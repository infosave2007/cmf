// NVG Ť on CUDA (`gpu_cuda/nvgt.rs`): the kernels of the layer-streamed
// distillation trainer beside the bake's (`bake.cu`, compiled into the same
// module: conv, GDN forward, causal softmax, RMSNorm). Activations are
// row-major `[tokens][width]` f32; the operands of the tensor-core GEMMs are
// staged in f16 (a column factor and a power-of-two scale folded in), as the
// wgpu trainer stages them. Column reductions run in two ordered passes, so
// a step is bit-reproducible. A kernel over a matrix takes its row from the
// grid's y and its column from x: no 64-bit division per element.

typedef unsigned int u32;
typedef unsigned short u16;

__device__ __forceinline__ u16 nt_f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float nt_h2f(u16 h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ float nt_sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ bool nt_fin(float v) { return (__float_as_uint(v) & 0x7f800000u) != 0x7f800000u; }

// Σ v over the block (blockDim a multiple of 32); every thread gets it.
__device__ __forceinline__ float nt_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
    const int nw = blockDim.x >> 5;
    for (int i = 0; i < nw; i++) s += red[i];
    return s;
}

// Four sums over the block at once (blockDim a multiple of 32, `red` ≥
// 4·blockDim/32 floats): one pair of barriers instead of four.
__device__ __forceinline__ void nt_block_sum4(float& a, float& b, float& c, float& d, float* red) {
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, o);
        b += __shfl_xor_sync(0xffffffffu, b, o);
        c += __shfl_xor_sync(0xffffffffu, c, o);
        d += __shfl_xor_sync(0xffffffffu, d, o);
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) {
        red[w] = a;
        red[nw + w] = b;
        red[2 * nw + w] = c;
        red[3 * nw + w] = d;
    }
    __syncthreads();
    a = b = c = d = 0.f;
    for (int i = 0; i < nw; i++) {
        a += red[i];
        b += red[nw + i];
        c += red[2 * nw + i];
        d += red[3 * nw + i];
    }
}

// ───────────────────────────── staging / planes ─────────────────────────────

// dst (f16 [m][k]) = src (f32 [m][k]) · col[k] (when given) · scale. Grid (k / 256, m).
extern "C" __global__ void nt_stage(const float* __restrict__ src, const float* __restrict__ col, u16* __restrict__ dst,
                                    u32 k, float scale, u32 has_col)
{
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    const size_t i = (size_t)blockIdx.y * k + c;
    float v = src[i] * scale;
    if (has_col) v *= col[c];
    dst[i] = nt_f2h(v);
}

// A sign bitmap [rows][wpr] (bit k of row i = rank k) as the f16 plane
// [rows][r] of ±sv[row]·mul: the row's scale (h of an output row, g of an
// input row) rides in the plane, so one staged activation serves every path
// and linear that reads it and a path's product lands in its sum directly.
// `mul` is a power of two that brings the largest scale under 1 (the GEMM's
// α takes it back). Grid (r / 256, rows).
extern "C" __global__ void nt_expand_s(const u32* __restrict__ bits, u16* __restrict__ pl, const float* __restrict__ sv,
                                       u32 r, u32 wpr, float mul)
{
    const u32 row = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= r) return;
    const u32 w = bits[(size_t)row * wpr + (k >> 5)];
    const float v = sv[row] * mul;
    pl[(size_t)row * r + k] = nt_f2h(((w >> (k & 31)) & 1u) ? v : -v);
}

// G ([rows][r]: a sign matrix's gradient before its row scale, G'[i][k] =
// ∂L/∂(s_i·S[i][k])): the scale's own gradient acc[row] += Σ_k ±G'[row][k]
// (the signs as they were in the forward), then G[row][k] = s[row]·G'[row][k],
// the gradient of the signs. A block a row, the sum in order.
extern "C" __global__ void nt_g_fold(float* __restrict__ G, const u32* __restrict__ bits, const float* __restrict__ sv,
                                     float* __restrict__ acc, u32 r, u32 wpr)
{
    __shared__ float red[32];
    const size_t row = blockIdx.x;
    const u32 t = threadIdx.x;
    const float s = sv[row];
    float a = 0.f;
    for (u32 k = t; k < r; k += blockDim.x) {
        const float g = G[row * r + k];
        const u32 w = bits[row * wpr + (k >> 5)];
        a += ((w >> (k & 31)) & 1u) ? g : -g;
        G[row * r + k] = g * s;
    }
    const float tot = nt_block_sum(a, red);
    if (t == 0) acc[row] += tot;
}

// out[i] = f32(src[i]) for n halves.
extern "C" __global__ void nt_h2f_n(const u16* __restrict__ src, float* __restrict__ out, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = nt_h2f(src[i]);
}

// out = a ⊙ col per row (in place when out = a): the DTG-MA gate on the MLP's
// activation and on its gradient. Grid (k / 256, rows).
extern "C" __global__ void nt_scale_cols(const float* a, const float* __restrict__ col, float* out, u32 k)
{
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    const size_t i = (size_t)blockIdx.y * k + c;
    out[i] = a[i] * col[c];
}

// Column reduction, first pass: part[blk·k + j] = Σ over the rows of block blk
// of a·b (mode 0), a·b·f[row] (mode 1) or a (mode 2). Grid (k / 256, blocks).
extern "C" __global__ void nt_colsum_part(const float* __restrict__ a, const float* __restrict__ b,
                                          const float* __restrict__ f, float* __restrict__ part, u32 m, u32 k, u32 rpb,
                                          u32 mode)
{
    const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    const u32 r0 = blockIdx.y * rpb, r1 = min(r0 + rpb, m);
    float acc = 0.f;
    if (mode == 0) {
        for (u32 r = r0; r < r1; r++) acc += a[(size_t)r * k + j] * b[(size_t)r * k + j];
    } else if (mode == 1) {
        for (u32 r = r0; r < r1; r++) acc += a[(size_t)r * k + j] * b[(size_t)r * k + j] * f[r];
    } else {
        for (u32 r = r0; r < r1; r++) acc += a[(size_t)r * k + j];
    }
    part[(size_t)blockIdx.y * k + j] = acc;
}

// Second pass: out[j] (+)= Σ_blk part[blk·k + j], the blocks in order.
extern "C" __global__ void nt_colsum_fin(const float* __restrict__ part, float* __restrict__ out, u32 k, u32 nblk, u32 acc)
{
    const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    float s = 0.f;
    for (u32 b = 0; b < nblk; b++) s += part[(size_t)b * k + j];
    out[j] = acc ? out[j] + s : s;
}

// ───────────────────────────── elementwise ─────────────────────────────

// h = silu(g)·u
extern "C" __global__ void nt_swiglu_fwd(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h,
                                         size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i];
    h[i] = a * nt_sigm(a) * u[i];
}

extern "C" __global__ void nt_swiglu_bwd(const float* __restrict__ g, const float* __restrict__ u,
                                         const float* __restrict__ dh, float* __restrict__ dg, float* __restrict__ du,
                                         size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i], s = nt_sigm(a);
    dg[i] = dh[i] * u[i] * s * (1.f + a * (1.f - s));
    du[i] = dh[i] * a * s;
}

// h = silu(g)·u staged in f16 for the linear that reads it (what `nt_stage`
// makes of `nt_swiglu_fwd`'s h, in one pass and without the f32 copy).
extern "C" __global__ void nt_swiglu_fwd16(const float* __restrict__ g, const float* __restrict__ u, u16* __restrict__ h,
                                           size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i];
    const float v = a * nt_sigm(a) * u[i];
    h[i] = nt_f2h(v);
}

// `nt_swiglu_bwd`'s two gradients staged in f16 times `scale` (the gradient
// scale) for their linears' backward — the values `nt_stage` makes of them.
extern "C" __global__ void nt_swiglu_bwd16(const float* __restrict__ g, const float* __restrict__ u,
                                           const float* __restrict__ dh, u16* __restrict__ dg, u16* __restrict__ du,
                                           size_t n, float scale)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i], s = nt_sigm(a);
    const float vg = dh[i] * u[i] * s * (1.f + a * (1.f - s));
    const float vu = dh[i] * a * s;
    dg[i] = nt_f2h(vg * scale);
    du[i] = nt_f2h(vu * scale);
}

// og = o·σ(gate)
extern "C" __global__ void nt_gate_fwd(const float* __restrict__ o, const float* __restrict__ gate, float* __restrict__ og,
                                       size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) og[i] = o[i] * nt_sigm(gate[i]);
}

extern "C" __global__ void nt_gate_bwd(const float* __restrict__ o, const float* __restrict__ gate,
                                       const float* __restrict__ dog, float* __restrict__ dout, float* __restrict__ dgate,
                                       size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = nt_sigm(gate[i]);
    dout[i] = dog[i] * s;
    dgate[i] = dog[i] * o[i] * s * (1.f - s);
}

// Per-head [q(hd); gate(hd)] rows of qg ([m][heads][2·hd]) → q, gate
// ([m][heads·hd] each); merge: the reverse.
extern "C" __global__ void nt_split_qg(float* qg, float* q, float* gate, u32 heads, u32 hd, u32 merge)
{
    const u32 kq = heads * hd, row = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kq) return;
    const u32 h = c / hd, d = c % hd;
    const size_t i = (size_t)row * kq + c;
    const size_t src = (size_t)row * 2 * kq + (size_t)h * 2 * hd + d;
    if (!merge) {
        q[i] = qg[src];
        gate[i] = qg[src + hd];
    } else {
        qg[src] = q[i];
        qg[src + hd] = gate[i];
    }
}

// y = a + b (bk_add twin kept here for a size_t count)
extern "C" __global__ void nt_add(const float* a, const float* b, float* y, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

// Depthwise causal conv + SiLU of the GDN's raw projection (`bk_conv_fwd`):
// pre = Σ_j raw[ti-(kk-1)+j]·tap[c][j], cq = silu(pre). Grid (cdim / 256, rows).
extern "C" __global__ void nt_conv_fwd(const float* __restrict__ raw, const float* __restrict__ taps,
                                       float* __restrict__ pre, float* __restrict__ cq, int t, int cdim, int kk)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= cdim) return;
    const int ti = r % t;
    float acc = 0.f;
    for (int j = 0; j < kk; j++) {
        const int p = ti - (kk - 1) + j;
        if (p >= 0) acc += raw[(size_t)(r - ti + p) * cdim + c] * taps[c * kk + j];
    }
    const size_t idx = (size_t)r * cdim + c;
    pre[idx] = acc;
    cq[idx] = bk_silu(acc);
}

// Its backward (`bk_conv_bwd`): draw[p][c] = Σ_j dcq[p+(kk-1)-j][c]·silu'(pre)·tap[c][j].
extern "C" __global__ void nt_conv_bwd(const float* __restrict__ pre, const float* __restrict__ dcq,
                                       const float* __restrict__ taps, float* __restrict__ draw, int t, int cdim, int kk)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= cdim) return;
    const int p = r % t;
    float acc = 0.f;
    for (int j = 0; j < kk; j++) {
        const int ti = p + (kk - 1) - j;
        if (ti < t) {
            const size_t q = (size_t)(r - p + ti) * cdim + c;
            acc += dcq[q] * bk_silu_bwd(pre[q]) * taps[c * kk + j];
        }
    }
    draw[(size_t)r * cdim + c] = acc;
}

// dst[(bi·nh + h)][p][c] = src[(bi·t + p)·(sh·hd) + (h / rep)·hd + c]
// (`bk_attn_perm`). Grid (hd / 256, t, nb·nh): x covers c, y = p, z = bi·nh + h.
extern "C" __global__ void nt_attn_perm(const float* __restrict__ src, float* __restrict__ dst, u32 t, u32 nh, u32 sh,
                                        u32 hd, u32 rep)
{
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hd) return;
    const u32 p = blockIdx.y, u = blockIdx.z, h = u % nh, bi = u / nh;
    dst[((size_t)u * t + p) * hd + c] = src[((size_t)bi * t + p) * sh * hd + (size_t)(h / rep) * hd + c];
}

// dst[(bi·t + p)·(dh·hd) + g·hd + c] = Σ over the rep q-heads h of group g of
// src[(bi·nh + h)][p][c] (`bk_attn_unperm`). Grid (dh·hd / 256, b·t).
extern "C" __global__ void nt_attn_unperm(const float* __restrict__ src, float* __restrict__ dst, u32 t, u32 nh, u32 dh,
                                          u32 hd, u32 rep)
{
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= dh * hd) return;
    const u32 row = blockIdx.y, p = row % t, bi = row / t, g = x / hd, c = x % hd;
    float acc = 0.f;
    for (u32 k = 0; k < rep; k++) acc += src[(((size_t)bi * nh + g * rep + k) * t + p) * hd + c];
    dst[(size_t)row * dh * hd + x] = acc;
}

// ───────────────────────────── q/k norm + RoPE ─────────────────────────────

// Per-head RMSNorm (over hd ≤ 256) then RoPE on the first 2·half dims (pairs
// (i, i + half)); INV per (row, head). A block of hd threads per (row, head).
extern "C" __global__ void nt_qkr_fwd(const float* __restrict__ X, const float* __restrict__ W, float* __restrict__ Y,
                                      float* __restrict__ INV, const float* __restrict__ ROPE, u32 heads, u32 hd, u32 half,
                                      u32 t, float eps, u32 gemma)
{
    __shared__ float red[32];
    __shared__ float buf[256];
    const size_t u = blockIdx.x;
    const u32 tid = threadIdx.x, pos = (u32)((u / heads) % t);
    const size_t base = u * hd;
    const float x = tid < hd ? X[base + tid] : 0.f;
    const float inv = 1.f / sqrtf(nt_block_sum(x * x, red) / (float)hd + eps);
    if (tid < hd) buf[tid] = x * inv * (gemma ? 1.f + W[tid] : W[tid]);
    __syncthreads();
    if (tid < hd) {
        float y = buf[tid];
        if (tid < half) {
            const float c = ROPE[((size_t)pos * half + tid) * 2], s = ROPE[((size_t)pos * half + tid) * 2 + 1];
            y = buf[tid] * c - buf[tid + half] * s;
        } else if (tid < 2 * half) {
            const u32 i = tid - half;
            const float c = ROPE[((size_t)pos * half + i) * 2], s = ROPE[((size_t)pos * half + i) * 2 + 1];
            y = buf[i] * s + buf[tid] * c;
        }
        Y[base + tid] = y;
    }
    if (tid == 0) INV[u] = inv;
}

// Inverse rotation of dY, then the RMSNorm backward; the gain gradient of
// each (row, head) goes to DW[u·hd + j] (summed by a column reduction).
extern "C" __global__ void nt_qkr_bwd(const float* __restrict__ X, const float* __restrict__ W,
                                      const float* __restrict__ INV, const float* __restrict__ ROPE,
                                      const float* __restrict__ DY, float* __restrict__ DX, float* __restrict__ DW,
                                      u32 heads, u32 hd, u32 half, u32 t, u32 gemma)
{
    __shared__ float red[32];
    __shared__ float buf[256];
    const size_t u = blockIdx.x;
    const u32 tid = threadIdx.x, pos = (u32)((u / heads) % t);
    const size_t base = u * hd;
    if (tid < hd) buf[tid] = DY[base + tid];
    __syncthreads();
    float g = 0.f;
    if (tid < hd) {
        g = buf[tid];
        if (tid < half) {
            const float c = ROPE[((size_t)pos * half + tid) * 2], s = ROPE[((size_t)pos * half + tid) * 2 + 1];
            g = buf[tid] * c + buf[tid + half] * s;
        } else if (tid < 2 * half) {
            const u32 i = tid - half;
            const float c = ROPE[((size_t)pos * half + i) * 2], s = ROPE[((size_t)pos * half + i) * 2 + 1];
            g = -buf[i] * s + buf[tid] * c;
        }
    }
    const float iv = INV[u];
    const float x = tid < hd ? X[base + tid] : 0.f;
    const float w = tid < hd ? (gemma ? 1.f + W[tid] : W[tid]) : 0.f;
    const float coef = nt_block_sum(g * w * x, red) / (float)hd * iv * iv * iv;
    if (tid < hd) {
        DX[base + tid] = iv * w * g - x * coef;
        DW[base + tid] = g * x * iv;
    }
}

// ───────────────────────────── GDN recurrence ─────────────────────────────
//
// The gated delta rule of `bk_gdn_fwd` / `bk_gdn_bwd` (bake.cu) in blocks of
// DV·GC threads (GC from the build, `GDN_GC` of nvgt.rs: a block's threads
// share an SM's 64k registers, so 1024 threads leave 64 each — too few for
// the reverse sweep's two state columns): a thread holds GR = DK / GC rows of
// one column of the state in registers, so a position is two loops of GR and
// two barriers forward, three backward (a position's q, k and scalar sums are
// published a position early; the forward's output norm rides the next
// position's barrier). (In the
// bake's block of DV threads a thread walks all DK rows of its column — the
// state spills to local memory — and an SM's schedulers wait on four warps'
// dependency chains: 3.4 µs a position forward and 17 µs backward on an RTX
// PRO 6000.) Column sums (Sᵀk̂, Sᵀq̂, dSᵀk̂) add the GC chunks' partials in
// order through shared memory; row sums (S·do, dS·u, S·dkv) reduce a warp's
// 32 columns by shuffles and add the DV / 32 warps' partials in order: a
// step stays bit-reproducible. The BPTT's state history is f16 and holds
// every second state ([unit][⌈t/2⌉][DK][DV], unit = local sequence · nv + h;
// the states after positions t-1, t-3, …): the reverse sweep rebuilds the
// state between two stored ones by one rank-one step from the forward's kv
// (no sums). A quarter of the traffic of an f32 history of every state,
// written once by the forward that the reverse sweep's layer recompute runs
// anyway and read once.

#ifndef GC
#define GC 4
#endif
#define GR (DK / GC)
#define GW (DV / 32)
// A row sum over a warp's 32 columns: GSH shuffle steps leave the sums of
// 32 >> GSH lanes each with their first lanes, which store them; the row's
// owner adds the GW·GP partials in order. (All five steps in the warp are
// 32 additions of which one is kept, twice a state row a position: two
// thirds of the reverse sweep's instructions.)
#define GSH 2
// The largest chunk of the chunked reverse sweep (`nt_gdc_*`).
#define GDC_MAX 256
#define GP (32 >> GSH)

// The GW warps' partials in order.
__device__ __forceinline__ float nt_gsum(const float* r) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < GW; i++) s += r[i];
    return s;
}

__device__ __forceinline__ u16 nt_f2h_sat(float f) { return nt_f2h(fminf(fmaxf(f, -65504.f), 65504.f)); }

// A position's q, k and the sums of their squares go to shared memory a
// position early (slot = the position's parity): the block needs no barrier
// of its own for them.
#define GDN_PUB_QK(SL, QV, KV_)                                    \
    if (tid < DK) {                                                \
        qs[SL][tid] = QV;                                          \
        ks[SL][tid] = KV_;                                         \
        float pq = QV * QV, pk = KV_ * KV_;                        \
        for (int o = 16; o > 0; o >>= 1) {                         \
            pq += __shfl_xor_sync(0xffffffffu, pq, o);             \
            pk += __shfl_xor_sync(0xffffffffu, pk, o);             \
        }                                                          \
        if ((tid & 31) == 0) {                                     \
            rq[SL][tid >> 5] = pq;                                 \
            rk[SL][tid >> 5] = pk;                                 \
        }                                                          \
    }

// Forward: grid (nv, sequences of this launch), block DV·GC.
// out[row][h·DV + j] = rms(o)·w·silu(z) when `out` is given; with `hist` the
// state after every second position (the last one among them), and kv and o
// of every position, are kept for `nt_gdn_rev`. Two barriers a position: q
// and k are published a position early, the output's norm is taken under the
// next position's first barrier.
extern "C" __global__ void nt_gdn_fwd(const float* cq, const float* z, const float* a, const float* bb,
                                      const float* a_log, const float* dt_bias, const float* normw, float* out,
                                      u16* hist, float* kvh, float* oh, int t, int nv, int nk, float eps, int seq0,
                                      int stride)
{
    const int h = blockIdx.x, lb = blockIdx.y, bi = seq0 + lb;
    const int tid = threadIdx.x, j = tid % DV, c = tid / DV, i0 = c * GR;
    const int rep = nv / nk, ko = h / rep, kd = nk * DK, vd = nv * DV, cdim = 2 * kd + vd;
    __shared__ float qs[2][DK], ks[2][DK], rq[2][GW], rk[2][GW], ro[GW], cpa[GC][DV], cpb[GC][DV];
    const long unit = (long)lb * nv + h;
    // the states kept: after the positions t-1, t-1-stride, … (slot k from the end)
    u16* H = hist ? hist + unit * (long)((t + stride - 1) / stride) * DK * DV : 0;
    float* KV = hist ? kvh + unit * (long)t * DV : 0;
    float* O = hist ? oh + unit * (long)t * DV : 0;
    const float ea = __expf(a_log[h]), dtb = dt_bias[h], wn = normw[j];
    const long row0 = (long)bi * t;
    float S[GR];
#pragma unroll
    for (int i = 0; i < GR; i++) S[i] = 0.f;
    // inputs are read ahead — q and k two positions (they are published one
    // position early), the rest one: the latency of their loads (rows 6 KB
    // apart, nothing prefetches them) passes under the arithmetic before
    float qa = 0.f, ka = 0.f, vn, an, bn, zn = 0.f;
    {
        const float* cr = cq + row0 * cdim;
        if (tid < DK) {
            qa = cr[ko * DK + tid];
            ka = cr[kd + ko * DK + tid];
        }
        GDN_PUB_QK(0, qa, ka)
        vn = cr[2 * kd + h * DV + j];
        an = a[row0 * nv + h];
        bn = bb[row0 * nv + h];
        if (c == 0 && out) zn = z[row0 * vd + h * DV + j];
        if (t > 1 && tid < DK) {
            qa = cr[cdim + ko * DK + tid];
            ka = cr[cdim + kd + ko * DK + tid];
        }
    }
    __syncthreads();
    float o_prev = 0.f, z_prev = 0.f;
    for (int ti = 0; ti < t; ti++) {
        const int p = ti & 1;
        const long row = row0 + ti;
        const float vj = vn, av = an, bv = bn, zv = zn;
        if (ti + 1 < t) {
            GDN_PUB_QK(1 - p, qa, ka)
            const float* cr = cq + (row + 1) * cdim;
            vn = cr[2 * kd + h * DV + j];
            an = a[(row + 1) * nv + h];
            bn = bb[(row + 1) * nv + h];
            if (c == 0 && out) zn = z[(row + 1) * vd + h * DV + j];
            if (ti + 2 < t && tid < DK) {
                qa = cr[cdim + ko * DK + tid];
                ka = cr[cdim + kd + ko * DK + tid];
            }
        }
        const float nq2 = nt_gsum(rq[p]) + 1e-6f, nk2 = nt_gsum(rk[p]) + 1e-6f;
        const float invq = 1.f / (sqrtf(nq2) * sqrtf((float)DK)), invk = 1.f / sqrtf(nk2);
        const float g = __expf(-ea * bk_softplus(av + dtb)), beta = bk_sigmoid(bv);
        float kvp = 0.f;
#pragma unroll
        for (int i = 0; i < GR; i++) {
            S[i] *= g;
            kvp += S[i] * ks[p][i0 + i] * invk;
        }
        cpa[c][j] = kvp;
        __syncthreads();
        // the position before: its output under its norm (the sums landed before this barrier)
        if (c == 0 && out && ti > 0) {
            const float inv = rsqrtf(nt_gsum(ro) / DV + eps);
            out[(row - 1) * vd + h * DV + j] = o_prev * inv * wn * bk_silu(z_prev);
        }
        float kv = 0.f;
#pragma unroll
        for (int q = 0; q < GC; q++) kv += cpa[q][j];
        const float u = beta * (vj - kv);
        float op = 0.f;
#pragma unroll
        for (int i = 0; i < GR; i++) {
            S[i] += ks[p][i0 + i] * invk * u;
            op += qs[p][i0 + i] * invq * S[i];
        }
        if (H && (t - 1 - ti) % stride == 0) {
            u16* Hn = H + (long)((t - 1 - ti) / stride) * DK * DV;
#pragma unroll
            for (int i = 0; i < GR; i++) Hn[(long)(i0 + i) * DV + j] = nt_f2h_sat(S[i]);
        }
        cpb[c][j] = op;
        __syncthreads();
        if (c == 0) {
            float o = 0.f;
#pragma unroll
            for (int q = 0; q < GC; q++) o += cpb[q][j];
            float so = o * o;
            for (int x = 16; x > 0; x >>= 1) so += __shfl_xor_sync(0xffffffffu, so, x);
            if ((tid & 31) == 0) ro[tid >> 5] = so;
            if (H) {
                KV[(long)ti * DV + j] = kv;
                O[(long)ti * DV + j] = o;
            }
            o_prev = o;
            z_prev = zv;
        }
    }
    __syncthreads();
    if (c == 0 && out) {
        const float inv = rsqrtf(nt_gsum(ro) / DV + eps);
        out[(row0 + t - 1) * vd + h * DV + j] = o_prev * inv * wn * bk_silu(z_prev);
    }
}

// The reverse sweep's sums of a position — q, k, their squares, and the gated
// RMSNorm's two sums — published a position early like the forward's.
#define GDN_PUB_REV(SL, QV, KV_, OV, ZV, DOV)                      \
    if (tid < DK) {                                                \
        qs[SL][tid] = QV;                                          \
        ks[SL][tid] = KV_;                                         \
        float pq = QV * QV, pk = KV_ * KV_, so = OV * OV, sd = DOV * (wn * bk_silu(ZV)) * OV; \
        for (int o = 16; o > 0; o >>= 1) {                         \
            pq += __shfl_xor_sync(0xffffffffu, pq, o);             \
            pk += __shfl_xor_sync(0xffffffffu, pk, o);             \
            so += __shfl_xor_sync(0xffffffffu, so, o);             \
            sd += __shfl_xor_sync(0xffffffffu, sd, o);             \
        }                                                          \
        if (lane == 0) {                                           \
            rq[SL][w] = pq;                                        \
            rk[SL][w] = pk;                                        \
            ro[SL][w] = so;                                        \
            rs[SL][w] = sd;                                        \
        }                                                          \
    }

// Reverse sweep (BPTT) over the history `nt_gdn_fwd` left: grid (nv,
// sequences of this launch), block DV·GC. Each v-head writes its q̂/k̂
// through-gradients into DQK ([row][nv][2·DK]); `nt_gdn_qk_sum` adds the heads
// of a k-group in order. dz, da, db and dcq's v channels are stored.
//
// The history holds the state after every second position (the last one
// among them). At a stored position the state before it is the stored state
// two back advanced by the position between (g·S + k̂ ⊗ u, u = β·(v − kv) from
// the forward's kv: no sums); at the position between, the state after it is
// the stored state before it advanced by the position itself. One state
// array in registers either way, half the history's traffic. Three barriers
// a position: the sums of a position are published a position early.
extern "C" __global__ void nt_gdn_rev(const float* cq, const float* z, const float* a, const float* bb,
                                      const float* a_log, const float* dt_bias, const float* normw, const float* dout,
                                      float* dcq, float* dz, float* da, float* db, float* dqk, const u16* hist,
                                      const float* kvh, const float* oh, int t, int nv, int nk, float eps, int seq0)
{
    const int h = blockIdx.x, lb = blockIdx.y, bi = seq0 + lb;
    const int tid = threadIdx.x, j = tid % DV, c = tid / DV, i0 = c * GR;
    const int lane = tid & 31, w = tid >> 5, wc = w % GW;
    const int rep = nv / nk, ko = h / rep, kd = nk * DK, vd = nv * DV, cdim = 2 * kd + vd;
    __shared__ float qs[2][DK], ks[2][DK], rq[2][GW], rk[2][GW], ro[2][GW], rs[2][GW], rb[GW], rqd[GW], rkd[GW],
        rg[GC * GW], cp[GC][DV], pq_[GW * GP][DK], pk_[GW * GP][DK];
    const int lead = (lane & ((1 << GSH) - 1)) == 0, slot = wc * GP + (lane >> GSH);
    const long unit = (long)lb * nv + h;
    const int par = (t - 1) & 1;
    const u16* H = hist + unit * (long)((t + 1) / 2) * DK * DV;
    const float* KV = kvh + unit * (long)t * DV;
    const float* O = oh + unit * (long)t * DV;
    const float ea = __expf(a_log[h]), dtb = dt_bias[h], wn = normw[j];
    const long row0 = (long)bi * t;
    float dS[GR], St[GR];
    {
        // slot 0: the state after the last position
        const u16* Hl = H;
#pragma unroll
        for (int i = 0; i < GR; i++) {
            dS[i] = 0.f;
            St[i] = nt_h2f(Hl[(long)(i0 + i) * DV + j]);
        }
    }
    // the current position's inputs, and the next one's (read two positions
    // ahead: its sums are published a position early; see `nt_gdn_fwd`)
    float oc, zc, dc, vc, ac, bc;
    float qn = 0.f, kn = 0.f, on = 0.f, zn = 0.f, dn = 0.f, vn = 0.f, an = 0.f, bn = 0.f;
    {
        const long row = row0 + (t - 1);
        const float* cr = cq + row * cdim;
        float q0 = 0.f, k0 = 0.f;
        if (tid < DK) {
            q0 = cr[ko * DK + tid];
            k0 = cr[kd + ko * DK + tid];
        }
        oc = O[(long)(t - 1) * DV + j];
        zc = z[row * vd + h * DV + j];
        dc = dout[row * vd + h * DV + j];
        vc = cr[2 * kd + h * DV + j] - KV[(long)(t - 1) * DV + j];
        ac = a[row * nv + h];
        bc = bb[row * nv + h];
        GDN_PUB_REV((t - 1) & 1, q0, k0, oc, zc, dc)
        // the other slot's keys are read (times zero) before a position fills them
        if (tid < DK) ks[1 - ((t - 1) & 1)][tid] = 0.f;
        if (t > 1) {
            const float* c1 = cr - cdim;
            if (tid < DK) {
                qn = c1[ko * DK + tid];
                kn = c1[kd + ko * DK + tid];
            }
            on = O[(long)(t - 2) * DV + j];
            zn = z[(row - 1) * vd + h * DV + j];
            dn = dout[(row - 1) * vd + h * DV + j];
            vn = c1[2 * kd + h * DV + j] - KV[(long)(t - 2) * DV + j];
            an = a[(row - 1) * nv + h];
            bn = bb[(row - 1) * nv + h];
        }
    }
    __syncthreads();
    for (int ti = t - 1; ti >= 0; ti--) {
        const int p = ti & 1;
        const bool stored = p == par;
        const long row = row0 + ti;
        const float oj = oc, zv = zc, dof = dc, vmk = vc, av = ac, bv = bc;
        // the position before, for the state between two stored ones
        const float vmkn = vn, avn = an, bvn = bn;
        if (ti > 0) {
            GDN_PUB_REV(1 - p, qn, kn, on, zn, dn)
            oc = on;
            zc = zn;
            dc = dn;
            vc = vn;
            ac = an;
            bc = bn;
            if (ti > 1) {
                const float* c2 = cq + (row - 2) * cdim;
                if (tid < DK) {
                    qn = c2[ko * DK + tid];
                    kn = c2[kd + ko * DK + tid];
                }
                on = O[(long)(ti - 2) * DV + j];
                zn = z[(row - 2) * vd + h * DV + j];
                dn = dout[(row - 2) * vd + h * DV + j];
                vn = c2[2 * kd + h * DV + j] - KV[(long)(ti - 2) * DV + j];
                an = a[(row - 2) * nv + h];
                bn = bb[(row - 2) * nv + h];
            }
        }
        const float weff = wn * bk_silu(zv);
        const float nq2 = nt_gsum(rq[p]) + 1e-6f, nk2 = nt_gsum(rk[p]) + 1e-6f;
        const float invq = 1.f / (sqrtf(nq2) * sqrtf((float)DK)), invk = 1.f / sqrtf(nk2);
        const float arg = av + dtb;
        const float g = __expf(-ea * bk_softplus(arg)), beta = bk_sigmoid(bv);
        const float inv = rsqrtf(nt_gsum(ro[p]) / DV + eps), sdot = nt_gsum(rs[p]);
        const float doj = inv * weff * dof - oj * (sdot / DV) * inv * inv * inv;
        if (c == 0) dz[row * vd + h * DV + j] = dof * oj * inv * wn * bk_silu_bwd(zv);
        // the state after this position: St itself at a stored position, St
        // (the stored state before) advanced by this position otherwise
        const float cs = stored ? 1.f : g, ck = stored ? 0.f : invk * (beta * vmk);
        // dS += q̂ ⊗ do; dq̂ = S_t·do (across columns); du = dSᵀk̂ (across chunks)
        float dup = 0.f;
#pragma unroll
        for (int i = 0; i < GR; i++) {
            const float kx = ks[p][i0 + i];
            dS[i] += qs[p][i0 + i] * invq * doj;
            dup += dS[i] * kx * invk;
            float v = (St[i] * cs + kx * ck) * doj;
            for (int o = 1; o < (1 << GSH); o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lead) pq_[slot][i0 + i] = v;
        }
        cp[c][j] = dup;
        // at a stored position: the stored state two back, read here — its
        // loads pass under the barrier and the column sum; it serves this
        // position (advanced) and the next two
        if (stored) {
            if (ti > 1) {
                const u16* Hp = H + (long)((t + 1 - ti) >> 1) * DK * DV;
#pragma unroll
                for (int i = 0; i < GR; i++) St[i] = nt_h2f(Hp[(long)(i0 + i) * DV + j]);
            } else {
#pragma unroll
                for (int i = 0; i < GR; i++) St[i] = 0.f;
            }
        }
        __syncthreads();
        float du = 0.f;
#pragma unroll
        for (int q = 0; q < GC; q++) du += cp[q][j];
        const float dkv = -(beta * du);
        if (c == 0) {
            dcq[row * cdim + (long)(2 * kd + h * DV + j)] = beta * du;
            float pb = du * vmk;
            for (int o = 16; o > 0; o >>= 1) pb += __shfl_xor_sync(0xffffffffu, pb, o);
            if (lane == 0) rb[w] = pb;
        }
        // the state before this position: St itself between stored positions,
        // St advanced by the position before at a stored one (nothing before
        // position 0)
        float cs2 = 1.f, ck2 = 0.f;
        if (stored) {
            if (ti > 0) {
                const float invkn = 1.f / sqrtf(nt_gsum(rk[1 - p]) + 1e-6f);
                cs2 = __expf(-ea * bk_softplus(avn + dtb));
                ck2 = invkn * (bk_sigmoid(bvn) * vmkn);
            } else {
                cs2 = 0.f;
            }
        }
        // dk̂ = dS·u + S_pre·dkv (across columns); dg = ⟨dS_pre, S_{t-1}⟩; dS ← g·dS_pre
        float dgp = 0.f;
#pragma unroll
        for (int i = 0; i < GR; i++) {
            const float sp = St[i] * cs2 + ks[1 - p][i0 + i] * ck2;
            float v = dS[i] * vmk * beta + sp * g * dkv;
            for (int o = 1; o < (1 << GSH); o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lead) pk_[slot][i0 + i] = v;
            const float dspre = dS[i] + ks[p][i0 + i] * invk * dkv;
            dgp += dspre * sp;
            dS[i] = g * dspre;
        }
        for (int o = 16; o > 0; o >>= 1) dgp += __shfl_xor_sync(0xffffffffu, dgp, o);
        if (lane == 0) rg[w] = dgp;
        __syncthreads();
        // the warps' partials in order, then the l2-norm through-grads of this head
        float aq = 0.f, ak = 0.f;
        if (tid < DK) {
#pragma unroll
            for (int x = 0; x < GW * GP; x++) {
                aq += pq_[x][tid];
                ak += pk_[x][tid];
            }
            float pq = aq * qs[p][tid], pk = ak * ks[p][tid];
            for (int o = 16; o > 0; o >>= 1) {
                pq += __shfl_xor_sync(0xffffffffu, pq, o);
                pk += __shfl_xor_sync(0xffffffffu, pk, o);
            }
            if (lane == 0) {
                rqd[w] = pq;
                rkd[w] = pk;
            }
        }
        if (tid == 0) {
            float dg = 0.f;
            for (int x = 0; x < GC * GW; x++) dg += rg[x];
            da[row * nv + h] = dg * g * (-ea) * bk_sigmoid(arg);
            db[row * nv + h] = nt_gsum(rb) * beta * (1.f - beta);
        }
        __syncthreads();
        if (tid < DK) {
            const float qdot = nt_gsum(rqd), kdot = nt_gsum(rkd);
            float* dst = dqk + ((size_t)row * nv + h) * 2 * DK;
            dst[tid] = invq * aq - qs[p][tid] * qdot * invq / nq2;
            dst[DK + tid] = invk * ak - ks[p][tid] * kdot * invk / nk2;
        }
    }
}

// dcq[row][ko·DK + i] = Σ over the v-heads h of k-group ko (in order) of
// DQK[row][h][i]; the k̂ channels likewise at kd + ko·DK + i.
extern "C" __global__ void nt_gdn_qk_sum(const float* __restrict__ dqk, float* __restrict__ dcq, int nv, int nk)
{
    const int kd = nk * DK, rep = nv / nk, cdim = 2 * kd + nv * DV;
    const size_t row = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 2 * kd) return;
    const int part = c / kd, ko = (c % kd) / DK, i = c % DK;
    float acc = 0.f;
    for (int r = 0; r < rep; r++) acc += dqk[(row * nv + ko * rep + r) * 2 * DK + part * DK + i];
    dcq[row * cdim + c] = acc;
}

// ───────────────────────────── GDN reverse sweep in chunks ─────────────────────────────
//
// The reverse sweep as batched matrix products over chunks of C positions (a
// unit = one sequence's v-head; D = DK = DV). With the chunk's rows K̂, Q̂, U
// ([C][D]; u_t = β_t·(v_t − kv_t) from the forward's kv), its start state S0
// ([D][D], rows the key index), the decays c_τ = Π_{σ≤τ} g_σ inside the
// chunk and r(τ,σ) = c_τ/c_σ:
//   kv_τ  = c_τ·(K̂ S0)_τ + Σ_{σ<τ} r(τ,σ)(k̂_τ·k̂_σ)·u_σ
//   o_τ   = c_τ·(Q̂ S0)_τ + Σ_{σ≤τ} r(τ,σ)(q̂_τ·k̂_σ)·u_σ
//   S_end = c_end·S0 + K̂ᵀ·diag(c_end/c_σ)·U
// so the adjoints are products of [C][D], [C][C] and [D][D] matrices and one
// back substitution (`nt_gdc_solve`): X_τ = Ū_τ − Σ_{τ'>τ} β_τ'·P[τ'][τ]·X_τ'
// with P = r ⊙ K̂K̂ᵀ below the diagonal. X is what the sequential sweep calls
// du. The kernels here are the elementwise and per-row parts; the products
// are cuBLAS batched SGEMMs issued by `gdn_core_bwd_chunked`.
//
// Per (unit, τ) scalars SC[8]: β, log g, 1/(|q|·√D), 1/|k|, |q|², |k|²,
// d(log g)/da, Σ_{σ≤τ} log g.

// One position of a chunk: the normalized keys and queries, u, the gradient
// at o through the gated RMSNorm (dz stored), and the scalars. Grid (L,
// units), block DV.
extern "C" __global__ void nt_gdc_prep(const float* cq, const float* z, const float* a, const float* bb,
                                       const float* a_log, const float* dt_bias, const float* normw, const float* dout,
                                       const float* kvh, const float* oh, float* dz, float* KH, float* QH, float* UU,
                                       float* OB, float* SC, int t, int nv, int nk, float eps, int seq0, int p0, int C)
{
    __shared__ float red[4 * (DV / 32)];
    const int tau = blockIdx.x, unit = blockIdx.y, h = unit % nv, lb = unit / nv, bi = seq0 + lb, ti = p0 + tau;
    const int j = threadIdx.x;
    const int rep = nv / nk, ko = h / rep, kd = nk * DK, vd = nv * DV, cdim = 2 * kd + vd;
    const long row = (long)bi * t + ti;
    const float* cr = cq + row * cdim;
    const float q = cr[ko * DK + j], k = cr[kd + ko * DK + j], v = cr[2 * kd + h * DV + j];
    const long hp = ((long)unit * t + ti) * DV + j;
    const float kv = kvh[hp], o = oh[hp];
    const float zv = z[row * vd + h * DV + j], dof = dout[row * vd + h * DV + j];
    const float wn = normw[j], weff = wn * bk_silu(zv);
    float nq2 = q * q, nk2 = k * k, so = o * o, sd = dof * weff * o;
    nt_block_sum4(nq2, nk2, so, sd, red);
    nq2 += 1e-6f;
    nk2 += 1e-6f;
    const float invq = 1.f / (sqrtf(nq2) * sqrtf((float)DK)), invk = 1.f / sqrtf(nk2);
    const float arg = a[row * nv + h] + dt_bias[h], ea = __expf(a_log[h]);
    const float beta = bk_sigmoid(bb[row * nv + h]);
    const float inv = rsqrtf(so / DV + eps);
    dz[row * vd + h * DV + j] = dof * o * inv * wn * bk_silu_bwd(zv);
    const size_t m = ((size_t)unit * C + tau) * DV + j;
    KH[m] = k * invk;
    QH[m] = q * invq;
    UU[m] = beta * (v - kv);
    OB[m] = inv * weff * dof - o * (sd / DV) * inv * inv * inv;
    if (j == 0) {
        float* sc = SC + ((size_t)unit * C + tau) * 8;
        sc[0] = beta;
        sc[1] = -ea * bk_softplus(arg);
        sc[2] = invq;
        sc[3] = invk;
        sc[4] = nq2;
        sc[5] = nk2;
        sc[6] = -ea * bk_sigmoid(arg);
    }
}

// Σ_{σ≤τ} log g of a unit's chunk. Grid (units, 1), block 32 (one thread works).
extern "C" __global__ void nt_gdc_cum(float* SC, int C, int L)
{
    if (threadIdx.x != 0) return;
    float* sc = SC + (size_t)blockIdx.x * C * 8;
    float acc = 0.f;
    for (int tau = 0; tau < L; tau++) {
        acc += sc[tau * 8 + 1];
        sc[tau * 8 + 7] = acc;
    }
}

// M[τ][σ] = r(τ,σ)·QK[τ][σ] for σ ≤ τ (0 above), and the back substitution's
// rows LT[σ][τ] = β_τ·r(τ,σ)·KK[τ][σ] for σ < τ. Grid (C, units), block ≥ C.
extern "C" __global__ void nt_gdc_mask(const float* KK, const float* QK, const float* SC, float* M, float* LT, int C,
                                       int L)
{
    const int tau = blockIdx.x, unit = blockIdx.y, sg = threadIdx.x;
    if (sg >= C) return;
    const size_t ub = (size_t)unit * C;
    const size_t e = (ub + tau) * C + sg;
    if (tau >= L || sg > tau) {
        M[e] = 0.f;
        return;
    }
    const float* sc = SC + ub * 8;
    const float r = __expf(sc[tau * 8 + 7] - sc[sg * 8 + 7]);
    M[e] = r * QK[e];
    if (sg < tau) LT[(ub + sg) * C + tau] = sc[tau * 8] * r * KK[e];
}

// X = (I + diag(β)P)⁻ᵀ·(T1 + diag(w)·T2), a column of the chunk a thread:
// back substitution from the chunk's last position. Stores X, Ā = −β·c·X,
// the v-channel gradient β·X, and the two scaled copies the products after
// it read: CO = diag(c)·Ō, WU = diag(w)·U (w_τ = c_end/c_τ). The kernel runs
// at the speed of its nine reads and writes a position (the substitution's
// own loop in shared memory or four positions a load changed nothing).
// Grid (units, 1), block DV.
extern "C" __global__ void nt_gdc_solve(const float* T1, const float* T2, const float* LT, const float* SC,
                                        const float* OB, const float* UU, float* X, float* AB, float* CO, float* WU,
                                        float* dcq, int t, int nv, int nk, int seq0, int p0, int C, int L)
{
    const int unit = blockIdx.x, h = unit % nv, lb = unit / nv, bi = seq0 + lb, j = threadIdx.x;
    const int kd = nk * DK, vd = nv * DV, cdim = 2 * kd + vd;
    const size_t ub = (size_t)unit * C;
    const float* sc = SC + ub * 8;
    const float lend = sc[(L - 1) * 8 + 7];
    float x[GDC_MAX];
    for (int tau = L - 1; tau >= 0; tau--) {
        const size_t m = (ub + tau) * DV + j;
        const float lg = sc[tau * 8 + 7], beta = sc[tau * 8], c = __expf(lg), w = __expf(lend - lg);
        float acc = T1[m] + w * T2[m];
        const float* lt = LT + (ub + tau) * C;
        for (int tp = tau + 1; tp < L; tp++) acc -= lt[tp] * x[tp];
        x[tau] = acc;
        X[m] = acc;
        AB[m] = -beta * c * acc;
        CO[m] = c * OB[m];
        WU[m] = w * UU[m];
        dcq[((long)bi * t + p0 + tau) * cdim + (long)(2 * kd + h * DV + j)] = beta * acc;
    }
}

// The adjoints of the two Gram matrices and the decay's pairwise part:
// W1 = Z + Zᵀ with Z[τ][σ] = −β_τ·r(τ,σ)·XU[τ][σ] (σ < τ); Y[τ][σ] =
// r(τ,σ)·OU[τ][σ] (σ ≤ τ); E1[τ][σ] = Z·KK + Y·QK for σ < τ (it enters
// d log c with + at τ and − at σ). Grid (C, units), block ≥ C.
extern "C" __global__ void nt_gdc_zy(const float* XU, const float* OU, const float* KK, const float* QK,
                                     const float* SC, float* W1, float* Y, float* E1, int C, int L)
{
    const int tau = blockIdx.x, unit = blockIdx.y, sg = threadIdx.x;
    if (sg >= C) return;
    const size_t ub = (size_t)unit * C;
    const size_t e = (ub + tau) * C + sg;
    if (tau >= L || sg >= L) {
        W1[e] = 0.f;
        Y[e] = 0.f;
        E1[e] = 0.f;
        return;
    }
    const float* sc = SC + ub * 8;
    const float lt = sc[tau * 8 + 7], ls = sc[sg * 8 + 7];
    if (sg < tau) {
        const float r = __expf(lt - ls);
        const float zv = -sc[tau * 8] * r * XU[e], yv = r * OU[e];
        W1[e] = zv;
        Y[e] = yv;
        E1[e] = zv * KK[e] + yv * QK[e];
    } else if (sg == tau) {
        W1[e] = 0.f;
        Y[e] = OU[e];
        E1[e] = 0.f;
    } else {
        // above the diagonal: Zᵀ
        W1[e] = -sc[sg * 8] * __expf(ls - lt) * XU[(ub + sg) * C + tau];
        Y[e] = 0.f;
        E1[e] = 0.f;
    }
}

// Per position: db, and the decay's row terms R1 = ⟨Ā,A⟩ + ⟨CO,B⟩ − ⟨T2,WU⟩,
// R2 = ⟨T2,WU⟩. With A = K̂·S0 and B = Q̂·S0: ⟨Ā_τ,A_τ⟩ = ⟨k̂_τ,(Ā·S0ᵀ)_τ⟩ and
// ⟨CO_τ,B_τ⟩ = ⟨q̂_τ,(CO·S0ᵀ)_τ⟩ — the two products the adjoints of K̂ and Q̂
// start from (AS, CS here), so A and B are never formed. Grid (L, units),
// block DV.
extern "C" __global__ void nt_gdc_rows(const float* X, const float* UU, const float* KH, const float* AS,
                                       const float* QH, const float* CS, const float* T2, const float* WU,
                                       const float* SC, float* R, float* db, int t, int nv, int seq0, int p0, int C)
{
    __shared__ float red[4 * (DV / 32)];
    const int tau = blockIdx.x, unit = blockIdx.y, h = unit % nv, lb = unit / nv, bi = seq0 + lb, j = threadIdx.x;
    const size_t m = ((size_t)unit * C + tau) * DV + j;
    float xu = X[m] * UU[m], r2 = T2[m] * WU[m], r1 = KH[m] * AS[m] + QH[m] * CS[m], unused = 0.f;
    nt_block_sum4(xu, r2, r1, unused, red);
    if (j == 0) {
        const float beta = SC[((size_t)unit * C + tau) * 8];
        db[((long)bi * t + p0 + tau) * nv + h] = xu * (1.f - beta);
        float* r = R + ((size_t)unit * C + tau) * 2;
        r[0] = r1 - r2;
        r[1] = r2;
    }
}

// A chunk's start state from the f16 history (slot `slot` of every unit; a
// negative slot: the zero state before the window) and ⟨S̄_end, S0⟩ of the
// unit, in one pass. Grid (units, 1), block 256.
extern "C" __global__ void nt_gdc_s0(const u16* hist, const float* SE, float* S0, float* SD, int slots, int slot)
{
    __shared__ float red[8];
    const size_t unit = blockIdx.x, ub = unit * DK * DV;
    const u16* hs = hist + (unit * slots + (slot < 0 ? 0 : slot)) * (size_t)(DK * DV);
    float acc = 0.f;
    for (int e = threadIdx.x; e < DK * DV; e += blockDim.x) {
        const float v = slot < 0 ? 0.f : nt_h2f(hs[e]);
        S0[ub + e] = v;
        acc += SE[ub + e] * v;
    }
    const float s = nt_block_sum(acc, red);
    if (threadIdx.x == 0) SD[unit] = s;
}

// da of a chunk: d log c_τ = Σ_σ E1[τ][σ] − Σ_τ' E1[τ'][τ] + R1_τ (+ at the
// last position: Σ R2 + c_end·⟨S̄_end, S0⟩), summed from the chunk's end
// (log c_τ holds every log g up to τ), times d(log g)/da. Grid (units, 1),
// block ≥ C (a thread a position).
extern "C" __global__ void nt_gdc_fin(const float* E1, const float* R, const float* SD, const float* SC, float* da,
                                      int t, int nv, int seq0, int p0, int C, int L)
{
    __shared__ float lg[GDC_MAX];
    const int unit = blockIdx.x, h = unit % nv, lb = unit / nv, bi = seq0 + lb, tau = threadIdx.x;
    const size_t ub = (size_t)unit * C;
    if (tau < L) {
        float acc = R[(ub + tau) * 2];
        for (int sg = 0; sg < tau; sg++) acc += E1[(ub + tau) * C + sg];
        for (int tp = tau + 1; tp < L; tp++) acc -= E1[(ub + tp) * C + tau];
        if (tau == L - 1) {
            float r2 = 0.f;
            for (int sg = 0; sg < L; sg++) r2 += R[(ub + sg) * 2 + 1];
            acc += r2 + __expf(SC[(ub + tau) * 8 + 7]) * SD[unit];
        }
        lg[tau] = acc;
    }
    __syncthreads();
    if (tau == 0) {
        float acc = 0.f;
        for (int x = L - 1; x >= 0; x--) {
            acc += lg[x];
            lg[x] = acc;
        }
    }
    __syncthreads();
    if (tau < L) da[((long)bi * t + p0 + tau) * nv + h] = lg[tau] * SC[(ub + tau) * 8 + 6];
}

// The l2-norm through-gradients of a position's q and k from the adjoints of
// q̂ and k̂ (as `nt_gdn_rev` ends a position). Grid (L, units), block DV.
extern "C" __global__ void nt_gdc_post(const float* cq, const float* KB, const float* QB, const float* SC, float* dqk,
                                       int t, int nv, int nk, int seq0, int p0, int C)
{
    __shared__ float red[4 * (DV / 32)];
    const int tau = blockIdx.x, unit = blockIdx.y, h = unit % nv, lb = unit / nv, bi = seq0 + lb, i = threadIdx.x;
    const int rep = nv / nk, ko = h / rep, kd = nk * DK, vd = nv * DV, cdim = 2 * kd + vd;
    const long row = (long)bi * t + p0 + tau;
    const float* cr = cq + row * cdim;
    const float q = cr[ko * DK + i], k = cr[kd + ko * DK + i];
    const size_t m = ((size_t)unit * C + tau) * DV + i;
    const float aq = QB[m], ak = KB[m];
    float qdot = aq * q, kdot = ak * k, u0 = 0.f, u1 = 0.f;
    nt_block_sum4(qdot, kdot, u0, u1, red);
    const float* sc = SC + ((size_t)unit * C + tau) * 8;
    float* dst = dqk + ((size_t)row * nv + h) * 2 * DK;
    dst[i] = sc[2] * aq - q * qdot * sc[2] / sc[4];
    dst[DK + i] = sc[3] * ak - k * kdot * sc[3] / sc[5];
}

// S̄0 += c_end·S̄_end (the state's own decay through the chunk).
extern "C" __global__ void nt_gdc_snext(float* SN, const float* SE, const float* SC, int C, int L, size_t n)
{
    const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const size_t unit = e / (DK * DV);
    SN[e] += __expf(SC[(unit * C + (L - 1)) * 8 + 7]) * SE[e];
}

// ───────────────────────────── head: KL to the teacher's top-k ─────────────────────────────

// Per token (a block a row): loss over the teacher's support, the logits
// replaced by coef·ws·p_S − coef·p_T (ws = the row's weight when weighted).
extern "C" __global__ void nt_kl_rows(float* __restrict__ LG, const u32* __restrict__ TID, const float* __restrict__ TP,
                                      float* __restrict__ LOSS, u32 v, u32 k, float coef, u32 weighted)
{
    __shared__ float red[32];
    const size_t row = blockIdx.x;
    const u32 t = threadIdx.x, nt = blockDim.x;
    float* lg = LG + row * v;
    const u32* tid = TID + row * k;
    const float* tp = TP + row * k;
    float mx = -3.0e38f;
    for (u32 j = t; j < v; j += nt) mx = fmaxf(mx, lg[j]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((t & 31) == 0) red[t >> 5] = mx;
    __syncthreads();
    mx = -3.0e38f;
    for (u32 i = 0; i < (nt >> 5); i++) mx = fmaxf(mx, red[i]);
    __syncthreads();
    float sum = 0.f;
    for (u32 j = t; j < v; j += nt) sum += __expf(lg[j] - mx);
    const float lse = mx + __logf(nt_block_sum(sum, red));
    __syncthreads();
    float wsum = 0.f;
    for (u32 q = t; q < k; q += nt) wsum += tp[q];
    const float wtot = nt_block_sum(wsum, red);
    const float ws = weighted ? wtot : 1.f;
    __syncthreads();
    float lp = 0.f;
    for (u32 q = t; q < k; q += nt) {
        const float pt = tp[q];
        if (pt > 0.f) lp += pt * (__logf(pt / ws) - (lg[tid[q]] - lse));
    }
    const float ls = nt_block_sum(lp, red);
    if (t == 0) LOSS[row] = ls;
    __syncthreads();
    for (u32 j = t; j < v; j += nt) lg[j] = __expf(lg[j] - lse) * (coef * ws);
    __syncthreads();
    for (u32 q = t; q < k; q += nt) lg[tid[q]] -= tp[q] * coef;
}

// Teacher top-k per row (a block of 256 a row): each thread keeps its best 8
// of a strided scan, the 2048 candidates are bitonic-sorted in shared memory,
// the first k are written with probabilities renormalized over them.
extern "C" __global__ void nt_topk_rows(const float* __restrict__ LG, u32* __restrict__ TID, float* __restrict__ TP, u32 v,
                                        u32 k)
{
    __shared__ float cv[2048];
    __shared__ u32 ci[2048];
    __shared__ float red[32];
    const size_t row = blockIdx.x;
    const u32 t = threadIdx.x;
    const float* lg = LG + row * v;
    float bv[8];
    u32 bi[8];
#pragma unroll
    for (int q = 0; q < 8; q++) {
        bv[q] = -3.0e38f;
        bi[q] = 0;
    }
    for (u32 j = t; j < v; j += 256u) {
        const float x = lg[j];
        if (x > bv[7]) {
            int pos = 7;
            while (pos > 0 && bv[pos - 1] < x) {
                bv[pos] = bv[pos - 1];
                bi[pos] = bi[pos - 1];
                pos--;
            }
            bv[pos] = x;
            bi[pos] = j;
        }
    }
#pragma unroll
    for (int q = 0; q < 8; q++) {
        cv[t * 8 + q] = bv[q];
        ci[t * 8 + q] = bi[q];
    }
    __syncthreads();
    // bitonic sort, descending
    for (u32 kk = 2; kk <= 2048u; kk <<= 1) {
        for (u32 jj = kk >> 1; jj > 0; jj >>= 1) {
            for (u32 e = t; e < 2048u; e += 256u) {
                const u32 x = e ^ jj;
                if (x > e) {
                    const bool desc = (e & kk) == 0;
                    const float a = cv[e], b = cv[x];
                    if ((desc && a < b) || (!desc && a > b)) {
                        cv[e] = b;
                        cv[x] = a;
                        const u32 ti = ci[e];
                        ci[e] = ci[x];
                        ci[x] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
    const float mx = cv[0];
    float s = 0.f;
    for (u32 q = t; q < k; q += 256u) s += __expf(cv[q] - mx);
    const float zsum = nt_block_sum(s, red);
    for (u32 q = t; q < k; q += 256u) {
        TID[row * k + q] = ci[q];
        TP[row * k + q] = __expf(cv[q] - mx) / zsum;
    }
}

// ───────────────────────────── optimizer ─────────────────────────────

// Adam (no weight decay) over every parameter entry: mask 0 = frozen; relf > 0:
// the step is relative to the value's magnitude, floored at relf. The gradient
// is cleared.
extern "C" __global__ void nt_adam(float* __restrict__ w, float* __restrict__ g, float* __restrict__ m1,
                                   float* __restrict__ m2, const float* __restrict__ mask, const float* __restrict__ relf,
                                   size_t n, float lr, float bc1, float bc2, float b1, float b2, float eps, float gmul)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gr = g[i] * gmul;
    g[i] = 0.f;
    if (mask[i] == 0.f) return;
    const float a1 = b1 * m1[i] + (1.f - b1) * gr;
    const float a2 = b2 * m2[i] + (1.f - b2) * gr * gr;
    m1[i] = a1;
    m2[i] = a2;
    const float st = lr * (a1 / bc1) / (sqrtf(a2 / bc2) + eps);
    const float rf = relf[i];
    if (rf > 0.f) w[i] = w[i] - st * fmaxf(fabsf(w[i]), rf);
    else w[i] = w[i] - st;
}

// Partial sums of squares of a[0..n): out[block] (non-finite values propagate).
extern "C" __global__ void nt_sumsq(const float* __restrict__ a, float* __restrict__ out, size_t n, u32 nblk)
{
    __shared__ float red[32];
    float acc = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)blockDim.x * nblk) acc += a[i] * a[i];
    const float s = nt_block_sum(acc, red);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}
