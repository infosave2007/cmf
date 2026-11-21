// Sign training of the binary low-rank paths on CUDA (`gpu_cuda/nvgt.rs`):
// one straight-through latent per sign, updated right after the path's own
// backward. Built without fused multiply-adds (`-fmad=false`): every entry's
// arithmetic is that of `gpu_wgpu/train_sign.rs` and of its host mirrors
// (`host_step_fact`), bit for bit from the same gradient and statistics.

typedef unsigned int u32;

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

// int16 latents two per u32 in the bitmap's row layout, a bf16 first moment,
// a factored second moment, stochastic rounding from PCG.

struct NtSP {
    u32 rows, r, rp, wpr;
    u32 id, seed, step, slot;
    float eta, b1, b2, eps;
    float clampv, hyst, kss, bc2;
    float bc1, tau;
    u32 chunks, pad0;
};

#define NT_CNT_SKIP 63

__device__ __forceinline__ u32 nt_pcg(u32 v) {
    const u32 s = v * 747796405u + 2891336453u;
    const u32 w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
__device__ __forceinline__ float nt_unif24(u32 h) { return (float)(h >> 8u) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float nt_gauss(u32 base, u32 e) {
    const float u1 = nt_unif24(nt_pcg(base ^ nt_pcg(2u * e))) + (1.0f / 16777216.0f);
    const float u2 = nt_unif24(nt_pcg(base ^ nt_pcg(2u * e + 1u)));
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}
__device__ __forceinline__ int nt_init_q(const NtSP& p, u32 base, u32 row, u32 k, bool pos) {
    int mag = 1;
    if (k < p.r) {
        const float z = nt_gauss(base, row * p.rp + k);
        mag = min(max((int)roundf(32767.0f * p.tau * fabsf(z)), 1), 32767);
    }
    return pos ? mag : -mag;
}
__device__ __forceinline__ u32 nt_live_mask(const NtSP& p, u32 w) {
    const u32 k0 = w * 32u;
    if (k0 >= p.r) return 0u;
    if (k0 + 32u <= p.r) return 0xffffffffu;
    return (1u << (p.r - k0)) - 1u;
}
__device__ __forceinline__ u32 nt_newbit(const NtSP& p, int q, u32 old) {
    if (p.hyst <= 0.f) return q >= 0 ? 1u : 0u;
    const int th = (int)(32767.0f * p.hyst);
    if (q > th) return 1u;
    if (q < -th) return 0u;
    return old;
}
__device__ __forceinline__ u32 nt_bf16(float v) {
    const u32 b = __float_as_uint(v);
    return (b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u;
}
__device__ __forceinline__ int nt_half_of(u32 w, u32 h) { return h == 0 ? ((int)(w << 16u)) >> 16 : ((int)w) >> 16; }

// int16 latents from the bitmap: q = ±clamp(round(32767·τ·|z|), 1, 32767),
// sign from the bit; pad ranks get ±1. One thread per bitmap word.
extern "C" __global__ void nt_lat_init(NtSP p, const u32* __restrict__ BITS, u32* __restrict__ LAT)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)p.rows * p.wpr) return;
    const u32 row = (u32)(i / p.wpr), w = (u32)(i % p.wpr);
    const u32 bw = BITS[i];
    const u32 base = nt_pcg(p.seed ^ nt_pcg(p.id ^ 0x5bd1e995u));
    const u32 half = p.rp / 2u;
    for (u32 j = 0; j < 32u; j += 2u) {
        u32 pair = 0;
        for (u32 h = 0; h < 2u; h++) {
            const int q = nt_init_q(p, base, row, w * 32u + j + h, ((bw >> (j + h)) & 1u) == 1u);
            pair |= ((u32)q & 0xffffu) << (16u * h);
        }
        LAT[(size_t)row * half + w * 16u + j / 2u] = pair;
    }
}

// The same values as f32 latents [rows][rp] (the reference optimizer).
extern "C" __global__ void nt_lat_init_f32(NtSP p, const u32* __restrict__ BITS, float* __restrict__ LAT)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)p.rows * p.wpr) return;
    const u32 row = (u32)(i / p.wpr), w = (u32)(i % p.wpr);
    const u32 bw = BITS[i];
    const u32 base = nt_pcg(p.seed ^ nt_pcg(p.id ^ 0x5bd1e995u));
    for (u32 j = 0; j < 32u; j++) {
        const int q = nt_init_q(p, base, row, w * 32u + j, ((bw >> j) & 1u) == 1u);
        LAT[(size_t)row * p.rp + w * 32u + j] = (float)q / 32767.0f;
    }
}

// Row statistics: R_i ← b2·R_i + (1−b2)·mean_{k<r} G_ik² (non-finite = 0,
// counted). A block of 256 a row.
extern "C" __global__ void nt_g_rowsq(NtSP p, const float* __restrict__ G, float* __restrict__ VROW, u32* CNT)
{
    __shared__ float red[32];
    __shared__ u32 redn[32];
    const u32 row = blockIdx.x, t = threadIdx.x;
    float acc = 0.f;
    u32 bad = 0;
    for (u32 k = t; k < p.r; k += 256u) {
        const float v = G[(size_t)row * p.rp + k];
        if (nt_fin(v)) acc += v * v;
        else bad++;
    }
    for (int o = 16; o > 0; o >>= 1) bad += __shfl_xor_sync(0xffffffffu, bad, o);
    if ((t & 31) == 0) redn[t >> 5] = bad;
    const float s = nt_block_sum(acc, red);
    if (t == 0 && CNT[NT_CNT_SKIP] == 0u) {
        u32 nb = 0;
        for (u32 i = 0; i < 8u; i++) nb += redn[i];
        VROW[row] = p.b2 * VROW[row] + (1.0f - p.b2) * (s / (float)p.r);
        if (nb > 0) atomicAdd(&CNT[40], nb);
    }
}

// CNT[SKIP] = 1 when any of the step's n per-token losses is not finite. One block.
extern "C" __global__ void nt_loss_flag(const float* __restrict__ LOSS, u32* CNT, u32 n)
{
    u32 bad = 0;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
        if (!nt_fin(LOSS[i])) bad = 1;
    }
    if (bad) CNT[NT_CNT_SKIP] = 1u;
}

// Column partial sums of G² over 512-row chunks: PART[chunk][col].
extern "C" __global__ void nt_g_colsq(NtSP p, const float* __restrict__ G, float* __restrict__ PART)
{
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= p.rp) return;
    const u32 r0 = blockIdx.y * 512u, r1 = min(r0 + 512u, p.rows);
    float acc = 0.f;
    for (u32 row = r0; row < r1; row++) {
        const float v = G[(size_t)row * p.rp + col];
        if (nt_fin(v)) acc += v * v;
    }
    PART[(size_t)blockIdx.y * p.rp + col] = acc;
}

// C_k ← b2·C_k + (1−b2)·mean_i G_ik² for k < r.
extern "C" __global__ void nt_g_colfin(NtSP p, const float* __restrict__ PART, float* __restrict__ VCOL, const u32* CNT)
{
    const u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= p.r || CNT[NT_CNT_SKIP] != 0u) return;
    float s = 0.f;
    for (u32 c = 0; c < p.chunks; c++) s += PART[(size_t)c * p.rp + col];
    VCOL[col] = p.b2 * VCOL[col] + (1.0f - p.b2) * (s / (float)p.rows);
}

// SCAL[0] = mean_{k<r} C_k. One block of 256.
extern "C" __global__ void nt_g_cmean(NtSP p, const float* __restrict__ VCOL, float* __restrict__ SCAL, const u32* CNT)
{
    __shared__ float red[32];
    float acc = 0.f;
    for (u32 k = threadIdx.x; k < p.r; k += 256u) acc += VCOL[k];
    const float s = nt_block_sum(acc, red);
    if (threadIdx.x == 0 && CNT[NT_CNT_SKIP] == 0u) SCAL[0] = s / (float)p.r;
}

// The factored-second-moment step, with a bf16 first moment when `mom`
// (MOM: two per u32, the latent layout). One thread per (row, word).
extern "C" __global__ void nt_sign_upd(NtSP p, const float* __restrict__ G, u32* __restrict__ LAT, u32* __restrict__ MOM,
                                       const float* __restrict__ VROW, const float* __restrict__ VCOL,
                                       u32* __restrict__ BITS, const float* __restrict__ SCAL, u32* CNT, u32 mom)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)p.rows * p.wpr || CNT[NT_CNT_SKIP] != 0u) return;
    const u32 row = (u32)(i / p.wpr), w = (u32)(i % p.wpr);
    const u32 live = nt_live_mask(p, w);
    if (live == 0u) return;
    const float mm = SCAL[0];
    const float inv = 1.0f / (mm * p.bc2);
    const float epsd = p.eps * sqrtf(mm / p.bc2);
    const float rv = VROW[row];
    const u32 base = nt_pcg(p.seed ^ nt_pcg(p.step ^ nt_pcg(p.id ^ 0x2545f491u)));
    const u32 bw = BITS[i];
    u32 nb = bw, sat = 0, clp = 0;
    const u32 half = p.rp / 2u;
    for (u32 j = 0; j < 32u; j += 2u) {
        const size_t li = (size_t)row * half + w * 16u + j / 2u;
        const u32 pw = LAT[li];
        const u32 mw = mom ? MOM[li] : 0u;
        u32 outw = 0, outm = 0;
        for (u32 h = 0; h < 2u; h++) {
            int q = nt_half_of(pw, h);
            const u32 jb = j + h;
            u32 mb = (mw >> (16u * h)) & 0xffffu;
            if (((live >> jb) & 1u) == 1u) {
                const u32 k = w * 32u + jb;
                const u32 e = row * p.rp + k;
                float gv = G[(size_t)row * p.rp + k];
                if (!nt_fin(gv)) gv = 0.f;
                float num = gv;
                if (mom) {
                    const float mn = p.b1 * __uint_as_float(mb << 16u) + (1.0f - p.b1) * gv;
                    mb = nt_bf16(mn);
                    num = mn / p.bc1;
                }
                const float den = sqrtf(fmaxf(rv * VCOL[k] * inv, 0.f)) + epsd;
                float u = 0.f;
                if (den > 0.f) u = num / den;
                if (!nt_fin(u)) u = 0.f;
                if (fabsf(u) > p.clampv) {
                    u = fminf(fmaxf(u, -p.clampv), p.clampv);
                    clp++;
                }
                const float delta = -32767.0f * p.eta * u;
                const float fl = floorf(delta);
                const float fr = delta - fl;
                int qn = q + (int)fl + (nt_unif24(nt_pcg(base ^ nt_pcg(e))) < fr ? 1 : 0);
                if (qn > 32767 || qn < -32767) {
                    qn = min(max(qn, -32767), 32767);
                    sat++;
                }
                q = qn;
                const u32 b = nt_newbit(p, q, (bw >> jb) & 1u);
                nb = (nb & ~(1u << jb)) | (b << jb);
            }
            outw |= ((u32)q & 0xffffu) << (16u * h);
            outm |= mb << (16u * h);
        }
        LAT[li] = outw;
        if (mom) MOM[li] = outm;
    }
    BITS[i] = nb;
    const u32 f = __popc(bw ^ nb);
    if (f > 0) atomicAdd(&CNT[p.slot], f);
    if (sat > 0) atomicAdd(&CNT[41], sat);
    if (clp > 0) atomicAdd(&CNT[42], clp);
}

// Reference: f32 latent, Adam over the SmoothSign gradient G·κ·(1 − tanh²(κΛ)).
extern "C" __global__ void nt_sign_upd_ref(NtSP p, const float* __restrict__ G, float* __restrict__ LAT,
                                           float* __restrict__ MOM, float* __restrict__ VV, u32* __restrict__ BITS,
                                           u32* CNT)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)p.rows * p.wpr || CNT[NT_CNT_SKIP] != 0u) return;
    const u32 row = (u32)(i / p.wpr), w = (u32)(i % p.wpr);
    const u32 live = nt_live_mask(p, w);
    if (live == 0u) return;
    const u32 bw = BITS[i];
    u32 nb = bw;
    for (u32 jb = 0; jb < 32u; jb++) {
        if (((live >> jb) & 1u) == 1u) {
            const size_t e = (size_t)row * p.rp + w * 32u + jb;
            float lam = LAT[e];
            float gv = G[e];
            if (!nt_fin(gv)) gv = 0.f;
            const float th = tanhf(p.kss * lam);
            const float gs = gv * p.kss * (1.0f - th * th);
            const float m = p.b1 * MOM[e] + (1.0f - p.b1) * gs;
            const float v = p.b2 * VV[e] + (1.0f - p.b2) * gs * gs;
            MOM[e] = m;
            VV[e] = v;
            const float den = sqrtf(v / p.bc2) + 1e-30f;
            lam = lam - p.eta * (m / p.bc1) / den;
            LAT[e] = lam;
            u32 b = lam >= 0.f ? 1u : 0u;
            if (p.hyst > 0.f) {
                b = (bw >> jb) & 1u;
                if (lam > p.hyst) b = 1u;
                if (lam < -p.hyst) b = 0u;
            }
            nb = (nb & ~(1u << jb)) | (b << jb);
        }
    }
    BITS[i] = nb;
    const u32 f = __popc(bw ^ nb);
    if (f > 0) atomicAdd(&CNT[p.slot], f);
}

// DIFF[slot] += popcount((BITS ^ BITS0) & live): distinct flips of one matrix.
extern "C" __global__ void nt_bits_diff(NtSP p, const u32* __restrict__ BITS, const u32* __restrict__ BITS0, u32* DIFF)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)p.rows * p.wpr) return;
    const u32 f = __popc((BITS[i] ^ BITS0[i]) & nt_live_mask(p, (u32)(i % p.wpr)));
    if (f > 0) atomicAdd(&DIFF[p.slot], f);
}
