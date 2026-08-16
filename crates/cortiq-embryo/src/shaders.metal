// Cortiq Embryo — training kernels (Metal, Apple Silicon).
//
// Everything the birth/growth trainer needs on the device, hand-written:
// no MPS, no framework. The GEMM is the workhorse (all three orientations
// of a linear layer's forward/backward through the TA/TB function
// constants); the rest are the elementwise/reduction companions of the
// fixed graph. f32 storage, f32 accumulate — this is a TRAINER, the
// gradients want the precision (the runtime's inference kernels use
// half tiles; this is a deliberate difference).

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------
// GEMM:  C[M,N] = alpha · op(A)[M,K] · op(B)[K,N] + beta · C[M,N]
//
//   TA = false: A stored row-major [M,K], lda ≥ K   (A[m,k] = A[m·lda + k])
//   TA = true : A stored row-major [K,M], lda ≥ M   (A[m,k] = A[k·lda + m])
//   TB = false: B stored row-major [K,N], ldb ≥ N   (B[k,n] = B[k·ldb + n])
//   TB = true : B stored row-major [N,K], ldb ≥ K   (B[k,n] = B[n·ldb + k])
//
// A linear layer y = x·Wᵀ (x[M,K], W[N,K]) is (TA=0,TB=1);
// its input grad dx = dy·W (dy[M,N], W[N,K]→[K',N'] with K'=N) is (0,0);
// its weight grad dW = dyᵀ·x (dyᵀ: [N,M] from dy[M,N]; x[M,K]) is (1,0).
//
// Tile 64×64×32, 128 threads = 4 simdgroups, each simdgroup owns a 32×32
// quadrant as 4×4 simdgroup_float8x8 accumulators. Host guarantees
// M%64 == 0, N%64 == 0, K%32 == 0 and 16-byte aligned rows (all Embryo
// shapes are multiples of 64 by construction; the host asserts).
// ---------------------------------------------------------------------

constant bool TA [[function_constant(0)]];
constant bool TB [[function_constant(1)]];

struct GemmArgs {
    // batch strides (elements) for the (b, h, c) decomposition of tgid.z
    ulong sa_b, sa_h, sa_c;
    ulong sb_b, sb_h, sb_c;
    ulong sc_b, sc_h, sc_c;
    uint M, N, K;
    uint lda, ldb, ldc;
    float alpha, beta;
    uint nb_h, nb_c;   // z = (b·nb_h + h)·nb_c + c
    uint mask;         // 1: causal — C[i,j] = 0 for j > i (global indices)
    uint kdyn;         // 1: K = min(round64(kcount[z]), K) — dynamic reduction length
};

#define BM 64u
#define BN 64u
#define BK 32u
#define NTHREADS 128u

kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant GemmArgs&  g [[buffer(3)]],
    device const uint*  kcount [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    uint Kdim = g.K;
    {
        uint z = tgid.z;
        if (g.kdyn == 1u) { Kdim = min(((kcount[z] + 63u) / 64u) * 64u, g.K); }
        uint cb = z % g.nb_c;
        uint rem = z / g.nb_c;
        uint hb = rem % g.nb_h;
        uint bb = rem / g.nb_h;
        A += bb * g.sa_b + hb * g.sa_h + cb * g.sa_c;
        B += bb * g.sb_b + hb * g.sb_h + cb * g.sb_c;
        C += bb * g.sc_b + hb * g.sc_h + cb * g.sc_c;
    }
    // One 16 KB arena: sA = [BM][BK] (m-major, k contiguous),
    // sB = [BK][BN] (k-major, n contiguous). After the K loop the same
    // 4096 floats are the [BM][BN] C staging tile.
    threadgroup float smem[BM * BN];
    threadgroup float* sA = smem;
    threadgroup float* sB = smem + BM * BK;

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint sm = (sgid >> 1) * 32u;   // this simdgroup's rows in the tile
    const uint sn = (sgid & 1u) * 32u;   // this simdgroup's cols in the tile

    simdgroup_float8x8 acc[4][4];
    #pragma clang loop unroll(full)
    for (short i = 0; i < 4; ++i) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < Kdim; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- A tile: BM×BK = 2048 floats, 512 float4, 4 per thread ----
        if (!TA) {
            // contiguous along k: 64 rows × 8 float4
            #pragma clang loop unroll(full)
            for (uint it = 0; it < 4u; ++it) {
                uint i = tid + it * NTHREADS;
                uint r = i >> 3, c4 = i & 7u;
                float4 v = *(device const float4*)(A + (ulong)(m0 + r) * g.lda + k0 + c4 * 4u);
                *(threadgroup float4*)(sA + r * BK + c4 * 4u) = v;
            }
        } else {
            // A stored [K,M]: contiguous along m: 32 k-rows × 16 float4,
            // scattered into sA transposed (scalar stores; see the
            // runtime's note on threadgroup pointer casts).
            #pragma clang loop unroll(full)
            for (uint it = 0; it < 4u; ++it) {
                uint i = tid + it * NTHREADS;
                uint kk = i >> 4, c4 = i & 15u;
                float4 v = *(device const float4*)(A + (ulong)(k0 + kk) * g.lda + m0 + c4 * 4u);
                uint r = c4 * 4u;
                sA[(r + 0u) * BK + kk] = v.x;
                sA[(r + 1u) * BK + kk] = v.y;
                sA[(r + 2u) * BK + kk] = v.z;
                sA[(r + 3u) * BK + kk] = v.w;
            }
        }
        // ---- B tile: BK×BN = 2048 floats ----
        if (!TB) {
            // B[K,N]: contiguous along n: 32 k-rows × 16 float4
            #pragma clang loop unroll(full)
            for (uint it = 0; it < 4u; ++it) {
                uint i = tid + it * NTHREADS;
                uint kk = i >> 4, c4 = i & 15u;
                float4 v = *(device const float4*)(B + (ulong)(k0 + kk) * g.ldb + n0 + c4 * 4u);
                *(threadgroup float4*)(sB + kk * BN + c4 * 4u) = v;
            }
        } else {
            // B stored [N,K]: contiguous along k: 64 n-rows × 8 float4,
            // scattered into sB transposed.
            #pragma clang loop unroll(full)
            for (uint it = 0; it < 4u; ++it) {
                uint i = tid + it * NTHREADS;
                uint r = i >> 3, c4 = i & 7u;
                float4 v = *(device const float4*)(B + (ulong)(n0 + r) * g.ldb + k0 + c4 * 4u);
                uint kk = c4 * 4u;
                sB[(kk + 0u) * BN + r] = v.x;
                sB[(kk + 1u) * BN + r] = v.y;
                sB[(kk + 2u) * BN + r] = v.z;
                sB[(kk + 3u) * BN + r] = v.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[4];
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(a[i], sA + (sm + i * 8u) * BK + kk, BK);
            }
            #pragma clang loop unroll(full)
            for (short j = 0; j < 4; ++j) {
                simdgroup_load(b[j], sB + kk * BN + sn + j * 8u, BN);
            }
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                #pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
    }

    // ---- epilogue: stage the 64×64 tile, then alpha/beta with float4 stores ----
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (short i = 0; i < 4; ++i) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], smem + (sm + i * 8u) * BN + sn + j * 8u, BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool accumulate = g.beta != 0.0f;
    #pragma clang loop unroll(full)
    for (uint it = 0; it < 8u; ++it) {
        uint i = tid + it * NTHREADS;      // 0..1023 float4 slots
        uint r = i >> 4, c4 = i & 15u;
        float4 v = *(threadgroup float4*)(smem + r * BN + c4 * 4u) * g.alpha;
        device float4* dst = (device float4*)(C + (ulong)(m0 + r) * g.ldc + n0 + c4 * 4u);
        if (accumulate) { v += *dst * g.beta; }
        if (g.mask == 1u) {
            uint i = m0 + r, j0 = n0 + c4 * 4u;
            if (j0 + 0u > i) v.x = 0.0f;
            if (j0 + 1u > i) v.y = 0.0f;
            if (j0 + 2u > i) v.z = 0.0f;
            if (j0 + 3u > i) v.w = 0.0f;
        }
        *dst = v;
    }
}

// ---------------------------------------------------------------------
// Elementwise / reduction companions (one thread per element or row).
// ---------------------------------------------------------------------

// y = a*x + b*y  (axpby over n floats)
kernel void axpby_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant float&     a [[buffer(2)]],
    constant float&     b [[buffer(3)]],
    constant uint&      n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { y[gid] = a * x[gid] + b * y[gid]; }
}

// AdamW (decoupled weight decay), one thread per parameter.
// m = b1·m + (1−b1)·g ; v = b2·v + (1−b2)·g² ;
// p -= lr·( m̂/(√v̂+eps) + wd·p ),  m̂ = m/(1−b1ᵗ), v̂ = v/(1−b2ᵗ)
struct AdamArgs {
    uint  n;
    float lr, beta1, beta2, eps, wd;
    float bc1, bc2;      // 1/(1−b1ᵗ), 1/(1−b2ᵗ)
    float gscale;        // global grad scale (1/accum · clip factor)
};
kernel void adamw_f32(
    device float*       p [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device float*       m [[buffer(2)]],
    device float*       v [[buffer(3)]],
    constant AdamArgs&  a [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.n) return;
    float gr = g[gid] * a.gscale;
    float mm = a.beta1 * m[gid] + (1.0f - a.beta1) * gr;
    float vv = a.beta2 * v[gid] + (1.0f - a.beta2) * gr * gr;
    m[gid] = mm;
    v[gid] = vv;
    float upd = (mm * a.bc1) / (sqrt(vv * a.bc2) + a.eps);
    p[gid] -= a.lr * (upd + a.wd * p[gid]);
}

// Sum of squares of n floats into partial[tg] (one threadgroup = 256
// threads, 4 floats each per step) — the grad-norm clip reads the
// partials back and finishes on the host.
kernel void sumsq_f32(
    device const float* x       [[buffer(0)]],
    device float*       partial [[buffer(1)]],
    constant uint&      n       [[buffer(2)]],
    uint gid  [[thread_position_in_grid]],
    uint tpg  [[threads_per_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[32];
    float s = 0.0f;
    for (uint i = gid; i < n; i += tpg) { s += x[i] * x[i]; }
    s = simd_sum(s);
    if (lane == 0) { red[sgid] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float t = (lane < (tgs + 31u) / 32u) ? red[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) { partial[tgid] = t; }
    }
}

// RMSNorm forward, one threadgroup (128 threads) per row of width d
// (d ≤ 128·4 handled by the loop). y = x · inv · w,  inv = 1/√(mean x²+eps).
// Stores inv per row for the backward.
kernel void rmsnorm_fwd_f32(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    device float*       inv [[buffer(3)]],
    constant uint&      d   [[buffer(4)]],
    constant float&     eps [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[4];
    device const float* xr = x + (ulong)row * d;
    device float* yr = y + (ulong)row * d;
    float s = 0.0f;
    for (uint i = tid; i < d; i += 128u) { float v = xr[i]; s += v * v; }
    s = simd_sum(s);
    if (lane == 0) { red[sgid] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3];
    float r = rsqrt(tot / (float)d + eps);
    if (tid == 0) { inv[row] = r; }
    for (uint i = tid; i < d; i += 128u) { yr[i] = xr[i] * r * w[i]; }
}

// RMSNorm backward (per row):  g = dy·w ;  dx = inv·(g − x·inv²·(g·x)/d)
// dw is accumulated on the host side via a GEMM-free column reduction
// kernel (rmsnorm_dw_f32) to keep this one race-free.
kernel void rmsnorm_bwd_dx_f32(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device const float* dy  [[buffer(2)]],
    device const float* inv [[buffer(3)]],
    device float*       dx  [[buffer(4)]],
    constant uint&      d   [[buffer(5)]],
    constant float&     beta [[buffer(6)]],   // dx = dx·beta + result
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[4];
    device const float* xr = x + (ulong)row * d;
    device const float* dyr = dy + (ulong)row * d;
    device float* dxr = dx + (ulong)row * d;
    float r = inv[row];
    float dot = 0.0f;
    for (uint i = tid; i < d; i += 128u) { dot += dyr[i] * w[i] * xr[i]; }
    dot = simd_sum(dot);
    if (lane == 0) { red[sgid] = dot; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = red[0] + red[1] + red[2] + red[3];
    float c = r * r * r * tot / (float)d;
    for (uint i = tid; i < d; i += 128u) {
        float v = r * dyr[i] * w[i] - c * xr[i];
        dxr[i] = (beta == 0.0f) ? v : (dxr[i] * beta + v);
    }
}

// dw[j] += Σ_rows dy[row,j]·x[row,j]·inv[row]   — one thread per column j,
// looping over rows (rows ~ B·T = 16k: fine for d ≤ 1k columns).
kernel void rmsnorm_dw_f32(
    device const float* x    [[buffer(0)]],
    device const float* dy   [[buffer(1)]],
    device const float* inv  [[buffer(2)]],
    device float*       dw   [[buffer(3)]],
    constant uint&      d    [[buffer(4)]],
    constant uint&      rows [[buffer(5)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= d) return;
    float s = 0.0f;
    for (uint r = 0; r < rows; ++r) {
        s += dy[(ulong)r * d + j] * x[(ulong)r * d + j] * inv[r];
    }
    dw[j] += s;
}

// SwiGLU forward: h = silu(gate) · up   (elementwise over n)
kernel void swiglu_fwd_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       h    [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float gv = gate[gid];
    float sg = gv / (1.0f + exp(-gv));
    h[gid] = sg * up[gid];
}

// SwiGLU backward: dgate = dh·up·silu'(gate) ; dup = dh·silu(gate)
kernel void swiglu_bwd_f32(
    device const float* gate  [[buffer(0)]],
    device const float* up    [[buffer(1)]],
    device const float* dh    [[buffer(2)]],
    device float*       dgate [[buffer(3)]],
    device float*       dup   [[buffer(4)]],
    constant uint&      n     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float gv = gate[gid];
    float sig = 1.0f / (1.0f + exp(-gv));
    float sg = gv * sig;
    float dsg = sig * (1.0f + gv * (1.0f - sig));
    float d = dh[gid];
    dgate[gid] = d * up[gid] * dsg;
    dup[gid] = d * sg;
}

// Embedding gather: out[row,:] = E[tok[row],:]  (d floats per row)
kernel void embed_gather_f32(
    device const float* E   [[buffer(0)]],
    device const uint*  tok [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      d   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])   // x: column, y: row
{
    if (gid.x >= d) return;
    out[(ulong)gid.y * d + gid.x] = E[(ulong)tok[gid.y] * d + gid.x];
}

// Fused softmax cross-entropy over a row of `n` logits, one threadgroup
// (256 threads) per row. Writes loss[row] = −log p[target] and, in
// place of the logits, dlogits = (p − onehot)·scale. Row-wise two-pass
// (max, then sum) in f32 with the max subtracted — n ≤ 64k.
kernel void softmax_ce_f32(
    device float*       logits [[buffer(0)]],
    device const uint*  target [[buffer(1)]],
    device float*       loss   [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    constant float&     scale  [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    device float* lr = logits + (ulong)row * n;
    float mx = -INFINITY;
    for (uint i = tid; i < n; i += 256u) { mx = max(mx, lr[i]); }
    mx = simd_max(mx);
    if (lane == 0) { red[sgid] = mx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint s = 1; s < 8u; ++s) { mx = max(mx, red[s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = tid; i < n; i += 256u) { sum += exp(lr[i] - mx); }
    sum = simd_sum(sum);
    if (lane == 0) { red[sgid] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; ++s) { tot += red[s]; }
    float lse = mx + log(tot);
    uint t = target[row];
    if (tid == 0) { loss[row] = lse - lr[t]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n; i += 256u) {
        float p = exp(lr[i] - lse);
        lr[i] = (p - ((i == t) ? 1.0f : 0.0f)) * scale;
    }
}

// ---------------------------------------------------------------------
// hybrid_k mixer — chunked scan (chunk C = 64), the runtime's vmf_phase
// core + κ write gate (linear_core.rs::phase_step) made trainable.
//
//   S_t = γ ⊙ S_{t−1} + κ_t·φk_t ⊗ v_t ,  o_t = φq_tᵀ·S_t ,  φ = [cos θ; sin θ]
//
// Layouts (row-major, one row per (b,t)):
//   phq, phk : [B·T, nh·P2]    kv = κ⊙v, out, dout, dkv : [B·T, nh·dv]
//   pow      : [nh, C+1, P2]   γ_{h,f}^δ, δ = 0..C
//   states   : [B, nh, nchunks+1, P2, dv]  S entering chunk c (S_0 = 0)
//   dstates  : [B, nh, nchunks+1, P2, dv]  ∂L/∂S_c from chunks ≥ c only
// P2 ≤ 64 (nph ≤ 32), dv ≤ 128 (one thread per value channel).
// ---------------------------------------------------------------------

#define HK_C   64u
#define HK_P2  64u

struct HkArgs {
    uint B, T, nh, nph, dv;   // P2 = 2·nph
};

// φ tables from θ:  phq[row][h·P2 + i] = cos θ, [.. + nph + i] = sin θ
kernel void hk_phi_f32(
    device const float* th  [[buffer(0)]],
    device float*       ph  [[buffer(1)]],
    constant HkArgs&    a   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh·nph
{
    uint rows = a.B * a.T;
    uint per_row = a.nh * a.nph;
    if (gid >= rows * per_row) return;
    uint row = gid / per_row, r = gid % per_row;
    uint h = r / a.nph, i = r % a.nph;
    uint p2 = 2u * a.nph;
    float t = th[gid];
    ph[(ulong)row * a.nh * p2 + h * p2 + i]         = cos(t);
    ph[(ulong)row * a.nh * p2 + h * p2 + a.nph + i] = sin(t);
}

// dθ from dφ:  dθ_i = −sin θ_i·dφ[i] + cos θ_i·dφ[nph+i]
kernel void hk_dtheta_f32(
    device const float* th  [[buffer(0)]],
    device const float* dph [[buffer(1)]],
    device float*       dth [[buffer(2)]],
    constant HkArgs&    a   [[buffer(3)]],
    constant float&     beta [[buffer(4)]],   // dth = beta·dth + result
    uint gid [[thread_position_in_grid]])
{
    uint rows = a.B * a.T;
    uint per_row = a.nh * a.nph;
    if (gid >= rows * per_row) return;
    uint row = gid / per_row, r = gid % per_row;
    uint h = r / a.nph, i = r % a.nph;
    uint p2 = 2u * a.nph;
    float t = th[gid];
    ulong base = (ulong)row * a.nh * p2 + h * p2;
    float g = -sin(t) * dph[base + i] + cos(t) * dph[base + a.nph + i];
    dth[gid] = (beta == 0.0f) ? g : (beta * dth[gid] + g);
}

// kv = κ ⊙ v  (κ per (row, head))
kernel void hk_kv_f32(
    device const float* v   [[buffer(0)]],
    device const float* kap [[buffer(1)]],
    device float*       kv  [[buffer(2)]],
    constant HkArgs&    a   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh·dv
{
    uint rows = a.B * a.T;
    uint per_row = a.nh * a.dv;
    if (gid >= rows * per_row) return;
    uint row = gid / per_row, h = (gid % per_row) / a.dv;
    kv[gid] = v[gid] * kap[row * a.nh + h];
}

// dv = κ⊙dkv ;  dκ = Σ_d dkv·v   (one thread per (row, head))
kernel void hk_dkv_split_f32(
    device const float* v    [[buffer(0)]],
    device const float* kap  [[buffer(1)]],
    device const float* dkv  [[buffer(2)]],
    device float*       dv_o [[buffer(3)]],
    device float*       dkap [[buffer(4)]],
    constant HkArgs&    a    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh
{
    if (gid >= a.B * a.T * a.nh) return;
    uint row = gid / a.nh, h = gid % a.nh;
    float k = kap[gid];
    ulong base = (ulong)row * a.nh * a.dv + h * a.dv;
    float s = 0.0f;
    for (uint d = 0; d < a.dv; ++d) {
        float g = dkv[base + d];
        dv_o[base + d] = k * g;
        s += g * v[base + d];
    }
    dkap[gid] = s;
}

// Forward states: one threadgroup per (b,h), threads = dv × FS where each
// thread owns value channel d and a 16-feature slice fg (FS = ceil(P2/16))
// — 16 accumulators per thread keeps the register file small enough for
// several threadgroups per core; φk of the chunk is staged once per
// threadgroup. The literal recurrence; S at every chunk boundary written.
#define HK_FT 16u
kernel void hk_states_fwd_f32(
    device const float* phk    [[buffer(0)]],
    device const float* kv     [[buffer(1)]],
    device const float* pow_t  [[buffer(2)]],
    device float*       states [[buffer(3)]],
    constant HkArgs&    a      [[buffer(4)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float sk[HK_C * HK_P2];   // φk of the current chunk
    uint b = tg / a.nh, h = tg % a.nh;
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    uint d = tid % a.dv, fg = tid / a.dv;
    uint f0 = fg * HK_FT;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    float S[HK_FT], gm[HK_FT];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < HK_FT; ++i) { S[i] = 0.0f; gm[i] = (f0 + i < p2) ? gam[f0 + i] : 0.0f; }
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    for (uint c = 0; c < nchunks; ++c) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < HK_FT; ++i) {
            if (f0 + i < p2) states[st_base + ((ulong)c * p2 + f0 + i) * a.dv + d] = S[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < HK_C * HK_P2; i += nth) {
            uint s = i / HK_P2, f = i % HK_P2;
            sk[i] = (f < p2) ? phk[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * p2 + f] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < HK_C; ++s) {
            float kvv = kv[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * a.dv + d];
            threadgroup const float* skr = sk + s * HK_P2 + f0;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < HK_FT; ++i) S[i] = gm[i] * S[i] + skr[i] * kvv;
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < HK_FT; ++i) {
        if (f0 + i < p2) states[st_base + ((ulong)nchunks * p2 + f0 + i) * a.dv + d] = S[i];
    }
}

// Forward per chunk: one threadgroup per (b,h,c), nth threads (≥ dv).
//   A[t,s] = Σ_f φq_t[f]·φk_s[f]·γ_f^{t−s}  (s ≤ t)
//   o_t[d] = Σ_{s≤t} A[t,s]·kv_s[d] + Σ_f φq_t[f]·γ_f^{t−t0+1}·S_c[f][d]
kernel void hk_chunk_fwd_f32(
    device const float* phq    [[buffer(0)]],
    device const float* phk    [[buffer(1)]],
    device const float* kv     [[buffer(2)]],
    device const float* pow_t  [[buffer(3)]],
    device const float* states [[buffer(4)]],
    device float*       out    [[buffer(5)]],
    constant HkArgs&    a      [[buffer(6)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float A[HK_C * HK_C];
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    uint c = tg % nchunks, bh = tg / nchunks;
    uint b = bh / a.nh, h = bh % a.nh;
    uint t0 = c * HK_C;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;   // pw[δ·p2 + f]
    // stage 1: A
    for (uint idx = tid; idx < HK_C * HK_C; idx += nth) {
        uint t = idx / HK_C, s = idx % HK_C;
        float acc = 0.0f;
        if (s <= t) {
            device const float* q = phq + ((ulong)(b * a.T + t0 + t) * a.nh + h) * p2;
            device const float* k = phk + ((ulong)(b * a.T + t0 + s) * a.nh + h) * p2;
            device const float* g = pw + (t - s) * p2;
            for (uint f = 0; f < p2; ++f) acc += q[f] * k[f] * g[f];
        }
        A[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // stage 2: outputs, thread per d
    if (tid < a.dv) {
        uint d = tid;
        float Sc[HK_P2];
        device const float* st = states + (((ulong)bh * (nchunks + 1u) + c) * p2) * a.dv + d;
        for (uint f = 0; f < p2; ++f) Sc[f] = st[(ulong)f * a.dv];
        for (uint t = 0; t < HK_C; ++t) {
            float o = 0.0f;
            for (uint s = 0; s <= t; ++s) {
                o += A[t * HK_C + s] * kv[((ulong)(b * a.T + t0 + s) * a.nh + h) * a.dv + d];
            }
            device const float* q = phq + ((ulong)(b * a.T + t0 + t) * a.nh + h) * p2;
            device const float* g = pw + (t + 1u) * p2;
            for (uint f = 0; f < p2; ++f) o += q[f] * g[f] * Sc[f];
            out[((ulong)(b * a.T + t0 + t) * a.nh + h) * a.dv + d] = o;
        }
    }
}

// Backward state gradients (reverse over positions), same thread layout
// as hk_states_fwd_f32:
//   G ← γ ⊙ (G + φq_t ⊗ do_t)   for t = T−1 … 0,
// G at each chunk boundary written to dstates[c] = ∂L/∂S_c restricted
// to reads by chunks ≥ c (dstates[nchunks] = 0).
kernel void hk_dstates_bwd_f32(
    device const float* phq     [[buffer(0)]],
    device const float* dout    [[buffer(1)]],
    device const float* pow_t   [[buffer(2)]],
    device float*       dstates [[buffer(3)]],
    constant HkArgs&    a       [[buffer(4)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float sq[HK_C * HK_P2];
    uint b = tg / a.nh, h = tg % a.nh;
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    uint d = tid % a.dv, fg = tid / a.dv;
    uint f0 = fg * HK_FT;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    float G[HK_FT], gm[HK_FT];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < HK_FT; ++i) { G[i] = 0.0f; gm[i] = (f0 + i < p2) ? gam[f0 + i] : 0.0f; }
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < HK_FT; ++i) {
        if (f0 + i < p2) dstates[st_base + ((ulong)nchunks * p2 + f0 + i) * a.dv + d] = 0.0f;
    }
    for (uint cc = 0; cc < nchunks; ++cc) {
        uint c = nchunks - 1u - cc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < HK_C * HK_P2; i += nth) {
            uint s = i / HK_P2, f = i % HK_P2;
            sq[i] = (f < p2) ? phq[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * p2 + f] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ss = 0; ss < HK_C; ++ss) {
            uint s = HK_C - 1u - ss;
            float dov = dout[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * a.dv + d];
            threadgroup const float* sqr = sq + s * HK_P2 + f0;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < HK_FT; ++i) G[i] = gm[i] * (G[i] + sqr[i] * dov);
        }
        #pragma clang loop unroll(full)
        for (uint i = 0; i < HK_FT; ++i) {
            if (f0 + i < p2) dstates[st_base + ((ulong)c * p2 + f0 + i) * a.dv + d] = G[i];
        }
    }
}

// Backward per chunk. Writes dkv (→ hk_dkv_split), dphq/dphk (→ hk_dtheta).
//   dkv_s[d]  = Σ_{t≥s} A[t,s]·do_t[d] + Σ_f φk_s[f]·γ_f^{C−1−s}·Gn[f][d]
//   dA[t,s]   = Σ_d do_t[d]·kv_s[d]
//   dφq_t[f]  = Σ_{s≤t} dA[t,s]·φk_s[f]·γ_f^{t−s} + γ_f^{t+1}·Σ_d do_t[d]·Sc[f][d]
//   dφk_s[f]  = Σ_{t≥s} dA[t,s]·φq_t[f]·γ_f^{t−s} + γ_f^{C−1−s}·Σ_d kv_s[d]·Gn[f][d]
// (t, s relative to the chunk; Sc = states[c], Gn = dstates[c+1])
kernel void hk_chunk_bwd_f32(
    device const float* phq     [[buffer(0)]],
    device const float* phk     [[buffer(1)]],
    device const float* kv      [[buffer(2)]],
    device const float* pow_t   [[buffer(3)]],
    device const float* states  [[buffer(4)]],
    device const float* dstates [[buffer(5)]],
    device const float* dout    [[buffer(6)]],
    device float*       dkv     [[buffer(7)]],
    device float*       dphq    [[buffer(8)]],
    device float*       dphk    [[buffer(9)]],
    constant HkArgs&    a       [[buffer(10)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float A[HK_C * HK_C];    // A, then reused for dA
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    uint c = tg % nchunks, bh = tg / nchunks;
    uint b = bh / a.nh, h = bh % a.nh;
    uint t0 = c * HK_C;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;
    #define ROW(t) ((ulong)(b * a.T + t0 + (t)) * a.nh + h)
    // stage 1: A
    for (uint idx = tid; idx < HK_C * HK_C; idx += nth) {
        uint t = idx / HK_C, s = idx % HK_C;
        float acc = 0.0f;
        if (s <= t) {
            device const float* q = phq + ROW(t) * p2;
            device const float* k = phk + ROW(s) * p2;
            device const float* g = pw + (t - s) * p2;
            for (uint f = 0; f < p2; ++f) acc += q[f] * k[f] * g[f];
        }
        A[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // stage 2: dkv, thread per d
    if (tid < a.dv) {
        uint d = tid;
        float Gn[HK_P2];
        device const float* gn = dstates + (((ulong)bh * (nchunks + 1u) + c + 1u) * p2) * a.dv + d;
        for (uint f = 0; f < p2; ++f) Gn[f] = gn[(ulong)f * a.dv];
        for (uint s = 0; s < HK_C; ++s) {
            float acc = 0.0f;
            for (uint t = s; t < HK_C; ++t) {
                acc += A[t * HK_C + s] * dout[ROW(t) * a.dv + d];
            }
            device const float* k = phk + ROW(s) * p2;
            device const float* g = pw + (HK_C - 1u - s) * p2;
            for (uint f = 0; f < p2; ++f) acc += k[f] * g[f] * Gn[f];
            dkv[ROW(s) * a.dv + d] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // stage 3: dA (overwrites A)
    for (uint idx = tid; idx < HK_C * HK_C; idx += nth) {
        uint t = idx / HK_C, s = idx % HK_C;
        float acc = 0.0f;
        if (s <= t) {
            device const float* dq = dout + ROW(t) * a.dv;
            device const float* kvs = kv + ROW(s) * a.dv;
            for (uint d = 0; d < a.dv; ++d) acc += dq[d] * kvs[d];
        }
        A[idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // stage 4: dφq, dφk — thread per (t, f)
    for (uint idx = tid; idx < HK_C * p2; idx += nth) {
        uint t = idx / p2, f = idx % p2;
        // dφq_t[f]
        float acc = 0.0f;
        for (uint s = 0; s <= t; ++s) {
            acc += A[t * HK_C + s] * phk[ROW(s) * p2 + f] * pw[(t - s) * p2 + f];
        }
        {
            device const float* dq = dout + ROW(t) * a.dv;
            device const float* sc = states + (((ulong)bh * (nchunks + 1u) + c) * p2 + f) * a.dv;
            float dot = 0.0f;
            for (uint d = 0; d < a.dv; ++d) dot += dq[d] * sc[d];
            acc += pw[(t + 1u) * p2 + f] * dot;
        }
        dphq[ROW(t) * p2 + f] = acc;
        // dφk_s[f] with s = t (same index space)
        uint s = t;
        float acck = 0.0f;
        for (uint tt = s; tt < HK_C; ++tt) {
            acck += A[tt * HK_C + s] * phq[ROW(tt) * p2 + f] * pw[(tt - s) * p2 + f];
        }
        {
            device const float* kvs = kv + ROW(s) * a.dv;
            device const float* gn = dstates + (((ulong)bh * (nchunks + 1u) + c + 1u) * p2 + f) * a.dv;
            float dot = 0.0f;
            for (uint d = 0; d < a.dv; ++d) dot += kvs[d] * gn[d];
            acck += pw[(HK_C - 1u - s) * p2 + f] * dot;
        }
        dphk[ROW(s) * p2 + f] = acck;
    }
    #undef ROW
}

// ---------------------------------------------------------------------
// Anchor attention companions (the softmax layer: GQA, RoPE, causal).
// The matmuls are the generic GEMM with column-block offsets; only the
// row-wise softmax and RoPE are custom.
// ---------------------------------------------------------------------

// RoPE (neox halves: pair (i, i+hd/2)) in place on x [rows, nheads·hd];
// position = row % T. sign=+1 forward, −1 = inverse rotation (backward).
struct RopeArgs { uint T, nheads, hd; float base; float sign; };
kernel void rope_f32(
    device float*      x [[buffer(0)]],
    constant RopeArgs& a [[buffer(1)]],
    uint gid [[thread_position_in_grid]])   // over rows·nheads·(hd/2)
{
    uint half_hd = a.hd / 2u;
    uint per_row = a.nheads * half_hd;
    uint row = gid / per_row, r = gid % per_row;
    uint h = r / half_hd, i = r % half_hd;
    uint pos = row % a.T;
    float inv_freq = pow(a.base, -(float)(2u * i) / (float)a.hd);
    float ang = (float)pos * inv_freq;
    float c = cos(ang), s = sin(ang) * a.sign;
    device float* p = x + (ulong)row * a.nheads * a.hd + h * a.hd;
    float x0 = p[i], x1 = p[i + half_hd];
    p[i] = x0 * c - x1 * s;
    p[i + half_hd] = x0 * s + x1 * c;
}

// Causal softmax over the rows of an [T,T] score block, in place:
// P[t,j] = softmax_j(S[t,j]) for j ≤ t, 0 beyond. One threadgroup per
// row (256 threads). `n` = T (row length); grid over T rows.
kernel void causal_softmax_rows_f32(
    device float*   S [[buffer(0)]],
    constant uint&  n [[buffer(1)]],
    uint2 tgp [[threadgroup_position_in_grid]],   // x: row, y: [T,T] block
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    uint row = tgp.x;
    device float* r = S + (ulong)tgp.y * n * n + (ulong)row * n;
    uint len = row + 1u;
    float mx = -INFINITY;
    for (uint j = tid; j < len; j += 256u) mx = max(mx, r[j]);
    mx = simd_max(mx);
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint s = 1; s < 8u; ++s) mx = max(mx, red[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint j = tid; j < len; j += 256u) { float e = exp(r[j] - mx); r[j] = e; sum += e; }
    sum = simd_sum(sum);
    if (lane == 0) red[sgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; ++s) tot += red[s];
    float inv = 1.0f / tot;
    for (uint j = tid; j < len; j += 256u) r[j] *= inv;
    for (uint j = len + tid; j < n; j += 256u) r[j] = 0.0f;
}

// Softmax backward on rows: dS = P ⊙ (dP − Σ_j P·dP), in place on dP.
kernel void softmax_bwd_rows_f32(
    device const float* P  [[buffer(0)]],
    device float*       dP [[buffer(1)]],
    constant uint&      n  [[buffer(2)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    uint row = tgp.x;
    device const float* p = P + (ulong)tgp.y * n * n + (ulong)row * n;
    device float* d = dP + (ulong)tgp.y * n * n + (ulong)row * n;
    float dot = 0.0f;
    for (uint j = tid; j < n; j += 256u) dot += p[j] * d[j];
    dot = simd_sum(dot);
    if (lane == 0) red[sgid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; ++s) tot += red[s];
    for (uint j = tid; j < n; j += 256u) d[j] = p[j] * (d[j] - tot);
}

// σ(x + bias) forward (y) and backward (dx = dy·y·(1−y)) — the κ gate.
kernel void sigmoid_fwd_f32(
    device const float* x    [[buffer(0)]],
    device float*       y    [[buffer(1)]],
    constant float&     bias [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    y[gid] = 1.0f / (1.0f + exp(-(x[gid] + bias)));
}
kernel void sigmoid_bwd_f32(
    device const float* y  [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float*       dx [[buffer(2)]],
    constant uint&      n  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float s = y[gid];
    dx[gid] = dy[gid] * s * (1.0f - s);
}

// dE[tok[row], :] += dx[row, :]  (atomic float adds; tied head)
kernel void embed_scatter_add_f32(
    device atomic_float* dE  [[buffer(0)]],
    device const uint*   tok [[buffer(1)]],
    device const float*  dx  [[buffer(2)]],
    constant uint&       d   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])   // x: column, y: row
{
    if (gid.x >= d) return;
    atomic_fetch_add_explicit(&dE[(ulong)tok[gid.y] * d + gid.x], dx[(ulong)gid.y * d + gid.x], memory_order_relaxed);
}

// Copy `n` floats: dst = src (buffer-to-buffer with offsets on the host side).
kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) dst[gid] = src[gid];
}

// κ gate with a padded pre-activation (the projection GEMM writes 64
// columns; only nh are real): kap[row·nh + h] = σ(pre[row·ld + h] + bias)
struct KapArgs { uint rows, nh, ld; float bias; };
kernel void kappa_fwd_f32(
    device const float* pre [[buffer(0)]],
    device float*       kap [[buffer(1)]],
    constant KapArgs&   a   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])   // over rows·nh
{
    if (gid >= a.rows * a.nh) return;
    uint row = gid / a.nh, h = gid % a.nh;
    kap[gid] = 1.0f / (1.0f + exp(-(pre[(ulong)row * a.ld + h] + a.bias)));
}
// dpre[row·ld + j] = j < nh ? dkap[row·nh + j]·k(1−k) : 0
kernel void kappa_bwd_f32(
    device const float* kap  [[buffer(0)]],
    device const float* dkap [[buffer(1)]],
    device float*       dpre [[buffer(2)]],
    constant KapArgs&   a    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])   // over rows·ld
{
    if (gid >= a.rows * a.ld) return;
    uint row = gid / a.ld, j = gid % a.ld;
    float g = 0.0f;
    if (j < a.nh) {
        float k = kap[(ulong)row * a.nh + j];
        g = dkap[(ulong)row * a.nh + j] * k * (1.0f - k);
    }
    dpre[gid] = g;
}

// hybrid_k chunk scan, GEMM formulation (log-space decay trick):
//   Q̃[t][f] = φq·γ^t,  K̃[s][f] = φk·γ^{−s},  Q⁺[t][f] = φq·γ^{t+1},  K̂[s][f] = φk·γ^{C−1−s}
// (t, s relative to the chunk; f32 range is ample for horizons ≥ 1 at C = 64).
// Chunk-major layout [B, nh, nchunks, 64, P2]; source phq/phk are the row-major
// [B·T, nh·P2] tables. One thread per (row, h, f).
kernel void hk_scale_f32(
    device const float* phq   [[buffer(0)]],
    device const float* phk   [[buffer(1)]],
    device const float* pow_t [[buffer(2)]],
    device float*       qt    [[buffer(3)]],
    device float*       kt    [[buffer(4)]],
    device float*       qp    [[buffer(5)]],
    device float*       kh    [[buffer(6)]],
    constant HkArgs&    a     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])   // over B·T·nh·P2
{
    uint p2 = 2u * a.nph;
    uint rows = a.B * a.T;
    if (gid >= rows * a.nh * p2) return;
    uint row = gid / (a.nh * p2), r = gid % (a.nh * p2);
    uint h = r / p2, f = r % p2;
    uint b = row / a.T, tt = row % a.T;
    uint c = tt / HK_C, t = tt % HK_C;
    uint nch = a.T / HK_C;
    ulong dst = (((ulong)(b * a.nh + h) * nch + c) * HK_C + t) * p2 + f;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;
    float q = phq[gid], k = phk[gid];
    float g_t = pw[t * p2 + f];
    qt[dst] = q * g_t;
    kt[dst] = k / g_t;
    qp[dst] = q * pw[(t + 1u) * p2 + f];
    kh[dst] = k * pw[(HK_C - 1u - t) * p2 + f];
}

// Inverse: dφq = dQ̃·γ^t + dqi·γ^{t+1};  dφk = dK̃·γ^{−s} + dki·γ^{C−1−s}
// (chunk-major inputs → row-major dphq/dphk).
kernel void hk_unscale_f32(
    device const float* dqt   [[buffer(0)]],
    device const float* dkt   [[buffer(1)]],
    device const float* dqi   [[buffer(2)]],
    device const float* dki   [[buffer(3)]],
    device const float* pow_t [[buffer(4)]],
    device float*       dphq  [[buffer(5)]],
    device float*       dphk  [[buffer(6)]],
    constant HkArgs&    a     [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint p2 = 2u * a.nph;
    uint rows = a.B * a.T;
    if (gid >= rows * a.nh * p2) return;
    uint row = gid / (a.nh * p2), r = gid % (a.nh * p2);
    uint h = r / p2, f = r % p2;
    uint b = row / a.T, tt = row % a.T;
    uint c = tt / HK_C, t = tt % HK_C;
    uint nch = a.T / HK_C;
    ulong src = (((ulong)(b * a.nh + h) * nch + c) * HK_C + t) * p2 + f;
    device const float* pw = pow_t + (ulong)h * (HK_C + 1u) * p2;
    float g_t = pw[t * p2 + f];
    dphq[gid] = dqt[src] * g_t + dqi[src] * pw[(t + 1u) * p2 + f];
    dphk[gid] = dkt[src] / g_t + dki[src] * pw[(HK_C - 1u - t) * p2 + f];
}

// Chunk-state scans, cell-parallel: the recurrence is independent per
// (f, d) cell, sequential only in t — one thread per cell, threadgroup =
// 8 features × 32 value channels (φk broadcast across d, kv coalesced
// across d). Grid: (B·nh, ceil(P2/8), ceil(dv/32)).
#define HK_FB 8u
#define HK_DB 32u

kernel void hk_states_fwd_par_f32(
    device const float* phk    [[buffer(0)]],
    device const float* kv     [[buffer(1)]],
    device const float* pow_t  [[buffer(2)]],
    device float*       states [[buffer(3)]],
    constant HkArgs&    a      [[buffer(4)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])   // x: d (32), y: f (8)
{
    uint b = tg.x / a.nh, h = tg.x % a.nh;
    uint p2 = 2u * a.nph;
    uint f = tg.y * HK_FB + tid.y;
    uint d = tg.z * HK_DB + tid.x;
    if (f >= p2 || d >= a.dv) return;
    uint nchunks = a.T / HK_C;
    float gam = pow_t[((ulong)h * (HK_C + 1u) + 1u) * p2 + f];
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    device const float* pk = phk + ((ulong)b * a.T * a.nh + h) * p2 + f;
    device const float* kvp = kv + ((ulong)b * a.T * a.nh + h) * a.dv + d;
    ulong step_k = (ulong)a.nh * p2, step_v = (ulong)a.nh * a.dv;
    float S = 0.0f;
    states[st_base + (ulong)f * a.dv + d] = 0.0f;
    for (uint c = 0; c < nchunks; ++c) {
        // 16 positions a batch: all loads issued before the dependent FMAs
        for (uint s0 = 0; s0 < HK_C; s0 += 16u) {
            float kk[16], vv[16];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                ulong t = (ulong)c * HK_C + s0 + i;
                kk[i] = pk[t * step_k];
                vv[i] = kvp[t * step_v];
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) S = gam * S + kk[i] * vv[i];
        }
        states[st_base + ((ulong)(c + 1u) * p2 + f) * a.dv + d] = S;
    }
}

kernel void hk_dstates_bwd_par_f32(
    device const float* phq     [[buffer(0)]],
    device const float* dout    [[buffer(1)]],
    device const float* pow_t   [[buffer(2)]],
    device float*       dstates [[buffer(3)]],
    constant HkArgs&    a       [[buffer(4)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    uint b = tg.x / a.nh, h = tg.x % a.nh;
    uint p2 = 2u * a.nph;
    uint f = tg.y * HK_FB + tid.y;
    uint d = tg.z * HK_DB + tid.x;
    if (f >= p2 || d >= a.dv) return;
    uint nchunks = a.T / HK_C;
    float gam = pow_t[((ulong)h * (HK_C + 1u) + 1u) * p2 + f];
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    device const float* pq = phq + ((ulong)b * a.T * a.nh + h) * p2 + f;
    device const float* dop = dout + ((ulong)b * a.T * a.nh + h) * a.dv + d;
    ulong step_q = (ulong)a.nh * p2, step_o = (ulong)a.nh * a.dv;
    float G = 0.0f;
    dstates[st_base + ((ulong)nchunks * p2 + f) * a.dv + d] = 0.0f;
    for (uint cc = 0; cc < nchunks; ++cc) {
        uint c = nchunks - 1u - cc;
        for (uint s0 = 0; s0 < HK_C; s0 += 16u) {
            float qq[16], oo[16];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                ulong t = (ulong)c * HK_C + (HK_C - 1u - (s0 + i));
                qq[i] = pq[t * step_q];
                oo[i] = dop[t * step_o];
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) G = gam * (G + qq[i] * oo[i]);
        }
        dstates[st_base + ((ulong)c * p2 + f) * a.dv + d] = G;
    }
}

// ---------------------------------------------------------------------
// Hierarchical head companions (128 clusters × 256, tied to the embedding):
// rows are grouped by target cluster on the host; these gather/scatter the
// grouped rows and run the within-cluster CE with an index map.
// ---------------------------------------------------------------------

// dst[i,:] = idx[i] ≥ 0 ? src[idx[i],:] : 0
kernel void gather_rows_f32(
    device const float* src [[buffer(0)]],
    device const int*   idx [[buffer(1)]],
    device float*       dst [[buffer(2)]],
    constant uint&      d   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= d) return;
    int i = idx[gid.y];
    dst[(ulong)gid.y * d + gid.x] = (i >= 0) ? src[(ulong)i * d + gid.x] : 0.0f;
}

// dst[idx[i],:] += src[i,:]  for idx[i] ≥ 0 (indices unique: no atomics)
kernel void scatter_add_rows_f32(
    device float*       dst [[buffer(0)]],
    device const int*   idx [[buffer(1)]],
    device const float* src [[buffer(2)]],
    constant uint&      d   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= d) return;
    int i = idx[gid.y];
    if (i >= 0) dst[(ulong)i * d + gid.x] += src[(ulong)gid.y * d + gid.x];
}

// Within-cluster softmax-CE over rows of n logits with an index map:
// row r stands for token idx[r] (< 0: padding → dlogits row = 0, no loss);
// target = tgt[idx[r]] mod n; loss2[idx[r]] = −log p; logits ← (p−onehot)·scale.
kernel void softmax_ce_idx_f32(
    device float*       logits [[buffer(0)]],
    device const int*   idx    [[buffer(1)]],
    device const uint*  tgt    [[buffer(2)]],
    device float*       loss2  [[buffer(3)]],
    constant uint&      n      [[buffer(4)]],
    constant float&     scale  [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    device float* lr = logits + (ulong)row * n;
    int i = idx[row];
    if (i < 0) {
        for (uint j = tid; j < n; j += 256u) lr[j] = 0.0f;
        return;
    }
    uint t = tgt[i] % n;
    float mx = -INFINITY;
    for (uint j = tid; j < n; j += 256u) mx = max(mx, lr[j]);
    mx = simd_max(mx);
    if (lane == 0) red[sgid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = red[0];
    for (uint s = 1; s < 8u; ++s) mx = max(mx, red[s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint j = tid; j < n; j += 256u) sum += exp(lr[j] - mx);
    sum = simd_sum(sum);
    if (lane == 0) red[sgid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; ++s) tot += red[s];
    float lse = mx + log(tot);
    if (tid == 0) loss2[i] = lse - lr[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < n; j += 256u) {
        float p = exp(lr[j] - lse);
        lr[j] = (p - ((j == t) ? 1.0f : 0.0f)) * scale;
    }
}

// GQA group reduction: src is head-major [B][qh][T][hd] (per-q-head partial
// dK or dV), dst is row-major [B·T, kvh·hd]; dst[b,t][g·hd + d] = Σ_j src[b][g·group + j][t][d].
struct GroupSumArgs { uint B, T, qh, kvh, hd; };
kernel void group_sum_heads_f32(
    device const float*    src [[buffer(0)]],
    device float*          dst [[buffer(1)]],
    constant GroupSumArgs& a   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])   // over B·T·kvh·hd
{
    uint kd = a.kvh * a.hd;
    uint rows = a.B * a.T;
    if (gid >= rows * kd) return;
    uint row = gid / kd, r = gid % kd;
    uint g = r / a.hd, d = r % a.hd;
    uint b = row / a.T, t = row % a.T;
    uint group = a.qh / a.kvh;
    float s = 0.0f;
    for (uint j = 0; j < group; ++j) {
        uint i = g * group + j;
        s += src[(((ulong)b * a.qh + i) * a.T + t) * a.hd + d];
    }
    dst[gid] = s;
}

// ---------------------------------------------------------------------
// Routed experts (top-1 by RESONANCE, no gate — P1): expert e has a
// descriptor μ_e (+ later a k-dim principal subspace U_e); resonance =
// reconstruction error ‖(x−μ_e) − U_eᵀU_e(x−μ_e)‖²; route to argmin of
// (resonance − bias_e), bias_e = loss-free load balancing.
// Tokens are placed in per-expert slots deterministically (rank among the
// tokens of that expert); slots ≥ cap are dropped (shared expert only).
// ---------------------------------------------------------------------

struct RouteArgs { uint rows, H, E, k, cap; };

// assign[row] = argmin_e (‖x−μ_e‖² − ‖U_e(x−μ_e)‖² − bias_e); one threadgroup
// (E ≤ 64 threads... use 64 threads: thread e computes expert e's score)
kernel void route_f32(
    device const float* x      [[buffer(0)]],   // [rows, H]
    device const float* mu     [[buffer(1)]],   // [E, H]
    device const float* U      [[buffer(2)]],   // [E, k, H] (rows orthonormal; k may be 0)
    device const float* bias   [[buffer(3)]],   // [E]
    device uint*        assign [[buffer(4)]],   // [rows]
    device float*       res    [[buffer(5)]],   // [rows] resonance of the chosen expert
    constant RouteArgs& a      [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint e   [[thread_index_in_threadgroup]])
{
    threadgroup float sc[64];
    threadgroup float rs[64];
    float score = INFINITY;
    float r = 0.0f;
    if (e < a.E) {
        device const float* xr = x + (ulong)row * a.H;
        device const float* m = mu + (ulong)e * a.H;
        float d2 = 0.0f;
        for (uint j = 0; j < a.H; ++j) { float d = xr[j] - m[j]; d2 += d * d; }
        float proj = 0.0f;
        for (uint i = 0; i < a.k; ++i) {
            device const float* u = U + ((ulong)e * a.k + i) * a.H;
            float p = 0.0f;
            for (uint j = 0; j < a.H; ++j) p += (xr[j] - m[j]) * u[j];
            proj += p * p;
        }
        r = d2 - proj;
        score = r - bias[e];
    }
    sc[e] = score;
    rs[e] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        uint best = 0;
        float bs = sc[0];
        for (uint i = 1; i < a.E; ++i) { if (sc[i] < bs) { bs = sc[i]; best = i; } }
        assign[row] = best;
        res[row] = rs[best];
    }
}

// Deterministic slot assignment: slot[row] = rank of row among rows with
// the same expert (serial pass, one thread); count[e] = tokens per expert.
kernel void route_group_f32(
    device const uint* assign [[buffer(0)]],
    device uint*       slot   [[buffer(1)]],
    device uint*       count  [[buffer(2)]],   // [E]
    constant RouteArgs& a     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    uint c[64];
    for (uint e = 0; e < a.E; ++e) c[e] = 0;
    for (uint r = 0; r < a.rows; ++r) {
        uint e = assign[r];
        slot[r] = c[e];
        c[e] += 1u;
    }
    for (uint e = 0; e < a.E; ++e) count[e] = c[e];
}

// hg[e][slot][:] = x[row][:] for slot < cap (buffer pre-zeroed)
kernel void moe_gather_f32(
    device const float* x      [[buffer(0)]],
    device const uint*  assign [[buffer(1)]],
    device const uint*  slot   [[buffer(2)]],
    device float*       hg     [[buffer(3)]],   // [E, cap, H]
    constant RouteArgs& a      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])   // x: column, y: row
{
    if (gid.x >= a.H) return;
    uint s = slot[gid.y];
    if (s >= a.cap) return;
    uint e = assign[gid.y];
    hg[((ulong)e * a.cap + s) * a.H + gid.x] = x[(ulong)gid.y * a.H + gid.x];
}

// out[row][:] += yh[e][slot][:] for slot < cap
kernel void moe_scatter_add_f32(
    device float*       out    [[buffer(0)]],
    device const uint*  assign [[buffer(1)]],
    device const uint*  slot   [[buffer(2)]],
    device const float* yh     [[buffer(3)]],
    constant RouteArgs& a      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= a.H) return;
    uint s = slot[gid.y];
    if (s >= a.cap) return;
    uint e = assign[gid.y];
    out[(ulong)gid.y * a.H + gid.x] += yh[((ulong)e * a.cap + s) * a.H + gid.x];
}

// Descriptor statistics: sums[e][j] = Σ_{slots of e} hg[e][slot][j] (over
// the filled slots — zero rows contribute nothing); one thread per (e, j).
kernel void moe_stats_f32(
    device const float* hg    [[buffer(0)]],
    device const uint*  count [[buffer(1)]],
    device float*       sums  [[buffer(2)]],   // [E, H]
    constant RouteArgs& a     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])   // over E·H
{
    if (gid >= a.E * a.H) return;
    uint e = gid / a.H, j = gid % a.H;
    uint n = min(count[e], a.cap);
    float s = 0.0f;
    for (uint r = 0; r < n; ++r) s += hg[((ulong)e * a.cap + r) * a.H + j];
    sums[gid] = s;
}

// Descriptor update after a step: μ_e ← (1−α)·μ_e + α·sums_e/count_e for
// experts that received tokens; bias_e += η·(1/E − count_e/rows) (loss-free
// balancing toward equal load). One thread per (e, j); thread j == 0 also
// moves the bias.
struct MoeUpdArgs { uint rows, H, E; float alpha, eta; };
kernel void moe_update_f32(
    device float*       mu    [[buffer(0)]],
    device float*       bias  [[buffer(1)]],
    device const float* sums  [[buffer(2)]],
    device const uint*  count [[buffer(3)]],
    constant MoeUpdArgs& a    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= a.E * a.H) return;
    uint e = gid / a.H, j = gid % a.H;
    uint n = count[e];
    if (n > 0) {
        float mean = sums[gid] / (float)n;
        mu[gid] = (1.0f - a.alpha) * mu[gid] + a.alpha * mean;
    }
    if (j == 0) {
        float frac = (float)n / (float)a.rows;
        bias[e] += a.eta * (1.0f / (float)a.E - frac);
    }
}

// Indirect dispatch arguments for the per-expert GEMMs: args[e] =
// {ntiles_n, ceil(min(count_e, cap)/64), 1} for two column counts (n1, n2).
struct IndArgs { uint E, cap, n1, n2; };
kernel void moe_indirect_args_f32(
    device const uint* count [[buffer(0)]],
    device uint*       args  [[buffer(1)]],   // [2, E, 3]
    constant IndArgs&  a     [[buffer(2)]],
    uint e [[thread_position_in_grid]])
{
    if (e >= a.E) return;
    uint m = min(count[e], a.cap);
    uint mt = (m + 63u) / 64u;
    args[(0u * a.E + e) * 3u + 0u] = a.n1 / 64u;
    args[(0u * a.E + e) * 3u + 1u] = mt;
    args[(0u * a.E + e) * 3u + 2u] = 1u;
    args[(1u * a.E + e) * 3u + 0u] = a.n2 / 64u;
    args[(1u * a.E + e) * 3u + 1u] = mt;
    args[(1u * a.E + e) * 3u + 2u] = 1u;
}
