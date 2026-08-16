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
    uint M, N, K;
    uint lda, ldb, ldc;
    float alpha, beta;
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
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
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

    for (uint k0 = 0; k0 < g.K; k0 += BK) {
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
