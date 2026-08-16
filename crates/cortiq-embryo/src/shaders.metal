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

// Forward states: one threadgroup per (b,h), one thread per value channel
// d; the literal recurrence over all T positions, S at every chunk
// boundary written out. Threadgroup = dv threads.
kernel void hk_states_fwd_f32(
    device const float* phk    [[buffer(0)]],
    device const float* kv     [[buffer(1)]],
    device const float* pow_t  [[buffer(2)]],
    device float*       states [[buffer(3)]],
    constant HkArgs&    a      [[buffer(4)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint d   [[thread_index_in_threadgroup]])
{
    threadgroup float sk[HK_C * HK_P2];   // φk of the current chunk
    uint b = tg / a.nh, h = tg % a.nh;
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    // γ_f = pow[h][1][f]
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    float S[HK_P2];
    for (uint f = 0; f < HK_P2; ++f) S[f] = 0.0f;
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    for (uint c = 0; c < nchunks; ++c) {
        // S entering chunk c
        for (uint f = 0; f < p2; ++f) states[st_base + ((ulong)c * p2 + f) * a.dv + d] = S[f];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = d; i < HK_C * p2; i += a.dv) {
            uint s = i / p2, f = i % p2;
            sk[s * HK_P2 + f] = phk[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * p2 + f];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < HK_C; ++s) {
            float kvv = kv[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * a.dv + d];
            for (uint f = 0; f < p2; ++f) {
                S[f] = gam[f] * S[f] + sk[s * HK_P2 + f] * kvv;
            }
        }
    }
    for (uint f = 0; f < p2; ++f) states[st_base + ((ulong)nchunks * p2 + f) * a.dv + d] = S[f];
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

// Backward state gradients (reverse over positions):
//   G ← γ ⊙ (G + φq_t ⊗ do_t)   for t = T−1 … 0,
// G at each chunk boundary written to dstates[c] = ∂L/∂S_c restricted
// to reads by chunks ≥ c (dstates[nchunks] = 0). Threadgroup = dv threads.
kernel void hk_dstates_bwd_f32(
    device const float* phq     [[buffer(0)]],
    device const float* dout    [[buffer(1)]],
    device const float* pow_t   [[buffer(2)]],
    device float*       dstates [[buffer(3)]],
    constant HkArgs&    a       [[buffer(4)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint d   [[thread_index_in_threadgroup]])
{
    threadgroup float sq[HK_C * HK_P2];
    uint b = tg / a.nh, h = tg % a.nh;
    uint p2 = 2u * a.nph;
    uint nchunks = a.T / HK_C;
    device const float* gam = pow_t + ((ulong)h * (HK_C + 1u) + 1u) * p2;
    float G[HK_P2];
    for (uint f = 0; f < HK_P2; ++f) G[f] = 0.0f;
    ulong st_base = ((ulong)b * a.nh + h) * (nchunks + 1u) * p2 * a.dv;
    for (uint f = 0; f < p2; ++f) dstates[st_base + ((ulong)nchunks * p2 + f) * a.dv + d] = 0.0f;
    for (uint cc = 0; cc < nchunks; ++cc) {
        uint c = nchunks - 1u - cc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = d; i < HK_C * p2; i += a.dv) {
            uint s = i / p2, f = i % p2;
            sq[s * HK_P2 + f] = phq[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * p2 + f];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ss = 0; ss < HK_C; ++ss) {
            uint s = HK_C - 1u - ss;
            float dov = dout[((ulong)(b * a.T + c * HK_C + s) * a.nh + h) * a.dv + d];
            for (uint f = 0; f < p2; ++f) {
                G[f] = gam[f] * (G[f] + sq[s * HK_P2 + f] * dov);
            }
        }
        for (uint f = 0; f < p2; ++f) dstates[st_base + ((ulong)c * p2 + f) * a.dv + d] = G[f];
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
