// Device linear algebra of the NVG `blr` encoder on CUDA (`gpu_cuda/nvg_dev.rs`):
// the elementwise / reduction kernels, the output-side and input-side sign
// knots and the Householder steps of `gpu_wgpu/nvg_dev.rs`, f32, over
// row-major matrices. The dense products, the Cholesky panels and the
// triangular solves are cuBLAS calls on the host side. A kernel over a matrix
// takes its row from the grid's y and its column from x (no 64-bit division
// per element): at most 65535 rows a launch.

typedef unsigned int u32;

#ifndef NQ
#define NQ 1
#endif

// Σ v over the block (blockDim a multiple of 32); every thread gets it.
__device__ __forceinline__ float nd_block_sum(float v, float* red) {
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

__device__ __forceinline__ float nd_sgn(float v) { return v >= 0.f ? 1.f : -1.f; }

// C[oc + i·ldc + j] = alpha·f(X[i,j])·R[i]·S[j] (+ beta·C), X = A (or C when in place).
// flags: 1 row scale, 2 column scale, 4 accumulate (beta), 8 lower mask (j > i → 0),
// 16 sign, 32 abs, 64 in place.
extern "C" __global__ void nd_scale_rc(const float* __restrict__ A, const float* __restrict__ R, const float* __restrict__ S,
                                       float* C, u32 rows, u32 cols, u32 lda, u32 oa, u32 ldc, u32 oc, u32 flags,
                                       float alpha, float beta)
{
    const u32 i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    const size_t ix = (size_t)oc + (size_t)i * ldc + j;
    float v = (flags & 64u) ? C[ix] : A[(size_t)oa + (size_t)i * lda + j];
    if (flags & 16u) v = nd_sgn(v);
    if (flags & 32u) v = fabsf(v);
    v = v * alpha;
    if (flags & 1u) v = v * R[i];
    if (flags & 2u) v = v * S[j];
    if ((flags & 8u) && j > i) v = 0.f;
    if (flags & 4u) v = v + beta * C[ix];
    C[ix] = v;
}

// O[i] = Σ_j A[oa + i·lda + j]·B[ob + i·ldb + j]; a block a row.
extern "C" __global__ void nd_rowdot(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ O,
                                     u32 cols, u32 lda, u32 oa, u32 ldb, u32 ob)
{
    __shared__ float red[32];
    const size_t i = blockIdx.x;
    const float* a = A + oa + i * lda;
    const float* b = B + ob + i * ldb;
    float acc = 0.f;
    for (u32 j = threadIdx.x; j < cols; j += blockDim.x) acc += a[j] * b[j];
    const float s = nd_block_sum(acc, red);
    if (threadIdx.x == 0) O[i] = s;
}

// In place: C[oc + i·ldc + j] /= max(‖row i‖, tiny); a block a row.
extern "C" __global__ void nd_rownorm(float* C, u32 cols, u32 ldc, u32 oc)
{
    __shared__ float red[32];
    float* c = C + oc + (size_t)blockIdx.x * ldc;
    float acc = 0.f;
    for (u32 j = threadIdx.x; j < cols; j += blockDim.x) acc += c[j] * c[j];
    const float s = nd_block_sum(acc, red);
    const float inv = 1.f / fmaxf(sqrtf(s), 1.0e-30f);
    for (u32 j = threadIdx.x; j < cols; j += blockDim.x) c[j] *= inv;
}

// Column partial sums: O[ch·cols + j] = Σ over the rows of chunk ch of A·B.
// Grid (cols / 256, chunks), block 256.
extern "C" __global__ void nd_colsum_part(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ O,
                                          u32 rows, u32 cols, u32 lda, u32 oa, u32 ldb, u32 ob, u32 per)
{
    const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    const u32 ch = blockIdx.y, r0 = ch * per, r1 = min(r0 + per, rows);
    float acc = 0.f;
    for (u32 i = r0; i < r1; i++) acc += A[(size_t)oa + (size_t)i * lda + j] * B[(size_t)ob + (size_t)i * ldb + j];
    O[(size_t)ch * cols + j] = acc;
}

// C[oc + j] = Σ_ch A[ch·cols + j]
extern "C" __global__ void nd_colsum_red(const float* __restrict__ A, float* __restrict__ C, u32 nch, u32 cols, u32 oc)
{
    const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float acc = 0.f;
    for (u32 ch = 0; ch < nch; ch++) acc += A[(size_t)ch * cols + j];
    C[oc + j] = acc;
}

// Frobenius partials: O[block] = Σ over the block's stride of A[i]·B[i], i < len; nwg blocks of 256.
extern "C" __global__ void nd_frob_part(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ O,
                                        size_t len, u32 nwg)
{
    __shared__ float red[32];
    float acc = 0.f;
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < len; i += (size_t)256 * nwg) acc += A[i] * B[i];
    const float s = nd_block_sum(acc, red);
    if (threadIdx.x == 0) O[blockIdx.x] = s;
}

// O[i] = A[oa + i·lda + i]
extern "C" __global__ void nd_diag(const float* __restrict__ A, float* __restrict__ O, u32 n, u32 lda, u32 oa)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) O[i] = A[(size_t)oa + (size_t)i * lda + i];
}

// C[i,j] = (A[a,b] + (a == b)·ridge) / (R[a]·R[b]); (a, b) = (i, j), or with
// `rev` the reversed act order a = U[n-1-i], b = U[n-1-j].
extern "C" __global__ void nd_equil(const float* __restrict__ A, const float* __restrict__ R, const u32* __restrict__ U,
                                    float* __restrict__ C, u32 n, u32 rev, float ridge)
{
    const u32 i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const size_t t = (size_t)i * n + j;
    u32 a = i, b = j;
    if (rev) {
        a = U[n - 1 - i];
        b = U[n - 1 - j];
    }
    float v = A[(size_t)a * n + b];
    if (a == b) v += ridge;
    C[t] = v / (R[a] * R[b]);
}

// C[j][k] = A[(n-1-j)·n + (n-1-k)] / R[k] for k >= j, else 0.
extern "C" __global__ void nd_make_uc(const float* __restrict__ A, const float* __restrict__ R, float* __restrict__ C, u32 n)
{
    const u32 j = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const size_t t = (size_t)j * n + k;
    C[t] = k >= j ? A[(size_t)(n - 1 - j) * n + (n - 1 - k)] / R[k] : 0.f;
}

// Row gather (scatter = 0): C[j,:] = A[U[j],:]; scatter: C[U[j],:] = A[j,:].
extern "C" __global__ void nd_permute_rows(const float* __restrict__ A, const u32* __restrict__ U, float* __restrict__ C,
                                           u32 rows, u32 cols, u32 scatter)
{
    const u32 j = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const size_t t = (size_t)j * cols + c;
    if (scatter) C[(size_t)U[j] * cols + c] = A[t];
    else C[t] = A[(size_t)U[j] * cols + c];
}

// C (cols × rows) = A (rows × cols) transposed.
extern "C" __global__ void nd_transpose(const float* __restrict__ A, float* __restrict__ C, u32 rows, u32 cols)
{
    const u32 c = blockIdx.y, r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    C[(size_t)c * rows + r] = A[(size_t)r * cols + c];
}

// C[oc + i·ldc + j] = (T[i·n + j]·alpha)·E[oc + i·ldc + j] + beta·C[…]: the
// Hadamard epilogue of a product staged in T (m × n, dense).
extern "C" __global__ void nd_had(const float* __restrict__ T, const float* __restrict__ E, float* C, u32 m, u32 n,
                                  u32 ldc, u32 oc, float beta, u32 acc)
{
    const u32 i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const size_t t = (size_t)i * n + j;
    const size_t ix = (size_t)oc + (size_t)i * ldc + j;
    float v = T[t] * E[ix];
    if (acc) v += beta * C[ix];
    C[ix] = v;
}

// X = I (n × n)
extern "C" __global__ void nd_eye(float* __restrict__ X, u32 n)
{
    const u32 i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    X[(size_t)i * n + j] = i == j ? 1.f : 0.f;
}

// One block: mean diagonal of the n × n A; RG[0] = ridge = rel·mean (retry 0)
// or RG[0]·100 (a failed factorisation), RG[1] = mean; C[i] = √max(A_ii + ridge, tiny).
extern "C" __global__ void nd_diag_ridge(const float* __restrict__ A, float* __restrict__ C, float* RG, u32 n, u32 retry,
                                         float rel)
{
    __shared__ float red[32];
    float acc = 0.f;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) acc += A[(size_t)i * n + i];
    const float mean = nd_block_sum(acc, red) / (float)n;
    float ridge = rel * fmaxf(mean, 1.0e-30f);
    if (retry) ridge = RG[0] * 100.f;
    __syncthreads();
    if (threadIdx.x == 0) {
        RG[0] = ridge;
        RG[1] = mean;
    }
    for (u32 i = threadIdx.x; i < n; i += blockDim.x) C[i] = sqrtf(fmaxf(A[(size_t)i * n + i] + ridge, 1.0e-30f));
}

// F = (A + RG[0]·I) / (d dᵀ), d = R.
extern "C" __global__ void nd_equil_rg(const float* __restrict__ A, const float* __restrict__ R, const float* __restrict__ RG,
                                       float* __restrict__ C, u32 n)
{
    const u32 i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const size_t t = (size_t)i * n + j;
    float v = A[t];
    if (i == j) v += RG[0];
    C[t] = v / (R[i] * R[j]);
}

// C[oc + i] = A[oa + i] / R[i]
extern "C" __global__ void nd_div_vec(const float* __restrict__ A, const float* __restrict__ R, float* __restrict__ C,
                                      u32 len, u32 oa, u32 oc)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) C[oc + i] = A[oa + i] / R[i];
}

// Per-row h solve (np ∈ {1, 2}) as the host ALS: old values stay when singular.
extern "C" __global__ void nd_hsolve(const float* __restrict__ rh0, const float* __restrict__ rh1,
                                     const float* __restrict__ g00, const float* __restrict__ g01,
                                     const float* __restrict__ g11, float* __restrict__ h0, float* __restrict__ h1, u32 m,
                                     u32 np)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    if (np == 1) {
        const float ridge = 1.0e-10f * fmaxf(g00[i], 1.0e-30f);
        h0[i] = rh0[i] / (g00[i] + ridge);
        return;
    }
    const float tr = 0.5f * (g00[i] + g11[i]);
    const float ridge = 1.0e-10f * fmaxf(tr, 1.0e-30f);
    const float a11 = g00[i] + ridge, a12 = g01[i], a22 = g11[i] + ridge;
    const float det = a11 * a22 - a12 * a12;
    if (fabsf(det) > 0.f) {
        h0[i] = (a22 * rh0[i] - a12 * rh1[i]) / det;
        h1[i] = (a11 * rh1[i] - a12 * rh0[i]) / det;
    }
}

// C[U[j]] = A[so + j], j < n
extern "C" __global__ void nd_scatter_vec(const float* __restrict__ A, const u32* __restrict__ U, float* __restrict__ C,
                                          u32 n, u32 so)
{
    const u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) C[U[j]] = A[so + j];
}

// One block: RG[slot] = Σ of the `len` partials in A.
extern "C" __global__ void nd_sum_into(const float* __restrict__ A, float* RG, u32 len, u32 slot)
{
    __shared__ float red[32];
    float acc = 0.f;
    for (u32 i = threadIdx.x; i < len; i += blockDim.x) acc += A[i];
    const float s = nd_block_sum(acc, red);
    if (threadIdx.x == 0) RG[slot] = s;
}

// C[i] = A[i] / √RG[slot]
extern "C" __global__ void nd_scale_rsqrt(const float* __restrict__ A, const float* __restrict__ RG, float* __restrict__ C,
                                          size_t len, u32 slot)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) C[i] = A[i] / sqrtf(fmaxf(RG[slot], 1.0e-30f));
}

// ───────────────────────────── discrete knots ─────────────────────────────

// Blocked output-side flips: the kk axis in blocks of 32. One thread per row
// decides the block's 32 bits in order (the block's qs entries in registers,
// updated through the block's 32×32 corner of Q in shared memory) and records
// the flips F (m × 32, entries 2·s_old or 0); the rest of every row's qs then
// moves with one GEMM `qs −= F·Q[block, :]` on the host side.
// su, qs, whb row-major m × r; Q r × r; h (m), l (r). Block 64, grid m / 64.
extern "C" __global__ void nd_flipblk(const float* __restrict__ whb, const float* __restrict__ Q,
                                      const float* __restrict__ H, const float* __restrict__ L, float* __restrict__ su,
                                      const float* __restrict__ qs, float* __restrict__ F, u32* CNT, u32 m, u32 r, u32 b0,
                                      float thr)
{
    __shared__ float qb[1024];
    __shared__ float lb[32];
    const u32 li = threadIdx.x, bs = min(32u, r - b0);
    for (u32 e = li; e < 1024u; e += blockDim.x) {
        const u32 a = e / 32u, b = e % 32u;
        qb[e] = (a < bs && b < bs) ? Q[(size_t)(b0 + a) * r + b0 + b] : 0.f;
    }
    if (li < 32u) lb[li] = li < bs ? L[b0 + li] : 0.f;
    __syncthreads();
    const size_t i = (size_t)blockIdx.x * blockDim.x + li;
    if (i >= m) return;
    const float h = H[i];
    const size_t base = i * r + b0;
    float s[32], q[32], w[32], f[32];
#pragma unroll
    for (u32 c = 0; c < 32u; c++) {
        s[c] = 0.f;
        q[c] = 0.f;
        w[c] = 0.f;
        f[c] = 0.f;
        if (c < bs) {
            s[c] = su[base + c];
            q[c] = qs[base + c];
            w[c] = whb[base + c] * lb[c];
        }
    }
    u32 nf = 0;
#pragma unroll
    for (u32 c = 0; c < 32u; c++) {
        if (c < bs) {
            const float qd = qb[c * 33u];
            const float d = -4.f * h * h * s[c] * (q[c] - qd * s[c]) + 4.f * h * s[c] * w[c];
            if (d < -thr * (h * h * qd + 1.0e-30f)) {
                f[c] = 2.f * s[c];
                s[c] = -s[c];
                nf++;
#pragma unroll
                for (u32 c2 = c + 1; c2 < 32u; c2++) q[c2] = q[c2] - f[c] * qb[c * 32u + c2];
            }
        }
    }
#pragma unroll
    for (u32 c = 0; c < 32u; c++) {
        if (c < bs) su[base + c] = s[c];
        F[i * 32u + c] = f[c];
    }
    if (nf > 0) atomicAdd(CNT, nf);
}

// Output-side knots, one thread per output row i, column-major operands:
// sut = S_Uᵀ (r × m, ±1), qst = qsᵀ (r × m), whbt (r × m), Q (r × r), h, l.
// Greedy passes (≤ 4) of exact single-bit flips.
extern "C" __global__ void nd_flipu(const float* __restrict__ whbt, const float* __restrict__ Q,
                                    const float* __restrict__ H, const float* __restrict__ L, float* __restrict__ sut,
                                    float* __restrict__ qst, u32* CNT, u32 m, u32 r, float thr)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const float h = H[i];
    u32 nf = 0;
    for (u32 sweep = 0; sweep < 4u; sweep++) {
        u32 changed = 0;
        for (u32 kk = 0; kk < r; kk++) {
            const float s = sut[(size_t)kk * m + i];
            const float qd = Q[(size_t)kk * r + kk];
            const float bm = whbt[(size_t)kk * m + i] * L[kk];
            const float d = -4.f * h * h * s * (qst[(size_t)kk * m + i] - qd * s) + 4.f * h * s * bm;
            if (d < -thr * (h * h * qd + 1.0e-30f)) {
                for (u32 t = 0; t < r; t++) qst[(size_t)t * m + i] -= 2.f * s * Q[(size_t)kk * r + t];
                sut[(size_t)kk * m + i] = -s;
                changed++;
            }
        }
        nf += changed;
        if (changed == 0) break;
    }
    if (nf > 0) atomicAdd(CNT, nf);
}

// Input-side knots of one 128-column block [b0, b1) in act order, in the
// latent space (n columns, rank r): BV (n × r: Mᵀw_j, trailing feedback
// applied), S (n × r, ±1), KS0 (n × r: K s_j of the incoming signs), K (r × r),
// Uc (n × n), EG = [E (128 × r) | G (n)]. Column j: bv_j minus the in-block
// feedback, best-improvement flips (≤ 3 rounds of ≤ maxf), e_j = (bv_j − g_j K s_j) / Uc[j][j].
// One block; a thread holds its NQ = ceil(r / blockDim) lanes in registers.
struct NdBest {
    float v;
    u32 i;
    float s;
};

__device__ __forceinline__ void nd_sum2(float& x, float& y, float* pa, float* pb, float* fin)
{
    for (int o = 16; o > 0; o >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, o);
        y += __shfl_xor_sync(0xffffffffu, y, o);
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) {
        pa[w] = x;
        pb[w] = y;
    }
    __syncthreads();
    if (w == 0) {
        float a = l < nw ? pa[l] : 0.f, b = l < nw ? pb[l] : 0.f;
        for (int o = 16; o > 0; o >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, o);
            b += __shfl_xor_sync(0xffffffffu, b, o);
        }
        if (l == 0) {
            fin[0] = a;
            fin[1] = b;
        }
    }
    __syncthreads();
    x = fin[0];
    y = fin[1];
}

__device__ __forceinline__ void nd_best_merge(float& v, u32& i, float& s, float ov, u32 oi, float os)
{
    if (ov < v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
        s = os;
    }
}

extern "C" __global__ void nd_gv_block(const float* __restrict__ KS0, const float* __restrict__ K,
                                       const float* __restrict__ Uc, const float* __restrict__ BV, float* __restrict__ S,
                                       float* EG, u32* CNT, u32 n, u32 r, u32 b0, u32 b1, u32 maxf, float tol, float gjtol)
{
    __shared__ float pa[32], pb[32], fin[2];
    __shared__ float pv[32], ps[32], fv, fs;
    __shared__ u32 pi[32], fi;
    const u32 t = threadIdx.x, nt = blockDim.x, lane = t & 31, w = t >> 5, nw = nt >> 5;
    const size_t goff = (size_t)128 * r;
    float kd[NQ], s[NQ], ks[NQ], bv[NQ];
#pragma unroll
    for (u32 q = 0; q < NQ; q++) {
        const u32 x = t + nt * q;
        kd[q] = x < r ? K[(size_t)x * r + x] : 0.f;
    }
    u32 flips = 0;
    for (u32 j = b0; j < b1; j++) {
#pragma unroll
        for (u32 q = 0; q < NQ; q++) {
            const u32 x = t + nt * q;
            s[q] = 0.f;
            ks[q] = 0.f;
            bv[q] = 0.f;
            if (x < r) {
                s[q] = S[(size_t)j * r + x];
                ks[q] = KS0[(size_t)j * r + x];
                bv[q] = BV[(size_t)j * r + x];
            }
        }
        for (u32 jj = b0; jj < j; jj++) {
            const float u = Uc[(size_t)jj * n + j];
            const size_t eb = (size_t)(jj - b0) * r;
#pragma unroll
            for (u32 q = 0; q < NQ; q++) {
                const u32 x = t + nt * q;
                if (x < r) bv[q] -= u * EG[eb + x];
            }
        }
        float a1 = 0.f, a2 = 0.f;
#pragma unroll
        for (u32 q = 0; q < NQ; q++) {
            a1 += bv[q] * s[q];
            a2 += s[q] * ks[q];
        }
        nd_sum2(a1, a2, pa, pb, fin);
        float gj = a1 / fmaxf(a2, 1.0e-30f);
        if (gj < 0.f) {
#pragma unroll
            for (u32 q = 0; q < NQ; q++) {
                s[q] = -s[q];
                ks[q] = -ks[q];
            }
            gj = -gj;
        }
        for (u32 rd = 0; rd < 3u; rd++) {
            u32 nf = 0;
            while (nf < maxf) {
                float bvl = 3.4e38f, bsl = 0.f;
                u32 bil = 0xffffffffu;
#pragma unroll
                for (u32 q = 0; q < NQ; q++) {
                    const u32 x = t + nt * q;
                    if (x < r) {
                        const float d = -4.f * gj * gj * s[q] * (ks[q] - kd[q] * s[q]) + 4.f * gj * s[q] * bv[q];
                        if (d < bvl) {
                            bvl = d;
                            bil = x;
                            bsl = s[q];
                        }
                    }
                }
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, bvl, o);
                    const u32 oi = __shfl_xor_sync(0xffffffffu, bil, o);
                    const float os = __shfl_xor_sync(0xffffffffu, bsl, o);
                    nd_best_merge(bvl, bil, bsl, ov, oi, os);
                }
                if (lane == 0) {
                    pv[w] = bvl;
                    pi[w] = bil;
                    ps[w] = bsl;
                }
                __syncthreads();
                if (w == 0) {
                    float gv = lane < nw ? pv[lane] : 3.4e38f, gs = lane < nw ? ps[lane] : 0.f;
                    u32 gi = lane < nw ? pi[lane] : 0xffffffffu;
                    for (int o = 16; o > 0; o >>= 1) {
                        const float ov = __shfl_xor_sync(0xffffffffu, gv, o);
                        const u32 oi = __shfl_xor_sync(0xffffffffu, gi, o);
                        const float os = __shfl_xor_sync(0xffffffffu, gs, o);
                        nd_best_merge(gv, gi, gs, ov, oi, os);
                    }
                    if (lane == 0) {
                        fv = gv;
                        fi = gi;
                        fs = gs;
                    }
                }
                __syncthreads();
                const float bvm = fv, sb = fs;
                const u32 bi = fi;
                if (bi >= r) break;
                const size_t kr = (size_t)bi * r;
                const float kdb = K[kr + bi];
                if (bvm >= -tol * (gj * gj * kdb + 1.0e-30f)) break;
#pragma unroll
                for (u32 q = 0; q < NQ; q++) {
                    const u32 x = t + nt * q;
                    if (x == bi) s[q] = -s[q];
                    if (x < r) ks[q] -= 2.f * sb * K[kr + x];
                }
                nf++;
                // the shared winner is rewritten in the next round: every thread has read it
                __syncthreads();
            }
            flips += nf;
            float c1 = 0.f, c2 = 0.f;
#pragma unroll
            for (u32 q = 0; q < NQ; q++) {
                c1 += bv[q] * s[q];
                c2 += s[q] * ks[q];
            }
            nd_sum2(c1, c2, pa, pb, fin);
            float g2 = c1 / fmaxf(c2, 1.0e-30f);
            if (g2 < 0.f) {
#pragma unroll
                for (u32 q = 0; q < NQ; q++) {
                    s[q] = -s[q];
                    ks[q] = -ks[q];
                }
                g2 = -g2;
            }
            const bool done = nf == 0 || fabsf(g2 - gj) <= gjtol * fabsf(gj);
            gj = g2;
            if (done) break;
        }
        const float ujj = Uc[(size_t)j * n + j];
#pragma unroll
        for (u32 q = 0; q < NQ; q++) {
            const u32 x = t + nt * q;
            if (x < r) {
                EG[(size_t)(j - b0) * r + x] = (bv[q] - gj * ks[q]) / ujj;
                S[(size_t)j * r + x] = s[q];
            }
        }
        if (t == 0) EG[goff + j] = gj;
        // the next column reads this one's E row
        __syncthreads();
    }
    if (t == 0) atomicAdd(CNT, flips);
}

// ───────────────────────────── symmetric eigensolver ─────────────────────────────

// Reflector of step i (LAPACK dlarfg on A[i+1.., i]): v (v0 = 1) into VS and
// below the subdiagonal of column i; d_i, beta (= e_i), tau into DE. One block.
extern "C" __global__ void nd_hh_vec(float* A, float* __restrict__ VS, float* __restrict__ DE, u32 k, u32 i)
{
    __shared__ float red[32];
    const u32 t = threadIdx.x, L = k - i - 1;
    const float x0 = A[(size_t)(i + 1) * k + i];
    float ss = 0.f;
    for (u32 j = 1 + t; j < L; j += blockDim.x) {
        const float v = A[(size_t)(i + 1 + j) * k + i];
        ss += v * v;
    }
    const float xn2 = nd_block_sum(ss, red);
    float tau = 0.f, beta = x0, scale = 0.f;
    if (xn2 > 0.f) {
        const float nrm = sqrtf(x0 * x0 + xn2);
        beta = x0 >= 0.f ? -nrm : nrm;
        tau = (beta - x0) / beta;
        scale = 1.f / (x0 - beta);
    }
    for (u32 j = t; j < L; j += blockDim.x) {
        float v = 1.f;
        if (j > 0) {
            v = A[(size_t)(i + 1 + j) * k + i] * scale;
            A[(size_t)(i + 1 + j) * k + i] = v;
        }
        VS[j] = v;
    }
    if (t == 0) {
        DE[i] = A[(size_t)i * k + i];
        DE[k + i] = beta;
        DE[2 * k + i] = tau;
    }
}

// p = tau·A22·v, a block a row of the trailing block.
extern "C" __global__ void nd_hh_matvec(const float* __restrict__ A, const float* __restrict__ VS, float* __restrict__ PV,
                                        const float* __restrict__ DE, u32 k, u32 i)
{
    __shared__ float red[32];
    const u32 L = k - i - 1, j = blockIdx.x;
    const float* row = A + (size_t)(i + 1 + j) * k + i + 1;
    float acc = 0.f;
    for (u32 c = threadIdx.x; c < L; c += blockDim.x) acc += row[c] * VS[c];
    const float s = nd_block_sum(acc, red);
    if (threadIdx.x == 0) PV[j] = DE[2 * k + i] * s;
}

// w = p − (tau/2)(p·v) v. One block.
extern "C" __global__ void nd_hh_w(const float* __restrict__ PV, const float* __restrict__ VS, float* __restrict__ WV,
                                   const float* __restrict__ DE, u32 k, u32 i)
{
    __shared__ float red[32];
    const u32 L = k - i - 1;
    float acc = 0.f;
    for (u32 j = threadIdx.x; j < L; j += blockDim.x) acc += PV[j] * VS[j];
    const float kk = 0.5f * DE[2 * k + i] * nd_block_sum(acc, red);
    for (u32 j = threadIdx.x; j < L; j += blockDim.x) WV[j] = PV[j] - kk * VS[j];
}

// A22 −= v wᵀ + w vᵀ
extern "C" __global__ void nd_hh_rank2(float* __restrict__ A, const float* __restrict__ VS, const float* __restrict__ WV,
                                       u32 k, u32 i)
{
    const u32 L = k - i - 1;
    const u32 j = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= L) return;
    A[(size_t)(i + 1 + j) * k + i + 1 + c] -= VS[j] * WV[c] + WV[j] * VS[c];
}

extern "C" __global__ void nd_hh_last(const float* __restrict__ A, float* __restrict__ DE, u32 k)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) DE[k - 1] = A[(size_t)(k - 1) * k + k - 1];
}

// Block of b reflectors starting at i0 (below the subdiagonal of the k × k A)
// as a dense k × b matrix C: column c is reflector i = i0 + c (1 at row i+1,
// A[row][i] below, 0 above).
extern "C" __global__ void nd_vblock(const float* __restrict__ A, float* __restrict__ C, u32 k, u32 b, u32 i0)
{
    const u32 row = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, i = i0 + c;
    if (c >= b) return;
    const size_t e = (size_t)row * b + c;
    float v = 0.f;
    if (i + 1 < k) {
        if (row == i + 1) v = 1.f;
        else if (row > i + 1) v = A[(size_t)row * k + i];
    }
    C[e] = v;
}

// Compact-WY T of one reflector block (dlarft, forward, columnwise), one
// block of 64 threads: VtV (bb × bb), tau at TAU[to..], b live columns; T (bb × bb).
extern "C" __global__ void nd_wy_t(const float* __restrict__ VtV, const float* __restrict__ TAU, float* T, u32 bb, u32 b,
                                   u32 to)
{
    const u32 t = threadIdx.x;
    for (u32 e = t; e < bb * bb; e += blockDim.x) T[e] = 0.f;
    __syncthreads();
    for (u32 i = 0; i < b; i++) {
        const float ti = TAU[to + i];
        float v = 0.f;
        if (t < i && ti != 0.f) {
            for (u32 q = t; q < i; q++) v += T[t * bb + q] * VtV[q * bb + i];
            v = -ti * v;
        }
        __syncthreads();
        if (t < i) T[t * bb + i] = v;
        if (t == 0) T[i * bb + i] = ti;
        __syncthreads();
    }
}
