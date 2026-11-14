// Video sparse attention (VSA) for the MiniMax-H3 chain: FastVideo's VSA-H3
// as its FastH3 students were trained with it. Compiled after h3_flash.cu
// (its mma, copy and swizzle helpers, its K / V lane layouts) with -DHD.
//
// The rows live in TILE-PADDED order: tile t owns slots 64 t .. 64 t + 63,
// of which the first size[t] hold rows (`h3_qkv_split` writes row r to its
// slot) and the rest stay zero. Per head:
//   pool     the tile means of q, k (and v) in f32           PQ, PK, PV [nh][T][HD]
//   scores   tile against tile, q̄ · k̄ · scale              S [nh][T][T]
//   select   a video query tile's key tiles: the P non-video tiles, then
//            its top-k video tiles by score, ascending       L [nh][T][P + k]
//   flash    each query tile through the rows of its key tiles (a
//            non-video query tile through all of them)       AOP [np][nh·HD] f16
//   compress softmax(S) · PV, the gated branch's tile output OC [nh][T][HD]
//   gather   back to row order, + gate · OC of the row's tile  AO [rows][nh·HD] f16

#define VT 64
#define CPT (VT / BN)

__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(f) : "h"(h));
    return f;
}

// the element of V [planes][ns/16][8 gid][4 tig][HD/8][4] holding (key, dim d)
__device__ __forceinline__ size_t vsa_vpos(int plane, int ns, int key, int d) {
    const int j = d >> 3, gid = d & 7;
    return ((size_t)plane * (ns >> 4) * 8 + gid) * (size_t)(HD >> 1) * 4
         + ((((size_t)(key >> 4) * 8) * 4 + ((key & 7) >> 1)) * (HD >> 1) + (size_t)j * 4 + ((key & 15) >> 3) * 2 + (key & 1));
}

// a block per (tile, head), a thread per dim
extern "C" __global__ void h3_vsa_pool(const u16* __restrict__ Q, const u16* __restrict__ K, const u16* __restrict__ V,
                                       const int* __restrict__ size, float* __restrict__ PQ, float* __restrict__ PK,
                                       float* __restrict__ PV, int np, int ns, int T, int wantv)
{
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (d >= HD) return;
    const int sz = size[t];
    const int m = d >> 3, tig = (d & 7) >> 1, e = d & 1;
    const size_t kd = (size_t)tig * (HD >> 2) + 2 * m + e;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (int i = 0; i < sz; i++) {
        const int s = t * VT + i;
        const size_t row = ((size_t)h * np + s) * HD;
        aq += h2f(Q[row + d]);
        ak += h2f(K[row + kd]);
        if (wantv) av += h2f(V[vsa_vpos(h, ns, s, d)]);
    }
    const float inv = 1.0f / (float)sz;
    const size_t o = ((size_t)h * T + t) * HD + d;
    PQ[o] = aq * inv;
    PK[o] = ak * inv;
    if (wantv) PV[o] = av * inv;
}

// 16 x 16 tiles of S a block (256 threads), the dims through shared memory
extern "C" __global__ void h3_vsa_scores(const float* __restrict__ PQ, const float* __restrict__ PK,
                                         float* __restrict__ S, int T, float scale)
{
    __shared__ float a[16][HD + 1], b[16][HD + 1];
    const int i0 = blockIdx.x * 16, j0 = blockIdx.y * 16, h = blockIdx.z;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    for (int c = tid; c < 16 * HD; c += 256) {
        const int r = c / HD, d = c - r * HD;
        a[r][d] = i0 + r < T ? PQ[((size_t)h * T + i0 + r) * HD + d] : 0.0f;
        b[r][d] = j0 + r < T ? PK[((size_t)h * T + j0 + r) * HD + d] : 0.0f;
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll 8
    for (int d = 0; d < HD; d++) acc += a[ty][d] * b[tx][d];
    const int i = i0 + ty, j = j0 + tx;
    if (i < T && j < T) S[((size_t)h * T + i) * T + j] = acc * scale;
}

__device__ __forceinline__ unsigned int ord_key(float f) {
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// A block (256 threads) per video query tile and head: radix-select the
// k-th largest of the row's video scores, then write the non-video tiles and
// the chosen video tiles in ascending order (a block scan, no atomics on
// the list: the same list every run).
extern "C" __global__ void h3_vsa_select(const float* __restrict__ S, int* __restrict__ L, int T, int P, int k)
{
    __shared__ unsigned int hist[256];
    __shared__ unsigned int sh_prefix, sh_left;
    __shared__ int scan[256];
    const int i = P + blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* row = S + ((size_t)h * T + i) * T;
    const int nv = T - P;
    int* out = L + ((size_t)h * T + i) * (P + k);
    for (int c = tid; c < P; c += 256) out[c] = c;
    unsigned int prefix = 0, mask = 0, left = (unsigned int)k;
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        hist[tid] = 0;
        __syncthreads();
        for (int c = tid; c < nv; c += 256) {
            const unsigned int key = ord_key(row[P + c]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned int cum = 0;
            int digit = 0;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= left) {
                    digit = b;
                    break;
                }
                cum += hist[b];
            }
            sh_prefix = prefix | ((unsigned int)digit << shift);
            sh_left = left - cum;
        }
        __syncthreads();
        prefix = sh_prefix;
        left = sh_left;
        mask |= 255u << shift;
        __syncthreads();
    }
    // `prefix` is the k-th largest key; `left` of its equals are taken
    const int gt_total = k - (int)left;
    const int per = (nv + 255) / 256, c0 = tid * per, c1 = min(nv, c0 + per);
    for (int which = 0; which < 2; which++) {
        int cnt = 0;
        for (int c = c0; c < c1; c++) {
            const unsigned int key = ord_key(row[P + c]);
            cnt += which == 0 ? key > prefix : key == prefix;
        }
        scan[tid] = cnt;
        __syncthreads();
        for (int o = 1; o < 256; o <<= 1) {
            const int v = tid >= o ? scan[tid - o] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        int pos = scan[tid] - cnt;
        for (int c = c0; c < c1; c++) {
            const unsigned int key = ord_key(row[P + c]);
            if (which == 0 && key > prefix) {
                out[P + pos++] = P + c;
            } else if (which == 1 && key == prefix) {
                if (pos < (int)left) out[P + gt_total + pos] = P + c;
                pos++;
            }
        }
        __syncthreads();
    }
}

// `h3_flash2` over a query tile's key tiles: a block per (query tile, head),
// its four warps the tile's 64 slots; the key chunks of BN rows walk the
// tile list, and a key past its tile's size is masked.
extern "C" __global__ void __launch_bounds__(128) h3_vsa_flash(const u16* __restrict__ Q, const u16* __restrict__ K,
                                                               const u16* __restrict__ V, u16* __restrict__ AO,
                                                               const int* __restrict__ L, const int* __restrict__ size,
                                                               int np, int ns, int nh, int T, int P, int k, float scale)
{
    __shared__ __align__(16) uint4 sK[2][KCH];
    __shared__ __align__(16) uint4 sV[2][KCH];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int qt = blockIdx.x, h = blockIdx.y;
    const int b0 = qt * VT;
    const int q0 = b0 + warp * 16;
    const u16* Kg = K + (size_t)h * np * HD;
    const u16* Vg = V + (size_t)h * ns * HD;
    const int nsub = ns >> 4;
    const int rA = q0 + gid, rB = q0 + gid + 8;
    const u16* Qp = Q + (size_t)h * np * HD;
    const u16* qA = Qp + (size_t)rA * HD + 2 * tig;
    const u16* qB = Qp + (size_t)rB * HD + 2 * tig;
    u32 qa[NS][4];
#pragma unroll
    for (int s = 0; s < NS; s++) {
        qa[s][0] = *(const u32*)(qA + 16 * s);
        qa[s][1] = *(const u32*)(qB + 16 * s);
        qa[s][2] = *(const u32*)(qA + 16 * s + 8);
        qa[s][3] = *(const u32*)(qB + 16 * s + 8);
    }
    float o[ND][4];
#pragma unroll
    for (int j = 0; j < ND; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float mA = -INF, mB = -INF, lA = 0.0f, lB = 0.0f;
    const float sl = scale * LOG2E;
    const bool dense = qt < P;
    const int* list = L + ((size_t)h * T + qt) * (P + k);
    const int ntiles = dense ? T : P + k;
    auto tile_of = [&](int c) { return dense ? c / CPT : list[c / CPT]; };
    auto load = [&](int st, int k0) {
#pragma unroll
        for (int c = tid; c < KCH; c += 128) {
            const int row = c / (HD / 8), cc = c - row * (HD / 8);
            cp16(&sK[st][swz(c)], Kg + (size_t)min(k0 + row, np - 1) * HD + cc * 8);
            const int sub = c / (2 * HD), w = c - sub * (2 * HD);
            cp16(&sV[st][swz(c)], Vg + (size_t)min((k0 >> 4) + sub, nsub - 1) * (16 * HD) + w * 8);
        }
        asm volatile("cp.async.commit_group;");
    };
    const int nt = ntiles * CPT;
    {
        const int kt = tile_of(0);
        load(0, kt * VT);
    }
    for (int t = 0; t < nt; t++) {
        if (t + 1 < nt) {
            const int kn = tile_of(t + 1);
            load((t + 1) & 1, kn * VT + ((t + 1) % CPT) * BN);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const int kt = tile_of(t);
        const int kbase = kt * VT + (t % CPT) * BN, kend = kt * VT + size[kt];
        const uint4* tk = sK[t & 1];
        const uint4* tv = sV[t & 1];
#pragma unroll
        for (int u = 0; u < BN / 16; u++) {
            const int k0 = kbase + u * 16;
            if (k0 >= kend) break;
            const int ra = (16 * u + gid) * (HD / 8) + tig * (HD / 32), rb = ra + 8 * (HD / 8);
            float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int s = 0; s < NS; s += 2) {
                const uint4 a = tk[swz(ra + (s >> 1))], b = tk[swz(rb + (s >> 1))];
                mma_f16(c0, qa[s][0], qa[s][1], qa[s][2], qa[s][3], a.x, a.y);
                mma_f16(c1, qa[s][0], qa[s][1], qa[s][2], qa[s][3], b.x, b.y);
                mma_f16(c0, qa[s + 1][0], qa[s + 1][1], qa[s + 1][2], qa[s + 1][3], a.z, a.w);
                mma_f16(c1, qa[s + 1][0], qa[s + 1][1], qa[s + 1][2], qa[s + 1][3], b.z, b.w);
            }
            const int kc = k0 + 2 * tig;
            float sA[4] = {c0[0], c0[1], c1[0], c1[1]};
            float sB[4] = {c0[2], c0[3], c1[2], c1[3]};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int key = kc + (i & 1) + (i >> 1) * 8;
                sA[i] = key < kend ? sA[i] * sl : -INF;
                sB[i] = key < kend ? sB[i] * sl : -INF;
            }
            float tA = fmaxf(fmaxf(sA[0], sA[1]), fmaxf(sA[2], sA[3]));
            float tB = fmaxf(fmaxf(sB[0], sB[1]), fmaxf(sB[2], sB[3]));
            tA = fmaxf(tA, __shfl_xor_sync(0xffffffffu, tA, 1));
            tB = fmaxf(tB, __shfl_xor_sync(0xffffffffu, tB, 1));
            tA = fmaxf(tA, __shfl_xor_sync(0xffffffffu, tA, 2));
            tB = fmaxf(tB, __shfl_xor_sync(0xffffffffu, tB, 2));
            const float nA = fmaxf(mA, tA), nB = fmaxf(mB, tB);
            const float fA = ex2(mA - nA), fB = ex2(mB - nB);
            mA = nA;
            mB = nB;
            float eA[4], eB[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                eA[i] = ex2(sA[i] - nA);
                eB[i] = ex2(sB[i] - nB);
            }
            lA = lA * fA + (eA[0] + eA[1]) + (eA[2] + eA[3]);
            lB = lB * fB + (eB[0] + eB[1]) + (eB[2] + eB[3]);
            if (fA != 1.0f || fB != 1.0f) {
#pragma unroll
                for (int j = 0; j < ND; j++) {
                    o[j][0] *= fA;
                    o[j][1] *= fA;
                    o[j][2] *= fB;
                    o[j][3] *= fB;
                }
            }
            const u32 p0 = pack2h(eA[0] * PS, eA[1] * PS), p1 = pack2h(eB[0] * PS, eB[1] * PS);
            const u32 p2 = pack2h(eA[2] * PS, eA[3] * PS), p3 = pack2h(eB[2] * PS, eB[3] * PS);
            const int vb = u * (2 * HD) + lane * (HD / 16);
#pragma unroll
            for (int j = 0; j < ND; j += 2) {
                const uint4 v = tv[swz(vb + (j >> 1))];
                mma_f16(o[j], p0, p1, p2, p3, v.x, v.y);
                mma_f16(o[j + 1], p0, p1, p2, p3, v.z, v.w);
            }
        }
        __syncthreads();
    }
    lA += __shfl_xor_sync(0xffffffffu, lA, 1);
    lB += __shfl_xor_sync(0xffffffffu, lB, 1);
    lA += __shfl_xor_sync(0xffffffffu, lA, 2);
    lB += __shfl_xor_sync(0xffffffffu, lB, 2);
    const float iA = 1.0f / (lA * PS), iB = 1.0f / (lB * PS);
    const size_t inner = (size_t)nh * HD;
    const int qend = b0 + size[qt];
    u16* oA = AO + (size_t)rA * inner + (size_t)h * HD + 2 * tig;
    u16* oB = AO + (size_t)rB * inner + (size_t)h * HD + 2 * tig;
#pragma unroll
    for (int j = 0; j < ND; j++) {
        if (rA < qend) *(u32*)(oA + 8 * j) = pack2h_sat(o[j][0] * iA, o[j][1] * iA);
        if (rB < qend) *(u32*)(oB + 8 * j) = pack2h_sat(o[j][2] * iB, o[j][3] * iB);
    }
}

// the compression branch: a block per (tile, head), a thread per dim
extern "C" __global__ void h3_vsa_compress(const float* __restrict__ S, const float* __restrict__ PV,
                                           float* __restrict__ OC, int T)
{
    __shared__ float red[32];
    __shared__ float sh_m, sh_z;
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x, lane = d & 31, warp = d >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    const float* row = S + ((size_t)h * T + i) * T;
    float m = -INF;
    for (int j = d; j < T; j += blockDim.x) m = fmaxf(m, row[j]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (d == 0) {
        float v = -INF;
        for (int w = 0; w < nw; w++) v = fmaxf(v, red[w]);
        sh_m = v;
    }
    __syncthreads();
    m = sh_m;
    float z = 0.0f;
    for (int j = d; j < T; j += blockDim.x) z += expf(row[j] - m);
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    __syncthreads();
    if (lane == 0) red[warp] = z;
    __syncthreads();
    if (d == 0) {
        float v = 0.0f;
        for (int w = 0; w < nw; w++) v += red[w];
        sh_z = v;
    }
    __syncthreads();
    if (d >= HD) return;
    const float inv = 1.0f / sh_z;
    const float* pv = PV + (size_t)h * T * HD + d;
    float acc = 0.0f;
    for (int j = 0; j < T; j++) acc += expf(row[j] - m) * pv[(size_t)j * HD];
    OC[((size_t)h * T + i) * HD + d] = acc * inv;
}

// rows r0 .. r0 + rows back from their slots, + gate · OC of their tile
extern "C" __global__ void h3_vsa_gather(const u16* __restrict__ AOP, const float* __restrict__ G,
                                         const float* __restrict__ OC, const int* __restrict__ slot,
                                         u16* __restrict__ AO, int r0, int rows, int nh, int T)
{
    const size_t inner = (size_t)nh * HD;
    const size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t)rows * inner) return;
    const int rr = (int)(e / inner);
    const int c = (int)(e - (size_t)rr * inner);
    const int s = slot[r0 + rr];
    float v = h2f(AOP[(size_t)s * inner + c]);
    if (G) {
        const int h = c / HD, d = c - h * HD, t = s / VT;
        v += G[e] * OC[((size_t)h * T + t) * HD + d];
    }
    AO[e] = pack2h_sat(v, 0.0f) & 0xffffu;
}
