// Attention with nothing materialized: no score rows, no weight rows.
// Compiled per head width (-DHD=<multiple of 32>) and mask (-DCAUSAL=0: the
// full bidirectional attention of the MiniMax-H3 chain; 1: a language
// model's, a query attending the keys up to itself).
//
// A warp owns 16 query rows of one (group, head) plane and walks the keys 16
// at a time. A step is two tensor-core products and a softmax between them,
// all in registers: scores = Q · Kᵀ (m16n8k16, f16 operands, f32 sums), the
// running row maximum and row sum of the online softmax, then O = O · f +
// P · V with the weights as the left operand exactly as the lanes already
// hold them.
//
// What bounds it is reading the key and value tiles, not the products: with
// each operand a 4-byte load of its own the kernel spent nine tenths of its
// time in loads (measured by leaving parts out). So the split kernel writes
// K and V in the order the lanes read them, and a lane takes its operands of
// a whole tile as a few 16-byte loads of one contiguous run.
//
// Layouts (see `h3_qkv_split`, lanes = 1):
//   Q   f16 [planes][n][HD]
//   K   f16 [planes][n][4 tig][HD / 8][2]         dims 8 m + 2 tig + e
//   V   f16 [planes][ns / 16][8 gid][4 tig][HD / 8][4]
//                                    dim 8 j + gid; keys 2 tig + e, then
//                                    2 tig + 8 + e, of the tile (zeros past n)
//   AO  f16 [groups * n][nh * HD]    the output projection's operand
typedef unsigned int u32;
typedef unsigned short u16;

#define INF __int_as_float(0x7f800000)
#define NS (HD / 16)
#define ND (HD / 8)
// weights are stored times this on their way through a half
#define PS 1024.0f

__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(h) : "f"(f));
    return h;
}

__device__ __forceinline__ u32 pack2h(float a, float b) {
    return (u32)f2h(a) | ((u32)f2h(b) << 16);
}

__device__ __forceinline__ u32 pack2h_sat(float a, float b) {
    return pack2h(fminf(fmaxf(a, -65504.0f), 65504.0f), fminf(fmaxf(b, -65504.0f), 65504.0f));
}

// c += A · B: A 16x16 row-major, B 16x8 column-major, both f16; c f32.
__device__ __forceinline__ void mma_f16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A block of W warps takes 16 W query rows of plane blockIdx.y. `hpk` query
// planes share one key / value plane (grouped-query attention; 1 otherwise).
extern "C" __global__ void h3_flash(const u16* __restrict__ Q, const u16* __restrict__ K,
                                    const u16* __restrict__ V, u16* __restrict__ AO, int n, int ns, int nh,
                                    float scale, int hpk)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int plane = blockIdx.y, g = plane / nh, h = plane - g * nh;
    const int q0 = ((int)blockIdx.x * (int)(blockDim.x >> 5) + warp) * 16;
    if (q0 >= n) return;
    const int kplane = plane / hpk;
    const u16* Qp = Q + (size_t)plane * n * HD;
    const u16* Kp = K + (size_t)kplane * n * HD + tig * (HD / 4);
    const u16* Vp = V + (size_t)kplane * ns * HD + (size_t)(gid * 4 + tig) * (HD / 2);
    // rows past the end read the last row; their output is not written
    const int rA = q0 + gid, rB = q0 + gid + 8;
    const u16* qA = Qp + (size_t)min(rA, n - 1) * HD + 2 * tig;
    const u16* qB = Qp + (size_t)min(rB, n - 1) * HD + 2 * tig;
    // the query tile as the left operand of every score product
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
#if CAUSAL
    // the tile's last row sees the keys up to itself
    const int kend = min(n, q0 + 16);
#else
    const int kend = n;
#endif
    for (int k0 = 0; k0 < kend; k0 += 16) {
        // scores of the tile's 16 keys: fragment 0 keys k0 + gid, fragment 1
        // keys k0 + 8 + gid (a key past the end reads the last one and is
        // masked below); a 16-byte load is the operands of two steps
        const uint4* ka = (const uint4*)(Kp + (size_t)min(k0 + gid, n - 1) * HD);
        const uint4* kb = (const uint4*)(Kp + (size_t)min(k0 + 8 + gid, n - 1) * HD);
        float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < NS; s += 2) {
            const uint4 a = ka[s >> 1], b = kb[s >> 1];
            mma_f16(c0, qa[s][0], qa[s][1], qa[s][2], qa[s][3], a.x, a.y);
            mma_f16(c1, qa[s][0], qa[s][1], qa[s][2], qa[s][3], b.x, b.y);
            mma_f16(c0, qa[s + 1][0], qa[s + 1][1], qa[s + 1][2], qa[s + 1][3], a.z, a.w);
            mma_f16(c1, qa[s + 1][0], qa[s + 1][1], qa[s + 1][2], qa[s + 1][3], b.z, b.w);
        }
        // this lane's scores: rows gid and gid + 8 against keys kc, kc + 1,
        // kc + 8, kc + 9
        const int kc = k0 + 2 * tig;
        float sA[4] = {c0[0], c0[1], c1[0], c1[1]};
        float sB[4] = {c0[2], c0[3], c1[2], c1[3]};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int key = kc + (i & 1) + (i >> 1) * 8;
#if CAUSAL
            sA[i] = (key < n && key <= rA) ? sA[i] * scale : -INF;
            sB[i] = (key < n && key <= rB) ? sB[i] * scale : -INF;
#else
            if (key < n) {
                sA[i] *= scale;
                sB[i] *= scale;
            } else {
                sA[i] = -INF;
                sB[i] = -INF;
            }
#endif
        }
        // the tile's row maxima, across the four lanes that share a row
        float tA = fmaxf(fmaxf(sA[0], sA[1]), fmaxf(sA[2], sA[3]));
        float tB = fmaxf(fmaxf(sB[0], sB[1]), fmaxf(sB[2], sB[3]));
        tA = fmaxf(tA, __shfl_xor_sync(0xffffffffu, tA, 1));
        tB = fmaxf(tB, __shfl_xor_sync(0xffffffffu, tB, 1));
        tA = fmaxf(tA, __shfl_xor_sync(0xffffffffu, tA, 2));
        tB = fmaxf(tB, __shfl_xor_sync(0xffffffffu, tB, 2));
        const float nA = fmaxf(mA, tA), nB = fmaxf(mB, tB);
        // what the sums so far shrink by under the new maximum (0 at the
        // first tile, where the old one is -inf)
        const float fA = expf(mA - nA), fB = expf(mB - nB);
        mA = nA;
        mB = nB;
        float eA[4], eB[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            eA[i] = expf(sA[i] - nA);
            eB[i] = expf(sB[i] - nB);
        }
        lA = lA * fA + (eA[0] + eA[1]) + (eA[2] + eA[3]);
        lB = lB * fB + (eB[0] + eB[1]) + (eB[2] + eB[3]);
        // once a row's maximum has settled the factor is exactly one, and
        // for most tiles that is every row of the warp
        if (fA != 1.0f || fB != 1.0f) {
#pragma unroll
            for (int j = 0; j < ND; j++) {
                o[j][0] *= fA;
                o[j][1] *= fA;
                o[j][2] *= fB;
                o[j][3] *= fB;
            }
        }
        // the weights are the left operand as the lanes hold them
        const u32 p0 = pack2h(eA[0] * PS, eA[1] * PS), p1 = pack2h(eB[0] * PS, eB[1] * PS);
        const u32 p2 = pack2h(eA[2] * PS, eA[3] * PS), p3 = pack2h(eB[2] * PS, eB[3] * PS);
        const uint4* vp = (const uint4*)(Vp + (size_t)(k0 >> 4) * (16 * HD));
#pragma unroll
        for (int j = 0; j < ND; j += 2) {
            const uint4 v = vp[j >> 1];
            mma_f16(o[j], p0, p1, p2, p3, v.x, v.y);
            mma_f16(o[j + 1], p0, p1, p2, p3, v.z, v.w);
        }
    }
    lA += __shfl_xor_sync(0xffffffffu, lA, 1);
    lB += __shfl_xor_sync(0xffffffffu, lB, 1);
    lA += __shfl_xor_sync(0xffffffffu, lA, 2);
    lB += __shfl_xor_sync(0xffffffffu, lB, 2);
    const float iA = 1.0f / (lA * PS), iB = 1.0f / (lB * PS);
    const size_t inner = (size_t)nh * HD;
    u16* oA = AO + ((size_t)g * n + rA) * inner + (size_t)h * HD + 2 * tig;
    u16* oB = AO + ((size_t)g * n + rB) * inner + (size_t)h * HD + 2 * tig;
#pragma unroll
    for (int j = 0; j < ND; j++) {
        if (rA < n) *(u32*)(oA + 8 * j) = pack2h_sat(o[j][0] * iA, o[j][1] * iA);
        if (rB < n) *(u32*)(oB + 8 * j) = pack2h_sat(o[j][2] * iB, o[j][3] * iB);
    }
}
