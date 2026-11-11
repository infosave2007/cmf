// The tensor-core small-batch q4tp matvec over the repacked ("mma") layout,
// standalone: the repack on the host, a check against the f32 scalar kernel
// of hyb.cu, timing on the shapes of Qwen3.8-27B.
//   nvcc -O3 -arch=sm_86 -DHPK=6 -DHD=256 -w mmp_test.cu -o mmp_test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "hyb.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

// Repacked layout of a q4tp matrix (rows a multiple of 16): for tile t (16
// rows) and group g (32 columns), 32 lanes x 8 bytes. Lane l = 4 gid + tig
// holds two words X0 (k 0..15 of the group) and X1 (k 16..31). Word X_s's
// nibble m (m < 4) and nibble m + 4 are the low and high element of the
// mma A register a_m of step s:
//   a0: row gid,     k = 16 s + 2 tig (+1)
//   a1: row gid + 8, k = 16 s + 2 tig (+1)
//   a2: row gid,     k = 16 s + 2 tig + 8 (+1)
//   a3: row gid + 8, k = 16 s + 2 tig + 8 (+1)
// so (X_s >> 4m) & 0x000F000F | 0x64006400 is a_m + (1032, 1032) in f16.
// Codes: [t][g][16 rows] bytes.
__device__ __forceinline__ u32 a_reg(u32 X, int m) {
    u32 h;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(h) : "r"(X >> (4 * m)), "r"(0x000F000Fu), "r"(0x64006400u));
    u32 r;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(h), "r"(0x64086408u));
    return r;
}

__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A block of 32 KS threads: TB tiles (16 TB rows), its KS warps over
// contiguous quarters (eighths) of the groups; every warp runs all the tiles
// of the block on its groups and the partials are summed in a fixed order.
// xh, xl [NT * 8][cols] f16, xsc [NT * 8] power-of-two scales; rows past
// `nrows` are not read.
template <int NT, int TB, int KS>
__device__ __forceinline__ void mmp_body(const HyJob* __restrict__ jobs, int njobs, const u16* __restrict__ xh,
                                         const u16* __restrict__ xl, const float* __restrict__ xsc, int nrows,
                                         int cols)
{
    __shared__ float lad_s[16 * TB][33];
    __shared__ float part[KS][TB][NT][32][4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int jx = 0;
    while (jx + 1 < njobs && (int)blockIdx.x >= jobs[jx + 1].blk0) jx++;
    const HyJob jb = jobs[jx];
    const int tile0 = ((int)blockIdx.x - jb.blk0) * TB;
    const int rows = jb.rows;
    const int ntile = rows >> 4;
    for (int i = tid; i < 16 * TB * 32; i += blockDim.x) {
        const int rr = tile0 * 16 + (i >> 5);
        lad_s[i >> 5][i & 31] = rr < rows ? ((const float*)jb.aux1)[(size_t)rr * 32 + (i & 31)] : 0.0f;
    }
    __syncthreads();
    const int gid = lane >> 2, tig = lane & 3;
    const int gpr = cols >> 5;
    const int per = (gpr + KS - 1) / KS;
    const int g0 = min(gpr, warp * per), g1 = min(gpr, g0 + per);
    float acc[TB][NT][4];
#pragma unroll
    for (int b = 0; b < TB; b++)
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[b][t][i] = 0.0f;
    const uint2* T = (const uint2*)jb.w;
    const u8* Cc = (const u8*)jb.aux0;
    const bool rd[NT] = {};
    (void)rd;
    for (int g = g0; g < g1; g++) {
        // this group's x columns for every 8-row n tile (zero past nrows)
        u32 bh[NT][4], bl[NT][4];
        const int k0 = g * 32 + 2 * tig;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const int j = t * 8 + gid;
            if (j < nrows) {
                const size_t xo = (size_t)j * cols + k0;
                bh[t][0] = *(const u32*)(xh + xo);
                bh[t][1] = *(const u32*)(xh + xo + 8);
                bh[t][2] = *(const u32*)(xh + xo + 16);
                bh[t][3] = *(const u32*)(xh + xo + 24);
                bl[t][0] = *(const u32*)(xl + xo);
                bl[t][1] = *(const u32*)(xl + xo + 8);
                bl[t][2] = *(const u32*)(xl + xo + 16);
                bl[t][3] = *(const u32*)(xl + xo + 24);
            } else {
#pragma unroll
                for (int i = 0; i < 4; i++) bh[t][i] = bl[t][i] = 0u;
            }
        }
#pragma unroll
        for (int b = 0; b < TB; b++) {
            const int tile = tile0 + b;
            if (tile >= ntile) break;
            const size_t tg = (size_t)tile * gpr + g;
            const uint2 X = T[tg * 32 + lane];
            const float sa = lad_s[b * 16 + gid][Cc[tg * 16 + gid]];
            const float sb = lad_s[b * 16 + gid + 8][Cc[tg * 16 + gid + 8]];
            const u32 a00 = a_reg(X.x, 0), a01 = a_reg(X.x, 1), a02 = a_reg(X.x, 2), a03 = a_reg(X.x, 3);
            const u32 a10 = a_reg(X.y, 0), a11 = a_reg(X.y, 1), a12 = a_reg(X.y, 2), a13 = a_reg(X.y, 3);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (t * 8 >= nrows) break;
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma16816(c, a00, a01, a02, a03, bh[t][0], bh[t][1]);
                mma16816(c, a00, a01, a02, a03, bl[t][0], bl[t][1]);
                mma16816(c, a10, a11, a12, a13, bh[t][2], bh[t][3]);
                mma16816(c, a10, a11, a12, a13, bl[t][2], bl[t][3]);
                acc[b][t][0] = fmaf(sa, c[0], acc[b][t][0]);
                acc[b][t][1] = fmaf(sa, c[1], acc[b][t][1]);
                acc[b][t][2] = fmaf(sb, c[2], acc[b][t][2]);
                acc[b][t][3] = fmaf(sb, c[3], acc[b][t][3]);
            }
        }
    }
#pragma unroll
    for (int b = 0; b < TB; b++)
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) part[warp][b][t][lane][i] = acc[b][t][i];
    __syncthreads();
    // the KS partials of every output, summed in order: a thread an output
    float* y = (float*)jb.y;
    const float* res = (const float*)jb.res;
    for (int o = tid; o < TB * NT * 32 * 4; o += blockDim.x) {
        const int i = o & 3, ln = (o >> 2) & 31, t = (o >> 7) % NT, b = (o >> 7) / NT;
        const int gd = ln >> 2, tg = ln & 3;
        const int j = t * 8 + 2 * tg + (i & 1);
        const int r = (tile0 + b) * 16 + gd + (i < 2 ? 0 : 8);
        if (j >= nrows || r >= rows) continue;
        float v = part[0][b][t][ln][i];
#pragma unroll
        for (int k = 1; k < KS; k++) v = __fadd_rn(v, part[k][b][t][ln][i]);
        v = __fmul_rn(v, xsc[j]);
        const size_t at = (size_t)j * jb.ystride + r;
        y[at] = res ? __fadd_rn(res[at], v) : v;
    }
}

#define MMP(NAME, NT, TB, KS)                                                                                     \
    extern "C" __global__ void __launch_bounds__(32 * KS) NAME(const HyJob* j, int nj, const u16* xh, const u16* xl, \
                                                              const float* xs, int n, int cols)                \
    {                                                                                                           \
        mmp_body<NT, TB, KS>(j, nj, xh, xl, xs, n, cols);                                                       \
    }
MMP(p1_t2_k8, 1, 2, 8)
MMP(p1_t1_k8, 1, 1, 8)
MMP(p1_t1_k16, 1, 1, 16)
MMP(p1_t2_k16, 1, 2, 16)
MMP(p1_t4_k8, 1, 4, 8)
MMP(p2_t1_k8, 2, 1, 8)
MMP(p2_t2_k8, 2, 2, 8)
MMP(p2_t1_k16, 2, 1, 16)
MMP(p2_t4_k8, 2, 4, 8)
MMP(p2_t2_k16, 2, 2, 16)

typedef void (*mmp_fn)(const HyJob*, int, const u16*, const u16*, const float*, int, int);

static unsigned short h_of(float f) {
    unsigned u; memcpy(&u, &f, 4);
    unsigned s = (u >> 16) & 0x8000u; int e = (int)((u >> 23) & 0xff) - 127 + 15; unsigned m = u & 0x7fffffu;
    if (e <= 0) return (unsigned short)s;
    unsigned r = (m >> 13); unsigned rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) { r++; if (r == 0x400) { r = 0; e++; } }
    return (unsigned short)(s | (e << 10) | r);
}
static float f_of(unsigned short h) {
    unsigned s = (h & 0x8000u) << 16; int e = (h >> 10) & 0x1f; unsigned m = h & 0x3ff;
    unsigned u = e == 0 ? s : (s | ((e - 15 + 127) << 23) | (m << 13)); float f; memcpy(&f, &u, 4); return f;
}

// nibble k of row r's group g in the original layout
static inline unsigned nib_at(const unsigned char* W, int gpr, int r, int g, int k) {
    const unsigned char* p = W + ((size_t)r * gpr + g) * 16;
    const unsigned w = p[(k >> 3) * 4 + ((k & 7) >> 1)];
    return (k & 1) ? (w >> 4) : (w & 15);
}

static void repack(const unsigned char* W, const unsigned char* C, int rows, int cols, std::vector<unsigned>& out,
                   std::vector<unsigned char>& oc) {
    const int gpr = cols / 32, nt = rows / 16;
    out.assign((size_t)nt * gpr * 64, 0);
    oc.assign((size_t)nt * gpr * 16, 0);
    for (int t = 0; t < nt; t++)
        for (int g = 0; g < gpr; g++) {
            for (int l = 0; l < 32; l++) {
                const int gid = l >> 2, tig = l & 3;
                for (int s = 0; s < 2; s++) {
                    unsigned X = 0;
                    for (int m = 0; m < 4; m++) {
                        const int r = t * 16 + gid + ((m & 1) ? 8 : 0);
                        const int k = 16 * s + 2 * tig + ((m & 2) ? 8 : 0);
                        X |= nib_at(W, gpr, r, g, k) << (4 * m);
                        X |= nib_at(W, gpr, r, g, k + 1) << (4 * m + 16);
                    }
                    out[(((size_t)t * gpr + g) * 32 + l) * 2 + s] = X;
                }
            }
            for (int i = 0; i < 16; i++) oc[((size_t)t * gpr + g) * 16 + i] = C[(size_t)(t * 16 + i) * gpr + g];
        }
}

int main() {
    srand(7);
    int shapes[][2] = {{248320, 5120}, {65536, 5120}, {5120, 17408}, {5120, 6144}, {12288, 5120}, {1024, 5120}, {10240, 5120}, {17408, 5120}};
    struct V { const char* name; void* f; int nt, tb, ks; } vars[] = {
        {"t2k8", (void*)p1_t2_k8, 1, 2, 8}, {"t2k16", (void*)p1_t2_k16, 1, 2, 16}, {"t1k16", (void*)p1_t1_k16, 1, 1, 16},
        {"2:t2k8", (void*)p2_t2_k8, 2, 2, 8}, {"2:t2k16", (void*)p2_t2_k16, 2, 2, 16}, {"2:t1k16", (void*)p2_t1_k16, 2, 1, 16}};
    for (auto& s : shapes) {
        const int rows = s[0], cols = s[1], gpr = cols / 32;
        size_t nib = (size_t)rows * gpr * 16, nc = (size_t)rows * gpr, nl = (size_t)rows * 32;
        std::vector<unsigned char> hW(nib), hC(nc);
        std::vector<float> hL(nl);
        for (auto& b : hW) b = rand() & 0xff;
        for (auto& b : hC) b = rand() & 31;
        for (auto& f : hL) f = (rand() / (float)RAND_MAX) * 0.02f + 0.001f;
        std::vector<unsigned> pW; std::vector<unsigned char> pC;
        repack(hW.data(), hC.data(), rows, cols, pW, pC);
        const int copies = 4;
        std::vector<void*> W(copies), C(copies), PW(copies), PC(copies), L(copies);
        for (int i = 0; i < copies; i++) {
            CK(cudaMalloc(&W[i], nib)); CK(cudaMemcpy(W[i], hW.data(), nib, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&C[i], nc)); CK(cudaMemcpy(C[i], hC.data(), nc, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&PW[i], nib)); CK(cudaMemcpy(PW[i], pW.data(), nib, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&PC[i], nc)); CK(cudaMemcpy(PC[i], pC.data(), nc, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&L[i], nl * 4)); CK(cudaMemcpy(L[i], hL.data(), nl * 4, cudaMemcpyHostToDevice));
        }
        double bytes = (double)nib + nc + nl * 4;
        for (int N : {1, 8, 16}) {
            const int NP = N <= 8 ? 8 : 16;
            std::vector<float> hx((size_t)N * cols);
            for (auto& f : hx) f = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
            std::vector<unsigned short> hh((size_t)NP * cols, 0), hl((size_t)NP * cols, 0);
            std::vector<float> hs(NP, 1.0f);
            for (int j = 0; j < N; j++)
                for (int c = 0; c < cols; c++) {
                    float v = hx[(size_t)j * cols + c];
                    unsigned short a = h_of(v);
                    hh[(size_t)j * cols + c] = a;
                    hl[(size_t)j * cols + c] = h_of(v - f_of(a));
                }
            float *x, *y, *y2, *xs; u16 *xh, *xl;
            CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&xh, hh.size() * 2)); CK(cudaMemcpy(xh, hh.data(), hh.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&xl, hl.size() * 2)); CK(cudaMemcpy(xl, hl.data(), hl.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&xs, NP * 4)); CK(cudaMemcpy(xs, hs.data(), NP * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&y, (size_t)N * rows * 4)); CK(cudaMalloc(&y2, (size_t)N * rows * 4));
            // scalar reference
            {
                int R = N <= 8 ? 8 : 4;
                HyJob bb{(u64)W[0], (u64)C[0], (u64)L[0], (u64)y2, 0, rows, 6 | (R << 16), 0, rows};
                HyJob* jd; CK(cudaMalloc(&jd, sizeof bb)); CK(cudaMemcpy(jd, &bb, sizeof bb, cudaMemcpyHostToDevice));
                int shm = N * 32 * 33 * 4;
                void* f = N == 1 ? (void*)hy_mvn_1 : N == 4 ? (void*)hy_mvn_4 : N == 8 ? (void*)hy_mvn_8 : (void*)hy_mvn_16;
                CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, shm));
                ((void (*)(const HyJob*, int, const float*, int, int))f)<<<(rows + 8 * R - 1) / (8 * R), 256, shm>>>(jd, 1, x, cols, cols);
                CK(cudaDeviceSynchronize());
                cudaFree(jd);
            }
            std::vector<float> o2((size_t)N * rows);
            CK(cudaMemcpy(o2.data(), y2, o2.size() * 4, cudaMemcpyDeviceToHost));
            for (auto& v : vars) {
                if ((NP == 16) != (v.nt == 2)) continue;
                std::vector<HyJob*> jm(copies);
                for (int i = 0; i < copies; i++) {
                    HyJob a{(u64)PW[i], (u64)PC[i], (u64)L[i], (u64)y, 0, rows, 6, 0, rows};
                    CK(cudaMalloc(&jm[i], sizeof a)); CK(cudaMemcpy(jm[i], &a, sizeof a, cudaMemcpyHostToDevice));
                }
                const int blocks = (rows / 16 + v.tb - 1) / v.tb;
                mmp_fn fn = (mmp_fn)v.f;
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                for (int i = 0; i < 4; i++) fn<<<blocks, 32 * v.ks>>>(jm[i % copies], 1, xh, xl, xs, N, cols);
                CK(cudaEventRecord(a));
                const int reps = 40;
                for (int i = 0; i < reps; i++) fn<<<blocks, 32 * v.ks>>>(jm[i % copies], 1, xh, xl, xs, N, cols);
                CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
                float ms; cudaEventElapsedTime(&ms, a, b);
                float us = ms * 1000 / reps;
                fn<<<blocks, 32 * v.ks>>>(jm[0], 1, xh, xl, xs, N, cols);
                CK(cudaDeviceSynchronize());
                std::vector<float> o1((size_t)N * rows);
                CK(cudaMemcpy(o1.data(), y, o1.size() * 4, cudaMemcpyDeviceToHost));
                double md = 0, mx = 0;
                for (size_t i = 0; i < o1.size(); i++) { md = fmax(md, fabs(o1[i] - o2[i])); mx = fmax(mx, fabs(o2[i])); }
                printf("%-7s N=%2d %6dx%-6d %7.1f us %5.0f GB/s | max |d| %.2g of %.3g\n", v.name, N, rows, cols, us,
                       bytes / us / 1e3, md, mx);
                for (int i = 0; i < copies; i++) cudaFree(jm[i]);
            }
            cudaFree(x); cudaFree(xh); cudaFree(xl); cudaFree(xs); cudaFree(y); cudaFree(y2);
        }
        for (int i = 0; i < copies; i++) { cudaFree(W[i]); cudaFree(C[i]); cudaFree(PW[i]); cudaFree(PC[i]); cudaFree(L[i]); }
    }
}
