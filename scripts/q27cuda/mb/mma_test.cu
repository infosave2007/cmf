// The tensor-core small-batch q4tp matvec, standalone: timing on the shapes of
// Qwen3.8-27B and a check against the f32 scalar kernel of hyb.cu.
//   nvcc -O3 -arch=sm_86 -DHPK=6 -DHD=256 -w mma_test.cu -o mma_test
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "hyb.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

// byte `tig` of a nibble word as f16x2 (low nibble, high nibble), each minus 8
__device__ __forceinline__ u32 nib2h(u32 w, int sh) {
    const u32 b = (w >> sh) & 0xFFu;
    const u32 h = (b & 0x0Fu) | ((b & 0xF0u) << 12) | 0x64006400u;
    u32 r;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(r) : "r"(h), "r"(0x64086408u));
    return r;
}

__device__ __forceinline__ void mma16816(float* c, const u32* a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// xh, xl [NP][cols] f16 (NP = 8 NT, rows past N zero), xsc [NP] the rows'
// power-of-two scales: x = (xh + xl) * xsc. A block (4 warps) a 16-row tile,
// the warps on the groups g = warp (mod KS), summed in a fixed order.
template <int NT, int KS>
__device__ __forceinline__ void mma_body(const HyJob* __restrict__ jobs, int njobs, const u16* __restrict__ xh,
                                         const u16* __restrict__ xl, const float* __restrict__ xsc, int nrows,
                                         int cols)
{
    __shared__ float lad_s[16][33];
    __shared__ float part[KS][NT][32][4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int jx = 0;
    while (jx + 1 < njobs && (int)blockIdx.x >= jobs[jx + 1].blk0) jx++;
    const HyJob jb = jobs[jx];
    const int r0 = ((int)blockIdx.x - jb.blk0) * 16;
    const int rows = jb.rows;
    for (int i = tid; i < 16 * 32; i += blockDim.x) {
        const int rr = r0 + (i >> 5);
        lad_s[i >> 5][i & 31] = rr < rows ? ((const float*)jb.aux1)[(size_t)rr * 32 + (i & 31)] : 0.0f;
    }
    __syncthreads();
    const int gid = lane >> 2, tig = lane & 3, sh = tig * 8;
    const int ra = r0 + gid, rb = r0 + gid + 8;
    const bool oka = ra < rows, okb = rb < rows;
    const int gpr = cols >> 5;
    const uint4* Wa = (const uint4*)jb.w + (size_t)ra * gpr;
    const uint4* Wb = (const uint4*)jb.w + (size_t)rb * gpr;
    const u8* Ca = (const u8*)jb.aux0 + (size_t)ra * gpr;
    const u8* Cb = (const u8*)jb.aux0 + (size_t)rb * gpr;
    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; t++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[t][i] = 0.0f;
    const uint4 zw = make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);
#pragma unroll 4
    for (int g = warp; g < gpr; g += KS) {
        const uint4 wa = oka ? Wa[g] : zw;
        const uint4 wb = okb ? Wb[g] : zw;
        const float sa = lad_s[gid][oka ? Ca[g] : 0];
        const float sb = lad_s[gid + 8][okb ? Cb[g] : 0];
        u32 A0[4] = {nib2h(wa.x, sh), nib2h(wb.x, sh), nib2h(wa.y, sh), nib2h(wb.y, sh)};
        u32 A1[4] = {nib2h(wa.z, sh), nib2h(wb.z, sh), nib2h(wa.w, sh), nib2h(wb.w, sh)};
        const int k0 = g * 32 + 2 * tig;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const size_t xo = (size_t)(t * 8 + gid) * cols + k0;
            const u32 h0 = *(const u32*)(xh + xo), h1 = *(const u32*)(xh + xo + 8);
            const u32 h2 = *(const u32*)(xh + xo + 16), h3 = *(const u32*)(xh + xo + 24);
            const u32 l0 = *(const u32*)(xl + xo), l1 = *(const u32*)(xl + xo + 8);
            const u32 l2 = *(const u32*)(xl + xo + 16), l3 = *(const u32*)(xl + xo + 24);
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma16816(c, A0, h0, h1);
            mma16816(c, A0, l0, l1);
            mma16816(c, A1, h2, h3);
            mma16816(c, A1, l2, l3);
            acc[t][0] = fmaf(sa, c[0], acc[t][0]);
            acc[t][1] = fmaf(sa, c[1], acc[t][1]);
            acc[t][2] = fmaf(sb, c[2], acc[t][2]);
            acc[t][3] = fmaf(sb, c[3], acc[t][3]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++)
#pragma unroll
        for (int i = 0; i < 4; i++) part[warp][t][lane][i] = acc[t][i];
    __syncthreads();
    if (warp == 0) {
        float* y = (float*)jb.y;
        const float* res = (const float*)jb.res;
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float v = part[0][t][lane][i];
#pragma unroll
                for (int k = 1; k < KS; k++) v = __fadd_rn(v, part[k][t][lane][i]);
                const int j = t * 8 + 2 * tig + (i & 1);
                const int r = i < 2 ? ra : rb;
                if (j < nrows && r < rows) {
                    v = __fmul_rn(v, xsc[j]);
                    const size_t at = (size_t)j * jb.ystride + r;
                    y[at] = res ? __fadd_rn(res[at], v) : v;
                }
            }
    }
}

extern "C" __global__ void __launch_bounds__(128) mma1(const HyJob* j, int nj, const u16* xh, const u16* xl,
                                                       const float* xs, int n, int cols)
{
    mma_body<1, 4>(j, nj, xh, xl, xs, n, cols);
}
extern "C" __global__ void __launch_bounds__(128) mma2(const HyJob* j, int nj, const u16* xh, const u16* xl,
                                                       const float* xs, int n, int cols)
{
    mma_body<2, 4>(j, nj, xh, xl, xs, n, cols);
}


// The mma-friendly layout: a 16-row tile's group is 256 bytes, lane l's eight
// bytes at l * 8 (row gid: bytes tig, 4+tig, 8+tig, 12+tig of its group, then
// the same of row gid + 8); the tile's 16 codes of the group follow apart.
template <int NT, int KS>
__device__ __forceinline__ void mmp_body(const HyJob* __restrict__ jobs, int njobs, const u16* __restrict__ xh,
                                         const u16* __restrict__ xl, const float* __restrict__ xsc, int nrows,
                                         int cols)
{
    __shared__ float lad_s[16][33];
    __shared__ float part[KS][NT][32][4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const HyJob jb = jobs[0];
    const int tile = blockIdx.x;
    const int r0 = tile * 16;
    const int rows = jb.rows;
    for (int i = tid; i < 16 * 32; i += blockDim.x) {
        const int rr = r0 + (i >> 5);
        lad_s[i >> 5][i & 31] = rr < rows ? ((const float*)jb.aux1)[(size_t)rr * 32 + (i & 31)] : 0.0f;
    }
    __syncthreads();
    const int gid = lane >> 2, tig = lane & 3;
    const int ra = r0 + gid, rb = r0 + gid + 8;
    const int gpr = cols >> 5;
    const uint2* T = (const uint2*)jb.w + (size_t)tile * gpr * 32;
    const u8* Cc = (const u8*)jb.aux0 + (size_t)tile * gpr * 16;
    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; t++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[t][i] = 0.0f;
    const int per = (gpr + KS - 1) / KS;
    const int g0 = warp * per, g1 = min(gpr, g0 + per);
#pragma unroll 4
    for (int g = g0; g < g1; g++) {
        const uint2 wv = T[(size_t)g * 32 + lane];
        const float sa = lad_s[gid][Cc[(size_t)g * 16 + gid]];
        const float sb = lad_s[gid + 8][Cc[(size_t)g * 16 + gid + 8]];
        u32 A0[4] = {nib2h(wv.x, 0), nib2h(wv.y, 0), nib2h(wv.x, 8), nib2h(wv.y, 8)};
        u32 A1[4] = {nib2h(wv.x, 16), nib2h(wv.y, 16), nib2h(wv.x, 24), nib2h(wv.y, 24)};
        const int k0 = g * 32 + 2 * tig;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const size_t xo = (size_t)(t * 8 + gid) * cols + k0;
            const u32 h0 = *(const u32*)(xh + xo), h1 = *(const u32*)(xh + xo + 8);
            const u32 h2 = *(const u32*)(xh + xo + 16), h3 = *(const u32*)(xh + xo + 24);
            const u32 l0 = *(const u32*)(xl + xo), l1 = *(const u32*)(xl + xo + 8);
            const u32 l2 = *(const u32*)(xl + xo + 16), l3 = *(const u32*)(xl + xo + 24);
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma16816(c, A0, h0, h1);
            mma16816(c, A0, l0, l1);
            mma16816(c, A1, h2, h3);
            mma16816(c, A1, l2, l3);
            acc[t][0] = fmaf(sa, c[0], acc[t][0]);
            acc[t][1] = fmaf(sa, c[1], acc[t][1]);
            acc[t][2] = fmaf(sb, c[2], acc[t][2]);
            acc[t][3] = fmaf(sb, c[3], acc[t][3]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++)
#pragma unroll
        for (int i = 0; i < 4; i++) part[warp][t][lane][i] = acc[t][i];
    __syncthreads();
    if (warp == 0) {
        float* y = (float*)jb.y;
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float v = part[0][t][lane][i];
#pragma unroll
                for (int k = 1; k < KS; k++) v = __fadd_rn(v, part[k][t][lane][i]);
                const int j = t * 8 + 2 * tig + (i & 1);
                const int r = i < 2 ? ra : rb;
                if (j < nrows && r < rows) y[(size_t)j * jb.ystride + r] = __fmul_rn(v, xsc[j]);
            }
    }
}
extern "C" __global__ void __launch_bounds__(128) mmp1(const HyJob* j, int nj, const u16* xh, const u16* xl,
                                                       const float* xs, int n, int cols)
{
    mmp_body<1, 4>(j, nj, xh, xl, xs, n, cols);
}
extern "C" __global__ void __launch_bounds__(128) mmp2(const HyJob* j, int nj, const u16* xh, const u16* xl,
                                                       const float* xs, int n, int cols)
{
    mmp_body<2, 4>(j, nj, xh, xl, xs, n, cols);
}
extern "C" __global__ void __launch_bounds__(256) mmp1w(const HyJob* j, int nj, const u16* xh, const u16* xl,
                                                       const float* xs, int n, int cols)
{
    mmp_body<1, 8>(j, nj, xh, xl, xs, n, cols);
}

static unsigned short h_of(float f) {
    // host f32 -> f16, round to nearest even (normal range only, as used)
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

int main() {
    srand(7);
    int shapes[][2] = {{17408, 5120}, {5120, 17408}, {10240, 5120}, {5120, 6144}, {12288, 5120}, {1024, 5120}};
    for (auto& s : shapes) {
        const int rows = s[0], cols = s[1], gpr = cols / 32;
        size_t nib = (size_t)rows * gpr * 16, nc = (size_t)rows * gpr, nl = (size_t)rows * 32;
        std::vector<unsigned char> hW(nib), hC(nc);
        std::vector<float> hL(nl);
        for (auto& b : hW) b = rand() & 0xff;
        for (auto& b : hC) b = rand() & 31;
        for (auto& f : hL) f = (rand() / (float)RAND_MAX) * 0.02f + 0.001f;
        const int copies = 4;
        std::vector<void*> W(copies), C(copies), L(copies);
        for (int i = 0; i < copies; i++) {
            CK(cudaMalloc(&W[i], nib)); CK(cudaMemcpy(W[i], hW.data(), nib, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&C[i], nc)); CK(cudaMemcpy(C[i], hC.data(), nc, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&L[i], nl * 4)); CK(cudaMemcpy(L[i], hL.data(), nl * 4, cudaMemcpyHostToDevice));
        }
        for (int N : {1, 2, 4, 8, 16}) {
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
            std::vector<HyJob*> jm(copies), js(copies);
            for (int i = 0; i < copies; i++) {
                HyJob a{(u64)W[i], (u64)C[i], (u64)L[i], (u64)y, 0, rows, 6, 0, rows};
                HyJob b{(u64)W[i], (u64)C[i], (u64)L[i], (u64)y2, 0, rows, 6 | (8 << 16), 0, rows};
                CK(cudaMalloc(&jm[i], sizeof a)); CK(cudaMemcpy(jm[i], &a, sizeof a, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&js[i], sizeof b)); CK(cudaMemcpy(js[i], &b, sizeof b, cudaMemcpyHostToDevice));
            }
            const int blocks = (rows + 15) / 16;
            auto launch = [&](int i) {
                if (NP == 8) mma1<<<blocks, 128>>>(jm[i % copies], 1, xh, xl, xs, N, cols);
                else mma2<<<blocks, 128>>>(jm[i % copies], 1, xh, xl, xs, N, cols);
            };
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            for (int i = 0; i < 4; i++) launch(i);
            CK(cudaEventRecord(a));
            const int reps = 40;
            for (int i = 0; i < reps; i++) launch(i);
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            float ms; cudaEventElapsedTime(&ms, a, b);
            float us = ms * 1000 / reps;
            // the scalar kernel's answer on the same rows
            int shm = N * 32 * 33 * 4;
            void* f = N == 1 ? (void*)hy_mvn_1 : N == 2 ? (void*)hy_mvn_2 : N == 4 ? (void*)hy_mvn_4 : N == 8 ? (void*)hy_mvn_8 : (void*)hy_mvn_16;
            CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, shm));
            int R = N <= 8 ? 8 : 4;
            int sblocks = (rows + 8 * R - 1) / (8 * R);
            if (R == 4) { HyJob bb{(u64)W[0], (u64)C[0], (u64)L[0], (u64)y2, 0, rows, 6 | (4 << 16), 0, rows}; CK(cudaMemcpy(js[0], &bb, sizeof bb, cudaMemcpyHostToDevice)); }
            ((void (*)(const HyJob*, int, const float*, int, int))f)<<<sblocks, 256, shm>>>(js[0], 1, x, cols, cols);
            CK(cudaDeviceSynchronize());
            launch(0);
            CK(cudaDeviceSynchronize());
            std::vector<float> o1((size_t)N * rows), o2((size_t)N * rows);
            CK(cudaMemcpy(o1.data(), y, o1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(o2.data(), y2, o2.size() * 4, cudaMemcpyDeviceToHost));
            double md = 0, mx = 0;
            for (size_t i = 0; i < o1.size(); i++) { md = fmax(md, fabs(o1[i] - o2[i])); mx = fmax(mx, fabs(o2[i])); }
            double bytes = (double)nib + nc + nl * 4;
            printf("mma N=%2d %6dx%-6d %8.1f us %6.0f GB/s | vs scalar: max |d| %.3g of max |y| %.3g\n", N, rows, cols, us,
                   bytes / us / 1e3, md, mx);
            for (int var = 0; var < 2; var++) {
                auto lp = [&](int i) {
                    if (var == 1 && NP == 8) mmp1w<<<blocks, 256>>>(jm[i % copies], 1, xh, xl, xs, N, cols);
                    else if (NP == 8) mmp1<<<blocks, 128>>>(jm[i % copies], 1, xh, xl, xs, N, cols);
                    else mmp2<<<blocks, 128>>>(jm[i % copies], 1, xh, xl, xs, N, cols);
                };
                for (int i = 0; i < 4; i++) lp(i);
                CK(cudaEventRecord(a));
                for (int i = 0; i < reps; i++) lp(i);
                CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
                cudaEventElapsedTime(&ms, a, b);
                float up = ms * 1000 / reps;
                printf("    permuted layout%s: %8.1f us %6.0f GB/s\n", var ? " KS=8" : "", up, bytes / up / 1e3);
            }
            cudaFree(x); cudaFree(xh); cudaFree(xl); cudaFree(xs); cudaFree(y); cudaFree(y2);
            for (int i = 0; i < copies; i++) { cudaFree(jm[i]); cudaFree(js[i]); }
        }
        for (int i = 0; i < copies; i++) { cudaFree(W[i]); cudaFree(C[i]); cudaFree(L[i]); }
    }
}
