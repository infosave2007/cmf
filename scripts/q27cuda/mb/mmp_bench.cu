// The small path's tensor-core matvec (hyb.cu `hy_mmp_*`) against candidate
// variants on the shapes of Qwen3.8-27B: GB/s of weights read, and whether
// the outputs are the same bits. Random repacked weights (timing only).
//   nvcc -O3 -arch=sm_86 -DHPK=6 -DHD=256 -w mmp_bench.cu -o mmp_bench
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "hyb.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

// Candidate: mmp_body with the next group's weight word and codes loaded
// before this group's mma (PF), same arithmetic and order as mmp_body.
template <int NT, int TB, int KS>
__device__ __forceinline__ void mmp_pf_body(const HyJob* __restrict__ jobs, int njobs, const u16* __restrict__ x0,
                                            const u16* __restrict__ x1, int xstride, int nrows, int cols)
{
    __shared__ float2 par_s[16 * TB];
    __shared__ float part[KS][TB][NT][32][4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int jx = 0;
    while (jx + 1 < njobs && (int)blockIdx.x >= jobs[jx + 1].blk0) jx++;
    const HyJob jb = jobs[jx];
    const int tile0 = ((int)blockIdx.x - jb.blk0) * TB;
    const int rows = jb.rows;
    const int ntile = (rows + 15) >> 4;
    for (int i = tid; i < 16 * TB; i += blockDim.x) {
        const int rr = tile0 * 16 + i;
        par_s[i] = rr < rows ? ((const float2*)jb.aux1)[rr] : make_float2(0.0f, 0.0f);
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
    uint2 Xn[TB];
    u8 ca[TB], cb[TB];
    auto load = [&](int g) {
#pragma unroll
        for (int b = 0; b < TB; b++) {
            const int tile = min(tile0 + b, ntile - 1);
            const size_t tg = (size_t)tile * gpr + g;
            Xn[b] = T[tg * 32 + lane];
            ca[b] = Cc[tg * 16 + gid];
            cb[b] = Cc[tg * 16 + gid + 8];
        }
    };
    if (g0 < g1) load(g0);
    for (int g = g0; g < g1; g++) {
        uint2 X[TB];
        u8 cA[TB], cB[TB];
#pragma unroll
        for (int b = 0; b < TB; b++) {
            X[b] = Xn[b];
            cA[b] = ca[b];
            cB[b] = cb[b];
        }
        if (g + 1 < g1) load(g + 1);
        u32 bx[NT][2][4];
        const int k0 = g * 32 + 2 * tig;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const int j = t * 8 + gid;
            if (j < nrows) {
                const size_t xo = (size_t)j * xstride + k0;
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    bx[t][0][q] = *(const u32*)(x0 + xo + 8 * q);
                    bx[t][1][q] = *(const u32*)(x1 + xo + 8 * q);
                }
            } else {
#pragma unroll
                for (int pl = 0; pl < 2; pl++)
#pragma unroll
                    for (int q = 0; q < 4; q++) bx[t][pl][q] = 0u;
            }
        }
#pragma unroll
        for (int b = 0; b < TB; b++) {
            const int tile = tile0 + b;
            if (tile >= ntile) break;
            const float2 pa = par_s[b * 16 + gid], pb = par_s[b * 16 + gid + 8];
            const float sa = exp2f(fmaf((float)cA[b], pa.y, pa.x));
            const float sb = exp2f(fmaf((float)cB[b], pb.y, pb.x));
            const u32 a00 = a_reg(X[b].x, 0), a01 = a_reg(X[b].x, 1), a02 = a_reg(X[b].x, 2), a03 = a_reg(X[b].x, 3);
            const u32 a10 = a_reg(X[b].y, 0), a11 = a_reg(X[b].y, 1), a12 = a_reg(X[b].y, 2), a13 = a_reg(X[b].y, 3);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                if (t * 8 >= nrows) break;
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int pl = 0; pl < 2; pl++) {
                    mma16816(c, a00, a01, a02, a03, bx[t][pl][0], bx[t][pl][1]);
                    mma16816(c, a10, a11, a12, a13, bx[t][pl][2], bx[t][pl][3]);
                }
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
    float* y = (float*)jb.y;
    for (int o = tid; o < TB * NT * 32 * 4; o += blockDim.x) {
        const int i = o & 3, ln = (o >> 2) & 31, t = (o >> 7) % NT, b = (o >> 7) / NT;
        const int gd = ln >> 2, tg = ln & 3;
        const int j = t * 8 + 2 * tg + (i & 1);
        const int r = (tile0 + b) * 16 + gd + (i < 2 ? 0 : 8);
        if (j >= nrows || r >= rows) continue;
        float v = part[0][b][t][ln][i];
#pragma unroll
        for (int k = 1; k < KS; k++) v = __fadd_rn(v, part[k][b][t][ln][i]);
        y[(size_t)j * jb.ystride + r] = v;
    }
}

#define PF(NT, TB, KS)                                                                                            \
    extern "C" __global__ void __launch_bounds__(32 * KS) pf_##NT##_##TB##_##KS(                                   \
        const HyJob* __restrict__ j, int nj, const u16* __restrict__ x0, const u16* __restrict__ x1, int xstride, \
        int n, int cols)                                                                                           \
    {                                                                                                              \
        mmp_pf_body<NT, TB, KS>(j, nj, x0, x1, xstride, n, cols);                                                  \
    }
PF(1, 1, 8)
PF(1, 2, 8)
PF(1, 4, 8)
PF(1, 2, 16)
PF(1, 1, 16)

typedef void (*KF)(const HyJob*, int, const u16*, const u16*, const u16*, int, int, int);
typedef void (*PFK)(const HyJob*, int, const u16*, const u16*, int, int, int);

int main() {
    // rows, cols: gate/up pair, down, GDN in (qkv+z), attention q, out, head, head shortlist
    int shapes[][2] = {{34816, 5120}, {5120, 17408}, {16384, 5120}, {12288, 5120}, {5120, 6144}, {248080, 5120}, {65536, 5120}};
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (auto& s : shapes) {
        const int rows = s[0], cols = s[1], gpr = cols / 32, ntile = rows / 16;
        const size_t nw = (size_t)ntile * gpr * 32 * 2, nc = (size_t)ntile * gpr * 16;
        // a few copies so the weights come from memory
        const int copies = rows * (size_t)cols / 2 > 200000000 ? 1 : 4;
        std::vector<u32*> W(copies);
        std::vector<u8*> C(copies);
        std::vector<float2*> P(copies);
        std::vector<u32> hw(nw);
        for (auto& v : hw) v = (u32)rand() * 2654435761u;
        std::vector<u8> hc(nc);
        for (auto& v : hc) v = rand() & 31;
        std::vector<float2> hp(rows);
        for (auto& v : hp) v = make_float2(-8.0f + (rand() % 100) * 0.01f, 0.1f);
        for (int i = 0; i < copies; i++) {
            CK(cudaMalloc(&W[i], nw * 4));
            CK(cudaMemcpy(W[i], hw.data(), nw * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&C[i], nc));
            CK(cudaMemcpy(C[i], hc.data(), nc, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&P[i], rows * 8));
            CK(cudaMemcpy(P[i], hp.data(), rows * 8, cudaMemcpyHostToDevice));
        }
        const double bytes = (double)nw * 4 + nc + rows * 8;
        for (int N : {1, 5}) {
            std::vector<u16> hx((size_t)16 * cols);
            for (auto& v : hx) v = 0x3c00 ^ (rand() & 0x83ff);
            u16 *x0, *x1;
            CK(cudaMalloc(&x0, hx.size() * 2));
            CK(cudaMalloc(&x1, hx.size() * 2));
            CK(cudaMemcpy(x0, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(x1, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            float *y, *yr;
            CK(cudaMalloc(&y, (size_t)N * rows * 4));
            CK(cudaMalloc(&yr, (size_t)N * rows * 4));
            const int ks = rows >= 16384 ? 8 : 16;
            struct Var { const char* name; void* f; int tb, ks; bool pf; } vars[] = {
                {"hy_mmp_1_2", ks == 8 ? (void*)hy_mmp_1_2_8 : (void*)hy_mmp_1_2_16, 2, ks, false},
                {"hy_mmp_1_1", ks == 8 ? (void*)hy_mmp_1_1_8 : (void*)hy_mmp_1_1_16, 1, ks, false},
                {"pf_1_1", ks == 8 ? (void*)pf_1_1_8 : (void*)pf_1_1_16, 1, ks, true},
                {"pf_1_2", ks == 8 ? (void*)pf_1_2_8 : (void*)pf_1_2_16, 2, ks, true},
                {"pf_1_4 ks8", (void*)pf_1_4_8, 4, 8, true},
            };
            std::vector<float> ref((size_t)N * rows), got((size_t)N * rows);
            bool first = true;
            for (auto& v : vars) {
                // a job table a copy
                std::vector<HyJob*> jd(copies);
                const int blocks = (ntile + v.tb - 1) / v.tb;
                for (int i = 0; i < copies; i++) {
                    HyJob jb{};
                    jb.w = (u64)W[i];
                    jb.aux0 = (u64)C[i];
                    jb.aux1 = (u64)P[i];
                    jb.y = (u64)y;
                    jb.res = 0;
                    jb.rows = rows;
                    jb.kind = 6;
                    jb.ystride = rows;
                    jb.blk0 = 0;
                    CK(cudaMalloc(&jd[i], sizeof(HyJob)));
                    CK(cudaMemcpy(jd[i], &jb, sizeof(HyJob), cudaMemcpyHostToDevice));
                }
                float best = 1e9f;
                for (int rep = 0; rep < 7; rep++) {
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < copies; i++) {
                        if (v.pf)
                            ((PFK)v.f)<<<blocks, 32 * v.ks>>>(jd[i], 1, x0, x1, cols, N, cols);
                        else
                            ((KF)v.f)<<<blocks, 32 * v.ks>>>(jd[i], 1, x0, x1, x1, cols, N, cols);
                    }
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    float ms;
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    best = ms < best ? ms : best;
                }
                CK(cudaGetLastError());
                CK(cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost));
                if (first) ref = got;
                const bool same = memcmp(ref.data(), got.data(), got.size() * 4) == 0;
                printf("%6dx%-6d N=%d %-11s ks%-2d %7.1f us  %6.1f GB/s  %s\n", rows, cols, N, v.name, v.ks,
                       best * 1e3f / copies, bytes * copies / (best * 1e-3) / 1e9, first ? "ref" : (same ? "same bits" : "DIFFERENT"));
                first = false;
                for (auto p : jd) cudaFree(p);
            }
            cudaFree(x0); cudaFree(x1); cudaFree(y); cudaFree(yr);
        }
        for (int i = 0; i < copies; i++) { cudaFree(W[i]); cudaFree(C[i]); cudaFree(P[i]); }
    }
    return 0;
}
