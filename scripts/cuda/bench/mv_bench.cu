// Standalone micro-benchmark of the decode matvec kernel (`mv_multi`), built
// with nvcc on a CUDA box:
//
//   nvcc -O3 -arch=sm_86 -o mv_bench mv_bench.cu && ./mv_bench
//
// It includes the engine's own kernel source, fills a random q4tp / q8_2f
// matrix of a real shape, and reports the time of one launch (best of N) and
// the rate at which the weight bytes are read, against a load-only kernel
// (the card's bus) — so a kernel variant is compared inside one process.
// Variants under test are compiled with -DVARIANT=n into `mv_var`.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../../../crates/cortiq-engine/src/gpu_cuda/kernels.cu"

// Load-only: every weight word is read and folded, nothing else.
__global__ void mv_loadonly(const uint4* __restrict__ W, float* __restrict__ y, int rows, int words, int rpb)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, NW = blockDim.x >> 5;
    const int row0 = blockIdx.x * rpb, row1 = min(row0 + rpb, rows);
    for (int r = row0 + warp; r < row1; r += NW) {
        u32 acc = 0;
        const uint4* w = W + (size_t)r * words;
        for (int g = lane; g < words; g += 32) {
            const uint4 v = w[g];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        for (int o = 16; o > 0; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) y[r] = (float)(acc & 1023u);
    }
}

#ifdef VARIANT
#include "mv_variants.cuh"
#endif

static double now_ms(cudaEvent_t a, cudaEvent_t b) { float ms; cudaEventElapsedTime(&ms, a, b); return ms; }

int main(int argc, char** argv)
{
    struct Shape { const char* name; int rows, cols, kind; };
    std::vector<Shape> shapes = {
        {"gate  q4tp 10240x2560 ", 10240, 2560, 6}, {"down  q4tp 2560x10240 ", 2560, 10240, 6},
        {"q     q4tp 4096x2560  ", 4096, 2560, 6},  {"k     q4tp 1024x2560  ", 1024, 2560, 6},
        {"o     q4tp 2560x4096  ", 2560, 4096, 6},  {"head  q4tp 131072x2560", 131072, 2560, 6},
        {"gate  q8_2f 10240x2560", 10240, 2560, 7}, {"down  q8_2f 2560x10240", 2560, 10240, 7},
    };
    int threads = argc > 1 ? atoi(argv[1]) : 256;
    int rpb_arg = argc > 2 ? atoi(argv[2]) : 0;
    int reps = 30;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (auto& s : shapes) {
        const int rows = s.rows, cols = s.cols, gpr = cols / 32;
        size_t wbytes = s.kind == 6 ? (size_t)rows * gpr * 16 : (size_t)rows * cols;
        std::vector<unsigned char> hw(wbytes);
        for (size_t i = 0; i < wbytes; i++) hw[i] = (unsigned char)(rand() & 255);
        std::vector<unsigned char> hc((size_t)rows * gpr);
        for (auto& c : hc) c = rand() & 31;
        std::vector<float> hl(s.kind == 6 ? (size_t)rows * 32 : (size_t)rows), hx(cols), hcol(cols, 1.0f);
        for (auto& v : hl) v = 0.001f + 1e-4f * (rand() & 255);
        for (auto& v : hx) v = (rand() & 1023) / 512.0f - 1.0f;
        unsigned char *dw, *dc; float *dl, *dx, *dy, *dcol; MvJob* dj;
        // MV_ALIGN=<bytes>: the weight plane placed at that offset past a 2 MiB
        // boundary (0 = on it) instead of wherever the allocator put it — a
        // matrix's rate was seen to differ between runs of one binary.
        unsigned char* dw_raw;
        cudaMalloc(&dw_raw, wbytes + (4 << 20));
        dw = dw_raw;
        if (const char* al = getenv("MV_ALIGN")) {
            size_t a = ((size_t)dw_raw + (2 << 20) - 1) & ~(size_t)((2 << 20) - 1);
            dw = (unsigned char*)(a + (size_t)atol(al));
        }
        if (getenv("MV_ADDR")) printf("[w at %p, %zu past 2 MiB] ", (void*)dw, (size_t)dw & ((2 << 20) - 1));
        cudaMalloc(&dc, hc.size() + 64); cudaMalloc(&dl, hl.size() * 4 + 64);
        cudaMalloc(&dx, cols * 4); cudaMalloc(&dy, rows * 4); cudaMalloc(&dj, sizeof(MvJob)); cudaMalloc(&dcol, cols * 4);
        cudaMemcpy(dw, hw.data(), wbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dc, hc.data(), hc.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dl, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx.data(), cols * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dcol, hcol.data(), cols * 4, cudaMemcpyHostToDevice);
        int rpb = rpb_arg ? rpb_arg : ((rows / 256 > 0 ? rows / 256 : 1) + 7) / 8 * 8;
        if (!rpb_arg) { if (rpb < 8) rpb = 8; if (rpb > 64) rpb = 64; }
        MvJob job;
        job.w = (u64)dw; job.aux0 = s.kind == 6 ? (u64)dc : (u64)dl; job.aux1 = s.kind == 6 ? (u64)dl : (u64)dcol;
        job.y = (u64)dy; job.res = 0; job.rows = rows; job.kind = s.kind; job.blk0 = 0; job.rpb = rpb;
        cudaMemcpy(dj, &job, sizeof(job), cudaMemcpyHostToDevice);
        const int blocks = (rows + rpb - 1) / rpb;
        const int cpad = (cols + 31) & ~31;
        const size_t shm = 4 * (16 + cpad + 64 + cpad / 32 + 8);
        cudaFuncSetAttribute(mv_multi, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 << 10);
        double best[3] = {1e30, 1e30, 1e30};
        for (int it = 0; it < reps; it++) {
            cudaEventRecord(e0);
            mv_multi<<<blocks, threads, shm>>>(dj, 1, dx, nullptr, -1, nullptr, 1e-6f, 0, cols);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            double t = now_ms(e0, e1); if (t < best[0]) best[0] = t;
            cudaEventRecord(e0);
            mv_loadonly<<<blocks, threads>>>((const uint4*)dw, dy, rows, (int)(wbytes / rows / 16), rpb);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            t = now_ms(e0, e1); if (t < best[1]) best[1] = t;
#ifdef VARIANT
            cudaEventRecord(e0);
            mv_var_launch(blocks, threads, shm, dj, dx, cols);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            t = now_ms(e0, e1); if (t < best[2]) best[2] = t;
#endif
        }
        cudaError_t err = cudaGetLastError();
        printf("%s rpb %2d x %3d thr: mv_multi %7.1f us %6.0f GB/s | load-only %7.1f us %6.0f GB/s", s.name, rpb, threads,
               best[0] * 1e3, wbytes / best[0] / 1e6, best[1] * 1e3, wbytes / best[1] / 1e6);
#ifdef VARIANT
        printf(" | variant %7.1f us %6.0f GB/s", best[2] * 1e3, wbytes / best[2] / 1e6);
#endif
        printf("%s\n", err == cudaSuccess ? "" : cudaGetErrorString(err));
        cudaFree(dw_raw); cudaFree(dc); cudaFree(dl); cudaFree(dx); cudaFree(dy); cudaFree(dj); cudaFree(dcol);
    }
    return 0;
}
