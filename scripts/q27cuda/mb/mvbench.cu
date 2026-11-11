// Standalone timing of the hybrid arm's small-path matvec kernels on the
// shapes of Qwen3.8-27B (nvcc -O3 -arch=sm_86 -DHPK=6 -DHD=256 mvbench.cu).
//   ./mvbench [kernel-src]   (the kernel source is #included at build time)
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef SRC
#define SRC "hyb.cu"
#endif
#include SRC

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

typedef void (*mvn_fn)(const HyJob*, int, const float*, int, int);

static float run(void* f, int N, int rows, int cols, int R, int rpw, int reps) {
    const int gpr = cols / 32;
    size_t nib = (size_t)rows * gpr * 16, nc = (size_t)rows * gpr, nl = (size_t)rows * 32;
    // three copies of the weights so a launch does not read them from L2
    const int copies = 4;
    std::vector<void*> W(copies), C(copies), L(copies);
    for (int i = 0; i < copies; i++) {
        CK(cudaMalloc(&W[i], nib)); CK(cudaMemset(W[i], 0x5a, nib));
        CK(cudaMalloc(&C[i], nc)); CK(cudaMemset(C[i], 3, nc));
        CK(cudaMalloc(&L[i], nl * 4)); CK(cudaMemset(L[i], 0, nl * 4));
    }
    float *x, *y; CK(cudaMalloc(&x, (size_t)N * cols * 4)); CK(cudaMemset(x, 0, (size_t)N * cols * 4));
    CK(cudaMalloc(&y, (size_t)N * rows * 4));
    std::vector<HyJob*> jobs(copies);
    int blocks = (rows + 8 * rpw - 1) / (8 * rpw);
    for (int i = 0; i < copies; i++) {
        HyJob j{(u64)W[i], (u64)C[i], (u64)L[i], (u64)y, 0, rows, 6 | (rpw << 16), 0, rows};
        CK(cudaMalloc(&jobs[i], sizeof(HyJob))); CK(cudaMemcpy(jobs[i], &j, sizeof j, cudaMemcpyHostToDevice));
    }
    mvn_fn fn = (mvn_fn)f;
    int shm = N * 32 * 33 * 4;
    CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, shm));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 4; i++) fn<<<blocks, 256, shm>>>(jobs[i % copies], 1, x, cols, cols);
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; i++) fn<<<blocks, 256, shm>>>(jobs[i % copies], 1, x, cols, cols);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    float us = ms * 1000 / reps;
    double bytes = (double)nib + nc + nl * 4;
    printf("N=%2d R=%d rpw=%d %6dx%-6d %8.1f us  %6.0f GB/s (weights %.0f)\n", N, R, rpw, rows, cols, us, bytes / us / 1e3,
           (double)nib / us / 1e3);
    for (int i = 0; i < copies; i++) { cudaFree(W[i]); cudaFree(C[i]); cudaFree(L[i]); cudaFree(jobs[i]); }
    cudaFree(x); cudaFree(y);
    return us;
}

extern "C" __global__ void __launch_bounds__(256) mv1_r16(const HyJob* __restrict__ jobs, int njobs, const float* __restrict__ x, int xstride, int cols) { mvn_body<1, 16>(jobs, njobs, x, xstride, cols); }
extern "C" __global__ void __launch_bounds__(256) mv2_r16(const HyJob* __restrict__ jobs, int njobs, const float* __restrict__ x, int xstride, int cols) { mvn_body<2, 16>(jobs, njobs, x, xstride, cols); }
int main(int argc, char** argv) {
    int shapes[][2] = {{17408, 5120}, {5120, 17408}, {10240, 5120}, {5120, 6144}, {12288, 5120}};
    for (auto& s : shapes) {
        run((void*)hy_mvn_1, 1, s[0], s[1], 8, 8, 50);
        run((void*)mv1_r16, 1, s[0], s[1], 16, 16, 50);
        run((void*)mv2_r16, 2, s[0], s[1], 16, 16, 50);
    }
}
