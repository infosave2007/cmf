// The HunyuanVideo VAE conv on CUDA as the engine runs it (hv_im2col, an f16
// cuBLAS product, hv_store) at the decoder's widest shapes: the seconds of
// each part, and a plain f16 GEMM of the same size for the ceiling.
// nvcc -O3 -arch=sm_86 conv_bench.cu -lcublas -o cb && ./cb
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "/root/h3i/srcrs/crates/cortiq-engine/src/gpu_cuda/s2v_vae.cu"

int main() {
    cublasHandle_t hb;
    cublasCreate(&hb);
    struct S { int c, F, H, W; } shapes[] = {{128, 17, 512, 512}, {256, 17, 256, 256}, {512, 9, 128, 128}};
    const size_t PANEL = 192u << 20;
    for (auto s : shapes) {
        const int k = 3, taps = 27;
        const size_t P = (size_t)s.F * s.H * s.W, kk = (size_t)s.c * taps;
        const size_t np = std::min(PANEL / (kk * 2), PANEL / ((size_t)s.c * 4));
        u16 *x, *cols, *w, *dst;
        float *yc, *bias;
        cudaMalloc(&x, (size_t)s.c * P * 2);
        cudaMalloc(&dst, (size_t)s.c * P * 2);
        cudaMalloc(&cols, np * kk * 2);
        cudaMalloc(&w, (size_t)s.c * kk * 2);
        cudaMalloc(&yc, np * s.c * 4);
        cudaMalloc(&bias, s.c * 4);
        cudaMemset(x, 0, (size_t)s.c * P * 2);
        cudaMemset(w, 0, (size_t)s.c * kk * 2);
        cudaMemset(bias, 0, s.c * 4);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        float t_im = 0, t_mm = 0, t_st = 0, ms;
        const float one = 1.0f, zero = 0.0f;
        for (size_t p0 = 0; p0 < P; p0 += np) {
            const size_t n = std::min(np, P - p0);
            cudaEventRecord(e0);
            const unsigned long long tot = n * s.c;
            hv_im2col<<<(tot + 255) / 256, 256>>>(x, cols, s.c, s.F, s.H, s.W, k, k, k, s.H, s.W, 1, 1, 1, p0, n);
            cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); t_im += ms;
            cudaEventRecord(e0);
            cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, s.c, (int)kk, &one, cols, CUDA_R_16F, (int)n, w, CUDA_R_16F, (int)kk,
                         &zero, yc, CUDA_R_32F, (int)n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); t_mm += ms;
            cudaEventRecord(e0);
            const unsigned long long ts = n * s.c;
            hv_store<<<(ts + 255) / 256, 256>>>(yc, dst, bias, nullptr, s.c, n, P, p0);
            cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); t_st += ms;
        }
        const double fl = 2.0 * P * kk * s.c;
        printf("c=%d %dx%dx%d: im2col %.1f ms, gemm %.1f ms (%.1f TF), store %.1f ms, total %.1f ms; chunk %zu\n", s.c, s.F, s.H, s.W,
               t_im, t_mm, fl / t_mm / 1e9, t_st, t_im + t_mm + t_st, np);
        cudaFree(x); cudaFree(dst); cudaFree(cols); cudaFree(w); cudaFree(yc); cudaFree(bias);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("cuda error %s\n", cudaGetErrorString(e));
    return 0;
}
