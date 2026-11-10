// cuBLAS GEMM rates on the card for the prompt's shapes, per precision mode:
//
//   nvcc -O3 -o gemm_bench gemm_bench.cu -lcublas && ./gemm_bench [b]
//
// Y [rows x b] = W^T [rows x cols] * X [cols x b] (column-major), the shape of
// a chunk's projection. Modes: f32 (cublasSgemm), TF32 (GemmEx FAST_TF32),
// f16 in / f32 out (GemmEx 16F,16F -> 32F), f16 in / f16 out, FAST_16F.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void to_half(const float* x, __half* h, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = __float2half(x[i]);
}

int main(int argc, char** argv)
{
    int b = argc > 1 ? atoi(argv[1]) : 1024;
    struct Shape { const char* name; int rows, cols; } shapes[] = {
        {"gate 10240x2560", 10240, 2560}, {"down 2560x10240", 2560, 10240}, {"q 4096x2560", 4096, 2560}, {"k 1024x2560", 1024, 2560}};
    cublasHandle_t h; cublasCreate(&h);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (auto& s : shapes) {
        size_t wn = (size_t)s.rows * s.cols, xn = (size_t)s.cols * b, yn = (size_t)s.rows * b;
        std::vector<float> hw(wn), hx(xn);
        for (auto& v : hw) v = (rand() & 1023) / 5120.0f - 0.1f;
        for (auto& v : hx) v = (rand() & 1023) / 512.0f - 1.0f;
        float *W, *X, *Y; __half *Wh, *Xh, *Yh;
        cudaMalloc(&W, wn * 4); cudaMalloc(&X, xn * 4); cudaMalloc(&Y, yn * 4);
        cudaMalloc(&Wh, wn * 2); cudaMalloc(&Xh, xn * 2); cudaMalloc(&Yh, yn * 2);
        cudaMemcpy(W, hw.data(), wn * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(X, hx.data(), xn * 4, cudaMemcpyHostToDevice);
        to_half<<<(wn + 255) / 256, 256>>>(W, Wh, wn);
        to_half<<<(xn + 255) / 256, 256>>>(X, Xh, xn);
        const float one = 1.0f, zero = 0.0f;
        const __half hone = __float2half(1.0f), hzero = __float2half(0.0f);
        const double flop = 2.0 * s.rows * (double)s.cols * b;
        printf("%-16s b=%d:", s.name, b);
        for (int mode = 0; mode < 5; mode++) {
            double best = 1e30; cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
            for (int it = 0; it < 6; it++) {
                cudaEventRecord(e0);
                switch (mode) {
                case 0: st = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, s.rows, b, s.cols, &one, W, s.cols, X, s.cols, &zero, Y, s.rows); break;
                case 1: st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.rows, b, s.cols, &one, W, CUDA_R_32F, s.cols, X, CUDA_R_32F, s.cols, &zero, Y, CUDA_R_32F, s.rows, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP); break;
                case 2: st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.rows, b, s.cols, &one, Wh, CUDA_R_16F, s.cols, Xh, CUDA_R_16F, s.cols, &zero, Y, CUDA_R_32F, s.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); break;
                case 3: st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.rows, b, s.cols, &hone, Wh, CUDA_R_16F, s.cols, Xh, CUDA_R_16F, s.cols, &hzero, Yh, CUDA_R_16F, s.rows, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); break;
                case 4: st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.rows, b, s.cols, &one, W, CUDA_R_32F, s.cols, X, CUDA_R_32F, s.cols, &zero, Y, CUDA_R_32F, s.rows, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); break;
                }
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) best = ms;
            }
            const char* names[] = {"f32", "tf32", "h->f32", "h->h", "fast16"};
            if (st != CUBLAS_STATUS_SUCCESS) printf("  %s: status %d", names[mode], (int)st);
            else printf("  %s %.2f ms %.0f TFLOPS", names[mode], best, flop / best / 1e9);
        }
        printf("\n");
        cudaFree(W); cudaFree(X); cudaFree(Y); cudaFree(Wh); cudaFree(Xh); cudaFree(Yh);
    }
    return 0;
}
