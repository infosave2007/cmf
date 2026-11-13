// h3_flash vs h3_flash2 on random data: max error against each other and
// against an f32 reference on a few rows, and the time of each.
// nvcc -O3 -arch=sm_86 -DHD=128 -DCAUSAL=0 flash_bench.cu -o fb && ./fb n planes
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "/root/h3i/srcrs/crates/cortiq-engine/src/gpu_cuda/h3_flash.cu"

__device__ u16 f2h_sat(float f) { return f2h(fminf(fmaxf(f, -65504.0f), 65504.0f)); }
static u16 h(float f) { __half x = __float2half(f); return *(u16*)&x; }

// the pack of h3.cu: mode 0 queries, 1 keys in lane order, 2 values in lane order
__global__ void pack(const float* src, u16* dst, int n, int ns, int planes, int hds, int hd, int mode) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)planes * ns * hd) return;
    const int d = (int)(idx % hd);
    const size_t t = idx / hd;
    const int row = (int)(t % ns);
    const size_t plane = t / ns;
    const u16 v = (row < n && d < hds) ? f2h_sat(src[(plane * n + row) * hds + d]) : (u16)0;
    if (mode == 2) {
        const int j = d >> 3, gid = d & 7;
        dst[plane * ns * hd + ((size_t)(row >> 4) * 32 + gid * 4 + ((row & 7) >> 1)) * (hd >> 1) + j * 4 + ((row & 15) >> 3) * 2 + (row & 1)] = v;
    } else if (row < n) {
        if (mode == 0) dst[(plane * n + row) * hd + d] = v;
        else {
            const int m = d >> 3, tig = (d & 7) >> 1, e = d & 1;
            dst[(plane * n + row) * hd + (size_t)tig * (hd >> 2) + 2 * m + e] = v;
        }
    }
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 36736, planes = argc > 2 ? atoi(argv[2]) : 16;
    const int ns = (n + 15) / 16 * 16;
    const size_t tot = (size_t)planes * n * HD;
    std::vector<float> q(tot), k(tot), v(tot);
    srand(7);
    for (size_t i = 0; i < tot; i++) {
        q[i] = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
        k[i] = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
        v[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }
    float *dq, *dk, *dv;
    u16 *Q, *K, *V, *O1, *O2;
    cudaMalloc(&dq, tot * 4); cudaMalloc(&dk, tot * 4); cudaMalloc(&dv, tot * 4);
    cudaMalloc(&Q, tot * 2); cudaMalloc(&K, tot * 2); cudaMalloc(&V, (size_t)planes * ns * HD * 2);
    cudaMalloc(&O1, tot * 2); cudaMalloc(&O2, tot * 2);
    cudaMemcpy(dq, q.data(), tot * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), tot * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), tot * 4, cudaMemcpyHostToDevice);
    const size_t pq = (size_t)planes * ns * HD;
    pack<<<(pq + 255) / 256, 256>>>(dq, Q, n, ns, planes, HD, HD, 0);
    pack<<<(pq + 255) / 256, 256>>>(dk, K, n, ns, planes, HD, HD, 1);
    pack<<<(pq + 255) / 256, 256>>>(dv, V, n, ns, planes, HD, HD, 2);
    const float scale = 1.0f / sqrtf((float)HD);
    dim3 grid((n + 63) / 64, planes);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float t1 = 0, t2 = 0;
    for (int it = 0; it < 4; it++) {
        cudaEventRecord(a);
        h3_flash<<<grid, 128>>>(Q, K, V, O1, n, ns, 1, scale, 1);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); if (it) t1 += ms / 3;
        cudaEventRecord(a);
        h3_flash2<<<grid, 128>>>(Q, K, V, O2, n, ns, 1, scale, 1);
        cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (it) t2 += ms / 3;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("cuda error %s\n", cudaGetErrorString(e)); return 1; }
    // planes are laid out as groups (nh = 1): AO [planes * n][HD]
    std::vector<u16> o1(tot), o2(tot);
    cudaMemcpy(o1.data(), O1, tot * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o2.data(), O2, tot * 2, cudaMemcpyDeviceToHost);
    auto hf = [](u16 x) { __half y = *(__half*)&x; return __half2float(y); };
    double d12 = 0, dr1 = 0, dr2 = 0;
    for (size_t i = 0; i < tot; i++) d12 = fmax(d12, fabs(hf(o1[i]) - hf(o2[i])));
    // f32 reference for a few rows of plane 0 and the last plane
    const int rows[] = {0, 1, 17, n / 2, n - 1};
    for (int pl : {0, planes - 1})
        for (int r : rows) {
            const float* qr = &q[((size_t)pl * n + r) * HD];
            std::vector<double> s(n);
            double mx = -1e30;
            for (int j = 0; j < n; j++) {
                const float* kr = &k[((size_t)pl * n + j) * HD];
                double d = 0;
                for (int i = 0; i < HD; i++) d += (double)hf(h(qr[i])) * hf(h(kr[i]));
                s[j] = d * scale;
                mx = fmax(mx, s[j]);
            }
            double z = 0;
            for (int j = 0; j < n; j++) { s[j] = exp(s[j] - mx); z += s[j]; }
            for (int i = 0; i < HD; i++) {
                double acc = 0;
                for (int j = 0; j < n; j++) acc += s[j] * hf(h(v[((size_t)pl * n + j) * HD + i]));
                acc /= z;
                const size_t oi = ((size_t)pl * n + r) * HD + i;
                dr1 = fmax(dr1, fabs(acc - hf(o1[oi])));
                dr2 = fmax(dr2, fabs(acc - hf(o2[oi])));
            }
        }
    const double fl = 4.0 * n * (double)n * HD * planes;
    printf("HD=%d n=%d planes=%d: flash %.2f ms (%.1f TF), flash2 %.2f ms (%.1f TF), max|1-2| %.3g, ref err %.3g / %.3g\n", HD, n,
           planes, t1, fl / t1 / 1e9, t2, fl / t2 / 1e9, d12, dr1, dr2);
    return 0;
}
