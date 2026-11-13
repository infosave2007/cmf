// vramfill MIB: hold card memory until only MIB stay free (an emulated smaller card), then sleep until killed.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: vramfill FREE_MIB\n"); return 2; }
    const size_t keep = (size_t)std::atoll(argv[1]) << 20;
    size_t f = 0, t = 0;
    cudaFree(0);
    cudaMemGetInfo(&f, &t);
    size_t held = 0;
    for (size_t step = (size_t)1 << 30; step >= ((size_t)16 << 20); ) {
        cudaMemGetInfo(&f, &t);
        if (f <= keep) break;
        size_t want = f - keep < step ? (step >> 1) : step;
        if (f - keep < want) { step >>= 1; continue; }
        void* p = nullptr;
        if (cudaMalloc(&p, want) != cudaSuccess) { step >>= 1; continue; }
        held += want;
    }
    cudaMemGetInfo(&f, &t);
    std::printf("vramfill: holding %zu MiB, %zu MiB free of %zu\n", held >> 20, f >> 20, t >> 20);
    std::fflush(stdout);
    for (;;) pause();
}
