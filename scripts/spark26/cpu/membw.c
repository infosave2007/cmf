// membw.c: the memory-bandwidth probe of the CPU test server (build: gcc -O2 -fopenmp membw.c).
// A 1 GiB float triad (b = 1.5·a + b, three streams) and a 1 GiB read sum, four passes each; the
// thread count comes from OMP_NUM_THREADS. Decode of a 2.16 GB q4tp file is bound by this figure.
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    const size_t n = (size_t)1 << 28; // 256 Mi floats = 1 GiB per array
    float *a = malloc(n * sizeof(float)), *b = malloc(n * sizeof(float));
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    int th = omp_get_max_threads();
    for (int pass = 0; pass < 4; pass++) {
        double t0 = omp_get_wtime();
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) b[i] = a[i] * 1.5f + b[i];
        double dt = omp_get_wtime() - t0;
        printf("triad %d threads: %.1f GB/s\n", th, 3.0 * n * sizeof(float) / dt / 1e9);
    }
    for (int pass = 0; pass < 4; pass++) {
        double s = 0;
        double t0 = omp_get_wtime();
#pragma omp parallel for reduction(+ : s)
        for (size_t i = 0; i < n; i++) s += a[i];
        double dt = omp_get_wtime() - t0;
        printf("read %d threads: %.1f GB/s (sum %.0f)\n", th, n * sizeof(float) / dt / 1e9, s);
    }
    free(a); free(b);
    return 0;
}
