// hvx_test.c - accuracy and speed of the module against a reference computed here.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
int cq_npu_init(const char *);
void *cq_npu_alloc(size_t);
int cq_npu_add(void *, int, int, int);
int cq_npu_run(int, int, const int8_t *, int32_t *, int);
int cq_npu_gemm(int, int, const int8_t *, int32_t *, int);
const char *cq_npu_error(void);
long long cq_npu_last_dsp_usec(void);
int cq_npu_last_map(void);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t s = 88172645463325252ull;
static uint64_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
int main(int argc, char **argv) {
    int rows = argc > 1 ? atoi(argv[1]) : 7680, cols = argc > 2 ? atoi(argv[2]) : 2048, shift = argc > 3 ? atoi(argv[3]) : 8;
    int tiles = (rows + 31) / 32, gpr = cols / 32;
    if (cq_npu_init(NULL)) { printf("init: %s\n", cq_npu_error()); return 1; }
    printf("session: %s | rows %d cols %d (tiles %d, gpr %d), %.1f MB on the DSP\n", cq_npu_error(), rows, cols, tiles, gpr, tiles * (double)gpr * 640 / 1e6);
    uint8_t *w = cq_npu_alloc((size_t)tiles * gpr * 640);
    if (!w) { printf("alloc failed\n"); return 1; }
    // the plain form: nib[r][c] in 0..15, q15[r][g]
    uint8_t *nib = malloc((size_t)tiles * 32 * cols); int16_t *q = malloc((size_t)tiles * 32 * gpr * 2);
    for (size_t i = 0; i < (size_t)tiles * 32 * cols; i++) nib[i] = rnd() & 15;
    for (size_t i = 0; i < (size_t)tiles * 32 * gpr; i++) q[i] = (int16_t)(rnd() % 32768);
    for (int t = 0; t < tiles; t++) for (int g = 0; g < gpr; g++) {
        uint8_t *rec = w + ((size_t)t * gpr + g) * 640;
        for (int j = 0; j < 4; j++) for (int r = 0; r < 32; r++) for (int b = 0; b < 4; b++) {
            const uint8_t *row = nib + ((size_t)t * 32 + r) * cols + g * 32;
            rec[j * 128 + r * 4 + b] = row[8 * j + 2 * b] | (row[8 * j + 2 * b + 1] << 4);
        }
        int32_t *sc = (int32_t *)(rec + 512);
        for (int r = 0; r < 32; r++) sc[r] = (int32_t)q[((size_t)t * 32 + r) * gpr + g] << 16;
    }
    int8_t *x = malloc(cols), *xo = malloc(cols); int32_t *y = malloc((size_t)tiles * 32 * 4);
    for (int c = 0; c < cols; c++) x[c] = (int8_t)((int)(rnd() % 255) - 127);
    for (int g = 0; g < gpr; g++) for (int j = 0; j < 4; j++) for (int b = 0; b < 4; b++) {
        xo[g * 32 + j * 8 + b] = x[g * 32 + 8 * j + 2 * b];
        xo[g * 32 + j * 8 + 4 + b] = x[g * 32 + 8 * j + 2 * b + 1];
    }
    int h = cq_npu_add(w, tiles, gpr, shift);
    if (h < 0) { printf("add: %s\n", cq_npu_error()); return 1; }
    printf("persistent map: %s (0x%x)\n", cq_npu_last_map() == 0 ? "yes" : "no", cq_npu_last_map());
    if (cq_npu_run(h, -1, xo, y, 1)) { printf("run: %s\n", cq_npu_error()); return 1; }
    double num = 0, den = 0; long maxdiff = 0;
    for (int r = 0; r < tiles * 32; r++) {
        double ref = 0;
        for (int g = 0; g < gpr; g++) {
            long d = 0; const uint8_t *row = nib + (size_t)r * cols + g * 32;
            for (int i = 0; i < 32; i++) d += ((int)row[i] - 8) * x[g * 32 + i];
            ref += (double)(d << shift) * q[(size_t)r * gpr + g] / 32768.0;
        }
        double df = y[r] - ref; num += df * df; den += ref * ref; if (labs((long)df) > maxdiff) maxdiff = labs((long)df);
    }
    printf("relative L2 error %.2e | largest |diff| %ld counts (of up to %d groups' rounding)\n", sqrt(num / den), maxdiff, gpr);
    for (int th = 1; th <= 4; th++) {
        double best = 1e9; long long dsp = 0;
        for (int it = 0; it < 3; it++) { double t = now(); for (int i = 0; i < 20; i++) cq_npu_run(h, -1, xo, y, th); double d = (now() - t) / 20; if (d < best) { best = d; dsp = cq_npu_last_dsp_usec(); } }
        printf("threads %d: %.2f ms per matvec end to end (DSP %lld us) | %.1f G weights/s\n", th, best * 1e3, dsp, tiles * 32.0 * cols / best / 1e9);
    }
    // the batch product: n vectors; vector b is x rotated by b groups, so its answer is checkable
    int nrows = tiles * 32;
    for (int n = 8; n <= 64; n *= 2) {
        int8_t *bx = malloc((size_t)n * cols); int32_t *by = malloc((size_t)n * nrows * 4);
        for (int b = 0; b < n; b++) for (int c = 0; c < cols; c++) bx[(size_t)b * cols + c] = xo[(c + 32 * b) % cols];
        if (cq_npu_gemm(h, n, bx, by, 1)) { printf("gemm: %s\n", cq_npu_error()); return 1; }
        int bad = 0; for (int r = 0; r < nrows; r++) if (by[r] != y[r]) bad++;   // vector 0 is the matvec's x
        long sum_last = 0; for (int r = 0; r < nrows; r++) sum_last += by[(size_t)(n - 1) * nrows + r];
        for (int th = 1; th <= 4; th++) {
            double best = 1e9; long long dsp = 0;
            for (int it = 0; it < 3; it++) { double t = now(); cq_npu_gemm(h, n, bx, by, th); double d = now() - t; if (d < best) { best = d; dsp = cq_npu_last_dsp_usec(); } }
            printf("gemm n=%2d threads %d: %.2f ms (DSP %lld us) = %.3f ms per vector | vector 0 mismatches %d\n", n, th, best * 1e3, dsp, best * 1e3 / n, bad);
        }
        (void)sum_last; free(bx); free(by);
    }
    return 0;
}
