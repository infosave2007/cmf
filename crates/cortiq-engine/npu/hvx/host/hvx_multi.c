// hvx_multi.c - several matrices, calls from another thread: what the engine does.
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int cq_npu_init(const char *);
void *cq_npu_alloc(size_t);
int cq_npu_add(void *, int, int, int);
int cq_npu_run(int, int, const int8_t *, int32_t *, int);
const char *cq_npu_error(void);
int cq_npu_last_map(void);
static int hs[8], n, tiles = 192, gpr = 64;
static void *runner(void *arg) {
    int8_t *x = calloc(1, gpr * 32); int32_t *y = malloc(tiles * 32 * 4);
    for (int i = 0; i < n; i++) { int e = cq_npu_run(hs[i], -1, x, y, 1); printf("  thread run lane %d -> %d %s\n", i, e, e ? cq_npu_error() : ""); }
    return NULL;
}
int main(int argc, char **argv) {
    n = argc > 1 ? atoi(argv[1]) : 3; if (argc > 2) tiles = atoi(argv[2]);
    if (cq_npu_init(NULL)) { printf("init: %s\n", cq_npu_error()); return 1; }
    for (int i = 0; i < n; i++) {
        void *w = cq_npu_alloc((size_t)tiles * gpr * 640); memset(w, 0, (size_t)tiles * gpr * 640);
        hs[i] = cq_npu_add(w, tiles, gpr, 8); printf("add %d -> handle %d, map 0x%x\n", i, hs[i], cq_npu_last_map());
    }
    int8_t *x = calloc(1, gpr * 32); int32_t *y = malloc(tiles * 32 * 4);
    for (int i = 0; i < n; i++) { int e = cq_npu_run(hs[i], -1, x, y, 1); printf("main run lane %d -> %d %s\n", i, e, e ? cq_npu_error() : ""); }
    if (n >= 2) { int32_t *y2 = malloc(tiles * 32 * 8); int e = cq_npu_run(hs[0], hs[1], x, y2, 1); printf("pair run -> %d %s\n", e, e ? cq_npu_error() : ""); }
    pthread_t t; pthread_create(&t, NULL, runner, NULL); pthread_join(t, NULL);
    return 0;
}
