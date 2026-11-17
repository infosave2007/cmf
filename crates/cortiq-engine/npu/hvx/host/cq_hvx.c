// cq_hvx.c - the Android side of the cortiq-npu shim over cortiq_hvx.idl (FastRPC, unsigned PD on the cDSP).
//
// C ABI used by the engine (src/npu.rs):
//   cq_npu_init(dir)            open the session; dir is where libcortiq_hvx_skel.so lives (may be NULL)
//   cq_npu_alloc(bytes)         a DSP-shareable buffer for a matrix (128-byte aligned), or NULL
//   cq_npu_add(w, tiles, gpr, shift)   register a tiled matrix that lives in a cq_npu_alloc buffer
//   cq_npu_run(h, h2, x, y, threads)   y[tiles*32] int32 = the matvec of x[gpr*32] int8 (already in the
//                                      module's input order); h2 >= 0: a second matrix, y twice as long
//   cq_npu_gemm(h, n, x, y, threads)   the same for n vectors: x[n][gpr*32], y[n][tiles*32]
//                                      all return 0, or -1 with cq_npu_error()
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cortiq_hvx.h"
#include "remote.h"
#include "rpcmem.h"

#pragma weak remote_session_control
#pragma weak fastrpc_mmap

#define MAX_MATS 512
typedef struct {
    uint8_t *w;
    int wlen, tiles, gpr, shift, fd;
    uint8_t *x;
} mat_t;

static remote_handle64 g_h = (remote_handle64)-1;
static mat_t g_mats[MAX_MATS];
static int g_n = 0;
static char g_err[256] = "";
static int64_t g_last_usec = 0;
static int g_mapped = -2;
static int32_t *g_by; static size_t g_by_len;
int cq_npu_last_map(void) { return g_mapped; }

const char *cq_npu_error(void) { return g_err; }
long long cq_npu_last_dsp_usec(void) { return g_last_usec; }

int cq_npu_init(const char *dir) {
    if (g_h != (remote_handle64)-1) return 0;
    (void)dir;
    if (remote_session_control) {
        struct remote_rpc_control_unsigned_module data;
        data.domain = CDSP_DOMAIN_ID;
        data.enable = 1;
        int e = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof data);
        if (e) {
            snprintf(g_err, sizeof g_err, "unsigned PD refused: 0x%x", e);
            return -1;
        }
    } else {
        snprintf(g_err, sizeof g_err, "remote_session_control missing (old FastRPC)");
        return -1;
    }
    int e = cortiq_hvx_open(cortiq_hvx_URI "&_dom=cdsp", &g_h);
    if (e) {
        snprintf(g_err, sizeof g_err, "cortiq_hvx_open: 0x%x", e);
        g_h = (remote_handle64)-1;
        return -1;
    }
    int y = 0, hvx = -1, power = -1;
    e = cortiq_hvx_ping(g_h, 41, &y, &hvx, &power);
    if (e || y != 42 || hvx != 0) {
        snprintf(g_err, sizeof g_err, "ping: err 0x%x y %d hvx %d power %d", e, y, hvx, power);
        cortiq_hvx_close(g_h);
        g_h = (remote_handle64)-1;
        return -1;
    }
    snprintf(g_err, sizeof g_err, "ok: power vote %d", power);
    return 0;
}

void *cq_npu_alloc(size_t bytes) {
    return rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)bytes);
}

void cq_npu_free(void *p) {
    if (p) rpcmem_free(p);
}

int cq_npu_add(void *w, int tiles, int gpr, int shift) {
    if (g_h == (remote_handle64)-1) {
        snprintf(g_err, sizeof g_err, "not initialised");
        return -1;
    }
    if (g_n >= MAX_MATS) {
        snprintf(g_err, sizeof g_err, "too many matrices");
        return -1;
    }
    mat_t *m = &g_mats[g_n];
    m->w = (uint8_t *)w;
    m->tiles = tiles;
    m->gpr = gpr;
    m->shift = shift;
    m->wlen = tiles * gpr * 640;
    // Keep the matrix mapped on the DSP for the session: without it FastRPC maps and unmaps the whole
    // buffer around every call. Older DSPs refuse (the call still works, just with that cost).
    m->fd = 0;
    if (fastrpc_mmap) {
        int fd = rpcmem_to_fd(w);
        g_mapped = fd > 0 ? fastrpc_mmap(CDSP_DOMAIN_ID, fd, w, 0, (size_t)m->wlen, FASTRPC_MAP_FD) : -1;
        if (g_mapped == 0) m->fd = fd;
    }
    m->x = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, gpr * 32);
    if (!m->x) {
        snprintf(g_err, sizeof g_err, "rpcmem_alloc failed");
        return -1;
    }
    return g_n++;
}

// y = the matvec of x on matrix h; with h2 >= 0 (a second matrix of the same shape, both mapped) y holds
// h's result and then h2's, from one call.
int cq_npu_run(int h, int h2, const int8_t *x, int32_t *y, int threads) {
    if (h < 0 || h >= g_n || h2 >= g_n) {
        snprintf(g_err, sizeof g_err, "bad handle");
        return -1;
    }
    mat_t *m = &g_mats[h];
    mat_t *m2 = h2 >= 0 ? &g_mats[h2] : NULL;
    int ny = m->tiles * 32 * (m2 ? 2 : 1);
    if (m2 && (!m->fd || !m2->fd || m2->tiles != m->tiles || m2->gpr != m->gpr || m2->shift != m->shift)) {
        snprintf(g_err, sizeof g_err, "pair needs two mapped matrices of one shape");
        return -1;
    }
    size_t yl = (size_t)ny * 4;
    if (yl > g_by_len) { if (g_by) rpcmem_free(g_by); g_by = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)yl); g_by_len = g_by ? yl : 0; }
    if (!g_by) {
        snprintf(g_err, sizeof g_err, "rpcmem_alloc failed");
        return -1;
    }
    memcpy(m->x, x, (size_t)m->gpr * 32);
    int64 usec = 0;
    int e = cortiq_hvx_q4_matvec(g_h, m->w, m->fd ? 0 : m->wlen, m->fd, m2 ? m2->fd : 0, m->tiles, m->gpr, m->shift, m->x,
                                 m->gpr * 32, (int *)g_by, ny, threads, &usec);
    if (e) {
        snprintf(g_err, sizeof g_err, "q4_matvec: 0x%x", e);
        return -1;
    }
    g_last_usec = usec;
    memcpy(y, g_by, yl);
    return 0;
}

static uint8_t *g_bx; static size_t g_bx_len;
// y[n][tiles*32] = the product of n vectors x[n][gpr*32] (a prompt chunk). The scratch buffers are shared
// across matrices and grown as needed.
int cq_npu_gemm(int h, int n, const int8_t *x, int32_t *y, int threads) {
    if (h < 0 || h >= g_n || n <= 0) {
        snprintf(g_err, sizeof g_err, "bad handle");
        return -1;
    }
    mat_t *m = &g_mats[h];
    size_t xl = (size_t)n * m->gpr * 32, yl = (size_t)n * m->tiles * 32 * 4;
    if (xl > g_bx_len) { if (g_bx) rpcmem_free(g_bx); g_bx = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)xl); g_bx_len = g_bx ? xl : 0; }
    if (yl > g_by_len) { if (g_by) rpcmem_free(g_by); g_by = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)yl); g_by_len = g_by ? yl : 0; }
    if (!g_bx || !g_by) {
        snprintf(g_err, sizeof g_err, "rpcmem_alloc failed");
        return -1;
    }
    memcpy(g_bx, x, xl);
    int64 usec = 0;
    int e = cortiq_hvx_q4_gemm(g_h, m->w, m->fd ? 0 : m->wlen, m->fd, m->tiles, m->gpr, m->shift, n, g_bx, (int)xl,
                               (int *)g_by, (int)(yl / 4), threads, &usec);
    if (e) {
        snprintf(g_err, sizeof g_err, "q4_gemm: 0x%x", e);
        return -1;
    }
    g_last_usec = usec;
    memcpy(y, g_by, yl);
    return 0;
}
