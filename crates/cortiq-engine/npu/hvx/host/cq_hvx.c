// cq_hvx.c - the Android side of the cortiq-npu shim over cortiq_hvx.idl (FastRPC, unsigned PD on the cDSP).
//
// C ABI used by the engine (src/npu.rs):
//   cq_npu_init(dir)            open the session; dir is where libcortiq_hvx_skel.so lives (may be NULL)
//   cq_npu_alloc(bytes)         a DSP-shareable buffer for a matrix (128-byte aligned), or NULL
//   cq_npu_add(w, tiles, gpr, shift)   register a tiled matrix that lives in a cq_npu_alloc buffer
//   cq_npu_run(h, x, y, threads)       y[tiles*32] int32 = the matvec of x[gpr*32] int8 (already in the
//                                      module's input order); returns 0, or -1 with cq_npu_error()
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
    int32_t *y;
} mat_t;

static remote_handle64 g_h = (remote_handle64)-1;
static mat_t g_mats[MAX_MATS];
static int g_n = 0;
static char g_err[256] = "";
static int64_t g_last_usec = 0;
static int g_mapped = -2;
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
    if (fastrpc_mmap && !getenv("CQ_NOMAP")) {
        int fd = rpcmem_to_fd(w);
        g_mapped = fd > 0 ? fastrpc_mmap(CDSP_DOMAIN_ID, fd, w, 0, (size_t)m->wlen, FASTRPC_MAP_FD) : -1;
        if (g_mapped == 0) m->fd = fd;
    }
    if (getenv("CQ_SHARED_XY")) {
        static uint8_t *sx; static int32_t *sy;
        if (!sx) { sx = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 1024 * 32); sy = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, 1 << 20); }
        m->x = sx; m->y = sy;
    } else {
    m->x = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, gpr * 32);
    m->y = (int32_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, tiles * 32 * 4);
    }
    if (!m->x || !m->y) {
        snprintf(g_err, sizeof g_err, "rpcmem_alloc failed");
        return -1;
    }
    return g_n++;
}

int cq_npu_run(int h, const int8_t *x, int32_t *y, int threads) {
    if (h < 0 || h >= g_n) {
        snprintf(g_err, sizeof g_err, "bad handle");
        return -1;
    }
    mat_t *m = &g_mats[h];
    memcpy(m->x, x, (size_t)m->gpr * 32);
    int64 usec = 0;
    int e = cortiq_hvx_q4_matvec(g_h, m->w, m->fd ? 0 : m->wlen, m->fd, m->tiles, m->gpr, m->shift, m->x, m->gpr * 32,
                                 (int *)m->y, m->tiles * 32, threads, &usec);
    if (e) {
        snprintf(g_err, sizeof g_err, "q4_matvec: 0x%x", e);
        return -1;
    }
    g_last_usec = usec;
    memcpy(y, m->y, (size_t)m->tiles * 32 * 4);
    return 0;
}
