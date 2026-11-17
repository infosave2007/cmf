// cortiq_hvx_imp.c - the DSP side of cortiq_hvx.idl: a 4-bit matvec in integer HVX.
//
// Rows live in the vector's lanes: one 128-byte vector carries 4 code bytes of each of 32 rows, and
// vrmpy(codes, four int8 inputs) leaves each row's partial sum in its own word. There is no horizontal
// reduction and no float; a group's 32 inputs take 8 such products, then one fractional multiply by the
// group's per-row scale.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "cortiq_hvx.h"
#include "qurt_hvx.h"
#include "qurt_thread.h"

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#define REC 640 /* bytes per (tile, group): 4 code vectors + 1 scale vector */
#ifndef FETCH_AHEAD
#define FETCH_AHEAD 6
#endif

typedef struct {
    int power;
} ctx_t;

// The DSP's clocks and bus at their highest while the session lives, HVX powered.
static int vote(void *ctx) {
    int kind = 0;
    HAP_power_request_t r;
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_DCVS_v3;
    r.dcvs_v3.set_dcvs_enable = TRUE;
    r.dcvs_v3.dcvs_enable = TRUE;
    r.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    r.dcvs_v3.set_bus_params = TRUE;
    r.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.set_core_params = TRUE;
    r.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.set_sleep_disable = TRUE;
    r.dcvs_v3.sleep_disable = TRUE;
    r.dcvs_v3.set_latency = TRUE;
    r.dcvs_v3.latency = 100;
    if (HAP_power_set(ctx, &r) == 0) kind = 3;
    if (!kind) {
        memset(&r, 0, sizeof r);
        r.type = HAP_power_set_DCVS_v2;
        r.dcvs_v2.dcvs_enable = TRUE;
        r.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
        r.dcvs_v2.set_latency = TRUE;
        r.dcvs_v2.latency = 100;
        r.dcvs_v2.set_dcvs_params = TRUE;
        r.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_MAX;
        r.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_MAX;
        r.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_MAX;
        if (HAP_power_set(ctx, &r) == 0) kind = 2;
    }
    memset(&r, 0, sizeof r);
    r.type = HAP_power_set_HVX;
    r.hvx.power_up = TRUE;
    if (HAP_power_set(ctx, &r) != 0) kind += 10;
    return kind;
}

int cortiq_hvx_open(const char *uri, remote_handle64 *h) {
    ctx_t *c = (ctx_t *)calloc(1, sizeof *c);
    if (!c) return AEE_ENOMEMORY;
    c->power = vote(c);
    *h = (remote_handle64)(uintptr_t)c;
    return 0;
}

int cortiq_hvx_close(remote_handle64 h) {
    ctx_t *c = (ctx_t *)(uintptr_t)h;
    if (c) {
        HAP_power_request_t r;
        memset(&r, 0, sizeof r);
        r.type = HAP_power_set_HVX;
        r.hvx.power_up = FALSE;
        HAP_power_set(c, &r);
        free(c);
    }
    return 0;
}

int cortiq_hvx_ping(remote_handle64 h, int x, int *y, int *hvx, int *power) {
    ctx_t *c = (ctx_t *)(uintptr_t)h;
    *y = x + 1;
    *power = c ? c->power : -1;
    int lock = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (lock == 0) {
        // 32 lanes of vrmpy(bytes 1 2 3 4, bytes 1 1 1 1) = 10 each.
        HVX_Vector v = Q6_Vw_vrmpy_VubRb(Q6_V_vsplat_R(0x04030201), 0x01010101);
        int32_t out[32] __attribute__((aligned(128)));
        *(HVX_Vector *)out = v;
        int s = 0;
        for (int i = 0; i < 32; i++) s += out[i];
        if (s != 320) lock = 1000 + s;
        qurt_hvx_unlock();
    }
    *hvx = lock;
    return 0;
}

typedef struct {
    const uint8_t *w;
    const int32_t *xw;   // per group 8 words: (even, odd) per j
    const int32_t *corr; // per group 8 * sum(x)
    int32_t *y;
    int gpr, shift, t0, t1, tiles;
    int ok;
} job_t;

static void run_tiles(job_t *j) {
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) {
        j->ok = 0;
        return;
    }
    const HVX_Vector m = Q6_V_vsplat_R(0x0f0f0f0f);
    const int gpr = j->gpr, shift = j->shift;
    const HVX_Vector *end = (const HVX_Vector *)(j->w + (size_t)j->tiles * gpr * REC);
    for (int t = j->t0; t < j->t1; t++) {
        const HVX_Vector *p = (const HVX_Vector *)(j->w + (size_t)t * gpr * REC);
        HVX_Vector acc_y = Q6_V_vzero();
        const int32_t *xw = j->xw;
        for (int g = 0; g < gpr; g++, xw += 8, p += 5) {
            // The weights stream from DDR and the DSP fetches nothing on its own: ask for the record
            // FETCH_AHEAD groups on (5 lines of 128 bytes) while this one is multiplied.
            // Never past the matrix: a fetch of an unmapped page is a fault in a user PD, not a no-op.
            if (p + 5 * (FETCH_AHEAD + 1) <= end) Q6_l2fetch_AR((void *)(p + 5 * FETCH_AHEAD), (128 << 16) | (128 << 8) | 5);
            HVX_Vector acc = Q6_V_vsplat_R(-j->corr[g]);
            for (int k = 0; k < 4; k++) {
                HVX_Vector v = p[k];
                HVX_Vector lo = Q6_V_vand_VV(v, m);
                HVX_Vector hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v, 4), m);
                acc = Q6_Vw_vrmpyacc_VwVubRb(acc, lo, xw[2 * k]);
                acc = Q6_Vw_vrmpyacc_VwVubRb(acc, hi, xw[2 * k + 1]);
            }
            acc_y = Q6_Vw_vadd_VwVw(acc_y, Q6_Vw_vmpyo_VwVh_s1_rnd_sat(Q6_Vw_vasl_VwR(acc, shift), p[4]));
        }
        *(HVX_Vector *)(j->y + (size_t)t * 32) = acc_y;
    }
    qurt_hvx_unlock();
    j->ok = 1;
}

static void worker(void *arg) {
    run_tiles((job_t *)arg);
    qurt_thread_exit(0);
}

static int q4_matvec(const unsigned char *w, int tiles, int gpr, int shift, const unsigned char *x, int xLen, int *y,
                     int yLen, int threads, int64 *usec);

#define MAX_THREADS 4
#define STACK 8192

int cortiq_hvx_q4_matvec(remote_handle64 h, const unsigned char *w, int wLen, int wfd, int tiles, int gpr, int shift,
                         const unsigned char *x, int xLen, int *y, int yLen, int threads, int64 *usec) {
    if (tiles <= 0 || gpr <= 0 || gpr > 1024 || shift < 0 || shift > 12) return AEE_EBADPARM;
    int held = 0;
    if (wfd > 0) {
        void *va = NULL;
        uint64 pa = 0;
        if (HAP_mmap_get(wfd, &va, &pa) != 0 || !va) return AEE_ENOSUCHMAP;
        w = (const unsigned char *)va;
        held = 1;
    } else if ((int64_t)wLen < (int64_t)tiles * gpr * REC) {
        return AEE_EBADPARM;
    }
    int rc = q4_matvec(w, tiles, gpr, shift, x, xLen, y, yLen, threads, usec);
    if (held) HAP_mmap_put(wfd);
    return rc;
}

static int q4_matvec(const unsigned char *w, int tiles, int gpr, int shift, const unsigned char *x, int xLen, int *y,
                     int yLen, int threads, int64 *usec) {
    if (xLen < gpr * 32 || yLen < tiles * 32) return AEE_EBADPARM;
    if (((uintptr_t)w & 127) || ((uintptr_t)y & 127) || ((uintptr_t)x & 3)) return AEE_EBADPARM;
    uint64_t t_start = HAP_perf_get_time_us();
    int32_t corr[1024];
    const int8_t *xs = (const int8_t *)x;
    for (int g = 0; g < gpr; g++) {
        int s = 0;
        for (int i = 0; i < 32; i++) s += xs[g * 32 + i];
        corr[g] = 8 * s;
    }
    if (threads < 1) threads = 1;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    if (threads > tiles) threads = tiles;
    job_t jobs[MAX_THREADS];
    qurt_thread_t tid[MAX_THREADS];
    void *stacks[MAX_THREADS] = {0};
    int started[MAX_THREADS] = {0};
    for (int i = 0; i < threads; i++) {
        jobs[i] = (job_t){.w = w, .xw = (const int32_t *)x, .corr = corr, .y = (int32_t *)y, .gpr = gpr, .shift = shift, .tiles = tiles,
                          .t0 = (int)((int64_t)tiles * i / threads), .t1 = (int)((int64_t)tiles * (i + 1) / threads), .ok = 0};
    }
    for (int i = 1; i < threads; i++) {
        stacks[i] = malloc(STACK);
        if (!stacks[i]) continue;
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, "cq_hvx");
        qurt_thread_attr_set_stack_addr(&attr, stacks[i]);
        qurt_thread_attr_set_stack_size(&attr, STACK);
        qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()));
        if (qurt_thread_create(&tid[i], &attr, worker, &jobs[i]) == 0) started[i] = 1;
    }
    run_tiles(&jobs[0]);
    int ok = jobs[0].ok;
    for (int i = 1; i < threads; i++) {
        if (started[i]) {
            int st;
            qurt_thread_join(tid[i], &st);
        } else {
            run_tiles(&jobs[i]); // could not start a thread: do its share here
        }
        ok &= jobs[i].ok;
        free(stacks[i]);
    }
    *usec = (int64)(HAP_perf_get_time_us() - t_start);
    return ok ? 0 : AEE_EFAILED;
}
