//! The converter's `q2tp` (v2) and `q4tp` encoders on the card, a thread a
//! row: the host's arithmetic step for step (`cortiq-cli` `convert.rs`:
//! `encode_q2tp_v2`, `encode_q4tp`), every f32 / f64 operation rounded on
//! its own (no fused multiply-add), so the payloads are the host's bytes.
//! The encoders' searches were the whole cost of rewriting a 2-bit expert
//! stack on a pod of 7.65 CPUs (Hy4-flash: ~220 s a layer of 512 experts).

use super::*;
use std::sync::Mutex;

const SRC: &str = r#"
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ float h2f(u16 h) {
    u32 sign = h >> 15, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign << 31;
        else {
            u32 e = 0, m = mant;
            while ((m & 0x400) == 0) { m <<= 1; e++; }
            m &= 0x3FF;
            bits = (sign << 31) | ((113 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    else bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    return __int_as_float((int)bits);
}

__device__ u16 f2h(float x) {
    u32 bits = (u32)__float_as_int(x);
    u16 sign = (u16)((bits >> 16) & 0x8000);
    int exp = (int)((bits >> 23) & 0xFF);
    u32 mant = bits & 0x7FFFFF;
    if (exp == 0xFF) return sign | 0x7C00 | (mant != 0 ? 0x200 : 0);
    exp -= 127 - 15;
    if (exp >= 0x1F) return sign | 0x7C00;
    if (exp <= 0) {
        if (exp < -10) return sign;
        u32 m = mant | 0x800000;
        u32 shift = (u32)(14 - exp);
        u32 half = m >> shift;
        u32 round = (m >> (shift - 1)) & 1;
        return sign | (u16)(half + round);
    }
    u32 half = ((u32)exp << 10) | (mant >> 13);
    u32 round = (mant >> 12) & 1;
    u32 sticky = (mant & 0xFFF) != 0;
    u32 bump = round & (sticky | (half & 1));
    return sign | (u16)(half + bump);
}

#define F16_TINY __int_as_float(0x38800000)
__device__ float f16_scale(float raw) {
    float v = h2f(f2h(raw));
    return v > F16_TINY ? v : (v != v ? F16_TINY : (v < F16_TINY ? F16_TINY : v));
}

__device__ float det_log2(float x) {
    if (x != x || x < 0.0f) return __int_as_float(0x7fc00000);
    if (x == 0.0f) return __int_as_float(0xff800000);
    if (isinf(x)) return __int_as_float(0x7f800000);
    double d = (double)x;
    u64 bits = (u64)__double_as_longlong(d);
    long long e = (long long)((bits >> 52) & 0x7ff) - 1023;
    double m = __longlong_as_double((long long)((bits & ((1ull << 52) - 1)) | (1023ull << 52)));
    double t = __ddiv_rn(__dsub_rn(m, 1.0), __dadd_rn(m, 1.0));
    double t2 = __dmul_rn(t, t);
    double term = t, sum = 0.0, k = 1.0;
    for (int i = 0; i < 24; i++) {
        sum = __dadd_rn(sum, __ddiv_rn(term, k));
        term = __dmul_rn(term, t2);
        k = __dadd_rn(k, 2.0);
    }
    double ln_m = __dmul_rn(2.0, sum);
    const double LOG2E = __longlong_as_double(0x3FF71547652B82FEll);
    return __double2float_rn(__dadd_rn((double)e, __dmul_rn(ln_m, LOG2E)));
}

__device__ float det_exp2(float y) {
    if (y != y) return y;
    if (y > 128.0f) return __int_as_float(0x7f800000);
    if (y < -160.0f) return 0.0f;
    double yd = (double)y;
    double n = floor(yd);
    double f = __dsub_rn(yd, n);
    const double LN2 = __longlong_as_double(0x3FE62E42FEFA39EFll);
    double z = __dmul_rn(f, LN2);
    double term = 1.0, sum = 1.0;
    for (int k = 1; k <= 24; k++) {
        term = __dmul_rn(term, __ddiv_rn(z, (double)k));
        sum = __dadd_rn(sum, term);
    }
    double scale = __longlong_as_double((long long)(((u64)((long long)n + 1023)) << 52));
    return __double2float_rn(__dmul_rn(sum, scale));
}

// det_q4tp_ladder of params (lo_h, st_h)
__device__ void ladder4(u16 lo_h, u16 st_h, float* t) {
    float lo = h2f(lo_h), st = h2f(st_h);
    float ratio = det_exp2(st);
    t[0] = det_exp2(lo);
    for (int c = 1; c < 32; c++) t[c] = __fmul_rn(t[c - 1], ratio);
}

__device__ float clampf(float x, float a, float b) {
    if (x < a) x = a;
    if (x > b) x = b;
    return x;
}

__device__ float q2_level(float x) {
    const float MAGIC = 12582912.0f;
    float c = clampf(x, -1.0f, 4.0f);
    return clampf(__fsub_rn(__fadd_rn(c, MAGIC), MAGIC), 0.0f, 3.0f);
}

__device__ float q2_group_err(const float* tile, float s) {
    float inv = __fdiv_rn(1.0f, s);
    float acc[8];
    for (int l = 0; l < 8; l++) acc[l] = 0.0f;
    for (int c = 0; c < 4; c++) {
        for (int l = 0; l < 8; l++) {
            float w = tile[c * 8 + l];
            float lv = q2_level(__fadd_rn(__fmul_rn(w, inv), 1.5f));
            float d = __fsub_rn(w, __fmul_rn(__fsub_rn(lv, 1.5f), s));
            acc[l] = __fadd_rn(acc[l], __fmul_rn(__fmul_rn(1.0f, d), d));
        }
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3])),
                     __fadd_rn(__fadd_rn(acc[4], acc[5]), __fadd_rn(acc[6], acc[7])));
}

// q2tp_row_params: rung 1 at lo_h, rung 31 reaching top
__device__ u16 q2_step(u16 lo_h, float top) {
    float lo_r = h2f(lo_h);
    float span = __fsub_rn(top, lo_r);
    if (!(span >= 0.0f)) span = 0.0f;
    u16 st_h = f2h(__fdiv_rn(span, 30.0f));
    for (int i = 0; i < 64; i++) {
        float st = h2f(st_h);
        if (st > 0.0f && __fadd_rn(lo_r, __fmul_rn(30.0f, st)) >= top) break;
        st_h += 1;
    }
    return st_h;
}

__device__ void put_code(u8* codes, int g, int code) {
    int bit = g * 5, b = bit / 8, sh = bit % 8;
    u32 v = (u32)(code & 0x1F);
    codes[b] |= (u8)(v << sh);
    if (sh > 3) codes[b + 1] |= (u8)(v >> (8 - sh));
}

__device__ int get_code(const u8* codes, int g) {
    int bit = g * 5, b = bit / 8, sh = bit % 8;
    u32 lo = codes[b], hi = sh > 3 ? codes[b + 1] : 0;
    return (int)(((lo | (hi << 8)) >> sh) & 0x1F);
}

#define GMAX 512

// encode_q2tp_v2, a thread a row. out: chunks [rows][gpr*8], params
// [rows][4], codes [rows][stride] (codes zeroed by the caller).
extern "C" __global__ void tp_q2(const float* __restrict__ vals, int rows, int cols,
                                 u8* __restrict__ chunks, u8* __restrict__ params, u8* __restrict__ codes)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int gpr = cols / 32, stride = (gpr * 5 + 7) / 8;
    const float* row = vals + (size_t)r * cols;
    float lg[GMAX];
    u8 dead[GMAX], rungs[GMAX], cand[GMAX];
    float lo = __int_as_float(0x7f800000), hi = __int_as_float(0xff800000);
    for (int g = 0; g < gpr; g++) {
        const float* tile = row + g * 32;
        float am = 0.0f;
        for (int j = 0; j < 32; j++) { float a = fabsf(tile[j]); am = am > a ? am : (a != a ? am : a); }
        dead[g] = am == 0.0f;
        float base = __fdiv_rn(am, 1.5f);
        float be = __int_as_float(0x7f800000), bs = base;
        if (!dead[g]) {
            for (int k = 0; k < 31; k++) {
                float mult = __double2float_rn(__ddiv_rn((double)(300 + 25 * k), 1000.0));
                float s = __fmul_rn(base, mult);
                if (!(s > 0.0f)) continue;
                float e = q2_group_err(tile, s);
                if (e < be) { be = e; bs = s; }
            }
        }
        lg[g] = det_log2(f16_scale(bs));
        if (!dead[g]) { lo = fminf(lo, lg[g]); hi = fmaxf(hi, lg[g]); }
    }
    if (!isfinite(lo)) { lo = lg[0]; hi = lo; }
    u16 lo_h = f2h(lo);
    float lo_r = h2f(lo_h);
    const float fr[4] = {1.0f, 0.9f, 0.8f, 0.7f};
    double best_err = 0.0;
    bool have = false;
    u16 best_st = 0;
    float tab[32];
    for (int fi = 0; fi < 4; fi++) {
        u16 st_h = q2_step(lo_h, __fadd_rn(lo_r, __fmul_rn(__fsub_rn(hi, lo_r), fr[fi])));
        float base[32];
        ladder4(lo_h, st_h, base);
        tab[0] = 0.0f;
        for (int c = 1; c < 32; c++) tab[c] = base[c - 1];
        double err = 0.0;
        for (int g = 0; g < gpr; g++) {
            if (dead[g]) { cand[g] = 0; continue; }
            const float* tile = row + g * 32;
            float be = __int_as_float(0x7f800000);
            int bc = 1;
            for (int c = 1; c < 32; c++) {
                float s = tab[c];
                if (!(s > 0.0f)) continue;
                float e = q2_group_err(tile, s);
                if (e < be) { be = e; bc = c; }
            }
            err = __dadd_rn(err, (double)be);
            cand[g] = (u8)bc;
        }
        if (!have || err < best_err) {
            have = true;
            best_err = err;
            best_st = st_h;
            for (int g = 0; g < gpr; g++) rungs[g] = cand[g];
        }
    }
    // pack at the chosen ladder
    {
        float base[32];
        ladder4(lo_h, best_st, base);
        tab[0] = 0.0f;
        for (int c = 1; c < 32; c++) tab[c] = base[c - 1];
    }
    u8* crow = codes + (size_t)r * stride;
    u8* ch = chunks + (size_t)r * gpr * 8;
    for (int g = 0; g < gpr; g++) {
        int c = rungs[g];
        put_code(crow, g, c);
        float inv = tab[c] > 0.0f ? __fdiv_rn(1.0f, tab[c]) : 0.0f;
        const float* tile = row + g * 32;
        for (int k = 0; k < 8; k++) {
            u32 b = 0;
            for (int j = 0; j < 4; j++) b |= ((u32)q2_level(__fadd_rn(__fmul_rn(tile[k * 4 + j], inv), 1.5f))) << (2 * j);
            ch[g * 8 + k] = (u8)b;
        }
    }
    // weight-space debias through lo
    double num = 0.0, den = 0.0;
    for (int g = 0; g < gpr; g++) {
        float s = tab[get_code(crow, g)];
        const float* tile = row + g * 32;
        for (int k = 0; k < 8; k++) {
            u32 byte = ch[g * 8 + k];
            for (int j = 0; j < 4; j++) {
                double w = (double)tile[k * 4 + j];
                double wh = (double)__fmul_rn(__fsub_rn((float)((byte >> (2 * j)) & 3), 1.5f), s);
                num = __dadd_rn(num, __dmul_rn(w, wh));
                den = __dadd_rn(den, __dmul_rn(w, w));
            }
        }
    }
    u16 lo_out = lo_h;
    if (den > 0.0) {
        double b = __ddiv_rn(num, den);
        if (isfinite(b)) {
            if (b < 0.8) b = 0.8;
            if (b > 1.25) b = 1.25;
            float lof = h2f(lo_h);
            float shifted = __double2float_rn(__dsub_rn((double)lof, (double)det_log2(__double2float_rn(b))));
            lo_out = f2h(shifted);
        }
    }
    u8* p = params + (size_t)r * 4;
    p[0] = (u8)(lo_out & 0xFF); p[1] = (u8)(lo_out >> 8);
    p[2] = (u8)(best_st & 0xFF); p[3] = (u8)(best_st >> 8);
}

// encode_q4tp with the rung search (CMF_Q2TP_SEARCH unset), a thread a row.
extern "C" __global__ void tp_q4(const float* __restrict__ vals, int rows, int cols, int search,
                                 u8* __restrict__ nib, u8* __restrict__ params, u8* __restrict__ codes)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int gpr = cols / 32, stride = (gpr * 5 + 7) / 8;
    const float* row = vals + (size_t)r * cols;
    float lg[GMAX];
    u8 dead[GMAX];
    float lo = __int_as_float(0x7f800000), hi = __int_as_float(0xff800000);
    for (int g = 0; g < gpr; g++) {
        const float* tile = row + g * 32;
        float am = 0.0f;
        for (int j = 0; j < 32; j++) { float a = fabsf(tile[j]); am = am > a ? am : (a != a ? am : a); }
        dead[g] = am == 0.0f;
        lg[g] = det_log2(f16_scale(__fdiv_rn(am, 7.0f)));
        if (!dead[g]) { lo = fminf(lo, lg[g]); hi = fmaxf(hi, lg[g]); }
    }
    if (!isfinite(lo)) { lo = lg[0]; hi = lo; }
    u16 lo_h = f2h(lo);
    float lo_r = h2f(lo_h);
    float span = __fsub_rn(hi, lo_r);
    if (!(span >= 0.0f)) span = 0.0f;
    u16 st_h = f2h(__fdiv_rn(span, 31.0f));
    for (int i = 0; i < 64; i++) {
        float st = h2f(st_h);
        if (st > 0.0f && __fadd_rn(lo_r, __fmul_rn(31.0f, st)) >= hi) break;
        st_h += 1;
    }
    u8* p = params + (size_t)r * 4;
    p[0] = (u8)(lo_h & 0xFF); p[1] = (u8)(lo_h >> 8);
    p[2] = (u8)(st_h & 0xFF); p[3] = (u8)(st_h >> 8);
    float st = h2f(st_h);
    float tab[32];
    ladder4(lo_h, st_h, tab);
    u8* crow = codes + (size_t)r * stride;
    u8* nrow = nib + (size_t)r * gpr * 16;
    for (int g = 0; g < gpr; g++) {
        int nominal = 0;
        if (!(dead[g] || st <= 0.0f)) nominal = (int)clampf(rintf(__fdiv_rn(__fsub_rn(lg[g], lo_r), st)), 0.0f, 31.0f);
        const float* tile = row + g * 32;
        int c = nominal;
        if (!(dead[g] || st <= 0.0f || !search)) {
            float be = __int_as_float(0x7f800000);
            int bc = nominal;
            int c0 = nominal >= 2 ? nominal - 2 : 0, c1 = nominal + 1 < 31 ? nominal + 1 : 31;
            for (int cd = c0; cd <= c1; cd++) {
                float sc = tab[cd];
                if (sc <= 0.0f) continue;
                float iv = __fdiv_rn(1.0f, sc);
                float err = 0.0f;
                for (int j = 0; j < 32; j++) {
                    float wv = tile[j];
                    float q = clampf(rintf(__fmul_rn(wv, iv)), -8.0f, 7.0f);
                    float d = __fsub_rn(wv, __fmul_rn(q, sc));
                    err = __fadd_rn(err, __fmul_rn(d, d));
                }
                if (err < be) { be = err; bc = cd; }
            }
            c = bc;
        }
        put_code(crow, g, c);
        float inv = tab[c] > 0.0f ? __fdiv_rn(1.0f, tab[c]) : 0.0f;
        for (int k = 0; k < 16; k++) {
            int q0 = (int)clampf(rintf(__fmul_rn(tile[k * 2], inv)), -8.0f, 7.0f) + 8;
            int q1 = (int)clampf(rintf(__fmul_rn(tile[k * 2 + 1], inv)), -8.0f, 7.0f) + 8;
            nrow[g * 16 + k] = (u8)((q0 & 0x0F) | (q1 << 4));
        }
    }
}
__device__ float q2_group_err_w(const float* tile, const float* hw, float s) {
    float inv = __fdiv_rn(1.0f, s);
    float acc[8];
    for (int l = 0; l < 8; l++) acc[l] = 0.0f;
    for (int c = 0; c < 4; c++) {
        for (int l = 0; l < 8; l++) {
            float w = tile[c * 8 + l];
            float lv = q2_level(__fadd_rn(__fmul_rn(w, inv), 1.5f));
            float d = __fsub_rn(w, __fmul_rn(__fsub_rn(lv, 1.5f), s));
            acc[l] = __fadd_rn(acc[l], __fmul_rn(__fmul_rn(hw[c * 8 + l], d), d));
        }
    }
    return __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3])),
                     __fadd_rn(__fadd_rn(acc[4], acc[5]), __fadd_rn(acc[6], acc[7])));
}

// q2tp_place_row with per-column weights hw: the row params only.
extern "C" __global__ void tp_place(const float* __restrict__ vals, const float* __restrict__ hw, int rows, int cols,
                                    u8* __restrict__ params)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int gpr = cols / 32;
    const float* row = vals + (size_t)r * cols;
    float lg[GMAX];
    u8 dead[GMAX];
    float lo = __int_as_float(0x7f800000), hi = __int_as_float(0xff800000);
    for (int g = 0; g < gpr; g++) {
        const float* tile = row + g * 32;
        const float* h = hw + g * 32;
        float am = 0.0f;
        for (int j = 0; j < 32; j++) { float a = fabsf(tile[j]); am = am > a ? am : (a != a ? am : a); }
        dead[g] = am == 0.0f;
        float base = __fdiv_rn(am, 1.5f);
        float be = __int_as_float(0x7f800000), bs = base;
        if (!dead[g]) {
            for (int k = 0; k < 31; k++) {
                float mult = __double2float_rn(__ddiv_rn((double)(300 + 25 * k), 1000.0));
                float s = __fmul_rn(base, mult);
                if (!(s > 0.0f)) continue;
                float e = q2_group_err_w(tile, h, s);
                if (e < be) { be = e; bs = s; }
            }
        }
        lg[g] = det_log2(f16_scale(bs));
        if (!dead[g]) { lo = fminf(lo, lg[g]); hi = fmaxf(hi, lg[g]); }
    }
    if (!isfinite(lo)) { lo = lg[0]; hi = lo; }
    u16 lo_h = f2h(lo);
    float lo_r = h2f(lo_h);
    const float fr[4] = {1.0f, 0.9f, 0.8f, 0.7f};
    double best_err = 0.0;
    bool have = false;
    u16 best_st = 0;
    for (int fi = 0; fi < 4; fi++) {
        u16 st_h = q2_step(lo_h, __fadd_rn(lo_r, __fmul_rn(__fsub_rn(hi, lo_r), fr[fi])));
        float base[32], tab[32];
        ladder4(lo_h, st_h, base);
        tab[0] = 0.0f;
        for (int c = 1; c < 32; c++) tab[c] = base[c - 1];
        double err = 0.0;
        for (int g = 0; g < gpr; g++) {
            if (dead[g]) continue;
            const float* tile = row + g * 32;
            const float* h = hw + g * 32;
            float be = __int_as_float(0x7f800000);
            for (int c = 1; c < 32; c++) {
                float s = tab[c];
                if (!(s > 0.0f)) continue;
                float e = q2_group_err_w(tile, h, s);
                if (e < be) be = e;
            }
            err = __dadd_rn(err, (double)be);
        }
        if (!have || err < best_err) { have = true; best_err = err; best_st = st_h; }
    }
    u8* p = params + (size_t)r * 4;
    p[0] = (u8)(lo_h & 0xFF); p[1] = (u8)(lo_h >> 8);
    p[2] = (u8)(best_st & 0xFF); p[3] = (u8)(best_st >> 8);
}
// GPTQ sweep, one 32-column group, a thread a row (gptq_dev.rs's WGSL
// q2tp_group / q4tp_group): rung choice, rounding with in-group feedback
// through the fold operator A (n x n, upper), codes into CODES, the words
// into QN, the scaled residuals into E (rows x 32) for the trail GEMM.
__device__ float q2l_round(float x) { return clampf(rintf(clampf(x, -1.0f, 4.0f)), 0.0f, 3.0f); }

extern "C" __global__ void gq_group(float* __restrict__ W, int n, int rows, int grp, int two_bit,
                                    const float* __restrict__ TAB, const float* __restrict__ LOST, const float* __restrict__ HD,
                                    const float* __restrict__ A, u32* __restrict__ QN, u32* __restrict__ CODES, float* __restrict__ E)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int c0 = grp * 32;
    float w[32];
    float amax = 0.0f;
    for (int t = 0; t < 32; t++) { w[t] = W[(size_t)row * n + c0 + t]; amax = fmaxf(amax, fabsf(w[t])); }
    const float* tab = TAB + (size_t)row * 32;
    int best_c = 0;
    if (two_bit) {
        if (amax > 0.0f) {
            float best_e = 3.4e38f;
            best_c = 1;
            for (int cand = 1; cand < 32; cand++) {
                float s_c = tab[cand];
                if (!(s_c > 0.0f)) continue;
                float iv = 1.0f / s_c;
                float err = 0.0f;
                for (int t = 0; t < 32; t++) {
                    float d = w[t] - (q2l_round(w[t] * iv + 1.5f) - 1.5f) * s_c;
                    err = err + HD[c0 + t] * d * d;
                }
                if (err < best_e) { best_e = err; best_c = cand; }
            }
        }
    } else {
        float lo = LOST[row * 2], st = LOST[row * 2 + 1];
        int nominal = 0;
        if (amax > 0.0f && st > 0.0f) {
            float x = (log2f(amax / 7.0f) - lo) / st;
            float ra = copysignf(floorf(fabsf(x) + 0.5f), x);
            nominal = (int)clampf(ra, 0.0f, 31.0f);
        }
        best_c = nominal;
        if (amax > 0.0f) {
            float best_e = 3.4e38f;
            int lo_c = nominal >= 2 ? nominal - 2 : 0, hi_c = nominal + 1 < 31 ? nominal + 1 : 31;
            for (int cand = lo_c; cand <= hi_c; cand++) {
                float s_c = tab[cand];
                if (s_c <= 0.0f) continue;
                float iv = 1.0f / s_c;
                float err = 0.0f;
                for (int t = 0; t < 32; t++) {
                    float q = clampf(rintf(w[t] * iv), -8.0f, 7.0f);
                    float d = w[t] - q * s_c;
                    err = err + HD[c0 + t] * d * d;
                }
                if (err < best_e) { best_e = err; best_c = cand; }
            }
        }
    }
    const float s = tab[best_c];
    CODES[(size_t)row * (n / 32) + grp] = (u32)best_c;
    u32 words[4] = {0u, 0u, 0u, 0u};
    for (int j = 0; j < 32; j++) {
        float q = 0.0f, deq;
        if (two_bit) {
            if (s > 0.0f) q = q2l_round(w[j] / s + 1.5f);
            words[j / 16] |= ((u32)q) << (2 * (j % 16));
            deq = (q - 1.5f) * s;
        } else {
            if (s > 0.0f) q = clampf(rintf(w[j] / s), -8.0f, 7.0f);
            u32 nb = (u32)((int)q + 8) & 15u;
            words[j / 8] |= nb << (4 * (j % 8));
            deq = q * s;
        }
        float dj = A[(size_t)(c0 + j) * n + c0 + j];
        float e = 0.0f;
        if (dj > 0.0f) e = (w[j] - deq) / dj;
        E[(size_t)row * 32 + j] = e;
        if (e != 0.0f) {
            for (int t = j + 1; t < 32; t++) w[t] = w[t] - e * A[(size_t)(c0 + j) * n + c0 + t];
        }
    }
    if (two_bit) {
        size_t base = ((size_t)row * (n / 32) + grp) * 2;
        QN[base] = words[0]; QN[base + 1] = words[1];
    } else {
        size_t base = ((size_t)row * (n / 32) + grp) * 4;
        for (int t = 0; t < 4; t++) QN[base + t] = words[t];
    }
}
"#;

struct TpState {
    f: Option<Result<(cs::CUfunction, cs::CUfunction, cs::CUfunction, cs::CUfunction), String>>,
    /// fold operators uploaded, by the caller's key (the last few)
    folds: Vec<(u64, usize, Buf)>,
    scr: std::collections::HashMap<&'static str, Buf>,
}

// SAFETY: used under the context's state lock.
unsafe impl Send for TpState {}

static STATE: Mutex<Option<TpState>> = Mutex::new(None);

impl TpState {
    fn scr(&mut self, name: &'static str, bytes: usize) -> Res<u64> {
        let bytes = bytes.max(64);
        if !self.scr.get(name).is_some_and(|b| b.bytes >= bytes) {
            self.scr.remove(name);
            self.scr.insert(name, Buf::direct(bytes.next_multiple_of(1 << 20))?);
        }
        Ok(self.scr[name].ptr)
    }
}

/// `CMF_TP_CUDA=0`: the host encodes (read each call: a test compares both).
fn on() -> bool {
    std::env::var("CMF_TP_CUDA").as_deref() != Ok("0")
}

/// A `[rows, cols]` matrix as the converter writes it: `q2tp` v2
/// (`two_bit`) or `q4tp` with the rung search (`search`, the host's
/// `CMF_Q2TP_SEARCH`), on the card. None: no card, a row past 16 384
/// columns, or a refusal (the caller encodes on the host).
pub fn tp_encode(vals: &[f32], rows: usize, cols: usize, two_bit: bool, search: bool) -> Option<Vec<u8>> {
    if !on() || rows == 0 || cols == 0 || cols % 32 != 0 || cols / 32 > 512 || vals.len() < rows * cols {
        return None;
    }
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(|| TpState { f: None, scr: Default::default(), folds: Vec::new() });
    let (fq2, fq4, _, _) = match s.f.get_or_insert_with(|| compile(SRC, vec![], c.arch).and_then(|m| Ok((func(m, "tp_q2")?, func(m, "tp_q4")?, func(m, "tp_place")?, func(m, "gq_group")?)))) {
        Ok(f) => *f,
        Err(e) => {
            decline("tp encode", e);
            return None;
        }
    };
    let gpr = cols / 32;
    let stride = (gpr * 5).div_ceil(8);
    let body = rows * gpr * if two_bit { 8 } else { 16 };
    let total = body + rows * 4 + rows * stride;
    let mut out = vec![0u8; total];
    let mut run = || -> Res<()> {
        let vp = s.scr("tp_v", rows * cols * 4)?;
        let op = s.scr("tp_o", total)?;
        unsafe {
            cu::memcpy_htod_async(vp, &vals[..rows * cols], c.stream).map_err(derr)?;
            // the codes start zeroed (the encoders OR their bits in)
            cu::memcpy_htod_async(op, out.as_slice(), c.stream).map_err(derr)?;
            let (rows_i, cols_i) = (rows as i32, cols as i32);
            let (bp, pp, cp) = (op, op + body as u64, op + (body + rows * 4) as u64);
            let blocks = rows.div_ceil(64) as u32;
            if two_bit {
                c.launch(fq2, (blocks, 1), 64, 0, kargs!(vp, rows_i, cols_i, bp, pp, cp))?;
            } else {
                let srch = search as i32;
                c.launch(fq4, (blocks, 1), 64, 0, kargs!(vp, rows_i, cols_i, srch, bp, pp, cp))?;
            }
            cu::memcpy_dtoh_async(out.as_mut_slice(), op, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run() {
        Ok(()) => Some(out),
        Err(e) => {
            let _ = c.sync();
            decline("tp encode", &e);
            None
        }
    }
}

/// `q2tp_place_row`'s row params (lo, step) of every row against the
/// per-column weights `hw`, on the card: the host's bytes. None: no card or
/// a refusal.
pub fn tp_place_q2(vals: &[f32], hw: &[f32], rows: usize, cols: usize) -> Option<Vec<u8>> {
    if !on() || rows == 0 || cols == 0 || cols % 32 != 0 || cols / 32 > 512 || vals.len() < rows * cols || hw.len() < cols {
        return None;
    }
    let c = ctx()?;
    if c.current().is_err() {
        return None;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(|| TpState { f: None, scr: Default::default(), folds: Vec::new() });
    let fp = match s.f.get_or_insert_with(|| compile(SRC, vec![], c.arch).and_then(|m| Ok((func(m, "tp_q2")?, func(m, "tp_q4")?, func(m, "tp_place")?, func(m, "gq_group")?)))) {
        Ok(f) => f.2,
        Err(e) => {
            decline("tp place", e);
            return None;
        }
    };
    let mut out = vec![0u8; rows * 4];
    let mut run = || -> Res<()> {
        let vp = s.scr("tp_v", rows * cols * 4)?;
        let hp = s.scr("tp_h", cols * 4)?;
        let op = s.scr("tp_o", rows * 4)?;
        unsafe {
            cu::memcpy_htod_async(vp, &vals[..rows * cols], c.stream).map_err(derr)?;
            cu::memcpy_htod_async(hp, &hw[..cols], c.stream).map_err(derr)?;
            let (rows_i, cols_i) = (rows as i32, cols as i32);
            c.launch(fp, (rows.div_ceil(64) as u32, 1), 64, 0, kargs!(vp, hp, rows_i, cols_i, op))?;
            cu::memcpy_dtoh_async(out.as_mut_slice(), op, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run() {
        Ok(()) => Some(out),
        Err(e) => {
            let _ = c.sync();
            decline("tp place", &e);
            None
        }
    }
}

/// The GPTQ sweep of `w` (`rows × n`, dead columns zeroed) on the card
/// against the fold operator `u` (`n × n` upper, row-major; uploaded once a
/// `ukey`): a thread a row a group (`gq_group`, the wgpu sweep's
/// arithmetic), the trail `W[:, c0+32..] -= E·U[c0..c0+32, c0+32..]` on
/// cuBLAS (plain f32). `tab` each row's ladder (q2tp: rung 0 = 0), `lost`
/// its (lo, step) (q4tp), `hd` the group error weights. Writes the code
/// plane (`rows·n/4` or `rows·n/2` bytes) and a rung code a (row, group).
#[allow(clippy::too_many_arguments)]
pub fn gptq_sweep(
    ukey: u64,
    u: &dyn Fn() -> Option<Vec<f32>>,
    n: usize,
    w: &[f32],
    rows: usize,
    tab: &[f32],
    lost: &[f32],
    hd: &[f32],
    two_bit: bool,
    plane: &mut [u8],
    codes: &mut [u32],
) -> bool {
    if !on() || n % 32 != 0 || w.len() != rows * n || tab.len() != rows * 32 || hd.len() != n || codes.len() != rows * (n / 32) {
        return false;
    }
    if plane.len() != rows * n / if two_bit { 4 } else { 2 } || (!two_bit && lost.len() != rows * 2) {
        return false;
    }
    let Some(c) = ctx() else { return false };
    if c.current().is_err() {
        return false;
    }
    let _st = c.st.lock().unwrap();
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(|| TpState { f: None, scr: Default::default(), folds: Vec::new() });
    let fg = match s.f.get_or_insert_with(|| compile(SRC, vec![], c.arch).and_then(|m| Ok((func(m, "tp_q2")?, func(m, "tp_q4")?, func(m, "tp_place")?, func(m, "gq_group")?)))) {
        Ok(f) => f.3,
        Err(e) => {
            decline("gptq sweep", e);
            return false;
        }
    };
    // the fold operator on the card (the last 8 kept: gate and up share one)
    let up = match s.folds.iter().position(|(k, nn, _)| *k == ukey && *nn == n) {
        Some(i) => s.folds[i].2.ptr,
        None => {
            let Some(hu) = u() else { return false };
            if hu.len() != n * n {
                return false;
            }
            let b = match Buf::direct(n * n * 4) {
                Ok(b) => b,
                Err(e) => {
                    decline("gptq sweep", &e);
                    return false;
                }
            };
            if unsafe { cu::memcpy_htod_async(b.ptr, hu.as_slice(), c.stream) }.is_err() {
                return false;
            }
            if s.folds.len() >= 8 {
                s.folds.remove(0);
            }
            let p = b.ptr;
            s.folds.push((ukey, n, b));
            p
        }
    };
    let gpr = n / 32;
    let mut run = || -> Res<()> {
        let wp = s.scr("gs_w", rows * n * 4)?;
        let tp = s.scr("gs_t", rows * 32 * 4)?;
        let lp = s.scr("gs_l", rows * 2 * 4)?;
        let hp = s.scr("gs_h", n * 4)?;
        let qp = s.scr("gs_q", plane.len())?;
        let cp = s.scr("gs_c", rows * gpr * 4)?;
        let ep = s.scr("gs_e", rows * 32 * 4)?;
        let lz = vec![0f32; rows * 2];
        unsafe {
            cu::memcpy_htod_async(wp, w, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(tp, tab, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(lp, if two_bit { lz.as_slice() } else { lost }, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(hp, hd, c.stream).map_err(derr)?;
            let (n_i, rows_i, tb) = (n as i32, rows as i32, two_bit as i32);
            let blocks = rows.div_ceil(128) as u32;
            for k in 0..gpr {
                let k_i = k as i32;
                c.launch(fg, (blocks, 1), 128, 0, kargs!(wp, n_i, rows_i, k_i, tb, tp, lp, hp, up, qp, cp, ep))?;
                let rest = n - (k + 1) * 32;
                if rest > 0 {
                    // column-major: W[:, c0+32..]ᵀ (rest × rows, ld n) -= U-block (rest × 32, ld n) · Eᵀ (32 × rows)
                    let c0 = k * 32;
                    let a = up + ((c0 * n + c0 + 32) * 4) as u64;
                    let cc = wp + ((c0 + 32) * 4) as u64;
                    c.gemm(false, false, rest, rows, 32, -1.0, a, n, ep, 32, 1.0, cc, n, Gemm::F32)?;
                }
            }
            cu::memcpy_dtoh_async(plane, qp, c.stream).map_err(derr)?;
            cu::memcpy_dtoh_async(codes, cp, c.stream).map_err(derr)?;
        }
        c.sync()
    };
    match run() {
        Ok(()) => true,
        Err(e) => {
            let _ = c.sync();
            decline("gptq sweep", &e);
            false
        }
    }
}
