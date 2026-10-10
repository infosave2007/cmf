#include <metal_stdlib>
using namespace metal;

// q8_2f routed experts on the Metal graphs (Mellum2.1 q8_2f).
//
// A q8_2f tensor in the file is [rows×cols i8][rows f16 row scales]
// [cols f16 input field]. The dense graph decodes the two f16 fields into
// cached f32 buffers per tensor; an expert job only knows its weight's
// absolute offset (the select kernel writes it), so these kernels read both
// fields straight from the blob behind the payload. f16 → f32 is exact, so
// x·col and the row scale are the same floats the dense kernels (and the
// CPU's `prescale`) use.
//
// The input field is per tensor: every job multiplies ITS OWN field into the
// shared activations while it reads them (the 0.8.14 per-op bug was one
// staged x·col reused across jobs — here nothing is staged at all).

// The windowed arena's `locate` (`moe_job_base` of the main library).
inline device const uchar* mq8_base(ulong jb, device const uchar* q,
                                   device const uchar* qw1, device const uchar* qw2,
                                   device const uchar* qw3, ulong wstride) {
    if (wstride == 0ul) return q + jb;
    uint wi = (uint)(jb / wstride);
    ulong rel = jb - (ulong)wi * wstride;
    return ((wi == 0u) ? q : (wi == 1u) ? qw1 : (wi == 2u) ? qw2 : qw3) + rel;
}

// Job-batched q8_2f matvec, four output rows per simdgroup: job j reads its
// weight at bases[j], its activations at x + j·xstride (float4 units; 0 =
// the shared input of gate/up) and writes y[j·rows + r]. The per-row math is
// `q8f_matvec_r4`'s, accumulation order included.
kernel void q8f_jobs_r4(
    device const uchar*  q       [[buffer(0)]],
    device const float4* x       [[buffer(1)]],
    device float*        y       [[buffer(2)]],
    constant uint&       cols4   [[buffer(3)]],
    constant uint&       rows    [[buffer(4)]],
    device const ulong*  bases   [[buffer(5)]],
    constant uint&       tg_per  [[buffer(6)]],
    constant uint&       xstride [[buffer(7)]],
    device const uchar*  qw1     [[buffer(8)]],
    device const uchar*  qw2     [[buffer(9)]],
    device const uchar*  qw3     [[buffer(10)]],
    constant ulong&      wstride [[buffer(11)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint j   = tgpos / tg_per;
    uint tgl = tgpos - j * tg_per;
    uint r0 = (tgl * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    device const uchar* qj = mq8_base(bases[j], q, qw1, qw2, qw3, wstride);
    device const char4* qq = (device const char4*)qj;
    ulong qlen = (ulong)rows * (ulong)cols4 * 4ul;
    device const half* rs = (device const half*)(qj + qlen);
    device const packed_half4* col = (device const packed_half4*)(qj + qlen + (ulong)rows * 2ul);
    device const float4* xs = x + (ulong)j * (ulong)xstride;
    device float* yj = y + (ulong)j * (ulong)rows;
    float4 a0 = float4(0.0f), a1 = float4(0.0f);
    float4 a2 = float4(0.0f), a3 = float4(0.0f);
    uint i = lane;
    for (; i + 96u < cols4; i += 128u) {
        float4 x0 = xs[i] * float4(col[i]);
        float4 x1 = xs[i + 32u] * float4(col[i + 32u]);
        float4 x2 = xs[i + 64u] * float4(col[i + 64u]);
        float4 x3 = xs[i + 96u] * float4(col[i + 96u]);
        for (uint ri = 0u; ri < nr; ++ri) {
            ulong base = (ulong)(r0 + ri) * cols4;
            float4 aa;
            aa.x = dot(float4(qq[base + i]), x0);
            aa.y = dot(float4(qq[base + i + 32u]), x1);
            aa.z = dot(float4(qq[base + i + 64u]), x2);
            aa.w = dot(float4(qq[base + i + 96u]), x3);
            if (ri == 0u) a0 += aa;
            else if (ri == 1u) a1 += aa;
            else if (ri == 2u) a2 += aa;
            else a3 += aa;
        }
    }
    for (; i < cols4; i += 32u) {
        float4 xv = xs[i] * float4(col[i]);
        for (uint ri = 0u; ri < nr; ++ri) {
            float v = dot(float4(qq[(ulong)(r0 + ri) * cols4 + i]), xv);
            if (ri == 0u) a0.x += v;
            else if (ri == 1u) a1.x += v;
            else if (ri == 2u) a2.x += v;
            else a3.x += v;
        }
    }
    float t0 = simd_sum(a0.x + a0.y + a0.z + a0.w);
    float t1 = simd_sum(a1.x + a1.y + a1.z + a1.w);
    float t2 = simd_sum(a2.x + a2.y + a2.z + a2.w);
    float t3 = simd_sum(a3.x + a3.y + a3.z + a3.w);
    if (lane == 0u) {
        yj[r0] = t0 * (float)rs[r0];
        if (nr > 1u) yj[r0 + 1u] = t1 * (float)rs[r0 + 1u];
        if (nr > 2u) yj[r0 + 2u] = t2 * (float)rs[r0 + 2u];
        if (nr > 3u) yj[r0 + 3u] = t3 * (float)rs[r0 + 3u];
    }
}

// The routed experts' gate|up|SiLU in ONE dispatch: job j's gate at
// bases[j], its up at bases[ne + j] (the select kernel's table), both over
// the shared input x with their own input fields, four rows of each per
// simdgroup; act[j·rows + r] = silu(g)·u (`moe_silu_jobs`'s expression).
// Per row the sums are `q8f_jobs_r4`'s; the SiLU epilogue in registers may
// contract differently from the separate pass, so the two agree to the last
// bits, not bit for bit. Opt-in (`CMF_MOE_Q8_GU=1`): slower on the M4.
kernel void q8f_moe_gu_r4(
    device const uchar*  q       [[buffer(0)]],
    device const float4* xs      [[buffer(1)]],
    device float*        act     [[buffer(2)]],
    constant uint&       cols4   [[buffer(3)]],
    constant uint&       rows    [[buffer(4)]],
    device const ulong*  bases   [[buffer(5)]],
    constant uint&       tg_per  [[buffer(6)]],
    constant uint&       ne      [[buffer(7)]],
    device const uchar*  qw1     [[buffer(8)]],
    device const uchar*  qw2     [[buffer(9)]],
    device const uchar*  qw3     [[buffer(10)]],
    constant ulong&      wstride [[buffer(11)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint j   = tgpos / tg_per;
    uint tgl = tgpos - j * tg_per;
    uint r0 = (tgl * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    ulong qlen = (ulong)rows * (ulong)cols4 * 4ul;
    device const uchar* pg = mq8_base(bases[j], q, qw1, qw2, qw3, wstride);
    device const uchar* pu = mq8_base(bases[ne + j], q, qw1, qw2, qw3, wstride);
    device const char4* qg = (device const char4*)pg;
    device const char4* qu = (device const char4*)pu;
    device const half* rsg = (device const half*)(pg + qlen);
    device const half* rsu = (device const half*)(pu + qlen);
    device const packed_half4* cg = (device const packed_half4*)(pg + qlen + (ulong)rows * 2ul);
    device const packed_half4* cu = (device const packed_half4*)(pu + qlen + (ulong)rows * 2ul);
    float4 g0 = float4(0.0f), g1 = float4(0.0f), g2 = float4(0.0f), g3 = float4(0.0f);
    float4 u0 = float4(0.0f), u1 = float4(0.0f), u2 = float4(0.0f), u3 = float4(0.0f);
    uint i = lane;
    for (; i + 96u < cols4; i += 128u) {
        float4 v0 = xs[i], v1 = xs[i + 32u], v2 = xs[i + 64u], v3 = xs[i + 96u];
        {
            float4 x0 = v0 * float4(cg[i]);
            float4 x1 = v1 * float4(cg[i + 32u]);
            float4 x2 = v2 * float4(cg[i + 64u]);
            float4 x3 = v3 * float4(cg[i + 96u]);
            for (uint ri = 0u; ri < nr; ++ri) {
                ulong base = (ulong)(r0 + ri) * cols4;
                float4 aa;
                aa.x = dot(float4(qg[base + i]), x0);
                aa.y = dot(float4(qg[base + i + 32u]), x1);
                aa.z = dot(float4(qg[base + i + 64u]), x2);
                aa.w = dot(float4(qg[base + i + 96u]), x3);
                if (ri == 0u) g0 += aa;
                else if (ri == 1u) g1 += aa;
                else if (ri == 2u) g2 += aa;
                else g3 += aa;
            }
        }
        {
            float4 x0 = v0 * float4(cu[i]);
            float4 x1 = v1 * float4(cu[i + 32u]);
            float4 x2 = v2 * float4(cu[i + 64u]);
            float4 x3 = v3 * float4(cu[i + 96u]);
            for (uint ri = 0u; ri < nr; ++ri) {
                ulong base = (ulong)(r0 + ri) * cols4;
                float4 aa;
                aa.x = dot(float4(qu[base + i]), x0);
                aa.y = dot(float4(qu[base + i + 32u]), x1);
                aa.z = dot(float4(qu[base + i + 64u]), x2);
                aa.w = dot(float4(qu[base + i + 96u]), x3);
                if (ri == 0u) u0 += aa;
                else if (ri == 1u) u1 += aa;
                else if (ri == 2u) u2 += aa;
                else u3 += aa;
            }
        }
    }
    for (; i < cols4; i += 32u) {
        float4 v = xs[i];
        float4 xg = v * float4(cg[i]);
        float4 xu = v * float4(cu[i]);
        for (uint ri = 0u; ri < nr; ++ri) {
            ulong o = (ulong)(r0 + ri) * cols4 + i;
            float vg = dot(float4(qg[o]), xg);
            float vu = dot(float4(qu[o]), xu);
            if (ri == 0u) { g0.x += vg; u0.x += vu; }
            else if (ri == 1u) { g1.x += vg; u1.x += vu; }
            else if (ri == 2u) { g2.x += vg; u2.x += vu; }
            else { g3.x += vg; u3.x += vu; }
        }
    }
    float tg0 = simd_sum(g0.x + g0.y + g0.z + g0.w);
    float tg1 = simd_sum(g1.x + g1.y + g1.z + g1.w);
    float tg2 = simd_sum(g2.x + g2.y + g2.z + g2.w);
    float tg3 = simd_sum(g3.x + g3.y + g3.z + g3.w);
    float tu0 = simd_sum(u0.x + u0.y + u0.z + u0.w);
    float tu1 = simd_sum(u1.x + u1.y + u1.z + u1.w);
    float tu2 = simd_sum(u2.x + u2.y + u2.z + u2.w);
    float tu3 = simd_sum(u3.x + u3.y + u3.z + u3.w);
    if (lane == 0u) {
        device float* aj = act + (ulong)j * (ulong)rows;
        float gv = tg0 * (float)rsg[r0];
        aj[r0] = (gv / (1.0f + exp(-gv))) * (tu0 * (float)rsu[r0]);
        if (nr > 1u) {
            gv = tg1 * (float)rsg[r0 + 1u];
            aj[r0 + 1u] = (gv / (1.0f + exp(-gv))) * (tu1 * (float)rsu[r0 + 1u]);
        }
        if (nr > 2u) {
            gv = tg2 * (float)rsg[r0 + 2u];
            aj[r0 + 2u] = (gv / (1.0f + exp(-gv))) * (tu2 * (float)rsu[r0 + 2u]);
        }
        if (nr > 3u) {
            gv = tg3 * (float)rsg[r0 + 3u];
            aj[r0 + 3u] = (gv / (1.0f + exp(-gv))) * (tu3 * (float)rsu[r0 + 3u]);
        }
    }
}

// The routed experts' down projections, their weighted mix and the residual
// in ONE dispatch: a simdgroup owns four output rows and walks the ne jobs
// in slot order, each with its own input field over its own activation row,
// so every row's mix accumulates as `moe_reduce_jobs` does (acc += w_e·y_e,
// e ascending from 0.0) and lands as h += acc.
kernel void q8f_moe_down_r4(
    device const uchar*  q       [[buffer(0)]],
    device const float4* a       [[buffer(1)]],
    device float*        h       [[buffer(2)]],
    constant uint&       cols4   [[buffer(3)]],
    constant uint&       rows    [[buffer(4)]],
    device const ulong*  bases   [[buffer(5)]],
    device const float*  wmix    [[buffer(6)]],
    constant uint&       ne      [[buffer(7)]],
    device const uchar*  qw1     [[buffer(8)]],
    device const uchar*  qw2     [[buffer(9)]],
    device const uchar*  qw3     [[buffer(10)]],
    constant ulong&      wstride [[buffer(11)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    ulong qlen = (ulong)rows * (ulong)cols4 * 4ul;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (uint e = 0u; e < ne; ++e) {
        device const uchar* pe = mq8_base(bases[e], q, qw1, qw2, qw3, wstride);
        device const char4* qq = (device const char4*)pe;
        device const half* rs = (device const half*)(pe + qlen);
        device const packed_half4* col = (device const packed_half4*)(pe + qlen + (ulong)rows * 2ul);
        device const float4* xs = a + (ulong)e * (ulong)cols4;
        float4 a0 = float4(0.0f), a1 = float4(0.0f);
        float4 a2 = float4(0.0f), a3 = float4(0.0f);
        uint i = lane;
        for (; i + 96u < cols4; i += 128u) {
            float4 x0 = xs[i] * float4(col[i]);
            float4 x1 = xs[i + 32u] * float4(col[i + 32u]);
            float4 x2 = xs[i + 64u] * float4(col[i + 64u]);
            float4 x3 = xs[i + 96u] * float4(col[i + 96u]);
            for (uint ri = 0u; ri < nr; ++ri) {
                ulong base = (ulong)(r0 + ri) * cols4;
                float4 aa;
                aa.x = dot(float4(qq[base + i]), x0);
                aa.y = dot(float4(qq[base + i + 32u]), x1);
                aa.z = dot(float4(qq[base + i + 64u]), x2);
                aa.w = dot(float4(qq[base + i + 96u]), x3);
                if (ri == 0u) a0 += aa;
                else if (ri == 1u) a1 += aa;
                else if (ri == 2u) a2 += aa;
                else a3 += aa;
            }
        }
        for (; i < cols4; i += 32u) {
            float4 xv = xs[i] * float4(col[i]);
            for (uint ri = 0u; ri < nr; ++ri) {
                float v = dot(float4(qq[(ulong)(r0 + ri) * cols4 + i]), xv);
                if (ri == 0u) a0.x += v;
                else if (ri == 1u) a1.x += v;
                else if (ri == 2u) a2.x += v;
                else a3.x += v;
            }
        }
        float we = wmix[e];
        o0 += we * (simd_sum(a0.x + a0.y + a0.z + a0.w) * (float)rs[r0]);
        if (nr > 1u) o1 += we * (simd_sum(a1.x + a1.y + a1.z + a1.w) * (float)rs[r0 + 1u]);
        if (nr > 2u) o2 += we * (simd_sum(a2.x + a2.y + a2.z + a2.w) * (float)rs[r0 + 2u]);
        if (nr > 3u) o3 += we * (simd_sum(a3.x + a3.y + a3.z + a3.w) * (float)rs[r0 + 3u]);
    }
    if (lane == 0u) {
        h[r0] = h[r0] + o0;
        if (nr > 1u) h[r0 + 1u] = h[r0 + 1u] + o1;
        if (nr > 2u) h[r0 + 2u] = h[r0 + 2u] + o2;
        if (nr > 3u) h[r0 + 3u] = h[r0 + 3u] + o3;
    }
}

// ── MoE chunk prefill ─────────────────────────────────────────────────

// SiLU·up per packed row with the down weight's input field folded in
// (the CPU's `prescale(act, col)`: (silu(g)·u)·col in f32), then the half
// guard of `moe_act_rows` on THAT row: the down GEMM stages X as half, and
// the field can lift a row past 65504 that silu·u alone kept under it.
// One threadgroup per packed row of ONE expert panel: `qd` is that expert's
// down weight in the blob, its input field `coloff` bytes in.
kernel void moe_act_rows_col(
    device const float* g      [[buffer(0)]],
    device const float* u      [[buffer(1)]],
    device float*       a      [[buffer(2)]],
    device float*       up     [[buffer(3)]],
    constant uint&      n      [[buffer(4)]],
    device const uchar* qd     [[buffer(5)]],
    constant uint&      coloff [[buffer(6)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint row  [[threadgroup_position_in_grid]])
{
    threadgroup float part[8];
    threadgroup float sc;
    device const half* col = (device const half*)(qd + coloff);
    ulong base = (ulong)row * n;
    float m = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float gv = g[base + i];
        float v = ((gv / (1.0f + exp(-gv))) * u[base + i]) * (float)col[i];
        a[base + i] = v;
        m = max(m, fabs(v));
    }
    m = simd_max(m);
    if (lane == 0u) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint q = 0u; q < 8u; ++q) tot = max(tot, part[q]);
        float e = tot > 16384.0f ? ceil(log2(tot / 16384.0f)) : 0.0f;
        sc = exp2(-e);
        up[row] = exp2(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float s = sc;
    if (s != 1.0f) {
        for (uint i = tid; i < n; i += 256u) a[base + i] = a[base + i] * s;
    }
}

// `q8_mul_mm`'s tiling (64 weight rows × 32 batch rows per 128-thread
// threadgroup, K steps of 32, both operand tiles 8×8-block-packed in
// threadgroup memory) for a q8_2f weight addressed by its blob offset alone:
// the row scales come from the f16 field behind the payload, and with
// `xcol != 0` the X tile is multiplied by the f16 input field while it is
// staged — the f32 x·col `col_scale_rows` would have written, without the
// staging pass or a host buffer per expert. Requires cols % 32 == 0.
//
// Q8MM_MODE picks the operand precision:
//   0 = half tiles, W = q·scale rounded to half (`q8_mul_mm` verbatim);
//   1 = half tiles, W = q exact in half, the row scale applied to the f32
//       result (only X is rounded);
//   2 = f32 tiles: q·scale is exact in f32 (7 × 11 significant bits) and
//       x·col stays f32 — the CPU's products, the GEMM's summation order.
#ifndef Q8MM_MODE
#define Q8MM_MODE 2
#endif
#if Q8MM_MODE == 2
typedef float q8mm_t;
typedef simdgroup_float8x8 q8mm_mat;
#define Q8MM_SHMEM 12288
#else
typedef half q8mm_t;
typedef simdgroup_half8x8 q8mm_mat;
#define Q8MM_SHMEM 8192
#endif

kernel void q8f_mul_mm_blob(
    device const char*   q     [[buffer(0)]],
    device const float*  xs    [[buffer(1)]],
    device float*        y     [[buffer(2)]],
    constant uint&       cols  [[buffer(3)]],
    constant uint&       rows  [[buffer(4)]],
    constant uint&       nb    [[buffer(5)]],
    constant uint&       xcol  [[buffer(6)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    threadgroup char shmem[Q8MM_SHMEM];
    threadgroup q8mm_t* sa = (threadgroup q8mm_t*)shmem;
    threadgroup q8mm_t* sb = (threadgroup q8mm_t*)(shmem + 64 * 32 * sizeof(q8mm_t));
    const uint NK = 32u;
    uint r0 = tg.y * 64u;   // weight-row tile
    uint r1 = tg.x * 32u;   // batch-row tile
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);

    ulong qlen = (ulong)rows * (ulong)cols;
    device const half* rsh = (device const half*)(q + qlen);
    device const half* colh = (device const half*)(q + qlen + (ulong)rows * 2ul);
    device const char* xrow = q + (ulong)(r0 + lr0) * cols + 16u * il0;
    device const float* yrow = xs + (ulong)(r1 + lr1) * cols + iy;
    device const half* crow = colh + iy;
#if Q8MM_MODE == 1
    const float wscale = 1.0f;
#else
    const float wscale = (float)rsh[r0 + lr0];
#endif
    bool usecol = xcol != 0u;

    q8mm_mat ma[4];
    q8mm_mat mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            device const char4* x4 = (device const char4*)xrow;
            float4 w0 = float4(x4[0]) * wscale;
            float4 w1 = float4(x4[1]) * wscale;
            float4 w2 = float4(x4[2]) * wscale;
            float4 w3 = float4(x4[3]) * wscale;
            float wv[16] = {
                w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w,
                w2.x, w2.y, w2.z, w2.w, w3.x, w3.y, w3.z, w3.w,
            };
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = (q8mm_t)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (q8mm_t)wv[i + 8u];
            }
        }
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* y4 = (device const float4*)yrow;
            float4 v0 = y4[0];
            float4 v1 = y4[1];
            if (usecol) {
                v0 = v0 * float4((float)crow[0], (float)crow[1], (float)crow[2], (float)crow[3]);
                v1 = v1 * float4((float)crow[4], (float)crow[5], (float)crow[6], (float)crow[7]);
            }
            threadgroup q8mm_t* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (q8mm_t)v0.x; dst[1] = (q8mm_t)v0.y;
            dst[2] = (q8mm_t)v0.z; dst[3] = (q8mm_t)v0.w;
            dst[4] = (q8mm_t)v1.x; dst[5] = (q8mm_t)v1.y;
            dst[6] = (q8mm_t)v1.z; dst[7] = (q8mm_t)v1.w;
        }
        xrow += NK;
        yrow += NK;
        crow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const q8mm_t* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const q8mm_t* lsmb = sb + 2u * 64u * (sgitg / 2u);
        #pragma clang loop unroll(full)
        for (short ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

#if Q8MM_MODE != 1
    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        // Interior tile: straight to device.
        device float* C = y + (r0 + 32u * (sgitg & 1u))
            + (ulong)(r1 + 16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4),
                            rows, ulong2(0, 0), false);
        }
        return;
    }
#endif
    // Edge tiles (and mode 1, whose row scales multiply the f32 result):
    // stage C through the re-cast shmem, then write it out.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* temp_str = ((threadgroup float*)shmem)
        + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                        64, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tiitg; i < 32u * 64u; i += 128u) {
        uint m = i / 64u, n = i % 64u;
        if (m < nr1 && n < nr0) {
#if Q8MM_MODE == 1
            float s = (float)rsh[r0 + n];
#else
            float s = 1.0f;
#endif
            y[(ulong)(r1 + m) * rows + r0 + n] = ((threadgroup float*)shmem)[m * 64u + n] * s;
        }
    }
}

// ── Chunk attention with f32 operand tiles (q8_2f MoE layers) ─────────
// `mul_mm_f32nt` / `mul_mm_f32nn` of the main library stage Q, K, P and V
// as half; these twins keep every operand f32 in threadgroup memory (same
// 64×32 tiles, 8×8-block packing and summation order).

// C[m,n] = X[m,k] · W[n,k]ᵀ · scale   (scores: X = Q panel, W = K rows)
kernel void att_mm_nt_f32(
    device const float*  xw    [[buffer(0)]],
    device const float*  xs    [[buffer(1)]],
    device float*        y     [[buffer(2)]],
    constant uint&       cols  [[buffer(3)]],
    constant uint&       rows  [[buffer(4)]],
    constant uint&       nb    [[buffer(5)]],
    constant float&      scale [[buffer(6)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    threadgroup char shmem[12288];
    threadgroup float* sa = (threadgroup float*)shmem;
    threadgroup float* sb = (threadgroup float*)(shmem + 8192);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;
    uint r1 = tg.x * 32u;
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);
    device const float* wrow = xw + (ulong)(r0 + lr0) * cols + 16u * il0;
    device const float* yrow = xs + (ulong)(r1 + lr1) * cols + iy;
    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            uint kb = k0 + 16u * il0;
            float wv[16];
            for (uint i = 0; i < 16u; ++i) {
                wv[i] = kb + i < cols ? wrow[i] : 0.0f;
            }
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = wv[i];
                sa[64u * ib1 + 8u * i + lx] = wv[i + 8u];
            }
        }
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            threadgroup float* dst = sb + 64u * ib + 8u * ly;
            for (uint i = 0; i < 8u; ++i) {
                dst[i] = k0 + iy + i < cols ? yrow[i] : 0.0f;
            }
        }
        wrow += NK;
        yrow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const float* lsmb = sb + 2u * 64u * (sgitg / 2u);
        #pragma clang loop unroll(full)
        for (short ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* temp_str = ((threadgroup float*)shmem)
        + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                        64, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tiitg; i < 32u * 64u; i += 128u) {
        uint m = i / 64u, n = i % 64u;
        if (r1 + m < nb && r0 + n < rows) {
            y[(ulong)(r1 + m) * rows + r0 + n] =
                ((threadgroup float*)shmem)[m * 64u + n] * scale;
        }
    }
}

// C[m,d] = P[m,n] · V[n,d]   (attention P·V: W is NOT transposed)
kernel void att_mm_nn_f32(
    device const float*  vw    [[buffer(0)]],
    device const float*  xs    [[buffer(1)]],
    device float*        y     [[buffer(2)]],
    constant uint&       kdim  [[buffer(3)]],
    constant uint&       rows  [[buffer(4)]],
    constant uint&       nb    [[buffer(5)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    threadgroup char shmem[8192];
    threadgroup float* sa = (threadgroup float*)shmem;          // V tile [16k × 64d]
    threadgroup float* sb = (threadgroup float*)(shmem + 4096); // P tile [32m × 16k]
    const uint NK = 16u;
    uint r0 = tg.y * 64u;
    uint r1 = tg.x * 32u;
    uint nr1 = min(nb - r1, 32u);
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint vk = tiitg / 8u;
    uint vd = 8u * (tiitg % 8u);
    uint iyp = 4u * (tiitg % 4u);
    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k0 = 0; k0 < kdim; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint dblk = vd / 8u;
            uint kblk = vk / 8u;
            uint ib = 8u * kblk + dblk;
            uint krow = vk % 8u;
            threadgroup float* dst = sa + 64u * ib + 8u * krow;
            device const float* vr = vw + (ulong)(k0 + vk) * rows + r0 + vd;
            bool kok = k0 + vk < kdim;
            for (uint i = 0; i < 8u; ++i) {
                bool ok = kok && r0 + vd + i < rows;
                dst[i] = ok ? vr[i] : 0.0f;
            }
        }
        {
            uint kb4 = iyp;
            uint sx = kb4 / 8u;
            uint off = kb4 % 8u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float* pr = xs + (ulong)(r1 + lr1) * kdim + k0 + kb4;
            threadgroup float* dst = sb + 64u * ib + 8u * ly + off;
            for (uint i = 0; i < 4u; ++i) {
                dst[i] = k0 + kb4 + i < kdim ? pr[i] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const float* lsmb = sb + 2u * 64u * (sgitg / 2u);
        #pragma clang loop unroll(full)
        for (short ik = 0; ik < 2; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0, 0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* temp_str = ((threadgroup float*)shmem)
        + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                        64, ulong2(0, 0), false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tiitg; i < 32u * 64u; i += 128u) {
        uint m = i / 64u, n = i % 64u;
        if (r1 + m < nb && r0 + n < rows) {
            y[(ulong)(r1 + m) * rows + r0 + n] =
                ((threadgroup float*)shmem)[m * 64u + n];
        }
    }
}
