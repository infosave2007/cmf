//! GPU path (D5, MVP): Metal on Apple Silicon.
//!
//! Architecture key: the CMF weights section is page-aligned in mmap → the GPU sees
//! THE SAME bytes via `newBufferWithBytesNoCopy` (unified memory), without
//! loading and without a second copy — cold weights stay cold.
//!
//! MVP scope: q8_row/q8_2f matvec for LARGE matrices (rows ≥ threshold —
//! in practice lm_head, the dominant decode matvec with a huge
//! vocabulary). Small matrices stay on the CPU: the dispatch cost (~50–100 µs)
//! eats the gain. Enable: `CMF_GPU=1`; any initialization failure —
//! an honest warning and CPU fallback (no silent accuracy degradations:
//! the kernel is mathematically identical to the CPU path, the same prescale trick).

use crate::gpu::{BatchJob, MoeJob};
use cortiq_core::CmfModel;
use cortiq_core::quant::{GROUP_SIZE, Q1_TILE, Q1T_TILE, Q4_TILE, f16_to_f32};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// CMF low-bit payloads are byte-packed. In particular Q1T uses a 9-byte
// group tile, so typed ushort/uint pointer casts are unaligned on most groups
// and therefore undefined in MSL. Assemble little-endian fields explicitly.
inline ushort cmf_load_u16_le(device const uchar* p) {
    return (ushort)p[0] | ((ushort)p[1] << 8u);
}
inline uint cmf_load_u32_le(device const uchar* p) {
    return (uint)p[0] | ((uint)p[1] << 8u) |
           ((uint)p[2] << 16u) | ((uint)p[3] << 24u);
}

// Shape-specialized pipeline variants (the llama.cpp trick): cols/rows
// arrive as FUNCTION CONSTANTS so the K-loop trip count and address
// strides are compile-time — fully unrolled, strength-reduced. Built
// per weight shape by the chunk graph (cached); the generic pipelines
// bind the buffer params instead (guarded by
// is_function_constant_defined).
constant uint FC_COLS [[function_constant(0)]];
constant uint FC_ROWS [[function_constant(1)]];

// y[o] = rs[o] * Σ_i q[o,i]·xs[i]; xs already prescaled by the col field (like CPU).
// SIMD group (32 lanes) per row: adjacent lanes read adjacent
// char4 → coalesced 128-byte reads; simd_sum reduction.
kernel void q8_matvec(
    device const char4*  q     [[buffer(0)]],
    device const float4* xs    [[buffer(1)]],
    device const float*  rs    [[buffer(2)]],
    device float*        y     [[buffer(3)]],
    constant uint&       cols4 [[buffer(4)]],
    constant uint&       rows  [[buffer(5)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint row = tgpos * sgs + sg;
    if (row >= rows) return;
    ulong base = (ulong)row * cols4;
    float4 acc = float4(0.0f);
    uint i = lane;
    for (; i + 96 < cols4; i += 128) {
        char4 q0 = q[base + i];
        char4 q1 = q[base + i + 32];
        char4 q2 = q[base + i + 64];
        char4 q3 = q[base + i + 96];

        float4 x0 = xs[i];
        float4 x1 = xs[i + 32];
        float4 x2 = xs[i + 64];
        float4 x3 = xs[i + 96];

        acc.x += dot(float4(q0), x0);
        acc.y += dot(float4(q1), x1);
        acc.z += dot(float4(q2), x2);
        acc.w += dot(float4(q3), x3);
    }
    for (; i < cols4; i += 32) {
        acc.x += dot(float4(q[base + i]), xs[i]);
    }
    float total = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) y[row] = total * rs[row];
}

// q8_2f twin: the input-channel field is applied while x is read. Keeping
// this inside the projection avoids a separate prescale dispatch/buffer for
// every Q/K/V/Gate/Up/Down projection in the whole-token graph.
kernel void q8f_matvec(
    device const char4*  q     [[buffer(0)]],
    device const float4* xs    [[buffer(1)]],
    device const float*  rs    [[buffer(2)]],
    device float*        y     [[buffer(3)]],
    device const float4* col   [[buffer(4)]],
    constant uint&       cols4 [[buffer(5)]],
    constant uint&       rows  [[buffer(6)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint row = tgpos * sgs + sg;
    if (row >= rows) return;
    ulong base = (ulong)row * cols4;
    float4 acc = float4(0.0f);
    uint i = lane;
    for (; i + 96 < cols4; i += 128) {
        char4 q0 = q[base + i];
        char4 q1 = q[base + i + 32];
        char4 q2 = q[base + i + 64];
        char4 q3 = q[base + i + 96];
        float4 x0 = xs[i] * col[i];
        float4 x1 = xs[i + 32] * col[i + 32];
        float4 x2 = xs[i + 64] * col[i + 64];
        float4 x3 = xs[i + 96] * col[i + 96];
        acc.x += dot(float4(q0), x0);
        acc.y += dot(float4(q1), x1);
        acc.z += dot(float4(q2), x2);
        acc.w += dot(float4(q3), x3);
    }
    for (; i < cols4; i += 32) {
        acc.x += dot(float4(q[base + i]), xs[i] * col[i]);
    }
    float total = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) y[row] = total * rs[row];
}

// act[i] = silu(g[i])·u[i]·col[i] — down_proj input with the col field already
// applied (q8_2f prescale on the GPU, without returning to the CPU).
// GEMM prefill batch: y[bi, o] = rs[o]·Σ q[o,i]·xs[bi,i].
// SIMD group per (row, position); the row is hot in L2 across bi.
kernel void q8_matmat(
    device const char4*  q     [[buffer(0)]],
    device const float4* xs    [[buffer(1)]],
    device const float*  rs    [[buffer(2)]],
    device float*        y     [[buffer(3)]],
    constant uint&       cols4 [[buffer(4)]],
    constant uint&       rows  [[buffer(5)]],
    constant uint&       nb    [[buffer(6)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint row = tg.x * sgs + sg;
    uint bi = tg.y;
    if (row >= rows || bi >= nb) return;
    ulong qb = (ulong)row * cols4;
    ulong xb = (ulong)bi * cols4;
    float4 acc = float4(0.0f);
    uint i = lane;
    for (; i + 96 < cols4; i += 128) {
        char4 q0 = q[qb + i];
        char4 q1 = q[qb + i + 32];
        char4 q2 = q[qb + i + 64];
        char4 q3 = q[qb + i + 96];

        float4 x0 = xs[xb + i];
        float4 x1 = xs[xb + i + 32];
        float4 x2 = xs[xb + i + 64];
        float4 x3 = xs[xb + i + 96];

        acc.x += dot(float4(q0), x0);
        acc.y += dot(float4(q1), x1);
        acc.z += dot(float4(q2), x2);
        acc.w += dot(float4(q3), x3);
    }
    for (; i < cols4; i += 32) {
        acc.x += dot(float4(q[qb + i]), xs[xb + i]);
    }
    float total = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) y[(ulong)bi * rows + row] = total * rs[row];
}

// True GEMM tile kernel for the prefill batch — the ggml mul_mm layout
// ported to our q8_row format (per-row f32 scale folded in at the W
// load; |w·s| well inside half range, mul_mm precision class). C-tile
// 64 weight rows × 32 batch rows per 128-thread / 4-simdgroup
// threadgroup, K in steps of 32; BOTH operand tiles live in threadgroup
// memory PACKED AS CONTIGUOUS 8×8 BLOCKS (stride 8), so every
// simdgroup_load reads one dense 64-element block — the wide-row-stride
// layouts of the earlier variants were the throughput ceiling (~1.5
// TF); this one measures materially higher. Per-thread device reads are
// fully coalesced: 16 consecutive quants of one W row / 8 consecutive
// floats of one X row per K-step. Requires cols % 32 == 0 (the host
// falls back to the matvec-style kernel otherwise).
kernel void q8_mul_mm(
    device const char*   q     [[buffer(0)]],
    device const float*  xs    [[buffer(1)]],
    device const float*  rs    [[buffer(2)]],
    device float*        y     [[buffer(3)]],
    constant uint&       cols_b [[buffer(4)]],
    constant uint&       rows_b [[buffer(5)]],
    constant uint&       nb    [[buffer(6)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    uint cols = is_function_constant_defined(FC_COLS) ? FC_COLS : cols_b;
    uint rows = is_function_constant_defined(FC_ROWS) ? FC_ROWS : rows_b;
    // ggml's exact shmem shape: one 8 KB char arena, W/X tiles as
    // casted half views during the K loop, the same bytes re-cast to
    // float for EDGE-tile C staging only — interior tiles store straight
    // to device (their aligned fast path). An earlier float-typed arena
    // measured 4.7× slower; the char base + ggml's access pattern does
    // not trip that.
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;   // weight-row tile
    uint r1 = tg.x * 32u;   // batch-row tile
    // Clamped in-tile coordinates (edge tiles re-load a valid row; the
    // guarded C write drops the duplicates).
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);   // 0..63 W row in tile
    uint il0 = tiitg % 2u;                  // which 16-col half of NK
    uint lr1 = min(tiitg / 4u, nr1 - 1u);   // 0..31 X row in tile
    uint iy  = 8u * (tiitg % 4u);           // k offset of this thread's 8 floats

    device const char* xrow = q + (ulong)(r0 + lr0) * cols + 16u * il0;
    device const float* yrow = xs + (ulong)(r1 + lr1) * cols + iy;
    float wscale = rs[r0 + lr0];

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // W: 16 consecutive quants (4 vector loads) → one
        // 8x8-block-packed column pair. No bounds branches in here:
        // cols % 32 == 0 is a host gate, and the row clamps above keep
        // every pointer in range — ggml compiles its checks out with
        // function constants, we simply don't emit them.
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
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        // X: 8 consecutive floats → one 8x8-block row.
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* y4 = (device const float4*)yrow;
            float4 v0 = y4[0];
            float4 v1 = y4[1];
            // NOTE: half4 threadgroup stores here measured 2× slower —
            // threadgroup pointer casts defeat the alias analysis (same
            // lesson as the arena union). Scalar stores compile clean.
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (half)v0.x; dst[1] = (half)v0.y;
            dst[2] = (half)v0.z; dst[3] = (half)v0.w;
            dst[4] = (half)v1.x; dst[5] = (half)v1.y;
            dst[6] = (half)v1.z; dst[7] = (half)v1.w;
        }
        xrow += NK;
        yrow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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

    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        // Interior tile: straight to device (ggml's aligned fast path).
        device float* C = y + (r0 + 32u * (sgitg & 1u))
            + (ulong)(r1 + 16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4),
                            rows, ulong2(0, 0), false);
        }
    } else {
        // Edge tile: stage through the (re-cast) shmem, sg 0 writes out.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                            64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (uint j = tiitg; j < nr1; j += 128u) {
                device float* D = y + r0 + (ulong)(r1 + j) * rows;
                threadgroup const float* Cr = ((threadgroup float*)shmem) + j * 64u;
                for (uint i = 0; i < nr0; ++i) {
                    D[i] = Cr[i];
                }
            }
        }
    }
}

// Q1T Ternary Lookup Table: 243 entries. Packs 5 ternary signs into 10 bits.
// Each code 0,1,2 maps to bits: 00, 01, 10.
constant ushort Q1T_LUT[243] = {
    0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 10u, 16u, 17u, 18u, 20u, 21u, 22u, 24u,
    25u, 26u, 32u, 33u, 34u, 36u, 37u, 38u, 40u, 41u, 42u, 64u, 65u, 66u, 68u, 69u,
    70u, 72u, 73u, 74u, 80u, 81u, 82u, 84u, 85u, 86u, 88u, 89u, 90u, 96u, 97u, 98u,
    100u, 101u, 102u, 104u, 105u, 106u, 128u, 129u, 130u, 132u, 133u, 134u, 136u, 137u, 138u, 144u,
    145u, 146u, 148u, 149u, 150u, 152u, 153u, 154u, 160u, 161u, 162u, 164u, 165u, 166u, 168u, 169u,
    170u, 256u, 257u, 258u, 260u, 261u, 262u, 264u, 265u, 266u, 272u, 273u, 274u, 276u, 277u, 278u,
    280u, 281u, 282u, 288u, 289u, 290u, 292u, 293u, 294u, 296u, 297u, 298u, 320u, 321u, 322u, 324u,
    325u, 326u, 328u, 329u, 330u, 336u, 337u, 338u, 340u, 341u, 342u, 344u, 345u, 346u, 352u, 353u,
    354u, 356u, 357u, 358u, 360u, 361u, 362u, 384u, 385u, 386u, 388u, 389u, 390u, 392u, 393u, 394u,
    400u, 401u, 402u, 404u, 405u, 406u, 408u, 409u, 410u, 416u, 417u, 418u, 420u, 421u, 422u, 424u,
    425u, 426u, 512u, 513u, 514u, 516u, 517u, 518u, 520u, 521u, 522u, 528u, 529u, 530u, 532u, 533u,
    534u, 536u, 537u, 538u, 544u, 545u, 546u, 548u, 549u, 550u, 552u, 553u, 554u, 576u, 577u, 578u,
    580u, 581u, 582u, 584u, 585u, 586u, 592u, 593u, 594u, 596u, 597u, 598u, 600u, 601u, 602u, 608u,
    609u, 610u, 612u, 613u, 614u, 616u, 617u, 618u, 640u, 641u, 642u, 644u, 645u, 646u, 648u, 649u,
    650u, 656u, 657u, 658u, 660u, 661u, 662u, 664u, 665u, 666u, 672u, 673u, 674u, 676u, 677u, 678u,
    680u, 681u, 682u,
};

// q1t register-blocked GEMM (prefill): identical simdgroup-matrix machinery to
// q8_mul_mm; only the weight staging decodes base-3 ternary tiles (per-group
// f16 scale) instead of i8·row_scale. NK=32 == GROUP_SIZE so each K-step is one
// group; no row_scale buffer. The sparse overlay is added by q1t_overlay_mm.
kernel void q1t_mul_mm(
    device const uchar*  q      [[buffer(0)]],
    device const float*  xs     [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint&       cols_b [[buffer(3)]],
    constant uint&       rows_b [[buffer(4)]],
    constant uint&       nb     [[buffer(5)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    uint cols = cols_b;
    uint rows = rows_b;
    uint gpr = cols >> 5u;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;
    uint r1 = tg.x * 32u;
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);

    device const float* yrow = xs + (ulong)(r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // W: decode this thread's 16 ternary weights (row r0+lr0, K-half il0).
        {
            uint g = k0 >> 5u;
            device const uchar* tile = q + ((ulong)(r0 + lr0) * gpr + (ulong)g) * 9u;
            half scale = as_type<half>((ushort)((uint)tile[0] | ((uint)tile[1] << 8)));
            device const uchar* codes = tile + 2u;
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            float wv[16];
            for (uint i = 0; i < 16u; ++i) {
                uint p = 16u * il0 + i;
                ushort bb = Q1T_LUT[codes[p / 5u]];
                uint code = (bb >> ((p % 5u) * 2u)) & 3u;
                float sgn = (float)(code == 1u) - (float)(code == 2u);
                wv[i] = sgn * (float)scale;
            }
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        // X: 8 consecutive floats → one 8x8-block row (identical to q8).
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* y4 = (device const float4*)yrow;
            float4 v0 = y4[0];
            float4 v1 = y4[1];
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (half)v0.x; dst[1] = (half)v0.y;
            dst[2] = (half)v0.z; dst[3] = (half)v0.w;
            dst[4] = (half)v1.x; dst[5] = (half)v1.y;
            dst[6] = (half)v1.z; dst[7] = (half)v1.w;
        }
        yrow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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

    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        device float* C = y + (r0 + 32u * (sgitg & 1u))
            + (ulong)(r1 + 16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4),
                            rows, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                            64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (uint j = tiitg; j < nr1; j += 128u) {
                device float* D = y + r0 + (ulong)(r1 + j) * rows;
                threadgroup const float* Cr = ((threadgroup float*)shmem) + j * 64u;
                for (uint i = 0; i < nr0; ++i) {
                    D[i] = Cr[i];
                }
            }
        }
    }
}

// Batched q1t overlay: adds the sparse outlier overlay onto the GEMM output
// Y[bi*rows + r] for every batch column. One thread per row; byte-wise reads.
kernel void q1t_overlay_mm(
    device const uchar* q        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float*       y        [[buffer(2)]],
    constant uint&      base_len [[buffer(3)]],
    constant uint&      rows     [[buffer(4)]],
    constant uint&      cols     [[buffer(5)]],
    constant uint&      nb       [[buffer(6)]],
    uint rid [[thread_position_in_grid]])
{
    if (rid >= rows) return;
    uint c0 = cmf_load_u32_le(q + base_len + rid * 4u);
    uint c1 = cmf_load_u32_le(q + base_len + (rid + 1u) * 4u);
    uint ent = base_len + (rows + 1u) * 4u;
    for (uint p = c0; p < c1; ++p) {
        uint e = ent + p * 4u;
        uint col_val = cmf_load_u32_le(q + e);
        uint col = col_val & 0xFFFF;
        float fv = (float)as_type<half>((ushort)(col_val >> 16));
        for (uint bi = 0; bi < nb; ++bi) {
            y[(ulong)bi * rows + rid] += fv * x[(ulong)bi * cols + col];
        }
    }
}

// q8_mul_mm with the FFN activation fused into the X-tile load:
// x[i] = silu(g[i])·u[i] — the down GEMM consumes gate/up directly, no
// separate silu dispatch, no act-buffer round trip (profiled at 8% of
// the chunk as a standalone stage).
kernel void q8_mul_mm_silu(
    device const char*   q     [[buffer(0)]],
    device const float*  gs    [[buffer(1)]],
    device const float*  us    [[buffer(2)]],
    device const float*  rs    [[buffer(3)]],
    device float*        y     [[buffer(4)]],
    constant uint&       cols_b [[buffer(5)]],
    constant uint&       rows_b [[buffer(6)]],
    constant uint&       nb    [[buffer(7)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    uint cols = is_function_constant_defined(FC_COLS) ? FC_COLS : cols_b;
    uint rows = is_function_constant_defined(FC_ROWS) ? FC_ROWS : rows_b;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;
    uint r1 = tg.x * 32u;
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);
    device const char* xrow = q + (ulong)(r0 + lr0) * cols + 16u * il0;
    device const float* grow = gs + (ulong)(r1 + lr1) * cols + iy;
    device const float* urow = us + (ulong)(r1 + lr1) * cols + iy;
    float wscale = rs[r0 + lr0];
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
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
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* g4 = (device const float4*)grow;
            device const float4* u4 = (device const float4*)urow;
            float4 g0 = g4[0];
            float4 g1 = g4[1];
            float4 u0 = u4[0];
            float4 u1 = u4[1];
            float4 a0 = (g0 / (1.0f + exp(-g0))) * u0;
            float4 a1 = (g1 / (1.0f + exp(-g1))) * u1;
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (half)a0.x; dst[1] = (half)a0.y;
            dst[2] = (half)a0.z; dst[3] = (half)a0.w;
            dst[4] = (half)a1.x; dst[5] = (half)a1.y;
            dst[6] = (half)a1.z; dst[7] = (half)a1.w;
        }
        xrow += NK;
        grow += NK;
        urow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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
                ((threadgroup float*)shmem)[m * 64u + n];
        }
    }
}

// f32 GEMM twins of q8_mul_mm for the chunk attention (profiled: the
// streaming attend was 47% of the chunk — GEMM attention is the same
// two-GEMM shape the CPU AMX path uses). Same 64×32 tile / 8x8-block
// shared layout; K-tails guarded (n is arbitrary).
// C[m,n] = X[m,k] · W[n,k]ᵀ · scale   (scores: X=Q panel, W=K rows)
kernel void mul_mm_f32nt(
    device const float*  xw    [[buffer(0)]],   // W [rows × cols]
    device const float*  xs    [[buffer(1)]],   // X [nb × cols]
    device float*        y     [[buffer(2)]],   // C [nb × rows]
    constant uint&       cols_b [[buffer(3)]],
    constant uint&       rows  [[buffer(4)]],
    constant uint&       nb    [[buffer(5)]],
    constant float&      scale [[buffer(6)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    // cols = head_dim (64/128) is stable per model — specialized
    // pipelines unroll the whole K loop for the scores GEMM.
    uint cols = is_function_constant_defined(FC_COLS) ? FC_COLS : cols_b;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
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
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
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
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            for (uint i = 0; i < 8u; ++i) {
                dst[i] = k0 + iy + i < cols ? (half)yrow[i] : (half)0.0f;
            }
        }
        wrow += NK;
        yrow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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
kernel void mul_mm_f32nn(
    device const float*  vw    [[buffer(0)]],   // V [kdim × rows] row-major
    device const float*  xs    [[buffer(1)]],   // P [nb × kdim]
    device float*        y     [[buffer(2)]],   // C [nb × rows]
    constant uint&       kdim  [[buffer(3)]],
    constant uint&       rows_b [[buffer(4)]],
    constant uint&       nb    [[buffer(5)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    // rows = head_dim is stable; kdim (context) varies per chunk and
    // stays a buffer param.
    uint rows = is_function_constant_defined(FC_ROWS) ? FC_ROWS : rows_b;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;      // V tile [16k × 64d] packed
    threadgroup half* sb = (threadgroup half*)(shmem + 4096); // P tile [32m × 16k]
    const uint NK = 16u;
    uint r0 = tg.y * 64u;   // d tile
    uint r1 = tg.x * 32u;   // m tile
    uint nr1 = min(nb - r1, 32u);
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    // V tile loader coords: 128 threads cover 16×64 halfs, 8 per thread.
    // Thread t loads row kv = t/8, col span 8*(t%8).
    uint vk = tiitg / 8u;       // 0..15 k-row in tile
    uint vd = 8u * (tiitg % 8u); // 0..56 d-col start
    uint iyp = 4u * (tiitg % 4u); // P: 4 floats per thread per row
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k0 = 0; k0 < kdim; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // V tile: [k][d] 8x8-block packed: block ib = 8*(d_blk) + k_blk?
        // Keep the SAME packing convention as sa in the nt kernel:
        // ma fragment i covers d-range [8i, 8i+8) of the sg's 32-wide
        // strip; blocks indexed ib = 8*dblk + kblk over [64d × 16k]…
        // simpler: store [d][k] transposed so the fragment layout matches
        // the nt kernel exactly (ma loads want [k][d(row-major 8x8)] via
        // transpose=false on [d][k]? — no: multiply(mb[m,k], ma[k,d])
        // needs ma fragment [k][d]. Store blocks as [k][d]:
        // ib = 8*sxd + syk with row=k%8, col=d%8.
        {
            uint dblk = vd / 8u;        // 0..7
            uint kblk = vk / 8u;        // 0..1
            // Block index MUST be k-major (ib = 8·kblk + dblk): the
            // compute loop advances k with lsma += 8·64 and picks the
            // d-half with 4·64·(sgitg%2) — same convention as sa in nt.
            uint ib = 8u * kblk + dblk;
            uint krow = vk % 8u;
            threadgroup half* dst = sa + 64u * ib + 8u * krow;
            device const float* vr = vw + (ulong)(k0 + vk) * rows + r0 + vd;
            bool kok = k0 + vk < kdim;
            for (uint i = 0; i < 8u; ++i) {
                bool ok = kok && r0 + vd + i < rows;
                dst[i] = ok ? (half)vr[i] : (half)0.0f;
            }
        }
        // P tile [32m × 16k]: blocks ib = 4*kblk… same as sb in nt:
        // thread t: row m = t/4, 4 floats at 4*(t%4).
        {
            uint kb4 = iyp;
            uint sx = kb4 / 8u;         // which 8-k block half? kb4 in {0,4,8,12}
            uint off = kb4 % 8u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float* pr = xs + (ulong)(r1 + lr1) * kdim + k0 + kb4;
            threadgroup half* dst = sb + 64u * ib + 8u * ly + off;
            for (uint i = 0; i < 4u; ++i) {
                dst[i] = k0 + kb4 + i < kdim ? (half)pr[i] : (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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

// Causal softmax over score rows [m = hl·nb + bi], allowed = s0+bi+1;
// one simdgroup per row (lane-strided max / exp-sum / scale).
kernel void causal_softmax(
    device float*  p    [[buffer(0)]],
    constant uint& n    [[buffer(1)]],  // row length (stride)
    constant uint& s0   [[buffer(2)]],
    constant uint& nb   [[buffer(3)]],
    constant uint& m    [[buffer(4)]],  // rows
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgp [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]])
{
    uint row = tgp * sgs + sg;
    if (row >= m) return;
    uint allowed = s0 + (row % nb) + 1u;
    device float* r = p + (ulong)row * n;
    float mx = -INFINITY;
    for (uint i = lane; i < allowed; i += 32u) mx = max(mx, r[i]);
    mx = simd_max(mx);
    float sum = 0.0f;
    for (uint i = lane; i < allowed; i += 32u) {
        float e = exp(r[i] - mx);
        r[i] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint i = lane; i < allowed; i += 32u) r[i] *= inv;
    for (uint i = allowed + lane; i < n; i += 32u) r[i] = 0.0f;
}

// Born importance: imp[pos] += Σ over rows of P[row, pos] (masked
// column sums — the zeroed tail contributes nothing). One THREAD per
// position, rows walked inside: adjacent threads read adjacent
// positions, so every row pass is coalesced (the lane-per-column form
// read 4 of every 128 bytes and cost as much as the P·V GEMM). The
// KV groups' encoders serialize on this buffer — plain read-add is
// safe, no atomics.
kernel void imp_colsum(
    device const float* p   [[buffer(0)]],
    device atomic_float* imp [[buffer(1)]],
    constant uint& n   [[buffer(2)]],
    constant uint& m   [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    // x: position (adjacent threads → coalesced row reads); y: a chunk
    // of 32 row-slices so the grid stays wide enough to hide latency.
    uint pos = gid.x;
    if (pos >= n) return;
    uint step = (m + 31u) / 32u;
    uint r0 = gid.y * step;
    uint r1 = min(m, r0 + step);
    float acc = 0.0f;
    for (uint r = r0; r < r1; ++r) {
        acc += p[(ulong)r * n + pos];
    }
    atomic_fetch_add_explicit(&imp[pos], acc, memory_order_relaxed);
}

// Panel unstack: attn panel [head][bi][hd] → [bi][head·hd] for the O GEMM.
kernel void panel_unstack(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    constant uint& nh [[buffer(2)]],
    constant uint& nb [[buffer(3)]],
    constant uint& hd [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    uint total = nh * nb * hd;
    if (i >= total) return;
    uint h = i / (nb * hd);
    uint bi = (i / hd) % nb;
    uint d = i % hd;
    dst[((ulong)bi * nh + h) * hd + d] = src[i];
}

// Full-row softmax for the DiT's bidirectional attention: one
// 256-thread threadgroup per row, strided max/exp-sum reductions, in
// place. exp/order differ from the CPU softmax (tolerance-gated,
// like every GPU reduction here).
kernel void softmax_rows(
    device float*  s [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint TPT = 256u;
    device float* row = s + (ulong)tg * n;
    threadgroup float red[256];
    float mx = -INFINITY;
    for (uint i = tid; i < n; i += TPT) mx = max(mx, row[i]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TPT >> 1u; off > 0u; off >>= 1u) {
        if (tid < off) red[tid] = max(red[tid], red[tid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = tid; i < n; i += TPT) {
        float e = exp(row[i] - mx);
        row[i] = e;
        sum += e;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TPT >> 1u; off > 0u; off >>= 1u) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    for (uint i = tid; i < n; i += TPT) row[i] *= inv;
}

// DiT flash attention V2: bidirectional, online softmax, no n×n
// scores in device memory. V1 staged Q/K/V through a 31 KB
// threadgroup arena — occupancy collapsed to one group per core and
// every per-tile threadgroup barrier drained the pipeline. V2 loads
// the 8×8 operand blocks with simdgroup_load STRAIGHT from device
// memory (K transposed by the load), so the only threadgroup state
// is a per-simdgroup S/P tile + stats (~5.5 KB total) and the KV
// loop has NO threadgroup barriers at all — each simdgroup runs its
// 8 query rows independently; L1/SLC serve the K/V block reuse.
// f32 MACs throughout (same rate as half on Apple GPUs). The
// per-row rescale of the O accumulators multiplies by a diagonal
// float8x8 (simdgroup matrices have no per-row scalar op), skipped
// whenever a KV tile raises no row max. GQA: the K/V head is
// h / hpk. Buffers are padded to n32 = ceil(n/32)·32 rows per head
// with ZEROED tails (host contract) — tail keys mask to p = 0 in
// the scalar phase. Output goes straight into the [n][nh·hd] layout
// the O projection reads. Host gates: hd ≤ 128, hd % 8 == 0.
kernel void dit_flash_attend(
    device const float* qh  [[buffer(0)]],   // [nh][n32][hd] head-major
    device const float* kh  [[buffer(1)]],   // [nkv][n32][hd]
    device const float* vh  [[buffer(2)]],   // [nkv][n32][hd]
    device float*       out [[buffer(3)]],   // [n][nh·hd]
    constant uint&  n     [[buffer(4)]],
    constant uint&  hd    [[buffer(5)]],
    constant uint&  nh    [[buffer(6)]],
    constant uint&  hpk   [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint&  n32   [[buffer(9)]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint2 tg   [[threadgroup_position_in_grid]])
{
    const uint QT = 32u;                 // query tile
    const uint KT = 32u;                 // kv tile
    uint qbase = tg.x * QT;
    uint h = tg.y;
    uint kv = h / hpk;
    device const float* qsrc = qh + (ulong)h * n32 * hd;
    device const float* ksrc = kh + (ulong)kv * n32 * hd;
    device const float* vsrc = vh + (ulong)kv * n32 * hd;

    // Per-simdgroup shmem only — nothing crosses simdgroups.
    threadgroup float ssm[4 * 8 * 32];   // S then P, per sg
    threadgroup float sdm[4 * 64];       // diagonal per sg
    threadgroup float smm[4 * 8];        // running row max per sg
    threadgroup float slm[4 * 8];        // running row sum per sg
    threadgroup float* ss = ssm + sgitg * (8 * 32);
    threadgroup float* sd = sdm + sgitg * 64;
    threadgroup float* sm = smm + sgitg * 8;
    threadgroup float* sl = slm + sgitg * 8;

    device const float* qrow = qsrc + (ulong)(qbase + 8u * sgitg) * hd;
    uint srow = lane / 4u;               // scalar phase: 4 lanes per row
    uint schunk = lane % 4u;

    if (lane < 8u) {
        sm[lane] = -1e30f;
        sl[lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 a8, b8;
    simdgroup_float8x8 o8[16];           // 8 rows × hd cols, hd/8 ≤ 16 blocks
    uint nob = hd / 8u;
    for (uint i = 0; i < nob; ++i) {
        o8[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint kb0 = 0; kb0 < n; kb0 += KT) {
        // S[8×32] = Q·Kᵀ, operands straight from device (K blocks
        // load transposed — measured FASTER than a pre-transposed K
        // whose n32-strided block rows lose cache locality; padded
        // tail rows are zero).
        for (uint cb = 0; cb < KT / 8u; ++cb) {
            simdgroup_float8x8 s8 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            device const float* krow = ksrc + (ulong)(kb0 + cb * 8u) * hd;
            for (uint kb = 0; kb < nob; ++kb) {
                simdgroup_load(a8, qrow + kb * 8u, hd, ulong2(0, 0), false);
                simdgroup_load(b8, krow + kb * 8u, hd, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s8, a8, b8, s8);
            }
            simdgroup_store(s8, ss + cb * 8u, KT, ulong2(0, 0), false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax (scale folded here), P overwrites S in ss.
        float mprev = sm[srow];
        float lprev = sl[srow];
        float lmax = -1e30f;
        for (uint j = 0; j < 8u; ++j) {
            uint col = schunk * 8u + j;
            if (kb0 + col < n) {
                lmax = max(lmax, ss[srow * KT + col] * scale);
            }
        }
        lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
        lmax = max(lmax, simd_shuffle_xor(lmax, 2u));
        float mnew = max(mprev, lmax);
        float alpha = exp(mprev - mnew);
        float psum = 0.0f;
        for (uint j = 0; j < 8u; ++j) {
            uint col = schunk * 8u + j;
            float p =
                (kb0 + col < n) ? exp(ss[srow * KT + col] * scale - mnew) : 0.0f;
            ss[srow * KT + col] = p;
            psum += p;
        }
        psum += simd_shuffle_xor(psum, 1u);
        psum += simd_shuffle_xor(psum, 2u);
        if (schunk == 0u) {
            sm[srow] = mnew;
            sl[srow] = alpha * lprev + psum;
        }
        // Rescale O by diag(alpha) only when some row max moved.
        if (simd_any(alpha != 1.0f)) {
            for (uint i = lane; i < 64u; i += 32u) sd[i] = 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (schunk == 0u) sd[srow * 8u + srow] = alpha;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 d8;
            simdgroup_load(d8, sd, 8u, ulong2(0, 0), false);
            for (uint i = 0; i < nob; ++i) {
                simdgroup_float8x8 t8;
                simdgroup_multiply(t8, d8, o8[i]);
                o8[i] = t8;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // O += P·V, V blocks straight from device (padded tails zero,
        // and their P is zero anyway).
        for (uint kb = 0; kb < KT / 8u; ++kb) {
            simdgroup_load(a8, ss + kb * 8u, KT, ulong2(0, 0), false);
            device const float* vrow = vsrc + (ulong)(kb0 + kb * 8u) * hd;
            for (uint i = 0; i < nob; ++i) {
                simdgroup_load(b8, vrow + i * 8u, hd, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(o8[i], a8, b8, o8[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O /= l (diagonal), then store. Full tiles go straight to device
    // (row stride nh·hd IS the output layout); the edge q-tile stages
    // each 8×8 block through sd and copies the valid rows.
    {
        float linv = 1.0f / max(sl[srow], 1e-30f);
        for (uint i = lane; i < 64u; i += 32u) sd[i] = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (schunk == 0u) sd[srow * 8u + srow] = linv;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 d8;
        simdgroup_load(d8, sd, 8u, ulong2(0, 0), false);
        uint row0 = qbase + 8u * sgitg;
        for (uint i = 0; i < nob; ++i) {
            simdgroup_float8x8 t8;
            simdgroup_multiply(t8, d8, o8[i]);
            if (row0 + 8u <= n) {
                simdgroup_store(t8, out + ((ulong)row0 * nh + h) * hd + i * 8u,
                                (ulong)nh * hd, ulong2(0, 0), false);
            } else {
                simdgroup_store(t8, sd, 8u, ulong2(0, 0), false);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64u; e += 32u) {
                    uint r = e / 8u;
                    if (row0 + r < n) {
                        out[((ulong)(row0 + r) * nh + h) * hd + i * 8u + e % 8u] =
                            sd[e];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// VAE conv2d as implicit GEMM: the q8_mul_mm tile machinery, but the
// X staging gathers the 3×3 (or 1×1) receptive field straight from
// the NCHW image — the CPU path materializes a ≥2 GB im2col patch
// matrix per high-res conv, which is the VAE's real wall. W is dense
// f32 [oc, ic·k²]; K-tails zero-fill (ic·k² need not divide 32).
// Output is a [hw, oc] panel; panel_to_nchw adds bias and transposes.
kernel void conv_mul_mm(
    device const float* wt    [[buffer(0)]],   // [oc, ic·k²]
    device const float* img   [[buffer(1)]],   // [ic, h, w]
    device float*       y     [[buffer(2)]],   // [hw, oc] panel
    constant uint&      ick2  [[buffer(3)]],
    constant uint&      oc    [[buffer(4)]],
    constant uint&      hw    [[buffer(5)]],
    constant uint&      ih    [[buffer(6)]],
    constant uint&      iw    [[buffer(7)]],
    constant uint&      kk    [[buffer(8)]],   // kernel size k
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint rows = oc;
    uint nb = hw;
    uint r0 = tg.y * 64u;   // oc tile
    uint r1 = tg.x * 32u;   // output-position tile
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);
    uint pad = kk / 2u;
    uint k2 = kk * kk;
    // This thread's output position (clamped like the row clamps).
    uint pos = r1 + lr1;
    uint py = pos / iw;
    uint px = pos % iw;

    device const float* wrow = wt + (ulong)(r0 + lr0) * ick2 + 16u * il0;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < ick2; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // W: 16 dense f32 → half, K-tail zero-filled.
        {
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            uint kb = k0 + 16u * il0;
            float wv[16];
            for (uint i = 0; i < 16u; ++i) {
                wv[i] = kb + i < ick2 ? wrow[i] : 0.0f;
            }
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        // X: gather 8 receptive-field taps for this position.
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            for (uint i = 0; i < 8u; ++i) {
                uint kki = k0 + iy + i;
                float v = 0.0f;
                if (kki < ick2) {
                    uint c = kki / k2;
                    uint r = kki % k2;
                    int sy2 = (int)py + (int)(r / kk) - (int)pad;
                    int sx2 = (int)px + (int)(r % kk) - (int)pad;
                    if (sy2 >= 0 && sy2 < (int)ih && sx2 >= 0 && sx2 < (int)iw) {
                        v = img[(ulong)c * ih * iw + (ulong)sy2 * iw + (ulong)sx2];
                    }
                }
                dst[i] = (half)v;
            }
        }
        wrow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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

    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        device float* C = y + (r0 + 32u * (sgitg & 1u))
            + (ulong)(r1 + 16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4),
                            rows, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                            64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (uint j = tiitg; j < nr1; j += 128u) {
                device float* D = y + r0 + (ulong)(r1 + j) * rows;
                threadgroup const float* Cr = ((threadgroup float*)shmem) + j * 64u;
                for (uint i = 0; i < nr0; ++i) {
                    D[i] = Cr[i];
                }
            }
        }
    }
}

// GroupNorm pass 1: per-group mean and 1/σ. One 256-thread group per
// channel group, grid-stride partial sums (f32 partials —
// tolerance-class vs the CPU's f64, like every GPU reduction here).
kernel void gn_reduce(
    device const float* x  [[buffer(0)]],
    device float*       st [[buffer(1)]],   // [groups][2]: mean, inv
    constant uint& per_g [[buffer(2)]],
    constant uint& hw    [[buffer(3)]],
    constant float& eps  [[buffer(4)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint TPT = 256u;
    ulong base = (ulong)tg * per_g * hw;
    ulong count = (ulong)per_g * hw;
    threadgroup float rs[256];
    threadgroup float rq[256];
    float s = 0.0f, q = 0.0f;
    for (ulong i = tid; i < count; i += TPT) {
        float v = x[base + i];
        s += v;
        q += v * v;
    }
    rs[tid] = s;
    rq[tid] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TPT >> 1u; off > 0u; off >>= 1u) {
        if (tid < off) {
            rs[tid] += rs[tid + off];
            rq[tid] += rq[tid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float mean = rs[0] / (float)count;
        float var = rq[0] / (float)count - mean * mean;
        st[2u * tg] = mean;
        st[2u * tg + 1u] = rsqrt(max(var, 0.0f) + eps);
    }
}

// GroupNorm pass 2: normalize + affine, SiLU optionally fused (the
// decoder always follows norm with silu).
kernel void gn_apply(
    device const float* x  [[buffer(0)]],
    device float*       y  [[buffer(1)]],
    device const float* st [[buffer(2)]],
    device const float* wa [[buffer(3)]],
    device const float* ba [[buffer(4)]],
    constant uint& per_g   [[buffer(5)]],
    constant uint& hw      [[buffer(6)]],
    constant uint& total   [[buffer(7)]],
    constant uint& do_silu [[buffer(8)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    uint c = i / hw;
    uint g = c / per_g;
    float v = (x[i] - st[2u * g]) * st[2u * g + 1u] * wa[c] + ba[c];
    if (do_silu != 0u) v = v / (1.0f + exp(-v));
    y[i] = v;
}

// Nearest-neighbour ×2 upsample, NCHW.
kernel void upsample2x_k(
    device const float* x [[buffer(0)]],   // [c, h, w]
    device float*       y [[buffer(1)]],   // [c, 2h, 2w]
    constant uint& hw_in [[buffer(2)]],    // h·w
    constant uint& w_in  [[buffer(3)]],
    constant uint& total [[buffer(4)]],    // c·4·h·w
    uint i [[thread_position_in_grid]])
{
    if (i >= total) return;
    uint ci = i / (4u * hw_in);
    uint rem = i % (4u * hw_in);
    uint w2 = 2u * w_in;
    uint yy = rem / w2;
    uint xx = rem % w2;
    y[i] = x[(ulong)ci * hw_in + (ulong)(yy / 2u) * w_in + xx / 2u];
}

// [hw, oc] panel → NCHW [oc, hw] + bias.
kernel void panel_to_nchw(
    device const float* y    [[buffer(0)]],
    device float*       out  [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    constant uint& hw [[buffer(3)]],
    constant uint& oc [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= hw * oc) return;
    uint o = i / hw;
    uint p = i % hw;
    out[i] = y[(ulong)p * oc + o] + bias[o];
}

// ── whole-DiT-block kernels: the norm/modulation/residual glue that
// kept every stage bouncing back to the CPU between GEMMs. All
// f32-reduction tolerance-class vs the CPU's f64 accumulation. ──

// dst[p] = rms_norm(src[p], w, eps) · (1 + s)   (AdaLN scale, s
// optional). One 256-thread group per row.
kernel void rms_mod_rows(
    device const float* src [[buffer(0)]],
    device float*       dst [[buffer(1)]],
    device const float* w   [[buffer(2)]],
    device const float* s   [[buffer(3)]],
    constant uint&  h     [[buffer(4)]],
    constant float& eps   [[buffer(5)]],
    constant uint&  has_s [[buffer(6)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint TPT = 256u;
    device const float* row = src + (ulong)tg * h;
    device float* out = dst + (ulong)tg * h;
    threadgroup float red[256];
    float ss = 0.0f;
    for (uint i = tid; i < h; i += TPT) { float x = row[i]; ss += x * x; }
    red[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TPT >> 1u; off > 0u; off >>= 1u) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(red[0] / (float)h + eps);
    for (uint i = tid; i < h; i += TPT) {
        float v = row[i] * inv * w[i];
        if (has_s != 0u) v *= 1.0f + s[i];
        out[i] = v;
    }
}

// x[p] += gate ⊙ rms_norm(src[p], w, eps)   (gate pre-tanh'd
// host-side, optional). One 256-thread group per row.
kernel void rms_residual_rows(
    device const float* src  [[buffer(0)]],
    device float*       x    [[buffer(1)]],
    device const float* w    [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    constant uint&  h     [[buffer(4)]],
    constant float& eps   [[buffer(5)]],
    constant uint&  has_g [[buffer(6)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const uint TPT = 256u;
    device const float* row = src + (ulong)tg * h;
    device float* out = x + (ulong)tg * h;
    threadgroup float red[256];
    float ss = 0.0f;
    for (uint i = tid; i < h; i += TPT) { float v = row[i]; ss += v * v; }
    red[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TPT >> 1u; off > 0u; off >>= 1u) {
        if (tid < off) red[tid] += red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(red[0] / (float)h + eps);
    for (uint i = tid; i < h; i += TPT) {
        float v = row[i] * inv * w[i];
        out[i] += (has_g != 0u ? gate[i] : 1.0f) * v;
    }
}

// Per-(token,head) qk-norm + interleaved-pair RoPE + head-major pack
// (DiT 3-axis RoPE arrives as a precomputed per-token cos/sin table).
// One simdgroup per (token, head) row of hd.
kernel void dit_rope_pack(
    device const float* src [[buffer(0)]],  // [n][heads][hd] token-major
    device float*       dst [[buffer(1)]],  // [heads][nst][hd]
    device const float* w   [[buffer(2)]],  // [hd] rms weight
    device const float* cs  [[buffer(3)]],  // cos [n][hd/2]
    device const float* sn  [[buffer(4)]],  // sin [n][hd/2]
    constant uint&  n     [[buffer(5)]],
    constant uint&  heads [[buffer(6)]],
    constant uint&  hd    [[buffer(7)]],
    constant float& eps   [[buffer(8)]],
    constant uint&  nst   [[buffer(9)]],    // dst row stride (padded n)
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint p = tg / heads, hh = tg % heads;
    if (p >= n) return;
    device const float* v = src + ((ulong)p * heads + hh) * hd;
    float ss = 0.0f;
    for (uint i = lane; i < hd; i += 32u) { float x = v[i]; ss += x * x; }
    ss = simd_sum(ss);
    float inv = rsqrt(ss / (float)hd + eps);
    uint pairs = hd >> 1u;
    device float* d = dst + ((ulong)hh * nst + p) * hd;
    for (uint j = lane; j < pairs; j += 32u) {
        float a = v[2u * j] * inv * w[2u * j];
        float b = v[2u * j + 1u] * inv * w[2u * j + 1u];
        float c = cs[(ulong)p * pairs + j];
        float s = sn[(ulong)p * pairs + j];
        d[2u * j]      = a * c - b * s;
        d[2u * j + 1u] = a * s + b * c;
    }
}

// Plain token-major → head-major permute (V has no norm/rope).
kernel void pack_heads(
    device const float* src [[buffer(0)]],  // [n][heads][hd]
    device float*       dst [[buffer(1)]],  // [heads][nst][hd]
    constant uint& n     [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& hd    [[buffer(4)]],
    constant uint& nst   [[buffer(5)]],     // dst row stride (padded n)
    uint i [[thread_position_in_grid]])
{
    uint total = n * heads * hd;
    if (i >= total) return;
    uint p = i / (heads * hd);
    uint h = (i / hd) % heads;
    uint d = i % hd;
    dst[((ulong)h * nst + p) * hd + d] = src[i];
}

// q1: 6-byte tiles [f16 scale][4B sign bits] per 32-group; w = s*(2b-1).
// One SIMD group per FOUR rows, tiles of a pair processed one at a
// time: each activation float4 a lane loads is used against four rows'
// tiles, halving the L1 xs traffic per weight byte vs the former
// two-row kernel (the earlier four-row attempt cached the whole x
// block in registers and spilled; here only one float4 accumulator per
// row is live inside the tile loop). Tile pairs are 12 bytes = three
// aligned u32 loads; gpr must be even (CPU handles the rest).
kernel void q1_matvec(
    device const uchar*  q    [[buffer(0)]],
    device const float4* xs   [[buffer(1)]],
    device float*        y    [[buffer(2)]],
    constant uint&       gpr  [[buffer(3)]],
    constant uint&       rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    uint np = gpr >> 1;
    device const uint* q0 = (device const uint*)(q + (ulong)r0 * gpr * 6u);
    device const uint* q1p = (device const uint*)(q + (ulong)(r0 + (nr > 1u ? 1u : 0u)) * gpr * 6u);
    device const uint* q2p = (device const uint*)(q + (ulong)(r0 + (nr > 2u ? 2u : 0u)) * gpr * 6u);
    device const uint* q3p = (device const uint*)(q + (ulong)(r0 + (nr > 3u ? 3u : 0u)) * gpr * 6u);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint pidx = lane; pidx < np; pidx += 32u) {
        uint a0 = q0[pidx * 3u], a1 = q0[pidx * 3u + 1u], a2 = q0[pidx * 3u + 2u];
        uint b0 = q1p[pidx * 3u], b1 = q1p[pidx * 3u + 1u], b2 = q1p[pidx * 3u + 2u];
        uint c0 = q2p[pidx * 3u], c1 = q2p[pidx * 3u + 1u], c2 = q2p[pidx * 3u + 2u];
        uint d0 = q3p[pidx * 3u], d1 = q3p[pidx * 3u + 1u], d2 = q3p[pidx * 3u + 2u];
        ulong g = (ulong)pidx * 2u;
        // First tile of the pair: bits live in the middle of word 0/1.
        {
            uint ba = (a0 >> 16) | (a1 << 16);
            uint bb = (b0 >> 16) | (b1 << 16);
            uint bc = (c0 >> 16) | (c1 << 16);
            uint bd = (d0 >> 16) | (d1 << 16);
            float4 sA = float4(0.0f), sB = float4(0.0f);
            float4 sC = float4(0.0f), sD = float4(0.0f);
            for (uint j = 0; j < 8; ++j) {
                float4 x = xs[g * 8u + j];
                uint na = ba >> (j * 4u), nb = bb >> (j * 4u);
                uint nc = bc >> (j * 4u), nd = bd >> (j * 4u);
                sA += select(-x, x, bool4(na & 1u, na & 2u, na & 4u, na & 8u));
                sB += select(-x, x, bool4(nb & 1u, nb & 2u, nb & 4u, nb & 8u));
                sC += select(-x, x, bool4(nc & 1u, nc & 2u, nc & 4u, nc & 8u));
                sD += select(-x, x, bool4(nd & 1u, nd & 2u, nd & 4u, nd & 8u));
            }
            acc0 += (float)as_type<half>((ushort)(a0 & 0xFFFFu)) * (sA.x + sA.y + sA.z + sA.w);
            acc1 += (float)as_type<half>((ushort)(b0 & 0xFFFFu)) * (sB.x + sB.y + sB.z + sB.w);
            acc2 += (float)as_type<half>((ushort)(c0 & 0xFFFFu)) * (sC.x + sC.y + sC.z + sC.w);
            acc3 += (float)as_type<half>((ushort)(d0 & 0xFFFFu)) * (sD.x + sD.y + sD.z + sD.w);
        }
        // Second tile of the pair: bits are word 2, scale tops word 1.
        {
            float4 sA = float4(0.0f), sB = float4(0.0f);
            float4 sC = float4(0.0f), sD = float4(0.0f);
            for (uint j = 0; j < 8; ++j) {
                float4 x = xs[(g + 1u) * 8u + j];
                uint na = a2 >> (j * 4u), nb = b2 >> (j * 4u);
                uint nc = c2 >> (j * 4u), nd = d2 >> (j * 4u);
                sA += select(-x, x, bool4(na & 1u, na & 2u, na & 4u, na & 8u));
                sB += select(-x, x, bool4(nb & 1u, nb & 2u, nb & 4u, nb & 8u));
                sC += select(-x, x, bool4(nc & 1u, nc & 2u, nc & 4u, nc & 8u));
                sD += select(-x, x, bool4(nd & 1u, nd & 2u, nd & 4u, nd & 8u));
            }
            acc0 += (float)as_type<half>((ushort)(a1 >> 16)) * (sA.x + sA.y + sA.z + sA.w);
            acc1 += (float)as_type<half>((ushort)(b1 >> 16)) * (sB.x + sB.y + sB.z + sB.w);
            acc2 += (float)as_type<half>((ushort)(c1 >> 16)) * (sC.x + sC.y + sC.z + sC.w);
            acc3 += (float)as_type<half>((ushort)(d1 >> 16)) * (sD.x + sD.y + sD.z + sD.w);
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// Half-accumulation twin of q1_matvec (default; CMF_Q1_HALF=0 reverts
// to the f32 kernel): the select/add chains — this kernel's ALU wall —
// run in half4 (double-rate on Apple GPUs); each 32-group's partial
// sum converts to f32 exactly once, at the scale fma. The activation
// float4 converts to half4 once per lane iteration and serves all four
// rows. Not bit-stable vs the f32 kernel, but blessed by the gates:
// PPL identical to 3 decimals on 1.7B (23.969) and 27B (14.985),
// greedy text token-identical; decode +5% (1.7B), TTFT −5% (27B).
kernel void q1_matvec_h(
    device const uchar*  q    [[buffer(0)]],
    device const float4* xs   [[buffer(1)]],
    device float*        y    [[buffer(2)]],
    constant uint&       gpr  [[buffer(3)]],
    constant uint&       rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    uint np = gpr >> 1;
    device const uint* q0 = (device const uint*)(q + (ulong)r0 * gpr * 6u);
    device const uint* q1p = (device const uint*)(q + (ulong)(r0 + (nr > 1u ? 1u : 0u)) * gpr * 6u);
    device const uint* q2p = (device const uint*)(q + (ulong)(r0 + (nr > 2u ? 2u : 0u)) * gpr * 6u);
    device const uint* q3p = (device const uint*)(q + (ulong)(r0 + (nr > 3u ? 3u : 0u)) * gpr * 6u);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint pidx = lane; pidx < np; pidx += 32u) {
        uint a0 = q0[pidx * 3u], a1 = q0[pidx * 3u + 1u], a2 = q0[pidx * 3u + 2u];
        uint b0 = q1p[pidx * 3u], b1 = q1p[pidx * 3u + 1u], b2 = q1p[pidx * 3u + 2u];
        uint c0 = q2p[pidx * 3u], c1 = q2p[pidx * 3u + 1u], c2 = q2p[pidx * 3u + 2u];
        uint d0 = q3p[pidx * 3u], d1 = q3p[pidx * 3u + 1u], d2 = q3p[pidx * 3u + 2u];
        ulong g = (ulong)pidx * 2u;
        {
            uint ba = (a0 >> 16) | (a1 << 16);
            uint bb = (b0 >> 16) | (b1 << 16);
            uint bc = (c0 >> 16) | (c1 << 16);
            uint bd = (d0 >> 16) | (d1 << 16);
            half4 sA = half4(0.0h), sB = half4(0.0h);
            half4 sC = half4(0.0h), sD = half4(0.0h);
            for (uint j = 0; j < 8; ++j) {
                half4 x = half4(xs[g * 8u + j]);
                uint na = ba >> (j * 4u), nb = bb >> (j * 4u);
                uint nc = bc >> (j * 4u), nd = bd >> (j * 4u);
                sA += select(-x, x, bool4(na & 1u, na & 2u, na & 4u, na & 8u));
                sB += select(-x, x, bool4(nb & 1u, nb & 2u, nb & 4u, nb & 8u));
                sC += select(-x, x, bool4(nc & 1u, nc & 2u, nc & 4u, nc & 8u));
                sD += select(-x, x, bool4(nd & 1u, nd & 2u, nd & 4u, nd & 8u));
            }
            acc0 += (float)as_type<half>((ushort)(a0 & 0xFFFFu)) * (float)(sA.x + sA.y + sA.z + sA.w);
            acc1 += (float)as_type<half>((ushort)(b0 & 0xFFFFu)) * (float)(sB.x + sB.y + sB.z + sB.w);
            acc2 += (float)as_type<half>((ushort)(c0 & 0xFFFFu)) * (float)(sC.x + sC.y + sC.z + sC.w);
            acc3 += (float)as_type<half>((ushort)(d0 & 0xFFFFu)) * (float)(sD.x + sD.y + sD.z + sD.w);
        }
        {
            half4 sA = half4(0.0h), sB = half4(0.0h);
            half4 sC = half4(0.0h), sD = half4(0.0h);
            for (uint j = 0; j < 8; ++j) {
                half4 x = half4(xs[(g + 1u) * 8u + j]);
                uint na = a2 >> (j * 4u), nb = b2 >> (j * 4u);
                uint nc = c2 >> (j * 4u), nd = d2 >> (j * 4u);
                sA += select(-x, x, bool4(na & 1u, na & 2u, na & 4u, na & 8u));
                sB += select(-x, x, bool4(nb & 1u, nb & 2u, nb & 4u, nb & 8u));
                sC += select(-x, x, bool4(nc & 1u, nc & 2u, nc & 4u, nc & 8u));
                sD += select(-x, x, bool4(nd & 1u, nd & 2u, nd & 4u, nd & 8u));
            }
            acc0 += (float)as_type<half>((ushort)(a1 >> 16)) * (float)(sA.x + sA.y + sA.z + sA.w);
            acc1 += (float)as_type<half>((ushort)(b1 >> 16)) * (float)(sB.x + sB.y + sB.z + sB.w);
            acc2 += (float)as_type<half>((ushort)(c1 >> 16)) * (float)(sC.x + sC.y + sC.z + sC.w);
            acc3 += (float)as_type<half>((ushort)(d1 >> 16)) * (float)(sD.x + sD.y + sD.z + sD.w);
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

kernel void silu_mul_pre(
    device const float* g   [[buffer(0)]],
    device const float* u   [[buffer(1)]],
    device const float* col [[buffer(2)]],
    device float*       act [[buffer(3)]],
    constant uint&      n   [[buffer(4)]],
    constant uint&      has_col [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float gv = g[i];
    float cv = has_col != 0 ? col[i] : 1.0f;
    act[i] = (gv / (1.0f + exp(-gv))) * u[i] * cv;
}

// Full attention on the device — one simdgroup per head throughout.
// Dims contract (checked host-side): hd % 4 == 0, hd <= 256, and for
// RoPE lane-local pairing (rd/2) % 32 == 0 with rd <= hd.

// Per-head qk-norm + partial RoPE. Heads 0..nh are Q (optionally
// [q(hd); gate(hd)] interleaved in qraw), heads nh..nh+nkv are K rows
// normed+rotated in place. The gate half is copied out untouched
// (it is applied after the attend, sigmoid-gated).
kernel void attn_rope_qkn(
    device const float* qraw [[buffer(0)]],
    device float*       k    [[buffer(1)]],
    device float*       qout [[buffer(2)]],
    device float*       gout [[buffer(3)]],
    device const float* qnw  [[buffer(4)]],
    device const float* knw  [[buffer(5)]],
    device const float* invf [[buffer(6)]],
    constant uint&  nh    [[buffer(7)]],
    constant uint&  nkv   [[buffer(8)]],
    constant uint&  hd    [[buffer(9)]],
    constant uint&  rd    [[buffer(10)]],
    constant uint&  pos   [[buffer(11)]],
    constant uint&  flags [[buffer(12)]], // 1=gate 2=qnorm 4=knorm 8=gemma
    constant float& eps   [[buffer(13)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]])
{
    uint head = tg * sgs + sg;
    if (head >= nh + nkv) return;
    bool isq = head < nh;
    bool gate = (flags & 1u) != 0u;
    device const float* src = isq
        ? qraw + (ulong)head * (gate ? 2u : 1u) * hd
        : k + (ulong)(head - nh) * hd;
    uint nt = (hd + 31u) / 32u;
    float xv[8];
    float ss = 0.0f;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        xv[t] = d < hd ? src[d] : 0.0f;
        ss += xv[t] * xv[t];
    }
    ss = simd_sum(ss);
    bool normed = isq ? (flags & 2u) != 0u : (flags & 4u) != 0u;
    if (normed) {
        float inv = 1.0f / sqrt(ss / (float)hd + eps);
        device const float* w = isq ? qnw : knw;
        bool gemma = (flags & 8u) != 0u;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) {
                float wd = w[d];
                xv[t] = xv[t] * inv * (gemma ? (1.0f + wd) : wd);
            }
        }
    }
    // Partial RoPE: pair (i, i + rd/2); with (rd/2) % 32 == 0 both
    // halves live in the same lane, slots t and t + (rd/2)/32.
    uint hlf = rd / 2u;
    uint toff = hlf / 32u;
    for (uint t = 0; t < toff; ++t) {
        uint i = t * 32u + lane;
        if (i < hlf) {
            float angle = (float)pos * invf[i];
            float c = cos(angle), s = sin(angle);
            float x0 = xv[t], x1 = xv[t + toff];
            xv[t] = x0 * c - x1 * s;
            xv[t + toff] = x0 * s + x1 * c;
        }
    }
    device float* dst = isq ? qout + (ulong)head * hd : k + (ulong)(head - nh) * hd;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) dst[d] = xv[t];
    }
    if (isq && gate) {
        device const float* gsrc = qraw + (ulong)head * 2u * hd + hd;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) gout[(ulong)head * hd + d] = gsrc[d];
        }
    }
}

// Append this position's K/V rows into the device cache mirror
// ([nkv, cap, hd] each) at index `stored`.
kernel void kv_append(
    device const float* k    [[buffer(0)]],
    device const float* v    [[buffer(1)]],
    device float*       kbuf [[buffer(2)]],
    device float*       vbuf [[buffer(3)]],
    constant uint& nkv    [[buffer(4)]],
    constant uint& hd     [[buffer(5)]],
    constant uint& cap    [[buffer(6)]],
    constant uint& stored [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= nkv * hd) return;
    uint h = i / hd, d = i % hd;
    ulong dst = ((ulong)h * cap + stored) * hd + d;
    kbuf[dst] = k[i];
    vbuf[dst] = v[i];
}

// Grouped decode attention, flash-decoding shape: the threadgroup owns
// ONE Q-head and its `sgs` simdgroups split the stored positions between
// them, each running an online softmax over its own slice (lane-sliced
// dims, dim d lives in lane d%32 slot d/32); the partials are then
// combined through threadgroup memory. One simdgroup per head — the
// shape this replaced — put only nh simdgroups on the whole device (48
// for Nanbeige 4.2), nowhere near enough to hide the per-position
// simd_sum latency chain, so decode fell off a cliff with context depth.
// A second pass banks each position's probability mass into the
// Born-importance accumulator (the default eviction policy ranks by it).
// exp/order differ from the CPU attend (tolerance-gated, like every GPU
// reduction here).
kernel void gqa_attend(
    device const float* q    [[buffer(0)]],
    device const float* kbuf [[buffer(1)]],
    device const float* vbuf [[buffer(2)]],
    device float*       outb [[buffer(3)]],
    device atomic_float* imp [[buffer(4)]],
    constant uint& nh  [[buffer(5)]],
    constant uint& hpk [[buffer(6)]],
    constant uint& hd  [[buffer(7)]],
    constant uint& cap [[buffer(8)]],
    constant uint& n   [[buffer(9)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]])
{
    uint h = tg;
    if (h >= nh) return;
    uint kh = h / hpk;
    device const float* kh0 = kbuf + (ulong)kh * cap * hd;
    device const float* vh0 = vbuf + (ulong)kh * cap * hd;
    float scale = 1.0f / sqrt((float)hd);
    uint nt = (hd + 31u) / 32u;
    float qv[8];
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        qv[t] = d < hd ? q[(ulong)h * hd + d] * scale : 0.0f;
    }
    // This simdgroup's slice of the stored positions. Contiguous, so
    // the K/V walk stays sequential inside each slice.
    uint per = (n + sgs - 1u) / sgs;
    uint p0 = min(sg * per, n);
    uint p1 = min(p0 + per, n);
    float m = -INFINITY, l = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint p = p0; p < p1; ++p) {
        device const float* kr = kh0 + (ulong)p * hd;
        float partial = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) partial += qv[t] * kr[d];
        }
        float s = simd_sum(partial);
        float mp = max(m, s);
        float f = exp(m - mp), w = exp(s - mp);
        l = l * f + w;
        device const float* vr = vh0 + (ulong)p * hd;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) acc[t] = acc[t] * f + w * vr[d];
        }
        m = mp;
    }
    // Combine the slices: sh = [sgs × hd accumulators | sgs m | sgs l].
    // An empty slice contributes m = -INF, l = 0, acc = 0 — exp(-INF −
    // gm) = 0 kills it in both sums, and n ≥ 1 keeps gm finite.
    threadgroup float* sacc = sh;
    threadgroup float* sm = sh + sgs * hd;
    threadgroup float* sl = sm + sgs;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) sacc[sg * hd + d] = acc[t];
    }
    if (lane == 0) {
        sm[sg] = m;
        sl[sg] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = -INFINITY;
    for (uint s = 0; s < sgs; ++s) gm = max(gm, sm[s]);
    float gl = 0.0f;
    for (uint s = 0; s < sgs; ++s) gl += sl[s] * exp(sm[s] - gm);
    float invl = gl > 0.0f ? 1.0f / gl : 0.0f;
    if (sg == 0) {
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) {
                float a = 0.0f;
                for (uint s = 0; s < sgs; ++s) a += sacc[s * hd + d] * exp(sm[s] - gm);
                outb[(ulong)h * hd + d] = a * invl;
            }
        }
    }
    m = gm; // the importance pass below wants the head's final max
    // Born-importance pass: prob_p = exp(s_p − m)/l summed over heads.
    // The score is recomputed in the SAME lane-sliced layout as the main
    // loop. The obvious form — one position per lane, each lane walking
    // a whole K row — makes every lane touch a different 512 B row, so
    // the reads never coalesce: it cost more than the whole rest of the
    // kernel at decode depth (M4, 44 virtual layers: 0.145 ms/position
    // of context vs 0.003 ms bandwidth-bound). Four positions per step
    // so the simd_sum chains overlap; `qv` already carries `scale`.
    // Each simdgroup re-walks its own slice, now with the head's final
    // m/l — the slices tile [0, n), so every position is banked once.
    uint p = p0;
    for (; p + 4u <= p1; p += 4u) {
        device const float* r0 = kh0 + (ulong)p * hd;
        device const float* r1 = r0 + hd;
        device const float* r2 = r1 + hd;
        device const float* r3 = r2 + hd;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) {
                float qd = qv[t];
                a0 += qd * r0[d];
                a1 += qd * r1[d];
                a2 += qd * r2[d];
                a3 += qd * r3[d];
            }
        }
        float s0 = simd_sum(a0), s1 = simd_sum(a1);
        float s2 = simd_sum(a2), s3 = simd_sum(a3);
        if (lane == 0) {
            atomic_fetch_add_explicit(&imp[p], exp(s0 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 1u], exp(s1 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 2u], exp(s2 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 3u], exp(s3 - m) * invl, memory_order_relaxed);
        }
    }
    for (; p < p1; ++p) {
        device const float* kr = kh0 + (ulong)p * hd;
        float a = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) a += qv[t] * kr[d];
        }
        float s = simd_sum(a);
        if (lane == 0) {
            atomic_fetch_add_explicit(&imp[p], exp(s - m) * invl, memory_order_relaxed);
        }
    }
}

// Chunk (prefill) attend: gqa_attend batched over the chunk's query
// positions with the causal bound — query bi sees cache rows
// 0 .. s0+bi. One simdgroup per (query, head), online softmax, the
// same Born-importance second pass accumulated atomically across every
// query and head (matching the CPU chunk path's masked column sums).
// The chunk's own K/V rows must already sit in the mirror.
//
// TWO MEASURED DEAD ENDS on M4 (kept away from):
// - flash-TILED (8 queries sharing 16 KB staged K/V): pp512 1750→1680,
//   pp2048 937→783 — a layer's K/V fits UMA L2, so per-query device
//   reads were already cached and tiles only added barriers.
// - split-K (8 simdgroups per query over row segments + flash-decoding
//   combine): pp512 1800→1690, pp2048 949→825 — the softmax chain per
//   query was NOT the wall either; the plain streaming loop with no
//   barriers and no combine is simply the fastest form here.
// The pp2048 depth wall therefore stands (deep chunks fall back to the
// CPU GEMM-attend via the pos0 bound in the pipeline).
kernel void chunk_attend(
    device const float* q    [[buffer(0)]],   // [nb, nh, hd] post-rope
    device const float* kbuf [[buffer(1)]],
    device const float* vbuf [[buffer(2)]],
    device float*       outb [[buffer(3)]],   // [nb, nh, hd]
    device atomic_float* imp [[buffer(4)]],
    constant uint& nh  [[buffer(5)]],
    constant uint& hpk [[buffer(6)]],
    constant uint& hd  [[buffer(7)]],
    constant uint& cap [[buffer(8)]],
    constant uint& s0  [[buffer(9)]],
    constant uint& nb  [[buffer(10)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]])
{
    uint h = tg.x * sgs + sg;
    uint bi = tg.y;
    if (h >= nh || bi >= nb) return;
    uint n = s0 + bi + 1u;
    uint kh = h / hpk;
    device const float* kh0 = kbuf + (ulong)kh * cap * hd;
    device const float* vh0 = vbuf + (ulong)kh * cap * hd;
    device const float* qh = q + ((ulong)bi * nh + h) * hd;
    float scale = 1.0f / sqrt((float)hd);
    uint nt = (hd + 31u) / 32u;
    float qv[8];
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        qv[t] = d < hd ? qh[d] * scale : 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint p = 0; p < n; ++p) {
        device const float* kr = kh0 + (ulong)p * hd;
        float partial = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) partial += qv[t] * kr[d];
        }
        float sv = simd_sum(partial);
        float mp = max(m, sv);
        float f = exp(m - mp), w = exp(sv - mp);
        l = l * f + w;
        device const float* vr = vh0 + (ulong)p * hd;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) acc[t] = acc[t] * f + w * vr[d];
        }
        m = mp;
    }
    float invl = l > 0.0f ? 1.0f / l : 0.0f;
    device float* oh = outb + ((ulong)bi * nh + h) * hd;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) oh[d] = acc[t] * invl;
    }
    // Born importance, lane-sliced like the main loop (see gqa_attend:
    // the per-lane serial dot reads the mirror uncoalesced and dominated
    // the whole chunk). Four positions per step for reduction ILP.
    uint p = 0;
    for (; p + 4u <= n; p += 4u) {
        device const float* r0 = kh0 + (ulong)p * hd;
        device const float* r1 = r0 + hd;
        device const float* r2 = r1 + hd;
        device const float* r3 = r2 + hd;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) {
                float qd = qv[t];
                a0 += qd * r0[d];
                a1 += qd * r1[d];
                a2 += qd * r2[d];
                a3 += qd * r3[d];
            }
        }
        float s0 = simd_sum(a0), s1 = simd_sum(a1);
        float s2 = simd_sum(a2), s3 = simd_sum(a3);
        if (lane == 0) {
            atomic_fetch_add_explicit(&imp[p], exp(s0 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 1u], exp(s1 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 2u], exp(s2 - m) * invl, memory_order_relaxed);
            atomic_fetch_add_explicit(&imp[p + 3u], exp(s3 - m) * invl, memory_order_relaxed);
        }
    }
    for (; p < n; ++p) {
        device const float* kr = kh0 + (ulong)p * hd;
        float a = 0.0f;
        for (uint t = 0; t < nt; ++t) {
            uint d = t * 32u + lane;
            if (d < hd) a += qv[t] * kr[d];
        }
        float s = simd_sum(a);
        if (lane == 0) {
            atomic_fetch_add_explicit(&imp[p], exp(s - m) * invl, memory_order_relaxed);
        }
    }
}

// a *= sigmoid(g) — the Qwen3.5 attention output gate.
kernel void sig_gate(
    device float*       a [[buffer(0)]],
    device const float* g [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    a[i] = a[i] / (1.0f + exp(-g[i]));
}

kernel void axpy(
    device const float* d [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant float&     w [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    y[i] += w * d[i];
}

kernel void fill_zero(
    device float*  y [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) y[i] = 0.0f;
}

// Completion flag: the LAST encoder of every command buffer writes a
// monotone ticket into a shared buffer; the CPU spins on that word
// directly (UMA) instead of the driver's completion machinery, which
// costs ~1.3 ms per round trip. Reading every output buffer makes Metal
// order this pass after ALL producing passes (hazard tracking) —
// independent batch jobs may otherwise still be in flight when the
// flag lands. Unused slots are bound to y0.
kernel void write_flag(
    device const float* y0 [[buffer(0)]],
    device const float* y1 [[buffer(1)]],
    device const float* y2 [[buffer(2)]],
    device const float* y3 [[buffer(3)]],
    device atomic_uint* f  [[buffer(4)]],
    constant uint&      v  [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i == 0) {
        float probe = y0[0] + y1[0] + y2[0] + y3[0];
        uint bump = (probe == 123456789.0f) ? 1u : 0u; // never true: forces the reads
        atomic_store_explicit(f, v + bump, memory_order_relaxed);
    }
}

// ── Whole-block GDN kernels: an entire linear layer (norm → mixer →
// conv → recurrence → out_proj → norm → FFN) runs inside ONE command
// buffer, hidden state resident on device; the CPU sees one sync per
// BLOCK of consecutive GDN layers instead of ~12 per layer. ──

// Tiny f32 matvec (the GDN a/b gate projections live dequantized in
// RAM; they are uploaded once through the small-vector cache).
kernel void f32_matvec(
    device const float*  q    [[buffer(0)]],
    device const float*  xs   [[buffer(1)]],
    device float*        y    [[buffer(2)]],
    constant uint&       cols [[buffer(3)]],
    constant uint&       rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint row = tgpos * sgs + sg;
    if (row >= rows) return;
    ulong base = (ulong)row * cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += 32u) {
        acc += q[base + i] * xs[i];
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

kernel void rmsnorm_k(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float*       o [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    constant uint&  gemma [[buffer(4)]],
    constant float&   eps [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float part[8];
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256u) { float v = x[i]; acc += v * v; }
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k = 0; k < 8u; ++k) tot += part[k];
    float inv = rsqrt(tot / (float)n + eps);
    for (uint i = tid; i < n; i += 256u) {
        float wv = gemma != 0u ? (1.0f + w[i]) : w[i];
        o[i] = x[i] * inv * wv;
    }
}

// Embedding gather for the chunk graph: h[bi] = dequant(embed[ids[bi]])
// · multiplier — the 512 per-position CPU dequants and the h upload
// disappear.
kernel void embed_q8_rows(
    device const char*  q    [[buffer(0)]],
    device const float* rs   [[buffer(1)]],
    device const uint*  ids  [[buffer(2)]],
    device float*       h    [[buffer(3)]],
    constant uint&      hs   [[buffer(4)]],
    constant uint&      nb   [[buffer(5)]],
    constant float&     mult [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint d = gid.x;
    uint bi = gid.y;
    if (d >= hs || bi >= nb) return;
    uint id = ids[bi];
    h[(ulong)bi * hs + d] = (float)q[(ulong)id * hs + d] * rs[id] * mult;
}

// rmsnorm_k over a batch: one threadgroup per row.
kernel void rmsnorm_rows(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float*       o [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    constant uint&  gemma [[buffer(4)]],
    constant float&   eps [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint row  [[threadgroup_position_in_grid]])
{
    threadgroup float part[8];
    device const float* xr = x + (ulong)row * n;
    device float* orow = o + (ulong)row * n;
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256u) { float v = xr[i]; acc += v * v; }
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k = 0; k < 8u; ++k) tot += part[k];
    float inv = rsqrt(tot / (float)n + eps);
    for (uint i = tid; i < n; i += 256u) {
        float wv = gemma != 0u ? (1.0f + w[i]) : w[i];
        orow[i] = xr[i] * inv * wv;
    }
}

// Fused residual add + row RMSNorm: h += delta (in place), then
// o = rms(h)·w — one pass instead of an axpy encoder and a norm
// encoder back-to-back over the same rows.
kernel void add_rmsnorm_rows(
    device float*       h [[buffer(0)]],
    device const float* d [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device float*       o [[buffer(3)]],
    constant uint&      n [[buffer(4)]],
    constant uint&  gemma [[buffer(5)]],
    constant float&   eps [[buffer(6)]],
    constant uint&  hasd  [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint row  [[threadgroup_position_in_grid]])
{
    threadgroup float part[8];
    device float* hr = h + (ulong)row * n;
    device const float* dr = d + (ulong)row * n;
    device float* orow = o + (ulong)row * n;
    float acc = 0.0f;
    for (uint i = tid; i < n; i += 256u) {
        float v = hr[i] + (hasd != 0u ? dr[i] : 0.0f);
        hr[i] = v;
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k = 0; k < 8u; ++k) tot += part[k];
    float inv = rsqrt(tot / (float)n + eps);
    for (uint i = tid; i < n; i += 256u) {
        float wv = gemma != 0u ? (1.0f + w[i]) : w[i];
        orow[i] = hr[i] * inv * wv;
    }
}

// Chunk QKV finish: bias add + optional per-head qk-norm + RoPE at
// pos0+bi, K/V written STRAIGHT into the cache mirror at stored0+bi
// (fuses kv_append for the whole chunk). Head space: [0, nh) = Q,
// [nh, nh+nkv) = K, [nh+nkv, nh+2·nkv) = V (bias only). One simdgroup
// per (head, position). flags: 2=qnorm 4=knorm 8=gemma-norm 16=bias.
kernel void chunk_rope_kv(
    device const float* qraw [[buffer(0)]],   // [nb, nh·hd]
    device const float* kraw [[buffer(1)]],   // [nb, nkv·hd]
    device const float* vraw [[buffer(2)]],   // [nb, nkv·hd]
    device float*       qout [[buffer(3)]],   // [nb, nh, hd]
    device float*       kbuf [[buffer(4)]],
    device float*       vbuf [[buffer(5)]],
    device const float* bq   [[buffer(6)]],
    device const float* bk   [[buffer(7)]],
    device const float* bv   [[buffer(8)]],
    device const float* qnw  [[buffer(9)]],
    device const float* knw  [[buffer(10)]],
    device const float* invf [[buffer(11)]],
    constant uint&  nh    [[buffer(12)]],
    constant uint&  nkv   [[buffer(13)]],
    constant uint&  hd    [[buffer(14)]],
    constant uint&  rd    [[buffer(15)]],
    constant uint&  pos0  [[buffer(16)]],
    constant uint&  st0   [[buffer(17)]],
    constant uint&  cap   [[buffer(18)]],
    constant uint&  flags [[buffer(19)]],
    constant float& eps   [[buffer(20)]],
    constant uint&  nb    [[buffer(21)]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]])
{
    uint head = tg.x * sgs + sg;
    uint bi = tg.y;
    if (head >= nh + 2u * nkv || bi >= nb) return;
    bool isq = head < nh;
    bool isv = head >= nh + nkv;
    uint kvh = isv ? head - nh - nkv : head - nh;
    bool bias = (flags & 16u) != 0u;
    device const float* src = isq
        ? qraw + (ulong)bi * nh * hd + (ulong)head * hd
        : (isv ? vraw : kraw) + (ulong)bi * nkv * hd + (ulong)kvh * hd;
    device const float* brow = isq ? bq : (isv ? bv : bk);
    uint nt = (hd + 31u) / 32u;
    float xv[8];
    float ss = 0.0f;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        float v = d < hd ? src[d] : 0.0f;
        if (bias && d < hd) v += brow[(isq ? (ulong)head : (ulong)kvh) * hd + d];
        xv[t] = v;
        ss += v * v;
    }
    if (!isv) {
        ss = simd_sum(ss);
        bool normed = isq ? (flags & 2u) != 0u : (flags & 4u) != 0u;
        if (normed) {
            float inv = 1.0f / sqrt(ss / (float)hd + eps);
            device const float* w = isq ? qnw : knw;
            bool gm = (flags & 8u) != 0u;
            for (uint t = 0; t < nt; ++t) {
                uint d = t * 32u + lane;
                if (d < hd) {
                    float wd = w[d];
                    xv[t] = xv[t] * inv * (gm ? (1.0f + wd) : wd);
                }
            }
        }
        uint hlf = rd / 2u;
        uint toff = hlf / 32u;
        uint pos = pos0 + bi;
        for (uint t = 0; t < toff; ++t) {
            uint i = t * 32u + lane;
            if (i < hlf) {
                float angle = (float)pos * invf[i];
                float c = cos(angle), sn = sin(angle);
                float x0 = xv[t], x1 = xv[t + toff];
                xv[t] = x0 * c - x1 * sn;
                xv[t + toff] = x0 * sn + x1 * c;
            }
        }
    }
    // Q lands head-major ([head][bi][hd]) — the group panel the scores
    // GEMM consumes without a gather.
    device float* dst = isq
        ? qout + ((ulong)head * nb + bi) * hd
        : (isv ? vbuf : kbuf) + ((ulong)kvh * cap + st0 + bi) * hd;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) dst[d] = xv[t];
    }
}

// cq = silu(depthwise causal conv over [ring…, current qkv])
kernel void gdn_conv(
    device const float* qkv  [[buffer(0)]],
    device const float* ring [[buffer(1)]],
    device const float* taps [[buffer(2)]],
    device float*       cq   [[buffer(3)]],
    constant uint&     c_dim [[buffer(4)]],
    constant uint&        kk [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= c_dim) return;
    float acc = qkv[i] * taps[i * kk + kk - 1u];
    for (uint j = 0; j + 1u < kk; ++j) acc += ring[j * c_dim + i] * taps[i * kk + j];
    cq[i] = acc / (1.0f + exp(-acc));
}

// Ring shift: drop the oldest position, append the RAW current qkv.
kernel void gdn_ring_shift(
    device float*       ring [[buffer(0)]],
    device const float* qkv  [[buffer(1)]],
    constant uint&     c_dim [[buffer(2)]],
    constant uint&        kk [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= c_dim) return;
    for (uint j = 0; j + 2u < kk; ++j) ring[j * c_dim + i] = ring[(j + 1u) * c_dim + i];
    ring[(kk - 2u) * c_dim + i] = qkv[i];
}

// Per-head decay g and write strength beta.
kernel void gdn_gates(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* a_log   [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device float*       g       [[buffer(4)]],
    device float*       beta    [[buffer(5)]],
    constant uint&      nv      [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= nv) return;
    float x = a[i] + dt_bias[i];
    float sp = x > 20.0f ? x : log(1.0f + exp(x));
    g[i] = exp(-exp(a_log[i]) * sp);
    beta[i] = 1.0f / (1.0f + exp(-b[i]));
}

// l2-norm inverses of q/k per K head (one simdgroup per head).
kernel void gdn_qk_norms(
    device const float* cq   [[buffer(0)]],
    device float*       invq [[buffer(1)]],
    device float*       invk [[buffer(2)]],
    constant uint&      nk   [[buffer(3)]],
    constant uint&      dk   [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg   [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint h = tg * sgs + sg;
    if (h >= nk) return;
    uint kd = nk * dk;
    float nq = 0.0f, nkn = 0.0f;
    for (uint d = lane; d < dk; d += 32u) {
        float q = cq[h * dk + d];      nq  += q * q;
        float k = cq[kd + h * dk + d]; nkn += k * k;
    }
    nq = simd_sum(nq); nkn = simd_sum(nkn);
    if (lane == 0) {
        invq[h] = 1.0f / (sqrt(nq + 1e-6f) * sqrt((float)dk));
        invk[h] = 1.0f / sqrt(nkn + 1e-6f);
    }
}

// The GatedDeltaNet recurrence + gated RMSNorm, one threadgroup per V
// head (dv threads, thread dj owns one output column):
//   kv = k'ᵀ S_old;  Δ = β(v − g·kv);  S = g·S_old + k' ⊗ Δ;  o = q'ᵀ S
// S rows are read coalesced (threads span dj).
kernel void gdn_state_update(
    device float*       S     [[buffer(0)]],
    device const float* cq    [[buffer(1)]],
    device const float* z     [[buffer(2)]],
    device const float* g     [[buffer(3)]],
    device const float* beta  [[buffer(4)]],
    device const float* invq  [[buffer(5)]],
    device const float* invk  [[buffer(6)]],
    device const float* gnorm [[buffer(7)]],
    device float*       of    [[buffer(8)]],
    constant uint&      nv    [[buffer(9)]],
    constant uint&      nk    [[buffer(10)]],
    constant uint&      dk    [[buffer(11)]],
    constant uint&      dv    [[buffer(12)]],
    constant float&     eps   [[buffer(13)]],
    uint h    [[threadgroup_position_in_grid]],
    uint dj   [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    uint rep = nv / nk;
    uint ko = h / rep;
    uint kd = nk * dk;
    device float* s = S + (ulong)h * dk * dv;
    float gh = g[h];
    float bh = beta[h];
    float iq = invq[ko];
    float ik = invk[ko];
    float vt = cq[2u * kd + h * dv + dj];
    float kv = 0.0f;
    for (uint di = 0; di < dk; ++di) {
        kv += cq[kd + ko * dk + di] * ik * s[di * dv + dj];
    }
    float delta = (vt - gh * kv) * bh;
    float o = 0.0f;
    for (uint di = 0; di < dk; ++di) {
        float kf = cq[kd + ko * dk + di] * ik;
        float qf = cq[ko * dk + di] * iq;
        float cell = gh * s[di * dv + dj] + kf * delta;
        s[di * dv + dj] = cell;
        o += qf * cell;
    }
    threadgroup float part[32];
    float ss = simd_sum(o * o);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint k2 = 0; k2 < (dv + 31u) / 32u; ++k2) tot += part[k2];
    float inv = rsqrt(tot / (float)dv + eps);
    float zz = z[h * dv + dj];
    of[h * dv + dj] = o * inv * gnorm[dj] * (zz / (1.0f + exp(-zz)));
}

// q1t: 9-byte tiles [f16 scale][7B base-3 codes, 5 ternary/byte] per 32-group;
// code 0->0, 1->+s, 2->-s. This computes the BASE dot only (raw f32 x, full
// precision); the sparse outlier overlay is added on the CPU (the base code at
// every overlay position is 0, so there is no double count). 4 rows/simdgroup.
constant half Q1T_SIGN[1280] = {
    0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 1.0h, 0.0h, 0.0h, 0.0h, 0.0h,
    -1.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 1.0h, 0.0h, 0.0h, 0.0h,
    1.0h, 1.0h, 0.0h, 0.0h, 0.0h, -1.0h, 1.0h, 0.0h, 0.0h, 0.0h,
    0.0h, -1.0h, 0.0h, 0.0h, 0.0h, 1.0h, -1.0h, 0.0h, 0.0h, 0.0h,
    -1.0h, -1.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 1.0h, 0.0h, 0.0h,
    1.0h, 0.0h, 1.0h, 0.0h, 0.0h, -1.0h, 0.0h, 1.0h, 0.0h, 0.0h,
    0.0h, 1.0h, 1.0h, 0.0h, 0.0h, 1.0h, 1.0h, 1.0h, 0.0h, 0.0h,
    -1.0h, 1.0h, 1.0h, 0.0h, 0.0h, 0.0h, -1.0h, 1.0h, 0.0h, 0.0h,
    1.0h, -1.0h, 1.0h, 0.0h, 0.0h, -1.0h, -1.0h, 1.0h, 0.0h, 0.0h,
    0.0h, 0.0h, -1.0h, 0.0h, 0.0h, 1.0h, 0.0h, -1.0h, 0.0h, 0.0h,
    -1.0h, 0.0h, -1.0h, 0.0h, 0.0h, 0.0h, 1.0h, -1.0h, 0.0h, 0.0h,
    1.0h, 1.0h, -1.0h, 0.0h, 0.0h, -1.0h, 1.0h, -1.0h, 0.0h, 0.0h,
    0.0h, -1.0h, -1.0h, 0.0h, 0.0h, 1.0h, -1.0h, -1.0h, 0.0h, 0.0h,
    -1.0h, -1.0h, -1.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 1.0h, 0.0h,
    1.0h, 0.0h, 0.0h, 1.0h, 0.0h, -1.0h, 0.0h, 0.0h, 1.0h, 0.0h,
    0.0h, 1.0h, 0.0h, 1.0h, 0.0h, 1.0h, 1.0h, 0.0h, 1.0h, 0.0h,
    -1.0h, 1.0h, 0.0h, 1.0h, 0.0h, 0.0h, -1.0h, 0.0h, 1.0h, 0.0h,
    1.0h, -1.0h, 0.0h, 1.0h, 0.0h, -1.0h, -1.0h, 0.0h, 1.0h, 0.0h,
    0.0h, 0.0h, 1.0h, 1.0h, 0.0h, 1.0h, 0.0h, 1.0h, 1.0h, 0.0h,
    -1.0h, 0.0h, 1.0h, 1.0h, 0.0h, 0.0h, 1.0h, 1.0h, 1.0h, 0.0h,
    1.0h, 1.0h, 1.0h, 1.0h, 0.0h, -1.0h, 1.0h, 1.0h, 1.0h, 0.0h,
    0.0h, -1.0h, 1.0h, 1.0h, 0.0h, 1.0h, -1.0h, 1.0h, 1.0h, 0.0h,
    -1.0h, -1.0h, 1.0h, 1.0h, 0.0h, 0.0h, 0.0h, -1.0h, 1.0h, 0.0h,
    1.0h, 0.0h, -1.0h, 1.0h, 0.0h, -1.0h, 0.0h, -1.0h, 1.0h, 0.0h,
    0.0h, 1.0h, -1.0h, 1.0h, 0.0h, 1.0h, 1.0h, -1.0h, 1.0h, 0.0h,
    -1.0h, 1.0h, -1.0h, 1.0h, 0.0h, 0.0h, -1.0h, -1.0h, 1.0h, 0.0h,
    1.0h, -1.0h, -1.0h, 1.0h, 0.0h, -1.0h, -1.0h, -1.0h, 1.0h, 0.0h,
    0.0h, 0.0h, 0.0h, -1.0h, 0.0h, 1.0h, 0.0h, 0.0h, -1.0h, 0.0h,
    -1.0h, 0.0h, 0.0h, -1.0h, 0.0h, 0.0h, 1.0h, 0.0h, -1.0h, 0.0h,
    1.0h, 1.0h, 0.0h, -1.0h, 0.0h, -1.0h, 1.0h, 0.0h, -1.0h, 0.0h,
    0.0h, -1.0h, 0.0h, -1.0h, 0.0h, 1.0h, -1.0h, 0.0h, -1.0h, 0.0h,
    -1.0h, -1.0h, 0.0h, -1.0h, 0.0h, 0.0h, 0.0h, 1.0h, -1.0h, 0.0h,
    1.0h, 0.0h, 1.0h, -1.0h, 0.0h, -1.0h, 0.0h, 1.0h, -1.0h, 0.0h,
    0.0h, 1.0h, 1.0h, -1.0h, 0.0h, 1.0h, 1.0h, 1.0h, -1.0h, 0.0h,
    -1.0h, 1.0h, 1.0h, -1.0h, 0.0h, 0.0h, -1.0h, 1.0h, -1.0h, 0.0h,
    1.0h, -1.0h, 1.0h, -1.0h, 0.0h, -1.0h, -1.0h, 1.0h, -1.0h, 0.0h,
    0.0h, 0.0h, -1.0h, -1.0h, 0.0h, 1.0h, 0.0h, -1.0h, -1.0h, 0.0h,
    -1.0h, 0.0h, -1.0h, -1.0h, 0.0h, 0.0h, 1.0h, -1.0h, -1.0h, 0.0h,
    1.0h, 1.0h, -1.0h, -1.0h, 0.0h, -1.0h, 1.0h, -1.0h, -1.0h, 0.0h,
    0.0h, -1.0h, -1.0h, -1.0h, 0.0h, 1.0h, -1.0h, -1.0h, -1.0h, 0.0h,
    -1.0h, -1.0h, -1.0h, -1.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 1.0h,
    1.0h, 0.0h, 0.0h, 0.0h, 1.0h, -1.0h, 0.0h, 0.0h, 0.0h, 1.0h,
    0.0h, 1.0h, 0.0h, 0.0h, 1.0h, 1.0h, 1.0h, 0.0h, 0.0h, 1.0h,
    -1.0h, 1.0h, 0.0h, 0.0h, 1.0h, 0.0h, -1.0h, 0.0h, 0.0h, 1.0h,
    1.0h, -1.0h, 0.0h, 0.0h, 1.0h, -1.0h, -1.0h, 0.0h, 0.0h, 1.0h,
    0.0h, 0.0h, 1.0h, 0.0h, 1.0h, 1.0h, 0.0h, 1.0h, 0.0h, 1.0h,
    -1.0h, 0.0h, 1.0h, 0.0h, 1.0h, 0.0h, 1.0h, 1.0h, 0.0h, 1.0h,
    1.0h, 1.0h, 1.0h, 0.0h, 1.0h, -1.0h, 1.0h, 1.0h, 0.0h, 1.0h,
    0.0h, -1.0h, 1.0h, 0.0h, 1.0h, 1.0h, -1.0h, 1.0h, 0.0h, 1.0h,
    -1.0h, -1.0h, 1.0h, 0.0h, 1.0h, 0.0h, 0.0h, -1.0h, 0.0h, 1.0h,
    1.0h, 0.0h, -1.0h, 0.0h, 1.0h, -1.0h, 0.0h, -1.0h, 0.0h, 1.0h,
    0.0h, 1.0h, -1.0h, 0.0h, 1.0h, 1.0h, 1.0h, -1.0h, 0.0h, 1.0h,
    -1.0h, 1.0h, -1.0h, 0.0h, 1.0h, 0.0h, -1.0h, -1.0h, 0.0h, 1.0h,
    1.0h, -1.0h, -1.0h, 0.0h, 1.0h, -1.0h, -1.0h, -1.0h, 0.0h, 1.0h,
    0.0h, 0.0h, 0.0h, 1.0h, 1.0h, 1.0h, 0.0h, 0.0h, 1.0h, 1.0h,
    -1.0h, 0.0h, 0.0h, 1.0h, 1.0h, 0.0h, 1.0h, 0.0h, 1.0h, 1.0h,
    1.0h, 1.0h, 0.0h, 1.0h, 1.0h, -1.0h, 1.0h, 0.0h, 1.0h, 1.0h,
    0.0h, -1.0h, 0.0h, 1.0h, 1.0h, 1.0h, -1.0h, 0.0h, 1.0h, 1.0h,
    -1.0h, -1.0h, 0.0h, 1.0h, 1.0h, 0.0h, 0.0h, 1.0h, 1.0h, 1.0h,
    1.0h, 0.0h, 1.0h, 1.0h, 1.0h, -1.0h, 0.0h, 1.0h, 1.0h, 1.0h,
    0.0h, 1.0h, 1.0h, 1.0h, 1.0h, 1.0h, 1.0h, 1.0h, 1.0h, 1.0h,
    -1.0h, 1.0h, 1.0h, 1.0h, 1.0h, 0.0h, -1.0h, 1.0h, 1.0h, 1.0h,
    1.0h, -1.0h, 1.0h, 1.0h, 1.0h, -1.0h, -1.0h, 1.0h, 1.0h, 1.0h,
    0.0h, 0.0h, -1.0h, 1.0h, 1.0h, 1.0h, 0.0h, -1.0h, 1.0h, 1.0h,
    -1.0h, 0.0h, -1.0h, 1.0h, 1.0h, 0.0h, 1.0h, -1.0h, 1.0h, 1.0h,
    1.0h, 1.0h, -1.0h, 1.0h, 1.0h, -1.0h, 1.0h, -1.0h, 1.0h, 1.0h,
    0.0h, -1.0h, -1.0h, 1.0h, 1.0h, 1.0h, -1.0h, -1.0h, 1.0h, 1.0h,
    -1.0h, -1.0h, -1.0h, 1.0h, 1.0h, 0.0h, 0.0h, 0.0h, -1.0h, 1.0h,
    1.0h, 0.0h, 0.0h, -1.0h, 1.0h, -1.0h, 0.0h, 0.0h, -1.0h, 1.0h,
    0.0h, 1.0h, 0.0h, -1.0h, 1.0h, 1.0h, 1.0h, 0.0h, -1.0h, 1.0h,
    -1.0h, 1.0h, 0.0h, -1.0h, 1.0h, 0.0h, -1.0h, 0.0h, -1.0h, 1.0h,
    1.0h, -1.0h, 0.0h, -1.0h, 1.0h, -1.0h, -1.0h, 0.0h, -1.0h, 1.0h,
    0.0h, 0.0h, 1.0h, -1.0h, 1.0h, 1.0h, 0.0h, 1.0h, -1.0h, 1.0h,
    -1.0h, 0.0h, 1.0h, -1.0h, 1.0h, 0.0h, 1.0h, 1.0h, -1.0h, 1.0h,
    1.0h, 1.0h, 1.0h, -1.0h, 1.0h, -1.0h, 1.0h, 1.0h, -1.0h, 1.0h,
    0.0h, -1.0h, 1.0h, -1.0h, 1.0h, 1.0h, -1.0h, 1.0h, -1.0h, 1.0h,
    -1.0h, -1.0h, 1.0h, -1.0h, 1.0h, 0.0h, 0.0h, -1.0h, -1.0h, 1.0h,
    1.0h, 0.0h, -1.0h, -1.0h, 1.0h, -1.0h, 0.0h, -1.0h, -1.0h, 1.0h,
    0.0h, 1.0h, -1.0h, -1.0h, 1.0h, 1.0h, 1.0h, -1.0h, -1.0h, 1.0h,
    -1.0h, 1.0h, -1.0h, -1.0h, 1.0h, 0.0h, -1.0h, -1.0h, -1.0h, 1.0h,
    1.0h, -1.0h, -1.0h, -1.0h, 1.0h, -1.0h, -1.0h, -1.0h, -1.0h, 1.0h,
    0.0h, 0.0h, 0.0h, 0.0h, -1.0h, 1.0h, 0.0h, 0.0h, 0.0h, -1.0h,
    -1.0h, 0.0h, 0.0h, 0.0h, -1.0h, 0.0h, 1.0h, 0.0h, 0.0h, -1.0h,
    1.0h, 1.0h, 0.0h, 0.0h, -1.0h, -1.0h, 1.0h, 0.0h, 0.0h, -1.0h,
    0.0h, -1.0h, 0.0h, 0.0h, -1.0h, 1.0h, -1.0h, 0.0h, 0.0h, -1.0h,
    -1.0h, -1.0h, 0.0h, 0.0h, -1.0h, 0.0h, 0.0h, 1.0h, 0.0h, -1.0h,
    1.0h, 0.0h, 1.0h, 0.0h, -1.0h, -1.0h, 0.0h, 1.0h, 0.0h, -1.0h,
    0.0h, 1.0h, 1.0h, 0.0h, -1.0h, 1.0h, 1.0h, 1.0h, 0.0h, -1.0h,
    -1.0h, 1.0h, 1.0h, 0.0h, -1.0h, 0.0h, -1.0h, 1.0h, 0.0h, -1.0h,
    1.0h, -1.0h, 1.0h, 0.0h, -1.0h, -1.0h, -1.0h, 1.0h, 0.0h, -1.0h,
    0.0h, 0.0h, -1.0h, 0.0h, -1.0h, 1.0h, 0.0h, -1.0h, 0.0h, -1.0h,
    -1.0h, 0.0h, -1.0h, 0.0h, -1.0h, 0.0h, 1.0h, -1.0h, 0.0h, -1.0h,
    1.0h, 1.0h, -1.0h, 0.0h, -1.0h, -1.0h, 1.0h, -1.0h, 0.0h, -1.0h,
    0.0h, -1.0h, -1.0h, 0.0h, -1.0h, 1.0h, -1.0h, -1.0h, 0.0h, -1.0h,
    -1.0h, -1.0h, -1.0h, 0.0h, -1.0h, 0.0h, 0.0h, 0.0h, 1.0h, -1.0h,
    1.0h, 0.0h, 0.0h, 1.0h, -1.0h, -1.0h, 0.0h, 0.0h, 1.0h, -1.0h,
    0.0h, 1.0h, 0.0h, 1.0h, -1.0h, 1.0h, 1.0h, 0.0h, 1.0h, -1.0h,
    -1.0h, 1.0h, 0.0h, 1.0h, -1.0h, 0.0h, -1.0h, 0.0h, 1.0h, -1.0h,
    1.0h, -1.0h, 0.0h, 1.0h, -1.0h, -1.0h, -1.0h, 0.0h, 1.0h, -1.0h,
    0.0h, 0.0h, 1.0h, 1.0h, -1.0h, 1.0h, 0.0h, 1.0h, 1.0h, -1.0h,
    -1.0h, 0.0h, 1.0h, 1.0h, -1.0h, 0.0h, 1.0h, 1.0h, 1.0h, -1.0h,
    1.0h, 1.0h, 1.0h, 1.0h, -1.0h, -1.0h, 1.0h, 1.0h, 1.0h, -1.0h,
    0.0h, -1.0h, 1.0h, 1.0h, -1.0h, 1.0h, -1.0h, 1.0h, 1.0h, -1.0h,
    -1.0h, -1.0h, 1.0h, 1.0h, -1.0h, 0.0h, 0.0h, -1.0h, 1.0h, -1.0h,
    1.0h, 0.0h, -1.0h, 1.0h, -1.0h, -1.0h, 0.0h, -1.0h, 1.0h, -1.0h,
    0.0h, 1.0h, -1.0h, 1.0h, -1.0h, 1.0h, 1.0h, -1.0h, 1.0h, -1.0h,
    -1.0h, 1.0h, -1.0h, 1.0h, -1.0h, 0.0h, -1.0h, -1.0h, 1.0h, -1.0h,
    1.0h, -1.0h, -1.0h, 1.0h, -1.0h, -1.0h, -1.0h, -1.0h, 1.0h, -1.0h,
    0.0h, 0.0h, 0.0h, -1.0h, -1.0h, 1.0h, 0.0h, 0.0h, -1.0h, -1.0h,
    -1.0h, 0.0h, 0.0h, -1.0h, -1.0h, 0.0h, 1.0h, 0.0h, -1.0h, -1.0h,
    1.0h, 1.0h, 0.0h, -1.0h, -1.0h, -1.0h, 1.0h, 0.0h, -1.0h, -1.0h,
    0.0h, -1.0h, 0.0h, -1.0h, -1.0h, 1.0h, -1.0h, 0.0h, -1.0h, -1.0h,
    -1.0h, -1.0h, 0.0h, -1.0h, -1.0h, 0.0h, 0.0h, 1.0h, -1.0h, -1.0h,
    1.0h, 0.0h, 1.0h, -1.0h, -1.0h, -1.0h, 0.0h, 1.0h, -1.0h, -1.0h,
    0.0h, 1.0h, 1.0h, -1.0h, -1.0h, 1.0h, 1.0h, 1.0h, -1.0h, -1.0h,
    -1.0h, 1.0h, 1.0h, -1.0h, -1.0h, 0.0h, -1.0h, 1.0h, -1.0h, -1.0h,
    1.0h, -1.0h, 1.0h, -1.0h, -1.0h, -1.0h, -1.0h, 1.0h, -1.0h, -1.0h,
    0.0h, 0.0h, -1.0h, -1.0h, -1.0h, 1.0h, 0.0h, -1.0h, -1.0h, -1.0h,
    -1.0h, 0.0h, -1.0h, -1.0h, -1.0h, 0.0h, 1.0h, -1.0h, -1.0h, -1.0h,
    1.0h, 1.0h, -1.0h, -1.0h, -1.0h, -1.0h, 1.0h, -1.0h, -1.0h, -1.0h,
    0.0h, -1.0h, -1.0h, -1.0h, -1.0h, 1.0h, -1.0h, -1.0h, -1.0h, -1.0h,
    -1.0h, -1.0h, -1.0h, -1.0h, -1.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h,
    0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h,
    0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h,
    0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h,
    0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h,
    0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h,
    0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h
};

kernel void q1t_matvec(
    device const uchar* q    [[buffer(0)]],
    device const float* xs   [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    device const float4* xs4 = (device const float4*)xs;

    for (uint g = lane; g < gpr; g += 32u) {
        uint wbase = g * 32u;
        uint wbase4 = wbase / 4u;
        float4 xg[8];
        for (uint i = 0; i < 8u; ++i) {
            xg[i] = xs4[wbase4 + i];
        }
        // Keep activations and the reduction in f32. Real GDN checkpoints can
        // transiently exceed f16's finite range; converting x to half here
        // produced NaN logits and greedy decode repeatedly selected the last
        // vocabulary id. The ternary LUT may stay half (its values are only
        // -1/0/+1), but every multiply is promoted before accumulation.
        float xh[32];
        for (uint i = 0; i < 8u; ++i) {
            xh[4*i+0] = xg[i].x;
            xh[4*i+1] = xg[i].y;
            xh[4*i+2] = xg[i].z;
            xh[4*i+3] = xg[i].w;
        }

        for (uint ri = 0u; ri < nr; ++ri) {
            ulong base = ((ulong)(r0 + ri) * gpr + (ulong)g) * 9u;
            device const uchar* p = q + base;
            half scale = as_type<half>(cmf_load_u16_le(p));
            
            uint b2_5 = cmf_load_u32_le(p + 2u);
            ushort b6_7 = cmf_load_u16_le(p + 6u);
            uchar b8 = p[8];

            float gsum = 0.0f;
            constant half* pl;

            pl = &Q1T_SIGN[(b2_5 & 0xFF) * 5u];
            gsum += pl[0] * xh[0] + pl[1] * xh[1] + pl[2] * xh[2] + pl[3] * xh[3] + pl[4] * xh[4];

            pl = &Q1T_SIGN[((b2_5 >> 8u) & 0xFF) * 5u];
            gsum += pl[0] * xh[5] + pl[1] * xh[6] + pl[2] * xh[7] + pl[3] * xh[8] + pl[4] * xh[9];

            pl = &Q1T_SIGN[((b2_5 >> 16u) & 0xFF) * 5u];
            gsum += pl[0] * xh[10] + pl[1] * xh[11] + pl[2] * xh[12] + pl[3] * xh[13] + pl[4] * xh[14];

            pl = &Q1T_SIGN[(b2_5 >> 24u) * 5u];
            gsum += pl[0] * xh[15] + pl[1] * xh[16] + pl[2] * xh[17] + pl[3] * xh[18] + pl[4] * xh[19];

            pl = &Q1T_SIGN[(b6_7 & 0xFF) * 5u];
            gsum += pl[0] * xh[20] + pl[1] * xh[21] + pl[2] * xh[22] + pl[3] * xh[23] + pl[4] * xh[24];

            pl = &Q1T_SIGN[(b6_7 >> 8u) * 5u];
            gsum += pl[0] * xh[25] + pl[1] * xh[26] + pl[2] * xh[27] + pl[3] * xh[28] + pl[4] * xh[29];

            pl = &Q1T_SIGN[b8 * 5u];
            gsum += pl[0] * xh[30] + pl[1] * xh[31];

            float contrib = (float)scale * gsum;
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// q1t sparse overlay: adds Σ val·x[col] onto y (the base already there), one
// thread per row over its [row_ptr[rid], row_ptr[rid+1]) entries. All reads are
// byte-wise because base_len = rows·gpr·9 is not 4-aligned.
kernel void q1t_overlay(
    device const uchar* q        [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float*       y        [[buffer(2)]],
    constant uint&      base_len [[buffer(3)]],
    constant uint&      rows     [[buffer(4)]],
    uint rid [[thread_position_in_grid]])
{
    if (rid >= rows) return;
    uint c0 = cmf_load_u32_le(q + base_len + rid * 4u);
    uint c1 = cmf_load_u32_le(q + base_len + (rid + 1u) * 4u);
    uint ent = base_len + (rows + 1u) * 4u;
    float corr = 0.0f;
    for (uint p = c0; p < c1; ++p) {
        uint e = ent + p * 4u;
        uint col_val = cmf_load_u32_le(q + e);
        uint col = col_val & 0xFFFF;
        half val = as_type<half>((ushort)(col_val >> 16));
        corr += (float)val * x[col];
    }
    y[rid] += corr;
}

inline float q4_dot8_fast(uint b, float4 x_lo, float4 x_hi) {
    // Nibble order: byte0-lo, byte0-hi, byte1-lo, byte1-hi → x_lo;
    //               byte2-lo, byte2-hi, byte3-lo, byte3-hi → x_hi.
    float4 w_lo = float4((float)(b & 0xFu) - 8.0f,
                         (float)((b >> 4u) & 0xFu) - 8.0f,
                         (float)((b >> 8u) & 0xFu) - 8.0f,
                         (float)((b >> 12u) & 0xFu) - 8.0f);
    float4 w_hi = float4((float)((b >> 16u) & 0xFu) - 8.0f,
                          (float)((b >> 20u) & 0xFu) - 8.0f,
                          (float)((b >> 24u) & 0xFu) - 8.0f,
                          (float)(b >> 28u) - 8.0f);
    return dot(w_lo, x_lo) + dot(w_hi, x_hi);
}

// q4_block: [packed nibbles: rows·gpr·16 B][f16 scales: rows·gpr·2 B]. Group
// gi's nibbles at packed[gi·16], scale at scales[gi·2]; weight = (nib-8)·scale.
// Lets the token graph keep a precise down_proj (or lm_head) on-device without
// quantizing it to ternary. 4 rows/simdgroup, cached activations & hardware SIMD dot.
kernel void q4b_matvec(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    uint scales_off = rows * gpr * 16u;
    device const uchar* sc = q + scales_off;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g = lane; g < gpr; g += 32u) {
        uint xb = g * 32u;
        device const float4* xv = (device const float4*)(x + xb);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];

        for (uint ri = 0u; ri < nr; ++ri) {
            uint gi = (r0 + ri) * gpr + g;
            half scale = as_type<half>(*(device const ushort*)(sc + gi * 2u));
            uint4 pk4 = *(device const uint4*)(q + (ulong)gi * 16u);

            float gsum = q4_dot8_fast(pk4.x, x0, x1)
                       + q4_dot8_fast(pk4.y, x2, x3)
                       + q4_dot8_fast(pk4.z, x4, x5)
                       + q4_dot8_fast(pk4.w, x6, x7);

            float contrib = (float)scale * gsum;
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

inline half q4_dot8_half(uint b, half4 x_lo, half4 x_hi) {
    // Nibble order: byte0-lo, byte0-hi, byte1-lo, byte1-hi → x_lo;
    //               byte2-lo, byte2-hi, byte3-lo, byte3-hi → x_hi.
    half4 w_lo = half4((half)(b & 0xFu) - 8.0h,
                       (half)((b >> 4u) & 0xFu) - 8.0h,
                       (half)((b >> 8u) & 0xFu) - 8.0h,
                       (half)((b >> 12u) & 0xFu) - 8.0h);
    half4 w_hi = half4((half)((b >> 16u) & 0xFu) - 8.0h,
                       (half)((b >> 20u) & 0xFu) - 8.0h,
                       (half)((b >> 24u) & 0xFu) - 8.0h,
                       (half)(b >> 28u) - 8.0h);
    return dot(w_lo, x_lo) + dot(w_hi, x_hi);
}

// Half-ALU q4 twin. The per-group result converts back to f32 before the
// long reduction, keeping accumulation stable while using Apple's higher
// throughput half vector pipes for nibble×activation work.
kernel void q4b_matvec_h(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    uint scales_off = rows * gpr * 16u;
    device const uchar* sc = q + scales_off;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g = lane; g < gpr; g += 32u) {
        uint xb = g * 32u;
        device const float4* xv = (device const float4*)(x + xb);
        half4 x0 = half4(xv[0]), x1 = half4(xv[1]);
        half4 x2 = half4(xv[2]), x3 = half4(xv[3]);
        half4 x4 = half4(xv[4]), x5 = half4(xv[5]);
        half4 x6 = half4(xv[6]), x7 = half4(xv[7]);
        for (uint ri = 0u; ri < nr; ++ri) {
            uint gi = (r0 + ri) * gpr + g;
            half scale = as_type<half>(*(device const ushort*)(sc + gi * 2u));
            uint4 pk4 = *(device const uint4*)(q + (ulong)gi * 16u);
            half gsum = q4_dot8_half(pk4.x, x0, x1)
                       + q4_dot8_half(pk4.y, x2, x3)
                       + q4_dot8_half(pk4.z, x4, x5)
                       + q4_dot8_half(pk4.w, x6, x7);
            float contrib = (float)(scale * gsum);
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// q4_tiled: 18-byte tiles [f16 scale][16B nibbles] per 32-group — ONE
// sequential stream per row (the split q4b layout reads nibbles and
// scales from two distant regions). Nibble order and values match q4b
// (lo nibble = even element, hi = odd, value = nibble − 8), so
// q4_dot8_fast is reused as-is. 18B tiles are only 2-aligned → the
// nibble words go through the unaligned byte loaders.
kernel void q4t_matvec(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    uint r0 = (tgpos * sgs + sg) * 4u;
    if (r0 >= rows) return;
    uint nr = min(rows - r0, 4u);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    // MEASURED DEAD END (M4, Nanbeige 4.2 decode): hoisting the four
    // rows' tile loads into one unrolled block so their misses overlap
    // cost 19.0 → 14.0 tok/s. 16 packed uints + 8 float4 of x + the
    // accumulators overflow the register budget and the occupancy loss
    // beats the latency win. Keep the one-row-at-a-time inner loop.
    for (uint g = lane; g < gpr; g += 32u) {
        uint xb = g * 32u;
        device const float4* xv = (device const float4*)(x + xb);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
        for (uint ri = 0u; ri < nr; ++ri) {
            ulong t = ((ulong)(r0 + ri) * gpr + (ulong)g) * 18u;
            // 18B tiles are always 2-aligned (tensor blobs are 64-aligned,
            // 18 is even) → nine ushort loads, not sixteen byte loads.
            device const ushort* p16 = (device const ushort*)(q + t);
            half scale = as_type<half>(p16[0]);
            uint b0 = (uint)p16[1] | ((uint)p16[2] << 16);
            uint b1 = (uint)p16[3] | ((uint)p16[4] << 16);
            uint b2 = (uint)p16[5] | ((uint)p16[6] << 16);
            uint b3 = (uint)p16[7] | ((uint)p16[8] << 16);
            float gsum = q4_dot8_fast(b0, x0, x1)
                       + q4_dot8_fast(b1, x2, x3)
                       + q4_dot8_fast(b2, x4, x5)
                       + q4_dot8_fast(b3, x6, x7);
            float contrib = (float)scale * gsum;
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// q4tp: same nibble values and order as q4t, but the scale is a 5-bit rung
// on the row's ladder, kept in two side planes that follow all the nibbles.
// Two consequences here, both good: the nibble stream is a clean 16 B stride
// (4-aligned, so four uint loads replace q4t's nine unaligned ushorts), and
// the scale costs one exp2 — a hardware instruction on-device. The CPU
// expands the ladder geometrically instead, purely to avoid 32 libm calls
// per row; the two forms agree to ~2e-6 relative, which is nothing against
// a 4-bit grid (measured, see `q4tp_ladder`).
kernel void q4tp_matvec(
    device const uchar* q    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      gpr  [[buffer(3)]],
    constant uint&      rows [[buffer(4)]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint sgs  [[simdgroups_per_threadgroup]])
{
    // One rung per lane, four rows per simdgroup, eight simdgroups.
    threadgroup float lad[8u * 4u * 32u];

    uint r0 = (tgpos * sgs + sg) * 4u;
    bool active = r0 < rows;
    uint nr = active ? min(rows - r0, 4u) : 0u;

    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off  = params_off + (ulong)rows * 4ul;
    uint  stride     = (gpr * 5u + 7u) / 8u;

    // Expand each row's ladder ONCE. Evaluating 2^(lo + code*step) inside the
    // tile loop instead was measured to cost the model ~15% even though the
    // kernel benchmarked FASTER standalone: free-running dispatches hide the
    // dependent chain (code byte → exp2 → scale), and the model's dispatches
    // serialize on each other, which exposes it. The lane index IS the rung,
    // so one exp2 per lane per row covers all 32.
    for (uint ri = 0u; ri < nr; ++ri) {
        device const half* ph = (device const half*)(q + params_off + (ulong)(r0 + ri) * 4ul);
        lad[(sg * 4u + ri) * 32u + lane] = exp2((float)ph[0] + (float)lane * (float)ph[1]);
    }
    // Every thread reaches this, including the inactive tail simdgroups —
    // a barrier skipped by part of the threadgroup is undefined.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint g = lane; g < gpr; g += 32u) {
        uint xb = g * 32u;
        device const float4* xv = (device const float4*)(x + xb);
        float4 x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
        float4 x4 = xv[4], x5 = xv[5], x6 = xv[6], x7 = xv[7];
        uint bit = g * 5u;
        uint cb  = bit >> 3u;
        uint shf = bit & 7u;
        for (uint ri = 0u; ri < nr; ++ri) {
            uint r = r0 + ri;
            // 16 B tiles are 4-aligned (tensors are 64-aligned in the blob),
            // so four uint loads — q4t needs nine ushorts for its 18 B stride.
            device const uint* p32 = (device const uint*)(q + ((ulong)r * gpr + (ulong)g) * 16ul);
            uint b0 = p32[0], b1 = p32[1], b2 = p32[2], b3 = p32[3];
            // The 5-bit field spills into the next byte past bit 3; the row's
            // stride always holds that byte when it does.
            device const uchar* cp = q + codes_off + (ulong)r * (ulong)stride + cb;
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            float scale = lad[(sg * 4u + ri) * 32u + code];
            float gsum = q4_dot8_fast(b0, x0, x1)
                       + q4_dot8_fast(b1, x2, x3)
                       + q4_dot8_fast(b2, x4, x5)
                       + q4_dot8_fast(b3, x6, x7);
            float contrib = scale * gsum;
            if (ri == 0u) acc0 += contrib;
            else if (ri == 1u) acc1 += contrib;
            else if (ri == 2u) acc2 += contrib;
            else acc3 += contrib;
        }
    }
    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    if (lane == 0u) {
        y[r0] = acc0;
        if (nr > 1u) y[r0 + 1u] = acc1;
        if (nr > 2u) y[r0 + 2u] = acc2;
        if (nr > 3u) y[r0 + 3u] = acc3;
    }
}

// MEASURED NEUTRAL (M4, Lumina DiT 512² and the Nanbeige chunk prefill):
// giving these two kernels q8_mul_mm's cols/rows function-constant
// specialization changed nothing — paired runs at matched thermal state
// landed between −4% and +3%. The K loop is already tile-shaped (NK=32
// == one 18 B group per step), so a compile-time `cols` buys no unroll
// the shape does not already imply. Not worth the extra pipeline cache.
//
// q4t register-blocked GEMM: q8_mul_mm's simdgroup machinery, weight
// staging decodes 18-byte q4t tiles (f16 scale + 32 nibbles) in the
// K loop. NK=32 == GROUP_SIZE so each K-step is exactly one tile per
// row — the weights travel device→shmem as 0.56 B each instead of a
// dequanted f32 scratch re-read per batch tile (the two-pass variant
// measured bandwidth-bound: ~2.8 GB of W traffic per FFN-shaped op).
// q4tp twins of the two q4t GEMMs. Only the weight-staging block differs:
// the 16 B nibble stride replaces the 18 B tile, and the scale comes off the
// row's ladder instead of the tile header. Everything downstream — the
// simdgroup machinery, the shmem layout, the epilogue — is byte-for-byte the
// q4t kernel, because the decoded weights are the same numbers.
kernel void q4tp_mul_mm(
    device const uchar*  q      [[buffer(0)]],
    device const float*  xs     [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint&       cols_b [[buffer(3)]],
    constant uint&       rows_b [[buffer(4)]],
    constant uint&       nb     [[buffer(5)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    uint cols = cols_b;
    uint rows = rows_b;
    uint gpr = cols >> 5u;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;
    uint r1 = tg.x * 32u;
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);

    device const float* yrow = xs + (ulong)(r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off = params_off + (ulong)rows * 4ul;
    uint cstride = (gpr * 5u + 7u) / 8u;
    device const half* prow = (device const half*)(q + params_off + (ulong)(r0 + lr0) * 4ul);
    float row_lo = (float)prow[0];
    float row_st = (float)prow[1];
    device const uchar* codes_row = q + codes_off + (ulong)(r0 + lr0) * (ulong)cstride;
    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // W: this thread's 16 weights (row r0+lr0, K-half il0) — 8
        // nibble bytes of one tile, low nibble first.
        {
            uint g = k0 >> 5u;
            uint bit = g * 5u;
            uint shf = bit & 7u;
            device const uchar* cp = codes_row + (bit >> 3u);
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            // lo/step are loop-invariant for this thread (its row is fixed
            // across the whole K loop), so they are read ONCE above. Leaving
            // them in the loop cost Lumina's DiT enough that the runtime probe
            // preferred the CPU GEMM outright — the "the GEMM's arithmetic
            // hides the chain" argument held for FFN shapes and not for this.
            float scale = exp2(row_lo + (float)code * row_st);
            device const uchar* nib = q + ((ulong)wr * gpr + (ulong)g) * 16ul + 8u * il0;
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            float wv[16];
            for (uint i = 0; i < 8u; ++i) {
                uint bb = nib[i];
                wv[2u * i]      = ((float)(bb & 0xFu) - 8.0f) * scale;
                wv[2u * i + 1u] = ((float)(bb >> 4u) - 8.0f) * scale;
            }
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        // X: 8 consecutive floats → one 8x8-block row (identical to q8).
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* y4 = (device const float4*)yrow;
            float4 v0 = y4[0];
            float4 v1 = y4[1];
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (half)v0.x; dst[1] = (half)v0.y;
            dst[2] = (half)v0.z; dst[3] = (half)v0.w;
            dst[4] = (half)v1.x; dst[5] = (half)v1.y;
            dst[6] = (half)v1.z; dst[7] = (half)v1.w;
        }
        yrow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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

    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        device float* C = y + (r0 + 32u * (sgitg & 1u))
            + (ulong)(r1 + 16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4),
                            rows, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                            64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (uint j = tiitg; j < nr1; j += 128u) {
                device float* D = y + r0 + (ulong)(r1 + j) * rows;
                threadgroup const float* Cr = ((threadgroup float*)shmem) + j * 64u;
                for (uint i = 0; i < nr0; ++i) {
                    D[i] = Cr[i];
                }
            }
        }
    }
}

kernel void q4tp_mul_mm_silu(
    device const uchar*  q      [[buffer(0)]],
    device const float*  gs     [[buffer(1)]],
    device const float*  us     [[buffer(2)]],
    device float*        y      [[buffer(3)]],
    constant uint&       cols_b [[buffer(4)]],
    constant uint&       rows_b [[buffer(5)]],
    constant uint&       nb     [[buffer(6)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    uint cols = cols_b;
    uint rows = rows_b;
    uint gpr = cols >> 5u;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;
    uint r1 = tg.x * 32u;
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);

    device const float* grow = gs + (ulong)(r1 + lr1) * cols + iy;
    device const float* urow = us + (ulong)(r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    ulong params_off = (ulong)rows * (ulong)gpr * 16ul;
    ulong codes_off = params_off + (ulong)rows * 4ul;
    uint cstride = (gpr * 5u + 7u) / 8u;
    device const half* prow = (device const half*)(q + params_off + (ulong)(r0 + lr0) * 4ul);
    float row_lo = (float)prow[0];
    float row_st = (float)prow[1];
    device const uchar* codes_row = q + codes_off + (ulong)(r0 + lr0) * (ulong)cstride;
    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // W: this thread's 16 weights (row r0+lr0, K-half il0) — 8
        // nibble bytes of one tile, low nibble first.
        {
            uint g = k0 >> 5u;
            uint bit = g * 5u;
            uint shf = bit & 7u;
            device const uchar* cp = codes_row + (bit >> 3u);
            uint code = (((uint)cp[0] | ((shf > 3u) ? ((uint)cp[1] << 8) : 0u)) >> shf) & 31u;
            // lo/step are loop-invariant for this thread (its row is fixed
            // across the whole K loop), so they are read ONCE above. Leaving
            // them in the loop cost Lumina's DiT enough that the runtime probe
            // preferred the CPU GEMM outright — the "the GEMM's arithmetic
            // hides the chain" argument held for FFN shapes and not for this.
            float scale = exp2(row_lo + (float)code * row_st);
            device const uchar* nib = q + ((ulong)wr * gpr + (ulong)g) * 16ul + 8u * il0;
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            float wv[16];
            for (uint i = 0; i < 8u; ++i) {
                uint bb = nib[i];
                wv[2u * i]      = ((float)(bb & 0xFu) - 8.0f) * scale;
                wv[2u * i + 1u] = ((float)(bb >> 4u) - 8.0f) * scale;
            }
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        // X: silu(gate)·up staged straight into the tile — no act
        // buffer, exactly as q8_mul_mm_silu does it.
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* g4 = (device const float4*)grow;
            device const float4* u4 = (device const float4*)urow;
            float4 g0 = g4[0];
            float4 g1 = g4[1];
            float4 u0 = u4[0];
            float4 u1 = u4[1];
            float4 a0 = (g0 / (1.0f + exp(-g0))) * u0;
            float4 a1 = (g1 / (1.0f + exp(-g1))) * u1;
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (half)a0.x; dst[1] = (half)a0.y;
            dst[2] = (half)a0.z; dst[3] = (half)a0.w;
            dst[4] = (half)a1.x; dst[5] = (half)a1.y;
            dst[6] = (half)a1.z; dst[7] = (half)a1.w;
        }
        grow += NK;
        urow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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

    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        device float* C = y + (r0 + 32u * (sgitg & 1u))
            + (ulong)(r1 + 16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4),
                            rows, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                            64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (uint j = tiitg; j < nr1; j += 128u) {
                device float* D = y + r0 + (ulong)(r1 + j) * rows;
                threadgroup const float* Cr = ((threadgroup float*)shmem) + j * 64u;
                for (uint i = 0; i < nr0; ++i) {
                    D[i] = Cr[i];
                }
            }
        }
    }
}

kernel void q4t_mul_mm(
    device const uchar*  q      [[buffer(0)]],
    device const float*  xs     [[buffer(1)]],
    device float*        y      [[buffer(2)]],
    constant uint&       cols_b [[buffer(3)]],
    constant uint&       rows_b [[buffer(4)]],
    constant uint&       nb     [[buffer(5)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    uint cols = cols_b;
    uint rows = rows_b;
    uint gpr = cols >> 5u;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;
    uint r1 = tg.x * 32u;
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);

    device const float* yrow = xs + (ulong)(r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // W: this thread's 16 weights (row r0+lr0, K-half il0) — 8
        // nibble bytes of one tile, low nibble first.
        {
            uint g = k0 >> 5u;
            device const uchar* tile = q + ((ulong)(r0 + lr0) * gpr + (ulong)g) * 18u;
            float scale = (float)as_type<half>((ushort)((uint)tile[0] | ((uint)tile[1] << 8)));
            device const uchar* nib = tile + 2u + 8u * il0;
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            float wv[16];
            for (uint i = 0; i < 8u; ++i) {
                uint bb = nib[i];
                wv[2u * i]      = ((float)(bb & 0xFu) - 8.0f) * scale;
                wv[2u * i + 1u] = ((float)(bb >> 4u) - 8.0f) * scale;
            }
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        // X: 8 consecutive floats → one 8x8-block row (identical to q8).
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* y4 = (device const float4*)yrow;
            float4 v0 = y4[0];
            float4 v1 = y4[1];
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (half)v0.x; dst[1] = (half)v0.y;
            dst[2] = (half)v0.z; dst[3] = (half)v0.w;
            dst[4] = (half)v1.x; dst[5] = (half)v1.y;
            dst[6] = (half)v1.z; dst[7] = (half)v1.w;
        }
        yrow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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

    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        device float* C = y + (r0 + 32u * (sgitg & 1u))
            + (ulong)(r1 + 16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4),
                            rows, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                            64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (uint j = tiitg; j < nr1; j += 128u) {
                device float* D = y + r0 + (ulong)(r1 + j) * rows;
                threadgroup const float* Cr = ((threadgroup float*)shmem) + j * 64u;
                for (uint i = 0; i < nr0; ++i) {
                    D[i] = Cr[i];
                }
            }
        }
    }
}

// q4t_mul_mm with the FFN activation fused into the X-tile load:
// C = silu(gate)·up · dequant(down)ᵀ. The q8 twin (q8_mul_mm_silu) is
// what lets the chunk prefill skip an act buffer and its round trip;
// q4t models had no such kernel, which is why the whole chunk graph
// bailed to the CPU for them.
kernel void q4t_mul_mm_silu(
    device const uchar*  q      [[buffer(0)]],
    device const float*  gs     [[buffer(1)]],
    device const float*  us     [[buffer(2)]],
    device float*        y      [[buffer(3)]],
    constant uint&       cols_b [[buffer(4)]],
    constant uint&       rows_b [[buffer(5)]],
    constant uint&       nb     [[buffer(6)]],
    uint tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint2 tg  [[threadgroup_position_in_grid]])
{
    uint cols = cols_b;
    uint rows = rows_b;
    uint gpr = cols >> 5u;
    threadgroup char shmem[8192];
    threadgroup half* sa = (threadgroup half*)shmem;
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);
    const uint NK = 32u;
    uint r0 = tg.y * 64u;
    uint r1 = tg.x * 32u;
    uint nr0 = min(rows - r0, 64u);
    uint nr1 = min(nb - r1, 32u);
    uint lr0 = min(tiitg / 2u, nr0 - 1u);
    uint il0 = tiitg % 2u;
    uint lr1 = min(tiitg / 4u, nr1 - 1u);
    uint iy  = 8u * (tiitg % 4u);

    device const float* grow = gs + (ulong)(r1 + lr1) * cols + iy;
    device const float* urow = us + (ulong)(r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < cols; k0 += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // W: this thread's 16 weights (row r0+lr0, K-half il0) — 8
        // nibble bytes of one tile, low nibble first.
        {
            uint g = k0 >> 5u;
            device const uchar* tile = q + ((ulong)(r0 + lr0) * gpr + (ulong)g) * 18u;
            float scale = (float)as_type<half>((ushort)((uint)tile[0] | ((uint)tile[1] << 8)));
            device const uchar* nib = tile + 2u + 8u * il0;
            uint sy = (tiitg / 2u) / 8u;
            uint lx = (tiitg / 2u) % 8u;
            float wv[16];
            for (uint i = 0; i < 8u; ++i) {
                uint bb = nib[i];
                wv[2u * i]      = ((float)(bb & 0xFu) - 8.0f) * scale;
                wv[2u * i + 1u] = ((float)(bb >> 4u) - 8.0f) * scale;
            }
            uint ib0 = 8u * (2u * il0) + sy;
            uint ib1 = 8u * (2u * il0 + 1u) + sy;
            for (uint i = 0; i < 8u; ++i) {
                sa[64u * ib0 + 8u * i + lx] = (half)wv[i];
                sa[64u * ib1 + 8u * i + lx] = (half)wv[i + 8u];
            }
        }
        // X: silu(gate)·up staged straight into the tile — no act
        // buffer, exactly as q8_mul_mm_silu does it.
        {
            uint sx = tiitg % 4u;
            uint sy = (tiitg / 4u) / 8u;
            uint ly = (tiitg / 4u) % 8u;
            uint ib = 4u * sx + sy;
            device const float4* g4 = (device const float4*)grow;
            device const float4* u4 = (device const float4*)urow;
            float4 g0 = g4[0];
            float4 g1 = g4[1];
            float4 u0 = u4[0];
            float4 u1 = u4[1];
            float4 a0 = (g0 / (1.0f + exp(-g0))) * u0;
            float4 a1 = (g1 / (1.0f + exp(-g1))) * u1;
            threadgroup half* dst = sb + 64u * ib + 8u * ly;
            dst[0] = (half)a0.x; dst[1] = (half)a0.y;
            dst[2] = (half)a0.z; dst[3] = (half)a0.w;
            dst[4] = (half)a1.x; dst[5] = (half)a1.y;
            dst[6] = (half)a1.z; dst[7] = (half)a1.w;
        }
        grow += NK;
        urow += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + 4u * 64u * (sgitg % 2u);
        threadgroup const half* lsmb = sb + 2u * 64u * (sgitg / 2u);
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

    if (r0 + 64u <= rows && r1 + 32u <= nb) {
        device float* C = y + (r0 + 32u * (sgitg & 1u))
            + (ulong)(r1 + 16u * (sgitg >> 1u)) * rows;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * (ulong)rows * (i / 4),
                            rows, ulong2(0, 0), false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* temp_str = ((threadgroup float*)shmem)
            + 32u * (sgitg & 1u) + (16u * (sgitg >> 1u)) * 64u;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                            64, ulong2(0, 0), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (uint j = tiitg; j < nr1; j += 128u) {
                device float* D = y + r0 + (ulong)(r1 + j) * rows;
                threadgroup const float* Cr = ((threadgroup float*)shmem) + j * 64u;
                for (uint i = 0; i < nr0; ++i) {
                    D[i] = Cr[i];
                }
            }
        }
    }
}

"#;

struct Ctx {
    _device: Device,
    queue: CommandQueue,
    q8: ComputePipelineState,
    q8f: ComputePipelineState,
    q8mm: ComputePipelineState,
    q8mmm: ComputePipelineState,
    q1: ComputePipelineState,
    q1h: ComputePipelineState,
    q1t: ComputePipelineState,
    q1t_ov: ComputePipelineState,
    q1t_mm: ComputePipelineState,
    q1t_ovmm: ComputePipelineState,
    q4b: ComputePipelineState,
    q4bh: ComputePipelineState,
    q4t: ComputePipelineState,
    q4tp: ComputePipelineState,
    q4tmm: ComputePipelineState,
    q4tmmsilu: ComputePipelineState,
    q4tpmm: ComputePipelineState,
    q4tpmmsilu: ComputePipelineState,
    smaxrows: ComputePipelineState,
    flashatt: ComputePipelineState,
    convmm: ComputePipelineState,
    p2nchw: ComputePipelineState,
    gnred: ComputePipelineState,
    gnapp: ComputePipelineState,
    ups2x: ComputePipelineState,
    rmsmod: ComputePipelineState,
    rmsres: ComputePipelineState,
    ropepack: ComputePipelineState,
    packh: ComputePipelineState,
    flag: ComputePipelineState,
    rmsn: ComputePipelineState,
    f16mv: ComputePipelineState,
    conv: ComputePipelineState,
    ring: ComputePipelineState,
    gates: ComputePipelineState,
    qkn: ComputePipelineState,
    stateup: ComputePipelineState,
    silu: ComputePipelineState,
    axpy: ComputePipelineState,
    zero: ComputePipelineState,
    rqkn: ComputePipelineState,
    kvapp: ComputePipelineState,
    gqat: ComputePipelineState,
    cattend: ComputePipelineState,
    rmsrows: ComputePipelineState,
    cropekv: ComputePipelineState,
    mmf32nt: ComputePipelineState,
    q8mmsilu: ComputePipelineState,
    mmf32nn: ComputePipelineState,
    csmax: ComputePipelineState,
    impcol: ComputePipelineState,
    unstack: ComputePipelineState,
    embedq8: ComputePipelineState,
    addnorm: ComputePipelineState,
    sgate: ComputePipelineState,
    /// Compiled MSL library — shape-specialized pipelines are built
    /// from it lazily.
    lib: metal::Library,
    /// Shape-specialized mul_mm pipelines: (rows, cols, kind) where
    /// kind 0 = q8, 1 = q8+silu, 2 = f32nt, 3 = f32nn.
    mm_fc: Mutex<HashMap<(u32, u32, u8), ComputePipelineState>>,
    /// Device K/V cache mirrors keyed by (pipeline id, layer).
    kv_mirrors: Mutex<HashMap<(u64, usize), KvMirror>>,
    /// No-copy buffer per model. Retaining the Arc is essential: a Metal
    /// buffer does not own its mmap bytes, and pointer-only cache keys can be
    /// reused after a model is dropped (cross-model data corruption).
    file_bufs: Mutex<HashMap<usize, (Buffer, Arc<CmfModel>)>>,
    /// row_scale buffer per tensor (key — (stable model identity, idx)).
    rs_bufs: Mutex<HashMap<(usize, usize), Buffer>>,
    /// q8_2f input-channel field buffer per tensor.
    cf_bufs: Mutex<HashMap<(usize, usize), Buffer>>,
    /// Reusable xs/y buffers by size (no per-token allocations).
    io_bufs: Mutex<HashMap<usize, Buffer>>,
    /// Shared completion-flag word + monotone ticket (fast wait).
    flag_buf: Buffer,
    ticket: std::sync::atomic::AtomicU32,
}

// metal-rs objects — retained ObjC pointers; used under a Mutex
// or from a single decode thread.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();

fn ctx() -> Option<&'static Ctx> {
    let requested = std::env::var("CMF_GPU")
        .map(|v| v != "0")
        .unwrap_or_else(|_| {
            crate::pipeline::GLOBAL_USE_GPU.load(std::sync::atomic::Ordering::Relaxed)
        });
    if !requested {
        // Do not permanently cache the disabled state: callers may enable the
        // backend after process start (the CLI and tests both do this).
        return None;
    }
    match CTX.get_or_init(init) {
        Ok(c) => {
            tracing::info!("Metal GPU path: on ({})", c._device.name());
            Some(c)
        }
        Err(e) => {
            tracing::warn!("Metal init failed — CPU fallback: {e}");
            None
        }
    }
}

/// Returns the cached Metal initialization error, if initialization was tried.
/// Primarily useful for diagnostics and hardware-specific integration tests.
pub fn initialization_error() -> Option<&'static str> {
    CTX.get()
        .and_then(|result| result.as_ref().err().map(String::as_str))
}

fn init() -> Result<Ctx, String> {
    let device = Device::system_default().ok_or("no Metal device")?;
    // The zero-copy mmap buffers assume unified memory. On discrete-GPU
    // Macs (Intel-era) `newBufferWithBytesNoCopy` silently yields stale
    // data — measured max|Δ| ≈ 0.53 vs the f32 reference on a Radeon —
    // so refuse the device instead of returning wrong numbers.
    if !device.has_unified_memory() {
        return Err(format!(
            "device '{}' has no unified memory — no-copy mmap path needs UMA",
            device.name()
        ));
    }
    let opts = metal::CompileOptions::new();
    // atomic_float (Born-importance accumulation in gqa_attend) needs
    // MSL 3.0 — macOS 13+, a subset of what the UMA gate already implies.
    opts.set_language_version(metal::MTLLanguageVersion::V3_0);
    let lib = device
        .new_library_with_source(MSL, &opts)
        .map_err(|e| format!("MSL compile: {e}"))?;
    let pso = |name: &str| -> Result<ComputePipelineState, String> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| format!("kernel {name}: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline {name}: {e}"))
    };
    let q8 = pso("q8_matvec")?;
    let q8f = pso("q8f_matvec")?;
    let q8mm = pso("q8_matmat")?;
    // Functions referencing function constants must be fetched through
    // the constantValues API even for the generic (all-optional-unset)
    // variant.
    let pso_fc = |name: &str| -> Result<ComputePipelineState, String> {
        let fcv = metal::FunctionConstantValues::new();
        let f = lib
            .get_function(name, Some(fcv))
            .map_err(|e| format!("kernel {name}: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline {name}: {e}"))
    };
    let q8mmm = pso_fc("q8_mul_mm")?;
    let q1 = pso("q1_matvec")?;
    let q1h = pso("q1_matvec_h")?;
    let q1t = pso("q1t_matvec")?;
    let q1t_ov = pso("q1t_overlay")?;
    let q1t_mm = pso("q1t_mul_mm")?;
    let q1t_ovmm = pso("q1t_overlay_mm")?;
    let q4b = pso("q4b_matvec")?;
    let q4bh = pso("q4b_matvec_h")?;
    let q4t = pso("q4t_matvec")?;
    let q4tp = pso("q4tp_matvec")?;
    let q4tmm = pso("q4t_mul_mm")?;
    let q4tmmsilu = pso("q4t_mul_mm_silu")?;
    let q4tpmm = pso("q4tp_mul_mm")?;
    let q4tpmmsilu = pso("q4tp_mul_mm_silu")?;
    let smaxrows = pso("softmax_rows")?;
    let flashatt = pso("dit_flash_attend")?;
    let convmm = pso("conv_mul_mm")?;
    let p2nchw = pso("panel_to_nchw")?;
    let gnred = pso("gn_reduce")?;
    let gnapp = pso("gn_apply")?;
    let ups2x = pso("upsample2x_k")?;
    let rmsmod = pso("rms_mod_rows")?;
    let rmsres = pso("rms_residual_rows")?;
    let ropepack = pso("dit_rope_pack")?;
    let packh = pso("pack_heads")?;
    let flag = pso("write_flag")?;
    let rmsn = pso("rmsnorm_k")?;
    let f16mv = pso("f32_matvec")?;
    let conv = pso("gdn_conv")?;
    let ring = pso("gdn_ring_shift")?;
    let gates = pso("gdn_gates")?;
    let qkn = pso("gdn_qk_norms")?;
    let stateup = pso("gdn_state_update")?;
    let silu = pso("silu_mul_pre")?;
    let axpy = pso("axpy")?;
    let zero = pso("fill_zero")?;
    let rqkn = pso("attn_rope_qkn")?;
    let kvapp = pso("kv_append")?;
    let gqat = pso("gqa_attend")?;
    let cattend = pso("chunk_attend")?;
    let rmsrows = pso("rmsnorm_rows")?;
    let cropekv = pso("chunk_rope_kv")?;
    let mmf32nt = pso_fc("mul_mm_f32nt")?;
    let q8mmsilu = pso_fc("q8_mul_mm_silu")?;
    let mmf32nn = pso_fc("mul_mm_f32nn")?;
    let csmax = pso("causal_softmax")?;
    let impcol = pso("imp_colsum")?;
    let unstack = pso("panel_unstack")?;
    let embedq8 = pso("embed_q8_rows")?;
    let addnorm = pso("add_rmsnorm_rows")?;
    let sgate = pso("sig_gate")?;
    let queue = device.new_command_queue();
    let flag_buf = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
    unsafe { *(flag_buf.contents() as *mut u32) = 0 };
    Ok(Ctx {
        _device: device,
        queue,
        q8,
        q8f,
        q8mm,
        q8mmm,
        q1,
        q1h,
        q1t,
        q1t_ov,
        q1t_mm,
        q1t_ovmm,
        q4b,
        q4bh,
        q4t,
        q4tp,
        q4tmm,
        q4tmmsilu,
        q4tpmm,
        q4tpmmsilu,
        smaxrows,
        flashatt,
        convmm,
        p2nchw,
        gnred,
        gnapp,
        ups2x,
        rmsmod,
        rmsres,
        ropepack,
        packh,
        flag,
        rmsn,
        f16mv,
        conv,
        ring,
        gates,
        qkn,
        stateup,
        silu,
        axpy,
        zero,
        rqkn,
        kvapp,
        gqat,
        cattend,
        rmsrows,
        cropekv,
        mmf32nt,
        q8mmsilu,
        mmf32nn,
        csmax,
        impcol,
        unstack,
        embedq8,
        addnorm,
        sgate,
        lib,
        mm_fc: Mutex::new(HashMap::new()),
        kv_mirrors: Mutex::new(HashMap::new()),
        file_bufs: Mutex::new(HashMap::new()),
        rs_bufs: Mutex::new(HashMap::new()),
        cf_bufs: Mutex::new(HashMap::new()),
        io_bufs: Mutex::new(HashMap::new()),
        flag_buf,
        ticket: std::sync::atomic::AtomicU32::new(0),
    })
}

/// Is the GPU enabled and initialized?
pub fn enabled() -> bool {
    ctx().is_some()
}

/// Micro-bench hook: N empty command-buffer commit+wait round trips.
#[doc(hidden)]
pub fn empty_submit_bench(n: usize) -> f64 {
    let Some(c) = ctx() else { return f64::NAN };
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.end_encoding();
        cmd.commit();
        wait_fast(cmd);
    }
    t0.elapsed().as_secs_f64()
}

/// Micro-bench hook: N empty command buffers committed back-to-back,
/// ONE wait at the end — separates pipeline latency from per-submit cost.
#[doc(hidden)]
pub fn pipelined_submit_bench(n: usize) -> f64 {
    let Some(c) = ctx() else { return f64::NAN };
    let t0 = std::time::Instant::now();
    let mut last = None;
    for _ in 0..n {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.end_encoding();
        cmd.commit();
        last = Some(cmd.to_owned());
    }
    if let Some(cmd) = last {
        wait_fast(&cmd);
    }
    t0.elapsed().as_secs_f64()
}

/// Probe helper: weights are no-copy over the file mapping, so residency
/// is per FILE — true once the file buffer exists; otherwise create it
/// now (no dispatch, `may_upload` permitting) and report cold.
pub fn q8_resident_or_upload(model: &Arc<CmfModel>, _idx: usize, may_upload: bool) -> bool {
    let Some(c) = ctx() else { return false };
    let key = model_key(model);
    if c.file_bufs.lock().unwrap().contains_key(&key) {
        return true;
    }
    if may_upload {
        let _ = file_buffer(c, model);
    }
    false
}

/// Commit with a fast completion path: append a flag-writing encoder
/// (ordered after `last_out` via a read hazard), commit, and spin on
/// the shared flag word — the driver's status/completion machinery
/// costs ~1.3 ms per round trip, the UMA flag lands in ~0.1 ms. Status
/// polling stays as the timeout fallback.
fn submit_and_wait(c: &Ctx, cmd: &metal::CommandBufferRef, outs: &[&Buffer]) {
    // NOTE: a "fast flag" variant (last encoder writes a ticket into a
    // shared buffer, CPU spins on the word) was tried here and REVERTED:
    // the flag becoming visible does not imply the earlier passes' output
    // lines have been written back — GPU cache write-back is not ordered
    // across buffers, and the readback raced (parity tests passed, the
    // real 27B decode corrupted). Only command-buffer completion gives
    // the system-scope guarantee, and its ~1.3 ms latency is exactly why
    // the road to 10+ tok/s is FEWER submissions per token, not faster
    // waits.
    let _ = (c, outs);
    cmd.commit();
    wait_fast(cmd);
}

/// Latency-critical wait: spin-poll the status instead of
/// waitUntilCompleted (sleeping/waking the thread costs ~1–3 ms —
/// across 40 MoE layers/token this canceled out the kernel's gain).
fn wait_fast(cmd: &metal::CommandBufferRef) {
    use metal::MTLCommandBufferStatus as S;
    let t0 = std::time::Instant::now();
    loop {
        match cmd.status() {
            S::Completed | S::Error => return,
            _ => {
                if t0.elapsed().as_millis() > 200 {
                    cmd.wait_until_completed(); // safeguard against an infinite spin
                    return;
                }
                std::hint::spin_loop();
            }
        }
    }
}

fn page_size() -> usize {
    // Apple Silicon: 16 KiB; taken from sysconf without a libc dependency.
    unsafe { getpagesize() as usize }
}

unsafe extern "C" {
    fn getpagesize() -> i32;
}

#[inline]
fn model_key(model: &Arc<CmfModel>) -> usize {
    Arc::as_ptr(model) as usize
}

/// No-copy buffer over the file mapping. The cache retains the model Arc so
/// the mmap cannot disappear underneath Metal and its identity cannot be
/// recycled for another model in the same process.
fn file_buffer(c: &Ctx, model: &Arc<CmfModel>) -> Option<(Buffer, usize)> {
    let bytes = model.primary_bytes();
    let base = bytes.as_ptr() as usize;
    let key = model_key(model);
    let page = page_size();
    if base % page != 0 {
        return None; // mmap is always aligned, but we check honestly
    }
    let len = bytes.len() / page * page; // down to the page
    let mut cache = c.file_bufs.lock().unwrap();
    if let Some((b, _owner)) = cache.get(&key) {
        return Some((b.clone(), len));
    }
    crate::gpu::probe_note_cold();
    let buf = c._device.new_buffer_with_bytes_no_copy(
        bytes.as_ptr() as *const std::ffi::c_void,
        len as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    cache.insert(key, (buf.clone(), Arc::clone(model)));
    Some((buf, len))
}

/// q8_row/q8_2f matvec on the GPU. `xs` — already prescaled activations (the same
/// math as the CPU path). false = could not (the caller falls back to CPU).
#[allow(clippy::too_many_arguments)]
pub fn q8_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    q8_matvec_range(model, idx, 0, row_scale, xs, rows, cols, out)
}

/// Range variant (hybrid CPU∥GPU split): rows
/// [row0, row0+rows) of a large tensor.
#[allow(clippy::too_many_arguments)]
pub fn q8_matvec_range(
    model: &Arc<CmfModel>,
    idx: usize,
    row0: usize,
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    q8_matvec_range_field(model, idx, row0, row_scale, None, xs, rows, cols, out)
}

/// Direct q8_2f projection for parity/microbench use. The whole-token graph
/// uses the same kernel and cached field buffers.
#[allow(clippy::too_many_arguments)]
pub fn q8_2f_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    row_scale: &[f32],
    col_field: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    if col_field.len() != cols {
        return false;
    }
    q8_matvec_range_field(
        model,
        idx,
        0,
        row_scale,
        Some(col_field),
        xs,
        rows,
        cols,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn q8_matvec_range_field(
    model: &Arc<CmfModel>,
    idx: usize,
    row0: usize,
    row_scale: &[f32],
    col_field: Option<&[f32]>,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 4 != 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let Some(mut abs) = model.entry_abs_offset(entry) else {
        return false; // a neighboring shard — a different mapping; MVP: CPU
    };
    abs += row0 * cols; // offset into the sub-range (the GPU does not need 64-alignment)
    let _bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let qlen = rows * cols; // the int8 part of the blob (quants before scales)
    if abs + qlen > safe_len {
        return false; // the tail is past the buffer's page boundary
    }

    // row_scale — cached; xs/y — per call (small).
    let base = model_key(model);
    let rs_buf = {
        let mut cache = c.rs_bufs.lock().unwrap();
        cache
            .entry((base, idx + row0 * 1_000_003))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device.new_buffer_with_data(
                    row_scale.as_ptr() as *const std::ffi::c_void,
                    (row_scale.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    };
    let get_io = |nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(nbytes)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let y_buf = get_io(rows * 4 + 4); // +4: does not share a key with xs of the same length

    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(if col_field.is_some() { &c.q8f } else { &c.q8 });
    enc.set_buffer(0, Some(&fbuf), abs as u64);
    enc.set_buffer(1, Some(&xs_buf), 0);
    enc.set_buffer(2, Some(&rs_buf), 0);
    enc.set_buffer(3, Some(&y_buf), 0);
    let cols4 = (cols / 4) as u32;
    let rows_u = rows as u32;
    let base = if let Some(field) = col_field {
        let field_buf = const_buf(c, field);
        enc.set_buffer(4, Some(&field_buf), 0);
        5
    } else {
        4
    };
    enc.set_bytes(base, 4, &cols4 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(
        base + 1,
        4,
        &rows_u as *const u32 as *const std::ffi::c_void,
    );
    // 256 threads = 8 SIMD groups per threadgroup → 8 rows per group.
    let sgs = 8u64;
    let n_tg = (rows as u64).div_ceil(sgs);
    enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(sgs * 32, 1, 1));
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);

    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), rows);
    }
    true
}

/// q1 matvec on the GPU: xs is the RAW f32 activation (the scale lives
/// inside the 6-byte tiles). GPU math is plain f32 — no A8 activation
/// quantization at all, so this path is if anything more accurate than
/// the CPU int8 kernel. false = CPU fallback.
pub fn q1_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    // The kernel stages xs through threadgroup memory in tile PAIRS —
    // odd group counts (unseen in real shapes) honestly stay on CPU.
    if cols % GROUP_SIZE != 0 || (cols / GROUP_SIZE) % 2 != 0 {
        return false;
    }
    let gpr = cols / GROUP_SIZE;
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let _bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    if abs + rows * gpr * Q1_TILE > safe_len {
        return false;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(13_000_000_559 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let y_buf = get_io(14_000_000_573 + rows, rows * 4);

    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_q1_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), rows);
    }
    true
}

/// Encode one q1 matvec dispatch (shared by the single, batch and
/// MoE-chain paths).
#[allow(clippy::too_many_arguments)]
/// Kernel-pick test hook: 0 = env (default), 1 = force f32, 2 = force
/// half — lets the parity test cover BOTH kernels in one process (the
/// env choice is cached in a OnceLock and can't be toggled).
static Q1_KERNEL_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Half-accumulation q1 kernel, default on (quality gates in the
/// kernel header); CMF_Q1_HALF=0 reverts to the f32 twin.
fn q1_half() -> bool {
    match Q1_KERNEL_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            static HALF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *HALF.get_or_init(|| {
                std::env::var("CMF_Q1_HALF")
                    .map(|v| v != "0")
                    .unwrap_or(true)
            })
        }
    }
}

fn encode_q1_matvec(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    xs: &Buffer,
    y: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(if q1_half() { &c.q1h } else { &c.q1 });
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(y), 0);
    let gpr_u = gpr as u32;
    let rows_u = rows as u32;
    enc.set_bytes(3, 4, &gpr_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64; // × 4 rows per simdgroup
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs * 4), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// Encode a q1t BASE matvec (ternary, raw-f32 x). `abs` points at the tile
/// base; the overlay follows and is applied by `encode_q1t_overlay`.
fn encode_q1t_matvec(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    xs: &Buffer,
    y: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(&c.q1t);
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(y), 0);
    let gpr_u = gpr as u32;
    let rows_u = rows as u32;
    enc.set_bytes(3, 4, &gpr_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64; // × 4 rows per simdgroup
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs * 4), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// Weight layout a chunk-graph GEMM reads. Was a bare `q4t: bool`; q4tp
/// needs a third value, and a bool pair would let "neither" and "both" be
/// spelled at every call site.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MmKind {
    /// q8_row — row scales ride in a side buffer.
    Q8,
    /// q4_tiled — 18 B tiles, scale inline.
    Q4t,
    /// q4tp — 16 B nibble stride, scale from the row ladder.
    Q4tp,
}

/// Which GPU kernel a graph projection uses.
#[derive(Clone)]
enum ProjKind {
    Q1,
    Q1t,
    Q4b,
    Q4t,
    Q4tp,
    Q8 {
        row_scale: Buffer,
        col_field: Option<Buffer>,
    },
}

/// Encode q4_block matvec (precise 4-bit, no overlay). Split layout: packed
/// nibbles then scales — the shader locates the scales from rows·gpr.
fn encode_q4b_matvec(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    xs: &Buffer,
    y: &Buffer,
    rows: usize,
    gpr: usize,
) {
    static HALF: OnceLock<bool> = OnceLock::new();
    let half = *HALF.get_or_init(|| {
        std::env::var("CMF_Q4_HALF")
            .map(|v| v != "0" && v != "off")
            .unwrap_or(false)
    });
    enc.set_compute_pipeline_state(if half { &c.q4bh } else { &c.q4b });
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(y), 0);
    let gpr_u = gpr as u32;
    let rows_u = rows as u32;
    enc.set_bytes(3, 4, &gpr_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64;
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs * 4), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// Encode q4_tiled matvec: 18B interleaved tiles, one buffer, no
/// separate scale region.
#[allow(clippy::too_many_arguments)]
fn encode_q4t_matvec(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    xs: &Buffer,
    y: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(&c.q4t);
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(y), 0);
    let gpr_u = gpr as u32;
    let rows_u = rows as u32;
    enc.set_bytes(3, 4, &gpr_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64;
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs * 4), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// q4tp twin of `encode_q4t_matvec` — same 4-rows-per-simdgroup shape; the
/// kernel derives its three plane offsets from `rows`/`gpr`, so the argument
/// list stays identical to q4t's.
fn encode_q4tp_matvec(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    xs: &Buffer,
    y: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(&c.q4tp);
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(y), 0);
    let gpr_u = gpr as u32;
    let rows_u = rows as u32;
    enc.set_bytes(3, 4, &gpr_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64;
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs * 4), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
}

/// Encode a projection `in_buf → out_buf` for a Q1 / Q1T / Q4-block weight.
/// For Q1T the base matvec is followed by the on-device overlay add. Free fn so
/// it works inside the graph encode loops (which capture `c`/`fbuf`, not self).
#[allow(clippy::too_many_arguments)]
fn encode_proj(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    kind: &ProjKind,
    in_buf: &Buffer,
    out_buf: &Buffer,
    rows: usize,
    gpr: usize,
) {
    match kind {
        ProjKind::Q1t => {
            encode_q1t_matvec(c, enc, fbuf, abs, in_buf, out_buf, rows, gpr);
            encode_q1t_overlay(c, enc, fbuf, abs, in_buf, out_buf, rows, gpr);
        }
        ProjKind::Q4b => {
            encode_q4b_matvec(c, enc, fbuf, abs, in_buf, out_buf, rows, gpr);
        }
        ProjKind::Q4t => {
            encode_q4t_matvec(c, enc, fbuf, abs, in_buf, out_buf, rows, gpr);
        }
        ProjKind::Q4tp => {
            encode_q4tp_matvec(c, enc, fbuf, abs, in_buf, out_buf, rows, gpr);
        }
        ProjKind::Q1 => {
            encode_q1_matvec(c, enc, fbuf, abs, in_buf, out_buf, rows, gpr);
        }
        ProjKind::Q8 {
            row_scale,
            col_field,
        } => {
            encode_q8_matvec(
                c,
                enc,
                fbuf,
                abs,
                row_scale,
                col_field.as_ref(),
                in_buf,
                out_buf,
                rows,
                gpr,
            );
        }
    }
}

/// Encode q8_row matvec.
fn encode_q8_matvec(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    rs_buf: &Buffer,
    col_buf: Option<&Buffer>,
    in_buf: &Buffer,
    out_buf: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(if col_buf.is_some() { &c.q8f } else { &c.q8 });
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(in_buf), 0);
    enc.set_buffer(2, Some(rs_buf), 0);
    enc.set_buffer(3, Some(out_buf), 0);
    let cols4 = (gpr * (GROUP_SIZE / 4)) as u32;
    let rows_u = rows as u32;
    let base = if let Some(col) = col_buf {
        enc.set_buffer(4, Some(col), 0);
        5
    } else {
        4
    };
    enc.set_bytes(base, 4, &cols4 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(
        base + 1,
        4,
        &rows_u as *const u32 as *const std::ffi::c_void,
    );
    let sgs = 8u64;
    let n_tg = (rows as u64).div_ceil(sgs);
    enc.dispatch_thread_groups(MTLSize::new(n_tg, 1, 1), MTLSize::new(sgs * 32, 1, 1));
}

/// Encode the q1t sparse-overlay add onto `y` (base already there). Reads the
/// `[row_ptr][entries]` that follow the base at `abs`; one thread per row.
fn encode_q1t_overlay(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    xs: &Buffer,
    y: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(&c.q1t_ov);
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(y), 0);
    let base_len = (rows * gpr * Q1T_TILE) as u32;
    let rows_u = rows as u32;
    enc.set_bytes(3, 4, &base_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let tpt = 64u64;
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(tpt), 1, 1),
        MTLSize::new(tpt, 1, 1),
    );
}

/// Ternary (q1t) BASE matvec on the GPU (full-precision raw-f32 x). Fills
/// `out` with `Σ_group scale·Σ sign·x`; the caller adds the sparse outlier
/// overlay on the CPU. Returns false (→ CPU fallback) on any shape/residency
/// miss. Mirrors `q1_matvec` but reads 9-byte base-3 tiles.
pub fn q1t_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    q1t_matvec_impl(model, idx, xs, rows, cols, out, false)
}

/// Diagnostic entry point matching the whole-token graph's Q1T projection:
/// base matvec and sparse overlay are encoded in the same command buffer.
#[doc(hidden)]
pub fn q1t_matvec_full_for_test(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    q1t_matvec_impl(model, idx, xs, rows, cols, out, true)
}

fn q1t_matvec_impl(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    full: bool,
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % GROUP_SIZE != 0 {
        return false;
    }
    let gpr = cols / GROUP_SIZE;
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let _bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let need = if full {
        entry.nbytes as usize
    } else {
        rows * gpr * Q1T_TILE
    };
    if abs + need > safe_len {
        return false;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(13_000_000_559 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let y_buf = get_io(14_000_000_573 + rows, rows * 4);
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_q1t_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
    if full {
        encode_q1t_overlay(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
    }
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), rows);
    }
    true
}

/// Diagnostic entry point matching the whole-token graph's Q4Tiled
/// projection encode (no overlay — q4t is a fixed-length codec).
#[doc(hidden)]
pub fn q4t_matvec_for_test(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % GROUP_SIZE != 0 {
        return false;
    }
    let gpr = cols / GROUP_SIZE;
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let _bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    if abs + rows * gpr * (2 + GROUP_SIZE / 2) > safe_len {
        return false;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(13_000_000_559 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let y_buf = get_io(14_000_000_573 + rows, rows * 4);
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_q4t_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), rows);
    }
    true
}

/// q4tp twin of `q4t_matvec_for_test` — the same encode the whole-token
/// graph uses, exposed so a test can hold the GPU kernel against the CPU one.
#[doc(hidden)]
pub fn q4tp_matvec_for_test(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % GROUP_SIZE != 0 {
        return false;
    }
    let gpr = cols / GROUP_SIZE;
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let Some(need) = cortiq_core::quant::expected_nbytes(
        cortiq_core::TensorDtype::Q4TiledP,
        &[rows, cols],
    ) else {
        return false;
    };
    if abs + need > safe_len {
        return false;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(15_000_000_611 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let y_buf = get_io(16_000_000_627 + rows, rows * 4);
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_q4tp_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), rows);
    }
    true
}

/// Time `reps` back-to-back matvec dispatches in ONE command buffer, so the
/// number is kernel cost and not submit latency — a single-dispatch timing
/// is ~0.25 ms of round trip on Metal and hides everything smaller.
/// Returns seconds per dispatch. Picks the kernel from the tensor's dtype.
#[doc(hidden)]
pub fn q4_matvec_bench(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    reps: usize,
) -> Option<f64> {
    let c = ctx()?;
    if cols % GROUP_SIZE != 0 {
        return None;
    }
    let gpr = cols / GROUP_SIZE;
    let entry = &model.tensors[idx];
    let abs = model.entry_abs_offset(entry)?;
    let (fbuf, safe_len) = file_buffer(c, model)?;
    let need = cortiq_core::quant::expected_nbytes(entry.dtype, &[rows, cols])?;
    if abs + need > safe_len {
        return None;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(17_000_000_633 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let y_buf = get_io(18_000_000_641 + rows, rows * 4);
    let tp = entry.dtype == cortiq_core::TensorDtype::Q4TiledP;
    let mut best = f64::MAX;
    for _ in 0..3 {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..reps {
            if tp {
                encode_q4tp_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
            } else {
                encode_q4t_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
            }
        }
        enc.end_encoding();
        let t0 = std::time::Instant::now();
        submit_and_wait(c, cmd, &[&y_buf]);
        best = best.min(t0.elapsed().as_secs_f64() / reps as f64);
    }
    Some(best)
}

/// Sweep: ONE dispatch per tensor across a whole list, in one command
/// buffer. Repeating a single tensor keeps its scale planes cache-hot, which
/// flatters q4tp; the model touches each tensor once per token, so this is
/// the access pattern that decides. Returns seconds for the whole sweep.
#[doc(hidden)]
pub fn q4_matvec_sweep(
    model: &Arc<CmfModel>,
    tensors: &[(usize, usize, usize)],
    serial: bool,
) -> Option<f64> {
    let c = ctx()?;
    let (fbuf, safe_len) = file_buffer(c, model)?;
    let maxc = tensors.iter().map(|t| t.2).max()?;
    let maxr = tensors.iter().map(|t| t.1).max()?;
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(19_000_000_643 + maxc, maxc * 4);
    let y_buf = get_io(20_000_000_649 + maxr, maxr * 4);
    let mut plan = Vec::with_capacity(tensors.len());
    for &(idx, rows, cols) in tensors {
        let entry = &model.tensors[idx];
        let abs = model.entry_abs_offset(entry)?;
        let need = cortiq_core::quant::expected_nbytes(entry.dtype, &[rows, cols])?;
        if cols % GROUP_SIZE != 0 || abs + need > safe_len {
            return None;
        }
        plan.push((
            abs,
            rows,
            cols / GROUP_SIZE,
            entry.dtype == cortiq_core::TensorDtype::Q4TiledP,
        ));
    }
    // `serial` puts each dispatch in its OWN encoder. Metal fences tracked
    // buffers across encoder boundaries, so the dispatches stop overlapping —
    // which is the regime the token graph actually runs in, every projection
    // feeding the next. Overlapped, a sweep measures bandwidth; serialized, it
    // measures the per-dispatch latency the model pays.
    let mut best = f64::MAX;
    for _ in 0..4 {
        let cmd = c.queue.new_command_buffer();
        if serial {
            for &(abs, rows, gpr, tp) in &plan {
                let enc = cmd.new_compute_command_encoder();
                if tp {
                    encode_q4tp_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
                } else {
                    encode_q4t_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
                }
                enc.end_encoding();
            }
        } else {
            let enc = cmd.new_compute_command_encoder();
            for &(abs, rows, gpr, tp) in &plan {
                if tp {
                    encode_q4tp_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
                } else {
                    encode_q4t_matvec(c, enc, &fbuf, abs, &xs_buf, &y_buf, rows, gpr);
                }
            }
            enc.end_encoding();
        }
        let t0 = std::time::Instant::now();
        submit_and_wait(c, cmd, &[&y_buf]);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    Some(best)
}

/// GEMM prefill batch: pre — prescaled inputs row-major [b, cols],
/// out — row-major [b, rows]. false = CPU path.
#[allow(clippy::too_many_arguments)]
/// f32 → f16 bulk convert into a raw destination (the mul_mm X upload).
/// NEON vcvt on aarch64; scalar bit-twiddle elsewhere.
fn f32_to_f16_into(src: &[f32], dst: *mut u16) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let n = src.len();
        let sp = src.as_ptr();
        let mut i = 0usize;
        while i + 4 <= n {
            let v = vld1q_f32(sp.add(i));
            let h = vcvt_f16_f32(v);
            core::ptr::write_unaligned(
                dst.add(i) as *mut u64,
                core::mem::transmute::<float16x4_t, u64>(h),
            );
            i += 4;
        }
        while i < n {
            *dst.add(i) = cortiq_core::quant::f32_to_f16(*sp.add(i));
            i += 1;
        }
        return;
    }
    #[allow(unreachable_code)]
    for (i, &v) in src.iter().enumerate() {
        unsafe { *dst.add(i) = cortiq_core::quant::f32_to_f16(v) };
    }
}

/// Shape-specialized mul_mm pipeline (cols/rows as function constants —
/// fully unrolled K loop, strength-reduced addressing). Falls back to
/// the generic pipeline if specialization fails.
fn mm_pipeline(c: &Ctx, rows: usize, cols: usize, kind: u8) -> ComputePipelineState {
    let mut cache = c.mm_fc.lock().unwrap();
    cache
        .entry((rows as u32, cols as u32, kind))
        .or_insert_with(|| {
            crate::gpu::probe_note_cold();
            let fcv = metal::FunctionConstantValues::new();
            let cols_u = cols as u32;
            let rows_u = rows as u32;
            // f32nt specializes cols only (rows = context, varies);
            // f32nn specializes rows only (kdim varies).
            if kind != 3 {
                fcv.set_constant_value_at_index(
                    &cols_u as *const u32 as *const std::ffi::c_void,
                    metal::MTLDataType::UInt,
                    0,
                );
            }
            if kind != 2 {
                fcv.set_constant_value_at_index(
                    &rows_u as *const u32 as *const std::ffi::c_void,
                    metal::MTLDataType::UInt,
                    1,
                );
            }
            let (name, generic) = match kind {
                1 => ("q8_mul_mm_silu", &c.q8mmsilu),
                2 => ("mul_mm_f32nt", &c.mmf32nt),
                3 => ("mul_mm_f32nn", &c.mmf32nn),
                _ => ("q8_mul_mm", &c.q8mmm),
            };
            c.lib
                .get_function(name, Some(fcv))
                .ok()
                .and_then(|f| c._device.new_compute_pipeline_state_with_function(&f).ok())
                .unwrap_or_else(|| generic.clone())
        })
        .clone()
}

/// Encode one tiled GEMM into an open command buffer (device-resident X
/// and Y). `q4t` picks the 18-byte-tile kernel, which reads its scales
/// from the tiles and ignores `rs_buf`; otherwise q8_row with row
/// scales. Caller guarantees b ≥ 32 and cols % 4 == 0.
#[allow(clippy::too_many_arguments)]
fn enc_mul_mm(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    rs_buf: &Buffer,
    kind: MmKind,
    xs: &Buffer,
    y: &Buffer,
    b: usize,
    rows: usize,
    cols: usize,
) {
    // q4t has no function-constant twin (its K loop is already tile-shaped
    // and fully unrolled over the 18 B group), so it takes the generic
    // pipeline; q8 keeps the cols/rows-specialized one.
    let (cols_u, rows_u, b_u) = (cols as u32, rows as u32, b as u32);
    if kind != MmKind::Q8 {
        enc.set_compute_pipeline_state(if kind == MmKind::Q4tp {
            &c.q4tpmm
        } else {
            &c.q4tmm
        });
        enc.set_buffer(0, Some(fbuf), abs as u64);
        enc.set_buffer(1, Some(xs), 0);
        enc.set_buffer(2, Some(y), 0);
        enc.set_bytes(3, 4, &cols_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &b_u as *const u32 as *const std::ffi::c_void);
    } else {
        let pso = mm_pipeline(c, rows, cols, 0);
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(fbuf), abs as u64);
        enc.set_buffer(1, Some(xs), 0);
        enc.set_buffer(2, Some(rs_buf), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_bytes(4, 4, &cols_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &b_u as *const u32 as *const std::ffi::c_void);
    }
    enc.dispatch_thread_groups(
        MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
        MTLSize::new(128, 1, 1),
    );
}

#[allow(clippy::too_many_arguments)]
fn encode_mul_mm(
    c: &Ctx,
    cmd: &metal::CommandBufferRef,
    fbuf: &Buffer,
    abs: usize,
    rs_buf: &Buffer,
    kind: MmKind,
    xs: &Buffer,
    y: &Buffer,
    b: usize,
    rows: usize,
    cols: usize,
) {
    let enc = cmd.new_compute_command_encoder();
    enc_mul_mm(c, enc, fbuf, abs, rs_buf, kind, xs, y, b, rows, cols);
    enc.end_encoding();
}

/// One full-attention prefill layer on q8_row weights, device-resident
/// through the whole chunk (roadmap: the llama.cpp Metal pp512 class).
pub struct ChunkLayer<'a> {
    pub model: &'a Arc<CmfModel>,
    pub kv_id: u64,
    pub layer: usize,
    /// (idx, rows, cols, row_scale) per projection — all q8_row.
    pub wq: (usize, usize, usize, &'a [f32]),
    pub wk: (usize, usize, usize, &'a [f32]),
    pub wv: (usize, usize, usize, &'a [f32]),
    pub wo: (usize, usize, usize, &'a [f32]),
    pub gate: (usize, usize, usize, &'a [f32]),
    pub up: (usize, usize, usize, &'a [f32]),
    pub down: (usize, usize, usize, &'a [f32]),
    pub input_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub bias: Option<(&'a [f32], &'a [f32], &'a [f32])>,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub inv_freq: &'a [f32],
    pub rd: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub hs: usize,
    pub inter: usize,
    pub gemma: bool,
    pub eps: f32,
}

/// Run a RUN of consecutive prefill layers for the whole chunk in a
/// single submission: per layer — norm → QKV GEMMs → bias+qk-norm+RoPE
/// with fused mirror append → causal chunk attend (+Born importance) →
/// O GEMM → residual → norm → gate/up GEMMs → silu·mul → down GEMM →
/// residual. The hidden buffer stays device-resident across the whole
/// run; ONE wait at the end, then every layer's chunk K/V rows and
/// importance masses come back for the CPU caches (owners of record).
/// Validation is all-before-encoding; a layer that fails during mirror
/// prep leaves at most an advanced `stored` counter behind, which the
/// self-healing resync repairs on the next touch. Returns false with
/// nothing encoded if ANY layer of the run is ineligible — the caller
/// decides run boundaries.
pub struct ChunkIo<'a> {
    pub cpu_stored: usize,
    pub cpu_k: Vec<&'a [f32]>,
    pub cpu_v: Vec<&'a [f32]>,
    pub out_k: &'a mut [f32],
    pub out_v: &'a mut [f32],
    pub imp: &'a mut [f32],
}

struct ChunkPrep {
    abs: [usize; 7],
    rs: [Buffer; 7],
    /// Per-projection weight layout (`rs` carries row scales for Q8 only).
    kind: [MmKind; 7],
    k_mb: Buffer,
    v_mb: Buffer,
    imp_mb: Buffer,
    cap: usize,
    st0: usize,
}

/// GPU time of a completed command buffer (GPUEndTime − GPUStartTime),
/// in milliseconds — metal-rs does not surface the getters, raw objc
/// does. Gaps BETWEEN buffers are not attributed to either side, which
/// is exactly what per-stage attribution wants.
fn cmd_gpu_ms(cmd: &metal::CommandBufferRef) -> f64 {
    use metal::foreign_types::ForeignTypeRef;
    use metal::objc::{msg_send, sel, sel_impl};
    unsafe {
        let p = cmd.as_ptr();
        let s: f64 = msg_send![p, GPUStartTime];
        let e: f64 = msg_send![p, GPUEndTime];
        (e - s) * 1000.0
    }
}

/// Stage-attribution mode for the chunk graph (CMF_CHUNK_PROF=1): each
/// stage is committed as its OWN command buffer so its GPU time can be
/// read back per stage. The queue keeps ordering; wall time inflates
/// (submit per stage), the per-stage GPU times stay honest.
struct ChunkProf {
    on: bool,
    log: Vec<(&'static str, metal::CommandBuffer)>,
}

impl ChunkProf {
    fn new() -> Self {
        Self {
            on: std::env::var("CMF_CHUNK_PROF")
                .map(|v| v == "1")
                .unwrap_or(false),
            log: Vec::new(),
        }
    }
    /// Close the current buffer under `label` and open a fresh one.
    fn cut(
        &mut self,
        c: &Ctx,
        cmd: metal::CommandBuffer,
        label: &'static str,
    ) -> metal::CommandBuffer {
        if !self.on {
            return cmd;
        }
        cmd.commit();
        self.log.push((label, cmd));
        c.queue.new_command_buffer().to_owned()
    }
    fn report(&self) {
        if !self.on || self.log.is_empty() {
            return;
        }
        let mut agg: std::collections::HashMap<&'static str, (f64, usize)> =
            std::collections::HashMap::new();
        for (label, cmd) in &self.log {
            let e = agg.entry(label).or_insert((0.0, 0));
            e.0 += cmd_gpu_ms(cmd);
            e.1 += 1;
        }
        let mut rows: Vec<_> = agg.into_iter().collect();
        rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        let total: f64 = rows.iter().map(|r| r.1.0).sum();
        eprintln!("chunk prof (GPU ms per stage, one chunk):");
        for (label, (ms, n)) in rows {
            eprintln!(
                "  {label:<12} {ms:8.2} ms  ({n:3}×)  {:4.1}%",
                ms / total * 100.0
            );
        }
        eprintln!("  total GPU    {total:8.2} ms");
    }
}

/// Optional on-device embedding for the chunk: (tensor idx, vocab rows,
/// row_scale, token ids, multiplier). q8_row only — anything else keeps
/// the CPU embed.
pub struct ChunkEmbed<'a> {
    pub idx: usize,
    pub rows: usize,
    pub row_scale: &'a [f32],
    pub ids: &'a [u32],
    pub mult: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn chunk_run_gpu(
    layers: &[ChunkLayer],
    io: &mut [ChunkIo],
    h: &mut [f32],
    b: usize,
    pos0: usize,
    embed: Option<&ChunkEmbed>,
) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(first) = layers.first() else {
        return false;
    };
    if layers.len() != io.len() {
        return false;
    }
    let (nh, nkv, hd, hs, inter) = (first.nh, first.nkv, first.hd, first.hs, first.inter);
    if b < 32
        || hd % 4 != 0
        || hd > 256
        || first.rd < 2
        || first.rd > hd
        || (first.rd / 2) % 32 != 0
        || nh % nkv.max(1) != 0
        || hs % 4 != 0
        || inter % 4 != 0
        || h.len() < b * hs
    {
        return false;
    }
    let Some((fbuf, safe_len)) = file_buffer(c, first.model) else {
        return false;
    };
    let base = model_key(first.model);

    // ── Phase 1: validate every layer and build its prep (weights
    // resident, shapes uniform, mirror ready).
    let mut preps: Vec<ChunkPrep> = Vec::with_capacity(layers.len());
    for (l, lio) in layers.iter().zip(io.iter()) {
        if l.nh != nh || l.nkv != nkv || l.hd != hd || l.hs != hs || l.inter != inter {
            return false;
        }
        // An empty row_scale marks q4_tiled (scales inside the tiles);
        // its payload is 18 B per 32-weight group, not one byte per
        // weight, so the bounds check differs.
        // Layout comes from the tensor directory, not from "row_scale is
        // empty" — that heuristic could only ever spell two of the three.
        let kind_of = |t: &(usize, usize, usize, &[f32])| -> Option<MmKind> {
            Some(match l.model.tensors.get(t.0)?.dtype {
                cortiq_core::TensorDtype::Q4Tiled => MmKind::Q4t,
                cortiq_core::TensorDtype::Q4TiledP => MmKind::Q4tp,
                _ => MmKind::Q8,
            })
        };
        let abs_of = |t: &(usize, usize, usize, &[f32])| -> Option<usize> {
            let entry = l.model.tensors.get(t.0)?;
            let abs = l.model.entry_abs_offset(entry)?;
            let bytes = match kind_of(t)? {
                MmKind::Q4t => {
                    if t.2 % GROUP_SIZE != 0 {
                        return None;
                    }
                    t.1 * (t.2 / GROUP_SIZE) * Q4_TILE
                }
                MmKind::Q4tp => cortiq_core::quant::expected_nbytes(
                    cortiq_core::TensorDtype::Q4TiledP,
                    &[t.1, t.2],
                )?,
                MmKind::Q8 => t.1 * t.2,
            };
            (abs + bytes <= safe_len).then_some(abs)
        };
        let tens = [&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down];
        let mut abs = [0usize; 7];
        for (slot, t) in abs.iter_mut().zip(tens) {
            match abs_of(t) {
                Some(a) => *slot = a,
                None => return false,
            }
        }
        if l.wq.1 != nh * hd
            || l.wk.1 != nkv * hd
            || l.wv.1 != nkv * hd
            || l.wo.1 != hs
            || l.wo.2 != nh * hd
            || l.gate.1 != inter
            || l.up.1 != inter
            || l.down.1 != hs
            || l.down.2 != inter
            || l.inv_freq.len() < l.rd / 2
            || lio.out_k.len() < b * nkv * hd
            || lio.out_v.len() < b * nkv * hd
            || lio.imp.len() < lio.cpu_stored + b
        {
            return false;
        }
        let rs_of = |t: &(usize, usize, usize, &[f32])| -> Buffer {
            let mut cache = c.rs_bufs.lock().unwrap();
            cache
                .entry((base, t.0))
                .or_insert_with(|| {
                    crate::gpu::probe_note_cold();
                    // q4t carries no row scales; a zero-length Metal
                    // buffer is invalid, so bind a 4-byte placeholder the
                    // q4t kernels never read.
                    if t.3.is_empty() {
                        return c._device.new_buffer(4, MTLResourceOptions::StorageModeShared);
                    }
                    c._device.new_buffer_with_data(
                        t.3.as_ptr() as *const std::ffi::c_void,
                        (t.3.len() * 4) as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                })
                .clone()
        };
        let rs = [
            rs_of(&l.wq),
            rs_of(&l.wk),
            rs_of(&l.wv),
            rs_of(&l.wo),
            rs_of(&l.gate),
            rs_of(&l.up),
            rs_of(&l.down),
        ];
        let kind = match [
            kind_of(&l.wq),
            kind_of(&l.wk),
            kind_of(&l.wv),
            kind_of(&l.wo),
            kind_of(&l.gate),
            kind_of(&l.up),
            kind_of(&l.down),
        ] {
            [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f), Some(g)] => {
                [a, b, c, d, e, f, g]
            }
            _ => return false,
        };
        // KV mirror prep (self-healing contract of the decode graph),
        // reserving b rows for the chunk.
        let (k_mb, v_mb, imp_mb, cap, st0) = {
            let mut reg = c.kv_mirrors.lock().unwrap();
            let need = lio.cpu_stored + b;
            let entry = reg.entry((l.kv_id, l.layer)).or_insert_with(|| KvMirror {
                k: c._device
                    .new_buffer(0, MTLResourceOptions::StorageModeShared),
                v: c._device
                    .new_buffer(0, MTLResourceOptions::StorageModeShared),
                imp: c
                    ._device
                    .new_buffer(0, MTLResourceOptions::StorageModeShared),
                cap: 0,
                stored: usize::MAX,
            });
            if entry.cap < need {
                let cap = need.next_power_of_two().max(1024);
                let nb = (nkv * cap * hd * 4) as u64;
                entry.k = c
                    ._device
                    .new_buffer(nb, MTLResourceOptions::StorageModeShared);
                entry.v = c
                    ._device
                    .new_buffer(nb, MTLResourceOptions::StorageModeShared);
                entry.imp = c
                    ._device
                    .new_buffer((cap * 4) as u64, MTLResourceOptions::StorageModeShared);
                entry.cap = cap;
                entry.stored = usize::MAX;
            }
            if entry.stored != lio.cpu_stored {
                if lio.cpu_k.len() != nkv || lio.cpu_v.len() != nkv {
                    return false;
                }
                for hh in 0..nkv {
                    if lio.cpu_k[hh].len() != lio.cpu_stored * hd
                        || lio.cpu_v[hh].len() != lio.cpu_stored * hd
                    {
                        return false;
                    }
                    unsafe {
                        let kd = (entry.k.contents() as *mut f32).add(hh * entry.cap * hd);
                        std::ptr::copy_nonoverlapping(
                            lio.cpu_k[hh].as_ptr(),
                            kd,
                            lio.cpu_k[hh].len(),
                        );
                        let vd = (entry.v.contents() as *mut f32).add(hh * entry.cap * hd);
                        std::ptr::copy_nonoverlapping(
                            lio.cpu_v[hh].as_ptr(),
                            vd,
                            lio.cpu_v[hh].len(),
                        );
                    }
                }
                entry.stored = lio.cpu_stored;
            }
            unsafe {
                std::ptr::write_bytes(entry.imp.contents() as *mut u8, 0, need * 4);
            }
            let out = (
                entry.k.clone(),
                entry.v.clone(),
                entry.imp.clone(),
                entry.cap,
                entry.stored,
            );
            entry.stored += b;
            out
        };
        preps.push(ChunkPrep {
            abs,
            rs,
            kind,
            k_mb,
            v_mb,
            imp_mb,
            cap,
            st0,
        });
    }

    // ── Shared per-run buffers (pooled by size, reused across layers —
    // encoder ordering within one command buffer serializes access).
    let h_b = io_buf(c, 60_000_000_071 + b * hs, b * hs * 4);
    let n_b = io_buf(c, 61_000_000_091 + b * hs, b * hs * 4);
    let qraw = io_buf(c, 62_000_000_017 + b * nh * hd, b * nh * hd * 4);
    let kraw = io_buf(c, 63_000_000_029 + b * nkv * hd, b * nkv * hd * 4);
    let vraw = io_buf(c, 64_000_000_063 + b * nkv * hd, b * nkv * hd * 4);
    let qrope = io_buf(c, 65_000_000_087 + b * nh * hd, b * nh * hd * 4);
    let attn = io_buf(c, 66_000_000_103 + b * nh * hd, b * nh * hd * 4);
    let apanel = io_buf(c, 73_000_000_117 + b * nh * hd, b * nh * hd * 4);
    let ob = io_buf(c, 67_000_000_141 + b * hs, b * hs * 4);
    let gb = io_buf(c, 68_000_000_169 + b * inter, b * inter * 4);
    let ub = io_buf(c, 69_000_000_213 + b * inter, b * inter * 4);
    let db = io_buf(c, 71_000_000_073 + b * hs, b * hs * 4);
    // Embedding source: validated up front; refusal keeps the CPU h.
    let embed_prep: Option<(usize, Buffer, Buffer)> = embed.and_then(|e| {
        if e.ids.len() < b || e.row_scale.len() < e.rows {
            return None;
        }
        let entry = layers[0].model.tensors.get(e.idx)?;
        let abs = layers[0].model.entry_abs_offset(entry)?;
        if abs + e.rows * hs > safe_len || e.ids.iter().any(|&id| id as usize >= e.rows) {
            return None;
        }
        let rs_buf = {
            let mut cache = c.rs_bufs.lock().unwrap();
            cache
                .entry((base, e.idx))
                .or_insert_with(|| {
                    crate::gpu::probe_note_cold();
                    c._device.new_buffer_with_data(
                        e.row_scale.as_ptr() as *const std::ffi::c_void,
                        (e.row_scale.len() * 4) as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                })
                .clone()
        };
        let ids_buf = io_buf(c, 74_000_000_177 + b, b * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(e.ids.as_ptr(), ids_buf.contents() as *mut u32, b);
        }
        Some((abs, rs_buf, ids_buf))
    });
    if embed.is_some() && embed_prep.is_none() {
        // The caller deferred the CPU embed expecting the device to do
        // it — refuse the whole run (advanced mirror counters self-heal
        // on the next touch) rather than silently prefill from zeros.
        return false;
    }
    if embed_prep.is_none() {
        unsafe {
            std::ptr::copy_nonoverlapping(h.as_ptr(), h_b.contents() as *mut f32, b * hs);
        }
    }

    let mut prof = ChunkProf::new();
    // The last layer's down-delta rides into the NEXT layer's fused
    // add+norm; before the first layer there is nothing pending.
    let mut pending_delta = false;
    let mut cmd = c.queue.new_command_buffer().to_owned();
    if let (Some((abs, rs_buf, ids_buf)), Some(e)) = (&embed_prep, embed) {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.embedq8);
        enc.set_buffer(0, Some(&fbuf), *abs as u64);
        enc.set_buffer(1, Some(rs_buf), 0);
        enc.set_buffer(2, Some(ids_buf), 0);
        enc.set_buffer(3, Some(&h_b), 0);
        let (hs_u, nb_u) = (hs as u32, b as u32);
        enc.set_bytes(4, 4, &hs_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &e.mult as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            MTLSize::new(hs as u64, b as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd = prof.cut(c, cmd, "embed");
    }
    for (l, prep) in layers.iter().zip(&preps) {
        let inorm = const_buf(c, l.input_norm);
        let pnorm = const_buf(c, l.post_norm);
        let invf = const_buf(c, &l.inv_freq[..l.rd / 2]);
        let (bqb, bkb, bvb, has_bias) = match l.bias {
            Some((bq, bk, bv)) => (const_buf(c, bq), const_buf(c, bk), const_buf(c, bv), true),
            None => (invf.clone(), invf.clone(), invf.clone(), false),
        };
        let qn_b = l
            .q_norm
            .map(|w| const_buf(c, w))
            .unwrap_or_else(|| invf.clone());
        let kn_b = l
            .k_norm
            .map(|w| const_buf(c, w))
            .unwrap_or_else(|| invf.clone());
        let add_norm =
            |cmd: &metal::CommandBufferRef, delta: Option<&Buffer>, w: &Buffer, dst: &Buffer| {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&c.addnorm);
                enc.set_buffer(0, Some(&h_b), 0);
                enc.set_buffer(1, Some(delta.unwrap_or(&h_b)), 0);
                enc.set_buffer(2, Some(w), 0);
                enc.set_buffer(3, Some(dst), 0);
                let n_u = hs as u32;
                let g_u = l.gemma as u32;
                let hd_u = delta.is_some() as u32;
                enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &g_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &l.eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &hd_u as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(MTLSize::new(b as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            };

        // First stage folds the PREVIOUS layer's down-projection delta
        // into the residual stream together with this layer's input
        // norm — one pass, no standalone axpy encoder at layer end.
        add_norm(&cmd, pending_delta.then_some(&db), &inorm, &n_b);
        pending_delta = true;
        cmd = prof.cut(c, cmd, "norm");
        {
            // Independent outputs — one encoder, three dispatches.
            let enc = cmd.new_compute_command_encoder();
            enc_mul_mm(
                c,
                enc,
                &fbuf,
                prep.abs[0],
                &prep.rs[0],
                prep.kind[0],
                &n_b,
                &qraw,
                b,
                l.wq.1,
                l.wq.2,
            );
            enc_mul_mm(
                c,
                enc,
                &fbuf,
                prep.abs[1],
                &prep.rs[1],
                prep.kind[1],
                &n_b,
                &kraw,
                b,
                l.wk.1,
                l.wk.2,
            );
            enc_mul_mm(
                c,
                enc,
                &fbuf,
                prep.abs[2],
                &prep.rs[2],
                prep.kind[2],
                &n_b,
                &vraw,
                b,
                l.wv.1,
                l.wv.2,
            );
            enc.end_encoding();
        }
        cmd = prof.cut(c, cmd, "mm_qkv");
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.cropekv);
            for (i, buf) in [
                &qraw, &kraw, &vraw, &qrope, &prep.k_mb, &prep.v_mb, &bqb, &bkb, &bvb, &qn_b,
                &kn_b, &invf,
            ]
            .iter()
            .enumerate()
            {
                enc.set_buffer(i as u64, Some(buf), 0);
            }
            let flags = ((l.q_norm.is_some() as u32) << 1)
                | ((l.k_norm.is_some() as u32) << 2)
                | ((l.gemma as u32) << 3)
                | ((has_bias as u32) << 4);
            let words = [
                nh as u32,
                nkv as u32,
                hd as u32,
                l.rd as u32,
                pos0 as u32,
                prep.st0 as u32,
                prep.cap as u32,
                flags,
            ];
            for (i, w) in words.iter().enumerate() {
                enc.set_bytes(12 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
            }
            enc.set_bytes(20, 4, &l.eps as *const f32 as *const std::ffi::c_void);
            let nb_u = b as u32;
            enc.set_bytes(21, 4, &nb_u as *const u32 as *const std::ffi::c_void);
            let sgs = 8u64;
            enc.dispatch_thread_groups(
                MTLSize::new(((nh + 2 * nkv) as u64).div_ceil(sgs), b as u64, 1),
                MTLSize::new(sgs * 32, 1, 1),
            );
            enc.end_encoding();
        }
        cmd = prof.cut(c, cmd, "rope_kv");
        // GEMM attention (profiled: the streaming attend was 47% of the
        // chunk): scores = Qpanel·Kᵀ·scale per KV group, causal softmax
        // rows, Born column sums, attn = P·V. Groups get their own
        // score REGIONS so same-stage dispatches of every group share
        // one encoder and may overlap; the imp and P·V passes both only
        // read the softmaxed scores and merge into one encoder too.
        {
            let hpk = nh / nkv.max(1);
            let ncur = prep.st0 + b;
            let m_rows = hpk * b;
            let g_stride = (m_rows * ncur * 4) as u64;
            let scores = io_buf(
                c,
                72_000_000_089 + nkv * m_rows * ncur,
                nkv * m_rows * ncur * 4,
            );
            let scale = 1.0f32 / (hd as f32).sqrt();
            {
                let enc = cmd.new_compute_command_encoder();
                let pso = mm_pipeline(c, 0, hd, 2);
                enc.set_compute_pipeline_state(&pso);
                for g in 0..nkv {
                    let koff = (g * prep.cap * hd * 4) as u64;
                    let qoff = (g * hpk * b * hd * 4) as u64;
                    enc.set_buffer(0, Some(&prep.k_mb), koff);
                    enc.set_buffer(1, Some(&qrope), qoff);
                    enc.set_buffer(2, Some(&scores), g as u64 * g_stride);
                    let (cols_u, rows_u, nb_u) = (hd as u32, ncur as u32, m_rows as u32);
                    enc.set_bytes(3, 4, &cols_u as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &scale as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new((m_rows as u64).div_ceil(32), (ncur as u64).div_ceil(64), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc.end_encoding();
            }
            cmd = prof.cut(c, cmd, "att_qk");
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&c.csmax);
                for g in 0..nkv {
                    enc.set_buffer(0, Some(&scores), g as u64 * g_stride);
                    let words = [ncur as u32, prep.st0 as u32, b as u32, m_rows as u32];
                    for (i, w) in words.iter().enumerate() {
                        enc.set_bytes(1 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
                    }
                    let sgs = 8u64;
                    enc.dispatch_thread_groups(
                        MTLSize::new((m_rows as u64).div_ceil(sgs), 1, 1),
                        MTLSize::new(sgs * 32, 1, 1),
                    );
                }
                enc.end_encoding();
            }
            cmd = prof.cut(c, cmd, "att_sm");
            {
                // Born sums and P·V both only READ the softmaxed scores
                // — one encoder, they may overlap.
                let enc = cmd.new_compute_command_encoder();
                for g in 0..nkv {
                    enc.set_compute_pipeline_state(&c.impcol);
                    enc.set_buffer(0, Some(&scores), g as u64 * g_stride);
                    enc.set_buffer(1, Some(&prep.imp_mb), 0);
                    let words = [ncur as u32, m_rows as u32];
                    for (i, w) in words.iter().enumerate() {
                        enc.set_bytes(2 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
                    }
                    enc.dispatch_threads(MTLSize::new(ncur as u64, 32, 1), MTLSize::new(64, 4, 1));
                    let pso = mm_pipeline(c, hd, 0, 3);
                    enc.set_compute_pipeline_state(&pso);
                    let koff = (g * prep.cap * hd * 4) as u64;
                    let qoff = (g * hpk * b * hd * 4) as u64;
                    enc.set_buffer(0, Some(&prep.v_mb), koff);
                    enc.set_buffer(1, Some(&scores), g as u64 * g_stride);
                    enc.set_buffer(2, Some(&apanel), qoff);
                    let (k_u, rows_u, nb_u) = (ncur as u32, hd as u32, m_rows as u32);
                    enc.set_bytes(3, 4, &k_u as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new((m_rows as u64).div_ceil(32), (hd as u64).div_ceil(64), 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
                enc.end_encoding();
            }
            cmd = prof.cut(c, cmd, "att_pv");
            // panel [head][bi][hd] → [bi][nh·hd] for the O GEMM.
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.unstack);
            enc.set_buffer(0, Some(&apanel), 0);
            enc.set_buffer(1, Some(&attn), 0);
            let words = [nh as u32, b as u32, hd as u32];
            for (i, w) in words.iter().enumerate() {
                enc.set_bytes(2 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
            }
            enc.dispatch_threads(
                MTLSize::new((nh * b * hd) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }
        cmd = prof.cut(c, cmd, "attend");
        encode_mul_mm(
            c,
            &cmd,
            &fbuf,
            prep.abs[3],
            &prep.rs[3],
            prep.kind[3],
            &attn,
            &ob,
            b,
            l.wo.1,
            l.wo.2,
        );
        cmd = prof.cut(c, cmd, "mm_o");
        add_norm(&cmd, Some(&ob), &pnorm, &n_b);
        cmd = prof.cut(c, cmd, "axpy+norm");
        {
            let enc = cmd.new_compute_command_encoder();
            enc_mul_mm(
                c,
                enc,
                &fbuf,
                prep.abs[4],
                &prep.rs[4],
                prep.kind[4],
                &n_b,
                &gb,
                b,
                l.gate.1,
                l.gate.2,
            );
            enc_mul_mm(
                c,
                enc,
                &fbuf,
                prep.abs[5],
                &prep.rs[5],
                prep.kind[5],
                &n_b,
                &ub,
                b,
                l.up.1,
                l.up.2,
            );
            enc.end_encoding();
        }
        cmd = prof.cut(c, cmd, "mm_gateup");
        // down GEMM with silu(g)·u fused into the X-tile load — no
        // standalone activation stage, no act-buffer round trip.
        {
            let enc = cmd.new_compute_command_encoder();
            let (cols_u, rows_u, b_u) = (l.down.2 as u32, l.down.1 as u32, b as u32);
            // q4t drops the row-scale buffer, so every constant after it
            // shifts down one slot.
            let base = if prep.kind[6] != MmKind::Q8 {
                enc.set_compute_pipeline_state(if prep.kind[6] == MmKind::Q4tp {
                    &c.q4tpmmsilu
                } else {
                    &c.q4tmmsilu
                });
                enc.set_buffer(0, Some(&fbuf), prep.abs[6] as u64);
                enc.set_buffer(1, Some(&gb), 0);
                enc.set_buffer(2, Some(&ub), 0);
                enc.set_buffer(3, Some(&db), 0);
                4
            } else {
                let pso = mm_pipeline(c, l.down.1, l.down.2, 1);
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&fbuf), prep.abs[6] as u64);
                enc.set_buffer(1, Some(&gb), 0);
                enc.set_buffer(2, Some(&ub), 0);
                enc.set_buffer(3, Some(&prep.rs[6]), 0);
                enc.set_buffer(4, Some(&db), 0);
                5
            };
            enc.set_bytes(base, 4, &cols_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(base + 1, 4, &rows_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(base + 2, 4, &b_u as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((b as u64).div_ceil(32), (l.down.1 as u64).div_ceil(64), 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        }
        cmd = prof.cut(c, cmd, "mm_down");
        // Early commit (decode-graph lesson): hand this layer to the
        // GPU now and encode the next one while it runs — the queue
        // keeps ordering, only the last buffer is waited on. Without
        // this the GPU sits idle through the whole chunk's encode.
        if !prof.on {
            cmd.commit();
            cmd = c.queue.new_command_buffer().to_owned();
        }
    }

    // Flush the final layer's pending down-delta into the stream.
    if pending_delta {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.axpy);
        enc.set_buffer(0, Some(&db), 0);
        enc.set_buffer(1, Some(&h_b), 0);
        let w1 = 1.0f32;
        let n_u = (b * hs) as u32;
        enc.set_bytes(2, 4, &w1 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(3, 4, &n_u as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new((b * hs) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
    if prof.on {
        cmd.commit();
        cmd.wait_until_completed();
        prof.report();
    } else {
        cmd.commit();
        cmd.wait_until_completed();
    }

    // ── readback: hidden once, K/V rows + importance per layer.
    unsafe {
        std::ptr::copy_nonoverlapping(h_b.contents() as *const f32, h.as_mut_ptr(), b * hs);
    }
    for (prep, lio) in preps.iter().zip(io.iter_mut()) {
        unsafe {
            let kc = prep.k_mb.contents() as *const f32;
            let vc = prep.v_mb.contents() as *const f32;
            for hh in 0..nkv {
                for bi in 0..b {
                    let srck = kc.add((hh * prep.cap + prep.st0 + bi) * hd);
                    let srcv = vc.add((hh * prep.cap + prep.st0 + bi) * hd);
                    let dst = (bi * nkv + hh) * hd;
                    std::ptr::copy_nonoverlapping(srck, lio.out_k.as_mut_ptr().add(dst), hd);
                    std::ptr::copy_nonoverlapping(srcv, lio.out_v.as_mut_ptr().add(dst), hd);
                }
            }
            std::ptr::copy_nonoverlapping(
                prep.imp_mb.contents() as *const f32,
                lio.imp.as_mut_ptr(),
                lio.cpu_stored + b,
            );
        }
    }
    true
}

pub fn q8_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    row_scale: &[f32],
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 4 != 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let _bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    if abs + rows * cols > safe_len {
        return false;
    }
    let base = model_key(model);
    let rs_buf = {
        let mut cache = c.rs_bufs.lock().unwrap();
        cache
            .entry((base, idx))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device.new_buffer_with_data(
                    row_scale.as_ptr() as *const std::ffi::c_void,
                    (row_scale.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    };
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let use_mm = b >= 32 && cols % 32 == 0;
    let xs_buf = get_io(11_000_000_453 + pre.len(), pre.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(pre.as_ptr(), xs_buf.contents() as *mut f32, pre.len());
    }
    let y_buf = get_io(12_000_000_469 + b * rows, b * rows * 4);

    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    // Batches wide enough to fill a C-tile take the simdgroup GEMM;
    // narrow ones keep the row-streaming matvec-style kernel.
    enc.set_compute_pipeline_state(if use_mm { &c.q8mmm } else { &c.q8mm });
    enc.set_buffer(0, Some(&fbuf), abs as u64);
    enc.set_buffer(1, Some(&xs_buf), 0);
    enc.set_buffer(2, Some(&rs_buf), 0);
    enc.set_buffer(3, Some(&y_buf), 0);
    let rows_u = rows as u32;
    let b_u = b as u32;
    let k_arg = if use_mm {
        cols as u32
    } else {
        (cols / 4) as u32
    };
    enc.set_bytes(4, 4, &k_arg as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &b_u as *const u32 as *const std::ffi::c_void);
    if use_mm {
        enc.dispatch_thread_groups(
            MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
    } else {
        let sgs = 8u64;
        enc.dispatch_thread_groups(
            MTLSize::new((rows as u64).div_ceil(sgs), b as u64, 1),
            MTLSize::new(sgs * 32, 1, 1),
        );
    }
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);

    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), b * rows);
    }
    tracing::debug!("gpu matmat: {rows}x{cols} b={b}");
    true
}

/// q4t batched GEMM (imagegen DiT prefill shapes): one q4t_mul_mm
/// encoder reading the mmap-resident tiles straight from the file
/// buffer — no dequant scratch (the two-pass variant re-read an f32
/// W copy per 32-batch tile and was bandwidth-bound). Half
/// shared-memory tiles make this tolerance-class (like the LLM
/// prefill graph); the probe arbitrates vs the CPU AMX arm per
/// process.
pub fn q4tp_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 32 != 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let Some(need) =
        cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[rows, cols])
    else {
        return false;
    };
    if abs + need > safe_len {
        return false;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(21_000_000_659 + pre.len(), pre.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(pre.as_ptr(), xs_buf.contents() as *mut f32, pre.len());
    }
    let y_buf = get_io(22_000_000_663 + b * rows, b * rows * 4);

    let cmd = c.queue.new_command_buffer();
    {
        // C[b, rows] = X · dequant(W)ᵀ, tiles decoded in the K loop.
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.q4tpmm);
        enc.set_buffer(0, Some(&fbuf), abs as u64);
        enc.set_buffer(1, Some(&xs_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        let (cols_u, rows_u, nb_u) = (cols as u32, rows as u32, b as u32);
        enc.set_bytes(3, 4, &cols_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
    }
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), b * rows);
    }
    tracing::debug!("gpu q4tp matmat: {rows}x{cols} b={b}");
    true
}

pub fn q4t_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 32 != 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let tiles = rows * (cols / 32);
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    if abs + tiles * 18 > safe_len {
        return false;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(11_000_000_453 + pre.len(), pre.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(pre.as_ptr(), xs_buf.contents() as *mut f32, pre.len());
    }
    let y_buf = get_io(12_000_000_469 + b * rows, b * rows * 4);

    let cmd = c.queue.new_command_buffer();
    {
        // C[b, rows] = X · dequant(W)ᵀ, tiles decoded in the K loop.
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.q4tmm);
        enc.set_buffer(0, Some(&fbuf), abs as u64);
        enc.set_buffer(1, Some(&xs_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        let (cols_u, rows_u, nb_u) = (cols as u32, rows as u32, b as u32);
        enc.set_bytes(3, 4, &cols_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
    }
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), b * rows);
    }
    tracing::debug!("gpu q4t matmat: {rows}x{cols} b={b}");
    true
}

/// Fused DiT SwiGLU FFN, all on-device: g = X·W1ᵀ, u = X·W3ᵀ,
/// g = silu(g)·u (in place — thread-local read→write), y = g·W2ᵀ.
/// Four encoders in one command buffer; encoder order is the
/// dependency chain. The unfused path shipped the [b, inter]
/// intermediates across the CPU boundary twice per layer (~78 MB at
/// 512px) and ran the silu·u loop CPU-side between submits.
#[allow(clippy::too_many_arguments)]
/// q4tp twin of `q4t_ffn` — the fused DiT SwiGLU chain. Without it a q4tp
/// image model falls back to the unfused path, which ships the [b, inter]
/// intermediates across the CPU boundary twice per layer: measured 2x slower
/// end to end on Lumina at 256px (28 s against 14 s).
pub fn q4tp_ffn(
    model: &Arc<CmfModel>,
    w1: usize,
    w3: usize,
    w2: usize,
    xs: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if hidden % 32 != 0 || inter % 32 != 0 {
        return false;
    }
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let abs_ok = |idx: usize, rows: usize, cols: usize| -> Option<usize> {
        let abs = model.entry_abs_offset(&model.tensors[idx])?;
        (abs + rows * (cols / 32) * 18 <= safe_len).then_some(abs)
    };
    let (Some(a1), Some(a3), Some(a2)) = (
        abs_ok(w1, inter, hidden),
        abs_ok(w3, inter, hidden),
        abs_ok(w2, hidden, inter),
    ) else {
        return false;
    };
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(11_000_000_453 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let g_buf = get_io(14_000_000_071 + b * inter, b * inter * 4);
    let u_buf = get_io(15_000_000_083 + b * inter, b * inter * 4);
    let y_buf = get_io(12_000_000_469 + b * hidden, b * hidden * 4);

    let cmd = c.queue.new_command_buffer();
    let mm = |abs: usize, xb: &Buffer, yb: &Buffer, rows: usize, cols: usize| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.q4tpmm);
        enc.set_buffer(0, Some(&fbuf), abs as u64);
        enc.set_buffer(1, Some(xb), 0);
        enc.set_buffer(2, Some(yb), 0);
        let (cu, ru, nbu) = (cols as u32, rows as u32, b as u32);
        enc.set_bytes(3, 4, &cu as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &ru as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &nbu as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
    };
    mm(a1, &xs_buf, &g_buf, inter, hidden);
    mm(a3, &xs_buf, &u_buf, inter, hidden);
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.silu);
        enc.set_buffer(0, Some(&g_buf), 0);
        enc.set_buffer(1, Some(&u_buf), 0);
        enc.set_buffer(2, Some(&u_buf), 0); // col slot: unused (has_col=0)
        enc.set_buffer(3, Some(&g_buf), 0);
        let n_u = (b * inter) as u32;
        let has = 0u32;
        enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &has as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((b * inter) as u64).div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    mm(a2, &g_buf, &y_buf, hidden, inter);
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), b * hidden);
    }
    tracing::debug!("gpu q4tp ffn: {hidden}x{inter} b={b}");
    true
}

pub fn q4t_ffn(
    model: &Arc<CmfModel>,
    w1: usize,
    w3: usize,
    w2: usize,
    xs: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if hidden % 32 != 0 || inter % 32 != 0 {
        return false;
    }
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let abs_ok = |idx: usize, rows: usize, cols: usize| -> Option<usize> {
        let abs = model.entry_abs_offset(&model.tensors[idx])?;
        (abs + rows * (cols / 32) * 18 <= safe_len).then_some(abs)
    };
    let (Some(a1), Some(a3), Some(a2)) = (
        abs_ok(w1, inter, hidden),
        abs_ok(w3, inter, hidden),
        abs_ok(w2, hidden, inter),
    ) else {
        return false;
    };
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(11_000_000_453 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let g_buf = get_io(14_000_000_071 + b * inter, b * inter * 4);
    let u_buf = get_io(15_000_000_083 + b * inter, b * inter * 4);
    let y_buf = get_io(12_000_000_469 + b * hidden, b * hidden * 4);

    let cmd = c.queue.new_command_buffer();
    let mm = |abs: usize, xb: &Buffer, yb: &Buffer, rows: usize, cols: usize| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.q4tmm);
        enc.set_buffer(0, Some(&fbuf), abs as u64);
        enc.set_buffer(1, Some(xb), 0);
        enc.set_buffer(2, Some(yb), 0);
        let (cu, ru, nbu) = (cols as u32, rows as u32, b as u32);
        enc.set_bytes(3, 4, &cu as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &ru as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &nbu as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
    };
    mm(a1, &xs_buf, &g_buf, inter, hidden);
    mm(a3, &xs_buf, &u_buf, inter, hidden);
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.silu);
        enc.set_buffer(0, Some(&g_buf), 0);
        enc.set_buffer(1, Some(&u_buf), 0);
        enc.set_buffer(2, Some(&u_buf), 0); // col slot: unused (has_col=0)
        enc.set_buffer(3, Some(&g_buf), 0);
        let n_u = (b * inter) as u32;
        let has = 0u32;
        enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &has as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((b * inter) as u64).div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    mm(a2, &g_buf, &y_buf, hidden, inter);
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), b * hidden);
    }
    tracing::debug!("gpu q4t ffn: {hidden}x{inter} b={b}");
    true
}

/// Shared-mode io buffer from the per-context cache.
fn io_shared(c: &Ctx, key: usize, nbytes: usize) -> Buffer {
    let mut cache = c.io_bufs.lock().unwrap();
    cache
        .entry(key)
        .or_insert_with(|| {
            crate::gpu::probe_note_cold();
            c._device
                .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
        })
        .clone()
}

/// Weight/bias buffer keyed by heap address — stable for the owner's
/// lifetime, so each layer uploads its constants once per process.
fn cached_weight_buf(c: &Ctx, base: usize, data: &[f32]) -> Buffer {
    let key = base
        .wrapping_add(data.as_ptr() as usize)
        .wrapping_add(data.len());
    let mut cache = c.io_bufs.lock().unwrap();
    let mut fresh = false;
    let buf = cache
        .entry(key)
        .or_insert_with(|| {
            fresh = true;
            c._device
                .new_buffer((data.len() * 4) as u64, MTLResourceOptions::StorageModeShared)
        })
        .clone();
    if fresh {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut f32, data.len());
        }
    }
    buf
}

/// conv_mul_mm + panel_to_nchw as two encoders on an open command
/// buffer: img [ic,h,w] → out [oc,h,w] (+bias).
#[allow(clippy::too_many_arguments)]
fn encode_conv(
    c: &Ctx,
    cmd: &metal::CommandBufferRef,
    w_buf: &Buffer,
    b_buf: &Buffer,
    img: &Buffer,
    panel: &Buffer,
    out: &Buffer,
    ick2: usize,
    oc: usize,
    h: usize,
    w_img: usize,
    k: usize,
) {
    let hw = h * w_img;
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.convmm);
        enc.set_buffer(0, Some(w_buf), 0);
        enc.set_buffer(1, Some(img), 0);
        enc.set_buffer(2, Some(panel), 0);
        let words = [
            ick2 as u32,
            oc as u32,
            hw as u32,
            h as u32,
            w_img as u32,
            k as u32,
        ];
        for (i, wv) in words.iter().enumerate() {
            enc.set_bytes(3 + i as u64, 4, wv as *const u32 as *const std::ffi::c_void);
        }
        enc.dispatch_thread_groups(
            MTLSize::new((hw as u64).div_ceil(32), (oc as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
    }
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.p2nchw);
        enc.set_buffer(0, Some(panel), 0);
        enc.set_buffer(1, Some(out), 0);
        enc.set_buffer(2, Some(b_buf), 0);
        let words = [hw as u32, oc as u32];
        for (i, wv) in words.iter().enumerate() {
            enc.set_bytes(3 + i as u64, 4, wv as *const u32 as *const std::ffi::c_void);
        }
        enc.dispatch_threads(MTLSize::new((hw * oc) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
}

/// GroupNorm (+fused SiLU) as two encoders: reduce → apply.
#[allow(clippy::too_many_arguments)]
fn encode_groupnorm(
    c: &Ctx,
    cmd: &metal::CommandBufferRef,
    x: &Buffer,
    y: &Buffer,
    st: &Buffer,
    wa: &Buffer,
    ba: &Buffer,
    groups: usize,
    ch: usize,
    hw: usize,
    do_silu: bool,
) {
    let per_g = ch / groups;
    let eps = 1e-6f32;
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.gnred);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(st), 0);
        let (pg, hw_u) = (per_g as u32, hw as u32);
        enc.set_bytes(2, 4, &pg as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(3, 4, &hw_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.gnapp);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_buffer(2, Some(st), 0);
        enc.set_buffer(3, Some(wa), 0);
        enc.set_buffer(4, Some(ba), 0);
        let words = [
            per_g as u32,
            hw as u32,
            (ch * hw) as u32,
            do_silu as u32,
        ];
        for (i, wv) in words.iter().enumerate() {
            enc.set_bytes(5 + i as u64, 4, wv as *const u32 as *const std::ffi::c_void);
        }
        enc.dispatch_threads(MTLSize::new((ch * hw) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
}

/// VAE conv2d on the device (implicit GEMM — no im2col matrix). The
/// weight buffer is cached by (pointer, len) so each conv uploads its
/// weights once per process; the image and result cross per call.
#[allow(clippy::too_many_arguments)]
pub fn vae_conv2d(
    w: &[f32],
    bias: &[f32],
    x: &[f32],
    ic: usize,
    oc: usize,
    h: usize,
    w_img: usize,
    k: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let ick2 = ic * k * k;
    let hw = h * w_img;
    if w.len() != oc * ick2 || x.len() != ic * hw || out.len() != oc * hw {
        return false;
    }
    let w_buf = cached_weight_buf(c, 30_000_000_101, w);
    let b_buf = cached_weight_buf(c, 31_000_000_103, bias);
    let x_buf = io_shared(c, 32_000_000_119 + x.len(), x.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, x.len());
    }
    let panel = io_shared(c, 33_000_000_127 + hw * oc, hw * oc * 4);
    let o_buf = io_shared(c, 34_000_000_131 + hw * oc, hw * oc * 4);

    let cmd = c.queue.new_command_buffer();
    encode_conv(
        c, cmd, &w_buf, &b_buf, &x_buf, &panel, &o_buf, ick2, oc, h, w_img, k,
    );
    submit_and_wait(c, cmd, &[&o_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(o_buf.contents() as *const f32, out.as_mut_ptr(), oc * hw);
    }
    tracing::debug!("gpu vae conv: {ic}x{oc} k={k} {h}x{w_img}");
    true
}

/// One whole VAE resnet block on the device: norm1+silu → conv1 →
/// norm2+silu → conv2 → (+1×1 shortcut) → residual add, a single
/// command buffer — the image crosses the CPU boundary once each way
/// instead of 2–3 times per conv.
pub fn vae_resnet(a: &crate::gpu::VaeResnetArgs, x: &[f32], out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let (ic, oc, h, w) = (a.ic, a.oc, a.h, a.w);
    let hw = h * w;
    if x.len() != ic * hw || out.len() != oc * hw || ic % a.groups != 0 || oc % a.groups != 0 {
        return false;
    }
    if a.shortcut.is_none() && ic != oc {
        return false;
    }
    let n1w = cached_weight_buf(c, 35_000_000_107, a.n1w);
    let n1b = cached_weight_buf(c, 35_000_000_107, a.n1b);
    let n2w = cached_weight_buf(c, 35_000_000_107, a.n2w);
    let n2b = cached_weight_buf(c, 35_000_000_107, a.n2b);
    let c1w = cached_weight_buf(c, 30_000_000_101, a.c1w);
    let c1b = cached_weight_buf(c, 31_000_000_103, a.c1b);
    let c2w = cached_weight_buf(c, 30_000_000_101, a.c2w);
    let c2b = cached_weight_buf(c, 31_000_000_103, a.c2b);

    let xb = io_shared(c, 32_000_000_119 + ic * hw, ic * hw * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), xb.contents() as *mut f32, x.len());
    }
    let st = io_shared(c, 36_000_000_137 + a.groups, a.groups * 2 * 4);
    let t1 = io_shared(c, 37_000_000_139 + ic * hw, ic * hw * 4);
    let panel = io_shared(c, 33_000_000_127 + hw * oc, hw * oc * 4);
    let h1 = io_shared(c, 38_000_000_149 + oc * hw, oc * hw * 4);
    let t2 = io_shared(c, 39_000_000_157 + oc * hw, oc * hw * 4);
    let h2 = io_shared(c, 34_000_000_131 + hw * oc, hw * oc * 4);

    let cmd = c.queue.new_command_buffer();
    encode_groupnorm(c, cmd, &xb, &t1, &st, &n1w, &n1b, a.groups, ic, hw, true);
    encode_conv(
        c,
        cmd,
        &c1w,
        &c1b,
        &t1,
        &panel,
        &h1,
        ic * a.c1k * a.c1k,
        oc,
        h,
        w,
        a.c1k,
    );
    encode_groupnorm(c, cmd, &h1, &t2, &st, &n2w, &n2b, a.groups, oc, hw, true);
    encode_conv(
        c,
        cmd,
        &c2w,
        &c2b,
        &t2,
        &panel,
        &h2,
        oc * a.c2k * a.c2k,
        oc,
        h,
        w,
        a.c2k,
    );
    // Residual: h2 += shortcut(x) (1×1 conv through t2 as scratch) or
    // h2 += x directly.
    let skip: Buffer = match a.shortcut {
        Some((sw, sb, sk)) => {
            let sw_buf = cached_weight_buf(c, 30_000_000_101, sw);
            let sb_buf = cached_weight_buf(c, 31_000_000_103, sb);
            encode_conv(
                c,
                cmd,
                &sw_buf,
                &sb_buf,
                &xb,
                &panel,
                &t2,
                ic * sk * sk,
                oc,
                h,
                w,
                sk,
            );
            t2.clone()
        }
        None => xb.clone(),
    };
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.axpy);
        enc.set_buffer(0, Some(&skip), 0);
        enc.set_buffer(1, Some(&h2), 0);
        let one = 1.0f32;
        let n_u = (oc * hw) as u32;
        enc.set_bytes(2, 4, &one as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(3, 4, &n_u as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(MTLSize::new((oc * hw) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
    submit_and_wait(c, cmd, &[&h2]);
    unsafe {
        std::ptr::copy_nonoverlapping(h2.contents() as *const f32, out.as_mut_ptr(), oc * hw);
    }
    tracing::debug!("gpu vae resnet: {ic}->{oc} {h}x{w}");
    true
}

/// Nearest-2× upsample fused with the following conv — only the small
/// pre-upsample image is uploaded; the ×4 tensor lives on the device.
#[allow(clippy::too_many_arguments)]
pub fn vae_upsample_conv(
    w: &[f32],
    bias: &[f32],
    x: &[f32],
    ic: usize,
    oc: usize,
    h: usize,
    w_img: usize,
    k: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let ick2 = ic * k * k;
    let (h2, w2) = (2 * h, 2 * w_img);
    let hw2 = h2 * w2;
    if w.len() != oc * ick2 || x.len() != ic * h * w_img || out.len() != oc * hw2 {
        return false;
    }
    let w_buf = cached_weight_buf(c, 30_000_000_101, w);
    let b_buf = cached_weight_buf(c, 31_000_000_103, bias);
    let x_buf = io_shared(c, 32_000_000_119 + x.len(), x.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, x.len());
    }
    let up = io_shared(c, 40_000_000_163 + ic * hw2, ic * hw2 * 4);
    let panel = io_shared(c, 33_000_000_127 + hw2 * oc, hw2 * oc * 4);
    let o_buf = io_shared(c, 34_000_000_131 + hw2 * oc, hw2 * oc * 4);

    let cmd = c.queue.new_command_buffer();
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.ups2x);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&up), 0);
        let words = [(h * w_img) as u32, w_img as u32, (ic * hw2) as u32];
        for (i, wv) in words.iter().enumerate() {
            enc.set_bytes(2 + i as u64, 4, wv as *const u32 as *const std::ffi::c_void);
        }
        enc.dispatch_threads(MTLSize::new((ic * hw2) as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
    encode_conv(
        c, cmd, &w_buf, &b_buf, &up, &panel, &o_buf, ick2, oc, h2, w2, k,
    );
    submit_and_wait(c, cmd, &[&o_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(o_buf.contents() as *const f32, out.as_mut_ptr(), oc * hw2);
    }
    tracing::debug!("gpu vae upsample+conv: {ic}x{oc} {h}x{w_img} -> {h2}x{w2}");
    true
}

/// dit_flash_attend gate — EXPERIMENTAL, opt-in via `CMF_DIT_FLASH=1`.
/// V2 (device-direct simdgroup loads, per-simdgroup-only shmem, zero
/// threadgroup barriers in the KV loop) is 1.5× faster than V1 and
/// essentially exact (3.4e-8 vs the f64 reference — f32 MACs end to
/// end), but still trails the GEMM chain on M4 (15.5 vs 12 ms at
/// n=1064, 270 vs 124 ms at n=4136): one 8×8 MAC per two device
/// loads cannot match mul_mm's staged-tile arithmetic intensity, and
/// a pre-transposed K measured WORSE (n32-strided block rows lose
/// locality). Beating the chain needs the full flash_attn_ext-class
/// design — 64–128-row Q tiles, half operands, pipelined staging.
/// Until then the default stays on the 3-encoder GEMM chain.
fn flash_ok(hd: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_DIT_FLASH").is_ok_and(|v| v == "1"))
        && hd % 8 == 0
        && hd <= 128
}

/// Encode one all-heads flash-attend dispatch (out = [n][nh·hd]).
/// Inputs are head-major with row stride `n32` (padded, zeroed tails).
#[allow(clippy::too_many_arguments)]
fn encode_flash_attend(
    c: &Ctx,
    cmd: &metal::CommandBufferRef,
    qb: &Buffer,
    kb: &Buffer,
    vb: &Buffer,
    ob: &Buffer,
    nh: usize,
    nkv: usize,
    n: usize,
    n32: usize,
    hd: usize,
    scale: f32,
) {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.flashatt);
    enc.set_buffer(0, Some(qb), 0);
    enc.set_buffer(1, Some(kb), 0);
    enc.set_buffer(2, Some(vb), 0);
    enc.set_buffer(3, Some(ob), 0);
    let hpk = (nh / nkv.max(1)) as u32;
    let words = [n as u32, hd as u32, nh as u32, hpk];
    for (i, w) in words.iter().enumerate() {
        enc.set_bytes(4 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
    }
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    let n32_u = n32 as u32;
    enc.set_bytes(9, 4, &n32_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        MTLSize::new((n as u64).div_ceil(32), nh as u64, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
}

/// DiT full bidirectional attention, all heads on the device. Flash
/// path (default): one dit_flash_attend dispatch, online softmax, no
/// n×n scratch. Fallback (`CMF_DIT_FLASH=0` or an odd head shape):
/// per head scores = (Q·scale)·Kᵀ (f32nt), full-row softmax, P·V
/// (f32nn) into an [nh][n][hd] panel, then panel_unstack → [n][nh·hd],
/// the n×n scores scratch shared across heads via encoder order.
/// Inputs are head-major packs; GQA picks kv = h/(nh/nkv).
#[allow(clippy::too_many_arguments)]
pub fn dit_attention(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    nh: usize,
    nkv: usize,
    n: usize,
    hd: usize,
    scale: f32,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let ab = get_io(21_000_000_179 + nh * n * hd, nh * n * hd * 4);

    if flash_ok(hd) {
        // Padded uploads: row stride n32, zeroed tails (kernel contract).
        let n32 = n.div_ceil(32) * 32;
        let pad_up = |base: usize, src: &[f32], heads: usize| -> Buffer {
            let buf = get_io(base + heads * n32 * hd, heads * n32 * hd * 4);
            let dst = buf.contents() as *mut f32;
            for hh in 0..heads {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr().add(hh * n * hd),
                        dst.add(hh * n32 * hd),
                        n * hd,
                    );
                    std::ptr::write_bytes(
                        dst.add(hh * n32 * hd + n * hd),
                        0,
                        (n32 - n) * hd,
                    );
                }
            }
            buf
        };
        let qb = pad_up(16_000_000_123, qh, nh);
        let kb = pad_up(17_000_000_137, kh, nkv);
        let vb = pad_up(18_000_000_149, vh, nkv);
        let cmd = c.queue.new_command_buffer();
        encode_flash_attend(c, cmd, &qb, &kb, &vb, &ab, nh, nkv, n, n32, hd, scale);
        submit_and_wait(c, cmd, &[&ab]);
        unsafe {
            std::ptr::copy_nonoverlapping(
                ab.contents() as *const f32,
                out.as_mut_ptr(),
                n * nh * hd,
            );
        }
        tracing::debug!("gpu dit flash attention: nh={nh} n={n} hd={hd}");
        return true;
    }
    let qb = get_io(16_000_000_123 + qh.len(), qh.len() * 4);
    let kb = get_io(17_000_000_137 + kh.len(), kh.len() * 4);
    let vb = get_io(18_000_000_149 + vh.len(), vh.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(qh.as_ptr(), qb.contents() as *mut f32, qh.len());
        std::ptr::copy_nonoverlapping(kh.as_ptr(), kb.contents() as *mut f32, kh.len());
        std::ptr::copy_nonoverlapping(vh.as_ptr(), vb.contents() as *mut f32, vh.len());
    }

    let sc = get_io(19_000_000_151 + n * n, n * n * 4);
    let pb = get_io(20_000_000_167 + nh * n * hd, nh * n * hd * 4);
    let hpk = nh / nkv.max(1);

    let cmd = c.queue.new_command_buffer();
    for h in 0..nh {
        let kv = h / hpk;
        {
            let enc = cmd.new_compute_command_encoder();
            let pso = mm_pipeline(c, 0, hd, 2);
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&kb), (kv * n * hd * 4) as u64);
            enc.set_buffer(1, Some(&qb), (h * n * hd * 4) as u64);
            enc.set_buffer(2, Some(&sc), 0);
            let (cols_u, rows_u, nb_u) = (hd as u32, n as u32, n as u32);
            enc.set_bytes(3, 4, &cols_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &scale as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((n as u64).div_ceil(32), (n as u64).div_ceil(64), 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&c.smaxrows);
            enc.set_buffer(0, Some(&sc), 0);
            let n_u = n as u32;
            enc.set_bytes(1, 4, &n_u as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            let pso = mm_pipeline(c, hd, 0, 3);
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&vb), (kv * n * hd * 4) as u64);
            enc.set_buffer(1, Some(&sc), 0);
            enc.set_buffer(2, Some(&pb), (h * n * hd * 4) as u64);
            let (k_u, rows_u, nb_u) = (n as u32, hd as u32, n as u32);
            enc.set_bytes(3, 4, &k_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((n as u64).div_ceil(32), (hd as u64).div_ceil(64), 1),
                MTLSize::new(128, 1, 1),
            );
            enc.end_encoding();
        }
    }
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.unstack);
        enc.set_buffer(0, Some(&pb), 0);
        enc.set_buffer(1, Some(&ab), 0);
        let words = [nh as u32, n as u32, hd as u32];
        for (i, w) in words.iter().enumerate() {
            enc.set_bytes(2 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
        }
        enc.dispatch_threads(
            MTLSize::new((nh * n * hd) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    submit_and_wait(c, cmd, &[&ab]);
    unsafe {
        std::ptr::copy_nonoverlapping(ab.contents() as *const f32, out.as_mut_ptr(), n * nh * hd);
    }
    tracing::debug!("gpu dit attention: nh={nh} n={n} hd={hd}");
    true
}

/// One whole modulated DiT block in a single command buffer: x stays
/// device-resident through norm1·(1+s) → qkv GEMMs → qk-norm+RoPE+
/// head pack → per-head attention → unstack → O GEMM → gated
/// residual → ffn-norm·(1+s) → W1/W3 GEMMs → silu·u → W2 GEMM →
/// gated residual. Encoders separate dependent stages (the ordering
/// contract used everywhere in this file); independent dispatches
/// share one. Only x crosses the CPU boundary — the per-op path
/// shipped ~10 roundtrips per block.
pub fn dit_block(model: &Arc<CmfModel>, a: &crate::gpu::DitBlockArgs, x: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let (n, h, inter) = (a.n, a.hidden, a.inter);
    let (nh, nkv, hd) = (a.nh, a.nkv, a.hd);
    if h % 32 != 0 || inter % 32 != 0 || (nkv * hd) % 32 != 0 || hd % 2 != 0 {
        return false;
    }
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let abs_ok = |idx: usize, rows: usize, cols: usize| -> Option<usize> {
        let abs = model.entry_abs_offset(&model.tensors[idx])?;
        (abs + rows * (cols / 32) * 18 <= safe_len).then_some(abs)
    };
    let (Some(aq), Some(ak), Some(av), Some(ao)) = (
        abs_ok(a.wq, nh * hd, h),
        abs_ok(a.wk, nkv * hd, h),
        abs_ok(a.wv, nkv * hd, h),
        abs_ok(a.wo, h, nh * hd),
    ) else {
        return false;
    };
    let (Some(a1), Some(a3), Some(a2)) = (
        abs_ok(a.w1, inter, h),
        abs_ok(a.w3, inter, h),
        abs_ok(a.w2, h, inter),
    ) else {
        return false;
    };
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    // Params pack: 8 [h]-vectors + qk-norm weights + rope table, one
    // upload. Offsets in floats, all 4-byte aligned.
    let pairs = hd / 2;
    debug_assert_eq!(a.rope_cos.len(), n * pairs);
    let psz = 8 * h + 2 * hd + 2 * n * pairs;
    let p_buf = get_io(22_000_000_003 + psz, psz * 4);
    {
        let dst = p_buf.contents() as *mut f32;
        let mut off = 0usize;
        for v in [
            a.norm1, a.norm2, a.ffn_norm1, a.ffn_norm2, a.s_msa, a.gate_msa, a.s_mlp, a.gate_mlp,
            a.norm_q, a.norm_k, a.rope_cos, a.rope_sin,
        ] {
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), dst.add(off), v.len()) };
            off += v.len();
        }
        debug_assert_eq!(off, psz);
    }
    let fo = |floats: usize| (floats * 4) as u64; // float offset → bytes
    let (o_norm1, o_norm2, o_fn1, o_fn2) = (fo(0), fo(h), fo(2 * h), fo(3 * h));
    let (o_smsa, o_gmsa, o_smlp, o_gmlp) = (fo(4 * h), fo(5 * h), fo(6 * h), fo(7 * h));
    let (o_nq, o_nk) = (fo(8 * h), fo(8 * h + hd));
    let (o_cos, o_sin) = (fo(8 * h + 2 * hd), fo(8 * h + 2 * hd + n * pairs));

    let xb = get_io(23_000_000_017 + n * h, n * h * 4);
    unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), xb.contents() as *mut f32, n * h) };
    let xnb = get_io(24_000_000_029 + n * h, n * h * 4);
    let qtok = get_io(25_000_000_039 + n * nh * hd, n * nh * hd * 4);
    let ktok = get_io(26_000_000_047 + n * nkv * hd, n * nkv * hd * 4);
    let vtok = get_io(27_000_000_059 + n * nkv * hd, n * nkv * hd * 4);
    // Head-major packs use a 32-padded row stride (flash contract:
    // zeroed tails; the GEMM fallback just reads the first n rows).
    let n32 = n.div_ceil(32) * 32;
    let qhm = get_io(16_000_000_123 + n32 * nh * hd, n32 * nh * hd * 4);
    let khm = get_io(17_000_000_137 + n32 * nkv * hd, n32 * nkv * hd * 4);
    let vhm = get_io(18_000_000_149 + n32 * nkv * hd, n32 * nkv * hd * 4);
    let attnb = get_io(21_000_000_179 + n * nh * hd, n * nh * hd * 4);
    let projb = get_io(28_000_000_067 + n * h, n * h * 4);
    let gb = get_io(14_000_000_071 + n * inter, n * inter * 4);
    let ub = get_io(15_000_000_083 + n * inter, n * inter * 4);
    let db = get_io(29_000_000_073 + n * h, n * h * 4);

    let cmd = c.queue.new_command_buffer();
    let u32c = |v: usize| v as u32;
    // rms_mod_rows / rms_residual_rows share a binding shape.
    let rms = |pso: &ComputePipelineState,
               src: &Buffer,
               dst: &Buffer,
               w_off: u64,
               sg_off: u64,
               has: u32| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_buffer(2, Some(&p_buf), w_off);
        enc.set_buffer(3, Some(&p_buf), sg_off);
        let h_u = u32c(h);
        enc.set_bytes(4, 4, &h_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &a.eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &has as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    };
    let mm = |enc: &metal::ComputeCommandEncoderRef,
              abs: usize,
              xbuf: &Buffer,
              ybuf: &Buffer,
              rows: usize,
              cols: usize| {
        enc.set_compute_pipeline_state(&c.q4tmm);
        enc.set_buffer(0, Some(&fbuf), abs as u64);
        enc.set_buffer(1, Some(xbuf), 0);
        enc.set_buffer(2, Some(ybuf), 0);
        let (cu, ru, nbu) = (u32c(cols), u32c(rows), u32c(n));
        enc.set_bytes(3, 4, &cu as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &ru as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &nbu as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
    };

    // E1: attention pre-norm ·(1+s_msa)
    rms(&c.rmsmod, &xb, &xnb, o_norm1, o_smsa, 1);
    // E2: qkv GEMMs (independent — one encoder)
    {
        let enc = cmd.new_compute_command_encoder();
        mm(enc, aq, &xnb, &qtok, nh * hd, h);
        mm(enc, ak, &xnb, &ktok, nkv * hd, h);
        mm(enc, av, &xnb, &vtok, nkv * hd, h);
        enc.end_encoding();
    }
    // E3: qk-norm + RoPE + head-major packs (independent). When the
    // flash path is on, the padded tail rows are zeroed first (same
    // encoder — disjoint regions).
    {
        let enc = cmd.new_compute_command_encoder();
        if flash_ok(hd) && n32 > n {
            enc.set_compute_pipeline_state(&c.zero);
            for (buf, heads) in [(&qhm, nh), (&khm, nkv), (&vhm, nkv)] {
                for hh in 0..heads {
                    enc.set_buffer(0, Some(buf), ((hh * n32 + n) * hd * 4) as u64);
                    let cnt = ((n32 - n) * hd) as u32;
                    enc.set_bytes(1, 4, &cnt as *const u32 as *const std::ffi::c_void);
                    enc.dispatch_threads(
                        MTLSize::new(cnt as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
            }
        }
        for (src, dst, heads, w_off) in
            [(&qtok, &qhm, nh, o_nq), (&ktok, &khm, nkv, o_nk)]
        {
            enc.set_compute_pipeline_state(&c.ropepack);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_buffer(2, Some(&p_buf), w_off);
            enc.set_buffer(3, Some(&p_buf), o_cos);
            enc.set_buffer(4, Some(&p_buf), o_sin);
            let (n_u, h_u, hd_u) = (u32c(n), u32c(heads), u32c(hd));
            enc.set_bytes(5, 4, &n_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &h_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &hd_u as *const u32 as *const std::ffi::c_void);
            let qk_eps = 1e-5f32;
            enc.set_bytes(8, 4, &qk_eps as *const f32 as *const std::ffi::c_void);
            let nst_u = u32c(n32);
            enc.set_bytes(9, 4, &nst_u as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((n * heads) as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
        }
        enc.set_compute_pipeline_state(&c.packh);
        enc.set_buffer(0, Some(&vtok), 0);
        enc.set_buffer(1, Some(&vhm), 0);
        let words = [u32c(n), u32c(nkv), u32c(hd), u32c(n32)];
        for (i, w) in words.iter().enumerate() {
            enc.set_bytes(2 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
        }
        enc.dispatch_threads(
            MTLSize::new((n * nkv * hd) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    // E4: attention — flash by default (one dispatch, all heads,
    // straight into the [n][nh·hd] layout); the per-head GEMM chain
    // with the shared n×n scratch stays as the fallback.
    let scale = 1.0f32 / (hd as f32).sqrt();
    if flash_ok(hd) {
        encode_flash_attend(
            c, cmd, &qhm, &khm, &vhm, &attnb, nh, nkv, n, n32, hd, scale,
        );
    } else {
        let sc = get_io(19_000_000_151 + n * n, n * n * 4);
        let pb = get_io(20_000_000_167 + nh * n * hd, nh * n * hd * 4);
        let hpk = nh / nkv.max(1);
        for hh in 0..nh {
            let kv = hh / hpk;
            {
                let enc = cmd.new_compute_command_encoder();
                let pso = mm_pipeline(c, 0, hd, 2);
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&khm), (kv * n32 * hd * 4) as u64);
                enc.set_buffer(1, Some(&qhm), (hh * n32 * hd * 4) as u64);
                enc.set_buffer(2, Some(&sc), 0);
                let (cols_u, rows_u, nb_u) = (u32c(hd), u32c(n), u32c(n));
                enc.set_bytes(3, 4, &cols_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &scale as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((n as u64).div_ceil(32), (n as u64).div_ceil(64), 1),
                    MTLSize::new(128, 1, 1),
                );
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&c.smaxrows);
                enc.set_buffer(0, Some(&sc), 0);
                let n_u = u32c(n);
                enc.set_bytes(1, 4, &n_u as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(n as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_command_encoder();
                let pso = mm_pipeline(c, hd, 0, 3);
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&vhm), (kv * n32 * hd * 4) as u64);
                enc.set_buffer(1, Some(&sc), 0);
                enc.set_buffer(2, Some(&pb), (hh * n * hd * 4) as u64);
                let (k_u, rows_u, nb_u) = (u32c(n), u32c(hd), u32c(n));
                enc.set_bytes(3, 4, &k_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((n as u64).div_ceil(32), (hd as u64).div_ceil(64), 1),
                    MTLSize::new(128, 1, 1),
                );
                enc.end_encoding();
            }
        }
        // panel → [n][nh·hd]
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.unstack);
        enc.set_buffer(0, Some(&pb), 0);
        enc.set_buffer(1, Some(&attnb), 0);
        let words = [u32c(nh), u32c(n), u32c(hd)];
        for (i, w) in words.iter().enumerate() {
            enc.set_bytes(2 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
        }
        enc.dispatch_threads(
            MTLSize::new((nh * n * hd) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    // E6: O projection
    {
        let enc = cmd.new_compute_command_encoder();
        mm(enc, ao, &attnb, &projb, h, nh * hd);
        enc.end_encoding();
    }
    // E7: x += gate_msa ⊙ rms(proj)·norm2
    rms(&c.rmsres, &projb, &xb, o_norm2, o_gmsa, 1);
    // E8: ffn pre-norm ·(1+s_mlp)
    rms(&c.rmsmod, &xb, &xnb, o_fn1, o_smlp, 1);
    // E9: W1/W3 GEMMs (independent)
    {
        let enc = cmd.new_compute_command_encoder();
        mm(enc, a1, &xnb, &gb, inter, h);
        mm(enc, a3, &xnb, &ub, inter, h);
        enc.end_encoding();
    }
    // E10: silu(g)·u in place
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.silu);
        enc.set_buffer(0, Some(&gb), 0);
        enc.set_buffer(1, Some(&ub), 0);
        enc.set_buffer(2, Some(&ub), 0); // col slot: unused (has_col=0)
        enc.set_buffer(3, Some(&gb), 0);
        let n_u = u32c(n * inter);
        let has = 0u32;
        enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &has as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((n * inter) as u64).div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    }
    // E11: W2 GEMM
    {
        let enc = cmd.new_compute_command_encoder();
        mm(enc, a2, &gb, &db, h, inter);
        enc.end_encoding();
    }
    // E12: x += gate_mlp ⊙ rms(d)·ffn_norm2
    rms(&c.rmsres, &db, &xb, o_fn2, o_gmlp, 1);

    submit_and_wait(c, cmd, &[&xb]);
    unsafe {
        std::ptr::copy_nonoverlapping(xb.contents() as *const f32, x.as_mut_ptr(), n * h);
    }
    tracing::debug!("gpu dit block: n={n} h={h} inter={inter}");
    true
}

/// q1t batched GEMM (prefill): register-blocked base GEMM (q1t_mul_mm) then the
/// sparse overlay (q1t_overlay_mm), both on-device in one command buffer. Raw
/// f32 x, scales in the tiles. Only the wide path (b ≥ 32, cols % 32 == 0);
/// narrower batches return false → CPU.
pub fn q1t_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if b < 32 || cols % 32 != 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let _bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    if abs + entry.nbytes as usize > safe_len {
        return false;
    }
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let xs_buf = get_io(11_000_000_453 + xs.len(), xs.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(xs.as_ptr(), xs_buf.contents() as *mut f32, xs.len());
    }
    let y_buf = get_io(12_000_000_469 + b * rows, b * rows * 4);
    let gpr = cols / GROUP_SIZE;
    let (cols_u, rows_u, b_u) = (cols as u32, rows as u32, b as u32);

    let cmd = c.queue.new_command_buffer();
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.q1t_mm);
        enc.set_buffer(0, Some(&fbuf), abs as u64);
        enc.set_buffer(1, Some(&xs_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        enc.set_bytes(3, 4, &cols_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &b_u as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
            MTLSize::new(128, 1, 1),
        );
        enc.end_encoding();
    }
    {
        // Separate encoder → serialized after the GEMM (reads its y).
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&c.q1t_ovmm);
        enc.set_buffer(0, Some(&fbuf), abs as u64);
        enc.set_buffer(1, Some(&xs_buf), 0);
        enc.set_buffer(2, Some(&y_buf), 0);
        let base_len = (rows * gpr * Q1T_TILE) as u32;
        enc.set_bytes(3, 4, &base_len as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &cols_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &b_u as *const u32 as *const std::ffi::c_void);
        let tpt = 64u64;
        enc.dispatch_thread_groups(
            MTLSize::new((rows as u64).div_ceil(tpt), 1, 1),
            MTLSize::new(tpt, 1, 1),
        );
        enc.end_encoding();
    }
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), b * rows);
    }
    true
}

/// Layer MoE-FFN in a single command buffer: for each selected expert
/// gate/up-matvec → silu·mul·prescale → down-matvec → axpy into y;
/// intermediate buffers are GPU-resident, one sync per layer. D5 design:
/// amortizing the dispatch cost over ~25 MB of work instead of a single matvec.
pub fn moe_block(model: &Arc<CmfModel>, jobs: &[MoeJob], out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() {
        return false;
    }
    let _bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let base = model_key(model);

    // Validate all tensors before encoding (fail → CPU without partial work).
    let mut abs3 = Vec::with_capacity(jobs.len());
    for j in jobs {
        let mut trio = [0usize; 3];
        for (slot, (idx, rows, cols, _)) in [(0, &j.gate), (1, &j.up), (2, &j.down)] {
            let entry = &model.tensors[*idx];
            let Some(abs) = model.entry_abs_offset(entry) else {
                return false;
            };
            let qlen = if j.q1 {
                if cols % GROUP_SIZE != 0 || (cols / GROUP_SIZE) % 2 != 0 {
                    return false;
                }
                rows * (cols / GROUP_SIZE) * Q1_TILE
            } else if j.q4t {
                if cols % GROUP_SIZE != 0 {
                    return false;
                }
                rows * (cols / GROUP_SIZE) * 18
            } else {
                if cols % 4 != 0 {
                    return false;
                }
                rows * cols
            };
            if abs + qlen > safe_len {
                return false;
            }
            trio[slot] = abs;
        }
        abs3.push(trio);
    }

    let inter = jobs[0].gate.1;
    let hidden = jobs[0].down.1;
    if out.len() != hidden {
        return false;
    }

    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    // Salted keys — sizes may coincide between assignments.
    let g_buf = get_io(1_000_000_007 + inter, inter * 4);
    let u_buf = get_io(2_000_000_011 + inter, inter * 4);
    let a_buf = get_io(3_000_000_019 + inter, inter * 4);
    let d_buf = get_io(4_000_000_021 + hidden, hidden * 4);
    let y_buf = get_io(5_000_000_033 + hidden, hidden * 4);

    let rs_or_col = |idx: usize, data: &[f32], salt: usize| -> Buffer {
        let mut cache = c.rs_bufs.lock().unwrap();
        cache
            .entry((base + salt, idx))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device.new_buffer_with_data(
                    data.as_ptr() as *const std::ffi::c_void,
                    (data.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    };

    let cmd = c.queue.new_command_buffer();
    // Stage boundaries are ENCODER boundaries: Metal's automatic hazard
    // tracking fences tracked buffers between encoders, which on Apple
    // GPUs is far cheaper than memory_barrier_with_resources inside one
    // encoder (measured: the barrier variant cost ~2 ms extra per FFN
    // chain — more than all three matvecs together).
    let disp_elem =
        |enc: &metal::ComputeCommandEncoderRef, pso: &ComputePipelineState, n: usize| {
            enc.set_compute_pipeline_state(pso);
            enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        };

    // y = 0
    let hid_u = hidden as u32;
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_buffer(0, Some(&y_buf), 0);
        enc.set_bytes(1, 4, &hid_u as *const u32 as *const std::ffi::c_void);
        disp_elem(enc, &c.zero, hidden);
        enc.end_encoding();
    }

    let matvec = |enc: &metal::ComputeCommandEncoderRef,
                  abs: usize,
                  rows: usize,
                  cols: usize,
                  rs: Option<&Buffer>,
                  q4t: bool,
                  xs: &Buffer,
                  y: &Buffer| {
        match rs {
            None if q4t => encode_q4t_matvec(c, enc, &fbuf, abs, xs, y, rows, cols / GROUP_SIZE),
            None => encode_q1_matvec(c, enc, &fbuf, abs, xs, y, rows, cols / GROUP_SIZE),
            Some(rs) => {
                enc.set_compute_pipeline_state(&c.q8);
                enc.set_buffer(0, Some(&fbuf), abs as u64);
                enc.set_buffer(1, Some(xs), 0);
                enc.set_buffer(2, Some(rs), 0);
                enc.set_buffer(3, Some(y), 0);
                let cols4 = (cols / 4) as u32;
                let rows_u = rows as u32;
                enc.set_bytes(4, 4, &cols4 as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
                let sgs = 8u64;
                enc.dispatch_thread_groups(
                    MTLSize::new((rows as u64).div_ceil(sgs), 1, 1),
                    MTLSize::new(sgs * 32, 1, 1),
                );
            }
        }
    };

    for (j, trio) in jobs.iter().zip(&abs3) {
        let (gi, grows, gcols, grs) = &j.gate;
        let (ui, urows, ucols, urs) = &j.up;
        let (di, drows, dcols, drs) = &j.down;
        // q1/q4t: scales live in the tiles — no rs buffers at all.
        let rs3 = if j.q1 || j.q4t {
            [None, None, None]
        } else {
            [
                Some(rs_or_col(*gi, grs, 0)),
                Some(rs_or_col(*ui, urs, 0)),
                Some(rs_or_col(*di, drs, 0)),
            ]
        };
        let has_col = !j.down_col.is_empty();
        let dcol_b = if has_col {
            rs_or_col(*di, j.down_col, 7_777_777)
        } else {
            g_buf.clone() // never read: silu has_col = 0
        };
        // gate/up xs — per call (small, via the size-keyed io cache).
        let xsg = get_io(6_000_000_087 + j.xs_gate.len(), j.xs_gate.len() * 4);
        let xsu = get_io(7_000_000_103 + j.xs_up.len(), j.xs_up.len() * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(
                j.xs_gate.as_ptr(),
                xsg.contents() as *mut f32,
                j.xs_gate.len(),
            );
            std::ptr::copy_nonoverlapping(
                j.xs_up.as_ptr(),
                xsu.contents() as *mut f32,
                j.xs_up.len(),
            );
        }

        {
            let enc = cmd.new_compute_command_encoder();
            matvec(enc, trio[0], *grows, *gcols, rs3[0].as_ref(), j.q4t, &xsg, &g_buf);
            matvec(enc, trio[1], *urows, *ucols, rs3[1].as_ref(), j.q4t, &xsu, &u_buf);
            enc.end_encoding();
        }
        {
            // act = silu(g)·u·col_down (col skipped when the job has none)
            let enc = cmd.new_compute_command_encoder();
            enc.set_buffer(0, Some(&g_buf), 0);
            enc.set_buffer(1, Some(&u_buf), 0);
            enc.set_buffer(2, Some(&dcol_b), 0);
            enc.set_buffer(3, Some(&a_buf), 0);
            let n_u = inter as u32;
            let hc_u = has_col as u32;
            enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hc_u as *const u32 as *const std::ffi::c_void);
            disp_elem(enc, &c.silu, inter);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            matvec(
                enc,
                trio[2],
                *drows,
                *dcols,
                rs3[2].as_ref(),
                j.q4t,
                &a_buf,
                &d_buf,
            );
            enc.end_encoding();
        }
        {
            // y += w · d
            let enc = cmd.new_compute_command_encoder();
            enc.set_buffer(0, Some(&d_buf), 0);
            enc.set_buffer(1, Some(&y_buf), 0);
            enc.set_bytes(2, 4, &j.w as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &hid_u as *const u32 as *const std::ffi::c_void);
            disp_elem(enc, &c.axpy, hidden);
            enc.end_encoding();
        }
    }
    submit_and_wait(c, cmd, &[&y_buf]);

    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), hidden);
    }
    true
}

/// Several independent q8-matvec in a single command buffer (one sync).
/// outs[i].len() == jobs[i].rows.
pub fn matvec_batch(model: &Arc<CmfModel>, jobs: &[BatchJob], outs: &mut [&mut [f32]]) -> bool {
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() || jobs.len() != outs.len() {
        return false;
    }
    let _bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, model) else {
        return false;
    };
    let base = model_key(model);

    let mut abss = Vec::with_capacity(jobs.len());
    for j in jobs {
        let entry = &model.tensors[j.idx];
        let Some(abs) = model.entry_abs_offset(entry) else {
            return false;
        };
        let qlen = if j.q1 {
            if j.cols % GROUP_SIZE != 0 || (j.cols / GROUP_SIZE) % 2 != 0 {
                return false;
            }
            j.rows * (j.cols / GROUP_SIZE) * Q1_TILE
        } else {
            if j.cols % 4 != 0 {
                return false;
            }
            j.rows * j.cols
        };
        if abs + qlen > safe_len {
            return false;
        }
        abss.push(abs);
    }

    // Buffers: y per job (by size, via the io cache with a position salt),
    // xs per job, rs cached per-tensor.
    let get_io = |key: usize, nbytes: usize| -> Buffer {
        let mut cache = c.io_bufs.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device
                    .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    };
    let rs_of = |idx: usize, data: &[f32]| -> Buffer {
        let mut cache = c.rs_bufs.lock().unwrap();
        cache
            .entry((base, idx))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c._device.new_buffer_with_data(
                    data.as_ptr() as *const std::ffi::c_void,
                    (data.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    };

    let mut y_bufs = Vec::with_capacity(jobs.len());
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    for (slot, (j, abs)) in jobs.iter().zip(&abss).enumerate() {
        let xs_b = get_io(8_000_000_209 + slot * 131 + j.xs.len(), j.xs.len() * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(j.xs.as_ptr(), xs_b.contents() as *mut f32, j.xs.len());
        }
        let y_b = get_io(9_000_000_341 + slot * 137 + j.rows, j.rows * 4);
        if j.q1 {
            encode_q1_matvec(
                c,
                enc,
                &fbuf,
                *abs,
                &xs_b,
                &y_b,
                j.rows,
                j.cols / GROUP_SIZE,
            );
        } else {
            let rs_b = rs_of(j.idx, j.row_scale);
            enc.set_compute_pipeline_state(&c.q8);
            enc.set_buffer(0, Some(&fbuf), *abs as u64);
            enc.set_buffer(1, Some(&xs_b), 0);
            enc.set_buffer(2, Some(&rs_b), 0);
            enc.set_buffer(3, Some(&y_b), 0);
            let cols4 = (j.cols / 4) as u32;
            let rows_u = j.rows as u32;
            enc.set_bytes(4, 4, &cols4 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
            let sgs = 8u64;
            enc.dispatch_thread_groups(
                MTLSize::new((j.rows as u64).div_ceil(sgs), 1, 1),
                MTLSize::new(sgs * 32, 1, 1),
            );
        }
        y_bufs.push(y_b);
    }
    enc.end_encoding();
    if y_bufs.len() <= 4 {
        let refs: Vec<&Buffer> = y_bufs.iter().collect();
        submit_and_wait(c, cmd, &refs);
    } else {
        cmd.commit();
        wait_fast(cmd);
    }

    for ((y_b, j), out) in y_bufs.iter().zip(jobs).zip(outs.iter_mut()) {
        unsafe {
            std::ptr::copy_nonoverlapping(y_b.contents() as *const f32, out.as_mut_ptr(), j.rows);
        }
    }
    true
}

/// One GDN layer's worth of tensors/vectors for the whole-block GPU
/// path. Matvec tensors are (directory idx, rows, cols) of q1 weights.
pub struct GdnGpuLayer<'a> {
    pub attn_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub qkv: (usize, usize, usize),
    pub z: (usize, usize, usize),
    pub a: (&'a [f32], usize, usize),
    pub b: (&'a [f32], usize, usize),
    pub out: (usize, usize, usize),
    pub gate: (usize, usize, usize),
    pub up: (usize, usize, usize),
    pub down: (usize, usize, usize),
    pub conv1d: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub gnorm: &'a [f32],
}

/// Shared dims of the block (identical across GDN layers of a model).
#[derive(Clone, Copy)]
pub struct GdnGpuCfg {
    pub nv: usize,
    pub nk: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
    pub hidden: usize,
    pub inter: usize,
    pub c_dim: usize,
    pub eps: f32,
    /// Gemma-style norms: x̂·(1+w) (qwen3_5 family) vs Qwen x̂·w.
    pub gemma: bool,
}

/// Model-wide dims every token-graph layer agrees on.
#[derive(Clone, Copy)]
pub struct GraphDims {
    pub hidden: usize,
    pub eps: f32,
    /// Gemma-style norms: x̂·(1+w) (qwen3_5 family) vs Qwen x̂·w.
    pub gemma: bool,
}

/// One full-attention layer's q1 graph inputs: (directory idx, rows,
/// cols) triples; the qk-norms / RoPE / KV / attend stay on the CPU
/// between the graph's QKV prefix and O+FFN suffix.
pub struct AttnGpuLayer<'a> {
    pub attn_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub wq: (usize, usize, usize),
    pub wk: (usize, usize, usize),
    pub wv: (usize, usize, usize),
    pub wo: (usize, usize, usize),
    pub gate: (usize, usize, usize),
    pub up: (usize, usize, usize),
    pub down: (usize, usize, usize),
}

fn io_buf(c: &Ctx, key: usize, nbytes: usize) -> Buffer {
    let mut cache = c.io_bufs.lock().unwrap();
    cache
        .entry(key)
        .or_insert_with(|| {
            crate::gpu::probe_note_cold();
            c._device
                .new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
        })
        .clone()
}

/// Small constant vectors cached by their (stable) data pointer.
fn const_buf(c: &Ctx, data: &[f32]) -> Buffer {
    let mut cache = c.rs_bufs.lock().unwrap();
    cache
        .entry((data.as_ptr() as usize, usize::MAX - 2))
        .or_insert_with(|| {
            crate::gpu::probe_note_cold();
            c._device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .clone()
}

fn enc_simple(
    c_cmd: &metal::CommandBufferRef,
    pso: &ComputePipelineState,
    bufs: &[(&Buffer, u64)],
    words: &[u32],
    floats: &[f32],
    grid: (u64, u64),
) {
    let enc = c_cmd.new_compute_command_encoder();
    disp(enc, pso, bufs, words, floats, grid);
    enc.end_encoding();
}

/// One dispatch into an ALREADY OPEN compute encoder. Dispatches inside
/// a single encoder are serial on Apple Silicon (the default
/// MTLDispatchTypeSerial: each waits on the previous and sees its
/// writes), so a chain of data-dependent steps belongs in ONE encoder.
/// A new encoder is a new GPU pass with its own kick and flush — at 11
/// passes per layer × 44 virtual layers, that overhead was a large
/// slice of Nanbeige's per-token wall.
fn disp(
    enc: &metal::ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    bufs: &[(&Buffer, u64)],
    words: &[u32],
    floats: &[f32],
    grid: (u64, u64),
) {
    enc.set_compute_pipeline_state(pso);
    for (i, (b, off)) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), *off);
    }
    let base = bufs.len() as u64;
    for (i, w) in words.iter().enumerate() {
        enc.set_bytes(
            base + i as u64,
            4,
            w as *const u32 as *const std::ffi::c_void,
        );
    }
    for (i, f) in floats.iter().enumerate() {
        enc.set_bytes(
            base + words.len() as u64 + i as u64,
            4,
            f as *const f32 as *const std::ffi::c_void,
        );
    }
    enc.dispatch_threads(MTLSize::new(grid.0, 1, 1), MTLSize::new(grid.1, 1, 1));
}

/// `disp` plus a threadgroup-memory allocation at index 0 — for kernels
/// whose simdgroups combine partials through shared memory
/// (`gqa_attend`'s flash-decoding split). The length is encoder state,
/// so it is cleared again for the dispatches that follow.
#[allow(clippy::too_many_arguments)]
fn disp_tg(
    enc: &metal::ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    bufs: &[(&Buffer, u64)],
    words: &[u32],
    floats: &[f32],
    grid: (u64, u64),
    tg_bytes: u64,
) {
    enc.set_threadgroup_memory_length(0, tg_bytes);
    disp(enc, pso, bufs, words, floats, grid);
    enc.set_threadgroup_memory_length(0, 0);
}

/// Device mirror of one layer's K/V cache: `[nkv, cap, hd]` each, plus
/// the per-position Born-importance accumulator for this token. The
/// CPU cache stays the owner of record — `stored` tracks how many CPU
/// rows the mirror reflects, and any mismatch (eviction, rollback, a
/// non-graph path having appended) triggers a full re-upload.
pub struct KvMirror {
    k: Buffer,
    v: Buffer,
    imp: Buffer,
    cap: usize,
    stored: usize,
}

// Buffers are retained ObjC pointers, guarded by the registry Mutex.
unsafe impl Send for KvMirror {}

/// A token's worth of layers as few command buffers: hidden lives in a
/// device buffer across GDN runs AND full-attention layers; the only
/// syncs are where the CPU genuinely needs data (q/k/v before the KV
/// attend, recurrent states, the final hidden). Contract: validate
/// every layer (`gdn_ok`/`attn_ok`) BEFORE encoding — after the first
/// `sync` a refused encode would leave the token half-executed.
pub struct TokenGraph {
    c: &'static Ctx,
    model: Arc<CmfModel>,
    fbuf: Buffer,
    safe_len: usize,
    dims: GraphDims,
    cmd: Option<metal::CommandBuffer>,
    /// Committed-but-unawaited predecessor (see `commit`).
    in_flight: Option<metal::CommandBuffer>,
    h_b: Buffer,
    n_b: Buffer,
    d_b: Buffer,
    /// Recurrent-state buffers awaiting readback (buffer, f32 len).
    dirty: Vec<(Buffer, usize)>,
    /// Next state-buffer cache slot (reset when `dirty` drains).
    st_next: usize,
    /// q/k/v buffers of the last encoded attention prefix.
    qkv_bufs: Option<(Buffer, Buffer, Buffer)>,
    /// Logits buffer of an encoded final-norm+lm_head tail (rows).
    logits_b: Option<Buffer>,
}

impl TokenGraph {
    pub fn new(model: &Arc<CmfModel>, dims: GraphDims, h: &[f32]) -> Option<TokenGraph> {
        let c = ctx()?;
        if h.len() != dims.hidden {
            return None;
        }
        let (fbuf, safe_len) = file_buffer(c, model)?;
        let h_b = io_buf(c, 20_000_000_003 + dims.hidden, dims.hidden * 4);
        let n_b = io_buf(c, 21_000_000_011 + dims.hidden, dims.hidden * 4);
        let d_b = io_buf(c, 32_000_000_207 + dims.hidden, dims.hidden * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(h.as_ptr(), h_b.contents() as *mut f32, dims.hidden);
        }
        Some(TokenGraph {
            c,
            model: model.clone(),
            fbuf,
            safe_len,
            dims,
            cmd: None,
            in_flight: None,
            h_b,
            n_b,
            d_b,
            dirty: Vec::new(),
            st_next: 0,
            qkv_bufs: None,
            logits_b: None,
        })
    }

    /// Validate one q1 tensor and resolve its absolute payload offset.
    fn q1_abs(&self, t: (usize, usize, usize)) -> Option<usize> {
        let (idx, rows, cols) = t;
        if cols % GROUP_SIZE != 0 || (cols / GROUP_SIZE) % 2 != 0 {
            return None;
        }
        let entry = &self.model.tensors[idx];
        let abs = self.model.entry_abs_offset(entry)?;
        if abs + rows * (cols / GROUP_SIZE) * Q1_TILE > self.safe_len {
            return None;
        }
        Some(abs)
    }

    /// Validate one q1t tensor: base (9-byte tiles) then the per-row overlay
    /// must fit the safe mmap window. No gpr-parity constraint (the q1t kernel
    /// doesn't pair tiles).
    fn q1t_abs(&self, t: (usize, usize, usize)) -> Option<usize> {
        let (idx, _rows, cols) = t;
        if cols % GROUP_SIZE != 0 {
            return None;
        }
        let entry = &self.model.tensors[idx];
        let abs = self.model.entry_abs_offset(entry)?;
        // Whole variable-length payload (base + overlay) sits within nbytes.
        if abs + entry.nbytes as usize > self.safe_len {
            return None;
        }
        Some(abs)
    }

    /// Validate one q4_block tensor: `packed (rows·gpr·16) + scales
    /// (rows·gpr·2)` must fit the safe mmap window.
    fn q4b_abs(&self, t: (usize, usize, usize)) -> Option<usize> {
        let (idx, rows, cols) = t;
        if cols % GROUP_SIZE != 0 {
            return None;
        }
        let entry = &self.model.tensors[idx];
        let abs = self.model.entry_abs_offset(entry)?;
        let n_groups = rows * (cols / GROUP_SIZE);
        if abs + n_groups * 16 + n_groups * 2 > self.safe_len {
            return None;
        }
        Some(abs)
    }

    /// Validate one q4_tiled tensor: `rows·gpr·18` interleaved tile
    /// bytes must fit the safe mmap window.
    /// q4tp spans three planes, so the bound check must cover all of them —
    /// the kernel reads the code plane past the end of the nibbles.
    fn q4tp_abs(&self, t: (usize, usize, usize)) -> Option<usize> {
        let (idx, rows, cols) = t;
        if cols % GROUP_SIZE != 0 {
            return None;
        }
        let entry = &self.model.tensors[idx];
        let abs = self.model.entry_abs_offset(entry)?;
        let need = cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[
            rows, cols,
        ])?;
        if abs + need > self.safe_len {
            return None;
        }
        Some(abs)
    }

    fn q4t_abs(&self, t: (usize, usize, usize)) -> Option<usize> {
        let (idx, rows, cols) = t;
        if cols % GROUP_SIZE != 0 {
            return None;
        }
        let entry = &self.model.tensors[idx];
        let abs = self.model.entry_abs_offset(entry)?;
        let n_groups = rows * (cols / GROUP_SIZE);
        if abs + n_groups * (2 + GROUP_SIZE / 2) > self.safe_len {
            return None;
        }
        Some(abs)
    }

    /// Resolve a projection tensor accepting Q1 / Q1T / Q4-block/tiled.
    fn proj_abs(&self, t: (usize, usize, usize)) -> Option<(usize, ProjKind)> {
        match self.model.tensors[t.0].dtype {
            cortiq_core::TensorDtype::Q1 => self.q1_abs(t).map(|a| (a, ProjKind::Q1)),
            cortiq_core::TensorDtype::Q1T => self.q1t_abs(t).map(|a| (a, ProjKind::Q1t)),
            cortiq_core::TensorDtype::Q4Block => self.q4b_abs(t).map(|a| (a, ProjKind::Q4b)),
            cortiq_core::TensorDtype::Q4Tiled => self.q4t_abs(t).map(|a| (a, ProjKind::Q4t)),
            cortiq_core::TensorDtype::Q4TiledP => self.q4tp_abs(t).map(|a| (a, ProjKind::Q4tp)),
            cortiq_core::TensorDtype::Q8Row | cortiq_core::TensorDtype::Q8_2f => {
                self.q8_abs(t).map(|(a, row_scale, col_field)| {
                    (
                        a,
                        ProjKind::Q8 {
                            row_scale,
                            col_field,
                        },
                    )
                })
            }
            _ => None,
        }
    }

    /// Validate q8_row/q8_2f and cache their f16-decoded fields as f32 Metal
    /// constants. The payload keeps f16 on disk; treating its bytes as f32
    /// here used to corrupt whole-token Q8 execution.
    fn q8_abs(&self, t: (usize, usize, usize)) -> Option<(usize, Buffer, Option<Buffer>)> {
        let (idx, rows, cols) = t;
        if cols % 4 != 0 {
            return None;
        }
        let entry = &self.model.tensors[idx];
        let has_col = entry.dtype == cortiq_core::TensorDtype::Q8_2f;
        let abs = self.model.entry_abs_offset(entry)?;
        let qlen = rows * cols;
        let need = qlen + rows * 2 + if has_col { cols * 2 } else { 0 };
        if abs + need > self.safe_len || (entry.nbytes as usize) < need {
            return None;
        }

        let base = model_key(&self.model);
        let c = self.c;
        let rs_buf = {
            let mut cache = c.rs_bufs.lock().unwrap();
            cache
                .entry((base, idx))
                .or_insert_with(|| {
                    crate::gpu::probe_note_cold();
                    let bytes = self.model.entry_bytes(entry);
                    let scales: Vec<f32> = (0..rows)
                        .map(|r| {
                            let o = qlen + r * 2;
                            f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]))
                        })
                        .collect();
                    c._device.new_buffer_with_data(
                        scales.as_ptr() as *const std::ffi::c_void,
                        (rows * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    )
                })
                .clone()
        };
        let col_buf = has_col.then(|| {
            let mut cache = c.cf_bufs.lock().unwrap();
            cache
                .entry((base, idx))
                .or_insert_with(|| {
                    crate::gpu::probe_note_cold();
                    let bytes = self.model.entry_bytes(entry);
                    let off = qlen + rows * 2;
                    let field: Vec<f32> = (0..cols)
                        .map(|i| {
                            let o = off + i * 2;
                            f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]))
                        })
                        .collect();
                    c._device.new_buffer_with_data(
                        field.as_ptr() as *const std::ffi::c_void,
                        (cols * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    )
                })
                .clone()
        });
        Some((abs, rs_buf, col_buf))
    }

    /// Pre-flight check for a GDN layer (call before any encode).
    pub fn gdn_ok(&self, l: &GdnGpuLayer, cfg: &GdnGpuCfg) -> bool {
        if cfg.kk < 2 || cfg.dv % 32 != 0 || cfg.dv > 1024 || cfg.hidden != self.dims.hidden {
            return false;
        }
        if l.a.0.len() != l.a.1 * l.a.2 || l.b.0.len() != l.b.1 * l.b.2 {
            return false;
        }
        [l.qkv, l.z, l.out, l.gate, l.up, l.down]
            .iter()
            .all(|t| self.proj_abs(*t).is_some())
    }

    /// Pre-flight check for a full-attention layer.
    pub fn attn_ok(&self, l: &AttnGpuLayer) -> bool {
        // The suffix reads the attention output back through ao (wo
        // cols) and writes hidden (wo rows) — both must match dims.
        if l.wo.1 != self.dims.hidden || l.down.1 != self.dims.hidden {
            return false;
        }
        [l.wq, l.wk, l.wv, l.wo, l.gate, l.up, l.down]
            .iter()
            .all(|t| self.proj_abs(*t).is_some())
    }

    fn ensure_cmd(&mut self) -> metal::CommandBuffer {
        if self.cmd.is_none() {
            self.cmd = Some(self.c.queue.new_command_buffer().to_owned());
        }
        self.cmd.as_ref().unwrap().clone()
    }

    /// Commit the current command buffer WITHOUT waiting: the GPU
    /// starts on it while the CPU keeps encoding the next one. Queue
    /// order makes the eventual `sync` wait (on the last buffer) cover
    /// every earlier commit.
    pub fn commit(&mut self) {
        if let Some(cmd) = self.cmd.take() {
            cmd.commit();
            self.in_flight = Some(cmd);
        }
    }

    /// Submit everything encoded so far and wait for completion.
    pub fn sync(&mut self) {
        if let Some(cmd) = self.cmd.take() {
            cmd.commit();
            self.in_flight = Some(cmd);
        }
        if let Some(cmd) = self.in_flight.take() {
            wait_fast(&cmd);
        }
    }

    /// Copy finished recurrent states back to their CPU owners (call
    /// after `sync`; order matches the `encode_gdn_run` calls).
    pub fn read_states(&mut self, outs: &mut [&mut [f32]]) {
        debug_assert_eq!(outs.len(), self.dirty.len());
        for ((buf, len), out) in self.dirty.drain(..).zip(outs.iter_mut()) {
            debug_assert_eq!(len, out.len());
            unsafe {
                std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), len);
            }
        }
        self.st_next = 0;
    }

    /// Final sync + hidden readback.
    pub fn finish(mut self, h: &mut [f32]) {
        self.sync();
        debug_assert!(self.dirty.is_empty(), "unread recurrent states at finish");
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.h_b.contents() as *const f32,
                h.as_mut_ptr(),
                self.dims.hidden,
            );
        }
    }

    /// Looped Transformer: apply RMS norm to the hidden state on-device
    /// between loop iterations — avoids a CPU round-trip at the boundary.
    /// h_b → rmsn → n_b → blit back → h_b.
    pub fn encode_loop_norm(&mut self, norm: &[f32]) {
        let cmd = self.ensure_cmd();
        enc_simple(
            &cmd,
            &self.c.rmsn,
            &[
                (&self.h_b, 0),
                (&const_buf(self.c, norm), 0),
                (&self.n_b, 0),
            ],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(&self.n_b, 0, &self.h_b, 0, (self.dims.hidden * 4) as u64);
        blit.end_encoding();
    }

    /// Pre-flight for the final-norm + lm_head tail.
    pub fn lm_head_ok(&self, lm: (usize, usize, usize)) -> bool {
        lm.2 == self.dims.hidden && self.proj_abs(lm).is_some()
    }

    /// Final rmsnorm + lm_head matvec at the end of the last layer —
    /// rides in the same command buffer, so the logits come out of the
    /// sync this graph already pays instead of a separate per-op
    /// submit+wait round trip. Read with `read_logits` after `sync`.
    pub fn encode_lm_head(&mut self, norm: &[f32], lm: (usize, usize, usize)) {
        let cmd = self.ensure_cmd();
        enc_simple(
            &cmd,
            &self.c.rmsn,
            &[
                (&self.h_b, 0),
                (&const_buf(self.c, norm), 0),
                (&self.n_b, 0),
            ],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        let (abs, q1t) = self.proj_abs(lm).unwrap();
        let lg_b = io_buf(self.c, 44_000_000_077 + lm.1, lm.1 * 4);
        let enc = cmd.new_compute_command_encoder();
        encode_proj(
            self.c,
            enc,
            &self.fbuf,
            abs,
            &q1t,
            &self.n_b,
            &lg_b,
            lm.1,
            lm.2 / GROUP_SIZE,
        );
        enc.end_encoding();
        self.logits_b = Some(lg_b);
    }

    /// Copy the finished logits (call after `sync`; out may be shorter
    /// than the head's rows — trailing rows are padding vocab).
    pub fn read_logits(&mut self, out: &mut [f32]) {
        let lg_b = self
            .logits_b
            .take()
            .expect("read_logits without encode_lm_head");
        unsafe {
            std::ptr::copy_nonoverlapping(
                lg_b.contents() as *const f32,
                out.as_mut_ptr(),
                out.len(),
            );
        }
    }

    /// norm(h) → n_b, then QKV projections n_b → q/k/v buffers. The
    /// caller must `sync` + `read_qkv` before using the values.
    pub fn encode_attn_prefix(&mut self, l: &AttnGpuLayer) {
        let cmd = self.ensure_cmd();
        let aq = self.proj_abs(l.wq).unwrap();
        let ak = self.proj_abs(l.wk).unwrap();
        let av = self.proj_abs(l.wv).unwrap();
        enc_simple(
            &cmd,
            &self.c.rmsn,
            &[
                (&self.h_b, 0),
                (&const_buf(self.c, l.attn_norm), 0),
                (&self.n_b, 0),
            ],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        let q_b = io_buf(self.c, 40_000_000_003 + l.wq.1, l.wq.1 * 4);
        let k_b = io_buf(self.c, 41_000_000_019 + l.wk.1, l.wk.1 * 4);
        let v_b = io_buf(self.c, 42_000_000_037 + l.wv.1, l.wv.1 * 4);
        let enc = cmd.new_compute_command_encoder();
        encode_proj(
            self.c,
            enc,
            &self.fbuf,
            aq.0,
            &aq.1,
            &self.n_b,
            &q_b,
            l.wq.1,
            l.wq.2 / GROUP_SIZE,
        );
        encode_proj(
            self.c,
            enc,
            &self.fbuf,
            ak.0,
            &ak.1,
            &self.n_b,
            &k_b,
            l.wk.1,
            l.wk.2 / GROUP_SIZE,
        );
        encode_proj(
            self.c,
            enc,
            &self.fbuf,
            av.0,
            &av.1,
            &self.n_b,
            &v_b,
            l.wv.1,
            l.wv.2 / GROUP_SIZE,
        );
        enc.end_encoding();
        self.qkv_bufs = Some((q_b, k_b, v_b));
    }

    /// Read the prefix's q/k/v after `sync` (UMA memcpy).
    pub fn read_qkv(&mut self, q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
        let (q_b, k_b, v_b) = self.qkv_bufs.take().expect("read_qkv without prefix");
        unsafe {
            std::ptr::copy_nonoverlapping(q_b.contents() as *const f32, q.as_mut_ptr(), q.len());
            std::ptr::copy_nonoverlapping(k_b.contents() as *const f32, k.as_mut_ptr(), k.len());
            std::ptr::copy_nonoverlapping(v_b.contents() as *const f32, v.as_mut_ptr(), v.len());
        }
    }

    /// Upload the CPU-attended output `ao`, then O-projection +
    /// residual + post-norm + FFN + residual on the device.
    pub fn encode_attn_suffix(&mut self, l: &AttnGpuLayer, ao: &[f32]) {
        debug_assert_eq!(ao.len(), l.wo.2);
        let cmd = self.ensure_cmd();
        let ao_b = io_buf(self.c, 43_000_000_057 + ao.len(), ao.len() * 4);
        // Safe to write: the previous command buffer completed at the
        // prefix sync, and the new one has not been committed yet.
        unsafe {
            std::ptr::copy_nonoverlapping(ao.as_ptr(), ao_b.contents() as *mut f32, ao.len());
        }
        let enc = cmd.new_compute_command_encoder();
        self.encode_o_ffn(enc, l, &ao_b);
        enc.end_encoding();
    }

    /// O-projection from a device-resident attention output + residual
    /// + post-norm + FFN + residual.
    fn encode_o_ffn(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        l: &AttnGpuLayer,
        ao_b: &Buffer,
    ) {
        let (abs, q1t) = self.proj_abs(l.wo).unwrap();
        encode_proj(
            self.c,
            enc,
            &self.fbuf,
            abs,
            &q1t,
            ao_b,
            &self.d_b,
            l.wo.1,
            l.wo.2 / GROUP_SIZE,
        );
        // Fused: h += d_b, n = rmsnorm(h, post_norm) — one dispatch
        // instead of separate enc_axpy + rmsnorm.
        self.encode_post_ffn(enc, l.post_norm, l.gate, l.up, l.down, Some(&self.d_b));
    }

    /// Dims contract of the device-attend kernels (host-side check).
    pub fn attn_device_ok(&self, l: &AttnGpuLayer, p: &AttnDeviceParams) -> bool {
        self.attn_ok(l)
            && p.hd % 4 == 0
            && p.hd <= 256
            && p.rd <= p.hd
            && p.rd >= 2
            && (p.rd / 2) % 32 == 0
            && p.nh % p.nkv == 0
            && l.wq.1 == p.nh * p.hd * (1 + p.output_gate as usize)
            && l.wk.1 == p.nkv * p.hd
            && l.wv.1 == p.nkv * p.hd
            && l.wo.2 == p.nh * p.hd
            && p.cpu_k.len() == p.nkv
            && p.cpu_v.len() == p.nkv
            && p.inv_freq.len() >= p.rd / 2
    }

    /// One attention layer entirely on the device: norm → QKV →
    /// qk-norm+RoPE → KV append → grouped attend (+Born importance) →
    /// output gate → O → residual → FFN → residual. No sync — the KV
    /// mirror is prepared host-side first (self-healing: any mismatch
    /// with the CPU cache re-uploads it). Returns false without
    /// encoding anything if the mirror could not be prepared.
    pub fn encode_attn_device(&mut self, l: &AttnGpuLayer, p: &AttnDeviceParams) -> bool {
        // ── KV mirror prep (CPU side; previous token already synced).
        let (k_mb, v_mb, imp_mb, cap, stored) = {
            let mut reg = self.c.kv_mirrors.lock().unwrap();
            let need = p.cpu_stored + 1;
            let entry = reg.entry((p.kv_id, p.layer)).or_insert_with(|| KvMirror {
                k: self
                    .c
                    ._device
                    .new_buffer(0, MTLResourceOptions::StorageModeShared),
                v: self
                    .c
                    ._device
                    .new_buffer(0, MTLResourceOptions::StorageModeShared),
                imp: self
                    .c
                    ._device
                    .new_buffer(0, MTLResourceOptions::StorageModeShared),
                cap: 0,
                stored: usize::MAX, // force first-touch upload
            });
            if entry.cap < need {
                let cap = need.next_power_of_two().max(1024);
                let bytes = (p.nkv * cap * p.hd * 4) as u64;
                entry.k = self
                    .c
                    ._device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
                entry.v = self
                    .c
                    ._device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
                entry.imp = self
                    .c
                    ._device
                    .new_buffer((cap * 4) as u64, MTLResourceOptions::StorageModeShared);
                unsafe {
                    std::ptr::write_bytes(entry.imp.contents() as *mut u8, 0, cap * 4);
                }
                entry.cap = cap;
                entry.stored = usize::MAX;
            }
            if entry.stored != p.cpu_stored {
                // Resync from the owner of record (eviction, rollback,
                // a CPU-path append, or a fresh mirror).
                for h in 0..p.nkv {
                    if p.cpu_k[h].len() != p.cpu_stored * p.hd
                        || p.cpu_v[h].len() != p.cpu_stored * p.hd
                    {
                        return false;
                    }
                    unsafe {
                        let kd = (entry.k.contents() as *mut f32).add(h * entry.cap * p.hd);
                        std::ptr::copy_nonoverlapping(p.cpu_k[h].as_ptr(), kd, p.cpu_k[h].len());
                        let vd = (entry.v.contents() as *mut f32).add(h * entry.cap * p.hd);
                        std::ptr::copy_nonoverlapping(p.cpu_v[h].as_ptr(), vd, p.cpu_v[h].len());
                    }
                }
                entry.stored = p.cpu_stored;
            }
            let out = (
                entry.k.clone(),
                entry.v.clone(),
                entry.imp.clone(),
                entry.cap,
                entry.stored,
            );
            entry.stored += 1; // this token's append
            out
        };

        let cmd = self.ensure_cmd();
        // The whole layer — norm, QKV, RoPE, append, attend, O, FFN,
        // both residuals — is ONE encoder: every step reads the step
        // before it, which serial dispatch already guarantees, so the
        // per-pass kick was pure overhead (see `disp`).
        let enc = cmd.new_compute_command_encoder();
        // 1. attn rmsnorm h → n
        disp(
            enc,
            &self.c.rmsn,
            &[
                (&self.h_b, 0),
                (&const_buf(self.c, l.attn_norm), 0),
                (&self.n_b, 0),
            ],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        // 2. QKV projections n → q_raw / k / v
        let q_b = io_buf(self.c, 40_000_000_003 + l.wq.1, l.wq.1 * 4);
        let k_b = io_buf(self.c, 41_000_000_019 + l.wk.1, l.wk.1 * 4);
        let v_b = io_buf(self.c, 42_000_000_037 + l.wv.1, l.wv.1 * 4);
        {
            let (aq, ak, av) = (
                self.proj_abs(l.wq).unwrap(),
                self.proj_abs(l.wk).unwrap(),
                self.proj_abs(l.wv).unwrap(),
            );
            encode_proj(
                self.c,
                enc,
                &self.fbuf,
                aq.0,
                &aq.1,
                &self.n_b,
                &q_b,
                l.wq.1,
                l.wq.2 / GROUP_SIZE,
            );
            encode_proj(
                self.c,
                enc,
                &self.fbuf,
                ak.0,
                &ak.1,
                &self.n_b,
                &k_b,
                l.wk.1,
                l.wk.2 / GROUP_SIZE,
            );
            encode_proj(
                self.c,
                enc,
                &self.fbuf,
                av.0,
                &av.1,
                &self.n_b,
                &v_b,
                l.wv.1,
                l.wv.2 / GROUP_SIZE,
            );
        }
        // 3. per-head qk-norm + RoPE (gate split into g_b)
        let nhd = p.nh * p.hd;
        let qr_b = io_buf(self.c, 44_000_000_007 + nhd, nhd * 4);
        let g_b = io_buf(self.c, 45_000_000_039 + nhd, nhd * 4);
        let flags = (p.output_gate as u32)
            | ((p.q_norm.is_some() as u32) << 1)
            | ((p.k_norm.is_some() as u32) << 2)
            | ((p.gemma as u32) << 3);
        let qn_b = p
            .q_norm
            .map(|w| const_buf(self.c, w))
            .unwrap_or_else(|| qr_b.clone());
        let kn_b = p
            .k_norm
            .map(|w| const_buf(self.c, w))
            .unwrap_or_else(|| qr_b.clone());
        disp(
            enc,
            &self.c.rqkn,
            &[
                (&q_b, 0),
                (&k_b, 0),
                (&qr_b, 0),
                (&g_b, 0),
                (&qn_b, 0),
                (&kn_b, 0),
                (&const_buf(self.c, p.inv_freq), 0),
            ],
            &[
                p.nh as u32,
                p.nkv as u32,
                p.hd as u32,
                p.rd as u32,
                p.position as u32,
                flags,
            ],
            &[p.eps],
            (((p.nh + p.nkv) * 32) as u64, 256),
        );
        // 4. append this position's K/V into the mirror
        disp(
            enc,
            &self.c.kvapp,
            &[(&k_b, 0), (&v_b, 0), (&k_mb, 0), (&v_mb, 0)],
            &[p.nkv as u32, p.hd as u32, cap as u32, stored as u32],
            &[],
            ((p.nkv * p.hd) as u64, 256),
        );
        // 5. grouped attend (+ Born importance into the mirror's imp).
        //    Flash-decoding: one threadgroup per Q-head, its simdgroups
        //    splitting the stored positions. ~32 positions per simdgroup
        //    is the point where the split stops paying for itself.
        let ao_b = io_buf(self.c, 43_000_000_057 + nhd, nhd * 4);
        let n_pos = stored + 1;
        let cap_sgs = (self.c.gqat.max_total_threads_per_threadgroup() as usize / 32)
            .clamp(1, gqa_split_max());
        let sgs = n_pos.div_ceil(32).clamp(1, cap_sgs);
        let tg_threads = 32 * sgs;
        disp_tg(
            enc,
            &self.c.gqat,
            &[(&qr_b, 0), (&k_mb, 0), (&v_mb, 0), (&ao_b, 0), (&imp_mb, 0)],
            &[
                p.nh as u32,
                (p.nh / p.nkv) as u32,
                p.hd as u32,
                cap as u32,
                n_pos as u32,
            ],
            &[],
            ((p.nh * tg_threads) as u64, tg_threads as u64),
            ((sgs * p.hd + 2 * sgs) * 4) as u64,
        );
        // 6. output gate
        if p.output_gate {
            disp(
                enc,
                &self.c.sgate,
                &[(&ao_b, 0), (&g_b, 0)],
                &[nhd as u32],
                &[],
                (nhd as u64, 256),
            );
        }
        // 7. O + residual + FFN + residual
        self.encode_o_ffn(enc, l, &ao_b);
        enc.end_encoding();
        true
    }
    /// post-norm(h) → n_b, gate/up, SiLU·mul, down, h += d — shared by
    /// the GDN layer tail and the attention suffix. When `delta` is
    /// Some, fuses `h += delta` and `n = rmsnorm(h, post_norm)` into a
    /// single `add_rmsnorm_rows` dispatch instead of separate axpy +
    /// rmsnorm (saves one encoder round trip per call — 2/layer).
    fn encode_post_ffn(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        post_norm: &[f32],
        gate: (usize, usize, usize),
        up: (usize, usize, usize),
        down: (usize, usize, usize),
        delta: Option<&Buffer>,
    ) {
        let inter = gate.1;
        let fg_b = io_buf(self.c, 33_000_000_209 + inter, inter * 4);
        let fu_b = io_buf(self.c, 34_000_000_213 + inter, inter * 4);
        let fa_b = io_buf(self.c, 35_000_000_221 + inter, inter * 4);
        // Fused residual-add + RMSNorm: h += delta (when present),
        // n = rmsnorm(h, post_norm). Uses add_rmsnorm_rows which
        // already handles the `hasd` flag.
        {
            let pn_buf = const_buf(self.c, post_norm);
            enc.set_compute_pipeline_state(&self.c.addnorm);
            enc.set_buffer(0, Some(&self.h_b), 0);
            enc.set_buffer(1, Some(delta.unwrap_or(&self.h_b)), 0);
            enc.set_buffer(2, Some(&pn_buf), 0);
            enc.set_buffer(3, Some(&self.n_b), 0);
            let n_u = self.dims.hidden as u32;
            let g_u = self.dims.gemma as u32;
            let hd_u = delta.is_some() as u32;
            enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &g_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(
                6,
                4,
                &self.dims.eps as *const f32 as *const std::ffi::c_void,
            );
            enc.set_bytes(7, 4, &hd_u as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }
        {
            let (ag, au) = (self.proj_abs(gate).unwrap(), self.proj_abs(up).unwrap());
            encode_proj(
                self.c,
                enc,
                &self.fbuf,
                ag.0,
                &ag.1,
                &self.n_b,
                &fg_b,
                gate.1,
                gate.2 / GROUP_SIZE,
            );
            encode_proj(
                self.c,
                enc,
                &self.fbuf,
                au.0,
                &au.1,
                &self.n_b,
                &fu_b,
                up.1,
                up.2 / GROUP_SIZE,
            );
        }
        {
            enc.set_compute_pipeline_state(&self.c.silu);
            enc.set_buffer(0, Some(&fg_b), 0);
            enc.set_buffer(1, Some(&fu_b), 0);
            enc.set_buffer(2, Some(&fg_b), 0); // dummy col (has_col = 0)
            enc.set_buffer(3, Some(&fa_b), 0);
            let (n_u, hc) = (inter as u32, 0u32);
            enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hc as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        {
            let ad = self.proj_abs(down).unwrap();
            encode_proj(
                self.c,
                enc,
                &self.fbuf,
                ad.0,
                &ad.1,
                &fa_b,
                &self.d_b,
                down.1,
                down.2 / GROUP_SIZE,
            );
        }
        disp_axpy(self.c, enc, &self.d_b, &self.h_b, 1.0, self.dims.hidden);
    }

    /// Encode a run of consecutive GDN layers; recurrent states upload
    /// now and read back via `read_states` after the next `sync`.
    pub fn encode_gdn_run(
        &mut self,
        layers: &[GdnGpuLayer],
        states: &[&[f32]],
        cfg: &GdnGpuCfg,
    ) -> bool {
        if layers.is_empty() || layers.len() != states.len() {
            return false;
        }
        let c = self.c;
        let vd = cfg.nv * cfg.dv;
        let ring_len = (cfg.kk - 1) * cfg.c_dim;
        let s_len = cfg.nv * cfg.dk * cfg.dv;

        // Resolve and validate every projection (Q1 or Q1T) before encoding.
        let mut abss: Vec<[(usize, ProjKind); 6]> = Vec::with_capacity(layers.len());
        for (l, st) in layers.iter().zip(states) {
            if !self.gdn_ok(l, cfg) || st.len() != ring_len + s_len {
                return false;
            }
            let mut a8 = core::array::from_fn(|_| (0usize, ProjKind::Q1));
            for (slot, t) in [l.qkv, l.z, l.out, l.gate, l.up, l.down].iter().enumerate() {
                a8[slot] = self.proj_abs(*t).unwrap();
            }
            abss.push(a8);
        }

        let qkv_b = io_buf(c, 22_000_000_017 + cfg.c_dim, cfg.c_dim * 4);
        let z_b = io_buf(c, 23_000_000_021 + vd, vd * 4);
        let a_b = io_buf(c, 24_000_000_047 + cfg.nv, cfg.nv * 4);
        let b_b = io_buf(c, 25_000_000_071 + cfg.nv, cfg.nv * 4);
        let cq_b = io_buf(c, 26_000_000_081 + cfg.c_dim, cfg.c_dim * 4);
        let g_b = io_buf(c, 27_000_000_093 + cfg.nv, cfg.nv * 4);
        let bt_b = io_buf(c, 28_000_000_129 + cfg.nv, cfg.nv * 4);
        let iq_b = io_buf(c, 29_000_000_131 + cfg.nk, cfg.nk * 4);
        let ik_b = io_buf(c, 30_000_000_133 + cfg.nk, cfg.nk * 4);
        let of_b = io_buf(c, 31_000_000_161 + vd, vd * 4);
        let st_bs: Vec<Buffer> = (0..layers.len())
            .map(|i| {
                io_buf(
                    c,
                    36_000_000_223 + (self.st_next + i) * 613 + ring_len + s_len,
                    (ring_len + s_len) * 4,
                )
            })
            .collect();
        self.st_next += layers.len();

        // Upload states (UMA memcpy into shared buffers) — safe: these
        // slots were read back before the previous sync window closed.
        unsafe {
            for (st, sb) in states.iter().zip(&st_bs) {
                std::ptr::copy_nonoverlapping(st.as_ptr(), sb.contents() as *mut f32, st.len());
            }
        }

        let cmd = self.ensure_cmd();
        let fbuf = self.fbuf.clone();
        let (h_b, n_b, d_b) = (self.h_b.clone(), self.n_b.clone(), self.d_b.clone());
        let enc_one = |pso: &ComputePipelineState,
                       bufs: &[(&Buffer, u64)],
                       words: &[u32],
                       floats: &[f32],
                       grid: (u64, u64)| {
            enc_simple(&cmd, pso, bufs, words, floats, grid);
        };
        let vec_buf = |data: &[f32]| -> Buffer { const_buf(c, data) };

        for (l, (a8, sb)) in layers.iter().zip(abss.iter().zip(&st_bs)) {
            let s_off = (ring_len * 4) as u64;
            // 1. attn rmsnorm h → n
            enc_one(
                &c.rmsn,
                &[(&h_b, 0), (&vec_buf(l.attn_norm), 0), (&n_b, 0)],
                &[cfg.hidden as u32, cfg.gemma as u32],
                &[cfg.eps],
                (256, 256),
            );
            // 2. mixer: qkv, z, a, b (independent — one encoder)
            {
                let enc = cmd.new_compute_command_encoder();
                encode_proj(
                    c,
                    enc,
                    &fbuf,
                    a8[0].0,
                    &a8[0].1,
                    &n_b,
                    &qkv_b,
                    l.qkv.1,
                    l.qkv.2 / GROUP_SIZE,
                );
                encode_proj(
                    c,
                    enc,
                    &fbuf,
                    a8[1].0,
                    &a8[1].1,
                    &n_b,
                    &z_b,
                    l.z.1,
                    l.z.2 / GROUP_SIZE,
                );
                for (t, y) in [(&l.a, &a_b), (&l.b, &b_b)] {
                    let (data, rows, cols) = *t;
                    let wb = vec_buf(data);
                    enc.set_compute_pipeline_state(&c.f16mv);
                    enc.set_buffer(0, Some(&wb), 0);
                    enc.set_buffer(1, Some(&n_b), 0);
                    enc.set_buffer(2, Some(y), 0);
                    let (cu, ru) = (cols as u32, rows as u32);
                    enc.set_bytes(3, 4, &cu as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(4, 4, &ru as *const u32 as *const std::ffi::c_void);
                    let sgs = 8u64;
                    enc.dispatch_thread_groups(
                        MTLSize::new((rows as u64).div_ceil(sgs), 1, 1),
                        MTLSize::new(sgs * 32, 1, 1),
                    );
                }
                enc.end_encoding();
            }
            // 3. conv + silu (reads ring BEFORE the shift)
            enc_one(
                &c.conv,
                &[(&qkv_b, 0), (sb, 0), (&vec_buf(l.conv1d), 0), (&cq_b, 0)],
                &[cfg.c_dim as u32, cfg.kk as u32],
                &[],
                (cfg.c_dim as u64, 256),
            );
            // 4. ring shift + gates + qk norms (one encoder, independent)
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&c.ring);
                enc.set_buffer(0, Some(sb), 0);
                enc.set_buffer(1, Some(&qkv_b), 0);
                let (cd, kk) = (cfg.c_dim as u32, cfg.kk as u32);
                enc.set_bytes(2, 4, &cd as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(3, 4, &kk as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(cfg.c_dim as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.set_compute_pipeline_state(&c.gates);
                enc.set_buffer(0, Some(&a_b), 0);
                enc.set_buffer(1, Some(&b_b), 0);
                enc.set_buffer(2, Some(&vec_buf(l.a_log)), 0);
                enc.set_buffer(3, Some(&vec_buf(l.dt_bias)), 0);
                enc.set_buffer(4, Some(&g_b), 0);
                enc.set_buffer(5, Some(&bt_b), 0);
                let nv = cfg.nv as u32;
                enc.set_bytes(6, 4, &nv as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(MTLSize::new(cfg.nv as u64, 1, 1), MTLSize::new(64, 1, 1));
                enc.set_compute_pipeline_state(&c.qkn);
                enc.set_buffer(0, Some(&cq_b), 0);
                enc.set_buffer(1, Some(&iq_b), 0);
                enc.set_buffer(2, Some(&ik_b), 0);
                let (nk, dk) = (cfg.nk as u32, cfg.dk as u32);
                enc.set_bytes(3, 4, &nk as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(4, 4, &dk as *const u32 as *const std::ffi::c_void);
                let sgs = 8u64;
                enc.dispatch_thread_groups(
                    MTLSize::new((cfg.nk as u64).div_ceil(sgs), 1, 1),
                    MTLSize::new(sgs * 32, 1, 1),
                );
                enc.end_encoding();
            }
            // 5. recurrence + gated norm → of
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&c.stateup);
                enc.set_buffer(0, Some(sb), s_off);
                enc.set_buffer(1, Some(&cq_b), 0);
                enc.set_buffer(2, Some(&z_b), 0);
                enc.set_buffer(3, Some(&g_b), 0);
                enc.set_buffer(4, Some(&bt_b), 0);
                enc.set_buffer(5, Some(&iq_b), 0);
                enc.set_buffer(6, Some(&ik_b), 0);
                enc.set_buffer(7, Some(&vec_buf(l.gnorm)), 0);
                enc.set_buffer(8, Some(&of_b), 0);
                let w4 = [cfg.nv as u32, cfg.nk as u32, cfg.dk as u32, cfg.dv as u32];
                for (i, w) in w4.iter().enumerate() {
                    enc.set_bytes(9 + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
                }
                enc.set_bytes(13, 4, &cfg.eps as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(cfg.nv as u64, 1, 1),
                    MTLSize::new(cfg.dv as u64, 1, 1),
                );
                enc.end_encoding();
            }
            // 6. out_proj of → d;  7. h += d
            {
                let enc = cmd.new_compute_command_encoder();
                encode_proj(
                    c,
                    enc,
                    &fbuf,
                    a8[2].0,
                    &a8[2].1,
                    &of_b,
                    &d_b,
                    l.out.1,
                    l.out.2 / GROUP_SIZE,
                );
                enc.end_encoding();
            }
            // 8–12. post-norm + FFN + residual (shared with attn suffix)
            // Fused: h += d, n = rmsnorm(h, post_norm) — one dispatch.
            {
                let enc = cmd.new_compute_command_encoder();
                self.encode_post_ffn(enc, l.post_norm, l.gate, l.up, l.down, Some(&d_b));
                enc.end_encoding();
            }
        }

        for (sb, st) in st_bs.iter().zip(states) {
            self.dirty.push((sb.clone(), st.len()));
        }
        true
    }
}

/// Host-side inputs for a fully device-resident attention layer.
pub struct AttnDeviceParams<'a> {
    pub kv_id: u64,
    pub layer: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub rd: usize,
    pub position: usize,
    pub eps: f32,
    pub gemma: bool,
    pub output_gate: bool,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub inv_freq: &'a [f32],
    /// CPU rows per head (`[stored × hd]` each) — the owner of record,
    /// used to (re)build the mirror when it diverges.
    pub cpu_k: Vec<&'a [f32]>,
    pub cpu_v: Vec<&'a [f32]>,
    pub cpu_stored: usize,
}

/// After the token's final sync: copy the row the graph appended for
/// (kv_id, layer) out of the mirror (UMA memcpy). `k_out`/`v_out` are
/// `[nkv × hd]`.
pub fn kv_mirror_read_last(
    kv_id: u64,
    layer: usize,
    nkv: usize,
    hd: usize,
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let reg = c.kv_mirrors.lock().unwrap();
    let Some(m) = reg.get(&(kv_id, layer)) else {
        return false;
    };
    if m.stored == 0 || m.stored == usize::MAX || k_out.len() != nkv * hd {
        return false;
    }
    let row = m.stored - 1;
    unsafe {
        let ks = m.k.contents() as *const f32;
        let vs = m.v.contents() as *const f32;
        for h in 0..nkv {
            let off = (h * m.cap + row) * hd;
            std::ptr::copy_nonoverlapping(ks.add(off), k_out[h * hd..].as_mut_ptr(), hd);
            std::ptr::copy_nonoverlapping(vs.add(off), v_out[h * hd..].as_mut_ptr(), hd);
        }
    }
    true
}

/// Add this token's Born-importance mass (mirror accumulator) into
/// `imp_acc` and clear the accumulator. Call after the final sync.
pub fn kv_mirror_take_imp(kv_id: u64, layer: usize, imp_acc: &mut [f32]) {
    let Some(c) = ctx() else { return };
    let reg = c.kv_mirrors.lock().unwrap();
    let Some(m) = reg.get(&(kv_id, layer)) else {
        return;
    };
    let n = imp_acc.len().min(m.cap);
    unsafe {
        let src = m.imp.contents() as *mut f32;
        for (i, dst) in imp_acc.iter_mut().take(n).enumerate() {
            *dst += *src.add(i);
            *src.add(i) = 0.0;
        }
    }
}

/// Drop every mirror belonging to a pipeline (its Drop calls this).
pub fn kv_mirror_drop(kv_id: u64) {
    if let Some(c) = ctx() {
        c.kv_mirrors
            .lock()
            .unwrap()
            .retain(|(id, _), _| *id != kv_id);
    }
}

/// A BLOCK of consecutive GDN layers in one command buffer: hidden
/// state stays device-resident across norm → mixer → conv → recurrence
/// → out_proj → norm → FFN → residuals of every layer; per-layer
/// recurrent states round-trip through shared memory (the CPU remains
/// their owner, so every other path stays coherent for free). One sync
/// per block instead of ~12 per layer.
pub fn gdn_block(
    model: &Arc<CmfModel>,
    layers: &[GdnGpuLayer],
    states: &mut [&mut [f32]],
    cfg: &GdnGpuCfg,
    h: &mut [f32],
) -> bool {
    let dims = GraphDims {
        hidden: cfg.hidden,
        eps: cfg.eps,
        gemma: cfg.gemma,
    };
    let Some(mut g) = TokenGraph::new(model, dims, h) else {
        return false;
    };
    let ro: Vec<&[f32]> = states.iter().map(|s| &**s).collect();
    if !g.encode_gdn_run(layers, &ro, cfg) {
        return false;
    }
    g.sync();
    g.read_states(states);
    g.finish(h);
    true
}

/// `y += w·d` as its own encoder.
/// How many simdgroups may split one Q-head's positions in the decode
/// attend (`CMF_GQA_SPLIT`, default 8). More splitting buys parallelism
/// at depth and costs a wider threadgroup-memory combine.
fn gqa_split_max() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CMF_GQA_SPLIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(1, 32))
            .unwrap_or(8)
    })
}

/// `enc_axpy` into an already-open encoder.
fn disp_axpy(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    d: &Buffer,
    y: &Buffer,
    w: f32,
    n: usize,
) {
    enc.set_compute_pipeline_state(&c.axpy);
    enc.set_buffer(0, Some(d), 0);
    enc.set_buffer(1, Some(y), 0);
    let n_u = n as u32;
    enc.set_bytes(2, 4, &w as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &n_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qtensor::QTensor;
    use cortiq_core::{
        CMF_VERSION, CmfHeader, CmfModel, LayerType, ModelArch, NormStyle, QuantType, TensorDtype,
        TensorSpec,
    };

    /// GPU kernel == CPU path on an lm_head-class q8_row tensor over
    /// a REAL mmap (no-copy buffer). Skipped without a Metal device.
    #[test]
    fn gpu_q8_matvec_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "1") };
        if !enabled() {
            eprintln!("gpu test skipped: no Metal device");
            return;
        }
        let (rows, cols) = (crate::gpu::GPU_MIN_ROWS, 64);
        // Reference q8_row encoder (like tests/roundtrip.rs).
        let mut w = vec![0f32; rows * cols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (((i * 31 + 7) % 197) as f32 / 197.0 - 0.5) * 0.3;
        }
        let mut q = Vec::with_capacity(rows * cols);
        let mut scales = Vec::with_capacity(rows * 2);
        for o in 0..rows {
            let row = &w[o * cols..(o + 1) * cols];
            let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = if absmax == 0.0 { 1e-10 } else { absmax / 127.0 };
            let scale = {
                let h = cortiq_core::quant::f32_to_f16(scale);
                cortiq_core::quant::f16_to_f32(h)
            };
            for &v in row {
                q.push((v / scale).round().clamp(-128.0, 127.0) as i8 as u8);
            }
            scales.extend_from_slice(&cortiq_core::quant::f32_to_f16(scale).to_le_bytes());
        }
        q.extend_from_slice(&scales);

        let arch = ModelArch {
            arch_name: "tiny".into(),
            hidden_size: cols,
            intermediate_size: cols * 2,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: rows,
            layer_types: vec![LayerType::FullAttention],
            rms_norm_eps: 1e-6,
            norm_style: NormStyle::Qwen,
            rope_theta: 1e4,
            tie_word_embeddings: false,
            partial_rotary_factor: 1.0,
            yarn: None,
            attention_heads_per_layer: None,
            local_partial_rotary_factor: None,
            mtp: None,
            moe: None,
            linear_core: None,
            max_position_embeddings: 8,
            linear_conv_kernel_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            hidden_act: "silu".into(),
            embed_multiplier: 1.0,
            query_pre_attn_scalar: None,
            sliding_window: None,
            sliding_window_pattern: None,
            rope_local_base_freq: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            global_partial_rotary_factor: None,
            final_logit_softcapping: None,
        attn_logit_softcapping: None,
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
            attn_v_norm: false,
            num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
            loop_final_norm: false,
        };
        let header = CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch,
            quant_type: QuantType::Q8Row,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
        };
        let spec = TensorSpec {
            name: "lm_head.weight".into(),
            dtype: TensorDtype::Q8Row,
            shape: vec![rows, cols],
            data: q,
        };
        let dir = std::env::temp_dir().join(format!("cmf-gpu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gpu.cmf");
        CmfModel::write(&path, &header, &[spec], None, None).unwrap();
        let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
        let t = QTensor::from_model(&model, "lm_head.weight").unwrap();

        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 + 3) % 89) as f32 / 89.0 - 0.5)
            .collect();
        let mut cpu = vec![0f32; rows];
        // CPU reference: matvec with the GPU disabled is impossible via env
        // (OnceLock) — compute manually from the source weights.
        for o in 0..rows {
            let mut acc = 0f32;
            for i in 0..cols {
                acc += w[o * cols + i] * x[i];
            }
            cpu[o] = acc;
        }
        let mut gpu = vec![0f32; rows];
        t.matvec(&x, &mut gpu, None); // rows ≥ threshold → GPU path
        let mut max_d = 0f32;
        for o in 0..rows {
            max_d = max_d.max((cpu[o] - gpu[o]).abs());
        }
        // q8 grid tolerance: |w|≤0.15, step ≈ absmax/127, dot over 64.
        assert!(max_d < 2e-2, "GPU vs f32 reference: max|Δ| = {max_d}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// GPU q4_tiled kernel == exact f32 reference over a real mmap.
    /// cols=96 (gpr=3) puts tiles at both u32 parities — the unaligned
    /// byte-loader path. Skipped without a Metal device.
    #[test]
    fn gpu_q4t_matvec_matches_reference() {
        unsafe { std::env::set_var("CMF_GPU", "1") };
        if !enabled() {
            eprintln!("gpu test skipped: no Metal device");
            return;
        }
        // Big enough that the file spans several pages (file_buffer
        // rounds the no-copy window DOWN to a page); the trailing pad
        // tensor keeps `w` clear of the truncated last page.
        let (rows, cols) = (1024usize, 96usize);
        let gpr = cols / GROUP_SIZE;
        const TILE: usize = 18;
        let mut payload = vec![0u8; rows * gpr * TILE];
        for r in 0..rows {
            for g in 0..gpr {
                let t = (r * gpr + g) * TILE;
                let sc = 0.02 + 0.001 * ((r + g) % 11) as f32;
                payload[t..t + 2]
                    .copy_from_slice(&cortiq_core::quant::f32_to_f16(sc).to_le_bytes());
                for k in 0..16 {
                    payload[t + 2 + k] = ((r * 37 + g * 11 + k * 13) % 251) as u8;
                }
            }
        }
        let arch = ModelArch {
            arch_name: "tiny".into(),
            hidden_size: cols,
            intermediate_size: cols * 2,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: rows,
            layer_types: vec![LayerType::FullAttention],
            rms_norm_eps: 1e-6,
            norm_style: NormStyle::Qwen,
            rope_theta: 1e4,
            tie_word_embeddings: false,
            partial_rotary_factor: 1.0,
            yarn: None,
            attention_heads_per_layer: None,
            local_partial_rotary_factor: None,
            mtp: None,
            moe: None,
            linear_core: None,
            max_position_embeddings: 8,
            linear_conv_kernel_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            hidden_act: "silu".into(),
            embed_multiplier: 1.0,
            query_pre_attn_scalar: None,
            sliding_window: None,
            sliding_window_pattern: None,
            rope_local_base_freq: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            global_partial_rotary_factor: None,
            final_logit_softcapping: None,
        attn_logit_softcapping: None,
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
            attn_v_norm: false,
            num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
            loop_final_norm: false,
        };
        let header = CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch,
            quant_type: QuantType::Q4Block,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
        };
        let spec = TensorSpec {
            name: "w".into(),
            dtype: TensorDtype::Q4Tiled,
            shape: vec![rows, cols],
            data: payload.clone(),
        };
        let pad = TensorSpec {
            name: "pad".into(),
            dtype: TensorDtype::F32,
            shape: vec![page_size() / 4, 8],
            data: vec![0u8; page_size() * 8],
        };
        let dir = std::env::temp_dir().join(format!("cmf-gpu-q4t-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q4t.cmf");
        CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
        let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
        let idx = model.tensor_index("w").unwrap();
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 + 3) % 89) as f32 / 89.0 - 0.5)
            .collect();
        // Exact f32 reference from the tile bytes: lo nibble → even
        // element, hi → odd, value = (nibble − 8)·scale.
        let mut expect = vec![0f32; rows];
        for r in 0..rows {
            let mut acc = 0f32;
            for g in 0..gpr {
                let t = (r * gpr + g) * 18;
                let s = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([
                    payload[t],
                    payload[t + 1],
                ]));
                for k in 0..16 {
                    let b = payload[t + 2 + k];
                    acc += ((b & 0x0F) as f32 - 8.0) * s * x[g * 32 + 2 * k]
                        + (((b >> 4) & 0x0F) as f32 - 8.0) * s * x[g * 32 + 2 * k + 1];
                }
            }
            expect[r] = acc;
        }
        let mut gpu = vec![0f32; rows];
        assert!(
            q4t_matvec_for_test(&model, idx, &x, rows, cols, &mut gpu),
            "q4t GPU path refused"
        );
        let mut max_d = 0f32;
        for r in 0..rows {
            max_d = max_d.max((expect[r] - gpu[r]).abs());
        }
        assert!(max_d < 1e-3, "GPU q4t vs f32 reference: max|Δ| = {max_d}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// GPU q1 kernel == exact f32 reference over a real mmap. The GPU
    /// math is plain f32 (no A8 quantization), so the tolerance is pure
    /// float-summation noise. Skipped without a Metal device.
    #[test]
    fn gpu_q1_matvec_matches_reference() {
        // Two shapes: single-chunk (cols ≤ 4096) and the CHUNKED path
        // (cols 6144 → two threadgroup-memory chunks — the out_proj
        // shape that a small parity test would never touch).
        gpu_q1_case(512, 256);
        gpu_q1_case(256, 6144);
    }

    fn gpu_q1_case(rows: usize, cols: usize) {
        unsafe { std::env::set_var("CMF_GPU", "1") };
        if !enabled() {
            eprintln!("gpu test skipped: no Metal device");
            return;
        }
        let gpr = cols / GROUP_SIZE;
        // Binary weights ±s per group, packed as q1 tiles.
        let mut payload = Vec::with_capacity(rows * gpr * Q1_TILE);
        let mut w = vec![0f32; rows * cols];
        for o in 0..rows {
            for g in 0..gpr {
                let s = 0.004 + ((o * 7 + g) % 11) as f32 * 0.002;
                let s = cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(s));
                payload.extend_from_slice(&cortiq_core::quant::f32_to_f16(s).to_le_bytes());
                for j in 0..4 {
                    let mut byte = 0u8;
                    for k in 0..8 {
                        let i = g * GROUP_SIZE + j * 8 + k;
                        let bit = ((o * 37 + i * 13) % 5) < 2;
                        if bit {
                            byte |= 1 << k;
                        }
                        w[o * cols + i] = if bit { s } else { -s };
                    }
                    payload.push(byte);
                }
            }
        }
        let arch = ModelArch {
            arch_name: "tiny".into(),
            hidden_size: cols,
            intermediate_size: cols * 2,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: rows,
            layer_types: vec![LayerType::FullAttention],
            rms_norm_eps: 1e-6,
            norm_style: NormStyle::Qwen,
            rope_theta: 1e4,
            tie_word_embeddings: false,
            partial_rotary_factor: 1.0,
            yarn: None,
            attention_heads_per_layer: None,
            local_partial_rotary_factor: None,
            mtp: None,
            moe: None,
            linear_core: None,
            max_position_embeddings: 8,
            linear_conv_kernel_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            hidden_act: "silu".into(),
            embed_multiplier: 1.0,
            query_pre_attn_scalar: None,
            sliding_window: None,
            sliding_window_pattern: None,
            rope_local_base_freq: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            global_partial_rotary_factor: None,
            final_logit_softcapping: None,
        attn_logit_softcapping: None,
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
            attn_v_norm: false,
            num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
            loop_final_norm: false,
        };
        let header = CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch,
            quant_type: QuantType::Vbit,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
        };
        let spec = TensorSpec {
            name: "lm_head.weight".into(),
            dtype: TensorDtype::Q1,
            shape: vec![rows, cols],
            data: payload,
        };
        // The no-copy buffer is truncated to the last FULL page; a q1
        // payload has no trailing scales section, so pad the file past
        // the page boundary with a dummy tensor (in a real model some
        // other tensor plays this role; only the file's very last q1
        // tensor honestly falls back to CPU).
        let pad = TensorSpec {
            name: "pad.weight".into(),
            dtype: TensorDtype::F32,
            shape: vec![4096, 2],
            data: vec![0u8; 4096 * 2 * 4],
        };
        let dir =
            std::env::temp_dir().join(format!("cmf-gpu-q1-{}-{rows}x{cols}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gpu.cmf");
        CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
        let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
        let idx = model.tensor_index("lm_head.weight").unwrap();

        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 17 + 5) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let mut cpu = vec![0f32; rows];
        for o in 0..rows {
            cpu[o] = (0..cols).map(|i| w[o * cols + i] * x[i]).sum();
        }
        // Both kernels, each against its own bound: f32 is near-exact;
        // the half twin accumulates 32-groups in f16 (~1e-3-class) —
        // the loose bound still catches sign/order bugs (those are
        // O(1) wrong, not O(1e-3)).
        for (mode, tol) in [(1u8, 1e-4f32), (2u8, 1e-2f32)] {
            Q1_KERNEL_OVERRIDE.store(mode, std::sync::atomic::Ordering::Relaxed);
            let mut gpu = vec![0f32; rows];
            assert!(
                q1_matvec(&model, idx, &x, rows, cols, &mut gpu),
                "metal q1_matvec refused (mode {mode})"
            );
            let mut max_d = 0f32;
            for o in 0..rows {
                max_d = max_d.max((cpu[o] - gpu[o]).abs());
            }
            assert!(
                max_d < tol,
                "GPU q1 vs f32 reference (mode {mode}): max|Δ| = {max_d}"
            );
        }
        Q1_KERNEL_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
        std::fs::remove_dir_all(&dir).ok();
    }
}
