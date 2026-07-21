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
use cortiq_core::quant::{Q1_TILE, Q1T_TILE, GROUP_SIZE};
use cortiq_core::CmfModel;
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

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
    float acc = 0.0f;
    for (uint i = lane; i < cols4; i += 32) {
        acc += dot(float4(q[base + i]), xs[i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc * rs[row];
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
    float acc = 0.0f;
    for (uint i = lane; i < cols4; i += 32) {
        acc += dot(float4(q[qb + i]), xs[xb + i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[(ulong)bi * rows + row] = acc * rs[row];
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
    device const uint* rp_ptr = (device const uint*)(q + base_len + rid * 4u);
    uint c0 = rp_ptr[0];
    uint c1 = *(device const uint*)(q + base_len + (rid + 1u) * 4u);
    uint ent = base_len + (rows + 1u) * 4u;
    for (uint p = c0; p < c1; ++p) {
        uint e = ent + p * 4u;
        uint col_val = *(device const uint*)(q + e);
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
// Dims contract (checked host-side): hd % 4 == 0, hd <= 128, and for
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
    float xv[4];
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

// Grouped decode attention, one simdgroup per Q-head: online softmax
// over the n stored positions (lane-sliced dims, dim d lives in lane
// d%32 slot d/32), plus a second pass that banks each position's
// probability mass into the Born-importance accumulator (the default
// eviction policy ranks by it). exp/order differ from the CPU attend
// (tolerance-gated, like every GPU reduction here).
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
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgs [[simdgroups_per_threadgroup]])
{
    uint h = tg * sgs + sg;
    if (h >= nh) return;
    uint kh = h / hpk;
    device const float* kh0 = kbuf + (ulong)kh * cap * hd;
    device const float* vh0 = vbuf + (ulong)kh * cap * hd;
    float scale = 1.0f / sqrt((float)hd);
    uint nt = (hd + 31u) / 32u;
    float qv[4];
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        qv[t] = d < hd ? q[(ulong)h * hd + d] * scale : 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint p = 0; p < n; ++p) {
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
    float invl = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        if (d < hd) outb[(ulong)h * hd + d] = acc[t] * invl;
    }
    // Born-importance pass: prob_p = exp(s_p − m)/l summed over heads.
    for (uint p = lane; p < n; p += 32u) {
        device const float* kr = kh0 + (ulong)p * hd;
        float dot = 0.0f;
        for (uint d = 0; d < hd; ++d) {
            dot += q[(ulong)h * hd + d] * kr[d];
        }
        float prob = exp(dot * scale - m) * invl;
        atomic_fetch_add_explicit(&imp[p], prob, memory_order_relaxed);
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
    float qv[4];
    for (uint t = 0; t < nt; ++t) {
        uint d = t * 32u + lane;
        qv[t] = d < hd ? qh[d] * scale : 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
    for (uint p = lane; p < n; p += 32u) {
        device const float* kr = kh0 + (ulong)p * hd;
        float dotv = 0.0f;
        for (uint d = 0; d < hd; ++d) {
            dotv += qh[d] * kr[d];
        }
        float prob = exp(dotv * scale - m) * invl;
        atomic_fetch_add_explicit(&imp[p], prob, memory_order_relaxed);
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
    float xv[4];
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
    threadgroup ushort lut[256];
    for (uint i = lane; i < 243u; i += 32u) {
        lut[i] = Q1T_LUT[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

        for (uint ri = 0u; ri < nr; ++ri) {
            ulong base = ((ulong)(r0 + ri) * gpr + (ulong)g) * 9u;
            ushort scale_bits = (ushort)q[base] | ((ushort)q[base+1u] << 8u);
            half scale = as_type<half>(scale_bits);
            
            uint b2_5 = (uint)q[base+2u] | ((uint)q[base+3u]<<8u) | ((uint)q[base+4u]<<16u) | ((uint)q[base+5u]<<24u);
            ushort b6_7 = (ushort)q[base+6u] | ((ushort)q[base+7u]<<8u);
            uchar b8 = q[base + 8u];

            float gsum = 0.0f;
            ushort p;

            // Byte 2 (0..4)
            p = lut[b2_5 & 0xFF];
            gsum += ((float)((p & 3u) == 1u) - (float)((p & 3u) == 2u)) * xg[0].x;
            gsum += ((float)(((p >> 2u) & 3u) == 1u) - (float)(((p >> 2u) & 3u) == 2u)) * xg[0].y;
            gsum += ((float)(((p >> 4u) & 3u) == 1u) - (float)(((p >> 4u) & 3u) == 2u)) * xg[0].z;
            gsum += ((float)(((p >> 6u) & 3u) == 1u) - (float)(((p >> 6u) & 3u) == 2u)) * xg[0].w;
            gsum += ((float)(((p >> 8u) & 3u) == 1u) - (float)(((p >> 8u) & 3u) == 2u)) * xg[1].x;

            // Byte 3 (5..9)
            p = lut[(b2_5 >> 8u) & 0xFF];
            gsum += ((float)((p & 3u) == 1u) - (float)((p & 3u) == 2u)) * xg[1].y;
            gsum += ((float)(((p >> 2u) & 3u) == 1u) - (float)(((p >> 2u) & 3u) == 2u)) * xg[1].z;
            gsum += ((float)(((p >> 4u) & 3u) == 1u) - (float)(((p >> 4u) & 3u) == 2u)) * xg[1].w;
            gsum += ((float)(((p >> 6u) & 3u) == 1u) - (float)(((p >> 6u) & 3u) == 2u)) * xg[2].x;
            gsum += ((float)(((p >> 8u) & 3u) == 1u) - (float)(((p >> 8u) & 3u) == 2u)) * xg[2].y;

            // Byte 4 (10..14)
            p = lut[(b2_5 >> 16u) & 0xFF];
            gsum += ((float)((p & 3u) == 1u) - (float)((p & 3u) == 2u)) * xg[2].z;
            gsum += ((float)(((p >> 2u) & 3u) == 1u) - (float)(((p >> 2u) & 3u) == 2u)) * xg[2].w;
            gsum += ((float)(((p >> 4u) & 3u) == 1u) - (float)(((p >> 4u) & 3u) == 2u)) * xg[3].x;
            gsum += ((float)(((p >> 6u) & 3u) == 1u) - (float)(((p >> 6u) & 3u) == 2u)) * xg[3].y;
            gsum += ((float)(((p >> 8u) & 3u) == 1u) - (float)(((p >> 8u) & 3u) == 2u)) * xg[3].z;

            // Byte 5 (15..19)
            p = lut[b2_5 >> 24u];
            gsum += ((float)((p & 3u) == 1u) - (float)((p & 3u) == 2u)) * xg[3].w;
            gsum += ((float)(((p >> 2u) & 3u) == 1u) - (float)(((p >> 2u) & 3u) == 2u)) * xg[4].x;
            gsum += ((float)(((p >> 4u) & 3u) == 1u) - (float)(((p >> 4u) & 3u) == 2u)) * xg[4].y;
            gsum += ((float)(((p >> 6u) & 3u) == 1u) - (float)(((p >> 6u) & 3u) == 2u)) * xg[4].z;
            gsum += ((float)(((p >> 8u) & 3u) == 1u) - (float)(((p >> 8u) & 3u) == 2u)) * xg[4].w;

            // Byte 6 (20..24)
            p = lut[b6_7 & 0xFF];
            gsum += ((float)((p & 3u) == 1u) - (float)((p & 3u) == 2u)) * xg[5].x;
            gsum += ((float)(((p >> 2u) & 3u) == 1u) - (float)(((p >> 2u) & 3u) == 2u)) * xg[5].y;
            gsum += ((float)(((p >> 4u) & 3u) == 1u) - (float)(((p >> 4u) & 3u) == 2u)) * xg[5].z;
            gsum += ((float)(((p >> 6u) & 3u) == 1u) - (float)(((p >> 6u) & 3u) == 2u)) * xg[5].w;
            gsum += ((float)(((p >> 8u) & 3u) == 1u) - (float)(((p >> 8u) & 3u) == 2u)) * xg[6].x;

            // Byte 7 (25..29)
            p = lut[b6_7 >> 8u];
            gsum += ((float)((p & 3u) == 1u) - (float)((p & 3u) == 2u)) * xg[6].y;
            gsum += ((float)(((p >> 2u) & 3u) == 1u) - (float)(((p >> 2u) & 3u) == 2u)) * xg[6].z;
            gsum += ((float)(((p >> 4u) & 3u) == 1u) - (float)(((p >> 4u) & 3u) == 2u)) * xg[6].w;
            gsum += ((float)(((p >> 6u) & 3u) == 1u) - (float)(((p >> 6u) & 3u) == 2u)) * xg[7].x;
            gsum += ((float)(((p >> 8u) & 3u) == 1u) - (float)(((p >> 8u) & 3u) == 2u)) * xg[7].y;

            // Byte 8 (30..31)
            p = lut[b8];
            gsum += ((float)((p & 3u) == 1u) - (float)((p & 3u) == 2u)) * xg[7].z;
            gsum += ((float)(((p >> 2u) & 3u) == 1u) - (float)(((p >> 2u) & 3u) == 2u)) * xg[7].w;

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
    device const uint* rp_ptr = (device const uint*)(q + base_len + rid * 4u);
    uint c0 = rp_ptr[0];
    uint c1 = *(device const uint*)(q + base_len + (rid + 1u) * 4u);
    uint ent = base_len + (rows + 1u) * 4u;
    float corr = 0.0f;
    for (uint p = c0; p < c1; ++p) {
        uint e = ent + p * 4u;
        uint col_val = *(device const uint*)(q + e);
        uint col = col_val & 0xFFFF;
        half val = as_type<half>((ushort)(col_val >> 16));
        corr += (float)val * x[col];
    }
    y[rid] += corr;
}

// q4_block: [packed nibbles: rows·gpr·16 B][f16 scales: rows·gpr·2 B]. Group
// gi's nibbles at packed[gi·16], scale at scales[gi·2]; weight = (nib-8)·scale.
// Lets the token graph keep a precise down_proj (or lm_head) on-device without
// quantizing it to ternary. 4 rows/simdgroup, like q1t_matvec.
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
        for (uint ri = 0u; ri < nr; ++ri) {
            uint gi = (r0 + ri) * gpr + g;
            half scale = as_type<half>((ushort)((uint)sc[gi * 2u] | ((uint)sc[gi * 2u + 1u] << 8)));
            device const uchar* pk = q + (ulong)gi * 16u;
            float gsum = 0.0f;
            for (uint k = 0u; k < 16u; ++k) {
                uint b = pk[k];
                gsum += ((float)(b & 0xFu) - 8.0f) * x[xb + k * 2u]
                      + ((float)((b >> 4) & 0xFu) - 8.0f) * x[xb + k * 2u + 1u];
            }
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
"#;

struct Ctx {
    _device: Device,
    queue: CommandQueue,
    q8: ComputePipelineState,
    q8mm: ComputePipelineState,
    q8mmm: ComputePipelineState,
    q1: ComputePipelineState,
    q1h: ComputePipelineState,
    q1t: ComputePipelineState,
    q1t_ov: ComputePipelineState,
    q1t_mm: ComputePipelineState,
    q1t_ovmm: ComputePipelineState,
    q4b: ComputePipelineState,
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
    /// no-copy buffer per file (key — the base address of the mapping).
    file_bufs: Mutex<HashMap<usize, Buffer>>,
    /// row_scale buffer per tensor (key — (base, idx)).
    rs_bufs: Mutex<HashMap<(usize, usize), Buffer>>,
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

static CTX: OnceLock<Option<Ctx>> = OnceLock::new();

fn ctx() -> Option<&'static Ctx> {
    CTX.get_or_init(|| {
        if std::env::var("CMF_GPU").map(|v| v != "0").unwrap_or_else(|_| crate::pipeline::GLOBAL_USE_GPU.load(std::sync::atomic::Ordering::Relaxed)) {
            match init() {
                Ok(c) => {
                    tracing::info!("Metal GPU path: on ({})", c._device.name());
                    Some(c)
                }
                Err(e) => {
                    tracing::warn!("Metal init failed — CPU fallback: {e}");
                    None
                }
            }
        } else {
            None
        }
    })
    .as_ref()
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
        q8mm,
        q8mmm,
        q1,
        q1h,
        q1t,
        q1t_ov,
        q1t_mm,
        q1t_ovmm,
        q4b,
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
    let bytes = model.primary_bytes();
    if c.file_bufs.lock().unwrap().contains_key(&(bytes.as_ptr() as usize)) {
        return true;
    }
    if may_upload {
        let _ = file_buffer(c, bytes);
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

/// no-copy buffer over the file mapping (cached per file).
fn file_buffer(c: &Ctx, bytes: &[u8]) -> Option<(Buffer, usize)> {
    let base = bytes.as_ptr() as usize;
    let page = page_size();
    if base % page != 0 {
        return None; // mmap is always aligned, but we check honestly
    }
    let len = bytes.len() / page * page; // down to the page
    let mut cache = c.file_bufs.lock().unwrap();
    if let Some(b) = cache.get(&base) {
        return Some((b.clone(), len));
    }
    crate::gpu::probe_note_cold();
    let buf = c._device.new_buffer_with_bytes_no_copy(
        bytes.as_ptr() as *const std::ffi::c_void,
        len as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    cache.insert(base, buf.clone());
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
    let Some(c) = ctx() else { return false };
    if cols % 4 != 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let Some(mut abs) = model.entry_abs_offset(entry) else {
        return false; // a neighboring shard — a different mapping; MVP: CPU
    };
    abs += row0 * cols; // offset into the sub-range (the GPU does not need 64-alignment)
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    let qlen = rows * cols; // the int8 part of the blob (quants before scales)
    if abs + qlen > safe_len {
        return false; // the tail is past the buffer's page boundary
    }

    // row_scale — cached; xs/y — per call (small).
    let base = bytes.as_ptr() as usize;
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
        std::ptr::copy_nonoverlapping(
            xs.as_ptr(),
            xs_buf.contents() as *mut f32,
            xs.len(),
        );
    }
    let y_buf = get_io(rows * 4 + 4); // +4: does not share a key with xs of the same length

    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.q8);
    enc.set_buffer(0, Some(&fbuf), abs as u64);
    enc.set_buffer(1, Some(&xs_buf), 0);
    enc.set_buffer(2, Some(&rs_buf), 0);
    enc.set_buffer(3, Some(&y_buf), 0);
    let cols4 = (cols / 4) as u32;
    let rows_u = rows as u32;
    enc.set_bytes(4, 4, &cols4 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    // 256 threads = 8 SIMD groups per threadgroup → 8 rows per group.
    let sgs = 8u64;
    let n_tg = (rows as u64).div_ceil(sgs);
    enc.dispatch_thread_groups(
        MTLSize::new(n_tg, 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);

    unsafe {
        std::ptr::copy_nonoverlapping(
            y_buf.contents() as *const f32,
            out.as_mut_ptr(),
            rows,
        );
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
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
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
            *HALF
                .get_or_init(|| std::env::var("CMF_Q1_HALF").map(|v| v != "0").unwrap_or(true))
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

/// Which GPU kernel a graph projection uses.
#[derive(Clone)]
enum ProjKind {
    Q1,
    Q1t,
    Q4b,
    Q8(Buffer),
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
    enc.set_compute_pipeline_state(&c.q4b);
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
        ProjKind::Q1 => {
            encode_q1_matvec(c, enc, fbuf, abs, in_buf, out_buf, rows, gpr);
        }
        ProjKind::Q8(rs_buf) => {
            encode_q8_matvec(c, enc, fbuf, abs, rs_buf, in_buf, out_buf, rows, gpr);
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
    in_buf: &Buffer,
    out_buf: &Buffer,
    rows: usize,
    gpr: usize,
) {
    enc.set_compute_pipeline_state(&c.q8);
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(in_buf), 0);
    enc.set_buffer(2, Some(rs_buf), 0);
    enc.set_buffer(3, Some(out_buf), 0);
    let cols4 = (gpr * (GROUP_SIZE / 4)) as u32;
    let rows_u = rows as u32;
    enc.set_bytes(4, 4, &cols4 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64;
    let n_tg = (rows as u64).div_ceil(sgs);
    enc.dispatch_thread_groups(
        MTLSize::new(n_tg, 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
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
    let Some(c) = ctx() else { return false };
    if cols % GROUP_SIZE != 0 {
        return false;
    }
    let gpr = cols / GROUP_SIZE;
    let entry = &model.tensors[idx];
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    if abs + rows * gpr * Q1T_TILE > safe_len {
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
    enc.set_compute_pipeline_state(&c.q1t);
    enc.set_buffer(0, Some(&fbuf), abs as u64);
    enc.set_buffer(1, Some(&xs_buf), 0);
    enc.set_buffer(2, Some(&y_buf), 0);
    let gpr_u = gpr as u32;
    let rows_u = rows as u32;
    enc.set_bytes(3, 4, &gpr_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    let sgs = 8u64; // × 4 rows per simdgroup
    enc.dispatch_thread_groups(
        MTLSize::new((rows as u64).div_ceil(sgs * 4), 1, 1),
        MTLSize::new(sgs * 32, 1, 1),
    );
    enc.end_encoding();
    submit_and_wait(c, cmd, &[&y_buf]);
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, out.as_mut_ptr(), rows);
    }
    true
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
            core::ptr::write_unaligned(dst.add(i) as *mut u64, core::mem::transmute::<
                float16x4_t,
                u64,
            >(h));
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

/// Encode one tiled q8 GEMM into an open command buffer (device-resident
/// X and Y). Caller guarantees b ≥ 32 and cols % 4 == 0.
#[allow(clippy::too_many_arguments)]
fn enc_mul_mm(
    c: &Ctx,
    enc: &metal::ComputeCommandEncoderRef,
    fbuf: &Buffer,
    abs: usize,
    rs_buf: &Buffer,
    xs: &Buffer,
    y: &Buffer,
    b: usize,
    rows: usize,
    cols: usize,
) {
    let pso = mm_pipeline(c, rows, cols, 0);
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(fbuf), abs as u64);
    enc.set_buffer(1, Some(xs), 0);
    enc.set_buffer(2, Some(rs_buf), 0);
    enc.set_buffer(3, Some(y), 0);
    let (cols_u, rows_u, b_u) = (cols as u32, rows as u32, b as u32);
    enc.set_bytes(4, 4, &cols_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &rows_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &b_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        MTLSize::new((b as u64).div_ceil(32), (rows as u64).div_ceil(64), 1),
        MTLSize::new(128, 1, 1),
    );
}

fn encode_mul_mm(
    c: &Ctx,
    cmd: &metal::CommandBufferRef,
    fbuf: &Buffer,
    abs: usize,
    rs_buf: &Buffer,
    xs: &Buffer,
    y: &Buffer,
    b: usize,
    rows: usize,
    cols: usize,
) {
    let enc = cmd.new_compute_command_encoder();
    enc_mul_mm(c, enc, fbuf, abs, rs_buf, xs, y, b, rows, cols);
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
            on: std::env::var("CMF_CHUNK_PROF").map(|v| v == "1").unwrap_or(false),
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
            eprintln!("  {label:<12} {ms:8.2} ms  ({n:3}×)  {:4.1}%", ms / total * 100.0);
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
    let Some(first) = layers.first() else { return false };
    if layers.len() != io.len() {
        return false;
    }
    let (nh, nkv, hd, hs, inter) = (first.nh, first.nkv, first.hd, first.hs, first.inter);
    if b < 32
        || hd % 4 != 0
        || hd > 128
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
    let bytes = first.model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    let base = bytes.as_ptr() as usize;

    // ── Phase 1: validate every layer and build its prep (weights
    // resident, shapes uniform, mirror ready).
    let mut preps: Vec<ChunkPrep> = Vec::with_capacity(layers.len());
    for (l, lio) in layers.iter().zip(io.iter()) {
        if l.nh != nh || l.nkv != nkv || l.hd != hd || l.hs != hs || l.inter != inter {
            return false;
        }
        let abs_of = |t: &(usize, usize, usize, &[f32])| -> Option<usize> {
            let entry = l.model.tensors.get(t.0)?;
            let abs = l.model.entry_abs_offset(entry)?;
            (abs + t.1 * t.2 <= safe_len).then_some(abs)
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
        // KV mirror prep (self-healing contract of the decode graph),
        // reserving b rows for the chunk.
        let (k_mb, v_mb, imp_mb, cap, st0) = {
            let mut reg = c.kv_mirrors.lock().unwrap();
            let need = lio.cpu_stored + b;
            let entry = reg.entry((l.kv_id, l.layer)).or_insert_with(|| KvMirror {
                k: c._device.new_buffer(0, MTLResourceOptions::StorageModeShared),
                v: c._device.new_buffer(0, MTLResourceOptions::StorageModeShared),
                imp: c._device.new_buffer(0, MTLResourceOptions::StorageModeShared),
                cap: 0,
                stored: usize::MAX,
            });
            if entry.cap < need {
                let cap = need.next_power_of_two().max(1024);
                let nb = (nkv * cap * hd * 4) as u64;
                entry.k = c._device.new_buffer(nb, MTLResourceOptions::StorageModeShared);
                entry.v = c._device.new_buffer(nb, MTLResourceOptions::StorageModeShared);
                entry.imp =
                    c._device.new_buffer((cap * 4) as u64, MTLResourceOptions::StorageModeShared);
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
                        std::ptr::copy_nonoverlapping(lio.cpu_k[hh].as_ptr(), kd, lio.cpu_k[hh].len());
                        let vd = (entry.v.contents() as *mut f32).add(hh * entry.cap * hd);
                        std::ptr::copy_nonoverlapping(lio.cpu_v[hh].as_ptr(), vd, lio.cpu_v[hh].len());
                    }
                }
                entry.stored = lio.cpu_stored;
            }
            unsafe {
                std::ptr::write_bytes(entry.imp.contents() as *mut u8, 0, need * 4);
            }
            let out = (entry.k.clone(), entry.v.clone(), entry.imp.clone(), entry.cap, entry.stored);
            entry.stored += b;
            out
        };
        preps.push(ChunkPrep { abs, rs, k_mb, v_mb, imp_mb, cap, st0 });
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
        let qn_b = l.q_norm.map(|w| const_buf(c, w)).unwrap_or_else(|| invf.clone());
        let kn_b = l.k_norm.map(|w| const_buf(c, w)).unwrap_or_else(|| invf.clone());
        let add_norm = |cmd: &metal::CommandBufferRef,
                        delta: Option<&Buffer>,
                        w: &Buffer,
                        dst: &Buffer| {
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
            enc_mul_mm(c, enc, &fbuf, prep.abs[0], &prep.rs[0], &n_b, &qraw, b, l.wq.1, l.wq.2);
            enc_mul_mm(c, enc, &fbuf, prep.abs[1], &prep.rs[1], &n_b, &kraw, b, l.wk.1, l.wk.2);
            enc_mul_mm(c, enc, &fbuf, prep.abs[2], &prep.rs[2], &n_b, &vraw, b, l.wv.1, l.wv.2);
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
            let scores =
                io_buf(c, 72_000_000_089 + nkv * m_rows * ncur, nkv * m_rows * ncur * 4);
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
                    enc.dispatch_threads(
                        MTLSize::new(ncur as u64, 32, 1),
                        MTLSize::new(64, 4, 1),
                    );
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
        encode_mul_mm(c, &cmd, &fbuf, prep.abs[3], &prep.rs[3], &attn, &ob, b, l.wo.1, l.wo.2);
        cmd = prof.cut(c, cmd, "mm_o");
        add_norm(&cmd, Some(&ob), &pnorm, &n_b);
        cmd = prof.cut(c, cmd, "axpy+norm");
        {
            let enc = cmd.new_compute_command_encoder();
            enc_mul_mm(c, enc, &fbuf, prep.abs[4], &prep.rs[4], &n_b, &gb, b, l.gate.1, l.gate.2);
            enc_mul_mm(c, enc, &fbuf, prep.abs[5], &prep.rs[5], &n_b, &ub, b, l.up.1, l.up.2);
            enc.end_encoding();
        }
        cmd = prof.cut(c, cmd, "mm_gateup");
        // down GEMM with silu(g)·u fused into the X-tile load — no
        // standalone activation stage, no act-buffer round trip.
        {
            let enc = cmd.new_compute_command_encoder();
            let pso = mm_pipeline(c, l.down.1, l.down.2, 1);
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&fbuf), prep.abs[6] as u64);
            enc.set_buffer(1, Some(&gb), 0);
            enc.set_buffer(2, Some(&ub), 0);
            enc.set_buffer(3, Some(&prep.rs[6]), 0);
            enc.set_buffer(4, Some(&db), 0);
            let (cols_u, rows_u, b_u) = (l.down.2 as u32, l.down.1 as u32, b as u32);
            enc.set_bytes(5, 4, &cols_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &rows_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &b_u as *const u32 as *const std::ffi::c_void);
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
    let Some(abs) = model.entry_abs_offset(entry) else { return false };
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    if abs + rows * cols > safe_len {
        return false;
    }
    let base = bytes.as_ptr() as usize;
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
    let k_arg = if use_mm { cols as u32 } else { (cols / 4) as u32 };
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
        std::ptr::copy_nonoverlapping(
            y_buf.contents() as *const f32, out.as_mut_ptr(), b * rows);
    }
    tracing::debug!("gpu matmat: {rows}x{cols} b={b}");
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
    let Some(abs) = model.entry_abs_offset(entry) else { return false };
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
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
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    let base = bytes.as_ptr() as usize;

    // Validate all tensors before encoding (fail → CPU without partial work).
    let mut abs3 = Vec::with_capacity(jobs.len());
    for j in jobs {
        let mut trio = [0usize; 3];
        for (slot, (idx, rows, cols, _)) in
            [(0, &j.gate), (1, &j.up), (2, &j.down)]
        {
            let entry = &model.tensors[*idx];
            let Some(abs) = model.entry_abs_offset(entry) else { return false };
            let qlen = if j.q1 {
                if cols % GROUP_SIZE != 0 || (cols / GROUP_SIZE) % 2 != 0 {
                    return false;
                }
                rows * (cols / GROUP_SIZE) * Q1_TILE
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
    let disp_elem = |enc: &metal::ComputeCommandEncoderRef,
                     pso: &ComputePipelineState,
                     n: usize| {
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
                  abs: usize, rows: usize, cols: usize, rs: Option<&Buffer>,
                  xs: &Buffer, y: &Buffer| {
        match rs {
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
        // q1: scales live in the tiles — no rs buffers at all.
        let rs3 = if j.q1 {
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
                j.xs_gate.as_ptr(), xsg.contents() as *mut f32, j.xs_gate.len());
            std::ptr::copy_nonoverlapping(
                j.xs_up.as_ptr(), xsu.contents() as *mut f32, j.xs_up.len());
        }

        {
            let enc = cmd.new_compute_command_encoder();
            matvec(enc, trio[0], *grows, *gcols, rs3[0].as_ref(), &xsg, &g_buf);
            matvec(enc, trio[1], *urows, *ucols, rs3[1].as_ref(), &xsu, &u_buf);
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
            matvec(enc, trio[2], *drows, *dcols, rs3[2].as_ref(), &a_buf, &d_buf);
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
        std::ptr::copy_nonoverlapping(
            y_buf.contents() as *const f32, out.as_mut_ptr(), hidden);
    }
    true
}

/// Several independent q8-matvec in a single command buffer (one sync).
/// outs[i].len() == jobs[i].rows.
pub fn matvec_batch(
    model: &Arc<CmfModel>,
    jobs: &[BatchJob],
    outs: &mut [&mut [f32]],
) -> bool {
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() || jobs.len() != outs.len() {
        return false;
    }
    let bytes = model.primary_bytes();
    let Some((fbuf, safe_len)) = file_buffer(c, bytes) else { return false };
    let base = bytes.as_ptr() as usize;

    let mut abss = Vec::with_capacity(jobs.len());
    for j in jobs {
        let entry = &model.tensors[j.idx];
        let Some(abs) = model.entry_abs_offset(entry) else { return false };
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
        let xs_b = get_io(
            8_000_000_209 + slot * 131 + j.xs.len(),
            j.xs.len() * 4,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                j.xs.as_ptr(), xs_b.contents() as *mut f32, j.xs.len());
        }
        let y_b = get_io(9_000_000_341 + slot * 137 + j.rows, j.rows * 4);
        if j.q1 {
            encode_q1_matvec(c, enc, &fbuf, *abs, &xs_b, &y_b, j.rows, j.cols / GROUP_SIZE);
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
            std::ptr::copy_nonoverlapping(
                y_b.contents() as *const f32, out.as_mut_ptr(), j.rows);
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
            c._device.new_buffer(nbytes as u64, MTLResourceOptions::StorageModeShared)
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
    enc.set_compute_pipeline_state(pso);
    for (i, (b, off)) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), *off);
    }
    let base = bufs.len() as u64;
    for (i, w) in words.iter().enumerate() {
        enc.set_bytes(base + i as u64, 4, w as *const u32 as *const std::ffi::c_void);
    }
    for (i, f) in floats.iter().enumerate() {
        enc.set_bytes(
            base + words.len() as u64 + i as u64,
            4,
            f as *const f32 as *const std::ffi::c_void,
        );
    }
    enc.dispatch_threads(MTLSize::new(grid.0, 1, 1), MTLSize::new(grid.1, 1, 1));
    enc.end_encoding();
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
        let (fbuf, safe_len) = file_buffer(c, model.primary_bytes())?;
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

    /// Resolve a projection tensor accepting Q1 / Q1T / Q4-block.
    fn proj_abs(&self, t: (usize, usize, usize)) -> Option<(usize, ProjKind)> {
        match self.model.tensors[t.0].dtype {
            cortiq_core::TensorDtype::Q1 => self.q1_abs(t).map(|a| (a, ProjKind::Q1)),
            cortiq_core::TensorDtype::Q1T => self.q1t_abs(t).map(|a| (a, ProjKind::Q1t)),
            cortiq_core::TensorDtype::Q4Block => self.q4b_abs(t).map(|a| (a, ProjKind::Q4b)),
            cortiq_core::TensorDtype::Q8Row => self.q8_abs(t).map(|(a, rs)| (a, ProjKind::Q8(rs))),
            _ => None,
        }
    }

    /// Validate one q8_row tensor and cache its row_scale buffer.
    fn q8_abs(&self, t: (usize, usize, usize)) -> Option<(usize, Buffer)> {
        let (idx, rows, cols) = t;
        if cols % 4 != 0 {
            return None;
        }
        let entry = &self.model.tensors[idx];
        let abs = self.model.entry_abs_offset(entry)?;
        let qlen = rows * cols;
        if abs + qlen + rows * 4 > self.safe_len {
            return None;
        }

        let base = self.model.primary_bytes().as_ptr() as usize;
        let c = self.c;
        let rs_buf = {
            let mut cache = c.rs_bufs.lock().unwrap();
            cache
                .entry((base, idx))
                .or_insert_with(|| {
                    crate::gpu::probe_note_cold();
                    let bytes = self.model.entry_bytes(entry);
                    let scales_bytes = &bytes[qlen..qlen + rows * 4];
                    c._device.new_buffer_with_data(
                        scales_bytes.as_ptr() as *const std::ffi::c_void,
                        (rows * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    )
                })
                .clone()
        };
        Some((abs, rs_buf))
    }

    /// Pre-flight check for a GDN layer (call before any encode).
    pub fn gdn_ok(&self, l: &GdnGpuLayer, cfg: &GdnGpuCfg) -> bool {
        if cfg.kk < 2 || cfg.dv % 32 != 0 || cfg.dv > 1024 || cfg.hidden != self.dims.hidden {
            return false;
        }
        if l.a.0.len() != l.a.1 * l.a.2 || l.b.0.len() != l.b.1 * l.b.2 {
            return false;
        }
        [l.qkv, l.z, l.out, l.gate, l.up, l.down].iter().all(|t| self.proj_abs(*t).is_some())
    }

    /// Pre-flight check for a full-attention layer.
    pub fn attn_ok(&self, l: &AttnGpuLayer) -> bool {
        // The suffix reads the attention output back through ao (wo
        // cols) and writes hidden (wo rows) — both must match dims.
        if l.wo.1 != self.dims.hidden || l.down.1 != self.dims.hidden {
            return false;
        }
        [l.wq, l.wk, l.wv, l.wo, l.gate, l.up, l.down].iter().all(|t| self.proj_abs(*t).is_some())
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
            &[(&self.h_b, 0), (&const_buf(self.c, norm), 0), (&self.n_b, 0)],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        let (abs, q1t) = self.proj_abs(lm).unwrap();
        let lg_b = io_buf(self.c, 44_000_000_077 + lm.1, lm.1 * 4);
        let enc = cmd.new_compute_command_encoder();
        encode_proj(self.c, enc, &self.fbuf, abs, &q1t, &self.n_b, &lg_b, lm.1, lm.2 / GROUP_SIZE);
        enc.end_encoding();
        self.logits_b = Some(lg_b);
    }

    /// Copy the finished logits (call after `sync`; out may be shorter
    /// than the head's rows — trailing rows are padding vocab).
    pub fn read_logits(&mut self, out: &mut [f32]) {
        let lg_b = self.logits_b.take().expect("read_logits without encode_lm_head");
        unsafe {
            std::ptr::copy_nonoverlapping(lg_b.contents() as *const f32, out.as_mut_ptr(), out.len());
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
            &[(&self.h_b, 0), (&const_buf(self.c, l.attn_norm), 0), (&self.n_b, 0)],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        let q_b = io_buf(self.c, 40_000_000_003 + l.wq.1, l.wq.1 * 4);
        let k_b = io_buf(self.c, 41_000_000_019 + l.wk.1, l.wk.1 * 4);
        let v_b = io_buf(self.c, 42_000_000_037 + l.wv.1, l.wv.1 * 4);
        let enc = cmd.new_compute_command_encoder();
        encode_proj(self.c, enc, &self.fbuf, aq.0, &aq.1, &self.n_b, &q_b, l.wq.1, l.wq.2 / GROUP_SIZE);
        encode_proj(self.c, enc, &self.fbuf, ak.0, &ak.1, &self.n_b, &k_b, l.wk.1, l.wk.2 / GROUP_SIZE);
        encode_proj(self.c, enc, &self.fbuf, av.0, &av.1, &self.n_b, &v_b, l.wv.1, l.wv.2 / GROUP_SIZE);
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
        self.encode_o_ffn(&cmd, l, &ao_b);
    }

    /// O-projection from a device-resident attention output + residual
    /// + post-norm + FFN + residual.
    fn encode_o_ffn(&self, cmd: &metal::CommandBufferRef, l: &AttnGpuLayer, ao_b: &Buffer) {
        {
            let enc = cmd.new_compute_command_encoder();
            let (abs, q1t) = self.proj_abs(l.wo).unwrap();
            encode_proj(self.c, enc, &self.fbuf, abs, &q1t, ao_b, &self.d_b, l.wo.1, l.wo.2 / GROUP_SIZE);
            enc.end_encoding();
        }
        // Fused: h += d_b, n = rmsnorm(h, post_norm) — one dispatch
        // instead of separate enc_axpy + rmsnorm.
        self.encode_post_ffn(cmd, l.post_norm, l.gate, l.up, l.down, Some(&self.d_b));
    }

    /// Dims contract of the device-attend kernels (host-side check).
    pub fn attn_device_ok(&self, l: &AttnGpuLayer, p: &AttnDeviceParams) -> bool {
        self.attn_ok(l)
            && p.hd % 4 == 0
            && p.hd <= 128
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
                k: self.c._device.new_buffer(0, MTLResourceOptions::StorageModeShared),
                v: self.c._device.new_buffer(0, MTLResourceOptions::StorageModeShared),
                imp: self.c._device.new_buffer(0, MTLResourceOptions::StorageModeShared),
                cap: 0,
                stored: usize::MAX, // force first-touch upload
            });
            if entry.cap < need {
                let cap = need.next_power_of_two().max(1024);
                let bytes = (p.nkv * cap * p.hd * 4) as u64;
                entry.k = self.c._device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
                entry.v = self.c._device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
                entry.imp =
                    self.c._device.new_buffer((cap * 4) as u64, MTLResourceOptions::StorageModeShared);
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
            let out =
                (entry.k.clone(), entry.v.clone(), entry.imp.clone(), entry.cap, entry.stored);
            entry.stored += 1; // this token's append
            out
        };

        let cmd = self.ensure_cmd();
        // 1. attn rmsnorm h → n
        enc_simple(
            &cmd,
            &self.c.rmsn,
            &[(&self.h_b, 0), (&const_buf(self.c, l.attn_norm), 0), (&self.n_b, 0)],
            &[self.dims.hidden as u32, self.dims.gemma as u32],
            &[self.dims.eps],
            (256, 256),
        );
        // 2. QKV projections n → q_raw / k / v
        let q_b = io_buf(self.c, 40_000_000_003 + l.wq.1, l.wq.1 * 4);
        let k_b = io_buf(self.c, 41_000_000_019 + l.wk.1, l.wk.1 * 4);
        let v_b = io_buf(self.c, 42_000_000_037 + l.wv.1, l.wv.1 * 4);
        {
            let enc = cmd.new_compute_command_encoder();
            let (aq, ak, av) = (
                self.proj_abs(l.wq).unwrap(),
                self.proj_abs(l.wk).unwrap(),
                self.proj_abs(l.wv).unwrap(),
            );
            encode_proj(self.c, enc, &self.fbuf, aq.0, &aq.1, &self.n_b, &q_b, l.wq.1, l.wq.2 / GROUP_SIZE);
            encode_proj(self.c, enc, &self.fbuf, ak.0, &ak.1, &self.n_b, &k_b, l.wk.1, l.wk.2 / GROUP_SIZE);
            encode_proj(self.c, enc, &self.fbuf, av.0, &av.1, &self.n_b, &v_b, l.wv.1, l.wv.2 / GROUP_SIZE);
            enc.end_encoding();
        }
        // 3. per-head qk-norm + RoPE (gate split into g_b)
        let nhd = p.nh * p.hd;
        let qr_b = io_buf(self.c, 44_000_000_007 + nhd, nhd * 4);
        let g_b = io_buf(self.c, 45_000_000_039 + nhd, nhd * 4);
        let flags = (p.output_gate as u32)
            | ((p.q_norm.is_some() as u32) << 1)
            | ((p.k_norm.is_some() as u32) << 2)
            | ((p.gemma as u32) << 3);
        let qn_b = p.q_norm.map(|w| const_buf(self.c, w)).unwrap_or_else(|| qr_b.clone());
        let kn_b = p.k_norm.map(|w| const_buf(self.c, w)).unwrap_or_else(|| qr_b.clone());
        enc_simple(
            &cmd,
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
        enc_simple(
            &cmd,
            &self.c.kvapp,
            &[(&k_b, 0), (&v_b, 0), (&k_mb, 0), (&v_mb, 0)],
            &[p.nkv as u32, p.hd as u32, cap as u32, stored as u32],
            &[],
            ((p.nkv * p.hd) as u64, 256),
        );
        // 5. grouped attend (+ Born importance into the mirror's imp)
        let ao_b = io_buf(self.c, 43_000_000_057 + nhd, nhd * 4);
        enc_simple(
            &cmd,
            &self.c.gqat,
            &[(&qr_b, 0), (&k_mb, 0), (&v_mb, 0), (&ao_b, 0), (&imp_mb, 0)],
            &[
                p.nh as u32,
                (p.nh / p.nkv) as u32,
                p.hd as u32,
                cap as u32,
                (stored + 1) as u32,
            ],
            &[],
            ((p.nh * 32) as u64, 256),
        );
        // 6. output gate
        if p.output_gate {
            enc_simple(
                &cmd,
                &self.c.sgate,
                &[(&ao_b, 0), (&g_b, 0)],
                &[nhd as u32],
                &[],
                (nhd as u64, 256),
            );
        }
        // 7. O + residual + FFN + residual
        self.encode_o_ffn(&cmd, l, &ao_b);
        true
    }
    /// post-norm(h) → n_b, gate/up, SiLU·mul, down, h += d — shared by
    /// the GDN layer tail and the attention suffix. When `delta` is
    /// Some, fuses `h += delta` and `n = rmsnorm(h, post_norm)` into a
    /// single `add_rmsnorm_rows` dispatch instead of separate axpy +
    /// rmsnorm (saves one encoder round trip per call — 2/layer).
    fn encode_post_ffn(
        &self,
        cmd: &metal::CommandBufferRef,
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
            let enc = cmd.new_compute_command_encoder();
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
            enc.set_bytes(6, 4, &self.dims.eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &hd_u as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            let (ag, au) = (self.proj_abs(gate).unwrap(), self.proj_abs(up).unwrap());
            encode_proj(self.c, enc, &self.fbuf, ag.0, &ag.1, &self.n_b, &fg_b, gate.1, gate.2 / GROUP_SIZE);
            encode_proj(self.c, enc, &self.fbuf, au.0, &au.1, &self.n_b, &fu_b, up.1, up.2 / GROUP_SIZE);
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.c.silu);
            enc.set_buffer(0, Some(&fg_b), 0);
            enc.set_buffer(1, Some(&fu_b), 0);
            enc.set_buffer(2, Some(&fg_b), 0); // dummy col (has_col = 0)
            enc.set_buffer(3, Some(&fa_b), 0);
            let (n_u, hc) = (inter as u32, 0u32);
            enc.set_bytes(4, 4, &n_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hc as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }
        {
            let enc = cmd.new_compute_command_encoder();
            let ad = self.proj_abs(down).unwrap();
            encode_proj(self.c, enc, &self.fbuf, ad.0, &ad.1, &fa_b, &self.d_b, down.1, down.2 / GROUP_SIZE);
            enc.end_encoding();
        }
        enc_axpy(self.c, cmd, &self.d_b, &self.h_b, 1.0, self.dims.hidden);
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
            encode_proj(c, enc, &fbuf, a8[0].0, &a8[0].1, &n_b, &qkv_b, l.qkv.1, l.qkv.2 / GROUP_SIZE);
            encode_proj(c, enc, &fbuf, a8[1].0, &a8[1].1, &n_b, &z_b, l.z.1, l.z.2 / GROUP_SIZE);
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
            enc.dispatch_threads(MTLSize::new(cfg.c_dim as u64, 1, 1), MTLSize::new(256, 1, 1));
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
            encode_proj(c, enc, &fbuf, a8[2].0, &a8[2].1, &of_b, &d_b, l.out.1, l.out.2 / GROUP_SIZE);
            enc.end_encoding();
        }
        // 8–12. post-norm + FFN + residual (shared with attn suffix)
        // Fused: h += d, n = rmsnorm(h, post_norm) — one dispatch.
        self.encode_post_ffn(&cmd, l.post_norm, l.gate, l.up, l.down, Some(&d_b));
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
    let Some(m) = reg.get(&(kv_id, layer)) else { return false };
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
    let Some(m) = reg.get(&(kv_id, layer)) else { return };
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
        c.kv_mirrors.lock().unwrap().retain(|(id, _), _| *id != kv_id);
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
    let dims = GraphDims { hidden: cfg.hidden, eps: cfg.eps, gemma: cfg.gemma };
    let Some(mut g) = TokenGraph::new(model, dims, h) else { return false };
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
fn enc_axpy(c: &Ctx, cmd: &metal::CommandBufferRef, d: &Buffer, y: &Buffer, w: f32, n: usize) {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&c.axpy);
    enc.set_buffer(0, Some(d), 0);
    enc.set_buffer(1, Some(y), 0);
    let n_u = n as u32;
    enc.set_bytes(2, 4, &w as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &n_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::{
        CmfHeader, CmfModel, LayerType, ModelArch, NormStyle, QuantType, TensorDtype,
        TensorSpec, CMF_VERSION,
    };
    use crate::qtensor::QTensor;

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
            scales.extend_from_slice(
                &cortiq_core::quant::f32_to_f16(scale).to_le_bytes());
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
            attn_v_norm: false,
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
            attn_v_norm: false,
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
        let dir = std::env::temp_dir().join(format!("cmf-gpu-q1-{}-{rows}x{cols}", std::process::id()));
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
            assert!(max_d < tol, "GPU q1 vs f32 reference (mode {mode}): max|Δ| = {max_d}");
        }
        Q1_KERNEL_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
        std::fs::remove_dir_all(&dir).ok();
    }
}
