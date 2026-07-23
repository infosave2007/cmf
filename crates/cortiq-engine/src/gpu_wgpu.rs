//! Cross-platform GPU backend (C1): wgpu → Vulkan / DX12 / Metal
//! (NVIDIA, AMD Radeon, Intel Arc, Apple). Implements the same contract as
//! `gpu_metal.rs`, behind the `gpu.rs` facade — runtime call-sites do not change.
//!
//! Difference from the Metal path: a discrete card has no unified memory, so
//! the quantized weights are LOADED into VRAM ONCE (residency cache keyed by
//! tensor index) — that is where the win lives (VRAM bandwidth ×5–10 vs CPU). The math
//! is identical to CPU/Metal: y[o] = row_scale[o]·Σ q[o,i]·xs[i], where xs is already
//! prescaled by the θ field (the two-field q8_2f folds into the input prescale).
//!
//! Enabling: `CMF_GPU=wgpu` (or `=1` on non-macOS, where wgpu is the only backend).
//! Any init/limit failure — `false` and an honest CPU path.

use crate::gpu::{BatchJob, MoeJob};
use cortiq_core::CmfModel;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::util::DeviceExt;

/// Workgroup limit per dimension (WebGPU minimum; lm_head has more
/// rows — we use grid-stride in the shader).
const MAX_WG: u32 = 65_535;

const WGSL: &str = r#"
struct Params { cols4: u32, rows: u32, row0_words: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       q  : array<u32>;   // 4×i8 packed into u32, row-major
@group(0) @binding(1) var<storage, read>       xs : array<f32>;   // cols, already prescaled by the θ field
@group(0) @binding(2) var<storage, read>       rs : array<f32>;   // row scales for the range
@group(0) @binding(3) var<storage, read_write> y  : array<f32>;   // output: rows
@group(0) @binding(4) var<uniform>             p  : Params;

var<workgroup> partial: array<f32, 64>;

// Exact unpack of 4 signed bytes from u32 (little-endian) — like char4→
// float4 on Metal, without snorm error.
fn i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    let b0 = (s << 24u) >> 24u;
    let b1 = (s << 16u) >> 24u;
    let b2 = (s <<  8u) >> 24u;
    let b3 =  s          >> 24u;
    return vec4<f32>(f32(b0), f32(b1), f32(b2), f32(b3));
}

// Grid-stride over rows: the number of workgroups is capped at 65535/dimension,
// while rows (lm_head) number in the hundreds of thousands; one group processes rows
// wid.x, wid.x+nwg.x, … , reducing each with 64 threads.
@compute @workgroup_size(64)
fn q8_matvec(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    var row = wid.x;
    loop {
        if (row >= p.rows) { break; }
        let base = p.row0_words + row * p.cols4;
        var acc = 0.0;
        var i = lid;
        loop {
            if (i >= p.cols4) { break; }
            let v = i8x4(q[base + i]);
            let xi = i * 4u;
            let xv = vec4<f32>(xs[xi], xs[xi + 1u], xs[xi + 2u], xs[xi + 3u]);
            acc = acc + dot(v, xv);
            i = i + 64u;
        }
        partial[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial[lid] = partial[lid] + partial[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) { y[row] = partial[0] * rs[row]; }
        workgroupBarrier(); // before partial is reused by the next row
        row = row + nwg.x;
    }
}

// GEMM of the prefill batch: y[bi, o] = rs[o]·Σ q[o,i]·xs[bi,i]. One workgroup
// per (row, position); the quant row stays hot in cache across bi.
struct MMParams { cols4: u32, rows: u32, nb: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       qm  : array<u32>;
@group(0) @binding(1) var<storage, read>       xsm : array<f32>;  // [nb, cols] row-major
@group(0) @binding(2) var<storage, read>       rsm : array<f32>;  // [rows]
@group(0) @binding(3) var<storage, read_write> ym  : array<f32>;  // [nb, rows] row-major
@group(0) @binding(4) var<uniform>             pm  : MMParams;

var<workgroup> partial_mm: array<f32, 64>;

@compute @workgroup_size(64)
fn q8_matmat(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let bi = wid.y;
    if (bi >= pm.nb) { return; }
    let xb = bi * pm.cols4 * 4u;
    var row = wid.x;
    loop {
        if (row >= pm.rows) { break; }
        let qb = row * pm.cols4;
        var acc = 0.0;
        var i = lid;
        loop {
            if (i >= pm.cols4) { break; }
            let v = i8x4(qm[qb + i]);
            let xi = xb + i * 4u;
            let xv = vec4<f32>(xsm[xi], xsm[xi + 1u], xsm[xi + 2u], xsm[xi + 3u]);
            acc = acc + dot(v, xv);
            i = i + 64u;
        }
        partial_mm[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial_mm[lid] = partial_mm[lid] + partial_mm[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) { ym[bi * pm.rows + row] = partial_mm[0] * rsm[row]; }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// q1: 6-byte tiles [f16 scale][4B sign bits] per 32-group; gpr is even,
// so a row is whole 12-byte tile PAIRS = 3 u32 each (same layout walk
// as the Metal kernel). Bit set → +x; np = gpr/2 tile-pairs/row (64 cols each).
//
// FAST kernel (the FFN q1 matvecs are ~59% of a 27B decode token): one
// workgroup owns 16 output ROWS, 16 lanes/row (256 threads). Activations are
// staged into shared memory in 1024-col tiles and REUSED across the 16 rows
// (16× fewer activation loads). Sign unpack is a branchless XOR sign-flip
// (bit clear ⇒ flip the f32 sign bit) instead of 32 vec4 selects.
struct Q1Params { np: u32, rows: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       q1w : array<u32>;
@group(0) @binding(1) var<storage, read>       q1x : array<f32>;   // raw f32 activations
@group(0) @binding(2) var<storage, read_write> q1y : array<f32>;
@group(0) @binding(3) var<uniform>             q1p : Q1Params;

var<workgroup> partial_q1: array<f32, 256>;   // 16 rows × 16 lanes
// 1024-col activation tile, PADDED to 33 slots per 32-col group. The read
// pattern is lane*64 + j*4 (all 16 lanes share bank (j*4) mod 32 with a flat
// 1024 tile => 16-way bank conflict, ~8x LSU penalty on the dominant inner
// loop). Padding to stride-33 spreads the lanes across 16 distinct banks
// (66 mod 32 = 2). Same math/accumulation order => token-identical.
var<workgroup> q1xs: array<f32, 1056>;        // 32 groups × 33

// Sum of ±x over one 32-weight group; x read from the shared tile at xbase.
// bit=1 → +x, bit=0 → -x, done by XORing the f32 sign bit (no select chain).
fn q1_tile_sum(bits: u32, xbase: u32) -> f32 {
    var s = vec4<f32>(0.0);
    let pb = (xbase >> 5u) * 33u;   // xbase is a multiple of 32 => padded group base
    for (var j = 0u; j < 8u; j = j + 1u) {
        let nib = bits >> (j * 4u);
        let o = pb + j * 4u;         // j*4+{0..3} stays in [0,32) < 33: no group crossing
        let x = vec4<f32>(q1xs[o], q1xs[o + 1u], q1xs[o + 2u], q1xs[o + 3u]);
        let m = vec4<u32>(
            ((nib & 1u) ^ 1u) << 31u,
            (((nib >> 1u) & 1u) ^ 1u) << 31u,
            (((nib >> 2u) & 1u) ^ 1u) << 31u,
            (((nib >> 3u) & 1u) ^ 1u) << 31u);
        s = s + bitcast<vec4<f32>>(bitcast<vec4<u32>>(x) ^ m);
    }
    return s.x + s.y + s.z + s.w;
}

@compute @workgroup_size(128)
fn q1_matvec(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let cols = q1p.np * 64u;
    let r = lid / 16u;      // which of the 8 rows this thread serves
    let lane = lid % 16u;   // which tile-pair lane within a column tile
    var row0 = wid.x * 8u;
    loop {
        if (row0 >= q1p.rows) { break; }
        let row = row0 + r;
        var acc = 0.0;
        var ti = 0u;                       // column tile start, in tile-pairs
        loop {
            if (ti >= q1p.np) { break; }
            // Cooperatively stage 1024 activations (16 tile-pairs) into shared.
            let c0 = ti * 64u;
            var k = lid;
            loop {
                if (k >= 1024u) { break; }
                let c = c0 + k;
                q1xs[(k >> 5u) * 33u + (k & 31u)] = select(0.0, q1x[c], c < cols);
                k = k + 128u;
            }
            workgroupBarrier();
            let pi = ti + lane;            // this lane's tile-pair
            if (row < q1p.rows && pi < q1p.np) {
                let base = row * q1p.np * 3u + pi * 3u;
                let a0 = q1w[base]; let a1 = q1w[base + 1u]; let a2 = q1w[base + 2u];
                let s0 = unpack2x16float(a0).x;
                let s1 = unpack2x16float(a1).y;
                let bits0 = (a0 >> 16u) | (a1 << 16u);
                let xb = lane * 64u;       // local offset of this pair in q1xs
                acc = acc + s0 * q1_tile_sum(bits0, xb) + s1 * q1_tile_sum(a2, xb + 32u);
            }
            workgroupBarrier();
            ti = ti + 16u;
        }
        partial_q1[lid] = acc;
        workgroupBarrier();
        // reduce the 16 lanes of each row (blocks of 16 in partial_q1)
        if (lane < 8u) { partial_q1[lid] = partial_q1[lid] + partial_q1[lid + 8u]; }
        workgroupBarrier();
        if (lane < 4u) { partial_q1[lid] = partial_q1[lid] + partial_q1[lid + 4u]; }
        workgroupBarrier();
        if (lane < 2u) { partial_q1[lid] = partial_q1[lid] + partial_q1[lid + 2u]; }
        workgroupBarrier();
        if (lane < 1u) { partial_q1[lid] = partial_q1[lid] + partial_q1[lid + 1u]; }
        workgroupBarrier();
        if (lane == 0u && row < q1p.rows) { q1y[row] = partial_q1[lid]; }
        workgroupBarrier();
        row0 = row0 + nwg.x * 8u;
    }
}

// Tiled GEMM for wide prefill batches (the WGSL cousin of Metal's
// q8_mul_mm; WGSL has no subgroup matrices, so this is the classic
// register-blocked form): a 64(b)×64(rows) C-tile per 16×16 workgroup,
// each thread owning a 4×4 accumulator block; X and dequantized W stage
// through 8 KB of workgroup memory in K-steps of 16. The naive kernel
// above re-reads every W row per position — here W is read once per 64
// positions. Perf is hardware-dependent by design: the runtime probe
// decides per machine whether this beats the CPU, so a card where it
// loses simply keeps the CPU path.
var<workgroup> mm_at: array<f32, 64 * 16>;
var<workgroup> mm_wt: array<f32, 64 * 16>;

@compute @workgroup_size(16, 16)
fn q8_mul_mm(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = pm.cols4 * 4u;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) { acc[i][j] = 0.0; }
    }
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        // Stage X tile [64×16] (4 f32 per thread) and W tile [64×16]
        // (one u32 = 4 quants per thread per round).
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let m = t / 4u;
            let k4 = t % 4u;
            var xv = vec4<f32>(0.0);
            if (m0 + m < pm.nb && (k0 / 4u) + k4 < pm.cols4) {
                let xi = (m0 + m) * cols + k0 + k4 * 4u;
                xv = vec4<f32>(xsm[xi], xsm[xi + 1u], xsm[xi + 2u], xsm[xi + 3u]);
            }
            let dst = m * 16u + k4 * 4u;
            mm_at[dst] = xv.x;
            mm_at[dst + 1u] = xv.y;
            mm_at[dst + 2u] = xv.z;
            mm_at[dst + 3u] = xv.w;
        }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let n = t / 4u;
            let k4 = t % 4u;
            var wv = vec4<f32>(0.0);
            if (n0 + n < pm.rows && (k0 / 4u) + k4 < pm.cols4) {
                wv = i8x4(qm[(n0 + n) * pm.cols4 + (k0 / 4u) + k4]);
            }
            let dst = n * 16u + k4 * 4u;
            mm_wt[dst] = wv.x;
            mm_wt[dst + 1u] = wv.y;
            mm_wt[dst + 2u] = wv.z;
            mm_wt[dst + 3u] = wv.w;
        }
        workgroupBarrier();
        // 4×4 outer-product accumulation over the 16 staged K values.
        for (var k = 0u; k < 16u; k = k + 1u) {
            var av: array<f32, 4>;
            var wv: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                av[i] = mm_at[(lid.y * 4u + i) * 16u + k];
                wv[i] = mm_wt[(lid.x * 4u + i) * 16u + k];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i][j] = acc[i][j] + av[i] * wv[j];
                }
            }
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + lid.y * 4u + i;
        if (m >= pm.nb) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let n = n0 + lid.x * 4u + j;
            if (n < pm.rows) {
                ym[m * pm.rows + n] = acc[i][j] * rsm[n];
            }
        }
    }
}

// Tiled q1 GEMM for wide batches (prefill / speculative K-token decode): the
// q1 twin of q8_mul_mm. Reuses the mul_mm bindings (rsm is unused — q1's scale
// is per-32-group and folded into the staged weight). Decode a 4-wide run of
// weights for one output row: 4 cols in one 32-group share a bit-word + scale;
// bit set → +scale, clear → −scale (XOR the sign bit). cols4 = cols/4, so the
// row has np = cols4/16 six-byte tile-pairs (64 cols each, 2 groups of 32).
fn q1_w4(n: u32, k: u32, np: u32) -> vec4<f32> {
    let pi = k / 64u;
    let off = k % 64u;                 // 4-aligned ⇒ never straddles a 32-group
    let base = n * np * 3u + pi * 3u;
    let a0 = qm[base]; let a1 = qm[base + 1u]; let a2 = qm[base + 2u];
    var bits: u32;
    var scale: f32;
    if (off < 32u) { bits = (a0 >> 16u) | (a1 << 16u); scale = unpack2x16float(a0).x; }
    else           { bits = a2;                        scale = unpack2x16float(a1).y; }
    let bo = off & 31u;
    let m = vec4<u32>(
        (((bits >> bo)        & 1u) ^ 1u) << 31u,
        (((bits >> (bo + 1u)) & 1u) ^ 1u) << 31u,
        (((bits >> (bo + 2u)) & 1u) ^ 1u) << 31u,
        (((bits >> (bo + 3u)) & 1u) ^ 1u) << 31u);
    let sv = vec4<f32>(scale, scale, scale, scale);
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(sv) ^ m);
}

@compute @workgroup_size(16, 16)
fn q1_mul_mm(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = pm.cols4 * 4u;
    let np = pm.cols4 / 16u;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) { acc[i][j] = 0.0; }
    }
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let m = t / 4u;
            let k4 = t % 4u;
            var xv = vec4<f32>(0.0);
            if (m0 + m < pm.nb && (k0 / 4u) + k4 < pm.cols4) {
                let xi = (m0 + m) * cols + k0 + k4 * 4u;
                xv = vec4<f32>(xsm[xi], xsm[xi + 1u], xsm[xi + 2u], xsm[xi + 3u]);
            }
            let dst = m * 16u + k4 * 4u;
            mm_at[dst] = xv.x; mm_at[dst + 1u] = xv.y; mm_at[dst + 2u] = xv.z; mm_at[dst + 3u] = xv.w;
        }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let n = t / 4u;
            let k4 = t % 4u;
            var wv = vec4<f32>(0.0);
            if (n0 + n < pm.rows && (k0 / 4u) + k4 < pm.cols4) {
                wv = q1_w4(n0 + n, k0 + k4 * 4u, np);
            }
            let dst = n * 16u + k4 * 4u;
            mm_wt[dst] = wv.x; mm_wt[dst + 1u] = wv.y; mm_wt[dst + 2u] = wv.z; mm_wt[dst + 3u] = wv.w;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            var av: array<f32, 4>;
            var wv: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                av[i] = mm_at[(lid.y * 4u + i) * 16u + k];
                wv[i] = mm_wt[(lid.x * 4u + i) * 16u + k];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) { acc[i][j] = acc[i][j] + av[i] * wv[j]; }
            }
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + lid.y * 4u + i;
        if (m >= pm.nb) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let n = n0 + lid.x * 4u + j;
            if (n < pm.rows) { ym[m * pm.rows + n] = acc[i][j]; }
        }
    }
}

// ── Element-wise kernels of the MoE block (silu·mul·col, axpy, zeroing) ──
struct N1 { n: u32, f: u32, _b: u32, _c: u32 };

@group(0) @binding(0) var<storage, read>       sg   : array<f32>;
@group(0) @binding(1) var<storage, read>       su   : array<f32>;
@group(0) @binding(2) var<storage, read>       scol : array<f32>;
@group(0) @binding(3) var<storage, read_write> sact : array<f32>;
@group(0) @binding(4) var<uniform>             snp  : N1;
@compute @workgroup_size(256)
fn silu_mul_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= snp.n) { return; }
    let gv = sg[i];
    var v = (gv / (1.0 + exp(-gv))) * su[i];
    if (snp.f == 1u) { v = v * scol[i]; }
    sact[i] = v;
}

struct AxpyP { w: f32, n: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       ad : array<f32>;
@group(0) @binding(1) var<storage, read_write> ay : array<f32>;
@group(0) @binding(2) var<uniform>             ap : AxpyP;
@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ap.n) { return; }
    ay[i] = ay[i] + ap.w * ad[i];
}

// Qwen3.5 output gate: attn_out *= sigmoid(gate), element-wise over nh·hd.
@group(0) @binding(0) var<storage, read>       gm_g : array<f32>;
@group(0) @binding(1) var<storage, read_write> gm_o : array<f32>;
@group(0) @binding(2) var<uniform>             gm_p : N1;
@compute @workgroup_size(256)
fn gate_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gm_p.n) { return; }
    gm_o[i] = gm_o[i] * (1.0 / (1.0 + exp(-gm_g[i])));
}

@group(0) @binding(0) var<storage, read_write> zy  : array<f32>;
@group(0) @binding(1) var<uniform>             znp : N1;
@compute @workgroup_size(256)
fn fill_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < znp.n) { zy[i] = 0.0; }
}

// Plain f32 matvec (for small unquantized projections like GDN in_proj_a/b):
// y[o] = Σ_i W[o,i]·x[i]. One workgroup per output row.
struct F32P { cols: u32, rows: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       f32w : array<f32>;
@group(0) @binding(1) var<storage, read>       f32x : array<f32>;
@group(0) @binding(2) var<storage, read_write> f32y : array<f32>;
@group(0) @binding(3) var<uniform>             f32p : F32P;
var<workgroup> f32part: array<f32, 64>;
@compute @workgroup_size(64)
fn f32_matvec(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    if (row >= f32p.rows) { return; }
    let base = row * f32p.cols;
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= f32p.cols) { break; }
        acc = acc + f32w[base + i] * f32x[i];
        i = i + 64u;
    }
    f32part[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { f32part[lid] = f32part[lid] + f32part[lid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid == 0u) { f32y[row] = f32part[0]; }
}

// RMSNorm of one row (WGSL twin of Metal rmsnorm_k): o = x·rsqrt(mean(x²)+eps)·w',
// w' = w or (1+w) for gemma. One workgroup, 256-thread tree reduction — the
// building block that keeps the token graph's hidden resident across the norm.
struct RmsP { n: u32, gemma: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       rn_x : array<f32>;
@group(0) @binding(1) var<storage, read>       rn_w : array<f32>;
@group(0) @binding(2) var<storage, read_write> rn_o : array<f32>;
@group(0) @binding(3) var<uniform>             rn_p : RmsP;
var<workgroup> rn_part: array<f32, 256>;
@compute @workgroup_size(256)
fn rmsnorm(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = rn_p.n;
    var acc = 0.0;
    var i = tid;
    loop {
        if (i >= n) { break; }
        let v = rn_x[i];
        acc = acc + v * v;
        i = i + 256u;
    }
    rn_part[tid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { rn_part[tid] = rn_part[tid] + rn_part[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(rn_part[0] / f32(n) + rn_p.eps);
    i = tid;
    loop {
        if (i >= n) { break; }
        var wv = rn_w[i];
        if (rn_p.gemma == 1u) { wv = 1.0 + wv; }
        rn_o[i] = rn_x[i] * inv * wv;
        i = i + 256u;
    }
}

// GDN depthwise causal conv + SiLU over the ring buffer of the last kk-1
// positions plus the current qkv, then shift the ring (drop oldest, append
// current). One thread per conv channel. WGSL twin of the Metal gdn_conv.
struct GcP { cdim: u32, kk: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       gc_qkv  : array<f32>;
@group(0) @binding(1) var<storage, read>       gc_taps : array<f32>;
@group(0) @binding(2) var<storage, read_write> gc_ring : array<f32>;
@group(0) @binding(3) var<storage, read_write> gc_cq   : array<f32>;
@group(0) @binding(4) var<uniform>             gc_p    : GcP;
@compute @workgroup_size(256)
fn gdn_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let cdim = gc_p.cdim;
    if (c >= cdim) { return; }
    let kk = gc_p.kk;
    let tb = c * kk;
    var acc = gc_qkv[c] * gc_taps[tb + kk - 1u];
    for (var j = 0u; j + 1u < kk; j = j + 1u) {
        acc = acc + gc_ring[j * cdim + c] * gc_taps[tb + j];
    }
    gc_cq[c] = acc / (1.0 + exp(-acc));
    // ring shift (columns are independent per thread c)
    for (var j = 0u; j + 2u < kk; j = j + 1u) {
        gc_ring[j * cdim + c] = gc_ring[(j + 1u) * cdim + c];
    }
    if (kk > 1u) {
        gc_ring[(kk - 2u) * cdim + c] = gc_qkv[c];
    }
}

// ── GDN (gated DeltaNet / linear attention) decode step ──────────────────
// One workgroup per v-head. From the conv output cq it l2-norms q/k, forms the
// decay g and gate β, runs the delta-rule state recurrence S ← g·S + kf⊗β(v −
// kfᵀS) with o = qfᵀS, then the gated RMSNorm o·norm·silu(z). S ([nv,dk,dv])
// persists across tokens (device state buffer). WGSL twin of the Metal GDN
// state-update kernel; dk,dv ≤ 256.
struct GdnP { nv: u32, dk: u32, dv: u32, kd: u32, rep: u32, cdim: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       gd_cq   : array<f32>;
@group(0) @binding(1) var<storage, read>       gd_z    : array<f32>;
@group(0) @binding(2) var<storage, read>       gd_a    : array<f32>;
@group(0) @binding(3) var<storage, read>       gd_b    : array<f32>;
@group(0) @binding(4) var<storage, read>       gd_alog : array<f32>;
@group(0) @binding(5) var<storage, read>       gd_dtb  : array<f32>;
@group(0) @binding(6) var<storage, read>       gd_norm : array<f32>;
@group(0) @binding(7) var<storage, read_write> gd_S    : array<f32>;
@group(0) @binding(8) var<storage, read_write> gd_o    : array<f32>;
@group(0) @binding(9) var<uniform>             gd_p    : GdnP;
var<workgroup> gd_kf: array<f32, 256>;
var<workgroup> gd_qf: array<f32, 256>;
var<workgroup> gd_ov: array<f32, 256>;
var<workgroup> gd_red: array<f32, 256>;
fn gd_softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}
fn gd_reduce(t: u32) -> f32 {
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { gd_red[t] = gd_red[t] + gd_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return gd_red[0];
}
@compute @workgroup_size(256)
fn gdn_step(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let t = lid.x;
    if (h >= gd_p.nv) { return; }
    let dk = gd_p.dk;
    let dv = gd_p.dv;
    let ko = h / gd_p.rep;
    let qs = ko * dk;
    let ks = gd_p.kd + ko * dk;
    // l2-norm of q then k over dk
    gd_red[t] = select(0.0, gd_cq[qs + t] * gd_cq[qs + t], t < dk);
    workgroupBarrier();
    let nq = gd_reduce(t);
    workgroupBarrier();
    gd_red[t] = select(0.0, gd_cq[ks + t] * gd_cq[ks + t], t < dk);
    workgroupBarrier();
    let nkn = gd_reduce(t);
    workgroupBarrier();
    let invq = 1.0 / (sqrt(nq + 1e-6) * sqrt(f32(dk)));
    let invk = 1.0 / sqrt(nkn + 1e-6);
    if (t < dk) {
        gd_qf[t] = gd_cq[qs + t] * invq;
        gd_kf[t] = gd_cq[ks + t] * invk;
    }
    workgroupBarrier();
    let g = exp(-exp(gd_alog[h]) * gd_softplus(gd_a[h] + gd_dtb[h]));
    let beta = 1.0 / (1.0 + exp(-gd_b[h]));
    let sbase = h * dk * dv;
    if (t < dv) {
        let dj = t;
        let vt = gd_cq[2u * gd_p.kd + h * dv + dj];
        var kv = 0.0;
        for (var di = 0u; di < dk; di = di + 1u) { kv = kv + gd_S[sbase + di * dv + dj] * gd_kf[di]; }
        let delta = (vt - g * kv) * beta;
        var o = 0.0;
        for (var di = 0u; di < dk; di = di + 1u) {
            let idx = sbase + di * dv + dj;
            let cell = g * gd_S[idx] + gd_kf[di] * delta;
            gd_S[idx] = cell;
            o = o + gd_qf[di] * cell;
        }
        gd_ov[dj] = o;
    }
    workgroupBarrier();
    // gated RMSNorm over dv
    gd_red[t] = select(0.0, gd_ov[t] * gd_ov[t], t < dv);
    workgroupBarrier();
    let ss = gd_reduce(t);
    workgroupBarrier();
    let inv = 1.0 / sqrt(ss / f32(dv) + gd_p.eps);
    if (t < dv) {
        let zz = gd_z[h * dv + t];
        gd_o[h * dv + t] = gd_ov[t] * inv * gd_norm[t] * (zz / (1.0 + exp(-zz)));
    }
}

// Fused residual-add + RMSNorm (WGSL twin of Metal add_rmsnorm_rows): h += d
// in place, then o = rms(h)·w. Collapses an axpy + an rmsnorm dispatch into
// one — cuts two launches per layer off the token graph.
struct ArP { n: u32, gemma: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> ar_h : array<f32>;
@group(0) @binding(1) var<storage, read>       ar_d : array<f32>;
@group(0) @binding(2) var<storage, read>       ar_w : array<f32>;
@group(0) @binding(3) var<storage, read_write> ar_o : array<f32>;
@group(0) @binding(4) var<uniform>             ar_p : ArP;
var<workgroup> ar_part: array<f32, 256>;
@compute @workgroup_size(256)
fn add_rmsnorm(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = ar_p.n;
    var acc = 0.0;
    var i = tid;
    loop {
        if (i >= n) { break; }
        let v = ar_h[i] + ar_d[i];
        ar_h[i] = v;
        acc = acc + v * v;
        i = i + 256u;
    }
    ar_part[tid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { ar_part[tid] = ar_part[tid] + ar_part[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(ar_part[0] / f32(n) + ar_p.eps);
    i = tid;
    loop {
        if (i >= n) { break; }
        var wv = ar_w[i];
        if (ar_p.gemma == 1u) { wv = 1.0 + wv; }
        ar_o[i] = ar_h[i] * inv * wv;
        i = i + 256u;
    }
}

// Batched RMSNorm for prefill: one workgroup per row (wid.x), row r reads/writes
// rn_x[r*n..] → rn_o[r*n..]; the weight rn_w[n] is shared. K prompt positions
// norm in one dispatch (twin of `rmsnorm`, strided by row).
@compute @workgroup_size(256)
fn rmsnorm_b(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = rn_p.n;
    let base = wid.x * n;
    var acc = 0.0;
    var i = tid;
    loop { if (i >= n) { break; } let v = rn_x[base + i]; acc = acc + v * v; i = i + 256u; }
    rn_part[tid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop { if (stride == 0u) { break; } if (tid < stride) { rn_part[tid] = rn_part[tid] + rn_part[tid + stride]; } workgroupBarrier(); stride = stride / 2u; }
    let inv = inverseSqrt(rn_part[0] / f32(n) + rn_p.eps);
    i = tid;
    loop { if (i >= n) { break; } var wv = rn_w[i]; if (rn_p.gemma == 1u) { wv = 1.0 + wv; } rn_o[base + i] = rn_x[base + i] * inv * wv; i = i + 256u; }
}

// Batched fused residual-add + RMSNorm (one workgroup per row): ar_h[r] += ar_d[r]
// in place, then ar_o[r] = rms(ar_h[r])·w. Prefill twin of `add_rmsnorm`.
@compute @workgroup_size(256)
fn add_rmsnorm_b(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = ar_p.n;
    let base = wid.x * n;
    var acc = 0.0;
    var i = tid;
    loop { if (i >= n) { break; } let v = ar_h[base + i] + ar_d[base + i]; ar_h[base + i] = v; acc = acc + v * v; i = i + 256u; }
    ar_part[tid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop { if (stride == 0u) { break; } if (tid < stride) { ar_part[tid] = ar_part[tid] + ar_part[tid + stride]; } workgroupBarrier(); stride = stride / 2u; }
    let inv = inverseSqrt(ar_part[0] / f32(n) + ar_p.eps);
    i = tid;
    loop { if (i >= n) { break; } var wv = ar_w[i]; if (ar_p.gemma == 1u) { wv = 1.0 + wv; } ar_o[base + i] = ar_h[base + i] * inv * wv; i = i + 256u; }
}

// RoPE + optional qk-norm + gate-split, one 32-thread workgroup per head
// (WGSL twin of Metal attn_rope_qkn; the qk-norm sum-of-squares reduces in
// workgroup memory — no subgroup ops, portable). Heads [0,nh)=Q (2·hd each
// when gated: q||gate), [nh,nh+nkv)=K. flags: 1=gate 2=qnorm 4=knorm 8=gemma.
struct RqP { nh: u32, nkv: u32, hd: u32, rd: u32, pos: u32, flags: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       rq_qraw : array<f32>;
@group(0) @binding(1) var<storage, read_write> rq_k    : array<f32>;
@group(0) @binding(2) var<storage, read_write> rq_qout : array<f32>;
@group(0) @binding(3) var<storage, read_write> rq_gout : array<f32>;
@group(0) @binding(4) var<storage, read>       rq_qnw  : array<f32>;
@group(0) @binding(5) var<storage, read>       rq_knw  : array<f32>;
@group(0) @binding(6) var<storage, read>       rq_invf : array<f32>;
@group(0) @binding(7) var<uniform>             rq_p    : RqP;
var<workgroup> rq_red: array<f32, 32>;
var<workgroup> rq_head: array<f32, 256>;
@compute @workgroup_size(32)
fn attn_rope_qkn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;
    let nh = rq_p.nh;
    let hd = rq_p.hd;
    if (head >= nh + rq_p.nkv) { return; }
    let isq = head < nh;
    let gate = (rq_p.flags & 1u) != 0u;
    let src_base = select((head - nh) * hd, head * select(1u, 2u, gate) * hd, isq);
    let nt = (hd + 31u) / 32u;  // ≤ 8 for head_dim ≤ 256 (Qwen3.5 uses 256)
    var xv: array<f32, 8>;
    var ss = 0.0;
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        var val = 0.0;
        if (d < hd) { val = select(rq_k[src_base + d], rq_qraw[src_base + d], isq); }
        xv[t] = val;
        ss = ss + val * val;
    }
    rq_red[lane] = ss;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) { rq_red[lane] = rq_red[lane] + rq_red[lane + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let normed = select((rq_p.flags & 4u) != 0u, (rq_p.flags & 2u) != 0u, isq);
    if (normed) {
        let inv = 1.0 / sqrt(rq_red[0] / f32(hd) + rq_p.eps);
        let gemma = (rq_p.flags & 8u) != 0u;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) {
                var wd = select(rq_knw[d], rq_qnw[d], isq);
                if (gemma) { wd = 1.0 + wd; }
                xv[t] = xv[t] * inv * wd;
            }
        }
    }
    // RoPE over the first rd dims, pairing dim i with dim i+hlf. Staged through
    // workgroup memory because the pair partner lands on a DIFFERENT lane when
    // hlf isn't a multiple of 32 (partial RoPE — Qwen3.5 rotates head_dim/4, so
    // hlf can be 16). The old register tiling (xv[t+toff], toff=hlf/32) silently
    // did nothing for hlf<32; here each lane ropes the pairs i=lane,lane+32,…
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) { rq_head[d] = xv[t]; }
    }
    workgroupBarrier();
    let hlf = rq_p.rd / 2u;
    var ri = lane;
    loop {
        if (ri >= hlf) { break; }
        let angle = f32(rq_p.pos) * rq_invf[ri];
        let cc = cos(angle);
        let sfac = sin(angle);
        let x0 = rq_head[ri];
        let x1 = rq_head[ri + hlf];
        rq_head[ri] = x0 * cc - x1 * sfac;
        rq_head[ri + hlf] = x0 * sfac + x1 * cc;
        ri = ri + 32u;
    }
    workgroupBarrier();
    let dst_base = select((head - nh) * hd, head * hd, isq);
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) {
            if (isq) { rq_qout[dst_base + d] = rq_head[d]; } else { rq_k[dst_base + d] = rq_head[d]; }
        }
    }
    if (isq && gate) {
        let gbase = head * 2u * hd + hd;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) { rq_gout[head * hd + d] = rq_qraw[gbase + d]; }
        }
    }
}

// Append this position's K/V rows into the device cache mirror ([nkv,cap,hd]
// each) at row `stored`. WGSL twin of Metal kv_append.
struct KvP { nkv: u32, hd: u32, cap: u32, stored: u32 };
@group(0) @binding(0) var<storage, read>       kv_k  : array<f32>;
@group(0) @binding(1) var<storage, read>       kv_v  : array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_kb : array<f32>;
@group(0) @binding(3) var<storage, read_write> kv_vb : array<f32>;
@group(0) @binding(4) var<uniform>             kv_p  : KvP;
@compute @workgroup_size(256)
fn kv_append(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= kv_p.nkv * kv_p.hd) { return; }
    let h = i / kv_p.hd;
    let d = i % kv_p.hd;
    let dst = (h * kv_p.cap + kv_p.stored) * kv_p.hd + d;
    kv_kb[dst] = kv_k[i];
    kv_vb[dst] = kv_v[i];
}

// Grouped decode attention, one 32-thread workgroup per Q-head. Dims sliced
// across lanes (dim d in lane d%32, slot d/32); online softmax over the n
// cached positions with the per-position q·k dot reduced in workgroup memory
// (portable — no subgroup ops). WGSL twin of Metal gqa_attend (output only;
// Born-importance is handled on the CPU side when eviction is active).
struct AtP { nh: u32, hpk: u32, hd: u32, cap: u32, n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       at_q : array<f32>;
@group(0) @binding(1) var<storage, read>       at_k : array<f32>;
@group(0) @binding(2) var<storage, read>       at_v : array<f32>;
@group(0) @binding(3) var<storage, read_write> at_o : array<f32>;
@group(0) @binding(4) var<uniform>             at_p : AtP;
// Flash-decoding: split the n cached positions across the 32 lanes. Each lane
// runs an INDEPENDENT online softmax over positions lane, lane+32, … with NO
// barrier in the loop (the old kernel barriered twice PER position — O(ctx)
// serial chain), then a 5-step 32-way log-sum-exp merge. Serial steps: n → n/32.
var<workgroup> at_acc: array<f32, 8224>; // [lane*257 + d], stride 257 dodges 32-bank conflicts, hd ≤ 256 (Qwen3.5=256)
var<workgroup> at_m: array<f32, 32>;
var<workgroup> at_l: array<f32, 32>;
@compute @workgroup_size(32)
fn gqa_attend(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let lane = lid.x;
    if (h >= at_p.nh) { return; }
    let hd = at_p.hd;
    let n = at_p.n;
    let kbase = (h / at_p.hpk) * at_p.cap * hd;
    let qbase = h * hd;
    let scale = 1.0 / sqrt(f32(hd));
    let base = lane * 257u;
    for (var d = 0u; d < hd; d = d + 1u) { at_acc[base + d] = 0.0; }
    var m = -1e30;
    var l = 0.0;
    var p = lane;
    loop {
        if (p >= n) { break; }
        let krow = kbase + p * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) { dot = dot + at_q[qbase + d] * at_k[krow + d]; }
        dot = dot * scale;
        let mp = max(m, dot);
        let f = exp(m - mp);
        let w = exp(dot - mp);
        l = l * f + w;
        for (var d = 0u; d < hd; d = d + 1u) { at_acc[base + d] = at_acc[base + d] * f + w * at_v[krow + d]; }
        m = mp;
        p = p + 32u;
    }
    at_m[lane] = m;
    at_l[lane] = l;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            let o = lane + stride;
            let m1 = at_m[lane];
            let m2 = at_m[o];
            let mm = max(m1, m2);
            let f1 = exp(m1 - mm);
            let f2 = exp(m2 - mm);
            at_l[lane] = at_l[lane] * f1 + at_l[o] * f2;
            let bo = o * 257u;
            for (var d = 0u; d < hd; d = d + 1u) {
                at_acc[base + d] = at_acc[base + d] * f1 + at_acc[bo + d] * f2;
            }
            at_m[lane] = mm;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let invl = select(0.0, 1.0 / at_l[0], at_l[0] > 0.0);
    for (var d = lane; d < hd; d = d + 32u) {
        at_o[h * hd + d] = at_acc[d] * invl;
    }
}

// q1t (ternary base-3) + q4_block matvec — reuse the q1 bindings (q1w/q1x/q1y/
// q1p) and its 4-slot layout. Weights arrive as array<u32>, so bytes come out
// with shift+mask (q1t_byte). q1p fields are reinterpreted: np=gpr, _p0=cols.
var<workgroup> partial_q1t: array<f32, 64>;
fn q1t_byte(off: u32) -> u32 {
    return (q1w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}
const Q1T_LUT: array<u32, 243> = array<u32, 243>(
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
    680u, 681u, 682u
);

@compute @workgroup_size(64)
fn q1t_matvec(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let base_len = rows * gpr * 9u;
    let ent_off = base_len + (rows + 1u) * 4u;
    var row = wid.x;
    loop {
        if (row >= rows) { break; }
        var acc = 0.0;
        var g = lid;
        loop {
            if (g >= gpr) { break; }
            let toff = (row * gpr + g) * 9u;
            let sc16 = q1t_byte(toff) | (q1t_byte(toff + 1u) << 8u);
            let scale = unpack2x16float(sc16).x;
            let codes = toff + 2u;
            let xb = g * 32u;
            var gsum = 0.0;
            for (var k = 0u; k < 32u; k = k + 1u) {
                let b = q1t_byte(codes + k / 5u);
                let p = Q1T_LUT[b];
                let code = (p >> ((k % 5u) * 2u)) & 3u;
                let sgn = select(0.0, 1.0, code == 1u) - select(0.0, 1.0, code == 2u);
                gsum = gsum + sgn * q1x[xb + k];
            }
            acc = acc + scale * gsum;
            g = g + 64u;
        }
        partial_q1t[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial_q1t[lid] = partial_q1t[lid] + partial_q1t[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) {
            var corr = 0.0;
            let rp0 = base_len + row * 4u;
            let c0 = q1t_byte(rp0) | (q1t_byte(rp0 + 1u) << 8u) | (q1t_byte(rp0 + 2u) << 16u) | (q1t_byte(rp0 + 3u) << 24u);
            let rp1 = base_len + (row + 1u) * 4u;
            let c1 = q1t_byte(rp1) | (q1t_byte(rp1 + 1u) << 8u) | (q1t_byte(rp1 + 2u) << 16u) | (q1t_byte(rp1 + 3u) << 24u);
            for (var p = c0; p < c1; p = p + 1u) {
                let e = ent_off + p * 4u;
                let col = q1t_byte(e) | (q1t_byte(e + 1u) << 8u);
                let val16 = q1t_byte(e + 2u) | (q1t_byte(e + 3u) << 8u);
                corr = corr + unpack2x16float(val16).x * q1x[col];
            }
            q1y[row] = partial_q1t[0] + corr;
        }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// 8 nibbles from one u32 word dot 8 activations (fully unrolled FMA chain).
fn q4b_dot8(w: u32, xi: u32) -> f32 {
    return (f32(w & 0xFu) - 8.0) * q1x[xi]
         + (f32((w >> 4u) & 0xFu) - 8.0) * q1x[xi + 1u]
         + (f32((w >> 8u) & 0xFu) - 8.0) * q1x[xi + 2u]
         + (f32((w >> 12u) & 0xFu) - 8.0) * q1x[xi + 3u]
         + (f32((w >> 16u) & 0xFu) - 8.0) * q1x[xi + 4u]
         + (f32((w >> 20u) & 0xFu) - 8.0) * q1x[xi + 5u]
         + (f32((w >> 24u) & 0xFu) - 8.0) * q1x[xi + 6u]
         + (f32((w >> 28u) & 0xFu) - 8.0) * q1x[xi + 7u];
}

@compute @workgroup_size(64)
fn q4b_matvec(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let scales_off = rows * gpr * 16u;
    var row = wid.x;
    loop {
        if (row >= rows) { break; }
        var acc = 0.0;
        var g = lid;
        loop {
            if (g >= gpr) { break; }
            let gi = row * gpr + g;
            // Scale: one u32 read instead of two byte reads.
            let sc_byte = scales_off + gi * 2u;
            let sc16 = (q1w[sc_byte >> 2u] >> ((sc_byte & 3u) * 8u)) & 0xFFFFu;
            let scale = unpack2x16float(sc16).x;
            // 4 u32 reads = 16 bytes = 32 weights (4× fewer array accesses
            // than the per-byte path, ~40% fewer ALU per group).
            let pk4 = gi * 4u;
            let xb = g * 32u;
            let gsum = q4b_dot8(q1w[pk4], xb)
                     + q4b_dot8(q1w[pk4 + 1u], xb + 8u)
                     + q4b_dot8(q1w[pk4 + 2u], xb + 16u)
                     + q4b_dot8(q1w[pk4 + 3u], xb + 24u);
            acc = acc + scale * gsum;
            g = g + 64u;
        }
        partial_q1t[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial_q1t[lid] = partial_q1t[lid] + partial_q1t[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) { q1y[row] = partial_q1t[0]; }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// Fused SiLU(gate)·up → Q4Block down-proj matvec: eliminates the standalone
// silu dispatch (saves one inter-pass pipeline flush per layer).
@group(0) @binding(0) var<storage, read>       sd_w : array<u32>;
@group(0) @binding(1) var<storage, read>       sd_gate : array<f32>;
@group(0) @binding(2) var<storage, read>       sd_up : array<f32>;
@group(0) @binding(3) var<storage, read_write> sd_y : array<f32>;
@group(0) @binding(4) var<uniform>             sd_p : Q1Params;

var<workgroup> partial_sd: array<f32, 64>;

fn sd_dot8(w: u32, xi: u32) -> f32 {
    let g0 = sd_gate[xi];     let g1 = sd_gate[xi + 1u];
    let g2 = sd_gate[xi + 2u]; let g3 = sd_gate[xi + 3u];
    let g4 = sd_gate[xi + 4u]; let g5 = sd_gate[xi + 5u];
    let g6 = sd_gate[xi + 6u]; let g7 = sd_gate[xi + 7u];
    return (f32(w & 0xFu) - 8.0) * (g0 / (1.0 + exp(-g0)) * sd_up[xi])
         + (f32((w >> 4u) & 0xFu) - 8.0) * (g1 / (1.0 + exp(-g1)) * sd_up[xi + 1u])
         + (f32((w >> 8u) & 0xFu) - 8.0) * (g2 / (1.0 + exp(-g2)) * sd_up[xi + 2u])
         + (f32((w >> 12u) & 0xFu) - 8.0) * (g3 / (1.0 + exp(-g3)) * sd_up[xi + 3u])
         + (f32((w >> 16u) & 0xFu) - 8.0) * (g4 / (1.0 + exp(-g4)) * sd_up[xi + 4u])
         + (f32((w >> 20u) & 0xFu) - 8.0) * (g5 / (1.0 + exp(-g5)) * sd_up[xi + 5u])
         + (f32((w >> 24u) & 0xFu) - 8.0) * (g6 / (1.0 + exp(-g6)) * sd_up[xi + 6u])
         + (f32((w >> 28u) & 0xFu) - 8.0) * (g7 / (1.0 + exp(-g7)) * sd_up[xi + 7u]);
}

@compute @workgroup_size(64)
fn silu_down_matvec(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(num_workgroups) nwg: vec3<u32>,
                    @builtin(local_invocation_index) lid: u32) {
    let gpr = sd_p.np;
    let rows = sd_p.rows;
    let scales_off = rows * gpr * 16u;
    var row = wid.x;
    loop {
        if (row >= rows) { break; }
        var acc = 0.0;
        var g = lid;
        loop {
            if (g >= gpr) { break; }
            let gi = row * gpr + g;
            let sc_byte = scales_off + gi * 2u;
            let sc16 = (sd_w[sc_byte >> 2u] >> ((sc_byte & 3u) * 8u)) & 0xFFFFu;
            let scale = unpack2x16float(sc16).x;
            let pk4 = gi * 4u;
            let xb = g * 32u;
            let gsum = sd_dot8(sd_w[pk4], xb)
                     + sd_dot8(sd_w[pk4 + 1u], xb + 8u)
                     + sd_dot8(sd_w[pk4 + 2u], xb + 16u)
                     + sd_dot8(sd_w[pk4 + 3u], xb + 24u);
            acc = acc + scale * gsum;
            g = g + 64u;
        }
        partial_sd[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial_sd[lid] = partial_sd[lid] + partial_sd[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) { sd_y[row] = partial_sd[0]; }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// q1t register-blocked GEMM (prefill) — the WGSL cousin of the Metal q1t_mul_mm
// and structurally identical to q8_mul_mm here; only the W staging decodes
// base-3 ternary × per-group f16 scale (no row_scale; scale folds into the
// staged weight). Own 4-slot bindings. The overlay is a second pass.
struct Q1tMmP { cols4: u32, rows: u32, nb: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       qmm : array<u32>;
@group(0) @binding(1) var<storage, read>       xmm : array<f32>;
@group(0) @binding(2) var<storage, read_write> ymm : array<f32>;
@group(0) @binding(3) var<uniform>             pmm : Q1tMmP;

fn qmm_byte(off: u32) -> u32 {
    return (qmm[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}
var<workgroup> q1t_at: array<f32, 64 * 16>;
var<workgroup> q1t_wt: array<f32, 64 * 16>;

@compute @workgroup_size(16, 16)
fn q1t_mul_mm(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = pmm.cols4 * 4u;
    let gpr = cols >> 5u;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) { acc[i][j] = 0.0; }
    }
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let m = t / 4u;
            let k4 = t % 4u;
            var xv = vec4<f32>(0.0);
            let col0 = k0 + k4 * 4u;
            if (m0 + m < pmm.nb && col0 < cols) {
                let xi = (m0 + m) * cols + col0;
                xv = vec4<f32>(xmm[xi], xmm[xi + 1u], xmm[xi + 2u], xmm[xi + 3u]);
            }
            let dst = m * 16u + k4 * 4u;
            q1t_at[dst] = xv.x; q1t_at[dst + 1u] = xv.y;
            q1t_at[dst + 2u] = xv.z; q1t_at[dst + 3u] = xv.w;
        }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let n = t / 4u;
            let k4 = t % 4u;
            var wv = vec4<f32>(0.0);
            let col0 = k0 + k4 * 4u;
            if (n0 + n < pmm.rows && col0 < cols) {
                let g = col0 >> 5u;
                let toff = ((n0 + n) * gpr + g) * 9u;
                let sc16 = qmm_byte(toff) | (qmm_byte(toff + 1u) << 8u);
                let scale = unpack2x16float(sc16).x;
                let codes = toff + 2u;
                for (var d = 0u; d < 4u; d = d + 1u) {
                    let p = (col0 + d) - g * 32u;
                    let b = qmm_byte(codes + p / 5u);
                    let code = (Q1T_LUT[b] >> ((p % 5u) * 2u)) & 3u;
                    var sgn = 0.0;
                    if (code == 1u) { sgn = 1.0; } else if (code == 2u) { sgn = -1.0; }
                    wv[d] = sgn * scale;
                }
            }
            let dst = n * 16u + k4 * 4u;
            q1t_wt[dst] = wv.x; q1t_wt[dst + 1u] = wv.y;
            q1t_wt[dst + 2u] = wv.z; q1t_wt[dst + 3u] = wv.w;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            var av: array<f32, 4>;
            var wv: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                av[i] = q1t_at[(lid.y * 4u + i) * 16u + k];
                wv[i] = q1t_wt[(lid.x * 4u + i) * 16u + k];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i][j] = acc[i][j] + av[i] * wv[j];
                }
            }
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + lid.y * 4u + i;
        if (m >= pmm.nb) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let n = n0 + lid.x * 4u + j;
            if (n < pmm.rows) { ymm[m * pmm.rows + n] = acc[i][j]; }
        }
    }
}

@compute @workgroup_size(64)
fn q1t_overlay_mm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= pmm.rows) { return; }
    let cols = pmm.cols4 * 4u;
    let gpr = cols >> 5u;
    let base_len = pmm.rows * gpr * 9u;
    let ent = base_len + (pmm.rows + 1u) * 4u;
    let rp0 = base_len + row * 4u;
    let c0 = qmm_byte(rp0) | (qmm_byte(rp0 + 1u) << 8u) | (qmm_byte(rp0 + 2u) << 16u) | (qmm_byte(rp0 + 3u) << 24u);
    let rp1 = base_len + (row + 1u) * 4u;
    let c1 = qmm_byte(rp1) | (qmm_byte(rp1 + 1u) << 8u) | (qmm_byte(rp1 + 2u) << 16u) | (qmm_byte(rp1 + 3u) << 24u);
    for (var p = c0; p < c1; p = p + 1u) {
        let e = ent + p * 4u;
        let col = qmm_byte(e) | (qmm_byte(e + 1u) << 8u);
        let val = unpack2x16float(qmm_byte(e + 2u) | (qmm_byte(e + 3u) << 8u)).x;
        for (var bi = 0u; bi < pmm.nb; bi = bi + 1u) {
            ymm[bi * pmm.rows + row] = ymm[bi * pmm.rows + row] + val * xmm[bi * cols + col];
        }
    }
}
"#;

struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matvec: wgpu::ComputePipeline,
    matmat: wgpu::ComputePipeline,
    mul_mm: wgpu::ComputePipeline,
    q1_mm: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    axpy: wgpu::ComputePipeline,
    gate_mul: wgpu::ComputePipeline,
    zero: wgpu::ComputePipeline,
    q1: wgpu::ComputePipeline,
    q1t: wgpu::ComputePipeline,
    q4b: wgpu::ComputePipeline,
    silu_down: wgpu::ComputePipeline,
    q1t_mm: wgpu::ComputePipeline,
    q1t_ovmm: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    add_rmsnorm: wgpu::ComputePipeline,
    rmsnorm_b: wgpu::ComputePipeline,
    add_rmsnorm_b: wgpu::ComputePipeline,
    attn_rope: wgpu::ComputePipeline,
    kv_append: wgpu::ComputePipeline,
    gqa_attend: wgpu::ComputePipeline,
    gdn_step: wgpu::ComputePipeline,
    gdn_conv: wgpu::ComputePipeline,
    f32_matvec: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    layout_mm: wgpu::BindGroupLayout,
    layout_mmm: wgpu::BindGroupLayout,
    layout_q1mm: wgpu::BindGroupLayout,
    layout_silu: wgpu::BindGroupLayout,
    layout_axpy: wgpu::BindGroupLayout,
    layout_gate_mul: wgpu::BindGroupLayout,
    layout_zero: wgpu::BindGroupLayout,
    layout_q1: wgpu::BindGroupLayout,
    layout_rmsnorm: wgpu::BindGroupLayout,
    layout_add_rmsnorm: wgpu::BindGroupLayout,
    layout_rmsnorm_b: wgpu::BindGroupLayout,
    layout_add_rmsnorm_b: wgpu::BindGroupLayout,
    layout_attn_rope: wgpu::BindGroupLayout,
    layout_kv: wgpu::BindGroupLayout,
    layout_attend: wgpu::BindGroupLayout,
    layout_gdn: wgpu::BindGroupLayout,
    layout_gdn_conv: wgpu::BindGroupLayout,
    layout_f32: wgpu::BindGroupLayout,
    layout_silu_down: wgpu::BindGroupLayout,
    /// Discrete card (PCIe VRAM) vs UMA — thresholds and budgets differ.
    discrete: bool,
    /// Weight-residency budget in bytes (CMF_GPU_VRAM_MB override). On a
    /// 24 GB card holding a 35 GB model, the first-touched tensors (=
    /// the first layers, decode touches them in order) stay resident and
    /// the rest honestly fall back to CPU — ngl-style offload without an
    /// explicit layer list, and no OOM.
    vram_budget: u64,
    /// Bytes currently resident in `weight_bufs`.
    resident: std::sync::atomic::AtomicU64,
    /// Pooled per-op scratch (grow-only): xs upload, y output, uniform
    /// params, readback staging. Every op used to CREATE all four (plus
    /// a bind group) and map_async-poll a fresh staging buffer — pure
    /// allocator traffic on the hot path. The lock is held across the
    /// whole op (encode → submit → poll): ops already serialize on the
    /// single queue.
    scratch: Mutex<Scratch>,
    /// Resident quant weights in VRAM — the WHOLE tensor is loaded once
    /// (key (base_ptr, idx)); ranges/batches address it by offset.
    weight_bufs: Mutex<HashMap<(usize, usize), wgpu::Buffer>>,
    /// row_scale buffer per (idx, row0) — small, cached.
    rs_bufs: Mutex<HashMap<(usize, usize), wgpu::Buffer>>,
    /// Device K/V cache mirror per (kv_id, layer) for the token graph:
    /// [nkv, cap, hd] each, persists across decode tokens. `synced` counts
    /// the positions already resident (prefill sync + graph appends).
    attn_kv: Mutex<HashMap<(u64, usize), KvMirror>>,
    /// GDN recurrent state per (kv_id, layer): (conv ring, S), persists across
    /// decode tokens (created zeroed on first touch).
    gdn_state: Mutex<HashMap<(u64, usize), (wgpu::Buffer, wgpu::Buffer)>>,
    /// Immutable [rows,cols,…] uniforms cached by content — the ~800 matvec
    /// param buffers per token are token-invariant, so uploading them once
    /// keeps them off the per-token encode critical path.
    uniforms: Mutex<HashMap<[u32; 4], wgpu::Buffer>>,
    /// Immutable norm/small weight buffers cached by (data ptr, len) — the
    /// ~200 per-layer norm uploads per token are token-invariant. Sentinel
    /// key (0, n) holds shared zero buffers. Assumes stable weight pointers
    /// (mmap), same as `weight_bufs`.
    const_bufs: Mutex<HashMap<(usize, usize), wgpu::Buffer>>,
    /// Pooled graph scratch: eliminates per-token buffer allocations in the
    /// whole-token graph path (the dominant decode cost on Vulkan/DX12).
    graph_scratch: Mutex<GraphScratch>,
}

struct KvMirror {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    synced: usize,
}

#[derive(Default)]
struct Scratch {
    xs: Option<(wgpu::Buffer, u64)>,
    y: Option<(wgpu::Buffer, u64)>,
    stage: Option<(wgpu::Buffer, u64)>,
    params: Option<wgpu::Buffer>,
}

impl Scratch {
    /// Grow-only slot: reuse when big enough, else recreate.
    fn ensure(
        dev: &wgpu::Device,
        slot: &mut Option<(wgpu::Buffer, u64)>,
        need: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> wgpu::Buffer {
        match slot {
            Some((b, cap)) if *cap >= need => b.clone(),
            _ => {
                crate::gpu::probe_note_cold();
                let cap = need.next_power_of_two().max(4096);
                let b = dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: cap,
                    usage,
                    mapped_at_creation: false,
                });
                *slot = Some((b.clone(), cap));
                b
            }
        }
    }
}

/// Pooled scratch for the whole-token graph path. Grow-only: each slot is
/// allocated once (or grown) and reused across tokens — eliminates the ~20
/// Vulkan buffer allocations per token that dominated decode latency.
#[derive(Default)]
struct GraphScratch {
    h: Option<(wgpu::Buffer, u64)>,
    n1: Option<(wgpu::Buffer, u64)>,
    qraw: Option<(wgpu::Buffer, u64)>,
    kb: Option<(wgpu::Buffer, u64)>,
    vb: Option<(wgpu::Buffer, u64)>,
    qout: Option<(wgpu::Buffer, u64)>,
    gout: Option<(wgpu::Buffer, u64)>,
    attn: Option<(wgpu::Buffer, u64)>,
    ob: Option<(wgpu::Buffer, u64)>,
    gbuf: Option<(wgpu::Buffer, u64)>,
    ubuf: Option<(wgpu::Buffer, u64)>,
    abuf: Option<(wgpu::Buffer, u64)>,
    // GDN intermediates
    qkv_b: Option<(wgpu::Buffer, u64)>,
    cq_b: Option<(wgpu::Buffer, u64)>,
    z_b: Option<(wgpu::Buffer, u64)>,
    a_b: Option<(wgpu::Buffer, u64)>,
    b_b: Option<(wgpu::Buffer, u64)>,
    gdo_b: Option<(wgpu::Buffer, u64)>,
    // Logits output + readback staging
    logits: Option<(wgpu::Buffer, u64)>,
    stage: Option<(wgpu::Buffer, u64)>,
    // Position-dependent uniforms (fixed size, write_buffer each token)
    kv_u: Option<wgpu::Buffer>,    // 16 bytes: [nkv, hd, cap, position]
    at_u: Option<wgpu::Buffer>,    // 32 bytes: [nh, nh/nkv, hd, cap, pos+1, 0, 0, 0]
    rope_u: Option<wgpu::Buffer>,  // 32 bytes: [nh, nkv, hd, rd, pos, flags, eps, 0]
}

impl GraphScratch {
    fn ensure(
        dev: &wgpu::Device,
        slot: &mut Option<(wgpu::Buffer, u64)>,
        need: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> wgpu::Buffer {
        match slot {
            Some((b, cap)) if *cap >= need => b.clone(),
            _ => {
                let cap = need.next_power_of_two().max(256);
                let b = dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: cap,
                    usage,
                    mapped_at_creation: false,
                });
                *slot = Some((b.clone(), cap));
                b
            }
        }
    }
    /// Pooled uniform buffer of `size` bytes (created once, write_buffer'd each token).
    fn ensure_uniform(dev: &wgpu::Device, slot: &mut Option<wgpu::Buffer>, size: u64) -> wgpu::Buffer {
        match slot {
            Some(b) => b.clone(),
            None => {
                let b = dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("g-unif"),
                    size,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                *slot = Some(b.clone());
                b
            }
        }
    }
}

static CTX: OnceLock<Option<Ctx>> = OnceLock::new();

/// Whether the wgpu path is selected by env (the facade asks before `enabled()`):
/// `CMF_GPU=wgpu` — always; `CMF_GPU=1` (≠0) — only on non-macOS, where
/// there is no native Metal (on macOS `=1` goes to Metal).
pub fn selected() -> bool {
    match std::env::var("CMF_GPU") {
        Ok(v) if v == "wgpu" => true,
        Ok(v) if v != "0" => !cfg!(target_os = "macos"),
        _ => crate::pipeline::GLOBAL_USE_GPU.load(std::sync::atomic::Ordering::Relaxed) && !cfg!(target_os = "macos"),
    }
}

fn ctx() -> Option<&'static Ctx> {
    CTX.get_or_init(|| {
        if !selected() {
            return None;
        }
        match init() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!("wgpu init failed — CPU fallback: {e}");
                None
            }
        }
    })
    .as_ref()
}

fn init() -> Result<Ctx, String> {
    // Backend selection is automatic (wgpu picks the platform's best:
    // DX12 on Windows, Vulkan on Linux, Metal on macOS), but the
    // standard WGPU_BACKEND env (vulkan|dx12|metal|gl) forces one.
    let backends = std::env::var("WGPU_BACKEND")
        .ok()
        .map(|v| match v.to_lowercase().as_str() {
            "vulkan" | "vk" => wgpu::Backends::VULKAN,
            "dx12" | "d3d12" => wgpu::Backends::DX12,
            "metal" | "mtl" => wgpu::Backends::METAL,
            "gl" | "gles" => wgpu::Backends::GL,
            _ => wgpu::Backends::all(),
        })
        .unwrap_or(wgpu::Backends::all());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no adapter: {e}"))?;

    // Take the card's maximum limits — large tensors (lm_head ≈ 254 MB
    // int8) require a raised storage buffer; a discrete card handles GB.
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cortiq-wgpu"),
        required_limits: limits,
        ..Default::default()
    }))
    .map_err(|e| format!("request_device: {e}"))?;

    let info = adapter.get_info();
    let discrete = info.device_type == wgpu::DeviceType::DiscreteGpu;
    let vram_budget = std::env::var("CMF_GPU_VRAM_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb * 1024 * 1024)
        .unwrap_or(if discrete {
            // Conservative default for unknown cards; 4090-class users
            // should set CMF_GPU_VRAM_MB=20000.
            8 * 1024 * 1024 * 1024
        } else {
            u64::MAX // UMA: the OS pages shared memory
        });
    tracing::info!(
        "wgpu GPU path: on ({} / {:?}, {}, weight budget {})",
        info.name,
        info.backend,
        if discrete { "discrete" } else { "uma" },
        if vram_budget == u64::MAX { "unlimited".to_string() } else { format!("{} MB", vram_budget / 1024 / 1024) },
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("q8"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    // Auto layout: the bind group layout is inferred from the shader.
    let pipe = |ep: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: None, // auto: layout is inferred from the shader
            module: &module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let matvec = pipe("q8_matvec");
    let matmat = pipe("q8_matmat");
    let mul_mm = pipe("q8_mul_mm");
    let q1_mm = pipe("q1_mul_mm");
    let silu = pipe("silu_mul_pre");
    let axpy = pipe("axpy");
    let gate_mul = pipe("gate_mul");
    let zero = pipe("fill_zero");
    let q1 = pipe("q1_matvec");
    let q1t = pipe("q1t_matvec");
    let q4b = pipe("q4b_matvec");
    let silu_down = pipe("silu_down_matvec");
    let q1t_mm = pipe("q1t_mul_mm");
    let q1t_ovmm = pipe("q1t_overlay_mm");
    let rmsnorm = pipe("rmsnorm");
    let add_rmsnorm = pipe("add_rmsnorm");
    let rmsnorm_b = pipe("rmsnorm_b");
    let add_rmsnorm_b = pipe("add_rmsnorm_b");
    let attn_rope = pipe("attn_rope_qkn");
    let kv_append = pipe("kv_append");
    let gqa_attend = pipe("gqa_attend");
    let gdn_step = pipe("gdn_step");
    let gdn_conv = pipe("gdn_conv");
    let f32_matvec = pipe("f32_matvec");
    let layout = matvec.get_bind_group_layout(0);
    let layout_q1 = q1.get_bind_group_layout(0);
    let layout_rmsnorm = rmsnorm.get_bind_group_layout(0);
    let layout_add_rmsnorm = add_rmsnorm.get_bind_group_layout(0);
    let layout_rmsnorm_b = rmsnorm_b.get_bind_group_layout(0);
    let layout_add_rmsnorm_b = add_rmsnorm_b.get_bind_group_layout(0);
    let layout_attn_rope = attn_rope.get_bind_group_layout(0);
    let layout_kv = kv_append.get_bind_group_layout(0);
    let layout_attend = gqa_attend.get_bind_group_layout(0);
    let layout_gdn = gdn_step.get_bind_group_layout(0);
    let layout_gdn_conv = gdn_conv.get_bind_group_layout(0);
    let layout_f32 = f32_matvec.get_bind_group_layout(0);
    let layout_silu_down = silu_down.get_bind_group_layout(0);
    let layout_mm = matmat.get_bind_group_layout(0);
    let layout_mmm = mul_mm.get_bind_group_layout(0);
    let layout_q1mm = q1_mm.get_bind_group_layout(0);
    let layout_silu = silu.get_bind_group_layout(0);
    let layout_axpy = axpy.get_bind_group_layout(0);
    let layout_gate_mul = gate_mul.get_bind_group_layout(0);
    let layout_zero = zero.get_bind_group_layout(0);

    Ok(Ctx {
        device,
        queue,
        matvec,
        matmat,
        mul_mm,
        q1_mm,
        silu,
        axpy,
        gate_mul,
        zero,
        q1,
        q1t,
        q4b,
        silu_down,
        q1t_mm,
        q1t_ovmm,
        rmsnorm,
        add_rmsnorm,
        rmsnorm_b,
        add_rmsnorm_b,
        attn_rope,
        kv_append,
        gqa_attend,
        gdn_step,
        gdn_conv,
        f32_matvec,
        layout,
        layout_mm,
        layout_mmm,
        layout_q1mm,
        layout_silu,
        layout_axpy,
        layout_gate_mul,
        layout_zero,
        layout_q1,
        layout_rmsnorm,
        layout_add_rmsnorm,
        layout_rmsnorm_b,
        layout_add_rmsnorm_b,
        layout_attn_rope,
        layout_kv,
        layout_attend,
        layout_gdn,
        layout_gdn_conv,
        layout_f32,
        layout_silu_down,
        discrete,
        vram_budget,
        resident: std::sync::atomic::AtomicU64::new(0),
        scratch: Mutex::new(Scratch::default()),
        weight_bufs: Mutex::new(HashMap::new()),
        uniforms: Mutex::new(HashMap::new()),
        const_bufs: Mutex::new(HashMap::new()),
        rs_bufs: Mutex::new(HashMap::new()),
        attn_kv: Mutex::new(HashMap::new()),
        gdn_state: Mutex::new(HashMap::new()),
        graph_scratch: Mutex::new(GraphScratch::default()),
    })
}

/// Is the active adapter a discrete card? (facade: threshold policy)
pub fn is_discrete() -> bool {
    ctx().map(|c| c.discrete).unwrap_or(false)
}

/// Resident quant weights of the WHOLE tensor in VRAM (loaded once per
/// (file, idx)), guarded by the VRAM budget: once the budget is spent,
/// new tensors return None and their ops run on the CPU. Decode touches
/// layers in order, so the resident set is deterministically the first
/// layers — ngl-style offload without configuration.
fn weight_buffer(c: &Ctx, key: (usize, usize), full_quant: &[u8]) -> Option<wgpu::Buffer> {
    use std::sync::atomic::Ordering;
    let mut map = c.weight_bufs.lock().unwrap();
    if let Some(b) = map.get(&key) {
        return Some(b.clone());
    }
    let len = full_quant.len() as u64;
    if c.resident.load(Ordering::Relaxed) + len > c.vram_budget {
        return None; // budget spent — this tensor stays on the CPU
    }
    crate::gpu::probe_note_cold(); // first touch = upload, not a steady sample
    // DEVICE-LOCAL residency: create_buffer_init maps at creation → the buffer
    // lands in a HOST_VISIBLE heap and every matvec streams its weights over
    // PCIe (~25 GB/s) every token. A plain create_buffer + staged write_buffer
    // lets the allocator pick DEVICE_LOCAL VRAM (~1 TB/s on a 4090). This is
    // THE discrete-GPU decode fix; on UMA it's a wash.
    let buf = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("q1-weights"),
        size: len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue.write_buffer(&buf, 0, full_quant);
    c.resident.fetch_add(len, Ordering::Relaxed);
    map.insert(key, buf.clone());
    Some(buf)
}

/// GPU enabled and initialized?
pub fn enabled() -> bool {
    ctx().is_some()
}

/// Probe helper: true — tensor `idx`'s weights are already resident;
/// false — not yet (with `may_upload`, the upload happens NOW within the
/// budget, without a dispatch, so the next touch is warm) or the tensor
/// can't be resolved.
pub fn q8_resident_or_upload(model: &Arc<CmfModel>, idx: usize, may_upload: bool) -> bool {
    let Some(c) = ctx() else { return false };
    let entry = &model.tensors[idx];
    let rows_total = entry.shape.first().copied().unwrap_or(0);
    let cols = entry.shape.get(1).copied().unwrap_or(0);
    if rows_total == 0 || cols == 0 {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    if abs + rows_total * cols > bytes.len() {
        return false;
    }
    let key = (bytes.as_ptr() as usize, idx);
    if c.weight_bufs.lock().unwrap().contains_key(&key) {
        return true;
    }
    if may_upload {
        let _ = weight_buffer(c, key, &bytes[abs..abs + rows_total * cols]);
    }
    false
}

/// q8_row/q8_2f matvec on the GPU, rows [row0, row0+rows). `xs` are already
/// prescaled activations. false = could not (the caller falls back to CPU).
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
    if cols % 4 != 0 || rows == 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    let rows_total = entry.shape.first().copied().unwrap_or(0);
    if rows_total < row0 + rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false; // neighboring shard — different mapping; CPU
    };
    let bytes = model.primary_bytes();
    if abs + rows_total * cols > bytes.len() {
        return false;
    }
    let full_quant = &bytes[abs..abs + rows_total * cols];
    let key = (bytes.as_ptr() as usize, idx);
    dispatch_matvec(c, Some(key), full_quant, row0, row_scale, xs, rows, cols, out)
}

/// matvec kernel: resident weights of the WHOLE tensor + row0 offset, rs, xs,
/// dispatch, readback. `weight_key = None` — no cache (test).
#[allow(clippy::too_many_arguments)]
fn dispatch_matvec(
    c: &Ctx,
    weight_key: Option<(usize, usize)>,
    full_quant: &[u8],
    row0: usize,
    row_scale: &[f32],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    if row_scale.len() < rows || xs.len() < cols || full_quant.len() < (row0 + rows) * cols {
        return false;
    }
    let q_buf = match weight_key {
        Some(k) => match weight_buffer(c, k, full_quant) {
            Some(b) => b,
            None => return false, // over VRAM budget — honest CPU path
        },
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q8-weights"),
            contents: full_quant,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    let make_rs = || {
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q8-rs"),
            contents: bytemuck::cast_slice(&row_scale[..rows]),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let rs_buf = match weight_key {
        Some((base, idx)) => c
            .rs_bufs
            .lock()
            .unwrap()
            .entry((base ^ idx.wrapping_mul(1_000_003), row0))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                make_rs()
            })
            .clone(),
        None => make_rs(),
    };

    // Pooled scratch for the whole op (encode → submit → poll).
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q8-xs",
    );
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
    let y_size = (rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q8-y",
    );
    let params = [(cols / 4) as u32, rows as u32, (row0 * cols / 4) as u32, 0u32];
    let p_buf = match &sc.params {
        Some(b) => b.clone(),
        None => {
            let b = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q8-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(b.clone());
            b
        }
    };
    c.queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q8-stage",
    );

    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q8-bg"),
        layout: &c.layout,
        entries: &[
            bind_buf(0, &q_buf),
            bind_buf(1, &xs_buf),
            bind_buf(2, &rs_buf),
            bind_buf(3, &y_buf),
            bind_buf(4, &p_buf),
        ],
    });

    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q8") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q8"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.matvec);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1); // grid-stride over rows
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..rows]);
    drop(sc);
    ok
}

/// q1t (base+overlay) / q4_block matvec on wgpu — raw f32 x, scales embedded.
/// The kernel decodes bytes out of the u32 weight buffer; params carry
/// (gpr, rows, cols). Weights resident under the shared VRAM budget.
pub fn q1t_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    q1t_like(model, idx, xs, rows, cols, out, false)
}

/// q4_block matvec on wgpu (nibbles + trailing scales, no overlay).
pub fn q4b_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    q1t_like(model, idx, xs, rows, cols, out, true)
}

fn q1t_like(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    q4: bool,
) -> bool {
    let Some(c) = ctx() else { return false };
    let gpr = cols / 32;
    if rows == 0 || cols % 32 != 0 || xs.len() < cols || out.len() < rows {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else { return false };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    // sanity: the base must at least fit (q1t base 9 B/group, q4b 18 B/group).
    let min_base = if q4 { rows * gpr * 18 } else { rows * gpr * 9 };
    if plen < min_base || abs + plen > bytes.len() {
        return false;
    }
    let pipeline = if q4 { &c.q4b } else { &c.q1t };
    dispatch_q1t(c, pipeline, Some((bytes.as_ptr() as usize, idx)), &bytes[abs..abs + plen], xs, rows, cols, out)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_q1t(
    c: &Ctx,
    pipeline: &wgpu::ComputePipeline,
    weight_key: Option<(usize, usize)>,
    payload: &[u8],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let gpr = cols / 32;
    let q_buf = match weight_key {
        Some(k) => match weight_buffer(c, k, payload) {
            Some(b) => b,
            None => return false,
        },
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q1t-weights"),
            contents: payload,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q1t-xs",
    );
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
    let y_size = (rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q1t-y",
    );
    let params = [gpr as u32, rows as u32, cols as u32, 0u32];
    let p_buf = match &sc.params {
        Some(b) => b.clone(),
        None => {
            let b = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q1t-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(b.clone());
            b
        }
    };
    c.queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q1t-stage",
    );
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q1t-bg"),
        // Must be THIS pipeline's layout (wgpu treats each pipeline's layout as
        // distinct even when structurally identical to q1's).
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            bind_buf(0, &q_buf),
            bind_buf(1, &xs_buf),
            bind_buf(2, &y_buf),
            bind_buf(3, &p_buf),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q1t") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q1t"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..rows]);
    drop(sc);
    ok
}

/// q1 matvec: raw f32 activations, tile-embedded scales (no rs buffer).
/// Weights resident under the same VRAM budget as q8; false = CPU path.
pub fn q1_matvec(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let gpr = cols / 32;
    if rows == 0 || cols % 32 != 0 || gpr % 2 != 0 || xs.len() < cols || out.len() < rows {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = rows * gpr * 6;
    if abs + plen > bytes.len() {
        return false;
    }
    dispatch_q1(c, Some((bytes.as_ptr() as usize, idx)), &bytes[abs..abs + plen], xs, rows, cols, out)
}

/// GPU RMSNorm of one row — the token-graph building block that keeps the
/// hidden state resident across the norm→matvec boundary. One workgroup,
/// direct buffers (no residency cache). Returns false without a GPU context.
pub fn rmsnorm_row(x: &[f32], w: &[f32], out: &mut [f32], gemma: bool, eps: f32) -> bool {
    let Some(c) = ctx() else { return false };
    let n = x.len();
    if n == 0 || w.len() < n || out.len() < n {
        return false;
    }
    let x_b = storage_bytes(c, bytemuck::cast_slice(x));
    let w_b = storage_bytes(c, bytemuck::cast_slice(&w[..n]));
    let o_b = rw_f32(c, n, true);
    let p_buf = uniform_u32x4(c, [n as u32, gemma as u32, eps.to_bits(), 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rms-bg"),
        layout: &c.layout_rmsnorm,
        entries: &[
            bind_buf(0, &x_b),
            bind_buf(1, &w_b),
            bind_buf(2, &o_b),
            bind_buf(3, &p_buf),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rms") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.rmsnorm);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let size = (n * 4) as u64;
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "rms-stage",
    );
    let ok = readback(c, enc, &o_b, &stage, size, &mut out[..n]);
    drop(sc);
    ok
}

/// GPU RoPE + qk-norm + gate-split building block (bring-up / parity). One
/// workgroup per head; writes qout[nh·hd], k in place[nkv·hd], gout[nh·hd].
/// qnw/knw must be hd-long (dummy ok if the norm flag is off), invf rd/2-long.
#[allow(clippy::too_many_arguments)]
pub fn attn_rope_qkn_gpu(
    qraw: &[f32],
    k_in: &[f32],
    qnw: &[f32],
    knw: &[f32],
    invf: &[f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    pos: usize,
    flags: u32,
    eps: f32,
    qout: &mut [f32],
    k_out: &mut [f32],
    gout: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let qraw_b = storage_bytes(c, bytemuck::cast_slice(qraw));
    let k_b = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rq-k"),
        contents: bytemuck::cast_slice(&k_in[..nkv * hd]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let qout_b = rw_f32(c, nh * hd, true);
    let gout_b = rw_f32(c, nh * hd, true);
    let qnw_b = storage_bytes(c, bytemuck::cast_slice(qnw));
    let knw_b = storage_bytes(c, bytemuck::cast_slice(knw));
    let invf_b = storage_bytes(c, bytemuck::cast_slice(invf));
    let p_data = [
        nh as u32, nkv as u32, hd as u32, rd as u32, pos as u32, flags, eps.to_bits(), 0u32,
    ];
    let p_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rq-p"),
        contents: bytemuck::cast_slice(&p_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rq-bg"),
        layout: &c.layout_attn_rope,
        entries: &[
            bind_buf(0, &qraw_b),
            bind_buf(1, &k_b),
            bind_buf(2, &qout_b),
            bind_buf(3, &gout_b),
            bind_buf(4, &qnw_b),
            bind_buf(5, &knw_b),
            bind_buf(6, &invf_b),
            bind_buf(7, &p_buf),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rq") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rq"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.attn_rope);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((nh + nkv) as u32, 1, 1);
    }
    let mk_stage = |n: usize| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rq-stage"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let sq = mk_stage(nh * hd);
    let sk = mk_stage(nkv * hd);
    let sgt = mk_stage(nh * hd);
    enc.copy_buffer_to_buffer(&qout_b, 0, &sq, 0, (nh * hd * 4) as u64);
    enc.copy_buffer_to_buffer(&k_b, 0, &sk, 0, (nkv * hd * 4) as u64);
    enc.copy_buffer_to_buffer(&gout_b, 0, &sgt, 0, (nh * hd * 4) as u64);
    c.queue.submit(Some(enc.finish()));
    for s in [&sq, &sk, &sgt] {
        s.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    }
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    let (Ok(dq), Ok(dk), Ok(dg)) = (
        sq.slice(..).get_mapped_range(),
        sk.slice(..).get_mapped_range(),
        sgt.slice(..).get_mapped_range(),
    ) else {
        return false;
    };
    qout[..nh * hd].copy_from_slice(bytemuck::cast_slice(&dq[..nh * hd * 4]));
    k_out[..nkv * hd].copy_from_slice(bytemuck::cast_slice(&dk[..nkv * hd * 4]));
    gout[..nh * hd].copy_from_slice(bytemuck::cast_slice(&dg[..nh * hd * 4]));
    true
}

/// GPU grouped decode attention (bring-up / parity). K/V caches are laid out
/// [nkv, cap, hd]; attends q[nh·hd] over the first `n` rows, writes out[nh·hd].
#[allow(clippy::too_many_arguments)]
pub fn gqa_attend_gpu(
    q: &[f32],
    kcache: &[f32],
    vcache: &[f32],
    nh: usize,
    hpk: usize,
    hd: usize,
    cap: usize,
    n: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let q_b = storage_bytes(c, bytemuck::cast_slice(q));
    let k_b = storage_bytes(c, bytemuck::cast_slice(kcache));
    let v_b = storage_bytes(c, bytemuck::cast_slice(vcache));
    let o_b = rw_f32(c, nh * hd, true);
    let p_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("at-p"),
        contents: bytemuck::cast_slice(&[
            nh as u32, hpk as u32, hd as u32, cap as u32, n as u32, 0u32, 0u32, 0u32,
        ]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("at-bg"),
        layout: &c.layout_attend,
        entries: &[
            bind_buf(0, &q_b),
            bind_buf(1, &k_b),
            bind_buf(2, &v_b),
            bind_buf(3, &o_b),
            bind_buf(4, &p_buf),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("at") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("at"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.gqa_attend);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(nh as u32, 1, 1);
    }
    let size = (nh * hd * 4) as u64;
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "at-stage",
    );
    let ok = readback(c, enc, &o_b, &stage, size, &mut out[..nh * hd]);
    drop(sc);
    ok
}

/// Resident q1 weight for a model tensor (cached in VRAM by (ptr, idx)).
/// Returns (buffer, rows, cols). None on budget/shape refusal.
fn q1_weight(c: &Ctx, model: &Arc<CmfModel>, idx: usize) -> Option<(wgpu::Buffer, usize, usize)> {
    let entry = model.tensors.get(idx)?;
    let rows = *entry.shape.first()? as usize;
    let cols = *entry.shape.get(1)? as usize;
    if cols % 32 != 0 {
        return None;
    }
    let abs = model.entry_abs_offset(entry)?;
    let bytes = model.primary_bytes();
    let plen = rows * (cols / 32) * 6;
    if abs + plen > bytes.len() {
        return None;
    }
    let buf = weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen])?;
    Some((buf, rows, cols))
}

/// Production drop-in for the attention sub-block on the token graph: takes
/// the already-normed hidden and returns the O-projection output (pre-
/// residual) — exactly where `qwen_attention` slots in. QKV/O weights are
/// resident (VRAM cache), the K/V cache is a persistent device mirror keyed
/// by (kv_id, layer) that is synced once from the CPU cache (prefill) then
/// appended to each token. Everything runs in ONE command encoder; only the
/// attention output reads back. false = refusal (caller keeps the CPU path).
#[allow(clippy::too_many_arguments)]
pub fn attn_dropin_gpu(
    model: &Arc<CmfModel>,
    kv_id: u64,
    layer: usize,
    normed: &[f32],
    wq_idx: usize,
    wk_idx: usize,
    wv_idx: usize,
    wo_idx: usize,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    invf: &[f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    pos: usize,
    cap: usize,
    gemma: bool,
    eps: f32,
    cpu_k: &[Vec<f32>],
    cpu_v: &[Vec<f32>],
    attn_out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if pos >= cap {
        return false;
    }
    let (wq, rq, cq) = q1_weight(c, model, wq_idx).unwrap_or((c.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }), 0, 0));
    if rq != nh * hd || cq != hidden {
        return false; // gated arch (e.g. output_gate doubles rows) → CPU path
    }
    let Some((wk, _, _)) = q1_weight(c, model, wk_idx) else { return false };
    let Some((wv, _, _)) = q1_weight(c, model, wv_idx) else { return false };
    let Some((wo, ro, co)) = q1_weight(c, model, wo_idx) else { return false };
    if ro != hidden || co != nh * hd {
        return false;
    }
    // Device K/V mirror (persist across tokens).
    let mut kvm = c.attn_kv.lock().unwrap();
    let entry = kvm.entry((kv_id, layer)).or_insert_with(|| {
        let sz = (nkv * cap * hd * 4) as u64;
        let mk = || c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv-mirror"),
            size: sz,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        KvMirror { k: mk(), v: mk(), synced: 0 }
    });
    // Sync prefill history 0..pos from the CPU cache (once).
    if entry.synced < pos {
        for h in 0..nkv {
            let src_k = &cpu_k[h];
            let src_v = &cpu_v[h];
            let from = entry.synced;
            let take = pos.min(src_k.len() / hd);
            if take > from {
                let off = ((h * cap + from) * hd * 4) as u64;
                c.queue.write_buffer(&entry.k, off, bytemuck::cast_slice(&src_k[from * hd..take * hd]));
                c.queue.write_buffer(&entry.v, off, bytemuck::cast_slice(&src_v[from * hd..take * hd]));
            }
        }
        entry.synced = pos;
    }
    let kbuf = entry.k.clone();
    let vbuf = entry.v.clone();
    drop(kvm);

    let stor = |data: &[u8]| c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: data, usage: wgpu::BufferUsages::STORAGE });
    let dummy = vec![0f32; hd];
    let qnw_b = stor(bytemuck::cast_slice(q_norm.unwrap_or(&dummy)));
    let knw_b = stor(bytemuck::cast_slice(k_norm.unwrap_or(&dummy)));
    let invf_b = stor(bytemuck::cast_slice(invf));
    let normed_b = stor(bytemuck::cast_slice(&normed[..hidden]));
    let qraw_b = rw_f32(c, nh * hd, false);
    let k_b = rw_f32(c, nkv * hd, false);
    let v_b = rw_f32(c, nkv * hd, false);
    let qout_b = rw_f32(c, nh * hd, false);
    let gout_b = rw_f32(c, nh * hd, false);
    let attn_b = rw_f32(c, nh * hd, false);
    let o_b = rw_f32(c, hidden, true);
    let flags = if q_norm.is_some() { 2u32 } else { 0 } | if k_norm.is_some() { 4 } else { 0 } | if gemma { 8 } else { 0 };
    let unif = |d: &[u32]| c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(d), usage: wgpu::BufferUsages::UNIFORM });
    let bg = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let e: Vec<_> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &e })
    };
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("attn-dropin") });
    let go = |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, b: &wgpu::BindGroup, g: u32| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(p);
        pass.set_bind_group(0, b, &[]);
        pass.dispatch_workgroups(g, 1, 1);
    };
    encode_matvec_q1(c, &mut enc, &wq, &normed_b, &qraw_b, nh * hd, hidden);
    encode_matvec_q1(c, &mut enc, &wk, &normed_b, &k_b, nkv * hd, hidden);
    encode_matvec_q1(c, &mut enc, &wv, &normed_b, &v_b, nkv * hd, hidden);
    let rq_p = unif(&[nh as u32, nkv as u32, hd as u32, rd as u32, pos as u32, flags, eps.to_bits(), 0]);
    go(&mut enc, &c.attn_rope, &bg(&c.layout_attn_rope, &[&qraw_b, &k_b, &qout_b, &gout_b, &qnw_b, &knw_b, &invf_b, &rq_p]), (nh + nkv) as u32);
    let kv_p = unif(&[nkv as u32, hd as u32, cap as u32, pos as u32]);
    go(&mut enc, &c.kv_append, &bg(&c.layout_kv, &[&k_b, &v_b, &kbuf, &vbuf, &kv_p]), ((nkv * hd) as u32).div_ceil(256));
    let at_p = unif(&[nh as u32, (nh / nkv) as u32, hd as u32, cap as u32, (pos + 1) as u32, 0, 0, 0]);
    go(&mut enc, &c.gqa_attend, &bg(&c.layout_attend, &[&qout_b, &kbuf, &vbuf, &attn_b, &at_p]), nh as u32);
    encode_matvec_q1(c, &mut enc, &wo, &attn_b, &o_b, hidden, nh * hd);
    let size = (hidden * 4) as u64;
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(&c.device, &mut sc.stage, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "dropin-stage");
    let ok = readback(c, enc, &o_b, &stage, size, &mut attn_out[..hidden]);
    drop(sc);
    if ok {
        c.attn_kv.lock().unwrap().get_mut(&(kv_id, layer)).map(|m| m.synced = pos + 1);
    }
    ok
}

/// WHOLE-TOKEN decode graph: the entire layer stack (rmsnorm → attention →
/// residual → rmsnorm → SiLU-FFN → residual, every layer) encoded into ONE
/// command buffer with the hidden RESIDENT on the GPU — only the final hidden
/// reads back (one submit/token instead of ~2 per layer). This is what lifts
/// the submit-latency wall. Returns false on any refusal (caller keeps CPU).
#[allow(clippy::too_many_arguments)]
pub fn forward_token_graph(
    model: &Arc<CmfModel>,
    kv_id: u64,
    layers: &[crate::gpu::GraphLayer],
    invf: &[f32],
    h: &mut [f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    inter: usize,
    position: usize,
    cap: usize,
    gemma: bool,
    eps: f32,
    // Optional final-norm + lm_head fold: (weight, rows). When Some and the
    // weight resolves, the graph rides the final RMSNorm and lm_head in the
    // same submit and reads back `logits` (rows) instead of the hidden — one
    // fewer op + sync per token, and the lm_head stays on-device.
    lm_head: Option<(&crate::gpu::GraphW, usize)>,
    final_norm: &[f32],
    logits: &mut Vec<f32>,
    loop_norm_at: &[usize],
) -> bool {
    let Some(c) = ctx() else { return false };
    if position >= cap {
        return false;
    }
    let t_start = std::time::Instant::now();
    // A resolved matvec weight: the device-local buffer, (q8 only) its row
    // scales, and the codec kind (0=q8_row 1=q1 2=q4_tiled 3=q1t).
    struct GMat {
        buf: wgpu::Buffer,
        rs: Option<wgpu::Buffer>,
        kind: u8,
    }
    enum LAttn {
        Full { wq: GMat, wk: GMat, wv: GMat, wo: GMat },
        Gdn { qkv: GMat, z: GMat, a: GMat, b: GMat, out: GMat, nv: usize, nk: usize, dk: usize, dv: usize, kk: usize, cdim: usize },
    }
    struct LW {
        attn: LAttn,
        gate: GMat,
        up: GMat,
        down: GMat,
    }
    // Resolve + cache every layer's weights (q8_row or q1) up front; bail (CPU)
    // on any refusal (budget/shape/dtype).
    let resolve = |gw: &crate::gpu::GraphW, rows: usize, cols: usize| -> Option<GMat> {
        match gw.kind {
            0 => {
                // q8_row: weight bytes = rows*cols, plus per-row scales.
                if gw.row_scale.len() < rows {
                    return None;
                }
                let b = tensor_weight(c, model, gw.idx, rows, cols)?; // device-local
                // Row scales are token-invariant — cache by (ptr,rows).
                let key = (gw.row_scale.as_ptr() as usize, rows);
                let mut cb = c.const_bufs.lock().unwrap();
                let rsb = if let Some(x) = cb.get(&key) {
                    x.clone()
                } else {
                    let x = c.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("g-rs"),
                        size: (rows * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    c.queue.write_buffer(&x, 0, bytemuck::cast_slice(&gw.row_scale[..rows]));
                    cb.insert(key, x.clone());
                    x
                };
                Some(GMat { buf: b, rs: Some(rsb), kind: 0 })
            }
            1 => {
                let (b, r, cc) = q1_weight(c, model, gw.idx)?;
                if r != rows || cc != cols {
                    return None;
                }
                Some(GMat { buf: b, rs: None, kind: 1 })
            }
            2 | 3 => {
                // q4_tiled / q1t: the tensor carries its own byte length (tiles
                // + q1t's sparse overlay) — fetch it whole, device-local.
                let entry = model.tensors.get(gw.idx)?;
                if *entry.shape.first()? as usize != rows || *entry.shape.get(1)? as usize != cols {
                    return None;
                }
                let abs = model.entry_abs_offset(entry)?;
                let plen = entry.nbytes as usize;
                let bytes = model.primary_bytes();
                if abs + plen > bytes.len() {
                    return None;
                }
                let b = weight_buffer(c, (bytes.as_ptr() as usize, gw.idx), &bytes[abs..abs + plen])?;
                Some(GMat { buf: b, rs: None, kind: gw.kind })
            }
            4 => {
                // f32 weight (small unquantized projection, e.g. GDN a/b) —
                // token-invariant: cache device-local by (ptr, rows*cols)
                // instead of re-uploading it every token.
                if gw.data.len() < rows * cols {
                    return None;
                }
                let key = (gw.data.as_ptr() as usize, rows * cols);
                let mut cb = c.const_bufs.lock().unwrap();
                let b = if let Some(x) = cb.get(&key) {
                    x.clone()
                } else {
                    let x = c.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("g-f32w"),
                        size: (rows * cols * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    c.queue.write_buffer(&x, 0, bytemuck::cast_slice(&gw.data[..rows * cols]));
                    cb.insert(key, x.clone());
                    x
                };
                Some(GMat { buf: b, rs: None, kind: 4 })
            }
            _ => None,
        }
    };
    let mut lws = Vec::with_capacity(layers.len());
    let mut gdn_dims: Option<(usize, usize, usize, usize, usize, usize)> = None; // nv,nk,dk,dv,kk,cdim
    for l in layers {
        let attn = match &l.attn {
            crate::gpu::GraphAttn::Full { wq, wk, wv, wo, output_gate, .. } => {
                // Gated attention: wq packs q||gate per head → 2·nh·hd rows.
                let qrows = nh * hd * (1 + *output_gate as usize);
                let (Some(wq), Some(wk), Some(wv), Some(wo)) = (
                    resolve(wq, qrows, hidden),
                    resolve(wk, nkv * hd, hidden),
                    resolve(wv, nkv * hd, hidden),
                    resolve(wo, hidden, nh * hd),
                ) else {
                    return false;
                };
                LAttn::Full { wq, wk, wv, wo }
            }
            crate::gpu::GraphAttn::Gdn { qkv, z, a, b, out, nv, nk, dk, dv, kk, .. } => {
                let cdim = 2 * nk * dk + nv * dv;
                gdn_dims = Some((*nv, *nk, *dk, *dv, *kk, cdim));
                let (Some(qkv), Some(z), Some(a), Some(b), Some(out)) = (
                    resolve(qkv, cdim, hidden),
                    resolve(z, nv * dv, hidden),
                    resolve(a, *nv, hidden),
                    resolve(b, *nv, hidden),
                    resolve(out, hidden, nv * dv),
                ) else {
                    return false;
                };
                LAttn::Gdn { qkv, z, a, b, out, nv: *nv, nk: *nk, dk: *dk, dv: *dv, kk: *kk, cdim }
            }
        };
        let (Some(gate), Some(up), Some(down)) = (
            resolve(&l.gate, inter, hidden),
            resolve(&l.up, inter, hidden),
            resolve(&l.down, hidden, inter),
        ) else {
            return false;
        };
        lws.push(LW { attn, gate, up, down });
    }
    // DEVICE-LOCAL + content-cached: create_buffer + write_buffer keeps norm
    // weights in VRAM (not the HOST_VISIBLE heap create_buffer_init forces);
    // caching by (ptr,len) uploads each token-invariant norm buffer once.
    let stor = |data: &[u8]| {
        let key = (data.as_ptr() as usize, data.len());
        let mut cb = c.const_bufs.lock().unwrap();
        if let Some(b) = cb.get(&key) {
            return b.clone();
        }
        let b = c.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: data.len() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        c.queue.write_buffer(&b, 0, data);
        cb.insert(key, b.clone());
        b
    };
    // Shared zero buffer of `n` f32 (sentinel key (0,n)) — for absent q/k-norms
    // and the silu bias slot, so no per-token zero Vec is allocated/uploaded.
    let zeros = |n: usize| -> wgpu::Buffer {
        let key = (0usize, n * 4);
        let mut cb = c.const_bufs.lock().unwrap();
        if let Some(b) = cb.get(&key) {
            return b.clone();
        }
        let b = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("g-zero"), size: (n * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        c.queue.write_buffer(&b, 0, &vec![0u8; n * 4]);
        cb.insert(key, b.clone());
        b
    };
    let unif = |d: &[u32]| c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(d), usage: wgpu::BufferUsages::UNIFORM });
    let bg = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let e: Vec<_> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &e })
    };
    // ── Pooled scratch: all intermediate buffers are reused across tokens ──
    let mut gs = c.graph_scratch.lock().unwrap();
    let st = wgpu::BufferUsages::STORAGE;
    let h_buf = GraphScratch::ensure(&c.device, &mut gs.h, (hidden * 4) as u64, st | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, "g-h");
    c.queue.write_buffer(&h_buf, 0, bytemuck::cast_slice(&h[..hidden]));
    let n1 = GraphScratch::ensure(&c.device, &mut gs.n1, (hidden * 4) as u64, st | wgpu::BufferUsages::COPY_SRC, "g-n1");
    // Gated attention (Qwen3.5) makes wq emit 2·nh·hd (q||gate per head), so the
    // raw-QKV scratch must hold the widened q output for any gated layer.
    let any_gate = layers.iter().any(|l| matches!(&l.attn, crate::gpu::GraphAttn::Full { output_gate: true, .. }));
    let qraw = GraphScratch::ensure(&c.device, &mut gs.qraw, (nh * hd * (1 + any_gate as usize) * 4) as u64, st, "g-qraw");
    let kb = GraphScratch::ensure(&c.device, &mut gs.kb, (nkv * hd * 4) as u64, st, "g-kb");
    let vb = GraphScratch::ensure(&c.device, &mut gs.vb, (nkv * hd * 4) as u64, st, "g-vb");
    let qout = GraphScratch::ensure(&c.device, &mut gs.qout, (nh * hd * 4) as u64, st, "g-qout");
    let gout = GraphScratch::ensure(&c.device, &mut gs.gout, (nh * hd * 4) as u64, st, "g-gout");
    let attn = GraphScratch::ensure(&c.device, &mut gs.attn, (nh * hd * 4) as u64, st, "g-attn");
    let ob = GraphScratch::ensure(&c.device, &mut gs.ob, (hidden * 4) as u64, st, "g-ob");
    let gbuf = GraphScratch::ensure(&c.device, &mut gs.gbuf, (inter * 4) as u64, st, "g-gbuf");
    let ubuf = GraphScratch::ensure(&c.device, &mut gs.ubuf, (inter * 4) as u64, st, "g-ubuf");
    let abuf = GraphScratch::ensure(&c.device, &mut gs.abuf, (inter * 4) as u64, st, "g-abuf");
    let invf_b = stor(bytemuck::cast_slice(invf));
    let dummy_hd = zeros(hd);
    // GDN intermediates (sized to the model's GDN geometry; 1 if no GDN layer).
    let (gnv, _gnk, gdk, gdv, _gkk, gcdim) = gdn_dims.unwrap_or((1, 1, 1, 1, 1, 1));
    let qkv_b = GraphScratch::ensure(&c.device, &mut gs.qkv_b, (gcdim * 4) as u64, st, "g-qkv");
    let cq_b = GraphScratch::ensure(&c.device, &mut gs.cq_b, (gcdim * 4) as u64, st, "g-cq");
    let z_b = GraphScratch::ensure(&c.device, &mut gs.z_b, (gnv * gdv * 4) as u64, st, "g-z");
    let a_b = GraphScratch::ensure(&c.device, &mut gs.a_b, (gnv * 4) as u64, st, "g-a");
    let b_b = GraphScratch::ensure(&c.device, &mut gs.b_b, (gnv * 4) as u64, st, "g-b");
    let gdo_b = GraphScratch::ensure(&c.device, &mut gs.gdo_b, (gnv * gdv * 4) as u64, st, "g-gdo");
    // Sync each Full layer's device K/V mirror from the CPU cache (once);
    // GDN layers carry a persistent (ring, S) recurrent state instead.
    let mut kvbufs: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>> = Vec::with_capacity(layers.len());
    let mut gdnbufs: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>> = Vec::with_capacity(layers.len());
    {
        let mut kvm = c.attn_kv.lock().unwrap();
        let mut gsm = c.gdn_state.lock().unwrap();
        for (li, l) in layers.iter().enumerate() {
            match &l.attn {
                crate::gpu::GraphAttn::Full { cpu_k, cpu_v, .. } => {
                    let e = kvm.entry((kv_id, li)).or_insert_with(|| {
                        let sz = (nkv * cap * hd * 4) as u64;
                        let mk = || c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("kv"), size: sz, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                        KvMirror { k: mk(), v: mk(), synced: 0 }
                    });
                    if e.synced < position {
                        for hh in 0..nkv {
                            let take = position.min(cpu_k[hh].len() / hd);
                            if take > e.synced {
                                let off = ((hh * cap + e.synced) * hd * 4) as u64;
                                c.queue.write_buffer(&e.k, off, bytemuck::cast_slice(&cpu_k[hh][e.synced * hd..take * hd]));
                                c.queue.write_buffer(&e.v, off, bytemuck::cast_slice(&cpu_v[hh][e.synced * hd..take * hd]));
                            }
                        }
                        e.synced = position;
                    }
                    kvbufs.push(Some((e.k.clone(), e.v.clone())));
                    gdnbufs.push(None);
                }
                crate::gpu::GraphAttn::Gdn { .. } => {
                    let e = gsm.entry((kv_id, li)).or_insert_with(|| {
                        let ring_sz = ((_gkk.max(1) - 0) * 0 + (gcdim * (_gkk.max(1).saturating_sub(1))) * 4) as u64;
                        let s_sz = (gnv * gdk * gdv * 4) as u64;
                        let mk = |sz: u64| {
                            let bf = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("gdn-state"), size: sz.max(4), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                            c.queue.write_buffer(&bf, 0, &vec![0u8; sz.max(4) as usize]);
                            bf
                        };
                        (mk(ring_sz), mk(s_sz))
                    });
                    gdnbufs.push(Some((e.0.clone(), e.1.clone())));
                    kvbufs.push(None);
                }
            }
        }
    }
    let prof = std::env::var("CMF_GRAPH_PROF").is_ok();
    // Group mutually-independent projections (that all read the same normed
    // hidden) into ONE compute pass — the GPU can overlap them, cutting the
    // per-pass barrier bubbles that dominate single-token decode. Default on
    // (measured +5-8% token-identical across q1/q8/GDN); CMF_GPU_GROUP=0 off.
    let group = std::env::var("CMF_GPU_GROUP").map(|v| v != "0").unwrap_or(true);
    let t_enc0 = std::time::Instant::now();
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("token-graph") });
    let go = |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, b: &wgpu::BindGroup, g: u32| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(p);
        pass.set_bind_group(0, b, &[]);
        pass.dispatch_workgroups(g, 1, 1);
    };
    let flags = |qn: bool, kn: bool| (if qn { 2u32 } else { 0 }) | (if kn { 4 } else { 0 }) | (if gemma { 8 } else { 0 });
    // Constant uniforms for the whole token (position is fixed for this call).
    // Token-invariant ones use the content-keyed cache; position-dependent ones
    // use pooled buffers updated via write_buffer (no allocation after first token).
    let g = if gemma { 1u32 } else { 0 };
    let rms_u = uniform_u32x4(c, [hidden as u32, g, eps.to_bits(), 0]);
    let ax_u = uniform_u32x4(c, [1.0f32.to_bits(), hidden as u32, 0, 0]);
    let silu_u = uniform_u32x4(c, [inter as u32, 0, 0, 0]);
    let kv_u = GraphScratch::ensure_uniform(&c.device, &mut gs.kv_u, 16);
    c.queue.write_buffer(&kv_u, 0, bytemuck::cast_slice(&[nkv as u32, hd as u32, cap as u32, position as u32]));
    let at_u = GraphScratch::ensure_uniform(&c.device, &mut gs.at_u, 32);
    c.queue.write_buffer(&at_u, 0, bytemuck::cast_slice(&[nh as u32, (nh / nkv) as u32, hd as u32, cap as u32, (position + 1) as u32, 0, 0, 0]));
    let rope_u = GraphScratch::ensure_uniform(&c.device, &mut gs.rope_u, 32);
    // Encode one matvec, dtype-dispatched: q8_row (encode_matvec + row scales)
    // or q1 (encode_matvec_q1). Each is its own pass — pass-grouping measured
    // as a no-op (the wall is per-dispatch, not per-barrier).
    let emat = |enc: &mut wgpu::CommandEncoder, m: &GMat, xs: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, cols: usize| {
        match m.kind {
            0 => encode_matvec(c, enc, &m.buf, xs, m.rs.as_ref().unwrap(), y, rows, cols),
            1 => encode_matvec_q1(c, enc, &m.buf, xs, y, rows, cols),
            2 => encode_q1t_like(c, enc, &c.q4b, &m.buf, xs, y, rows, cols),
            3 => encode_q1t_like(c, enc, &c.q1t, &m.buf, xs, y, rows, cols),
            _ => encode_f32matvec(c, enc, &m.buf, xs, y, rows, cols),
        }
    };
    // Prep a matvec (pipeline, bind group, workgroups) WITHOUT opening a pass —
    // so several independent ones can share a pass. None = a dtype we don't
    // group (q4t/q1t) → caller falls back to per-op emat. The bind group keeps
    // its uniform buffer alive, so returning it alone is enough.
    let prep = |m: &GMat, xs: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, cols: usize| -> Option<(&wgpu::ComputePipeline, wgpu::BindGroup, u32)> {
        match m.kind {
            0 => {
                let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, 0, 0]);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &c.layout, entries: &[bind_buf(0, &m.buf), bind_buf(1, xs), bind_buf(2, m.rs.as_ref().unwrap()), bind_buf(3, y), bind_buf(4, &p_buf)] });
                Some((&c.matvec, bind, (rows as u32).min(MAX_WG)))
            }
            1 => {
                let gpr = cols / 32;
                let p_buf = uniform_u32x4(c, [(gpr / 2) as u32, rows as u32, 0, 0]);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &c.layout_q1, entries: &[bind_buf(0, &m.buf), bind_buf(1, xs), bind_buf(2, y), bind_buf(3, &p_buf)] });
                Some((&c.q1, bind, (rows as u32).div_ceil(8).min(MAX_WG)))
            }
            4 => {
                let p_buf = uniform_u32x4(c, [cols as u32, rows as u32, 0, 0]);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &c.layout_f32, entries: &[bind_buf(0, &m.buf), bind_buf(1, xs), bind_buf(2, y), bind_buf(3, &p_buf)] });
                Some((&c.f32_matvec, bind, (rows as u32).min(MAX_WG)))
            }
            2 => {
                let gpr = cols / 32;
                let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
                let layout = c.q4b.get_bind_group_layout(0);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[bind_buf(0, &m.buf), bind_buf(1, xs), bind_buf(2, y), bind_buf(3, &p_buf)] });
                Some((&c.q4b, bind, (rows as u32).min(MAX_WG)))
            }
            3 => {
                let gpr = cols / 32;
                let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
                let layout = c.q1t.get_bind_group_layout(0);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &[bind_buf(0, &m.buf), bind_buf(1, xs), bind_buf(2, y), bind_buf(3, &p_buf)] });
                Some((&c.q1t, bind, (rows as u32).min(MAX_WG)))
            }
            _ => None,
        }
    };
    // Emit a set of mutually-INDEPENDENT matvecs. When grouping is on and every
    // one preps, they share a single compute pass (no barrier between them);
    // otherwise each goes through emat as its own pass. Correctness rests on the
    // caller passing only matvecs with no read-after-write among them.
    let group_mats = |enc: &mut wgpu::CommandEncoder, mats: &[(&GMat, &wgpu::Buffer, &wgpu::Buffer, usize, usize)]| {
        if group {
            let prepped: Vec<_> = mats.iter().filter_map(|(m, xs, y, r, cc)| prep(m, xs, y, *r, *cc)).collect();
            if prepped.len() == mats.len() {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                for (p, b, g) in &prepped {
                    pass.set_pipeline(p);
                    pass.set_bind_group(0, b, &[]);
                    pass.dispatch_workgroups(*g, 1, 1);
                }
                return;
            }
        }
        for (m, xs, y, r, cc) in mats {
            emat(enc, m, xs, y, *r, *cc);
        }
    };
    // Bootstrap the first layer's attention norm; thereafter each residual is
    // fused with the following norm (add_rmsnorm), saving two dispatches/layer.
    let inw0 = stor(bytemuck::cast_slice(layers[0].input_norm));
    go(&mut enc, &c.rmsnorm, &bg(&c.layout_rmsnorm, &[&h_buf, &inw0, &n1, &rms_u]), 1);
    for (li, l) in layers.iter().enumerate() {
        let lw = &lws[li];
        let pnw = stor(bytemuck::cast_slice(l.post_norm));
        // ── token mixing (attention or GDN) → ob ──
        match (&lw.attn, &l.attn) {
            (LAttn::Full { wq, wk, wv, wo }, crate::gpu::GraphAttn::Full { q_norm, k_norm, bias, output_gate, .. }) => {
                let (kbuf, vbuf) = kvbufs[li].as_ref().unwrap();
                let qnw = q_norm.map(|q| stor(bytemuck::cast_slice(q))).unwrap_or_else(|| zeros(hd));
                let knw = k_norm.map(|k| stor(bytemuck::cast_slice(k))).unwrap_or_else(|| zeros(hd));
                let gate_flag = if *output_gate { 1u32 } else { 0 };
                c.queue.write_buffer(&rope_u, 0, bytemuck::cast_slice(&[nh as u32, nkv as u32, hd as u32, rd as u32, position as u32, flags(q_norm.is_some(), k_norm.is_some()) | gate_flag, eps.to_bits(), 0]));
                // Gated wq emits 2·nh·hd (q||gate interleaved per head); the rope
                // kernel splits it, roping q and passing gate through to `gout`.
                let qrows = nh * hd * (1 + *output_gate as usize);
                group_mats(&mut enc, &[(wq, &n1, &qraw, qrows, hidden), (wk, &n1, &kb, nkv * hd, hidden), (wv, &n1, &vb, nkv * hd, hidden)]);
                if let Some((bq, bk, bv)) = bias {
                    let (bqb, bkb, bvb) = (stor(bytemuck::cast_slice(bq)), stor(bytemuck::cast_slice(bk)), stor(bytemuck::cast_slice(bv)));
                    let axq = uniform_u32x4(c, [1.0f32.to_bits(), (nh * hd) as u32, 0, 0]);
                    let axkv = uniform_u32x4(c, [1.0f32.to_bits(), (nkv * hd) as u32, 0, 0]);
                    go(&mut enc, &c.axpy, &bg(&c.layout_axpy, &[&bqb, &qraw, &axq]), ((nh * hd) as u32).div_ceil(256));
                    go(&mut enc, &c.axpy, &bg(&c.layout_axpy, &[&bkb, &kb, &axkv]), ((nkv * hd) as u32).div_ceil(256));
                    go(&mut enc, &c.axpy, &bg(&c.layout_axpy, &[&bvb, &vb, &axkv]), ((nkv * hd) as u32).div_ceil(256));
                }
                // rope + kv_append are independent (both read kb, neither
                // writes it) — share ONE compute pass to avoid the inter-pass
                // pipeline flush (~78 μs on NVIDIA Vulkan).
                {
                    let bg_rope = bg(&c.layout_attn_rope, &[&qraw, &kb, &qout, &gout, &qnw, &knw, &invf_b, &rope_u]);
                    let bg_kv = bg(&c.layout_kv, &[&kb, &vb, kbuf, vbuf, &kv_u]);
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                    pass.set_pipeline(&c.attn_rope);
                    pass.set_bind_group(0, &bg_rope, &[]);
                    pass.dispatch_workgroups((nh + nkv) as u32, 1, 1);
                    pass.set_pipeline(&c.kv_append);
                    pass.set_bind_group(0, &bg_kv, &[]);
                    pass.dispatch_workgroups(((nkv * hd) as u32).div_ceil(256), 1, 1);
                }
                go(&mut enc, &c.gqa_attend, &bg(&c.layout_attend, &[&qout, kbuf, vbuf, &attn, &at_u]), nh as u32);
                // attn_out *= sigmoid(gate) before the O projection.
                if *output_gate {
                    let gm_u = uniform_u32x4(c, [(nh * hd) as u32, 0, 0, 0]);
                    go(&mut enc, &c.gate_mul, &bg(&c.layout_gate_mul, &[&gout, &attn, &gm_u]), ((nh * hd) as u32).div_ceil(256));
                }
                emat(&mut enc, wo, &attn, &ob, hidden, nh * hd);
            }
            (LAttn::Gdn { qkv, z, a, b, out, nv, nk, dk, dv, kk, cdim }, crate::gpu::GraphAttn::Gdn { conv1d, a_log, dt_bias, norm, .. }) => {
                let (ring, s) = gdnbufs[li].as_ref().unwrap();
                let taps = stor(bytemuck::cast_slice(conv1d));
                let alog = stor(bytemuck::cast_slice(a_log));
                let dtb = stor(bytemuck::cast_slice(dt_bias));
                let gnorm = stor(bytemuck::cast_slice(norm));
                group_mats(&mut enc, &[(qkv, &n1, &qkv_b, *cdim, hidden), (z, &n1, &z_b, nv * dv, hidden), (a, &n1, &a_b, *nv, hidden), (b, &n1, &b_b, *nv, hidden)]);
                let gc_p = uniform_u32x4(c, [*cdim as u32, *kk as u32, 0, 0]);
                go(&mut enc, &c.gdn_conv, &bg(&c.layout_gdn_conv, &[&qkv_b, &taps, ring, &cq_b, &gc_p]), (*cdim as u32).div_ceil(256));
                let gd_p = unif(&[*nv as u32, *dk as u32, *dv as u32, (nk * dk) as u32, (nv / nk) as u32, *cdim as u32, eps.to_bits(), 0]);
                go(&mut enc, &c.gdn_step, &bg(&c.layout_gdn, &[&cq_b, &z_b, &a_b, &b_b, &alog, &dtb, &gnorm, s, &gdo_b, &gd_p]), *nv as u32);
                emat(&mut enc, out, &gdo_b, &ob, hidden, nv * dv);
            }
            _ => return false,
        }
        // token-mix residual + FFN-norm fused: h += ob, n1 = rms(h, post_norm)
        go(&mut enc, &c.add_rmsnorm, &bg(&c.layout_add_rmsnorm, &[&h_buf, &ob, &pnw, &n1, &rms_u]), 1);
        // SiLU FFN: gate+up matvecs + silu fused in ONE compute pass
        // (dispatches within a pass are serialized — silu safely reads gate/up output).
        {
            let pg = prep(&lw.gate, &n1, &gbuf, inter, hidden);
            let pu = prep(&lw.up, &n1, &ubuf, inter, hidden);
            if let (Some((pgp, bg_g, wg)), Some((pup, bg_u, wu))) = (pg, pu) {
                let bg_silu = bg(&c.layout_silu, &[&gbuf, &ubuf, &dummy_hd, &abuf, &silu_u]);
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(pgp);
                pass.set_bind_group(0, &bg_g, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
                pass.set_pipeline(pup);
                pass.set_bind_group(0, &bg_u, &[]);
                pass.dispatch_workgroups(wu, 1, 1);
                pass.set_pipeline(&c.silu);
                pass.set_bind_group(0, &bg_silu, &[]);
                pass.dispatch_workgroups((inter as u32).div_ceil(256), 1, 1);
            } else {
                group_mats(&mut enc, &[(&lw.gate, &n1, &gbuf, inter, hidden), (&lw.up, &n1, &ubuf, inter, hidden)]);
                go(&mut enc, &c.silu, &bg(&c.layout_silu, &[&gbuf, &ubuf, &dummy_hd, &abuf, &silu_u]), (inter as u32).div_ceil(256));
            }
        }
        emat(&mut enc, &lw.down, &abuf, &ob, hidden, inter);
        // FFN-residual + next layer's attn-norm fused (plain residual on the last).
        // At loop boundaries (Looped Transformer), insert final_norm between the
        // residual and the next iteration's input norm.
        if li + 1 < layers.len() {
            if loop_norm_at.contains(&li) {
                // h += ob; n1 = rms(h, final_norm); copy n1→h; n1 = rms(h, next_input_norm)
                let fnw = stor(bytemuck::cast_slice(final_norm));
                let inw_next = stor(bytemuck::cast_slice(layers[li + 1].input_norm));
                go(&mut enc, &c.add_rmsnorm, &bg(&c.layout_add_rmsnorm, &[&h_buf, &ob, &fnw, &n1, &rms_u]), 1);
                enc.copy_buffer_to_buffer(&n1, 0, &h_buf, 0, (hidden * 4) as u64);
                go(&mut enc, &c.rmsnorm, &bg(&c.layout_rmsnorm, &[&h_buf, &inw_next, &n1, &rms_u]), 1);
            } else {
                let inw_next = stor(bytemuck::cast_slice(layers[li + 1].input_norm));
                go(&mut enc, &c.add_rmsnorm, &bg(&c.layout_add_rmsnorm, &[&h_buf, &ob, &inw_next, &n1, &rms_u]), 1);
            }
        } else {
            go(&mut enc, &c.axpy, &bg(&c.layout_axpy, &[&ob, &h_buf, &ax_u]), (hidden as u32).div_ceil(256));
        }
    }
    let t_enc = t_enc0.elapsed().as_secs_f64() * 1000.0;
    let t_sub0 = std::time::Instant::now();
    // h_buf now holds the final hidden. Either ride final-norm + lm_head and
    // read back logits, or (no lm / unresolved weight) read back the hidden.
    let lm_resolved = lm_head.and_then(|(gw, rows)| resolve(gw, rows, hidden).map(|m| (m, rows)));
    let ok = if let Some((lm, lrows)) = lm_resolved {
        let fnw = stor(bytemuck::cast_slice(final_norm));
        go(&mut enc, &c.rmsnorm, &bg(&c.layout_rmsnorm, &[&h_buf, &fnw, &n1, &rms_u]), 1);
        let lsize = (lrows * 4) as u64;
        let lbuf = GraphScratch::ensure(&c.device, &mut gs.logits, lsize, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "g-logits");
        emat(&mut enc, &lm, &n1, &lbuf, lrows, hidden);
        logits.resize(lrows, 0.0);
        let stage = GraphScratch::ensure(&c.device, &mut gs.stage, lsize, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "g-stage");
        let r = readback(c, enc, &lbuf, &stage, lsize, &mut logits[..lrows]);
        drop(gs);
        r
    } else {
        let size = (hidden * 4) as u64;
        let stage = GraphScratch::ensure(&c.device, &mut gs.stage, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "g-stage");
        let r = readback(c, enc, &h_buf, &stage, size, &mut h[..hidden]);
        drop(gs);
        r
    };
    if ok {
        // The append at `position` is now durable — advance each mirror.
        let mut kvm = c.attn_kv.lock().unwrap();
        for li in 0..layers.len() {
            if let Some(m) = kvm.get_mut(&(kv_id, li)) {
                m.synced = position + 1;
            }
        }
    }
    if prof {
        let setup = t_enc0.duration_since(t_start).as_secs_f64() * 1000.0;
        eprintln!("token-graph: setup {setup:.2} ms | encode {t_enc:.2} ms | submit+readback {:.2} ms", t_sub0.elapsed().as_secs_f64() * 1000.0);
    }
    ok
}

/// Batched prefill: K prompt positions through the whole layer stack in ONE
/// submit. Projections & FFN run as resident GEMMs (each weight read once per K
/// columns instead of once per position); attention and GDN loop the existing
/// per-position kernels over scratch slices (KV mirror / recurrent S persist).
/// Cuts graph prefill from N whole-graph submits to N/K. Returns false on any
/// unsupported case (bias, q4t/q1t projections) → caller keeps the per-position
/// graph. positions[i] = absolute sequence position of batch row i (contiguous
/// causal run starting at positions[0]); `h` is [k·hidden] in/out.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_graph(
    model: &Arc<CmfModel>,
    kv_id: u64,
    layers: &[crate::gpu::GraphLayer],
    invf: &[f32],
    h: &mut [f32],
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    inter: usize,
    positions: &[usize],
    cap: usize,
    gemma: bool,
    eps: f32,
    k: usize,
) -> bool {
    let Some(c) = ctx() else { return false };
    if k == 0 || positions.len() != k {
        return false;
    }
    let pos0 = positions[0];
    if pos0 + k > cap {
        return false;
    }
    struct GMat { buf: wgpu::Buffer, rs: Option<wgpu::Buffer>, kind: u8 }
    enum LAttn {
        Full { wq: GMat, wk: GMat, wv: GMat, wo: GMat },
        Gdn { qkv: GMat, z: GMat, a: GMat, b: GMat, out: GMat, nv: usize, nk: usize, dk: usize, dv: usize, kk: usize, cdim: usize },
    }
    struct LW { attn: LAttn, gate: GMat, up: GMat, down: GMat }
    let resolve = |gw: &crate::gpu::GraphW, rows: usize, cols: usize| -> Option<GMat> {
        match gw.kind {
            0 => {
                if gw.row_scale.len() < rows { return None; }
                let b = tensor_weight(c, model, gw.idx, rows, cols)?;
                let rsb = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("bg-rs"), size: (rows * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                c.queue.write_buffer(&rsb, 0, bytemuck::cast_slice(&gw.row_scale[..rows]));
                Some(GMat { buf: b, rs: Some(rsb), kind: 0 })
            }
            1 => {
                let (b, r, cc) = q1_weight(c, model, gw.idx)?;
                if r != rows || cc != cols { return None; }
                Some(GMat { buf: b, rs: None, kind: 1 })
            }
            4 => {
                if gw.data.len() < rows * cols { return None; }
                let b = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("bg-f32w"), size: (rows * cols * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                c.queue.write_buffer(&b, 0, bytemuck::cast_slice(&gw.data[..rows * cols]));
                Some(GMat { buf: b, rs: None, kind: 4 })
            }
            _ => None, // q4t/q1t not batched here → CPU/per-position path
        }
    };
    // GEMM-able projection? (q8_row/q1). f32 (a/b) is per-position; anything else bails.
    let gemmable = |m: &GMat| m.kind == 0 || m.kind == 1;
    let mut lws = Vec::with_capacity(layers.len());
    let mut gdn_dims: Option<(usize, usize, usize, usize, usize, usize)> = None;
    for l in layers {
        let attn = match &l.attn {
            crate::gpu::GraphAttn::Full { wq, wk, wv, wo, output_gate, bias, .. } => {
                if bias.is_some() { return false; } // batched bias axpy not wired
                let qrows = nh * hd * (1 + *output_gate as usize);
                let (Some(wq), Some(wk), Some(wv), Some(wo)) = (resolve(wq, qrows, hidden), resolve(wk, nkv * hd, hidden), resolve(wv, nkv * hd, hidden), resolve(wo, hidden, nh * hd)) else { return false };
                if !(gemmable(&wq) && gemmable(&wk) && gemmable(&wv) && gemmable(&wo)) { return false; }
                LAttn::Full { wq, wk, wv, wo }
            }
            crate::gpu::GraphAttn::Gdn { qkv, z, a, b, out, nv, nk, dk, dv, kk, .. } => {
                let cdim = 2 * nk * dk + nv * dv;
                gdn_dims = Some((*nv, *nk, *dk, *dv, *kk, cdim));
                let (Some(qkv), Some(z), Some(a), Some(b), Some(out)) = (resolve(qkv, cdim, hidden), resolve(z, nv * dv, hidden), resolve(a, *nv, hidden), resolve(b, *nv, hidden), resolve(out, hidden, nv * dv)) else { return false };
                if !(gemmable(&qkv) && gemmable(&z) && gemmable(&out) && a.kind == 4 && b.kind == 4) { return false; }
                LAttn::Gdn { qkv, z, a, b, out, nv: *nv, nk: *nk, dk: *dk, dv: *dv, kk: *kk, cdim }
            }
        };
        let (Some(gate), Some(up), Some(down)) = (resolve(&l.gate, inter, hidden), resolve(&l.up, inter, hidden), resolve(&l.down, hidden, inter)) else { return false };
        if !(gemmable(&gate) && gemmable(&up) && gemmable(&down)) { return false; }
        lws.push(LW { attn, gate, up, down });
    }
    let stor = |data: &[u8]| { let b = c.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: data.len() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }); c.queue.write_buffer(&b, 0, data); b };
    let unif = |d: &[u32]| c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(d), usage: wgpu::BufferUsages::UNIFORM });
    let bg = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| { let e: Vec<_> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect(); c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &e }) };
    // Buffers usable both as compute storage and copy src/dst (K-loop slicing).
    let rwc = |n: usize| c.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n.max(1) * 4) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let h_buf = rwc(k * hidden);
    c.queue.write_buffer(&h_buf, 0, bytemuck::cast_slice(&h[..k * hidden]));
    let n1 = rwc(k * hidden);
    let any_gate = layers.iter().any(|l| matches!(&l.attn, crate::gpu::GraphAttn::Full { output_gate: true, .. }));
    let qdim = nh * hd * (1 + any_gate as usize);
    let (gnv, _gnk, gdk, gdv, _gkk, gcdim) = gdn_dims.unwrap_or((1, 1, 1, 1, 1, 1));
    // batched GEMM outputs
    let qraw_b = rwc(k * qdim);
    let kb_b = rwc(k * nkv * hd);
    let vb_b = rwc(k * nkv * hd);
    let attn_bb = rwc(k * nh * hd);
    let qkv_b = rwc(k * gcdim);
    let z_b = rwc(k * gnv * gdv);
    let gdo_b = rwc(k * gnv * gdv);
    let ob = rwc(k * hidden);
    let gbuf = rwc(k * inter);
    let ubuf = rwc(k * inter);
    let abuf = rwc(k * inter);
    // per-position scratch
    let n1_s = rwc(hidden);
    let qraw_s = rwc(qdim);
    let kb_s = rwc(nkv * hd);
    let vb_s = rwc(nkv * hd);
    let qout_s = rwc(nh * hd);
    let gout_s = rwc(nh * hd);
    let attn_s = rwc(nh * hd);
    let qkv_s = rwc(gcdim);
    let cq_s = rwc(gcdim);
    let z_s = rwc(gnv * gdv);
    let a_s = rwc(gnv);
    let b_s = rwc(gnv);
    let gdo_s = rwc(gnv * gdv);
    let invf_b = stor(bytemuck::cast_slice(invf));
    let dummy_hd = stor(bytemuck::cast_slice(&vec![0f32; hd]));
    // KV mirror + GDN state (fresh; batch appends positions pos0..pos0+k).
    let mut kvbufs: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>> = Vec::with_capacity(layers.len());
    let mut gdnbufs: Vec<Option<(wgpu::Buffer, wgpu::Buffer)>> = Vec::with_capacity(layers.len());
    {
        let mut kvm = c.attn_kv.lock().unwrap();
        let mut gsm = c.gdn_state.lock().unwrap();
        for (li, l) in layers.iter().enumerate() {
            match &l.attn {
                crate::gpu::GraphAttn::Full { .. } => {
                    let e = kvm.entry((kv_id, li)).or_insert_with(|| {
                        let sz = (nkv * cap * hd * 4) as u64;
                        let mk = || c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("kv"), size: sz, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                        KvMirror { k: mk(), v: mk(), synced: 0 }
                    });
                    kvbufs.push(Some((e.k.clone(), e.v.clone())));
                    gdnbufs.push(None);
                }
                crate::gpu::GraphAttn::Gdn { .. } => {
                    let e = gsm.entry((kv_id, li)).or_insert_with(|| {
                        let ring_sz = (gcdim * (_gkk.max(1).saturating_sub(1)) * 4) as u64;
                        let s_sz = (gnv * gdk * gdv * 4) as u64;
                        let mk = |sz: u64| { let bf = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("gdn-state"), size: sz.max(4), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }); c.queue.write_buffer(&bf, 0, &vec![0u8; sz.max(4) as usize]); bf };
                        (mk(ring_sz), mk(s_sz))
                    });
                    gdnbufs.push(Some((e.0.clone(), e.1.clone())));
                    kvbufs.push(None);
                }
            }
        }
    }
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch-graph") });
    let go = |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, b: &wgpu::BindGroup, g: u32| { let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None }); pass.set_pipeline(p); pass.set_bind_group(0, b, &[]); pass.dispatch_workgroups(g, 1, 1); };
    let flags = |qn: bool, kn: bool| (if qn { 2u32 } else { 0 }) | (if kn { 4 } else { 0 }) | (if gemma { 8 } else { 0 });
    let rms_u = unif(&[hidden as u32, if gemma { 1 } else { 0 }, eps.to_bits(), 0]);
    let silu_u = unif(&[(k * inter) as u32, 0, 0, 0]);
    // Batched GEMM matvec (q8_row / q1) into a [k·rows] output.
    let ematb = |enc: &mut wgpu::CommandEncoder, m: &GMat, xs: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, cols: usize| {
        match m.kind {
            0 => encode_q8_mm(c, enc, &m.buf, m.rs.as_ref().unwrap(), xs, y, rows, cols, k),
            _ => encode_q1_mm(c, enc, &m.buf, xs, y, rows, cols, k),
        }
    };
    let cp = |enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, so: usize, dst: &wgpu::Buffer, n: usize| enc.copy_buffer_to_buffer(src, (so * 4) as u64, dst, 0, (n * 4) as u64);
    let cpo = |enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, dst: &wgpu::Buffer, dof: usize, n: usize| enc.copy_buffer_to_buffer(src, 0, dst, (dof * 4) as u64, (n * 4) as u64);
    // Bootstrap first layer's input norm over all k rows.
    let inw0 = stor(bytemuck::cast_slice(layers[0].input_norm));
    go(&mut enc, &c.rmsnorm_b, &bg(&c.layout_rmsnorm_b, &[&h_buf, &inw0, &n1, &rms_u]), k as u32);
    for (li, l) in layers.iter().enumerate() {
        let lw = &lws[li];
        let pnw = stor(bytemuck::cast_slice(l.post_norm));
        match (&lw.attn, &l.attn) {
            (LAttn::Full { wq, wk, wv, wo }, crate::gpu::GraphAttn::Full { q_norm, k_norm, output_gate, .. }) => {
                let (kbuf, vbuf) = kvbufs[li].as_ref().unwrap();
                let qnw = stor(bytemuck::cast_slice(q_norm.unwrap_or(&vec![0f32; hd])));
                let knw = stor(bytemuck::cast_slice(k_norm.unwrap_or(&vec![0f32; hd])));
                let qrows = nh * hd * (1 + *output_gate as usize);
                ematb(&mut enc, wq, &n1, &qraw_b, qrows, hidden);
                ematb(&mut enc, wk, &n1, &kb_b, nkv * hd, hidden);
                ematb(&mut enc, wv, &n1, &vb_b, nkv * hd, hidden);
                for i in 0..k {
                    cp(&mut enc, &qraw_b, i * qrows, &qraw_s, qrows);
                    cp(&mut enc, &kb_b, i * nkv * hd, &kb_s, nkv * hd);
                    cp(&mut enc, &vb_b, i * nkv * hd, &vb_s, nkv * hd);
                    let p = positions[i];
                    let gate_flag = if *output_gate { 1u32 } else { 0 };
                    let rope_u = unif(&[nh as u32, nkv as u32, hd as u32, rd as u32, p as u32, flags(q_norm.is_some(), k_norm.is_some()) | gate_flag, eps.to_bits(), 0]);
                    let kv_u = unif(&[nkv as u32, hd as u32, cap as u32, p as u32]);
                    let at_u = unif(&[nh as u32, (nh / nkv) as u32, hd as u32, cap as u32, (p + 1) as u32, 0, 0, 0]);
                    go(&mut enc, &c.attn_rope, &bg(&c.layout_attn_rope, &[&qraw_s, &kb_s, &qout_s, &gout_s, &qnw, &knw, &invf_b, &rope_u]), (nh + nkv) as u32);
                    go(&mut enc, &c.kv_append, &bg(&c.layout_kv, &[&kb_s, &vb_s, kbuf, vbuf, &kv_u]), ((nkv * hd) as u32).div_ceil(256));
                    go(&mut enc, &c.gqa_attend, &bg(&c.layout_attend, &[&qout_s, kbuf, vbuf, &attn_s, &at_u]), nh as u32);
                    if *output_gate {
                        let gm_u = unif(&[(nh * hd) as u32, 0, 0, 0]);
                        go(&mut enc, &c.gate_mul, &bg(&c.layout_gate_mul, &[&gout_s, &attn_s, &gm_u]), ((nh * hd) as u32).div_ceil(256));
                    }
                    cpo(&mut enc, &attn_s, &attn_bb, i * nh * hd, nh * hd);
                }
                ematb(&mut enc, wo, &attn_bb, &ob, hidden, nh * hd);
            }
            (LAttn::Gdn { qkv, z, a, b, out, nv, nk, dk, dv, kk, cdim }, crate::gpu::GraphAttn::Gdn { conv1d, a_log, dt_bias, norm, .. }) => {
                let (ring, s) = gdnbufs[li].as_ref().unwrap();
                let taps = stor(bytemuck::cast_slice(conv1d));
                let alog = stor(bytemuck::cast_slice(a_log));
                let dtb = stor(bytemuck::cast_slice(dt_bias));
                let gnorm = stor(bytemuck::cast_slice(norm));
                ematb(&mut enc, qkv, &n1, &qkv_b, *cdim, hidden);
                ematb(&mut enc, z, &n1, &z_b, nv * dv, hidden);
                let gc_p = unif(&[*cdim as u32, *kk as u32, 0, 0]);
                let gd_p = unif(&[*nv as u32, *dk as u32, *dv as u32, (nk * dk) as u32, (nv / nk) as u32, *cdim as u32, eps.to_bits(), 0]);
                for i in 0..k {
                    cp(&mut enc, &qkv_b, i * cdim, &qkv_s, *cdim);
                    cp(&mut enc, &z_b, i * nv * dv, &z_s, nv * dv);
                    cp(&mut enc, &n1, i * hidden, &n1_s, hidden);
                    encode_f32matvec(c, &mut enc, &a.buf, &n1_s, &a_s, *nv, hidden);
                    encode_f32matvec(c, &mut enc, &b.buf, &n1_s, &b_s, *nv, hidden);
                    go(&mut enc, &c.gdn_conv, &bg(&c.layout_gdn_conv, &[&qkv_s, &taps, ring, &cq_s, &gc_p]), (*cdim as u32).div_ceil(256));
                    go(&mut enc, &c.gdn_step, &bg(&c.layout_gdn, &[&cq_s, &z_s, &a_s, &b_s, &alog, &dtb, &gnorm, s, &gdo_s, &gd_p]), *nv as u32);
                    cpo(&mut enc, &gdo_s, &gdo_b, i * nv * dv, nv * dv);
                }
                ematb(&mut enc, out, &gdo_b, &ob, hidden, nv * dv);
            }
            _ => return false,
        }
        go(&mut enc, &c.add_rmsnorm_b, &bg(&c.layout_add_rmsnorm_b, &[&h_buf, &ob, &pnw, &n1, &rms_u]), k as u32);
        ematb(&mut enc, &lw.gate, &n1, &gbuf, inter, hidden);
        ematb(&mut enc, &lw.up, &n1, &ubuf, inter, hidden);
        go(&mut enc, &c.silu, &bg(&c.layout_silu, &[&gbuf, &ubuf, &dummy_hd, &abuf, &silu_u]), ((k * inter) as u32).div_ceil(256));
        ematb(&mut enc, &lw.down, &abuf, &ob, hidden, inter);
        if li + 1 < layers.len() {
            let inw_next = stor(bytemuck::cast_slice(layers[li + 1].input_norm));
            go(&mut enc, &c.add_rmsnorm_b, &bg(&c.layout_add_rmsnorm_b, &[&h_buf, &ob, &inw_next, &n1, &rms_u]), k as u32);
        } else {
            let ax_u = unif(&[1.0f32.to_bits(), (k * hidden) as u32, 0, 0]);
            go(&mut enc, &c.axpy, &bg(&c.layout_axpy, &[&ob, &h_buf, &ax_u]), ((k * hidden) as u32).div_ceil(256));
        }
    }
    let size = (k * hidden * 4) as u64;
    let stage = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("bg-stage"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let ok = readback(c, enc, &h_buf, &stage, size, &mut h[..k * hidden]);
    if ok {
        let mut kvm = c.attn_kv.lock().unwrap();
        for li in 0..layers.len() {
            if let Some(m) = kvm.get_mut(&(kv_id, li)) { m.synced = pos0 + k; }
        }
    }
    ok
}

/// Drop the device K/V mirror for a pipeline (called on cache clear).
pub fn kv_mirror_reset(kv_id: u64) {
    if let Some(c) = ctx() {
        c.attn_kv.lock().unwrap().retain(|(id, _), _| *id != kv_id);
        c.gdn_state.lock().unwrap().retain(|(id, _), _| *id != kv_id);
    }
}

/// GDN depthwise conv step (bring-up / parity): updates cq [cdim] and shifts
/// the ring [(kk-1)·cdim] in place.
pub fn gdn_conv_gpu(qkv: &[f32], taps: &[f32], ring: &mut [f32], cdim: usize, kk: usize, cq: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let qb = storage_bytes(c, bytemuck::cast_slice(qkv));
    let tb = storage_bytes(c, bytemuck::cast_slice(taps));
    let rb = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(ring),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let cb = rw_f32(c, cdim, true);
    let p = uniform_u32x4(c, [cdim as u32, kk as u32, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_gdn_conv,
        entries: &[bind_buf(0, &qb), bind_buf(1, &tb), bind_buf(2, &rb), bind_buf(3, &cb), bind_buf(4, &p)],
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&c.gdn_conv);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((cdim as u32).div_ceil(256), 1, 1);
    }
    let rsz = (ring.len() * 4) as u64;
    let csz = (cdim * 4) as u64;
    let sr = c.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: rsz, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let scq = c.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: csz, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&rb, 0, &sr, 0, rsz);
    enc.copy_buffer_to_buffer(&cb, 0, &scq, 0, csz);
    c.queue.submit(Some(enc.finish()));
    sr.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    scq.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    let (Ok(dr), Ok(dc)) = (sr.slice(..).get_mapped_range(), scq.slice(..).get_mapped_range()) else {
        return false;
    };
    ring.copy_from_slice(bytemuck::cast_slice(&dr[..ring.len() * 4]));
    cq[..cdim].copy_from_slice(bytemuck::cast_slice(&dc[..cdim * 4]));
    true
}

/// GDN decode step (bring-up / parity): one workgroup per v-head. `s` is the
/// [nv·dk·dv] recurrent state, updated in place; writes `o` [nv·dv].
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_gpu(
    cq: &[f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
    alog: &[f32],
    dtb: &[f32],
    norm: &[f32],
    s: &mut [f32],
    nv: usize,
    dk: usize,
    dv: usize,
    kd: usize,
    rep: usize,
    cdim: usize,
    eps: f32,
    o: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let sb = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gdn-s"),
        contents: bytemuck::cast_slice(s),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let ob = rw_f32(c, nv * dv, true);
    let p = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gdn-p"),
        contents: bytemuck::cast_slice(&[
            nv as u32, dk as u32, dv as u32, kd as u32, rep as u32, cdim as u32, eps.to_bits(), 0u32,
        ]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sbuf = |d: &[f32]| storage_bytes(c, bytemuck::cast_slice(d));
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gdn-bg"),
        layout: &c.layout_gdn,
        entries: &[
            bind_buf(0, &sbuf(cq)),
            bind_buf(1, &sbuf(z)),
            bind_buf(2, &sbuf(a)),
            bind_buf(3, &sbuf(b)),
            bind_buf(4, &sbuf(alog)),
            bind_buf(5, &sbuf(dtb)),
            bind_buf(6, &sbuf(norm)),
            bind_buf(7, &sb),
            bind_buf(8, &ob),
            bind_buf(9, &p),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gdn") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gdn"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.gdn_step);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(nv as u32, 1, 1);
    }
    // read back updated S and o
    let ssz = (s.len() * 4) as u64;
    let osz = (nv * dv * 4) as u64;
    let stage_s = c.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: ssz, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let stage_o = c.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: osz, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    enc.copy_buffer_to_buffer(&sb, 0, &stage_s, 0, ssz);
    enc.copy_buffer_to_buffer(&ob, 0, &stage_o, 0, osz);
    c.queue.submit(Some(enc.finish()));
    stage_s.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    stage_o.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    let (Ok(ds), Ok(dobuf)) = (stage_s.slice(..).get_mapped_range(), stage_o.slice(..).get_mapped_range()) else {
        return false;
    };
    s.copy_from_slice(bytemuck::cast_slice(&ds[..s.len() * 4]));
    o[..nv * dv].copy_from_slice(bytemuck::cast_slice(&dobuf[..nv * dv * 4]));
    true
}

/// One full attention sub-block resident on the GPU in a SINGLE command
/// encoder: rmsnorm → QKV (q1) → rope/qk-norm → kv_append → attend → O (q1)
/// → residual. The K/V cache lives on the device ([nkv,cap,hd]) and persists
/// across tokens; only the updated hidden is read back. This is the token
/// graph's attention half — it collapses ~6 per-op submits into one.
/// `flags` follows attn_rope_qkn (2=qnorm 4=knorm 8=gemma; gate unsupported
/// here). Weights are raw q1 payloads (bring-up path; production keys the
/// resident VRAM cache). Returns false without a GPU context.
#[allow(clippy::too_many_arguments)]
pub fn attn_block_gpu(
    h_in: &[f32],
    attn_norm_w: &[f32],
    wq: &[u8],
    wk: &[u8],
    wv: &[u8],
    wo: &[u8],
    qnw: &[f32],
    knw: &[f32],
    invf: &[f32],
    kbuf: &wgpu::Buffer,
    vbuf: &wgpu::Buffer,
    nh: usize,
    nkv: usize,
    hd: usize,
    rd: usize,
    hidden: usize,
    cap: usize,
    stored: usize,
    flags: u32,
    eps: f32,
    h_out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let unif = |data: &[u32]| {
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blk-u"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };
    let stor = |data: &[u8]| {
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blk-w"),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    // Resident buffers.
    let h_buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("blk-h"),
        contents: bytemuck::cast_slice(&h_in[..hidden]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let normw_b = stor(bytemuck::cast_slice(&attn_norm_w[..hidden]));
    let normed_b = rw_f32(c, hidden, false);
    let wq_b = stor(wq);
    let wk_b = stor(wk);
    let wv_b = stor(wv);
    let wo_b = stor(wo);
    let qraw_b = rw_f32(c, nh * hd, false);
    let k_b = rw_f32(c, nkv * hd, false);
    let v_b = rw_f32(c, nkv * hd, false);
    let qout_b = rw_f32(c, nh * hd, false);
    let gout_b = rw_f32(c, nh * hd, false);
    let qnw_b = stor(bytemuck::cast_slice(qnw));
    let knw_b = stor(bytemuck::cast_slice(knw));
    let invf_b = stor(bytemuck::cast_slice(invf));
    let attn_b = rw_f32(c, nh * hd, false);
    let o_b = rw_f32(c, hidden, false);
    let bg = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let entries: Vec<wgpu::BindGroupEntry> =
            bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
    };
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("attn-block") });
    let dispatch = |enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, groups: u32| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    };
    // 1. rmsnorm(h) -> normed
    let rms_p = unif(&[hidden as u32, 0, eps.to_bits(), 0]);
    dispatch(&mut enc, &c.rmsnorm, &bg(&c.layout_rmsnorm, &[&h_buf, &normw_b, &normed_b, &rms_p]), 1);
    // 2. QKV (q1) from normed
    encode_matvec_q1(c, &mut enc, &wq_b, &normed_b, &qraw_b, nh * hd, hidden);
    encode_matvec_q1(c, &mut enc, &wk_b, &normed_b, &k_b, nkv * hd, hidden);
    encode_matvec_q1(c, &mut enc, &wv_b, &normed_b, &v_b, nkv * hd, hidden);
    // 3. rope + qk-norm
    let rq_p = unif(&[nh as u32, nkv as u32, hd as u32, rd as u32, stored as u32, flags, eps.to_bits(), 0]);
    dispatch(&mut enc, &c.attn_rope, &bg(&c.layout_attn_rope, &[&qraw_b, &k_b, &qout_b, &gout_b, &qnw_b, &knw_b, &invf_b, &rq_p]), (nh + nkv) as u32);
    // 4. kv_append
    let kv_p = unif(&[nkv as u32, hd as u32, cap as u32, stored as u32]);
    let kv_groups = ((nkv * hd) as u32).div_ceil(256);
    dispatch(&mut enc, &c.kv_append, &bg(&c.layout_kv, &[&k_b, &v_b, kbuf, vbuf, &kv_p]), kv_groups);
    // 5. attend
    let at_p = unif(&[nh as u32, (nh / nkv) as u32, hd as u32, cap as u32, (stored + 1) as u32, 0, 0, 0]);
    dispatch(&mut enc, &c.gqa_attend, &bg(&c.layout_attend, &[&qout_b, kbuf, vbuf, &attn_b, &at_p]), nh as u32);
    // 6. O (q1)
    encode_matvec_q1(c, &mut enc, &wo_b, &attn_b, &o_b, hidden, nh * hd);
    // 7. residual h += o
    let ax_p = unif(&[1.0f32.to_bits(), hidden as u32, 0, 0]);
    dispatch(&mut enc, &c.axpy, &bg(&c.layout_axpy, &[&o_b, &h_buf, &ax_p]), (hidden as u32).div_ceil(256));
    // readback updated hidden
    let size = (hidden * 4) as u64;
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "blk-stage",
    );
    let ok = readback(c, enc, &h_buf, &stage, size, &mut h_out[..hidden]);
    drop(sc);
    ok
}

/// q1 kernel body (weight_key = None — no residency cache; test path).
fn dispatch_q1(
    c: &Ctx,
    weight_key: Option<(usize, usize)>,
    payload: &[u8],
    xs: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let gpr = cols / 32;
    let q_buf = match weight_key {
        Some(k) => match weight_buffer(c, k, payload) {
            Some(b) => b,
            None => return false, // over VRAM budget — honest CPU path
        },
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q1-weights"),
            contents: payload,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q1-xs",
    );
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
    let y_size = (rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q1-y",
    );
    let params = [(gpr / 2) as u32, rows as u32, 0u32, 0u32];
    let p_buf = match &sc.params {
        Some(b) => b.clone(),
        None => {
            let b = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q1-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(b.clone());
            b
        }
    };
    c.queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q1-stage",
    );
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q1-bg"),
        layout: &c.layout_q1,
        entries: &[
            bind_buf(0, &q_buf),
            bind_buf(1, &xs_buf),
            bind_buf(2, &y_buf),
            bind_buf(3, &p_buf),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q1") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q1"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q1);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(8).min(MAX_WG), 1, 1);
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..rows]);
    drop(sc);
    ok
}

/// GEMM of the prefill batch: `pre` are prescaled inputs row-major [b, cols],
/// out — row-major [b, rows]. Weights are resident in VRAM. false = CPU path.
#[allow(clippy::too_many_arguments)]
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
    if cols % 4 != 0 || rows == 0 || b == 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    if abs + rows * cols > bytes.len()
        || row_scale.len() < rows
        || pre.len() < b * cols
        || out.len() < b * rows
    {
        return false;
    }
    let full_quant = &bytes[abs..abs + rows * cols];
    dispatch_matmat(
        c,
        Some((bytes.as_ptr() as usize, idx)),
        full_quant,
        row_scale,
        pre,
        b,
        rows,
        cols,
        out,
    )
}

/// Batched q1 GEMM (prefill): resident 1-bit weight, batch of raw-f32 inputs,
/// one 2D dispatch of q1_mul_mm, one readback. cols must be a 64-multiple (the
/// q1 format packs whole tile-pairs). Weights resident + cached; x through the
/// pooled scratch.
pub fn q1_matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 64 != 0 || rows == 0 || b == 0 || pre.len() < b * cols || out.len() < b * rows {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else { return false };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    if abs + plen > bytes.len() {
        return false;
    }
    let Some(w) = weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen]) else {
        return false; // over VRAM budget → CPU path
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(&c.device, &mut sc.xs, (b * cols * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, "q1mm-xs");
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&pre[..b * cols]));
    let y_size = (b * rows * 4) as u64;
    let y_buf = Scratch::ensure(&c.device, &mut sc.y, y_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, "q1mm-y");
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, b as u32, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q1mm-bg"),
        layout: &c.layout_q1mm, // q1_mul_mm omits binding 2 (no row-scale)
        entries: &[bind_buf(0, &w), bind_buf(1, &xs_buf), bind_buf(3, &y_buf), bind_buf(4, &p_buf)],
    });
    let stage_buf = Scratch::ensure(&c.device, &mut sc.stage, y_size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "q1mm-stage");
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q1mm") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("q1mm"), timestamp_writes: None });
        pass.set_pipeline(&c.q1_mm);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(64).min(MAX_WG), (b as u32).div_ceil(64), 1);
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..b * rows]);
    drop(sc);
    ok
}

/// matmat kernel: resident weights + rs + batch of inputs, 2D dispatch, readback.
#[allow(clippy::too_many_arguments)]
fn dispatch_matmat(
    c: &Ctx,
    weight_key: Option<(usize, usize)>,
    full_quant: &[u8],
    row_scale: &[f32],
    pre: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    if full_quant.len() < rows * cols
        || row_scale.len() < rows
        || pre.len() < b * cols
        || out.len() < b * rows
    {
        return false;
    }
    let q_buf = match weight_key {
        Some(k) => match weight_buffer(c, k, full_quant) {
            Some(b) => b,
            None => return false, // over VRAM budget — honest CPU path
        },
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mm-weights"),
            contents: full_quant,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    // rs cached per tensor (row0 sentinel = full-tensor scales).
    let rs_buf = match weight_key {
        Some((base, idx)) => c
            .rs_bufs
            .lock()
            .unwrap()
            .entry((base ^ idx.wrapping_mul(1_000_003), usize::MAX))
            .or_insert_with(|| {
                crate::gpu::probe_note_cold();
                c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mm-rs"),
                    contents: bytemuck::cast_slice(&row_scale[..rows]),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .clone(),
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mm-rs"),
            contents: bytemuck::cast_slice(&row_scale[..rows]),
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    // Pooled scratch for the whole op (encode → submit → poll).
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "mm-xs",
    );
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&pre[..b * cols]));
    let y_size = (b * rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "mm-y",
    );
    let params = [(cols / 4) as u32, rows as u32, b as u32, 0u32];
    let p_buf = match &sc.params {
        Some(bf) => bf.clone(),
        None => {
            let bf = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mm-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(bf.clone());
            bf
        }
    };
    c.queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "mm-stage",
    );
    let use_mm = b >= 32;
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mm-bg"),
        // Auto bind-group layouts are pipeline-exclusive in wgpu — pick
        // the layout of the pipeline this dispatch actually uses.
        layout: if use_mm { &c.layout_mmm } else { &c.layout_mm },
        entries: &[
            bind_buf(0, &q_buf),
            bind_buf(1, &xs_buf),
            bind_buf(2, &rs_buf),
            bind_buf(3, &y_buf),
            bind_buf(4, &p_buf),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mm") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mm"),
            timestamp_writes: None,
        });
        if use_mm {
            pass.set_pipeline(&c.mul_mm);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                (rows as u32).div_ceil(64).min(MAX_WG),
                (b as u32).div_ceil(64),
                1,
            );
        } else {
            pass.set_pipeline(&c.matmat);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((rows as u32).min(MAX_WG), b as u32, 1);
        }
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..b * rows]);
    drop(sc);
    ok
}

/// q1t batched GEMM (prefill) on wgpu — register-blocked base GEMM then the
/// sparse overlay, two passes in one encoder. Raw f32 x, scales in the tiles.
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
    let gpr = cols / 32;
    if cols % 32 != 0 || rows == 0 || b == 0 {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else { return false };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    if plen < rows * gpr * 9 || abs + plen > bytes.len() || xs.len() < b * cols || out.len() < b * rows {
        return false;
    }
    dispatch_q1t_mm(c, Some((bytes.as_ptr() as usize, idx)), &bytes[abs..abs + plen], xs, b, rows, cols, out)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_q1t_mm(
    c: &Ctx,
    weight_key: Option<(usize, usize)>,
    payload: &[u8],
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let q_buf = match weight_key {
        Some(k) => match weight_buffer(c, k, payload) {
            Some(bf) => bf,
            None => return false,
        },
        None => c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q1tmm-weights"),
            contents: payload,
            usage: wgpu::BufferUsages::STORAGE,
        }),
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q1tmm-xs",
    );
    c.queue.write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..b * cols]));
    let y_size = (b * rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q1tmm-y",
    );
    let params = [(cols / 4) as u32, rows as u32, b as u32, 0u32];
    let p_buf = match &sc.params {
        Some(bf) => bf.clone(),
        None => {
            let bf = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q1tmm-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(bf.clone());
            bf
        }
    };
    c.queue.write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q1tmm-stage",
    );
    let entries = [
        bind_buf(0, &q_buf),
        bind_buf(1, &xs_buf),
        bind_buf(2, &y_buf),
        bind_buf(3, &p_buf),
    ];
    let bind_mm = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q1tmm-bg"),
        layout: &c.q1t_mm.get_bind_group_layout(0),
        entries: &entries,
    });
    let bind_ov = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q1tov-bg"),
        layout: &c.q1t_ovmm.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q1tmm") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q1tmm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q1t_mm);
        pass.set_bind_group(0, &bind_mm, &[]);
        pass.dispatch_workgroups(
            (rows as u32).div_ceil(64).min(MAX_WG),
            (b as u32).div_ceil(64),
            1,
        );
    }
    {
        // Separate pass = a barrier, so the overlay reads the finished base.
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q1tov"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q1t_ovmm);
        pass.set_bind_group(0, &bind_ov, &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(64).min(MAX_WG), 1, 1);
    }
    let ok = readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..b * rows]);
    drop(sc);
    ok
}

/// Copy the output buffer GPU→staging→CPU (map+poll). Single readback path
/// for matvec/matmat.
fn readback(
    c: &Ctx,
    mut enc: wgpu::CommandEncoder,
    y_buf: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    y_size: u64,
    out: &mut [f32],
) -> bool {
    enc.copy_buffer_to_buffer(y_buf, 0, staging, 0, y_size);
    c.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..y_size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    {
        let Ok(data) = slice.get_mapped_range() else { return false };
        out.copy_from_slice(bytemuck::cast_slice(&data[..out.len() * 4]));
    }
    staging.unmap();
    true
}

fn bind_buf(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn storage_bytes(c: &Ctx, data: &[u8]) -> wgpu::Buffer {
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn uniform_u32x4(c: &Ctx, v: [u32; 4]) -> wgpu::Buffer {
    // Content-keyed cache: these params (rows/cols/flags) repeat every token,
    // so build each once and clone the handle thereafter.
    let mut u = c.uniforms.lock().unwrap();
    if let Some(b) = u.get(&v) {
        return b.clone();
    }
    let b = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&v),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    u.insert(v, b.clone());
    b
}

fn rw_f32(c: &Ctx, n: usize, copy_src: bool) -> wgpu::Buffer {
    let usage = if copy_src {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
    } else {
        wgpu::BufferUsages::STORAGE
    };
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * 4) as u64,
        usage,
        mapped_at_creation: false,
    })
}

/// Resident quant weights of tensor `idx` (the whole tensor, cached by (file,idx)).
fn tensor_weight(c: &Ctx, model: &Arc<CmfModel>, idx: usize, rows: usize, cols: usize) -> Option<wgpu::Buffer> {
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return None;
    }
    let abs = model.entry_abs_offset(entry)?;
    let bytes = model.primary_bytes();
    if abs + rows * cols > bytes.len() {
        return None;
    }
    weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + rows * cols])
}

/// Encodes q8-matvec (row0=0) into the given encoder, writes to `y`. The bind
/// group and uniform are ref-counted by the command buffer until submit.
fn encode_matvec(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    rs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) {
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout,
        entries: &[
            bind_buf(0, weight),
            bind_buf(1, xs),
            bind_buf(2, rs),
            bind_buf(3, y),
            bind_buf(4, &p_buf),
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.matvec);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
}

/// q1 cousin of `encode_matvec`: the q1 pipeline + `layout_q1` (4 bindings,
/// no row-scale — q1 carries its scales inside the tiles). params = the
/// `dispatch_q1` layout `[gpr/2, rows, 0, 0]`. Lets q1 QKV share one encoder.
fn encode_matvec_q1(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) {
    let gpr = cols / 32;
    let p_buf = uniform_u32x4(c, [(gpr / 2) as u32, rows as u32, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_q1,
        entries: &[
            bind_buf(0, weight),
            bind_buf(1, xs),
            bind_buf(2, y),
            bind_buf(3, &p_buf),
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.q1);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).div_ceil(8).min(MAX_WG), 1, 1);
}

/// Encode a resident q1 GEMM (batched prefill): Y[k,rows] = X[k,cols] @ Wᵀ, all
/// buffers already on the device. q1_mul_mm omits binding 2 (no row scale).
#[allow(dead_code)] // wired by forward_batch_graph (batched prefill, in progress)
fn encode_q1_mm(c: &Ctx, enc: &mut wgpu::CommandEncoder, weight: &wgpu::Buffer, xs: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, cols: usize, k: usize) {
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, k as u32, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &c.layout_q1mm,
        entries: &[bind_buf(0, weight), bind_buf(1, xs), bind_buf(3, y), bind_buf(4, &p_buf)],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(&c.q1_mm);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).div_ceil(64).min(MAX_WG), (k as u32).div_ceil(64), 1);
}

/// Encode a resident q8 GEMM (int8 weight + per-row f32 scale) into `enc`.
#[allow(dead_code)] // wired by forward_batch_graph (batched prefill, in progress)
fn encode_q8_mm(c: &Ctx, enc: &mut wgpu::CommandEncoder, weight: &wgpu::Buffer, rs: &wgpu::Buffer, xs: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, cols: usize, k: usize) {
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, k as u32, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &c.layout_mmm,
        entries: &[bind_buf(0, weight), bind_buf(1, xs), bind_buf(2, rs), bind_buf(3, y), bind_buf(4, &p_buf)],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(&c.mul_mm);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).div_ceil(64).min(MAX_WG), (k as u32).div_ceil(64), 1);
}

/// Encode a plain f32 matvec (small unquantized projections) into `enc`.
fn encode_f32matvec(c: &Ctx, enc: &mut wgpu::CommandEncoder, weight: &wgpu::Buffer, xs: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, cols: usize) {
    let p_buf = uniform_u32x4(c, [cols as u32, rows as u32, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_f32,
        entries: &[bind_buf(0, weight), bind_buf(1, xs), bind_buf(2, y), bind_buf(3, &p_buf)],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(&c.f32_matvec);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
}

/// Encode a q4_tiled or q1t matvec into `enc` (same 4-slot layout as q1, but
/// params are [gpr, rows, cols]; q1t reads its sparse overlay from the tail of
/// the same buffer). `pipeline` is c.q4b or c.q1t.
fn encode_q1t_like(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) {
    let gpr = cols / 32;
    let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
    let layout = pipeline.get_bind_group_layout(0);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[bind_buf(0, weight), bind_buf(1, xs), bind_buf(2, y), bind_buf(3, &p_buf)],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
}

/// Fused SiLU(gate)·up → Q4Block down-proj: one dispatch instead of silu + matvec.
fn encode_silu_down(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) {
    let gpr = cols / 32;
    let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_silu_down,
        entries: &[bind_buf(0, weight), bind_buf(1, gate), bind_buf(2, up), bind_buf(3, y), bind_buf(4, &p_buf)],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.silu_down);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
}

/// q1 batched matvec: N q1 projections (e.g. QKV) in ONE submit + one
/// readback — the chain-fusion that `matvec_batch` does for q8, now for
/// 1-bit weights. Bails to `false` (→ CPU) on any budget/shape refusal so
/// the caller's fallback stays intact.
fn matvec_batch_q1(model: &Arc<CmfModel>, jobs: &[BatchJob], out: &mut [&mut [f32]]) -> bool {
    let Some(c) = ctx() else { return false };
    let bytes = model.primary_bytes();
    // Resident weight per job (VRAM cache; over-budget/oob → honest CPU).
    let mut weights = Vec::with_capacity(jobs.len());
    for j in jobs {
        let gpr = j.cols / 32;
        if j.rows == 0 || j.cols % 32 != 0 || gpr % 2 != 0 || j.xs.len() < j.cols {
            return false;
        }
        let entry = &model.tensors[j.idx];
        if entry.shape.first().copied().unwrap_or(0) < j.rows {
            return false;
        }
        let Some(abs) = model.entry_abs_offset(entry) else {
            return false;
        };
        let plen = j.rows * gpr * 6;
        if abs + plen > bytes.len() {
            return false;
        }
        let Some(w) = weight_buffer(c, (bytes.as_ptr() as usize, j.idx), &bytes[abs..abs + plen])
        else {
            return false;
        };
        weights.push(w);
    }
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q1-batch") });
    let mut y_bufs = Vec::with_capacity(jobs.len());
    for (j, w) in jobs.iter().zip(&weights) {
        let xs_b = storage_bytes(c, bytemuck::cast_slice(&j.xs[..j.cols]));
        let y_b = rw_f32(c, j.rows, true);
        encode_matvec_q1(c, &mut enc, w, &xs_b, &y_b, j.rows, j.cols);
        y_bufs.push(y_b);
    }
    // ONE pooled staging buffer for all outputs, one map (mirror the q8 path).
    let total: u64 = jobs.iter().map(|j| (j.rows * 4) as u64).sum();
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        total,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q1-batch-stage",
    );
    let mut off = 0u64;
    for (y_b, j) in y_bufs.iter().zip(jobs) {
        enc.copy_buffer_to_buffer(y_b, 0, &stage, off, (j.rows * 4) as u64);
        off += (j.rows * 4) as u64;
    }
    c.queue.submit(Some(enc.finish()));
    stage.slice(..total).map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    {
        let Ok(data) = stage.slice(..total).get_mapped_range() else {
            return false;
        };
        let mut off = 0usize;
        for (j, o) in jobs.iter().zip(out.iter_mut()) {
            o[..j.rows].copy_from_slice(bytemuck::cast_slice(&data[off..off + j.rows * 4]));
            off += j.rows * 4;
        }
    }
    stage.unmap();
    drop(sc);
    true
}

/// Layer MoE-FFN in a single submission: for each expert gate/up-matvec →
/// silu·mul·col_down → down-matvec → y += w·d. Intermediate buffers are
/// GPU-resident, one sync per layer.
pub fn moe_block(model: &Arc<CmfModel>, jobs: &[MoeJob], out: &mut [f32]) -> bool {
    if jobs.iter().any(|j| j.q1) {
        return false; // q1 WGSL kernel not implemented yet — honest CPU
    }
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() {
        return false;
    }
    let inter = jobs[0].gate.1;
    let hidden = jobs[0].down.1;
    if out.len() != hidden {
        return false;
    }
    // Resident weights of all triples — validate first (fail → CPU entirely).
    let mut w3 = Vec::with_capacity(jobs.len());
    for j in jobs {
        let (gi, gr, gc, _) = j.gate;
        let (ui, ur, uc, _) = j.up;
        let (di, dr, dc, _) = j.down;
        if gc % 4 != 0 || uc % 4 != 0 || dc % 4 != 0 {
            return false;
        }
        let (Some(gw), Some(uw), Some(dw)) = (
            tensor_weight(c, model, gi, gr, gc),
            tensor_weight(c, model, ui, ur, uc),
            tensor_weight(c, model, di, dr, dc),
        ) else {
            return false;
        };
        w3.push((gw, uw, dw));
    }

    let g_buf = rw_f32(c, inter, false);
    let u_buf = rw_f32(c, inter, false);
    let a_buf = rw_f32(c, inter, false);
    let d_buf = rw_f32(c, hidden, false);
    let y_buf = rw_f32(c, hidden, true);

    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("moe") });

    // y = 0
    {
        let np = uniform_u32x4(c, [hidden as u32, 0, 0, 0]);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.layout_zero,
            entries: &[bind_buf(0, &y_buf), bind_buf(1, &np)],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.zero);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((hidden as u32).div_ceil(256), 1, 1);
    }

    for (j, (gw, uw, dw)) in jobs.iter().zip(&w3) {
        let (_, gr, gc, grs) = &j.gate;
        let (_, ur, uc, urs) = &j.up;
        let (_, dr, dc, drs) = &j.down;
        // Per-tensor scale/col buffers are stable across tokens — cache
        // them like the matvec row-scales instead of re-uploading.
        let mut rs_map = c.rs_bufs.lock().unwrap();
        let mut cached = |tag: usize, idx: usize, data: &[f32]| -> wgpu::Buffer {
            rs_map
                .entry((idx.wrapping_mul(1_000_003) ^ tag, usize::MAX - 1))
                .or_insert_with(|| {
                    crate::gpu::probe_note_cold();
                    storage_bytes(c, bytemuck::cast_slice(data))
                })
                .clone()
        };
        let grs_b = cached(1, j.gate.0, grs);
        let urs_b = cached(2, j.up.0, urs);
        let drs_b = cached(3, j.down.0, drs);
        let has_col = !j.down_col.is_empty();
        let col_b = if has_col {
            cached(4, j.down.0, j.down_col)
        } else {
            cached(5, usize::MAX, &[0f32]) // dummy, gated by f=0
        };
        drop(rs_map);
        let xsg = storage_bytes(c, bytemuck::cast_slice(&j.xs_gate));
        let xsu = storage_bytes(c, bytemuck::cast_slice(&j.xs_up));

        encode_matvec(c, &mut enc, gw, &xsg, &grs_b, &g_buf, *gr, *gc);
        encode_matvec(c, &mut enc, uw, &xsu, &urs_b, &u_buf, *ur, *uc);
        // act = silu(g)·u·col_down
        {
            let np = uniform_u32x4(c, [inter as u32, has_col as u32, 0, 0]);
            let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &c.layout_silu,
                entries: &[
                    bind_buf(0, &g_buf),
                    bind_buf(1, &u_buf),
                    bind_buf(2, &col_b),
                    bind_buf(3, &a_buf),
                    bind_buf(4, &np),
                ],
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.silu);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((inter as u32).div_ceil(256), 1, 1);
        }
        encode_matvec(c, &mut enc, dw, &a_buf, &drs_b, &d_buf, *dr, *dc);
        // y += w·d
        {
            let wp = uniform_u32x4(c, [j.w.to_bits(), hidden as u32, 0, 0]);
            let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &c.layout_axpy,
                entries: &[bind_buf(0, &d_buf), bind_buf(1, &y_buf), bind_buf(2, &wp)],
            });
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.axpy);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((hidden as u32).div_ceil(256), 1, 1);
        }
    }
    // Hold the scratch lock across the readback: with concurrent server
    // slots two ops must not share the staging buffer mid-flight.
    let mut sc = c.scratch.lock().unwrap();
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (hidden * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "moe-stage",
    );
    let ok = readback(c, enc, &y_buf, &stage_buf, (hidden * 4) as u64, out);
    drop(sc);
    ok
}

/// N independent q8-matvec (GDN projections of one input) in a single submission.
pub fn matvec_batch(model: &Arc<CmfModel>, jobs: &[BatchJob], out: &mut [&mut [f32]]) -> bool {
    let Some(c) = ctx() else { return false };
    if jobs.is_empty() || jobs.len() != out.len() {
        return false;
    }
    // q1 jobs carry tile-embedded scales (empty row_scale) and need the q1
    // pipeline — route the whole batch to the q1 encoder. Mixed batches
    // (shouldn't happen: QKV share a dtype) fall to the CPU path.
    let n_q1 = jobs.iter().filter(|j| j.q1).count();
    if n_q1 == jobs.len() {
        return matvec_batch_q1(model, jobs, out);
    }
    if n_q1 != 0 {
        return false;
    }
    let mut weights = Vec::with_capacity(jobs.len());
    for j in jobs {
        if j.cols % 4 != 0 {
            return false;
        }
        let Some(w) = tensor_weight(c, model, j.idx, j.rows, j.cols) else {
            return false;
        };
        weights.push(w);
    }
    let mut y_bufs = Vec::with_capacity(jobs.len());
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") });
    for (j, w) in jobs.iter().zip(&weights) {
        let rs_b = storage_bytes(c, bytemuck::cast_slice(j.row_scale));
        let xs_b = storage_bytes(c, bytemuck::cast_slice(&j.xs));
        let y_b = rw_f32(c, j.rows, true);
        encode_matvec(c, &mut enc, w, &xs_b, &rs_b, &y_b, j.rows, j.cols);
        y_bufs.push(y_b);
    }
    // ONE pooled staging buffer for all outputs (per-job offsets),
    // one map — instead of N fresh MAP_READ allocations per call.
    let total: u64 = jobs.iter().map(|j| (j.rows * 4) as u64).sum();
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        total,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "batch-stage",
    );
    let mut off = 0u64;
    for (y_b, j) in y_bufs.iter().zip(jobs) {
        enc.copy_buffer_to_buffer(y_b, 0, &stage, off, (j.rows * 4) as u64);
        off += (j.rows * 4) as u64;
    }
    c.queue.submit(Some(enc.finish()));
    stage.slice(..total).map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    {
        let Ok(data) = stage.slice(..total).get_mapped_range() else { return false };
        let mut off = 0usize;
        for (j, o) in jobs.iter().zip(out.iter_mut()) {
            o[..j.rows]
                .copy_from_slice(bytemuck::cast_slice(&data[off..off + j.rows * 4]));
            off += j.rows * 4;
        }
    }
    stage.unmap();
    drop(sc);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_q8_matvec_matches_cpu_reference() {
        // Force the wgpu path on (Metal-via-wgpu locally; Vulkan on the server).
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping parity test");
            return;
        };
        let (rows, cols) = (256usize, 64usize); // cols % 4 == 0
        // Synthetic int8 weights + row scales + pre-scaled activations.
        let mut q = vec![0i8; rows * cols];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (((i * 37 + 11) % 255) as i32 - 127) as i8;
        }
        let rs: Vec<f32> = (0..rows).map(|r| 0.01 + (r % 7) as f32 * 0.003).collect();
        let xs: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        // CPU reference: y[o] = rs[o] * Σ q[o,i]·xs[i].
        let mut want = vec![0f32; rows];
        for o in 0..rows {
            let mut acc = 0f32;
            for i in 0..cols {
                acc += q[o * cols + i] as f32 * xs[i];
            }
            want[o] = acc * rs[o];
        }

        let qbytes: &[u8] = bytemuck::cast_slice(&q);
        let mut got = vec![0f32; rows];
        assert!(dispatch_matvec(c, None, qbytes, 0, &rs, &xs, rows, cols, &mut got));

        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q8_matvec ≠ CPU: max|Δ| = {max_d}");

        // Also check the row0 offset: the range [rows/2, rows) of the full
        // tensor must match the tail of the reference.
        let r0 = rows / 2;
        let mut got2 = vec![0f32; rows - r0];
        assert!(dispatch_matvec(
            c, None, qbytes, r0, &rs[r0..], &xs, rows - r0, cols, &mut got2
        ));
        let max_d2 = want[r0..]
            .iter()
            .zip(&got2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d2 < 1e-3, "wgpu row0 offset ≠ CPU: max|Δ| = {max_d2}");
    }

    /// Quantifies the whole-token-graph ceiling on THIS device: K chained
    /// matvecs run as K separate submit+readback ops (today's per-op path)
    /// vs the same K dispatches in ONE command buffer with a single readback
    /// (intermediates stay on the GPU — what the graph does). The ratio is how
    /// much the submit/PCIe-readback wall is costing per token.
    /// Run: `CMF_GPU=wgpu cargo test -p cortiq-engine --release --features gpu
    ///       --test-threads 1 wgpu_chain_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn wgpu_chain_probe() {
        use std::time::Instant;
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping");
            return;
        };
        let n: usize = std::env::var("CMF_CHAIN_N").ok().and_then(|v| v.parse().ok()).unwrap_or(896);
        let k: usize = std::env::var("CMF_CHAIN_K").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
        assert!(n % 4 == 0);
        // Resident n×n q8 weights + row scales (values irrelevant — timing only).
        let q = vec![1i8; n * n];
        let w = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe-w"),
            contents: bytemuck::cast_slice(&q),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rs = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe-rs"),
            contents: bytemuck::cast_slice(&vec![1f32; n]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let p = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe-p"),
            contents: bytemuck::cast_slice(&[(n / 4) as u32, n as u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mkbuf = |lbl| {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(lbl),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let a = mkbuf("probe-a");
        let b = mkbuf("probe-b");
        c.queue.write_buffer(&a, 0, bytemuck::cast_slice(&vec![0.01f32; n]));
        let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe-stage"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = |xs: &wgpu::Buffer, y: &wgpu::Buffer| {
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe-bg"),
                layout: &c.layout,
                entries: &[bind_buf(0, &w), bind_buf(1, xs), bind_buf(2, &rs), bind_buf(3, y), bind_buf(4, &p)],
            })
        };
        let bg_ab = bg(&a, &b);
        let bg_ba = bg(&b, &a);
        let wg = (n as u32).min(MAX_WG);
        let readback = |buf: &wgpu::Buffer, enc: wgpu::CommandEncoder| {
            let mut enc = enc;
            enc.copy_buffer_to_buffer(buf, 0, &stage, 0, (n * 4) as u64);
            c.queue.submit(Some(enc.finish()));
            stage.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            let _ = stage.slice(..).get_mapped_range();
            stage.unmap();
        };
        let dispatch = |enc: &mut wgpu::CommandEncoder, even: bool| {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&c.matvec);
            pass.set_bind_group(0, if even { &bg_ab } else { &bg_ba }, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        };
        // Warm.
        for _ in 0..3 {
            let mut e = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch(&mut e, true);
            readback(&b, e);
        }
        // Per-op: K submits + K readbacks.
        let t = Instant::now();
        for i in 0..k {
            let mut e = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch(&mut e, i % 2 == 0);
            readback(if i % 2 == 0 { &b } else { &a }, e);
        }
        let per_op = t.elapsed().as_secs_f64();
        // Fused: K dispatches, ONE submit + ONE readback.
        let t = Instant::now();
        let mut e = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for i in 0..k {
            dispatch(&mut e, i % 2 == 0);
        }
        readback(if (k - 1) % 2 == 0 { &b } else { &a }, e);
        let fused = t.elapsed().as_secs_f64();
        eprintln!(
            "CHAIN PROBE n={n} k={k}: per-op {:.2} ms ({:.3} ms/op) | fused {:.2} ms | speedup {:.2}× | submit+readback wall ≈ {:.3} ms/op",
            per_op * 1e3,
            per_op * 1e3 / k as f64,
            fused * 1e3,
            per_op / fused,
            (per_op - fused) * 1e3 / (k - 1) as f64,
        );
    }

    #[test]
    fn wgpu_q1_matvec_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping q1 parity test");
            return;
        };
        let (rows, cols) = (33usize, 256usize); // gpr = 8 (even), odd rows
        let gpr = cols / 32;
        let mut payload = Vec::new();
        for t in 0..rows * gpr {
            let sc = 0.005 + (t % 9) as f32 * 0.004;
            payload.extend_from_slice(&cortiq_core::quant::f32_to_f16(sc).to_le_bytes());
            for j in 0..4 {
                payload.push(((t * 41 + j * 71 + 13) % 253) as u8);
            }
        }
        let xs: Vec<f32> = (0..cols).map(|i| ((i * 7 + 3) % 29) as f32 / 29.0 - 0.5).collect();
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q1(&payload, &mut w);
        let mut want = vec![0f32; rows];
        for o in 0..rows {
            want[o] = (0..cols).map(|i| w[o * cols + i] * xs[i]).sum();
        }
        let mut got = vec![0f32; rows];
        assert!(dispatch_q1(c, None, &payload, &xs, rows, cols, &mut got));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q1_matvec ≠ CPU: max|Δ| = {max_d}");
    }

    #[test]
    fn wgpu_rmsnorm_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        if ctx().is_none() {
            eprintln!("no wgpu adapter — skipping rmsnorm parity test");
            return;
        }
        let n = 896usize;
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 101) as f32 / 101.0 - 0.5).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + ((i * 5 + 1) % 17) as f32 / 17.0).collect();
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / n as f32 + eps).sqrt();
        // plain RMSNorm
        let want: Vec<f32> = (0..n).map(|i| x[i] * inv * w[i]).collect();
        let mut got = vec![0f32; n];
        assert!(rmsnorm_row(&x, &w, &mut got, false, eps));
        let md = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(md < 1e-4, "wgpu rmsnorm ≠ CPU: max|Δ| = {md}");
        // gemma variant: w' = 1 + w
        let wantg: Vec<f32> = (0..n).map(|i| x[i] * inv * (1.0 + w[i])).collect();
        let mut gotg = vec![0f32; n];
        assert!(rmsnorm_row(&x, &w, &mut gotg, true, eps));
        let mdg = wantg.iter().zip(&gotg).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(mdg < 1e-4, "wgpu rmsnorm(gemma) ≠ CPU: max|Δ| = {mdg}");
    }

    #[test]
    fn wgpu_add_rmsnorm_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping");
            return;
        };
        let n = 896usize;
        let eps = 1e-6f32;
        let h: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 101) as f32 / 101.0 - 0.5).collect();
        let d: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 61) as f32 / 61.0 - 0.5).collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + ((i * 5 + 1) % 17) as f32 / 17.0).collect();
        // CPU reference: h += d, then rmsnorm(h, w)
        let hd: Vec<f32> = (0..n).map(|i| h[i] + d[i]).collect();
        let ss: f32 = hd.iter().map(|x| x * x).sum();
        let inv = 1.0 / (ss / n as f32 + eps).sqrt();
        let want: Vec<f32> = (0..n).map(|i| hd[i] * inv * w[i]).collect();
        // GPU add_rmsnorm
        let hb = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&h),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let db = storage_bytes(c, bytemuck::cast_slice(&d));
        let wb = storage_bytes(c, bytemuck::cast_slice(&w));
        let ob = rw_f32(c, n, true);
        let pb = uniform_u32x4(c, [n as u32, 0, eps.to_bits(), 0]);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.layout_add_rmsnorm,
            entries: &[bind_buf(0, &hb), bind_buf(1, &db), bind_buf(2, &wb), bind_buf(3, &ob), bind_buf(4, &pb)],
        });
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            p.set_pipeline(&c.add_rmsnorm);
            p.set_bind_group(0, &bind, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        let mut got = vec![0f32; n];
        let sz = (n * 4) as u64;
        let mut sc = c.scratch.lock().unwrap();
        let stage = Scratch::ensure(&c.device, &mut sc.stage, sz, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "arn-stage");
        assert!(readback(c, enc, &ob, &stage, sz, &mut got));
        drop(sc);
        let md = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(md < 1e-4, "wgpu add_rmsnorm ≠ CPU: max|Δ| = {md}");
    }

    #[test]
    fn wgpu_attn_rope_qkn_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        if ctx().is_none() {
            eprintln!("no wgpu adapter — skipping attn_rope parity test");
            return;
        }
        // head_dim 256 with partial RoPE (rd=64) — the Qwen3.5 geometry: nt=8
        // (>4-slot xv) and hlf=32 exercise the paths that broke the graph.
        let (nh, nkv, hd, rd, pos) = (4usize, 2usize, 256usize, 64usize, 5usize);
        let eps = 1e-6f32;
        let flags = 1u32 | 2u32 | 4u32; // gate + qnorm + knorm, non-gemma
        let jitter = |a: usize, b: usize| ((a * 31 + b * 17 + 7) % 97) as f32 / 97.0 - 0.5;
        // qraw: nh heads × 2·hd (q part || gate part); k: nkv × hd
        let qraw: Vec<f32> = (0..nh * 2 * hd).map(|i| jitter(i, 1)).collect();
        let k_in: Vec<f32> = (0..nkv * hd).map(|i| jitter(i, 2)).collect();
        let qnw: Vec<f32> = (0..hd).map(|d| 0.7 + jitter(d, 3)).collect();
        let knw: Vec<f32> = (0..hd).map(|d| 0.7 + jitter(d, 4)).collect();
        let invf: Vec<f32> = (0..rd / 2).map(|i| 1.0 / (10000f32).powf(2.0 * i as f32 / rd as f32)).collect();
        // CPU reference: qk-norm then half-split partial RoPE.
        let norm_rope = |v: &mut [f32], w: &[f32]| {
            let ss: f32 = v.iter().map(|x| x * x).sum();
            let inv = 1.0 / (ss / hd as f32 + eps).sqrt();
            for d in 0..hd {
                v[d] = v[d] * inv * w[d];
            }
            let hlf = rd / 2;
            for i in 0..hlf {
                let ang = pos as f32 * invf[i];
                let (c, s) = (ang.cos(), ang.sin());
                let (x0, x1) = (v[i], v[i + hlf]);
                v[i] = x0 * c - x1 * s;
                v[i + hlf] = x0 * s + x1 * c;
            }
        };
        let mut want_q = vec![0f32; nh * hd];
        let mut want_g = vec![0f32; nh * hd];
        for h in 0..nh {
            let mut q: Vec<f32> = qraw[h * 2 * hd..h * 2 * hd + hd].to_vec();
            norm_rope(&mut q, &qnw);
            want_q[h * hd..(h + 1) * hd].copy_from_slice(&q);
            want_g[h * hd..(h + 1) * hd].copy_from_slice(&qraw[h * 2 * hd + hd..h * 2 * hd + 2 * hd]);
        }
        let mut want_k = k_in.clone();
        for kh in 0..nkv {
            let mut kk = want_k[kh * hd..(kh + 1) * hd].to_vec();
            norm_rope(&mut kk, &knw);
            want_k[kh * hd..(kh + 1) * hd].copy_from_slice(&kk);
        }
        let mut got_q = vec![0f32; nh * hd];
        let mut got_k = vec![0f32; nkv * hd];
        let mut got_g = vec![0f32; nh * hd];
        assert!(attn_rope_qkn_gpu(
            &qraw, &k_in, &qnw, &knw, &invf, nh, nkv, hd, rd, pos, flags, eps,
            &mut got_q, &mut got_k, &mut got_g,
        ));
        let md = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(md(&want_q, &got_q) < 1e-4, "q mismatch: {}", md(&want_q, &got_q));
        assert!(md(&want_k, &got_k) < 1e-4, "k mismatch: {}", md(&want_k, &got_k));
        assert!(md(&want_g, &got_g) < 1e-4, "gate mismatch: {}", md(&want_g, &got_g));
    }

    #[test]
    fn wgpu_gqa_attend_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        if ctx().is_none() {
            eprintln!("no wgpu adapter — skipping gqa_attend parity test");
            return;
        }
        // head_dim 256 (Qwen3.5): exercises the at_acc stride-257 accumulator.
        let (nh, hpk, hd, cap, n) = (4usize, 2usize, 256usize, 16usize, 5usize);
        let nkv = nh / hpk;
        let jit = |a: usize, b: usize| ((a * 29 + b * 13 + 5) % 89) as f32 / 89.0 - 0.5;
        let q: Vec<f32> = (0..nh * hd).map(|i| jit(i, 1)).collect();
        // caches laid out [nkv, cap, hd]; only first n rows are valid.
        let mut kc = vec![0f32; nkv * cap * hd];
        let mut vc = vec![0f32; nkv * cap * hd];
        for kh in 0..nkv {
            for p in 0..n {
                for d in 0..hd {
                    kc[(kh * cap + p) * hd + d] = jit(kh * 1000 + p * 10 + d, 2);
                    vc[(kh * cap + p) * hd + d] = jit(kh * 1000 + p * 10 + d, 3);
                }
            }
        }
        // CPU reference: scaled softmax attention per head.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut want = vec![0f32; nh * hd];
        for h in 0..nh {
            let kh = h / hpk;
            let mut sc: Vec<f32> = (0..n)
                .map(|p| (0..hd).map(|d| q[h * hd + d] * kc[(kh * cap + p) * hd + d]).sum::<f32>() * scale)
                .collect();
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0.0;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                den += *s;
            }
            for d in 0..hd {
                want[h * hd + d] = (0..n).map(|p| sc[p] * vc[(kh * cap + p) * hd + d]).sum::<f32>() / den;
            }
        }
        let mut got = vec![0f32; nh * hd];
        assert!(gqa_attend_gpu(&q, &kc, &vc, nh, hpk, hd, cap, n, &mut got));
        let md = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(md < 1e-4, "wgpu gqa_attend ≠ CPU: max|Δ| = {md}");
    }

    #[test]
    fn wgpu_gdn_step_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        if ctx().is_none() {
            eprintln!("no wgpu adapter — skipping gdn_step test");
            return;
        }
        let (nv, nk, dk, dv) = (4usize, 2usize, 8usize, 8usize);
        let kd = nk * dk;
        let rep = nv / nk;
        let cdim = 2 * kd + nv * dv;
        let eps = 1e-6f32;
        let jit = |a: usize, b: usize| ((a * 23 + b * 11 + 5) % 71) as f32 / 71.0 - 0.5;
        let cq: Vec<f32> = (0..cdim).map(|i| jit(i, 1)).collect();
        let z: Vec<f32> = (0..nv * dv).map(|i| jit(i, 2)).collect();
        let a: Vec<f32> = (0..nv).map(|i| jit(i, 3)).collect();
        let b: Vec<f32> = (0..nv).map(|i| jit(i, 4)).collect();
        let alog: Vec<f32> = (0..nv).map(|i| jit(i, 5) - 0.5).collect();
        let dtb: Vec<f32> = (0..nv).map(|i| jit(i, 6)).collect();
        let norm: Vec<f32> = (0..dv).map(|i| 0.8 + jit(i, 7)).collect();
        let s0: Vec<f32> = (0..nv * dk * dv).map(|i| jit(i, 8) * 0.3).collect();
        // CPU reference (mirrors linear_core::gdn_step).
        let sp = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let mut sc = s0.clone();
        let mut want = vec![0f32; nv * dv];
        for h in 0..nv {
            let ko = h / rep;
            let (qs, ks) = (ko * dk, kd + ko * dk);
            let nq: f32 = (0..dk).map(|d| cq[qs + d] * cq[qs + d]).sum();
            let nkn: f32 = (0..dk).map(|d| cq[ks + d] * cq[ks + d]).sum();
            let invq = 1.0 / ((nq + 1e-6).sqrt() * (dk as f32).sqrt());
            let invk = 1.0 / (nkn + 1e-6).sqrt();
            let qf: Vec<f32> = (0..dk).map(|d| cq[qs + d] * invq).collect();
            let kf: Vec<f32> = (0..dk).map(|d| cq[ks + d] * invk).collect();
            let g = (-(alog[h].exp()) * sp(a[h] + dtb[h])).exp();
            let beta = sig(b[h]);
            let sbase = h * dk * dv;
            let mut o = vec![0f32; dv];
            for dj in 0..dv {
                let vt = cq[2 * kd + h * dv + dj];
                let mut kv = 0.0;
                for di in 0..dk {
                    kv += sc[sbase + di * dv + dj] * kf[di];
                }
                let delta = (vt - g * kv) * beta;
                for di in 0..dk {
                    let idx = sbase + di * dv + dj;
                    let cell = g * sc[idx] + kf[di] * delta;
                    sc[idx] = cell;
                    o[dj] += qf[di] * cell;
                }
            }
            let ss: f32 = o.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / dv as f32 + eps).sqrt();
            for dj in 0..dv {
                want[h * dv + dj] = o[dj] * inv * norm[dj] * silu(z[h * dv + dj]);
            }
        }
        // GPU
        let mut sg = s0.clone();
        let mut got = vec![0f32; nv * dv];
        assert!(gdn_step_gpu(&cq, &z, &a, &b, &alog, &dtb, &norm, &mut sg, nv, dk, dv, kd, rep, cdim, eps, &mut got));
        let mo = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let msd = sc.iter().zip(&sg).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(mo < 2e-3, "wgpu gdn_step o ≠ CPU: max|Δ| = {mo}");
        assert!(msd < 2e-3, "wgpu gdn_step S ≠ CPU: max|Δ| = {msd}");
    }

    #[test]
    fn wgpu_gdn_conv_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        if ctx().is_none() {
            eprintln!("no wgpu adapter — skipping gdn_conv test");
            return;
        }
        let (cdim, kk) = (48usize, 4usize);
        let jit = |a: usize, b: usize| ((a * 19 + b * 7 + 3) % 61) as f32 / 61.0 - 0.5;
        let qkv: Vec<f32> = (0..cdim).map(|i| jit(i, 1)).collect();
        let taps: Vec<f32> = (0..cdim * kk).map(|i| jit(i, 2)).collect();
        let ring0: Vec<f32> = (0..(kk - 1) * cdim).map(|i| jit(i, 3)).collect();
        let silu = |x: f32| x / (1.0 + (-x).exp());
        // CPU reference
        let mut rc = ring0.clone();
        let mut want_cq = vec![0f32; cdim];
        for c in 0..cdim {
            let t = &taps[c * kk..(c + 1) * kk];
            let mut acc = qkv[c] * t[kk - 1];
            for j in 0..kk - 1 {
                acc += rc[j * cdim + c] * t[j];
            }
            want_cq[c] = silu(acc);
        }
        rc.copy_within(cdim.., 0);
        let tail = (kk - 2) * cdim;
        rc[tail..tail + cdim].copy_from_slice(&qkv[..cdim]);
        // GPU
        let mut rg = ring0.clone();
        let mut got_cq = vec![0f32; cdim];
        assert!(gdn_conv_gpu(&qkv, &taps, &mut rg, cdim, kk, &mut got_cq));
        let mc = want_cq.iter().zip(&got_cq).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let mr = rc.iter().zip(&rg).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(mc < 1e-5, "wgpu gdn_conv cq ≠ CPU: {mc}");
        assert!(mr < 1e-6, "wgpu gdn_conv ring ≠ CPU: {mr}");
    }

    // Build a deterministic q1 payload for a [rows, cols] weight + its dequant.
    #[cfg(test)]
    fn mk_q1(rows: usize, cols: usize, seed: usize) -> (Vec<u8>, Vec<f32>) {
        let gpr = cols / 32;
        let mut payload = Vec::new();
        for t in 0..rows * gpr {
            let sc = 0.004 + ((t + seed) % 9) as f32 * 0.003;
            payload.extend_from_slice(&cortiq_core::quant::f32_to_f16(sc).to_le_bytes());
            for j in 0..4 {
                payload.push(((t * 37 + j * 53 + seed * 7 + 11) % 251) as u8);
            }
        }
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q1(&payload, &mut w);
        (payload, w)
    }

    #[test]
    fn wgpu_attn_block_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping attn_block test");
            return;
        };
        let (nh, nkv, hd, rd, hidden, cap, stored) = (4usize, 2usize, 64usize, 64usize, 128usize, 8usize, 2usize);
        let hpk = nh / nkv;
        let eps = 1e-6f32;
        let flags = 2u32 | 4u32; // qnorm + knorm, no gate
        let jit = |a: usize, b: usize| ((a * 31 + b * 17 + 3) % 83) as f32 / 83.0 - 0.5;
        let h_in: Vec<f32> = (0..hidden).map(|i| jit(i, 1)).collect();
        let norm_w: Vec<f32> = (0..hidden).map(|i| 0.8 + jit(i, 2)).collect();
        let (wq_p, wq) = mk_q1(nh * hd, hidden, 1);
        let (wk_p, wk) = mk_q1(nkv * hd, hidden, 2);
        let (wv_p, wv) = mk_q1(nkv * hd, hidden, 3);
        let (wo_p, wo) = mk_q1(hidden, nh * hd, 4);
        let qnw: Vec<f32> = (0..hd).map(|d| 0.7 + jit(d, 5)).collect();
        let knw: Vec<f32> = (0..hd).map(|d| 0.7 + jit(d, 6)).collect();
        let invf: Vec<f32> = (0..rd / 2).map(|i| 1.0 / (10000f32).powf(2.0 * i as f32 / rd as f32)).collect();
        // Pre-filled device K/V caches [nkv, cap, hd] (first `stored` rows valid).
        let mut kc = vec![0f32; nkv * cap * hd];
        let mut vc = vec![0f32; nkv * cap * hd];
        for kh in 0..nkv {
            for p in 0..stored {
                for d in 0..hd {
                    kc[(kh * cap + p) * hd + d] = jit(kh * 900 + p * 30 + d, 7);
                    vc[(kh * cap + p) * hd + d] = jit(kh * 900 + p * 30 + d, 8);
                }
            }
        }
        let mkcache = |data: &[f32]| {
            c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cache"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })
        };
        let kbuf = mkcache(&kc);
        let vbuf = mkcache(&vc);
        // ---- CPU reference ----
        let ss: f32 = h_in.iter().map(|x| x * x).sum();
        let rinv = 1.0 / (ss / hidden as f32 + eps).sqrt();
        let normed: Vec<f32> = (0..hidden).map(|i| h_in[i] * rinv * norm_w[i]).collect();
        let matvec = |w: &[f32], rows: usize, cols: usize, x: &[f32]| -> Vec<f32> {
            (0..rows).map(|o| (0..cols).map(|i| w[o * cols + i] * x[i]).sum()).collect()
        };
        let qraw = matvec(&wq, nh * hd, hidden, &normed);
        let kv_k = matvec(&wk, nkv * hd, hidden, &normed);
        let kv_v = matvec(&wv, nkv * hd, hidden, &normed);
        let norm_rope = |v: &mut [f32], w: &[f32]| {
            let s: f32 = v.iter().map(|x| x * x).sum();
            let inv = 1.0 / (s / hd as f32 + eps).sqrt();
            for d in 0..hd {
                v[d] = v[d] * inv * w[d];
            }
            for i in 0..rd / 2 {
                let ang = stored as f32 * invf[i];
                let (co, si) = (ang.cos(), ang.sin());
                let (x0, x1) = (v[i], v[i + rd / 2]);
                v[i] = x0 * co - x1 * si;
                v[i + rd / 2] = x0 * si + x1 * co;
            }
        };
        let mut qout = vec![0f32; nh * hd];
        for h in 0..nh {
            let mut q = qraw[h * hd..(h + 1) * hd].to_vec();
            norm_rope(&mut q, &qnw);
            qout[h * hd..(h + 1) * hd].copy_from_slice(&q);
        }
        for kh in 0..nkv {
            let mut kk = kv_k[kh * hd..(kh + 1) * hd].to_vec();
            norm_rope(&mut kk, &knw);
            kc[(kh * cap + stored) * hd..(kh * cap + stored) * hd + hd].copy_from_slice(&kk);
            vc[(kh * cap + stored) * hd..(kh * cap + stored) * hd + hd]
                .copy_from_slice(&kv_v[kh * hd..(kh + 1) * hd]);
        }
        let n = stored + 1;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn = vec![0f32; nh * hd];
        for h in 0..nh {
            let kh = h / hpk;
            let mut sc: Vec<f32> = (0..n)
                .map(|p| (0..hd).map(|d| qout[h * hd + d] * kc[(kh * cap + p) * hd + d]).sum::<f32>() * scale)
                .collect();
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0.0;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                den += *s;
            }
            for d in 0..hd {
                attn[h * hd + d] = (0..n).map(|p| sc[p] * vc[(kh * cap + p) * hd + d]).sum::<f32>() / den;
            }
        }
        let o = matvec(&wo, hidden, nh * hd, &attn);
        let want: Vec<f32> = (0..hidden).map(|i| h_in[i] + o[i]).collect();
        // ---- GPU block ----
        let mut got = vec![0f32; hidden];
        assert!(attn_block_gpu(
            &h_in, &norm_w, &wq_p, &wk_p, &wv_p, &wo_p, &qnw, &knw, &invf,
            &kbuf, &vbuf, nh, nkv, hd, rd, hidden, cap, stored, flags, eps, &mut got,
        ));
        let md = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(md < 2e-3, "wgpu attn_block ≠ CPU: max|Δ| = {md}");
    }

    // Payoff microbench: the resident attention block (ONE submit) vs the same
    // steps as separate submit+readback ops (today's per-op decode). Run with
    //   cargo test -p cortiq-engine --release --features gpu attn_block_timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn wgpu_attn_block_timing() {
        use std::time::Instant;
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping");
            return;
        };
        // 1.7B-ish attention geometry.
        let (nh, nkv, hd, rd, hidden, cap, stored) = (16usize, 8usize, 128usize, 128usize, 2048usize, 256usize, 128usize);
        let hpk = nh / nkv;
        let eps = 1e-6f32;
        let flags = 2u32 | 4u32;
        let h_in = vec![0.01f32; hidden];
        let norm_w = vec![1.0f32; hidden];
        let (wq_p, _) = mk_q1(nh * hd, hidden, 1);
        let (wk_p, _) = mk_q1(nkv * hd, hidden, 2);
        let (wv_p, _) = mk_q1(nkv * hd, hidden, 3);
        let (wo_p, _) = mk_q1(hidden, nh * hd, 4);
        let qnw = vec![1.0f32; hd];
        let knw = vec![1.0f32; hd];
        let invf: Vec<f32> = (0..rd / 2).map(|i| 1.0 / (10000f32).powf(2.0 * i as f32 / rd as f32)).collect();
        let kc = vec![0.01f32; nkv * cap * hd];
        let vc = vec![0.01f32; nkv * cap * hd];
        let mkc = |d: &[f32]| c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(d),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let (kbuf, vbuf) = (mkc(&kc), mkc(&vc));
        let iters = 200;
        let mut hout = vec![0f32; hidden];
        // FUSED: the resident block, one submit + one readback per call.
        for _ in 0..20 {
            attn_block_gpu(&h_in, &norm_w, &wq_p, &wk_p, &wv_p, &wo_p, &qnw, &knw, &invf, &kbuf, &vbuf, nh, nkv, hd, rd, hidden, cap, stored, flags, eps, &mut hout);
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            attn_block_gpu(&h_in, &norm_w, &wq_p, &wk_p, &wv_p, &wo_p, &qnw, &knw, &invf, &kbuf, &vbuf, nh, nkv, hd, rd, hidden, cap, stored, flags, eps, &mut hout);
        }
        let fused = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        // UNFUSED: each step its own submit+readback (rmsnorm, QKV×3, rope, attend, O).
        let mut normed = vec![0f32; hidden];
        let mut qraw = vec![0f32; nh * hd];
        let mut kk = vec![0f32; nkv * hd];
        let mut vv = vec![0f32; nkv * hd];
        let mut qout = vec![0f32; nh * hd];
        let mut kout = vec![0f32; nkv * hd];
        let mut gout = vec![0f32; nh * hd];
        let mut attn = vec![0f32; nh * hd];
        let mut oout = vec![0f32; hidden];
        let unfused_once = |normed: &mut [f32], qraw: &mut [f32], kk: &mut [f32], vv: &mut [f32], qout: &mut [f32], kout: &mut [f32], gout: &mut [f32], attn: &mut [f32], oout: &mut [f32]| {
            rmsnorm_row(&h_in, &norm_w, normed, false, eps);
            dispatch_q1(c, None, &wq_p, normed, nh * hd, hidden, qraw);
            dispatch_q1(c, None, &wk_p, normed, nkv * hd, hidden, kk);
            dispatch_q1(c, None, &wv_p, normed, nkv * hd, hidden, vv);
            attn_rope_qkn_gpu(qraw, kk, &qnw, &knw, &invf, nh, nkv, hd, rd, stored, flags, eps, qout, kout, gout);
            gqa_attend_gpu(qout, &kc, &vc, nh, hpk, hd, cap, stored + 1, attn);
            dispatch_q1(c, None, &wo_p, attn, hidden, nh * hd, oout);
        };
        for _ in 0..20 {
            unfused_once(&mut normed, &mut qraw, &mut kk, &mut vv, &mut qout, &mut kout, &mut gout, &mut attn, &mut oout);
        }
        let t1 = Instant::now();
        for _ in 0..iters {
            unfused_once(&mut normed, &mut qraw, &mut kk, &mut vv, &mut qout, &mut kout, &mut gout, &mut attn, &mut oout);
        }
        let unfused = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        eprintln!(
            "ATTN BLOCK 1.7B-dims: fused(1 submit) {fused:.3} ms/layer | unfused(per-op) {unfused:.3} ms/layer | speedup {:.2}×",
            unfused / fused
        );
    }

    #[test]
    fn wgpu_q1t_matvec_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping q1t parity test");
            return;
        };
        use cortiq_core::quant::{f32_to_f16, q1t_pack, GROUP_SIZE};
        let (rows, cols) = (33usize, 256usize);
        let gpr = cols / GROUP_SIZE;
        let outliers: [(usize, f32); 3] = [(5, 3.0), (300, -2.0), (600, 1.5)]; // sorted
        let is_out = |flat: usize| outliers.iter().any(|&(i, _)| i == flat);
        let mut payload = Vec::new();
        for r in 0..rows {
            for g in 0..gpr {
                let s = 0.02 + ((r + g) % 7) as f32 * 0.01;
                payload.extend_from_slice(&f32_to_f16(s).to_le_bytes());
                let mut cc = [0u8; 7];
                for k in 0..GROUP_SIZE {
                    let code = if is_out(r * cols + g * GROUP_SIZE + k) {
                        0
                    } else {
                        ((k * 7 + r + g) % 3) as u8
                    };
                    q1t_pack(&mut cc, k, code);
                }
                payload.extend_from_slice(&cc);
            }
        }
        let mut row_ptr = vec![0u32; rows + 1];
        for &(idx, _) in &outliers {
            row_ptr[idx / cols + 1] += 1;
        }
        for r in 0..rows {
            row_ptr[r + 1] += row_ptr[r];
        }
        for &p in &row_ptr {
            payload.extend_from_slice(&p.to_le_bytes());
        }
        for &(idx, v) in &outliers {
            payload.extend_from_slice(&((idx % cols) as u16).to_le_bytes());
            payload.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        let xs: Vec<f32> = (0..cols).map(|i| ((i * 7 + 3) % 29) as f32 / 29.0 - 0.5).collect();
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q1t(&payload, rows, cols, &mut w);
        let mut want = vec![0f32; rows];
        for o in 0..rows {
            want[o] = (0..cols).map(|i| w[o * cols + i] * xs[i]).sum();
        }
        let mut got = vec![0f32; rows];
        assert!(dispatch_q1t(c, &c.q1t, None, &payload, &xs, rows, cols, &mut got));
        let max_d = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_d < 1e-2, "wgpu q1t_matvec ≠ CPU: max|Δ| = {max_d}");
    }

    #[test]
    fn wgpu_q4b_matvec_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping q4b parity test");
            return;
        };
        use cortiq_core::quant::{f32_to_f16, GROUP_SIZE};
        let (rows, cols) = (33usize, 256usize);
        let n_groups = rows * (cols / GROUP_SIZE);
        let mut payload = vec![0u8; n_groups * 16]; // packed nibbles
        for g in 0..n_groups {
            for k in 0..16 {
                let lo = ((g * 3 + k) % 16) as u8;
                let hi = ((g * 5 + k * 2) % 16) as u8;
                payload[g * 16 + k] = lo | (hi << 4);
            }
        }
        for g in 0..n_groups {
            let s = 0.02 + (g % 7) as f32 * 0.01;
            payload.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        }
        let xs: Vec<f32> = (0..cols).map(|i| ((i * 7 + 3) % 29) as f32 / 29.0 - 0.5).collect();
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q4_block(&payload, &mut w);
        let mut want = vec![0f32; rows];
        for o in 0..rows {
            want[o] = (0..cols).map(|i| w[o * cols + i] * xs[i]).sum();
        }
        let mut got = vec![0f32; rows];
        assert!(dispatch_q1t(c, &c.q4b, None, &payload, &xs, rows, cols, &mut got));
        let max_d = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_d < 1e-2, "wgpu q4b_matvec ≠ CPU: max|Δ| = {max_d}");
    }

    #[test]
    fn wgpu_q1t_matmat_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping q1t GEMM parity test");
            return;
        };
        use cortiq_core::quant::{f32_to_f16, q1t_pack, GROUP_SIZE};
        let (b, rows, cols) = (40usize, 64usize, 256usize);
        let gpr = cols / GROUP_SIZE;
        let outliers: [(usize, f32); 4] = [(5, 3.0), (300, -2.0), (600, 1.5), (2000, -1.0)];
        let is_out = |flat: usize| outliers.iter().any(|&(i, _)| i == flat);
        let mut payload = Vec::new();
        for r in 0..rows {
            for g in 0..gpr {
                let s = 0.02 + ((r + g) % 7) as f32 * 0.01;
                payload.extend_from_slice(&f32_to_f16(s).to_le_bytes());
                let mut cc = [0u8; 7];
                for k in 0..GROUP_SIZE {
                    let code = if is_out(r * cols + g * GROUP_SIZE + k) {
                        0
                    } else {
                        ((k * 7 + r + g) % 3) as u8
                    };
                    q1t_pack(&mut cc, k, code);
                }
                payload.extend_from_slice(&cc);
            }
        }
        let mut row_ptr = vec![0u32; rows + 1];
        for &(idx, _) in &outliers {
            row_ptr[idx / cols + 1] += 1;
        }
        for r in 0..rows {
            row_ptr[r + 1] += row_ptr[r];
        }
        for &p in &row_ptr {
            payload.extend_from_slice(&p.to_le_bytes());
        }
        for &(idx, v) in &outliers {
            payload.extend_from_slice(&((idx % cols) as u16).to_le_bytes());
            payload.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        let xs: Vec<f32> = (0..b * cols).map(|i| ((i * 13 + 7) % 31) as f32 / 31.0 - 0.5).collect();
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q1t(&payload, rows, cols, &mut w);
        let mut want = vec![0f32; b * rows];
        for bi in 0..b {
            for o in 0..rows {
                want[bi * rows + o] =
                    (0..cols).map(|i| w[o * cols + i] * xs[bi * cols + i]).sum();
            }
        }
        let mut got = vec![0f32; b * rows];
        assert!(dispatch_q1t_mm(c, None, &payload, &xs, b, rows, cols, &mut got));
        let max_d = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_d < 2e-2, "wgpu q1t_mul_mm ≠ CPU: max|Δ| = {max_d}");
    }

    #[test]
    fn wgpu_q8_matmat_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping matmat test");
            return;
        };
        let (rows, cols, b) = (128usize, 64usize, 5usize);
        let mut q = vec![0i8; rows * cols];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (((i * 53 + 3) % 255) as i32 - 127) as i8;
        }
        let rs: Vec<f32> = (0..rows).map(|r| 0.01 + (r % 5) as f32 * 0.004).collect();
        let pre: Vec<f32> = (0..b * cols).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        // CPU ref: out[bi, o] = rs[o]·Σ q[o,i]·pre[bi,i].
        let mut want = vec![0f32; b * rows];
        for bi in 0..b {
            for o in 0..rows {
                let mut acc = 0f32;
                for i in 0..cols {
                    acc += q[o * cols + i] as f32 * pre[bi * cols + i];
                }
                want[bi * rows + o] = acc * rs[o];
            }
        }
        let qbytes: &[u8] = bytemuck::cast_slice(&q);
        let mut got = vec![0f32; b * rows];
        assert!(dispatch_matmat(c, None, qbytes, &rs, &pre, b, rows, cols, &mut got));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q8_matmat ≠ CPU: max|Δ| = {max_d}");
    }

    /// The tiled kernel (b ≥ 32) on deliberately awkward shapes: rows
    /// not a multiple of the 64-tile, cols not a multiple of the K-step
    /// — every edge guard fires.
    #[test]
    fn wgpu_q8_mul_mm_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping mul_mm test");
            return;
        };
        let (rows, cols, b) = (100usize, 52usize, 70usize);
        let mut q = vec![0i8; rows * cols];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (((i * 31 + 7) % 255) as i32 - 127) as i8;
        }
        let rs: Vec<f32> = (0..rows).map(|r| 0.01 + (r % 7) as f32 * 0.003).collect();
        let pre: Vec<f32> = (0..b * cols).map(|i| ((i % 19) as f32 - 9.0) * 0.04).collect();
        let mut want = vec![0f32; b * rows];
        for bi in 0..b {
            for o in 0..rows {
                let mut acc = 0f32;
                for i in 0..cols {
                    acc += q[o * cols + i] as f32 * pre[bi * cols + i];
                }
                want[bi * rows + o] = acc * rs[o];
            }
        }
        let qbytes: &[u8] = bytemuck::cast_slice(&q);
        let mut got = vec![0f32; b * rows];
        assert!(dispatch_matmat(c, None, qbytes, &rs, &pre, b, rows, cols, &mut got));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q8_mul_mm ≠ CPU: max|Δ| = {max_d}");
    }

    // Tiled q1 GEMM on an awkward shape (rows/batch not 64-multiples, cols a
    // 64-multiple as the format requires): the prefill / speculative-batch path.
    #[test]
    fn wgpu_q1_mul_mm_matches_cpu_reference() {
        use cortiq_core::quant::{f16_to_f32, f32_to_f16};
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping q1_mul_mm test");
            return;
        };
        let (rows, cols, b) = (100usize, 128usize, 70usize); // cols % 64 == 0
        let np = cols / 64;
        let jit = |a: usize| ((a * 2654435761usize) >> 13) as u32; // cheap hash → bits
        // Build the q1 weight blob + a decoded f32 reference weight in lock-step.
        let mut q1w = vec![0u32; rows * np * 3];
        let mut wref = vec![0f32; rows * cols];
        for o in 0..rows {
            for pi in 0..np {
                let s0 = 0.02 + ((o * 7 + pi) % 11) as f32 * 0.005;
                let s1 = 0.03 + ((o * 3 + pi * 5) % 9) as f32 * 0.004;
                let (h0, h1) = (f32_to_f16(s0), f32_to_f16(s1));
                let (sf0, sf1) = (f16_to_f32(h0), f16_to_f32(h1));
                let bits0 = jit(o * 131 + pi * 17 + 1);
                let bits1 = jit(o * 131 + pi * 17 + 2);
                let base = o * np * 3 + pi * 3;
                q1w[base] = (h0 as u32) | ((bits0 & 0xFFFF) << 16);
                q1w[base + 1] = (bits0 >> 16) | ((h1 as u32) << 16);
                q1w[base + 2] = bits1;
                for j in 0..32usize {
                    let sgn0 = if (bits0 >> j) & 1 != 0 { sf0 } else { -sf0 };
                    let sgn1 = if (bits1 >> j) & 1 != 0 { sf1 } else { -sf1 };
                    wref[o * cols + pi * 64 + j] = sgn0;
                    wref[o * cols + pi * 64 + 32 + j] = sgn1;
                }
            }
        }
        let x: Vec<f32> = (0..b * cols).map(|i| ((i % 23) as f32 - 11.0) * 0.03).collect();
        let mut want = vec![0f32; b * rows];
        for bi in 0..b {
            for o in 0..rows {
                let mut acc = 0f32;
                for i in 0..cols {
                    acc += wref[o * cols + i] * x[bi * cols + i];
                }
                want[bi * rows + o] = acc;
            }
        }
        // GPU dispatch (inline — q1_mm is not yet wired into a public entry).
        let qbuf = storage_bytes(c, bytemuck::cast_slice(&q1w));
        let xbuf = storage_bytes(c, bytemuck::cast_slice(&x));
        let ybuf = rw_f32(c, b * rows, true);
        let pbuf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, b as u32, 0]);
        // q1_mul_mm never reads rs → its auto layout omits binding 2.
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.layout_q1mm,
            entries: &[bind_buf(0, &qbuf), bind_buf(1, &xbuf), bind_buf(3, &ybuf), bind_buf(4, &pbuf)],
        });
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&c.q1_mm);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(64), (b as u32).div_ceil(64), 1);
        }
        let mut sc = c.scratch.lock().unwrap();
        let stage = Scratch::ensure(&c.device, &mut sc.stage, (b * rows * 4) as u64, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, "q1mm-stage");
        let mut got = vec![0f32; b * rows];
        assert!(readback(c, enc, &ybuf, &stage, (b * rows * 4) as u64, &mut got));
        drop(sc);
        let max_d = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q1_mul_mm ≠ CPU: max|Δ| = {max_d}");
    }
}
