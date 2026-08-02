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

/// Subgroup-accelerated MoE select: the top-k rounds ride subgroupMax /
/// subgroupMin (barrier-free within a subgroup) — two barriers per slot
/// against the tree version's eight. Lives in its OWN module: `enable
/// subgroups` fails validation on devices without the feature, and one
/// invalid function kills every entry point of a module (0.5.40's Metal
/// lesson, re-learned on Vulkan this afternoon).
const SELECT_SG_SRC: &str = r#"
struct MoeSelP { n_exp: u32, top_k: u32, norm: u32, pk: u32 };
@group(0) @binding(0) var<storage, read>       sg_logit : array<f32>;
@group(0) @binding(1) var<storage, read>       sg_slog  : array<f32>;
@group(0) @binding(2) var<storage, read_write> sg_sel   : array<u32>;
@group(0) @binding(3) var<storage, read_write> sg_w     : array<f32>;
@group(0) @binding(4) var<uniform>             sg_p     : MoeSelP;
@group(0) @binding(5) var<storage, read>       sg_sgw   : array<u32>;
@group(0) @binding(6) var<storage, read>       sg_x     : array<f32>;

var<workgroup> sgm_lg:  array<f32, 256>;
var<workgroup> sgm_red: array<f32, 256>;
var<workgroup> sgm_pv:  array<f32, 8>;
var<workgroup> sgm_pi:  array<u32, 8>;
var<workgroup> sgm_gate: f32;

@compute @workgroup_size(256)
fn moe_select_sg(@builtin(local_invocation_index) lid: u32,
                 @builtin(subgroup_invocation_id) sl: u32,
                 @builtin(subgroup_size) ssz: u32) {
    let sgid = lid / ssz;
    let n = sg_p.n_exp;
    let sg_kind = sg_p.pk & 0xFFu;
    let sg_hidden = sg_p.pk >> 8u;
    // shared-expert gate (same math as the tree kernel)
    if (sg_kind == 4u) {
        var d = 0.0;
        var i = lid;
        loop {
            if (i >= sg_hidden) { break; }
            d = d + bitcast<f32>(sg_sgw[i]) * sg_x[i];
            i = i + 256u;
        }
        sgm_red[lid] = d;
        workgroupBarrier();
        var st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { sgm_red[lid] = sgm_red[lid] + sgm_red[lid + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        if (lid == 0u) { sgm_gate = sgm_red[0]; }
    } else {
        if (lid == 0u) { sgm_gate = sg_slog[0]; }
    }
    workgroupBarrier();
    var v = -3.0e38;
    if (lid < n) { v = sg_logit[lid]; }
    sgm_lg[lid] = v;
    // global max + softmax denom (subgroup sums, one barrier each)
    let m1 = subgroupMax(v);
    if (sl == 0u) { sgm_red[sgid] = m1; }
    workgroupBarrier();
    var mx = -3.0e38;
    if (lid < 8u) { mx = sgm_red[lid]; }
    mx = subgroupMax(mx);
    mx = subgroupBroadcast(mx, 0u);
    if (lid == 0u) { sgm_red[255] = mx; }
    workgroupBarrier();
    mx = sgm_red[255];
    let ev = select(0.0, exp(v - mx), lid < n);
    let s1 = subgroupAdd(ev);
    if (sl == 0u) { sgm_red[sgid] = s1; }
    workgroupBarrier();
    var denom = 0.0;
    if (lid < 8u) { denom = sgm_red[lid]; }
    denom = subgroupAdd(denom);
    denom = subgroupBroadcast(denom, 0u);
    if (lid == 0u) { sgm_red[254] = denom; }
    workgroupBarrier();
    denom = sgm_red[254];
    // top-k rounds: subgroup argmax (value then lowest index), then an
    // 8-wide final in subgroup 0.
    let k = sg_p.top_k;
    var wsum = 0.0;
    for (var slot = 0u; slot < k; slot = slot + 1u) {
        let lv = sgm_lg[lid];
        let sm = subgroupMax(lv);
        let cand = select(0xFFFFFFFFu, lid, lv == sm);
        let si = subgroupMin(cand);
        if (sl == 0u) {
            sgm_pv[sgid] = sm;
            sgm_pi[sgid] = si;
        }
        workgroupBarrier();
        if (sgid == 0u) {
            var pv = -3.0e38;
            var pi = 0xFFFFFFFFu;
            if (sl < 8u) {
                pv = sgm_pv[sl];
                pi = sgm_pi[sl];
            }
            let bm = subgroupMax(pv);
            let bc = select(0xFFFFFFFFu, pi, pv == bm);
            let bi = subgroupMin(bc);
            if (sl == 0u) {
                sgm_pv[0] = bm;
                sgm_pi[0] = bi;
            }
        }
        workgroupBarrier();
        let bi = sgm_pi[0];
        let w = exp(sgm_pv[0] - mx) / denom;
        if (lid == 0u) {
            sg_sel[slot] = bi;
            sg_w[slot] = w;
        }
        wsum = wsum + w;
        if (lid == bi) { sgm_lg[lid] = -3.0e38; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        if (sg_p.norm != 0u) {
            for (var slot = 0u; slot < k; slot = slot + 1u) { sg_w[slot] = sg_w[slot] / wsum; }
        }
        sg_sel[k] = n;
        sg_w[k] = 1.0 / (1.0 + exp(-sgm_gate));
    }
}
"#;

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

fn mm_store4(m: u32, n0: u32, v0: f32, v1: f32, v2: f32, v3: f32) {
    if (m >= pm.nb) { return; }
    let base = m * pm.rows + n0;
    if (n0 < pm.rows) { ymm[base] = v0; }
    if (n0 + 1u < pm.rows) { ymm[base + 1u] = v1; }
    if (n0 + 2u < pm.rows) { ymm[base + 2u] = v2; }
    if (n0 + 3u < pm.rows) { ymm[base + 3u] = v3; }
}

fn q8_store4(m: u32, n0: u32, v0: f32, v1: f32, v2: f32, v3: f32) {
    if (m >= pm.nb) { return; }
    let base = m * pm.rows + n0;
    if (n0 < pm.rows) { ym[base] = v0 * rsm[n0]; }
    if (n0 + 1u < pm.rows) { ym[base + 1u] = v1 * rsm[n0 + 1u]; }
    if (n0 + 2u < pm.rows) { ym[base + 2u] = v2 * rsm[n0 + 2u]; }
    if (n0 + 3u < pm.rows) { ym[base + 3u] = v3 * rsm[n0 + 3u]; }
}

fn q1m_store4(m: u32, n0: u32, v0: f32, v1: f32, v2: f32, v3: f32) {
    if (m >= pm.nb) { return; }
    let base = m * pm.rows + n0;
    if (n0 < pm.rows) { ym[base] = v0; }
    if (n0 + 1u < pm.rows) { ym[base + 1u] = v1; }
    if (n0 + 2u < pm.rows) { ym[base + 2u] = v2; }
    if (n0 + 3u < pm.rows) { ym[base + 3u] = v3; }
}

@compute @workgroup_size(16, 16)
fn q8_mul_mm(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = pm.cols4 * 4u;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    // Sixteen named scalars, not array<array<f32,4>,4> — see q4t_mul_mm.
    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;
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
        let ab = lid.y * 64u;
        let wb = lid.x * 64u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let x0 = mm_at[ab + k];
            let x1 = mm_at[ab + 16u + k];
            let x2 = mm_at[ab + 32u + k];
            let x3 = mm_at[ab + 48u + k];
            let y0 = mm_wt[wb + k];
            let y1 = mm_wt[wb + 16u + k];
            let y2 = mm_wt[wb + 32u + k];
            let y3 = mm_wt[wb + 48u + k];
            a00 = a00 + x0 * y0; a01 = a01 + x0 * y1;
            a02 = a02 + x0 * y2; a03 = a03 + x0 * y3;
            a10 = a10 + x1 * y0; a11 = a11 + x1 * y1;
            a12 = a12 + x1 * y2; a13 = a13 + x1 * y3;
            a20 = a20 + x2 * y0; a21 = a21 + x2 * y1;
            a22 = a22 + x2 * y2; a23 = a23 + x2 * y3;
            a30 = a30 + x3 * y0; a31 = a31 + x3 * y1;
            a32 = a32 + x3 * y2; a33 = a33 + x3 * y3;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    let mb = m0 + lid.y * 4u;
    let nb2 = n0 + lid.x * 4u;
    q8_store4(mb, nb2, a00, a01, a02, a03);
    q8_store4(mb + 1u, nb2, a10, a11, a12, a13);
    q8_store4(mb + 2u, nb2, a20, a21, a22, a23);
    q8_store4(mb + 3u, nb2, a30, a31, a32, a33);
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
    // Sixteen named scalars, not array<array<f32,4>,4> — see q4t_mul_mm.
    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;
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
        let ab = lid.y * 64u;
        let wb = lid.x * 64u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let x0 = mm_at[ab + k];
            let x1 = mm_at[ab + 16u + k];
            let x2 = mm_at[ab + 32u + k];
            let x3 = mm_at[ab + 48u + k];
            let y0 = mm_wt[wb + k];
            let y1 = mm_wt[wb + 16u + k];
            let y2 = mm_wt[wb + 32u + k];
            let y3 = mm_wt[wb + 48u + k];
            a00 = a00 + x0 * y0; a01 = a01 + x0 * y1;
            a02 = a02 + x0 * y2; a03 = a03 + x0 * y3;
            a10 = a10 + x1 * y0; a11 = a11 + x1 * y1;
            a12 = a12 + x1 * y2; a13 = a13 + x1 * y3;
            a20 = a20 + x2 * y0; a21 = a21 + x2 * y1;
            a22 = a22 + x2 * y2; a23 = a23 + x2 * y3;
            a30 = a30 + x3 * y0; a31 = a31 + x3 * y1;
            a32 = a32 + x3 * y2; a33 = a33 + x3 * y3;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    let mb = m0 + lid.y * 4u;
    let nb2 = n0 + lid.x * 4u;
    q1m_store4(mb, nb2, a00, a01, a02, a03);
    q1m_store4(mb + 1u, nb2, a10, a11, a12, a13);
    q1m_store4(mb + 2u, nb2, a20, a21, a22, a23);
    q1m_store4(mb + 3u, nb2, a30, a31, a32, a33);
}

// ── Element-wise kernels of the MoE block (silu·mul·col, axpy, zeroing) ──
struct N1 { n: u32, f: u32, lim: f32, _c: u32 };

@group(0) @binding(0) var<storage, read>       sg   : array<f32>;
@group(0) @binding(1) var<storage, read>       su   : array<f32>;
@group(0) @binding(2) var<storage, read>       scol : array<f32>;
@group(0) @binding(3) var<storage, read_write> sact : array<f32>;
@group(0) @binding(4) var<uniform>             snp  : N1;
@compute @workgroup_size(256)
fn silu_mul_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= snp.n) { return; }
    var gv = sg[i];
    var uv = su[i];
    // swiglu_limit, and its asymmetry is the reference's: `up` is clamped on
    // BOTH sides, `gate` only from above. A device that skips it diverges
    // from the CPU path exactly on the tokens that saturate.
    if (snp.lim > 0.0) {
        uv = clamp(uv, -snp.lim, snp.lim);
        gv = min(gv, snp.lim);
    }
    var v = (gv / (1.0 + exp(-gv))) * uv;
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

// silu(g)·u in place on g — the glue pass of the fused imagegen FFN
// (w1/w3/silu/w2 in one submission, one readback).
@group(0) @binding(0) var<storage, read_write> fsg : array<f32>;
@group(0) @binding(1) var<storage, read>       fsu : array<f32>;
@group(0) @binding(2) var<uniform>             fsp : N1;
@compute @workgroup_size(256)
fn ffn_silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= fsp.n) { return; }
    let g = fsg[i];
    fsg[i] = (g / (1.0 + exp(-g))) * fsu[i];
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

// ── Two independent projections of the SAME input, in ONE dispatch.
//
// A GDN layer projects its input four ways (qkv, z, a, b); a MoE layer
// projects it twice (router, shared gate). Those are independent, but
// dispatches inside a compute pass are serialized by wgpu's
// memory-visibility guarantee — so each costs a full launch (~29 µs on
// this Vulkan stack) for work the card could overlap. Measured on a
// 5090, the 40-layer decode spends 15.3 of its 16.4 ms in
// submit+readback across ~520 dispatches, so collapsing pairs is worth
// more than any arithmetic in the kernels.
//
// The two jobs are laid end-to-end in one row space: workgroup r < rows0
// is job 0, the rest is job 1. Whole workgroups branch together, so the
// per-kind branch never diverges within a workgroup. Kinds: 4 = f32,
// 6 = q4tp; the caller keeps the unfused path for anything else.
struct MvP2 {
    rows0: u32, cols0: u32, kind0: u32, _pa: u32,
    rows1: u32, cols1: u32, kind1: u32, _pb: u32,
};
@group(0) @binding(0) var<storage, read>       m2w0 : array<u32>;
@group(0) @binding(1) var<storage, read>       m2w1 : array<u32>;
@group(0) @binding(2) var<storage, read>       m2x  : array<f32>;
@group(0) @binding(3) var<storage, read_write> m2y0 : array<f32>;
@group(0) @binding(4) var<storage, read_write> m2y1 : array<f32>;
@group(0) @binding(5) var<uniform>             m2p  : MvP2;
var<workgroup> m2part: array<f32, 64>;
var<workgroup> m2lad:  array<f32, 32>;

fn m2_dot8(w: u32, xi: u32) -> f32 {
    return (f32(w & 0xFu) - 8.0) * m2x[xi]
         + (f32((w >> 4u) & 0xFu) - 8.0) * m2x[xi + 1u]
         + (f32((w >> 8u) & 0xFu) - 8.0) * m2x[xi + 2u]
         + (f32((w >> 12u) & 0xFu) - 8.0) * m2x[xi + 3u]
         + (f32((w >> 16u) & 0xFu) - 8.0) * m2x[xi + 4u]
         + (f32((w >> 20u) & 0xFu) - 8.0) * m2x[xi + 5u]
         + (f32((w >> 24u) & 0xFu) - 8.0) * m2x[xi + 6u]
         + (f32((w >> 28u) & 0xFu) - 8.0) * m2x[xi + 7u];
}

// One partial dot over job 0's weights. Split per buffer because WGSL has
// no way to pick a binding at runtime.
fn m2_part0(kind: u32, row: u32, cols: u32, lid: u32) -> f32 {
    var acc = 0.0;
    if (kind == 4u) {
        let base = row * cols;
        var i = lid;
        loop {
            if (i >= cols) { break; }
            acc = acc + bitcast<f32>(m2w0[base + i]) * m2x[i];
            i = i + 64u;
        }
        return acc;
    }
    let gpr = cols / 32u;
    let rows = m2p.rows0;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    if (lid < 32u) {
        let pr = unpack2x16float(m2w0[params_w + row]);
        m2lad[lid] = exp2(pr.x + f32(lid) * pr.y);
    }
    workgroupBarrier();
    var g = lid;
    loop {
        if (g >= gpr) { break; }
        let bit = g * 5u;
        let cb = codes_b + row * cstride + (bit >> 3u);
        let sh = bit & 7u;
        var cv = (m2w0[cb >> 2u] >> ((cb & 3u) * 8u)) & 0xFFu;
        if (sh > 3u) {
            let cb1 = cb + 1u;
            cv = cv | (((m2w0[cb1 >> 2u] >> ((cb1 & 3u) * 8u)) & 0xFFu) << 8u);
        }
        let scale = m2lad[(cv >> sh) & 31u];
        let base = (row * gpr + g) * 4u;
        let xb = g * 32u;
        var gs = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            gs = gs + m2_dot8(m2w0[base + k], xb + 8u * k);
        }
        acc = acc + scale * gs;
        g = g + 64u;
    }
    return acc;
}

fn m2_part1(kind: u32, row: u32, cols: u32, lid: u32) -> f32 {
    var acc = 0.0;
    if (kind == 4u) {
        let base = row * cols;
        var i = lid;
        loop {
            if (i >= cols) { break; }
            acc = acc + bitcast<f32>(m2w1[base + i]) * m2x[i];
            i = i + 64u;
        }
        return acc;
    }
    let gpr = cols / 32u;
    let rows = m2p.rows1;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    if (lid < 32u) {
        let pr = unpack2x16float(m2w1[params_w + row]);
        m2lad[lid] = exp2(pr.x + f32(lid) * pr.y);
    }
    workgroupBarrier();
    var g = lid;
    loop {
        if (g >= gpr) { break; }
        let bit = g * 5u;
        let cb = codes_b + row * cstride + (bit >> 3u);
        let sh = bit & 7u;
        var cv = (m2w1[cb >> 2u] >> ((cb & 3u) * 8u)) & 0xFFu;
        if (sh > 3u) {
            let cb1 = cb + 1u;
            cv = cv | (((m2w1[cb1 >> 2u] >> ((cb1 & 3u) * 8u)) & 0xFFu) << 8u);
        }
        let scale = m2lad[(cv >> sh) & 31u];
        let base = (row * gpr + g) * 4u;
        let xb = g * 32u;
        var gs = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            gs = gs + m2_dot8(m2w1[base + k], xb + 8u * k);
        }
        acc = acc + scale * gs;
        g = g + 64u;
    }
    return acc;
}

@compute @workgroup_size(64)
fn matvec_pair(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let total = m2p.rows0 + m2p.rows1;
    var flat = wid.x;
    loop {
        if (flat >= total) { break; }
        var acc = 0.0;
        let second = flat >= m2p.rows0;
        if (second) {
            acc = m2_part1(m2p.kind1, flat - m2p.rows0, m2p.cols1, lid);
        } else {
            acc = m2_part0(m2p.kind0, flat, m2p.cols0, lid);
        }
        m2part[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { m2part[lid] = m2part[lid] + m2part[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) {
            if (second) { m2y1[flat - m2p.rows0] = m2part[0]; }
            else { m2y0[flat] = m2part[0]; }
        }
        // The next iteration rewrites m2lad/m2part; make sure every thread
        // is done reading them first.
        workgroupBarrier();
        flat = flat + nwg.x;
    }
}

// f32 matvec with a token axis for the batch graph: wid.y = token, and
// the PER-ROW math is f32_matvec verbatim — same lane stride, same tree
// reduction — so the logits it produces are bit-identical to k separate
// dispatches of the single-token kernel. That equivalence is what lets
// the batch prefill's router and GDN a/b projections collapse from one
// dispatch per token per layer (~3200 a chunk) to one per layer.
struct F32BP { cols: u32, rows: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       fb_w : array<f32>;
@group(0) @binding(1) var<storage, read>       fb_x : array<f32>;
@group(0) @binding(2) var<storage, read_write> fb_y : array<f32>;
@group(0) @binding(3) var<uniform>             fb_p : F32BP;
var<workgroup> fb_part: array<f32, 64>;
@compute @workgroup_size(64)
fn f32_matvec_b(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let t = wid.y;
    if (row >= fb_p.rows) { return; }
    let base = row * fb_p.cols;
    let xoff = t * fb_p.cols;
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= fb_p.cols) { break; }
        acc = acc + fb_w[base + i] * fb_x[xoff + i];
        i = i + 64u;
    }
    fb_part[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { fb_part[lid] = fb_part[lid] + fb_part[lid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid == 0u) { fb_y[t * fb_p.rows + row] = fb_part[0]; }
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
struct GcP { cdim: u32, kk: u32, xoff: u32, _b: u32 };
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
    var acc = gc_qkv[gc_p.xoff + c] * gc_taps[tb + kk - 1u];
    for (var j = 0u; j + 1u < kk; j = j + 1u) {
        acc = acc + gc_ring[j * cdim + c] * gc_taps[tb + j];
    }
    gc_cq[c] = acc / (1.0 + exp(-acc));
    // ring shift (columns are independent per thread c)
    for (var j = 0u; j + 2u < kk; j = j + 1u) {
        gc_ring[j * cdim + c] = gc_ring[(j + 1u) * cdim + c];
    }
    if (kk > 1u) {
        gc_ring[(kk - 2u) * cdim + c] = gc_qkv[gc_p.xoff + c];
    }
}

// ── GDN (gated DeltaNet / linear attention) decode step ──────────────────
// One workgroup per v-head. From the conv output cq it l2-norms q/k, forms the
// decay g and gate β, runs the delta-rule state recurrence S ← g·S + kf⊗β(v −
// kfᵀS) with o = qfᵀS, then the gated RMSNorm o·norm·silu(z). S ([nv,dk,dv])
// persists across tokens (device state buffer). WGSL twin of the Metal GDN
// state-update kernel; dk,dv ≤ 256.
struct GdnP { nv: u32, dk: u32, dv: u32, kd: u32, rep: u32, cdim: u32, eps: f32, tok: u32 };
@group(0) @binding(0) var<storage, read>       gd_cq   : array<f32>;
@group(0) @binding(1) var<storage, read>       gd_z    : array<f32>;
@group(0) @binding(2) var<storage, read>       gd_a    : array<f32>;
@group(0) @binding(3) var<storage, read>       gd_b    : array<f32>;
@group(0) @binding(4) var<storage, read>       gd_alog : array<f32>;
@group(0) @binding(5) var<storage, read>       gd_dtb  : array<f32>;
@group(0) @binding(6) var<storage, read>       gd_norm : array<f32>;
@group(0) @binding(7) var<storage, read_write> gd_S    : array<f32>;
@group(0) @binding(8) var<storage, read_write> gd_o    : array<f32>;
// S and o again as vec4 (same-slot rule): a state row is dv-contiguous,
// so a lane's four columns are ONE 16-byte access, not four scattered.
@group(0) @binding(7) var<storage, read_write> gd_S4   : array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> gd_o4   : array<vec4<f32>>;
@group(0) @binding(9) var<uniform>             gd_p    : GdnP;
var<workgroup> gd_kf: array<f32, 256>;
var<workgroup> gd_qf: array<f32, 256>;
var<workgroup> gd_ov: array<f32, 256>;
var<workgroup> gd_red: array<f32, 256>;
var<workgroup> gd_red2: array<f32, 256>;
var<workgroup> gd_red3: array<f32, 256>;
var<workgroup> gd_red4: array<f32, 256>;
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
    let abo = gd_p.tok * gd_p.nv;
    let g = exp(-exp(gd_alog[h]) * gd_softplus(gd_a[abo + h] + gd_dtb[h]));
    let beta = 1.0 / (1.0 + exp(-gd_b[abo + h]));
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
        let zo = gd_p.tok * gd_p.nv * dv;
        let zz = gd_z[zo + h * dv + t];
        gd_o[zo + h * dv + t] = gd_ov[t] * inv * gd_norm[t] * (zz / (1.0 + exp(-zz)));
    }
}

// ── GDN step, parallel edition: one WORKGROUP PER (head, column). The
// one-workgroup-per-head kernel put 32 workgroups on a 188-SM card — 8%
// occupancy, 3.7 ms/token of a 12.5 ms frame on the 2-bit 35B. Column j
// is independent under the delta rule, and both dk-loops become 128-lane
// tree reductions. Reduction order differs from the serial kernel, so
// bits differ within the documented GPU tie class; CMF_GDN_PAR=0 keeps
// the old kernel for A/B. The raw o lands in gd_o and gdn_step_norm
// applies the gated RMSNorm in place.
@compute @workgroup_size(128)
fn gdn_step_par(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let dj4 = wid.y;   // FOUR columns per workgroup, vec4 access
    let t = lid.x;
    let dk = gd_p.dk;
    let dv = gd_p.dv;
    if (h >= gd_p.nv || dj4 * 4u >= dv) { return; }
    let ko = h / gd_p.rep;
    let qs = ko * dk;
    let ks = gd_p.kd + ko * dk;
    // q/k l2 norms over dk (identical formulas, tree order)
    gd_red[t] = select(0.0, gd_cq[qs + t] * gd_cq[qs + t], t < dk);
    workgroupBarrier();
    var stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { gd_red[t] = gd_red[t] + gd_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let nq = gd_red[0];
    workgroupBarrier();
    gd_red[t] = select(0.0, gd_cq[ks + t] * gd_cq[ks + t], t < dk);
    workgroupBarrier();
    stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { gd_red[t] = gd_red[t] + gd_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let nkn = gd_red[0];
    workgroupBarrier();
    let invq = 1.0 / (sqrt(nq + 1e-6) * sqrt(f32(dk)));
    let invk = 1.0 / sqrt(nkn + 1e-6);
    let abo = gd_p.tok * gd_p.nv;
    let g = exp(-exp(gd_alog[h]) * gd_softplus(gd_a[abo + h] + gd_dtb[h]));
    let beta = 1.0 / (1.0 + exp(-gd_b[abo + h]));
    let s4base = (h * dk * dv) >> 2u;
    let dv4 = dv >> 2u;
    let vto = 2u * gd_p.kd + h * dv + dj4 * 4u;
    let vt = vec4<f32>(gd_cq[vto], gd_cq[vto + 1u], gd_cq[vto + 2u], gd_cq[vto + 3u]);
    let kf_t = select(0.0, gd_cq[ks + t] * invk, t < dk);
    let qf_t = select(0.0, gd_cq[qs + t] * invq, t < dk);
    // kv = kfᵀ S[:, j..j+3] — the four column reductions ride together
    var kv4 = vec4<f32>(0.0);
    if (t < dk) {
        kv4 = gd_S4[s4base + t * dv4 + dj4] * kf_t;
    }
    gd_red[t] = kv4.x;
    gd_red2[t] = kv4.y;
    gd_red3[t] = kv4.z;
    gd_red4[t] = kv4.w;
    workgroupBarrier();
    stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            gd_red[t] = gd_red[t] + gd_red[t + stride];
            gd_red2[t] = gd_red2[t] + gd_red2[t + stride];
            gd_red3[t] = gd_red3[t] + gd_red3[t + stride];
            gd_red4[t] = gd_red4[t] + gd_red4[t + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let kv = vec4<f32>(gd_red[0], gd_red2[0], gd_red3[0], gd_red4[0]);
    workgroupBarrier();
    let delta = (vt - g * kv) * beta;
    var contrib = vec4<f32>(0.0);
    if (t < dk) {
        let idx = s4base + t * dv4 + dj4;
        let cell = g * gd_S4[idx] + kf_t * delta;
        gd_S4[idx] = cell;
        contrib = qf_t * cell;
    }
    gd_red[t] = contrib.x;
    gd_red2[t] = contrib.y;
    gd_red3[t] = contrib.z;
    gd_red4[t] = contrib.w;
    workgroupBarrier();
    stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            gd_red[t] = gd_red[t] + gd_red[t + stride];
            gd_red2[t] = gd_red2[t] + gd_red2[t + stride];
            gd_red3[t] = gd_red3[t] + gd_red3[t + stride];
            gd_red4[t] = gd_red4[t] + gd_red4[t + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let zo4 = (gd_p.tok * gd_p.nv * dv) >> 2u;
        gd_o4[zo4 + h * dv4 + dj4] =
            vec4<f32>(gd_red[0], gd_red2[0], gd_red3[0], gd_red4[0]);
    }
}

// v2 of the parallel pair: the conv is INLINE (the same taps math, the
// same order, computed per element from the PRE-shift ring), so the par
// kernel no longer waits on a conv dispatch — and the ring shift rides
// the norm kernel, which was going to run anyway. One dependent hop
// fewer per GDN layer, thirty layers a frame.
struct GciP { kk: u32, xoff: u32, _a: u32, _b: u32 };
@group(0) @binding(10) var<storage, read>       gi_qkv  : array<f32>;
@group(0) @binding(11) var<storage, read_write> gi_ring : array<f32>;
@group(0) @binding(12) var<storage, read>       gi_taps : array<f32>;
@group(0) @binding(13) var<uniform>             gi_p    : GciP;

fn gi_cq(c: u32) -> f32 {
    let kk = gi_p.kk;
    let tb = c * kk;
    var acc = gi_qkv[gi_p.xoff + c] * gi_taps[tb + kk - 1u];
    for (var j = 0u; j + 1u < kk; j = j + 1u) {
        acc = acc + gi_ring[j * gd_p.cdim + c] * gi_taps[tb + j];
    }
    return acc / (1.0 + exp(-acc));
}

@compute @workgroup_size(128)
fn gdn_step_par2(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let dj = wid.y;
    let t = lid.x;
    let dk = gd_p.dk;
    let dv = gd_p.dv;
    if (h >= gd_p.nv || dj >= dv) { return; }
    let ko = h / gd_p.rep;
    let qs = ko * dk;
    let ks = gd_p.kd + ko * dk;
    let cq_q = select(0.0, gi_cq(qs + t), t < dk);
    let cq_k = select(0.0, gi_cq(ks + t), t < dk);
    gd_red[t] = cq_q * cq_q;
    workgroupBarrier();
    var stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { gd_red[t] = gd_red[t] + gd_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let nq = gd_red[0];
    workgroupBarrier();
    gd_red[t] = cq_k * cq_k;
    workgroupBarrier();
    stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { gd_red[t] = gd_red[t] + gd_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let nkn = gd_red[0];
    workgroupBarrier();
    let invq = 1.0 / (sqrt(nq + 1e-6) * sqrt(f32(dk)));
    let invk = 1.0 / sqrt(nkn + 1e-6);
    let abo = gd_p.tok * gd_p.nv;
    let g = exp(-exp(gd_alog[h]) * gd_softplus(gd_a[abo + h] + gd_dtb[h]));
    let beta = 1.0 / (1.0 + exp(-gd_b[abo + h]));
    let sbase = h * dk * dv;
    let vt = gi_cq(2u * gd_p.kd + h * dv + dj);
    let kf_t = cq_k * invk;
    let qf_t = cq_q * invq;
    gd_red[t] = select(0.0, gd_S[sbase + t * dv + dj] * kf_t, t < dk);
    workgroupBarrier();
    stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { gd_red[t] = gd_red[t] + gd_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let kv = gd_red[0];
    workgroupBarrier();
    let delta = (vt - g * kv) * beta;
    var contrib = 0.0;
    if (t < dk) {
        let idx = sbase + t * dv + dj;
        let cell = g * gd_S[idx] + kf_t * delta;
        gd_S[idx] = cell;
        contrib = qf_t * cell;
    }
    gd_red[t] = contrib;
    workgroupBarrier();
    stride = 64u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { gd_red[t] = gd_red[t] + gd_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) {
        let zo = gd_p.tok * gd_p.nv * dv;
        gd_o[zo + h * dv + dj] = gd_red[0];
    }
}

// norm v2: the gated RMSNorm PLUS the ring shift the conv kernel used to
// do — its writers (par2's gi_cq readers) are all upstream in the pass.
@compute @workgroup_size(256)
fn gdn_step_norm2(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let t = lid.x;
    let dv = gd_p.dv;
    if (h >= gd_p.nv) { return; }
    let zo = gd_p.tok * gd_p.nv * dv;
    gd_red[t] = select(0.0, gd_o[zo + h * dv + t] * gd_o[zo + h * dv + t], t < dv);
    workgroupBarrier();
    let ss = gd_reduce(t);
    workgroupBarrier();
    let inv = 1.0 / sqrt(ss / f32(dv) + gd_p.eps);
    if (t < dv) {
        let zz = gd_z[zo + h * dv + t];
        gd_o[zo + h * dv + t] =
            gd_o[zo + h * dv + t] * inv * gd_norm[t] * (zz / (1.0 + exp(-zz)));
    }
    // ring shift, strided over cdim across all norm workgroups
    let kk = gi_p.kk;
    let cdim = gd_p.cdim;
    var c = wid.x * 256u + t;
    loop {
        if (c >= cdim) { break; }
        for (var j = 0u; j + 2u < kk; j = j + 1u) {
            gi_ring[j * cdim + c] = gi_ring[(j + 1u) * cdim + c];
        }
        if (kk > 1u) {
            gi_ring[(kk - 2u) * cdim + c] = gi_qkv[gi_p.xoff + c];
        }
        c = c + gd_p.nv * 256u;
    }
}

// Gated RMSNorm tail of the parallel GDN step: in place over gd_o.
@compute @workgroup_size(256)
fn gdn_step_norm(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let t = lid.x;
    let dv = gd_p.dv;
    if (h >= gd_p.nv) { return; }
    let zo = gd_p.tok * gd_p.nv * dv;
    gd_red[t] = select(0.0, gd_o[zo + h * dv + t] * gd_o[zo + h * dv + t], t < dv);
    workgroupBarrier();
    let ss = gd_reduce(t);
    workgroupBarrier();
    let inv = 1.0 / sqrt(ss / f32(dv) + gd_p.eps);
    if (t < dv) {
        let zz = gd_z[zo + h * dv + t];
        gd_o[zo + h * dv + t] =
            gd_o[zo + h * dv + t] * inv * gd_norm[t] * (zz / (1.0 + exp(-zz)));
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
struct RqP { nh: u32, nkv: u32, hd: u32, rd: u32, pos: u32, flags: u32, eps: f32, tok: u32 };
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
    // Batch-graph token offsets (0 in the token graph): q rows live in the
    // batched projection output, K is rotated IN PLACE in its batch slice.
    let qoff = rq_p.tok * nh * select(1u, 2u, gate) * hd;
    let koff = rq_p.tok * rq_p.nkv * hd;
    let nt = (hd + 31u) / 32u;  // ≤ 8 for head_dim ≤ 256 (Qwen3.5 uses 256)
    var xv: array<f32, 8>;
    var ss = 0.0;
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        var val = 0.0;
        if (d < hd) { val = select(rq_k[koff + src_base + d], rq_qraw[qoff + src_base + d], isq); }
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
            if (isq) { rq_qout[dst_base + d] = rq_head[d]; } else { rq_k[koff + dst_base + d] = rq_head[d]; }
        }
    }
    if (isq && gate) {
        let gbase = head * 2u * hd + hd;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) { rq_gout[head * hd + d] = rq_qraw[qoff + gbase + d]; }
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
// `stored` carries the batch token index in its high bits (pos | tok<<20):
// the batch graph appends straight from its batched K/V buffers, the token
// graph passes tok=0 and reads from offset zero as before. Positions stay
// under 2^20, far above any cap the cache allows.
@compute @workgroup_size(256)
fn kv_append(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= kv_p.nkv * kv_p.hd) { return; }
    let stored = kv_p.stored & 0xFFFFFu;
    let toff = (kv_p.stored >> 20u) * kv_p.nkv * kv_p.hd;
    let h = i / kv_p.hd;
    let d = i % kv_p.hd;
    let dst = (h * kv_p.cap + stored) * kv_p.hd + d;
    kv_kb[dst] = kv_k[toff + i];
    kv_vb[dst] = kv_v[toff + i];
}

// Grouped decode attention, one 32-thread workgroup per Q-head. Dims sliced
// across lanes (dim d in lane d%32, slot d/32); online softmax over the n
// cached positions with the per-position q·k dot reduced in workgroup memory
// (portable — no subgroup ops). WGSL twin of Metal gqa_attend (output only;
// Born-importance is handled on the CPU side when eviction is active).
struct AtP { nh: u32, hpk: u32, hd: u32, cap: u32, n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       at_q : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       at_k : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       at_v : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> at_o : array<f32>;
@group(0) @binding(4) var<uniform>             at_p : AtP;
// Flash-decoding: split the n cached positions across the 32 lanes. Each lane
// runs an INDEPENDENT online softmax over positions lane, lane+32, … with NO
// barrier in the loop (the old kernel barriered twice PER position — O(ctx)
// serial chain), then a 5-step 32-way log-sum-exp merge. Serial steps: n → n/32.
// K/V/Q are vec4 bindings (hd % 4 == 0, gated by the Rust callers): each lane
// reads a DIFFERENT cache row, so f32 loads were 4B-used-per-32B-sector — the
// depth wall of the decode graph (4090, 1.7B q1 @ctx512: attend dominated the
// 15 ms/token submit). vec4 quarters the wasted sectors. The workgroup
// accumulator stays SCALAR at stride 257 — (lane·257 + d) mod 32 is unique per
// lane, bank-conflict-free; a vec4 accumulator array cannot be (stride must be
// ≡1 mod 32 AND a multiple of 4 — impossible).
var<workgroup> at_acc: array<f32, 8224>; // [lane*257 + d], stride 257 dodges 32-bank conflicts, hd ≤ 256 (Qwen3.5=256)
var<workgroup> at_m: array<f32, 32>;
var<workgroup> at_l: array<f32, 32>;
// Decode-regime attend: 256 threads per head instead of one warp. Lanes
// are POSITIONS for the score pass (dot over hd each) and DIMENSIONS for
// the value pass (coalesced v reads, one output dim per lane, hd <= 256).
// Online softmax over 256-position chunks; per-chunk stats via one tree.
// The 32-lane kernel above kept a 257-stride accumulator per lane and a
// five-level 256-wide merge — 137 us per layer at fifty positions of
// context. This shape does the same math in the natural order.
var<workgroup> ad_sc: array<f32, 256>;
var<workgroup> ad_red: array<f32, 256>;

@compute @workgroup_size(256)
fn gqa_attend_dec(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    if (h >= at_p.nh) { return; }
    let hd = at_p.hd;
    let hd4 = hd / 4u;
    let n = at_p.n;
    let kbase = (h / at_p.hpk) * at_p.cap * hd4;
    let qbase = h * hd4;
    let scale = 1.0 / sqrt(f32(hd));
    var m = -1.0e30;
    var l = 0.0;
    var acc = 0.0;                      // this lane's output dim (lid < hd)
    var c0 = 0u;
    loop {
        if (c0 >= n) { break; }
        let cn = min(256u, n - c0);
        // scores: lane p of the chunk
        var sc = -1.0e30;
        if (lid < cn) {
            let krow = kbase + (c0 + lid) * hd4;
            var dot4 = vec4<f32>(0.0);
            for (var d = 0u; d < hd4; d = d + 1u) {
                dot4 = dot4 + at_q[qbase + d] * at_k[krow + d];
            }
            sc = (dot4.x + dot4.y + dot4.z + dot4.w) * scale;
        }
        ad_sc[lid] = sc;
        ad_red[lid] = sc;
        workgroupBarrier();
        var st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { ad_red[lid] = max(ad_red[lid], ad_red[lid + st]); }
            workgroupBarrier();
            st = st >> 1u;
        }
        let cm = ad_red[0];
        workgroupBarrier();
        let mp = max(m, cm);
        let f = exp(m - mp);
        // weights into shared, denom via tree
        let w = select(0.0, exp(ad_sc[lid] - mp), lid < cn);
        ad_sc[lid] = w;
        ad_red[lid] = w;
        workgroupBarrier();
        st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { ad_red[lid] = ad_red[lid] + ad_red[lid + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        l = l * f + ad_red[0];
        workgroupBarrier();
        // value pass: lane = output dim, coalesced across lanes
        if (lid < hd) {
            acc = acc * f;
            let dw = lid >> 2u;
            let dc = lid & 3u;
            for (var p = 0u; p < cn; p = p + 1u) {
                acc = acc + ad_sc[p] * at_v[kbase + (c0 + p) * hd4 + dw][dc];
            }
        }
        m = mp;
        c0 = c0 + 256u;
        workgroupBarrier();
    }
    if (lid < hd) {
        at_o[h * hd + lid] = acc / l;
    }
}

@compute @workgroup_size(32)
fn gqa_attend(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let lane = lid.x;
    if (h >= at_p.nh) { return; }
    let hd = at_p.hd;
    let hd4 = hd / 4u;
    let n = at_p.n;
    let kbase = (h / at_p.hpk) * at_p.cap * hd4;
    let qbase = h * hd4;
    let scale = 1.0 / sqrt(f32(hd));
    let base = lane * 257u;
    for (var d = 0u; d < hd; d = d + 1u) { at_acc[base + d] = 0.0; }
    var m = -1e30;
    var l = 0.0;
    var p = lane;
    loop {
        if (p >= n) { break; }
        let krow = kbase + p * hd4;
        var dot4 = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d = d + 1u) { dot4 = dot4 + at_q[qbase + d] * at_k[krow + d]; }
        let dot = (dot4.x + dot4.y + dot4.z + dot4.w) * scale;
        let mp = max(m, dot);
        let f = exp(m - mp);
        let w = exp(dot - mp);
        l = l * f + w;
        for (var d = 0u; d < hd4; d = d + 1u) {
            let vv = at_v[krow + d] * w;
            let a = base + d * 4u;
            at_acc[a]      = at_acc[a]      * f + vv.x;
            at_acc[a + 1u] = at_acc[a + 1u] * f + vv.y;
            at_acc[a + 2u] = at_acc[a + 2u] * f + vv.z;
            at_acc[a + 3u] = at_acc[a + 3u] * f + vv.w;
        }
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

// head_dim ≤ 256 on a 32 KB device: same stride 257, HALF the lanes.
//
// The 32-lane kernel above needs 32·257·4 = 32 896 B of workgroup memory
// and cannot be created where the limit is 32 768 — wgpu-Metal and mobile.
// The stride cannot shrink (it must exceed hd, and 257 is what dodges the
// 32-bank conflicts), so the lane count is the only free dimension:
// 16·257·4 = 16 448 B fits with room to spare.
//
// Without this, `hd_cap` on Apple was 128 and the whole-token graph
// silently declined for the ENTIRE Qwen3.5/3.6 family (head_dim 256) —
// every layer walked the host on a machine whose GPU could have run it.
// Halving the lanes halves the position parallelism, which this kernel
// can afford: it is bound by the vec4 K/V reads, not by lane occupancy.
var<workgroup> at_acc16: array<f32, 4112>; // [lane*257 + d], 16 lanes
var<workgroup> at_m16: array<f32, 16>;
var<workgroup> at_l16: array<f32, 16>;
@compute @workgroup_size(16)
fn gqa_attend_w16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let lane = lid.x;
    if (h >= at_p.nh) { return; }
    let hd = at_p.hd;
    let hd4 = hd / 4u;
    let n = at_p.n;
    let kbase = (h / at_p.hpk) * at_p.cap * hd4;
    let qbase = h * hd4;
    let scale = 1.0 / sqrt(f32(hd));
    let base = lane * 257u;
    for (var d = 0u; d < hd; d = d + 1u) { at_acc16[base + d] = 0.0; }
    var m = -1e30;
    var l = 0.0;
    var p = lane;
    loop {
        if (p >= n) { break; }
        let krow = kbase + p * hd4;
        var dot4 = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d = d + 1u) { dot4 = dot4 + at_q[qbase + d] * at_k[krow + d]; }
        let dot = (dot4.x + dot4.y + dot4.z + dot4.w) * scale;
        let mp = max(m, dot);
        let f = exp(m - mp);
        let w = exp(dot - mp);
        l = l * f + w;
        for (var d = 0u; d < hd4; d = d + 1u) {
            let vv = at_v[krow + d] * w;
            let a = base + d * 4u;
            at_acc16[a]      = at_acc16[a]      * f + vv.x;
            at_acc16[a + 1u] = at_acc16[a + 1u] * f + vv.y;
            at_acc16[a + 2u] = at_acc16[a + 2u] * f + vv.z;
            at_acc16[a + 3u] = at_acc16[a + 3u] * f + vv.w;
        }
        m = mp;
        p = p + 16u;
    }
    at_m16[lane] = m;
    at_l16[lane] = l;
    workgroupBarrier();
    var stride = 8u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            let o = lane + stride;
            let m1 = at_m16[lane];
            let m2 = at_m16[o];
            let mm = max(m1, m2);
            let f1 = exp(m1 - mm);
            let f2 = exp(m2 - mm);
            at_l16[lane] = at_l16[lane] * f1 + at_l16[o] * f2;
            let bo = o * 257u;
            for (var d = 0u; d < hd; d = d + 1u) {
                at_acc16[base + d] = at_acc16[base + d] * f1 + at_acc16[bo + d] * f2;
            }
            at_m16[lane] = mm;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let invl = select(0.0, 1.0 / at_l16[0], at_l16[0] > 0.0);
    for (var d = lane; d < hd; d = d + 16u) {
        at_o[h * hd + d] = at_acc16[d] * invl;
    }
}

// hd <= 128 twin of gqa_attend at stride 129 — 16.5 KB of workgroup
// memory instead of 33 KB. Mobile GPUs (Adreno/Mali) and wgpu-Metal cap
// maxComputeWorkgroupStorageSize at 32768 B, where the 257-stride kernel
// cannot even be created: the invalid pipeline turned every dispatch
// into a no-op and the graph decoded garbage on phones. (lane*129 + d)
// mod 32 == (lane + d) mod 32 — still bank-conflict-free.
var<workgroup> at_acc_s: array<f32, 4128>;
@compute @workgroup_size(32)
fn gqa_attend_s(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let lane = lid.x;
    if (h >= at_p.nh) { return; }
    let hd = at_p.hd;
    let hd4 = hd / 4u;
    let n = at_p.n;
    let kbase = (h / at_p.hpk) * at_p.cap * hd4;
    let qbase = h * hd4;
    let scale = 1.0 / sqrt(f32(hd));
    let base = lane * 129u;
    for (var d = 0u; d < hd; d = d + 1u) { at_acc_s[base + d] = 0.0; }
    var m = -1e30;
    var l = 0.0;
    var p = lane;
    loop {
        if (p >= n) { break; }
        let krow = kbase + p * hd4;
        var dot4 = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d = d + 1u) { dot4 = dot4 + at_q[qbase + d] * at_k[krow + d]; }
        let dot = (dot4.x + dot4.y + dot4.z + dot4.w) * scale;
        let mp = max(m, dot);
        let f = exp(m - mp);
        let w = exp(dot - mp);
        l = l * f + w;
        for (var d = 0u; d < hd4; d = d + 1u) {
            let vv = at_v[krow + d] * w;
            let a = base + d * 4u;
            at_acc_s[a]      = at_acc_s[a]      * f + vv.x;
            at_acc_s[a + 1u] = at_acc_s[a + 1u] * f + vv.y;
            at_acc_s[a + 2u] = at_acc_s[a + 2u] * f + vv.z;
            at_acc_s[a + 3u] = at_acc_s[a + 3u] * f + vv.w;
        }
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
            let bo = o * 129u;
            for (var d = 0u; d < hd; d = d + 1u) {
                at_acc_s[base + d] = at_acc_s[base + d] * f1 + at_acc_s[bo + d] * f2;
            }
            at_m[lane] = mm;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let invl = select(0.0, 1.0 / at_l[0], at_l[0] > 0.0);
    for (var d = lane; d < hd; d = d + 32u) {
        at_o[h * hd + d] = at_acc_s[d] * invl;
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
            // One byte carries FIVE base-3 codes: read (and LUT) it once
            // and spend it on all five, instead of re-reading per weight —
            // 7 byte loads a group against 32. Same k order, same adds.
            var k = 0u;
            for (var bi = 0u; bi < 7u; bi = bi + 1u) {
                let p = Q1T_LUT[q1t_byte(codes + bi)];
                let n = min(5u, 32u - k);
                for (var j = 0u; j < n; j = j + 1u) {
                    let code = (p >> (j * 2u)) & 3u;
                    let sgn = select(0.0, 1.0, code == 1u) - select(0.0, 1.0, code == 2u);
                    gsum = gsum + sgn * q1x[xb + k + j];
                }
                k = k + n;
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

// q4b, tall edition: 8 rows per 256-thread workgroup in pairs with vec4
// activations — the same recipe as q4tp_matvec4, on the split layout
// (nibbles and f16 scales in two distant planes). Per-row group order
// and add order match the one-row kernel, so parity carries.
var<workgroup> p8a_q4b: array<f32, 256>;
var<workgroup> p8b_q4b: array<f32, 256>;

@compute @workgroup_size(256)
fn q4b_matvec8(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let scales_off = rows * gpr * 16u;
    let sub = lid >> 6u;
    let l = lid & 63u;
    var base = wid.x * 8u;
    loop {
        if (base >= rows) { break; }
        let row_a = base + sub;
        let row_b = row_a + 4u;
        var acc_a = 0.0;
        var acc_b = 0.0;
        if (row_a < rows) {
            let live_b = row_b < rows;
            var g = l;
            loop {
                if (g >= gpr) { break; }
                let xq = g * 8u;
                let x0 = q4v_x[xq];      let x1 = q4v_x[xq + 1u];
                let x2 = q4v_x[xq + 2u]; let x3 = q4v_x[xq + 3u];
                let x4 = q4v_x[xq + 4u]; let x5 = q4v_x[xq + 5u];
                let x6 = q4v_x[xq + 6u]; let x7 = q4v_x[xq + 7u];
                let ga = row_a * gpr + g;
                let sab = scales_off + ga * 2u;
                let sa = unpack2x16float((q1w[sab >> 2u] >> ((sab & 3u) * 8u)) & 0xFFFFu).x;
                let va = q4v_w[ga];
                acc_a = acc_a + sa
                    * (q4v_dot8(va.x, x0, x1) + q4v_dot8(va.y, x2, x3)
                     + q4v_dot8(va.z, x4, x5) + q4v_dot8(va.w, x6, x7));
                if (live_b) {
                    let gb = row_b * gpr + g;
                    let sbb = scales_off + gb * 2u;
                    let sb = unpack2x16float((q1w[sbb >> 2u] >> ((sbb & 3u) * 8u)) & 0xFFFFu).x;
                    let vb = q4v_w[gb];
                    acc_b = acc_b + sb
                        * (q4v_dot8(vb.x, x0, x1) + q4v_dot8(vb.y, x2, x3)
                         + q4v_dot8(vb.z, x4, x5) + q4v_dot8(vb.w, x6, x7));
                }
                g = g + 64u;
            }
        }
        p8a_q4b[lid] = acc_a;
        p8b_q4b[lid] = acc_b;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) {
                p8a_q4b[lid] = p8a_q4b[lid] + p8a_q4b[lid + stride];
                p8b_q4b[lid] = p8b_q4b[lid] + p8b_q4b[lid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (l == 0u) {
            if (row_a < rows) { q1y[row_a] = p8a_q4b[sub << 6u]; }
            if (row_b < rows) { q1y[row_b] = p8b_q4b[sub << 6u]; }
        }
        workgroupBarrier();
        base = base + nwg.x * 8u;
    }
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

// q4_tiled matvec: 18-byte interleaved tiles [f16 scale][16B nibbles] — ONE
// stream per row (the split q4b layout above reads nibbles and scales from
// two distant regions; feeding TILED bytes to it produced garbage — caught by
// an end-to-end answer check on real Vulkan). Tiles are 2-aligned, so words
// assemble from u16 halves of the u32 weight array.
fn q4t_u16(off16: u32) -> u32 {
    return (q1w[off16 >> 1u] >> ((off16 & 1u) * 16u)) & 0xFFFFu;
}
@compute @workgroup_size(64)
fn q4t_matvec(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    var row = wid.x;
    loop {
        if (row >= rows) { break; }
        var acc = 0.0;
        var g = lid;
        loop {
            if (g >= gpr) { break; }
            let t16 = (row * gpr + g) * 9u;
            let scale = unpack2x16float(q4t_u16(t16)).x;
            let xb = g * 32u;
            var gsum = 0.0;
            for (var k = 0u; k < 4u; k = k + 1u) {
                let w = q4t_u16(t16 + 1u + 2u * k) | (q4t_u16(t16 + 2u + 2u * k) << 16u);
                gsum = gsum + q4b_dot8(w, xb + 8u * k);
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
        if (lid == 0u) { q1y[row] = partial_q1t[0]; }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// q4t, tall edition: 8 rows per 256-thread workgroup in pairs, vec4
// activations. The 18-byte tile stride is 2-aligned, not 4, so the
// weights stay u16-assembled (that is the layout's own cost) — but the
// activation side vectorizes exactly as in q4tp, and every x vec4 feeds
// two rows. Per-row group order and add order are the one-row kernel's.
var<workgroup> lad_q4t8: array<f32, 8>;
var<workgroup> p8a_q4t: array<f32, 256>;
var<workgroup> p8b_q4t: array<f32, 256>;

fn q4t_dot8v(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return (f32(w & 0xFu) - 8.0) * a.x
         + (f32((w >> 4u) & 0xFu) - 8.0) * a.y
         + (f32((w >> 8u) & 0xFu) - 8.0) * a.z
         + (f32((w >> 12u) & 0xFu) - 8.0) * a.w
         + (f32((w >> 16u) & 0xFu) - 8.0) * b.x
         + (f32((w >> 20u) & 0xFu) - 8.0) * b.y
         + (f32((w >> 24u) & 0xFu) - 8.0) * b.z
         + (f32((w >> 28u) & 0xFu) - 8.0) * b.w;
}

@compute @workgroup_size(256)
fn q4t_matvec8(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let sub = lid >> 6u;
    let l = lid & 63u;
    var base = wid.x * 8u;
    loop {
        if (base >= rows) { break; }
        let row_a = base + sub;
        let row_b = row_a + 4u;
        var acc_a = 0.0;
        var acc_b = 0.0;
        if (row_a < rows) {
            let live_b = row_b < rows;
            var g = l;
            loop {
                if (g >= gpr) { break; }
                let xq = g * 8u;
                let x0 = q4v_x[xq];      let x1 = q4v_x[xq + 1u];
                let x2 = q4v_x[xq + 2u]; let x3 = q4v_x[xq + 3u];
                let x4 = q4v_x[xq + 4u]; let x5 = q4v_x[xq + 5u];
                let x6 = q4v_x[xq + 6u]; let x7 = q4v_x[xq + 7u];
                let ta = (row_a * gpr + g) * 9u;
                let sa = unpack2x16float(q4t_u16(ta)).x;
                let wa0 = q4t_u16(ta + 1u) | (q4t_u16(ta + 2u) << 16u);
                let wa1 = q4t_u16(ta + 3u) | (q4t_u16(ta + 4u) << 16u);
                let wa2 = q4t_u16(ta + 5u) | (q4t_u16(ta + 6u) << 16u);
                let wa3 = q4t_u16(ta + 7u) | (q4t_u16(ta + 8u) << 16u);
                acc_a = acc_a + sa
                    * (q4t_dot8v(wa0, x0, x1) + q4t_dot8v(wa1, x2, x3)
                     + q4t_dot8v(wa2, x4, x5) + q4t_dot8v(wa3, x6, x7));
                if (live_b) {
                    let tb = (row_b * gpr + g) * 9u;
                    let sb = unpack2x16float(q4t_u16(tb)).x;
                    let wb0 = q4t_u16(tb + 1u) | (q4t_u16(tb + 2u) << 16u);
                    let wb1 = q4t_u16(tb + 3u) | (q4t_u16(tb + 4u) << 16u);
                    let wb2 = q4t_u16(tb + 5u) | (q4t_u16(tb + 6u) << 16u);
                    let wb3 = q4t_u16(tb + 7u) | (q4t_u16(tb + 8u) << 16u);
                    acc_b = acc_b + sb
                        * (q4t_dot8v(wb0, x0, x1) + q4t_dot8v(wb1, x2, x3)
                         + q4t_dot8v(wb2, x4, x5) + q4t_dot8v(wb3, x6, x7));
                }
                g = g + 64u;
            }
        }
        p8a_q4t[lid] = acc_a;
        p8b_q4t[lid] = acc_b;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) {
                p8a_q4t[lid] = p8a_q4t[lid] + p8a_q4t[lid + stride];
                p8b_q4t[lid] = p8b_q4t[lid] + p8b_q4t[lid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (l == 0u) {
            if (row_a < rows) { q1y[row_a] = p8a_q4t[sub << 6u]; }
            if (row_b < rows) { q1y[row_b] = p8b_q4t[sub << 6u]; }
        }
        workgroupBarrier();
        base = base + nwg.x * 8u;
    }
}

// q4tp matvec: same nibble values as q4t, but the stride is a clean 16 B —
// so the words come straight off the u32 array instead of being assembled
// from u16 halves the way q4t's 2-aligned 18 B tiles force. The scale is a
// 5-bit rung on the row's ladder, kept in two planes after the nibbles.
//
// A workgroup owns one row at a time, so it expands that row's 32 rungs once
// into workgroup memory. Evaluating 2^(lo + code*step) per tile instead was
// measured on Metal to cost the model ~15% even though the kernel benchmarked
// faster standalone: the graph's dispatches serialize on each other, which
// exposes the dependent chain (code byte → exp2 → scale) that a free-running
// benchmark hides.
var<workgroup> lad_q4tp: array<f32, 32>;

fn q4tp_byte(off: u32) -> u32 {
    return (q1w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn q4tp_matvec(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let params_w = rows * gpr * 4u;                  // u32 index of row params
    let codes_b = rows * gpr * 16u + rows * 4u;      // byte offset of the codes
    let cstride = (gpr * 5u + 7u) / 8u;
    var row = wid.x;
    loop {
        if (row >= rows) { break; }
        if (lid < 32u) {
            let pr = unpack2x16float(q1w[params_w + row]);
            lad_q4tp[lid] = exp2(pr.x + f32(lid) * pr.y);
        }
        workgroupBarrier();
        var acc = 0.0;
        var g = lid;
        loop {
            if (g >= gpr) { break; }
            let bit = g * 5u;
            let cb = codes_b + row * cstride + (bit >> 3u);
            let sh = bit & 7u;
            // The 5-bit field spills into the next byte past bit 3; the row's
            // stride always holds that byte when it does.
            var cv = q4tp_byte(cb);
            if (sh > 3u) { cv = cv | (q4tp_byte(cb + 1u) << 8u); }
            let scale = lad_q4tp[(cv >> sh) & 31u];
            let base = (row * gpr + g) * 4u;
            let xb = g * 32u;
            var gsum = 0.0;
            for (var k = 0u; k < 4u; k = k + 1u) {
                gsum = gsum + q4b_dot8(q1w[base + k], xb + 8u * k);
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
        if (lid == 0u) { q1y[row] = partial_q1t[0]; }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// Narrow-matrix edition: 16 rows per workgroup for gpr <= 64 shapes
// (cols <= 2048: the GDN projections, o/qkv projections, lm_head), where
// the 8-row kernel gives each lane exactly ONE group and nothing to
// amortize. Four rows per 64-lane sub-block share every activation vec4
// four ways. Per-row lane layout and add order match the one-row kernel.
var<workgroup> lad_q16: array<f32, 512>;
var<workgroup> p16_a: array<f32, 256>;
var<workgroup> p16_b: array<f32, 256>;
var<workgroup> p16_c: array<f32, 256>;
var<workgroup> p16_d: array<f32, 256>;

@compute @workgroup_size(256)
fn q4tp_matvec16(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(num_workgroups) nwg: vec3<u32>,
                 @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let sub = lid >> 6u;
    let l = lid & 63u;
    var base = wid.x * 16u;
    loop {
        if (base >= rows) { break; }
        // 16 rows x 32 rungs: each thread stages two.
        for (var q = lid; q < 512u; q = q + 256u) {
            let r = base + (q >> 5u);
            if (r < rows) {
                let pr = unpack2x16float(q1w[params_w + r]);
                lad_q16[q] = exp2(pr.x + f32(q & 31u) * pr.y);
            }
        }
        workgroupBarrier();
        let r_a = base + sub * 4u;
        let r_b = r_a + 1u;
        let r_c = r_a + 2u;
        let r_d = r_a + 3u;
        var aa = 0.0;
        var ab = 0.0;
        var ac = 0.0;
        var ad = 0.0;
        if (r_a < rows) {
            let all_live = r_d < rows;
            var g = l;
            loop {
                if (g >= gpr) { break; }
                let bit = g * 5u;
                let cbo = bit >> 3u;
                let sh = bit & 7u;
                let xq = g * 8u;
                let x0 = q4v_x[xq];      let x1 = q4v_x[xq + 1u];
                let x2 = q4v_x[xq + 2u]; let x3 = q4v_x[xq + 3u];
                let x4 = q4v_x[xq + 4u]; let x5 = q4v_x[xq + 5u];
                let x6 = q4v_x[xq + 6u]; let x7 = q4v_x[xq + 7u];
                let cra = codes_b + r_a * cstride + cbo;
                var cva = q4tp_byte(cra);
                if (sh > 3u) { cva = cva | (q4tp_byte(cra + 1u) << 8u); }
                let sa = lad_q16[(sub * 4u << 5u) + ((cva >> sh) & 31u)];
                let va = q4v_w[r_a * gpr + g];
                aa = aa + sa
                    * (q4v_dot8(va.x, x0, x1) + q4v_dot8(va.y, x2, x3)
                     + q4v_dot8(va.z, x4, x5) + q4v_dot8(va.w, x6, x7));
                if (all_live || r_b < rows) {
                    let crb = codes_b + r_b * cstride + cbo;
                    var cvb = q4tp_byte(crb);
                    if (sh > 3u) { cvb = cvb | (q4tp_byte(crb + 1u) << 8u); }
                    let sb = lad_q16[((sub * 4u + 1u) << 5u) + ((cvb >> sh) & 31u)];
                    let vb = q4v_w[r_b * gpr + g];
                    ab = ab + sb
                        * (q4v_dot8(vb.x, x0, x1) + q4v_dot8(vb.y, x2, x3)
                         + q4v_dot8(vb.z, x4, x5) + q4v_dot8(vb.w, x6, x7));
                }
                if (all_live || r_c < rows) {
                    let crc = codes_b + r_c * cstride + cbo;
                    var cvc = q4tp_byte(crc);
                    if (sh > 3u) { cvc = cvc | (q4tp_byte(crc + 1u) << 8u); }
                    let sc = lad_q16[((sub * 4u + 2u) << 5u) + ((cvc >> sh) & 31u)];
                    let vc = q4v_w[r_c * gpr + g];
                    ac = ac + sc
                        * (q4v_dot8(vc.x, x0, x1) + q4v_dot8(vc.y, x2, x3)
                         + q4v_dot8(vc.z, x4, x5) + q4v_dot8(vc.w, x6, x7));
                }
                if (all_live || r_d < rows) {
                    let crd = codes_b + r_d * cstride + cbo;
                    var cvd = q4tp_byte(crd);
                    if (sh > 3u) { cvd = cvd | (q4tp_byte(crd + 1u) << 8u); }
                    let sd = lad_q16[((sub * 4u + 3u) << 5u) + ((cvd >> sh) & 31u)];
                    let vd = q4v_w[r_d * gpr + g];
                    ad = ad + sd
                        * (q4v_dot8(vd.x, x0, x1) + q4v_dot8(vd.y, x2, x3)
                         + q4v_dot8(vd.z, x4, x5) + q4v_dot8(vd.w, x6, x7));
                }
                g = g + 64u;
            }
        }
        p16_a[lid] = aa;
        p16_b[lid] = ab;
        p16_c[lid] = ac;
        p16_d[lid] = ad;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) {
                p16_a[lid] = p16_a[lid] + p16_a[lid + stride];
                p16_b[lid] = p16_b[lid] + p16_b[lid + stride];
                p16_c[lid] = p16_c[lid] + p16_c[lid + stride];
                p16_d[lid] = p16_d[lid] + p16_d[lid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (l == 0u) {
            if (r_a < rows) { q1y[r_a] = p16_a[sub << 6u]; }
            if (r_b < rows) { q1y[r_b] = p16_b[sub << 6u]; }
            if (r_c < rows) { q1y[r_c] = p16_c[sub << 6u]; }
            if (r_d < rows) { q1y[r_d] = p16_d[sub << 6u]; }
        }
        workgroupBarrier();
        base = base + nwg.x * 16u;
    }
}

// q4tp matvec, tall edition: 4 rows per 256-thread workgroup, and the group's
// 16 B of nibbles arrive as ONE vec4<u32> load instead of four scalar loads.
// Written for dense-FFN shapes (17408x5120: the one-row kernel left a 27B
// dense model at ~5% of the card's bandwidth); the weight buffer is bound
// TWICE — the scalar u32 view for params and 5-bit codes (they live at
// unaligned offsets, and the buffer tail may not be 16 B-round, which a vec4
// view would silently clamp) and a vec4 view for the nibble tiles, whose
// region is 16 B-exact by construction. Each row's lane layout, add order and
// 64-slot reduction tree are byte-identical to q4tp_matvec, so the kernels
// are interchangeable under greedy parity.
@group(0) @binding(4) var<storage, read> q4v_w : array<vec4<u32>>;
// The activations again, as vec4: the scalar kernel issues 32 x-loads per
// 16 B of weights and is LSU-bound long before it is bandwidth-bound
// (measured 190 GB/s of 1.79 TB/s on the dense-FFN shapes). Components are
// consumed in the exact q4b_dot8 order.
@group(0) @binding(5) var<storage, read> q4v_x : array<vec4<f32>>;

var<workgroup> lad_q4v: array<f32, 256>;
var<workgroup> partial_q4v: array<f32, 256>;
var<workgroup> partial_q4vb: array<f32, 256>;

fn q4v_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return (f32(w & 0xFu) - 8.0) * a.x
         + (f32((w >> 4u) & 0xFu) - 8.0) * a.y
         + (f32((w >> 8u) & 0xFu) - 8.0) * a.z
         + (f32((w >> 12u) & 0xFu) - 8.0) * a.w
         + (f32((w >> 16u) & 0xFu) - 8.0) * b.x
         + (f32((w >> 20u) & 0xFu) - 8.0) * b.y
         + (f32((w >> 24u) & 0xFu) - 8.0) * b.z
         + (f32((w >> 28u) & 0xFu) - 8.0) * b.w;
}

@compute @workgroup_size(256)
fn q4tp_matvec4(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(num_workgroups) nwg: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let sub = lid >> 6u;
    let l = lid & 63u;
    // 8 rows per workgroup, register-blocked in pairs: sub-block `sub` owns
    // rows base+sub and base+sub+4, and every x vec4 fetched for a group
    // feeds BOTH rows' dot chains — the x side of the LSU load nearly
    // halves. Each row's group order and add order stay those of the
    // one-row kernel.
    var base = wid.x * 8u;
    loop {
        if (base >= rows) { break; }
        {
            let r = base + (lid >> 5u);
            if (r < rows) {
                let pr = unpack2x16float(q1w[params_w + r]);
                lad_q4v[lid] = exp2(pr.x + f32(lid & 31u) * pr.y);
            }
        }
        workgroupBarrier();
        let row_a = base + sub;
        let row_b = base + sub + 4u;
        let live_a = row_a < rows;
        let live_b = row_b < rows;
        var acc_a = 0.0;
        var acc_b = 0.0;
        if (live_a) {
            let crow_a = codes_b + row_a * cstride;
            let crow_b = codes_b + row_b * cstride;
            var g = l;
            loop {
                if (g >= gpr) { break; }
                let bit = g * 5u;
                let cbo = bit >> 3u;
                let sh = bit & 7u;
                var cv_a = q4tp_byte(crow_a + cbo);
                if (sh > 3u) { cv_a = cv_a | (q4tp_byte(crow_a + cbo + 1u) << 8u); }
                let v_a = q4v_w[row_a * gpr + g];
                let xq = g * 8u;
                let x0 = q4v_x[xq];      let x1 = q4v_x[xq + 1u];
                let x2 = q4v_x[xq + 2u]; let x3 = q4v_x[xq + 3u];
                let x4 = q4v_x[xq + 4u]; let x5 = q4v_x[xq + 5u];
                let x6 = q4v_x[xq + 6u]; let x7 = q4v_x[xq + 7u];
                let sa = lad_q4v[(sub << 5u) + ((cv_a >> sh) & 31u)];
                acc_a = acc_a + sa
                    * (q4v_dot8(v_a.x, x0, x1) + q4v_dot8(v_a.y, x2, x3)
                     + q4v_dot8(v_a.z, x4, x5) + q4v_dot8(v_a.w, x6, x7));
                if (live_b) {
                    var cv_b = q4tp_byte(crow_b + cbo);
                    if (sh > 3u) { cv_b = cv_b | (q4tp_byte(crow_b + cbo + 1u) << 8u); }
                    let v_b = q4v_w[row_b * gpr + g];
                    let sb = lad_q4v[128u + (sub << 5u) + ((cv_b >> sh) & 31u)];
                    acc_b = acc_b + sb
                        * (q4v_dot8(v_b.x, x0, x1) + q4v_dot8(v_b.y, x2, x3)
                         + q4v_dot8(v_b.z, x4, x5) + q4v_dot8(v_b.w, x6, x7));
                }
                g = g + 64u;
            }
        }
        partial_q4v[lid] = acc_a;
        partial_q4vb[lid] = acc_b;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) {
                partial_q4v[lid] = partial_q4v[lid] + partial_q4v[lid + stride];
                partial_q4vb[lid] = partial_q4vb[lid] + partial_q4vb[lid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (l == 0u && row_a < rows) { q1y[row_a] = partial_q4v[sub << 6u]; }
        if (l == 0u && row_b < rows) { q1y[row_b] = partial_q4vb[sub << 6u]; }
        workgroupBarrier();
        base = base + nwg.x * 8u;
    }
}

// ── Fold-select MoE twins: gu/down recompute the top-k FROM THE ROUTER
// LOGITS inside every workgroup — redundant arithmetic, but the serial
// select hop disappears and the layer chain loses one ~25 us dispatch
// latency. The comparator (max, lowest index on ties) is order-free, so
// every workgroup lands on the same experts; softmax summation order
// differs from the retired select kernel only in reduction shape.
// Slot 3 carries the LOGITS where the plain twins carry the selection.
@group(0) @binding(3) var<storage, read> mgf_logit : array<f32>;
struct MgfP { n_exp: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(7) var<uniform> mgf_p : MgfP;

var<workgroup> mgf_lg: array<f32, 256>;
var<workgroup> mgf_v:  array<f32, 64>;
var<workgroup> mgf_i:  array<u32, 64>;

// top-(slot+1) of n logits with 64 lanes; returns the slot'th expert id.
fn mgf_pick(slot: u32, n: u32, lid: u32) -> u32 {
    var chosen = 0u;
    for (var s = 0u; s <= slot; s = s + 1u) {
        var best = -3.0e38;
        var bi = 0xFFFFu;
        var i = lid;
        loop {
            if (i >= n) { break; }
            let v = mgf_lg[i];
            if (v > best || (v == best && i < bi)) { best = v; bi = i; }
            i = i + 64u;
        }
        mgf_v[lid] = best;
        mgf_i[lid] = bi;
        workgroupBarrier();
        var st = 32u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) {
                let b = mgf_v[lid + st];
                let ib = mgf_i[lid + st];
                if (b > mgf_v[lid] || (b == mgf_v[lid] && ib < mgf_i[lid])) {
                    mgf_v[lid] = b;
                    mgf_i[lid] = ib;
                }
            }
            workgroupBarrier();
            st = st >> 1u;
        }
        chosen = mgf_i[0];
        workgroupBarrier();
        if (lid == 0u && s < slot) { mgf_lg[chosen] = -3.0e38; }
        workgroupBarrier();
    }
    return chosen;
}

@compute @workgroup_size(64)
fn moe_gate_up_q2tp_f(@builtin(workgroup_id) wid: vec3<u32>,
                      @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let slot = wid.y;
    let gpr = mg_p.gpr;
    let rows = mg_p.inter;
    let n = mgf_p.n_exp;
    let mat16 = mg_p.mat16;
    // Stage logits once (shared expert = last slot, id n).
    var i = lid;
    loop {
        if (i >= n) { break; }
        mgf_lg[i] = mgf_logit[i];
        i = i + 64u;
    }
    workgroupBarrier();
    var id = n;
    if (slot < mg_p.slots - 1u) {
        id = mgf_pick(slot, n, lid);
    }
    let base16 = id * mat16;
    let nib16 = base16 + row * gpr * 4u;
    let par16 = base16 + rows * gpr * 4u + row * 2u;
    let cst = (gpr * 5u + 7u) / 8u;
    let cod8 = (base16 + rows * gpr * 4u + rows * 2u) * 2u + row * cst;

    let gl = unpack2x16float(mg_g16(par16) | (mg_g16(par16 + 1u) << 16u));
    let ul = unpack2x16float(mg_u16f(par16) | (mg_u16f(par16 + 1u) << 16u));
    var ag = 0.0;
    var au = 0.0;
    for (var g = lid; g < gpr; g = g + 64u) {
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        var cg = mgp_gu8(cod8 + cb);
        var cu = mgp_uu8(cod8 + cb);
        if (shf > 3u) {
            cg = cg | (mgp_gu8(cod8 + cb + 1u) << 8u);
            cu = cu | (mgp_uu8(cod8 + cb + 1u) << 8u);
        }
        let cgv = (cg >> shf) & 31u;
        let cuv = (cu >> shf) & 31u;
        let sg = select(exp2(gl.x + f32(max(cgv, 1u) - 1u) * gl.y), 0.0, cgv == 0u);
        let su = select(exp2(ul.x + f32(max(cuv, 1u) - 1u) * ul.y), 0.0, cuv == 0u);
        let w32 = (nib16 + g * 4u) >> 1u;
        let xq = g * 8u;
        let x0 = mg_xv[xq];      let x1 = mg_xv[xq + 1u];
        let x2 = mg_xv[xq + 2u]; let x3 = mg_xv[xq + 3u];
        let x4 = mg_xv[xq + 4u]; let x5 = mg_xv[xq + 5u];
        let x6 = mg_xv[xq + 6u]; let x7 = mg_xv[xq + 7u];
        let dg = mg_dot16v(mg_gw[w32], x0, x1, x2, x3)
               + mg_dot16v(mg_gw[w32 + 1u], x4, x5, x6, x7);
        let du = mg_dot16v(mg_uw[w32], x0, x1, x2, x3)
               + mg_dot16v(mg_uw[w32 + 1u], x4, x5, x6, x7);
        ag = ag + sg * dg;
        au = au + su * du;
    }
    mg_pg[lid] = ag;
    mg_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            mg_pg[lid] = mg_pg[lid] + mg_pg[lid + stride];
            mg_pu[lid] = mg_pu[lid] + mg_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        let g = mg_pg[0];
        var gg = g;
        var uu = mg_pu[0];
        if (mg_p.lim > 0.0) {
            uu = clamp(uu, -mg_p.lim, mg_p.lim);
            gg = min(gg, mg_p.lim);
        }
        mg_act[slot * mg_p.inter + row] = (gg / (1.0 + exp(-gg))) * uu;
    }
}

// down twin: recomputes ids AND weights (softmax over picked + shared
// sigmoid). Slot 2 = logits, slot 3 = shared-gate weight (f32 bits),
// slot 6 = the token's activations for the shared-gate dot.
@group(0) @binding(2) var<storage, read> mdf_logit : array<f32>;
@group(0) @binding(3) var<storage, read> mdf_sgw   : array<u32>;
@group(0) @binding(6) var<storage, read> mdf_x     : array<f32>;
struct MdfP { n_exp: u32, top_k: u32, norm: u32, pk: u32 };
@group(0) @binding(7) var<uniform> mdf_p : MdfP;

var<workgroup> mdf_lg: array<f32, 256>;
var<workgroup> mdf_v:  array<f32, 64>;
var<workgroup> mdf_i:  array<u32, 64>;
var<workgroup> mdf_sel: array<u32, 16>;
var<workgroup> mdf_wt:  array<f32, 16>;

@compute @workgroup_size(64)
fn moe_down_q4tp_f(@builtin(workgroup_id) wid: vec3<u32>,
                   @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let gpr = md_p.gpr;
    let rows = md_p.hidden;
    let n = mdf_p.n_exp;
    let kk = mdf_p.top_k;
    let sg_kind = mdf_p.pk & 0xFFu;
    let sg_hidden = mdf_p.pk >> 8u;
    // shared gate dot (same strided shape as the retired select kernel,
    // 64 lanes instead of 256)
    var sgv = 0.0;
    if (sg_kind == 4u) {
        var d = 0.0;
        var i = lid;
        loop {
            if (i >= sg_hidden) { break; }
            d = d + bitcast<f32>(mdf_sgw[i]) * mdf_x[i];
            i = i + 64u;
        }
        mdf_v[lid] = d;
        workgroupBarrier();
        var st = 32u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { mdf_v[lid] = mdf_v[lid] + mdf_v[lid + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        sgv = mdf_v[0];
        workgroupBarrier();
    }
    // logits + max + denom with 64-lane reductions
    var i2 = lid;
    loop {
        if (i2 >= n) { break; }
        mdf_lg[i2] = mdf_logit[i2];
        i2 = i2 + 64u;
    }
    workgroupBarrier();
    var mbest = -3.0e38;
    var i3 = lid;
    loop {
        if (i3 >= n) { break; }
        mbest = max(mbest, mdf_lg[i3]);
        i3 = i3 + 64u;
    }
    mdf_v[lid] = mbest;
    workgroupBarrier();
    var st2 = 32u;
    loop {
        if (st2 == 0u) { break; }
        if (lid < st2) { mdf_v[lid] = max(mdf_v[lid], mdf_v[lid + st2]); }
        workgroupBarrier();
        st2 = st2 >> 1u;
    }
    let mx = mdf_v[0];
    workgroupBarrier();
    var dsum = 0.0;
    var i4 = lid;
    loop {
        if (i4 >= n) { break; }
        dsum = dsum + exp(mdf_lg[i4] - mx);
        i4 = i4 + 64u;
    }
    mdf_v[lid] = dsum;
    workgroupBarrier();
    st2 = 32u;
    loop {
        if (st2 == 0u) { break; }
        if (lid < st2) { mdf_v[lid] = mdf_v[lid] + mdf_v[lid + st2]; }
        workgroupBarrier();
        st2 = st2 >> 1u;
    }
    let denom = mdf_v[0];
    workgroupBarrier();
    // top-k, weights, optional renorm; shared expert last
    var wsum = 0.0;
    for (var s = 0u; s < kk; s = s + 1u) {
        var best = -3.0e38;
        var bi = 0xFFFFu;
        var i5 = lid;
        loop {
            if (i5 >= n) { break; }
            let v = mdf_lg[i5];
            if (v > best || (v == best && i5 < bi)) { best = v; bi = i5; }
            i5 = i5 + 64u;
        }
        mdf_v[lid] = best;
        mdf_i[lid] = bi;
        workgroupBarrier();
        var st3 = 32u;
        loop {
            if (st3 == 0u) { break; }
            if (lid < st3) {
                let b = mdf_v[lid + st3];
                let ib = mdf_i[lid + st3];
                if (b > mdf_v[lid] || (b == mdf_v[lid] && ib < mdf_i[lid])) {
                    mdf_v[lid] = b;
                    mdf_i[lid] = ib;
                }
            }
            workgroupBarrier();
            st3 = st3 >> 1u;
        }
        if (lid == 0u) {
            mdf_sel[s] = mdf_i[0];
            mdf_wt[s] = exp(mdf_v[0] - mx) / denom;
        }
        workgroupBarrier();
        wsum = wsum + exp(mdf_v[0] - mx) / denom;
        if (lid == 0u) { mdf_lg[mdf_i[0]] = -3.0e38; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        if (mdf_p.norm != 0u) {
            for (var s = 0u; s < kk; s = s + 1u) { mdf_wt[s] = mdf_wt[s] / wsum; }
        }
        mdf_sel[kk] = n;
        mdf_wt[kk] = 1.0 / (1.0 + exp(-sgv));
    }
    workgroupBarrier();
    let cst = (gpr * 5u + 7u) / 8u;
    let total = md_p.slots * gpr;
    var acc = 0.0;
    for (var i6 = lid; i6 < total; i6 = i6 + 64u) {
        let slot = i6 / gpr;
        let g = i6 % gpr;
        let base16 = mdf_sel[slot] * md_p.mat16;
        let par16 = base16 + rows * gpr * 8u + row * 2u;
        let cod8 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row * cst;
        let pl = unpack2x16float(md_u16(par16) | (md_u16(par16 + 1u) << 16u));
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        var cv = mdp_u8(cod8 + cb);
        if (shf > 3u) { cv = cv | (mdp_u8(cod8 + cb + 1u) << 8u); }
        let scale = exp2(pl.x + f32((cv >> shf) & 31u) * pl.y);
        let t16 = base16 + (row * gpr + g) * 8u;
        let xb = (slot * gpr + g) * 32u;
        var d = 0.0;
        for (var k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            let w = md_u16(t16 + 2u * k2) | (md_u16(t16 + 1u + 2u * k2) << 16u);
            d = d + md_dot8(w, xb + 8u * k2);
        }
        acc = acc + mdf_wt[slot] * scale * d;
    }
    md_pt[lid] = acc;
    workgroupBarrier();
    var st4 = 32u;
    loop {
        if (st4 == 0u) { break; }
        if (lid < st4) { md_pt[lid] = md_pt[lid] + md_pt[lid + st4]; }
        workgroupBarrier();
        st4 = st4 >> 1u;
    }
    if (lid == 0u) { md_y[row] = md_pt[0]; }
}

// ── Multi-step greedy tail: argmax over the logits on the device, then
// re-embed the winner — k decode steps ride ONE submit and the CPU sees
// k token ids instead of k megabytes of logits. Ties pick an arbitrary
// maximal index (same class as the documented GPU float-order ties).
struct AmP { n: u32, parts: u32, st: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       am_x   : array<f32>;
@group(0) @binding(1) var<storage, read_write> am_pv  : array<f32>;
@group(0) @binding(2) var<storage, read_write> am_pi  : array<u32>;
@group(0) @binding(3) var<uniform>             am_p   : AmP;
var<workgroup> am_wv: array<f32, 256>;
var<workgroup> am_wi: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax_part(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    var best = -3.0e38;
    var bi = 0u;
    var i = wid.x * 256u + lid;
    let stride = am_p.parts * 256u;
    loop {
        if (i >= am_p.n) { break; }
        let v = am_x[i];
        if (v > best) { best = v; bi = i; }
        i = i + stride;
    }
    am_wv[lid] = best;
    am_wi[lid] = bi;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s && am_wv[lid + s] > am_wv[lid]) {
            am_wv[lid] = am_wv[lid + s];
            am_wi[lid] = am_wi[lid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) {
        am_pv[wid.x] = am_wv[0];
        am_pi[wid.x] = am_wi[0];
    }
}

@group(0) @binding(0) var<storage, read>       af_pv : array<f32>;
@group(0) @binding(1) var<storage, read>       af_pi : array<u32>;
@group(0) @binding(2) var<storage, read_write> af_ids: array<u32>;
@group(0) @binding(3) var<uniform>             af_p  : AmP;
var<workgroup> af_wv: array<f32, 256>;
var<workgroup> af_wi: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax_final(@builtin(local_invocation_index) lid: u32) {
    var best = -3.0e38;
    var bi = 0u;
    var i = lid;
    loop {
        if (i >= af_p.parts) { break; }
        if (af_pv[i] > best) { best = af_pv[i]; bi = af_pi[i]; }
        i = i + 256u;
    }
    af_wv[lid] = best;
    af_wi[lid] = bi;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s && af_wv[lid + s] > af_wv[lid]) {
            af_wv[lid] = af_wv[lid + s];
            af_wi[lid] = af_wi[lid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) { af_ids[af_p.st] = af_wi[0]; }
}

// One thread = one hidden element of the winner's q4tp embedding row.
// `mult` carries the model's embed multiplier as f32 bits.
struct EgP { hidden: u32, gpr: u32, rows: u32, st: u32, mult: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       eg_w  : array<u32>;
@group(0) @binding(1) var<storage, read>       eg_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> eg_h  : array<f32>;
@group(0) @binding(3) var<uniform>             eg_p  : EgP;

fn eg_byte(off: u32) -> u32 {
    return (eg_w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn embed_gather_q4tp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= eg_p.hidden) { return; }
    let r = eg_ids[eg_p.st];
    let gpr = eg_p.gpr;
    let g = i / 32u;
    let k = i % 32u;
    let params_w = eg_p.rows * gpr * 4u;
    let codes_b = eg_p.rows * gpr * 16u + eg_p.rows * 4u;
    let cst = (gpr * 5u + 7u) / 8u;
    let pr = unpack2x16float(eg_w[params_w + r]);
    let bit = g * 5u;
    let cb = codes_b + r * cst + (bit >> 3u);
    let sh = bit & 7u;
    var cv = eg_byte(cb);
    if (sh > 3u) { cv = cv | (eg_byte(cb + 1u) << 8u); }
    let sc = exp2(pr.x + f32((cv >> sh) & 31u) * pr.y);
    let nb = (r * gpr + g) * 16u + (k >> 1u);
    let byte = eg_byte(nb);
    let q = select(byte & 0xFu, (byte >> 4u) & 0xFu, (k & 1u) == 1u);
    eg_h[i] = (f32(q) - 8.0) * sc * bitcast<f32>(eg_p.mult);
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
// The same activations as vec4 (same-slot rule): the GEMM stages four
// consecutive floats per thread per K-step, and col0 is a multiple of 4,
// so that is one 16-byte load instead of four scalar ones.
@group(0) @binding(1) var<storage, read>       xmm4 : array<vec4<f32>>;
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
    // Sixteen named scalars, not array<array<f32,4>,4> — see q4t_mul_mm.
    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let m = t / 4u;
            let k4 = t % 4u;
            var xv = vec4<f32>(0.0);
            let col0 = k0 + k4 * 4u;
            if (m0 + m < pmm.nb && col0 < cols) {
                // cols is a multiple of 32 and col0 of 4 — vec4-aligned.
                xv = xmm4[((m0 + m) * cols + col0) >> 2u];
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
        let ab = lid.y * 64u;
        let wb = lid.x * 64u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let x0 = q1t_at[ab + k];
            let x1 = q1t_at[ab + 16u + k];
            let x2 = q1t_at[ab + 32u + k];
            let x3 = q1t_at[ab + 48u + k];
            let y0 = q1t_wt[wb + k];
            let y1 = q1t_wt[wb + 16u + k];
            let y2 = q1t_wt[wb + 32u + k];
            let y3 = q1t_wt[wb + 48u + k];
            a00 = a00 + x0 * y0; a01 = a01 + x0 * y1;
            a02 = a02 + x0 * y2; a03 = a03 + x0 * y3;
            a10 = a10 + x1 * y0; a11 = a11 + x1 * y1;
            a12 = a12 + x1 * y2; a13 = a13 + x1 * y3;
            a20 = a20 + x2 * y0; a21 = a21 + x2 * y1;
            a22 = a22 + x2 * y2; a23 = a23 + x2 * y3;
            a30 = a30 + x3 * y0; a31 = a31 + x3 * y1;
            a32 = a32 + x3 * y2; a33 = a33 + x3 * y3;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    let mb = m0 + lid.y * 4u;
    let nb2 = n0 + lid.x * 4u;
    q4t_store4(mb, nb2, a00, a01, a02, a03);
    q4t_store4(mb + 1u, nb2, a10, a11, a12, a13);
    q4t_store4(mb + 2u, nb2, a20, a21, a22, a23);
    q4t_store4(mb + 3u, nb2, a30, a31, a32, a33);
}

// q4t register-blocked GEMM (imagegen DiT prefill / any wide q4t
// batch) — the WGSL cousin of the Metal q4t_mul_mm and structurally
// identical to q1t_mul_mm above; only the W staging decodes 18-byte
// q4t tiles (f16 scale + 16 nibble bytes per 32-weight group).
// Shares the 4-slot qmm/xmm/ymm/pmm bindings.
var<workgroup> q4t_at: array<f32, 64 * 16>;
var<workgroup> q4t_wt: array<f32, 64 * 16>;

fn q4t_store4(m: u32, n0: u32, v0: f32, v1: f32, v2: f32, v3: f32) {
    if (m >= pmm.nb) { return; }
    let base = m * pmm.rows + n0;
    if (n0 < pmm.rows) { ymm[base] = v0; }
    if (n0 + 1u < pmm.rows) { ymm[base + 1u] = v1; }
    if (n0 + 2u < pmm.rows) { ymm[base + 2u] = v2; }
    if (n0 + 3u < pmm.rows) { ymm[base + 3u] = v3; }
}

// q4tp register-blocked GEMM — the q4t kernel above with one block swapped:
// a 16 B nibble stride instead of the 18 B tile, and the scale off the row's
// ladder. Shares q4t_store4 and the q4t_at/q4t_wt staging arrays; only one
// entry point runs per dispatch, so the workgroup allocation is not doubled.
@compute @workgroup_size(16, 16)
fn q4tp_mul_mm(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = pmm.cols4 * 4u;
    let gpr = cols >> 5u;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    // The 4x4 register block is SIXTEEN NAMED SCALARS, not
    // array<array<f32,4>,4>: indexed by loop variables the array is a
    // private array, which this backend puts in stack memory — the
    // accumulators leave registers and the GEMM runs at a fraction of
    // the card (measured 373 GFLOP/s of an RTX 3090's ~35 TFLOP/s).
    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let m = t / 4u;
            let k4 = t % 4u;
            var xv = vec4<f32>(0.0);
            let col0 = k0 + k4 * 4u;
            if (m0 + m < pmm.nb && col0 < cols) {
                // cols is a multiple of 32 and col0 of 4 — vec4-aligned.
                xv = xmm4[((m0 + m) * cols + col0) >> 2u];
            }
            let dst = m * 16u + k4 * 4u;
            q4t_at[dst] = xv.x; q4t_at[dst + 1u] = xv.y;
            q4t_at[dst + 2u] = xv.z; q4t_at[dst + 3u] = xv.w;
        }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let n = t / 4u;
            let k4 = t % 4u;
            var wv = vec4<f32>(0.0);
            let col0 = k0 + k4 * 4u;
            if (n0 + n < pmm.rows && col0 < cols) {
                let g = col0 >> 5u;
                let wrow = n0 + n;
                let params_b = pmm.rows * gpr * 16u;
                let codes_b = params_b + pmm.rows * 4u;
                let cstride = (gpr * 5u + 7u) / 8u;
                let bit = g * 5u;
                let cb = codes_b + wrow * cstride + (bit >> 3u);
                let sh = bit & 7u;
                var cv = qmm_byte(cb);
                if (sh > 3u) { cv = cv | (qmm_byte(cb + 1u) << 8u); }
                // One exp2 per staged group of 4 — this thread stages exactly
                // one such group per K-step, and the GEMM's arithmetic hides
                // the chain that the matvec had to hoist out of its tile loop.
                let pr = unpack2x16float(qmm[(params_b >> 2u) + wrow]);
                let scale = exp2(pr.x + f32((cv >> sh) & 31u) * pr.y);
                // 4 consecutive weights = 2 nibble bytes (col0 is even).
                let toff = (wrow * gpr + g) * 16u;
                let p = col0 - g * 32u;
                // The two nibble bytes are adjacent: one u32 covers both
                // unless they straddle a word boundary (one case in four).
                let bo = toff + p / 2u;
                let w32 = qmm[bo >> 2u];
                let sh0 = (bo & 3u) * 8u;
                let b0 = (w32 >> sh0) & 0xFFu;
                var b1 = 0u;
                if ((bo & 3u) == 3u) {
                    b1 = qmm[(bo >> 2u) + 1u] & 0xFFu;
                } else {
                    b1 = (w32 >> (sh0 + 8u)) & 0xFFu;
                }
                wv[0u] = (f32(b0 & 0xFu) - 8.0) * scale;
                wv[1u] = (f32(b0 >> 4u) - 8.0) * scale;
                wv[2u] = (f32(b1 & 0xFu) - 8.0) * scale;
                wv[3u] = (f32(b1 >> 4u) - 8.0) * scale;
            }
            let dst = n * 16u + k4 * 4u;
            q4t_wt[dst] = wv.x; q4t_wt[dst + 1u] = wv.y;
            q4t_wt[dst + 2u] = wv.z; q4t_wt[dst + 3u] = wv.w;
        }
        workgroupBarrier();
        let ab = lid.y * 64u;
        let wb = lid.x * 64u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let x0 = q4t_at[ab + k];
            let x1 = q4t_at[ab + 16u + k];
            let x2 = q4t_at[ab + 32u + k];
            let x3 = q4t_at[ab + 48u + k];
            let y0 = q4t_wt[wb + k];
            let y1 = q4t_wt[wb + 16u + k];
            let y2 = q4t_wt[wb + 32u + k];
            let y3 = q4t_wt[wb + 48u + k];
            a00 = a00 + x0 * y0; a01 = a01 + x0 * y1;
            a02 = a02 + x0 * y2; a03 = a03 + x0 * y3;
            a10 = a10 + x1 * y0; a11 = a11 + x1 * y1;
            a12 = a12 + x1 * y2; a13 = a13 + x1 * y3;
            a20 = a20 + x2 * y0; a21 = a21 + x2 * y1;
            a22 = a22 + x2 * y2; a23 = a23 + x2 * y3;
            a30 = a30 + x3 * y0; a31 = a31 + x3 * y1;
            a32 = a32 + x3 * y2; a33 = a33 + x3 * y3;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    let mb = m0 + lid.y * 4u;
    let nb2 = n0 + lid.x * 4u;
    q4t_store4(mb, nb2, a00, a01, a02, a03);
    q4t_store4(mb + 1u, nb2, a10, a11, a12, a13);
    q4t_store4(mb + 2u, nb2, a20, a21, a22, a23);
    q4t_store4(mb + 3u, nb2, a30, a31, a32, a33);
}

@compute @workgroup_size(16, 16)
fn q4t_mul_mm(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = pmm.cols4 * 4u;
    let gpr = cols >> 5u;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    // The 4x4 register block is SIXTEEN NAMED SCALARS, not
    // array<array<f32,4>,4>: indexed by loop variables the array is a
    // private array, which this backend puts in stack memory — the
    // accumulators leave registers and the GEMM runs at a fraction of
    // the card (measured 373 GFLOP/s of an RTX 3090's ~35 TFLOP/s).
    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let m = t / 4u;
            let k4 = t % 4u;
            var xv = vec4<f32>(0.0);
            let col0 = k0 + k4 * 4u;
            if (m0 + m < pmm.nb && col0 < cols) {
                // cols is a multiple of 32 and col0 of 4 — vec4-aligned.
                xv = xmm4[((m0 + m) * cols + col0) >> 2u];
            }
            let dst = m * 16u + k4 * 4u;
            q4t_at[dst] = xv.x; q4t_at[dst + 1u] = xv.y;
            q4t_at[dst + 2u] = xv.z; q4t_at[dst + 3u] = xv.w;
        }
        for (var t = tid; t < 64u * 4u; t = t + 256u) {
            let n = t / 4u;
            let k4 = t % 4u;
            var wv = vec4<f32>(0.0);
            let col0 = k0 + k4 * 4u;
            if (n0 + n < pmm.rows && col0 < cols) {
                let g = col0 >> 5u;
                let toff = ((n0 + n) * gpr + g) * 18u;
                let sc16 = qmm_byte(toff) | (qmm_byte(toff + 1u) << 8u);
                let scale = unpack2x16float(sc16).x;
                // 4 consecutive weights = 2 nibble bytes (col0 is even).
                let p = col0 - g * 32u;
                let b0 = qmm_byte(toff + 2u + p / 2u);
                let b1 = qmm_byte(toff + 3u + p / 2u);
                wv[0u] = (f32(b0 & 0xFu) - 8.0) * scale;
                wv[1u] = (f32(b0 >> 4u) - 8.0) * scale;
                wv[2u] = (f32(b1 & 0xFu) - 8.0) * scale;
                wv[3u] = (f32(b1 >> 4u) - 8.0) * scale;
            }
            let dst = n * 16u + k4 * 4u;
            q4t_wt[dst] = wv.x; q4t_wt[dst + 1u] = wv.y;
            q4t_wt[dst + 2u] = wv.z; q4t_wt[dst + 3u] = wv.w;
        }
        workgroupBarrier();
        let ab = lid.y * 64u;
        let wb = lid.x * 64u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let x0 = q4t_at[ab + k];
            let x1 = q4t_at[ab + 16u + k];
            let x2 = q4t_at[ab + 32u + k];
            let x3 = q4t_at[ab + 48u + k];
            let y0 = q4t_wt[wb + k];
            let y1 = q4t_wt[wb + 16u + k];
            let y2 = q4t_wt[wb + 32u + k];
            let y3 = q4t_wt[wb + 48u + k];
            a00 = a00 + x0 * y0; a01 = a01 + x0 * y1;
            a02 = a02 + x0 * y2; a03 = a03 + x0 * y3;
            a10 = a10 + x1 * y0; a11 = a11 + x1 * y1;
            a12 = a12 + x1 * y2; a13 = a13 + x1 * y3;
            a20 = a20 + x2 * y0; a21 = a21 + x2 * y1;
            a22 = a22 + x2 * y2; a23 = a23 + x2 * y3;
            a30 = a30 + x3 * y0; a31 = a31 + x3 * y1;
            a32 = a32 + x3 * y2; a33 = a33 + x3 * y3;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    let mb = m0 + lid.y * 4u;
    let nb2 = n0 + lid.x * 4u;
    q4t_store4(mb, nb2, a00, a01, a02, a03);
    q4t_store4(mb + 1u, nb2, a10, a11, a12, a13);
    q4t_store4(mb + 2u, nb2, a20, a21, a22, a23);
    q4t_store4(mb + 3u, nb2, a30, a31, a32, a33);
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

// ── MoE inside the whole-token graph ────────────────────────────────────────
// Router logits/shared-gate logit arrive from ordinary matvecs; these three
// kernels keep the routing DECISION and every selected expert on-device, so
// a MoE layer costs one extra pass over a dense one instead of a CPU sync.
// Expert weights live in three per-layer concat buffers (q4t tiles, expert e
// at u16 offset e·mat16); the SHARED expert is the last block, pinned by the
// select kernel at slot top_k with a sigmoid weight.

// `sg_kind` = 4 means this kernel computes the shared-expert gate itself
// from `ms_sgw` · `ms_x` and the host skips that matvec entirely. It is a
// ONE-ROW projection: 2048 multiply-adds for a whole dispatch, and a
// dispatch costs ~23 us on this stack against ~5 us for a pass — measured
// by sweeping the layer count. Folding it into a kernel that already runs
// one workgroup is free. Any other dtype keeps the separate matvec and
// this kernel reads its result from `ms_slog`.
struct MoeSelP { n_exp: u32, top_k: u32, norm: u32, pk: u32 };
@group(0) @binding(0) var<storage, read>       ms_logit : array<f32>;
@group(0) @binding(1) var<storage, read>       ms_slog  : array<f32>;
@group(0) @binding(2) var<storage, read_write> ms_sel   : array<u32>;
@group(0) @binding(3) var<storage, read_write> ms_w     : array<f32>;
@group(0) @binding(4) var<uniform>             ms_p     : MoeSelP;
@group(0) @binding(5) var<storage, read>       ms_sgw   : array<u32>;
@group(0) @binding(6) var<storage, read>       ms_x     : array<f32>;
var<workgroup> ms_lg:  array<f32, 256>;
var<workgroup> ms_red: array<f32, 256>;
var<workgroup> ms_ri:  array<u32, 256>;
var<workgroup> ms_pick: u32;
var<workgroup> ms_sg:  f32;

// One workgroup, ALL-parallel: softmax reductions, then k rounds of an
// argmax reduce (the selected logit is neutralized between rounds). A
// serial one-thread top-k here measured ~270 ns per L2-latency-bound
// probe — 22 ms/token across 40 layers at k=8, the whole decode wall.
// Ties pick the LOWEST index, matching the CPU scan. n_exp ≤ 256.
@compute @workgroup_size(256)
fn moe_select(@builtin(local_invocation_index) lid: u32) {
    // Shared-expert gate first: the reduction scratch below is reused, so
    // this has to land in ms_sg before the router work starts.
    let sg_kind = ms_p.pk & 0xFFu;
    let sg_hidden = ms_p.pk >> 8u;
    if (sg_kind == 4u) {
        var d = 0.0;
        var i = lid;
        loop {
            if (i >= sg_hidden) { break; }
            d = d + bitcast<f32>(ms_sgw[i]) * ms_x[i];
            i = i + 256u;
        }
        ms_red[lid] = d;
        workgroupBarrier();
        var st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { ms_red[lid] = ms_red[lid] + ms_red[lid + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        if (lid == 0u) { ms_sg = ms_red[0]; }
    } else {
        if (lid == 0u) { ms_sg = ms_slog[0]; }
    }
    workgroupBarrier();
    let n = ms_p.n_exp;
    var v = -3.0e38;
    if (lid < n) { v = ms_logit[lid]; }
    ms_lg[lid] = v;
    ms_red[lid] = v;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { ms_red[lid] = max(ms_red[lid], ms_red[lid + stride]); }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let mx = ms_red[0];
    workgroupBarrier();
    ms_red[lid] = select(0.0, exp(v - mx), lid < n);
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { ms_red[lid] = ms_red[lid] + ms_red[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let denom = ms_red[0];
    workgroupBarrier();
    let k = ms_p.top_k;
    var wsum = 0.0;
    for (var slot = 0u; slot < k; slot = slot + 1u) {
        ms_red[lid] = ms_lg[lid];
        ms_ri[lid] = lid;
        workgroupBarrier();
        stride = 128u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) {
                let a = ms_red[lid];
                let b = ms_red[lid + stride];
                let ia = ms_ri[lid];
                let ib = ms_ri[lid + stride];
                if (b > a || (b == a && ib < ia)) {
                    ms_red[lid] = b;
                    ms_ri[lid] = ib;
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) {
            let bi = ms_ri[0];
            ms_sel[slot] = bi;
            ms_w[slot] = exp(ms_red[0] - mx) / denom;
            ms_pick = bi;
        }
        workgroupBarrier();
        wsum = wsum + exp(ms_red[0] - mx) / denom;
        if (lid == ms_pick) { ms_lg[lid] = -3.0e38; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        if (ms_p.norm != 0u) {
            for (var slot = 0u; slot < k; slot = slot + 1u) { ms_w[slot] = ms_w[slot] / wsum; }
        }
        ms_sel[k] = n;
        ms_w[k] = 1.0 / (1.0 + exp(-ms_sg));
    }
}

// gate+up+SiLU for every selected expert: workgroup (row, slot) does BOTH q4t
// row dots (they share the activation reads) and writes act = silu(g)·u.
struct MoeGuP { gpr: u32, inter: u32, slots: u32, mat16: u32 , lim: f32, _p0: u32, _p1: u32, _p2: u32 };
// Three scalars, not a vec3: a vec3<u32> aligns to 16 in uniform layout and
// pushes the struct to 48 bytes, while the buffer handed in is 32.
// `lim` is DeepSeek-V4's swiglu_limit: the up projection is clamped both
// ways, the gate only from above. Zero means no clamp, which is every other
// architecture that reaches these kernels.
@group(0) @binding(0) var<storage, read>       mg_gw  : array<u32>;
@group(0) @binding(1) var<storage, read>       mg_uw  : array<u32>;
@group(0) @binding(2) var<storage, read>       mg_x   : array<f32>;
@group(0) @binding(3) var<storage, read>       mg_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> mg_act : array<f32>;
@group(0) @binding(5) var<uniform>             mg_p   : MoeGuP;
var<workgroup> mg_pg: array<f32, 64>;
var<workgroup> mg_pu: array<f32, 64>;

// The same activations as `mg_x`, at the SAME SLOT, seen as vec4. The
// scalar view costs one load per weight, which left the dense FFN kernel
// at ~11% of the card's bandwidth until the vec4 rewrite (+2.7x there).
//
// A second global on slot 2 rather than a new slot 6, because an auto
// layout lists only the bindings its entry point actually USES: the q2tp
// kernel stopped touching `mg_x`, naga dropped slot 2, and the 7-entry
// bind group met a 6-entry layout. Same slot = the bind group is
// unchanged for every kernel here.
@group(0) @binding(2) var<storage, read> mg_xv : array<vec4<f32>>;

fn mg_g16(o: u32) -> u32 { return (mg_gw[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn mg_u16f(o: u32) -> u32 { return (mg_uw[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn mg_dot8(w: u32, xi: u32) -> f32 {
    return (f32(w & 0xFu) - 8.0) * mg_x[xi]
         + (f32((w >> 4u) & 0xFu) - 8.0) * mg_x[xi + 1u]
         + (f32((w >> 8u) & 0xFu) - 8.0) * mg_x[xi + 2u]
         + (f32((w >> 12u) & 0xFu) - 8.0) * mg_x[xi + 3u]
         + (f32((w >> 16u) & 0xFu) - 8.0) * mg_x[xi + 4u]
         + (f32((w >> 20u) & 0xFu) - 8.0) * mg_x[xi + 5u]
         + (f32((w >> 24u) & 0xFu) - 8.0) * mg_x[xi + 6u]
         + (f32((w >> 28u) & 0xFu) - 8.0) * mg_x[xi + 7u];
}

@compute @workgroup_size(64)
fn moe_gate_up(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let slot = wid.y;
    let gpr = mg_p.gpr;
    let base = mg_sel[slot] * mg_p.mat16 + row * gpr * 9u;
    var ag = 0.0;
    var au = 0.0;
    for (var g = lid; g < gpr; g = g + 64u) {
        let t16 = base + g * 9u;
        let sg = unpack2x16float(mg_g16(t16)).x;
        let su = unpack2x16float(mg_u16f(t16)).x;
        let xb = g * 32u;
        var dg = 0.0;
        var du = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let wg = mg_g16(t16 + 1u + 2u * k) | (mg_g16(t16 + 2u + 2u * k) << 16u);
            let wu = mg_u16f(t16 + 1u + 2u * k) | (mg_u16f(t16 + 2u + 2u * k) << 16u);
            dg = dg + mg_dot8(wg, xb + 8u * k);
            du = du + mg_dot8(wu, xb + 8u * k);
        }
        ag = ag + sg * dg;
        au = au + su * du;
    }
    mg_pg[lid] = ag;
    mg_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            mg_pg[lid] = mg_pg[lid] + mg_pg[lid + stride];
            mg_pu[lid] = mg_pu[lid] + mg_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        let g = mg_pg[0];
        var gg = g;
        var uu = mg_pu[0];
        if (mg_p.lim > 0.0) {
            uu = clamp(uu, -mg_p.lim, mg_p.lim);
            gg = min(gg, mg_p.lim);
        }
        mg_act[slot * mg_p.inter + row] = (gg / (1.0 + exp(-gg))) * uu;
    }
}

// Weighted down-projection: one workgroup per hidden row accumulates
// Σ_slot w[slot]·(down[sel[slot]] row · act[slot]) over the flattened
// (slot, group) space, then overwrites y[row] (the graph's usual FFN
// output slot — the existing fused residual add consumes it).
struct MoeDnP { gpr: u32, hidden: u32, slots: u32, mat16: u32 };
@group(0) @binding(0) var<storage, read>       md_w   : array<u32>;
@group(0) @binding(1) var<storage, read>       md_act : array<f32>;
@group(0) @binding(2) var<storage, read>       md_sel : array<u32>;
@group(0) @binding(3) var<storage, read>       md_wt  : array<f32>;
@group(0) @binding(4) var<storage, read_write> md_y   : array<f32>;
@group(0) @binding(5) var<uniform>             md_p   : MoeDnP;
var<workgroup> md_pt: array<f32, 64>;

// The activations again as vec4, on the SAME slot — the scalar view costs
// one load per weight and put the down kernel at 33 us for ~5 MB of reads.
@group(0) @binding(1) var<storage, read> md_actv : array<vec4<f32>>;

fn md_u16(o: u32) -> u32 { return (md_w[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn md_dot8v(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return (f32(w & 0xFu) - 8.0) * a.x
         + (f32((w >> 4u) & 0xFu) - 8.0) * a.y
         + (f32((w >> 8u) & 0xFu) - 8.0) * a.z
         + (f32((w >> 12u) & 0xFu) - 8.0) * a.w
         + (f32((w >> 16u) & 0xFu) - 8.0) * b.x
         + (f32((w >> 20u) & 0xFu) - 8.0) * b.y
         + (f32((w >> 24u) & 0xFu) - 8.0) * b.z
         + (f32((w >> 28u) & 0xFu) - 8.0) * b.w;
}
fn md_dot8(w: u32, xi: u32) -> f32 {
    return (f32(w & 0xFu) - 8.0) * md_act[xi]
         + (f32((w >> 4u) & 0xFu) - 8.0) * md_act[xi + 1u]
         + (f32((w >> 8u) & 0xFu) - 8.0) * md_act[xi + 2u]
         + (f32((w >> 12u) & 0xFu) - 8.0) * md_act[xi + 3u]
         + (f32((w >> 16u) & 0xFu) - 8.0) * md_act[xi + 4u]
         + (f32((w >> 20u) & 0xFu) - 8.0) * md_act[xi + 5u]
         + (f32((w >> 24u) & 0xFu) - 8.0) * md_act[xi + 6u]
         + (f32((w >> 28u) & 0xFu) - 8.0) * md_act[xi + 7u];
}

@compute @workgroup_size(64)
fn moe_down(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let gpr = md_p.gpr;
    let total = md_p.slots * gpr;
    var acc = 0.0;
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i % gpr;
        let t16 = md_sel[slot] * md_p.mat16 + (row * gpr + g) * 9u;
        let scale = unpack2x16float(md_u16(t16)).x;
        let xb = (slot * gpr + g) * 32u;
        var d = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let w = md_u16(t16 + 1u + 2u * k) | (md_u16(t16 + 2u + 2u * k) << 16u);
            d = d + md_dot8(w, xb + 8u * k);
        }
        acc = acc + md_wt[slot] * scale * d;
    }
    md_pt[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { md_pt[lid] = md_pt[lid] + md_pt[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) { md_y[row] = md_pt[0]; }
}

// ── q4tp twins of the two MoE kernels. Identical nibble math and
// identical bindings; only where the scale comes from differs. q4t
// carries an f16 scale inside each 18-byte tile, q4tp packs the nibbles
// 16-byte tight and puts a 5-bit rung index into a side plane, read
// against the row's geometric ladder `2^(lo + code·step)`. Per expert
// the blob is [nibbles | row params (f16 lo, f16 step) | 5-bit codes],
// so the two extra plane offsets fall out of rows/gpr.
fn mgp_gu8(o: u32) -> u32 { return (mg_gw[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu; }
fn mgp_uu8(o: u32) -> u32 { return (mg_uw[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu; }

@compute @workgroup_size(64)
fn moe_gate_up_q4tp(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let slot = wid.y;
    let gpr = mg_p.gpr;
    let rows = mg_p.inter;
    let base16 = mg_sel[slot] * mg_p.mat16;
    let nib16 = base16 + row * gpr * 8u;
    let par16 = base16 + rows * gpr * 8u + row * 2u;
    let cst = (gpr * 5u + 7u) / 8u;
    let cod8 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row * cst;

    let gl = unpack2x16float(mg_g16(par16) | (mg_g16(par16 + 1u) << 16u));
    let ul = unpack2x16float(mg_u16f(par16) | (mg_u16f(par16 + 1u) << 16u));
    var ag = 0.0;
    var au = 0.0;
    for (var g = lid; g < gpr; g = g + 64u) {
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        var cg = mgp_gu8(cod8 + cb);
        var cu = mgp_uu8(cod8 + cb);
        // A 5-bit field starting past bit 3 spills into the next byte.
        if (shf > 3u) {
            cg = cg | (mgp_gu8(cod8 + cb + 1u) << 8u);
            cu = cu | (mgp_uu8(cod8 + cb + 1u) << 8u);
        }
        let sg = exp2(gl.x + f32((cg >> shf) & 31u) * gl.y);
        let su = exp2(ul.x + f32((cu >> shf) & 31u) * ul.y);
        let t16 = nib16 + g * 8u;
        let xb = g * 32u;
        var dg = 0.0;
        var du = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let wg = mg_g16(t16 + 2u * k) | (mg_g16(t16 + 1u + 2u * k) << 16u);
            let wu = mg_u16f(t16 + 2u * k) | (mg_u16f(t16 + 1u + 2u * k) << 16u);
            dg = dg + mg_dot8(wg, xb + 8u * k);
            du = du + mg_dot8(wu, xb + 8u * k);
        }
        ag = ag + sg * dg;
        au = au + su * du;
    }
    mg_pg[lid] = ag;
    mg_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            mg_pg[lid] = mg_pg[lid] + mg_pg[lid + stride];
            mg_pu[lid] = mg_pu[lid] + mg_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        let g = mg_pg[0];
        var gg = g;
        var uu = mg_pu[0];
        if (mg_p.lim > 0.0) {
            uu = clamp(uu, -mg_p.lim, mg_p.lim);
            gg = min(gg, mg_p.lim);
        }
        mg_act[slot * mg_p.inter + row] = (gg / (1.0 + exp(-gg))) * uu;
    }
}

// q2tp gate/up: the q4tp kernel with a 2-bit weight plane. A group is
// 32 weights in 8 bytes (4 u16 units) instead of 16, and one u32 carries
// SIXTEEN weights, so the group is two words and two dot16s. The params
// and 5-bit code planes are byte-identical to q4tp — only the plane
// offsets move, since they sit behind a half-size weight plane.
// 16 two-bit weights against four staged vec4s — same add order as the
// scalar mg_dot16 below, which greedy parity depends on.
fn mg_dot16v(w: u32, a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>) -> f32 {
    return (f32(w & 3u) - 1.5) * a.x
         + (f32((w >> 2u) & 3u) - 1.5) * a.y
         + (f32((w >> 4u) & 3u) - 1.5) * a.z
         + (f32((w >> 6u) & 3u) - 1.5) * a.w
         + (f32((w >> 8u) & 3u) - 1.5) * b.x
         + (f32((w >> 10u) & 3u) - 1.5) * b.y
         + (f32((w >> 12u) & 3u) - 1.5) * b.z
         + (f32((w >> 14u) & 3u) - 1.5) * b.w
         + (f32((w >> 16u) & 3u) - 1.5) * c.x
         + (f32((w >> 18u) & 3u) - 1.5) * c.y
         + (f32((w >> 20u) & 3u) - 1.5) * c.z
         + (f32((w >> 22u) & 3u) - 1.5) * c.w
         + (f32((w >> 24u) & 3u) - 1.5) * d.x
         + (f32((w >> 26u) & 3u) - 1.5) * d.y
         + (f32((w >> 28u) & 3u) - 1.5) * d.z
         + (f32((w >> 30u) & 3u) - 1.5) * d.w;
}

fn mg_dot16(w: u32, xi: u32) -> f32 {
    return (f32(w & 3u) - 1.5) * mg_x[xi]
         + (f32((w >> 2u) & 3u) - 1.5) * mg_x[xi + 1u]
         + (f32((w >> 4u) & 3u) - 1.5) * mg_x[xi + 2u]
         + (f32((w >> 6u) & 3u) - 1.5) * mg_x[xi + 3u]
         + (f32((w >> 8u) & 3u) - 1.5) * mg_x[xi + 4u]
         + (f32((w >> 10u) & 3u) - 1.5) * mg_x[xi + 5u]
         + (f32((w >> 12u) & 3u) - 1.5) * mg_x[xi + 6u]
         + (f32((w >> 14u) & 3u) - 1.5) * mg_x[xi + 7u]
         + (f32((w >> 16u) & 3u) - 1.5) * mg_x[xi + 8u]
         + (f32((w >> 18u) & 3u) - 1.5) * mg_x[xi + 9u]
         + (f32((w >> 20u) & 3u) - 1.5) * mg_x[xi + 10u]
         + (f32((w >> 22u) & 3u) - 1.5) * mg_x[xi + 11u]
         + (f32((w >> 24u) & 3u) - 1.5) * mg_x[xi + 12u]
         + (f32((w >> 26u) & 3u) - 1.5) * mg_x[xi + 13u]
         + (f32((w >> 28u) & 3u) - 1.5) * mg_x[xi + 14u]
         + (f32((w >> 30u) & 3u) - 1.5) * mg_x[xi + 15u];
}

@compute @workgroup_size(64)
fn moe_gate_up_q2tp(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let slot = wid.y;
    let gpr = mg_p.gpr;
    let rows = mg_p.inter;
    let base16 = mg_sel[slot] * mg_p.mat16;
    let nib16 = base16 + row * gpr * 4u;
    let par16 = base16 + rows * gpr * 4u + row * 2u;
    let cst = (gpr * 5u + 7u) / 8u;
    let cod8 = (base16 + rows * gpr * 4u + rows * 2u) * 2u + row * cst;

    let gl = unpack2x16float(mg_g16(par16) | (mg_g16(par16 + 1u) << 16u));
    let ul = unpack2x16float(mg_u16f(par16) | (mg_u16f(par16 + 1u) << 16u));
    var ag = 0.0;
    var au = 0.0;
    for (var g = lid; g < gpr; g = g + 64u) {
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        var cg = mgp_gu8(cod8 + cb);
        var cu = mgp_uu8(cod8 + cb);
        if (shf > 3u) {
            cg = cg | (mgp_gu8(cod8 + cb + 1u) << 8u);
            cu = cu | (mgp_uu8(cod8 + cb + 1u) << 8u);
        }
        // Rung 0 is the format's exact zero (the ±0.5/±1.5 grid has no
        // zero of its own); live rungs are the ladder shifted down one.
        let cgv = (cg >> shf) & 31u;
        let cuv = (cu >> shf) & 31u;
        let sg = select(exp2(gl.x + f32(max(cgv, 1u) - 1u) * gl.y), 0.0, cgv == 0u);
        let su = select(exp2(ul.x + f32(max(cuv, 1u) - 1u) * ul.y), 0.0, cuv == 0u);
        // Group base in u16 units is a multiple of 4, so the two 32-bit
        // words land on u32 lanes (nib16 >> 1) and (nib16 >> 1) + 1.
        let w32 = (nib16 + g * 4u) >> 1u;
        // One group = 32 activations = 8 vec4s, shared by gate and up.
        let xq = g * 8u;
        let x0 = mg_xv[xq];      let x1 = mg_xv[xq + 1u];
        let x2 = mg_xv[xq + 2u]; let x3 = mg_xv[xq + 3u];
        let x4 = mg_xv[xq + 4u]; let x5 = mg_xv[xq + 5u];
        let x6 = mg_xv[xq + 6u]; let x7 = mg_xv[xq + 7u];
        let dg = mg_dot16v(mg_gw[w32], x0, x1, x2, x3)
               + mg_dot16v(mg_gw[w32 + 1u], x4, x5, x6, x7);
        let du = mg_dot16v(mg_uw[w32], x0, x1, x2, x3)
               + mg_dot16v(mg_uw[w32 + 1u], x4, x5, x6, x7);
        ag = ag + sg * dg;
        au = au + su * du;
    }
    mg_pg[lid] = ag;
    mg_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            mg_pg[lid] = mg_pg[lid] + mg_pg[lid + stride];
            mg_pu[lid] = mg_pu[lid] + mg_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        let g = mg_pg[0];
        var gg = g;
        var uu = mg_pu[0];
        if (mg_p.lim > 0.0) {
            uu = clamp(uu, -mg_p.lim, mg_p.lim);
            gg = min(gg, mg_p.lim);
        }
        mg_act[slot * mg_p.inter + row] = (gg / (1.0 + exp(-gg))) * uu;
    }
}

fn mdp_u8(o: u32) -> u32 { return (md_w[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu; }

@compute @workgroup_size(64)
fn moe_down_q4tp(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let gpr = md_p.gpr;
    let rows = md_p.hidden;
    let cst = (gpr * 5u + 7u) / 8u;
    let total = md_p.slots * gpr;
    var acc = 0.0;
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i % gpr;
        let base16 = md_sel[slot] * md_p.mat16;
        let par16 = base16 + rows * gpr * 8u + row * 2u;
        let cod8 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row * cst;
        let pl = unpack2x16float(md_u16(par16) | (md_u16(par16 + 1u) << 16u));
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        var cv = mdp_u8(cod8 + cb);
        if (shf > 3u) { cv = cv | (mdp_u8(cod8 + cb + 1u) << 8u); }
        let scale = exp2(pl.x + f32((cv >> shf) & 31u) * pl.y);
        let t16 = base16 + (row * gpr + g) * 8u;
        let xq = (slot * gpr + g) * 8u;
        let x0 = md_actv[xq];      let x1 = md_actv[xq + 1u];
        let x2 = md_actv[xq + 2u]; let x3 = md_actv[xq + 3u];
        let x4 = md_actv[xq + 4u]; let x5 = md_actv[xq + 5u];
        let x6 = md_actv[xq + 6u]; let x7 = md_actv[xq + 7u];
        let w0 = md_u16(t16) | (md_u16(t16 + 1u) << 16u);
        let w1 = md_u16(t16 + 2u) | (md_u16(t16 + 3u) << 16u);
        let w2 = md_u16(t16 + 4u) | (md_u16(t16 + 5u) << 16u);
        let w3 = md_u16(t16 + 6u) | (md_u16(t16 + 7u) << 16u);
        let d = md_dot8v(w0, x0, x1) + md_dot8v(w1, x2, x3)
              + md_dot8v(w2, x4, x5) + md_dot8v(w3, x6, x7);
        acc = acc + md_wt[slot] * scale * d;
    }
    md_pt[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { md_pt[lid] = md_pt[lid] + md_pt[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) { md_y[row] = md_pt[0]; }
}

// ── Batched MoE: the whole layer's routing and experts with a TOKEN
// dimension, for the batch (prefill) graph. The per-token encoding of
// this block cost ~7 commands per token per layer — at k=32 over 40
// layers that is ~9000 encoder commands a chunk, and the chunk clocked
// at the same 16 ms/token as the per-position path it was meant to
// beat. These three kernels replace all of it with THREE dispatches per
// layer and zero buffer-to-buffer copies.
//
// The router matvec lives INSIDE the select kernel: one thread = one
// expert row (n_exp <= 256 = workgroup size), reading x straight from
// the batch hidden at its token offset. No logits buffer, no row
// staging. f32 router weights only — the converter leaves the router
// unquantized; anything else falls back to the per-token path.
struct MoeSelBP { n_exp: u32, top_k: u32, norm: u32, pk: u32 };
@group(0) @binding(0) var<storage, read>       sb_lgin: array<f32>;
@group(0) @binding(1) var<storage, read>       sb_x   : array<f32>;
@group(0) @binding(2) var<storage, read_write> sb_sel : array<u32>;
@group(0) @binding(3) var<storage, read_write> sb_w   : array<f32>;
@group(0) @binding(4) var<uniform>             sb_p   : MoeSelBP;
@group(0) @binding(5) var<storage, read>       sb_sgw : array<u32>;
var<workgroup> sb_lg:  array<f32, 256>;
var<workgroup> sb_red: array<f32, 256>;
var<workgroup> sb_ri:  array<u32, 256>;
var<workgroup> sb_sg:  f32;

@compute @workgroup_size(256)
fn moe_select_b(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let n = sb_p.n_exp;
    let sg_kind = sb_p.pk & 0xFFu;
    let hidden = sb_p.pk >> 8u;
    let xb = t * hidden;
    // Logits arrive from the same f32 matvec kernel the parity-proven
    // path used, one slice per token — computing them here with a
    // different summation order shifted near-tied experts and broke
    // token parity with the CPU.
    var v = -3.0e38;
    if (lid < n) { v = sb_lgin[t * n + lid]; }
    // Shared-expert gate on this token's x (f32 weights, same fold as
    // the single-token kernel).
    if (sg_kind == 4u) {
        var d = 0.0;
        var i = lid;
        loop {
            if (i >= hidden) { break; }
            d = d + bitcast<f32>(sb_sgw[i]) * sb_x[xb + i];
            i = i + 256u;
        }
        sb_red[lid] = d;
        workgroupBarrier();
        var st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { sb_red[lid] = sb_red[lid] + sb_red[lid + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        if (lid == 0u) { sb_sg = sb_red[0]; }
        workgroupBarrier();
    }
    sb_lg[lid] = v;
    sb_red[lid] = v;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { sb_red[lid] = max(sb_red[lid], sb_red[lid + stride]); }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let mx = sb_red[0];
    workgroupBarrier();
    sb_red[lid] = select(0.0, exp(v - mx), lid < n);
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { sb_red[lid] = sb_red[lid] + sb_red[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let denom = sb_red[0];
    workgroupBarrier();
    let kk = sb_p.top_k;
    let ob = t * (kk + 1u);
    var wsum = 0.0;
    for (var slot = 0u; slot < kk; slot = slot + 1u) {
        sb_red[lid] = sb_lg[lid];
        sb_ri[lid] = lid;
        workgroupBarrier();
        stride = 128u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) {
                let a = sb_red[lid];
                let b = sb_red[lid + stride];
                let ia = sb_ri[lid];
                let ib = sb_ri[lid + stride];
                if (b > a || (b == a && ib < ia)) {
                    sb_red[lid] = b;
                    sb_ri[lid] = ib;
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) {
            let bi = sb_ri[0];
            sb_sel[ob + slot] = bi;
            sb_w[ob + slot] = exp(sb_red[0] - mx) / denom;
        }
        workgroupBarrier();
        wsum = wsum + exp(sb_red[0] - mx) / denom;
        if (lid == sb_ri[0]) { sb_lg[lid] = -3.0e38; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        if (sb_p.norm != 0u) {
            for (var slot = 0u; slot < kk; slot = slot + 1u) {
                sb_w[ob + slot] = sb_w[ob + slot] / wsum;
            }
        }
        sb_sel[ob + kk] = n;
        sb_w[ob + kk] = 1.0 / (1.0 + exp(-sb_sg));
    }
}

// gate+up+SiLU with a token axis: workgroup (row, slot, token).
struct MoeGuBP { gpr: u32, inter: u32, slots: u32, mat16: u32 , lim: f32, _p0: u32, _p1: u32, _p2: u32 };
// Three scalars, not a vec3: a vec3<u32> aligns to 16 in uniform layout and
// pushes the struct to 48 bytes, while the buffer handed in is 32.
// `lim` is DeepSeek-V4's swiglu_limit: the up projection is clamped both
// ways, the gate only from above. Zero means no clamp, which is every other
// architecture that reaches these kernels.
@group(0) @binding(0) var<storage, read>       gb_gw  : array<u32>;
@group(0) @binding(1) var<storage, read>       gb_uw  : array<u32>;
@group(0) @binding(2) var<storage, read>       gb_x   : array<f32>;
@group(0) @binding(3) var<storage, read>       gb_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> gb_act : array<f32>;
@group(0) @binding(5) var<uniform>             gb_p   : MoeGuBP;
var<workgroup> gb_pg: array<f32, 64>;
var<workgroup> gb_pu: array<f32, 64>;

fn gb_g16(o: u32) -> u32 { return (gb_gw[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn gb_u16(o: u32) -> u32 { return (gb_uw[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn gb_gu8(o: u32) -> u32 { return (gb_gw[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu; }
fn gb_uu8(o: u32) -> u32 { return (gb_uw[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu; }
fn gb_dot8(w: u32, xi: u32) -> f32 {
    return (f32(w & 0xFu) - 8.0) * gb_x[xi]
         + (f32((w >> 4u) & 0xFu) - 8.0) * gb_x[xi + 1u]
         + (f32((w >> 8u) & 0xFu) - 8.0) * gb_x[xi + 2u]
         + (f32((w >> 12u) & 0xFu) - 8.0) * gb_x[xi + 3u]
         + (f32((w >> 16u) & 0xFu) - 8.0) * gb_x[xi + 4u]
         + (f32((w >> 20u) & 0xFu) - 8.0) * gb_x[xi + 5u]
         + (f32((w >> 24u) & 0xFu) - 8.0) * gb_x[xi + 6u]
         + (f32((w >> 28u) & 0xFu) - 8.0) * gb_x[xi + 7u];
}

@compute @workgroup_size(64)
fn moe_gate_up_q4tp_b(@builtin(workgroup_id) wid: vec3<u32>,
                      @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let slot = wid.y;
    let t = wid.z;
    let gpr = gb_p.gpr;
    let rows = gb_p.inter;
    let hidden = gpr * 32u;
    let xoff = t * hidden;
    let base16 = gb_sel[t * gb_p.slots + slot] * gb_p.mat16;
    let nib16 = base16 + row * gpr * 8u;
    let par16 = base16 + rows * gpr * 8u + row * 2u;
    let cst = (gpr * 5u + 7u) / 8u;
    let cod8 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row * cst;
    let gl = unpack2x16float(gb_g16(par16) | (gb_g16(par16 + 1u) << 16u));
    let ul = unpack2x16float(gb_u16(par16) | (gb_u16(par16 + 1u) << 16u));
    var ag = 0.0;
    var au = 0.0;
    for (var g = lid; g < gpr; g = g + 64u) {
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        var cg = gb_gu8(cod8 + cb);
        var cu = gb_uu8(cod8 + cb);
        if (shf > 3u) {
            cg = cg | (gb_gu8(cod8 + cb + 1u) << 8u);
            cu = cu | (gb_uu8(cod8 + cb + 1u) << 8u);
        }
        let sg = exp2(gl.x + f32((cg >> shf) & 31u) * gl.y);
        let su = exp2(ul.x + f32((cu >> shf) & 31u) * ul.y);
        let t16 = nib16 + g * 8u;
        let xb = xoff + g * 32u;
        var dg = 0.0;
        var du = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let wg = gb_g16(t16 + 2u * k) | (gb_g16(t16 + 1u + 2u * k) << 16u);
            let wu = gb_u16(t16 + 2u * k) | (gb_u16(t16 + 1u + 2u * k) << 16u);
            dg = dg + gb_dot8(wg, xb + 8u * k);
            du = du + gb_dot8(wu, xb + 8u * k);
        }
        ag = ag + sg * dg;
        au = au + su * du;
    }
    gb_pg[lid] = ag;
    gb_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            gb_pg[lid] = gb_pg[lid] + gb_pg[lid + stride];
            gb_pu[lid] = gb_pu[lid] + gb_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        let g = gb_pg[0];
        var gg = g;
        var uu = gb_pu[0];
        if (gb_p.lim > 0.0) {
            uu = clamp(uu, -gb_p.lim, gb_p.lim);
            gg = min(gg, gb_p.lim);
        }
        gb_act[(t * gb_p.slots + slot) * gb_p.inter + row] =
            (gg / (1.0 + exp(-gg))) * uu;
    }
}

// Weighted down-projection with a token axis; writes STRAIGHT into the
// batch FFN output at the token's row — no staging row, no copy back.
struct MoeDnBP { gpr: u32, hidden: u32, slots: u32, mat16: u32 };
@group(0) @binding(0) var<storage, read>       db_w   : array<u32>;
@group(0) @binding(1) var<storage, read>       db_act : array<f32>;
@group(0) @binding(2) var<storage, read>       db_sel : array<u32>;
@group(0) @binding(3) var<storage, read>       db_wt  : array<f32>;
@group(0) @binding(4) var<storage, read_write> db_y   : array<f32>;
@group(0) @binding(5) var<uniform>             db_p   : MoeDnBP;
var<workgroup> db_pt: array<f32, 64>;

fn db_u16(o: u32) -> u32 { return (db_w[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn db_u8(o: u32) -> u32 { return (db_w[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu; }
fn db_dot8(w: u32, xi: u32) -> f32 {
    return (f32(w & 0xFu) - 8.0) * db_act[xi]
         + (f32((w >> 4u) & 0xFu) - 8.0) * db_act[xi + 1u]
         + (f32((w >> 8u) & 0xFu) - 8.0) * db_act[xi + 2u]
         + (f32((w >> 12u) & 0xFu) - 8.0) * db_act[xi + 3u]
         + (f32((w >> 16u) & 0xFu) - 8.0) * db_act[xi + 4u]
         + (f32((w >> 20u) & 0xFu) - 8.0) * db_act[xi + 5u]
         + (f32((w >> 24u) & 0xFu) - 8.0) * db_act[xi + 6u]
         + (f32((w >> 28u) & 0xFu) - 8.0) * db_act[xi + 7u];
}

@compute @workgroup_size(64)
fn moe_down_q4tp_b(@builtin(workgroup_id) wid: vec3<u32>,
                   @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let t = wid.y;
    let gpr = db_p.gpr;
    let rows = db_p.hidden;
    let inter = gpr * 32u;
    let cst = (gpr * 5u + 7u) / 8u;
    let sb = t * db_p.slots;
    let ab = t * db_p.slots * inter;
    let total = db_p.slots * gpr;
    var acc = 0.0;
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i % gpr;
        let base16 = db_sel[sb + slot] * db_p.mat16;
        let par16 = base16 + rows * gpr * 8u + row * 2u;
        let cod8 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row * cst;
        let pl = unpack2x16float(db_u16(par16) | (db_u16(par16 + 1u) << 16u));
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        var cv = db_u8(cod8 + cb);
        if (shf > 3u) { cv = cv | (db_u8(cod8 + cb + 1u) << 8u); }
        let scale = exp2(pl.x + f32((cv >> shf) & 31u) * pl.y);
        let t16 = base16 + (row * gpr + g) * 8u;
        let xb = ab + (slot * gpr + g) * 32u;
        var d = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let w = db_u16(t16 + 2u * k) | (db_u16(t16 + 1u + 2u * k) << 16u);
            d = d + db_dot8(w, xb + 8u * k);
        }
        acc = acc + db_wt[sb + slot] * scale * d;
    }
    db_pt[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { db_pt[lid] = db_pt[lid] + db_pt[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) { db_y[t * rows + row] = db_pt[0]; }
}


// ── O(1) Nystrom attention on the graph (spec: nystrom.rs step/far_insert).
// State lives on the device after one upload per seal epoch: ring window,
// sinks, landmarks, and each head's flash-scaled far skeleton. Per o1
// layer per token: THREE dispatches replacing kv_append+attend, and the
// work is O(m + w) instead of O(ctx) — the graph's exact attend was
// 54 -> 37.8 tok/s from 4K to 16K while o1 holds flat by construction.
struct O1P { hpg: u32, m: u32, w: u32, nsrect: u32, d: u32, dv: u32, scale: f32, _p: u32 };

@group(0) @binding(0) var<storage, read_write> of_meta : array<u32>;
@group(0) @binding(1) var<storage, read>       of_rk   : array<f32>;
@group(0) @binding(2) var<storage, read>       of_rv   : array<f32>;
@group(0) @binding(3) var<storage, read>       of_qt   : array<f32>;
@group(0) @binding(4) var<storage, read_write> of_mz   : array<f32>;
@group(0) @binding(5) var<storage, read_write> of_th   : array<f32>;
@group(0) @binding(6) var<uniform>             of_p    : O1P;
var<workgroup> of_part: array<f32, 64>;
var<workgroup> of_rs: f32;
var<workgroup> of_e: f32;

// One workgroup per (group, head, landmark): absorb the evicted window
// slot into this head's far accumulators (nystrom.rs far_insert).
@compute @workgroup_size(64)
fn o1_far(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let hm = of_p.hpg * of_p.m;
    let g = wid.x / hm;
    let rr = wid.x % hm;
    let h = rr / of_p.m;
    let i = rr % of_p.m;
    let len = of_meta[g * 4u];
    if (len < of_p.w) { return; }
    let slot = of_meta[g * 4u + 1u];
    let d = of_p.d;
    let qb = ((g * of_p.hpg + h) * of_p.m + i) * d;
    let kb = (g * of_p.w + slot) * d;
    var acc = 0.0;
    var t = lid;
    loop {
        if (t >= d) { break; }
        acc = acc + of_qt[qb + t] * of_rk[kb + t];
        t = t + 64u;
    }
    of_part[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { of_part[lid] = of_part[lid] + of_part[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let mzb = (g * of_p.hpg + h) * 2u * of_p.m;
    if (lid == 0u) {
        let l = of_part[0] * of_p.scale;
        var mm = of_mz[mzb + i];
        var rs = 1.0;
        if (l > mm) {
            rs = exp(mm - l);
            of_mz[mzb + of_p.m + i] = of_mz[mzb + of_p.m + i] * rs;
            mm = l;
            of_mz[mzb + i] = l;
        }
        let e = exp(l - mm);
        of_mz[mzb + of_p.m + i] = of_mz[mzb + of_p.m + i] + e;
        of_rs = rs;
        of_e = e;
    }
    workgroupBarrier();
    let rs = of_rs;
    let e = of_e;
    let thb = ((g * of_p.hpg + h) * of_p.m + i) * of_p.dv;
    let vb = (g * of_p.w + slot) * of_p.dv;
    var u = lid;
    loop {
        if (u >= of_p.dv) { break; }
        of_th[thb + u] = of_th[thb + u] * rs + e * of_rv[vb + u];
        u = u + 64u;
    }
}

@group(0) @binding(0) var<storage, read_write> op_meta : array<u32>;
@group(0) @binding(1) var<storage, read>       op_k    : array<f32>;
@group(0) @binding(2) var<storage, read>       op_v    : array<f32>;
@group(0) @binding(3) var<storage, read_write> op_rk   : array<f32>;
@group(0) @binding(4) var<storage, read_write> op_rv   : array<f32>;
@group(0) @binding(5) var<uniform>             op_p    : O1P;

// One workgroup per group: push this token's rotated K and V into the
// window ring (after o1_far has read the slot being overwritten).
@compute @workgroup_size(256)
fn o1_push(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x;
    let len = op_meta[g * 4u];
    let head = op_meta[g * 4u + 1u];
    var slot = len;
    if (len == op_p.w) { slot = head; }
    let d = op_p.d;
    var t = lid;
    loop {
        if (t >= d) { break; }
        op_rk[(g * op_p.w + slot) * d + t] = op_k[g * d + t];
        t = t + 256u;
    }
    t = lid;
    loop {
        if (t >= op_p.dv) { break; }
        op_rv[(g * op_p.w + slot) * op_p.dv + t] = op_v[g * op_p.dv + t];
        t = t + 256u;
    }
    workgroupBarrier();
    if (lid == 0u) {
        if (len == op_p.w) {
            op_meta[g * 4u + 1u] = (head + 1u) % op_p.w;
            op_meta[g * 4u + 2u] = op_meta[g * 4u + 2u] + 1u;
        } else {
            op_meta[g * 4u] = len + 1u;
        }
    }
}

@group(0) @binding(0)  var<storage, read>       oa_meta : array<u32>;
@group(0) @binding(1)  var<storage, read>       oa_q    : array<f32>;
@group(0) @binding(2)  var<storage, read>       oa_rk   : array<f32>;
@group(0) @binding(3)  var<storage, read>       oa_rv   : array<f32>;
@group(0) @binding(4)  var<storage, read>       oa_sk   : array<f32>;
@group(0) @binding(5)  var<storage, read>       oa_sv   : array<f32>;
@group(0) @binding(6)  var<storage, read>       oa_kt   : array<f32>;
@group(0) @binding(7)  var<storage, read>       oa_mu   : array<f32>;
@group(0) @binding(8)  var<storage, read>       oa_mz   : array<f32>;
@group(0) @binding(9)  var<storage, read>       oa_th   : array<f32>;
@group(0) @binding(10) var<storage, read_write> oa_out  : array<f32>;
@group(0) @binding(11) var<uniform>             oa_p    : O1P;
var<workgroup> oa_qs:  array<f32, 256>;
var<workgroup> oa_scr: array<f32, 160>;
var<workgroup> oa_f:   array<f32, 32>;
var<workgroup> oa_u:   array<f32, 32>;
var<workgroup> oa_red: array<f32, 256>;
var<workgroup> oa_sc:  array<f32, 4>; // [c_all, far_den, den, have_far]

// One workgroup per (group, head): the whole Nystrom step output.
// Per-score dots run one THREAD per key/landmark (serial over d) — no
// barriers in the hot part, and the same product order as the CPU's
// scalar loop.
@compute @workgroup_size(256)
fn o1_attend(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x / oa_p.hpg;
    let h = wid.x % oa_p.hpg;
    let d = oa_p.d;
    let dv = oa_p.dv;
    let m = oa_p.m;
    let ns = oa_p.nsrect & 0xFFu;
    let rect_fm = (oa_p.nsrect >> 8u) != 0u;
    let len = oa_meta[g * 4u];
    let farl = oa_meta[g * 4u + 2u];
    let n = ns + len;
    let gh = g * oa_p.hpg + h;
    // q into shared
    var t = lid;
    loop {
        if (t >= d) { break; }
        oa_qs[t] = oa_q[gh * d + t];
        t = t + 256u;
    }
    workgroupBarrier();
    // near scores: thread s owns key s
    if (lid < n) {
        var acc = 0.0;
        if (lid < ns) {
            let kb = (g * ns + lid) * d;
            for (var j = 0u; j < d; j = j + 1u) { acc = acc + oa_qs[j] * oa_sk[kb + j]; }
        } else {
            let kb = (g * oa_p.w + (lid - ns)) * d;
            for (var j = 0u; j < d; j = j + 1u) { acc = acc + oa_qs[j] * oa_rk[kb + j]; }
        }
        oa_scr[lid] = acc * oa_p.scale;
    }
    // landmark scores: thread 200+a owns landmark a (disjoint from keys)
    if (lid >= 200u && lid < 200u + m && farl > 0u) {
        let a = lid - 200u;
        var acc = 0.0;
        let ktb = (g * m + a) * d;
        for (var j = 0u; j < d; j = j + 1u) {
            acc = acc + oa_qs[j] * oa_kt[ktb + j];
        }
        oa_f[a] = acc * oa_p.scale;
    }
    workgroupBarrier();
    // c = max near score (single thread — n <= 136, trivial)
    if (lid == 0u) {
        var c = -3.0e38;
        for (var sidx = 0u; sidx < n; sidx = sidx + 1u) { c = max(c, oa_scr[sidx]); }
        var c_all = c;
        var far_den = 0.0;
        var have_far = 0.0;
        if (farl > 0u) {
            var f = -3.0e38;
            for (var a = 0u; a < m; a = a + 1u) { f = max(f, oa_f[a]); }
            for (var a = 0u; a < m; a = a + 1u) { oa_f[a] = exp(oa_f[a] - f); }
            for (var b = 0u; b < m; b = b + 1u) {
                var uacc = 0.0;
                for (var a = 0u; a < m; a = a + 1u) {
                    uacc = uacc + oa_f[a] * oa_mu[(gh * m + a) * m + b];
                }
                if (rect_fm) { uacc = max(uacc, 0.0); }
                oa_u[b] = uacc;
            }
            let mzb = gh * 2u * m;
            for (var b = 0u; b < m; b = b + 1u) {
                c_all = max(c_all, f + oa_mz[mzb + b]);
            }
            for (var b = 0u; b < m; b = b + 1u) {
                let gain = oa_u[b] * exp(f + oa_mz[mzb + b] - c_all);
                oa_u[b] = gain;
                far_den = far_den + gain * oa_mz[mzb + m + b];
            }
            if (far_den >= 0.0) { have_far = 1.0; } else { far_den = 0.0; }
        }
        var den = far_den;
        for (var sidx = 0u; sidx < n; sidx = sidx + 1u) {
            let pv = exp(oa_scr[sidx] - c_all);
            oa_scr[sidx] = pv;
            den = den + pv;
        }
        oa_sc[0] = c_all;
        oa_sc[1] = far_den;
        oa_sc[2] = max(den, 1e-30);
        oa_sc[3] = have_far;
    }
    workgroupBarrier();
    let den = oa_sc[2];
    let have_far = oa_sc[3] > 0.5;
    t = lid;
    loop {
        if (t >= dv) { break; }
        var acc = 0.0;
        if (have_far) {
            for (var b = 0u; b < m; b = b + 1u) {
                acc = acc + oa_u[b] * oa_th[(gh * m + b) * dv + t];
            }
        }
        for (var sidx = 0u; sidx < ns; sidx = sidx + 1u) {
            acc = acc + oa_scr[sidx] * oa_sv[(g * ns + sidx) * dv + t];
        }
        for (var sidx = ns; sidx < n; sidx = sidx + 1u) {
            acc = acc + oa_scr[sidx] * oa_rv[(g * oa_p.w + (sidx - ns)) * dv + t];
        }
        oa_out[gh * dv + t] = acc / den;
        t = t + 256u;
    }
}

// ── DiT attention (imagegen): scores GEMM -> row softmax -> P·V.
// Same 64x64 tile / 4x4 named-scalar register block as the quantized
// GEMMs; the operands are plain f32 here, so the staging is a copy.
struct DitP { m: u32, k: u32, n: u32, scale: f32, s0: u32, causal: u32, _p0: u32, _p1: u32, };
@group(0) @binding(0) var<storage, read> da: array<f32>;
@group(0) @binding(1) var<storage, read> db: array<f32>;
@group(0) @binding(2) var<storage, read_write> dc: array<f32>;
@group(0) @binding(3) var<uniform> dp: DitP;
var<workgroup> dit_at: array<f32, 1024>;
var<workgroup> dit_bt: array<f32, 1024>;

fn dit_gemm(wid: vec3<u32>, lid: vec3<u32>, bt: bool) {
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let tid = lid.y * 16u + lid.x;
    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;
    var k0 = 0u;
    loop {
        if (k0 >= dp.k) { break; }
        for (var q = 0u; q < 4u; q = q + 1u) {
            let r = tid / 4u + q * 64u;
            if (r < 64u) {
                let c4 = (tid % 4u) * 4u;
                for (var e = 0u; e < 4u; e = e + 1u) {
                    let kk = k0 + c4 + e;
                    var va = 0.0;
                    if (m0 + r < dp.m && kk < dp.k) { va = da[(m0 + r) * dp.k + kk]; }
                    dit_at[r * 16u + c4 + e] = va;
                    var vb = 0.0;
                    if (n0 + r < dp.n && kk < dp.k) {
                        if (bt) { vb = db[(n0 + r) * dp.k + kk]; }
                        else { vb = db[kk * dp.n + n0 + r]; }
                    }
                    dit_bt[r * 16u + c4 + e] = vb;
                }
            }
        }
        workgroupBarrier();
        let ab = lid.y * 64u;
        let wb = lid.x * 64u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let x0 = dit_at[ab + k];
            let x1 = dit_at[ab + 16u + k];
            let x2 = dit_at[ab + 32u + k];
            let x3 = dit_at[ab + 48u + k];
            let y0 = dit_bt[wb + k];
            let y1 = dit_bt[wb + 16u + k];
            let y2 = dit_bt[wb + 32u + k];
            let y3 = dit_bt[wb + 48u + k];
            a00 = a00 + x0 * y0; a01 = a01 + x0 * y1;
            a02 = a02 + x0 * y2; a03 = a03 + x0 * y3;
            a10 = a10 + x1 * y0; a11 = a11 + x1 * y1;
            a12 = a12 + x1 * y2; a13 = a13 + x1 * y3;
            a20 = a20 + x2 * y0; a21 = a21 + x2 * y1;
            a22 = a22 + x2 * y2; a23 = a23 + x2 * y3;
            a30 = a30 + x3 * y0; a31 = a31 + x3 * y1;
            a32 = a32 + x3 * y2; a33 = a33 + x3 * y3;
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    let mb = m0 + lid.y * 4u;
    let nb2 = n0 + lid.x * 4u;
    dit_store4(mb, nb2, a00, a01, a02, a03);
    dit_store4(mb + 1u, nb2, a10, a11, a12, a13);
    dit_store4(mb + 2u, nb2, a20, a21, a22, a23);
    dit_store4(mb + 3u, nb2, a30, a31, a32, a33);
}

fn dit_store4(m: u32, n0: u32, v0: f32, v1: f32, v2: f32, v3: f32) {
    if (m >= dp.m) { return; }
    let base = m * dp.n + n0;
    if (n0 < dp.n) { dc[base] = v0 * dp.scale; }
    if (n0 + 1u < dp.n) { dc[base + 1u] = v1 * dp.scale; }
    if (n0 + 2u < dp.n) { dc[base + 2u] = v2 * dp.scale; }
    if (n0 + 3u < dp.n) { dc[base + 3u] = v3 * dp.scale; }
}

@compute @workgroup_size(16, 16)
fn dit_qk(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    dit_gemm(wid, lid, true);
}

@compute @workgroup_size(16, 16)
fn dit_pv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    dit_gemm(wid, lid, false);
}

var<workgroup> dit_red: array<f32, 256>;

// Row softmax over dc, one workgroup per row of dp.n columns.
@compute @workgroup_size(256)
fn dit_softmax(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x * dp.n;
    let t = lid.x;
    // Causal bound: query `wid.x` may see keys 0..=s0+wid.x. Masked
    // entries are zeroed rather than set to -inf so the P·V GEMM that
    // follows reads a clean matrix.
    var lim = dp.n;
    if (dp.causal != 0u) { lim = min(dp.n, dp.s0 + wid.x + 1u); }
    var mx = -3.4e38;
    for (var j = t; j < lim; j = j + 256u) { mx = max(mx, dc[row + j]); }
    dit_red[t] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { dit_red[t] = max(dit_red[t], dit_red[t + s]); }
        workgroupBarrier();
    }
    let m = dit_red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var j = t; j < lim; j = j + 256u) {
        let e = exp(dc[row + j] - m);
        dc[row + j] = e;
        sum = sum + e;
    }
    for (var j = lim + t; j < dp.n; j = j + 256u) { dc[row + j] = 0.0; }
    dit_red[t] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { dit_red[t] = dit_red[t] + dit_red[t + s]; }
        workgroupBarrier();
    }
    let inv = 1.0 / dit_red[0];
    for (var j = t; j < lim; j = j + 256u) { dc[row + j] = dc[row + j] * inv; }
}

// [nh][n][hd] panel -> [n][nh*hd].
@compute @workgroup_size(256)
fn dit_unstack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = dp.m * dp.k * dp.n;   // nh * n * hd
    if (i >= total) { return; }
    let hd = dp.n;
    let n = dp.k;
    let h = i / (n * hd);
    let rest = i % (n * hd);
    let tok = rest / hd;
    let d = rest % hd;
    dc[tok * dp.m * hd + h * hd + d] = da[i];
}


// ── Per-head RMS and the rope tail (DeepSeek-V4) ────────────────────────────
//
// Three places need this and they differ only in flags: the queries take a
// second RMS on each head after wq_b and then a forward rotation; the shared
// KV vector takes a forward rotation alone; and attention's output takes the
// INVERSE rotation, a detail no naming convention would suggest.
//
// The rotation pairs ADJACENT coordinates — the reference builds its complex
// numbers with unflatten(-1, (2)) — and pairing halves instead agrees with it
// exactly at position 0 and nowhere else. That cost a night once.

struct RpP { nh: u32, hd: u32, rd: u32, flags: u32 };   // flags: 1 = rms, 2 = inverse
@group(0) @binding(0) var<storage, read_write> rp_x    : array<f32>;   // nh*hd
@group(0) @binding(1) var<storage, read>       rp_freq : array<f32>;   // rd/2
@group(0) @binding(2) var<uniform>             rp_p    : RpP;
@group(0) @binding(3) var<storage, read>       rp_pos  : array<f32>;   // [position, eps]

var<workgroup> rp_red: array<f32, 256>;

@compute @workgroup_size(256)
fn rope_heads(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    if (h >= rp_p.nh) { return; }
    let hd = rp_p.hd;
    let rd = rp_p.rd;
    let b = h * hd;

    if ((rp_p.flags & 1u) != 0u) {
        var acc = 0.0;
        var i = lid;
        loop {
            if (i >= hd) { break; }
            let v = rp_x[b + i];
            acc = acc + v * v;
            i = i + 256u;
        }
        rp_red[lid] = acc;
        workgroupBarrier();
        var stride = 128u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { rp_red[lid] = rp_red[lid] + rp_red[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        let inv = inverseSqrt(rp_red[0] / f32(hd) + rp_pos[1]);
        workgroupBarrier();
        var j = lid;
        loop {
            if (j >= hd) { break; }
            rp_x[b + j] = rp_x[b + j] * inv;
            j = j + 256u;
        }
        workgroupBarrier();
    }

    // The tail only, adjacent pairs.
    let base = b + hd - rd;
    let pos = rp_pos[0];
    var t = lid;
    loop {
        if (t >= rd / 2u) { break; }
        let th = pos * rp_freq[t];
        var sn = sin(th);
        let cs = cos(th);
        if ((rp_p.flags & 2u) != 0u) { sn = -sn; }
        let a = rp_x[base + 2u * t];
        let c = rp_x[base + 2u * t + 1u];
        rp_x[base + 2u * t] = a * cs - c * sn;
        rp_x[base + 2u * t + 1u] = a * sn + c * cs;
        t = t + 256u;
    }
}

// ── Grouped low-rank output projection, stage A (DeepSeek-V4) ───────────────
//
// wo_a is block-diagonal wearing a dense disguise. It is stored as one
// [groups*lora, per_group] matrix, but row i multiplies ONLY the slice of the
// attention output that group i/lora owns — a matvec whose activation window
// slides with the row. One added term in the index buys the whole operator.
//
// It earns a kernel of its own by size: on the release checkpoint wo_a is 33M
// weights read once per layer per token, the largest single thing still on
// the CPU once the experts are away.
//
// q4tp layout; the per-row math is copied from q4tp_matvec unchanged, so the
// two agree bit for bit wherever they overlap — that is, at lora = rows,
// where the window stops sliding.

@compute @workgroup_size(64)
fn o_lora_a(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let lora = q1p._p0;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    var row = wid.x;
    loop {
        if (row >= rows) { break; }
        if (lid < 32u) {
            let pr = unpack2x16float(q1w[params_w + row]);
            lad_q4tp[lid] = exp2(pr.x + f32(lid) * pr.y);
        }
        workgroupBarrier();
        // A row is exactly one group's width, so the slice offset needs no
        // parameter of its own: per_group = gpr * 32.
        let xoff = (row / lora) * gpr * 32u;
        var acc = 0.0;
        var g = lid;
        loop {
            if (g >= gpr) { break; }
            let bit = g * 5u;
            let cb = codes_b + row * cstride + (bit >> 3u);
            let sh = bit & 7u;
            var cv = q4tp_byte(cb);
            if (sh > 3u) { cv = cv | (q4tp_byte(cb + 1u) << 8u); }
            let scale = lad_q4tp[(cv >> sh) & 31u];
            let base = (row * gpr + g) * 4u;
            let xb = xoff + g * 32u;
            var gsum = 0.0;
            for (var k = 0u; k < 4u; k = k + 1u) {
                gsum = gsum + q4b_dot8(q1w[base + k], xb + 8u * k);
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
        if (lid == 0u) { q1y[row] = partial_q1t[0]; }
        workgroupBarrier();
        row = row + nwg.x;
    }
}

// ── Sparse attention, split in two (DeepSeek-V4) ────────────────────────────
//
// The one-workgroup-per-head version leaves 64 workgroups on a card with
// 150-odd multiprocessors, and measured 0.54 ms a layer — the whole cost of
// the attention block once its encoding was cached away. Scores are cheap and
// genuinely per-head; the weighted sum is nh*hd independent outputs. So:
// scores in one dispatch of nh groups, the sum in another of nh*hd/256.

struct Sa2P { nh: u32, hd: u32, m: u32, scale: f32 };

@group(0) @binding(0) var<storage, read>       s2_q    : array<f32>;   // nh*hd
@group(0) @binding(1) var<storage, read>       s2_kv   : array<f32>;
@group(0) @binding(2) var<storage, read>       s2_idx  : array<u32>;   // m
@group(0) @binding(3) var<storage, read>       s2_sink : array<f32>;   // nh
@group(0) @binding(4) var<storage, read_write> s2_w    : array<f32>;   // nh*m
@group(0) @binding(5) var<uniform>             s2_p    : Sa2P;

var<workgroup> s2_red: array<f32, 256>;
var<workgroup> s2_sc:  array<f32, 1024>;
var<workgroup> s2_max: f32;
var<workgroup> s2_den: f32;

@compute @workgroup_size(256)
fn sa_scores(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    if (h >= s2_p.nh) { return; }
    let hd = s2_p.hd;
    let m = s2_p.m;
    let qb = h * hd;

    var mx = s2_sink[h];
    var t = lid;
    loop {
        if (t >= m) { break; }
        let kb = s2_idx[t] * hd;
        var d = 0.0;
        for (var i = 0u; i < hd; i = i + 1u) { d = d + s2_q[qb + i] * s2_kv[kb + i]; }
        d = d * s2_p.scale;
        s2_sc[t] = d;
        mx = max(mx, d);
        t = t + 256u;
    }
    s2_red[lid] = mx;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { s2_red[lid] = max(s2_red[lid], s2_red[lid + st]); }
        workgroupBarrier();
        st = st >> 1u;
    }
    if (lid == 0u) { s2_max = s2_red[0]; }
    workgroupBarrier();

    // The learned sink enters the denominator and NOT the numerator: that is
    // what lets a head attend to nothing at all.
    var acc = 0.0;
    var u = lid;
    loop {
        if (u >= m) { break; }
        let e = exp(s2_sc[u] - s2_max);
        s2_sc[u] = e;
        acc = acc + e;
        u = u + 256u;
    }
    s2_red[lid] = acc;
    workgroupBarrier();
    st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { s2_red[lid] = s2_red[lid] + s2_red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    if (lid == 0u) { s2_den = s2_red[0] + exp(s2_sink[h] - s2_max); }
    workgroupBarrier();
    let inv = 1.0 / s2_den;
    var v = lid;
    loop {
        if (v >= m) { break; }
        s2_w[h * m + v] = s2_sc[v] * inv;
        v = v + 256u;
    }
}

@group(0) @binding(0) var<storage, read>       sy_w   : array<f32>;   // nh*m
@group(0) @binding(1) var<storage, read>       sy_kv  : array<f32>;
@group(0) @binding(2) var<storage, read>       sy_idx : array<u32>;
@group(0) @binding(3) var<storage, read_write> sy_out : array<f32>;   // nh*hd
@group(0) @binding(4) var<uniform>             sy_p   : Sa2P;

@compute @workgroup_size(256)
fn sa_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let hd = sy_p.hd;
    if (i >= sy_p.nh * hd) { return; }
    let h = i / hd;
    let d = i % hd;
    let m = sy_p.m;
    let wb = h * m;
    var acc = 0.0;
    for (var t = 0u; t < m; t = t + 1u) {
        acc = acc + sy_w[wb + t] * sy_kv[sy_idx[t] * hd + d];
    }
    sy_out[i] = acc;
}

// ── The KV compressor's pooling step (DeepSeek-V4) ──────────────────────────
//
// A softmax over the slot axis taken PER DIMENSION — not per token — then the
// weighted sum. Both compressors end here; they differ only in how the slots
// are gathered, so the gather lives inside the kernel and the graph never has
// to materialise the interleaved copy the CPU builds.
//
// Overlapping (ratio 4 in the release): each token contributes 2*width
// values, the first half belonging to the window that began half a stride
// earlier. Fold time pools 2*ratio slots — the previous window's taking their
// first half, the current window's taking their second. A missing previous
// window votes with -inf, which is also how a whole column of absent slots
// leaves the output at zero instead of dividing by nothing.

struct KpP { slots: u32, width: u32, ratio: u32, flags: u32 };
// flags: 1 = overlapping, 2 = a previous window exists, 4 = add the APE bias

@group(0) @binding(0) var<storage, read>       kp_pkv : array<f32>;
@group(0) @binding(1) var<storage, read>       kp_psc : array<f32>;
@group(0) @binding(2) var<storage, read>       kp_ckv : array<f32>;
@group(0) @binding(3) var<storage, read>       kp_csc : array<f32>;
@group(0) @binding(4) var<storage, read>       kp_ape : array<f32>;
@group(0) @binding(5) var<storage, read_write> kp_out : array<f32>;
@group(0) @binding(6) var<uniform>             kp_p   : KpP;

const KP_NINF: f32 = -3.0e38;

@compute @workgroup_size(256)
fn kv_pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let w = kp_p.width;
    if (d >= w) { return; }
    let slots = kp_p.slots;
    let r = kp_p.ratio;
    let overlap = (kp_p.flags & 1u) != 0u;
    let have_prev = (kp_p.flags & 2u) != 0u;
    let use_ape = (kp_p.flags & 4u) != 0u;

    // Pass one: the maximum, so the exponentials cannot overflow. A column
    // that is entirely absent stays at -inf and the slot is left at zero.
    var mx = KP_NINF;
    for (var t = 0u; t < slots; t = t + 1u) {
        var sc = KP_NINF;
        if (overlap) {
            if (t < r) {
                if (have_prev) { sc = kp_psc[t * 2u * w + d]; }
            } else {
                sc = kp_csc[(t - r) * 2u * w + w + d];
            }
        } else {
            sc = kp_csc[t * w + d];
            if (use_ape) { sc = sc + kp_ape[t * w + d]; }
        }
        mx = max(mx, sc);
    }
    if (mx <= KP_NINF) { kp_out[d] = 0.0; return; }

    var den = 0.0;
    var acc = 0.0;
    for (var t = 0u; t < slots; t = t + 1u) {
        var sc = KP_NINF;
        var kv = 0.0;
        if (overlap) {
            if (t < r) {
                if (have_prev) {
                    sc = kp_psc[t * 2u * w + d];
                    kv = kp_pkv[t * 2u * w + d];
                }
            } else {
                sc = kp_csc[(t - r) * 2u * w + w + d];
                kv = kp_ckv[(t - r) * 2u * w + w + d];
            }
        } else {
            sc = kp_csc[t * w + d];
            if (use_ape) { sc = sc + kp_ape[t * w + d]; }
            kv = kp_ckv[t * w + d];
        }
        if (sc > KP_NINF) {
            let e = exp(sc - mx);
            den = den + e;
            acc = acc + e * kv;
        }
    }
    if (den <= 0.0) { kp_out[d] = 0.0; return; }
    kp_out[d] = acc / den;
}

// ── The sparse indexer: scores, then the top-k (DeepSeek-V4) ────────────────
//
// The relu comes BEFORE the per-head weighting, so a head can vote for a
// position or abstain but never against it. Getting that order wrong produces
// scores that look reasonable and a top-k that is quietly different.

struct IxP { nh: u32, hd: u32, n_pos: u32, limit: u32 };

@group(0) @binding(0) var<storage, read>       ix_q   : array<f32>;   // nh*hd
@group(0) @binding(1) var<storage, read>       ix_kv  : array<f32>;   // n_pos*hd
@group(0) @binding(2) var<storage, read>       ix_w   : array<f32>;   // nh
@group(0) @binding(3) var<storage, read_write> ix_out : array<f32>;   // n_pos
@group(0) @binding(4) var<uniform>             ix_p   : IxP;

var<workgroup> ix_red: array<f32, 256>;

@compute @workgroup_size(256)
fn index_scores(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    if (t >= ix_p.n_pos) { return; }
    if (t >= ix_p.limit) {
        if (lid == 0u) { ix_out[t] = KP_NINF; }
        return;
    }
    let hd = ix_p.hd;
    let kb = t * hd;
    // One lane per head: the relu makes the heads non-additive before their
    // weights, so a head's dot has to be finished by whoever owns it.
    var acc = 0.0;
    var h = lid;
    loop {
        if (h >= ix_p.nh) { break; }
        var dot = 0.0;
        let qb = h * hd;
        for (var i = 0u; i < hd; i = i + 1u) {
            dot = dot + ix_q[qb + i] * ix_kv[kb + i];
        }
        acc = acc + max(dot, 0.0) * ix_w[h];
        h = h + 256u;
    }
    ix_red[lid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { ix_red[lid] = ix_red[lid] + ix_red[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) { ix_out[t] = ix_red[0]; }
}

// Top-k without a sort. The CPU picks by repeated argmax, first maximum wins,
// and returns the winners in index order; the same set falls out of a rank —
// how many positions beat me, counting an equal score as a win only if it
// sits at a lower index — kept when it is below k. Two O(n^2) passes over one
// workgroup, which for the compressed axis is a few thousand comparisons and
// needs neither a scan nor an atomic to stay deterministic.

struct TkP { n: u32, k: u32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       tk_s   : array<f32>;   // n
@group(0) @binding(1) var<storage, read_write> tk_idx : array<u32>;   // k
@group(0) @binding(2) var<storage, read_write> tk_cnt : array<u32>;   // 1
@group(0) @binding(3) var<uniform>             tk_p   : TkP;

var<workgroup> tk_keep: array<u32, 4096>;

@compute @workgroup_size(256)
fn top_k_index(@builtin(local_invocation_index) lid: u32) {
    let n = tk_p.n;
    var i = lid;
    loop {
        if (i >= n) { break; }
        let si = tk_s[i];
        var keep = 0u;
        if (si > KP_NINF) {
            var rank = 0u;
            for (var j = 0u; j < n; j = j + 1u) {
                let sj = tk_s[j];
                if (sj > KP_NINF) {
                    if (sj > si || (sj == si && j < i)) { rank = rank + 1u; }
                }
            }
            if (rank < tk_p.k) { keep = 1u; }
        }
        tk_keep[i] = keep;
        i = i + 256u;
    }
    workgroupBarrier();
    // Position among the kept, by index — counted rather than scanned, which
    // costs one more pass and removes every ordering question.
    var m = lid;
    loop {
        if (m >= n) { break; }
        if (tk_keep[m] == 1u) {
            var before = 0u;
            for (var j = 0u; j < m; j = j + 1u) { before = before + tk_keep[j]; }
            tk_idx[before] = m;
        }
        m = m + 256u;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var total = 0u;
        for (var j = 0u; j < n; j = j + 1u) { total = total + tk_keep[j]; }
        tk_cnt[0] = total;
    }
}

// ── MoE routing: sqrt-softplus, noaux_tc bias, top-k (DeepSeek-V4) ──────────
//
// The bias shifts the CHOICE and never the weight: the weight of a chosen
// expert is its pre-bias score. Swapping those two — an easy thing to do when
// the bias is right there — leaves a model that still speaks and routes
// slightly wrong forever.
//
// Ranking replaces the repeated argmax and gives selection order for free:
// rank i = how many experts beat it, an equal score counting only from a
// lower index, which is precisely what "first maximum wins" means. Ranks of
// the finite entries are dense from zero, so the count is just how many
// slots got filled.

struct RtP { n: u32, top_k: u32, flags: u32, scale: f32 };
// flags: 1 = bias present, 2 = mask present, 4 = indices forced (hash layers),
//        8 = pin the shared expert in slot top_k with weight 1,
//       16 = the packed set is a SUBSET: rt_map turns a global expert id into
//            a slot, or 0xFFFFFFFF when that expert did not fit on the card.
//
// A cold pick is not dropped and not substituted — it is handed back. The
// slot gets weight zero so the device contributes nothing for it, and the
// expert's global id and its real weight go into rt_cold for the host to
// finish. Routing therefore still ranges over every expert, which is the
// whole difference between this and a mask.
//
// With bit 8 the output is the `msel`/`mwt` pair the batched expert kernels
// read: top_k routed slots then the shared one, every slot written. Slots the
// router could not fill (a mask that closes too much) get weight ZERO rather
// than being left short — the kernels downstream take a fixed slot count, and
// a stale index with a live weight is how a token gets an expert nobody chose.

@group(0) @binding(0) var<storage, read>       rt_s      : array<f32>;   // n
@group(0) @binding(1) var<storage, read>       rt_bias   : array<f32>;   // n
@group(0) @binding(2) var<storage, read>       rt_mask   : array<u32>;   // n
@group(0) @binding(3) var<storage, read>       rt_forced : array<u32>;   // top_k
@group(0) @binding(4) var<storage, read_write> rt_idx    : array<u32>;   // top_k
@group(0) @binding(5) var<storage, read_write> rt_w      : array<f32>;   // top_k
@group(0) @binding(6) var<storage, read_write> rt_cnt    : array<u32>;   // 1
@group(0) @binding(7) var<uniform>             rt_p      : RtP;
@group(0) @binding(8) var<storage, read>       rt_map    : array<u32>;   // n
@group(0) @binding(9) var<storage, read_write> rt_cold   : array<u32>;   // 2*top_k

var<workgroup> rt_sc:   array<f32, 1024>;   // sqrt(softplus(score))
var<workgroup> rt_sh:   array<f32, 1024>;   // the same, biased and masked
var<workgroup> rt_used: array<u32, 64>;

@compute @workgroup_size(256)
fn moe_route(@builtin(local_invocation_index) lid: u32) {
    let n = rt_p.n;
    let k = rt_p.top_k;
    let has_bias = (rt_p.flags & 1u) != 0u;
    let has_mask = (rt_p.flags & 2u) != 0u;
    let forced = (rt_p.flags & 4u) != 0u;

    // `shared` is a WGSL reserved word. Naming it that compiled here and
    // failed at pipeline creation, which took the whole context down and made
    // every GPU test pass by skipping.
    let pin_shared = (rt_p.flags & 8u) != 0u;
    let subset = (rt_p.flags & 16u) != 0u;
    if (lid < k) {
        rt_used[lid] = 0u;
        rt_idx[lid] = 0u;
        rt_w[lid] = 0.0;
        rt_cold[2u * lid] = 0xFFFFFFFFu;
        rt_cold[2u * lid + 1u] = 0u;
        // Diagnostic mirror, second half: every winner regardless of where it
        // lives. Nothing reads it but a human, so it cannot change an answer.
        rt_cold[2u * k + 2u * lid] = 0xFFFFFFFFu;
        rt_cold[2u * k + 2u * lid + 1u] = 0u;
    }
    // The shared expert sits LAST in the packing, which is `n` only when
    // every expert was packed. With a subset it is n_pack, carried in the
    // flags' upper bits — writing `n` there pointed the kernel past the end
    // of the buffer at whatever followed.
    let shared_slot = rt_p.flags >> 8u;
    if (pin_shared && lid == 0u) {
        rt_idx[k] = shared_slot;
        rt_w[k] = 1.0;
    }
    // Same reason: the zero-fill is a storage write that the ranking lanes
    // must not race with.
    storageBarrier();
    var i = lid;
    loop {
        if (i >= n) { break; }
        let v = rt_s[i];
        // softplus, guarded past 20 the way the reference's F.softplus is
        var sp = v;
        if (v <= 20.0) { sp = log(1.0 + exp(v)); }
        let sc = sqrt(sp);
        rt_sc[i] = sc;
        var sh = sc;
        if (has_bias) { sh = sh + rt_bias[i]; }
        if (has_mask && rt_mask[i] == 0u) { sh = KP_NINF; }
        rt_sh[i] = sh;
        i = i + 256u;
    }
    workgroupBarrier();

    if (forced) {
        if (lid < k) {
            let e = rt_forced[lid];
            rt_idx[lid] = e;
            var w = 0.0;
            if (e < n) { w = rt_sc[e]; }
            rt_w[lid] = w;
            rt_used[lid] = 1u;
        }
    } else {
        var m = lid;
        loop {
            if (m >= n) { break; }
            let si = rt_sh[m];
            if (si > KP_NINF) {
                var rank = 0u;
                for (var j = 0u; j < n; j = j + 1u) {
                    let sj = rt_sh[j];
                    if (sj > KP_NINF) {
                        if (sj > si || (sj == si && j < m)) { rank = rank + 1u; }
                    }
                }
                if (rank < k) {
                    rt_used[rank] = 1u;
                    rt_cold[2u * k + 2u * rank] = m;
                    rt_cold[2u * k + 2u * rank + 1u] = bitcast<u32>(rt_sh[m]);
                    // What the kernel believes it was handed, in the slots the
                    // winners do not use.
                    if (subset) {
                        let slot = rt_map[m];
                        if (slot == 0xFFFFFFFFu) {
                            // Cold: the device computes nothing for it and the
                            // host is told which expert and with what weight.
                            rt_idx[rank] = 0u;
                            rt_w[rank] = 0.0;
                            rt_cold[2u * rank] = m;
                            rt_cold[2u * rank + 1u] = bitcast<u32>(rt_sc[m]);
                        } else {
                            rt_idx[rank] = slot;
                            rt_w[rank] = rt_sc[m];
                        }
                    } else {
                        rt_idx[rank] = m;
                        rt_w[rank] = rt_sc[m];
                    }
                }
            }
            m = m + 256u;
        }
    }
    // BOTH barriers. The ranking above writes rt_idx/rt_w, which are STORAGE
    // buffers, and the lane that normalises them below reads what every other
    // lane wrote. workgroupBarrier orders workgroup memory only; without the
    // storage barrier those writes need not be visible yet. With 8 experts it
    // happened to work, with 256 it did not — and the failure is a routing
    // weight quietly attached to the wrong expert.
    workgroupBarrier();
    storageBarrier();

    // Normalisation is a handful of terms; one lane keeps the add order fixed.
    if (lid == 0u) {

        var cnt = 0u;
        for (var j = 0u; j < k; j = j + 1u) {
            if (rt_used[j] == 1u) { cnt = cnt + 1u; }
        }
        rt_cnt[0] = cnt;
        // The sum runs over the chosen experts INCLUDING the cold ones — the
        // reference normalises across the whole top-k, and leaving them out
        // would inflate every surviving weight.
        var sum = 0.0;
        for (var j = 0u; j < cnt; j = j + 1u) {
            sum = sum + rt_w[j];
            if (rt_cold[2u * j] != 0xFFFFFFFFu) {
                sum = sum + bitcast<f32>(rt_cold[2u * j + 1u]);
            }
        }
        if (sum > 0.0) {
            let inv = rt_p.scale / sum;
            for (var j = 0u; j < cnt; j = j + 1u) {
                rt_w[j] = rt_w[j] * inv;
                if (rt_cold[2u * j] != 0xFFFFFFFFu) {
                    rt_cold[2u * j + 1u] =
                        bitcast<u32>(bitcast<f32>(rt_cold[2u * j + 1u]) * inv);
                }
            }
        }
        // The shared expert is not part of that normalisation — it rides at
        // weight 1 whatever the router decided.
    }
}

// ── Sparse attention over an index list (DeepSeek-V4) ───────────────────────
//
// Not the sliding-window attention the canonical graph encodes: the keys are
// named by an INDEX LIST — the window's positions followed by whichever
// compressed ones the indexer chose — and one KV vector of a single head's
// width serves all query heads. The learned sink enters the denominator and
// contributes nothing to the numerator, which is what lets a head attend to
// nothing at all.
//
// One workgroup per head. `hd` is 512 in the release, so the accumulator
// fits in workgroup storage and the whole head is one pass.

struct SaP { nh: u32, hd: u32, m: u32, scale: f32 };

@group(0) @binding(0) var<storage, read>       sa_q    : array<f32>;   // nh*hd
@group(0) @binding(1) var<storage, read>       sa_kv   : array<f32>;   // n*hd
@group(0) @binding(2) var<storage, read>       sa_idx  : array<u32>;   // m
@group(0) @binding(3) var<storage, read>       sa_sink : array<f32>;   // nh
@group(0) @binding(4) var<storage, read_write> sa_out  : array<f32>;   // nh*hd
@group(0) @binding(5) var<uniform>             sa_p    : SaP;

var<workgroup> sa_red: array<f32, 256>;
// Scores, then weights, for every attended position. Bounding it here bounds
// the index list: window (128) + index_topk (512) fits with room.
var<workgroup> sa_w: array<f32, 1024>;
var<workgroup> sa_max: f32;
var<workgroup> sa_den: f32;

@compute @workgroup_size(256)
fn sparse_attend(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    if (h >= sa_p.nh) { return; }
    let hd = sa_p.hd;
    let m = sa_p.m;
    let qb = h * hd;

    // 1. scores, kept — recomputing them per output dimension would cost
    //    hd times more, and computing them twice was the first draft's waste.
    var mx = sa_sink[h];
    var t = lid;
    loop {
        if (t >= m) { break; }
        let p = sa_idx[t];
        var d = 0.0;
        for (var k = 0u; k < hd; k = k + 1u) {
            d = d + sa_q[qb + k] * sa_kv[p * hd + k];
        }
        let sc = d * sa_p.scale;
        sa_w[t] = sc;
        mx = max(mx, sc);
        t = t + 256u;
    }
    sa_red[lid] = mx;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { sa_red[lid] = max(sa_red[lid], sa_red[lid + stride]); }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let mval = sa_red[0];
    workgroupBarrier();

    // 2. weights and the denominator, the sink taking its share of the latter
    var den = 0.0;
    t = lid;
    loop {
        if (t >= m) { break; }
        let w = exp(sa_w[t] - mval);
        sa_w[t] = w;
        den = den + w;
        t = t + 256u;
    }
    sa_red[lid] = den;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { sa_red[lid] = sa_red[lid] + sa_red[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) { sa_den = sa_red[0] + exp(sa_sink[h] - mval); }
    workgroupBarrier();
    let inv = 1.0 / sa_den;

    // 3. the weighted sum, parallel over the OUTPUT dimension. Splitting by
    //    position instead makes every thread accumulate into the same
    //    sa_acc[k] — a data race that the first draft called "serialised".
    var k = lid;
    loop {
        if (k >= hd) { break; }
        var acc = 0.0;
        for (var i = 0u; i < m; i = i + 1u) {
            acc = acc + sa_w[i] * sa_kv[sa_idx[i] * hd + k];
        }
        sa_out[qb + k] = acc * inv;
        k = k + 256u;
    }
}

// ── Hyper-connections on the device (DeepSeek-V4) ───────────────────────────
//
// The hidden state is `hc` copies of a `dim` vector, and a block folds them
// to one, runs, then expands back through a Sinkhorn-normalized mixing
// matrix. There is no ordinary residual, so this is not an add — it is the
// join between every pair of blocks, and leaving it on the CPU is what
// forces a round trip per layer.
//
// Sizes are small where it matters: hc is 4, mix_hc is 24, and only the fold
// runs over dim. One workgroup owns the whole thing.

struct HcP { hc: u32, dim: u32, iters: u32, eps: f32 };

@group(0) @binding(0) var<storage, read>       hc_state : array<f32>;   // hc*dim
@group(0) @binding(1) var<storage, read>       hc_mix   : array<f32>;   // mix_hc, raw
@group(0) @binding(2) var<storage, read>       hc_sc    : array<f32>;   // 3
@group(0) @binding(3) var<storage, read>       hc_base  : array<f32>;   // mix_hc
@group(0) @binding(4) var<storage, read_write> hc_fold  : array<f32>;   // dim
@group(0) @binding(5) var<storage, read_write> hc_post  : array<f32>;   // hc
@group(0) @binding(6) var<storage, read_write> hc_comb  : array<f32>;   // hc*hc
@group(0) @binding(7) var<uniform>             hc_p     : HcP;

var<workgroup> hc_red: array<f32, 256>;
var<workgroup> hc_pre_w: array<f32, 8>;
var<workgroup> hc_cmb_w: array<f32, 64>;
var<workgroup> hc_rsq: f32;

@compute @workgroup_size(256)
fn hc_pre_fold(@builtin(local_invocation_index) lid: u32) {
    let hc = hc_p.hc;
    let dim = hc_p.dim;
    let n = hc * dim;

    // rsqrt(mean(state^2) + eps) — the reference scales the mixes by this,
    // and it is a mean over ALL copies, not per copy.
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= n) { break; }
        let v = hc_state[i];
        acc = acc + v * v;
        i = i + 256u;
    }
    hc_red[lid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { hc_red[lid] = hc_red[lid] + hc_red[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        hc_rsq = inverseSqrt(hc_red[0] / f32(n) + hc_p.eps);
    }
    workgroupBarrier();
    let rsq = hc_rsq;

    // pre / post / comb. Thread 0 does it: hc is 4, and the Sinkhorn is a
    // sequential fixed point over a 4x4 — parallelising it would cost more
    // in barriers than it saves.
    if (lid == 0u) {
        for (var j = 0u; j < hc; j = j + 1u) {
            let m = hc_mix[j] * rsq * hc_sc[0] + hc_base[j];
            hc_pre_w[j] = 1.0 / (1.0 + exp(-m)) + hc_p.eps;
            let m2 = hc_mix[hc + j] * rsq * hc_sc[1] + hc_base[hc + j];
            hc_post[j] = 2.0 / (1.0 + exp(-m2));
        }
        // row softmax, then the alternating normalisation
        for (var j = 0u; j < hc; j = j + 1u) {
            var mx = -1e30;
            for (var k = 0u; k < hc; k = k + 1u) {
                let v = hc_mix[2u * hc + j * hc + k] * rsq * hc_sc[2]
                      + hc_base[2u * hc + j * hc + k];
                hc_cmb_w[j * hc + k] = v;
                mx = max(mx, v);
            }
            var sum = 0.0;
            for (var k = 0u; k < hc; k = k + 1u) {
                let e = exp(hc_cmb_w[j * hc + k] - mx);
                hc_cmb_w[j * hc + k] = e;
                sum = sum + e;
            }
            for (var k = 0u; k < hc; k = k + 1u) {
                hc_cmb_w[j * hc + k] = hc_cmb_w[j * hc + k] / sum + hc_p.eps;
            }
        }
        for (var k = 0u; k < hc; k = k + 1u) {          // first column pass
            var sum = 0.0;
            for (var j = 0u; j < hc; j = j + 1u) { sum = sum + hc_cmb_w[j * hc + k]; }
            for (var j = 0u; j < hc; j = j + 1u) {
                hc_cmb_w[j * hc + k] = hc_cmb_w[j * hc + k] / (sum + hc_p.eps);
            }
        }
        for (var it = 1u; it < hc_p.iters; it = it + 1u) {
            for (var j = 0u; j < hc; j = j + 1u) {
                var sum = 0.0;
                for (var k = 0u; k < hc; k = k + 1u) { sum = sum + hc_cmb_w[j * hc + k]; }
                for (var k = 0u; k < hc; k = k + 1u) {
                    hc_cmb_w[j * hc + k] = hc_cmb_w[j * hc + k] / (sum + hc_p.eps);
                }
            }
            for (var k = 0u; k < hc; k = k + 1u) {
                var sum = 0.0;
                for (var j = 0u; j < hc; j = j + 1u) { sum = sum + hc_cmb_w[j * hc + k]; }
                for (var j = 0u; j < hc; j = j + 1u) {
                    hc_cmb_w[j * hc + k] = hc_cmb_w[j * hc + k] / (sum + hc_p.eps);
                }
            }
        }
        for (var j = 0u; j < hc * hc; j = j + 1u) { hc_comb[j] = hc_cmb_w[j]; }
    }
    workgroupBarrier();

    // fold: y[d] = sum_j pre[j] * state[j*dim + d]
    var d = lid;
    loop {
        if (d >= dim) { break; }
        var y = 0.0;
        for (var j = 0u; j < hc; j = j + 1u) {
            y = y + hc_pre_w[j] * hc_state[j * dim + d];
        }
        hc_fold[d] = y;
        d = d + 256u;
    }
}

// expand: state[j*dim+d] = post[j]*x[d] + sum_k comb[k*hc+j]*residual[k*dim+d]
// Summing over the FIRST index of comb, as the reference does — reading it
// the other way transposes the mixing and is not detectable by eye.
@group(0) @binding(0) var<storage, read>       he_x    : array<f32>;   // dim
@group(0) @binding(1) var<storage, read>       he_res  : array<f32>;   // hc*dim
@group(0) @binding(2) var<storage, read>       he_post : array<f32>;   // hc
@group(0) @binding(3) var<storage, read>       he_comb : array<f32>;   // hc*hc
@group(0) @binding(4) var<storage, read_write> he_out  : array<f32>;   // hc*dim
@group(0) @binding(5) var<uniform>             he_p    : HcP;
@compute @workgroup_size(256)
fn hc_post_expand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hc = he_p.hc;
    let dim = he_p.dim;
    let i = gid.x;
    if (i >= hc * dim) { return; }
    let j = i / dim;
    let d = i % dim;
    var y = he_post[j] * he_x[d];
    for (var k = 0u; k < hc; k = k + 1u) {
        y = y + he_comb[k * hc + j] * he_res[k * dim + d];
    }
    he_out[i] = y;
}

"#;

// Split-K decode attention (its own module: the main module's at_* binding
// slots are taken, and WGSL forbids two resource vars on one binding).
// `gqa_attend_part` runs the flash-decoding loop over ONE ck-position chunk
// per workgroup — grid (nh, nchunks) instead of nh, which left a discrete GPU
// at 16 resident workgroups and latency-bound at depth — and stores each
// chunk's unnormalized accumulator plus its (m, l) softmax frame.
// `gqa_attend_merge` (grid nh) rescales the chunk frames into the global max
// and normalizes. Same math as `gqa_attend` up to one extra merge rounding.
const ATTEND_SPLIT_SRC: &str = r#"
struct ApP { nh: u32, hpk: u32, hd: u32, cap: u32, n: u32, ck: u32, nc: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       ap_q  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       ap_k  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       ap_v  : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ap_acc: array<f32>;
@group(0) @binding(4) var<storage, read_write> ap_ml : array<vec2<f32>>;
@group(0) @binding(5) var<uniform>             ap_p  : ApP;
@group(0) @binding(6) var<storage, read_write> ap_o  : array<f32>;
var<workgroup> app_acc: array<f32, 8224>;
var<workgroup> app_m: array<f32, 32>;
var<workgroup> app_l: array<f32, 32>;
@compute @workgroup_size(32)
fn gqa_attend_part(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let ch = wid.y;
    let lane = lid.x;
    if (h >= ap_p.nh) { return; }
    let hd = ap_p.hd;
    let hd4 = hd / 4u;
    let p0 = ch * ap_p.ck;
    let pend = min(ap_p.n, p0 + ap_p.ck);
    let kbase = (h / ap_p.hpk) * ap_p.cap * hd4;
    let qbase = h * hd4;
    let scale = 1.0 / sqrt(f32(hd));
    let base = lane * 257u;
    for (var d = 0u; d < hd; d = d + 1u) { app_acc[base + d] = 0.0; }
    var m = -1e30;
    var l = 0.0;
    var p = p0 + lane;
    loop {
        if (p >= pend) { break; }
        let krow = kbase + p * hd4;
        var dot4 = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d = d + 1u) { dot4 = dot4 + ap_q[qbase + d] * ap_k[krow + d]; }
        let dot = (dot4.x + dot4.y + dot4.z + dot4.w) * scale;
        let mp = max(m, dot);
        let f = exp(m - mp);
        let w = exp(dot - mp);
        l = l * f + w;
        for (var d = 0u; d < hd4; d = d + 1u) {
            let vv = ap_v[krow + d] * w;
            let a = base + d * 4u;
            app_acc[a]      = app_acc[a]      * f + vv.x;
            app_acc[a + 1u] = app_acc[a + 1u] * f + vv.y;
            app_acc[a + 2u] = app_acc[a + 2u] * f + vv.z;
            app_acc[a + 3u] = app_acc[a + 3u] * f + vv.w;
        }
        m = mp;
        p = p + 32u;
    }
    app_m[lane] = m;
    app_l[lane] = l;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            let o = lane + stride;
            let m1 = app_m[lane];
            let m2 = app_m[o];
            let mm = max(m1, m2);
            let f1 = exp(m1 - mm);
            let f2 = exp(m2 - mm);
            app_l[lane] = app_l[lane] * f1 + app_l[o] * f2;
            let bo = o * 257u;
            for (var d = 0u; d < hd; d = d + 1u) {
                app_acc[base + d] = app_acc[base + d] * f1 + app_acc[bo + d] * f2;
            }
            app_m[lane] = mm;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let idx = h * ap_p.nc + ch;
    for (var d = lane; d < hd; d = d + 32u) {
        ap_acc[idx * hd + d] = app_acc[d];
    }
    if (lane == 0u) {
        ap_ml[idx] = vec2<f32>(app_m[0], app_l[0]);
    }
}
// hd <= 128 twin of gqa_attend_part at stride 129 (16.5 KB workgroup
// memory — fits the 32 KB mobile/Metal limit; see gqa_attend_s).
var<workgroup> app_acc_s: array<f32, 4128>;
@compute @workgroup_size(32)
fn gqa_attend_part_s(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let ch = wid.y;
    let lane = lid.x;
    if (h >= ap_p.nh) { return; }
    let hd = ap_p.hd;
    let hd4 = hd / 4u;
    let p0 = ch * ap_p.ck;
    let pend = min(ap_p.n, p0 + ap_p.ck);
    let kbase = (h / ap_p.hpk) * ap_p.cap * hd4;
    let qbase = h * hd4;
    let scale = 1.0 / sqrt(f32(hd));
    let base = lane * 129u;
    for (var d = 0u; d < hd; d = d + 1u) { app_acc_s[base + d] = 0.0; }
    var m = -1e30;
    var l = 0.0;
    var p = p0 + lane;
    loop {
        if (p >= pend) { break; }
        let krow = kbase + p * hd4;
        var dot4 = vec4<f32>(0.0);
        for (var d = 0u; d < hd4; d = d + 1u) { dot4 = dot4 + ap_q[qbase + d] * ap_k[krow + d]; }
        let dot = (dot4.x + dot4.y + dot4.z + dot4.w) * scale;
        let mp = max(m, dot);
        let f = exp(m - mp);
        let w = exp(dot - mp);
        l = l * f + w;
        for (var d = 0u; d < hd4; d = d + 1u) {
            let vv = ap_v[krow + d] * w;
            let a = base + d * 4u;
            app_acc_s[a]      = app_acc_s[a]      * f + vv.x;
            app_acc_s[a + 1u] = app_acc_s[a + 1u] * f + vv.y;
            app_acc_s[a + 2u] = app_acc_s[a + 2u] * f + vv.z;
            app_acc_s[a + 3u] = app_acc_s[a + 3u] * f + vv.w;
        }
        m = mp;
        p = p + 32u;
    }
    app_m[lane] = m;
    app_l[lane] = l;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            let o = lane + stride;
            let m1 = app_m[lane];
            let m2 = app_m[o];
            let mm = max(m1, m2);
            let f1 = exp(m1 - mm);
            let f2 = exp(m2 - mm);
            app_l[lane] = app_l[lane] * f1 + app_l[o] * f2;
            let bo = o * 129u;
            for (var d = 0u; d < hd; d = d + 1u) {
                app_acc_s[base + d] = app_acc_s[base + d] * f1 + app_acc_s[bo + d] * f2;
            }
            app_m[lane] = mm;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let idx = h * ap_p.nc + ch;
    for (var d = lane; d < hd; d = d + 32u) {
        ap_acc[idx * hd + d] = app_acc_s[d];
    }
    if (lane == 0u) {
        ap_ml[idx] = vec2<f32>(app_m[0], app_l[0]);
    }
}
@compute @workgroup_size(32)
fn gqa_attend_merge(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let lane = lid.x;
    if (h >= ap_p.nh) { return; }
    let hd = ap_p.hd;
    let nc = (ap_p.n + ap_p.ck - 1u) / ap_p.ck;
    var mg = -1e30;
    for (var ci = 0u; ci < nc; ci = ci + 1u) { mg = max(mg, ap_ml[h * ap_p.nc + ci].x); }
    var lg = 0.0;
    for (var ci = 0u; ci < nc; ci = ci + 1u) {
        let ml = ap_ml[h * ap_p.nc + ci];
        lg = lg + ml.y * exp(ml.x - mg);
    }
    let invl = select(0.0, 1.0 / lg, lg > 0.0);
    for (var d = lane; d < hd; d = d + 32u) {
        var a = 0.0;
        for (var ci = 0u; ci < nc; ci = ci + 1u) {
            let idx = h * ap_p.nc + ci;
            a = a + ap_acc[idx * hd + d] * exp(ap_ml[idx].x - mg);
        }
        ap_o[h * hd + d] = a * invl;
    }
}
"#;

/// Positions per split-K attend chunk; the split path engages past
/// `ATTEND_SPLIT_MIN` cached positions (below it the single-workgroup
/// kernel's one dispatch wins).
const ATTEND_CK: usize = 128;
const ATTEND_SPLIT_MIN: usize = 256;

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
    q4t_mv: wgpu::ComputePipeline,
    /// Hyper-connection fold (with the Sinkhorn) and expand — the join
    /// between blocks in DeepSeek-V4, where an ordinary model has a residual.
    /// Attention over an index list with a learned sink — DeepSeek-V4's,
    /// not the canonical sliding window.
    /// Per-head RMS and the rope tail, forward or inverse.
    rope_heads: wgpu::ComputePipeline,
    o_lora_a: wgpu::ComputePipeline,
    kv_pool: wgpu::ComputePipeline,
    index_scores: wgpu::ComputePipeline,
    top_k_index: wgpu::ComputePipeline,
    moe_route: wgpu::ComputePipeline,
    sparse_attend: wgpu::ComputePipeline,
    sa_scores: wgpu::ComputePipeline,
    sa_apply: wgpu::ComputePipeline,
    hc_pre_fold: wgpu::ComputePipeline,
    hc_post_expand: wgpu::ComputePipeline,
    q4tp_mv: wgpu::ComputePipeline,
    /// Tall-matrix q4tp matvec (4 rows/workgroup, vec4 nibble loads); the
    /// per-row math is byte-identical to `q4tp_mv`. `CMF_MV4=0` reverts.
    q4tp_mv4: wgpu::ComputePipeline,
    use_mv4: bool,
    q4tp_mv16: wgpu::ComputePipeline,
    q4t_mv8: wgpu::ComputePipeline,
    q4b_mv8: wgpu::ComputePipeline,
    q4tp_mm: wgpu::ComputePipeline,
    argmax_part: wgpu::ComputePipeline,
    gdn_step_par: wgpu::ComputePipeline,
    gdn_step_norm: wgpu::ComputePipeline,
    gdn_par: bool,
    ts_query: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)>,
    ts_period: f32,
    argmax_final: wgpu::ComputePipeline,
    embed_gather_q4tp: wgpu::ComputePipeline,
    silu_down: wgpu::ComputePipeline,
    q1t_mm: wgpu::ComputePipeline,
    q4t_mm: wgpu::ComputePipeline,
    dit_qk: wgpu::ComputePipeline,
    dit_pv: wgpu::ComputePipeline,
    dit_softmax: wgpu::ComputePipeline,
    dit_unstack: wgpu::ComputePipeline,
    ffn_silu: wgpu::ComputePipeline,
    q1t_ovmm: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    add_rmsnorm: wgpu::ComputePipeline,
    rmsnorm_b: wgpu::ComputePipeline,
    add_rmsnorm_b: wgpu::ComputePipeline,
    attn_rope: wgpu::ComputePipeline,
    kv_append: wgpu::ComputePipeline,
    gqa_attend: wgpu::ComputePipeline,
    gqa_attend_s: wgpu::ComputePipeline,
    attend_part: wgpu::ComputePipeline,
    attend_part_s: wgpu::ComputePipeline,
    attend_merge: wgpu::ComputePipeline,
    /// Max head_dim the attend kernels can serve on this device: 256
    /// when 33 KB of workgroup storage fits (desktop), 128 on 32 KB
    /// devices (Adreno/Mali/wgpu-Metal) where only the stride-129
    /// kernels exist.
    hd_cap: usize,
    /// 32 KB+ of workgroup storage: the split-K attend parts need it.
    big_attend: bool,
    gdn_step: wgpu::ComputePipeline,
    gdn_conv: wgpu::ComputePipeline,
    f32_matvec: wgpu::ComputePipeline,
    f32_matvec_b: wgpu::ComputePipeline,
    layout_f32b: wgpu::BindGroupLayout,
    o1_far: wgpu::ComputePipeline,
    o1_push: wgpu::ComputePipeline,
    o1_attend: wgpu::ComputePipeline,
    layout_o1_far: wgpu::BindGroupLayout,
    layout_o1_push: wgpu::BindGroupLayout,
    layout_o1_attend: wgpu::BindGroupLayout,
    /// Device o1 state per (kv_id, layer); re-uploaded when the seal
    /// epoch changes (each generate seals fresh CPU state).
    o1m: Mutex<HashMap<(u64, usize), O1Dev>>,
    moe_select: wgpu::ComputePipeline,
    /// Two independent projections of one input in a single dispatch.
    matvec_pair: wgpu::ComputePipeline,
    layout_mv2: wgpu::BindGroupLayout,
    moe_gate_up: wgpu::ComputePipeline,
    moe_down: wgpu::ComputePipeline,
    /// q4tp twins — same bindings, ladder-plane scale decode.
    moe_gate_up_q4tp: wgpu::ComputePipeline,
    moe_down_q4tp: wgpu::ComputePipeline,
    /// Batched (token-axis) MoE for the batch graph; select_b carries the
    /// f32 router matvec inside itself.
    moe_select_b: wgpu::ComputePipeline,
    moe_gate_up_q4tp_b: wgpu::ComputePipeline,
    moe_down_q4tp_b: wgpu::ComputePipeline,
    layout_moe_sel_b: wgpu::BindGroupLayout,
    layout_moe_gu_b: wgpu::BindGroupLayout,
    layout_moe_dn_b: wgpu::BindGroupLayout,
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
    layout_attend_s: wgpu::BindGroupLayout,
    layout_attend_part: wgpu::BindGroupLayout,
    layout_attend_part_s: wgpu::BindGroupLayout,
    layout_attend_merge: wgpu::BindGroupLayout,
    layout_gdn: wgpu::BindGroupLayout,
    layout_gdn_conv: wgpu::BindGroupLayout,
    layout_f32: wgpu::BindGroupLayout,
    layout_silu_down: wgpu::BindGroupLayout,
    layout_moe_sel: wgpu::BindGroupLayout,
    layout_moe_gu: wgpu::BindGroupLayout,
    layout_moe_dn: wgpu::BindGroupLayout,
    /// wgpu treats an auto-derived layout as exclusive to the pipeline it
    /// came from, so the q4tp twins need their own even though the
    /// binding lists are identical.
    layout_moe_gu_q4tp: wgpu::BindGroupLayout,
    moe_gate_up_q2tp: wgpu::ComputePipeline,
    layout_moe_gu_q2tp: wgpu::BindGroupLayout,
    moe_gate_up_q2tp_f: wgpu::ComputePipeline,
    moe_down_q4tp_f: wgpu::ComputePipeline,
    gqa_attend_dec: wgpu::ComputePipeline,
    moe_select_sg: Option<wgpu::ComputePipeline>,
    gdn_step_par2: wgpu::ComputePipeline,
    gdn_step_norm2: wgpu::ComputePipeline,
    gdn_inline: bool,
    attend_dec: bool,
    foldsel: bool,
    layout_moe_dn_q4tp: wgpu::BindGroupLayout,
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
    ///
    /// Residency is DEMAND-DRIVEN and evicting: nothing is preloaded, the
    /// set grows as the model routes, and when the budget is full the
    /// least-valuable tensor makes room. Without eviction the first
    /// tensors to arrive owned the device for the process's life — a
    /// model that switched from prose to code kept the prose experts and
    /// ran the code ones on the CPU forever. The two working sets overlap
    /// at a Jaccard of 0.095, measured, so that is not a corner case.
    weight_bufs: Mutex<HashMap<(usize, usize), Resident>>,
    /// DeepSeek-V4's attention cache, per (kv id, layer), living on the card
    /// between tokens. Re-uploading it each token costs more than the frame
    /// saves: at 4K context it is megabytes per layer.
    dsv4_kv: Mutex<HashMap<(u64, usize), (wgpu::Buffer, usize)>>,
    /// The frame's working buffers, keyed by role and length. Their sizes do
    /// not change from token to token, and allocating ten of them per layer
    /// per token — 430 allocations a token on the release — is most of what a
    /// submission costs. Created once, written thereafter.
    dsv4_scratch: Mutex<HashMap<(u8, usize), wgpu::Buffer>>,
    /// (epoch, bind groups) for the dsv4 frames — see `cached_bind`.
    dsv4_binds: Mutex<(u64, HashMap<(u8, u64, usize), wgpu::BindGroup>)>,
    /// Access clock for the aging above — one tick per weight lookup.
    res_clock: std::sync::atomic::AtomicU64,
    /// row_scale buffer per (idx, row0) — small, cached.
    rs_bufs: Mutex<HashMap<(usize, usize), wgpu::Buffer>>,
    /// Device K/V cache mirror per (kv_id, layer) for the token graph:
    /// [nkv, cap, hd] each, persists across decode tokens. `synced` counts
    /// the positions already resident (prefill sync + graph appends).
    attn_kv: Mutex<HashMap<(u64, usize), KvMirror>>,
    /// GDN recurrent state per (kv_id, layer): (conv ring, S), persists across
    /// decode tokens (created zeroed on first touch).
    gdn_state: Mutex<HashMap<(u64, usize), (wgpu::Buffer, wgpu::Buffer)>>,
    /// Per-layer concatenated MoE expert weights (gate_all, up_all, down_all)
    /// keyed by (file base ptr, first gate idx) — every routed expert plus the
    /// shared one as the trailing block, uploaded once, addressed by expert id
    /// inside the kernels. Counted against `resident` like any weight.
    moe_expw: Mutex<HashMap<(usize, usize), (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)>>,
    /// Immutable [rows,cols,…] uniforms cached by content — the ~800 matvec
    /// param buffers per token are token-invariant, so uploading them once
    /// keeps them off the per-token encode critical path.
    uniforms: Mutex<HashMap<[u32; 4], wgpu::Buffer>>,
    uniforms8: Mutex<HashMap<[u32; 8], wgpu::Buffer>>,
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
    /// Fused-FFN intermediates (gate / up panels).
    g: Option<(wgpu::Buffer, u64)>,
    u: Option<(wgpu::Buffer, u64)>,
    /// DiT attention: Q/K/V uploads, scores, panel, output, staging.
    dq: Option<(wgpu::Buffer, u64)>,
    dk: Option<(wgpu::Buffer, u64)>,
    dv: Option<(wgpu::Buffer, u64)>,
    dsc: Option<(wgpu::Buffer, u64)>,
    dpan: Option<(wgpu::Buffer, u64)>,
    dout: Option<(wgpu::Buffer, u64)>,
    dstage: Option<(wgpu::Buffer, u64)>,
    dpar: Option<wgpu::Buffer>,
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
    // Split-K attend partials: [nh·nchunks·hd] accumulators + [nh·nchunks] (m,l)
    apacc: Option<(wgpu::Buffer, u64)>,
    apml: Option<(wgpu::Buffer, u64)>,
    // MoE routing intermediates: router logits, shared-gate logit, selected
    // expert ids + weights, per-slot activations
    m_logit: Option<(wgpu::Buffer, u64)>,
    m_slog: Option<(wgpu::Buffer, u64)>,
    m_sel: Option<(wgpu::Buffer, u64)>,
    m_wt: Option<(wgpu::Buffer, u64)>,
    m_act: Option<(wgpu::Buffer, u64)>,
    // Logits output + readback staging
    logits: Option<(wgpu::Buffer, u64)>,
    stage: Option<(wgpu::Buffer, u64)>,
    // Position-dependent uniforms (fixed size, write_buffer each token)
    kv_u: Option<wgpu::Buffer>,   // 16 bytes: [nkv, hd, cap, position]
    at_u: Option<wgpu::Buffer>,   // 32 bytes: [nh, nh/nkv, hd, cap, pos+1, 0, 0, 0]
    rope_u: Option<wgpu::Buffer>, // 32 bytes: [nh, nkv, hd, rd, pos, flags, eps, 0]
    // Multi-step slots: one uniform PER STEP with a stable identity, so the
    // attention bind groups survive across chunks (write_buffer runs at
    // submit — a single shared uniform would collapse every step to the
    // last position written).
    kv_us: Vec<wgpu::Buffer>,
    at_us: Vec<wgpu::Buffer>,
    rope_us: Vec<wgpu::Buffer>,
    ids: Option<(wgpu::Buffer, u64)>,
    ids_stage: Option<(wgpu::Buffer, u64)>,
    am_pv: Option<(wgpu::Buffer, u64)>,
    am_pi: Option<(wgpu::Buffer, u64)>,
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
    fn ensure_uniform(
        dev: &wgpu::Device,
        slot: &mut Option<wgpu::Buffer>,
        size: u64,
    ) -> wgpu::Buffer {
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
        _ => {
            crate::pipeline::GLOBAL_USE_GPU.load(std::sync::atomic::Ordering::Relaxed)
                && !cfg!(target_os = "macos")
        }
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
                // Tests install no subscriber, so a tracing-only report makes
                // an init failure look exactly like "no GPU here".
                tracing::warn!("wgpu init failed — CPU fallback: {e}");
                if std::env::var("CMF_GPU_DEBUG").is_ok()
                    || std::env::var("CMF_DSV4_FRAME_DEBUG").is_ok()
                {
                    eprintln!("wgpu init не удался — откат на CPU: {e}");
                }
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
    // 33 152 B = the stride-257 attend kernels' workgroup footprint.
    // Adreno/Mali/wgpu-Metal report 32 768 — there only the stride-129
    // (hd <= 128) kernels are created.
    let big_attend = limits.max_compute_workgroup_storage_size >= 33_152;
    // GPU timestamps (CMF_GPU_TS=1): ask for the query features when the
    // adapter has them — the frame profiler below is the only consumer.
    let ts_features = wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    let want_ts = adapter.features().contains(ts_features);
    let want_sg = adapter.features().contains(wgpu::Features::SUBGROUP);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cortiq-wgpu"),
        required_limits: limits,
        required_features: if want_ts {
            ts_features
        } else {
            wgpu::Features::empty()
        } | if want_sg {
            wgpu::Features::SUBGROUP
        } else {
            wgpu::Features::empty()
        },
        ..Default::default()
    }))
    .map_err(|e| format!("request_device: {e}"))?;
    // Every shader-module and pipeline validation error below must fail
    // init: an invalid pipeline silently turns its dispatches into
    // no-ops and the graph decodes garbage (seen on phones before this
    // scope existed). Err here = clean CPU fallback.
    let vscope = device.push_error_scope(wgpu::ErrorFilter::Validation);

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
        if vram_budget == u64::MAX {
            "unlimited".to_string()
        } else {
            format!("{} MB", vram_budget / 1024 / 1024)
        },
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
    let q4t_mv = pipe("q4t_matvec");
    let rope_heads = pipe("rope_heads");
    let o_lora_a = pipe("o_lora_a");
    let kv_pool = pipe("kv_pool");
    let index_scores = pipe("index_scores");
    let top_k_index = pipe("top_k_index");
    let moe_route = pipe("moe_route");
    let sparse_attend = pipe("sparse_attend");
    let sa_scores = pipe("sa_scores");
    let sa_apply = pipe("sa_apply");
    let hc_pre_fold = pipe("hc_pre_fold");
    let hc_post_expand = pipe("hc_post_expand");
    let q4tp_mv = pipe("q4tp_matvec");
    let q4tp_mv4 = pipe("q4tp_matvec4");
    let use_mv4 = std::env::var("CMF_MV4").map(|v| v != "0").unwrap_or(true);
    let q4tp_mv16 = pipe("q4tp_matvec16");
    let q4t_mv8 = pipe("q4t_matvec8");
    let q4b_mv8 = pipe("q4b_matvec8");
    let q4tp_mm = pipe("q4tp_mul_mm");
    let argmax_part = pipe("argmax_part");
    let gdn_step_par = pipe("gdn_step_par");
    let gdn_step_par2 = pipe("gdn_step_par2");
    let gdn_step_norm2 = pipe("gdn_step_norm2");
    // Measured -1 tok/s on RTX PRO 6000: every dv-workgroup of a head
    // recomputes the conv reads, 128-fold traffic amplification against
    // one saved hop. Kept for narrow-dv models; CMF_GDN_INLINE=1 enables.
    let gdn_inline = std::env::var("CMF_GDN_INLINE").as_deref() == Ok("1");
    let gdn_step_norm = pipe("gdn_step_norm");
    let gdn_par = std::env::var("CMF_GDN_PAR")
        .map(|v| v != "0")
        .unwrap_or(true);
    // Frame profiler (CMF_GPU_TS=1): 256 timestamp slots + resolve/stage
    // buffers. Created only when the device carries the feature.
    let ts_query = if want_ts && matches!(std::env::var("CMF_GPU_TS").as_deref(), Ok("1") | Ok("2"))
    {
        let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("g-ts"),
            ty: wgpu::QueryType::Timestamp,
            count: 256,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g-ts-resolve"),
            size: 256 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g-ts-stage"),
            size: 256 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some((qs, resolve, stage))
    } else {
        None
    };
    let ts_period = queue.get_timestamp_period();
    let argmax_final = pipe("argmax_final");
    let embed_gather_q4tp = pipe("embed_gather_q4tp");
    let silu_down = pipe("silu_down_matvec");
    let q1t_mm = pipe("q1t_mul_mm");
    let q4t_mm = pipe("q4t_mul_mm");
    let dit_qk = pipe("dit_qk");
    let dit_pv = pipe("dit_pv");
    let dit_softmax = pipe("dit_softmax");
    let dit_unstack = pipe("dit_unstack");
    let ffn_silu = pipe("ffn_silu_mul");
    let q1t_ovmm = pipe("q1t_overlay_mm");
    let rmsnorm = pipe("rmsnorm");
    let add_rmsnorm = pipe("add_rmsnorm");
    let rmsnorm_b = pipe("rmsnorm_b");
    let add_rmsnorm_b = pipe("add_rmsnorm_b");
    let attn_rope = pipe("attn_rope_qkn");
    let kv_append = pipe("kv_append");
    let gqa_attend_s = pipe("gqa_attend_s");
    // 32 lanes where the workgroup budget allows it, 16 lanes where it does
    // not: both cover head_dim 256, so hd_cap is 256 everywhere now.
    let gqa_attend = if big_attend {
        pipe("gqa_attend")
    } else {
        pipe("gqa_attend_w16")
    };
    let gdn_step = pipe("gdn_step");
    let gdn_conv = pipe("gdn_conv");
    let f32_matvec = pipe("f32_matvec");
    let f32_matvec_b = pipe("f32_matvec_b");
    let layout_f32b = f32_matvec_b.get_bind_group_layout(0);
    let o1_far = pipe("o1_far");
    let o1_push = pipe("o1_push");
    let o1_attend = pipe("o1_attend");
    let layout_o1_far = o1_far.get_bind_group_layout(0);
    let layout_o1_push = o1_push.get_bind_group_layout(0);
    let layout_o1_attend = o1_attend.get_bind_group_layout(0);
    let matvec_pair = pipe("matvec_pair");
    let layout_mv2 = matvec_pair.get_bind_group_layout(0);
    let moe_select = pipe("moe_select");
    let moe_gate_up = pipe("moe_gate_up");
    let moe_down = pipe("moe_down");
    let moe_gate_up_q4tp = pipe("moe_gate_up_q4tp");
    let moe_down_q4tp = pipe("moe_down_q4tp");
    let moe_select_b = pipe("moe_select_b");
    let moe_gate_up_q4tp_b = pipe("moe_gate_up_q4tp_b");
    let moe_down_q4tp_b = pipe("moe_down_q4tp_b");
    let layout_moe_sel_b = moe_select_b.get_bind_group_layout(0);
    let layout_moe_gu_b = moe_gate_up_q4tp_b.get_bind_group_layout(0);
    let layout_moe_dn_b = moe_down_q4tp_b.get_bind_group_layout(0);
    let layout_moe_sel = moe_select.get_bind_group_layout(0);
    let layout_moe_gu = moe_gate_up.get_bind_group_layout(0);
    let layout_moe_dn = moe_down.get_bind_group_layout(0);
    let layout_moe_gu_q4tp = moe_gate_up_q4tp.get_bind_group_layout(0);
    let moe_gate_up_q2tp = pipe("moe_gate_up_q2tp");
    let moe_gate_up_q2tp_f = pipe("moe_gate_up_q2tp_f");
    let moe_down_q4tp_f = pipe("moe_down_q4tp_f");
    let gqa_attend_dec = pipe("gqa_attend_dec");
    let attend_dec = std::env::var("CMF_ATTEND_DEC")
        .map(|v| v != "0")
        .unwrap_or(true);
    // Measured NEGATIVE on RTX PRO 6000 (72.6 vs 79.0 tok/s): the redundant
    // per-workgroup top-k costs more than the retired select hop — in-pass
    // dispatches overlap more than the latency model assumed. Kept for
    // study; CMF_MOE_FOLDSEL=1 enables.
    let foldsel = std::env::var("CMF_MOE_FOLDSEL").as_deref() == Ok("1");
    let layout_moe_gu_q2tp = moe_gate_up_q2tp.get_bind_group_layout(0);
    let layout_moe_dn_q4tp = moe_down_q4tp.get_bind_group_layout(0);
    let layout = matvec.get_bind_group_layout(0);
    let layout_q1 = q1.get_bind_group_layout(0);
    let layout_rmsnorm = rmsnorm.get_bind_group_layout(0);
    let layout_add_rmsnorm = add_rmsnorm.get_bind_group_layout(0);
    let layout_rmsnorm_b = rmsnorm_b.get_bind_group_layout(0);
    let layout_add_rmsnorm_b = add_rmsnorm_b.get_bind_group_layout(0);
    let layout_attn_rope = attn_rope.get_bind_group_layout(0);
    let layout_kv = kv_append.get_bind_group_layout(0);
    let layout_attend = gqa_attend.get_bind_group_layout(0);
    let layout_attend_s = gqa_attend_s.get_bind_group_layout(0);
    let split_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-attend-split"),
        source: wgpu::ShaderSource::Wgsl(ATTEND_SPLIT_SRC.into()),
    });
    let pipe_split = |ep: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: None,
            module: &split_module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let attend_part_s = pipe_split("gqa_attend_part_s");
    let attend_part = if big_attend {
        pipe_split("gqa_attend_part")
    } else {
        attend_part_s.clone()
    };
    let attend_merge = pipe_split("gqa_attend_merge");
    // Subgroup select: its own module — `enable subgroups` must never
    // reach a device without the feature.
    let moe_select_sg = if want_sg && std::env::var("CMF_SELECT_SG").as_deref() != Ok("0") {
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cmf-select-sg"),
            source: wgpu::ShaderSource::Wgsl(SELECT_SG_SRC.into()),
        });
        Some(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("moe_select_sg"),
                layout: None,
                module: &m,
                entry_point: Some("moe_select_sg"),
                compilation_options: Default::default(),
                cache: None,
            }),
        )
    } else {
        None
    };
    let layout_attend_part = attend_part.get_bind_group_layout(0);
    let layout_attend_part_s = attend_part_s.get_bind_group_layout(0);
    let layout_attend_merge = attend_merge.get_bind_group_layout(0);
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

    if let Some(e) = pollster::block_on(vscope.pop()) {
        return Err(format!("wgpu pipeline validation: {e}"));
    }

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
        q4t_mv,
        rope_heads,
        o_lora_a,
        kv_pool,
        index_scores,
        top_k_index,
        moe_route,
        sparse_attend,
        sa_scores,
        sa_apply,
        hc_pre_fold,
        hc_post_expand,
        q4tp_mv,
        q4tp_mv4,
        use_mv4,
        q4tp_mv16,
        q4t_mv8,
        q4b_mv8,
        q4tp_mm,
        argmax_part,
        gdn_step_par,
        gdn_step_par2,
        gdn_step_norm2,
        gdn_inline,
        gdn_step_norm,
        gdn_par,
        ts_query,
        ts_period,
        argmax_final,
        embed_gather_q4tp,
        silu_down,
        q1t_mm,
        q4t_mm,
        dit_qk,
        dit_pv,
        dit_softmax,
        dit_unstack,
        ffn_silu,
        q1t_ovmm,
        rmsnorm,
        add_rmsnorm,
        rmsnorm_b,
        add_rmsnorm_b,
        attn_rope,
        kv_append,
        gqa_attend,
        gqa_attend_s,
        attend_part,
        attend_part_s,
        attend_merge,
        hd_cap: 256,
        big_attend,
        gdn_step,
        gdn_conv,
        f32_matvec,
        f32_matvec_b,
        layout_f32b,
        o1_far,
        o1_push,
        o1_attend,
        layout_o1_far,
        layout_o1_push,
        layout_o1_attend,
        o1m: Mutex::new(HashMap::new()),
        matvec_pair,
        layout_mv2,
        moe_select,
        moe_gate_up,
        moe_down,
        moe_gate_up_q4tp,
        moe_down_q4tp,
        moe_select_b,
        moe_gate_up_q4tp_b,
        moe_down_q4tp_b,
        layout_moe_sel_b,
        layout_moe_gu_b,
        layout_moe_dn_b,
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
        layout_attend_s,
        layout_attend_part,
        layout_attend_part_s,
        layout_attend_merge,
        layout_gdn,
        layout_gdn_conv,
        layout_f32,
        layout_silu_down,
        layout_moe_sel,
        layout_moe_gu,
        layout_moe_dn,
        layout_moe_gu_q4tp,
        moe_gate_up_q2tp,
        moe_gate_up_q2tp_f,
        moe_down_q4tp_f,
        gqa_attend_dec,
        moe_select_sg,
        attend_dec,
        foldsel,
        layout_moe_gu_q2tp,
        layout_moe_dn_q4tp,
        discrete,
        vram_budget,
        resident: std::sync::atomic::AtomicU64::new(0),
        scratch: Mutex::new(Scratch::default()),
        weight_bufs: Mutex::new(HashMap::new()),
        dsv4_kv: Mutex::new(HashMap::new()),
        dsv4_scratch: Mutex::new(HashMap::new()),
        dsv4_binds: Mutex::new((0, HashMap::new())),
        res_clock: std::sync::atomic::AtomicU64::new(0),
        uniforms: Mutex::new(HashMap::new()),
        uniforms8: Mutex::new(HashMap::new()),
        const_bufs: Mutex::new(HashMap::new()),
        rs_bufs: Mutex::new(HashMap::new()),
        attn_kv: Mutex::new(HashMap::new()),
        gdn_state: Mutex::new(HashMap::new()),
        moe_expw: Mutex::new(HashMap::new()),
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
/// One tensor living on the device, with what the eviction policy needs.
struct Resident {
    buf: wgpu::Buffer,
    bytes: u64,
    /// Use count, aged lazily: the score at time `t` is `uses ·
    /// DECAY^(t − last)`. Plain frequency ossifies — an expert that was
    /// popular during the first prompt outranks one being used right now,
    /// forever.
    uses: f32,
    last: u64,
}

/// Per-tick multiplier for the aged use count. 0.999 halves a score over
/// ~700 lookups: long enough that a steady working set is never disturbed,
/// short enough that a change of task migrates within a prompt or two.
const RES_DECAY: f32 = 0.999;

/// A tensor touched within this many lookups is not evicted, whatever its
/// score. Without it a budget slightly smaller than the working set evicts
/// and re-uploads on every token, which is slower than never using the
/// device at all.
const RES_HYSTERESIS: u64 = 512;

fn res_score(e: &Resident, now: u64) -> f32 {
    e.uses * RES_DECAY.powi((now.saturating_sub(e.last)).min(4096) as i32)
}

fn weight_buffer(c: &Ctx, key: (usize, usize), full_quant: &[u8]) -> Option<wgpu::Buffer> {
    use std::sync::atomic::Ordering;
    let now = c.res_clock.fetch_add(1, Ordering::Relaxed);
    let mut map = c.weight_bufs.lock().unwrap();
    if let Some(e) = map.get_mut(&key) {
        e.uses = res_score(e, now) + 1.0;
        e.last = now;
        return Some(e.buf.clone());
    }
    let len = full_quant.len() as u64;
    if len > c.vram_budget {
        return None; // one tensor larger than the whole budget
    }
    // Make room by evicting the least valuable, skipping anything touched
    // recently. Failing to free enough is not an error: the tensor stays on
    // the CPU, which is the pressure valve that keeps a too-small budget
    // from thrashing the bus.
    if c.resident.load(Ordering::Relaxed) + len > c.vram_budget {
        let mut cand: Vec<((usize, usize), f32, u64)> = map
            .iter()
            .filter(|(_, e)| now.saturating_sub(e.last) > RES_HYSTERESIS)
            .map(|(k, e)| (*k, res_score(e, now), e.bytes))
            .collect();
        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut freed = 0u64;
        for (k, _, bytes) in cand {
            if c.resident.load(Ordering::Relaxed) - freed + len <= c.vram_budget {
                break;
            }
            map.remove(&k);
            freed += bytes;
        }
        if freed > 0 {
            c.resident.fetch_sub(freed, Ordering::Relaxed);
            crate::gpu::probe_note_cold();
        }
        if c.resident.load(Ordering::Relaxed) + len > c.vram_budget {
            return None; // still no room — honest CPU
        }
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
    map.insert(
        key,
        Resident {
            buf: buf.clone(),
            bytes: len,
            uses: 1.0,
            last: now,
        },
    );
    Some(buf)
}

/// One MoE layer's expert weights as three concatenated device buffers
/// (gate_all, up_all, down_all), q4t payloads back to back in `experts`
/// order (routed experts then the shared one) — the kernels address
/// expert e at u16 offset e·mat16. Uploaded once per layer (keyed by the
/// first gate idx), budget-guarded like every resident weight; the copy
/// walks the per-tensor directory, so no file-order contiguity is assumed.
/// One-shot reason the whole-token graph declined. Without it the fallback
/// to the per-op path is invisible, which is how a q4tp model looked
/// GPU-accelerated while every layer walked the host.
fn graph_refused(why: &'static str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        tracing::warn!("wgpu token graph declined: {why}");
    }
}

fn moe_expert_bufs(
    c: &Ctx,
    model: &Arc<CmfModel>,
    experts: &[(usize, usize, usize)],
    inter: usize,
    hidden: usize,
    q4tp: bool,
    gu_q2: bool,
) -> Option<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> {
    use std::sync::atomic::Ordering;
    if hidden % 32 != 0 || inter % 32 != 0 {
        graph_refused("moe_expert_bufs: hidden/inter not 32-aligned");
        return None;
    }
    let bytes = model.primary_bytes();
    let key = (bytes.as_ptr() as usize, experts.first()?.0);
    if let Some(t) = c.moe_expw.lock().unwrap().get(&key) {
        return Some(t.clone());
    }
    // q4t is 18 B a group flat; q4tp is 16 B of nibbles plus the row
    // params and 5-bit code planes, which the format's own accessor sizes.
    let plen = |rows: usize, cols: usize| -> Option<usize> {
        if q4tp {
            cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[rows, cols])
        } else {
            Some(rows * (cols / 32) * 18)
        }
    };
    let gu_len = if gu_q2 {
        cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q2TiledP, &[inter, hidden])?
    } else {
        plen(inter, hidden)?
    };
    let d_len = plen(hidden, inter)?;
    let total = (experts.len() * (2 * gu_len + d_len)) as u64;
    if c.resident.load(Ordering::Relaxed) + total > c.vram_budget {
        // Over budget — the whole graph falls to CPU. Say so ONCE with the
        // numbers: the default budget is a conservative 8 GB on discrete
        // cards, so a 32 GB card running a big MoE lands here and every
        // expert quietly walks the host. That refusal used to be silent.
        use std::sync::atomic::AtomicBool;
        static SAID: AtomicBool = AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            let mb = |b: u64| b / (1024 * 1024);
            tracing::warn!(
                "MoE experts need {} MB for this layer on top of {} MB resident, \
                 over the {} MB weight budget — the whole-token graph falls back to \
                 the CPU. Raise it with CMF_GPU_VRAM_MB (e.g. {} on this card).",
                mb(total),
                mb(c.resident.load(Ordering::Relaxed)),
                mb(c.vram_budget),
                mb(c.vram_budget) * 3,
            );
        }
        return None;
    }
    crate::gpu::probe_note_cold();
    // Every byte range this layer ships to the card, for the post-upload
    // evict below.
    let uploaded = std::cell::RefCell::new(Vec::<(usize, usize)>::new());
    let mk = |role: &dyn Fn(&(usize, usize, usize)) -> usize,
              rows: usize,
              cols: usize,
              plen: usize|
     -> Option<wgpu::Buffer> {
        // Check every expert BEFORE allocating: a shape mismatch found
        // halfway used to leave a gigabyte-scale buffer behind.
        let mut offs = Vec::with_capacity(experts.len());
        for t in experts {
            let e = model.tensors.get(role(t))?;
            if *e.shape.first()? as usize != rows
                || *e.shape.get(1)? as usize != cols
                || e.nbytes as usize != plen
            {
                return None;
            }
            let abs = model.entry_abs_offset(e)?;
            bytes.get(abs..abs + plen)?; // in range
            offs.push(abs);
        }
        let b = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moe-experts"),
            size: (experts.len() * plen) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Straight from the mapping to the queue, expert by expert. Gathering
        // them into one Vec first meant a 2.2 GB allocation and a full extra
        // memcpy PER LAYER — 94 GB of pointless copying across the release,
        // on top of the 94 GB that has to move anyway.
        for (i, &abs) in offs.iter().enumerate() {
            c.queue
                .write_buffer(&b, (i * plen) as u64, &bytes[abs..abs + plen]);
        }
        uploaded.borrow_mut().extend(offs.iter().map(|&a| (a, plen)));
        Some(b)
    };
    let (g, u, d) = match (
        mk(&|t| t.0, inter, hidden, gu_len),
        mk(&|t| t.1, inter, hidden, gu_len),
        mk(&|t| t.2, hidden, inter, d_len),
    ) {
        (Some(g), Some(u), Some(d)) => (g, u, d),
        _ => {
            graph_refused("moe_expert_bufs: expert tensor shape/nbytes mismatch");
            return None;
        }
    };
    // Flush the write_buffer staging belt NOW: with 40 MoE layers the
    // pending uploads (~17 GB) would otherwise coexist with their device
    // copies until the graph's first submit — twice the expert weights in
    // memory = device OOM on discrete cards. One submit+wait per layer
    // bounds transient staging to this layer's three buffers.
    c.queue.submit(std::iter::empty());
    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    // The card holds these bytes now, so the host copy is dead weight: the
    // page cache otherwise keeps every resident expert a second time, and on
    // a 112 GB model that second copy IS the machine's RAM (measured: 172 of
    // 176 GB cached).
    //
    // Alternated off/on/off/on from a warmed file, 256 tokens each:
    //
    //   off  94 GB resident  1.0 tok/s      off  91 GB  0.5 tok/s
    //   on    5 GB resident  6.9 tok/s      on    5 GB  6.1 tok/s
    //
    // It is not a trade — holding a second copy of the weights is what was
    // costing the speed. At 94 GB resident the machine has nothing left for
    // the pages it does need, and reclaim churns; at 5 GB it does not.
    //
    // Pairs with `open()` skipping its whole-file `WillNeed` when this is on:
    // reading 104 GB ahead only to drop it behind the uploader had the kernel
    // fetching the same bytes twice. `CMF_UPLOAD_EVICT=0` opts out; discrete
    // only, as on UMA the mapping IS the working copy.
    if c.discrete
        && std::env::var("CMF_UPLOAD_EVICT")
            .map(|v| v != "0")
            .unwrap_or(true)
    {
        model.evict_ranges(&uploaded.borrow());
    }
    c.resident.fetch_add(total, Ordering::Relaxed);
    c.moe_expw
        .lock()
        .unwrap()
        .insert(key, (g.clone(), u.clone(), d.clone()));
    Some((g, u, d))
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
    dispatch_matvec(
        c,
        Some(key),
        full_quant,
        row0,
        row_scale,
        xs,
        rows,
        cols,
        out,
    )
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
        None => c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("q8-weights"),
                contents: full_quant,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    };
    let make_rs = || {
        c.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
    let y_size = (rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q8-y",
    );
    let params = [
        (cols / 4) as u32,
        rows as u32,
        (row0 * cols / 4) as u32,
        0u32,
    ];
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
    c.queue
        .write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
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
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    // sanity: the base must at least fit (q1t base 9 B/group, q4b 18 B/group).
    let min_base = if q4 { rows * gpr * 18 } else { rows * gpr * 9 };
    if plen < min_base || abs + plen > bytes.len() {
        return false;
    }
    let pipeline = if q4 { &c.q4b } else { &c.q1t };
    dispatch_q1t(
        c,
        pipeline,
        Some((bytes.as_ptr() as usize, idx)),
        &bytes[abs..abs + plen],
        xs,
        rows,
        cols,
        out,
    )
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
        None => c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
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
    c.queue
        .write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
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
    dispatch_q1(
        c,
        Some((bytes.as_ptr() as usize, idx)),
        &bytes[abs..abs + plen],
        xs,
        rows,
        cols,
        out,
    )
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
    let k_b = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        nh as u32,
        nkv as u32,
        hd as u32,
        rd as u32,
        pos as u32,
        flags,
        eps.to_bits(),
        0u32,
    ];
    let p_buf = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    if hd % 4 != 0 || hd > c.hd_cap {
        return false; // vec4 K/V reads; hd_cap = workgroup-storage limit
    }
    let q_b = storage_bytes(c, bytemuck::cast_slice(q));
    let k_b = storage_bytes(c, bytemuck::cast_slice(kcache));
    let v_b = storage_bytes(c, bytemuck::cast_slice(vcache));
    let o_b = rw_f32(c, nh * hd, true);
    let p_buf = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("at-p"),
            contents: bytemuck::cast_slice(&[
                nh as u32, hpk as u32, hd as u32, cap as u32, n as u32, 0u32, 0u32, 0u32,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("at-bg"),
        layout: attend_pipes(c, hd).1,
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
        pass.set_pipeline(attend_pipes(c, hd).0);
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
/// Is wgpu initialized on a DISCRETE adapter? Gates the whole-token
/// graph default (see gpu::wgpu_graph_default).
pub(crate) fn discrete_active() -> bool {
    ctx().map(|c| c.discrete).unwrap_or(false)
}

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

/// A q4_tiled / q4tp weight as one device buffer — the whole tensor, since
/// both layouts keep their scales inside (q4t) or in trailing planes (q4tp)
/// and the kernels index them from the same base.
fn tile_weight(c: &Ctx, model: &Arc<CmfModel>, idx: usize) -> Option<(wgpu::Buffer, usize, usize)> {
    let entry = model.tensors.get(idx)?;
    let rows = *entry.shape.first()? as usize;
    let cols = *entry.shape.get(1)? as usize;
    if cols % 32 != 0 {
        return None;
    }
    let abs = model.entry_abs_offset(entry)?;
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
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
    if pos >= cap || hd % 4 != 0 || hd > c.hd_cap {
        return false; // vec4 K/V reads; hd_cap = workgroup-storage limit
    }
    let (wq, rq, cq) = q1_weight(c, model, wq_idx).unwrap_or((
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }),
        0,
        0,
    ));
    if rq != nh * hd || cq != hidden {
        return false; // gated arch (e.g. output_gate doubles rows) → CPU path
    }
    let Some((wk, _, _)) = q1_weight(c, model, wk_idx) else {
        return false;
    };
    let Some((wv, _, _)) = q1_weight(c, model, wv_idx) else {
        return false;
    };
    let Some((wo, ro, co)) = q1_weight(c, model, wo_idx) else {
        return false;
    };
    if ro != hidden || co != nh * hd {
        return false;
    }
    // Device K/V mirror (persist across tokens).
    let mut kvm = c.attn_kv.lock().unwrap();
    let entry = kvm.entry((kv_id, layer)).or_insert_with(|| {
        let sz = (nkv * cap * hd * 4) as u64;
        let mk = || {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kv-mirror"),
                size: sz,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        KvMirror {
            k: mk(),
            v: mk(),
            synced: 0,
        }
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
                c.queue.write_buffer(
                    &entry.k,
                    off,
                    bytemuck::cast_slice(&src_k[from * hd..take * hd]),
                );
                c.queue.write_buffer(
                    &entry.v,
                    off,
                    bytemuck::cast_slice(&src_v[from * hd..take * hd]),
                );
            }
        }
        entry.synced = pos;
    }
    let kbuf = entry.k.clone();
    let vbuf = entry.v.clone();
    drop(kvm);

    let stor = |data: &[u8]| {
        c.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
    };
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
    let flags = if q_norm.is_some() { 2u32 } else { 0 }
        | if k_norm.is_some() { 4 } else { 0 }
        | if gemma { 8 } else { 0 };
    let unif = |d: &[u32]| {
        c.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(d),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    };
    let bg = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let e: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| bind_buf(i as u32, b))
            .collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &e,
        })
    };
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn-dropin"),
        });
    let go =
        |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, b: &wgpu::BindGroup, g: u32| {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(p);
            pass.set_bind_group(0, b, &[]);
            pass.dispatch_workgroups(g, 1, 1);
        };
    encode_matvec_q1(c, &mut enc, &wq, &normed_b, &qraw_b, nh * hd, hidden);
    encode_matvec_q1(c, &mut enc, &wk, &normed_b, &k_b, nkv * hd, hidden);
    encode_matvec_q1(c, &mut enc, &wv, &normed_b, &v_b, nkv * hd, hidden);
    let rq_p = unif(&[
        nh as u32,
        nkv as u32,
        hd as u32,
        rd as u32,
        pos as u32,
        flags,
        eps.to_bits(),
        0,
    ]);
    go(
        &mut enc,
        &c.attn_rope,
        &bg(
            &c.layout_attn_rope,
            &[
                &qraw_b, &k_b, &qout_b, &gout_b, &qnw_b, &knw_b, &invf_b, &rq_p,
            ],
        ),
        (nh + nkv) as u32,
    );
    let kv_p = unif(&[nkv as u32, hd as u32, cap as u32, pos as u32]);
    go(
        &mut enc,
        &c.kv_append,
        &bg(&c.layout_kv, &[&k_b, &v_b, &kbuf, &vbuf, &kv_p]),
        ((nkv * hd) as u32).div_ceil(256),
    );
    let at_p = unif(&[
        nh as u32,
        (nh / nkv) as u32,
        hd as u32,
        cap as u32,
        (pos + 1) as u32,
        0,
        0,
        0,
    ]);
    {
        let (ap, al) = attend_pipes(c, hd);
        go(
            &mut enc,
            ap,
            &bg(al, &[&qout_b, &kbuf, &vbuf, &attn_b, &at_p]),
            nh as u32,
        );
    }
    encode_matvec_q1(c, &mut enc, &wo, &attn_b, &o_b, hidden, nh * hd);
    let size = (hidden * 4) as u64;
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "dropin-stage",
    );
    let ok = readback(c, enc, &o_b, &stage, size, &mut attn_out[..hidden]);
    drop(sc);
    if ok {
        c.attn_kv
            .lock()
            .unwrap()
            .get_mut(&(kv_id, layer))
            .map(|m| m.synced = pos + 1);
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
    o1: &[Option<Vec<crate::nystrom::O1DeviceView<'_>>>],
    o1_epoch: u64,
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
    // Multi-step greedy: encode `steps` whole frames in THIS submit, argmax
    // and re-embed on the device, and return the k winner ids instead of
    // logits. Needs `embed` = (q4tp embedding weight, vocab rows, multiplier).
    steps: usize,
    embed: Option<(&crate::gpu::GraphW, usize, f32)>,
    ids_out: Option<&mut Vec<u32>>,
) -> bool {
    let Some(c) = ctx() else {
        graph_refused("no ctx");
        return false;
    };
    if position >= cap || hd % 4 != 0 || hd > c.hd_cap {
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "wgpu token graph declined: position {position} >= cap {cap}, or head_dim \
                     {hd} (must be %4 and <= hd_cap {})",
                    c.hd_cap
                );
            }
        }
        return false; // vec4 K/V reads; hd_cap = workgroup-storage limit
    }
    let t_start = std::time::Instant::now();
    // A resolved matvec weight: the device-local buffer, (q8 only) its row
    // scales, and the codec kind (0=q8_row 1=q1 2=q4_block 3=q1t 4=f32 5=q4_tiled).
    struct GMat {
        buf: wgpu::Buffer,
        rs: Option<wgpu::Buffer>,
        kind: u8,
    }
    enum LAttn {
        Full {
            wq: GMat,
            wk: GMat,
            wv: GMat,
            wo: GMat,
        },
        Gdn {
            qkv: GMat,
            z: GMat,
            a: GMat,
            b: GMat,
            out: GMat,
            nv: usize,
            nk: usize,
            dk: usize,
            dv: usize,
            kk: usize,
            cdim: usize,
        },
    }
    enum LFfn {
        Dense {
            gate: GMat,
            up: GMat,
            down: GMat,
        },
        Moe {
            router: GMat,
            sgate: GMat,
            gate_all: wgpu::Buffer,
            up_all: wgpu::Buffer,
            down_all: wgpu::Buffer,
            n_exp: usize,
            top_k: usize,
            inter: usize,
            norm_topk: bool,
            q4tp: bool,
            gu_q2: bool,
        },
    }
    struct LW {
        attn: LAttn,
        ffn: LFfn,
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
                    c.queue
                        .write_buffer(&x, 0, bytemuck::cast_slice(&gw.row_scale[..rows]));
                    cb.insert(key, x.clone());
                    x
                };
                Some(GMat {
                    buf: b,
                    rs: Some(rsb),
                    kind: 0,
                })
            }
            1 => {
                let (b, r, cc) = q1_weight(c, model, gw.idx)?;
                if r != rows || cc != cols {
                    return None;
                }
                Some(GMat {
                    buf: b,
                    rs: None,
                    kind: 1,
                })
            }
            2 | 3 | 5 | 6 => {
                // q4_block / q1t / q4_tiled / q4tp: the tensor carries its own byte
                // length (tiles + q1t's sparse overlay) — fetch whole,
                // device-local.
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
                let b = weight_buffer(
                    c,
                    (bytes.as_ptr() as usize, gw.idx),
                    &bytes[abs..abs + plen],
                )?;
                Some(GMat {
                    buf: b,
                    rs: None,
                    kind: gw.kind,
                })
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
                    c.queue
                        .write_buffer(&x, 0, bytemuck::cast_slice(&gw.data[..rows * cols]));
                    cb.insert(key, x.clone());
                    x
                };
                Some(GMat {
                    buf: b,
                    rs: None,
                    kind: 4,
                })
            }
            _ => None,
        }
    };
    let mut lws = Vec::with_capacity(layers.len());
    let mut gdn_dims: Option<(usize, usize, usize, usize, usize, usize)> = None; // nv,nk,dk,dv,kk,cdim
    for l in layers {
        let attn = match &l.attn {
            crate::gpu::GraphAttn::Full {
                wq,
                wk,
                wv,
                wo,
                output_gate,
                ..
            } => {
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
            crate::gpu::GraphAttn::Gdn {
                qkv,
                z,
                a,
                b,
                out,
                nv,
                nk,
                dk,
                dv,
                kk,
                ..
            } => {
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
                LAttn::Gdn {
                    qkv,
                    z,
                    a,
                    b,
                    out,
                    nv: *nv,
                    nk: *nk,
                    dk: *dk,
                    dv: *dv,
                    kk: *kk,
                    cdim,
                }
            }
        };
        let ffn = match &l.ffn {
            crate::gpu::GraphFfn::Dense { gate, up, down } => {
                let (Some(gate), Some(up), Some(down)) = (
                    resolve(gate, inter, hidden),
                    resolve(up, inter, hidden),
                    resolve(down, hidden, inter),
                ) else {
                    return false;
                };
                LFfn::Dense { gate, up, down }
            }
            crate::gpu::GraphFfn::Moe {
                router,
                shared_gate,
                experts,
                n_exp,
                top_k,
                inter: mi,
                norm_topk,
                q4tp,
                gu_q2,
            } => {
                // Select kernel: logits live in a 256-slot workgroup array;
                // slot top_k+1 holds the shared expert.
                if *top_k >= 16 || *n_exp > 256 || experts.len() != n_exp + 1 {
                    return false;
                }
                let (Some(router), Some(sgate)) = (
                    resolve(router, *n_exp, hidden),
                    resolve(shared_gate, 1, hidden),
                ) else {
                    return false;
                };
                let Some((gate_all, up_all, down_all)) =
                    moe_expert_bufs(c, model, experts, *mi, hidden, *q4tp, *gu_q2)
                else {
                    return false;
                };
                LFfn::Moe {
                    router,
                    sgate,
                    gate_all,
                    up_all,
                    down_all,
                    n_exp: *n_exp,
                    top_k: *top_k,
                    inter: *mi,
                    norm_topk: *norm_topk,
                    q4tp: *q4tp,
                    gu_q2: *gu_q2,
                }
            }
        };
        lws.push(LW { attn, ffn });
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
        let b = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        let b = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g-zero"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&b, 0, &vec![0u8; n * 4]);
        cb.insert(key, b.clone());
        b
    };
    let unif = |d: &[u32]| {
        c.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(d),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    };
    let bg = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let e: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| bind_buf(i as u32, b))
            .collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &e,
        })
    };
    // ── Pooled scratch: all intermediate buffers are reused across tokens ──
    let mut gs = c.graph_scratch.lock().unwrap();
    let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC; // COPY_SRC: debug taps (CMF_O1_TRACE)
    let h_buf = GraphScratch::ensure(
        &c.device,
        &mut gs.h,
        (hidden * 4) as u64,
        st | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        "g-h",
    );
    c.queue
        .write_buffer(&h_buf, 0, bytemuck::cast_slice(&h[..hidden]));
    let n1 = GraphScratch::ensure(
        &c.device,
        &mut gs.n1,
        (hidden * 4) as u64,
        st | wgpu::BufferUsages::COPY_SRC,
        "g-n1",
    );
    // Gated attention (Qwen3.5) makes wq emit 2·nh·hd (q||gate per head), so the
    // raw-QKV scratch must hold the widened q output for any gated layer.
    let any_gate = layers.iter().any(|l| {
        matches!(
            &l.attn,
            crate::gpu::GraphAttn::Full {
                output_gate: true,
                ..
            }
        )
    });
    let qraw = GraphScratch::ensure(
        &c.device,
        &mut gs.qraw,
        (nh * hd * (1 + any_gate as usize) * 4) as u64,
        st,
        "g-qraw",
    );
    let kb = GraphScratch::ensure(&c.device, &mut gs.kb, (nkv * hd * 4) as u64, st, "g-kb");
    let vb = GraphScratch::ensure(&c.device, &mut gs.vb, (nkv * hd * 4) as u64, st, "g-vb");
    let qout = GraphScratch::ensure(&c.device, &mut gs.qout, (nh * hd * 4) as u64, st, "g-qout");
    let gout = GraphScratch::ensure(&c.device, &mut gs.gout, (nh * hd * 4) as u64, st, "g-gout");
    let attn = GraphScratch::ensure(&c.device, &mut gs.attn, (nh * hd * 4) as u64, st, "g-attn");
    let ob = GraphScratch::ensure(&c.device, &mut gs.ob, (hidden * 4) as u64, st, "g-ob");
    let gbuf = GraphScratch::ensure(&c.device, &mut gs.gbuf, (inter * 4) as u64, st, "g-gbuf");
    let ubuf = GraphScratch::ensure(&c.device, &mut gs.ubuf, (inter * 4) as u64, st, "g-ubuf");
    let abuf = GraphScratch::ensure(&c.device, &mut gs.abuf, (inter * 4) as u64, st, "g-abuf");
    // MoE routing scratch, sized to the largest MoE layer (absent → skipped).
    let moe_geom = lws
        .iter()
        .filter_map(|lw| match &lw.ffn {
            LFfn::Moe {
                n_exp,
                top_k,
                inter,
                ..
            } => Some((*n_exp, *top_k + 1, *inter)),
            _ => None,
        })
        .reduce(|a, b| (a.0.max(b.0), a.1.max(b.1), a.2.max(b.2)));
    let moe_bufs = moe_geom.map(|(mn, ms, mi)| {
        (
            GraphScratch::ensure(&c.device, &mut gs.m_logit, (mn * 4) as u64, st, "g-mlogit"),
            GraphScratch::ensure(&c.device, &mut gs.m_slog, 4, st, "g-mslog"),
            GraphScratch::ensure(&c.device, &mut gs.m_sel, (ms * 4) as u64, st, "g-msel"),
            GraphScratch::ensure(&c.device, &mut gs.m_wt, (ms * 4) as u64, st, "g-mwt"),
            GraphScratch::ensure(&c.device, &mut gs.m_act, (ms * mi * 4) as u64, st, "g-mact"),
        )
    });
    let invf_b = stor(bytemuck::cast_slice(invf));
    let dummy_hd = zeros(hd);
    // GDN intermediates (sized to the model's GDN geometry; 1 if no GDN layer).
    let (gnv, _gnk, gdk, gdv, _gkk, gcdim) = gdn_dims.unwrap_or((1, 1, 1, 1, 1, 1));
    let qkv_b = GraphScratch::ensure(&c.device, &mut gs.qkv_b, (gcdim * 4) as u64, st, "g-qkv");
    let cq_b = GraphScratch::ensure(&c.device, &mut gs.cq_b, (gcdim * 4) as u64, st, "g-cq");
    let z_b = GraphScratch::ensure(&c.device, &mut gs.z_b, (gnv * gdv * 4) as u64, st, "g-z");
    let a_b = GraphScratch::ensure(&c.device, &mut gs.a_b, (gnv * 4) as u64, st, "g-a");
    let b_b = GraphScratch::ensure(&c.device, &mut gs.b_b, (gnv * 4) as u64, st, "g-b");
    let gdo_b = GraphScratch::ensure(
        &c.device,
        &mut gs.gdo_b,
        (gnv * gdv * 4) as u64,
        st,
        "g-gdo",
    );
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
                    if o1.get(li).is_some_and(|v| v.is_some()) {
                        // o1 replaces this layer's KV attention outright —
                        // no mirror, and no prefill K/V upload (16K of it
                        // at long context) that nothing would read.
                        kvbufs.push(None);
                        gdnbufs.push(None);
                        continue;
                    }
                    let e = kvm.entry((kv_id, li)).or_insert_with(|| {
                        let sz = (nkv * cap * hd * 4) as u64;
                        let mk = || {
                            c.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("kv"),
                                size: sz,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            })
                        };
                        KvMirror {
                            k: mk(),
                            v: mk(),
                            synced: 0,
                        }
                    });
                    if e.synced < position {
                        for hh in 0..nkv {
                            let take = position.min(cpu_k[hh].len() / hd);
                            if take > e.synced {
                                let off = ((hh * cap + e.synced) * hd * 4) as u64;
                                c.queue.write_buffer(
                                    &e.k,
                                    off,
                                    bytemuck::cast_slice(&cpu_k[hh][e.synced * hd..take * hd]),
                                );
                                c.queue.write_buffer(
                                    &e.v,
                                    off,
                                    bytemuck::cast_slice(&cpu_v[hh][e.synced * hd..take * hd]),
                                );
                            }
                        }
                        e.synced = position;
                    }
                    kvbufs.push(Some((e.k.clone(), e.v.clone())));
                    gdnbufs.push(None);
                }
                crate::gpu::GraphAttn::Gdn { cpu_state, .. } => {
                    let e = gsm.entry((kv_id, li)).or_insert_with(|| {
                        let ring_sz = ((gcdim * (_gkk.max(1).saturating_sub(1))) * 4) as u64;
                        let s_sz = (gnv * gdk * gdv * 4) as u64;
                        let mk = |sz: u64| {
                            let bf = c.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("gdn-state"),
                                size: sz.max(4),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            });
                            c.queue.write_buffer(&bf, 0, &vec![0u8; sz.max(4) as usize]);
                            bf
                        };
                        let (ring, sbuf) = (mk(ring_sz), mk(s_sz));
                        // Seed from the CPU recurrence when the host ran the
                        // prefill (o1 collection, CPU fallback). A fresh entry
                        // with an EMPTY cpu_state is the graph-prefill flow —
                        // the graph builds the state itself from position 0.
                        // Zero-initialized device state at decode is the
                        // "coherent but contextless" failure this closes.
                        let want = (ring_sz + s_sz) as usize / 4;
                        if cpu_state.len() == want && want > 0 {
                            let ring_n = ring_sz as usize / 4;
                            c.queue.write_buffer(
                                &ring,
                                0,
                                bytemuck::cast_slice(&cpu_state[..ring_n]),
                            );
                            c.queue.write_buffer(
                                &sbuf,
                                0,
                                bytemuck::cast_slice(&cpu_state[ring_n..]),
                            );
                        }
                        (ring, sbuf)
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
    let group = std::env::var("CMF_GPU_GROUP")
        .map(|v| v != "0")
        .unwrap_or(true);
    // Hand strictly-serial single-dispatch stages to the NEXT pass instead of
    // opening a pass for each. Dispatch ORDER is unchanged, so the answer is
    // unchanged; only pass boundaries move. CMF_PASSFUSE=0 reverts.
    let passfuse = std::env::var("CMF_PASSFUSE")
        .map(|v| v != "0")
        .unwrap_or(true);
    // CMF_SKIP_PROBE=moe|gdn — TIMING ONLY, the answer is garbage. Drops a
    // whole stage's dispatches while leaving every buffer, pass and shape
    // in place, so the delta is that stage's real share of the frame. The
    // arithmetic-only probe (CMF_TOPK_PROBE) says MoE math is ~2 ms of 17;
    // this one says where the rest actually goes, which neither dispatch
    // counting nor pass counting predicted correctly.
    let skip = std::env::var("CMF_SKIP_PROBE").unwrap_or_default();
    let (skip_moe, skip_gdn) = (skip.contains("moe"), skip.contains("gdn"));
    // Skeleton pieces, so the 7.5 ms that is neither MoE nor GDN can be
    // attributed instead of guessed at: the four GDN input projections,
    // the GDN output projection, the fused residual+norm, the router, and
    // the whole full-attention chain.
    let skip_proj = skip.contains("proj");
    let skip_outp = skip.contains("outp");
    let skip_norm = skip.contains("norm");
    let skip_router = skip.contains("router");
    let skip_attn = skip.contains("attn");
    // CMF_LAYERS_PROBE=N — TIMING ONLY, the answer is garbage. Encodes just
    // the first N layers, leaving the final norm + lm_head + readback in
    // place. Decode time against N is a straight line whose slope is the
    // per-layer cost and whose intercept is everything that happens once a
    // token: the submit, the ~1 MB logits readback and the lm_head. Neither
    // dispatch counting nor pass counting predicted the frame correctly, so
    // this splits it by measurement instead.
    let layer_cap = std::env::var("CMF_LAYERS_PROBE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let t_enc0 = std::time::Instant::now();
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("token-graph"),
        });
    let go =
        |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, b: &wgpu::BindGroup, g: u32| {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(p);
            pass.set_bind_group(0, b, &[]);
            pass.dispatch_workgroups(g, 1, 1);
        };
    let flags = |qn: bool, kn: bool| {
        (if qn { 2u32 } else { 0 }) | (if kn { 4 } else { 0 }) | (if gemma { 8 } else { 0 })
    };
    // Constant uniforms for the whole token (position is fixed for this call).
    // Token-invariant ones use the content-keyed cache; position-dependent ones
    // use pooled buffers updated via write_buffer (no allocation after first token).
    let g = if gemma { 1u32 } else { 0 };
    let rms_u = uniform_u32x4(c, [hidden as u32, g, eps.to_bits(), 0]);
    let ax_u = uniform_u32x4(c, [1.0f32.to_bits(), hidden as u32, 0, 0]);
    let silu_u = uniform_u32x4(c, [inter as u32, 0, 0, 0]);
    let steps = steps.max(1);
    // One uniform PER STEP, with stable identities: write_buffer lands at
    // submit, so a single shared buffer would collapse every step to the
    // last position written. Slot 0 is the plain single-step path.
    let mku = |v: &mut Vec<wgpu::Buffer>, size: u64| {
        while v.len() < steps {
            v.push(c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("g-step-u"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    };
    mku(&mut gs.kv_us, 16);
    mku(&mut gs.at_us, 32);
    mku(&mut gs.rope_us, 32);
    for st in 0..steps {
        let p = position + st;
        c.queue.write_buffer(
            &gs.kv_us[st],
            0,
            bytemuck::cast_slice(&[nkv as u32, hd as u32, cap as u32, p as u32]),
        );
        c.queue.write_buffer(
            &gs.at_us[st],
            0,
            bytemuck::cast_slice(&[
                nh as u32,
                (nh / nkv) as u32,
                hd as u32,
                cap as u32,
                (p + 1) as u32,
                0,
                0,
                0,
            ]),
        );
    }
    let kv_us = std::mem::take(&mut gs.kv_us);
    let at_us = std::mem::take(&mut gs.at_us);
    let rope_us = std::mem::take(&mut gs.rope_us);
    // Encode one matvec, dtype-dispatched: q8_row (encode_matvec + row scales)
    // or q1 (encode_matvec_q1). Each is its own pass — pass-grouping measured
    // as a no-op (the wall is per-dispatch, not per-barrier).
    let emat = |enc: &mut wgpu::CommandEncoder,
                m: &GMat,
                xs: &wgpu::Buffer,
                y: &wgpu::Buffer,
                rows: usize,
                cols: usize| {
        match m.kind {
            0 => encode_matvec(c, enc, &m.buf, xs, m.rs.as_ref().unwrap(), y, rows, cols),
            1 => encode_matvec_q1(c, enc, &m.buf, xs, y, rows, cols),
            2 => encode_q1t_like(c, enc, &c.q4b, &m.buf, xs, y, rows, cols),
            3 => encode_q1t_like(c, enc, &c.q1t, &m.buf, xs, y, rows, cols),
            5 => {
                if c.use_mv4 {
                    let gpr = cols / 32;
                    let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
                    let layout = c.q4t_mv8.get_bind_group_layout(0);
                    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layout,
                        entries: &[
                            bind_buf(0, &m.buf),
                            bind_buf(2, y),
                            bind_buf(3, &p_buf),
                            bind_buf(5, xs),
                        ],
                    });
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&c.q4t_mv8);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups((rows as u32).div_ceil(8).min(MAX_WG), 1, 1);
                } else {
                    encode_q1t_like(c, enc, &c.q4t_mv, &m.buf, xs, y, rows, cols)
                }
            }
            6 => {
                if c.use_mv4 {
                    encode_q4tp_mv4(c, enc, &m.buf, xs, y, rows, cols)
                } else {
                    encode_q1t_like(c, enc, &c.q4tp_mv, &m.buf, xs, y, rows, cols)
                }
            }
            _ => encode_f32matvec(c, enc, &m.buf, xs, y, rows, cols),
        }
    };
    // Prep a matvec (pipeline, bind group, workgroups) WITHOUT opening a pass —
    // so several independent ones can share a pass. None = a dtype we don't
    // group (q4t/q1t) → caller falls back to per-op emat. The bind group keeps
    // its uniform buffer alive, so returning it alone is enough.
    let prep = |m: &GMat,
                xs: &wgpu::Buffer,
                y: &wgpu::Buffer,
                rows: usize,
                cols: usize|
     -> Option<(&wgpu::ComputePipeline, wgpu::BindGroup, u32)> {
        match m.kind {
            0 => {
                let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, 0, 0]);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &c.layout,
                    entries: &[
                        bind_buf(0, &m.buf),
                        bind_buf(1, xs),
                        bind_buf(2, m.rs.as_ref().unwrap()),
                        bind_buf(3, y),
                        bind_buf(4, &p_buf),
                    ],
                });
                Some((&c.matvec, bind, (rows as u32).min(MAX_WG)))
            }
            1 => {
                let gpr = cols / 32;
                let p_buf = uniform_u32x4(c, [(gpr / 2) as u32, rows as u32, 0, 0]);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &c.layout_q1,
                    entries: &[
                        bind_buf(0, &m.buf),
                        bind_buf(1, xs),
                        bind_buf(2, y),
                        bind_buf(3, &p_buf),
                    ],
                });
                Some((&c.q1, bind, (rows as u32).div_ceil(8).min(MAX_WG)))
            }
            4 => {
                let p_buf = uniform_u32x4(c, [cols as u32, rows as u32, 0, 0]);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &c.layout_f32,
                    entries: &[
                        bind_buf(0, &m.buf),
                        bind_buf(1, xs),
                        bind_buf(2, y),
                        bind_buf(3, &p_buf),
                    ],
                });
                Some((&c.f32_matvec, bind, (rows as u32).min(MAX_WG)))
            }
            // These arms must exist: without them `prep` returned None for
            // every q4t/q4tp projection, `group_mats` fell back to one
            // compute pass PER matvec and the MoE layer took its per-op
            // branch — a pass costs ~60 us on this Vulkan stack against
            // ~2 ms of arithmetic for the whole MoE block.
            2 | 5 | 6 => {
                let gpr = cols / 32;
                let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
                if m.kind == 2 && c.use_mv4 {
                    let layout = c.q4b_mv8.get_bind_group_layout(0);
                    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layout,
                        entries: &[
                            bind_buf(0, &m.buf),
                            bind_buf(2, y),
                            bind_buf(3, &p_buf),
                            bind_buf(4, &m.buf),
                            bind_buf(5, xs),
                        ],
                    });
                    return Some((&c.q4b_mv8, bind, (rows as u32).div_ceil(8).min(MAX_WG)));
                }
                if m.kind == 5 && c.use_mv4 {
                    // q4t's twin takes the same five bindings as q4tp's.
                    let layout = c.q4t_mv8.get_bind_group_layout(0);
                    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layout,
                        // NO slot 4: q4t assembles weights from u16 halves
                        // (18 B tiles are 2-aligned), so the entry point
                        // never reads the vec4 weight view and its auto
                        // layout does not carry that binding.
                        entries: &[
                            bind_buf(0, &m.buf),
                            bind_buf(2, y),
                            bind_buf(3, &p_buf),
                            bind_buf(5, xs),
                        ],
                    });
                    return Some((&c.q4t_mv8, bind, (rows as u32).div_ceil(8).min(MAX_WG)));
                }
                if m.kind == 6 && c.use_mv4 {
                    let (pipe6, per_wg) = if gpr <= 64 {
                        (&c.q4tp_mv16, 16u32)
                    } else {
                        (&c.q4tp_mv4, 8u32)
                    };
                    let layout = pipe6.get_bind_group_layout(0);
                    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layout,
                        entries: &[
                            bind_buf(0, &m.buf),
                            bind_buf(2, y),
                            bind_buf(3, &p_buf),
                            bind_buf(4, &m.buf),
                            bind_buf(5, xs),
                        ],
                    });
                    return Some((pipe6, bind, (rows as u32).div_ceil(per_wg).min(MAX_WG)));
                }
                let pl = match m.kind {
                    5 => &c.q4t_mv,
                    6 => &c.q4tp_mv,
                    _ => &c.q4b,
                };
                let layout = pl.get_bind_group_layout(0);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &[
                        bind_buf(0, &m.buf),
                        bind_buf(1, xs),
                        bind_buf(2, y),
                        bind_buf(3, &p_buf),
                    ],
                });
                Some((pl, bind, (rows as u32).min(MAX_WG)))
            }
            3 => {
                let gpr = cols / 32;
                let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
                let layout = c.q1t.get_bind_group_layout(0);
                let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &[
                        bind_buf(0, &m.buf),
                        bind_buf(1, xs),
                        bind_buf(2, y),
                        bind_buf(3, &p_buf),
                    ],
                });
                Some((&c.q1t, bind, (rows as u32).min(MAX_WG)))
            }
            _ => None,
        }
    };
    // Emit a set of mutually-INDEPENDENT matvecs. When grouping is on and every
    // one preps, they share a single compute pass (no barrier between them);
    // otherwise each goes through emat as its own pass. Correctness rests on the
    // caller passing only matvecs with no read-after-write among them.
    let group_mats =
        |enc: &mut wgpu::CommandEncoder,
         mats: &[(&GMat, &wgpu::Buffer, &wgpu::Buffer, usize, usize)]| {
            if group {
                let prepped: Vec<_> = mats
                    .iter()
                    .filter_map(|(m, xs, y, r, cc)| prep(m, xs, y, *r, *cc))
                    .collect();
                if prepped.len() == mats.len() {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
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
    // Two projections of the same input under ONE dispatch. `false` = a
    // kind the paired kernel does not cover, caller keeps `group_mats`.
    let pair_mats = |enc: &mut wgpu::CommandEncoder,
                     a: (&GMat, &wgpu::Buffer, usize, usize),
                     b: (&GMat, &wgpu::Buffer, usize, usize),
                     xs: &wgpu::Buffer|
     -> bool {
        let ok = |k: u8| k == 4 || k == 6;
        if !ok(a.0.kind) || !ok(b.0.kind) {
            return false;
        }
        let p = unif(&[
            a.2 as u32,
            a.3 as u32,
            a.0.kind as u32,
            0,
            b.2 as u32,
            b.3 as u32,
            b.0.kind as u32,
            0,
        ]);
        let bind = bg(&c.layout_mv2, &[&a.0.buf, &b.0.buf, xs, a.1, b.1, &p]);
        go(enc, &c.matvec_pair, &bind, ((a.2 + b.2) as u32).min(MAX_WG));
        true
    };
    let mut o1_dbg: Vec<(usize, wgpu::Buffer)> = Vec::new();
    // ── Frame profiler (CMF_GPU_TS=1, single-step): GPU timestamps at pass
    // granularity, aggregated per (stage, layer-kind). The microscope that
    // replaces cost-model guessing.
    let mut ts_n: u32 = 0;
    let mut ts_lbl: Vec<(u8, u8)> = Vec::new();
    let ts_fine = std::env::var("CMF_GPU_TS").as_deref() == Ok("2");
    macro_rules! ts {
        ($enc:expr, $stage:expr, $kind:expr) => {
            if steps == 1 {
                if let Some((qs, _, _)) = &c.ts_query {
                    if ts_n < 255 {
                        $enc.write_timestamp(qs, ts_n);
                        ts_lbl.push(($stage, $kind));
                        ts_n += 1;
                    }
                }
            }
        };
    }
    // Per-dispatch stamps INSIDE a pass (CMF_GPU_TS=2), for the first layer
    // of each kind only — 256 slots cannot carry every dispatch of a frame.
    macro_rules! tsp {
        ($pass:expr, $on:expr, $stage:expr) => {
            if ts_fine && $on && steps == 1 {
                if let Some((qs, _, _)) = &c.ts_query {
                    if ts_n < 255 {
                        $pass.write_timestamp(qs, ts_n);
                        ts_lbl.push(($stage, 9));
                        ts_n += 1;
                    }
                }
            }
        };
    }
    // ── Multi-step prerequisites: the lm_head fold and a q4tp embedding,
    // both resolved up front. Anything missing refuses the WHOLE call so
    // the pipeline can fall back to single-step.
    // "multi" really means: the DEVICE picks the token(s) and the CPU
    // reads ids, not logits. k=1 rides it too — a 4-byte readback against
    // a megabyte of logits.
    let multi = ids_out.is_some();
    let lm_pre = lm_head.and_then(|(gw, rows)| resolve(gw, rows, hidden).map(|m| (m, rows)));
    let emb_pre =
        embed.and_then(|(gw, rows, mult)| resolve(gw, rows, hidden).map(|m| (m, rows, mult)));
    if multi {
        let embed_ok = matches!(&emb_pre, Some((m, _, _)) if m.kind == 6);
        if lm_pre.is_none() || !embed_ok {
            graph_refused("multi-step needs the lm_head fold and a q4tp embedding");
            return false;
        }
    }
    const AM_PARTS: u32 = 512;
    let lm_rows_pre = lm_pre.as_ref().map(|(_, r)| *r).unwrap_or(0);
    let (lbuf_pre, am_pv, am_pi, ids_buf, ids_stage) = if multi {
        let lsize = (lm_rows_pre * 4) as u64;
        (
            Some(GraphScratch::ensure(
                &c.device,
                &mut gs.logits,
                lsize,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "g-logits",
            )),
            Some(GraphScratch::ensure(
                &c.device,
                &mut gs.am_pv,
                (AM_PARTS * 4) as u64,
                wgpu::BufferUsages::STORAGE,
                "g-am-pv",
            )),
            Some(GraphScratch::ensure(
                &c.device,
                &mut gs.am_pi,
                (AM_PARTS * 4) as u64,
                wgpu::BufferUsages::STORAGE,
                "g-am-pi",
            )),
            Some(GraphScratch::ensure(
                &c.device,
                &mut gs.ids,
                (steps * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                "g-ids",
            )),
            Some(GraphScratch::ensure(
                &c.device,
                &mut gs.ids_stage,
                (steps * 4) as u64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                "g-ids-stage",
            )),
        )
    } else {
        (None, None, None, None, None)
    };
    for stp in 0..steps {
        let kv_u = kv_us[stp].clone();
        let at_u = at_us[stp].clone();
        let rope_u = rope_us[stp].clone();
        let position = position + stp;
        // Bootstrap the first layer's attention norm; thereafter each residual is
        // fused with the following norm (add_rmsnorm), saving two dispatches/layer.
        let inw0 = stor(bytemuck::cast_slice(layers[0].input_norm));
        ts!(enc, 0, 0);
        go(
            &mut enc,
            &c.rmsnorm,
            &bg(&c.layout_rmsnorm, &[&h_buf, &inw0, &n1, &rms_u]),
            1,
        );
        for (li, l) in layers.iter().enumerate() {
            if li >= layer_cap {
                break;
            }
            let lw = &lws[li];
            let lkind: u8 = if matches!(lw.attn, LAttn::Full { .. }) {
                1
            } else {
                0
            };
            let pnw = stor(bytemuck::cast_slice(l.post_norm));
            // ── token mixing (attention or GDN) → ob ──
            match (&lw.attn, &l.attn) {
                (
                    LAttn::Full { wq, wk, wv, wo },
                    crate::gpu::GraphAttn::Full {
                        q_norm,
                        k_norm,
                        bias,
                        output_gate,
                        ..
                    },
                ) => {
                    let o1_here = o1.get(li).and_then(|v| v.as_ref());
                    // true = the fused short-context arm already ran the output
                    // gate and the O projection inside its pass.
                    let mut attn_done = false;
                    let qnw = q_norm
                        .map(|q| stor(bytemuck::cast_slice(q)))
                        .unwrap_or_else(|| zeros(hd));
                    let knw = k_norm
                        .map(|k| stor(bytemuck::cast_slice(k)))
                        .unwrap_or_else(|| zeros(hd));
                    let gate_flag = if *output_gate { 1u32 } else { 0 };
                    c.queue.write_buffer(
                        &rope_u,
                        0,
                        bytemuck::cast_slice(&[
                            nh as u32,
                            nkv as u32,
                            hd as u32,
                            rd as u32,
                            position as u32,
                            flags(q_norm.is_some(), k_norm.is_some()) | gate_flag,
                            eps.to_bits(),
                            0,
                        ]),
                    );
                    // Gated wq emits 2·nh·hd (q||gate interleaved per head); the rope
                    // kernel splits it, roping q and passing gate through to `gout`.
                    let qrows = nh * hd * (1 + *output_gate as usize);
                    group_mats(
                        &mut enc,
                        &[
                            (wq, &n1, &qraw, qrows, hidden),
                            (wk, &n1, &kb, nkv * hd, hidden),
                            (wv, &n1, &vb, nkv * hd, hidden),
                        ],
                    );
                    if let Some((bq, bk, bv)) = bias {
                        let (bqb, bkb, bvb) = (
                            stor(bytemuck::cast_slice(bq)),
                            stor(bytemuck::cast_slice(bk)),
                            stor(bytemuck::cast_slice(bv)),
                        );
                        let axq = uniform_u32x4(c, [1.0f32.to_bits(), (nh * hd) as u32, 0, 0]);
                        let axkv = uniform_u32x4(c, [1.0f32.to_bits(), (nkv * hd) as u32, 0, 0]);
                        go(
                            &mut enc,
                            &c.axpy,
                            &bg(&c.layout_axpy, &[&bqb, &qraw, &axq]),
                            ((nh * hd) as u32).div_ceil(256),
                        );
                        go(
                            &mut enc,
                            &c.axpy,
                            &bg(&c.layout_axpy, &[&bkb, &kb, &axkv]),
                            ((nkv * hd) as u32).div_ceil(256),
                        );
                        go(
                            &mut enc,
                            &c.axpy,
                            &bg(&c.layout_axpy, &[&bvb, &vb, &axkv]),
                            ((nkv * hd) as u32).div_ceil(256),
                        );
                    }
                    if let Some(views) = o1_here {
                        // O(1) attention: rope as usual, then the three o1
                        // kernels replace kv_append + attend. State mirrors on
                        // the device once per seal epoch; kv mirrors are not
                        // touched for this layer at all.
                        if o1_ensure(c, kv_id, li, views, o1_epoch).is_none() {
                            graph_refused("o1 state not portable");
                            return false;
                        }
                        let (
                            dmeta,
                            drk,
                            drv,
                            dsk,
                            dsv,
                            dkt,
                            dqt,
                            dmu,
                            dmz,
                            dth,
                            gg,
                            hh_,
                            mm,
                            ww,
                            nns,
                            sc,
                        ) = {
                            let map = c.o1m.lock().unwrap();
                            let d = map.get(&(kv_id, li)).unwrap();
                            (
                                d.meta.clone(),
                                d.ring_k.clone(),
                                d.ring_v.clone(),
                                d.sink_k.clone(),
                                d.sink_v.clone(),
                                d.k_tilde.clone(),
                                d.qt.clone(),
                                d.mu.clone(),
                                d.mz.clone(),
                                d.that.clone(),
                                d.g,
                                d.h,
                                d.m,
                                d.w,
                                d.ns,
                                d.scale,
                            )
                        };
                        let rect_fm = views
                            .first()
                            .and_then(|v| v.heads.first())
                            .is_some_and(|h| h.rect_fm);
                        let o1_u = uniform_u32x8(
                            c,
                            [
                                hh_ as u32,
                                mm as u32,
                                ww as u32,
                                (nns as u32) | (u32::from(rect_fm) << 8),
                                hd as u32,
                                hd as u32,
                                sc.to_bits(),
                                0,
                            ],
                        );
                        let bg_rope = bg(
                            &c.layout_attn_rope,
                            &[&qraw, &kb, &qout, &gout, &qnw, &knw, &invf_b, &rope_u],
                        );
                        let bg_far = bg(
                            &c.layout_o1_far,
                            &[&dmeta, &drk, &drv, &dqt, &dmz, &dth, &o1_u],
                        );
                        let bg_push = bg(&c.layout_o1_push, &[&dmeta, &kb, &vb, &drk, &drv, &o1_u]);
                        let bg_att = bg(
                            &c.layout_o1_attend,
                            &[
                                &dmeta, &qout, &drk, &drv, &dsk, &dsv, &dkt, &dmu, &dmz, &dth,
                                &attn, &o1_u,
                            ],
                        );
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&c.attn_rope);
                        pass.set_bind_group(0, &bg_rope, &[]);
                        pass.dispatch_workgroups((nh + nkv) as u32, 1, 1);
                        pass.set_pipeline(&c.o1_far);
                        pass.set_bind_group(0, &bg_far, &[]);
                        pass.dispatch_workgroups((gg * hh_ * mm) as u32, 1, 1);
                        pass.set_pipeline(&c.o1_push);
                        pass.set_bind_group(0, &bg_push, &[]);
                        pass.dispatch_workgroups(gg as u32, 1, 1);
                        pass.set_pipeline(&c.o1_attend);
                        pass.set_bind_group(0, &bg_att, &[]);
                        pass.dispatch_workgroups((gg * hh_) as u32, 1, 1);
                        drop(pass);
                        if std::env::var("CMF_O1_TRACE").is_ok() {
                            // Debug-only: stage this layer's o1 attention output
                            // for a post-submit dump (the attn buffer itself is
                            // reused by every later layer).
                            let dbgb = c.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("o1-dbg"),
                                size: (nh * hd * 4) as u64,
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            enc.copy_buffer_to_buffer(&attn, 0, &dbgb, 0, (nh * hd * 4) as u64);
                            o1_dbg.push((li, dbgb));
                            let dbgq = c.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("o1-dbg-q"),
                                size: 64,
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            enc.copy_buffer_to_buffer(&qout, 0, &dbgq, 0, 16);
                            enc.copy_buffer_to_buffer(&kb, 0, &dbgq, 16, 16);
                            enc.copy_buffer_to_buffer(&vb, 0, &dbgq, 32, 16);
                            o1_dbg.push((li + 10_000, dbgq));
                        }
                    } else {
                        let (kbuf, vbuf) = kvbufs[li].as_ref().unwrap();
                        // rope + kv_append are independent (both read kb, neither
                        // writes it) — share ONE compute pass to avoid the inter-pass
                        // pipeline flush (~78 μs on NVIDIA Vulkan).
                        let n_ctx = position + 1;
                        // Short context (the decode regime): attend + output gate +
                        // O-projection ride the SAME pass as rope/kv when they prep —
                        // five passes become one, and dispatch order is unchanged.
                        let short_ctx = !(n_ctx > ATTEND_SPLIT_MIN && (hd <= 128 || c.big_attend));
                        if passfuse && short_ctx && !skip_attn {
                            attn_done = true;
                            let (mut ap, al) = attend_pipes(c, hd);
                            let dec_l;
                            let bg_att = if c.attend_dec && hd <= 256 {
                                ap = &c.gqa_attend_dec;
                                // Auto layouts are pipeline-exclusive — the twin's
                                // binding SET matches, its layout object does not.
                                dec_l = c.gqa_attend_dec.get_bind_group_layout(0);
                                bg(&dec_l, &[&qout, kbuf, vbuf, &attn, &at_u])
                            } else {
                                bg(al, &[&qout, kbuf, vbuf, &attn, &at_u])
                            };
                            let gm = if *output_gate {
                                let gm_u = uniform_u32x4(c, [(nh * hd) as u32, 0, 0, 0]);
                                Some(bg(&c.layout_gate_mul, &[&gout, &attn, &gm_u]))
                            } else {
                                None
                            };
                            let wo_prep = prep(wo, &attn, &ob, hidden, nh * hd);
                            {
                                let bg_rope = bg(
                                    &c.layout_attn_rope,
                                    &[&qraw, &kb, &qout, &gout, &qnw, &knw, &invf_b, &rope_u],
                                );
                                let bg_kv = bg(&c.layout_kv, &[&kb, &vb, kbuf, vbuf, &kv_u]);
                                let mut pass =
                                    enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: None,
                                        timestamp_writes: None,
                                    });
                                let fine = li < 4;
                                tsp!(pass, fine, 20); // pass start (after qkv projections)
                                pass.set_pipeline(&c.attn_rope);
                                pass.set_bind_group(0, &bg_rope, &[]);
                                pass.dispatch_workgroups((nh + nkv) as u32, 1, 1);
                                tsp!(pass, fine, 21); // rope
                                pass.set_pipeline(&c.kv_append);
                                pass.set_bind_group(0, &bg_kv, &[]);
                                pass.dispatch_workgroups(((nkv * hd) as u32).div_ceil(256), 1, 1);
                                tsp!(pass, fine, 22); // kv append
                                pass.set_pipeline(ap);
                                pass.set_bind_group(0, &bg_att, &[]);
                                pass.dispatch_workgroups(nh as u32, 1, 1);
                                tsp!(pass, fine, 23); // attend
                                if let Some(bg_gm) = &gm {
                                    pass.set_pipeline(&c.gate_mul);
                                    pass.set_bind_group(0, bg_gm, &[]);
                                    pass.dispatch_workgroups(
                                        ((nh * hd) as u32).div_ceil(256),
                                        1,
                                        1,
                                    );
                                }
                                tsp!(pass, fine, 24); // gate
                                if let Some((wp, wb, ww)) = &wo_prep {
                                    pass.set_pipeline(wp);
                                    pass.set_bind_group(0, wb, &[]);
                                    pass.dispatch_workgroups(*ww, 1, 1);
                                }
                                tsp!(pass, fine, 25); // o-proj
                            }
                            if wo_prep.is_none() {
                                emat(&mut enc, wo, &attn, &ob, hidden, nh * hd);
                            }
                        } else {
                            {
                                let bg_rope = bg(
                                    &c.layout_attn_rope,
                                    &[&qraw, &kb, &qout, &gout, &qnw, &knw, &invf_b, &rope_u],
                                );
                                let bg_kv = bg(&c.layout_kv, &[&kb, &vb, kbuf, vbuf, &kv_u]);
                                let mut pass =
                                    enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: None,
                                        timestamp_writes: None,
                                    });
                                pass.set_pipeline(&c.attn_rope);
                                pass.set_bind_group(0, &bg_rope, &[]);
                                pass.dispatch_workgroups((nh + nkv) as u32, 1, 1);
                                pass.set_pipeline(&c.kv_append);
                                pass.set_bind_group(0, &bg_kv, &[]);
                                pass.dispatch_workgroups(((nkv * hd) as u32).div_ceil(256), 1, 1);
                            }
                            if n_ctx > ATTEND_SPLIT_MIN && (hd <= 128 || c.big_attend) {
                                // Split-K attend: (nh × chunks) part workgroups + a
                                // per-head merge, both in ONE pass (WebGPU orders
                                // dispatches within a pass, so the merge sees the
                                // partials without an inter-pass flush).
                                let nc = cap.div_ceil(ATTEND_CK);
                                let nc_used = n_ctx.div_ceil(ATTEND_CK);
                                let pacc = GraphScratch::ensure(
                                    &c.device,
                                    &mut gs.apacc,
                                    (nh * nc * hd * 4) as u64,
                                    st,
                                    "g-apacc",
                                );
                                let pml = GraphScratch::ensure(
                                    &c.device,
                                    &mut gs.apml,
                                    (nh * nc * 8) as u64,
                                    st,
                                    "g-apml",
                                );
                                let ap_u = unif(&[
                                    nh as u32,
                                    (nh / nkv) as u32,
                                    hd as u32,
                                    cap as u32,
                                    n_ctx as u32,
                                    ATTEND_CK as u32,
                                    nc as u32,
                                    0,
                                ]);
                                let (pp, pl) = attend_part_pipes(c, hd);
                                let bg_part = bg(pl, &[&qout, kbuf, vbuf, &pacc, &pml, &ap_u]);
                                let bg_merge =
                                    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: None,
                                        layout: &c.layout_attend_merge,
                                        entries: &[
                                            bind_buf(3, &pacc),
                                            bind_buf(4, &pml),
                                            bind_buf(5, &ap_u),
                                            bind_buf(6, &attn),
                                        ],
                                    });
                                let mut pass =
                                    enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: None,
                                        timestamp_writes: None,
                                    });
                                pass.set_pipeline(pp);
                                pass.set_bind_group(0, &bg_part, &[]);
                                pass.dispatch_workgroups(nh as u32, nc_used as u32, 1);
                                pass.set_pipeline(&c.attend_merge);
                                pass.set_bind_group(0, &bg_merge, &[]);
                                pass.dispatch_workgroups(nh as u32, 1, 1);
                            } else if !skip_attn {
                                let (ap, al) = attend_pipes(c, hd);
                                go(
                                    &mut enc,
                                    ap,
                                    &bg(al, &[&qout, kbuf, vbuf, &attn, &at_u]),
                                    nh as u32,
                                );
                            }
                        } // fused-vs-split attend arms
                        // attn_out *= sigmoid(gate) before the O projection.
                    }
                    if *output_gate && !attn_done {
                        let gm_u = uniform_u32x4(c, [(nh * hd) as u32, 0, 0, 0]);
                        go(
                            &mut enc,
                            &c.gate_mul,
                            &bg(&c.layout_gate_mul, &[&gout, &attn, &gm_u]),
                            ((nh * hd) as u32).div_ceil(256),
                        );
                    }
                    if !attn_done {
                        emat(&mut enc, wo, &attn, &ob, hidden, nh * hd);
                    }
                }
                (
                    LAttn::Gdn {
                        qkv,
                        z,
                        a,
                        b,
                        out,
                        nv,
                        nk,
                        dk,
                        dv,
                        kk,
                        cdim,
                    },
                    crate::gpu::GraphAttn::Gdn {
                        conv1d,
                        a_log,
                        dt_bias,
                        norm,
                        ..
                    },
                ) => {
                    let (ring, s) = gdnbufs[li].as_ref().unwrap();
                    let taps = stor(bytemuck::cast_slice(conv1d));
                    let alog = stor(bytemuck::cast_slice(a_log));
                    let dtb = stor(bytemuck::cast_slice(dt_bias));
                    let gnorm = stor(bytemuck::cast_slice(norm));
                    let gc_p = uniform_u32x4(c, [*cdim as u32, *kk as u32, 0, 0]);
                    let gd_p = unif(&[
                        *nv as u32,
                        *dk as u32,
                        *dv as u32,
                        (nk * dk) as u32,
                        (nv / nk) as u32,
                        *cdim as u32,
                        eps.to_bits(),
                        0,
                    ]);
                    let bg_conv = bg(&c.layout_gdn_conv, &[&qkv_b, &taps, ring, &cq_b, &gc_p]);
                    let bg_step = bg(
                        &c.layout_gdn,
                        &[
                            &cq_b, &z_b, &a_b, &b_b, &alog, &dtb, &gnorm, s, &gdo_b, &gd_p,
                        ],
                    );
                    // The parallel step/norm entries use SUBSETS of the gdn
                    // binding set, and an auto layout lists only what its entry
                    // reads — each gets its own bind group (lesson of the day).
                    let bg_par = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &c.gdn_step_par.get_bind_group_layout(0),
                        entries: &[
                            bind_buf(0, &cq_b),
                            bind_buf(2, &a_b),
                            bind_buf(3, &b_b),
                            bind_buf(4, &alog),
                            bind_buf(5, &dtb),
                            bind_buf(7, s),
                            bind_buf(8, &gdo_b),
                            bind_buf(9, &gd_p),
                        ],
                    });
                    let bg_snorm = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &c.gdn_step_norm.get_bind_group_layout(0),
                        entries: &[
                            bind_buf(1, &z_b),
                            bind_buf(6, &gnorm),
                            bind_buf(8, &gdo_b),
                            bind_buf(9, &gd_p),
                        ],
                    });
                    let gi_u = uniform_u32x4(c, [*kk as u32, 0, 0, 0]);
                    let (bg_par2, bg_snorm2) = if c.gdn_inline {
                        (
                            Some(c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: &c.gdn_step_par2.get_bind_group_layout(0),
                                entries: &[
                                    bind_buf(2, &a_b),
                                    bind_buf(3, &b_b),
                                    bind_buf(4, &alog),
                                    bind_buf(5, &dtb),
                                    bind_buf(7, s),
                                    bind_buf(8, &gdo_b),
                                    bind_buf(9, &gd_p),
                                    bind_buf(10, &qkv_b),
                                    bind_buf(11, ring),
                                    bind_buf(12, &taps),
                                    bind_buf(13, &gi_u),
                                ],
                            })),
                            Some(c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: &c.gdn_step_norm2.get_bind_group_layout(0),
                                entries: &[
                                    bind_buf(1, &z_b),
                                    bind_buf(6, &gnorm),
                                    bind_buf(8, &gdo_b),
                                    bind_buf(9, &gd_p),
                                    bind_buf(10, &qkv_b),
                                    bind_buf(11, ring),
                                    bind_buf(13, &gi_u),
                                ],
                            })),
                        )
                    } else {
                        (None, None)
                    };
                    // The whole GDN chain in ONE compute pass: projections →
                    // conv → step → out_proj. Each stage reads the previous
                    // stage's output, which is exactly what a pass guarantees
                    // (dispatches inside it are ordered, with memory visible
                    // between them — the same rule the fused SiLU FFN relies on).
                    //
                    // A pass, not a dispatch, is the unit that costs here:
                    // teaching `prep` the q4tp kind collapsed this layer's four
                    // projection passes into one and bought 1.46 ms a token
                    // across 30 layers — ~16 us per pass. Four passes become one.
                    let projs = [
                        prep(qkv, &n1, &qkv_b, *cdim, hidden),
                        prep(z, &n1, &z_b, nv * dv, hidden),
                        prep(a, &n1, &a_b, *nv, hidden),
                        prep(b, &n1, &b_b, *nv, hidden),
                    ];
                    let outp = prep(out, &gdo_b, &ob, hidden, nv * dv);
                    if projs.iter().all(|p| p.is_some()) && outp.is_some() {
                        let _ = (skip_proj, skip_outp);
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        if !skip_proj {
                            for p in projs.iter().flatten() {
                                pass.set_pipeline(p.0);
                                pass.set_bind_group(0, &p.1, &[]);
                                pass.dispatch_workgroups(p.2, 1, 1);
                            }
                        }
                        let fine = li == 0;
                        tsp!(pass, fine, 10); // after projections
                        if !skip_gdn {
                            if c.gdn_par && c.gdn_inline {
                                pass.set_pipeline(&c.gdn_step_par2);
                                pass.set_bind_group(0, bg_par2.as_ref().unwrap(), &[]);
                                pass.dispatch_workgroups(*nv as u32, *dv as u32, 1);
                                tsp!(pass, fine, 12); // step_par (conv inline)
                                pass.set_pipeline(&c.gdn_step_norm2);
                                pass.set_bind_group(0, bg_snorm2.as_ref().unwrap(), &[]);
                                pass.dispatch_workgroups(*nv as u32, 1, 1);
                                tsp!(pass, fine, 13); // step_norm (+ring shift)
                            } else {
                                pass.set_pipeline(&c.gdn_conv);
                                pass.set_bind_group(0, &bg_conv, &[]);
                                pass.dispatch_workgroups((*cdim as u32).div_ceil(256), 1, 1);
                                tsp!(pass, fine, 11); // conv
                                if c.gdn_par {
                                    pass.set_pipeline(&c.gdn_step_par);
                                    pass.set_bind_group(0, &bg_par, &[]);
                                    pass.dispatch_workgroups(
                                        *nv as u32,
                                        (*dv as u32).div_ceil(4),
                                        1,
                                    );
                                    tsp!(pass, fine, 12); // step_par
                                    pass.set_pipeline(&c.gdn_step_norm);
                                    pass.set_bind_group(0, &bg_snorm, &[]);
                                    pass.dispatch_workgroups(*nv as u32, 1, 1);
                                    tsp!(pass, fine, 13); // step_norm
                                } else {
                                    pass.set_pipeline(&c.gdn_step);
                                    pass.set_bind_group(0, &bg_step, &[]);
                                    pass.dispatch_workgroups(*nv as u32, 1, 1);
                                    tsp!(pass, fine, 12);
                                }
                            } // gdn_inline arms
                        }
                        if !skip_outp {
                            let o = outp.as_ref().unwrap();
                            pass.set_pipeline(o.0);
                            pass.set_bind_group(0, &o.1, &[]);
                            pass.dispatch_workgroups(o.2, 1, 1);
                            tsp!(pass, fine, 14); // out-proj
                        }
                    } else {
                        group_mats(
                            &mut enc,
                            &[
                                (qkv, &n1, &qkv_b, *cdim, hidden),
                                (z, &n1, &z_b, nv * dv, hidden),
                                (a, &n1, &a_b, *nv, hidden),
                                (b, &n1, &b_b, *nv, hidden),
                            ],
                        );
                        go(
                            &mut enc,
                            &c.gdn_conv,
                            &bg_conv,
                            (*cdim as u32).div_ceil(256),
                        );
                        if c.gdn_par {
                            {
                                let mut pass =
                                    enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: None,
                                        timestamp_writes: None,
                                    });
                                pass.set_pipeline(&c.gdn_step_par);
                                pass.set_bind_group(0, &bg_par, &[]);
                                pass.dispatch_workgroups(*nv as u32, (*dv as u32).div_ceil(4), 1);
                                pass.set_pipeline(&c.gdn_step_norm);
                                pass.set_bind_group(0, &bg_snorm, &[]);
                                pass.dispatch_workgroups(*nv as u32, 1, 1);
                            }
                        } else {
                            go(&mut enc, &c.gdn_step, &bg_step, *nv as u32);
                        }
                        emat(&mut enc, out, &gdo_b, &ob, hidden, nv * dv);
                    }
                }
                _ => return false,
            }
            ts!(enc, 1, lkind);
            // token-mix residual + FFN-norm fused: h += ob, n1 = rms(h, post_norm).
            // It used to open its own compute pass. On this Vulkan stack a PASS
            // BOUNDARY is the expensive part (the MoE block is built entirely
            // around that fact), and this one sits between two passes that are
            // strictly serial anyway — so hand it to the FFN pass as a prologue
            // and let within-pass serialization do the same job for free.
            // CMF_PASSFUSE=0 puts it back in its own pass.
            let mut ffn_pre: Option<(&wgpu::ComputePipeline, wgpu::BindGroup, u32)> = None;
            if !skip_norm {
                let nbg = bg(&c.layout_add_rmsnorm, &[&h_buf, &ob, &pnw, &n1, &rms_u]);
                if passfuse {
                    ffn_pre = Some((&c.add_rmsnorm, nbg, 1));
                } else {
                    go(&mut enc, &c.add_rmsnorm, &nbg, 1);
                }
            }
            // …and the layer's TAIL (FFN residual + the next layer's input norm)
            // rides out on the same pass. It reads `ob`, which that pass's last
            // dispatch writes — the same within-pass ordering the block above
            // relies on. With both ends folded in, a layer is TWO passes
            // (token-mix, then FFN) instead of four.
            let simple_tail = passfuse && !loop_norm_at.contains(&li);
            let mut ffn_post: Option<(&wgpu::ComputePipeline, wgpu::BindGroup, u32)> = None;
            let mut tail_done = false;
            if simple_tail {
                ffn_post = Some(if li + 1 < layers.len() {
                    let inw_next = stor(bytemuck::cast_slice(layers[li + 1].input_norm));
                    (
                        &c.add_rmsnorm,
                        bg(
                            &c.layout_add_rmsnorm,
                            &[&h_buf, &ob, &inw_next, &n1, &rms_u],
                        ),
                        1,
                    )
                } else {
                    (
                        &c.axpy,
                        bg(&c.layout_axpy, &[&ob, &h_buf, &ax_u]),
                        (hidden as u32).div_ceil(256),
                    )
                });
            }
            // SiLU FFN: gate+up matvecs + silu fused in ONE compute pass
            // (dispatches within a pass are serialized — silu safely reads gate/up output).
            match &lw.ffn {
                LFfn::Dense { gate, up, down } => {
                    let pg = prep(gate, &n1, &gbuf, inter, hidden);
                    let pu = prep(up, &n1, &ubuf, inter, hidden);
                    if let (Some((pgp, bg_g, wg)), Some((pup, bg_u, wu))) = (pg, pu) {
                        let bg_silu =
                            bg(&c.layout_silu, &[&gbuf, &ubuf, &dummy_hd, &abuf, &silu_u]);
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        if let Some((p, b, w)) = &ffn_pre {
                            pass.set_pipeline(p);
                            pass.set_bind_group(0, b, &[]);
                            pass.dispatch_workgroups(*w, 1, 1);
                        }
                        pass.set_pipeline(pgp);
                        pass.set_bind_group(0, &bg_g, &[]);
                        pass.dispatch_workgroups(wg, 1, 1);
                        pass.set_pipeline(pup);
                        pass.set_bind_group(0, &bg_u, &[]);
                        pass.dispatch_workgroups(wu, 1, 1);
                        pass.set_pipeline(&c.silu);
                        pass.set_bind_group(0, &bg_silu, &[]);
                        pass.dispatch_workgroups((inter as u32).div_ceil(256), 1, 1);
                        // NOTE: the dense arm still emits `down` outside this pass
                        // (see emat below), so the tail cannot ride here.
                    } else {
                        group_mats(
                            &mut enc,
                            &[
                                (gate, &n1, &gbuf, inter, hidden),
                                (up, &n1, &ubuf, inter, hidden),
                            ],
                        );
                        go(
                            &mut enc,
                            &c.silu,
                            &bg(&c.layout_silu, &[&gbuf, &ubuf, &dummy_hd, &abuf, &silu_u]),
                            (inter as u32).div_ceil(256),
                        );
                    }
                    emat(&mut enc, down, &abuf, &ob, hidden, inter);
                }
                LFfn::Moe {
                    router,
                    sgate,
                    gate_all,
                    up_all,
                    down_all,
                    n_exp,
                    top_k,
                    inter: mi,
                    norm_topk,
                    q4tp,
                    gu_q2,
                } => {
                    // The WHOLE MoE FFN — router + shared-gate matvecs, top-k
                    // select, fused gate+up+SiLU over the selected experts, and
                    // the weighted down accumulation into ob — rides in ONE
                    // compute pass: dispatches within a pass serialize with
                    // memory visibility (same guarantee the dense fused FFN
                    // uses), and the inter-pass pipeline flush (~78 µs on
                    // NVIDIA Vulkan) is what dominates a 40-layer decode.
                    let (mlogit, mslog, msel, mwt, mact) = moe_bufs.as_ref().unwrap();
                    let slots = *top_k + 1;
                    // sg_kind = 4 tells the select kernel to compute the shared
                    // gate itself; then the sgate matvec below is not encoded.
                    let sg_fold = sgate.kind == 4;
                    // Cached uniform: `unif` mints a fresh buffer per call, and one
                    // per MoE layer per token exhausted the device. hidden and the
                    // fold flag share the spare word.
                    let sel_u = uniform_u32x4(
                        c,
                        [
                            *n_exp as u32,
                            *top_k as u32,
                            *norm_topk as u32,
                            (hidden as u32) << 8 | u32::from(sg_fold) * 4,
                        ],
                    );
                    // Per-expert stride in u16 units: q4t is 9 per group flat,
                    // q4tp adds the row params and code planes on top of 8.
                    let mat16 = |rows: usize, cols: usize| -> u32 {
                        let n = if *q4tp {
                            cortiq_core::quant::expected_nbytes(
                                cortiq_core::TensorDtype::Q4TiledP,
                                &[rows, cols],
                            )
                            .unwrap_or(0)
                        } else {
                            rows * (cols / 32) * 18
                        };
                        (n / 2) as u32
                    };
                    // gate/up may be a HALF-WIDTH plane (q2tp experts against a
                    // q4tp down), so its per-expert stride is its own.
                    let gu_mat16 = if *gu_q2 {
                        (cortiq_core::quant::expected_nbytes(
                            cortiq_core::TensorDtype::Q2TiledP,
                            &[*mi, hidden],
                        )
                        .unwrap_or(0)
                            / 2) as u32
                    } else {
                        mat16(*mi, hidden)
                    };
                    // Eight words now: the fifth is the swiglu limit, zero
                    // for every architecture but DeepSeek-V4.
                    let gu_u = uniform_u32x8(
                        c,
                        [(hidden / 32) as u32, *mi as u32, slots as u32, gu_mat16, 0, 0, 0, 0],
                    );
                    let dn_u = uniform_u32x4(
                        c,
                        [
                            (*mi / 32) as u32,
                            hidden as u32,
                            slots as u32,
                            mat16(hidden, *mi),
                        ],
                    );
                    let (p_gu, p_dn, l_gu, l_dn) = if *gu_q2 {
                        (
                            &c.moe_gate_up_q2tp,
                            &c.moe_down_q4tp,
                            &c.layout_moe_gu_q2tp,
                            &c.layout_moe_dn_q4tp,
                        )
                    } else if *q4tp {
                        (
                            &c.moe_gate_up_q4tp,
                            &c.moe_down_q4tp,
                            &c.layout_moe_gu_q4tp,
                            &c.layout_moe_dn_q4tp,
                        )
                    } else {
                        (
                            &c.moe_gate_up,
                            &c.moe_down,
                            &c.layout_moe_gu,
                            &c.layout_moe_dn,
                        )
                    };
                    let bg_sel = bg(
                        &c.layout_moe_sel,
                        &[mlogit, mslog, msel, mwt, &sel_u, &sgate.buf, &n1],
                    );
                    let bg_sel_sg = c.moe_select_sg.as_ref().map(|p| {
                        let l = p.get_bind_group_layout(0);
                        bg(&l, &[mlogit, mslog, msel, mwt, &sel_u, &sgate.buf, &n1])
                    });
                    let bg_gu = bg(l_gu, &[gate_all, up_all, &n1, msel, mact, &gu_u]);
                    let bg_dn = bg(l_dn, &[down_all, mact, msel, mwt, &ob, &dn_u]);
                    let pr = prep(router, &n1, mlogit, *n_exp, hidden);
                    let ps = prep(sgate, &n1, mslog, 1, hidden);
                    let mut continue_moe_std = true;
                    // Fold-select (q2tp + folded shared gate): router feeds the
                    // gu/down twins DIRECTLY — the select hop and the sgate
                    // matvec disappear from the layer's dependency chain.
                    let fold = c.foldsel && *gu_q2 && sg_fold && !skip_router;
                    if fold {
                        if let Some((prp, bgr, wr)) = prep(router, &n1, mlogit, *n_exp, hidden) {
                            let mgf_u = uniform_u32x4(c, [*n_exp as u32, 0, 0, 0]);
                            let l_guf = c.moe_gate_up_q2tp_f.get_bind_group_layout(0);
                            let bg_guf = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: &l_guf,
                                entries: &[
                                    bind_buf(0, gate_all),
                                    bind_buf(1, up_all),
                                    bind_buf(2, &n1),
                                    bind_buf(3, mlogit),
                                    bind_buf(4, mact),
                                    bind_buf(5, &gu_u),
                                    bind_buf(7, &mgf_u),
                                ],
                            });
                            let mdf_u = uniform_u32x4(
                                c,
                                [
                                    *n_exp as u32,
                                    *top_k as u32,
                                    *norm_topk as u32,
                                    (hidden as u32) << 8 | 4,
                                ],
                            );
                            let l_dnf = c.moe_down_q4tp_f.get_bind_group_layout(0);
                            let bg_dnf = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: &l_dnf,
                                entries: &[
                                    bind_buf(0, down_all),
                                    bind_buf(1, mact),
                                    bind_buf(2, mlogit),
                                    bind_buf(3, &sgate.buf),
                                    bind_buf(4, &ob),
                                    bind_buf(5, &dn_u),
                                    bind_buf(6, &n1),
                                    bind_buf(7, &mdf_u),
                                ],
                            });
                            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: None,
                                timestamp_writes: None,
                            });
                            if let Some((p, b, w)) = &ffn_pre {
                                pass.set_pipeline(p);
                                pass.set_bind_group(0, b, &[]);
                                pass.dispatch_workgroups(*w, 1, 1);
                            }
                            pass.set_pipeline(prp);
                            pass.set_bind_group(0, &bgr, &[]);
                            pass.dispatch_workgroups(wr, 1, 1);
                            if !skip_moe {
                                pass.set_pipeline(&c.moe_gate_up_q2tp_f);
                                pass.set_bind_group(0, &bg_guf, &[]);
                                pass.dispatch_workgroups(*mi as u32, slots as u32, 1);
                                pass.set_pipeline(&c.moe_down_q4tp_f);
                                pass.set_bind_group(0, &bg_dnf, &[]);
                                pass.dispatch_workgroups(hidden as u32, 1, 1);
                                if let Some((p, b, w)) = &ffn_post {
                                    pass.set_pipeline(p);
                                    pass.set_bind_group(0, b, &[]);
                                    pass.dispatch_workgroups(*w, 1, 1);
                                    tail_done = true;
                                }
                            }
                            drop(pass);
                            enc.copy_buffer_to_buffer(&n1, 0, &h_buf, 0, 0);
                            // (zero-length copy: keeps the borrow checker shape
                            // identical to the non-fold arm; no-op on device)
                            continue_moe_std = false;
                        }
                    }
                    if continue_moe_std {
                        if let (Some((prp, bgr, wr)), Some((psp, bgs, ws))) = (pr, ps) {
                            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: None,
                                timestamp_writes: None,
                            });
                            if let Some((p, b, w)) = &ffn_pre {
                                pass.set_pipeline(p);
                                pass.set_bind_group(0, b, &[]);
                                pass.dispatch_workgroups(*w, 1, 1);
                            }
                            let fine = li == 0;
                            tsp!(pass, fine, 30); // pass start (after prologue norm)
                            if !skip_router {
                                pass.set_pipeline(prp);
                                pass.set_bind_group(0, &bgr, &[]);
                                pass.dispatch_workgroups(wr, 1, 1);
                            }
                            tsp!(pass, fine, 31); // router
                            if !sg_fold {
                                pass.set_pipeline(psp);
                                pass.set_bind_group(0, &bgs, &[]);
                                pass.dispatch_workgroups(ws, 1, 1);
                            }
                            if let Some(sgp) = &c.moe_select_sg {
                                // Same binding ORDER as the tree kernel's bg_sel —
                                // but its OWN layout (auto layouts are exclusive).
                                pass.set_pipeline(sgp);
                                pass.set_bind_group(0, bg_sel_sg.as_ref().unwrap(), &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            } else {
                                pass.set_pipeline(&c.moe_select);
                                pass.set_bind_group(0, &bg_sel, &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            tsp!(pass, fine, 32); // select
                            if !skip_moe {
                                pass.set_pipeline(p_gu);
                                pass.set_bind_group(0, &bg_gu, &[]);
                                pass.dispatch_workgroups(*mi as u32, slots as u32, 1);
                                tsp!(pass, fine, 33); // gate/up experts
                                pass.set_pipeline(p_dn);
                                pass.set_bind_group(0, &bg_dn, &[]);
                                pass.dispatch_workgroups(hidden as u32, 1, 1);
                                tsp!(pass, fine, 34); // down experts
                                if let Some((p, b, w)) = &ffn_post {
                                    pass.set_pipeline(p);
                                    pass.set_bind_group(0, b, &[]);
                                    pass.dispatch_workgroups(*w, 1, 1);
                                    tail_done = true;
                                }
                            }
                        } else {
                            // Un-preppable router dtype: per-op passes (correct, rare).
                            group_mats(
                                &mut enc,
                                &[
                                    (router, &n1, mlogit, *n_exp, hidden),
                                    (sgate, &n1, mslog, 1, hidden),
                                ],
                            );
                            go(&mut enc, &c.moe_select, &bg_sel, 1);
                            {
                                let mut pass =
                                    enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                        label: None,
                                        timestamp_writes: None,
                                    });
                                pass.set_pipeline(p_gu);
                                pass.set_bind_group(0, &bg_gu, &[]);
                                pass.dispatch_workgroups(*mi as u32, slots as u32, 1);
                            }
                            go(&mut enc, p_dn, &bg_dn, hidden as u32);
                        }
                    } // continue_moe_std
                }
            }
            // FFN-residual + next layer's attn-norm fused (plain residual on the last).
            // At loop boundaries (Looped Transformer), insert final_norm between the
            // residual and the next iteration's input norm.
            ts!(enc, 2, lkind);
            if tail_done {
                // already emitted at the end of the FFN pass
            } else if li + 1 < layers.len() {
                if loop_norm_at.contains(&li) {
                    // h += ob; n1 = rms(h, final_norm); copy n1→h; n1 = rms(h, next_input_norm)
                    let fnw = stor(bytemuck::cast_slice(final_norm));
                    let inw_next = stor(bytemuck::cast_slice(layers[li + 1].input_norm));
                    go(
                        &mut enc,
                        &c.add_rmsnorm,
                        &bg(&c.layout_add_rmsnorm, &[&h_buf, &ob, &fnw, &n1, &rms_u]),
                        1,
                    );
                    enc.copy_buffer_to_buffer(&n1, 0, &h_buf, 0, (hidden * 4) as u64);
                    go(
                        &mut enc,
                        &c.rmsnorm,
                        &bg(&c.layout_rmsnorm, &[&h_buf, &inw_next, &n1, &rms_u]),
                        1,
                    );
                } else {
                    let inw_next = stor(bytemuck::cast_slice(layers[li + 1].input_norm));
                    go(
                        &mut enc,
                        &c.add_rmsnorm,
                        &bg(
                            &c.layout_add_rmsnorm,
                            &[&h_buf, &ob, &inw_next, &n1, &rms_u],
                        ),
                        1,
                    );
                }
            } else {
                go(
                    &mut enc,
                    &c.axpy,
                    &bg(&c.layout_axpy, &[&ob, &h_buf, &ax_u]),
                    (hidden as u32).div_ceil(256),
                );
            }
        }
        // ── Multi-step tail: final norm + lm_head + on-device argmax; the
        // winner's embedding becomes the next step's h. All inside the SAME
        // encoder — one submit carries every step.
        if multi {
            let (lm, lrows) = lm_pre.as_ref().unwrap();
            let lrows = *lrows;
            let lbuf = lbuf_pre.as_ref().unwrap();
            let fnw = stor(bytemuck::cast_slice(final_norm));
            go(
                &mut enc,
                &c.rmsnorm,
                &bg(&c.layout_rmsnorm, &[&h_buf, &fnw, &n1, &rms_u]),
                1,
            );
            emat(&mut enc, lm, &n1, lbuf, lrows, hidden);
            let am_u = uniform_u32x4(c, [lrows as u32, AM_PARTS, stp as u32, 0]);
            let l_ap = c.argmax_part.get_bind_group_layout(0);
            let bg_ap = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &l_ap,
                entries: &[
                    bind_buf(0, lbuf),
                    bind_buf(1, am_pv.as_ref().unwrap()),
                    bind_buf(2, am_pi.as_ref().unwrap()),
                    bind_buf(3, &am_u),
                ],
            });
            go(&mut enc, &c.argmax_part, &bg_ap, AM_PARTS);
            let l_af = c.argmax_final.get_bind_group_layout(0);
            let bg_af = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &l_af,
                entries: &[
                    bind_buf(0, am_pv.as_ref().unwrap()),
                    bind_buf(1, am_pi.as_ref().unwrap()),
                    bind_buf(2, ids_buf.as_ref().unwrap()),
                    bind_buf(3, &am_u),
                ],
            });
            go(&mut enc, &c.argmax_final, &bg_af, 1);
            if stp + 1 < steps {
                let (em, e_rows, mult) = emb_pre.as_ref().unwrap();
                let eg_u = uniform_u32x8(
                    c,
                    [
                        hidden as u32,
                        (hidden / 32) as u32,
                        *e_rows as u32,
                        stp as u32,
                        mult.to_bits(),
                        0,
                        0,
                        0,
                    ],
                );
                let l_eg = c.embed_gather_q4tp.get_bind_group_layout(0);
                let bg_eg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &l_eg,
                    entries: &[
                        bind_buf(0, &em.buf),
                        bind_buf(1, ids_buf.as_ref().unwrap()),
                        bind_buf(2, &h_buf),
                        bind_buf(3, &eg_u),
                    ],
                });
                go(
                    &mut enc,
                    &c.embed_gather_q4tp,
                    &bg_eg,
                    (hidden as u32).div_ceil(256),
                );
            }
        }
    } // for stp (multi-step frames)
    let t_enc = t_enc0.elapsed().as_secs_f64() * 1000.0;
    let t_sub0 = std::time::Instant::now();
    // Return the step-slot uniforms to the scratch pool.
    gs.kv_us = kv_us;
    gs.at_us = at_us;
    gs.rope_us = rope_us;
    // ── Multi-step exit: one submit, one k×u32 readback, no logits. ──
    if multi {
        let ids_b = ids_buf.as_ref().unwrap();
        let stage = ids_stage.as_ref().unwrap();
        let sz = (steps * 4) as u64;
        enc.copy_buffer_to_buffer(ids_b, 0, stage, 0, sz);
        c.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        stage.map_async(wgpu::MapMode::Read, ..sz, move |r| {
            let _ = tx.send(r);
        });
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        let ok = rx.recv().map(|r| r.is_ok()).unwrap_or(false);
        if ok {
            let raw = stage.get_mapped_range(..sz).unwrap();
            let ids: &[u32] = bytemuck::cast_slice(&raw);
            if let Some(out) = ids_out {
                out.clear();
                out.extend_from_slice(ids);
            }
            drop(raw);
        }
        stage.unmap();
        if ok {
            let mut kvm = c.attn_kv.lock().unwrap();
            for li in 0..layers.len() {
                if let Some(m) = kvm.get_mut(&(kv_id, li)) {
                    m.synced = position + steps;
                }
            }
            let mut gsm = c.gdn_state.lock().unwrap();
            let _ = &mut gsm; // states advanced on-device; nothing to sync
        }
        drop(gs);
        if prof {
            let setup = t_enc0.duration_since(t_start).as_secs_f64() * 1000.0;
            eprintln!(
                "token-graph[x{steps}]: setup {setup:.2} ms | encode {t_enc:.2} ms | submit+ids {:.2} ms",
                t_sub0.elapsed().as_secs_f64() * 1000.0
            );
        }
        return ok;
    }
    // h_buf now holds the final hidden. Either ride final-norm + lm_head and
    // read back logits, or (no lm / unresolved weight) read back the hidden.
    let lm_resolved = lm_head.and_then(|(gw, rows)| resolve(gw, rows, hidden).map(|m| (m, rows)));
    let ok = if let Some((lm, lrows)) = lm_resolved {
        let fnw = stor(bytemuck::cast_slice(final_norm));
        go(
            &mut enc,
            &c.rmsnorm,
            &bg(&c.layout_rmsnorm, &[&h_buf, &fnw, &n1, &rms_u]),
            1,
        );
        let lsize = (lrows * 4) as u64;
        let lbuf = GraphScratch::ensure(
            &c.device,
            &mut gs.logits,
            lsize,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "g-logits",
        );
        emat(&mut enc, &lm, &n1, &lbuf, lrows, hidden);
        ts!(enc, 3, 0);
        if let Some((qs, resolve, tstage)) = &c.ts_query {
            if steps == 1 && ts_n > 0 {
                enc.resolve_query_set(qs, 0..ts_n, resolve, 0);
                enc.copy_buffer_to_buffer(resolve, 0, tstage, 0, ts_n as u64 * 8);
            }
        }
        logits.resize(lrows, 0.0);
        let stage = GraphScratch::ensure(
            &c.device,
            &mut gs.stage,
            lsize,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "g-stage",
        );
        let r = readback(c, enc, &lbuf, &stage, lsize, &mut logits[..lrows]);
        drop(gs);
        r
    } else {
        let size = (hidden * 4) as u64;
        let stage = GraphScratch::ensure(
            &c.device,
            &mut gs.stage,
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "g-stage",
        );
        let r = readback(c, enc, &h_buf, &stage, size, &mut h[..hidden]);
        drop(gs);
        r
    };
    if ok && ts_n > 1 {
        if let Some((_, _, tstage)) = &c.ts_query {
            let (tx, rx) = std::sync::mpsc::channel();
            tstage.map_async(wgpu::MapMode::Read, ..(ts_n as u64 * 8), move |r| {
                let _ = tx.send(r);
            });
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            if rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
                let raw = tstage.get_mapped_range(..(ts_n as u64 * 8)).unwrap();
                let t: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&raw).to_vec();
                drop(raw);
                // Attribute each delta to the LATER stamp's (stage, kind).
                let mut agg = std::collections::BTreeMap::<(u8, u8), (f64, u32)>::new();
                for i in 1..ts_n as usize {
                    let dt = t[i].saturating_sub(t[i - 1]) as f64 * c.ts_period as f64 / 1000.0;
                    let e = agg.entry(ts_lbl[i]).or_insert((0.0, 0));
                    e.0 += dt;
                    e.1 += 1;
                }
                let name = |k: (u8, u8)| match k {
                    (1, 0) => "gdn-mix",
                    (1, 1) => "attn-mix",
                    (2, 0) => "ffn@gdn",
                    (2, 1) => "ffn@attn",
                    (3, _) => "tail(norm+lm)",
                    (10, _) => "|gdn:proj",
                    (11, _) => "|gdn:conv",
                    (12, _) => "|gdn:step",
                    (13, _) => "|gdn:snorm",
                    (14, _) => "|gdn:outp",
                    (20, _) => "|attn:qkv",
                    (21, _) => "|attn:rope",
                    (22, _) => "|attn:kv",
                    (23, _) => "|attn:attend",
                    (24, _) => "|attn:gate",
                    (25, _) => "|attn:wo",
                    (30, _) => "|moe:pre",
                    (31, _) => "|moe:router",
                    (32, _) => "|moe:select",
                    (33, _) => "|moe:gu",
                    (34, _) => "|moe:dn",
                    _ => "start",
                };
                let mut line = String::from("gpu-ts:");
                let total: f64 = agg.values().map(|v| v.0).sum();
                for (k, (us, n)) in &agg {
                    line.push_str(&format!(" {}={:.0}us/{}", name(*k), us, n));
                }
                line.push_str(&format!(" | total {:.2} ms", total / 1000.0));
                eprintln!("{line}");
            }
            tstage.unmap();
        }
    }
    if ok {
        for (li, b) in &o1_dbg {
            let (tx, rx) = std::sync::mpsc::channel();
            b.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r);
            });
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            if rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
                let raw = b.get_mapped_range(..).unwrap();
                let all: &[f32] = bytemuck::cast_slice(&raw);
                let v: Vec<f32> = all[..all.len().min(16)].to_vec();
                drop(raw);
                if *li >= 10_000 {
                    eprintln!(
                        "o1-trace L{} gpu q[..4]={:?} k[..4]={:?} v[..4]={:?}",
                        li - 10_000,
                        &v[..4],
                        &v[4..8],
                        &v[8..12]
                    );
                } else {
                    eprintln!("o1-trace L{li} gpu attn[..8] = {v:?}");
                }
            }
        }
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
        eprintln!(
            "token-graph: setup {setup:.2} ms | encode {t_enc:.2} ms | submit+readback {:.2} ms",
            t_sub0.elapsed().as_secs_f64() * 1000.0
        );
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
    let Some(c) = ctx() else {
        bgraph_refused("no ctx");
        return false;
    };
    if k == 0 || positions.len() != k {
        bgraph_refused("k/positions mismatch");
        return false;
    }
    let pos0 = positions[0];
    if pos0 + k > cap || hd % 4 != 0 || hd > c.hd_cap {
        bgraph_refused("pos+k past cap, or head_dim not %4 / over hd_cap");
        return false; // vec4 K/V reads; hd_cap = workgroup-storage limit
    }
    struct GMat {
        buf: wgpu::Buffer,
        rs: Option<wgpu::Buffer>,
        kind: u8,
    }
    enum LAttn {
        Full {
            wq: GMat,
            wk: GMat,
            wv: GMat,
            wo: GMat,
        },
        Gdn {
            qkv: GMat,
            z: GMat,
            a: GMat,
            b: GMat,
            out: GMat,
            nv: usize,
            nk: usize,
            dk: usize,
            dv: usize,
            kk: usize,
            cdim: usize,
        },
    }
    /// Батчевый FFN слоя. MoE маршрутизируется ПО ТОКЕНАМ, поэтому его
    /// эксперты кодируются в цикле внутри того же submit'а, тогда как
    /// attention и проекции остаются батчевыми GEMM'ами. Раньше здесь
    /// допускался только Dense, и любая MoE-модель уходила на путь
    /// «одна позиция за submit»: префилл 33 tok/s против 54 на декоде,
    /// то есть промпт обрабатывался медленнее, чем генерация.
    enum BFfn {
        Dense {
            gate: GMat,
            up: GMat,
            down: GMat,
        },
        Moe {
            router: GMat,
            sgate: GMat,
            gate_all: wgpu::Buffer,
            up_all: wgpu::Buffer,
            down_all: wgpu::Buffer,
            n_exp: usize,
            top_k: usize,
            inter: usize,
            norm_topk: bool,
            q4tp: bool,
        },
    }
    struct LW {
        attn: LAttn,
        ffn: BFfn,
    }
    let resolve = |gw: &crate::gpu::GraphW, rows: usize, cols: usize| -> Option<GMat> {
        match gw.kind {
            0 => {
                if gw.row_scale.len() < rows {
                    return None;
                }
                let b = tensor_weight(c, model, gw.idx, rows, cols)?;
                let rsb = c.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bg-rs"),
                    size: (rows * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                c.queue
                    .write_buffer(&rsb, 0, bytemuck::cast_slice(&gw.row_scale[..rows]));
                Some(GMat {
                    buf: b,
                    rs: Some(rsb),
                    kind: 0,
                })
            }
            1 => {
                let (b, r, cc) = q1_weight(c, model, gw.idx)?;
                if r != rows || cc != cols {
                    return None;
                }
                Some(GMat {
                    buf: b,
                    rs: None,
                    kind: 1,
                })
            }
            4 => {
                if gw.data.len() < rows * cols {
                    return None;
                }
                let b = c.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bg-f32w"),
                    size: (rows * cols * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                c.queue
                    .write_buffer(&b, 0, bytemuck::cast_slice(&gw.data[..rows * cols]));
                Some(GMat {
                    buf: b,
                    rs: None,
                    kind: 4,
                })
            }
            // q4_tiled and q4tp: same buffer shape, the kernel differs.
            // Leaving these out is what kept every q4t/q4tp model off the
            // batched path — including its GDN projections, which is where
            // the refusal actually landed.
            k @ (5 | 6) => {
                let (b, r, cc) = tile_weight(c, model, gw.idx)?;
                if r != rows || cc != cols {
                    return None;
                }
                Some(GMat {
                    buf: b,
                    rs: None,
                    kind: k,
                })
            }
            _ => None, // q1t not batched here → CPU/per-position path
        }
    };
    // GEMM-able projection? (q8_row/q1). f32 (a/b) is per-position; anything else bails.
    // kinds 5/6 (q4_tiled, q4tp) have tile GEMMs too — admitting only 0/1
    // is what kept every q4tp model off the batched path.
    let gemmable = |m: &GMat| matches!(m.kind, 0 | 1 | 5 | 6);
    let mut lws = Vec::with_capacity(layers.len());
    let mut gdn_dims: Option<(usize, usize, usize, usize, usize, usize)> = None;
    for l in layers {
        let attn = match &l.attn {
            crate::gpu::GraphAttn::Full {
                wq,
                wk,
                wv,
                wo,
                output_gate,
                bias,
                ..
            } => {
                if bias.is_some() {
                    bgraph_refused("site:5889");
                    return false;
                } // batched bias axpy not wired
                let qrows = nh * hd * (1 + *output_gate as usize);
                let (Some(wq), Some(wk), Some(wv), Some(wo)) = (
                    resolve(wq, qrows, hidden),
                    resolve(wk, nkv * hd, hidden),
                    resolve(wv, nkv * hd, hidden),
                    resolve(wo, hidden, nh * hd),
                ) else {
                    bgraph_refused("site:5898");
                    return false;
                };
                if !(gemmable(&wq) && gemmable(&wk) && gemmable(&wv) && gemmable(&wo)) {
                    bgraph_refused("attention weights not gemmable");
                    return false;
                }
                LAttn::Full { wq, wk, wv, wo }
            }
            crate::gpu::GraphAttn::Gdn {
                qkv,
                z,
                a,
                b,
                out,
                nv,
                nk,
                dk,
                dv,
                kk,
                ..
            } => {
                let cdim = 2 * nk * dk + nv * dv;
                gdn_dims = Some((*nv, *nk, *dk, *dv, *kk, cdim));
                let (Some(qkv), Some(z), Some(a), Some(b), Some(out)) = (
                    resolve(qkv, cdim, hidden),
                    resolve(z, nv * dv, hidden),
                    resolve(a, *nv, hidden),
                    resolve(b, *nv, hidden),
                    resolve(out, hidden, nv * dv),
                ) else {
                    bgraph_refused("site:5928");
                    return false;
                };
                if !(gemmable(&qkv) && gemmable(&z) && gemmable(&out) && a.kind == 4 && b.kind == 4)
                {
                    bgraph_refused("site:5932");
                    return false;
                }
                LAttn::Gdn {
                    qkv,
                    z,
                    a,
                    b,
                    out,
                    nv: *nv,
                    nk: *nk,
                    dk: *dk,
                    dv: *dv,
                    kk: *kk,
                    cdim,
                }
            }
        };
        let bffn = match &l.ffn {
            crate::gpu::GraphFfn::Dense {
                gate: lg,
                up: lu,
                down: ld,
            } => {
                let (Some(gate), Some(up), Some(down)) = (
                    resolve(lg, inter, hidden),
                    resolve(lu, inter, hidden),
                    resolve(ld, hidden, inter),
                ) else {
                    bgraph_refused("site:5960");
                    return false;
                };
                if !(gemmable(&gate) && gemmable(&up) && gemmable(&down)) {
                    bgraph_refused("dense FFN not gemmable");
                    return false;
                }
                BFfn::Dense { gate, up, down }
            }
            crate::gpu::GraphFfn::Moe {
                router,
                shared_gate,
                experts,
                n_exp,
                top_k,
                inter: mi,
                norm_topk,
                q4tp,
                gu_q2,
            } => {
                if *top_k >= 16 || *n_exp > 256 || experts.len() != n_exp + 1 {
                    bgraph_refused("site:5979");
                    return false;
                }
                let (Some(router), Some(sgate)) = (
                    resolve(router, *n_exp, hidden),
                    resolve(shared_gate, 1, hidden),
                ) else {
                    bgraph_refused("site:5985");
                    return false;
                };
                let Some((gate_all, up_all, down_all)) =
                    moe_expert_bufs(c, model, experts, *mi, hidden, *q4tp, false)
                else {
                    bgraph_refused("site:5990");
                    return false;
                };
                BFfn::Moe {
                    router,
                    sgate,
                    gate_all,
                    up_all,
                    down_all,
                    n_exp: *n_exp,
                    top_k: *top_k,
                    inter: *mi,
                    norm_topk: *norm_topk,
                    q4tp: *q4tp,
                }
            }
        };
        lws.push(LW { attn, ffn: bffn });
    }
    let stor = |data: &[u8]| {
        let b = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&b, 0, data);
        b
    };
    let unif = |d: &[u32]| {
        c.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(d),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    };
    let bg = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let e: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| bind_buf(i as u32, b))
            .collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &e,
        })
    };
    // Buffers usable both as compute storage and copy src/dst (K-loop slicing).
    let rwc = |n: usize| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let h_buf = rwc(k * hidden);
    c.queue
        .write_buffer(&h_buf, 0, bytemuck::cast_slice(&h[..k * hidden]));
    let n1 = rwc(k * hidden);
    let any_gate = layers.iter().any(|l| {
        matches!(
            &l.attn,
            crate::gpu::GraphAttn::Full {
                output_gate: true,
                ..
            }
        )
    });
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
    // Whole-batch a/b planes: one token-axis matvec per layer fills them,
    // and gdn_step reads its token's row via GdnP.tok.
    let a_bb = rwc(k * gnv);
    let b_bb = rwc(k * gnv);
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
                        let mk = || {
                            c.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("kv"),
                                size: sz,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            })
                        };
                        KvMirror {
                            k: mk(),
                            v: mk(),
                            synced: 0,
                        }
                    });
                    kvbufs.push(Some((e.k.clone(), e.v.clone())));
                    gdnbufs.push(None);
                }
                crate::gpu::GraphAttn::Gdn { .. } => {
                    let e = gsm.entry((kv_id, li)).or_insert_with(|| {
                        let ring_sz = (gcdim * (_gkk.max(1).saturating_sub(1)) * 4) as u64;
                        let s_sz = (gnv * gdk * gdv * 4) as u64;
                        let mk = |sz: u64| {
                            let bf = c.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("gdn-state"),
                                size: sz.max(4),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            });
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
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch-graph"),
        });
    let go =
        |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, b: &wgpu::BindGroup, g: u32| {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(p);
            pass.set_bind_group(0, b, &[]);
            pass.dispatch_workgroups(g, 1, 1);
        };
    let flags = |qn: bool, kn: bool| {
        (if qn { 2u32 } else { 0 }) | (if kn { 4 } else { 0 }) | (if gemma { 8 } else { 0 })
    };
    let rms_u = unif(&[hidden as u32, if gemma { 1 } else { 0 }, eps.to_bits(), 0]);
    let silu_u = unif(&[(k * inter) as u32, 0, 0, 0]);
    // Batched GEMM matvec (q8_row / q1) into a [k·rows] output.
    let ematb = |enc: &mut wgpu::CommandEncoder,
                 m: &GMat,
                 xs: &wgpu::Buffer,
                 y: &wgpu::Buffer,
                 rows: usize,
                 cols: usize| {
        match m.kind {
            0 => encode_q8_mm(c, enc, &m.buf, m.rs.as_ref().unwrap(), xs, y, rows, cols, k),
            5 => encode_q4_tile_mm(c, enc, &c.q4t_mm, &m.buf, xs, y, rows, cols, k),
            6 => encode_q4_tile_mm(c, enc, &c.q4tp_mm, &m.buf, xs, y, rows, cols, k),
            _ => encode_q1_mm(c, enc, &m.buf, xs, y, rows, cols, k),
        }
    };
    // SINGLE-row matvec for the per-token stretches inside the batch (the MoE
    // router/gate run once per token). `ematb` bakes nb=k into the GEMM: fed a
    // one-row buffer it reads k rows past the end and writes k rows into a
    // one-row output — and it has no f32 arm at all, so a kind-4 router fell
    // into the q1 decoder. Both were enough to turn the answer into noise.
    let emat1 = |enc: &mut wgpu::CommandEncoder,
                 m: &GMat,
                 xs: &wgpu::Buffer,
                 y: &wgpu::Buffer,
                 rows: usize,
                 cols: usize| {
        match m.kind {
            0 => encode_matvec(c, enc, &m.buf, xs, m.rs.as_ref().unwrap(), y, rows, cols),
            1 => encode_matvec_q1(c, enc, &m.buf, xs, y, rows, cols),
            5 => {
                if c.use_mv4 {
                    let gpr = cols / 32;
                    let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
                    let layout = c.q4t_mv8.get_bind_group_layout(0);
                    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &layout,
                        entries: &[
                            bind_buf(0, &m.buf),
                            bind_buf(2, y),
                            bind_buf(3, &p_buf),
                            bind_buf(4, &m.buf),
                            bind_buf(5, xs),
                        ],
                    });
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&c.q4t_mv8);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups((rows as u32).div_ceil(8).min(MAX_WG), 1, 1);
                } else {
                    encode_q1t_like(c, enc, &c.q4t_mv, &m.buf, xs, y, rows, cols)
                }
            }
            6 => {
                if c.use_mv4 {
                    encode_q4tp_mv4(c, enc, &m.buf, xs, y, rows, cols)
                } else {
                    encode_q1t_like(c, enc, &c.q4tp_mv, &m.buf, xs, y, rows, cols)
                }
            }
            _ => encode_f32matvec(c, enc, &m.buf, xs, y, rows, cols),
        }
    };
    let cp =
        |enc: &mut wgpu::CommandEncoder,
         src: &wgpu::Buffer,
         so: usize,
         dst: &wgpu::Buffer,
         n: usize| enc.copy_buffer_to_buffer(src, (so * 4) as u64, dst, 0, (n * 4) as u64);
    // Однострочные срезы батча для MoE: его ядра написаны на ОДИН токен,
    // поэтому i-я строка копируется сюда, считается и уезжает обратно.
    let row_in = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bg-row-in"),
        size: (hidden * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let row_out = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bg-row-out"),
        size: (hidden * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let moe_bufs = lws.iter().find_map(|w| match &w.ffn {
        BFfn::Moe {
            n_exp,
            top_k,
            inter: mi,
            ..
        } => Some((*n_exp, *top_k + 1, *mi)),
        _ => None,
    });
    let moe_bufs = moe_bufs.map(|(mn, ms, mi)| {
        let mk = |n: usize, label: &str| {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * 4).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        (
            mk(k * mn, "bg-mlogit"),
            mk(1, "bg-mslog"),
            mk(k * ms, "bg-msel"),
            mk(k * ms, "bg-mwt"),
            mk(k * ms * mi, "bg-mact"),
        )
    });
    let cpo =
        |enc: &mut wgpu::CommandEncoder,
         src: &wgpu::Buffer,
         dst: &wgpu::Buffer,
         dof: usize,
         n: usize| enc.copy_buffer_to_buffer(src, 0, dst, (dof * 4) as u64, (n * 4) as u64);
    // Bootstrap first layer's input norm over all k rows.
    let inw0 = stor(bytemuck::cast_slice(layers[0].input_norm));
    go(
        &mut enc,
        &c.rmsnorm_b,
        &bg(&c.layout_rmsnorm_b, &[&h_buf, &inw0, &n1, &rms_u]),
        k as u32,
    );
    for (li, l) in layers.iter().enumerate() {
        let lw = &lws[li];
        let pnw = stor(bytemuck::cast_slice(l.post_norm));
        match (&lw.attn, &l.attn) {
            (
                LAttn::Full { wq, wk, wv, wo },
                crate::gpu::GraphAttn::Full {
                    q_norm,
                    k_norm,
                    output_gate,
                    ..
                },
            ) => {
                let (kbuf, vbuf) = kvbufs[li].as_ref().unwrap();
                let qnw = stor(bytemuck::cast_slice(q_norm.unwrap_or(&vec![0f32; hd])));
                let knw = stor(bytemuck::cast_slice(k_norm.unwrap_or(&vec![0f32; hd])));
                let qrows = nh * hd * (1 + *output_gate as usize);
                ematb(&mut enc, wq, &n1, &qraw_b, qrows, hidden);
                ematb(&mut enc, wk, &n1, &kb_b, nkv * hd, hidden);
                ematb(&mut enc, wv, &n1, &vb_b, nkv * hd, hidden);
                for i in 0..k {
                    let p = positions[i];
                    let gate_flag = if *output_gate { 1u32 } else { 0 };
                    // Token offsets ride the kernels' spare uniform words —
                    // the three staging copies this loop carried were half
                    // its commands.
                    let rope_u = uniform_u32x8(
                        c,
                        [
                            nh as u32,
                            nkv as u32,
                            hd as u32,
                            rd as u32,
                            p as u32,
                            flags(q_norm.is_some(), k_norm.is_some()) | gate_flag,
                            eps.to_bits(),
                            i as u32,
                        ],
                    );
                    let kv_u = uniform_u32x4(
                        c,
                        [nkv as u32, hd as u32, cap as u32, (p | (i << 20)) as u32],
                    );
                    let at_u = unif(&[
                        nh as u32,
                        (nh / nkv) as u32,
                        hd as u32,
                        cap as u32,
                        (p + 1) as u32,
                        0,
                        0,
                        0,
                    ]);
                    go(
                        &mut enc,
                        &c.attn_rope,
                        &bg(
                            &c.layout_attn_rope,
                            &[
                                &qraw_b, &kb_b, &qout_s, &gout_s, &qnw, &knw, &invf_b, &rope_u,
                            ],
                        ),
                        (nh + nkv) as u32,
                    );
                    go(
                        &mut enc,
                        &c.kv_append,
                        &bg(&c.layout_kv, &[&kb_b, &vb_b, kbuf, vbuf, &kv_u]),
                        ((nkv * hd) as u32).div_ceil(256),
                    );
                    let (ap, al) = attend_pipes(c, hd);
                    go(
                        &mut enc,
                        ap,
                        &bg(al, &[&qout_s, kbuf, vbuf, &attn_s, &at_u]),
                        nh as u32,
                    );
                    if *output_gate {
                        let gm_u = unif(&[(nh * hd) as u32, 0, 0, 0]);
                        go(
                            &mut enc,
                            &c.gate_mul,
                            &bg(&c.layout_gate_mul, &[&gout_s, &attn_s, &gm_u]),
                            ((nh * hd) as u32).div_ceil(256),
                        );
                    }
                    cpo(&mut enc, &attn_s, &attn_bb, i * nh * hd, nh * hd);
                }
                ematb(&mut enc, wo, &attn_bb, &ob, hidden, nh * hd);
            }
            (
                LAttn::Gdn {
                    qkv,
                    z,
                    a,
                    b,
                    out,
                    nv,
                    nk,
                    dk,
                    dv,
                    kk,
                    cdim,
                },
                crate::gpu::GraphAttn::Gdn {
                    conv1d,
                    a_log,
                    dt_bias,
                    norm,
                    ..
                },
            ) => {
                let (ring, s) = gdnbufs[li].as_ref().unwrap();
                let taps = stor(bytemuck::cast_slice(conv1d));
                let alog = stor(bytemuck::cast_slice(a_log));
                let dtb = stor(bytemuck::cast_slice(dt_bias));
                let gnorm = stor(bytemuck::cast_slice(norm));
                ematb(&mut enc, qkv, &n1, &qkv_b, *cdim, hidden);
                ematb(&mut enc, z, &n1, &z_b, nv * dv, hidden);
                let gc_p = unif(&[*cdim as u32, *kk as u32, 0, 0]);
                let gd_p = unif(&[
                    *nv as u32,
                    *dk as u32,
                    *dv as u32,
                    (nk * dk) as u32,
                    (nv / nk) as u32,
                    *cdim as u32,
                    eps.to_bits(),
                    0,
                ]);
                // Token offsets ride in the kernels' spare uniform words:
                // conv reads its token's qkv slice, step reads/writes its
                // token's z/output rows in the BATCH buffers. The staging
                // copies this replaces were 4 of the 8 commands per token
                // per GDN layer of a chunk.
                // a/b for EVERY token in one dispatch each — the per-token
                // matvecs were 1920 of the chunk's ~4800 remaining commands.
                let fb_u = uniform_u32x4(c, [hidden as u32, *nv as u32, 0, 0]);
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    for (w, y) in [(&a.buf, &a_bb), (&b.buf, &b_bb)] {
                        pass.set_pipeline(&c.f32_matvec_b);
                        pass.set_bind_group(0, &bg(&c.layout_f32b, &[w, &n1, y, &fb_u]), &[]);
                        pass.dispatch_workgroups((*nv as u32).min(MAX_WG), k as u32, 1);
                    }
                }
                for i in 0..k {
                    let gc_pt = uniform_u32x4(c, [*cdim as u32, *kk as u32, (i * cdim) as u32, 0]);
                    go(
                        &mut enc,
                        &c.gdn_conv,
                        &bg(&c.layout_gdn_conv, &[&qkv_b, &taps, ring, &cq_s, &gc_pt]),
                        (*cdim as u32).div_ceil(256),
                    );
                    let gd_pt = uniform_u32x8(
                        c,
                        [
                            *nv as u32,
                            *dk as u32,
                            *dv as u32,
                            (nk * dk) as u32,
                            (nv / nk) as u32,
                            *cdim as u32,
                            eps.to_bits(),
                            i as u32,
                        ],
                    );
                    go(
                        &mut enc,
                        &c.gdn_step,
                        &bg(
                            &c.layout_gdn,
                            &[
                                &cq_s, &z_b, &a_bb, &b_bb, &alog, &dtb, &gnorm, s, &gdo_b, &gd_pt,
                            ],
                        ),
                        *nv as u32,
                    );
                }
                ematb(&mut enc, out, &gdo_b, &ob, hidden, nv * dv);
            }
            _ => return false,
        }
        go(
            &mut enc,
            &c.add_rmsnorm_b,
            &bg(&c.layout_add_rmsnorm_b, &[&h_buf, &ob, &pnw, &n1, &rms_u]),
            k as u32,
        );
        match &lw.ffn {
            BFfn::Dense { gate, up, down } => {
                ematb(&mut enc, gate, &n1, &gbuf, inter, hidden);
                ematb(&mut enc, up, &n1, &ubuf, inter, hidden);
                go(
                    &mut enc,
                    &c.silu,
                    &bg(&c.layout_silu, &[&gbuf, &ubuf, &dummy_hd, &abuf, &silu_u]),
                    ((k * inter) as u32).div_ceil(256),
                );
                ematb(&mut enc, down, &abuf, &ob, hidden, inter);
            }
            // Routing is per token, so the experts run token by token —
            // but inside THIS submit, next to the batched attention and
            // projections. Same four kernels the token graph uses, fed a
            // one-row slice of the batch and writing one row back.
            BFfn::Moe {
                router,
                sgate,
                gate_all,
                up_all,
                down_all,
                n_exp,
                top_k,
                inter: mi,
                norm_topk,
                q4tp,
            } => {
                let (mlogit, mslog, msel, mwt, mact) = moe_bufs.as_ref().unwrap();
                let mut continue_ffn = true;
                let slots = *top_k + 1;
                let mat16 = |rows: usize, cols: usize| -> u32 {
                    let n = if *q4tp {
                        cortiq_core::quant::expected_nbytes(
                            cortiq_core::TensorDtype::Q4TiledP,
                            &[rows, cols],
                        )
                        .unwrap_or(0)
                    } else {
                        rows * (cols / 32) * 18
                    };
                    (n / 2) as u32
                };
                let sg_fold = sgate.kind == 4;
                let sel_u = uniform_u32x4(
                    c,
                    [
                        *n_exp as u32,
                        *top_k as u32,
                        *norm_topk as u32,
                        (hidden as u32) << 8 | u32::from(sg_fold) * 4,
                    ],
                );
                let gu_u = uniform_u32x8(
                    c,
                    [(hidden / 32) as u32, *mi as u32, slots as u32, mat16(*mi, hidden), 0, 0, 0, 0],
                );
                let dn_u = uniform_u32x4(
                    c,
                    [
                        (*mi / 32) as u32,
                        hidden as u32,
                        slots as u32,
                        mat16(hidden, *mi),
                    ],
                );
                let (p_gu, p_dn, l_gu, l_dn) = if *q4tp {
                    (
                        &c.moe_gate_up_q4tp,
                        &c.moe_down_q4tp,
                        &c.layout_moe_gu_q4tp,
                        &c.layout_moe_dn_q4tp,
                    )
                } else {
                    (
                        &c.moe_gate_up,
                        &c.moe_down,
                        &c.layout_moe_gu,
                        &c.layout_moe_dn,
                    )
                };
                if *q4tp && router.kind == 4 && sgate.kind == 4 && *n_exp <= 256 {
                    // Uniform q4tp experts + f32 router/gate: k router
                    // matvecs (offset bindings, no staging rows) plus THREE
                    // token-axis dispatches for select/experts/down. The
                    // loop below is ~7 commands per token per layer and
                    // clocks the chunk at per-position speed.
                    // Router for every token in ONE dispatch; per-row math is
                    // f32_matvec verbatim, so the logits stay bit-identical.
                    let fr_u = uniform_u32x4(c, [hidden as u32, *n_exp as u32, 0, 0]);
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&c.f32_matvec_b);
                        pass.set_bind_group(
                            0,
                            &bg(&c.layout_f32b, &[&router.buf, &n1, mlogit, &fr_u]),
                            &[],
                        );
                        pass.dispatch_workgroups((*n_exp as u32).min(MAX_WG), k as u32, 1);
                    }
                    let bg_sel = bg(
                        &c.layout_moe_sel_b,
                        &[mlogit, &n1, msel, mwt, &sel_u, &sgate.buf],
                    );
                    let bg_gu = bg(
                        &c.layout_moe_gu_b,
                        &[gate_all, up_all, &n1, msel, mact, &gu_u],
                    );
                    let bg_dn = bg(&c.layout_moe_dn_b, &[down_all, mact, msel, mwt, &ob, &dn_u]);
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&c.moe_select_b);
                    pass.set_bind_group(0, &bg_sel, &[]);
                    pass.dispatch_workgroups(k as u32, 1, 1);
                    pass.set_pipeline(&c.moe_gate_up_q4tp_b);
                    pass.set_bind_group(0, &bg_gu, &[]);
                    pass.dispatch_workgroups(*mi as u32, slots as u32, k as u32);
                    pass.set_pipeline(&c.moe_down_q4tp_b);
                    pass.set_bind_group(0, &bg_dn, &[]);
                    pass.dispatch_workgroups(hidden as u32, k as u32, 1);
                    drop(pass);
                    continue_ffn = false;
                }
                if continue_ffn {
                    for i in 0..k {
                        cp(&mut enc, &n1, i * hidden, &row_in, hidden);
                        let bg_sel = bg(
                            &c.layout_moe_sel,
                            &[mlogit, mslog, msel, mwt, &sel_u, &sgate.buf, &row_in],
                        );
                        let bg_gu = bg(l_gu, &[gate_all, up_all, &row_in, msel, mact, &gu_u]);
                        let bg_dn = bg(l_dn, &[down_all, mact, msel, mwt, &row_out, &dn_u]);
                        emat1(&mut enc, router, &row_in, mlogit, *n_exp, hidden);
                        if !sg_fold {
                            emat1(&mut enc, sgate, &row_in, mslog, 1, hidden);
                        }
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&c.moe_select);
                        pass.set_bind_group(0, &bg_sel, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                        pass.set_pipeline(p_gu);
                        pass.set_bind_group(0, &bg_gu, &[]);
                        pass.dispatch_workgroups(*mi as u32, slots as u32, 1);
                        pass.set_pipeline(p_dn);
                        pass.set_bind_group(0, &bg_dn, &[]);
                        pass.dispatch_workgroups(hidden as u32, 1, 1);
                        drop(pass);
                        cpo(&mut enc, &row_out, &ob, i * hidden, hidden);
                    }
                }
            }
        }
        if li + 1 < layers.len() {
            let inw_next = stor(bytemuck::cast_slice(layers[li + 1].input_norm));
            go(
                &mut enc,
                &c.add_rmsnorm_b,
                &bg(
                    &c.layout_add_rmsnorm_b,
                    &[&h_buf, &ob, &inw_next, &n1, &rms_u],
                ),
                k as u32,
            );
        } else {
            let ax_u = unif(&[1.0f32.to_bits(), (k * hidden) as u32, 0, 0]);
            go(
                &mut enc,
                &c.axpy,
                &bg(&c.layout_axpy, &[&ob, &h_buf, &ax_u]),
                ((k * hidden) as u32).div_ceil(256),
            );
        }
    }
    let size = (k * hidden * 4) as u64;
    let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bg-stage"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let ok = readback(c, enc, &h_buf, &stage, size, &mut h[..k * hidden]);
    if ok {
        let mut kvm = c.attn_kv.lock().unwrap();
        for li in 0..layers.len() {
            if let Some(m) = kvm.get_mut(&(kv_id, li)) {
                m.synced = pos0 + k;
            }
        }
    }
    ok
}

/// Drop the device K/V mirror for a pipeline (called on cache clear).
pub fn kv_mirror_reset(kv_id: u64) {
    if let Some(c) = ctx() {
        c.attn_kv.lock().unwrap().retain(|(id, _), _| *id != kv_id);
        c.gdn_state
            .lock()
            .unwrap()
            .retain(|(id, _), _| *id != kv_id);
    }
}

/// GDN depthwise conv step (bring-up / parity): updates cq [cdim] and shifts
/// the ring [(kk-1)·cdim] in place.
pub fn gdn_conv_gpu(
    qkv: &[f32],
    taps: &[f32],
    ring: &mut [f32],
    cdim: usize,
    kk: usize,
    cq: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let qb = storage_bytes(c, bytemuck::cast_slice(qkv));
    let tb = storage_bytes(c, bytemuck::cast_slice(taps));
    let rb = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(ring),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let cb = rw_f32(c, cdim, true);
    let p = uniform_u32x4(c, [cdim as u32, kk as u32, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_gdn_conv,
        entries: &[
            bind_buf(0, &qb),
            bind_buf(1, &tb),
            bind_buf(2, &rb),
            bind_buf(3, &cb),
            bind_buf(4, &p),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.gdn_conv);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((cdim as u32).div_ceil(256), 1, 1);
    }
    let rsz = (ring.len() * 4) as u64;
    let csz = (cdim * 4) as u64;
    let sr = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: rsz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let scq = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: csz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&rb, 0, &sr, 0, rsz);
    enc.copy_buffer_to_buffer(&cb, 0, &scq, 0, csz);
    c.queue.submit(Some(enc.finish()));
    sr.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    scq.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    let (Ok(dr), Ok(dc)) = (
        sr.slice(..).get_mapped_range(),
        scq.slice(..).get_mapped_range(),
    ) else {
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
    let sb = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gdn-s"),
            contents: bytemuck::cast_slice(s),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let ob = rw_f32(c, nv * dv, true);
    let p = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gdn-p"),
            contents: bytemuck::cast_slice(&[
                nv as u32,
                dk as u32,
                dv as u32,
                kd as u32,
                rep as u32,
                cdim as u32,
                eps.to_bits(),
                0u32,
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
    let stage_s = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: ssz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let stage_o = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: osz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&sb, 0, &stage_s, 0, ssz);
    enc.copy_buffer_to_buffer(&ob, 0, &stage_o, 0, osz);
    c.queue.submit(Some(enc.finish()));
    stage_s.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    stage_o.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    let (Ok(ds), Ok(dobuf)) = (
        stage_s.slice(..).get_mapped_range(),
        stage_o.slice(..).get_mapped_range(),
    ) else {
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
        c.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blk-u"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    };
    let stor = |data: &[u8]| {
        c.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("blk-w"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
    };
    // Resident buffers.
    let h_buf = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| bind_buf(i as u32, b))
            .collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    };
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attn-block"),
        });
    let dispatch = |enc: &mut wgpu::CommandEncoder,
                    pipe: &wgpu::ComputePipeline,
                    bind: &wgpu::BindGroup,
                    groups: u32| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    };
    // 1. rmsnorm(h) -> normed
    let rms_p = unif(&[hidden as u32, 0, eps.to_bits(), 0]);
    dispatch(
        &mut enc,
        &c.rmsnorm,
        &bg(&c.layout_rmsnorm, &[&h_buf, &normw_b, &normed_b, &rms_p]),
        1,
    );
    // 2. QKV (q1) from normed
    encode_matvec_q1(c, &mut enc, &wq_b, &normed_b, &qraw_b, nh * hd, hidden);
    encode_matvec_q1(c, &mut enc, &wk_b, &normed_b, &k_b, nkv * hd, hidden);
    encode_matvec_q1(c, &mut enc, &wv_b, &normed_b, &v_b, nkv * hd, hidden);
    // 3. rope + qk-norm
    let rq_p = unif(&[
        nh as u32,
        nkv as u32,
        hd as u32,
        rd as u32,
        stored as u32,
        flags,
        eps.to_bits(),
        0,
    ]);
    dispatch(
        &mut enc,
        &c.attn_rope,
        &bg(
            &c.layout_attn_rope,
            &[
                &qraw_b, &k_b, &qout_b, &gout_b, &qnw_b, &knw_b, &invf_b, &rq_p,
            ],
        ),
        (nh + nkv) as u32,
    );
    // 4. kv_append
    let kv_p = unif(&[nkv as u32, hd as u32, cap as u32, stored as u32]);
    let kv_groups = ((nkv * hd) as u32).div_ceil(256);
    dispatch(
        &mut enc,
        &c.kv_append,
        &bg(&c.layout_kv, &[&k_b, &v_b, kbuf, vbuf, &kv_p]),
        kv_groups,
    );
    // 5. attend
    let at_p = unif(&[
        nh as u32,
        (nh / nkv) as u32,
        hd as u32,
        cap as u32,
        (stored + 1) as u32,
        0,
        0,
        0,
    ]);
    {
        let (ap, al) = attend_pipes(c, hd);
        dispatch(
            &mut enc,
            ap,
            &bg(al, &[&qout_b, kbuf, vbuf, &attn_b, &at_p]),
            nh as u32,
        );
    }
    // 6. O (q1)
    encode_matvec_q1(c, &mut enc, &wo_b, &attn_b, &o_b, hidden, nh * hd);
    // 7. residual h += o
    let ax_p = unif(&[1.0f32.to_bits(), hidden as u32, 0, 0]);
    dispatch(
        &mut enc,
        &c.axpy,
        &bg(&c.layout_axpy, &[&o_b, &h_buf, &ax_p]),
        (hidden as u32).div_ceil(256),
    );
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
        None => c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
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
    c.queue
        .write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
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
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    if abs + plen > bytes.len() {
        return false;
    }
    let Some(w) = weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen]) else {
        return false; // over VRAM budget → CPU path
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q1mm-xs",
    );
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&pre[..b * cols]));
    let y_size = (b * rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q1mm-y",
    );
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, b as u32, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q1mm-bg"),
        layout: &c.layout_q1mm, // q1_mul_mm omits binding 2 (no row-scale)
        entries: &[
            bind_buf(0, &w),
            bind_buf(1, &xs_buf),
            bind_buf(3, &y_buf),
            bind_buf(4, &p_buf),
        ],
    });
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q1mm-stage",
    );
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q1mm"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q1mm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q1_mm);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(
            (rows as u32).div_ceil(64).min(MAX_WG),
            (b as u32).div_ceil(64),
            1,
        );
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
        None => c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                c.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mm-rs"),
                        contents: bytemuck::cast_slice(&row_scale[..rows]),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            })
            .clone(),
        None => c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&pre[..b * cols]));
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
    c.queue
        .write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
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
/// Batched q4t GEMM (imagegen DiT prefill shapes) — the wgpu twin of
/// the Metal q4t_matmat: one q4t_mul_mm dispatch reading the 18-byte
/// tiles from the cached weight buffer. The CPU/GPU probe arbitrates
/// per process exactly as on Metal.
/// q4tp twin of `q4t_matmat` — the batched GEMM the wide-batch arm of
/// `QTensor::matmat` reaches for (DiT prefill, MoE experts, dense FFN
/// batches). Without it a q4tp model kept that arm on the CPU while q4t
/// went to the device.
pub fn q4tp_matmat(
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
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    let Some(need) =
        cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[rows, cols])
    else {
        return false;
    };
    if plen < need || abs + plen > bytes.len() || xs.len() < b * cols || out.len() < b * rows {
        return false;
    }
    let q_buf = match weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen]) {
        Some(bf) => bf,
        None => return false,
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q4tpmm-xs",
    );
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..b * cols]));
    let y_size = (b * rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q4tpmm-y",
    );
    let params = [(cols / 4) as u32, rows as u32, b as u32, 0u32];
    let p_buf = match &sc.params {
        Some(bf) => bf.clone(),
        None => {
            let bf = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q4tpmm-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(bf.clone());
            bf
        }
    };
    c.queue
        .write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q4tpmm-stage",
    );
    let entries = [
        bind_buf(0, &q_buf),
        bind_buf(1, &xs_buf),
        bind_buf(2, &y_buf),
        bind_buf(3, &p_buf),
    ];
    let bind_mm = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q4tpmm-bg"),
        layout: &c.q4tp_mm.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q4tpmm"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q4tpmm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q4tp_mm);
        pass.set_bind_group(0, &bind_mm, &[]);
        pass.dispatch_workgroups(
            (rows as u32).div_ceil(64).min(MAX_WG),
            (b as u32).div_ceil(64),
            1,
        );
    }
    readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..b * rows])
}

pub fn q4t_matmat(
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
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    if plen < rows * gpr * 18
        || abs + plen > bytes.len()
        || xs.len() < b * cols
        || out.len() < b * rows
    {
        return false;
    }
    let q_buf = match weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen]) {
        Some(bf) => bf,
        None => return false,
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q4tmm-xs",
    );
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..b * cols]));
    let y_size = (b * rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q4tmm-y",
    );
    let params = [(cols / 4) as u32, rows as u32, b as u32, 0u32];
    let p_buf = match &sc.params {
        Some(bf) => bf.clone(),
        None => {
            let bf = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q4tmm-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            sc.params = Some(bf.clone());
            bf
        }
    };
    c.queue
        .write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q4tmm-stage",
    );
    let entries = [
        bind_buf(0, &q_buf),
        bind_buf(1, &xs_buf),
        bind_buf(2, &y_buf),
        bind_buf(3, &p_buf),
    ];
    let bind_mm = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q4tmm-bg"),
        layout: &c.q4t_mm.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q4tmm"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q4tmm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q4t_mm);
        pass.set_bind_group(0, &bind_mm, &[]);
        pass.dispatch_workgroups(
            (rows as u32).div_ceil(64).min(MAX_WG),
            (b as u32).div_ceil(64),
            1,
        );
    }
    readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..b * rows])
}

/// One q4tp matvec through the WGSL kernel, weight fetched from the model.
/// Exists so the shader can be pinned to `dequant_q4tp` in a test: the token
/// graph is the only other caller, and a wrong kernel there still produces
/// fluent text.
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
    if cols % 32 != 0 || rows == 0 || xs.len() < cols || out.len() < rows {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.dtype != cortiq_core::TensorDtype::Q4TiledP {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    let Some(need) =
        cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[rows, cols])
    else {
        return false;
    };
    if plen < need || abs + plen > bytes.len() {
        return false;
    }
    let Some(q_buf) = weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen])
    else {
        return false;
    };
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q4tp-xs",
    );
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..cols]));
    let y_size = (rows * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q4tp-y",
    );
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q4tp-stage",
    );
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q4tp-mv"),
        });
    // The hook follows the runtime's kernel choice, so the dequant-pinned
    // test exercises whichever variant real decodes will use.
    if c.use_mv4 {
        encode_q4tp_mv4(c, &mut enc, &q_buf, &xs_buf, &y_buf, rows, cols);
    } else {
        encode_q1t_like(c, &mut enc, &c.q4tp_mv, &q_buf, &xs_buf, &y_buf, rows, cols);
    }
    readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..rows])
}

/// DiT attention on wgpu: per head, scores = scale·Q·Kᵀ → row softmax →
/// P·V, then one unstack of the [nh][n][hd] panel into [n][nh·hd]. All
/// of it in ONE submission with the scores and the panel resident on the
/// device — the CPU only ships Q/K/V in and the result out. Head-major
/// inputs, matching `gpu_metal::dit_attention`.
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
    if nh == 0 || nkv == 0 || n == 0 || hd == 0 || nh % nkv != 0 {
        return false;
    }
    if qh.len() < nh * n * hd || kh.len() < nkv * n * hd || vh.len() < nkv * n * hd {
        return false;
    }
    if out.len() < n * nh * hd {
        return false;
    }
    let dev = &c.device;
    // Grow-only slots, not fresh allocations per call: a render calls
    // this 26 times per forward and the driver's allocator is not free.
    // `Scratch::ensure` also flags the cold call so the contention
    // tripwire does not read a one-off buffer creation as a busy device.
    let mut sc = c.scratch.lock().unwrap();
    let st = wgpu::BufferUsages::STORAGE;
    let up = |slot: &mut Option<(wgpu::Buffer, u64)>, data: &[f32], label: &str| -> wgpu::Buffer {
        let b = Scratch::ensure(
            dev,
            slot,
            (data.len() * 4) as u64,
            st | wgpu::BufferUsages::COPY_DST,
            label,
        );
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        b
    };
    let qb = up(&mut sc.dq, &qh[..nh * n * hd], "dit-q");
    let kb = up(&mut sc.dk, &kh[..nkv * n * hd], "dit-k");
    let vb = up(&mut sc.dv, &vh[..nkv * n * hd], "dit-v");
    let scb = Scratch::ensure(dev, &mut sc.dsc, (n * n * 4) as u64, st, "dit-scores");
    let pb = Scratch::ensure(dev, &mut sc.dpan, (nh * n * hd * 4) as u64, st, "dit-panel");
    let ab = Scratch::ensure(
        dev,
        &mut sc.dout,
        (n * nh * hd * 4) as u64,
        st | wgpu::BufferUsages::COPY_SRC,
        "dit-out",
    );
    let stage = Scratch::ensure(
        dev,
        &mut sc.dstage,
        (n * nh * hd * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "dit-stage",
    );
    drop(sc);

    // One uniform per distinct (m, k, n, scale) shape; the head offset
    // rides in the bound slice, not the params.
    let params = |m: u32, k: u32, nn: u32, sc: f32| -> wgpu::Buffer {
        let raw = [m, k, nn, sc.to_bits(), 0u32, 0u32, 0u32, 0u32];
        let b = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dit-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(&raw));
        b
    };
    let p_qk = params(n as u32, hd as u32, n as u32, scale);
    let p_sm = params(n as u32, hd as u32, n as u32, 1.0);
    let p_pv = params(n as u32, n as u32, hd as u32, 1.0);
    let p_un = params(nh as u32, n as u32, hd as u32, 1.0);

    let bind = |pipe: &wgpu::ComputePipeline,
                a: &wgpu::Buffer,
                ao: u64,
                al: u64,
                b: &wgpu::Buffer,
                bo: u64,
                bl: u64,
                cc: &wgpu::Buffer,
                co: u64,
                cl: u64,
                pp: &wgpu::Buffer|
     -> wgpu::BindGroup {
        dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dit-bg"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: a,
                        offset: ao,
                        size: std::num::NonZeroU64::new(al),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b,
                        offset: bo,
                        size: std::num::NonZeroU64::new(bl),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: cc,
                        offset: co,
                        size: std::num::NonZeroU64::new(cl),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pp.as_entire_binding(),
                },
            ],
        })
    };

    let hpk = nh / nkv;
    let head = (n * hd * 4) as u64;
    let sc_len = (n * n * 4) as u64;
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dit-attn"),
    });
    for h in 0..nh {
        let kv = (h / hpk) as u64;
        let bg_qk = bind(
            &c.dit_qk,
            &qb,
            h as u64 * head,
            head,
            &kb,
            kv * head,
            head,
            &scb,
            0,
            sc_len,
            &p_qk,
        );
        // Naga derives each pipeline's layout from the bindings it
        // actually uses: softmax touches only the scores and the params,
        // so its group has two entries, not four.
        let bg_sm = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dit-sm-bg"),
            layout: &c.dit_softmax.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_sm.as_entire_binding(),
                },
            ],
        });
        let bg_pv = bind(
            &c.dit_pv,
            &scb,
            0,
            sc_len,
            &vb,
            kv * head,
            head,
            &pb,
            h as u64 * head,
            head,
            &p_pv,
        );
        // Each stage reads what the previous one wrote to the SAME
        // scores buffer, so each gets its own pass: wgpu inserts the
        // memory barrier at pass boundaries, and three dispatches inside
        // one pass raced (max pixel error 38/255 against the CPU path).
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dit-qk"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.dit_qk);
            pass.set_bind_group(0, &bg_qk, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), (n as u32).div_ceil(64), 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dit-sm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.dit_softmax);
            pass.set_bind_group(0, &bg_sm, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dit-pv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.dit_pv);
            pass.set_bind_group(0, &bg_pv, &[]);
            pass.dispatch_workgroups((hd as u32).div_ceil(64), (n as u32).div_ceil(64), 1);
        }
    }
    {
        let total = (nh * n * hd) as u32;
        // unstack reads the panel (0) and writes the output (2).
        let bg_un = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dit-un-bg"),
            layout: &c.dit_unstack.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ab.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_un.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dit-unstack"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.dit_unstack);
        pass.set_bind_group(0, &bg_un, &[]);
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }
    readback(
        c,
        enc,
        &ab,
        &stage,
        (n * nh * hd * 4) as u64,
        &mut out[..n * nh * hd],
    )
}

/// Causal chunk attention on wgpu: `b` new queries against `s0 + b`
/// cached keys, per head, with the causal bound applied in the softmax.
/// Same three kernels as the DiT path, rectangular this time.
///
/// This is the prefill attention the CPU path only has on aarch64 — its
/// batched attend needs Accelerate or the NEON micro-GEMM, so x86 fell
/// back to a per-position scalar loop. Measured on a 256-core EPYC that
/// loop was 30% of a 512-token prefill and 46% of a 1024-token one.
///
/// `q` is head-major [nh][b][hd] (post-RoPE); `k`/`v` are per-kv-head
/// contiguous [s0+b][hd] — the cache's own layout. `out` is
/// [b][nh·hd].
#[allow(clippy::too_many_arguments)]
pub fn chunk_attend(
    q: &[f32],
    k: &[&[f32]],
    v: &[&[f32]],
    b: usize,
    s0: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    scale: f32,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let n = s0 + b;
    if nh == 0 || nkv == 0 || b == 0 || hd == 0 || nh % nkv != 0 || n == 0 {
        return false;
    }
    if q.len() < nh * b * hd || k.len() != nkv || v.len() != nkv {
        return false;
    }
    for h in 0..nkv {
        if k[h].len() < n * hd || v[h].len() < n * hd {
            return false;
        }
    }
    if out.len() < b * nh * hd {
        return false;
    }
    let dev = &c.device;
    let mut sc = c.scratch.lock().unwrap();
    let st = wgpu::BufferUsages::STORAGE;
    let qb = Scratch::ensure(
        dev,
        &mut sc.dq,
        (nh * b * hd * 4) as u64,
        st | wgpu::BufferUsages::COPY_DST,
        "ca-q",
    );
    c.queue
        .write_buffer(&qb, 0, bytemuck::cast_slice(&q[..nh * b * hd]));
    // K/V are per-head slices of the CPU cache: pack them back to back
    // so one buffer serves every head at a known stride.
    let kvsz = (nkv * n * hd * 4) as u64;
    let kb = Scratch::ensure(
        dev,
        &mut sc.dk,
        kvsz,
        st | wgpu::BufferUsages::COPY_DST,
        "ca-k",
    );
    let vb = Scratch::ensure(
        dev,
        &mut sc.dv,
        kvsz,
        st | wgpu::BufferUsages::COPY_DST,
        "ca-v",
    );
    for h in 0..nkv {
        let off = (h * n * hd * 4) as u64;
        c.queue
            .write_buffer(&kb, off, bytemuck::cast_slice(&k[h][..n * hd]));
        c.queue
            .write_buffer(&vb, off, bytemuck::cast_slice(&v[h][..n * hd]));
    }
    let scb = Scratch::ensure(dev, &mut sc.dsc, (b * n * 4) as u64, st, "ca-scores");
    let pb = Scratch::ensure(dev, &mut sc.dpan, (nh * b * hd * 4) as u64, st, "ca-panel");
    let ab = Scratch::ensure(
        dev,
        &mut sc.dout,
        (b * nh * hd * 4) as u64,
        st | wgpu::BufferUsages::COPY_SRC,
        "ca-out",
    );
    let stage = Scratch::ensure(
        dev,
        &mut sc.dstage,
        (b * nh * hd * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "ca-stage",
    );
    drop(sc);

    let params = |m: u32, kk: u32, nn: u32, s: f32, s0v: u32, caus: u32| -> wgpu::Buffer {
        let raw = [m, kk, nn, s.to_bits(), s0v, caus, 0u32, 0u32];
        let bf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ca-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&bf, 0, bytemuck::cast_slice(&raw));
        bf
    };
    let p_qk = params(b as u32, hd as u32, n as u32, scale, s0 as u32, 0);
    let p_sm = params(b as u32, hd as u32, n as u32, 1.0, s0 as u32, 1);
    let p_pv = params(b as u32, n as u32, hd as u32, 1.0, s0 as u32, 0);
    let p_un = params(nh as u32, b as u32, hd as u32, 1.0, 0, 0);

    fn slot(bf: &wgpu::Buffer, off: u64, len: u64, bind: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: bind,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: bf,
                offset: off,
                size: std::num::NonZeroU64::new(len),
            }),
        }
    }
    let qhead = (b * hd * 4) as u64;
    let khead = (n * hd * 4) as u64;
    let sc_len = (b * n * 4) as u64;
    let hpk = nh / nkv;
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("chunk-attend"),
    });
    for h in 0..nh {
        let kv = (h / hpk) as u64;
        let bg_qk = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ca-qk"),
            layout: &c.dit_qk.get_bind_group_layout(0),
            entries: &[
                slot(&qb, h as u64 * qhead, qhead, 0),
                slot(&kb, kv * khead, khead, 1),
                slot(&scb, 0, sc_len, 2),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_qk.as_entire_binding(),
                },
            ],
        });
        let bg_sm = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ca-sm"),
            layout: &c.dit_softmax.get_bind_group_layout(0),
            entries: &[
                slot(&scb, 0, sc_len, 2),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_sm.as_entire_binding(),
                },
            ],
        });
        let bg_pv = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ca-pv"),
            layout: &c.dit_pv.get_bind_group_layout(0),
            entries: &[
                slot(&scb, 0, sc_len, 0),
                slot(&vb, kv * khead, khead, 1),
                slot(&pb, h as u64 * qhead, qhead, 2),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_pv.as_entire_binding(),
                },
            ],
        });
        // One pass per stage: each reads what the previous wrote to the
        // same scores buffer, and dispatches inside one pass do not
        // order against each other.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ca-qk"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.dit_qk);
            pass.set_bind_group(0, &bg_qk, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), (b as u32).div_ceil(64), 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ca-sm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.dit_softmax);
            pass.set_bind_group(0, &bg_sm, &[]);
            pass.dispatch_workgroups(b as u32, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ca-pv"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.dit_pv);
            pass.set_bind_group(0, &bg_pv, &[]);
            pass.dispatch_workgroups((hd as u32).div_ceil(64), (b as u32).div_ceil(64), 1);
        }
    }
    {
        let bg_un = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ca-un"),
            layout: &c.dit_unstack.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ab.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_un.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ca-un"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.dit_unstack);
        pass.set_bind_group(0, &bg_un, &[]);
        pass.dispatch_workgroups(((nh * b * hd) as u32).div_ceil(256), 1, 1);
    }
    readback(
        c,
        enc,
        &ab,
        &stage,
        (b * nh * hd * 4) as u64,
        &mut out[..b * nh * hd],
    )
}

/// Fused QKV on wgpu: one upload of the normed chunk, three GEMMs, one
/// readback of Q|K|V laid out back to back. The unfused route pays three
/// submits and three uploads of the same X — at a 512-token chunk that
/// is the same 6 MB shipped three times, 44 times per prefill.
/// Weights stay cached in VRAM. `out` receives q (b·rq), then k (b·rk),
/// then v (b·rv).
#[allow(clippy::too_many_arguments)]
pub fn q4t_qkv(
    model: &Arc<CmfModel>,
    wq: usize,
    wk: usize,
    wv: usize,
    xs: &[f32],
    b: usize,
    cols: usize,
    rq: usize,
    rk: usize,
    rv: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 32 != 0 || b == 0 {
        return false;
    }
    let need = b * (rq + rk + rv);
    if xs.len() < b * cols || out.len() < need {
        return false;
    }
    let bytes = model.primary_bytes();
    let gpr = cols / 32;
    let wbuf = |idx: usize, rows: usize| -> Option<wgpu::Buffer> {
        let entry = &model.tensors[idx];
        if entry.shape.first().copied().unwrap_or(0) < rows {
            return None;
        }
        let abs = model.entry_abs_offset(entry)?;
        let plen = entry.nbytes as usize;
        if plen < rows * gpr * 18 || abs + plen > bytes.len() {
            return None;
        }
        weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen])
    };
    let (Some(bq), Some(bk), Some(bv)) = (wbuf(wq, rq), wbuf(wk, rk), wbuf(wv, rv)) else {
        return false;
    };

    let dev = &c.device;
    let mut sc = c.scratch.lock().unwrap();
    let st = wgpu::BufferUsages::STORAGE;
    let xs_buf = Scratch::ensure(
        dev,
        &mut sc.xs,
        (b * cols * 4) as u64,
        st | wgpu::BufferUsages::COPY_DST,
        "qkv-xs",
    );
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..b * cols]));
    let y_size = (need * 4) as u64;
    let y_buf = Scratch::ensure(
        dev,
        &mut sc.y,
        y_size,
        st | wgpu::BufferUsages::COPY_SRC,
        "qkv-y",
    );
    let stage = Scratch::ensure(
        dev,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "qkv-stage",
    );
    drop(sc);

    let params = |rows: usize| -> wgpu::Buffer {
        let raw = [(cols / 4) as u32, rows as u32, b as u32, 0u32];
        let bf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qkv-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&bf, 0, bytemuck::cast_slice(&raw));
        bf
    };
    let layout = c.q4t_mm.get_bind_group_layout(0);
    let mut enc =
        dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("qkv") });
    let mut off = 0u64;
    for (wbf, rows) in [(&bq, rq), (&bk, rk), (&bv, rv)] {
        let pbf = params(rows);
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qkv-bg"),
            layout: &layout,
            entries: &[
                bind_buf(0, wbf),
                bind_buf(1, &xs_buf),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &y_buf,
                        offset: off,
                        size: std::num::NonZeroU64::new((b * rows * 4) as u64),
                    }),
                },
                bind_buf(3, &pbf),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("qkv-mm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q4t_mm);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(
            (rows as u32).div_ceil(64).min(MAX_WG),
            (b as u32).div_ceil(64),
            1,
        );
        drop(pass);
        off += (b * rows * 4) as u64;
    }
    readback(c, enc, &y_buf, &stage, y_size, &mut out[..need])
}

/// Fused DiT SwiGLU FFN on wgpu: g=X·W1ᵀ, u=X·W3ᵀ, silu(g)·u,
/// y=·W2ᵀ — four passes, ONE submission, one readback. The unfused
/// per-op route pays 3 submits and ships the [b, inter] intermediates
/// across PCIe twice; on discrete cards that overhead dominates the
/// GEMM itself. Weights stay cached in VRAM.
#[allow(clippy::too_many_arguments)]
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
    if hidden % 32 != 0 || inter % 32 != 0 || b == 0 {
        return false;
    }
    let bytes = model.primary_bytes();
    let wbuf = |idx: usize, rows: usize, cols: usize| -> Option<wgpu::Buffer> {
        let entry = &model.tensors[idx];
        let abs = model.entry_abs_offset(entry)?;
        let plen = entry.nbytes as usize;
        if plen < rows * (cols / 32) * 18 || abs + plen > bytes.len() {
            return None;
        }
        weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen])
    };
    let (Some(q1), Some(q3), Some(q2)) = (
        wbuf(w1, inter, hidden),
        wbuf(w3, inter, hidden),
        wbuf(w2, hidden, inter),
    ) else {
        return false;
    };
    if xs.len() < b * hidden || out.len() < b * hidden {
        return false;
    }
    let mut sc = c.scratch.lock().unwrap();
    let xs_buf = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * hidden * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "q4tffn-xs",
    );
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..b * hidden]));
    let panel = (b * inter * 4) as u64;
    let g_buf = Scratch::ensure(
        &c.device,
        &mut sc.g,
        panel,
        wgpu::BufferUsages::STORAGE,
        "q4tffn-g",
    );
    let u_buf = Scratch::ensure(
        &c.device,
        &mut sc.u,
        panel,
        wgpu::BufferUsages::STORAGE,
        "q4tffn-u",
    );
    let y_size = (b * hidden * 4) as u64;
    let y_buf = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "q4tffn-y",
    );
    let stage_buf = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "q4tffn-stage",
    );
    // Content-keyed uniforms: three shapes live in one submission, so
    // one rewritable buffer cannot serve them.
    let p13 = uniform_u32x4(c, [(hidden / 4) as u32, inter as u32, b as u32, 0]);
    let p2 = uniform_u32x4(c, [(inter / 4) as u32, hidden as u32, b as u32, 0]);
    let psilu = uniform_u32x4(c, [(b * inter) as u32, 0, 0, 0]);
    let mm_layout = c.q4t_mm.get_bind_group_layout(0);
    let bind_mm = |q: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer, p: &wgpu::Buffer| {
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q4tffn-bg"),
            layout: &mm_layout,
            entries: &[
                bind_buf(0, q),
                bind_buf(1, x),
                bind_buf(2, y),
                bind_buf(3, p),
            ],
        })
    };
    let bg1 = bind_mm(&q1, &xs_buf, &g_buf, &p13);
    let bg3 = bind_mm(&q3, &xs_buf, &u_buf, &p13);
    let bg2 = bind_mm(&q2, &g_buf, &y_buf, &p2);
    let bg_silu = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q4tffn-silu-bg"),
        layout: &c.ffn_silu.get_bind_group_layout(0),
        entries: &[
            bind_buf(0, &g_buf),
            bind_buf(1, &u_buf),
            bind_buf(2, &psilu),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q4tffn"),
        });
    let mm_pass = |enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup, rows: usize| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q4tffn-mm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.q4t_mm);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(
            (rows as u32).div_ceil(64).min(MAX_WG),
            (b as u32).div_ceil(64),
            1,
        );
    };
    mm_pass(&mut enc, &bg1, inter);
    mm_pass(&mut enc, &bg3, inter);
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("q4tffn-silu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.ffn_silu);
        pass.set_bind_group(0, &bg_silu, &[]);
        pass.dispatch_workgroups(((b * inter) as u32).div_ceil(256).min(MAX_WG), 1, 1);
    }
    mm_pass(&mut enc, &bg2, hidden);
    readback(c, enc, &y_buf, &stage_buf, y_size, &mut out[..b * hidden])
}

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
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    if plen < rows * gpr * 9
        || abs + plen > bytes.len()
        || xs.len() < b * cols
        || out.len() < b * rows
    {
        return false;
    }
    dispatch_q1t_mm(
        c,
        Some((bytes.as_ptr() as usize, idx)),
        &bytes[abs..abs + plen],
        xs,
        b,
        rows,
        cols,
        out,
    )
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
        None => c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    c.queue
        .write_buffer(&xs_buf, 0, bytemuck::cast_slice(&xs[..b * cols]));
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
    c.queue
        .write_buffer(&p_buf, 0, bytemuck::cast_slice(&params));
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q1tmm"),
        });
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
/// Spin briefly for a submission to land instead of sleeping on it.
fn spin_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_GPU_SPIN").map(|v| v != "0").unwrap_or(true))
}

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
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d2 = done.clone();
    slice.map_async(wgpu::MapMode::Read, move |_| {
        d2.store(true, std::sync::atomic::Ordering::Release);
    });
    // A blocking wait hands the thread to the OS scheduler, and getting it
    // back costs more than the work did: these submissions finish in tens of
    // microseconds and there are 86 of them a token. Spin on the queue for a
    // short while first, then block — a decode that stalls for a real reason
    // must not burn a core forever. CMF_GPU_SPIN=0 reverts.
    if spin_wait() {
        let t0 = std::time::Instant::now();
        loop {
            let _ = c.device.poll(wgpu::PollType::Poll);
            if done.load(std::sync::atomic::Ordering::Acquire) {
                break;
            }
            if t0.elapsed() > std::time::Duration::from_millis(2) {
                if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
                    return false;
                }
                break;
            }
            std::hint::spin_loop();
        }
    } else if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    {
        let Ok(data) = slice.get_mapped_range() else {
            return false;
        };
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
    c.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
    let b = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    u.insert(v, b.clone());
    b
}

/// Three words and a float, in one uniform. The cache is keyed on the bits,
/// which is exactly right: two params differ iff their bytes differ.
fn uniform_mixed(c: &Ctx, v: [u32; 3], f: f32) -> wgpu::Buffer {
    uniform_u32x4(c, [v[0], v[1], v[2], f.to_bits()])
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
fn tensor_weight(
    c: &Ctx,
    model: &Arc<CmfModel>,
    idx: usize,
    rows: usize,
    cols: usize,
) -> Option<wgpu::Buffer> {
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return None;
    }
    let abs = model.entry_abs_offset(entry)?;
    let bytes = model.primary_bytes();
    if abs + rows * cols > bytes.len() {
        return None;
    }
    weight_buffer(
        c,
        (bytes.as_ptr() as usize, idx),
        &bytes[abs..abs + rows * cols],
    )
}

/// `tensor_weight` for tile-packed dtypes whose payload length differs
/// from rows·cols (q4_tiled: 18 B per 32-weight group).
fn tensor_weight_sized(
    c: &Ctx,
    model: &Arc<CmfModel>,
    idx: usize,
    rows: usize,
    payload: usize,
) -> Option<wgpu::Buffer> {
    let entry = &model.tensors[idx];
    if entry.shape.first().copied().unwrap_or(0) < rows {
        return None;
    }
    let abs = model.entry_abs_offset(entry)?;
    let bytes = model.primary_bytes();
    if abs + payload > bytes.len() {
        return None;
    }
    weight_buffer(
        c,
        (bytes.as_ptr() as usize, idx),
        &bytes[abs..abs + payload],
    )
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
/// Batched q4_tiled / q4tp GEMM into `enc` — the tile GEMMs the imagegen
/// path already used, wired for the graph.
///
/// `ematb` matched kind 0 and sent EVERYTHING else to the q1 kernel, which
/// for a q4tp weight is simply the wrong decoder. Together with `gemmable`
/// admitting only kinds 0 and 1, that shut batched prefill out of every
/// q4t and q4tp file — the same shape of bug as the `prep()` hole, and the
/// reason prefill ran one position at a time at 33 tok/s against 54 on
/// decode.
/// One-shot reason the BATCHED graph declined. Three silent fallbacks in a
/// row this session cost hours; a refusal that says nothing is the most
/// expensive kind of bug in this file.
fn bgraph_refused(why: &'static str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        tracing::warn!("batch graph declined: {why}");
    }
}

/// Device mirror of one layer's sealed o1 state.
struct O1Dev {
    epoch: u64,
    meta: wgpu::Buffer,
    ring_k: wgpu::Buffer,
    ring_v: wgpu::Buffer,
    sink_k: wgpu::Buffer,
    sink_v: wgpu::Buffer,
    k_tilde: wgpu::Buffer,
    qt: wgpu::Buffer,
    mu: wgpu::Buffer,
    mz: wgpu::Buffer,
    that: wgpu::Buffer,
    g: usize,
    h: usize,
    m: usize,
    w: usize,
    ns: usize,
    scale: f32,
}

/// Upload (or reuse) a layer's o1 state. One upload per seal epoch: the
/// window ring and far skeleton then live and MUTATE on the device, and
/// the CPU copy is stale by design — the same one-way discipline as the
/// KV mirror.
fn o1_ensure(
    c: &Ctx,
    kv_id: u64,
    li: usize,
    views: &[crate::nystrom::O1DeviceView<'_>],
    epoch: u64,
) -> Option<()> {
    {
        let m = c.o1m.lock().unwrap();
        if let Some(d) = m.get(&(kv_id, li)) {
            if d.epoch == epoch {
                return Some(());
            }
        }
    }
    tracing::info!("o1_ensure: UPLOADING layer {li} (epoch {epoch})");
    let g0 = views.first()?;
    let (gcnt, hcnt, m, w, ns) = (views.len(), g0.heads.len(), g0.m_eff, g0.w, g0.sink_len);
    // Landmark threads park at lane 200+ in the attend kernel.
    if ns + w > 196 || m > 32 || g0.d > 256 || g0.dv > 256 {
        return None;
    }
    for v in views {
        if v.m_eff != m || v.w != w || v.sink_len != ns || v.heads.len() != hcnt {
            return None;
        }
    }
    let (d, dv) = (g0.d, g0.dv);
    let stor_f = |data: &[f32], label: &str| -> wgpu::Buffer {
        let b = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: ((data.len() * 4).max(4)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        b
    };
    let mut meta = Vec::with_capacity(gcnt * 4);
    let (mut rk, mut rv, mut sk, mut sv, mut kt) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut qt, mut mu, mut mz, mut th) = (vec![], vec![], vec![], vec![]);
    for v in views {
        meta.extend_from_slice(&[v.win_len as u32, v.win_head as u32, v.far_len as u32, 0]);
        // Ring buffers are cap-sized already (cap = w in skeleton mode).
        rk.extend_from_slice(v.win_k);
        rk.resize(rk.len() + (w * d - v.win_k.len().min(w * d)), 0.0);
        rv.extend_from_slice(v.win_v);
        rv.resize(rv.len() + (w * dv - v.win_v.len().min(w * dv)), 0.0);
        sk.extend_from_slice(v.sink_k);
        sv.extend_from_slice(v.sink_v);
        kt.extend_from_slice(v.k_tilde);
        for hh in &v.heads {
            qt.extend_from_slice(hh.q_tilde);
            mu.extend_from_slice(hh.mu);
            mz.extend_from_slice(hh.m_max);
            mz.extend_from_slice(hh.z_hat);
            th.extend_from_slice(hh.t_hat);
        }
    }
    let meta_b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("o1-meta"),
        size: (meta.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue
        .write_buffer(&meta_b, 0, bytemuck::cast_slice(&meta));
    let dev = O1Dev {
        epoch,
        meta: meta_b,
        ring_k: stor_f(&rk, "o1-rk"),
        ring_v: stor_f(&rv, "o1-rv"),
        sink_k: stor_f(&sk, "o1-sk"),
        sink_v: stor_f(&sv, "o1-sv"),
        k_tilde: stor_f(&kt, "o1-kt"),
        qt: stor_f(&qt, "o1-qt"),
        mu: stor_f(&mu, "o1-mu"),
        mz: stor_f(&mz, "o1-mz"),
        that: stor_f(&th, "o1-th"),
        g: gcnt,
        h: hcnt,
        m,
        w,
        ns,
        scale: g0.scale,
    };
    c.o1m.lock().unwrap().insert((kv_id, li), dev);
    Some(())
}

fn encode_q4_tile_mm(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
    k: usize,
) {
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, k as u32, 0]);
    let layout = pipeline.get_bind_group_layout(0);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
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
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(
        (rows as u32).div_ceil(64).min(MAX_WG),
        (k as u32).div_ceil(64),
        1,
    );
}

fn encode_q1_mm(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
    k: usize,
) {
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, k as u32, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_q1mm,
        entries: &[
            bind_buf(0, weight),
            bind_buf(1, xs),
            bind_buf(3, y),
            bind_buf(4, &p_buf),
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.q1_mm);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(
        (rows as u32).div_ceil(64).min(MAX_WG),
        (k as u32).div_ceil(64),
        1,
    );
}

/// Encode a resident q8 GEMM (int8 weight + per-row f32 scale) into `enc`.
#[allow(dead_code)] // wired by forward_batch_graph (batched prefill, in progress)
fn encode_q8_mm(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    rs: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
    k: usize,
) {
    let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, k as u32, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_mmm,
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
    pass.set_pipeline(&c.mul_mm);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(
        (rows as u32).div_ceil(64).min(MAX_WG),
        (k as u32).div_ceil(64),
        1,
    );
}

/// Encode a plain f32 matvec (small unquantized projections) into `enc`.
fn encode_f32matvec(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) {
    let p_buf = uniform_u32x4(c, [cols as u32, rows as u32, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_f32,
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
    pass.set_pipeline(&c.f32_matvec);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
}

/// `encode_f32matvec` with byte offsets into `xs` and `y` — the batched MoE
/// router runs the SAME f32 kernel per token (bit-for-bit the logits the
/// parity-proven path produced) but reads its token's row of the batch
/// hidden and writes its token's slice of the logit plane directly. Both
/// offsets land on 256-byte boundaries (t·hidden·4 and t·n_exp·4 with
/// hidden=2048, n_exp≤256), which is all wgpu asks of a buffer binding.
#[allow(clippy::too_many_arguments)]
/// Content-keyed cache for 8-word uniforms — the per-token GDN params of a
/// batched chunk repeat every chunk, and `unif` mints a fresh buffer per
/// call (the OOM lesson of the folded-gate work).
fn uniform_u32x8(c: &Ctx, v: [u32; 8]) -> wgpu::Buffer {
    let mut u = c.uniforms8.lock().unwrap();
    if let Some(b) = u.get(&v) {
        return b.clone();
    }
    let b = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    u.insert(v, b.clone());
    b
}

fn encode_f32matvec_off(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    xs_off: u64,
    y: &wgpu::Buffer,
    y_off: u64,
    y_len: u64,
    rows: usize,
    cols: usize,
) {
    let p_buf = uniform_u32x4(c, [cols as u32, rows as u32, 0, 0]);
    let entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: weight.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: xs,
                offset: xs_off,
                size: wgpu::BufferSize::new((cols * 4) as u64),
            }),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: y,
                offset: y_off,
                size: wgpu::BufferSize::new(y_len),
            }),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: p_buf.as_entire_binding(),
        },
    ];
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.layout_f32,
        entries: &entries,
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.f32_matvec);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
}

/// Encode a q4_tiled or q1t matvec into `enc` (same 4-slot layout as q1, but
/// params are [gpr, rows, cols]; q1t reads its sparse overlay from the tail of
/// the same buffer). `pipeline` is c.q4b or c.q1t.
/// Attend kernel flavor by head_dim: stride-129 (16.5 KB of workgroup
/// memory, exists on every device) for hd <= 128, stride-257 for larger
/// heads (desktop-only — see Ctx::hd_cap).
fn attend_pipes(c: &Ctx, hd: usize) -> (&wgpu::ComputePipeline, &wgpu::BindGroupLayout) {
    if hd <= 128 {
        (&c.gqa_attend_s, &c.layout_attend_s)
    } else {
        (&c.gqa_attend, &c.layout_attend)
    }
}

fn attend_part_pipes(c: &Ctx, hd: usize) -> (&wgpu::ComputePipeline, &wgpu::BindGroupLayout) {
    if hd <= 128 {
        (&c.attend_part_s, &c.layout_attend_part_s)
    } else {
        (&c.attend_part, &c.layout_attend_part)
    }
}

/// `q4tp_matvec4`: the q1t-like binding set plus the weight buffer AGAIN at
/// slot 4 as the kernel's vec4 nibble view.
fn encode_q4tp_mv4(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) {
    let gpr = cols / 32;
    // Narrow shapes (one group per lane in the 8-row kernel) go 16-rows.
    let (pipe, per_wg) = if gpr <= 64 {
        (&c.q4tp_mv16, 16u32)
    } else {
        (&c.q4tp_mv4, 8u32)
    };
    let p_buf = uniform_u32x4(c, [gpr as u32, rows as u32, cols as u32, 0]);
    let layout = pipe.get_bind_group_layout(0);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            bind_buf(0, weight),
            bind_buf(2, y),
            bind_buf(3, &p_buf),
            bind_buf(4, weight),
            bind_buf(5, xs),
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).div_ceil(per_wg).min(MAX_WG), 1, 1);
}

/// The one-row q4tp kernel, which is the one `gpu_q4tp_parity` blesses.
/// `encode_q4tp_mv4` picks a wider variant by shape; when a frame has to
/// agree with the CPU to the last bit, agreement beats throughput.
#[allow(clippy::too_many_arguments)]
fn encode_q4tp_mv1(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
    bkey: (u8, u64, usize),
) {
    let bind = cached_bind(c, bkey, || {
        let p_buf = uniform_u32x4(c, [(cols / 32) as u32, rows as u32, cols as u32, 0]);
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.q4tp_mv.get_bind_group_layout(0),
            entries: &[
                bind_buf(0, weight),
                bind_buf(1, xs),
                bind_buf(2, y),
                bind_buf(3, &p_buf),
            ],
        })
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.q4tp_mv);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
}

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
        entries: &[
            bind_buf(0, weight),
            bind_buf(1, gate),
            bind_buf(2, up),
            bind_buf(3, y),
            bind_buf(4, &p_buf),
        ],
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q1-batch"),
        });
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
    let q4t = jobs[0].q4t;
    let q4tp = jobs[0].q4tp;
    if jobs.iter().any(|j| j.q4t != q4t || j.q4tp != q4tp) {
        return false; // mixed job kinds — honest CPU
    }
    if q4t && q4tp {
        return false; // a trio is one layout or the other
    }
    let inter = jobs[0].gate.1;
    let hidden = jobs[0].down.1;
    if out.len() != hidden {
        return false;
    }
    // Resident weights of all triples — validate first (fail → CPU entirely).
    let fetch = |idx: usize, rows: usize, cols: usize| -> Option<wgpu::Buffer> {
        if q4tp {
            // Three planes, not a flat tile: nibbles, then the per-row
            // (lo, step) pair, then the 5-bit rung codes. Only the layout
            // owner knows the total, so ask it rather than re-deriving.
            let n = cortiq_core::quant::expected_nbytes(
                cortiq_core::TensorDtype::Q4TiledP,
                &[rows, cols],
            )?;
            tensor_weight_sized(c, model, idx, rows, n)
        } else if q4t {
            tensor_weight_sized(c, model, idx, rows, rows * (cols / 32) * 18)
        } else {
            tensor_weight(c, model, idx, rows, cols)
        }
    };
    let mut w3 = Vec::with_capacity(jobs.len());
    for j in jobs {
        let (gi, gr, gc, _) = j.gate;
        let (ui, ur, uc, _) = j.up;
        let (di, dr, dc, _) = j.down;
        let align = if q4t || q4tp { 32 } else { 4 };
        if gc % align != 0 || uc % align != 0 || dc % align != 0 {
            return false;
        }
        let (Some(gw), Some(uw), Some(dw)) =
            (fetch(gi, gr, gc), fetch(ui, ur, uc), fetch(di, dr, dc))
        else {
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

        if q4tp {
            encode_q1t_like(c, &mut enc, &c.q4tp_mv, gw, &xsg, &g_buf, *gr, *gc);
            encode_q1t_like(c, &mut enc, &c.q4tp_mv, uw, &xsu, &u_buf, *ur, *uc);
        } else if q4t {
            encode_q1t_like(c, &mut enc, &c.q4t_mv, gw, &xsg, &g_buf, *gr, *gc);
            encode_q1t_like(c, &mut enc, &c.q4t_mv, uw, &xsu, &u_buf, *ur, *uc);
        } else {
            encode_matvec(c, &mut enc, gw, &xsg, &grs_b, &g_buf, *gr, *gc);
            encode_matvec(c, &mut enc, uw, &xsu, &urs_b, &u_buf, *ur, *uc);
        }
        // act = silu(g)·u·col_down
        {
            let np = uniform_u32x4(c, [inter as u32, has_col as u32, j.swiglu_limit.to_bits(), 0]);
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
        if q4tp {
            encode_q1t_like(c, &mut enc, &c.q4tp_mv, dw, &a_buf, &d_buf, *dr, *dc);
        } else if q4t {
            encode_q1t_like(c, &mut enc, &c.q4t_mv, dw, &a_buf, &d_buf, *dr, *dc);
        } else {
            encode_matvec(c, &mut enc, dw, &a_buf, &drs_b, &d_buf, *dr, *dc);
        }
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
    // wgpu has a q1 batched kernel and no q4t/q4tp twin, so those layouts
    // keep the CPU path here rather than being fed to the wrong kernel.
    if jobs.iter().any(|j| {
        matches!(
            j.layout,
            crate::gpu::BatchLayout::Q4t | crate::gpu::BatchLayout::Q4tp
        )
    }) {
        return false;
    }
    let n_q1 = jobs
        .iter()
        .filter(|j| j.layout == crate::gpu::BatchLayout::Q1)
        .count();
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batch"),
        });
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
        assert!(dispatch_matvec(
            c, None, qbytes, 0, &rs, &xs, rows, cols, &mut got
        ));

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
            c,
            None,
            qbytes,
            r0,
            &rs[r0..],
            &xs,
            rows - r0,
            cols,
            &mut got2
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
        let n: usize = std::env::var("CMF_CHAIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(896);
        let k: usize = std::env::var("CMF_CHAIN_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        assert!(n % 4 == 0);
        // Resident n×n q8 weights + row scales (values irrelevant — timing only).
        let q = vec![1i8; n * n];
        let w = c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("probe-w"),
                contents: bytemuck::cast_slice(&q),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let rs = c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("probe-rs"),
                contents: bytemuck::cast_slice(&vec![1f32; n]),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let p = c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        c.queue
            .write_buffer(&a, 0, bytemuck::cast_slice(&vec![0.01f32; n]));
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
                entries: &[
                    bind_buf(0, &w),
                    bind_buf(1, xs),
                    bind_buf(2, &rs),
                    bind_buf(3, y),
                    bind_buf(4, &p),
                ],
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
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.matvec);
            pass.set_bind_group(0, if even { &bg_ab } else { &bg_ba }, &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        };
        // Warm.
        for _ in 0..3 {
            let mut e = c
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch(&mut e, true);
            readback(&b, e);
        }
        // Per-op: K submits + K readbacks.
        let t = Instant::now();
        for i in 0..k {
            let mut e = c
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch(&mut e, i % 2 == 0);
            readback(if i % 2 == 0 { &b } else { &a }, e);
        }
        let per_op = t.elapsed().as_secs_f64();
        // Fused: K dispatches, ONE submit + ONE readback.
        let t = Instant::now();
        let mut e = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
        let xs: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 + 3) % 29) as f32 / 29.0 - 0.5)
            .collect();
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
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 13 + 7) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let w: Vec<f32> = (0..n)
            .map(|i| 0.5 + ((i * 5 + 1) % 17) as f32 / 17.0)
            .collect();
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / n as f32 + eps).sqrt();
        // plain RMSNorm
        let want: Vec<f32> = (0..n).map(|i| x[i] * inv * w[i]).collect();
        let mut got = vec![0f32; n];
        assert!(rmsnorm_row(&x, &w, &mut got, false, eps));
        let md = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-4, "wgpu rmsnorm ≠ CPU: max|Δ| = {md}");
        // gemma variant: w' = 1 + w
        let wantg: Vec<f32> = (0..n).map(|i| x[i] * inv * (1.0 + w[i])).collect();
        let mut gotg = vec![0f32; n];
        assert!(rmsnorm_row(&x, &w, &mut gotg, true, eps));
        let mdg = wantg
            .iter()
            .zip(&gotg)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
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
        let h: Vec<f32> = (0..n)
            .map(|i| ((i * 13 + 7) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let d: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3) % 61) as f32 / 61.0 - 0.5)
            .collect();
        let w: Vec<f32> = (0..n)
            .map(|i| 0.5 + ((i * 5 + 1) % 17) as f32 / 17.0)
            .collect();
        // CPU reference: h += d, then rmsnorm(h, w)
        let hd: Vec<f32> = (0..n).map(|i| h[i] + d[i]).collect();
        let ss: f32 = hd.iter().map(|x| x * x).sum();
        let inv = 1.0 / (ss / n as f32 + eps).sqrt();
        let want: Vec<f32> = (0..n).map(|i| hd[i] * inv * w[i]).collect();
        // GPU add_rmsnorm
        let hb = c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
            entries: &[
                bind_buf(0, &hb),
                bind_buf(1, &db),
                bind_buf(2, &wb),
                bind_buf(3, &ob),
                bind_buf(4, &pb),
            ],
        });
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            p.set_pipeline(&c.add_rmsnorm);
            p.set_bind_group(0, &bind, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        let mut got = vec![0f32; n];
        let sz = (n * 4) as u64;
        let mut sc = c.scratch.lock().unwrap();
        let stage = Scratch::ensure(
            &c.device,
            &mut sc.stage,
            sz,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "arn-stage",
        );
        assert!(readback(c, enc, &ob, &stage, sz, &mut got));
        drop(sc);
        let md = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
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
        let invf: Vec<f32> = (0..rd / 2)
            .map(|i| 1.0 / (10000f32).powf(2.0 * i as f32 / rd as f32))
            .collect();
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
            want_g[h * hd..(h + 1) * hd]
                .copy_from_slice(&qraw[h * 2 * hd + hd..h * 2 * hd + 2 * hd]);
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
            &qraw, &k_in, &qnw, &knw, &invf, nh, nkv, hd, rd, pos, flags, eps, &mut got_q,
            &mut got_k, &mut got_g,
        ));
        let md = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            md(&want_q, &got_q) < 1e-4,
            "q mismatch: {}",
            md(&want_q, &got_q)
        );
        assert!(
            md(&want_k, &got_k) < 1e-4,
            "k mismatch: {}",
            md(&want_k, &got_k)
        );
        assert!(
            md(&want_g, &got_g) < 1e-4,
            "gate mismatch: {}",
            md(&want_g, &got_g)
        );
    }

    #[test]
    fn wgpu_gqa_attend_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping gqa_attend parity test");
            return;
        };
        // hd=128 exercises the stride-129 kernel (exists everywhere);
        // hd=256 exercises stride-257 where the device's workgroup
        // storage allows it (32 KB devices — Adreno/Mali/wgpu-Metal —
        // honestly refuse: hd_cap gates them to the small kernel).
        attend_case(128);
        if c.hd_cap >= 256 {
            attend_case(256);
        } else {
            eprintln!("hd_cap {} — skipping hd=256 attend case", c.hd_cap);
        }
    }

    fn attend_case(hd: usize) {
        let (nh, hpk, cap, n) = (4usize, 2usize, 16usize, 5usize);
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
                .map(|p| {
                    (0..hd)
                        .map(|d| q[h * hd + d] * kc[(kh * cap + p) * hd + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0.0;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                den += *s;
            }
            for d in 0..hd {
                want[h * hd + d] = (0..n)
                    .map(|p| sc[p] * vc[(kh * cap + p) * hd + d])
                    .sum::<f32>()
                    / den;
            }
        }
        let mut got = vec![0f32; nh * hd];
        assert!(gqa_attend_gpu(&q, &kc, &vc, nh, hpk, hd, cap, n, &mut got));
        let md = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-4, "wgpu gqa_attend hd={hd} ≠ CPU: max|Δ| = {md}");
    }

    #[test]
    fn wgpu_o1_step_matches_cpu() {
        // End-to-end: the REAL NystromState is the reference — prefill a
        // group, clone it, advance the clone one token on the CPU, and
        // require the device mirror (upload + o1_far/o1_push/o1_attend)
        // to produce the same attention output from the same state.
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping o1 test");
            return;
        };
        // Production geometry (Qwen3.6): the small-dim version passed while
        // the real model garbled, so the test runs BOTH.
        for (d, dv, m, w, sink, hpg, t) in [
            (8usize, 8usize, 4usize, 8usize, 2usize, 2usize, 40usize),
            (256, 256, 32, 128, 4, 8, 430),
        ] {
            let jit = |a: usize, b: usize| ((a * 37 + b * 13 + 3) % 83) as f32 / 83.0 - 0.5;
            let ks: Vec<f32> = (0..t * d).map(|i| jit(i, 1)).collect();
            let vs: Vec<f32> = (0..t * dv).map(|i| jit(i, 2)).collect();
            let qs_own: Vec<Vec<f32>> = (0..hpg)
                .map(|h| (0..t * d).map(|i| jit(i, 3 + h)).collect())
                .collect();
            let qs_refs: Vec<&[f32]> = qs_own.iter().map(|v| v.as_slice()).collect();
            let mut st = crate::nystrom::NystromState::new_group(m, w, sink, hpg);
            st.prefill_group(&qs_refs, &ks, &vs, t, d, dv);
            // CPU ground truth for the next token (built below per group).
            // TWO groups — the production model has nkv=2, and the group
            // concatenation in the upload plus every g-offset in the kernels
            // is exactly what a single-group test cannot catch.
            let mut st2 = crate::nystrom::NystromState::new_group(m, w, sink, hpg);
            let qs2_own: Vec<Vec<f32>> = (0..hpg)
                .map(|h| (0..t * d).map(|i| jit(i, 23 + h)).collect())
                .collect();
            let qs2_refs: Vec<&[f32]> = qs2_own.iter().map(|v| v.as_slice()).collect();
            let ks2: Vec<f32> = (0..t * d).map(|i| jit(i, 21)).collect();
            let vs2: Vec<f32> = (0..t * dv).map(|i| jit(i, 22)).collect();
            st2.prefill_group(&qs2_refs, &ks2, &vs2, t, d, dv);
            let gcnt = 2usize;
            let q_new: Vec<f32> = (0..gcnt * hpg * d).map(|i| jit(i, 5)).collect();
            let k_new: Vec<f32> = (0..gcnt * d).map(|i| jit(i, 6)).collect();
            let v_new: Vec<f32> = (0..gcnt * dv).map(|i| jit(i, 7)).collect();
            let mut want = vec![0f32; gcnt * hpg * dv];
            let mut cpu1 = st.clone();
            let mut cpu2 = st2.clone();
            cpu1.step_group(
                &q_new[..hpg * d],
                &k_new[..d],
                &v_new[..dv],
                &mut want[..hpg * dv],
            );
            cpu2.step_group(
                &q_new[hpg * d..],
                &k_new[d..],
                &v_new[dv..],
                &mut want[hpg * dv..],
            );
            // Device: upload the PRE-step states, run the three kernels.
            let views = vec![st.device_view(), st2.device_view()];
            assert!(!views[0].exact_only, "t must exceed w+8 for this test");
            let mv = views[0].m_eff;
            o1_ensure(c, u64::MAX, usize::MAX, &views, 1).expect("o1 upload");
            let dev_bufs = {
                let map = c.o1m.lock().unwrap();
                let dref = map.get(&(u64::MAX, usize::MAX)).unwrap();
                (
                    dref.meta.clone(),
                    dref.ring_k.clone(),
                    dref.ring_v.clone(),
                    dref.sink_k.clone(),
                    dref.sink_v.clone(),
                    dref.k_tilde.clone(),
                    dref.qt.clone(),
                    dref.mu.clone(),
                    dref.mz.clone(),
                    dref.that.clone(),
                    dref.scale,
                )
            };
            let (dmeta, drk, drv, dsk, dsv, dkt, dqt, dmu, dmz, dth, sc) = dev_bufs;
            let rect_fm = views[0].heads[0].rect_fm;
            let stor = |data: &[f32]| {
                use wgpu::util::DeviceExt;
                c.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    })
            };
            let qb = stor(&q_new);
            let kb = stor(&k_new);
            let vb = stor(&v_new);
            let ob = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (gcnt * hpg * dv * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let o1_u = uniform_u32x8(
                c,
                [
                    hpg as u32,
                    mv as u32,
                    w as u32,
                    (sink as u32) | (u32::from(rect_fm) << 8),
                    d as u32,
                    dv as u32,
                    sc.to_bits(),
                    0,
                ],
            );
            let bgf = |layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
                let entries: Vec<_> = bufs
                    .iter()
                    .enumerate()
                    .map(|(i, b)| bind_buf(i as u32, b))
                    .collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout,
                    entries: &entries,
                })
            };
            let bg_far = bgf(
                &c.layout_o1_far,
                &[&dmeta, &drk, &drv, &dqt, &dmz, &dth, &o1_u],
            );
            let bg_push = bgf(&c.layout_o1_push, &[&dmeta, &kb, &vb, &drk, &drv, &o1_u]);
            let bg_att = bgf(
                &c.layout_o1_attend,
                &[
                    &dmeta, &qb, &drk, &drv, &dsk, &dsv, &dkt, &dmu, &dmz, &dth, &ob, &o1_u,
                ],
            );
            let mut enc = c
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&c.o1_far);
                pass.set_bind_group(0, &bg_far, &[]);
                pass.dispatch_workgroups((gcnt * hpg * mv) as u32, 1, 1);
                pass.set_pipeline(&c.o1_push);
                pass.set_bind_group(0, &bg_push, &[]);
                pass.dispatch_workgroups(gcnt as u32, 1, 1);
                pass.set_pipeline(&c.o1_attend);
                pass.set_bind_group(0, &bg_att, &[]);
                pass.dispatch_workgroups((gcnt * hpg) as u32, 1, 1);
            }
            let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (gcnt * hpg * dv * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_buffer_to_buffer(&ob, 0, &stage, 0, (gcnt * hpg * dv * 4) as u64);
            c.queue.submit([enc.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            stage.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().unwrap().unwrap();
            let got: Vec<f32> = bytemuck::cast_slice(&stage.get_mapped_range(..).unwrap()).to_vec();
            c.o1m.lock().unwrap().remove(&(u64::MAX, usize::MAX));
            let md = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                md < 1e-3,
                "wgpu o1 step ≠ CPU (d={d} m={m} w={w} hpg={hpg}): max|Δ| = {md}"
            );
        }
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
        assert!(gdn_step_gpu(
            &cq, &z, &a, &b, &alog, &dtb, &norm, &mut sg, nv, dk, dv, kd, rep, cdim, eps, &mut got
        ));
        let mo = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let msd = sc
            .iter()
            .zip(&sg)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
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
        let mc = want_cq
            .iter()
            .zip(&got_cq)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mr = rc
            .iter()
            .zip(&rg)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
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
        let (nh, nkv, hd, rd, hidden, cap, stored) =
            (4usize, 2usize, 64usize, 64usize, 128usize, 8usize, 2usize);
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
        let invf: Vec<f32> = (0..rd / 2)
            .map(|i| 1.0 / (10000f32).powf(2.0 * i as f32 / rd as f32))
            .collect();
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
            c.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cache"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
        };
        let kbuf = mkcache(&kc);
        let vbuf = mkcache(&vc);
        // ---- CPU reference ----
        let ss: f32 = h_in.iter().map(|x| x * x).sum();
        let rinv = 1.0 / (ss / hidden as f32 + eps).sqrt();
        let normed: Vec<f32> = (0..hidden).map(|i| h_in[i] * rinv * norm_w[i]).collect();
        let matvec = |w: &[f32], rows: usize, cols: usize, x: &[f32]| -> Vec<f32> {
            (0..rows)
                .map(|o| (0..cols).map(|i| w[o * cols + i] * x[i]).sum())
                .collect()
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
                .map(|p| {
                    (0..hd)
                        .map(|d| qout[h * hd + d] * kc[(kh * cap + p) * hd + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0.0;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                den += *s;
            }
            for d in 0..hd {
                attn[h * hd + d] = (0..n)
                    .map(|p| sc[p] * vc[(kh * cap + p) * hd + d])
                    .sum::<f32>()
                    / den;
            }
        }
        let o = matvec(&wo, hidden, nh * hd, &attn);
        let want: Vec<f32> = (0..hidden).map(|i| h_in[i] + o[i]).collect();
        // ---- GPU block ----
        let mut got = vec![0f32; hidden];
        assert!(attn_block_gpu(
            &h_in, &norm_w, &wq_p, &wk_p, &wv_p, &wo_p, &qnw, &knw, &invf, &kbuf, &vbuf, nh, nkv,
            hd, rd, hidden, cap, stored, flags, eps, &mut got,
        ));
        let md = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
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
        let (nh, nkv, hd, rd, hidden, cap, stored) = (
            16usize, 8usize, 128usize, 128usize, 2048usize, 256usize, 128usize,
        );
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
        let invf: Vec<f32> = (0..rd / 2)
            .map(|i| 1.0 / (10000f32).powf(2.0 * i as f32 / rd as f32))
            .collect();
        let kc = vec![0.01f32; nkv * cap * hd];
        let vc = vec![0.01f32; nkv * cap * hd];
        let mkc = |d: &[f32]| {
            c.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(d),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
        };
        let (kbuf, vbuf) = (mkc(&kc), mkc(&vc));
        let iters = 200;
        let mut hout = vec![0f32; hidden];
        // FUSED: the resident block, one submit + one readback per call.
        for _ in 0..20 {
            attn_block_gpu(
                &h_in, &norm_w, &wq_p, &wk_p, &wv_p, &wo_p, &qnw, &knw, &invf, &kbuf, &vbuf, nh,
                nkv, hd, rd, hidden, cap, stored, flags, eps, &mut hout,
            );
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            attn_block_gpu(
                &h_in, &norm_w, &wq_p, &wk_p, &wv_p, &wo_p, &qnw, &knw, &invf, &kbuf, &vbuf, nh,
                nkv, hd, rd, hidden, cap, stored, flags, eps, &mut hout,
            );
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
        let unfused_once = |normed: &mut [f32],
                            qraw: &mut [f32],
                            kk: &mut [f32],
                            vv: &mut [f32],
                            qout: &mut [f32],
                            kout: &mut [f32],
                            gout: &mut [f32],
                            attn: &mut [f32],
                            oout: &mut [f32]| {
            rmsnorm_row(&h_in, &norm_w, normed, false, eps);
            dispatch_q1(c, None, &wq_p, normed, nh * hd, hidden, qraw);
            dispatch_q1(c, None, &wk_p, normed, nkv * hd, hidden, kk);
            dispatch_q1(c, None, &wv_p, normed, nkv * hd, hidden, vv);
            attn_rope_qkn_gpu(
                qraw, kk, &qnw, &knw, &invf, nh, nkv, hd, rd, stored, flags, eps, qout, kout, gout,
            );
            gqa_attend_gpu(qout, &kc, &vc, nh, hpk, hd, cap, stored + 1, attn);
            dispatch_q1(c, None, &wo_p, attn, hidden, nh * hd, oout);
        };
        for _ in 0..20 {
            unfused_once(
                &mut normed,
                &mut qraw,
                &mut kk,
                &mut vv,
                &mut qout,
                &mut kout,
                &mut gout,
                &mut attn,
                &mut oout,
            );
        }
        let t1 = Instant::now();
        for _ in 0..iters {
            unfused_once(
                &mut normed,
                &mut qraw,
                &mut kk,
                &mut vv,
                &mut qout,
                &mut kout,
                &mut gout,
                &mut attn,
                &mut oout,
            );
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
        use cortiq_core::quant::{GROUP_SIZE, f32_to_f16, q1t_pack};
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
        let xs: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 + 3) % 29) as f32 / 29.0 - 0.5)
            .collect();
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q1t(&payload, rows, cols, &mut w);
        let mut want = vec![0f32; rows];
        for o in 0..rows {
            want[o] = (0..cols).map(|i| w[o * cols + i] * xs[i]).sum();
        }
        let mut got = vec![0f32; rows];
        assert!(dispatch_q1t(
            c, &c.q1t, None, &payload, &xs, rows, cols, &mut got
        ));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_d < 1e-2, "wgpu q1t_matvec ≠ CPU: max|Δ| = {max_d}");
    }

    #[test]
    fn wgpu_q4b_matvec_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping q4b parity test");
            return;
        };
        use cortiq_core::quant::{GROUP_SIZE, f32_to_f16};
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
        let xs: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 + 3) % 29) as f32 / 29.0 - 0.5)
            .collect();
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q4_block(&payload, &mut w);
        let mut want = vec![0f32; rows];
        for o in 0..rows {
            want[o] = (0..cols).map(|i| w[o * cols + i] * xs[i]).sum();
        }
        let mut got = vec![0f32; rows];
        assert!(dispatch_q1t(
            c, &c.q4b, None, &payload, &xs, rows, cols, &mut got
        ));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_d < 1e-2, "wgpu q4b_matvec ≠ CPU: max|Δ| = {max_d}");
    }

    #[test]
    fn wgpu_q1t_matmat_matches_cpu_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping q1t GEMM parity test");
            return;
        };
        use cortiq_core::quant::{GROUP_SIZE, f32_to_f16, q1t_pack};
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
        let xs: Vec<f32> = (0..b * cols)
            .map(|i| ((i * 13 + 7) % 31) as f32 / 31.0 - 0.5)
            .collect();
        let mut w = vec![0f32; rows * cols];
        cortiq_core::quant::dequant_q1t(&payload, rows, cols, &mut w);
        let mut want = vec![0f32; b * rows];
        for bi in 0..b {
            for o in 0..rows {
                want[bi * rows + o] = (0..cols).map(|i| w[o * cols + i] * xs[bi * cols + i]).sum();
            }
        }
        let mut got = vec![0f32; b * rows];
        assert!(dispatch_q1t_mm(
            c, None, &payload, &xs, b, rows, cols, &mut got
        ));
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
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
        let pre: Vec<f32> = (0..b * cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
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
        assert!(dispatch_matmat(
            c, None, qbytes, &rs, &pre, b, rows, cols, &mut got
        ));
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
        let pre: Vec<f32> = (0..b * cols)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
            .collect();
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
        assert!(dispatch_matmat(
            c, None, qbytes, &rs, &pre, b, rows, cols, &mut got
        ));
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
        let x: Vec<f32> = (0..b * cols)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
            .collect();
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
            entries: &[
                bind_buf(0, &qbuf),
                bind_buf(1, &xbuf),
                bind_buf(3, &ybuf),
                bind_buf(4, &pbuf),
            ],
        });
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&c.q1_mm);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((rows as u32).div_ceil(64), (b as u32).div_ceil(64), 1);
        }
        let mut sc = c.scratch.lock().unwrap();
        let stage = Scratch::ensure(
            &c.device,
            &mut sc.stage,
            (b * rows * 4) as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "q1mm-stage",
        );
        let mut got = vec![0f32; b * rows];
        assert!(readback(
            c,
            enc,
            &ybuf,
            &stage,
            (b * rows * 4) as u64,
            &mut got
        ));
        drop(sc);
        let max_d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-3, "wgpu q1_mul_mm ≠ CPU: max|Δ| = {max_d}");
    }
}

/// Was the wgpu backend ASKED for, and did it come up? The two halves must
/// be told apart: a machine nobody pointed at wgpu is a legitimate skip, a
/// machine that was pointed at it and produced no context is a failure. A
/// reserved word in one shader once took the whole context down and every
/// GPU test reported success by skipping.
pub fn selected_and_up() -> Option<bool> {
    if !selected() {
        return None; // nobody asked
    }
    Some(ctx().is_some())
}

/// Cheap device probe for `gpu::backend_available`: can wgpu bring an
/// adapter up here at all? One instance, no device/queue, no caching —
/// the caller caches.
/// Every adapter wgpu can see, and which one would be chosen. Three times in
/// one night the question "is the GPU actually visible?" was answered by
/// inference from a missing log line; this answers it directly.
/// Per-head RMS and the rope tail on the device — `rms` for the queries'
/// second normalisation, `inverse` for attention's output.
#[allow(clippy::too_many_arguments)]
pub fn rope_heads_for_test(
    x: &mut [f32],
    inv_freq: &[f32],
    nh: usize,
    hd: usize,
    rd: usize,
    pos: usize,
    eps: f32,
    rms: bool,
    inverse: bool,
) -> bool {
    let Some(c) = ctx() else { return false };
    if x.len() != nh * hd || rd > hd || rd % 2 != 0 || inv_freq.len() * 2 < rd {
        return false;
    }
    // In place: seeded from the host and read back afterwards, so the buffer
    // needs both directions. rw_f32 gives STORAGE|COPY_SRC and refuses the
    // write; storage_bytes gives STORAGE and refuses the read.
    let xb = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope-x"),
            contents: bytemuck::cast_slice(x),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let fb = storage_bytes(c, bytemuck::cast_slice(&inv_freq[..rd / 2]));
    let pb = storage_bytes(c, bytemuck::cast_slice(&[pos as f32, eps]));
    let flags = (rms as u32) | ((inverse as u32) << 1);
    let p = uniform_u32x4(c, [nh as u32, hd as u32, rd as u32, flags]);
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rope") });
    {
        let layout = c.rope_heads.get_bind_group_layout(0);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                bind_buf(0, &xb),
                bind_buf(1, &fb),
                bind_buf(2, &p),
                bind_buf(3, &pb),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.rope_heads);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(nh as u32, 1, 1);
    }
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (nh * hd * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "rope-stage",
    );
    let ok = readback(c, enc, &xb, &stage, (nh * hd * 4) as u64, x);
    drop(sc);
    ok
}

/// Stage A of the grouped low-rank output projection on the device.
///
/// `lora` rows share each group's slice of `attn`; `rows` is `groups * lora`.
/// q4tp only — the release stores wo_a that way in both published variants,
/// and guessing at a layout is how a kernel returns plausible nonsense.
pub fn o_lora_a_for_test(
    model: &Arc<CmfModel>,
    idx: usize,
    attn: &[f32],
    rows: usize,
    lora: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if rows == 0 || lora == 0 || rows % lora != 0 || out.len() < rows {
        return false;
    }
    let entry = &model.tensors[idx];
    if entry.dtype != cortiq_core::TensorDtype::Q4TiledP || entry.shape.len() != 2 {
        return false;
    }
    let (trows, cols) = (entry.shape[0], entry.shape[1]);
    let groups = rows / lora;
    if trows < rows || cols % 32 != 0 || attn.len() < groups * cols {
        return false;
    }
    let Some(abs) = model.entry_abs_offset(entry) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = entry.nbytes as usize;
    if abs + plen > bytes.len() {
        return false;
    }
    let Some(w) = weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen]) else {
        return false; // over budget → the caller keeps it on the CPU
    };
    let xb = storage_bytes(c, bytemuck::cast_slice(&attn[..groups * cols]));
    let yb = rw_f32(c, rows, true);
    let p = uniform_u32x4(c, [(cols / 32) as u32, rows as u32, lora as u32, 0]);
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("o-lora-a"),
        });
    {
        let layout = c.o_lora_a.get_bind_group_layout(0);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                bind_buf(0, &w),
                bind_buf(1, &xb),
                bind_buf(2, &yb),
                bind_buf(3, &p),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.o_lora_a);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
    }
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (rows * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "o-lora-stage",
    );
    let ok = readback(c, enc, &yb, &stage, (rows * 4) as u64, &mut out[..rows]);
    drop(sc);
    ok
}

/// The compressor's pooling step on the device — `overlap` picks the folding
/// the release uses at ratio 4, `ape` the positional bias the plain one adds.
#[allow(clippy::too_many_arguments)]
pub fn kv_pool_for_test(
    prev_kv: &[f32],
    prev_score: &[f32],
    cur_kv: &[f32],
    cur_score: &[f32],
    ape: Option<&[f32]>,
    ratio: usize,
    width: usize,
    overlap: bool,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if width == 0 || ratio == 0 || out.len() < width {
        return false;
    }
    let slots = if overlap { 2 * ratio } else { ratio };
    let stride = if overlap { 2 * width } else { width };
    if cur_kv.len() < ratio * stride || cur_score.len() < ratio * stride {
        return false;
    }
    let have_prev = overlap && prev_kv.len() >= ratio * stride && prev_score.len() >= ratio * stride;
    // Unused bindings still have to point somewhere; the current window is as
    // good a placeholder as an empty buffer and costs no allocation.
    let ckv = storage_bytes(c, bytemuck::cast_slice(cur_kv));
    let csc = storage_bytes(c, bytemuck::cast_slice(cur_score));
    let pkv = if have_prev {
        storage_bytes(c, bytemuck::cast_slice(prev_kv))
    } else {
        ckv.clone()
    };
    let psc = if have_prev {
        storage_bytes(c, bytemuck::cast_slice(prev_score))
    } else {
        csc.clone()
    };
    let use_ape = ape.is_some_and(|a| a.len() >= ratio * width) && !overlap;
    let apb = if use_ape {
        storage_bytes(c, bytemuck::cast_slice(ape.unwrap()))
    } else {
        csc.clone()
    };
    let yb = rw_f32(c, width, true);
    let flags = (overlap as u32) | ((have_prev as u32) << 1) | ((use_ape as u32) << 2);
    let p = uniform_u32x4(c, [slots as u32, width as u32, ratio as u32, flags]);
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kv-pool") });
    {
        let layout = c.kv_pool.get_bind_group_layout(0);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                bind_buf(0, &pkv),
                bind_buf(1, &psc),
                bind_buf(2, &ckv),
                bind_buf(3, &csc),
                bind_buf(4, &apb),
                bind_buf(5, &yb),
                bind_buf(6, &p),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.kv_pool);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((width as u32).div_ceil(256), 1, 1);
    }
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (width * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "kv-pool-stage",
    );
    let ok = readback(c, enc, &yb, &stage, (width * 4) as u64, &mut out[..width]);
    drop(sc);
    ok
}

/// The indexer's scoring pass on the device.
#[allow(clippy::too_many_arguments)]
pub fn index_scores_for_test(
    q: &[f32],
    kv: &[f32],
    hw: &[f32],
    nh: usize,
    hd: usize,
    n_pos: usize,
    limit: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if q.len() < nh * hd || kv.len() < n_pos * hd || hw.len() < nh || out.len() < n_pos {
        return false;
    }
    if n_pos == 0 {
        return true;
    }
    let qb = storage_bytes(c, bytemuck::cast_slice(&q[..nh * hd]));
    let kb = storage_bytes(c, bytemuck::cast_slice(&kv[..n_pos * hd]));
    let wb = storage_bytes(c, bytemuck::cast_slice(&hw[..nh]));
    let yb = rw_f32(c, n_pos, true);
    let p = uniform_u32x4(c, [nh as u32, hd as u32, n_pos as u32, limit as u32]);
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ix") });
    {
        let layout = c.index_scores.get_bind_group_layout(0);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                bind_buf(0, &qb),
                bind_buf(1, &kb),
                bind_buf(2, &wb),
                bind_buf(3, &yb),
                bind_buf(4, &p),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.index_scores);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((n_pos as u32).min(MAX_WG), 1, 1);
    }
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (n_pos * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "ix-stage",
    );
    let ok = readback(c, enc, &yb, &stage, (n_pos * 4) as u64, &mut out[..n_pos]);
    drop(sc);
    ok
}

/// Top-k positions on the device, in index order — the list `sparse_attend`
/// consumes. Bounded by the kernel's workgroup array; beyond it the caller
/// keeps the CPU's version rather than getting a truncated answer.
pub fn top_k_for_test(scores: &[f32], k: usize, out: &mut Vec<u32>) -> bool {
    let Some(c) = ctx() else { return false };
    let n = scores.len();
    if n == 0 || n > 4096 || k == 0 {
        out.clear();
        return n == 0;
    }
    let kk = k.min(n);
    let sb = storage_bytes(c, bytemuck::cast_slice(scores));
    let ib = rw_f32(c, kk, true);
    let cb = rw_f32(c, 1, true);
    let p = uniform_u32x4(c, [n as u32, k as u32, 0, 0]);
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("topk") });
    {
        let layout = c.top_k_index.get_bind_group_layout(0);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                bind_buf(0, &sb),
                bind_buf(1, &ib),
                bind_buf(2, &cb),
                bind_buf(3, &p),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.top_k_index);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let bytes = ((kk + 1) * 4) as u64;
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "topk-stage",
    );
    enc.copy_buffer_to_buffer(&ib, 0, &stage, 0, (kk * 4) as u64);
    enc.copy_buffer_to_buffer(&cb, 0, &stage, (kk * 4) as u64, 4);
    c.queue.submit(Some(enc.finish()));
    let slice = stage.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    let mut ok = false;
    if let Ok(data) = slice.get_mapped_range() {
        let words: &[u32] = bytemuck::cast_slice(&data[..bytes as usize]);
        let cnt = (words[kk] as usize).min(kk);
        out.clear();
        out.extend_from_slice(&words[..cnt]);
        ok = true;
    }
    stage.unmap();
    drop(sc);
    ok
}

#[allow(clippy::too_many_arguments)]
fn encode_hc_fold(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    state: &wgpu::Buffer,
    mixes: &wgpu::Buffer,
    sc: &wgpu::Buffer,
    base: &wgpu::Buffer,
    fold: &wgpu::Buffer,
    post: &wgpu::Buffer,
    comb: &wgpu::Buffer,
    p: &wgpu::Buffer,
) {
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.hc_pre_fold.get_bind_group_layout(0),
        entries: &[
            bind_buf(0, state),
            bind_buf(1, mixes),
            bind_buf(2, sc),
            bind_buf(3, base),
            bind_buf(4, fold),
            bind_buf(5, post),
            bind_buf(6, comb),
            bind_buf(7, p),
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.hc_pre_fold);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(1, 1, 1);
}

#[allow(clippy::too_many_arguments)]
fn encode_hc_expand(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    res: &wgpu::Buffer,
    post: &wgpu::Buffer,
    comb: &wgpu::Buffer,
    out: &wgpu::Buffer,
    p: &wgpu::Buffer,
    hc: usize,
    dim: usize,
) {
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.hc_post_expand.get_bind_group_layout(0),
        entries: &[
            bind_buf(0, x),
            bind_buf(1, res),
            bind_buf(2, post),
            bind_buf(3, comb),
            bind_buf(4, out),
            bind_buf(5, p),
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.hc_post_expand);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(((hc * dim) as u32).div_ceil(256), 1, 1);
}

/// The attention chain from an already-normed LoRA vector to the block output.
#[allow(clippy::too_many_arguments)]
fn encode_attn_chain(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    wb: &[wgpu::Buffer],
    qn: &wgpu::Buffer,
    q: &wgpu::Buffer,
    attn: &wgpu::Buffer,
    mid: &wgpu::Buffer,
    out: &wgpu::Buffer,
    cache: &wgpu::Buffer,
    ixb: &wgpu::Buffer,
    sink: &wgpu::Buffer,
    freq: &wgpu::Buffer,
    posb: &wgpu::Buffer,
    g: Dsv4AttnGeom,
    kv_id: u64,
    li: usize,
    m: usize,
) {
    encode_q4tp_mv1(c, enc, &wb[1], qn, q, g.nh * g.hd, g.q_lora, (60, kv_id, li));
    encode_rope_heads(c, enc, q, freq, posb, g.nh, g.hd, g.rd, true, false, (61, kv_id, li));
    encode_sparse_attend2(c, enc, q, cache, ixb, sink, attn, g.nh, g.hd, m, g.scale);
    encode_rope_heads(c, enc, attn, freq, posb, g.nh, g.hd, g.rd, false, true, (62, kv_id, li));
    {
        let rows = g.o_groups * g.o_lora;
        let cols = g.nh * g.hd / g.o_groups;
        let bind = cached_bind(c, (63, kv_id, li), || {
            let p = uniform_u32x4(c, [(cols / 32) as u32, rows as u32, g.o_lora as u32, 0]);
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &c.o_lora_a.get_bind_group_layout(0),
                entries: &[
                    bind_buf(0, &wb[2]),
                    bind_buf(1, attn),
                    bind_buf(2, mid),
                    bind_buf(3, &p),
                ],
            })
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.o_lora_a);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
    }
    encode_q4tp_mv1(
        c,
        enc,
        &wb[3],
        mid,
        out,
        g.dim,
        g.o_groups * g.o_lora,
        (64, kv_id, li),
    );
}

/// Route, then the chosen experts and the shared one.
#[allow(clippy::too_many_arguments)]
fn encode_moe_chain(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    logits: &wgpu::Buffer,
    x: &wgpu::Buffer,
    msel: &wgpu::Buffer,
    mwt: &wgpu::Buffer,
    mcnt: &wgpu::Buffer,
    mact: &wgpu::Buffer,
    out: &wgpu::Buffer,
    gate_all: &wgpu::Buffer,
    up_all: &wgpu::Buffer,
    down_all: &wgpu::Buffer,
    w: &Dsv4LayerW,
    g: Dsv4MoeGeom,
    n_pack: usize,
    slots: usize,
) {
    // Same address-keying rule as everywhere: the bias is built per layer, so
    // it goes through the per-call pool, not the const cache.
    let bs = match w.moe.bias {
        Some(b) if b.len() >= n_pack => frame_up(c, 25, bytemuck::cast_slice(&b[..n_pack])),
        _ => logits.clone(),
    };
    let mk = frame_buf(c, 17, n_pack * 4, true);
    let fc = match w.moe.forced {
        Some(f) if f.len() >= g.top_k => {
            let v: Vec<u32> = f[..g.top_k].iter().map(|&i| i as u32).collect();
            frame_up(c, 18, bytemuck::cast_slice(&v))
        }
        _ => frame_buf(c, 18, g.top_k * 4, true),
    };
    let rflags = (w.moe.bias.is_some_and(|b| b.len() >= n_pack) as u32)
        | ((w.moe.forced.is_some_and(|f| f.len() >= g.top_k) as u32) << 2)
        | 8
        | ((n_pack as u32) << 8);
    let rp = uniform_mixed(c, [n_pack as u32, g.top_k as u32, rflags], g.route_scale);
    let stride16 = |rows: usize, cols: usize, q2: bool| -> u32 {
        let dt = if q2 {
            cortiq_core::TensorDtype::Q2TiledP
        } else {
            cortiq_core::TensorDtype::Q4TiledP
        };
        (cortiq_core::quant::expected_nbytes(dt, &[rows, cols]).unwrap_or(0) / 2) as u32
    };
    let gu_u = uniform_u32x8(
        c,
        [
            (g.hidden / 32) as u32,
            g.inter as u32,
            slots as u32,
            stride16(g.inter, g.hidden, g.gu_q2),
            g.swiglu_limit.to_bits(),
            0,
            0,
            0,
        ],
    );
    let dn_u = uniform_u32x4(
        c,
        [
            (g.inter / 32) as u32,
            g.hidden as u32,
            slots as u32,
            stride16(g.hidden, g.inter, false),
        ],
    );
    let (p_gu, p_dn, l_gu, l_dn) = if g.gu_q2 {
        (
            &c.moe_gate_up_q2tp,
            &c.moe_down_q4tp,
            &c.layout_moe_gu_q2tp,
            &c.layout_moe_dn_q4tp,
        )
    } else {
        (
            &c.moe_gate_up_q4tp_b,
            &c.moe_down_q4tp_b,
            &c.layout_moe_gu_b,
            &c.layout_moe_dn_b,
        )
    };
    let bind_r = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &c.moe_route.get_bind_group_layout(0),
        entries: &[
            bind_buf(0, logits),
            bind_buf(1, &bs),
            bind_buf(2, &mk),
            bind_buf(3, &fc),
            bind_buf(4, msel),
            bind_buf(5, mwt),
            bind_buf(6, mcnt),
            bind_buf(7, &rp),
            bind_buf(8, &frame_buf(c, 26, n_pack.max(1) * 4, true)),
            bind_buf(9, &frame_buf(c, 27, 2 * g.top_k * 4, false)),
        ],
    });
    let bg_gu = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: l_gu,
        entries: &[
            bind_buf(0, gate_all),
            bind_buf(1, up_all),
            bind_buf(2, x),
            bind_buf(3, msel),
            bind_buf(4, mact),
            bind_buf(5, &gu_u),
        ],
    });
    let bg_dn = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: l_dn,
        entries: &[
            bind_buf(0, down_all),
            bind_buf(1, mact),
            bind_buf(2, msel),
            bind_buf(3, mwt),
            bind_buf(4, out),
            bind_buf(5, &dn_u),
        ],
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.moe_route);
    pass.set_bind_group(0, &bind_r, &[]);
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(p_gu);
    pass.set_bind_group(0, &bg_gu, &[]);
    pass.dispatch_workgroups(g.inter as u32, slots as u32, 1);
    pass.set_pipeline(p_dn);
    pass.set_bind_group(0, &bg_dn, &[]);
    pass.dispatch_workgroups(g.hidden as u32, 1, 1);
}

/// Everything one DeepSeek-V4 layer does, in ONE submission.
///
/// The two frames before this cost two barriers a layer — 30 ms of a 76 ms
/// token, spent waiting rather than computing. Fusing them means the
/// hyper-connection glue between the halves has to run on the device too,
/// which is what `hc_pre_fold` (mixes, Sinkhorn and the fold in one kernel)
/// and `hc_post_expand` were built for.
///
/// The frame is shifted by one on purpose: it ENDS by folding and norming the
/// state for the NEXT layer's attention half and projecting its LoRA vector,
/// then reads back that normed hidden. The host needs exactly that one vector
/// — for the kv projection, the compressor and the indexer, which still live
/// there — and nothing else. Layer zero's opening fold is done on the host
/// once, which costs nothing at all.
pub struct Dsv4LayerW<'a> {
    pub attn: Dsv4AttnW<'a>,
    pub moe: Dsv4MoeW<'a>,
    /// Hyper-connection projection of the FFN half: `[mix_hc, hc*dim]` f32.
    pub hc_ffn_fn: &'a [f32],
    pub hc_ffn_scale: &'a [f32; 3],
    pub hc_ffn_base: &'a [f32],
    /// The same for the NEXT layer's attention half — absent on the last.
    pub hc_next_fn: Option<&'a [f32]>,
    pub hc_next_scale: &'a [f32; 3],
    pub hc_next_base: &'a [f32],
    pub ffn_norm: &'a [f32],
    /// The next layer's input norm and its q_norm, for the tail that
    /// prepares the following frame.
    pub next_norm: &'a [f32],
    pub next_q_norm: &'a [f32],
    /// The next layer's wq_a, by directory index.
    pub next_wq_a: Option<usize>,
    /// Router logits weight, f32 `[n_exp, dim]`.
    pub router: &'a [f32],
}

#[derive(Clone, Copy)]
pub struct Dsv4LayerGeom {
    pub attn: Dsv4AttnGeom,
    pub moe: Dsv4MoeGeom,
    pub hc: usize,
    pub hc_eps: f32,
    pub sinkhorn_iters: usize,
}

/// Returns the next layer's normed hidden in `folded_next` (or this layer's
/// state contribution when there is no next layer).
#[allow(clippy::too_many_arguments)]
pub fn dsv4_layer_frame(
    model: &Arc<CmfModel>,
    w: &Dsv4LayerW,
    g: Dsv4LayerGeom,
    kv_id: u64,
    li: usize,
    // `None` uses the LoRA vector the previous frame left on the card — the
    // host never sees it and never projects it.
    qn: Option<&[f32]>,
    idxs: &[u32],
    inv_freq: &[f32],
    pos: usize,
    folded_next: &mut [f32],
) -> bool {
    macro_rules! no {
        ($($t:tt)*) => {{
            if std::env::var("CMF_DSV4_FRAME_DEBUG").is_ok() {
                eprintln!("кадр слоя отклонён: {}", format_args!($($t)*));
            }
            return false;
        }};
    }
    let Some(c) = ctx() else { no!("нет контекста wgpu") };
    let a = g.attn;
    let m = g.moe;
    let (hc, dim) = (g.hc, a.dim);
    let mix_hc = (2 + hc) * hc;
    if qn.is_some_and(|v| v.len() < a.q_lora)
        || idxs.is_empty()
        || idxs.len() > 1024
        || folded_next.len() < dim
    {
        no!("формы: idx {} out {}", idxs.len(), folded_next.len());
    }
    if w.hc_ffn_fn.len() < mix_hc * hc * dim || w.router.len() < m.hidden {
        no!("гипер-связи или роутер не той формы");
    }

    // ── weights ──
    let bytes = model.primary_bytes();
    let mut wb = Vec::with_capacity(5);
    for &idx in &[
        w.attn.wq_a,
        w.attn.wq_b,
        w.attn.wo_a,
        w.attn.wo_b,
        w.next_wq_a.unwrap_or(w.attn.wq_a),
    ] {
        let Some(e) = model.tensors.get(idx) else {
            no!("тензора {idx} нет");
        };
        if e.dtype != cortiq_core::TensorDtype::Q4TiledP {
            no!("{} не q4tp", e.name);
        }
        let (Some(abs), plen) = (model.entry_abs_offset(e), e.nbytes as usize) else {
            no!("{} без смещения", e.name);
        };
        let Some(b) = weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen])
        else {
            no!("{} не влез в VRAM", e.name);
        };
        wb.push(b);
    }
    let Some((gate_all, up_all, down_all)) = moe_expert_bufs(
        c,
        model,
        w.moe.experts,
        m.inter,
        m.hidden,
        true,
        m.gu_q2,
    ) else {
        no!("эксперты не влезли в VRAM");
    };
    let cache = {
        let map = c.dsv4_kv.lock().unwrap();
        match map.get(&(kv_id, li)) {
            Some((b, _)) => b.clone(),
            None => no!("кеш ({kv_id}, {li}) не заведён"),
        }
    };
    // The hyper-connection state lives on the card for the whole token; the
    // host seeds it once at layer zero.
    let state = frame_buf(c, 40, hc * dim * 4, true);

    // ── constants (model-owned, address keying is sound) ──
    let qnw = const_buf(c, bytemuck::cast_slice(&w.attn.q_norm[..a.q_lora]));
    let sink = const_buf(c, bytemuck::cast_slice(&w.attn.sink[..a.nh]));
    let freq = const_buf(c, bytemuck::cast_slice(&inv_freq[..a.rd / 2]));
    let ffn_fn = const_buf(c, bytemuck::cast_slice(w.hc_ffn_fn));
    let ffn_sc = const_buf(c, bytemuck::cast_slice(w.hc_ffn_scale));
    let ffn_bs = const_buf(c, bytemuck::cast_slice(&w.hc_ffn_base[..mix_hc]));
    let ffn_nw = const_buf(c, bytemuck::cast_slice(&w.ffn_norm[..dim]));
    let n_exp = w.moe.experts.len().saturating_sub(1);
    if w.router.len() < m.hidden * n_exp {
        no!("роутер короче {} × {}", n_exp, m.hidden);
    }
    let router = const_buf(c, bytemuck::cast_slice(&w.router[..m.hidden * n_exp]));
    let next_nw = const_buf(c, bytemuck::cast_slice(&w.next_norm[..dim]));
    let next_qn = const_buf(c, bytemuck::cast_slice(&w.next_q_norm[..a.q_lora]));

    // ── per-call uploads ──
    let posb = frame_up(c, 1, bytemuck::cast_slice(&[pos as f32, a.eps]));
    let ixb = {
        let cap = idxs.len().next_power_of_two().max(64);
        let b = frame_buf(c, 2, cap * 4, true);
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(idxs));
        b
    };
    let qnb = match qn {
        Some(v) => frame_up(c, 4, bytemuck::cast_slice(&v[..a.q_lora])),
        None => frame_buf(c, 4, a.q_lora * 4, true),
    };

    // ── working buffers ──
    let n_pack = n_exp;
    let slots = m.top_k + 1;
    let q = frame_buf(c, 5, a.nh * a.hd * 4, false);
    let attn = frame_buf(c, 6, a.nh * a.hd * 4, false);
    let mid = frame_buf(c, 7, a.o_groups * a.o_lora * 4, false);
    let ao = frame_buf(c, 8, dim * 4, false);
    let mixes = frame_buf(c, 41, mix_hc * 4, false);
    let folded = frame_buf(c, 42, dim * 4, false);
    let hpost = frame_buf(c, 43, hc * 4, true);
    let hcomb = frame_buf(c, 44, hc * hc * 4, true);
    let x2 = frame_buf(c, 45, dim * 4, false);
    let state2 = frame_buf(c, 46, hc * dim * 4, false);
    let logit_b = frame_buf(c, 47, n_pack * 4, false);
    let msel = frame_buf(c, 19, slots * 4, false);
    let mwt = frame_buf(c, 20, slots * 4, false);
    let mcnt = frame_buf(c, 21, 4, false);
    let mact = frame_buf(c, 22, slots * m.inter * 4, false);
    let mo = frame_buf(c, 24, dim * 4, false);
    let qr2 = frame_buf(c, 48, a.q_lora * 4, false);
    let qn2 = frame_buf(c, 49, a.q_lora * 4, false);

    let hcp = uniform_mixed(
        c,
        [hc as u32, dim as u32, g.sinkhorn_iters as u32],
        g.hc_eps,
    );

    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dsv4-layer"),
        });

    // ── attention half (the fold for it was prepared by the previous frame) ──
    encode_attn_chain(c, &mut enc, &wb, &qnb, &q, &attn, &mid, &ao, &cache, &ixb,
                      &sink, &freq, &posb, a, kv_id, li, idxs.len());

    // ── glue: expand, then the FFN half's fold and norm ──
    encode_hc_expand(c, &mut enc, &ao, &state, &hpost, &hcomb, &state2, &hcp, hc, dim);
    encode_f32matvec(c, &mut enc, &ffn_fn, &state2, &mixes, mix_hc, hc * dim);
    encode_hc_fold(c, &mut enc, &state2, &mixes, &ffn_sc, &ffn_bs, &folded, &hpost, &hcomb, &hcp);
    encode_rmsnorm(c, &mut enc, &folded, &ffn_nw, &x2, dim, a.eps, (50, kv_id, li));

    // ── MoE half ──
    encode_f32matvec(c, &mut enc, &router, &x2, &logit_b, n_pack, m.hidden);
    encode_moe_chain(c, &mut enc, &logit_b, &x2, &msel, &mwt, &mcnt, &mact, &mo,
                     &gate_all, &up_all, &down_all, w, m, n_pack, slots);

    // ── expand, then prepare the NEXT layer ──
    encode_hc_expand(c, &mut enc, &mo, &state2, &hpost, &hcomb, &state, &hcp, hc, dim);
    if let Some(nf) = w.hc_next_fn {
        let nfn = const_buf(c, bytemuck::cast_slice(nf));
        let nsc = const_buf(c, bytemuck::cast_slice(w.hc_next_scale));
        let nbs = const_buf(c, bytemuck::cast_slice(&w.hc_next_base[..mix_hc]));
        encode_f32matvec(c, &mut enc, &nfn, &state, &mixes, mix_hc, hc * dim);
        encode_hc_fold(c, &mut enc, &state, &mixes, &nsc, &nbs, &folded, &hpost, &hcomb, &hcp);
        encode_rmsnorm(c, &mut enc, &folded, &next_nw, &x2, dim, a.eps, (51, kv_id, li));
        // The next layer's LoRA vector, but only when the host is not going
        // to hand it over anyway — the indexer needs `qr` there, so today it
        // projects it regardless and computing it twice is waste.
        if qn.is_none() {
            encode_q4tp_mv1(c, &mut enc, &wb[4], &x2, &qr2, a.q_lora, dim, (52, kv_id, li));
            encode_rmsnorm(c, &mut enc, &qr2, &next_qn, &qn2, a.q_lora, a.eps, (53, kv_id, li));
            enc.copy_buffer_to_buffer(&qn2, 0, &qnb, 0, (a.q_lora * 4) as u64);
        }
    }

    let src = if w.hc_next_fn.is_some() { &x2 } else { &ao };
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (dim * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "dsv4-layer-stage",
    );
    let ok = readback(c, enc, src, &stage, (dim * 4) as u64, &mut folded_next[..dim]);
    drop(sc);
    ok
}

/// Is this tensor resident, or can it be made so? Uploads it if it can.
pub fn dsv4_weight_ready(model: &Arc<CmfModel>, idx: usize) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(e) = model.tensors.get(idx) else {
        return false;
    };
    let Some(abs) = model.entry_abs_offset(e) else {
        return false;
    };
    let bytes = model.primary_bytes();
    let plen = e.nbytes as usize;
    if abs + plen > bytes.len() {
        return false;
    }
    weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen]).is_some()
}

/// How many experts of this shape still fit on the card. The caller packs
/// that many and leaves the rest to the host — per EXPERT, so no layer ever
/// has to leave the device wholesale.
pub fn dsv4_experts_fit(inter: usize, hidden: usize, gu_q2: bool) -> usize {
    use std::sync::atomic::Ordering;
    let Some(c) = ctx() else { return 0 };
    let gu = if gu_q2 {
        cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q2TiledP, &[inter, hidden])
    } else {
        cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[inter, hidden])
    }
    .unwrap_or(0);
    let dn = cortiq_core::quant::expected_nbytes(
        cortiq_core::TensorDtype::Q4TiledP,
        &[hidden, inter],
    )
    .unwrap_or(0);
    let per = (2 * gu + dn) as u64;
    if per == 0 {
        return 0;
    }
    let used = c.resident.load(Ordering::Relaxed);
    ((c.vram_budget.saturating_sub(used)) / per) as usize
}

/// Can this layer's experts live on the card? Uploads them if they can, so a
/// caller that pre-flights every layer has also paid the upload before it
/// commits to the device path.
/// Ask for the attention weights BEFORE the experts, or the experts take the
/// card and the skeleton — two orders of magnitude smaller — has nowhere left.
pub fn dsv4_experts_ready(
    model: &Arc<CmfModel>,
    experts: &[(usize, usize, usize)],
    inter: usize,
    hidden: usize,
    gu_q2: bool,
) -> bool {
    let Some(c) = ctx() else { return false };
    moe_expert_bufs(c, model, experts, inter, hidden, true, gu_q2).is_some()
}

/// Seed the attention half's `post`/`comb` from the host. The frame's opening
/// expand reads what the PREVIOUS frame's tail left there; layer zero has no
/// previous frame, and neither does the layer after one that ran on the host.
/// Without this both read whatever was in the buffer — perplexity 1470.
pub fn dsv4_hc_write(post: &[f32], comb: &[f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let pb = frame_buf(c, 43, post.len() * 4, true);
    let cb = frame_buf(c, 44, comb.len() * 4, true);
    c.queue.write_buffer(&pb, 0, bytemuck::cast_slice(post));
    c.queue.write_buffer(&cb, 0, bytemuck::cast_slice(comb));
    true
}

/// Seed the layer-frame's hyper-connection state from the host (layer zero).
pub fn dsv4_state_write(state: &[f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let b = frame_buf(c, 40, state.len() * 4, true);
    c.queue.write_buffer(&b, 0, bytemuck::cast_slice(state));
    true
}

/// Read the hyper-connection state back (end of token, for the head).
pub fn dsv4_state_read(state: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let b = frame_buf(c, 40, state.len() * 4, true);
    let bytes = (state.len() * 4) as u64;
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "dsv4-state-stage",
    );
    let enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("st") });
    let ok = readback(c, enc, &b, &stage, bytes, state);
    drop(sc);
    ok
}

/// MoE routing on the device: scores in, chosen experts and their normalised
/// weights out, in selection order. `forced` is the hash layers' table row.
#[allow(clippy::too_many_arguments)]
pub fn moe_route_for_test(
    scores: &[f32],
    bias: Option<&[f32]>,
    mask: Option<&[bool]>,
    forced: Option<&[usize]>,
    top_k: usize,
    route_scale: f32,
    // Pin the shared expert in slot `top_k` at weight 1, and write every slot
    // — the `msel`/`mwt` pair the batched expert kernels take. Off returns
    // just what the router chose.
    shared_slot: bool,
    idx_out: &mut Vec<usize>,
    w_out: &mut Vec<f32>,
) -> bool {
    let Some(c) = ctx() else { return false };
    let n = scores.len();
    if n == 0 || n > 1024 || top_k == 0 || top_k > 64 {
        return false;
    }
    let sb = storage_bytes(c, bytemuck::cast_slice(scores));
    let bb = match bias {
        Some(b) if b.len() >= n => storage_bytes(c, bytemuck::cast_slice(&b[..n])),
        _ => sb.clone(),
    };
    let mb = match mask {
        Some(m) if m.len() >= n => {
            let v: Vec<u32> = m[..n].iter().map(|&x| x as u32).collect();
            storage_bytes(c, bytemuck::cast_slice(&v))
        }
        _ => storage_bytes(c, bytemuck::cast_slice(&vec![1u32; n])),
    };
    let fb = match forced {
        Some(f) if f.len() >= top_k => {
            let v: Vec<u32> = f[..top_k].iter().map(|&x| x as u32).collect();
            storage_bytes(c, bytemuck::cast_slice(&v))
        }
        _ => storage_bytes(c, bytemuck::cast_slice(&vec![0u32; top_k])),
    };
    let slots = top_k + shared_slot as usize;
    let ib = rw_f32(c, slots, true);
    let wb = rw_f32(c, slots, true);
    let cb = rw_f32(c, 1, true);
    let rmb = storage_bytes(c, bytemuck::cast_slice(&vec![0u32; n]));
    let coldb = rw_f32(c, 4 * top_k, false);
    let flags = (bias.is_some_and(|b| b.len() >= n) as u32)
        | ((mask.is_some_and(|m| m.len() >= n) as u32) << 1)
        | ((forced.is_some_and(|f| f.len() >= top_k) as u32) << 2)
        | ((shared_slot as u32) << 3)
        // Nothing is packed away here, so the shared expert sits at n — the
        // same slot the frame computes from its packing.
        | ((n as u32) << 8);
    let p = uniform_mixed(c, [n as u32, top_k as u32, flags], route_scale);
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("route") });
    {
        let layout = c.moe_route.get_bind_group_layout(0);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                bind_buf(0, &sb),
                bind_buf(1, &bb),
                bind_buf(2, &mb),
                bind_buf(3, &fb),
                bind_buf(4, &ib),
                bind_buf(5, &wb),
                bind_buf(6, &cb),
                bind_buf(7, &p),
                // The router grew a remap and a winners buffer; a standalone
                // caller that skips them is a validation error, not a
                // silently different answer.
                bind_buf(8, &rmb),
                bind_buf(9, &coldb),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.moe_route);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    // idx | weights | count, one map.
    let bytes = ((2 * slots + 1) * 4) as u64;
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "route-stage",
    );
    enc.copy_buffer_to_buffer(&ib, 0, &stage, 0, (slots * 4) as u64);
    enc.copy_buffer_to_buffer(&wb, 0, &stage, (slots * 4) as u64, (slots * 4) as u64);
    enc.copy_buffer_to_buffer(&cb, 0, &stage, (2 * slots * 4) as u64, 4);
    c.queue.submit(Some(enc.finish()));
    let slice = stage.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    let mut ok = false;
    if let Ok(data) = slice.get_mapped_range() {
        let words: &[u32] = bytemuck::cast_slice(&data[..bytes as usize]);
        let ws: &[f32] = bytemuck::cast_slice(&data[slots * 4..2 * slots * 4]);
        // With a shared slot the caller wants every slot, filled or not; the
        // count is what the kernels use to skip nothing.
        let take = if shared_slot {
            slots
        } else {
            (words[2 * slots] as usize).min(top_k)
        };
        idx_out.clear();
        w_out.clear();
        idx_out.extend(words[..take].iter().map(|&x| x as usize));
        w_out.extend_from_slice(&ws[..take]);
        ok = true;
    }
    stage.unmap();
    drop(sc);
    ok
}

fn encode_rmsnorm(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    o: &wgpu::Buffer,
    n: usize,
    eps: f32,
    bkey: (u8, u64, usize),
) {
    let bind = cached_bind(c, bkey, || {
        let p = uniform_u32x4(c, [n as u32, 0, eps.to_bits(), 0]);
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.rmsnorm.get_bind_group_layout(0),
            entries: &[
                bind_buf(0, x),
                bind_buf(1, w),
                bind_buf(2, o),
                bind_buf(3, &p),
            ],
        })
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.rmsnorm);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(1, 1, 1);
}

#[allow(clippy::too_many_arguments)]
fn encode_rope_heads(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    freq: &wgpu::Buffer,
    posb: &wgpu::Buffer,
    nh: usize,
    hd: usize,
    rd: usize,
    rms: bool,
    inverse: bool,
    bkey: (u8, u64, usize),
) {
    let bind = cached_bind(c, bkey, || {
        let flags = (rms as u32) | ((inverse as u32) << 1);
        let p = uniform_u32x4(c, [nh as u32, hd as u32, rd as u32, flags]);
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.rope_heads.get_bind_group_layout(0),
            entries: &[
                bind_buf(0, x),
                bind_buf(1, freq),
                bind_buf(2, &p),
                bind_buf(3, posb),
            ],
        })
    });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&c.rope_heads);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(nh as u32, 1, 1);
}

/// A constant vector (a norm weight, the sinks, the frequency table) parked
/// on the card and keyed on its host address — the same bytes arrive every
/// token, and re-uploading them 43 times a token is pure waste.
fn const_buf(c: &Ctx, data: &[u8]) -> wgpu::Buffer {
    let key = (data.as_ptr() as usize, data.len());
    let mut m = c.const_bufs.lock().unwrap();
    if let Some(b) = m.get(&key) {
        return b.clone();
    }
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dsv4-const"),
        size: data.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue.write_buffer(&b, 0, data);
    m.insert(key, b.clone());
    b
}

/// A frame working buffer, created on first use and reused for good. `tag`
/// separates roles that happen to share a length — two buffers of the same
/// size are not interchangeable when both are live in one encoder.
fn frame_buf(c: &Ctx, tag: u8, len_bytes: usize, upload: bool) -> wgpu::Buffer {
    let mut m = c.dsv4_scratch.lock().unwrap();
    if let Some(b) = m.get(&(tag, len_bytes)) {
        return b.clone();
    }
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    if upload {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dsv4-frame"),
        size: len_bytes.max(4) as u64,
        usage,
        mapped_at_creation: false,
    });
    m.insert((tag, len_bytes), b.clone());
    b
}

/// Upload into a reused buffer instead of minting one per call.
fn frame_up(c: &Ctx, tag: u8, data: &[u8]) -> wgpu::Buffer {
    let b = frame_buf(c, tag, data.len(), true);
    c.queue.write_buffer(&b, 0, data);
    b
}

fn sa_split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_SA_SPLIT").is_ok_and(|v| v != "0"))
}

/// The two-dispatch sparse attention: scores per head, then the weighted sum
/// over nh*hd independent outputs. Same numbers as the one-workgroup-per-head
/// kernel, spread across the card instead of 64 groups of it — which measured
/// 0.54 ms a layer and was the whole cost of the block once its encoding was
/// cached away.
#[allow(clippy::too_many_arguments)]
fn encode_sparse_attend2(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    q: &wgpu::Buffer,
    kv: &wgpu::Buffer,
    ixb: &wgpu::Buffer,
    sink: &wgpu::Buffer,
    out: &wgpu::Buffer,
    nh: usize,
    hd: usize,
    m: usize,
    scale: f32,
) {
    // ONE workgroup per head after all. The split into scores + apply spread
    // the work across the card and bought 0.54 -> 0.49 ms a layer — nothing —
    // while moving the model's perplexity by 0.7% through a different
    // accumulation order. Faster would have justified that; a wash does not.
    // CMF_SA_SPLIT=1 runs the split pair for anyone who wants to retry it on
    // a part where occupancy actually bites.
    if !sa_split() {
        let p = uniform_mixed(c, [nh as u32, hd as u32, m as u32], scale);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.sparse_attend.get_bind_group_layout(0),
            entries: &[
                bind_buf(0, q),
                bind_buf(1, kv),
                bind_buf(2, ixb),
                bind_buf(3, sink),
                bind_buf(4, out),
                bind_buf(5, &p),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.sparse_attend);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(nh as u32, 1, 1);
        return;
    }
    let wbuf = frame_buf(c, 9, nh * m.max(1) * 4, false);
    let p = uniform_mixed(c, [nh as u32, hd as u32, m as u32], scale);
    {
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.sa_scores.get_bind_group_layout(0),
            entries: &[
                bind_buf(0, q),
                bind_buf(1, kv),
                bind_buf(2, ixb),
                bind_buf(3, sink),
                bind_buf(4, &wbuf),
                bind_buf(5, &p),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.sa_scores);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(nh as u32, 1, 1);
    }
    {
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &c.sa_apply.get_bind_group_layout(0),
            entries: &[
                bind_buf(0, &wbuf),
                bind_buf(1, kv),
                bind_buf(2, ixb),
                bind_buf(3, out),
                bind_buf(4, &p),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.sa_apply);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(((nh * hd) as u32).div_ceil(256), 1, 1);
    }
}

/// The one thing that has to survive between tokens. `off` and `data` are in
/// floats; the buffer is created on first use at `cap` and never shrinks.
pub fn dsv4_cache_write(kv_id: u64, li: usize, off: usize, data: &[f32], cap: usize) -> bool {
    let dbg = std::env::var("CMF_DSV4_FRAME_DEBUG").is_ok();
    let Some(c) = ctx() else {
        if dbg {
            eprintln!("кеш dsv4: нет контекста wgpu");
        }
        return false;
    };
    if off + data.len() > cap {
        if dbg {
            eprintln!("кеш dsv4: {off}+{} не влезает в {cap}", data.len());
        }
        return false;
    }
    // Storage buffers have a size ceiling of their own, well under VRAM, and
    // silently refusing at it reads as "no device" from the caller's side.
    if (cap * 4) as u64 > c.device.limits().max_storage_buffer_binding_size as u64 {
        tracing::warn!(
            "кеш dsv4: {} МБ превышает предел одного буфера {} МБ — слой остаётся на CPU",
            cap * 4 / (1 << 20),
            c.device.limits().max_storage_buffer_binding_size / (1 << 20)
        );
        return false;
    }
    let mut map = c.dsv4_kv.lock().unwrap();
    // Grow rather than refuse: the compressed axis lengthens as the sequence
    // does, and a buffer sized for token 100 is not a reason to fall off the
    // device at token 1000. The caller rewrites both regions each token, so
    // losing the old contents costs nothing.
    if map.get(&(kv_id, li)).is_some_and(|(_, have)| *have < cap) {
        map.remove(&(kv_id, li));
        GREW.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let e = map.entry((kv_id, li)).or_insert_with(|| {
        (
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dsv4-kv"),
                size: (cap * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cap,
        )
    });
    if e.1 < off + data.len() {
        return false; // a longer context than the cache was built for
    }
    if !data.is_empty() {
        c.queue
            .write_buffer(&e.0, (off * 4) as u64, bytemuck::cast_slice(data));
    }
    true
}

/// Bind groups for the dsv4 frames, keyed by role and layer. Their buffers
/// are pooled and stable between tokens, so building 11 of them per layer per
/// token — 473 a token on the release — was pure host overhead. The epoch
/// invalidates the lot whenever a pooled buffer is rebuilt underneath them.
fn cached_bind<F>(c: &Ctx, key: (u8, u64, usize), build: F) -> wgpu::BindGroup
where
    F: FnOnce() -> wgpu::BindGroup,
{
    use std::sync::atomic::Ordering;
    let epoch = GREW.load(Ordering::Relaxed);
    let mut m = c.dsv4_binds.lock().unwrap();
    if m.0 != epoch {
        m.0 = epoch;
        m.1.clear();
    }
    if let Some(b) = m.1.get(&key) {
        return b.clone();
    }
    let b = build();
    m.1.insert(key, b.clone());
    b
}

/// Bumped whenever a cache buffer is reallocated. A caller that writes only
/// the tail has to notice, because the new buffer holds nothing.
pub static GREW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Drop a conversation's caches (a new sequence, or the pipeline resetting).
pub fn dsv4_cache_clear(kv_id: u64) {
    if let Some(c) = ctx() {
        c.dsv4_kv.lock().unwrap().retain(|k, _| k.0 != kv_id);
    }
}

/// The quantized tensors one DeepSeek-V4 attention block reads, by directory
/// index, plus the two small f32 vectors it needs whole.
pub struct Dsv4AttnW<'a> {
    pub wq_a: usize,
    pub wq_b: usize,
    pub wo_a: usize,
    pub wo_b: usize,
    pub q_norm: &'a [f32],
    pub sink: &'a [f32],
}

/// Shapes for one block. Separate from the weights so the caller can build it
/// once per layer and keep it.
#[derive(Clone, Copy)]
pub struct Dsv4AttnGeom {
    pub dim: usize,
    pub nh: usize,
    pub hd: usize,
    pub rd: usize,
    pub q_lora: usize,
    pub o_lora: usize,
    pub o_groups: usize,
    pub eps: f32,
    pub scale: f32,
}

/// DeepSeek-V4's attention block, start to finish, in ONE submission.
///
/// Eight operations that were eight round trips: the query LoRA and its two
/// norms, the rope tail, attention over the index list, the inverse rope, and
/// the grouped output projection. Nothing between them touches the host — the
/// intermediate vectors never leave the card, and the KV cache is already
/// there.
///
/// The kv vector itself stays on the CPU deliberately: the compressor's
/// pending windows are host state, and pulling one 512-wide vector back is
/// cheaper than moving that state too. That is the next frame to fuse, not a
/// thing forgotten here.
#[allow(clippy::too_many_arguments)]
pub fn dsv4_attn_frame(
    model: &Arc<CmfModel>,
    w: &Dsv4AttnW,
    g: Dsv4AttnGeom,
    hidden: &[f32],
    // The layer's own `q_norm(wq_a(x))`, when the caller already has it — the
    // indexer needs that vector on the host anyway, and computing it twice is
    // worse than uploading 1536 floats. `None` puts both ops in the frame.
    qn_in: Option<&[f32]>,
    kv_id: u64,
    li: usize,
    idxs: &[u32],
    inv_freq: &[f32],
    pos: usize,
    out: &mut [f32],
) -> bool {
    // A refusal used to be a silent `false`, and three of them in a row cost
    // an evening of guessing which guard had fired.
    macro_rules! no {
        ($($t:tt)*) => {{
            tracing::debug!("кадр dsv4 отклонён: {}", format_args!($($t)*));
            if std::env::var("CMF_DSV4_FRAME_DEBUG").is_ok() {
                eprintln!("кадр dsv4 отклонён: {}", format_args!($($t)*));
            }
            return false;
        }};
    }
    let Some(c) = ctx() else { no!("нет контекста wgpu") };
    // `hidden` is only read when the frame has to build the LoRA vector
    // itself; demanding it regardless refused every caller that had one.
    if (qn_in.is_none() && hidden.len() < g.dim)
        || out.len() < g.dim
        || w.sink.len() < g.nh
        || w.q_norm.len() < g.q_lora
        || idxs.is_empty()
        || idxs.len() > 1024
        || inv_freq.len() * 2 < g.rd
    {
        no!(
            "формы: hidden {} dim {} out {} sink {} nh {} q_norm {} q_lora {} idx {} freq {} rd {}",
            hidden.len(), g.dim, out.len(), w.sink.len(), g.nh,
            w.q_norm.len(), g.q_lora, idxs.len(), inv_freq.len(), g.rd
        );
    }
    let bytes = model.primary_bytes();
    // Every weight q4tp, or the frame declines: a mixed layer would need the
    // per-op branches back and this is not the place to guess a layout.
    let mut wb = Vec::with_capacity(4);
    for &idx in &[w.wq_a, w.wq_b, w.wo_a, w.wo_b] {
        let Some(e) = model.tensors.get(idx) else {
            no!("тензора {idx} нет в каталоге");
        };
        if e.dtype != cortiq_core::TensorDtype::Q4TiledP || e.shape.len() != 2 {
            no!("{} не q4tp ({:?}, {:?})", e.name, e.dtype, e.shape);
        }
        let Some(abs) = model.entry_abs_offset(e) else {
            no!("{} без абсолютного смещения", e.name);
        };
        let plen = e.nbytes as usize;
        if abs + plen > bytes.len() {
            no!("{} выходит за файл", e.name);
        }
        let Some(b) = weight_buffer(c, (bytes.as_ptr() as usize, idx), &bytes[abs..abs + plen])
        else {
            no!("{} не поместился в бюджет VRAM", e.name);
        };
        wb.push(b);
    }
    let cache = {
        let map = c.dsv4_kv.lock().unwrap();
        match map.get(&(kv_id, li)) {
            Some((b, _)) => b.clone(),
            None => no!("кеш ({kv_id}, {li}) не заведён"),
        }
    };

    if let Some(v) = qn_in {
        if v.len() < g.q_lora {
            no!("готовый qn короче q_lora: {} < {}", v.len(), g.q_lora);
        }
    }
    // Constants (q_norm, sink, inv_freq) go through the const cache keyed on
    // their address — they are the same bytes every token. Everything else is
    // a reused buffer written in place.
    let hb = match qn_in {
        None => frame_up(c, 0, bytemuck::cast_slice(&hidden[..g.dim])),
        Some(_) => frame_buf(c, 0, 4, true),
    };
    // These three ARE model-owned and outlive the run, so address keying is
    // sound for them — unlike anything built per call.
    let qnw = const_buf(c, bytemuck::cast_slice(&w.q_norm[..g.q_lora]));
    let sink = const_buf(c, bytemuck::cast_slice(&w.sink[..g.nh]));
    let freq = const_buf(c, bytemuck::cast_slice(&inv_freq[..g.rd / 2]));
    let posb = frame_up(c, 1, bytemuck::cast_slice(&[pos as f32, g.eps]));
    // The list length changes token to token; round the buffer up so it is
    // not reallocated on every step, and pass the true count in the uniform.
    let ixb = {
        let cap = idxs.len().next_power_of_two().max(64);
        let b = frame_buf(c, 2, cap * 4, true);
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(idxs));
        b
    };

    // Readable, all of them: `CMF_DSV4_FRAME_TAP` reads back an intermediate
    // instead of the output. Eight verified kernels can still be wired wrong,
    // and a single number at the end says only that they were.
    let qr = frame_buf(c, 3, g.q_lora * 4, false);
    let qn = match qn_in {
        Some(v) => frame_up(c, 4, bytemuck::cast_slice(&v[..g.q_lora])),
        None => frame_buf(c, 4, g.q_lora * 4, true),
    };
    let q = frame_buf(c, 5, g.nh * g.hd * 4, false);
    let attn = frame_buf(c, 6, g.nh * g.hd * 4, false);
    let mid = frame_buf(c, 7, g.o_groups * g.o_lora * 4, false);
    let yb = frame_buf(c, 8, g.dim * 4, false);

    let t_all = std::time::Instant::now();
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dsv4-attn"),
        });
    if qn_in.is_none() {
        encode_q4tp_mv1(c, &mut enc, &wb[0], &hb, &qr, g.q_lora, g.dim, (30, kv_id, li));
        encode_rmsnorm(c, &mut enc, &qr, &qnw, &qn, g.q_lora, g.eps, (31, kv_id, li));
    }
    encode_q4tp_mv1(c, &mut enc, &wb[1], &qn, &q, g.nh * g.hd, g.q_lora, (32, kv_id, li));
    encode_rope_heads(
        c, &mut enc, &q, &freq, &posb, g.nh, g.hd, g.rd, true, false, (33, kv_id, li),
    );
    encode_sparse_attend2(
        c, &mut enc, &q, &cache, &ixb, &sink, &attn, g.nh, g.hd, idxs.len(), g.scale,
    );
    encode_rope_heads(
        c, &mut enc, &attn, &freq, &posb, g.nh, g.hd, g.rd, false, true, (34, kv_id, li),
    );
    {
        let rows = g.o_groups * g.o_lora;
        let cols = g.nh * g.hd / g.o_groups;
        let bind = cached_bind(c, (36, kv_id, li), || {
            let p = uniform_u32x4(c, [(cols / 32) as u32, rows as u32, g.o_lora as u32, 0]);
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &c.o_lora_a.get_bind_group_layout(0),
                entries: &[
                    bind_buf(0, &wb[2]),
                    bind_buf(1, &attn),
                    bind_buf(2, &mid),
                    bind_buf(3, &p),
                ],
            })
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.o_lora_a);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
    }
    encode_q4tp_mv1(
        c,
        &mut enc,
        &wb[3],
        &mid,
        &yb,
        g.dim,
        g.o_groups * g.o_lora,
        (35, kv_id, li),
    );

    let tap = std::env::var("CMF_DSV4_FRAME_TAP").unwrap_or_default();
    let (src, n) = match tap.as_str() {
        "qr" => (&qr, g.q_lora),
        "qn" => (&qn, g.q_lora),
        "q" => (&q, g.nh * g.hd),
        "attn" => (&attn, g.nh * g.hd),
        "mid" => (&mid, g.o_groups * g.o_lora),
        _ => (&yb, g.dim),
    };
    if out.len() < n {
        no!("отвод {tap} нуждается в {n} значениях, дано {}", out.len());
    }
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (n * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "dsv4-attn-stage",
    );
    let t_enc = std::time::Instant::now();
    let ok = readback(c, enc, src, &stage, (n * 4) as u64, &mut out[..n]);
    drop(sc);
    ATT_ENC_NS.fetch_add(
        t_enc.duration_since(t_all).as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    ATT_WAIT_NS.fetch_add(
        t_enc.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    ok
}

/// Time inside `dsv4_attn_frame`, split at the submit — the same question the
/// MoE frame already answers, asked of the block that now costs more.
pub static ATT_ENC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ATT_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One DeepSeek-V4 MoE block on the device: route, run the chosen experts and
/// the shared one, sum. One submission.
///
/// The experts arrive PACKED — a subset chosen by the host (the hot set a
/// mask keeps), shared expert last — and the router's logits arrive already
/// renumbered into that packing. That removes the whole global-to-slot remap
/// the obvious design needs, and it makes the mask implicit: an expert not in
/// the packing has no logit and cannot be chosen.
pub struct Dsv4MoeW<'a> {
    /// `(gate, up, down)` directory indices per packed expert, shared LAST.
    pub experts: &'a [(usize, usize, usize)],
    /// Router logits over the packed routed experts (shared excluded).
    pub logits: &'a [f32],
    /// noaux_tc selection bias, same numbering. Absent on the hash layers.
    pub bias: Option<&'a [f32]>,
    /// Hash-layer row, already in packed numbering.
    pub forced: Option<&'a [usize]>,
    /// global expert id -> packed slot, `u32::MAX` where the expert did not
    /// fit. When present the router ranges over ALL experts and hands the
    /// cold picks back instead of avoiding them.
    pub remap: Option<&'a [u32]>,
}

#[derive(Clone, Copy)]
pub struct Dsv4MoeGeom {
    pub hidden: usize,
    pub inter: usize,
    pub top_k: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    /// gate/up are q2tp against a q4tp down — the mixed 2-bit profile.
    pub gu_q2: bool,
}

pub fn dsv4_moe_frame(
    model: &Arc<CmfModel>,
    w: &Dsv4MoeW,
    g: Dsv4MoeGeom,
    x: &[f32],
    // `(expert, weight)` pairs the device left for the host, empty when the
    // whole packing was resident.
    cold_out: &mut Vec<(usize, f32)>,
    out: &mut [f32],
) -> bool {
    macro_rules! no {
        ($($t:tt)*) => {{
            tracing::debug!("кадр MoE отклонён: {}", format_args!($($t)*));
            if std::env::var("CMF_DSV4_FRAME_DEBUG").is_ok() {
                eprintln!("кадр MoE отклонён: {}", format_args!($($t)*));
            }
            return false;
        }};
    }
    let t_all = std::time::Instant::now();
    let Some(c) = ctx() else { no!("нет контекста wgpu") };
    let n_pack = w.experts.len().saturating_sub(1); // routed; shared is last
    let slots = g.top_k + 1;
    let n_all = w.logits.len();
    let subset = w.remap.is_some_and(|r| r.len() >= n_all) && n_all > 0;
    // ONE width for the whole routing side: the scores, the bias and the
    // uniform must agree, and they did not. The bias went in n_pack long
    // while the kernel ranked over n_all, so every index past the packing
    // boundary read the LAST bias entry — WGSL clamps an out-of-bounds read
    // rather than faulting, so it looked like a plausible number and the
    // router quietly preferred the packed experts.
    let n_route = if subset { n_all } else { n_pack };
    if n_pack == 0
        || w.logits.len() < n_pack
        || n_pack > 1024
        || g.top_k == 0
        || g.top_k > 63
        || x.len() < g.hidden
        || out.len() < g.hidden
        || g.hidden % 32 != 0
        || g.inter % 32 != 0
    {
        no!(
            "формы: упаковано {n_pack} логитов {} top_k {} hidden {} inter {}",
            w.logits.len(),
            g.top_k,
            g.hidden,
            g.inter
        );
    }
    let Some((gate_all, up_all, down_all)) =
        moe_expert_bufs(c, model, w.experts, g.inter, g.hidden, true, g.gu_q2)
    else {
        no!("эксперты не поместились в бюджет VRAM");
    };

    // ── routing, on the device, straight into the msel/mwt the kernels read ──
    let lg = frame_up(c, 16, bytemuck::cast_slice(&w.logits[..n_route]));
    // NOT const_buf: that cache is keyed on the host ADDRESS, which is only
    // meaningful for model weights that outlive the process. The bias arrives
    // in a Vec built per layer, and the allocator hands back the same address
    // layer after layer — so every layer was routed with layer zero's bias.
    // The toys never caught it because they carry no expert_bias at all.
    let bs = match w.bias {
        Some(b) if b.len() >= n_route => {
            frame_up(c, 25, bytemuck::cast_slice(&b[..n_route]))
        }
        _ => lg.clone(),
    };
    let rmb = match w.remap {
        Some(r) if subset => frame_up(c, 26, bytemuck::cast_slice(&r[..n_all])),
        _ => frame_buf(c, 26, n_all.max(1) * 4, true),
    };
    let coldb = frame_buf(c, 27, 4 * g.top_k * 4, false);
    let mk = frame_buf(c, 17, n_pack * 4, true);
    let fc = match w.forced {
        Some(f) if f.len() >= g.top_k => {
            let v: Vec<u32> = f[..g.top_k].iter().map(|&i| i as u32).collect();
            frame_up(c, 18, bytemuck::cast_slice(&v))
        }
        _ => frame_buf(c, 18, g.top_k * 4, true),
    };
    let msel = frame_buf(c, 19, slots * 4, false);
    let mwt = frame_buf(c, 20, slots * 4, false);
    let mcnt = frame_buf(c, 21, 4, false);
    let mact = frame_buf(c, 22, slots * g.inter * 4, false);
    let xb = frame_up(c, 23, bytemuck::cast_slice(&x[..g.hidden]));
    let ob = frame_buf(c, 24, g.hidden * 4, false);

    let rflags = (w.bias.is_some_and(|b| b.len() >= n_route) as u32)
        | ((w.forced.is_some_and(|f| f.len() >= g.top_k) as u32) << 2)
        | 8 // always pin the shared slot: these kernels take a fixed count
        | ((subset as u32) << 4)
        | ((n_pack as u32) << 8); // where the shared expert actually sits
    if std::env::var("CMF_DSV4_MOE_CHECK").is_ok() {
        eprintln!(
            "[маршрут] n_all={n_all} n_pack={n_pack} subset={subset} flags={rflags} \
             remap[0..12]={:?}",
            w.remap.map(|r| &r[..12.min(r.len())])
        );
    }
    // Ranking ranges over EVERY expert when the packing is a subset — that is
    // the whole point. Passing n_pack here silently turned it back into a
    // mask that also indexed the packed buffer with global ids.
    let rp = uniform_mixed(
        c,
        [n_route as u32, g.top_k as u32, rflags],
        g.route_scale,
    );

    let stride16 = |rows: usize, cols: usize, q2: bool| -> u32 {
        let dt = if q2 {
            cortiq_core::TensorDtype::Q2TiledP
        } else {
            cortiq_core::TensorDtype::Q4TiledP
        };
        (cortiq_core::quant::expected_nbytes(dt, &[rows, cols]).unwrap_or(0) / 2) as u32
    };
    let gu_u = uniform_u32x8(
        c,
        [
            (g.hidden / 32) as u32,
            g.inter as u32,
            slots as u32,
            stride16(g.inter, g.hidden, g.gu_q2),
            g.swiglu_limit.to_bits(),
            0,
            0,
            0,
        ],
    );
    let dn_u = uniform_u32x4(
        c,
        [
            (g.inter / 32) as u32,
            g.hidden as u32,
            slots as u32,
            stride16(g.hidden, g.inter, false),
        ],
    );
    let (p_gu, p_dn, l_gu, l_dn) = if g.gu_q2 {
        (
            &c.moe_gate_up_q2tp,
            &c.moe_down_q4tp,
            &c.layout_moe_gu_q2tp,
            &c.layout_moe_dn_q4tp,
        )
    } else {
        (
            &c.moe_gate_up_q4tp_b,
            &c.moe_down_q4tp_b,
            &c.layout_moe_gu_b,
            &c.layout_moe_dn_b,
        )
    };

    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dsv4-moe"),
        });
    // The layer's identity for the bind cache: its first expert's directory
    // index, which is unique per layer and already at hand.
    let lkey = w.experts.first().map(|e| e.0).unwrap_or(0);
    {
        // NOT cached. This group holds `rp`, a CONTENT-keyed uniform: change a
        // flag and the uniform becomes a different buffer while the cached
        // group keeps pointing at the old one — the layer then routes with
        // yesterday's flags forever. Encoding it costs 0.01 ms a layer; being
        // wrong costs a model.
        let _ = lkey;
        let bind = {
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &c.moe_route.get_bind_group_layout(0),
                entries: &[
                    bind_buf(0, &lg),
                    bind_buf(1, &bs),
                    bind_buf(2, &mk),
                    bind_buf(3, &fc),
                    bind_buf(4, &msel),
                    bind_buf(5, &mwt),
                    bind_buf(6, &mcnt),
                    bind_buf(7, &rp),
                    bind_buf(8, &rmb),
                    bind_buf(9, &coldb),
                ],
            })
        };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.moe_route);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);

        let bg_gu = cached_bind(c, (41, 0, lkey), || {
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: l_gu,
                entries: &[
                    bind_buf(0, &gate_all),
                    bind_buf(1, &up_all),
                    bind_buf(2, &xb),
                    bind_buf(3, &msel),
                    bind_buf(4, &mact),
                    bind_buf(5, &gu_u),
                ],
            })
        });
        pass.set_pipeline(p_gu);
        pass.set_bind_group(0, &bg_gu, &[]);
        pass.dispatch_workgroups(g.inter as u32, slots as u32, 1);

        let bg_dn = cached_bind(c, (42, 0, lkey), || {
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: l_dn,
                entries: &[
                    bind_buf(0, &down_all),
                    bind_buf(1, &mact),
                    bind_buf(2, &msel),
                    bind_buf(3, &mwt),
                    bind_buf(4, &ob),
                    bind_buf(5, &dn_u),
                ],
            })
        });
        pass.set_pipeline(p_dn);
        pass.set_bind_group(0, &bg_dn, &[]);
        pass.dispatch_workgroups(g.hidden as u32, 1, 1);
    }
    let t_enc = std::time::Instant::now();
    let mut sc = c.scratch.lock().unwrap();
    // The cold list rides the SAME staging buffer and the SAME fence, so the
    // host learns which picks it owes without paying a second barrier.
    //
    // This block went missing once — `cold_out` was declared, threaded all
    // the way down and never filled — and the empty list read exactly like a
    // router that had chosen no cold experts. An unconditional probe written
    // from the kernel is what proved otherwise.
    let cold_bytes = (4 * g.top_k * 4) as u64;
    let total = (g.hidden * 4) as u64 + cold_bytes;
    // ONE ensure for the whole readback. A first call sized to the hidden
    // state alone used to run before this one, on the same slot: it built a
    // buffer that the next line immediately outgrew and replaced.
    let stage2 = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        total,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "dsv4-moe-stage",
    );
    enc.copy_buffer_to_buffer(&ob, 0, &stage2, 0, (g.hidden * 4) as u64);
    enc.copy_buffer_to_buffer(&coldb, 0, &stage2, (g.hidden * 4) as u64, cold_bytes);
    c.queue.submit(Some(enc.finish()));
    let slice = stage2.slice(..total);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    let mut ok = false;
    if let Ok(data) = slice.get_mapped_range() {
        out[..g.hidden].copy_from_slice(bytemuck::cast_slice(&data[..g.hidden * 4]));
        let tail: &[u32] = bytemuck::cast_slice(&data[g.hidden * 4..total as usize]);
        cold_out.clear();
        for t in 0..g.top_k {
            if tail[2 * t] != u32::MAX {
                cold_out.push((tail[2 * t] as usize, f32::from_bits(tail[2 * t + 1])));
            }
        }
        if std::env::var("CMF_DSV4_MOE_CHECK").is_ok() {
            let picks: Vec<(u32, f32)> = (0..g.top_k)
                .map(|t| {
                    (
                        tail[2 * g.top_k + 2 * t],
                        f32::from_bits(tail[2 * g.top_k + 2 * t + 1]),
                    )
                })
                .collect();
            eprintln!("[победители карты] {picks:?}");
        }
        ok = true;
    }
    stage2.unmap();
    drop(sc);
    // Encoding and waiting are different problems with different fixes, and
    // the layer total cannot tell them apart. Costs one Instant per layer.
    MOE_ENC_NS.fetch_add(
        t_enc.duration_since(t_all).as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    MOE_WAIT_NS.fetch_add(
        t_enc.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    ok
}

/// Time inside `dsv4_moe_frame`, split at the submit. Read by the dsv4
/// profile so a slow block can be blamed on the right half.
pub static MOE_ENC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MOE_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Attention over an index list with the learned sink, on the device.
///
/// Step two of the whole-token graph. Verified against `dsv4::sparse_attend`
/// before anything depends on it: a sink that contributes to the numerator,
/// or a denominator missing its share, changes every head's output by a
/// factor that no generated text would reveal.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attend_for_test(
    q: &[f32],
    kv: &[f32],
    idxs: &[u32],
    sink: &[f32],
    scale: f32,
    nh: usize,
    hd: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if q.len() != nh * hd || out.len() != nh * hd || sink.len() != nh || idxs.len() > 1024 {
        return false;
    }
    let qb = storage_bytes(c, bytemuck::cast_slice(q));
    let kvb = storage_bytes(c, bytemuck::cast_slice(kv));
    let ib = storage_bytes(c, bytemuck::cast_slice(idxs));
    let sb = storage_bytes(c, bytemuck::cast_slice(sink));
    let ob = rw_f32(c, nh * hd, true);
    let p = uniform_u32x4(
        c,
        [nh as u32, hd as u32, idxs.len() as u32, scale.to_bits()],
    );
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sa") });
    let _ = &p; // the split path builds its own params
    encode_sparse_attend2(
        c, &mut enc, &qb, &kvb, &ib, &sb, &ob, nh, hd, idxs.len(), scale,
    );
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        (nh * hd * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "sa-stage",
    );
    let ok = readback(c, enc, &ob, &stage, (nh * hd * 4) as u64, out);
    drop(sc);
    ok
}

/// One hyper-connection join on the device: fold the copies (with the
/// Sinkhorn) and expand them back around a block output computed elsewhere.
///
/// Step one of the whole-token graph, and deliberately useless on its own —
/// it costs a submission to save none. It exists so the join can be checked
/// against the CPU before anything is built on top of it, because a
/// transposed mixing matrix or a Sinkhorn off by one iteration produces
/// output that looks entirely reasonable.
#[allow(clippy::too_many_arguments)]
pub fn hc_join_for_test(
    state: &[f32],
    mixes: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    block_out: &[f32],
    hc: usize,
    dim: usize,
    iters: u32,
    eps: f32,
    folded: &mut [f32],
    expanded: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if state.len() != hc * dim || folded.len() != dim || expanded.len() != hc * dim {
        return false;
    }
    let st = storage_bytes(c, bytemuck::cast_slice(state));
    let mx = storage_bytes(c, bytemuck::cast_slice(mixes));
    let sc = storage_bytes(c, bytemuck::cast_slice(&scale[..]));
    let bs = storage_bytes(c, bytemuck::cast_slice(base));
    let fo = rw_f32(c, dim, true);
    let po = rw_f32(c, hc, false);
    let cb = rw_f32(c, hc * hc, false);
    let params = uniform_u32x4(c, [hc as u32, dim as u32, iters, eps.to_bits()]);

    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("hc") });
    {
        let layout = c.hc_pre_fold.get_bind_group_layout(0);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                bind_buf(0, &st),
                bind_buf(1, &mx),
                bind_buf(2, &sc),
                bind_buf(3, &bs),
                bind_buf(4, &fo),
                bind_buf(5, &po),
                bind_buf(6, &cb),
                bind_buf(7, &params),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.hc_pre_fold);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let bo = storage_bytes(c, bytemuck::cast_slice(block_out));
    let ex = rw_f32(c, hc * dim, true);
    {
        let layout = c.hc_post_expand.get_bind_group_layout(0);
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                bind_buf(0, &bo),
                bind_buf(1, &st),
                bind_buf(2, &po),
                bind_buf(3, &cb),
                bind_buf(4, &ex),
                bind_buf(5, &params),
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&c.hc_post_expand);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(((hc * dim) as u32).div_ceil(256), 1, 1);
    }
    // Two readbacks because the two results have different lengths; this is
    // a check, not a hot path.
    let mut sc_lock = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc_lock.stage,
        ((hc * dim).max(dim) * 4) as u64,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "hc-stage",
    );
    if !readback(c, enc, &fo, &stage, (dim * 4) as u64, folded) {
        return false;
    }
    let enc2 = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("hc2") });
    if !readback(c, enc2, &ex, &stage, ((hc * dim) * 4) as u64, expanded) {
        return false;
    }
    drop(sc_lock);
    true
}

pub fn adapter_report() -> Vec<String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let mut out: Vec<String> =
        pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .iter()
            .map(|a| {
                let i = a.get_info();
                let l = a.limits();
                format!(
                    "{:?} | {} | {:?} | буфер до {:.1} ГБ | рабочая группа {}",
                    i.backend,
                    i.name,
                    i.device_type,
                    l.max_buffer_size as f64 / 1e9,
                    l.max_compute_workgroup_size_x
                )
            })
            .collect();
    if out.is_empty() {
        out.push("адаптеров не найдено".into());
    }
    match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    })) {
        Ok(a) => out.push(format!("выбран: {}", a.get_info().name)),
        Err(e) => out.push(format!("выбрать не удалось: {e}")),
    }
    out
}

pub fn adapter_probe() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .is_ok()
}
