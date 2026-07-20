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
// as the Metal kernel). Bit set → +x. Grid-stride over rows, 64
// threads reduce a row; np = gpr/2.
struct Q1Params { np: u32, rows: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       q1w : array<u32>;
@group(0) @binding(1) var<storage, read>       q1x : array<f32>;   // raw f32 activations
@group(0) @binding(2) var<storage, read_write> q1y : array<f32>;
@group(0) @binding(3) var<uniform>             q1p : Q1Params;

var<workgroup> partial_q1: array<f32, 64>;

fn q1_tile_sum(bits: u32, xbase: u32) -> f32 {
    var s = vec4<f32>(0.0);
    for (var j = 0u; j < 8u; j = j + 1u) {
        let nib = bits >> (j * 4u);
        let xi = xbase + j * 4u;
        let x = vec4<f32>(q1x[xi], q1x[xi + 1u], q1x[xi + 2u], q1x[xi + 3u]);
        s = s + select(-x, x,
            vec4<bool>((nib & 1u) != 0u, (nib & 2u) != 0u, (nib & 4u) != 0u, (nib & 8u) != 0u));
    }
    return s.x + s.y + s.z + s.w;
}

@compute @workgroup_size(64)
fn q1_matvec(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    var row = wid.x;
    loop {
        if (row >= q1p.rows) { break; }
        let base = row * q1p.np * 3u;
        var acc = 0.0;
        var pi = lid;
        loop {
            if (pi >= q1p.np) { break; }
            let a0 = q1w[base + pi * 3u];
            let a1 = q1w[base + pi * 3u + 1u];
            let a2 = q1w[base + pi * 3u + 2u];
            // pair words: [s0 | bits0.lo] [bits0.hi | s1] [bits1]
            let s0 = unpack2x16float(a0).x;
            let s1 = unpack2x16float(a1).y;
            let bits0 = (a0 >> 16u) | (a1 << 16u);
            let g = pi * 2u;
            acc = acc + s0 * q1_tile_sum(bits0, g * 32u)
                      + s1 * q1_tile_sum(a2, (g + 1u) * 32u);
            pi = pi + 64u;
        }
        partial_q1[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (lid < stride) { partial_q1[lid] = partial_q1[lid] + partial_q1[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u) { q1y[row] = partial_q1[0]; }
        workgroupBarrier();
        row = row + nwg.x;
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

@group(0) @binding(0) var<storage, read_write> zy  : array<f32>;
@group(0) @binding(1) var<uniform>             znp : N1;
@compute @workgroup_size(256)
fn fill_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < znp.n) { zy[i] = 0.0; }
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
    let nt = (hd + 31u) / 32u;
    var xv: array<f32, 4>;
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
    let hlf = rq_p.rd / 2u;
    let toff = hlf / 32u;
    for (var t = 0u; t < toff; t = t + 1u) {
        let i = t * 32u + lane;
        if (i < hlf) {
            let angle = f32(rq_p.pos) * rq_invf[i];
            let cc = cos(angle);
            let sfac = sin(angle);
            let x0 = xv[t];
            let x1 = xv[t + toff];
            xv[t] = x0 * cc - x1 * sfac;
            xv[t + toff] = x0 * sfac + x1 * cc;
        }
    }
    let dst_base = select((head - nh) * hd, head * hd, isq);
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) {
            if (isq) { rq_qout[dst_base + d] = xv[t]; } else { rq_k[dst_base + d] = xv[t]; }
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
var<workgroup> at_red: array<f32, 32>;
@compute @workgroup_size(32)
fn gqa_attend(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let lane = lid.x;
    if (h >= at_p.nh) { return; }
    let hd = at_p.hd;
    let kbase = (h / at_p.hpk) * at_p.cap * hd;
    let scale = 1.0 / sqrt(f32(hd));
    let nt = (hd + 31u) / 32u;
    var qv: array<f32, 4>;
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        qv[t] = select(0.0, at_q[h * hd + d] * scale, d < hd);
    }
    var m = -1e30;
    var l = 0.0;
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var p = 0u; p < at_p.n; p = p + 1u) {
        let row = kbase + p * hd;
        var partial = 0.0;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) { partial = partial + qv[t] * at_k[row + d]; }
        }
        at_red[lane] = partial;
        workgroupBarrier();
        var s = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { s = s + at_red[j]; }
        workgroupBarrier();
        let mp = max(m, s);
        let f = exp(m - mp);
        let w = exp(s - mp);
        l = l * f + w;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) { acc[t] = acc[t] * f + w * at_v[row + d]; }
        }
        m = mp;
    }
    let invl = select(0.0, 1.0 / l, l > 0.0);
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) { at_o[h * hd + d] = acc[t] * invl; }
    }
}

// q1t (ternary base-3) + q4_block matvec — reuse the q1 bindings (q1w/q1x/q1y/
// q1p) and its 4-slot layout. Weights arrive as array<u32>, so bytes come out
// with shift+mask (q1t_byte). q1p fields are reinterpreted: np=gpr, _p0=cols.
var<workgroup> partial_q1t: array<f32, 64>;
fn q1t_byte(off: u32) -> u32 {
    return (q1w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}
fn pow3t(i: u32) -> u32 {
    switch i {
        case 0u: { return 1u; }
        case 1u: { return 3u; }
        case 2u: { return 9u; }
        case 3u: { return 27u; }
        default: { return 81u; }
    }
}

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
                let code = (b / pow3t(k % 5u)) % 3u;
                var sgn = 0.0;
                if (code == 1u) { sgn = 1.0; } else if (code == 2u) { sgn = -1.0; }
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
            let sc_off = scales_off + gi * 2u;
            let sc16 = q1t_byte(sc_off) | (q1t_byte(sc_off + 1u) << 8u);
            let scale = unpack2x16float(sc16).x;
            let pk = gi * 16u;
            let xb = g * 32u;
            var gsum = 0.0;
            for (var k = 0u; k < 16u; k = k + 1u) {
                let b = q1t_byte(pk + k);
                gsum = gsum + (f32(b & 0xFu) - 8.0) * q1x[xb + k * 2u]
                            + (f32((b >> 4u) & 0xFu) - 8.0) * q1x[xb + k * 2u + 1u];
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
                    let code = (b / pow3t(p % 5u)) % 3u;
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
    silu: wgpu::ComputePipeline,
    axpy: wgpu::ComputePipeline,
    zero: wgpu::ComputePipeline,
    q1: wgpu::ComputePipeline,
    q1t: wgpu::ComputePipeline,
    q4b: wgpu::ComputePipeline,
    q1t_mm: wgpu::ComputePipeline,
    q1t_ovmm: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    attn_rope: wgpu::ComputePipeline,
    kv_append: wgpu::ComputePipeline,
    gqa_attend: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    layout_mm: wgpu::BindGroupLayout,
    layout_mmm: wgpu::BindGroupLayout,
    layout_silu: wgpu::BindGroupLayout,
    layout_axpy: wgpu::BindGroupLayout,
    layout_zero: wgpu::BindGroupLayout,
    layout_q1: wgpu::BindGroupLayout,
    layout_rmsnorm: wgpu::BindGroupLayout,
    layout_attn_rope: wgpu::BindGroupLayout,
    layout_kv: wgpu::BindGroupLayout,
    layout_attend: wgpu::BindGroupLayout,
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

static CTX: OnceLock<Option<Ctx>> = OnceLock::new();

/// Whether the wgpu path is selected by env (the facade asks before `enabled()`):
/// `CMF_GPU=wgpu` — always; `CMF_GPU=1` (≠0) — only on non-macOS, where
/// there is no native Metal (on macOS `=1` goes to Metal).
pub fn selected() -> bool {
    match std::env::var("CMF_GPU") {
        Ok(v) if v == "wgpu" => true,
        Ok(v) if v != "0" => !cfg!(target_os = "macos"),
        _ => false,
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
    let silu = pipe("silu_mul_pre");
    let axpy = pipe("axpy");
    let zero = pipe("fill_zero");
    let q1 = pipe("q1_matvec");
    let q1t = pipe("q1t_matvec");
    let q4b = pipe("q4b_matvec");
    let q1t_mm = pipe("q1t_mul_mm");
    let q1t_ovmm = pipe("q1t_overlay_mm");
    let rmsnorm = pipe("rmsnorm");
    let attn_rope = pipe("attn_rope_qkn");
    let kv_append = pipe("kv_append");
    let gqa_attend = pipe("gqa_attend");
    let layout = matvec.get_bind_group_layout(0);
    let layout_q1 = q1.get_bind_group_layout(0);
    let layout_rmsnorm = rmsnorm.get_bind_group_layout(0);
    let layout_attn_rope = attn_rope.get_bind_group_layout(0);
    let layout_kv = kv_append.get_bind_group_layout(0);
    let layout_attend = gqa_attend.get_bind_group_layout(0);
    let layout_mm = matmat.get_bind_group_layout(0);
    let layout_mmm = mul_mm.get_bind_group_layout(0);
    let layout_silu = silu.get_bind_group_layout(0);
    let layout_axpy = axpy.get_bind_group_layout(0);
    let layout_zero = zero.get_bind_group_layout(0);

    Ok(Ctx {
        device,
        queue,
        matvec,
        matmat,
        mul_mm,
        silu,
        axpy,
        zero,
        q1,
        q1t,
        q4b,
        q1t_mm,
        q1t_ovmm,
        rmsnorm,
        attn_rope,
        kv_append,
        gqa_attend,
        layout,
        layout_mm,
        layout_mmm,
        layout_silu,
        layout_axpy,
        layout_zero,
        layout_q1,
        layout_rmsnorm,
        layout_attn_rope,
        layout_kv,
        layout_attend,
        discrete,
        vram_budget,
        resident: std::sync::atomic::AtomicU64::new(0),
        scratch: Mutex::new(Scratch::default()),
        weight_bufs: Mutex::new(HashMap::new()),
        rs_bufs: Mutex::new(HashMap::new()),
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
    let buf = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q8-weights"),
        contents: full_quant,
        usage: wgpu::BufferUsages::STORAGE,
    });
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
        pass.dispatch_workgroups((rows as u32).min(MAX_WG), 1, 1);
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
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&v),
        usage: wgpu::BufferUsages::UNIFORM,
    })
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
    fn wgpu_attn_rope_qkn_matches_cpu() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        if ctx().is_none() {
            eprintln!("no wgpu adapter — skipping attn_rope parity test");
            return;
        }
        let (nh, nkv, hd, rd, pos) = (4usize, 2usize, 64usize, 64usize, 5usize);
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
        let (nh, hpk, hd, cap, n) = (4usize, 2usize, 64usize, 16usize, 5usize);
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
}
