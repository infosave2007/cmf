//! Device kernels of the layer-streamed trainer: RMSNorm (both gain
//! styles) with the gain gradient, SwiGLU, the sigmoid output gate, the
//! per-head q/k norm + partial RoPE of full-attention layers, and exact
//! causal GQA attention (tiled forward with an online softmax, the
//! FlashAttention-2 backward split into a dK/dV pass over key blocks and
//! a dQ pass over query blocks). Every op reads and writes device buffers;
//! the host wrappers at the bottom upload/download for the parity tests.
//!
//! Layouts are row-major over tokens: activations `[n, d]`; q `[n, nh·hd]`,
//! k/v `[n, nkv·hd]` with n = windows·T; attention never crosses windows.

use super::{Ctx, submit};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const NORM_SRC: &str = r#"
struct P { n: u32, d: u32, gemma: u32, pad: u32, eps: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> INV: array<f32>;
@group(0) @binding(5) var<storage, read_write> DY: array<f32>;
@group(0) @binding(6) var<storage, read_write> DX: array<f32>;
@group(0) @binding(7) var<storage, read_write> DW: array<f32>;
var<workgroup> red: array<f32, 256>;

fn weff(j: u32) -> f32 { if (p.gemma != 0u) { return 1.0 + W[j]; } return W[j]; }
fn wg_sum(t: u32, v: f32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn rms_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let r = wg.x + wg.y * 65535u;
    if (r >= p.n) { return; }
    var ss = 0.0;
    for (var j = t; j < p.d; j = j + 256u) { let v = X[r * p.d + j]; ss = ss + v * v; }
    let inv = 1.0 / sqrt(wg_sum(t, ss) / f32(p.d) + p.eps);
    for (var j = t; j < p.d; j = j + 256u) { Y[r * p.d + j] = X[r * p.d + j] * inv * weff(j); }
    if (t == 0u) { INV[r] = inv; }
}

// dx = inv·w·dy − x·inv³/d·Σ dy·w·x   (written, not accumulated)
@compute @workgroup_size(256)
fn rms_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let r = wg.x + wg.y * 65535u;
    if (r >= p.n) { return; }
    let iv = INV[r];
    var s = 0.0;
    for (var j = t; j < p.d; j = j + 256u) { s = s + DY[r * p.d + j] * weff(j) * X[r * p.d + j]; }
    let coef = wg_sum(t, s) / f32(p.d) * iv * iv * iv;
    for (var j = t; j < p.d; j = j + 256u) {
        DX[r * p.d + j] = iv * weff(j) * DY[r * p.d + j] - X[r * p.d + j] * coef;
    }
}

// dw[j] += Σ_r dy·x·inv (one thread per column)
@compute @workgroup_size(256)
fn rms_dw(@builtin(global_invocation_id) g: vec3<u32>) {
    let j = g.x;
    if (j >= p.d) { return; }
    var acc = 0.0;
    for (var r = 0u; r < p.n; r = r + 1u) { acc = acc + DY[r * p.d + j] * X[r * p.d + j] * INV[r]; }
    DW[j] = DW[j] + acc;
}
"#;

const ELT_SRC: &str = r#"
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read_write> DC: array<f32>;
@group(0) @binding(5) var<storage, read_write> DA: array<f32>;
@group(0) @binding(6) var<storage, read_write> DB: array<f32>;

fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn idx(g: vec3<u32>) -> u32 { return g.x + g.y * 65535u * 256u; }

// C = silu(A)·B
@compute @workgroup_size(256)
fn swiglu_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    let a = A[i];
    C[i] = a * sigm(a) * B[i];
}

// C = s·silu(min(A, L))·clamp(B, −L, L): a SwiGLU with DeepSeek's
// `swiglu_limit` L (bits in pad0; ≤ 0 = no clamp) and an output scale s
// (bits in pad1).
@compute @workgroup_size(256)
fn swiglu_lim_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    let lim = bitcast<f32>(p.pad0);
    var a = A[i];
    var b = B[i];
    if (lim > 0.0) {
        a = min(a, lim);
        b = clamp(b, -lim, lim);
    }
    C[i] = bitcast<f32>(p.pad1) * a * sigm(a) * b;
}

// swiglu_lim_fwd with a weight per row of `pad2` elements, from DC:
// C = s·DC[i / pad2]·silu(min(A, L))·clamp(B, −L, L) — DeepSeek's route
// weight on an expert's down input.
@compute @workgroup_size(256)
fn swiglu_lim_rows_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    let lim = bitcast<f32>(p.pad0);
    var a = A[i];
    var b = B[i];
    if (lim > 0.0) {
        a = min(a, lim);
        b = clamp(b, -lim, lim);
    }
    C[i] = bitcast<f32>(p.pad1) * DC[i / p.pad2] * a * sigm(a) * b;
}

@compute @workgroup_size(256)
fn swiglu_bwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    let a = A[i];
    let s = sigm(a);
    DA[i] = DC[i] * B[i] * s * (1.0 + a * (1.0 - s));
    DB[i] = DC[i] * a * s;
}

// C = silu(A)·B·S[i mod w], S in DC, w in pad2: a SwiGLU behind one gate
// per column — the DTG-MA mask σ(m) on an MLP's intermediate.
@compute @workgroup_size(256)
fn swiglu_cols_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    let a = A[i];
    C[i] = a * sigm(a) * B[i] * DC[i % p.pad2];
}

// swiglu_bwd of DC·S[i mod w] (S in C, w in pad2): the gradient through
// the column gate into the SwiGLU's inputs.
@compute @workgroup_size(256)
fn swiglu_cols_bwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    let a = A[i];
    let s = sigm(a);
    let dc = DC[i] * C[i % p.pad2];
    DA[i] = dc * B[i] * s * (1.0 + a * (1.0 - s));
    DB[i] = dc * a * s;
}

// C = A·σ(B)   (attention output gate)
@compute @workgroup_size(256)
fn gate_fwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    C[i] = A[i] * sigm(B[i]);
}

@compute @workgroup_size(256)
fn gate_bwd(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    let s = sigm(B[i]);
    DA[i] = DC[i] * s;
    DB[i] = DC[i] * A[i] * s * (1.0 - s);
}

// C = A + B
@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = idx(g);
    if (i >= p.n) { return; }
    C[i] = A[i] + B[i];
}
"#;

// Per-head RMSNorm (over hd) then RoPE on the first 2·half dims
// (half-split pairing (i, i+half)), in place on Y; INV per (row, head).
const QKR_SRC: &str = r#"
struct P { n: u32, heads: u32, hd: u32, half: u32, t: u32, gemma: u32, pad0: u32, pad1: u32, eps: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read_write> INV: array<f32>;
@group(0) @binding(5) var<storage, read_write> ROPE: array<f32>;   // [t][half][2] cos, sin
@group(0) @binding(6) var<storage, read_write> DY: array<f32>;
@group(0) @binding(7) var<storage, read_write> DX: array<f32>;
@group(0) @binding(8) var<storage, read_write> DW: array<f32>;      // [rows_of_heads][hd] partials
var<workgroup> red: array<f32, 256>;
var<workgroup> buf: array<f32, 256>;

fn weff(j: u32) -> f32 { if (p.gemma != 0u) { return 1.0 + W[j]; } return W[j]; }
fn wg_sum(t: u32, v: f32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// one workgroup per (row, head), hd ≤ 256
@compute @workgroup_size(256)
fn qkr_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let u = wg.x + wg.y * 65535u;
    if (u >= p.n * p.heads) { return; }
    let row = u / p.heads;
    let pos = row % p.t;
    let base = u * p.hd;
    var x = 0.0;
    if (t < p.hd) { x = X[base + t]; }
    let inv = 1.0 / sqrt(wg_sum(t, x * x) / f32(p.hd) + p.eps);
    if (t < p.hd) { buf[t] = x * inv * weff(t); }
    workgroupBarrier();
    if (t < p.hd) {
        var y = buf[t];
        if (t < p.half) {
            let c = ROPE[(pos * p.half + t) * 2u];
            let s = ROPE[(pos * p.half + t) * 2u + 1u];
            y = buf[t] * c - buf[t + p.half] * s;
        } else if (t < 2u * p.half) {
            let i = t - p.half;
            let c = ROPE[(pos * p.half + i) * 2u];
            let s = ROPE[(pos * p.half + i) * 2u + 1u];
            y = buf[i] * s + buf[t] * c;
        }
        Y[base + t] = y;
    }
    if (t == 0u) { INV[u] = inv; }
}

// inverse rotation of dY, then the RMSNorm backward; the gain gradient of
// each (row, head) goes to DW[u*hd + j] (summed by the host-side reduce).
@compute @workgroup_size(256)
fn qkr_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let u = wg.x + wg.y * 65535u;
    if (u >= p.n * p.heads) { return; }
    let row = u / p.heads;
    let pos = row % p.t;
    let base = u * p.hd;
    if (t < p.hd) { buf[t] = DY[base + t]; }
    workgroupBarrier();
    var g = 0.0;
    if (t < p.hd) {
        g = buf[t];
        if (t < p.half) {
            let c = ROPE[(pos * p.half + t) * 2u];
            let s = ROPE[(pos * p.half + t) * 2u + 1u];
            g = buf[t] * c + buf[t + p.half] * s;
        } else if (t < 2u * p.half) {
            let i = t - p.half;
            let c = ROPE[(pos * p.half + i) * 2u];
            let s = ROPE[(pos * p.half + i) * 2u + 1u];
            g = -buf[i] * s + buf[t] * c;
        }
    }
    let iv = INV[u];
    var x = 0.0;
    if (t < p.hd) { x = X[base + t]; }
    var w = 0.0;
    if (t < p.hd) { w = weff(t); }
    let coef = wg_sum(t, g * w * x) / f32(p.hd) * iv * iv * iv;
    if (t < p.hd) {
        DX[base + t] = iv * w * g - x * coef;
        DW[base + t] = g * x * iv;
    }
}

// DW column sums: W_grad[j] += Σ_u DW[u*hd + j]   (one thread per j)
@compute @workgroup_size(256)
fn qkr_dw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= p.hd) { return; }
    var acc = 0.0;
    for (var u = 0u; u < p.n * p.heads; u = u + 1u) { acc = acc + DW[u * p.hd + j]; }
    W[j] = acc;   // reused as the gradient output buffer
}
"#;

const ATTN_SRC: &str = r#"
struct P { nb: u32, t: u32, nh: u32, nkv: u32, hd: u32, pad0: u32, pad1: u32, pad2: u32, scale: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<storage, read_write> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<storage, read_write> LSE: array<f32>;
@group(0) @binding(6) var<storage, read_write> DO: array<f32>;
@group(0) @binding(7) var<storage, read_write> DQ: array<f32>;
@group(0) @binding(8) var<storage, read_write> DK: array<f32>;
@group(0) @binding(9) var<storage, read_write> DV: array<f32>;
@group(0) @binding(10) var<storage, read_write> DS: array<f32>;   // [n, nh] Σ dO·O

var<workgroup> sh_a: array<f32, 4096>;
var<workgroup> sh_b: array<f32, 4096>;
var<workgroup> sh_p: array<f32, 256>;
var<workgroup> sh_m: array<f32, 16>;
var<workgroup> sh_l: array<f32, 16>;
var<workgroup> sh_c: array<f32, 16>;

// ---- forward: one workgroup per (window, head, block of 16 queries) ----
@compute @workgroup_size(256)
fn attn_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let nqb = (p.t + 15u) / 16u;
    let qb = wg.x % nqb;
    let bh = wg.x / nqb;
    let h = bh % p.nh;
    let b = bh / p.nh;
    let g = h / (p.nh / p.nkv);
    let hd = p.hd;
    let qs = p.nh * hd;
    let ks = p.nkv * hd;
    let r = tid / 16u;          // query row in the block
    let l = tid % 16u;          // lane: columns l*cpl.., keys l
    let cpl = hd / 16u;
    let q0 = qb * 16u;
    for (var e = tid; e < 16u * hd; e = e + 256u) {
        let rr = e / hd;
        let c = e % hd;
        var v = 0.0;
        if (q0 + rr < p.t) { v = Q[(b * p.t + q0 + rr) * qs + h * hd + c]; }
        sh_a[e] = v;
    }
    if (tid < 16u) { sh_m[tid] = -3.0e38; sh_l[tid] = 0.0; }
    var acc: array<f32, 16>;
    for (var c = 0u; c < 16u; c = c + 1u) { acc[c] = 0.0; }
    let qi = q0 + r;
    let last = min(q0 + 15u, p.t - 1u);
    let nkb = last / 16u + 1u;
    workgroupBarrier();
    for (var kb = 0u; kb < nkb; kb = kb + 1u) {
        let k0 = kb * 16u;
        for (var e = tid; e < 16u * hd; e = e + 256u) {
            let kk = e / hd;
            let c = e % hd;
            var v = 0.0;
            if (k0 + kk < p.t) { v = K[(b * p.t + k0 + kk) * ks + g * hd + c]; }
            sh_b[e] = v;
        }
        workgroupBarrier();
        // score (r, key l)
        var s = -3.0e38;
        let kj = k0 + l;
        if (kj <= qi && qi < p.t) {
            var d = 0.0;
            for (var c = 0u; c < hd; c = c + 1u) { d = d + sh_a[r * hd + c] * sh_b[l * hd + c]; }
            s = d * p.scale;
        }
        sh_p[r * 16u + l] = s;
        workgroupBarrier();
        if (l == 0u) {
            var mx = sh_m[r];
            for (var k = 0u; k < 16u; k = k + 1u) { mx = max(mx, sh_p[r * 16u + k]); }
            let corr = exp(sh_m[r] - mx);
            var sum = 0.0;
            for (var k = 0u; k < 16u; k = k + 1u) {
                let sv = sh_p[r * 16u + k];
                var pk = 0.0;
                if (sv > -1.0e38) { pk = exp(sv - mx); }
                sh_p[r * 16u + k] = pk;
                sum = sum + pk;
            }
            sh_l[r] = sh_l[r] * corr + sum;
            sh_m[r] = mx;
            sh_c[r] = corr;
        }
        workgroupBarrier();
        for (var e = tid; e < 16u * hd; e = e + 256u) {
            let kk = e / hd;
            let c = e % hd;
            var v = 0.0;
            if (k0 + kk < p.t) { v = V[(b * p.t + k0 + kk) * ks + g * hd + c]; }
            sh_b[e] = v;
        }
        workgroupBarrier();
        let corr = sh_c[r];
        for (var cc = 0u; cc < cpl; cc = cc + 1u) {
            let c = l * cpl + cc;
            var a = acc[cc] * corr;
            for (var k = 0u; k < 16u; k = k + 1u) { a = a + sh_p[r * 16u + k] * sh_b[k * hd + c]; }
            acc[cc] = a;
        }
        workgroupBarrier();
    }
    if (qi < p.t) {
        let li = sh_l[r];
        for (var cc = 0u; cc < cpl; cc = cc + 1u) {
            O[(b * p.t + qi) * qs + h * hd + l * cpl + cc] = acc[cc] / li;
        }
        if (l == 0u) { LSE[(b * p.t + qi) * p.nh + h] = sh_m[r] + log(li); }
    }
}

// ---- D = Σ_c dO·O per (row, head): one workgroup (256) per (row, head) ----
@compute @workgroup_size(256)
fn attn_dsum(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let u = wg.x + wg.y * 65535u;
    if (u >= p.nb * p.t * p.nh) { return; }
    var s = 0.0;
    for (var c = tid; c < p.hd; c = c + 256u) { s = s + DO[u * p.hd + c] * O[u * p.hd + c]; }
    sh_p[tid] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) { sh_p[tid] = sh_p[tid] + sh_p[tid + st]; }
        workgroupBarrier();
    }
    if (tid == 0u) { DS[u] = sh_p[0]; }
}

// ---- dK, dV: one workgroup per (window, kv-head, block of 8 keys);
// loops over the rep query heads and the query blocks that see the keys.
// sh_a: K (8×hd) | V (8×hd) ; sh_b: Q (8×hd) | dO (8×hd) ; sh_p: P | dS (8×8 each)
@compute @workgroup_size(256)
fn attn_dkv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let nkb = (p.t + 7u) / 8u;
    let kb = wg.x % nkb;
    let bg = wg.x / nkb;
    let g = bg % p.nkv;
    let b = bg / p.nkv;
    let rep = p.nh / p.nkv;
    let hd = p.hd;
    let qs = p.nh * hd;
    let ks = p.nkv * hd;
    let k0 = kb * 8u;
    for (var e = tid; e < 8u * hd; e = e + 256u) {
        let kk = e / hd;
        let c = e % hd;
        var kv = 0.0;
        var vv = 0.0;
        if (k0 + kk < p.t) {
            kv = K[(b * p.t + k0 + kk) * ks + g * hd + c];
            vv = V[(b * p.t + k0 + kk) * ks + g * hd + c];
        }
        sh_a[e] = kv;
        sh_a[8u * hd + e] = vv;
    }
    let kr = tid / 32u;             // key row owned for accumulation
    let cl = tid % 32u;             // column lane: columns cl*cpl..
    let cpl = hd / 32u;
    var dk: array<f32, 8>;
    var dv: array<f32, 8>;
    for (var c = 0u; c < 8u; c = c + 1u) { dk[c] = 0.0; dv[c] = 0.0; }
    // dot assignment: pair (qq, kk) with a 4-way split of the hd axis
    let pq = tid / 32u;
    let pk = (tid / 4u) % 8u;
    let part = tid % 4u;
    let chunk = hd / 4u;
    workgroupBarrier();
    let nqb = (p.t + 7u) / 8u;
    for (var hh = 0u; hh < rep; hh = hh + 1u) {
        let h = g * rep + hh;
        for (var qb = k0 / 8u; qb < nqb; qb = qb + 1u) {
            let q0 = qb * 8u;
            for (var e = tid; e < 8u * hd; e = e + 256u) {
                let qq = e / hd;
                let c = e % hd;
                var qv = 0.0;
                var dov = 0.0;
                if (q0 + qq < p.t) {
                    qv = Q[(b * p.t + q0 + qq) * qs + h * hd + c];
                    dov = DO[(b * p.t + q0 + qq) * qs + h * hd + c];
                }
                sh_b[e] = qv;
                sh_b[8u * hd + e] = dov;
            }
            workgroupBarrier();
            // partial dots s = q·k and dp = dO·v over a quarter of hd
            var s = 0.0;
            var dp = 0.0;
            for (var c = part * chunk; c < (part + 1u) * chunk; c = c + 1u) {
                s = s + sh_b[pq * hd + c] * sh_a[pk * hd + c];
                dp = dp + sh_b[8u * hd + pq * hd + c] * sh_a[8u * hd + pk * hd + c];
            }
            // reduce the 4 parts through sh_m-free shared slots (reuse sh_p as 2×64 + scratch)
            workgroupBarrier();
            if (part == 0u) { sh_p[pq * 8u + pk] = 0.0; sh_p[64u + pq * 8u + pk] = 0.0; }
            workgroupBarrier();
            // serialised accumulation of the 4 parts (no float atomics in WGSL)
            for (var w = 0u; w < 4u; w = w + 1u) {
                if (part == w) {
                    sh_p[pq * 8u + pk] = sh_p[pq * 8u + pk] + s;
                    sh_p[64u + pq * 8u + pk] = sh_p[64u + pq * 8u + pk] + dp;
                }
                workgroupBarrier();
            }
            if (part == 0u) {
                let qi = q0 + pq;
                let kj = k0 + pk;
                var pr = 0.0;
                var dsv = 0.0;
                if (qi < p.t && kj < p.t && kj <= qi) {
                    let lse = LSE[(b * p.t + qi) * p.nh + h];
                    pr = exp(sh_p[pq * 8u + pk] * p.scale - lse);
                    let dsum = DS[(b * p.t + qi) * p.nh + h];
                    dsv = pr * (sh_p[64u + pq * 8u + pk] - dsum) * p.scale;
                }
                sh_p[128u + pq * 8u + pk] = pr;
                sh_p[192u + pq * 8u + pk] = dsv;
            }
            workgroupBarrier();
            for (var cc = 0u; cc < cpl; cc = cc + 1u) {
                let c = cl * cpl + cc;
                var a = dk[cc];
                var bv = dv[cc];
                for (var qq = 0u; qq < 8u; qq = qq + 1u) {
                    a = a + sh_p[192u + qq * 8u + kr] * sh_b[qq * hd + c];
                    bv = bv + sh_p[128u + qq * 8u + kr] * sh_b[8u * hd + qq * hd + c];
                }
                dk[cc] = a;
                dv[cc] = bv;
            }
            workgroupBarrier();
        }
    }
    if (k0 + kr < p.t) {
        for (var cc = 0u; cc < cpl; cc = cc + 1u) {
            let c = cl * cpl + cc;
            DK[(b * p.t + k0 + kr) * ks + g * hd + c] = dk[cc];
            DV[(b * p.t + k0 + kr) * ks + g * hd + c] = dv[cc];
        }
    }
}

// ---- dQ: one workgroup per (window, head, block of 8 queries) ----
// sh_a: Q (8×hd) | dO (8×hd) ; sh_b: K (8×hd) | V (8×hd)
@compute @workgroup_size(256)
fn attn_dq(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let nqb = (p.t + 7u) / 8u;
    let qb = wg.x % nqb;
    let bh = wg.x / nqb;
    let h = bh % p.nh;
    let b = bh / p.nh;
    let g = h / (p.nh / p.nkv);
    let hd = p.hd;
    let qs = p.nh * hd;
    let ks = p.nkv * hd;
    let q0 = qb * 8u;
    for (var e = tid; e < 8u * hd; e = e + 256u) {
        let qq = e / hd;
        let c = e % hd;
        var qv = 0.0;
        var dov = 0.0;
        if (q0 + qq < p.t) {
            qv = Q[(b * p.t + q0 + qq) * qs + h * hd + c];
            dov = DO[(b * p.t + q0 + qq) * qs + h * hd + c];
        }
        sh_a[e] = qv;
        sh_a[8u * hd + e] = dov;
    }
    let qr = tid / 32u;
    let cl = tid % 32u;
    let cpl = hd / 32u;
    var dq: array<f32, 8>;
    for (var c = 0u; c < 8u; c = c + 1u) { dq[c] = 0.0; }
    let pq = tid / 32u;
    let pk = (tid / 4u) % 8u;
    let part = tid % 4u;
    let chunk = hd / 4u;
    let last = min(q0 + 7u, p.t - 1u);
    workgroupBarrier();
    for (var kb = 0u; kb <= last / 8u; kb = kb + 1u) {
        let k0 = kb * 8u;
        for (var e = tid; e < 8u * hd; e = e + 256u) {
            let kk = e / hd;
            let c = e % hd;
            var kv = 0.0;
            var vv = 0.0;
            if (k0 + kk < p.t) {
                kv = K[(b * p.t + k0 + kk) * ks + g * hd + c];
                vv = V[(b * p.t + k0 + kk) * ks + g * hd + c];
            }
            sh_b[e] = kv;
            sh_b[8u * hd + e] = vv;
        }
        workgroupBarrier();
        var s = 0.0;
        var dp = 0.0;
        for (var c = part * chunk; c < (part + 1u) * chunk; c = c + 1u) {
            s = s + sh_a[pq * hd + c] * sh_b[pk * hd + c];
            dp = dp + sh_a[8u * hd + pq * hd + c] * sh_b[8u * hd + pk * hd + c];
        }
        workgroupBarrier();
        if (part == 0u) { sh_p[pq * 8u + pk] = 0.0; sh_p[64u + pq * 8u + pk] = 0.0; }
        workgroupBarrier();
        for (var w = 0u; w < 4u; w = w + 1u) {
            if (part == w) {
                sh_p[pq * 8u + pk] = sh_p[pq * 8u + pk] + s;
                sh_p[64u + pq * 8u + pk] = sh_p[64u + pq * 8u + pk] + dp;
            }
            workgroupBarrier();
        }
        if (part == 0u) {
            let qi = q0 + pq;
            let kj = k0 + pk;
            var dsv = 0.0;
            if (qi < p.t && kj < p.t && kj <= qi) {
                let lse = LSE[(b * p.t + qi) * p.nh + h];
                let pr = exp(sh_p[pq * 8u + pk] * p.scale - lse);
                let dsum = DS[(b * p.t + qi) * p.nh + h];
                dsv = pr * (sh_p[64u + pq * 8u + pk] - dsum) * p.scale;
            }
            sh_p[192u + pq * 8u + pk] = dsv;
        }
        workgroupBarrier();
        for (var cc = 0u; cc < cpl; cc = cc + 1u) {
            let c = cl * cpl + cc;
            var a = dq[cc];
            for (var kk = 0u; kk < 8u; kk = kk + 1u) { a = a + sh_p[192u + qr * 8u + kk] * sh_b[kk * hd + c]; }
            dq[cc] = a;
        }
        workgroupBarrier();
    }
    if (q0 + qr < p.t) {
        for (var cc = 0u; cc < cpl; cc = cc + 1u) {
            DQ[(b * p.t + q0 + qr) * qs + h * hd + cl * cpl + cc] = dq[cc];
        }
    }
}
"#;

/// A trainer shader module. Built without naga's injected bounds checks
/// and loop bounding (as `zimage` builds `zi_mm`): the clamps and the loop
/// guard keep the drivers from unrolling the kernels' serial loops (column
/// reductions, the GDN recurrence), so every iteration paid a full memory
/// latency. `CMF_TRAIN_CHECKED=1` restores the checked build.
pub(super) fn trainer_module(c: &Ctx, name: &str, src: &str) -> wgpu::ShaderModule {
    let desc = wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) };
    if std::env::var("CMF_TRAIN_CHECKED").as_deref() == Ok("1") {
        return c.device.create_shader_module(desc);
    }
    // SAFETY: every trainer kernel guards its own indices against the
    // dispatch geometry, the buffers are sized by their owners for that
    // geometry, and every loop has a uniform, finite trip count.
    unsafe { c.device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
}

pub(super) struct Mod {
    pub(super) p: HashMap<&'static str, wgpu::ComputePipeline>,
}

static MODS: OnceLock<Mutex<HashMap<&'static str, Option<&'static Mod>>>> = OnceLock::new();

pub(super) fn module(c: &Ctx, name: &'static str, src: &'static str, entries: &[&'static str]) -> Option<&'static Mod> {
    let map = MODS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = map.lock().unwrap();
    if let Some(m) = g.get(name) {
        return *m;
    }
    let built = (|| {
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = trainer_module(c, name, src);
        if let Some(e) = pollster::block_on(sc.pop()) {
            tracing::warn!("{name} module rejected: {e}");
            return None;
        }
        let mut p = HashMap::new();
        for &ep in entries {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipe = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &m,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            });
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("{name}/{ep} rejected: {e}");
                return None;
            }
            p.insert(ep, pipe);
        }
        let leaked: &'static Mod = Box::leak(Box::new(Mod { p }));
        for (ep, pipe) in &leaked.p {
            super::train_prof::name_pipe(pipe, ep);
        }
        Some(leaked)
    })();
    g.insert(name, built);
    built
}

fn norm_mod(c: &Ctx) -> Option<&'static Mod> {
    module(c, "train_norm", NORM_SRC, &["rms_fwd", "rms_bwd", "rms_dw"])
}
fn elt_mod(c: &Ctx) -> Option<&'static Mod> {
    module(c, "train_elt", ELT_SRC, &["swiglu_fwd", "swiglu_lim_fwd", "swiglu_lim_rows_fwd", "swiglu_bwd", "swiglu_cols_fwd", "swiglu_cols_bwd", "gate_fwd", "gate_bwd", "add"])
}
fn qkr_mod(c: &Ctx) -> Option<&'static Mod> {
    module(c, "train_qkr", QKR_SRC, &["qkr_fwd", "qkr_bwd", "qkr_dw"])
}
fn attn_mod(c: &Ctx) -> Option<&'static Mod> {
    module(c, "train_attn", ATTN_SRC, &["attn_fwd", "attn_dsum", "attn_dkv", "attn_dq"])
}

/// A dispatch: pipeline, uniform words, (binding, buffer) pairs, grid.
pub struct Disp<'a> {
    pub(super) pipe: &'a wgpu::ComputePipeline,
    pub(super) uni: Vec<u32>,
    pub(super) bufs: Vec<(u32, &'a wgpu::Buffer)>,
    pub(super) grid: (u32, u32, u32),
}

/// Encode dispatches into one compute pass and submit.
pub fn run(c: &Ctx, ds: &[Disp]) {
    let prof = super::train_prof::on();
    let mut keep = Vec::new();
    let mut groups = Vec::new();
    for d in ds {
        let mut words = d.uni.clone();
        while words.len() % 4 != 0 {
            words.push(0);
        }
        if prof {
            use super::train_prof::{Cnt, count};
            count(Cnt::Buf, 1);
            count(Cnt::Write, 1);
            count(Cnt::WriteBytes, (words.len() * 4) as u64);
            count(Cnt::BindGroup, 1);
        }
        let u = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("train_p"),
            size: (words.len() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&u, 0, bytemuck::cast_slice(&words));
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }];
        for &(b, buf) in &d.bufs {
            entries.push(wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() });
        }
        let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("train"),
            layout: &d.pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        keep.push(u);
        groups.push(bg);
    }
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("train") });
    {
        let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        let mut mk = super::train_prof::pass_begin(c, &mut ps);
        for (d, bg) in ds.iter().zip(&groups) {
            ps.set_pipeline(d.pipe);
            ps.set_bind_group(0, bg, &[]);
            ps.dispatch_workgroups(d.grid.0, d.grid.1, d.grid.2);
            if prof {
                super::train_prof::pass_mark(c, &mut ps, &mut mk, super::train_prof::pipe_name(d.pipe), [0; 3]);
            }
        }
    }
    submit(c, enc.finish());
}

pub(super) fn wg_fold(n: usize) -> (u32, u32, u32) {
    if n <= 65535 { (n as u32, 1, 1) } else { (65535, n.div_ceil(65535) as u32, 1) }
}
pub(super) fn elt_fold(n: usize) -> (u32, u32, u32) {
    let g = n.div_ceil(256);
    if g <= 65535 { (g as u32, 1, 1) } else { (65535, g.div_ceil(65535) as u32, 1) }
}

/// RMSNorm forward over `n` rows of width `d`.
pub fn rms_fwd<'a>(c: &'a Ctx, x: &'a wgpu::Buffer, w: &'a wgpu::Buffer, y: &'a wgpu::Buffer, inv: &'a wgpu::Buffer, n: usize, d: usize, eps: f32, gemma: bool) -> Option<Disp<'a>> {
    let m = norm_mod(c)?;
    Some(Disp {
        pipe: &m.p["rms_fwd"],
        uni: vec![n as u32, d as u32, gemma as u32, 0, eps.to_bits(), 0, 0, 0],
        bufs: vec![(1, x), (2, w), (3, y), (4, inv)],
        grid: wg_fold(n),
    })
}

/// RMSNorm backward (dx written).
#[allow(clippy::too_many_arguments)]
pub fn rms_bwd<'a>(c: &'a Ctx, x: &'a wgpu::Buffer, w: &'a wgpu::Buffer, inv: &'a wgpu::Buffer, dy: &'a wgpu::Buffer, dx: &'a wgpu::Buffer, n: usize, d: usize, gemma: bool) -> Option<Disp<'a>> {
    let m = norm_mod(c)?;
    Some(Disp {
        pipe: &m.p["rms_bwd"],
        uni: vec![n as u32, d as u32, gemma as u32, 0, 0, 0, 0, 0],
        bufs: vec![(1, x), (2, w), (4, inv), (5, dy), (6, dx)],
        grid: wg_fold(n),
    })
}

/// RMSNorm gain gradient `dw[j] += Σ_r dy·x·inv`, accumulated into `dw`
/// (the two-pass column reduction of `train_mm`).
pub fn rms_dw<'a>(c: &'a Ctx, x: &'a wgpu::Buffer, inv: &'a wgpu::Buffer, dy: &'a wgpu::Buffer, dw: &'a wgpu::Buffer, n: usize, d: usize) -> Option<[Disp<'a>; 2]> {
    super::train_mm::colsum(c, dy, super::train_mm::ColTerm::ProdRow(x, inv), dw, n, d, true)
}

/// Elementwise ops: `swiglu_fwd` (C = silu(A)·B), `swiglu_bwd`
/// (DA, DB from DC), `gate_fwd` (C = A·σ(B)), `gate_bwd`, `add`.
#[allow(clippy::too_many_arguments)]
pub fn elt<'a>(c: &'a Ctx, op: &str, n: usize, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, cc: Option<&'a wgpu::Buffer>, dc: Option<&'a wgpu::Buffer>, da: Option<&'a wgpu::Buffer>, db: Option<&'a wgpu::Buffer>) -> Option<Disp<'a>> {
    let m = elt_mod(c)?;
    let (key, bufs): (&'static str, Vec<(u32, &wgpu::Buffer)>) = match op {
        "swiglu_fwd" => ("swiglu_fwd", vec![(1, a), (2, b), (3, cc?)]),
        "gate_fwd" => ("gate_fwd", vec![(1, a), (2, b), (3, cc?)]),
        "add" => ("add", vec![(1, a), (2, b), (3, cc?)]),
        "swiglu_bwd" => ("swiglu_bwd", vec![(1, a), (2, b), (4, dc?), (5, da?), (6, db?)]),
        "gate_bwd" => ("gate_bwd", vec![(1, a), (2, b), (4, dc?), (5, da?), (6, db?)]),
        _ => return None,
    };
    Some(Disp { pipe: &m.p[key], uni: vec![n as u32, 0, 0, 0], bufs, grid: elt_fold(n) })
}

/// `C = scale·silu(min(A, limit))·clamp(B, −limit, limit)` over `n`
/// elements (`limit ≤ 0`: no clamp) — the expert activation of DeepSeek's
/// `swiglu_limit`, with an output scale for weighted rows.
pub fn swiglu_lim<'a>(c: &'a Ctx, n: usize, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, cc: &'a wgpu::Buffer, limit: f32, scale: f32) -> Option<Disp<'a>> {
    let m = elt_mod(c)?;
    Some(Disp { pipe: &m.p["swiglu_lim_fwd"], uni: vec![n as u32, limit.to_bits(), scale.to_bits(), 0], bufs: vec![(1, a), (2, b), (3, cc)], grid: elt_fold(n) })
}

/// [`swiglu_lim`] with one weight per row of `width` elements:
/// `C = scale·rw[row]·silu(min(A, limit))·clamp(B, −limit, limit)` — the
/// route-weighted down input of a DeepSeek expert.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_lim_rows<'a>(c: &'a Ctx, n: usize, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, cc: &'a wgpu::Buffer, rw: &'a wgpu::Buffer, width: usize, limit: f32, scale: f32) -> Option<Disp<'a>> {
    let m = elt_mod(c)?;
    Some(Disp { pipe: &m.p["swiglu_lim_rows_fwd"], uni: vec![n as u32, limit.to_bits(), scale.to_bits(), width.max(1) as u32], bufs: vec![(1, a), (2, b), (3, cc), (4, rw)], grid: elt_fold(n) })
}

/// `C = silu(A)·B·s[col]` over `n` elements in rows of `width`: a SwiGLU
/// behind a gate per column (DTG-MA's mask σ(m) on an MLP intermediate).
#[allow(clippy::too_many_arguments)]
pub fn swiglu_cols<'a>(c: &'a Ctx, n: usize, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, cc: &'a wgpu::Buffer, s: &'a wgpu::Buffer, width: usize) -> Option<Disp<'a>> {
    let m = elt_mod(c)?;
    Some(Disp { pipe: &m.p["swiglu_cols_fwd"], uni: vec![n as u32, 0, 0, width.max(1) as u32], bufs: vec![(1, a), (2, b), (3, cc), (4, s)], grid: elt_fold(n) })
}

/// Backward of [`swiglu_cols`]: `da`, `db` from `dc` (the gradient at the
/// gated output) through the gate `s`.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_cols_bwd<'a>(c: &'a Ctx, n: usize, a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, s: &'a wgpu::Buffer, dc: &'a wgpu::Buffer, da: &'a wgpu::Buffer, db: &'a wgpu::Buffer, width: usize) -> Option<Disp<'a>> {
    let m = elt_mod(c)?;
    Some(Disp { pipe: &m.p["swiglu_cols_bwd"], uni: vec![n as u32, 0, 0, width.max(1) as u32], bufs: vec![(1, a), (2, b), (3, s), (4, dc), (5, da), (6, db)], grid: elt_fold(n) })
}

/// Geometry of a per-head norm + RoPE.
#[derive(Clone, Copy)]
pub struct QkrGeom {
    pub n: usize,
    pub heads: usize,
    pub hd: usize,
    pub half: usize,
    pub t: usize,
    pub gemma: bool,
    pub eps: f32,
}

impl QkrGeom {
    fn uni(&self) -> Vec<u32> {
        vec![self.n as u32, self.heads as u32, self.hd as u32, self.half as u32, self.t as u32, self.gemma as u32, 0, 0, self.eps.to_bits(), 0, 0, 0]
    }
}

pub fn qkr_fwd<'a>(c: &'a Ctx, g: QkrGeom, x: &'a wgpu::Buffer, w: &'a wgpu::Buffer, y: &'a wgpu::Buffer, inv: &'a wgpu::Buffer, rope: &'a wgpu::Buffer) -> Option<Disp<'a>> {
    let m = qkr_mod(c)?;
    Some(Disp { pipe: &m.p["qkr_fwd"], uni: g.uni(), bufs: vec![(1, x), (2, w), (3, y), (4, inv), (5, rope)], grid: wg_fold(g.n * g.heads) })
}

/// Backward of [`qkr_fwd`]: dx, and per-(row, head) gain partials in `dwp`.
#[allow(clippy::too_many_arguments)]
pub fn qkr_bwd<'a>(c: &'a Ctx, g: QkrGeom, x: &'a wgpu::Buffer, w: &'a wgpu::Buffer, inv: &'a wgpu::Buffer, rope: &'a wgpu::Buffer, dy: &'a wgpu::Buffer, dx: &'a wgpu::Buffer, dwp: &'a wgpu::Buffer) -> Option<Disp<'a>> {
    let m = qkr_mod(c)?;
    Some(Disp {
        pipe: &m.p["qkr_bwd"],
        uni: g.uni(),
        bufs: vec![(1, x), (2, w), (4, inv), (5, rope), (6, dy), (7, dx), (8, dwp)],
        grid: wg_fold(g.n * g.heads),
    })
}

/// Sum the gain partials of [`qkr_bwd`] into `dw` (overwritten, or
/// accumulated with `acc`), as a two-pass column reduction.
pub fn qkr_dw<'a>(c: &'a Ctx, g: QkrGeom, dwp: &'a wgpu::Buffer, dw: &'a wgpu::Buffer, acc: bool) -> Option<[Disp<'a>; 2]> {
    super::train_mm::colsum(c, dwp, super::train_mm::ColTerm::Plain, dw, g.n * g.heads, g.hd, acc)
}

/// Geometry of a causal GQA attention over `nb` windows of `t` tokens.
#[derive(Clone, Copy)]
pub struct AttnGeom {
    pub nb: usize,
    pub t: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
}

impl AttnGeom {
    fn uni(&self) -> Vec<u32> {
        let scale = 1.0 / (self.hd as f32).sqrt();
        vec![self.nb as u32, self.t as u32, self.nh as u32, self.nkv as u32, self.hd as u32, 0, 0, 0, scale.to_bits(), 0, 0, 0]
    }
    fn ok(&self) -> bool {
        self.hd % 32 == 0 && self.hd <= 256 && self.nh % self.nkv == 0
    }
}

/// Attention forward: O and the per-(row, head) log-sum-exp.
pub fn attn_fwd<'a>(c: &'a Ctx, g: AttnGeom, q: &'a wgpu::Buffer, k: &'a wgpu::Buffer, v: &'a wgpu::Buffer, o: &'a wgpu::Buffer, lse: &'a wgpu::Buffer) -> Option<Disp<'a>> {
    if !g.ok() {
        return None;
    }
    let m = attn_mod(c)?;
    let n = g.nb * g.nh * g.t.div_ceil(16);
    Some(Disp { pipe: &m.p["attn_fwd"], uni: g.uni(), bufs: vec![(1, q), (2, k), (3, v), (4, o), (5, lse)], grid: (n as u32, 1, 1) })
}

/// Attention backward: three dispatches (D, dK/dV, dQ).
#[allow(clippy::too_many_arguments)]
pub fn attn_bwd<'a>(
    c: &'a Ctx,
    g: AttnGeom,
    q: &'a wgpu::Buffer,
    k: &'a wgpu::Buffer,
    v: &'a wgpu::Buffer,
    o: &'a wgpu::Buffer,
    lse: &'a wgpu::Buffer,
    dout: &'a wgpu::Buffer,
    dsum: &'a wgpu::Buffer,
    dq: &'a wgpu::Buffer,
    dk: &'a wgpu::Buffer,
    dv: &'a wgpu::Buffer,
) -> Option<Vec<Disp<'a>>> {
    if !g.ok() {
        return None;
    }
    let m = attn_mod(c)?;
    let u = g.uni();
    Some(vec![
        Disp { pipe: &m.p["attn_dsum"], uni: u.clone(), bufs: vec![(4, o), (6, dout), (10, dsum)], grid: wg_fold(g.nb * g.t * g.nh) },
        Disp {
            pipe: &m.p["attn_dkv"],
            uni: u.clone(),
            bufs: vec![(1, q), (2, k), (3, v), (5, lse), (6, dout), (8, dk), (9, dv), (10, dsum)],
            grid: ((g.nb * g.nkv * g.t.div_ceil(8)) as u32, 1, 1),
        },
        Disp {
            pipe: &m.p["attn_dq"],
            uni: u,
            bufs: vec![(1, q), (2, k), (3, v), (5, lse), (6, dout), (7, dq), (10, dsum)],
            grid: ((g.nb * g.nh * g.t.div_ceil(8)) as u32, 1, 1),
        },
    ])
}

// ───────────────────────────── host helpers ─────────────────────────────

/// A storage buffer holding `data` (or zeros of `n` floats).
pub fn upload(c: &Ctx, data: &[f32]) -> wgpu::Buffer {
    if super::train_prof::on() {
        use super::train_prof::{Cnt, count};
        count(Cnt::Buf, 1);
        count(Cnt::Write, !data.is_empty() as u64);
        count(Cnt::WriteBytes, (data.len() * 4) as u64);
    }
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("train_buf"),
        size: (data.len().max(4) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
    }
    b
}

/// Overwrite the start of `b` with `data`.
pub fn write(c: &Ctx, b: &wgpu::Buffer, data: &[f32]) {
    if super::train_prof::on() {
        super::train_prof::count(super::train_prof::Cnt::Write, 1);
        super::train_prof::count(super::train_prof::Cnt::WriteBytes, (data.len() * 4) as u64);
    }
    if !data.is_empty() {
        c.queue.write_buffer(b, 0, bytemuck::cast_slice(data));
    }
}

pub fn zeros(c: &Ctx, n: usize) -> wgpu::Buffer {
    upload(c, &vec![0f32; n.max(1)])
}

pub fn download(c: &Ctx, b: &wgpu::Buffer, n: usize) -> Option<Vec<f32>> {
    let size = (n * 4) as u64;
    let st = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("train_rb"),
        size: size.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let prof = super::train_prof::on();
    let mut mk = super::train_prof::enc_begin(c, &mut enc);
    enc.copy_buffer_to_buffer(b, 0, &st, 0, size);
    if prof {
        super::train_prof::count(super::train_prof::Cnt::Buf, 1);
        super::train_prof::enc_mark(c, &mut enc, &mut mk, "readback");
    }
    submit(c, enc.finish());
    let slice = st.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let tw = prof.then(std::time::Instant::now);
    c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    if let Some(tw) = tw {
        super::train_prof::count(super::train_prof::Cnt::Wait, 1);
        super::train_prof::count(super::train_prof::Cnt::WaitNs, tw.elapsed().as_nanos() as u64);
    }
    let out = {
        let data = slice.get_mapped_range().ok()?;
        let mut v = vec![0f32; n];
        super::par_copy(&mut v, bytemuck::cast_slice(&data[..size as usize]));
        v
    };
    st.unmap();
    Some(out)
}

/// `n` u32 words of `b` (bit patterns kept exactly).
pub fn download_u32(c: &Ctx, b: &wgpu::Buffer, n: usize) -> Option<Vec<u32>> {
    let size = (n * 4) as u64;
    let st = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("train_rb_u32"),
        size: size.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let prof = super::train_prof::on();
    let mut mk = super::train_prof::enc_begin(c, &mut enc);
    enc.copy_buffer_to_buffer(b, 0, &st, 0, size);
    if prof {
        super::train_prof::count(super::train_prof::Cnt::Buf, 1);
        super::train_prof::enc_mark(c, &mut enc, &mut mk, "readback");
    }
    submit(c, enc.finish());
    let slice = st.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let tw = prof.then(std::time::Instant::now);
    c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    if let Some(tw) = tw {
        super::train_prof::count(super::train_prof::Cnt::Wait, 1);
        super::train_prof::count(super::train_prof::Cnt::WaitNs, tw.elapsed().as_nanos() as u64);
    }
    let out = {
        let data = slice.get_mapped_range().ok()?;
        bytemuck::cast_slice::<u8, u32>(&data[..size as usize]).to_vec()
    };
    st.unmap();
    Some(out)
}

/// A zero-filled storage buffer of `n` u32 words (no host staging: wgpu
/// zero-initializes new buffers).
pub fn zeros_u32(c: &Ctx, n: usize) -> wgpu::Buffer {
    super::train_prof::count(super::train_prof::Cnt::Buf, 1);
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("train_u32"),
        size: (n.max(4) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The trainer's device context, for callers outside `gpu_wgpu` (on macOS
/// a wgpu context of its own beside the runtime's native Metal).
pub fn dev() -> Option<&'static Ctx> {
    super::trainer_ctx()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fcd_ops::{attn_head_bwd, attn_head_fwd, rmsnorm_bwd, rmsnorm_fwd, rope_bwd, rope_fwd};

    /// The elementwise module (with the clamped SwiGLU entry) parses and
    /// validates without a device: one bad entry would break every kernel
    /// of the module.
    #[test]
    fn elt_wgsl_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(ELT_SRC).expect("ELT_SRC must parse");
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .expect("ELT_SRC must validate");
        for e in ["swiglu_fwd", "swiglu_lim_fwd", "swiglu_bwd", "swiglu_cols_fwd", "swiglu_cols_bwd", "gate_fwd", "gate_bwd", "add"] {
            assert!(module.entry_points.iter().any(|p| p.name == e), "entry {e}");
        }
    }

    fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32 * scale
            })
            .collect()
    }
    fn rel(name: &str, got: &[f32], want: &[f64], tol: f64) {
        let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).powi(2)).sum();
        let den: f64 = want.iter().map(|w| w * w).sum::<f64>().max(1e-30);
        let r = (num / den).sqrt();
        eprintln!("{name}: relative error {r:e}");
        assert!(r < tol, "{name}: relative error {r:e}");
    }
    fn f64v(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| x as f64).collect()
    }

    #[test]
    fn rms_and_elementwise_match_host() {
        let Some(c) = dev() else { return };
        let (n, d) = (37usize, 5120usize);
        let x = rnd(n * d, 1, 3.0);
        let w = rnd(d, 2, 0.5);
        let dy = rnd(n * d, 3, 1.0);
        for gemma in [false, true] {
            let (bx, bw, by, binv, bdy, bdx, bdw) = (upload(c, &x), upload(c, &w), zeros(c, n * d), zeros(c, n), upload(c, &dy), zeros(c, n * d), zeros(c, d));
            let [dw1, dw2] = rms_dw(c, &bx, &binv, &bdy, &bdw, n, d).unwrap();
            let ds = vec![
                rms_fwd(c, &bx, &bw, &by, &binv, n, d, 1e-6, gemma).unwrap(),
                rms_bwd(c, &bx, &bw, &binv, &bdy, &bdx, n, d, gemma).unwrap(),
                dw1,
                dw2,
            ];
            run(c, &ds);
            let (xf, wf, dyf) = (f64v(&x), f64v(&w), f64v(&dy));
            let (mut y, mut inv, mut dx, mut dw) = (vec![0f64; n * d], vec![0f64; n], vec![0f64; n * d], vec![0f64; d]);
            rmsnorm_fwd(&xf, &wf, 1e-6, gemma, &mut y, &mut inv);
            rmsnorm_bwd(&xf, &wf, &inv, &dyf, gemma, &mut dx, Some(&mut dw));
            rel("rms y", &download(c, &by, n * d).unwrap(), &y, 1e-6);
            rel("rms dx", &download(c, &bdx, n * d).unwrap(), &dx, 1e-5);
            rel("rms dw", &download(c, &bdw, d).unwrap(), &dw, 1e-5);
        }
        let a = rnd(4097, 4, 4.0);
        let b = rnd(4097, 5, 2.0);
        let dc = rnd(4097, 6, 1.0);
        let (ba, bb, bc, bdc, bda, bdb) = (upload(c, &a), upload(c, &b), zeros(c, 4097), upload(c, &dc), zeros(c, 4097), zeros(c, 4097));
        run(c, &[elt(c, "swiglu_fwd", 4097, &ba, &bb, Some(&bc), None, None, None).unwrap(), elt(c, "swiglu_bwd", 4097, &ba, &bb, None, Some(&bdc), Some(&bda), Some(&bdb)).unwrap()]);
        let sg = |x: f64| 1.0 / (1.0 + (-x).exp());
        let want: Vec<f64> = a.iter().zip(&b).map(|(&a, &b)| a as f64 * sg(a as f64) * b as f64).collect();
        rel("swiglu", &download(c, &bc, 4097).unwrap(), &want, 1e-6);
        let wda: Vec<f64> = (0..4097).map(|i| { let (a, s) = (a[i] as f64, sg(a[i] as f64)); dc[i] as f64 * b[i] as f64 * s * (1.0 + a * (1.0 - s)) }).collect();
        rel("swiglu da", &download(c, &bda, 4097).unwrap(), &wda, 1e-6);
        // behind a column gate (rows of 17 → 241 rows, the last one short)
        let w = 17usize;
        let gs: Vec<f32> = (0..w).map(|j| 0.1 + 0.05 * j as f32).collect();
        let (bs, bg, bga, bgb) = (upload(c, &gs), zeros(c, 4097), zeros(c, 4097), zeros(c, 4097));
        run(c, &[swiglu_cols(c, 4097, &ba, &bb, &bg, &bs, w).unwrap(), swiglu_cols_bwd(c, 4097, &ba, &bb, &bs, &bdc, &bga, &bgb, w).unwrap()]);
        let want: Vec<f64> = (0..4097).map(|i| a[i] as f64 * sg(a[i] as f64) * b[i] as f64 * gs[i % w] as f64).collect();
        rel("swiglu cols", &download(c, &bg, 4097).unwrap(), &want, 1e-6);
        let wga: Vec<f64> = (0..4097).map(|i| wda[i] * gs[i % w] as f64).collect();
        rel("swiglu cols da", &download(c, &bga, 4097).unwrap(), &wga, 1e-6);
        let wgb: Vec<f64> = (0..4097).map(|i| { let (a, s) = (a[i] as f64, sg(a[i] as f64)); dc[i] as f64 * gs[i % w] as f64 * a * s }).collect();
        rel("swiglu cols db", &download(c, &bgb, 4097).unwrap(), &wgb, 1e-6);
        // the clamped, scaled form (DeepSeek swiglu_limit), limit inside the data range
        let (lim, s) = (1.5f32, 0.75f32);
        let bl = zeros(c, 4097);
        run(c, &[swiglu_lim(c, 4097, &ba, &bb, &bl, lim, s).unwrap()]);
        let want: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| {
                let (a, b) = ((a as f64).min(lim as f64), (b as f64).clamp(-lim as f64, lim as f64));
                s as f64 * a * sg(a) * b
            })
            .collect();
        rel("swiglu_lim", &download(c, &bl, 4097).unwrap(), &want, 1e-6);
        // the same with a weight per row (17 rows of 241, the route weights)
        let (rows, width) = (17usize, 241usize);
        let rw: Vec<f32> = rnd(rows, 7, 1.0).iter().map(|v| v.abs()).collect();
        let (brw, blr) = (upload(c, &rw), zeros(c, rows * width));
        run(c, &[swiglu_lim_rows(c, rows * width, &ba, &bb, &blr, &brw, width, lim, s).unwrap()]);
        let want_r: Vec<f64> = (0..rows * width).map(|i| want[i] * rw[i / width] as f64).collect();
        rel("swiglu_lim_rows", &download(c, &blr, rows * width).unwrap(), &want_r, 1e-6);
    }

    #[test]
    fn qk_norm_rope_matches_host() {
        let Some(c) = dev() else { return };
        let (t, nb, heads, hd, half) = (19usize, 2usize, 3usize, 256usize, 32usize);
        let n = nb * t;
        let x = rnd(n * heads * hd, 7, 2.0);
        let w = rnd(hd, 8, 0.3);
        let dy = rnd(n * heads * hd, 9, 1.0);
        let inv_freq: Vec<f64> = (0..half).map(|i| 1.0 / 1e7f64.powf(2.0 * i as f64 / (2 * half) as f64)).collect();
        let mut tab = Vec::new();
        for pos in 0..t {
            for &f in &inv_freq {
                let a = pos as f64 * f;
                tab.push(a.cos() as f32);
                tab.push(a.sin() as f32);
            }
        }
        let g = QkrGeom { n, heads, hd, half, t, gemma: true, eps: 1e-6 };
        let (bx, bw, by, binv, brope, bdy, bdx, bdwp, bdw) = (upload(c, &x), upload(c, &w), zeros(c, x.len()), zeros(c, n * heads), upload(c, &tab), upload(c, &dy), zeros(c, x.len()), zeros(c, x.len()), zeros(c, hd));
        run(c, &[qkr_fwd(c, g, &bx, &bw, &by, &binv, &brope).unwrap(), qkr_bwd(c, g, &bx, &bw, &binv, &brope, &bdy, &bdx, &bdwp).unwrap()]);
        run(c, &qkr_dw(c, g, &bdwp, &bdw, false).unwrap());
        let (xf, wf) = (f64v(&x), f64v(&w));
        let (mut y, mut dx, mut dw) = (vec![0f64; x.len()], vec![0f64; x.len()], vec![0f64; hd]);
        for u in 0..n * heads {
            let pos = (u / heads) % t;
            let xr = &xf[u * hd..(u + 1) * hd];
            let mut yr = vec![0f64; hd];
            let mut iv = vec![0f64; 1];
            rmsnorm_fwd(xr, &wf, 1e-6, true, &mut yr, &mut iv);
            rope_fwd(&mut yr[..2 * half], pos, &inv_freq);
            y[u * hd..(u + 1) * hd].copy_from_slice(&yr);
            let mut g = f64v(&dy[u * hd..(u + 1) * hd]);
            rope_bwd(&mut g[..2 * half], pos, &inv_freq);
            let mut dxr = vec![0f64; hd];
            rmsnorm_bwd(xr, &wf, &iv, &g, true, &mut dxr, Some(&mut dw));
            dx[u * hd..(u + 1) * hd].copy_from_slice(&dxr);
        }
        rel("qkr y", &download(c, &by, x.len()).unwrap(), &y, 1e-6);
        rel("qkr dx", &download(c, &bdx, x.len()).unwrap(), &dx, 1e-5);
        rel("qkr dw", &download(c, &bdw, hd).unwrap(), &dw, 1e-5);
    }

    #[test]
    fn attention_matches_host() {
        let Some(c) = dev() else { return };
        let g = AttnGeom { nb: 2, t: 37, nh: 6, nkv: 2, hd: 256 };
        let (n, qs, ks) = (g.nb * g.t, g.nh * g.hd, g.nkv * g.hd);
        let q = rnd(n * qs, 11, 1.0);
        let k = rnd(n * ks, 12, 1.0);
        let v = rnd(n * ks, 13, 1.0);
        let dout = rnd(n * qs, 14, 1.0);
        let (bq, bk, bv, bo, blse, bdo, bds, bdq, bdk, bdv) = (upload(c, &q), upload(c, &k), upload(c, &v), zeros(c, n * qs), zeros(c, n * g.nh), upload(c, &dout), zeros(c, n * g.nh), zeros(c, n * qs), zeros(c, n * ks), zeros(c, n * ks));
        run(c, &[attn_fwd(c, g, &bq, &bk, &bv, &bo, &blse).unwrap()]);
        run(c, &attn_bwd(c, g, &bq, &bk, &bv, &bo, &blse, &bdo, &bds, &bdq, &bdk, &bdv).unwrap());
        let (mut o, mut dq, mut dk, mut dv) = (vec![0f64; n * qs], vec![0f64; n * qs], vec![0f64; n * ks], vec![0f64; n * ks]);
        let rep = g.nh / g.nkv;
        let hd = g.hd;
        for b in 0..g.nb {
            for h in 0..g.nh {
                let kv = h / rep;
                let gather = |src: &[f32], stride: usize, head: usize| -> Vec<f64> {
                    (0..g.t).flat_map(|ti| (0..hd).map(move |c| src[(b * g.t + ti) * stride + head * hd + c] as f64)).collect()
                };
                let (qh, kh, vh, doh) = (gather(&q, qs, h), gather(&k, ks, kv), gather(&v, ks, kv), gather(&dout, qs, h));
                let mut oh = vec![0f64; g.t * hd];
                attn_head_fwd(&qh, &kh, &vh, g.t, hd, hd, &mut oh);
                let (mut dqh, mut dkh, mut dvh) = (vec![0f64; g.t * hd], vec![0f64; g.t * hd], vec![0f64; g.t * hd]);
                attn_head_bwd(&qh, &kh, &vh, &doh, g.t, hd, hd, &mut dqh, &mut dkh, &mut dvh);
                for ti in 0..g.t {
                    for cc in 0..hd {
                        o[(b * g.t + ti) * qs + h * hd + cc] = oh[ti * hd + cc];
                        dq[(b * g.t + ti) * qs + h * hd + cc] = dqh[ti * hd + cc];
                        dk[(b * g.t + ti) * ks + kv * hd + cc] += dkh[ti * hd + cc];
                        dv[(b * g.t + ti) * ks + kv * hd + cc] += dvh[ti * hd + cc];
                    }
                }
            }
        }
        rel("attn o", &download(c, &bo, n * qs).unwrap(), &o, 1e-5);
        rel("attn dq", &download(c, &bdq, n * qs).unwrap(), &dq, 1e-5);
        rel("attn dk", &download(c, &bdk, n * ks).unwrap(), &dk, 1e-5);
        rel("attn dv", &download(c, &bdv, n * ks).unwrap(), &dv, 1e-5);
    }
}
