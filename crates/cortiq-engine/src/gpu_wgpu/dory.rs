//! Dory's token path on wgpu, resident: every matrix in VRAM, the KV of each
//! attention layer (and loop) on the device, one submit per forward. `dory` has
//! the semantics. A forward replays a recorded plan of dispatches, because the
//! only things that change from token to token are the two uniforms, the rotary
//! tables, the embedded rows and the head's output.
//!
//! An attention layer: the q,k,v and g projections, `dory_prep` (rotate, unit
//! heads, the q scale, K and V into the slot's ring), `dory_attend` (one 256-lane
//! workgroup per KV head holds its four query heads: each key is read once,
//! four dots at a time), `dory_norm`, the output projection, `dory_slerp`.
//! An MLP layer: gate and up, `dory_silu`, down, `dory_slerp`.

use super::*;
use crate::dory::{Dory, ExitRate, Gate, KvMode, Layer, Want, top_prob};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const SRC: &str = r#"
struct P {
    t: u32, pos0: u32, cap: u32, window: u32,
    nh: u32, nk: u32, hd: u32, hidden: u32,
    ffn: u32, qsize: u32, ncap: u32, rep: u32,
    scale: f32, eps: f32, ch: u32, pad1: u32,
};
@group(0) @binding(0) var<uniform> p: P;

// ── Rotation, unit heads, the query scale, K and V into the ring.
@group(0) @binding(1) var<storage, read> yq: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> yk: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> yv: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> rope: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> sq: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> qo: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> ko: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> vo: array<vec4<f32>>;

// One workgroup per head, one lane per rotated pair of vec4s: nothing to wait for
// but the reduction, and the workgroups spread over the SMs.
@compute @workgroup_size(32)
fn dory_prep(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let a = w.x;
    let row = w.y;
    let n = p.hd / 4u;
    let half = n / 2u;
    let rb = row * n;
    let e = lid;
    let c = rope[rb + e];
    let sn = rope[rb + half + e];
    if (a < p.nh) {
        let at = (row * p.nh + a) * n;
        let x = yq[at + e];
        let y = yq[at + half + e];
        let u = x * c - y * sn;
        let v = y * c + x * sn;
        let inv = inverseSqrt(rsum32(lid, dot(u, u) + dot(v, v)) + p.eps);
        let sb = a * n;
        qo[at + e] = u * inv * sq[sb + e];
        qo[at + half + e] = v * inv * sq[sb + half + e];
        return;
    }
    let kh = a - p.nh;
    let at = (row * p.nk + kh) * n;
    let ring = (((p.pos0 + row) % p.cap) * p.nk + kh) * n;
    let x = yk[at + e];
    let y = yk[at + half + e];
    let u = x * c - y * sn;
    let v = y * c + x * sn;
    let inv = inverseSqrt(rsum32(lid, dot(u, u) + dot(v, v)) + p.eps);
    ko[ring + e] = u * inv;
    ko[ring + half + e] = v * inv;
    vo[ring + e] = yv[at + e];
    vo[ring + half + e] = yv[at + half + e];
}

// ── Scores: one thread per key of a KV head, its four query heads at once, so a
// key is read once. Short threads and no shared memory: a workgroup stays on one
// SM, and a layer has 4 KV heads, so workgroups that held a whole head left most
// of the card idle (230 µs a layer at a full window).
@group(0) @binding(10) var<storage, read> aq: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> ak: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> sc: array<vec4<f32>>;

fn hs(v: vec4<f32>) -> f32 { return v.x + v.y + v.z + v.w; }

@compute @workgroup_size(32)
fn dory_score(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    let kh = g.y;
    let row = g.z;
    let pos = p.pos0 + row;
    let lo = select(0u, pos + 1u - p.window, pos + 1u > p.window);
    let n = pos + 1u - lo;
    if (i >= n) { return; }
    let e4 = p.hd / 4u;
    let q0 = (row * p.nh + kh * 4u) * e4;
    let q1 = q0 + e4;
    let q2 = q1 + e4;
    let q3 = q2 + e4;
    let kb = (((lo + i) % p.cap) * p.nk + kh) * e4;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var e = 0u; e < e4; e += 1u) {
        let k = ak[kb + e];
        a0 += k * aq[q0 + e];
        a1 += k * aq[q1 + e];
        a2 += k * aq[q2 + e];
        a3 += k * aq[q3 + e];
    }
    sc[(row * p.nk + kh) * p.ncap + i] = vec4<f32>(hs(a0), hs(a1), hs(a2), hs(a3)) * p.scale;
}

// ── The softmax of a KV head's four queries: each key's weight, and 1/Σ.
@group(0) @binding(17) var<storage, read_write> sf: array<vec4<f32>>;
@group(0) @binding(18) var<storage, read_write> sfi: array<vec4<f32>>;

@compute @workgroup_size(LANES)
fn dory_soft(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kh = w.x;
    let row = w.y;
    let pos = p.pos0 + row;
    let lo = select(0u, pos + 1u - p.window, pos + 1u > p.window);
    let n = pos + 1u - lo;
    let sb = (row * p.nk + kh) * p.ncap;
    var m = vec4<f32>(-1.0e30);
    for (var i = lid; i < n; i += LANESu) { m = max(m, sf[sb + i]); }
    let mx = rmax4(lid, m);
    var den = vec4<f32>(0.0);
    for (var i = lid; i < n; i += LANESu) {
        let e = exp(sf[sb + i] - mx);
        sf[sb + i] = e;
        den += e;
    }
    let total = rsum4(lid, den);
    if (lid == 0u) { sfi[row * p.nk + kh] = 1.0 / total; }
}

// ── The value sum in pieces: one thread per (KV head, chunk of keys, vec4 of the
// head) over the same short-thread grid; the next stage adds the pieces. A short
// window has fewer (`p.ch`).
@group(0) @binding(12) var<storage, read> av: array<vec4<f32>>;
@group(0) @binding(19) var<storage, read> vsc: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> pz: array<vec4<f32>>;
const CH: u32 = 8u;

@compute @workgroup_size(32)
fn dory_sum(@builtin(global_invocation_id) g: vec3<u32>) {
    let e4 = p.hd / 4u;
    let t = g.x;
    let d = t % e4;
    let ch = (t / e4) % p.ch;
    let kh = t / (e4 * p.ch);
    let row = g.y;
    if (kh >= p.nk) { return; }
    let pos = p.pos0 + row;
    let lo = select(0u, pos + 1u - p.window, pos + 1u > p.window);
    let n = pos + 1u - lo;
    let per = (n + p.ch - 1u) / p.ch;
    let sb = (row * p.nk + kh) * p.ncap;
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    let last = min((ch + 1u) * per, n);
    for (var i = ch * per; i < last; i += 1u) {
        let a = vsc[sb + i];
        let v = av[(((lo + i) % p.cap) * p.nk + kh) * e4 + d];
        o0 += a.x * v;
        o1 += a.y * v;
        o2 += a.z * v;
        o3 += a.w * v;
    }
    let at = (((row * p.nk + kh) * CH + ch) * 4u) * e4 + d;
    pz[at] = o0;
    pz[at + e4] = o1;
    pz[at + 2u * e4] = o2;
    pz[at + 3u * e4] = o3;
}

// ── The pieces added, 1/Σ and the gate. The reference normalizes this row before
// the output projection, and the residual normalizes what comes out of it: a scale
// goes through a matrix unchanged, so only the second one runs.
@group(0) @binding(13) var<storage, read> ag: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read> asg: array<vec4<f32>>;
@group(0) @binding(20) var<storage, read> gpz: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> gsi: array<vec4<f32>>;
@group(0) @binding(31) var<storage, read_write> nz: array<vec4<f32>>;

@compute @workgroup_size(32)
fn dory_gate(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.y;
    let e = g.x;
    let e4 = p.hd / 4u;
    let n = p.nh * e4;
    if (e >= n) { return; }
    let base = row * n;
    let d = e % e4;
    let hi = e / e4;
    let kh = hi / 4u;
    let h = hi % 4u;
    var acc = vec4<f32>(0.0);
    let pb = (row * p.nk + kh) * CH;
    for (var ch = 0u; ch < p.ch; ch += 1u) {
        acc += gpz[((pb + ch) * 4u + h) * e4 + d];
    }
    let inv = gsi[row * p.nk + kh][h];
    nz[base + e] = acc * inv / (1.0 + exp(-(ag[base + e] * asg[e])));
}

// ── The MLP activation, in place in the gate's output (the norm as in `dory_gate`).
@group(0) @binding(21) var<storage, read_write> sg_g: array<vec4<f32>>;
@group(0) @binding(22) var<storage, read> sg_u: array<vec4<f32>>;
@group(0) @binding(23) var<storage, read> sg_a: array<vec4<f32>>;
@group(0) @binding(24) var<storage, read> sg_b: array<vec4<f32>>;

@compute @workgroup_size(32)
fn dory_silu(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p.ffn / 4u;
    let e = id.x;
    if (e >= n) { return; }
    let base = id.y * n;
    let g = sg_g[base + e] * sg_a[e];
    let u = sg_u[base + e] * sg_b[e];
    sg_g[base + e] = g / (1.0 + exp(-g)) * u;
}

// ── The nGPT residual.
@group(0) @binding(25) var<storage, read_write> sl_h: array<vec4<f32>>;
@group(0) @binding(26) var<storage, read> sl_o: array<vec4<f32>>;
@group(0) @binding(27) var<storage, read> sl_a: array<vec4<f32>>;

@compute @workgroup_size(LANES)
fn dory_slerp(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = p.hidden / 4u;
    let base = w.x * n;
    var sx = 0.0;
    var so = 0.0;
    for (var e = lid; e < n; e += LANESu) {
        let x = sl_h[base + e];
        let o = sl_o[base + e];
        sx += dot(x, x);
        so += dot(o, o);
    }
    let ix = inverseSqrt(rsum1(lid, sx) + p.eps);
    let io = inverseSqrt(rsum1(lid, so) + p.eps);
    var t = 0.0;
    for (var e = lid; e < n; e += LANESu) {
        let a = sl_h[base + e] * ix;
        let y = a + sl_a[e] * (sl_o[base + e] * io - a);
        sl_h[base + e] = y;
        t += dot(y, y);
    }
    let r = inverseSqrt(rsum1(lid, t) + p.eps);
    for (var e = lid; e < n; e += LANESu) {
        sl_h[base + e] = sl_h[base + e] * r;
    }
}

// ── The loop's edges: mode 0 enters the first pass (input = state = h),
// mode 1 goes between passes (state = ĥ), mode 2 leaves the last (h = ĥ).
@group(0) @binding(28) var<storage, read_write> mx_h: array<vec4<f32>>;
@group(0) @binding(29) var<storage, read_write> mx_i: array<vec4<f32>>;
@group(0) @binding(30) var<storage, read_write> mx_s: array<vec4<f32>>;

@compute @workgroup_size(LANES)
fn dory_mix0(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = p.hidden / 4u;
    let base = w.x * n;
    var s = 0.0;
    for (var e = lid; e < n; e += LANESu) {
        let v = mx_h[base + e];
        mx_i[base + e] = v;
        mx_s[base + e] = v;
        let y = v + v;
        mx_h[base + e] = y;
        s += dot(y, y);
    }
    let r = inverseSqrt(rsum1(lid, s) + p.eps);
    for (var e = lid; e < n; e += LANESu) {
        mx_h[base + e] = mx_h[base + e] * r;
    }
}

@compute @workgroup_size(LANES)
fn dory_mix1(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = p.hidden / 4u;
    let base = w.x * n;
    var s = 0.0;
    for (var e = lid; e < n; e += LANESu) {
        let v = mx_h[base + e];
        s += dot(v, v);
    }
    let r = inverseSqrt(rsum1(lid, s) + p.eps);
    var t = 0.0;
    for (var e = lid; e < n; e += LANESu) {
        let st = mx_h[base + e] * r;
        mx_s[base + e] = st;
        let y = st + mx_i[base + e];
        mx_h[base + e] = y;
        t += dot(y, y);
    }
    let r2 = inverseSqrt(rsum1(lid, t) + p.eps);
    for (var e = lid; e < n; e += LANESu) {
        mx_h[base + e] = mx_h[base + e] * r2;
    }
}

@compute @workgroup_size(LANES)
fn dory_mix2(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = p.hidden / 4u;
    let base = w.x * n;
    var s = 0.0;
    for (var e = lid; e < n; e += LANESu) {
        let v = mx_h[base + e];
        s += dot(v, v);
    }
    let r = inverseSqrt(rsum1(lid, s) + p.eps);
    for (var e = lid; e < n; e += LANESu) {
        mx_h[base + e] = mx_h[base + e] * r;
    }
}

// ── The gate's two edges: mode 3 reads out after a pass and keeps the state,
// mode 4 starts the next pass from the kept state.
@compute @workgroup_size(LANES)
fn dory_mix3(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = p.hidden / 4u;
    let base = w.x * n;
    var s = 0.0;
    for (var e = lid; e < n; e += LANESu) {
        let v = mx_h[base + e];
        s += dot(v, v);
    }
    let r = inverseSqrt(rsum1(lid, s) + p.eps);
    for (var e = lid; e < n; e += LANESu) {
        let st = mx_h[base + e] * r;
        mx_h[base + e] = st;
        mx_s[base + e] = st;
    }
}

@compute @workgroup_size(LANES)
fn dory_mix4(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = p.hidden / 4u;
    let base = w.x * n;
    var t = 0.0;
    for (var e = lid; e < n; e += LANESu) {
        let y = mx_s[base + e] + mx_i[base + e];
        mx_h[base + e] = y;
        t += dot(y, y);
    }
    let r2 = inverseSqrt(rsum1(lid, t) + p.eps);
    for (var e = lid; e < n; e += LANESu) {
        mx_h[base + e] = mx_h[base + e] * r2;
    }
}

// ── O(1) on a full layer (`nystrom`): the far, push and attend kernels are the
// engine's. These take the prompt's queries for the seal, a later pass's rewrite
// of this position, and the gate over the kernel's output.
const QT_STRIDE: u32 = 8u;
@group(0) @binding(34) var<storage, read> qc_in: array<vec4<f32>>;
@group(0) @binding(35) var<storage, read_write> qc_out: array<vec4<f32>>;
@compute @workgroup_size(32)
fn dory_qcopy(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.nh * (p.hd / 4u);
    let e = g.x;
    let at = p.pos0 + g.y;
    if (e >= n || at % QT_STRIDE != 0u) { return; }
    qc_out[at / QT_STRIDE * n + e] = qc_in[g.y * n + e];
}

@group(0) @binding(33) var<storage, read_write> ob_meta: array<u32>;
@compute @workgroup_size(1)
fn dory_o1_back(@builtin(global_invocation_id) id: vec3<u32>) {
    let g = id.x * 4u;
    let wg = ob_meta[g + 3u];
    if (ob_meta[g] == wg) {
        ob_meta[g + 1u] = (ob_meta[g + 1u] + wg - 1u) % wg;
        ob_meta[g + 2u] = ob_meta[g + 2u] - 1u;
    } else {
        ob_meta[g] = ob_meta[g] - 1u;
    }
}

@group(0) @binding(32) var<storage, read> g1_z: array<vec4<f32>>;
@compute @workgroup_size(32)
fn dory_gate1(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.nh * (p.hd / 4u);
    let e = g.x;
    if (e >= n) { return; }
    let at = g.y * n + e;
    nz[at] = g1_z[at] / (1.0 + exp(-(ag[at] * asg[e])));
}

// ── A sealed prompt chunk's keys and values, kept for the choice of exact rows.
@group(0) @binding(60) var<storage, read> ar_k: array<vec4<f32>>;
@group(0) @binding(61) var<storage, read> ar_v: array<vec4<f32>>;
@group(0) @binding(62) var<storage, read_write> ar_ko: array<vec4<f32>>;
@group(0) @binding(63) var<storage, read_write> ar_vo: array<vec4<f32>>;

@compute @workgroup_size(32)
fn dory_arch(@builtin(global_invocation_id) g: vec3<u32>) {
    let n = p.nk * (p.hd / 4u);
    let e = g.x;
    if (e >= n) { return; }
    let ring = ((p.pos0 + g.y) % p.cap) * n + e;
    ar_ko[ring] = ar_k[g.y * n + e];
    ar_vo[ring] = ar_v[g.y * n + e];
}

// ── The far field's attention, one row, in the exact path's short threads: a
// workgroup stays on one SM, and the far state's four KV heads would hold four of them.
struct OU { hpg: u32, m: u32, w: u32, ns: u32, d: u32, dv: u32, scale: f32, rect: u32 };
@group(0) @binding(50) var<uniform> ou: OU;
@group(0) @binding(40) var<storage, read> om: array<u32>;
@group(0) @binding(41) var<storage, read> ork: array<vec4<f32>>;
@group(0) @binding(42) var<storage, read> osk: array<vec4<f32>>;
@group(0) @binding(43) var<storage, read> okt: array<vec4<f32>>;
@group(0) @binding(44) var<storage, read> omu: array<f32>;
@group(0) @binding(45) var<storage, read> omz: array<f32>;
@group(0) @binding(46) var<storage, read> oth: array<vec4<f32>>;
@group(0) @binding(47) var<storage, read> orv: array<vec4<f32>>;
@group(0) @binding(48) var<storage, read> osv: array<vec4<f32>>;
@group(0) @binding(51) var<storage, read_write> ofs: array<vec4<f32>>;
@group(0) @binding(52) var<storage, read> ofsr: array<vec4<f32>>;
@group(0) @binding(53) var<storage, read_write> onm: array<vec4<f32>>;
@group(0) @binding(54) var<storage, read> onmr: array<vec4<f32>>;
@group(0) @binding(55) var<storage, read_write> ohd: array<vec4<f32>>;
@group(0) @binding(56) var<storage, read> ohdr: array<vec4<f32>>;
@group(0) @binding(57) var<storage, read_write> ogn: array<f32>;
@group(0) @binding(58) var<storage, read> ognr: array<f32>;

// Near scores: a thread per key of the sink and the window.
@compute @workgroup_size(32)
fn dory_os(@builtin(global_invocation_id) g: vec3<u32>) {
    let s = g.x;
    let kh = g.y;
    let n = ou.ns + om[kh * 4u];
    if (s >= n) { return; }
    let e4 = p.hd / 4u;
    let q0 = kh * 4u * e4;
    let q1 = q0 + e4;
    let q2 = q1 + e4;
    let q3 = q2 + e4;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    if (s < ou.ns) {
        let kb = (kh * ou.ns + s) * e4;
        for (var e = 0u; e < e4; e += 1u) {
            let k = osk[kb + e];
            a0 += k * aq[q0 + e];
            a1 += k * aq[q1 + e];
            a2 += k * aq[q2 + e];
            a3 += k * aq[q3 + e];
        }
    } else {
        let kb = (kh * ou.w + s - ou.ns) * e4;
        for (var e = 0u; e < e4; e += 1u) {
            let k = ork[kb + e];
            a0 += k * aq[q0 + e];
            a1 += k * aq[q1 + e];
            a2 += k * aq[q2 + e];
            a3 += k * aq[q3 + e];
        }
    }
    sc[kh * p.ncap + s] = vec4<f32>(hs(a0), hs(a1), hs(a2), hs(a3)) * p.scale;
}

// Landmark scores, the same way.
@compute @workgroup_size(32)
fn dory_ol(@builtin(global_invocation_id) g: vec3<u32>) {
    let a = g.x;
    let kh = g.y;
    if (a >= ou.m) { return; }
    let e4 = p.hd / 4u;
    let q0 = kh * 4u * e4;
    let q1 = q0 + e4;
    let q2 = q1 + e4;
    let q3 = q2 + e4;
    let kb = (kh * ou.m + a) * e4;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var e = 0u; e < e4; e += 1u) {
        let k = okt[kb + e];
        a0 += k * aq[q0 + e];
        a1 += k * aq[q1 + e];
        a2 += k * aq[q2 + e];
        a3 += k * aq[q3 + e];
    }
    ofs[kh * ou.m + a] = vec4<f32>(hs(a0), hs(a1), hs(a2), hs(a3)) * p.scale;
}

// The largest near score of a KV head's four queries.
@compute @workgroup_size(LANES)
fn dory_omax(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kh = w.x;
    let n = ou.ns + om[kh * 4u];
    let sb = kh * p.ncap;
    var m = vec4<f32>(-1.0e30);
    for (var i = lid; i < n; i += LANESu) { m = max(m, sf[sb + i]); }
    let mx = rmax4(lid, m);
    if (lid == 0u) { onm[kh] = mx; }
}

// A query head's far part: the landmark weights through its inverse, one gain per
// landmark, the far mass, and the common exponent the near scores will use.
@compute @workgroup_size(8)
fn dory_ofar(@builtin(global_invocation_id) g: vec3<u32>) {
    let gh = g.x;
    if (gh >= p.nh) { return; }
    let kh = gh / 4u;
    let h = gh % 4u;
    let m = ou.m;
    var c_all = onmr[kh][h];
    var far_den = 0.0;
    var have = 0.0;
    if (om[kh * 4u + 2u] > 0u) {
        var f = -3.0e38;
        var fe: array<f32, 32>;
        var u: array<f32, 32>;
        for (var a = 0u; a < m; a += 1u) {
            let v = ofsr[kh * m + a][h];
            fe[a] = v;
            f = max(f, v);
        }
        for (var a = 0u; a < m; a += 1u) {
            let e = exp(fe[a] - f);
            let row = (gh * m + a) * m;
            for (var b = 0u; b < m; b += 1u) { u[b] += e * omu[row + b]; }
        }
        let mzb = gh * 2u * m;
        for (var b = 0u; b < m; b += 1u) { c_all = max(c_all, f + omz[mzb + b]); }
        for (var b = 0u; b < m; b += 1u) {
            let gain = u[b] * exp(f + omz[mzb + b] - c_all);
            ogn[gh * m + b] = gain;
            far_den += gain * omz[mzb + m + b];
        }
        if (far_den >= 0.0) { have = 1.0; } else { far_den = 0.0; }
    }
    ohd[gh] = vec4<f32>(c_all, far_den, have, 0.0);
}

// The near weights under that exponent, and 1/Σ with the far mass in it.
@compute @workgroup_size(LANES)
fn dory_osoft(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kh = w.x;
    let n = ou.ns + om[kh * 4u];
    let sb = kh * p.ncap;
    let h0 = ohdr[kh * 4u];
    let h1 = ohdr[kh * 4u + 1u];
    let h2 = ohdr[kh * 4u + 2u];
    let h3 = ohdr[kh * 4u + 3u];
    let c = vec4<f32>(h0.x, h1.x, h2.x, h3.x);
    let far = vec4<f32>(h0.y, h1.y, h2.y, h3.y);
    var den = vec4<f32>(0.0);
    for (var i = lid; i < n; i += LANESu) {
        let e = exp(sf[sb + i] - c);
        sf[sb + i] = e;
        den += e;
    }
    let total = rsum4(lid, den) + far;
    if (lid == 0u) { sfi[kh] = 1.0 / max(total, vec4<f32>(1.0e-30)); }
}

// The near value sum in eight pieces over the short-thread grid.
@compute @workgroup_size(32)
fn dory_osum(@builtin(global_invocation_id) g: vec3<u32>) {
    let e4 = p.hd / 4u;
    let t = g.x;
    let d = t % e4;
    let ch = (t / e4) % CH;
    let kh = t / (e4 * CH);
    if (kh >= p.nk) { return; }
    let n = ou.ns + om[kh * 4u];
    let per = (n + CH - 1u) / CH;
    let sb = kh * p.ncap;
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    var o2 = vec4<f32>(0.0);
    var o3 = vec4<f32>(0.0);
    let last = min((ch + 1u) * per, n);
    for (var i = ch * per; i < last; i += 1u) {
        let a = vsc[sb + i];
        var v: vec4<f32>;
        if (i < ou.ns) { v = osv[(kh * ou.ns + i) * e4 + d]; } else { v = orv[(kh * ou.w + i - ou.ns) * e4 + d]; }
        o0 += a.x * v;
        o1 += a.y * v;
        o2 += a.z * v;
        o3 += a.w * v;
    }
    let at = ((kh * CH + ch) * 4u) * e4 + d;
    pz[at] = o0;
    pz[at + e4] = o1;
    pz[at + 2u * e4] = o2;
    pz[at + 3u * e4] = o3;
}

// The pieces, the far values, 1/Σ and the gate.
@compute @workgroup_size(32)
fn dory_ogate(@builtin(global_invocation_id) g: vec3<u32>) {
    let e = g.x;
    let e4 = p.hd / 4u;
    if (e >= p.nh * e4) { return; }
    let d = e % e4;
    let gh = e / e4;
    let kh = gh / 4u;
    let h = gh % 4u;
    var acc = vec4<f32>(0.0);
    let pb = kh * CH;
    for (var ch = 0u; ch < CH; ch += 1u) {
        acc += gpz[((pb + ch) * 4u + h) * e4 + d];
    }
    if (ohdr[gh].z > 0.5) {
        for (var b = 0u; b < ou.m; b += 1u) {
            acc += ognr[gh * ou.m + b] * oth[(gh * ou.m + b) * e4 + d];
        }
    }
    nz[e] = acc * gsi[kh][h] / (1.0 + exp(-(ag[e] * asg[e])));
}
"#;

/// A head's reduction, on a device whose subgroups are the 32 lanes of the workgroup.
const REDUCE_SG: &str = r#"
fn rsum32(lid: u32, v: f32) -> f32 { return subgroupAdd(v); }
"#;

/// The same on any device.
const REDUCE_TREE32: &str = r#"
var<workgroup> r_c: array<f32, 32>;
fn rsum32(lid: u32, v: f32) -> f32 {
    r_c[lid] = v;
    workgroupBarrier();
    for (var st = 16u; st > 0u; st = st >> 1u) {
        if (lid < st) { r_c[lid] = r_c[lid] + r_c[lid + st]; }
        workgroupBarrier();
    }
    let r = r_c[0];
    workgroupBarrier();
    return r;
}
"#;

/// The row-wide reductions: a workgroup of two warps and a barrier tree beat one
/// warp and `subgroupAdd` (16 against 10 µs a residual), which stalls on every load.
const REDUCE_TREE: &str = r#"
var<workgroup> r_a: array<f32, 64>;
var<workgroup> r_b: array<vec4<f32>, 64>;
fn rsum1(lid: u32, v: f32) -> f32 {
    r_a[lid] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {
        if (lid < st) { r_a[lid] = r_a[lid] + r_a[lid + st]; }
        workgroupBarrier();
    }
    let r = r_a[0];
    workgroupBarrier();
    return r;
}
fn rsum4(lid: u32, v: vec4<f32>) -> vec4<f32> {
    r_b[lid] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {
        if (lid < st) { r_b[lid] = r_b[lid] + r_b[lid + st]; }
        workgroupBarrier();
    }
    let r = r_b[0];
    workgroupBarrier();
    return r;
}
fn rmax4(lid: u32, v: vec4<f32>) -> vec4<f32> {
    r_b[lid] = v;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {
        if (lid < st) { r_b[lid] = max(r_b[lid], r_b[lid + st]); }
        workgroupBarrier();
    }
    let r = r_b[0];
    workgroupBarrier();
    return r;
}
"#;

/// One row in this many keeps its queries for the seal: landmark queries are means
/// over long segments, and a row's queries take 16 KB per full layer and loop.
const QT_STRIDE: usize = 8;
/// A prompt this long seals before it ends: 8 KB a row per full layer and loop.
const SEAL_MAX: usize = 8192;
/// Rows of a sealed prompt kept on the card for the choice of exact ones.
const ARCH_MAX: usize = 65536;
/// Rows of a prompt chunk.
pub(crate) const CHUNK: usize = 64;
/// Pieces of a value sum (`CH` in the shader).
const CH: usize = 8;
/// `dory_attend` assumes this many query heads per KV head.
const REP: usize = 4;
const WG_BYTES: u32 = 2 * 1024;

struct Pipe {
    p: wgpu::ComputePipeline,
    l: wgpu::BindGroupLayout,
}

struct Pipes {
    prep: Pipe,
    score: Pipe,
    soft: Pipe,
    sum: Pipe,
    gate: Pipe,
    /// Threads of the row-sized stages' workgroup.
    lanes: u32,
    silu: Pipe,
    slerp: Pipe,
    mix: [Pipe; 5],
    qcopy: Pipe,
    o1_back: Pipe,
    gate1: Pipe,
    arch: Pipe,
    far: FarPipes,
}

/// The far field's attention stages, in order.
struct FarPipes {
    os: Pipe,
    ol: Pipe,
    omax: Pipe,
    ofar: Pipe,
    osoft: Pipe,
    osum: Pipe,
    ogate: Pipe,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            if c.device.limits().max_compute_workgroup_storage_size < WG_BYTES {
                return None;
            }
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let sg = c.device.features().contains(wgpu::Features::SUBGROUP)
                && c.adapter_info.subgroup_min_size == 32
                && c.adapter_info.subgroup_max_size == 32
                && std::env::var("CMF_DORY_SG").as_deref() != Ok("0");
            let lanes = 64u32;
            let body = SRC.replace("LANESu", &format!("{lanes}u")).replace("LANES", &lanes.to_string());
            let source = format!("{}{body}{REDUCE_TREE}", if sg { REDUCE_SG } else { REDUCE_TREE32 });
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("dory"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let mk = |entry: &str| {
                let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &m,
                    entry_point: Some(entry),
                    compilation_options: pco(),
                    cache: None,
                });
                let l = p.get_bind_group_layout(0);
                Pipe { p, l }
            };
            let out = Pipes {
                prep: mk("dory_prep"),
                score: mk("dory_score"),
                soft: mk("dory_soft"),
                sum: mk("dory_sum"),
                gate: mk("dory_gate"),
                lanes,
                silu: mk("dory_silu"),
                slerp: mk("dory_slerp"),
                mix: [mk("dory_mix0"), mk("dory_mix1"), mk("dory_mix2"), mk("dory_mix3"), mk("dory_mix4")],
                qcopy: mk("dory_qcopy"),
                o1_back: mk("dory_o1_back"),
                gate1: mk("dory_gate1"),
                arch: mk("dory_arch"),
                far: FarPipes {
                    os: mk("dory_os"),
                    ol: mk("dory_ol"),
                    omax: mk("dory_omax"),
                    ofar: mk("dory_ofar"),
                    osoft: mk("dory_osoft"),
                    osum: mk("dory_osum"),
                    ogate: mk("dory_ogate"),
                },
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("dory kernels rejected: {e}");
                return None;
            }
            Some(out)
        })
        .as_ref()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Codec {
    Q4,
    Q8,
}

struct Mat {
    buf: wgpu::Buffer,
    codec: Codec,
    rows: usize,
    cols: usize,
}

struct AttnDev {
    q: Mat,
    k: Mat,
    v: Mat,
    g: Mat,
    o: Mat,
    sq: wgpu::Buffer,
    sg: wgpu::Buffer,
    alpha: wgpu::Buffer,
    sliding: bool,
    ordinal: usize,
}

struct MlpDev {
    gate: Mat,
    up: Mat,
    down: Mat,
    sg: wgpu::Buffer,
    su: wgpu::Buffer,
    alpha: wgpu::Buffer,
}

enum LayerDev {
    Attn(Box<AttnDev>),
    Mlp(Box<MlpDev>),
}

struct Slot {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    cap: usize,
}

#[derive(Clone)]
struct Step {
    pipe: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    x: u32,
    y: u32,
    /// Its stage in `CMF_DORY_KEEP`.
    kind: Kind,
}

/// The stages of a forward, for a keep-one table: `CMF_DORY_KEEP=prep,attend`
/// runs only these dispatches (the logits are then garbage, the timing is not).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Qkv,
    G,
    Prep,
    Score,
    Soft,
    Sum,
    Gate,
    O,
    Gu,
    Silu,
    D,
    Slerp,
    Mix,
    Far,
    Push,
    FarAttend,
}

fn keep_mask() -> u32 {
    static K: OnceLock<u32> = OnceLock::new();
    *K.get_or_init(|| {
        let Ok(v) = std::env::var("CMF_DORY_KEEP") else { return !0 };
        let mut m = 0u32;
        for w in v.split(',') {
            m |= match w {
                "qkv" => 1 << Kind::Qkv as u32,
                "g" => 1 << Kind::G as u32,
                "prep" => 1 << Kind::Prep as u32,
                "score" => 1 << Kind::Score as u32,
                "soft" => 1 << Kind::Soft as u32,
                "sum" => 1 << Kind::Sum as u32,
                "gate" => 1 << Kind::Gate as u32,
                "o" => 1 << Kind::O as u32,
                "gu" => 1 << Kind::Gu as u32,
                "silu" => 1 << Kind::Silu as u32,
                "d" => 1 << Kind::D as u32,
                "slerp" => 1 << Kind::Slerp as u32,
                "mix" => 1 << Kind::Mix as u32,
                "far" => 1 << Kind::Far as u32,
                "push" => 1 << Kind::Push as u32,
                "farattend" => 1 << Kind::FarAttend as u32,
                _ => 0,
            };
        }
        m
    })
}

struct Scratch {
    h: wgpu::Buffer,
    input: wgpu::Buffer,
    state: wgpu::Buffer,
    hlast: wgpu::Buffer,
    yq: wgpu::Buffer,
    yk: wgpu::Buffer,
    yv: wgpu::Buffer,
    yg: wgpu::Buffer,
    qo: wgpu::Buffer,
    z: wgpu::Buffer,
    /// The value sum's pieces and each KV head's 1/Σ.
    pz: wgpu::Buffer,
    sinv: wgpu::Buffer,
    fg: wgpu::Buffer,
    fu: wgpu::Buffer,
    o: wgpu::Buffer,
    /// One head row, and the whole chunk's for the scoring path.
    logits: wgpu::Buffer,
    logits_k: wgpu::Buffer,
    staging: wgpu::Buffer,
    staging_k: wgpu::Buffer,
    scores: wgpu::Buffer,
    scores_cap: usize,
    /// Uniform per layer kind: sliding, full, and a sealed full layer's
    /// (its K and V go to `kt`, `vt`, a one-row ring).
    u: [wgpu::Buffer; 3],
    rope: [wgpu::Buffer; 2],
    kt: wgpu::Buffer,
    vt: wgpu::Buffer,
    attn: wgpu::Buffer,
    /// `o1_batch`'s scratch.
    ob: wgpu::Buffer,
    /// The far stages': landmark scores, near maxima, per head and gains.
    ofs: wgpu::Buffer,
    onm: wgpu::Buffer,
    ohd: wgpu::Buffer,
    ogn: wgpu::Buffer,
}

pub(crate) struct Device {
    layers: std::sync::Arc<Vec<LayerDev>>,
    head: std::sync::Arc<Mat>,
    s: Scratch,
    slots: Vec<Vec<Slot>>,
    /// `(loops, rows)` of a recorded plan; dropped when a buffer it binds grows.
    plans: HashMap<(usize, usize), std::sync::Arc<Vec<Step>>>,
    head_plan: HashMap<usize, std::sync::Arc<Step>>,
    /// The parts of a token the early exit runs one by one, on the last-loop cache.
    parts: Option<std::sync::Arc<Parts>>,
    gate_tokens: u64,
    gate_one: u64,
    exits: ExitRate,
    /// Key of the sequence's O(1) states in `Ctx::o1m`.
    id: u64,
    sealed: bool,
    /// Rows of the sealed prompt still kept for the choice of exact ones; 0: none.
    arch_rows: usize,
    seal_epoch: u64,
    o1_refused: bool,
    /// The queries of a full slot up to the seal.
    qtrace: HashMap<(usize, usize), wgpu::Buffer>,
}

struct Parts {
    input: Vec<Step>,
    pass: Vec<Step>,
    /// A later pass and read-out: on a sealed layer they rewrite this position.
    pass2: Vec<Step>,
    out: Vec<Step>,
    out2: Vec<Step>,
    /// `mix[i]` of the shader.
    mix: Vec<Step>,
}

static DEVICES: Mutex<Option<HashMap<u64, Device>>> = Mutex::new(None);
/// One copy of a file's weights, however many sequences (server slots) read it.
#[allow(clippy::type_complexity)]
static WEIGHTS: Mutex<Option<HashMap<usize, (std::sync::Arc<Vec<LayerDev>>, std::sync::Arc<Mat>)>>> = Mutex::new(None);

fn buf(c: &Ctx, label: &str, bytes: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16) as u64,
        usage,
        mapped_at_creation: false,
    })
}

const STORE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

fn store_f32(c: &Ctx, label: &str, v: &[f32]) -> wgpu::Buffer {
    let b = buf(c, label, v.len() * 4, STORE);
    c.queue.write_buffer(&b, 0, bytemuck::cast_slice(v));
    b
}

fn upload(c: &Ctx, t: &crate::qtensor::QTensor, pending: &mut u64) -> Option<Mat> {
    let codec = if t.q4tp_mapped().is_some() {
        Codec::Q4
    } else if t.q8_2f_codes().is_some() {
        Codec::Q8
    } else {
        return None;
    };
    let bytes = t.mapped_bytes()?;
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dory-weight"),
        // A 4-byte word kernel over a tail that may be shorter.
        size: (bytes.len() as u64 + 3) & !3,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue.write_buffer(&b, 0, bytes);
    *pending += bytes.len() as u64;
    // A long run of write_buffer without a submit holds its staging in host RAM.
    if *pending > 512 << 20 {
        c.queue.submit(None);
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        *pending = 0;
    }
    Some(Mat { buf: b, codec, rows: t.rows(), cols: t.cols() })
}

/// Can the device path run this file? Decided before anything moves.
pub(crate) fn admits(m: &Dory) -> bool {
    let c = &m.cfg;
    if c.nh != c.nk * REP || c.hd != 256 || c.hidden % 64 != 0 || c.ffn % 64 != 0 {
        return false;
    }
    let plain = |t: &crate::qtensor::QTensor| t.q4tp_mapped().is_some() || t.q8_2f_codes().is_some();
    plain(&m.head)
        && m.layers.iter().all(|l| match l {
            Layer::Attn(w) => plain(&w.q) && plain(&w.k) && plain(&w.v) && plain(&w.g) && plain(&w.o),
            Layer::Mlp(w) => plain(&w.gate) && plain(&w.up) && plain(&w.down),
        })
}

fn build(c: &Ctx, m: &Dory) -> Option<Device> {
    pipes(c)?;
    let cf = &m.cfg;
    let t0 = std::time::Instant::now();
    let mut pending = 0u64;
    let key = m.file.as_ref().map_or(m as *const Dory as usize, |f| Arc::as_ptr(f) as usize);
    let cached = WEIGHTS.lock().unwrap().as_ref().and_then(|w| w.get(&key).cloned());
    let mut layers = Vec::with_capacity(m.layers.len());
    for l in m.layers.iter().filter(|_| cached.is_none()) {
        layers.push(match l {
            Layer::Attn(w) => LayerDev::Attn(Box::new(AttnDev {
                q: upload(c, &w.q, &mut pending)?,
                k: upload(c, &w.k, &mut pending)?,
                v: upload(c, &w.v, &mut pending)?,
                g: upload(c, &w.g, &mut pending)?,
                o: upload(c, &w.o, &mut pending)?,
                sq: store_f32(c, "dory-sq", &w.sq),
                sg: store_f32(c, "dory-sg", &w.sg),
                alpha: store_f32(c, "dory-alpha", &w.alpha),
                sliding: w.sliding,
                ordinal: w.ordinal,
            })),
            Layer::Mlp(w) => LayerDev::Mlp(Box::new(MlpDev {
                gate: upload(c, &w.gate, &mut pending)?,
                up: upload(c, &w.up, &mut pending)?,
                down: upload(c, &w.down, &mut pending)?,
                sg: store_f32(c, "dory-sg", &w.sg),
                su: store_f32(c, "dory-su", &w.su),
                alpha: store_f32(c, "dory-alpha", &w.alpha),
            })),
        });
    }
    let (layers, head) = match cached {
        Some(w) => w,
        None => {
            let w = (Arc::new(layers), Arc::new(upload(c, &m.head, &mut pending)?));
            WEIGHTS.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, w.clone());
            w
        }
    };
    c.queue.submit(None);
    let k = CHUNK;
    let q_size = cf.nh * cf.hd;
    let kv_row = cf.nk * cf.hd;
    let hk = k * cf.hidden * 4;
    let rd = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
    let unif = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let s = Scratch {
        h: buf(c, "dory-h", hk, STORE),
        input: buf(c, "dory-input", hk, STORE),
        state: buf(c, "dory-state", hk, STORE),
        hlast: buf(c, "dory-hlast", cf.hidden * 4, STORE),
        yq: buf(c, "dory-yq", k * q_size * 4, STORE),
        yk: buf(c, "dory-yk", k * kv_row * 4, STORE),
        yv: buf(c, "dory-yv", k * kv_row * 4, STORE),
        yg: buf(c, "dory-yg", k * q_size * 4, STORE),
        qo: buf(c, "dory-qo", k * q_size * 4, STORE),
        z: buf(c, "dory-z", k * q_size * 4, STORE),
        pz: buf(c, "dory-pz", k * q_size * CH * 4, STORE),
        sinv: buf(c, "dory-sinv", k * cf.nk * 16, STORE),
        fg: buf(c, "dory-fg", k * cf.ffn * 4, STORE),
        fu: buf(c, "dory-fu", k * cf.ffn * 4, STORE),
        o: buf(c, "dory-o", hk, STORE),
        logits: buf(c, "dory-logits", cf.vocab * 4, STORE),
        logits_k: buf(c, "dory-logits-k", k * cf.vocab * 4, STORE),
        staging: buf(c, "dory-staging", cf.vocab * 4, rd),
        staging_k: buf(c, "dory-staging-k", k * cf.vocab * 4, rd),
        scores: buf(c, "dory-scores", (cf.window + k) * k * cf.nk * 16, STORE),
        scores_cap: cf.window + k,
        u: [buf(c, "dory-u", 64, unif), buf(c, "dory-u", 64, unif), buf(c, "dory-u", 64, unif)],
        rope: [buf(c, "dory-rope", k * cf.hd * 4, STORE), buf(c, "dory-rope", k * cf.hd * 4, STORE)],
        kt: buf(c, "dory-kt", k * kv_row * 4, STORE),
        vt: buf(c, "dory-vt", k * kv_row * 4, STORE),
        attn: buf(c, "dory-attn", k * q_size * 4, STORE),
        ob: buf(c, "dory-ob", o1_batch::Geom { g: cf.nk, hpg: cf.nh / cf.nk, m: 32, w: 0, ns: 0, rect_fm: false, d: cf.hd, dv: cf.hd, scale: 0.0 }.scratch_len() * 4, STORE),
        ofs: buf(c, "dory-ofs", cf.nk * 32 * 16, STORE),
        onm: buf(c, "dory-onm", cf.nk * 16, STORE),
        ohd: buf(c, "dory-ohd", cf.nh * 16, STORE),
        ogn: buf(c, "dory-ogn", cf.nh * 32 * 4, STORE),
    };
    let mut bytes = 0u64;
    for l in layers.iter() {
        match l {
            LayerDev::Attn(a) => {
                for x in [&a.q, &a.k, &a.v, &a.g, &a.o] {
                    bytes += x.buf.size();
                }
            }
            LayerDev::Mlp(a) => {
                for x in [&a.gate, &a.up, &a.down] {
                    bytes += x.buf.size();
                }
            }
        }
    }
    bytes += head.buf.size();
    tracing::info!(
        "dory resident graph: {:.2} GB of weights in {:.1} s",
        bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    let slots = (0..m.attn_layers).map(|_| Vec::new()).collect();
    Some(Device { layers, head, s, slots, plans: HashMap::new(), head_plan: HashMap::new(), parts: None, gate_tokens: 0, gate_one: 0, exits: ExitRate::default(), id: 0, sealed: false, arch_rows: 0, seal_epoch: 0, o1_refused: false, qtrace: HashMap::new() })
}

/// Pieces of a value sum over `n` keys: short threads on every SM beat long ones
/// on a few (one piece took 35 µs a layer at 60 keys, eight 7).
fn pieces(n: usize) -> usize {
    (n / 8).clamp(1, CH)
}

fn slot_bytes(cap: usize, kv_row: usize) -> usize {
    cap * kv_row * 4
}

impl Device {
    /// Slot `sl` of an attention layer, large enough for `need` rows.
    fn ensure_slot(&mut self, c: &Ctx, ord: usize, sl: usize, need: usize, floor: usize, kv_row: usize) -> bool {
        let row = &mut self.slots[ord];
        while row.len() <= sl {
            let cap = need.max(floor);
            row.push(Slot {
                k: buf(c, "dory-k", slot_bytes(cap, kv_row), STORE),
                v: buf(c, "dory-v", slot_bytes(cap, kv_row), STORE),
                cap,
            });
            self.plans.clear();
            self.parts = None;
        }
        if row[sl].cap >= need {
            return true;
        }
        // Double a full-attention slot: its rows are never rewritten.
        let cap = need.max(row[sl].cap * 2);
        let (k, v) = (buf(c, "dory-k", slot_bytes(cap, kv_row), STORE), buf(c, "dory-v", slot_bytes(cap, kv_row), STORE));
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dory-grow") });
        let old = slot_bytes(row[sl].cap, kv_row) as u64;
        enc.copy_buffer_to_buffer(&row[sl].k, 0, &k, 0, old);
        enc.copy_buffer_to_buffer(&row[sl].v, 0, &v, 0, old);
        c.queue.submit(Some(enc.finish()));
        row[sl] = Slot { k, v, cap };
        self.plans.clear();
        self.parts = None;
        true
    }
}

fn mv_steps(c: &'static Ctx, xs: &wgpu::Buffer, cols: usize, mats: &[(&Mat, &wgpu::Buffer)]) -> Option<Vec<Step>> {
    let mut out = Vec::new();
    let codec = mats[0].0.codec;
    if mats.iter().all(|m| m.0.codec == codec) && mats.len() <= 3 {
        let list: Vec<(&wgpu::Buffer, &wgpu::Buffer, usize)> = mats.iter().map(|m| (&m.0.buf, m.1, m.0.rows)).collect();
        let t = match codec {
            Codec::Q4 => dense_mv::sg_prep(c, xs, cols, &list).map(|(p, b, g)| (p.clone(), b, g)),
            Codec::Q8 => moe_r4::q82sg_prep(c, xs, cols, &list),
        };
        if let Some((pipe, bg, x)) = t {
            out.push(Step { pipe, bg, x, y: 1, kind: Kind::Qkv });
            return Some(out);
        }
    }
    for (m, y) in mats {
        out.push(mv_one(c, m, xs, y)?);
    }
    Some(out)
}

fn mv_one(c: &'static Ctx, m: &Mat, xs: &wgpu::Buffer, y: &wgpu::Buffer) -> Option<Step> {
    match m.codec {
        Codec::Q4 => {
            if let Some((p, bg, x)) = dense_mv::sg_prep(c, xs, m.cols, &[(&m.buf, y, m.rows)]) {
                return Some(Step { pipe: p.clone(), bg, x, y: 1, kind: Kind::Qkv });
            }
            let u = uniform_u32x4(c, [(m.cols / 32) as u32, m.rows as u32, 1, 0]);
            let pipe = &c.q4tp_mv16nl;
            let bg = bind_pairs(c, &pipe.get_bind_group_layout(0), &[(0, &m.buf), (2, y), (3, &u), (4, &m.buf), (5, xs)]);
            Some(Step { pipe: pipe.clone(), bg, x: (m.rows as u32).div_ceil(16), y: 1, kind: Kind::Qkv })
        }
        Codec::Q8 => {
            if let Some((pipe, bg, x)) = moe_r4::q82sg_prep(c, xs, m.cols, &[(&m.buf, y, m.rows)]) {
                return Some(Step { pipe, bg, x, y: 1, kind: Kind::Qkv });
            }
            let u = uniform_u32x4(c, [(m.cols / 4) as u32, m.rows as u32, m.cols as u32, 0]);
            let (pipe, per) = q82_pipe(c, m.cols);
            let bg = bind_pairs(c, &pipe.get_bind_group_layout(0), &[(0, &m.buf), (1, xs), (2, y), (3, &u)]);
            Some(Step { pipe: pipe.clone(), bg, x: (m.rows as u32).div_ceil(per), y: 1, kind: Kind::Qkv })
        }
    }
}

fn mm_one(c: &'static Ctx, m: &Mat, xs: &wgpu::Buffer, y: &wgpu::Buffer, k: usize) -> Option<Step> {
    let (pipe, bg, a, b) = match m.codec {
        Codec::Q4 => dense_batch::q4tp_mm_r_prep(c, &m.buf, xs, y, m.rows, m.cols, k)?,
        Codec::Q8 => gemma_batch::q82_mm_prep(c, &m.buf, xs, y, m.rows, m.cols, k)?,
    };
    Some(Step { pipe, bg, x: a, y: b, kind: Kind::Qkv })
}

fn tag(mut v: Vec<Step>, kind: Kind) -> Vec<Step> {
    v.iter_mut().for_each(|s| s.kind = kind);
    v
}

thread_local! {
    static PARTS_SPLIT: std::cell::Cell<[usize; 5]> = const { std::cell::Cell::new([0; 5]) };
}

fn o1_key(ordinal: usize, sl: usize) -> usize {
    ordinal * 16 + sl
}

/// The dispatches of one forward over `k` rows.
fn record(c: &'static Ctx, d: &Device, m: &Dory, loops: usize, k: usize) -> Option<Vec<Step>> {
    let p = pipes(c)?;
    let cf = &m.cfg;
    let s = &d.s;
    let mut out = Vec::with_capacity(16 * m.layers.len());
    let kx = k as u32;
    let u = |sliding: bool| if sliding { &s.u[0] } else { &s.u[1] };
    let layer = |out: &mut Vec<Step>, li: usize, lp: usize| -> Option<()> {
        match &d.layers[li] {
            LayerDev::Attn(a) => {
                let rec = li >= cf.n_in && li < cf.n_in + cf.n_rec;
                let sl = if rec && cf.kv == KvMode::PerLoop { lp } else { 0 };
                let slot = d.slots[a.ordinal].get(sl)?;
                if d.sealed && !a.sliding {
                    // A sealed full layer: the window and the far skeleton.
                    if k == 1 {
                        out.extend(mv_steps(c, &s.h, cf.hidden, &[(&a.q, &s.yq), (&a.k, &s.yk), (&a.v, &s.yv)])?);
                        out.extend(tag(mv_steps(c, &s.h, cf.hidden, &[(&a.g, &s.yg)])?, Kind::G));
                    } else {
                        out.push(mm_one(c, &a.q, &s.h, &s.yq, k)?);
                        out.push(mm_one(c, &a.k, &s.h, &s.yk, k)?);
                        out.push(mm_one(c, &a.v, &s.h, &s.yv, k)?);
                        out.extend(tag(vec![mm_one(c, &a.g, &s.h, &s.yg, k)?], Kind::G));
                    }
                    out.push(Step {
                        pipe: p.prep.p.clone(),
                        bg: bind_pairs(c, &p.prep.l, &[
                            (0, &s.u[2]), (1, &s.yq), (2, &s.yk), (3, &s.yv), (4, &s.rope[1]), (5, &a.sq),
                            (6, &s.qo), (7, &s.kt), (8, &s.vt),
                        ]),
                        x: (cf.nh + cf.nk) as u32,
                        y: kx,
                        kind: Kind::Prep,
                    });
                    if k > 1 && d.arch_rows > 0 {
                        let slot = &d.slots[a.ordinal][sl];
                        out.push(Step {
                            pipe: p.arch.p.clone(),
                            bg: bind_pairs(c, &p.arch.l, &[(0, &s.u[1]), (60, &s.kt), (61, &s.vt), (62, &slot.k), (63, &slot.v)]),
                            x: ((cf.nk * cf.hd / 4) as u32).div_ceil(32),
                            y: kx,
                            kind: Kind::Prep,
                        });
                        if let Some(tr) = d.qtrace.get(&(a.ordinal, sl)) {
                            out.push(Step {
                                pipe: p.qcopy.p.clone(),
                                bg: bind_pairs(c, &p.qcopy.l, &[(0, &s.u[1]), (34, &s.qo), (35, tr)]),
                                x: ((cf.nh * cf.hd / 4) as u32).div_ceil(32),
                                y: kx,
                                kind: Kind::Prep,
                            });
                        }
                    }
                    let map = c.o1m.lock().unwrap();
                    let o = map.get(&(d.id, o1_key(a.ordinal, sl)))?;
                    let uo = uniform_u32x8(c, [o.h as u32, o.m as u32, o.w as u32, o.ns as u32, cf.hd as u32, cf.hd as u32, o.scale.to_bits(), 0]);
                    if k > 1 {
                        // A prompt chunk: `o1_batch`, the state moving only with the
                        // chunk's last pass on the last-loop cache.
                        let ob = o1_batch::pipes(c)?;
                        let st = o1_batch::State {
                            meta: &o.meta,
                            ring_k: &o.ring_k,
                            ring_v: &o.ring_v,
                            sink_k: &o.sink_k,
                            sink_v: &o.sink_v,
                            k_tilde: &o.k_tilde,
                            qt: &o.qt,
                            mu: &o.mu,
                            mz: &o.mz,
                            that: &o.that,
                        };
                        let geom = o1_batch::Geom { g: o.g, hpg: o.h, m: o.m, w: o.w, ns: o.ns, rect_fm: false, d: cf.hd, dv: cf.hd, scale: o.scale };
                        let commit = !(rec && cf.kv == KvMode::LastLoop && lp + 1 < loops);
                        for (pipe, bg, x, y) in o1_batch::steps(c, ob, &st, geom, &s.qo, &s.kt, &s.vt, &s.ob, &s.attn, k, commit) {
                            out.push(Step { pipe, bg, x, y, kind: Kind::FarAttend });
                        }
                    } else if lp > 0 && cf.kv == KvMode::LastLoop {
                        out.push(Step {
                            pipe: p.o1_back.p.clone(),
                            bg: bind_pairs(c, &p.o1_back.l, &[(33, &o.meta)]),
                            x: o.g as u32,
                            y: 1,
                            kind: Kind::Far,
                        });
                    } else {
                        out.push(Step {
                            pipe: c.o1_far.clone(),
                            bg: bind_pairs(c, &c.layout_o1_far, &[(0, &o.meta), (1, &o.ring_k), (2, &o.ring_v), (3, &o.qt), (4, &o.mz), (5, &o.that), (6, &uo)]),
                            x: (o.g * o.h * o.m) as u32,
                            y: 1,
                            kind: Kind::Far,
                        });
                    }
                    if k == 1 {
                    out.push(Step {
                        pipe: c.o1_push.clone(),
                        bg: bind_pairs(c, &c.layout_o1_push, &[(0, &o.meta), (1, &s.kt), (2, &s.vt), (3, &o.ring_k), (4, &o.ring_v), (5, &uo)]),
                        x: o.g as u32,
                        y: 1,
                        kind: Kind::Push,
                    });
                    let f = &p.far;
                    let nk = cf.nk as u32;
                    let u1 = &s.u[1];
                    let mut far = |pipe: &Pipe, pairs: &[(u32, &wgpu::Buffer)], x: u32, y: u32, kind: Kind| {
                        out.push(Step { pipe: pipe.p.clone(), bg: bind_pairs(c, &pipe.l, pairs), x, y, kind });
                    };
                    far(&f.os, &[(0, u1), (50, &uo), (40, &o.meta), (10, &s.qo), (41, &o.ring_k), (42, &o.sink_k), (15, &s.scores)], ((o.w + o.ns) as u32).div_ceil(32), nk, Kind::FarAttend);
                    far(&f.ol, &[(0, u1), (50, &uo), (10, &s.qo), (43, &o.k_tilde), (51, &s.ofs)], (o.m as u32).div_ceil(32), nk, Kind::FarAttend);
                    far(&f.omax, &[(0, u1), (50, &uo), (40, &o.meta), (17, &s.scores), (53, &s.onm)], nk, 1, Kind::FarAttend);
                    far(&f.ofar, &[(0, u1), (50, &uo), (40, &o.meta), (52, &s.ofs), (54, &s.onm), (44, &o.mu), (45, &o.mz), (55, &s.ohd), (57, &s.ogn)], (cf.nh as u32).div_ceil(8), 1, Kind::FarAttend);
                    far(&f.osoft, &[(0, u1), (50, &uo), (40, &o.meta), (17, &s.scores), (56, &s.ohd), (18, &s.sinv)], nk, 1, Kind::FarAttend);
                    far(&f.osum, &[(0, u1), (50, &uo), (40, &o.meta), (19, &s.scores), (47, &o.ring_v), (48, &o.sink_v), (16, &s.pz)], ((cf.nk * CH * cf.hd / 4) as u32).div_ceil(32), 1, Kind::FarAttend);
                    far(&f.ogate, &[(0, u1), (50, &uo), (20, &s.pz), (56, &s.ohd), (58, &s.ogn), (46, &o.that), (9, &s.sinv), (13, &s.yg), (14, &a.sg), (31, &s.z)], ((cf.nh * cf.hd / 4) as u32).div_ceil(32), 1, Kind::Gate);
                    } else {
                        // A chunk's rows come out of `o1_batch` without the gate.
                        out.push(Step {
                            pipe: p.gate1.p.clone(),
                            bg: bind_pairs(c, &p.gate1.l, &[(0, &s.u[0]), (13, &s.yg), (14, &a.sg), (32, &s.attn), (31, &s.z)]),
                            x: ((cf.nh * cf.hd / 4) as u32).div_ceil(32),
                            y: kx,
                            kind: Kind::Gate,
                        });
                    }
                    drop(map);
                    out.extend(tag(vec![if k == 1 { mv_one(c, &a.o, &s.z, &s.o)? } else { mm_one(c, &a.o, &s.z, &s.o, k)? }], Kind::O));
                    out.push(Step {
                        pipe: p.slerp.p.clone(),
                        bg: bind_pairs(c, &p.slerp.l, &[(0, &s.u[0]), (25, &s.h), (26, &s.o), (27, &a.alpha)]),
                        x: kx,
                        y: 1,
                        kind: Kind::Slerp,
                    });
                    return Some(());
                }
                if k == 1 {
                    out.extend(mv_steps(c, &s.h, cf.hidden, &[(&a.q, &s.yq), (&a.k, &s.yk), (&a.v, &s.yv)])?);
                    out.extend(tag(mv_steps(c, &s.h, cf.hidden, &[(&a.g, &s.yg)])?, Kind::G));
                } else {
                    out.push(mm_one(c, &a.q, &s.h, &s.yq, k)?);
                    out.push(mm_one(c, &a.k, &s.h, &s.yk, k)?);
                    out.push(mm_one(c, &a.v, &s.h, &s.yv, k)?);
                    out.extend(tag(vec![mm_one(c, &a.g, &s.h, &s.yg, k)?], Kind::G));
                }
                let r = if a.sliding { &s.rope[0] } else { &s.rope[1] };
                out.push(Step {
                    pipe: p.prep.p.clone(),
                    bg: bind_pairs(c, &p.prep.l, &[
                        (0, u(a.sliding)), (1, &s.yq), (2, &s.yk), (3, &s.yv), (4, r), (5, &a.sq),
                        (6, &s.qo), (7, &slot.k), (8, &slot.v),
                    ]),
                    x: (cf.nh + cf.nk) as u32,
                    y: kx,
                    kind: Kind::Prep,
                });
                if let Some(tr) = d.qtrace.get(&(a.ordinal, sl)) {
                    out.push(Step {
                        pipe: p.qcopy.p.clone(),
                        bg: bind_pairs(c, &p.qcopy.l, &[(0, &s.u[1]), (34, &s.qo), (35, tr)]),
                        x: ((cf.nh * cf.hd / 4) as u32).div_ceil(32),
                        y: kx,
                        kind: Kind::Prep,
                    });
                }
                out.push(Step {
                    pipe: p.score.p.clone(),
                    bg: bind_pairs(c, &p.score.l, &[(0, u(a.sliding)), (10, &s.qo), (11, &slot.k), (15, &s.scores)]),
                    // 0: a sliding layer; else the workgroups come from the position.
                    x: if a.sliding { 0 } else { 1 },
                    y: cf.nk as u32,
                    kind: Kind::Score,
                });
                out.push(Step {
                    pipe: p.soft.p.clone(),
                    bg: bind_pairs(c, &p.soft.l, &[(0, u(a.sliding)), (17, &s.scores), (18, &s.sinv)]),
                    x: cf.nk as u32,
                    y: kx,
                    kind: Kind::Soft,
                });
                out.push(Step {
                    pipe: p.sum.p.clone(),
                    bg: bind_pairs(c, &p.sum.l, &[(0, u(a.sliding)), (12, &slot.v), (19, &s.scores), (16, &s.pz)]),
                    x: if a.sliding { 0 } else { 1 },
                    y: kx,
                    kind: Kind::Sum,
                });
                out.push(Step {
                    pipe: p.gate.p.clone(),
                    bg: bind_pairs(c, &p.gate.l, &[(0, u(a.sliding)), (13, &s.yg), (14, &a.sg), (20, &s.pz), (9, &s.sinv), (31, &s.z)]),
                    x: ((cf.nh * cf.hd / 4) as u32).div_ceil(32),
                    y: kx,
                    kind: Kind::Gate,
                });
                out.extend(tag(vec![if k == 1 { mv_one(c, &a.o, &s.z, &s.o)? } else { mm_one(c, &a.o, &s.z, &s.o, k)? }], Kind::O));
                out.push(Step {
                    pipe: p.slerp.p.clone(),
                    bg: bind_pairs(c, &p.slerp.l, &[(0, &s.u[0]), (25, &s.h), (26, &s.o), (27, &a.alpha)]),
                    x: kx,
                    y: 1,
                    kind: Kind::Slerp,
                });
            }
            LayerDev::Mlp(a) => {
                if k == 1 {
                    out.extend(tag(mv_steps(c, &s.h, cf.hidden, &[(&a.gate, &s.fg), (&a.up, &s.fu)])?, Kind::Gu));
                } else {
                    out.extend(tag(vec![mm_one(c, &a.gate, &s.h, &s.fg, k)?, mm_one(c, &a.up, &s.h, &s.fu, k)?], Kind::Gu));
                }
                out.push(Step {
                    pipe: p.silu.p.clone(),
                    bg: bind_pairs(c, &p.silu.l, &[(0, &s.u[0]), (21, &s.fg), (22, &s.fu), (23, &a.sg), (24, &a.su)]),
                    x: ((cf.ffn / 4) as u32).div_ceil(32),
                    y: kx,
                    kind: Kind::Silu,
                });
                out.extend(tag(vec![if k == 1 { mv_one(c, &a.down, &s.fg, &s.o)? } else { mm_one(c, &a.down, &s.fg, &s.o, k)? }], Kind::D));
                out.push(Step {
                    pipe: p.slerp.p.clone(),
                    bg: bind_pairs(c, &p.slerp.l, &[(0, &s.u[0]), (25, &s.h), (26, &s.o), (27, &a.alpha)]),
                    x: kx,
                    y: 1,
                    kind: Kind::Slerp,
                });
            }
        }
        Some(())
    };
    let mix = |out: &mut Vec<Step>, mode: usize| {
        let pp = &p.mix[mode];
        let pairs: Vec<(u32, &wgpu::Buffer)> = match mode {
            0 | 1 | 4 => vec![(0, &s.u[0]), (28, &s.h), (29, &s.input), (30, &s.state)],
            3 => vec![(0, &s.u[0]), (28, &s.h), (30, &s.state)],
            _ => vec![(0, &s.u[0]), (28, &s.h)],
        };
        out.push(Step { pipe: pp.p.clone(), bg: bind_pairs(c, &pp.l, &pairs), x: kx, y: 1, kind: Kind::Mix });
    };
    if loops == 0 {
        // The parts of a token for the early exit (`Parts`).
        let mut lists: [Vec<Step>; 6] = Default::default();
        for li in 0..cf.n_in {
            layer(&mut lists[0], li, 0)?;
        }
        mix(&mut lists[0], 0);
        for li in cf.n_in..cf.n_in + cf.n_rec {
            layer(&mut lists[1], li, 0)?;
            layer(&mut lists[2], li, 1)?;
        }
        for li in cf.n_in + cf.n_rec..m.layers.len() {
            layer(&mut lists[3], li, 0)?;
            layer(&mut lists[4], li, 1)?;
        }
        for i in 0..5 {
            mix(&mut lists[5], i);
        }
        PARTS_SPLIT.with(|c| c.set([lists[0].len(), lists[1].len(), lists[2].len(), lists[3].len(), lists[4].len()]));
        return Some(lists.into_iter().flatten().collect());
    }
    for li in 0..cf.n_in {
        layer(&mut out, li, 0)?;
    }
    mix(&mut out, 0);
    for i in 0..loops {
        for li in cf.n_in..cf.n_in + cf.n_rec {
            layer(&mut out, li, i)?;
        }
        mix(&mut out, if i + 1 == loops { 2 } else { 1 });
    }
    for li in cf.n_in + cf.n_rec..m.layers.len() {
        layer(&mut out, li, 0)?;
    }
    Some(out)
}

/// `Some` when this forward ran on the device.
pub(crate) fn forward(
    id: u64,
    m: &Dory,
    ids: &[u32],
    pos0: usize,
    loops: usize,
    want: Want,
    logits: &mut Vec<f32>,
) -> bool {
    let Some(c) = ctx() else { return false };
    if !admits(m) || !selected() {
        return false;
    }
    // An early exit leaves a pass's rows only on the last-loop cache.
    if m.gate.is_some() && cf_kv(m) != KvMode::LastLoop {
        tracing::warn!("CMF_DORY_GATE needs --kv-cache last_loop: not applied");
    }
    let mut guard = DEVICES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if !map.contains_key(&id) {
        match build(c, m) {
            Some(mut d) => {
                d.id = id;
                map.insert(id, d);
            }
            None => return false,
        }
    }
    let d = map.get_mut(&id).expect("just inserted");
    let cf = &m.cfg;
    let kv_row = cf.nk * cf.hd;
    let mut at = 0usize;
    while at < ids.len() {
        // Decode rows one at a time reads each weight once per row; a whole
        // chunk reads it once per 64 rows.
        let k = (ids.len() - at).min(CHUNK);
        // A sealed prompt chunk needs a window the batch kernels can evict from.
        let batch = !d.sealed
            || (o1_batch::pipes(c).is_some()
                && m.o1.is_some_and(|o| {
                    let extra = if d.arch_rows > 0 { 0 } else { m.exact.as_ref().map_or(0, |x| x.budget) };
                    o.w >= o1_batch::OB_K && o.sink + extra <= 255 && o.w + o.sink + extra <= 2052
                }));
        let k = if k < 8 || !batch { 1 } else { k };
        // The prompt's last rows, rows that take the decode path, or rows that would pass
        // what is kept: the choice comes now.
        let tail_from = (pos0 + ids.len()).saturating_sub(crate::dory::EXACT_TAIL);
        if d.arch_rows > 0 && (k == 1 || pos0 + at + k > ARCH_MAX || pos0 + at >= tail_from) {
            if !m.o1.is_some_and(|o1| seal(c, d, m, o1, pos0 + at, true)) {
                return false;
            }
            continue;
        }
        let last = at + k == ids.len();
        let w = if last || want == Want::All { want } else { Want::None };
        if !run(c, d, m, &ids[at..at + k], pos0 + at, m.chunk_loops(loops, w), w, kv_row, logits) {
            return false;
        }
        at += k;
        if d.arch_rows > 0 && k > 1 {
            d.arch_rows = pos0 + at;
        }
        if let (Some(o1), false, false) = (m.o1, d.sealed, d.o1_refused) {
            let rows = pos0 + at;
            // At a prompt's end: its rows go fastest exact, and a long one seals
            // anyway before they outgrow the card. A refused state only costs the
            // layer its far field: it stays exact.
            let due = at == ids.len() || rows >= SEAL_MAX;
            if due && rows >= o1.seal && !seal(c, d, m, o1, rows, at == ids.len()) {
                d.o1_refused = true;
                d.qtrace.clear();
                d.plans.clear();
                d.parts = None;
            }
        }
    }
    if d.arch_rows > 0 && !m.o1.is_some_and(|o1| seal(c, d, m, o1, pos0 + ids.len(), true)) {
        return false;
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn run(
    c: &'static Ctx,
    d: &mut Device,
    m: &Dory,
    ids: &[u32],
    pos: usize,
    loops: usize,
    want: Want,
    kv_row: usize,
    logits: &mut Vec<f32>,
) -> bool {
    let cf = &m.cfg;
    let k = ids.len();
    // The KV rings. A sliding ring holds the window, a chunk and a turn's rewind.
    let sl_cap = cf.window + CHUNK + crate::dory::REWIND;
    let need_full = pos + k;
    let per_loop = cf.kv == KvMode::PerLoop;
    let layers = d.layers.clone();
    let attn: Vec<(usize, usize, bool)> = layers
        .iter()
        .enumerate()
        .filter_map(|(li, l)| match l {
            LayerDev::Attn(a) => Some((li, a.ordinal, a.sliding)),
            _ => None,
        })
        .collect();
    for (li, ord, sliding) in attn {
        let rec = li >= cf.n_in && li < cf.n_in + cf.n_rec;
        let n = if rec && per_loop { loops } else { 1 };
        for sl in 0..n {
            if !sliding && d.sealed && !(d.arch_rows > 0 && k > 1) {
                continue;
            }
            if let (false, Some(o1), false, false, false) = (sliding, m.o1, d.sealed, d.o1_refused, d.qtrace.contains_key(&(ord, sl))) {
                let top = if m.exact.is_some() { ARCH_MAX } else { SEAL_MAX };
                let b = buf(c, "dory-qtrace", (top + CHUNK) / QT_STRIDE * cf.nh * cf.hd * 4, STORE);
                d.qtrace.insert((ord, sl), b);
                d.plans.clear();
                d.parts = None;
            }
            // A sliding ring has its size; a full slot grows from 4096 rows, or is a
            // ring of its own when the full layers are cut to a window.
            if sliding {
                d.ensure_slot(c, ord, sl, sl_cap, sl_cap, kv_row);
            } else if let Some(w) = cf.full_window {
                let cap = w + CHUNK + crate::dory::REWIND;
                d.ensure_slot(c, ord, sl, cap, cap, kv_row);
            } else {
                d.ensure_slot(c, ord, sl, need_full, 4096, kv_row);
            }
        }
    }
    // The score plane.
    let ncap_f = pos + k;
    if ncap_f > d.s.scores_cap && !d.sealed {
        let cap = ncap_f.max(d.s.scores_cap * 2);
        d.s.scores = buf(c, "dory-scores", cap * CHUNK * cf.nk * 16, STORE);
        d.s.scores_cap = cap;
        d.plans.clear();
        d.parts = None;
    }
    // The uniforms of the two layer kinds, and the rotary tables.
    let full_cap = d
        .layers
        .iter()
        .find_map(|l| match l {
            LayerDev::Attn(a) if !a.sliding => Some(d.slots[a.ordinal][0].cap),
            _ => None,
        })
        .unwrap_or(need_full);
    let full_w = cf.full_window.unwrap_or(u32::MAX as usize);
    let words = |cap: usize, window: u32, ncap: usize, n: usize| -> [u32; 16] {
        [
            k as u32,
            pos as u32,
            cap as u32,
            window,
            cf.nh as u32,
            cf.nk as u32,
            cf.hd as u32,
            cf.hidden as u32,
            cf.ffn as u32,
            (cf.nh * cf.hd) as u32,
            ncap as u32,
            REP as u32,
            cf.attn_scale.to_bits(),
            crate::dory::EPS.to_bits(),
            pieces(n) as u32,
            0,
        ]
    };
    c.queue.write_buffer(&d.s.u[0], 0, bytemuck::cast_slice(&words(sl_cap, cf.window as u32, d.s.scores_cap, (pos + k).min(cf.window))));
    c.queue.write_buffer(&d.s.u[1], 0, bytemuck::cast_slice(&words(full_cap, full_w as u32, d.s.scores_cap, (pos + k).min(full_w))));
    let mut w2 = words(k, u32::MAX, d.s.scores_cap, 1);
    w2[1] = 0; // Its one-chunk ring starts at this chunk.
    c.queue.write_buffer(&d.s.u[2], 0, bytemuck::cast_slice(&w2));
    let half = cf.hd / 2;
    let mut table = vec![0f32; k * cf.hd];
    for (ti, inv) in [&m.inv_sliding, &m.inv_full].iter().enumerate() {
        for r in 0..k {
            let at = (pos + r) as f64;
            let row = &mut table[r * cf.hd..(r + 1) * cf.hd];
            for j in 0..half {
                let (sn, cs) = (at * inv[j] as f64).sin_cos();
                row[j] = cs as f32;
                row[half + j] = sn as f32;
            }
        }
        c.queue.write_buffer(&d.s.rope[ti], 0, bytemuck::cast_slice(&table));
    }
    // The embedded rows.
    let mut rows = vec![0f32; k * cf.hidden];
    for (i, row) in rows.chunks_exact_mut(cf.hidden).enumerate() {
        if (ids[i] as usize) < m.embed.rows() {
            m.embed.row_f32(ids[i] as usize, row);
        }
    }
    c.queue.write_buffer(&d.s.h, 0, bytemuck::cast_slice(&rows));
    // One row and a gate: the parts of the token, one by one.
    if k == 1 && want != Want::None {
        if let Some(g) = &m.gate {
            return run_gated(c, d, m, want, g, pos, logits);
        }
    }
    // The plan.
    let plan = match d.plans.get(&(loops, k)) {
        Some(p) => p.clone(),
        None => {
            let Some(p) = record(c, d, m, loops, k) else { return false };
            let p = std::sync::Arc::new(p);
            d.plans.insert((loops, k), p.clone());
            p
        }
    };
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dory-token") });
    let _merge = PassMergeGuard::new(&enc);
    {
        let mut pass = begin_pass(&mut enc);
        replay(&mut pass, &plan, cf, pos, k);
    }
    match want {
        Want::None => {
            flush_pass(&enc);
            submit(c, finish_enc(enc));
            let _ = c.device.poll(wgpu::PollType::Poll);
            true
        }
        Want::Last => {
            let st = match d.head_plan.get(&1) {
                Some(s) => s.clone(),
                None => {
                    let Some(s) = mv_one(c, &d.head, &d.s.hlast, &d.s.logits) else { return false };
                    let s = std::sync::Arc::new(s);
                    d.head_plan.insert(1, s.clone());
                    s
                }
            };
            flush_pass(&enc);
            let off = ((k - 1) * cf.hidden * 4) as u64;
            enc.copy_buffer_to_buffer(&d.s.h, off, &d.s.hlast, 0, (cf.hidden * 4) as u64);
            {
                let mut pass = begin_pass(&mut enc);
                pass.set_pipeline(&st.pipe);
                pass.set_bind_group(0, &st.bg, &[]);
                pass.dispatch_workgroups(st.x, st.y, 1);
            }
            logits.clear();
            logits.resize(cf.vocab, 0.0);
            let size = (cf.vocab * 4) as u64;
            if !readback(c, enc, &d.s.logits, &d.s.staging, size, logits) {
                return false;
            }
            for (x, s) in logits.iter_mut().zip(&m.logit_scale) {
                *x *= *s;
            }
            true
        }
        Want::All => {
            let Some(st) = (if k == 1 {
                mv_one(c, &d.head, &d.s.h, &d.s.logits_k)
            } else {
                mm_one(c, &d.head, &d.s.h, &d.s.logits_k, k)
            }) else {
                return false;
            };
            {
                let mut pass = begin_pass(&mut enc);
                pass.set_pipeline(&st.pipe);
                pass.set_bind_group(0, &st.bg, &[]);
                pass.dispatch_workgroups(st.x, st.y, 1);
            }
            let from = logits.len();
            logits.resize(from + k * cf.vocab, 0.0);
            let size = (k * cf.vocab * 4) as u64;
            if !readback(c, enc, &d.s.logits_k, &d.s.staging_k, size, &mut logits[from..]) {
                return false;
            }
            for row in logits[from..].chunks_exact_mut(cf.vocab) {
                for (x, s) in row.iter_mut().zip(&m.logit_scale) {
                    *x *= *s;
                }
            }
            true
        }
    }
}

fn replay(pass: &mut PassHandle, steps: &[Step], cf: &crate::dory::DoryCfg, pos: usize, k: usize) {
    let keep = keep_mask();
    for st in steps {
        if keep & (1 << st.kind as u32) == 0 {
            continue;
        }
        pass.set_pipeline(&st.pipe);
        pass.set_bind_group(0, &st.bg, &[]);
        if st.kind == Kind::Score || st.kind == Kind::Sum {
            // As many keys as the longest row has: the plan outlives a position.
            let sliding = st.x == 0;
            let n = if sliding { (pos + k).min(cf.window) } else { (pos + k).min(cf.full_window.unwrap_or(usize::MAX)) };
            if st.kind == Kind::Score {
                pass.dispatch_workgroups((n as u32).div_ceil(32), st.y, k as u32);
            } else {
                let threads = (cf.nk * pieces(n) * cf.hd / 4) as u32;
                pass.dispatch_workgroups(threads.div_ceil(32), st.y, 1);
            }
        } else {
            pass.dispatch_workgroups(st.x, st.y, 1);
        }
    }
}

/// A token with an early exit on the last-loop cache: every pass writes its rows
/// over the same ring positions, so a token that stops after one pass leaves no hole.
fn run_gated(c: &'static Ctx, d: &mut Device, m: &Dory, want: Want, g: &Gate, pos: usize, logits: &mut Vec<f32>) -> bool {
    let cf = &m.cfg;
    if d.parts.is_none() {
        let Some(flat) = record(c, d, m, 0, 1) else { return false };
        let [a, b, b2, o, o2] = PARTS_SPLIT.with(|c| c.get());
        let cut = |from: usize, n: usize| flat[from..from + n].to_vec();
        d.parts = Some(std::sync::Arc::new(Parts {
            input: cut(0, a),
            pass: cut(a, b),
            pass2: cut(a + b, b2),
            out: cut(a + b + b2, o),
            out2: cut(a + b + b2 + o, o2),
            mix: flat[a + b + b2 + o + o2..].to_vec(),
        }));
    }
    let parts = d.parts.as_ref().expect("just built").clone();
    let head = match d.head_plan.get(&0) {
        Some(s) => s.clone(),
        None => {
            let Some(s) = mv_one(c, &d.head, &d.s.h, &d.s.logits) else { return false };
            let s = std::sync::Arc::new(s);
            d.head_plan.insert(0, s.clone());
            s
        }
    };
    let head: &Step = &head;
    let size = (cf.vocab * 4) as u64;
    let mut read = |c: &'static Ctx, enc: wgpu::CommandEncoder, d: &Device, logits: &mut Vec<f32>| -> bool {
        logits.clear();
        logits.resize(cf.vocab, 0.0);
        if !readback(c, enc, &d.s.logits, &d.s.staging, size, logits) {
            return false;
        }
        for (x, s) in logits.iter_mut().zip(&m.logit_scale) {
            *x *= *s;
        }
        true
    };
    let encoder = |c: &'static Ctx| c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dory-gate") });
    let mut lg = Vec::new();
    let one = match g {
        Gate::Conf(th) => {
            let probe = d.exits.probe();
            let mut enc = encoder(c);
            {
                let _merge = PassMergeGuard::new(&enc);
                let mut pass = begin_pass(&mut enc);
                replay(&mut pass, &parts.input, cf, pos, 1);
                replay(&mut pass, &parts.pass, cf, pos, 1);
                replay(&mut pass, &parts.mix[3..4], cf, pos, 1);
                if probe {
                    replay(&mut pass, &parts.out, cf, pos, 1);
                    replay(&mut pass, std::slice::from_ref(head), cf, pos, 1);
                }
                drop(pass);
                flush_pass(&enc);
            }
            let mut one = false;
            if probe {
                if !read(c, enc, d, &mut lg) {
                    return false;
                }
                one = top_prob(&lg) >= *th;
                d.exits.note(one);
            } else {
                submit(c, finish_enc(enc));
            }
            if !one {
                let mut enc = encoder(c);
                {
                    let _merge = PassMergeGuard::new(&enc);
                    let mut pass = begin_pass(&mut enc);
                    replay(&mut pass, &parts.mix[4..5], cf, pos, 1);
                    replay(&mut pass, &parts.pass2, cf, pos, 1);
                    replay(&mut pass, &parts.mix[2..3], cf, pos, 1);
                    // A second read-out rewrites this position in the output block.
                    replay(&mut pass, if probe { &parts.out2 } else { &parts.out }, cf, pos, 1);
                    replay(&mut pass, std::slice::from_ref(head), cf, pos, 1);
                    drop(pass);
                    flush_pass(&enc);
                }
                if !read(c, enc, d, &mut lg) {
                    return false;
                }
            }
            one
        }
        Gate::Recon(t, bias) => {
            let mut enc = encoder(c);
            {
                let _merge = PassMergeGuard::new(&enc);
                let mut pass = begin_pass(&mut enc);
                replay(&mut pass, &parts.input, cf, pos, 1);
                replay(&mut pass, &parts.pass, cf, pos, 1);
                replay(&mut pass, &parts.mix[3..4], cf, pos, 1);
                drop(pass);
                flush_pass(&enc);
            }
            let hb = (cf.hidden * 4) as u64;
            let mut phi = vec![0f32; cf.hidden];
            if !readback(c, enc, &d.s.state, &d.s.staging, hb, &mut phi) {
                return false;
            }
            let easy = t.score(&phi) >= *bias;
            let mut enc = encoder(c);
            {
                let _merge = PassMergeGuard::new(&enc);
                let mut pass = begin_pass(&mut enc);
                if !easy {
                    replay(&mut pass, &parts.mix[4..5], cf, pos, 1);
                    replay(&mut pass, &parts.pass2, cf, pos, 1);
                    replay(&mut pass, &parts.mix[2..3], cf, pos, 1);
                }
                replay(&mut pass, &parts.out, cf, pos, 1);
                replay(&mut pass, std::slice::from_ref(head), cf, pos, 1);
                drop(pass);
                flush_pass(&enc);
            }
            if !read(c, enc, d, &mut lg) {
                return false;
            }
            easy
        }
    };
    d.gate_tokens += 1;
    d.gate_one += u64::from(one);
    if d.gate_tokens % 64 == 0 && std::env::var_os("CMF_DORY_GATE_LOG").is_some() {
        eprintln!("dory gate: {} tokens, {:.1} % on one pass", d.gate_tokens, 100.0 * d.gate_one as f64 / d.gate_tokens as f64);
    }
    match want {
        Want::All => logits.extend_from_slice(&lg),
        _ => *logits = lg,
    }
    true
}

fn cf_kv(m: &Dory) -> KvMode {
    m.cfg.kv
}

/// O(1) on the full layers from here on: read the slots back, freeze landmarks over
/// them (`nystrom`), upload the states, free the exact rows.
fn seal(c: &'static Ctx, d: &mut Device, m: &Dory, o1: crate::dory::DoryO1, rows: usize, last: bool) -> bool {
    let cf = &m.cfg;
    let (nh, nk, hd) = (cf.nh, cf.nk, cf.hd);
    let rep = nh / nk;
    let kv_row = nk * hd;
    let t0 = std::time::Instant::now();
    let full: Vec<usize> = d
        .layers
        .iter()
        .filter_map(|l| match l {
            LayerDev::Attn(a) if !a.sliding => Some(a.ordinal),
            _ => None,
        })
        .collect();
    let mut got: Vec<(usize, usize, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
    for &o in &full {
        for sl in 0..d.slots[o].len() {
            let slot = &d.slots[o][sl];
            let Some(tr) = d.qtrace.get(&(o, sl)) else { return false };
            let (Some(ks), Some(vs), Some(qs)) = (
                train_ops::download(c, &slot.k, rows * kv_row),
                train_ops::download(c, &slot.v, rows * kv_row),
                train_ops::download(c, tr, rows.div_ceil(QT_STRIDE) * nh * hd),
            ) else {
                return false;
            };
            got.push((o, sl, ks, vs, qs));
        }
    }
    // The chosen rows leave the far field: a copy of a fact there drowns the exact one.
    let keep = match (&m.exact, last) {
        (Some(x), true) => {
            let views: Vec<crate::dory::FlatRows> = got.iter().filter(|e| e.1 == 0).map(|e| crate::dory::FlatRows { k: &e.2, v: &e.3, nk, hd }).collect();
            let layers: Vec<&dyn crate::dory::LayerRows> = views.iter().map(|v| v as &dyn crate::dory::LayerRows).collect();
            crate::dory::pick_exact_rows(x, &layers, rows, nk, hd, o1.m, o1.sink, rows.saturating_sub(o1.w))
        }
        _ => Vec::new(),
    };
    let order = crate::dory::seal_order(rows, o1.sink, &keep);
    d.seal_epoch += 1;
    for (o, sl, ks, vs, qs) in &got {
        let mut states = Vec::with_capacity(nk);
        std::thread::scope(|sc| {
            let mut hs = Vec::with_capacity(nk);
            for kh in 0..nk {
                let (ks, vs, qs, order, keep) = (ks, vs, qs, &order, &keep);
                hs.push(sc.spawn(move || {
                    let mut kg = vec![0f32; rows * hd];
                    let mut vg = vec![0f32; rows * hd];
                    let mut qg: Vec<Vec<f32>> = (0..rep).map(|_| vec![0f32; rows * hd]).collect();
                    for (r, &from) in order.iter().enumerate() {
                        let from = from as usize;
                        let at = (from * nk + kh) * hd;
                        kg[r * hd..(r + 1) * hd].copy_from_slice(&ks[at..at + hd]);
                        vg[r * hd..(r + 1) * hd].copy_from_slice(&vs[at..at + hd]);
                        for (h, q) in qg.iter_mut().enumerate() {
                            let at = (from / QT_STRIDE * nh + kh * rep + h) * hd;
                            q[r * hd..(r + 1) * hd].copy_from_slice(&qs[at..at + hd]);
                        }
                    }
                    let mut n = crate::nystrom::NystromState::new_group(o1.m, o1.w, o1.sink + keep.len(), rep);
                    let refs: Vec<&[f32]> = qg.iter().map(|q| &q[..]).collect();
                    n.prefill_group(&refs, &kg, &vg, rows, hd, hd);
                    n
                }));
            }
            for h in hs {
                states.push(h.join().expect("seal worker"));
            }
        });
        let views: Vec<_> = states.iter().map(|n| n.device_view()).collect();
        if !o1_views_valid_near(&views, nh, nk, hd, crate::dory::MAX_NEAR)
            || o1_ensure_near(c, d.id, o1_key(*o, *sl), &views, d.seal_epoch, crate::dory::MAX_NEAR).is_none()
        {
            tracing::warn!("dory: the far state does not fit the kernels, the layer stays exact");
            return false;
        }
    }
    if m.exact.is_some() && !last {
        d.arch_rows = rows;
    } else {
        for (o, sl, _, _, _) in &got {
            d.slots[*o][*sl].k = buf(c, "dory-sealed", 16, STORE);
            d.slots[*o][*sl].v = buf(c, "dory-sealed", 16, STORE);
            d.slots[*o][*sl].cap = 1;
        }
        d.arch_rows = 0;
        d.qtrace.clear();
    }
    d.sealed = true;
    d.plans.clear();
    d.parts = None;
    tracing::info!("dory: O(1) sealed {rows} rows, {} kept exact, in {:.2} s", keep.len(), t0.elapsed().as_secs_f64());
    true
}

/// Has this sequence gone over to the far field?
pub(crate) fn sealed(id: u64) -> bool {
    DEVICES.lock().unwrap().as_ref().and_then(|g| g.get(&id)).is_some_and(|d| d.sealed)
}

/// The KV bytes of a sequence on the device.
pub(crate) fn kv_bytes(id: u64) -> Option<u64> {
    let g = DEVICES.lock().unwrap();
    let d = g.as_ref()?.get(&id)?;
    let exact: u64 = d.slots.iter().flatten().map(|s| s.k.size() + s.v.size()).sum();
    let far: u64 = ctx().map_or(0, |c| c.o1m.lock().unwrap().iter().filter(|(k, _)| k.0 == id).map(|(_, o)| o.bytes).sum());
    Some(exact + far)
}

/// A new sequence: O(1) states go, and exact slots come back on demand.
pub(crate) fn reset(id: u64) {
    if let Some(map) = DEVICES.lock().unwrap().as_mut() {
        if let Some(d) = map.get_mut(&id) {
            if d.sealed {
                if let Some(c) = ctx() {
                    c.o1m.lock().unwrap().retain(|(i, _), _| *i != id);
                }
                d.sealed = false;
                d.arch_rows = 0;
                d.qtrace.clear();
                d.o1_refused = false;
                d.slots.iter_mut().for_each(|r| r.clear());
                d.plans.clear();
                d.parts = None;
            }
        }
    }
}

/// Free the model's device state.
pub(crate) fn drop_model(id: u64) {
    if let Some(map) = DEVICES.lock().unwrap().as_mut() {
        map.remove(&id);
    }
}
