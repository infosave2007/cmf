//! Inkling-2-Mini on wgpu (Vulkan, DX12, Metal through wgpu), the twin of the
//! CUDA arm (`gpu_cuda/hyb/inkling.rs`; `inkling` has the semantics).
//! Resident: every matrix in VRAM — the dense ones as the backend's q4tp /
//! q8_2f matvecs read them, a layer's routed experts in one bank buffer that
//! this arm's kernels address by expert (the router, the top-k and every
//! pick's gate/up and down run on the device: nothing returns to the host
//! inside a call). Every one of the 48 layer applications keeps its own keys
//! and values (f32; a local layer's in a ring of its window and a call's rows,
//! a full layer's growing) and its four convolution rings. A call (a token, or
//! up to 16 prompt rows) replays a plan recorded once per row count.

use super::*;
use crate::inkling::{Cfg, Globals, Layer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Rows of a call (`CMF_INKLING_ROWS`, 128): a prompt chunk's; a decode call
/// is one.
fn rows_cap() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("CMF_INKLING_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).map_or(128, |n| n.clamp(1, 1024)))
}
/// Rows of logits a call returns (every row's, 16 at a time).
const LROWS: usize = 16;
/// Block states a row holds.
const MAXB: usize = 4;

const SRC: &str = r#"
struct IP {
    h: u32, nh: u32, nkv: u32, hd: u32,
    dr: u32, kc: u32, nb: u32, bstride: u32,
    ostride: u32, ring: u32, win: u32, e: u32,
    full: u32, rows: u32, r: u32, inter: u32,
    v: u32, bslot: u32, topk: u32, nq: u32,
    eps: f32, qclip: f32, rclip: f32, alpha: f32,
    floor_: f32, scale: f32, cap: f32, ucap: f32,
    rscale: f32, pad1: f32, pad2: f32, pad3: f32,
};
// the call's first position and rows
struct CR { pos0: u32, n: u32, pad0: u32, pad1: u32 };

@group(0) @binding(0) var<uniform> p: IP;
@group(0) @binding(1) var<storage, read> cr: CR;
@group(0) @binding(2) var<storage, read> a0: array<f32>;
@group(0) @binding(3) var<storage, read> a1: array<f32>;
@group(0) @binding(4) var<storage, read> a2: array<f32>;
@group(0) @binding(5) var<storage, read> a3: array<f32>;
@group(0) @binding(6) var<storage, read> a4: array<f32>;
@group(0) @binding(7) var<storage, read_write> o0: array<f32>;
@group(0) @binding(8) var<storage, read_write> o1: array<f32>;
@group(0) @binding(9) var<storage, read_write> o2: array<f32>;
@group(0) @binding(10) var<storage, read_write> o3: array<f32>;
@group(0) @binding(11) var<storage, read> bank: array<u32>;
@group(0) @binding(12) var<storage, read> ui: array<u32>;
@group(0) @binding(13) var<storage, read_write> uo: array<u32>;
// the bank as 16-byte words (one q4tp nibble group) and a row source as vec4s
@group(0) @binding(14) var<storage, read> bank4: array<vec4<u32>>;
@group(0) @binding(15) var<storage, read> xv4: array<vec4<f32>>;

var<workgroup> red: array<f32, 256>;
var<workgroup> sh8: array<f32, 16>;

// the workgroup's sum of v (256 lanes); every lane gets it
fn wsum(lid: u32, v: f32) -> f32 {
    workgroupBarrier();
    red[lid] = v;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { red[lid] = red[lid] + red[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let t = red[0];
    workgroupBarrier();
    return t;
}

// the sum of v over 128 lanes
fn wsum128(lid: u32, v: f32) -> f32 {
    workgroupBarrier();
    red[lid] = v;
    workgroupBarrier();
    var s = 64u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { red[lid] = red[lid] + red[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let t = red[0];
    workgroupBarrier();
    return t;
}

fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

// ── The block attention residual: a0 blocks, a1 x, a2 queries → o0.
// Grid (nq, n), 256.
@compute @workgroup_size(256)
fn ink_mix(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let qi = w.x; let r = w.y; let H = p.h; let nb = p.nb; let ns = nb + 1u;
    var lg: array<f32, 8>;
    var rn: array<f32, 8>;
    for (var s = 0u; s < ns; s = s + 1u) {
        var ss = 0.0; var dt = 0.0;
        for (var i = lid; i < H; i = i + 256u) {
            var v: f32;
            if (s < nb) { v = a0[r * p.bstride + s * H + i]; } else { v = a1[r * H + i]; }
            ss = ss + v * v;
            dt = dt + v * a2[qi * H + i];
        }
        let s2 = wsum(lid, ss);
        let d2 = wsum(lid, dt);
        let inv = inverseSqrt(s2 / f32(H) + 1e-6);
        rn[s] = inv;
        lg[s] = d2 * inv * inverseSqrt(f32(H));
    }
    var mx = -3.0e38;
    for (var s = 0u; s < ns; s = s + 1u) { mx = max(mx, lg[s]); }
    var wt: array<f32, 8>;
    var den = 0.0;
    for (var s = 0u; s < ns; s = s + 1u) { wt[s] = exp(lg[s] - mx); den = den + wt[s]; }
    for (var s = 0u; s < ns; s = s + 1u) { wt[s] = wt[s] / den * rn[s]; }
    for (var i = lid; i < H; i = i + 256u) {
        var acc = 0.0;
        for (var s = 0u; s < ns; s = s + 1u) {
            var v: f32;
            if (s < nb) { v = a0[r * p.bstride + s * H + i]; } else { v = a1[r * H + i]; }
            acc = acc + wt[s] * v;
        }
        o0[qi * p.ostride + r * H + i] = acc;
    }
}

// ── Weighted RMS norm: a0 x, a1 w → o0. Grid n, 256.
@compute @workgroup_size(256)
fn ink_rms(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = w.x; let H = p.h;
    var ss = 0.0;
    for (var i = lid; i < H; i = i + 256u) { let v = a0[r * H + i]; ss = ss + v * v; }
    let inv = inverseSqrt(wsum(lid, ss) / f32(H) + p.eps);
    for (var i = lid; i < H; i = i + 256u) { o0[r * H + i] = a1[i] * (a0[r * H + i] * inv); }
}

// ── Message gate, first half: a0 x, a1 m, a2 down → o0 d. Grid (R, n), 256.
@compute @workgroup_size(256)
fn ink_gate_down(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let k = w.x; let r = w.y; let H = p.h;
    var sx = 0.0; var sm = 0.0; var ax = 0.0; var am = 0.0;
    for (var i = lid; i < H; i = i + 256u) {
        let a = a0[r * H + i]; let b = a1[r * H + i];
        sx = sx + a * a; sm = sm + b * b;
        ax = ax + a2[k * 2u * H + i] * a;
        am = am + a2[k * 2u * H + H + i] * b;
    }
    let sx2 = wsum(lid, sx); let sm2 = wsum(lid, sm);
    let ax2 = wsum(lid, ax); let am2 = wsum(lid, am);
    if (lid == 0u) {
        o0[r * p.r + k] = ax2 * inverseSqrt(sx2 / f32(H) + p.eps) + am2 * inverseSqrt(sm2 / f32(H) + p.eps);
    }
}

// ── Second half: a0 x, a1 m, a2 d, a3 F, a4 G → o0; 32 lanes an output
// channel. Grid (H/8, n), 256.
@compute @workgroup_size(256)
fn ink_gate_out(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let sg = lid >> 5u; let lane = lid & 31u;
    let i = w.x * 8u + sg; let r = w.y; let H = p.h; let R = p.r;
    var f = 0.0; var g = 0.0;
    if (i < H) {
        for (var k = lane; k < R; k = k + 32u) {
            let dk = a2[r * R + k];
            f = f + a3[i * R + k] * dk;
            g = g + a4[i * R + k] * dk;
        }
    }
    red[lid] = f;
    workgroupBarrier();
    var fs = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) { fs = fs + red[sg * 32u + j]; }
    workgroupBarrier();
    red[lid] = g;
    workgroupBarrier();
    var gs = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) { gs = gs + red[sg * 32u + j]; }
    if (lane == 0u && i < H) {
        let at = r * H + i;
        o0[at] = (1.0 - tanh(fs)) * a0[at] + tanh(gs) * a1[at];
    }
}

// ── y[r][o] = W[o]·x[r], f32: a0 W, a1 x → o0. Grid (rows/8, n), 256 (32 lanes a row).
@compute @workgroup_size(256)
fn ink_fmv(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let sg = lid >> 5u; let lane = lid & 31u;
    let o = w.x * 8u + sg; let r = w.y; let H = p.h;
    var acc = 0.0;
    if (o < p.rows) {
        for (var i = lane; i < H; i = i + 32u) { acc = acc + a0[o * H + i] * a1[r * H + i]; }
    }
    red[lid] = acc;
    workgroupBarrier();
    if (lane == 0u && o < p.rows) {
        var t = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { t = t + red[sg * 32u + j]; }
        o0[r * p.rows + o] = t;
    }
}

fn tau_of(pos: u32) -> f32 { return 1.0 + p.alpha * log(max(f32(pos + 1u) / p.floor_, 1.0)); }

// ── Queries (o0) and relative states (o1) in place; a0 q scale, a1 rel scale.
// Grid (NH, n), 128.
@compute @workgroup_size(128)
fn ink_qr(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = w.x; let r = w.y; let HD = p.hd; let DR = p.dr; let NH = p.nh;
    var tu = 1.0;
    if (p.full != 0u) { tu = tau_of(cr.pos0 + r); }
    let qa = (r * NH + h) * HD + lid;
    let v = o0[qa] * a0[h * HD + lid];
    let n2 = wsum128(lid, v * v);
    o0[qa] = v * min(p.qclip / (sqrt(n2) + 1e-6), 1.0) * tu;
    var a = 0.0;
    if (lid < DR) { a = o1[(r * NH + h) * DR + lid] * a1[h * DR + lid]; }
    let t = wsum128(lid, a * a);
    if (lid < DR) { o1[(r * NH + h) * DR + lid] = a * min(p.rclip / (sqrt(t) + 1e-6), 1.0) * tu; }
}

// ── Keys (y 0) or values (1) into the cache, a KV head a workgroup of HD lanes.
// a0 yk, a1 yv, a2 dk, a3 dv, a4 taps [k | v] (C·KC each), o0 ring k, o1 ring v,
// o2 kcache, o3 vcache. Grid (NKV, 2), 128.
@compute @workgroup_size(128)
fn ink_kv(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = w.x; let isv = w.y; let HD = p.hd; let NKV = p.nkv; let KC = p.kc;
    let C = NKV * HD; let ch = g * HD + lid;
    let toff = isv * C * KC;
    var hist: array<f32, 8>;
    hist[0] = 0.0;
    for (var l = 1u; l < KC; l = l + 1u) {
        if (isv == 0u) { hist[l] = o0[(l - 1u) * C + ch]; } else { hist[l] = o1[(l - 1u) * C + ch]; }
    }
    let dyn = p.full != 0u;
    var sc = 1.0 / sqrt(f32(KC + 1u));
    if (dyn) { sc = 1.0 / sqrt(f32(2u * KC + 1u)); }
    for (var j = 0u; j < cr.n; j = j + 1u) {
        var x: f32;
        if (isv == 0u) { x = a0[j * C + ch]; } else { x = a1[j * C + ch]; }
        hist[0] = x;
        var acc = 0.0;
        for (var l = 0u; l < KC; l = l + 1u) {
            var wv = a4[toff + ch * KC + l];
            if (dyn) {
                if (isv == 0u) { wv = wv + a2[(j * NKV + g) * KC + l]; } else { wv = wv + a3[(j * NKV + g) * KC + l]; }
            }
            acc = acc + wv * hist[l];
        }
        var v = (x + acc) * sc;
        for (var l = KC - 1u; l > 0u; l = l - 1u) { hist[l] = hist[l - 1u]; }
        let n2 = wsum128(lid, v * v);
        if (isv == 1u) { v = v * inverseSqrt(n2 / f32(HD) + p.eps); } else { v = v * min(p.qclip / (sqrt(n2) + 1e-6), 1.0); }
        let pos = cr.pos0 + j;
        var row = pos;
        if (p.ring > 0u) { row = pos % p.ring; }
        if (isv == 0u) { o2[row * C + ch] = v; } else { o3[row * C + ch] = v; }
    }
    for (var l = 1u; l < KC; l = l + 1u) {
        if (isv == 0u) { o0[(l - 1u) * C + ch] = hist[l]; } else { o1[(l - 1u) * C + ch] = hist[l]; }
    }
}

var<workgroup> bias_s: array<f32, 1024>;
var<workgroup> sc_s: array<f32, 1024>;
var<workgroup> q_s: array<f32, 128>;
var<workgroup> part_s: array<vec4<f32>, 256>;

// the max of v over 256 lanes
fn wmax(lid: u32, v: f32) -> f32 {
    workgroupBarrier();
    red[lid] = v;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { red[lid] = max(red[lid], red[lid + s]); }
        workgroupBarrier();
        s = s >> 1u;
    }
    let t = red[0];
    workgroupBarrier();
    return t;
}

// ── The attention of row r and head h, keys in tiles of 1024: a lane a key
// for the scores (q·k·scale + the relative bias), the tile's softmax step
// (online across tiles, the sink logit first), then eight groups of 32 lanes
// over the tile's keys, a lane four channels, for the weighted values. a0 yq,
// a1 yr, a2 sinks, a3 table, a4 kcache, o0 out, o1 vcache. Grid (NH, n), 256
// (HD 128).
@compute @workgroup_size(256)
fn ink_attn(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = w.x; let r = w.y; let HD = p.hd; let NKV = p.nkv; let NH = p.nh; let DR = p.dr; let E = p.e;
    let C = NKV * HD; let g = h / (NH / NKV);
    let pos = cr.pos0 + r;
    var first = 0u;
    if (p.win > 0u && pos + 1u > p.win) { first = pos + 1u - p.win; }
    let nk = pos + 1u - first;
    let near = min(E, nk);
    for (var d = lid; d < near; d = d + 256u) {
        var b = 0.0;
        for (var j = 0u; j < DR; j = j + 1u) { b = b + a1[(r * NH + h) * DR + j] * a3[j * E + d]; }
        bias_s[d] = b;
    }
    if (lid < 128u) { q_s[lid] = a0[(r * NH + h) * HD + lid]; }
    workgroupBarrier();
    let grp = lid >> 5u; let lane = lid & 31u;
    // the sink: one logit with a zero value
    var m = a2[r * NH + h];
    var l = 1.0;
    var acc = vec4<f32>(0.0);
    var t0 = 0u;
    loop {
        if (t0 >= nk) { break; }
        let tn = min(1024u, nk - t0);
        var tmax = -3.0e38;
        for (var i = lid; i < tn; i = i + 256u) {
            let tpos = first + t0 + i;
            var row = tpos;
            if (p.ring > 0u) { row = tpos % p.ring; }
            let kb = row * C + g * HD;
            var s = 0.0;
            for (var c = 0u; c < 128u; c = c + 4u) {
                s = s + q_s[c] * a4[kb + c] + q_s[c + 1u] * a4[kb + c + 1u] + q_s[c + 2u] * a4[kb + c + 2u] + q_s[c + 3u] * a4[kb + c + 3u];
            }
            s = s * p.scale;
            let dd = pos - tpos;
            if (dd < near) { s = s + bias_s[dd]; }
            sc_s[i] = s;
            tmax = max(tmax, s);
        }
        let mt = wmax(lid, tmax);
        let mn = max(m, mt);
        let cfac = exp(m - mn);
        var ts = 0.0;
        for (var i = lid; i < tn; i = i + 256u) {
            let e = exp(sc_s[i] - mn);
            sc_s[i] = e;
            ts = ts + e;
        }
        let tsum = wsum(lid, ts);
        l = l * cfac + tsum;
        acc = acc * cfac;
        for (var i = grp; i < tn; i = i + 8u) {
            let tpos = first + t0 + i;
            var row = tpos;
            if (p.ring > 0u) { row = tpos % p.ring; }
            let vb = row * C + g * HD + lane * 4u;
            acc = acc + sc_s[i] * vec4<f32>(o1[vb], o1[vb + 1u], o1[vb + 2u], o1[vb + 3u]);
        }
        m = mn;
        t0 = t0 + tn;
        workgroupBarrier();
    }
    part_s[lid] = acc;
    workgroupBarrier();
    if (lid < 32u) {
        var t = vec4<f32>(0.0);
        for (var k = 0u; k < 8u; k = k + 1u) { t = t + part_s[k * 32u + lid]; }
        let ob = (r * NH + h) * HD + lid * 4u;
        o0[ob] = t.x / l;
        o0[ob + 1u] = t.y / l;
        o0[ob + 2u] = t.z / l;
        o0[ob + 3u] = t.w / l;
    }
}

// ── o · σ(gate): o0 in place, a0 gate. Grid (NH·HD/256, n), 256.
@compute @workgroup_size(256)
fn ink_ogate(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let W = p.nh * p.hd;
    let i = w.x * 256u + lid; let r = w.y;
    if (i >= W) { return; }
    let at = r * W + i;
    o0[at] = o0[at] * sigm(a0[at]);
}

// ── A sub-layer's output convolution: a0 a, a1 taps, a2 res; o0 ring, o1 blk,
// o2 out; bslot 0xFFFFFFFF: no block closes. Grid H/256, 256.
@compute @workgroup_size(256)
fn ink_post(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = w.x * 256u + lid; let H = p.h; let KC = p.kc;
    if (i >= H) { return; }
    var hist: array<f32, 8>;
    hist[0] = 0.0;
    for (var l = 1u; l < KC; l = l + 1u) { hist[l] = o0[(l - 1u) * H + i]; }
    let sc = 1.0 / sqrt(f32(KC + 1u));
    let close = p.bslot != 0xFFFFFFFFu;
    for (var j = 0u; j < cr.n; j = j + 1u) {
        let at = j * H + i;
        let x = a0[at];
        hist[0] = x;
        var acc = 0.0;
        for (var l = 0u; l < KC; l = l + 1u) { acc = acc + a1[i * KC + l] * hist[l]; }
        let cv = (x + acc) * sc;
        for (var l = KC - 1u; l > 0u; l = l - 1u) { hist[l] = hist[l - 1u]; }
        let hv = a2[at];
        if (close) {
            o1[j * p.bstride + p.bslot * H + i] = hv;
            o2[at] = cv;
        } else {
            o2[at] = hv + cv;
        }
    }
    for (var l = 1u; l < KC; l = l + 1u) { o0[(l - 1u) * H + i] = hist[l]; }
}

// ── cap · tanh(x / cap) in place: o0. Grid (V/256, n), 256.
@compute @workgroup_size(256)
fn ink_softcap(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = w.x * 256u + lid; let r = w.y;
    if (i >= p.v) { return; }
    let at = r * p.v + i;
    o0[at] = tanh(o0[at] / p.cap) * p.cap;
}

// ── o0[i] = a0[i] for i < p.rows. Grid rows/256, 256.
@compute @workgroup_size(256)
fn ink_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    if (i < p.rows) { o0[i] = a0[i]; }
}

// ── Router logits: a0 W [E][H], a1 x → o0 [n][E]. Grid (E/8, n), 256.
@compute @workgroup_size(256)
fn ink_router(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let sg = lid >> 5u; let lane = lid & 31u;
    let e = w.x * 8u + sg; let r = w.y; let H = p.h;
    var acc = 0.0;
    if (e < p.e) {
        for (var i = lane; i < H; i = i + 32u) { acc = acc + a0[e * H + i] * a1[r * H + i]; }
    }
    red[lid] = acc;
    workgroupBarrier();
    if (lane == 0u && e < p.e) {
        var t = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { t = t + red[sg * 32u + j]; }
        o0[r * p.e + e] = t;
    }
}

var<workgroup> key_s: array<f32, 128>;
var<workgroup> idx_s: array<u32, 128>;

// ── A row's top-k by σ(l) + bias (ties to the lower expert): a0 logits, a1 bias
// → uo picks, o0 weights (renormalized σ times the route scale). A lane an
// expert (E ≤ 128), k argmax rounds over the workgroup. Grid n, 128.
@compute @workgroup_size(128)
fn ink_select(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = w.x; let E = p.e; let K = p.topk;
    var key = -3.0e38;
    if (lid < E) { key = sigm(a0[r * E + lid]) + a1[lid]; }
    var sum = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
        key_s[lid] = key;
        idx_s[lid] = lid;
        workgroupBarrier();
        var st = 64u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) {
                let ka = key_s[lid]; let kb = key_s[lid + st];
                let ia = idx_s[lid]; let ib = idx_s[lid + st];
                if (kb > ka || (kb == ka && ib < ia)) { key_s[lid] = kb; idx_s[lid] = ib; }
            }
            workgroupBarrier();
            st = st >> 1u;
        }
        let bi = idx_s[0];
        workgroupBarrier();
        if (lid == bi) { key = -3.0e38; }
        if (lid == 0u) {
            uo[r * K + k] = bi;
            let sc = sigm(a0[r * E + bi]);
            o0[r * K + k] = sc;
        }
    }
    workgroupBarrier();
    if (lid == 0u) {
        var tot = 0.0;
        for (var k = 0u; k < K; k = k + 1u) { tot = tot + o0[r * K + k]; }
        for (var k = 0u; k < K; k = k + 1u) { o0[r * K + k] = o0[r * K + k] / (tot + 1e-20) * p.rscale; }
    }
}

fn nib(w: u32, s: u32) -> f32 { return f32((w >> s) & 0xFu) - 8.0; }
fn dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return nib(w, 0u) * a.x + nib(w, 4u) * a.y + nib(w, 8u) * a.z + nib(w, 12u) * a.w
         + nib(w, 16u) * b.x + nib(w, 20u) * b.y + nib(w, 24u) * b.z + nib(w, 28u) * b.w;
}
fn bank_byte(off: u32) -> u32 { return (bank[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }

// One q4tp row of the bank matrix at word `wb` (rows × gpr groups) against x
// (32·gpr floats of `xs` from `xo`): this lane's groups g ≡ lane (mod 32).
fn q4row(wb: u32, rows: u32, gpr: u32, r: u32, lane: u32, xo: u32, which: u32) -> f32 {
    let params_w = wb + rows * gpr * 4u;
    let codes_b = (wb + rows * gpr * 4u + rows) * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let pr = unpack2x16float(bank[params_w + r]);
    var acc = 0.0;
    for (var g = lane; g < gpr; g = g + 32u) {
        let bit = g * 5u;
        let cbo = bit >> 3u; let sh = bit & 7u;
        var cv = bank_byte(codes_b + r * cstride + cbo);
        if (sh > 3u) { cv = cv | (bank_byte(codes_b + r * cstride + cbo + 1u) << 8u); }
        let scale = exp2(pr.x + f32((cv >> sh) & 31u) * pr.y);
        let wq = wb + (r * gpr + g) * 4u;
        let x0 = xo + g * 32u;
        var s = 0.0;
        for (var q = 0u; q < 4u; q = q + 1u) {
            let xa = q * 8u;
            var av: vec4<f32>; var bv: vec4<f32>;
            if (which == 0u) {
                av = vec4<f32>(a2[x0 + xa], a2[x0 + xa + 1u], a2[x0 + xa + 2u], a2[x0 + xa + 3u]);
                bv = vec4<f32>(a2[x0 + xa + 4u], a2[x0 + xa + 5u], a2[x0 + xa + 6u], a2[x0 + xa + 7u]);
            } else {
                av = vec4<f32>(a3[x0 + xa], a3[x0 + xa + 1u], a3[x0 + xa + 2u], a3[x0 + xa + 3u]);
                bv = vec4<f32>(a3[x0 + xa + 4u], a3[x0 + xa + 5u], a3[x0 + xa + 6u], a3[x0 + xa + 7u]);
            }
            s = s + dot8(bank[wq + q], av, bv);
        }
        acc = acc + scale * s;
    }
    return acc;
}

// ── Every pick's gate and up rows, the softcapped SwiGLU into o0 act[r][k][inter].
// ui [0..E·3): the experts' word offsets (gate, up, down) in the bank, then the
// picks at ui[E·3 + r·K + k] (written by `ink_select` through uo, the same buffer).
// a2 x. Grid (inter/8, K, n), 256 (32 lanes a row).
@compute @workgroup_size(256)
fn ink_moe_gu(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let sg = lid >> 5u; let lane = lid & 31u;
    let k = w.y; let r = w.z; let inter = p.inter; let gpr = p.h / 32u;
    let e = ui[p.e * 3u + r * p.topk + k];
    let row = w.x * 8u + sg;
    var g = 0.0; var u = 0.0;
    if (row < inter) {
        g = q4row(ui[e * 3u], inter, gpr, row, lane, r * p.h, 0u);
        u = q4row(ui[e * 3u + 1u], inter, gpr, row, lane, r * p.h, 0u);
    }
    red[lid] = g;
    workgroupBarrier();
    var gs = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) { gs = gs + red[sg * 32u + j]; }
    workgroupBarrier();
    red[lid] = u;
    workgroupBarrier();
    var us = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) { us = us + red[sg * 32u + j]; }
    if (lane == 0u && row < inter) {
        o0[(r * p.topk + k) * inter + row] = (p.cap * tanh(gs / p.cap)) * sigm(gs) * (p.ucap * tanh(us / p.ucap));
    }
}

// ── Every row's mixture of its picks' downs: a3 act, a4 weights → o0 y[r].
// Grid (H/8, n), 256.
@compute @workgroup_size(256)
fn ink_moe_down(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let sg = lid >> 5u; let lane = lid & 31u;
    let r = w.y; let H = p.h; let inter = p.inter; let gpr = inter / 32u;
    let row = w.x * 8u + sg;
    var tot = 0.0;
    for (var k = 0u; k < p.topk; k = k + 1u) {
        let e = ui[p.e * 3u + r * p.topk + k];
        var acc = 0.0;
        if (row < H) { acc = q4row(ui[e * 3u + 2u], H, gpr, row, lane, (r * p.topk + k) * inter, 1u); }
        red[lid] = acc;
        workgroupBarrier();
        var s = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { s = s + red[sg * 32u + j]; }
        workgroupBarrier();
        tot = tot + a4[r * p.topk + k] * s;
    }
    if (lane == 0u && row < H) { o0[r * H + row] = tot; }
}

// ── The decode's experts four rows a workgroup (the Mellum/Flash-Next `moe_r4`
// shape on this bank): a lane's 32-column group of the row source read once
// for every row it meets, each row's nibble group one vec4<u32>, the partial
// sums through one 64-lane tree.
var<workgroup> m4a: array<vec4<f32>, 64>;
var<workgroup> m4b: array<vec4<f32>, 64>;

fn m4_n4(w: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(f32((w >> sh) & 0xFu), f32((w >> (sh + 4u)) & 0xFu),
                     f32((w >> (sh + 8u)) & 0xFu), f32((w >> (sh + 12u)) & 0xFu)) - vec4<f32>(8.0);
}
fn m4_d8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 { return dot(m4_n4(w, 0u), a) + dot(m4_n4(w, 16u), b); }
fn m4_d32(w: vec4<u32>, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>,
          x4: vec4<f32>, x5: vec4<f32>, x6: vec4<f32>, x7: vec4<f32>) -> f32 {
    return (m4_d8(w.x, x0, x1) + m4_d8(w.y, x2, x3)) + (m4_d8(w.z, x4, x5) + m4_d8(w.w, x6, x7));
}
// a group's 5-bit scale code: `ob` the row's code bytes, `bit` the group's bit
fn m4_code(ob: u32, bit: u32) -> u32 {
    let o = ob + (bit >> 3u); let sh = bit & 7u;
    var c = bank_byte(o);
    if (sh > 3u) { c = c | (bank_byte(o + 1u) << 8u); }
    return (c >> sh) & 31u;
}
// the scales of rows row0..row0+3 of the matrix at word `wb` (rows × gpr) at group g
fn m4_scales(wb: u32, rows: u32, gpr: u32, row0: u32, g: u32) -> vec4<f32> {
    let par = wb + rows * gpr * 4u + row0;
    let a = unpack2x16float(bank[par]); let b = unpack2x16float(bank[par + 1u]);
    let c = unpack2x16float(bank[par + 2u]); let d = unpack2x16float(bank[par + 3u]);
    let cst = (gpr * 5u + 7u) / 8u;
    let ob = (wb + rows * gpr * 4u + rows) * 4u + row0 * cst;
    let bit = g * 5u;
    let cv = vec4<u32>(m4_code(ob, bit), m4_code(ob + cst, bit), m4_code(ob + 2u * cst, bit), m4_code(ob + 3u * cst, bit));
    return exp2(vec4<f32>(a.x, b.x, c.x, d.x) + vec4<f32>(cv) * vec4<f32>(a.y, b.y, c.y, d.y));
}
fn m4_reduce(lid: u32) {
    workgroupBarrier();
    var st = 32u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { m4a[lid] = m4a[lid] + m4a[lid + st]; m4b[lid] = m4b[lid] + m4b[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
}

// Every pick's gate and up, rows 4·w.x .. +3: the softcapped SwiGLU into o0
// act[r][k][inter]. xv4 x (rows H apart), ui the table and picks.
// Grid (inter/4, K, n), 64.
@compute @workgroup_size(64)
fn ink_moe_gu4(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let k = w.y; let r = w.z; let inter = p.inter; let gpr = p.h / 32u;
    let e = ui[p.e * 3u + r * p.topk + k];
    let gw = ui[e * 3u]; let uw = ui[e * 3u + 1u];
    let row0 = w.x * 4u;
    let xb = r * (p.h / 4u);
    var ag = vec4<f32>(0.0); var au = vec4<f32>(0.0);
    for (var g = lid; g < gpr; g = g + 64u) {
        let xo = xb + g * 8u;
        let x0 = xv4[xo]; let x1 = xv4[xo + 1u]; let x2 = xv4[xo + 2u]; let x3 = xv4[xo + 3u];
        let x4 = xv4[xo + 4u]; let x5 = xv4[xo + 5u]; let x6 = xv4[xo + 6u]; let x7 = xv4[xo + 7u];
        let wg = gw / 4u + row0 * gpr + g;
        let wu = uw / 4u + row0 * gpr + g;
        let dg = vec4<f32>(m4_d32(bank4[wg], x0, x1, x2, x3, x4, x5, x6, x7), m4_d32(bank4[wg + gpr], x0, x1, x2, x3, x4, x5, x6, x7),
                           m4_d32(bank4[wg + 2u * gpr], x0, x1, x2, x3, x4, x5, x6, x7), m4_d32(bank4[wg + 3u * gpr], x0, x1, x2, x3, x4, x5, x6, x7));
        let du = vec4<f32>(m4_d32(bank4[wu], x0, x1, x2, x3, x4, x5, x6, x7), m4_d32(bank4[wu + gpr], x0, x1, x2, x3, x4, x5, x6, x7),
                           m4_d32(bank4[wu + 2u * gpr], x0, x1, x2, x3, x4, x5, x6, x7), m4_d32(bank4[wu + 3u * gpr], x0, x1, x2, x3, x4, x5, x6, x7));
        ag = ag + m4_scales(gw, inter, gpr, row0, g) * dg;
        au = au + m4_scales(uw, inter, gpr, row0, g) * du;
    }
    m4a[lid] = ag; m4b[lid] = au;
    m4_reduce(lid);
    if (lid < 4u && row0 + lid < inter) {
        let gs = m4a[0][lid]; let us = m4b[0][lid];
        o0[(r * p.topk + k) * inter + row0 + lid] = (p.cap * tanh(gs / p.cap)) * sigm(gs) * (p.ucap * tanh(us / p.ucap));
    }
}

// Every row's mixture of its picks' downs, rows 4·w.x .. +3: xv4 act
// [r][k][inter], a4 the picks' weights → o0 y[r]. Grid (H/4, n), 64.
@compute @workgroup_size(64)
fn ink_moe_dn4(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = w.y; let H = p.h; let inter = p.inter; let gpr = inter / 32u; let K = p.topk;
    let row0 = w.x * 4u;
    var acc = vec4<f32>(0.0);
    for (var i = lid; i < K * gpr; i = i + 64u) {
        let k = i / gpr; let g = i - k * gpr;
        let e = ui[p.e * 3u + r * K + k];
        let dw = ui[e * 3u + 2u];
        let xo = (r * K + k) * (inter / 4u) + g * 8u;
        let a0 = xv4[xo]; let a1 = xv4[xo + 1u]; let a2v = xv4[xo + 2u]; let a3v = xv4[xo + 3u];
        let a4v = xv4[xo + 4u]; let a5 = xv4[xo + 5u]; let a6 = xv4[xo + 6u]; let a7 = xv4[xo + 7u];
        let wv = dw / 4u + row0 * gpr + g;
        let d = vec4<f32>(m4_d32(bank4[wv], a0, a1, a2v, a3v, a4v, a5, a6, a7), m4_d32(bank4[wv + gpr], a0, a1, a2v, a3v, a4v, a5, a6, a7),
                          m4_d32(bank4[wv + 2u * gpr], a0, a1, a2v, a3v, a4v, a5, a6, a7), m4_d32(bank4[wv + 3u * gpr], a0, a1, a2v, a3v, a4v, a5, a6, a7));
        acc = acc + a4[r * K + k] * (m4_scales(dw, H, gpr, row0, g) * d);
    }
    m4a[lid] = acc; m4b[lid] = vec4<f32>(0.0);
    m4_reduce(lid);
    if (lid < 4u && row0 + lid < H) { o0[r * H + row0 + lid] = m4a[0][lid]; }
}

// ── A prompt call's routed experts grouped by expert: an expert's rows read its
// weights once a call (from the cache after the first entry), not once a pick.

var<workgroup> gcnt: array<u32, 1024>;
var<workgroup> g_i0: u32;
var<workgroup> g_i1: u32;

// The (row, pick) entries by expert: ui the picks (the table from E·3), uo
// [0..E) the first entry of each expert, uo[E] the count, uo[E + 1 + q] the
// q-th entry (r·K + k), uo[E + 1 + nk + entry] its place q. p.rows: the
// call's rows. Grid 1, 256.
@compute @workgroup_size(256)
fn ink_group(@builtin(local_invocation_index) lid: u32) {
    let E = p.e; let nk = p.rows * p.topk;
    for (var e = lid; e < E; e = e + 256u) {
        var c = 0u;
        for (var i = 0u; i < nk; i = i + 1u) { if (ui[i] == e) { c = c + 1u; } }
        gcnt[e] = c;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var o = 0u;
        for (var e = 0u; e < E; e = e + 1u) { let c = gcnt[e]; gcnt[e] = o; o = o + c; }
        uo[E] = o;
    }
    workgroupBarrier();
    for (var e = lid; e < E; e = e + 256u) {
        var q = gcnt[e];
        uo[e] = q;
        for (var i = 0u; i < nk; i = i + 1u) { if (ui[i] == e) { uo[E + 1u + nk + i] = q; uo[E + 1u + q] = i; q = q + 1u; } }
    }
}

// An expert's entries, uniform over the workgroup: [i0, i1) of uo's list.
fn entries(lid: u32, e: u32) -> vec2<u32> {
    if (lid == 0u) {
        g_i0 = uo[e];
        if (e + 1u < p.e) { g_i1 = uo[e + 1u]; } else { g_i1 = uo[p.e]; }
    }
    return vec2<u32>(workgroupUniformLoad(&g_i0), workgroupUniformLoad(&g_i1));
}

// An expert's gate and up rows over its entries, the softcapped SwiGLU into
// o0 act[entry][inter]. a2 x, ui the table, uo the grouped entries.
// Grid (inter/8, E), 256 (32 lanes a row).
@compute @workgroup_size(256)
fn ink_moe_gu_g(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let sg = lid >> 5u; let lane = lid & 31u;
    let e = w.y; let inter = p.inter; let gpr = p.h / 32u;
    let row = w.x * 8u + sg;
    let span = entries(lid, e);
    for (var i = span.x; i < span.y; i = i + 1u) {
        let ent = uo[p.e + 1u + i];
        let r = ent / p.topk;
        var g = 0.0; var u = 0.0;
        if (row < inter) {
            g = q4row(ui[e * 3u], inter, gpr, row, lane, r * p.h, 0u);
            u = q4row(ui[e * 3u + 1u], inter, gpr, row, lane, r * p.h, 0u);
        }
        red[lid] = g;
        workgroupBarrier();
        var gs = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { gs = gs + red[sg * 32u + j]; }
        workgroupBarrier();
        red[lid] = u;
        workgroupBarrier();
        var us = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { us = us + red[sg * 32u + j]; }
        workgroupBarrier();
        if (lane == 0u && row < inter) {
            o0[ent * inter + row] = (p.cap * tanh(gs / p.cap)) * sigm(gs) * (p.ucap * tanh(us / p.ucap));
        }
    }
}

// An expert's down rows over its entries into o0 yp[entry][H] (unweighted).
// a3 act. Grid (H/8, E), 256.
@compute @workgroup_size(256)
fn ink_moe_down_g(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let sg = lid >> 5u; let lane = lid & 31u;
    let e = w.y; let H = p.h; let inter = p.inter; let gpr = inter / 32u;
    let row = w.x * 8u + sg;
    let span = entries(lid, e);
    for (var i = span.x; i < span.y; i = i + 1u) {
        let ent = uo[p.e + 1u + i];
        var acc = 0.0;
        if (row < H) { acc = q4row(ui[e * 3u + 2u], H, gpr, row, lane, ent * inter, 1u); }
        red[lid] = acc;
        workgroupBarrier();
        var s = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { s = s + red[sg * 32u + j]; }
        workgroupBarrier();
        if (lane == 0u && row < H) { o0[ent * H + row] = s; }
    }
}

// ── The grouped experts as tiled GEMMs: o0 Y[q][rows] = W_e · X[q] over the
// places q of expert e. p.v 0 / 1: the gate / up of e over the gathered rows
// a2 x[entry / K] (cols H); 2: the down of e over a2 A[q] (cols inter). ui the
// table, uo the grouped entries. A workgroup a tile of 64 output rows × 32
// places, K in steps of 32 through workgroup memory (the weights decoded once
// a tile), 8 sums a lane. Grid (rows of a call / 32, out rows / 64, E), 256.
var<workgroup> wt_s: array<f32, 2112>;
var<workgroup> xt_s: array<f32, 1056>;
@compute @workgroup_size(256)
fn ink_moe_mm(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let which = p.v; let e = w.z; let E = p.e; let K = p.topk;
    let span = entries(lid, e);
    let nb = span.y - span.x;
    let r1 = w.x * 32u;
    if (r1 >= nb) { return; }
    var cols = p.h; var rows = p.inter;
    if (which == 2u) { cols = p.inter; rows = p.h; }
    let r0 = w.y * 64u;
    if (r0 >= rows) { return; }
    let wb = ui[e * 3u + which];
    let gpr = cols / 32u;
    let params_w = wb + rows * gpr * 4u;
    let codes_b = (wb + rows * gpr * 4u + rows) * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    // the loads: a weight row's quarter group (one word, 8 nibbles), an input row's 4 floats
    let lrow = lid >> 2u; let quarter = lid & 3u;
    let wrow = min(r0 + lrow, rows - 1u);
    let pr = unpack2x16float(bank[params_w + wrow]);
    let xent = lid >> 3u; let x4 = (lid & 7u) * 4u;
    let qp = span.x + r1 + min(xent, nb - r1 - 1u);
    var xrow = qp;
    if (which != 2u) { xrow = uo[E + 1u + qp] / K; }
    let xbase = xrow * cols;
    // the sums: rows rb..rb+3, places eb, eb+1
    let rb = (lid & 15u) * 4u; let eb = (lid >> 4u) * 2u;
    var c00 = 0.0; var c01 = 0.0; var c10 = 0.0; var c11 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c30 = 0.0; var c31 = 0.0;
    for (var g = 0u; g < gpr; g = g + 1u) {
        workgroupBarrier();
        let bit = g * 5u; let cbo = bit >> 3u; let sh = bit & 7u;
        var cv = bank_byte(codes_b + wrow * cstride + cbo);
        if (sh > 3u) { cv = cv | (bank_byte(codes_b + wrow * cstride + cbo + 1u) << 8u); }
        let scale = exp2(pr.x + f32((cv >> sh) & 31u) * pr.y);
        let wq = bank[wb + (wrow * gpr + g) * 4u + quarter];
        let wo = lrow * 33u + quarter * 8u;
        for (var i = 0u; i < 8u; i = i + 1u) { wt_s[wo + i] = (f32((wq >> (4u * i)) & 0xFu) - 8.0) * scale; }
        let xo = xbase + g * 32u + x4;
        let xs = xent * 33u + x4;
        xt_s[xs] = a2[xo]; xt_s[xs + 1u] = a2[xo + 1u]; xt_s[xs + 2u] = a2[xo + 2u]; xt_s[xs + 3u] = a2[xo + 3u];
        workgroupBarrier();
        for (var k = 0u; k < 32u; k = k + 1u) {
            let a0 = wt_s[rb * 33u + k]; let a1 = wt_s[(rb + 1u) * 33u + k];
            let a2v = wt_s[(rb + 2u) * 33u + k]; let a3 = wt_s[(rb + 3u) * 33u + k];
            let b0 = xt_s[eb * 33u + k]; let b1 = xt_s[(eb + 1u) * 33u + k];
            c00 = c00 + a0 * b0; c01 = c01 + a0 * b1;
            c10 = c10 + a1 * b0; c11 = c11 + a1 * b1;
            c20 = c20 + a2v * b0; c21 = c21 + a2v * b1;
            c30 = c30 + a3 * b0; c31 = c31 + a3 * b1;
        }
    }
    // the places eb, eb+1 of the tile that hold an entry, the rows that exist
    let base = span.x + r1;
    if (eb < nb - r1) {
        let o = (base + eb) * rows + r0 + rb;
        if (r0 + rb < rows) { o0[o] = c00; }
        if (r0 + rb + 1u < rows) { o0[o + 1u] = c10; }
        if (r0 + rb + 2u < rows) { o0[o + 2u] = c20; }
        if (r0 + rb + 3u < rows) { o0[o + 3u] = c30; }
    }
    if (eb + 1u < nb - r1) {
        let o = (base + eb + 1u) * rows + r0 + rb;
        if (r0 + rb < rows) { o0[o] = c01; }
        if (r0 + rb + 1u < rows) { o0[o + 1u] = c11; }
        if (r0 + rb + 2u < rows) { o0[o + 2u] = c21; }
        if (r0 + rb + 3u < rows) { o0[o + 3u] = c31; }
    }
}

// The softcapped SwiGLU of the places' gate and up rows: o0 A = f(a0 G, a1 U).
// Grid (inter/256, nk), 256.
@compute @workgroup_size(256)
fn ink_moe_act(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = w.x * 256u + lid; let q = w.y; let inter = p.inter;
    if (i >= inter || q >= p.rows * p.topk) { return; }
    let at = q * inter + i;
    let g = a0[at]; let u = a1[at];
    o0[at] = (p.cap * tanh(g / p.cap)) * sigm(g) * (p.ucap * tanh(u / p.ucap));
}

// Every row's mixture of its picks' downs from the places: o0 y[r] = Σ_k
// a1 wt[r][k]·a0 Yp[place(r·K + k)], ui the grouped entries. Grid (H/256, n), 256.
@compute @workgroup_size(256)
fn ink_moe_mixq(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = w.x * 256u + lid; let r = w.y; let H = p.h; let nk = p.rows * p.topk;
    if (i >= H) { return; }
    var acc = 0.0;
    for (var k = 0u; k < p.topk; k = k + 1u) {
        let q = ui[p.e + 1u + nk + r * p.topk + k];
        acc = acc + a1[r * p.topk + k] * a0[q * H + i];
    }
    o0[r * H + i] = acc;
}

// Every row's mixture of its picks' downs: o0 y[r] = Σ_k a1 wt[r][k]·a0 yp[r·K + k].
// Grid (H/256, n), 256.
@compute @workgroup_size(256)
fn ink_moe_mix(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = w.x * 256u + lid; let r = w.y; let H = p.h;
    if (i >= H) { return; }
    var acc = 0.0;
    for (var k = 0u; k < p.topk; k = k + 1u) { acc = acc + a1[r * p.topk + k] * a0[(r * p.topk + k) * H + i]; }
    o0[r * H + i] = acc;
}
"#;

/// `IP` of the shader.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct IP {
    h: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
    dr: u32,
    kc: u32,
    nb: u32,
    bstride: u32,
    ostride: u32,
    ring: u32,
    win: u32,
    e: u32,
    full: u32,
    rows: u32,
    r: u32,
    inter: u32,
    v: u32,
    bslot: u32,
    topk: u32,
    nq: u32,
    eps: f32,
    qclip: f32,
    rclip: f32,
    alpha: f32,
    floor_: f32,
    scale: f32,
    cap: f32,
    ucap: f32,
    rscale: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
}

struct Pipes {
    by_name: HashMap<&'static str, wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

const KERNELS: [&str; 25] = [
    "ink_mix", "ink_rms", "ink_gate_down", "ink_gate_out", "ink_fmv", "ink_qr", "ink_kv", "ink_attn", "ink_ogate", "ink_post", "ink_softcap", "ink_copy", "ink_router",
    "ink_select", "ink_moe_gu", "ink_moe_down", "ink_group", "ink_moe_gu_g", "ink_moe_down_g", "ink_moe_mix", "ink_moe_mm", "ink_moe_act", "ink_moe_mixq", "ink_moe_gu4",
    "ink_moe_dn4",
];

/// The writable bindings of the shader (the others read-only storage,
/// binding 0 the uniform).
const WRITABLE: [u32; 5] = [7, 8, 9, 10, 13];

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            // one layout for every entry point: a dispatch binds all sixteen
            // (a dummy where a kernel reads nothing)
            let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..16u32)
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if b == 0 { wgpu::BufferBindingType::Uniform } else { wgpu::BufferBindingType::Storage { read_only: !WRITABLE.contains(&b) } },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            let layout = c.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("inkling"), entries: &entries });
            let pl = c.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("inkling"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("inkling"), source: wgpu::ShaderSource::Wgsl(SRC.into()) });
            let mut by_name = HashMap::new();
            for k in KERNELS {
                let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(k),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(k),
                    compilation_options: Default::default(),
                    cache: None,
                });
                by_name.insert(k, p);
            }
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("inkling2_loop on wgpu: shaders: {e}");
                return None;
            }
            Some(Pipes { by_name, layout })
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

struct LayerW {
    sliding: bool,
    boundary: bool,
    aq: Option<(wgpu::Buffer, usize)>,
    mq: wgpu::Buffer,
    norms: [wgpu::Buffer; 3],
    post_norm: wgpu::Buffer,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    rel: Mat,
    sink: wgpu::Buffer,
    q_scale: wgpu::Buffer,
    rel_scale: wgpu::Buffer,
    rel_table: wgpu::Buffer,
    extent: usize,
    /// k taps then v taps.
    taps: wgpu::Buffer,
    dk: Option<wgpu::Buffer>,
    dv: Option<wgpu::Buffer>,
    tpa: wgpu::Buffer,
    tpm: wgpu::Buffer,
    router: wgpu::Buffer,
    bias: wgpu::Buffer,
    /// The layer's experts as one buffer, and the u32 table of their word
    /// offsets `[E][3]` followed by room for the picks (`ink_select` writes them).
    bank: wgpu::Buffer,
    table: wgpu::Buffer,
    e: usize,
    inter: usize,
    gates: Option<[(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, usize); 4]>,
}

struct SlotW {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    cap: usize,
    ring: usize,
    rings: [wgpu::Buffer; 4],
}

#[derive(Clone)]
struct Step {
    /// The kernel (`CMF_INKLING_PROF=1` times each by name).
    name: &'static str,
    pipe: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    x: u32,
    y: u32,
    z: u32,
}

struct Dev {
    layers: Vec<LayerW>,
    slots: Vec<SlotW>,
    head: Mat,
    msg_q: wgpu::Buffer,
    final_q: wgpu::Buffer,
    final_norm: wgpu::Buffer,
    cr: wgpu::Buffer,
    h: [wgpu::Buffer; 2],
    blk: wgpu::Buffer,
    pre_h: wgpu::Buffer,
    bank: wgpu::Buffer,
    bank_out: wgpu::Buffer,
    msg: wgpu::Buffer,
    in3: wgpu::Buffer,
    gin: wgpu::Buffer,
    gd: wgpu::Buffer,
    x: [wgpu::Buffer; 3],
    yq: wgpu::Buffer,
    yg: wgpu::Buffer,
    yr: wgpu::Buffer,
    yk: wgpu::Buffer,
    yv: wgpu::Buffer,
    sk: wgpu::Buffer,
    dkb: wgpu::Buffer,
    dvb: wgpu::Buffer,
    att: wgpu::Buffer,
    ya: wgpu::Buffer,
    r: wgpu::Buffer,
    m: wgpu::Buffer,
    mg: wgpu::Buffer,
    xm: wgpu::Buffer,
    lg: wgpu::Buffer,
    wt: wgpu::Buffer,
    act: wgpu::Buffer,
    /// A multi-row call's (row, pick) entries grouped by expert, and the
    /// picks' down outputs `[row·topk + k][H]`.
    grp: wgpu::Buffer,
    yp: wgpu::Buffer,
    /// The places' gate and up rows of the grouped GEMMs `[place][inter]`.
    gq: wgpu::Buffer,
    uq: wgpu::Buffer,
    y: wgpu::Buffer,
    fin: wgpu::Buffer,
    xf: wgpu::Buffer,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
    dummy: wgpu::Buffer,
    dummy_w: wgpu::Buffer,
    /// Recorded plans by (rows, logits rows).
    plans: HashMap<(usize, usize), Arc<Vec<Step>>>,
}

static DEVICES: Mutex<Option<HashMap<u64, Dev>>> = Mutex::new(None);
static DECLINED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn buf(c: &Ctx, label: &str, bytes: usize) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (bytes.max(16) as u64 + 15) & !15, usage: STORE, mapped_at_creation: false })
}

const STORE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST).union(wgpu::BufferUsages::COPY_SRC);

fn store<T: bytemuck::Pod>(c: &Ctx, label: &str, v: &[T]) -> wgpu::Buffer {
    let b = buf(c, label, std::mem::size_of_val(v));
    if !v.is_empty() {
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(v));
    }
    b
}

fn flush_uploads(c: &Ctx, pending: &mut u64) {
    // a long run of write_buffer without a submit holds its staging in host RAM
    if *pending > 512 << 20 {
        c.queue.submit(None);
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        *pending = 0;
    }
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
    let b = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("inkling-weight"), size: (bytes.len() as u64 + 3) & !3, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    c.queue.write_buffer(&b, 0, &bytes[..bytes.len() / 4 * 4]);
    if bytes.len() % 4 != 0 {
        let mut tail = [0u8; 4];
        let rest = &bytes[bytes.len() / 4 * 4..];
        tail[..rest.len()].copy_from_slice(rest);
        c.queue.write_buffer(&b, (bytes.len() / 4 * 4) as u64, &tail);
    }
    *pending += bytes.len() as u64;
    flush_uploads(c, pending);
    Some(Mat { buf: b, codec, rows: t.rows(), cols: t.cols() })
}

fn admits(layers: &[Layer], cfg: &Cfg) -> bool {
    cfg.hd == 128
        && cfg.nh % cfg.nkv == 0
        && cfg.dim % 256 == 0
        && cfg.d_rel <= 32
        && cfg.window <= 1024
        && cfg.rel_extent <= 1024
        && cfg.top_k <= 16
        && cfg.layers / cfg.block.max(1) <= MAXB
        && layers.iter().all(|l| l.moe.bias.len() <= 128)
        && layers.iter().all(|l| l.moe.experts.iter().all(|[g, u, d]| g.q4tp_mapped().is_some() && u.q4tp_mapped().is_some() && d.q4tp_mapped().is_some()))
}

fn build(c: &'static Ctx, g: &Globals, layers: &[Layer], cfg: &Cfg) -> Result<Dev, String> {
    pipes(c).ok_or("shaders")?;
    let t0 = std::time::Instant::now();
    let (hidden, nh, nkv, hd, dr) = (cfg.dim, cfg.nh, cfg.nkv, cfg.hd, cfg.d_rel);
    let mut pending = 0u64;
    let mut dl = Vec::with_capacity(layers.len());
    let max_bind = c.device.limits().max_storage_buffer_binding_size as usize;
    for (li, l) in layers.iter().enumerate() {
        let a = &l.attn;
        let mut up = |t: &crate::qtensor::QTensor| upload(c, t, &mut pending).ok_or_else(|| format!("layer {li}: a matrix outside q4tp / q8_2f"));
        let (q, k, v, o, gate, rel) = (up(&a.q)?, up(&a.k)?, up(&a.v)?, up(&a.o)?, up(&a.gate)?, up(&a.rel)?);
        // the experts in one bank: each matrix 16-byte aligned, the word offsets in the table
        let mut total = 0usize;
        let mut offs = Vec::with_capacity(l.moe.experts.len() * 3);
        for x in &l.moe.experts {
            for t in x {
                let n = t.mapped_bytes().ok_or("expert bytes")?.len();
                offs.push((total / 4) as u32);
                total += n.div_ceil(16) * 16;
            }
        }
        if total > max_bind {
            return Err(format!("layer {li}: an expert bank of {} MB past the binding limit of {} MB", total >> 20, max_bind >> 20));
        }
        let bank = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("inkling-bank"), size: total as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut at = 0u64;
        for x in &l.moe.experts {
            for t in x {
                let b = t.mapped_bytes().ok_or("expert bytes")?;
                let whole = b.len() / 4 * 4;
                c.queue.write_buffer(&bank, at, &b[..whole]);
                if whole < b.len() {
                    let mut tail = [0u8; 4];
                    tail[..b.len() - whole].copy_from_slice(&b[whole..]);
                    c.queue.write_buffer(&bank, at + whole as u64, &tail);
                }
                at += (b.len().div_ceil(16) * 16) as u64;
                pending += b.len() as u64;
                flush_uploads(c, &mut pending);
            }
        }
        let e = l.moe.bias.len();
        let mut table = offs;
        table.resize(e * 3 + rows_cap() * cfg.top_k, 0);
        let gates = match &l.gates {
            Some(gs) => {
                let mk = |x: &crate::inkling::Gate| (store(c, "gd", &x.down), store(c, "gf", &x.forget), store(c, "gg", &x.gain), x.down.len() / (2 * hidden));
                Some([mk(&gs[0]), mk(&gs[1]), mk(&gs[2]), mk(&gs[3])])
            }
            None => None,
        };
        let mut taps = a.k_conv.taps.clone();
        taps.extend_from_slice(&a.v_conv.taps);
        dl.push(LayerW {
            sliding: l.sliding,
            boundary: li % cfg.block == 0,
            aq: l.attn_query.as_ref().map(|q| (store(c, "aq", q), q.len() / hidden)),
            mq: store(c, "mq", &l.mlp_query),
            norms: [store(c, "n0", &l.in_norm[0]), store(c, "n1", &l.in_norm[1]), store(c, "n2", &l.in_norm[2])],
            post_norm: store(c, "pn", &l.post_norm),
            q,
            k,
            v,
            o,
            gate,
            rel,
            sink: store(c, "sink", &a.sink),
            q_scale: store(c, "qs", &a.q_scale),
            rel_scale: store(c, "rs", &a.rel_scale),
            rel_table: store(c, "rt", &a.rel_table),
            extent: a.rel_table.len() / dr,
            taps: store(c, "taps", &taps),
            dk: a.k_conv.proj.as_ref().map(|p| store(c, "dk", p)),
            dv: a.v_conv.proj.as_ref().map(|p| store(c, "dv", p)),
            tpa: store(c, "tpa", &l.post_attn_conv.taps),
            tpm: store(c, "tpm", &l.post_mlp_conv.taps),
            router: store(c, "router", &l.moe.router),
            bias: store(c, "bias", &l.moe.bias),
            bank,
            table: store(c, "table", &table),
            e,
            inter: l.moe.experts.first().map_or(0, |x| x[0].rows()),
            gates,
        });
    }
    let head = upload(c, &g.lm_head, &mut pending).ok_or("head codec")?;
    c.queue.submit(None);
    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    let kvs = nkv * hd;
    let mut slots: Vec<Option<SlotW>> = (0..cfg.slots()).map(|_| None).collect();
    for (li, pass) in cfg.plan() {
        let ring = if layers[li].sliding { cfg.window + rows_cap() } else { 0 };
        let cap = if ring > 0 { ring } else { 4096 };
        slots[cfg.slot(li, pass)] = Some(SlotW {
            k: buf(c, "kc", cap * kvs * 4),
            v: buf(c, "vc", cap * kvs * 4),
            cap,
            ring,
            rings: [buf(c, "rk", 8 * kvs * 4), buf(c, "rv", 8 * kvs * 4), buf(c, "ra", 8 * hidden * 4), buf(c, "rm", 8 * hidden * 4)],
        });
    }
    let slots: Vec<SlotW> = slots.into_iter().map(|s| s.ok_or_else(|| "a slot without its application".to_string())).collect::<Result<_, _>>()?;
    let rows = rows_cap();
    let f4 = |label: &str, n: usize| buf(c, label, n * 4);
    let (qs, rs) = (nh * hd, nh * dr);
    let e = dl.first().map_or(0, |l| l.e);
    let inter = dl.first().map_or(0, |l| l.inter);
    let vocab = head.rows;
    let d = Dev {
        layers: dl,
        slots,
        msg_q: store(c, "msgq", &g.message_query),
        final_q: store(c, "finq", &g.final_query),
        final_norm: store(c, "finn", &g.norm),
        cr: buf(c, "cr", 16),
        h: [f4("h0", rows * hidden), f4("h1", rows * hidden)],
        blk: f4("blk", rows * MAXB * hidden),
        pre_h: f4("preh", rows * hidden),
        bank: f4("bank", rows * MAXB * hidden),
        bank_out: f4("bout", rows * hidden),
        msg: f4("msg", 4 * rows * hidden),
        in3: f4("in3", 3 * rows * hidden),
        gin: f4("gin", 3 * rows * hidden),
        gd: f4("gd", rows * 256),
        x: [f4("x0", rows * hidden), f4("x1", rows * hidden), f4("x2", rows * hidden)],
        yq: f4("yq", rows * qs),
        yg: f4("yg", rows * qs),
        yr: f4("yr", rows * rs),
        yk: f4("yk", rows * kvs),
        yv: f4("yv", rows * kvs),
        sk: f4("sk", rows * nh),
        dkb: f4("dkb", rows * nkv * 8),
        dvb: f4("dvb", rows * nkv * 8),
        att: f4("att", rows * qs),
        ya: f4("ya", rows * hidden),
        r: f4("r", rows * hidden),
        m: f4("m", rows * hidden),
        mg: f4("mg", rows * hidden),
        xm: f4("xm", rows * hidden),
        lg: f4("lg", rows * e),
        wt: f4("wt", rows * cfg.top_k),
        act: f4("act", rows * cfg.top_k * inter),
        grp: buf(c, "grp", (e + 1 + 2 * rows * cfg.top_k) * 4),
        yp: f4("yp", rows * cfg.top_k * hidden),
        gq: f4("gq", rows * cfg.top_k * inter),
        uq: f4("uq", rows * cfg.top_k * inter),
        y: f4("y", rows * hidden),
        fin: f4("fin", rows * hidden),
        xf: f4("xf", rows * hidden),
        logits: f4("logits", LROWS * vocab),
        staging: c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("inkling-staging"), size: (LROWS * vocab * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
        dummy: buf(c, "dummy", 256),
        dummy_w: buf(c, "dummy-w", 256),
        head,
        plans: HashMap::new(),
    };
    tracing::info!("inkling2_loop on wgpu: {} layers resident, built in {:.1} s", layers.len(), t0.elapsed().as_secs_f64());
    Ok(d)
}

impl Dev {
    /// Room for positions `..need` in every full slot (a device copy); the
    /// plans hold the old buffers, so they go.
    fn ensure(&mut self, c: &Ctx, need: usize, kvs: usize) {
        let mut grew = false;
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("inkling-grow") });
        for s in self.slots.iter_mut().filter(|s| s.ring == 0 && s.cap < need) {
            let cap = need.max(s.cap * 2).next_power_of_two();
            let (k, v) = (buf(c, "kc", cap * kvs * 4), buf(c, "vc", cap * kvs * 4));
            let old = (s.cap * kvs * 4) as u64;
            enc.copy_buffer_to_buffer(&s.k, 0, &k, 0, old);
            enc.copy_buffer_to_buffer(&s.v, 0, &v, 0, old);
            s.k = k;
            s.v = v;
            s.cap = cap;
            grew = true;
        }
        if grew {
            c.queue.submit(Some(enc.finish()));
            self.plans.clear();
        }
    }

    fn zero_rings(&self, c: &Ctx) {
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("inkling-zero") });
        for s in &self.slots {
            for r in &s.rings {
                enc.clear_buffer(r, 0, None);
            }
        }
        c.queue.submit(Some(enc.finish()));
    }
}

/// `true` when this call ran on the device. `ids` from position `pos0`; the
/// last row's logits (`all`: every row's) when `want`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(id: u64, g: &Globals, layers: &[Layer], cfg: &Cfg, ids: &[u32], pos0: usize, want: bool, all: bool, logits: &mut Vec<f32>) -> bool {
    let Some(c) = ctx() else { return false };
    if std::env::var("CMF_INKLING_WGPU").as_deref() == Ok("0") || !admits(layers, cfg) {
        return false;
    }
    if DECLINED.lock().unwrap().contains(&id) {
        return false;
    }
    let mut guard = DEVICES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if !map.contains_key(&id) {
        // one sequence at a time: a new one takes the old one's device
        let old = map.drain().next();
        let built = match old {
            Some((_, d)) => Ok(d),
            None => build(c, g, layers, cfg),
        };
        match built {
            Ok(d) => {
                map.insert(id, d);
            }
            Err(e) => {
                tracing::warn!("inkling2_loop on wgpu declined: {e}; the host runs this sequence");
                DECLINED.lock().unwrap().push(id);
                return false;
            }
        }
    }
    let d = map.get_mut(&id).expect("device");
    if pos0 == 0 {
        d.zero_rings(c);
    }
    d.ensure(c, pos0 + ids.len(), cfg.nkv * cfg.hd);
    if all {
        logits.clear();
    }
    let hidden = cfg.dim;
    let vocab = d.head.rows;
    let mut at = 0usize;
    while at < ids.len() {
        let n = (ids.len() - at).min(if all { LROWS } else { rows_cap() });
        let last = at + n == ids.len();
        let lrows = if all { n } else if want && last { 1 } else { 0 };
        let mut rows = vec![0f32; n * hidden];
        let mut e = vec![0f32; hidden];
        for (i, &tid) in ids[at..at + n].iter().enumerate() {
            e.fill(0.0);
            if (tid as usize) < g.embed.rows() {
                g.embed.row_f32(&g.model, tid as usize, &mut e);
            }
            let ms = e.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
            let r = (1.0 / (ms + cfg.eps).sqrt()) as f32;
            for (o, (&v, &w)) in rows[i * hidden..(i + 1) * hidden].iter_mut().zip(e.iter().zip(&g.embed_norm)) {
                *o = w * (v * r);
            }
        }
        c.queue.write_buffer(&d.h[0], 0, bytemuck::cast_slice(&rows));
        c.queue.write_buffer(&d.cr, 0, bytemuck::cast_slice(&[(pos0 + at) as u32, n as u32, 0, 0]));
        let plan = match d.plans.get(&(n, lrows)) {
            Some(p) => p.clone(),
            None => {
                let p = Arc::new(record(c, d, cfg, n, lrows));
                d.plans.insert((n, lrows), p.clone());
                p
            }
        };
        if prof() {
            profile(c, &plan, n);
        }
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("inkling-call") });
        if !prof() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("inkling"), timestamp_writes: None });
            for st in plan.iter() {
                pass.set_pipeline(&st.pipe);
                pass.set_bind_group(0, &st.bg, &[]);
                pass.dispatch_workgroups(st.x, st.y, st.z);
            }
        }
        if lrows > 0 {
            let mut out = vec![0f32; lrows * vocab];
            if !readback(c, enc, &d.logits, &d.staging, (lrows * vocab * 4) as u64, &mut out) {
                return false;
            }
            if all {
                logits.extend_from_slice(&out);
            } else {
                *logits = out;
            }
        } else {
            c.queue.submit(Some(enc.finish()));
        }
        at += n;
    }
    true
}

/// A chunk's routed experts (`CMF_INKLING_MOE`): `gemm` (the default) groups
/// the (row, pick) entries by expert and runs each expert as a tiled GEMM over
/// its rows; `loop` an expert's rows a matvec at a time; `pick` the decode
/// kernels, a (row, pick) at a time.
fn moe_mode() -> u8 {
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("CMF_INKLING_MOE").as_deref() {
        Ok("pick") => 0,
        Ok("loop") => 1,
        _ => 2,
    })
}

/// `CMF_INKLING_COOP=0`: a prompt chunk's 4-bit dense matrices on the scalar
/// tile GEMM instead of the cooperative-matrix one.
fn coop_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_COOP").as_deref() != Ok("0"))
}

/// `CMF_INKLING_MOE4=0`: the decode's experts a row a subgroup (`ink_moe_gu` /
/// `ink_moe_down`) instead of four rows a workgroup (`ink_moe_gu4` / `ink_moe_dn4`).
fn moe4_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_MOE4").as_deref() != Ok("0"))
}

fn prof() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_PROF").as_deref() == Ok("1"))
}

/// Milliseconds by kernel and calls, by the call's row count (one row: decode).
static PROF: Mutex<Option<HashMap<(bool, &'static str), (f64, u64)>>> = Mutex::new(None);
static PROF_CALLS: Mutex<[u64; 2]> = Mutex::new([0; 2]);

/// `CMF_INKLING_PROF=1`: a call's steps each in a compute pass of its own
/// with GPU timestamps at its ends, all in one submission (the times are the
/// device's own, without a submission per step). Every 32 calls of a kind:
/// the table by kernel in ms a call and their sum — against a plain run's ms
/// a token, the difference is what the dispatches and the barriers between
/// them cost. Without timestamp queries a submission a step, the host timing
/// it.
fn profile(c: &Ctx, plan: &[Step], n: usize) {
    let multi = n > 1;
    let ts = c.device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    if ts {
        const CHUNK: usize = 2000;
        let period = c.queue.get_timestamp_period() as f64;
        for part in plan.chunks(CHUNK) {
            let cnt = (part.len() * 2) as u32;
            let qs = c.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("inkling-prof"), ty: wgpu::QueryType::Timestamp, count: cnt });
            let size = cnt as u64 * 8;
            let rb = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("inkling-prof"), size, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let mb = c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("inkling-prof-map"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("inkling-prof") });
            for (i, st) in part.iter().enumerate() {
                let tw = wgpu::ComputePassTimestampWrites { query_set: &qs, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) };
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("inkling"), timestamp_writes: Some(tw) });
                pass.set_pipeline(&st.pipe);
                pass.set_bind_group(0, &st.bg, &[]);
                pass.dispatch_workgroups(st.x, st.y, st.z);
            }
            enc.resolve_query_set(&qs, 0..cnt, &rb, 0);
            enc.copy_buffer_to_buffer(&rb, 0, &mb, 0, size);
            c.queue.submit(Some(enc.finish()));
            let slice = mb.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            let raw: Vec<u64> = match slice.get_mapped_range() {
                Ok(v) => v.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect(),
                Err(_) => return,
            };
            mb.unmap();
            let mut g = PROF.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            for (i, st) in part.iter().enumerate() {
                let ms = raw[2 * i + 1].saturating_sub(raw[2 * i]) as f64 * period / 1e6;
                let e = m.entry((multi, st.name)).or_insert((0.0, 0));
                e.0 += ms;
                e.1 += 1;
            }
        }
    } else {
        for st in plan {
            let t = std::time::Instant::now();
            let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("inkling-prof") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("inkling"), timestamp_writes: None });
                pass.set_pipeline(&st.pipe);
                pass.set_bind_group(0, &st.bg, &[]);
                pass.dispatch_workgroups(st.x, st.y, st.z);
            }
            c.queue.submit(Some(enc.finish()));
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let mut g = PROF.lock().unwrap();
            let e = g.get_or_insert_with(HashMap::new).entry((multi, st.name)).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += 1;
        }
    }
    let calls = {
        let mut k = PROF_CALLS.lock().unwrap();
        k[multi as usize] += 1;
        k[multi as usize]
    };
    if calls % 32 == 0 {
        let g = PROF.lock().unwrap();
        let mut rows: Vec<(&str, f64, u64)> = g.as_ref().map_or(Vec::new(), |m| m.iter().filter(|((mm, _), _)| *mm == multi).map(|((_, k), (ms, cnt))| (*k, *ms / calls as f64, *cnt / calls)).collect());
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let tot: f64 = rows.iter().map(|r| r.1).sum();
        let steps: u64 = rows.iter().map(|r| r.2).sum();
        let line: Vec<String> = rows.iter().map(|(k, ms, cnt)| format!("{k} {ms:.2} ({cnt})")).collect();
        tracing::info!(
            "inkling2_loop wgpu ms a {} call ({calls} calls, {steps} dispatches, kernels {tot:.1} ms): {}",
            if multi { "prompt" } else { "decode" },
            line.join(", ")
        );
    }
}

/// A new sequence: its plans stay (the buffers are the next one's).
pub(crate) fn reset(id: u64) {
    DECLINED.lock().unwrap().retain(|&x| x != id);
}

/// One dispatch: the kernel, its uniform, the buffers by binding (with byte offsets).
struct Rec<'a> {
    c: &'a Ctx,
    p: &'a Pipes,
    out: Vec<Step>,
    cr: &'a wgpu::Buffer,
    dummy_r: &'a wgpu::Buffer,
    dummy_w: &'a wgpu::Buffer,
}

impl Rec<'_> {
    fn go(&mut self, kernel: &'static str, ip: &IP, binds: &[(u32, &wgpu::Buffer, u64)], grid: (u32, u32, u32)) {
        let pipe = &self.p.by_name[kernel];
        let u = self.c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("inkling-ip"), size: std::mem::size_of::<IP>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.c.queue.write_buffer(&u, 0, bytemuck::bytes_of(ip));
        let mut entries: Vec<wgpu::BindGroupEntry> = vec![wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: self.cr.as_entire_binding() }];
        for b in 2..16u32 {
            let (buf, off) = match binds.iter().find(|x| x.0 == b) {
                Some(&(_, buf, off)) => (buf, off),
                // a read binding and a writable one never share a buffer in a dispatch
                None => (if WRITABLE.contains(&b) { self.dummy_w } else { self.dummy_r }, 0),
            };
            entries.push(wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buf, offset: off, size: None }) });
        }
        let bg = self.c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(kernel), layout: &self.p.layout, entries: &entries });
        self.out.push(Step { name: kernel, pipe: pipe.clone(), bg, x: grid.0.max(1), y: grid.1.max(1), z: grid.2.max(1) });
    }
}

/// `m · x` for `n` rows (`x` rows `cols` apart, `y` rows `rows` apart).
fn mv(c: &'static Ctx, rec: &mut Rec, m: &Mat, xs: &wgpu::Buffer, y: &wgpu::Buffer, n: usize) -> Option<()> {
    // the backend's batch kernels for a prompt chunk, its matvec a row
    if n > 1 {
        // a 4-bit matrix on the cooperative-matrix GEMM (f16 operands, in-kernel
        // dequant) where the device has matrix units
        if let (Codec::Q4, Some(pipe), true) = (&m.codec, c.q4tp_mm_coop.as_ref(), coop_on()) {
            if m.cols % 32 == 0 {
                let u = uniform_u32x4(c, [(m.cols / 4) as u32, m.rows as u32, n as u32, 0]);
                let layout = pipe.get_bind_group_layout(0);
                let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("inkling-coop"), layout: &layout, entries: &[bind_buf(0, &m.buf), bind_buf(1, xs), bind_buf(2, y), bind_buf(3, &u)] });
                rec.out.push(Step { name: "mm (coop)", pipe: pipe.clone(), bg, x: (m.rows as u32).div_ceil(64).min(MAX_WG), y: (n as u32).div_ceil(64), z: 1 });
                return Some(());
            }
        }
        let (pipe, bg, a, b) = match m.codec {
            Codec::Q4 => dense_batch::q4tp_mm_r_prep(c, &m.buf, xs, y, m.rows, m.cols, n)?,
            Codec::Q8 => gemma_batch::q82_mm_prep(c, &m.buf, xs, y, m.rows, m.cols, n)?,
        };
        rec.out.push(Step { name: "mm (rows)", pipe, bg, x: a, y: b, z: 1 });
        return Some(());
    }
    let (pipe, bg, x) = match m.codec {
        Codec::Q4 => dense_mv::sg_prep(c, xs, m.cols, &[(&m.buf, y, m.rows)]).map(|(p, b, g)| (p.clone(), b, g))?,
        Codec::Q8 => moe_r4::q82sg_prep(c, xs, m.cols, &[(&m.buf, y, m.rows)])?,
    };
    rec.out.push(Step { name: "mv", pipe, bg, x, y: 1, z: 1 });
    Some(())
}

fn base_ip(cfg: &Cfg) -> IP {
    IP {
        h: cfg.dim as u32,
        nh: cfg.nh as u32,
        nkv: cfg.nkv as u32,
        hd: cfg.hd as u32,
        dr: cfg.d_rel as u32,
        kc: 4,
        bstride: (MAXB * cfg.dim) as u32,
        ostride: (rows_cap() * cfg.dim) as u32,
        topk: cfg.top_k as u32,
        eps: cfg.eps as f32,
        qclip: cfg.qk_clip,
        rclip: cfg.rel_clip,
        alpha: cfg.log_alpha as f32,
        floor_: cfg.log_floor as f32,
        scale: 1.0 / (cfg.hd as f32).sqrt(),
        cap: cfg.gate_cap,
        ucap: cfg.up_cap,
        rscale: cfg.route_scale,
        bslot: u32::MAX,
        ..IP::default()
    }
}

/// The plan of one call of `n` rows: the 48 applications, the final mixture,
/// the head over the last `lrows` rows.
fn record(c: &'static Ctx, d: &Dev, cfg: &Cfg, n: usize, lrows: usize) -> Vec<Step> {
    let p = pipes(c).expect("shaders");
    let mut rec = Rec { c, p, out: Vec::with_capacity(48 * 40), cr: &d.cr, dummy_r: &d.dummy, dummy_w: &d.dummy_w };
    let hidden = cfg.dim;
    let base = base_ip(cfg);
    let nu = n as u32;
    let plane = (rows_cap() * hidden * 4) as u64;
    let copy = |rec: &mut Rec, src: &wgpu::Buffer, dst: &wgpu::Buffer, floats: usize| {
        let ip = IP { rows: floats as u32, ..base };
        rec.go("ink_copy", &ip, &[(2, src, 0), (7, dst, 0)], ((floats as u32).div_ceil(256), 1, 1));
    };
    let mut cur = 0usize;
    let mut nb = 0usize;
    let mut npre = 0usize;
    let mut hin: &wgpu::Buffer = &d.h[0];
    for (li, pass) in cfg.plan() {
        let l = &d.layers[li];
        let s = &d.slots[cfg.slot(li, pass)];
        if li == cfg.core_start {
            if pass == 0 {
                copy(&mut rec, hin, &d.pre_h, n * hidden);
                npre = nb;
            } else {
                copy(&mut rec, &d.blk, &d.bank, n * MAXB * hidden);
                copy(&mut rec, hin, &d.bank_out, n * hidden);
                hin = &d.pre_h;
                nb = npre;
            }
        }
        let hout = &d.h[1 - cur];
        // 1. the attention inputs
        let mut inp: [(&wgpu::Buffer, u64); 3] = [(hin, 0); 3];
        if let Some((aq, nq)) = &l.aq {
            let ip = IP { nb: nb as u32, ..base };
            rec.go("ink_mix", &ip, &[(2, &d.blk, 0), (3, hin, 0), (4, aq, 0), (7, &d.in3, 0)], (*nq as u32, nu, 1));
            inp = if *nq == 3 { [(&d.in3, 0), (&d.in3, plane), (&d.in3, 2 * plane)] } else { [(&d.in3, 0); 3] };
        }
        // 2. the messages of a later pass, gated into q, k, v
        let gates = if pass > 0 { l.gates.as_ref() } else { None };
        if let Some(gs) = gates {
            let i = li - cfg.core_start;
            let ip = IP { nb: (cfg.loop_len() / cfg.block - 1) as u32, ..base };
            rec.go("ink_mix", &ip, &[(2, &d.bank, ((npre + 1) * hidden * 4) as u64), (3, &d.bank_out, 0), (4, &d.msg_q, (i * 4 * hidden * 4) as u64), (7, &d.msg, 0)], (4, nu, 1));
            for j in 0..3 {
                gate(&mut rec, &base, &gs[j], inp[j], (&d.msg, j as u64 * plane), &d.gd, (&d.gin, j as u64 * plane), n, hidden);
                inp[j] = (&d.gin, j as u64 * plane);
            }
        }
        // 3. the input norms
        let one = l.sliding && gates.is_none();
        for j in 0..(if one { 1 } else { 3 }) {
            rec.go("ink_rms", &base, &[(2, inp[j].0, inp[j].1), (3, &l.norms[j], 0), (7, &d.x[j], 0)], (nu, 1, 1));
        }
        let (xk, xv) = if one { (0, 0) } else { (1, 2) };
        // 4. the projections
        for (m, x, y) in [(&l.q, &d.x[0], &d.yq), (&l.gate, &d.x[0], &d.yg), (&l.rel, &d.x[0], &d.yr), (&l.k, &d.x[xk], &d.yk), (&l.v, &d.x[xv], &d.yv)] {
            if mv(c, &mut rec, m, x, y, n).is_none() {
                tracing::error!("inkling2_loop on wgpu: no matvec kernel for a {}x{} matrix", m.rows, m.cols);
            }
        }
        let fmv = |rec: &mut Rec, w: &wgpu::Buffer, rows: usize, x: &wgpu::Buffer, y: &wgpu::Buffer| {
            let ip = IP { rows: rows as u32, ..base };
            rec.go("ink_fmv", &ip, &[(2, w, 0), (3, x, 0), (7, y, 0)], ((rows as u32).div_ceil(8), nu, 1));
        };
        fmv(&mut rec, &l.sink, cfg.nh, &d.x[0], &d.sk);
        let full = l.dk.is_some();
        if let (Some(wk), Some(wv)) = (&l.dk, &l.dv) {
            fmv(&mut rec, wk, cfg.nkv * 4, &d.x[xk], &d.dkb);
            fmv(&mut rec, wv, cfg.nkv * 4, &d.x[xv], &d.dvb);
        }
        // 5. queries and relative states; keys and values into the cache
        let ip = IP { full: (!l.sliding) as u32, ..base };
        rec.go("ink_qr", &ip, &[(2, &l.q_scale, 0), (3, &l.rel_scale, 0), (7, &d.yq, 0), (8, &d.yr, 0)], (cfg.nh as u32, nu, 1));
        let ip = IP { full: full as u32, ring: s.ring as u32, ..base };
        rec.go(
            "ink_kv",
            &ip,
            &[(2, &d.yk, 0), (3, &d.yv, 0), (4, if full { &d.dkb } else { &d.dummy }, 0), (5, if full { &d.dvb } else { &d.dummy }, 0), (6, &l.taps, 0), (7, &s.rings[0], 0), (8, &s.rings[1], 0), (9, &s.k, 0), (10, &s.v, 0)],
            (cfg.nkv as u32, 2, 1),
        );
        // 6. the attention, its gate, the O projection
        let ip = IP { ring: s.ring as u32, win: if s.ring > 0 { cfg.window as u32 } else { 0 }, e: l.extent as u32, ..base };
        rec.go("ink_attn", &ip, &[(2, &d.yq, 0), (3, &d.yr, 0), (4, &d.sk, 0), (5, &l.rel_table, 0), (6, &s.k, 0), (7, &d.att, 0), (8, &s.v, 0)], (cfg.nh as u32, nu, 1));
        let qs = cfg.nh * cfg.hd;
        rec.go("ink_ogate", &base, &[(2, &d.yg, 0), (7, &d.att, 0)], ((qs as u32).div_ceil(256), nu, 1));
        let _ = mv(c, &mut rec, &l.o, &d.att, &d.ya, n);
        // 7. the output convolution; a boundary closes the running sum into a block
        let ip = IP { bslot: if l.boundary { nb as u32 } else { u32::MAX }, ..base };
        rec.go("ink_post", &ip, &[(2, &d.ya, 0), (3, &l.tpa, 0), (4, hin, 0), (7, &s.rings[2], 0), (8, &d.blk, 0), (9, &d.r, 0)], ((hidden as u32).div_ceil(256), 1, 1));
        if l.boundary {
            nb += 1;
        }
        // 8. the MoE input
        let ip = IP { nb: nb as u32, ..base };
        rec.go("ink_mix", &ip, &[(2, &d.blk, 0), (3, &d.r, 0), (4, &l.mq, 0), (7, &d.m, 0)], (1, nu, 1));
        let mut mi: (&wgpu::Buffer, u64) = (&d.m, 0);
        if let Some(gs) = gates {
            gate(&mut rec, &base, &gs[3], mi, (&d.msg, 3 * plane), &d.gd, (&d.mg, 0), n, hidden);
            mi = (&d.mg, 0);
        }
        rec.go("ink_rms", &base, &[(2, mi.0, mi.1), (3, &l.post_norm, 0), (7, &d.xm, 0)], (nu, 1, 1));
        // 9. the routed experts into y
        let ip = IP { e: l.e as u32, inter: l.inter as u32, ..base };
        rec.go("ink_router", &ip, &[(2, &l.router, 0), (3, &d.xm, 0), (7, &d.lg, 0)], ((l.e as u32).div_ceil(8), nu, 1));
        // the picks land behind the expert offsets in the layer's table
        rec.go("ink_select", &ip, &[(2, &d.lg, 0), (3, &l.bias, 0), (7, &d.wt, 0), (13, &l.table, (l.e * 3 * 4) as u64)], (nu, 1, 1));
        if n > 1 && moe_mode() == 2 {
            // the entries by expert, each expert's rows a tiled GEMM against its weights
            let ipg = IP { rows: nu, ..ip };
            rec.go("ink_group", &ipg, &[(12, &l.table, (l.e * 3 * 4) as u64), (13, &d.grp, 0)], (1, 1, 1));
            let tiles = nu.div_ceil(32);
            for (which, x, y, out) in [(0u32, &d.xm, &d.gq, l.inter), (1, &d.xm, &d.uq, l.inter)] {
                let ipm = IP { v: which, ..ipg };
                rec.go("ink_moe_mm", &ipm, &[(4, x, 0), (7, y, 0), (11, &l.bank, 0), (12, &l.table, 0), (13, &d.grp, 0)], (tiles, (out as u32).div_ceil(64), l.e as u32));
            }
            let nk = nu * cfg.top_k as u32;
            rec.go("ink_moe_act", &ipg, &[(2, &d.gq, 0), (3, &d.uq, 0), (7, &d.act, 0)], ((l.inter as u32).div_ceil(256), nk, 1));
            let ipm = IP { v: 2, ..ipg };
            rec.go("ink_moe_mm", &ipm, &[(4, &d.act, 0), (7, &d.yp, 0), (11, &l.bank, 0), (12, &l.table, 0), (13, &d.grp, 0)], (tiles, (hidden as u32).div_ceil(64), l.e as u32));
            rec.go("ink_moe_mixq", &ipg, &[(2, &d.yp, 0), (3, &d.wt, 0), (7, &d.y, 0), (12, &d.grp, 0)], ((hidden as u32).div_ceil(256), nu, 1));
        } else if n > 1 && moe_mode() == 1 {
            // the entries by expert, an expert's rows against its weights once
            let ipg = IP { rows: nu, ..ip };
            rec.go("ink_group", &ipg, &[(12, &l.table, (l.e * 3 * 4) as u64), (13, &d.grp, 0)], (1, 1, 1));
            rec.go("ink_moe_gu_g", &ipg, &[(4, &d.xm, 0), (7, &d.act, 0), (11, &l.bank, 0), (12, &l.table, 0), (13, &d.grp, 0)], ((l.inter as u32).div_ceil(8), l.e as u32, 1));
            rec.go("ink_moe_down_g", &ipg, &[(5, &d.act, 0), (7, &d.yp, 0), (11, &l.bank, 0), (12, &l.table, 0), (13, &d.grp, 0)], ((hidden as u32).div_ceil(8), l.e as u32, 1));
            rec.go("ink_moe_mix", &ipg, &[(2, &d.yp, 0), (3, &d.wt, 0), (7, &d.y, 0)], ((hidden as u32).div_ceil(256), nu, 1));
        } else if moe4_on() && l.inter % 32 == 0 && hidden % 32 == 0 {
            rec.go("ink_moe_gu4", &ip, &[(7, &d.act, 0), (11, &l.bank, 0), (12, &l.table, 0), (14, &l.bank, 0), (15, &d.xm, 0)], ((l.inter as u32) / 4, cfg.top_k as u32, nu));
            rec.go("ink_moe_dn4", &ip, &[(6, &d.wt, 0), (7, &d.y, 0), (11, &l.bank, 0), (12, &l.table, 0), (14, &l.bank, 0), (15, &d.act, 0)], ((hidden as u32) / 4, nu, 1));
        } else {
            rec.go("ink_moe_gu", &ip, &[(4, &d.xm, 0), (7, &d.act, 0), (11, &l.bank, 0), (12, &l.table, 0)], ((l.inter as u32).div_ceil(8), cfg.top_k as u32, nu));
            rec.go("ink_moe_down", &ip, &[(5, &d.act, 0), (6, &d.wt, 0), (7, &d.y, 0), (11, &l.bank, 0), (12, &l.table, 0)], ((hidden as u32).div_ceil(8), nu, 1));
        }
        // 10. the output convolution onto the running sum
        rec.go("ink_post", &base, &[(2, &d.y, 0), (3, &l.tpm, 0), (4, &d.r, 0), (7, &s.rings[3], 0), (8, &d.dummy_w, 0), (9, hout, 0)], ((hidden as u32).div_ceil(256), 1, 1));
        cur = 1 - cur;
        hin = hout;
    }
    // the final mixture, its norm and the head
    let ip = IP { nb: nb as u32, ..base };
    rec.go("ink_mix", &ip, &[(2, &d.blk, 0), (3, hin, 0), (4, &d.final_q, 0), (7, &d.fin, 0)], (1, nu, 1));
    if lrows > 0 {
        let r0 = n - lrows;
        rec.go("ink_rms", &base, &[(2, &d.fin, (r0 * hidden * 4) as u64), (3, &d.final_norm, 0), (7, &d.xf, 0)], (lrows as u32, 1, 1));
        let _ = mv(c, &mut rec, &d.head, &d.xf, &d.logits, lrows);
        if cfg.logit_cap > 0.0 {
            let ip = IP { v: d.head.rows as u32, cap: cfg.logit_cap, ..base };
            rec.go("ink_softcap", &ip, &[(7, &d.logits, 0)], ((d.head.rows as u32).div_ceil(256), lrows as u32, 1));
        }
    }
    rec.out
}

/// A message gate.
#[allow(clippy::too_many_arguments)]
fn gate(rec: &mut Rec, base: &IP, g: &(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, usize), x: (&wgpu::Buffer, u64), m: (&wgpu::Buffer, u64), dbuf: &wgpu::Buffer, out: (&wgpu::Buffer, u64), n: usize, hidden: usize) {
    let ip = IP { r: g.3 as u32, ..*base };
    rec.go("ink_gate_down", &ip, &[(2, x.0, x.1), (3, m.0, m.1), (4, &g.0, 0), (7, dbuf, 0)], (g.3 as u32, n as u32, 1));
    rec.go("ink_gate_out", &ip, &[(2, x.0, x.1), (3, m.0, m.1), (4, dbuf, 0), (5, &g.1, 0), (6, &g.2, 0), (7, out.0, out.1)], ((hidden as u32).div_ceil(8), n as u32, 1));
}

#[cfg(test)]
mod tests {
    /// The arm's shader (every entry point, the decode's four-row expert
    /// kernels with the bank's vec4 view among them) parses and validates
    /// without a device.
    #[test]
    fn inkling_wgsl_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(super::SRC).unwrap_or_else(|e| panic!("inkling WGSL must parse: {}", e.emit_to_string(super::SRC)));
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("inkling WGSL must validate: {}", e.emit_to_string(super::SRC)));
        for k in super::KERNELS {
            assert!(module.entry_points.iter().any(|e| e.name == k), "{k}");
        }
    }
}
