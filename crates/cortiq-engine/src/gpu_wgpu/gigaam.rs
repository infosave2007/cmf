//! The GigaAM encoder resident on wgpu (`gpu::gigaam_encode`): the three
//! Zapformer stacks of one segment with the activations on the card from the
//! first projection to the output norm — one upload, ONE submit, one readback.
//!
//! The weights go up once a model (`key`) as f32 planes, one buffer a weight.
//! The products are the backend's NT GEMM over resident buffers: `gemm_nt_coop`
//! on the matrix units (f16 operands, an f32 accumulator), `gemm_nt_f32t` (the
//! plain-f32 tile kernel) where the device has none or under
//! `CMF_GIGAAM_GEMM=f32`. Attention is the DiT's kernels over head-major
//! planes: every head of a layer in one dispatch each on the shader cores
//! (`dit_qk` → `dit_softmax` → `dit_pv`), or — with matrix units, for stacks of
//! more than `CMF_GIGAAM_ATTN_ROWS` (384) rows — a head a dispatch on them
//! (`dit_gemm_coop`, v transposed first by `dit_v_transpose`).
//! `CMF_GIGAAM_ATTN=scalar` keeps the shader cores, `=ref` runs the per-head
//! plan through a plain kernel of the same contract (a check of that plan on a
//! device without matrix units, slow). Everything between is the kernels of
//! this file: the CUDA port's (`gpu_cuda/gigaam.cu`) with the biases and the
//! additions folded into their neighbours.
//!
//! A segment is a PLAN: the list of (pipeline, bind group, grid) built once a
//! model and replayed into one compute pass. What changes with the segment —
//! the row count of each stack — sits in one uniform buffer written once a
//! segment (a 256-byte slot an op shape), so no buffer or bind group is made
//! on the hot path. `CMF_PASS_MERGE=0` gives every dispatch its own pass.
//!
//! Short segments: a product over one or two rows of tiles is bound by the K
//! loop of its workgroups, not by the device (Apple M4: 61 ms of products for
//! 50 frames, 189 ms for 750). The f32 tile kernel can split K over workgroups
//! (`kspan`), so there a stack of up to `CMF_GIGAAM_KSPLIT` / 2 tile rows
//! (default 4: 128 rows) runs its products as K slices into planes and sums
//! them (`ga_fold`): 61 → 34 ms. The matrix-unit kernel has no such switch.
//!
//! `CMF_GIGAAM_PROF=1` prints a line a segment (milliseconds, dispatches);
//! with `CMF_GIGAAM_STAGES=1` the segment is then replayed one kind of step a
//! submit — products, attention, the rest — and each is timed (diagnostics).
//!
//! The front end (`gpu::gigaam_encode_feats`) is a second plan in front of the
//! stacks', in the same pass: the log-mel rows go up, the six convolutions run
//! as direct sums (`ga_conv2d`, an invocation an output, plain f32) over
//! `[channel][frame][bin]` volumes, the last one lays each frame's channels
//! flat, and the projection — a product like any other — leaves the stacks'
//! input where a host upload would have put it.
//!
//! The voice-activity model (`gpu::gigaam_vad`) is a plan of its own over
//! `windows · steps` rows, attention inside a window: the windows' heads are
//! the DiT kernels' heads (`dit_qk` → `dit_softmax` → `dit_pv`, every head of
//! every window in one dispatch each). Plain f32 everywhere — the products are
//! `gemm_nt_f32t` whatever the device has. One submit a call; a batch whose
//! buffers or grids pass the device's limits (or `CMF_GIGAAM_VAD_WINDOWS`
//! windows) runs as equal chunks of windows inside that submit.

use super::{Ctx, ctx};
use crate::gpu::{GaFront, GaLin, GaVadArgs, GigaamArgs};
use std::sync::{Mutex, OnceLock};

/// `P`: an op's own numbers, fixed for the model. `R`: the rows of the op in
/// this call — `n` in all, `r1` a window (an encoder segment is one window),
/// `r2` a window before the op's stride of two, `r3` the windows.
const PRE: &str = r#"
struct P { a: u32, b: u32, c: u32, d: u32, x: f32, y: f32, z: f32, w: f32 }
struct R { n: u32, r1: u32, r2: u32, r3: u32 }
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(v, -80.0, 80.0))); }
"#;

// y += b a row of 4·p.a lanes; p.b = 1: SiLU after, 2: ReLU.
const ACT_SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bb: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_bias_act(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= r.n * p.a) { return; }
    var v = y[i] + bb[i % p.a];
    if (p.b == 1u) { v = v / (1.0 + exp(-clamp(v, vec4<f32>(-80.0), vec4<f32>(80.0)))); }
    if (p.b == 2u) { v = max(v, vec4<f32>(0.0)); }
    y[i] = v;
}
"#;

// dst = a + y + b a row of 4·p.a lanes.
const ADD3_SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> y: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bb: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_add3(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= r.n * p.a) { return; }
    dst[i] = a[i] + y[i] + bb[i % p.a];
}
"#;

// dst += y + b over r.n rows of 4·p.a lanes; the bias repeats every 4·p.b
// lanes (a stack's way out: frame rows, the bias a folded row).
const ACC_SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> y: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bb: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_acc(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= r.n * p.a) { return; }
    dst[i] = dst[i] + y[i] + bb[i % p.b];
}
"#;

// dst = the sum of r.r1 planes of part, each r.n rows of 4·p.a lanes (a
// product whose K was split over workgroups, `Dyn::MmK`).
const FOLD_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> part: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_fold(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let plane = r.n * p.a;
    if (i >= plane) { return; }
    var acc = part[i];
    for (var z = 1u; z < r.r1; z = z + 1u) { acc = acc + part[z * plane + i]; }
    dst[i] = acc;
}
"#;

// A stack's way in, after its projection: v = h + b; bypass = p.x·v; h = p.y·v.
const FORK_SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> h: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bb: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> byp: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_fork(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= r.n * p.a) { return; }
    let v = h[i] + bb[i % p.a];
    byp[i] = p.x * v;
    h[i] = p.y * v;
}
"#;

// dst = src · rsqrt(mean(src²) + p.x) · w a row of p.a lanes, a workgroup a row.
const RMS_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> r: R;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn ga_rms(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wid.x + wid.y * 65535u;
    if (row >= r.n) { return; }
    let d = p.a;
    let base = row * d;
    var acc = 0.0;
    for (var i = t; i < d; i = i + 256u) { let v = src[base + i]; acc = acc + v * v; }
    red[t] = acc;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(d) + p.x);
    for (var i = t; i < d; i = i + 256u) { dst[base + i] = src[base + i] * inv * w[i]; }
}
"#;

// A layer's way out, a workgroup a row: with v = h + p.y·((s + y + b) − h)
// (the last addition of the stream folded in), h = v · rsqrt(mean(v²) + p.x) · w
// and bypass += p.z · h.
const TAIL_SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> bb: array<f32>;
@group(0) @binding(4) var<storage, read> w: array<f32>;
@group(0) @binding(5) var<storage, read_write> byp: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@group(0) @binding(7) var<uniform> r: R;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn ga_tail(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wid.x + wid.y * 65535u;
    if (row >= r.n) { return; }
    let d = p.a;
    let base = row * d;
    var acc = 0.0;
    for (var i = t; i < d; i = i + 256u) {
        let hh = h[base + i];
        let v = hh + p.y * (s[base + i] + y[base + i] + bb[i] - hh);
        acc = acc + v * v;
    }
    red[t] = acc;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(d) + p.x);
    // a lane is read and written by its own thread alone
    for (var i = t; i < d; i = i + 256u) {
        let hh = h[base + i];
        let v = hh + p.y * (s[base + i] + y[base + i] + bb[i] - hh);
        let o = v * inv * w[i];
        h[base + i] = o;
        byp[base + i] = byp[base + i] + p.z * o;
    }
}
"#;

// qk rows [n][2·hw] and vg rows [n][3·hw] (their biases in ab: qk's, then
// vg's) into head-major planes [heads][n][hd]: q and k rotated in neighbour
// pairs by the frame's cosines and sines (cs: the cosines [frame][hd / 2],
// the sines from p.c on), v multiplied by the sigmoid of its input gate (the
// second third of vg). p.a = heads, p.b = hd.
const SPLIT_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> qk: array<f32>;
@group(0) @binding(1) var<storage, read> vg: array<f32>;
@group(0) @binding(2) var<storage, read> ab: array<f32>;
@group(0) @binding(3) var<storage, read> cs: array<f32>;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> k: array<f32>;
@group(0) @binding(6) var<storage, read_write> v: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
@group(0) @binding(8) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_split(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let n = r.n;
    let hd = p.b;
    let half = hd / 2u;
    let hw = p.a * hd;
    if (i >= p.a * n * half) { return; }
    let pr = i % half;
    let row = (i / half) % n;
    let hh = i / (half * n);
    let c = cs[row * half + pr];
    let s = cs[p.c + row * half + pr];
    let col = hh * hd + 2u * pr;
    let dst = (hh * n + row) * hd + 2u * pr;
    let qb = row * 2u * hw + col;
    let q0 = qk[qb] + ab[col];
    let q1 = qk[qb + 1u] + ab[col + 1u];
    let k0 = qk[qb + hw] + ab[hw + col];
    let k1 = qk[qb + hw + 1u] + ab[hw + col + 1u];
    q[dst] = q0 * c - q1 * s;
    q[dst + 1u] = q0 * s + q1 * c;
    k[dst] = k0 * c - k1 * s;
    k[dst + 1u] = k0 * s + k1 * c;
    let vb = row * 3u * hw + col;
    let bv = 2u * hw + col;
    v[dst] = (vg[vb] + ab[bv]) * sigm(vg[vb + hw] + ab[bv + hw]);
    v[dst + 1u] = (vg[vb + 1u] + ab[bv + 1u]) * sigm(vg[vb + hw + 1u] + ab[bv + hw + 1u]);
}
"#;

// The attention context from head-major planes [heads][n][hd] to rows
// [n][hw], times the sigmoid of the output gate (the last third of vg).
const GATE_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> o: array<f32>;
@group(0) @binding(1) var<storage, read> vg: array<f32>;
@group(0) @binding(2) var<storage, read> ab: array<f32>;
@group(0) @binding(3) var<storage, read_write> cx: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_gate(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let hd = p.b;
    let hw = p.a * hd;
    if (i >= r.n * hw) { return; }
    let col = i % hw;
    let row = i / hw;
    let hh = col / hd;
    let e = col % hd;
    cx[i] = o[(hh * r.n + row) * hd + e] * sigm(vg[row * 3u * hw + 2u * hw + col] + ab[4u * hw + col]);
}
"#;

// The convolution module's constants in one buffer cw: the taps [kernel][d],
// the convolution's bias [d], the mean's weights [d]. cg rows are 2·d wide
// (the stream, then its gate), bi their product's bias. p.a = d, p.b = kernel.
//
// av[c] = mean_w[c] · mean over frames of the stream, a workgroup a channel.
const MEAN_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> cg: array<f32>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(2) var<storage, read> bi: array<f32>;
@group(0) @binding(3) var<storage, read_write> av: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<uniform> r: R;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn ga_mean(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let c = wid.x;
    let d = p.a;
    if (c >= d) { return; }
    var acc = 0.0;
    for (var i = t; i < r.n; i = i + 256u) { acc = acc + cg[i * 2u * d + c]; }
    red[t] = acc;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    if (t == 0u) { av[c] = (red[0] / f32(r.n) + bi[c]) * cw[(p.b + 1u) * d + c]; }
}
"#;

// z[frame][c] = silu((conv + av[c]) · σ(gate)): the depthwise convolution over
// time, (kernel − 1) / 2 zero frames on both sides.
const CONV_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> cg: array<f32>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(2) var<storage, read> bi: array<f32>;
@group(0) @binding(3) var<storage, read> av: array<f32>;
@group(0) @binding(4) var<storage, read_write> z: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@group(0) @binding(6) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_conv(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let d = p.a;
    let kn = p.b;
    if (i >= r.n * d) { return; }
    let c = i % d;
    let row = i / d;
    let half = (kn - 1u) / 2u;
    var acc = cw[kn * d + c];
    for (var j = 0u; j < kn; j = j + 1u) {
        let at = row + j;
        if (at >= half && at - half < r.n) { acc = acc + cw[j * d + c] * (cg[(at - half) * 2u * d + c] + bi[c]); }
    }
    let v = (acc + av[c]) * sigm(cg[row * 2u * d + d + c] + bi[d + c]);
    z[i] = v * sigm(v);
}
"#;

// `dit_gemm_coop`'s contract in plain f32, a 64×64 tile a workgroup:
// y[c_off + m·rows + n] = scale · Σ x[a_off + m·cols + k] · w[b_off + n·cols + k].
// Only `CMF_GIGAAM_ATTN=ref` runs it: the per-head plan of the matrix units on a
// device without them.
const NTREF_SRC: &str = r#"
struct DcP { cols4: u32, rows: u32, nb: u32, scale: f32, a_off: u32, b_off: u32, c_off: u32, kk: u32 };
@group(0) @binding(0) var<storage, read> dcw: array<f32>;
@group(0) @binding(1) var<storage, read> dcx: array<f32>;
@group(0) @binding(2) var<storage, read_write> dcy: array<f32>;
@group(0) @binding(3) var<uniform> dcp: DcP;
@compute @workgroup_size(8, 8)
fn ga_nt_ref(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = select(dcp.cols4 * 4u, dcp.kk, dcp.kk > 0u);
    for (var e = 0u; e < 64u; e = e + 1u) {
        let m = wid.y * 64u + lid.y * 8u + e / 8u;
        let n = wid.x * 64u + lid.x * 8u + e % 8u;
        if (m >= dcp.nb || n >= dcp.rows) { continue; }
        var acc = 0.0;
        for (var kx = 0u; kx < cols; kx = kx + 1u) { acc = acc + dcx[dcp.a_off + m * cols + kx] * dcw[dcp.b_off + n * cols + kx]; }
        dcy[dcp.c_off + m * dcp.rows + n] = acc * dcp.scale;
    }
}
"#;

// ---- the front end ------------------------------------------------------

// A k×k convolution (k 1 or 3) of x [ci][t][f] into [co][to][fo], an
// invocation an output: `pt` zero frames before the first (frames after the
// last are zeros too), `pf` zero bins on both sides (pad = pt | pf << 16),
// stride `st` on both axes — at stride 2 the input has r.r2 frames, otherwise
// r.r1 like the output. w is [co][ci][k][k]; flag 1: depthwise, [co][1][k][k].
// Flag 2: SiLU after the bias. Flag 4: the result is added to ad
// ([co][to][fo]). Flag 8: the output is laid as rows [to][co · fo] (a frame's
// channels flat, for the projection).
const CONV2D_SRC: &str = r#"
struct CP { ci: u32, co: u32, f: u32, fo: u32, k: u32, st: u32, pad: u32, flags: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bb: array<f32>;
@group(0) @binding(3) var<storage, read> ad: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: CP;
@group(0) @binding(6) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_conv2d(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let to = r.r1;
    let t = select(r.r1, r.r2, p.st == 2u);
    if (i >= p.co * to * p.fo) { return; }
    let ff = i % p.fo;
    let tt = (i / p.fo) % to;
    let c = i / (p.fo * to);
    let pt = p.pad & 0xffffu;
    let pf = p.pad >> 16u;
    let kk = p.k * p.k;
    // the channels read, and the first of the weight's kernels for them
    var c0 = 0u;
    var c1 = p.ci;
    var wb = c * p.ci * kk;
    if ((p.flags & 1u) != 0u) { c0 = c; c1 = c + 1u; wb = c * kk; }
    let plane = t * p.f;
    var acc = bb[c];
    for (var kt = 0u; kt < p.k; kt = kt + 1u) {
        let at = tt * p.st + kt;
        if (at < pt || at - pt >= t) { continue; }
        for (var kf = 0u; kf < p.k; kf = kf + 1u) {
            let af = ff * p.st + kf;
            if (af < pf || af - pf >= p.f) { continue; }
            let xo = (at - pt) * p.f + af - pf;
            let wo = wb + kt * p.k + kf;
            for (var cc = c0; cc < c1; cc = cc + 1u) { acc = acc + w[wo + (cc - c0) * kk] * x[cc * plane + xo]; }
        }
    }
    if ((p.flags & 2u) != 0u) { acc = acc * sigm(acc); }
    if ((p.flags & 4u) != 0u) { acc = acc + ad[i]; }
    if ((p.flags & 8u) != 0u) { y[(tt * p.co + c) * p.fo + ff] = acc; } else { y[i] = acc; }
}
"#;

// ---- the voice-activity model -------------------------------------------

// Patches of a kernel-3 stride-2 convolution over time, one zero frame on
// both sides of a window: x [windows · r.r2][p.a] to y [windows · r.r1][3][p.a].
const PATCH_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_patch1d(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let ci = p.a;
    if (i >= r.n * 3u * ci) { return; }
    let c = i % ci;
    let k = (i / ci) % 3u;
    let row = i / (ci * 3u);
    let t = row % r.r1;
    let wi = row / r.r1;
    let s = 2u * t + k;
    var v = 0.0;
    if (s >= 1u && s - 1u < r.r2) { v = x[(wi * r.r2 + s - 1u) * ci + c]; }
    y[i] = v;
}
"#;

// LayerNorm a row of p.a lanes, a workgroup a row: dst = (src − mean) ·
// rsqrt(var + p.x) · w + b, nb the weight and then the bias; p.b = 1: SiLU after.
const LN_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> nb: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> r: R;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn ga_ln(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wid.x + wid.y * 65535u;
    if (row >= r.n) { return; }
    let d = p.a;
    let base = row * d;
    var acc = 0.0;
    for (var i = t; i < d; i = i + 256u) { acc = acc + src[base + i]; }
    red[t] = acc;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let mean = red[0] / f32(d);
    workgroupBarrier();
    acc = 0.0;
    for (var i = t; i < d; i = i + 256u) { let v = src[base + i] - mean; acc = acc + v * v; }
    red[t] = acc;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(d) + p.x);
    for (var i = t; i < d; i = i + 256u) {
        var v = (src[base + i] - mean) * inv * nb[i] + nb[d + i];
        if (p.b == 1u) { v = v * sigm(v); }
        dst[base + i] = v;
    }
}
"#;

// Rows [n][heads · hd] rotated by the angles of the row's step in its window
// (cs: the cosines [step][hd / 2], the sines from p.c on): lane j of a head
// with lane j + hd / 2. p.a = heads, p.b = hd.
const ROT_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> cs: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_rot_half(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let hd = p.b;
    let half = hd / 2u;
    if (i >= r.n * p.a * half) { return; }
    let j = i % half;
    let hh = (i / half) % p.a;
    let row = i / (half * p.a);
    let pos = row % r.r1;
    let c = cs[pos * half + j];
    let s = cs[p.c + pos * half + j];
    let at = (row * p.a + hh) * hd + j;
    let lo = x[at];
    let hi = x[at + half];
    y[at] = lo * c - hi * s;
    y[at + half] = hi * c + lo * s;
}
"#;

// Rows [windows · win][heads · hd] plus their bias to planes
// [windows][heads][win][hd]; p.c = 1: planes back to rows (no bias).
// p.a = heads, p.b = hd.
const PLANES_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> bb: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_planes(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let hd = p.b;
    let d = p.a * hd;
    if (i >= r.n * d) { return; }
    let col = i % d;
    let row = i / d;
    let t = row % r.r1;
    let wi = row / r.r1;
    let plane = ((wi * p.a + col / hd) * r.r1 + t) * hd + col % hd;
    if (p.c == 1u) { dst[i] = src[plane]; } else { dst[plane] = src[i] + bb[col]; }
}
"#;

// The convolution module's middle: u rows are [n][2·d] (values, then gates;
// bi their product's bias), g = value · σ(gate), and y[row][c] = bias[c] +
// Σ taps[j][c] · g of the row j − (kernel − 1) / 2 away inside the same window
// (zero outside). cw: the taps [kernel][d], then the bias. p.a = d, p.b = kernel.
const GLU_SRC: &str = r#"
@group(0) @binding(0) var<storage, read> u: array<f32>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(2) var<storage, read> bi: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_glu_conv(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    let d = p.a;
    let kn = p.b;
    if (i >= r.n * d) { return; }
    let c = i % d;
    let row = i / d;
    let t = row % r.r1;
    let half = (kn - 1u) / 2u;
    var acc = cw[kn * d + c];
    for (var j = 0u; j < kn; j = j + 1u) {
        let at = t + j;
        if (at >= half && at - half < r.r1) {
            let ub = (row - t + at - half) * 2u * d + c;
            acc = acc + cw[j * d + c] * (u[ub] + bi[c]) * sigm(u[ub + d] + bi[d + c]);
        }
    }
    y[i] = acc;
}
"#;

// dst += p.x · (y + b) a row of 4·p.a lanes.
const AXPB_SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> y: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bb: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<uniform> r: R;
@compute @workgroup_size(256)
fn ga_axpb(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= r.n * p.a) { return; }
    dst[i] = dst[i] + p.x * (y[i] + bb[i % p.a]);
}
"#;

struct Pipes {
    act: wgpu::ComputePipeline,
    add3: wgpu::ComputePipeline,
    acc: wgpu::ComputePipeline,
    fork: wgpu::ComputePipeline,
    fold: wgpu::ComputePipeline,
    rms: wgpu::ComputePipeline,
    tail: wgpu::ComputePipeline,
    split: wgpu::ComputePipeline,
    gate: wgpu::ComputePipeline,
    mean: wgpu::ComputePipeline,
    conv: wgpu::ComputePipeline,
    nt_ref: wgpu::ComputePipeline,
}

/// The front end's kernel.
struct FrontPipes {
    conv2d: wgpu::ComputePipeline,
}

/// The voice-activity model's kernels (with `Pipes::act`).
struct VadPipes {
    patch: wgpu::ComputePipeline,
    ln: wgpu::ComputePipeline,
    rot: wgpu::ComputePipeline,
    planes: wgpu::ComputePipeline,
    glu: wgpu::ComputePipeline,
    axpb: wgpu::ComputePipeline,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();
static FRONT_PIPES: OnceLock<Option<FrontPipes>> = OnceLock::new();
static VAD_PIPES: OnceLock<Option<VadPipes>> = OnceLock::new();

/// The pipelines `make` creates with `mk(source, entry)`; None where the
/// device rejects one.
fn compiled<T>(c: &Ctx, make: impl FnOnce(&dyn Fn(&str, &str) -> wgpu::ComputePipeline) -> T) -> Option<T> {
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mk = |src: &str, entry: &str| {
        let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(format!("{PRE}{src}").into()),
        });
        c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &m,
            entry_point: Some(entry),
            compilation_options: crate::gpu_wgpu::pco(),
            cache: None,
        })
    };
    let p = make(&mk);
    if let Some(e) = pollster::block_on(sc.pop()) {
        tracing::warn!("gigaam kernels rejected: {e}");
        return None;
    }
    Some(p)
}

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            compiled(c, |mk| Pipes {
                act: mk(ACT_SRC, "ga_bias_act"),
                add3: mk(ADD3_SRC, "ga_add3"),
                acc: mk(ACC_SRC, "ga_acc"),
                fork: mk(FORK_SRC, "ga_fork"),
                fold: mk(FOLD_SRC, "ga_fold"),
                rms: mk(RMS_SRC, "ga_rms"),
                tail: mk(TAIL_SRC, "ga_tail"),
                split: mk(SPLIT_SRC, "ga_split"),
                gate: mk(GATE_SRC, "ga_gate"),
                mean: mk(MEAN_SRC, "ga_mean"),
                conv: mk(CONV_SRC, "ga_conv"),
                nt_ref: mk(NTREF_SRC, "ga_nt_ref"),
            })
        })
        .as_ref()
}

fn front_pipes(c: &Ctx) -> Option<&'static FrontPipes> {
    FRONT_PIPES.get_or_init(|| compiled(c, |mk| FrontPipes { conv2d: mk(CONV2D_SRC, "ga_conv2d") })).as_ref()
}

fn vad_pipes(c: &Ctx) -> Option<&'static VadPipes> {
    VAD_PIPES
        .get_or_init(|| {
            compiled(c, |mk| VadPipes {
                patch: mk(PATCH_SRC, "ga_patch1d"),
                ln: mk(LN_SRC, "ga_ln"),
                rot: mk(ROT_SRC, "ga_rot_half"),
                planes: mk(PLANES_SRC, "ga_planes"),
                glu: mk(GLU_SRC, "ga_glu_conv"),
                axpb: mk(AXPB_SRC, "ga_axpb"),
            })
        })
        .as_ref()
}

/// A product on the card: the weight plane and its bias (zeros where the
/// product has none).
struct LinDev {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

struct ConvDev {
    inp: LinDev,
    /// taps `[kernel][dim]`, the convolution's bias, the mean's weights
    cw: wgpu::Buffer,
    kernel: usize,
    out: LinDev,
}

struct LayerDev {
    ff1: (LinDev, LinDev),
    in_norm: wgpu::Buffer,
    qk: LinDev,
    vg: LinDev,
    /// the biases of qk, then of vg
    ab: wgpu::Buffer,
    attn_out: LinDev,
    conv: Option<ConvDev>,
    ff2: Option<(LinDev, LinDev)>,
    norm: wgpu::Buffer,
}

struct StackDev {
    proj: LinDev,
    proj_t: LinDev,
    layers: Vec<LayerDev>,
}

struct Model {
    key: u64,
    stacks: Vec<StackDev>,
    /// the cosines `[position][head_dim / 2]`, then the sines
    cs: wgpu::Buffer,
    /// the output norm's scale in every lane
    out_w: wgpu::Buffer,
}

/// The working buffers of a segment of up to `cap` frames.
struct Work {
    cap: usize,
    xb: wgpu::Buffer,
    xout: wgpu::Buffer,
    h: wgpu::Buffer,
    s: wgpu::Buffer,
    res: wgpu::Buffer,
    tmp: wgpu::Buffer,
    byp: wgpu::Buffer,
    wide: wgpu::Buffer,
    qkb: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    vt: wgpu::Buffer,
    pan: wgpu::Buffer,
    sc: wgpu::Buffer,
    mean: wgpu::Buffer,
    /// the planes of a product with K split over workgroups
    part: wgpu::Buffer,
    stage: wgpu::Buffer,
    /// the per-segment uniforms, a 256-byte slot each
    dynb: wgpu::Buffer,
}

/// The front end on the card: the three convolutions (weight, bias, channels
/// out), the depthwise one, the two pointwise ones as convolutions, the projection.
struct FrontDev {
    conv: Vec<(wgpu::Buffer, wgpu::Buffer, usize)>,
    dw: (wgpu::Buffer, wgpu::Buffer),
    pw1: LinDev,
    pw2: LinDev,
    out: LinDev,
}

/// The front end's volumes for up to `cap` mel frames.
struct FrontWork {
    cap: usize,
    fin: wgpu::Buffer,
    a1: wgpu::Buffer,
    a2: wgpu::Buffer,
    a3: wgpu::Buffer,
    a4: wgpu::Buffer,
    up: wgpu::Buffer,
    dynb: wgpu::Buffer,
}

/// The front end beside its model's stacks: its plan ends in their input.
struct Front {
    dev: FrontDev,
    work: Option<FrontWork>,
    plan: Option<Plan>,
}

struct VadLayerDev {
    /// a norm: the weight, then the bias
    norm_ff1: wgpu::Buffer,
    ff1: (LinDev, LinDev),
    norm_att: wgpu::Buffer,
    q: LinDev,
    k: LinDev,
    v: LinDev,
    o: LinDev,
    norm_conv: wgpu::Buffer,
    pw1: LinDev,
    /// taps `[kernel][dim]`, then the convolution's bias
    cw: wgpu::Buffer,
    conv_norm: wgpu::Buffer,
    pw2: LinDev,
    norm_ff2: wgpu::Buffer,
    ff2: (LinDev, LinDev),
    norm_out: wgpu::Buffer,
}

struct VadDev {
    /// the two convolutions over time as products over patches
    pre: Vec<LinDev>,
    layers: Vec<VadLayerDev>,
    head: LinDev,
    /// the cosines `[step][head_dim / 2]`, then the sines
    cs: wgpu::Buffer,
}

/// The voice-activity working buffers for up to `windows` windows of up to
/// `frames` mel frames.
struct VadWork {
    windows: usize,
    frames: usize,
    fin: wgpu::Buffer,
    patch: wgpu::Buffer,
    y1: wgpu::Buffer,
    r: wgpu::Buffer,
    r2: wgpu::Buffer,
    xn: wgpu::Buffer,
    tmp: wgpu::Buffer,
    rot: wgpu::Buffer,
    up: wgpu::Buffer,
    qp: wgpu::Buffer,
    kp: wgpu::Buffer,
    vp: wgpu::Buffer,
    sc: wgpu::Buffer,
    logit: wgpu::Buffer,
    stage: wgpu::Buffer,
    dynb: wgpu::Buffer,
}

/// A call in chunks of windows: every window's features, every step's logit
/// and their staging, for up to `feats` and `logits` floats.
struct VadIo {
    feats: usize,
    logits: usize,
    fall: wgpu::Buffer,
    oall: wgpu::Buffer,
    stage: wgpu::Buffer,
}

struct VadState {
    key: u64,
    dev: VadDev,
    work: Option<VadWork>,
    plan: Option<Plan>,
    io: Option<VadIo>,
}

/// The uniform slots a plan may take: the stacks', the front end's, the
/// voice-activity model's.
const MAX_DYN: usize = 1024;
const FRONT_DYN: usize = 16;
const VAD_DYN: usize = 64;

/// The rows of a call: `mult` windows of `t` frames, each on its own (an
/// encoder segment is one window of its frames). A step of fold `f` works on
/// `mult · ⌈t / f⌉` rows.
#[derive(Clone, Copy)]
struct Span {
    t: usize,
    mult: usize,
}

impl Span {
    fn win(self, fold: usize) -> usize {
        self.t.div_ceil(fold)
    }

    fn rows(self, fold: usize) -> usize {
        self.mult * self.win(fold)
    }
}

/// A uniform that follows the call: its words from the rows of its step.
#[derive(Clone, Copy, PartialEq)]
enum Dyn {
    /// `R` of this file's kernels
    Rows,
    /// `MmP` of the NT GEMMs
    Mm { cols4: u32, rows: u32 },
    /// `MmP` with K split over the workgroups of z (`ksplit`), and `R` of the
    /// fold after it: the rows and the planes
    MmK { cols4: u32, rows: u32, tiles: u32 },
    Fold { cols4: u32, tiles: u32 },
    /// `DitP` of `dit_qk` / `dit_softmax` / `dit_pv`, the heads (of every
    /// window) on z
    Qk { hd: u32, scale: f32 },
    Sm { hd: u32 },
    Pv { hd: u32 },
    /// `DcP` of `dit_gemm_coop`, a head a dispatch
    CoopQk { h: u32, hd: u32, scale: f32 },
    CoopPv { h: u32, hd: u32 },
    /// `VtP` of `dit_v_transpose`
    Vt { nh: u32, hd: u32 },
}

impl Dyn {
    /// For a step of `fold` in a call over `sp`: `n` rows in all, `win` a window.
    fn words(self, sp: Span, fold: usize) -> [u32; 8] {
        let one = 1f32.to_bits();
        let (n, win) = (sp.rows(fold) as u32, sp.win(fold) as u32);
        match self {
            Dyn::Rows => [n, win, sp.win((fold / 2).max(1)) as u32, sp.mult as u32, 0, 0, 0, 0],
            Dyn::Mm { cols4, rows } => [cols4, rows, n, 0, 0, 0, 0, 0],
            Dyn::MmK { cols4, rows, tiles } => [cols4, rows, n, ksplit(n, cols4 * 4, tiles).0, 0, 0, 0, 0],
            Dyn::Fold { cols4, tiles } => [n, ksplit(n, cols4 * 4, tiles).1, 0, 0, 0, 0, 0, 0],
            Dyn::Qk { hd, scale } => [win, hd, win, scale.to_bits(), 0, 0, 1, win],
            Dyn::Sm { hd } => [win, hd, win, one, 0, 0, 1, win],
            Dyn::Pv { hd } => [win, win, hd, one, 0, 0, 1, win],
            Dyn::CoopQk { h, hd, scale } => [hd / 4, n, n, scale.to_bits(), h * n * hd, h * n * hd, h * n * n, hd],
            Dyn::CoopPv { h, hd } => [n / 4, hd, n, one, h * n * n, h * n * hd, h * n * hd, n],
            Dyn::Vt { nh, hd } => [n, nh, hd, 0, 0, 0, 0, 0],
        }
    }

    /// bytes of the kernel's struct
    fn size(self) -> u64 {
        match self {
            Dyn::Rows | Dyn::Mm { .. } | Dyn::MmK { .. } | Dyn::Fold { .. } | Dyn::Vt { .. } => 16,
            _ => 32,
        }
    }
}

/// The NT GEMM of this device and its name in the profile.
fn gemm_of(c: &Ctx) -> (&wgpu::ComputePipeline, &'static str) {
    match c.gemm_nt_coop.as_ref() {
        Some(g) if std::env::var("CMF_GIGAAM_GEMM").as_deref() != Ok("f32") => (g, "matrix units"),
        _ => (&c.gemm_nt_f32t, "f32 tiles"),
    }
}

/// Tile rows × K slices a short segment's product spreads over
/// (`CMF_GIGAAM_KSPLIT`, at most 32); 0: no split — asked for, or the GEMM is
/// not the f32 tile kernel, whose switch this is.
fn k_tiles(c: &Ctx) -> u32 {
    let want = std::env::var("CMF_GIGAAM_KSPLIT").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(4).min(32);
    if want >= 2 && std::ptr::eq(gemm_of(c).0, &c.gemm_nt_f32t) { want } else { 0 }
}

/// A short segment's products are bound by the K loop of their few tiles, not
/// by the device: `gemm_nt_f32t` can split K over workgroups (its `kspan`),
/// each slice into its own plane, summed after (`ga_fold`). For `n` rows and
/// `cols` of K, `tiles` tile rows × slices wanted: (the slice, the slices).
fn ksplit(n: u32, cols: u32, tiles: u32) -> (u32, u32) {
    let z = (tiles / n.div_ceil(64).max(1)).clamp(1, cols.div_ceil(64).max(1));
    let kspan = cols.div_ceil(z).next_multiple_of(16);
    (kspan, cols.div_ceil(kspan))
}

/// A dispatch grid from the rows of the op's step: `n` in all, `win` a
/// window, `windows` of them.
#[derive(Clone, Copy)]
enum Grid {
    /// `n · per` invocations, 256 a workgroup
    Flat(usize),
    /// a workgroup a row
    Rows,
    /// 64×64 tiles of an `[n][out]` product
    Tile(u32),
    /// the same with K split on z: (out, cols, tiles)
    TileK(u32, u32, u32),
    /// 64×64 tiles of `[win][win]`, `windows ·` the heads on z
    Square(u32),
    /// a workgroup a row of `[windows · heads][win]`
    Sm(u32),
    /// `(x, win / 64, windows · heads)`
    Pv(u32, u32),
    /// `x` workgroups whatever the rows
    Fixed(u32),
}

fn grid(total: usize) -> (u32, u32) {
    let wgs = total.div_ceil(256).max(1);
    let x = wgs.min(65_535);
    (x as u32, wgs.div_ceil(x) as u32)
}

impl Grid {
    fn dims(self, sp: Span, fold: usize) -> (u32, u32, u32) {
        let (n, win, windows) = (sp.rows(fold), sp.win(fold) as u32, sp.mult as u32);
        let t64 = (n as u32).div_ceil(64).max(1);
        let w64 = win.div_ceil(64).max(1);
        match self {
            Grid::Flat(per) => {
                let (x, y) = grid(n * per);
                (x, y, 1)
            }
            Grid::Rows => (n.clamp(1, 65_535) as u32, n.div_ceil(65_535).max(1) as u32, 1),
            Grid::Tile(out) => (out.div_ceil(64), t64, 1),
            Grid::TileK(out, cols, tiles) => (out.div_ceil(64), t64, ksplit(n as u32, cols, tiles).1),
            Grid::Square(z) => (w64, w64, windows * z),
            Grid::Sm(nh) => (win, windows * nh, 1),
            Grid::Pv(x, nh) => (x, w64, windows * nh),
            Grid::Fixed(x) => (x, 1, 1),
        }
    }
}

/// What a step is, for the profile.
const PRODUCT: usize = 0;
const ATTENTION: usize = 1;
const GLUE: usize = 2;
const FRONT: usize = 3;

/// Which segments run a step: a plan holds both ways of a product (K split
/// or whole) and of attention (every head in a dispatch, or a head a dispatch
/// on the matrix units), and the rows of the step's stack choose.
#[derive(Clone, Copy, PartialEq)]
enum When {
    Always,
    /// at most / more than `Plan::k_rows` rows
    KSplit,
    KWhole,
    /// at most / more than `Plan::head_rows` rows
    AttnShort,
    AttnLong,
}

struct Step {
    pipe: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    /// frames in a row of the step (see `Span`)
    fold: usize,
    grid: Grid,
    kind: usize,
    when: When,
}

struct Plan {
    steps: Vec<Step>,
    dyns: Vec<(Dyn, usize)>,
    k_rows: usize,
    head_rows: usize,
    /// (products, attention) as the profile line names them
    names: (&'static str, String),
}

impl Plan {
    fn runs(&self, step: &Step, n: usize) -> bool {
        match step.when {
            When::Always => true,
            When::KSplit => n <= self.k_rows,
            When::KWhole => n > self.k_rows,
            When::AttnShort => n <= self.head_rows,
            When::AttnLong => n > self.head_rows,
        }
    }

    /// The plan's uniforms for a call over `sp`, a 256-byte slot each.
    fn uniforms(&self, sp: Span) -> Vec<u32> {
        let mut du = vec![0u32; self.dyns.len() * 64];
        for (i, (d, fold)) in self.dyns.iter().enumerate() {
            du[i * 64..i * 64 + 8].copy_from_slice(&d.words(sp, *fold));
        }
        du
    }

    /// The plan's steps (all, or one kind of them) for a call over `sp` into
    /// `pass`, opened in `enc` where there is none; `count`: (passes, steps).
    fn replay(&self, sp: Span, only: Option<usize>, enc: &mut wgpu::CommandEncoder, pass: &mut Option<wgpu::ComputePass<'static>>, count: &mut (usize, usize)) {
        let merge = std::env::var("CMF_PASS_MERGE").as_deref() != Ok("0");
        for step in self.steps.iter().filter(|s| only.is_none_or(|k| s.kind == k)) {
            if !self.runs(step, sp.rows(step.fold)) {
                continue;
            }
            count.1 += 1;
            if !merge {
                drop(pass.take());
            }
            if pass.is_none() {
                count.0 += 1;
                *pass = Some(enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ga"), timestamp_writes: None }).forget_lifetime());
            }
            let ps = pass.as_mut().expect("set above");
            let (gx, gy, gz) = step.grid.dims(sp, step.fold);
            ps.set_pipeline(&step.pipe);
            ps.set_bind_group(0, &step.bg, &[]);
            ps.dispatch_workgroups(gx, gy, gz);
        }
    }
}

struct State {
    model: Model,
    work: Option<Work>,
    plan: Option<Plan>,
    front: Option<Front>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static VAD: Mutex<Option<VadState>> = Mutex::new(None);

const USE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST).union(wgpu::BufferUsages::COPY_SRC);

/// Uploads queued since the last flush (bytes): the queue keeps a staging
/// copy of every write until a submit.
struct Up<'a> {
    c: &'a Ctx,
    pending: usize,
    total: usize,
}

impl Up<'_> {
    fn stor(&mut self, data: &[f32]) -> wgpu::Buffer {
        let b = self.c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ga-w"),
            size: (data.len().max(4).next_multiple_of(4) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        self.pending += data.len() * 4;
        self.total += data.len() * 4;
        if self.pending >= 96 << 20 {
            self.pending = 0;
            super::submit_empty(self.c);
            let _ = self.c.device.poll(wgpu::PollType::Poll);
        }
        b
    }

    fn lin(&mut self, l: &GaLin, scratch: &mut Vec<f32>) -> LinDev {
        scratch.clear();
        scratch.resize(l.rows * l.cols, 0.0);
        l.weight.fill(scratch);
        let w = self.stor(scratch);
        let b = if l.bias.is_empty() { self.stor(&vec![0f32; l.rows]) } else { self.stor(l.bias) };
        LinDev { w, b, rows: l.rows, cols: l.cols }
    }
}

fn upload(c: &Ctx, a: &GigaamArgs) -> (Model, usize) {
    crate::gpu::probe_note_cold();
    let mut up = Up { c, pending: 0, total: 0 };
    let mut scratch = Vec::new();
    let mut stacks = Vec::with_capacity(a.stacks.len());
    for st in &a.stacks {
        let hw = st.heads * a.head_dim;
        let mut layers = Vec::with_capacity(st.layers.len());
        for l in &st.layers {
            let conv = l.conv.as_ref().map(|cv| {
                let cw: Vec<f32> = cv.taps.iter().chain(cv.bias).chain(cv.mean_w).copied().collect();
                ConvDev { inp: up.lin(&cv.inp, &mut scratch), cw: up.stor(&cw), kernel: cv.kernel, out: up.lin(&cv.out, &mut scratch) }
            });
            let ff2 = l.ff2.as_ref().map(|(i, o)| (up.lin(i, &mut scratch), up.lin(o, &mut scratch)));
            let mut ab = vec![0f32; 5 * hw];
            if !l.qk.bias.is_empty() {
                ab[..2 * hw].copy_from_slice(l.qk.bias);
            }
            if !l.vg.bias.is_empty() {
                ab[2 * hw..].copy_from_slice(l.vg.bias);
            }
            layers.push(LayerDev {
                ff1: (up.lin(&l.ff1_in, &mut scratch), up.lin(&l.ff1_out, &mut scratch)),
                in_norm: up.stor(l.in_norm),
                qk: up.lin(&l.qk, &mut scratch),
                vg: up.lin(&l.vg, &mut scratch),
                ab: up.stor(&ab),
                attn_out: up.lin(&l.attn_out, &mut scratch),
                conv,
                ff2,
                norm: up.stor(l.norm),
            });
        }
        stacks.push(StackDev { proj: up.lin(&st.proj, &mut scratch), proj_t: up.lin(&st.proj_t, &mut scratch), layers });
    }
    let cs: Vec<f32> = a.cos.iter().chain(a.sin).copied().collect();
    let model = Model { key: a.key, stacks, cs: up.stor(&cs), out_w: up.stor(&vec![a.out_scale; a.width]) };
    (model, up.total)
}

fn upload_front(c: &Ctx, fr: &GaFront) -> FrontDev {
    crate::gpu::probe_note_cold();
    let mut up = Up { c, pending: 0, total: 0 };
    let mut scratch = Vec::new();
    FrontDev {
        conv: fr.conv.iter().map(|(w, b, co)| (up.stor(w), up.stor(b), *co)).collect(),
        dw: (up.stor(fr.dw.0), up.stor(fr.dw.1)),
        pw1: up.lin(&fr.pw1, &mut scratch),
        pw2: up.lin(&fr.pw2, &mut scratch),
        out: up.lin(&fr.out, &mut scratch),
    }
}

fn upload_vad(c: &Ctx, a: &GaVadArgs) -> VadDev {
    crate::gpu::probe_note_cold();
    let mut up = Up { c, pending: 0, total: 0 };
    let mut sc = Vec::new();
    // two slices back to back: a norm's weight and bias, the taps and their bias
    let pair = |up: &mut Up, x: &[f32], y: &[f32]| up.stor(&x.iter().chain(y).copied().collect::<Vec<f32>>());
    let pre = a.pre.iter().map(|(w, b, co, ci)| LinDev { w: up.stor(w), b: up.stor(b), rows: *co, cols: 3 * ci }).collect();
    let mut layers = Vec::with_capacity(a.layers.len());
    for l in &a.layers {
        layers.push(VadLayerDev {
            norm_ff1: pair(&mut up, l.norm_ff1.0, l.norm_ff1.1),
            ff1: (up.lin(&l.ff1.0, &mut sc), up.lin(&l.ff1.1, &mut sc)),
            norm_att: pair(&mut up, l.norm_att.0, l.norm_att.1),
            q: up.lin(&l.q, &mut sc),
            k: up.lin(&l.k, &mut sc),
            v: up.lin(&l.v, &mut sc),
            o: up.lin(&l.o, &mut sc),
            norm_conv: pair(&mut up, l.norm_conv.0, l.norm_conv.1),
            pw1: up.lin(&l.pw1, &mut sc),
            cw: pair(&mut up, l.taps, l.dw_b),
            conv_norm: pair(&mut up, l.conv_norm.0, l.conv_norm.1),
            pw2: up.lin(&l.pw2, &mut sc),
            norm_ff2: pair(&mut up, l.norm_ff2.0, l.norm_ff2.1),
            ff2: (up.lin(&l.ff2.0, &mut sc), up.lin(&l.ff2.1, &mut sc)),
            norm_out: pair(&mut up, l.norm_out.0, l.norm_out.1),
        });
    }
    VadDev { pre, layers, head: up.lin(&a.head, &mut sc), cs: pair(&mut up, a.cos, a.sin) }
}

/// Does every shape sit inside the kernels' contract?
fn shapes_ok(a: &GigaamArgs, x: &[f32], t: usize) -> bool {
    let (w, hd) = (a.width, a.head_dim);
    let lin = |l: &GaLin, rows: usize, cols: usize| l.rows == rows && l.cols == cols && rows % 4 == 0 && cols % 4 == 0 && (l.bias.is_empty() || l.bias.len() == rows);
    if t == 0 || t > 65_535 || x.len() != t * w || w % 4 != 0 || hd == 0 || hd % 4 != 0 || a.cos.len() < t * hd / 2 || a.sin.len() != a.cos.len() {
        return false;
    }
    a.stacks.iter().all(|st| {
        let (d, hw) = (st.dim, st.heads * hd);
        (st.fold == 1 || st.fold == 2)
            && st.heads > 0
            && st.residual_scales.len() == st.layers.len() + 1
            && lin(&st.proj, d, st.fold * w)
            && lin(&st.proj_t, st.fold * w, d)
            && st.layers.iter().all(|l| {
                let inter = l.ff1_in.rows;
                lin(&l.ff1_in, inter, d)
                    && lin(&l.ff1_out, d, inter)
                    && l.in_norm.len() == d
                    && l.norm.len() == d
                    && lin(&l.qk, 2 * hw, d)
                    && lin(&l.vg, 3 * hw, d)
                    && lin(&l.attn_out, d, hw)
                    && l.conv.as_ref().is_none_or(|cv| {
                        cv.kernel > 0 && lin(&cv.inp, 2 * d, d) && lin(&cv.out, d, d) && cv.taps.len() == cv.kernel * d && cv.bias.len() == d && cv.mean_w.len() == d
                    })
                    && l.ff2.as_ref().is_none_or(|(i, o)| lin(i, i.rows, d) && lin(o, d, i.rows))
            })
    })
}

/// The bins after the front end's two strided convolutions.
fn front_bins(n_mel: usize) -> (usize, usize) {
    let f2 = (n_mel.max(3) - 3) / 2 + 1;
    (f2, (f2.max(3) - 3) / 2 + 1)
}

/// Does the front end sit inside its kernels' contract, `feats` being
/// `frames` rows of it for a model `width` wide?
fn front_ok(fr: &GaFront, width: usize, feats: &[f32], frames: usize) -> bool {
    let n_mel = fr.n_mel;
    let (c0, c1, c2) = (fr.conv[0].2, fr.conv[1].2, fr.conv[2].2);
    let (f2, f3) = front_bins(n_mel);
    let conv = |i: usize, ci: usize| fr.conv[i].0.len() == fr.conv[i].2 * ci * 9 && fr.conv[i].1.len() == fr.conv[i].2;
    let lin = |l: &GaLin, rows: usize, cols: usize| l.rows == rows && l.cols == cols && (l.bias.is_empty() || l.bias.len() == rows);
    frames > 0
        && frames <= 65_535
        && n_mel >= 3
        && f2 >= 3
        && feats.len() == frames * n_mel
        && c0 > 0
        && c1 > 0
        && c2 > 0
        && conv(0, 1)
        && conv(1, c0)
        && conv(2, c1)
        && fr.dw.0.len() == c2 * 9
        && fr.dw.1.len() == c2
        && fr.pw1.rows > 0
        && lin(&fr.pw1, fr.pw1.rows, c2)
        && lin(&fr.pw2, c2, fr.pw1.rows)
        && lin(&fr.out, width, c2 * f3)
        && (c2 * f3) % 4 == 0
        && width % 4 == 0
}

/// Does the voice-activity model sit inside its kernels' contract for
/// `windows` windows of `frames` mel frames in `feats`, their logits in `out`?
fn vad_ok(a: &GaVadArgs, feats: &[f32], windows: usize, frames: usize, out: &[f32]) -> bool {
    let (d, nh, n_mel) = (a.dim, a.heads, a.n_mel);
    let lin = |l: &GaLin, rows: usize, cols: usize| l.rows == rows && l.cols == cols && rows % 4 == 0 && cols % 4 == 0 && (l.bias.is_empty() || l.bias.len() == rows);
    let norm = |n: (&[f32], &[f32])| n.0.len() == d && n.1.len() == d;
    let pre = |i: usize, co: usize, ci: usize| {
        let (w, b, rows, cols) = a.pre[i];
        rows == co && cols == ci && w.len() == co * 3 * ci && b.len() == co && co % 4 == 0 && ci % 4 == 0
    };
    if windows == 0 || frames == 0 || d == 0 || nh == 0 || d % nh != 0 {
        return false;
    }
    let (hd, steps, c0) = (d / nh, frames.div_ceil(4), a.pre[0].2);
    hd % 2 == 0
        && steps <= 65_535
        && feats.len() == windows * frames * n_mel
        && out.len() == windows * steps
        && a.cos.len() >= steps * hd / 2
        && a.sin.len() == a.cos.len()
        && c0 > 0
        && pre(0, c0, n_mel)
        && pre(1, d, c0)
        && a.kernel % 2 == 1
        && a.head.rows == 1
        && a.head.cols == d
        && a.head.bias.len() <= 1
        && a.layers.iter().all(|l| {
            let ff = |f: &(GaLin, GaLin)| f.0.rows > 0 && lin(&f.0, f.0.rows, d) && lin(&f.1, d, f.0.rows);
            norm(l.norm_ff1)
                && ff(&l.ff1)
                && norm(l.norm_att)
                && lin(&l.q, d, d)
                && lin(&l.k, d, d)
                && lin(&l.v, d, d)
                && lin(&l.o, d, d)
                && norm(l.norm_conv)
                && lin(&l.pw1, 2 * d, d)
                && l.taps.len() == a.kernel * d
                && l.dw_b.len() == d
                && norm(l.conv_norm)
                && lin(&l.pw2, d, d)
                && norm(l.norm_ff2)
                && ff(&l.ff2)
                && norm(l.norm_out)
        })
}

fn alloc(c: &Ctx, a: &GigaamArgs, cap: usize) -> Option<Work> {
    let w = a.width;
    let hd = a.head_dim;
    let max_dim = a.stacks.iter().map(|st| st.dim).max().unwrap_or(w);
    let max_hw = a.stacks.iter().map(|st| st.heads * hd).max().unwrap_or(hd);
    let wide_cols = a
        .stacks
        .iter()
        .flat_map(|st| st.layers.iter())
        .map(|l| l.ff1_in.rows.max(l.vg.rows).max(l.conv.as_ref().map_or(0, |cv| cv.inp.rows)).max(l.ff2.as_ref().map_or(0, |f| f.0.rows)))
        .max()
        .unwrap_or(max_dim);
    let max_out = wide_cols.max(max_dim).max(a.stacks.iter().map(|st| st.fold * w).max().unwrap_or(w));
    let scores = a.stacks.iter().map(|st| st.heads * cap.div_ceil(st.fold).pow(2)).max().unwrap_or(1);
    let lim = c.device.limits();
    let max_bind = (lim.max_storage_buffer_binding_size as u64).min(lim.max_buffer_size);
    let sizes = [(cap + 1) * w, cap * max_dim.max(max_hw), cap * wide_cols, cap * 2 * max_hw, scores];
    if sizes.iter().any(|s| (*s as u64 * 4).next_multiple_of(16) > max_bind) {
        return None;
    }
    let make = |floats: usize, usage: wgpu::BufferUsages| {
        let size = ((floats.max(4) * 4) as u64).next_multiple_of(16);
        c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ga-act"), size, usage, mapped_at_creation: false })
    };
    let st = |floats: usize| make(floats, USE);
    let rows_d = cap * max_dim.max(max_hw);
    let planes = cap * max_hw;
    Some(Work {
        cap,
        xb: st((cap + 1) * w),
        xout: st((cap + 1) * w),
        h: st(rows_d),
        s: st(rows_d),
        res: st(rows_d),
        tmp: st(rows_d),
        byp: st(rows_d),
        wide: st(cap * wide_cols),
        qkb: st(cap * 2 * max_hw),
        q: st(planes),
        k: st(planes),
        v: st(planes),
        vt: st(planes),
        pan: st(planes),
        sc: st(scores),
        mean: st(max_dim),
        part: st(k_tiles(c) as usize * 64 * max_out),
        stage: make(cap * w, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        dynb: make(MAX_DYN * 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
    })
}

/// A buffer of `floats` (16 bytes at least, a multiple of 16).
fn buffer(c: &Ctx, floats: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let size = ((floats.max(4) * 4) as u64).next_multiple_of(16);
    c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ga-act"), size, usage, mapped_at_creation: false })
}

/// Does a storage buffer of each of `floats` fit a binding of this device?
fn binds(c: &Ctx, floats: &[usize]) -> bool {
    let lim = c.device.limits();
    let max_bind = (lim.max_storage_buffer_binding_size as u64).min(lim.max_buffer_size);
    floats.iter().all(|s| (*s as u64 * 4).next_multiple_of(16) <= max_bind)
}

fn alloc_front(c: &Ctx, fr: &GaFront, cap: usize) -> Option<FrontWork> {
    let (c0, c1, c2) = (fr.conv[0].2, fr.conv[1].2, fr.conv[2].2);
    let (f2, f3) = front_bins(fr.n_mel);
    let (t2, t3) = (cap.div_ceil(2), cap.div_ceil(4));
    // the volumes after each convolution, the block's wide one last
    let sizes = [cap * fr.n_mel, c0 * cap * fr.n_mel, c1 * t2 * f2, c2 * t3 * f3, fr.pw1.rows * t3 * f3];
    if !binds(c, &sizes) {
        return None;
    }
    let st = |floats: usize| buffer(c, floats, USE);
    Some(FrontWork {
        cap,
        fin: st(sizes[0]),
        a1: st(sizes[1]),
        a2: st(sizes[2]),
        a3: st(sizes[3]),
        a4: st(sizes[3]),
        up: st(sizes[4]),
        dynb: buffer(c, FRONT_DYN * 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
    })
}

/// Floats of the voice-activity working buffers for `windows` windows of
/// `frames` mel frames: the features, the patches, the first convolution, a
/// plane of rows, the wide rows, the scores.
fn vad_sizes(a: &GaVadArgs, windows: usize, frames: usize) -> [usize; 6] {
    let (l1, steps) = (frames.div_ceil(2), frames.div_ceil(4));
    let (c0, d) = (a.pre[0].2, a.dim);
    let wide = a.layers.iter().map(|l| l.ff1.0.rows.max(l.ff2.0.rows).max(l.pw1.rows)).max().unwrap_or(d);
    let patches = (l1 * 3 * a.n_mel).max(steps * 3 * c0);
    [windows * frames * a.n_mel, windows * patches, windows * l1 * c0, windows * steps * d, windows * steps * wide, windows * a.heads * steps * steps]
}

fn alloc_vad(c: &Ctx, a: &GaVadArgs, windows: usize, frames: usize) -> Option<VadWork> {
    let sizes = vad_sizes(a, windows, frames);
    if !binds(c, &sizes) {
        return None;
    }
    let [fin, patch, y1, rows, wide, scores] = sizes;
    let st = |floats: usize| buffer(c, floats, USE);
    let steps = windows * frames.div_ceil(4);
    Some(VadWork {
        windows,
        frames,
        fin: st(fin),
        patch: st(patch),
        y1: st(y1),
        r: st(rows),
        r2: st(rows),
        xn: st(rows),
        tmp: st(rows),
        rot: st(rows),
        up: st(wide),
        qp: st(rows),
        kp: st(rows),
        vp: st(rows),
        sc: st(scores),
        logit: st(steps),
        stage: buffer(c, steps, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        dynb: buffer(c, VAD_DYN * 64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
    })
}

struct Builder<'a> {
    c: &'a Ctx,
    dynb: &'a wgpu::Buffer,
    /// the NT GEMM, and with K split: (the fold, its planes, tile rows × slices)
    gemm: &'a wgpu::ComputePipeline,
    split: Option<(&'a wgpu::ComputePipeline, &'a wgpu::Buffer, u32)>,
    when: When,
    steps: Vec<Step>,
    dyns: Vec<(Dyn, usize)>,
    fixed: std::collections::HashMap<[u32; 8], wgpu::Buffer>,
}

impl Builder<'_> {
    /// The slot of the uniform `d` for a stack of `fold` frames a row.
    fn slot(&mut self, d: Dyn, fold: usize) -> u64 {
        let at = match self.dyns.iter().position(|e| *e == (d, fold)) {
            Some(i) => i,
            None => {
                self.dyns.push((d, fold));
                self.dyns.len() - 1
            }
        };
        (at * 256) as u64
    }

    /// The uniform of an op's fixed numbers (`P`), one buffer a distinct set.
    fn fixed_buf(&mut self, words: [u32; 8]) -> wgpu::Buffer {
        let c = self.c;
        self.fixed
            .entry(words)
            .or_insert_with(|| {
                use wgpu::util::DeviceExt;
                let contents = bytemuck::cast_slice(&words);
                c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("ga-p"), contents, usage: wgpu::BufferUsages::UNIFORM })
            })
            .clone()
    }

    fn push(&mut self, pipe: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry], fold: usize, grid: Grid, kind: usize) {
        let bg = self.c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ga"), layout: &pipe.get_bind_group_layout(0), entries });
        self.steps.push(Step { pipe: pipe.clone(), bg, fold, grid, kind, when: self.when });
    }

    /// A kernel of this file: the storage buffers in binding order, then `P`
    /// (`a b c d` and four floats), then `R`.
    #[allow(clippy::too_many_arguments)]
    fn op(&mut self, pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer], ints: [u32; 4], floats: [f32; 4], fold: usize, grid: Grid, kind: usize) {
        let words = [ints[0], ints[1], ints[2], ints[3], floats[0].to_bits(), floats[1].to_bits(), floats[2].to_bits(), floats[3].to_bits()];
        self.op_words(pipe, bufs, words, fold, grid, kind);
    }

    /// The same for a kernel with eight words of its own in place of `P`.
    fn op_words(&mut self, pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer], words: [u32; 8], fold: usize, grid: Grid, kind: usize) {
        let pb = self.fixed_buf(words);
        let off = self.slot(Dyn::Rows, fold);
        let dynb = self.dynb;
        let mut e: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| super::bind_buf(i as u32, b)).collect();
        e.push(super::bind_buf(bufs.len() as u32, &pb));
        e.push(super::bind_buf_off(bufs.len() as u32 + 1, dynb, off, 16));
        self.push(pipe, &e, fold, grid, kind);
    }

    /// A kernel of the backend: its buffers by binding, its own uniform at `ub`.
    #[allow(clippy::too_many_arguments)]
    fn ext(&mut self, pipe: &wgpu::ComputePipeline, bufs: &[(u32, &wgpu::Buffer)], ub: u32, d: Dyn, fold: usize, grid: Grid, kind: usize) {
        let off = self.slot(d, fold);
        let dynb = self.dynb;
        let mut e: Vec<wgpu::BindGroupEntry> = bufs.iter().map(|(i, b)| super::bind_buf(*i, b)).collect();
        e.push(super::bind_buf_off(ub, dynb, off, d.size()));
        self.push(pipe, &e, fold, grid, kind);
    }

    /// y `[n][rows]` = x `[n][cols]` · wᵀ.
    fn mm(&mut self, l: &LinDev, x: &wgpu::Buffer, y: &wgpu::Buffer, fold: usize) {
        let (gemm, cols4, rows) = (self.gemm, (l.cols / 4) as u32, l.rows as u32);
        if let Some((fold_pipe, part, tiles)) = self.split {
            // a short segment: K over workgroups into planes, then their sum
            self.when = When::KSplit;
            self.ext(gemm, &[(0, &l.w), (1, x), (2, part)], 3, Dyn::MmK { cols4, rows, tiles }, fold, Grid::TileK(rows, l.cols as u32, tiles), PRODUCT);
            let off = self.slot(Dyn::Fold { cols4, tiles }, fold);
            let pb = self.fixed_buf([rows / 4, 0, 0, 0, 0, 0, 0, 0]);
            let e = [super::bind_buf(0, part), super::bind_buf(1, y), super::bind_buf(2, &pb), super::bind_buf_off(3, self.dynb, off, 16)];
            self.push(fold_pipe, &e, fold, Grid::Flat(l.rows / 4), PRODUCT);
            self.when = When::KWhole;
        }
        self.ext(gemm, &[(0, &l.w), (1, x), (2, y)], 3, Dyn::Mm { cols4, rows }, fold, Grid::Tile(rows), PRODUCT);
        self.when = When::Always;
    }
}

/// The segment as a list of dispatches over the model's and the working buffers.
fn build(c: &Ctx, p: &Pipes, a: &GigaamArgs, m: &Model, wk: &Work) -> Option<Plan> {
    let env = |k: &str| std::env::var(k).ok();
    let f32_only = env("CMF_GIGAAM_GEMM").as_deref() == Some("f32");
    let (gemm, gemm_name) = gemm_of(c);
    // K split for short segments (see the head of the file)
    let k_tiles = k_tiles(c);
    let split = (k_tiles >= 2).then_some((&p.fold, &wk.part, k_tiles));
    // Attention a head a dispatch: the matrix units (`CMF_GIGAAM_ATTN=ref`: the
    // plain kernel of the same contract), for stacks of more than
    // `CMF_GIGAAM_ATTN_ROWS` rows — below, the two dispatches a head cost more
    // than the shader cores take for all heads at once. `=scalar`: never.
    let per_head = match (env("CMF_GIGAAM_ATTN").as_deref(), c.dit_gemm_coop.as_ref(), c.dit_v_transpose.as_ref()) {
        (Some("scalar"), _, _) => None,
        (Some("ref"), _, Some(vt)) => Some((&p.nt_ref, vt, "the per-head reference")),
        (Some("coop"), Some(g), Some(vt)) => Some((g, vt, "the matrix units")),
        (_, Some(g), Some(vt)) if !f32_only && env("CMF_DIT_ATTN_COOP").as_deref() != Some("0") => Some((g, vt, "the matrix units")),
        _ => None,
    };
    let head_rows = env("CMF_GIGAAM_ATTN_ROWS").and_then(|v| v.parse::<usize>().ok()).unwrap_or(384);
    let (w, hd) = (a.width, a.head_dim);
    let scale = (hd as f32).powf(-0.5);
    let mut b = Builder { c, dynb: &wk.dynb, gemm, split, when: When::Always, steps: Vec::new(), dyns: Vec::new(), fixed: Default::default() };
    let none = [0f32; 4];
    for (st, dev) in a.stacks.iter().zip(&m.stacks) {
        let (fold, d, nh) = (st.fold, st.dim, st.heads);
        let hw = nh * hd;
        let d4 = (d / 4) as u32;
        // rows of this stack: the frames, or pairs of them (the row after the
        // last frame is zero, see the upload)
        b.mm(&dev.proj, &wk.xb, &wk.h, fold);
        b.op(&p.fork, &[&wk.h, &dev.proj.b, &wk.byp], [d4, 0, 0, 0], [st.residual_scales[0], st.input_scale, 0.0, 0.0], fold, Grid::Flat(d / 4), GLUE);
        for (li, l) in dev.layers.iter().enumerate() {
            // the layer input stays in h; the stream is s
            let inter = l.ff1.0.rows;
            b.mm(&l.ff1.0, &wk.h, &wk.wide, fold);
            b.op(&p.act, &[&wk.wide, &l.ff1.0.b], [(inter / 4) as u32, 1, 0, 0], none, fold, Grid::Flat(inter / 4), GLUE);
            b.mm(&l.ff1.1, &wk.wide, &wk.tmp, fold);
            b.op(&p.add3, &[&wk.s, &wk.h, &wk.tmp, &l.ff1.1.b], [d4, 0, 0, 0], none, fold, Grid::Flat(d / 4), GLUE);
            // attention
            b.op(&p.rms, &[&wk.h, &l.in_norm, &wk.res], [d as u32, 0, 0, 0], [1e-6, 0.0, 0.0, 0.0], fold, Grid::Rows, GLUE);
            b.mm(&l.qk, &wk.res, &wk.qkb, fold);
            b.mm(&l.vg, &wk.s, &wk.wide, fold);
            let (nh32, hd32) = (nh as u32, hd as u32);
            let planes = [&wk.qkb, &wk.wide, &l.ab, &m.cs, &wk.q, &wk.k, &wk.v];
            b.op(&p.split, &planes, [nh32, hd32, a.cos.len() as u32, 0], none, fold, Grid::Flat(nh * hd / 2), GLUE);
            let sm = |b: &mut Builder| b.ext(&c.dit_softmax, &[(2, &wk.sc)], 3, Dyn::Sm { hd: hd32 }, fold, Grid::Sm(nh32), ATTENTION);
            // every head in one dispatch: scores, softmax, context
            b.when = if per_head.is_some() { When::AttnShort } else { When::Always };
            b.ext(&c.dit_qk, &[(0, &wk.q), (1, &wk.k), (2, &wk.sc)], 3, Dyn::Qk { hd: hd32, scale }, fold, Grid::Square(nh32), ATTENTION);
            sm(&mut b);
            b.ext(&c.dit_pv, &[(0, &wk.sc), (1, &wk.v), (2, &wk.pan)], 3, Dyn::Pv { hd: hd32 }, fold, Grid::Pv(hd32.div_ceil(64), nh32), ATTENTION);
            if let Some((nt, vt, _)) = per_head {
                // a head a dispatch, v as [hd][n] for the second product
                b.when = When::AttnLong;
                b.ext(vt, &[(0, &wk.v), (1, &wk.vt)], 2, Dyn::Vt { nh: nh32, hd: hd32 }, fold, Grid::Flat(hw), ATTENTION);
                for h in 0..nh32 {
                    b.ext(nt, &[(0, &wk.k), (1, &wk.q), (2, &wk.sc)], 3, Dyn::CoopQk { h, hd: hd32, scale }, fold, Grid::Square(1), ATTENTION);
                }
                sm(&mut b);
                for h in 0..nh32 {
                    b.ext(nt, &[(0, &wk.vt), (1, &wk.sc), (2, &wk.pan)], 3, Dyn::CoopPv { h, hd: hd32 }, fold, Grid::Tile(hd32), ATTENTION);
                }
            }
            b.when = When::Always;
            b.op(&p.gate, &[&wk.pan, &wk.wide, &l.ab, &wk.res], [nh32, hd32, 0, 0], none, fold, Grid::Flat(hw), GLUE);
            b.mm(&l.attn_out, &wk.res, &wk.tmp, fold);
            // the stream's last addition waits in (tmp, bias): the next
            // module adds it, or the layer's way out folds it in
            let mut last = &l.attn_out.b;
            let add_last = |b: &mut Builder, bias: &wgpu::Buffer| b.op(&p.acc, &[&wk.s, &wk.tmp, bias], [d4, d4, 0, 0], none, fold, Grid::Flat(d / 4), GLUE);
            if let Some(cv) = &l.conv {
                add_last(&mut b, last);
                let dk = [d as u32, cv.kernel as u32, 0, 0];
                b.mm(&cv.inp, &wk.s, &wk.wide, fold);
                b.op(&p.mean, &[&wk.wide, &cv.cw, &cv.inp.b, &wk.mean], dk, none, fold, Grid::Fixed(d as u32), GLUE);
                b.op(&p.conv, &[&wk.wide, &cv.cw, &cv.inp.b, &wk.mean, &wk.res], dk, none, fold, Grid::Flat(d), GLUE);
                b.mm(&cv.out, &wk.res, &wk.tmp, fold);
                last = &cv.out.b;
            }
            if let Some(ff2) = &l.ff2 {
                add_last(&mut b, last);
                let inter = ff2.0.rows;
                b.mm(&ff2.0, &wk.s, &wk.wide, fold);
                b.op(&p.act, &[&wk.wide, &ff2.0.b], [(inter / 4) as u32, 1, 0, 0], none, fold, Grid::Flat(inter / 4), GLUE);
                b.mm(&ff2.1, &wk.wide, &wk.tmp, fold);
                last = &ff2.1.b;
            }
            // h = norm(h + ¼ (stream − h)), bypass += scale · h
            b.op(&p.tail, &[&wk.h, &wk.s, &wk.tmp, last, &l.norm, &wk.byp], [d as u32, 0, 0, 0], [1e-6, 0.25, st.residual_scales[li + 1], 0.0], fold, Grid::Rows, GLUE);
        }
        // x += proj_t(bypass): the frames' lanes of the stack's rows
        b.mm(&dev.proj_t, &wk.byp, &wk.xout, fold);
        b.op(&p.acc, &[&wk.xb, &wk.xout, &dev.proj_t.b], [(w / 4) as u32, (fold * w / 4) as u32, 0, 0], none, 1, Grid::Flat(w / 4), GLUE);
    }
    // the output norm: one scale for every lane
    b.op(&p.rms, &[&wk.xb, &m.out_w, &wk.xout], [w as u32, 0, 0, 0], [a.out_eps * a.out_eps, 0.0, 0.0, 0.0], 1, Grid::Rows, GLUE);
    if b.dyns.len() > MAX_DYN {
        return None;
    }
    let attn_name = match per_head {
        Some((_, _, name)) => format!("{name} above {head_rows} rows, the shader cores below"),
        None => "the shader cores".to_string(),
    };
    // the rows up to which `ksplit` gives two slices or more
    let k_rows = if split.is_some() { (k_tiles as usize / 2) * 64 } else { 0 };
    Some(Plan { steps: b.steps, dyns: b.dyns, k_rows, head_rows, names: (gemm_name, attn_name) })
}

/// The front end as a list of dispatches, from the features in `fw.fin` to
/// the stacks' input rows in `wk.xb`. Its folds count mel frames: 1, 2 and 4
/// of them a frame of the volume after each stride.
#[allow(clippy::too_many_arguments)]
fn build_front(c: &Ctx, p: &Pipes, fp: &FrontPipes, a: &GigaamArgs, fr: &GaFront, dev: &FrontDev, fw: &FrontWork, wk: &Work) -> Option<Plan> {
    const DEPTHWISE: u32 = 1;
    const SILU: u32 = 2;
    const ADD: u32 = 4;
    const ROWS: u32 = 8;
    let (gemm, gemm_name) = gemm_of(c);
    let k_tiles = k_tiles(c);
    let split = (k_tiles >= 2).then_some((&p.fold, &wk.part, k_tiles));
    let mut b = Builder { c, dynb: &fw.dynb, gemm, split, when: When::Always, steps: Vec::new(), dyns: Vec::new(), fixed: Default::default() };
    let n_mel = fr.n_mel as u32;
    let (f2, f3) = front_bins(fr.n_mel);
    let (f2, f3) = (f2 as u32, f3 as u32);
    let (c0, c1, c2, wide) = (dev.conv[0].2 as u32, dev.conv[1].2 as u32, dev.conv[2].2 as u32, dev.pw1.rows as u32);
    let pad = |pt: u32, pf: u32| pt | pf << 16;
    // (x, weight, bias, added, y) and `CP`: ci, co, f, fo, k, stride, pad, flags
    let mut conv = |bufs: [&wgpu::Buffer; 5], cp: [u32; 8], fold: usize| b.op_words(&fp.conv2d, &bufs, cp, fold, Grid::Flat((cp[1] * cp[3]) as usize), FRONT);
    conv([&fw.fin, &dev.conv[0].0, &dev.conv[0].1, &fw.fin, &fw.a1], [1, c0, n_mel, n_mel, 3, 1, pad(2, 1), SILU], 1);
    conv([&fw.a1, &dev.conv[1].0, &dev.conv[1].1, &fw.a1, &fw.a2], [c0, c1, n_mel, f2, 3, 2, pad(1, 0), SILU], 2);
    conv([&fw.a2, &dev.conv[2].0, &dev.conv[2].1, &fw.a2, &fw.a3], [c1, c2, f2, f3, 3, 2, pad(1, 0), SILU], 4);
    // ConvNeXt: depthwise 3×3, 1×1 up with SiLU, 1×1 down added to the block
    // input — laid as rows for the projection
    conv([&fw.a3, &dev.dw.0, &dev.dw.1, &fw.a3, &fw.a4], [c2, c2, f3, f3, 3, 1, pad(2, 1), DEPTHWISE], 4);
    conv([&fw.a4, &dev.pw1.w, &dev.pw1.b, &fw.a4, &fw.up], [c2, wide, f3, f3, 1, 1, 0, SILU], 4);
    conv([&fw.up, &dev.pw2.w, &dev.pw2.b, &fw.a3, &fw.a4], [wide, c2, f3, f3, 1, 1, 0, ADD | ROWS], 4);
    let steps = b.steps.len();
    b.mm(&dev.out, &fw.a4, &wk.xb, 4);
    b.op(&p.act, &[&wk.xb, &dev.out.b], [(a.width / 4) as u32, 0, 0, 0], [0f32; 4], 4, Grid::Flat(a.width / 4), FRONT);
    for step in &mut b.steps[steps..] {
        step.kind = FRONT;
    }
    if b.dyns.len() > FRONT_DYN {
        return None;
    }
    let k_rows = if split.is_some() { (k_tiles as usize / 2) * 64 } else { 0 };
    Some(Plan { steps: b.steps, dyns: b.dyns, k_rows, head_rows: 0, names: (gemm_name, String::new()) })
}

/// The voice-activity model as a list of dispatches, from the features in
/// `wk.fin` to a logit a step in `wk.logit` (the head's bias is the host's to
/// add). Its folds count the mel frames of a window. Plain f32 throughout.
fn build_vad(c: &Ctx, p: &Pipes, vp: &VadPipes, a: &GaVadArgs, m: &VadDev, wk: &VadWork) -> Option<Plan> {
    let (d, nh) = (a.dim, a.heads);
    let hd = d / nh;
    let (d4, nh32, hd32) = ((d / 4) as u32, nh as u32, hd as u32);
    let scale = (hd as f32).powf(-0.5);
    let none = [0f32; 4];
    let mut b = Builder { c, dynb: &wk.dynb, gemm: &c.gemm_nt_f32t, split: None, when: When::Always, steps: Vec::new(), dyns: Vec::new(), fixed: Default::default() };
    // the two strided convolutions: patches, a product, the bias and ReLU
    let mut src = &wk.fin;
    for (pre, (dst, fold)) in m.pre.iter().zip([(&wk.y1, 2), (&wk.r, 4)]) {
        let ci = pre.cols / 3;
        b.op(&vp.patch, &[src, &wk.patch], [ci as u32, 0, 0, 0], none, fold, Grid::Flat(3 * ci), GLUE);
        b.mm(pre, &wk.patch, dst, fold);
        b.op(&p.act, &[dst, &pre.b], [(pre.rows / 4) as u32, 2, 0, 0], none, fold, Grid::Flat(pre.rows / 4), GLUE);
        src = dst;
    }
    let ln = |b: &mut Builder, src: &wgpu::Buffer, nb: &wgpu::Buffer, dst: &wgpu::Buffer, silu: u32| {
        b.op(&vp.ln, &[src, nb, dst], [d as u32, silu, 0, 0], [1e-5, 0.0, 0.0, 0.0], 4, Grid::Rows, GLUE)
    };
    // r += weight · (y + bias)
    let add = |b: &mut Builder, r: &wgpu::Buffer, y: &wgpu::Buffer, bias: &wgpu::Buffer, weight: f32| {
        b.op(&vp.axpb, &[r, y, bias], [d4, 0, 0, 0], [weight, 0.0, 0.0, 0.0], 4, Grid::Flat(d / 4), GLUE)
    };
    // a feed-forward module at half weight
    let ff = |b: &mut Builder, r: &wgpu::Buffer, nb: &wgpu::Buffer, f: &(LinDev, LinDev)| {
        let inter = f.0.rows;
        ln(b, r, nb, &wk.xn, 0);
        b.mm(&f.0, &wk.xn, &wk.up, 4);
        b.op(&p.act, &[&wk.up, &f.0.b], [(inter / 4) as u32, 1, 0, 0], none, 4, Grid::Flat(inter / 4), GLUE);
        b.mm(&f.1, &wk.up, &wk.tmp, 4);
        add(b, r, &wk.tmp, &f.1.b, 0.5);
    };
    // the stream, and the rows a layer's output norm writes
    let (mut r, mut spare) = (&wk.r, &wk.r2);
    for l in &m.layers {
        ff(&mut b, r, &l.norm_ff1, &l.ff1);
        // attention inside each window: queries and keys from the rotated
        // rows, values from the rows themselves
        ln(&mut b, r, &l.norm_att, &wk.xn, 0);
        b.op(&vp.rot, &[&wk.xn, &m.cs, &wk.rot], [nh32, hd32, a.cos.len() as u32, 0], none, 4, Grid::Flat(nh * hd / 2), GLUE);
        for (lin, x, plane) in [(&l.q, &wk.rot, &wk.qp), (&l.k, &wk.rot, &wk.kp), (&l.v, &wk.xn, &wk.vp)] {
            b.mm(lin, x, &wk.tmp, 4);
            b.op(&vp.planes, &[&wk.tmp, &lin.b, plane], [nh32, hd32, 0, 0], none, 4, Grid::Flat(d), GLUE);
        }
        b.ext(&c.dit_qk, &[(0, &wk.qp), (1, &wk.kp), (2, &wk.sc)], 3, Dyn::Qk { hd: hd32, scale }, 4, Grid::Square(nh32), ATTENTION);
        b.ext(&c.dit_softmax, &[(2, &wk.sc)], 3, Dyn::Sm { hd: hd32 }, 4, Grid::Sm(nh32), ATTENTION);
        b.ext(&c.dit_pv, &[(0, &wk.sc), (1, &wk.vp), (2, &wk.qp)], 3, Dyn::Pv { hd: hd32 }, 4, Grid::Pv(hd32.div_ceil(64), nh32), ATTENTION);
        b.op(&vp.planes, &[&wk.qp, &l.q.b, &wk.xn], [nh32, hd32, 1, 0], none, 4, Grid::Flat(d), GLUE);
        b.mm(&l.o, &wk.xn, &wk.tmp, 4);
        add(&mut b, r, &wk.tmp, &l.o.b, 1.0);
        // the convolution module: 1×1 up, GLU, depthwise over time, norm and
        // SiLU, 1×1 down
        ln(&mut b, r, &l.norm_conv, &wk.xn, 0);
        b.mm(&l.pw1, &wk.xn, &wk.up, 4);
        b.op(&vp.glu, &[&wk.up, &l.cw, &l.pw1.b, &wk.tmp], [d as u32, a.kernel as u32, 0, 0], none, 4, Grid::Flat(d), GLUE);
        ln(&mut b, &wk.tmp, &l.conv_norm, &wk.xn, 1);
        b.mm(&l.pw2, &wk.xn, &wk.tmp, 4);
        add(&mut b, r, &wk.tmp, &l.pw2.b, 1.0);
        ff(&mut b, r, &l.norm_ff2, &l.ff2);
        // the layer's output norm into the spare rows: the stream from here on
        ln(&mut b, r, &l.norm_out, spare, 0);
        std::mem::swap(&mut r, &mut spare);
    }
    b.mm(&m.head, r, &wk.logit, 4);
    if b.dyns.len() > VAD_DYN {
        return None;
    }
    Some(Plan { steps: b.steps, dyns: b.dyns, k_rows: 0, head_rows: 0, names: ("f32 tiles", "the shader cores".to_string()) })
}

/// Says why a path declined, once a path (`said` is its flag).
fn declined(said: &std::sync::atomic::AtomicBool, path: &str, what: &str) -> bool {
    if !said.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!("gigaam {path} (wgpu) declined: {what}");
    }
    false
}

fn decline(what: &str) -> bool {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    declined(&SAID, "encoder", what)
}

fn decline_front(what: &str) -> bool {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    declined(&SAID, "front end", what)
}

fn decline_vad(what: &str) -> bool {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    declined(&SAID, "voice activity", what)
}

/// The encoder stacks over one segment `x` (`[t, width]`) in place;
/// `false` = declined, the caller runs its own path.
pub fn encode(a: &GigaamArgs, x: &mut [f32], t: usize) -> bool {
    run(a, x, t, None)
}

/// The front end and the stacks of one segment: `feats` (`[frames, n_mel]`)
/// in, `out` (`[t, width]`) the encoder output; `false` = declined (the
/// caller runs the front end on the host and calls [`encode`]).
pub fn encode_feats(a: &GigaamArgs, feats: &[f32], frames: usize, out: &mut [f32]) -> bool {
    if a.front.is_none() {
        return false;
    }
    run(a, out, frames.div_ceil(4), Some((feats, frames)))
}

/// The stacks over the `t` rows of `x` in place — or, with `feats` (the
/// log-mel rows and how many), the front end over them first, its rows never
/// leaving the device, and the encoder output into `x`.
fn run(a: &GigaamArgs, x: &mut [f32], t: usize, feats: Option<(&[f32], usize)>) -> bool {
    let Some(c) = ctx() else { return false };
    if !shapes_ok(a, x, t) {
        return decline("an encoder shape outside the kernels");
    }
    let front = match (feats, a.front.as_ref()) {
        (Some((fe, frames)), Some(fr)) if front_ok(fr, a.width, fe, frames) => Some((fr, fe, frames)),
        (Some(_), _) => return decline_front("a shape outside the kernels"),
        (None, _) => None,
    };
    let Some(p) = pipes(c) else { return false };
    let fp = match front {
        Some(_) => match front_pipes(c) {
            Some(fp) => Some(fp),
            None => return false,
        },
        None => None,
    };
    let w = a.width;
    let _op = c.op_gate.enter();
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    // the cold part: the weights, the working buffers, the plans
    let other = |s: &State| s.model.key != a.key || s.model.stacks.len() != a.stacks.len();
    let no_front = |s: &State, frames: usize| s.front.as_ref().is_none_or(|f| f.plan.is_none() || f.work.as_ref().is_none_or(|fw| fw.cap < frames));
    let cold = g.as_ref().is_none_or(|s| {
        other(s) || s.work.as_ref().is_none_or(|wk| wk.cap < t) || s.plan.is_none() || front.is_some_and(|(_, _, frames)| no_front(s, frames))
    });
    if cold {
        let tc = std::time::Instant::now();
        let oom = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let val = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        if g.as_ref().is_none_or(other) {
            *g = None;
            let lim = c.device.limits();
            let max_bind = (lim.max_storage_buffer_binding_size as u64).min(lim.max_buffer_size);
            let lins = a.stacks.iter().flat_map(|st| {
                let per = st.layers.iter().flat_map(|l| {
                    let conv = l.conv.iter().flat_map(|cv| [&cv.inp, &cv.out]);
                    let ff2 = l.ff2.iter().flat_map(|f| [&f.0, &f.1]);
                    [&l.ff1_in, &l.ff1_out, &l.qk, &l.vg, &l.attn_out].into_iter().chain(conv).chain(ff2)
                });
                [&st.proj, &st.proj_t].into_iter().chain(per)
            });
            let planes: Vec<u64> = lins.map(|l| (l.rows * l.cols * 4) as u64).collect();
            if planes.iter().any(|bytes| *bytes > max_bind) || super::resident_bytes() + planes.iter().sum::<u64>() > c.vram_budget {
                drop((val, oom));
                return decline("the weights do not fit the device's buffers or its budget");
            }
            let (model, bytes) = upload(c, a);
            if a.prof {
                eprintln!("gigaam (wgpu): weights of {:.0} MB to the device, {:.0} ms", bytes as f64 / 1048576.0, tc.elapsed().as_secs_f64() * 1e3);
            }
            *g = Some(State { model, work: None, plan: None, front: None });
        }
        let st = g.as_mut().expect("set above");
        if st.work.as_ref().is_none_or(|wk| wk.cap < t) {
            // the plans bind the buffers: new ones with them
            st.plan = None;
            if let Some(f) = st.front.as_mut() {
                f.plan = None;
            }
            // a 30 s segment is 750 frames: one size for every segment of a recording
            st.work = alloc(c, a, t.max(750));
        }
        if st.plan.is_none() {
            st.plan = st.work.as_ref().and_then(|wk| build(c, p, a, &st.model, wk));
        }
        if let (Some((fr, _, frames)), Some(fp)) = (front, fp) {
            let f = st.front.get_or_insert_with(|| Front { dev: upload_front(c, fr), work: None, plan: None });
            if f.work.as_ref().is_none_or(|fw| fw.cap < frames) {
                f.plan = None;
                // and 3000 mel frames at most
                f.work = alloc_front(c, fr, frames.max(3000));
            }
            if f.plan.is_none() {
                f.plan = match (f.work.as_ref(), st.work.as_ref()) {
                    (Some(fw), Some(wk)) => build_front(c, p, fp, a, fr, &f.dev, fw, wk),
                    _ => None,
                };
            }
        }
        let bad_val = pollster::block_on(val.pop());
        let bad_oom = pollster::block_on(oom.pop());
        if let (true, Some(wk), Some(plan)) = (a.prof, st.work.as_ref(), st.plan.as_ref()) {
            let front_steps = st.front.as_ref().and_then(|f| f.plan.as_ref()).map_or(0, |fpl| fpl.steps.len());
            eprintln!(
                "gigaam (wgpu): buffers for {} frames and a plan of {} steps ({front_steps} more of the front end), {:.0} ms from the start",
                wk.cap,
                plan.steps.len(),
                tc.elapsed().as_secs_f64() * 1e3
            );
        }
        if bad_val.is_some() || bad_oom.is_some() || st.plan.is_none() {
            *g = None;
            return decline(&match (bad_val, bad_oom) {
                (Some(e), _) => format!("{e}"),
                (_, Some(e)) => format!("{e}"),
                _ => "a segment past the device's buffer size".to_string(),
            });
        }
        if front.is_some_and(|(_, _, frames)| no_front(st, frames)) {
            // the stacks stay: the caller's front end feeds them
            st.front = None;
            return decline_front("a segment past the device's buffer size");
        }
    }
    let st = g.as_ref().expect("set above");
    let (Some(wk), Some(plan)) = (st.work.as_ref(), st.plan.as_ref()) else { return false };
    // the front end's buffers and plan with the features and their frames
    let front = match front {
        Some((_, fe, frames)) => match st.front.as_ref().map(|f| (f.work.as_ref(), f.plan.as_ref())) {
            Some((Some(fw), Some(fplan))) => Some((fw, fplan, fe, Span { t: frames, mult: 1 })),
            _ => return false,
        },
        None => None,
    };
    let sp = Span { t, mult: 1 };
    let t0 = std::time::Instant::now();
    match front {
        Some((fw, fplan, fe, fsp)) => {
            c.queue.write_buffer(&fw.fin, 0, bytemuck::cast_slice(fe));
            c.queue.write_buffer(&fw.dynb, 0, bytemuck::cast_slice(&fplan.uniforms(fsp)));
            // the front end writes the frames' rows; the row after them (the
            // pair of an odd last frame) is zero
            c.queue.write_buffer(&wk.xb, (t * w * 4) as u64, bytemuck::cast_slice(&vec![0f32; w]));
        }
        None => {
            // the frames and one zero row after them (the pair of an odd last frame)
            let mut rows = Vec::with_capacity((t + 1) * w);
            rows.extend_from_slice(x);
            rows.resize((t + 1) * w, 0.0);
            c.queue.write_buffer(&wk.xb, 0, bytemuck::cast_slice(&rows));
        }
    }
    c.queue.write_buffer(&wk.dynb, 0, bytemuck::cast_slice(&plan.uniforms(sp)));
    // the plans' steps (all, or one kind of them) into a pass of an encoder
    let record = |only: Option<usize>| {
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ga") });
        let mut pass: Option<wgpu::ComputePass<'static>> = None;
        let mut count = (0usize, 0usize);
        if let Some((_, fplan, _, fsp)) = front {
            fplan.replay(fsp, only, &mut enc, &mut pass, &mut count);
        }
        plan.replay(sp, only, &mut enc, &mut pass, &mut count);
        drop(pass.take());
        (enc, count.0, count.1)
    };
    let (enc, passes, steps) = record(None);
    let t_enc = t0.elapsed().as_secs_f64() * 1e3;
    if !super::readback(c, enc, &wk.xout, &wk.stage, (t * w * 4) as u64, x) {
        *g = None;
        return decline("the readback failed");
    }
    let total = t0.elapsed().as_secs_f64() * 1e3;
    if !x.iter().all(|v| v.is_finite()) {
        return decline("the encoder came back with a value that is not finite");
    }
    if a.prof {
        let from = front.map_or(String::new(), |(_, _, _, fsp)| format!(" from {} mel frames", fsp.t));
        eprintln!(
            "gigaam (wgpu, resident): {t} frames{from}, {total:.1} ms (recorded in {t_enc:.2} ms: {steps} dispatches, {passes} pass(es), 1 submit; \
             products on {}, attention on {})",
            plan.names.0,
            plan.names.1,
        );
        // `CMF_GIGAAM_STAGES=1`: the same segment again, one kind of step a
        // submit (the buffers hold the finished segment: the numbers are only
        // timed, never read), and an empty submit for the price of the wait.
        if std::env::var("CMF_GIGAAM_STAGES").as_deref() == Ok("1") {
            let mut alone = [(0f64, 0usize); 5];
            for (kind, slot) in alone.iter_mut().enumerate() {
                let ts = std::time::Instant::now();
                let (enc, _, steps) = record(Some(kind));
                super::submit(c, enc.finish());
                let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
                *slot = (ts.elapsed().as_secs_f64() * 1e3, steps);
            }
            let [pr, at, gl, fe, empty] = alone;
            eprintln!(
                "gigaam (wgpu) stages, each alone: products {:.1} ms ({}) | attention {:.1} ms ({}) | glue {:.1} ms ({}) | front end {:.1} ms ({}) | \
                 an empty submit {:.1} ms",
                pr.0, pr.1, at.0, at.1, gl.0, gl.1, fe.0, fe.1, empty.0,
            );
        }
    }
    true
}

/// The voice-activity logits of `windows` windows of `frames` mel frames
/// (`feats` is `[windows·frames, n_mel]`) into `out`; `false` = declined
/// (the caller runs the host model).
pub fn vad(a: &GaVadArgs, feats: &[f32], windows: usize, frames: usize, out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    if !vad_ok(a, feats, windows, frames, out) {
        return decline_vad("a shape outside the kernels");
    }
    let (Some(p), Some(vp)) = (pipes(c), vad_pipes(c)) else { return false };
    let (n_mel, steps) = (a.n_mel, frames.div_ceil(4));
    // The windows of a chunk: what the device's bindings hold of the widest
    // buffer (the scores, as a rule) and its grids of (window, head) on an
    // axis — or fewer, when asked.
    let lim = c.device.limits();
    let max_bind = (lim.max_storage_buffer_binding_size as u64).min(lim.max_buffer_size);
    let widest = vad_sizes(a, 1, frames).into_iter().max().unwrap_or(1) as u64;
    let asked = std::env::var("CMF_GIGAAM_VAD_WINDOWS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|v| *v > 0);
    let axis = lim.max_compute_workgroups_per_dimension.min(65_535) as usize;
    let most = ((max_bind / 4 / widest) as usize).min(axis / a.heads).min(asked.unwrap_or(usize::MAX));
    if most == 0 {
        return decline_vad("a window past the device's buffer size");
    }
    let chunk = windows.min(most);
    let chunks = windows.div_ceil(chunk);
    let _op = c.op_gate.enter();
    let mut g = VAD.lock().unwrap_or_else(|e| e.into_inner());
    // the cold part: the weights, the working buffers, the plan
    let other = |s: &VadState| s.key != a.key || s.dev.layers.len() != a.layers.len();
    let fits = |wk: &VadWork| wk.windows >= chunk && wk.frames >= frames;
    let no_io = |s: &VadState| chunks > 1 && s.io.as_ref().is_none_or(|io| io.feats < feats.len() || io.logits < out.len());
    let cold = g.as_ref().is_none_or(|s| other(s) || !s.work.as_ref().is_some_and(fits) || s.plan.is_none() || no_io(s));
    if cold {
        let oom = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let val = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        if g.as_ref().is_none_or(other) {
            *g = None;
            *g = Some(VadState { key: a.key, dev: upload_vad(c, a), work: None, plan: None, io: None });
        }
        let st = g.as_mut().expect("set above");
        if !st.work.as_ref().is_some_and(fits) {
            // the plan binds the buffers: a new one with them. They only
            // grow, where the device takes both sizes at once.
            st.plan = None;
            let (had_w, had_f) = st.work.take().map_or((0, 0), |wk| (wk.windows, wk.frames));
            st.work = alloc_vad(c, a, chunk.max(had_w), frames.max(had_f)).or_else(|| alloc_vad(c, a, chunk, frames));
        }
        if st.plan.is_none() {
            st.plan = st.work.as_ref().and_then(|wk| build_vad(c, p, vp, a, &st.dev, wk));
        }
        if no_io(st) {
            let copy = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
            st.io = None;
            st.io = Some(VadIo {
                feats: feats.len(),
                logits: out.len(),
                fall: buffer(c, feats.len(), copy),
                oall: buffer(c, out.len(), copy),
                stage: buffer(c, out.len(), wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            });
        }
        let bad_val = pollster::block_on(val.pop());
        let bad_oom = pollster::block_on(oom.pop());
        if bad_val.is_some() || bad_oom.is_some() || st.plan.is_none() {
            *g = None;
            return decline_vad(&match (bad_val, bad_oom) {
                (Some(e), _) => format!("{e}"),
                (_, Some(e)) => format!("{e}"),
                _ => "a batch past the device's buffer size".to_string(),
            });
        }
    }
    let st = g.as_ref().expect("set above");
    let (Some(wk), Some(plan)) = (st.work.as_ref(), st.plan.as_ref()) else { return false };
    let io = match (chunks > 1, st.io.as_ref()) {
        (true, Some(io)) => Some(io),
        (true, None) => return false,
        (false, _) => None,
    };
    let t0 = std::time::Instant::now();
    // Every chunk is as many windows as the first — the last one starts
    // earlier where they do not divide (a window is on its own: its logits
    // come out the same, twice) — so one set of uniforms serves them all.
    let sp = Span { t: frames, mult: chunk };
    c.queue.write_buffer(io.map_or(&wk.fin, |io| &io.fall), 0, bytemuck::cast_slice(feats));
    c.queue.write_buffer(&wk.dynb, 0, bytemuck::cast_slice(&plan.uniforms(sp)));
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ga-vad") });
    let mut count = (0usize, 0usize);
    for i in 0..chunks {
        let w0 = (i * chunk).min(windows - chunk);
        if let Some(io) = io {
            enc.copy_buffer_to_buffer(&io.fall, (w0 * frames * n_mel * 4) as u64, &wk.fin, 0, (chunk * frames * n_mel * 4) as u64);
        }
        let mut pass: Option<wgpu::ComputePass<'static>> = None;
        plan.replay(sp, None, &mut enc, &mut pass, &mut count);
        drop(pass.take());
        if let Some(io) = io {
            enc.copy_buffer_to_buffer(&wk.logit, 0, &io.oall, (w0 * steps * 4) as u64, (chunk * steps * 4) as u64);
        }
    }
    let (logits, stage) = io.map_or((&wk.logit, &wk.stage), |io| (&io.oall, &io.stage));
    if !super::readback(c, enc, logits, stage, (out.len() * 4) as u64, out) {
        *g = None;
        return decline_vad("the readback failed");
    }
    // the head's bias: one float, added here
    if let Some(bias) = a.head.bias.first() {
        out.iter_mut().for_each(|v| *v += bias);
    }
    if !out.iter().all(|v| v.is_finite()) {
        return decline_vad("the model came back with a value that is not finite");
    }
    if std::env::var("CMF_GIGAAM_PROF").as_deref() == Ok("1") {
        eprintln!(
            "gigaam vad (wgpu, resident): {windows} window(s) of {frames} mel frames, {:.1} ms ({} dispatches, {} pass(es), {chunks} chunk(s) of {chunk}, 1 submit)",
            t0.elapsed().as_secs_f64() * 1e3,
            count.1,
            count.0,
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_validate() {
        use wgpu::naga;
        let kernels = [
            ("ga_bias_act", ACT_SRC),
            ("ga_add3", ADD3_SRC),
            ("ga_acc", ACC_SRC),
            ("ga_fold", FOLD_SRC),
            ("ga_fork", FORK_SRC),
            ("ga_rms", RMS_SRC),
            ("ga_tail", TAIL_SRC),
            ("ga_split", SPLIT_SRC),
            ("ga_gate", GATE_SRC),
            ("ga_mean", MEAN_SRC),
            ("ga_conv", CONV_SRC),
            ("ga_nt_ref", NTREF_SRC),
            ("ga_conv2d", CONV2D_SRC),
            ("ga_patch1d", PATCH_SRC),
            ("ga_ln", LN_SRC),
            ("ga_rot_half", ROT_SRC),
            ("ga_planes", PLANES_SRC),
            ("ga_glu_conv", GLU_SRC),
            ("ga_axpb", AXPB_SRC),
        ];
        for (name, src) in kernels {
            let src = format!("{PRE}{src}");
            let m = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&m)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
        }
    }

    #[test]
    fn rows_follow_the_strides() {
        // a 30 s segment through the front end, then a stack of frame pairs
        let mel = Span { t: 2998, mult: 1 };
        assert_eq!((mel.rows(1), mel.rows(2), mel.rows(4)), (2998, 1499, 750));
        assert_eq!(Dyn::Rows.words(mel, 4)[..4], [750, 750, 1499, 1]);
        assert_eq!(Span { t: 750, mult: 1 }.rows(2), 375);
        // 64 voice-activity windows of 10 s: the rows of all, the steps of one
        let vad = Span { t: 1000, mult: 64 };
        assert_eq!(Dyn::Rows.words(vad, 2)[..4], [32_000, 500, 1000, 64]);
        assert_eq!(Dyn::Rows.words(vad, 4)[..4], [16_000, 250, 500, 64]);
        assert_eq!(Dyn::Qk { hd: 32, scale: 1.0 }.words(vad, 4)[..3], [250, 32, 250]);
        assert_eq!(Grid::Square(4).dims(vad, 4), (4, 4, 256));
        assert_eq!(Grid::Sm(4).dims(vad, 4), (250, 256, 1));
        assert_eq!(Grid::Tile(128).dims(vad, 4), (2, 250, 1));
        // a short last window: 37 mel frames
        let tail = Span { t: 37, mult: 1 };
        assert_eq!(Dyn::Rows.words(tail, 4)[..4], [10, 10, 19, 1]);
        assert_eq!(front_bins(64), (31, 15));
    }

    struct Plane(Vec<f32>);

    impl crate::gpu::GaWeight for Plane {
        fn fill(&self, dst: &mut [f32]) {
            dst.copy_from_slice(&self.0);
        }
    }

    fn rnd(st: &mut u64, n: usize, a: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *st ^= *st << 13;
                *st ^= *st >> 7;
                *st ^= *st << 17;
                ((*st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * a
            })
            .collect()
    }

    fn silu(x: &mut [f32]) {
        x.iter_mut().for_each(|v| *v /= 1.0 + (-*v).exp());
    }

    /// A k×k convolution as `gpu::GaFront` words it, the direct sum: x
    /// `[ci][t][f]` to `[co][⌈t / stride⌉][fo]`, `pt` zero frames before the
    /// first and `pf` zero bins on both sides; (the volume, its frames, its bins).
    #[allow(clippy::too_many_arguments)]
    fn conv(
        x: &[f32],
        (ci, t, f): (usize, usize, usize),
        w: &[f32],
        b: &[f32],
        co: usize,
        k: usize,
        stride: usize,
        (pt, pf): (usize, usize),
        depthwise: bool,
    ) -> (Vec<f32>, usize, usize) {
        let (to, fo) = (t.div_ceil(stride), (f + 2 * pf - k) / stride + 1);
        let mut y = vec![0f32; co * to * fo];
        for c in 0..co {
            for tt in 0..to {
                for ff in 0..fo {
                    let mut acc = b[c];
                    for cc in if depthwise { c..c + 1 } else { 0..ci } {
                        let kernel = if depthwise { c } else { c * ci + cc };
                        let wk = &w[kernel * k * k..][..k * k];
                        for kt in (0..k).filter(|kt| tt * stride + kt >= pt && tt * stride + kt - pt < t) {
                            for kf in (0..k).filter(|kf| ff * stride + kf >= pf && ff * stride + kf - pf < f) {
                                acc += wk[kt * k + kf] * x[(cc * t + tt * stride + kt - pt) * f + ff * stride + kf - pf];
                            }
                        }
                    }
                    y[(c * to + tt) * fo + ff] = acc;
                }
            }
        }
        (y, to, fo)
    }

    /// A small front end with no stack behind it — the output norm alone —
    /// against the direct sums on the host: a short odd segment after a long
    /// one that crosses the tiles and takes the whole-K product.
    #[test]
    fn front_end_matches_host() {
        let Some(c) = ctx() else { return };
        let (n_mel, c0, c1, c2, wide, w) = (16usize, 4usize, 8usize, 8usize, 16usize, 32usize);
        let (f2, f3) = front_bins(n_mel);
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let wts = [rnd(&mut st, c0 * 9, 0.6), rnd(&mut st, c1 * c0 * 9, 0.3), rnd(&mut st, c2 * c1 * 9, 0.3), rnd(&mut st, c2 * 9, 0.6)];
        // the biases: the three convolutions, the depthwise one, the pointwise pair, the projection
        let bias = [c0, c1, c2, c2, wide, c2, w].map(|n| rnd(&mut st, n, 0.2));
        let (pw1, pw2, out) = (Plane(rnd(&mut st, wide * c2, 0.7)), Plane(rnd(&mut st, c2 * wide, 0.5)), Plane(rnd(&mut st, w * c2 * f3, 0.4)));
        let zeros = vec![0f32; 256 * w / 2];
        let front = GaFront {
            n_mel,
            conv: [(&wts[0], &bias[0], c0), (&wts[1], &bias[1], c1), (&wts[2], &bias[2], c2)],
            dw: (&wts[3], &bias[3]),
            pw1: GaLin { rows: wide, cols: c2, weight: &pw1, bias: &bias[4] },
            pw2: GaLin { rows: c2, cols: wide, weight: &pw2, bias: &bias[5] },
            out: GaLin { rows: w, cols: c2 * f3, weight: &out, bias: &bias[6] },
        };
        let (key, stacks) = (0x6761_6672_6f6e_7400, Vec::new());
        let a = GigaamArgs { key, width: w, head_dim: w, stacks, cos: &zeros, sin: &zeros, out_scale: 1.3, out_eps: 0.05, prof: false, front: Some(front) };
        // the matrix units take f16 operands for the projection
        let tol = if gemm_of(c).1 == "f32 tiles" { 1e-4 } else { 4e-3 };
        for frames in [521usize, 37] {
            let feats = rnd(&mut st, frames * n_mel, 4.0);
            let t = frames.div_ceil(4);
            let mut got = vec![0f32; t * w];
            assert!(encode_feats(&a, &feats, frames, &mut got), "the device declined {frames} frames");
            // the host: three convolutions with SiLU, the ConvNeXt block, the projection, the norm
            let (mut x, t1, _) = conv(&feats, (1, frames, n_mel), &wts[0], &bias[0], c0, 3, 1, (2, 1), false);
            silu(&mut x);
            let (mut x, t2, _) = conv(&x, (c0, t1, n_mel), &wts[1], &bias[1], c1, 3, 2, (1, 0), false);
            silu(&mut x);
            let (mut x, t3, _) = conv(&x, (c1, t2, f2), &wts[2], &bias[2], c2, 3, 2, (1, 0), false);
            silu(&mut x);
            assert_eq!(t3, t);
            let (y, ..) = conv(&x, (c2, t, f3), &wts[3], &bias[3], c2, 3, 1, (2, 1), true);
            let (mut h, ..) = conv(&y, (c2, t, f3), &pw1.0, &bias[4], wide, 1, 1, (0, 0), false);
            silu(&mut h);
            let (z, ..) = conv(&h, (wide, t, f3), &pw2.0, &bias[5], c2, 1, 1, (0, 0), false);
            let mut worst = 0f32;
            for tt in 0..t {
                let flat: Vec<f32> = (0..c2 * f3).map(|i| x[(i / f3 * t + tt) * f3 + i % f3] + z[(i / f3 * t + tt) * f3 + i % f3]).collect();
                let row: Vec<f32> = (0..w).map(|j| bias[6][j] + flat.iter().zip(&out.0[j * c2 * f3..]).map(|(p, q)| p * q).sum::<f32>()).collect();
                let k = a.out_scale / (row.iter().map(|v| v * v).sum::<f32>() / w as f32 + a.out_eps * a.out_eps).sqrt();
                for (j, v) in row.iter().enumerate() {
                    worst = worst.max((v * k - got[tt * w + j]).abs());
                }
            }
            assert!(worst < tol, "{frames} frames: the device is {worst} away from the host");
        }
    }
}
