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

use super::{Ctx, ctx};
use crate::gpu::{GaLin, GigaamArgs};
use std::sync::{Mutex, OnceLock};

/// `P`: an op's own numbers, fixed for the model. `R`: the rows of the op's
/// stack in this segment.
const PRE: &str = r#"
struct P { a: u32, b: u32, c: u32, d: u32, x: f32, y: f32, z: f32, w: f32 }
struct R { n: u32, r1: u32, r2: u32, r3: u32 }
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-clamp(v, -80.0, 80.0))); }
"#;

// y += b a row of 4·p.a lanes; p.b = 1: SiLU after.
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

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
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
            let p = Pipes {
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
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("gigaam kernels rejected: {e}");
                return None;
            }
            Some(p)
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

/// The uniform slots a plan may take.
const MAX_DYN: usize = 1024;

/// A uniform that follows the segment: its words from the rows `n` of its stack.
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
    /// `DitP` of `dit_qk` / `dit_softmax` / `dit_pv`, the heads on z
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
    fn words(self, n: u32) -> [u32; 8] {
        let one = 1f32.to_bits();
        match self {
            Dyn::Rows => [n, 0, 0, 0, 0, 0, 0, 0],
            Dyn::Mm { cols4, rows } => [cols4, rows, n, 0, 0, 0, 0, 0],
            Dyn::MmK { cols4, rows, tiles } => [cols4, rows, n, ksplit(n, cols4 * 4, tiles).0, 0, 0, 0, 0],
            Dyn::Fold { cols4, tiles } => [n, ksplit(n, cols4 * 4, tiles).1, 0, 0, 0, 0, 0, 0],
            Dyn::Qk { hd, scale } => [n, hd, n, scale.to_bits(), 0, 0, 1, n],
            Dyn::Sm { hd } => [n, hd, n, one, 0, 0, 1, n],
            Dyn::Pv { hd } => [n, n, hd, one, 0, 0, 1, n],
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

/// A dispatch grid from the rows `n` of the op's stack.
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
    /// 64×64 tiles of `[n][n]`, the heads on z
    Square(u32),
    /// a workgroup a row of `[heads][n]`
    Sm(u32),
    /// `(x, n / 64, heads)`
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
    fn dims(self, n: usize) -> (u32, u32, u32) {
        let t64 = (n as u32).div_ceil(64).max(1);
        match self {
            Grid::Flat(per) => {
                let (x, y) = grid(n * per);
                (x, y, 1)
            }
            Grid::Rows => (n.clamp(1, 65_535) as u32, n.div_ceil(65_535).max(1) as u32, 1),
            Grid::Tile(out) => (out.div_ceil(64), t64, 1),
            Grid::TileK(out, cols, tiles) => (out.div_ceil(64), t64, ksplit(n as u32, cols, tiles).1),
            Grid::Square(z) => (t64, t64, z),
            Grid::Sm(nh) => (n as u32, nh, 1),
            Grid::Pv(x, nh) => (x, t64, nh),
            Grid::Fixed(x) => (x, 1, 1),
        }
    }
}

/// What a step is, for the profile.
const PRODUCT: usize = 0;
const ATTENTION: usize = 1;
const GLUE: usize = 2;

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
    /// frames in a row of the step's stack
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
}

struct State {
    model: Model,
    work: Option<Work>,
    plan: Option<Plan>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

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
        let pb = self.fixed_buf([ints[0], ints[1], ints[2], ints[3], floats[0].to_bits(), floats[1].to_bits(), floats[2].to_bits(), floats[3].to_bits()]);
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

fn decline(what: &str) -> bool {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!("gigaam encoder (wgpu) declined: {what}");
    }
    false
}

/// The encoder stacks over one segment `x` (`[t, width]`) in place;
/// `false` = declined, the caller runs its own path.
pub fn encode(a: &GigaamArgs, x: &mut [f32], t: usize) -> bool {
    let Some(c) = ctx() else { return false };
    if !shapes_ok(a, x, t) {
        return decline("an encoder shape outside the kernels");
    }
    let Some(p) = pipes(c) else { return false };
    let w = a.width;
    let _op = c.op_gate.enter();
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    // the cold part: the weights, the working buffers, the plan
    let other = |s: &State| s.model.key != a.key || s.model.stacks.len() != a.stacks.len();
    let cold = g.as_ref().is_none_or(|s| other(s) || s.work.as_ref().is_none_or(|wk| wk.cap < t) || s.plan.is_none());
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
            *g = Some(State { model, work: None, plan: None });
        }
        let st = g.as_mut().expect("set above");
        if st.work.as_ref().is_none_or(|wk| wk.cap < t) {
            // the plan binds the buffers: a new one with them
            st.plan = None;
            // a 30 s segment is 750 frames: one size for every segment of a recording
            st.work = alloc(c, a, t.max(750));
        }
        if st.plan.is_none() {
            st.plan = st.work.as_ref().and_then(|wk| build(c, p, a, &st.model, wk));
        }
        let bad_val = pollster::block_on(val.pop());
        let bad_oom = pollster::block_on(oom.pop());
        if let (true, Some(wk), Some(plan)) = (a.prof, st.work.as_ref(), st.plan.as_ref()) {
            eprintln!("gigaam (wgpu): buffers for {} frames and a plan of {} steps, {:.0} ms from the start", wk.cap, plan.steps.len(), tc.elapsed().as_secs_f64() * 1e3);
        }
        if bad_val.is_some() || bad_oom.is_some() || st.plan.is_none() {
            *g = None;
            return decline(&match (bad_val, bad_oom) {
                (Some(e), _) => format!("{e}"),
                (_, Some(e)) => format!("{e}"),
                _ => "a segment past the device's buffer size".to_string(),
            });
        }
    }
    let st = g.as_ref().expect("set above");
    let (Some(wk), Some(plan)) = (st.work.as_ref(), st.plan.as_ref()) else { return false };
    let t0 = std::time::Instant::now();
    // the frames and one zero row after them (the pair of an odd last frame)
    let mut rows = Vec::with_capacity((t + 1) * w);
    rows.extend_from_slice(x);
    rows.resize((t + 1) * w, 0.0);
    c.queue.write_buffer(&wk.xb, 0, bytemuck::cast_slice(&rows));
    let mut du = vec![0u32; plan.dyns.len() * 64];
    for (i, (d, fold)) in plan.dyns.iter().enumerate() {
        du[i * 64..i * 64 + 8].copy_from_slice(&d.words(t.div_ceil(*fold) as u32));
    }
    c.queue.write_buffer(&wk.dynb, 0, bytemuck::cast_slice(&du));
    let merge = std::env::var("CMF_PASS_MERGE").as_deref() != Ok("0");
    // the plan's steps (all, or one kind of them) into a pass of an encoder
    let record = |only: Option<usize>| {
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ga") });
        let mut pass: Option<wgpu::ComputePass<'static>> = None;
        let (mut passes, mut steps) = (0usize, 0usize);
        for step in plan.steps.iter().filter(|s| only.is_none_or(|k| s.kind == k)) {
            let n = t.div_ceil(step.fold);
            if !plan.runs(step, n) {
                continue;
            }
            steps += 1;
            if !merge {
                drop(pass.take());
            }
            if pass.is_none() {
                passes += 1;
                pass = Some(enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ga"), timestamp_writes: None }).forget_lifetime());
            }
            let ps = pass.as_mut().expect("set above");
            let (gx, gy, gz) = step.grid.dims(n);
            ps.set_pipeline(&step.pipe);
            ps.set_bind_group(0, &step.bg, &[]);
            ps.dispatch_workgroups(gx, gy, gz);
        }
        drop(pass.take());
        (enc, passes, steps)
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
        eprintln!(
            "gigaam (wgpu, resident): {t} frames, {total:.1} ms (recorded in {t_enc:.2} ms: {steps} dispatches, {passes} pass(es), 1 submit; \
             products on {}, attention on {})",
            plan.names.0,
            plan.names.1,
        );
        // `CMF_GIGAAM_STAGES=1`: the same segment again, one kind of step a
        // submit (the buffers hold the finished segment: the numbers are only
        // timed, never read), and an empty submit for the price of the wait.
        if std::env::var("CMF_GIGAAM_STAGES").as_deref() == Ok("1") {
            let mut alone = [(0f64, 0usize); 4];
            for (kind, slot) in alone.iter_mut().enumerate() {
                let ts = std::time::Instant::now();
                let (enc, _, steps) = record(Some(kind));
                super::submit(c, enc.finish());
                let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
                *slot = (ts.elapsed().as_secs_f64() * 1e3, steps);
            }
            let [pr, at, gl, empty] = alone;
            eprintln!(
                "gigaam (wgpu) stages, each alone: products {:.1} ms ({}) | attention {:.1} ms ({}) | glue {:.1} ms ({}) | an empty submit {:.1} ms",
                pr.0, pr.1, at.0, at.1, gl.0, gl.1, empty.0,
            );
        }
    }
    true
}

/// The front end and the stacks of one segment: `feats` (`[frames, n_mel]`)
/// in, `out` (`[t, width]`) the encoder output; `false` = declined (the
/// caller runs the front end on the host and calls [`encode`]).
pub fn encode_feats(_a: &crate::gpu::GigaamArgs, _feats: &[f32], _frames: usize, _out: &mut [f32]) -> bool {
    false
}

/// The voice-activity logits of `windows` windows of `frames` mel frames
/// (`feats` is `[windows·frames, n_mel]`) into `out`; `false` = declined
/// (the caller runs the host model).
pub fn vad(_a: &crate::gpu::GaVadArgs, _feats: &[f32], _windows: usize, _frames: usize, _out: &mut [f32]) -> bool {
    false
}
