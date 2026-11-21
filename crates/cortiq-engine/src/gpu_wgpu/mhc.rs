//! SenseNova-Loop's mHC streams inside the wgpu token graph
//! (`sensenova::Mhc`, two streams). The residual of an mHC layer is two
//! copies of the hidden, `r = [r0; r1]`; around each sub-layer:
//!
//! - `mhc_route` (one workgroup): optionally copies the incoming hidden into
//!   both streams (the range's first layer), takes the RMS of the flat
//!   `[2·d]` streams, the router's 8 logits (`W·x̂ + b`), `pre = σ`, `post =
//!   2σ`, the 2×2 Sinkhorn (log space, row then column logsumexp per
//!   iteration), and writes the branch `Σ pre_s·r_s · pre_scale` into the
//!   hidden buffer and `post`/`mix` into the coefficient buffer;
//! - `mhc_merge` (one thread per lane): `r_i ← Σ_j mix_ij·r_j + post_i·y`
//!   with `y` the sub-layer's output, and, after the range's last FFN, the
//!   hidden becomes the streams' mean.
//!
//! The streams live only within one token's pass, so their buffer needs no
//! persistence between tokens.

use super::Ctx;
use std::sync::OnceLock;

const ROUTE_SRC: &str = r#"
struct MP { d: u32, s: u32, outn: u32, flags: u32, iters: u32, pre_scale: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read_write> r: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> coef: array<f32>;
@group(0) @binding(5) var<uniform> p: MP;
var<workgroup> red: array<f32, 2304>;
var<workgroup> pre_s: array<f32, 2>;

fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn lse2(a: f32, b: f32) -> f32 { let m = max(a, b); return m + log(exp(a - m) + exp(b - m)); }

@compute @workgroup_size(256)
fn mhc_route(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let d = p.d;
    let n = 2u * d;
    // One workgroup per row of a batch: row offsets into h, r and coef.
    let hb = wid.x * d;
    let rb = wid.x * n;
    let cb = wid.x * 8u;
    if ((p.flags & 1u) != 0u) {
        for (var e = t; e < n; e = e + 256u) { r[rb + e] = h[hb + e % d]; }
    }
    storageBarrier();
    workgroupBarrier();
    // Nine scalar sums, not an indexed array: the array lived in local memory and the
    // loop took ~100 us on an RTX A5000 (SenseNova-Loop-9B, 8192 lanes). Same sums,
    // same order.
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0; var a4 = 0.0;
    var a5 = 0.0; var a6 = 0.0; var a7 = 0.0; var a8 = 0.0;
    // Four lanes' loads in flight before their sums, which keep the lane order: one
    // workgroup reads 288 KB here and waits on each load otherwise.
    var e = t;
    loop {
        if (e + 768u >= n) { break; }
        let x0 = r[rb + e];
        let w00 = w[e];
        let w01 = w[n + e];
        let w02 = w[2u * n + e];
        let w03 = w[3u * n + e];
        let w04 = w[4u * n + e];
        let w05 = w[5u * n + e];
        let w06 = w[6u * n + e];
        let w07 = w[7u * n + e];
        let x1 = r[rb + e + 256u];
        let w10 = w[e + 256u];
        let w11 = w[n + e + 256u];
        let w12 = w[2u * n + e + 256u];
        let w13 = w[3u * n + e + 256u];
        let w14 = w[4u * n + e + 256u];
        let w15 = w[5u * n + e + 256u];
        let w16 = w[6u * n + e + 256u];
        let w17 = w[7u * n + e + 256u];
        let x2 = r[rb + e + 512u];
        let w20 = w[e + 512u];
        let w21 = w[n + e + 512u];
        let w22 = w[2u * n + e + 512u];
        let w23 = w[3u * n + e + 512u];
        let w24 = w[4u * n + e + 512u];
        let w25 = w[5u * n + e + 512u];
        let w26 = w[6u * n + e + 512u];
        let w27 = w[7u * n + e + 512u];
        let x3 = r[rb + e + 768u];
        let w30 = w[e + 768u];
        let w31 = w[n + e + 768u];
        let w32 = w[2u * n + e + 768u];
        let w33 = w[3u * n + e + 768u];
        let w34 = w[4u * n + e + 768u];
        let w35 = w[5u * n + e + 768u];
        let w36 = w[6u * n + e + 768u];
        let w37 = w[7u * n + e + 768u];
        a0 = a0 + x0 * x0;
        a1 = a1 + w00 * x0;
        a2 = a2 + w01 * x0;
        a3 = a3 + w02 * x0;
        a4 = a4 + w03 * x0;
        a5 = a5 + w04 * x0;
        a6 = a6 + w05 * x0;
        a7 = a7 + w06 * x0;
        a8 = a8 + w07 * x0;
        a0 = a0 + x1 * x1;
        a1 = a1 + w10 * x1;
        a2 = a2 + w11 * x1;
        a3 = a3 + w12 * x1;
        a4 = a4 + w13 * x1;
        a5 = a5 + w14 * x1;
        a6 = a6 + w15 * x1;
        a7 = a7 + w16 * x1;
        a8 = a8 + w17 * x1;
        a0 = a0 + x2 * x2;
        a1 = a1 + w20 * x2;
        a2 = a2 + w21 * x2;
        a3 = a3 + w22 * x2;
        a4 = a4 + w23 * x2;
        a5 = a5 + w24 * x2;
        a6 = a6 + w25 * x2;
        a7 = a7 + w26 * x2;
        a8 = a8 + w27 * x2;
        a0 = a0 + x3 * x3;
        a1 = a1 + w30 * x3;
        a2 = a2 + w31 * x3;
        a3 = a3 + w32 * x3;
        a4 = a4 + w33 * x3;
        a5 = a5 + w34 * x3;
        a6 = a6 + w35 * x3;
        a7 = a7 + w36 * x3;
        a8 = a8 + w37 * x3;
        e = e + 1024u;
    }
    for (; e < n; e = e + 256u) {
        let x = r[rb + e];
        a0 = a0 + x * x;
        a1 = a1 + w[e] * x;
        a2 = a2 + w[n + e] * x;
        a3 = a3 + w[2u * n + e] * x;
        a4 = a4 + w[3u * n + e] * x;
        a5 = a5 + w[4u * n + e] * x;
        a6 = a6 + w[5u * n + e] * x;
        a7 = a7 + w[6u * n + e] * x;
        a8 = a8 + w[7u * n + e] * x;
    }
    red[t] = a0; red[256u + t] = a1; red[512u + t] = a2; red[768u + t] = a3; red[1024u + t] = a4;
    red[1280u + t] = a5; red[1536u + t] = a6; red[1792u + t] = a7; red[2048u + t] = a8;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) {
            for (var k = 0u; k < 9u; k = k + 1u) { red[k * 256u + t] = red[k * 256u + t] + red[k * 256u + t + st]; }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        let inv = inverseSqrt(red[0] / f32(n) + 1e-6);
        var l: array<f32, 8>;
        for (var k = 0u; k < 8u; k = k + 1u) { l[k] = red[(k + 1u) * 256u] * inv + bias[k]; }
        pre_s[0] = sig(l[0]);
        pre_s[1] = sig(l[1]);
        coef[cb] = 2.0 * sig(l[2]);
        coef[cb + 1u] = 2.0 * sig(l[3]);
        var m00 = l[4]; var m01 = l[5]; var m10 = l[6]; var m11 = l[7];
        for (var it = 0u; it < p.iters; it = it + 1u) {
            let r0 = lse2(m00, m01); m00 = m00 - r0; m01 = m01 - r0;
            let r1 = lse2(m10, m11); m10 = m10 - r1; m11 = m11 - r1;
            let c0 = lse2(m00, m10); m00 = m00 - c0; m10 = m10 - c0;
            let c1 = lse2(m01, m11); m01 = m01 - c1; m11 = m11 - c1;
        }
        coef[cb + 2u] = exp(m00); coef[cb + 3u] = exp(m01); coef[cb + 4u] = exp(m10); coef[cb + 5u] = exp(m11);
    }
    workgroupBarrier();
    let p0 = pre_s[0] * p.pre_scale;
    let p1 = pre_s[1] * p.pre_scale;
    for (var e = t; e < d; e = e + 256u) { h[hb + e] = p0 * r[rb + e] + p1 * r[rb + d + e]; }
}
"#;

const MERGE_SRC: &str = r#"
struct MP { d: u32, s: u32, outn: u32, flags: u32, iters: u32, pre_scale: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read_write> r: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> coef: array<f32>;
@group(0) @binding(4) var<uniform> p: MP;

@compute @workgroup_size(256)
fn mhc_merge(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= p.d * max(p.pad0, 1u)) { return; }
    let row = i / p.d;
    let e = i % p.d;
    let rb = row * 2u * p.d;
    let cb = row * 8u;
    let yv = y[i];
    let r0 = r[rb + e];
    let r1 = r[rb + p.d + e];
    let n0 = coef[cb + 2u] * r0 + coef[cb + 3u] * r1 + coef[cb] * yv;
    let n1 = coef[cb + 4u] * r0 + coef[cb + 5u] * r1 + coef[cb + 1u] * yv;
    r[rb + e] = n0;
    r[rb + p.d + e] = n1;
    if ((p.flags & 2u) != 0u) { h[i] = 0.5 * (n0 + n1); }
}
"#;

// ── DeepSeek-V4's four-stream form (GLM-5.3, `dsv4::hc_block`) ──
// The streams `r` are [rows][4][d]; a row's coefficients [post(4) | mix(16)]
// with mix_ij the weight of old stream j in new stream i (comb transposed).
const MHC4_SRC: &str = r#"
struct MP4 { d: u32, flags: u32, iters: u32, rows: u32, s0: f32, s1: f32, s2: f32, eps: f32, rms_eps: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read_write> h: array<f32>;
@group(0) @binding(1) var<storage, read_write> r: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> base: array<f32>;
@group(0) @binding(4) var<storage, read_write> coef: array<f32>;
@group(0) @binding(5) var<storage, read_write> lg: array<f32>;
@group(0) @binding(6) var<uniform> p: MP4;
var<workgroup> red2: array<f32, 512>;
var<workgroup> pre_s: array<f32, 4>;

fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

// enter: every stream takes the row's hidden (a thread an element)
@compute @workgroup_size(256)
fn mhc4_enter(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= p.d * max(p.rows, 1u)) { return; }
    let row = i / p.d;
    let e = i % p.d;
    let v = h[i];
    for (var s = 0u; s < 4u; s = s + 1u) { r[row * 4u * p.d + s * p.d + e] = v; }
}

// a workgroup a (logit k, row): W_k over the row's flat streams, scaled by
// their RMS
@compute @workgroup_size(256)
fn mhc4_logits(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = wid.x;
    let row = wid.y;
    let n = 4u * p.d;
    let rb = row * n;
    var s2 = 0.0;
    var d = 0.0;
    for (var e = t; e < n; e = e + 256u) {
        let x = r[rb + e];
        s2 = s2 + x * x;
        d = d + w[k * n + e] * x;
    }
    red2[t] = s2;
    red2[256u + t] = d;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) {
            red2[t] = red2[t] + red2[t + st];
            red2[256u + t] = red2[256u + t] + red2[256u + t + st];
        }
        workgroupBarrier();
    }
    // a negative epsilon is added to the RMS itself (Maxwell: 1 / (rms + eps))
    if (t == 0u) {
        let ms = red2[0] / f32(n);
        lg[row * 24u + k] = red2[256u] / select(sqrt(ms + p.rms_eps), sqrt(ms) - p.rms_eps, p.rms_eps < 0.0);
    }
}

// a workgroup a row: pre / post / the Sinkhorn mix (thread 0), the branch
// sum_s pre_s r_s into h
@compute @workgroup_size(256)
fn mhc4_route(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let d = p.d;
    let rb = row * 4u * d;
    let cb = row * 20u;
    if (t == 0u && (p.flags & 4u) != 0u) {
        // Hy4-flash's identity form: post = mag sigmoid + eps, no mix (the
        // closing fold reads the pre weights only)
        let lb = row * 24u;
        let mag = bitcast<f32>(p.p0);
        for (var s = 0u; s < 4u; s = s + 1u) {
            pre_s[s] = sig(lg[lb + s] * p.s0 + base[s]) + p.eps;
            if ((p.flags & 8u) == 0u) { coef[cb + s] = mag * sig(lg[lb + 4u + s] * p.s1 + base[4u + s]) + p.eps; }
        }
        if ((p.flags & 8u) == 0u) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) { coef[cb + 4u + i * 4u + j] = select(0.0, 1.0, i == j); }
            }
        }
    } else if (t == 0u) {
        let lb = row * 24u;
        for (var s = 0u; s < 4u; s = s + 1u) {
            pre_s[s] = sig(lg[lb + s] * p.s0 + base[s]) + p.eps;
            coef[cb + s] = 2.0 * sig(lg[lb + 4u + s] * p.s1 + base[4u + s]);
        }
        var m: array<f32, 16>;
        for (var i = 0u; i < 4u; i = i + 1u) {
            var mx = -3.0e38;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let kk = 8u + i * 4u + j;
                m[i * 4u + j] = lg[lb + kk] * p.s2 + base[kk];
                mx = max(mx, m[i * 4u + j]);
            }
            var se = 0.0;
            for (var j = 0u; j < 4u; j = j + 1u) {
                m[i * 4u + j] = exp(m[i * 4u + j] - mx);
                se = se + m[i * 4u + j];
            }
            for (var j = 0u; j < 4u; j = j + 1u) { m[i * 4u + j] = m[i * 4u + j] / se + p.eps; }
        }
        for (var it = 0u; it < p.iters; it = it + 1u) {
            if (it > 0u) {
                for (var i = 0u; i < 4u; i = i + 1u) {
                    var se = 0.0;
                    for (var j = 0u; j < 4u; j = j + 1u) { se = se + m[i * 4u + j]; }
                    se = se + p.eps;
                    for (var j = 0u; j < 4u; j = j + 1u) { m[i * 4u + j] = m[i * 4u + j] / se; }
                }
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                var se = p.eps;
                for (var i = 0u; i < 4u; i = i + 1u) { se = se + m[i * 4u + j]; }
                for (var i = 0u; i < 4u; i = i + 1u) { m[i * 4u + j] = m[i * 4u + j] / se; }
            }
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
            for (var j = 0u; j < 4u; j = j + 1u) { coef[cb + 4u + i * 4u + j] = m[j * 4u + i]; }
        }
    }
    workgroupBarrier();
    let a = pre_s[0];
    let b = pre_s[1];
    let c = pre_s[2];
    let dd = pre_s[3];
    for (var e = t; e < d; e = e + 256u) {
        h[row * d + e] = a * r[rb + e] + b * r[rb + d + e] + c * r[rb + 2u * d + e] + dd * r[rb + 3u * d + e];
    }
}

// a thread an element: new stream i = post_i y + sum_j mix_ij r_j; leaving,
// the hidden becomes the streams' mean
@compute @workgroup_size(256)
fn mhc4_merge(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= p.d * max(p.rows, 1u)) { return; }
    let row = i / p.d;
    let e = i % p.d;
    let rb = row * 4u * p.d;
    let cb = row * 20u;
    let yv = lg[i];
    var old: array<f32, 4>;
    for (var s = 0u; s < 4u; s = s + 1u) { old[s] = r[rb + s * p.d + e]; }
    var sum = 0.0;
    for (var a = 0u; a < 4u; a = a + 1u) {
        var v = coef[cb + a] * yv;
        for (var j = 0u; j < 4u; j = j + 1u) { v = v + coef[cb + 4u + a * 4u + j] * old[j]; }
        r[rb + a * p.d + e] = v;
        sum = sum + v;
    }
    if ((p.flags & 2u) != 0u) { h[i] = sum / 4.0; }
}
"#;

pub(crate) struct Pipes4 {
    pub enter: wgpu::ComputePipeline,
    pub logits: wgpu::ComputePipeline,
    pub route: wgpu::ComputePipeline,
    pub merge: wgpu::ComputePipeline,
}

static PIPES4: OnceLock<Option<Pipes4>> = OnceLock::new();

/// The four-stream kernels, compiled on first use.
pub(crate) fn pipes4(c: &Ctx) -> Option<&'static Pipes4> {
    PIPES4
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mhc4"),
                source: wgpu::ShaderSource::Wgsl(MHC4_SRC.into()),
            });
            let mk = |entry: &str| {
                c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &m,
                    entry_point: Some(entry),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                })
            };
            let p = Pipes4 { enter: mk("mhc4_enter"), logits: mk("mhc4_logits"), route: mk("mhc4_route"), merge: mk("mhc4_merge") };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("four-stream mHC kernels rejected: {e}");
                return None;
            }
            Some(p)
        })
        .as_ref()
}

/// One four-stream sub-layer step's uniform: `ffn` picks the FFN's group
/// scales, `leave` sets the merge's mean.
pub(crate) fn params4(d: usize, rows: usize, leave: bool, iters: usize, scale: [f32; 3], eps: f32, rms_eps: f32) -> [u32; 12] {
    params4x(d, rows, leave, iters, scale, eps, rms_eps, None, false)
}

/// `params4` with Hy4-flash's identity form (`ihc` = the post magnitude) and
/// its closing fold (`fold`: the route writes the pre-weighted fold only).
#[allow(clippy::too_many_arguments)]
pub(crate) fn params4x(d: usize, rows: usize, leave: bool, iters: usize, scale: [f32; 3], eps: f32, rms_eps: f32, ihc: Option<f32>, fold: bool) -> [u32; 12] {
    [
        d as u32,
        ((leave as u32) << 1) | ((ihc.is_some() as u32) << 2) | ((fold as u32) << 3),
        iters as u32,
        rows as u32,
        scale[0].to_bits(),
        scale[1].to_bits(),
        scale[2].to_bits(),
        eps.to_bits(),
        rms_eps.to_bits(),
        ihc.unwrap_or(0.0).to_bits(),
        0,
        0,
    ]
}

static BUFS4: std::sync::Mutex<Option<(usize, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)>> = std::sync::Mutex::new(None);

/// The four streams (`[rows · 4·d]`), the coefficients (`[rows · 20]`) and
/// the logits (`[rows · 24]`, also the merge's product rows when larger is
/// needed it is the caller's buffer), kept and grown.
pub(crate) fn bufs4(c: &Ctx, d: usize, rows: usize) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let rows = rows.max(1);
    let mut g = BUFS4.lock().unwrap();
    if let Some((cap, r, k, l)) = g.as_ref() {
        if *cap >= rows && r.size() >= (rows * 4 * d * 4) as u64 {
            return (r.clone(), k.clone(), l.clone());
        }
    }
    let mk = |label: &str, size: u64| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let (r, k, l) = (mk("mhc4_r", (rows * 4 * d * 4) as u64), mk("mhc4_coef", (rows * 20 * 4).max(64) as u64), mk("mhc4_lg", (rows * 24 * 4).max(64) as u64));
    *g = Some((rows, r.clone(), k.clone(), l.clone()));
    (r, k, l)
}

/// Encode a four-stream sub-layer's route: (entering) the streams take the
/// hidden rows, the logits, then the coefficients and the branch into `h`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_route4(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes4,
    h: &wgpu::Buffer,
    w: &wgpu::Buffer,
    base: &wgpu::Buffer,
    u: &wgpu::Buffer,
    d: usize,
    rows: usize,
    enter: bool,
) {
    encode_route4n(c, enc, p, h, w, base, u, d, rows, enter, 24)
}

/// `encode_route4` over the first `logits` router rows (Hy4-flash's identity
/// form has 8, its closing fold 4).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_route4n(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes4,
    h: &wgpu::Buffer,
    w: &wgpu::Buffer,
    base: &wgpu::Buffer,
    u: &wgpu::Buffer,
    d: usize,
    rows: usize,
    enter: bool,
    logits: u32,
) {
    let (r, k, l) = bufs4(c, d, rows);
    let rows = rows.max(1) as u32;
    let b_enter = bind_at(c, &p.enter, &[(0, h), (1, &r), (6, u)]);
    let b_logits = bind_at(c, &p.logits, &[(1, &r), (2, w), (5, &l), (6, u)]);
    let b_route = bind_at(c, &p.route, &[(0, h), (1, &r), (3, base), (4, &k), (5, &l), (6, u)]);
    let mut pass = crate::gpu_wgpu::begin_pass(enc);
    if enter {
        let (gx, gy) = merge_grid(d * rows as usize);
        pass.set_pipeline(&p.enter);
        pass.set_bind_group(0, &b_enter, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    pass.set_pipeline(&p.logits);
    pass.set_bind_group(0, &b_logits, &[]);
    pass.dispatch_workgroups(logits, rows, 1);
    pass.set_pipeline(&p.route);
    pass.set_bind_group(0, &b_route, &[]);
    pass.dispatch_workgroups(rows, 1, 1);
}

/// A bind group of the bindings an entry point reads (its auto layout holds
/// those only).
fn bind_at(c: &Ctx, pipe: &wgpu::ComputePipeline, bufs: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let layout = pipe.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry> =
        bufs.iter().map(|(i, b)| wgpu::BindGroupEntry { binding: *i, resource: b.as_entire_binding() }).collect();
    c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("mhc4"), layout: &layout, entries: &entries })
}

/// Encode a four-stream merge of the sub-layer's product rows `y` (and,
/// leaving, the streams' mean into `h`). `y` rides the logits binding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_merge4(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes4,
    h: &wgpu::Buffer,
    y: &wgpu::Buffer,
    u: &wgpu::Buffer,
    d: usize,
    rows: usize,
) {
    let (r, k, _) = bufs4(c, d, rows);
    let b = bind_at(c, &p.merge, &[(0, h), (1, &r), (4, &k), (5, y), (6, u)]);
    let (gx, gy) = merge_grid(d * rows.max(1));
    let mut pass = crate::gpu_wgpu::begin_pass(enc);
    pass.set_pipeline(&p.merge);
    pass.set_bind_group(0, &b, &[]);
    pass.dispatch_workgroups(gx, gy, 1);
}

pub(crate) struct Pipes {
    pub route: wgpu::ComputePipeline,
    pub merge: wgpu::ComputePipeline,
    /// Their bind-group layouts, taken once (the token graph caches its groups).
    pub route_layout: wgpu::BindGroupLayout,
    pub merge_layout: wgpu::BindGroupLayout,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

pub(crate) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mk = |label: &str, src: &str, entry: &str| {
                let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
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
            let (route, merge) = (mk("mhc_route", ROUTE_SRC, "mhc_route"), mk("mhc_merge", MERGE_SRC, "mhc_merge"));
            let (route_layout, merge_layout) = (route.get_bind_group_layout(0), merge.get_bind_group_layout(0));
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("mHC kernels rejected: {e}");
                return None;
            }
            Some(Pipes { route, merge, route_layout, merge_layout })
        })
        .as_ref()
}

static UNIFORMS: std::sync::Mutex<Option<std::collections::HashMap<[u32; 8], wgpu::Buffer>>> = std::sync::Mutex::new(None);

/// A params uniform by content: the same words every token, so one buffer each.
pub(crate) fn uniform8(c: &Ctx, v: [u32; 8]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let mut g = UNIFORMS.lock().unwrap();
    let m = g.get_or_insert_with(Default::default);
    if let Some(b) = m.get(&v) {
        return b.clone();
    }
    let b = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mhc-params"),
        contents: bytemuck::cast_slice(&v),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    m.insert(v, b.clone());
    b
}

/// The uniform words: `[d, 2, 8, flags, iters, pre_scale, rows, 0]` with
/// flags bit 0 = enter (copy the hidden into both streams), bit 1 = leave
/// (the hidden becomes the streams' mean). `mhc_route` takes one workgroup
/// per row, `mhc_merge` one thread per element of all rows.
pub(crate) fn params(d: usize, enter: bool, leave: bool, iters: usize, pre_scale: f32) -> [u32; 8] {
    params_k(d, 1, enter, leave, iters, pre_scale)
}

pub(crate) fn params_k(d: usize, rows: usize, enter: bool, leave: bool, iters: usize, pre_scale: f32) -> [u32; 8] {
    [d as u32, 2, 8, enter as u32 | (leave as u32) << 1, iters as u32, pre_scale.to_bits(), rows as u32, 0]
}

/// The merge's workgroup grid for `n` elements (≤ 65 535 per axis).
pub(crate) fn merge_grid(n: usize) -> (u32, u32) {
    let wgs = n.div_ceil(256).max(1);
    let x = wgs.min(65_535);
    (x as u32, wgs.div_ceil(x) as u32)
}

/// Bind group of `mhc_route`: hidden, streams, router W, router b, coef, uniform.
pub(crate) fn route_bg(c: &Ctx, p: &Pipes, bufs: [&wgpu::Buffer; 6]) -> wgpu::BindGroup {
    bind(c, &p.route, &bufs)
}

/// Bind group of `mhc_merge`: hidden, streams, sub-layer output, coef, uniform.
pub(crate) fn merge_bg(c: &Ctx, p: &Pipes, bufs: [&wgpu::Buffer; 5]) -> wgpu::BindGroup {
    bind(c, &p.merge, &bufs)
}

fn bind(c: &Ctx, pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let layout = pipe.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("mhc"), layout: &layout, entries: &entries })
}

static BUFS: std::sync::Mutex<Option<(usize, wgpu::Buffer, wgpu::Buffer)>> = std::sync::Mutex::new(None);

/// The streams buffer (`[2·d]` f32) and the coefficient buffer, kept.
pub(crate) fn bufs(c: &Ctx, d: usize) -> (wgpu::Buffer, wgpu::Buffer) {
    bufs_k(c, d, 1)
}

/// The streams (`[rows · 2·d]` f32) and coefficient (`[rows · 8]` f32)
/// buffers of a batch, kept and grown (`d` is fixed per model).
pub(crate) fn bufs_k(c: &Ctx, d: usize, rows: usize) -> (wgpu::Buffer, wgpu::Buffer) {
    let rows = rows.max(1);
    let mut g = BUFS.lock().unwrap();
    if let Some((cap, r, k)) = g.as_ref() {
        if *cap >= rows && r.size() >= (rows * 2 * d * 4) as u64 {
            return (r.clone(), k.clone());
        }
    }
    let mk = |label: &str, size: u64| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let (r, k) = (mk("mhc_r", (rows * 2 * d * 4) as u64), mk("mhc_coef", (rows * 8 * 4).max(64) as u64));
    *g = Some((rows, r.clone(), k.clone()));
    (r, k)
}

#[cfg(test)]
mod mhc4_tests {
    #[test]
    fn four_stream_wgsl_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(super::MHC4_SRC).expect("four-stream mHC WGSL must parse");
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .expect("four-stream mHC WGSL must validate");
    }

    #[test]
    fn engram_wgsl_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(super::ENGRAM_SRC).expect("Engram WGSL must parse");
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .expect("Engram WGSL must validate");
    }

    #[test]
    fn two_stream_wgsl_validates() {
        for (name, src) in [("route", super::ROUTE_SRC), ("merge", super::MERGE_SRC)] {
            let module = wgpu::naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} WGSL must parse: {e}"));
            wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name} WGSL must validate: {e:?}"));
        }
    }
}

// ── Hy4-flash's Engram on the four streams (`hy4::Engram::apply`) ──
// v: the value projection (no bias yet), k: the four key projections [4][d],
// c1: [value bias | gamma], c2: [key bias | key norm | query norm | hc scale],
// each [4][d]. A workgroup a stream: the value normed (no weight) times gamma,
// the stream's key and query normed, their scaled dot through the signed
// square root and a sigmoid, then stream += (gate + hc_scale) * value.
const ENGRAM_SRC: &str = r#"
struct EP { d: u32, eps: f32, rows: u32, _b: u32 }
@group(0) @binding(0) var<storage, read_write> r: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> c1: array<f32>;
@group(0) @binding(4) var<storage, read> c2: array<f32>;
@group(0) @binding(5) var<uniform> p: EP;
var<workgroup> red: array<f32, 256>;

fn wsum(t: u32, x: f32) -> f32 {
    red[t] = x;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let s = red[0];
    workgroupBarrier();
    return s;
}

// a workgroup a (stream j, row): r [rows][4][d], v [rows][d], k [4][rows][d]
@compute @workgroup_size(256)
fn engram4(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let j = wid.x;
    let row = wid.y;
    let d = p.d;
    let fd = f32(d);
    let rb = (row * 4u + j) * d;
    let vb = row * d;
    let kb = (j * p.rows + row) * d;
    var sv = 0.0;
    var sk = 0.0;
    var ss = 0.0;
    for (var e = t; e < d; e = e + 256u) {
        let a = v[vb + e] + c1[e];
        sv = sv + a * a;
        let b = k[kb + e] + c2[j * d + e];
        sk = sk + b * b;
        let s = r[rb + e];
        ss = ss + s * s;
    }
    let rv = 1.0 / sqrt(wsum(t, sv) / fd + p.eps);
    let rk = 1.0 / sqrt(wsum(t, sk) / fd + p.eps);
    let rs = 1.0 / sqrt(wsum(t, ss) / fd + p.eps);
    var dt = 0.0;
    for (var e = t; e < d; e = e + 256u) {
        let kn = (k[kb + e] + c2[j * d + e]) * rk * c2[4u * d + j * d + e];
        let qn = r[rb + e] * rs * c2[8u * d + j * d + e];
        dt = dt + kn * qn;
    }
    let g0 = wsum(t, dt) / sqrt(fd);
    var g = 0.0;
    if (g0 != 0.0) { g = sqrt(max(abs(g0), 1e-6)) * sign(g0); }
    let gate = 1.0 / (1.0 + exp(-g));
    for (var e = t; e < d; e = e + 256u) {
        let m = (v[vb + e] + c1[e]) * rv * c1[d + e];
        r[rb + e] = r[rb + e] + gate * m + c2[12u * d + j * d + e] * m;
    }
}
"#;

static ENGRAM: OnceLock<Option<wgpu::ComputePipeline>> = OnceLock::new();

/// Hy4-flash's Engram step over the four streams of `rows` rows (`bufs4`):
/// `v` the value projection `[rows][d]`, `k` the four key projections
/// `[4][rows][d]`, `c1` `[value bias | gamma]`, `c2` `[key bias | key norm |
/// query norm | hc scale]` (each `[4][d]`). None: the kernel did not build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_engram4(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    d: usize,
    rows: usize,
    eps: f32,
    v: &wgpu::Buffer,
    k: &wgpu::Buffer,
    c1: &wgpu::Buffer,
    c2: &wgpu::Buffer,
) -> Option<()> {
    let pipe = ENGRAM
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("engram4"), source: wgpu::ShaderSource::Wgsl(ENGRAM_SRC.into()) });
            let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("engram4"),
                layout: None,
                module: &m,
                entry_point: Some("engram4"),
                compilation_options: crate::gpu_wgpu::pco(),
                cache: None,
            });
            match pollster::block_on(sc.pop()) {
                None => Some(p),
                Some(e) => {
                    tracing::warn!("wgpu Engram kernel: {e}");
                    None
                }
            }
        })
        .as_ref()?;
    use wgpu::util::DeviceExt;
    let rows = rows.max(1);
    let (r, _, _) = bufs4(c, d, rows);
    let u = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("engram4-u"),
        contents: bytemuck::cast_slice(&[d as u32, eps.to_bits(), rows as u32, 0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let b = bind_at(c, pipe, &[(0, &r), (1, v), (2, k), (3, c1), (4, c2), (5, &u)]);
    let mut pass = crate::gpu_wgpu::begin_pass(enc);
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &b, &[]);
    pass.dispatch_workgroups(4, rows as u32, 1);
    Some(())
}
