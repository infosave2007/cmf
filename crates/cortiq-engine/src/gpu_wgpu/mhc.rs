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
    var acc: array<f32, 9>;
    for (var k = 0u; k < 9u; k = k + 1u) { acc[k] = 0.0; }
    for (var e = t; e < n; e = e + 256u) {
        let x = r[rb + e];
        acc[0] = acc[0] + x * x;
        for (var k = 0u; k < 8u; k = k + 1u) { acc[k + 1u] = acc[k + 1u] + w[k * n + e] * x; }
    }
    for (var k = 0u; k < 9u; k = k + 1u) { red[k * 256u + t] = acc[k]; }
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

pub(crate) struct Pipes {
    pub route: wgpu::ComputePipeline,
    pub merge: wgpu::ComputePipeline,
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
            let p = Pipes { route: mk("mhc_route", ROUTE_SRC, "mhc_route"), merge: mk("mhc_merge", MERGE_SRC, "mhc_merge") };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("mHC kernels rejected: {e}");
                return None;
            }
            Some(p)
        })
        .as_ref()
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
