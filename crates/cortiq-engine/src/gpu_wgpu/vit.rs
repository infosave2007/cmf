//! A vision tower's encoder layers resident on wgpu (`gpu::vit_encode`):
//! the pre-norm ViT blocks of Mage-VL's OneVision tower and SenseNova's
//! Qwen3-VL tower with the activations on the card from the first block to
//! the last — one upload of the embedded rows, one readback.
//!
//! Per block: `vt_ln` (LayerNorm with bias, one workgroup a row) → the
//! matrix-unit NT GEMM (`gemm_nt_coop`, the same kernel as `gemm_nt_f32`) for
//! qkv → `vt_bias_act` → `vt_rope` (head-major q/k/v planes, q and k rotated
//! by host cos/sin tables: interleaved pairs for OneVision, split halves for
//! Qwen3-VL) → the DiT attention in its resident mode (the planes are already
//! in its scratch slots, the output panel stays on the card) → proj GEMM →
//! `vt_add` (bias + residual) → LayerNorm → fc1 GEMM → `vt_bias_act` with the
//! block's GELU (exact or tanh) → fc2 GEMM → `vt_add`.
//!
//! The weights upload once per tower (`key`) and stay. The per-op door
//! (`gemm_nt_f32` per projection) moved every activation across the bus
//! twice and re-uploaded each weight on its first sighting.

use super::{Ctx, Scratch, ctx, submit};
use std::sync::{Mutex, OnceLock};

const LN_SRC: &str = r#"
struct VP { rows: u32, cols: u32, a: u32, b: u32, eps: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bb: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> p: VP;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn vt_ln(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = wid.x + wid.y * 65535u;
    if (row >= p.rows) { return; }
    let d = p.cols;
    let base = row * d;
    var s = 0.0;
    for (var i = t; i < d; i = i + 256u) { s = s + src[base + i]; }
    red[t] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let mean = red[0] / f32(d);
    workgroupBarrier();
    var v = 0.0;
    for (var i = t; i < d; i = i + 256u) { let x = src[base + i] - mean; v = v + x * x; }
    red[t] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(d) + p.eps);
    for (var i = t; i < d; i = i + 256u) { dst[base + i] = (src[base + i] - mean) * inv * w[i] + bb[i]; }
}
"#;

const ACT_SRC: &str = r#"
struct VP { rows: u32, cols: u32, act: u32, b: u32, eps: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> bb: array<f32>;
@group(0) @binding(2) var<uniform> p: VP;
// erf by Abramowitz–Stegun 7.1.26 (|error| ≤ 1.5e-7)
fn erf_(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return s * (1.0 - poly * exp(-a * a));
}
@compute @workgroup_size(256)
fn vt_bias_act(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= p.rows * p.cols) { return; }
    var v = y[i] + bb[i % p.cols];
    if (p.act == 1u) {
        v = 0.5 * v * (1.0 + erf_(v * 0.70710678118654752));
    } else if (p.act == 2u) {
        let u = clamp(0.7978845608 * (v + 0.044715 * v * v * v), -15.0, 15.0);
        v = 0.5 * v * (1.0 + tanh(u));
    }
    y[i] = v;
}
"#;

const ADD_SRC: &str = r#"
struct VP { rows: u32, cols: u32, a: u32, b: u32, eps: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> bb: array<f32>;
@group(0) @binding(3) var<uniform> p: VP;
@compute @workgroup_size(256)
fn vt_add(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= p.rows * p.cols) { return; }
    x[i] = x[i] + y[i] + bb[i % p.cols];
}
"#;

const ROPE_SRC: &str = r#"
// rows = n patches, cols = heads, a = head_dim, b = mode (0 interleaved pairs, 1 split halves)
struct VP { rows: u32, cols: u32, a: u32, b: u32, eps: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> cs: array<f32>;
@group(0) @binding(2) var<storage, read> sn: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> k: array<f32>;
@group(0) @binding(5) var<storage, read_write> v: array<f32>;
@group(0) @binding(6) var<uniform> p: VP;
@compute @workgroup_size(256)
fn vt_rope(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = p.rows;
    let nh = p.cols;
    let hd = p.a;
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= nh * n * hd) { return; }
    let h = i / (n * hd);
    let pp = (i / hd) % n;
    let j = i % hd;
    let d = nh * hd;
    let base = pp * 3u * d + h * hd;
    var pj = 0u;
    var sg = 1.0;
    if (p.b == 0u) {
        pj = j ^ 1u;
        if ((j & 1u) == 0u) { sg = -1.0; }
    } else {
        let half = hd / 2u;
        if (j < half) { pj = j + half; sg = -1.0; } else { pj = j - half; }
    }
    let c = cs[pp * hd + j];
    let s = sn[pp * hd + j];
    q[i] = qkv[base + j] * c + sg * qkv[base + pj] * s;
    k[i] = qkv[base + d + j] * c + sg * qkv[base + d + pj] * s;
    v[i] = qkv[base + 2u * d + j];
}
"#;

struct Pipes {
    ln: wgpu::ComputePipeline,
    act: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mk = |src: &str, entry: &str| {
                let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
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
            let p = Pipes {
                ln: mk(LN_SRC, "vt_ln"),
                act: mk(ACT_SRC, "vt_bias_act"),
                add: mk(ADD_SRC, "vt_add"),
                rope: mk(ROPE_SRC, "vt_rope"),
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("vit kernels rejected: {e}");
                return None;
            }
            Some(p)
        })
        .as_ref()
}

struct LayerBufs {
    ln1_w: wgpu::Buffer,
    ln1_b: wgpu::Buffer,
    qkv: wgpu::Buffer,
    qkv_b: wgpu::Buffer,
    proj: wgpu::Buffer,
    proj_b: wgpu::Buffer,
    ln2_w: wgpu::Buffer,
    ln2_b: wgpu::Buffer,
    fc1: wgpu::Buffer,
    fc1_b: wgpu::Buffer,
    fc2: wgpu::Buffer,
    fc2_b: wgpu::Buffer,
}

struct State {
    key: u64,
    layers: Vec<LayerBufs>,
    /// x, xn, qkv/ff, tmp — grown as needed.
    act: Vec<(wgpu::Buffer, u64)>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn stor(c: &Ctx, label: &str, data: &[f32]) -> wgpu::Buffer {
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (data.len().max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
    b
}

fn grid(n: usize) -> (u32, u32) {
    let wgs = n.div_ceil(256).max(1);
    let x = wgs.min(65_535);
    (x as u32, wgs.div_ceil(x) as u32)
}

/// Run the tower's encoder layers over `x` (`[n, d]`, embedded and
/// pre-normed rows in the tower's patch order), in place. `false` =
/// refused (no device, no matrix units, a shape outside the kernels'
/// contract); the caller runs its per-op path.
pub fn encode(a: &crate::gpu::VitArgs, x: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(p) = pipes(c) else { return false };
    let Some(gemm) = c.gemm_nt_coop.as_ref() else { return false };
    let (n, d, nh, hd, inter) = (a.n, a.d, a.heads, a.head_dim, a.inter);
    if n == 0
        || nh * hd != d
        || d % 4 != 0
        || inter % 4 != 0
        || x.len() != n * d
        || a.cos.len() != n * hd
        || a.sin.len() != n * hd
        || n.div_ceil(64) > 65_535
        || n > 65_535
    {
        return false;
    }
    let lim = c.device.limits();
    let max_bind = (lim.max_storage_buffer_binding_size as u64).min(lim.max_buffer_size);
    let big = [n * 3 * d, n * inter, n * n].into_iter().max().unwrap();
    if (big * 4) as u64 > max_bind {
        return false;
    }
    let _op = c.op_gate.enter();
    let sc_err = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut g = STATE.lock().unwrap();
    if g.as_ref().is_none_or(|s| s.key != a.key || s.layers.len() != a.layers.len()) {
        let layers = a
            .layers
            .iter()
            .map(|l| LayerBufs {
                ln1_w: stor(c, "vt", l.ln1_w),
                ln1_b: stor(c, "vt", l.ln1_b),
                qkv: stor(c, "vt", l.qkv),
                qkv_b: stor(c, "vt", l.qkv_b),
                proj: stor(c, "vt", l.proj),
                proj_b: stor(c, "vt", l.proj_b),
                ln2_w: stor(c, "vt", l.ln2_w),
                ln2_b: stor(c, "vt", l.ln2_b),
                fc1: stor(c, "vt", l.fc1),
                fc1_b: stor(c, "vt", l.fc1_b),
                fc2: stor(c, "vt", l.fc2),
                fc2_b: stor(c, "vt", l.fc2_b),
            })
            .collect();
        *g = Some(State { key: a.key, layers, act: Vec::new() });
    }
    let st = g.as_mut().expect("set above");
    let sizes = [n * d, n * d, n * (3 * d).max(inter), n * d];
    while st.act.len() < sizes.len() {
        st.act.push((c.device.create_buffer(&wgpu::BufferDescriptor { label: Some("vt-act"), size: 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }), 4));
    }
    for (slot, &sz) in st.act.iter_mut().zip(&sizes) {
        let need = (sz * 4) as u64;
        if slot.1 < need {
            *slot = (
                c.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vt-act"),
                    size: need,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                need,
            );
        }
    }
    // The attention's input planes: its own scratch slots (`resident`).
    let plane = (nh * n * hd * 4) as u64;
    let (qb, kb, vb) = {
        let mut sc = c.scratch.lock().unwrap();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        (
            Scratch::ensure(&c.device, &mut sc.dq, plane, usage, "dit-q"),
            Scratch::ensure(&c.device, &mut sc.dk, plane, usage, "dit-k"),
            Scratch::ensure(&c.device, &mut sc.dv, plane, usage, "dit-v"),
        )
    };
    if pollster::block_on(sc_err.pop()).is_some() {
        *g = None;
        return false;
    }
    let (xb, xnb, wide, tmp) = (st.act[0].0.clone(), st.act[1].0.clone(), st.act[2].0.clone(), st.act[3].0.clone());
    let csb = stor(c, "vt-cos", a.cos);
    let snb = stor(c, "vt-sin", a.sin);
    c.queue.write_buffer(&xb, 0, bytemuck::cast_slice(x));
    let unif = |w: [u32; 8]| {
        let b = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vt-p"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(&w));
        b
    };
    let bind = |pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| {
        let layout = pipe.get_bind_group_layout(0);
        let e: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("vt"), layout: &layout, entries: &e })
    };
    let go = |enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, gx: u32, gy: u32| {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("vt"), timestamp_writes: None });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    };
    let eps = (a.eps as f32).to_bits();
    let ln = |enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer, w: &wgpu::Buffer, b: &wgpu::Buffer, dst: &wgpu::Buffer| {
        let u = unif([n as u32, d as u32, 0, 0, eps, 0, 0, 0]);
        go(enc, &p.ln, &bind(&p.ln, &[src, w, b, dst, &u]), (n as u32).min(65_535), (n as u32).div_ceil(65_535));
    };
    // y[n, m] = x[n, k] · w[m, k]ᵀ on the matrix units
    let mm = |enc: &mut wgpu::CommandEncoder, w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer, k: usize, m: usize| {
        let u = unif([(k / 4) as u32, m as u32, n as u32, 0, 0, 0, 0, 0]);
        go(enc, gemm, &bind(gemm, &[w, x, y, &u]), (m as u32).div_ceil(64), (n as u32).div_ceil(64));
    };
    let bias_act = |enc: &mut wgpu::CommandEncoder, y: &wgpu::Buffer, b: &wgpu::Buffer, cols: usize, act: u32| {
        let u = unif([n as u32, cols as u32, act, 0, 0, 0, 0, 0]);
        let (gx, gy) = grid(n * cols);
        go(enc, &p.act, &bind(&p.act, &[y, b, &u]), gx, gy);
    };
    let add = |enc: &mut wgpu::CommandEncoder, xb: &wgpu::Buffer, y: &wgpu::Buffer, b: &wgpu::Buffer| {
        let u = unif([n as u32, d as u32, 0, 0, 0, 0, 0, 0]);
        let (gx, gy) = grid(n * d);
        go(enc, &p.add, &bind(&p.add, &[xb, y, b, &u]), gx, gy);
    };
    let t0 = std::time::Instant::now();
    let scale = (hd as f32).powf(-0.5);
    // `CMF_VIT_PROF=1`: drain the queue after each stage and split the time
    // (pre-attention, attention, post-attention); slower, diagnostics only.
    let stage_prof = std::env::var("CMF_VIT_PROF").as_deref() == Ok("1");
    let drain = || {
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    };
    let mut split = [0f64; 3];
    for l in &st.layers {
        let ts = std::time::Instant::now();
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vt-a") });
        ln(&mut enc, &xb, &l.ln1_w, &l.ln1_b, &xnb);
        mm(&mut enc, &l.qkv, &xnb, &wide, d, 3 * d);
        bias_act(&mut enc, &wide, &l.qkv_b, 3 * d, 0);
        let u = unif([n as u32, nh as u32, hd as u32, a.rope_halves as u32, 0, 0, 0, 0]);
        let (gx, gy) = grid(nh * n * hd);
        go(&mut enc, &p.rope, &bind(&p.rope, &[&wide, &csb, &snb, &qb, &kb, &vb, &u]), gx, gy);
        submit(c, enc.finish());
        if stage_prof {
            drain();
            split[0] += ts.elapsed().as_secs_f64();
        }
        let ts = std::time::Instant::now();
        let mut keep: Option<wgpu::Buffer> = None;
        if !super::dit_attention_inner(&[], &[], &[], nh, nh, n, hd, scale, &mut [], true, Some(&mut keep)) {
            return false;
        }
        let Some(ab) = keep else { return false };
        if stage_prof {
            drain();
            split[1] += ts.elapsed().as_secs_f64();
        }
        let ts = std::time::Instant::now();
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vt-b") });
        mm(&mut enc, &l.proj, &ab, &tmp, d, d);
        add(&mut enc, &xb, &tmp, &l.proj_b);
        ln(&mut enc, &xb, &l.ln2_w, &l.ln2_b, &xnb);
        mm(&mut enc, &l.fc1, &xnb, &wide, d, inter);
        bias_act(&mut enc, &wide, &l.fc1_b, inter, if a.gelu_tanh { 2 } else { 1 });
        mm(&mut enc, &l.fc2, &wide, &tmp, inter, d);
        add(&mut enc, &xb, &tmp, &l.fc2_b);
        submit(c, enc.finish());
        if stage_prof {
            drain();
            split[2] += ts.elapsed().as_secs_f64();
        }
    }
    if stage_prof {
        eprintln!("vit stages: ln+qkv+rope {:.3} s, attention {:.3} s, proj+mlp {:.3} s", split[0], split[1], split[2]);
    }
    // readback
    let bytes = (n * d * 4) as u64;
    let staging = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vt-rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vt-rb") });
    enc.copy_buffer_to_buffer(&xb, 0, &staging, 0, bytes);
    submit(c, enc.finish());
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    {
        let Ok(data) = slice.get_mapped_range() else {
            staging.unmap();
            return false;
        };
        x.copy_from_slice(bytemuck::cast_slice(&data[..bytes as usize]));
    }
    staging.unmap();
    if a.prof {
        eprintln!("vit (wgpu, resident): {} layers over {n} rows, {:.3} s", st.layers.len(), t0.elapsed().as_secs_f64());
    }
    true
}
