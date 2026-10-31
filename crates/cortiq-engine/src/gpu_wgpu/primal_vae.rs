//! Primal VAE on wgpu, resident: a whole encode or decode (`primal_vae::PvPlan`)
//! as one command buffer — one upload of the input, one readback of the
//! output, every activation on the device in between.
//!
//! Four kernels, f32 NCHW throughout:
//! - `pv_conv`: a 3×3 convolution with dilation = padding as an implicit
//!   GEMM (64 pixels × 64 output channels per workgroup, 16×16 threads with
//!   4×4 outputs each, 16 K values — `(cin, tap)` pairs — per shared-memory
//!   step), bias and an optional in-place residual (`dst += conv`) in the
//!   epilogue;
//! - `pv_norm`: the per-pixel RMSNorm over the channels × γ, then SiLU;
//! - `pv_unshuf` / `pv_shuf`: `pixel_unshuffle(2)` / `pixel_shuffle(2)`, the
//!   latter with an optional in-place add (the U-Net skip).
//!
//! Weights upload once per model (`key`) and stay. `CMF_PRIMAL_VAE_GPU=0`
//! (checked by the caller) keeps the host path.

use super::{Ctx, ctx, submit};
use crate::primal_vae::{PvPlan, PvStep, PvWeights};
use std::sync::{Mutex, OnceLock};

const SRC: &str = r#"
struct P { a: u32, b: u32, h: u32, w: u32, dil: u32, flag: u32, k: u32, n: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> ta: array<f32, 1024>;
var<workgroup> tb: array<f32, 1024>;

// a = cin, b = cout, k = 9·cin; flag = residual in place.
@compute @workgroup_size(16, 16, 1)
fn pv_conv(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_index) li: u32,
           @builtin(local_invocation_id) lid: vec3<u32>) {
    let hw = p.h * p.w;
    let m0 = wg.x * 64u;
    let n0 = wg.y * 64u;
    let dil = i32(p.dil);
    var acc: array<vec4<f32>, 4>;
    for (var r = 0u; r < 4u; r = r + 1u) { acc[r] = vec4<f32>(0.0); }
    var k0 = 0u;
    loop {
        if (k0 >= p.k) { break; }
        for (var e = 0u; e < 4u; e = e + 1u) {
            let idx = li + e * 256u;
            let kk = idx / 64u;
            let mm = idx % 64u;
            let k = k0 + kk;
            let pix = m0 + mm;
            var va = 0.0;
            if (k < p.k && pix < hw) {
                let ci = k / 9u;
                let tap = k - ci * 9u;
                let y = i32(pix / p.w) + (i32(tap / 3u) - 1) * dil;
                let x = i32(pix % p.w) + (i32(tap % 3u) - 1) * dil;
                if (y >= 0 && x >= 0 && y < i32(p.h) && x < i32(p.w)) {
                    va = src[ci * hw + u32(y) * p.w + u32(x)];
                }
            }
            ta[idx] = va;
            let nn = idx / 16u;
            let kb = idx % 16u;
            var vb = 0.0;
            if (n0 + nn < p.b && k0 + kb < p.k) { vb = wt[(n0 + nn) * p.k + k0 + kb]; }
            tb[kb * 64u + nn] = vb;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let ra = kk * 64u + lid.x * 4u;
            let rb = kk * 64u + lid.y * 4u;
            let a = vec4<f32>(ta[ra], ta[ra + 1u], ta[ra + 2u], ta[ra + 3u]);
            let bv = vec4<f32>(tb[rb], tb[rb + 1u], tb[rb + 2u], tb[rb + 3u]);
            acc[0] = fma(vec4<f32>(a.x), bv, acc[0]);
            acc[1] = fma(vec4<f32>(a.y), bv, acc[1]);
            acc[2] = fma(vec4<f32>(a.z), bv, acc[2]);
            acc[3] = fma(vec4<f32>(a.w), bv, acc[3]);
        }
        workgroupBarrier();
        k0 = k0 + 16u;
    }
    for (var r = 0u; r < 4u; r = r + 1u) {
        let pix = m0 + lid.x * 4u + r;
        if (pix < hw) {
            for (var c = 0u; c < 4u; c = c + 1u) {
                let oc = n0 + lid.y * 4u + c;
                if (oc < p.b) {
                    let o = oc * hw + pix;
                    var v = acc[r][c] + bias[oc];
                    if (p.flag != 0u) { v = v + dst[o]; }
                    dst[o] = v;
                }
            }
        }
    }
}

fn gidx(gid: vec3<u32>, nwg: vec3<u32>) -> u32 { return gid.x + gid.y * nwg.x * 256u; }

// a = channels; wt = γ.
@compute @workgroup_size(256, 1, 1)
fn pv_norm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let hw = p.h * p.w;
    let pix = gidx(gid, nwg);
    if (pix >= hw) { return; }
    var ss = 0.0;
    for (var c = 0u; c < p.a; c = c + 1u) { let v = src[c * hw + pix]; ss = ss + v * v; }
    let inv = inverseSqrt(ss / f32(p.a) + 1e-6);
    for (var c = 0u; c < p.a; c = c + 1u) {
        let v = src[c * hw + pix] * inv * wt[c];
        dst[c * hw + pix] = v / (1.0 + exp(-v));
    }
}

// a = source channels, (h, w) = source size; one thread per output element.
@compute @workgroup_size(256, 1, 1)
fn pv_unshuf(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let o = gidx(gid, nwg);
    if (o >= p.n) { return; }
    let oh = p.h / 2u;
    let ow = p.w / 2u;
    let x = o % ow;
    let y = (o / ow) % oh;
    let oc = o / (ow * oh);
    let c = oc / 4u;
    let i = (oc / 2u) % 2u;
    let j = oc % 2u;
    dst[o] = src[(c * p.h + 2u * y + i) * p.w + 2u * x + j];
}

// a = source channels (4× the output's), (h, w) = source size; flag = add.
@compute @workgroup_size(256, 1, 1)
fn pv_shuf(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let o = gidx(gid, nwg);
    if (o >= p.n) { return; }
    let oh = 2u * p.h;
    let ow = 2u * p.w;
    let ox = o % ow;
    let oy = (o / ow) % oh;
    let c = o / (ow * oh);
    let ic = c * 4u + (oy % 2u) * 2u + (ox % 2u);
    var v = src[(ic * p.h + oy / 2u) * p.w + ox / 2u];
    if (p.flag != 0u) { v = v + dst[o]; }
    dst[o] = v;
}
"#;

struct Pipes {
    conv: wgpu::ComputePipeline,
    norm: wgpu::ComputePipeline,
    unshuf: wgpu::ComputePipeline,
    shuf: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let ro = |b: u32| wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let layout = c.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pv_bgl"),
                entries: &[
                    ro(0),
                    ro(1),
                    ro(2),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let pl = c.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pv_pl"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("primal_vae"),
                source: wgpu::ShaderSource::Wgsl(SRC.into()),
            });
            let mk = |e: &str| {
                c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pl),
                    module: &m,
                    entry_point: Some(e),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                })
            };
            let p = Pipes { conv: mk("pv_conv"), norm: mk("pv_norm"), unshuf: mk("pv_unshuf"), shuf: mk("pv_shuf"), layout };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("primal_vae kernels rejected: {e}");
                return None;
            }
            Some(p)
        })
        .as_ref()
}

struct DevWeights {
    key: u64,
    convs: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    gammas: Vec<wgpu::Buffer>,
    dummy: wgpu::Buffer,
}

struct State {
    weights: Option<DevWeights>,
    bufs: Vec<(wgpu::Buffer, u64)>,
}

static STATE: Mutex<State> = Mutex::new(State { weights: None, bufs: Vec::new() });

fn storage(c: &Ctx, label: &str, data: &[f32]) -> wgpu::Buffer {
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: ((data.len().max(1)) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
    }
    b
}

fn grid(n: usize) -> (u32, u32) {
    let wgs = n.div_ceil(256).max(1);
    let x = wgs.min(65_535);
    (x as u32, wgs.div_ceil(x) as u32)
}

/// Run `plan` on the device. `false` = refused (no device, a buffer over
/// the binding limit, a device error); the caller runs the host.
pub fn run(key: u64, wts: &PvWeights, plan: &PvPlan, input: &[f32], out: &mut [f32]) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(p) = pipes(c) else { return false };
    let _op = c.op_gate.enter();
    let lim = c.device.limits();
    let max_bind = lim.max_storage_buffer_binding_size as u64;
    if plan.buf_elems.iter().any(|&n| (n * 4) as u64 > max_bind.min(lim.max_buffer_size)) {
        return false;
    }
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut st = STATE.lock().unwrap();
    if st.weights.as_ref().is_none_or(|w| w.key != key) {
        st.weights = Some(DevWeights {
            key,
            convs: wts.convs.iter().map(|cv| (storage(c, "pv_w", &cv.w), storage(c, "pv_b", &cv.b))).collect(),
            gammas: wts.gammas.iter().map(|g| storage(c, "pv_g", g)).collect(),
            dummy: storage(c, "pv_dummy", &[0.0]),
        });
    }
    while st.bufs.len() < plan.buf_elems.len() {
        st.bufs.push((c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pv_act"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }), 4));
    }
    for (i, &n) in plan.buf_elems.iter().enumerate() {
        let need = ((n.max(1)) * 4) as u64;
        if st.bufs[i].1 < need {
            st.bufs[i] = (c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pv_act"),
                size: need,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }), need);
        }
    }
    if pollster::block_on(sc.pop()).is_some() {
        st.bufs.clear();
        return false;
    }
    let st = &*st;
    let dw = st.weights.as_ref().expect("uploaded above");
    let buf = |i: u8| &st.bufs[i as usize].0;
    c.queue.write_buffer(buf(plan.input), 0, bytemuck::cast_slice(input));
    // Per-step uniforms, one 256-byte slot each.
    let slot = 256u64;
    let ubuf = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pv_p"),
        size: slot * plan.steps.len().max(1) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut ubytes = vec![0u8; (slot as usize) * plan.steps.len().max(1)];
    let mut binds = Vec::with_capacity(plan.steps.len());
    for (si, s) in plan.steps.iter().enumerate() {
        let (pr, srcb, wb, bb, dstb, groups, pipe): ([u32; 8], u8, &wgpu::Buffer, &wgpu::Buffer, u8, (u32, u32), &wgpu::ComputePipeline) = match *s {
            PvStep::Conv { wi, src, dst, res, dil, h, w } => {
                let cv = &wts.convs[wi];
                debug_assert!(res.is_none_or(|r| r == dst));
                let (wb, bb) = &dw.convs[wi];
                (
                    [cv.cin as u32, cv.cout as u32, h as u32, w as u32, dil, res.is_some() as u32, (cv.cin * 9) as u32, 0],
                    src, wb, bb, dst,
                    ((h * w).div_ceil(64) as u32, cv.cout.div_ceil(64) as u32),
                    &p.conv,
                )
            }
            PvStep::NormSilu { gi, src, dst, c: ch, h, w } => (
                [ch as u32, 0, h as u32, w as u32, 0, 0, 0, 0],
                src, &dw.gammas[gi], &dw.dummy, dst, grid(h * w), &p.norm,
            ),
            PvStep::Unshuffle { src, dst, c: ch, h, w } => (
                [ch as u32, 0, h as u32, w as u32, 0, 0, 0, (ch * h * w) as u32],
                src, &dw.dummy, &dw.dummy, dst, grid(ch * h * w), &p.unshuf,
            ),
            PvStep::Shuffle { src, dst, add, c: ch, h, w } => {
                debug_assert!(add.is_none_or(|a| a == dst));
                (
                    [ch as u32, 0, h as u32, w as u32, 0, add.is_some() as u32, 0, (ch * h * w) as u32],
                    src, &dw.dummy, &dw.dummy, dst, grid(ch * h * w), &p.shuf,
                )
            }
        };
        if groups.0 > 65_535 || groups.1 > 65_535 {
            return false;
        }
        ubytes[si * slot as usize..si * slot as usize + 32].copy_from_slice(bytemuck::cast_slice(&pr));
        let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pv_bg"),
            layout: &p.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf(srcb).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf(dstb).as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &ubuf,
                        offset: si as u64 * slot,
                        size: std::num::NonZeroU64::new(32),
                    }),
                },
            ],
        });
        binds.push((bg, groups, pipe));
    }
    c.queue.write_buffer(&ubuf, 0, &ubytes);
    let t0 = std::time::Instant::now();
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("primal_vae") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("primal_vae"), timestamp_writes: None });
        for (bg, (gx, gy), pipe) in &binds {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(*gx, *gy, 1);
        }
    }
    let bytes = (plan.out_elems * 4) as u64;
    let staging = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pv_rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(buf(plan.output), 0, &staging, 0, bytes);
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
        out[..plan.out_elems].copy_from_slice(bytemuck::cast_slice(&data[..bytes as usize]));
    }
    staging.unmap();
    if std::env::var_os("CMF_PRIMAL_VAE_PROF").is_some() {
        eprintln!("primal_vae (wgpu): {} steps, {:.3} s on the device", plan.steps.len(), t0.elapsed().as_secs_f64());
    }
    true
}
