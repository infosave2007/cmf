//! Ripple's value residual inside the wgpu graphs: a layer's V projection
//! reads `lerp(e0, normed, blend)` — the row the layer's input norm wrote,
//! pulled toward the scaled embedding of the same position. One dispatch per
//! layer over all rows of the call, in the form `pipeline::vres_lerp` uses on
//! the host (the smaller of `blend` and `1 − blend` multiplies the difference).

use super::Ctx;
use std::sync::OnceLock;

const LERP_SRC: &str = r#"
struct VP { blend: f32, n: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> vx: array<f32>;
@group(0) @binding(1) var<storage, read> ve: array<f32>;
@group(0) @binding(2) var<storage, read_write> vy: array<f32>;
@group(0) @binding(3) var<uniform> vp: VP;

@compute @workgroup_size(256)
fn vres_lerp(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= vp.n) { return; }
    let e = ve[i];
    let x = vx[i];
    if (vp.blend < 0.5) { vy[i] = e + vp.blend * (x - e); } else { vy[i] = x - (x - e) * (1.0 - vp.blend); }
}
"#;

static PIPE: OnceLock<Option<wgpu::ComputePipeline>> = OnceLock::new();

pub(crate) fn pipe(c: &Ctx) -> Option<&'static wgpu::ComputePipeline> {
    PIPE.get_or_init(|| {
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vres_lerp"),
            source: wgpu::ShaderSource::Wgsl(LERP_SRC.into()),
        });
        let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vres_lerp"),
            layout: None,
            module: &m,
            entry_point: Some("vres_lerp"),
            compilation_options: crate::gpu_wgpu::pco(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(sc.pop()) {
            tracing::warn!("value-residual kernel rejected: {e}");
            return None;
        }
        Some(p)
    })
    .as_ref()
}

/// The uniform words of one layer's blend over `n` floats.
pub(crate) fn params(blend: f32, n: usize) -> [u32; 4] {
    [blend.to_bits(), n as u32, 0, 0]
}

/// The workgroup grid for `n` floats (≤ 65 535 per axis).
pub(crate) fn grid(n: usize) -> (u32, u32) {
    let wgs = n.div_ceil(256).max(1);
    let x = wgs.min(65_535);
    (x as u32, wgs.div_ceil(x) as u32)
}

/// Bind group: normed rows, embedding rows, the V input, the uniform.
pub(crate) fn bind(c: &Ctx, p: &wgpu::ComputePipeline, bufs: [&wgpu::Buffer; 4]) -> wgpu::BindGroup {
    let layout = p.get_bind_group_layout(0);
    let entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("vres"), layout: &layout, entries: &entries })
}
