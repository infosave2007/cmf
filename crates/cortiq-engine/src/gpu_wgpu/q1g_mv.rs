//! Subgroup q1g decode matvec for the whole-token graph and the per-op path.
//!
//! `q1g_matvec` keeps `q1_matvec`'s workgroup shape on purpose (that is what
//! makes it bit-identical to q1): every 1024 columns it stages the
//! activation through shared memory behind a barrier, and every eight rows
//! it pays a four-step shared-memory tree with a barrier per step. On
//! Bonsai-1.7B's decode shapes (2048-6144 columns) those barriers, not the
//! sign arithmetic, are most of the kernel: the 5090 moved 30-76 GB/s.
//!
//! `q1g_mv_sg` gives each half of a 32-lane subgroup one row. A lane walks
//! exactly the tile pairs its `q1g_matvec` lane walks (pairs l, l+16, …),
//! with the same per-pair `acc + s·Σ(tile0) + s·Σ(tile1)` over the same
//! values: the activation comes straight from the (L1-resident) input
//! instead of a staged copy. The row tree is the shared-memory tree's own
//! order (stride 8, 4, 2, 1; own + partner) done with subgroup shuffles, so
//! the output is `q1g_matvec`'s — and therefore q1's — bit for bit, with no
//! barrier and no shared memory at all.
//!
//! Own module (subgroup builtins must never reach a device without the
//! feature), built lazily; 32-wide subgroups only. Opt-in for long rows
//! (see `wanted` for the measurements that keep it off by default).

use super::*;

pub(crate) struct Q1gSg {
    pub(crate) mv: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
}

/// Rows per 256-thread workgroup: eight 32-lane subgroups, two rows each.
pub(crate) const ROWS_PER_WG: u32 = 16;

// No `enable subgroups;` directive: this Naga rejects it as "not yet
// supported" and accepts the subgroup builtins on a device created with
// `Features::SUBGROUP` (as the `dense_mv` kernels rely on).
const SRC: &str = r#"
// np = tile pairs per row (cols/64); sbase = word offset of the f16 scale
// plane (behind the sign plane of ALL rows of the tensor).
struct P { np: u32, rows: u32, sbase: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       gw : array<u32>;
@group(0) @binding(1) var<storage, read>       gx : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> gy : array<f32>;
@group(0) @binding(3) var<uniform>             gp : P;

// q1_tile_sum with the activation read as eight vec4 from the input:
// the same values, the same vec4 accumulation, the same final sum.
fn tile_sum(bits: u32, x4: u32) -> f32 {
    var s = vec4<f32>(0.0);
    for (var j = 0u; j < 8u; j = j + 1u) {
        let nib = bits >> (j * 4u);
        let x = gx[x4 + j];
        let m = vec4<u32>(
            ((nib & 1u) ^ 1u) << 31u,
            (((nib >> 1u) & 1u) ^ 1u) << 31u,
            (((nib >> 2u) & 1u) ^ 1u) << 31u,
            (((nib >> 3u) & 1u) ^ 1u) << 31u);
        s = s + bitcast<vec4<f32>>(bitcast<vec4<u32>>(x) ^ m);
    }
    return s.x + s.y + s.z + s.w;
}

@compute @workgroup_size(256)
fn q1g_mv_sg(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32,
             @builtin(subgroup_invocation_id) sl: u32) {
    let gpr = gp.np * 2u;
    let half = sl >> 4u;
    let lane = sl & 15u;
    // Every loop bound below is workgroup-uniform (WGSL's uniformity rules
    // for the shuffles): rows and pairs a lane does not own are masked by
    // `if`, never by an early exit, exactly as q1_matvec masks them.
    var blk = wid.x;
    loop {
        if (blk * 16u >= gp.rows) { break; }
        let row = blk * 16u + (lid >> 5u) * 2u + half;
        let live = row < gp.rows;
        var acc = 0.0;
        for (var ti = 0u; ti < gp.np; ti = ti + 16u) {
            let pi = ti + lane;
            if (live && pi < gp.np) {
                let base = row * gpr + pi * 2u;
                let b0 = gw[base]; let b1 = gw[base + 1u];
                let g = row * (gpr >> 2u) + (pi >> 1u);
                let sw = unpack2x16float(gw[gp.sbase + (g >> 1u)]);
                let s = select(sw.x, sw.y, (g & 1u) == 1u);
                let x4 = pi * 16u;
                acc = acc + s * tile_sum(b0, x4) + s * tile_sum(b1, x4 + 8u);
            }
        }
        var v = acc;
        v = v + subgroupShuffleDown(v, 8u);
        v = v + subgroupShuffleDown(v, 4u);
        v = v + subgroupShuffleDown(v, 2u);
        v = v + subgroupShuffleDown(v, 1u);
        if (lane == 0u && live) { gy[row] = v; }
        blk = blk + nwg.x;
    }
}
"#;

/// Validation error of the subgroup module on this device, if any (None
/// also when the device has no 32-wide subgroups). For tests: `build`
/// only logs it.
#[cfg(test)]
pub(crate) fn module_error(c: &Ctx) -> Option<String> {
    if !sg32(c) {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-q1g-sg-test"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let _ = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q1g_mv_sg-test"),
            layout: None,
            module: &module,
            entry_point: Some("q1g_mv_sg"),
            compilation_options: Default::default(),
            cache: None,
        });
    pollster::block_on(scope.pop()).map(|e| format!("{e:?}"))
}

pub(crate) fn sg32(c: &Ctx) -> bool {
    c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
}

/// Take the subgroup kernel for a `cols`-wide matvec? Opt-in
/// (`CMF_Q1G_SG=1`), and then only for rows of ≥ 4096 columns, because the
/// measurements disagree by device (`q1g_matvec_speed`, interleaved runs,
/// both cards shared with another job):
/// - RTX 5090: down_proj 2048×6144 took 23-25 µs against 43-61 staged and
///   44-54 for q1 (four runs). lm_head 151669×2048 took 239-419 against
///   117-168 staged: every row re-reads the activation from L1, where the
///   staged kernel shares one shared-memory copy across eight rows.
/// - RTX 2000 Ada: slower on every shape, down_proj included (329-367
///   against 219-252 staged).
///
/// End to end (Bonsai-1.7B decode, tok/s, medians of 6), staged-only vs
/// with this kernel on down_proj: 113.8 vs 116.1 on the 5090, 25.9 vs 24.2
/// on the 2000 Ada (q1: 110.5 and 25.0). The default stays the staged kernel.
/// Same bits either way.
pub(crate) fn wanted(cols: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_Q1G_SG").as_deref() == Ok("1")) && cols >= 4096
}

fn build(c: &Ctx) -> Option<Q1gSg> {
    if !sg32(c) {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-q1g-sg"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let mv = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q1g_mv_sg"),
            layout: None,
            module: &module,
            entry_point: Some("q1g_mv_sg"),
            compilation_options: Default::default(),
            cache: c.pipeline_cache.as_ref(),
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-q1g-sg module rejected ({e}): q1g keeps the staged kernel");
        return None;
    }
    Some(Q1gSg {
        layout: mv.get_bind_group_layout(0),
        mv,
    })
}

pub(crate) fn pipes(c: &Ctx) -> Option<&Q1gSg> {
    c.q1g_sg_pipes.get_or_init(|| build(c)).as_ref()
}
