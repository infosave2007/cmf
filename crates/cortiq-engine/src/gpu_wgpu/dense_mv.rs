//! Warp-per-row q4tp decode matvecs for the whole-token graph.
//!
//! `q4tp_matvec16nl` gives a 64-lane sub-block four rows, walks the row's
//! 16-byte groups two and a half at a time (5120 columns) and then pays a
//! six-barrier workgroup tree for every sixteen rows. On an RTX 3090 that
//! held Qwen3.8-27B's narrow projections at 37-52 % of the bus (GDN qkv+z
//! 349 GB/s, out_proj 357, FFN down 488). These kernels give a 32-lane
//! subgroup R rows (`CMF_MV_SG_R`, 1/2/4/8), U groups a lane a step
//! (`CMF_MV_SG_U`, 2/4) with every load of the step issued before the
//! first FMA (clamped, unconditional), no barrier at all, and finish with
//! subgroup shuffles.
//!
//! Each row is the 16nl row to the bit: lane l holds the partials of the
//! 16nl lanes l (groups l, l+64, …) and l+32 (groups l+32, l+96, …) in two
//! accumulators, adds them (the tree's stride-32 step), and the shuffle
//! steps 16, 8, 4, 2, 1 take the partner's partial in the tree's own
//! order. Up to three matrices of one input per dispatch, and the fused
//! gate+up+SiLU of `q4tp_matvec16nl_gu`.
//!
//! `CMF_MV_SG=0` keeps the 16nl kernels.

use super::*;

pub(crate) struct MvSg {
    pub(crate) mv: wgpu::ComputePipeline,
    pub(crate) mv_l: wgpu::BindGroupLayout,
    pub(crate) gu: wgpu::ComputePipeline,
    pub(crate) gu_l: wgpu::BindGroupLayout,
    /// Rows a subgroup owns.
    pub(crate) rpw: usize,
    /// Rows a subgroup owns in the fused gate+up.
    pub(crate) grpw: usize,
    /// Never-written output for the unused matrix slots.
    dummy_y: wgpu::Buffer,
}

fn sg32(c: &Ctx) -> bool {
    c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
}

fn shape(var_r: &str, var_u: &str, r_default: usize) -> (usize, usize) {
    let r = match std::env::var(var_r).as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        Ok("4") => 4,
        Ok("8") => 8,
        _ => r_default,
    };
    let u = match std::env::var(var_u).as_deref() {
        Ok("4") => 4,
        _ => 2,
    };
    (r, u)
}

fn compile(c: &Ctx, label: &str, src: &str, entry: &str) -> Option<wgpu::ComputePipeline> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let p = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: c.pipeline_cache.as_ref(),
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("{label} module rejected ({e}): q4tp decode keeps the 16nl kernels");
        return None;
    }
    Some(p)
}

fn build(c: &Ctx) -> Option<MvSg> {
    if std::env::var("CMF_MV_SG").as_deref() == Ok("0") || !sg32(c) {
        return None;
    }
    // Rows a subgroup owns (R) and 16-byte groups a lane loads a step (U);
    // every load of a step is issued before its first FMA. The fused
    // gate+up carries two weight rows per output row, so it has its own.
    let (rpw, unroll) = shape("CMF_MV_SG_R", "CMF_MV_SG_U", 4);
    let (grpw, gunroll) = shape("CMF_GU_SG_R", "CMF_GU_SG_U", 2);
    let src = match (rpw, unroll) {
        (1, _) => MV_SG_R1U2,
        (2, 4) => MV_SG_R2U4,
        (2, _) => MV_SG_R2U2,
        (8, _) => MV_SG_R8U2,
        (_, 4) => MV_SG_R4U4,
        _ => MV_SG_R4U2,
    };
    let (gsrc, grpw) = match (grpw, gunroll) {
        (1, 4) => (GU_SG_R1U4, 1),
        (1, _) => (GU_SG_R1U2, 1),
        (4, _) => (GU_SG_R4U2, 4),
        (_, 4) => (GU_SG_R2U4, 2),
        _ => (GU_SG_R2U2, 2),
    };
    let mv = compile(c, "cmf-mv-sg", src, "q4tp_mv_sg")?;
    let gu = compile(c, "cmf-gu-sg", gsrc, "q4tp_gu_sg")?;
    let dummy_y = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mv-sg-dummy"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    Some(MvSg {
        mv_l: mv.get_bind_group_layout(0),
        mv,
        gu_l: gu.get_bind_group_layout(0),
        gu,
        rpw,
        grpw,
        dummy_y,
    })
}

pub(crate) fn pipes(c: &Ctx) -> Option<&MvSg> {
    c.dense_mv_pipes.get_or_init(|| build(c)).as_ref()
}

/// Up to three q4tp matvecs of one input `xs` (`mats`: weight, output,
/// rows) as ONE dispatch: (pipeline, bind group, workgroups). None when the
/// kernel is missing or the width is not whole groups.
pub(crate) fn sg_prep<'a>(
    c: &'a Ctx,
    xs: &wgpu::Buffer,
    cols: usize,
    mats: &[(&wgpu::Buffer, &wgpu::Buffer, usize)],
) -> Option<(&'a wgpu::ComputePipeline, wgpu::BindGroup, u32)> {
    let p = pipes(c)?;
    if cols % 32 != 0 || mats.is_empty() || mats.len() > 3 {
        return None;
    }
    let per = 8 * p.rpw;
    let r = |i: usize| mats.get(i).map_or(0, |m| m.2);
    let groups: usize = (0..3).map(|i| r(i).div_ceil(per)).sum();
    if groups == 0 || groups as u32 > MAX_WG {
        return None;
    }
    let u = uniform_u32x8(
        c,
        [r(0) as u32, r(1) as u32, r(2) as u32, (cols / 32) as u32, 0, 0, 0, 0],
    );
    let w = |i: usize| mats.get(i).map_or(mats[0].0, |m| m.0);
    let y = |i: usize| mats.get(i).map_or(&p.dummy_y, |m| m.1);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mv-sg"),
        layout: &p.mv_l,
        entries: &[
            bind_buf(0, w(0)),
            bind_buf(1, w(1)),
            bind_buf(2, w(2)),
            bind_buf(3, w(0)),
            bind_buf(4, w(1)),
            bind_buf(5, w(2)),
            bind_buf(6, xs),
            bind_buf(7, y(0)),
            bind_buf(8, y(1)),
            bind_buf(9, y(2)),
            bind_buf(10, &u),
        ],
    });
    Some((&p.mv, bind, groups as u32))
}

/// gate+up+activation of `q4tp_matvec16nl_gu` (act code 0 SiLU, 1 exact
/// GELU; `lim` the swiglu limit, 0 for none) as ONE dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sg_gu_prep<'a>(
    c: &'a Ctx,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    act: &wgpu::Buffer,
    inter: usize,
    cols: usize,
    act_code: u32,
) -> Option<(&'a wgpu::ComputePipeline, wgpu::BindGroup, u32)> {
    let p = pipes(c)?;
    if cols % 32 != 0 || inter == 0 {
        return None;
    }
    let groups = inter.div_ceil(8 * p.grpw);
    if groups as u32 > MAX_WG {
        return None;
    }
    let u = uniform_u32x8(
        c,
        [inter as u32, inter as u32, 0, (cols / 32) as u32, 0, act_code, 0, 0],
    );
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gu-sg"),
        layout: &p.gu_l,
        entries: &[
            bind_buf(0, gate),
            bind_buf(1, up),
            bind_buf(3, gate),
            bind_buf(4, up),
            bind_buf(6, xs),
            bind_buf(7, act),
            bind_buf(10, &u),
        ],
    });
    Some((&p.gu, bind, groups as u32))
}

pub(crate) const MV_SG_R1U2: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_a(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 1u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wa[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_a(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_a(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4a[q0 * gpr + g0];
        var cvm01 = sg_byte_a(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_a(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4a[q0 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    if (lane == 0u) {
        if (l0) { sg_ya[r0] = t0; }
    }
}
fn run_b(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_b;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 1u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wb[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_b(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_b(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4b[q0 * gpr + g0];
        var cvm01 = sg_byte_b(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_b(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4b[q0 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    if (lane == 0u) {
        if (l0) { sg_yb[r0] = t0; }
    }
}
fn run_c(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_c;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 1u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wc[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_c(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_c(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4c[q0 * gpr + g0];
        var cvm01 = sg_byte_c(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_c(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4c[q0 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    if (lane == 0u) {
        if (l0) { sg_yc[r0] = t0; }
    }
}

@compute @workgroup_size(256)
fn q4tp_mv_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    let per = 8u * 1u;
    let warp = lid >> 5u;
    let ba = (sg_p.rows_a + per - 1u) / per;
    let bb = (sg_p.rows_b + per - 1u) / per;
    let wb = wid.x;
    if (wb < ba) {
        run_a(wb, warp, lane);
    } else if (wb < ba + bb) {
        run_b(wb - ba, warp, lane);
    } else {
        run_c(wb - ba - bb, warp, lane);
    }
}

"#;

pub(crate) const MV_SG_R2U2: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_a(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 2u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wa[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wa[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_a(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_a(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4a[q0 * gpr + g0];
        var cvm01 = sg_byte_a(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_a(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4a[q0 * gpr + g1];
        var cvm10 = sg_byte_a(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_a(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4a[q1 * gpr + g0];
        var cvm11 = sg_byte_a(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_a(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4a[q1 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    if (lane == 0u) {
        if (l0) { sg_ya[r0] = t0; }
        if (l1) { sg_ya[r1] = t1; }
    }
}
fn run_b(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_b;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 2u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wb[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wb[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_b(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_b(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4b[q0 * gpr + g0];
        var cvm01 = sg_byte_b(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_b(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4b[q0 * gpr + g1];
        var cvm10 = sg_byte_b(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_b(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4b[q1 * gpr + g0];
        var cvm11 = sg_byte_b(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_b(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4b[q1 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    if (lane == 0u) {
        if (l0) { sg_yb[r0] = t0; }
        if (l1) { sg_yb[r1] = t1; }
    }
}
fn run_c(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_c;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 2u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wc[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wc[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_c(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_c(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4c[q0 * gpr + g0];
        var cvm01 = sg_byte_c(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_c(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4c[q0 * gpr + g1];
        var cvm10 = sg_byte_c(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_c(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4c[q1 * gpr + g0];
        var cvm11 = sg_byte_c(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_c(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4c[q1 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    if (lane == 0u) {
        if (l0) { sg_yc[r0] = t0; }
        if (l1) { sg_yc[r1] = t1; }
    }
}

@compute @workgroup_size(256)
fn q4tp_mv_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    let per = 8u * 2u;
    let warp = lid >> 5u;
    let ba = (sg_p.rows_a + per - 1u) / per;
    let bb = (sg_p.rows_b + per - 1u) / per;
    let wb = wid.x;
    if (wb < ba) {
        run_a(wb, warp, lane);
    } else if (wb < ba + bb) {
        run_b(wb - ba, warp, lane);
    } else {
        run_c(wb - ba - bb, warp, lane);
    }
}

"#;

pub(crate) const MV_SG_R4U2: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_a(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 4u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wa[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wa[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wa[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wa[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_a(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_a(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4a[q0 * gpr + g0];
        var cvm01 = sg_byte_a(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_a(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4a[q0 * gpr + g1];
        var cvm10 = sg_byte_a(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_a(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4a[q1 * gpr + g0];
        var cvm11 = sg_byte_a(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_a(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4a[q1 * gpr + g1];
        var cvm20 = sg_byte_a(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_a(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4a[q2 * gpr + g0];
        var cvm21 = sg_byte_a(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_a(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4a[q2 * gpr + g1];
        var cvm30 = sg_byte_a(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_a(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4a[q3 * gpr + g0];
        var cvm31 = sg_byte_a(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_a(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4a[q3 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    if (lane == 0u) {
        if (l0) { sg_ya[r0] = t0; }
        if (l1) { sg_ya[r1] = t1; }
        if (l2) { sg_ya[r2] = t2; }
        if (l3) { sg_ya[r3] = t3; }
    }
}
fn run_b(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_b;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 4u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wb[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wb[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wb[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wb[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_b(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_b(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4b[q0 * gpr + g0];
        var cvm01 = sg_byte_b(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_b(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4b[q0 * gpr + g1];
        var cvm10 = sg_byte_b(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_b(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4b[q1 * gpr + g0];
        var cvm11 = sg_byte_b(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_b(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4b[q1 * gpr + g1];
        var cvm20 = sg_byte_b(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_b(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4b[q2 * gpr + g0];
        var cvm21 = sg_byte_b(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_b(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4b[q2 * gpr + g1];
        var cvm30 = sg_byte_b(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_b(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4b[q3 * gpr + g0];
        var cvm31 = sg_byte_b(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_b(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4b[q3 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    if (lane == 0u) {
        if (l0) { sg_yb[r0] = t0; }
        if (l1) { sg_yb[r1] = t1; }
        if (l2) { sg_yb[r2] = t2; }
        if (l3) { sg_yb[r3] = t3; }
    }
}
fn run_c(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_c;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 4u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wc[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wc[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wc[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wc[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_c(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_c(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4c[q0 * gpr + g0];
        var cvm01 = sg_byte_c(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_c(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4c[q0 * gpr + g1];
        var cvm10 = sg_byte_c(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_c(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4c[q1 * gpr + g0];
        var cvm11 = sg_byte_c(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_c(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4c[q1 * gpr + g1];
        var cvm20 = sg_byte_c(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_c(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4c[q2 * gpr + g0];
        var cvm21 = sg_byte_c(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_c(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4c[q2 * gpr + g1];
        var cvm30 = sg_byte_c(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_c(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4c[q3 * gpr + g0];
        var cvm31 = sg_byte_c(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_c(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4c[q3 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    if (lane == 0u) {
        if (l0) { sg_yc[r0] = t0; }
        if (l1) { sg_yc[r1] = t1; }
        if (l2) { sg_yc[r2] = t2; }
        if (l3) { sg_yc[r3] = t3; }
    }
}

@compute @workgroup_size(256)
fn q4tp_mv_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    let per = 8u * 4u;
    let warp = lid >> 5u;
    let ba = (sg_p.rows_a + per - 1u) / per;
    let bb = (sg_p.rows_b + per - 1u) / per;
    let wb = wid.x;
    if (wb < ba) {
        run_a(wb, warp, lane);
    } else if (wb < ba + bb) {
        run_b(wb - ba, warp, lane);
    } else {
        run_c(wb - ba - bb, warp, lane);
    }
}

"#;

pub(crate) const MV_SG_R8U2: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_a(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 8u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wa[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wa[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wa[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wa[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    let r4 = base + 4u;
    let l4 = r4 < rows;
    let q4 = select(base, r4, l4);
    let cm4 = codes_b + q4 * cstride;
    let pm4 = unpack2x16float(sg_wa[params_w + q4]);
    var ma4 = 0.0;
    var mb4 = 0.0;
    let r5 = base + 5u;
    let l5 = r5 < rows;
    let q5 = select(base, r5, l5);
    let cm5 = codes_b + q5 * cstride;
    let pm5 = unpack2x16float(sg_wa[params_w + q5]);
    var ma5 = 0.0;
    var mb5 = 0.0;
    let r6 = base + 6u;
    let l6 = r6 < rows;
    let q6 = select(base, r6, l6);
    let cm6 = codes_b + q6 * cstride;
    let pm6 = unpack2x16float(sg_wa[params_w + q6]);
    var ma6 = 0.0;
    var mb6 = 0.0;
    let r7 = base + 7u;
    let l7 = r7 < rows;
    let q7 = select(base, r7, l7);
    let cm7 = codes_b + q7 * cstride;
    let pm7 = unpack2x16float(sg_wa[params_w + q7]);
    var ma7 = 0.0;
    var mb7 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_a(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_a(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4a[q0 * gpr + g0];
        var cvm01 = sg_byte_a(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_a(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4a[q0 * gpr + g1];
        var cvm10 = sg_byte_a(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_a(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4a[q1 * gpr + g0];
        var cvm11 = sg_byte_a(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_a(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4a[q1 * gpr + g1];
        var cvm20 = sg_byte_a(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_a(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4a[q2 * gpr + g0];
        var cvm21 = sg_byte_a(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_a(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4a[q2 * gpr + g1];
        var cvm30 = sg_byte_a(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_a(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4a[q3 * gpr + g0];
        var cvm31 = sg_byte_a(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_a(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4a[q3 * gpr + g1];
        var cvm40 = sg_byte_a(cm4 + cbo0);
        if (sh0 > 3u) { cvm40 = cvm40 | (sg_byte_a(cm4 + cbo0 + 1u) << 8u); }
        let vm40 = sg_w4a[q4 * gpr + g0];
        var cvm41 = sg_byte_a(cm4 + cbo1);
        if (sh1 > 3u) { cvm41 = cvm41 | (sg_byte_a(cm4 + cbo1 + 1u) << 8u); }
        let vm41 = sg_w4a[q4 * gpr + g1];
        var cvm50 = sg_byte_a(cm5 + cbo0);
        if (sh0 > 3u) { cvm50 = cvm50 | (sg_byte_a(cm5 + cbo0 + 1u) << 8u); }
        let vm50 = sg_w4a[q5 * gpr + g0];
        var cvm51 = sg_byte_a(cm5 + cbo1);
        if (sh1 > 3u) { cvm51 = cvm51 | (sg_byte_a(cm5 + cbo1 + 1u) << 8u); }
        let vm51 = sg_w4a[q5 * gpr + g1];
        var cvm60 = sg_byte_a(cm6 + cbo0);
        if (sh0 > 3u) { cvm60 = cvm60 | (sg_byte_a(cm6 + cbo0 + 1u) << 8u); }
        let vm60 = sg_w4a[q6 * gpr + g0];
        var cvm61 = sg_byte_a(cm6 + cbo1);
        if (sh1 > 3u) { cvm61 = cvm61 | (sg_byte_a(cm6 + cbo1 + 1u) << 8u); }
        let vm61 = sg_w4a[q6 * gpr + g1];
        var cvm70 = sg_byte_a(cm7 + cbo0);
        if (sh0 > 3u) { cvm70 = cvm70 | (sg_byte_a(cm7 + cbo0 + 1u) << 8u); }
        let vm70 = sg_w4a[q7 * gpr + g0];
        var cvm71 = sg_byte_a(cm7 + cbo1);
        if (sh1 > 3u) { cvm71 = cvm71 | (sg_byte_a(cm7 + cbo1 + 1u) << 8u); }
        let vm71 = sg_w4a[q7 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        ma4 = ma4 + exp2(pm4.x + f32((cvm40 >> sh0) & 31u) * pm4.y)
            * (sg_dot8(vm40.x, xa0, xb0) + sg_dot8(vm40.y, xc0, xd0)
             + sg_dot8(vm40.z, xe0, xf0) + sg_dot8(vm40.w, xg0, xh0));
        ma5 = ma5 + exp2(pm5.x + f32((cvm50 >> sh0) & 31u) * pm5.y)
            * (sg_dot8(vm50.x, xa0, xb0) + sg_dot8(vm50.y, xc0, xd0)
             + sg_dot8(vm50.z, xe0, xf0) + sg_dot8(vm50.w, xg0, xh0));
        ma6 = ma6 + exp2(pm6.x + f32((cvm60 >> sh0) & 31u) * pm6.y)
            * (sg_dot8(vm60.x, xa0, xb0) + sg_dot8(vm60.y, xc0, xd0)
             + sg_dot8(vm60.z, xe0, xf0) + sg_dot8(vm60.w, xg0, xh0));
        ma7 = ma7 + exp2(pm7.x + f32((cvm70 >> sh0) & 31u) * pm7.y)
            * (sg_dot8(vm70.x, xa0, xb0) + sg_dot8(vm70.y, xc0, xd0)
             + sg_dot8(vm70.z, xe0, xf0) + sg_dot8(vm70.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
            mb4 = mb4 + exp2(pm4.x + f32((cvm41 >> sh1) & 31u) * pm4.y)
                * (sg_dot8(vm41.x, xa1, xb1) + sg_dot8(vm41.y, xc1, xd1)
                 + sg_dot8(vm41.z, xe1, xf1) + sg_dot8(vm41.w, xg1, xh1));
            mb5 = mb5 + exp2(pm5.x + f32((cvm51 >> sh1) & 31u) * pm5.y)
                * (sg_dot8(vm51.x, xa1, xb1) + sg_dot8(vm51.y, xc1, xd1)
                 + sg_dot8(vm51.z, xe1, xf1) + sg_dot8(vm51.w, xg1, xh1));
            mb6 = mb6 + exp2(pm6.x + f32((cvm61 >> sh1) & 31u) * pm6.y)
                * (sg_dot8(vm61.x, xa1, xb1) + sg_dot8(vm61.y, xc1, xd1)
                 + sg_dot8(vm61.z, xe1, xf1) + sg_dot8(vm61.w, xg1, xh1));
            mb7 = mb7 + exp2(pm7.x + f32((cvm71 >> sh1) & 31u) * pm7.y)
                * (sg_dot8(vm71.x, xa1, xb1) + sg_dot8(vm71.y, xc1, xd1)
                 + sg_dot8(vm71.z, xe1, xf1) + sg_dot8(vm71.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    let t4 = sg_tree(ma4 + mb4);
    let t5 = sg_tree(ma5 + mb5);
    let t6 = sg_tree(ma6 + mb6);
    let t7 = sg_tree(ma7 + mb7);
    if (lane == 0u) {
        if (l0) { sg_ya[r0] = t0; }
        if (l1) { sg_ya[r1] = t1; }
        if (l2) { sg_ya[r2] = t2; }
        if (l3) { sg_ya[r3] = t3; }
        if (l4) { sg_ya[r4] = t4; }
        if (l5) { sg_ya[r5] = t5; }
        if (l6) { sg_ya[r6] = t6; }
        if (l7) { sg_ya[r7] = t7; }
    }
}
fn run_b(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_b;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 8u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wb[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wb[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wb[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wb[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    let r4 = base + 4u;
    let l4 = r4 < rows;
    let q4 = select(base, r4, l4);
    let cm4 = codes_b + q4 * cstride;
    let pm4 = unpack2x16float(sg_wb[params_w + q4]);
    var ma4 = 0.0;
    var mb4 = 0.0;
    let r5 = base + 5u;
    let l5 = r5 < rows;
    let q5 = select(base, r5, l5);
    let cm5 = codes_b + q5 * cstride;
    let pm5 = unpack2x16float(sg_wb[params_w + q5]);
    var ma5 = 0.0;
    var mb5 = 0.0;
    let r6 = base + 6u;
    let l6 = r6 < rows;
    let q6 = select(base, r6, l6);
    let cm6 = codes_b + q6 * cstride;
    let pm6 = unpack2x16float(sg_wb[params_w + q6]);
    var ma6 = 0.0;
    var mb6 = 0.0;
    let r7 = base + 7u;
    let l7 = r7 < rows;
    let q7 = select(base, r7, l7);
    let cm7 = codes_b + q7 * cstride;
    let pm7 = unpack2x16float(sg_wb[params_w + q7]);
    var ma7 = 0.0;
    var mb7 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_b(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_b(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4b[q0 * gpr + g0];
        var cvm01 = sg_byte_b(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_b(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4b[q0 * gpr + g1];
        var cvm10 = sg_byte_b(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_b(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4b[q1 * gpr + g0];
        var cvm11 = sg_byte_b(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_b(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4b[q1 * gpr + g1];
        var cvm20 = sg_byte_b(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_b(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4b[q2 * gpr + g0];
        var cvm21 = sg_byte_b(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_b(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4b[q2 * gpr + g1];
        var cvm30 = sg_byte_b(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_b(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4b[q3 * gpr + g0];
        var cvm31 = sg_byte_b(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_b(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4b[q3 * gpr + g1];
        var cvm40 = sg_byte_b(cm4 + cbo0);
        if (sh0 > 3u) { cvm40 = cvm40 | (sg_byte_b(cm4 + cbo0 + 1u) << 8u); }
        let vm40 = sg_w4b[q4 * gpr + g0];
        var cvm41 = sg_byte_b(cm4 + cbo1);
        if (sh1 > 3u) { cvm41 = cvm41 | (sg_byte_b(cm4 + cbo1 + 1u) << 8u); }
        let vm41 = sg_w4b[q4 * gpr + g1];
        var cvm50 = sg_byte_b(cm5 + cbo0);
        if (sh0 > 3u) { cvm50 = cvm50 | (sg_byte_b(cm5 + cbo0 + 1u) << 8u); }
        let vm50 = sg_w4b[q5 * gpr + g0];
        var cvm51 = sg_byte_b(cm5 + cbo1);
        if (sh1 > 3u) { cvm51 = cvm51 | (sg_byte_b(cm5 + cbo1 + 1u) << 8u); }
        let vm51 = sg_w4b[q5 * gpr + g1];
        var cvm60 = sg_byte_b(cm6 + cbo0);
        if (sh0 > 3u) { cvm60 = cvm60 | (sg_byte_b(cm6 + cbo0 + 1u) << 8u); }
        let vm60 = sg_w4b[q6 * gpr + g0];
        var cvm61 = sg_byte_b(cm6 + cbo1);
        if (sh1 > 3u) { cvm61 = cvm61 | (sg_byte_b(cm6 + cbo1 + 1u) << 8u); }
        let vm61 = sg_w4b[q6 * gpr + g1];
        var cvm70 = sg_byte_b(cm7 + cbo0);
        if (sh0 > 3u) { cvm70 = cvm70 | (sg_byte_b(cm7 + cbo0 + 1u) << 8u); }
        let vm70 = sg_w4b[q7 * gpr + g0];
        var cvm71 = sg_byte_b(cm7 + cbo1);
        if (sh1 > 3u) { cvm71 = cvm71 | (sg_byte_b(cm7 + cbo1 + 1u) << 8u); }
        let vm71 = sg_w4b[q7 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        ma4 = ma4 + exp2(pm4.x + f32((cvm40 >> sh0) & 31u) * pm4.y)
            * (sg_dot8(vm40.x, xa0, xb0) + sg_dot8(vm40.y, xc0, xd0)
             + sg_dot8(vm40.z, xe0, xf0) + sg_dot8(vm40.w, xg0, xh0));
        ma5 = ma5 + exp2(pm5.x + f32((cvm50 >> sh0) & 31u) * pm5.y)
            * (sg_dot8(vm50.x, xa0, xb0) + sg_dot8(vm50.y, xc0, xd0)
             + sg_dot8(vm50.z, xe0, xf0) + sg_dot8(vm50.w, xg0, xh0));
        ma6 = ma6 + exp2(pm6.x + f32((cvm60 >> sh0) & 31u) * pm6.y)
            * (sg_dot8(vm60.x, xa0, xb0) + sg_dot8(vm60.y, xc0, xd0)
             + sg_dot8(vm60.z, xe0, xf0) + sg_dot8(vm60.w, xg0, xh0));
        ma7 = ma7 + exp2(pm7.x + f32((cvm70 >> sh0) & 31u) * pm7.y)
            * (sg_dot8(vm70.x, xa0, xb0) + sg_dot8(vm70.y, xc0, xd0)
             + sg_dot8(vm70.z, xe0, xf0) + sg_dot8(vm70.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
            mb4 = mb4 + exp2(pm4.x + f32((cvm41 >> sh1) & 31u) * pm4.y)
                * (sg_dot8(vm41.x, xa1, xb1) + sg_dot8(vm41.y, xc1, xd1)
                 + sg_dot8(vm41.z, xe1, xf1) + sg_dot8(vm41.w, xg1, xh1));
            mb5 = mb5 + exp2(pm5.x + f32((cvm51 >> sh1) & 31u) * pm5.y)
                * (sg_dot8(vm51.x, xa1, xb1) + sg_dot8(vm51.y, xc1, xd1)
                 + sg_dot8(vm51.z, xe1, xf1) + sg_dot8(vm51.w, xg1, xh1));
            mb6 = mb6 + exp2(pm6.x + f32((cvm61 >> sh1) & 31u) * pm6.y)
                * (sg_dot8(vm61.x, xa1, xb1) + sg_dot8(vm61.y, xc1, xd1)
                 + sg_dot8(vm61.z, xe1, xf1) + sg_dot8(vm61.w, xg1, xh1));
            mb7 = mb7 + exp2(pm7.x + f32((cvm71 >> sh1) & 31u) * pm7.y)
                * (sg_dot8(vm71.x, xa1, xb1) + sg_dot8(vm71.y, xc1, xd1)
                 + sg_dot8(vm71.z, xe1, xf1) + sg_dot8(vm71.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    let t4 = sg_tree(ma4 + mb4);
    let t5 = sg_tree(ma5 + mb5);
    let t6 = sg_tree(ma6 + mb6);
    let t7 = sg_tree(ma7 + mb7);
    if (lane == 0u) {
        if (l0) { sg_yb[r0] = t0; }
        if (l1) { sg_yb[r1] = t1; }
        if (l2) { sg_yb[r2] = t2; }
        if (l3) { sg_yb[r3] = t3; }
        if (l4) { sg_yb[r4] = t4; }
        if (l5) { sg_yb[r5] = t5; }
        if (l6) { sg_yb[r6] = t6; }
        if (l7) { sg_yb[r7] = t7; }
    }
}
fn run_c(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_c;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 8u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wc[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wc[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wc[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wc[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    let r4 = base + 4u;
    let l4 = r4 < rows;
    let q4 = select(base, r4, l4);
    let cm4 = codes_b + q4 * cstride;
    let pm4 = unpack2x16float(sg_wc[params_w + q4]);
    var ma4 = 0.0;
    var mb4 = 0.0;
    let r5 = base + 5u;
    let l5 = r5 < rows;
    let q5 = select(base, r5, l5);
    let cm5 = codes_b + q5 * cstride;
    let pm5 = unpack2x16float(sg_wc[params_w + q5]);
    var ma5 = 0.0;
    var mb5 = 0.0;
    let r6 = base + 6u;
    let l6 = r6 < rows;
    let q6 = select(base, r6, l6);
    let cm6 = codes_b + q6 * cstride;
    let pm6 = unpack2x16float(sg_wc[params_w + q6]);
    var ma6 = 0.0;
    var mb6 = 0.0;
    let r7 = base + 7u;
    let l7 = r7 < rows;
    let q7 = select(base, r7, l7);
    let cm7 = codes_b + q7 * cstride;
    let pm7 = unpack2x16float(sg_wc[params_w + q7]);
    var ma7 = 0.0;
    var mb7 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvm00 = sg_byte_c(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_c(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4c[q0 * gpr + g0];
        var cvm01 = sg_byte_c(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_c(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4c[q0 * gpr + g1];
        var cvm10 = sg_byte_c(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_c(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4c[q1 * gpr + g0];
        var cvm11 = sg_byte_c(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_c(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4c[q1 * gpr + g1];
        var cvm20 = sg_byte_c(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_c(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4c[q2 * gpr + g0];
        var cvm21 = sg_byte_c(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_c(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4c[q2 * gpr + g1];
        var cvm30 = sg_byte_c(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_c(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4c[q3 * gpr + g0];
        var cvm31 = sg_byte_c(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_c(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4c[q3 * gpr + g1];
        var cvm40 = sg_byte_c(cm4 + cbo0);
        if (sh0 > 3u) { cvm40 = cvm40 | (sg_byte_c(cm4 + cbo0 + 1u) << 8u); }
        let vm40 = sg_w4c[q4 * gpr + g0];
        var cvm41 = sg_byte_c(cm4 + cbo1);
        if (sh1 > 3u) { cvm41 = cvm41 | (sg_byte_c(cm4 + cbo1 + 1u) << 8u); }
        let vm41 = sg_w4c[q4 * gpr + g1];
        var cvm50 = sg_byte_c(cm5 + cbo0);
        if (sh0 > 3u) { cvm50 = cvm50 | (sg_byte_c(cm5 + cbo0 + 1u) << 8u); }
        let vm50 = sg_w4c[q5 * gpr + g0];
        var cvm51 = sg_byte_c(cm5 + cbo1);
        if (sh1 > 3u) { cvm51 = cvm51 | (sg_byte_c(cm5 + cbo1 + 1u) << 8u); }
        let vm51 = sg_w4c[q5 * gpr + g1];
        var cvm60 = sg_byte_c(cm6 + cbo0);
        if (sh0 > 3u) { cvm60 = cvm60 | (sg_byte_c(cm6 + cbo0 + 1u) << 8u); }
        let vm60 = sg_w4c[q6 * gpr + g0];
        var cvm61 = sg_byte_c(cm6 + cbo1);
        if (sh1 > 3u) { cvm61 = cvm61 | (sg_byte_c(cm6 + cbo1 + 1u) << 8u); }
        let vm61 = sg_w4c[q6 * gpr + g1];
        var cvm70 = sg_byte_c(cm7 + cbo0);
        if (sh0 > 3u) { cvm70 = cvm70 | (sg_byte_c(cm7 + cbo0 + 1u) << 8u); }
        let vm70 = sg_w4c[q7 * gpr + g0];
        var cvm71 = sg_byte_c(cm7 + cbo1);
        if (sh1 > 3u) { cvm71 = cvm71 | (sg_byte_c(cm7 + cbo1 + 1u) << 8u); }
        let vm71 = sg_w4c[q7 * gpr + g1];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        ma4 = ma4 + exp2(pm4.x + f32((cvm40 >> sh0) & 31u) * pm4.y)
            * (sg_dot8(vm40.x, xa0, xb0) + sg_dot8(vm40.y, xc0, xd0)
             + sg_dot8(vm40.z, xe0, xf0) + sg_dot8(vm40.w, xg0, xh0));
        ma5 = ma5 + exp2(pm5.x + f32((cvm50 >> sh0) & 31u) * pm5.y)
            * (sg_dot8(vm50.x, xa0, xb0) + sg_dot8(vm50.y, xc0, xd0)
             + sg_dot8(vm50.z, xe0, xf0) + sg_dot8(vm50.w, xg0, xh0));
        ma6 = ma6 + exp2(pm6.x + f32((cvm60 >> sh0) & 31u) * pm6.y)
            * (sg_dot8(vm60.x, xa0, xb0) + sg_dot8(vm60.y, xc0, xd0)
             + sg_dot8(vm60.z, xe0, xf0) + sg_dot8(vm60.w, xg0, xh0));
        ma7 = ma7 + exp2(pm7.x + f32((cvm70 >> sh0) & 31u) * pm7.y)
            * (sg_dot8(vm70.x, xa0, xb0) + sg_dot8(vm70.y, xc0, xd0)
             + sg_dot8(vm70.z, xe0, xf0) + sg_dot8(vm70.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
            mb4 = mb4 + exp2(pm4.x + f32((cvm41 >> sh1) & 31u) * pm4.y)
                * (sg_dot8(vm41.x, xa1, xb1) + sg_dot8(vm41.y, xc1, xd1)
                 + sg_dot8(vm41.z, xe1, xf1) + sg_dot8(vm41.w, xg1, xh1));
            mb5 = mb5 + exp2(pm5.x + f32((cvm51 >> sh1) & 31u) * pm5.y)
                * (sg_dot8(vm51.x, xa1, xb1) + sg_dot8(vm51.y, xc1, xd1)
                 + sg_dot8(vm51.z, xe1, xf1) + sg_dot8(vm51.w, xg1, xh1));
            mb6 = mb6 + exp2(pm6.x + f32((cvm61 >> sh1) & 31u) * pm6.y)
                * (sg_dot8(vm61.x, xa1, xb1) + sg_dot8(vm61.y, xc1, xd1)
                 + sg_dot8(vm61.z, xe1, xf1) + sg_dot8(vm61.w, xg1, xh1));
            mb7 = mb7 + exp2(pm7.x + f32((cvm71 >> sh1) & 31u) * pm7.y)
                * (sg_dot8(vm71.x, xa1, xb1) + sg_dot8(vm71.y, xc1, xd1)
                 + sg_dot8(vm71.z, xe1, xf1) + sg_dot8(vm71.w, xg1, xh1));
        }
        g = g + 64u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    let t4 = sg_tree(ma4 + mb4);
    let t5 = sg_tree(ma5 + mb5);
    let t6 = sg_tree(ma6 + mb6);
    let t7 = sg_tree(ma7 + mb7);
    if (lane == 0u) {
        if (l0) { sg_yc[r0] = t0; }
        if (l1) { sg_yc[r1] = t1; }
        if (l2) { sg_yc[r2] = t2; }
        if (l3) { sg_yc[r3] = t3; }
        if (l4) { sg_yc[r4] = t4; }
        if (l5) { sg_yc[r5] = t5; }
        if (l6) { sg_yc[r6] = t6; }
        if (l7) { sg_yc[r7] = t7; }
    }
}

@compute @workgroup_size(256)
fn q4tp_mv_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    let per = 8u * 8u;
    let warp = lid >> 5u;
    let ba = (sg_p.rows_a + per - 1u) / per;
    let bb = (sg_p.rows_b + per - 1u) / per;
    let wb = wid.x;
    if (wb < ba) {
        run_a(wb, warp, lane);
    } else if (wb < ba + bb) {
        run_b(wb - ba, warp, lane);
    } else {
        run_c(wb - ba - bb, warp, lane);
    }
}

"#;

pub(crate) const MV_SG_R2U4: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_a(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 2u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wa[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wa[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        let g2 = min(g + 64u, gpr - 1u);
        let bit2 = g2 * 5u;
        let cbo2 = bit2 >> 3u;
        let sh2 = bit2 & 7u;
        let xo2 = g2 * 8u;
        let xa2 = sg_x[xo2 + 0u];
        let xb2 = sg_x[xo2 + 1u];
        let xc2 = sg_x[xo2 + 2u];
        let xd2 = sg_x[xo2 + 3u];
        let xe2 = sg_x[xo2 + 4u];
        let xf2 = sg_x[xo2 + 5u];
        let xg2 = sg_x[xo2 + 6u];
        let xh2 = sg_x[xo2 + 7u];
        let g3 = min(g + 96u, gpr - 1u);
        let bit3 = g3 * 5u;
        let cbo3 = bit3 >> 3u;
        let sh3 = bit3 & 7u;
        let xo3 = g3 * 8u;
        let xa3 = sg_x[xo3 + 0u];
        let xb3 = sg_x[xo3 + 1u];
        let xc3 = sg_x[xo3 + 2u];
        let xd3 = sg_x[xo3 + 3u];
        let xe3 = sg_x[xo3 + 4u];
        let xf3 = sg_x[xo3 + 5u];
        let xg3 = sg_x[xo3 + 6u];
        let xh3 = sg_x[xo3 + 7u];
        var cvm00 = sg_byte_a(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_a(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4a[q0 * gpr + g0];
        var cvm01 = sg_byte_a(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_a(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4a[q0 * gpr + g1];
        var cvm02 = sg_byte_a(cm0 + cbo2);
        if (sh2 > 3u) { cvm02 = cvm02 | (sg_byte_a(cm0 + cbo2 + 1u) << 8u); }
        let vm02 = sg_w4a[q0 * gpr + g2];
        var cvm03 = sg_byte_a(cm0 + cbo3);
        if (sh3 > 3u) { cvm03 = cvm03 | (sg_byte_a(cm0 + cbo3 + 1u) << 8u); }
        let vm03 = sg_w4a[q0 * gpr + g3];
        var cvm10 = sg_byte_a(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_a(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4a[q1 * gpr + g0];
        var cvm11 = sg_byte_a(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_a(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4a[q1 * gpr + g1];
        var cvm12 = sg_byte_a(cm1 + cbo2);
        if (sh2 > 3u) { cvm12 = cvm12 | (sg_byte_a(cm1 + cbo2 + 1u) << 8u); }
        let vm12 = sg_w4a[q1 * gpr + g2];
        var cvm13 = sg_byte_a(cm1 + cbo3);
        if (sh3 > 3u) { cvm13 = cvm13 | (sg_byte_a(cm1 + cbo3 + 1u) << 8u); }
        let vm13 = sg_w4a[q1 * gpr + g3];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
        }
        if (g + 64u < gpr) {
            ma0 = ma0 + exp2(pm0.x + f32((cvm02 >> sh2) & 31u) * pm0.y)
                * (sg_dot8(vm02.x, xa2, xb2) + sg_dot8(vm02.y, xc2, xd2)
                 + sg_dot8(vm02.z, xe2, xf2) + sg_dot8(vm02.w, xg2, xh2));
            ma1 = ma1 + exp2(pm1.x + f32((cvm12 >> sh2) & 31u) * pm1.y)
                * (sg_dot8(vm12.x, xa2, xb2) + sg_dot8(vm12.y, xc2, xd2)
                 + sg_dot8(vm12.z, xe2, xf2) + sg_dot8(vm12.w, xg2, xh2));
        }
        if (g + 96u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm03 >> sh3) & 31u) * pm0.y)
                * (sg_dot8(vm03.x, xa3, xb3) + sg_dot8(vm03.y, xc3, xd3)
                 + sg_dot8(vm03.z, xe3, xf3) + sg_dot8(vm03.w, xg3, xh3));
            mb1 = mb1 + exp2(pm1.x + f32((cvm13 >> sh3) & 31u) * pm1.y)
                * (sg_dot8(vm13.x, xa3, xb3) + sg_dot8(vm13.y, xc3, xd3)
                 + sg_dot8(vm13.z, xe3, xf3) + sg_dot8(vm13.w, xg3, xh3));
        }
        g = g + 128u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    if (lane == 0u) {
        if (l0) { sg_ya[r0] = t0; }
        if (l1) { sg_ya[r1] = t1; }
    }
}
fn run_b(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_b;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 2u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wb[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wb[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        let g2 = min(g + 64u, gpr - 1u);
        let bit2 = g2 * 5u;
        let cbo2 = bit2 >> 3u;
        let sh2 = bit2 & 7u;
        let xo2 = g2 * 8u;
        let xa2 = sg_x[xo2 + 0u];
        let xb2 = sg_x[xo2 + 1u];
        let xc2 = sg_x[xo2 + 2u];
        let xd2 = sg_x[xo2 + 3u];
        let xe2 = sg_x[xo2 + 4u];
        let xf2 = sg_x[xo2 + 5u];
        let xg2 = sg_x[xo2 + 6u];
        let xh2 = sg_x[xo2 + 7u];
        let g3 = min(g + 96u, gpr - 1u);
        let bit3 = g3 * 5u;
        let cbo3 = bit3 >> 3u;
        let sh3 = bit3 & 7u;
        let xo3 = g3 * 8u;
        let xa3 = sg_x[xo3 + 0u];
        let xb3 = sg_x[xo3 + 1u];
        let xc3 = sg_x[xo3 + 2u];
        let xd3 = sg_x[xo3 + 3u];
        let xe3 = sg_x[xo3 + 4u];
        let xf3 = sg_x[xo3 + 5u];
        let xg3 = sg_x[xo3 + 6u];
        let xh3 = sg_x[xo3 + 7u];
        var cvm00 = sg_byte_b(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_b(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4b[q0 * gpr + g0];
        var cvm01 = sg_byte_b(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_b(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4b[q0 * gpr + g1];
        var cvm02 = sg_byte_b(cm0 + cbo2);
        if (sh2 > 3u) { cvm02 = cvm02 | (sg_byte_b(cm0 + cbo2 + 1u) << 8u); }
        let vm02 = sg_w4b[q0 * gpr + g2];
        var cvm03 = sg_byte_b(cm0 + cbo3);
        if (sh3 > 3u) { cvm03 = cvm03 | (sg_byte_b(cm0 + cbo3 + 1u) << 8u); }
        let vm03 = sg_w4b[q0 * gpr + g3];
        var cvm10 = sg_byte_b(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_b(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4b[q1 * gpr + g0];
        var cvm11 = sg_byte_b(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_b(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4b[q1 * gpr + g1];
        var cvm12 = sg_byte_b(cm1 + cbo2);
        if (sh2 > 3u) { cvm12 = cvm12 | (sg_byte_b(cm1 + cbo2 + 1u) << 8u); }
        let vm12 = sg_w4b[q1 * gpr + g2];
        var cvm13 = sg_byte_b(cm1 + cbo3);
        if (sh3 > 3u) { cvm13 = cvm13 | (sg_byte_b(cm1 + cbo3 + 1u) << 8u); }
        let vm13 = sg_w4b[q1 * gpr + g3];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
        }
        if (g + 64u < gpr) {
            ma0 = ma0 + exp2(pm0.x + f32((cvm02 >> sh2) & 31u) * pm0.y)
                * (sg_dot8(vm02.x, xa2, xb2) + sg_dot8(vm02.y, xc2, xd2)
                 + sg_dot8(vm02.z, xe2, xf2) + sg_dot8(vm02.w, xg2, xh2));
            ma1 = ma1 + exp2(pm1.x + f32((cvm12 >> sh2) & 31u) * pm1.y)
                * (sg_dot8(vm12.x, xa2, xb2) + sg_dot8(vm12.y, xc2, xd2)
                 + sg_dot8(vm12.z, xe2, xf2) + sg_dot8(vm12.w, xg2, xh2));
        }
        if (g + 96u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm03 >> sh3) & 31u) * pm0.y)
                * (sg_dot8(vm03.x, xa3, xb3) + sg_dot8(vm03.y, xc3, xd3)
                 + sg_dot8(vm03.z, xe3, xf3) + sg_dot8(vm03.w, xg3, xh3));
            mb1 = mb1 + exp2(pm1.x + f32((cvm13 >> sh3) & 31u) * pm1.y)
                * (sg_dot8(vm13.x, xa3, xb3) + sg_dot8(vm13.y, xc3, xd3)
                 + sg_dot8(vm13.z, xe3, xf3) + sg_dot8(vm13.w, xg3, xh3));
        }
        g = g + 128u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    if (lane == 0u) {
        if (l0) { sg_yb[r0] = t0; }
        if (l1) { sg_yb[r1] = t1; }
    }
}
fn run_c(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_c;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 2u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wc[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wc[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        let g2 = min(g + 64u, gpr - 1u);
        let bit2 = g2 * 5u;
        let cbo2 = bit2 >> 3u;
        let sh2 = bit2 & 7u;
        let xo2 = g2 * 8u;
        let xa2 = sg_x[xo2 + 0u];
        let xb2 = sg_x[xo2 + 1u];
        let xc2 = sg_x[xo2 + 2u];
        let xd2 = sg_x[xo2 + 3u];
        let xe2 = sg_x[xo2 + 4u];
        let xf2 = sg_x[xo2 + 5u];
        let xg2 = sg_x[xo2 + 6u];
        let xh2 = sg_x[xo2 + 7u];
        let g3 = min(g + 96u, gpr - 1u);
        let bit3 = g3 * 5u;
        let cbo3 = bit3 >> 3u;
        let sh3 = bit3 & 7u;
        let xo3 = g3 * 8u;
        let xa3 = sg_x[xo3 + 0u];
        let xb3 = sg_x[xo3 + 1u];
        let xc3 = sg_x[xo3 + 2u];
        let xd3 = sg_x[xo3 + 3u];
        let xe3 = sg_x[xo3 + 4u];
        let xf3 = sg_x[xo3 + 5u];
        let xg3 = sg_x[xo3 + 6u];
        let xh3 = sg_x[xo3 + 7u];
        var cvm00 = sg_byte_c(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_c(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4c[q0 * gpr + g0];
        var cvm01 = sg_byte_c(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_c(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4c[q0 * gpr + g1];
        var cvm02 = sg_byte_c(cm0 + cbo2);
        if (sh2 > 3u) { cvm02 = cvm02 | (sg_byte_c(cm0 + cbo2 + 1u) << 8u); }
        let vm02 = sg_w4c[q0 * gpr + g2];
        var cvm03 = sg_byte_c(cm0 + cbo3);
        if (sh3 > 3u) { cvm03 = cvm03 | (sg_byte_c(cm0 + cbo3 + 1u) << 8u); }
        let vm03 = sg_w4c[q0 * gpr + g3];
        var cvm10 = sg_byte_c(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_c(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4c[q1 * gpr + g0];
        var cvm11 = sg_byte_c(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_c(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4c[q1 * gpr + g1];
        var cvm12 = sg_byte_c(cm1 + cbo2);
        if (sh2 > 3u) { cvm12 = cvm12 | (sg_byte_c(cm1 + cbo2 + 1u) << 8u); }
        let vm12 = sg_w4c[q1 * gpr + g2];
        var cvm13 = sg_byte_c(cm1 + cbo3);
        if (sh3 > 3u) { cvm13 = cvm13 | (sg_byte_c(cm1 + cbo3 + 1u) << 8u); }
        let vm13 = sg_w4c[q1 * gpr + g3];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
        }
        if (g + 64u < gpr) {
            ma0 = ma0 + exp2(pm0.x + f32((cvm02 >> sh2) & 31u) * pm0.y)
                * (sg_dot8(vm02.x, xa2, xb2) + sg_dot8(vm02.y, xc2, xd2)
                 + sg_dot8(vm02.z, xe2, xf2) + sg_dot8(vm02.w, xg2, xh2));
            ma1 = ma1 + exp2(pm1.x + f32((cvm12 >> sh2) & 31u) * pm1.y)
                * (sg_dot8(vm12.x, xa2, xb2) + sg_dot8(vm12.y, xc2, xd2)
                 + sg_dot8(vm12.z, xe2, xf2) + sg_dot8(vm12.w, xg2, xh2));
        }
        if (g + 96u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm03 >> sh3) & 31u) * pm0.y)
                * (sg_dot8(vm03.x, xa3, xb3) + sg_dot8(vm03.y, xc3, xd3)
                 + sg_dot8(vm03.z, xe3, xf3) + sg_dot8(vm03.w, xg3, xh3));
            mb1 = mb1 + exp2(pm1.x + f32((cvm13 >> sh3) & 31u) * pm1.y)
                * (sg_dot8(vm13.x, xa3, xb3) + sg_dot8(vm13.y, xc3, xd3)
                 + sg_dot8(vm13.z, xe3, xf3) + sg_dot8(vm13.w, xg3, xh3));
        }
        g = g + 128u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    if (lane == 0u) {
        if (l0) { sg_yc[r0] = t0; }
        if (l1) { sg_yc[r1] = t1; }
    }
}

@compute @workgroup_size(256)
fn q4tp_mv_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    let per = 8u * 2u;
    let warp = lid >> 5u;
    let ba = (sg_p.rows_a + per - 1u) / per;
    let bb = (sg_p.rows_b + per - 1u) / per;
    let wb = wid.x;
    if (wb < ba) {
        run_a(wb, warp, lane);
    } else if (wb < ba + bb) {
        run_b(wb - ba, warp, lane);
    } else {
        run_c(wb - ba - bb, warp, lane);
    }
}

"#;

pub(crate) const MV_SG_R4U4: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_a(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 4u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wa[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wa[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wa[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wa[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        let g2 = min(g + 64u, gpr - 1u);
        let bit2 = g2 * 5u;
        let cbo2 = bit2 >> 3u;
        let sh2 = bit2 & 7u;
        let xo2 = g2 * 8u;
        let xa2 = sg_x[xo2 + 0u];
        let xb2 = sg_x[xo2 + 1u];
        let xc2 = sg_x[xo2 + 2u];
        let xd2 = sg_x[xo2 + 3u];
        let xe2 = sg_x[xo2 + 4u];
        let xf2 = sg_x[xo2 + 5u];
        let xg2 = sg_x[xo2 + 6u];
        let xh2 = sg_x[xo2 + 7u];
        let g3 = min(g + 96u, gpr - 1u);
        let bit3 = g3 * 5u;
        let cbo3 = bit3 >> 3u;
        let sh3 = bit3 & 7u;
        let xo3 = g3 * 8u;
        let xa3 = sg_x[xo3 + 0u];
        let xb3 = sg_x[xo3 + 1u];
        let xc3 = sg_x[xo3 + 2u];
        let xd3 = sg_x[xo3 + 3u];
        let xe3 = sg_x[xo3 + 4u];
        let xf3 = sg_x[xo3 + 5u];
        let xg3 = sg_x[xo3 + 6u];
        let xh3 = sg_x[xo3 + 7u];
        var cvm00 = sg_byte_a(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_a(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4a[q0 * gpr + g0];
        var cvm01 = sg_byte_a(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_a(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4a[q0 * gpr + g1];
        var cvm02 = sg_byte_a(cm0 + cbo2);
        if (sh2 > 3u) { cvm02 = cvm02 | (sg_byte_a(cm0 + cbo2 + 1u) << 8u); }
        let vm02 = sg_w4a[q0 * gpr + g2];
        var cvm03 = sg_byte_a(cm0 + cbo3);
        if (sh3 > 3u) { cvm03 = cvm03 | (sg_byte_a(cm0 + cbo3 + 1u) << 8u); }
        let vm03 = sg_w4a[q0 * gpr + g3];
        var cvm10 = sg_byte_a(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_a(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4a[q1 * gpr + g0];
        var cvm11 = sg_byte_a(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_a(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4a[q1 * gpr + g1];
        var cvm12 = sg_byte_a(cm1 + cbo2);
        if (sh2 > 3u) { cvm12 = cvm12 | (sg_byte_a(cm1 + cbo2 + 1u) << 8u); }
        let vm12 = sg_w4a[q1 * gpr + g2];
        var cvm13 = sg_byte_a(cm1 + cbo3);
        if (sh3 > 3u) { cvm13 = cvm13 | (sg_byte_a(cm1 + cbo3 + 1u) << 8u); }
        let vm13 = sg_w4a[q1 * gpr + g3];
        var cvm20 = sg_byte_a(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_a(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4a[q2 * gpr + g0];
        var cvm21 = sg_byte_a(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_a(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4a[q2 * gpr + g1];
        var cvm22 = sg_byte_a(cm2 + cbo2);
        if (sh2 > 3u) { cvm22 = cvm22 | (sg_byte_a(cm2 + cbo2 + 1u) << 8u); }
        let vm22 = sg_w4a[q2 * gpr + g2];
        var cvm23 = sg_byte_a(cm2 + cbo3);
        if (sh3 > 3u) { cvm23 = cvm23 | (sg_byte_a(cm2 + cbo3 + 1u) << 8u); }
        let vm23 = sg_w4a[q2 * gpr + g3];
        var cvm30 = sg_byte_a(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_a(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4a[q3 * gpr + g0];
        var cvm31 = sg_byte_a(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_a(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4a[q3 * gpr + g1];
        var cvm32 = sg_byte_a(cm3 + cbo2);
        if (sh2 > 3u) { cvm32 = cvm32 | (sg_byte_a(cm3 + cbo2 + 1u) << 8u); }
        let vm32 = sg_w4a[q3 * gpr + g2];
        var cvm33 = sg_byte_a(cm3 + cbo3);
        if (sh3 > 3u) { cvm33 = cvm33 | (sg_byte_a(cm3 + cbo3 + 1u) << 8u); }
        let vm33 = sg_w4a[q3 * gpr + g3];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
        }
        if (g + 64u < gpr) {
            ma0 = ma0 + exp2(pm0.x + f32((cvm02 >> sh2) & 31u) * pm0.y)
                * (sg_dot8(vm02.x, xa2, xb2) + sg_dot8(vm02.y, xc2, xd2)
                 + sg_dot8(vm02.z, xe2, xf2) + sg_dot8(vm02.w, xg2, xh2));
            ma1 = ma1 + exp2(pm1.x + f32((cvm12 >> sh2) & 31u) * pm1.y)
                * (sg_dot8(vm12.x, xa2, xb2) + sg_dot8(vm12.y, xc2, xd2)
                 + sg_dot8(vm12.z, xe2, xf2) + sg_dot8(vm12.w, xg2, xh2));
            ma2 = ma2 + exp2(pm2.x + f32((cvm22 >> sh2) & 31u) * pm2.y)
                * (sg_dot8(vm22.x, xa2, xb2) + sg_dot8(vm22.y, xc2, xd2)
                 + sg_dot8(vm22.z, xe2, xf2) + sg_dot8(vm22.w, xg2, xh2));
            ma3 = ma3 + exp2(pm3.x + f32((cvm32 >> sh2) & 31u) * pm3.y)
                * (sg_dot8(vm32.x, xa2, xb2) + sg_dot8(vm32.y, xc2, xd2)
                 + sg_dot8(vm32.z, xe2, xf2) + sg_dot8(vm32.w, xg2, xh2));
        }
        if (g + 96u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm03 >> sh3) & 31u) * pm0.y)
                * (sg_dot8(vm03.x, xa3, xb3) + sg_dot8(vm03.y, xc3, xd3)
                 + sg_dot8(vm03.z, xe3, xf3) + sg_dot8(vm03.w, xg3, xh3));
            mb1 = mb1 + exp2(pm1.x + f32((cvm13 >> sh3) & 31u) * pm1.y)
                * (sg_dot8(vm13.x, xa3, xb3) + sg_dot8(vm13.y, xc3, xd3)
                 + sg_dot8(vm13.z, xe3, xf3) + sg_dot8(vm13.w, xg3, xh3));
            mb2 = mb2 + exp2(pm2.x + f32((cvm23 >> sh3) & 31u) * pm2.y)
                * (sg_dot8(vm23.x, xa3, xb3) + sg_dot8(vm23.y, xc3, xd3)
                 + sg_dot8(vm23.z, xe3, xf3) + sg_dot8(vm23.w, xg3, xh3));
            mb3 = mb3 + exp2(pm3.x + f32((cvm33 >> sh3) & 31u) * pm3.y)
                * (sg_dot8(vm33.x, xa3, xb3) + sg_dot8(vm33.y, xc3, xd3)
                 + sg_dot8(vm33.z, xe3, xf3) + sg_dot8(vm33.w, xg3, xh3));
        }
        g = g + 128u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    if (lane == 0u) {
        if (l0) { sg_ya[r0] = t0; }
        if (l1) { sg_ya[r1] = t1; }
        if (l2) { sg_ya[r2] = t2; }
        if (l3) { sg_ya[r3] = t3; }
    }
}
fn run_b(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_b;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 4u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wb[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wb[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wb[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wb[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        let g2 = min(g + 64u, gpr - 1u);
        let bit2 = g2 * 5u;
        let cbo2 = bit2 >> 3u;
        let sh2 = bit2 & 7u;
        let xo2 = g2 * 8u;
        let xa2 = sg_x[xo2 + 0u];
        let xb2 = sg_x[xo2 + 1u];
        let xc2 = sg_x[xo2 + 2u];
        let xd2 = sg_x[xo2 + 3u];
        let xe2 = sg_x[xo2 + 4u];
        let xf2 = sg_x[xo2 + 5u];
        let xg2 = sg_x[xo2 + 6u];
        let xh2 = sg_x[xo2 + 7u];
        let g3 = min(g + 96u, gpr - 1u);
        let bit3 = g3 * 5u;
        let cbo3 = bit3 >> 3u;
        let sh3 = bit3 & 7u;
        let xo3 = g3 * 8u;
        let xa3 = sg_x[xo3 + 0u];
        let xb3 = sg_x[xo3 + 1u];
        let xc3 = sg_x[xo3 + 2u];
        let xd3 = sg_x[xo3 + 3u];
        let xe3 = sg_x[xo3 + 4u];
        let xf3 = sg_x[xo3 + 5u];
        let xg3 = sg_x[xo3 + 6u];
        let xh3 = sg_x[xo3 + 7u];
        var cvm00 = sg_byte_b(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_b(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4b[q0 * gpr + g0];
        var cvm01 = sg_byte_b(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_b(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4b[q0 * gpr + g1];
        var cvm02 = sg_byte_b(cm0 + cbo2);
        if (sh2 > 3u) { cvm02 = cvm02 | (sg_byte_b(cm0 + cbo2 + 1u) << 8u); }
        let vm02 = sg_w4b[q0 * gpr + g2];
        var cvm03 = sg_byte_b(cm0 + cbo3);
        if (sh3 > 3u) { cvm03 = cvm03 | (sg_byte_b(cm0 + cbo3 + 1u) << 8u); }
        let vm03 = sg_w4b[q0 * gpr + g3];
        var cvm10 = sg_byte_b(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_b(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4b[q1 * gpr + g0];
        var cvm11 = sg_byte_b(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_b(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4b[q1 * gpr + g1];
        var cvm12 = sg_byte_b(cm1 + cbo2);
        if (sh2 > 3u) { cvm12 = cvm12 | (sg_byte_b(cm1 + cbo2 + 1u) << 8u); }
        let vm12 = sg_w4b[q1 * gpr + g2];
        var cvm13 = sg_byte_b(cm1 + cbo3);
        if (sh3 > 3u) { cvm13 = cvm13 | (sg_byte_b(cm1 + cbo3 + 1u) << 8u); }
        let vm13 = sg_w4b[q1 * gpr + g3];
        var cvm20 = sg_byte_b(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_b(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4b[q2 * gpr + g0];
        var cvm21 = sg_byte_b(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_b(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4b[q2 * gpr + g1];
        var cvm22 = sg_byte_b(cm2 + cbo2);
        if (sh2 > 3u) { cvm22 = cvm22 | (sg_byte_b(cm2 + cbo2 + 1u) << 8u); }
        let vm22 = sg_w4b[q2 * gpr + g2];
        var cvm23 = sg_byte_b(cm2 + cbo3);
        if (sh3 > 3u) { cvm23 = cvm23 | (sg_byte_b(cm2 + cbo3 + 1u) << 8u); }
        let vm23 = sg_w4b[q2 * gpr + g3];
        var cvm30 = sg_byte_b(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_b(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4b[q3 * gpr + g0];
        var cvm31 = sg_byte_b(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_b(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4b[q3 * gpr + g1];
        var cvm32 = sg_byte_b(cm3 + cbo2);
        if (sh2 > 3u) { cvm32 = cvm32 | (sg_byte_b(cm3 + cbo2 + 1u) << 8u); }
        let vm32 = sg_w4b[q3 * gpr + g2];
        var cvm33 = sg_byte_b(cm3 + cbo3);
        if (sh3 > 3u) { cvm33 = cvm33 | (sg_byte_b(cm3 + cbo3 + 1u) << 8u); }
        let vm33 = sg_w4b[q3 * gpr + g3];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
        }
        if (g + 64u < gpr) {
            ma0 = ma0 + exp2(pm0.x + f32((cvm02 >> sh2) & 31u) * pm0.y)
                * (sg_dot8(vm02.x, xa2, xb2) + sg_dot8(vm02.y, xc2, xd2)
                 + sg_dot8(vm02.z, xe2, xf2) + sg_dot8(vm02.w, xg2, xh2));
            ma1 = ma1 + exp2(pm1.x + f32((cvm12 >> sh2) & 31u) * pm1.y)
                * (sg_dot8(vm12.x, xa2, xb2) + sg_dot8(vm12.y, xc2, xd2)
                 + sg_dot8(vm12.z, xe2, xf2) + sg_dot8(vm12.w, xg2, xh2));
            ma2 = ma2 + exp2(pm2.x + f32((cvm22 >> sh2) & 31u) * pm2.y)
                * (sg_dot8(vm22.x, xa2, xb2) + sg_dot8(vm22.y, xc2, xd2)
                 + sg_dot8(vm22.z, xe2, xf2) + sg_dot8(vm22.w, xg2, xh2));
            ma3 = ma3 + exp2(pm3.x + f32((cvm32 >> sh2) & 31u) * pm3.y)
                * (sg_dot8(vm32.x, xa2, xb2) + sg_dot8(vm32.y, xc2, xd2)
                 + sg_dot8(vm32.z, xe2, xf2) + sg_dot8(vm32.w, xg2, xh2));
        }
        if (g + 96u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm03 >> sh3) & 31u) * pm0.y)
                * (sg_dot8(vm03.x, xa3, xb3) + sg_dot8(vm03.y, xc3, xd3)
                 + sg_dot8(vm03.z, xe3, xf3) + sg_dot8(vm03.w, xg3, xh3));
            mb1 = mb1 + exp2(pm1.x + f32((cvm13 >> sh3) & 31u) * pm1.y)
                * (sg_dot8(vm13.x, xa3, xb3) + sg_dot8(vm13.y, xc3, xd3)
                 + sg_dot8(vm13.z, xe3, xf3) + sg_dot8(vm13.w, xg3, xh3));
            mb2 = mb2 + exp2(pm2.x + f32((cvm23 >> sh3) & 31u) * pm2.y)
                * (sg_dot8(vm23.x, xa3, xb3) + sg_dot8(vm23.y, xc3, xd3)
                 + sg_dot8(vm23.z, xe3, xf3) + sg_dot8(vm23.w, xg3, xh3));
            mb3 = mb3 + exp2(pm3.x + f32((cvm33 >> sh3) & 31u) * pm3.y)
                * (sg_dot8(vm33.x, xa3, xb3) + sg_dot8(vm33.y, xc3, xd3)
                 + sg_dot8(vm33.z, xe3, xf3) + sg_dot8(vm33.w, xg3, xh3));
        }
        g = g + 128u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    if (lane == 0u) {
        if (l0) { sg_yb[r0] = t0; }
        if (l1) { sg_yb[r1] = t1; }
        if (l2) { sg_yb[r2] = t2; }
        if (l3) { sg_yb[r3] = t3; }
    }
}
fn run_c(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_c;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 4u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cm0 = codes_b + q0 * cstride;
    let pm0 = unpack2x16float(sg_wc[params_w + q0]);
    var ma0 = 0.0;
    var mb0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cm1 = codes_b + q1 * cstride;
    let pm1 = unpack2x16float(sg_wc[params_w + q1]);
    var ma1 = 0.0;
    var mb1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cm2 = codes_b + q2 * cstride;
    let pm2 = unpack2x16float(sg_wc[params_w + q2]);
    var ma2 = 0.0;
    var mb2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cm3 = codes_b + q3 * cstride;
    let pm3 = unpack2x16float(sg_wc[params_w + q3]);
    var ma3 = 0.0;
    var mb3 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        let g2 = min(g + 64u, gpr - 1u);
        let bit2 = g2 * 5u;
        let cbo2 = bit2 >> 3u;
        let sh2 = bit2 & 7u;
        let xo2 = g2 * 8u;
        let xa2 = sg_x[xo2 + 0u];
        let xb2 = sg_x[xo2 + 1u];
        let xc2 = sg_x[xo2 + 2u];
        let xd2 = sg_x[xo2 + 3u];
        let xe2 = sg_x[xo2 + 4u];
        let xf2 = sg_x[xo2 + 5u];
        let xg2 = sg_x[xo2 + 6u];
        let xh2 = sg_x[xo2 + 7u];
        let g3 = min(g + 96u, gpr - 1u);
        let bit3 = g3 * 5u;
        let cbo3 = bit3 >> 3u;
        let sh3 = bit3 & 7u;
        let xo3 = g3 * 8u;
        let xa3 = sg_x[xo3 + 0u];
        let xb3 = sg_x[xo3 + 1u];
        let xc3 = sg_x[xo3 + 2u];
        let xd3 = sg_x[xo3 + 3u];
        let xe3 = sg_x[xo3 + 4u];
        let xf3 = sg_x[xo3 + 5u];
        let xg3 = sg_x[xo3 + 6u];
        let xh3 = sg_x[xo3 + 7u];
        var cvm00 = sg_byte_c(cm0 + cbo0);
        if (sh0 > 3u) { cvm00 = cvm00 | (sg_byte_c(cm0 + cbo0 + 1u) << 8u); }
        let vm00 = sg_w4c[q0 * gpr + g0];
        var cvm01 = sg_byte_c(cm0 + cbo1);
        if (sh1 > 3u) { cvm01 = cvm01 | (sg_byte_c(cm0 + cbo1 + 1u) << 8u); }
        let vm01 = sg_w4c[q0 * gpr + g1];
        var cvm02 = sg_byte_c(cm0 + cbo2);
        if (sh2 > 3u) { cvm02 = cvm02 | (sg_byte_c(cm0 + cbo2 + 1u) << 8u); }
        let vm02 = sg_w4c[q0 * gpr + g2];
        var cvm03 = sg_byte_c(cm0 + cbo3);
        if (sh3 > 3u) { cvm03 = cvm03 | (sg_byte_c(cm0 + cbo3 + 1u) << 8u); }
        let vm03 = sg_w4c[q0 * gpr + g3];
        var cvm10 = sg_byte_c(cm1 + cbo0);
        if (sh0 > 3u) { cvm10 = cvm10 | (sg_byte_c(cm1 + cbo0 + 1u) << 8u); }
        let vm10 = sg_w4c[q1 * gpr + g0];
        var cvm11 = sg_byte_c(cm1 + cbo1);
        if (sh1 > 3u) { cvm11 = cvm11 | (sg_byte_c(cm1 + cbo1 + 1u) << 8u); }
        let vm11 = sg_w4c[q1 * gpr + g1];
        var cvm12 = sg_byte_c(cm1 + cbo2);
        if (sh2 > 3u) { cvm12 = cvm12 | (sg_byte_c(cm1 + cbo2 + 1u) << 8u); }
        let vm12 = sg_w4c[q1 * gpr + g2];
        var cvm13 = sg_byte_c(cm1 + cbo3);
        if (sh3 > 3u) { cvm13 = cvm13 | (sg_byte_c(cm1 + cbo3 + 1u) << 8u); }
        let vm13 = sg_w4c[q1 * gpr + g3];
        var cvm20 = sg_byte_c(cm2 + cbo0);
        if (sh0 > 3u) { cvm20 = cvm20 | (sg_byte_c(cm2 + cbo0 + 1u) << 8u); }
        let vm20 = sg_w4c[q2 * gpr + g0];
        var cvm21 = sg_byte_c(cm2 + cbo1);
        if (sh1 > 3u) { cvm21 = cvm21 | (sg_byte_c(cm2 + cbo1 + 1u) << 8u); }
        let vm21 = sg_w4c[q2 * gpr + g1];
        var cvm22 = sg_byte_c(cm2 + cbo2);
        if (sh2 > 3u) { cvm22 = cvm22 | (sg_byte_c(cm2 + cbo2 + 1u) << 8u); }
        let vm22 = sg_w4c[q2 * gpr + g2];
        var cvm23 = sg_byte_c(cm2 + cbo3);
        if (sh3 > 3u) { cvm23 = cvm23 | (sg_byte_c(cm2 + cbo3 + 1u) << 8u); }
        let vm23 = sg_w4c[q2 * gpr + g3];
        var cvm30 = sg_byte_c(cm3 + cbo0);
        if (sh0 > 3u) { cvm30 = cvm30 | (sg_byte_c(cm3 + cbo0 + 1u) << 8u); }
        let vm30 = sg_w4c[q3 * gpr + g0];
        var cvm31 = sg_byte_c(cm3 + cbo1);
        if (sh1 > 3u) { cvm31 = cvm31 | (sg_byte_c(cm3 + cbo1 + 1u) << 8u); }
        let vm31 = sg_w4c[q3 * gpr + g1];
        var cvm32 = sg_byte_c(cm3 + cbo2);
        if (sh2 > 3u) { cvm32 = cvm32 | (sg_byte_c(cm3 + cbo2 + 1u) << 8u); }
        let vm32 = sg_w4c[q3 * gpr + g2];
        var cvm33 = sg_byte_c(cm3 + cbo3);
        if (sh3 > 3u) { cvm33 = cvm33 | (sg_byte_c(cm3 + cbo3 + 1u) << 8u); }
        let vm33 = sg_w4c[q3 * gpr + g3];
        ma0 = ma0 + exp2(pm0.x + f32((cvm00 >> sh0) & 31u) * pm0.y)
            * (sg_dot8(vm00.x, xa0, xb0) + sg_dot8(vm00.y, xc0, xd0)
             + sg_dot8(vm00.z, xe0, xf0) + sg_dot8(vm00.w, xg0, xh0));
        ma1 = ma1 + exp2(pm1.x + f32((cvm10 >> sh0) & 31u) * pm1.y)
            * (sg_dot8(vm10.x, xa0, xb0) + sg_dot8(vm10.y, xc0, xd0)
             + sg_dot8(vm10.z, xe0, xf0) + sg_dot8(vm10.w, xg0, xh0));
        ma2 = ma2 + exp2(pm2.x + f32((cvm20 >> sh0) & 31u) * pm2.y)
            * (sg_dot8(vm20.x, xa0, xb0) + sg_dot8(vm20.y, xc0, xd0)
             + sg_dot8(vm20.z, xe0, xf0) + sg_dot8(vm20.w, xg0, xh0));
        ma3 = ma3 + exp2(pm3.x + f32((cvm30 >> sh0) & 31u) * pm3.y)
            * (sg_dot8(vm30.x, xa0, xb0) + sg_dot8(vm30.y, xc0, xd0)
             + sg_dot8(vm30.z, xe0, xf0) + sg_dot8(vm30.w, xg0, xh0));
        if (g + 32u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm01 >> sh1) & 31u) * pm0.y)
                * (sg_dot8(vm01.x, xa1, xb1) + sg_dot8(vm01.y, xc1, xd1)
                 + sg_dot8(vm01.z, xe1, xf1) + sg_dot8(vm01.w, xg1, xh1));
            mb1 = mb1 + exp2(pm1.x + f32((cvm11 >> sh1) & 31u) * pm1.y)
                * (sg_dot8(vm11.x, xa1, xb1) + sg_dot8(vm11.y, xc1, xd1)
                 + sg_dot8(vm11.z, xe1, xf1) + sg_dot8(vm11.w, xg1, xh1));
            mb2 = mb2 + exp2(pm2.x + f32((cvm21 >> sh1) & 31u) * pm2.y)
                * (sg_dot8(vm21.x, xa1, xb1) + sg_dot8(vm21.y, xc1, xd1)
                 + sg_dot8(vm21.z, xe1, xf1) + sg_dot8(vm21.w, xg1, xh1));
            mb3 = mb3 + exp2(pm3.x + f32((cvm31 >> sh1) & 31u) * pm3.y)
                * (sg_dot8(vm31.x, xa1, xb1) + sg_dot8(vm31.y, xc1, xd1)
                 + sg_dot8(vm31.z, xe1, xf1) + sg_dot8(vm31.w, xg1, xh1));
        }
        if (g + 64u < gpr) {
            ma0 = ma0 + exp2(pm0.x + f32((cvm02 >> sh2) & 31u) * pm0.y)
                * (sg_dot8(vm02.x, xa2, xb2) + sg_dot8(vm02.y, xc2, xd2)
                 + sg_dot8(vm02.z, xe2, xf2) + sg_dot8(vm02.w, xg2, xh2));
            ma1 = ma1 + exp2(pm1.x + f32((cvm12 >> sh2) & 31u) * pm1.y)
                * (sg_dot8(vm12.x, xa2, xb2) + sg_dot8(vm12.y, xc2, xd2)
                 + sg_dot8(vm12.z, xe2, xf2) + sg_dot8(vm12.w, xg2, xh2));
            ma2 = ma2 + exp2(pm2.x + f32((cvm22 >> sh2) & 31u) * pm2.y)
                * (sg_dot8(vm22.x, xa2, xb2) + sg_dot8(vm22.y, xc2, xd2)
                 + sg_dot8(vm22.z, xe2, xf2) + sg_dot8(vm22.w, xg2, xh2));
            ma3 = ma3 + exp2(pm3.x + f32((cvm32 >> sh2) & 31u) * pm3.y)
                * (sg_dot8(vm32.x, xa2, xb2) + sg_dot8(vm32.y, xc2, xd2)
                 + sg_dot8(vm32.z, xe2, xf2) + sg_dot8(vm32.w, xg2, xh2));
        }
        if (g + 96u < gpr) {
            mb0 = mb0 + exp2(pm0.x + f32((cvm03 >> sh3) & 31u) * pm0.y)
                * (sg_dot8(vm03.x, xa3, xb3) + sg_dot8(vm03.y, xc3, xd3)
                 + sg_dot8(vm03.z, xe3, xf3) + sg_dot8(vm03.w, xg3, xh3));
            mb1 = mb1 + exp2(pm1.x + f32((cvm13 >> sh3) & 31u) * pm1.y)
                * (sg_dot8(vm13.x, xa3, xb3) + sg_dot8(vm13.y, xc3, xd3)
                 + sg_dot8(vm13.z, xe3, xf3) + sg_dot8(vm13.w, xg3, xh3));
            mb2 = mb2 + exp2(pm2.x + f32((cvm23 >> sh3) & 31u) * pm2.y)
                * (sg_dot8(vm23.x, xa3, xb3) + sg_dot8(vm23.y, xc3, xd3)
                 + sg_dot8(vm23.z, xe3, xf3) + sg_dot8(vm23.w, xg3, xh3));
            mb3 = mb3 + exp2(pm3.x + f32((cvm33 >> sh3) & 31u) * pm3.y)
                * (sg_dot8(vm33.x, xa3, xb3) + sg_dot8(vm33.y, xc3, xd3)
                 + sg_dot8(vm33.z, xe3, xf3) + sg_dot8(vm33.w, xg3, xh3));
        }
        g = g + 128u;
    }
    let t0 = sg_tree(ma0 + mb0);
    let t1 = sg_tree(ma1 + mb1);
    let t2 = sg_tree(ma2 + mb2);
    let t3 = sg_tree(ma3 + mb3);
    if (lane == 0u) {
        if (l0) { sg_yc[r0] = t0; }
        if (l1) { sg_yc[r1] = t1; }
        if (l2) { sg_yc[r2] = t2; }
        if (l3) { sg_yc[r3] = t3; }
    }
}

@compute @workgroup_size(256)
fn q4tp_mv_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    let per = 8u * 4u;
    let warp = lid >> 5u;
    let ba = (sg_p.rows_a + per - 1u) / per;
    let bb = (sg_p.rows_b + per - 1u) / per;
    let wb = wid.x;
    if (wb < ba) {
        run_a(wb, warp, lane);
    } else if (wb < ba + bb) {
        run_b(wb - ba, warp, lane);
    } else {
        run_c(wb - ba - bb, warp, lane);
    }
}

"#;

pub(crate) const GU_SG_R1U2: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_gu(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 1u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cg0 = codes_b + q0 * cstride;
    let pg0 = unpack2x16float(sg_wa[params_w + q0]);
    var ga0 = 0.0;
    var gb0 = 0.0;
    let cu0 = codes_b + q0 * cstride;
    let pu0 = unpack2x16float(sg_wb[params_w + q0]);
    var ua0 = 0.0;
    var ub0 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvg00 = sg_byte_a(cg0 + cbo0);
        if (sh0 > 3u) { cvg00 = cvg00 | (sg_byte_a(cg0 + cbo0 + 1u) << 8u); }
        let vg00 = sg_w4a[q0 * gpr + g0];
        var cvg01 = sg_byte_a(cg0 + cbo1);
        if (sh1 > 3u) { cvg01 = cvg01 | (sg_byte_a(cg0 + cbo1 + 1u) << 8u); }
        let vg01 = sg_w4a[q0 * gpr + g1];
        var cvu00 = sg_byte_b(cu0 + cbo0);
        if (sh0 > 3u) { cvu00 = cvu00 | (sg_byte_b(cu0 + cbo0 + 1u) << 8u); }
        let vu00 = sg_w4b[q0 * gpr + g0];
        var cvu01 = sg_byte_b(cu0 + cbo1);
        if (sh1 > 3u) { cvu01 = cvu01 | (sg_byte_b(cu0 + cbo1 + 1u) << 8u); }
        let vu01 = sg_w4b[q0 * gpr + g1];
        ga0 = ga0 + exp2(pg0.x + f32((cvg00 >> sh0) & 31u) * pg0.y)
            * (sg_dot8(vg00.x, xa0, xb0) + sg_dot8(vg00.y, xc0, xd0)
             + sg_dot8(vg00.z, xe0, xf0) + sg_dot8(vg00.w, xg0, xh0));
        ua0 = ua0 + exp2(pu0.x + f32((cvu00 >> sh0) & 31u) * pu0.y)
            * (sg_dot8(vu00.x, xa0, xb0) + sg_dot8(vu00.y, xc0, xd0)
             + sg_dot8(vu00.z, xe0, xf0) + sg_dot8(vu00.w, xg0, xh0));
        if (g + 32u < gpr) {
            gb0 = gb0 + exp2(pg0.x + f32((cvg01 >> sh1) & 31u) * pg0.y)
                * (sg_dot8(vg01.x, xa1, xb1) + sg_dot8(vg01.y, xc1, xd1)
                 + sg_dot8(vg01.z, xe1, xf1) + sg_dot8(vg01.w, xg1, xh1));
            ub0 = ub0 + exp2(pu0.x + f32((cvu01 >> sh1) & 31u) * pu0.y)
                * (sg_dot8(vu01.x, xa1, xb1) + sg_dot8(vu01.y, xc1, xd1)
                 + sg_dot8(vu01.z, xe1, xf1) + sg_dot8(vu01.w, xg1, xh1));
        }
        g = g + 64u;
    }
    var gg0 = sg_tree(ga0 + gb0);
    var uu0 = sg_tree(ua0 + ub0);
    if (lane == 0u) {
        let lim = bitcast<f32>(sg_p.lim);
        if (l0) {
            if (lim > 0.0) { uu0 = clamp(uu0, -lim, lim); gg0 = min(gg0, lim); }
            if (sg_p.act == 1u) { sg_ya[r0] = sg_gelu_erf(gg0) * uu0; }
            else { sg_ya[r0] = (gg0 / (1.0 + exp(-gg0))) * uu0; }
        }
    }
}

@compute @workgroup_size(256)
fn q4tp_gu_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    run_gu(wid.x, lid >> 5u, lane);
}
"#;

pub(crate) const GU_SG_R2U2: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_gu(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 2u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cg0 = codes_b + q0 * cstride;
    let pg0 = unpack2x16float(sg_wa[params_w + q0]);
    var ga0 = 0.0;
    var gb0 = 0.0;
    let cu0 = codes_b + q0 * cstride;
    let pu0 = unpack2x16float(sg_wb[params_w + q0]);
    var ua0 = 0.0;
    var ub0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cg1 = codes_b + q1 * cstride;
    let pg1 = unpack2x16float(sg_wa[params_w + q1]);
    var ga1 = 0.0;
    var gb1 = 0.0;
    let cu1 = codes_b + q1 * cstride;
    let pu1 = unpack2x16float(sg_wb[params_w + q1]);
    var ua1 = 0.0;
    var ub1 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvg00 = sg_byte_a(cg0 + cbo0);
        if (sh0 > 3u) { cvg00 = cvg00 | (sg_byte_a(cg0 + cbo0 + 1u) << 8u); }
        let vg00 = sg_w4a[q0 * gpr + g0];
        var cvg01 = sg_byte_a(cg0 + cbo1);
        if (sh1 > 3u) { cvg01 = cvg01 | (sg_byte_a(cg0 + cbo1 + 1u) << 8u); }
        let vg01 = sg_w4a[q0 * gpr + g1];
        var cvg10 = sg_byte_a(cg1 + cbo0);
        if (sh0 > 3u) { cvg10 = cvg10 | (sg_byte_a(cg1 + cbo0 + 1u) << 8u); }
        let vg10 = sg_w4a[q1 * gpr + g0];
        var cvg11 = sg_byte_a(cg1 + cbo1);
        if (sh1 > 3u) { cvg11 = cvg11 | (sg_byte_a(cg1 + cbo1 + 1u) << 8u); }
        let vg11 = sg_w4a[q1 * gpr + g1];
        var cvu00 = sg_byte_b(cu0 + cbo0);
        if (sh0 > 3u) { cvu00 = cvu00 | (sg_byte_b(cu0 + cbo0 + 1u) << 8u); }
        let vu00 = sg_w4b[q0 * gpr + g0];
        var cvu01 = sg_byte_b(cu0 + cbo1);
        if (sh1 > 3u) { cvu01 = cvu01 | (sg_byte_b(cu0 + cbo1 + 1u) << 8u); }
        let vu01 = sg_w4b[q0 * gpr + g1];
        var cvu10 = sg_byte_b(cu1 + cbo0);
        if (sh0 > 3u) { cvu10 = cvu10 | (sg_byte_b(cu1 + cbo0 + 1u) << 8u); }
        let vu10 = sg_w4b[q1 * gpr + g0];
        var cvu11 = sg_byte_b(cu1 + cbo1);
        if (sh1 > 3u) { cvu11 = cvu11 | (sg_byte_b(cu1 + cbo1 + 1u) << 8u); }
        let vu11 = sg_w4b[q1 * gpr + g1];
        ga0 = ga0 + exp2(pg0.x + f32((cvg00 >> sh0) & 31u) * pg0.y)
            * (sg_dot8(vg00.x, xa0, xb0) + sg_dot8(vg00.y, xc0, xd0)
             + sg_dot8(vg00.z, xe0, xf0) + sg_dot8(vg00.w, xg0, xh0));
        ga1 = ga1 + exp2(pg1.x + f32((cvg10 >> sh0) & 31u) * pg1.y)
            * (sg_dot8(vg10.x, xa0, xb0) + sg_dot8(vg10.y, xc0, xd0)
             + sg_dot8(vg10.z, xe0, xf0) + sg_dot8(vg10.w, xg0, xh0));
        ua0 = ua0 + exp2(pu0.x + f32((cvu00 >> sh0) & 31u) * pu0.y)
            * (sg_dot8(vu00.x, xa0, xb0) + sg_dot8(vu00.y, xc0, xd0)
             + sg_dot8(vu00.z, xe0, xf0) + sg_dot8(vu00.w, xg0, xh0));
        ua1 = ua1 + exp2(pu1.x + f32((cvu10 >> sh0) & 31u) * pu1.y)
            * (sg_dot8(vu10.x, xa0, xb0) + sg_dot8(vu10.y, xc0, xd0)
             + sg_dot8(vu10.z, xe0, xf0) + sg_dot8(vu10.w, xg0, xh0));
        if (g + 32u < gpr) {
            gb0 = gb0 + exp2(pg0.x + f32((cvg01 >> sh1) & 31u) * pg0.y)
                * (sg_dot8(vg01.x, xa1, xb1) + sg_dot8(vg01.y, xc1, xd1)
                 + sg_dot8(vg01.z, xe1, xf1) + sg_dot8(vg01.w, xg1, xh1));
            gb1 = gb1 + exp2(pg1.x + f32((cvg11 >> sh1) & 31u) * pg1.y)
                * (sg_dot8(vg11.x, xa1, xb1) + sg_dot8(vg11.y, xc1, xd1)
                 + sg_dot8(vg11.z, xe1, xf1) + sg_dot8(vg11.w, xg1, xh1));
            ub0 = ub0 + exp2(pu0.x + f32((cvu01 >> sh1) & 31u) * pu0.y)
                * (sg_dot8(vu01.x, xa1, xb1) + sg_dot8(vu01.y, xc1, xd1)
                 + sg_dot8(vu01.z, xe1, xf1) + sg_dot8(vu01.w, xg1, xh1));
            ub1 = ub1 + exp2(pu1.x + f32((cvu11 >> sh1) & 31u) * pu1.y)
                * (sg_dot8(vu11.x, xa1, xb1) + sg_dot8(vu11.y, xc1, xd1)
                 + sg_dot8(vu11.z, xe1, xf1) + sg_dot8(vu11.w, xg1, xh1));
        }
        g = g + 64u;
    }
    var gg0 = sg_tree(ga0 + gb0);
    var uu0 = sg_tree(ua0 + ub0);
    var gg1 = sg_tree(ga1 + gb1);
    var uu1 = sg_tree(ua1 + ub1);
    if (lane == 0u) {
        let lim = bitcast<f32>(sg_p.lim);
        if (l0) {
            if (lim > 0.0) { uu0 = clamp(uu0, -lim, lim); gg0 = min(gg0, lim); }
            if (sg_p.act == 1u) { sg_ya[r0] = sg_gelu_erf(gg0) * uu0; }
            else { sg_ya[r0] = (gg0 / (1.0 + exp(-gg0))) * uu0; }
        }
        if (l1) {
            if (lim > 0.0) { uu1 = clamp(uu1, -lim, lim); gg1 = min(gg1, lim); }
            if (sg_p.act == 1u) { sg_ya[r1] = sg_gelu_erf(gg1) * uu1; }
            else { sg_ya[r1] = (gg1 / (1.0 + exp(-gg1))) * uu1; }
        }
    }
}

@compute @workgroup_size(256)
fn q4tp_gu_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    run_gu(wid.x, lid >> 5u, lane);
}
"#;

pub(crate) const GU_SG_R4U2: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_gu(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 4u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cg0 = codes_b + q0 * cstride;
    let pg0 = unpack2x16float(sg_wa[params_w + q0]);
    var ga0 = 0.0;
    var gb0 = 0.0;
    let cu0 = codes_b + q0 * cstride;
    let pu0 = unpack2x16float(sg_wb[params_w + q0]);
    var ua0 = 0.0;
    var ub0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cg1 = codes_b + q1 * cstride;
    let pg1 = unpack2x16float(sg_wa[params_w + q1]);
    var ga1 = 0.0;
    var gb1 = 0.0;
    let cu1 = codes_b + q1 * cstride;
    let pu1 = unpack2x16float(sg_wb[params_w + q1]);
    var ua1 = 0.0;
    var ub1 = 0.0;
    let r2 = base + 2u;
    let l2 = r2 < rows;
    let q2 = select(base, r2, l2);
    let cg2 = codes_b + q2 * cstride;
    let pg2 = unpack2x16float(sg_wa[params_w + q2]);
    var ga2 = 0.0;
    var gb2 = 0.0;
    let cu2 = codes_b + q2 * cstride;
    let pu2 = unpack2x16float(sg_wb[params_w + q2]);
    var ua2 = 0.0;
    var ub2 = 0.0;
    let r3 = base + 3u;
    let l3 = r3 < rows;
    let q3 = select(base, r3, l3);
    let cg3 = codes_b + q3 * cstride;
    let pg3 = unpack2x16float(sg_wa[params_w + q3]);
    var ga3 = 0.0;
    var gb3 = 0.0;
    let cu3 = codes_b + q3 * cstride;
    let pu3 = unpack2x16float(sg_wb[params_w + q3]);
    var ua3 = 0.0;
    var ub3 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        var cvg00 = sg_byte_a(cg0 + cbo0);
        if (sh0 > 3u) { cvg00 = cvg00 | (sg_byte_a(cg0 + cbo0 + 1u) << 8u); }
        let vg00 = sg_w4a[q0 * gpr + g0];
        var cvg01 = sg_byte_a(cg0 + cbo1);
        if (sh1 > 3u) { cvg01 = cvg01 | (sg_byte_a(cg0 + cbo1 + 1u) << 8u); }
        let vg01 = sg_w4a[q0 * gpr + g1];
        var cvg10 = sg_byte_a(cg1 + cbo0);
        if (sh0 > 3u) { cvg10 = cvg10 | (sg_byte_a(cg1 + cbo0 + 1u) << 8u); }
        let vg10 = sg_w4a[q1 * gpr + g0];
        var cvg11 = sg_byte_a(cg1 + cbo1);
        if (sh1 > 3u) { cvg11 = cvg11 | (sg_byte_a(cg1 + cbo1 + 1u) << 8u); }
        let vg11 = sg_w4a[q1 * gpr + g1];
        var cvg20 = sg_byte_a(cg2 + cbo0);
        if (sh0 > 3u) { cvg20 = cvg20 | (sg_byte_a(cg2 + cbo0 + 1u) << 8u); }
        let vg20 = sg_w4a[q2 * gpr + g0];
        var cvg21 = sg_byte_a(cg2 + cbo1);
        if (sh1 > 3u) { cvg21 = cvg21 | (sg_byte_a(cg2 + cbo1 + 1u) << 8u); }
        let vg21 = sg_w4a[q2 * gpr + g1];
        var cvg30 = sg_byte_a(cg3 + cbo0);
        if (sh0 > 3u) { cvg30 = cvg30 | (sg_byte_a(cg3 + cbo0 + 1u) << 8u); }
        let vg30 = sg_w4a[q3 * gpr + g0];
        var cvg31 = sg_byte_a(cg3 + cbo1);
        if (sh1 > 3u) { cvg31 = cvg31 | (sg_byte_a(cg3 + cbo1 + 1u) << 8u); }
        let vg31 = sg_w4a[q3 * gpr + g1];
        var cvu00 = sg_byte_b(cu0 + cbo0);
        if (sh0 > 3u) { cvu00 = cvu00 | (sg_byte_b(cu0 + cbo0 + 1u) << 8u); }
        let vu00 = sg_w4b[q0 * gpr + g0];
        var cvu01 = sg_byte_b(cu0 + cbo1);
        if (sh1 > 3u) { cvu01 = cvu01 | (sg_byte_b(cu0 + cbo1 + 1u) << 8u); }
        let vu01 = sg_w4b[q0 * gpr + g1];
        var cvu10 = sg_byte_b(cu1 + cbo0);
        if (sh0 > 3u) { cvu10 = cvu10 | (sg_byte_b(cu1 + cbo0 + 1u) << 8u); }
        let vu10 = sg_w4b[q1 * gpr + g0];
        var cvu11 = sg_byte_b(cu1 + cbo1);
        if (sh1 > 3u) { cvu11 = cvu11 | (sg_byte_b(cu1 + cbo1 + 1u) << 8u); }
        let vu11 = sg_w4b[q1 * gpr + g1];
        var cvu20 = sg_byte_b(cu2 + cbo0);
        if (sh0 > 3u) { cvu20 = cvu20 | (sg_byte_b(cu2 + cbo0 + 1u) << 8u); }
        let vu20 = sg_w4b[q2 * gpr + g0];
        var cvu21 = sg_byte_b(cu2 + cbo1);
        if (sh1 > 3u) { cvu21 = cvu21 | (sg_byte_b(cu2 + cbo1 + 1u) << 8u); }
        let vu21 = sg_w4b[q2 * gpr + g1];
        var cvu30 = sg_byte_b(cu3 + cbo0);
        if (sh0 > 3u) { cvu30 = cvu30 | (sg_byte_b(cu3 + cbo0 + 1u) << 8u); }
        let vu30 = sg_w4b[q3 * gpr + g0];
        var cvu31 = sg_byte_b(cu3 + cbo1);
        if (sh1 > 3u) { cvu31 = cvu31 | (sg_byte_b(cu3 + cbo1 + 1u) << 8u); }
        let vu31 = sg_w4b[q3 * gpr + g1];
        ga0 = ga0 + exp2(pg0.x + f32((cvg00 >> sh0) & 31u) * pg0.y)
            * (sg_dot8(vg00.x, xa0, xb0) + sg_dot8(vg00.y, xc0, xd0)
             + sg_dot8(vg00.z, xe0, xf0) + sg_dot8(vg00.w, xg0, xh0));
        ga1 = ga1 + exp2(pg1.x + f32((cvg10 >> sh0) & 31u) * pg1.y)
            * (sg_dot8(vg10.x, xa0, xb0) + sg_dot8(vg10.y, xc0, xd0)
             + sg_dot8(vg10.z, xe0, xf0) + sg_dot8(vg10.w, xg0, xh0));
        ga2 = ga2 + exp2(pg2.x + f32((cvg20 >> sh0) & 31u) * pg2.y)
            * (sg_dot8(vg20.x, xa0, xb0) + sg_dot8(vg20.y, xc0, xd0)
             + sg_dot8(vg20.z, xe0, xf0) + sg_dot8(vg20.w, xg0, xh0));
        ga3 = ga3 + exp2(pg3.x + f32((cvg30 >> sh0) & 31u) * pg3.y)
            * (sg_dot8(vg30.x, xa0, xb0) + sg_dot8(vg30.y, xc0, xd0)
             + sg_dot8(vg30.z, xe0, xf0) + sg_dot8(vg30.w, xg0, xh0));
        ua0 = ua0 + exp2(pu0.x + f32((cvu00 >> sh0) & 31u) * pu0.y)
            * (sg_dot8(vu00.x, xa0, xb0) + sg_dot8(vu00.y, xc0, xd0)
             + sg_dot8(vu00.z, xe0, xf0) + sg_dot8(vu00.w, xg0, xh0));
        ua1 = ua1 + exp2(pu1.x + f32((cvu10 >> sh0) & 31u) * pu1.y)
            * (sg_dot8(vu10.x, xa0, xb0) + sg_dot8(vu10.y, xc0, xd0)
             + sg_dot8(vu10.z, xe0, xf0) + sg_dot8(vu10.w, xg0, xh0));
        ua2 = ua2 + exp2(pu2.x + f32((cvu20 >> sh0) & 31u) * pu2.y)
            * (sg_dot8(vu20.x, xa0, xb0) + sg_dot8(vu20.y, xc0, xd0)
             + sg_dot8(vu20.z, xe0, xf0) + sg_dot8(vu20.w, xg0, xh0));
        ua3 = ua3 + exp2(pu3.x + f32((cvu30 >> sh0) & 31u) * pu3.y)
            * (sg_dot8(vu30.x, xa0, xb0) + sg_dot8(vu30.y, xc0, xd0)
             + sg_dot8(vu30.z, xe0, xf0) + sg_dot8(vu30.w, xg0, xh0));
        if (g + 32u < gpr) {
            gb0 = gb0 + exp2(pg0.x + f32((cvg01 >> sh1) & 31u) * pg0.y)
                * (sg_dot8(vg01.x, xa1, xb1) + sg_dot8(vg01.y, xc1, xd1)
                 + sg_dot8(vg01.z, xe1, xf1) + sg_dot8(vg01.w, xg1, xh1));
            gb1 = gb1 + exp2(pg1.x + f32((cvg11 >> sh1) & 31u) * pg1.y)
                * (sg_dot8(vg11.x, xa1, xb1) + sg_dot8(vg11.y, xc1, xd1)
                 + sg_dot8(vg11.z, xe1, xf1) + sg_dot8(vg11.w, xg1, xh1));
            gb2 = gb2 + exp2(pg2.x + f32((cvg21 >> sh1) & 31u) * pg2.y)
                * (sg_dot8(vg21.x, xa1, xb1) + sg_dot8(vg21.y, xc1, xd1)
                 + sg_dot8(vg21.z, xe1, xf1) + sg_dot8(vg21.w, xg1, xh1));
            gb3 = gb3 + exp2(pg3.x + f32((cvg31 >> sh1) & 31u) * pg3.y)
                * (sg_dot8(vg31.x, xa1, xb1) + sg_dot8(vg31.y, xc1, xd1)
                 + sg_dot8(vg31.z, xe1, xf1) + sg_dot8(vg31.w, xg1, xh1));
            ub0 = ub0 + exp2(pu0.x + f32((cvu01 >> sh1) & 31u) * pu0.y)
                * (sg_dot8(vu01.x, xa1, xb1) + sg_dot8(vu01.y, xc1, xd1)
                 + sg_dot8(vu01.z, xe1, xf1) + sg_dot8(vu01.w, xg1, xh1));
            ub1 = ub1 + exp2(pu1.x + f32((cvu11 >> sh1) & 31u) * pu1.y)
                * (sg_dot8(vu11.x, xa1, xb1) + sg_dot8(vu11.y, xc1, xd1)
                 + sg_dot8(vu11.z, xe1, xf1) + sg_dot8(vu11.w, xg1, xh1));
            ub2 = ub2 + exp2(pu2.x + f32((cvu21 >> sh1) & 31u) * pu2.y)
                * (sg_dot8(vu21.x, xa1, xb1) + sg_dot8(vu21.y, xc1, xd1)
                 + sg_dot8(vu21.z, xe1, xf1) + sg_dot8(vu21.w, xg1, xh1));
            ub3 = ub3 + exp2(pu3.x + f32((cvu31 >> sh1) & 31u) * pu3.y)
                * (sg_dot8(vu31.x, xa1, xb1) + sg_dot8(vu31.y, xc1, xd1)
                 + sg_dot8(vu31.z, xe1, xf1) + sg_dot8(vu31.w, xg1, xh1));
        }
        g = g + 64u;
    }
    var gg0 = sg_tree(ga0 + gb0);
    var uu0 = sg_tree(ua0 + ub0);
    var gg1 = sg_tree(ga1 + gb1);
    var uu1 = sg_tree(ua1 + ub1);
    var gg2 = sg_tree(ga2 + gb2);
    var uu2 = sg_tree(ua2 + ub2);
    var gg3 = sg_tree(ga3 + gb3);
    var uu3 = sg_tree(ua3 + ub3);
    if (lane == 0u) {
        let lim = bitcast<f32>(sg_p.lim);
        if (l0) {
            if (lim > 0.0) { uu0 = clamp(uu0, -lim, lim); gg0 = min(gg0, lim); }
            if (sg_p.act == 1u) { sg_ya[r0] = sg_gelu_erf(gg0) * uu0; }
            else { sg_ya[r0] = (gg0 / (1.0 + exp(-gg0))) * uu0; }
        }
        if (l1) {
            if (lim > 0.0) { uu1 = clamp(uu1, -lim, lim); gg1 = min(gg1, lim); }
            if (sg_p.act == 1u) { sg_ya[r1] = sg_gelu_erf(gg1) * uu1; }
            else { sg_ya[r1] = (gg1 / (1.0 + exp(-gg1))) * uu1; }
        }
        if (l2) {
            if (lim > 0.0) { uu2 = clamp(uu2, -lim, lim); gg2 = min(gg2, lim); }
            if (sg_p.act == 1u) { sg_ya[r2] = sg_gelu_erf(gg2) * uu2; }
            else { sg_ya[r2] = (gg2 / (1.0 + exp(-gg2))) * uu2; }
        }
        if (l3) {
            if (lim > 0.0) { uu3 = clamp(uu3, -lim, lim); gg3 = min(gg3, lim); }
            if (sg_p.act == 1u) { sg_ya[r3] = sg_gelu_erf(gg3) * uu3; }
            else { sg_ya[r3] = (gg3 / (1.0 + exp(-gg3))) * uu3; }
        }
    }
}

@compute @workgroup_size(256)
fn q4tp_gu_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    run_gu(wid.x, lid >> 5u, lane);
}
"#;

pub(crate) const GU_SG_R1U4: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_gu(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 1u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cg0 = codes_b + q0 * cstride;
    let pg0 = unpack2x16float(sg_wa[params_w + q0]);
    var ga0 = 0.0;
    var gb0 = 0.0;
    let cu0 = codes_b + q0 * cstride;
    let pu0 = unpack2x16float(sg_wb[params_w + q0]);
    var ua0 = 0.0;
    var ub0 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        let g2 = min(g + 64u, gpr - 1u);
        let bit2 = g2 * 5u;
        let cbo2 = bit2 >> 3u;
        let sh2 = bit2 & 7u;
        let xo2 = g2 * 8u;
        let xa2 = sg_x[xo2 + 0u];
        let xb2 = sg_x[xo2 + 1u];
        let xc2 = sg_x[xo2 + 2u];
        let xd2 = sg_x[xo2 + 3u];
        let xe2 = sg_x[xo2 + 4u];
        let xf2 = sg_x[xo2 + 5u];
        let xg2 = sg_x[xo2 + 6u];
        let xh2 = sg_x[xo2 + 7u];
        let g3 = min(g + 96u, gpr - 1u);
        let bit3 = g3 * 5u;
        let cbo3 = bit3 >> 3u;
        let sh3 = bit3 & 7u;
        let xo3 = g3 * 8u;
        let xa3 = sg_x[xo3 + 0u];
        let xb3 = sg_x[xo3 + 1u];
        let xc3 = sg_x[xo3 + 2u];
        let xd3 = sg_x[xo3 + 3u];
        let xe3 = sg_x[xo3 + 4u];
        let xf3 = sg_x[xo3 + 5u];
        let xg3 = sg_x[xo3 + 6u];
        let xh3 = sg_x[xo3 + 7u];
        var cvg00 = sg_byte_a(cg0 + cbo0);
        if (sh0 > 3u) { cvg00 = cvg00 | (sg_byte_a(cg0 + cbo0 + 1u) << 8u); }
        let vg00 = sg_w4a[q0 * gpr + g0];
        var cvg01 = sg_byte_a(cg0 + cbo1);
        if (sh1 > 3u) { cvg01 = cvg01 | (sg_byte_a(cg0 + cbo1 + 1u) << 8u); }
        let vg01 = sg_w4a[q0 * gpr + g1];
        var cvg02 = sg_byte_a(cg0 + cbo2);
        if (sh2 > 3u) { cvg02 = cvg02 | (sg_byte_a(cg0 + cbo2 + 1u) << 8u); }
        let vg02 = sg_w4a[q0 * gpr + g2];
        var cvg03 = sg_byte_a(cg0 + cbo3);
        if (sh3 > 3u) { cvg03 = cvg03 | (sg_byte_a(cg0 + cbo3 + 1u) << 8u); }
        let vg03 = sg_w4a[q0 * gpr + g3];
        var cvu00 = sg_byte_b(cu0 + cbo0);
        if (sh0 > 3u) { cvu00 = cvu00 | (sg_byte_b(cu0 + cbo0 + 1u) << 8u); }
        let vu00 = sg_w4b[q0 * gpr + g0];
        var cvu01 = sg_byte_b(cu0 + cbo1);
        if (sh1 > 3u) { cvu01 = cvu01 | (sg_byte_b(cu0 + cbo1 + 1u) << 8u); }
        let vu01 = sg_w4b[q0 * gpr + g1];
        var cvu02 = sg_byte_b(cu0 + cbo2);
        if (sh2 > 3u) { cvu02 = cvu02 | (sg_byte_b(cu0 + cbo2 + 1u) << 8u); }
        let vu02 = sg_w4b[q0 * gpr + g2];
        var cvu03 = sg_byte_b(cu0 + cbo3);
        if (sh3 > 3u) { cvu03 = cvu03 | (sg_byte_b(cu0 + cbo3 + 1u) << 8u); }
        let vu03 = sg_w4b[q0 * gpr + g3];
        ga0 = ga0 + exp2(pg0.x + f32((cvg00 >> sh0) & 31u) * pg0.y)
            * (sg_dot8(vg00.x, xa0, xb0) + sg_dot8(vg00.y, xc0, xd0)
             + sg_dot8(vg00.z, xe0, xf0) + sg_dot8(vg00.w, xg0, xh0));
        ua0 = ua0 + exp2(pu0.x + f32((cvu00 >> sh0) & 31u) * pu0.y)
            * (sg_dot8(vu00.x, xa0, xb0) + sg_dot8(vu00.y, xc0, xd0)
             + sg_dot8(vu00.z, xe0, xf0) + sg_dot8(vu00.w, xg0, xh0));
        if (g + 32u < gpr) {
            gb0 = gb0 + exp2(pg0.x + f32((cvg01 >> sh1) & 31u) * pg0.y)
                * (sg_dot8(vg01.x, xa1, xb1) + sg_dot8(vg01.y, xc1, xd1)
                 + sg_dot8(vg01.z, xe1, xf1) + sg_dot8(vg01.w, xg1, xh1));
            ub0 = ub0 + exp2(pu0.x + f32((cvu01 >> sh1) & 31u) * pu0.y)
                * (sg_dot8(vu01.x, xa1, xb1) + sg_dot8(vu01.y, xc1, xd1)
                 + sg_dot8(vu01.z, xe1, xf1) + sg_dot8(vu01.w, xg1, xh1));
        }
        if (g + 64u < gpr) {
            ga0 = ga0 + exp2(pg0.x + f32((cvg02 >> sh2) & 31u) * pg0.y)
                * (sg_dot8(vg02.x, xa2, xb2) + sg_dot8(vg02.y, xc2, xd2)
                 + sg_dot8(vg02.z, xe2, xf2) + sg_dot8(vg02.w, xg2, xh2));
            ua0 = ua0 + exp2(pu0.x + f32((cvu02 >> sh2) & 31u) * pu0.y)
                * (sg_dot8(vu02.x, xa2, xb2) + sg_dot8(vu02.y, xc2, xd2)
                 + sg_dot8(vu02.z, xe2, xf2) + sg_dot8(vu02.w, xg2, xh2));
        }
        if (g + 96u < gpr) {
            gb0 = gb0 + exp2(pg0.x + f32((cvg03 >> sh3) & 31u) * pg0.y)
                * (sg_dot8(vg03.x, xa3, xb3) + sg_dot8(vg03.y, xc3, xd3)
                 + sg_dot8(vg03.z, xe3, xf3) + sg_dot8(vg03.w, xg3, xh3));
            ub0 = ub0 + exp2(pu0.x + f32((cvu03 >> sh3) & 31u) * pu0.y)
                * (sg_dot8(vu03.x, xa3, xb3) + sg_dot8(vu03.y, xc3, xd3)
                 + sg_dot8(vu03.z, xe3, xf3) + sg_dot8(vu03.w, xg3, xh3));
        }
        g = g + 128u;
    }
    var gg0 = sg_tree(ga0 + gb0);
    var uu0 = sg_tree(ua0 + ub0);
    if (lane == 0u) {
        let lim = bitcast<f32>(sg_p.lim);
        if (l0) {
            if (lim > 0.0) { uu0 = clamp(uu0, -lim, lim); gg0 = min(gg0, lim); }
            if (sg_p.act == 1u) { sg_ya[r0] = sg_gelu_erf(gg0) * uu0; }
            else { sg_ya[r0] = (gg0 / (1.0 + exp(-gg0))) * uu0; }
        }
    }
}

@compute @workgroup_size(256)
fn q4tp_gu_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    run_gu(wid.x, lid >> 5u, lane);
}
"#;

pub(crate) const GU_SG_R2U4: &str = r#"
struct SgP { rows_a: u32, rows_b: u32, rows_c: u32, gpr: u32, lim: u32, act: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0)  var<storage, read>       sg_wa  : array<u32>;
@group(0) @binding(1)  var<storage, read>       sg_wb  : array<u32>;
@group(0) @binding(2)  var<storage, read>       sg_wc  : array<u32>;
@group(0) @binding(3)  var<storage, read>       sg_w4a : array<vec4<u32>>;
@group(0) @binding(4)  var<storage, read>       sg_w4b : array<vec4<u32>>;
@group(0) @binding(5)  var<storage, read>       sg_w4c : array<vec4<u32>>;
@group(0) @binding(6)  var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(7)  var<storage, read_write> sg_ya  : array<f32>;
@group(0) @binding(8)  var<storage, read_write> sg_yb  : array<f32>;
@group(0) @binding(9)  var<storage, read_write> sg_yc  : array<f32>;
@group(0) @binding(10) var<uniform>             sg_p   : SgP;
fn sg_byte_a(off: u32) -> u32 { return (sg_wa[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_b(off: u32) -> u32 { return (sg_wb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_byte_c(off: u32) -> u32 { return (sg_wc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sg_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x
         + sg_nib(w, 4u) * a.y
         + sg_nib(w, 8u) * a.z
         + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x
         + sg_nib(w, 20u) * b.y
         + sg_nib(w, 24u) * b.z
         + sg_nib(w, 28u) * b.w;
}
// The 64-lane tree of `q4tp_matvec16nl` from stride 16 down: lane l takes
// lane l+s's partial, in the same order (own + partner).
fn sg_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
fn sg_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn sg_gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + sg_erf(x * 0.70710678));
}
fn run_gu(blk: u32, warp: u32, lane: u32) {
    let rows = sg_p.rows_a;
    let gpr = sg_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let base = (blk * 8u + warp) * 2u;
    let r0 = base + 0u;
    let l0 = r0 < rows;
    let q0 = select(base, r0, l0);
    let cg0 = codes_b + q0 * cstride;
    let pg0 = unpack2x16float(sg_wa[params_w + q0]);
    var ga0 = 0.0;
    var gb0 = 0.0;
    let cu0 = codes_b + q0 * cstride;
    let pu0 = unpack2x16float(sg_wb[params_w + q0]);
    var ua0 = 0.0;
    var ub0 = 0.0;
    let r1 = base + 1u;
    let l1 = r1 < rows;
    let q1 = select(base, r1, l1);
    let cg1 = codes_b + q1 * cstride;
    let pg1 = unpack2x16float(sg_wa[params_w + q1]);
    var ga1 = 0.0;
    var gb1 = 0.0;
    let cu1 = codes_b + q1 * cstride;
    let pu1 = unpack2x16float(sg_wb[params_w + q1]);
    var ua1 = 0.0;
    var ub1 = 0.0;
    if (!l0) { return; }
    var g = lane;
    loop {
        if (g >= gpr) { break; }
        let g0 = min(g + 0u, gpr - 1u);
        let bit0 = g0 * 5u;
        let cbo0 = bit0 >> 3u;
        let sh0 = bit0 & 7u;
        let xo0 = g0 * 8u;
        let xa0 = sg_x[xo0 + 0u];
        let xb0 = sg_x[xo0 + 1u];
        let xc0 = sg_x[xo0 + 2u];
        let xd0 = sg_x[xo0 + 3u];
        let xe0 = sg_x[xo0 + 4u];
        let xf0 = sg_x[xo0 + 5u];
        let xg0 = sg_x[xo0 + 6u];
        let xh0 = sg_x[xo0 + 7u];
        let g1 = min(g + 32u, gpr - 1u);
        let bit1 = g1 * 5u;
        let cbo1 = bit1 >> 3u;
        let sh1 = bit1 & 7u;
        let xo1 = g1 * 8u;
        let xa1 = sg_x[xo1 + 0u];
        let xb1 = sg_x[xo1 + 1u];
        let xc1 = sg_x[xo1 + 2u];
        let xd1 = sg_x[xo1 + 3u];
        let xe1 = sg_x[xo1 + 4u];
        let xf1 = sg_x[xo1 + 5u];
        let xg1 = sg_x[xo1 + 6u];
        let xh1 = sg_x[xo1 + 7u];
        let g2 = min(g + 64u, gpr - 1u);
        let bit2 = g2 * 5u;
        let cbo2 = bit2 >> 3u;
        let sh2 = bit2 & 7u;
        let xo2 = g2 * 8u;
        let xa2 = sg_x[xo2 + 0u];
        let xb2 = sg_x[xo2 + 1u];
        let xc2 = sg_x[xo2 + 2u];
        let xd2 = sg_x[xo2 + 3u];
        let xe2 = sg_x[xo2 + 4u];
        let xf2 = sg_x[xo2 + 5u];
        let xg2 = sg_x[xo2 + 6u];
        let xh2 = sg_x[xo2 + 7u];
        let g3 = min(g + 96u, gpr - 1u);
        let bit3 = g3 * 5u;
        let cbo3 = bit3 >> 3u;
        let sh3 = bit3 & 7u;
        let xo3 = g3 * 8u;
        let xa3 = sg_x[xo3 + 0u];
        let xb3 = sg_x[xo3 + 1u];
        let xc3 = sg_x[xo3 + 2u];
        let xd3 = sg_x[xo3 + 3u];
        let xe3 = sg_x[xo3 + 4u];
        let xf3 = sg_x[xo3 + 5u];
        let xg3 = sg_x[xo3 + 6u];
        let xh3 = sg_x[xo3 + 7u];
        var cvg00 = sg_byte_a(cg0 + cbo0);
        if (sh0 > 3u) { cvg00 = cvg00 | (sg_byte_a(cg0 + cbo0 + 1u) << 8u); }
        let vg00 = sg_w4a[q0 * gpr + g0];
        var cvg01 = sg_byte_a(cg0 + cbo1);
        if (sh1 > 3u) { cvg01 = cvg01 | (sg_byte_a(cg0 + cbo1 + 1u) << 8u); }
        let vg01 = sg_w4a[q0 * gpr + g1];
        var cvg02 = sg_byte_a(cg0 + cbo2);
        if (sh2 > 3u) { cvg02 = cvg02 | (sg_byte_a(cg0 + cbo2 + 1u) << 8u); }
        let vg02 = sg_w4a[q0 * gpr + g2];
        var cvg03 = sg_byte_a(cg0 + cbo3);
        if (sh3 > 3u) { cvg03 = cvg03 | (sg_byte_a(cg0 + cbo3 + 1u) << 8u); }
        let vg03 = sg_w4a[q0 * gpr + g3];
        var cvg10 = sg_byte_a(cg1 + cbo0);
        if (sh0 > 3u) { cvg10 = cvg10 | (sg_byte_a(cg1 + cbo0 + 1u) << 8u); }
        let vg10 = sg_w4a[q1 * gpr + g0];
        var cvg11 = sg_byte_a(cg1 + cbo1);
        if (sh1 > 3u) { cvg11 = cvg11 | (sg_byte_a(cg1 + cbo1 + 1u) << 8u); }
        let vg11 = sg_w4a[q1 * gpr + g1];
        var cvg12 = sg_byte_a(cg1 + cbo2);
        if (sh2 > 3u) { cvg12 = cvg12 | (sg_byte_a(cg1 + cbo2 + 1u) << 8u); }
        let vg12 = sg_w4a[q1 * gpr + g2];
        var cvg13 = sg_byte_a(cg1 + cbo3);
        if (sh3 > 3u) { cvg13 = cvg13 | (sg_byte_a(cg1 + cbo3 + 1u) << 8u); }
        let vg13 = sg_w4a[q1 * gpr + g3];
        var cvu00 = sg_byte_b(cu0 + cbo0);
        if (sh0 > 3u) { cvu00 = cvu00 | (sg_byte_b(cu0 + cbo0 + 1u) << 8u); }
        let vu00 = sg_w4b[q0 * gpr + g0];
        var cvu01 = sg_byte_b(cu0 + cbo1);
        if (sh1 > 3u) { cvu01 = cvu01 | (sg_byte_b(cu0 + cbo1 + 1u) << 8u); }
        let vu01 = sg_w4b[q0 * gpr + g1];
        var cvu02 = sg_byte_b(cu0 + cbo2);
        if (sh2 > 3u) { cvu02 = cvu02 | (sg_byte_b(cu0 + cbo2 + 1u) << 8u); }
        let vu02 = sg_w4b[q0 * gpr + g2];
        var cvu03 = sg_byte_b(cu0 + cbo3);
        if (sh3 > 3u) { cvu03 = cvu03 | (sg_byte_b(cu0 + cbo3 + 1u) << 8u); }
        let vu03 = sg_w4b[q0 * gpr + g3];
        var cvu10 = sg_byte_b(cu1 + cbo0);
        if (sh0 > 3u) { cvu10 = cvu10 | (sg_byte_b(cu1 + cbo0 + 1u) << 8u); }
        let vu10 = sg_w4b[q1 * gpr + g0];
        var cvu11 = sg_byte_b(cu1 + cbo1);
        if (sh1 > 3u) { cvu11 = cvu11 | (sg_byte_b(cu1 + cbo1 + 1u) << 8u); }
        let vu11 = sg_w4b[q1 * gpr + g1];
        var cvu12 = sg_byte_b(cu1 + cbo2);
        if (sh2 > 3u) { cvu12 = cvu12 | (sg_byte_b(cu1 + cbo2 + 1u) << 8u); }
        let vu12 = sg_w4b[q1 * gpr + g2];
        var cvu13 = sg_byte_b(cu1 + cbo3);
        if (sh3 > 3u) { cvu13 = cvu13 | (sg_byte_b(cu1 + cbo3 + 1u) << 8u); }
        let vu13 = sg_w4b[q1 * gpr + g3];
        ga0 = ga0 + exp2(pg0.x + f32((cvg00 >> sh0) & 31u) * pg0.y)
            * (sg_dot8(vg00.x, xa0, xb0) + sg_dot8(vg00.y, xc0, xd0)
             + sg_dot8(vg00.z, xe0, xf0) + sg_dot8(vg00.w, xg0, xh0));
        ga1 = ga1 + exp2(pg1.x + f32((cvg10 >> sh0) & 31u) * pg1.y)
            * (sg_dot8(vg10.x, xa0, xb0) + sg_dot8(vg10.y, xc0, xd0)
             + sg_dot8(vg10.z, xe0, xf0) + sg_dot8(vg10.w, xg0, xh0));
        ua0 = ua0 + exp2(pu0.x + f32((cvu00 >> sh0) & 31u) * pu0.y)
            * (sg_dot8(vu00.x, xa0, xb0) + sg_dot8(vu00.y, xc0, xd0)
             + sg_dot8(vu00.z, xe0, xf0) + sg_dot8(vu00.w, xg0, xh0));
        ua1 = ua1 + exp2(pu1.x + f32((cvu10 >> sh0) & 31u) * pu1.y)
            * (sg_dot8(vu10.x, xa0, xb0) + sg_dot8(vu10.y, xc0, xd0)
             + sg_dot8(vu10.z, xe0, xf0) + sg_dot8(vu10.w, xg0, xh0));
        if (g + 32u < gpr) {
            gb0 = gb0 + exp2(pg0.x + f32((cvg01 >> sh1) & 31u) * pg0.y)
                * (sg_dot8(vg01.x, xa1, xb1) + sg_dot8(vg01.y, xc1, xd1)
                 + sg_dot8(vg01.z, xe1, xf1) + sg_dot8(vg01.w, xg1, xh1));
            gb1 = gb1 + exp2(pg1.x + f32((cvg11 >> sh1) & 31u) * pg1.y)
                * (sg_dot8(vg11.x, xa1, xb1) + sg_dot8(vg11.y, xc1, xd1)
                 + sg_dot8(vg11.z, xe1, xf1) + sg_dot8(vg11.w, xg1, xh1));
            ub0 = ub0 + exp2(pu0.x + f32((cvu01 >> sh1) & 31u) * pu0.y)
                * (sg_dot8(vu01.x, xa1, xb1) + sg_dot8(vu01.y, xc1, xd1)
                 + sg_dot8(vu01.z, xe1, xf1) + sg_dot8(vu01.w, xg1, xh1));
            ub1 = ub1 + exp2(pu1.x + f32((cvu11 >> sh1) & 31u) * pu1.y)
                * (sg_dot8(vu11.x, xa1, xb1) + sg_dot8(vu11.y, xc1, xd1)
                 + sg_dot8(vu11.z, xe1, xf1) + sg_dot8(vu11.w, xg1, xh1));
        }
        if (g + 64u < gpr) {
            ga0 = ga0 + exp2(pg0.x + f32((cvg02 >> sh2) & 31u) * pg0.y)
                * (sg_dot8(vg02.x, xa2, xb2) + sg_dot8(vg02.y, xc2, xd2)
                 + sg_dot8(vg02.z, xe2, xf2) + sg_dot8(vg02.w, xg2, xh2));
            ga1 = ga1 + exp2(pg1.x + f32((cvg12 >> sh2) & 31u) * pg1.y)
                * (sg_dot8(vg12.x, xa2, xb2) + sg_dot8(vg12.y, xc2, xd2)
                 + sg_dot8(vg12.z, xe2, xf2) + sg_dot8(vg12.w, xg2, xh2));
            ua0 = ua0 + exp2(pu0.x + f32((cvu02 >> sh2) & 31u) * pu0.y)
                * (sg_dot8(vu02.x, xa2, xb2) + sg_dot8(vu02.y, xc2, xd2)
                 + sg_dot8(vu02.z, xe2, xf2) + sg_dot8(vu02.w, xg2, xh2));
            ua1 = ua1 + exp2(pu1.x + f32((cvu12 >> sh2) & 31u) * pu1.y)
                * (sg_dot8(vu12.x, xa2, xb2) + sg_dot8(vu12.y, xc2, xd2)
                 + sg_dot8(vu12.z, xe2, xf2) + sg_dot8(vu12.w, xg2, xh2));
        }
        if (g + 96u < gpr) {
            gb0 = gb0 + exp2(pg0.x + f32((cvg03 >> sh3) & 31u) * pg0.y)
                * (sg_dot8(vg03.x, xa3, xb3) + sg_dot8(vg03.y, xc3, xd3)
                 + sg_dot8(vg03.z, xe3, xf3) + sg_dot8(vg03.w, xg3, xh3));
            gb1 = gb1 + exp2(pg1.x + f32((cvg13 >> sh3) & 31u) * pg1.y)
                * (sg_dot8(vg13.x, xa3, xb3) + sg_dot8(vg13.y, xc3, xd3)
                 + sg_dot8(vg13.z, xe3, xf3) + sg_dot8(vg13.w, xg3, xh3));
            ub0 = ub0 + exp2(pu0.x + f32((cvu03 >> sh3) & 31u) * pu0.y)
                * (sg_dot8(vu03.x, xa3, xb3) + sg_dot8(vu03.y, xc3, xd3)
                 + sg_dot8(vu03.z, xe3, xf3) + sg_dot8(vu03.w, xg3, xh3));
            ub1 = ub1 + exp2(pu1.x + f32((cvu13 >> sh3) & 31u) * pu1.y)
                * (sg_dot8(vu13.x, xa3, xb3) + sg_dot8(vu13.y, xc3, xd3)
                 + sg_dot8(vu13.z, xe3, xf3) + sg_dot8(vu13.w, xg3, xh3));
        }
        g = g + 128u;
    }
    var gg0 = sg_tree(ga0 + gb0);
    var uu0 = sg_tree(ua0 + ub0);
    var gg1 = sg_tree(ga1 + gb1);
    var uu1 = sg_tree(ua1 + ub1);
    if (lane == 0u) {
        let lim = bitcast<f32>(sg_p.lim);
        if (l0) {
            if (lim > 0.0) { uu0 = clamp(uu0, -lim, lim); gg0 = min(gg0, lim); }
            if (sg_p.act == 1u) { sg_ya[r0] = sg_gelu_erf(gg0) * uu0; }
            else { sg_ya[r0] = (gg0 / (1.0 + exp(-gg0))) * uu0; }
        }
        if (l1) {
            if (lim > 0.0) { uu1 = clamp(uu1, -lim, lim); gg1 = min(gg1, lim); }
            if (sg_p.act == 1u) { sg_ya[r1] = sg_gelu_erf(gg1) * uu1; }
            else { sg_ya[r1] = (gg1 / (1.0 + exp(-gg1))) * uu1; }
        }
    }
}

@compute @workgroup_size(256)
fn q4tp_gu_sg(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    run_gu(wid.x, lid >> 5u, lane);
}
"#;
