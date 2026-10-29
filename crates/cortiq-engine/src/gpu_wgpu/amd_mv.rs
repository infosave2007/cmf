//! q4tp decode matvecs for AMD GPUs under Metal (wave64 subgroups: the
//! Radeon Pro of Intel Macs).
//!
//! On a Radeon Pro 560X (Polaris, Metal 2, ~81 GB/s) the 16nl kernels stream
//! 20-47 GB/s of weights; a load-only twin of the same walk reaches 70-85, so
//! the kernel, not the bus, is the wall: over a 2560-wide row a lane walks one
//! or two groups, then a ladder table, a workgroup barrier and a six-step tree.
//! These dequantize with masked nibbles (x pre-scaled by 1, 1/16, 1/256,
//! 1/4096 and -8·Σx folded per group) and reduce with wave64 shuffles:
//! `k16` gives a wave sixteen rows, sixteen lanes a row walking groups l,
//! l+16, ... with the rung by exp2 in place (no table, no barrier) — the tall
//! shapes; `w64m` gives a wave four rows, 64 lanes a row — the rest. Not the
//! CPU's add order (|Δ|/max ≈ 2e-6). Measured standalone on the 560X, GB/s
//! (16nl → this): 4096x2560 29 → 33, 1024x2560 21 → 25, 2560x4096 36 → 35,
//! 10240x2560 33 → 42 (k16), 2560x10240 46 → 51, 131072x2560 37 → 54 (k16).
//! `CMF_AMD_MV=0` keeps the 16nl kernels.

use super::*;

pub(crate) struct AmdMv {
    k16: wgpu::ComputePipeline,
    w64m: wgpu::ComputePipeline,
}

fn eligible(c: &Ctx) -> bool {
    std::env::var("CMF_AMD_MV").as_deref() != Ok("0")
        && c.adapter_info.backend == wgpu::Backend::Metal
        && (c.adapter_info.name.contains("AMD") || c.adapter_info.name.contains("Radeon"))
        && c.device.features().contains(wgpu::Features::SUBGROUP)
}

fn build(c: &Ctx) -> Option<AmdMv> {
    if !eligible(c) {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-amd-mv"),
        source: wgpu::ShaderSource::Wgsl(AMD_MV_SRC.into()),
    });
    let pipe = |entry: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: crate::gpu_wgpu::pco(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let k16 = pipe("amd_q4tp_mv_k16");
    let w64m = pipe("amd_q4tp_mv_w64m");
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-amd-mv module rejected ({e}): q4tp decode keeps the 16nl kernels");
        return None;
    }
    Some(AmdMv { k16, w64m })
}

pub(crate) fn pipes(c: &Ctx) -> Option<&AmdMv> {
    c.amd_mv_pipes.get_or_init(|| build(c)).as_ref()
}

/// Whether this device decodes q4tp through these kernels (the pair and
/// fused gate+up kernels then stand down: on the 560X they gained nothing,
/// 62.9 / 61.2 / 62.1 ms a token with both, without x2, without both).
pub(crate) fn active(c: &Ctx) -> bool {
    pipes(c).is_some()
}

/// One q4tp matvec `w · xs → y` (`rows` x `cols`): (pipeline, bind group,
/// workgroups). `k16` from 8192 rows, `w64m` below.
pub(crate) fn prep<'a>(
    c: &'a Ctx,
    w: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> Option<(&'a wgpu::ComputePipeline, wgpu::BindGroup, u32)> {
    let p = pipes(c)?;
    if cols % 32 != 0 || rows == 0 {
        return None;
    }
    let (pipe, per) = if rows >= 8192 { (&p.k16, 16u32) } else { (&p.w64m, 4u32) };
    let p_buf = uniform_u32x4(c, [(cols / 32) as u32, rows as u32, cols as u32, 0]);
    let layout = pipe.get_bind_group_layout(0);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("amd-mv"),
        layout: &layout,
        entries: &[
            bind_buf(0, w),
            bind_buf(2, y),
            bind_buf(3, &p_buf),
            bind_buf(4, w),
            bind_buf(5, xs),
        ],
    });
    Some((pipe, bind, (rows as u32).div_ceil(per).min(MAX_WG)))
}

const AMD_MV_SRC: &str = r#"
struct Q1Params { np: u32, rows: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       q1w : array<u32>;
@group(0) @binding(2) var<storage, read_write> q1y : array<f32>;
@group(0) @binding(3) var<uniform>             q1p : Q1Params;
@group(0) @binding(4) var<storage, read> q4v_w : array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> q4v_x : array<vec4<f32>>;

fn q4tp_byte(off: u32) -> u32 {
    return (q1w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}
// Masked nibbles: four nibbles of a 16-bit half used in place, the x lanes
// pre-scaled by 1, 1/16, 1/256, 1/4096; "-8" folds into -8·Σx per group.
fn q4m_nib4(u: u32) -> vec4<f32> {
    return vec4<f32>(f32(u & 0xFu), f32(u & 0xF0u), f32(u & 0xF00u), f32(u & 0xF000u));
}
fn q4m_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(q4m_nib4(w), a) + dot(q4m_nib4(w >> 16u), b);
}

// w64m: one wave, four consecutive rows, 64 lanes a row (the 16nl lane walk).
var<workgroup> lad_m64: array<f32, 128>;
var<workgroup> part_m64: array<vec4<f32>, 64>;
@compute @workgroup_size(64)
fn amd_q4tp_mv_w64m(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) lid: u32,
               @builtin(subgroup_size) ssz: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let rows_t = select(rows, q1p._p1, q1p._p0 == 0xFFFFFFFEu);
    let params_w = rows_t * gpr * 4u;
    let codes_b = rows_t * gpr * 16u + rows_t * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let l = lid;
    let blocks = (rows + 3u) / 4u;
    var wb = wid.x;
    loop {
        if (wb >= blocks) { break; }
        let base = wb * 4u;
        for (var t = lid; t < 128u; t = t + 64u) {
            let r = base + (t >> 5u);
            if (r < rows) {
                let pr = unpack2x16float(q1w[params_w + r]);
                lad_m64[t] = exp2(pr.x + f32(t & 31u) * pr.y);
            }
        }
        workgroupBarrier();
        let r0 = base;
        let r1 = base + 1u;
        let r2 = base + 2u;
        let r3 = base + 3u;
        var acc = vec4<f32>(0.0);
        let c0 = codes_b + r0 * cstride;
        let c1 = codes_b + r1 * cstride;
        let c2 = codes_b + r2 * cstride;
        let c3 = codes_b + r3 * cstride;
        let l1 = r1 < rows;
        let l2 = r2 < rows;
        let l3 = r3 < rows;
        var g = l;
        loop {
            if (g >= gpr) { break; }
            let bit = g * 5u;
            let cbo = bit >> 3u;
            let sh = bit & 7u;
            let x0 = g * 8u;
            let ya = q4v_x[x0];      let yb = q4v_x[x0 + 1u];
            let yc = q4v_x[x0 + 2u]; let yd = q4v_x[x0 + 3u];
            let ye = q4v_x[x0 + 4u]; let yf = q4v_x[x0 + 5u];
            let yg = q4v_x[x0 + 6u]; let yh = q4v_x[x0 + 7u];
            let s4 = ya + yb + yc + yd + ye + yf + yg + yh;
            let m8 = -8.0 * (s4.x + s4.y + s4.z + s4.w);
            let sc = vec4<f32>(1.0, 0.0625, 0.00390625, 0.000244140625);
            let xa = ya * sc; let xb = yb * sc; let xc = yc * sc; let xd = yd * sc;
            let xe = ye * sc; let xf = yf * sc; let xg = yg * sc; let xh = yh * sc;
            var cv = q4tp_byte(c0 + cbo);
            if (sh > 3u) { cv = cv | (q4tp_byte(c0 + cbo + 1u) << 8u); }
            var v = q4v_w[r0 * gpr + g];
            acc.x = acc.x + lad_m64[((cv >> sh) & 31u)]
                * (q4m_dot8(v.x, xa, xb) + q4m_dot8(v.y, xc, xd)
                 + q4m_dot8(v.z, xe, xf) + q4m_dot8(v.w, xg, xh) + m8);
            if (l1) {
                cv = q4tp_byte(c1 + cbo);
                if (sh > 3u) { cv = cv | (q4tp_byte(c1 + cbo + 1u) << 8u); }
                v = q4v_w[r1 * gpr + g];
                acc.y = acc.y + lad_m64[32u + ((cv >> sh) & 31u)]
                    * (q4m_dot8(v.x, xa, xb) + q4m_dot8(v.y, xc, xd)
                     + q4m_dot8(v.z, xe, xf) + q4m_dot8(v.w, xg, xh) + m8);
            }
            if (l2) {
                cv = q4tp_byte(c2 + cbo);
                if (sh > 3u) { cv = cv | (q4tp_byte(c2 + cbo + 1u) << 8u); }
                v = q4v_w[r2 * gpr + g];
                acc.z = acc.z + lad_m64[64u + ((cv >> sh) & 31u)]
                    * (q4m_dot8(v.x, xa, xb) + q4m_dot8(v.y, xc, xd)
                     + q4m_dot8(v.z, xe, xf) + q4m_dot8(v.w, xg, xh) + m8);
            }
            if (l3) {
                cv = q4tp_byte(c3 + cbo);
                if (sh > 3u) { cv = cv | (q4tp_byte(c3 + cbo + 1u) << 8u); }
                v = q4v_w[r3 * gpr + g];
                acc.w = acc.w + lad_m64[96u + ((cv >> sh) & 31u)]
                    * (q4m_dot8(v.x, xa, xb) + q4m_dot8(v.y, xc, xd)
                     + q4m_dot8(v.z, xe, xf) + q4m_dot8(v.w, xg, xh) + m8);
            }
            g = g + 64u;
        }
        if (ssz == 64u) {
            acc = acc + subgroupShuffleDown(acc, 32u);
            acc = acc + subgroupShuffleDown(acc, 16u);
            acc = acc + subgroupShuffleDown(acc, 8u);
            acc = acc + subgroupShuffleDown(acc, 4u);
            acc = acc + subgroupShuffleDown(acc, 2u);
            acc = acc + subgroupShuffleDown(acc, 1u);
        } else {
            part_m64[lid] = acc;
            workgroupBarrier();
            var stride = 32u;
            loop {
                if (stride == 0u) { break; }
                if (l < stride) { part_m64[lid] = part_m64[lid] + part_m64[lid + stride]; }
                workgroupBarrier();
                stride = stride >> 1u;
            }
            acc = part_m64[0];
        }
        if (l == 0u) {
            if (r0 < rows) { q1y[r0] = acc.x; }
            if (l1) { q1y[r1] = acc.y; }
            if (l2) { q1y[r2] = acc.z; }
            if (l3) { q1y[r3] = acc.w; }
        }
        workgroupBarrier();
        wb = wb + nwg.x;
    }
}

// k16: one wave, sixteen rows, sixteen lanes a row walking groups l, l+16, ...
@compute @workgroup_size(64)
fn amd_q4tp_mv_k16(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let gpr = q1p.np;
    let rows = q1p.rows;
    let rows_t = select(rows, q1p._p1, q1p._p0 == 0xFFFFFFFEu);
    let params_w = rows_t * gpr * 4u;
    let codes_b = rows_t * gpr * 16u + rows_t * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let seg = lid >> 4u;
    let l = lid & 15u;
    let blocks = (rows + 15u) / 16u;
    var wb = wid.x;
    loop {
        if (wb >= blocks) { break; }
        let r0 = wb * 16u + seg * 4u;
        let r1 = r0 + 1u;
        let r2 = r0 + 2u;
        let r3 = r0 + 3u;
        let l0 = r0 < rows; let l1 = r1 < rows; let l2 = r2 < rows; let l3 = r3 < rows;
        var p0 = vec2<f32>(0.0); var p1 = vec2<f32>(0.0);
        var p2 = vec2<f32>(0.0); var p3 = vec2<f32>(0.0);
        if (l0) { p0 = unpack2x16float(q1w[params_w + r0]); }
        if (l1) { p1 = unpack2x16float(q1w[params_w + r1]); }
        if (l2) { p2 = unpack2x16float(q1w[params_w + r2]); }
        if (l3) { p3 = unpack2x16float(q1w[params_w + r3]); }
        var acc = vec4<f32>(0.0);
        var g = l;
        loop {
            if (g >= gpr) { break; }
            let bit = g * 5u;
            let cbo = bit >> 3u;
            let sh = bit & 7u;
            let x0 = g * 8u;
            let ya = q4v_x[x0];      let yb = q4v_x[x0 + 1u];
            let yc = q4v_x[x0 + 2u]; let yd = q4v_x[x0 + 3u];
            let ye = q4v_x[x0 + 4u]; let yf = q4v_x[x0 + 5u];
            let yg = q4v_x[x0 + 6u]; let yh = q4v_x[x0 + 7u];
            let s4 = ya + yb + yc + yd + ye + yf + yg + yh;
            let m8 = -8.0 * (s4.x + s4.y + s4.z + s4.w);
            let sc = vec4<f32>(1.0, 0.0625, 0.00390625, 0.000244140625);
            let xa = ya * sc; let xb = yb * sc; let xc = yc * sc; let xd = yd * sc;
            let xe = ye * sc; let xf = yf * sc; let xg = yg * sc; let xh = yh * sc;
            if (l0) {
                let cr = codes_b + r0 * cstride;
                var cv = q4tp_byte(cr + cbo);
                if (sh > 3u) { cv = cv | (q4tp_byte(cr + cbo + 1u) << 8u); }
                let v = q4v_w[r0 * gpr + g];
                acc.x = acc.x + exp2(p0.x + f32((cv >> sh) & 31u) * p0.y)
                    * (q4m_dot8(v.x, xa, xb) + q4m_dot8(v.y, xc, xd) + q4m_dot8(v.z, xe, xf) + q4m_dot8(v.w, xg, xh) + m8);
            }
            if (l1) {
                let cr = codes_b + r1 * cstride;
                var cv = q4tp_byte(cr + cbo);
                if (sh > 3u) { cv = cv | (q4tp_byte(cr + cbo + 1u) << 8u); }
                let v = q4v_w[r1 * gpr + g];
                acc.y = acc.y + exp2(p1.x + f32((cv >> sh) & 31u) * p1.y)
                    * (q4m_dot8(v.x, xa, xb) + q4m_dot8(v.y, xc, xd) + q4m_dot8(v.z, xe, xf) + q4m_dot8(v.w, xg, xh) + m8);
            }
            if (l2) {
                let cr = codes_b + r2 * cstride;
                var cv = q4tp_byte(cr + cbo);
                if (sh > 3u) { cv = cv | (q4tp_byte(cr + cbo + 1u) << 8u); }
                let v = q4v_w[r2 * gpr + g];
                acc.z = acc.z + exp2(p2.x + f32((cv >> sh) & 31u) * p2.y)
                    * (q4m_dot8(v.x, xa, xb) + q4m_dot8(v.y, xc, xd) + q4m_dot8(v.z, xe, xf) + q4m_dot8(v.w, xg, xh) + m8);
            }
            if (l3) {
                let cr = codes_b + r3 * cstride;
                var cv = q4tp_byte(cr + cbo);
                if (sh > 3u) { cv = cv | (q4tp_byte(cr + cbo + 1u) << 8u); }
                let v = q4v_w[r3 * gpr + g];
                acc.w = acc.w + exp2(p3.x + f32((cv >> sh) & 31u) * p3.y)
                    * (q4m_dot8(v.x, xa, xb) + q4m_dot8(v.y, xc, xd) + q4m_dot8(v.z, xe, xf) + q4m_dot8(v.w, xg, xh) + m8);
            }
            g = g + 16u;
        }
        acc = acc + subgroupShuffleDown(acc, 8u);
        acc = acc + subgroupShuffleDown(acc, 4u);
        acc = acc + subgroupShuffleDown(acc, 2u);
        acc = acc + subgroupShuffleDown(acc, 1u);
        if (l == 0u) {
            if (l0) { q1y[r0] = acc.x; }
            if (l1) { q1y[r1] = acc.y; }
            if (l2) { q1y[r2] = acc.z; }
            if (l3) { q1y[r3] = acc.w; }
        }
        wb = wb + nwg.x;
    }
}
"#;
