//! Column-split q4tp decode matvec for the SHORT, NARROW projections.
//!
//! `dense_mv`'s warp-per-row kernel gives a workgroup 32 rows (eight
//! subgroups of four) and the whole width of each row: a 5120-row matrix
//! is 160 workgroups, a 20480-row one 640. On an RTX PRO 6000 (188 SMs)
//! that is one and three workgroups an SM, eight warps each, and every
//! warp has only two to five dependent load-FMA steps before it is done
//! — the kernel never reaches the streaming state the long FFN rows do.
//! Measured on Ripple-65B q4tp: the o projection (5120 × 10240) at
//! 740 GB/s and q (20480 × 5120) at ~850 against the fused gate+up's
//! 1400 on the same card and the same bytes-per-row shape. More rows a
//! warp (R2) halved the loads in flight a warp and LOST; fewer rows a
//! warp doubled the activation traffic and LOST.
//!
//! This kernel keeps four rows a workgroup and splits the COLUMNS over
//! the workgroup's S subgroups instead: subgroup s owns the 32-column
//! groups of blocks [sU, sU+U) (a block is 32 groups, one a lane), every
//! load of the row is issued at once, the per-block partials meet in
//! shared memory and one subgroup a row folds them. Workgroups: rows/4
//! — 1280 for the o projection, 5120 for q.
//!
//! Each output is `q4tp_matvec16nl`'s to the bit. That kernel's lane l
//! (of 64) sums the groups l, l+64, l+128, … in order, lane l+32 the
//! groups l+32, l+96, …, the stride-32 tree step adds the two, and the
//! shuffle tree finishes. In blocks of 32 groups: lane l holds the even
//! blocks, lane l+32 the odd ones. The fold here is exactly that:
//! `((b0 + b2) + b4 …) + ((b1 + b3) + …)`, then `sg_tree`.
//!
//! One matrix a dispatch. MEASURED AND NOT TAKEN by default: on the RTX
//! PRO 6000 the same o projection ran 36 us a layer here against 31 on
//! the warp-per-row kernel (160 workgroups), and q 51 against 48 — the
//! card wants fewer, fatter warps (eight loads a lane a step) more than
//! it wants workgroups; the fold through shared memory and the
//! five-warp workgroup shape cost more than the grid gave. Kept as the
//! record and as an opt-in: `CMF_MV_SPLIT=1` takes it for every eligible
//! shape, `CMF_MV_SPLIT=auto` where the warp-per-row grid would give the
//! card fewer than `CMF_MV_SPLIT_WG` (8) workgroups an SM.

use super::*;

pub(crate) struct Split {
    pub(crate) pipe: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
}

pub(crate) struct SplitPipes {
    /// Pipelines by (subgroups, blocks a subgroup); None = rejected.
    by_shape: Mutex<HashMap<(u32, u32), Option<&'static Split>>>,
    /// Force (Some(true)), forbid (Some(false)) or decide by the grid.
    force: Option<bool>,
    /// Warp-per-row workgroups an SM below which this kernel takes over.
    per_sm: u32,
}

fn sg32(c: &Ctx) -> bool {
    c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
}

pub(crate) fn pipes(c: &Ctx) -> Option<&SplitPipes> {
    c.dense_split_pipes
        .get_or_init(|| {
            let force = match std::env::var("CMF_MV_SPLIT").as_deref() {
                Ok("1") => Some(true),
                Ok("auto") => None,
                _ => return None,
            };
            if !sg32(c) {
                return None;
            }
            let per_sm = std::env::var("CMF_MV_SPLIT_WG")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8);
            Some(SplitPipes {
                by_shape: Mutex::new(HashMap::new()),
                force,
                per_sm,
            })
        })
        .as_ref()
}

/// Subgroups and blocks a subgroup for `gpr` 32-column groups, or None
/// when the shape is not this kernel's (wider than 16 blocks: the FFN
/// down, which streams well as it is; fewer than 4 blocks: too narrow to
/// give four subgroups a row each in the fold).
fn shape(gpr: usize) -> Option<(u32, u32)> {
    let nblk = gpr.div_ceil(32);
    if !(4..=16).contains(&nblk) {
        return None;
    }
    let u = nblk.div_ceil(8) as u32;
    let s = (nblk as u32).div_ceil(u);
    Some((s, u))
}

fn wgsl(s: u32, u: u32) -> String {
    let mut o = String::new();
    o.push_str(
        r#"
struct SpP { rows: u32, gpr: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       sp_w  : array<u32>;
@group(0) @binding(1) var<storage, read>       sp_w4 : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       sp_x  : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> sp_y  : array<f32>;
@group(0) @binding(4) var<uniform>             sp_p  : SpP;
fn sp_byte(off: u32) -> u32 { return (sp_w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn sp_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sp_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sp_nib(w, 0u) * a.x
         + sp_nib(w, 4u) * a.y
         + sp_nib(w, 8u) * a.z
         + sp_nib(w, 12u) * a.w
         + sp_nib(w, 16u) * b.x
         + sp_nib(w, 20u) * b.y
         + sp_nib(w, 24u) * b.z
         + sp_nib(w, 28u) * b.w;
}
fn sp_tree(v0: f32) -> f32 {
    var v = v0;
    v = v + subgroupShuffleDown(v, 16u);
    v = v + subgroupShuffleDown(v, 8u);
    v = v + subgroupShuffleDown(v, 4u);
    v = v + subgroupShuffleDown(v, 2u);
    v = v + subgroupShuffleDown(v, 1u);
    return v;
}
"#,
    );
    let nb = s * u;
    // Partials: [block][row][lane].
    o.push_str(&format!(
        "var<workgroup> sp_part: array<f32, {}>;\n",
        nb * 4 * 32
    ));
    o.push_str(&format!(
        "@compute @workgroup_size({})\n\
         fn q4tp_mv_split(@builtin(workgroup_id) wid: vec3<u32>,\n\
                          @builtin(local_invocation_index) lid: u32,\n\
                          @builtin(subgroup_invocation_id) lane: u32) {{\n",
        s * 32
    ));
    o.push_str(
        r#"    let rows = sp_p.rows;
    let gpr = sp_p.gpr;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let sub = lid >> 5u;
    let base = wid.x * 4u;
"#,
    );
    for i in 0..4 {
        o.push_str(&format!(
            "    let r{i} = base + {i}u;\n\
             \x20   let l{i} = r{i} < rows;\n\
             \x20   let q{i} = select(base, r{i}, l{i});\n\
             \x20   let cm{i} = codes_b + q{i} * cstride;\n\
             \x20   let pm{i} = unpack2x16float(sp_w[params_w + q{i}]);\n"
        ));
    }
    // Every load of every block first, then the arithmetic.
    for k in 0..u {
        o.push_str(&format!(
            "    let g{k} = (sub * {u}u + {k}u) * 32u + lane;\n\
             \x20   let ok{k} = g{k} < gpr;\n\
             \x20   let gg{k} = min(g{k}, gpr - 1u);\n\
             \x20   let bit{k} = gg{k} * 5u;\n\
             \x20   let cbo{k} = bit{k} >> 3u;\n\
             \x20   let sh{k} = bit{k} & 7u;\n\
             \x20   let xo{k} = gg{k} * 8u;\n\
             \x20   let xa{k} = sp_x[xo{k} + 0u];\n\
             \x20   let xb{k} = sp_x[xo{k} + 1u];\n\
             \x20   let xc{k} = sp_x[xo{k} + 2u];\n\
             \x20   let xd{k} = sp_x[xo{k} + 3u];\n\
             \x20   let xe{k} = sp_x[xo{k} + 4u];\n\
             \x20   let xf{k} = sp_x[xo{k} + 5u];\n\
             \x20   let xg{k} = sp_x[xo{k} + 6u];\n\
             \x20   let xh{k} = sp_x[xo{k} + 7u];\n"
        ));
        for i in 0..4 {
            o.push_str(&format!(
                "    var cv{i}_{k} = sp_byte(cm{i} + cbo{k});\n\
                 \x20   if (sh{k} > 3u) {{ cv{i}_{k} = cv{i}_{k} | (sp_byte(cm{i} + cbo{k} + 1u) << 8u); }}\n\
                 \x20   let vm{i}_{k} = sp_w4[q{i} * gpr + gg{k}];\n"
            ));
        }
    }
    for k in 0..u {
        for i in 0..4 {
            o.push_str(&format!(
                "    var p{i}_{k} = 0.0;\n\
                 \x20   if (ok{k}) {{\n\
                 \x20       p{i}_{k} = exp2(pm{i}.x + f32((cv{i}_{k} >> sh{k}) & 31u) * pm{i}.y)\n\
                 \x20           * (sp_dot8(vm{i}_{k}.x, xa{k}, xb{k}) + sp_dot8(vm{i}_{k}.y, xc{k}, xd{k})\n\
                 \x20            + sp_dot8(vm{i}_{k}.z, xe{k}, xf{k}) + sp_dot8(vm{i}_{k}.w, xg{k}, xh{k}));\n\
                 \x20   }}\n\
                 \x20   sp_part[((sub * {u}u + {k}u) * 4u + {i}u) * 32u + lane] = p{i}_{k};\n"
            ));
        }
    }
    o.push_str("    workgroupBarrier();\n");
    // Subgroup i folds row i: even blocks, odd blocks, their sum, the tree.
    o.push_str("    if (sub < 4u) {\n");
    o.push_str("        let row = sub;\n");
    o.push_str("        var even = 0.0;\n        var odd = 0.0;\n");
    for b in 0..nb {
        let acc = if b % 2 == 0 { "even" } else { "odd" };
        o.push_str(&format!(
            "        {acc} = {acc} + sp_part[({b}u * 4u + row) * 32u + lane];\n"
        ));
    }
    o.push_str("        let t = sp_tree(even + odd);\n");
    o.push_str("        if (lane == 0u && base + row < rows) { sp_y[base + row] = t; }\n");
    o.push_str("    }\n}\n");
    o
}

fn build(c: &Ctx, s: u32, u: u32) -> Option<&'static Split> {
    let src = wgsl(s, u);
    let label = format!("cmf-mv-split-{s}x{u}");
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipe = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q4tp_mv_split"),
            layout: None,
            module: &module,
            entry_point: Some("q4tp_mv_split"),
            compilation_options: crate::gpu_wgpu::pco(),
            cache: c.pipeline_cache.as_ref(),
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("{label} rejected ({e}): the warp-per-row kernel stays");
        return None;
    }
    let layout = pipe.get_bind_group_layout(0);
    // The context is a process-wide static; a handful of pipelines live
    // as long as it does.
    Some(Box::leak(Box::new(Split { pipe, layout })))
}

/// One q4tp matvec `w` (rows × cols) of `xs` into `y` as rows/4
/// column-split workgroups, or None when the shape, the card or the
/// grid rule says the warp-per-row kernel should have it.
pub(crate) fn prep<'a>(
    c: &'a Ctx,
    xs: &wgpu::Buffer,
    cols: usize,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
) -> Option<(&'a wgpu::ComputePipeline, wgpu::BindGroup, u32)> {
    let p = pipes(c)?;
    if cols % 32 != 0 || rows == 0 {
        return None;
    }
    let gpr = cols / 32;
    let (s, u) = shape(gpr)?;
    let groups = rows.div_ceil(4);
    if groups as u32 > MAX_WG {
        return None;
    }
    match p.force {
        Some(false) => return None,
        Some(true) => {}
        None => {
            // The warp-per-row grid (32 rows a workgroup) against the SMs.
            let sm = c.sm_count.unwrap_or(48);
            if rows.div_ceil(32) as u32 >= p.per_sm * sm {
                return None;
            }
        }
    }
    let split = {
        let mut m = p.by_shape.lock().unwrap_or_else(|e| e.into_inner());
        *m.entry((s, u)).or_insert_with(|| {
            let b = build(c, s, u);
            if std::env::var("CMF_GPU_DEBUG").is_ok() {
                eprintln!(
                    "mv-split: {rows}x{cols} -> {s} subgroups x {u} blocks, {} workgroups, built {}",
                    groups,
                    b.is_some()
                );
            }
            b
        })
    }?;
    let uni = uniform_u32x4(c, [rows as u32, gpr as u32, 0, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mv-split"),
        layout: &split.layout,
        entries: &[
            bind_buf(0, w),
            bind_buf(1, w),
            bind_buf(2, xs),
            bind_buf(3, y),
            bind_buf(4, &uni),
        ],
    });
    Some((&split.pipe, bind, groups as u32))
}
