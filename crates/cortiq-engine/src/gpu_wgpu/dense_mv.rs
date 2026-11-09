//! Warp-per-row q4tp decode matvecs for the whole-token graph.
//!
//! `q4tp_matvec16nl` gives a 64-lane sub-block four rows, walks the row's
//! 16-byte groups two and a half at a time (5120 columns) and then pays a
//! six-barrier workgroup tree for every sixteen rows. On an RTX 3090 that
//! held Qwen3.8-27B's narrow projections at 37-52 % of the bus (GDN qkv+z
//! 349 GB/s, out_proj 357, FFN down 488). These kernels give a 32-lane
//! subgroup R rows (`CMF_MV_SG_R`, 4 or 2; gate+up `CMF_GU_SG_R`, 2 or 4),
//! two groups a lane a step with every load of the step issued before the
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
    /// The column-split twin (`ks_src`) with its own auto layout (the
    /// bindings are the same, the layouts are per pipeline); built only
    /// for `CMF_MV_SG_KS=1` (measured worse, see `sg_prep`).
    pub(crate) ks: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// The two-rows-a-subgroup kernel for the thin grids (`sg_prep`), built
    /// for `CMF_MV_SG_THIN=1` only (measured worse).
    pub(crate) thin: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// The plain matvec at two warps a workgroup, for the thin grids of a
    /// card with at most 96 SMs (`sg_prep`; `CMF_MV_SG_SMALL=0` keeps the
    /// eight-warp kernel for every shape).
    pub(crate) small: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    pub(crate) gu: wgpu::ComputePipeline,
    pub(crate) gu_l: wgpu::BindGroupLayout,
    /// Rows a subgroup owns.
    pub(crate) rpw: usize,
    /// Subgroups (warps) a workgroup of the plain matvec holds.
    pub(crate) wpg: usize,
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

fn shape(var_r: &str, r_default: usize) -> usize {
    match std::env::var(var_r).as_deref() {
        Ok("2") => 2,
        Ok("4") => 4,
        _ => r_default,
    }
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
            compilation_options: crate::gpu_wgpu::pco(),
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
    // Rows a subgroup owns (R); a lane loads two 16-byte groups a step,
    // every load of the step issued before its first FMA. The fused
    // gate+up carries two weight rows per output row, so it has its own.
    let rpw = shape("CMF_MV_SG_R", 4);
    let grpw = shape("CMF_GU_SG_R", 2);
    // Measured on Qwen3.8-27B / RTX 3090 plain decode (tok/s): R4U2 36.3,
    // R2U2 33.0, R8U2 31.3, R2U4 32.2, R4U4 32.5, R1 24.0; only R4U2 and
    // R2U2 are kept. Gate+up: R2U2 36.4, R4U2 36.1, R2U4 35.5, R1 33.1.
    let (src, rpw) = match rpw {
        2 => (MV_SG_R2U2, 2),
        _ => (MV_SG_R4U2, 4),
    };
    // Warps a workgroup (`CMF_MV_SG_W`: 1, 2, 4 or 8; default 8). On a
    // 188-SM RTX PRO 6000 a 5120-row matrix is 160 workgroups of eight
    // warps — 28 SMs idle — and the obvious cure, two warps a workgroup
    // (640 workgroups), MEASURED WORSE: the FFN down 87 → 100 us a layer,
    // the o projection 31 → 35, q 48 → 52 (one and four warps the same).
    // Eight warps of one workgroup share an SM's L1 for the activation
    // vector; spread thin, every warp streams it from L2. Same rows,
    // same bits whichever way: a warp's work does not depend on which
    // workgroup holds it. Kept as the record and the knob.
    let wpg: usize = match std::env::var("CMF_MV_SG_W").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        Ok("4") => 4,
        _ => 8,
    };
    let at_wpg = |src: &str, wpg: usize| {
        src.replace(
            "@compute @workgroup_size(256)\nfn q4tp_mv_sg(",
            &format!("@compute @workgroup_size({})\nfn q4tp_mv_sg(", 32 * wpg),
        )
        .replace("let per = 8u * 4u;", &format!("let per = {wpg}u * 4u;"))
        .replace("let per = 8u * 2u;", &format!("let per = {wpg}u * 2u;"))
        .replace("(blk * 8u + warp)", &format!("(blk * {wpg}u + warp)"))
    };
    let with_wpg = |src: &str| at_wpg(src, wpg);
    // Two warps a workgroup for the THIN grids on a smaller card: a
    // 2560-row matrix is 80 eight-warp workgroups on a 64-SM RTX A5000,
    // 320 this way (Qwen3.5-4B decode: FFN down 1449 -> 1117 us a token,
    // GDN out 629 -> 512, o 226 -> 185; the wide matrices unchanged). The
    // 188-SM PRO 6000 measured two warps worse (see `wpg` above), so only
    // cards of at most 96 SMs take it.
    let small_src = (wpg == 8
        && c.sm_count.is_some_and(|n| n <= 96)
        && std::env::var("CMF_MV_SG_SMALL").as_deref() != Ok("0"))
        .then(|| at_wpg(src, 2));
    let src = with_wpg(src);
    debug_assert!(src.contains(&format!("@workgroup_size({})", 32 * wpg)));
    let (gsrc, grpw) = match grpw {
        4 => (GU_SG_R4U2, 4),
        _ => (GU_SG_R2U2, 2),
    };
    let mv = compile(c, "cmf-mv-sg", &src, "q4tp_mv_sg")?;
    // The column-split twin: the eight warps of a workgroup take the SAME
    // four rows, each an eighth of the columns, and the eight partials of
    // a row are added in a fixed order. Four rows a workgroup puts a
    // 5120-row matrix on 1280 workgroups (6.8 an SM on the 188-SM RTX PRO
    // 6000) where the row-split kernel has 160 and leaves 28 SMs idle.
    // MEASURED WORSE on that card for every shape (`sg_prep`); a knob.
    // Two rows a subgroup for the matrices whose four-row grid leaves SMs
    // idle: a 5120-row projection is 160 workgroups of 32 rows on a
    // 188-SM card, 320 of 16 rows this way. The same per-row arithmetic
    // (two groups a lane a step, the same tree), so the same bits — and
    // MEASURED WORSE on that card (RTX PRO 6000, 65B): the FFN down 87 ->
    // 97 us a layer, the o projection 31 -> 35. Four rows a warp keep
    // more loads in flight than the idle SMs cost. A knob: `CMF_MV_SG_THIN=1`.
    let thin = (rpw == 4 && std::env::var("CMF_MV_SG_THIN").as_deref() == Ok("1"))
        .then(|| compile(c, "cmf-mv-sg-thin", &with_wpg(MV_SG_R2U2), "q4tp_mv_sg"))
        .flatten()
        .map(|p| {
            let l = p.get_bind_group_layout(0);
            (p, l)
        });
    let ks = (rpw == 4 && std::env::var("CMF_MV_SG_KS").as_deref() == Ok("1"))
        .then(|| compile(c, "cmf-mv-sg-ks", &ks_src(), "q4tp_mv_sg_ks"))
        .flatten()
        .map(|p| {
            let l = p.get_bind_group_layout(0);
            (p, l)
        });
    let small = small_src
        .and_then(|ss| compile(c, "cmf-mv-sg-w2", &ss, "q4tp_mv_sg"))
        .map(|p| {
            let l = p.get_bind_group_layout(0);
            (p, l)
        });
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
        ks,
        thin,
        small,
        gu_l: gu.get_bind_group_layout(0),
        gu,
        rpw,
        wpg,
        grpw,
        dummy_y,
    })
}

/// `MV_SG_R4U2` with the columns split across the eight warps: `run_*`
/// get the row block from the workgroup alone, walk the warp's eighth of
/// the groups, and hand their four warp sums to shared memory; the entry
/// point adds the eight in warp order and writes the rows.
fn ks_src() -> String {
    let mut s = MV_SG_R4U2.to_string();
    // The warp's slice of the groups.
    assert_eq!(s.matches("    let base = (blk * 8u + warp) * 4u;\n").count(), 3);
    s = s.replace(
        "    let base = (blk * 8u + warp) * 4u;\n",
        "    let base = blk * 4u;\n    let gq = (gpr + 7u) / 8u;\n    let gend = min(gpr, (warp + 1u) * gq);\n",
    );
    // A dead row block must still reach the barrier: no early return, an
    // empty slice instead.
    let old_loop = "    if (!l0) { return; }\n    var g = lane;\n    loop {\n        if (g >= gpr) { break; }\n";
    assert_eq!(s.matches(old_loop).count(), 3);
    s = s.replace(
        old_loop,
        "    var g = warp * gq + lane;\n    if (!l0) { g = gend; }\n    loop {\n        if (g >= gend) { break; }\n",
    );
    assert_eq!(s.matches("        if (g + 32u < gpr) {\n").count(), 3);
    s = s.replace("        if (g + 32u < gpr) {\n", "        if (g + 32u < gend) {\n");
    // The warp sums go to shared memory instead of the output.
    for m in ["a", "b", "c"] {
        let old = format!(
            "    if (lane == 0u) {{\n        if (l0) {{ sg_y{m}[r0] = t0; }}\n        if (l1) {{ sg_y{m}[r1] = t1; }}\n        if (l2) {{ sg_y{m}[r2] = t2; }}\n        if (l3) {{ sg_y{m}[r3] = t3; }}\n    }}\n"
        );
        assert_eq!(s.matches(old.as_str()).count(), 1, "sum store of run_{m}");
        s = s.replace(
            old.as_str(),
            "    if (lane == 0u) {\n        ks_part[warp * 4u + 0u] = t0;\n        ks_part[warp * 4u + 1u] = t1;\n        ks_part[warp * 4u + 2u] = t2;\n        ks_part[warp * 4u + 3u] = t3;\n    }\n",
        );
    }
    s = s.replace("var<storage, read_write> sg_yc  : array<f32>;\n", "var<storage, read_write> sg_yc  : array<f32>;\nvar<workgroup> ks_part: array<f32, 32>;\n");
    let old_entry = s.find("@compute @workgroup_size(256)\nfn q4tp_mv_sg(").expect("sg entry");
    s.truncate(old_entry);
    s.push_str(
        "@compute @workgroup_size(256)\nfn q4tp_mv_sg_ks(@builtin(workgroup_id) wid: vec3<u32>,\n                 @builtin(local_invocation_index) lid: u32,\n                 @builtin(subgroup_invocation_id) lane: u32) {\n    let warp = lid >> 5u;\n    let ba = (sg_p.rows_a + 3u) / 4u;\n    let bb = (sg_p.rows_b + 3u) / 4u;\n    let wb = wid.x;\n    var m = 0u;\n    var blk = wb;\n    if (wb < ba) {\n        run_a(wb, warp, lane);\n    } else if (wb < ba + bb) {\n        m = 1u;\n        blk = wb - ba;\n        run_b(blk, warp, lane);\n    } else {\n        m = 2u;\n        blk = wb - ba - bb;\n        run_c(blk, warp, lane);\n    }\n    workgroupBarrier();\n    if (lid < 4u) {\n        var t = ks_part[lid];\n        for (var w = 1u; w < 8u; w = w + 1u) { t = t + ks_part[w * 4u + lid]; }\n        let r = blk * 4u + lid;\n        if (m == 0u) { if (r < sg_p.rows_a) { sg_ya[r] = t; } }\n        else if (m == 1u) { if (r < sg_p.rows_b) { sg_yb[r] = t; } }\n        else { if (r < sg_p.rows_c) { sg_yc[r] = t; } }\n    }\n}\n",
    );
    s
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
    let r = |i: usize| mats.get(i).map_or(0, |m| m.2);
    // Two knobs for the thin grids (fewer four-row workgroups than two an
    // SM), both MEASURED WORSE on an RTX PRO 6000 and off by default: the
    // two-row kernel (`CMF_MV_SG_THIN=1`, twice the workgroups, the same
    // bits: down 87 -> 97 us a layer, o 31 -> 35) and the column-split one
    // (`CMF_MV_SG_KS=1`: q 48 -> 72, o 31 -> 43, down 87 -> 97 — eight
    // warps reading eight segments of the same four rows fragment the
    // memory streams). Four rows a warp, eight warps a workgroup, keep the
    // most bytes in flight; the 28 idle SMs cost less than either cure.
    let per_rs = p.wpg * p.rpw;
    let groups_rs: usize = (0..3).map(|i| r(i).div_ceil(per_rs)).sum();
    let sm = c.sm_count.unwrap_or(48) as usize;
    let thin_grid = groups_rs < 2 * sm;
    let (pipe, layout, per) = match (p.ks.as_ref(), p.thin.as_ref(), p.small.as_ref()) {
        (Some((ks, l)), _, _) if ks_forced() => (ks, l, 4),
        (_, Some((t, l)), _) if thin_grid => (t, l, p.wpg * 2),
        (_, _, Some((sp, l))) if thin_grid => (sp, l, 2 * p.rpw),
        _ => (&p.mv, &p.mv_l, per_rs),
    };
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
        layout,
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
    Some((pipe, bind, groups as u32))
}

/// `CMF_MV_SG_KS=1`: the column-split kernel for every shape (a measurement knob).
fn ks_forced() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("CMF_MV_SG_KS").as_deref() == Ok("1"))
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
    lim_bits: u32,
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
        [inter as u32, inter as u32, 0, (cols / 32) as u32, lim_bits, act_code, 0, 0],
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
fn sg_gelu_tanh(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x * x * x)));
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
fn sg_gelu_tanh(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x * x * x)));
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
fn sg_gelu_tanh(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x * x * x)));
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
            else if (sg_p.act == 2u) { sg_ya[r0] = sg_gelu_tanh(gg0) * uu0; }
            else { sg_ya[r0] = (gg0 / (1.0 + exp(-gg0))) * uu0; }
        }
        if (l1) {
            if (lim > 0.0) { uu1 = clamp(uu1, -lim, lim); gg1 = min(gg1, lim); }
            if (sg_p.act == 1u) { sg_ya[r1] = sg_gelu_erf(gg1) * uu1; }
            else if (sg_p.act == 2u) { sg_ya[r1] = sg_gelu_tanh(gg1) * uu1; }
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
fn sg_gelu_tanh(x: f32) -> f32 {
    return 0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x * x * x)));
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
            else if (sg_p.act == 2u) { sg_ya[r0] = sg_gelu_tanh(gg0) * uu0; }
            else { sg_ya[r0] = (gg0 / (1.0 + exp(-gg0))) * uu0; }
        }
        if (l1) {
            if (lim > 0.0) { uu1 = clamp(uu1, -lim, lim); gg1 = min(gg1, lim); }
            if (sg_p.act == 1u) { sg_ya[r1] = sg_gelu_erf(gg1) * uu1; }
            else if (sg_p.act == 2u) { sg_ya[r1] = sg_gelu_tanh(gg1) * uu1; }
            else { sg_ya[r1] = (gg1 / (1.0 + exp(-gg1))) * uu1; }
        }
        if (l2) {
            if (lim > 0.0) { uu2 = clamp(uu2, -lim, lim); gg2 = min(gg2, lim); }
            if (sg_p.act == 1u) { sg_ya[r2] = sg_gelu_erf(gg2) * uu2; }
            else if (sg_p.act == 2u) { sg_ya[r2] = sg_gelu_tanh(gg2) * uu2; }
            else { sg_ya[r2] = (gg2 / (1.0 + exp(-gg2))) * uu2; }
        }
        if (l3) {
            if (lim > 0.0) { uu3 = clamp(uu3, -lim, lim); gg3 = min(gg3, lim); }
            if (sg_p.act == 1u) { sg_ya[r3] = sg_gelu_erf(gg3) * uu3; }
            else if (sg_p.act == 2u) { sg_ya[r3] = sg_gelu_tanh(gg3) * uu3; }
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



/// `gdn_step_par` (decode) and `gdn_step_par_k` (batch) with their tree
/// reductions finished by subgroup shuffles: the stride-64 step through
/// workgroup memory, the stride-32 step and the 16..1 strides inside the
/// first subgroup, in the tree's own pair order — the same sums to the
/// bit with four barriers a reduction where the loop takes nine.
pub(crate) const GDN_SG_WGSL: &str = r#"
struct GdnP { nv: u32, dk: u32, dv: u32, kd: u32, rep: u32, cdim: u32, eps: f32, tok: u32 };
@group(0) @binding(0) var<storage, read>       gd_cq   : array<f32>;
@group(0) @binding(2) var<storage, read>       gd_a    : array<f32>;
@group(0) @binding(3) var<storage, read>       gd_b    : array<f32>;
@group(0) @binding(4) var<storage, read>       gd_alog : array<f32>;
@group(0) @binding(5) var<storage, read>       gd_dtb  : array<f32>;
@group(0) @binding(7) var<storage, read_write> gd_S4   : array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> gd_o4   : array<vec4<f32>>;
@group(0) @binding(9) var<uniform>             gd_p    : GdnP;
var<workgroup> sr1: array<f32, 128>;
var<workgroup> sr4: array<vec4<f32>, 128>;
var<workgroup> sbc: vec4<f32>;
fn gd_softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}
// The 128-lane tree `for stride 64..1: r[t] += r[t + stride]` and its
// trailing barrier, value of r[0] to every lane.
fn sgr1(t: u32, v: f32) -> f32 {
    sr1[t] = v;
    workgroupBarrier();
    if (t < 64u) { sr1[t] = sr1[t] + sr1[t + 64u]; }
    workgroupBarrier();
    if (t < 32u) {
        var s = sr1[t] + sr1[t + 32u];
        s = s + subgroupShuffleDown(s, 16u);
        s = s + subgroupShuffleDown(s, 8u);
        s = s + subgroupShuffleDown(s, 4u);
        s = s + subgroupShuffleDown(s, 2u);
        s = s + subgroupShuffleDown(s, 1u);
        if (t == 0u) { sbc.x = s; }
    }
    workgroupBarrier();
    let r = sbc.x;
    workgroupBarrier();
    return r;
}
fn sgr4(t: u32, v: vec4<f32>) -> vec4<f32> {
    sr4[t] = v;
    workgroupBarrier();
    if (t < 64u) { sr4[t] = sr4[t] + sr4[t + 64u]; }
    workgroupBarrier();
    if (t < 32u) {
        var s = sr4[t] + sr4[t + 32u];
        s = s + subgroupShuffleDown(s, 16u);
        s = s + subgroupShuffleDown(s, 8u);
        s = s + subgroupShuffleDown(s, 4u);
        s = s + subgroupShuffleDown(s, 2u);
        s = s + subgroupShuffleDown(s, 1u);
        if (t == 0u) { sbc = s; }
    }
    workgroupBarrier();
    let r = sbc;
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(128)
fn gdn_step_par_sg(@builtin(workgroup_id) wid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let dj4 = wid.y;
    let t = lid.x;
    let dk = gd_p.dk;
    let dv = gd_p.dv;
    if (h >= gd_p.nv || dj4 * 4u >= dv) { return; }
    let ko = h / gd_p.rep;
    let qs = ko * dk;
    let ks = gd_p.kd + ko * dk;
    let nq = sgr1(t, select(0.0, gd_cq[qs + t] * gd_cq[qs + t], t < dk));
    let nkn = sgr1(t, select(0.0, gd_cq[ks + t] * gd_cq[ks + t], t < dk));
    let invq = 1.0 / (sqrt(nq + 1e-6) * sqrt(f32(dk)));
    let invk = 1.0 / sqrt(nkn + 1e-6);
    let abo = gd_p.tok * gd_p.nv;
    let g = exp(-exp(gd_alog[h]) * gd_softplus(gd_a[abo + h] + gd_dtb[h]));
    let beta = 1.0 / (1.0 + exp(-gd_b[abo + h]));
    let s4base = (h * dk * dv) >> 2u;
    let dv4 = dv >> 2u;
    let vto = 2u * gd_p.kd + h * dv + dj4 * 4u;
    let vt = vec4<f32>(gd_cq[vto], gd_cq[vto + 1u], gd_cq[vto + 2u], gd_cq[vto + 3u]);
    let kf_t = select(0.0, gd_cq[ks + t] * invk, t < dk);
    let qf_t = select(0.0, gd_cq[qs + t] * invq, t < dk);
    var kv4 = vec4<f32>(0.0);
    if (t < dk) {
        kv4 = gd_S4[s4base + t * dv4 + dj4] * kf_t;
    }
    let kv = sgr4(t, kv4);
    let delta = (vt - g * kv) * beta;
    var contrib = vec4<f32>(0.0);
    if (t < dk) {
        let idx = s4base + t * dv4 + dj4;
        let cell = g * gd_S4[idx] + kf_t * delta;
        gd_S4[idx] = cell;
        contrib = qf_t * cell;
    }
    let o = sgr4(t, contrib);
    if (t == 0u) {
        let zo4 = (gd_p.tok * gd_p.nv * dv) >> 2u;
        gd_o4[zo4 + h * dv4 + dj4] = o;
    }
}
"#;

pub(crate) const GDNK_SG_WGSL: &str = r#"
struct GdKP {
    nv: u32, dk: u32, dv: u32, kd: u32,
    rep: u32, cdim: u32, eps: f32, kb: u32,
    stride: u32, ring_els: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read>       gdk_cq   : array<f32>;
@group(0) @binding(2) var<storage, read>       gdk_a    : array<f32>;
@group(0) @binding(3) var<storage, read>       gdk_b    : array<f32>;
@group(0) @binding(4) var<storage, read>       gdk_alog : array<f32>;
@group(0) @binding(5) var<storage, read>       gdk_dtb  : array<f32>;
@group(0) @binding(7) var<storage, read_write> gdk_S4   : array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> gdk_o4   : array<vec4<f32>>;
@group(0) @binding(9) var<uniform>             gdk_p    : GdKP;
@group(0) @binding(10) var<storage, read_write> gdk_snap4 : array<vec4<f32>>;
var<workgroup> sr1: array<f32, 128>;
var<workgroup> sr4: array<vec4<f32>, 128>;
var<workgroup> sbc: vec4<f32>;
fn gd_softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}
fn sgr1(t: u32, v: f32) -> f32 {
    sr1[t] = v;
    workgroupBarrier();
    if (t < 64u) { sr1[t] = sr1[t] + sr1[t + 64u]; }
    workgroupBarrier();
    if (t < 32u) {
        var s = sr1[t] + sr1[t + 32u];
        s = s + subgroupShuffleDown(s, 16u);
        s = s + subgroupShuffleDown(s, 8u);
        s = s + subgroupShuffleDown(s, 4u);
        s = s + subgroupShuffleDown(s, 2u);
        s = s + subgroupShuffleDown(s, 1u);
        if (t == 0u) { sbc.x = s; }
    }
    workgroupBarrier();
    let r = sbc.x;
    workgroupBarrier();
    return r;
}
fn sgr4(t: u32, v: vec4<f32>) -> vec4<f32> {
    sr4[t] = v;
    workgroupBarrier();
    if (t < 64u) { sr4[t] = sr4[t] + sr4[t + 64u]; }
    workgroupBarrier();
    if (t < 32u) {
        var s = sr4[t] + sr4[t + 32u];
        s = s + subgroupShuffleDown(s, 16u);
        s = s + subgroupShuffleDown(s, 8u);
        s = s + subgroupShuffleDown(s, 4u);
        s = s + subgroupShuffleDown(s, 2u);
        s = s + subgroupShuffleDown(s, 1u);
        if (t == 0u) { sbc = s; }
    }
    workgroupBarrier();
    let r = sbc;
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(128)
fn gdn_step_par_k_sg(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let dj4 = wid.y;
    let t = lid.x;
    let dk = gdk_p.dk;
    let dv = gdk_p.dv;
    if (h >= gdk_p.nv || dj4 * 4u >= dv) { return; }
    let ko = h / gdk_p.rep;
    let dv4 = dv >> 2u;
    let s4base = (h * dk * dv) >> 2u;
    for (var i = 0u; i < gdk_p.kb; i = i + 1u) {
        let cq0 = i * gdk_p.cdim;
        let qs = cq0 + ko * dk;
        let ks = cq0 + gdk_p.kd + ko * dk;
        let nq = sgr1(t, select(0.0, gdk_cq[qs + t] * gdk_cq[qs + t], t < dk));
        let nkn = sgr1(t, select(0.0, gdk_cq[ks + t] * gdk_cq[ks + t], t < dk));
        let invq = 1.0 / (sqrt(nq + 1e-6) * sqrt(f32(dk)));
        let invk = 1.0 / sqrt(nkn + 1e-6);
        let abo = i * gdk_p.nv;
        let g = exp(-exp(gdk_alog[h]) * gd_softplus(gdk_a[abo + h] + gdk_dtb[h]));
        let beta = 1.0 / (1.0 + exp(-gdk_b[abo + h]));
        let vto = cq0 + 2u * gdk_p.kd + h * dv + dj4 * 4u;
        let vt = vec4<f32>(gdk_cq[vto], gdk_cq[vto + 1u], gdk_cq[vto + 2u], gdk_cq[vto + 3u]);
        let kf_t = select(0.0, gdk_cq[ks + t] * invk, t < dk);
        let qf_t = select(0.0, gdk_cq[qs + t] * invq, t < dk);
        var kv4 = vec4<f32>(0.0);
        if (t < dk) {
            kv4 = gdk_S4[s4base + t * dv4 + dj4] * kf_t;
        }
        let kv = sgr4(t, kv4);
        let delta = (vt - g * kv) * beta;
        var contrib = vec4<f32>(0.0);
        if (t < dk) {
            let idx = s4base + t * dv4 + dj4;
            let cell = g * gdk_S4[idx] + kf_t * delta;
            gdk_S4[idx] = cell;
            if (gdk_p.stride != 0u) {
                gdk_snap4[(i * gdk_p.stride + gdk_p.ring_els) / 4u + idx] = cell;
            }
            contrib = qf_t * cell;
        }
        let o = sgr4(t, contrib);
        if (t == 0u) {
            gdk_o4[(i * gdk_p.nv * dv) / 4u + h * dv4 + dj4] = o;
        }
    }
}
"#;

/// The subgroup-tree GDN step twins (`CMF_GDN_SG=0` keeps the loops).
pub(crate) struct GdnSg {
    pub(crate) par: wgpu::ComputePipeline,
    pub(crate) par_l: wgpu::BindGroupLayout,
    pub(crate) park: wgpu::ComputePipeline,
    pub(crate) park_l: wgpu::BindGroupLayout,
    /// The register-resident prompt scan (`gdn_scan_k`, 128-wide keys):
    /// None under `CMF_GDN_SCAN=0` or on rejection.
    pub(crate) scan: Option<(wgpu::ComputePipeline, usize)>,
    /// The short-chain scan (`gdn_scan2_k`) and its position-parallel
    /// pre-pass (`gdn_prep_k`): None under `CMF_GDN_SCAN2=0` or on
    /// rejection (the scan above then runs).
    pub(crate) scan2: Option<(wgpu::ComputePipeline, wgpu::ComputePipeline, usize)>,
    /// Position-parallel twins of the chunk's conv and gated norm
    /// (`gdn_conv_par` + `gdn_ring_next`, `gdn_norm_rows`): None under
    /// `CMF_GDN_PAR=0` or on rejection.
    pub(crate) par_k: Option<GdnPar>,
}

pub(crate) struct GdnPar {
    pub(crate) conv: wgpu::ComputePipeline,
    pub(crate) ring: wgpu::ComputePipeline,
    pub(crate) norm: wgpu::ComputePipeline,
}

/// `gdn_conv_k` and `gdn_step_norm_k` walk the chunk's positions one after
/// another — the conv per column with its ring in device memory, the norm
/// per head with a 256-lane tree per position: about 0.2 ms each of a
/// 128-row chunk per layer on an RTX A5000, where neither carries state
/// between positions. Here every (position, column) of the conv is its own
/// lane, reading the chunk's inputs or the ring for the positions before
/// it, in the k-looped kernel's order of sums (bit-identical); the ring
/// takes the chunk's last inputs in a second dispatch. The norm is one
/// workgroup per (head, position), a subgroup sum per 32 columns.
pub(crate) const GDN_PAR_WGSL: &str = r#"
struct GcKP { cdim: u32, kk: u32, kb: u32, snap: u32, stride: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       gp_qkv : array<f32>;
@group(0) @binding(1) var<storage, read>       gp_taps: array<f32>;
@group(0) @binding(2) var<storage, read_write> gp_ring: array<f32>;
@group(0) @binding(3) var<storage, read_write> gp_cq  : array<f32>;
@group(0) @binding(4) var<uniform>             gp_p   : GcKP;
// Input row p of the chunk (p >= kk-1) or the ring row before it, for
// relative position p - (kk-1).
fn gp_x(p: u32, c: u32) -> f32 {
    let kk = gp_p.kk;
    if (p + 1u >= kk) { return gp_qkv[(p + 1u - kk) * gp_p.cdim + c]; }
    return gp_ring[p * gp_p.cdim + c];
}
@compute @workgroup_size(256)
fn gdn_conv_par(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.x * 256u + lid;
    let i = wid.y;
    let cdim = gp_p.cdim;
    if (c >= cdim || i >= gp_p.kb) { return; }
    let kk = gp_p.kk;
    let tb = c * kk;
    var acc = gp_qkv[i * cdim + c] * gp_taps[tb + kk - 1u];
    for (var j = 0u; j + 1u < kk; j = j + 1u) {
        acc = acc + gp_x(i + j, c) * gp_taps[tb + j];
    }
    gp_cq[i * cdim + c] = acc / (1.0 + exp(-acc));
}
@compute @workgroup_size(256)
fn gdn_ring_next(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let c = wid.x * 256u + lid;
    if (c >= gp_p.cdim || gp_p.kk < 2u) { return; }
    let kk = gp_p.kk;
    // The new ring row j holds relative position kb - (kk-1) + j, read
    // whole before any is written (a short chunk keeps old ring rows).
    var nr: array<f32, 8>;
    for (var j = 0u; j + 1u < kk && j < 8u; j = j + 1u) { nr[j] = gp_x(gp_p.kb + j, c); }
    for (var j = 0u; j + 1u < kk && j < 8u; j = j + 1u) { gp_ring[j * gp_p.cdim + c] = nr[j]; }
}

struct GnP {
    nv: u32, dk: u32, dv: u32, kd: u32,
    rep: u32, cdim: u32, eps: f32, kb: u32,
    stride: u32, ring_els: u32, p0: u32, p1: u32,
};
@group(0) @binding(5) var<storage, read>       gn_z : array<f32>;
@group(0) @binding(6) var<storage, read>       gn_w : array<f32>;
@group(0) @binding(7) var<storage, read_write> gn_o : array<f32>;
@group(0) @binding(8) var<uniform>             gn_p : GnP;
var<workgroup> gn_part: array<f32, 4>;
@compute @workgroup_size(128)
fn gdn_norm_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = wid.x;
    let i = wid.y;
    let dv = gn_p.dv;
    let idx = (i * gn_p.nv + h) * dv + t;
    var x = 0.0;
    if (t < dv) { x = gn_o[idx]; }
    let ss = subgroupAdd(x * x);
    if ((t & 31u) == 0u) { gn_part[t >> 5u] = ss; }
    workgroupBarrier();
    let tot = (gn_part[0] + gn_part[1]) + (gn_part[2] + gn_part[3]);
    let inv = 1.0 / sqrt(tot / f32(dv) + gn_p.eps);
    if (t < dv) {
        let zz = gn_z[idx];
        gn_o[idx] = x * inv * gn_w[t] * (zz / (1.0 + exp(-zz)));
    }
}
"#;

/// Key width the prompt scan is generated for (Qwen3.5 / 3.6 / 3.8).
pub(crate) const GDN_SCAN_DK: usize = 128;

/// The batch graph's GDN recurrence over a chunk with the state in
/// registers. `gdn_step_par_k_sg` keeps the state in device memory and
/// takes four 128-lane reductions per position (16 barriers): 1.7 ms of a
/// 128-row chunk per layer on an RTX A5000, a third of a Qwen3.5-4B prompt.
/// Here lane `l` of a workgroup owns column `l / P` of a head's state, rows
/// `(l % P) · R .. + R` (R = dk / P) as named registers; the q/k norms and
/// both state dot products are partial sums over a lane's rows, finished by
/// `log2 P` subgroup shuffles among the column's P adjacent lanes. No
/// barrier, no state traffic inside the position loop. Same recurrence as
/// the step kernels; the sums run in another order (not bit-identical).
fn gdn_scan_src(parts: usize) -> String {
    let rows = GDN_SCAN_DK / parts;
    let cols = 128 / parts;
    let dk = GDN_SCAN_DK;
    let mut s = String::new();
    s.push_str(
        r#"struct GsP {
    nv: u32, dk: u32, dv: u32, kd: u32,
    rep: u32, cdim: u32, eps: f32, kb: u32,
    stride: u32, ring_els: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read>       gs_cq   : array<f32>;
@group(0) @binding(2) var<storage, read>       gs_a    : array<f32>;
@group(0) @binding(3) var<storage, read>       gs_b    : array<f32>;
@group(0) @binding(4) var<storage, read>       gs_alog : array<f32>;
@group(0) @binding(5) var<storage, read>       gs_dtb  : array<f32>;
@group(0) @binding(7) var<storage, read_write> gs_S    : array<f32>;
@group(0) @binding(8) var<storage, read_write> gs_o    : array<f32>;
@group(0) @binding(9) var<uniform>             gs_p    : GsP;
fn gs_softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}
"#,
    );
    let red = |v: &str| {
        let mut t = String::new();
        let mut m = 1;
        while m < parts {
            t.push_str(&format!("        {v} = {v} + subgroupShuffleXor({v}, {m}u);\n"));
            m *= 2;
        }
        t
    };
    // Four partial chains per dot product (ILP), summed at the end.
    let dot = |name: &str, a: &str, b: &str| {
        let mut t = String::new();
        for c in 0..4 {
            let terms: Vec<String> = (c..rows).step_by(4).map(|u| format!("{a}{u} * {b}{u}")).collect();
            t.push_str(&format!("        let {name}_{c} = {};\n", terms.join(" + ")));
        }
        t.push_str(&format!("        var {name} = ({name}_0 + {name}_1) + ({name}_2 + {name}_3);\n"));
        t
    };
    // Position `at`'s inputs into the `n*` registers: loaded during the
    // position before, so the loop's critical path does not wait on memory.
    let load = |at: &str, ind: &str| {
        let mut t = String::new();
        t.push_str(&format!("{ind}let cqn = ({at}) * gs_p.cdim;\n"));
        t.push_str(&format!("{ind}let qsn = cqn + ko * {dk}u + r * {rows}u;\n{ind}let ksn = qsn + gs_p.kd;\n"));
        for u in 0..rows {
            t.push_str(&format!("{ind}nq{u} = gs_cq[qsn + {u}u];\n{ind}nk{u} = gs_cq[ksn + {u}u];\n"));
        }
        t.push_str(&format!(
            "{ind}na = gs_a[({at}) * gs_p.nv + h];\n{ind}nb = gs_b[({at}) * gs_p.nv + h];\n{ind}nvv = gs_cq[cqn + 2u * gs_p.kd + h * dv + j];\n"
        ));
        t
    };
    s.push_str(&format!(
        "@compute @workgroup_size(128)\nfn gdn_scan_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{\n    let r = lid % {parts}u;\n    let j0 = wid.y * {cols}u + lid / {parts}u;\n"
    ));
    s.push_str(
        "    let dv = gs_p.dv;\n    let live = wid.x < gs_p.nv && j0 < dv;\n    let h = select(0u, wid.x, live);\n    let j = select(0u, j0, live);\n    let ko = h / gs_p.rep;\n",
    );
    s.push_str(&format!("    let sb = h * {dk}u * dv + r * {rows}u * dv + j;\n"));
    for u in 0..rows {
        s.push_str(&format!("    var s{u} = gs_S[sb + {u}u * dv];\n"));
    }
    s.push_str("    let al = exp(gs_alog[h]);\n    let dtb = gs_dtb[h];\n");
    s.push_str(&format!("    let qsc = 1.0 / sqrt(f32({dk}));\n"));
    for u in 0..rows {
        s.push_str(&format!("    var nq{u} = 0.0;\n    var nk{u} = 0.0;\n"));
    }
    s.push_str("    var na = 0.0;\n    var nb = 0.0;\n    var nvv = 0.0;\n    if (gs_p.kb > 0u) {\n");
    s.push_str(&load("0u", "        "));
    s.push_str("    }\n    for (var i = 0u; i < gs_p.kb; i = i + 1u) {\n");
    for u in 0..rows {
        s.push_str(&format!("        let q{u} = nq{u};\n        let k{u} = nk{u};\n"));
    }
    s.push_str("        let pa = na;\n        let pb = nb;\n        let vt = nvv;\n        if (i + 1u < gs_p.kb) {\n");
    s.push_str(&load("i + 1u", "            "));
    s.push_str("        }\n");
    s.push_str(&dot("qq", "q", "q"));
    s.push_str(&red("qq"));
    s.push_str(&dot("kk", "k", "k"));
    s.push_str(&red("kk"));
    s.push_str(
        "        let invq = 1.0 / sqrt(qq + 1e-6) * qsc;\n        let invk = 1.0 / sqrt(kk + 1e-6);\n        let g = exp(-al * gs_softplus(pa + dtb));\n        let beta = 1.0 / (1.0 + exp(-pb));\n",
    );
    s.push_str(&dot("kv", "s", "k"));
    s.push_str(&red("kv"));
    s.push_str("        let dlt = (vt - g * (kv * invk)) * beta * invk;\n");
    for u in 0..rows {
        s.push_str(&format!("        s{u} = g * s{u} + k{u} * dlt;\n"));
    }
    s.push_str(&dot("o", "q", "s"));
    s.push_str(&red("o"));
    s.push_str("        if (live && r == 0u) { gs_o[i * gs_p.nv * dv + h * dv + j] = o * invq; }\n    }\n    if (live) {\n");
    for u in 0..rows {
        s.push_str(&format!("        gs_S[sb + {u}u * dv] = s{u};\n"));
    }
    s.push_str("    }\n}\n");
    s
}

/// The pre-pass of `gdn_scan2_k`, a workgroup per (position, key head):
/// the normalized q (times 1/sqrt(dk)) and k rows, their dot, and the
/// decay g and gate beta of the key head's value heads — everything a scan
/// step needs that does not depend on the state. Bindings: 0 the conv
/// output, 2/3 a/b, 4/5 A_log/dt_bias, 9 the step uniform, 10 q|k rows
/// `[pos][nk][2][128]`, 11 k·q `[pos][nk]`, 12 (g, beta) `[pos][nv]`.
pub(crate) const GDN_PREP_WGSL: &str = r#"
struct GsP {
    nv: u32, dk: u32, dv: u32, kd: u32,
    rep: u32, cdim: u32, eps: f32, kb: u32,
    stride: u32, ring_els: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read>       gp_cq   : array<f32>;
@group(0) @binding(2) var<storage, read>       gp_a    : array<f32>;
@group(0) @binding(3) var<storage, read>       gp_b    : array<f32>;
@group(0) @binding(4) var<storage, read>       gp_alog : array<f32>;
@group(0) @binding(5) var<storage, read>       gp_dtb  : array<f32>;
@group(0) @binding(9) var<uniform>             gp_p    : GsP;
@group(0) @binding(10) var<storage, read_write> gp_qk  : array<f32>;
@group(0) @binding(11) var<storage, read_write> gp_kq  : array<f32>;
@group(0) @binding(12) var<storage, read_write> gp_gb  : array<vec2<f32>>;
var<workgroup> gp_r: array<vec4<f32>, 4>;
fn gp_softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}
@compute @workgroup_size(128)
fn gdn_prep_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let ko = wid.x;
    let i = wid.y;
    let nk = gp_p.kd / 128u;
    let base = i * gp_p.cdim + ko * 128u + t;
    let q = gp_cq[base];
    let k = gp_cq[base + gp_p.kd];
    let s1 = subgroupAdd(q * q);
    let s2 = subgroupAdd(k * k);
    if ((t & 31u) == 0u) { gp_r[t >> 5u] = vec4<f32>(s1, s2, 0.0, 0.0); }
    workgroupBarrier();
    let tot = (gp_r[0] + gp_r[1]) + (gp_r[2] + gp_r[3]);
    let invq = 1.0 / sqrt(tot.x + 1e-6) * (1.0 / sqrt(128.0));
    let invk = 1.0 / sqrt(tot.y + 1e-6);
    let qf = q * invq;
    let kf = k * invk;
    let ob = (i * nk + ko) * 256u + t;
    gp_qk[ob] = qf;
    gp_qk[ob + 128u] = kf;
    let s3 = subgroupAdd(qf * kf);
    workgroupBarrier();
    if ((t & 31u) == 0u) { gp_r[t >> 5u].z = s3; }
    workgroupBarrier();
    if (t == 0u) { gp_kq[i * nk + ko] = (gp_r[0].z + gp_r[1].z) + (gp_r[2].z + gp_r[3].z); }
    if (t < gp_p.rep) {
        let h = ko * gp_p.rep + t;
        let abo = i * gp_p.nv + h;
        let g = exp(-exp(gp_alog[h]) * gp_softplus(gp_a[abo] + gp_dtb[h]));
        let beta = 1.0 / (1.0 + exp(-gp_b[abo]));
        gp_gb[abo] = vec2<f32>(g, beta);
    }
}
"#;

/// `gdn_scan_k` with the step reordered so its critical path is short:
/// o_t = g·(S_{t-1}ᵀq) + (k·q)·δ, so both state dot products read the
/// previous state at once, and the norms, g, beta and k·q come from
/// `gdn_prep_k` instead of the chain. Per step on the chain: two
/// independent partial dots, their `log2 P` shuffles side by side, δ, the
/// state update. `gdn_scan_k` spent ~1.7 µs a position (21 of a 512-row
/// Qwen3.5-4B chunk's GDN 24 ms on an RTX A5000) with the norms, a second
/// reduction and the output dot behind every update.
fn gdn_scan2_src(parts: usize) -> String {
    let rows = GDN_SCAN_DK / parts;
    let cols = 128 / parts;
    let dk = GDN_SCAN_DK;
    let mut s = String::new();
    s.push_str(
        r#"struct GsP {
    nv: u32, dk: u32, dv: u32, kd: u32,
    rep: u32, cdim: u32, eps: f32, kb: u32,
    stride: u32, ring_els: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read>       gs_cq   : array<f32>;
@group(0) @binding(7) var<storage, read_write> gs_S    : array<f32>;
@group(0) @binding(8) var<storage, read_write> gs_o    : array<f32>;
@group(0) @binding(9) var<uniform>             gs_p    : GsP;
@group(0) @binding(10) var<storage, read>      gs_qk   : array<f32>;
@group(0) @binding(11) var<storage, read>      gs_kq   : array<f32>;
@group(0) @binding(12) var<storage, read>      gs_gb   : array<vec2<f32>>;
"#,
    );
    let red2 = |a: &str, b: &str| {
        let mut t = String::new();
        let mut m = 1;
        while m < parts {
            t.push_str(&format!(
                "        {a} = {a} + subgroupShuffleXor({a}, {m}u);
        {b} = {b} + subgroupShuffleXor({b}, {m}u);
"
            ));
            m *= 2;
        }
        t
    };
    let dot = |name: &str, a: &str, b: &str| {
        let mut t = String::new();
        for c in 0..4 {
            let terms: Vec<String> = (c..rows).step_by(4).map(|u| format!("{a}{u} * {b}{u}")).collect();
            t.push_str(&format!("        let {name}_{c} = {};
", terms.join(" + ")));
        }
        t.push_str(&format!("        var {name} = ({name}_0 + {name}_1) + ({name}_2 + {name}_3);
"));
        t
    };
    let load = |at: &str, ind: &str| {
        let mut t = String::new();
        t.push_str(&format!("{ind}let qkb = (({at}) * nk + ko) * 256u + r * {rows}u;
"));
        for u in 0..rows {
            t.push_str(&format!("{ind}nq{u} = gs_qk[qkb + {u}u];
{ind}nk{u} = gs_qk[qkb + 128u + {u}u];
"));
        }
        t.push_str(&format!(
            "{ind}ngb = gs_gb[({at}) * gs_p.nv + h];
{ind}nkq = gs_kq[({at}) * nk + ko];
{ind}nvv = gs_cq[({at}) * gs_p.cdim + 2u * gs_p.kd + h * dv + j];
"
        ));
        t
    };
    s.push_str(&format!(
        "@compute @workgroup_size(128)
fn gdn_scan2_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let r = lid % {parts}u;
    let j0 = wid.y * {cols}u + lid / {parts}u;
"
    ));
    s.push_str(
        "    let dv = gs_p.dv;
    let nk = gs_p.kd / 128u;
    let live = wid.x < gs_p.nv && j0 < dv;
    let h = select(0u, wid.x, live);
    let j = select(0u, j0, live);
    let ko = h / gs_p.rep;
",
    );
    s.push_str(&format!("    let sb = h * {dk}u * dv + r * {rows}u * dv + j;
"));
    for u in 0..rows {
        s.push_str(&format!("    var s{u} = gs_S[sb + {u}u * dv];
"));
    }
    for u in 0..rows {
        s.push_str(&format!("    var nq{u} = 0.0;
    var nk{u} = 0.0;
"));
    }
    s.push_str("    var ngb = vec2<f32>(0.0);
    var nkq = 0.0;
    var nvv = 0.0;
    if (gs_p.kb > 0u) {
");
    s.push_str(&load("0u", "        "));
    s.push_str("    }
    for (var i = 0u; i < gs_p.kb; i = i + 1u) {
");
    for u in 0..rows {
        s.push_str(&format!("        let q{u} = nq{u};
        let k{u} = nk{u};
"));
    }
    s.push_str("        let gb = ngb;
        let kq = nkq;
        let vt = nvv;
        if (i + 1u < gs_p.kb) {
");
    s.push_str(&load("i + 1u", "            "));
    s.push_str("        }
");
    s.push_str(&dot("kv", "s", "k"));
    s.push_str(&dot("sq", "s", "q"));
    s.push_str(&red2("kv", "sq"));
    s.push_str("        let g = gb.x;
        let dlt = (vt - g * kv) * gb.y;
");
    s.push_str("        if (live && r == 0u) { gs_o[i * gs_p.nv * dv + h * dv + j] = g * sq + kq * dlt; }
");
    for u in 0..rows {
        s.push_str(&format!("        s{u} = g * s{u} + k{u} * dlt;
"));
    }
    s.push_str("    }
    if (live) {
");
    for u in 0..rows {
        s.push_str(&format!("        gs_S[sb + {u}u * dv] = s{u};
"));
    }
    s.push_str("    }
}
");
    s
}

/// Positions a `gdn_scan3_k` block stages at once.
const SCAN3_T: usize = 16;

/// `gdn_scan2_k` with each block of 16 positions staged in workgroup
/// memory: the workgroup loads the next block's q|k rows (from
/// `gdn_prep_k`), its own v columns, g, beta and k·q with coalesced vec4
/// loads while it runs the current block, and a step reads them back with
/// ten vec4 shared loads. `gdn_scan2_k` fetched every step's 32 values a
/// lane from device memory one step ahead — the prep rows of a 512-row
/// chunk (8 MB) do not stay in a 6 MB L2, and the scan ran at ~1.6 µs a
/// position whatever the work a lane did. q|k groups of 16 floats sit at a
/// stride of 20, so the eight row parts of a warp's lanes hit distinct
/// banks. Same arithmetic as `gdn_scan2_k`.
fn gdn_scan3_src(parts: usize) -> String {
    let rows = GDN_SCAN_DK / parts;
    let cols = 128 / parts;
    let dk = GDN_SCAN_DK;
    let t_blk = SCAN3_T;
    // Shared row of a position, in vec4: q|k 16 groups × 5, v cols/4, (g, beta, kq, 0).
    let qk4 = 16 * 5;
    let v4 = cols / 4;
    let st4 = qk4 + v4 + 1;
    let per = t_blk * (64 + v4); // vec4 loads of a block
    let nld = per.div_ceil(128);
    let mut s = String::new();
    s.push_str(
        r#"struct GsP {
    nv: u32, dk: u32, dv: u32, kd: u32,
    rep: u32, cdim: u32, eps: f32, kb: u32,
    stride: u32, ring_els: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read>       gs_cq   : array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> gs_S    : array<f32>;
@group(0) @binding(8) var<storage, read_write> gs_o    : array<f32>;
@group(0) @binding(9) var<uniform>             gs_p    : GsP;
@group(0) @binding(10) var<storage, read>      gs_qk   : array<vec4<f32>>;
@group(0) @binding(11) var<storage, read>      gs_kq   : array<f32>;
@group(0) @binding(12) var<storage, read>      gs_gb   : array<vec2<f32>>;
"#,
    );
    s.push_str(&format!("var<workgroup> sh: array<vec4<f32>, {}>;
", t_blk * st4));
    let red2 = |a: &str, b: &str| {
        let mut t = String::new();
        let mut m = 1;
        while m < parts {
            t.push_str(&format!(
                "            {a} = {a} + subgroupShuffleXor({a}, {m}u);
            {b} = {b} + subgroupShuffleXor({b}, {m}u);
"
            ));
            m *= 2;
        }
        t
    };
    // A block's loads into registers: vec4 index v → (position, slot).
    let load = |blk: &str, s: &mut String| {
        for l in 0..nld {
            s.push_str(&format!(
                "        {{ let vi = lid + {o}u; var x = vec4<f32>(0.0); let tp = vi / {w}u; let c = vi % {w}u; let pos = {blk} * {t_blk}u + tp; if (vi < {per}u && pos < gs_p.kb) {{ if (c < 64u) {{ x = gs_qk[(pos * nk + ko) * 64u + c]; }} else {{ x = gs_cq[(pos * gs_p.cdim + 2u * gs_p.kd + h * dv + col0) / 4u + c - 64u]; }} }} ld{l} = x; }}
",
                o = l * 128,
                w = 64 + v4
            ));
        }
        s.push_str(&format!(
            "        {{ let pos = {blk} * {t_blk}u + lid; var e = vec4<f32>(0.0); if (lid < {t_blk}u && pos < gs_p.kb) {{ let gb = gs_gb[pos * gs_p.nv + h]; e = vec4<f32>(gb.x, gb.y, gs_kq[pos * nk + ko], 0.0); }} lde = e; }}
"
        ));
    };
    s.push_str(&format!(
        "@compute @workgroup_size(128)
fn gdn_scan3_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
    let r = lid % {parts}u;
    let jl = lid / {parts}u;
    let col0 = wid.y * {cols}u;
    let j0 = col0 + jl;
"
    ));
    s.push_str(
        "    let dv = gs_p.dv;
    let nk = gs_p.kd / 128u;
    let live = wid.x < gs_p.nv && j0 < dv;
    let h = min(wid.x, gs_p.nv - 1u);
    let j = select(0u, j0, live);
    let ko = h / gs_p.rep;
",
    );
    s.push_str(&format!("    let sb = h * {dk}u * dv + r * {rows}u * dv + j;
"));
    for u in 0..rows {
        s.push_str(&format!("    var s{u} = 0.0; if (live) {{ s{u} = gs_S[sb + {u}u * dv]; }}
"));
    }
    for l in 0..nld {
        s.push_str(&format!("    var ld{l} = vec4<f32>(0.0);
"));
    }
    s.push_str("    var lde = vec4<f32>(0.0);
");
    s.push_str(&format!("    let nb = (gs_p.kb + {}u) / {t_blk}u;
", t_blk - 1));
    load("0u", &mut s);
    s.push_str("    for (var b = 0u; b < nb; b = b + 1u) {
        workgroupBarrier();
");
    // registers → shared (q|k groups at stride 5 vec4, then v, then e)
    for l in 0..nld {
        s.push_str(&format!(
            "        {{ let vi = lid + {o}u; let tp = vi / {w}u; let c = vi % {w}u; if (vi < {per}u) {{ var d = tp * {st4}u + {qk4}u + (c - 64u); if (c < 64u) {{ d = tp * {st4}u + (c / 4u) * 5u + c % 4u; }} sh[d] = ld{l}; }} }}
",
            o = l * 128,
            w = 64 + v4
        ));
    }
    s.push_str(&format!("        if (lid < {t_blk}u) {{ sh[lid * {st4}u + {}u] = lde; }}
", qk4 + v4));
    s.push_str("        workgroupBarrier();
        if (b + 1u < nb) {
");
    load("(b + 1u)", &mut s);
    s.push_str("        }
");
    s.push_str(&format!("        let tn = min({t_blk}u, gs_p.kb - b * {t_blk}u);
"));
    s.push_str("        for (var t = 0u; t < tn; t = t + 1u) {
");
    s.push_str(&format!("            let ro = t * {st4}u;
"));
    let g4 = rows / 4;
    for v in 0..g4 {
        // part r's group of 16 rows: q group r, k group 8 + r (for 128/16 = 8 groups each)
        let _ = v;
    }
    // q|k element e of part r: group (r*rows + e) / 16, offset % 16
    for u4 in 0..g4 {
        s.push_str(&format!(
            "            let qv{u4} = sh[ro + ((r * {rows}u + {e}u) / 16u) * 5u + ((r * {rows}u + {e}u) % 16u) / 4u];
            let kv{u4} = sh[ro + (8u + (r * {rows}u + {e}u) / 16u) * 5u + ((r * {rows}u + {e}u) % 16u) / 4u];
",
            e = u4 * 4
        ));
    }
    s.push_str(&format!(
        "            let vq = sh[ro + {qk4}u + jl / 4u];
            let vt = select(select(select(vq.w, vq.z, (jl & 3u) == 2u), vq.y, (jl & 3u) == 1u), vq.x, (jl & 3u) == 0u);
            let e = sh[ro + {}u];
",
        qk4 + v4
    ));
    // dots: kv = Σ s_u k_u, sq = Σ s_u q_u, four chains each
    let comp = ["x", "y", "z", "w"];
    let mut kv_terms = vec![Vec::new(); 4];
    let mut sq_terms = vec![Vec::new(); 4];
    for u in 0..rows {
        kv_terms[u % 4].push(format!("s{u} * kv{}.{}", u / 4, comp[u % 4]));
        sq_terms[u % 4].push(format!("s{u} * qv{}.{}", u / 4, comp[u % 4]));
    }
    for c in 0..4 {
        s.push_str(&format!("            let kd_{c} = {};
            let sd_{c} = {};
", kv_terms[c].join(" + "), sq_terms[c].join(" + ")));
    }
    s.push_str("            var kvs = (kd_0 + kd_1) + (kd_2 + kd_3);
            var sqs = (sd_0 + sd_1) + (sd_2 + sd_3);
");
    s.push_str(&red2("kvs", "sqs"));
    s.push_str("            let g = e.x;
            let dlt = (vt - g * kvs) * e.y;
");
    s.push_str(&format!(
        "            if (live && r == 0u) {{ gs_o[(b * {t_blk}u + t) * gs_p.nv * dv + h * dv + j] = g * sqs + e.z * dlt; }}
"
    ));
    for u in 0..rows {
        s.push_str(&format!("            s{u} = g * s{u} + kv{}.{} * dlt;
", u / 4, comp[u % 4]));
    }
    s.push_str("        }
    }
    if (live) {
");
    for u in 0..rows {
        s.push_str(&format!("        gs_S[sb + {u}u * dv] = s{u};
"));
    }
    s.push_str("    }
}
");
    s
}

/// Lanes a state column spans in the prompt scans (`CMF_GDN_SCAN_P`: 2, 4
/// or 8, default 4; `CMF_GDN_SCAN=0` turns the scans off).
fn gdn_scan_parts() -> usize {
    match std::env::var("CMF_GDN_SCAN").as_deref() {
        Ok("0") => 0,
        // RTX A5000, Qwen3.5-4B, 512-row chunks, the block-staged scan:
        // 12.4 ms of the chunk's GDN stage at 4 lanes a column, 12.9 at 8
        // (the one-step-prefetch scans: 21.4 / 22.0, 58 at 2).
        _ => match std::env::var("CMF_GDN_SCAN_P").as_deref() {
            Ok("8") => 8,
            Ok("2") => 2,
            _ => 4,
        },
    }
}

fn build_gdn(c: &Ctx) -> Option<GdnSg> {
    if std::env::var("CMF_GDN_SG").as_deref() == Ok("0") || !sg32(c) {
        return None;
    }
    let par = compile(c, "cmf-gdn-sg", GDN_SG_WGSL, "gdn_step_par_sg")?;
    let park = compile(c, "cmf-gdnk-sg", GDNK_SG_WGSL, "gdn_step_par_k_sg")?;
    let parts = gdn_scan_parts();
    let scan = (parts > 0)
        .then(|| compile(c, "cmf-gdn-scan", &gdn_scan_src(parts), "gdn_scan_k"))
        .flatten()
        .map(|p| (p, 128 / parts));
    let scan2 = (parts > 0 && std::env::var("CMF_GDN_SCAN2").as_deref() != Ok("0"))
        .then(|| {
            let prep = compile(c, "cmf-gdn-prep", GDN_PREP_WGSL, "gdn_prep_k")?;
            // The block-staged scan by default; `CMF_GDN_SCAN3=0` keeps
            // the one-step prefetch.
            let sc = if std::env::var("CMF_GDN_SCAN3").as_deref() != Ok("0") {
                compile(c, "cmf-gdn-scan3", &gdn_scan3_src(parts), "gdn_scan3_k")?
            } else {
                compile(c, "cmf-gdn-scan2", &gdn_scan2_src(parts), "gdn_scan2_k")?
            };
            Some((prep, sc, 128 / parts))
        })
        .flatten();
    let par_k = (std::env::var("CMF_GDN_PAR").as_deref() != Ok("0"))
        .then(|| {
            Some(GdnPar {
                conv: compile(c, "cmf-gdn-par", GDN_PAR_WGSL, "gdn_conv_par")?,
                ring: compile(c, "cmf-gdn-par", GDN_PAR_WGSL, "gdn_ring_next")?,
                norm: compile(c, "cmf-gdn-par", GDN_PAR_WGSL, "gdn_norm_rows")?,
            })
        })
        .flatten();
    Some(GdnSg {
        par_l: par.get_bind_group_layout(0),
        par,
        park_l: park.get_bind_group_layout(0),
        park,
        scan,
        scan2,
        par_k,
    })
}

pub(crate) fn gdn(c: &Ctx) -> Option<&GdnSg> {
    c.gdn_sg_pipes.get_or_init(|| build_gdn(c)).as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prompt scan parses and validates for every lane split it may be
    /// generated with (naga, subgroups on), with no GPU.
    #[test]
    fn gdn_scan_sources_validate() {
        for parts in [2, 4, 8] {
            let s = gdn_scan_src(parts);
            let module = wgpu::naga::front::wgsl::parse_str(&s)
                .unwrap_or_else(|e| panic!("parts {parts}: {e}"));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("parts {parts}: {e:?}"));
            assert!(s.contains(&format!("var s{} = gs_S", GDN_SCAN_DK / parts - 1)));
        }
        for parts in [4, 8] {
            for src in [gdn_scan2_src(parts), gdn_scan3_src(parts), GDN_PREP_WGSL.to_string()] {
                let module = wgpu::naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{e}"));
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|e| panic!("{e:?}"));
            }
        }
        let module = wgpu::naga::front::wgsl::parse_str(GDN_PAR_WGSL).expect("gdn par parses");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("gdn par validates");
    }
}
