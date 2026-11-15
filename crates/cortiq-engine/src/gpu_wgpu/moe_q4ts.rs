//! q4_tiled MoE experts on the wgpu graphs in a split layout ("q4ts").
//!
//! A q4_tiled tile is 18 bytes: an f16 scale, then 16 bytes of nibbles. The
//! stride is two-aligned, so the generic `moe_gate_up` / `moe_down` pair
//! assembles every weight word from two u16 halves, puts one output row in a
//! workgroup and reads the activations as scalars; the batch graph walks the
//! prompt one token at a time through the same pair. On an RTX A5000 that
//! was 0.46 ms a Kimi-Linear-48B layer for 36 MB of expert weights (~80
//! GB/s), 12.5 of the 17-21 ms token, and a 4k prompt at 73 tok/s.
//!
//! Here each packed expert matrix is rewritten once on the device, byte for
//! byte, into its nibbles (`[rows][gpr][16 B]`, 16-byte aligned) followed by
//! the f16 scales (`[rows][gpr]`) — the same size, so the expert stride does
//! not change — and served by the `moe_r4` kernel shapes: four rows a
//! workgroup with vec4 activations and `vec4<u32>` nibble words for decode,
//! and the expert-grouped prefill (`moeg_group` / `moeg_sum` of `moe_r4`
//! with gate/up and down kernels of this layout). The scale of a tile is the
//! file's f16, the dot product the tile's nibbles − 8 against x, as the
//! one-row kernels compute it (sums in another order).

use super::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const COMMON: &str = r#"
fn m4_n4(w: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(f32((w >> sh) & 0xFu), f32((w >> (sh + 4u)) & 0xFu),
                     f32((w >> (sh + 8u)) & 0xFu), f32((w >> (sh + 12u)) & 0xFu))
        - vec4<f32>(8.0);
}
fn m4_d8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(m4_n4(w, 0u), a) + dot(m4_n4(w, 16u), b);
}
fn m4_d32(w: vec4<u32>, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>,
          x4: vec4<f32>, x5: vec4<f32>, x6: vec4<f32>, x7: vec4<f32>) -> f32 {
    return (m4_d8(w.x, x0, x1) + m4_d8(w.y, x2, x3)) + (m4_d8(w.z, x4, x5) + m4_d8(w.w, x6, x7));
}
"#;

/// The repack and the decode pair.
const DECODE: &str = r#"
// ── the one-time repack: two tiles a thread, 18-byte tiles in, nibbles then
// scales out (per expert, at the expert's own offset) ──
struct RpP { tpe: u32, n_exp: u32, row_pairs: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       rp_src : array<u32>;
@group(0) @binding(1) var<storage, read_write> rp_dst : array<u32>;
@group(0) @binding(2) var<uniform>             rp_p   : RpP;
fn rp_u16(o: u32) -> u32 { return (rp_src[o >> 1u] >> ((o & 1u) * 16u)) & 0xFFFFu; }
@compute @workgroup_size(256)
fn q4ts_repack(@builtin(global_invocation_id) gid: vec3<u32>) {
    // pair index over the whole buffer (tpe is even)
    let p = gid.y * rp_p.row_pairs + gid.x;
    let half = rp_p.tpe >> 1u;
    if (p >= half * rp_p.n_exp) { return; }
    let e = p / half;
    let t = (p - e * half) * 2u;
    let eb = e * ((rp_p.tpe * 9u) >> 1u);
    for (var i = 0u; i < 2u; i = i + 1u) {
        let s = (e * rp_p.tpe + t + i) * 9u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            rp_dst[eb + (t + i) * 4u + k] = rp_u16(s + 1u + 2u * k) | (rp_u16(s + 2u + 2u * k) << 16u);
        }
    }
    let s0 = (e * rp_p.tpe + t) * 9u;
    rp_dst[eb + rp_p.tpe * 4u + (t >> 1u)] = rp_u16(s0) | (rp_u16(s0 + 9u) << 16u);
}

// ── gate+up+SiLU, four rows a workgroup (`moe4_gu`'s bindings) ──
struct GuP { gpr: u32, inter: u32, slots: u32, mat16: u32, lim: f32, fs: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       gu_gw  : array<u32>;
@group(0) @binding(1) var<storage, read>       gu_uw  : array<u32>;
@group(0) @binding(2) var<storage, read>       gu_x   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       gu_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> gu_act : array<f32>;
@group(0) @binding(5) var<uniform>             gu_p   : GuP;
@group(0) @binding(6) var<storage, read>       gu_gw4 : array<vec4<u32>>;
@group(0) @binding(7) var<storage, read>       gu_uw4 : array<vec4<u32>>;
var<workgroup> gu_pg: array<vec4<f32>, 64>;
var<workgroup> gu_pu: array<vec4<f32>, 64>;
fn gu_gs(o: u32) -> f32 { return unpack2x16float(gu_gw[o >> 1u] >> ((o & 1u) * 16u)).x; }
fn gu_us(o: u32) -> f32 { return unpack2x16float(gu_uw[o >> 1u] >> ((o & 1u) * 16u)).x; }

@compute @workgroup_size(64)
fn q4ts_gu(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let slot = wid.y;
    let gpr = gu_p.gpr;
    let rows = gu_p.inter;
    let base16 = gu_sel[slot] * gu_p.mat16;
    // u16 index of the scale of (row0, group 0); a row is gpr scales on
    let sp = base16 + rows * gpr * 8u + row0 * gpr;
    // 16-byte word of (row0, group 0); the next row is gpr words on
    let nb = (base16 >> 3u) + row0 * gpr;
    var ag = vec4<f32>(0.0);
    var au = vec4<f32>(0.0);
    for (var g = lid; g < gpr; g = g + 64u) {
        let xo = g * 8u;
        let x0 = gu_x[xo];      let x1 = gu_x[xo + 1u];
        let x2 = gu_x[xo + 2u]; let x3 = gu_x[xo + 3u];
        let x4 = gu_x[xo + 4u]; let x5 = gu_x[xo + 5u];
        let x6 = gu_x[xo + 6u]; let x7 = gu_x[xo + 7u];
        let so = sp + g;
        let sg = vec4<f32>(gu_gs(so), gu_gs(so + gpr), gu_gs(so + 2u * gpr), gu_gs(so + 3u * gpr));
        let su = vec4<f32>(gu_us(so), gu_us(so + gpr), gu_us(so + 2u * gpr), gu_us(so + 3u * gpr));
        let w = nb + g;
        let dg = vec4<f32>(m4_d32(gu_gw4[w], x0, x1, x2, x3, x4, x5, x6, x7),
                           m4_d32(gu_gw4[w + gpr], x0, x1, x2, x3, x4, x5, x6, x7),
                           m4_d32(gu_gw4[w + 2u * gpr], x0, x1, x2, x3, x4, x5, x6, x7),
                           m4_d32(gu_gw4[w + 3u * gpr], x0, x1, x2, x3, x4, x5, x6, x7));
        let du = vec4<f32>(m4_d32(gu_uw4[w], x0, x1, x2, x3, x4, x5, x6, x7),
                           m4_d32(gu_uw4[w + gpr], x0, x1, x2, x3, x4, x5, x6, x7),
                           m4_d32(gu_uw4[w + 2u * gpr], x0, x1, x2, x3, x4, x5, x6, x7),
                           m4_d32(gu_uw4[w + 3u * gpr], x0, x1, x2, x3, x4, x5, x6, x7));
        ag = ag + sg * dg;
        au = au + su * du;
    }
    gu_pg[lid] = ag;
    gu_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            gu_pg[lid] = gu_pg[lid] + gu_pg[lid + stride];
            gu_pu[lid] = gu_pu[lid] + gu_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid < 4u) {
        var gg = gu_pg[0][lid];
        var uu = gu_pu[0][lid];
        if (gu_p.lim > 0.0) {
            uu = clamp(uu, -gu_p.lim, gu_p.lim);
            gg = min(gg, gu_p.lim);
        }
        gu_act[slot * rows + row0 + lid] = (gg / (1.0 + exp(-gg))) * uu;
    }
}

// ── weighted down projection, four rows a workgroup (`moe4_dn`'s bindings) ──
struct DnP { gpr: u32, hidden: u32, slots: u32, mat16: u32 };
@group(0) @binding(8)  var<storage, read>       dn_w   : array<u32>;
@group(0) @binding(9)  var<storage, read>       dn_w4  : array<vec4<u32>>;
@group(0) @binding(10) var<storage, read>       dn_act : array<vec4<f32>>;
@group(0) @binding(11) var<storage, read>       dn_sel : array<u32>;
@group(0) @binding(12) var<storage, read>       dn_wt  : array<f32>;
@group(0) @binding(13) var<storage, read_write> dn_y   : array<f32>;
@group(0) @binding(14) var<uniform>             dn_p   : DnP;
var<workgroup> dn_pt: array<vec4<f32>, 64>;
var<workgroup> dn_ss: array<u32, 32>;
var<workgroup> dn_sw: array<f32, 32>;
fn dn_s(o: u32) -> f32 { return unpack2x16float(dn_w[o >> 1u] >> ((o & 1u) * 16u)).x; }

@compute @workgroup_size(64)
fn q4ts_dn(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    if (lid < dn_p.slots) {
        dn_ss[lid] = dn_sel[lid];
        dn_sw[lid] = dn_wt[lid];
    }
    workgroupBarrier();
    let row0 = wid.x * 4u;
    let gpr = dn_p.gpr;
    let rows = dn_p.hidden;
    let total = dn_p.slots * gpr;
    var acc = vec4<f32>(0.0);
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i - slot * gpr;
        let base16 = dn_ss[slot] * dn_p.mat16;
        let so = base16 + rows * gpr * 8u + row0 * gpr + g;
        let scale = vec4<f32>(dn_s(so), dn_s(so + gpr), dn_s(so + 2u * gpr), dn_s(so + 3u * gpr));
        let xo = (slot * gpr + g) * 8u;
        let a0 = dn_act[xo];      let a1 = dn_act[xo + 1u];
        let a2 = dn_act[xo + 2u]; let a3 = dn_act[xo + 3u];
        let a4 = dn_act[xo + 4u]; let a5 = dn_act[xo + 5u];
        let a6 = dn_act[xo + 6u]; let a7 = dn_act[xo + 7u];
        let w = (base16 >> 3u) + row0 * gpr + g;
        let d = vec4<f32>(m4_d32(dn_w4[w], a0, a1, a2, a3, a4, a5, a6, a7),
                          m4_d32(dn_w4[w + gpr], a0, a1, a2, a3, a4, a5, a6, a7),
                          m4_d32(dn_w4[w + 2u * gpr], a0, a1, a2, a3, a4, a5, a6, a7),
                          m4_d32(dn_w4[w + 3u * gpr], a0, a1, a2, a3, a4, a5, a6, a7));
        acc = acc + dn_sw[slot] * (scale * d);
    }
    dn_pt[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { dn_pt[lid] = dn_pt[lid] + dn_pt[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid < 4u) {
        dn_y[row0 + lid] = dn_pt[0][lid];
    }
}
"#;

/// The expert-grouped prefill's gate/up and down for this layout (the
/// bindings of `moe_r4`'s `moeg_gu` / `moeg_dn`; its `moeg_group` and
/// `moeg_sum` run unchanged around them).
const GROUPED_T: &str = r#"
struct GgP { gpr: u32, inter: u32, slots: u32, mat16: u32 };
@group(0) @binding(0) var<storage, read>       gg_gw  : array<u32>;
@group(0) @binding(1) var<storage, read>       gg_uw  : array<u32>;
@group(0) @binding(2) var<storage, read>       gg_gw4 : array<vec4<u32>>;
@group(0) @binding(3) var<storage, read>       gg_uw4 : array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       gg_x   : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read>       gg_off : array<u32>;
@group(0) @binding(6) var<storage, read>       gg_ent : array<u32>;
@group(0) @binding(7) var<storage, read_write> gg_act : array<f32>;
@group(0) @binding(8) var<uniform>             gg_p   : GgP;
fn gg_gs(o: u32) -> f32 { return unpack2x16float(gg_gw[o >> 1u] >> ((o & 1u) * 16u)).x; }
fn gg_us(o: u32) -> f32 { return unpack2x16float(gg_uw[o >> 1u] >> ((o & 1u) * 16u)).x; }
@compute @workgroup_size(64)
fn q4ts_ggu(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let ex = wid.y;
    let beg = gg_off[ex] + wid.z * {NT}u;
    let end = min(gg_off[ex + 1u], beg + {NT}u);
    if (beg >= end) { return; }
    let nt = end - beg;
    let gpr = gg_p.gpr;
    let rows = gg_p.inter;
    let row = wid.x * 64u + lid;
    if (row >= rows) { return; }
    let base16 = ex * gg_p.mat16;
    let sp = base16 + rows * gpr * 8u + row * gpr;
    let wb = (base16 >> 3u) + row * gpr;
{GU_INIT}
    for (var g = 0u; g < gpr; g = g + 1u) {
        let sg = gg_gs(sp + g);
        let su = gg_us(sp + g);
        let wg = gg_gw4[wb + g];
        let wu = gg_uw4[wb + g];
        let gx = g * 8u;
{GU_BODY}
    }
{GU_OUT}
}

struct GdP { gpr: u32, hidden: u32, slots: u32, mat16: u32 };
@group(0) @binding(0) var<storage, read>       gd_w   : array<u32>;
@group(0) @binding(1) var<storage, read>       gd_w4  : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       gd_act : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       gd_off : array<u32>;
@group(0) @binding(4) var<storage, read>       gd_ent : array<u32>;
@group(0) @binding(5) var<storage, read_write> gd_y   : array<f32>;
@group(0) @binding(6) var<uniform>             gd_p   : GdP;
fn gd_s(o: u32) -> f32 { return unpack2x16float(gd_w[o >> 1u] >> ((o & 1u) * 16u)).x; }
@compute @workgroup_size(64)
fn q4ts_gdn(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let ex = wid.y;
    let beg = gd_off[ex] + wid.z * {NT}u;
    let end = min(gd_off[ex + 1u], beg + {NT}u);
    if (beg >= end) { return; }
    let nt = end - beg;
    let gpr = gd_p.gpr;
    let rows = gd_p.hidden;
    let row = wid.x * 64u + lid;
    if (row >= rows) { return; }
    let base16 = ex * gd_p.mat16;
    let sp = base16 + rows * gpr * 8u + row * gpr;
    let wb = (base16 >> 3u) + row * gpr;
{DN_INIT}
    for (var g = 0u; g < gpr; g = g + 1u) {
        let sc = gd_s(sp + g);
        let w = gd_w4[wb + g];
        let gx = g * 8u;
{DN_BODY}
    }
{DN_OUT}
}
"#;

/// Entries of a group a gate/up or down workgroup walks (`moe_r4`'s).
const NT: usize = super::moe_r4::MOEG_TOKENS;

fn grouped_src() -> String {
    let (mut gi, mut gb, mut go, mut di, mut db, mut dout) =
        (String::new(), String::new(), String::new(), String::new(), String::new(), String::new());
    for j in 0..NT {
        gi.push_str(&format!(
            "    let e{j} = gg_ent[min(beg + {j}u, end - 1u)];\n    let t{j} = (e{j} / gg_p.slots) * (gpr * 8u);\n    var ag{j} = 0.0; var au{j} = 0.0;\n"
        ));
        gb.push_str(&format!(
            "        {{ let xo = t{j} + gx;\n          let x0 = gg_x[xo]; let x1 = gg_x[xo + 1u]; let x2 = gg_x[xo + 2u]; let x3 = gg_x[xo + 3u];\n          let x4 = gg_x[xo + 4u]; let x5 = gg_x[xo + 5u]; let x6 = gg_x[xo + 6u]; let x7 = gg_x[xo + 7u];\n          ag{j} = ag{j} + sg * m4_d32(wg, x0, x1, x2, x3, x4, x5, x6, x7);\n          au{j} = au{j} + su * m4_d32(wu, x0, x1, x2, x3, x4, x5, x6, x7); }}\n"
        ));
        go.push_str(&format!(
            "    if ({j}u < nt) {{ gg_act[e{j} * rows + row] = (ag{j} / (1.0 + exp(-ag{j}))) * au{j}; }}\n"
        ));
        di.push_str(&format!(
            "    let e{j} = gd_ent[min(beg + {j}u, end - 1u)];\n    let t{j} = e{j} * (gpr * 8u);\n    var ac{j} = 0.0;\n"
        ));
        db.push_str(&format!(
            "        {{ let ao = t{j} + gx;\n          let a0 = gd_act[ao]; let a1 = gd_act[ao + 1u]; let a2 = gd_act[ao + 2u]; let a3 = gd_act[ao + 3u];\n          let a4 = gd_act[ao + 4u]; let a5 = gd_act[ao + 5u]; let a6 = gd_act[ao + 6u]; let a7 = gd_act[ao + 7u];\n          ac{j} = ac{j} + sc * m4_d32(w, a0, a1, a2, a3, a4, a5, a6, a7); }}\n"
        ));
        dout.push_str(&format!("    if ({j}u < nt) {{ gd_y[e{j} * rows + row] = ac{j}; }}\n"));
    }
    let mut s = String::from(COMMON);
    s.push_str(
        &GROUPED_T
            .replace("{GU_INIT}", &gi)
            .replace("{GU_BODY}", &gb)
            .replace("{GU_OUT}", &go)
            .replace("{DN_INIT}", &di)
            .replace("{DN_BODY}", &db)
            .replace("{DN_OUT}", &dout)
            .replace("{NT}", &NT.to_string()),
    );
    s
}

fn decode_src() -> String {
    format!("{COMMON}{DECODE}")
}

pub(crate) struct Pipe {
    pub(crate) p: wgpu::ComputePipeline,
    pub(crate) l: wgpu::BindGroupLayout,
}

pub(crate) struct Pipes {
    repack: Pipe,
    pub(crate) gu: Pipe,
    pub(crate) dn: Pipe,
    pub(crate) ggu: Pipe,
    pub(crate) gdn: Pipe,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn build_module(c: &Ctx, label: &str, src: String, entries: &[&str]) -> Option<Vec<Pipe>> {
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let out: Vec<Pipe> = entries
        .iter()
        .map(|e| {
            let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: None,
                module: &m,
                entry_point: Some(e),
                compilation_options: crate::gpu_wgpu::pco(),
                cache: c.pipeline_cache.as_ref(),
            });
            let l = p.get_bind_group_layout(0);
            Pipe { p, l }
        })
        .collect();
    if let Some(e) = pollster::block_on(sc.pop()) {
        tracing::warn!("{label} kernels rejected ({e}): q4_tiled experts keep the one-row kernels");
        return None;
    }
    Some(out)
}

pub(crate) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let mut d = build_module(c, "cmf-moe-q4ts", decode_src(), &["q4ts_repack", "q4ts_gu", "q4ts_dn"])?.into_iter();
            let mut g = build_module(c, "cmf-moe-q4ts-g", grouped_src(), &["q4ts_ggu", "q4ts_gdn"])?.into_iter();
            Some(Pipes {
                repack: d.next()?,
                gu: d.next()?,
                dn: d.next()?,
                ggu: g.next()?,
                gdn: g.next()?,
            })
        })
        .as_ref()
}

/// Do q4_tiled MoE layers of this geometry take the split layout? Needs the
/// kernels (and `moe_r4`'s grouped prefill, whose group and sum kernels the
/// batch graph runs around these), 32-wide groups, rows in fours and every
/// expert a whole number of 16-byte words; `CMF_MOE_Q4TS=0` keeps the
/// one-row kernels and the original layout.
pub(crate) fn on(c: &Ctx, hidden: usize, inter: usize) -> bool {
    if std::env::var("CMF_MOE_Q4TS").as_deref() == Ok("0") {
        return false;
    }
    let tiles = |rows: usize, cols: usize| rows * (cols / 32);
    hidden % 32 == 0
        && inter % 32 == 0
        && hidden % 4 == 0
        && inter % 4 == 0
        && tiles(inter, hidden) % 8 == 0
        && tiles(hidden, inter) % 8 == 0
        && super::moe_r4::prefill(c).is_some()
        && pipes(c).is_some()
}

type Bank = (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer);
static BANKS: Mutex<Option<HashMap<(usize, usize), Bank>>> = Mutex::new(None);

/// One repacked matrix stack: `n_exp` experts of `rows × cols`.
fn repack(c: &Ctx, src: &wgpu::Buffer, n_exp: usize, rows: usize, cols: usize) -> Option<wgpu::Buffer> {
    let p = pipes(c)?;
    let tpe = rows * (cols / 32);
    let bytes = (n_exp * tpe * 18) as u64;
    if src.size() < bytes || tpe % 2 != 0 {
        return None;
    }
    let dst = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("moe-q4ts"),
        size: src.size(),
        usage: src.usage(),
        mapped_at_creation: false,
    });
    let pairs = n_exp * tpe / 2;
    let row_pairs = 256 * 16384usize;
    let u = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("moe-q4ts-u"),
        contents: bytemuck::cast_slice(&[tpe as u32, n_exp as u32, row_pairs as u32, 0u32]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("moe-q4ts-repack"),
        layout: &p.repack.l,
        entries: &[bind_buf(0, src), bind_buf(1, &dst), bind_buf(2, &u)],
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("moe-q4ts-repack") });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&p.repack.p);
        pass.set_bind_group(0, &bg, &[]);
        let y = pairs.div_ceil(row_pairs);
        let x = if y > 1 { row_pairs / 256 } else { pairs.div_ceil(256) };
        pass.dispatch_workgroups(x as u32, y as u32, 1);
    }
    submit(c, enc.finish());
    Some(dst)
}

/// A q4_tiled MoE layer's expert stacks in the split layout, uploaded (the
/// ordinary `moe_expert_bufs`) and repacked once; the original stacks are
/// dropped from the upload cache, their budget kept by these.
pub(crate) fn banks(
    c: &Ctx,
    model: &Arc<CmfModel>,
    experts: &[(usize, usize, usize)],
    inter: usize,
    hidden: usize,
) -> Option<Bank> {
    let key = (model.uid() as usize, experts.first()?.0);
    if let Some(b) = BANKS.lock().unwrap().get_or_insert_with(HashMap::new).get(&key) {
        return Some(b.clone());
    }
    let (g, u, d) = moe_expert_bufs(c, model, experts, inter, hidden, false, false, false)?;
    let n = experts.len();
    let out = (repack(c, &g, n, inter, hidden)?, repack(c, &u, n, inter, hidden)?, repack(c, &d, n, hidden, inter)?);
    c.moe_expw.lock().unwrap().remove(&key);
    BANKS.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, out.clone());
    Some(out)
}

/// Is this layer's bank (model uid, first expert) on the card already.
pub(crate) fn resident(key: (usize, usize)) -> bool {
    BANKS.lock().unwrap().as_ref().is_some_and(|m| m.contains_key(&key))
}

/// Drop a model's banks (`release_idle_model_buffers`, beside the upload
/// cache): their bytes leave the resident count, which is returned.
pub(crate) fn release(c: &Ctx, uid: usize) -> u64 {
    let mut g = BANKS.lock().unwrap();
    let Some(m) = g.as_mut() else { return 0 };
    let mut bytes = 0u64;
    m.retain(|k, (a, b, d)| {
        if k.0 != uid {
            return true;
        }
        bytes = bytes.saturating_add(a.size()).saturating_add(b.size()).saturating_add(d.size());
        false
    });
    resident_sub(c, bytes);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn f(&mut self) -> f32 {
            (self.u() as f64 / (1u64 << 31) as f64) as f32 * 2.0 - 1.0
        }
    }

    /// `n` q4_tiled matrices `rows × cols`: (bytes, dequantized f32).
    fn q4t(r: &mut Rng, n: usize, rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        let gpr = cols / 32;
        let mut b = Vec::with_capacity(n * rows * gpr * 18);
        let mut w = vec![0f32; n * rows * cols];
        for e in 0..n {
            for row in 0..rows {
                for g in 0..gpr {
                    let bits = cortiq_core::quant::f32_to_f16(0.01 + 0.05 * r.f().abs());
                    let sc = cortiq_core::quant::f16_to_f32(bits);
                    b.extend_from_slice(&bits.to_le_bytes());
                    for byte in 0..16 {
                        let v = (r.u() & 0xFF) as u8;
                        b.push(v);
                        let base = (e * rows + row) * cols + g * 32 + 2 * byte;
                        w[base] = sc * ((v & 0xF) as f32 - 8.0);
                        w[base + 1] = sc * ((v >> 4) as f32 - 8.0);
                    }
                }
            }
        }
        (b, w)
    }

    fn upload(c: &Ctx, bytes: &[u8]) -> wgpu::Buffer {
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q4ts-test"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn download(c: &Ctx, b: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let st = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(b, 0, &st, 0, (n * 4) as u64);
        c.queue.submit([enc.finish()]);
        let sl = st.slice(..);
        sl.map_async(wgpu::MapMode::Read, |_| {});
        c.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let v = bytemuck::cast_slice::<u8, f32>(&sl.get_mapped_range().unwrap()).to_vec();
        st.unmap();
        v
    }

    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// The repack, the decode pair and the grouped pair against the
    /// dequantized experts (one-row semantics: silu(g)·u, Σ slot w·down).
    #[test]
    fn q4ts_kernels_match_host() {
        let Some(c) = ctx().or_else(trainer_ctx) else {
            eprintln!("q4ts_kernels_match_host: no wgpu device, skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("q4ts_kernels_match_host: kernels unavailable, skipped");
            return;
        };
        let (n, inter, hidden) = (3usize, 64usize, 128usize);
        let mut r = Rng(5);
        let (gb, gw) = q4t(&mut r, n, inter, hidden);
        let (ub, uw) = q4t(&mut r, n, inter, hidden);
        let (db, dw) = q4t(&mut r, n, hidden, inter);
        let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
        let sel = [2u32, 0u32];
        let wt = [0.6f32, 0.4f32];
        let slots = sel.len();
        // host
        let mut act = vec![0f64; slots * inter];
        for (s, &e) in sel.iter().enumerate() {
            for row in 0..inter {
                let (mut g, mut u) = (0f64, 0f64);
                for i in 0..hidden {
                    let o = (e as usize * inter + row) * hidden + i;
                    g += gw[o] as f64 * x[i] as f64;
                    u += uw[o] as f64 * x[i] as f64;
                }
                act[s * inter + row] = silu(g) * u;
            }
        }
        let mut y = vec![0f64; hidden];
        for (s, &e) in sel.iter().enumerate() {
            for row in 0..hidden {
                let mut d = 0f64;
                for i in 0..inter {
                    d += dw[(e as usize * hidden + row) * inter + i] as f64 * act[s * inter + i];
                }
                y[row] += wt[s] as f64 * d;
            }
        }
        // device: repack, then the decode pair
        let (g0, u0, d0) = (upload(c, &gb), upload(c, &ub), upload(c, &db));
        let g1 = repack(c, &g0, n, inter, hidden).unwrap();
        let u1 = repack(c, &u0, n, inter, hidden).unwrap();
        let d1 = repack(c, &d0, n, hidden, inter).unwrap();
        let mat16 = |rows: usize, cols: usize| (rows * (cols / 32) * 9) as u32;
        let xb = upload(c, bytemuck::cast_slice(&x));
        let selb = upload(c, bytemuck::cast_slice(&sel));
        let wtb = upload(c, bytemuck::cast_slice(&wt));
        let actb = upload(c, &vec![0u8; slots * inter * 4]);
        let yb = upload(c, &vec![0u8; hidden * 4]);
        let u32s = |v: &[u32]| upload(c, bytemuck::cast_slice(v));
        let gu_u = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[(hidden / 32) as u32, inter as u32, slots as u32, mat16(inter, hidden), 0, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dn_u = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[(inter / 32) as u32, hidden as u32, slots as u32, mat16(hidden, inter)]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bgu = bind_pairs(c, &p.gu.l, &[(0, &g1), (1, &u1), (2, &xb), (3, &selb), (4, &actb), (5, &gu_u), (6, &g1), (7, &u1)]);
        let bdn = bind_pairs(c, &p.dn.l, &[(8, &d1), (9, &d1), (10, &actb), (11, &selb), (12, &wtb), (13, &yb), (14, &dn_u)]);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&p.gu.p);
            pass.set_bind_group(0, &bgu, &[]);
            pass.dispatch_workgroups((inter / 4) as u32, slots as u32, 1);
            pass.set_pipeline(&p.dn.p);
            pass.set_bind_group(0, &bdn, &[]);
            pass.dispatch_workgroups((hidden / 4) as u32, 1, 1);
        }
        c.queue.submit([enc.finish()]);
        let ag = download(c, &actb, slots * inter);
        let yg = download(c, &yb, hidden);
        let err = |a: &[f32], b: &[f64]| a.iter().zip(b).map(|(x, y)| (*x as f64 - y).abs() / (1.0 + y.abs())).fold(0f64, f64::max);
        let (ea, ey) = (err(&ag, &act), err(&yg, &y));
        // grouped: two tokens (the same x twice), entries t·slots + s, bucketed by expert
        let k = 2usize;
        let x2: Vec<f32> = x.iter().chain(x.iter()).copied().collect();
        let ents: Vec<u32> = (0..(k * slots) as u32).collect();
        let mut by: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &e in &ents {
            by[sel[e as usize % slots] as usize].push(e);
        }
        let mut off = vec![0u32];
        let mut gent = Vec::new();
        for b in &by {
            gent.extend_from_slice(b);
            off.push(gent.len() as u32);
        }
        let x2b = upload(c, bytemuck::cast_slice(&x2));
        let (offb, entb) = (u32s(&off), u32s(&gent));
        let act2 = upload(c, &vec![0u8; k * slots * inter * 4]);
        let ypart = upload(c, &vec![0u8; k * slots * hidden * 4]);
        let ggu_u = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[(hidden / 32) as u32, inter as u32, slots as u32, mat16(inter, hidden), 0, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bggu = bind_pairs(c, &p.ggu.l, &[(0, &g1), (1, &u1), (2, &g1), (3, &u1), (4, &x2b), (5, &offb), (6, &entb), (7, &act2), (8, &ggu_u)]);
        let bgdn = bind_pairs(c, &p.gdn.l, &[(0, &d1), (1, &d1), (2, &act2), (3, &offb), (4, &entb), (5, &ypart), (6, &dn_u)]);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&p.ggu.p);
            pass.set_bind_group(0, &bggu, &[]);
            pass.dispatch_workgroups((inter as u32).div_ceil(64), n as u32, 1);
            pass.set_pipeline(&p.gdn.p);
            pass.set_bind_group(0, &bgdn, &[]);
            pass.dispatch_workgroups((hidden as u32).div_ceil(64), n as u32, 1);
        }
        c.queue.submit([enc.finish()]);
        let a2 = download(c, &act2, k * slots * inter);
        let y2 = download(c, &ypart, k * slots * hidden);
        let mut eg = 0f64;
        for e in 0..k * slots {
            let s = e % slots;
            eg = eg.max(err(&a2[e * inter..(e + 1) * inter], &act[s * inter..(s + 1) * inter]));
            let ex = sel[s] as usize;
            let want: Vec<f64> = (0..hidden)
                .map(|row| (0..inter).map(|i| dw[(ex * hidden + row) * inter + i] as f64 * act[s * inter + i]).sum())
                .collect();
            eg = eg.max(err(&y2[e * hidden..(e + 1) * hidden], &want));
        }
        eprintln!("q4ts kernels vs host: decode act {ea:.2e}, y {ey:.2e}; grouped {eg:.2e}");
        assert!(ea < 1e-4 && ey < 1e-4 && eg < 1e-4, "act {ea} y {ey} grouped {eg}");
    }

    fn validate(src: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("q4ts WGSL must parse: {}", e.emit_to_string(src)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("q4ts WGSL must validate: {}", e.emit_to_string(src)));
    }

    #[test]
    fn q4ts_shaders_validate() {
        validate(&super::decode_src());
        validate(&super::grouped_src());
    }
}
