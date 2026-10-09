//! Row-blocked decode kernels for the generic q4tp MoE block of the
//! whole-token graph (Mellum2.1: 64 experts, top-8, no shared expert).
//!
//! The generic `moe_gate_up_q4tp` / `moe_down_q4tp` kernels put ONE output
//! row in a workgroup, read the activation vector as scalars and assemble
//! every weight word from two u16 halves. On an RTX 3090 that measured
//! 143 µs + 49 µs a layer for 17.2 + 8.7 MB of expert weights (120 and
//! 177 GB/s, an eighth of the bus). These are the `q4_gu_q4tp4` /
//! `q4_dn_q4tp4` shapes of the Qwen3.8-Flash-Next resident path on the
//! graph's ordinary per-layer expert buffers: four rows a workgroup, the
//! lane's activation group loaded once as vec4s for all four rows (and for
//! gate and up), and each row's 16-byte nibble group read as one
//! `vec4<u32>`. Same select output (`sel`, `wt`), same `act` layout, same
//! uniforms as the generic pair, so the host swaps pipelines and bind
//! groups only.

use super::*;

pub(crate) const MOE4_WGSL: &str = r#"
// ── gate+up+SiLU, four rows a workgroup ──
struct GuP { gpr: u32, inter: u32, slots: u32, mat16: u32, lim: f32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       gu_gw  : array<u32>;
@group(0) @binding(1) var<storage, read>       gu_uw  : array<u32>;
@group(0) @binding(2) var<storage, read>       gu_x   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       gu_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> gu_act : array<f32>;
@group(0) @binding(5) var<uniform>             gu_p   : GuP;
// The same two weight buffers seen as 16-byte words: one nibble group.
@group(0) @binding(6) var<storage, read>       gu_gw4 : array<vec4<u32>>;
@group(0) @binding(7) var<storage, read>       gu_uw4 : array<vec4<u32>>;
var<workgroup> gu_pg: array<vec4<f32>, 64>;
var<workgroup> gu_pu: array<vec4<f32>, 64>;

fn m4_n4(w: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(f32((w >> sh) & 0xFu), f32((w >> (sh + 4u)) & 0xFu),
                     f32((w >> (sh + 8u)) & 0xFu), f32((w >> (sh + 12u)) & 0xFu))
        - vec4<f32>(8.0);
}
fn m4_d8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(m4_n4(w, 0u), a) + dot(m4_n4(w, 16u), b);
}
// One row's 32-column group: word k of the group covers columns 8k..8k+7.
fn m4_d32(w: vec4<u32>, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>,
          x4: vec4<f32>, x5: vec4<f32>, x6: vec4<f32>, x7: vec4<f32>) -> f32 {
    return (m4_d8(w.x, x0, x1) + m4_d8(w.y, x2, x3)) + (m4_d8(w.z, x4, x5) + m4_d8(w.w, x6, x7));
}
fn gu_gc(o: u32, shf: u32) -> u32 {
    var c = (gu_gw[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
    if (shf > 3u) {
        let o1 = o + 1u;
        c = c | (((gu_gw[o1 >> 2u] >> ((o1 & 3u) * 8u)) & 0xFFu) << 8u);
    }
    return (c >> shf) & 31u;
}
fn gu_uc(o: u32, shf: u32) -> u32 {
    var c = (gu_uw[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
    if (shf > 3u) {
        let o1 = o + 1u;
        c = c | (((gu_uw[o1 >> 2u] >> ((o1 & 3u) * 8u)) & 0xFFu) << 8u);
    }
    return (c >> shf) & 31u;
}

@compute @workgroup_size(64)
fn moe4_gu(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let slot = wid.y;
    let gpr = gu_p.gpr;
    let rows = gu_p.inter;
    let base16 = gu_sel[slot] * gu_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    // Row params: one word (f16 lo, f16 step) per row after the nibbles.
    let parw = (base16 >> 1u) + rows * gpr * 4u + row0;
    let g0 = unpack2x16float(gu_gw[parw]);
    let g1 = unpack2x16float(gu_gw[parw + 1u]);
    let g2 = unpack2x16float(gu_gw[parw + 2u]);
    let g3 = unpack2x16float(gu_gw[parw + 3u]);
    let u0 = unpack2x16float(gu_uw[parw]);
    let u1 = unpack2x16float(gu_uw[parw + 1u]);
    let u2 = unpack2x16float(gu_uw[parw + 2u]);
    let u3 = unpack2x16float(gu_uw[parw + 3u]);
    let glo = vec4<f32>(g0.x, g1.x, g2.x, g3.x);
    let gst = vec4<f32>(g0.y, g1.y, g2.y, g3.y);
    let ulo = vec4<f32>(u0.x, u1.x, u2.x, u3.x);
    let ust = vec4<f32>(u0.y, u1.y, u2.y, u3.y);
    let cod0 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row0 * cst;
    // 16-byte word of (row0, group 0); the next row is gpr words on.
    let nb = (base16 >> 3u) + row0 * gpr;
    var ag = vec4<f32>(0.0);
    var au = vec4<f32>(0.0);
    for (var g = lid; g < gpr; g = g + 64u) {
        let xo = g * 8u;
        let x0 = gu_x[xo];      let x1 = gu_x[xo + 1u];
        let x2 = gu_x[xo + 2u]; let x3 = gu_x[xo + 3u];
        let x4 = gu_x[xo + 4u]; let x5 = gu_x[xo + 5u];
        let x6 = gu_x[xo + 6u]; let x7 = gu_x[xo + 7u];
        let bit = g * 5u;
        let cb = cod0 + (bit >> 3u);
        let shf = bit & 7u;
        let cg = vec4<u32>(gu_gc(cb, shf), gu_gc(cb + cst, shf),
                           gu_gc(cb + 2u * cst, shf), gu_gc(cb + 3u * cst, shf));
        let cu = vec4<u32>(gu_uc(cb, shf), gu_uc(cb + cst, shf),
                           gu_uc(cb + 2u * cst, shf), gu_uc(cb + 3u * cst, shf));
        let sg = exp2(glo + vec4<f32>(cg) * gst);
        let su = exp2(ulo + vec4<f32>(cu) * ust);
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

// ── weighted down projection, four rows a workgroup ──
struct DnP { gpr: u32, hidden: u32, slots: u32, mat16: u32 };
@group(0) @binding(8)  var<storage, read>       dn_w   : array<u32>;
@group(0) @binding(9)  var<storage, read>       dn_w4  : array<vec4<u32>>;
@group(0) @binding(10) var<storage, read>       dn_act : array<vec4<f32>>;
@group(0) @binding(11) var<storage, read>       dn_sel : array<u32>;
@group(0) @binding(12) var<storage, read>       dn_wt  : array<f32>;
@group(0) @binding(13) var<storage, read_write> dn_y   : array<f32>;
@group(0) @binding(14) var<uniform>             dn_p   : DnP;
var<workgroup> dn_pt: array<vec4<f32>, 64>;

fn dn_c(o: u32, shf: u32) -> u32 {
    var c = (dn_w[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
    if (shf > 3u) {
        let o1 = o + 1u;
        c = c | (((dn_w[o1 >> 2u] >> ((o1 & 3u) * 8u)) & 0xFFu) << 8u);
    }
    return (c >> shf) & 31u;
}

@compute @workgroup_size(64)
fn moe4_dn(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let gpr = dn_p.gpr;
    let rows = dn_p.hidden;
    let cst = (gpr * 5u + 7u) / 8u;
    let total = dn_p.slots * gpr;
    var acc = vec4<f32>(0.0);
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i - slot * gpr;
        let base16 = dn_sel[slot] * dn_p.mat16;
        let parw = (base16 >> 1u) + rows * gpr * 4u + row0;
        let p0 = unpack2x16float(dn_w[parw]);
        let p1 = unpack2x16float(dn_w[parw + 1u]);
        let p2 = unpack2x16float(dn_w[parw + 2u]);
        let p3 = unpack2x16float(dn_w[parw + 3u]);
        let cod0 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row0 * cst;
        let bit = g * 5u;
        let cb = cod0 + (bit >> 3u);
        let shf = bit & 7u;
        let cv = vec4<u32>(dn_c(cb, shf), dn_c(cb + cst, shf),
                           dn_c(cb + 2u * cst, shf), dn_c(cb + 3u * cst, shf));
        let scale = exp2(vec4<f32>(p0.x, p1.x, p2.x, p3.x)
                         + vec4<f32>(cv) * vec4<f32>(p0.y, p1.y, p2.y, p3.y));
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
        acc = acc + dn_wt[slot] * (scale * d);
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

/// Expert-grouped prefill MoE (the batch graph). The per-token batch
/// kernels walk every (token, slot) pair through its expert, so a 32-token
/// chunk streamed each selected expert once per token that chose it — 222 ms
/// of a 330 ms chunk on Mellum2.1 (RTX 3090), slower per token than decode.
/// Here the select output is bucketed by expert (`moeg_group`), and one
/// thread per output row walks its expert's weights ONCE for up to eight of
/// that expert's tokens (`moeg_gu`, `moeg_dn`); the per-slot down outputs
/// are then mixed per token (`moeg_sum`) in slot order. Every (row, token)
/// sum is computed by one thread in group order, so the result does not
/// depend on where in its bucket an entry landed.
pub(crate) const MOEG_TOKENS: usize = 8;

fn moeg_wgsl() -> String {
    let mut gu_init = String::new();
    let mut gu_body = String::new();
    let mut gu_out = String::new();
    let mut dn_init = String::new();
    let mut dn_body = String::new();
    let mut dn_out = String::new();
    for j in 0..MOEG_TOKENS {
        gu_init.push_str(&format!(
            "    let e{j} = gg_ent[min(beg + {j}u, end - 1u)];\n    let t{j} = (e{j} / gg_p.slots) * (gpr * 8u);\n    var ag{j} = 0.0; var au{j} = 0.0;\n"
        ));
        gu_body.push_str(&format!(
            "        {{ let xo = t{j} + gx;\n          let x0 = gg_x[xo]; let x1 = gg_x[xo + 1u]; let x2 = gg_x[xo + 2u]; let x3 = gg_x[xo + 3u];\n          let x4 = gg_x[xo + 4u]; let x5 = gg_x[xo + 5u]; let x6 = gg_x[xo + 6u]; let x7 = gg_x[xo + 7u];\n          ag{j} = ag{j} + sg * m4_d32(wg, x0, x1, x2, x3, x4, x5, x6, x7);\n          au{j} = au{j} + su * m4_d32(wu, x0, x1, x2, x3, x4, x5, x6, x7); }}\n"
        ));
        gu_out.push_str(&format!(
            "    if ({j}u < nt) {{ gg_act[e{j} * rows + row] = (ag{j} / (1.0 + exp(-ag{j}))) * au{j}; }}\n"
        ));
        dn_init.push_str(&format!(
            "    let e{j} = gd_ent[min(beg + {j}u, end - 1u)];\n    let t{j} = e{j} * (gpr * 8u);\n    var ac{j} = 0.0;\n"
        ));
        dn_body.push_str(&format!(
            "        {{ let ao = t{j} + gx;\n          let a0 = gd_act[ao]; let a1 = gd_act[ao + 1u]; let a2 = gd_act[ao + 2u]; let a3 = gd_act[ao + 3u];\n          let a4 = gd_act[ao + 4u]; let a5 = gd_act[ao + 5u]; let a6 = gd_act[ao + 6u]; let a7 = gd_act[ao + 7u];\n          ac{j} = ac{j} + sc * m4_d32(w, a0, a1, a2, a3, a4, a5, a6, a7); }}\n"
        ));
        dn_out.push_str(&format!(
            "    if ({j}u < nt) {{ gd_y[e{j} * rows + row] = ac{j}; }}\n"
        ));
    }
    MOEG_WGSL_T
        .replace("{GU_INIT}", &gu_init)
        .replace("{GU_BODY}", &gu_body)
        .replace("{GU_OUT}", &gu_out)
        .replace("{DN_INIT}", &dn_init)
        .replace("{DN_BODY}", &dn_body)
        .replace("{DN_OUT}", &dn_out)
        .replace("{NT}", &MOEG_TOKENS.to_string())
}

const MOEG_WGSL_T: &str = r#"
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

// ── bucket the k·slots select entries by expert ──
struct GrP { n_exp: u32, n: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       gr_sel : array<u32>;
@group(0) @binding(1) var<storage, read_write> gr_off : array<u32>;
@group(0) @binding(2) var<storage, read_write> gr_ent : array<u32>;
@group(0) @binding(3) var<uniform>             gr_p   : GrP;
var<workgroup> gr_cnt: array<atomic<u32>, 256>;
var<workgroup> gr_base: array<u32, 256>;
@compute @workgroup_size(256)
fn moeg_group(@builtin(local_invocation_index) lid: u32) {
    atomicStore(&gr_cnt[lid], 0u);
    workgroupBarrier();
    for (var i = lid; i < gr_p.n; i = i + 256u) { atomicAdd(&gr_cnt[gr_sel[i]], 1u); }
    workgroupBarrier();
    if (lid == 0u) {
        var s = 0u;
        for (var e = 0u; e < gr_p.n_exp; e = e + 1u) {
            gr_base[e] = s;
            gr_off[e] = s;
            s = s + atomicLoad(&gr_cnt[e]);
        }
        gr_off[gr_p.n_exp] = s;
    }
    workgroupBarrier();
    atomicStore(&gr_cnt[lid], 0u);
    workgroupBarrier();
    for (var i = lid; i < gr_p.n; i = i + 256u) {
        let e = gr_sel[i];
        gr_ent[gr_base[e] + atomicAdd(&gr_cnt[e], 1u)] = i;
    }
}

// ── gate+up+SiLU: one thread a row of expert wid.y, {NT} of its entries ──
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
fn gg_gc(o: u32, shf: u32) -> u32 {
    var c = (gg_gw[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
    if (shf > 3u) {
        let o1 = o + 1u;
        c = c | (((gg_gw[o1 >> 2u] >> ((o1 & 3u) * 8u)) & 0xFFu) << 8u);
    }
    return (c >> shf) & 31u;
}
fn gg_uc(o: u32, shf: u32) -> u32 {
    var c = (gg_uw[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
    if (shf > 3u) {
        let o1 = o + 1u;
        c = c | (((gg_uw[o1 >> 2u] >> ((o1 & 3u) * 8u)) & 0xFFu) << 8u);
    }
    return (c >> shf) & 31u;
}
@compute @workgroup_size(64)
fn moeg_gu(@builtin(workgroup_id) wid: vec3<u32>,
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
    let cst = (gpr * 5u + 7u) / 8u;
    let parw = (base16 >> 1u) + rows * gpr * 4u + row;
    let gl = unpack2x16float(gg_gw[parw]);
    let ul = unpack2x16float(gg_uw[parw]);
    let cod0 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row * cst;
    let wb = (base16 >> 3u) + row * gpr;
{GU_INIT}
    for (var g = 0u; g < gpr; g = g + 1u) {
        let bit = g * 5u;
        let cb = cod0 + (bit >> 3u);
        let shf = bit & 7u;
        let sg = exp2(gl.x + f32(gg_gc(cb, shf)) * gl.y);
        let su = exp2(ul.x + f32(gg_uc(cb, shf)) * ul.y);
        let wg = gg_gw4[wb + g];
        let wu = gg_uw4[wb + g];
        let gx = g * 8u;
{GU_BODY}
    }
{GU_OUT}
}

// ── down: one thread a hidden row of expert wid.y, per-entry outputs ──
struct GdP { gpr: u32, hidden: u32, slots: u32, mat16: u32 };
@group(0) @binding(0) var<storage, read>       gd_w   : array<u32>;
@group(0) @binding(1) var<storage, read>       gd_w4  : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       gd_act : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       gd_off : array<u32>;
@group(0) @binding(4) var<storage, read>       gd_ent : array<u32>;
@group(0) @binding(5) var<storage, read_write> gd_y   : array<f32>;
@group(0) @binding(6) var<uniform>             gd_p   : GdP;
fn gd_c(o: u32, shf: u32) -> u32 {
    var c = (gd_w[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
    if (shf > 3u) {
        let o1 = o + 1u;
        c = c | (((gd_w[o1 >> 2u] >> ((o1 & 3u) * 8u)) & 0xFFu) << 8u);
    }
    return (c >> shf) & 31u;
}
@compute @workgroup_size(64)
fn moeg_dn(@builtin(workgroup_id) wid: vec3<u32>,
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
    let cst = (gpr * 5u + 7u) / 8u;
    let pl = unpack2x16float(gd_w[(base16 >> 1u) + rows * gpr * 4u + row]);
    let cod0 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row * cst;
    let wb = (base16 >> 3u) + row * gpr;
{DN_INIT}
    for (var g = 0u; g < gpr; g = g + 1u) {
        let bit = g * 5u;
        let sc = exp2(pl.x + f32(gd_c(cod0 + (bit >> 3u), bit & 7u)) * pl.y);
        let w = gd_w4[wb + g];
        let gx = g * 8u;
{DN_BODY}
    }
{DN_OUT}
}

// ── per-token mix of the slot outputs, in slot order ──
struct GsP { hidden: u32, slots: u32, n: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       gs_y  : array<f32>;
@group(0) @binding(1) var<storage, read>       gs_wt : array<f32>;
@group(0) @binding(2) var<storage, read_write> gs_o  : array<f32>;
@group(0) @binding(3) var<uniform>             gs_p  : GsP;
@compute @workgroup_size(256)
fn moeg_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gs_p.n) { return; }
    let t = i / gs_p.hidden;
    let row = i - t * gs_p.hidden;
    var s = 0.0;
    for (var k = 0u; k < gs_p.slots; k = k + 1u) {
        let e = t * gs_p.slots + k;
        s = s + gs_wt[e] * gs_y[e * gs_p.hidden + row];
    }
    gs_o[i] = s;
}
"#;

pub(crate) struct Prefill {
    pub(crate) group: wgpu::ComputePipeline,
    pub(crate) group_l: wgpu::BindGroupLayout,
    pub(crate) gu: wgpu::ComputePipeline,
    pub(crate) gu_l: wgpu::BindGroupLayout,
    pub(crate) dn: wgpu::ComputePipeline,
    pub(crate) dn_l: wgpu::BindGroupLayout,
    pub(crate) sum: wgpu::ComputePipeline,
    pub(crate) sum_l: wgpu::BindGroupLayout,
}

pub(crate) struct Pipes {
    pub(crate) gu: wgpu::ComputePipeline,
    pub(crate) gu_l: wgpu::BindGroupLayout,
    pub(crate) dn: wgpu::ComputePipeline,
    pub(crate) dn_l: wgpu::BindGroupLayout,
    /// The expert-grouped batch kernels; None when the device rejected
    /// them (or `CMF_MOEG=0`): the batch graph keeps its per-token arm.
    pub(crate) prefill: Option<Prefill>,
}

fn build_prefill(c: &Ctx) -> Option<Prefill> {
    if std::env::var("CMF_MOEG").as_deref() == Ok("0") {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-moeg"),
        source: wgpu::ShaderSource::Wgsl(moeg_wgsl().into()),
    });
    let pipe = |ep: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let (group, gu, dn, sum) = (pipe("moeg_group"), pipe("moeg_gu"), pipe("moeg_dn"), pipe("moeg_sum"));
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-moeg module rejected ({e}): batch MoE keeps the per-token kernels");
        return None;
    }
    Some(Prefill {
        group_l: group.get_bind_group_layout(0),
        gu_l: gu.get_bind_group_layout(0),
        dn_l: dn.get_bind_group_layout(0),
        sum_l: sum.get_bind_group_layout(0),
        group,
        gu,
        dn,
        sum,
    })
}

fn build(c: &Ctx) -> Option<Pipes> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-moe-r4"),
        source: wgpu::ShaderSource::Wgsl(MOE4_WGSL.into()),
    });
    let pipe = |ep: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let gu = pipe("moe4_gu");
    let dn = pipe("moe4_dn");
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-moe-r4 module rejected ({e}): MoE keeps the one-row kernels");
        return None;
    }
    let gu_l = gu.get_bind_group_layout(0);
    let dn_l = dn.get_bind_group_layout(0);
    let prefill = build_prefill(c);
    Some(Pipes { gu, gu_l, dn, dn_l, prefill })
}

/// The module's pipelines, or None when the device rejected them.
pub(crate) fn pipes(c: &Ctx) -> Option<&Pipes> {
    c.moe_r4_pipes.get_or_init(|| build(c)).as_ref()
}

/// The decode pair, unless `CMF_MOE_R4=0` (the A/B switch back to the
/// one-row kernels).
pub(crate) fn decode(c: &Ctx) -> Option<&Pipes> {
    if std::env::var("CMF_MOE_R4").as_deref() == Ok("0") {
        return None;
    }
    pipes(c)
}

/// The expert-grouped batch kernels (`CMF_MOEG=0` keeps the per-token arm).
pub(crate) fn prefill(c: &Ctx) -> Option<&Prefill> {
    pipes(c).and_then(|p| p.prefill.as_ref())
}

/// Can the row-blocked pair serve this q4tp geometry? Rows in fours, and
/// every expert's blob a whole number of 16-byte words (the nibble groups
/// are read as `vec4<u32>`, so each expert must start 16-byte aligned).
pub(crate) fn fits(hidden: usize, inter: usize, gu_mat16: u32, dn_mat16: u32) -> bool {
    hidden % 32 == 0
        && inter % 32 == 0
        && hidden % 4 == 0
        && inter % 4 == 0
        && gu_mat16 % 8 == 0
        && dn_mat16 % 8 == 0
}
