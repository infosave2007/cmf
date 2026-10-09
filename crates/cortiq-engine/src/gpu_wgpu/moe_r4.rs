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
// `fs` (word 5) = n_exp when the select is folded in (`moe4_gu_f`).
struct GuP { gpr: u32, inter: u32, slots: u32, mat16: u32, lim: f32, fs: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       gu_gw  : array<u32>;
@group(0) @binding(1) var<storage, read>       gu_uw  : array<u32>;
@group(0) @binding(2) var<storage, read>       gu_x   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       gu_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> gu_act : array<f32>;
@group(0) @binding(5) var<uniform>             gu_p   : GuP;
// The same two weight buffers seen as 16-byte words: one nibble group.
@group(0) @binding(6) var<storage, read>       gu_gw4 : array<vec4<u32>>;
@group(0) @binding(7) var<storage, read>       gu_uw4 : array<vec4<u32>>;
// Router logits for the folded select.
@group(0) @binding(15) var<storage, read>      fs_lg  : array<f32>;
var<workgroup> gu_pg: array<vec4<f32>, 64>;
var<workgroup> gu_pu: array<vec4<f32>, 64>;
var<workgroup> fs_v: array<f32, 64>;
var<workgroup> fs_pick: u32;

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

fn gu_body(wid: vec3<u32>, lid: u32, expert: u32) {
    let row0 = wid.x * 4u;
    let slot = wid.y;
    let gpr = gu_p.gpr;
    let rows = gu_p.inter;
    let base16 = expert * gu_p.mat16;
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

@compute @workgroup_size(64)
fn moe4_gu(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    gu_body(wid, lid, gu_sel[wid.y]);
}

// The softmax top-k select folded in (no bias, no shared expert,
// n_exp <= 64): slot wid.y's expert is the logit of rank wid.y, ties to
// the lower index — the order `moe_select`'s argmax rounds produce.
@compute @workgroup_size(64)
fn moe4_gu_f(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let n = gu_p.fs;
    fs_v[lid] = select(-3.0e38, fs_lg[lid], lid < n);
    workgroupBarrier();
    if (lid < n) {
        let v = fs_v[lid];
        var r = 0u;
        for (var j = 0u; j < n; j = j + 1u) {
            let u = fs_v[j];
            if (u > v || (u == v && j < lid)) { r = r + 1u; }
        }
        if (r == wid.y) { fs_pick = lid; }
    }
    workgroupBarrier();
    gu_body(wid, lid, fs_pick);
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
// Folded select: (n_exp, top_k, renorm, routed scale bits).
struct FdP { n: u32, k: u32, norm: u32, scale: f32 };
@group(0) @binding(16) var<uniform>             fd_p   : FdP;
var<workgroup> dn_pt: array<vec4<f32>, 64>;
var<workgroup> dn_ss: array<u32, 32>;
var<workgroup> dn_sw: array<f32, 32>;
var<workgroup> fd_r: array<f32, 64>;
var<workgroup> fd_sc: array<f32, 64>;

fn dn_c(o: u32, shf: u32) -> u32 {
    var c = (dn_w[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
    if (shf > 3u) {
        let o1 = o + 1u;
        c = c | (((dn_w[o1 >> 2u] >> ((o1 & 3u) * 8u)) & 0xFFu) << 8u);
    }
    return (c >> shf) & 31u;
}

// The slots' experts and weights are in dn_ss / dn_sw.
fn dn_body(wid: vec3<u32>, lid: u32) {
    let row0 = wid.x * 4u;
    let gpr = dn_p.gpr;
    let rows = dn_p.hidden;
    let cst = (gpr * 5u + 7u) / 8u;
    let total = dn_p.slots * gpr;
    var acc = vec4<f32>(0.0);
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i - slot * gpr;
        let base16 = dn_ss[slot] * dn_p.mat16;
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

@compute @workgroup_size(64)
fn moe4_dn(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    if (lid < dn_p.slots) {
        dn_ss[lid] = dn_sel[lid];
        dn_sw[lid] = dn_wt[lid];
    }
    workgroupBarrier();
    dn_body(wid, lid);
}

// `moe_select`'s softmax path, bit for bit, inside every down workgroup:
// max and sum through the same trees (a 256-lane tree padded with -3e38 /
// zeros is this 64-lane tree), probabilities exp(v - max)/sum, the top-k
// by rank (ties to the lower index), the renorm sum taken in slot order,
// then the routed scale.
@compute @workgroup_size(64)
fn moe4_dn_f(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let n = fd_p.n;
    let v = select(-3.0e38, fs_lg[lid], lid < n);
    fs_v[lid] = v;
    fd_r[lid] = v;
    workgroupBarrier();
    var st = 32u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { fd_r[lid] = max(fd_r[lid], fd_r[lid + st]); }
        workgroupBarrier();
        st = st >> 1u;
    }
    let mx = fd_r[0];
    workgroupBarrier();
    fd_r[lid] = select(0.0, exp(v - mx), lid < n);
    workgroupBarrier();
    st = 32u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { fd_r[lid] = fd_r[lid] + fd_r[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let denom = fd_r[0];
    var msc = 0.0;
    if (lid < n) { msc = exp(v - mx) / denom; }
    fd_sc[lid] = msc;
    if (lid < n) {
        var r = 0u;
        for (var j = 0u; j < n; j = j + 1u) {
            let u = fs_v[j];
            if (u > v || (u == v && j < lid)) { r = r + 1u; }
        }
        if (r < fd_p.k) { dn_ss[r] = lid; }
    }
    workgroupBarrier();
    if (lid == 0u) {
        var wsum = 0.0;
        for (var s = 0u; s < fd_p.k; s = s + 1u) { wsum = wsum + fd_sc[dn_ss[s]]; }
        for (var s = 0u; s < fd_p.k; s = s + 1u) {
            var w = fd_sc[dn_ss[s]];
            if (fd_p.norm != 0u) { w = w / wsum; }
            dn_sw[s] = w * fd_p.scale;
        }
    }
    workgroupBarrier();
    dn_body(wid, lid);
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

/// Q, K and V q8_2f projections of one input in ONE dispatch: the three
/// matrices' four-row blocks laid end to end over the grid, each block
/// running `q8_2f_matvec4`'s row walk, lane order, reduction tree and scale
/// exactly — so every output equals the three-dispatch path's bit for bit.
/// Two dispatch boundaries fewer per attention layer.
fn mv3_wgsl() -> String {
    let mut body = String::new();
    for m in ["a", "b", "c"] {
        body.push_str(&MV3_FN_T.replace("{M}", m));
    }
    MV3_WGSL_T.replace("{FNS}", &body)
}

const MV3_FN_T: &str = r#"
fn m3f16x4_{M}(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(m3_w{M}[w]);
    let b = unpack2x16float(m3_w{M}[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(m3_w{M}[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
fn m3_row_{M}(wb: u32, lid: u32) {
    let rows = m3_p.rows_{M};
    let qbytes = rows * m3_p.cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    let ngrp = m3_p.cols / 4u;
    let sub = lid >> 6u;
    let l = lid & 63u;
    let row = wb * 4u + sub;
    var acc = 0.0;
    if (row < rows) {
        let roww = row * ngrp;
        var i = l;
        loop {
            if (i + 192u >= ngrp) { break; }
            let w0 = m3_w{M}[roww + i];
            let w1 = m3_w{M}[roww + i + 64u];
            let w2 = m3_w{M}[roww + i + 128u];
            let w3 = m3_w{M}[roww + i + 192u];
            let x0 = m3_x[i];
            let x1 = m3_x[i + 64u];
            let x2 = m3_x[i + 128u];
            let x3 = m3_x[i + 192u];
            let s0 = m3f16x4_{M}(cs0h + i * 4u);
            let s1 = m3f16x4_{M}(cs0h + (i + 64u) * 4u);
            let s2 = m3f16x4_{M}(cs0h + (i + 128u) * 4u);
            let s3 = m3f16x4_{M}(cs0h + (i + 192u) * 4u);
            acc = acc + dot(m3i8x4(w0), x0 * s0);
            acc = acc + dot(m3i8x4(w1), x1 * s1);
            acc = acc + dot(m3i8x4(w2), x2 * s2);
            acc = acc + dot(m3i8x4(w3), x3 * s3);
            i = i + 256u;
        }
        loop {
            if (i >= ngrp) { break; }
            acc = acc + dot(m3i8x4(m3_w{M}[roww + i]), m3_x[i] * m3f16x4_{M}(cs0h + i * 4u));
            i = i + 64u;
        }
    }
    m3_part[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (l < stride) { m3_part[lid] = m3_part[lid] + m3_part[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (l == 0u && row < rows) {
        let rw = unpack2x16float(m3_w{M}[rs0 + (row >> 1u)]);
        var sc = rw.x;
        if ((row & 1u) == 1u) { sc = rw.y; }
        m3_y{M}[row] = m3_part[lid] * sc;
    }
    workgroupBarrier();
}
"#;

const MV3_WGSL_T: &str = r#"
struct M3P { rows_a: u32, rows_b: u32, rows_c: u32, cols: u32 };
@group(0) @binding(0) var<storage, read>       m3_wa : array<u32>;
@group(0) @binding(1) var<storage, read>       m3_wb : array<u32>;
@group(0) @binding(2) var<storage, read>       m3_wc : array<u32>;
@group(0) @binding(3) var<storage, read>       m3_x  : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> m3_ya : array<f32>;
@group(0) @binding(5) var<storage, read_write> m3_yb : array<f32>;
@group(0) @binding(6) var<storage, read_write> m3_yc : array<f32>;
@group(0) @binding(7) var<uniform>             m3_p  : M3P;
var<workgroup> m3_part: array<f32, 256>;
fn m3i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    let b0 = (s << 24u) >> 24u;
    let b1 = (s << 16u) >> 24u;
    let b2 = (s <<  8u) >> 24u;
    let b3 =  s          >> 24u;
    return vec4<f32>(f32(b0), f32(b1), f32(b2), f32(b3));
}
{FNS}
@compute @workgroup_size(256)
fn q82_mv3(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let ba = (m3_p.rows_a + 3u) / 4u;
    let bb = (m3_p.rows_b + 3u) / 4u;
    let bc = (m3_p.rows_c + 3u) / 4u;
    var wb = wid.x;
    loop {
        if (wb >= ba + bb + bc) { break; }
        if (wb < ba) {
            m3_row_a(wb, lid);
        } else if (wb < ba + bb) {
            m3_row_b(wb - ba, lid);
        } else {
            m3_row_c(wb - ba - bb, lid);
        }
        wb = wb + nwg.x;
    }
}
"#;

/// Subgroup variant of the folded-select gate/up: ONE expert row per
/// 32-lane subgroup, a lane per 32-bit nibble word (128-byte coalesced
/// loads per warp instruction), the nibble converted by the exponent trick,
/// `subgroupAdd` for the row sum — the design the CUDA experiment found
/// fastest for b=1 q4tp GEMV, ported to the graph's expert buffers with the
/// select folded in exactly as `moe4_gu_f` does it. Its down twin measured
/// slower than `moe4_dn_f` (1.07 vs 0.98 ms a token) and was dropped. Built only
/// where the device has SUBGROUP with a fixed 32-lane width (NVIDIA); naga
/// (wgpu 30) takes the builtins without an `enable` directive.
const MOESG_WGSL: &str = r#"
struct SgGuP { gpr: u32, inter: u32, slots: u32, mat16: u32, lim: f32, fs: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read>       sg_gw  : array<u32>;
@group(0) @binding(1) var<storage, read>       sg_uw  : array<u32>;
@group(0) @binding(2) var<storage, read>       sg_x   : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> sg_act : array<f32>;
@group(0) @binding(5) var<uniform>             sg_p   : SgGuP;
@group(0) @binding(15) var<storage, read>      sg_lg  : array<f32>;
var<workgroup> sg_v: array<f32, 128>;
var<workgroup> sg_pick: u32;

fn sg_nib(v: u32, sh: u32) -> f32 {
    return bitcast<f32>(((v >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn sg_d8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return sg_nib(w, 0u) * a.x + sg_nib(w, 4u) * a.y + sg_nib(w, 8u) * a.z + sg_nib(w, 12u) * a.w
         + sg_nib(w, 16u) * b.x + sg_nib(w, 20u) * b.y + sg_nib(w, 24u) * b.z + sg_nib(w, 28u) * b.w;
}
// 5-bit field at bit offset `sh` of the word pair (lo, hi).
fn sg_c5(lo: u32, hi: u32, sh: u32) -> u32 {
    return select((lo >> sh) | (hi << (32u - sh)), lo, sh == 0u) & 31u;
}

@compute @workgroup_size(128)
fn moesg_gu_f(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32,
              @builtin(subgroup_size) ssz: u32) {
    let n = sg_p.fs;
    sg_v[lid] = select(-3.0e38, sg_lg[min(lid, n - 1u)], lid < n);
    workgroupBarrier();
    if (lid < n) {
        let v = sg_v[lid];
        var r = 0u;
        for (var j = 0u; j < n; j = j + 1u) {
            let u = sg_v[j];
            if (u > v || (u == v && j < lid)) { r = r + 1u; }
        }
        if (r == wid.y) { sg_pick = lid; }
    }
    workgroupBarrier();
    let expert = sg_pick;
    let slot = wid.y;
    let rows = sg_p.inter;
    let gpr = sg_p.gpr;
    let row = wid.x * (128u / ssz) + lid / ssz;
    let live = row < rows;
    let r = select(rows - 1u, row, live);
    let nw = gpr * 4u;
    let base16 = expert * sg_p.mat16;
    let wb = base16 >> 1u;
    let pg = unpack2x16float(sg_gw[wb + rows * nw + r]);
    let pu = unpack2x16float(sg_uw[wb + rows * nw + r]);
    let crow = base16 * 2u + rows * gpr * 16u + rows * 4u + r * ((gpr * 5u + 7u) / 8u);
    var ag = 0.0;
    var au = 0.0;
    var k = lane;
    loop {
        if (k >= nw) { break; }
        let bit = (k >> 2u) * 5u;
        let cb = crow + (bit >> 3u);
        let wi = cb >> 2u;
        let sh = (cb & 3u) * 8u + (bit & 7u);
        let sgv = exp2(pg.x + f32(sg_c5(sg_gw[wi], sg_gw[wi + 1u], sh)) * pg.y);
        let suv = exp2(pu.x + f32(sg_c5(sg_uw[wi], sg_uw[wi + 1u], sh)) * pu.y);
        let wg = sg_gw[wb + r * nw + k];
        let wu = sg_uw[wb + r * nw + k];
        let xa = sg_x[k * 2u];
        let xb = sg_x[k * 2u + 1u];
        ag = ag + sgv * sg_d8(wg, xa, xb);
        au = au + suv * sg_d8(wu, xa, xb);
        k = k + ssz;
    }
    let tg = subgroupAdd(ag);
    let tu = subgroupAdd(au);
    if (lane == 0u && live) {
        var gg = tg;
        var uu = tu;
        if (sg_p.lim > 0.0) {
            uu = clamp(uu, -sg_p.lim, sg_p.lim);
            gg = min(gg, sg_p.lim);
        }
        sg_act[slot * rows + row] = (gg / (1.0 + exp(-gg))) * uu;
    }
}

"#;

pub(crate) struct Sg {
    pub(crate) gu: wgpu::ComputePipeline,
    pub(crate) gu_l: wgpu::BindGroupLayout,
}

fn sg32(c: &Ctx) -> bool {
    c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
}

fn build_q82sg(c: &Ctx) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    if std::env::var("CMF_Q82_SG").as_deref() == Ok("0") || !sg32(c) {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-q82-sg"),
        source: wgpu::ShaderSource::Wgsl(q82sg_wgsl().into()),
    });
    let p = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q82_sg"),
            layout: None,
            module: &module,
            entry_point: Some("q82_sg"),
            compilation_options: Default::default(),
            cache: c.pipeline_cache.as_ref(),
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-q82-sg module rejected ({e}): q8_2f keeps the 64-lane kernels");
        return None;
    }
    let l = p.get_bind_group_layout(0);
    Some((p, l))
}

/// Encode up to three q8_2f matvecs of ONE input `x` (`mats`: weight,
/// output, rows) through `q82_sg`. False when the kernel is missing or the
/// width is outside it (cols % 16, cols <= 4096); nothing is encoded then.
pub(crate) fn q82sg_encode(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    cols: usize,
    mats: &[(&wgpu::Buffer, &wgpu::Buffer, usize)],
) -> bool {
    let Some(ps) = pipes(c) else { return false };
    let Some((p, l)) = ps.q82sg.as_ref() else {
        return false;
    };
    if cols % 16 != 0 || cols > 4096 || mats.is_empty() || mats.len() > 3 {
        return false;
    }
    let rows_max = mats.iter().map(|m| m.2).max().unwrap_or(0);
    // Rows per warp: enough workgroups to fill the card on the narrow
    // matrices, fewer activation stagings on the wide ones (the head).
    // Rows per warp (handled in pairs).
    let rpw: usize = if rows_max >= 32768 { 4 } else { 2 };
    let per = 8 * rpw;
    let r = |i: usize| mats.get(i).map_or(0, |m| m.2);
    let u = uniform_u32x8(
        c,
        [r(0) as u32, r(1) as u32, r(2) as u32, cols as u32, rpw as u32, 0, 0, 0],
    );
    let w = |i: usize| mats.get(i).map_or(mats[0].0, |m| m.0);
    let y = |i: usize| mats.get(i).map_or(&ps.dummy_y, |m| m.1);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q82-sg"),
        layout: l,
        entries: &[
            bind_buf(0, w(0)),
            bind_buf(1, w(1)),
            bind_buf(2, w(2)),
            bind_buf(3, x),
            bind_buf(4, y(0)),
            bind_buf(5, y(1)),
            bind_buf(6, y(2)),
            bind_buf(7, &u),
        ],
    });
    let groups: usize = (0..3).map(|i| r(i).div_ceil(per)).sum();
    if groups as u32 > MAX_WG {
        return false;
    }
    let mut pass = begin_pass(enc);
    pass.set_pipeline(p);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(groups as u32, 1, 1);
    true
}

fn build_sg(c: &Ctx) -> Option<Sg> {
    if std::env::var("CMF_MOE_SG").as_deref() == Ok("0") || !sg32(c) {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-moe-sg"),
        source: wgpu::ShaderSource::Wgsl(MOESG_WGSL.into()),
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
    let gu = pipe("moesg_gu_f");
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-moe-sg module rejected ({e}): MoE keeps the four-row kernels");
        return None;
    }
    Some(Sg {
        gu_l: gu.get_bind_group_layout(0),
        gu,
    })
}

/// q8_2f projections with the subgroup design: a WARP per row (32 lanes
/// over the row's 32-bit int8 words, 128-byte coalesced loads), the
/// activations pre-scaled by the column scales ONCE per workgroup into
/// workgroup memory (`q8_2f_matvec4` re-reads the f16 column scales for
/// every row), `subgroupAdd` for the row sum, then the row scale. Up to
/// three matrices of one input per dispatch (Q/K/V); one for O and the head.
/// Same products as `q8_2f_matvec4` (x·colscale, then the int8 dot), a
/// different summation order. cols <= 4096, cols % 16 == 0.
fn q82sg_wgsl() -> String {
    let mut body = String::new();
    for m in ["a", "b", "c"] {
        body.push_str(&Q82SG_FN_T.replace("{M}", m));
    }
    Q82SG_WGSL_T.replace("{FNS}", &body)
}

const Q82SG_FN_T: &str = r#"
fn qs_cs4_{M}(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(qs_w{M}[w]);
    let b = unpack2x16float(qs_w{M}[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(qs_w{M}[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
fn qs_run_{M}(blk: u32, lid: u32, lane: u32, ssz: u32) {
    let rows = qs_p.rows_{M};
    let cols = qs_p.cols;
    let words = cols / 4u;
    let qbytes = rows * cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    for (var i = lid; i < words; i = i + 256u) {
        qs_x4[i] = qs_x[i] * qs_cs4_{M}(cs0h + i * 4u);
    }
    workgroupBarrier();
    let wpw = 256u / ssz;
    let warp = lid / ssz;
    // Two rows a warp at once, six words a lane per step: twelve loads in
    // flight before the first dot.
    for (var r = 0u; r < qs_p.rpw; r = r + 2u) {
        let row0 = (blk * qs_p.rpw + r) * wpw + warp * 2u;
        let live0 = row0 < rows;
        let live1 = row0 + 1u < rows;
        let r0 = select(rows - 1u, row0, live0);
        let r1 = select(rows - 1u, row0 + 1u, live1);
        let b0 = r0 * words;
        let b1 = r1 * words;
        var a0 = 0.0;
        var a1 = 0.0;
        var k = lane;
        loop {
            if (k + 5u * ssz >= words) { break; }
            let p0 = qs_w{M}[b0 + k];
            let p1 = qs_w{M}[b0 + k + ssz];
            let p2 = qs_w{M}[b0 + k + 2u * ssz];
            let p3 = qs_w{M}[b0 + k + 3u * ssz];
            let p4 = qs_w{M}[b0 + k + 4u * ssz];
            let p5 = qs_w{M}[b0 + k + 5u * ssz];
            let q0 = qs_w{M}[b1 + k];
            let q1 = qs_w{M}[b1 + k + ssz];
            let q2 = qs_w{M}[b1 + k + 2u * ssz];
            let q3 = qs_w{M}[b1 + k + 3u * ssz];
            let q4 = qs_w{M}[b1 + k + 4u * ssz];
            let q5 = qs_w{M}[b1 + k + 5u * ssz];
            let x0 = qs_x4[k];
            let x1 = qs_x4[k + ssz];
            let x2 = qs_x4[k + 2u * ssz];
            let x3 = qs_x4[k + 3u * ssz];
            let x4 = qs_x4[k + 4u * ssz];
            let x5 = qs_x4[k + 5u * ssz];
            a0 = a0 + dot(qs_i8x4(p0), x0) + dot(qs_i8x4(p1), x1) + dot(qs_i8x4(p2), x2)
                 + dot(qs_i8x4(p3), x3) + dot(qs_i8x4(p4), x4) + dot(qs_i8x4(p5), x5);
            a1 = a1 + dot(qs_i8x4(q0), x0) + dot(qs_i8x4(q1), x1) + dot(qs_i8x4(q2), x2)
                 + dot(qs_i8x4(q3), x3) + dot(qs_i8x4(q4), x4) + dot(qs_i8x4(q5), x5);
            k = k + 6u * ssz;
        }
        loop {
            if (k >= words) { break; }
            let x = qs_x4[k];
            a0 = a0 + dot(qs_i8x4(qs_w{M}[b0 + k]), x);
            a1 = a1 + dot(qs_i8x4(qs_w{M}[b1 + k]), x);
            k = k + ssz;
        }
        let t0 = subgroupAdd(a0);
        let t1 = subgroupAdd(a1);
        if (lane == 0u && live0) {
            let rw = unpack2x16float(qs_w{M}[rs0 + (r0 >> 1u)]);
            qs_y{M}[r0] = t0 * select(rw.x, rw.y, (r0 & 1u) == 1u);
        }
        if (lane == 0u && live1) {
            let rw = unpack2x16float(qs_w{M}[rs0 + (r1 >> 1u)]);
            qs_y{M}[r1] = t1 * select(rw.x, rw.y, (r1 & 1u) == 1u);
        }
    }
}
"#;

const Q82SG_WGSL_T: &str = r#"
struct QsP { rows_a: u32, rows_b: u32, rows_c: u32, cols: u32, rpw: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       qs_wa : array<u32>;
@group(0) @binding(1) var<storage, read>       qs_wb : array<u32>;
@group(0) @binding(2) var<storage, read>       qs_wc : array<u32>;
@group(0) @binding(3) var<storage, read>       qs_x  : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> qs_ya : array<f32>;
@group(0) @binding(5) var<storage, read_write> qs_yb : array<f32>;
@group(0) @binding(6) var<storage, read_write> qs_yc : array<f32>;
@group(0) @binding(7) var<uniform>             qs_p  : QsP;
var<workgroup> qs_x4: array<vec4<f32>, 1024>;
fn qs_i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    let b0 = (s << 24u) >> 24u;
    let b1 = (s << 16u) >> 24u;
    let b2 = (s <<  8u) >> 24u;
    let b3 =  s          >> 24u;
    return vec4<f32>(f32(b0), f32(b1), f32(b2), f32(b3));
}
{FNS}
@compute @workgroup_size(256)
fn q82_sg(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32,
          @builtin(subgroup_invocation_id) lane: u32,
          @builtin(subgroup_size) ssz: u32) {
    let per = (256u / ssz) * qs_p.rpw;
    let ba = (qs_p.rows_a + per - 1u) / per;
    let bb = (qs_p.rows_b + per - 1u) / per;
    let wb = wid.x;
    if (wb < ba) {
        qs_run_a(wb, lid, lane, ssz);
    } else if (wb < ba + bb) {
        qs_run_b(wb - ba, lid, lane, ssz);
    } else {
        qs_run_c(wb - ba - bb, lid, lane, ssz);
    }
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
    /// The pair with the softmax top-k select folded in (no select
    /// dispatch): `moe4_gu_f`, `moe4_dn_f`.
    pub(crate) gu_f: wgpu::ComputePipeline,
    pub(crate) gu_f_l: wgpu::BindGroupLayout,
    pub(crate) dn_f: wgpu::ComputePipeline,
    pub(crate) dn_f_l: wgpu::BindGroupLayout,
    /// `q82_mv3` (Q/K/V q8_2f in one dispatch); None on rejection or
    /// `CMF_MV3=0`.
    pub(crate) mv3: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// Subgroup folded-select pair (four rows a 128-lane workgroup).
    pub(crate) sg: Option<Sg>,
    /// `q82_sg`: warp-per-row q8_2f projections (Q/K/V together, O, head).
    pub(crate) q82sg: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    /// Never-written output slot for the unused matrices of a `q82_sg` call.
    pub(crate) dummy_y: wgpu::Buffer,
    /// The expert-grouped batch kernels; None when the device rejected
    /// them (or `CMF_MOEG=0`): the batch graph keeps its per-token arm.
    pub(crate) prefill: Option<Prefill>,
}

fn build_mv3(c: &Ctx) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    if std::env::var("CMF_MV3").as_deref() == Ok("0") {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-mv3"),
        source: wgpu::ShaderSource::Wgsl(mv3_wgsl().into()),
    });
    let p = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("q82_mv3"),
            layout: None,
            module: &module,
            entry_point: Some("q82_mv3"),
            compilation_options: Default::default(),
            cache: c.pipeline_cache.as_ref(),
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-mv3 module rejected ({e}): Q/K/V keep three dispatches");
        return None;
    }
    let l = p.get_bind_group_layout(0);
    Some((p, l))
}

/// The fused Q/K/V q8_2f projection, when the module came up.
pub(crate) fn mv3(c: &Ctx) -> Option<&(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    pipes(c).and_then(|p| p.mv3.as_ref())
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
    let gu_f = pipe("moe4_gu_f");
    let dn_f = pipe("moe4_dn_f");
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-moe-r4 module rejected ({e}): MoE keeps the one-row kernels");
        return None;
    }
    let gu_l = gu.get_bind_group_layout(0);
    let dn_l = dn.get_bind_group_layout(0);
    let gu_f_l = gu_f.get_bind_group_layout(0);
    let dn_f_l = dn_f.get_bind_group_layout(0);
    let prefill = build_prefill(c);
    let mv3 = build_mv3(c);
    let sg = build_sg(c);
    let q82sg = build_q82sg(c);
    let dummy_y = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("q82-sg-dummy"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    Some(Pipes {
        gu, gu_l, dn, dn_l, gu_f, gu_f_l, dn_f, dn_f_l, mv3, sg, q82sg, dummy_y, prefill,
    })
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
