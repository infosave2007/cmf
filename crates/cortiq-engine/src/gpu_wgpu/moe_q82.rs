//! q8_2f MoE experts on the whole-token graph (decode) and the batch graph
//! (prefill) — Mellum2.1's quality file: 64 experts, top-8, gate, up and
//! down all q8_2f.
//!
//! A q8_2f expert blob is `[int8: rows·cols][f16 row scale: rows][f16 col:
//! cols]` and `w[o,i] = q[o,i]·rs[o]·col[i]`. The host contract (strict CPU
//! `qmatvec`) multiplies the activation by the column field FIRST, then
//! takes the int8 dot, then the row scale — so every expert needs its own
//! `x·col`: gate and up each have one, and so does every expert's down (a
//! 0.8.14 Metal bug came from sharing one staged input across expert jobs).
//! The products here are those same f32 multiplies; only the summation
//! order differs from the host.
//!
//! Decode (`m8_gu`, `m8_dn`): a 256-lane workgroup, one 32-lane warp per
//! row, 32-bit int8 words read coalesced. Gate/up stage `x·col_gate` and
//! `x·col_up` of their slot's expert in workgroup memory once, and write
//! `silu(g)·u` already multiplied by the down's column field, so the down
//! kernel reads its input straight from the act buffer. The softmax top-k
//! select can ride inside both (bit-identical to `moe_select`, as in
//! `moe_r4`'s folded pair).
//!
//! Prefill (`m8g_gu`, `m8g_dn`): the expert-grouped batch shape of
//! `moe_r4`'s moeg kernels (bucketed by `moeg_group`, mixed by `moeg_sum`):
//! one thread per output row, up to eight entries of one expert per
//! workgroup, the entries' `x·col` staged per 128-column tile.

use super::*;

/// Prefill entries per workgroup (`CMF_M8_NT`: 4, 8, 16 or 32). Every
/// workgroup streams its expert's rows once, so the expert's bytes are read
/// once per group of entries.
fn nt() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("CMF_M8_NT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| matches!(v, 4 | 8 | 16 | 32))
            .unwrap_or(16)
    })
}
/// Prefill column tile, in 32-bit words (four columns each).
const TW: usize = 32;

/// Byte length of one q8_2f expert matrix.
pub(crate) fn blob_bytes(rows: usize, cols: usize) -> usize {
    rows * cols + 2 * rows + 2 * cols
}

fn sg32(c: &Ctx) -> bool {
    c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
}

/// Geometry both kernel sets can address: word-aligned rows and blobs, the
/// scale planes starting on a word (even row counts), and 16-byte groups for
/// the prefill's vector weight reads.
pub(crate) fn fits(hidden: usize, inter: usize) -> bool {
    hidden % 16 == 0
        && inter % 16 == 0
        && hidden > 0
        && inter > 0
        && blob_bytes(inter, hidden) % 16 == 0
        && blob_bytes(hidden, inter) % 16 == 0
}

fn env_rpw(key: &str, def: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| (1..=64).contains(v))
        .unwrap_or(def)
}

/// Rows per warp of the decode gate/up (`CMF_M8_RPW_GU`) and down
/// (`CMF_M8_RPW_DN`) kernels.
fn rpw_gu() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| env_rpw("CMF_M8_RPW_GU", 4))
}
fn rpw_dn() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| env_rpw("CMF_M8_RPW_DN", 1))
}

const COMMON_WGSL: &str = r#"
struct M8P { hidden: u32, inter: u32, slots: u32, gsw: u32, dsw: u32, fold: u32, rpw: u32, _p: u32 };
fn q8x4(w: u32) -> vec4<f32> {
    let s = bitcast<i32>(w);
    return vec4<f32>(f32((s << 24u) >> 24u), f32((s << 16u) >> 24u),
                     f32((s << 8u) >> 24u), f32(s >> 24u));
}
fn h2(w: u32, odd: bool) -> f32 {
    let v = unpack2x16float(w);
    return select(v.x, v.y, odd);
}
"#;

const DECODE_WGSL: &str = r#"
// ── gate + up + SiLU (· down's column field), one warp a row ──
@group(0) @binding(0) var<storage, read>       g8_gw  : array<u32>;
@group(0) @binding(1) var<storage, read>       g8_uw  : array<u32>;
@group(0) @binding(2) var<storage, read>       g8_dw  : array<u32>;
@group(0) @binding(3) var<storage, read>       g8_x   : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       g8_sel : array<u32>;
@group(0) @binding(5) var<storage, read_write> g8_act : array<f32>;
@group(0) @binding(6) var<uniform>             g8_p   : M8P;
@group(0) @binding(7) var<storage, read>       g8_lg  : array<f32>;
var<workgroup> g8_xg: array<vec4<f32>, {HV4}>;
var<workgroup> g8_xu: array<vec4<f32>, {HV4}>;
var<workgroup> g8_v: array<f32, 64>;
var<workgroup> g8_pick: u32;

@compute @workgroup_size(256)
fn m8_gu(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(local_invocation_index) lid: u32,
         @builtin(subgroup_invocation_id) lane: u32,
         @builtin(subgroup_size) ssz: u32) {
    let slot = wid.y;
    let n = g8_p.fold;
    if (n != 0u) {
        // `moe4_gu_f`'s folded select: slot wid.y's expert is the logit of
        // rank wid.y, ties to the lower index.
        if (lid < 64u) { g8_v[lid] = select(-3.0e38, g8_lg[min(lid, n - 1u)], lid < n); }
        workgroupBarrier();
        if (lid < n) {
            let v = g8_v[lid];
            var r = 0u;
            for (var j = 0u; j < n; j = j + 1u) {
                let u = g8_v[j];
                if (u > v || (u == v && j < lid)) { r = r + 1u; }
            }
            if (r == slot) { g8_pick = lid; }
        }
    } else if (lid == 0u) {
        g8_pick = g8_sel[slot];
    }
    let expert = workgroupUniformLoad(&g8_pick);
    let hv4 = g8_p.hidden >> 2u;
    let rows = g8_p.inter;
    let gb = expert * g8_p.gsw;
    let rsw = gb + rows * hv4;
    let csw = rsw + (rows >> 1u);
    for (var i = lid; i < hv4; i = i + 256u) {
        let xv = g8_x[i];
        let c = csw + 2u * i;
        g8_xg[i] = xv * vec4<f32>(unpack2x16float(g8_gw[c]), unpack2x16float(g8_gw[c + 1u]));
        g8_xu[i] = xv * vec4<f32>(unpack2x16float(g8_uw[c]), unpack2x16float(g8_uw[c + 1u]));
    }
    workgroupBarrier();
    let nwarp = 256u / ssz;
    let warp = lid / ssz;
    let dcw = expert * g8_p.dsw + g8_p.hidden * (rows >> 2u) + (g8_p.hidden >> 1u);
    for (var rr = 0u; rr < g8_p.rpw; rr = rr + 1u) {
        let row = (wid.x * g8_p.rpw + rr) * nwarp + warp;
        let live = row < rows;
        let r = select(rows - 1u, row, live);
        let w0 = gb + r * hv4;
        var ag = 0.0;
        var au = 0.0;
        // {UG} words of gate and of up a lane per pass, all loads issued
        // before the first dot; each lane still sums its words in order.
        for (var kb = 0u; kb < hv4; kb = kb + {UG}u * ssz) {
{GU_LOAD}
{GU_DOT}
        }
        let tg = subgroupAdd(ag);
        let tu = subgroupAdd(au);
        if (lane == 0u && live) {
            let odd = (row & 1u) == 1u;
            let gg = tg * h2(g8_gw[rsw + (row >> 1u)], odd);
            let uu = tu * h2(g8_uw[rsw + (row >> 1u)], odd);
            let a = (gg / (1.0 + exp(-gg))) * uu;
            g8_act[slot * rows + row] = a * h2(g8_dw[dcw + (row >> 1u)], odd);
        }
    }
}

// ── weighted down over the slots, one warp a hidden row ──
struct FdP { n: u32, k: u32, norm: u32, scale: f32 };
@group(0) @binding(8)  var<storage, read>       d8_w   : array<u32>;
@group(0) @binding(9)  var<storage, read>       d8_act : array<vec4<f32>>;
@group(0) @binding(10) var<storage, read>       d8_sel : array<u32>;
@group(0) @binding(11) var<storage, read>       d8_wt  : array<f32>;
@group(0) @binding(12) var<storage, read_write> d8_y   : array<f32>;
@group(0) @binding(13) var<uniform>             d8_p   : M8P;
@group(0) @binding(14) var<uniform>             d8_fd  : FdP;
@group(0) @binding(15) var<storage, read>       d8_lg  : array<f32>;
var<workgroup> d8_ss: array<u32, 32>;
var<workgroup> d8_sw: array<f32, 32>;
var<workgroup> d8_v: array<f32, 64>;
var<workgroup> d8_r: array<f32, 64>;
var<workgroup> d8_sc: array<f32, 64>;

@compute @workgroup_size(256)
fn m8_dn(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(local_invocation_index) lid: u32,
         @builtin(subgroup_invocation_id) lane: u32,
         @builtin(subgroup_size) ssz: u32) {
    if (d8_p.fold != 0u) {
        // `moe4_dn_f`'s select (= `moe_select`'s softmax path, bit for
        // bit) on the first 64 lanes.
        let n = d8_fd.n;
        var v = -3.0e38;
        if (lid < 64u) {
            v = select(-3.0e38, d8_lg[min(lid, n - 1u)], lid < n);
            d8_v[lid] = v;
            d8_r[lid] = v;
        }
        workgroupBarrier();
        var st = 32u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { d8_r[lid] = max(d8_r[lid], d8_r[lid + st]); }
            workgroupBarrier();
            st = st >> 1u;
        }
        let mx = d8_r[0];
        workgroupBarrier();
        if (lid < 64u) { d8_r[lid] = select(0.0, exp(v - mx), lid < n); }
        workgroupBarrier();
        st = 32u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { d8_r[lid] = d8_r[lid] + d8_r[lid + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        let denom = d8_r[0];
        var msc = 0.0;
        if (lid < n) { msc = exp(v - mx) / denom; }
        if (lid < 64u) { d8_sc[lid] = msc; }
        if (lid < n) {
            var r = 0u;
            for (var j = 0u; j < n; j = j + 1u) {
                let u = d8_v[j];
                if (u > v || (u == v && j < lid)) { r = r + 1u; }
            }
            if (r < d8_fd.k) { d8_ss[r] = lid; }
        }
        workgroupBarrier();
        if (lid == 0u) {
            var wsum = 0.0;
            for (var s = 0u; s < d8_fd.k; s = s + 1u) { wsum = wsum + d8_sc[d8_ss[s]]; }
            for (var s = 0u; s < d8_fd.k; s = s + 1u) {
                var w = d8_sc[d8_ss[s]];
                if (d8_fd.norm != 0u) { w = w / wsum; }
                d8_sw[s] = w * d8_fd.scale;
            }
        }
    } else if (lid < d8_p.slots) {
        d8_ss[lid] = d8_sel[lid];
        d8_sw[lid] = d8_wt[lid];
    }
    workgroupBarrier();
    let iv4 = d8_p.inter >> 2u;
    let rows = d8_p.hidden;
    let qw = rows * iv4;
    let nwarp = 256u / ssz;
    let warp = lid / ssz;
    for (var rr = 0u; rr < d8_p.rpw; rr = rr + 1u) {
        let row = (wid.x * d8_p.rpw + rr) * nwarp + warp;
        let live = row < rows;
        let r = select(rows - 1u, row, live);
        let odd = (r & 1u) == 1u;
        var acc = 0.0;
        // Two slots at a time, {UD} words of each a lane per pass, all
        // loads issued before the first dot; per lane the words of a slot
        // still sum in order and the slots fold into `acc` in slot order.
        for (var s = 0u; s < d8_p.slots; s = s + 2u) {
            let s1 = min(s + 1u, d8_p.slots - 1u);
            let b0 = d8_ss[s] * d8_p.dsw;
            let b1 = d8_ss[s1] * d8_p.dsw;
            let w0 = b0 + r * iv4;
            let w1 = b1 + r * iv4;
            let x0 = s * iv4;
            let x1 = s1 * iv4;
            var p0 = 0.0;
            var p1 = 0.0;
            for (var kb = 0u; kb < iv4; kb = kb + {UD}u * ssz) {
{DN_LOAD}
{DN_DOT}
            }
            acc = acc + (d8_sw[s] * h2(d8_w[b0 + qw + (r >> 1u)], odd)) * p0;
            if (s + 1u < d8_p.slots) {
                acc = acc + (d8_sw[s1] * h2(d8_w[b1 + qw + (r >> 1u)], odd)) * p1;
            }
        }
        let t = subgroupAdd(acc);
        if (lane == 0u && live) { d8_y[row] = t; }
    }
}

// ── f32 router logits, one 32-lane workgroup a row ──
// `f32_matvec` walks a row with one dependent load a step (~18 us a layer
// for 64 x 2304 on an RTX 3090); here every lane issues its vec4 loads
// before the first dot.
struct RtP { rows: u32, cols: u32, _a: u32, _b: u32 };
@group(0) @binding(16) var<storage, read>       r8_w : array<vec4<f32>>;
@group(0) @binding(17) var<storage, read>       r8_x : array<vec4<f32>>;
@group(0) @binding(18) var<storage, read_write> r8_y : array<f32>;
@group(0) @binding(19) var<uniform>             r8_p : RtP;

@compute @workgroup_size(32)
fn m8_rt(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(subgroup_invocation_id) lane: u32,
         @builtin(subgroup_size) ssz: u32) {
    let row = wid.x;
    let c4 = r8_p.cols >> 2u;
    let b = row * c4;
    var acc = 0.0;
    for (var kb = 0u; kb < c4; kb = kb + {UR}u * ssz) {
{RT_LOAD}
{RT_DOT}
    }
    let t = subgroupAdd(acc);
    if (lane == 0u) { r8_y[row] = t; }
}
"#;

const PREFILL_WGSL_T: &str = r#"
// ── grouped gate + up + SiLU: a thread a row of expert wid.y, {NT} entries ──
@group(0) @binding(0) var<storage, read>       pg_g4  : array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       pg_u4  : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       pg_g1  : array<u32>;
@group(0) @binding(3) var<storage, read>       pg_u1  : array<u32>;
@group(0) @binding(4) var<storage, read>       pg_d1  : array<u32>;
@group(0) @binding(5) var<storage, read>       pg_x   : array<vec4<f32>>;
@group(0) @binding(6) var<storage, read>       pg_off : array<u32>;
@group(0) @binding(7) var<storage, read>       pg_ent : array<u32>;
@group(0) @binding(8) var<storage, read_write> pg_act : array<f32>;
@group(0) @binding(9) var<uniform>             pg_p   : M8P;
// [entry j][gate 0 | up 1][TW words] of x·col for the current tile.
var<workgroup> pg_s: array<vec4<f32>, {NT2TW}>;
var<workgroup> pg_beg: u32;
var<workgroup> pg_nt: u32;

@compute @workgroup_size(64)
fn m8g_gu(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
    let ex = wid.y;
    if (lid == 0u) {
        let b = pg_off[ex] + wid.z * {NT}u;
        let e = min(pg_off[ex + 1u], b + {NT}u);
        pg_beg = b;
        pg_nt = select(0u, e - b, e > b);
    }
    let nt = workgroupUniformLoad(&pg_nt);
    if (nt == 0u) { return; }
    let beg = workgroupUniformLoad(&pg_beg);
    let hv4 = pg_p.hidden >> 2u;
    let h16 = pg_p.hidden >> 4u;
    let rows = pg_p.inter;
    let row = wid.x * 64u + lid;
    let live = row < rows;
    let r = select(rows - 1u, row, live);
    let gb = ex * pg_p.gsw;
    let rsw = gb + rows * hv4;
    let csw = rsw + (rows >> 1u);
    let wb = (gb >> 2u) + r * h16;
{GU_INIT}
    for (var c0 = 0u; c0 < hv4; c0 = c0 + {TW}u) {
        workgroupBarrier();
        for (var i = lid; i < {NT2TW}u; i = i + 64u) {
            let j = i / {TW2}u;
            let m = (i / {TW}u) & 1u;
            let col = c0 + (i % {TW}u);
            var v = vec4<f32>(0.0);
            if (j < nt && col < hv4) {
                let xo = (pg_ent[beg + j] / pg_p.slots) * hv4 + col;
                let c = csw + 2u * col;
                if (m == 0u) {
                    v = pg_x[xo] * vec4<f32>(unpack2x16float(pg_g1[c]), unpack2x16float(pg_g1[c + 1u]));
                } else {
                    v = pg_x[xo] * vec4<f32>(unpack2x16float(pg_u1[c]), unpack2x16float(pg_u1[c + 1u]));
                }
            }
            pg_s[i] = v;
        }
        workgroupBarrier();
        for (var g = 0u; g < {TW4}u; g = g + 1u) {
            let gc = (c0 >> 2u) + g;
            if (gc >= h16) { break; }
            let wg = pg_g4[wb + gc];
            let wu = pg_u4[wb + gc];
            let g0 = q8x4(wg.x); let g1 = q8x4(wg.y); let g2 = q8x4(wg.z); let g3 = q8x4(wg.w);
            let u0 = q8x4(wu.x); let u1 = q8x4(wu.y); let u2 = q8x4(wu.z); let u3 = q8x4(wu.w);
            let so = g * 4u;
{GU_BODY}
        }
    }
    if (live) {
        let odd = (row & 1u) == 1u;
        let gs = h2(pg_g1[rsw + (row >> 1u)], odd);
        let us = h2(pg_u1[rsw + (row >> 1u)], odd);
        let dc = h2(pg_d1[ex * pg_p.dsw + pg_p.hidden * (rows >> 2u) + (pg_p.hidden >> 1u) + (row >> 1u)], odd);
{GU_OUT}
    }
}

// ── grouped down: a thread a hidden row of expert wid.y, per-entry outputs ──
@group(0) @binding(10) var<storage, read>       pd_w4  : array<vec4<u32>>;
@group(0) @binding(11) var<storage, read>       pd_w1  : array<u32>;
@group(0) @binding(12) var<storage, read>       pd_act : array<vec4<f32>>;
@group(0) @binding(13) var<storage, read>       pd_off : array<u32>;
@group(0) @binding(14) var<storage, read>       pd_ent : array<u32>;
@group(0) @binding(15) var<storage, read_write> pd_y   : array<f32>;
@group(0) @binding(16) var<uniform>             pd_p   : M8P;
var<workgroup> pd_s: array<vec4<f32>, {NTTW}>;
var<workgroup> pd_beg: u32;
var<workgroup> pd_nt: u32;

@compute @workgroup_size(64)
fn m8g_dn(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
    let ex = wid.y;
    if (lid == 0u) {
        let b = pd_off[ex] + wid.z * {NT}u;
        let e = min(pd_off[ex + 1u], b + {NT}u);
        pd_beg = b;
        pd_nt = select(0u, e - b, e > b);
    }
    let nt = workgroupUniformLoad(&pd_nt);
    if (nt == 0u) { return; }
    let beg = workgroupUniformLoad(&pd_beg);
    let iv4 = pd_p.inter >> 2u;
    let i16 = pd_p.inter >> 4u;
    let rows = pd_p.hidden;
    let row = wid.x * 64u + lid;
    let live = row < rows;
    let r = select(rows - 1u, row, live);
    let db = ex * pd_p.dsw;
    let wb = (db >> 2u) + r * i16;
{DN_INIT}
    for (var c0 = 0u; c0 < iv4; c0 = c0 + {TW}u) {
        workgroupBarrier();
        for (var i = lid; i < {NTTW}u; i = i + 64u) {
            let j = i / {TW}u;
            let col = c0 + (i % {TW}u);
            var v = vec4<f32>(0.0);
            if (j < nt && col < iv4) { v = pd_act[pd_ent[beg + j] * iv4 + col]; }
            pd_s[i] = v;
        }
        workgroupBarrier();
        for (var g = 0u; g < {TW4}u; g = g + 1u) {
            let gc = (c0 >> 2u) + g;
            if (gc >= i16) { break; }
            let w = pd_w4[wb + gc];
            let a0 = q8x4(w.x); let a1 = q8x4(w.y); let a2 = q8x4(w.z); let a3 = q8x4(w.w);
            let so = g * 4u;
{DN_BODY}
        }
    }
    if (live) {
        let sc = h2(pd_w1[db + rows * iv4 + (row >> 1u)], (row & 1u) == 1u);
{DN_OUT}
    }
}
"#;

fn prefill_wgsl() -> String {
    let (mut gi, mut gb, mut go, mut di, mut dbody, mut dout) = (
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    );
    let nt = nt();
    // The entry count is workgroup-uniform: an entry past it costs nothing
    // (at a 32-token chunk an expert holds about four of the eight).
    for j in 0..nt {
        gi.push_str(&format!(
            "    let e{j} = pg_ent[beg + min({j}u, nt - 1u)];\n    var ag{j} = 0.0; var au{j} = 0.0;\n"
        ));
        let sg = j * 2 * TW;
        let su = sg + TW;
        gb.push_str(&format!(
            "            if ({j}u < nt) {{\n            ag{j} = ag{j} + dot(g0, pg_s[{sg}u + so]) + dot(g1, pg_s[{sg}u + so + 1u])\n                  + dot(g2, pg_s[{sg}u + so + 2u]) + dot(g3, pg_s[{sg}u + so + 3u]);\n            au{j} = au{j} + dot(u0, pg_s[{su}u + so]) + dot(u1, pg_s[{su}u + so + 1u])\n                  + dot(u2, pg_s[{su}u + so + 2u]) + dot(u3, pg_s[{su}u + so + 3u]);\n            }}\n"
        ));
        go.push_str(&format!(
            "        if ({j}u < nt) {{ let gg = ag{j} * gs; let uu = au{j} * us; pg_act[e{j} * rows + row] = ((gg / (1.0 + exp(-gg))) * uu) * dc; }}\n"
        ));
        di.push_str(&format!(
            "    let e{j} = pd_ent[beg + min({j}u, nt - 1u)];\n    var ac{j} = 0.0;\n"
        ));
        let sd = j * TW;
        dbody.push_str(&format!(
            "            if ({j}u < nt) {{ ac{j} = ac{j} + dot(a0, pd_s[{sd}u + so]) + dot(a1, pd_s[{sd}u + so + 1u])\n                  + dot(a2, pd_s[{sd}u + so + 2u]) + dot(a3, pd_s[{sd}u + so + 3u]); }}\n"
        ));
        dout.push_str(&format!(
            "        if ({j}u < nt) {{ pd_y[e{j} * rows + row] = ac{j} * sc; }}\n"
        ));
    }
    let body = PREFILL_WGSL_T
        .replace("{GU_INIT}", &gi)
        .replace("{GU_BODY}", &gb)
        .replace("{GU_OUT}", &go)
        .replace("{DN_INIT}", &di)
        .replace("{DN_BODY}", &dbody)
        .replace("{DN_OUT}", &dout)
        .replace("{NT2TW}", &(nt * 2 * TW).to_string())
        .replace("{NTTW}", &(nt * TW).to_string())
        .replace("{TW2}", &(2 * TW).to_string())
        .replace("{TW4}", &(TW / 4).to_string())
        .replace("{TW}", &TW.to_string())
        .replace("{NT}", &nt.to_string());
    format!("{COMMON_WGSL}{body}")
}

/// Words a lane issues per pass: gate/up (`CMF_M8_UGU`), down
/// (`CMF_M8_UDN`, per slot of a pair) and router vec4s (`CMF_M8_URT`).
fn unroll(key: &str, def: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| (1..=32).contains(v))
        .unwrap_or(def)
}
fn unrolls() -> (usize, usize, usize) {
    static V: OnceLock<(usize, usize, usize)> = OnceLock::new();
    *V.get_or_init(|| (unroll("CMF_M8_UGU", 6), unroll("CMF_M8_UDN", 8), unroll("CMF_M8_URT", 6)))
}

fn decode_wgsl(hidden: usize) -> String {
    let (ug, ud, ur) = unrolls();
    let (mut gl, mut gd, mut dl, mut dd, mut rl, mut rd) = (
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    );
    for j in 0..ug {
        gl.push_str(&format!(
            "            let k{j} = kb + lane + {j}u * ssz;\n            let a{j} = g8_gw[w0 + k{j}];\n            let b{j} = g8_uw[w0 + k{j}];\n"
        ));
        gd.push_str(&format!(
            "            if (k{j} < hv4) {{ ag = ag + dot(q8x4(a{j}), g8_xg[k{j}]); au = au + dot(q8x4(b{j}), g8_xu[k{j}]); }}\n"
        ));
    }
    for j in 0..ud {
        dl.push_str(&format!(
            "                let k{j} = kb + lane + {j}u * ssz;\n                let c{j} = d8_w[w0 + k{j}];\n                let e{j} = d8_w[w1 + k{j}];\n"
        ));
        dd.push_str(&format!(
            "                if (k{j} < iv4) {{ p0 = p0 + dot(q8x4(c{j}), d8_act[x0 + k{j}]); p1 = p1 + dot(q8x4(e{j}), d8_act[x1 + k{j}]); }}\n"
        ));
    }
    for j in 0..ur {
        rl.push_str(&format!(
            "        let k{j} = kb + lane + {j}u * ssz;\n        let w{j} = r8_w[b + k{j}];\n        let x{j} = r8_x[k{j}];\n"
        ));
        rd.push_str(&format!(
            "        if (k{j} < c4) {{ acc = acc + dot(w{j}, x{j}); }}\n"
        ));
    }
    let body = DECODE_WGSL
        .replace("{HV4}", &(hidden / 4).to_string())
        .replace("{GU_LOAD}", &gl)
        .replace("{GU_DOT}", &gd)
        .replace("{DN_LOAD}", &dl)
        .replace("{DN_DOT}", &dd)
        .replace("{RT_LOAD}", &rl)
        .replace("{RT_DOT}", &rd)
        .replace("{UG}", &ug.to_string())
        .replace("{UD}", &ud.to_string())
        .replace("{UR}", &ur.to_string());
    format!("{COMMON_WGSL}{body}")
}

type Pl = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

fn build(c: &Ctx, label: &str, src: &str, eps: &[&str]) -> Option<Vec<Pl>> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let ps: Vec<wgpu::ComputePipeline> = eps
        .iter()
        .map(|ep| {
            c.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(ep),
                    layout: None,
                    module: &module,
                    entry_point: Some(ep),
                    compilation_options: Default::default(),
                    cache: c.pipeline_cache.as_ref(),
                })
        })
        .collect();
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("{label} module rejected ({e}): q8_2f experts stay off the graph");
        return None;
    }
    Some(
        ps.into_iter()
            .map(|p| {
                let l = p.get_bind_group_layout(0);
                (p, l)
            })
            .collect(),
    )
}

pub(crate) struct Dec {
    pub(crate) gu: wgpu::ComputePipeline,
    gu_l: wgpu::BindGroupLayout,
    pub(crate) dn: wgpu::ComputePipeline,
    dn_l: wgpu::BindGroupLayout,
    pub(crate) rt: wgpu::ComputePipeline,
    rt_l: wgpu::BindGroupLayout,
}

pub(crate) struct Pre {
    pub(crate) gu: wgpu::ComputePipeline,
    gu_l: wgpu::BindGroupLayout,
    pub(crate) dn: wgpu::ComputePipeline,
    dn_l: wgpu::BindGroupLayout,
}

type DecMap = Mutex<HashMap<(usize, usize), Option<Arc<Dec>>>>;
type PreMap = Mutex<HashMap<usize, Option<Arc<Pre>>>>;

/// The decode pair for this `hidden` (its workgroup staging is sized by it),
/// or None: no 32-lane subgroups, the staging over the device's workgroup
/// memory, or the module rejected.
pub(crate) fn decode(c: &Ctx, hidden: usize) -> Option<Arc<Dec>> {
    static M: OnceLock<DecMap> = OnceLock::new();
    let key = (c as *const Ctx as usize, hidden);
    let mut m = M.get_or_init(Default::default).lock().unwrap();
    if let Some(v) = m.get(&key) {
        return v.clone();
    }
    let smem = 2 * (hidden / 4) * 16 + 64 * 4 + 4;
    let v = (sg32(c)
        && hidden % 16 == 0
        && smem as u32 <= c.device.limits().max_compute_workgroup_storage_size)
        .then(|| build(c, "cmf-moe-q82", &decode_wgsl(hidden), &["m8_gu", "m8_dn", "m8_rt"]))
        .flatten()
        .map(|mut v| {
            let (rt, rt_l) = v.pop().unwrap();
            let (dn, dn_l) = v.pop().unwrap();
            let (gu, gu_l) = v.pop().unwrap();
            Arc::new(Dec { gu, gu_l, dn, dn_l, rt, rt_l })
        });
    m.insert(key, v.clone());
    v
}

/// The grouped prefill pair (`moe_r4`'s `moeg_group` / `moeg_sum` bucket
/// and mix around it); `CMF_MOEG=0` turns it off with the q4tp one.
pub(crate) fn prefill(c: &Ctx) -> Option<Arc<Pre>> {
    static M: OnceLock<PreMap> = OnceLock::new();
    if std::env::var("CMF_MOEG").as_deref() == Ok("0") {
        return None;
    }
    let key = c as *const Ctx as usize;
    let mut m = M.get_or_init(Default::default).lock().unwrap();
    if let Some(v) = m.get(&key) {
        return v.clone();
    }
    let v = build(c, "cmf-moeg-q82", &prefill_wgsl(), &["m8g_gu", "m8g_dn"]).map(|mut v| {
        let (dn, dn_l) = v.pop().unwrap();
        let (gu, gu_l) = v.pop().unwrap();
        Arc::new(Pre { gu, gu_l, dn, dn_l })
    });
    m.insert(key, v.clone());
    v
}

fn params(c: &Ctx, hidden: usize, inter: usize, slots: usize, fold: u32, rpw: u32) -> wgpu::Buffer {
    uniform_u32x8(
        c,
        [
            hidden as u32,
            inter as u32,
            slots as u32,
            (blob_bytes(inter, hidden) / 4) as u32,
            (blob_bytes(hidden, inter) / 4) as u32,
            fold,
            rpw,
            0,
        ],
    )
}

/// One decode layer's bindings and grids.
pub(crate) struct DecodeJob {
    pub(crate) gu_bg: wgpu::BindGroup,
    pub(crate) dn_bg: wgpu::BindGroup,
    pub(crate) gu_x: u32,
    pub(crate) dn_x: u32,
}

/// `fold` = (n_exp, top_k, renorm, routed scale) for the folded softmax
/// select (no bias, no shared expert, n_exp <= 64); None reads `sel`/`wt`
/// written by `moe_select`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_job(
    c: &Ctx,
    d: &Dec,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    down: &wgpu::Buffer,
    x: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    wt: &wgpu::Buffer,
    act: &wgpu::Buffer,
    y: &wgpu::Buffer,
    logits: &wgpu::Buffer,
    hidden: usize,
    inter: usize,
    slots: usize,
    fold: Option<(usize, usize, bool, f32)>,
) -> DecodeJob {
    let fold_n = fold.map_or(0, |f| f.0 as u32);
    let (rg, rd) = (rpw_gu(), rpw_dn());
    let gu_u = params(c, hidden, inter, slots, fold_n, rg);
    let dn_u = params(c, hidden, inter, slots, fold_n, rd);
    let fd_u = uniform_u32x4(
        c,
        match fold {
            Some((n, k, norm, scale)) => [n as u32, k as u32, u32::from(norm), scale.to_bits()],
            None => [0, 0, 0, 1.0f32.to_bits()],
        },
    );
    let gu_bg = bind_pairs(
        c,
        &d.gu_l,
        &[
            (0, gate),
            (1, up),
            (2, down),
            (3, x),
            (4, sel),
            (5, act),
            (6, &gu_u),
            (7, logits),
        ],
    );
    let dn_bg = bind_pairs(
        c,
        &d.dn_l,
        &[
            (8, down),
            (9, act),
            (10, sel),
            (11, wt),
            (12, y),
            (13, &dn_u),
            (14, &fd_u),
            (15, logits),
        ],
    );
    // Eight 32-lane warps a workgroup (`decode` admits 32-lane subgroups only).
    DecodeJob {
        gu_bg,
        dn_bg,
        gu_x: (inter as u32).div_ceil(8 * rg),
        dn_x: (hidden as u32).div_ceil(8 * rd),
    }
}

/// The warp-a-row f32 router (`CMF_M8_ROUTER=0` keeps `f32_matvec`):
/// bind group and grid, or None when the router is outside it (cols % 4).
pub(crate) fn router_job(
    c: &Ctx,
    d: &Dec,
    w: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> Option<(wgpu::BindGroup, u32)> {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("CMF_M8_ROUTER").as_deref() != Ok("0"))
        || cols % 4 != 0
        || rows == 0
        || rows as u32 > MAX_WG
    {
        return None;
    }
    let u = uniform_u32x4(c, [rows as u32, cols as u32, 0, 0]);
    let bg = bind_pairs(c, &d.rt_l, &[(16, w), (17, x), (18, y), (19, &u)]);
    Some((bg, rows as u32))
}

/// One prefill layer's grouped bindings and grids (z = entry groups).
pub(crate) struct PrefillJob {
    pub(crate) gu_bg: wgpu::BindGroup,
    pub(crate) dn_bg: wgpu::BindGroup,
    pub(crate) gu_grid: (u32, u32, u32),
    pub(crate) dn_grid: (u32, u32, u32),
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_job(
    c: &Ctx,
    p: &Pre,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    down: &wgpu::Buffer,
    x: &wgpu::Buffer,
    off: &wgpu::Buffer,
    ent: &wgpu::Buffer,
    act: &wgpu::Buffer,
    ypart: &wgpu::Buffer,
    hidden: usize,
    inter: usize,
    slots: usize,
    n_exp: usize,
    k: usize,
) -> PrefillJob {
    let u = params(c, hidden, inter, slots, 0, 0);
    let gu_bg = bind_pairs(
        c,
        &p.gu_l,
        &[
            (0, gate),
            (1, up),
            (2, gate),
            (3, up),
            (4, down),
            (5, x),
            (6, off),
            (7, ent),
            (8, act),
            (9, &u),
        ],
    );
    let dn_bg = bind_pairs(
        c,
        &p.dn_l,
        &[
            (10, down),
            (11, down),
            (12, act),
            (13, off),
            (14, ent),
            (15, ypart),
            (16, &u),
        ],
    );
    let z = k.div_ceil(nt()) as u32;
    PrefillJob {
        gu_bg,
        dn_bg,
        gu_grid: ((inter as u32).div_ceil(64), n_exp as u32, z),
        dn_grid: ((hidden as u32).div_ceil(64), n_exp as u32, z),
    }
}
