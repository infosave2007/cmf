//! Batch-graph kernels for Gemma-4 (the humanizer rewriter, a 12B dense
//! model): its prompt ingest on the batched wgpu graph.
//!
//! The token graph learned Gemma-4 first (512-wide global heads over
//! 256-wide sliding ones, the scale-less V norm, sandwich norms, the layer
//! scalar, tanh-GELU); the batch graph refused all of it, so a prompt went
//! through the host-driven chunk walk at ~91 tok/s on an RTX PRO 4000 —
//! a 12B dense stack whose GEMMs alone allow several hundred. These are
//! the pieces the batch graph was missing, each with a TOKEN AXIS so one
//! dispatch serves the whole chunk:
//!
//! * `g4_rope_bt` — Q/K norm + RoPE of every row (heads up to 512 wide),
//!   and the V norm of every row's V heads in the same dispatch. Per head
//!   it is `attn_rope_qkn`'s arithmetic, expression for expression.
//! * `g4_kv_bt` — every row's K/V into the (ring) mirror.
//! * `g4_attend_bt` — causal (windowed) attention, one workgroup per (query
//!   head, block of 8 tokens): each K row is read once for the block's
//!   eight queries and each V row once for their eight value sums.
//! * `g4_add_norm_b` — h = (h + d)·layer_scalar, then rms(h)·w, per row.
//! * `q8_2f_mm_r2` — a register-blocked int8 GEMM for the q8_2f
//!   projections (the batch graph's q8_2f arm streamed each weight word
//!   for eight rows at a time; a 128-row chunk read the 12 GB of weights
//!   sixteen times). 64×64 tiles, `q4tp_mm_r2`'s staging and FMA order,
//!   exact f32 (the column field folds into the staged activations, the
//!   row scale multiplies the finished sum — the decode kernel's factors).
//!
//! `CMF_G4_BATCH=0` keeps Gemma-4 on the chunked host prefill;
//! `CMF_G4_MM=0` keeps the batch graph's eight-row q8_2f arm.

use super::*;

pub(crate) const G4_WGSL: &str = r#"
// ── Q/K norm + RoPE with a token axis (wid.y = token t of a run starting
// at position `pos`), and the scale-less V norm of the same rows. Heads
// [0, nh) are Q, [nh, nh+nkv) K, [nh+nkv, nh+2·nkv) V (when `vnorm`).
struct GrP {
    nh: u32, nkv: u32, hd: u32, rd: u32,
    pos: u32, flags: u32, eps: f32, dv: u32,
    rope_scale: f32, vnorm: u32, _p1: u32, _p2: u32,
};
@group(0) @binding(0) var<storage, read>       gr_qraw : array<f32>;
@group(0) @binding(1) var<storage, read_write> gr_k    : array<f32>;
@group(0) @binding(2) var<storage, read_write> gr_qout : array<f32>;
@group(0) @binding(3) var<storage, read_write> gr_v    : array<f32>;
@group(0) @binding(4) var<storage, read>       gr_qnw  : array<f32>;
@group(0) @binding(5) var<storage, read>       gr_knw  : array<f32>;
@group(0) @binding(6) var<storage, read>       gr_invf : array<f32>;
@group(0) @binding(7) var<uniform>             gr_p    : GrP;
var<workgroup> gr_red: array<f32, 32>;
var<workgroup> gr_head: array<f32, 512>;
@compute @workgroup_size(32)
fn g4_rope_bt(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let t = wid.y;
    let lane = lid.x;
    let nh = gr_p.nh;
    let nkv = gr_p.nkv;
    let nvh = select(0u, nkv, gr_p.vnorm != 0u);
    if (head >= nh + nkv + nvh) { return; }
    if (head >= nh + nkv) {
        // V norm: rms over dv, no weight, no rotation — the token graph's
        // attn_rope_qkn pass over the V heads with unit weights.
        let dv = gr_p.dv;
        let vbase = (t * nkv + (head - nh - nkv)) * dv;
        let ntv = (dv + 31u) / 32u;
        var vv: array<f32, 16>;
        var vs = 0.0;
        for (var i = 0u; i < ntv; i = i + 1u) {
            let d = i * 32u + lane;
            var val = 0.0;
            if (d < dv) { val = gr_v[vbase + d]; }
            vv[i] = val;
            vs = vs + val * val;
        }
        gr_red[lane] = vs;
        workgroupBarrier();
        var vst = 16u;
        loop {
            if (vst == 0u) { break; }
            if (lane < vst) { gr_red[lane] = gr_red[lane] + gr_red[lane + vst]; }
            workgroupBarrier();
            vst = vst / 2u;
        }
        let vinv = 1.0 / sqrt(gr_red[0] / f32(dv) + gr_p.eps);
        for (var i = 0u; i < ntv; i = i + 1u) {
            let d = i * 32u + lane;
            if (d < dv) { gr_v[vbase + d] = vv[i] * vinv * 1.0; }
        }
        return;
    }
    let hd = gr_p.hd;
    let pos = gr_p.pos + t;
    let isq = head < nh;
    let src_base = select((head - nh) * hd, head * hd, isq);
    let qoff = t * nh * hd;
    let koff = t * nkv * hd;
    let nt = (hd + 31u) / 32u;
    var xv: array<f32, 16>;
    var ss = 0.0;
    for (var i = 0u; i < nt; i = i + 1u) {
        let d = i * 32u + lane;
        var val = 0.0;
        if (d < hd) { val = select(gr_k[koff + src_base + d], gr_qraw[qoff + src_base + d], isq); }
        xv[i] = val;
        ss = ss + val * val;
    }
    gr_red[lane] = ss;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) { gr_red[lane] = gr_red[lane] + gr_red[lane + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let normed = select((gr_p.flags & 4u) != 0u, (gr_p.flags & 2u) != 0u, isq);
    let late = (gr_p.flags & 32u) != 0u;
    let hlf = gr_p.rd / 2u;
    for (var i = 0u; i < nt; i = i + 1u) {
        let d = i * 32u + lane;
        if (d < hd) { gr_head[d] = xv[i]; }
    }
    workgroupBarrier();
    if (late) {
        var ri0 = lane;
        loop {
            if (ri0 >= hlf) { break; }
            let angle0 = f32(pos) * gr_invf[ri0];
            let cc0 = cos(angle0);
            let sf0 = sin(angle0);
            let y0 = gr_head[ri0];
            let y1 = gr_head[ri0 + hlf];
            gr_head[ri0] = (y0 * cc0 - y1 * sf0) * gr_p.rope_scale;
            gr_head[ri0 + hlf] = (y0 * sf0 + y1 * cc0) * gr_p.rope_scale;
            ri0 = ri0 + 32u;
        }
    }
    workgroupBarrier();
    for (var i = 0u; i < nt; i = i + 1u) {
        let d = i * 32u + lane;
        if (d < hd) { xv[i] = gr_head[d]; }
    }
    workgroupBarrier();
    if (normed) {
        let inv = 1.0 / sqrt(gr_red[0] / f32(hd) + gr_p.eps);
        let gemma = (gr_p.flags & 8u) != 0u;
        for (var i = 0u; i < nt; i = i + 1u) {
            let d = i * 32u + lane;
            if (d < hd) {
                var wd = select(gr_knw[d], gr_qnw[d], isq);
                if (gemma) { wd = 1.0 + wd; }
                xv[i] = xv[i] * inv * wd;
            }
        }
    }
    for (var i = 0u; i < nt; i = i + 1u) {
        let d = i * 32u + lane;
        if (d < hd) { gr_head[d] = xv[i]; }
    }
    workgroupBarrier();
    if (!late) {
        var ri = lane;
        loop {
            if (ri >= hlf) { break; }
            let angle = f32(pos) * gr_invf[ri];
            let cc = cos(angle);
            let sfac = sin(angle);
            let x0 = gr_head[ri];
            let x1 = gr_head[ri + hlf];
            gr_head[ri] = (x0 * cc - x1 * sfac) * gr_p.rope_scale;
            gr_head[ri + hlf] = (x0 * sfac + x1 * cc) * gr_p.rope_scale;
            ri = ri + 32u;
        }
    }
    workgroupBarrier();
    let dst_base = select((head - nh) * hd, head * hd, isq);
    for (var i = 0u; i < nt; i = i + 1u) {
        let d = i * 32u + lane;
        if (d < hd) {
            if (isq) { gr_qout[qoff + dst_base + d] = gr_head[d]; } else { gr_k[koff + dst_base + d] = gr_head[d]; }
        }
    }
}

// ── Every row's K (hd wide) and V (dv wide) heads into mirror row
// (pos + t) % cap (gid.y = token t).
struct GkP { nkv: u32, hd: u32, dv: u32, cap: u32, pos: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(8)  var<storage, read>       gk_k  : array<f32>;
@group(0) @binding(9)  var<storage, read>       gk_v  : array<f32>;
@group(0) @binding(10) var<storage, read_write> gk_kc : array<f32>;
@group(0) @binding(11) var<storage, read_write> gk_vc : array<f32>;
@group(0) @binding(12) var<uniform>             gk_p  : GkP;
@compute @workgroup_size(256)
fn g4_kv_bt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    let slot = (gk_p.pos + t) % gk_p.cap;
    let kw = gk_p.nkv * gk_p.hd;
    if (i < kw) {
        let h = i / gk_p.hd;
        gk_kc[(h * gk_p.cap + slot) * gk_p.hd + (i % gk_p.hd)] = gk_k[t * kw + i];
    }
    let vw = gk_p.nkv * gk_p.dv;
    if (i < vw) {
        let h = i / gk_p.dv;
        gk_vc[(h * gk_p.cap + slot) * gk_p.dv + (i % gk_p.dv)] = gk_v[t * vw + i];
    }
}

// ── Causal (windowed) attention of a token block: workgroup (head h, block
// wid.y of 8 tokens t = 8·wid.y + j, position pos0 + t). Token t sees the
// mirror rows of positions [t_lo, pos0 + t] (t_lo = pos0 + t + 1 - win for
// a sliding layer, else 0), row p at slot p % cap. Positions walk in
// 256-wide chunks over the block's union range. Scores: lanes are
// positions, each K row read once for the block's eight queries. Values:
// lanes are (position quarter, dim quad) — a vec4 of V for eight tokens a
// load — and the quarters' partial sums meet once, at the end.
// `g4_attend_bt` serves dv <= 256, `g4_attend_bt2` dv <= 512 (second quad
// set at +64): the 512-wide accumulators would halve the narrow layers'
// occupancy.
struct GaP {
    nh: u32, hpk: u32, hd: u32, dv: u32,
    cap: u32, pos0: u32, win: u32, scale: f32,
    k: u32, _a: u32, _b: u32, _c: u32,
};
@group(0) @binding(13) var<storage, read>       ga_q : array<vec4<f32>>;
@group(0) @binding(14) var<storage, read>       ga_k : array<vec4<f32>>;
@group(0) @binding(15) var<storage, read>       ga_v : array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> ga_o : array<vec4<f32>>;
@group(0) @binding(17) var<uniform>             ga_p : GaP;
var<workgroup> ga_sq: array<vec4<f32>, 1024>;  // [j·hd4 + d4]; then the quarter sums
var<workgroup> ga_w: array<vec4<f32>, 512>;    // [p·2 + j/4], lanes j%4
var<workgroup> ga_red: array<f32, 2048>;       // [j·256 + lane]
@compute @workgroup_size(256)
fn g4_attend_bt(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let tb0 = wid.y * 8u;
    let nh = ga_p.nh;
    if (h >= nh || tb0 >= ga_p.k) { return; }
    let nb = min(8u, ga_p.k - tb0);
    let g = h / ga_p.hpk;
    let hd4 = ga_p.hd / 4u;
    let dv4 = ga_p.dv / 4u;
    let cap = ga_p.cap;
    let win = ga_p.win;
    let scale = ga_p.scale;
    for (var i = lid; i < 8u * hd4; i = i + 256u) {
        let j = i / hd4;
        let d = i - j * hd4;
        var qv = vec4<f32>(0.0);
        if (j < nb) { qv = ga_q[((tb0 + j) * nh + h) * hd4 + d]; }
        ga_sq[i] = qv;
    }
    workgroupBarrier();
    let hi0 = ga_p.pos0 + tb0;          // token 0's position
    let p_last = hi0 + nb - 1u;
    var p_first = 0u;
    if (win != 0u && hi0 + 1u > win) { p_first = hi0 + 1u - win; }
    let kbase4 = g * cap * hd4;
    let vbase4 = g * cap * dv4;
    let pg = lid >> 6u;
    let q4 = lid & 63u;
    let vlive = q4 < dv4;
    // Per-token softmax frame and value sums, unrolled by hand (constant
    // indices only: the accumulators must stay in registers).
    var m0 = -1.0e30; var l0 = 0.0; var ac0 = vec4<f32>(0.0);
    var m1 = -1.0e30; var l1 = 0.0; var ac1 = vec4<f32>(0.0);
    var m2 = -1.0e30; var l2 = 0.0; var ac2 = vec4<f32>(0.0);
    var m3 = -1.0e30; var l3 = 0.0; var ac3 = vec4<f32>(0.0);
    var m4 = -1.0e30; var l4 = 0.0; var ac4 = vec4<f32>(0.0);
    var m5 = -1.0e30; var l5 = 0.0; var ac5 = vec4<f32>(0.0);
    var m6 = -1.0e30; var l6 = 0.0; var ac6 = vec4<f32>(0.0);
    var m7 = -1.0e30; var l7 = 0.0; var ac7 = vec4<f32>(0.0);
    var c0 = p_first;
    loop {
        if (c0 > p_last) { break; }
        let cn = min(256u, p_last + 1u - c0);
        let p = c0 + lid;
        var s0 = -1.0e30;
        var s1 = -1.0e30;
        var s2 = -1.0e30;
        var s3 = -1.0e30;
        var s4 = -1.0e30;
        var s5 = -1.0e30;
        var s6 = -1.0e30;
        var s7 = -1.0e30;
        var okm = 0u;
        if (lid < cn) {
            let krow = kbase4 + (p % cap) * hd4;
            var d0 = vec4<f32>(0.0);
            var d1 = vec4<f32>(0.0);
            var d2 = vec4<f32>(0.0);
            var d3 = vec4<f32>(0.0);
            var d4 = vec4<f32>(0.0);
            var d5 = vec4<f32>(0.0);
            var d6 = vec4<f32>(0.0);
            var d7 = vec4<f32>(0.0);
            for (var d = 0u; d < hd4; d = d + 1u) {
                let kv = ga_k[krow + d];
                d0 = d0 + ga_sq[0u * hd4 + d] * kv;
                d1 = d1 + ga_sq[1u * hd4 + d] * kv;
                d2 = d2 + ga_sq[2u * hd4 + d] * kv;
                d3 = d3 + ga_sq[3u * hd4 + d] * kv;
                d4 = d4 + ga_sq[4u * hd4 + d] * kv;
                d5 = d5 + ga_sq[5u * hd4 + d] * kv;
                d6 = d6 + ga_sq[6u * hd4 + d] * kv;
                d7 = d7 + ga_sq[7u * hd4 + d] * kv;
            }
            let tp0 = hi0 + 0u; if (0u < nb && p <= tp0 && (win == 0u || p + win > tp0)) { s0 = (d0.x + d0.y + d0.z + d0.w) * scale; okm = okm | (1u << 0u); }
            let tp1 = hi0 + 1u; if (1u < nb && p <= tp1 && (win == 0u || p + win > tp1)) { s1 = (d1.x + d1.y + d1.z + d1.w) * scale; okm = okm | (1u << 1u); }
            let tp2 = hi0 + 2u; if (2u < nb && p <= tp2 && (win == 0u || p + win > tp2)) { s2 = (d2.x + d2.y + d2.z + d2.w) * scale; okm = okm | (1u << 2u); }
            let tp3 = hi0 + 3u; if (3u < nb && p <= tp3 && (win == 0u || p + win > tp3)) { s3 = (d3.x + d3.y + d3.z + d3.w) * scale; okm = okm | (1u << 3u); }
            let tp4 = hi0 + 4u; if (4u < nb && p <= tp4 && (win == 0u || p + win > tp4)) { s4 = (d4.x + d4.y + d4.z + d4.w) * scale; okm = okm | (1u << 4u); }
            let tp5 = hi0 + 5u; if (5u < nb && p <= tp5 && (win == 0u || p + win > tp5)) { s5 = (d5.x + d5.y + d5.z + d5.w) * scale; okm = okm | (1u << 5u); }
            let tp6 = hi0 + 6u; if (6u < nb && p <= tp6 && (win == 0u || p + win > tp6)) { s6 = (d6.x + d6.y + d6.z + d6.w) * scale; okm = okm | (1u << 6u); }
            let tp7 = hi0 + 7u; if (7u < nb && p <= tp7 && (win == 0u || p + win > tp7)) { s7 = (d7.x + d7.y + d7.z + d7.w) * scale; okm = okm | (1u << 7u); }
        }
        ga_red[0u * 256u + lid] = s0;
        ga_red[1u * 256u + lid] = s1;
        ga_red[2u * 256u + lid] = s2;
        ga_red[3u * 256u + lid] = s3;
        ga_red[4u * 256u + lid] = s4;
        ga_red[5u * 256u + lid] = s5;
        ga_red[6u * 256u + lid] = s6;
        ga_red[7u * 256u + lid] = s7;
        workgroupBarrier();
        var st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) {
                ga_red[0u * 256u + lid] = max(ga_red[0u * 256u + lid], ga_red[0u * 256u + lid + st]);
                ga_red[1u * 256u + lid] = max(ga_red[1u * 256u + lid], ga_red[1u * 256u + lid + st]);
                ga_red[2u * 256u + lid] = max(ga_red[2u * 256u + lid], ga_red[2u * 256u + lid + st]);
                ga_red[3u * 256u + lid] = max(ga_red[3u * 256u + lid], ga_red[3u * 256u + lid + st]);
                ga_red[4u * 256u + lid] = max(ga_red[4u * 256u + lid], ga_red[4u * 256u + lid + st]);
                ga_red[5u * 256u + lid] = max(ga_red[5u * 256u + lid], ga_red[5u * 256u + lid + st]);
                ga_red[6u * 256u + lid] = max(ga_red[6u * 256u + lid], ga_red[6u * 256u + lid + st]);
                ga_red[7u * 256u + lid] = max(ga_red[7u * 256u + lid], ga_red[7u * 256u + lid + st]);
            }
            workgroupBarrier();
            st = st >> 1u;
        }
        let mp0 = max(m0, ga_red[0u * 256u]);
        let mp1 = max(m1, ga_red[1u * 256u]);
        let mp2 = max(m2, ga_red[2u * 256u]);
        let mp3 = max(m3, ga_red[3u * 256u]);
        let mp4 = max(m4, ga_red[4u * 256u]);
        let mp5 = max(m5, ga_red[5u * 256u]);
        let mp6 = max(m6, ga_red[6u * 256u]);
        let mp7 = max(m7, ga_red[7u * 256u]);
        workgroupBarrier();
        let w0 = select(0.0, exp(s0 - mp0), (okm & (1u << 0u)) != 0u); ga_red[0u * 256u + lid] = w0;
        let w1 = select(0.0, exp(s1 - mp1), (okm & (1u << 1u)) != 0u); ga_red[1u * 256u + lid] = w1;
        let w2 = select(0.0, exp(s2 - mp2), (okm & (1u << 2u)) != 0u); ga_red[2u * 256u + lid] = w2;
        let w3 = select(0.0, exp(s3 - mp3), (okm & (1u << 3u)) != 0u); ga_red[3u * 256u + lid] = w3;
        let w4 = select(0.0, exp(s4 - mp4), (okm & (1u << 4u)) != 0u); ga_red[4u * 256u + lid] = w4;
        let w5 = select(0.0, exp(s5 - mp5), (okm & (1u << 5u)) != 0u); ga_red[5u * 256u + lid] = w5;
        let w6 = select(0.0, exp(s6 - mp6), (okm & (1u << 6u)) != 0u); ga_red[6u * 256u + lid] = w6;
        let w7 = select(0.0, exp(s7 - mp7), (okm & (1u << 7u)) != 0u); ga_red[7u * 256u + lid] = w7;
        ga_w[lid * 2u] = vec4<f32>(w0, w1, w2, w3);
        ga_w[lid * 2u + 1u] = vec4<f32>(w4, w5, w6, w7);
        workgroupBarrier();
        st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) {
                ga_red[0u * 256u + lid] = ga_red[0u * 256u + lid] + ga_red[0u * 256u + lid + st];
                ga_red[1u * 256u + lid] = ga_red[1u * 256u + lid] + ga_red[1u * 256u + lid + st];
                ga_red[2u * 256u + lid] = ga_red[2u * 256u + lid] + ga_red[2u * 256u + lid + st];
                ga_red[3u * 256u + lid] = ga_red[3u * 256u + lid] + ga_red[3u * 256u + lid + st];
                ga_red[4u * 256u + lid] = ga_red[4u * 256u + lid] + ga_red[4u * 256u + lid + st];
                ga_red[5u * 256u + lid] = ga_red[5u * 256u + lid] + ga_red[5u * 256u + lid + st];
                ga_red[6u * 256u + lid] = ga_red[6u * 256u + lid] + ga_red[6u * 256u + lid + st];
                ga_red[7u * 256u + lid] = ga_red[7u * 256u + lid] + ga_red[7u * 256u + lid + st];
            }
            workgroupBarrier();
            st = st >> 1u;
        }
        let f0 = exp(m0 - mp0); l0 = l0 * f0 + ga_red[0u * 256u]; ac0 = ac0 * f0; m0 = mp0;
        let f1 = exp(m1 - mp1); l1 = l1 * f1 + ga_red[1u * 256u]; ac1 = ac1 * f1; m1 = mp1;
        let f2 = exp(m2 - mp2); l2 = l2 * f2 + ga_red[2u * 256u]; ac2 = ac2 * f2; m2 = mp2;
        let f3 = exp(m3 - mp3); l3 = l3 * f3 + ga_red[3u * 256u]; ac3 = ac3 * f3; m3 = mp3;
        let f4 = exp(m4 - mp4); l4 = l4 * f4 + ga_red[4u * 256u]; ac4 = ac4 * f4; m4 = mp4;
        let f5 = exp(m5 - mp5); l5 = l5 * f5 + ga_red[5u * 256u]; ac5 = ac5 * f5; m5 = mp5;
        let f6 = exp(m6 - mp6); l6 = l6 * f6 + ga_red[6u * 256u]; ac6 = ac6 * f6; m6 = mp6;
        let f7 = exp(m7 - mp7); l7 = l7 * f7 + ga_red[7u * 256u]; ac7 = ac7 * f7; m7 = mp7;
        if (vlive) {
            let sl0 = c0 % cap;
            for (var pp = pg; pp < cn; pp = pp + 4u) {
                var sl = sl0 + pp;
                if (sl >= cap) { sl = sl - cap; }
                let r4 = vbase4 + sl * dv4;
                let v = ga_v[r4 + q4];
                let wa = ga_w[pp * 2u];
                let wb = ga_w[pp * 2u + 1u];
                ac0 = ac0 + wa.x * v; ac1 = ac1 + wa.y * v; ac2 = ac2 + wa.z * v; ac3 = ac3 + wa.w * v;
                ac4 = ac4 + wb.x * v; ac5 = ac5 + wb.y * v; ac6 = ac6 + wb.z * v; ac7 = ac7 + wb.w * v;
            }
        }
        c0 = c0 + 256u;
        workgroupBarrier();
    }
    // The four position quarters' partial sums meet in workgroup memory,
    // two tokens a round: slot [(pg·2 + jj)·128 + quad].
    workgroupBarrier();
    if (vlive) {
        ga_sq[(pg * 2u) * 128u + q4] = ac0;
        ga_sq[(pg * 2u + 1u) * 128u + q4] = ac1;
    }
    workgroupBarrier();
    if (pg == 0u && vlive) {
        if (0u < nb) {
            let ob = ((tb0 + 0u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[0u * 128u + q4] + ga_sq[2u * 128u + q4])
                + (ga_sq[4u * 128u + q4] + ga_sq[6u * 128u + q4])) / l0;
        }
        if (1u < nb) {
            let ob = ((tb0 + 1u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[1u * 128u + q4] + ga_sq[3u * 128u + q4])
                + (ga_sq[5u * 128u + q4] + ga_sq[7u * 128u + q4])) / l1;
        }
    }
    workgroupBarrier();
    if (vlive) {
        ga_sq[(pg * 2u) * 128u + q4] = ac2;
        ga_sq[(pg * 2u + 1u) * 128u + q4] = ac3;
    }
    workgroupBarrier();
    if (pg == 0u && vlive) {
        if (2u < nb) {
            let ob = ((tb0 + 2u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[0u * 128u + q4] + ga_sq[2u * 128u + q4])
                + (ga_sq[4u * 128u + q4] + ga_sq[6u * 128u + q4])) / l2;
        }
        if (3u < nb) {
            let ob = ((tb0 + 3u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[1u * 128u + q4] + ga_sq[3u * 128u + q4])
                + (ga_sq[5u * 128u + q4] + ga_sq[7u * 128u + q4])) / l3;
        }
    }
    workgroupBarrier();
    if (vlive) {
        ga_sq[(pg * 2u) * 128u + q4] = ac4;
        ga_sq[(pg * 2u + 1u) * 128u + q4] = ac5;
    }
    workgroupBarrier();
    if (pg == 0u && vlive) {
        if (4u < nb) {
            let ob = ((tb0 + 4u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[0u * 128u + q4] + ga_sq[2u * 128u + q4])
                + (ga_sq[4u * 128u + q4] + ga_sq[6u * 128u + q4])) / l4;
        }
        if (5u < nb) {
            let ob = ((tb0 + 5u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[1u * 128u + q4] + ga_sq[3u * 128u + q4])
                + (ga_sq[5u * 128u + q4] + ga_sq[7u * 128u + q4])) / l5;
        }
    }
    workgroupBarrier();
    if (vlive) {
        ga_sq[(pg * 2u) * 128u + q4] = ac6;
        ga_sq[(pg * 2u + 1u) * 128u + q4] = ac7;
    }
    workgroupBarrier();
    if (pg == 0u && vlive) {
        if (6u < nb) {
            let ob = ((tb0 + 6u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[0u * 128u + q4] + ga_sq[2u * 128u + q4])
                + (ga_sq[4u * 128u + q4] + ga_sq[6u * 128u + q4])) / l6;
        }
        if (7u < nb) {
            let ob = ((tb0 + 7u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[1u * 128u + q4] + ga_sq[3u * 128u + q4])
                + (ga_sq[5u * 128u + q4] + ga_sq[7u * 128u + q4])) / l7;
        }
    }
}

@compute @workgroup_size(256)
fn g4_attend_bt2(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let tb0 = wid.y * 8u;
    let nh = ga_p.nh;
    if (h >= nh || tb0 >= ga_p.k) { return; }
    let nb = min(8u, ga_p.k - tb0);
    let g = h / ga_p.hpk;
    let hd4 = ga_p.hd / 4u;
    let dv4 = ga_p.dv / 4u;
    let cap = ga_p.cap;
    let win = ga_p.win;
    let scale = ga_p.scale;
    for (var i = lid; i < 8u * hd4; i = i + 256u) {
        let j = i / hd4;
        let d = i - j * hd4;
        var qv = vec4<f32>(0.0);
        if (j < nb) { qv = ga_q[((tb0 + j) * nh + h) * hd4 + d]; }
        ga_sq[i] = qv;
    }
    workgroupBarrier();
    let hi0 = ga_p.pos0 + tb0;          // token 0's position
    let p_last = hi0 + nb - 1u;
    var p_first = 0u;
    if (win != 0u && hi0 + 1u > win) { p_first = hi0 + 1u - win; }
    let kbase4 = g * cap * hd4;
    let vbase4 = g * cap * dv4;
    let pg = lid >> 6u;
    let q4 = lid & 63u;
    let vlive = q4 < dv4;
    // Per-token softmax frame and value sums, unrolled by hand (constant
    // indices only: the accumulators must stay in registers).
    var m0 = -1.0e30; var l0 = 0.0; var ac0 = vec4<f32>(0.0);
    var m1 = -1.0e30; var l1 = 0.0; var ac1 = vec4<f32>(0.0);
    var m2 = -1.0e30; var l2 = 0.0; var ac2 = vec4<f32>(0.0);
    var m3 = -1.0e30; var l3 = 0.0; var ac3 = vec4<f32>(0.0);
    var m4 = -1.0e30; var l4 = 0.0; var ac4 = vec4<f32>(0.0);
    var m5 = -1.0e30; var l5 = 0.0; var ac5 = vec4<f32>(0.0);
    var m6 = -1.0e30; var l6 = 0.0; var ac6 = vec4<f32>(0.0);
    var m7 = -1.0e30; var l7 = 0.0; var ac7 = vec4<f32>(0.0);
    var ad0 = vec4<f32>(0.0);
    var ad1 = vec4<f32>(0.0);
    var ad2 = vec4<f32>(0.0);
    var ad3 = vec4<f32>(0.0);
    var ad4 = vec4<f32>(0.0);
    var ad5 = vec4<f32>(0.0);
    var ad6 = vec4<f32>(0.0);
    var ad7 = vec4<f32>(0.0);
    var c0 = p_first;
    loop {
        if (c0 > p_last) { break; }
        let cn = min(256u, p_last + 1u - c0);
        let p = c0 + lid;
        var s0 = -1.0e30;
        var s1 = -1.0e30;
        var s2 = -1.0e30;
        var s3 = -1.0e30;
        var s4 = -1.0e30;
        var s5 = -1.0e30;
        var s6 = -1.0e30;
        var s7 = -1.0e30;
        var okm = 0u;
        if (lid < cn) {
            let krow = kbase4 + (p % cap) * hd4;
            var d0 = vec4<f32>(0.0);
            var d1 = vec4<f32>(0.0);
            var d2 = vec4<f32>(0.0);
            var d3 = vec4<f32>(0.0);
            var d4 = vec4<f32>(0.0);
            var d5 = vec4<f32>(0.0);
            var d6 = vec4<f32>(0.0);
            var d7 = vec4<f32>(0.0);
            for (var d = 0u; d < hd4; d = d + 1u) {
                let kv = ga_k[krow + d];
                d0 = d0 + ga_sq[0u * hd4 + d] * kv;
                d1 = d1 + ga_sq[1u * hd4 + d] * kv;
                d2 = d2 + ga_sq[2u * hd4 + d] * kv;
                d3 = d3 + ga_sq[3u * hd4 + d] * kv;
                d4 = d4 + ga_sq[4u * hd4 + d] * kv;
                d5 = d5 + ga_sq[5u * hd4 + d] * kv;
                d6 = d6 + ga_sq[6u * hd4 + d] * kv;
                d7 = d7 + ga_sq[7u * hd4 + d] * kv;
            }
            let tp0 = hi0 + 0u; if (0u < nb && p <= tp0 && (win == 0u || p + win > tp0)) { s0 = (d0.x + d0.y + d0.z + d0.w) * scale; okm = okm | (1u << 0u); }
            let tp1 = hi0 + 1u; if (1u < nb && p <= tp1 && (win == 0u || p + win > tp1)) { s1 = (d1.x + d1.y + d1.z + d1.w) * scale; okm = okm | (1u << 1u); }
            let tp2 = hi0 + 2u; if (2u < nb && p <= tp2 && (win == 0u || p + win > tp2)) { s2 = (d2.x + d2.y + d2.z + d2.w) * scale; okm = okm | (1u << 2u); }
            let tp3 = hi0 + 3u; if (3u < nb && p <= tp3 && (win == 0u || p + win > tp3)) { s3 = (d3.x + d3.y + d3.z + d3.w) * scale; okm = okm | (1u << 3u); }
            let tp4 = hi0 + 4u; if (4u < nb && p <= tp4 && (win == 0u || p + win > tp4)) { s4 = (d4.x + d4.y + d4.z + d4.w) * scale; okm = okm | (1u << 4u); }
            let tp5 = hi0 + 5u; if (5u < nb && p <= tp5 && (win == 0u || p + win > tp5)) { s5 = (d5.x + d5.y + d5.z + d5.w) * scale; okm = okm | (1u << 5u); }
            let tp6 = hi0 + 6u; if (6u < nb && p <= tp6 && (win == 0u || p + win > tp6)) { s6 = (d6.x + d6.y + d6.z + d6.w) * scale; okm = okm | (1u << 6u); }
            let tp7 = hi0 + 7u; if (7u < nb && p <= tp7 && (win == 0u || p + win > tp7)) { s7 = (d7.x + d7.y + d7.z + d7.w) * scale; okm = okm | (1u << 7u); }
        }
        ga_red[0u * 256u + lid] = s0;
        ga_red[1u * 256u + lid] = s1;
        ga_red[2u * 256u + lid] = s2;
        ga_red[3u * 256u + lid] = s3;
        ga_red[4u * 256u + lid] = s4;
        ga_red[5u * 256u + lid] = s5;
        ga_red[6u * 256u + lid] = s6;
        ga_red[7u * 256u + lid] = s7;
        workgroupBarrier();
        var st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) {
                ga_red[0u * 256u + lid] = max(ga_red[0u * 256u + lid], ga_red[0u * 256u + lid + st]);
                ga_red[1u * 256u + lid] = max(ga_red[1u * 256u + lid], ga_red[1u * 256u + lid + st]);
                ga_red[2u * 256u + lid] = max(ga_red[2u * 256u + lid], ga_red[2u * 256u + lid + st]);
                ga_red[3u * 256u + lid] = max(ga_red[3u * 256u + lid], ga_red[3u * 256u + lid + st]);
                ga_red[4u * 256u + lid] = max(ga_red[4u * 256u + lid], ga_red[4u * 256u + lid + st]);
                ga_red[5u * 256u + lid] = max(ga_red[5u * 256u + lid], ga_red[5u * 256u + lid + st]);
                ga_red[6u * 256u + lid] = max(ga_red[6u * 256u + lid], ga_red[6u * 256u + lid + st]);
                ga_red[7u * 256u + lid] = max(ga_red[7u * 256u + lid], ga_red[7u * 256u + lid + st]);
            }
            workgroupBarrier();
            st = st >> 1u;
        }
        let mp0 = max(m0, ga_red[0u * 256u]);
        let mp1 = max(m1, ga_red[1u * 256u]);
        let mp2 = max(m2, ga_red[2u * 256u]);
        let mp3 = max(m3, ga_red[3u * 256u]);
        let mp4 = max(m4, ga_red[4u * 256u]);
        let mp5 = max(m5, ga_red[5u * 256u]);
        let mp6 = max(m6, ga_red[6u * 256u]);
        let mp7 = max(m7, ga_red[7u * 256u]);
        workgroupBarrier();
        let w0 = select(0.0, exp(s0 - mp0), (okm & (1u << 0u)) != 0u); ga_red[0u * 256u + lid] = w0;
        let w1 = select(0.0, exp(s1 - mp1), (okm & (1u << 1u)) != 0u); ga_red[1u * 256u + lid] = w1;
        let w2 = select(0.0, exp(s2 - mp2), (okm & (1u << 2u)) != 0u); ga_red[2u * 256u + lid] = w2;
        let w3 = select(0.0, exp(s3 - mp3), (okm & (1u << 3u)) != 0u); ga_red[3u * 256u + lid] = w3;
        let w4 = select(0.0, exp(s4 - mp4), (okm & (1u << 4u)) != 0u); ga_red[4u * 256u + lid] = w4;
        let w5 = select(0.0, exp(s5 - mp5), (okm & (1u << 5u)) != 0u); ga_red[5u * 256u + lid] = w5;
        let w6 = select(0.0, exp(s6 - mp6), (okm & (1u << 6u)) != 0u); ga_red[6u * 256u + lid] = w6;
        let w7 = select(0.0, exp(s7 - mp7), (okm & (1u << 7u)) != 0u); ga_red[7u * 256u + lid] = w7;
        ga_w[lid * 2u] = vec4<f32>(w0, w1, w2, w3);
        ga_w[lid * 2u + 1u] = vec4<f32>(w4, w5, w6, w7);
        workgroupBarrier();
        st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) {
                ga_red[0u * 256u + lid] = ga_red[0u * 256u + lid] + ga_red[0u * 256u + lid + st];
                ga_red[1u * 256u + lid] = ga_red[1u * 256u + lid] + ga_red[1u * 256u + lid + st];
                ga_red[2u * 256u + lid] = ga_red[2u * 256u + lid] + ga_red[2u * 256u + lid + st];
                ga_red[3u * 256u + lid] = ga_red[3u * 256u + lid] + ga_red[3u * 256u + lid + st];
                ga_red[4u * 256u + lid] = ga_red[4u * 256u + lid] + ga_red[4u * 256u + lid + st];
                ga_red[5u * 256u + lid] = ga_red[5u * 256u + lid] + ga_red[5u * 256u + lid + st];
                ga_red[6u * 256u + lid] = ga_red[6u * 256u + lid] + ga_red[6u * 256u + lid + st];
                ga_red[7u * 256u + lid] = ga_red[7u * 256u + lid] + ga_red[7u * 256u + lid + st];
            }
            workgroupBarrier();
            st = st >> 1u;
        }
        let f0 = exp(m0 - mp0); l0 = l0 * f0 + ga_red[0u * 256u]; ac0 = ac0 * f0; m0 = mp0; ad0 = ad0 * f0;
        let f1 = exp(m1 - mp1); l1 = l1 * f1 + ga_red[1u * 256u]; ac1 = ac1 * f1; m1 = mp1; ad1 = ad1 * f1;
        let f2 = exp(m2 - mp2); l2 = l2 * f2 + ga_red[2u * 256u]; ac2 = ac2 * f2; m2 = mp2; ad2 = ad2 * f2;
        let f3 = exp(m3 - mp3); l3 = l3 * f3 + ga_red[3u * 256u]; ac3 = ac3 * f3; m3 = mp3; ad3 = ad3 * f3;
        let f4 = exp(m4 - mp4); l4 = l4 * f4 + ga_red[4u * 256u]; ac4 = ac4 * f4; m4 = mp4; ad4 = ad4 * f4;
        let f5 = exp(m5 - mp5); l5 = l5 * f5 + ga_red[5u * 256u]; ac5 = ac5 * f5; m5 = mp5; ad5 = ad5 * f5;
        let f6 = exp(m6 - mp6); l6 = l6 * f6 + ga_red[6u * 256u]; ac6 = ac6 * f6; m6 = mp6; ad6 = ad6 * f6;
        let f7 = exp(m7 - mp7); l7 = l7 * f7 + ga_red[7u * 256u]; ac7 = ac7 * f7; m7 = mp7; ad7 = ad7 * f7;
        if (vlive) {
            let sl0 = c0 % cap;
            for (var pp = pg; pp < cn; pp = pp + 4u) {
                var sl = sl0 + pp;
                if (sl >= cap) { sl = sl - cap; }
                let r4 = vbase4 + sl * dv4;
                let v = ga_v[r4 + q4];
                let wa = ga_w[pp * 2u];
                let wb = ga_w[pp * 2u + 1u];
                ac0 = ac0 + wa.x * v; ac1 = ac1 + wa.y * v; ac2 = ac2 + wa.z * v; ac3 = ac3 + wa.w * v;
                ac4 = ac4 + wb.x * v; ac5 = ac5 + wb.y * v; ac6 = ac6 + wb.z * v; ac7 = ac7 + wb.w * v;
                let v2 = ga_v[r4 + q4 + 64u];
                ad0 = ad0 + wa.x * v2; ad1 = ad1 + wa.y * v2; ad2 = ad2 + wa.z * v2; ad3 = ad3 + wa.w * v2;
                ad4 = ad4 + wb.x * v2; ad5 = ad5 + wb.y * v2; ad6 = ad6 + wb.z * v2; ad7 = ad7 + wb.w * v2;
            }
        }
        c0 = c0 + 256u;
        workgroupBarrier();
    }
    // The four position quarters' partial sums meet in workgroup memory,
    // two tokens a round: slot [(pg·2 + jj)·128 + quad].
    workgroupBarrier();
    if (vlive) {
        ga_sq[(pg * 2u) * 128u + q4] = ac0;
        ga_sq[(pg * 2u + 1u) * 128u + q4] = ac1;
        ga_sq[(pg * 2u) * 128u + q4 + 64u] = ad0;
        ga_sq[(pg * 2u + 1u) * 128u + q4 + 64u] = ad1;
    }
    workgroupBarrier();
    if (pg == 0u && vlive) {
        if (0u < nb) {
            let ob = ((tb0 + 0u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[0u * 128u + q4] + ga_sq[2u * 128u + q4])
                + (ga_sq[4u * 128u + q4] + ga_sq[6u * 128u + q4])) / l0;
            ga_o[ob + q4 + 64u] = ((ga_sq[0u * 128u + q4 + 64u] + ga_sq[2u * 128u + q4 + 64u])
                + (ga_sq[4u * 128u + q4 + 64u] + ga_sq[6u * 128u + q4 + 64u])) / l0;
        }
        if (1u < nb) {
            let ob = ((tb0 + 1u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[1u * 128u + q4] + ga_sq[3u * 128u + q4])
                + (ga_sq[5u * 128u + q4] + ga_sq[7u * 128u + q4])) / l1;
            ga_o[ob + q4 + 64u] = ((ga_sq[1u * 128u + q4 + 64u] + ga_sq[3u * 128u + q4 + 64u])
                + (ga_sq[5u * 128u + q4 + 64u] + ga_sq[7u * 128u + q4 + 64u])) / l1;
        }
    }
    workgroupBarrier();
    if (vlive) {
        ga_sq[(pg * 2u) * 128u + q4] = ac2;
        ga_sq[(pg * 2u + 1u) * 128u + q4] = ac3;
        ga_sq[(pg * 2u) * 128u + q4 + 64u] = ad2;
        ga_sq[(pg * 2u + 1u) * 128u + q4 + 64u] = ad3;
    }
    workgroupBarrier();
    if (pg == 0u && vlive) {
        if (2u < nb) {
            let ob = ((tb0 + 2u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[0u * 128u + q4] + ga_sq[2u * 128u + q4])
                + (ga_sq[4u * 128u + q4] + ga_sq[6u * 128u + q4])) / l2;
            ga_o[ob + q4 + 64u] = ((ga_sq[0u * 128u + q4 + 64u] + ga_sq[2u * 128u + q4 + 64u])
                + (ga_sq[4u * 128u + q4 + 64u] + ga_sq[6u * 128u + q4 + 64u])) / l2;
        }
        if (3u < nb) {
            let ob = ((tb0 + 3u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[1u * 128u + q4] + ga_sq[3u * 128u + q4])
                + (ga_sq[5u * 128u + q4] + ga_sq[7u * 128u + q4])) / l3;
            ga_o[ob + q4 + 64u] = ((ga_sq[1u * 128u + q4 + 64u] + ga_sq[3u * 128u + q4 + 64u])
                + (ga_sq[5u * 128u + q4 + 64u] + ga_sq[7u * 128u + q4 + 64u])) / l3;
        }
    }
    workgroupBarrier();
    if (vlive) {
        ga_sq[(pg * 2u) * 128u + q4] = ac4;
        ga_sq[(pg * 2u + 1u) * 128u + q4] = ac5;
        ga_sq[(pg * 2u) * 128u + q4 + 64u] = ad4;
        ga_sq[(pg * 2u + 1u) * 128u + q4 + 64u] = ad5;
    }
    workgroupBarrier();
    if (pg == 0u && vlive) {
        if (4u < nb) {
            let ob = ((tb0 + 4u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[0u * 128u + q4] + ga_sq[2u * 128u + q4])
                + (ga_sq[4u * 128u + q4] + ga_sq[6u * 128u + q4])) / l4;
            ga_o[ob + q4 + 64u] = ((ga_sq[0u * 128u + q4 + 64u] + ga_sq[2u * 128u + q4 + 64u])
                + (ga_sq[4u * 128u + q4 + 64u] + ga_sq[6u * 128u + q4 + 64u])) / l4;
        }
        if (5u < nb) {
            let ob = ((tb0 + 5u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[1u * 128u + q4] + ga_sq[3u * 128u + q4])
                + (ga_sq[5u * 128u + q4] + ga_sq[7u * 128u + q4])) / l5;
            ga_o[ob + q4 + 64u] = ((ga_sq[1u * 128u + q4 + 64u] + ga_sq[3u * 128u + q4 + 64u])
                + (ga_sq[5u * 128u + q4 + 64u] + ga_sq[7u * 128u + q4 + 64u])) / l5;
        }
    }
    workgroupBarrier();
    if (vlive) {
        ga_sq[(pg * 2u) * 128u + q4] = ac6;
        ga_sq[(pg * 2u + 1u) * 128u + q4] = ac7;
        ga_sq[(pg * 2u) * 128u + q4 + 64u] = ad6;
        ga_sq[(pg * 2u + 1u) * 128u + q4 + 64u] = ad7;
    }
    workgroupBarrier();
    if (pg == 0u && vlive) {
        if (6u < nb) {
            let ob = ((tb0 + 6u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[0u * 128u + q4] + ga_sq[2u * 128u + q4])
                + (ga_sq[4u * 128u + q4] + ga_sq[6u * 128u + q4])) / l6;
            ga_o[ob + q4 + 64u] = ((ga_sq[0u * 128u + q4 + 64u] + ga_sq[2u * 128u + q4 + 64u])
                + (ga_sq[4u * 128u + q4 + 64u] + ga_sq[6u * 128u + q4 + 64u])) / l6;
        }
        if (7u < nb) {
            let ob = ((tb0 + 7u) * nh + h) * dv4;
            ga_o[ob + q4] = ((ga_sq[1u * 128u + q4] + ga_sq[3u * 128u + q4])
                + (ga_sq[5u * 128u + q4] + ga_sq[7u * 128u + q4])) / l7;
            ga_o[ob + q4 + 64u] = ((ga_sq[1u * 128u + q4 + 64u] + ga_sq[3u * 128u + q4 + 64u])
                + (ga_sq[5u * 128u + q4 + 64u] + ga_sq[7u * 128u + q4 + 64u])) / l7;
        }
    }
}

// ── Gemma sandwich tail, per row (wid.x): h = (h + d)·scale (scale bits 0:
// no scale), then o = rms(h)·w — the token graph's add_rmsnorm with its
// layer-scale word, strided by row.
struct GnP { n: u32, gemma: u32, eps: f32, scale: u32 };
@group(0) @binding(18) var<storage, read_write> gn_h : array<f32>;
@group(0) @binding(19) var<storage, read>       gn_d : array<f32>;
@group(0) @binding(20) var<storage, read>       gn_w : array<f32>;
@group(0) @binding(21) var<storage, read_write> gn_o : array<f32>;
@group(0) @binding(22) var<uniform>             gn_p : GnP;
var<workgroup> gn_part: array<f32, 256>;
@compute @workgroup_size(256)
fn g4_add_norm_b(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = gn_p.n;
    let base = wid.x * n;
    var acc = 0.0;
    var i = tid;
    loop {
        if (i >= n) { break; }
        var v = gn_h[base + i] + gn_d[base + i];
        if (gn_p.scale != 0u) { v = v * bitcast<f32>(gn_p.scale); }
        gn_h[base + i] = v;
        acc = acc + v * v;
        i = i + 256u;
    }
    gn_part[tid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { gn_part[tid] = gn_part[tid] + gn_part[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(gn_part[0] / f32(n) + gn_p.eps);
    i = tid;
    loop {
        if (i >= n) { break; }
        var wv = gn_w[i];
        if (gn_p.gemma == 1u) { wv = 1.0 + wv; }
        gn_o[base + i] = gn_h[base + i] * inv * wv;
        i = i + 256u;
    }
}
"#;

/// The q8_2f prompt GEMM: `q4tp_mm_r2`'s tile, staging layout and FMA
/// order over an int8 weight. Weight buffer = the whole q8_2f tensor:
/// int8 body `[rows][cols]`, f16 row scales, f16 column field.
pub(crate) const Q82_MM_WGSL: &str = r#"
struct MmP { cols4: u32, rows: u32, nb: u32, cols: u32 };
@group(0) @binding(0) var<storage, read>       qmm  : array<u32>;
@group(0) @binding(1) var<storage, read>       xmm4 : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ymm  : array<f32>;
@group(0) @binding(3) var<uniform>             pmm  : MmP;
var<workgroup> rx: array<vec4<f32>, 1024>;
var<workgroup> rw: array<vec4<f32>, 1024>;
// Four f16 of the column field from half index `hh` (any alignment).
fn cf4(hh: u32) -> vec4<f32> {
    let w = hh >> 1u;
    let a = unpack2x16float(qmm[w]);
    let b = unpack2x16float(qmm[w + 1u]);
    if ((hh & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(qmm[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
fn i8f(w: u32, b: u32) -> f32 {
    return f32((i32(w << (24u - 8u * b))) >> 24u);
}
@compute @workgroup_size(128)
fn q8_2f_mm_r2(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) t: u32) {
    let cols4 = pmm.cols4;
    let cols = pmm.cols;
    let rows = pmm.rows;
    let nb = pmm.nb;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let rs0 = (rows * cols) >> 2u;          // word of the f16 row scales
    let cs0h = ((rows * cols) >> 1u) + rows; // half index of the column field
    // Weight staging role: words 2kw, 2kw+1 of the step (k = 8kw .. 8kw+7)
    // of rows 4ng .. 4ng+3.
    let kw = t & 7u;
    let ng = t >> 3u;
    let wr = n0 + ng * 4u;
    let tm = t & 15u;
    let tn = t >> 4u;
    let r0 = min(wr + 0u, rows - 1u);
    let r1 = min(wr + 1u, rows - 1u);
    let r2 = min(wr + 2u, rows - 1u);
    let r3 = min(wr + 3u, rows - 1u);
    let wb0 = r0 * cols4;
    let wb1 = r1 * cols4;
    let wb2 = r2 * cols4;
    let wb3 = r3 * cols4;
    var a00 = 0.0; var a01 = 0.0; var a02 = 0.0; var a03 = 0.0;
    var a04 = 0.0; var a05 = 0.0; var a06 = 0.0; var a07 = 0.0;
    var a10 = 0.0; var a11 = 0.0; var a12 = 0.0; var a13 = 0.0;
    var a14 = 0.0; var a15 = 0.0; var a16 = 0.0; var a17 = 0.0;
    var a20 = 0.0; var a21 = 0.0; var a22 = 0.0; var a23 = 0.0;
    var a24 = 0.0; var a25 = 0.0; var a26 = 0.0; var a27 = 0.0;
    var a30 = 0.0; var a31 = 0.0; var a32 = 0.0; var a33 = 0.0;
    var a34 = 0.0; var a35 = 0.0; var a36 = 0.0; var a37 = 0.0;
    // x staging tasks u = t and t + 128: kq = u & 15, mg = u >> 4.
    let kqa = t & 15u;
    let mga = t >> 4u;
    let mgb = mga + 8u;
    let ma = m0 + mga * 4u;
    let mb = m0 + mgb * 4u;
    var xa0: vec4<f32>; var xa1: vec4<f32>; var xa2: vec4<f32>; var xa3: vec4<f32>;
    var xb0: vec4<f32>; var xb1: vec4<f32>; var xb2: vec4<f32>; var xb3: vec4<f32>;
    var cf: vec4<f32>;
    var q0a: u32; var q0b: u32; var q1a: u32; var q1b: u32;
    var q2a: u32; var q2b: u32; var q3a: u32; var q3b: u32;
    {
        let ci = kqa;
        xa0 = select(vec4<f32>(0.0), xmm4[min(ma + 0u, nb - 1u) * cols4 + ci], ma + 0u < nb);
        xa1 = select(vec4<f32>(0.0), xmm4[min(ma + 1u, nb - 1u) * cols4 + ci], ma + 1u < nb);
        xa2 = select(vec4<f32>(0.0), xmm4[min(ma + 2u, nb - 1u) * cols4 + ci], ma + 2u < nb);
        xa3 = select(vec4<f32>(0.0), xmm4[min(ma + 3u, nb - 1u) * cols4 + ci], ma + 3u < nb);
        xb0 = select(vec4<f32>(0.0), xmm4[min(mb + 0u, nb - 1u) * cols4 + ci], mb + 0u < nb);
        xb1 = select(vec4<f32>(0.0), xmm4[min(mb + 1u, nb - 1u) * cols4 + ci], mb + 1u < nb);
        xb2 = select(vec4<f32>(0.0), xmm4[min(mb + 2u, nb - 1u) * cols4 + ci], mb + 2u < nb);
        xb3 = select(vec4<f32>(0.0), xmm4[min(mb + 3u, nb - 1u) * cols4 + ci], mb + 3u < nb);
        cf = cf4(cs0h + ci * 4u);
        let wo = 2u * kw;
        q0a = qmm[wb0 + wo]; q0b = qmm[wb0 + wo + 1u];
        q1a = qmm[wb1 + wo]; q1b = qmm[wb1 + wo + 1u];
        q2a = qmm[wb2 + wo]; q2b = qmm[wb2 + wo + 1u];
        q3a = qmm[wb3 + wo]; q3b = qmm[wb3 + wo + 1u];
    }
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        {
            let kb = kqa * 64u;
            let sx = mga ^ kqa;
            let y0 = xa0 * cf; let y1 = xa1 * cf; let y2 = xa2 * cf; let y3 = xa3 * cf;
            rx[kb + sx] = vec4<f32>(y0.x, y1.x, y2.x, y3.x);
            rx[kb + 16u + sx] = vec4<f32>(y0.y, y1.y, y2.y, y3.y);
            rx[kb + 32u + sx] = vec4<f32>(y0.z, y1.z, y2.z, y3.z);
            rx[kb + 48u + sx] = vec4<f32>(y0.w, y1.w, y2.w, y3.w);
        }
        {
            let kb = kqa * 64u;
            let sx = mgb ^ kqa;
            let y0 = xb0 * cf; let y1 = xb1 * cf; let y2 = xb2 * cf; let y3 = xb3 * cf;
            rx[kb + sx] = vec4<f32>(y0.x, y1.x, y2.x, y3.x);
            rx[kb + 16u + sx] = vec4<f32>(y0.y, y1.y, y2.y, y3.y);
            rx[kb + 32u + sx] = vec4<f32>(y0.z, y1.z, y2.z, y3.z);
            rx[kb + 48u + sx] = vec4<f32>(y0.w, y1.w, y2.w, y3.w);
        }
        {
            let sw = ng ^ kw;
            let kb = kw * 128u;
            rw[kb + 0u + sw] = vec4<f32>(i8f(q0a, 0u), i8f(q1a, 0u), i8f(q2a, 0u), i8f(q3a, 0u));
            rw[kb + 16u + sw] = vec4<f32>(i8f(q0a, 1u), i8f(q1a, 1u), i8f(q2a, 1u), i8f(q3a, 1u));
            rw[kb + 32u + sw] = vec4<f32>(i8f(q0a, 2u), i8f(q1a, 2u), i8f(q2a, 2u), i8f(q3a, 2u));
            rw[kb + 48u + sw] = vec4<f32>(i8f(q0a, 3u), i8f(q1a, 3u), i8f(q2a, 3u), i8f(q3a, 3u));
            rw[kb + 64u + sw] = vec4<f32>(i8f(q0b, 0u), i8f(q1b, 0u), i8f(q2b, 0u), i8f(q3b, 0u));
            rw[kb + 80u + sw] = vec4<f32>(i8f(q0b, 1u), i8f(q1b, 1u), i8f(q2b, 1u), i8f(q3b, 1u));
            rw[kb + 96u + sw] = vec4<f32>(i8f(q0b, 2u), i8f(q1b, 2u), i8f(q2b, 2u), i8f(q3b, 2u));
            rw[kb + 112u + sw] = vec4<f32>(i8f(q0b, 3u), i8f(q1b, 3u), i8f(q2b, 3u), i8f(q3b, 3u));
        }
        workgroupBarrier();
        let kn = k0 + 64u;
        if (kn < cols) {
            let ci = (kn >> 2u) + kqa;
            xa0 = select(vec4<f32>(0.0), xmm4[min(ma + 0u, nb - 1u) * cols4 + ci], ma + 0u < nb);
            xa1 = select(vec4<f32>(0.0), xmm4[min(ma + 1u, nb - 1u) * cols4 + ci], ma + 1u < nb);
            xa2 = select(vec4<f32>(0.0), xmm4[min(ma + 2u, nb - 1u) * cols4 + ci], ma + 2u < nb);
            xa3 = select(vec4<f32>(0.0), xmm4[min(ma + 3u, nb - 1u) * cols4 + ci], ma + 3u < nb);
            xb0 = select(vec4<f32>(0.0), xmm4[min(mb + 0u, nb - 1u) * cols4 + ci], mb + 0u < nb);
            xb1 = select(vec4<f32>(0.0), xmm4[min(mb + 1u, nb - 1u) * cols4 + ci], mb + 1u < nb);
            xb2 = select(vec4<f32>(0.0), xmm4[min(mb + 2u, nb - 1u) * cols4 + ci], mb + 2u < nb);
            xb3 = select(vec4<f32>(0.0), xmm4[min(mb + 3u, nb - 1u) * cols4 + ci], mb + 3u < nb);
            cf = cf4(cs0h + ci * 4u);
            let wo = (kn >> 2u) + 2u * kw;
            q0a = qmm[wb0 + wo]; q0b = qmm[wb0 + wo + 1u];
            q1a = qmm[wb1 + wo]; q1b = qmm[wb1 + wo + 1u];
            q2a = qmm[wb2 + wo]; q2b = qmm[wb2 + wo + 1u];
            q3a = qmm[wb3 + wo]; q3b = qmm[wb3 + wo + 1u];
        }
        for (var kq = 0u; kq < 16u; kq = kq + 1u) {
            let sx = tm ^ kq;
            let sw = kq >> 1u;
            let na = (tn * 2u) ^ sw;
            let nbb = (tn * 2u + 1u) ^ sw;
            for (var c4 = 0u; c4 < 4u; c4 = c4 + 1u) {
                let kk = kq * 4u + c4;
                let xv = rx[kk * 16u + sx];
                let wa = rw[kk * 16u + na];
                let wb = rw[kk * 16u + nbb];
                let x0 = xv.x; let x1 = xv.y; let x2 = xv.z; let x3 = xv.w;
                let y0 = wa.x; let y1 = wa.y; let y2 = wa.z; let y3 = wa.w;
                let y4 = wb.x; let y5 = wb.y; let y6 = wb.z; let y7 = wb.w;
                a00 = a00 + x0 * y0; a01 = a01 + x0 * y1; a02 = a02 + x0 * y2; a03 = a03 + x0 * y3; a04 = a04 + x0 * y4; a05 = a05 + x0 * y5; a06 = a06 + x0 * y6; a07 = a07 + x0 * y7;
                a10 = a10 + x1 * y0; a11 = a11 + x1 * y1; a12 = a12 + x1 * y2; a13 = a13 + x1 * y3; a14 = a14 + x1 * y4; a15 = a15 + x1 * y5; a16 = a16 + x1 * y6; a17 = a17 + x1 * y7;
                a20 = a20 + x2 * y0; a21 = a21 + x2 * y1; a22 = a22 + x2 * y2; a23 = a23 + x2 * y3; a24 = a24 + x2 * y4; a25 = a25 + x2 * y5; a26 = a26 + x2 * y6; a27 = a27 + x2 * y7;
                a30 = a30 + x3 * y0; a31 = a31 + x3 * y1; a32 = a32 + x3 * y2; a33 = a33 + x3 * y3; a34 = a34 + x3 * y4; a35 = a35 + x3 * y5; a36 = a36 + x3 * y6; a37 = a37 + x3 * y7;
            }
        }
        workgroupBarrier();
        k0 = kn;
    }
    // Row scales of this lane's eight output rows.
    let nq = n0 + tn * 8u;
    var rsv: array<f32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let r = min(nq + i, rows - 1u);
        let pr = unpack2x16float(qmm[rs0 + (r >> 1u)]);
        rsv[i] = select(pr.x, pr.y, (r & 1u) == 1u);
    }
    {
        let m = m0 + tm * 4u + 0u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a00 * rsv[0]; }
            if (nq + 1u < rows) { ymm[o + 1u] = a01 * rsv[1]; }
            if (nq + 2u < rows) { ymm[o + 2u] = a02 * rsv[2]; }
            if (nq + 3u < rows) { ymm[o + 3u] = a03 * rsv[3]; }
            if (nq + 4u < rows) { ymm[o + 4u] = a04 * rsv[4]; }
            if (nq + 5u < rows) { ymm[o + 5u] = a05 * rsv[5]; }
            if (nq + 6u < rows) { ymm[o + 6u] = a06 * rsv[6]; }
            if (nq + 7u < rows) { ymm[o + 7u] = a07 * rsv[7]; }
        }
    }
    {
        let m = m0 + tm * 4u + 1u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a10 * rsv[0]; }
            if (nq + 1u < rows) { ymm[o + 1u] = a11 * rsv[1]; }
            if (nq + 2u < rows) { ymm[o + 2u] = a12 * rsv[2]; }
            if (nq + 3u < rows) { ymm[o + 3u] = a13 * rsv[3]; }
            if (nq + 4u < rows) { ymm[o + 4u] = a14 * rsv[4]; }
            if (nq + 5u < rows) { ymm[o + 5u] = a15 * rsv[5]; }
            if (nq + 6u < rows) { ymm[o + 6u] = a16 * rsv[6]; }
            if (nq + 7u < rows) { ymm[o + 7u] = a17 * rsv[7]; }
        }
    }
    {
        let m = m0 + tm * 4u + 2u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a20 * rsv[0]; }
            if (nq + 1u < rows) { ymm[o + 1u] = a21 * rsv[1]; }
            if (nq + 2u < rows) { ymm[o + 2u] = a22 * rsv[2]; }
            if (nq + 3u < rows) { ymm[o + 3u] = a23 * rsv[3]; }
            if (nq + 4u < rows) { ymm[o + 4u] = a24 * rsv[4]; }
            if (nq + 5u < rows) { ymm[o + 5u] = a25 * rsv[5]; }
            if (nq + 6u < rows) { ymm[o + 6u] = a26 * rsv[6]; }
            if (nq + 7u < rows) { ymm[o + 7u] = a27 * rsv[7]; }
        }
    }
    {
        let m = m0 + tm * 4u + 3u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a30 * rsv[0]; }
            if (nq + 1u < rows) { ymm[o + 1u] = a31 * rsv[1]; }
            if (nq + 2u < rows) { ymm[o + 2u] = a32 * rsv[2]; }
            if (nq + 3u < rows) { ymm[o + 3u] = a33 * rsv[3]; }
            if (nq + 4u < rows) { ymm[o + 4u] = a34 * rsv[4]; }
            if (nq + 5u < rows) { ymm[o + 5u] = a35 * rsv[5]; }
            if (nq + 6u < rows) { ymm[o + 6u] = a36 * rsv[6]; }
            if (nq + 7u < rows) { ymm[o + 7u] = a37 * rsv[7]; }
        }
    }
}
"#;

type Pipe = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

pub(crate) struct Pipes {
    pub(crate) rope: Pipe,
    pub(crate) kv: Pipe,
    /// dv <= 256 / dv <= 512 (Gemma-4's global layers).
    pub(crate) attend: Pipe,
    pub(crate) attend2: Pipe,
    pub(crate) add_norm: Pipe,
    /// None under `CMF_G4_MM=0` or when the module was rejected.
    pub(crate) mm: Option<Pipe>,
}

fn module_pipes(c: &Ctx, label: &str, src: &str, entries: &[&str]) -> Option<Vec<Pipe>> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipes: Vec<wgpu::ComputePipeline> = entries
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
        tracing::warn!("{label} module rejected ({e}): Gemma-4 keeps the chunked host prefill");
        return None;
    }
    Some(
        pipes
            .into_iter()
            .map(|p| {
                let l = p.get_bind_group_layout(0);
                (p, l)
            })
            .collect(),
    )
}

fn build(c: &Ctx) -> Option<Pipes> {
    // The attend's workgroup arrays (16 KB of queries, 2×8 KB of scores and
    // reductions) and the GEMM's two 16 KB tiles.
    if c.device.limits().max_compute_workgroup_storage_size < 32 * 1024 {
        return None;
    }
    let mut v = module_pipes(
        c,
        "cmf-g4-batch",
        G4_WGSL,
        &["g4_rope_bt", "g4_kv_bt", "g4_attend_bt", "g4_attend_bt2", "g4_add_norm_b"],
    )?
    .into_iter();
    let (rope, kv, attend, attend2, add_norm) =
        (v.next()?, v.next()?, v.next()?, v.next()?, v.next()?);
    let mm = if std::env::var("CMF_G4_MM").as_deref() == Ok("0") {
        None
    } else {
        module_pipes(c, "cmf-q82-mm", Q82_MM_WGSL, &["q8_2f_mm_r2"]).and_then(|v| v.into_iter().next())
    };
    Some(Pipes {
        rope,
        kv,
        attend,
        attend2,
        add_norm,
        mm,
    })
}

/// The Gemma-4 batch kernels, when the module came up (`CMF_G4_BATCH=0`:
/// never — the batch graph then refuses Gemma-4 as before).
pub(crate) fn pipes(c: &Ctx) -> Option<&Pipes> {
    if !crate::gpu::g4_batch_on() {
        return None;
    }
    c.g4_batch_pipes.get_or_init(|| build(c)).as_ref()
}

fn fresh_uniform(c: &Ctx, words: &[u32]) -> wgpu::Buffer {
    c.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("g4-u"),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

fn bind(c: &Ctx, l: &wgpu::BindGroupLayout, e: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let entries: Vec<_> = e.iter().map(|(i, b)| bind_buf(*i, b)).collect();
    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("g4-batch"),
        layout: l,
        entries: &entries,
    })
}

/// Attention geometry of one Gemma-4 layer in a run of `k` rows.
pub(crate) struct G4Attn {
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub rd: usize,
    pub dv: usize,
    pub flags: u32,
    pub eps: f32,
    pub rope_scale: f32,
    pub v_norm: bool,
    pub cap: usize,
    pub pos0: usize,
    pub window: Option<usize>,
    pub scale: f32,
    pub k: usize,
}

/// One Gemma-4 attention layer of the batch: norm + rope of Q/K (in
/// `qall` / in place in `kb`), V norm in place in `vb`, every row into the
/// mirror, then the blocked attend into `out` ([k][nh][dv]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attn(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes,
    g: &G4Attn,
    qraw: &wgpu::Buffer,
    kb: &wgpu::Buffer,
    vb: &wgpu::Buffer,
    qall: &wgpu::Buffer,
    qnw: &wgpu::Buffer,
    knw: &wgpu::Buffer,
    invf: &wgpu::Buffer,
    kcache: &wgpu::Buffer,
    vcache: &wgpu::Buffer,
    out: &wgpu::Buffer,
) {
    let rope_u = fresh_uniform(
        c,
        &[
            g.nh as u32,
            g.nkv as u32,
            g.hd as u32,
            g.rd as u32,
            g.pos0 as u32,
            g.flags,
            g.eps.to_bits(),
            g.dv as u32,
            g.rope_scale.to_bits(),
            u32::from(g.v_norm),
            0,
            0,
        ],
    );
    let kv_u = fresh_uniform(
        c,
        &[g.nkv as u32, g.hd as u32, g.dv as u32, g.cap as u32, g.pos0 as u32, 0, 0, 0],
    );
    let at_u = fresh_uniform(
        c,
        &[
            g.nh as u32,
            (g.nh / g.nkv) as u32,
            g.hd as u32,
            g.dv as u32,
            g.cap as u32,
            g.pos0 as u32,
            g.window.unwrap_or(0) as u32,
            g.scale.to_bits(),
            g.k as u32,
            0,
            0,
            0,
        ],
    );
    let bg_rope = bind(
        c,
        &p.rope.1,
        &[(0, qraw), (1, kb), (2, qall), (3, vb), (4, qnw), (5, knw), (6, invf), (7, &rope_u)],
    );
    let bg_kv = bind(c, &p.kv.1, &[(8, kb), (9, vb), (10, kcache), (11, vcache), (12, &kv_u)]);
    let at = if g.dv > 256 { &p.attend2 } else { &p.attend };
    let bg_at = bind(c, &at.1, &[(13, qall), (14, kcache), (15, vcache), (16, out), (17, &at_u)]);
    let heads = g.nh + g.nkv + if g.v_norm { g.nkv } else { 0 };
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.rope.0);
    pass.set_bind_group(0, &bg_rope, &[]);
    pass.dispatch_workgroups(heads as u32, g.k as u32, 1);
    pass.set_pipeline(&p.kv.0);
    pass.set_bind_group(0, &bg_kv, &[]);
    pass.dispatch_workgroups(
        ((g.nkv * g.hd.max(g.dv)) as u32).div_ceil(256),
        g.k as u32,
        1,
    );
    pass.set_pipeline(&at.0);
    pass.set_bind_group(0, &bg_at, &[]);
    pass.dispatch_workgroups(g.nh as u32, g.k.div_ceil(8) as u32, 1);
}

/// Can `encode_attn` take this geometry? (Mirror rows are vec4 rows; the
/// attend stages eight 512-wide queries and owns two V dims a lane.)
pub(crate) fn attn_fits(nh: usize, nkv: usize, hd: usize, dv: usize, rd: usize) -> bool {
    nkv > 0
        && nh % nkv == 0
        && hd % 4 == 0
        && (4..=512).contains(&hd)
        && dv % 4 == 0
        && ((4..=256).contains(&dv) || dv == 512)
        && rd <= hd
}

/// The sandwich tail over `k` rows: h = (h + d)·scale, o = rms(h)·w.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_add_norm(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes,
    h: &wgpu::Buffer,
    d: &wgpu::Buffer,
    w: &wgpu::Buffer,
    o: &wgpu::Buffer,
    hidden: usize,
    gemma: bool,
    eps: f32,
    scale: Option<f32>,
    k: usize,
) {
    let u = uniform_u32x4(
        c,
        [
            hidden as u32,
            u32::from(gemma),
            eps.to_bits(),
            scale.map_or(0, f32::to_bits),
        ],
    );
    let bgp = bind(c, &p.add_norm.1, &[(18, h), (19, d), (20, w), (21, o), (22, &u)]);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.add_norm.0);
    pass.set_bind_group(0, &bgp, &[]);
    pass.dispatch_workgroups(k as u32, 1, 1);
}

/// `y[nb][rows] = x[nb][cols] · Wᵀ` for a whole q8_2f tensor buffer through
/// `q8_2f_mm_r2`. False (nothing encoded) when the kernel is missing or the
/// width is not a multiple of its 64-column step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q82_mm(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
    nb: usize,
) -> bool {
    let Some((pipe, l)) = p.mm.as_ref() else {
        return false;
    };
    if cols % 64 != 0 || rows == 0 || nb == 0 || rows.div_ceil(64) > MAX_WG as usize {
        return false;
    }
    let u = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, nb as u32, cols as u32]);
    let bgm = bind(c, l, &[(0, weight), (1, xs), (2, y), (3, &u)]);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bgm, &[]);
    pass.dispatch_workgroups(rows.div_ceil(64) as u32, nb.div_ceil(64) as u32, 1);
    true
}

#[cfg(test)]
mod shader_tests {
    use super::*;

    /// Both modules parse and validate (naga) without a device.
    #[test]
    fn g4_batch_shaders_validate() {
        for (name, src) in [("g4", G4_WGSL), ("q82mm", Q82_MM_WGSL)] {
            let module = wgpu::naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
