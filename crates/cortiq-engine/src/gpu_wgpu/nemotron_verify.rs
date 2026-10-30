//! Two-position decode step of the Nemotron-H frame: the verify pass of
//! draft speculation.
//!
//! The decode kernels of `nemotron.rs` read every weight once per position.
//! Here the same fused kernels take two consecutive positions (the token and
//! its draft) in one pass: every q8_2f row is read once and dotted with both
//! staged inputs, the Mamba-2 step runs both positions in order inside the
//! scan kernel, the routed experts are the union of both positions' picks
//! (twelve bank slots), attention runs causally per position. After the
//! first position the conv rings and scan states are also written to
//! snapshots, so a rejected draft rolls the state back to the token.
//!
//! The arithmetic is the decode frame's model (same codecs, norms and
//! epilogues per position); only the summation order of the q8_2f row loop
//! differs, so an emitted token is the model's argmax as in plain decoding.

use super::*;

/// Positions of one verify pass.
pub(crate) const NR: usize = 2;

pub(crate) const VER_WGSL: &str = r#"
const NR: u32 = 2u;

fn vf_relu2(v: f32) -> f32 {
    let r = max(v, 0.0);
    return r * r;
}
fn vf_i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    return vec4<f32>(f32((s << 24u) >> 24u), f32((s << 16u) >> 24u), f32((s << 8u) >> 24u), f32(s >> 24u));
}

// ── q8_2f matvec of up to three matrices (+ f32 rows) on NR inputs.
// Rows of every buffer are per position: x [NR · cols], zx [NR · zs],
// y/h [NR · rows], eo [NR · k · rows], sw [NR · k]. Prologue/epilogue codes
// as `nf_q8`; epi 4 writes K/V at meta pos + r. The ring shift of the
// B/C channels advances NR positions and keeps the state after the first
// in `snap` (same layout as the ring).
struct VqP {
    rows_a: u32, rows_b: u32, rows_c: u32, cols: u32,
    rpw: u32, pro: u32, epi: u32, eps: f32,
    gs: u32, frows: u32, k: u32, kvw: u32,
    ring_lo: u32, ring_hi: u32, ring_n: u32, ring_off: u32,
    xo: u32, zs: u32, _a: u32, _b: u32,
};
@group(0) @binding(0)  var<storage, read>       vq_wa   : array<u32>;
@group(0) @binding(1)  var<storage, read>       vq_wb   : array<u32>;
@group(0) @binding(2)  var<storage, read>       vq_wc   : array<u32>;
@group(0) @binding(3)  var<storage, read>       vq_x    : array<vec4<f32>>;
@group(0) @binding(4)  var<storage, read>       vq_nw   : array<vec4<f32>>;
@group(0) @binding(5)  var<storage, read>       vq_zx   : array<f32>;
@group(0) @binding(6)  var<storage, read_write> vq_ya   : array<f32>;
@group(0) @binding(7)  var<storage, read_write> vq_yb   : array<f32>;
@group(0) @binding(8)  var<storage, read_write> vq_yc   : array<f32>;
@group(0) @binding(9)  var<storage, read_write> vq_h    : array<f32>;
@group(0) @binding(10) var<storage, read_write> vq_xo   : array<vec4<f32>>;
@group(0) @binding(11) var<storage, read>       vq_eo   : array<f32>;
@group(0) @binding(12) var<storage, read>       vq_sw   : array<f32>;
@group(0) @binding(13) var<storage, read>       vq_fw   : array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> vq_fy   : array<f32>;
@group(0) @binding(15) var<storage, read>       vq_meta : array<u32>;
@group(0) @binding(16) var<storage, read_write> vq_kc   : array<f32>;
@group(0) @binding(17) var<storage, read_write> vq_vc   : array<f32>;
@group(0) @binding(18) var<storage, read_write> vq_ring : array<f32>;
@group(0) @binding(19) var<storage, read_write> vq_snap : array<f32>;
@group(0) @binding(20) var<uniform>             vq_p    : VqP;
var<workgroup> vq_xs: array<vec4<f32>, 2048>;
var<workgroup> vq_red: array<f32, 16>;
var<workgroup> vq_gin: array<f32, 64>;

fn vq_out(m: u32, r: u32, row: u32, v: f32) {
    let e = vq_p.epi;
    if (m == 0u) {
        let o = r * vq_p.rows_a + row;
        if (e == 1u) {
            vq_ya[o] = vf_relu2(v);
        } else if (e == 2u) {
            vq_h[o] = vq_h[o] + v;
        } else if (e == 3u) {
            var s = vq_h[o] + v;
            for (var j = 0u; j < vq_p.k; j = j + 1u) {
                s = s + vq_sw[r * vq_p.k + j] * vq_eo[(r * vq_p.k + j) * vq_p.rows_a + row];
            }
            vq_h[o] = s;
        } else {
            vq_ya[o] = v;
        }
    } else if (m == 1u) {
        if (e == 4u) {
            vq_kc[(vq_meta[0] + r) * vq_p.kvw + row] = v;
        } else if (e == 1u) {
            vq_yb[r * vq_p.rows_b + row] = vf_relu2(v);
        } else {
            vq_yb[r * vq_p.rows_b + row] = v;
        }
    } else {
        if (e == 4u) {
            vq_vc[(vq_meta[0] + r) * vq_p.kvw + row] = v;
        } else if (e == 1u) {
            vq_yc[r * vq_p.rows_c + row] = vf_relu2(v);
        } else {
            vq_yc[r * vq_p.rows_c + row] = v;
        }
    }
}

{VQ_FNS}

fn vq_rows_f(blk: u32, lid: u32, lane: u32) {
    let c4 = vq_p.cols / 4u;
    let row = blk * 8u + lid / 32u;
    let live = row < vq_p.frows;
    let b = select(vq_p.frows - 1u, row, live) * c4;
    var a0 = 0.0;
    var a1 = 0.0;
    for (var k = lane; k < c4; k = k + 32u) {
        let w = vq_fw[b + k];
        a0 = a0 + dot(w, vq_xs[k]);
        a1 = a1 + dot(w, vq_xs[c4 + k]);
    }
    let t0 = subgroupAdd(a0);
    let t1 = subgroupAdd(a1);
    if (lane == 0u && live) {
        vq_fy[row] = t0;
        vq_fy[vq_p.frows + row] = t1;
    }
}

@compute @workgroup_size(256)
fn vf_q8(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(local_invocation_index) lid: u32,
         @builtin(subgroup_invocation_id) lane: u32) {
    let c4 = vq_p.cols / 4u;
    let per = 8u * vq_p.rpw;
    let ba = (vq_p.rows_a + per - 1u) / per;
    let bb = (vq_p.rows_b + per - 1u) / per;
    let bc = (vq_p.rows_c + per - 1u) / per;
    var m = 0u;
    var blk = wid.x;
    if (blk >= ba) {
        blk = blk - ba;
        m = 1u;
        if (blk >= bb) {
            blk = blk - bb;
            m = 2u;
            if (blk >= bc) {
                blk = blk - bc;
                m = 3u;
            }
        }
    }
    var rows = vq_p.rows_a;
    if (m == 1u) { rows = vq_p.rows_b; }
    if (m == 2u) { rows = vq_p.rows_c; }
    let cs0h = ((rows * vq_p.cols) >> 1u) + rows;
    var inv0 = 1.0;
    var inv1 = 1.0;
    let warp = lid / 32u;
    if (vq_p.pro == 1u) {
        var s0 = 0.0;
        var s1 = 0.0;
        for (var i = lid; i < c4; i = i + 256u) {
            let v0 = vq_x[i];
            let v1 = vq_x[c4 + i];
            s0 = s0 + dot(v0, v0);
            s1 = s1 + dot(v1, v1);
        }
        let t0 = subgroupAdd(s0);
        let t1 = subgroupAdd(s1);
        if (lane == 0u) {
            vq_red[warp] = t0;
            vq_red[8u + warp] = t1;
        }
        workgroupBarrier();
        var a = 0.0;
        var b = 0.0;
        for (var w = 0u; w < 8u; w = w + 1u) {
            a = a + vq_red[w];
            b = b + vq_red[8u + w];
        }
        inv0 = inverseSqrt(a / f32(vq_p.cols) + vq_p.eps);
        inv1 = inverseSqrt(b / f32(vq_p.cols) + vq_p.eps);
    }
    if (vq_p.pro == 2u) {
        for (var i = lid; i < NR * c4; i = i + 256u) {
            let r = i / c4;
            let j = i - r * c4;
            let zo = r * vq_p.zs + j * 4u;
            let z = vec4<f32>(vq_zx[zo], vq_zx[zo + 1u], vq_zx[zo + 2u], vq_zx[zo + 3u]);
            vq_xs[i] = vq_x[i] * (z / (vec4<f32>(1.0) + exp(-z)));
        }
        workgroupBarrier();
        let g4 = vq_p.gs / 4u;
        let ng = c4 / g4;
        for (var g = warp; g < NR * ng; g = g + 8u) {
            var acc = 0.0;
            for (var j = lane; j < g4; j = j + 32u) {
                let v = vq_xs[g * g4 + j];
                acc = acc + dot(v, v);
            }
            let t = subgroupAdd(acc);
            if (lane == 0u) { vq_gin[g] = inverseSqrt(t / f32(vq_p.gs) + vq_p.eps); }
        }
        workgroupBarrier();
    }
    for (var i = lid; i < NR * c4; i = i + 256u) {
        let r = i / c4;
        let j = i - r * c4;
        var v = vq_x[i];
        if (vq_p.pro == 1u) {
            v = v * select(inv0, inv1, r == 1u) * vq_nw[j];
            if (vq_p.xo != 0u && wid.x == 0u) { vq_xo[i] = v; }
        } else if (vq_p.pro == 2u) {
            v = vq_xs[i] * vq_gin[(i * 4u) / vq_p.gs] * vq_nw[j];
        }
        if (m == 0u) { v = v * vq_cs4_a(cs0h + j * 4u); }
        if (m == 1u) { v = v * vq_cs4_b(cs0h + j * 4u); }
        if (m == 2u) { v = v * vq_cs4_c(cs0h + j * 4u); }
        vq_xs[i] = v;
    }
    workgroupBarrier();
    if (wid.x == 0u && vq_p.ring_n > 0u) {
        // B/C ring channels: (r0, r1, r2) → after x0: (r1, r2, x0) [snapshot]
        // → after x1: (r2, x0, x1)
        let n = vq_p.ring_n;
        for (var c = vq_p.ring_lo + lid; c < vq_p.ring_hi; c = c + 256u) {
            let r1 = vq_ring[n + c];
            let r2 = vq_ring[2u * n + c];
            let x0 = vq_zx[vq_p.ring_off + c];
            let x1 = vq_zx[vq_p.zs + vq_p.ring_off + c];
            vq_snap[c] = r1;
            vq_snap[n + c] = r2;
            vq_snap[2u * n + c] = x0;
            vq_ring[c] = r2;
            vq_ring[n + c] = x0;
            vq_ring[2u * n + c] = x1;
        }
    }
    if (m == 0u) {
        vq_rows_a(0u, blk, lid, lane);
    } else if (m == 1u) {
        vq_rows_b(1u, blk, lid, lane);
    } else if (m == 2u) {
        vq_rows_c(2u, blk, lid, lane);
    } else {
        vq_rows_f(blk, lid, lane);
    }
}

// ── Mamba-2 step of one head over NR positions: conv taps of each
// position from the ring and the pass's own inputs, then the scan in order;
// after the first position the head's state and x-channel ring go to the
// snapshots. zx rows [NR · zs], y rows [NR · inner].
struct VsP { inner: u32, dt_off: u32, ng: u32, ns: u32, hpg: u32, hdim: u32, nh: u32, cdim: u32, zs: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       vs_zx    : array<f32>;
@group(0) @binding(1) var<storage, read>       vs_cw    : array<f32>;
@group(0) @binding(2) var<storage, read>       vs_cb    : array<f32>;
@group(0) @binding(3) var<storage, read_write> vs_ring  : array<f32>;
@group(0) @binding(4) var<storage, read_write> vs_s     : array<f32>;
@group(0) @binding(5) var<storage, read>       vs_par   : array<f32>;
@group(0) @binding(6) var<storage, read_write> vs_y     : array<f32>;
@group(0) @binding(7) var<storage, read_write> vs_snap  : array<f32>;
@group(0) @binding(8) var<storage, read_write> vs_ssnap : array<f32>;
@group(0) @binding(9) var<uniform>             vs_p     : VsP;
var<workgroup> vs_b: array<f32, 512>;
var<workgroup> vs_c: array<f32, 512>;
var<workgroup> vs_x: array<f32, 128>;
var<workgroup> vs_part: array<f32, 256>;
@compute @workgroup_size(256)
fn vf_scan(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let ns = vs_p.ns;
    let hdim = vs_p.hdim;
    let inner = vs_p.inner;
    let n = vs_p.cdim;
    let g = h / vs_p.hpg;
    for (var j = lid; j < hdim + 2u * ns; j = j + 256u) {
        var c = h * hdim + j;
        if (j >= hdim + ns) {
            c = inner + vs_p.ng * ns + g * ns + (j - hdim - ns);
        } else if (j >= hdim) {
            c = inner + g * ns + (j - hdim);
        }
        let x0 = vs_zx[inner + c];
        let x1 = vs_zx[vs_p.zs + inner + c];
        let r0 = vs_ring[c];
        let r1 = vs_ring[n + c];
        let r2 = vs_ring[2u * n + c];
        let w = c * 4u;
        let a0 = vs_cb[c] + vs_cw[w] * r0 + vs_cw[w + 1u] * r1 + vs_cw[w + 2u] * r2 + vs_cw[w + 3u] * x0;
        let a1 = vs_cb[c] + vs_cw[w] * r1 + vs_cw[w + 1u] * r2 + vs_cw[w + 2u] * x0 + vs_cw[w + 3u] * x1;
        let v0 = a0 / (1.0 + exp(-a0));
        let v1 = a1 / (1.0 + exp(-a1));
        if (j < hdim) {
            vs_x[j] = v0;
            vs_x[64u + j] = v1;
            vs_snap[c] = r1;
            vs_snap[n + c] = r2;
            vs_snap[2u * n + c] = x0;
            vs_ring[c] = r2;
            vs_ring[n + c] = x0;
            vs_ring[2u * n + c] = x1;
        } else if (j < hdim + ns) {
            vs_b[j - hdim] = v0;
            vs_b[256u + j - hdim] = v1;
        } else {
            vs_c[j - hdim - ns] = v0;
            vs_c[256u + j - hdim - ns] = v1;
        }
    }
    workgroupBarrier();
    let p = lid % 64u;
    let kq = lid / 64u;
    let q = ns / 4u;
    let base = h * ns * hdim + p;
    for (var r = 0u; r < NR; r = r + 1u) {
        let raw = vs_zx[r * vs_p.zs + vs_p.dt_off + h] + vs_par[2u * vs_p.nh + h];
        var dt = log(1.0 + exp(raw));
        if (raw > 20.0) { dt = raw; }
        if (raw < -15.0) { dt = exp(raw); }
        let da = exp(dt * vs_par[h]);
        let x = vs_x[r * 64u + p];
        let dx = dt * x;
        var acc = 0.0;
        for (var k = kq * q; k < (kq + 1u) * q; k = k + 1u) {
            let i = base + k * hdim;
            let v = da * vs_s[i] + dx * vs_b[r * 256u + k];
            vs_s[i] = v;
            if (r == 0u) { vs_ssnap[i] = v; }
            acc = acc + v * vs_c[r * 256u + k];
        }
        vs_part[lid] = acc;
        workgroupBarrier();
        if (kq == 0u) {
            vs_y[r * inner + h * hdim + p] = vs_part[p] + vs_part[64u + p] + vs_part[128u + p] + vs_part[192u + p]
                + vs_par[vs_p.nh + h] * x;
        }
        workgroupBarrier();
    }
}

// ── routed experts over NR positions: workgroup (row block, slot), slot
// = position·k + rank. `route` (up): each workgroup ranks its position's
// router scores; workgroup (0, r·k) stores the position's picks/weights.
// x row: (slot / xdiv)·x_stride; y row: slot·y_stride. Full banks.
struct VbP {
    rows: u32, gpr: u32, stride_w: u32, act: u32,
    x_stride: u32, y_stride: u32, route: u32, n_exp: u32,
    k: u32, scale: f32, norm: u32, xdiv: u32,
    cols: u32, _a: u32, _b: u32, _c: u32,
};
@group(0) @binding(0) var<storage, read>       vb_w    : array<u32>;
@group(0) @binding(1) var<storage, read>       vb_w4   : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       vb_x    : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       vb_lg   : array<f32>;
@group(0) @binding(4) var<storage, read>       vb_bias : array<f32>;
@group(0) @binding(5) var<storage, read_write> vb_sel  : array<u32>;
@group(0) @binding(6) var<storage, read_write> vb_sw   : array<f32>;
@group(0) @binding(7) var<storage, read_write> vb_y    : array<f32>;
@group(0) @binding(8) var<uniform>             vb_p    : VbP;
var<workgroup> vb_xs: array<vec4<f32>, 1024>;
var<workgroup> vb_c: array<f32, 256>;
var<workgroup> vb_s: array<u32, 16>;
var<workgroup> vb_slot: u32;

fn vb_prologue(wid: vec3<u32>, lid: u32) -> u32 {
    let n = vb_p.n_exp;
    let k = vb_p.k;
    let r = wid.y / k;
    if (lid < 16u) { vb_s[lid] = 0u; }
    if (vb_p.route == 1u) {
        if (lid < n) { vb_c[lid] = 1.0 / (1.0 + exp(-vb_lg[r * n + lid])) + vb_bias[lid]; }
        workgroupBarrier();
        if (lid < n) {
            let ci = vb_c[lid];
            var rank = 0u;
            for (var j = 0u; j < n; j = j + 1u) {
                let cj = vb_c[j];
                if (cj > ci || (cj == ci && j < lid)) { rank = rank + 1u; }
            }
            if (rank < k) { vb_s[rank] = lid; }
        }
        workgroupBarrier();
        if (wid.x == 0u && wid.y % k == 0u && lid == 0u) {
            var sum = 0.0;
            for (var t = 0u; t < k; t = t + 1u) {
                sum = sum + 1.0 / (1.0 + exp(-vb_lg[r * n + vb_s[t]]));
            }
            var den = 1.0;
            if (vb_p.norm != 0u) { den = sum + 1e-20; }
            for (var t = 0u; t < k; t = t + 1u) {
                let e = vb_s[t];
                vb_sel[r * k + t] = e;
                vb_sw[r * k + t] = (1.0 / (1.0 + exp(-vb_lg[r * n + e]))) / den * vb_p.scale;
            }
        }
    } else {
        workgroupBarrier();
        if (lid < k) { vb_s[lid] = vb_sel[r * k + lid]; }
        workgroupBarrier();
    }
    if (lid == 0u) { vb_slot = vb_s[wid.y % k]; }
    return workgroupUniformLoad(&vb_slot);
}

fn vb_byte(off: u32) -> u32 { return (vb_w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn vb_code(cb: u32, g: u32) -> u32 {
    let bit = g * 5u;
    let off = cb + (bit >> 3u);
    return ((vb_byte(off) | (vb_byte(off + 1u) << 8u)) >> (bit & 7u)) & 31u;
}
fn vb_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn vb_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return vb_nib(w, 0u) * a.x + vb_nib(w, 4u) * a.y + vb_nib(w, 8u) * a.z + vb_nib(w, 12u) * a.w
         + vb_nib(w, 16u) * b.x + vb_nib(w, 20u) * b.y + vb_nib(w, 24u) * b.z + vb_nib(w, 28u) * b.w;
}
fn vb_dot32(n: vec4<u32>, g: u32, gpr: u32) -> f32 {
    return vb_dot8(n.x, vb_xs[g], vb_xs[gpr + g]) + vb_dot8(n.y, vb_xs[2u * gpr + g], vb_xs[3u * gpr + g])
         + vb_dot8(n.z, vb_xs[4u * gpr + g], vb_xs[5u * gpr + g]) + vb_dot8(n.w, vb_xs[6u * gpr + g], vb_xs[7u * gpr + g]);
}
fn vb_out(slot: u32, row: u32, v: f32) {
    var o = v;
    if (vb_p.act == 1u) { o = vf_relu2(o); }
    vb_y[slot * vb_p.y_stride + row] = o;
}

@compute @workgroup_size(256)
fn vf_bank4(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32,
            @builtin(subgroup_invocation_id) lane: u32) {
    let ds = vb_prologue(wid, lid);
    let slot = wid.y;
    let rows = vb_p.rows;
    let gpr = vb_p.gpr;
    let c4 = gpr * 8u;
    let xo = (slot / vb_p.xdiv) * vb_p.x_stride / 4u;
    for (var i = lid; i < c4; i = i + 256u) {
        vb_xs[(i % 8u) * gpr + i / 8u] = vb_x[xo + i];
    }
    workgroupBarrier();
    let bw = ds * vb_p.stride_w;
    let params_w = bw + rows * gpr * 4u;
    let codes_b = (params_w + rows) * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let row0 = (wid.x * 8u + lid / 32u) * 4u;
    let r0 = min(row0, rows - 1u);
    let r1 = min(row0 + 1u, rows - 1u);
    let r2 = min(row0 + 2u, rows - 1u);
    let r3 = min(row0 + 3u, rows - 1u);
    let p0 = unpack2x16float(vb_w[params_w + r0]);
    let p1 = unpack2x16float(vb_w[params_w + r1]);
    let p2 = unpack2x16float(vb_w[params_w + r2]);
    let p3 = unpack2x16float(vb_w[params_w + r3]);
    let nb = bw / 4u;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    for (var g = lane; g < gpr; g = g + 32u) {
        let w0 = vb_w4[nb + r0 * gpr + g];
        let w1 = vb_w4[nb + r1 * gpr + g];
        let w2 = vb_w4[nb + r2 * gpr + g];
        let w3 = vb_w4[nb + r3 * gpr + g];
        let k0 = vb_code(codes_b + r0 * cstride, g);
        let k1 = vb_code(codes_b + r1 * cstride, g);
        let k2 = vb_code(codes_b + r2 * cstride, g);
        let k3 = vb_code(codes_b + r3 * cstride, g);
        a0 = a0 + exp2(p0.x + f32(k0) * p0.y) * vb_dot32(w0, g, gpr);
        a1 = a1 + exp2(p1.x + f32(k1) * p1.y) * vb_dot32(w1, g, gpr);
        a2 = a2 + exp2(p2.x + f32(k2) * p2.y) * vb_dot32(w2, g, gpr);
        a3 = a3 + exp2(p3.x + f32(k3) * p3.y) * vb_dot32(w3, g, gpr);
    }
    let t0 = subgroupAdd(a0);
    let t1 = subgroupAdd(a1);
    let t2 = subgroupAdd(a2);
    let t3 = subgroupAdd(a3);
    if (lane == 0u) {
        if (row0 < rows) { vb_out(slot, row0, t0); }
        if (row0 + 1u < rows) { vb_out(slot, row0 + 1u, t1); }
        if (row0 + 2u < rows) { vb_out(slot, row0 + 2u, t2); }
        if (row0 + 3u < rows) { vb_out(slot, row0 + 3u, t3); }
    }
}

fn vb_cs4(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(vb_w[w]);
    let b = unpack2x16float(vb_w[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(vb_w[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
fn vb_rs(rs0: u32, r: u32) -> f32 {
    let rw = unpack2x16float(vb_w[rs0 + (r >> 1u)]);
    return select(rw.x, rw.y, (r & 1u) == 1u);
}

@compute @workgroup_size(256)
fn vf_bank8(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32,
            @builtin(subgroup_invocation_id) lane: u32) {
    let ds = vb_prologue(wid, lid);
    let slot = wid.y;
    let rows = vb_p.rows;
    let words = vb_p.cols / 4u;
    let bw = ds * vb_p.stride_w;
    let cs0h = bw * 2u + ((rows * vb_p.cols) >> 1u) + rows;
    let xo = (slot / vb_p.xdiv) * vb_p.x_stride / 4u;
    for (var i = lid; i < words; i = i + 256u) {
        vb_xs[i] = vb_x[xo + i] * vb_cs4(cs0h + i * 4u);
    }
    workgroupBarrier();
    let row0 = (wid.x * 8u + lid / 32u) * 4u;
    let b0 = bw + min(row0, rows - 1u) * words;
    let b1 = bw + min(row0 + 1u, rows - 1u) * words;
    let b2 = bw + min(row0 + 2u, rows - 1u) * words;
    let b3 = bw + min(row0 + 3u, rows - 1u) * words;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    for (var k = lane; k < words; k = k + 32u) {
        let x = vb_xs[k];
        a0 = a0 + dot(vf_i8x4(vb_w[b0 + k]), x);
        a1 = a1 + dot(vf_i8x4(vb_w[b1 + k]), x);
        a2 = a2 + dot(vf_i8x4(vb_w[b2 + k]), x);
        a3 = a3 + dot(vf_i8x4(vb_w[b3 + k]), x);
    }
    // keep the vec4 view in the binding set (auto layout): never true
    if (rows == 0xFFFFFFFFu) { a0 = a0 + f32(vb_w4[0].x); }
    let t0 = subgroupAdd(a0);
    let t1 = subgroupAdd(a1);
    let t2 = subgroupAdd(a2);
    let t3 = subgroupAdd(a3);
    let rs0 = bw + ((rows * vb_p.cols) >> 2u);
    if (lane == 0u) {
        if (row0 < rows) { vb_out(slot, row0, t0 * vb_rs(rs0, row0)); }
        if (row0 + 1u < rows) { vb_out(slot, row0 + 1u, t1 * vb_rs(rs0, row0 + 1u)); }
        if (row0 + 2u < rows) { vb_out(slot, row0 + 2u, t2 * vb_rs(rs0, row0 + 2u)); }
        if (row0 + 3u < rows) { vb_out(slot, row0 + 3u, t3 * vb_rs(rs0, row0 + 3u)); }
    }
}

// ── final RMSNorm, workgroup per position ──
struct VrP { n: u32, eps: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       vr_h : array<f32>;
@group(0) @binding(1) var<storage, read>       vr_w : array<f32>;
@group(0) @binding(2) var<storage, read_write> vr_y : array<f32>;
@group(0) @binding(3) var<uniform>             vr_p : VrP;
var<workgroup> vr_part: array<f32, 256>;
@compute @workgroup_size(256)
fn vf_rms(@builtin(workgroup_id) wid: vec3<u32>,
          @builtin(local_invocation_index) lid: u32) {
    let n = vr_p.n;
    let o = wid.x * n;
    var acc = 0.0;
    for (var i = lid; i < n; i = i + 256u) {
        let v = vr_h[o + i];
        acc = acc + v * v;
    }
    vr_part[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { vr_part[lid] = vr_part[lid] + vr_part[lid + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(vr_part[0] / f32(n) + vr_p.eps);
    for (var i = lid; i < n; i = i + 256u) {
        vr_y[o + i] = vr_w[i] * (vr_h[o + i] * inv);
    }
}

// ── argmax of each position's logits: workgroup per position ──
struct VxP { vocab: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       vx_l   : array<f32>;
@group(0) @binding(1) var<storage, read_write> vx_ids : array<u32>;
@group(0) @binding(2) var<uniform>             vx_p   : VxP;
var<workgroup> vx_v: array<f32, 256>;
var<workgroup> vx_i: array<u32, 256>;
@compute @workgroup_size(256)
fn vf_argmax(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let o = wid.x * vx_p.vocab;
    var bv = -3.0e38;
    var bi = 0u;
    for (var i = lid; i < vx_p.vocab; i = i + 256u) {
        let v = vx_l[o + i];
        if (v > bv) { bv = v; bi = i; }
    }
    vx_v[lid] = bv;
    vx_i[lid] = bi;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let a = vx_v[lid];
            let c = vx_v[lid + s];
            if (c > a || (c == a && vx_i[lid + s] < vx_i[lid])) {
                vx_v[lid] = c;
                vx_i[lid] = vx_i[lid + s];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) { vx_ids[wid.x] = vx_i[0]; }
}
"#;

/// The q8_2f row loop of `vf_q8` for both positions, one copy per matrix
/// binding: two rows a warp, four words a lane per step, each word dotted
/// with both staged inputs.
const VQ_FN_T: &str = r#"
fn vq_cs4_{M}(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(vq_w{M}[w]);
    let b = unpack2x16float(vq_w{M}[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(vq_w{M}[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
fn vq_rows_{M}(m: u32, blk: u32, lid: u32, lane: u32) {
    let rows = vq_p.rows_{M};
    let words = vq_p.cols / 4u;
    let rs0 = (rows * vq_p.cols) >> 2u;
    let warp = lid / 32u;
    for (var r = 0u; r < vq_p.rpw; r = r + 2u) {
        let row0 = (blk * vq_p.rpw + r) * 8u + warp * 2u;
        let live0 = row0 < rows;
        let live1 = row0 + 1u < rows;
        let r0 = select(rows - 1u, row0, live0);
        let r1 = select(rows - 1u, row0 + 1u, live1);
        let b0 = r0 * words;
        let b1 = r1 * words;
        var a00 = 0.0;
        var a01 = 0.0;
        var a10 = 0.0;
        var a11 = 0.0;
        var k = lane;
        loop {
            if (k + 96u >= words) { break; }
            let p0 = vf_i8x4(vq_w{M}[b0 + k]);
            let p1 = vf_i8x4(vq_w{M}[b0 + k + 32u]);
            let p2 = vf_i8x4(vq_w{M}[b0 + k + 64u]);
            let p3 = vf_i8x4(vq_w{M}[b0 + k + 96u]);
            let q0 = vf_i8x4(vq_w{M}[b1 + k]);
            let q1 = vf_i8x4(vq_w{M}[b1 + k + 32u]);
            let q2 = vf_i8x4(vq_w{M}[b1 + k + 64u]);
            let q3 = vf_i8x4(vq_w{M}[b1 + k + 96u]);
            let x0 = vq_xs[k];
            let x1 = vq_xs[k + 32u];
            let x2 = vq_xs[k + 64u];
            let x3 = vq_xs[k + 96u];
            let y0 = vq_xs[words + k];
            let y1 = vq_xs[words + k + 32u];
            let y2 = vq_xs[words + k + 64u];
            let y3 = vq_xs[words + k + 96u];
            a00 = a00 + dot(p0, x0) + dot(p1, x1) + dot(p2, x2) + dot(p3, x3);
            a01 = a01 + dot(p0, y0) + dot(p1, y1) + dot(p2, y2) + dot(p3, y3);
            a10 = a10 + dot(q0, x0) + dot(q1, x1) + dot(q2, x2) + dot(q3, x3);
            a11 = a11 + dot(q0, y0) + dot(q1, y1) + dot(q2, y2) + dot(q3, y3);
            k = k + 128u;
        }
        loop {
            if (k >= words) { break; }
            let p = vf_i8x4(vq_w{M}[b0 + k]);
            let q = vf_i8x4(vq_w{M}[b1 + k]);
            let x = vq_xs[k];
            let y = vq_xs[words + k];
            a00 = a00 + dot(p, x);
            a01 = a01 + dot(p, y);
            a10 = a10 + dot(q, x);
            a11 = a11 + dot(q, y);
            k = k + 32u;
        }
        let t00 = subgroupAdd(a00);
        let t01 = subgroupAdd(a01);
        let t10 = subgroupAdd(a10);
        let t11 = subgroupAdd(a11);
        if (lane == 0u && live0) {
            let rw = unpack2x16float(vq_w{M}[rs0 + (r0 >> 1u)]);
            let sc = select(rw.x, rw.y, (r0 & 1u) == 1u);
            vq_out(m, 0u, r0, t00 * sc);
            vq_out(m, 1u, r0, t01 * sc);
        }
        if (lane == 0u && live1) {
            let rw = unpack2x16float(vq_w{M}[rs0 + (r1 >> 1u)]);
            let sc = select(rw.x, rw.y, (r1 & 1u) == 1u);
            vq_out(m, 0u, r1, t10 * sc);
            vq_out(m, 1u, r1, t11 * sc);
        }
    }
}
"#;

/// The verify module with the per-matrix row loops expanded.
pub(crate) fn ver_wgsl() -> String {
    let mut fns = String::new();
    for m in ["a", "b", "c"] {
        fns.push_str(&VQ_FN_T.replace("{M}", m));
    }
    VER_WGSL.replace("{VQ_FNS}", &fns)
}
