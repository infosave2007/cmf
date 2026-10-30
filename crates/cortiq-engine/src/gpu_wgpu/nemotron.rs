//! Nemotron-H (Nemotron 3.5 Transcribe decoder) device-resident token path.
//!
//! The whole decode step runs in one compute pass of ~190 dispatches. Every
//! RMSNorm is folded into the projection that reads it (each workgroup
//! reduces the residual itself, workgroup 0 keeps the normed row when a
//! later kernel needs it), every residual add into the epilogue of the
//! projection that produces it, the Mamba-2 conv into the scan, the gated
//! group norm into the out projection, the router's top-k into the expert
//! kernel, the argmax's last stage into the embedding of the winner.
//! Weights: q8_2f projections, q4tp or q8_2f routed experts in per-layer
//! banks, f32 router. `steps` greedy tokens ride one submit and only their
//! ids come back; with full banks two submits are kept in flight.
//!
//! A bank may hold only some of its layer's experts (the card's budget, or
//! `CMF_NEMO_EXPERT_GB`): a per-layer map gives each expert's slot. A token
//! that routes to a missing expert stops at that layer — the expert kernel
//! raises `abort` in the meta words and every later kernel returns at once
//! — the host completes the missing experts on the CPU from the mapped file
//! and the step resumes at the layer's expert kernels. Experts the decode
//! keeps routing to are admitted in place of the least used.
//!
//! The frame holds ONE sequence. It is seeded from the host state after a
//! host prefill (conv rings and KV rows verbatim, the scan state transposed
//! to the device layout `[head][state][head_dim]`) or prefilled in place.

use super::*;

pub(crate) const NEMO_WGSL: &str = r#"
// meta words: 0 pos, 1 abort, 2 abort layer, 3 abort pos, 4 provided layer,
// 5 provided slot mask, 6 step (ids index). Every kernel that changes
// persistent state returns at once while `abort` is set.

fn nf_relu2(v: f32) -> f32 {
    let r = max(v, 0.0);
    return r * r;
}
fn nf_i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    let b0 = (s << 24u) >> 24u;
    let b1 = (s << 16u) >> 24u;
    let b2 = (s <<  8u) >> 24u;
    let b3 =  s          >> 24u;
    return vec4<f32>(f32(b0), f32(b1), f32(b2), f32(b3));
}

// ── q8_2f matvec of up to three matrices on one input plus an optional
// segment of f32 rows (the router), one dispatch.
// Prologue `pro`: 0 x as is; 1 x = rms(h)·w (workgroup 0 stores it in `xo`
// when `xo` is set); 2 x = groupRMS(y·silu(z))·w over groups of `gs`.
// Epilogue `epi` (matrix a): 0 store; 1 relu²; 2 h += y; 3 h += y +
// Σ_s sw[s]·eo[s] (the MoE sum); 4 qkv: a → ya, b/c → K/V cache rows at
// meta pos. relu² applies to b/c as well when epi = 1.
// Workgroup 0 also shifts the conv ring channels [ring_lo, ring_hi) when
// ring_n > 0 (the Mamba B/C channels; the scan shifts its own x channels).
// `row_cols` > 0: the matrices are `row_cols` wide and the pass reads the
// `cols` columns from `col_off` (the halves of the draft's fusion).
struct NqP {
    rows_a: u32, rows_b: u32, rows_c: u32, cols: u32,
    rpw: u32, pro: u32, epi: u32, eps: f32,
    gs: u32, frows: u32, k: u32, kvw: u32,
    ring_lo: u32, ring_hi: u32, ring_n: u32, ring_off: u32,
    xo: u32, col_off: u32, row_cols: u32, _c: u32,
};
@group(0) @binding(0)  var<storage, read>       nq_wa   : array<u32>;
@group(0) @binding(1)  var<storage, read>       nq_wb   : array<u32>;
@group(0) @binding(2)  var<storage, read>       nq_wc   : array<u32>;
@group(0) @binding(3)  var<storage, read>       nq_x    : array<vec4<f32>>;
@group(0) @binding(4)  var<storage, read>       nq_nw   : array<vec4<f32>>;
@group(0) @binding(5)  var<storage, read>       nq_zx   : array<f32>;
@group(0) @binding(6)  var<storage, read_write> nq_ya   : array<f32>;
@group(0) @binding(7)  var<storage, read_write> nq_yb   : array<f32>;
@group(0) @binding(8)  var<storage, read_write> nq_yc   : array<f32>;
@group(0) @binding(9)  var<storage, read_write> nq_h    : array<f32>;
@group(0) @binding(10) var<storage, read_write> nq_xo   : array<vec4<f32>>;
@group(0) @binding(11) var<storage, read>       nq_eo   : array<f32>;
@group(0) @binding(12) var<storage, read>       nq_sw   : array<f32>;
@group(0) @binding(13) var<storage, read>       nq_fw   : array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> nq_fy   : array<f32>;
@group(0) @binding(15) var<storage, read_write> nq_meta : array<u32>;
@group(0) @binding(16) var<storage, read_write> nq_kc   : array<f32>;
@group(0) @binding(17) var<storage, read_write> nq_vc   : array<f32>;
@group(0) @binding(18) var<storage, read_write> nq_ring : array<f32>;
@group(0) @binding(19) var<uniform>             nq_p    : NqP;
var<workgroup> nq_xs: array<vec4<f32>, 1024>;
var<workgroup> nq_red: array<f32, 8>;
var<workgroup> nq_gin: array<f32, 32>;
var<workgroup> nq_flag: u32;

fn nq_out(m: u32, r: u32, v: f32) {
    let e = nq_p.epi;
    if (m == 0u) {
        if (e == 1u) {
            nq_ya[r] = nf_relu2(v);
        } else if (e == 2u) {
            nq_h[r] = nq_h[r] + v;
        } else if (e == 3u) {
            var s = nq_h[r] + v;
            for (var j = 0u; j < nq_p.k; j = j + 1u) {
                s = s + nq_sw[j] * nq_eo[j * nq_p.rows_a + r];
            }
            nq_h[r] = s;
        } else {
            nq_ya[r] = v;
        }
    } else if (m == 1u) {
        if (e == 4u) {
            nq_kc[nq_meta[0] * nq_p.kvw + r] = v;
        } else if (e == 1u) {
            nq_yb[r] = nf_relu2(v);
        } else {
            nq_yb[r] = v;
        }
    } else {
        if (e == 4u) {
            nq_vc[nq_meta[0] * nq_p.kvw + r] = v;
        } else if (e == 1u) {
            nq_yc[r] = nf_relu2(v);
        } else {
            nq_yc[r] = v;
        }
    }
}

{NQ_FNS}

fn nq_rows_f(blk: u32, lid: u32, lane: u32) {
    let c4 = nq_p.cols / 4u;
    let row = blk * 8u + lid / 32u;
    let live = row < nq_p.frows;
    let b = select(nq_p.frows - 1u, row, live) * c4;
    var acc = 0.0;
    var k = lane;
    loop {
        if (k + 96u >= c4) { break; }
        acc = acc + dot(nq_fw[b + k], nq_xs[k]) + dot(nq_fw[b + k + 32u], nq_xs[k + 32u])
            + dot(nq_fw[b + k + 64u], nq_xs[k + 64u]) + dot(nq_fw[b + k + 96u], nq_xs[k + 96u]);
        k = k + 128u;
    }
    loop {
        if (k >= c4) { break; }
        acc = acc + dot(nq_fw[b + k], nq_xs[k]);
        k = k + 32u;
    }
    let t = subgroupAdd(acc);
    if (lane == 0u && live) { nq_fy[row] = t; }
}

@compute @workgroup_size(256)
fn nf_q8(@builtin(workgroup_id) wid: vec3<u32>,
         @builtin(local_invocation_index) lid: u32,
         @builtin(subgroup_invocation_id) lane: u32) {
    if (lid == 0u) { nq_flag = nq_meta[1]; }
    if (workgroupUniformLoad(&nq_flag) != 0u) { return; }
    let c4 = nq_p.cols / 4u;
    let per = 8u * nq_p.rpw;
    let ba = (nq_p.rows_a + per - 1u) / per;
    let bb = (nq_p.rows_b + per - 1u) / per;
    let bc = (nq_p.rows_c + per - 1u) / per;
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
    var rows = nq_p.rows_a;
    if (m == 1u) { rows = nq_p.rows_b; }
    if (m == 2u) { rows = nq_p.rows_c; }
    let rc = select(nq_p.cols, nq_p.row_cols, nq_p.row_cols != 0u);
    let cs0h = ((rows * rc) >> 1u) + rows + nq_p.col_off;
    var inv = 1.0;
    if (nq_p.pro == 1u) {
        var acc = 0.0;
        for (var i = lid; i < c4; i = i + 256u) {
            let v = nq_x[i];
            acc = acc + dot(v, v);
        }
        let t = subgroupAdd(acc);
        if (lane == 0u) { nq_red[lid / 32u] = t; }
        workgroupBarrier();
        var tot = 0.0;
        for (var w = 0u; w < 8u; w = w + 1u) { tot = tot + nq_red[w]; }
        inv = inverseSqrt(tot / f32(nq_p.cols) + nq_p.eps);
    }
    if (nq_p.pro == 2u) {
        for (var i = lid; i < c4; i = i + 256u) {
            let z = vec4<f32>(nq_zx[i * 4u], nq_zx[i * 4u + 1u], nq_zx[i * 4u + 2u], nq_zx[i * 4u + 3u]);
            nq_xs[i] = nq_x[i] * (z / (vec4<f32>(1.0) + exp(-z)));
        }
        workgroupBarrier();
        let g4 = nq_p.gs / 4u;
        let ng = c4 / g4;
        let warp = lid / 32u;
        for (var g = warp; g < ng; g = g + 8u) {
            var acc = 0.0;
            for (var j = lane; j < g4; j = j + 32u) {
                let v = nq_xs[g * g4 + j];
                acc = acc + dot(v, v);
            }
            let t = subgroupAdd(acc);
            if (lane == 0u) { nq_gin[g] = inverseSqrt(t / f32(nq_p.gs) + nq_p.eps); }
        }
        workgroupBarrier();
    }
    for (var i = lid; i < c4; i = i + 256u) {
        var v = nq_x[i];
        if (nq_p.pro == 1u) {
            v = v * inv * nq_nw[i];
            if (nq_p.xo != 0u && wid.x == 0u) { nq_xo[i] = v; }
        } else if (nq_p.pro == 2u) {
            v = nq_xs[i] * nq_gin[(i * 4u) / nq_p.gs] * nq_nw[i];
        }
        if (m == 0u) { v = v * nq_cs4_a(cs0h + i * 4u); }
        if (m == 1u) { v = v * nq_cs4_b(cs0h + i * 4u); }
        if (m == 2u) { v = v * nq_cs4_c(cs0h + i * 4u); }
        nq_xs[i] = v;
    }
    workgroupBarrier();
    if (wid.x == 0u) {
        if (nq_p.ring_n > 0u) {
            let n = nq_p.ring_n;
            for (var c = nq_p.ring_lo + lid; c < nq_p.ring_hi; c = c + 256u) {
                nq_ring[c] = nq_ring[n + c];
                nq_ring[n + c] = nq_ring[2u * n + c];
                nq_ring[2u * n + c] = nq_zx[nq_p.ring_off + c];
            }
        }
        // the MoE sum closes the layer: drop the host's provided slots
        if (nq_p.epi == 3u && lid == 0u) { nq_meta[4] = 0xFFFFFFFFu; }
    }
    if (m == 0u) {
        nq_rows_a(0u, blk, lid, lane);
    } else if (m == 1u) {
        nq_rows_b(1u, blk, lid, lane);
    } else if (m == 2u) {
        nq_rows_c(2u, blk, lid, lane);
    } else {
        nq_rows_f(blk, lid, lane);
    }
}

// ── Mamba-2 step of one head: the depthwise conv (kernel 4) + SiLU of the
// head's x channels and its group's B/C channels, then the selective scan.
// The head shifts the ring of its own x channels; the B/C channels, read
// by every head of the group, are shifted by the out projection.
// State layout [head][state][head_dim]; par = [A(nh) | D(nh) | dt_bias(nh)].
struct ScP { inner: u32, dt_off: u32, ng: u32, ns: u32, hpg: u32, hdim: u32, nh: u32, cdim: u32 };
@group(0) @binding(0) var<storage, read>       sc_zx   : array<f32>;
@group(0) @binding(1) var<storage, read>       sc_cw   : array<f32>;
@group(0) @binding(2) var<storage, read>       sc_cb   : array<f32>;
@group(0) @binding(3) var<storage, read_write> sc_ring : array<f32>;
@group(0) @binding(4) var<storage, read_write> sc_s    : array<f32>;
@group(0) @binding(5) var<storage, read>       sc_par  : array<f32>;
@group(0) @binding(6) var<storage, read_write> sc_y    : array<f32>;
@group(0) @binding(7) var<storage, read>       sc_meta : array<u32>;
@group(0) @binding(8) var<uniform>             sc_p    : ScP;
var<workgroup> sc_b: array<f32, 256>;
var<workgroup> sc_c: array<f32, 256>;
var<workgroup> sc_x: array<f32, 64>;
var<workgroup> sc_part: array<f32, 256>;
var<workgroup> sc_flag: u32;
@compute @workgroup_size(256)
fn nf_scan(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    if (lid == 0u) { sc_flag = sc_meta[1]; }
    if (workgroupUniformLoad(&sc_flag) != 0u) { return; }
    let h = wid.x;
    let ns = sc_p.ns;
    let hdim = sc_p.hdim;
    let inner = sc_p.inner;
    let n = sc_p.cdim;
    let g = h / sc_p.hpg;
    for (var j = lid; j < hdim + 2u * ns; j = j + 256u) {
        var c = h * hdim + j;
        if (j >= hdim + ns) {
            c = inner + sc_p.ng * ns + g * ns + (j - hdim - ns);
        } else if (j >= hdim) {
            c = inner + g * ns + (j - hdim);
        }
        let xv = sc_zx[inner + c];
        let r0 = sc_ring[c];
        let r1 = sc_ring[n + c];
        let r2 = sc_ring[2u * n + c];
        let w = c * 4u;
        let acc = sc_cb[c] + sc_cw[w] * r0 + sc_cw[w + 1u] * r1 + sc_cw[w + 2u] * r2 + sc_cw[w + 3u] * xv;
        let v = acc / (1.0 + exp(-acc));
        if (j < hdim) {
            sc_x[j] = v;
            sc_ring[c] = r1;
            sc_ring[n + c] = r2;
            sc_ring[2u * n + c] = xv;
        } else if (j < hdim + ns) {
            sc_b[j - hdim] = v;
        } else {
            sc_c[j - hdim - ns] = v;
        }
    }
    workgroupBarrier();
    let p = lid % 64u;
    let kq = lid / 64u;
    let raw = sc_zx[sc_p.dt_off + h] + sc_par[2u * sc_p.nh + h];
    var dt = log(1.0 + exp(raw));
    if (raw > 20.0) { dt = raw; }
    if (raw < -15.0) { dt = exp(raw); }
    let da = exp(dt * sc_par[h]);
    let x = sc_x[p];
    let dx = dt * x;
    var acc = 0.0;
    let q = ns / 4u;
    let base = h * ns * hdim + p;
    for (var k = kq * q; k < (kq + 1u) * q; k = k + 1u) {
        let i = base + k * hdim;
        let v = da * sc_s[i] + dx * sc_b[k];
        sc_s[i] = v;
        acc = acc + v * sc_c[k];
    }
    sc_part[lid] = acc;
    workgroupBarrier();
    if (kq == 0u) {
        sc_y[h * hdim + p] = sc_part[p] + sc_part[64u + p] + sc_part[128u + p] + sc_part[192u + p]
            + sc_par[sc_p.nh + h] * x;
    }
}

// ── routed experts from a per-layer bank: workgroup (row block, slot).
// `route` 1 (the up projection): every workgroup takes the top-k of the
// router logits itself (rank by score + bias, ties to the lower index) and
// workgroup (0,0) stores the picks and their weights; `route` 0 (down)
// reads the stored picks. With `remap` the bank holds only some experts:
// map[layer·n + e] is the bank slot or MISS. A missing expert the host has
// provided for this layer (meta 4/5) is skipped; otherwise the up
// projection raises `abort` and the host completes the layer.
// q4tp payload: [nibbles rows·gpr·16][params (f16 lo, f16 step) rows·4]
// [5-bit codes]; q8_2f: [int8 rows·cols][f16 row scales][f16 col field].
// A 32-lane subgroup owns 4 rows, all four in flight; act 1 = relu².
struct BkP {
    rows: u32, gpr: u32, stride_w: u32, act: u32,
    x_stride: u32, y_stride: u32, route: u32, n_exp: u32,
    k: u32, scale: f32, norm: u32, layer: u32,
    remap: u32, cols: u32, nrep: u32, _b: u32,
};
@group(0) @binding(0)  var<storage, read>       bk_w    : array<u32>;
@group(0) @binding(1)  var<storage, read>       bk_w4   : array<vec4<u32>>;
@group(0) @binding(2)  var<storage, read>       bk_x    : array<vec4<f32>>;
@group(0) @binding(3)  var<storage, read>       bk_lg   : array<f32>;
@group(0) @binding(4)  var<storage, read>       bk_bias : array<f32>;
@group(0) @binding(5)  var<storage, read_write> bk_sel  : array<u32>;
@group(0) @binding(6)  var<storage, read_write> bk_sw   : array<f32>;
@group(0) @binding(7)  var<storage, read_write> bk_y    : array<f32>;
@group(0) @binding(8)  var<storage, read_write> bk_meta : array<u32>;
@group(0) @binding(9)  var<storage, read>       bk_map  : array<u32>;
@group(0) @binding(10) var<storage, read_write> bk_use  : array<u32>;
@group(0) @binding(11) var<uniform>             bk_p    : BkP;
var<workgroup> bk_xs: array<vec4<f32>, 1024>;
var<workgroup> bk_c: array<f32, 256>;
var<workgroup> bk_s: array<u32, 16>;
var<workgroup> bk_flag: u32;
var<workgroup> bk_slot: u32;

fn bk_prologue(wid: vec3<u32>, lid: u32) -> u32 {
    if (lid == 0u) { bk_flag = bk_meta[1]; }
    if (workgroupUniformLoad(&bk_flag) != 0u) { return 0xFFFFFFFFu; }
    let n = bk_p.n_exp;
    if (lid < 16u) { bk_s[lid] = 0u; }
    if (bk_p.route == 1u) {
        if (lid < n) { bk_c[lid] = 1.0 / (1.0 + exp(-bk_lg[lid])) + bk_bias[lid]; }
        workgroupBarrier();
        if (lid < n) {
            let ci = bk_c[lid];
            var rank = 0u;
            for (var j = 0u; j < n; j = j + 1u) {
                let cj = bk_c[j];
                if (cj > ci || (cj == ci && j < lid)) { rank = rank + 1u; }
            }
            if (rank < bk_p.k) { bk_s[rank] = lid; }
        }
        workgroupBarrier();
        if (wid.x == 0u && wid.y == 0u && lid == 0u) {
            var sum = 0.0;
            for (var t = 0u; t < bk_p.k; t = t + 1u) {
                sum = sum + 1.0 / (1.0 + exp(-bk_lg[bk_s[t]]));
            }
            var den = 1.0;
            if (bk_p.norm != 0u) { den = sum + 1e-20; }
            for (var t = 0u; t < bk_p.k; t = t + 1u) {
                let e = bk_s[t];
                bk_sel[t] = e;
                bk_sw[t] = (1.0 / (1.0 + exp(-bk_lg[e]))) / den * bk_p.scale;
                bk_use[bk_p.layer * n + e] = bk_use[bk_p.layer * n + e] + 1u;
            }
        }
    } else {
        workgroupBarrier();
        if (lid < bk_p.k) { bk_s[lid] = bk_sel[lid]; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let slot = wid.y;
        let e = bk_s[slot];
        var ds = e;
        if (bk_p.remap != 0u) { ds = bk_map[bk_p.layer * n + e]; }
        if (ds == 0xFFFFFFFFu) {
            let prov = bk_meta[4] == bk_p.layer && ((bk_meta[5] >> slot) & 1u) == 1u;
            if (!prov && bk_p.route == 1u) {
                bk_meta[2] = bk_p.layer;
                bk_meta[3] = bk_meta[0];
                bk_meta[1] = 1u;
            }
        }
        bk_slot = ds;
    }
    return workgroupUniformLoad(&bk_slot);
}

fn bk_byte(off: u32) -> u32 { return (bk_w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn bk_code(cb: u32, g: u32) -> u32 {
    let bit = g * 5u;
    let off = cb + (bit >> 3u);
    return ((bk_byte(off) | (bk_byte(off + 1u) << 8u)) >> (bit & 7u)) & 31u;
}
fn bk_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn bk_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return bk_nib(w, 0u) * a.x + bk_nib(w, 4u) * a.y + bk_nib(w, 8u) * a.z + bk_nib(w, 12u) * a.w
         + bk_nib(w, 16u) * b.x + bk_nib(w, 20u) * b.y + bk_nib(w, 24u) * b.z + bk_nib(w, 28u) * b.w;
}
// x staged transposed, [i][group] (vec4 i of group g at i·gpr + g): the
// lanes of a subgroup read consecutive groups, no bank conflicts
fn bk_dot32(n: vec4<u32>, g: u32, gpr: u32) -> f32 {
    return bk_dot8(n.x, bk_xs[g], bk_xs[gpr + g]) + bk_dot8(n.y, bk_xs[2u * gpr + g], bk_xs[3u * gpr + g])
         + bk_dot8(n.z, bk_xs[4u * gpr + g], bk_xs[5u * gpr + g]) + bk_dot8(n.w, bk_xs[6u * gpr + g], bk_xs[7u * gpr + g]);
}
fn bk_out(slot: u32, row: u32, v: f32) {
    var o = v;
    if (bk_p.act == 1u) { o = nf_relu2(o); }
    bk_y[slot * bk_p.y_stride + row] = o;
}

@compute @workgroup_size(256)
fn nf_bank4(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32,
            @builtin(subgroup_invocation_id) lane: u32) {
    let ds = bk_prologue(wid, lid);
    if (ds == 0xFFFFFFFFu) { return; }
    let slot = wid.y;
    let rows = bk_p.rows;
    let gpr = bk_p.gpr;
    let c4 = gpr * 8u;
    let xo = slot * bk_p.x_stride / 4u;
    for (var i = lid; i < c4; i = i + 256u) {
        bk_xs[(i % 8u) * gpr + i / 8u] = bk_x[xo + i];
    }
    workgroupBarrier();
    let bw = ds * bk_p.stride_w;
    let params_w = bw + rows * gpr * 4u;
    let codes_b = (params_w + rows) * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    for (var rep = 0u; rep < bk_p.nrep; rep = rep + 1u) {
    let row0 = ((wid.x * bk_p.nrep + rep) * 8u + lid / 32u) * 4u;
    let r0 = min(row0, rows - 1u);
    let r1 = min(row0 + 1u, rows - 1u);
    let r2 = min(row0 + 2u, rows - 1u);
    let r3 = min(row0 + 3u, rows - 1u);
    let p0 = unpack2x16float(bk_w[params_w + r0]);
    let p1 = unpack2x16float(bk_w[params_w + r1]);
    let p2 = unpack2x16float(bk_w[params_w + r2]);
    let p3 = unpack2x16float(bk_w[params_w + r3]);
    let nb = bw / 4u;
    let n0 = nb + r0 * gpr;
    let n1 = nb + r1 * gpr;
    let n2 = nb + r2 * gpr;
    let n3 = nb + r3 * gpr;
    let c0 = codes_b + r0 * cstride;
    let c1 = codes_b + r1 * cstride;
    let c2 = codes_b + r2 * cstride;
    let c3 = codes_b + r3 * cstride;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    for (var g = lane; g < gpr; g = g + 32u) {
        let w0 = bk_w4[n0 + g];
        let w1 = bk_w4[n1 + g];
        let w2 = bk_w4[n2 + g];
        let w3 = bk_w4[n3 + g];
        let k0 = bk_code(c0, g);
        let k1 = bk_code(c1, g);
        let k2 = bk_code(c2, g);
        let k3 = bk_code(c3, g);
        a0 = a0 + exp2(p0.x + f32(k0) * p0.y) * bk_dot32(w0, g, gpr);
        a1 = a1 + exp2(p1.x + f32(k1) * p1.y) * bk_dot32(w1, g, gpr);
        a2 = a2 + exp2(p2.x + f32(k2) * p2.y) * bk_dot32(w2, g, gpr);
        a3 = a3 + exp2(p3.x + f32(k3) * p3.y) * bk_dot32(w3, g, gpr);
    }
    let t0 = subgroupAdd(a0);
    let t1 = subgroupAdd(a1);
    let t2 = subgroupAdd(a2);
    let t3 = subgroupAdd(a3);
    if (lane == 0u) {
        if (row0 < rows) { bk_out(slot, row0, t0); }
        if (row0 + 1u < rows) { bk_out(slot, row0 + 1u, t1); }
        if (row0 + 2u < rows) { bk_out(slot, row0 + 2u, t2); }
        if (row0 + 3u < rows) { bk_out(slot, row0 + 3u, t3); }
    }
    }
}

fn bk_cs4(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(bk_w[w]);
    let b = unpack2x16float(bk_w[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(bk_w[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
fn bk_rs(rs0: u32, r: u32) -> f32 {
    let rw = unpack2x16float(bk_w[rs0 + (r >> 1u)]);
    return select(rw.x, rw.y, (r & 1u) == 1u);
}

@compute @workgroup_size(256)
fn nf_bank8(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32,
            @builtin(subgroup_invocation_id) lane: u32) {
    let ds = bk_prologue(wid, lid);
    if (ds == 0xFFFFFFFFu) { return; }
    let slot = wid.y;
    let rows = bk_p.rows;
    let words = bk_p.cols / 4u;
    let bw = ds * bk_p.stride_w;
    let cs0h = bw * 2u + ((rows * bk_p.cols) >> 1u) + rows;
    let xo = slot * bk_p.x_stride / 4u;
    for (var i = lid; i < words; i = i + 256u) {
        bk_xs[i] = bk_x[xo + i] * bk_cs4(cs0h + i * 4u);
    }
    workgroupBarrier();
    for (var rep = 0u; rep < bk_p.nrep; rep = rep + 1u) {
    let row0 = ((wid.x * bk_p.nrep + rep) * 8u + lid / 32u) * 4u;
    let b0 = bw + min(row0, rows - 1u) * words;
    let b1 = bw + min(row0 + 1u, rows - 1u) * words;
    let b2 = bw + min(row0 + 2u, rows - 1u) * words;
    let b3 = bw + min(row0 + 3u, rows - 1u) * words;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    var k = lane;
    loop {
        if (k + 32u >= words) { break; }
        let xa = bk_xs[k];
        let xb = bk_xs[k + 32u];
        let u0 = bk_w[b0 + k];
        let u1 = bk_w[b1 + k];
        let u2 = bk_w[b2 + k];
        let u3 = bk_w[b3 + k];
        let v0 = bk_w[b0 + k + 32u];
        let v1 = bk_w[b1 + k + 32u];
        let v2 = bk_w[b2 + k + 32u];
        let v3 = bk_w[b3 + k + 32u];
        a0 = a0 + dot(nf_i8x4(u0), xa) + dot(nf_i8x4(v0), xb);
        a1 = a1 + dot(nf_i8x4(u1), xa) + dot(nf_i8x4(v1), xb);
        a2 = a2 + dot(nf_i8x4(u2), xa) + dot(nf_i8x4(v2), xb);
        a3 = a3 + dot(nf_i8x4(u3), xa) + dot(nf_i8x4(v3), xb);
        k = k + 64u;
    }
    if (k < words) {
        let x = bk_xs[k];
        a0 = a0 + dot(nf_i8x4(bk_w[b0 + k]), x);
        a1 = a1 + dot(nf_i8x4(bk_w[b1 + k]), x);
        a2 = a2 + dot(nf_i8x4(bk_w[b2 + k]), x);
        a3 = a3 + dot(nf_i8x4(bk_w[b3 + k]), x);
    }
    // keep the vec4 view in the binding set (auto layout): never true
    if (rows == 0xFFFFFFFFu) { a0 = a0 + f32(bk_w4[0].x); }
    let t0 = subgroupAdd(a0);
    let t1 = subgroupAdd(a1);
    let t2 = subgroupAdd(a2);
    let t3 = subgroupAdd(a3);
    let rs0 = bw + ((rows * bk_p.cols) >> 2u);
    if (lane == 0u) {
        if (row0 < rows) { bk_out(slot, row0, t0 * bk_rs(rs0, row0)); }
        if (row0 + 1u < rows) { bk_out(slot, row0 + 1u, t1 * bk_rs(rs0, row0 + 1u)); }
        if (row0 + 2u < rows) { bk_out(slot, row0 + 2u, t2 * bk_rs(rs0, row0 + 2u)); }
        if (row0 + 3u < rows) { bk_out(slot, row0 + 3u, t3 * bk_rs(rs0, row0 + 3u)); }
    }
    }
}

// ── GQA decode attention, split-K partials. Workgroup (split, kv head,
// block of 4 q heads); eight 32-lane subgroups walk the split's positions;
// a lane owns 4 dims of each of the 4 heads (hd = 128), and one vec4
// subgroup sum gives the 4 scores of a position. Partial per (split,
// subgroup, head): acc[hd], max, sum.
struct AtP { nh: u32, nkv: u32, hd: u32, npos: u32, chunk: u32, scale: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       at_q    : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       at_kc   : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       at_vc   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> at_part : array<f32>;
@group(0) @binding(4) var<uniform>             at_p    : AtP;
fn at_put(base: u32, lane: u32, a: vec4<f32>, m: f32, l: f32) {
    at_part[base + lane * 4u] = a.x;
    at_part[base + lane * 4u + 1u] = a.y;
    at_part[base + lane * 4u + 2u] = a.z;
    at_part[base + lane * 4u + 3u] = a.w;
    if (lane == 0u) {
        at_part[base + at_p.hd] = m;
        at_part[base + at_p.hd + 1u] = l;
    }
}
@compute @workgroup_size(256)
fn nm_attn(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32,
           @builtin(subgroup_invocation_id) lane: u32) {
    let s = wid.x;
    let g = wid.y;
    let w = lid / 32u;
    let rep = at_p.nh / at_p.nkv;
    let d4 = at_p.hd / 4u;
    let h0 = g * rep + wid.z * 4u;
    let q0 = at_q[h0 * d4 + lane] * at_p.scale;
    let q1 = at_q[(h0 + 1u) * d4 + lane] * at_p.scale;
    let q2 = at_q[(h0 + 2u) * d4 + lane] * at_p.scale;
    let q3 = at_q[(h0 + 3u) * d4 + lane] * at_p.scale;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var m = vec4<f32>(-1e30);
    var l = vec4<f32>(0.0);
    let start = s * at_p.chunk;
    let end = min(start + at_p.chunk, at_p.npos);
    for (var j = start + w; j < end; j = j + 8u) {
        let kk = at_kc[(j * at_p.nkv + g) * d4 + lane];
        let vv = at_vc[(j * at_p.nkv + g) * d4 + lane];
        let sc = subgroupAdd(vec4<f32>(dot(q0, kk), dot(q1, kk), dot(q2, kk), dot(q3, kk)));
        let mn = max(m, sc);
        let al = exp(m - mn);
        let pp = exp(sc - mn);
        l = l * al + pp;
        a0 = a0 * al.x + pp.x * vv;
        a1 = a1 * al.y + pp.y * vv;
        a2 = a2 * al.z + pp.z * vv;
        a3 = a3 * al.w + pp.w * vv;
        m = mn;
    }
    let stride = at_p.hd + 2u;
    let base = ((s * 8u + w) * at_p.nh + h0) * stride;
    at_put(base, lane, a0, m.x, l.x);
    at_put(base + stride, lane, a1, m.y, l.y);
    at_put(base + 2u * stride, lane, a2, m.z, l.z);
    at_put(base + 3u * stride, lane, a3, m.w, l.w);
}

struct AmP { nh: u32, hd: u32, nparts: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       am_part : array<f32>;
@group(0) @binding(1) var<storage, read_write> am_o    : array<f32>;
@group(0) @binding(2) var<uniform>             am_p    : AmP;
@compute @workgroup_size(128)
fn nm_attn_merge(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) d: u32) {
    let h = wid.x;
    let stride = am_p.hd + 2u;
    var mx = -1e30;
    for (var p = 0u; p < am_p.nparts; p = p + 1u) {
        mx = max(mx, am_part[(p * am_p.nh + h) * stride + am_p.hd]);
    }
    var den = 0.0;
    var num = 0.0;
    for (var p = 0u; p < am_p.nparts; p = p + 1u) {
        let base = (p * am_p.nh + h) * stride;
        let f = exp(am_part[base + am_p.hd] - mx);
        den = den + am_part[base + am_p.hd + 1u] * f;
        num = num + am_part[base + d] * f;
    }
    if (d < am_p.hd) { am_o[h * am_p.hd + d] = num / den; }
}

// ── final RMSNorm (one workgroup) ──
struct RmP { n: u32, eps: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       rm_h    : array<f32>;
@group(0) @binding(1) var<storage, read>       rm_w    : array<f32>;
@group(0) @binding(2) var<storage, read_write> rm_y    : array<f32>;
@group(0) @binding(3) var<storage, read>       rm_meta : array<u32>;
@group(0) @binding(4) var<uniform>             rm_p    : RmP;
var<workgroup> rm_part: array<f32, 256>;
var<workgroup> rm_flag: u32;
@compute @workgroup_size(256)
fn nf_rms(@builtin(local_invocation_index) lid: u32) {
    if (lid == 0u) { rm_flag = rm_meta[1]; }
    if (workgroupUniformLoad(&rm_flag) != 0u) { return; }
    let n = rm_p.n;
    var acc = 0.0;
    for (var i = lid; i < n; i = i + 256u) {
        let v = rm_h[i];
        acc = acc + v * v;
    }
    rm_part[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { rm_part[lid] = rm_part[lid] + rm_part[lid + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(rm_part[0] / f32(n) + rm_p.eps);
    for (var i = lid; i < n; i = i + 256u) {
        rm_y[i] = rm_w[i] * (rm_h[i] * inv);
    }
}

// ── argmax stage 1: best of each 1024-wide block ──
struct AxP { n: u32, blocks: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       ax_x  : array<f32>;
@group(0) @binding(1) var<storage, read_write> ax_bv : array<f32>;
@group(0) @binding(2) var<storage, read_write> ax_bi : array<u32>;
@group(0) @binding(3) var<uniform>             ax_p  : AxP;
var<workgroup> ax_v: array<f32, 256>;
var<workgroup> ax_i: array<u32, 256>;
@compute @workgroup_size(256)
fn nf_argmax1(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    var bv = -3.0e38;
    var bi = 0u;
    let base = wid.x * 1024u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let i = base + j * 256u + lid;
        if (i < ax_p.n) {
            let v = ax_x[i];
            if (v > bv) { bv = v; bi = i; }
        }
    }
    ax_v[lid] = bv;
    ax_i[lid] = bi;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let a = ax_v[lid];
            let b = ax_v[lid + s];
            if (b > a || (b == a && ax_i[lid + s] < ax_i[lid])) {
                ax_v[lid] = b;
                ax_i[lid] = ax_i[lid + s];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        ax_bv[wid.x] = ax_v[0];
        ax_bi[wid.x] = ax_i[0];
    }
}

// ── argmax stage 2 + the q8_2f embedding of the winner into h; the step
// ends here: ids[step] = winner, pos and step advance ──
struct PkP { blocks: u32, rows: u32, cols: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       pk_bv   : array<f32>;
@group(0) @binding(1) var<storage, read>       pk_bi   : array<u32>;
@group(0) @binding(2) var<storage, read_write> pk_ids  : array<u32>;
@group(0) @binding(3) var<storage, read>       pk_emb  : array<u32>;
@group(0) @binding(4) var<storage, read_write> pk_h    : array<f32>;
@group(0) @binding(5) var<storage, read_write> pk_meta : array<u32>;
@group(0) @binding(6) var<uniform>             pk_p    : PkP;
var<workgroup> pk_v: array<f32, 256>;
var<workgroup> pk_i: array<u32, 256>;
var<workgroup> pk_flag: u32;
fn pk_f16(byte_off: u32) -> f32 {
    let w = unpack2x16float(pk_emb[byte_off >> 2u]);
    return select(w.x, w.y, (byte_off & 2u) != 0u);
}
@compute @workgroup_size(256)
fn nf_pick(@builtin(local_invocation_index) lid: u32) {
    if (lid == 0u) { pk_flag = pk_meta[1]; }
    if (workgroupUniformLoad(&pk_flag) != 0u) { return; }
    var bv = -3.0e38;
    var bi = 0u;
    for (var i = lid; i < pk_p.blocks; i = i + 256u) {
        let v = pk_bv[i];
        if (v > bv || (v == bv && pk_bi[i] < bi)) { bv = v; bi = pk_bi[i]; }
    }
    pk_v[lid] = bv;
    pk_i[lid] = bi;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let a = pk_v[lid];
            let b = pk_v[lid + s];
            if (b > a || (b == a && pk_i[lid + s] < pk_i[lid])) {
                pk_v[lid] = b;
                pk_i[lid] = pk_i[lid + s];
            }
        }
        workgroupBarrier();
    }
    let id = pk_i[0];
    let cols = pk_p.cols;
    let body = pk_p.rows * cols;
    let rs = pk_f16(body + id * 2u);
    for (var c = lid; c < cols; c = c + 256u) {
        let off = id * cols + c;
        let q = i32(((pk_emb[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu) << 24u) >> 24u;
        let cf = pk_f16(body + pk_p.rows * 2u + c * 2u);
        pk_h[c] = f32(q) * rs * cf;
    }
    if (lid == 0u) {
        let st = pk_meta[6];
        pk_ids[st] = id;
        pk_meta[6] = st + 1u;
        pk_meta[0] = pk_meta[0] + 1u;
    }
}
"#;

/// The q8_2f row loop of `nf_q8`, one copy per matrix binding: two rows a
/// warp at once, six words a lane per step (twelve loads in flight).
const NQ_FN_T: &str = r#"
fn nq_cs4_{M}(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(nq_w{M}[w]);
    let b = unpack2x16float(nq_w{M}[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(nq_w{M}[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
fn nq_rows_{M}(m: u32, blk: u32, lid: u32, lane: u32) {
    let rows = nq_p.rows_{M};
    let words = nq_p.cols / 4u;
    let rc = select(nq_p.cols, nq_p.row_cols, nq_p.row_cols != 0u);
    let rs0 = (rows * rc) >> 2u;
    let warp = lid / 32u;
    for (var r = 0u; r < nq_p.rpw; r = r + 2u) {
        let row0 = (blk * nq_p.rpw + r) * 8u + warp * 2u;
        let live0 = row0 < rows;
        let live1 = row0 + 1u < rows;
        let r0 = select(rows - 1u, row0, live0);
        let r1 = select(rows - 1u, row0 + 1u, live1);
        let b0 = r0 * (rc / 4u) + nq_p.col_off / 4u;
        let b1 = r1 * (rc / 4u) + nq_p.col_off / 4u;
        var a0 = 0.0;
        var a1 = 0.0;
        var k = lane;
        loop {
            if (k + 160u >= words) { break; }
            let p0 = nq_w{M}[b0 + k];
            let p1 = nq_w{M}[b0 + k + 32u];
            let p2 = nq_w{M}[b0 + k + 64u];
            let p3 = nq_w{M}[b0 + k + 96u];
            let p4 = nq_w{M}[b0 + k + 128u];
            let p5 = nq_w{M}[b0 + k + 160u];
            let q0 = nq_w{M}[b1 + k];
            let q1 = nq_w{M}[b1 + k + 32u];
            let q2 = nq_w{M}[b1 + k + 64u];
            let q3 = nq_w{M}[b1 + k + 96u];
            let q4 = nq_w{M}[b1 + k + 128u];
            let q5 = nq_w{M}[b1 + k + 160u];
            let x0 = nq_xs[k];
            let x1 = nq_xs[k + 32u];
            let x2 = nq_xs[k + 64u];
            let x3 = nq_xs[k + 96u];
            let x4 = nq_xs[k + 128u];
            let x5 = nq_xs[k + 160u];
            a0 = a0 + dot(nf_i8x4(p0), x0) + dot(nf_i8x4(p1), x1) + dot(nf_i8x4(p2), x2)
                 + dot(nf_i8x4(p3), x3) + dot(nf_i8x4(p4), x4) + dot(nf_i8x4(p5), x5);
            a1 = a1 + dot(nf_i8x4(q0), x0) + dot(nf_i8x4(q1), x1) + dot(nf_i8x4(q2), x2)
                 + dot(nf_i8x4(q3), x3) + dot(nf_i8x4(q4), x4) + dot(nf_i8x4(q5), x5);
            k = k + 192u;
        }
        loop {
            if (k >= words) { break; }
            let x = nq_xs[k];
            a0 = a0 + dot(nf_i8x4(nq_w{M}[b0 + k]), x);
            a1 = a1 + dot(nf_i8x4(nq_w{M}[b1 + k]), x);
            k = k + 32u;
        }
        let t0 = subgroupAdd(a0);
        let t1 = subgroupAdd(a1);
        if (lane == 0u && live0) {
            let rw = unpack2x16float(nq_w{M}[rs0 + (r0 >> 1u)]);
            nq_out(m, r0, t0 * select(rw.x, rw.y, (r0 & 1u) == 1u));
        }
        if (lane == 0u && live1) {
            let rw = unpack2x16float(nq_w{M}[rs0 + (r1 >> 1u)]);
            nq_out(m, r1, t1 * select(rw.x, rw.y, (r1 & 1u) == 1u));
        }
    }
}
"#;

/// The decode module: `NEMO_WGSL` with the per-matrix row loops expanded.
pub(crate) fn nemo_wgsl() -> String {
    let mut fns = String::new();
    for m in ["a", "b", "c"] {
        fns.push_str(&NQ_FN_T.replace("{M}", m));
    }
    NEMO_WGSL.replace("{NQ_FNS}", &fns)
}

struct Pipes {
    q8: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    bank4: wgpu::ComputePipeline,
    bank8: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
    attn_merge: wgpu::ComputePipeline,
    rms: wgpu::ComputePipeline,
    argmax1: wgpu::ComputePipeline,
    pick: wgpu::ComputePipeline,
}

static PIPES: std::sync::OnceLock<Option<Pipes>> = std::sync::OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES.get_or_init(|| build_pipes(c)).as_ref()
}

fn build_pipes(c: &Ctx) -> Option<Pipes> {
    // The kernels split a workgroup into 32-lane subgroups. Apple GPUs run
    // compute in 32-wide SIMD groups, while wgpu's Metal backend reports a
    // range around it, so Metal admits the range; elsewhere it must be 32.
    let (lo, hi) = (
        c.adapter_info.subgroup_min_size,
        c.adapter_info.subgroup_max_size,
    );
    let sg32 = (lo == 32 && hi == 32)
        || (c.adapter_info.backend == wgpu::Backend::Metal && lo <= 32 && hi >= 32);
    if !c.device.features().contains(wgpu::Features::SUBGROUP) || !sg32 {
        tracing::warn!("nemotron frame: needs 32-lane subgroups (adapter reports {lo}..{hi})");
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-nemotron"),
        source: wgpu::ShaderSource::Wgsl(nemo_wgsl().into()),
    });
    let pipe = |ep: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: pco(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let p = Pipes {
        q8: pipe("nf_q8"),
        scan: pipe("nf_scan"),
        bank4: pipe("nf_bank4"),
        bank8: pipe("nf_bank8"),
        attn: pipe("nm_attn"),
        attn_merge: pipe("nm_attn_merge"),
        rms: pipe("nf_rms"),
        argmax1: pipe("nf_argmax1"),
        pick: pipe("nf_pick"),
    };
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("nemotron frame: module rejected ({e})");
        return None;
    }
    Some(p)
}

/// Layout of the routed experts' payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertCodec {
    Q4tp,
    Q8,
}

/// Host description of the model for [`NemoFrame::build`].
pub struct NemoDesc<'a> {
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub m_heads: usize,
    pub m_head_dim: usize,
    pub m_groups: usize,
    pub m_state: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub n_exp: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub route_scale: f32,
    pub norm_topk: bool,
    pub blocks: Vec<NemoBlockDesc<'a>>,
    /// q8_2f payloads
    pub embed: &'a [u8],
    pub lm_head: &'a [u8],
    pub final_norm: &'a [f32],
    /// KV capacity (positions)
    pub cap: usize,
    /// The mapped file the expert payloads live in: a partial bank admits
    /// experts from it later.
    pub model: Option<std::sync::Arc<CmfModel>>,
    /// Device bytes other residents need (the audio towers).
    pub reserve: u64,
    /// The draft (multi-token prediction) layer, when speculation is on.
    pub mtp: Option<NemoMtpDesc<'a>>,
}

/// The draft layer: the fusion of a hidden row with the next token's
/// embedding (`eh` q8_2f `[hidden, 2·hidden]`), an attention block and a
/// MoE block, the output norm; the main head drafts from it.
pub struct NemoMtpDesc<'a> {
    pub enorm: &'a [f32],
    pub hnorm: &'a [f32],
    pub eh: &'a [u8],
    /// the hidden half comes first in `eh`'s columns
    pub h_first: bool,
    pub n_attn: &'a [f32],
    pub q: &'a [u8],
    pub k: &'a [u8],
    pub v: &'a [u8],
    pub o: &'a [u8],
    pub n_moe: &'a [f32],
    pub router: &'a [f32],
    pub bias: &'a [f32],
    pub up: Vec<&'a [u8]>,
    pub down: Vec<&'a [u8]>,
    pub sh_up: &'a [u8],
    pub sh_down: &'a [u8],
    pub fnorm: &'a [f32],
}

/// The draft layer on the device.
struct DevMtp {
    enorm: wgpu::Buffer,
    hnorm: wgpu::Buffer,
    eh: wgpu::Buffer,
    h_first: bool,
    n_attn: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    o: wgpu::Buffer,
    kc: wgpu::Buffer,
    vc: wgpu::Buffer,
    n_moe: wgpu::Buffer,
    router: wgpu::Buffer,
    bias: wgpu::Buffer,
    up: wgpu::Buffer,
    up_stride: usize,
    down: wgpu::Buffer,
    down_stride: usize,
    sh_up: wgpu::Buffer,
    sh_down: wgpu::Buffer,
    fnorm: wgpu::Buffer,
    /// input rows: the hidden state, the next token's embedding; the
    /// fused stream; the output (normed) row
    hin: wgpu::Buffer,
    e: wgpu::Buffer,
    x: wgpu::Buffer,
    out: wgpu::Buffer,
}

pub enum NemoBlockDesc<'a> {
    Mamba {
        norm: &'a [f32],
        /// q8_2f
        in_proj: &'a [u8],
        out_proj: &'a [u8],
        conv_w: &'a [f32],
        conv_b: &'a [f32],
        /// `[A = −exp(A_log) | D | dt_bias]`
        par: Vec<f32>,
        gnorm: &'a [f32],
    },
    Attn {
        norm: &'a [f32],
        q: &'a [u8],
        k: &'a [u8],
        v: &'a [u8],
        o: &'a [u8],
    },
    Moe {
        norm: &'a [f32],
        router: &'a [f32],
        bias: &'a [f32],
        codec: ExpertCodec,
        /// one payload per expert
        up: Vec<&'a [u8]>,
        down: Vec<&'a [u8]>,
        /// the order to fill a partial bank in (most used first), if known
        hot: Option<Vec<usize>>,
        /// q8_2f
        sh_up: &'a [u8],
        sh_down: &'a [u8],
    },
}

enum DevBlock {
    Mamba {
        norm: wgpu::Buffer,
        in_proj: wgpu::Buffer,
        out_proj: wgpu::Buffer,
        conv_w: wgpu::Buffer,
        conv_b: wgpu::Buffer,
        par: wgpu::Buffer,
        gnorm: wgpu::Buffer,
        ring: wgpu::Buffer,
        state: wgpu::Buffer,
    },
    Attn {
        norm: wgpu::Buffer,
        q: wgpu::Buffer,
        k: wgpu::Buffer,
        v: wgpu::Buffer,
        o: wgpu::Buffer,
        kc: wgpu::Buffer,
        vc: wgpu::Buffer,
    },
    Moe {
        norm: wgpu::Buffer,
        router: wgpu::Buffer,
        bias: wgpu::Buffer,
        /// banks of `slots` experts, strides in u32 words
        up: wgpu::Buffer,
        up_stride: usize,
        down: wgpu::Buffer,
        down_stride: usize,
        sh_up: wgpu::Buffer,
        sh_down: wgpu::Buffer,
        /// index among the MoE blocks
        moe: usize,
    },
}

/// Scratch activations of one step.
struct Scratch {
    h: wgpu::Buffer,
    xn: wgpu::Buffer,
    zx: wgpu::Buffer,
    y: wgpu::Buffer,
    q: wgpu::Buffer,
    att: wgpu::Buffer,
    part: wgpu::Buffer,
    logits_r: wgpu::Buffer,
    sel: wgpu::Buffer,
    sw: wgpu::Buffer,
    mid: wgpu::Buffer,
    eo: wgpu::Buffer,
    smid: wgpu::Buffer,
    logits: wgpu::Buffer,
    bv: wgpu::Buffer,
    bi: wgpu::Buffer,
    ids: wgpu::Buffer,
    meta: wgpu::Buffer,
    /// expert → bank slot per MoE layer (partial banks)
    map: wgpu::Buffer,
    /// routing counts per MoE layer and expert
    usage: wgpu::Buffer,
    /// bound where a kernel's binding is unused
    ro: wgpu::Buffer,
    rw: wgpu::Buffer,
    /// readback of ids + meta (+ usage, picks and the MoE input on a miss)
    stage: [wgpu::Buffer; 2],
    stage_logits: wgpu::Buffer,
}

/// One dispatch of the step plan. `Fixed` ones are bound once at build;
/// the position-dependent attention core is bound per step.
enum Op {
    Fixed {
        pipe: &'static wgpu::ComputePipeline,
        bg: wgpu::BindGroup,
        groups: (u32, u32),
        tag: &'static str,
    },
    /// split-K attention + merge for attention block `block`
    Attn { block: usize },
    /// the same over the draft layer's KV cache
    MtpAttn,
}

const MISS: u32 = u32::MAX;
/// Positions of one verify pass (the token + its drafts).
pub const VR_MAX: usize = 4;
const STAGE_IDS: usize = 64;
const META_WORDS: usize = 8;

/// Host side of partial expert banks.
struct Arena {
    slots: usize,
    /// `[moe · n_exp]` → slot or MISS (mirror of `Scratch::map`)
    map: Vec<u32>,
    /// `[moe][slot]` → expert or MISS
    owner: Vec<Vec<u32>>,
    /// decayed routing counts `[moe · n_exp]`
    score: Vec<f32>,
    /// the device usage counters at the last read
    seen: Vec<u32>,
    /// every expert's payloads in the mapped file: `[moe][e]` = (up, down)
    src: Vec<Vec<[(usize, usize); 2]>>,
    _model: std::sync::Arc<CmfModel>,
    misses: u64,
    admitted: u64,
    /// `pos / CMF_NEMO_ADMIT` at the last admission
    admit_mark: usize,
}

/// A step stopped at a MoE layer whose routed experts are not all resident.
pub struct NemoMiss {
    /// index among the MoE blocks
    pub moe: usize,
    /// the layer's input row (normed) and its top-k (expert, weight)
    pub xn: Vec<f32>,
    pub picks: Vec<(usize, f32)>,
    /// positions in `picks` the host must complete
    pub missing: Vec<usize>,
}

/// What a greedy run returned: the ids of the finished steps, and the miss
/// that stopped the run early (resume with [`NemoFrame::resume`]).
pub struct NemoRun {
    pub ids: Vec<u32>,
    pub miss: Option<NemoMiss>,
}

/// A submitted greedy batch awaiting readback.
struct Flight {
    stage: usize,
    index: wgpu::SubmissionIndex,
    steps: usize,
}

/// The rest of a run that stopped on a miss.
struct Pending {
    /// steps of the run not yet finished, the stopped one included
    left: usize,
    pos: usize,
    moe: usize,
    ids: Vec<u32>,
}

pub struct NemoFrame {
    c: &'static Ctx,
    p: &'static Pipes,
    blocks: Vec<DevBlock>,
    plan: Vec<Op>,
    /// plan index of each MoE layer's up-projection kernel (resume point)
    moe_at: Vec<usize>,
    /// plan index of the final norm (end of the blocks) and past the head
    body_end: usize,
    head_end: usize,
    embed: wgpu::Buffer,
    lm_head: wgpu::Buffer,
    final_norm: wgpu::Buffer,
    s: Scratch,
    hidden: usize,
    vocab: usize,
    eps: f32,
    m_heads: usize,
    m_head_dim: usize,
    m_groups: usize,
    m_state: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    top_k: usize,
    moe_inter: usize,
    shared_inter: usize,
    route_scale: f32,
    norm_topk: bool,
    n_exp: usize,
    n_moe: usize,
    codec: ExpertCodec,
    cap: usize,
    max_parts: usize,
    arena: Option<Arena>,
    flights: std::collections::VecDeque<Flight>,
    next_stage: usize,
    pending: Option<Pending>,
    /// Batched-prefill buffers and plan, built on first use.
    pf: Option<Prefill>,
    /// positions of the last verify pass awaiting [`Self::accept`]
    pending_verify: usize,
    /// The two-position verify pass, built on first use.
    vf: Option<Verify>,
    /// the draft layer and its step plan (`mtp_head_at`: where the head,
    /// argmax and embedding of the draft start)
    mtp: Option<DevMtp>,
    mtp_plan: Vec<Op>,
    mtp_head_at: usize,
    /// `CMF_NEMO_TS=1`: GPU timestamps around every dispatch of the first
    /// step of a batch (query set, resolve buffer, readback buffer).
    ts: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)>,
    /// Positions already in the device state (with every submitted step).
    pub pos: usize,
    pub bytes: u64,
}

/// `CMF_NEMO_PROF=1`: host encode and device time of every frame submit.
fn prof() -> bool {
    std::env::var("CMF_NEMO_PROF").as_deref() == Ok("1")
}

/// `CMF_NEMO_SKIP=tag,tag`: leave these dispatch classes out of the step
/// (garbage output; the time difference is the class's cost). Tags:
/// in_proj scan out_proj qkv attn o_proj router exp_up exp_down sh_down
/// final head.
fn skip_tags() -> Vec<&'static str> {
    static TAGS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    TAGS.get_or_init(|| {
        std::env::var("CMF_NEMO_SKIP")
            .map(|v| {
                v.split(',')
                    .map(|t| &*Box::leak(t.trim().to_string().into_boxed_str()))
                    .collect()
            })
            .unwrap_or_default()
    })
    .clone()
}
const ATTN_SPLITS: usize = 256;

fn uwords(c: &Ctx, v: &[u32]) -> wgpu::Buffer {
    c.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// Arguments of one `nf_q8` dispatch; unset buffers bind the dummies.
#[derive(Default)]
struct Nq<'b> {
    /// (weights, output, rows) of matrices a, b, c
    mats: Vec<(&'b wgpu::Buffer, Option<&'b wgpu::Buffer>, usize)>,
    cols: usize,
    x: Option<&'b wgpu::Buffer>,
    pro: u32,
    nw: Option<&'b wgpu::Buffer>,
    zx: Option<&'b wgpu::Buffer>,
    gs: usize,
    epi: u32,
    h: Option<&'b wgpu::Buffer>,
    xo: Option<&'b wgpu::Buffer>,
    eo: Option<&'b wgpu::Buffer>,
    sw: Option<&'b wgpu::Buffer>,
    /// f32 rows: (weights, output, rows)
    f: Option<(&'b wgpu::Buffer, &'b wgpu::Buffer, usize)>,
    kv: Option<(&'b wgpu::Buffer, &'b wgpu::Buffer)>,
    kvw: usize,
    /// (ring, lo, hi, row stride, zx offset)
    ring: Option<(&'b wgpu::Buffer, usize, usize, usize, usize)>,
    /// (first column, full row width): a column range of wider matrices
    span: Option<(usize, usize)>,
}

impl NemoFrame {
    /// Upload the model; None when the device lacks a feature or a tensor
    /// is outside the frame's contract (the caller keeps the host path).
    pub fn build(d: &NemoDesc) -> Option<Self> {
        let c = ctx()?;
        // KV positions: the model's 65 536 on a discrete card; unified memory
        // (Apple) shares the weights' RAM, so ~27 min of audio by default
        let cap = std::env::var("CMF_NEMO_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(if c.discrete { d.cap } else { d.cap.min(20480) });
        let inner = d.m_heads * d.m_head_dim;
        let cdim = inner + 2 * d.m_groups * d.m_state;
        let kvw = d.nkv * d.hd;
        let gs = inner / d.m_groups.max(1);
        if d.hd != 128
            || (d.nh / d.nkv) % 4 != 0
            || d.m_head_dim != 64
            || d.m_state > 256
            || d.m_state % 4 != 0
            || d.m_groups > 32
            || gs % 4 != 0
            || d.n_exp > 256
            || d.top_k > 16
            || [d.hidden, inner, d.shared_inter, d.nh * d.hd, d.moe_inter]
                .iter()
                .any(|&n| n > 4096 || n % 32 != 0)
        {
            tracing::warn!("nemotron frame: geometry outside the kernels");
            return None;
        }
        let p = pipes(c)?;
        // Uploads go straight from the mmap through `write_buffer`, flushed
        // every ~1 GB so the staging copies never pile up for the whole
        // model (17+ GB of host staging otherwise sat in RAM until the end).
        let since_flush = std::cell::Cell::new(0usize);
        let total = std::cell::Cell::new(0u64);
        let put = |buf: &wgpu::Buffer, off: u64, data: &[u8]| {
            let body = data.len() / 4 * 4;
            if body > 0 {
                c.queue.write_buffer(buf, off, &data[..body]);
            }
            if body < data.len() {
                let mut tail = data[body..].to_vec();
                tail.resize(4, 0);
                c.queue.write_buffer(buf, off + body as u64, &tail);
            }
            total.set(total.get() + data.len() as u64);
            since_flush.set(since_flush.get() + data.len());
            if since_flush.get() > (1 << 30) {
                c.queue.submit(None);
                let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
                since_flush.set(0);
            }
        };
        let alloc = |label: &str, size: usize| -> wgpu::Buffer {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                // padded: vec4 views and the code lookup past a row's end
                size: (size.div_ceil(16) * 16 + 16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let up = |label: &str, data: &[u8]| -> wgpu::Buffer {
            let b = alloc(label, data.len());
            put(&b, 0, data);
            b
        };
        let f32b = |v: &[f32]| -> Vec<u8> { bytemuck::cast_slice(v).to_vec() };
        let zeros = |label: &str, n: usize| -> wgpu::Buffer {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n.max(4) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        // ── residency: every expert when the card holds them, else the
        // same number of slots in every MoE layer ──
        let mut n_moe = 0usize;
        let mut codec = ExpertCodec::Q4tp;
        let (mut up_bytes, mut down_bytes) = (0usize, 0usize);
        let mut other = 0u64;
        for b in &d.blocks {
            match b {
                NemoBlockDesc::Mamba {
                    in_proj, out_proj, ..
                } => other += (in_proj.len() + out_proj.len()) as u64,
                NemoBlockDesc::Attn { q, k, v, o, .. } => {
                    other += (q.len() + k.len() + v.len() + o.len()) as u64
                        + (2 * cap * kvw * 4) as u64
                }
                NemoBlockDesc::Moe {
                    codec: cd,
                    up,
                    down,
                    sh_up,
                    sh_down,
                    router,
                    ..
                } => {
                    if n_moe > 0 && *cd != codec {
                        tracing::warn!("nemotron frame: mixed expert codecs");
                        return None;
                    }
                    codec = *cd;
                    n_moe += 1;
                    let one_up = up.first()?.len();
                    let one_down = down.first()?.len();
                    if up.len() != d.n_exp
                        || down.len() != d.n_exp
                        || up.iter().any(|m| m.len() != one_up)
                        || down.iter().any(|m| m.len() != one_down)
                    {
                        return None;
                    }
                    up_bytes = one_up.div_ceil(16) * 16;
                    down_bytes = one_down.div_ceil(16) * 16;
                    other += (sh_up.len() + sh_down.len() + router.len() * 4) as u64;
                }
            }
        }
        other += (d.embed.len() + d.lm_head.len()) as u64 + (256 << 20);
        let mtp_bytes: u64 = d.mtp.as_ref().map_or(0, |m| {
            (m.eh.len() + m.q.len() + m.k.len() + m.v.len() + m.o.len() + m.sh_up.len() + m.sh_down.len()) as u64
                + m.up.iter().chain(&m.down).map(|e| e.len() as u64).sum::<u64>()
                + (2 * cap * kvw * 4) as u64
        });
        let per_layer = d.n_exp * (up_bytes + down_bytes);
        let budget = std::env::var("CMF_NEMO_EXPERT_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|gb| (gb * 1e9) as u64)
            .unwrap_or_else(|| {
                device_vram_budget().saturating_sub(other.saturating_add(d.reserve))
            });
        let slots = if n_moe == 0 || (n_moe * per_layer) as u64 + mtp_bytes <= budget {
            d.n_exp
        } else {
            ((budget / (n_moe * (up_bytes + down_bytes)) as u64) as usize).min(d.n_exp)
        };
        if slots < d.top_k {
            tracing::warn!(
                "nemotron frame: {:.1} GB for experts holds {slots} per layer, fewer than top-{}",
                budget as f64 / 1e9,
                d.top_k
            );
            return None;
        }
        let partial = slots < d.n_exp;
        if partial && d.model.is_none() {
            return None;
        }
        let mut arena = partial.then(|| Arena {
            slots,
            map: vec![MISS; n_moe * d.n_exp],
            owner: vec![vec![MISS; slots]; n_moe],
            score: vec![0.0; n_moe * d.n_exp],
            seen: vec![0; n_moe * d.n_exp],
            src: Vec::with_capacity(n_moe),
            _model: d.model.clone().unwrap(),
            misses: 0,
            admitted: 0,
            admit_mark: 0,
        });

        let mut blocks = Vec::with_capacity(d.blocks.len());
        let mut moe = 0usize;
        for b in &d.blocks {
            blocks.push(match b {
                NemoBlockDesc::Mamba {
                    norm,
                    in_proj,
                    out_proj,
                    conv_w,
                    conv_b,
                    par,
                    gnorm,
                } => DevBlock::Mamba {
                    norm: up("nm-norm", &f32b(norm)),
                    in_proj: up("nm-in", in_proj),
                    out_proj: up("nm-out", out_proj),
                    conv_w: up("nm-convw", &f32b(conv_w)),
                    conv_b: up("nm-convb", &f32b(conv_b)),
                    par: up("nm-par", &f32b(par)),
                    gnorm: up("nm-gnorm", &f32b(gnorm)),
                    ring: zeros("nm-ring", 3 * cdim),
                    state: zeros("nm-ssm", d.m_heads * d.m_head_dim * d.m_state),
                },
                NemoBlockDesc::Attn { norm, q, k, v, o } => DevBlock::Attn {
                    norm: up("nm-norm", &f32b(norm)),
                    q: up("nm-q", q),
                    k: up("nm-k", k),
                    v: up("nm-v", v),
                    o: up("nm-o", o),
                    kc: zeros("nm-kc", cap * kvw),
                    vc: zeros("nm-vc", cap * kvw),
                },
                NemoBlockDesc::Moe {
                    norm,
                    router,
                    bias,
                    up: ups,
                    down,
                    hot,
                    sh_up,
                    sh_down,
                    ..
                } => {
                    let upb = alloc("nm-up-bank", up_bytes * slots);
                    let dnb = alloc("nm-down-bank", down_bytes * slots);
                    // a full bank in expert order; a partial one with the
                    // hottest experts known, else the first ones
                    let order: Vec<usize> = match (partial, hot) {
                        (true, Some(h)) => h
                            .iter()
                            .copied()
                            .filter(|&e| e < d.n_exp)
                            .chain(0..d.n_exp)
                            .fold(Vec::new(), |mut v, e| {
                                if !v.contains(&e) {
                                    v.push(e);
                                }
                                v
                            }),
                        _ => (0..d.n_exp).collect(),
                    };
                    for (slot, &e) in order.iter().take(slots).enumerate() {
                        put(&upb, (slot * up_bytes) as u64, ups[e]);
                        put(&dnb, (slot * down_bytes) as u64, down[e]);
                        if let Some(a) = arena.as_mut() {
                            a.map[moe * d.n_exp + e] = slot as u32;
                            a.owner[moe][slot] = e as u32;
                        }
                    }
                    if let Some(a) = arena.as_mut() {
                        a.src.push(
                            (0..d.n_exp)
                                .map(|e| {
                                    [
                                        (ups[e].as_ptr() as usize, ups[e].len()),
                                        (down[e].as_ptr() as usize, down[e].len()),
                                    ]
                                })
                                .collect(),
                        );
                    }
                    moe += 1;
                    DevBlock::Moe {
                        norm: up("nm-norm", &f32b(norm)),
                        router: up("nm-router", &f32b(router)),
                        bias: up("nm-bias", &f32b(bias)),
                        up: upb,
                        up_stride: up_bytes / 4,
                        down: dnb,
                        down_stride: down_bytes / 4,
                        sh_up: up("nm-shup", sh_up),
                        sh_down: up("nm-shdown", sh_down),
                        moe: moe - 1,
                    }
                }
            });
        }
        // the draft layer rides along with full banks only (speculation
        // needs them; a partial bank keeps its slots for the experts)
        let mtp = match &d.mtp {
            Some(m) if !partial && m.up.len() == d.n_exp && m.down.len() == d.n_exp => {
                let one_up = m.up[0].len().div_ceil(16) * 16;
                let one_down = m.down[0].len().div_ceil(16) * 16;
                let upb = alloc("nm-mtp-up", one_up * d.n_exp);
                let dnb = alloc("nm-mtp-down", one_down * d.n_exp);
                for e in 0..d.n_exp {
                    put(&upb, (e * one_up) as u64, m.up[e]);
                    put(&dnb, (e * one_down) as u64, m.down[e]);
                }
                Some(DevMtp {
                    enorm: up("nm-mtp-enorm", &f32b(m.enorm)),
                    hnorm: up("nm-mtp-hnorm", &f32b(m.hnorm)),
                    eh: up("nm-mtp-eh", m.eh),
                    h_first: m.h_first,
                    n_attn: up("nm-mtp-na", &f32b(m.n_attn)),
                    q: up("nm-mtp-q", m.q),
                    k: up("nm-mtp-k", m.k),
                    v: up("nm-mtp-v", m.v),
                    o: up("nm-mtp-o", m.o),
                    kc: zeros("nm-mtp-kc", cap * kvw),
                    vc: zeros("nm-mtp-vc", cap * kvw),
                    n_moe: up("nm-mtp-nm", &f32b(m.n_moe)),
                    router: up("nm-mtp-router", &f32b(m.router)),
                    bias: up("nm-mtp-bias", &f32b(m.bias)),
                    up: upb,
                    up_stride: one_up / 4,
                    down: dnb,
                    down_stride: one_down / 4,
                    sh_up: up("nm-mtp-shup", m.sh_up),
                    sh_down: up("nm-mtp-shdown", m.sh_down),
                    fnorm: up("nm-mtp-fnorm", &f32b(m.fnorm)),
                    hin: zeros("nm-mtp-hin", d.hidden),
                    e: zeros("nm-mtp-e", d.hidden),
                    x: zeros("nm-mtp-x", d.hidden),
                    out: zeros("nm-mtp-out", d.hidden),
                })
            }
            _ => None,
        };
        let embed = up("nm-embed", d.embed);
        let lm_head = up("nm-head", d.lm_head);
        let final_norm = up("nm-fnorm", &f32b(d.final_norm));
        let in_rows = inner + cdim + d.m_heads;
        let max_parts = ATTN_SPLITS * 8;
        let blocks_ax = d.vocab.div_ceil(1024);
        let map_words = ((n_moe + 1) * d.n_exp).max(4);
        let stage_words = STAGE_IDS + META_WORDS + map_words + 16 + d.hidden;
        let stage = || {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nm-stage"),
                size: (stage_words * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let s = Scratch {
            h: zeros("nm-h", d.hidden),
            xn: zeros("nm-xn", d.hidden),
            zx: zeros("nm-zx", in_rows),
            y: zeros("nm-y", inner.max(d.nh * d.hd)),
            q: zeros("nm-q", d.nh * d.hd),
            att: zeros("nm-att", d.nh * d.hd),
            part: zeros("nm-part", max_parts * d.nh * (d.hd + 2)),
            logits_r: zeros("nm-rl", d.n_exp),
            sel: zeros("nm-sel", 16),
            sw: zeros("nm-sw", 16),
            mid: zeros("nm-mid", d.top_k * d.moe_inter),
            eo: zeros("nm-eo", d.top_k * d.hidden),
            smid: zeros("nm-smid", d.shared_inter),
            logits: zeros("nm-logits", d.vocab),
            bv: zeros("nm-bv", blocks_ax),
            bi: zeros("nm-bi", blocks_ax),
            ids: zeros("nm-ids", STAGE_IDS),
            meta: zeros("nm-meta", META_WORDS),
            map: zeros("nm-map", map_words),
            usage: zeros("nm-usage", map_words),
            ro: zeros("nm-ro", 64),
            rw: zeros("nm-rw", 64),
            stage: [stage(), stage()],
            stage_logits: c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nm-stage-logits"),
                size: (d.vocab * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        if let Some(a) = arena.as_ref() {
            // the draft layer's bank is always whole: identity after the layers
            let mut map = a.map.clone();
            map.extend((0..d.n_exp as u32).collect::<Vec<_>>());
            c.queue
                .write_buffer(&s.map, 0, bytemuck::cast_slice(&map));
        }
        c.queue.submit(None);
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        if partial {
            tracing::info!(
                "nemotron frame: {slots} of {} experts resident per layer ({:.1} GB), the rest from the file",
                d.n_exp,
                (n_moe * slots * (up_bytes + down_bytes)) as f64 / 1e9
            );
        }
        let bytes = total.get();
        let mut f = Self {
            c,
            p,
            blocks,
            plan: Vec::new(),
            moe_at: Vec::new(),
            body_end: 0,
            head_end: 0,
            pf: None,
            embed,
            lm_head,
            final_norm,
            s,
            hidden: d.hidden,
            vocab: d.vocab,
            eps: d.eps,
            m_heads: d.m_heads,
            m_head_dim: d.m_head_dim,
            m_groups: d.m_groups,
            m_state: d.m_state,
            nh: d.nh,
            nkv: d.nkv,
            hd: d.hd,
            top_k: d.top_k,
            moe_inter: d.moe_inter,
            shared_inter: d.shared_inter,
            route_scale: d.route_scale,
            norm_topk: d.norm_topk,
            n_exp: d.n_exp,
            n_moe,
            codec,
            cap,
            max_parts,
            arena,
            flights: std::collections::VecDeque::new(),
            next_stage: 0,
            pending: None,
            pending_verify: 0,
            vf: None,
            mtp,
            mtp_plan: Vec::new(),
            mtp_head_at: 0,
            ts: None,
            pos: 0,
            bytes,
        };
        f.build_plan();
        f.build_mtp_plan();
        if std::env::var("CMF_NEMO_TS").as_deref() == Ok("1")
            && c.device.features().contains(
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
            )
        {
            let n = f.plan.len() as u32 + 16;
            let qs = c.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("nm-ts"),
                ty: wgpu::QueryType::Timestamp,
                count: n,
            });
            let resolve = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nm-ts-resolve"),
                size: n as u64 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nm-ts-read"),
                size: n as u64 * 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            f.ts = Some((qs, resolve, read));
        }
        Some(f)
    }

    /// True when every expert is resident (no misses, pipelined submits).
    pub fn full(&self) -> bool {
        self.arena.is_none()
    }

    /// Seed the device state from the host recurrence after a host prefill:
    /// per Mamba block its conv ring `[(k−1)·cdim]` and scan state
    /// `[head][head_dim][state]`, per attention block its K/V rows.
    pub fn seed(
        &mut self,
        conv: &[Vec<f32>],
        ssm: &[Vec<f32>],
        kc: &[Vec<f32>],
        vc: &[Vec<f32>],
        pos: usize,
    ) -> bool {
        if pos > self.cap || !self.flights.is_empty() {
            return false;
        }
        self.pending = None;
        let (nh, hdim, ns) = (self.m_heads, self.m_head_dim, self.m_state);
        let (mut mi, mut ai) = (0, 0);
        for b in &self.blocks {
            match b {
                DevBlock::Mamba { ring, state, .. } => {
                    self.c
                        .queue
                        .write_buffer(ring, 0, bytemuck::cast_slice(&conv[mi]));
                    let src = &ssm[mi];
                    let mut t = vec![0f32; src.len()];
                    for h in 0..nh {
                        for p in 0..hdim {
                            for k in 0..ns {
                                t[(h * ns + k) * hdim + p] = src[(h * hdim + p) * ns + k];
                            }
                        }
                    }
                    self.c
                        .queue
                        .write_buffer(state, 0, bytemuck::cast_slice(&t));
                    mi += 1;
                }
                DevBlock::Attn { kc: kb, vc: vb, .. } => {
                    let n = pos * self.nkv * self.hd;
                    if kc[ai].len() < n || vc[ai].len() < n {
                        return false;
                    }
                    if n > 0 {
                        self.c
                            .queue
                            .write_buffer(kb, 0, bytemuck::cast_slice(&kc[ai][..n]));
                        self.c
                            .queue
                            .write_buffer(vb, 0, bytemuck::cast_slice(&vc[ai][..n]));
                    }
                    ai += 1;
                }
                DevBlock::Moe { .. } => {}
            }
        }
        self.pos = pos;
        true
    }

    fn bind(
        &self,
        pipe: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        u: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = pipe.get_bind_group_layout(0);
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| bind_buf(i as u32, b))
            .collect();
        entries.push(bind_buf(bufs.len() as u32, u));
        self.c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        })
    }

    fn u(&self, v: [u32; 8]) -> wgpu::Buffer {
        uniform_u32x8(self.c, v)
    }

    fn fixed(
        &self,
        tag: &'static str,
        pipe: &'static wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        u: &[u32],
        groups: (u32, u32),
    ) -> Op {
        let ub = uwords(self.c, u);
        Op::Fixed {
            pipe,
            bg: self.bind(pipe, bufs, &ub),
            groups,
            tag,
        }
    }

    fn nq(&self, tag: &'static str, a: Nq) -> Op {
        let s = &self.s;
        let rows = |i: usize| a.mats.get(i).map_or(0, |m| m.2);
        let rows_max = rows(0).max(rows(1)).max(rows(2));
        let rpw: usize = if rows_max >= 32768 { 4 } else { 2 };
        let per = 8 * rpw;
        let frows = a.f.map_or(0, |f| f.2);
        let groups = (0..3).map(|i| rows(i).div_ceil(per)).sum::<usize>() + frows.div_ceil(8);
        let w = |i: usize| a.mats.get(i).map_or(&s.ro, |m| m.0);
        let y = |i: usize| a.mats.get(i).and_then(|m| m.1).unwrap_or(&s.rw);
        let (kc, vc) = a.kv.unwrap_or((&s.rw, &s.rw));
        let (ring, lo, hi, n, off) = a.ring.unwrap_or((&s.rw, 0, 0, 0, 0));
        let u = [
            rows(0) as u32,
            rows(1) as u32,
            rows(2) as u32,
            a.cols as u32,
            rpw as u32,
            a.pro,
            a.epi,
            self.eps.to_bits(),
            a.gs as u32,
            frows as u32,
            self.top_k as u32,
            a.kvw as u32,
            lo as u32,
            hi as u32,
            n as u32,
            off as u32,
            a.xo.is_some() as u32,
            a.span.map_or(0, |s| s.0) as u32,
            a.span.map_or(0, |s| s.1) as u32,
            0,
        ];
        let bufs: [&wgpu::Buffer; 19] = [
            w(0),
            w(1),
            w(2),
            a.x.unwrap_or(&s.ro),
            a.nw.unwrap_or(&s.ro),
            a.zx.unwrap_or(&s.ro),
            y(0),
            y(1),
            y(2),
            a.h.unwrap_or(&s.rw),
            a.xo.unwrap_or(&s.rw),
            a.eo.unwrap_or(&s.ro),
            a.sw.unwrap_or(&s.ro),
            a.f.map_or(&s.ro, |f| f.0),
            a.f.map_or(&s.rw, |f| f.1),
            &s.meta,
            kc,
            vc,
            ring,
        ];
        self.fixed(tag, &self.p.q8, &bufs, &u, (groups as u32, 1))
    }

    /// Routed experts of one MoE layer: `up` picks the experts itself.
    #[allow(clippy::too_many_arguments)]
    fn bank(
        &self,
        tag: &'static str,
        w: &wgpu::Buffer,
        stride_w: usize,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        up: Option<(&wgpu::Buffer, &wgpu::Buffer)>,
        moe: usize,
    ) -> Op {
        let s = &self.s;
        let (lg, bias) = up.unwrap_or((&s.ro, &s.ro));
        let route = up.is_some();
        // rows per workgroup = 32·nrep (CMF_NEMO_BANK_REP, A/B knob)
        let nrep: usize = std::env::var("CMF_NEMO_BANK_REP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| (1..=8).contains(&n))
            .unwrap_or(1);
        let u = [
            rows as u32,
            (cols / 32) as u32,
            stride_w as u32,
            route as u32,
            if route { 0 } else { self.moe_inter as u32 },
            rows as u32,
            route as u32,
            self.n_exp as u32,
            self.top_k as u32,
            self.route_scale.to_bits(),
            self.norm_topk as u32,
            moe as u32,
            self.arena.is_some() as u32,
            cols as u32,
            nrep as u32,
            0,
        ];
        let pipe = match self.codec {
            ExpertCodec::Q4tp => &self.p.bank4,
            ExpertCodec::Q8 => &self.p.bank8,
        };
        let bufs = [
            w, w, x, lg, bias, &s.sel, &s.sw, y, &s.meta, &s.map, &s.usage,
        ];
        self.fixed(
            tag,
            pipe,
            &bufs,
            &u,
            (rows.div_ceil(32 * nrep) as u32, self.top_k as u32),
        )
    }

    /// The position-independent dispatch list of one token step.
    fn build_plan(&mut self) {
        let s = &self.s;
        let p = self.p;
        let hid = self.hidden;
        let eps = self.eps.to_bits();
        let inner = self.m_heads * self.m_head_dim;
        let cdim = inner + 2 * self.m_groups * self.m_state;
        let in_rows = inner + cdim + self.m_heads;
        let kvw = self.nkv * self.hd;
        let mut plan = Vec::new();
        let mut moe_at = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            match b {
                DevBlock::Mamba {
                    norm,
                    in_proj,
                    out_proj,
                    conv_w,
                    conv_b,
                    par,
                    gnorm,
                    ring,
                    state,
                } => {
                    plan.push(self.nq(
                        "in_proj",
                        Nq {
                            mats: vec![(in_proj, Some(&s.zx), in_rows)],
                            cols: hid,
                            x: Some(&s.h),
                            pro: 1,
                            nw: Some(norm),
                            ..Default::default()
                        },
                    ));
                    plan.push(self.fixed(
                        "scan",
                        &p.scan,
                        &[&s.zx, conv_w, conv_b, ring, state, par, &s.y, &s.meta],
                        &[
                            inner as u32,
                            (inner + cdim) as u32,
                            self.m_groups as u32,
                            self.m_state as u32,
                            (self.m_heads / self.m_groups) as u32,
                            self.m_head_dim as u32,
                            self.m_heads as u32,
                            cdim as u32,
                        ],
                        (self.m_heads as u32, 1),
                    ));
                    plan.push(self.nq(
                        "out_proj",
                        Nq {
                            mats: vec![(out_proj, None, hid)],
                            cols: inner,
                            x: Some(&s.y),
                            pro: 2,
                            nw: Some(gnorm),
                            zx: Some(&s.zx),
                            gs: inner / self.m_groups,
                            epi: 2,
                            h: Some(&s.h),
                            ring: Some((ring, inner, cdim, cdim, inner)),
                            ..Default::default()
                        },
                    ));
                }
                DevBlock::Attn {
                    norm,
                    q,
                    k,
                    v,
                    o,
                    kc,
                    vc,
                } => {
                    plan.push(self.nq(
                        "qkv",
                        Nq {
                            mats: vec![
                                (q, Some(&s.q), self.nh * self.hd),
                                (k, None, kvw),
                                (v, None, kvw),
                            ],
                            cols: hid,
                            x: Some(&s.h),
                            pro: 1,
                            nw: Some(norm),
                            epi: 4,
                            kv: Some((kc, vc)),
                            kvw,
                            ..Default::default()
                        },
                    ));
                    plan.push(Op::Attn { block: bi });
                    plan.push(self.nq(
                        "o_proj",
                        Nq {
                            mats: vec![(o, None, hid)],
                            cols: self.nh * self.hd,
                            x: Some(&s.att),
                            epi: 2,
                            h: Some(&s.h),
                            ..Default::default()
                        },
                    ));
                }
                DevBlock::Moe {
                    norm,
                    router,
                    bias,
                    up,
                    up_stride,
                    down,
                    down_stride,
                    sh_up,
                    sh_down,
                    moe,
                } => {
                    plan.push(self.nq(
                        "router",
                        Nq {
                            mats: vec![(sh_up, Some(&s.smid), self.shared_inter)],
                            cols: hid,
                            x: Some(&s.h),
                            pro: 1,
                            nw: Some(norm),
                            epi: 1,
                            xo: Some(&s.xn),
                            f: Some((router, &s.logits_r, self.n_exp)),
                            ..Default::default()
                        },
                    ));
                    moe_at.push(plan.len());
                    plan.push(self.bank(
                        "exp_up",
                        up,
                        *up_stride,
                        &s.xn,
                        &s.mid,
                        self.moe_inter,
                        hid,
                        Some((&s.logits_r, bias)),
                        *moe,
                    ));
                    plan.push(self.bank(
                        "exp_down",
                        down,
                        *down_stride,
                        &s.mid,
                        &s.eo,
                        hid,
                        self.moe_inter,
                        None,
                        *moe,
                    ));
                    plan.push(self.nq(
                        "sh_down",
                        Nq {
                            mats: vec![(sh_down, None, hid)],
                            cols: self.shared_inter,
                            x: Some(&s.smid),
                            epi: 3,
                            h: Some(&s.h),
                            eo: Some(&s.eo),
                            sw: Some(&s.sw),
                            ..Default::default()
                        },
                    ));
                }
            }
        }
        let body_end = plan.len();
        plan.push(self.fixed(
            "final",
            &p.rms,
            &[&s.h, &self.final_norm, &s.xn, &s.meta],
            &[hid as u32, eps, 0, 0],
            (1, 1),
        ));
        plan.push(self.nq(
            "head",
            Nq {
                mats: vec![(&self.lm_head, Some(&s.logits), self.vocab)],
                cols: hid,
                x: Some(&s.xn),
                ..Default::default()
            },
        ));
        let head_end = plan.len();
        let blocks_ax = self.vocab.div_ceil(1024);
        plan.push(self.fixed(
            "argmax",
            &p.argmax1,
            &[&s.logits, &s.bv, &s.bi],
            &[self.vocab as u32, blocks_ax as u32, 0, 0],
            (blocks_ax as u32, 1),
        ));
        plan.push(self.fixed(
            "pick",
            &p.pick,
            &[&s.bv, &s.bi, &s.ids, &self.embed, &s.h, &s.meta],
            &[blocks_ax as u32, self.vocab as u32, hid as u32, 0],
            (1, 1),
        ));
        self.plan = plan;
        self.moe_at = moe_at;
        self.body_end = body_end;
        self.head_end = head_end;
    }

    /// The draft layer's step: fusion (two column halves of `eh`, each with
    /// its own norm), attention over the draft KV, MoE, output norm; then
    /// (from `mtp_head_at`) the head, argmax and the draft's embedding into
    /// the next step's input row.
    fn build_mtp_plan(&mut self) {
        let Some(m) = self.mtp.as_ref() else {
            return;
        };
        let s = &self.s;
        let p = self.p;
        let hid = self.hidden;
        let kvw = self.nkv * self.hd;
        let (e_off, h_off) = if m.h_first { (hid, 0) } else { (0, hid) };
        let mut plan = Vec::new();
        plan.push(self.nq(
            "mtp",
            Nq {
                mats: vec![(&m.eh, Some(&m.x), hid)],
                cols: hid,
                span: Some((e_off, 2 * hid)),
                x: Some(&m.e),
                pro: 1,
                nw: Some(&m.enorm),
                ..Default::default()
            },
        ));
        plan.push(self.nq(
            "mtp",
            Nq {
                mats: vec![(&m.eh, None, hid)],
                cols: hid,
                span: Some((h_off, 2 * hid)),
                x: Some(&m.hin),
                pro: 1,
                nw: Some(&m.hnorm),
                epi: 2,
                h: Some(&m.x),
                ..Default::default()
            },
        ));
        plan.push(self.nq(
            "mtp",
            Nq {
                mats: vec![(&m.q, Some(&s.q), self.nh * self.hd), (&m.k, None, kvw), (&m.v, None, kvw)],
                cols: hid,
                x: Some(&m.x),
                pro: 1,
                nw: Some(&m.n_attn),
                epi: 4,
                kv: Some((&m.kc, &m.vc)),
                kvw,
                ..Default::default()
            },
        ));
        plan.push(Op::MtpAttn);
        plan.push(self.nq(
            "mtp",
            Nq {
                mats: vec![(&m.o, None, hid)],
                cols: self.nh * self.hd,
                x: Some(&s.att),
                epi: 2,
                h: Some(&m.x),
                ..Default::default()
            },
        ));
        plan.push(self.nq(
            "mtp",
            Nq {
                mats: vec![(&m.sh_up, Some(&s.smid), self.shared_inter)],
                cols: hid,
                x: Some(&m.x),
                pro: 1,
                nw: Some(&m.n_moe),
                epi: 1,
                xo: Some(&s.xn),
                f: Some((&m.router, &s.logits_r, self.n_exp)),
                ..Default::default()
            },
        ));
        let mut ops = vec![
            self.bank(
                "mtp",
                &m.up,
                m.up_stride,
                &s.xn,
                &s.mid,
                self.moe_inter,
                hid,
                Some((&s.logits_r, &m.bias)),
                self.n_moe,
            ),
            self.bank("mtp", &m.down, m.down_stride, &s.mid, &s.eo, hid, self.moe_inter, None, self.n_moe),
        ];
        plan.append(&mut ops);
        plan.push(self.nq(
            "mtp",
            Nq {
                mats: vec![(&m.sh_down, None, hid)],
                cols: self.shared_inter,
                x: Some(&s.smid),
                epi: 3,
                h: Some(&m.x),
                eo: Some(&s.eo),
                sw: Some(&s.sw),
                ..Default::default()
            },
        ));
        plan.push(self.fixed(
            "mtp",
            &p.rms,
            &[&m.x, &m.fnorm, &m.out, &s.meta],
            &[hid as u32, self.eps.to_bits(), 0, 0],
            (1, 1),
        ));
        let head_at = plan.len();
        plan.push(self.nq(
            "mtp",
            Nq {
                mats: vec![(&self.lm_head, Some(&s.logits), self.vocab)],
                cols: hid,
                x: Some(&m.out),
                ..Default::default()
            },
        ));
        let blocks_ax = self.vocab.div_ceil(1024);
        plan.push(self.fixed(
            "mtp",
            &p.argmax1,
            &[&s.logits, &s.bv, &s.bi],
            &[self.vocab as u32, blocks_ax as u32, 0, 0],
            (blocks_ax as u32, 1),
        ));
        plan.push(self.fixed(
            "mtp",
            &p.pick,
            &[&s.bv, &s.bi, &s.ids, &self.embed, &m.e, &s.meta],
            &[blocks_ax as u32, self.vocab as u32, hid as u32, 0],
            (1, 1),
        ));
        self.mtp_plan = plan;
        self.mtp_head_at = head_at;
    }

    /// True when the file's draft layer is on the device.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Draft-layer steps: each `(hidden, next, pos)` fuses a final hidden
    /// row with the next token's embedding and extends the draft KV at
    /// `pos`; with `draft`, the last step also runs the head and returns
    /// its argmax (and leaves its output row for a deeper draft).
    pub fn mtp_steps(&mut self, steps: &[(&[f32], &[f32], usize)], draft: bool) -> Option<u32> {
        let m = self.mtp.as_ref()?;
        let c = self.c;
        let hid = self.hidden;
        for (i, &(h, e, pos)) in steps.iter().enumerate() {
            if h.len() != hid || e.len() != hid || pos >= self.cap {
                return None;
            }
            c.queue.write_buffer(&m.hin, 0, bytemuck::cast_slice(h));
            c.queue.write_buffer(&m.e, 0, bytemuck::cast_slice(e));
            self.write_meta(pos, None);
            let last = i + 1 == steps.len();
            let end = if last && draft { self.mtp_plan.len() } else { self.mtp_head_at };
            let mut enc = c
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nm-mtp") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nm-mtp"),
                    timestamp_writes: None,
                });
                self.encode_mtp_ops(&mut pass, 0, end, pos);
            }
            if last && draft {
                enc.copy_buffer_to_buffer(&self.s.ids, 0, &self.s.stage[0], 0, 4);
            }
            c.queue.submit(Some(enc.finish()));
        }
        if !draft {
            return Some(0);
        }
        let st = &self.s.stage[0];
        let slice = st.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let id = {
            let d = slice.get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&d)[0]
        };
        st.unmap();
        Some(id)
    }

    fn encode_mtp_ops(&self, pass: &mut wgpu::ComputePass, a: usize, b: usize, pos: usize) {
        let s = &self.s;
        let p = self.p;
        let Some(m) = self.mtp.as_ref() else {
            return;
        };
        for op in &self.mtp_plan[a..b] {
            match op {
                Op::Fixed { pipe, bg, groups, .. } => {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(groups.0, groups.1, 1);
                }
                Op::MtpAttn | Op::Attn { .. } => {
                    let npos = pos + 1;
                    let chunk = npos.div_ceil(ATTN_SPLITS).max(64);
                    let splits = npos.div_ceil(chunk);
                    let scale = (1.0 / (self.hd as f32).sqrt()).to_bits();
                    self.run3(
                        pass,
                        &p.attn,
                        &[&s.q, &m.kc, &m.vc, &s.part],
                        [self.nh as u32, self.nkv as u32, self.hd as u32, npos as u32, chunk as u32, scale, 0, 0],
                        (splits as u32, self.nkv as u32),
                        (self.nh / self.nkv / 4) as u32,
                    );
                    self.run(
                        pass,
                        &p.attn_merge,
                        &[&s.part, &s.att],
                        [self.nh as u32, self.hd as u32, (splits * 8) as u32, 0, 0, 0, 0, 0],
                        (self.nh as u32, 1),
                    );
                }
            }
        }
    }

    fn run(
        &self,
        pass: &mut wgpu::ComputePass,
        pipe: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        u: [u32; 8],
        groups: (u32, u32),
    ) {
        self.run3(pass, pipe, bufs, u, groups, 1)
    }

    fn run3(
        &self,
        pass: &mut wgpu::ComputePass,
        pipe: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        u: [u32; 8],
        groups: (u32, u32),
        z: u32,
    ) {
        let ub = self.u(u);
        let bg = self.bind(pipe, bufs, &ub);
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, z);
    }

    /// Encode plan ops `[a, b)` of the step at `pos`; with `ts`, a
    /// timestamp before the first op and after every op.
    fn encode_ops(&self, pass: &mut wgpu::ComputePass, a: usize, b: usize, pos: usize) {
        self.encode_ops_ts(pass, a, b, pos, None)
    }

    fn encode_ops_ts(
        &self,
        pass: &mut wgpu::ComputePass,
        a: usize,
        b: usize,
        pos: usize,
        ts: Option<&wgpu::QuerySet>,
    ) {
        let s = &self.s;
        let p = self.p;
        let skip = skip_tags();
        if let Some(q) = ts {
            pass.write_timestamp(q, 0);
        }
        for (i, op) in self.plan[a..b].iter().enumerate() {
            if let Some(q) = ts
                && i > 0
            {
                pass.write_timestamp(q, i as u32);
            }
            match op {
                Op::Fixed { tag, .. } if skip.contains(tag) => {}
                Op::Attn { .. } if skip.contains(&"attn") => {}
                Op::MtpAttn => {}
                Op::Fixed {
                    pipe, bg, groups, ..
                } => {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(groups.0, groups.1, 1);
                }
                Op::Attn { block } => {
                    let DevBlock::Attn { kc, vc, .. } = &self.blocks[*block] else {
                        unreachable!()
                    };
                    let npos = pos + 1;
                    let chunk = npos.div_ceil(ATTN_SPLITS).max(64);
                    let splits = npos.div_ceil(chunk);
                    let scale = (1.0 / (self.hd as f32).sqrt()).to_bits();
                    self.run3(
                        pass,
                        &p.attn,
                        &[&s.q, kc, vc, &s.part],
                        [
                            self.nh as u32,
                            self.nkv as u32,
                            self.hd as u32,
                            npos as u32,
                            chunk as u32,
                            scale,
                            0,
                            0,
                        ],
                        (splits as u32, self.nkv as u32),
                        (self.nh / self.nkv / 4) as u32,
                    );
                    self.run(
                        pass,
                        &p.attn_merge,
                        &[&s.part, &s.att],
                        [
                            self.nh as u32,
                            self.hd as u32,
                            (splits * 8) as u32,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ],
                        (self.nh as u32, 1),
                    );
                }
            }
        }
        if let Some(q) = ts {
            pass.write_timestamp(q, (b - a) as u32);
        }
    }

    fn write_meta(&self, pos: usize, prov: Option<(usize, u32)>) {
        let (pl, pm) = prov.map_or((MISS, 0), |(l, m)| (l as u32, m));
        let meta: [u32; META_WORDS] = [pos as u32, 0, 0, 0, pl, pm, 0, 0];
        self.c
            .queue
            .write_buffer(&self.s.meta, 0, bytemuck::cast_slice(&meta));
    }

    /// Encode `steps` greedy steps from `pos` (the first one from plan op
    /// `from`), copy the readback into stage `st` and submit.
    fn submit_steps(&mut self, pos: usize, steps: usize, from: usize, st: usize) -> Flight {
        let t0 = std::time::Instant::now();
        let mut enc = self
            .c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nm-step"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nm-step"),
                timestamp_writes: None,
            });
            for t in 0..steps {
                let a = if t == 0 { from } else { 0 };
                let ts = (t == 0 && a == 0).then_some(self.ts.as_ref().map(|t| &t.0)).flatten();
                self.encode_ops_ts(&mut pass, a, self.plan.len(), pos + t, ts);
            }
        }
        if let Some((qs, resolve, read)) = &self.ts
            && from == 0
        {
            let n = self.plan.len() as u32 + 1;
            enc.resolve_query_set(qs, 0..n, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, read, 0, n as u64 * 8);
        }
        let s = &self.s;
        let stage = &s.stage[st];
        enc.copy_buffer_to_buffer(&s.ids, 0, stage, 0, (STAGE_IDS * 4) as u64);
        enc.copy_buffer_to_buffer(
            &s.meta,
            0,
            stage,
            (STAGE_IDS * 4) as u64,
            (META_WORDS * 4) as u64,
        );
        if self.arena.is_some() {
            let base = (STAGE_IDS + META_WORDS) * 4;
            let mw = self.n_moe * self.n_exp;
            enc.copy_buffer_to_buffer(&s.usage, 0, stage, base as u64, (mw * 4) as u64);
            let b2 = base + mw.max(4) * 4;
            enc.copy_buffer_to_buffer(&s.sel, 0, stage, b2 as u64, 32);
            enc.copy_buffer_to_buffer(&s.sw, 0, stage, b2 as u64 + 32, 32);
            enc.copy_buffer_to_buffer(
                &s.xn,
                0,
                stage,
                b2 as u64 + 64,
                (self.hidden * 4) as u64,
            );
        }
        let index = self.c.queue.submit(Some(enc.finish()));
        if prof() {
            eprintln!(
                "nemotron frame: {steps} steps at pos {pos}: encode {:.2} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Flight { stage: st, index, steps }
    }

    /// Wait for a flight and map its stage: (ids of the finished steps,
    /// meta words, the rest of the stage).
    fn land(&self, f: &Flight) -> Option<(Vec<u32>, [u32; META_WORDS], Vec<u32>)> {
        let t0 = std::time::Instant::now();
        let stage = &self.s.stage[f.stage];
        let slice = stage.slice(..);
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d2 = done.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                d2.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        if self
            .c
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(f.index.clone()),
                timeout: None,
            })
            .is_err()
        {
            stage.unmap();
            return None;
        }
        // the map callback runs on a later poll when the wait returned first
        while !done.load(std::sync::atomic::Ordering::Acquire) {
            if self.c.device.poll(wgpu::PollType::Poll).is_err() {
                stage.unmap();
                return None;
            }
            std::thread::yield_now();
        }
        let words: Vec<u32> = {
            let data = slice.get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&data).to_vec()
        };
        stage.unmap();
        if let Some((_, _, read)) = &self.ts {
            self.print_ts(read);
        }
        let mut meta = [0u32; META_WORDS];
        meta.copy_from_slice(&words[STAGE_IDS..STAGE_IDS + META_WORDS]);
        let n = (meta[6] as usize).min(f.steps);
        if prof() {
            eprintln!(
                "nemotron frame: {} steps landed, wait {:.2} ms",
                n,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Some((
            words[..n].to_vec(),
            meta,
            words[STAGE_IDS + META_WORDS..].to_vec(),
        ))
    }

    /// Per-class device time of the first step of the landed batch.
    fn print_ts(&self, read: &wgpu::Buffer) {
        let n = self.plan.len() + 1;
        let slice = read.slice(..(n * 8) as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.c.device.poll(wgpu::PollType::wait_indefinitely());
        let Ok(data) = slice.get_mapped_range() else {
            read.unmap();
            return;
        };
        let t: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&data).to_vec();
        drop(data);
        read.unmap();
        let period = self.c.queue.get_timestamp_period() as f64;
        let mut by: Vec<(&str, f64, usize)> = Vec::new();
        for (i, op) in self.plan.iter().enumerate() {
            let tag = match op {
                Op::Fixed { tag, .. } => *tag,
                Op::Attn { .. } | Op::MtpAttn => "attn",
            };
            let us = t[i + 1].saturating_sub(t[i]) as f64 * period / 1e3;
            match by.iter_mut().find(|b| b.0 == tag) {
                Some(b) => {
                    b.1 += us;
                    b.2 += 1;
                }
                None => by.push((tag, us, 1)),
            }
        }
        let total: f64 = by.iter().map(|b| b.1).sum();
        let line: Vec<String> = by
            .iter()
            .map(|(t, us, n)| format!("{t} {:.0}us/{n}", us))
            .collect();
        eprintln!("nemotron frame step: {:.0} us = {}", total, line.join(", "));
    }

    /// Start `steps` greedy steps at `self.pos`. `row`: the next input
    /// embedding; None continues from the embedding the last step left in
    /// `h`. Full banks only (a partial bank can stop on a miss: use
    /// [`Self::greedy`]); at most two batches in flight.
    pub fn greedy_start(&mut self, row: Option<&[f32]>, steps: usize) -> bool {
        if steps == 0
            || steps > STAGE_IDS
            || self.pos + steps > self.cap
            || self.arena.is_some()
            || self.flights.len() >= 2
            || row.is_some_and(|r| r.len() != self.hidden)
        {
            return false;
        }
        if let Some(r) = row {
            self.c
                .queue
                .write_buffer(&self.s.h, 0, bytemuck::cast_slice(r));
        }
        self.write_meta(self.pos, None);
        let st = self.next_stage;
        self.next_stage ^= 1;
        let f = self.submit_steps(self.pos, steps, 0, st);
        self.pos += steps;
        self.flights.push_back(f);
        true
    }

    /// The ids of the oldest batch in flight.
    pub fn greedy_finish(&mut self) -> Option<Vec<u32>> {
        let f = self.flights.pop_front()?;
        let (ids, meta, _) = self.land(&f)?;
        (meta[1] == 0 && ids.len() == f.steps).then_some(ids)
    }

    /// Drop the batches still in flight (the transcript ended): their steps
    /// leave the state ahead of the last id taken.
    pub fn greedy_drain(&mut self) {
        while let Some(f) = self.flights.pop_front() {
            let _ = self.land(&f);
        }
    }

    /// Greedy decode: `row` (the next input embedding) at `self.pos`, then
    /// `steps − 1` more positions fed by their own argmax, in one submit.
    /// The run stops early on a miss of a partial bank: complete the
    /// missing experts and call [`Self::resume`].
    pub fn greedy(&mut self, row: Option<&[f32]>, steps: usize) -> Option<NemoRun> {
        if steps == 0
            || steps > STAGE_IDS
            || self.pos + steps > self.cap
            || !self.flights.is_empty()
            || row.is_some_and(|r| r.len() != self.hidden)
        {
            return None;
        }
        self.pending = None;
        if let Some(r) = row {
            self.c
                .queue
                .write_buffer(&self.s.h, 0, bytemuck::cast_slice(r));
        }
        let pos = self.pos;
        self.write_meta(pos, None);
        self.run_from(pos, steps, 0, Vec::new())
    }

    fn run_from(&mut self, pos: usize, steps: usize, from: usize, mut ids: Vec<u32>) -> Option<NemoRun> {
        let st = self.next_stage;
        self.next_stage ^= 1;
        let f = self.submit_steps(pos, steps, from, st);
        let (got, meta, rest) = self.land(&f)?;
        let done = got.len();
        ids.extend_from_slice(&got);
        self.observe(&rest);
        if meta[1] == 0 {
            if done != steps {
                return None;
            }
            self.pos = pos + steps;
            self.admit();
            return Some(NemoRun { ids, miss: None });
        }
        // stopped inside step `done` at MoE layer meta[2]
        let moe = meta[2] as usize;
        let at = meta[3] as usize;
        if at != pos + done || moe >= self.n_moe {
            return None;
        }
        let a = self.arena.as_mut()?;
        let mw = (self.n_moe * self.n_exp).max(4);
        let k = self.top_k;
        let sel = &rest[mw..mw + k];
        let sw = &rest[mw + 8..mw + 8 + k];
        let xn: Vec<f32> = rest[mw + 16..mw + 16 + self.hidden]
            .iter()
            .map(|w| f32::from_bits(*w))
            .collect();
        let picks: Vec<(usize, f32)> = sel
            .iter()
            .zip(sw)
            .map(|(&e, &w)| (e as usize, f32::from_bits(w)))
            .collect();
        let missing: Vec<usize> = picks
            .iter()
            .enumerate()
            .filter(|(_, (e, _))| a.map[moe * self.n_exp + e] == MISS)
            .map(|(i, _)| i)
            .collect();
        a.misses += missing.len() as u64;
        for &i in &missing {
            // a miss weighs like a few uses: the admission prefers experts
            // that keep stopping the decode
            a.score[moe * self.n_exp + picks[i].0] += 4.0;
        }
        self.pos = at;
        self.pending = Some(Pending {
            left: steps - done,
            pos: at,
            moe,
            ids,
        });
        Some(NemoRun {
            ids: Vec::new(),
            miss: Some(NemoMiss {
                moe,
                xn,
                picks,
                missing,
            }),
        })
    }

    /// Continue a run that stopped on a miss: `outs` are the outputs of the
    /// missing experts (unweighted, `hidden` each) in the order of
    /// `NemoMiss::missing`.
    pub fn resume(&mut self, missing: &[usize], outs: &[Vec<f32>]) -> Option<NemoRun> {
        let pd = self.pending.take()?;
        let hid = self.hidden;
        let mut mask = 0u32;
        for (&slot, o) in missing.iter().zip(outs) {
            if o.len() != hid || slot >= self.top_k {
                return None;
            }
            mask |= 1 << slot;
            self.c.queue.write_buffer(
                &self.s.eo,
                (slot * hid * 4) as u64,
                bytemuck::cast_slice(o),
            );
        }
        let mut meta: [u32; META_WORDS] = [pd.pos as u32, 0, 0, 0, pd.moe as u32, mask, 0, 0];
        // the ids of the finished steps were read; the stopped step counts on
        meta[6] = 0;
        self.c
            .queue
            .write_buffer(&self.s.meta, 0, bytemuck::cast_slice(&meta));
        let from = self.moe_at[pd.moe];
        self.run_from(pd.pos, pd.left, from, pd.ids)
    }

    /// Greedy decode with a partial bank, one MoE layer at a time: each
    /// submit runs up to the next layer's expert kernel and waits, so a miss
    /// stops there instead of leaving the rest of the step's dispatches to
    /// run idle. `complete(moe, xn, picks, missing)` returns the missing
    /// experts' outputs; they are written where the bank would have.
    #[allow(clippy::type_complexity)]
    pub fn greedy_seg(
        &mut self,
        row: Option<&[f32]>,
        steps: usize,
        complete: &mut dyn FnMut(usize, &[f32], &[(usize, f32)], &[usize]) -> Vec<Vec<f32>>,
    ) -> Option<Vec<u32>> {
        if self.arena.is_none()
            || steps == 0
            || steps > STAGE_IDS
            || self.pos + steps > self.cap
            || !self.flights.is_empty()
            || row.is_some_and(|r| r.len() != self.hidden)
        {
            return None;
        }
        let c = self.c;
        let hid = self.hidden;
        if let Some(r) = row {
            c.queue.write_buffer(&self.s.h, 0, bytemuck::cast_slice(r));
        }
        self.write_meta(self.pos, None);
        let cuts: Vec<usize> = self.moe_at.iter().map(|&i| i + 1).collect();
        let mut ids = Vec::with_capacity(steps);
        for t in 0..steps {
            let pos = self.pos + t;
            let mut cursor = 0usize;
            while cursor < self.plan.len() {
                let end = cuts
                    .iter()
                    .copied()
                    .find(|&e| e > cursor)
                    .unwrap_or(self.plan.len());
                let mut enc = c
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nm-seg") });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("nm-seg"),
                        timestamp_writes: None,
                    });
                    self.encode_ops(&mut pass, cursor, end, pos);
                }
                let st = &self.s.stage[0];
                let mw = (self.n_moe * self.n_exp).max(4);
                let base = (STAGE_IDS + META_WORDS) * 4;
                enc.copy_buffer_to_buffer(&self.s.ids, 0, st, 0, (STAGE_IDS * 4) as u64);
                enc.copy_buffer_to_buffer(&self.s.meta, 0, st, (STAGE_IDS * 4) as u64, (META_WORDS * 4) as u64);
                enc.copy_buffer_to_buffer(&self.s.usage, 0, st, base as u64, ((self.n_moe * self.n_exp) * 4) as u64);
                let b2 = (base + mw * 4) as u64;
                enc.copy_buffer_to_buffer(&self.s.sel, 0, st, b2, 32);
                enc.copy_buffer_to_buffer(&self.s.sw, 0, st, b2 + 32, 32);
                enc.copy_buffer_to_buffer(&self.s.xn, 0, st, b2 + 64, (hid * 4) as u64);
                c.queue.submit(Some(enc.finish()));
                let slice = st.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
                let words: Vec<u32> = {
                    let d = slice.get_mapped_range().ok()?;
                    bytemuck::cast_slice::<u8, u32>(&d).to_vec()
                };
                st.unmap();
                let meta = &words[STAGE_IDS..STAGE_IDS + META_WORDS];
                if meta[1] == 0 {
                    if end == self.plan.len() {
                        ids.push(words[t]);
                        let rest = words[STAGE_IDS + META_WORDS..].to_vec();
                        self.observe(&rest);
                    }
                    cursor = end;
                    continue;
                }
                // the expert kernel of MoE layer meta[2] stopped on a miss
                let moe = meta[2] as usize;
                let rest = &words[STAGE_IDS + META_WORDS..];
                let k = self.top_k;
                let picks: Vec<(usize, f32)> = (0..k)
                    .map(|i| (rest[mw + i] as usize, f32::from_bits(rest[mw + 8 + i])))
                    .collect();
                let xn: Vec<f32> = rest[mw + 16..mw + 16 + hid].iter().map(|w| f32::from_bits(*w)).collect();
                let a = self.arena.as_mut()?;
                let missing: Vec<usize> = (0..k)
                    .filter(|&i| a.map[moe * self.n_exp + picks[i].0] == MISS)
                    .collect();
                a.misses += missing.len() as u64;
                for &i in &missing {
                    a.score[moe * self.n_exp + picks[i].0] += 4.0;
                }
                let outs = complete(moe, &xn, &picks, &missing);
                if outs.len() != missing.len() {
                    return None;
                }
                let mut mask = 0u32;
                for (&slot, o) in missing.iter().zip(&outs) {
                    mask |= 1 << slot;
                    c.queue
                        .write_buffer(&self.s.eo, (slot * hid * 4) as u64, bytemuck::cast_slice(o));
                }
                let m: [u32; META_WORDS] = [pos as u32, 0, 0, 0, moe as u32, mask, t as u32, 0];
                c.queue.write_buffer(&self.s.meta, 0, bytemuck::cast_slice(&m));
                cursor = self.moe_at[moe];
            }
        }
        self.pos += steps;
        self.admit();
        Some(ids)
    }

    /// Fold the device routing counts into the arena's scores.
    fn observe(&mut self, rest: &[u32]) {
        let Some(a) = self.arena.as_mut() else {
            return;
        };
        let mw = self.n_moe * self.n_exp;
        for i in 0..mw {
            let now = rest[i];
            let d = now.wrapping_sub(a.seen[i]);
            a.seen[i] = now;
            a.score[i] = a.score[i] * 0.98 + d as f32;
        }
    }

    /// Admit the best-scored missing experts in place of the worst-scored
    /// residents (each is one expert's upload): one per `CMF_NEMO_ADMIT`
    /// decoded positions (default 4), banked while no upload is due.
    fn admit(&mut self) {
        let every: usize = std::env::var("CMF_NEMO_ADMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
            .max(1);
        let Some(a) = self.arena.as_mut() else {
            return;
        };
        let due = (self.pos / every).saturating_sub(a.admit_mark);
        a.admit_mark = self.pos / every;
        self.admit_n(due.min(8));
    }

    fn admit_n(&mut self, budget: usize) {
        if budget == 0 {
            return;
        }
        let n_exp = self.n_exp;
        let c = self.c;
        let Some(a) = self.arena.as_mut() else {
            return;
        };
        let mut moves: Vec<(f32, usize, usize, usize)> = Vec::new();
        for m in 0..self.n_moe {
            let sc = &a.score[m * n_exp..(m + 1) * n_exp];
            let best = (0..n_exp)
                .filter(|&e| a.map[m * n_exp + e] == MISS)
                .max_by(|&x, &y| sc[x].total_cmp(&sc[y]));
            let worst = (0..a.slots).min_by(|&x, &y| {
                let ex = a.owner[m][x];
                let ey = a.owner[m][y];
                let fx = if ex == MISS { -1.0 } else { sc[ex as usize] };
                let fy = if ey == MISS { -1.0 } else { sc[ey as usize] };
                fx.total_cmp(&fy)
            });
            if let (Some(b), Some(w)) = (best, worst) {
                let ew = a.owner[m][w];
                let fw = if ew == MISS { -1.0 } else { sc[ew as usize] };
                if sc[b] > fw * 1.5 + 1.0 {
                    moves.push((sc[b] - fw, m, b, w));
                }
            }
        }
        moves.sort_by(|x, y| y.0.total_cmp(&x.0));
        for &(_, m, e, slot) in moves.iter().take(budget) {
            let Some(DevBlock::Moe {
                up,
                up_stride,
                down,
                down_stride,
                ..
            }) = self
                .blocks
                .iter()
                .find(|b| matches!(b, DevBlock::Moe { moe, .. } if *moe == m))
            else {
                continue;
            };
            let [(ua, ul), (da, dl)] = a.src[m][e];
            // SAFETY: the payloads live in the mapped file `_model` keeps open
            let (ub, db) = unsafe {
                (
                    std::slice::from_raw_parts(ua as *const u8, ul),
                    std::slice::from_raw_parts(da as *const u8, dl),
                )
            };
            for (buf, off, data) in [
                (up, slot * up_stride * 4, ub),
                (down, slot * down_stride * 4, db),
            ] {
                let body = data.len() / 4 * 4;
                c.queue.write_buffer(buf, off as u64, &data[..body]);
                if body < data.len() {
                    let mut tail = data[body..].to_vec();
                    tail.resize(4, 0);
                    c.queue.write_buffer(buf, (off + body) as u64, &tail);
                }
            }
            let old = a.owner[m][slot];
            if old != MISS {
                a.map[m * n_exp + old as usize] = MISS;
                c.queue.write_buffer(
                    &self.s.map,
                    ((m * n_exp + old as usize) * 4) as u64,
                    bytemuck::cast_slice(&[MISS]),
                );
            }
            a.owner[m][slot] = e as u32;
            a.map[m * n_exp + e] = slot as u32;
            c.queue.write_buffer(
                &self.s.map,
                ((m * n_exp + e) * 4) as u64,
                bytemuck::cast_slice(&[slot as u32]),
            );
            a.admitted += 1;
        }
    }

    /// (misses completed on the host, experts admitted) of a partial bank.
    pub fn arena_stats(&self) -> Option<(u64, u64)> {
        self.arena.as_ref().map(|a| (a.misses, a.admitted))
    }

    /// Greedy batches submitted and not yet read back.
    pub fn in_flight(&self) -> usize {
        self.flights.len()
    }

    /// One position from a host row, logits read back.
    pub fn logits(&mut self, row: &[f32], out: &mut Vec<f32>) -> bool {
        if self.pos + 1 > self.cap
            || row.len() != self.hidden
            || self.arena.is_some()
            || !self.flights.is_empty()
        {
            return false;
        }
        self.c
            .queue
            .write_buffer(&self.s.h, 0, bytemuck::cast_slice(row));
        self.write_meta(self.pos, None);
        let mut enc = self
            .c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nm-logits"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nm-logits"),
                timestamp_writes: None,
            });
            self.encode_ops(&mut pass, 0, self.head_end, self.pos);
        }
        let bytes = (self.vocab * 4) as u64;
        enc.copy_buffer_to_buffer(&self.s.logits, 0, &self.s.stage_logits, 0, bytes);
        self.c.queue.submit(Some(enc.finish()));
        let slice = self.s.stage_logits.slice(..);
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d2 = done.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            if r.is_ok() {
                d2.store(true, std::sync::atomic::Ordering::Release);
            }
        });
        if self
            .c
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_err()
            || !done.load(std::sync::atomic::Ordering::Acquire)
        {
            self.s.stage_logits.unmap();
            return false;
        }
        {
            let Ok(data) = slice.get_mapped_range() else {
                return false;
            };
            out.clear();
            out.extend(
                bytemuck::cast_slice::<u8, u32>(&data)
                    .iter()
                    .map(|w| f32::from_bits(*w)),
            );
        }
        self.s.stage_logits.unmap();
        self.pos += 1;
        true
    }

    /// Positions without readback (prefill through the frame): `rows` are
    /// the input embeddings, one step per row. Full banks only.
    pub fn feed(&mut self, rows: &[f32]) -> bool {
        let n = rows.len() / self.hidden;
        if n == 0 || self.pos + n > self.cap || self.arena.is_some() || !self.flights.is_empty()
        {
            return false;
        }
        // one submit per row: the host row is written into `h` before each
        for t in 0..n {
            self.c.queue.write_buffer(
                &self.s.h,
                0,
                bytemuck::cast_slice(&rows[t * self.hidden..(t + 1) * self.hidden]),
            );
            self.write_meta(self.pos + t, None);
            let mut enc = self
                .c
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nm-feed"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nm-feed"),
                    timestamp_writes: None,
                });
                self.encode_ops(&mut pass, 0, self.body_end, self.pos + t);
            }
            self.c.queue.submit(Some(enc.finish()));
        }
        self.pos += n;
        true
    }

    pub fn max_parts(&self) -> usize {
        self.max_parts
    }

    fn prefill_build(&self) -> Option<Prefill> {
        use super::nemotron_prefill::{PB_MAX, PB_WGSL, PQ_RW};
        let c = self.c;
        let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cmf-nemotron-prefill"),
            source: wgpu::ShaderSource::Wgsl(PB_WGSL.into()),
        });
        let pipe = |ep: &str| {
            c.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(ep),
                    layout: None,
                    module: &module,
                    entry_point: Some(ep),
                    compilation_options: pco(),
                    cache: c.pipeline_cache.as_ref(),
                })
        };
        let pp = PfPipes {
            add_rms: pipe("pb_add_rms"),
            f32mm: pipe("pb_f32_mm"),
            conv: pipe("pb_conv"),
            scan: pipe("pb_scan"),
            gnorm: pipe("pb_gnorm"),
            kv: pipe("pb_kv_append"),
            attn: pipe("pb_attn"),
            merge: pipe("pb_attn_merge"),
            route: pipe("pb_route"),
            lists: pipe("pb_lists"),
            bank: pipe("pb_q4_bank"),
            bank8: pipe("pb_q8_bank"),
            relu2: pipe("pb_relu2"),
            argmax: pipe("pb_argmax"),
        };
        if let Some(e) = pollster::block_on(scope.pop()) {
            tracing::warn!(
                "nemotron prefill: module rejected ({e}); the frame feeds position by position"
            );
            return None;
        }
        let pp: &'static PfPipes = Box::leak(Box::new(pp));
        let b = PB_MAX;
        let zeros = |label: &str, n: usize| -> wgpu::Buffer {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n.max(4) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let inner = self.m_heads * self.m_head_dim;
        let cdim = inner + 2 * self.m_groups * self.m_state;
        let in_rows = inner + cdim + self.m_heads;
        let kvw = self.nkv * self.hd;
        let (hid, k) = (self.hidden, self.top_k);
        let cap = b * k;
        let maxsplit = 16usize;
        let pf = PfBufs {
            h: zeros("pb-h", b * hid),
            o: zeros("pb-o", b * hid),
            xn: zeros("pb-xn", b * hid),
            zx: zeros("pb-zx", b * in_rows),
            xbc: zeros("pb-xbc", b * cdim),
            y: zeros("pb-y", b * inner.max(self.nh * self.hd)),
            yn: zeros("pb-yn", b * inner),
            q: zeros("pb-q", b * self.nh * self.hd),
            k: zeros("pb-k", b * kvw),
            v: zeros("pb-v", b * kvw),
            att: zeros("pb-att", b * self.nh * self.hd),
            part: zeros("pb-part", b * maxsplit * 8 * self.nh * (self.hd + 2)),
            rl: zeros("pb-rl", b * self.n_exp),
            sel: zeros("pb-sel", b * k),
            sw: zeros("pb-sw", b * k),
            list: zeros("pb-list", self.n_exp * cap),
            cnt: zeros("pb-cnt", self.n_exp),
            mid: zeros("pb-mid", cap * self.moe_inter),
            eo: zeros("pb-eo", cap * hid),
            smid: zeros("pb-smid", b * self.shared_inter),
            so: zeros("pb-so", b * hid),
            meta: zeros("pb-meta", 4),
            dummy: zeros("pb-dummy", 4),
            logits: zeros("pb-logits", b * self.vocab),
            ids: zeros("pb-ids", b),
        };
        // per Mamba block: conv rings and scan states after each of a verify
        // pass's first VR_MAX − 1 positions
        let state_n = self.m_heads * self.m_head_dim * self.m_state;
        let snaps: Vec<(wgpu::Buffer, wgpu::Buffer)> = self
            .blocks
            .iter()
            .filter(|b| matches!(b, DevBlock::Mamba { .. }))
            .map(|_| {
                let mk = |label: &str, n: usize| {
                    c.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: (n * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                };
                (mk("pb-snap-ring", (VR_MAX - 1) * 3 * cdim), mk("pb-snap-ssm", (VR_MAX - 1) * state_n))
            })
            .collect();
        let eps = self.eps.to_bits();
        let mut plan: Vec<Op> = Vec::new();
        let mut lists_at = Vec::new();
        let fixed = |tag: &'static str,
                     pipe: &'static wgpu::ComputePipeline,
                     bufs: &[&wgpu::Buffer],
                     u: [u32; 8],
                     groups: (u32, u32)|
         -> Op {
            let ub = uniform_u32x8(c, u);
            Op::Fixed {
                pipe,
                bg: self.bind(pipe, bufs, &ub),
                groups,
                tag,
            }
        };
        // q8_2f projections: the batch matmat of `gemma_batch` (64 rows ×
        // 64 positions a workgroup, the activations staged once per tile),
        // always over PB_MAX positions — rows past the pass's count are
        // computed and never read.
        let gb = gemma_batch::pipes(c)?;
        let (mm_pipe, mm_layout) = gb.mm.as_ref().map(|m| (&m.0, &m.1))?;
        let q82_one = |tag: &'static str,
                       w: &wgpu::Buffer,
                       x: &wgpu::Buffer,
                       y: &wgpu::Buffer,
                       rows: usize,
                       cols: usize|
         -> Op {
            let u = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, b as u32, cols as u32]);
            let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pb-q82"),
                layout: mm_layout,
                entries: &[
                    bind_buf(0, w),
                    bind_buf(1, x),
                    bind_buf(2, y),
                    bind_buf(3, &u),
                ],
            });
            Op::Fixed {
                pipe: mm_pipe,
                bg,
                groups: (rows.div_ceil(64) as u32, 1),
                tag,
            }
        };
        if [hid, inner, self.shared_inter, self.nh * self.hd]
            .iter()
            .any(|&c| c % 64 != 0)
        {
            return None;
        }
        let mut mi = 0usize;
        for (bi, blk) in self.blocks.iter().enumerate() {
            let norm = match blk {
                DevBlock::Mamba { norm, .. }
                | DevBlock::Attn { norm, .. }
                | DevBlock::Moe { norm, .. } => norm,
            };
            let prev_moe = bi > 0 && matches!(self.blocks[bi - 1], DevBlock::Moe { .. });
            let add = if bi == 0 {
                0
            } else if prev_moe {
                2
            } else {
                1
            };
            let o_src = match add {
                0 => &pf.dummy,
                2 => &pf.so,
                _ => &pf.o,
            };
            plan.push(fixed(
                "rms",
                &pp.add_rms,
                &[&pf.h, o_src, norm, &pf.xn, &pf.eo, &pf.sw, &pf.meta],
                [hid as u32, add, eps, k as u32, 0, 0, 0, 0],
                (b as u32, 1),
            ));
            match blk {
                DevBlock::Mamba {
                    in_proj,
                    out_proj,
                    conv_w,
                    conv_b,
                    par,
                    gnorm,
                    ring,
                    state,
                    ..
                } => {
                    plan.push(q82_one("in_proj", in_proj, &pf.xn, &pf.zx, in_rows, hid));
                    plan.push(fixed(
                        "conv",
                        &pp.conv,
                        &[&pf.zx, conv_w, conv_b, ring, &pf.xbc, &pf.meta, &snaps[mi].0],
                        [cdim as u32, inner as u32, in_rows as u32, 0, 0, 0, 0, 0],
                        (cdim.div_ceil(256) as u32, 1),
                    ));
                    plan.push(fixed(
                        "scan",
                        &pp.scan,
                        &[&pf.zx, &pf.xbc, state, par, &pf.y, &pf.meta, &snaps[mi].1],
                        [
                            inner as u32,
                            (inner + cdim) as u32,
                            self.m_groups as u32,
                            self.m_state as u32,
                            (self.m_heads / self.m_groups) as u32,
                            self.m_head_dim as u32,
                            self.m_heads as u32,
                            in_rows as u32,
                        ],
                        (self.m_heads as u32, 1),
                    ));
                    plan.push(fixed(
                        "gnorm",
                        &pp.gnorm,
                        &[&pf.y, &pf.zx, gnorm, &pf.yn, &pf.meta],
                        [
                            (inner / self.m_groups) as u32,
                            eps,
                            inner as u32,
                            in_rows as u32,
                            0,
                            0,
                            0,
                            0,
                        ],
                        (self.m_groups as u32, b as u32),
                    ));
                    plan.push(q82_one("out_proj", out_proj, &pf.yn, &pf.o, hid, inner));
                    mi += 1;
                }
                DevBlock::Attn {
                    q,
                    k: kw,
                    v,
                    o,
                    kc,
                    vc,
                    ..
                } => {
                    plan.push(q82_one("qkv", q, &pf.xn, &pf.q, self.nh * self.hd, hid));
                    plan.push(q82_one("qkv", kw, &pf.xn, &pf.k, kvw, hid));
                    plan.push(q82_one("qkv", v, &pf.xn, &pf.v, kvw, hid));
                    plan.push(fixed(
                        "attn",
                        &pp.kv,
                        &[&pf.k, &pf.v, kc, vc, &pf.meta],
                        [kvw as u32, 0, 0, 0, 0, 0, 0, 0],
                        ((b * kvw).div_ceil(256) as u32, 1),
                    ));
                    plan.push(Op::Attn { block: bi });
                    plan.push(q82_one("o_proj", o, &pf.att, &pf.o, hid, self.nh * self.hd));
                }
                DevBlock::Moe {
                    router,
                    bias,
                    up,
                    up_stride,
                    down,
                    down_stride,
                    sh_up,
                    sh_down,
                    moe,
                    ..
                } => {
                    plan.push(fixed(
                        "router",
                        &pp.f32mm,
                        &[router, &pf.xn, &pf.rl, &pf.meta],
                        [self.n_exp as u32, hid as u32, 0, 0, 0, 0, 0, 0],
                        (self.n_exp.div_ceil(8) as u32, 1),
                    ));
                    plan.push(fixed(
                        "route",
                        &pp.route,
                        &[&pf.rl, bias, &pf.sel, &pf.sw, &pf.meta],
                        [
                            self.n_exp as u32,
                            k as u32,
                            self.route_scale.to_bits(),
                            self.norm_topk as u32,
                            0,
                            0,
                            0,
                            0,
                        ],
                        (b as u32, 1),
                    ));
                    lists_at.push(plan.len());
                    plan.push(fixed(
                        "route",
                        &pp.lists,
                        &[&pf.sel, &pf.list, &pf.cnt, &pf.meta],
                        [k as u32, cap as u32, 0, 0, 0, 0, 0, 0],
                        (self.n_exp.div_ceil(64) as u32, 1),
                    ));
                    let (bank, q8) = match self.codec {
                        ExpertCodec::Q4tp => (&pp.bank, false),
                        ExpertCodec::Q8 => (&pp.bank8, true),
                    };
                    let remap = self.arena.is_some() as u32;
                    plan.push(fixed(
                        "exp_up",
                        bank,
                        &[up, up, &pf.xn, &pf.list, &pf.cnt, &pf.mid, &self.s.map],
                        [
                            self.moe_inter as u32,
                            if q8 { hid } else { hid / 32 } as u32,
                            *up_stride as u32,
                            1,
                            k as u32,
                            cap as u32,
                            *moe as u32,
                            remap,
                        ],
                        (self.moe_inter.div_ceil(8 * PQ_RW) as u32, self.n_exp as u32),
                    ));
                    plan.push(fixed(
                        "exp_down",
                        bank,
                        &[down, down, &pf.mid, &pf.list, &pf.cnt, &pf.eo, &self.s.map],
                        [
                            hid as u32,
                            if q8 { self.moe_inter } else { self.moe_inter / 32 } as u32,
                            *down_stride as u32,
                            0,
                            1,
                            cap as u32,
                            *moe as u32,
                            remap,
                        ],
                        (hid.div_ceil(8 * PQ_RW) as u32, self.n_exp as u32),
                    ));
                    plan.push(q82_one(
                        "sh_up",
                        sh_up,
                        &pf.xn,
                        &pf.smid,
                        self.shared_inter,
                        hid,
                    ));
                    let nr = b * self.shared_inter;
                    plan.push(fixed(
                        "relu2",
                        &pp.relu2,
                        &[&pf.smid],
                        [nr as u32, 0, 0, 0, 0, 0, 0, 0],
                        (nr.div_ceil(256) as u32, 1),
                    ));
                    plan.push(q82_one(
                        "sh_down",
                        sh_down,
                        &pf.smid,
                        &pf.so,
                        hid,
                        self.shared_inter,
                    ));
                }
            }
        }
        // the MoE sum of the last block (the residual itself is not needed:
        // the pass only advances the conv rings, scan states and KV rows)
        // verify tail: the last block's sum + final norm, head, argmax
        let last_moe = matches!(self.blocks.last(), Some(DevBlock::Moe { .. }));
        let tail = vec![
            fixed(
                "final",
                &pp.add_rms,
                &[&pf.h, if last_moe { &pf.so } else { &pf.o }, &self.final_norm, &pf.xn, &pf.eo, &pf.sw, &pf.meta],
                [hid as u32, if last_moe { 2 } else { 1 }, eps, k as u32, 0, 0, 0, 0],
                (b as u32, 1),
            ),
            q82_one("head", &self.lm_head, &pf.xn, &pf.logits, self.vocab, hid),
            fixed(
                "argmax",
                &pp.argmax,
                &[&pf.logits, &pf.ids, &pf.meta],
                [self.vocab as u32, 0, 0, 0, 0, 0, 0, 0],
                (b as u32, 1),
            ),
        ];
        let vstage = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pb-vstage"),
            size: ((VR_MAX + 2 * VR_MAX * hid) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stage = self.arena.is_some().then(|| {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pb-stage"),
                size: ((self.n_exp + self.n_exp * cap + b * hid) * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Some(Prefill {
            pp,
            pf,
            plan,
            maxsplit,
            lists_at,
            stage,
            tail,
            snaps,
            vstage,
            draft_rows: zeros("pb-draft-rows", b * hid),
            draft_pos: zeros("pb-draft-pos", b),
        })
    }

    /// A verify pass: `rows` (≤ VR_MAX input embeddings) at `self.pos` in
    /// one batched pass, every position's argmax and its normed and
    /// residual final rows `[n · hidden]` read back. The state then holds
    /// all n positions; [`Self::accept`] keeps a prefix. Full banks only.
    pub fn verify(&mut self, rows: &[f32]) -> Option<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        let hid = self.hidden;
        let n = rows.len() / hid;
        if n == 0 || n > VR_MAX || self.pos + n > self.cap || self.arena.is_some() || !self.flights.is_empty() {
            return None;
        }
        if self.pf.is_none() {
            self.pf = Some(self.prefill_build()?);
        }
        let c = self.c;
        let pos0 = self.pos;
        let pfx = self.pf.as_ref().unwrap();
        c.queue
            .write_buffer(&pfx.pf.meta, 0, bytemuck::cast_slice(&[n as u32, pos0 as u32, 1, 0]));
        c.queue.write_buffer(&pfx.pf.h, 0, bytemuck::cast_slice(rows));
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pb-verify") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pb-verify"),
                timestamp_writes: None,
            });
            self.encode_pf_ops(&mut pass, 0, pfx.plan.len(), pos0, n);
            for op in &pfx.tail {
                if let Op::Fixed { pipe, bg, groups, .. } = op {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(groups.0, groups.1, 1);
                }
            }
        }
        let st = &pfx.vstage;
        enc.copy_buffer_to_buffer(&pfx.pf.ids, 0, st, 0, (n * 4) as u64);
        enc.copy_buffer_to_buffer(&pfx.pf.xn, 0, st, (VR_MAX * 4) as u64, (n * hid * 4) as u64);
        enc.copy_buffer_to_buffer(
            &pfx.pf.h,
            0,
            st,
            ((VR_MAX + VR_MAX * hid) * 4) as u64,
            (n * hid * 4) as u64,
        );
        c.queue.submit(Some(enc.finish()));
        let slice = st.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let words: Vec<u32> = {
            let d = slice.get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&d).to_vec()
        };
        st.unmap();
        let f = |a: usize| -> Vec<f32> { words[a..a + n * hid].iter().map(|w| f32::from_bits(*w)).collect() };
        self.pending_verify = n;
        Some((words[..n].to_vec(), f(VR_MAX), f(VR_MAX + VR_MAX * hid)))
    }

    /// Keep the first `keep` positions of the last verify pass (rolling the
    /// Mamba states back to the snapshot after position `keep − 1` when
    /// fewer than all were kept); KV rows past them are overwritten later.
    pub fn accept(&mut self, keep: usize) -> bool {
        let n = std::mem::take(&mut self.pending_verify);
        if keep == 0 || keep > n {
            return false;
        }
        if keep < n {
            let pfx = self.pf.as_ref().unwrap();
            let state_n = self.m_heads * self.m_head_dim * self.m_state;
            let cdim = self.m_heads * self.m_head_dim + 2 * self.m_groups * self.m_state;
            let mut enc = self
                .c
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pb-rollback") });
            let mut mi = 0;
            for b in &self.blocks {
                if let DevBlock::Mamba { ring, state, .. } = b {
                    let (sr, ss) = &pfx.snaps[mi];
                    let j = keep - 1;
                    enc.copy_buffer_to_buffer(sr, (j * 3 * cdim * 4) as u64, ring, 0, (3 * cdim * 4) as u64);
                    enc.copy_buffer_to_buffer(ss, (j * state_n * 4) as u64, state, 0, (state_n * 4) as u64);
                    mi += 1;
                }
            }
            self.c.queue.submit(Some(enc.finish()));
        }
        self.pos += keep;
        true
    }

    /// Prefill `rows` (input embeddings) through the batched kernels,
    /// `PB_MAX` positions per pass. With a partial expert bank every MoE
    /// layer of a pass stops after its expert lists: `complete(moe, expert,
    /// rows, n)` returns the missing expert's outputs `[n · hidden]` for its
    /// normed input rows, written where the bank would have. False when the
    /// batched path is unavailable (nothing was encoded) or the device failed.
    ///
    /// With `draft` (the draft layer on, full banks) every pass also runs
    /// the draft layer over its positions — the history its attention
    /// reads during speculation; `next` is the input row after `rows`.
    #[allow(clippy::type_complexity)]
    pub fn feed_batched(
        &mut self,
        rows: &[f32],
        mut complete: Option<&mut dyn FnMut(usize, usize, &[f32], usize) -> Vec<f32>>,
        draft: bool,
        next: Option<&[f32]>,
    ) -> bool {
        use super::nemotron_prefill::PB_MAX;
        let hid = self.hidden;
        let n_all = rows.len() / hid;
        if n_all == 0
            || self.pos + n_all > self.cap
            || !self.flights.is_empty()
            || (self.arena.is_some() && complete.is_none())
        {
            return false;
        }
        if self.pf.is_none() {
            match self.prefill_build() {
                Some(p) => self.pf = Some(p),
                None => return false,
            }
        }
        let t0 = std::time::Instant::now();
        let c = self.c;
        let draft = draft && self.mtp.is_some() && self.arena.is_none();
        if draft {
            // the bank kernels of the draft layer read the meta words
            self.write_meta(self.pos, None);
        }
        let mut done = 0;
        let mut host_rows = 0usize;
        while done < n_all {
            let n = (n_all - done).min(PB_MAX);
            let pos0 = self.pos + done;
            let pfx = self.pf.as_ref().unwrap();
            c.queue.write_buffer(
                &pfx.pf.meta,
                0,
                bytemuck::cast_slice(&[n as u32, pos0 as u32, 0, 0]),
            );
            c.queue.write_buffer(
                &pfx.pf.h,
                0,
                bytemuck::cast_slice(&rows[done * hid..(done + n) * hid]),
            );
            let cuts: Vec<usize> = if self.arena.is_some() {
                pfx.lists_at.iter().map(|&i| i + 1).collect()
            } else {
                Vec::new()
            };
            let mut from = 0usize;
            for seg in 0..=cuts.len() {
                let to = cuts.get(seg).copied().unwrap_or(pfx.plan.len());
                let pfx = self.pf.as_ref().unwrap();
                let mut enc = c
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pb") });
                let pts = (done == PB_MAX && cuts.is_empty() && std::env::var("CMF_NEMO_TS").as_deref() == Ok("1"))
                    .then(|| {
                        c.device.create_query_set(&wgpu::QuerySetDescriptor {
                            label: Some("pb-ts"),
                            ty: wgpu::QueryType::Timestamp,
                            count: pfx.plan.len() as u32 + 1,
                        })
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("pb"),
                        timestamp_writes: None,
                    });
                    self.encode_pf_ops_ts(&mut pass, from, to, pos0, n, pts.as_ref());
                    if draft && seg == cuts.len() {
                        // the pass's final normed rows (the draft layer's input)
                        if let Some(Op::Fixed { pipe, bg, groups, .. }) = pfx.tail.first() {
                            pass.set_pipeline(pipe);
                            pass.set_bind_group(0, bg, &[]);
                            pass.dispatch_workgroups(groups.0, groups.1, 1);
                        }
                    }
                }
                if draft && seg == cuts.len() {
                    self.encode_draft_history(&mut enc, rows, next, done, n, pos0);
                }
                if seg == cuts.len() {
                    if let Some(q) = &pts {
                        self.print_pf_ts(enc, q);
                    } else {
                        c.queue.submit(Some(enc.finish()));
                    }
                    break;
                }
                // a partial bank: read the layer's lists and rows back
                let stage = pfx.stage.as_ref().unwrap();
                let (ne, cap) = (self.n_exp, PB_MAX * self.top_k);
                enc.copy_buffer_to_buffer(&pfx.pf.cnt, 0, stage, 0, (ne * 4) as u64);
                enc.copy_buffer_to_buffer(&pfx.pf.list, 0, stage, (ne * 4) as u64, (ne * cap * 4) as u64);
                enc.copy_buffer_to_buffer(
                    &pfx.pf.xn,
                    0,
                    stage,
                    ((ne + ne * cap) * 4) as u64,
                    (n * hid * 4) as u64,
                );
                c.queue.submit(Some(enc.finish()));
                let slice = stage.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
                    return false;
                }
                let words: Vec<u32> = match slice.get_mapped_range() {
                    Ok(d) => bytemuck::cast_slice::<u8, u32>(&d).to_vec(),
                    Err(_) => return false,
                };
                stage.unmap();
                let moe = seg;
                let k = self.top_k;
                let xs: Vec<f32> = words[ne + ne * cap..ne + ne * cap + n * hid]
                    .iter()
                    .map(|w| f32::from_bits(*w))
                    .collect();
                let a = self.arena.as_mut().unwrap();
                for e in 0..ne {
                    let cnt = words[e] as usize;
                    a.score[moe * ne + e] += cnt as f32;
                    if cnt == 0 || a.map[moe * ne + e] != MISS {
                        continue;
                    }
                    let ents = &words[ne + e * cap..ne + e * cap + cnt];
                    let mut x = Vec::with_capacity(cnt * hid);
                    for &j in ents {
                        let p = j as usize / k;
                        x.extend_from_slice(&xs[p * hid..(p + 1) * hid]);
                    }
                    let y = (complete.as_mut().unwrap())(moe, e, &x, cnt);
                    if y.len() != cnt * hid {
                        return false;
                    }
                    let pf = &self.pf.as_ref().unwrap().pf;
                    for (i, &j) in ents.iter().enumerate() {
                        c.queue.write_buffer(
                            &pf.eo,
                            (j as usize * hid * 4) as u64,
                            bytemuck::cast_slice(&y[i * hid..(i + 1) * hid]),
                        );
                    }
                    a.misses += cnt as u64;
                    host_rows += cnt;
                }
                from = to;
            }
            done += n;
        }
        if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return false;
        }
        if prof() {
            eprintln!(
                "nemotron prefill: {n_all} positions in {} passes, {:.1} ms ({:.2} ms/position), {host_rows} expert rows on the host",
                n_all.div_ceil(PB_MAX),
                t0.elapsed().as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3 / n_all as f64
            );
        }
        self.pos += n_all;
        if let Some(a) = self.arena.as_mut() {
            // rebalance the banks toward the prompt's experts
            a.admit_mark = self.pos / 4;
            self.admit_n(2 * self.n_moe);
        }
        true
    }

    /// Submit a profiled prefill pass and print its per-class device time.
    fn print_pf_ts(&self, mut enc: wgpu::CommandEncoder, q: &wgpu::QuerySet) {
        let c = self.c;
        let pfx = self.pf.as_ref().unwrap();
        let n = pfx.plan.len() as u64 + 1;
        let resolve = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.resolve_query_set(q, 0..n as u32, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, n * 8);
        c.queue.submit(Some(enc.finish()));
        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        let Ok(data) = slice.get_mapped_range() else {
            return;
        };
        let t: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&data).to_vec();
        drop(data);
        let period = c.queue.get_timestamp_period() as f64;
        let mut by: Vec<(&str, f64, usize)> = Vec::new();
        for (i, op) in pfx.plan.iter().enumerate() {
            let tag = match op {
                Op::Fixed { tag, .. } => *tag,
                _ => "attn",
            };
            let us = t[i + 1].saturating_sub(t[i]) as f64 * period / 1e3;
            match by.iter_mut().find(|b| b.0 == tag) {
                Some(b) => {
                    b.1 += us;
                    b.2 += 1;
                }
                None => by.push((tag, us, 1)),
            }
        }
        let total: f64 = by.iter().map(|b| b.1).sum();
        let line: Vec<String> = by.iter().map(|(t, us, n)| format!("{t} {:.0}us/{n}", us)).collect();
        eprintln!("nemotron prefill pass: {:.0} us = {}", total, line.join(", "));
    }

    /// The draft layer over a prefill pass's positions (`n` from row
    /// `first` of `rows`, at `pos0`): each position fuses its final normed
    /// row with the next input row and extends the draft KV.
    fn encode_draft_history(
        &self,
        enc: &mut wgpu::CommandEncoder,
        rows: &[f32],
        next: Option<&[f32]>,
        first: usize,
        n: usize,
        pos0: usize,
    ) {
        let (Some(m), Some(pfx)) = (self.mtp.as_ref(), self.pf.as_ref()) else {
            return;
        };
        let hid = self.hidden;
        let total = rows.len() / hid;
        let c = self.c;
        // the next rows of the pass's positions, and their positions
        let mut nx = Vec::with_capacity(n * hid);
        let mut ps = Vec::with_capacity(n);
        for t in 0..n {
            let j = first + t + 1;
            let r = if j < total {
                &rows[j * hid..(j + 1) * hid]
            } else if let Some(nr) = next {
                nr
            } else {
                break;
            };
            nx.extend_from_slice(r);
            ps.push((pos0 + t) as u32);
        }
        c.queue.write_buffer(&pfx.draft_rows, 0, bytemuck::cast_slice(&nx));
        c.queue.write_buffer(&pfx.draft_pos, 0, bytemuck::cast_slice(&ps));
        let rb = (hid * 4) as u64;
        for (t, &p) in ps.iter().enumerate() {
            enc.copy_buffer_to_buffer(&pfx.pf.xn, t as u64 * rb, &m.hin, 0, rb);
            enc.copy_buffer_to_buffer(&pfx.draft_rows, t as u64 * rb, &m.e, 0, rb);
            enc.copy_buffer_to_buffer(&pfx.draft_pos, t as u64 * 4, &self.s.meta, 0, 4);
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nm-mtp-history"),
                timestamp_writes: None,
            });
            self.encode_mtp_ops(&mut pass, 0, self.mtp_head_at, p as usize);
        }
    }

    /// Encode prefill plan ops `[a, b)` of the pass at `pos0` with `n` positions.
    fn encode_pf_ops(&self, pass: &mut wgpu::ComputePass, a: usize, b: usize, pos0: usize, n: usize) {
        self.encode_pf_ops_ts(pass, a, b, pos0, n, None)
    }

    fn encode_pf_ops_ts(
        &self,
        pass: &mut wgpu::ComputePass,
        a: usize,
        b: usize,
        pos0: usize,
        n: usize,
        ts: Option<&wgpu::QuerySet>,
    ) {
        let pfx = self.pf.as_ref().unwrap();
        let skip = skip_tags();
        if let Some(q) = ts {
            pass.write_timestamp(q, 0);
        }
        for (i, op) in pfx.plan[a..b].iter().enumerate() {
            if let Some(q) = ts
                && i > 0
            {
                pass.write_timestamp(q, i as u32);
            }
            match op {
                Op::Fixed { tag, .. } if skip.contains(tag) => {}
                Op::Fixed {
                    pipe, bg, groups, ..
                } => {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(groups.0, groups.1, 1);
                }
                Op::MtpAttn => {}
                Op::Attn { block } => {
                    let DevBlock::Attn { kc, vc, .. } = &self.blocks[*block] else {
                        unreachable!()
                    };
                    let npos = pos0 + n;
                    let chunk = npos.div_ceil(pfx.maxsplit).max(64);
                    let splits = npos.div_ceil(chunk);
                    let scale = (1.0 / (self.hd as f32).sqrt()).to_bits();
                    let pf = &pfx.pf;
                    let ub = self.u([
                        self.nh as u32,
                        self.nkv as u32,
                        self.hd as u32,
                        chunk as u32,
                        scale,
                        pfx.maxsplit as u32,
                        0,
                        0,
                    ]);
                    let bg = self.bind(&pfx.pp.attn, &[&pf.q, kc, vc, &pf.part, &pf.meta], &ub);
                    pass.set_pipeline(&pfx.pp.attn);
                    pass.set_bind_group(0, &bg, &[]);
                    // grid z: the pass's positions
                    pass.dispatch_workgroups(splits as u32, (self.nh / 4) as u32, n as u32);
                    self.run(
                        pass,
                        &pfx.pp.merge,
                        &[&pf.part, &pf.att, &pf.meta],
                        [
                            self.nh as u32,
                            self.hd as u32,
                            splits as u32,
                            pfx.maxsplit as u32,
                            0,
                            0,
                            0,
                            0,
                        ],
                        (self.nh as u32, n as u32),
                    );
                }
            }
        }
        if let Some(q) = ts {
            pass.write_timestamp(q, (b - a) as u32);
        }
    }
}

/// The two-position verify pass (`nemotron_verify`): pipelines, rows of
/// both positions, per-Mamba-block snapshots after the first position.
struct Verify {
    plan: Vec<Op>,
    tail: Vec<Op>,
    attn: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    h: wgpu::Buffer,
    xn: wgpu::Buffer,
    q: wgpu::Buffer,
    att: wgpu::Buffer,
    part: wgpu::Buffer,
    ids: wgpu::Buffer,
    meta: wgpu::Buffer,
    stage: wgpu::Buffer,
    /// per Mamba block (ring, scan state) after the first position
    snaps: Vec<(wgpu::Buffer, wgpu::Buffer)>,
}

impl NemoFrame {
    fn verify_build(&self) -> Option<Verify> {
        use super::nemotron_verify::{NR, ver_wgsl};
        let c = self.c;
        if self.arena.is_some() {
            return None;
        }
        let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cmf-nemotron-verify"),
            source: wgpu::ShaderSource::Wgsl(ver_wgsl().into()),
        });
        let pipe = |ep: &str| -> &'static wgpu::ComputePipeline {
            Box::leak(Box::new(c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: pco(),
                cache: c.pipeline_cache.as_ref(),
            })))
        };
        let (q8p, scanp, bank4, bank8, rmsp, axp) = (
            pipe("vf_q8"),
            pipe("vf_scan"),
            pipe("vf_bank4"),
            pipe("vf_bank8"),
            pipe("vf_rms"),
            pipe("vf_argmax"),
        );
        if let Some(e) = pollster::block_on(scope.pop()) {
            tracing::warn!("nemotron verify: module rejected ({e})");
            return None;
        }
        let zeros = |label: &str, n: usize| -> wgpu::Buffer {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n.max(64) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let hid = self.hidden;
        let inner = self.m_heads * self.m_head_dim;
        let cdim = inner + 2 * self.m_groups * self.m_state;
        let in_rows = inner + cdim + self.m_heads;
        let kvw = self.nkv * self.hd;
        let k = self.top_k;
        let state_n = self.m_heads * self.m_head_dim * self.m_state;
        let h = zeros("vf-h", NR * hid);
        let xn = zeros("vf-xn", NR * hid);
        let zx = zeros("vf-zx", NR * in_rows);
        let y = zeros("vf-y", NR * inner);
        let q = zeros("vf-q", NR * self.nh * self.hd);
        let att = zeros("vf-att", NR * self.nh * self.hd);
        let part = zeros("vf-part", self.max_parts * self.nh * (self.hd + 2));
        let rl = zeros("vf-rl", NR * self.n_exp);
        let sel = zeros("vf-sel", NR * k);
        let sw = zeros("vf-sw", NR * k);
        let mid = zeros("vf-mid", NR * k * self.moe_inter);
        let eo = zeros("vf-eo", NR * k * hid);
        let smid = zeros("vf-smid", NR * self.shared_inter);
        let logits = zeros("vf-logits", NR * self.vocab);
        let ids = zeros("vf-ids", NR);
        let meta = zeros("vf-meta", 8);
        let (ro, rw) = (zeros("vf-ro", 64), zeros("vf-rw", 64));
        let snaps: Vec<(wgpu::Buffer, wgpu::Buffer)> = self
            .blocks
            .iter()
            .filter(|b| matches!(b, DevBlock::Mamba { .. }))
            .map(|_| (zeros("vf-snap-ring", 3 * cdim), zeros("vf-snap-ssm", state_n)))
            .collect();
        let eps = self.eps.to_bits();
        // one vf_q8 dispatch (same argument shape as `nq`, plus the ring
        // snapshot and the per-position zx stride)
        let vq = |tag: &'static str, a: Nq, snap: Option<&wgpu::Buffer>| -> Op {
            let rows = |i: usize| a.mats.get(i).map_or(0, |m| m.2);
            let rpw: usize = if rows(0).max(rows(1)).max(rows(2)) >= 32768 { 4 } else { 2 };
            let frows = a.f.map_or(0, |f| f.2);
            let groups = (0..3).map(|i| rows(i).div_ceil(8 * rpw)).sum::<usize>() + frows.div_ceil(8);
            let w = |i: usize| a.mats.get(i).map_or(&ro, |m| m.0);
            let yb = |i: usize| a.mats.get(i).and_then(|m| m.1).unwrap_or(&rw);
            let (kc, vc) = a.kv.unwrap_or((&rw, &rw));
            let (ring, lo, hi, n, off) = a.ring.unwrap_or((&rw, 0, 0, 0, 0));
            let u = [
                rows(0) as u32,
                rows(1) as u32,
                rows(2) as u32,
                a.cols as u32,
                rpw as u32,
                a.pro,
                a.epi,
                eps,
                a.gs as u32,
                frows as u32,
                k as u32,
                a.kvw as u32,
                lo as u32,
                hi as u32,
                n as u32,
                off as u32,
                a.xo.is_some() as u32,
                in_rows as u32,
                0,
                0,
            ];
            let bufs: [&wgpu::Buffer; 20] = [
                w(0),
                w(1),
                w(2),
                a.x.unwrap_or(&ro),
                a.nw.unwrap_or(&ro),
                a.zx.unwrap_or(&ro),
                yb(0),
                yb(1),
                yb(2),
                a.h.unwrap_or(&rw),
                a.xo.unwrap_or(&rw),
                a.eo.unwrap_or(&ro),
                a.sw.unwrap_or(&ro),
                a.f.map_or(&ro, |f| f.0),
                a.f.map_or(&rw, |f| f.1),
                &meta,
                kc,
                vc,
                ring,
                snap.unwrap_or(&rw),
            ];
            let ub = uwords(c, &u);
            Op::Fixed {
                pipe: q8p,
                bg: self.bind(q8p, &bufs, &ub),
                groups: (groups as u32, 1),
                tag,
            }
        };
        let fixed = |tag: &'static str, pipe: &'static wgpu::ComputePipeline, bufs: &[&wgpu::Buffer], u: &[u32], groups: (u32, u32)| -> Op {
            let ub = uwords(c, u);
            Op::Fixed {
                pipe,
                bg: self.bind(pipe, bufs, &ub),
                groups,
                tag,
            }
        };
        let bank = |tag: &'static str, w: &wgpu::Buffer, stride: usize, x: &wgpu::Buffer, yv: &wgpu::Buffer, rows: usize, cols: usize, up: Option<&wgpu::Buffer>| -> Op {
            let route = up.is_some();
            let (pipe, gpr) = match self.codec {
                ExpertCodec::Q4tp => (bank4, cols / 32),
                ExpertCodec::Q8 => (bank8, cols),
            };
            let u = [
                rows as u32,
                gpr as u32,
                stride as u32,
                route as u32,
                cols as u32,
                rows as u32,
                route as u32,
                self.n_exp as u32,
                k as u32,
                self.route_scale.to_bits(),
                self.norm_topk as u32,
                if route { k } else { 1 } as u32,
                cols as u32,
                0,
                0,
                0,
            ];
            fixed(
                tag,
                pipe,
                &[w, w, x, if route { &rl } else { &ro }, up.unwrap_or(&ro), &sel, &sw, yv],
                &u,
                (rows.div_ceil(32) as u32, (NR * k) as u32),
            )
        };
        let mut plan = Vec::new();
        let mut mi = 0usize;
        for (bi, b) in self.blocks.iter().enumerate() {
            match b {
                DevBlock::Mamba { norm, in_proj, out_proj, conv_w, conv_b, par, gnorm, ring, state } => {
                    plan.push(vq(
                        "in_proj",
                        Nq { mats: vec![(in_proj, Some(&zx), in_rows)], cols: hid, x: Some(&h), pro: 1, nw: Some(norm), ..Default::default() },
                        None,
                    ));
                    plan.push(fixed(
                        "scan",
                        scanp,
                        &[&zx, conv_w, conv_b, ring, state, par, &y, &snaps[mi].0, &snaps[mi].1],
                        &[
                            inner as u32,
                            (inner + cdim) as u32,
                            self.m_groups as u32,
                            self.m_state as u32,
                            (self.m_heads / self.m_groups) as u32,
                            self.m_head_dim as u32,
                            self.m_heads as u32,
                            cdim as u32,
                            in_rows as u32,
                            0,
                            0,
                            0,
                        ],
                        (self.m_heads as u32, 1),
                    ));
                    plan.push(vq(
                        "out_proj",
                        Nq {
                            mats: vec![(out_proj, None, hid)],
                            cols: inner,
                            x: Some(&y),
                            pro: 2,
                            nw: Some(gnorm),
                            zx: Some(&zx),
                            gs: inner / self.m_groups,
                            epi: 2,
                            h: Some(&h),
                            ring: Some((ring, inner, cdim, cdim, inner)),
                            ..Default::default()
                        },
                        Some(&snaps[mi].0),
                    ));
                    mi += 1;
                }
                DevBlock::Attn { norm, q: wq, k: wk, v: wv, o, kc, vc } => {
                    plan.push(vq(
                        "qkv",
                        Nq {
                            mats: vec![(wq, Some(&q), self.nh * self.hd), (wk, None, kvw), (wv, None, kvw)],
                            cols: hid,
                            x: Some(&h),
                            pro: 1,
                            nw: Some(norm),
                            epi: 4,
                            kv: Some((kc, vc)),
                            kvw,
                            ..Default::default()
                        },
                        None,
                    ));
                    plan.push(Op::Attn { block: bi });
                    plan.push(vq(
                        "o_proj",
                        Nq { mats: vec![(o, None, hid)], cols: self.nh * self.hd, x: Some(&att), epi: 2, h: Some(&h), ..Default::default() },
                        None,
                    ));
                }
                DevBlock::Moe { norm, router, bias, up, up_stride, down, down_stride, sh_up, sh_down, .. } => {
                    plan.push(vq(
                        "router",
                        Nq {
                            mats: vec![(sh_up, Some(&smid), self.shared_inter)],
                            cols: hid,
                            x: Some(&h),
                            pro: 1,
                            nw: Some(norm),
                            epi: 1,
                            xo: Some(&xn),
                            f: Some((router, &rl, self.n_exp)),
                            ..Default::default()
                        },
                        None,
                    ));
                    plan.push(bank("exp_up", up, *up_stride, &xn, &mid, self.moe_inter, hid, Some(bias)));
                    plan.push(bank("exp_down", down, *down_stride, &mid, &eo, hid, self.moe_inter, None));
                    plan.push(vq(
                        "sh_down",
                        Nq {
                            mats: vec![(sh_down, None, hid)],
                            cols: self.shared_inter,
                            x: Some(&smid),
                            epi: 3,
                            h: Some(&h),
                            eo: Some(&eo),
                            sw: Some(&sw),
                            ..Default::default()
                        },
                        None,
                    ));
                }
            }
        }
        let tail = vec![
            fixed("final", rmsp, &[&h, &self.final_norm, &xn], &[hid as u32, eps, 0, 0], (NR as u32, 1)),
            vq("head", Nq { mats: vec![(&self.lm_head, Some(&logits), self.vocab)], cols: hid, x: Some(&xn), ..Default::default() }, None),
            fixed("argmax", axp, &[&logits, &ids], &[self.vocab as u32, 0, 0, 0], (NR as u32, 1)),
        ];
        let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vf-stage"),
            size: ((NR + NR * hid) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Verify {
            plan,
            tail,
            attn: self.p.attn.clone(),
            merge: self.p.attn_merge.clone(),
            h,
            xn,
            q,
            att,
            part,
            ids,
            meta,
            stage,
            snaps,
        })
    }

    /// The two-position verify pass at `self.pos`: `rows` = the token's and
    /// the draft's embeddings. Returns both argmaxes and both final normed
    /// rows; the state then holds both positions until [`Self::accept2`].
    pub fn verify2(&mut self, rows: &[f32]) -> Option<(Vec<u32>, Vec<f32>)> {
        use super::nemotron_verify::NR;
        let hid = self.hidden;
        if rows.len() != NR * hid || self.pos + NR > self.cap || !self.flights.is_empty() {
            return None;
        }
        if self.vf.is_none() {
            self.vf = Some(self.verify_build()?);
        }
        let c = self.c;
        let pos = self.pos;
        let v = self.vf.as_ref().unwrap();
        c.queue.write_buffer(&v.h, 0, bytemuck::cast_slice(rows));
        c.queue.write_buffer(&v.meta, 0, bytemuck::cast_slice(&[pos as u32, 0, 0, 0]));
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vf") });
        let vts = (std::env::var("CMF_NEMO_TS").as_deref() == Ok("1") && pos % 64 == 0).then(|| {
            c.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("vf-ts"),
                ty: wgpu::QueryType::Timestamp,
                count: (v.plan.len() + v.tail.len() + 1) as u32,
            })
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vf"),
                timestamp_writes: None,
            });
            let qb = (self.nh * self.hd * 4) as u64;
            if let Some(q) = &vts {
                pass.write_timestamp(q, 0);
            }
            for (oi, op) in v.plan.iter().chain(&v.tail).enumerate() {
                if let Some(q) = &vts
                    && oi > 0
                {
                    pass.write_timestamp(q, oi as u32);
                }
                match op {
                    Op::Fixed { pipe, bg, groups, .. } => {
                        pass.set_pipeline(pipe);
                        pass.set_bind_group(0, bg, &[]);
                        pass.dispatch_workgroups(groups.0, groups.1, 1);
                    }
                    Op::Attn { block } => {
                        let DevBlock::Attn { kc, vc, .. } = &self.blocks[*block] else {
                            unreachable!()
                        };
                        for r in 0..NR {
                            let npos = pos + r + 1;
                            let chunk = npos.div_ceil(ATTN_SPLITS).max(64);
                            let splits = npos.div_ceil(chunk);
                            let scale = (1.0 / (self.hd as f32).sqrt()).to_bits();
                            let ub = self.u([self.nh as u32, self.nkv as u32, self.hd as u32, npos as u32, chunk as u32, scale, 0, 0]);
                            let layout = v.attn.get_bind_group_layout(0);
                            let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: &layout,
                                entries: &[
                                    bind_buf_off(0, &v.q, r as u64 * qb, qb),
                                    bind_buf(1, kc),
                                    bind_buf(2, vc),
                                    bind_buf(3, &v.part),
                                    bind_buf(4, &ub),
                                ],
                            });
                            pass.set_pipeline(&v.attn);
                            pass.set_bind_group(0, &bg, &[]);
                            pass.dispatch_workgroups(splits as u32, self.nkv as u32, (self.nh / self.nkv / 4) as u32);
                            let ub = self.u([self.nh as u32, self.hd as u32, (splits * 8) as u32, 0, 0, 0, 0, 0]);
                            let layout = v.merge.get_bind_group_layout(0);
                            let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: &layout,
                                entries: &[
                                    bind_buf(0, &v.part),
                                    bind_buf_off(1, &v.att, r as u64 * qb, qb),
                                    bind_buf(2, &ub),
                                ],
                            });
                            pass.set_pipeline(&v.merge);
                            pass.set_bind_group(0, &bg, &[]);
                            pass.dispatch_workgroups(self.nh as u32, 1, 1);
                        }
                    }
                    Op::MtpAttn => {}
                }
            }
            if let Some(q) = &vts {
                pass.write_timestamp(q, (v.plan.len() + v.tail.len()) as u32);
            }
        }
        enc.copy_buffer_to_buffer(&v.ids, 0, &v.stage, 0, (NR * 4) as u64);
        enc.copy_buffer_to_buffer(&v.xn, 0, &v.stage, (NR * 4) as u64, (NR * hid * 4) as u64);
        let vtsr = vts.as_ref().map(|q| {
            let n = (v.plan.len() + v.tail.len() + 1) as u64;
            let resolve = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: n * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: n * 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.resolve_query_set(q, 0..n as u32, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, n * 8);
            read
        });
        c.queue.submit(Some(enc.finish()));
        if let Some(read) = &vtsr {
            let sl = read.slice(..);
            sl.map_async(wgpu::MapMode::Read, |_| {});
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            if let Ok(d) = sl.get_mapped_range() {
                let t: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&d).to_vec();
                let period = c.queue.get_timestamp_period() as f64;
                let mut by: Vec<(&str, f64, usize)> = Vec::new();
                for (i, op) in v.plan.iter().chain(&v.tail).enumerate() {
                    let tag = match op {
                        Op::Fixed { tag, .. } => *tag,
                        _ => "attn",
                    };
                    let us = t[i + 1].saturating_sub(t[i]) as f64 * period / 1e3;
                    match by.iter_mut().find(|b| b.0 == tag) {
                        Some(b) => {
                            b.1 += us;
                            b.2 += 1;
                        }
                        None => by.push((tag, us, 1)),
                    }
                }
                let total: f64 = by.iter().map(|b| b.1).sum();
                let line: Vec<String> = by.iter().map(|(t, us, n)| format!("{t} {:.0}us/{n}", us)).collect();
                eprintln!("nemotron verify pass at {pos}: {:.0} us = {}", total, line.join(", "));
            }
        }
        let slice = v.stage.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let words: Vec<u32> = {
            let d = slice.get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&d).to_vec()
        };
        v.stage.unmap();
        self.pending_verify = NR;
        Some((words[..NR].to_vec(), words[NR..].iter().map(|w| f32::from_bits(*w)).collect()))
    }

    /// True when the two-position verify pass is built.
    pub fn verify_ready(&self) -> bool {
        self.vf.is_some()
    }

    /// Build the two-position verify pass now (pipelines compile at load,
    /// not in the first transcription's decode).
    pub fn prepare_verify(&mut self) -> bool {
        if self.vf.is_none() {
            self.vf = self.verify_build();
        }
        self.vf.is_some()
    }

    /// One decode step from `row` at `self.pos`: its argmax and final
    /// normed row (the draft layer's input).
    pub fn step_hidden(&mut self, row: &[f32]) -> Option<(u32, Vec<f32>)> {
        let hid = self.hidden;
        if row.len() != hid || self.pos + 1 > self.cap || !self.flights.is_empty() || self.arena.is_some() {
            return None;
        }
        let c = self.c;
        c.queue.write_buffer(&self.s.h, 0, bytemuck::cast_slice(row));
        self.write_meta(self.pos, None);
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nm-step1") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nm-step1"),
                timestamp_writes: None,
            });
            self.encode_ops(&mut pass, 0, self.plan.len(), self.pos);
        }
        let st = &self.s.stage[0];
        enc.copy_buffer_to_buffer(&self.s.ids, 0, st, 0, 4);
        enc.copy_buffer_to_buffer(&self.s.xn, 0, st, 4 * 16, (hid * 4) as u64);
        c.queue.submit(Some(enc.finish()));
        let slice = st.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let words: Vec<u32> = {
            let d = slice.get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&d).to_vec()
        };
        st.unmap();
        self.pos += 1;
        Some((words[0], words[16..16 + hid].iter().map(|w| f32::from_bits(*w)).collect()))
    }

    /// Keep 1 or 2 positions of the last two-position pass (1: the Mamba
    /// states roll back to the first position's snapshot).
    pub fn accept2(&mut self, keep: usize) -> bool {
        let n = std::mem::take(&mut self.pending_verify);
        if keep == 0 || keep > n || self.vf.is_none() {
            return false;
        }
        if keep < n {
            let v = self.vf.as_ref().unwrap();
            let mut enc = self
                .c
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vf-rollback") });
            let mut mi = 0;
            for b in &self.blocks {
                if let DevBlock::Mamba { ring, state, .. } = b {
                    let (sr, ss) = &v.snaps[mi];
                    enc.copy_buffer_to_buffer(sr, 0, ring, 0, ring.size());
                    enc.copy_buffer_to_buffer(ss, 0, state, 0, state.size());
                    mi += 1;
                }
            }
            self.c.queue.submit(Some(enc.finish()));
        }
        self.pos += keep;
        true
    }
}

struct PfPipes {
    add_rms: wgpu::ComputePipeline,
    f32mm: wgpu::ComputePipeline,
    conv: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    gnorm: wgpu::ComputePipeline,
    kv: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
    merge: wgpu::ComputePipeline,
    route: wgpu::ComputePipeline,
    lists: wgpu::ComputePipeline,
    bank: wgpu::ComputePipeline,
    bank8: wgpu::ComputePipeline,
    relu2: wgpu::ComputePipeline,
    argmax: wgpu::ComputePipeline,
}

struct PfBufs {
    h: wgpu::Buffer,
    o: wgpu::Buffer,
    xn: wgpu::Buffer,
    zx: wgpu::Buffer,
    xbc: wgpu::Buffer,
    y: wgpu::Buffer,
    yn: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    att: wgpu::Buffer,
    part: wgpu::Buffer,
    rl: wgpu::Buffer,
    sel: wgpu::Buffer,
    sw: wgpu::Buffer,
    list: wgpu::Buffer,
    cnt: wgpu::Buffer,
    mid: wgpu::Buffer,
    eo: wgpu::Buffer,
    smid: wgpu::Buffer,
    so: wgpu::Buffer,
    meta: wgpu::Buffer,
    dummy: wgpu::Buffer,
    /// verify passes: every position's logits and argmax
    logits: wgpu::Buffer,
    ids: wgpu::Buffer,
}

struct Prefill {
    pp: &'static PfPipes,
    pf: PfBufs,
    plan: Vec<Op>,
    maxsplit: usize,
    /// plan index of each MoE layer's expert lists (a partial bank syncs
    /// there: the host completes the missing experts)
    lists_at: Vec<usize>,
    /// readback of cnt, lists and the normed rows (partial banks)
    stage: Option<wgpu::Buffer>,
    /// final norm, head and argmax of every position (verify passes)
    tail: Vec<Op>,
    /// per Mamba block (ring, scan state) snapshots of a verify pass
    snaps: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    /// verify readback: ids, normed rows, residual rows
    vstage: wgpu::Buffer,
    /// the draft history: next input rows and positions of a pass
    draft_rows: wgpu::Buffer,
    draft_pos: wgpu::Buffer,
}
