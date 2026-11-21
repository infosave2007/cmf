//! Maxwell's block-sparse attention on the wgpu graphs (`GraphBlockRoute`):
//! the CUDA hybrid arm's `hy_route_*` kernels and the host's
//! `maxwell::route_blocks` / `Attn::forward` are its specification.
//!
//! A query attends to the first `prefix` blocks, to its own block and the
//! `local - 1` before it, and to the best `remote` blocks between those by
//! the block's largest token score of the indexer (each KV group its own
//! choice; ties to the lower block). While the blocks in between do not
//! outnumber `remote`, that is every block up to the query's own: the dense
//! range.
//!
//! The graph's Full arm (the per-layer-geometry one) projects q / k / v; everything after is
//! here, the layer's mirrors included (f16 keys and values of the KV groups and the index key,
//! as the CUDA arm keeps them: 104 KB a position over 45 layers):
//!
//! * `br_rope_b`: the rows' per-head norms and rotation, the queries into the call's `q`, the
//!   keys and values into the mirrors;
//! * `br_proj1` / `br_proj`: the indexer's projections of the rows (`[nkv·hd + hd]` a row: the
//!   queries of the KV groups, then the key) over the normed rows;
//! * `br_put`: their norms and rotation, the queries into `iq`, the key into the index-key
//!   mirror;
//! * `br_score` / `br_sel`: past the dense range, the candidates' scores; every (row, group)'s
//!   block list, ascending (inside the dense range every block up to the row's own);
//! * `br_part` / `br_merge`: the attention over the listed blocks, in parts of 64 positions a
//!   workgroup for a token and 256 for a prompt chunk, merged a head a row into `attn`.

use super::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Blocks a (row, group) attends to at most: `prefix + local + remote`.
const LIST_BLOCKS: usize = 32;
/// Positions a part workgroup takes: a prompt chunk's, and a token's (a shorter chain of value
/// loads a thread, four times the workgroups).
const PART: usize = 256;
const PART_TOKEN: usize = 64;
/// Rows a prompt call takes (the batch graph's largest chunk).
const ROWS_MAX: usize = 1024;

const SRC: &str = r#"
struct RP {
    hidden: u32, hd: u32, nkv: u32, nh: u32,
    pos0: u32, rows: u32, cap: u32, block: u32,
    prefix: u32, local: u32, remote: u32, nblk: u32,
    eps: f32, scale: f32, nch: u32, nparts: u32,
    plen: u32, used: u32, pad1: u32, pad2: u32,
};
@group(0) @binding(0) var<uniform> p: RP;

// Does the row at `pos` pick its remote blocks (past the dense range)?
fn picks(pos: u32) -> bool {
    let cur = pos / p.block;
    let local0 = select(0u, cur + 1u - p.local, cur + 1u > p.local);
    return local0 > p.prefix + p.remote;
}

// ---- br_proj1: raw[row][nch] = W[nch][hidden] . x[row] for a few rows: 4 channels a
// workgroup, 64 threads a channel, every row in turn
@group(0) @binding(1) var<storage, read> pj_x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pj_w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pj_raw: array<f32>;
var<workgroup> pr: array<f32, 256>;
@compute @workgroup_size(256)
fn br_proj1(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let ch = w.x * 4u + t / 64u;
    let lane = t % 64u;
    let nq = p.nkv * p.hd;
    let n = p.hidden / 4u;
    let live = ch < p.nch;
    for (var row = 0u; row < p.rows; row = row + 1u) {
        // a row inside the dense range makes no index query (uniform across the workgroup:
        // its four channels are all queries or all the key)
        let skip = !live || (ch < nq && !picks(p.pos0 + row));
        var acc = vec4<f32>(0.0);
        if (!skip) {
            for (var c = lane; c < n; c = c + 64u) { acc = acc + pj_w[ch * n + c] * pj_x[row * n + c]; }
        }
        pr[t] = acc.x + acc.y + acc.z + acc.w;
        workgroupBarrier();
        var st = 32u;
        loop {
            if (st == 0u) { break; }
            if (lane < st) { pr[t] = pr[t] + pr[t + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        if (lane == 0u && !skip) { pj_raw[row * p.nch + ch] = pr[t]; }
        workgroupBarrier();
    }
}

// ---- br_proj: the same for a prompt chunk, a 16 x 16 tile of (channel, row) a workgroup over
// 64-column steps staged in workgroup memory
var<workgroup> tw: array<vec4<f32>, 256>;
var<workgroup> tx: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16)
fn br_proj(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let ch0 = w.x * 16u;
    let row0 = w.y * 16u;
    let ch = ch0 + l.x;
    let row = row0 + l.y;
    let nq = p.nkv * p.hd;
    let n = p.hidden / 4u;
    // a tile of queries whose rows are all inside the dense range has nothing to do
    let last = min(row0 + 15u, p.rows - 1u);
    if (ch0 + 15u < nq && !picks(p.pos0 + last)) { return; }
    var acc = 0.0;
    for (var c0 = 0u; c0 < n; c0 = c0 + 16u) {
        // thread (x, y) stages vec4 x of channel row y and of input row y
        let wc = ch0 + l.y;
        let xr = row0 + l.y;
        tw[l.y * 16u + l.x] = select(vec4<f32>(0.0), pj_w[wc * n + c0 + l.x], wc < p.nch && c0 + l.x < n);
        tx[l.y * 16u + l.x] = select(vec4<f32>(0.0), pj_x[xr * n + c0 + l.x], xr < p.rows && c0 + l.x < n);
        workgroupBarrier();
        var a4 = vec4<f32>(0.0);
        for (var e = 0u; e < 16u; e = e + 1u) { a4 = a4 + tw[l.x * 16u + e] * tx[l.y * 16u + e]; }
        acc = acc + a4.x + a4.y + a4.z + a4.w;
        workgroupBarrier();
    }
    if (ch < p.nch && row < p.rows && (ch >= nq || picks(p.pos0 + row))) { pj_raw[row * p.nch + ch] = acc; }
}

// The rotated element `e` of the normed head in `xs` at `pos` (NeoX half-split).
fn rot(e: u32, pos: u32, half: u32) -> f32 {
    if (e < half) {
        let a = f32(pos) * pt_invf[e];
        return xs[e] * cos(a) - xs[e + half] * sin(a);
    }
    let a = f32(pos) * pt_invf[e - half];
    return xs[e - half] * sin(a) + xs[e] * cos(a);
}

// ---- br_put: a head's norm and rotation (workgroup = hd threads); head u < nkv the index
// query of KV group u into iq[row][u][hd], head nkv the index key into ik[pos][hd / 2] (f16 pairs)
@group(0) @binding(4) var<storage, read> pt_raw: array<f32>;
@group(0) @binding(5) var<storage, read> pt_qn: array<f32>;
@group(0) @binding(6) var<storage, read> pt_kn: array<f32>;
@group(0) @binding(7) var<storage, read> pt_invf: array<f32>;
@group(0) @binding(8) var<storage, read_write> pt_iq: array<f32>;
@group(0) @binding(9) var<storage, read_write> pt_ik: array<u32>;
var<workgroup> xs: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(128)
fn br_put(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let u = w.x;
    let row = w.y;
    if (row >= p.rows) { return; }
    let pos = p.pos0 + row;
    let isq = u < p.nkv;
    if (isq && !picks(pos)) { return; }
    let hd = p.hd;
    var x = 0.0;
    if (t < hd) { x = pt_raw[row * p.nch + u * hd + t]; }
    red[t] = x * x;
    workgroupBarrier();
    var st = 64u;
    loop {
        if (st == 0u) { break; }
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let inv = 1.0 / sqrt(red[0] / f32(hd) + p.eps);
    if (t < hd) {
        let wn = select(pt_kn[t], pt_qn[t], isq);
        xs[t] = x * inv * wn;
    }
    workgroupBarrier();
    let half = hd / 2u;
    if (isq) {
        if (t < hd) { pt_iq[(row * p.nkv + u) * hd + t] = rot(t, pos, half); }
    } else if (t < half) {
        pt_ik[pos * half + t] = pack2x16float(vec2<f32>(rot(2u * t, pos, half), rot(2u * t + 1u, pos, half)));
    }
}

// ---- br_score: candidate block `prefix + w.x` of row w.y for every KV group: its largest
// token score (a thread a token) into sc[row][group][nblk]
@group(0) @binding(10) var<storage, read> sc_iq: array<f32>;
@group(0) @binding(11) var<storage, read> sc_ik: array<u32>;
@group(0) @binding(12) var<storage, read_write> sc_out: array<f32>;
var<workgroup> qsh: array<f32, 1024>;
var<workgroup> mx: array<f32, 1024>;
@compute @workgroup_size(128)
fn br_score(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let row = w.y;
    if (row >= p.rows) { return; }
    let pos = p.pos0 + row;
    let cb = p.prefix + w.x;
    let cur = pos / p.block;
    let local0 = select(0u, cur + 1u - p.local, cur + 1u > p.local);
    if (!picks(pos) || cb >= local0) { return; }
    let hd = p.hd;
    for (var i = t; i < p.nkv * hd; i = i + 128u) { qsh[i] = sc_iq[row * p.nkv * hd + i]; }
    workgroupBarrier();
    for (var g = 0u; g < p.nkv; g = g + 1u) {
        var m = -3.0e38;
        for (var tok = t; tok < p.block; tok = tok + 128u) {
            let kb = (cb * p.block + tok) * (hd / 2u);
            var a = 0.0;
            for (var d = 0u; d < hd / 2u; d = d + 1u) {
                let kk = unpack2x16float(sc_ik[kb + d]);
                a = a + qsh[g * hd + 2u * d] * kk.x + qsh[g * hd + 2u * d + 1u] * kk.y;
            }
            m = max(m, a);
        }
        mx[g * 128u + t] = m;
    }
    workgroupBarrier();
    var st = 64u;
    loop {
        if (st == 0u) { break; }
        if (t < st) {
            for (var g = 0u; g < p.nkv; g = g + 1u) { mx[g * 128u + t] = max(mx[g * 128u + t], mx[g * 128u + t + st]); }
        }
        workgroupBarrier();
        st = st >> 1u;
    }
    if (t < p.nkv) { sc_out[(row * p.nkv + t) * p.nblk + cb] = mx[t * 128u]; }
}

// ---- br_sel: the blocks a (row, group) attends to, ascending: bl[(row * nkv + g) * 33] = the
// count, then the blocks
@group(0) @binding(13) var<storage, read> sl_sc: array<f32>;
@group(0) @binding(14) var<storage, read_write> sl_bl: array<u32>;
@compute @workgroup_size(1)
fn br_sel(@builtin(workgroup_id) w: vec3<u32>) {
    let g = w.x;
    let row = w.y;
    if (row >= p.rows || g >= p.nkv) { return; }
    let pos = p.pos0 + row;
    let cur = pos / p.block;
    let o = (row * p.nkv + g) * 33u;
    if (!picks(pos)) {
        // the dense range: every block up to the row's own
        for (var b = 0u; b <= cur; b = b + 1u) { sl_bl[o + 1u + b] = b; }
        sl_bl[o] = cur + 1u;
        return;
    }
    let local0 = cur + 1u - p.local;
    var vals: array<f32, 16>;
    var ids: array<u32, 16>;
    var cnt = 0u;
    let r = min(p.remote, 16u);
    let sb = (row * p.nkv + g) * p.nblk;
    for (var cb = p.prefix; cb < local0; cb = cb + 1u) {
        let m = sl_sc[sb + cb];
        if (cnt < r || m > vals[r - 1u]) {
            var j = min(cnt, r - 1u);
            loop {
                if (j == 0u) { break; }
                if (!(vals[j - 1u] < m)) { break; }
                vals[j] = vals[j - 1u];
                ids[j] = ids[j - 1u];
                j = j - 1u;
            }
            vals[j] = m;
            ids[j] = cb;
            cnt = min(cnt + 1u, r);
        }
    }
    // the picks ascending
    for (var i = 1u; i < cnt; i = i + 1u) {
        let v = ids[i];
        var j = i;
        loop {
            if (j == 0u) { break; }
            if (ids[j - 1u] <= v) { break; }
            ids[j] = ids[j - 1u];
            j = j - 1u;
        }
        ids[j] = v;
    }
    var n = 0u;
    for (var b = 0u; b < p.prefix; b = b + 1u) { sl_bl[o + 1u + n] = b; n = n + 1u; }
    for (var i = 0u; i < cnt; i = i + 1u) { sl_bl[o + 1u + n] = ids[i]; n = n + 1u; }
    for (var b = local0; b <= cur; b = b + 1u) { sl_bl[o + 1u + n] = b; n = n + 1u; }
    sl_bl[o] = n;
}

// ---- br_part: the heads of KV group w.x of row w.z over list positions [w.y * plen, + plen)
// (plen 64 for a token, 256 for a prompt chunk): each head's largest score in the part, its
// sum of exponentials and its weighted values. A key row is read once for the group's heads.
@group(0) @binding(15) var<storage, read> at_q: array<f32>;
@group(0) @binding(16) var<storage, read> at_k: array<u32>;
@group(0) @binding(17) var<storage, read> at_v: array<u32>;
@group(0) @binding(18) var<storage, read> at_bl: array<u32>;
@group(0) @binding(19) var<storage, read_write> at_acc: array<f32>;
@group(0) @binding(20) var<storage, read_write> at_ml: array<vec2<f32>>;
var<workgroup> wsc: array<f32, 2048>;
var<workgroup> wrow: array<u32, 256>;
var<workgroup> qv: array<vec4<f32>, 256>;
var<workgroup> hm: array<f32, 8>;
var<workgroup> hs: array<f32, 8>;
const NONE: u32 = 0xffffffffu;
@compute @workgroup_size(256)
fn br_part(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let g = w.x;
    let part = w.y;
    let row = w.z;
    if (row >= p.rows || g >= p.nkv) { return; }
    let pos = p.pos0 + row;
    let hd = p.hd;
    let lanes = hd / 4u;
    let hpk = p.nh / p.nkv;
    let o = (row * p.nkv + g) * 33u;
    let nb = at_bl[o];
    let h0 = g * hpk;
    let np = p.plen;
    if (part * np >= nb * p.block) {
        for (var i = t; i < hpk * hd; i = i + 256u) {
            let h = h0 + i / hd;
            let mi = (row * p.nh + h) * p.nparts + part;
            at_acc[mi * hd + i % hd] = 0.0;
            if (i % hd == 0u) { at_ml[mi] = vec2<f32>(-3.0e38, 0.0); }
        }
        return;
    }
    for (var i = t; i < hpk * lanes; i = i + 256u) {
        let h = h0 + i / lanes;
        let e = i % lanes;
        qv[i] = vec4<f32>(at_q[(row * p.nh + h) * hd + e * 4u], at_q[(row * p.nh + h) * hd + e * 4u + 1u],
                          at_q[(row * p.nh + h) * hd + e * 4u + 2u], at_q[(row * p.nh + h) * hd + e * 4u + 3u]);
    }
    // this thread's position of the part, and the heads it scores there
    let ip = t % np;
    let li = part * np + ip;
    let slot = li / p.block;
    var r = NONE;
    if (slot < nb) {
        let pp = at_bl[o + 1u + slot] * p.block + li % p.block;
        if (pp <= pos) { r = pp; }
    }
    if (t < np) { wrow[ip] = r; }
    workgroupBarrier();
    let hstep = 256u / np;
    if (r != NONE) {
        let kb = (g * p.cap + r) * (hd / 2u);
        if (np == 256u) {
            // a thread a position, the group's eight heads at once: a key read once
            var a8: array<vec4<f32>, 8>;
            for (var h = 0u; h < 8u; h = h + 1u) { a8[h] = vec4<f32>(0.0); }
            for (var e = 0u; e < lanes; e = e + 1u) {
                let k0 = unpack2x16float(at_k[kb + 2u * e]);
                let k1 = unpack2x16float(at_k[kb + 2u * e + 1u]);
                let kv = vec4<f32>(k0.x, k0.y, k1.x, k1.y);
                for (var h = 0u; h < 8u; h = h + 1u) { a8[h] = a8[h] + qv[(h % hpk) * lanes + e] * kv; }
            }
            for (var h = 0u; h < 8u; h = h + 1u) {
                if (h < hpk) { wsc[h * np + ip] = (a8[h].x + a8[h].y + a8[h].z + a8[h].w) * p.scale; }
            }
        } else {
            for (var h = t / np; h < hpk; h = h + hstep) {
                var a = vec4<f32>(0.0);
                for (var e = 0u; e < lanes; e = e + 1u) {
                    let k0 = unpack2x16float(at_k[kb + 2u * e]);
                    let k1 = unpack2x16float(at_k[kb + 2u * e + 1u]);
                    a = a + qv[h * lanes + e] * vec4<f32>(k0.x, k0.y, k1.x, k1.y);
                }
                wsc[h * np + ip] = (a.x + a.y + a.z + a.w) * p.scale;
            }
        }
    }
    workgroupBarrier();
    // each head's largest score and sum: 32 threads a head
    if (t < hpk * 32u) {
        let h = t / 32u;
        let ln = t % 32u;
        var m = -3.0e38;
        for (var i = ln; i < np; i = i + 32u) { if (wrow[i] != NONE) { m = max(m, wsc[h * np + i]); } }
        pr[t] = m;
    }
    workgroupBarrier();
    if (t < hpk) {
        var m = -3.0e38;
        for (var i = 0u; i < 32u; i = i + 1u) { m = max(m, pr[t * 32u + i]); }
        hm[t] = m;
    }
    workgroupBarrier();
    for (var i = t; i < hpk * np; i = i + 256u) {
        let h = i / np;
        let ii = i % np;
        if (wrow[ii] != NONE) { wsc[i] = exp(wsc[i] - hm[h]); } else { wsc[i] = 0.0; }
    }
    workgroupBarrier();
    if (t < hpk * 32u) {
        let h = t / 32u;
        let ln = t % 32u;
        var sm = 0.0;
        for (var i = ln; i < np; i = i + 32u) { sm = sm + wsc[h * np + i]; }
        pr[t] = sm;
    }
    workgroupBarrier();
    if (t < hpk) {
        var sm = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sm = sm + pr[t * 32u + i]; }
        hs[t] = sm;
    }
    workgroupBarrier();
    // values: eight positions' rows at a time staged in workgroup memory (the queries' room,
    // done with), a thread a (head, four dims) of the group
    let hh = t / 32u;
    let lane = t % 32u;
    var acc4 = vec4<f32>(0.0);
    for (var t0 = 0u; t0 < np; t0 = t0 + 8u) {
        workgroupBarrier();
        let rr = wrow[t0 + t / 32u];
        var vv = vec4<f32>(0.0);
        if (rr != NONE) {
            let vb = (g * p.cap + rr) * (hd / 2u) + 2u * lane;
            let a0 = unpack2x16float(at_v[vb]);
            let a1 = unpack2x16float(at_v[vb + 1u]);
            vv = vec4<f32>(a0.x, a0.y, a1.x, a1.y);
        }
        qv[t] = vv;
        workgroupBarrier();
        if (hh < hpk) {
            for (var j = 0u; j < 8u; j = j + 1u) { acc4 = acc4 + wsc[hh * np + t0 + j] * qv[j * 32u + lane]; }
        }
    }
    if (hh < hpk) {
        let mi = (row * p.nh + h0 + hh) * p.nparts + part;
        at_acc[mi * hd + 4u * lane] = acc4.x;
        at_acc[mi * hd + 4u * lane + 1u] = acc4.y;
        at_acc[mi * hd + 4u * lane + 2u] = acc4.z;
        at_acc[mi * hd + 4u * lane + 3u] = acc4.w;
        if (lane == 0u) { at_ml[mi] = vec2<f32>(hm[hh], hs[hh]); }
    }
}

// ---- br_merge: a head of a row from its parts into attn[row][nh][hd]
@group(0) @binding(21) var<storage, read> mg_acc: array<f32>;
@group(0) @binding(22) var<storage, read> mg_ml: array<vec2<f32>>;
@group(0) @binding(23) var<storage, read_write> mg_out: array<f32>;
@compute @workgroup_size(128)
fn br_merge(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let h = w.x;
    let row = w.y;
    if (row >= p.rows || h >= p.nh || t >= p.hd) { return; }
    let b = (row * p.nh + h) * p.nparts;
    var m = -3.0e38;
    for (var c = 0u; c < p.used; c = c + 1u) { m = max(m, mg_ml[b + c].x); }
    var l = 0.0;
    var a = 0.0;
    for (var c = 0u; c < p.used; c = c + 1u) {
        let ml = mg_ml[b + c];
        if (ml.y > 0.0) {
            let f = exp(ml.x - m);
            l = l + ml.y * f;
            a = a + mg_acc[(b + c) * p.hd + t] * f;
        }
    }
    mg_out[(row * p.nh + h) * p.hd + t] = a / l;
}

// ---- br_rope_b: the call's heads (w.x < nh: a query head, else a KV head) normed and rotated by
// their row's position; queries into rq_q[row][nh][hd] (f32), keys and values into the mirrors
// [group][cap][hd / 2] (f16 pairs)
@group(0) @binding(24) var<storage, read> rq_qraw: array<f32>;
@group(0) @binding(25) var<storage, read> rq_kraw: array<f32>;
@group(0) @binding(26) var<storage, read> rq_vraw: array<f32>;
@group(0) @binding(27) var<storage, read> rq_qn: array<f32>;
@group(0) @binding(28) var<storage, read> rq_kn: array<f32>;
@group(0) @binding(30) var<storage, read_write> rq_q: array<f32>;
@group(0) @binding(31) var<storage, read_write> rq_kc: array<u32>;
@group(0) @binding(32) var<storage, read_write> rq_vc: array<u32>;
@compute @workgroup_size(128)
fn br_rope_b(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let u = w.x;
    let row = w.y;
    if (row >= p.rows) { return; }
    let pos = p.pos0 + row;
    let hd = p.hd;
    let isq = u < p.nh;
    let g = u - select(p.nh, 0u, isq);
    var x = 0.0;
    if (t < hd) {
        if (isq) { x = rq_qraw[(row * p.nh + u) * hd + t]; } else { x = rq_kraw[(row * p.nkv + g) * hd + t]; }
    }
    red[t] = x * x;
    workgroupBarrier();
    var st = 64u;
    loop {
        if (st == 0u) { break; }
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let inv = 1.0 / sqrt(red[0] / f32(hd) + p.eps);
    if (t < hd) {
        let wn = select(rq_kn[t], rq_qn[t], isq);
        xs[t] = x * inv * wn;
    }
    workgroupBarrier();
    let half = hd / 2u;
    if (isq) {
        if (t < hd) { rq_q[(row * p.nh + u) * hd + t] = rot(t, pos, half); }
    } else if (t < half) {
        let at = (g * p.cap + pos) * half + t;
        rq_kc[at] = pack2x16float(vec2<f32>(rot(2u * t, pos, half), rot(2u * t + 1u, pos, half)));
        let vb = (row * p.nkv + g) * hd + 2u * t;
        rq_vc[at] = pack2x16float(vec2<f32>(rq_vraw[vb], rq_vraw[vb + 1u]));
    }
}
"#;
struct Pipe {
    p: wgpu::ComputePipeline,
    l: wgpu::BindGroupLayout,
}

struct Pipes {
    proj1: Pipe,
    proj: Pipe,
    put: Pipe,
    score: Pipe,
    sel: Pipe,
    part: Pipe,
    merge: Pipe,
    rope_b: Pipe,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("broute"), source: wgpu::ShaderSource::Wgsl(SRC.into()) });
            let mk = |entry: &str| {
                let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &m,
                    entry_point: Some(entry),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                });
                let l = p.get_bind_group_layout(0);
                Pipe { p, l }
            };
            let out = Pipes {
                proj1: mk("br_proj1"),
                proj: mk("br_proj"),
                put: mk("br_put"),
                score: mk("br_score"),
                sel: mk("br_sel"),
                part: mk("br_part"),
                merge: mk("br_merge"),
                rope_b: mk("br_rope_b"),
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("block-route kernels rejected: {e}");
                return None;
            }
            Some(out)
        })
        .as_ref()
}

/// One layer's router on the device: its weights, the attention's q / k norms and the RoPE
/// table, and the bind groups over the call's buffers while those stay the same.
struct LayerDev {
    /// `[nkv·hd + hd][hidden]`: the index queries' rows, then the key's.
    w: wgpu::Buffer,
    qn: wgpu::Buffer,
    kn: wgpu::Buffer,
    invf: wgpu::Buffer,
    aqn: wgpu::Buffer,
    akn: wgpu::Buffer,
    bgs: Option<(Vec<wgpu::Buffer>, Bgs)>,
}

struct Bgs {
    rope: wgpu::BindGroup,
    proj1: wgpu::BindGroup,
    proj: wgpu::BindGroup,
    put: wgpu::BindGroup,
    score: wgpu::BindGroup,
    sel: wgpu::BindGroup,
    part: wgpu::BindGroup,
    merge: wgpu::BindGroup,
}

/// A sequence's rows of one layer: the keys and values of the KV groups `[nkv][cap][hd]` and
/// the index keys `[cap][hd]`, f16.
struct Rows {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    ik: wgpu::Buffer,
    cap: usize,
    /// Positions written (the next call starts at most here).
    rows: usize,
}

/// Scratch shared by the layers (one call's rows).
struct Scratch {
    raw: wgpu::Buffer,
    iq: wgpu::Buffer,
    sc: wgpu::Buffer,
    bl: wgpu::Buffer,
    acc: wgpu::Buffer,
    ml: wgpu::Buffer,
    /// rows, nh, hd, nkv, nblk it was sized for
    dims: (usize, usize, usize, usize, usize),
    /// rows x parts its partials hold
    acc_rows: usize,
}

struct State {
    layers: HashMap<usize, LayerDev>,
    rows: HashMap<(u64, usize), Rows>,
    scratch: Option<Scratch>,
    /// The call's parameters: every layer of a call has the same, and a call is one submit.
    u: Option<wgpu::Buffer>,
    words: [u32; 20],
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn buf(c: &Ctx, label: &str, size: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(4) as u64, usage, mapped_at_creation: false })
}

fn stor_f32(c: &Ctx, label: &str, data: &[f32]) -> wgpu::Buffer {
    let b = buf(c, label, data.len() * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
    b
}

/// Refuse a geometry the kernels cannot hold, before any device state moves.
pub(crate) fn admits(r: &crate::gpu::GraphBlockRoute<'_>, nh: usize, nkv: usize, hd: usize) -> bool {
    let hidden = if hd > 0 { r.k.len() / hd } else { 0 };
    hd == 128
        && nkv > 0
        && nkv <= 8
        && nh % nkv == 0
        && nh / nkv <= 8
        && hidden % 4 == 0
        && r.block > 0
        && r.block <= PART
        && PART % r.block == 0
        && r.prefix > 0
        && r.local > 0
        && (1..=16).contains(&r.remote)
        && r.prefix + r.local + r.remote <= LIST_BLOCKS
        && r.q.len() == nkv * hd * hidden
        && r.k.len() == hd * hidden
        && r.q_norm.len() == hd
        && r.k_norm.len() == hd
}

/// Every row of a sequence (`gpu::graph_kv_reset`).
pub(crate) fn reset(kv_id: u64) {
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.rows.retain(|k, _| k.0 != kv_id);
    }
}

/// Bytes a position takes over `layers` layers: f16 keys and values of the KV groups and the
/// index key.
pub fn position_bytes(layers: usize, nkv: usize, hd: usize) -> u64 {
    (layers * (2 * nkv + 1) * hd * 2) as u64
}

/// Everything an encode needs from the graph.
pub(crate) struct Enc<'a> {
    pub r: &'a crate::gpu::GraphBlockRoute<'a>,
    /// (sequence, layer): the rows' key.
    pub key: (u64, usize),
    /// The model's layer (its router's weights are uploaded once).
    pub layer: usize,
    /// The rows' normed input `[rows][hidden]` f32.
    pub n1: &'a wgpu::Buffer,
    /// The rows' raw projections: queries `[rows][nh][hd]`, keys and values `[rows][nkv][hd]`.
    pub qraw: &'a wgpu::Buffer,
    pub kraw: &'a wgpu::Buffer,
    pub vraw: &'a wgpu::Buffer,
    /// The attention's own q / k norms.
    pub qn: &'a [f32],
    pub kn: &'a [f32],
    /// Where the rows' rotated queries go `[rows][nh][hd]` (scratch of the call).
    pub q: &'a wgpu::Buffer,
    /// The layer's RoPE table (`hd / 2` inverse frequencies).
    pub invf: &'a [f32],
    pub attn: &'a wgpu::Buffer,
    pub pos0: usize,
    pub rows: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub hidden: usize,
    pub scale: f32,
    pub eps: f32,
}

/// Does any row of `rows` at `pos0` pick its remote blocks?
pub(crate) fn any_picks(r: &crate::gpu::GraphBlockRoute<'_>, pos0: usize, rows: usize) -> bool {
    let last = pos0 + rows - 1;
    let cur = last / r.block;
    (cur + 1).saturating_sub(r.local) > r.prefix + r.remote
}

/// The mirrors of `key` with room for `need` positions: made, or grown (the rows so far copied
/// at the new stride, the card waited for so the old buffers leave it at once).
fn ensure_rows(c: &Ctx, s: &mut State, key: (u64, usize), nkv: usize, hd: usize, need: usize) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, usize, usize) {
    if let Some(m) = s.rows.get(&key).filter(|m| m.cap >= need) {
        return (m.k.clone(), m.v.clone(), m.ik.clone(), m.cap, m.rows);
    }
    let cap = super::kv_capacity(1 << 20, need).max(need);
    let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let k = buf(c, "broute-k", nkv * cap * hd * 2, st);
    let v = buf(c, "broute-v", nkv * cap * hd * 2, st);
    let ik = buf(c, "broute-ik", cap * hd * 2, st);
    let mut rows = 0;
    if let Some(old) = s.rows.remove(&key) {
        rows = old.rows.min(cap);
        if rows > 0 {
            let mut ce = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("broute-grow") });
            let rb = (rows * hd * 2) as u64;
            for g in 0..nkv {
                let (src, dst) = ((g * old.cap * hd * 2) as u64, (g * cap * hd * 2) as u64);
                ce.copy_buffer_to_buffer(&old.k, src, &k, dst, rb);
                ce.copy_buffer_to_buffer(&old.v, src, &v, dst, rb);
            }
            ce.copy_buffer_to_buffer(&old.ik, 0, &ik, 0, rb);
            c.queue.submit(Some(ce.finish()));
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
    s.rows.insert(key, Rows { k: k.clone(), v: v.clone(), ik: ik.clone(), cap, rows });
    (k, v, ik, cap, rows)
}

/// Encode a call's rows: their norms, rotation and mirror append, the index keys, the block
/// route and the attention into `attn`. None: the kernels are not available, or the call does
/// not continue the layer's rows (nothing was encoded).
pub(crate) fn encode(c: &Ctx, enc: &mut wgpu::CommandEncoder, e: &Enc<'_>) -> Option<()> {
    let p = pipes(c)?;
    let r = e.r;
    let (rows, nh, nkv, hd) = (e.rows, e.nh, e.nkv, e.hd);
    if rows == 0 || rows > ROWS_MAX || hd != 128 || e.invf.len() < hd / 2 || e.qn.len() != hd || e.kn.len() != hd {
        return None;
    }
    let nch = nkv * hd + hd;
    let plen = if rows <= 16 { PART_TOKEN } else { PART };
    let nparts = (LIST_BLOCKS * r.block).div_ceil(plen);
    // the parts the call's longest list fills: every block up to the last row's own, at most
    // LIST_BLOCKS of them
    let cur = (e.pos0 + rows - 1) / r.block;
    let used = ((cur + 1).min(LIST_BLOCKS) * r.block).div_ceil(plen).min(nparts);
    let st = wgpu::BufferUsages::STORAGE;
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(|| State { layers: HashMap::new(), rows: HashMap::new(), scratch: None, u: None, words: [u32::MAX; 20] });
    let (mk, mv, ik, cap, have) = ensure_rows(c, s, e.key, nkv, hd, e.pos0 + rows);
    // a call continues the layer's rows (or writes over the last ones again)
    if e.pos0 > have {
        return None;
    }
    if let Some(m) = s.rows.get_mut(&e.key) {
        m.rows = e.pos0 + rows;
    }
    let nblk = cap.div_ceil(r.block);
    let acc_rows = rows * nparts;
    let fits = s.scratch.as_ref().is_some_and(|x| x.dims.0 >= rows && x.dims.1 >= nh && x.dims.2 >= hd && x.dims.3 >= nkv && x.dims.4 >= nblk && x.acc_rows >= acc_rows);
    if !fits {
        let rr = rows.max(s.scratch.as_ref().map_or(0, |x| x.dims.0));
        let ar = acc_rows.max(s.scratch.as_ref().map_or(0, |x| x.acc_rows));
        let bb = nblk.max(s.scratch.as_ref().map_or(0, |x| x.dims.4)).next_power_of_two();
        s.scratch = Some(Scratch {
            raw: buf(c, "broute-raw", rr * nch * 4, st),
            iq: buf(c, "broute-iq", rr * nkv * hd * 4, st),
            sc: buf(c, "broute-sc", rr * nkv * bb * 4, st),
            bl: buf(c, "broute-bl", rr * nkv * (LIST_BLOCKS + 1) * 4, st),
            acc: buf(c, "broute-acc", ar * nh * hd * 4, st),
            ml: buf(c, "broute-ml", ar * nh * 8, st),
            dims: (rr, nh, hd, nkv, bb),
            acc_rows: ar,
        });
    }
    let x = s.scratch.as_ref().expect("set above");
    let words: [u32; 20] = [
        e.hidden as u32,
        hd as u32,
        nkv as u32,
        nh as u32,
        e.pos0 as u32,
        rows as u32,
        cap as u32,
        r.block as u32,
        r.prefix as u32,
        r.local as u32,
        r.remote as u32,
        x.dims.4 as u32,
        e.eps.to_bits(),
        e.scale.to_bits(),
        nch as u32,
        nparts as u32,
        plen as u32,
        used as u32,
        0,
        0,
    ];
    let u = s.u.get_or_insert_with(|| buf(c, "broute-u", 80, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)).clone();
    if s.words != words {
        c.queue.write_buffer(&u, 0, bytemuck::cast_slice(&words));
        s.words = words;
    }
    let (sraw, siq, ssc, sbl, sacc, sml) = (x.raw.clone(), x.iq.clone(), x.sc.clone(), x.bl.clone(), x.acc.clone(), x.ml.clone());
    let l = s.layers.entry(e.layer).or_insert_with(|| {
        let mut w = Vec::with_capacity(r.q.len() + r.k.len());
        w.extend_from_slice(r.q);
        w.extend_from_slice(r.k);
        LayerDev {
            w: stor_f32(c, "broute-w", &w),
            qn: stor_f32(c, "broute-qn", r.q_norm),
            kn: stor_f32(c, "broute-kn", r.k_norm),
            invf: stor_f32(c, "broute-invf", &e.invf[..hd / 2]),
            aqn: stor_f32(c, "broute-aqn", e.qn),
            akn: stor_f32(c, "broute-akn", e.kn),
            bgs: None,
        }
    });
    let ids: Vec<wgpu::Buffer> = vec![
        e.n1.clone(),
        e.qraw.clone(),
        e.kraw.clone(),
        e.vraw.clone(),
        e.q.clone(),
        e.attn.clone(),
        mk.clone(),
        mv.clone(),
        ik.clone(),
        sraw.clone(),
        sacc.clone(),
        ssc.clone(),
    ];
    if l.bgs.as_ref().is_none_or(|(k, _)| *k != ids) {
        let b = Bgs {
            rope: bind_pairs(
                c,
                &p.rope_b.l,
                &[(0, &u), (7, &l.invf), (24, e.qraw), (25, e.kraw), (26, e.vraw), (27, &l.aqn), (28, &l.akn), (30, e.q), (31, &mk), (32, &mv)],
            ),
            proj1: bind_pairs(c, &p.proj1.l, &[(0, &u), (1, e.n1), (2, &l.w), (3, &sraw)]),
            proj: bind_pairs(c, &p.proj.l, &[(0, &u), (1, e.n1), (2, &l.w), (3, &sraw)]),
            put: bind_pairs(c, &p.put.l, &[(0, &u), (4, &sraw), (5, &l.qn), (6, &l.kn), (7, &l.invf), (8, &siq), (9, &ik)]),
            score: bind_pairs(c, &p.score.l, &[(0, &u), (10, &siq), (11, &ik), (12, &ssc)]),
            sel: bind_pairs(c, &p.sel.l, &[(0, &u), (13, &ssc), (14, &sbl)]),
            part: bind_pairs(c, &p.part.l, &[(0, &u), (15, e.q), (16, &mk), (17, &mv), (18, &sbl), (19, &sacc), (20, &sml)]),
            merge: bind_pairs(c, &p.merge.l, &[(0, &u), (21, &sacc), (22, &sml), (23, e.attn)]),
        };
        l.bgs = Some((ids, b));
    }
    let b = &l.bgs.as_ref().expect("set above").1;
    // `CMF_BROUTE_MEMLOG=1`: the allocator's bytes as the layers of a call are encoded
    if std::env::var("CMF_BROUTE_MEMLOG").as_deref() == Ok("1") && e.layer % 11 == 0 {
        tracing::info!(
            "broute memlog: layer {} rows {} at {} cap {}: {:.0} MiB in use",
            e.layer,
            rows,
            e.pos0,
            cap,
            c.device.generate_allocator_report().map_or(0, |r| r.total_reserved_bytes) as f64 / 1048576.0
        );
    }
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.rope_b.p);
    pass.set_bind_group(0, &b.rope, &[]);
    pass.dispatch_workgroups((nh + nkv) as u32, rows as u32, 1);
    if rows <= 16 {
        pass.set_pipeline(&p.proj1.p);
        pass.set_bind_group(0, &b.proj1, &[]);
        pass.dispatch_workgroups((nch as u32).div_ceil(4), 1, 1);
    } else {
        pass.set_pipeline(&p.proj.p);
        pass.set_bind_group(0, &b.proj, &[]);
        pass.dispatch_workgroups((nch as u32).div_ceil(16), (rows as u32).div_ceil(16), 1);
    }
    pass.set_pipeline(&p.put.p);
    pass.set_bind_group(0, &b.put, &[]);
    pass.dispatch_workgroups((nkv + 1) as u32, rows as u32, 1);
    // the candidates of the call's last row bound the grid
    let local0 = (cur + 1).saturating_sub(r.local);
    if local0 > r.prefix + r.remote {
        pass.set_pipeline(&p.score.p);
        pass.set_bind_group(0, &b.score, &[]);
        pass.dispatch_workgroups((local0 - r.prefix) as u32, rows as u32, 1);
    }
    pass.set_pipeline(&p.sel.p);
    pass.set_bind_group(0, &b.sel, &[]);
    pass.dispatch_workgroups(nkv as u32, rows as u32, 1);
    pass.set_pipeline(&p.part.p);
    pass.set_bind_group(0, &b.part, &[]);
    pass.dispatch_workgroups(nkv as u32, used as u32, rows as u32);
    pass.set_pipeline(&p.merge.p);
    pass.set_bind_group(0, &b.merge, &[]);
    pass.dispatch_workgroups(nh as u32, rows as u32, 1);
    Some(())
}
