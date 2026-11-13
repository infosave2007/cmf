//! Latent MoE layers (Edge0) on the wgpu device frame.
//!
//! A latent layer's routed experts read and write an `L`-wide copy of the token: expert `e` of group
//! `labels[e]` reads row `t·G + g` of `xl = D·x` and its weighted output is summed into the same row of
//! `acc`, which `U` brings back to the hidden width. The router and the shared expert are the base
//! model's, at full width. Every routed expert of a layer is resident in one storage buffer as an NVFP4
//! slot (gate, up, down, each padded to 16 bytes; codes, UE4M3 scales, f32 global), the kernels below
//! read them per pick, and the dense projections (D, U, the shared expert) go through the frame's own
//! q8_2f matvec.

use super::*;

/// WGSL of the latent stage.
pub(crate) const QWEN4_LAT_WGSL: &str = r#"
// ── router: y[t][r] = Σ_i W[r][i]·x[t][i], W f32 [rows][cols] (the router rows, then the shared gate) ──
struct LrP { rows: u32, cols: u32, xs: u32, ys: u32 };
@group(0) @binding(0) var<storage, read>       lr_w : array<f32>;
@group(0) @binding(1) var<storage, read>       lr_x : array<f32>;
@group(0) @binding(2) var<storage, read_write> lr_y : array<f32>;
@group(0) @binding(3) var<uniform>             lr_p : LrP;
var<workgroup> lr_part: array<f32, 256>;
@compute @workgroup_size(256)
fn lat_router(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = wid.x;
    let t = wid.y;
    if (r >= lr_p.rows) { return; }
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= lr_p.cols) { break; }
        acc = acc + lr_w[r * lr_p.cols + i] * lr_x[t * lr_p.xs + i];
        i = i + 256u;
    }
    lr_part[lid] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { lr_part[lid] = lr_part[lid] + lr_part[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) { lr_y[t * lr_p.ys + r] = lr_part[0]; }
}

// ── route: softmax over the experts, the top k, renormalised (Qwen's order: ties to the lower index) ──
struct LtP { ne: u32, k: u32, ls: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       lt_l : array<f32>;
@group(0) @binding(1) var<storage, read_write> lt_pk : array<i32>;
@group(0) @binding(2) var<storage, read_write> lt_w : array<f32>;
@group(0) @binding(3) var<uniform>             lt_p : LtP;
var<workgroup> lt_pe: array<f32, 1024>;
var<workgroup> lt_red: array<f32, 256>;
var<workgroup> lt_sp: array<i32, 64>;
var<workgroup> lt_sw: array<f32, 64>;
@compute @workgroup_size(256)
fn lat_route(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let ne = lt_p.ne;
    let base = t * lt_p.ls;
    var mx = -3.0e38;
    var i = lid;
    loop { if (i >= ne) { break; } mx = max(mx, lt_l[base + i]); i = i + 256u; }
    lt_red[lid] = mx;
    workgroupBarrier();
    var s = 128u;
    loop { if (s == 0u) { break; } if (lid < s) { lt_red[lid] = max(lt_red[lid], lt_red[lid + s]); } workgroupBarrier(); s = s >> 1u; }
    mx = lt_red[0];
    workgroupBarrier();
    i = lid;
    var part = 0.0;
    loop { if (i >= ne) { break; } let v = exp(lt_l[base + i] - mx); lt_pe[i] = v; part = part + v; i = i + 256u; }
    lt_red[lid] = part;
    workgroupBarrier();
    s = 128u;
    loop { if (s == 0u) { break; } if (lid < s) { lt_red[lid] = lt_red[lid] + lt_red[lid + s]; } workgroupBarrier(); s = s >> 1u; }
    let tot = lt_red[0];
    workgroupBarrier();
    i = lid;
    loop {
        if (i >= ne) { break; }
        let p = lt_pe[i];
        var rank = 0u;
        for (var j = 0u; j < ne; j = j + 1u) {
            let q = lt_pe[j];
            if (q > p || (q == p && j < i)) { rank = rank + 1u; }
        }
        if (rank < lt_p.k) { lt_sp[rank] = i32(i); lt_sw[rank] = p / tot; }
        i = i + 256u;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var ws = 0.0;
        for (var r = 0u; r < lt_p.k; r = r + 1u) { ws = ws + lt_sw[r]; }
        for (var r = 0u; r < lt_p.k; r = r + 1u) {
            lt_pk[t * lt_p.k + r] = lt_sp[r];
            lt_w[t * lt_p.k + r] = lt_sw[r] / ws;
        }
    }
}

// ── NVFP4 decode ──
fn lat_e2m1(c: u32) -> f32 {
    let e = (c >> 1u) & 3u;
    let m = c & 1u;
    var mag = 0u;
    if (e != 0u) { mag = ((e + 126u) << 23u) | (m << 22u); } else if (m != 0u) { mag = 0x3F000000u; }
    return bitcast<f32>(mag | ((c & 8u) << 28u));
}
fn lat_ue4m3(b0: u32) -> f32 {
    let b = b0 & 0x7Fu;
    if (b == 0u || b == 0x7Fu) { return 0.0; }
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    if (e != 0u) { return bitcast<f32>(((e + 120u) << 23u) | (m << 20u)); }
    return f32(m) * 0.001953125;
}

// ── gate/up of every pick: act[t·k + j][f] = silu(sg·(G x))·(su·(U x)), x the pick's latent row ──
// The bank: per expert `slot` words: gate, up, down payloads at `pw` words each (padded), a payload =
// codes (rows·cols/8 words), UE4M3 bytes (rows·cols/16 bytes), the f32 global.
struct LgP { ff: u32, l: u32, g: u32, k: u32, slot: u32, pw: u32, xs: u32, as_: u32, rb: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       lg_bank : array<u32>;
@group(0) @binding(1) var<storage, read>       lg_lab : array<u32>;
@group(0) @binding(2) var<storage, read>       lg_pk : array<i32>;
@group(0) @binding(3) var<storage, read>       lg_x : array<f32>;
@group(0) @binding(4) var<storage, read_write> lg_a : array<f32>;
@group(0) @binding(5) var<uniform>             lg_p : LgP;
var<workgroup> lg_xs: array<f32, 2048>;
var<workgroup> lg_rd: array<vec2<f32>, 256>;
fn lat_byte(base: u32, i: u32) -> u32 {
    return (lg_bank[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu;
}
fn lat_dot16(wlo: u32, whi: u32, sc: f32, xo: u32) -> f32 {
    var a = 0.0;
    for (var q = 0u; q < 4u; q = q + 1u) {
        let by = (wlo >> (8u * q)) & 0xFFu;
        a = a + lat_e2m1(by & 15u) * lg_xs[xo + 2u * q] + lat_e2m1(by >> 4u) * lg_xs[xo + 2u * q + 1u];
    }
    for (var q = 0u; q < 4u; q = q + 1u) {
        let by = (whi >> (8u * q)) & 0xFFu;
        a = a + lat_e2m1(by & 15u) * lg_xs[xo + 8u + 2u * q] + lat_e2m1(by >> 4u) * lg_xs[xo + 8u + 2u * q + 1u];
    }
    return a * sc;
}
@compute @workgroup_size(256)
fn lat_gu(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = wid.y;                  // pick: token r / k, place r % k
    let e = lg_pk[r];
    let valid = e >= 0;
    let ue = select(0u, u32(e), valid);
    let r0 = wid.x * lg_p.rb;
    // the token's row of xl holds its groups back to back, `xs` floats apart from the next token's
    let x0 = (r / lg_p.k) * lg_p.xs + lg_lab[ue] * lg_p.l;
    var i = lid;
    loop { if (i >= lg_p.l) { break; } lg_xs[i] = lg_x[x0 + i]; i = i + 256u; }
    workgroupBarrier();
    let gb = ue * lg_p.slot;
    let ub = gb + lg_p.pw;
    let cw = lg_p.ff * lg_p.l / 8u;           // code words of a matrix
    let sb = lg_p.ff * lg_p.l / 16u;          // scale bytes of a matrix
    let sg = bitcast<f32>(lg_bank[gb + cw + sb / 4u]);
    let su = bitcast<f32>(lg_bank[ub + cw + sb / 4u]);
    let gpr = lg_p.l / 16u;
    // 32 lanes a row, eight rows a step (rb is a multiple of eight)
    let lane = lid & 31u;
    let wr = lid >> 5u;
    for (var st = 0u; st < lg_p.rb / 8u; st = st + 1u) {
        let row = r0 + st * 8u + wr;
        var ag = 0.0;
        var au = 0.0;
        if (valid && row < lg_p.ff) {
            var kk = lane;
            loop {
                if (kk >= gpr) { break; }
                let o = row * gpr + kk;
                let gsc = lat_ue4m3(lat_byte(gb + cw, o));
                let usc = lat_ue4m3(lat_byte(ub + cw, o));
                ag = ag + lat_dot16(lg_bank[gb + 2u * o], lg_bank[gb + 2u * o + 1u], gsc, kk * 16u);
                au = au + lat_dot16(lg_bank[ub + 2u * o], lg_bank[ub + 2u * o + 1u], usc, kk * 16u);
                kk = kk + 32u;
            }
        }
        lg_rd[lid] = vec2<f32>(ag, au);
        workgroupBarrier();
        for (var s = 16u; s > 0u; s = s >> 1u) {
            if (lane < s) { lg_rd[lid] = lg_rd[lid] + lg_rd[lid + s]; }
            workgroupBarrier();
        }
        if (lane == 0u && row < lg_p.ff) {
            let v = lg_rd[lid];
            let gg = sg * v.x;
            lg_a[r * lg_p.as_ + row] = select(0.0, gg / (1.0 + exp(-gg)) * (su * v.y), valid);
        }
        workgroupBarrier();
    }
}

// ── down of every pick: y[r][j] = sd·(D act) ──
struct LdP { ff: u32, l: u32, slot: u32, pw: u32, as_: u32, ys: u32, rb: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       ld_bank : array<u32>;
@group(0) @binding(1) var<storage, read>       ld_pk : array<i32>;
@group(0) @binding(2) var<storage, read>       ld_a : array<f32>;
@group(0) @binding(3) var<storage, read_write> ld_y : array<f32>;
@group(0) @binding(4) var<uniform>             ld_p : LdP;
var<workgroup> ld_xs: array<f32, 1024>;
var<workgroup> ld_rd: array<f32, 256>;
fn ldn_byte(base: u32, i: u32) -> u32 {
    return (ld_bank[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu;
}
fn ldn_dot16(wlo: u32, whi: u32, sc: f32, xo: u32) -> f32 {
    var a = 0.0;
    for (var q = 0u; q < 4u; q = q + 1u) {
        let by = (wlo >> (8u * q)) & 0xFFu;
        a = a + lat_e2m1(by & 15u) * ld_xs[xo + 2u * q] + lat_e2m1(by >> 4u) * ld_xs[xo + 2u * q + 1u];
    }
    for (var q = 0u; q < 4u; q = q + 1u) {
        let by = (whi >> (8u * q)) & 0xFFu;
        a = a + lat_e2m1(by & 15u) * ld_xs[xo + 8u + 2u * q] + lat_e2m1(by >> 4u) * ld_xs[xo + 8u + 2u * q + 1u];
    }
    return a * sc;
}
@compute @workgroup_size(256)
fn lat_dn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = wid.y;
    let e = ld_pk[r];
    let valid = e >= 0;
    let ue = select(0u, u32(e), valid);
    let r0 = wid.x * ld_p.rb;
    var i = lid;
    loop { if (i >= ld_p.ff) { break; } ld_xs[i] = ld_a[r * ld_p.as_ + i]; i = i + 256u; }
    workgroupBarrier();
    let db = ue * ld_p.slot + 2u * ld_p.pw;
    let cw = ld_p.ff * ld_p.l / 8u;
    let sb = ld_p.ff * ld_p.l / 16u;
    let sd = bitcast<f32>(ld_bank[db + cw + sb / 4u]);
    let gpr = ld_p.ff / 16u;
    let lane = lid & 31u;
    let wr = lid >> 5u;
    for (var st = 0u; st < ld_p.rb / 8u; st = st + 1u) {
        let row = r0 + st * 8u + wr;
        var a = 0.0;
        if (valid && row < ld_p.l) {
            var kk = lane;
            loop {
                if (kk >= gpr) { break; }
                let o = row * gpr + kk;
                a = a + ldn_dot16(ld_bank[db + 2u * o], ld_bank[db + 2u * o + 1u], lat_ue4m3(ldn_byte(db + cw, o)), kk * 16u);
                kk = kk + 32u;
            }
        }
        ld_rd[lid] = a;
        workgroupBarrier();
        for (var s = 16u; s > 0u; s = s >> 1u) {
            if (lane < s) { ld_rd[lid] = ld_rd[lid] + ld_rd[lid + s]; }
            workgroupBarrier();
        }
        if (lane == 0u && row < ld_p.l) { ld_y[r * ld_p.ys + row] = select(0.0, sd * ld_rd[lid], valid); }
        workgroupBarrier();
    }
}

// ── a prompt frame's picks sorted by expert (one workgroup): off[e]..off[e+1] of `order` are expert e's picks ──
struct LoP { ne: u32, np: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       lo_pk : array<i32>;
@group(0) @binding(1) var<storage, read_write> lo_off : array<u32>;
@group(0) @binding(2) var<storage, read_write> lo_ord : array<u32>;
@group(0) @binding(3) var<uniform>             lo_p : LoP;
var<workgroup> lo_cnt: array<atomic<u32>, 1024>;
var<workgroup> lo_base: array<u32, 1025>;
@compute @workgroup_size(256)
fn lat_sort(@builtin(local_invocation_index) lid: u32) {
    for (var i = lid; i < lo_p.ne; i = i + 256u) { atomicStore(&lo_cnt[i], 0u); }
    workgroupBarrier();
    for (var q = lid; q < lo_p.np; q = q + 256u) {
        let e = lo_pk[q];
        if (e >= 0) { atomicAdd(&lo_cnt[u32(e)], 1u); }
    }
    workgroupBarrier();
    if (lid == 0u) {
        var acc = 0u;
        for (var i = 0u; i < lo_p.ne; i = i + 1u) {
            lo_base[i] = acc;
            acc = acc + atomicLoad(&lo_cnt[i]);
            atomicStore(&lo_cnt[i], 0u);
        }
        lo_base[lo_p.ne] = acc;
    }
    workgroupBarrier();
    for (var q = lid; q < lo_p.np; q = q + 256u) {
        let e = lo_pk[q];
        if (e >= 0) {
            let at = atomicAdd(&lo_cnt[u32(e)], 1u);
            lo_ord[lo_base[u32(e)] + at] = q;
        }
    }
    for (var i = lid; i <= lo_p.ne; i = i + 256u) { lo_off[i] = lo_base[i]; }
}

// ── a prompt frame's gate/up, by expert: workgroup (row tile of 32, expert); the expert's picks in tiles
// of 8 (a 256-token frame gives an expert about five), its rows decoded once per tile into shared memory,
// 64 columns at a time; a thread owns one row and one pick ──
struct LgG { ff: u32, l: u32, g: u32, k: u32, slot: u32, pw: u32, xs: u32, as_: u32 };
@group(0) @binding(0) var<storage, read>       gg_bank : array<u32>;
@group(0) @binding(1) var<storage, read>       gg_lab : array<u32>;
@group(0) @binding(2) var<storage, read>       gg_off : array<u32>;
@group(0) @binding(3) var<storage, read>       gg_ord : array<u32>;
@group(0) @binding(4) var<storage, read>       gg_x : array<f32>;
@group(0) @binding(5) var<storage, read_write> gg_a : array<f32>;
@group(0) @binding(6) var<uniform>             gg_p : LgG;
var<workgroup> gg_wg: array<f32, 2048>;   // gate tile [32 rows][64]
var<workgroup> gg_wu: array<f32, 2048>;   // up tile
var<workgroup> gg_xt: array<f32, 512>;    // x tile [8 picks][64]
var<workgroup> gg_n: vec2<u32>;
fn gg_byte(base: u32, i: u32) -> u32 { return (gg_bank[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu; }
@compute @workgroup_size(256)
fn lat_gu_g(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let e = wid.y;
    if (lid == 0u) { gg_n = vec2<u32>(gg_off[e], gg_off[e + 1u] - gg_off[e]); }
    let nn = workgroupUniformLoad(&gg_n);
    let p0 = nn.x;
    let np = nn.y;
    if (np == 0u) { return; }
    let r0 = wid.x * 32u;
    let gb = e * gg_p.slot;
    let ub = gb + gg_p.pw;
    let cw = gg_p.ff * gg_p.l / 8u;
    let sb = gg_p.ff * gg_p.l / 16u;
    let sg = bitcast<f32>(gg_bank[gb + cw + sb / 4u]);
    let su = bitcast<f32>(gg_bank[ub + cw + sb / 4u]);
    let grp = gg_lab[e];
    let gpr = gg_p.l / 16u;
    // this thread's output: row tr, pick tp
    let tr = lid >> 3u;
    let tp = lid & 7u;
    for (var pt = 0u; pt < np; pt = pt + 8u) {
        var ag = 0.0;
        var au = 0.0;
        for (var k0 = 0u; k0 < gg_p.l; k0 = k0 + 64u) {
            workgroupBarrier();
            // decode the tiles: 32 rows x 64 columns of gate and up (four 16-groups a row), x of 32 picks
            for (var i = lid; i < 32u * 4u; i = i + 256u) {
                let rr = i >> 2u;
                let q = i & 3u;
                let row = min(r0 + rr, gg_p.ff - 1u);
                let o = row * gpr + k0 / 16u + q;
                let gsc = lat_ue4m3(gg_byte(gb + cw, o));
                let usc = lat_ue4m3(gg_byte(ub + cw, o));
                let w0 = gg_bank[gb + 2u * o];
                let w1 = gg_bank[gb + 2u * o + 1u];
                let u0 = gg_bank[ub + 2u * o];
                let u1 = gg_bank[ub + 2u * o + 1u];
                for (var j = 0u; j < 8u; j = j + 1u) {
                    let wsel = select(w1, w0, j < 4u);
                    let usel = select(u1, u0, j < 4u);
                    let by = (wsel >> (8u * (j & 3u))) & 0xFFu;
                    let bu = (usel >> (8u * (j & 3u))) & 0xFFu;
                    let c = rr * 64u + q * 16u + 2u * j;
                    gg_wg[c] = lat_e2m1(by & 15u) * gsc;
                    gg_wg[c + 1u] = lat_e2m1(by >> 4u) * gsc;
                    gg_wu[c] = lat_e2m1(bu & 15u) * usc;
                    gg_wu[c + 1u] = lat_e2m1(bu >> 4u) * usc;
                }
            }
            for (var i = lid; i < 8u * 64u; i = i + 256u) {
                let pp = i >> 6u;
                let kk = i & 63u;
                var v = 0.0;
                if (pt + pp < np) {
                    let pick = gg_ord[p0 + pt + pp];
                    v = gg_x[(pick / gg_p.k) * gg_p.xs + grp * gg_p.l + k0 + kk];
                }
                gg_xt[i] = v;
            }
            workgroupBarrier();
            for (var kk = 0u; kk < 64u; kk = kk + 4u) {
                let w4 = vec4<f32>(gg_wg[tr * 64u + kk], gg_wg[tr * 64u + kk + 1u], gg_wg[tr * 64u + kk + 2u], gg_wg[tr * 64u + kk + 3u]);
                let u4 = vec4<f32>(gg_wu[tr * 64u + kk], gg_wu[tr * 64u + kk + 1u], gg_wu[tr * 64u + kk + 2u], gg_wu[tr * 64u + kk + 3u]);
                let x4 = vec4<f32>(gg_xt[tp * 64u + kk], gg_xt[tp * 64u + kk + 1u], gg_xt[tp * 64u + kk + 2u], gg_xt[tp * 64u + kk + 3u]);
                ag = ag + dot(w4, x4);
                au = au + dot(u4, x4);
            }
        }
        let row = r0 + tr;
        if (row < gg_p.ff && pt + tp < np) {
            let pick = gg_ord[p0 + pt + tp];
            let gv = sg * ag;
            gg_a[pick * gg_p.as_ + row] = gv / (1.0 + exp(-gv)) * (su * au);
        }
    }
}

// ── a prompt frame's down, by expert: workgroup (row tile of 32 of L, expert) ──
struct LdG { ff: u32, l: u32, slot: u32, pw: u32, as_: u32, ys: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       dg_bank : array<u32>;
@group(0) @binding(1) var<storage, read>       dg_off : array<u32>;
@group(0) @binding(2) var<storage, read>       dg_ord : array<u32>;
@group(0) @binding(3) var<storage, read>       dg_a : array<f32>;
@group(0) @binding(4) var<storage, read_write> dg_y : array<f32>;
@group(0) @binding(5) var<uniform>             dg_p : LdG;
var<workgroup> dg_wt: array<f32, 2048>;
var<workgroup> dg_at: array<f32, 512>;
var<workgroup> dg_n: vec2<u32>;
fn dg_byte(base: u32, i: u32) -> u32 { return (dg_bank[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu; }
@compute @workgroup_size(256)
fn lat_dn_g(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let e = wid.y;
    if (lid == 0u) { dg_n = vec2<u32>(dg_off[e], dg_off[e + 1u] - dg_off[e]); }
    let nn = workgroupUniformLoad(&dg_n);
    let p0 = nn.x;
    let np = nn.y;
    if (np == 0u) { return; }
    let r0 = wid.x * 32u;
    let db = e * dg_p.slot + 2u * dg_p.pw;
    let cw = dg_p.ff * dg_p.l / 8u;
    let sb = dg_p.ff * dg_p.l / 16u;
    let sd = bitcast<f32>(dg_bank[db + cw + sb / 4u]);
    let gpr = dg_p.ff / 16u;
    let tr = lid >> 3u;
    let tp = lid & 7u;
    for (var pt = 0u; pt < np; pt = pt + 8u) {
        var acc = 0.0;
        for (var k0 = 0u; k0 < dg_p.ff; k0 = k0 + 64u) {
            workgroupBarrier();
            for (var i = lid; i < 32u * 4u; i = i + 256u) {
                let rr = i >> 2u;
                let q = i & 3u;
                let row = min(r0 + rr, dg_p.l - 1u);
                let o = row * gpr + k0 / 16u + q;
                let sc = lat_ue4m3(dg_byte(db + cw, o));
                let w0 = dg_bank[db + 2u * o];
                let w1 = dg_bank[db + 2u * o + 1u];
                for (var j = 0u; j < 8u; j = j + 1u) {
                    let wsel = select(w1, w0, j < 4u);
                    let by = (wsel >> (8u * (j & 3u))) & 0xFFu;
                    let c = rr * 64u + q * 16u + 2u * j;
                    dg_wt[c] = lat_e2m1(by & 15u) * sc;
                    dg_wt[c + 1u] = lat_e2m1(by >> 4u) * sc;
                }
            }
            for (var i = lid; i < 8u * 64u; i = i + 256u) {
                let pp = i >> 6u;
                let kk = i & 63u;
                var v = 0.0;
                if (pt + pp < np) {
                    let pick = dg_ord[p0 + pt + pp];
                    v = dg_a[pick * dg_p.as_ + k0 + kk];
                }
                dg_at[i] = v;
            }
            workgroupBarrier();
            for (var kk = 0u; kk < 64u; kk = kk + 4u) {
                let w4 = vec4<f32>(dg_wt[tr * 64u + kk], dg_wt[tr * 64u + kk + 1u], dg_wt[tr * 64u + kk + 2u], dg_wt[tr * 64u + kk + 3u]);
                let a4 = vec4<f32>(dg_at[tp * 64u + kk], dg_at[tp * 64u + kk + 1u], dg_at[tp * 64u + kk + 2u], dg_at[tp * 64u + kk + 3u]);
                acc = acc + dot(w4, a4);
            }
        }
        let row = r0 + tr;
        if (row < dg_p.l && pt + tp < np) {
            let pick = dg_ord[p0 + pt + tp];
            dg_y[pick * dg_p.ys + row] = sd * acc;
        }
    }
}

// ── the groups' sums: acc[t][g·L + j] = Σ_{picks k of t in group g} w·y[t·K + k][j] ──
struct LsP { n: u32, g: u32, l: u32, k: u32, ys: u32, accs: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       ls_pk : array<i32>;
@group(0) @binding(1) var<storage, read>       ls_lab : array<u32>;
@group(0) @binding(2) var<storage, read>       ls_w : array<f32>;
@group(0) @binding(3) var<storage, read>       ls_y : array<f32>;
@group(0) @binding(4) var<storage, read_write> ls_acc : array<f32>;
@group(0) @binding(5) var<uniform>             ls_p : LsP;
@compute @workgroup_size(256)
fn lat_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gl = ls_p.g * ls_p.l;
    let i = gid.x;
    let t = gid.y;
    if (i >= gl || t >= ls_p.n) { return; }
    let g = i / ls_p.l;
    let j = i - g * ls_p.l;
    var a = 0.0;
    for (var q = 0u; q < ls_p.k; q = q + 1u) {
        let e = ls_pk[t * ls_p.k + q];
        if (e >= 0 && ls_lab[u32(e)] == g) { a = a + ls_w[t * ls_p.k + q] * ls_y[(t * ls_p.k + q) * ls_p.ys + j]; }
    }
    ls_acc[t * ls_p.accs + i] = a;
}

// ── the shared expert's act: a = silu(g)·u ──
struct LaP { n: u32, ff: u32, gs: u32, as_: u32 };
@group(0) @binding(0) var<storage, read>       la_g : array<f32>;
@group(0) @binding(1) var<storage, read>       la_u : array<f32>;
@group(0) @binding(2) var<storage, read_write> la_a : array<f32>;
@group(0) @binding(3) var<uniform>             la_p : LaP;
@compute @workgroup_size(256)
fn lat_silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= la_p.ff || t >= la_p.n) { return; }
    let g = la_g[t * la_p.gs + i];
    la_a[t * la_p.as_ + i] = g / (1.0 + exp(-g)) * la_u[t * la_p.gs + i];
}

// ── out[t][h] += σ(gate logit)·shared[t][h] ──
struct LhP { n: u32, h: u32, os: u32, ls: u32, gcol: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> lh_o : array<f32>;
@group(0) @binding(1) var<storage, read>       lh_s : array<f32>;
@group(0) @binding(2) var<storage, read>       lh_l : array<f32>;
@group(0) @binding(3) var<uniform>             lh_p : LhP;
@compute @workgroup_size(256)
fn lat_shadd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= lh_p.h || t >= lh_p.n) { return; }
    let gl = lh_l[t * lh_p.ls + lh_p.gcol];
    lh_o[t * lh_p.os + i] = lh_o[t * lh_p.os + i] + lh_s[t * lh_p.os + i] / (1.0 + exp(-gl));
}
"#;

/// Pipelines of the latent stage.
pub(crate) struct LatPipes {
    router: wgpu::ComputePipeline,
    route: wgpu::ComputePipeline,
    gu: wgpu::ComputePipeline,
    dn: wgpu::ComputePipeline,
    sum: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    shadd: wgpu::ComputePipeline,
    sort: wgpu::ComputePipeline,
    gu_g: wgpu::ComputePipeline,
    dn_g: wgpu::ComputePipeline,
}

static LAT_PIPES: std::sync::OnceLock<Option<LatPipes>> = std::sync::OnceLock::new();

fn lat_pipes(c: &Ctx) -> Option<&'static LatPipes> {
    LAT_PIPES
        .get_or_init(|| {
            let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = qwen_module(c, "qwen4-latent", QWEN4_LAT_WGSL);
            if let Some(e) = pollster::block_on(scope.pop()) {
                tracing::warn!("qwen4 latent shader module rejected: {e}");
                return None;
            }
            let pipe = |ep: &str| {
                c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(ep),
                    layout: None,
                    module: &module,
                    entry_point: Some(ep),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: qwen_checked(),
                    },
                    cache: c.pipeline_cache.as_ref(),
                })
            };
            Some(LatPipes {
                router: pipe("lat_router"),
                route: pipe("lat_route"),
                gu: pipe("lat_gu"),
                dn: pipe("lat_dn"),
                sum: pipe("lat_sum"),
                silu: pipe("lat_silu"),
                shadd: pipe("lat_shadd"),
                sort: pipe("lat_sort"),
                gu_g: pipe("lat_gu_g"),
                dn_g: pipe("lat_dn_g"),
            })
        })
        .as_ref()
}

/// A latent layer on the card: every routed expert (NVFP4 slots in one buffer), their groups, the
/// router (f32, the shared expert's gate row last), and the directory indices of D, U and the shared
/// expert (q8_2f, resident with the skeleton).
pub(crate) struct LatDev {
    bank: wgpu::Buffer,
    labels: wgpu::Buffer,
    router: wgpu::Buffer,
    d_idx: usize,
    u_idx: usize,
    sh_idx: [usize; 3],
    l: usize,
    groups: usize,
    ff: usize,
    ne: usize,
    k: usize,
    slot_w: u32,
    pw_w: u32,
}

impl LatDev {
    /// Upload a latent layer's experts, groups and router. None when a matrix is not where the stage
    /// expects it (the caller keeps the layer off the card).
    pub(crate) fn new(model: &CmfModel, lat: &crate::qwen4_exp::Latent, moe: &crate::pipeline::MoeFfn, hidden: usize) -> Option<Self> {
        let c = ctx()?;
        lat_pipes(c)?;
        let ne = moe.experts.len();
        let ff = moe.experts.first()?.gate_proj.rows();
        let (l, groups) = (lat.dim, lat.groups);
        if l > 2048 || ff > 1024 || ne > 1024 || moe.top_k > 64 || !l.is_multiple_of(16) || !ff.is_multiple_of(16) {
            return None;
        }
        let pad = |n: usize| n.div_ceil(16) * 16;
        let pg = cortiq_core::nvfp4::nbytes(ff, l)?;
        let pd = cortiq_core::nvfp4::nbytes(l, ff)?;
        if pg != pd {
            return None;
        }
        let pw = pad(pg);
        let slot = 3 * pw;
        let bank = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen4-latent-bank"),
            size: (ne * slot) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut stage = vec![0u8; slot * 32];
        for (c0, chunk) in moe.experts.chunks(32).enumerate() {
            stage.iter_mut().for_each(|b| *b = 0);
            for (j, e) in chunk.iter().enumerate() {
                for (k, (t, rows, cols)) in [(&e.gate_proj, ff, l), (&e.up_proj, ff, l), (&e.down_proj, l, ff)].into_iter().enumerate() {
                    if t.rows() != rows || t.cols() != cols || t.model_dtype() != Some(TensorDtype::Nvfp4) {
                        return None;
                    }
                    let b = t.mapped_bytes()?;
                    if b.len() != pg {
                        return None;
                    }
                    let at = j * slot + k * pw;
                    stage[at..at + pg].copy_from_slice(b);
                }
            }
            c.queue.write_buffer(&bank, (c0 * 32 * slot) as u64, &stage[..chunk.len() * slot]);
        }
        let labels_u: Vec<u32> = lat.labels.iter().map(|&g| g as u32).collect();
        let labels = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("qwen4-latent-labels"),
            contents: bytemuck::cast_slice(&labels_u),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let (shf, gate) = moe.shared.as_ref()?;
        let gate = gate.as_ref()?;
        if moe.router.rows() != ne || moe.router.cols() != hidden || gate.rows() != 1 || gate.cols() != hidden {
            return None;
        }
        let mut rw = vec![0f32; (ne + 1) * hidden];
        for r in 0..ne {
            moe.router.row_f32(r, &mut rw[r * hidden..(r + 1) * hidden]);
        }
        gate.row_f32(0, &mut rw[ne * hidden..]);
        let router = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("qwen4-latent-router"),
            contents: bytemuck::cast_slice(&rw),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let q8 = |t: &crate::qtensor::QTensor| -> Option<usize> {
            (t.model_dtype() == Some(TensorDtype::Q8_2f)).then(|| t.model_idx()).flatten()
        };
        let _ = model;
        Some(Self {
            bank,
            labels,
            router,
            d_idx: lat.down_idx.filter(|_| q8(&lat.down).is_some())?,
            u_idx: lat.up_idx.filter(|_| q8(&lat.up).is_some())?,
            sh_idx: [q8(&shf.gate_proj)?, q8(&shf.up_proj)?, q8(&shf.down_proj)?],
            l,
            groups,
            ff,
            ne,
            k: moe.top_k.min(ne),
            slot_w: (slot / 4) as u32,
            pw_w: (pw / 4) as u32,
        })
    }

    /// Bytes this layer holds on the card.
    pub(crate) fn bytes(&self) -> u64 {
        self.bank.size() + self.labels.size() + self.router.size()
    }

}

/// Frame-buffer tags of the latent stage.
const T_LLOG: u8 = 170;
const T_LPICK: u8 = 171;
const T_LWT: u8 = 172;
const T_LXL: u8 = 173;
const T_LACT: u8 = 174;
const T_LY: u8 = 175;
const T_LACC: u8 = 176;
const T_LSG: u8 = 177;
const T_LSU: u8 = 178;
const T_LSA: u8 = 168;
const T_LSO: u8 = 169;
const T_LOFF: u8 = 166;
const T_LORD: u8 = 167;

/// The expert-grouped kernels of a prompt frame (`CMF_QWEN_LAT_GROUPED=1`); off by default: on the RTX PRO 6000
/// they read a 4K prompt at 380 tok/s against 445 for the per-pick ones (an expert gets about five picks a frame).
fn grouped_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_QWEN_LAT_GROUPED").as_deref() == Ok("1"))
}

/// The latent MoE of `nt` token rows: `x2` (rows `hs` floats apart) in, `mo` (the same stride) out.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    bc: Bc,
    model: &Arc<CmfModel>,
    lat: &LatDev,
    hidden: usize,
    x2: &wgpu::Buffer,
    hs: u32,
    mo: &wgpu::Buffer,
    nt: usize,
) -> Option<()> {
    let lp = lat_pipes(c)?;
    let (l, g, ff, ne, k) = (lat.l, lat.groups, lat.ff, lat.ne, lat.k);
    let gl = g * l;
    let w = |idx: usize| weight(c, model, idx);
    let (wd, wu, wsg, wsu, wsd) = (w(lat.d_idx)?, w(lat.u_idx)?, w(lat.sh_idx[0])?, w(lat.sh_idx[1])?, w(lat.sh_idx[2])?);
    let logits = tbuf_exact(c, T_LLOG, (ne + 1) * 4, false);
    let picks = tbuf_exact(c, T_LPICK, k * 4, false);
    let wts = tbuf_exact(c, T_LWT, k * 4, false);
    let xl = tbuf(c, T_LXL, gl * 4, false);
    let xls = es(gl * 4);
    let act = tbuf_exact(c, T_LACT, k * ff * 4, false);
    let y = tbuf_exact(c, T_LY, k * l * 4, false);
    let acc = tbuf(c, T_LACC, gl * 4, false);
    let sgb = tbuf(c, T_LSG, ff * 4, false);
    let sub = tbuf(c, T_LSU, ff * 4, false);
    let sab = tbuf(c, T_LSA, ff * 4, false);
    let sob = tbuf(c, T_LSO, hidden * 4, false);
    let ffs = es(ff * 4);
    let rows = (ne + 1) as u32;
    // `CMF_Q4_SKIP=lat_r,lat_d,lat_x,lat_u,lat_s`: leave the router, D, the experts, U or the shared expert
    // out (a cost probe: the output goes wrong)
    // the router and the shared expert's gate
    let (rt, lg) = (lat.router.clone(), logits.clone());
    if !skip("lat_r") {
    dispatch(
        c,
        pass,
        &lp.router,
        bc,
        700,
        "qwen4-lat-router",
        || vec![rt, x2.clone(), lg, uniform_u32x4(c, [rows, hidden as u32, hs, rows])],
        (rows, nt as u32, 1),
    );
    let (lg, pk, wt) = (logits.clone(), picks.clone(), wts.clone());
    dispatch(
        c,
        pass,
        &lp.route,
        bc,
        701,
        "qwen4-lat-route",
        || vec![lg, pk, wt, uniform_u32x4(c, [ne as u32, k as u32, rows, 0])],
        (nt as u32, 1, 1),
    );
    }
    // D
    if !skip("lat_d") && !q82_t(c, p, pass, &wd, x2, hs, &xl, xls, nt, bc, 702) {
        return None;
    }
    if !skip("lat_x") {
    // a prompt frame: the picks sorted by expert, every expert's rows read once per tile of 32 picks
    if nt > TMAX && grouped_on() {
        let off = tbuf_exact(c, T_LOFF, (ne + 1) * 4 / TROWS + 4, false);
        let ord = tbuf_exact(c, T_LORD, k * 4, false);
        let (pk, o1, o2) = (picks.clone(), off.clone(), ord.clone());
        dispatch(
            c,
            pass,
            &lp.sort,
            bc,
            713,
            "qwen4-lat-sort",
            || vec![pk, o1, o2, uniform_u32x4(c, [ne as u32, (nt * k) as u32, 0, 0])],
            (1, 1, 1),
        );
        let (bk, lb, o1, o2, xx, aa) = (lat.bank.clone(), lat.labels.clone(), off.clone(), ord.clone(), xl.clone(), act.clone());
        dispatch(
            c,
            pass,
            &lp.gu_g,
            bc,
            703,
            "qwen4-lat-gu-g",
            || vec![bk, lb, o1, o2, xx, aa, uniform_u32x8(c, [ff as u32, l as u32, g as u32, k as u32, lat.slot_w, lat.pw_w, xls, ff as u32])],
            ((ff as u32).div_ceil(32), ne as u32, 1),
        );
        let (bk, o1, o2, aa, yy) = (lat.bank.clone(), off.clone(), ord.clone(), act.clone(), y.clone());
        dispatch(
            c,
            pass,
            &lp.dn_g,
            bc,
            704,
            "qwen4-lat-dn-g",
            || vec![bk, o1, o2, aa, yy, uniform_u32x8(c, [ff as u32, l as u32, lat.slot_w, lat.pw_w, ff as u32, l as u32, 0, 0])],
            ((l as u32).div_ceil(32), ne as u32, 1),
        );
    } else {
    // the picks' gate/up and down, the groups' sums
    let rb = 8u32;
    let (bk, lb, pk, xx, aa) = (lat.bank.clone(), lat.labels.clone(), picks.clone(), xl.clone(), act.clone());
    dispatch(
        c,
        pass,
        &lp.gu,
        bc,
        703,
        "qwen4-lat-gu",
        || {
            vec![
                bk,
                lb,
                pk,
                xx,
                aa,
                uniform_u32x16(c, [ff as u32, l as u32, g as u32, k as u32, lat.slot_w, lat.pw_w, xls, ff as u32, rb, 0, 0, 0, 0, 0, 0, 0]),
            ]
        },
        ((ff as u32).div_ceil(rb), (nt * k) as u32, 1),
    );
    let (bk, pk, aa, yy) = (lat.bank.clone(), picks.clone(), act.clone(), y.clone());
    dispatch(
        c,
        pass,
        &lp.dn,
        bc,
        704,
        "qwen4-lat-dn",
        || vec![bk, pk, aa, yy, uniform_u32x8(c, [ff as u32, l as u32, lat.slot_w, lat.pw_w, ff as u32, l as u32, rb, 0])],
        ((l as u32).div_ceil(rb), (nt * k) as u32, 1),
    );
    }
    }
    let (pk, lb, wt, yy, ac) = (picks.clone(), lat.labels.clone(), wts.clone(), y.clone(), acc.clone());
    dispatch(
        c,
        pass,
        &lp.sum,
        bc,
        705,
        "qwen4-lat-sum",
        || vec![pk, lb, wt, yy, ac, uniform_u32x8(c, [nt as u32, g as u32, l as u32, k as u32, l as u32, xls, 0, 0])],
        ((gl as u32).div_ceil(256), nt as u32, 1),
    );
    // U into the MoE output
    if !skip("lat_u") && !q82_t(c, p, pass, &wu, &acc, xls, mo, hs, nt, bc, 706) {
        return None;
    }
    if skip("lat_s") {
        return Some(());
    }
    // the shared expert at full width, gated
    if !q82_t(c, p, pass, &wsg, x2, hs, &sgb, ffs, nt, bc, 707)
        || !q82_t(c, p, pass, &wsu, x2, hs, &sub, ffs, nt, bc, 708)
    {
        return None;
    }
    let (s1, s2, s3) = (sgb.clone(), sub.clone(), sab.clone());
    dispatch(
        c,
        pass,
        &lp.silu,
        bc,
        710,
        "qwen4-lat-silu",
        || vec![s1, s2, s3, uniform_u32x4(c, [nt as u32, ff as u32, ffs, ffs])],
        ((ff as u32).div_ceil(256), nt as u32, 1),
    );
    if !q82_t(c, p, pass, &wsd, &sab, ffs, &sob, hs, nt, bc, 711) {
        return None;
    }
    let (o, so, lg) = (mo.clone(), sob.clone(), logits.clone());
    dispatch(
        c,
        pass,
        &lp.shadd,
        bc,
        712,
        "qwen4-lat-shadd",
        || vec![o, so, lg, uniform_u32x8(c, [nt as u32, hidden as u32, hs, rows, ne as u32, 0, 0, 0])],
        ((hidden as u32).div_ceil(256), nt as u32, 1),
    );
    Some(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn latent_wgsl_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(super::QWEN4_LAT_WGSL).expect("latent WGSL parses");
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .expect("latent WGSL validates");
        for ep in ["lat_router", "lat_route", "lat_gu", "lat_dn", "lat_sum", "lat_silu", "lat_shadd", "lat_sort", "lat_gu_g", "lat_dn_g"] {
            assert!(module.entry_points.iter().any(|e| e.name == ep), "{ep} missing");
        }
    }
}
