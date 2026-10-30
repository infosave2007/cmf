//! Batched prefill for the Nemotron-H device frame: up to `PB_MAX` prompt
//! positions per pass, every weight read once per pass.
//!
//! The decode frame reads all ~2.6 GB of active weights for every position;
//! a 30 s recording is ~420 prompt positions, so the prompt dominated the
//! whole transcription. Here each q8_2f projection keeps its row in the
//! lanes' registers and walks the pass's positions; the routed experts are
//! grouped on the device (per expert, the (position, slot) pairs that chose
//! it) so every selected expert is read once per pass; the Mamba-2 conv ring
//! and scan and the KV cache advance position by position inside their
//! kernels. The count and the first position of the pass live in a small
//! `meta` buffer, so every bind group is built once.
//!
//! The arithmetic is the decode frame's: same products, same state layout,
//! so a prompt fed here and decoded by the frame is one sequence.

/// Positions per pass.
pub(crate) const PB_MAX: usize = 64;
/// Rows a subgroup owns in the expert banks (`PQ_RW` in the WGSL).
pub(crate) const PQ_RW: usize = 16;

pub(crate) const PB_WGSL: &str = r#"
// n positions from pos0; `snap` > 0: the conv rings and scan states after
// every position but the last are also written to the snapshot buffers
// (a verify pass rolls back to them).
struct Meta { n: u32, pos0: u32, snap: u32, _b: u32 };

// ── h[b] += o[b] (add 1) or the MoE sum (add 2); y[b] = rms(h[b])·w ──
struct PaP { hid: u32, add: u32, eps: f32, k: u32 };
@group(0) @binding(0) var<storage, read_write> pa_h  : array<f32>;
@group(0) @binding(1) var<storage, read>       pa_o  : array<f32>;
@group(0) @binding(2) var<storage, read>       pa_w  : array<f32>;
@group(0) @binding(3) var<storage, read_write> pa_y  : array<f32>;
@group(0) @binding(4) var<storage, read>       pa_eo : array<f32>;
@group(0) @binding(5) var<storage, read>       pa_sw : array<f32>;
@group(0) @binding(6) var<storage, read>       pa_m  : Meta;
@group(0) @binding(7) var<uniform>             pa_p  : PaP;
var<workgroup> pa_part: array<f32, 256>;
@compute @workgroup_size(256)
fn pb_add_rms(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let b = wid.x;
    if (b >= pa_m.n) { return; }
    let n = pa_p.hid;
    let base = b * n;
    var acc = 0.0;
    for (var i = lid; i < n; i = i + 256u) {
        var v = pa_h[base + i];
        if (pa_p.add == 1u) {
            v = v + pa_o[base + i];
            pa_h[base + i] = v;
        } else if (pa_p.add == 2u) {
            var m = pa_o[base + i];
            for (var s = 0u; s < pa_p.k; s = s + 1u) {
                m = m + pa_sw[b * pa_p.k + s] * pa_eo[(b * pa_p.k + s) * n + i];
            }
            v = v + m;
            pa_h[base + i] = v;
        }
        acc = acc + v * v;
    }
    pa_part[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { pa_part[lid] = pa_part[lid] + pa_part[lid + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(pa_part[0] / f32(n) + pa_p.eps);
    for (var i = lid; i < n; i = i + 256u) {
        pa_y[base + i] = pa_w[i] * (pa_h[base + i] * inv);
    }
}

// ── in-place relu² over the pass ──
struct RqP { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> rq_y : array<f32>;
@group(0) @binding(1) var<uniform>             rq_p : RqP;
@compute @workgroup_size(256)
fn pb_relu2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 256u + gid.x;
    if (i >= rq_p.n) { return; }
    let v = max(rq_y[i], 0.0);
    rq_y[i] = v * v;
}

// ── f32 GEMM over the pass (the router) ──
struct FpP { rows: u32, cols: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       fp_w : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       fp_x : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> fp_y : array<f32>;
@group(0) @binding(3) var<storage, read>       fp_m : Meta;
@group(0) @binding(4) var<uniform>             fp_p : FpP;
@compute @workgroup_size(256)
fn pb_f32_mm(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32,
             @builtin(subgroup_invocation_id) lane: u32) {
    let rows = fp_p.rows;
    let c4 = fp_p.cols / 4u;
    let row = wid.x * 8u + lid / 32u;
    let live = row < rows;
    let r = select(rows - 1u, row, live);
    var wv: array<vec4<f32>, 32>;
    var nw = 0u;
    for (var k = lane; k < c4; k = k + 32u) {
        wv[nw] = fp_w[r * c4 + k];
        nw = nw + 1u;
    }
    for (var b = 0u; b < fp_m.n; b = b + 1u) {
        var acc = 0.0;
        for (var i = 0u; i < nw; i = i + 1u) {
            acc = acc + dot(wv[i], fp_x[b * c4 + lane + i * 32u]);
        }
        let t = subgroupAdd(acc);
        if (lane == 0u && live) { fp_y[b * rows + row] = t; }
    }
}

// ── Mamba-2 conv over the pass, ring [3][cdim] advanced per position ──
struct PcP { cdim: u32, off: u32, stride: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       pc_zx   : array<f32>;
@group(0) @binding(1) var<storage, read>       pc_w    : array<f32>;
@group(0) @binding(2) var<storage, read>       pc_b    : array<f32>;
@group(0) @binding(3) var<storage, read_write> pc_ring : array<f32>;
@group(0) @binding(4) var<storage, read_write> pc_out  : array<f32>;
@group(0) @binding(5) var<storage, read>       pc_m    : Meta;
@group(0) @binding(6) var<storage, read_write> pc_snap : array<f32>;
@group(0) @binding(7) var<uniform>             pc_p    : PcP;
@compute @workgroup_size(256)
fn pb_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let n = pc_p.cdim;
    if (c >= n) { return; }
    var r0 = pc_ring[c];
    var r1 = pc_ring[n + c];
    var r2 = pc_ring[2u * n + c];
    let w = c * 4u;
    for (var b = 0u; b < pc_m.n; b = b + 1u) {
        let x = pc_zx[b * pc_p.stride + pc_p.off + c];
        let acc = pc_b[c] + pc_w[w] * r0 + pc_w[w + 1u] * r1 + pc_w[w + 2u] * r2 + pc_w[w + 3u] * x;
        r0 = r1;
        r1 = r2;
        r2 = x;
        pc_out[b * n + c] = acc / (1.0 + exp(-acc));
        if (pc_m.snap != 0u && b + 1u < pc_m.n) {
            pc_snap[b * 3u * n + c] = r0;
            pc_snap[b * 3u * n + n + c] = r1;
            pc_snap[b * 3u * n + 2u * n + c] = r2;
        }
    }
    pc_ring[c] = r0;
    pc_ring[n + c] = r1;
    pc_ring[2u * n + c] = r2;
}

// ── Mamba-2 scan over the pass (decode frame's state layout) ──
struct PsP { inner: u32, dt_off: u32, ng: u32, ns: u32, hpg: u32, hdim: u32, nh: u32, zstride: u32 };
@group(0) @binding(0) var<storage, read>       ps_zx  : array<f32>;
@group(0) @binding(1) var<storage, read>       ps_xbc : array<f32>;
@group(0) @binding(2) var<storage, read_write> ps_s   : array<f32>;
@group(0) @binding(3) var<storage, read>       ps_par : array<f32>;
@group(0) @binding(4) var<storage, read_write> ps_y   : array<f32>;
@group(0) @binding(5) var<storage, read>       ps_m   : Meta;
@group(0) @binding(6) var<storage, read_write> ps_snap : array<f32>;
@group(0) @binding(7) var<uniform>             ps_p   : PsP;
var<workgroup> ps_b: array<f32, 256>;
var<workgroup> ps_c: array<f32, 256>;
var<workgroup> ps_part: array<f32, 256>;
@compute @workgroup_size(256)
fn pb_scan(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let ns = ps_p.ns;
    let hdim = ps_p.hdim;
    let p = lid % 64u;
    let kq = lid / 64u;
    let g = h / ps_p.hpg;
    let cdim = ps_p.inner + 2u * ps_p.ng * ns;
    let q = ns / 4u;
    let base = h * ns * hdim + p;
    let a = ps_par[h];
    let dpar = ps_par[ps_p.nh + h];
    let dtb = ps_par[2u * ps_p.nh + h];
    for (var b = 0u; b < ps_m.n; b = b + 1u) {
        let xo = b * cdim;
        for (var k = lid; k < ns; k = k + 256u) {
            ps_b[k] = ps_xbc[xo + ps_p.inner + g * ns + k];
            ps_c[k] = ps_xbc[xo + ps_p.inner + ps_p.ng * ns + g * ns + k];
        }
        workgroupBarrier();
        let raw = ps_zx[b * ps_p.zstride + ps_p.dt_off + h] + dtb;
        var dt = log(1.0 + exp(raw));
        if (raw > 20.0) { dt = raw; }
        if (raw < -15.0) { dt = exp(raw); }
        let da = exp(dt * a);
        let x = ps_xbc[xo + h * hdim + p];
        let dx = dt * x;
        var acc = 0.0;
        let snap = ps_m.snap != 0u && b + 1u < ps_m.n;
        let so = b * ps_p.nh * ns * hdim;
        for (var k = kq * q; k < (kq + 1u) * q; k = k + 1u) {
            let i = base + k * hdim;
            let v = da * ps_s[i] + dx * ps_b[k];
            ps_s[i] = v;
            if (snap) { ps_snap[so + i] = v; }
            acc = acc + v * ps_c[k];
        }
        ps_part[lid] = acc;
        workgroupBarrier();
        if (kq == 0u) {
            ps_y[b * ps_p.inner + h * hdim + p] = ps_part[p] + ps_part[64u + p] + ps_part[128u + p] + ps_part[192u + p] + dpar * x;
        }
        workgroupBarrier();
    }
}

// ── gated group RMSNorm, workgroup per (group, position) ──
struct PgP { gs: u32, eps: f32, inner: u32, zstride: u32 };
@group(0) @binding(0) var<storage, read>       pg_y  : array<f32>;
@group(0) @binding(1) var<storage, read>       pg_zx : array<f32>;
@group(0) @binding(2) var<storage, read>       pg_w  : array<f32>;
@group(0) @binding(3) var<storage, read_write> pg_o  : array<f32>;
@group(0) @binding(4) var<storage, read>       pg_m  : Meta;
@group(0) @binding(5) var<uniform>             pg_p  : PgP;
var<workgroup> pg_part: array<f32, 256>;
@compute @workgroup_size(256)
fn pb_gnorm(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let b = wid.y;
    if (b >= pg_m.n) { return; }
    let gs = pg_p.gs;
    let base = wid.x * gs;
    let yo = b * pg_p.inner;
    let zo = b * pg_p.zstride;
    var acc = 0.0;
    for (var i = lid; i < gs; i = i + 256u) {
        let z = pg_zx[zo + base + i];
        let v = pg_y[yo + base + i] * (z / (1.0 + exp(-z)));
        acc = acc + v * v;
    }
    pg_part[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { pg_part[lid] = pg_part[lid] + pg_part[lid + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(pg_part[0] / f32(gs) + pg_p.eps);
    for (var i = lid; i < gs; i = i + 256u) {
        let z = pg_zx[zo + base + i];
        let v = pg_y[yo + base + i] * (z / (1.0 + exp(-z)));
        pg_o[yo + base + i] = v * inv * pg_w[base + i];
    }
}

// ── KV rows of the pass at pos0.. ──
struct PkP { w: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       pk_k  : array<f32>;
@group(0) @binding(1) var<storage, read>       pk_v  : array<f32>;
@group(0) @binding(2) var<storage, read_write> pk_kc : array<f32>;
@group(0) @binding(3) var<storage, read_write> pk_vc : array<f32>;
@group(0) @binding(4) var<storage, read>       pk_m  : Meta;
@group(0) @binding(5) var<uniform>             pk_p  : PkP;
@compute @workgroup_size(256)
fn pb_kv_append(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let w = pk_p.w;
    if (i >= pk_m.n * w) { return; }
    pk_kc[pk_m.pos0 * w + i] = pk_k[i];
    pk_vc[pk_m.pos0 * w + i] = pk_v[i];
}

// ── causal GQA over the pass: workgroup (split, kv head × block of 4 q
// heads, position); a lane owns 4 dims of each of the 4 heads, one vec4
// subgroup sum gives a position's 4 scores ──
struct PtP { nh: u32, nkv: u32, hd: u32, chunk: u32, scale: f32, maxsplit: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       pt_q    : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       pt_kc   : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       pt_vc   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pt_part : array<f32>;
@group(0) @binding(4) var<storage, read>       pt_m    : Meta;
@group(0) @binding(5) var<uniform>             pt_p    : PtP;
fn pt_put(base: u32, lane: u32, a: vec4<f32>, m: f32, l: f32) {
    pt_part[base + lane * 4u] = a.x;
    pt_part[base + lane * 4u + 1u] = a.y;
    pt_part[base + lane * 4u + 2u] = a.z;
    pt_part[base + lane * 4u + 3u] = a.w;
    if (lane == 0u) {
        pt_part[base + pt_p.hd] = m;
        pt_part[base + pt_p.hd + 1u] = l;
    }
}
@compute @workgroup_size(256)
fn pb_attn(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32,
           @builtin(subgroup_invocation_id) lane: u32) {
    let s = wid.x;
    let rep = pt_p.nh / pt_p.nkv;
    let blocks = rep / 4u;
    let g = wid.y / blocks;
    let b = wid.z;
    if (b >= pt_m.n) { return; }
    let w = lid / 32u;
    let d4 = pt_p.hd / 4u;
    let npos = pt_m.pos0 + b + 1u;
    let h0 = g * rep + (wid.y % blocks) * 4u;
    let qb = b * pt_p.nh;
    let q0 = pt_q[(qb + h0) * d4 + lane] * pt_p.scale;
    let q1 = pt_q[(qb + h0 + 1u) * d4 + lane] * pt_p.scale;
    let q2 = pt_q[(qb + h0 + 2u) * d4 + lane] * pt_p.scale;
    let q3 = pt_q[(qb + h0 + 3u) * d4 + lane] * pt_p.scale;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var m = vec4<f32>(-1e30);
    var l = vec4<f32>(0.0);
    let start = s * pt_p.chunk;
    let end = min(start + pt_p.chunk, npos);
    for (var j = start + w; j < end; j = j + 8u) {
        let kk = pt_kc[(j * pt_p.nkv + g) * d4 + lane];
        let vv = pt_vc[(j * pt_p.nkv + g) * d4 + lane];
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
    let stride = pt_p.hd + 2u;
    let base = (((b * pt_p.maxsplit + s) * 8u + w) * pt_p.nh + h0) * stride;
    pt_put(base, lane, a0, m.x, l.x);
    pt_put(base + stride, lane, a1, m.y, l.y);
    pt_put(base + 2u * stride, lane, a2, m.z, l.z);
    pt_put(base + 3u * stride, lane, a3, m.w, l.w);
}

struct PmP { nh: u32, hd: u32, splits: u32, maxsplit: u32 };
@group(0) @binding(0) var<storage, read>       pm_part : array<f32>;
@group(0) @binding(1) var<storage, read_write> pm_o    : array<f32>;
@group(0) @binding(2) var<storage, read>       pm_m    : Meta;
@group(0) @binding(3) var<uniform>             pm_p    : PmP;
@compute @workgroup_size(128)
fn pb_attn_merge(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) d: u32) {
    let h = wid.x;
    let b = wid.y;
    if (b >= pm_m.n) { return; }
    let stride = pm_p.hd + 2u;
    let np = pm_p.splits * 8u;
    let pb = b * pm_p.maxsplit * 8u;
    var mx = -1e30;
    for (var p = 0u; p < np; p = p + 1u) {
        mx = max(mx, pm_part[((pb + p) * pm_p.nh + h) * stride + pm_p.hd]);
    }
    var den = 0.0;
    var num = 0.0;
    for (var p = 0u; p < np; p = p + 1u) {
        let base = ((pb + p) * pm_p.nh + h) * stride;
        let f = exp(pm_part[base + pm_p.hd] - mx);
        den = den + pm_part[base + pm_p.hd + 1u] * f;
        num = num + pm_part[base + d] * f;
    }
    if (d < pm_p.hd) { pm_o[(b * pm_p.nh + h) * pm_p.hd + d] = num / den; }
}

// ── router over the pass: workgroup per position ──
struct PrP { n: u32, k: u32, scale: f32, norm: u32 };
@group(0) @binding(0) var<storage, read>       pr_logit : array<f32>;
@group(0) @binding(1) var<storage, read>       pr_bias  : array<f32>;
@group(0) @binding(2) var<storage, read_write> pr_sel   : array<u32>;
@group(0) @binding(3) var<storage, read_write> pr_w     : array<f32>;
@group(0) @binding(4) var<storage, read>       pr_m     : Meta;
@group(0) @binding(5) var<uniform>             pr_p     : PrP;
var<workgroup> pr_choice: array<f32, 256>;
var<workgroup> pr_v: array<f32, 8>;
var<workgroup> pr_i: array<u32, 8>;
@compute @workgroup_size(256)
fn pb_route(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32,
            @builtin(subgroup_invocation_id) lane: u32) {
    let b = wid.x;
    if (b >= pr_m.n) { return; }
    let lo = b * pr_p.n;
    let so = b * pr_p.k;
    var c = -3.0e38;
    if (lid < pr_p.n) {
        c = 1.0 / (1.0 + exp(-pr_logit[lo + lid])) + pr_bias[lid];
    }
    pr_choice[lid] = c;
    let w = lid / 32u;
    for (var t = 0u; t < pr_p.k; t = t + 1u) {
        let v = pr_choice[lid];
        let mxv = subgroupMax(v);
        let idx = subgroupMin(select(0xFFFFFFFFu, lid, v == mxv));
        if (lane == 0u) {
            pr_v[w] = mxv;
            pr_i[w] = idx;
        }
        workgroupBarrier();
        if (lid == 0u) {
            var bv = pr_v[0];
            var bi = pr_i[0];
            for (var s = 1u; s < 8u; s = s + 1u) {
                if (pr_v[s] > bv) { bv = pr_v[s]; bi = pr_i[s]; }
            }
            pr_sel[so + t] = bi;
            pr_choice[bi] = -3.0e38;
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        var sum = 0.0;
        for (var t = 0u; t < pr_p.k; t = t + 1u) {
            sum = sum + 1.0 / (1.0 + exp(-pr_logit[lo + pr_sel[so + t]]));
        }
        var den = 1.0;
        if (pr_p.norm != 0u) { den = sum + 1e-20; }
        for (var t = 0u; t < pr_p.k; t = t + 1u) {
            pr_w[so + t] = (1.0 / (1.0 + exp(-pr_logit[lo + pr_sel[so + t]]))) / den * pr_p.scale;
        }
    }
}

// ── per expert, the pass's (position·k + slot) entries that chose it ──
struct PlP { k: u32, cap: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       pl_sel  : array<u32>;
@group(0) @binding(1) var<storage, read_write> pl_list : array<u32>;
@group(0) @binding(2) var<storage, read_write> pl_cnt  : array<u32>;
@group(0) @binding(3) var<storage, read>       pl_m    : Meta;
@group(0) @binding(4) var<uniform>             pl_p    : PlP;
@compute @workgroup_size(64)
fn pb_lists(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    var c = 0u;
    let total = pl_m.n * pl_p.k;
    for (var j = 0u; j < total; j = j + 1u) {
        if (pl_sel[j] == e) {
            pl_list[e * pl_p.cap + c] = j;
            c = c + 1u;
        }
    }
    pl_cnt[e] = c;
}

// ── q4tp expert bank over the pass: workgroup (row block, expert); each
// subgroup owns four rows and walks the expert's entries. x row of entry j:
// j / xdiv (stride cols); output row j (stride rows); act 1 = relu² ──
// With `remap` the bank holds only some experts: map[layer·n_exp + e] is
// the bank slot or MISS; a missing expert is skipped (the host writes its
// rows). `gpr` is the column count for the q8_2f bank.
struct PqP { rows: u32, gpr: u32, stride_w: u32, act: u32, xdiv: u32, cap: u32, layer: u32, remap: u32 };
@group(0) @binding(0) var<storage, read>       pq_w    : array<u32>;
@group(0) @binding(1) var<storage, read>       pq_w4   : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       pq_x    : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       pq_list : array<u32>;
@group(0) @binding(4) var<storage, read>       pq_cnt  : array<u32>;
@group(0) @binding(5) var<storage, read_write> pq_y    : array<f32>;
@group(0) @binding(6) var<storage, read>       pq_map  : array<u32>;
@group(0) @binding(7) var<uniform>             pq_p    : PqP;
fn pq_slot(e: u32, n_exp: u32) -> u32 {
    if (pq_p.remap == 0u) { return e; }
    return pq_map[pq_p.layer * n_exp + e];
}
fn pq_byte(off: u32) -> u32 { return (pq_w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn pq_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn pq_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return pq_nib(w, 0u) * a.x + pq_nib(w, 4u) * a.y + pq_nib(w, 8u) * a.z + pq_nib(w, 12u) * a.w
         + pq_nib(w, 16u) * b.x + pq_nib(w, 20u) * b.y + pq_nib(w, 24u) * b.z + pq_nib(w, 28u) * b.w;
}
var<workgroup> pq_xs: array<vec4<f32>, 2688>;
// rows a subgroup owns in the expert banks (a workgroup: 8·PQ_RW rows)
const PQ_RW: u32 = 16u;
// up to four entries staged at once, transposed per entry ([e][i][group]):
// every weight word is loaded and unpacked once for the four of them
@compute @workgroup_size(256)
fn pb_q4_bank(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    let e = wid.y;
    let cnt = pq_cnt[e];
    if (cnt == 0u) { return; }
    let ds = pq_slot(e, nwg.y);
    if (ds == 0xFFFFFFFFu) { return; }
    let rows = pq_p.rows;
    let gpr = pq_p.gpr;
    let bw = ds * pq_p.stride_w;
    let params_w = bw + rows * gpr * 4u;
    let codes_b = (bw + rows * gpr * 4u + rows) * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let warp = lid / 32u;
    let c8 = gpr * 8u;
    let es = c8;
    for (var j0 = 0u; j0 < cnt; j0 = j0 + 4u) {
        let ne = min(4u, cnt - j0);
        workgroupBarrier();
        for (var t = lid; t < ne * c8; t = t + 256u) {
            let q = t / c8;
            let i = t - q * c8;
            let ent = pq_list[e * pq_p.cap + j0 + q];
            let xo = (ent / pq_p.xdiv) * c8;
            pq_xs[q * es + (i % 8u) * gpr + i / 8u] = pq_x[xo + i];
        }
        workgroupBarrier();
        for (var rr = 0u; rr < PQ_RW; rr = rr + 1u) {
            let row = (wid.x * 8u + warp) * PQ_RW + rr;
            let live = row < rows;
            let r = select(rows - 1u, row, live);
            let pm = unpack2x16float(pq_w[params_w + r]);
            let cb = codes_b + r * cstride;
            var acc = vec4<f32>(0.0);
            for (var g = lane; g < gpr; g = g + 32u) {
                let bit = g * 5u;
                let off = cb + (bit >> 3u);
                let code = ((pq_byte(off) | (pq_byte(off + 1u) << 8u)) >> (bit & 7u)) & 31u;
                let sc = exp2(pm.x + f32(code) * pm.y);
                let v = pq_w4[bw / 4u + r * gpr + g];
                // the 32 weights of the group, unpacked once
                let w0 = vec4<f32>(pq_nib(v.x, 0u), pq_nib(v.x, 4u), pq_nib(v.x, 8u), pq_nib(v.x, 12u));
                let w1 = vec4<f32>(pq_nib(v.x, 16u), pq_nib(v.x, 20u), pq_nib(v.x, 24u), pq_nib(v.x, 28u));
                let w2 = vec4<f32>(pq_nib(v.y, 0u), pq_nib(v.y, 4u), pq_nib(v.y, 8u), pq_nib(v.y, 12u));
                let w3 = vec4<f32>(pq_nib(v.y, 16u), pq_nib(v.y, 20u), pq_nib(v.y, 24u), pq_nib(v.y, 28u));
                let w4 = vec4<f32>(pq_nib(v.z, 0u), pq_nib(v.z, 4u), pq_nib(v.z, 8u), pq_nib(v.z, 12u));
                let w5 = vec4<f32>(pq_nib(v.z, 16u), pq_nib(v.z, 20u), pq_nib(v.z, 24u), pq_nib(v.z, 28u));
                let w6 = vec4<f32>(pq_nib(v.w, 0u), pq_nib(v.w, 4u), pq_nib(v.w, 8u), pq_nib(v.w, 12u));
                let w7 = vec4<f32>(pq_nib(v.w, 16u), pq_nib(v.w, 20u), pq_nib(v.w, 24u), pq_nib(v.w, 28u));
                for (var q = 0u; q < ne; q = q + 1u) {
                    let xb = q * es + g;
                    let s = dot(w0, pq_xs[xb]) + dot(w1, pq_xs[gpr + xb]) + dot(w2, pq_xs[2u * gpr + xb])
                          + dot(w3, pq_xs[3u * gpr + xb]) + dot(w4, pq_xs[4u * gpr + xb]) + dot(w5, pq_xs[5u * gpr + xb])
                          + dot(w6, pq_xs[6u * gpr + xb]) + dot(w7, pq_xs[7u * gpr + xb]);
                    acc[q] = acc[q] + sc * s;
                }
            }
            let t = vec4<f32>(subgroupAdd(acc.x), subgroupAdd(acc.y), subgroupAdd(acc.z), subgroupAdd(acc.w));
            if (lane == 0u && live) {
                for (var q = 0u; q < ne; q = q + 1u) {
                    let ent = pq_list[e * pq_p.cap + j0 + q];
                    var o = t[q];
                    if (pq_p.act == 1u) { o = max(o, 0.0); o = o * o; }
                    pq_y[ent * rows + row] = o;
                }
            }
        }
    }
}

// ── q8_2f expert bank over the pass (payload [int8 rows·cols][f16 row
// scales][f16 col field]); as `pb_q4_bank`, `gpr` = cols ──
fn pq_cs4(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(pq_w[w]);
    let b = unpack2x16float(pq_w[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(pq_w[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
fn pq_i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    return vec4<f32>(f32((s << 24u) >> 24u), f32((s << 16u) >> 24u), f32((s << 8u) >> 24u), f32(s >> 24u));
}
fn pq_rs(rs0: u32, r: u32) -> f32 {
    let rw = unpack2x16float(pq_w[rs0 + (r >> 1u)]);
    return select(rw.x, rw.y, (r & 1u) == 1u);
}
@compute @workgroup_size(256)
fn pb_q8_bank(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) lane: u32) {
    let e = wid.y;
    let cnt = pq_cnt[e];
    if (cnt == 0u) { return; }
    let ds = pq_slot(e, nwg.y);
    if (ds == 0xFFFFFFFFu) { return; }
    let rows = pq_p.rows;
    let cols = pq_p.gpr;
    let words = cols / 4u;
    let bw = ds * pq_p.stride_w;
    let cs0h = bw * 2u + ((rows * cols) >> 1u) + rows;
    let rs0 = bw + ((rows * cols) >> 2u);
    for (var j = 0u; j < cnt; j = j + 1u) {
        let ent = pq_list[e * pq_p.cap + j];
        let xo = (ent / pq_p.xdiv) * words;
        workgroupBarrier();
        for (var i = lid; i < words; i = i + 256u) {
            pq_xs[i] = pq_x[xo + i] * pq_cs4(cs0h + i * 4u);
        }
        workgroupBarrier();
        for (var rq = 0u; rq < PQ_RW; rq = rq + 4u) {
            let row0 = (wid.x * 8u + lid / 32u) * PQ_RW + rq;
            let b0 = bw + min(row0, rows - 1u) * words;
            let b1 = bw + min(row0 + 1u, rows - 1u) * words;
            let b2 = bw + min(row0 + 2u, rows - 1u) * words;
            let b3 = bw + min(row0 + 3u, rows - 1u) * words;
            var a0 = 0.0;
            var a1 = 0.0;
            var a2 = 0.0;
            var a3 = 0.0;
            for (var k = lane; k < words; k = k + 32u) {
                let x = pq_xs[k];
                a0 = a0 + dot(pq_i8x4(pq_w[b0 + k]), x);
                a1 = a1 + dot(pq_i8x4(pq_w[b1 + k]), x);
                a2 = a2 + dot(pq_i8x4(pq_w[b2 + k]), x);
                a3 = a3 + dot(pq_i8x4(pq_w[b3 + k]), x);
            }
            // keep the vec4 view in the binding set (auto layout): never true
            if (rows == 0xFFFFFFFFu) { a0 = a0 + f32(pq_w4[0].x); }
            let t = vec4<f32>(subgroupAdd(a0), subgroupAdd(a1), subgroupAdd(a2), subgroupAdd(a3));
            if (lane == 0u) {
                for (var r = 0u; r < 4u; r = r + 1u) {
                    let row = row0 + r;
                    if (row < rows) {
                        var o = t[r] * pq_rs(rs0, row);
                        if (pq_p.act == 1u) { o = max(o, 0.0); o = o * o; }
                        pq_y[ent * rows + row] = o;
                    }
                }
            }
        }
    }
}

// ── argmax of every position's logits (verify passes): workgroup per position ──
struct PxP { vocab: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       px_l   : array<f32>;
@group(0) @binding(1) var<storage, read_write> px_ids : array<u32>;
@group(0) @binding(2) var<storage, read>       px_m   : Meta;
@group(0) @binding(3) var<uniform>             px_p   : PxP;
var<workgroup> px_v: array<f32, 256>;
var<workgroup> px_i: array<u32, 256>;
@compute @workgroup_size(256)
fn pb_argmax(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let b = wid.x;
    if (b >= px_m.n) { return; }
    let o = b * px_p.vocab;
    var bv = -3.0e38;
    var bi = 0u;
    for (var i = lid; i < px_p.vocab; i = i + 256u) {
        let v = px_l[o + i];
        if (v > bv) { bv = v; bi = i; }
    }
    px_v[lid] = bv;
    px_i[lid] = bi;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let a = px_v[lid];
            let c = px_v[lid + s];
            if (c > a || (c == a && px_i[lid + s] < px_i[lid])) {
                px_v[lid] = c;
                px_i[lid] = px_i[lid + s];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) { px_ids[b] = px_i[0]; }
}
"#;
