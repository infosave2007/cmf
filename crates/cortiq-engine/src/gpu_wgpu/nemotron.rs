//! Nemotron-H (Nemotron 3.5 Transcribe decoder) device-resident token path.
//!
//! The whole decode step runs in one compute pass: every block's RMSNorm,
//! the Mamba-2 mixer (q8_2f in/out projections, depthwise conv ring, the
//! selective scan over a device-resident state, the gated group RMSNorm),
//! the NoPE GQA attention over a device KV cache (split-K partials + merge),
//! the MoE (f32 router, sigmoid top-k with the selection bias on device,
//! q4tp routed experts read from per-layer banks by the device-chosen index,
//! relu² activations, the q8_2f shared expert), the final norm, the q8_2f
//! head, argmax and the q8_2f embedding of the winner for the next step.
//! Nothing crosses the bus inside a step; `steps` greedy tokens ride one
//! submit and only their ids come back.
//!
//! Kernels reused from the parent: `moe_r4`'s warp-per-row `q82_sg` for
//! every q8_2f projection. New here: the kernels in `NEMO_WGSL`.
//!
//! The frame holds ONE sequence. It is seeded from the host state after a
//! host (or per-op) prefill: conv rings and KV rows verbatim, the scan state
//! transposed to the device layout `[head][state][head_dim]`.

use super::*;

pub(crate) const NEMO_WGSL: &str = r#"
// ── h += o (add 1) or h += so + Σ_s w[s]·eo[s] (add 2, the MoE sum);
// y = rms(h)·w ──
struct ArP { n: u32, add: u32, eps: f32, k: u32 };
@group(0) @binding(0) var<storage, read_write> ar_h  : array<f32>;
@group(0) @binding(1) var<storage, read>       ar_o  : array<f32>;
@group(0) @binding(2) var<storage, read>       ar_w  : array<f32>;
@group(0) @binding(3) var<storage, read_write> ar_y  : array<f32>;
@group(0) @binding(4) var<storage, read>       ar_eo : array<f32>;
@group(0) @binding(5) var<storage, read>       ar_sw : array<f32>;
@group(0) @binding(6) var<uniform>             ar_p  : ArP;
var<workgroup> ar_part: array<f32, 256>;
@compute @workgroup_size(256)
fn nm_add_rms(@builtin(local_invocation_index) lid: u32) {
    let n = ar_p.n;
    var acc = 0.0;
    for (var i = lid; i < n; i = i + 256u) {
        var v = ar_h[i];
        if (ar_p.add == 1u) {
            v = v + ar_o[i];
            ar_h[i] = v;
        } else if (ar_p.add == 2u) {
            var m = ar_o[i];
            for (var s = 0u; s < ar_p.k; s = s + 1u) {
                m = m + ar_sw[s] * ar_eo[s * n + i];
            }
            v = v + m;
            ar_h[i] = v;
        }
        acc = acc + v * v;
    }
    ar_part[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { ar_part[lid] = ar_part[lid] + ar_part[lid + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(ar_part[0] / f32(n) + ar_p.eps);
    for (var i = lid; i < n; i = i + 256u) {
        ar_y[i] = ar_w[i] * (ar_h[i] * inv);
    }
}

// ── Mamba-2 depthwise causal conv (kernel 4) + SiLU, ring [3][cdim] ──
struct CvP { cdim: u32, off: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       cv_zx   : array<f32>;
@group(0) @binding(1) var<storage, read>       cv_w    : array<f32>;
@group(0) @binding(2) var<storage, read>       cv_b    : array<f32>;
@group(0) @binding(3) var<storage, read_write> cv_ring : array<f32>;
@group(0) @binding(4) var<storage, read_write> cv_out  : array<f32>;
@group(0) @binding(5) var<uniform>             cv_p    : CvP;
@compute @workgroup_size(256)
fn nm_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let n = cv_p.cdim;
    if (c >= n) { return; }
    let x = cv_zx[cv_p.off + c];
    let r0 = cv_ring[c];
    let r1 = cv_ring[n + c];
    let r2 = cv_ring[2u * n + c];
    let w = c * 4u;
    let acc = cv_b[c] + cv_w[w] * r0 + cv_w[w + 1u] * r1 + cv_w[w + 2u] * r2 + cv_w[w + 3u] * x;
    cv_ring[c] = r1;
    cv_ring[n + c] = r2;
    cv_ring[2u * n + c] = x;
    cv_out[c] = acc / (1.0 + exp(-acc));
}

// ── Mamba-2 selective scan, one workgroup per head, one lane per head dim.
// State layout [head][state][head_dim]; par = [A(nh) | D(nh) | dt_bias(nh)].
struct ScP { inner: u32, dt_off: u32, ng: u32, ns: u32, hpg: u32, hdim: u32, nh: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       sc_zx  : array<f32>;
@group(0) @binding(1) var<storage, read>       sc_xbc : array<f32>;
@group(0) @binding(2) var<storage, read_write> sc_s   : array<f32>;
@group(0) @binding(3) var<storage, read>       sc_par : array<f32>;
@group(0) @binding(4) var<storage, read_write> sc_y   : array<f32>;
@group(0) @binding(5) var<uniform>             sc_p   : ScP;
var<workgroup> sc_b: array<f32, 256>;
var<workgroup> sc_c: array<f32, 256>;
var<workgroup> sc_part: array<f32, 256>;
@compute @workgroup_size(256)
fn nm_scan(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let ns = sc_p.ns;
    let hdim = sc_p.hdim;
    let p = lid % 64u;
    let kq = lid / 64u;
    let g = h / sc_p.hpg;
    let boff = sc_p.inner + g * ns;
    let coff = sc_p.inner + sc_p.ng * ns + g * ns;
    for (var k = lid; k < ns; k = k + 256u) {
        sc_b[k] = sc_xbc[boff + k];
        sc_c[k] = sc_xbc[coff + k];
    }
    workgroupBarrier();
    let raw = sc_zx[sc_p.dt_off + h] + sc_par[2u * sc_p.nh + h];
    var dt = log(1.0 + exp(raw));
    if (raw > 20.0) { dt = raw; }
    if (raw < -15.0) { dt = exp(raw); }
    let da = exp(dt * sc_par[h]);
    let x = sc_xbc[h * hdim + p];
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

// ── y·silu(z), then RMSNorm over each group of `gs`, ·w ──
struct GqP { gs: u32, eps: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       gq_y  : array<f32>;
@group(0) @binding(1) var<storage, read>       gq_zx : array<f32>;
@group(0) @binding(2) var<storage, read>       gq_w  : array<f32>;
@group(0) @binding(3) var<storage, read_write> gq_o  : array<f32>;
@group(0) @binding(4) var<uniform>             gq_p  : GqP;
var<workgroup> gq_part: array<f32, 256>;
@compute @workgroup_size(256)
fn nm_gnorm(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let gs = gq_p.gs;
    let base = wid.x * gs;
    var acc = 0.0;
    for (var i = lid; i < gs; i = i + 256u) {
        let z = gq_zx[base + i];
        let v = gq_y[base + i] * (z / (1.0 + exp(-z)));
        acc = acc + v * v;
    }
    gq_part[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) { gq_part[lid] = gq_part[lid] + gq_part[lid + s]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(gq_part[0] / f32(gs) + gq_p.eps);
    for (var i = lid; i < gs; i = i + 256u) {
        let z = gq_zx[base + i];
        let v = gq_y[base + i] * (z / (1.0 + exp(-z)));
        gq_o[base + i] = v * inv * gq_w[base + i];
    }
}

// ── KV append at `pos` ──
struct KaP { n: u32, pos: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       ka_k  : array<f32>;
@group(0) @binding(1) var<storage, read>       ka_v  : array<f32>;
@group(0) @binding(2) var<storage, read_write> ka_kc : array<f32>;
@group(0) @binding(3) var<storage, read_write> ka_vc : array<f32>;
@group(0) @binding(4) var<uniform>             ka_p  : KaP;
@compute @workgroup_size(256)
fn nm_kv_append(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ka_p.n) { return; }
    ka_kc[ka_p.pos * ka_p.n + i] = ka_k[i];
    ka_vc[ka_p.pos * ka_p.n + i] = ka_v[i];
}

// ── GQA decode attention, split-K partials. Workgroup (split, kv head);
// eight 32-lane subgroups walk the split's positions; a lane owns 4 dims of
// every q head of the group (rep ≤ 16, hd = 128). Partial per (split,
// subgroup, head): acc[hd], max, sum.
struct AtP { nh: u32, nkv: u32, hd: u32, npos: u32, chunk: u32, scale: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       at_q    : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       at_kc   : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       at_vc   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> at_part : array<f32>;
@group(0) @binding(4) var<uniform>             at_p    : AtP;
@compute @workgroup_size(256)
fn nm_attn(@builtin(workgroup_id) wid: vec3<u32>,
           @builtin(local_invocation_index) lid: u32,
           @builtin(subgroup_invocation_id) lane: u32) {
    let s = wid.x;
    let g = wid.y;
    let w = lid / 32u;
    let rep = at_p.nh / at_p.nkv;
    let d4 = at_p.hd / 4u;
    var qv: array<vec4<f32>, 16>;
    var acc: array<vec4<f32>, 16>;
    var m: array<f32, 16>;
    var l: array<f32, 16>;
    for (var r = 0u; r < rep; r = r + 1u) {
        qv[r] = at_q[(g * rep + r) * d4 + lane] * at_p.scale;
        acc[r] = vec4<f32>(0.0);
        m[r] = -1e30;
        l[r] = 0.0;
    }
    let start = s * at_p.chunk;
    let end = min(start + at_p.chunk, at_p.npos);
    for (var j = start + w; j < end; j = j + 8u) {
        let kk = at_kc[(j * at_p.nkv + g) * d4 + lane];
        let vv = at_vc[(j * at_p.nkv + g) * d4 + lane];
        for (var r = 0u; r < rep; r = r + 1u) {
            let sc = subgroupAdd(dot(qv[r], kk));
            let mn = max(m[r], sc);
            let a = exp(m[r] - mn);
            let pp = exp(sc - mn);
            l[r] = l[r] * a + pp;
            acc[r] = acc[r] * a + pp * vv;
            m[r] = mn;
        }
    }
    let stride = at_p.hd + 2u;
    for (var r = 0u; r < rep; r = r + 1u) {
        let base = ((s * 8u + w) * at_p.nh + g * rep + r) * stride;
        let a4 = acc[r];
        at_part[base + lane * 4u] = a4.x;
        at_part[base + lane * 4u + 1u] = a4.y;
        at_part[base + lane * 4u + 2u] = a4.z;
        at_part[base + lane * 4u + 3u] = a4.w;
        if (lane == 0u) {
            at_part[base + at_p.hd] = m[r];
            at_part[base + at_p.hd + 1u] = l[r];
        }
    }
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

// ── f32 matvec, one 32-lane subgroup per row ──
struct FmP { rows: u32, cols: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       fm_w : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       fm_x : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> fm_y : array<f32>;
@group(0) @binding(3) var<uniform>             fm_p : FmP;
@compute @workgroup_size(256)
fn nm_f32mv(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32,
            @builtin(subgroup_invocation_id) lane: u32) {
    let row = wid.x * 8u + lid / 32u;
    let live = row < fm_p.rows;
    let r = select(fm_p.rows - 1u, row, live);
    let c4 = fm_p.cols / 4u;
    var acc = 0.0;
    for (var k = lane; k < c4; k = k + 32u) {
        acc = acc + dot(fm_w[r * c4 + k], fm_x[k]);
    }
    let t = subgroupAdd(acc);
    if (lane == 0u && live) { fm_y[row] = t; }
}

// ── sigmoid router: top-k by score + bias, weights = renormalized
// unbiased scores × scale ──
struct RtP { n: u32, k: u32, scale: f32, norm: u32 };
@group(0) @binding(0) var<storage, read>       rt_logit : array<f32>;
@group(0) @binding(1) var<storage, read>       rt_bias  : array<f32>;
@group(0) @binding(2) var<storage, read_write> rt_sel   : array<u32>;
@group(0) @binding(3) var<storage, read_write> rt_w     : array<f32>;
@group(0) @binding(4) var<uniform>             rt_p     : RtP;
var<workgroup> rt_choice: array<f32, 256>;
var<workgroup> rt_v: array<f32, 8>;
var<workgroup> rt_i: array<u32, 8>;
@compute @workgroup_size(256)
fn nm_route(@builtin(local_invocation_index) lid: u32,
            @builtin(subgroup_invocation_id) lane: u32) {
    var c = -3.0e38;
    if (lid < rt_p.n) {
        c = 1.0 / (1.0 + exp(-rt_logit[lid])) + rt_bias[lid];
    }
    rt_choice[lid] = c;
    let w = lid / 32u;
    for (var t = 0u; t < rt_p.k; t = t + 1u) {
        let v = rt_choice[lid];
        let m = subgroupMax(v);
        // lowest index holding the maximum within the subgroup
        let idx = subgroupMin(select(0xFFFFFFFFu, lid, v == m));
        if (lane == 0u) {
            rt_v[w] = m;
            rt_i[w] = idx;
        }
        workgroupBarrier();
        if (lid == 0u) {
            var bv = rt_v[0];
            var bi = rt_i[0];
            for (var s = 1u; s < 8u; s = s + 1u) {
                if (rt_v[s] > bv) { bv = rt_v[s]; bi = rt_i[s]; }
            }
            rt_sel[t] = bi;
            rt_choice[bi] = -3.0e38;
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        var sum = 0.0;
        for (var t = 0u; t < rt_p.k; t = t + 1u) {
            sum = sum + 1.0 / (1.0 + exp(-rt_logit[rt_sel[t]]));
        }
        var den = 1.0;
        if (rt_p.norm != 0u) { den = sum + 1e-20; }
        for (var t = 0u; t < rt_p.k; t = t + 1u) {
            rt_w[t] = (1.0 / (1.0 + exp(-rt_logit[rt_sel[t]]))) / den * rt_p.scale;
        }
    }
}

// ── q4tp routed-expert matvec from a per-layer bank: workgroup (row block,
// slot); the slot's expert id comes from the router. Payload per expert:
// [nibbles rows·gpr·16][params (f16 lo, f16 step) rows·4][5-bit codes].
// A 32-lane subgroup owns 4 rows; act 1 = relu². ──
struct EqP { rows: u32, gpr: u32, stride_w: u32, act: u32, x_stride: u32, y_stride: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       eq_w   : array<u32>;
@group(0) @binding(1) var<storage, read>       eq_w4  : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       eq_x   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       eq_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> eq_y   : array<f32>;
@group(0) @binding(5) var<uniform>             eq_p   : EqP;
var<workgroup> eq_xs: array<vec4<f32>, 1024>;
fn eq_byte(off: u32) -> u32 { return (eq_w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
fn eq_nib(w: u32, sh: u32) -> f32 {
    return bitcast<f32>(((w >> sh) & 0xFu) | 0x4B000000u) - 8388616.0;
}
fn eq_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return eq_nib(w, 0u) * a.x + eq_nib(w, 4u) * a.y + eq_nib(w, 8u) * a.z + eq_nib(w, 12u) * a.w
         + eq_nib(w, 16u) * b.x + eq_nib(w, 20u) * b.y + eq_nib(w, 24u) * b.z + eq_nib(w, 28u) * b.w;
}
@compute @workgroup_size(256)
fn nm_q4tp_bank(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32,
                @builtin(subgroup_invocation_id) lane: u32) {
    let slot = wid.y;
    let e = eq_sel[slot];
    let rows = eq_p.rows;
    let gpr = eq_p.gpr;
    let c4 = gpr * 8u;
    let xo = slot * eq_p.x_stride / 4u;
    for (var i = lid; i < c4; i = i + 256u) {
        eq_xs[i] = eq_x[xo + i];
    }
    workgroupBarrier();
    let bw = e * eq_p.stride_w;
    let params_w = bw + rows * gpr * 4u;
    let codes_b = (bw + rows * gpr * 4u + rows) * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let warp = lid / 32u;
    for (var rr = 0u; rr < 4u; rr = rr + 1u) {
        let row = (wid.x * 8u + warp) * 4u + rr;
        let live = row < rows;
        let r = select(rows - 1u, row, live);
        let pm = unpack2x16float(eq_w[params_w + r]);
        let cb = codes_b + r * cstride;
        var acc = 0.0;
        for (var g = lane; g < gpr; g = g + 32u) {
            let bit = g * 5u;
            let off = cb + (bit >> 3u);
            let code = ((eq_byte(off) | (eq_byte(off + 1u) << 8u)) >> (bit & 7u)) & 31u;
            let sc = exp2(pm.x + f32(code) * pm.y);
            let nv = eq_w4[bw / 4u + r * gpr + g];
            let xb = g * 8u;
            let s = eq_dot8(nv.x, eq_xs[xb], eq_xs[xb + 1u]) + eq_dot8(nv.y, eq_xs[xb + 2u], eq_xs[xb + 3u])
                  + eq_dot8(nv.z, eq_xs[xb + 4u], eq_xs[xb + 5u]) + eq_dot8(nv.w, eq_xs[xb + 6u], eq_xs[xb + 7u]);
            acc = acc + sc * s;
        }
        let t = subgroupAdd(acc);
        if (lane == 0u && live) {
            var v = t;
            if (eq_p.act == 1u) { v = max(v, 0.0); v = v * v; }
            eq_y[slot * eq_p.y_stride + row] = v;
        }
    }
}

// ── argmax: stage 1 per 1024-wide block, stage 2 over the blocks ──
struct AxP { n: u32, blocks: u32, slot: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       ax_x   : array<f32>;
@group(0) @binding(1) var<storage, read_write> ax_bv  : array<f32>;
@group(0) @binding(2) var<storage, read_write> ax_bi  : array<u32>;
@group(0) @binding(3) var<storage, read_write> ax_ids : array<u32>;
@group(0) @binding(4) var<uniform>             ax_p   : AxP;
var<workgroup> ax_v: array<f32, 256>;
var<workgroup> ax_i: array<u32, 256>;
fn ax_tree(lid: u32) {
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
}
@compute @workgroup_size(256)
fn nm_argmax1(@builtin(workgroup_id) wid: vec3<u32>,
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
    ax_tree(lid);
    if (lid == 0u) {
        ax_bv[wid.x] = ax_v[0];
        ax_bi[wid.x] = ax_i[0];
    }
    // keep the binding set total (auto layout): never true
    if (wid.x == 0xFFFFFFFFu) { ax_ids[0] = 0u; }
}
@compute @workgroup_size(256)
fn nm_argmax2(@builtin(local_invocation_index) lid: u32) {
    var bv = -3.0e38;
    var bi = 0u;
    for (var i = lid; i < ax_p.blocks; i = i + 256u) {
        let v = ax_bv[i];
        if (v > bv || (v == bv && ax_bi[i] < bi)) { bv = v; bi = ax_bi[i]; }
    }
    if (lid == 0xFFFFFFFFu) { bv = ax_x[0]; }
    ax_v[lid] = bv;
    ax_i[lid] = bi;
    workgroupBarrier();
    ax_tree(lid);
    if (lid == 0u) { ax_ids[ax_p.slot] = ax_i[0]; }
}

// ── q8_2f embedding row of ids[slot] into h ──
struct EmP { rows: u32, cols: u32, slot: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       em_w   : array<u32>;
@group(0) @binding(1) var<storage, read>       em_ids : array<u32>;
@group(0) @binding(2) var<storage, read_write> em_h   : array<f32>;
@group(0) @binding(3) var<uniform>             em_p   : EmP;
fn em_f16(byte_off: u32) -> f32 {
    let w = unpack2x16float(em_w[byte_off >> 2u]);
    return select(w.x, w.y, (byte_off & 2u) != 0u);
}
@compute @workgroup_size(256)
fn nm_embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= em_p.cols) { return; }
    let id = em_ids[em_p.slot];
    let off = id * em_p.cols + c;
    let q = i32(((em_w[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu) << 24u) >> 24u;
    let body = em_p.rows * em_p.cols;
    let rs = em_f16(body + id * 2u);
    let cf = em_f16(body + em_p.rows * 2u + c * 2u);
    em_h[c] = f32(q) * rs * cf;
}
"#;

struct Pipes {
    /// `moe_r4`'s `q82_sg` with a relu² epilogue (uniform word 5 = 1)
    q82: wgpu::ComputePipeline,
    add_rms: wgpu::ComputePipeline,
    conv: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    gnorm: wgpu::ComputePipeline,
    kv_append: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
    attn_merge: wgpu::ComputePipeline,
    f32mv: wgpu::ComputePipeline,
    route: wgpu::ComputePipeline,
    q4bank: wgpu::ComputePipeline,
    argmax1: wgpu::ComputePipeline,
    argmax2: wgpu::ComputePipeline,
    embed: wgpu::ComputePipeline,
}

static PIPES: std::sync::OnceLock<Option<Pipes>> = std::sync::OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES.get_or_init(|| build_pipes(c)).as_ref()
}

fn build_pipes(c: &Ctx) -> Option<Pipes> {
    if !c.device.features().contains(wgpu::Features::SUBGROUP)
        || c.adapter_info.subgroup_min_size != 32
        || c.adapter_info.subgroup_max_size != 32
    {
        tracing::warn!("nemotron frame: needs 32-lane subgroups");
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-nemotron"),
        source: wgpu::ShaderSource::Wgsl(NEMO_WGSL.into()),
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
    let q82_src = moe_r4::q82sg_wgsl()
        .replace("qs_y{M}", "qs_y{M}")
        .replace("= t0 * select(rw.x, rw.y, (r0 & 1u) == 1u);", "= qs_act(t0 * select(rw.x, rw.y, (r0 & 1u) == 1u));")
        .replace("= t1 * select(rw.x, rw.y, (r1 & 1u) == 1u);", "= qs_act(t1 * select(rw.x, rw.y, (r1 & 1u) == 1u));")
        .replace("fn qs_i8x4(w: u32) -> vec4<f32> {", "fn qs_act(v: f32) -> f32 {\n    if (qs_p._a == 1u) { let r = max(v, 0.0); return r * r; }\n    return v;\n}\nfn qs_i8x4(w: u32) -> vec4<f32> {");
    if !q82_src.contains("qs_act(t0") || !q82_src.contains("qs_act(t1") {
        tracing::warn!("nemotron frame: q82_sg source changed shape; relu² epilogue unavailable");
        return None;
    }
    let q82_module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-nemotron-q82"),
        source: wgpu::ShaderSource::Wgsl(q82_src.into()),
    });
    let q82 = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nm-q82"),
            layout: None,
            module: &q82_module,
            entry_point: Some("q82_sg"),
            compilation_options: pco(),
            cache: c.pipeline_cache.as_ref(),
        });
    let p = Pipes {
        q82,
        add_rms: pipe("nm_add_rms"),
        conv: pipe("nm_conv"),
        scan: pipe("nm_scan"),
        gnorm: pipe("nm_gnorm"),
        kv_append: pipe("nm_kv_append"),
        attn: pipe("nm_attn"),
        attn_merge: pipe("nm_attn_merge"),
        f32mv: pipe("nm_f32mv"),
        route: pipe("nm_route"),
        q4bank: pipe("nm_q4tp_bank"),
        argmax1: pipe("nm_argmax1"),
        argmax2: pipe("nm_argmax2"),
        embed: pipe("nm_embed"),
    };
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("nemotron frame: module rejected ({e})");
        return None;
    }
    Some(p)
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
        /// q4tp payloads, one per expert
        up: Vec<&'a [u8]>,
        down: Vec<&'a [u8]>,
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
        up: wgpu::Buffer,
        up_stride: usize,
        down: wgpu::Buffer,
        down_stride: usize,
        sh_up: wgpu::Buffer,
        sh_down: wgpu::Buffer,
    },
}

/// Scratch activations of one step.
struct Scratch {
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
    logits_r: wgpu::Buffer,
    sel: wgpu::Buffer,
    sw: wgpu::Buffer,
    mid: wgpu::Buffer,
    eo: wgpu::Buffer,
    smid: wgpu::Buffer,
    so: wgpu::Buffer,
    logits: wgpu::Buffer,
    bv: wgpu::Buffer,
    bi: wgpu::Buffer,
    ids: wgpu::Buffer,
    dummy: wgpu::Buffer,
    stage: wgpu::Buffer,
}

/// One dispatch of the step plan. `Fixed` ones are bound once at build;
/// the position-dependent attention core and the head tail are bound per
/// step.
enum Op {
    Fixed {
        pipe: &'static wgpu::ComputePipeline,
        bg: wgpu::BindGroup,
        groups: (u32, u32),
        tag: &'static str,
    },
    /// kv append + split-K attention + merge for attention block `block`
    Attn { block: usize },
}

pub struct NemoFrame {
    c: &'static Ctx,
    p: &'static Pipes,
    blocks: Vec<DevBlock>,
    plan: Vec<Op>,
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
    cap: usize,
    max_parts: usize,
    /// Batched-prefill buffers and plan, built on first use.
    pf: Option<Prefill>,
    /// Positions already in the device state.
    pub pos: usize,
    pub bytes: u64,
}

const STAGE_IDS: usize = 64;

/// `CMF_NEMO_PROF=1`: host encode and device time of every frame submit.
fn prof() -> bool {
    std::env::var("CMF_NEMO_PROF").as_deref() == Ok("1")
}

/// `CMF_NEMO_SKIP=tag,tag`: leave these dispatch classes out of the step
/// (garbage output; the time difference is the class's cost). Tags: rms
/// in_proj conv scan gnorm out_proj qkv attn o_proj router route exp_up
/// exp_down sh_up sh_down final.
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
        if d.hd != 128
            || d.nh / d.nkv > 16
            || d.m_head_dim != 64
            || d.m_state > 256
            || d.n_exp > 256
            || d.top_k > 16
        {
            tracing::warn!("nemotron frame: geometry outside the kernels");
            return None;
        }
        let q82 = moe_r4::pipes(c)?;
        q82.q82sg.as_ref()?;
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
        let inner = d.m_heads * d.m_head_dim;
        let cdim = inner + 2 * d.m_groups * d.m_state;
        let kvw = d.nkv * d.hd;
        let mut blocks = Vec::with_capacity(d.blocks.len());
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
                    sh_up,
                    sh_down,
                } => {
                    let bank = |label: &str, mats: &[&[u8]]| -> Option<(wgpu::Buffer, usize)> {
                        let one = mats[0].len();
                        if mats.iter().any(|m| m.len() != one) {
                            return None;
                        }
                        let stride = one.div_ceil(16) * 16;
                        let b = alloc(label, stride * mats.len());
                        for (i, m) in mats.iter().enumerate() {
                            put(&b, (i * stride) as u64, m);
                        }
                        Some((b, stride / 4))
                    };
                    let (upb, up_stride) = bank("nm-up-bank", ups)?;
                    let (dnb, down_stride) = bank("nm-down-bank", down)?;
                    DevBlock::Moe {
                        norm: up("nm-norm", &f32b(norm)),
                        router: up("nm-router", &f32b(router)),
                        bias: up("nm-bias", &f32b(bias)),
                        up: upb,
                        up_stride,
                        down: dnb,
                        down_stride,
                        sh_up: up("nm-shup", sh_up),
                        sh_down: up("nm-shdown", sh_down),
                    }
                }
            });
        }
        let embed = up("nm-embed", d.embed);
        let lm_head = up("nm-head", d.lm_head);
        let final_norm = up("nm-fnorm", &f32b(d.final_norm));
        c.queue.submit(None);
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        let bytes = total.get();
        let in_rows = inner + cdim + d.m_heads;
        let max_parts = ATTN_SPLITS * 8;
        let blocks_ax = d.vocab.div_ceil(1024);
        let s = Scratch {
            h: zeros("nm-h", d.hidden),
            o: zeros("nm-o", d.hidden),
            xn: zeros("nm-xn", d.hidden),
            zx: zeros("nm-zx", in_rows),
            xbc: zeros("nm-xbc", cdim),
            y: zeros("nm-y", inner.max(d.nh * d.hd)),
            yn: zeros("nm-yn", inner),
            q: zeros("nm-q", d.nh * d.hd),
            k: zeros("nm-k", kvw),
            v: zeros("nm-v", kvw),
            att: zeros("nm-att", d.nh * d.hd),
            part: zeros("nm-part", max_parts * d.nh * (d.hd + 2)),
            logits_r: zeros("nm-rl", d.n_exp),
            sel: zeros("nm-sel", d.top_k),
            sw: zeros("nm-sw", d.top_k),
            mid: zeros("nm-mid", d.top_k * d.moe_inter),
            eo: zeros("nm-eo", d.top_k * d.hidden),
            smid: zeros("nm-smid", d.shared_inter),
            so: zeros("nm-so", d.hidden),
            logits: zeros("nm-logits", d.vocab),
            bv: zeros("nm-bv", blocks_ax),
            bi: zeros("nm-bi", blocks_ax),
            ids: zeros("nm-ids", STAGE_IDS),
            dummy: zeros("nm-dummy", 4),
            stage: c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nm-stage"),
                size: (d.vocab.max(STAGE_IDS) * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        let mut f = Self {
            c,
            p,
            blocks,
            plan: Vec::new(),
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
            cap,
            max_parts,
            pos: 0,
            bytes,
        };
        f.plan = f.build_plan();
        Some(f)
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
        if pos > self.cap {
            return false;
        }
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
        u: [u32; 8],
        groups: (u32, u32),
    ) -> Op {
        let ub = self.u(u);
        Op::Fixed {
            pipe,
            bg: self.bind(pipe, bufs, &ub),
            groups,
            tag,
        }
    }

    fn q82op(
        &self,
        tag: &'static str,
        x: &wgpu::Buffer,
        cols: usize,
        mats: &[(&wgpu::Buffer, &wgpu::Buffer, usize)],
    ) -> Op {
        self.q82op_act(tag, x, cols, mats, 0)
    }

    /// q8_2f matvec(s) of one input (up to three matrices), `act` 1 = relu².
    fn q82op_act(
        &self,
        tag: &'static str,
        x: &wgpu::Buffer,
        cols: usize,
        mats: &[(&wgpu::Buffer, &wgpu::Buffer, usize)],
        act: u32,
    ) -> Op {
        let ps = moe_r4::pipes(self.c).expect("q82 pipes");
        let pipe: &'static wgpu::ComputePipeline = &self.p.q82;
        let rows_max = mats.iter().map(|m| m.2).max().unwrap_or(0);
        let rpw: usize = if rows_max >= 32768 { 4 } else { 2 };
        let per = 8 * rpw;
        let r = |i: usize| mats.get(i).map_or(0, |m| m.2);
        let u = self.u([
            r(0) as u32,
            r(1) as u32,
            r(2) as u32,
            cols as u32,
            rpw as u32,
            act,
            0,
            0,
        ]);
        let w = |i: usize| mats.get(i).map_or(mats[0].0, |m| m.0);
        let y = |i: usize| mats.get(i).map_or(&ps.dummy_y, |m| m.1);
        let bg = self.bind(pipe, &[w(0), w(1), w(2), x, y(0), y(1), y(2)], &u);
        let groups: usize = (0..3).map(|i| r(i).div_ceil(per)).sum();
        Op::Fixed {
            pipe,
            bg,
            groups: (groups as u32, 1),
            tag,
        }
    }

    /// The position-independent dispatch list of one token step, ending
    /// with the final norm and the head (argmax/embed are per step).
    fn build_plan(&self) -> Vec<Op> {
        let s = &self.s;
        let p = self.p;
        let hid = self.hidden;
        let eps = self.eps.to_bits();
        let inner = self.m_heads * self.m_head_dim;
        let cdim = inner + 2 * self.m_groups * self.m_state;
        let in_rows = inner + cdim + self.m_heads;
        let kvw = self.nkv * self.hd;
        let mut plan = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            let norm = match b {
                DevBlock::Mamba { norm, .. }
                | DevBlock::Attn { norm, .. }
                | DevBlock::Moe { norm, .. } => norm,
            };
            // add the previous mixer's output (the MoE sum is fused here)
            let prev_moe = bi > 0 && matches!(self.blocks[bi - 1], DevBlock::Moe { .. });
            let add = if bi == 0 {
                0
            } else if prev_moe {
                2
            } else {
                1
            };
            let o_src = match add {
                0 => &s.dummy,
                2 => &s.so,
                _ => &s.o,
            };
            plan.push(self.fixed(
                "rms",
                &p.add_rms,
                &[&s.h, o_src, norm, &s.xn, &s.eo, &s.sw],
                [hid as u32, add, eps, self.top_k as u32, 0, 0, 0, 0],
                (1, 1),
            ));
            match b {
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
                    plan.push(self.q82op("in_proj", &s.xn, hid, &[(in_proj, &s.zx, in_rows)]));
                    plan.push(self.fixed(
                        "conv",
                        &p.conv,
                        &[&s.zx, conv_w, conv_b, ring, &s.xbc],
                        [cdim as u32, inner as u32, 0, 0, 0, 0, 0, 0],
                        (cdim.div_ceil(256) as u32, 1),
                    ));
                    plan.push(self.fixed(
                        "scan",
                        &p.scan,
                        &[&s.zx, &s.xbc, state, par, &s.y],
                        [
                            inner as u32,
                            (inner + cdim) as u32,
                            self.m_groups as u32,
                            self.m_state as u32,
                            (self.m_heads / self.m_groups) as u32,
                            self.m_head_dim as u32,
                            self.m_heads as u32,
                            0,
                        ],
                        (self.m_heads as u32, 1),
                    ));
                    plan.push(self.fixed(
                        "gnorm",
                        &p.gnorm,
                        &[&s.y, &s.zx, gnorm, &s.yn],
                        [(inner / self.m_groups) as u32, eps, 0, 0, 0, 0, 0, 0],
                        (self.m_groups as u32, 1),
                    ));
                    plan.push(self.q82op("out_proj", &s.yn, inner, &[(out_proj, &s.o, hid)]));
                }
                DevBlock::Attn { q, k, v, o, .. } => {
                    plan.push(self.q82op(
                        "qkv",
                        &s.xn,
                        hid,
                        &[(q, &s.q, self.nh * self.hd), (k, &s.k, kvw), (v, &s.v, kvw)],
                    ));
                    plan.push(Op::Attn { block: bi });
                    plan.push(self.q82op("o_proj", &s.att, self.nh * self.hd, &[(o, &s.o, hid)]));
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
                    ..
                } => {
                    plan.push(self.fixed(
                        "router",
                        &p.f32mv,
                        &[router, &s.xn, &s.logits_r],
                        [self.n_exp as u32, hid as u32, 0, 0, 0, 0, 0, 0],
                        (self.n_exp.div_ceil(8) as u32, 1),
                    ));
                    plan.push(self.fixed(
                        "route",
                        &p.route,
                        &[&s.logits_r, bias, &s.sel, &s.sw],
                        [
                            self.n_exp as u32,
                            self.top_k as u32,
                            self.route_scale.to_bits(),
                            self.norm_topk as u32,
                            0,
                            0,
                            0,
                            0,
                        ],
                        (1, 1),
                    ));
                    let k = self.top_k as u32;
                    plan.push(self.fixed(
                        "exp_up",
                        &p.q4bank,
                        &[up, up, &s.xn, &s.sel, &s.mid],
                        [
                            self.moe_inter as u32,
                            (hid / 32) as u32,
                            *up_stride as u32,
                            1,
                            0,
                            self.moe_inter as u32,
                            0,
                            0,
                        ],
                        (self.moe_inter.div_ceil(32) as u32, k),
                    ));
                    plan.push(self.fixed(
                        "exp_down",
                        &p.q4bank,
                        &[down, down, &s.mid, &s.sel, &s.eo],
                        [
                            hid as u32,
                            (self.moe_inter / 32) as u32,
                            *down_stride as u32,
                            0,
                            self.moe_inter as u32,
                            hid as u32,
                            0,
                            0,
                        ],
                        (hid.div_ceil(32) as u32, k),
                    ));
                    plan.push(self.q82op_act(
                        "sh_up",
                        &s.xn,
                        hid,
                        &[(sh_up, &s.smid, self.shared_inter)],
                        1,
                    ));
                    plan.push(self.q82op(
                        "sh_down",
                        &s.smid,
                        self.shared_inter,
                        &[(sh_down, &s.so, hid)],
                    ));
                }
            }
        }
        let last_moe = matches!(self.blocks.last(), Some(DevBlock::Moe { .. }));
        let (add, o_src) = if last_moe { (2, &s.so) } else { (1, &s.o) };
        plan.push(self.fixed(
            "final",
            &p.add_rms,
            &[&s.h, o_src, &self.final_norm, &s.xn, &s.eo, &s.sw],
            [hid as u32, add, eps, self.top_k as u32, 0, 0, 0, 0],
            (1, 1),
        ));
        plan
    }

    fn run(
        &self,
        pass: &mut wgpu::ComputePass,
        pipe: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        u: [u32; 8],
        groups: (u32, u32),
    ) {
        let ub = self.u(u);
        let bg = self.bind(pipe, bufs, &ub);
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }

    /// Encode one token step at `pos`; the input row must already be in `h`.
    /// With `slot`, the step ends in argmax → `ids[slot]` → embed into `h`.
    fn encode_step(
        &self,
        pass: &mut wgpu::ComputePass,
        pos: usize,
        slot: Option<usize>,
        logits: bool,
    ) {
        let s = &self.s;
        let p = self.p;
        let kvw = self.nkv * self.hd;
        let skip = skip_tags();
        for op in &self.plan {
            match op {
                Op::Fixed { tag, .. } if skip.contains(tag) => {}
                Op::Attn { .. } if skip.contains(&"attn") => {}
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
                    self.run(
                        pass,
                        &p.kv_append,
                        &[&s.k, &s.v, kc, vc],
                        [kvw as u32, pos as u32, 0, 0, 0, 0, 0, 0],
                        (kvw.div_ceil(256) as u32, 1),
                    );
                    let npos = pos + 1;
                    let chunk = npos.div_ceil(ATTN_SPLITS).max(64);
                    let splits = npos.div_ceil(chunk);
                    let scale = (1.0 / (self.hd as f32).sqrt()).to_bits();
                    self.run(
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
        if !logits && slot.is_none() {
            return;
        }
        if let Op::Fixed {
            pipe, bg, groups, ..
        } = &self.head_op()
        {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        if let Some(slot) = slot {
            let hid = self.hidden;
            let blocks = self.vocab.div_ceil(1024);
            self.run(
                pass,
                &p.argmax1,
                &[&s.logits, &s.bv, &s.bi, &s.ids],
                [self.vocab as u32, blocks as u32, slot as u32, 0, 0, 0, 0, 0],
                (blocks as u32, 1),
            );
            self.run(
                pass,
                &p.argmax2,
                &[&s.logits, &s.bv, &s.bi, &s.ids],
                [self.vocab as u32, blocks as u32, slot as u32, 0, 0, 0, 0, 0],
                (1, 1),
            );
            self.run(
                pass,
                &p.embed,
                &[&self.embed, &s.ids, &s.h],
                [self.vocab as u32, hid as u32, slot as u32, 0, 0, 0, 0, 0],
                (hid.div_ceil(256) as u32, 1),
            );
        }
    }

    fn head_op(&self) -> Op {
        self.q82op(
            "head",
            &self.s.xn,
            self.hidden,
            &[(&self.lm_head, &self.s.logits, self.vocab)],
        )
    }

    fn read(
        &self,
        mut enc: wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        n_words: usize,
    ) -> Option<Vec<u32>> {
        let bytes = (n_words * 4) as u64;
        enc.copy_buffer_to_buffer(src, 0, &self.s.stage, 0, bytes);
        self.c.queue.submit(Some(enc.finish()));
        let slice = self.s.stage.slice(..bytes);
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
            self.s.stage.unmap();
            return None;
        }
        let out = {
            let data = slice.get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, u32>(&data[..bytes as usize]).to_vec()
        };
        self.s.stage.unmap();
        Some(out)
    }

    /// Greedy decode: `row` (the next input embedding) at `self.pos`, then
    /// `steps − 1` more positions fed by their own argmax, all in ONE submit.
    /// Returns the `steps` argmax ids; the state advances by `steps`.
    pub fn greedy(&mut self, row: &[f32], steps: usize) -> Option<Vec<u32>> {
        if steps == 0
            || steps > STAGE_IDS
            || self.pos + steps > self.cap
            || row.len() != self.hidden
        {
            return None;
        }
        let t0 = std::time::Instant::now();
        self.c
            .queue
            .write_buffer(&self.s.h, 0, bytemuck::cast_slice(row));
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
                self.encode_step(&mut pass, self.pos + t, Some(t), false);
            }
        }
        let t_enc = t0.elapsed();
        let ids = self.read(enc, &self.s.ids, steps)?;
        if prof() {
            eprintln!(
                "nemotron frame: {steps} steps at pos {}: encode {:.2} ms, submit+wait {:.2} ms, {:.2} ms/token",
                self.pos,
                t_enc.as_secs_f64() * 1e3,
                (t0.elapsed() - t_enc).as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3 / steps as f64
            );
        }
        self.pos += steps;
        Some(ids)
    }

    /// One position from a host row, logits read back.
    pub fn logits(&mut self, row: &[f32], out: &mut Vec<f32>) -> bool {
        if self.pos + 1 > self.cap || row.len() != self.hidden {
            return false;
        }
        self.c
            .queue
            .write_buffer(&self.s.h, 0, bytemuck::cast_slice(row));
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
            self.encode_step(&mut pass, self.pos, None, true);
        }
        let Some(words) = self.read(enc, &self.s.logits, self.vocab) else {
            return false;
        };
        out.clear();
        out.extend(words.iter().map(|w| f32::from_bits(*w)));
        self.pos += 1;
        true
    }

    /// Positions without readback (prefill through the frame): `rows` are
    /// the input embeddings, one step per row, one submit.
    pub fn feed(&mut self, rows: &[f32]) -> bool {
        let n = rows.len() / self.hidden;
        if n == 0 || self.pos + n > self.cap {
            return false;
        }
        // one submit per row: the host row is written into `h` before each
        for t in 0..n {
            self.c.queue.write_buffer(
                &self.s.h,
                0,
                bytemuck::cast_slice(&rows[t * self.hidden..(t + 1) * self.hidden]),
            );
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
                self.encode_step(&mut pass, self.pos + t, None, false);
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
        use super::nemotron_prefill::{PB_MAX, PB_WGSL};
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
            relu2: pipe("pb_relu2"),
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
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        };
        let eps = self.eps.to_bits();
        let mut plan: Vec<Op> = Vec::new();
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
                        &[&pf.zx, conv_w, conv_b, ring, &pf.xbc, &pf.meta],
                        [cdim as u32, inner as u32, in_rows as u32, 0, 0, 0, 0, 0],
                        (cdim.div_ceil(256) as u32, 1),
                    ));
                    plan.push(fixed(
                        "scan",
                        &pp.scan,
                        &[&pf.zx, &pf.xbc, state, par, &pf.y, &pf.meta],
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
                    plan.push(fixed(
                        "route",
                        &pp.lists,
                        &[&pf.sel, &pf.list, &pf.cnt, &pf.meta],
                        [k as u32, cap as u32, 0, 0, 0, 0, 0, 0],
                        (self.n_exp.div_ceil(64) as u32, 1),
                    ));
                    plan.push(fixed(
                        "exp_up",
                        &pp.bank,
                        &[up, up, &pf.xn, &pf.list, &pf.cnt, &pf.mid],
                        [
                            self.moe_inter as u32,
                            (hid / 32) as u32,
                            *up_stride as u32,
                            1,
                            k as u32,
                            cap as u32,
                            0,
                            0,
                        ],
                        (self.moe_inter.div_ceil(32) as u32, self.n_exp as u32),
                    ));
                    plan.push(fixed(
                        "exp_down",
                        &pp.bank,
                        &[down, down, &pf.mid, &pf.list, &pf.cnt, &pf.eo],
                        [
                            hid as u32,
                            (self.moe_inter / 32) as u32,
                            *down_stride as u32,
                            0,
                            1,
                            cap as u32,
                            0,
                            0,
                        ],
                        (hid.div_ceil(32) as u32, self.n_exp as u32),
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
        Some(Prefill {
            pp,
            pf,
            plan,
            maxsplit,
        })
    }

    /// Prefill `rows` (input embeddings) through the batched kernels,
    /// `PB_MAX` positions per pass; no readback. False when the batched
    /// path is unavailable (nothing was encoded) or the device failed.
    pub fn feed_batched(&mut self, rows: &[f32]) -> bool {
        use super::nemotron_prefill::PB_MAX;
        let hid = self.hidden;
        let n_all = rows.len() / hid;
        if n_all == 0 || self.pos + n_all > self.cap {
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
        let pfx = self.pf.as_ref().unwrap();
        let skip = skip_tags();
        let kvw = self.nkv * self.hd;
        let mut done = 0;
        while done < n_all {
            let n = (n_all - done).min(PB_MAX);
            let pos0 = self.pos + done;
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
            let mut enc = c
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pb") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pb"),
                    timestamp_writes: None,
                });
                for op in &pfx.plan {
                    match op {
                        Op::Fixed { tag, .. } if skip.contains(tag) => {}
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
                            let bg =
                                self.bind(&pfx.pp.attn, &[&pf.q, kc, vc, &pf.part, &pf.meta], &ub);
                            pass.set_pipeline(&pfx.pp.attn);
                            pass.set_bind_group(0, &bg, &[]);
                            // grid z: the pass's positions
                            pass.dispatch_workgroups(splits as u32, self.nkv as u32, n as u32);
                            let _ = kvw;
                            self.run(
                                &mut pass,
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
            }
            c.queue.submit(Some(enc.finish()));
            done += n;
        }
        if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return false;
        }
        if prof() {
            eprintln!(
                "nemotron prefill: {n_all} positions in {} passes, {:.1} ms ({:.2} ms/position)",
                n_all.div_ceil(PB_MAX),
                t0.elapsed().as_secs_f64() * 1e3,
                t0.elapsed().as_secs_f64() * 1e3 / n_all as f64
            );
        }
        self.pos += n_all;
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
    relu2: wgpu::ComputePipeline,
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
}

struct Prefill {
    pp: &'static PfPipes,
    pf: PfBufs,
    plan: Vec<Op>,
    maxsplit: usize,
}
