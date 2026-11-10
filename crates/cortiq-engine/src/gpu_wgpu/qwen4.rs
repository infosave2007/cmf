//! Qwen3.8-Flash-Next (`qwen4_exp`) device-resident token path.
//!
//! The host-owned runtime in `qwen4_exp.rs` runs the always-active skeleton
//! (hyper-connections, GDN, QSA, PLE, routers, lm_head) on the CPU and asks
//! the card only for the resident routed experts, one frame per layer with a
//! full readback. That is ~3 GB of q8_2f weight traffic per token on the host
//! memory bus plus 48 synchronous frames: the measured 7 tok/s.
//!
//! This module keeps the whole layer on the device: the four-stream hyper
//! state lives in VRAM for the token, every projection reads resident
//! weights, the GDN recurrent state and the QSA key/value caches are device
//! buffers, and the only thing that crosses the bus per layer is the routed
//! cold-expert list (a few words) and the MoE input (one hidden vector) the
//! CPU needs to complete the experts the arena does not hold. The exact CPU
//! completion and the global segmented expert arena are the same ones the
//! host path already uses, so the arithmetic contract does not change.
//!
//! Kernels reused from the parent module: q8_2f and f32 matvec, the token
//! graph's GDN conv/step/norm, `attn_rope_qkn` (q/k norm + partial RoPE +
//! gate split), `kv_append`, the DSV4 indexer scores and `top_k_index`, the
//! DSV4 `moe_route` (Qwen flag) and the global q2tp/q4tp expert kernels.
//! New here: group RMSNorm over the hyper streams, an f16 matvec for the
//! hyper-connection and router weights, the HC fold/inject, the PLE gate and
//! dilated conv, the QSA compressed-key builder, index list and GQA sparse
//! attention with the sigmoid output gate.

use super::*;
use cortiq_core::{CmfModel, TensorDtype};
use std::sync::Arc;

/// WGSL for the kernels this stack adds on top of the main module.
pub(crate) const QWEN4_WGSL: &str = r#"
enable wgpu_binding_array;
// ── group RMSNorm: `groups` rows of `n`, o = x·rsqrt(mean(x²)+eps)·(1+w) ──
struct GnP { groups: u32, n: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       gn_x : array<f32>;
@group(0) @binding(1) var<storage, read>       gn_w : array<f32>;
@group(0) @binding(2) var<storage, read_write> gn_o : array<f32>;
@group(0) @binding(3) var<uniform>             gn_p : GnP;
var<workgroup> gn_part: array<f32, 256>;
@compute @workgroup_size(256)
fn q4_group_rmsnorm(@builtin(workgroup_id) wid: vec3<u32>,
                    @builtin(local_invocation_index) lid: u32) {
    let g = wid.x;
    if (g >= gn_p.groups) { return; }
    let n = gn_p.n;
    let base = g * n;
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= n) { break; }
        let v = gn_x[base + i];
        acc = acc + v * v;
        i = i + 256u;
    }
    gn_part[lid] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { gn_part[lid] = gn_part[lid] + gn_part[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let inv = inverseSqrt(gn_part[0] / f32(n) + gn_p.eps);
    i = lid;
    loop {
        if (i >= n) { break; }
        gn_o[base + i] = gn_x[base + i] * inv * (1.0 + gn_w[base + i]);
        i = i + 256u;
    }
}

// ── f16 matvec: y[r] = Σ_i W[r,i]·x[i], W row-major f16 pairs in u32 ──
// cols must be even (every projection here is a multiple of 32).
struct HmP { cols: u32, rows: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       hm_w : array<u32>;
@group(0) @binding(1) var<storage, read>       hm_x : array<f32>;
@group(0) @binding(2) var<storage, read_write> hm_y : array<f32>;
@group(0) @binding(3) var<uniform>             hm_p : HmP;
var<workgroup> hm_part: array<f32, 64>;
@compute @workgroup_size(64)
fn q4_f16_matvec(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    if (row >= hm_p.rows) { return; }
    let cols = hm_p.cols;
    let base = row * cols;
    var acc = 0.0;
    var i = lid * 2u;
    loop {
        if (i >= cols) { break; }
        let w2 = unpack2x16float(hm_w[(base + i) >> 1u]);
        acc = acc + w2.x * hm_x[i] + w2.y * hm_x[i + 1u];
        i = i + 128u;
    }
    hm_part[lid] = acc;
    workgroupBarrier();
    var s = 32u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { hm_part[lid] = hm_part[lid] + hm_part[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) { hm_y[row] = hm_part[0]; }
}

// ── two f16 matrices on one input in one dispatch: rows_a of A, then rows_b of B ──
// act bit 1: ya = silu(ya·inv); act bit 2: yb = σ(yb). B's rows land at yb[yb_off..].
struct PmP { cols: u32, rows_a: u32, rows_b: u32, act: u32, inv: f32, yb_off: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       pm_a  : array<u32>;
@group(0) @binding(1) var<storage, read>       pm_b  : array<u32>;
@group(0) @binding(2) var<storage, read>       pm_x  : array<f32>;
@group(0) @binding(3) var<storage, read_write> pm_ya : array<f32>;
@group(0) @binding(4) var<storage, read_write> pm_yb : array<f32>;
@group(0) @binding(5) var<uniform>             pm_p  : PmP;
var<workgroup> pm_part: array<f32, 64>;
@compute @workgroup_size(64)
fn q4_f16_matvec2(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let cols = pm_p.cols;
    let is_b = row >= pm_p.rows_a;
    if (is_b && row - pm_p.rows_a >= pm_p.rows_b) { return; }
    let r = select(row, row - pm_p.rows_a, is_b);
    let base = r * cols;
    var acc = 0.0;
    var i = lid * 2u;
    loop {
        if (i >= cols) { break; }
        var w2: vec2<f32>;
        if (is_b) { w2 = unpack2x16float(pm_b[(base + i) >> 1u]); }
        else { w2 = unpack2x16float(pm_a[(base + i) >> 1u]); }
        acc = acc + w2.x * pm_x[i] + w2.y * pm_x[i + 1u];
        i = i + 128u;
    }
    pm_part[lid] = acc;
    workgroupBarrier();
    var s = 32u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { pm_part[lid] = pm_part[lid] + pm_part[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) {
        var v = pm_part[0];
        if (is_b) {
            if ((pm_p.act & 2u) != 0u) { v = 1.0 / (1.0 + exp(-v)); }
            pm_yb[pm_p.yb_off + r] = v;
        } else {
            if ((pm_p.act & 1u) != 0u) { let t = v * pm_p.inv; v = t / (1.0 + exp(-t)); }
            pm_ya[r] = v;
        }
    }
}

// ── hyper-connection up-projection and fold in one: out[d] = Σ_s σ(up[s,d]·low)·normed[s,d]·inv ──
// One workgroup per d, lanes over the low-rank axis, the hc streams in sequence.
struct UfP { hc: u32, hidden: u32, low: u32, inv: f32 };
@group(0) @binding(0) var<storage, read>       uf_w      : array<u32>;
@group(0) @binding(1) var<storage, read>       uf_low    : array<f32>;
@group(0) @binding(2) var<storage, read>       uf_normed : array<f32>;
@group(0) @binding(3) var<storage, read_write> uf_out    : array<f32>;
@group(0) @binding(4) var<uniform>             uf_p      : UfP;
var<workgroup> uf_part: array<f32, 64>;
@compute @workgroup_size(64)
fn q4_hc_upfold(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let d = wid.x;
    if (d >= uf_p.hidden) { return; }
    let low = uf_p.low;
    var acc = 0.0;
    for (var st = 0u; st < uf_p.hc; st = st + 1u) {
        let base = (st * uf_p.hidden + d) * low;
        var part = 0.0;
        var j = lid * 2u;
        loop {
            if (j >= low) { break; }
            let w2 = unpack2x16float(uf_w[(base + j) >> 1u]);
            part = part + w2.x * uf_low[j] + w2.y * uf_low[j + 1u];
            j = j + 128u;
        }
        uf_part[lid] = part;
        workgroupBarrier();
        var s = 32u;
        loop {
            if (s == 0u) { break; }
            if (lid < s) { uf_part[lid] = uf_part[lid] + uf_part[lid + s]; }
            workgroupBarrier();
            s = s >> 1u;
        }
        let m = uf_part[0];
        workgroupBarrier();
        acc = acc + (1.0 / (1.0 + exp(-m))) * uf_normed[st * uf_p.hidden + d] * uf_p.inv;
    }
    if (lid == 0u) { uf_out[d] = acc; }
}

// ── v = silu(v·inv), in place ──
struct SsP { n: u32, inv: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read_write> ss_v : array<f32>;
@group(0) @binding(1) var<uniform>             ss_p : SsP;
@compute @workgroup_size(256)
fn q4_silu_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= ss_p.n) { return; }
    let v = ss_v[i] * ss_p.inv;
    ss_v[i] = v / (1.0 + exp(-v));
}

// ── hyper-connection fold: out[d] = Σ_s sigmoid(mix[s,d])·normed[s,d]·inv ──
struct HfP { hc: u32, hidden: u32, inv: f32, _b: u32 };
@group(0) @binding(0) var<storage, read>       hf_mix    : array<f32>;
@group(0) @binding(1) var<storage, read>       hf_normed : array<f32>;
@group(0) @binding(2) var<storage, read_write> hf_out    : array<f32>;
@group(0) @binding(3) var<uniform>             hf_p      : HfP;
@compute @workgroup_size(256)
fn q4_hc_fold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= hf_p.hidden) { return; }
    var acc = 0.0;
    for (var s = 0u; s < hf_p.hc; s = s + 1u) {
        let o = s * hf_p.hidden + d;
        let m = hf_mix[o];
        acc = acc + (1.0 / (1.0 + exp(-m))) * hf_normed[o] * hf_p.inv;
    }
    hf_out[d] = acc;
}

// ── inject a block into the hyper streams: h[s,d] += 2·σ(w[s]·inv)·(blk[d] + cold[d] + blk2[d]) ──
// use: bit 1 = the host's cold completion, bit 2 = the card's own cold pass.
struct InP { hc: u32, hidden: u32, inv: f32, use_cold: u32 };
@group(0) @binding(0) var<storage, read_write> in_h    : array<f32>;
@group(0) @binding(1) var<storage, read>       in_blk  : array<f32>;
@group(0) @binding(2) var<storage, read>       in_w    : array<f32>;
@group(0) @binding(3) var<storage, read>       in_cold : array<f32>;
@group(0) @binding(4) var<uniform>             in_p    : InP;
@group(0) @binding(5) var<storage, read>       in_blk2 : array<f32>;
@compute @workgroup_size(256)
fn q4_inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = in_p.hc * in_p.hidden;
    if (i >= total) { return; }
    let s = i / in_p.hidden;
    let d = i - s * in_p.hidden;
    let w = 2.0 / (1.0 + exp(-in_w[s] * in_p.inv));
    var b = in_blk[d];
    if ((in_p.use_cold & 1u) != 0u) { b = b + in_cold[d]; }
    if ((in_p.use_cold & 2u) != 0u) { b = b + in_blk2[d]; }
    in_h[i] = in_h[i] + w * b;
}

// ── PLE gate: per stream g = σ(sign(dot)·sqrt(max(|dot|,1e-6))), out = g·value ──
struct PgP { hc: u32, hidden: u32, inv_sqrt: f32, _b: u32 };
@group(0) @binding(0) var<storage, read>       pg_key : array<f32>;
@group(0) @binding(1) var<storage, read>       pg_q   : array<f32>;
@group(0) @binding(2) var<storage, read>       pg_val : array<f32>;
@group(0) @binding(3) var<storage, read_write> pg_out : array<f32>;
@group(0) @binding(4) var<uniform>             pg_p   : PgP;
var<workgroup> pg_part: array<f32, 256>;
@compute @workgroup_size(256)
fn q4_ple_gate(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let s = wid.x;
    if (s >= pg_p.hc) { return; }
    let n = pg_p.hidden;
    let off = s * n;
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= n) { break; }
        acc = acc + pg_key[off + i] * pg_q[off + i];
        i = i + 256u;
    }
    pg_part[lid] = acc;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { pg_part[lid] = pg_part[lid] + pg_part[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let dot = pg_part[0] * pg_p.inv_sqrt;
    var sg = 1.0;
    if (dot < 0.0) { sg = -1.0; }
    let root = sg * sqrt(max(abs(dot), 1e-6));
    let g = 1.0 / (1.0 + exp(-root));
    i = lid;
    loop {
        if (i >= n) { break; }
        pg_out[off + i] = g * pg_val[i];
        i = i + 256u;
    }
}

// ── PLE dilated depthwise conv over the normalized history ring, + gated, into the hyper state ──
struct PcP { width: u32, kernel: u32, dil: u32, cap: u32, head: u32, rows: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       pc_norm  : array<f32>;
@group(0) @binding(1) var<storage, read>       pc_hist  : array<f32>;
@group(0) @binding(2) var<storage, read>       pc_taps  : array<f32>;
@group(0) @binding(3) var<storage, read>       pc_gated : array<f32>;
@group(0) @binding(4) var<storage, read_write> pc_h     : array<f32>;
@group(0) @binding(5) var<uniform>             pc_p     : PcP;
@compute @workgroup_size(256)
fn q4_ple_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= pc_p.width) { return; }
    let kk = pc_p.kernel;
    var sum = pc_taps[c * kk + kk - 1u] * pc_norm[c];
    for (var tap = 0u; tap + 1u < kk; tap = tap + 1u) {
        let lag = (kk - 1u - tap) * pc_p.dil;
        if (lag <= pc_p.rows) {
            let slot = (pc_p.head + pc_p.cap - lag) % pc_p.cap;
            sum = sum + pc_taps[c * kk + tap] * pc_hist[slot * pc_p.width + c];
        }
    }
    let v = sum / (1.0 + exp(-sum)) + pc_gated[c];
    pc_h[c] = pc_h[c] + v;
}

// ── QSA indexer: the compressed key of one completed block ──
// mean of `cr` raw keys, per-head RMSNorm (1+w), partial RoPE at the block start.
struct BkP { cr: u32, idim: u32, block: u32, rd: u32, eps: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       bk_raw  : array<f32>;
@group(0) @binding(1) var<storage, read>       bk_w    : array<f32>;
@group(0) @binding(2) var<storage, read>       bk_invf : array<f32>;
@group(0) @binding(3) var<storage, read_write> bk_out  : array<f32>;
@group(0) @binding(4) var<uniform>             bk_p    : BkP;
var<workgroup> bk_k: array<f32, 256>;
var<workgroup> bk_red: array<f32, 256>;
@compute @workgroup_size(256)
fn q4_qsa_block_key(@builtin(local_invocation_index) lid: u32) {
    let d = lid;
    let idim = bk_p.idim;
    var v = 0.0;
    if (d < idim) {
        for (var t = 0u; t < bk_p.cr; t = t + 1u) {
            v = v + bk_raw[(bk_p.block * bk_p.cr + t) * idim + d] / f32(bk_p.cr);
        }
    }
    bk_red[lid] = v * v;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { bk_red[lid] = bk_red[lid] + bk_red[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let inv = inverseSqrt(bk_red[0] / f32(idim) + bk_p.eps);
    if (d < idim) { v = v * inv * (1.0 + bk_w[d]); }
    bk_k[lid] = v;
    workgroupBarrier();
    let hlf = bk_p.rd / 2u;
    let ob = bk_p.block * idim;
    if (d < hlf) {
        let ang = f32(bk_p.block * bk_p.cr) * bk_invf[d];
        let x1 = bk_k[d];
        let x2 = bk_k[d + hlf];
        let ca = cos(ang);
        let sa = sin(ang);
        bk_out[ob + d] = x1 * ca - x2 * sa;
        bk_out[ob + d + hlf] = x2 * ca + x1 * sa;
    } else if (d >= bk_p.rd && d < idim) {
        bk_out[ob + d] = bk_k[d];
    }
}

// ── QSA attended-position list: the kept blocks' positions, then the open tail ──
struct IbP { keep: u32, cr: u32, complete: u32, npos: u32 };
@group(0) @binding(0) var<storage, read>       ib_pick : array<u32>;
@group(0) @binding(1) var<storage, read_write> ib_idx  : array<u32>;
@group(0) @binding(2) var<uniform>             ib_p    : IbP;
@compute @workgroup_size(256)
fn q4_qsa_idx_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let nsel = ib_p.keep * ib_p.cr;
    let tail = ib_p.npos - ib_p.complete * ib_p.cr;
    if (j >= nsel + tail) { return; }
    if (j < nsel) {
        ib_idx[j] = ib_pick[j / ib_p.cr] * ib_p.cr + (j % ib_p.cr);
    } else {
        ib_idx[j] = ib_p.complete * ib_p.cr + (j - nsel);
    }
}

// ── QSA grouped sparse attention over the index list, sigmoid output gate ──
// One workgroup per query head; K/V caches are [nkv][cap][hd].
struct QaP { nh: u32, hd: u32, m: u32, scale: f32, groups: u32, cap: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       qa_q    : array<f32>;
@group(0) @binding(1) var<storage, read>       qa_k    : array<f32>;
@group(0) @binding(2) var<storage, read>       qa_v    : array<f32>;
@group(0) @binding(3) var<storage, read>       qa_idx  : array<u32>;
@group(0) @binding(4) var<storage, read>       qa_gate : array<f32>;
@group(0) @binding(5) var<storage, read_write> qa_out  : array<f32>;
@group(0) @binding(6) var<uniform>             qa_p    : QaP;
var<workgroup> qa_red: array<f32, 256>;
var<workgroup> qa_w: array<f32, 2112>;
var<workgroup> qa_qs: array<f32, 256>;
@compute @workgroup_size(256)
fn q4_qsa_attend(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    if (h >= qa_p.nh) { return; }
    let hd = qa_p.hd;
    let m = qa_p.m;
    let kbase = (h / qa_p.groups) * qa_p.cap * hd;
    if (lid < hd) { qa_qs[lid] = qa_q[h * hd + lid]; }
    workgroupBarrier();
    var mx = -3.0e38;
    var t = lid;
    loop {
        if (t >= m) { break; }
        let p = qa_idx[t];
        var d = 0.0;
        let kb = kbase + p * hd;
        for (var k = 0u; k < hd; k = k + 1u) { d = d + qa_qs[k] * qa_k[kb + k]; }
        let sc = d * qa_p.scale;
        qa_w[t] = sc;
        mx = max(mx, sc);
        t = t + 256u;
    }
    qa_red[lid] = mx;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { qa_red[lid] = max(qa_red[lid], qa_red[lid + s]); }
        workgroupBarrier();
        s = s >> 1u;
    }
    let mval = qa_red[0];
    workgroupBarrier();
    var den = 0.0;
    t = lid;
    loop {
        if (t >= m) { break; }
        let w = exp(qa_w[t] - mval);
        qa_w[t] = w;
        den = den + w;
        t = t + 256u;
    }
    qa_red[lid] = den;
    workgroupBarrier();
    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { qa_red[lid] = qa_red[lid] + qa_red[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let inv = 1.0 / max(qa_red[0], 1.17549435e-38);
    var k = lid;
    loop {
        if (k >= hd) { break; }
        var acc = 0.0;
        for (var i = 0u; i < m; i = i + 1u) {
            acc = acc + qa_w[i] * qa_v[kbase + qa_idx[i] * hd + k];
        }
        let g = qa_gate[h * hd + k];
        qa_out[h * hd + k] = acc * inv * (1.0 / (1.0 + exp(-g)));
        k = k + 256u;
    }
}

// ── GDN output: per-head RMSNorm · w · σ(z), in place (Qwen3.8 gates with a sigmoid, not SiLU) ──
struct GoP { nv: u32, dv: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> go_o    : array<f32>;
@group(0) @binding(1) var<storage, read>       go_z    : array<f32>;
@group(0) @binding(2) var<storage, read>       go_norm : array<f32>;
@group(0) @binding(3) var<uniform>             go_p    : GoP;
var<workgroup> go_red: array<f32, 256>;
@compute @workgroup_size(256)
fn q4_gdn_norm(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    if (h >= go_p.nv) { return; }
    let dv = go_p.dv;
    let base = h * dv;
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= dv) { break; }
        let v = go_o[base + i];
        acc = acc + v * v;
        i = i + 256u;
    }
    go_red[lid] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { go_red[lid] = go_red[lid] + go_red[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let inv = 1.0 / sqrt(go_red[0] / f32(dv) + go_p.eps);
    i = lid;
    loop {
        if (i >= dv) { break; }
        let zz = go_z[base + i];
        go_o[base + i] = go_o[base + i] * inv * go_norm[i] * (1.0 / (1.0 + exp(-zz)));
        i = i + 256u;
    }
}

// ── chain gating: a layer's indirect dispatch sizes are its template, or zero once a miss happened ──
struct GtP { li: u32, slots: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       gt_tpl  : array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> gt_live : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       gt_miss : array<u32>;
@group(0) @binding(3) var<uniform>             gt_p    : GtP;
@compute @workgroup_size(64)
fn q4_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sl = gid.x;
    if (sl >= gt_p.slots) { return; }
    let i = gt_p.li * gt_p.slots + sl;
    if (gt_miss[0] != 0u) { gt_live[i] = vec4<u32>(0u, 0u, 0u, 0u); } else { gt_live[i] = gt_tpl[i]; }
}

// ── after a route: any cold winner raises the miss flag for the layers that follow ──
struct MsP { top_k: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       ms_cold : array<u32>;
@group(0) @binding(1) var<storage, read_write> ms_miss : array<u32>;
@group(0) @binding(2) var<uniform>             ms_p    : MsP;
@compute @workgroup_size(1)
fn q4_miss() {
    var hit = 0u;
    for (var j = 0u; j < ms_p.top_k; j = j + 1u) {
        if (ms_cold[2u * j] != 0xFFFFFFFFu) { hit = 1u; }
    }
    if (hit != 0u) { ms_miss[0] = 1u; }
}

// ── MTP input: R[s,d] = h_s[d] + e[d] for the hc streams (h_s from four per-stream projections) ──
struct MfP { hc: u32, hidden: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       mf_h0 : array<f32>;
@group(0) @binding(1) var<storage, read>       mf_h1 : array<f32>;
@group(0) @binding(2) var<storage, read>       mf_h2 : array<f32>;
@group(0) @binding(3) var<storage, read>       mf_h3 : array<f32>;
@group(0) @binding(4) var<storage, read>       mf_e  : array<f32>;
@group(0) @binding(5) var<storage, read_write> mf_r  : array<f32>;
@group(0) @binding(6) var<uniform>             mf_p  : MfP;
@compute @workgroup_size(256)
fn q4_mtp_fuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= mf_p.hc * mf_p.hidden) { return; }
    let s = i / mf_p.hidden;
    let d = i - s * mf_p.hidden;
    var h = 0.0;
    if (s == 0u) { h = mf_h0[d]; } else if (s == 1u) { h = mf_h1[d]; }
    else if (s == 2u) { h = mf_h2[d]; } else { h = mf_h3[d]; }
    mf_r[i] = h + mf_e[d];
}

// ── the shared expert's gate, as the route kernel wants it: σ(x) bit-cast into forced[index] ──
struct SbP { index: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       sb_in  : array<f32>;
@group(0) @binding(1) var<storage, read_write> sb_out : array<u32>;
@group(0) @binding(2) var<uniform>             sb_p   : SbP;
@compute @workgroup_size(1)
fn q4_sigmoid_bits() {
    sb_out[sb_p.index] = bitcast<u32>(1.0 / (1.0 + exp(-sb_in[0])));
}
"#;

/// Token-wide kernels: one dispatch covers every token slot of a frame.
/// Per-token rows live in one buffer at a fixed stride (in elements, from
/// the uniform), so a frame of T tokens costs the dispatches of one.
pub(crate) const QWEN4T_WGSL: &str = r#"
// ── group RMSNorm over T rows: grid (groups, nt) ──
struct TgP { groups: u32, n: u32, eps: f32, xs: u32, os: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       tg_x : array<f32>;
@group(0) @binding(1) var<storage, read>       tg_w : array<f32>;
@group(0) @binding(2) var<storage, read_write> tg_o : array<f32>;
@group(0) @binding(3) var<uniform>             tg_p : TgP;
var<workgroup> tg_part: array<f32, 256>;
@compute @workgroup_size(256)
fn q4t_group_rmsnorm(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(local_invocation_index) lid: u32) {
    let g = wid.x;
    let t = wid.y;
    if (g >= tg_p.groups) { return; }
    let n = tg_p.n;
    let bx = t * tg_p.xs + g * n;
    let bo = t * tg_p.os + g * n;
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= n) { break; }
        let v = tg_x[bx + i];
        acc = acc + v * v;
        i = i + 256u;
    }
    tg_part[lid] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { tg_part[lid] = tg_part[lid] + tg_part[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let inv = inverseSqrt(tg_part[0] / f32(n) + tg_p.eps);
    i = lid;
    loop {
        if (i >= n) { break; }
        tg_o[bo + i] = tg_x[bx + i] * inv * (1.0 + tg_w[g * n + i]);
        i = i + 256u;
    }
}

// ── two f16 matrices over T inputs: one workgroup per output row, the
// weights read once, the eight token accumulators in two vec4s (a
// runtime-indexed local array would spill). Lane walk, add order and
// reduction tree are q4_f16_matvec2's, so row t=0 is that kernel's bit
// for bit. Rows past `nt` are read (the buffers hold TMAX rows) and their
// results dropped. ──
struct TpP { cols: u32, rows_a: u32, rows_b: u32, act: u32, inv: f32, yb_off: u32, nt: u32, xs: u32,
             yas: u32, ybs: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32 };
@group(0) @binding(0) var<storage, read>       tp_a  : array<u32>;
@group(0) @binding(1) var<storage, read>       tp_b  : array<u32>;
@group(0) @binding(2) var<storage, read>       tp_x  : array<f32>;
@group(0) @binding(3) var<storage, read_write> tp_ya : array<f32>;
@group(0) @binding(4) var<storage, read_write> tp_yb : array<f32>;
@group(0) @binding(5) var<uniform>             tp_p  : TpP;
var<workgroup> tp_lo: array<vec4<f32>, 64>;
var<workgroup> tp_hi: array<vec4<f32>, 64>;
fn tp_x4(o: u32, xs: u32) -> vec4<f32> {
    return vec4<f32>(tp_x[o], tp_x[xs + o], tp_x[2u * xs + o], tp_x[3u * xs + o]);
}
@compute @workgroup_size(64)
fn q4t_f16_pair(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let cols = tp_p.cols;
    let is_b = row >= tp_p.rows_a;
    if (is_b && row - tp_p.rows_a >= tp_p.rows_b) { return; }
    let r = select(row, row - tp_p.rows_a, is_b);
    let base = r * cols;
    let nt = tp_p.nt;
    let xs = tp_p.xs;
    let hi_on = nt > 4u;
    var lo = vec4<f32>(0.0);
    var hi = vec4<f32>(0.0);
    var i = lid * 2u;
    loop {
        if (i >= cols) { break; }
        var w2: vec2<f32>;
        if (is_b) { w2 = unpack2x16float(tp_b[(base + i) >> 1u]); }
        else { w2 = unpack2x16float(tp_a[(base + i) >> 1u]); }
        lo = lo + w2.x * tp_x4(i, xs) + w2.y * tp_x4(i + 1u, xs);
        if (hi_on) {
            hi = hi + w2.x * tp_x4(4u * xs + i, xs) + w2.y * tp_x4(4u * xs + i + 1u, xs);
        }
        i = i + 128u;
    }
    tp_lo[lid] = lo;
    tp_hi[lid] = hi;
    workgroupBarrier();
    var s = 32u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) {
            tp_lo[lid] = tp_lo[lid] + tp_lo[lid + s];
            tp_hi[lid] = tp_hi[lid] + tp_hi[lid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) {
        let r0 = tp_lo[0];
        let r1 = tp_hi[0];
        for (var t = 0u; t < nt; t = t + 1u) {
            var v = r0[t & 3u];
            if (t >= 4u) { v = r1[t & 3u]; }
            if (is_b) {
                if ((tp_p.act & 2u) != 0u) { v = 1.0 / (1.0 + exp(-v)); }
                tp_yb[t * tp_p.ybs + tp_p.yb_off + r] = v;
            } else {
                if ((tp_p.act & 1u) != 0u) { let z = v * tp_p.inv; v = z / (1.0 + exp(-z)); }
                tp_ya[t * tp_p.yas + r] = v;
            }
        }
    }
}

// ── hyper-connection up-projection and fold over T tokens ──
struct TuP { hc: u32, hidden: u32, low: u32, inv: f32, nt: u32, ls: u32, ns: u32, os: u32 };
@group(0) @binding(0) var<storage, read>       tu_w      : array<u32>;
@group(0) @binding(1) var<storage, read>       tu_low    : array<f32>;
@group(0) @binding(2) var<storage, read>       tu_normed : array<f32>;
@group(0) @binding(3) var<storage, read_write> tu_out    : array<f32>;
@group(0) @binding(4) var<uniform>             tu_p      : TuP;
var<workgroup> tu_lo: array<vec4<f32>, 64>;
var<workgroup> tu_hi: array<vec4<f32>, 64>;
fn tu_l4(o: u32, ls: u32) -> vec4<f32> {
    return vec4<f32>(tu_low[o], tu_low[ls + o], tu_low[2u * ls + o], tu_low[3u * ls + o]);
}
fn tu_n4(o: u32, ns: u32) -> vec4<f32> {
    return vec4<f32>(tu_normed[o], tu_normed[ns + o], tu_normed[2u * ns + o], tu_normed[3u * ns + o]);
}
@compute @workgroup_size(64)
fn q4t_hc_upfold(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) lid: u32) {
    let d = wid.x;
    if (d >= tu_p.hidden) { return; }
    let low = tu_p.low;
    let nt = tu_p.nt;
    let ls = tu_p.ls;
    let ns = tu_p.ns;
    let hi_on = nt > 4u;
    var alo = vec4<f32>(0.0);
    var ahi = vec4<f32>(0.0);
    for (var st = 0u; st < tu_p.hc; st = st + 1u) {
        let base = (st * tu_p.hidden + d) * low;
        var plo = vec4<f32>(0.0);
        var phi = vec4<f32>(0.0);
        var j = lid * 2u;
        loop {
            if (j >= low) { break; }
            let w2 = unpack2x16float(tu_w[(base + j) >> 1u]);
            plo = plo + w2.x * tu_l4(j, ls) + w2.y * tu_l4(j + 1u, ls);
            if (hi_on) {
                phi = phi + w2.x * tu_l4(4u * ls + j, ls) + w2.y * tu_l4(4u * ls + j + 1u, ls);
            }
            j = j + 128u;
        }
        tu_lo[lid] = plo;
        tu_hi[lid] = phi;
        workgroupBarrier();
        var s = 32u;
        loop {
            if (s == 0u) { break; }
            if (lid < s) {
                tu_lo[lid] = tu_lo[lid] + tu_lo[lid + s];
                tu_hi[lid] = tu_hi[lid] + tu_hi[lid + s];
            }
            workgroupBarrier();
            s = s >> 1u;
        }
        let mlo = tu_lo[0];
        let mhi = tu_hi[0];
        let no = st * tu_p.hidden + d;
        alo = alo + (1.0 / (1.0 + exp(-mlo))) * tu_n4(no, ns) * tu_p.inv;
        if (hi_on) {
            ahi = ahi + (1.0 / (1.0 + exp(-mhi))) * tu_n4(4u * ns + no, ns) * tu_p.inv;
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        for (var t = 0u; t < nt; t = t + 1u) {
            var v = alo[t & 3u];
            if (t >= 4u) { v = ahi[t & 3u]; }
            tu_out[t * tu_p.os + d] = v;
        }
    }
}

// ── q8_2f matvec over T inputs: four rows a workgroup, a weight word read
// once for every token, the token accumulators in two vec4s. Lane order and
// reduction tree are q8_2f_matvec4's (row t=0 is its bit for bit).
// Word-aligned rows only (cols % 16 == 0; the caller checks). ──
struct TqP { ngrp: u32, rows: u32, cols: u32, nt: u32, xs4: u32, ys: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       tq_w : array<u32>;
@group(0) @binding(1) var<storage, read>       tq_x : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> tq_y : array<f32>;
@group(0) @binding(3) var<uniform>             tq_p : TqP;
var<workgroup> tq_lo: array<vec4<f32>, 256>;
var<workgroup> tq_hi: array<vec4<f32>, 256>;
fn tq_i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    let b0 = (s << 24u) >> 24u;
    let b1 = (s << 16u) >> 24u;
    let b2 = (s <<  8u) >> 24u;
    let b3 =  s          >> 24u;
    return vec4<f32>(f32(b0), f32(b1), f32(b2), f32(b3));
}
fn tq_f16x4(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(tq_w[w]);
    let b = unpack2x16float(tq_w[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(tq_w[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}
@compute @workgroup_size(256)
fn q4t_q82_matvec(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(num_workgroups) nwg: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
    let rows = tq_p.rows;
    let ngrp = tq_p.ngrp;
    let nt = tq_p.nt;
    let xs4 = tq_p.xs4;
    let hi_on = nt > 4u;
    let qbytes = rows * tq_p.cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    let sub = lid >> 6u;
    let l = lid & 63u;
    let blocks = (rows + 3u) / 4u;
    var wb = wid.x;
    loop {
        if (wb >= blocks) { break; }
        let row = wb * 4u + sub;
        var lo = vec4<f32>(0.0);
        var hi = vec4<f32>(0.0);
        if (row < rows) {
            let roww = row * ngrp;
            var i = l;
            loop {
                if (i >= ngrp) { break; }
                let wq = tq_i8x4(tq_w[roww + i]);
                let cs = tq_f16x4(cs0h + i * 4u);
                lo.x = lo.x + dot(wq, tq_x[i] * cs);
                lo.y = lo.y + dot(wq, tq_x[xs4 + i] * cs);
                lo.z = lo.z + dot(wq, tq_x[2u * xs4 + i] * cs);
                lo.w = lo.w + dot(wq, tq_x[3u * xs4 + i] * cs);
                if (hi_on) {
                    hi.x = hi.x + dot(wq, tq_x[4u * xs4 + i] * cs);
                    hi.y = hi.y + dot(wq, tq_x[5u * xs4 + i] * cs);
                    hi.z = hi.z + dot(wq, tq_x[6u * xs4 + i] * cs);
                    hi.w = hi.w + dot(wq, tq_x[7u * xs4 + i] * cs);
                }
                i = i + 64u;
            }
        }
        tq_lo[lid] = lo;
        tq_hi[lid] = hi;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) {
                tq_lo[lid] = tq_lo[lid] + tq_lo[lid + stride];
                tq_hi[lid] = tq_hi[lid] + tq_hi[lid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (l == 0u && row < rows) {
            let rw = unpack2x16float(tq_w[rs0 + (row >> 1u)]);
            var sc = rw.x;
            if ((row & 1u) == 1u) { sc = rw.y; }
            let r0 = tq_lo[lid];
            let r1 = tq_hi[lid];
            for (var t = 0u; t < nt; t = t + 1u) {
                var v = r0[t & 3u];
                if (t >= 4u) { v = r1[t & 3u]; }
                tq_y[t * tq_p.ys + row] = v * sc;
            }
        }
        workgroupBarrier();
        wb = wb + nwg.x;
    }
}

// ── inject over T rows: h[t,s,d] += 2·σ(w[t,s]·inv)·(blk + cold·bit1 + blk2·bit2) ──
struct TiP { hc: u32, hidden: u32, inv: f32, use_cold: u32, hs: u32, bs: u32, ws: u32, cs: u32,
             b2s: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32, _g: u32 };
@group(0) @binding(0) var<storage, read_write> ti_h    : array<f32>;
@group(0) @binding(1) var<storage, read>       ti_blk  : array<f32>;
@group(0) @binding(2) var<storage, read>       ti_w    : array<f32>;
@group(0) @binding(3) var<storage, read>       ti_cold : array<f32>;
@group(0) @binding(4) var<uniform>             ti_p    : TiP;
@group(0) @binding(5) var<storage, read>       ti_blk2 : array<f32>;
// word 0: flags OR-ed into use_cold (the pending inject's finalize writes it)
@group(0) @binding(6) var<storage, read>       ti_f    : array<u32>;
@compute @workgroup_size(256)
fn q4t_inject(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let i = wid.x * 256u + lid;
    let t = wid.y;
    let total = ti_p.hc * ti_p.hidden;
    if (i >= total) { return; }
    let s = i / ti_p.hidden;
    let d = i - s * ti_p.hidden;
    let use_cold = ti_p.use_cold | ti_f[0];
    let w = 2.0 / (1.0 + exp(-ti_w[t * ti_p.ws + s] * ti_p.inv));
    var b = ti_blk[t * ti_p.bs + d];
    if ((use_cold & 1u) != 0u) { b = b + ti_cold[t * ti_p.cs + d]; }
    if ((use_cold & 2u) != 0u) { b = b + ti_blk2[t * ti_p.b2s + d]; }
    let o = t * ti_p.hs + i;
    ti_h[o] = ti_h[o] + w * b;
}

// ── GDN gated RMSNorm over T rows: grid (nv, nt) ──
struct TnP { nv: u32, dv: u32, eps: f32, os: u32, zs: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> tn_o    : array<f32>;
@group(0) @binding(1) var<storage, read>       tn_z    : array<f32>;
@group(0) @binding(2) var<storage, read>       tn_norm : array<f32>;
@group(0) @binding(3) var<uniform>             tn_p    : TnP;
var<workgroup> tn_red: array<f32, 256>;
@compute @workgroup_size(256)
fn q4t_gdn_norm(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let t = wid.y;
    if (h >= tn_p.nv) { return; }
    let dv = tn_p.dv;
    let bo = t * tn_p.os + h * dv;
    let bz = t * tn_p.zs + h * dv;
    var acc = 0.0;
    var i = lid;
    loop {
        if (i >= dv) { break; }
        let v = tn_o[bo + i];
        acc = acc + v * v;
        i = i + 256u;
    }
    tn_red[lid] = acc;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { tn_red[lid] = tn_red[lid] + tn_red[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let inv = 1.0 / sqrt(tn_red[0] / f32(dv) + tn_p.eps);
    i = lid;
    loop {
        if (i >= dv) { break; }
        let zz = tn_z[bz + i];
        tn_o[bo + i] = tn_o[bo + i] * inv * tn_norm[i] * (1.0 / (1.0 + exp(-zz)));
        i = i + 256u;
    }
}

// ── Qwen routing over T rows: one workgroup per token (the main module's
// `moe_route` with the qwen/subset/pin_shared flags, per-row offsets). ──
struct TrP { n: u32, top_k: u32, flags: u32, scale: f32, ss: u32, fs: u32, is: u32, cs: u32 };
@group(0) @binding(0) var<storage, read>       tr_s      : array<f32>;
@group(0) @binding(1) var<storage, read>       tr_forced : array<u32>;
@group(0) @binding(2) var<storage, read_write> tr_idx    : array<u32>;
@group(0) @binding(3) var<storage, read_write> tr_w      : array<f32>;
@group(0) @binding(4) var<storage, read_write> tr_cnt    : array<u32>;
@group(0) @binding(5) var<uniform>             tr_p      : TrP;
@group(0) @binding(6) var<storage, read>       tr_map    : array<u32>;
@group(0) @binding(7) var<storage, read_write> tr_cold   : array<u32>;
var<workgroup> tr_sc:   array<f32, 1024>;
var<workgroup> tr_used: array<u32, 64>;
@compute @workgroup_size(1024)
fn q4t_route(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let n = tr_p.n;
    let k = tr_p.top_k;
    let s0 = t * tr_p.ss;
    let f0 = t * tr_p.fs;
    let i0 = t * tr_p.is;
    let c0 = t * tr_p.cs;
    let pin_shared = (tr_p.flags & 8u) != 0u;
    let subset = (tr_p.flags & 16u) != 0u;
    let qwen = (tr_p.flags & 32u) != 0u;
    let shared_gated = (tr_p.flags & 64u) != 0u;
    // flag 128 (with a remap): the ranking sees only the experts the arena holds
    let masked = subset && (tr_p.flags & 128u) != 0u;
    if (lid < k) {
        tr_used[lid] = 0u;
        tr_idx[i0 + lid] = 0u;
        tr_w[i0 + lid] = 0.0;
        tr_cold[c0 + 2u * lid] = 0xFFFFFFFFu;
        tr_cold[c0 + 2u * lid + 1u] = 0u;
        tr_cold[c0 + 2u * k + 2u * lid] = 0xFFFFFFFFu;
        tr_cold[c0 + 2u * k + 2u * lid + 1u] = 0u;
    }
    let shared_slot = tr_p.flags >> 8u;
    if (pin_shared && lid == 0u) {
        tr_idx[i0 + k] = shared_slot;
        if (shared_gated) {
            tr_w[i0 + k] = bitcast<f32>(tr_forced[f0 + k]);
        } else {
            tr_w[i0 + k] = 1.0;
        }
    }
    storageBarrier();
    var i = lid;
    loop {
        if (i >= n) { break; }
        let v = tr_s[s0 + i];
        var sc = v;
        if (!qwen) {
            var sp = v;
            if (v <= 20.0) { sp = log(1.0 + exp(v)); }
            sc = sqrt(sp);
        }
        if (masked && tr_map[i] == 0xFFFFFFFFu) { sc = bitcast<f32>(0xFF800000u); }
        tr_sc[i] = sc;
        i = i + 1024u;
    }
    workgroupBarrier();
    var m = lid;
    loop {
        if (m >= n) { break; }
        let si = tr_sc[m];
        var rank = 0u;
        for (var j = 0u; j < n; j = j + 1u) {
            let sj = tr_sc[j];
            if (sj > si || (sj == si && j < m)) { rank = rank + 1u; }
        }
        if (rank < k) {
            tr_used[rank] = 1u;
            tr_cold[c0 + 2u * k + 2u * rank] = m;
            tr_cold[c0 + 2u * k + 2u * rank + 1u] = bitcast<u32>(si);
            if (subset) {
                let slot = tr_map[m];
                if (slot == 0xFFFFFFFFu) {
                    tr_idx[i0 + rank] = 0u;
                    tr_w[i0 + rank] = 0.0;
                    tr_cold[c0 + 2u * rank] = m;
                    tr_cold[c0 + 2u * rank + 1u] = bitcast<u32>(si);
                } else {
                    tr_idx[i0 + rank] = slot;
                    tr_w[i0 + rank] = si;
                }
            } else {
                tr_idx[i0 + rank] = m;
                tr_w[i0 + rank] = si;
            }
        }
        m = m + 1024u;
    }
    workgroupBarrier();
    storageBarrier();
    if (lid == 0u) {
        var cnt = 0u;
        for (var j = 0u; j < k; j = j + 1u) {
            if (tr_used[j] == 1u) { cnt = cnt + 1u; }
        }
        tr_cnt[t * 4u] = cnt;
        var qmx = -3.0e38;
        if (qwen) {
            for (var j = 0u; j < cnt; j = j + 1u) {
                var v = tr_w[i0 + j];
                if (tr_cold[c0 + 2u * j] != 0xFFFFFFFFu) {
                    v = bitcast<f32>(tr_cold[c0 + 2u * j + 1u]);
                }
                qmx = max(qmx, v);
            }
            for (var j = 0u; j < cnt; j = j + 1u) {
                if (tr_cold[c0 + 2u * j] != 0xFFFFFFFFu) {
                    tr_cold[c0 + 2u * j + 1u] =
                        bitcast<u32>(exp(bitcast<f32>(tr_cold[c0 + 2u * j + 1u]) - qmx));
                } else {
                    tr_w[i0 + j] = exp(tr_w[i0 + j] - qmx);
                }
                tr_cold[c0 + 2u * k + 2u * j + 1u] =
                    bitcast<u32>(exp(bitcast<f32>(tr_cold[c0 + 2u * k + 2u * j + 1u]) - qmx));
            }
        }
        var sum = 0.0;
        for (var j = 0u; j < cnt; j = j + 1u) {
            sum = sum + tr_w[i0 + j];
            if (tr_cold[c0 + 2u * j] != 0xFFFFFFFFu) {
                sum = sum + bitcast<f32>(tr_cold[c0 + 2u * j + 1u]);
            }
        }
        if (sum > 0.0) {
            let inv = tr_p.scale / sum;
            for (var j = 0u; j < cnt; j = j + 1u) {
                tr_w[i0 + j] = tr_w[i0 + j] * inv;
                if (tr_cold[c0 + 2u * j] != 0xFFFFFFFFu) {
                    tr_cold[c0 + 2u * j + 1u] =
                        bitcast<u32>(bitcast<f32>(tr_cold[c0 + 2u * j + 1u]) * inv);
                }
                tr_cold[c0 + 2u * k + 2u * j + 1u] =
                    bitcast<u32>(bitcast<f32>(tr_cold[c0 + 2u * k + 2u * j + 1u]) * inv);
            }
        }
    }
}

// ── one row of a q8_2f embedding table, picked by `ids[st]`, as f32:
// w[r,i] = q[r,i]·row_scale[r]·col[i] (layout `[int8: rows·cols][f16: rows][f16: cols]`).
// The MTP draft chain re-embeds its own argmax without a host round trip. ──
struct EqP { cols: u32, rows: u32, st: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       eq_w   : array<u32>;
@group(0) @binding(1) var<storage, read>       eq_ids : array<u32>;
@group(0) @binding(2) var<storage, read_write> eq_out : array<f32>;
@group(0) @binding(3) var<uniform>             eq_p   : EqP;
@compute @workgroup_size(256)
fn q4_embed_gather_q82(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let cols = eq_p.cols;
    if (i >= cols) { return; }
    let r = min(eq_ids[eq_p.st], eq_p.rows - 1u);
    let qbytes = eq_p.rows * cols;
    let byte = r * cols + i;
    let q = i32((eq_w[byte >> 2u] >> ((byte & 3u) * 8u)) & 0xFFu);
    let qs = f32((q << 24u) >> 24u);
    let rs2 = unpack2x16float(eq_w[(qbytes >> 2u) + (r >> 1u)]);
    let rs = select(rs2.x, rs2.y, (r & 1u) == 1u);
    let ch = (qbytes >> 1u) + eq_p.rows + i;
    let cw = unpack2x16float(eq_w[ch >> 1u]);
    let cs = select(cw.x, cw.y, (ch & 1u) == 1u);
    eq_out[i] = qs * rs * cs;
}

// ── Hyper-connection down-projection with the group RMSNorm folded in:
// every workgroup (one output row of `down`, or one row of the injection
// gate after them) recomputes the per-stream inverse RMS of its T hyper
// rows and normalizes on the fly, so the norm never materializes and the
// separate dispatch goes. Workgroup 0 keeps the inverses for the up-fold.
// ──
struct ThP { cols: u32, rows_a: u32, rows_b: u32, act: u32, inv: f32, yb_off: u32, nt: u32, xs: u32,
             yas: u32, ybs: u32, hidden: u32, eps: f32, hc: u32, _d: u32, _e: u32, _f: u32 };
@group(0) @binding(0) var<storage, read>       th_a   : array<u32>;
@group(0) @binding(1) var<storage, read>       th_b   : array<u32>;
@group(0) @binding(2) var<storage, read>       th_h   : array<f32>;
@group(0) @binding(3) var<storage, read>       th_w   : array<f32>;
@group(0) @binding(4) var<storage, read_write> th_ya  : array<f32>;
@group(0) @binding(5) var<storage, read_write> th_yb  : array<f32>;
@group(0) @binding(6) var<storage, read_write> th_inv : array<f32>;
@group(0) @binding(7) var<uniform>             th_p   : ThP;
var<workgroup> th_lo: array<vec4<f32>, 64>;
var<workgroup> th_hi: array<vec4<f32>, 64>;
var<workgroup> th_ilo: array<vec4<f32>, 8>;
var<workgroup> th_ihi: array<vec4<f32>, 8>;
fn th_h4(o: u32, xs: u32) -> vec4<f32> {
    return vec4<f32>(th_h[o], th_h[xs + o], th_h[2u * xs + o], th_h[3u * xs + o]);
}
@compute @workgroup_size(64)
fn q4t_hc_down(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let cols = th_p.cols;
    let is_b = row >= th_p.rows_a;
    if (is_b && row - th_p.rows_a >= th_p.rows_b) { return; }
    let r = select(row, row - th_p.rows_a, is_b);
    let base = r * cols;
    let nt = th_p.nt;
    let xs = th_p.xs;
    let hidden = th_p.hidden;
    let hi_on = nt > 4u;
    // per-stream sums of squares of every token row
    for (var st = 0u; st < th_p.hc; st = st + 1u) {
        var slo = vec4<f32>(0.0);
        var shi = vec4<f32>(0.0);
        var i = st * hidden + lid * 2u;
        let end = (st + 1u) * hidden;
        loop {
            if (i >= end) { break; }
            let a = th_h4(i, xs);
            let b = th_h4(i + 1u, xs);
            slo = slo + a * a + b * b;
            if (hi_on) {
                let c = th_h4(4u * xs + i, xs);
                let d = th_h4(4u * xs + i + 1u, xs);
                shi = shi + c * c + d * d;
            }
            i = i + 128u;
        }
        th_lo[lid] = slo;
        th_hi[lid] = shi;
        workgroupBarrier();
        var s = 32u;
        loop {
            if (s == 0u) { break; }
            if (lid < s) {
                th_lo[lid] = th_lo[lid] + th_lo[lid + s];
                th_hi[lid] = th_hi[lid] + th_hi[lid + s];
            }
            workgroupBarrier();
            s = s >> 1u;
        }
        if (lid == 0u) {
            th_ilo[st] = inverseSqrt(th_lo[0] / f32(hidden) + th_p.eps);
            th_ihi[st] = inverseSqrt(th_hi[0] / f32(hidden) + th_p.eps);
        }
        workgroupBarrier();
    }
    if (row == 0u && lid < th_p.hc) {
        for (var t = 0u; t < nt; t = t + 1u) {
            var v = th_ilo[lid][t & 3u];
            if (t >= 4u) { v = th_ihi[lid][t & 3u]; }
            th_inv[t * th_p.hc + lid] = v;
        }
    }
    var lo = vec4<f32>(0.0);
    var hi = vec4<f32>(0.0);
    var i = lid * 2u;
    loop {
        if (i >= cols) { break; }
        var w2: vec2<f32>;
        if (is_b) { w2 = unpack2x16float(th_b[(base + i) >> 1u]); }
        else { w2 = unpack2x16float(th_a[(base + i) >> 1u]); }
        let st = i / hidden;
        let ilo = th_ilo[st];
        let g0 = 1.0 + th_w[i];
        let g1 = 1.0 + th_w[i + 1u];
        lo = lo + w2.x * (th_h4(i, xs) * ilo * g0) + w2.y * (th_h4(i + 1u, xs) * ilo * g1);
        if (hi_on) {
            let ihi = th_ihi[st];
            hi = hi + w2.x * (th_h4(4u * xs + i, xs) * ihi * g0) + w2.y * (th_h4(4u * xs + i + 1u, xs) * ihi * g1);
        }
        i = i + 128u;
    }
    th_lo[lid] = lo;
    th_hi[lid] = hi;
    workgroupBarrier();
    var s = 32u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) {
            th_lo[lid] = th_lo[lid] + th_lo[lid + s];
            th_hi[lid] = th_hi[lid] + th_hi[lid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (lid == 0u) {
        let r0 = th_lo[0];
        let r1 = th_hi[0];
        for (var t = 0u; t < nt; t = t + 1u) {
            var v = r0[t & 3u];
            if (t >= 4u) { v = r1[t & 3u]; }
            if (is_b) {
                if ((th_p.act & 2u) != 0u) { v = 1.0 / (1.0 + exp(-v)); }
                th_yb[t * th_p.ybs + th_p.yb_off + r] = v;
            } else {
                if ((th_p.act & 1u) != 0u) { let z = v * th_p.inv; v = z / (1.0 + exp(-z)); }
                th_ya[t * th_p.yas + r] = v;
            }
        }
    }
}

// ── the up-fold on the un-normalized hyper rows: normed[t,s,d] =
// h[t,s,d]·inv[t,s]·(1 + w[s,d]) with the inverses `q4t_hc_down` left ──
struct TvP { hc: u32, hidden: u32, low: u32, inv: f32, nt: u32, ls: u32, hs: u32, os: u32 };
@group(0) @binding(0) var<storage, read>       tv_w    : array<u32>;
@group(0) @binding(1) var<storage, read>       tv_low  : array<f32>;
@group(0) @binding(2) var<storage, read>       tv_h    : array<f32>;
@group(0) @binding(3) var<storage, read>       tv_nw   : array<f32>;
@group(0) @binding(4) var<storage, read>       tv_inv  : array<f32>;
@group(0) @binding(5) var<storage, read_write> tv_out  : array<f32>;
@group(0) @binding(6) var<uniform>             tv_p    : TvP;
var<workgroup> tv_lo: array<vec4<f32>, 64>;
var<workgroup> tv_hi: array<vec4<f32>, 64>;
fn tv_l4(o: u32, ls: u32) -> vec4<f32> {
    return vec4<f32>(tv_low[o], tv_low[ls + o], tv_low[2u * ls + o], tv_low[3u * ls + o]);
}
fn tv_n4(o: u32, hs: u32, st: u32, t0: u32) -> vec4<f32> {
    let hc = tv_p.hc;
    let g = 1.0 + tv_nw[o];
    return vec4<f32>(
        tv_h[t0 * hs + o] * tv_inv[t0 * hc + st] * g,
        tv_h[(t0 + 1u) * hs + o] * tv_inv[(t0 + 1u) * hc + st] * g,
        tv_h[(t0 + 2u) * hs + o] * tv_inv[(t0 + 2u) * hc + st] * g,
        tv_h[(t0 + 3u) * hs + o] * tv_inv[(t0 + 3u) * hc + st] * g,
    );
}
@compute @workgroup_size(64)
fn q4t_hc_upfold2(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
    let d = wid.x;
    if (d >= tv_p.hidden) { return; }
    let low = tv_p.low;
    let nt = tv_p.nt;
    let ls = tv_p.ls;
    let hs = tv_p.hs;
    let hi_on = nt > 4u;
    var alo = vec4<f32>(0.0);
    var ahi = vec4<f32>(0.0);
    for (var st = 0u; st < tv_p.hc; st = st + 1u) {
        let base = (st * tv_p.hidden + d) * low;
        var plo = vec4<f32>(0.0);
        var phi = vec4<f32>(0.0);
        var j = lid * 2u;
        loop {
            if (j >= low) { break; }
            let w2 = unpack2x16float(tv_w[(base + j) >> 1u]);
            plo = plo + w2.x * tv_l4(j, ls) + w2.y * tv_l4(j + 1u, ls);
            if (hi_on) {
                phi = phi + w2.x * tv_l4(4u * ls + j, ls) + w2.y * tv_l4(4u * ls + j + 1u, ls);
            }
            j = j + 128u;
        }
        tv_lo[lid] = plo;
        tv_hi[lid] = phi;
        workgroupBarrier();
        var s = 32u;
        loop {
            if (s == 0u) { break; }
            if (lid < s) {
                tv_lo[lid] = tv_lo[lid] + tv_lo[lid + s];
                tv_hi[lid] = tv_hi[lid] + tv_hi[lid + s];
            }
            workgroupBarrier();
            s = s >> 1u;
        }
        let mlo = tv_lo[0];
        let mhi = tv_hi[0];
        let no = st * tv_p.hidden + d;
        alo = alo + (1.0 / (1.0 + exp(-mlo))) * tv_n4(no, hs, st, 0u) * tv_p.inv;
        if (hi_on) {
            ahi = ahi + (1.0 / (1.0 + exp(-mhi))) * tv_n4(no, hs, st, 4u) * tv_p.inv;
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        for (var t = 0u; t < nt; t = t + 1u) {
            var v = alo[t & 3u];
            if (t >= 4u) { v = ahi[t & 3u]; }
            tv_out[t * tv_p.os + d] = v;
        }
    }
}

// ── Token-wide QSA. The frame's token table `tab[t] = (pos, complete
// blocks, kept blocks, attended count)` drives every per-token quantity, so
// the twelve QSA layers run one dispatch per stage for the whole frame. ──

// RoPE + qk-norm + gate split over T rows (the main module's
// `attn_rope_qkn`, per-row position from the table; no late-norm order).
struct TrqP { nh: u32, nkv: u32, hd: u32, rd: u32, flags: u32, eps: f32, qs: u32, ks: u32,
              qos: u32, gos: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32 };
@group(0) @binding(0) var<storage, read>       rq2_qraw : array<f32>;
@group(0) @binding(1) var<storage, read_write> rq2_k    : array<f32>;
@group(0) @binding(2) var<storage, read_write> rq2_qout : array<f32>;
@group(0) @binding(3) var<storage, read_write> rq2_gout : array<f32>;
@group(0) @binding(4) var<storage, read>       rq2_qnw  : array<f32>;
@group(0) @binding(5) var<storage, read>       rq2_knw  : array<f32>;
@group(0) @binding(6) var<storage, read>       rq2_invf : array<f32>;
@group(0) @binding(7) var<storage, read>       rq2_tab  : array<vec4<u32>>;
@group(0) @binding(8) var<uniform>             rq2_p    : TrqP;
var<workgroup> rq2_red: array<f32, 32>;
var<workgroup> rq2_head: array<f32, 256>;
@compute @workgroup_size(32)
fn q4t_rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let tk = wid.y;
    let lane = lid.x;
    let nh = rq2_p.nh;
    let hd = rq2_p.hd;
    if (head >= nh + rq2_p.nkv) { return; }
    let pos = rq2_tab[tk].x;
    let isq = head < nh;
    let gate = (rq2_p.flags & 1u) != 0u;
    let src_base = select((head - nh) * hd, head * select(1u, 2u, gate) * hd, isq);
    let qoff = tk * rq2_p.qs;
    let koff = tk * rq2_p.ks;
    let nt = (hd + 31u) / 32u;
    var xv: array<f32, 8>;
    var ss = 0.0;
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        var val = 0.0;
        if (d < hd) { val = select(rq2_k[koff + src_base + d], rq2_qraw[qoff + src_base + d], isq); }
        xv[t] = val;
        ss = ss + val * val;
    }
    rq2_red[lane] = ss;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) { rq2_red[lane] = rq2_red[lane] + rq2_red[lane + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let normed = select((rq2_p.flags & 4u) != 0u, (rq2_p.flags & 2u) != 0u, isq);
    let hlf = rq2_p.rd / 2u;
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) { rq2_head[d] = xv[t]; }
    }
    workgroupBarrier();
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) { xv[t] = rq2_head[d]; }
    }
    workgroupBarrier();
    if (normed) {
        let inv = 1.0 / sqrt(rq2_red[0] / f32(hd) + rq2_p.eps);
        let gemma = (rq2_p.flags & 8u) != 0u;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) {
                var wd = select(rq2_knw[d], rq2_qnw[d], isq);
                if (gemma) { wd = 1.0 + wd; }
                xv[t] = xv[t] * inv * wd;
            }
        }
    }
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) { rq2_head[d] = xv[t]; }
    }
    workgroupBarrier();
    var ri = lane;
    loop {
        if (ri >= hlf) { break; }
        let angle = f32(pos) * rq2_invf[ri];
        let cc = cos(angle);
        let sfac = sin(angle);
        let x0 = rq2_head[ri];
        let x1 = rq2_head[ri + hlf];
        rq2_head[ri] = x0 * cc - x1 * sfac;
        rq2_head[ri + hlf] = x0 * sfac + x1 * cc;
        ri = ri + 32u;
    }
    workgroupBarrier();
    let dst_base = select((head - nh) * hd, head * hd, isq);
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) {
            if (isq) { rq2_qout[tk * rq2_p.qos + dst_base + d] = rq2_head[d]; }
            else { rq2_k[koff + dst_base + d] = rq2_head[d]; }
        }
    }
    if (isq && gate) {
        let gbase = head * 2u * hd + hd;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) { rq2_gout[tk * rq2_p.gos + head * hd + d] = rq2_qraw[qoff + gbase + d]; }
        }
    }
}

// K/V rows of every token into the caches ([nkv][cap][hd]) at their positions.
struct TkvP { nkv: u32, hd: u32, cap: u32, ks: u32, vs: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       kv2_k  : array<f32>;
@group(0) @binding(1) var<storage, read>       kv2_v  : array<f32>;
@group(0) @binding(2) var<storage, read_write> kv2_kb : array<f32>;
@group(0) @binding(3) var<storage, read_write> kv2_vb : array<f32>;
@group(0) @binding(4) var<storage, read>       kv2_tab: array<vec4<u32>>;
@group(0) @binding(5) var<uniform>             kv2_p  : TkvP;
@compute @workgroup_size(256)
fn q4t_kv_append(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= kv2_p.nkv * kv2_p.hd) { return; }
    let pos = kv2_tab[t].x;
    let h = i / kv2_p.hd;
    let d = i - h * kv2_p.hd;
    let dst = (h * kv2_p.cap + pos) * kv2_p.hd + d;
    kv2_kb[dst] = kv2_k[t * kv2_p.ks + i];
    kv2_vb[dst] = kv2_v[t * kv2_p.vs + i];
}

// The raw indexer key of every token into its cache row.
struct TrkP { idim: u32, ioff: u32, is: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       rk_iqk : array<f32>;
@group(0) @binding(1) var<storage, read_write> rk_raw : array<f32>;
@group(0) @binding(2) var<storage, read>       rk_tab : array<vec4<u32>>;
@group(0) @binding(3) var<uniform>             rk_p   : TrkP;
@compute @workgroup_size(256)
fn q4t_rawk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let t = gid.y;
    if (d >= rk_p.idim) { return; }
    let pos = rk_tab[t].x;
    rk_raw[pos * rk_p.idim + d] = rk_iqk[t * rk_p.is + rk_p.ioff + d];
}

// Indexer scores of every token against the blocks it has completed.
struct TixP { nh: u32, hd: u32, qs: u32, os: u32 };
@group(0) @binding(0) var<storage, read>       ix2_q   : array<f32>;
@group(0) @binding(1) var<storage, read>       ix2_kv  : array<f32>;
@group(0) @binding(2) var<storage, read>       ix2_w   : array<f32>;
@group(0) @binding(3) var<storage, read_write> ix2_out : array<f32>;
@group(0) @binding(4) var<storage, read>       ix2_tab : array<vec4<u32>>;
@group(0) @binding(5) var<uniform>             ix2_p   : TixP;
var<workgroup> ix2_red: array<f32, 256>;
@compute @workgroup_size(256)
fn q4t_ix_scores(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) lid: u32) {
    let blk = wid.x;
    let t = wid.y;
    if (blk >= ix2_tab[t].y) { return; }
    let hd = ix2_p.hd;
    let kb = blk * hd;
    let q0 = t * ix2_p.qs;
    var acc = 0.0;
    var h = lid;
    loop {
        if (h >= ix2_p.nh) { break; }
        var dot = 0.0;
        let qb = q0 + h * hd;
        for (var i = 0u; i < hd; i = i + 1u) {
            dot = dot + ix2_q[qb + i] * ix2_kv[kb + i];
        }
        acc = acc + max(dot, 0.0) * ix2_w[h];
        h = h + 256u;
    }
    ix2_red[lid] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { ix2_red[lid] = ix2_red[lid] + ix2_red[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) { ix2_out[t * ix2_p.os + blk] = ix2_red[0]; }
}

// Top-k of every token's block scores (the main module's `top_k_index`).
struct TtkP { ss: u32, ps: u32, cs: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       tk2_s   : array<f32>;
@group(0) @binding(1) var<storage, read_write> tk2_idx : array<u32>;
@group(0) @binding(2) var<storage, read_write> tk2_cnt : array<u32>;
@group(0) @binding(3) var<storage, read>       tk2_tab : array<vec4<u32>>;
@group(0) @binding(4) var<uniform>             tk2_p   : TtkP;
var<workgroup> tk2_keep: array<u32, 4096>;
@compute @workgroup_size(1024)
fn q4t_topk(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let n = tk2_tab[t].y;
    let k = tk2_tab[t].z;
    let s0 = t * tk2_p.ss;
    let p0 = t * tk2_p.ps;
    if (n == 0u) {
        if (lid == 0u) { tk2_cnt[t * tk2_p.cs] = 0u; }
        return;
    }
    var i = lid;
    loop {
        if (i >= n) { break; }
        let si = tk2_s[s0 + i];
        var rank = 0u;
        for (var j = 0u; j < n; j = j + 1u) {
            let sj = tk2_s[s0 + j];
            if (sj > si || (sj == si && j < i)) { rank = rank + 1u; }
        }
        var keep = 0u;
        if (rank < k) { keep = 1u; }
        tk2_keep[i] = keep;
        i = i + 1024u;
    }
    workgroupBarrier();
    var m = lid;
    loop {
        if (m >= n) { break; }
        if (tk2_keep[m] == 1u) {
            var before = 0u;
            for (var j = 0u; j < m; j = j + 1u) { before = before + tk2_keep[j]; }
            tk2_idx[p0 + before] = m;
        }
        m = m + 1024u;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var total = 0u;
        for (var j = 0u; j < n; j = j + 1u) { total = total + tk2_keep[j]; }
        tk2_cnt[t * tk2_p.cs] = total;
    }
}

// The indexer's block scores with a lane per block (`q4t_ix_scores` spends a
// 256-lane workgroup per block on its four heads): the token's index queries in
// workgroup memory, every lane one block's dots. nh·hd <= 1024.
var<workgroup> ix3_q: array<f32, 1024>;
@compute @workgroup_size(256)
fn q4t_ix_scores_l(@builtin(workgroup_id) wid: vec3<u32>,
                   @builtin(local_invocation_index) lid: u32) {
    let t = wid.y;
    let hd = ix2_p.hd;
    let nh = ix2_p.nh;
    for (var i = lid; i < nh * hd; i = i + 256u) { ix3_q[i] = ix2_q[t * ix2_p.qs + i]; }
    workgroupBarrier();
    let blk = wid.x * 256u + lid;
    if (blk >= ix2_tab[t].y) { return; }
    let kb = blk * hd;
    var acc = 0.0;
    for (var h = 0u; h < nh; h = h + 1u) {
        var d = 0.0;
        let qb = h * hd;
        for (var i = 0u; i < hd; i = i + 4u) {
            let kv = vec4<f32>(ix2_kv[kb + i], ix2_kv[kb + i + 1u], ix2_kv[kb + i + 2u], ix2_kv[kb + i + 3u]);
            let qv = vec4<f32>(ix3_q[qb + i], ix3_q[qb + i + 1u], ix3_q[qb + i + 2u], ix3_q[qb + i + 3u]);
            d = d + dot(kv, qv);
        }
        acc = acc + max(d, 0.0) * ix2_w[h];
    }
    ix2_out[t * ix2_p.os + blk] = acc;
}

// Top-k of a token's block scores for any count (`q4t_topk` ranks every block
// against every other and keeps a 4096-entry table): the k-th largest score by
// a 32-step radix search on the (non-negative) score bits, ties to the lower
// index, the kept blocks written in index order. Same set as `q4t_topk`.
var<workgroup> tk3_acc: atomic<u32>;
var<workgroup> tk3_tot: u32;
var<workgroup> tk3_hdr: vec2<u32>;
var<workgroup> tk3_sa: array<u32, 1024>;
var<workgroup> tk3_sb: array<u32, 1024>;
var<workgroup> tk3_h: array<atomic<u32>, 256>;
var<workgroup> tk3_hs: array<u32, 256>;
var<workgroup> tk3_sel: vec2<u32>;
fn tk3_key(v: f32) -> u32 { return select(bitcast<u32>(v), 0u, v <= 0.0); }
// workgroup-wide sum of `c`, uniform
fn tk3_sum(c: u32, lid: u32) -> u32 {
    if (lid == 0u) { atomicStore(&tk3_acc, 0u); }
    workgroupBarrier();
    atomicAdd(&tk3_acc, c);
    workgroupBarrier();
    if (lid == 0u) { tk3_tot = atomicLoad(&tk3_acc); }
    return workgroupUniformLoad(&tk3_tot);
}
@compute @workgroup_size(1024)
fn q4t_topk_r(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    if (lid == 0u) { tk3_hdr = vec2<u32>(tk2_tab[t].y, tk2_tab[t].z); }
    let hdr = workgroupUniformLoad(&tk3_hdr);
    let n = hdr.x;
    let k = hdr.y;
    let s0 = t * tk2_p.ss;
    let p0 = t * tk2_p.ps;
    let per = (n + 1023u) / 1024u;
    let lo = min(n, lid * per);
    let hi = min(n, lo + per);
    // every block kept (k >= n, up to top_k·ratio positions): index order, as below
    if (k >= n) {
        for (var i = lo; i < hi; i = i + 1u) { tk2_idx[p0 + i] = i; }
        if (lid == 0u) { tk2_cnt[t * tk2_p.cs] = n; }
        return;
    }
    // the k-th largest key, eight bits a pass: a 256-bin histogram of the keys
    // that match the digits found so far, then the digit whose suffix count
    // first reaches what is still needed (the same value the bitwise search
    // finds: the largest thr with count(key >= thr) >= k)
    var thr = 0u;
    var kk = k;
    for (var d = 0u; d < 4u; d = d + 1u) {
        let sh = 24u - 8u * d;
        let hm = select(0xFFFFFFFFu << (sh + 8u), 0u, d == 0u);
        if (lid < 256u) { atomicStore(&tk3_h[lid], 0u); }
        workgroupBarrier();
        for (var i = lo; i < hi; i = i + 1u) {
            let key = tk3_key(tk2_s[s0 + i]);
            if ((key & hm) == thr) { atomicAdd(&tk3_h[(key >> sh) & 255u], 1u); }
        }
        workgroupBarrier();
        if (lid < 256u) { tk3_hs[lid] = atomicLoad(&tk3_h[lid]); }
        workgroupBarrier();
        for (var o = 1u; o < 256u; o = o << 1u) {
            var v = 0u;
            if (lid < 256u) {
                v = tk3_hs[lid];
                if (lid + o < 256u) { v = v + tk3_hs[lid + o]; }
            }
            workgroupBarrier();
            if (lid < 256u) { tk3_hs[lid] = v; }
            workgroupBarrier();
        }
        if (lid < 256u) {
            var above = 0u;
            if (lid < 255u) { above = tk3_hs[lid + 1u]; }
            if (above < kk && kk <= tk3_hs[lid]) { tk3_sel = vec2<u32>(lid, kk - above); }
        }
        let sel = workgroupUniformLoad(&tk3_sel);
        thr = thr | (sel.x << sh);
        kk = sel.y;
    }
    // keep everything when k >= n
    if (k >= n) { thr = 0u; }
    var gt = 0u;
    var eq = 0u;
    for (var i = lo; i < hi; i = i + 1u) {
        let key = tk3_key(tk2_s[s0 + i]);
        if (key > thr) { gt = gt + 1u; } else if (key == thr) { eq = eq + 1u; }
    }
    if (k >= n) { gt = gt + eq; eq = 0u; }
    let gt_all = tk3_sum(gt, lid);
    let need = select(0u, k - gt_all, k > gt_all);
    // exclusive prefix of the ties, then of what each lane keeps
    tk3_sa[lid] = eq;
    workgroupBarrier();
    for (var o = 1u; o < 1024u; o = o << 1u) {
        var v = tk3_sa[lid];
        if (lid >= o) { v = v + tk3_sa[lid - o]; }
        workgroupBarrier();
        tk3_sa[lid] = v;
        workgroupBarrier();
    }
    let eq_before = tk3_sa[lid] - eq;
    let take = min(eq, select(0u, need - eq_before, need > eq_before));
    tk3_sb[lid] = gt + take;
    workgroupBarrier();
    for (var o = 1u; o < 1024u; o = o << 1u) {
        var v = tk3_sb[lid];
        if (lid >= o) { v = v + tk3_sb[lid - o]; }
        workgroupBarrier();
        tk3_sb[lid] = v;
        workgroupBarrier();
    }
    var out = tk3_sb[lid] - (gt + take);
    var taken = 0u;
    for (var i = lo; i < hi; i = i + 1u) {
        let key = tk3_key(tk2_s[s0 + i]);
        var keep = key > thr || k >= n;
        if (!keep && key == thr && taken < take) { keep = true; taken = taken + 1u; }
        if (keep) {
            tk2_idx[p0 + out] = i;
            out = out + 1u;
        }
    }
    if (lid == 1023u) { tk2_cnt[t * tk2_p.cs] = tk3_sb[1023]; }
}

// Every token's attended-position list: kept blocks, then its open tail.
struct TibP { cr: u32, ps: u32, is: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       ib2_pick : array<u32>;
@group(0) @binding(1) var<storage, read_write> ib2_idx  : array<u32>;
@group(0) @binding(2) var<storage, read>       ib2_tab  : array<vec4<u32>>;
@group(0) @binding(3) var<uniform>             ib2_p    : TibP;
@compute @workgroup_size(256)
fn q4t_idx_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let t = gid.y;
    let tab = ib2_tab[t];
    let cr = ib2_p.cr;
    let nsel = tab.z * cr;
    let tail = tab.x + 1u - tab.y * cr;
    if (j >= nsel + tail) { return; }
    let o = t * ib2_p.is + j;
    if (j < nsel) {
        ib2_idx[o] = ib2_pick[t * ib2_p.ps + j / cr] * cr + (j % cr);
    } else {
        ib2_idx[o] = tab.y * cr + (j - nsel);
    }
}

// Grouped sparse attention of every token over its index list.
struct TqaP { nh: u32, hd: u32, scale: f32, groups: u32, cap: u32, qs: u32, is: u32, gs: u32,
              os: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32, _g: u32 };
@group(0) @binding(0) var<storage, read>       qa2_q    : array<f32>;
@group(0) @binding(1) var<storage, read>       qa2_k    : array<f32>;
@group(0) @binding(2) var<storage, read>       qa2_v    : array<f32>;
@group(0) @binding(3) var<storage, read>       qa2_idx  : array<u32>;
@group(0) @binding(4) var<storage, read>       qa2_gate : array<f32>;
@group(0) @binding(5) var<storage, read_write> qa2_out  : array<f32>;
@group(0) @binding(6) var<storage, read>       qa2_tab  : array<vec4<u32>>;
@group(0) @binding(7) var<uniform>             qa2_p    : TqaP;
var<workgroup> qa2_red: array<f32, 256>;
var<workgroup> qa2_w: array<f32, 2112>;
var<workgroup> qa2_qs: array<f32, 256>;
@compute @workgroup_size(256)
fn q4t_attend(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let t = wid.y;
    if (h >= qa2_p.nh) { return; }
    let hd = qa2_p.hd;
    let m = qa2_tab[t].w;
    let kbase = (h / qa2_p.groups) * qa2_p.cap * hd;
    let i0 = t * qa2_p.is;
    if (lid < hd) { qa2_qs[lid] = qa2_q[t * qa2_p.qs + h * hd + lid]; }
    workgroupBarrier();
    var mx = -3.0e38;
    var tt = lid;
    loop {
        if (tt >= m) { break; }
        let p = qa2_idx[i0 + tt];
        var d = 0.0;
        let kb = kbase + p * hd;
        for (var k = 0u; k < hd; k = k + 1u) { d = d + qa2_qs[k] * qa2_k[kb + k]; }
        let sc = d * qa2_p.scale;
        qa2_w[tt] = sc;
        mx = max(mx, sc);
        tt = tt + 256u;
    }
    qa2_red[lid] = mx;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { qa2_red[lid] = max(qa2_red[lid], qa2_red[lid + s]); }
        workgroupBarrier();
        s = s >> 1u;
    }
    let mval = qa2_red[0];
    workgroupBarrier();
    var den = 0.0;
    tt = lid;
    loop {
        if (tt >= m) { break; }
        let w = exp(qa2_w[tt] - mval);
        qa2_w[tt] = w;
        den = den + w;
        tt = tt + 256u;
    }
    qa2_red[lid] = den;
    workgroupBarrier();
    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) { qa2_red[lid] = qa2_red[lid] + qa2_red[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let inv = 1.0 / max(qa2_red[0], 1.17549435e-38);
    var k = lid;
    loop {
        if (k >= hd) { break; }
        var acc = 0.0;
        for (var i = 0u; i < m; i = i + 1u) {
            acc = acc + qa2_w[i] * qa2_v[kbase + qa2_idx[i0 + i] * hd + k];
        }
        let g = qa2_gate[t * qa2_p.gs + h * hd + k];
        qa2_out[t * qa2_p.os + h * hd + k] = acc * inv * (1.0 / (1.0 + exp(-g)));
        k = k + 256u;
    }
}

// Split-key grouped attention (flash decoding): one workgroup per (KV head, chunk of
// `ch` attended keys, token). The GQA group's `groups` query heads share every K/V row
// the workgroup reads (the per-head kernel above reads each one `groups` times, one
// workgroup a head walking all `m` keys alone). Each workgroup leaves its chunk's
// running max, exp-sum and exp-weighted V rows; `q4t_attend_cmb` folds the chunks of a
// head and applies the output gate. Limits: groups <= 16, ch <= 32, hd <= 256.
struct TspP { nh: u32, hd: u32, scale: f32, groups: u32, cap: u32, qs: u32, is: u32, ch: u32,
              nsp: u32, ps: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32 };
@group(0) @binding(0) var<storage, read>       sp_q   : array<f32>;
@group(0) @binding(1) var<storage, read>       sp_k   : array<f32>;
@group(0) @binding(2) var<storage, read>       sp_v   : array<f32>;
@group(0) @binding(3) var<storage, read>       sp_idx : array<u32>;
@group(0) @binding(4) var<storage, read>       sp_tab : array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> sp_o   : array<f32>;
@group(0) @binding(6) var<storage, read_write> sp_ml  : array<vec2<f32>>;
@group(0) @binding(7) var<uniform>             sp_p   : TspP;
var<workgroup> sp_qs: array<f32, 4096>;   // [g][d] the group's queries
var<workgroup> sp_sc: array<f32, 512>;    // [g][j] scores, then weights
var<workgroup> sp_kp: array<u32, 32>;     // the chunk's key positions
var<workgroup> sp_mx: array<f32, 16>;
@compute @workgroup_size(256)
fn q4t_attend_sp(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_index) lid: u32) {
    let kvh = wid.x;
    let s = wid.y;
    let t = wid.z;
    let hd = sp_p.hd;
    let gq = sp_p.groups;
    let ch = sp_p.ch;
    let m = sp_tab[t].w;
    let j0 = s * ch;
    let n = select(0u, min(ch, m - j0), j0 < m);
    let h0 = kvh * gq;
    // partial slot of (token, chunk, head)
    let pbase = (t * sp_p.nsp + s) * sp_p.nh;
    if (n == 0u) {
        if (lid < gq) { sp_ml[pbase + h0 + lid] = vec2<f32>(-3.0e38, 0.0); }
        return;
    }
    for (var i = lid; i < gq * hd; i = i + 256u) {
        let g = i / hd;
        let d = i - g * hd;
        sp_qs[i] = sp_q[t * sp_p.qs + (h0 + g) * hd + d];
    }
    // every block kept (kept == complete): the list is 0..m, as q4t_idx_build
    // would have written it from the radix top-k's index-ordered picks
    let every = sp_tab[t].z == sp_tab[t].y;
    if (lid < n) {
        if (every) { sp_kp[lid] = j0 + lid; } else { sp_kp[lid] = sp_idx[t * sp_p.is + j0 + lid]; }
    }
    workgroupBarrier();
    let kbase = kvh * sp_p.cap * hd;
    // scores: lane -> (key j, heads g, g + 256/ch, ...)
    let j = lid % ch;
    let gl = lid / ch;
    let gstep = 256u / ch;
    if (j < n) {
        let kb = kbase + sp_kp[j] * hd;
        for (var g = gl; g < gq; g = g + gstep) {
            var d0 = 0.0;
            let qo = g * hd;
            for (var d = 0u; d < hd; d = d + 4u) {
                let kv = vec4<f32>(sp_k[kb + d], sp_k[kb + d + 1u], sp_k[kb + d + 2u], sp_k[kb + d + 3u]);
                let qv = vec4<f32>(sp_qs[qo + d], sp_qs[qo + d + 1u], sp_qs[qo + d + 2u], sp_qs[qo + d + 3u]);
                d0 = d0 + dot(kv, qv);
            }
            sp_sc[g * ch + j] = d0 * sp_p.scale;
        }
    }
    workgroupBarrier();
    // per head: the chunk's max, then weights and their sum
    if (lid < gq) {
        var mx = -3.0e38;
        for (var i = 0u; i < n; i = i + 1u) { mx = max(mx, sp_sc[lid * ch + i]); }
        var den = 0.0;
        for (var i = 0u; i < n; i = i + 1u) {
            let w = exp(sp_sc[lid * ch + i] - mx);
            sp_sc[lid * ch + i] = w;
            den = den + w;
        }
        sp_mx[lid] = mx;
        sp_ml[pbase + h0 + lid] = vec2<f32>(mx, den);
    }
    workgroupBarrier();
    // exp-weighted V rows: lane -> dimension d, head by head (the chunk's V rows,
    // 32 x hd floats, stay in L1 across the group's heads; a runtime-indexed
    // accumulator array would spill)
    for (var d = lid; d < hd; d = d + 256u) {
        for (var g = 0u; g < gq; g = g + 1u) {
            var acc = 0.0;
            for (var i = 0u; i < n; i = i + 1u) {
                acc = acc + sp_sc[g * ch + i] * sp_v[kbase + sp_kp[i] * hd + d];
            }
            sp_o[(pbase + h0 + g) * sp_p.ps + d] = acc;
        }
    }
}

// `q4t_attend_sp` with coalesced reads: eight lanes a key read its K row as
// vec4s interleaved over the row (lane sl: vec4s sl, sl+8, …), each against
// all the group's queries from workgroup memory; the slices' partial dots
// meet in workgroup memory. The V pass keeps every head's accumulator in
// registers (a constant-bound loop, unrolled), so each V element is read once.
// Same bindings, grid, outputs; needs ch == 32, hd % 32 == 0, hd <= 256,
// groups <= 12 (the host checks).
@group(0) @binding(1) var<storage, read> sp2_k4 : array<vec4<f32>>;
var<workgroup> sp2_part: array<f32, 3072>;   // [g][j][slice], g < 12 (Apple's 32 KB)
@compute @workgroup_size(256)
fn q4t_attend_sp2(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
    let kvh = wid.x;
    let s = wid.y;
    let t = wid.z;
    let hd = sp_p.hd;
    let gq = sp_p.groups;
    let ch = sp_p.ch;
    let m = sp_tab[t].w;
    let j0 = s * ch;
    let n = select(0u, min(ch, m - j0), j0 < m);
    let h0 = kvh * gq;
    let pbase = (t * sp_p.nsp + s) * sp_p.nh;
    if (n == 0u) {
        if (lid < gq) { sp_ml[pbase + h0 + lid] = vec2<f32>(-3.0e38, 0.0); }
        return;
    }
    for (var i = lid; i < gq * hd; i = i + 256u) {
        let g = i / hd;
        let d = i - g * hd;
        sp_qs[i] = sp_q[t * sp_p.qs + (h0 + g) * hd + d];
    }
    let every = sp_tab[t].z == sp_tab[t].y;
    if (lid < n) {
        if (every) { sp_kp[lid] = j0 + lid; } else { sp_kp[lid] = sp_idx[t * sp_p.is + j0 + lid]; }
    }
    workgroupBarrier();
    let kbase = kvh * sp_p.cap * hd;
    let j = lid >> 3u;
    let sl = lid & 7u;
    let nv = hd >> 5u;   // vec4s of a lane's slice
    if (j < n) {
        let row4 = (kbase + sp_kp[j] * hd) >> 2u;
        var kr: array<vec4<f32>, 8>;
        for (var e = 0u; e < 8u; e = e + 1u) {
            if (e < nv) { kr[e] = sp2_k4[row4 + sl + 8u * e]; }
        }
        for (var g = 0u; g < gq; g = g + 1u) {
            var acc = 0.0;
            for (var e = 0u; e < 8u; e = e + 1u) {
                if (e < nv) {
                    let qo = g * hd + 4u * (sl + 8u * e);
                    acc = acc + dot(kr[e], vec4<f32>(sp_qs[qo], sp_qs[qo + 1u], sp_qs[qo + 2u], sp_qs[qo + 3u]));
                }
            }
            sp2_part[(g * 32u + j) * 8u + sl] = acc;
        }
    }
    workgroupBarrier();
    for (var i = lid; i < gq * 32u; i = i + 256u) {
        let g = i >> 5u;
        let jj = i & 31u;
        if (jj < n) {
            let o = (g * 32u + jj) * 8u;
            var v = 0.0;
            for (var q = 0u; q < 8u; q = q + 1u) { v = v + sp2_part[o + q]; }
            sp_sc[g * ch + jj] = v * sp_p.scale;
        }
    }
    workgroupBarrier();
    if (lid < gq) {
        var mx = -3.0e38;
        for (var i = 0u; i < n; i = i + 1u) { mx = max(mx, sp_sc[lid * ch + i]); }
        var den = 0.0;
        for (var i = 0u; i < n; i = i + 1u) {
            let w = exp(sp_sc[lid * ch + i] - mx);
            sp_sc[lid * ch + i] = w;
            den = den + w;
        }
        sp_ml[pbase + h0 + lid] = vec2<f32>(mx, den);
    }
    workgroupBarrier();
    let d = lid;
    if (d < hd) {
        var acc: array<f32, 16>;
        for (var i = 0u; i < n; i = i + 1u) {
            let v = sp_v[kbase + sp_kp[i] * hd + d];
            for (var g = 0u; g < 16u; g = g + 1u) {
                if (g < gq) { acc[g] = acc[g] + sp_sc[g * ch + i] * v; }
            }
        }
        for (var g = 0u; g < 16u; g = g + 1u) {
            if (g < gq) { sp_o[(pbase + h0 + g) * sp_p.ps + d] = acc[g]; }
        }
    }
}

// Fold the chunks of every (head, token): out = gate(σ) · Σ e^{m_s−M} o_s / Σ e^{m_s−M} l_s.
struct TcbP { nh: u32, hd: u32, ch: u32, nsp: u32, ps: u32, gs: u32, os: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       cb_o    : array<f32>;
@group(0) @binding(1) var<storage, read>       cb_ml   : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       cb_gate : array<f32>;
@group(0) @binding(3) var<storage, read>       cb_tab  : array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> cb_out  : array<f32>;
@group(0) @binding(5) var<uniform>             cb_p    : TcbP;
var<workgroup> cb_w: array<f32, 128>;
@compute @workgroup_size(256)
fn q4t_attend_cmb(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let t = wid.y;
    if (h >= cb_p.nh) { return; }
    let m = cb_tab[t].w;
    let ns = min((m + cb_p.ch - 1u) / cb_p.ch, cb_p.nsp);
    if (lid == 0u) {
        var mx = -3.0e38;
        for (var s = 0u; s < ns; s = s + 1u) { mx = max(mx, cb_ml[(t * cb_p.nsp + s) * cb_p.nh + h].x); }
        var den = 0.0;
        for (var s = 0u; s < ns; s = s + 1u) {
            let ml = cb_ml[(t * cb_p.nsp + s) * cb_p.nh + h];
            let w = exp(ml.x - mx);
            cb_w[s] = w;
            den = den + w * ml.y;
        }
        cb_w[127] = 1.0 / max(den, 1.17549435e-38);
    }
    workgroupBarrier();
    let inv = cb_w[127];
    for (var d = lid; d < cb_p.hd; d = d + 256u) {
        var acc = 0.0;
        for (var s = 0u; s < ns; s = s + 1u) {
            acc = acc + cb_w[s] * cb_o[((t * cb_p.nsp + s) * cb_p.nh + h) * cb_p.ps + d];
        }
        let g = cb_gate[t * cb_p.gs + h * cb_p.hd + d];
        cb_out[t * cb_p.os + h * cb_p.hd + d] = acc * inv * (1.0 / (1.0 + exp(-g)));
    }
}

// Prompt frames past eight tokens (resident cut models): the f16 pair kernel as a
// tiled GEMM. A workgroup holds 64 rows of [A; B] against 16 tokens, 64 columns a
// step through workgroup memory; the epilogue is `q4t_f16_pair`'s (A rows to ya,
// SiLU(v·inv) under act & 1; B rows to yb at yb_off, σ under act & 2).
struct TmmP { cols: u32, rows_a: u32, rows_b: u32, act: u32, inv: f32, yb_off: u32, nt: u32, xs: u32,
              yas: u32, ybs: u32, kc: u32, ks: u32, _c: u32, _d: u32, _e: u32, _f: u32 };
@group(0) @binding(0) var<storage, read>       tmm_a  : array<u32>;
@group(0) @binding(1) var<storage, read>       tmm_b  : array<u32>;
@group(0) @binding(2) var<storage, read>       tmm_x  : array<f32>;
@group(0) @binding(3) var<storage, read_write> tmm_ya : array<f32>;
@group(0) @binding(4) var<storage, read_write> tmm_yb : array<f32>;
@group(0) @binding(5) var<uniform>             tmm_p  : TmmP;
// split-K partials [ks][nt][rows_a + rows_b] (wid.z = the column slice), when ks > 1
@group(0) @binding(6) var<storage, read_write> tmm_part : array<f32>;
var<workgroup> tmm_ws: array<f32, 4096>;  // [64 rows][64 cols]
var<workgroup> tmm_xs: array<f32, 1024>;  // [16 tokens][64 cols]
@compute @workgroup_size(256)
fn q4t_f16_pair_mm(@builtin(workgroup_id) wid: vec3<u32>,
                   @builtin(local_invocation_index) lid: u32) {
    let cols = tmm_p.cols;
    let ra = tmm_p.rows_a;
    let total = ra + tmm_p.rows_b;
    let row0 = wid.x * 64u;
    let t0 = wid.y * 16u;
    let nt = tmm_p.nt;
    let r = lid % 64u;
    let tq = lid / 64u;
    var acc = vec4<f32>(0.0);
    let kbeg = wid.z * tmm_p.kc;
    let kend = min(cols, kbeg + tmm_p.kc);
    for (var k0 = kbeg; k0 < kend; k0 = k0 + 64u) {
        // 64 x 64 weights: 2048 f16 pairs, eight a lane
        for (var i = lid; i < 2048u; i = i + 256u) {
            let rr = i / 32u;
            let cc = (i % 32u) * 2u;
            let row = row0 + rr;
            let col = k0 + cc;
            var w2 = vec2<f32>(0.0);
            if (row < total && col < kend) {
                if (row < ra) { w2 = unpack2x16float(tmm_a[(row * cols + col) >> 1u]); }
                else { w2 = unpack2x16float(tmm_b[((row - ra) * cols + col) >> 1u]); }
            }
            tmm_ws[rr * 64u + cc] = w2.x;
            tmm_ws[rr * 64u + cc + 1u] = w2.y;
        }
        // 16 tokens x 64 columns of the inputs
        for (var i = lid; i < 1024u; i = i + 256u) {
            let tt = i / 64u;
            let cc = i % 64u;
            let t = t0 + tt;
            var xv = 0.0;
            if (t < nt && k0 + cc < kend) { xv = tmm_x[t * tmm_p.xs + k0 + cc]; }
            tmm_xs[i] = xv;
        }
        workgroupBarrier();
        let wo = r * 64u;
        let xo = tq * 4u * 64u;
        for (var kk = 0u; kk < 64u; kk = kk + 1u) {
            let w = tmm_ws[wo + kk];
            acc = acc + w * vec4<f32>(tmm_xs[xo + kk], tmm_xs[xo + 64u + kk],
                                      tmm_xs[xo + 128u + kk], tmm_xs[xo + 192u + kk]);
        }
        workgroupBarrier();
    }
    let row = row0 + r;
    if (row >= total) { return; }
    if (tmm_p.ks > 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let t = t0 + tq * 4u + j;
            if (t >= nt) { break; }
            tmm_part[(wid.z * nt + t) * total + row] = acc[j];
        }
        return;
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        let t = t0 + tq * 4u + j;
        if (t >= nt) { break; }
        var v = acc[j];
        if (row >= ra) {
            if ((tmm_p.act & 2u) != 0u) { v = 1.0 / (1.0 + exp(-v)); }
            tmm_yb[t * tmm_p.ybs + tmm_p.yb_off + row - ra] = v;
        } else {
            if ((tmm_p.act & 1u) != 0u) { let z = v * tmm_p.inv; v = z / (1.0 + exp(-z)); }
            tmm_ya[t * tmm_p.yas + row] = v;
        }
    }
}

// The split-K partials of `q4t_f16_pair_mm` summed in slice order, then its epilogue.
@group(0) @binding(0) var<storage, read>       tps_part : array<f32>;
@group(0) @binding(1) var<storage, read_write> tps_ya   : array<f32>;
@group(0) @binding(2) var<storage, read_write> tps_yb   : array<f32>;
@group(0) @binding(3) var<uniform>             tps_p    : TmmP;
@compute @workgroup_size(256)
fn q4t_pair_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let t = gid.y;
    let ra = tps_p.rows_a;
    let total = ra + tps_p.rows_b;
    let nt = tps_p.nt;
    if (row >= total || t >= nt) { return; }
    var v = 0.0;
    for (var z = 0u; z < tps_p.ks; z = z + 1u) { v = v + tps_part[(z * nt + t) * total + row]; }
    if (row >= ra) {
        if ((tps_p.act & 2u) != 0u) { v = 1.0 / (1.0 + exp(-v)); }
        tps_yb[t * tps_p.ybs + tps_p.yb_off + row - ra] = v;
    } else {
        if ((tps_p.act & 1u) != 0u) { let z = v * tps_p.inv; v = z / (1.0 + exp(-z)); }
        tps_ya[t * tps_p.yas + row] = v;
    }
}

// The hyper-connection fold of `q4t_hc_upfold` once the up-projection is a GEMM:
// x[t][d] = inv · Σ_s σ(y[t][s·hidden + d]) · normed[t][s·hidden + d].
struct TfdP { hc: u32, hidden: u32, nt: u32, inv: f32, ys: u32, ns: u32, os: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       tfd_y  : array<f32>;
@group(0) @binding(1) var<storage, read>       tfd_n  : array<f32>;
@group(0) @binding(2) var<storage, read_write> tfd_o  : array<f32>;
@group(0) @binding(3) var<uniform>             tfd_p  : TfdP;
@compute @workgroup_size(256)
fn q4t_hc_fold_t(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let t = gid.y;
    if (d >= tfd_p.hidden || t >= tfd_p.nt) { return; }
    var a = 0.0;
    for (var st = 0u; st < tfd_p.hc; st = st + 1u) {
        let o = st * tfd_p.hidden + d;
        a = a + (1.0 / (1.0 + exp(-tfd_y[t * tfd_p.ys + o]))) * tfd_n[t * tfd_p.ns + o] * tfd_p.inv;
    }
    tfd_o[t * tfd_p.os + d] = a;
}

// ── Resident experts, four rows a workgroup. The arena kernels put one
// output row per workgroup, so every row re-reads the whole activation
// vector (10 KB) for 1.3 KB of weights; here a lane loads its 32-column
// group of x once and feeds four rows' codes with it. Same arena layout
// (eight storage segments, q2tp gate/up planes, q4tp down), same
// `(token, slot)` indexing as the arena kernels. ──
struct GqBank { words: array<u32> };
struct GqP { gpr: u32, inter: u32, slots: u32, mat16: u32, lim: f32, segment_slots: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read>       gq_gw  : binding_array<GqBank, 8>;
@group(0) @binding(1) var<storage, read>       gq_uw  : binding_array<GqBank, 8>;
@group(0) @binding(2) var<storage, read>       gq_x   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       gq_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> gq_act : array<f32>;
@group(1) @binding(0) var<uniform>             gq_p   : GqP;
var<workgroup> gq_pg: array<vec4<f32>, 64>;
var<workgroup> gq_pu: array<vec4<f32>, 64>;
fn gq_g32(seg: u32, o: u32) -> u32 { return gq_gw[seg].words[o]; }
fn gq_u32(seg: u32, o: u32) -> u32 { return gq_uw[seg].words[o]; }
fn gq_g16(seg: u32, o: u32) -> u32 { return (gq_g32(seg, o >> 1u) >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn gq_u16(seg: u32, o: u32) -> u32 { return (gq_u32(seg, o >> 1u) >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn gq_g8(seg: u32, o: u32) -> u32 { return (gq_g32(seg, o >> 2u) >> ((o & 3u) * 8u)) & 0xFFu; }
fn gq_u8(seg: u32, o: u32) -> u32 { return (gq_u32(seg, o >> 2u) >> ((o & 3u) * 8u)) & 0xFFu; }
fn gq_c4(w: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(f32((w >> sh) & 3u), f32((w >> (sh + 2u)) & 3u), f32((w >> (sh + 4u)) & 3u), f32((w >> (sh + 6u)) & 3u))
        - vec4<f32>(1.5);
}
fn gq_dot16(w: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>) -> f32 {
    return dot(gq_c4(w, 0u), x0) + dot(gq_c4(w, 8u), x1) + dot(gq_c4(w, 16u), x2) + dot(gq_c4(w, 24u), x3);
}
fn gq_scale(code: u32, pl: vec2<f32>) -> f32 {
    return select(exp2(pl.x + f32(max(code, 1u) - 1u) * pl.y), 0.0, code == 0u);
}
@compute @workgroup_size(64)
fn q4_gu_q2tp4(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let slot = wid.y;
    let batch = wid.z;
    let bslot = batch * gq_p.slots + slot;
    let flat = gq_sel[bslot];
    let seg = flat / gq_p.segment_slots;
    let local = flat - seg * gq_p.segment_slots;
    let gpr = gq_p.gpr;
    let rows = gq_p.inter;
    let base16 = local * gq_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let par0 = base16 + rows * gpr * 4u;
    let cod0 = (par0 + rows * 2u) * 2u;
    var gl: array<vec2<f32>, 4>;
    var ul: array<vec2<f32>, 4>;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let par16 = par0 + (row0 + k) * 2u;
        gl[k] = unpack2x16float(gq_g16(seg, par16) | (gq_g16(seg, par16 + 1u) << 16u));
        ul[k] = unpack2x16float(gq_u16(seg, par16) | (gq_u16(seg, par16 + 1u) << 16u));
    }
    var ag = vec4<f32>(0.0);
    var au = vec4<f32>(0.0);
    let xb4 = batch * gpr * 8u;
    for (var g = lid; g < gpr; g = g + 64u) {
        let xo = xb4 + g * 8u;
        let x0 = gq_x[xo];
        let x1 = gq_x[xo + 1u];
        let x2 = gq_x[xo + 2u];
        let x3 = gq_x[xo + 3u];
        let x4 = gq_x[xo + 4u];
        let x5 = gq_x[xo + 5u];
        let x6 = gq_x[xo + 6u];
        let x7 = gq_x[xo + 7u];
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let row = row0 + k;
            let cod8 = cod0 + row * cst;
            var cg = gq_g8(seg, cod8 + cb);
            var cu = gq_u8(seg, cod8 + cb);
            if (shf > 3u) {
                cg = cg | (gq_g8(seg, cod8 + cb + 1u) << 8u);
                cu = cu | (gq_u8(seg, cod8 + cb + 1u) << 8u);
            }
            let sg = gq_scale((cg >> shf) & 31u, gl[k]);
            let su = gq_scale((cu >> shf) & 31u, ul[k]);
            let w32 = (base16 + row * gpr * 4u + g * 4u) >> 1u;
            let dg = gq_dot16(gq_g32(seg, w32), x0, x1, x2, x3) + gq_dot16(gq_g32(seg, w32 + 1u), x4, x5, x6, x7);
            let du = gq_dot16(gq_u32(seg, w32), x0, x1, x2, x3) + gq_dot16(gq_u32(seg, w32 + 1u), x4, x5, x6, x7);
            ag[k] = ag[k] + sg * dg;
            au[k] = au[k] + su * du;
        }
    }
    gq_pg[lid] = ag;
    gq_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            gq_pg[lid] = gq_pg[lid] + gq_pg[lid + stride];
            gq_pu[lid] = gq_pu[lid] + gq_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid < 4u) {
        let row = row0 + lid;
        if (row < rows) {
            let gate = gq_pg[0][lid];
            let up = gq_pu[0][lid];
            gq_act[bslot * gq_p.inter + row] = (gate / (1.0 + exp(-gate))) * up;
        }
    }
}

struct GvP { gpr: u32, hidden: u32, slots: u32, mat16: u32, segment_slots: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read>       gv_w   : binding_array<GqBank, 8>;
@group(0) @binding(1) var<storage, read>       gv_act : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       gv_sel : array<u32>;
@group(0) @binding(3) var<storage, read>       gv_wt  : array<f32>;
@group(0) @binding(4) var<storage, read_write> gv_y   : array<f32>;
@group(1) @binding(0) var<uniform>             gv_p   : GvP;
var<workgroup> gv_pt: array<vec4<f32>, 64>;
fn gv_32(seg: u32, o: u32) -> u32 { return gv_w[seg].words[o]; }
fn gv_16(seg: u32, o: u32) -> u32 { return (gv_32(seg, o >> 1u) >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn gv_8(seg: u32, o: u32) -> u32 { return (gv_32(seg, o >> 2u) >> ((o & 3u) * 8u)) & 0xFFu; }
// The banks as 16-byte words (the same bindings 0/1): a 32-column group's
// nibbles in one load. Declared for the kernels that never touch the u32 views.
struct GqBank4 { w4: array<vec4<u32>> };
@group(0) @binding(0) var<storage, read>       gq4_gw : binding_array<GqBank4, 8>;
@group(0) @binding(1) var<storage, read>       gq4_uw : binding_array<GqBank4, 8>;
@group(0) @binding(0) var<storage, read>       gv4_w  : binding_array<GqBank4, 8>;
fn gq4_g32(seg: u32, o: u32) -> u32 { let v = gq4_gw[seg].w4[o >> 2u]; return v[o & 3u]; }
fn gq4_u32(seg: u32, o: u32) -> u32 { let v = gq4_uw[seg].w4[o >> 2u]; return v[o & 3u]; }
fn gq4_g8(seg: u32, o: u32) -> u32 { return (gq4_g32(seg, o >> 2u) >> ((o & 3u) * 8u)) & 0xFFu; }
fn gq4_u8(seg: u32, o: u32) -> u32 { return (gq4_u32(seg, o >> 2u) >> ((o & 3u) * 8u)) & 0xFFu; }
fn gq4_g16(seg: u32, o: u32) -> u32 { return (gq4_g32(seg, o >> 1u) >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn gq4_u16(seg: u32, o: u32) -> u32 { return (gq4_u32(seg, o >> 1u) >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn gq4_gc5(seg: u32, o: u32, shf: u32) -> u32 {
    var c = gq4_g8(seg, o);
    if (shf > 3u) { c = c | (gq4_g8(seg, o + 1u) << 8u); }
    return (c >> shf) & 31u;
}
fn gq4_uc5(seg: u32, o: u32, shf: u32) -> u32 {
    var c = gq4_u8(seg, o);
    if (shf > 3u) { c = c | (gq4_u8(seg, o + 1u) << 8u); }
    return (c >> shf) & 31u;
}
fn gq4_gpl(seg: u32, par16: u32) -> vec2<f32> {
    return unpack2x16float(gq4_g16(seg, par16) | (gq4_g16(seg, par16 + 1u) << 16u));
}
fn gq4_upl(seg: u32, par16: u32) -> vec2<f32> {
    return unpack2x16float(gq4_u16(seg, par16) | (gq4_u16(seg, par16 + 1u) << 16u));
}
// gq_gd32 of a group already loaded (the same sum order)
fn gq4_dot(q: vec4<u32>, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>,
           x4: vec4<f32>, x5: vec4<f32>, x6: vec4<f32>, x7: vec4<f32>) -> f32 {
    return gv_dot8(q.x, x0, x1) + gv_dot8(q.y, x2, x3) + gv_dot8(q.z, x4, x5) + gv_dot8(q.w, x6, x7);
}
fn gv4_32(seg: u32, o: u32) -> u32 { let v = gv4_w[seg].w4[o >> 2u]; return v[o & 3u]; }
fn gv4_16(seg: u32, o: u32) -> u32 { return (gv4_32(seg, o >> 1u) >> ((o & 1u) * 16u)) & 0xFFFFu; }
fn gv4_8(seg: u32, o: u32) -> u32 { return (gv4_32(seg, o >> 2u) >> ((o & 3u) * 8u)) & 0xFFu; }
fn gv_n4(w: u32, sh: u32) -> vec4<f32> {
    return vec4<f32>(f32((w >> sh) & 0xFu), f32((w >> (sh + 4u)) & 0xFu), f32((w >> (sh + 8u)) & 0xFu), f32((w >> (sh + 12u)) & 0xFu))
        - vec4<f32>(8.0);
}
fn gv_dot8(w: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
    return dot(gv_n4(w, 0u), xa) + dot(gv_n4(w, 16u), xb);
}
@compute @workgroup_size(64)
fn q4_dn_q4tp4(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let batch = wid.y;
    let gpr = gv_p.gpr;
    let rows = gv_p.hidden;
    let cst = (gpr * 5u + 7u) / 8u;
    let total = gv_p.slots * gpr;
    var acc = vec4<f32>(0.0);
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i - slot * gpr;
        let bslot = batch * gv_p.slots + slot;
        let flat = gv_sel[bslot];
        let seg = flat / gv_p.segment_slots;
        let local = flat - seg * gv_p.segment_slots;
        let base16 = local * gv_p.mat16;
        let par0 = base16 + rows * gpr * 8u;
        let cod0 = (par0 + rows * 2u) * 2u;
        let xo = (bslot * gpr + g) * 8u;
        let a0 = gv_act[xo];
        let a1 = gv_act[xo + 1u];
        let a2 = gv_act[xo + 2u];
        let a3 = gv_act[xo + 3u];
        let a4 = gv_act[xo + 4u];
        let a5 = gv_act[xo + 5u];
        let a6 = gv_act[xo + 6u];
        let a7 = gv_act[xo + 7u];
        let wt = gv_wt[bslot];
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let row = row0 + k;
            let par16 = par0 + row * 2u;
            let pl = unpack2x16float(gv_16(seg, par16) | (gv_16(seg, par16 + 1u) << 16u));
            let cod8 = cod0 + row * cst;
            var cv = gv_8(seg, cod8 + cb);
            if (shf > 3u) { cv = cv | (gv_8(seg, cod8 + cb + 1u) << 8u); }
            let scale = exp2(pl.x + f32((cv >> shf) & 31u) * pl.y);
            let t16 = base16 + (row * gpr + g) * 8u;
            var d = 0.0;
            d = d + gv_dot8(gv_16(seg, t16) | (gv_16(seg, t16 + 1u) << 16u), a0, a1);
            d = d + gv_dot8(gv_16(seg, t16 + 2u) | (gv_16(seg, t16 + 3u) << 16u), a2, a3);
            d = d + gv_dot8(gv_16(seg, t16 + 4u) | (gv_16(seg, t16 + 5u) << 16u), a4, a5);
            d = d + gv_dot8(gv_16(seg, t16 + 6u) | (gv_16(seg, t16 + 7u) << 16u), a6, a7);
            acc[k] = acc[k] + wt * scale * d;
        }
    }
    gv_pt[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { gv_pt[lid] = gv_pt[lid] + gv_pt[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid < 4u) {
        let row = row0 + lid;
        if (row < rows) {
            gv_y[batch * gv_p.hidden + row] = gv_pt[0][lid];
        }
    }
}

// ── The q4tp twin of q4_gu_q2tp4: gate/up planes in q4tp (16 nibble bytes
// a 32-column group). Same bindings, uniform, dispatch geometry and output
// as q4_gu_q2tp4; each group decoded as `dsv4_global_gate_up_q4tp` decodes
// it (nibble − 8, scale exp2(lo + code·step), no zero rung). The four rows
// ride in the lanes of vec4s, so nothing private is indexed at run time.
// Word reads need `mat16` even, which inter % 4 == 0 gives (every section
// of a q4tp matrix is then a whole number of words). ──
fn gq_gc5(seg: u32, o: u32, shf: u32) -> u32 {
    var c = gq_g8(seg, o);
    if (shf > 3u) { c = c | (gq_g8(seg, o + 1u) << 8u); }
    return (c >> shf) & 31u;
}
fn gq_uc5(seg: u32, o: u32, shf: u32) -> u32 {
    var c = gq_u8(seg, o);
    if (shf > 3u) { c = c | (gq_u8(seg, o + 1u) << 8u); }
    return (c >> shf) & 31u;
}
// one row's 32-column group: four nibble words from word `w`
fn gq_gd32(seg: u32, w: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>,
           x4: vec4<f32>, x5: vec4<f32>, x6: vec4<f32>, x7: vec4<f32>) -> f32 {
    return gv_dot8(gq_g32(seg, w), x0, x1) + gv_dot8(gq_g32(seg, w + 1u), x2, x3)
         + gv_dot8(gq_g32(seg, w + 2u), x4, x5) + gv_dot8(gq_g32(seg, w + 3u), x6, x7);
}
fn gq_ud32(seg: u32, w: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>, x3: vec4<f32>,
           x4: vec4<f32>, x5: vec4<f32>, x6: vec4<f32>, x7: vec4<f32>) -> f32 {
    return gv_dot8(gq_u32(seg, w), x0, x1) + gv_dot8(gq_u32(seg, w + 1u), x2, x3)
         + gv_dot8(gq_u32(seg, w + 2u), x4, x5) + gv_dot8(gq_u32(seg, w + 3u), x6, x7);
}
// a row's (lo, step) pair at u16 offset `par16` of the gate / up plane
fn gq_gpl(seg: u32, par16: u32) -> vec2<f32> {
    return unpack2x16float(gq_g16(seg, par16) | (gq_g16(seg, par16 + 1u) << 16u));
}
fn gq_upl(seg: u32, par16: u32) -> vec2<f32> {
    return unpack2x16float(gq_u16(seg, par16) | (gq_u16(seg, par16 + 1u) << 16u));
}
@compute @workgroup_size(64)
fn q4_gu_q4tp4(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let slot = wid.y;
    let batch = wid.z;
    let bslot = batch * gq_p.slots + slot;
    let flat = gq_sel[bslot];
    let seg = flat / gq_p.segment_slots;
    let local = flat - seg * gq_p.segment_slots;
    let gpr = gq_p.gpr;
    let rows = gq_p.inter;
    let base16 = local * gq_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let par0 = base16 + rows * gpr * 8u + row0 * 2u;
    let cod0 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row0 * cst;
    let g0 = gq_gpl(seg, par0);
    let g1 = gq_gpl(seg, par0 + 2u);
    let g2 = gq_gpl(seg, par0 + 4u);
    let g3 = gq_gpl(seg, par0 + 6u);
    let u0 = gq_upl(seg, par0);
    let u1 = gq_upl(seg, par0 + 2u);
    let u2 = gq_upl(seg, par0 + 4u);
    let u3 = gq_upl(seg, par0 + 6u);
    let glo = vec4<f32>(g0.x, g1.x, g2.x, g3.x);
    let gst = vec4<f32>(g0.y, g1.y, g2.y, g3.y);
    let ulo = vec4<f32>(u0.x, u1.x, u2.x, u3.x);
    let ust = vec4<f32>(u0.y, u1.y, u2.y, u3.y);
    // nibble word of (row0, group 0); the next row is gpr·4 words on
    let wr = gpr * 4u;
    let w0 = (base16 >> 1u) + row0 * wr;
    var ag = vec4<f32>(0.0);
    var au = vec4<f32>(0.0);
    let xb4 = batch * gpr * 8u;
    for (var g = lid; g < gpr; g = g + 64u) {
        let xo = xb4 + g * 8u;
        let x0 = gq_x[xo];
        let x1 = gq_x[xo + 1u];
        let x2 = gq_x[xo + 2u];
        let x3 = gq_x[xo + 3u];
        let x4 = gq_x[xo + 4u];
        let x5 = gq_x[xo + 5u];
        let x6 = gq_x[xo + 6u];
        let x7 = gq_x[xo + 7u];
        let bit = g * 5u;
        let cb = cod0 + (bit >> 3u);
        let shf = bit & 7u;
        let cg = vec4<u32>(gq_gc5(seg, cb, shf), gq_gc5(seg, cb + cst, shf),
                           gq_gc5(seg, cb + 2u * cst, shf), gq_gc5(seg, cb + 3u * cst, shf));
        let cu = vec4<u32>(gq_uc5(seg, cb, shf), gq_uc5(seg, cb + cst, shf),
                           gq_uc5(seg, cb + 2u * cst, shf), gq_uc5(seg, cb + 3u * cst, shf));
        let sg = exp2(glo + vec4<f32>(cg) * gst);
        let su = exp2(ulo + vec4<f32>(cu) * ust);
        let w = w0 + g * 4u;
        let dg = vec4<f32>(gq_gd32(seg, w, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_gd32(seg, w + wr, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_gd32(seg, w + 2u * wr, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_gd32(seg, w + 3u * wr, x0, x1, x2, x3, x4, x5, x6, x7));
        let du = vec4<f32>(gq_ud32(seg, w, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_ud32(seg, w + wr, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_ud32(seg, w + 2u * wr, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_ud32(seg, w + 3u * wr, x0, x1, x2, x3, x4, x5, x6, x7));
        ag = ag + sg * dg;
        au = au + su * du;
    }
    gq_pg[lid] = ag;
    gq_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            gq_pg[lid] = gq_pg[lid] + gq_pg[lid + stride];
            gq_pu[lid] = gq_pu[lid] + gq_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid < 4u) {
        let row = row0 + lid;
        if (row < rows) {
            let gate = gq_pg[0][lid];
            let up = gq_pu[0][lid];
            gq_act[bslot * gq_p.inter + row] = (gate / (1.0 + exp(-gate))) * up;
        }
    }
}

// q4_gu_q4tp4 with the nibble words read as vec4s (bank offsets 4-word aligned).
@compute @workgroup_size(64)
fn q4_gu_q4tp4v(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let slot = wid.y;
    let batch = wid.z;
    let bslot = batch * gq_p.slots + slot;
    let flat = gq_sel[bslot];
    let seg = flat / gq_p.segment_slots;
    let local = flat - seg * gq_p.segment_slots;
    let gpr = gq_p.gpr;
    let rows = gq_p.inter;
    let base16 = local * gq_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let par0 = base16 + rows * gpr * 8u + row0 * 2u;
    let cod0 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row0 * cst;
    let g0 = gq4_gpl(seg, par0);
    let g1 = gq4_gpl(seg, par0 + 2u);
    let g2 = gq4_gpl(seg, par0 + 4u);
    let g3 = gq4_gpl(seg, par0 + 6u);
    let u0 = gq4_upl(seg, par0);
    let u1 = gq4_upl(seg, par0 + 2u);
    let u2 = gq4_upl(seg, par0 + 4u);
    let u3 = gq4_upl(seg, par0 + 6u);
    let glo = vec4<f32>(g0.x, g1.x, g2.x, g3.x);
    let gst = vec4<f32>(g0.y, g1.y, g2.y, g3.y);
    let ulo = vec4<f32>(u0.x, u1.x, u2.x, u3.x);
    let ust = vec4<f32>(u0.y, u1.y, u2.y, u3.y);
    // nibble word of (row0, group 0); the next row is gpr·4 words on
    let wr = gpr * 4u;
    let w0 = (base16 >> 1u) + row0 * wr;
    var ag = vec4<f32>(0.0);
    var au = vec4<f32>(0.0);
    let xb4 = batch * gpr * 8u;
    for (var g = lid; g < gpr; g = g + 64u) {
        let xo = xb4 + g * 8u;
        let x0 = gq_x[xo];
        let x1 = gq_x[xo + 1u];
        let x2 = gq_x[xo + 2u];
        let x3 = gq_x[xo + 3u];
        let x4 = gq_x[xo + 4u];
        let x5 = gq_x[xo + 5u];
        let x6 = gq_x[xo + 6u];
        let x7 = gq_x[xo + 7u];
        let bit = g * 5u;
        let cb = cod0 + (bit >> 3u);
        let shf = bit & 7u;
        let cg = vec4<u32>(gq4_gc5(seg, cb, shf), gq4_gc5(seg, cb + cst, shf),
                           gq4_gc5(seg, cb + 2u * cst, shf), gq4_gc5(seg, cb + 3u * cst, shf));
        let cu = vec4<u32>(gq4_uc5(seg, cb, shf), gq4_uc5(seg, cb + cst, shf),
                           gq4_uc5(seg, cb + 2u * cst, shf), gq4_uc5(seg, cb + 3u * cst, shf));
        let sg = exp2(glo + vec4<f32>(cg) * gst);
        let su = exp2(ulo + vec4<f32>(cu) * ust);
        let w = w0 + g * 4u;
        let dg = vec4<f32>(gq4_dot(gq4_gw[seg].w4[(w) >> 2u], x0, x1, x2, x3, x4, x5, x6, x7),
                           gq4_dot(gq4_gw[seg].w4[(w + wr) >> 2u], x0, x1, x2, x3, x4, x5, x6, x7),
                           gq4_dot(gq4_gw[seg].w4[(w + 2u * wr) >> 2u], x0, x1, x2, x3, x4, x5, x6, x7),
                           gq4_dot(gq4_gw[seg].w4[(w + 3u * wr) >> 2u], x0, x1, x2, x3, x4, x5, x6, x7));
        let du = vec4<f32>(gq4_dot(gq4_uw[seg].w4[(w) >> 2u], x0, x1, x2, x3, x4, x5, x6, x7),
                           gq4_dot(gq4_uw[seg].w4[(w + wr) >> 2u], x0, x1, x2, x3, x4, x5, x6, x7),
                           gq4_dot(gq4_uw[seg].w4[(w + 2u * wr) >> 2u], x0, x1, x2, x3, x4, x5, x6, x7),
                           gq4_dot(gq4_uw[seg].w4[(w + 3u * wr) >> 2u], x0, x1, x2, x3, x4, x5, x6, x7));
        ag = ag + sg * dg;
        au = au + su * du;
    }
    gq_pg[lid] = ag;
    gq_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            gq_pg[lid] = gq_pg[lid] + gq_pg[lid + stride];
            gq_pu[lid] = gq_pu[lid] + gq_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid < 4u) {
        let row = row0 + lid;
        if (row < rows) {
            let gate = gq_pg[0][lid];
            let up = gq_pu[0][lid];
            gq_act[bslot * gq_p.inter + row] = (gate / (1.0 + exp(-gate))) * up;
        }
    }
}


// ── Prompt frames past eight tokens: the frame's (token, slot) winners sorted by
// arena slot (`q4g_sort`), so the workgroups that read one expert's weights run
// back to back and its rows come from L2 instead of DRAM once per token. ──
@group(0) @binding(5) var<storage, read> gq_ord : array<u32>;
@compute @workgroup_size(64)
fn q4g_gu_q4tp4(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let bslot = gq_ord[wid.y];
    let batch = bslot / gq_p.slots;
    let flat = gq_sel[bslot];
    if (flat == 0xFFFFFFFFu) { return; }
    let seg = flat / gq_p.segment_slots;
    let local = flat - seg * gq_p.segment_slots;
    let gpr = gq_p.gpr;
    let rows = gq_p.inter;
    let base16 = local * gq_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let par0 = base16 + rows * gpr * 8u + row0 * 2u;
    let cod0 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + row0 * cst;
    let g0 = gq_gpl(seg, par0);
    let g1 = gq_gpl(seg, par0 + 2u);
    let g2 = gq_gpl(seg, par0 + 4u);
    let g3 = gq_gpl(seg, par0 + 6u);
    let u0 = gq_upl(seg, par0);
    let u1 = gq_upl(seg, par0 + 2u);
    let u2 = gq_upl(seg, par0 + 4u);
    let u3 = gq_upl(seg, par0 + 6u);
    let glo = vec4<f32>(g0.x, g1.x, g2.x, g3.x);
    let gst = vec4<f32>(g0.y, g1.y, g2.y, g3.y);
    let ulo = vec4<f32>(u0.x, u1.x, u2.x, u3.x);
    let ust = vec4<f32>(u0.y, u1.y, u2.y, u3.y);
    // nibble word of (row0, group 0); the next row is gpr·4 words on
    let wr = gpr * 4u;
    let w0 = (base16 >> 1u) + row0 * wr;
    var ag = vec4<f32>(0.0);
    var au = vec4<f32>(0.0);
    let xb4 = batch * gpr * 8u;
    for (var g = lid; g < gpr; g = g + 64u) {
        let xo = xb4 + g * 8u;
        let x0 = gq_x[xo];
        let x1 = gq_x[xo + 1u];
        let x2 = gq_x[xo + 2u];
        let x3 = gq_x[xo + 3u];
        let x4 = gq_x[xo + 4u];
        let x5 = gq_x[xo + 5u];
        let x6 = gq_x[xo + 6u];
        let x7 = gq_x[xo + 7u];
        let bit = g * 5u;
        let cb = cod0 + (bit >> 3u);
        let shf = bit & 7u;
        let cg = vec4<u32>(gq_gc5(seg, cb, shf), gq_gc5(seg, cb + cst, shf),
                           gq_gc5(seg, cb + 2u * cst, shf), gq_gc5(seg, cb + 3u * cst, shf));
        let cu = vec4<u32>(gq_uc5(seg, cb, shf), gq_uc5(seg, cb + cst, shf),
                           gq_uc5(seg, cb + 2u * cst, shf), gq_uc5(seg, cb + 3u * cst, shf));
        let sg = exp2(glo + vec4<f32>(cg) * gst);
        let su = exp2(ulo + vec4<f32>(cu) * ust);
        let w = w0 + g * 4u;
        let dg = vec4<f32>(gq_gd32(seg, w, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_gd32(seg, w + wr, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_gd32(seg, w + 2u * wr, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_gd32(seg, w + 3u * wr, x0, x1, x2, x3, x4, x5, x6, x7));
        let du = vec4<f32>(gq_ud32(seg, w, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_ud32(seg, w + wr, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_ud32(seg, w + 2u * wr, x0, x1, x2, x3, x4, x5, x6, x7),
                           gq_ud32(seg, w + 3u * wr, x0, x1, x2, x3, x4, x5, x6, x7));
        ag = ag + sg * dg;
        au = au + su * du;
    }
    gq_pg[lid] = ag;
    gq_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            gq_pg[lid] = gq_pg[lid] + gq_pg[lid + stride];
            gq_pu[lid] = gq_pu[lid] + gq_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid < 4u) {
        let row = row0 + lid;
        if (row < rows) {
            let gate = gq_pg[0][lid];
            let up = gq_pu[0][lid];
            gq_act[bslot * gq_p.inter + row] = (gate / (1.0 + exp(-gate))) * up;
        }
    }
}


// The down projection per sorted winner: sixteen rows a workgroup, four lanes a
// row over its 32-column groups; the weighted rows land per winner and
// `q4g_sum` adds a token's winners.
@group(0) @binding(5) var<storage, read> gv_ord : array<u32>;
var<workgroup> gd_red: array<f32, 64>;
@compute @workgroup_size(64)
fn q4g_dn_q4tp4(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 16u;
    let bslot = gv_ord[wid.y];
    let gpr = gv_p.gpr;
    let rows = gv_p.hidden;
    let cst = (gpr * 5u + 7u) / 8u;
    let flat = gv_sel[bslot];
    let r = lid & 15u;
    let gl = lid >> 4u;
    let row = row0 + r;
    var acc = 0.0;
    if (flat != 0xFFFFFFFFu && row < rows) {
        let seg = flat / gv_p.segment_slots;
        let local = flat - seg * gv_p.segment_slots;
        let base16 = local * gv_p.mat16;
        let par0 = base16 + rows * gpr * 8u;
        let cod0 = (par0 + rows * 2u) * 2u;
        let par16 = par0 + row * 2u;
        let pl = unpack2x16float(gv_16(seg, par16) | (gv_16(seg, par16 + 1u) << 16u));
        let cod8 = cod0 + row * cst;
        for (var g = gl; g < gpr; g = g + 4u) {
            let xo = (bslot * gpr + g) * 8u;
            let bit = g * 5u;
            let cb = bit >> 3u;
            let shf = bit & 7u;
            var cv = gv_8(seg, cod8 + cb);
            if (shf > 3u) { cv = cv | (gv_8(seg, cod8 + cb + 1u) << 8u); }
            let scale = exp2(pl.x + f32((cv >> shf) & 31u) * pl.y);
            let t16 = base16 + (row * gpr + g) * 8u;
            var d = 0.0;
            d = d + gv_dot8(gv_16(seg, t16) | (gv_16(seg, t16 + 1u) << 16u), gv_act[xo], gv_act[xo + 1u]);
            d = d + gv_dot8(gv_16(seg, t16 + 2u) | (gv_16(seg, t16 + 3u) << 16u), gv_act[xo + 2u], gv_act[xo + 3u]);
            d = d + gv_dot8(gv_16(seg, t16 + 4u) | (gv_16(seg, t16 + 5u) << 16u), gv_act[xo + 4u], gv_act[xo + 5u]);
            d = d + gv_dot8(gv_16(seg, t16 + 6u) | (gv_16(seg, t16 + 7u) << 16u), gv_act[xo + 6u], gv_act[xo + 7u]);
            acc = acc + scale * d;
        }
    }
    gd_red[lid] = acc;
    workgroupBarrier();
    if (lid < 16u && row0 + lid < rows && flat != 0xFFFFFFFFu) {
        let v = gd_red[lid] + gd_red[lid + 16u] + gd_red[lid + 32u] + gd_red[lid + 48u];
        gv_y[bslot * rows + row0 + lid] = gv_wt[bslot] * v;
    }
}

// A token's MoE output: the sum of its winners' rows (empty ranks skipped).
struct QsmP { hidden: u32, slots: u32, nt: u32, ys: u32 };
@group(0) @binding(0) var<storage, read>       qsm_part : array<f32>;
@group(0) @binding(1) var<storage, read>       qsm_sel  : array<u32>;
@group(0) @binding(2) var<storage, read_write> qsm_y    : array<f32>;
@group(0) @binding(3) var<uniform>             qsm_p    : QsmP;
@compute @workgroup_size(256)
fn q4g_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let t = gid.y;
    if (d >= qsm_p.hidden || t >= qsm_p.nt) { return; }
    var v = 0.0;
    for (var k = 0u; k < qsm_p.slots; k = k + 1u) {
        let b = t * qsm_p.slots + k;
        if (qsm_sel[b] != 0xFFFFFFFFu) { v = v + qsm_part[b * qsm_p.hidden + d]; }
    }
    qsm_y[t * qsm_p.ys + d] = v;
}

// The frame's winners ordered by (arena slot, winner index): one workgroup, a
// bitonic sort of up to 2048 keys in workgroup memory.
struct QsoP { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       qso_sel : array<u32>;
@group(0) @binding(1) var<storage, read_write> qso_ord : array<u32>;
@group(0) @binding(2) var<uniform>             qso_p   : QsoP;
var<workgroup> qso_k: array<u32, 4096>;
var<workgroup> qso_v: array<u32, 4096>;
@compute @workgroup_size(1024)
fn q4g_sort(@builtin(local_invocation_index) lid: u32) {
    let n = qso_p.n;
    for (var h = 0u; h < 4u; h = h + 1u) {
        let i = lid + h * 1024u;
        qso_k[i] = select(0xFFFFFFFFu, qso_sel[min(i, n - 1u)], i < n);
        qso_v[i] = i;
    }
    workgroupBarrier();
    for (var k = 2u; k <= 4096u; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            for (var h = 0u; h < 4u; h = h + 1u) {
                let i = lid + h * 1024u;
                let l = i ^ j;
                if (l > i) {
                    let ki = qso_k[i];
                    let kl = qso_k[l];
                    let vi = qso_v[i];
                    let vl = qso_v[l];
                    let gt = ki > kl || (ki == kl && vi > vl);
                    let up = (i & k) == 0u;
                    if (gt == up) {
                        qso_k[i] = kl;
                        qso_k[l] = ki;
                        qso_v[i] = vl;
                        qso_v[l] = vi;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    for (var h = 0u; h < 4u; h = h + 1u) {
        let i = lid + h * 1024u;
        if (i < n) { qso_ord[i] = qso_v[i]; }
    }
}


// The sorted winners' segments for the matrix-unit expert GEMMs: one per
// arena slot, (slot, first sorted winner, count, 0), the count of segments at
// [4096]; up to 4096 winners (`q4g_sort`'s order), four consecutive a lane.
struct QsgP { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       qsg_sel : array<u32>;
@group(0) @binding(1) var<storage, read>       qsg_ord : array<u32>;
@group(0) @binding(2) var<storage, read_write> qsg_seg : array<vec4<u32>>;
@group(0) @binding(3) var<uniform>             qsg_p   : QsgP;
var<workgroup> qsg_sc: array<u32, 1024>;
var<workgroup> qsg_nv: atomic<u32>;
@compute @workgroup_size(1024)
fn q4g_segs(@builtin(local_invocation_index) lid: u32) {
    let n = qsg_p.n;
    if (lid == 0u) { atomicStore(&qsg_nv, 0u); }
    let i0 = lid * 4u;
    var prev = 0xFFFFFFFFu;
    if (i0 > 0u && i0 - 1u < n) { prev = qsg_sel[qsg_ord[i0 - 1u]]; }
    var key = vec4<u32>(0xFFFFFFFFu);
    var st = vec4<u32>(0u);
    var nval = 0u;
    for (var h = 0u; h < 4u; h = h + 1u) {
        let i = i0 + h;
        var k = 0xFFFFFFFFu;
        if (i < n) { k = qsg_sel[qsg_ord[i]]; }
        key[h] = k;
        let valid = k != 0xFFFFFFFFu;
        st[h] = u32(valid && (i == 0u || k != prev));
        nval = nval + u32(valid);
        prev = k;
    }
    let nst = st.x + st.y + st.z + st.w;
    workgroupBarrier();
    atomicAdd(&qsg_nv, nval);
    qsg_sc[lid] = nst;
    workgroupBarrier();
    for (var o = 1u; o < 1024u; o = o << 1u) {
        var v = qsg_sc[lid];
        if (lid >= o) { v = v + qsg_sc[lid - o]; }
        workgroupBarrier();
        qsg_sc[lid] = v;
        workgroupBarrier();
    }
    var at = qsg_sc[lid] - nst;
    let nseg = qsg_sc[1023];
    let nvalid = atomicLoad(&qsg_nv);
    for (var h = 0u; h < 4u; h = h + 1u) {
        if (st[h] == 1u) {
            qsg_seg[at] = vec4<u32>(key[h], i0 + h, 0u, 0u);
            at = at + 1u;
        }
    }
    if (lid == 0u) { qsg_seg[4096] = vec4<u32>(nseg, 0u, 0u, 0u); }
    storageBarrier();
    workgroupBarrier();
    for (var sid = lid; sid < nseg; sid = sid + 1024u) {
        let e = qsg_seg[sid];
        var en = nvalid;
        if (sid + 1u < nseg) { en = qsg_seg[sid + 1u].y; }
        qsg_seg[sid] = vec4<u32>(e.x, e.y, en - e.y, 0u);
    }
}

// Frames past 4096 winners (`q4g_sort` holds 4096 keys in workgroup memory): a counting
// sort by arena slot in global memory, into `q4g_segs`' layout — counts, one workgroup's
// scan into each slot's first sorted position and the segment table, scatter. Winners of
// one slot come out in any order: every winner's rows are computed on their own (the
// GEMM rows are independent), so the results do not depend on it.
struct QcsP { n: u32, nslots: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       qcs_sel : array<u32>;
@group(0) @binding(1) var<storage, read_write> qcs_cnt : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> qcs_ord : array<u32>;
@group(0) @binding(3) var<storage, read_write> qcs_seg : array<vec4<u32>>;
@group(0) @binding(4) var<uniform>             qcs_p   : QcsP;
// every kernel names all five bindings (the auto layout keeps only the used ones, and
// the four share one bind group); the branch never runs
fn qcs_touch() {
    if (qcs_p._a == 0xFFFFFFFFu) {
        qcs_ord[0] = qcs_sel[0] + atomicLoad(&qcs_cnt[0]);
        qcs_seg[0] = vec4<u32>(0u);
    }
}
@compute @workgroup_size(256)
fn q4g_cs_clear(@builtin(global_invocation_id) g: vec3<u32>) {
    qcs_touch();
    if (g.x < qcs_p.nslots) { atomicStore(&qcs_cnt[g.x], 0u); }
}
@compute @workgroup_size(256)
fn q4g_cs_count(@builtin(global_invocation_id) g: vec3<u32>) {
    qcs_touch();
    if (g.x >= qcs_p.n) { return; }
    let sl = qcs_sel[g.x];
    if (sl < qcs_p.nslots) { atomicAdd(&qcs_cnt[sl], 1u); }
}
var<workgroup> qcs_sc: array<u32, 1024>;
@compute @workgroup_size(1024)
fn q4g_cs_scan(@builtin(local_invocation_index) lid: u32) {
    qcs_touch();
    let ns = qcs_p.nslots;
    let per = (ns + 1023u) / 1024u;
    let a = lid * per;
    // this lane's bins: their winners (low 16 bits) and how many hold any (high 16)
    var mine = 0u;
    for (var j = 0u; j < per; j = j + 1u) {
        let sl = a + j;
        if (sl < ns) {
            let c = atomicLoad(&qcs_cnt[sl]);
            mine = mine + c + (select(0u, 1u, c > 0u) << 16u);
        }
    }
    qcs_sc[lid] = mine;
    workgroupBarrier();
    for (var o = 1u; o < 1024u; o = o << 1u) {
        var v = qcs_sc[lid];
        if (lid >= o) { v = v + qcs_sc[lid - o]; }
        workgroupBarrier();
        qcs_sc[lid] = v;
        workgroupBarrier();
    }
    let ex = qcs_sc[lid] - mine;
    var st = ex & 0xFFFFu;
    var sid = ex >> 16u;
    for (var j = 0u; j < per; j = j + 1u) {
        let sl = a + j;
        if (sl < ns) {
            let c = atomicLoad(&qcs_cnt[sl]);
            if (c > 0u) {
                qcs_seg[sid] = vec4<u32>(sl, st, c, 0u);
                sid = sid + 1u;
            }
            atomicStore(&qcs_cnt[sl], st);
            st = st + c;
        }
    }
    if (lid == 1023u) { qcs_seg[4096] = vec4<u32>(qcs_sc[1023] >> 16u, 0u, 0u, 0u); }
}
@compute @workgroup_size(256)
fn q4g_cs_scatter(@builtin(global_invocation_id) g: vec3<u32>) {
    qcs_touch();
    if (g.x >= qcs_p.n) { return; }
    let sl = qcs_sel[g.x];
    if (sl < qcs_p.nslots) { qcs_ord[atomicAdd(&qcs_cnt[sl], 1u)] = g.x; }
}


// ── O(1) layers: the frame's rows packed for the engine's far-field kernels
// (dense `[t][nh·hd]` and `[t][nkv·hd]`), and while the prompt runs exact the
// query trace the seal reads: one row in `stride` at `pos / stride`.
struct ToP { nh: u32, nkv: u32, hd: u32, qs: u32, ks: u32, vs: u32, trace: u32, stride: u32,
             qcap: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32, _g: u32 };
@group(0) @binding(0) var<storage, read>       o1p_q   : array<f32>;
@group(0) @binding(1) var<storage, read>       o1p_k   : array<f32>;
@group(0) @binding(2) var<storage, read>       o1p_v   : array<f32>;
@group(0) @binding(3) var<storage, read_write> o1p_pq  : array<f32>;
@group(0) @binding(4) var<storage, read_write> o1p_pk  : array<f32>;
@group(0) @binding(5) var<storage, read_write> o1p_pv  : array<f32>;
@group(0) @binding(6) var<storage, read_write> o1p_qt  : array<f32>;
@group(0) @binding(7) var<storage, read>       o1p_tab : array<vec4<u32>>;
@group(0) @binding(8) var<uniform>             o1p_p   : ToP;
@compute @workgroup_size(256)
fn q4t_o1_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    let nq = o1p_p.nh * o1p_p.hd;
    let nk = o1p_p.nkv * o1p_p.hd;
    if (i < nq) {
        let q = o1p_q[t * o1p_p.qs + i];
        o1p_pq[t * nq + i] = q;
        if (o1p_p.trace != 0u) {
            let pos = o1p_tab[t].x;
            if (pos % o1p_p.stride == 0u) {
                let r = pos / o1p_p.stride;
                if (r < o1p_p.qcap) { o1p_qt[r * nq + i] = q; }
            }
        }
    }
    if (i < nk) {
        o1p_pk[t * nk + i] = o1p_k[t * o1p_p.ks + i];
        o1p_pv[t * nk + i] = o1p_v[t * o1p_p.vs + i];
    }
}

// ── O(1) layers: the far-field rows through the sigmoid output gate (as
// `q4t_attend` applies it), into the frame's strided attention output.
struct TogP { n: u32, gs: u32, os: u32, _a: u32 };
@group(0) @binding(0) var<storage, read>       o1g_in   : array<f32>;
@group(0) @binding(1) var<storage, read>       o1g_gate : array<f32>;
@group(0) @binding(2) var<storage, read_write> o1g_out  : array<f32>;
@group(0) @binding(3) var<uniform>             o1g_p    : TogP;
@compute @workgroup_size(256)
fn q4t_o1_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= o1g_p.n) { return; }
    let g = o1g_gate[t * o1g_p.gs + i];
    o1g_out[t * o1g_p.os + i] = o1g_in[t * o1g_p.n + i] * (1.0 / (1.0 + exp(-g)));
}
// ── The routed experts' hot neurons (`hot-bake --experts`): every expert carries a
// tube of `h` neurons at eight bits (q8_2f: int8 rows, f16 row scales, f16 column
// scales) whose rows are zero in its own triple. The banks hold the experts' tubes
// one after another at a fixed segment, gate and up `[h, hidden]`, down `[hidden, h]`.
// `q4t_tube_gu`: one workgroup per (winner j, token t, block of 32 rows), eight lanes a row:
// act[t][j][i] = silu(gate_i·x)·(up_i·x). `q4t_tube_dn`: a lane per (t, d) adds
// Σ_j w_j Σ_i act[t][j][i]·down[d][i] to the frame's MoE output.
// `cols` activation columns a token: the routed winners (`top_k`), or one for the shared
// expert's tube (`sh` = 1: bank entry 0, weight `tt_w[t·is + wi]`, the route's σ).
// `tt_slot[e]` is expert e's entry in the bank, 0xFFFFFFFF when it has no tube.
struct TtP { hidden: u32, h: u32, top_k: u32, experts: u32, xs: u32, cs: u32, gseg: u32, dseg: u32,
             os: u32, cols: u32, is: u32, wi: u32, sh: u32, _e: u32, _f: u32, _g: u32 };
@group(0) @binding(0) var<storage, read>       tt_g   : array<u32>;
@group(0) @binding(1) var<storage, read>       tt_u   : array<u32>;
@group(0) @binding(2) var<storage, read>       tt_x   : array<f32>;
@group(0) @binding(3) var<storage, read>       tt_win : array<u32>;
@group(0) @binding(4) var<storage, read_write> tt_act : array<f32>;
@group(0) @binding(5) var<uniform>             tt_p   : TtP;
@group(0) @binding(6) var<storage, read>       tt_slot: array<u32>;
var<workgroup> tt_rg: array<f32, 256>;
var<workgroup> tt_ru: array<f32, 256>;
fn tt_i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    return vec4<f32>(f32((s << 24u) >> 24u), f32((s << 16u) >> 24u), f32((s << 8u) >> 24u), f32(s >> 24u));
}
// f16 at half index `hi` of the gate bank / the up bank
fn tt_gh(hi: u32) -> f32 {
    let v = unpack2x16float(tt_g[hi >> 1u]);
    if ((hi & 1u) == 0u) { return v.x; }
    return v.y;
}
fn tt_uh(hi: u32) -> f32 {
    let v = unpack2x16float(tt_u[hi >> 1u]);
    if ((hi & 1u) == 0u) { return v.x; }
    return v.y;
}
@compute @workgroup_size(256)
fn q4t_tube_gu(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let j = wid.x;
    let t = wid.y;
    let k = tt_p.top_k;
    let h = tt_p.h;
    let hid = tt_p.hidden;
    var e = 0u;
    if (tt_p.sh == 0u) {
        let rec = t * tt_p.cs + 2u * k + 2u * j;
        e = tt_win[rec];
    }
    let abase = (t * tt_p.cols + j) * h;
    let sub = lid & 7u;
    // Workgroup z owns rows r0..r0+32 (a whole expert's 640 rows are 20 workgroups).
    let r0 = wid.z * 32u;
    var slot = 0xFFFFFFFFu;
    if (e < tt_p.experts) { slot = tt_slot[e]; }
    if (slot == 0xFFFFFFFFu) {
        let i = r0 + lid;
        if (lid < 32u && i < h) { tt_act[abase + i] = 0.0; }
        return;
    }
    let qw = hid >> 2u;
    let seg = slot * tt_p.gseg;         // words
    let rsh = seg * 2u + ((h * hid) >> 1u); // half index of the row scales
    let csh = rsh + h;                   // half index of the column scales
    {
        let row = r0 + (lid >> 3u);
        var ag = 0.0;
        var au = 0.0;
        if (row < h) {
            let rw = seg + row * qw;
            var w = sub;
            loop {
                if (w >= qw) { break; }
                let xo = t * tt_p.xs + 4u * w;
                let xv = vec4<f32>(tt_x[xo], tt_x[xo + 1u], tt_x[xo + 2u], tt_x[xo + 3u]);
                let cg = vec4<f32>(tt_gh(csh + 4u * w), tt_gh(csh + 4u * w + 1u), tt_gh(csh + 4u * w + 2u), tt_gh(csh + 4u * w + 3u));
                let cu = vec4<f32>(tt_uh(csh + 4u * w), tt_uh(csh + 4u * w + 1u), tt_uh(csh + 4u * w + 2u), tt_uh(csh + 4u * w + 3u));
                ag = ag + dot(tt_i8x4(tt_g[rw + w]), xv * cg);
                au = au + dot(tt_i8x4(tt_u[rw + w]), xv * cu);
                w = w + 8u;
            }
        }
        tt_rg[lid] = ag;
        tt_ru[lid] = au;
        workgroupBarrier();
        if (sub < 4u) { tt_rg[lid] = tt_rg[lid] + tt_rg[lid + 4u]; tt_ru[lid] = tt_ru[lid] + tt_ru[lid + 4u]; }
        workgroupBarrier();
        if (sub < 2u) { tt_rg[lid] = tt_rg[lid] + tt_rg[lid + 2u]; tt_ru[lid] = tt_ru[lid] + tt_ru[lid + 2u]; }
        workgroupBarrier();
        if (sub == 0u && row < h) {
            let g = (tt_rg[lid] + tt_rg[lid + 1u]) * tt_gh(rsh + row);
            let u = (tt_ru[lid] + tt_ru[lid + 1u]) * tt_uh(rsh + row);
            tt_act[abase + row] = g / (1.0 + exp(-g)) * u;
        }
        workgroupBarrier();
    }
}

@group(0) @binding(0) var<storage, read>       td_d   : array<u32>;
@group(0) @binding(1) var<storage, read>       td_act : array<f32>;
@group(0) @binding(2) var<storage, read>       td_win : array<u32>;
@group(0) @binding(3) var<storage, read_write> td_y   : array<f32>;
@group(0) @binding(4) var<uniform>             td_p   : TtP;
@group(0) @binding(5) var<storage, read>       td_w   : array<f32>;
@group(0) @binding(6) var<storage, read>       td_slot: array<u32>;
fn td_h(hi: u32) -> f32 {
    let v = unpack2x16float(td_d[hi >> 1u]);
    if ((hi & 1u) == 0u) { return v.x; }
    return v.y;
}
@compute @workgroup_size(256)
fn q4t_tube_dn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    let t = gid.y;
    let hid = td_p.hidden;
    if (d >= hid) { return; }
    let k = td_p.top_k;
    let h = td_p.h;
    let hw = h >> 2u;
    var acc = 0.0;
    for (var j = 0u; j < td_p.cols; j = j + 1u) {
        var e = 0u;
        var wj = 0.0;
        if (td_p.sh == 0u) {
            let rec = t * td_p.cs + 2u * k + 2u * j;
            e = td_win[rec];
            if (e >= td_p.experts) { continue; }
            wj = bitcast<f32>(td_win[rec + 1u]);
        } else {
            wj = td_w[t * td_p.is + td_p.wi];
        }
        let slot = td_slot[e];
        if (slot == 0xFFFFFFFFu) { continue; }
        let seg = slot * td_p.dseg;
        let rsh = seg * 2u + ((hid * h) >> 1u);
        let csh = rsh + hid;
        let abase = (t * td_p.cols + j) * h;
        var sum = 0.0;
        for (var w = 0u; w < hw; w = w + 1u) {
            let q = tt_i8x4(td_d[seg + d * hw + w]);
            let cs = vec4<f32>(td_h(csh + 4u * w), td_h(csh + 4u * w + 1u), td_h(csh + 4u * w + 2u), td_h(csh + 4u * w + 3u));
            let a = vec4<f32>(td_act[abase + 4u * w], td_act[abase + 4u * w + 1u], td_act[abase + 4u * w + 2u], td_act[abase + 4u * w + 3u]);
            sum = sum + dot(q * cs, a);
        }
        acc = acc + wj * sum * td_h(rsh + d);
    }
    let yo = t * td_p.os + d;
    td_y[yo] = td_y[yo] + acc;
}
"#;

/// Subgroup twins of qwen4 kernels (32-lane subgroups laid out over the
/// local index). Every twin keeps its original's arithmetic exactly: the
/// same lane walk and add order, and the same pairing in a 64-lane tree
/// (level 32 through workgroup memory, levels 16..1 as xor shuffles, which
/// pair lane i with lane i + s for every lane i < s, as the tree does). Its
/// own module (built on top of `QWEN4_WGSL` + `QWEN4T_WGSL` for the
/// bindings and helpers): the subgroup builtins need the device feature.
pub(crate) const QWEN4_SG_WGSL: &str = r#"
var<workgroup> g8_t: array<vec4<f32>, 64>;
fn g8_tail(v: vec4<f32>) -> vec4<f32> {
    var r = v;
    r = r + subgroupShuffleXor(r, 16u);
    r = r + subgroupShuffleXor(r, 8u);
    r = r + subgroupShuffleXor(r, 4u);
    r = r + subgroupShuffleXor(r, 2u);
    r = r + subgroupShuffleXor(r, 1u);
    return r;
}
// q4_gu_q2tp4 on 128-lane groups: two independent 64-lane halves, each
// four rows of the same (slot, token) with q4_gu_q2tp4's lane walk, add
// order and tree (finished in registers), so a group carries eight rows
// and the dispatch half the groups — one resident wave on cards where the
// 64-lane groups needed one and a third. Grid (inter / 8, slots, tokens).
var<workgroup> g8_h: array<vec4<f32>, 128>;
@compute @workgroup_size(128)
fn q4_gu_q2tp4w(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lidw: u32) {
    let half = lidw >> 6u;
    let lid = lidw & 63u;
    let row0 = (wid.x * 2u + half) * 4u;
    let slot = wid.y;
    let batch = wid.z;
    let bslot = batch * gq_p.slots + slot;
    let flat = gq_sel[bslot];
    let seg = flat / gq_p.segment_slots;
    let local = flat - seg * gq_p.segment_slots;
    let gpr = gq_p.gpr;
    let rows = gq_p.inter;
    let base16 = local * gq_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let par0 = base16 + rows * gpr * 4u;
    let cod0 = (par0 + rows * 2u) * 2u;
    var gl: array<vec2<f32>, 4>;
    var ul: array<vec2<f32>, 4>;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let par16 = par0 + (row0 + k) * 2u;
        gl[k] = unpack2x16float(gq_g16(seg, par16) | (gq_g16(seg, par16 + 1u) << 16u));
        ul[k] = unpack2x16float(gq_u16(seg, par16) | (gq_u16(seg, par16 + 1u) << 16u));
    }
    var ag = vec4<f32>(0.0);
    var au = vec4<f32>(0.0);
    let xb4 = batch * gpr * 8u;
    for (var g = lid; g < gpr; g = g + 64u) {
        let xo = xb4 + g * 8u;
        let x0 = gq_x[xo];
        let x1 = gq_x[xo + 1u];
        let x2 = gq_x[xo + 2u];
        let x3 = gq_x[xo + 3u];
        let x4 = gq_x[xo + 4u];
        let x5 = gq_x[xo + 5u];
        let x6 = gq_x[xo + 6u];
        let x7 = gq_x[xo + 7u];
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let row = row0 + k;
            let cod8 = cod0 + row * cst;
            var cg = gq_g8(seg, cod8 + cb);
            var cu = gq_u8(seg, cod8 + cb);
            if (shf > 3u) {
                cg = cg | (gq_g8(seg, cod8 + cb + 1u) << 8u);
                cu = cu | (gq_u8(seg, cod8 + cb + 1u) << 8u);
            }
            let sg = gq_scale((cg >> shf) & 31u, gl[k]);
            let su = gq_scale((cu >> shf) & 31u, ul[k]);
            let w32 = (base16 + row * gpr * 4u + g * 4u) >> 1u;
            let dg = gq_dot16(gq_g32(seg, w32), x0, x1, x2, x3) + gq_dot16(gq_g32(seg, w32 + 1u), x4, x5, x6, x7);
            let du = gq_dot16(gq_u32(seg, w32), x0, x1, x2, x3) + gq_dot16(gq_u32(seg, w32 + 1u), x4, x5, x6, x7);
            ag[k] = ag[k] + sg * dg;
            au[k] = au[k] + su * du;
        }
    }
    let hb = half * 64u;
    if (lid >= 32u) {
        g8_h[hb + lid - 32u] = ag;
        g8_h[hb + lid] = au;
    }
    workgroupBarrier();
    if (lid < 32u) {
        let ta = g8_tail(ag + g8_h[hb + lid]);
        let tu = g8_tail(au + g8_h[hb + 32u + lid]);
        if (lid == 0u) {
            for (var k = 0u; k < 4u; k = k + 1u) {
                let row = row0 + k;
                if (row < rows) {
                    let gate = ta[k];
                    let up = tu[k];
                    gq_act[bslot * gq_p.inter + row] = (gate / (1.0 + exp(-gate))) * up;
                }
            }
        }
    }
}

// q4_dn_q4tp4 with the tree finished in registers.
@compute @workgroup_size(64)
fn q4_dn_q4tp4s(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let batch = wid.y;
    let gpr = gv_p.gpr;
    let rows = gv_p.hidden;
    let cst = (gpr * 5u + 7u) / 8u;
    let total = gv_p.slots * gpr;
    var acc = vec4<f32>(0.0);
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i - slot * gpr;
        let bslot = batch * gv_p.slots + slot;
        let flat = gv_sel[bslot];
        let seg = flat / gv_p.segment_slots;
        let local = flat - seg * gv_p.segment_slots;
        let base16 = local * gv_p.mat16;
        let par0 = base16 + rows * gpr * 8u;
        let cod0 = (par0 + rows * 2u) * 2u;
        let xo = (bslot * gpr + g) * 8u;
        let a0 = gv_act[xo];
        let a1 = gv_act[xo + 1u];
        let a2 = gv_act[xo + 2u];
        let a3 = gv_act[xo + 3u];
        let a4 = gv_act[xo + 4u];
        let a5 = gv_act[xo + 5u];
        let a6 = gv_act[xo + 6u];
        let a7 = gv_act[xo + 7u];
        let wt = gv_wt[bslot];
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let row = row0 + k;
            let par16 = par0 + row * 2u;
            let pl = unpack2x16float(gv_16(seg, par16) | (gv_16(seg, par16 + 1u) << 16u));
            let cod8 = cod0 + row * cst;
            var cv = gv_8(seg, cod8 + cb);
            if (shf > 3u) { cv = cv | (gv_8(seg, cod8 + cb + 1u) << 8u); }
            let scale = exp2(pl.x + f32((cv >> shf) & 31u) * pl.y);
            let t16 = base16 + (row * gpr + g) * 8u;
            var d = 0.0;
            d = d + gv_dot8(gv_16(seg, t16) | (gv_16(seg, t16 + 1u) << 16u), a0, a1);
            d = d + gv_dot8(gv_16(seg, t16 + 2u) | (gv_16(seg, t16 + 3u) << 16u), a2, a3);
            d = d + gv_dot8(gv_16(seg, t16 + 4u) | (gv_16(seg, t16 + 5u) << 16u), a4, a5);
            d = d + gv_dot8(gv_16(seg, t16 + 6u) | (gv_16(seg, t16 + 7u) << 16u), a6, a7);
            acc[k] = acc[k] + wt * scale * d;
        }
    }
    if (lid >= 32u) {
        g8_t[lid - 32u] = acc;
    }
    workgroupBarrier();
    if (lid < 32u) {
        let ta = g8_tail(acc + g8_t[lid]);
        if (lid == 0u) {
            for (var k = 0u; k < 4u; k = k + 1u) {
                let row = row0 + k;
                if (row < rows) {
                    gv_y[batch * gv_p.hidden + row] = ta[k];
                }
            }
        }
    }
}

// q4_dn_q4tp4s with the nibble words read as vec4s.
@compute @workgroup_size(64)
fn q4_dn_q4tp4sv(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * 4u;
    let batch = wid.y;
    let gpr = gv_p.gpr;
    let rows = gv_p.hidden;
    let cst = (gpr * 5u + 7u) / 8u;
    let total = gv_p.slots * gpr;
    var acc = vec4<f32>(0.0);
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i - slot * gpr;
        let bslot = batch * gv_p.slots + slot;
        let flat = gv_sel[bslot];
        let seg = flat / gv_p.segment_slots;
        let local = flat - seg * gv_p.segment_slots;
        let base16 = local * gv_p.mat16;
        let par0 = base16 + rows * gpr * 8u;
        let cod0 = (par0 + rows * 2u) * 2u;
        let xo = (bslot * gpr + g) * 8u;
        let a0 = gv_act[xo];
        let a1 = gv_act[xo + 1u];
        let a2 = gv_act[xo + 2u];
        let a3 = gv_act[xo + 3u];
        let a4 = gv_act[xo + 4u];
        let a5 = gv_act[xo + 5u];
        let a6 = gv_act[xo + 6u];
        let a7 = gv_act[xo + 7u];
        let wt = gv_wt[bslot];
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let row = row0 + k;
            let par16 = par0 + row * 2u;
            let pl = unpack2x16float(gv4_16(seg, par16) | (gv4_16(seg, par16 + 1u) << 16u));
            let cod8 = cod0 + row * cst;
            var cv = gv4_8(seg, cod8 + cb);
            if (shf > 3u) { cv = cv | (gv4_8(seg, cod8 + cb + 1u) << 8u); }
            let scale = exp2(pl.x + f32((cv >> shf) & 31u) * pl.y);
            let t16 = base16 + (row * gpr + g) * 8u;
            let q = gv4_w[seg].w4[t16 >> 3u];
            var d = 0.0;
            d = d + gv_dot8(q.x, a0, a1);
            d = d + gv_dot8(q.y, a2, a3);
            d = d + gv_dot8(q.z, a4, a5);
            d = d + gv_dot8(q.w, a6, a7);
            acc[k] = acc[k] + wt * scale * d;
        }
    }
    if (lid >= 32u) {
        g8_t[lid - 32u] = acc;
    }
    workgroupBarrier();
    if (lid < 32u) {
        let ta = g8_tail(acc + g8_t[lid]);
        if (lid == 0u) {
            for (var k = 0u; k < 4u; k = k + 1u) {
                let row = row0 + k;
                if (row < rows) {
                    gv_y[batch * gv_p.hidden + row] = ta[k];
                }
            }
        }
    }
}

// q4_dn_q4tp4sv on 128 lanes (up to two (slot, group) items a lane for
// eleven slots of twenty groups); four subgroup sums meet in workgroup memory.
var<workgroup> d128_t: array<vec4<f32>, 4>;
@compute @workgroup_size(128)
fn q4_dn_q4tp4sv128(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_index) lid: u32,
                @builtin(subgroup_invocation_id) sgi: u32) {
    let row0 = wid.x * 4u;
    let batch = wid.y;
    let gpr = gv_p.gpr;
    let rows = gv_p.hidden;
    let cst = (gpr * 5u + 7u) / 8u;
    let total = gv_p.slots * gpr;
    var acc = vec4<f32>(0.0);
    for (var i = lid; i < total; i = i + 128u) {
        let slot = i / gpr;
        let g = i - slot * gpr;
        let bslot = batch * gv_p.slots + slot;
        let flat = gv_sel[bslot];
        let seg = flat / gv_p.segment_slots;
        let local = flat - seg * gv_p.segment_slots;
        let base16 = local * gv_p.mat16;
        let par0 = base16 + rows * gpr * 8u;
        let cod0 = (par0 + rows * 2u) * 2u;
        let xo = (bslot * gpr + g) * 8u;
        let a0 = gv_act[xo];
        let a1 = gv_act[xo + 1u];
        let a2 = gv_act[xo + 2u];
        let a3 = gv_act[xo + 3u];
        let a4 = gv_act[xo + 4u];
        let a5 = gv_act[xo + 5u];
        let a6 = gv_act[xo + 6u];
        let a7 = gv_act[xo + 7u];
        let wt = gv_wt[bslot];
        let bit = g * 5u;
        let cb = bit >> 3u;
        let shf = bit & 7u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let row = row0 + k;
            let par16 = par0 + row * 2u;
            let pl = unpack2x16float(gv4_16(seg, par16) | (gv4_16(seg, par16 + 1u) << 16u));
            let cod8 = cod0 + row * cst;
            var cv = gv4_8(seg, cod8 + cb);
            if (shf > 3u) { cv = cv | (gv4_8(seg, cod8 + cb + 1u) << 8u); }
            let scale = exp2(pl.x + f32((cv >> shf) & 31u) * pl.y);
            let t16 = base16 + (row * gpr + g) * 8u;
            let q = gv4_w[seg].w4[t16 >> 3u];
            var d = 0.0;
            d = d + gv_dot8(q.x, a0, a1);
            d = d + gv_dot8(q.y, a2, a3);
            d = d + gv_dot8(q.z, a4, a5);
            d = d + gv_dot8(q.w, a6, a7);
            acc[k] = acc[k] + wt * scale * d;
        }
    }
    let r = g8_tail(acc);
    if (sgi == 0u) { d128_t[lid >> 5u] = r; }
    workgroupBarrier();
    if (lid == 0u) {
        let ta = d128_t[0] + d128_t[1] + d128_t[2] + d128_t[3];
        for (var k = 0u; k < 4u; k = k + 1u) {
            let row = row0 + k;
            if (row < rows) {
                gv_y[batch * gv_p.hidden + row] = ta[k];
            }
        }
    }
}

// ── q4t_route with the ranking as k rounds of a workgroup argmax instead of
// every expert counting its rank against all n (one SM walking n² compares),
// and the softmax tail on workgroup memory instead of a single lane's chain
// of dependent global loads and stores. A round picks the largest live
// score, ties to the lowest index: exactly the expert whose rank in
// q4t_route is the round number. The tail repeats q4t_route's arithmetic
// in its order (max, exp(s − max), the sum in rank order, s·(scale/sum));
// a cold winner's resident weight stays 0, as there. n <= 512, one expert a
// lane; 32-lane subgroups laid out over the local index. ──
var<workgroup> r2_v: array<f32, 16>;
var<workgroup> r2_i: array<u32, 16>;
var<workgroup> r2_m: array<u32, 64>;
var<workgroup> r2_s: array<f32, 64>;
var<workgroup> r2_e: array<f32, 64>;
var<workgroup> r2_inv: f32;
fn r2_better(v2: f32, i2: u32, v: f32, i: u32) -> bool {
    return v2 > v || (v2 == v && i2 < i);
}
@compute @workgroup_size(512)
fn q4t_route2(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let n = tr_p.n;
    let k = tr_p.top_k;
    let s0 = t * tr_p.ss;
    let f0 = t * tr_p.fs;
    let i0 = t * tr_p.is;
    let c0 = t * tr_p.cs;
    let pin_shared = (tr_p.flags & 8u) != 0u;
    let subset = (tr_p.flags & 16u) != 0u;
    let qwen = (tr_p.flags & 32u) != 0u;
    let shared_gated = (tr_p.flags & 64u) != 0u;
    let masked = subset && (tr_p.flags & 128u) != 0u;
    let shared_slot = tr_p.flags >> 8u;
    if (pin_shared && lid == 0u) {
        tr_idx[i0 + k] = shared_slot;
        if (shared_gated) {
            tr_w[i0 + k] = bitcast<f32>(tr_forced[f0 + k]);
        } else {
            tr_w[i0 + k] = 1.0;
        }
    }
    let neg_inf = bitcast<f32>(0xFF800000u);
    var alive = lid < n;
    if (alive && masked) { alive = tr_map[lid] != 0xFFFFFFFFu; }
    var si = neg_inf;
    if (alive) {
        let v = tr_s[s0 + lid];
        si = v;
        if (!qwen) {
            var sp = v;
            if (v <= 20.0) { sp = log(1.0 + exp(v)); }
            si = sqrt(sp);
        }
    }
    let m = lid;
    for (var rank = 0u; rank < k; rank = rank + 1u) {
        var bv = select(neg_inf, si, alive);
        var bi = select(0xFFFFFFFFu, m, alive);
        for (var sft = 16u; sft > 0u; sft = sft >> 1u) {
            let ov = subgroupShuffleXor(bv, sft);
            let oi = subgroupShuffleXor(bi, sft);
            if (r2_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if ((lid & 31u) == 0u) {
            r2_v[lid >> 5u] = bv;
            r2_i[lid >> 5u] = bi;
        }
        workgroupBarrier();
        var wv = r2_v[0];
        var wi = r2_i[0];
        for (var j = 1u; j < 16u; j = j + 1u) {
            if (r2_better(r2_v[j], r2_i[j], wv, wi)) {
                wv = r2_v[j];
                wi = r2_i[j];
            }
        }
        workgroupBarrier();
        if (alive && wi == m) {
            alive = false;
            r2_m[rank] = m;
            r2_s[rank] = si;
        }
    }
    workgroupBarrier();
    // q4t_route's tail, over the k winners in rank order (n >= k: all used)
    if (lid == 0u) {
        tr_cnt[t * 4u] = k;
        var qmx = -3.0e38;
        if (qwen) {
            for (var j = 0u; j < k; j = j + 1u) { qmx = max(qmx, r2_s[j]); }
        }
        var sum = 0.0;
        for (var j = 0u; j < k; j = j + 1u) {
            var e = r2_s[j];
            if (qwen) { e = exp(r2_s[j] - qmx); }
            r2_e[j] = e;
            sum = sum + e;
        }
        var inv = 1.0;
        if (sum > 0.0) { inv = tr_p.scale / sum; }
        r2_inv = inv;
    }
    workgroupBarrier();
    if (lid < k) {
        let j = lid;
        let mj = r2_m[j];
        // q4t_route scales only when the sum was positive; inv is then 1
        // and e·1 is e
        let scaled = r2_e[j] * r2_inv;
        tr_cold[c0 + 2u * k + 2u * j] = mj;
        tr_cold[c0 + 2u * k + 2u * j + 1u] = bitcast<u32>(scaled);
        tr_cold[c0 + 2u * j] = 0xFFFFFFFFu;
        tr_cold[c0 + 2u * j + 1u] = 0u;
        if (subset) {
            let slot = tr_map[mj];
            if (slot == 0xFFFFFFFFu) {
                tr_idx[i0 + j] = 0u;
                tr_w[i0 + j] = 0.0;
                tr_cold[c0 + 2u * j] = mj;
                tr_cold[c0 + 2u * j + 1u] = bitcast<u32>(scaled);
            } else {
                tr_idx[i0 + j] = slot;
                tr_w[i0 + j] = scaled;
            }
        } else {
            tr_idx[i0 + j] = mj;
            tr_w[i0 + j] = scaled;
        }
    }
}

// ── q8_2f matvec, one token: `q8_2f_matvec4` (four rows a 256-lane group, a
// row on 64 lanes, the column walk unrolled four deep) with its tree's
// levels 16..1 as xor shuffles inside the row's lower subgroup, so a row
// block pays two barriers instead of seven. Lane walk, add order and the
// pairing of the tree are q8_2f_matvec4's: the same bits. ──
var<workgroup> q8s_t: array<f32, 128>;
@compute @workgroup_size(256)
fn q4_q82_sg(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>,
             @builtin(local_invocation_index) lid: u32) {
    let rows = tq_p.rows;
    let ngrp = tq_p.ngrp;
    let qbytes = rows * tq_p.cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    let sub = lid >> 6u;
    let l = lid & 63u;
    let blocks = (rows + 3u) / 4u;
    var wb = wid.x;
    loop {
        if (wb >= blocks) { break; }
        let row = wb * 4u + sub;
        var acc = 0.0;
        if (row < rows) {
            let roww = row * ngrp;
            var i = l;
            loop {
                if (i + 192u >= ngrp) { break; }
                let w0 = tq_w[roww + i];
                let w1 = tq_w[roww + i + 64u];
                let w2 = tq_w[roww + i + 128u];
                let w3 = tq_w[roww + i + 192u];
                let x0 = tq_x[i];
                let x1 = tq_x[i + 64u];
                let x2 = tq_x[i + 128u];
                let x3 = tq_x[i + 192u];
                let s0 = tq_f16x4(cs0h + i * 4u);
                let s1 = tq_f16x4(cs0h + (i + 64u) * 4u);
                let s2 = tq_f16x4(cs0h + (i + 128u) * 4u);
                let s3 = tq_f16x4(cs0h + (i + 192u) * 4u);
                acc = acc + dot(tq_i8x4(w0), x0 * s0);
                acc = acc + dot(tq_i8x4(w1), x1 * s1);
                acc = acc + dot(tq_i8x4(w2), x2 * s2);
                acc = acc + dot(tq_i8x4(w3), x3 * s3);
                i = i + 256u;
            }
            loop {
                if (i >= ngrp) { break; }
                acc = acc + dot(tq_i8x4(tq_w[roww + i]), tq_x[i] * tq_f16x4(cs0h + i * 4u));
                i = i + 64u;
            }
        }
        if (l >= 32u) {
            q8s_t[sub * 32u + l - 32u] = acc;
        }
        workgroupBarrier();
        if (l < 32u) {
            var r = acc + q8s_t[sub * 32u + l];
            r = r + subgroupShuffleXor(r, 16u);
            r = r + subgroupShuffleXor(r, 8u);
            r = r + subgroupShuffleXor(r, 4u);
            r = r + subgroupShuffleXor(r, 2u);
            r = r + subgroupShuffleXor(r, 1u);
            if (l == 0u && row < rows) {
                let rw = unpack2x16float(tq_w[rs0 + (row >> 1u)]);
                var sc = rw.x;
                if ((row & 1u) == 1u) { sc = rw.y; }
                tq_y[row] = r * sc;
            }
        }
        workgroupBarrier();
        wb = wb + nwg.x;
    }
}

// q4_q82_sg for long rows (the out projections, cols >= 4096): one row a
// 256-lane workgroup, so four times the lanes stream a row's words; the
// eight subgroup sums meet in workgroup memory. Same bindings and uniform.
var<workgroup> q8s1_t: array<f32, 8>;
@compute @workgroup_size(256)
fn q4_q82_sg1(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) lid: u32,
              @builtin(subgroup_invocation_id) sgi: u32) {
    let rows = tq_p.rows;
    let ngrp = tq_p.ngrp;
    let qbytes = rows * tq_p.cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    var row = wid.x;
    loop {
        if (row >= rows) { break; }
        let roww = row * ngrp;
        var acc = 0.0;
        var i = lid;
        loop {
            if (i + 256u >= ngrp) { break; }
            let w0 = tq_w[roww + i];
            let w1 = tq_w[roww + i + 256u];
            let x0 = tq_x[i];
            let x1 = tq_x[i + 256u];
            let s0 = tq_f16x4(cs0h + i * 4u);
            let s1 = tq_f16x4(cs0h + (i + 256u) * 4u);
            acc = acc + dot(tq_i8x4(w0), x0 * s0);
            acc = acc + dot(tq_i8x4(w1), x1 * s1);
            i = i + 512u;
        }
        loop {
            if (i >= ngrp) { break; }
            acc = acc + dot(tq_i8x4(tq_w[roww + i]), tq_x[i] * tq_f16x4(cs0h + i * 4u));
            i = i + 256u;
        }
        let r = subgroupAdd(acc);
        if (sgi == 0u) { q8s1_t[lid >> 5u] = r; }
        workgroupBarrier();
        if (lid == 0u) {
            var t = 0.0;
            for (var k = 0u; k < 8u; k = k + 1u) { t = t + q8s1_t[k]; }
            let rw = unpack2x16float(tq_w[rs0 + (row >> 1u)]);
            var sc = rw.x;
            if ((row & 1u) == 1u) { sc = rw.y; }
            tq_y[row] = t * sc;
        }
        workgroupBarrier();
        row = row + nwg.x;
    }
}
"#;

/// HC v3: one hyper-connection mix in two subgroup kernels (`hc3_down`:
/// group RMS norm folded into the down projection and the injection gate,
/// `hc3_upfold`: up-projection and sigmoid fold). Every 16-byte weight word
/// is fetched once per frame and multiplied into all `nt` token rows; token
/// rows past `nt` are never loaded. With `flags` bit 1 the pending block of
/// the previous sub-layer enters the state on the fly (h + 2σ(g/hc)·blk, the
/// `q4t_inject` arithmetic), and the up-fold writes the new hyper value back
/// in place: the separate inject dispatch goes.
///
/// Its own module, built only on devices with `Features::SUBGROUP` (and no
/// `enable subgroups;`: naga 30 rejects the directive, the device feature is
/// what admits the builtins). Host conditions (`encode_hc_v3`): hc == 4,
/// hidden % 8 == 0, low % 8 == 0, nt <= 8, subgroups of >= 32 lanes laid out
/// linearly over the local index (`hc3_subgroups_ok`).
pub(crate) const HC3_WGSL: &str = r#"
fn h2_dot8(w: vec4<u32>, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)), a)
         + dot(vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)), b);
}

// Pipeline-overridable chunk counts: 0 = take them from the uniform; the
// engine specializes them (ceil(cols/2048) and ceil(low/64): 5 and 5 for
// Qwen3.8-Flash-Next) so every loop below has a constant trip count.
override H3_KD: u32 = 0u;
override H3_KU: u32 = 0u;
struct H3dP { cols: u32, rows_a: u32, rows_b: u32, hidden: u32,
              eps: f32, inv_hc: f32, nt: u32, hs4: u32,
              yas: u32, ybs: u32, flags: u32, bs4: u32,
              gs: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       h3d_a   : array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       h3d_b   : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       h3d_h   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       h3d_w   : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> h3d_ya  : array<f32>;
@group(0) @binding(5) var<storage, read_write> h3d_yb  : array<f32>;
@group(0) @binding(6) var<storage, read_write> h3d_inv : array<f32>;
@group(0) @binding(7) var<uniform>             h3d_p   : H3dP;
@group(0) @binding(8) var<storage, read>       h3d_blk : array<vec4<f32>>;
@group(0) @binding(9) var<storage, read>       h3d_g   : array<f32>;
// Every slot read below is written earlier in the same dispatch (the
// pipelines run without workgroup zero-init): h3d_ss[t*32 + sg] by each
// subgroup's lane 0 for t < nt, sg < 256/ssz; h3d_sc[t*4 + s] (and the
// gates at 32 + t*4 + s when flagged) by lanes < 4·nt; h3d_red[sg*4 + 0..3]
// by each subgroup's lane 0.
var<workgroup> h3d_ss  : array<vec4<f32>, 256>;  // [t][sg] per-stream sums
var<workgroup> h3d_sc  : array<f32, 64>;          // [t][s]: inv, then [32 + t*4+s]: gate
var<workgroup> h3d_red : array<vec4<f32>, 128>;  // [sg][lo0, hi0, lo1, hi1]

// token t's chunk q (8 floats), the pending inject applied when flagged;
// `bo` is the chunk's column within its stream in vec4 units (q·2 − s·hidden/4)
fn h3d_row(t: u32, q: u32, s: u32, bo: u32) -> array<vec4<f32>, 2> {
    var a = h3d_h[t * h3d_p.hs4 + 2u * q];
    var b = h3d_h[t * h3d_p.hs4 + 2u * q + 1u];
    if ((h3d_p.flags & 1u) != 0u) {
        let gs = h3d_sc[32u + t * 4u + s];
        let o = t * h3d_p.bs4 + bo;
        a = a + gs * h3d_blk[o];
        b = b + gs * h3d_blk[o + 1u];
    }
    return array<vec4<f32>, 2>(a, b);
}

// the two row dots of token t's normalized chunk
fn h3d_dots(t: u32, q: u32, s: u32, bo: u32, wa: vec4<u32>, wb: vec4<u32>, ga: vec4<f32>, gb: vec4<f32>) -> vec2<f32> {
    let v = h3d_row(t, q, s, bo);
    let iv = h3d_sc[t * 4u + s];
    let na = v[0] * iv * ga;
    let nb = v[1] * iv * gb;
    return vec2<f32>(h2_dot8(wa, na, nb), h2_dot8(wb, na, nb));
}

// grid ((rows_a + rows_b) / 2): workgroup w owns rows 2w and 2w+1 of [A; B]
@compute @workgroup_size(256)
fn hc3_down(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32,
            @builtin(subgroup_invocation_id) sgi: u32,
            @builtin(subgroup_size) ssz: u32) {
    let nch = h3d_p.cols >> 3u;
    let hidden = h3d_p.hidden;
    let nt = h3d_p.nt;
    let nsg = 256u / ssz;
    let sg = lid / ssz;
    let total = h3d_p.rows_a + h3d_p.rows_b;
    let g0 = wid.x * 2u;
    let g1 = g0 + 1u;
    let kd = select((nch + 255u) >> 8u, H3_KD, H3_KD != 0u);
    if (lid < 4u * nt && (h3d_p.flags & 1u) != 0u) {
        let t = lid >> 2u;
        h3d_sc[32u + lid] = 2.0 / (1.0 + exp(-h3d_g[t * h3d_p.gs + (lid & 3u)] * h3d_p.inv_hc));
    }
    workgroupBarrier();
    // pass 1: per-token, per-stream sums of squares
    for (var t = 0u; t < nt; t = t + 1u) {
        var ss = vec4<f32>(0.0);
        for (var k = 0u; k < kd; k = k + 1u) {
            let q = lid + 256u * k;
            if (q >= nch) { break; }
            let s = (q * 8u) / hidden;
            let v = h3d_row(t, q, s, 2u * q - s * (hidden >> 2u));
            let e = dot(v[0], v[0]) + dot(v[1], v[1]);
            ss = ss + select(vec4<f32>(0.0), vec4<f32>(e), vec4<u32>(0u, 1u, 2u, 3u) == vec4<u32>(s));
        }
        ss = subgroupAdd(ss);
        if (sgi == 0u) { h3d_ss[t * 32u + sg] = ss; }
    }
    workgroupBarrier();
    if (lid < 4u * nt) {
        let t = lid >> 2u;
        var tot = 0.0;
        for (var i = 0u; i < nsg; i = i + 1u) { tot = tot + h3d_ss[t * 32u + i][lid & 3u]; }
        let inv = inverseSqrt(tot / f32(hidden) + h3d_p.eps);
        h3d_sc[lid] = inv;
        if (wid.x == 0u) { h3d_inv[lid] = inv; }
    }
    workgroupBarrier();
    // pass 2: each weight word once, every token row against it
    var lo0 = vec4<f32>(0.0);
    var hi0 = vec4<f32>(0.0);
    var lo1 = vec4<f32>(0.0);
    var hi1 = vec4<f32>(0.0);
    let ra = g0 < h3d_p.rows_a;
    let rb = g1 < h3d_p.rows_a;
    for (var k = 0u; k < kd; k = k + 1u) {
        let q = lid + 256u * k;
        if (q >= nch) { break; }
        let s = (q * 8u) / hidden;
        let bo = 2u * q - s * (hidden >> 2u);
        var wa = vec4<u32>(0u);
        var wb = vec4<u32>(0u);
        if (ra) { wa = h3d_a[g0 * nch + q]; } else if (g0 < total) { wa = h3d_b[(g0 - h3d_p.rows_a) * nch + q]; }
        if (rb) { wb = h3d_a[g1 * nch + q]; } else if (g1 < total) { wb = h3d_b[(g1 - h3d_p.rows_a) * nch + q]; }
        let ga = vec4<f32>(1.0) + h3d_w[2u * q];
        let gb = vec4<f32>(1.0) + h3d_w[2u * q + 1u];
        var d = h3d_dots(0u, q, s, bo, wa, wb, ga, gb);
        lo0.x = lo0.x + d.x; lo1.x = lo1.x + d.y;
        if (nt > 1u) { d = h3d_dots(1u, q, s, bo, wa, wb, ga, gb); lo0.y = lo0.y + d.x; lo1.y = lo1.y + d.y; }
        if (nt > 2u) { d = h3d_dots(2u, q, s, bo, wa, wb, ga, gb); lo0.z = lo0.z + d.x; lo1.z = lo1.z + d.y; }
        if (nt > 3u) { d = h3d_dots(3u, q, s, bo, wa, wb, ga, gb); lo0.w = lo0.w + d.x; lo1.w = lo1.w + d.y; }
        if (nt > 4u) {
            d = h3d_dots(4u, q, s, bo, wa, wb, ga, gb); hi0.x = hi0.x + d.x; hi1.x = hi1.x + d.y;
            if (nt > 5u) { d = h3d_dots(5u, q, s, bo, wa, wb, ga, gb); hi0.y = hi0.y + d.x; hi1.y = hi1.y + d.y; }
            if (nt > 6u) { d = h3d_dots(6u, q, s, bo, wa, wb, ga, gb); hi0.z = hi0.z + d.x; hi1.z = hi1.z + d.y; }
            if (nt > 7u) { d = h3d_dots(7u, q, s, bo, wa, wb, ga, gb); hi0.w = hi0.w + d.x; hi1.w = hi1.w + d.y; }
        }
    }
    lo0 = subgroupAdd(lo0);
    lo1 = subgroupAdd(lo1);
    if (nt > 4u) {
        hi0 = subgroupAdd(hi0);
        hi1 = subgroupAdd(hi1);
    }
    if (sgi == 0u) {
        h3d_red[sg * 4u] = lo0;
        h3d_red[sg * 4u + 1u] = hi0;
        h3d_red[sg * 4u + 2u] = lo1;
        h3d_red[sg * 4u + 3u] = hi1;
    }
    workgroupBarrier();
    if (lid < 2u * nt) {
        let r = lid / nt;
        let t = lid % nt;
        var v = 0.0;
        let slot = r * 2u + (t >> 2u);
        for (var i = 0u; i < nsg; i = i + 1u) { v = v + h3d_red[i * 4u + slot][t & 3u]; }
        let g = g0 + r;
        if (g < h3d_p.rows_a) {
            let z = v * h3d_p.inv_hc;
            h3d_ya[t * h3d_p.yas + g] = z / (1.0 + exp(-z));
        } else if (g < total) {
            h3d_yb[t * h3d_p.ybs + g - h3d_p.rows_a] = v;
        }
    }
}

// `hc3_down` in one pass over the hyper state (up to four tokens): every
// lane keeps, per token, the four streams' sums of squares and the two rows'
// per-stream dots of the (1 + w)-weighted state, and the stream inverses are
// applied after the reduction (z = Σ_s inv_s · dot_s), so the state is read
// once and the first pass with its barrier is gone. Same bindings, uniform,
// grid and outputs.
var<workgroup> h3e_red : array<vec4<f32>, 96>;  // [sg][t][ss, d0, d1]
@compute @workgroup_size(256)
fn hc3_down1(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32,
             @builtin(subgroup_invocation_id) sgi: u32,
             @builtin(subgroup_size) ssz: u32) {
    let nch = h3d_p.cols >> 3u;
    let hidden = h3d_p.hidden;
    let nt = h3d_p.nt;
    let nsg = 256u / ssz;
    let sg = lid / ssz;
    let total = h3d_p.rows_a + h3d_p.rows_b;
    let g0 = wid.x * 2u;
    let g1 = g0 + 1u;
    let kd = select((nch + 255u) >> 8u, H3_KD, H3_KD != 0u);
    if (lid < 4u * nt && (h3d_p.flags & 1u) != 0u) {
        let t = lid >> 2u;
        h3d_sc[32u + lid] = 2.0 / (1.0 + exp(-h3d_g[t * h3d_p.gs + (lid & 3u)] * h3d_p.inv_hc));
    }
    workgroupBarrier();
    var a = array<vec4<f32>, 12>();
    let ra = g0 < h3d_p.rows_a;
    let rb = g1 < h3d_p.rows_a;
    for (var k = 0u; k < kd; k = k + 1u) {
        let q = lid + 256u * k;
        if (q >= nch) { break; }
        let s = (q * 8u) / hidden;
        let bo = 2u * q - s * (hidden >> 2u);
        let m = select(vec4<f32>(0.0), vec4<f32>(1.0), vec4<u32>(0u, 1u, 2u, 3u) == vec4<u32>(s));
        var wa = vec4<u32>(0u);
        var wb = vec4<u32>(0u);
        if (ra) { wa = h3d_a[g0 * nch + q]; } else if (g0 < total) { wa = h3d_b[(g0 - h3d_p.rows_a) * nch + q]; }
        if (rb) { wb = h3d_a[g1 * nch + q]; } else if (g1 < total) { wb = h3d_b[(g1 - h3d_p.rows_a) * nch + q]; }
        let ga = vec4<f32>(1.0) + h3d_w[2u * q];
        let gb = vec4<f32>(1.0) + h3d_w[2u * q + 1u];
        for (var t = 0u; t < 4u; t = t + 1u) {
            if (t >= nt) { break; }
            let v = h3d_row(t, q, s, bo);
            let e = dot(v[0], v[0]) + dot(v[1], v[1]);
            let na = v[0] * ga;
            let nb = v[1] * gb;
            a[t * 3u] = a[t * 3u] + m * e;
            a[t * 3u + 1u] = a[t * 3u + 1u] + m * h2_dot8(wa, na, nb);
            a[t * 3u + 2u] = a[t * 3u + 2u] + m * h2_dot8(wb, na, nb);
        }
    }
    for (var i = 0u; i < 12u; i = i + 1u) {
        let v = subgroupAdd(a[i]);
        if (sgi == 0u) { h3e_red[sg * 12u + i] = v; }
    }
    workgroupBarrier();
    if (lid < 2u * nt) {
        let r = lid / nt;
        let t = lid % nt;
        var ss = vec4<f32>(0.0);
        var dd = vec4<f32>(0.0);
        for (var i = 0u; i < nsg; i = i + 1u) {
            ss = ss + h3e_red[i * 12u + t * 3u];
            dd = dd + h3e_red[i * 12u + t * 3u + 1u + r];
        }
        let inv = inverseSqrt(ss / f32(hidden) + vec4<f32>(h3d_p.eps));
        if (wid.x == 0u && r == 0u) {
            h3d_inv[t * 4u] = inv.x;
            h3d_inv[t * 4u + 1u] = inv.y;
            h3d_inv[t * 4u + 2u] = inv.z;
            h3d_inv[t * 4u + 3u] = inv.w;
        }
        let v = dot(inv, dd);
        let g = g0 + r;
        if (g < h3d_p.rows_a) {
            let z = v * h3d_p.inv_hc;
            h3d_ya[t * h3d_p.yas + g] = z / (1.0 + exp(-z));
        } else if (g < total) {
            h3d_yb[t * h3d_p.ybs + g - h3d_p.rows_a] = v;
        }
    }
}

// `hc3_down` split by stream: workgroup (w, s) owns rows 2w, 2w+1 over stream
// s's columns alone, with that stream's norm (an RMS of its own columns, so no
// other workgroup is waited on), and writes the stream's share of the rows:
// ya[t][s][g] = inv_s · Σ_c W·(1+w)·h (the A rows, before the SiLU) and the
// injection rows' shares in yb[t][s][g − rows_a]. `hc3_upfold4` adds the four
// shares (and applies the SiLU / writes the final injection logits). Four
// times the workgroups of `hc3_down`, one pass over the state, up to 8 tokens.
var<workgroup> h3s_red : array<vec4<f32>, 48>;  // [sg][ss lo, ss hi, d0 lo, d0 hi, d1 lo, d1 hi]
@compute @workgroup_size(256)
fn hc3_down4(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32,
             @builtin(subgroup_invocation_id) sgi: u32,
             @builtin(subgroup_size) ssz: u32) {
    let hidden = h3d_p.hidden;
    let nch = h3d_p.cols >> 3u;
    let cps = hidden >> 3u;
    let s = wid.y;
    let nt = h3d_p.nt;
    let nsg = 256u / ssz;
    let sg = lid / ssz;
    let total = h3d_p.rows_a + h3d_p.rows_b;
    let g0 = wid.x * 2u;
    let g1 = g0 + 1u;
    if (lid < 4u * nt && (h3d_p.flags & 1u) != 0u) {
        let t = lid >> 2u;
        h3d_sc[32u + lid] = 2.0 / (1.0 + exp(-h3d_g[t * h3d_p.gs + (lid & 3u)] * h3d_p.inv_hc));
    }
    workgroupBarrier();
    var ssl = vec4<f32>(0.0);
    var ssh = vec4<f32>(0.0);
    var d0l = vec4<f32>(0.0);
    var d0h = vec4<f32>(0.0);
    var d1l = vec4<f32>(0.0);
    var d1h = vec4<f32>(0.0);
    let ra = g0 < h3d_p.rows_a;
    let rb = g1 < h3d_p.rows_a;
    let kd = (cps + 255u) >> 8u;
    for (var k = 0u; k < kd; k = k + 1u) {
        let qs = lid + 256u * k;
        if (qs >= cps) { break; }
        let q = s * cps + qs;
        let bo = 2u * qs;
        var wa = vec4<u32>(0u);
        var wb = vec4<u32>(0u);
        if (ra) { wa = h3d_a[g0 * nch + q]; } else if (g0 < total) { wa = h3d_b[(g0 - h3d_p.rows_a) * nch + q]; }
        if (rb) { wb = h3d_a[g1 * nch + q]; } else if (g1 < total) { wb = h3d_b[(g1 - h3d_p.rows_a) * nch + q]; }
        let ga = vec4<f32>(1.0) + h3d_w[2u * q];
        let gb = vec4<f32>(1.0) + h3d_w[2u * q + 1u];
        for (var t = 0u; t < 8u; t = t + 1u) {
            if (t >= nt) { break; }
            let v = h3d_row(t, q, s, bo);
            let e = dot(v[0], v[0]) + dot(v[1], v[1]);
            let na = v[0] * ga;
            let nb = v[1] * gb;
            let x = h2_dot8(wa, na, nb);
            let y = h2_dot8(wb, na, nb);
            let m = select(vec4<f32>(0.0), vec4<f32>(1.0), vec4<u32>(0u, 1u, 2u, 3u) == vec4<u32>(t & 3u));
            if (t < 4u) {
                ssl = ssl + m * e;
                d0l = d0l + m * x;
                d1l = d1l + m * y;
            } else {
                ssh = ssh + m * e;
                d0h = d0h + m * x;
                d1h = d1h + m * y;
            }
        }
    }
    ssl = subgroupAdd(ssl);
    d0l = subgroupAdd(d0l);
    d1l = subgroupAdd(d1l);
    if (nt > 4u) {
        ssh = subgroupAdd(ssh);
        d0h = subgroupAdd(d0h);
        d1h = subgroupAdd(d1h);
    }
    if (sgi == 0u) {
        h3s_red[sg * 6u] = ssl;
        h3s_red[sg * 6u + 1u] = ssh;
        h3s_red[sg * 6u + 2u] = d0l;
        h3s_red[sg * 6u + 3u] = d0h;
        h3s_red[sg * 6u + 4u] = d1l;
        h3s_red[sg * 6u + 5u] = d1h;
    }
    workgroupBarrier();
    if (lid < 2u * nt) {
        let r = lid / nt;
        let t = lid % nt;
        let hi = t >> 2u;
        var ss = 0.0;
        var dd = 0.0;
        for (var i = 0u; i < nsg; i = i + 1u) {
            ss = ss + h3s_red[i * 6u + hi][t & 3u];
            dd = dd + h3s_red[i * 6u + 2u + 2u * r + hi][t & 3u];
        }
        let inv = inverseSqrt(ss / f32(hidden) + h3d_p.eps);
        if (wid.x == 0u && r == 0u) { h3d_inv[t * 4u + s] = inv; }
        let g = g0 + r;
        if (g < h3d_p.rows_a) {
            h3d_ya[t * h3d_p.yas + s * h3d_p.rows_a + g] = inv * dd;
        } else if (g < total) {
            h3d_yb[t * h3d_p.ybs + s * h3d_p.rows_b + g - h3d_p.rows_a] = inv * dd;
        }
    }
}

struct H3uP { hidden: u32, low: u32, nt: u32, inv_hc: f32, ls4: u32, hs: u32, os: u32, flags: u32,
              bs: u32, gs: u32, _a: u32, _b: u32, _c: u32, _d: u32, _e: u32, _f: u32 };
@group(0) @binding(0) var<storage, read>       h3u_w   : array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       h3u_low : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> h3u_h   : array<f32>;
@group(0) @binding(3) var<storage, read>       h3u_nw  : array<f32>;
@group(0) @binding(4) var<storage, read>       h3u_inv : array<f32>;
@group(0) @binding(5) var<storage, read_write> h3u_out : array<f32>;
@group(0) @binding(6) var<uniform>             h3u_p   : H3uP;
@group(0) @binding(7) var<storage, read>       h3u_blk : array<f32>;
@group(0) @binding(8) var<storage, read>       h3u_g   : array<f32>;

fn h3u_dot(wq: vec4<u32>, t: u32, c: u32) -> f32 {
    let lb = t * h3u_p.ls4 + 2u * c;
    return h2_dot8(wq, h3u_low[lb], h3u_low[lb + 1u]);
}

// σ-fold value of token t at (s, d), the hyper value updated in place when flagged
fn h3u_fold(t: u32, s: u32, o: u32, dd: u32, m: f32, nw1: f32, write_h: bool) -> f32 {
    var hv = h3u_h[t * h3u_p.hs + o];
    if ((h3u_p.flags & 1u) != 0u) {
        let gs = 2.0 / (1.0 + exp(-h3u_g[t * h3u_p.gs + s] * h3u_p.inv_hc));
        hv = hv + gs * h3u_blk[t * h3u_p.bs + dd];
        if (write_h) { h3u_h[t * h3u_p.hs + o] = hv; }
    }
    let n = hv * h3u_inv[t * 4u + s] * nw1;
    return n * h3u_p.inv_hc / (1.0 + exp(-m));
}

// grid (hidden / 8): a 32-lane group owns one output column d, lanes
// 8s..8s+7 the row s·hidden+d of `up`; xor 1/2/4 sums the octet, xor 8/16
// the four streams (subgroups of >= 32 lanes, linear over the local index)
@compute @workgroup_size(256)
fn hc3_upfold(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let hidden = h3u_p.hidden;
    let nt = h3u_p.nt;
    let lane = lid & 31u;
    let s = lane >> 3u;
    let j = lane & 7u;
    let d = wid.x * 8u + (lid >> 5u);
    let dd = min(d, hidden - 1u);
    let lc = h3u_p.low >> 3u;
    let o = s * hidden + dd;
    let base = o * lc;
    var lo = vec4<f32>(0.0);
    var hi = vec4<f32>(0.0);
    let ku = select((lc + 7u) >> 3u, H3_KU, H3_KU != 0u);
    for (var k = 0u; k < ku; k = k + 1u) {
        let c = j + 8u * k;
        if (c >= lc) { break; }
        let wq = h3u_w[base + c];
        lo.x = lo.x + h3u_dot(wq, 0u, c);
        if (nt > 1u) { lo.y = lo.y + h3u_dot(wq, 1u, c); }
        if (nt > 2u) { lo.z = lo.z + h3u_dot(wq, 2u, c); }
        if (nt > 3u) { lo.w = lo.w + h3u_dot(wq, 3u, c); }
        if (nt > 4u) {
            hi.x = hi.x + h3u_dot(wq, 4u, c);
            if (nt > 5u) { hi.y = hi.y + h3u_dot(wq, 5u, c); }
            if (nt > 6u) { hi.z = hi.z + h3u_dot(wq, 6u, c); }
            if (nt > 7u) { hi.w = hi.w + h3u_dot(wq, 7u, c); }
        }
    }
    lo = lo + subgroupShuffleXor(lo, 1u);
    lo = lo + subgroupShuffleXor(lo, 2u);
    lo = lo + subgroupShuffleXor(lo, 4u);
    if (nt > 4u) {
        hi = hi + subgroupShuffleXor(hi, 1u);
        hi = hi + subgroupShuffleXor(hi, 2u);
        hi = hi + subgroupShuffleXor(hi, 4u);
    }
    let nw1 = 1.0 + h3u_nw[o];
    let wh = j == 0u && d < hidden;
    // one lane per (stream, column) folds: it alone reads (and, with the
    // fused inject, rewrites) h[t,s,d]; the other seven contribute zero
    var vlo = vec4<f32>(0.0);
    var vhi = vec4<f32>(0.0);
    if (j == 0u) {
        vlo.x = h3u_fold(0u, s, o, dd, lo.x, nw1, wh);
        if (nt > 1u) { vlo.y = h3u_fold(1u, s, o, dd, lo.y, nw1, wh); }
        if (nt > 2u) { vlo.z = h3u_fold(2u, s, o, dd, lo.z, nw1, wh); }
        if (nt > 3u) { vlo.w = h3u_fold(3u, s, o, dd, lo.w, nw1, wh); }
        if (nt > 4u) {
            vhi.x = h3u_fold(4u, s, o, dd, hi.x, nw1, wh);
            if (nt > 5u) { vhi.y = h3u_fold(5u, s, o, dd, hi.y, nw1, wh); }
            if (nt > 6u) { vhi.z = h3u_fold(6u, s, o, dd, hi.z, nw1, wh); }
            if (nt > 7u) { vhi.w = h3u_fold(7u, s, o, dd, hi.w, nw1, wh); }
        }
    }
    vlo = vlo + subgroupShuffleXor(vlo, 8u);
    vlo = vlo + subgroupShuffleXor(vlo, 16u);
    if (nt > 4u) {
        vhi = vhi + subgroupShuffleXor(vhi, 8u);
        vhi = vhi + subgroupShuffleXor(vhi, 16u);
    }
    if (lane == 0u && d < hidden) {
        for (var t = 0u; t < nt; t = t + 1u) {
            var v = vlo[t & 3u];
            if (t >= 4u) { v = vhi[t & 3u]; }
            h3u_out[t * h3u_p.os + d] = v;
        }
    }
}

// hc3_upfold4's low row: the four streams' shares of chunk c added, the SiLU
// of hc3_down applied (`ls4` is then the share array's token stride in vec4s)
@group(0) @binding(9)  var<storage, read>       h3u_ybp : array<f32>;
@group(0) @binding(10) var<storage, read_write> h3u_yb  : array<f32>;
fn h3u_dot4(wq: vec4<u32>, t: u32, c: u32) -> f32 {
    let lq = h3u_p.low >> 2u;
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    for (var s4 = 0u; s4 < 4u; s4 = s4 + 1u) {
        let lb = t * h3u_p.ls4 + s4 * lq + 2u * c;
        a = a + h3u_low[lb];
        b = b + h3u_low[lb + 1u];
    }
    let za = a * h3u_p.inv_hc;
    let zb = b * h3u_p.inv_hc;
    return h2_dot8(wq, za / (vec4<f32>(1.0) + exp(-za)), zb / (vec4<f32>(1.0) + exp(-zb)));
}

@compute @workgroup_size(256)
fn hc3_upfold4(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let hidden = h3u_p.hidden;
    let nt = h3u_p.nt;
    // the injection logits: the four streams' shares added (workgroup 0)
    if (wid.x == 0u && lid < h3u_p._a * nt) {
        let t = lid / h3u_p._a;
        let r = lid % h3u_p._a;
        var v = 0.0;
        for (var s4 = 0u; s4 < 4u; s4 = s4 + 1u) { v = v + h3u_ybp[t * h3u_p._b + s4 * h3u_p._a + r]; }
        h3u_yb[t * h3u_p._c + r] = v;
    }
    let lane = lid & 31u;
    let s = lane >> 3u;
    let j = lane & 7u;
    let d = wid.x * 8u + (lid >> 5u);
    let dd = min(d, hidden - 1u);
    let lc = h3u_p.low >> 3u;
    let o = s * hidden + dd;
    let base = o * lc;
    var lo = vec4<f32>(0.0);
    var hi = vec4<f32>(0.0);
    let ku = select((lc + 7u) >> 3u, H3_KU, H3_KU != 0u);
    for (var k = 0u; k < ku; k = k + 1u) {
        let c = j + 8u * k;
        if (c >= lc) { break; }
        let wq = h3u_w[base + c];
        lo.x = lo.x + h3u_dot4(wq, 0u, c);
        if (nt > 1u) { lo.y = lo.y + h3u_dot4(wq, 1u, c); }
        if (nt > 2u) { lo.z = lo.z + h3u_dot4(wq, 2u, c); }
        if (nt > 3u) { lo.w = lo.w + h3u_dot4(wq, 3u, c); }
        if (nt > 4u) {
            hi.x = hi.x + h3u_dot4(wq, 4u, c);
            if (nt > 5u) { hi.y = hi.y + h3u_dot4(wq, 5u, c); }
            if (nt > 6u) { hi.z = hi.z + h3u_dot4(wq, 6u, c); }
            if (nt > 7u) { hi.w = hi.w + h3u_dot4(wq, 7u, c); }
        }
    }
    lo = lo + subgroupShuffleXor(lo, 1u);
    lo = lo + subgroupShuffleXor(lo, 2u);
    lo = lo + subgroupShuffleXor(lo, 4u);
    if (nt > 4u) {
        hi = hi + subgroupShuffleXor(hi, 1u);
        hi = hi + subgroupShuffleXor(hi, 2u);
        hi = hi + subgroupShuffleXor(hi, 4u);
    }
    let nw1 = 1.0 + h3u_nw[o];
    let wh = j == 0u && d < hidden;
    // one lane per (stream, column) folds: it alone reads (and, with the
    // fused inject, rewrites) h[t,s,d]; the other seven contribute zero
    var vlo = vec4<f32>(0.0);
    var vhi = vec4<f32>(0.0);
    if (j == 0u) {
        vlo.x = h3u_fold(0u, s, o, dd, lo.x, nw1, wh);
        if (nt > 1u) { vlo.y = h3u_fold(1u, s, o, dd, lo.y, nw1, wh); }
        if (nt > 2u) { vlo.z = h3u_fold(2u, s, o, dd, lo.z, nw1, wh); }
        if (nt > 3u) { vlo.w = h3u_fold(3u, s, o, dd, lo.w, nw1, wh); }
        if (nt > 4u) {
            vhi.x = h3u_fold(4u, s, o, dd, hi.x, nw1, wh);
            if (nt > 5u) { vhi.y = h3u_fold(5u, s, o, dd, hi.y, nw1, wh); }
            if (nt > 6u) { vhi.z = h3u_fold(6u, s, o, dd, hi.z, nw1, wh); }
            if (nt > 7u) { vhi.w = h3u_fold(7u, s, o, dd, hi.w, nw1, wh); }
        }
    }
    vlo = vlo + subgroupShuffleXor(vlo, 8u);
    vlo = vlo + subgroupShuffleXor(vlo, 16u);
    if (nt > 4u) {
        vhi = vhi + subgroupShuffleXor(vhi, 8u);
        vhi = vhi + subgroupShuffleXor(vhi, 16u);
    }
    if (lane == 0u && d < hidden) {
        for (var t = 0u; t < nt; t = t + 1u) {
            var v = vlo[t & 3u];
            if (t >= 4u) { v = vhi[t & 3u]; }
            h3u_out[t * h3u_p.os + d] = v;
        }
    }
}
"#;

/// Expert GEMMs of prompt frames on the matrix units (cooperative 16×16×16 f16
/// → f32): per (64-row tile, expert segment of the sorted winners) a workgroup
/// of four subgroups dequantizes its q4tp rows one 32-column group at a time
/// into workgroup memory (each lane one row: four nibble words, one rung code,
/// one exp2) and multiplies them into sixteen of the segment's tokens at once.
/// Same bindings and segment table (`q4g_segs`: arena slot, first sorted
/// winner, count; the count at [4096]) as the scalar tiles. Activations are
/// rounded to f16 (the down input scaled by 2⁻⁴ against overflow, the scale
/// undone in the f32 epilogue). Rules of naga's coop uniformity analysis kept:
/// no early return, no `&&` and no call result on a path to a coop op, every
/// workgroup variable a whole number of 16-byte units.
pub(crate) const QWEN4_MM_WGSL: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;
enable wgpu_binding_array;
diagnostic(off, derivative_uniformity);

struct GqBank { words: array<u32> };
struct GqP { gpr: u32, inter: u32, slots: u32, mat16: u32, lim: f32, segment_slots: u32, p0: u32, p1: u32 };
struct GvP { gpr: u32, hidden: u32, slots: u32, mat16: u32, segment_slots: u32, p0: u32, p1: u32, p2: u32 };

const MLD: u32 = 40u;

// ── gate/up: the gu0t layout (banks, x, act, sorted order, segments) ──
@group(0) @binding(0) var<storage, read>       mg_gw  : binding_array<GqBank, 8>;
@group(0) @binding(1) var<storage, read>       mg_uw  : binding_array<GqBank, 8>;
@group(0) @binding(2) var<storage, read>       mg_x   : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> mg_act : array<f32>;
@group(0) @binding(5) var<storage, read>       mg_ord : array<u32>;
@group(0) @binding(6) var<storage, read>       mg_seg : array<vec4<u32>>;
@group(1) @binding(0) var<uniform>             mg_p   : GqP;
var<workgroup> mg_sa: array<f16, 640>;    // [16 tokens][40]
var<workgroup> mg_sg: array<f16, 2560>;   // [64 rows][40] gate
var<workgroup> mg_su: array<f16, 2560>;   // [64 rows][40] up
var<workgroup> mg_c:  array<f32, 2048>;   // [subgroup][gate | up][16 tokens][16 rows]
var<workgroup> mg_hdr: vec4<u32>;

fn mg_gword(seg: u32, o: u32) -> u32 { return mg_gw[seg].words[o]; }
fn mg_uword(seg: u32, o: u32) -> u32 { return mg_uw[seg].words[o]; }

@compute @workgroup_size(128)
fn q4g_mm_gu(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32,
             @builtin(subgroup_id) sgid: u32) {
    if (lid == 0u) {
        var h = vec4<u32>(0u);
        if (wid.y < mg_seg[4096].x) { h = mg_seg[wid.y]; }
        mg_hdr = h;
    }
    let hdr = workgroupUniformLoad(&mg_hdr);
    let cnt = hdr.z;
    let flat = hdr.x;
    let seg = flat / mg_p.segment_slots;
    let local = flat - seg * mg_p.segment_slots;
    let gpr = mg_p.gpr;
    let rows = mg_p.inter;
    let base16 = local * mg_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let row0 = wid.x * 64u;
    // dequant role: lanes 0..63 a gate row, 64..127 the same up row
    let isup = lid >= 64u;
    let dr = lid & 63u;
    let r = min(row0 + dr, rows - 1u);
    let par16 = base16 + rows * gpr * 8u + r * 2u;
    var plw = 0u;
    if (isup) {
        plw = (mg_uword(seg, par16 >> 1u));
    } else {
        plw = (mg_gword(seg, par16 >> 1u));
    }
    let pl = unpack2x16float(plw);
    let cod8 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + r * cst;
    let w0 = (base16 >> 1u) + r * gpr * 4u;
    // A role: token (lid / 8) of the tile, columns (lid % 8)·4 .. +3 of the group
    let at = lid >> 3u;
    let ak = (lid & 7u) * 4u;
    var zc: coop_mat16x16<f32, C>;
    var cg: coop_mat16x16<f32, C>;
    var cu: coop_mat16x16<f32, C>;
    for (var tb = 0u; tb < cnt; tb = tb + 16u) {
        cg = zc;
        cu = zc;
        let jt = tb + at;
        let live = jt < cnt;
        var xrow = 0u;
        if (live) { xrow = (mg_ord[hdr.y + jt] / mg_p.slots) * gpr * 8u; }
        // registers of slice kt (loaded one slice ahead, under the MMAs)
        var av = vec4<f32>(0.0);
        var q = vec4<u32>(0u);
        var cw = 0u;
        if (live) { av = mg_x[xrow + (ak >> 2u)]; }
        if (isup) {
            q = vec4<u32>(mg_uword(seg, w0), mg_uword(seg, w0 + 1u), mg_uword(seg, w0 + 2u), mg_uword(seg, w0 + 3u));
            cw = mg_uword(seg, cod8 >> 2u) >> ((cod8 & 3u) * 8u);
        } else {
            q = vec4<u32>(mg_gword(seg, w0), mg_gword(seg, w0 + 1u), mg_gword(seg, w0 + 2u), mg_gword(seg, w0 + 3u));
            cw = mg_gword(seg, cod8 >> 2u) >> ((cod8 & 3u) * 8u);
        }
        for (var kt = 0u; kt < gpr; kt = kt + 1u) {
            let da = at * MLD + ak;
            mg_sa[da] = f16(av.x);
            mg_sa[da + 1u] = f16(av.y);
            mg_sa[da + 2u] = f16(av.z);
            mg_sa[da + 3u] = f16(av.w);
            // the rung code: five bits at bit kt·5 of the row's code bytes; `cw`
            // holds the byte at kt·5/8 and the one after it from the prefetch
            let shf = (kt * 5u) & 7u;
            let sc = exp2(pl.x + f32((cw >> shf) & 31u) * pl.y);
            let db = dr * MLD;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let wj = q[j];
                for (var b = 0u; b < 8u; b = b + 1u) {
                    let v = f16((f32((wj >> (4u * b)) & 0xFu) - 8.0) * sc);
                    if (isup) { mg_su[db + j * 8u + b] = v; } else { mg_sg[db + j * 8u + b] = v; }
                }
            }
            workgroupBarrier();
            // the next slice's registers while the matrix units run
            let kn = min(kt + 1u, gpr - 1u);
            av = vec4<f32>(0.0);
            if (live) { av = mg_x[xrow + kn * 8u + (ak >> 2u)]; }
            let wn = w0 + kn * 4u;
            let cbn = cod8 + ((kn * 5u) >> 3u);
            if (isup) {
                q = vec4<u32>(mg_uword(seg, wn), mg_uword(seg, wn + 1u), mg_uword(seg, wn + 2u), mg_uword(seg, wn + 3u));
                cw = (mg_uword(seg, cbn >> 2u) >> ((cbn & 3u) * 8u)) & 0xFFu;
                if (((kn * 5u) & 7u) > 3u) { cw = cw | (((mg_uword(seg, (cbn + 1u) >> 2u) >> (((cbn + 1u) & 3u) * 8u)) & 0xFFu) << 8u); }
            } else {
                q = vec4<u32>(mg_gword(seg, wn), mg_gword(seg, wn + 1u), mg_gword(seg, wn + 2u), mg_gword(seg, wn + 3u));
                cw = (mg_gword(seg, cbn >> 2u) >> ((cbn & 3u) * 8u)) & 0xFFu;
                if (((kn * 5u) & 7u) > 3u) { cw = cw | (((mg_gword(seg, (cbn + 1u) >> 2u) >> (((cbn + 1u) & 3u) * 8u)) & 0xFFu) << 8u); }
            }
            for (var kk = 0u; kk < 2u; kk = kk + 1u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(&mg_sa[kk * 16u], MLD);
                let bg = coopLoad<coop_mat16x16<f16, B>>(&mg_sg[sgid * 16u * MLD + kk * 16u], MLD);
                let bu = coopLoad<coop_mat16x16<f16, B>>(&mg_su[sgid * 16u * MLD + kk * 16u], MLD);
                cg = coopMultiplyAdd(a, bg, cg);
                cu = coopMultiplyAdd(a, bu, cu);
            }
            workgroupBarrier();
        }
        let og = sgid * 512u;
        let ou = og + 256u;
        coopStoreT(cg, &mg_c[og], 16u);
        coopStoreT(cu, &mg_c[ou], 16u);
        workgroupBarrier();
        for (var e = lid; e < 1024u; e = e + 128u) {
            let tt = e >> 6u;
            let rr = e & 63u;
            let o = (rr >> 4u) * 512u + tt * 16u + (rr & 15u);
            let gv = mg_c[o];
            let uv = mg_c[o + 256u];
            let jj = tb + tt;
            if (jj < cnt) {
                if (row0 + rr < rows) {
                    let bslot = mg_ord[hdr.y + jj];
                    mg_act[bslot * rows + row0 + rr] = (gv / (1.0 + exp(-gv))) * uv;
                }
            }
        }
        workgroupBarrier();
    }
}

// ── q4g_mm_gu with 64-column K slices (two groups a lane a slice): half the
// slices and barriers; needs an even group count (the host checks) ──
const MLD2: u32 = 72u;
var<workgroup> mg2_sa: array<f16, 1152>;   // [16 tokens][72]
var<workgroup> mg2_sg: array<f16, 4608>;   // [64 rows][72]
var<workgroup> mg2_su: array<f16, 4608>;
fn mg_code(isup: bool, seg: u32, cod8: u32, g: u32) -> u32 {
    let cb = cod8 + ((g * 5u) >> 3u);
    let shf = (g * 5u) & 7u;
    var cw = 0u;
    if (isup) {
        cw = (mg_uword(seg, cb >> 2u) >> ((cb & 3u) * 8u)) & 0xFFu;
        if (shf > 3u) { cw = cw | (((mg_uword(seg, (cb + 1u) >> 2u) >> (((cb + 1u) & 3u) * 8u)) & 0xFFu) << 8u); }
    } else {
        cw = (mg_gword(seg, cb >> 2u) >> ((cb & 3u) * 8u)) & 0xFFu;
        if (shf > 3u) { cw = cw | (((mg_gword(seg, (cb + 1u) >> 2u) >> (((cb + 1u) & 3u) * 8u)) & 0xFFu) << 8u); }
    }
    return (cw >> shf) & 31u;
}
fn mg_words(isup: bool, seg: u32, w: u32) -> vec4<u32> {
    if (isup) {
        return vec4<u32>(mg_uword(seg, w), mg_uword(seg, w + 1u), mg_uword(seg, w + 2u), mg_uword(seg, w + 3u));
    }
    return vec4<u32>(mg_gword(seg, w), mg_gword(seg, w + 1u), mg_gword(seg, w + 2u), mg_gword(seg, w + 3u));
}
@compute @workgroup_size(128)
fn q4g_mm_gu64(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32,
               @builtin(subgroup_id) sgid: u32) {
    if (lid == 0u) {
        var h = vec4<u32>(0u);
        if (wid.y < mg_seg[4096].x) { h = mg_seg[wid.y]; }
        mg_hdr = h;
    }
    let hdr = workgroupUniformLoad(&mg_hdr);
    let cnt = hdr.z;
    let flat = hdr.x;
    let seg = flat / mg_p.segment_slots;
    let local = flat - seg * mg_p.segment_slots;
    let gpr = mg_p.gpr;
    let npair = gpr >> 1u;
    let rows = mg_p.inter;
    let base16 = local * mg_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let row0 = wid.x * 64u;
    let isup = lid >= 64u;
    let dr = lid & 63u;
    let r = min(row0 + dr, rows - 1u);
    let par16 = base16 + rows * gpr * 8u + r * 2u;
    var plw = 0u;
    if (isup) { plw = mg_uword(seg, par16 >> 1u); } else { plw = mg_gword(seg, par16 >> 1u); }
    let pl = unpack2x16float(plw);
    let cod8 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + r * cst;
    let w0 = (base16 >> 1u) + r * gpr * 4u;
    let at = lid >> 3u;
    let ak = (lid & 7u) * 8u;
    var zc: coop_mat16x16<f32, C>;
    var cg: coop_mat16x16<f32, C>;
    var cu: coop_mat16x16<f32, C>;
    for (var tb = 0u; tb < cnt; tb = tb + 16u) {
        cg = zc;
        cu = zc;
        let jt = tb + at;
        let live = jt < cnt;
        var xrow = 0u;
        if (live) { xrow = (mg_ord[hdr.y + jt] / mg_p.slots) * gpr * 8u; }
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        if (live) { a0 = mg_x[xrow + (ak >> 2u)]; a1 = mg_x[xrow + (ak >> 2u) + 1u]; }
        var q0 = mg_words(isup, seg, w0);
        var q1 = mg_words(isup, seg, w0 + 4u);
        var c0 = mg_code(isup, seg, cod8, 0u);
        var c1 = mg_code(isup, seg, cod8, 1u);
        for (var kp = 0u; kp < npair; kp = kp + 1u) {
            let da = at * MLD2 + ak;
            mg2_sa[da] = f16(a0.x);
            mg2_sa[da + 1u] = f16(a0.y);
            mg2_sa[da + 2u] = f16(a0.z);
            mg2_sa[da + 3u] = f16(a0.w);
            mg2_sa[da + 4u] = f16(a1.x);
            mg2_sa[da + 5u] = f16(a1.y);
            mg2_sa[da + 6u] = f16(a1.z);
            mg2_sa[da + 7u] = f16(a1.w);
            let s0 = exp2(pl.x + f32(c0) * pl.y);
            let s1 = exp2(pl.x + f32(c1) * pl.y);
            let db = dr * MLD2;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let wa = q0[j];
                let wb = q1[j];
                for (var b = 0u; b < 8u; b = b + 1u) {
                    let va = f16((f32((wa >> (4u * b)) & 0xFu) - 8.0) * s0);
                    let vb = f16((f32((wb >> (4u * b)) & 0xFu) - 8.0) * s1);
                    if (isup) {
                        mg2_su[db + j * 8u + b] = va;
                        mg2_su[db + 32u + j * 8u + b] = vb;
                    } else {
                        mg2_sg[db + j * 8u + b] = va;
                        mg2_sg[db + 32u + j * 8u + b] = vb;
                    }
                }
            }
            workgroupBarrier();
            let kn = min(kp + 1u, npair - 1u);
            a0 = vec4<f32>(0.0);
            a1 = vec4<f32>(0.0);
            if (live) { a0 = mg_x[xrow + kn * 16u + (ak >> 2u)]; a1 = mg_x[xrow + kn * 16u + (ak >> 2u) + 1u]; }
            q0 = mg_words(isup, seg, w0 + kn * 8u);
            q1 = mg_words(isup, seg, w0 + kn * 8u + 4u);
            c0 = mg_code(isup, seg, cod8, 2u * kn);
            c1 = mg_code(isup, seg, cod8, 2u * kn + 1u);
            for (var kk = 0u; kk < 4u; kk = kk + 1u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(&mg2_sa[kk * 16u], MLD2);
                let bg = coopLoad<coop_mat16x16<f16, B>>(&mg2_sg[sgid * 16u * MLD2 + kk * 16u], MLD2);
                let bu = coopLoad<coop_mat16x16<f16, B>>(&mg2_su[sgid * 16u * MLD2 + kk * 16u], MLD2);
                cg = coopMultiplyAdd(a, bg, cg);
                cu = coopMultiplyAdd(a, bu, cu);
            }
            workgroupBarrier();
        }
        let og = sgid * 512u;
        let ou = og + 256u;
        coopStoreT(cg, &mg_c[og], 16u);
        coopStoreT(cu, &mg_c[ou], 16u);
        workgroupBarrier();
        for (var e = lid; e < 1024u; e = e + 128u) {
            let tt = e >> 6u;
            let rr = e & 63u;
            let o = (rr >> 4u) * 512u + tt * 16u + (rr & 15u);
            let gv = mg_c[o];
            let uv = mg_c[o + 256u];
            let jj = tb + tt;
            if (jj < cnt) {
                if (row0 + rr < rows) {
                    let bslot = mg_ord[hdr.y + jj];
                    mg_act[bslot * rows + row0 + rr] = (gv / (1.0 + exp(-gv))) * uv;
                }
            }
        }
        workgroupBarrier();
    }
}

// `q4g_mm_gu64` for q2tp gate/up banks (the q2tp-v2 and hot files): the 2-bit section
// is half the size, a slice's two groups are four words, code 0 is a zero scale.
@compute @workgroup_size(128)
fn q4g_mm_gu64q2(@builtin(workgroup_id) wid: vec3<u32>,
               @builtin(local_invocation_index) lid: u32,
               @builtin(subgroup_id) sgid: u32) {
    if (lid == 0u) {
        var h = vec4<u32>(0u);
        if (wid.y < mg_seg[4096].x) { h = mg_seg[wid.y]; }
        mg_hdr = h;
    }
    let hdr = workgroupUniformLoad(&mg_hdr);
    let cnt = hdr.z;
    let flat = hdr.x;
    let seg = flat / mg_p.segment_slots;
    let local = flat - seg * mg_p.segment_slots;
    let gpr = mg_p.gpr;
    let npair = gpr >> 1u;
    let rows = mg_p.inter;
    let base16 = local * mg_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let row0 = wid.x * 64u;
    let isup = lid >= 64u;
    let dr = lid & 63u;
    let r = min(row0 + dr, rows - 1u);
    let par16 = base16 + rows * gpr * 4u + r * 2u;
    var plw = 0u;
    if (isup) { plw = mg_uword(seg, par16 >> 1u); } else { plw = mg_gword(seg, par16 >> 1u); }
    let pl = unpack2x16float(plw);
    let cod8 = (base16 + rows * gpr * 4u + rows * 2u) * 2u + r * cst;
    let w0 = (base16 >> 1u) + r * gpr * 2u;
    let at = lid >> 3u;
    let ak = (lid & 7u) * 8u;
    var zc: coop_mat16x16<f32, C>;
    var cg: coop_mat16x16<f32, C>;
    var cu: coop_mat16x16<f32, C>;
    for (var tb = 0u; tb < cnt; tb = tb + 16u) {
        cg = zc;
        cu = zc;
        let jt = tb + at;
        let live = jt < cnt;
        var xrow = 0u;
        if (live) { xrow = (mg_ord[hdr.y + jt] / mg_p.slots) * gpr * 8u; }
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        if (live) { a0 = mg_x[xrow + (ak >> 2u)]; a1 = mg_x[xrow + (ak >> 2u) + 1u]; }
        var q0 = mg_words(isup, seg, w0);
        var c0 = mg_code(isup, seg, cod8, 0u);
        var c1 = mg_code(isup, seg, cod8, 1u);
        for (var kp = 0u; kp < npair; kp = kp + 1u) {
            let da = at * MLD2 + ak;
            mg2_sa[da] = f16(a0.x);
            mg2_sa[da + 1u] = f16(a0.y);
            mg2_sa[da + 2u] = f16(a0.z);
            mg2_sa[da + 3u] = f16(a0.w);
            mg2_sa[da + 4u] = f16(a1.x);
            mg2_sa[da + 5u] = f16(a1.y);
            mg2_sa[da + 6u] = f16(a1.z);
            mg2_sa[da + 7u] = f16(a1.w);
            // q2tp rungs: code 0 is a zero scale, code c the rung c - 1
            let s0 = select(exp2(pl.x + f32(max(c0, 1u) - 1u) * pl.y), 0.0, c0 == 0u);
            let s1 = select(exp2(pl.x + f32(max(c1, 1u) - 1u) * pl.y), 0.0, c1 == 0u);
            let db = dr * MLD2;
            // two 2-bit words a group: column j at bit 2·(j % 16) of word j / 16, value − 1.5
            for (var j = 0u; j < 2u; j = j + 1u) {
                let wa = q0[j];
                let wb = q0[2u + j];
                for (var b = 0u; b < 16u; b = b + 1u) {
                    let va = f16((f32((wa >> (2u * b)) & 3u) - 1.5) * s0);
                    let vb = f16((f32((wb >> (2u * b)) & 3u) - 1.5) * s1);
                    if (isup) {
                        mg2_su[db + j * 16u + b] = va;
                        mg2_su[db + 32u + j * 16u + b] = vb;
                    } else {
                        mg2_sg[db + j * 16u + b] = va;
                        mg2_sg[db + 32u + j * 16u + b] = vb;
                    }
                }
            }
            workgroupBarrier();
            let kn = min(kp + 1u, npair - 1u);
            a0 = vec4<f32>(0.0);
            a1 = vec4<f32>(0.0);
            if (live) { a0 = mg_x[xrow + kn * 16u + (ak >> 2u)]; a1 = mg_x[xrow + kn * 16u + (ak >> 2u) + 1u]; }
            q0 = mg_words(isup, seg, w0 + kn * 4u);
            c0 = mg_code(isup, seg, cod8, 2u * kn);
            c1 = mg_code(isup, seg, cod8, 2u * kn + 1u);
            for (var kk = 0u; kk < 4u; kk = kk + 1u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(&mg2_sa[kk * 16u], MLD2);
                let bg = coopLoad<coop_mat16x16<f16, B>>(&mg2_sg[sgid * 16u * MLD2 + kk * 16u], MLD2);
                let bu = coopLoad<coop_mat16x16<f16, B>>(&mg2_su[sgid * 16u * MLD2 + kk * 16u], MLD2);
                cg = coopMultiplyAdd(a, bg, cg);
                cu = coopMultiplyAdd(a, bu, cu);
            }
            workgroupBarrier();
        }
        let og = sgid * 512u;
        let ou = og + 256u;
        coopStoreT(cg, &mg_c[og], 16u);
        coopStoreT(cu, &mg_c[ou], 16u);
        workgroupBarrier();
        for (var e = lid; e < 1024u; e = e + 128u) {
            let tt = e >> 6u;
            let rr = e & 63u;
            let o = (rr >> 4u) * 512u + tt * 16u + (rr & 15u);
            let gv = mg_c[o];
            let uv = mg_c[o + 256u];
            let jj = tb + tt;
            if (jj < cnt) {
                if (row0 + rr < rows) {
                    let bslot = mg_ord[hdr.y + jj];
                    mg_act[bslot * rows + row0 + rr] = (gv / (1.0 + exp(-gv))) * uv;
                }
            }
        }
        workgroupBarrier();
    }
}

// ── `q4t_f16_pair_mm` on the matrix units: the f16 pair (rows_a of A, then
// rows_b of B) times every token's f32 inputs (rounded to f16), 64 rows × 64
// tokens a workgroup (four subgroups of 16 rows, four 16-token accumulators
// each), the same bindings, uniform, split-K partials and epilogue; grid
// (row tiles, token tiles of 64, ks). ──
struct TmmP { cols: u32, rows_a: u32, rows_b: u32, act: u32, inv: f32, yb_off: u32, nt: u32, xs: u32,
              yas: u32, ybs: u32, kc: u32, ks: u32, _c: u32, _d: u32, _e: u32, _f: u32 };
@group(0) @binding(0) var<storage, read>       pc_a  : array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       pc_b  : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       pc_x  : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pc_ya : array<f32>;
@group(0) @binding(4) var<storage, read_write> pc_yb : array<f32>;
@group(0) @binding(5) var<uniform>             pc_p  : TmmP;
@group(0) @binding(6) var<storage, read_write> pc_part : array<f32>;
var<workgroup> pc_sa: array<f16, 2560>;   // [64 tokens][40]
var<workgroup> pc_sb: array<f16, 2560>;   // [64 rows][40]
var<workgroup> pc_c:  array<f32, 4096>;   // [subgroup][token block][16 tokens][16 rows]
@compute @workgroup_size(128)
fn q4t_f16_pair_coop(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(local_invocation_index) lid: u32,
                     @builtin(subgroup_id) sgid: u32) {
    let cols = pc_p.cols;
    let ra = pc_p.rows_a;
    let total = ra + pc_p.rows_b;
    let row0 = wid.x * 64u;
    let t0 = wid.y * 64u;
    let nt = pc_p.nt;
    let kbeg = wid.z * pc_p.kc;
    let kend = min(cols, kbeg + pc_p.kc);
    // B role: row lid / 2, sixteen columns (lid & 1) of the 32-column slice
    let br = lid >> 1u;
    let bh = (lid & 1u) * 16u;
    let row_b = row0 + br;
    // A role: token lid / 2, the same sixteen columns
    let tok_a = t0 + br;
    var zc: coop_mat16x16<f32, C>;
    var c0: coop_mat16x16<f32, C>;
    var c1: coop_mat16x16<f32, C>;
    var c2: coop_mat16x16<f32, C>;
    var c3: coop_mat16x16<f32, C>;
    c0 = zc;
    c1 = zc;
    c2 = zc;
    c3 = zc;
    // this lane's sixteen columns of the slice, loaded one slice ahead: two
    // 16-byte words of weights (8 f16 pairs), four 16-byte words of inputs
    // (the host gives cols % 32 == 0 and an input row stride % 4 == 0)
    let wbase = select((row_b - ra) * cols, row_b * cols, row_b < ra) >> 3u;
    let live_b = row_b < total;
    let live_a = tok_a < nt;
    let xbase = (tok_a * pc_p.xs) >> 2u;
    var wq0 = vec4<u32>(0u);
    var wq1 = vec4<u32>(0u);
    var xa0 = vec4<f32>(0.0);
    var xa1 = vec4<f32>(0.0);
    var xa2 = vec4<f32>(0.0);
    var xa3 = vec4<f32>(0.0);
    if (kbeg < kend) {
        let o = (kbeg + bh) >> 3u;
        if (live_b) {
            if (row_b < ra) { wq0 = pc_a[wbase + o]; wq1 = pc_a[wbase + o + 1u]; }
            else { wq0 = pc_b[wbase + o]; wq1 = pc_b[wbase + o + 1u]; }
        }
        if (live_a) {
            let xo = xbase + ((kbeg + bh) >> 2u);
            xa0 = pc_x[xo];
            xa1 = pc_x[xo + 1u];
            xa2 = pc_x[xo + 2u];
            xa3 = pc_x[xo + 3u];
        }
    }
    for (var k0 = kbeg; k0 < kend; k0 = k0 + 32u) {
        let db = br * MLD + bh;
        let wv = array<u32, 8>(wq0.x, wq0.y, wq0.z, wq0.w, wq1.x, wq1.y, wq1.z, wq1.w);
        for (var j = 0u; j < 8u; j = j + 1u) {
            let w2 = unpack2x16float(wv[j]);
            pc_sb[db + 2u * j] = f16(w2.x);
            pc_sb[db + 2u * j + 1u] = f16(w2.y);
        }
        let xv = array<vec4<f32>, 4>(xa0, xa1, xa2, xa3);
        for (var j = 0u; j < 4u; j = j + 1u) {
            pc_sa[db + 4u * j] = f16(xv[j].x);
            pc_sa[db + 4u * j + 1u] = f16(xv[j].y);
            pc_sa[db + 4u * j + 2u] = f16(xv[j].z);
            pc_sa[db + 4u * j + 3u] = f16(xv[j].w);
        }
        // the next slice (the last one reloads itself: in bounds, unused)
        let kn = min(k0 + 32u, kend - 32u) + bh;
        if (live_b) {
            if (row_b < ra) { wq0 = pc_a[wbase + (kn >> 3u)]; wq1 = pc_a[wbase + (kn >> 3u) + 1u]; }
            else { wq0 = pc_b[wbase + (kn >> 3u)]; wq1 = pc_b[wbase + (kn >> 3u) + 1u]; }
        }
        if (live_a) {
            let xo = xbase + (kn >> 2u);
            xa0 = pc_x[xo];
            xa1 = pc_x[xo + 1u];
            xa2 = pc_x[xo + 2u];
            xa3 = pc_x[xo + 3u];
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 2u; kk = kk + 1u) {
            let b = coopLoad<coop_mat16x16<f16, B>>(&pc_sb[sgid * 16u * MLD + kk * 16u], MLD);
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&pc_sa[kk * 16u], MLD);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&pc_sa[16u * MLD + kk * 16u], MLD);
            let a2 = coopLoadT<coop_mat16x16<f16, A>>(&pc_sa[32u * MLD + kk * 16u], MLD);
            let a3 = coopLoadT<coop_mat16x16<f16, A>>(&pc_sa[48u * MLD + kk * 16u], MLD);
            c0 = coopMultiplyAdd(a0, b, c0);
            c1 = coopMultiplyAdd(a1, b, c1);
            c2 = coopMultiplyAdd(a2, b, c2);
            c3 = coopMultiplyAdd(a3, b, c3);
        }
        workgroupBarrier();
    }
    let o0 = sgid * 1024u;
    let o1 = o0 + 256u;
    let o2 = o0 + 512u;
    let o3 = o0 + 768u;
    coopStoreT(c0, &pc_c[o0], 16u);
    coopStoreT(c1, &pc_c[o1], 16u);
    coopStoreT(c2, &pc_c[o2], 16u);
    coopStoreT(c3, &pc_c[o3], 16u);
    workgroupBarrier();
    // 64 rows x 64 tokens, 32 a lane: lane -> row (lid & 63), tokens of half lid / 64
    let rr = lid & 63u;
    let row = row0 + rr;
    let th = lid >> 6u;
    if (row < total) {
        for (var j = 0u; j < 32u; j = j + 1u) {
            let tt = th * 32u + j;
            let t = t0 + tt;
            if (t < nt) {
                var v = pc_c[(rr >> 4u) * 1024u + (tt >> 4u) * 256u + (tt & 15u) * 16u + (rr & 15u)];
                if (pc_p.ks > 1u) {
                    pc_part[(wid.z * nt + t) * total + row] = v;
                } else if (row >= ra) {
                    if ((pc_p.act & 2u) != 0u) { v = 1.0 / (1.0 + exp(-v)); }
                    pc_yb[t * pc_p.ybs + pc_p.yb_off + row - ra] = v;
                } else {
                    if ((pc_p.act & 1u) != 0u) { let z = v * pc_p.inv; v = z / (1.0 + exp(-z)); }
                    pc_ya[t * pc_p.yas + row] = v;
                }
            }
        }
    }
}

// ── down: the dn0t layout; partial[bslot][row] = weight · row ──
@group(0) @binding(0) var<storage, read>       md_w   : binding_array<GqBank, 8>;
@group(0) @binding(1) var<storage, read>       md_act : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       md_wt  : array<f32>;
@group(0) @binding(4) var<storage, read_write> md_y   : array<f32>;
@group(0) @binding(5) var<storage, read>       md_ord : array<u32>;
@group(0) @binding(6) var<storage, read>       md_seg : array<vec4<u32>>;
@group(1) @binding(0) var<uniform>             md_p   : GvP;
var<workgroup> md_sa: array<f16, 640>;    // [16 tokens][40]
var<workgroup> md_sb: array<f16, 2560>;   // [64 rows][40]
var<workgroup> md_c:  array<f32, 1024>;   // [subgroup][16 tokens][16 rows]
var<workgroup> md_hdr: vec4<u32>;

fn md_word(seg: u32, o: u32) -> u32 { return md_w[seg].words[o]; }

@compute @workgroup_size(128)
fn q4g_mm_dn(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) lid: u32,
             @builtin(subgroup_id) sgid: u32) {
    if (lid == 0u) {
        var h = vec4<u32>(0u);
        if (wid.y < md_seg[4096].x) { h = md_seg[wid.y]; }
        md_hdr = h;
    }
    let hdr = workgroupUniformLoad(&md_hdr);
    let cnt = hdr.z;
    let flat = hdr.x;
    let seg = flat / md_p.segment_slots;
    let local = flat - seg * md_p.segment_slots;
    let gpr = md_p.gpr;              // 32-column groups of the inter axis
    let rows = md_p.hidden;
    let base16 = local * md_p.mat16;
    let cst = (gpr * 5u + 7u) / 8u;
    let row0 = wid.x * 64u;
    // dequant role: two lanes a row, two nibble words each
    let dr = lid >> 1u;
    let hw = lid & 1u;
    let r = min(row0 + dr, rows - 1u);
    let par16 = base16 + rows * gpr * 8u + r * 2u;
    let pl = unpack2x16float(md_word(seg, par16 >> 1u));
    let cod8 = (base16 + rows * gpr * 8u + rows * 2u) * 2u + r * cst;
    let w0 = (base16 >> 1u) + r * gpr * 4u + hw * 2u;
    let at = lid >> 3u;
    let ak = (lid & 7u) * 4u;
    let cols4 = gpr * 8u;
    var zc: coop_mat16x16<f32, C>;
    var cc: coop_mat16x16<f32, C>;
    for (var tb = 0u; tb < cnt; tb = tb + 16u) {
        cc = zc;
        let jt = tb + at;
        let live = jt < cnt;
        var arow = 0u;
        if (live) { arow = md_ord[hdr.y + jt] * cols4; }
        // registers of slice kt (loaded one slice ahead, under the MMAs)
        var av = vec4<f32>(0.0);
        if (live) { av = md_act[arow + (ak >> 2u)] * 0.0625; }
        var cw = md_word(seg, cod8 >> 2u) >> ((cod8 & 3u) * 8u);
        var q0 = md_word(seg, w0);
        var q1 = md_word(seg, w0 + 1u);
        for (var kt = 0u; kt < gpr; kt = kt + 1u) {
            let da = at * MLD + ak;
            md_sa[da] = f16(av.x);
            md_sa[da + 1u] = f16(av.y);
            md_sa[da + 2u] = f16(av.z);
            md_sa[da + 3u] = f16(av.w);
            let shf = (kt * 5u) & 7u;
            let sc = exp2(pl.x + f32((cw >> shf) & 31u) * pl.y);
            let db = dr * MLD + hw * 16u;
            for (var b = 0u; b < 8u; b = b + 1u) {
                md_sb[db + b] = f16((f32((q0 >> (4u * b)) & 0xFu) - 8.0) * sc);
                md_sb[db + 8u + b] = f16((f32((q1 >> (4u * b)) & 0xFu) - 8.0) * sc);
            }
            workgroupBarrier();
            let kn = min(kt + 1u, gpr - 1u);
            av = vec4<f32>(0.0);
            if (live) { av = md_act[arow + kn * 8u + (ak >> 2u)] * 0.0625; }
            let cbn = cod8 + ((kn * 5u) >> 3u);
            cw = (md_word(seg, cbn >> 2u) >> ((cbn & 3u) * 8u)) & 0xFFu;
            if (((kn * 5u) & 7u) > 3u) { cw = cw | (((md_word(seg, (cbn + 1u) >> 2u) >> (((cbn + 1u) & 3u) * 8u)) & 0xFFu) << 8u); }
            let wn = w0 + kn * 4u;
            q0 = md_word(seg, wn);
            q1 = md_word(seg, wn + 1u);
            for (var kk = 0u; kk < 2u; kk = kk + 1u) {
                let a = coopLoadT<coop_mat16x16<f16, A>>(&md_sa[kk * 16u], MLD);
                let b = coopLoad<coop_mat16x16<f16, B>>(&md_sb[sgid * 16u * MLD + kk * 16u], MLD);
                cc = coopMultiplyAdd(a, b, cc);
            }
            workgroupBarrier();
        }
        let oc = sgid * 256u;
        coopStoreT(cc, &md_c[oc], 16u);
        workgroupBarrier();
        for (var e = lid; e < 1024u; e = e + 128u) {
            let tt = e >> 6u;
            let rr = e & 63u;
            let v = md_c[(rr >> 4u) * 256u + tt * 16u + (rr & 15u)];
            let jj = tb + tt;
            if (jj < cnt) {
                if (row0 + rr < rows) {
                    let bslot = md_ord[hdr.y + jj];
                    md_y[bslot * rows + row0 + rr] = md_wt[bslot] * v * 16.0;
                }
            }
        }
        workgroupBarrier();
    }
}
"#;

pub(crate) struct Pipes {
    group_rmsnorm: wgpu::ComputePipeline,
    f16_matvec: wgpu::ComputePipeline,
    ple_gate: wgpu::ComputePipeline,
    ple_conv: wgpu::ComputePipeline,
    block_key: wgpu::ComputePipeline,
    idx_build: wgpu::ComputePipeline,
    qsa_attend: wgpu::ComputePipeline,
    gate: wgpu::ComputePipeline,
    miss: wgpu::ComputePipeline,
    mtp_fuse: wgpu::ComputePipeline,
    t_group_rmsnorm: wgpu::ComputePipeline,
    t_f16_pair: wgpu::ComputePipeline,
    t_hc_upfold: wgpu::ComputePipeline,
    t_q82_matvec: wgpu::ComputePipeline,
    t_inject: wgpu::ComputePipeline,
    t_gdn_norm: wgpu::ComputePipeline,
    t_route: wgpu::ComputePipeline,
    embed_gather_q82: wgpu::ComputePipeline,
    gu_q2tp4: wgpu::ComputePipeline,
    gu_q4tp4: wgpu::ComputePipeline,
    gu_q4tp4v: wgpu::ComputePipeline,
    dn_q4tp4: wgpu::ComputePipeline,
    t_hc_down: wgpu::ComputePipeline,
    t_hc_upfold2: wgpu::ComputePipeline,
    t_rope: wgpu::ComputePipeline,
    t_kv_append: wgpu::ComputePipeline,
    t_rawk: wgpu::ComputePipeline,
    t_ix_scores: wgpu::ComputePipeline,
    t_ix_scores_l: wgpu::ComputePipeline,
    t_topk_r: wgpu::ComputePipeline,
    t_topk: wgpu::ComputePipeline,
    t_idx_build: wgpu::ComputePipeline,
    t_attend: wgpu::ComputePipeline,
    t_attend_sp: wgpu::ComputePipeline,
    t_attend_cmb: wgpu::ComputePipeline,
    t_f16_pair_mm: wgpu::ComputePipeline,
    t_pair_sum: wgpu::ComputePipeline,
    g_gu_q4tp4: wgpu::ComputePipeline,
    g_dn_q4tp4: wgpu::ComputePipeline,
    g_sum: wgpu::ComputePipeline,
    g_sort: wgpu::ComputePipeline,
    g_segs: wgpu::ComputePipeline,
    /// The counting sort of frames past 4096 winners: clear, count, scan, scatter.
    g_cs: [wgpu::ComputePipeline; 4],
    /// `q4g_mm_gu` / `q4g_mm_dn` (`QWEN4_MM_WGSL`): the prompt frames'
    /// expert GEMMs on the matrix units; None without cooperative matrices,
    /// 32-lane subgroups, on rejection, or with `CMF_QWEN_MM=0`.
    mm: Option<(wgpu::ComputePipeline, wgpu::ComputePipeline)>,
    /// `q4g_mm_gu64q2`: the gate/up GEMM for q2tp banks (same layout as `mm.0`).
    mm_gu_q2: Option<wgpu::ComputePipeline>,
    /// `q4t_f16_pair_coop` (`QWEN4_MM_WGSL`): the prompt frames' f16 pair GEMMs
    /// (HC mixes, router, GDN gates) on the matrix units; `CMF_QWEN_PAIR_COOP=0` off.
    pair_coop: Option<wgpu::ComputePipeline>,
    t_hc_fold_t: wgpu::ComputePipeline,
    t_o1_pack: wgpu::ComputePipeline,
    t_o1_gate: wgpu::ComputePipeline,
    t_tube_gu: wgpu::ComputePipeline,
    t_tube_dn: wgpu::ComputePipeline,
    /// HC v3 (`HC3_WGSL`), where the device takes it; None keeps every
    /// hyper-connection mix on the kernels above.
    hc3: Option<Hc3>,
    /// `q4_gu_q2tp4w` (`CMF_QWEN_GU4W`: two four-row halves a 128-lane
    /// group) and `q4_dn_q4tp4s` (`CMF_QWEN_DN4S`: the tree finished in
    /// registers), subgroup twins of the four-row expert pair; None without
    /// 32-lane subgroups, on rejection, or with the switch at 0.
    gu4w: Option<wgpu::ComputePipeline>,
    dn4s: Option<wgpu::ComputePipeline>,
    dn4sv: Option<wgpu::ComputePipeline>,
    dn4sv128: Option<wgpu::ComputePipeline>,
    /// `q4t_route2` (`QWEN4_SG_WGSL`): the ranking as k argmax rounds;
    /// None without 32-lane subgroups or with `CMF_QWEN_ROUTE2=0`.
    route2: Option<wgpu::ComputePipeline>,
    /// `q4_f16_matvec2`: the single-token pair (`q4t_f16_pair` reads four
    /// token rows per column even for one token); `CMF_QWEN_PAIR1=0` off.
    f16_matvec2: wgpu::ComputePipeline,
    /// `q4_q82_sg` (`QWEN4_SG_WGSL`): one-token q8_2f matvecs with the
    /// shuffle tail; None without the module or with `CMF_QWEN_Q82SG=0`.
    q82sg: Option<wgpu::ComputePipeline>,
    /// `q4_q82_sg1`: the long-row twin (one row a workgroup); `CMF_QWEN_Q82SG1=0` off.
    q82sg1: Option<wgpu::ComputePipeline>,
}

impl Pipes {
    /// Gate/up rows per workgroup of the resident-expert pair: 8 on the
    /// two-half twin (q2tp gate/up, inter % 8 == 0), else 4.
    fn gu_rows(&self, g: &Geom) -> usize {
        if self.gu4w.is_some() && g.gu_q2 && g.inter % 8 == 0 { 8 } else { 4 }
    }

    /// The row-blocked (gate/up, down) pair: the subgroup twins where they
    /// came up.
    fn experts_blocked(&self, g: &Geom) -> (&wgpu::ComputePipeline, &wgpu::ComputePipeline) {
        // vec4 nibble loads where every bank matrix starts on a 16-byte word
        let v4 = v4load() && !g.gu_q2 && q4tp_words16(g.inter, g.hidden) && q4tp_words16(g.hidden, g.inter);
        let gu = match &self.gu4w {
            Some(p) if self.gu_rows(g) == 8 => p,
            _ if v4 => &self.gu_q4tp4v,
            _ => self.gu4(g.gu_q2),
        };
        let dn = match (&self.dn4sv128, &self.dn4sv, &self.dn4s) {
            (Some(p), _, _) if v4 => p,
            (_, Some(p), _) if v4 => p,
            (_, _, Some(p)) => p,
            _ => &self.dn_q4tp4,
        };
        (gu, dn)
    }

    /// The row-blocked gate/up kernel for the bank's gate/up dtype (q2tp or
    /// q4tp). Both take the same layout, so a cached bind group fits either.
    fn gu4(&self, gu_q2: bool) -> &wgpu::ComputePipeline {
        if gu_q2 {
            &self.gu_q2tp4
        } else {
            &self.gu_q4tp4
        }
    }
}

/// The HC v3 module and its pipelines, specialized on first use per loop
/// count (`H3_KD` for the down kernel, `H3_KU` for the up-fold).
pub(crate) struct Hc3 {
    module: wgpu::ShaderModule,
    pipes: std::sync::Mutex<HashMap<(bool, u32), Option<wgpu::ComputePipeline>>>,
}

/// `CMF_QWEN_CHECKED=1`: build the qwen4 modules with naga's runtime checks
/// (bounds clamping, loop bounding, division guards) and zero-initialized
/// workgroup memory, as wgpu does by default. Off, they are trusted: every
/// index is in range by construction (the host guards in `Dev::new` and
/// the frame encoders, the audit notes on `build_pipes`) and every
/// workgroup slot a kernel reads is written earlier in the same dispatch.
fn qwen_checked() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_CHECKED").as_deref() == Ok("1"))
}

/// `CMF_QWEN_HC_V3=0`: every hyper-connection mix on the pre-v3 kernels
/// (group norm + `q4t_f16_pair` + `q4t_hc_upfold`, or `CMF_QWEN_HC_FUSE=1`'s
/// pair), the pending block injected by its own dispatch. The A/B arm.
fn hc_v3_env() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_HC_V3").as_deref() != Ok("0"))
}

/// `CMF_QWEN_HC_V3_INJECT=0`: v3 mixes, but the pending block still enters
/// the state through `q4t_inject` in front of them (isolates the fusion).
fn hc_v3_inject_env() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_HC_V3_INJECT").as_deref() != Ok("0"))
}

/// A shader module for the qwen4 kernels: trusted (no injected runtime
/// checks) unless `CMF_QWEN_CHECKED=1`.
fn qwen_module(c: &Ctx, label: &str, src: &str) -> wgpu::ShaderModule {
    let desc = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    };
    if qwen_checked() {
        c.device.create_shader_module(desc)
    } else {
        // SAFETY: see `qwen_checked` — the kernels index only inside the
        // buffers and workgroup tables their callers size for them, every
        // loop has a finite, uniform-bounded trip count, and no divisor is
        // zero (the shapes are checked on the host before a frame is built).
        unsafe {
            c.device
                .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
        }
    }
}

/// Does the adapter guarantee what `hc3_upfold` assumes of a subgroup: at
/// least 32 lanes (its xor-8/16 shuffles), at most 128 (`hc3_down`'s
/// per-subgroup tables), laid out linearly over the local index? Vulkan and
/// DX12 report the range the driver may pick from (NVIDIA: 32..32). Metal
/// reports a blanket 4..64, but Apple GPUs run 32-wide SIMD-groups.
fn hc3_subgroups_ok(c: &Ctx) -> bool {
    if !c.device.features().contains(wgpu::Features::SUBGROUP) {
        return false;
    }
    let i = &c.adapter_info;
    (i.subgroup_min_size >= 32 && i.subgroup_max_size <= 128)
        || (i.backend == wgpu::Backend::Metal && i.name.starts_with("Apple"))
}

/// The HC v3 module, in isolation: a rejection here must leave the qwen4
/// module and the older mixes untouched.
/// The q2tp twin of the matrix-unit gate/up GEMM.
fn build_mm_q2(c: &Ctx, gu: &wgpu::PipelineLayout) -> Option<wgpu::ComputePipeline> {
    if std::env::var("CMF_QWEN_MM").as_deref() == Ok("0")
        || !super::coop_matrix_active()
        || !c.device.features().contains(wgpu::Features::SUBGROUP)
        || c.adapter_info.subgroup_min_size != 32
        || c.adapter_info.subgroup_max_size != 32
    {
        return None;
    }
    let si = c.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = qwen_module(c, "qwen4-mm-q2", QWEN4_MM_WGSL);
    let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("q4g_mm_gu64q2"),
        layout: Some(gu),
        module: &module,
        entry_point: Some("q4g_mm_gu64q2"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: qwen_checked(),
        },
        cache: c.pipeline_cache.as_ref(),
    });
    let ev = pollster::block_on(sv.pop());
    let ei = pollster::block_on(si.pop());
    if let Some(e) = ev.or(ei) {
        tracing::warn!("qwen4 matrix-unit q2tp gate/up GEMM rejected: {e}");
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        return None;
    }
    Some(p)
}

/// The matrix-unit f16 pair GEMM (auto layout: the pair's own bind groups).
fn build_pair_coop(c: &Ctx) -> Option<wgpu::ComputePipeline> {
    if std::env::var("CMF_QWEN_PAIR_COOP").as_deref() == Ok("0")
        || !super::coop_matrix_active()
        || !c.device.features().contains(wgpu::Features::SUBGROUP)
        || c.adapter_info.subgroup_min_size != 32
        || c.adapter_info.subgroup_max_size != 32
    {
        return None;
    }
    let si = c.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = qwen_module(c, "qwen4-mm-pair", QWEN4_MM_WGSL);
    let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("q4t_f16_pair_coop"),
        layout: None,
        module: &module,
        entry_point: Some("q4t_f16_pair_coop"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: qwen_checked(),
        },
        cache: c.pipeline_cache.as_ref(),
    });
    let ev = pollster::block_on(sv.pop());
    let ei = pollster::block_on(si.pop());
    if let Some(e) = ev.or(ei) {
        tracing::warn!("qwen4 matrix-unit pair GEMM rejected: {e}");
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        return None;
    }
    Some(p)
}

/// The matrix-unit expert GEMMs, in isolation like HC v3: a rejection keeps
/// the scalar sorted kernels.
fn build_mm(
    c: &Ctx,
    gu: &wgpu::PipelineLayout,
    dn: &wgpu::PipelineLayout,
) -> Option<(wgpu::ComputePipeline, wgpu::ComputePipeline)> {
    if std::env::var("CMF_QWEN_MM").as_deref() == Ok("0")
        || !super::coop_matrix_active()
        || !c.device.features().contains(wgpu::Features::SUBGROUP)
        || c.adapter_info.subgroup_min_size != 32
        || c.adapter_info.subgroup_max_size != 32
    {
        return None;
    }
    let si = c.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = qwen_module(c, "qwen4-mm", QWEN4_MM_WGSL);
    let mk = |ep: &str, layout: &wgpu::PipelineLayout| {
        c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: Some(layout),
            module: &module,
            entry_point: Some(ep),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: qwen_checked(),
            },
            cache: c.pipeline_cache.as_ref(),
        })
    };
    // 64-column K slices for the gate/up GEMM (an even group count; 4K prompt
    // 837 → 858 tok/s on the PRO 6000); `CMF_QWEN_MM64=0` the 32-column one
    let pg = mk(if std::env::var("CMF_QWEN_MM64").as_deref() == Ok("0") { "q4g_mm_gu" } else { "q4g_mm_gu64" }, gu);
    let pd = mk("q4g_mm_dn", dn);
    let ev = pollster::block_on(sv.pop());
    let ei = pollster::block_on(si.pop());
    if let Some(e) = ev.or(ei) {
        tracing::warn!("qwen4 matrix-unit expert GEMMs rejected: {e}");
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        return None;
    }
    tracing::info!("qwen4 prompt experts on the matrix units (q4g_mm_gu / q4g_mm_dn)");
    Some((pg, pd))
}

fn build_hc3(c: &Ctx) -> Option<Hc3> {
    if !hc_v3_env() {
        return None;
    }
    if !hc3_subgroups_ok(c) {
        tracing::info!(
            "qwen4 HC v3 off: subgroup feature {} / sizes {}..{} on {:?}",
            c.device.features().contains(wgpu::Features::SUBGROUP),
            c.adapter_info.subgroup_min_size,
            c.adapter_info.subgroup_max_size,
            c.adapter_info.backend
        );
        return None;
    }
    let si = c.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = qwen_module(c, "qwen4-hc3", HC3_WGSL);
    let ev = pollster::block_on(sv.pop());
    let ei = pollster::block_on(si.pop());
    if let Some(e) = ev.or(ei) {
        tracing::warn!("qwen4 HC v3 module rejected: {e}");
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        return None;
    }
    Some(Hc3 {
        module,
        pipes: std::sync::Mutex::new(HashMap::new()),
    })
}

/// The one-pass `hc3_down1` for a frame of `nt` tokens: by default the
/// two-to-four-token frames (the drafts' verify window: 40 → 36 µs a mix on
/// the PRO 6000, one token equal); `CMF_QWEN_HC1=1` also one token, `=0` never.
fn hc1(nt: usize) -> bool {
    static S: OnceLock<u8> = OnceLock::new();
    match S.get_or_init(|| match std::env::var("CMF_QWEN_HC1").as_deref() {
        Ok("0") => 0,
        Ok("1") => 1,
        _ => 2,
    }) {
        0 => false,
        1 => nt <= 4,
        _ => (2..=4).contains(&nt),
    }
}

/// The stream-split HC down (`hc3_down4` + `hc3_upfold4`): four times the
/// workgroups of `hc3_down`, a mix 20.8 → 18.8 µs at one token on the PRO 6000;
/// `CMF_QWEN_HC4=0` keeps the pair of `hc3_down` (or `hc3_down1`) and `hc3_upfold`.
fn hc4() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_HC4").as_deref() != Ok("0"))
}

/// `hc3_down` (`down`) or `hc3_upfold` with its loop count fixed at `k`,
/// built on first use; None (remembered) when the device rejects it.
fn hc3_pipe(c: &Ctx, h: &Hc3, down: bool, k: u32) -> Option<wgpu::ComputePipeline> {
    let mut m = h.pipes.lock().ok()?;
    if let Some(p) = m.get(&(down, k)) {
        return p.clone();
    }
    // bit 15 of `k` asks for the one-pass down (`hc3_down1`), bit 14 for the
    // stream-split pair (`hc3_down4` / `hc3_upfold4`)
    let one = down && k & 0x8000 != 0;
    let split = k & 0x4000 != 0;
    let (ep, knob) = if down {
        (if split { "hc3_down4" } else if one { "hc3_down1" } else { "hc3_down" }, "H3_KD")
    } else {
        (if split { "hc3_upfold4" } else { "hc3_upfold" }, "H3_KU")
    };
    // Validation and Internal both: a shader the translator or the driver
    // turns down (`CreateComputePipelineError::Internal`) is not a
    // validation error, and outside a scope that takes it wgpu's default
    // handler panics instead of letting this mix fall back.
    let si = c.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let p = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: None,
            module: &h.module,
            entry_point: Some(ep),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[(knob, f64::from(k & 0x3FFF))],
                zero_initialize_workgroup_memory: qwen_checked(),
            },
            cache: c.pipeline_cache.as_ref(),
        });
    // scopes pop in reverse order
    let ev = pollster::block_on(sv.pop());
    let ei = pollster::block_on(si.pop());
    let p = match ev.or(ei) {
        None => Some(p),
        Some(e) => {
            tracing::warn!("qwen4 HC v3 pipeline {ep} ({knob} = {k}) rejected: {e}");
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            None
        }
    };
    m.insert((down, k), p.clone());
    p
}

// Workgroup-memory audit for the trusted, non-zero-initialized build (every
// `var<workgroup>` of QWEN4_WGSL / QWEN4T_WGSL; HC3_WGSL notes its own):
// - tree reductions (gn_part, hm_part, pm_part, uf_part, pg_part, bk_red,
//   qa_red, go_red, tg_part, tp_lo/hi, tu_lo/hi, tq_lo/hi, tn_red, th_lo/hi,
//   tv_lo/hi, rq2_red, ix2_red, qa2_red, gq_pg/pu, gv_pt): every lane of the
//   workgroup stores its slot before the first barrier; the tree reads
//   lid + s < size only. No partial fill.
// - th_ilo/th_ihi[8]: written for st < hc by lane 0, read for st < hc;
//   the encoder takes that kernel only with hc <= 8.
// - tr_used[64]: zeroed for lid < top_k, then rank < top_k set, read for
//   j < top_k (Dev::new: top_k <= 64). tr_sc[1024]: written and read for
//   i < n_experts (Dev::new: n_experts <= 1024).
// - tk2_keep[4096]: written and read for i < complete blocks (the frame
//   declines past MAX_INDEX_BLOCKS = 4096).
// - qa_w / qa2_w[2112]: written and read for i < m attended positions (the
//   frame declines past MAX_ATTEND = 2112). qa_qs / qa2_qs[256]: written for
//   lid < head_dim, read for k < head_dim (Dev::new: head_dim <= 256).
// - rq2_head[256] and the private xv[8]: written for d < hd, the RoPE reads
//   d < rotary dims <= hd (Dev::new: head_dim, index_dim <= 256, rotary_dim
//   <= head_dim; the indexer's rotary dims are min(rd, idim)).
// - bk_k[256]: every lane writes it; the RoPE reads d + rd/2 < rd <= idim.
// Nothing relied on the zero fill or on clamped indices.
fn build_pipes(c: &Ctx) -> Option<Pipes> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = qwen_module(c, "qwen4", &format!("{QWEN4_WGSL}{QWEN4T_WGSL}"));
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("qwen4 shader module rejected: {e}");
        return None;
    }
    let opts = || wgpu::PipelineCompilationOptions {
        constants: &[],
        zero_initialize_workgroup_memory: qwen_checked(),
    };
    let pipe = |ep: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: opts(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    // The row-blocked expert kernels bind the arena's eight segments as
    // binding arrays, which the automatic layout cannot express: explicit
    // layouts, like the arena's own pipelines.
    let storage = |binding: u32, read_only: bool, count: Option<u32>| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: count.and_then(std::num::NonZeroU32::new),
    };
    let segs = Some(8u32);
    let gu0 = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen4-gu4-0"),
            entries: &[
                storage(0, true, segs),
                storage(1, true, segs),
                storage(2, true, None),
                storage(3, true, None),
                storage(4, false, None),
            ],
        });
    let dn0 = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen4-dn4-0"),
            entries: &[
                storage(0, true, segs),
                storage(1, true, None),
                storage(2, true, None),
                storage(3, true, None),
                storage(4, false, None),
            ],
        });
    // the sorted twins read the frame's winner order at binding 5
    let gu0s = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen4-gu4s-0"),
            entries: &[
                storage(0, true, segs),
                storage(1, true, segs),
                storage(2, true, None),
                storage(3, true, None),
                storage(4, false, None),
                storage(5, true, None),
            ],
        });
    let dn0s = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen4-dn4s-0"),
            entries: &[
                storage(0, true, segs),
                storage(1, true, None),
                storage(2, true, None),
                storage(3, true, None),
                storage(4, false, None),
                storage(5, true, None),
            ],
        });
    // the expert tiles: no winner table (they walk the sorted segments)
    let gu0t = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen4-gut-0"),
            entries: &[
                storage(0, true, segs),
                storage(1, true, segs),
                storage(2, true, None),
                storage(4, false, None),
                storage(5, true, None),
                storage(6, true, None),
            ],
        });
    let dn0t = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen4-dnt-0"),
            entries: &[
                storage(0, true, segs),
                storage(1, true, None),
                storage(3, true, None),
                storage(4, false, None),
                storage(5, true, None),
                storage(6, true, None),
            ],
        });
    let params = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qwen4-expert-params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let gu_layout = c
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qwen4-gu4-layout"),
            bind_group_layouts: &[Some(&gu0), Some(&params)],
            immediate_size: 0,
        });
    let dn_layout = c
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qwen4-dn4-layout"),
            bind_group_layouts: &[Some(&dn0), Some(&params)],
            immediate_size: 0,
        });
    let gu_layout_s = c
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qwen4-gu4s-layout"),
            bind_group_layouts: &[Some(&gu0s), Some(&params)],
            immediate_size: 0,
        });
    let dn_layout_s = c
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qwen4-dn4s-layout"),
            bind_group_layouts: &[Some(&dn0s), Some(&params)],
            immediate_size: 0,
        });
    let gu_layout_t = c
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qwen4-gut-layout"),
            bind_group_layouts: &[Some(&gu0t), Some(&params)],
            immediate_size: 0,
        });
    let dn_layout_t = c
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qwen4-dnt-layout"),
            bind_group_layouts: &[Some(&dn0t), Some(&params)],
            immediate_size: 0,
        });
    let pipe_l = |ep: &str, layout: &wgpu::PipelineLayout| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(layout),
                module: &module,
                entry_point: Some(ep),
                compilation_options: opts(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    // the subgroup module of this context's device, built on first use
    let sgm: SgModule = std::cell::OnceCell::new();
    Some(Pipes {
        group_rmsnorm: pipe("q4_group_rmsnorm"),
        f16_matvec: pipe("q4_f16_matvec"),
        ple_gate: pipe("q4_ple_gate"),
        ple_conv: pipe("q4_ple_conv"),
        block_key: pipe("q4_qsa_block_key"),
        idx_build: pipe("q4_qsa_idx_build"),
        qsa_attend: pipe("q4_qsa_attend"),
        gate: pipe("q4_gate"),
        miss: pipe("q4_miss"),
        mtp_fuse: pipe("q4_mtp_fuse"),
        t_group_rmsnorm: pipe("q4t_group_rmsnorm"),
        t_f16_pair: pipe("q4t_f16_pair"),
        t_hc_upfold: pipe("q4t_hc_upfold"),
        t_q82_matvec: pipe("q4t_q82_matvec"),
        t_inject: pipe("q4t_inject"),
        t_gdn_norm: pipe("q4t_gdn_norm"),
        t_route: pipe("q4t_route"),
        embed_gather_q82: pipe("q4_embed_gather_q82"),
        gu_q2tp4: pipe_l("q4_gu_q2tp4", &gu_layout),
        gu_q4tp4: pipe_l("q4_gu_q4tp4", &gu_layout),
        gu_q4tp4v: pipe_l("q4_gu_q4tp4v", &gu_layout),
        dn_q4tp4: pipe_l("q4_dn_q4tp4", &dn_layout),
        t_hc_down: pipe("q4t_hc_down"),
        t_hc_upfold2: pipe("q4t_hc_upfold2"),
        t_rope: pipe("q4t_rope"),
        t_kv_append: pipe("q4t_kv_append"),
        t_rawk: pipe("q4t_rawk"),
        t_ix_scores: pipe("q4t_ix_scores"),
        t_ix_scores_l: pipe("q4t_ix_scores_l"),
        t_topk_r: pipe("q4t_topk_r"),
        t_topk: pipe("q4t_topk"),
        t_idx_build: pipe("q4t_idx_build"),
        t_attend: pipe("q4t_attend"),
        t_attend_sp: pipe(if attn_sp2() { "q4t_attend_sp2" } else { "q4t_attend_sp" }),
        t_attend_cmb: pipe("q4t_attend_cmb"),
        t_f16_pair_mm: pipe("q4t_f16_pair_mm"),
        t_pair_sum: pipe("q4t_pair_sum"),
        g_gu_q4tp4: pipe_l("q4g_gu_q4tp4", &gu_layout_s),
        g_dn_q4tp4: pipe_l("q4g_dn_q4tp4", &dn_layout_s),
        g_sum: pipe("q4g_sum"),
        g_sort: pipe("q4g_sort"),
        g_segs: pipe("q4g_segs"),
        g_cs: [pipe("q4g_cs_clear"), pipe("q4g_cs_count"), pipe("q4g_cs_scan"), pipe("q4g_cs_scatter")],
        mm: build_mm(c, &gu_layout_t, &dn_layout_t),
        mm_gu_q2: build_mm_q2(c, &gu_layout_t),
        pair_coop: build_pair_coop(c),
        t_hc_fold_t: pipe("q4t_hc_fold_t"),
        t_o1_pack: pipe("q4t_o1_pack"),
        t_o1_gate: pipe("q4t_o1_gate"),
        t_tube_gu: pipe("q4t_tube_gu"),
        t_tube_dn: pipe("q4t_tube_dn"),
        hc3: build_hc3(c),
        gu4w: build_sg_l(c, &sgm, "q4_gu_q2tp4w", "CMF_QWEN_GU4W", true, &gu_layout, opts()),
        dn4s: build_sg_l(c, &sgm, "q4_dn_q4tp4s", "CMF_QWEN_DN4S", true, &dn_layout, opts()),
        dn4sv: build_sg_l(c, &sgm, "q4_dn_q4tp4sv", "CMF_QWEN_V4LOAD", true, &dn_layout, opts()),
        dn4sv128: build_sg_l(c, &sgm, "q4_dn_q4tp4sv128", "CMF_QWEN_DN128", true, &dn_layout, opts()),
        route2: build_sg_auto(c, &sgm, "q4t_route2", "CMF_QWEN_ROUTE2", opts()),
        f16_matvec2: pipe("q4_f16_matvec2"),
        q82sg: build_sg_auto(c, &sgm, "q4_q82_sg", "CMF_QWEN_Q82SG", opts()),
        q82sg1: build_sg_auto(c, &sgm, "q4_q82_sg1", "CMF_QWEN_Q82SG1", opts()),
    })
}

/// The subgroup module (`QWEN4_SG_WGSL` on top of the qwen4 sources) of
/// one context, built on first use: once per device, since a module (and
/// the subgroup-size check) belongs to the device its context drives.
type SgModule = std::cell::OnceCell<Option<wgpu::ShaderModule>>;

/// The subgroup module on 32-lane-subgroup devices; None elsewhere or on
/// rejection.
fn sg_module<'a>(c: &Ctx, m: &'a SgModule) -> Option<&'a wgpu::ShaderModule> {
    m.get_or_init(|| {
        if !c.device.features().contains(wgpu::Features::SUBGROUP)
            || c.adapter_info.subgroup_min_size != 32
            || c.adapter_info.subgroup_max_size != 32
        {
            return None;
        }
        let si = c.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = qwen_module(
            c,
            "qwen4-sg",
            &format!("{QWEN4_WGSL}{QWEN4T_WGSL}{QWEN4_SG_WGSL}"),
        );
        let ev = pollster::block_on(sv.pop());
        let ei = pollster::block_on(si.pop());
        if let Some(e) = ev.or(ei) {
            tracing::warn!("qwen4 subgroup module rejected: {e}");
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            return None;
        }
        Some(module)
    })
    .as_ref()
}

/// One entry point of the subgroup module on an automatic layout; None when
/// `env` is "0", without the module, or on rejection.
fn build_sg_auto(
    c: &Ctx,
    sgm: &SgModule,
    ep: &str,
    env: &str,
    opts: wgpu::PipelineCompilationOptions<'_>,
) -> Option<wgpu::ComputePipeline> {
    if std::env::var(env).as_deref() == Ok("0") {
        return None;
    }
    let module = sg_module(c, sgm)?;
    let si = c.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let p = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: None,
            module,
            entry_point: Some(ep),
            compilation_options: opts,
            cache: c.pipeline_cache.as_ref(),
        });
    let ev = pollster::block_on(sv.pop());
    let ei = pollster::block_on(si.pop());
    if let Some(e) = ev.or(ei) {
        tracing::warn!("qwen4 {ep} rejected: {e}");
        return None;
    }
    Some(p)
}

/// The eight-row expert pair, in its own module (subgroup builtins).
fn build_sg_l(
    c: &Ctx,
    sgm: &SgModule,
    ep: &str,
    env: &str,
    default_on: bool,
    layout: &wgpu::PipelineLayout,
    opts: wgpu::PipelineCompilationOptions<'_>,
) -> Option<wgpu::ComputePipeline> {
    let on = match std::env::var(env).as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => default_on,
    };
    if !on {
        return None;
    }
    let module = sg_module(c, sgm)?;
    let si = c.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let p = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: Some(layout),
            module,
            entry_point: Some(ep),
            compilation_options: opts,
            cache: c.pipeline_cache.as_ref(),
        });
    let ev = pollster::block_on(sv.pop());
    let ei = pollster::block_on(si.pop());
    if let Some(e) = ev.or(ei) {
        tracing::warn!("qwen4 {ep} rejected: {e}");
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        return None;
    }
    Some(p)
}

fn pipes(c: &Ctx) -> Option<&Pipes> {
    c.qwen4_pipes.get_or_init(|| build_pipes(c)).as_ref()
}

/// Is the device path available on this adapter at all (wgpu context,
/// the extra kernels compile, the segmented expert arena is supported)?
pub(crate) fn available() -> bool {
    let Some(c) = ctx() else { return false };
    pipes(c).is_some() && dsv4_global_moe_supported()
}

pub(crate) fn vram_budget() -> Option<u64> {
    ctx().map(|c| c.vram_budget)
}

/// Bytes of model weights currently resident through the per-tensor arena.
pub(crate) fn resident_bytes() -> u64 {
    ctx().map_or(0, |c| c.resident.load(std::sync::atomic::Ordering::Relaxed))
}

// ── geometry and weight descriptors ──

#[derive(Clone, Copy)]
pub(crate) struct GdnGeom {
    pub nv: usize,
    pub nk: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct Geom {
    pub hidden: usize,
    pub hc: usize,
    pub eps: f32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_budget: usize,
    pub compress_ratio: usize,
    pub gdn: GdnGeom,
    pub ple_kernel: usize,
    pub ple_dilation: usize,
    pub top_k: usize,
    pub n_experts: usize,
    pub inter: usize,
    pub gu_q2: bool,
}

pub(crate) struct HcW<'a> {
    pub norm: &'a [f32],
    pub down: usize,
    pub up: usize,
    pub inject: Option<usize>,
}

pub(crate) enum MixerW<'a> {
    Gdn {
        qkv: usize,
        z: usize,
        a: usize,
        b: usize,
        out: usize,
        conv1d: &'a [f32],
        a_log: &'a [f32],
        dt_bias: &'a [f32],
        norm: &'a [f32],
    },
    Qsa {
        q: usize,
        k: usize,
        v: usize,
        o: usize,
        index_qk: usize,
        q_norm: &'a [f32],
        k_norm: &'a [f32],
        iq_norm: &'a [f32],
        ik_norm: &'a [f32],
    },
}

pub(crate) struct PleW<'a> {
    pub key_proj: usize,
    pub value_proj: usize,
    pub norm_key: &'a [f32],
    pub norm_query: &'a [f32],
    pub norm_conv: &'a [f32],
    pub conv: &'a [f32],
}

pub(crate) struct LayerW<'a> {
    pub attn_hc: HcW<'a>,
    pub mlp_hc: HcW<'a>,
    pub mixer: MixerW<'a>,
    pub ple: Option<PleW<'a>>,
    pub router: usize,
    pub shared_gate: Option<usize>,
}

// ── device state ──

struct QsaDev {
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    /// Positions the K/V buffers hold.
    cap: usize,
    /// Raw indexer keys `[kcap][idim]` and compressed block keys
    /// `[kcap / cr][idim]`.
    rawk: wgpu::Buffer,
    ckeys: wgpu::Buffer,
    kcap: usize,
}

impl QsaDev {
    /// Caches for `cap` positions.
    fn new(c: &Ctx, g: &Geom, cap: usize) -> Self {
        QsaDev {
            k: storage_buf(c, "qwen4-k", (g.n_kv_heads * cap * g.head_dim * 4) as u64),
            v: storage_buf(c, "qwen4-v", (g.n_kv_heads * cap * g.head_dim * 4) as u64),
            cap,
            rawk: storage_buf(c, "qwen4-rawk", (cap * g.index_dim * 4) as u64),
            ckeys: storage_buf(c, "qwen4-ckeys", ((cap / g.compress_ratio.max(1)) * g.index_dim * 4) as u64),
            kcap: cap,
        }
    }
}

struct PleDev {
    hist: wgpu::Buffer,
    cap: usize,
    head: usize,
    rows: usize,
}

struct LayerDev {
    gdn: Option<(wgpu::Buffer, wgpu::Buffer)>,
    qsa: Option<QsaDev>,
    ple: Option<PleDev>,
}

/// Tokens one frame can carry (prefill chunks, the verify window).
pub(crate) const TMAX: usize = 8;
/// Rows of a prompt frame on a resident cut model (every expert on the card, so
/// no cold pass): the frame buffers hold this many tokens; frames past `TMAX`
/// take the GEMM kernels where the eight-token ones stop.
pub(crate) const TROWS: usize = 256;

/// One MoE layer's expert tubes on the card (`hot-bake --experts`): the q8_2f tube tensors
/// of the experts that have one, packed one after another at a fixed segment (`slots`
/// names each expert's entry), gate and up `[h, hidden]`, down `[hidden, h]`;
/// `q4t_tube_gu` / `q4t_tube_dn` read them for the frame's winners.
pub(crate) struct TubeBank {
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
    down: wgpu::Buffer,
    /// Expert e's entry in the bank (`u32::MAX`: no tube), `[experts]`.
    slots: wgpu::Buffer,
    /// Neurons a tube holds.
    h: usize,
    /// Words per expert in the gate / up banks and in the down bank.
    gseg: u32,
    dseg: u32,
}

/// The attention layers' O(1) far field (`crate::qwen4_o1`): the exact prompt leaves a
/// query trace, the seal builds a sealed state per layer in `Ctx::o1m`, and later frames
/// attend through `o1_batch`.
pub(crate) struct O1Run {
    pub(crate) cfg: crate::qwen4_o1::Qwen4O1,
    geom: Geom,
    /// Past the seal: the attention layers read the far field.
    pub(crate) sealed: bool,
    /// The far field did not fit the kernels once; the layers stay exact.
    pub(crate) refused: bool,
    epoch: u64,
    /// The exact prompt's rotated queries, one row in `QT_STRIDE`, per layer
    /// (`[row / QT_STRIDE][nh·hd]`; None on a GDN layer).
    qtrace: Vec<Option<wgpu::Buffer>>,
    /// Rows the trace holds.
    qcap: usize,
    /// The frame's rows packed for the far-field kernels: Q `[TMAX][nh·hd]`, K and V
    /// `[TMAX][nkv·hd]`, their attention `[TMAX][nh·hd]`, the batch kernels' scratch.
    pq: wgpu::Buffer,
    pk: wgpu::Buffer,
    pv: wgpu::Buffer,
    out: wgpu::Buffer,
    ob: wgpu::Buffer,
}

pub(crate) struct Dev {
    uid: u64,
    /// The hyper state of every token slot of a frame: TMAX rows of
    /// `hc·hidden` floats (a row is 256-byte aligned, so one slot binds).
    hyper: wgpu::Buffer,
    hh: usize,
    /// Word 0: the pending inject's cold flags (bit 1 host completion,
    /// bit 2 the card's cold pass), written at finalize.
    inj_flags: wgpu::Buffer,
    layers: Vec<LayerDev>,
    ixw: Vec<f32>,
    pub pos: usize,
    /// The previous layer's MoE output still has to enter the hyper state
    /// (with the host's cold-expert completion). Encoded by the next frame.
    pending_inject: bool,
    /// Bind groups by (layer, step): every buffer a frame binds has a
    /// permanent identity (pinned weights, grow-only frame scratch, per-slot
    /// uniforms), so a layer's groups are built once. Growing a layer's
    /// caches drops that layer's entries.
    binds: std::cell::RefCell<HashMap<(usize, u16, usize), wgpu::BindGroup>>,
    /// Two readback staging buffers: the next frame's copy lands in one
    /// while the host still reads the other.
    stages: [wgpu::Buffer; 2],
    stage_ix: usize,
    /// Indirect dispatch sizes of the cold-expert pass, written when the
    /// cold list is known — after the frame was encoded.
    cold_args: wgpu::Buffer,
    /// What `finalize_pending` last wrote to `cold_args` and `inj_flags`:
    /// most frames repeat it (no cold winner), and two queue writes per
    /// layer sit on the path between one frame's fence and the next submit.
    fin_last: std::cell::Cell<Option<([u32; 8], u32)>>,
    /// The cold pass's gate/up and down uniforms (`CMF_QWEN_COLD_K`): the
    /// slot count per token is the frame's largest cold list, written at
    /// finalize with the indirect sizes; `cold_tpl` holds the other words,
    /// `cold_k` the count last written (0: none yet).
    cold_u: [wgpu::Buffer; 2],
    cold_tpl: std::cell::Cell<Option<([u32; 8], [u32; 8])>>,
    cold_k: std::cell::Cell<usize>,
    /// Every dispatch of a layer reads its workgroup count from `args_live`,
    /// which the layer's gate kernel fills from `args_tpl` — or with zeros
    /// once an earlier layer of the chain routed to a cold expert. The host
    /// records the template (`tpl`) while encoding and uploads the chain's
    /// rows before the submit.
    args_tpl: wgpu::Buffer,
    args_live: wgpu::Buffer,
    tpl: std::cell::RefCell<Vec<u32>>,
    rows: usize,
    /// The chain's miss flag (one word), reset to zero at every submit.
    miss: wgpu::Buffer,
    /// PLE passes encoded per layer but not yet committed (the history ring
    /// advances on commit, once per token that actually ran).
    ple_touched: Vec<usize>,
    /// The final hyper state of the last processed position (R of the MTP's
    /// next cell), kept across forwards.
    pub(crate) r_last: wgpu::Buffer,
    /// Verify-window snapshots: per GDN layer one buffer of `rows` slots,
    /// each the conv ring then the recurrent state AFTER that token's step
    /// (the looped kernels write them as they go); per PLE layer the
    /// history ring before each token. A rejected draft is undone by
    /// copying them back.
    snaps: Vec<Option<(wgpu::Buffer, usize)>>,
    ple_snaps: Vec<Vec<(wgpu::Buffer, usize, usize)>>,
    /// Chains longer than one frame need the gate/miss machinery and
    /// indirect dispatch; single-frame chains dispatch directly (measured:
    /// ~1200 indirect dispatches a token cost 6-8 ms on the NVIDIA Vulkan
    /// stack, and an aborted frame still pays every launch).
    pub gated: bool,
    /// The far field of the attention layers, when the file or `CMF_QWEN_O1` asks for one.
    pub(crate) o1: Option<O1Run>,
    /// The experts' tube banks per layer (None: no tubes in that layer) and the frame's
    /// tube activations `[TMAX][top_k][h]`.
    tubes: Vec<Option<TubeBank>>,
    tube_act: Option<wgpu::Buffer>,
    /// The shared expert's tube per layer (`hot-bake --shared-q8`: the whole expert at
    /// eight bits) and its activations `[TMAX][h]`.
    tubes_shared: Vec<Option<TubeBank>>,
    tube_act_s: Option<wgpu::Buffer>,
    /// The streamed prompt (`forward_stream_device`): per frame of a chunk the state that
    /// crosses a layer boundary — the hyper rows, the MoE output and its inject weights —
    /// parked while the chunk's other frames run the same layer.
    sbank: Vec<[wgpu::Buffer; 3]>,
    /// The streamed prompt's routing: every (layer, frame)'s cold/pick rows copied out,
    /// read once when the chunk is done (`route_cap_read`).
    route_cap: Option<wgpu::Buffer>,
    /// Routing kept to the experts the arena holds (`set_masked`): the narrow bundle of a
    /// request decodes with no cold winner, as a resident model does.
    pub(crate) masked: bool,
}

/// Indirect-argument slots per layer row (16 bytes each).
const SLOTS: usize = 88;

/// A dispatch site's slot in its layer's argument row. Step ids are fixed
/// per site; this folds them into a dense index.
fn step_slot(step: u16) -> usize {
    let s = match step {
        0..=9 => step as usize,                  // in-chain inject (2), gate (9)
        10..=19 => 10 + (step as usize - 10),    // PLE
        100..=109 => 20 + (step as usize - 100), // attention HC
        200..=209 => 30 + (step as usize - 200), // MoE HC
        300..=309 => 20 + (step as usize - 300), // head HC (its own row)
        400..=409 => 40 + (step as usize - 400), // GDN (409: the prompt scan)
        500..=513 => 40 + (step as usize - 500), // QSA (never with GDN in one row)
        520..=523 => 70 + (step as usize - 520), // QSA O(1) pack / gate
        524 => 51,                               // split attention (in the per-head attend's slot)
        525 => 78,                               // its fold
        720..=723 => 74 + (step as usize - 720), // the experts' tubes, the shared expert's
        600 => 54,                               // attention inject
        700..=713 => 55 + (step as usize - 700), // MoE route / experts / miss
        714 => 58,                               // the route's other kernel
        800 => 69,                               // lm_head (head row)
        110..=113 => 70 + (step as usize - 100), // HC stream-split pair (attention)
        310..=313 => 70 + (step as usize - 300), // the head's (its own row)
        210..=213 => 74 + (step as usize - 200), // the MoE's
        _ => 79,
    };
    s.min(SLOTS - 1)
}

/// Cost probe, not a release path: `CMF_Q4_SKIP_STEP=a,b,..` leaves the
/// launches of those bind-cache steps out (the output goes wrong, the time
/// of the rest stays), `CMF_Q4_STEP_LOG=1` prints every distinct (step,
/// grid) once with its layer. True when the launch is to be left out.
fn step_probe(step: u16, li: usize, groups: (u32, u32, u32)) -> bool {
    static S: OnceLock<(Vec<u16>, bool)> = OnceLock::new();
    let (skip, log) = S.get_or_init(|| {
        let v = std::env::var("CMF_Q4_SKIP_STEP")
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_default();
        (v, std::env::var("CMF_Q4_STEP_LOG").as_deref() == Ok("1"))
    });
    if *log {
        static SEEN: std::sync::Mutex<Option<std::collections::HashSet<(u16, (u32, u32, u32))>>> = std::sync::Mutex::new(None);
        if let Ok(mut g) = SEEN.lock() {
            if g.get_or_insert_with(Default::default).insert((step, groups)) {
                eprintln!("[step] {step} li {li} grid {groups:?}");
            }
        }
    }
    !skip.is_empty() && skip.contains(&step)
}

/// `CMF_Q4_SKIP=hc,gdn,qsa,route,experts,ple,head`: leave a stage out of the
/// frame. The output is wrong; the timing says what the stage costs.
fn skip(stage: &str) -> bool {
    static S: OnceLock<Vec<String>> = OnceLock::new();
    S.get_or_init(|| {
        std::env::var("CMF_Q4_SKIP")
            .map(|v| v.split(',').map(|x| x.trim().to_string()).collect())
            .unwrap_or_default()
    })
    .iter()
    .any(|x| x == stage)
}

impl Dev {
    /// A cached bind group for `(layer, step, token slot)`, built on first use.
    fn bind<F: FnOnce() -> wgpu::BindGroup>(
        &self,
        li: usize,
        step: u16,
        tok: usize,
        build: F,
    ) -> wgpu::BindGroup {
        // Several cells of one submit (the MTP draft chain) salt their
        // frame: their per-position uniforms differ, so their bind groups do.
        let key = (li, step, tok + dsv4_frame_salt() * 4096);
        if let Some(b) = self.binds.borrow().get(&key) {
            return b.clone();
        }
        let b = build();
        self.binds.borrow_mut().insert(key, b.clone());
        b
    }

    /// The attention layers' far field (`crate::qwen4_o1`) for this sequence: the trace
    /// and packing buffers, the state itself at the seal.
    pub(crate) fn set_o1(&mut self, g: &Geom, cfg: crate::qwen4_o1::Qwen4O1) -> bool {
        let Some(c) = ctx() else { return false };
        if o1_batch::pipes(c).is_none() {
            tracing::warn!("qwen4 O(1): the batch kernels are unavailable, the layers stay exact");
            return false;
        }
        let (nh, nkv, hd) = (g.n_heads, g.n_kv_heads, g.head_dim);
        let qcap = (cfg.o1.seal + TROWS).div_ceil(crate::dory::QT_STRIDE);
        let qtrace = self
            .layers
            .iter()
            .map(|l| l.qsa.as_ref().map(|_| storage_buf(c, "qwen4-o1-qtrace", (qcap * nh * hd * 4) as u64)))
            .collect();
        let geom = o1_batch::Geom { g: nkv, hpg: nh / nkv, m: cfg.o1.m, w: cfg.o1.w, ns: cfg.near() - cfg.o1.w, rect_fm: false, d: hd, dv: hd, scale: 0.0 };
        self.o1 = Some(O1Run {
            cfg,
            geom: *g,
            sealed: false,
            refused: false,
            epoch: 0,
            qtrace,
            qcap,
            pq: storage_buf(c, "qwen4-o1-pq", (TROWS * nh * hd * 4) as u64),
            pk: storage_buf(c, "qwen4-o1-pk", (TROWS * nkv * hd * 4) as u64),
            pv: storage_buf(c, "qwen4-o1-pv", (TROWS * nkv * hd * 4) as u64),
            out: storage_buf(c, "qwen4-o1-out", (TROWS * nh * hd * 4) as u64),
            ob: storage_buf(c, "qwen4-o1-ob", (geom.scratch_len() * 4) as u64),
        });
        true
    }

    /// The experts' tubes of every MoE layer on the card: `segs[li][e]` names each expert's
    /// (gate, up, down) tube tensors (directory indices, q8_2f, every expert of a layer
    /// the same width). Bytes land resident; false when a layer's tubes disagree, a tensor
    /// is not q8_2f or the upload fails — the device path then stays off (its kernels
    /// would compute the zeroed rows and nothing else).
    pub(crate) fn set_tubes(&mut self, model: &CmfModel, g: &Geom, segs: &[Vec<Option<(usize, usize, usize)>>], shared: &[Option<(usize, usize, usize)>]) -> bool {
        let Some(c) = ctx() else { return false };
        // The banks stay resident beside the skeleton (already up): refuse what the budget
        // cannot hold with a slice left for the experts and the caches — a wgpu allocation
        // past the card's memory is fatal, and a device path without the tubes would run the
        // experts' zeroed rows.
        let bank_bytes = |layer: &[Option<(usize, usize, usize)>]| -> u64 {
            let Some((gi, _, _)) = layer.iter().flatten().next() else { return 0 };
            let Some(e) = model.tensors.get(*gi) else { return 0 };
            let Some(&h) = e.shape.first() else { return 0 };
            let (h, hidden) = (h as usize, g.hidden);
            let seg_g = cortiq_core::quant::expected_nbytes(TensorDtype::Q8_2f, &[h, hidden]).unwrap_or(0).div_ceil(16) * 16;
            let seg_d = cortiq_core::quant::expected_nbytes(TensorDtype::Q8_2f, &[hidden, h]).unwrap_or(0).div_ceil(16) * 16;
            (layer.iter().flatten().count() * (2 * seg_g + seg_d)) as u64
        };
        let need: u64 = segs.iter().map(|l| bank_bytes(l)).sum::<u64>() + shared.iter().map(|t| bank_bytes(std::slice::from_ref(t))).sum::<u64>();
        let resident = c.resident.load(std::sync::atomic::Ordering::Relaxed);
        let headroom = 2u64 << 30;
        if resident + need + headroom > c.vram_budget {
            tracing::warn!(
                "qwen4 tubes: {} MB of tubes do not fit the VRAM budget ({} MB, {} MB resident) with 2 GB left for the experts",
                need >> 20,
                c.vram_budget >> 20,
                resident >> 20
            );
            return false;
        }
        let mut banks = Vec::with_capacity(segs.len());
        let mut hmax = 0usize;
        for (li, layer) in segs.iter().enumerate() {
            match Self::tube_bank(c, model, g, li, layer) {
                Ok(b) => {
                    hmax = hmax.max(b.as_ref().map_or(0, |b| b.h));
                    banks.push(b);
                }
                Err(()) => return false,
            }
        }
        let mut sbanks = Vec::with_capacity(shared.len());
        let mut hsmax = 0usize;
        for (li, t) in shared.iter().enumerate() {
            match Self::tube_bank(c, model, g, li, std::slice::from_ref(t)) {
                Ok(b) => {
                    hsmax = hsmax.max(b.as_ref().map_or(0, |b| b.h));
                    sbanks.push(b);
                }
                Err(()) => return false,
            }
        }
        if hmax > 0 {
            self.tube_act = Some(storage_buf(c, "qwen4-tube-act", (TROWS * g.top_k * hmax * 4) as u64));
            self.tubes = banks;
        }
        if hsmax > 0 {
            self.tube_act_s = Some(storage_buf(c, "qwen4-tube-act-s", (TMAX * hsmax * 4) as u64));
            self.tubes_shared = sbanks;
        }
        c.queue.submit(std::iter::empty());
        true
    }

    /// One layer's tube bank from the tube tensors of its experts (None: no tubes there).
    fn tube_bank(c: &Ctx, model: &CmfModel, g: &Geom, li: usize, layer: &[Option<(usize, usize, usize)>]) -> Result<Option<TubeBank>, ()> {
        let hidden = g.hidden;
        let bytes_of = |idx: usize| -> Option<(&[u8], usize, usize)> {
            let e = model.tensors.get(idx)?;
            if e.dtype != TensorDtype::Q8_2f || e.shape.len() != 2 {
                return None;
            }
            let abs = model.entry_abs_offset(e)?;
            let b = model.primary_bytes().get(abs..abs + e.nbytes as usize)?;
            Some((b, e.shape[0] as usize, e.shape[1] as usize))
        };
        {
            if layer.is_empty() || layer.iter().all(Option::is_none) {
                return Ok(None);
            }
            let Some((gi, _, di)) = layer.iter().flatten().next().copied() else {
                return Ok(None);
            };
            let (Some((_, h, cols)), Some((_, drows, dcols))) = (bytes_of(gi), bytes_of(di)) else {
                tracing::warn!("qwen4 tubes: layer {li}: a tube tensor is not a 2-D q8_2f tensor");
                return Err(());
            };
            if cols != hidden || drows != hidden || dcols != h || h % 4 != 0 || h == 0 || hidden % 4 != 0 {
                tracing::warn!("qwen4 tubes: layer {li}: tube [{h}, {cols}] / down [{drows}, {dcols}] against hidden {hidden}");
                return Err(());
            }
            let seg_g = cortiq_core::quant::expected_nbytes(TensorDtype::Q8_2f, &[h, hidden]).unwrap_or(0).div_ceil(16) * 16;
            let seg_d = cortiq_core::quant::expected_nbytes(TensorDtype::Q8_2f, &[hidden, h]).unwrap_or(0).div_ceil(16) * 16;
            // The experts with a tube, packed: `slots[e]` names expert e's entry.
            let n = layer.iter().flatten().count();
            let mut slots = vec![u32::MAX; layer.len()];
            let (mut gb, mut ub, mut db) = (vec![0u8; n * seg_g], vec![0u8; n * seg_g], vec![0u8; n * seg_d]);
            let mut e = 0usize;
            for (ex, t) in layer.iter().enumerate() {
                let Some((gi, ui, di)) = t else {
                    continue;
                };
                slots[ex] = e as u32;
                for (idx, dst, seg, rows, cols) in [(*gi, &mut gb, seg_g, h, hidden), (*ui, &mut ub, seg_g, h, hidden), (*di, &mut db, seg_d, hidden, h)] {
                    let Some((b, r, cc)) = bytes_of(idx) else {
                        tracing::warn!("qwen4 tubes: layer {li}: expert {e}'s tube tensor is not q8_2f");
                        return Err(());
                    };
                    if r != rows || cc != cols || b.len() > seg {
                        tracing::warn!("qwen4 tubes: layer {li}: expert {e}'s tube is [{r}, {cc}], the layer's [{rows}, {cols}]");
                        return Err(());
                    }
                    dst[e * seg..e * seg + b.len()].copy_from_slice(b);
                }
                e += 1;
            }
            let up = |label: &str, data: &[u8]| -> wgpu::Buffer {
                let b = storage_buf(c, label, data.len() as u64);
                c.queue.write_buffer(&b, 0, data);
                b
            };
            c.resident.fetch_add((gb.len() + ub.len() + db.len()) as u64, std::sync::atomic::Ordering::Relaxed);
            let sb: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
            let bank = TubeBank { gate: up("qwen4-tube-g", &gb), up: up("qwen4-tube-u", &ub), down: up("qwen4-tube-d", &db), slots: up("qwen4-tube-slots", &sb), h, gseg: (seg_g / 4) as u32, dseg: (seg_d / 4) as u32 };
            // Flush this layer's writes: their staging memory is device memory too, and
            // 6 GB of banks staged at once left the expert pool's allocation no room.
            c.queue.submit(std::iter::empty());
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            Ok(Some(bank))
        }
    }

    /// Tube bytes resident on the card.
    pub(crate) fn tube_bytes(&self) -> u64 {
        self.tubes.iter().chain(self.tubes_shared.iter()).flatten().map(|b| b.gate.size() + b.up.size() + b.down.size()).sum()
    }

    /// The model's id, the key of its far states in `Ctx::o1m`.
    pub(crate) fn uid(&self) -> u64 {
        self.uid
    }

    /// Rows at which the sequence seals, when a far field is configured and not yet built.
    pub(crate) fn o1_seal_at(&self) -> Option<usize> {
        self.o1.as_ref().filter(|o| !o.sealed && !o.refused).map(|o| o.cfg.o1.seal)
    }

    /// Past the seal: the attention layers read the far field.
    pub(crate) fn o1_sealed(&self) -> bool {
        self.o1.as_ref().is_some_and(|o| o.sealed && !o.refused)
    }

    /// Seal the sequence's attention layers at `rows` positions: their keys, values and
    /// query traces come to the host, `nystrom` builds a sink + window + landmark state
    /// per KV head (the chosen rows exact among the sinks), the states go up as the
    /// layers' O(1) states and the exact caches leave the card. False, and the layers
    /// stay exact, when a state does not fit the kernels.
    pub(crate) fn seal(&mut self, g: &Geom, rows: usize) -> bool {
        let Some(c) = ctx() else { return false };
        let Some(o) = self.o1.as_ref() else { return false };
        if o.sealed || o.refused || rows == 0 {
            return false;
        }
        let (nh, nkv, hd) = (g.n_heads, g.n_kv_heads, g.head_dim);
        let o1 = o.cfg.o1;
        let stride = crate::dory::QT_STRIDE;
        let t0 = std::time::Instant::now();
        let refuse = |this: &mut Self, why: &str| -> bool {
            tracing::warn!("qwen4 O(1): {why}; the attention layers stay exact");
            c.o1m.lock().unwrap().retain(|(id, _), _| *id != this.uid);
            if let Some(o) = this.o1.as_mut() {
                o.refused = true;
            }
            false
        };
        if rows.div_ceil(stride) > o.qcap {
            return refuse(self, &format!("{rows} rows go past the query trace of {} rows", o.qcap * stride));
        }
        // the layers' rows, `[row][nkv][hd]`, and their query traces
        let mut got: Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
        for (li, l) in self.layers.iter().enumerate() {
            let (Some(q), Some(tr)) = (l.qsa.as_ref(), o.qtrace[li].as_ref()) else { continue };
            if rows > q.cap {
                return refuse(self, &format!("{rows} rows go past layer {li}'s cache of {}", q.cap));
            }
            let (Some(ks), Some(vs), Some(qs)) = (
                train_ops::download(c, &q.k, nkv * q.cap * hd),
                train_ops::download(c, &q.v, nkv * q.cap * hd),
                train_ops::download(c, tr, rows.div_ceil(stride) * nh * hd),
            ) else {
                return refuse(self, "the caches could not be read back");
            };
            got.push((
                li,
                crate::qwen4_o1::rows_from_heads(&ks, nkv, q.cap, hd, rows),
                crate::qwen4_o1::rows_from_heads(&vs, nkv, q.cap, hd, rows),
                qs,
            ));
        }
        if got.is_empty() {
            return refuse(self, "no attention layer to seal");
        }
        // The chosen rows leave the far field: a copy of a fact there drowns the exact one.
        let keep = match &o.cfg.exact {
            Some(x) => {
                let views: Vec<crate::dory::FlatRows> = got.iter().map(|e| crate::dory::FlatRows { k: &e.1, v: &e.2, nk: nkv, hd }).collect();
                let layers: Vec<&dyn crate::dory::LayerRows> = views.iter().map(|v| v as &dyn crate::dory::LayerRows).collect();
                crate::dory::pick_exact_rows(x, &layers, rows, nkv, hd, o1.m, o1.sink, rows.saturating_sub(o1.w))
            }
            None => Vec::new(),
        };
        let order = crate::dory::seal_order(rows, o1.sink, &keep);
        let epoch = o.epoch + 1;
        for (li, ks, vs, qs) in &got {
            let states = crate::qwen4_o1::far_states(nh, nkv, hd, o1, ks, vs, qs, rows, &order, keep.len());
            let views: Vec<_> = states.iter().map(|n| n.device_view()).collect();
            if !o1_views_valid_near(&views, nh, nkv, hd, crate::dory::MAX_NEAR)
                || o1_ensure_near(c, self.uid, *li, &views, epoch, crate::dory::MAX_NEAR).is_none()
            {
                return refuse(self, &format!("the far state of layer {li} does not fit the kernels"));
            }
        }
        // The exact caches leave the card; the cached bind groups held them.
        let cr = g.compress_ratio.max(1);
        for (li, _, _, _) in &got {
            if let Some(q) = self.layers[*li].qsa.as_mut() {
                *q = QsaDev::new(c, g, cr);
            }
            self.forget_layer(*li);
        }
        if let Some(o) = self.o1.as_mut() {
            o.sealed = true;
            o.epoch = epoch;
        }
        tracing::info!(
            "qwen4 O(1): sealed {rows} rows of {} attention layers, {} kept exact, in {:.2} s",
            got.len(),
            keep.len(),
            t0.elapsed().as_secs_f64()
        );
        true
    }

    /// A new sequence: the far states go, the exact caches come back.
    pub(crate) fn o1_reset(&mut self) {
        let Some(o) = self.o1.as_mut() else { return };
        if !o.sealed && !o.refused {
            return;
        }
        let g = &o.geom;
        let was_sealed = o.sealed;
        o.sealed = false;
        o.refused = false;
        if let Some(c) = ctx() {
            c.o1m.lock().unwrap().retain(|(id, _), _| *id != self.uid);
            if was_sealed {
                for l in &mut self.layers {
                    if let Some(q) = l.qsa.as_mut() {
                        *q = QsaDev::new(c, g, 4096);
                    }
                }
            }
        }
        for li in 0..self.layers.len() {
            self.forget_layer(li);
        }
    }

    fn forget_layer(&self, li: usize) {
        self.binds.borrow_mut().retain(|k, _| k.0 != li);
    }
}

// frame_buf tags for this stack (the DSV4 frames use 0..175; 200+ is ours)
const T_NORMED: u8 = 200;
const T_LOW: u8 = 201;
/// HC v3 stream-split shares: the A rows' `[t][s][row]`, the injection rows'.
const T_LOW4: u8 = 185;
const T_HCYB: u8 = 186;
const T_INJ_ATTN: u8 = 202;
const T_INJ_MLP: u8 = 203;
const T_X: u8 = 205;
const T_X2: u8 = 206;
const T_QKV: u8 = 207;
const T_Z: u8 = 208;
const T_A: u8 = 209;
const T_B: u8 = 210;
const T_CQ: u8 = 211;
const T_GDO: u8 = 212;
/// The staged GDN scan's prep rows of a prompt frame: q|k, k·q, (g, beta).
const T_GQK: u8 = 227;
const T_GKQ: u8 = 244;
const T_GGB: u8 = 247;
const T_BLK: u8 = 213;
const T_IQK: u8 = 214;
const T_QG: u8 = 215;
const T_K: u8 = 216;
const T_V: u8 = 217;
const T_IQ: u8 = 218;
const T_Q: u8 = 219;
const T_GATE: u8 = 220;
const T_SCORES: u8 = 221;
const T_PICK: u8 = 222;
const T_CNT: u8 = 223;
const T_IDX: u8 = 224;
const T_ATT: u8 = 225;
const T_LOGITS: u8 = 226;
const T_FORCED: u8 = 228;
const T_MSEL: u8 = 229;
const T_MWT: u8 = 230;
const T_MCNT: u8 = 231;
const T_MACT: u8 = 232;
const T_MO: u8 = 233;
/// Counting sort bins of frames past 4096 winners (`q4g_cs_*`).
const T_CSCNT: u8 = 21;
/// The prompt frame's activation scale and its partial maxima (`q82_t` on the matrix units).
const T_Q82ASC: u8 = 22;
const T_Q82AMP: u8 = 23;

/// `CMF_QWEN_Q82COOP=0`: prompt frames keep the scalar int8 GEMM for q8_2f projections.
fn q82_coop_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_QWEN_Q82COOP").as_deref() != Ok("0"))
}
const T_COLD: u8 = 234;
const T_REMAP: u8 = 235;
const T_COLDVEC: u8 = 236;
const T_EMB: u8 = 237;
const T_KEYRAW: u8 = 238;
const T_VAL: u8 = 239;
const T_KEYN: u8 = 240;
const T_QN: u8 = 241;
const T_GATED: u8 = 242;
const T_PNORM: u8 = 243;
const T_LMLOGITS: u8 = 245;
const T_HID: u8 = 246;
const T_CSEL: u8 = 248;
const T_CWT: u8 = 249;
const T_MACT2: u8 = 250;
const T_MOCOLD: u8 = 251;
const T_ZERO: u8 = 252;
const T_MIDS: u8 = 253;
const T_HINV: u8 = 199;
const T_QTAB: u8 = 198;
/// Split-key attention partials (`q4t_attend_sp`): weighted V rows, and (max, sum).
const T_ATSP_O: u8 = 192;
const T_ATSP_ML: u8 = 193;
/// The hyper-connection up-projection of a prompt frame past eight tokens.
const T_HCUP: u8 = 194;
/// Split-K partials of the f16 pair GEMM.
const T_PAIRK: u8 = 195;
/// A prompt frame's winners in arena-slot order, and their weighted down rows.
const T_ORD: u8 = 196;
const T_EPART: u8 = 197;
/// The sorted winners' segments per expert (`q4g_segs`): 4096 + a header.
const T_ESEG: u8 = 204;
const T_AMPV: u8 = 254;
const T_AMPI: u8 = 255;
const T_MEMB: u8 = 180;
const T_MEN: u8 = 181;
const T_ME: u8 = 182;
const T_MRN: u8 = 183;
const T_MRS0: u8 = 184; // ..187: one per stream
const T_MH0: u8 = 188; // ..191

// uniform slot tags (uni_slot keys: (tag, uid, li))
const U_PLE: u8 = 207;
const U_ROPE_IQ: u8 = 200;
const U_ROPE_QK: u8 = 201;
const U_KV: u8 = 202;
const U_IX: u8 = 203;
const U_TK: u8 = 204;
const U_IB: u8 = 205;
const U_QA: u8 = 206;
const U_BK: u8 = 208;

/// The indexer's top-k runs in one workgroup over at most this many
/// compressed blocks (the DSV4 kernel's table). With a compress ratio of 4
/// that is 16k positions; longer contexts stay on the host path for now.
/// Index blocks a token's scores row holds: 262 144 positions at a compression
/// of 4, the model's trained context (`q4t_topk_r`); the per-head path and the
/// old ranking kernel stop at `MAX_INDEX_BLOCKS_OLD`.
pub(crate) const MAX_INDEX_BLOCKS: usize = 65536;
pub(crate) const MAX_INDEX_BLOCKS_OLD: usize = 4096;
/// The sparse attention kernel's score table.
const MAX_ATTEND: usize = 2112;

/// `CMF_QWEN_QSA_T=0`: QSA token by token (offset-bound single-token
/// kernels) instead of the token-wide stages. The A/B arm.
/// `CMF_QWEN_ATTN_SPLIT=0`: the per-head QSA attention kernel instead of the
/// split-key GQA one (`q4t_attend_sp` + `q4t_attend_cmb`).
fn attend_split() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_ATTN_SPLIT").as_deref() != Ok("0"))
}

/// `CMF_QWEN_SORTED=0`: prompt frames past eight tokens keep the per-token
/// expert kernels instead of the arena-slot-sorted ones.
fn sorted_experts() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_SORTED").as_deref() != Ok("0"))
}

/// `CMF_QWEN_TOPK_R=0`: the indexer's per-block scores and quadratic top-k
/// instead of the lane-per-block scores and the radix top-k (whose block
/// table reaches the trained 262 144 positions).
/// `CMF_QWEN_QSA_ID=0`: rank and list the kept blocks even when every block
/// is kept (the identity list the split attention otherwise takes directly).
/// `CMF_QWEN_ATTN2=0`: the first split attention kernel (strided K reads).
fn attn_sp2() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_ATTN2").as_deref() != Ok("0"))
}

/// `CMF_QWEN_V4LOAD=0`: the decode expert kernels read their nibbles as u32s.
fn v4load() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_V4LOAD").as_deref() != Ok("0"))
}

/// Does a q4tp matrix take a whole number of 16-byte words (so bank slot i
/// starts on one: the vec4 views index from the slot base)?
fn q4tp_words16(rows: usize, cols: usize) -> bool {
    cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[rows, cols]).is_some_and(|n| n % 16 == 0)
}

fn qsa_identity() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_QSA_ID").as_deref() != Ok("0"))
}

fn topk_r() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_TOPK_R").as_deref() != Ok("0"))
}

fn index_blocks_cap() -> usize {
    if topk_r() { MAX_INDEX_BLOCKS } else { MAX_INDEX_BLOCKS_OLD }
}

fn qsa_tw() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_QSA_T").as_deref() != Ok("0"))
}

/// `CMF_QWEN_HC_FUSE=1`: fold the hyper-connection norm into the down and
/// up-fold kernels (two dispatches a mix instead of three). Measured 4%
/// slower on the RTX 4090 — every row workgroup recomputes the stream
/// norms — so the separate norm stays the default.
fn hc_fused() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("CMF_QWEN_HC_FUSE").as_deref() == Ok("1"))
}

/// Do the row-blocked expert kernels apply (gate/up in q2tp or q4tp, down in
/// q4tp, over an eight-segment arena, rows in fours)?
fn blocked_experts(g: &Geom, segments: usize) -> bool {
    segments == 8
        && g.inter % 4 == 0
        && g.hidden % 4 == 0
        && std::env::var("CMF_QWEN_EXPERT4").as_deref() != Ok("0")
}

/// Floats per snapshot slot of a GDN layer: the conv ring then S.
fn snap_stride(g: &Geom) -> usize {
    let cdim = 2 * g.gdn.nk * g.gdn.dk + g.gdn.nv * g.gdn.dv;
    (g.gdn.kk - 1) * cdim + g.gdn.nv * g.gdn.dk * g.gdn.dv
}

/// The readback stages' size at creation; `Dev::ensure_stage` grows them
/// when a forward needs more (a verify window's logits rows).
const STAGE_MIN: u64 = 4 << 20;

/// Bytes a readback stage is allocated with to hold `need`: whole MiB,
/// never below `STAGE_MIN`.
pub(crate) fn stage_size(need: u64) -> u64 {
    (need.div_ceil(1 << 20) * (1 << 20)).max(STAGE_MIN)
}

/// What one submitted chain of a forward copies into its stage, at most:
/// `layer_frames` layer frames of `frame_bytes` each (cold lists and MoE
/// inputs, 16-byte aligned), then `logit_rows` rows of the head's logits at
/// `logit_stride` bytes (one row for a plain token, every position in a
/// verify window). The layout `forward_tokens_device` encodes.
pub(crate) fn chain_stage_bytes(
    layer_frames: usize,
    frame_bytes: usize,
    logit_rows: usize,
    logit_stride: usize,
) -> u64 {
    (layer_frames * frame_bytes) as u64 + ((logit_rows * logit_stride) as u64).div_ceil(16) * 16
}

/// The row stride (bytes) of the logits `encode_head` leaves for the
/// vocabulary projection `lm_head` of `model`.
pub(crate) fn head_stride(model: &Arc<CmfModel>, lm_head: usize) -> Option<usize> {
    let e = model.tensors.get(lm_head)?;
    (e.shape.len() == 2).then(|| tstride(e.shape[0] * 4))
}

fn stage_buf(c: &Ctx, i: usize, bytes: u64) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(if i == 0 {
            "qwen4-stage-0"
        } else {
            "qwen4-stage-1"
        }),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn storage_buf(c: &Ctx, label: &str, bytes: u64) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl Dev {
    pub(crate) fn new(uid: u64, g: &Geom, kinds: &[(bool, bool)]) -> Option<Self> {
        let c = ctx()?;
        pipes(c)?;
        // one argument row per layer, one for the head
        let rows = kinds.len() + 1;
        let hh = g.hc * g.hidden;
        if (hh * 4) % 256 != 0 {
            return None;
        }
        // The workgroup tables the kernels size statically (and index without
        // runtime checks, see `qwen_checked`): a head of <= 256 lanes in the
        // RoPE / attention kernels, <= 1024 router scores, <= 64 winners, a
        // GQA group of at least one query head per KV head. Shapes outside
        // them were never computed right on the card; keep them on the host.
        if g.head_dim == 0
            || g.head_dim > 256
            || g.rotary_dim > g.head_dim
            || g.index_dim > 256
            || g.n_experts > 1024
            || g.top_k > 64
            || g.n_kv_heads == 0
            || g.n_heads < g.n_kv_heads
        {
            tracing::warn!(
                "qwen4 device path: shape outside the kernels' tables (head_dim {}, rotary {}, \
                 index_dim {}, experts {}, top_k {}, heads {}/{})",
                g.head_dim,
                g.rotary_dim,
                g.index_dim,
                g.n_experts,
                g.top_k,
                g.n_heads,
                g.n_kv_heads
            );
            return None;
        }
        let hyper = storage_buf(c, "qwen4-hyper", (TROWS * hh * 4) as u64);
        let inj_flags = storage_buf(c, "qwen4-inj-flags", 16);
        let mut layers = Vec::with_capacity(kinds.len());
        for &(is_gdn, has_ple) in kinds {
            let gdn = is_gdn.then(|| {
                let cdim = 2 * g.gdn.nk * g.gdn.dk + g.gdn.nv * g.gdn.dv;
                (
                    storage_buf(c, "qwen4-gdn-ring", ((g.gdn.kk - 1) * cdim * 4) as u64),
                    storage_buf(
                        c,
                        "qwen4-gdn-s",
                        (g.gdn.nv * g.gdn.dk * g.gdn.dv * 4) as u64,
                    ),
                )
            });
            let qsa = (!is_gdn).then(|| QsaDev::new(c, g, 4096));
            let ple = has_ple.then(|| {
                let cap = ((g.ple_kernel - 1) * g.ple_dilation).max(1);
                PleDev {
                    hist: storage_buf(c, "qwen4-ple-hist", (cap * g.hc * g.hidden * 4) as u64),
                    cap,
                    head: 0,
                    rows: 0,
                }
            });
            layers.push(LayerDev { gdn, qsa, ple });
        }
        Some(Self {
            uid,
            hyper,
            hh,
            inj_flags,
            layers,
            ixw: vec![(g.index_dim as f32).sqrt().recip(); g.index_heads],
            pos: 0,
            pending_inject: false,
            binds: std::cell::RefCell::new(HashMap::new()),
            stages: [stage_buf(c, 0, STAGE_MIN), stage_buf(c, 1, STAGE_MIN)],
            stage_ix: 0,
            cold_args: c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("qwen4-cold-args"),
                size: (TMAX * 32) as u64,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            fin_last: std::cell::Cell::new(None),
            cold_u: [0, 1].map(|_| {
                c.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("qwen4-cold-u"),
                    size: 32,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }),
            cold_tpl: std::cell::Cell::new(None),
            cold_k: std::cell::Cell::new(0),
            args_tpl: c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("qwen4-args-tpl"),
                size: (rows * SLOTS * 16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            args_live: c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("qwen4-args-live"),
                size: (rows * SLOTS * 16) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tpl: std::cell::RefCell::new(vec![0u32; rows * SLOTS * 4]),
            rows,
            miss: c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("qwen4-miss"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            ple_touched: vec![0; kinds.len()],
            snaps: vec![None; kinds.len()],
            ple_snaps: vec![Vec::new(); kinds.len()],
            r_last: storage_buf(c, "qwen4-r-last", (g.hc * g.hidden * 4) as u64),
            gated: false,
            o1: None,
            tubes: Vec::new(),
            tube_act: None,
            tubes_shared: Vec::new(),
            tube_act_s: None,
            sbank: Vec::new(),
            route_cap: None,
            masked: false,
        })
    }

    /// Start a new sequence: zero the recurrent GDN state and the PLE
    /// history, forget the caches' fill (they are overwritten in place).
    pub(crate) fn reset(&mut self) -> bool {
        let Some(c) = ctx() else { return false };
        self.o1_reset();
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qwen4-reset"),
            });
        for l in &mut self.layers {
            if let Some((ring, s)) = &l.gdn {
                enc.clear_buffer(ring, 0, None);
                enc.clear_buffer(s, 0, None);
            }
            if let Some(p) = &mut l.ple {
                p.head = 0;
                p.rows = 0;
            }
        }
        submit(c, enc.finish());
        self.pos = 0;
        self.pending_inject = false;
        true
    }

    /// Seed token slot `tok`'s hyper state with its embedding on every stream.
    pub(crate) fn seed(&self, tok: usize, emb: &[f32], hc: usize) {
        let Some(c) = ctx() else { return };
        let mut all = Vec::with_capacity(emb.len() * hc);
        for _ in 0..hc {
            all.extend_from_slice(emb);
        }
        c.queue.write_buffer(
            &self.hyper,
            (tok * self.hh * 4) as u64,
            bytemuck::cast_slice(&all),
        );
    }

    /// Drop the deferred injection (a token nobody reads the logits of).
    pub(crate) fn clear_pending(&mut self) {
        self.pending_inject = false;
    }

    /// Room in the streamed prompt's bank for `frames` frames of TROWS rows.
    pub(crate) fn stream_bank(&mut self, g: &Geom, frames: usize) -> bool {
        let Some(c) = ctx() else { return false };
        while self.sbank.len() < frames {
            self.sbank.push([
                storage_buf(c, "qwen4-sbank-hyper", (TROWS * self.hh * 4) as u64),
                storage_buf(c, "qwen4-sbank-mo", (TROWS * tstride(g.hidden * 4)) as u64),
                storage_buf(c, "qwen4-sbank-inj", (TROWS * tstride(g.hc * 4)) as u64),
            ]);
        }
        true
    }

    /// Keep the routing to the arena's experts (or not). The route kernels take the flag in
    /// a uniform their cached bind groups hold, so a change drops the bind cache.
    pub(crate) fn set_masked(&mut self, on: bool) {
        if self.masked != on {
            self.masked = on;
            self.binds.borrow_mut().clear();
        }
    }

    /// A routing capture of at least `bytes` (grow-only).
    pub(crate) fn route_cap_ensure(&mut self, bytes: u64) -> bool {
        let Some(c) = ctx() else { return false };
        if self.route_cap.as_ref().is_some_and(|b| b.size() >= bytes) {
            return true;
        }
        self.route_cap = Some(c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen4-route-cap"),
            size: bytes.max(16).div_ceil(16) * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        true
    }

    /// The frame's routing rows (`nt` tokens of `cold_stride` bytes: cold list, then
    /// every winner and its weight) into the capture at byte `off`.
    pub(crate) fn route_cap_copy(&self, enc: &mut wgpu::CommandEncoder, g: &Geom, off: u64, nt: usize) {
        let (Some(c), Some(cap)) = (ctx(), self.route_cap.as_ref()) else { return };
        let cold = tbuf(c, T_COLD, 4 * g.top_k * 4, false);
        let bytes = (nt * cold_stride(g)) as u64;
        if off + bytes <= cap.size() {
            flush_pass(&*enc);
            enc.copy_buffer_to_buffer(&cold, 0, cap, off, bytes);
        }
    }

    /// The capture's first `bytes`, once the card has written them.
    pub(crate) fn route_cap_read(&self, bytes: u64) -> Option<Vec<u8>> {
        let c = ctx()?;
        let cap = self.route_cap.as_ref()?;
        let bytes = bytes.min(cap.size()).div_ceil(16) * 16;
        let slice = cap.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        if !rx.recv().ok()? {
            return None;
        }
        let out = slice.get_mapped_range().ok()?.to_vec();
        cap.unmap();
        Some(out)
    }

    /// Seed token `tok` of the streamed chunk's frame `k` with its embedding on every
    /// stream (as `seed`, into the bank).
    pub(crate) fn seed_bank(&self, k: usize, tok: usize, emb: &[f32], hc: usize) {
        let (Some(c), Some(b)) = (ctx(), self.sbank.get(k)) else { return };
        let mut all = Vec::with_capacity(emb.len() * hc);
        for _ in 0..hc {
            all.extend_from_slice(emb);
        }
        c.queue.write_buffer(&b[0], (tok * self.hh * 4) as u64, bytemuck::cast_slice(&all));
    }

    /// Frame `k` (`nt` rows) of a streamed chunk between the bank and the frame buffers:
    /// `save` parks the live hyper rows (and, with `moe`, the MoE output and its inject
    /// weights the next layer adds), else the parked state becomes live.
    pub(crate) fn stream_swap(&self, enc: &mut wgpu::CommandEncoder, g: &Geom, k: usize, nt: usize, save: bool, moe: bool) -> bool {
        let (Some(c), Some(b)) = (ctx(), self.sbank.get(k)) else { return false };
        flush_pass(&*enc);
        let mo = tbuf(c, T_MO, g.hidden * 4, true);
        let inj = tbuf(c, T_INJ_MLP, g.hc * 4, true);
        let parts = [
            (&self.hyper, &b[0], nt * self.hh * 4),
            (&mo, &b[1], nt * tstride(g.hidden * 4)),
            (&inj, &b[2], nt * tstride(g.hc * 4)),
        ];
        for (i, (live, bank, bytes)) in parts.into_iter().enumerate() {
            if i > 0 && !moe {
                break;
            }
            if save {
                enc.copy_buffer_to_buffer(live, 0, bank, 0, bytes as u64);
            } else {
                enc.copy_buffer_to_buffer(bank, 0, live, 0, bytes as u64);
            }
        }
        true
    }
}

// ── small encoders ──

/// Where a dispatch's bind group is cached: the layer and a step id that is
/// fixed per dispatch site (never a running counter — the frame has
/// position-dependent branches).
#[derive(Clone, Copy)]
struct Bc<'a> {
    dev: &'a Dev,
    li: usize,
    /// The token slot of the frame this section works on.
    tok: usize,
    /// Dispatch directly instead of through the gated argument row: the
    /// chain head's pending inject, which always runs.
    direct: bool,
}

impl Bc<'_> {
    fn get<F: FnOnce() -> wgpu::BindGroup>(&self, step: u16, build: F) -> wgpu::BindGroup {
        self.dev.bind(self.li, step, self.tok, build)
    }

    /// The uniform-slot key of this section: layer and token slot.
    fn key(&self) -> usize {
        // token slots up to TW + TROWS + TMAX per layer
        self.li.wrapping_mul(1024) + self.tok
    }

    /// Record the dispatch size in the layer's template row and dispatch
    /// indirectly from the live row the gate kernel fills.
    fn launch(&self, pass: &mut wgpu::ComputePass<'_>, step: u16, groups: (u32, u32, u32)) {
        if step_probe(step, self.li, groups) {
            return;
        }
        if self.direct || !self.dev.gated {
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            return;
        }
        let row = self.dev.row(self.li);
        let slot = step_slot(step);
        let i = (row * SLOTS + slot) * 4;
        {
            let mut t = self.dev.tpl.borrow_mut();
            t[i] = groups.0;
            t[i + 1] = groups.1;
            t[i + 2] = groups.2;
            t[i + 3] = 0;
        }
        pass.dispatch_workgroups_indirect(&self.dev.args_live, ((row * SLOTS + slot) * 16) as u64);
    }
}

impl Dev {
    /// The argument row of a frame: layers by index, the head after them.
    fn row(&self, li: usize) -> usize {
        if li >= self.rows { self.rows - 1 } else { li }
    }

    /// Upload the template rows `lo..hi` (frames of the chain about to be
    /// submitted) and clear the miss flag.
    pub(crate) fn arm_chain(&self, lo: usize, hi: usize) {
        if !self.gated {
            return;
        }
        let Some(c) = ctx() else { return };
        let (lo, hi) = (self.row(lo), self.row(hi.saturating_sub(1)) + 1);
        let t = self.tpl.borrow();
        let a = lo * SLOTS * 4;
        let b = hi * SLOTS * 4;
        c.queue.write_buffer(
            &self.args_tpl,
            (lo * SLOTS * 16) as u64,
            bytemuck::cast_slice(&t[a..b]),
        );
        c.queue.write_buffer(&self.miss, 0, &[0u8; 16]);
    }

    /// The frame `li` ran for `n` tokens: advance the host-side bookkeeping
    /// it deferred.
    pub(crate) fn commit(&mut self, li: usize) {
        if li >= self.ple_touched.len() {
            return;
        }
        let n = std::mem::take(&mut self.ple_touched[li]);
        if let Some(pd) = self.layers[li].ple.as_mut() {
            for _ in 0..n {
                pd.head = (pd.head + 1) % pd.cap;
                pd.rows = (pd.rows + 1).min(pd.cap);
            }
        }
    }

    /// The frames `lo..hi` were encoded but did not run; forget their
    /// deferred bookkeeping.
    pub(crate) fn discard(&mut self, lo: usize, hi: usize) {
        for li in lo..hi.min(self.ple_touched.len()) {
            self.ple_touched[li] = 0;
        }
    }

    pub(crate) fn stage(&self) -> &wgpu::Buffer {
        &self.stages[self.stage_ix]
    }

    /// Token slot `tok`'s row of the hyper state.
    pub(crate) fn hyper(&self, tok: usize) -> Rng {
        Rng::row(&self.hyper, tok, self.hh * 4)
    }

    /// Remember slot `tok`'s hyper state as the last position's R.
    pub(crate) fn keep_r(&self, enc: &mut wgpu::CommandEncoder, tok: usize) {
        flush_pass(&*enc);
        enc.copy_buffer_to_buffer(
            &self.hyper,
            (tok * self.hh * 4) as u64,
            &self.r_last,
            0,
            self.r_last.size(),
        );
    }

    /// Make both readback stages hold at least `bytes`. Every copy of a
    /// frame names the stage it was encoded against, so a forward calls this
    /// before it encodes its first frame; a stage a finished readback still
    /// holds stays alive with it. The stages start at `STAGE_MIN` (4 MiB),
    /// which holds four rows of a 248k vocabulary's logits and no more: a
    /// verify window of five or more positions (MTP k >= 4) needs larger
    /// ones. False when `bytes` exceeds the device's buffer limit.
    pub(crate) fn ensure_stage(&mut self, bytes: u64) -> bool {
        let Some(c) = ctx() else { return false };
        let have = self.stages[0].size().min(self.stages[1].size());
        if bytes <= have {
            return true;
        }
        let size = stage_size(bytes);
        if size > c.device.limits().max_buffer_size {
            return false;
        }
        self.stages = [stage_buf(c, 0, size), stage_buf(c, 1, size)];
        true
    }

    /// VRAM of `rows` verify-window snapshot rows over layers of `kinds`
    /// (`(is_gdn, has_ple)`, as `Dev::new` takes them): what
    /// `ensure_snaps(g, rows)` allocates.
    pub(crate) fn snap_bytes(g: &Geom, kinds: &[(bool, bool)], rows: usize) -> u64 {
        let ple_cap = ((g.ple_kernel.max(1) - 1) * g.ple_dilation).max(1);
        kinds
            .iter()
            .map(|&(is_gdn, has_ple)| {
                let gdn = if is_gdn { snap_stride(g) * 4 } else { 0 };
                let ple = if has_ple {
                    ple_cap * g.hc * g.hidden * 4
                } else {
                    0
                };
                (rows * (gdn + ple)) as u64
            })
            .sum()
    }

    /// Make sure `nt` snapshot slots exist for every recurrent layer.
    pub(crate) fn ensure_snaps(&mut self, g: &Geom, nt: usize) {
        let Some(c) = ctx() else { return };
        let stride = snap_stride(g);
        for (li, l) in self.layers.iter().enumerate() {
            if l.gdn.is_some() && self.snaps[li].as_ref().is_none_or(|(_, rows)| *rows < nt) {
                self.snaps[li] = Some((
                    storage_buf(c, "qwen4-snap-gdn", (nt * stride * 4) as u64),
                    nt,
                ));
                // the GDN kernels' cached groups of a smaller window name
                // the replaced buffer: their snapshots would land there
                // while `restore` reads this one
                self.forget_layer(li);
            }
            if let Some(pd) = l.ple.as_ref() {
                while self.ple_snaps[li].len() < nt {
                    self.ple_snaps[li].push((
                        storage_buf(c, "qwen4-snap-ple", (pd.cap * g.hc * g.hidden * 4) as u64),
                        0,
                        0,
                    ));
                }
            }
        }
    }

    /// Roll the recurrent state back to what it was before token slot
    /// `tok` of the last frame (its snapshot), on every layer. The PLE
    /// history and counters follow. QSA caches need nothing: rejected rows
    /// are overwritten when their positions are processed again.
    pub(crate) fn restore(&mut self, g: &Geom, tok: usize) -> bool {
        let Some(c) = ctx() else { return false };
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qwen4-restore"),
            });
        let hh = g.hc * g.hidden;
        let stride = snap_stride(g);
        let ring_els = (g.gdn.kk - 1) * (2 * g.gdn.nk * g.gdn.dk + g.gdn.nv * g.gdn.dv);
        for li in 0..self.layers.len() {
            // the state before slot `tok` is the one slot `tok - 1` left
            if let (Some((ring, st)), Some((snap, rows))) =
                (self.layers[li].gdn.as_ref(), self.snaps[li].as_ref())
                && tok >= 1
                && tok - 1 < *rows
            {
                let base = ((tok - 1) * stride * 4) as u64;
                enc.copy_buffer_to_buffer(snap, base, ring, 0, ring.size());
                enc.copy_buffer_to_buffer(snap, base + (ring_els * 4) as u64, st, 0, st.size());
            }
            let snap = self.ple_snaps[li]
                .get(tok)
                .map(|(b, h, r)| (b.clone(), *h, *r));
            if let (Some(pd), Some((sb, head, rows))) = (self.layers[li].ple.as_mut(), snap) {
                enc.copy_buffer_to_buffer(&sb, 0, &pd.hist, 0, (pd.cap * hh * 4) as u64);
                pd.head = head;
                pd.rows = rows;
            }
        }
        submit(c, enc.finish());
        true
    }
}

/// The layer's gate: fill its live argument row from the template, or with
/// zeros once a miss happened earlier in the chain. Dispatched directly.
fn encode_gate(c: &Ctx, p: &Pipes, pass: &mut wgpu::ComputePass<'_>, dev: &Dev, li: usize) {
    if !dev.gated {
        return;
    }
    let row = dev.row(li) as u32;
    let bind = dev.bind(li, 9, 0, || {
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qwen4-gate"),
            layout: &p.gate.get_bind_group_layout(0),
            entries: &[
                bind_buf(0, &dev.args_tpl),
                bind_buf(1, &dev.args_live),
                bind_buf(2, &dev.miss),
                bind_buf(3, &uniform_u32x4(c, [row, SLOTS as u32, 0, 0])),
            ],
        })
    });
    pass.set_pipeline(&p.gate);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((SLOTS as u32).div_ceil(64), 1, 1);
}

const LI_PENDING: usize = usize::MAX;
const LI_MTP_IN: usize = usize::MAX - 64;

struct WeightRef {
    buf: wgpu::Buffer,
    dtype: TensorDtype,
    rows: usize,
    cols: usize,
}

fn weight(c: &Ctx, model: &Arc<CmfModel>, idx: usize) -> Option<WeightRef> {
    let e = model.tensors.get(idx)?;
    if e.shape.len() != 2 {
        return None;
    }
    let abs = model.entry_abs_offset(e)?;
    let plen = e.nbytes as usize;
    let bytes = model.primary_bytes();
    let slice = bytes.get(abs..abs + plen)?;
    let buf = weight_buffer_l(
        c,
        (model.uid() as usize, idx),
        slice,
        layer_of_name(&e.name),
    )?;
    Some(WeightRef {
        buf,
        dtype: e.dtype,
        rows: e.shape[0],
        cols: e.shape[1],
    })
}

/// Upload every tensor of the list to the arena (so the budget the expert
/// bank sees afterwards is the real one) and pin them against eviction.
pub(crate) fn prewarm(model: &Arc<CmfModel>, idxs: &[usize]) -> Option<u64> {
    let c = ctx()?;
    let mut bytes = 0u64;
    for &i in idxs {
        let w = weight(c, model, i)?;
        bytes += w.buf.size();
    }
    pin_weights(model, idxs);
    Some(bytes)
}

/// y = W·x for a q8_2f / f16 / f32 matrix resident on the card.
#[allow(clippy::too_many_arguments)]
fn encode_mv(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    w: &WeightRef,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    bc: Bc,
    step: u16,
) -> bool {
    match w.dtype {
        TensorDtype::Q8_2f => {
            let (pipe, per_wg) = q82_pipe(c, w.cols);
            let bind = bc.get(step, || {
                let pb = uniform_u32x4(c, [(w.cols / 4) as u32, w.rows as u32, w.cols as u32, 0]);
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-q82"),
                    layout: &pipe.get_bind_group_layout(0),
                    entries: &[
                        bind_buf(0, &w.buf),
                        bind_buf(1, xs),
                        bind_buf(2, y),
                        bind_buf(3, &pb),
                    ],
                })
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bind, &[]);
            bc.launch(
                pass,
                step,
                ((w.rows as u32).div_ceil(per_wg).min(MAX_WG), 1, 1),
            );
            true
        }
        TensorDtype::F16 if w.cols % 2 == 0 && w.rows <= MAX_WG as usize => {
            let bind = bc.get(step, || {
                let pb = uniform_u32x4(c, [w.cols as u32, w.rows as u32, 0, 0]);
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-f16"),
                    layout: &p.f16_matvec.get_bind_group_layout(0),
                    entries: &[
                        bind_buf(0, &w.buf),
                        bind_buf(1, xs),
                        bind_buf(2, y),
                        bind_buf(3, &pb),
                    ],
                })
            });
            pass.set_pipeline(&p.f16_matvec);
            pass.set_bind_group(0, &bind, &[]);
            bc.launch(pass, step, (w.rows as u32, 1, 1));
            true
        }
        TensorDtype::F32 if w.rows <= MAX_WG as usize => {
            let bind = bc.get(step, || {
                let pb = uniform_u32x4(c, [w.cols as u32, w.rows as u32, 0, 0]);
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-f32"),
                    layout: &c.layout_f32,
                    entries: &[
                        bind_buf(0, &w.buf),
                        bind_buf(1, xs),
                        bind_buf(2, y),
                        bind_buf(3, &pb),
                    ],
                })
            });
            pass.set_pipeline(&c.f32_matvec);
            pass.set_bind_group(0, &bind, &[]);
            bc.launch(pass, step, (w.rows as u32, 1, 1));
            true
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn mv(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    bc: Bc,
    step: u16,
) -> Option<WeightRef> {
    let w = weight(c, model, idx)?;
    encode_mv(c, p, pass, &w, xs, y, bc, step).then_some(w)
}

#[allow(clippy::too_many_arguments)]
fn group_rmsnorm(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    o: &wgpu::Buffer,
    groups: usize,
    n: usize,
    eps: f32,
    bc: Bc,
    step: u16,
) {
    let bind = bc.get(step, || {
        let pb = uniform_u32x4(c, [groups as u32, n as u32, eps.to_bits(), 0]);
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qwen4-gn"),
            layout: &p.group_rmsnorm.get_bind_group_layout(0),
            entries: &[
                bind_buf(0, x),
                bind_buf(1, w),
                bind_buf(2, o),
                bind_buf(3, &pb),
            ],
        })
    });
    pass.set_pipeline(&p.group_rmsnorm);
    pass.set_bind_group(0, &bind, &[]);
    bc.launch(pass, step, (groups as u32, 1, 1));
}

/// One dispatch with a cached bind group: `entries` built only on a miss.
#[allow(clippy::too_many_arguments)]
fn dispatch<F>(
    c: &Ctx,
    pass: &mut wgpu::ComputePass<'_>,
    pipe: &wgpu::ComputePipeline,
    bc: Bc,
    step: u16,
    label: &'static str,
    entries: F,
    groups: (u32, u32, u32),
) where
    F: FnOnce() -> Vec<wgpu::Buffer>,
{
    let bind = bc.get(step, || {
        let bufs = entries();
        let ents: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| bind_buf(i as u32, b))
            .collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipe.get_bind_group_layout(0),
            entries: &ents,
        })
    });
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bind, &[]);
    bc.launch(pass, step, groups);
}

// ── token-wide frame layout ──

/// Token index of the token-strided frame buffers in the frame pool (the
/// per-token scratch of the PLE and MTP-input paths keeps 0..TMAX).
const TROW: usize = 1000;
/// `Bc.tok` of a token-wide dispatch site: its bind group depends on the
/// token count (through the uniform), never on a token slot.
const TW: usize = 64;

/// Bytes per token row of a strided frame buffer: the row rounded up to
/// the storage offset alignment, so a single-token kernel can bind one row.
pub(crate) fn tstride(len: usize) -> usize {
    len.div_ceil(256) * 256
}

/// Elements per token row of a strided buffer holding `len` bytes a row.
fn es(len: usize) -> u32 {
    (tstride(len) / 4) as u32
}

/// A frame buffer of TMAX token rows at `tstride(len)`.
fn tbuf(c: &Ctx, tag: u8, len: usize, upload: bool) -> wgpu::Buffer {
    frame_buf_t(c, tag, TROW, TROWS * tstride(len), upload)
}

/// A frame buffer of TMAX token rows packed at exactly `len`: buffers only
/// token-wide kernels touch, whose row stride the kernel itself fixes.
fn tbuf_exact(c: &Ctx, tag: u8, len: usize, upload: bool) -> wgpu::Buffer {
    frame_buf_t(c, tag, TROW, TROWS * len, upload)
}

/// A buffer nothing ever writes (zeros), for bindings a flag turns off.
fn zero_buf(c: &Ctx) -> wgpu::Buffer {
    frame_buf_t(c, T_ZERO, TROW, 4096, false)
}

/// A whole buffer or one token row of a strided buffer, as a bind entry.
#[derive(Clone)]
pub(crate) struct Rng {
    buf: wgpu::Buffer,
    off: u64,
    len: u64,
}

impl Rng {
    fn all(b: &wgpu::Buffer) -> Self {
        Rng {
            buf: b.clone(),
            off: 0,
            len: 0,
        }
    }
    fn row(b: &wgpu::Buffer, t: usize, len: usize) -> Self {
        Rng {
            buf: b.clone(),
            off: (t * tstride(len)) as u64,
            len: len as u64,
        }
    }
    fn entry(&self, i: u32) -> wgpu::BindGroupEntry<'_> {
        if self.len == 0 {
            bind_buf(i, &self.buf)
        } else {
            bind_buf_off(i, &self.buf, self.off, self.len)
        }
    }
}

fn uniform_u32x16(c: &Ctx, v: [u32; 16]) -> wgpu::Buffer {
    let mut u = c.qwen4_uni16.lock().unwrap();
    if let Some(b) = u.get(&v) {
        return b.clone();
    }
    let b = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    u.insert(v, b.clone());
    b
}

/// `dispatch` with ranged entries (token rows of strided buffers).
#[allow(clippy::too_many_arguments)]
fn dispatch_r<F>(
    c: &Ctx,
    pass: &mut wgpu::ComputePass<'_>,
    pipe: &wgpu::ComputePipeline,
    bc: Bc,
    step: u16,
    label: &'static str,
    entries: F,
    groups: (u32, u32, u32),
) where
    F: FnOnce() -> Vec<Rng>,
{
    let bind = bc.get(step, || {
        let rs = entries();
        let ents: Vec<_> = rs
            .iter()
            .enumerate()
            .map(|(i, r)| r.entry(i as u32))
            .collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipe.get_bind_group_layout(0),
            entries: &ents,
        })
    });
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, &bind, &[]);
    bc.launch(pass, step, groups);
}

// ── token-wide encoders ──

/// Group RMSNorm of `nt` rows: `(1 + w)` scaled, `groups` of `n`.
#[allow(clippy::too_many_arguments)]
fn gn_t(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    x: &wgpu::Buffer,
    xs: u32,
    w: &wgpu::Buffer,
    o: &wgpu::Buffer,
    os: u32,
    groups: usize,
    n: usize,
    eps: f32,
    nt: usize,
    bc: Bc,
    step: u16,
) {
    dispatch(
        c,
        pass,
        &p.t_group_rmsnorm,
        bc,
        step,
        "qwen4t-gn",
        || {
            vec![
                x.clone(),
                w.clone(),
                o.clone(),
                uniform_u32x8(c, [groups as u32, n as u32, eps.to_bits(), xs, os, 0, 0, 0]),
            ]
        },
        (groups as u32, nt as u32, 1),
    );
}

/// Two f16 matrices over `nt` inputs in one dispatch (`q4t_f16_pair`):
/// A's rows to `ya` (silu·inv with act bit 1), B's to `yb[yb_off..]`
/// (sigmoid with act bit 2). None when a matrix is not f16.
#[allow(clippy::too_many_arguments)]
fn pair_t(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    model: &Arc<CmfModel>,
    idx_a: usize,
    idx_b: Option<usize>,
    x: &wgpu::Buffer,
    xs: u32,
    ya: &wgpu::Buffer,
    yas: u32,
    yb: &wgpu::Buffer,
    ybs: u32,
    act: u32,
    inv: f32,
    yb_off: usize,
    nt: usize,
    bc: Bc,
    step: u16,
) -> Option<()> {
    let a = weight(c, model, idx_a)?;
    let b = match idx_b {
        Some(i) => Some(weight(c, model, i)?),
        None => None,
    };
    pair_w(
        c,
        p,
        pass,
        &a,
        b.as_ref(),
        x,
        xs,
        ya,
        yas,
        yb,
        ybs,
        act,
        inv,
        yb_off,
        nt,
        bc,
        step,
    )
}

/// `pair_t` on matrices already resident.
#[allow(clippy::too_many_arguments)]
fn pair_w(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    a: &WeightRef,
    b: Option<&WeightRef>,
    x: &wgpu::Buffer,
    xs: u32,
    ya: &wgpu::Buffer,
    yas: u32,
    yb: &wgpu::Buffer,
    ybs: u32,
    act: u32,
    inv: f32,
    yb_off: usize,
    nt: usize,
    bc: Bc,
    step: u16,
) -> Option<()> {
    if a.dtype != TensorDtype::F16
        || a.cols % 2 != 0
        || b.is_some_and(|b| b.dtype != TensorDtype::F16 || b.cols != a.cols)
    {
        return None;
    }
    let rows_b = b.map_or(0, |b| b.rows);
    let total = a.rows + rows_b;
    if total > MAX_WG as usize {
        return None;
    }
    let bbuf = b.map_or_else(|| a.buf.clone(), |b| b.buf.clone());
    if nt > TMAX {
        // a prompt frame past eight tokens: the tiled GEMM, same epilogue.
        // Few rows over a long row (the hyper-connection's 324 x 10240) split
        // the columns so the card gets enough workgroups; the slices' sums go
        // through `q4t_pair_sum` in slice order.
        let row_tiles = total.div_ceil(64);
        // the matrix-unit twin takes 64-token tiles and 32-column slices
        let coop = p.pair_coop.as_ref().filter(|_| a.cols % 32 == 0 && xs % 4 == 0);
        let tok_tile = if coop.is_some() { 64 } else { 16 };
        let ks = if row_tiles * nt.div_ceil(tok_tile) < 256 { a.cols.div_ceil(1024).clamp(1, 16) } else { 1 };
        let kc = a.cols.div_ceil(ks).div_ceil(64) * 64;
        let part = if ks > 1 {
            frame_buf_t(c, T_PAIRK, 0, (ks * nt * total * 4).max(4), false)
        } else {
            zero_buf(c)
        };
        let u = uniform_u32x16(
            c,
            [
                a.cols as u32,
                a.rows as u32,
                rows_b as u32,
                act,
                inv.to_bits(),
                yb_off as u32,
                nt as u32,
                xs,
                yas,
                ybs,
                kc as u32,
                ks as u32,
                0,
                0,
                0,
                0,
            ],
        );
        let u2 = u.clone();
        let part2 = part.clone();
        dispatch(
            c,
            pass,
            coop.unwrap_or(&p.t_f16_pair_mm),
            bc,
            step,
            "qwen4t-pair-mm",
            || vec![a.buf.clone(), bbuf, x.clone(), ya.clone(), yb.clone(), u, part],
            (row_tiles as u32, (nt as u32).div_ceil(tok_tile as u32), ks as u32),
        );
        if ks > 1 {
            dispatch(
                c,
                pass,
                &p.t_pair_sum,
                bc,
                step + 20,
                "qwen4t-pair-sum",
                || vec![part2, ya.clone(), yb.clone(), u2],
                ((total as u32).div_ceil(256), nt as u32, 1),
            );
        }
        return Some(());
    }
    if nt == 1 && pair1_env() {
        // one token: the single-row pair kernel (row t = 0 of q4t_f16_pair
        // is its arithmetic bit for bit); the bind group is cached apart
        // from the frame's (its key carries nt)
        dispatch(
            c,
            pass,
            &p.f16_matvec2,
            bc,
            step,
            "qwen4-pair1",
            || {
                vec![
                    a.buf.clone(),
                    bbuf,
                    x.clone(),
                    ya.clone(),
                    yb.clone(),
                    uniform_u32x8(
                        c,
                        [
                            a.cols as u32,
                            a.rows as u32,
                            rows_b as u32,
                            act,
                            inv.to_bits(),
                            yb_off as u32,
                            0,
                            0,
                        ],
                    ),
                ]
            },
            (total as u32, 1, 1),
        );
        return Some(());
    }
    dispatch(
        c,
        pass,
        &p.t_f16_pair,
        bc,
        step,
        "qwen4t-pair",
        || {
            vec![
                a.buf.clone(),
                bbuf,
                x.clone(),
                ya.clone(),
                yb.clone(),
                uniform_u32x16(
                    c,
                    [
                        a.cols as u32,
                        a.rows as u32,
                        rows_b as u32,
                        act,
                        inv.to_bits(),
                        yb_off as u32,
                        nt as u32,
                        xs,
                        yas,
                        ybs,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                ),
            ]
        },
        (total as u32, 1, 1),
    );
    Some(())
}

/// A q8_2f matrix over `nt` inputs (`q4t_q82_matvec`); one token takes
/// the tuned single-row kernels as they are (row 0 sits at offset 0).
#[allow(clippy::too_many_arguments)]
fn q82_t(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    w: &WeightRef,
    x: &wgpu::Buffer,
    xs: u32,
    y: &wgpu::Buffer,
    ys: u32,
    nt: usize,
    bc: Bc,
    step: u16,
) -> bool {
    if nt == 1 {
        // long rows: one row a workgroup
        let long = w.cols >= 4096 && w.rows % 2 == 0;
        let sg_pick = if long { p.q82sg1.as_ref().or(p.q82sg.as_ref()) } else { p.q82sg.as_ref() };
        let one_row = long && p.q82sg1.is_some();
        if let Some(sg) = sg_pick
            && w.dtype == TensorDtype::Q8_2f
            && w.cols % 16 == 0
            && c.use_q82_mv4
        {
            // the bits of `q8_2f_matvec4`, which `encode_mv` picks for
            // these shapes, with the subgroup tail
            dispatch(
                c,
                pass,
                sg,
                bc,
                step,
                "qwen4-q82sg",
                || {
                    vec![
                        w.buf.clone(),
                        x.clone(),
                        y.clone(),
                        uniform_u32x8(
                            c,
                            [
                                (w.cols / 4) as u32,
                                w.rows as u32,
                                w.cols as u32,
                                1,
                                0,
                                0,
                                0,
                                0,
                            ],
                        ),
                    ]
                },
                ((w.rows as u32).div_ceil(if one_row { 1 } else { 4 }).min(MAX_WG), 1, 1),
            );
            return true;
        }
        return encode_mv(c, p, pass, w, x, y, bc, step);
    }
    if w.dtype != TensorDtype::Q8_2f || w.cols % 16 != 0 || xs % 4 != 0 {
        return false;
    }
    if nt > TMAX {
        // a prompt frame past eight tokens on the matrix units: the eight-bit prompt
        // GEMM of the dense models (`q82_mm_coop_s`: f16 operands, f32 sums), the
        // activations scaled by their absolute maximum (reduced on the card first).
        // The scalar int8 GEMM below held ~18 TFLOP/s on an RTX 5090, the GDN and QSA
        // projections half of a resident prompt frame. `CMF_QWEN_Q82COOP=0` keeps it.
        if xs as usize == w.cols
            && ys as usize == w.rows
            && w.cols % 32 == 0
            && w.rows >= 256
            && q82_coop_on()
            && let (Some(mm), Some(part), Some(fold)) = (c.q82_mm_coop_s.as_ref(), c.act_amax_part.as_ref(), c.act_amax_fold.as_ref())
        {
            let asc = frame_buf_t(c, T_Q82ASC, 0, 16, false);
            let parts = frame_buf_t(c, T_Q82AMP, 0, 512 * 4, false);
            let n = nt * w.cols;
            dispatch(
                c,
                pass,
                part,
                bc,
                step + 20000,
                "qwen4t-amax-part",
                || vec![x.clone(), parts.clone(), uniform_u32x4(c, [n as u32, 0, 0, 0])],
                (512, 1, 1),
            );
            dispatch(
                c,
                pass,
                fold,
                bc,
                step + 21000,
                "qwen4t-amax-fold",
                || vec![parts.clone(), asc.clone(), uniform_u32x4(c, [512, 0, 0, 0])],
                (1, 1, 1),
            );
            dispatch(
                c,
                pass,
                mm,
                bc,
                step + 22000,
                "qwen4t-q82-coop",
                || {
                    vec![
                        w.buf.clone(),
                        x.clone(),
                        y.clone(),
                        uniform_u32x4(c, [(w.cols / 4) as u32, w.rows as u32, nt as u32, 0xFFFF_FFFF]),
                        asc.clone(),
                    ]
                },
                ((w.rows as u32).div_ceil(64).min(MAX_WG), (nt as u32).div_ceil(64), 1),
            );
            return true;
        }
        // the batch graph's scalar q8_2f GEMM (`q8_2f_mm_r2`, 64 x 64 tiles), rows
        // packed at their width
        let Some((mm, _)) = super::gemma_batch::pipes(c).and_then(|g| g.mm.as_ref()) else {
            return false;
        };
        if xs as usize != w.cols || ys as usize != w.rows || w.cols % 64 != 0 {
            return false;
        }
        dispatch(
            c,
            pass,
            mm,
            bc,
            step,
            "qwen4t-q82-mm",
            || {
                vec![
                    w.buf.clone(),
                    x.clone(),
                    y.clone(),
                    uniform_u32x4(c, [(w.cols / 4) as u32, w.rows as u32, nt as u32, w.cols as u32]),
                ]
            },
            ((w.rows as u32).div_ceil(64), (nt as u32).div_ceil(64), 1),
        );
        return true;
    }
    dispatch(
        c,
        pass,
        &p.t_q82_matvec,
        bc,
        step,
        "qwen4t-q82",
        || {
            vec![
                w.buf.clone(),
                x.clone(),
                y.clone(),
                uniform_u32x8(
                    c,
                    [
                        (w.cols / 4) as u32,
                        w.rows as u32,
                        w.cols as u32,
                        nt as u32,
                        xs / 4,
                        ys,
                        0,
                        0,
                    ],
                ),
            ]
        },
        ((w.rows as u32).div_ceil(4).min(MAX_WG), 1, 1),
    );
    true
}

/// Model matrix `idx` over `nt` inputs, by dtype.
#[allow(clippy::too_many_arguments)]
fn mv_t(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    model: &Arc<CmfModel>,
    idx: usize,
    x: &wgpu::Buffer,
    xs: u32,
    y: &wgpu::Buffer,
    ys: u32,
    nt: usize,
    bc: Bc,
    step: u16,
) -> Option<()> {
    let w = weight(c, model, idx)?;
    if w.dtype == TensorDtype::F16 && nt > 1 {
        let zero = zero_buf(c);
        return pair_t(
            c, p, pass, model, idx, None, x, xs, y, ys, &zero, 0, 0, 1.0, 0, nt, bc, step,
        );
    }
    q82_t(c, p, pass, &w, x, xs, y, ys, nt, bc, step).then_some(())
}

/// Inject a block into the hyper streams of `nt` tokens. `use_cold` bits
/// (1: `cold`, 2: `blk2`) are OR-ed with word 0 of `flags`, a storage
/// buffer the pending inject's finalize rewrites.
#[allow(clippy::too_many_arguments)]
fn inject_t(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    g: &Geom,
    hyper: &wgpu::Buffer,
    blk: &wgpu::Buffer,
    bs: u32,
    w: &wgpu::Buffer,
    ws: u32,
    cold: (&wgpu::Buffer, u32),
    blk2: (&wgpu::Buffer, u32),
    use_cold: u32,
    flags: &wgpu::Buffer,
    nt: usize,
    bc: Bc,
    step: u16,
) {
    let hh = g.hc * g.hidden;
    dispatch(
        c,
        pass,
        &p.t_inject,
        bc,
        step,
        "qwen4t-inject",
        || {
            vec![
                hyper.clone(),
                blk.clone(),
                w.clone(),
                cold.0.clone(),
                uniform_u32x16(
                    c,
                    [
                        g.hc as u32,
                        g.hidden as u32,
                        (1.0 / g.hc as f32).to_bits(),
                        use_cold,
                        es(hh * 4),
                        bs,
                        ws,
                        cold.1,
                        blk2.1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                ),
                blk2.0.clone(),
                flags.clone(),
            ]
        },
        ((hh as u32).div_ceil(256), nt as u32, 1),
    );
}

/// The pending block of the previous sub-layer, entering the hyper state
/// in front of a mix: h[t,s,d] += 2σ(gate[t,s]/hc)·blk[t,d] (`q4t_inject`
/// with no cold parts). A v3 mix applies it inside its kernels; in front of
/// the older kernels it is its own dispatch, at the site (`step`) it always had.
#[derive(Clone, Copy)]
struct PreInject<'a> {
    blk: &'a wgpu::Buffer,
    /// `blk` row stride, floats.
    bs: u32,
    /// The raw injection-gate logits (the B rows of the mix that made `blk`).
    gate: &'a wgpu::Buffer,
    /// `gate` row stride, floats.
    gs: u32,
    step: u16,
}

/// `pre` as its own `q4t_inject` dispatch.
#[allow(clippy::too_many_arguments)]
fn inject_pre(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    g: &Geom,
    hyper: &wgpu::Buffer,
    pre: &PreInject,
    nt: usize,
    bc: Bc,
) {
    let zero = zero_buf(c);
    inject_t(
        c,
        p,
        pass,
        g,
        hyper,
        pre.blk,
        pre.bs,
        pre.gate,
        pre.gs,
        (&zero, 0),
        (&zero, 0),
        0,
        &zero,
        nt,
        bc,
        pre.step,
    );
}

/// Do the mixes of this device take the pending inject into their kernels
/// (HC v3 built, `CMF_QWEN_HC_V3_INJECT` not 0)? A mix whose shape v3 does
/// not take still injects first, through `inject_pre`.
/// `CMF_QWEN_PAIR1=0`: one-token f16 pairs on `q4t_f16_pair` as before.
fn pair1_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_QWEN_PAIR1").as_deref() != Ok("0"))
}

/// `CMF_QWEN_COLD_K=0`: the cold pass runs top_k slots for every token of
/// the frame (padded with weight-zero copies) instead of the frame's
/// longest cold list.
fn cold_k_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_QWEN_COLD_K").as_deref() != Ok("0"))
}

fn hc_v3_fuses(p: &Pipes) -> bool {
    p.hc3.is_some() && hc_v3_inject_env()
}

/// One hyper-connection mix over `nt` tokens, `pre` injected first: normed
/// streams → low-rank (down + injection gate) → up-projection and fold.
/// HC v3 where the device and the shape take it, the older kernels
/// otherwise. Needs the f16 mixer weights the converter emits; None
/// otherwise.
#[allow(clippy::too_many_arguments)]
fn encode_hc_t(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    model: &Arc<CmfModel>,
    g: &Geom,
    hc: &HcW,
    hyper: &wgpu::Buffer,
    x: &wgpu::Buffer,
    inj: Option<&wgpu::Buffer>,
    pre: Option<PreInject>,
    nt: usize,
    bc: Bc,
    base: u16,
) -> Option<()> {
    let down = weight(c, model, hc.down)?;
    let up = weight(c, model, hc.up)?;
    let inj_w = match hc.inject.filter(|_| inj.is_some()) {
        Some(i) => Some(weight(c, model, i)?),
        None => None,
    };
    encode_hc_w(
        c,
        p,
        pass,
        g,
        hc.norm,
        &down,
        &up,
        inj_w.as_ref(),
        hyper,
        x,
        inj,
        pre,
        nt,
        bc,
        base,
    )
}

/// `encode_hc_t` on matrices already resident (`norm`: the hc·hidden
/// stream weights).
#[allow(clippy::too_many_arguments)]
fn encode_hc_w(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    g: &Geom,
    norm: &[f32],
    down: &WeightRef,
    up: &WeightRef,
    inj_w: Option<&WeightRef>,
    hyper: &wgpu::Buffer,
    x: &wgpu::Buffer,
    inj: Option<&wgpu::Buffer>,
    pre: Option<PreInject>,
    nt: usize,
    bc: Bc,
    base: u16,
) -> Option<()> {
    let hh = g.hc * g.hidden;
    if down.dtype != TensorDtype::F16
        || down.rows % 2 != 0
        || up.dtype != TensorDtype::F16
        || up.cols != down.rows
        || up.rows != hh
        || norm.len() < hh
    {
        return None;
    }
    let nw = const_buf(c, bytemuck::cast_slice(&norm[..hh]));
    let low = tbuf(c, T_LOW, down.rows * 4, false);
    let zero = zero_buf(c);
    let injb = inj.cloned().unwrap_or_else(|| zero.clone());
    if encode_hc_v3(
        c,
        p,
        pass,
        g,
        &nw,
        down,
        up,
        inj_w,
        hyper,
        x,
        &low,
        &injb,
        pre.as_ref(),
        nt,
        bc,
        base,
    )
    .is_some()
    {
        return Some(());
    }
    if let Some(q) = &pre {
        inject_pre(c, p, pass, g, hyper, q, nt, bc);
    }
    encode_hc_old(
        c, p, pass, g, &nw, down, up, inj_w, hyper, x, &low, &injb, nt, bc, base,
    )
}

/// The v3 mix: `hc3_down` (stream norms, down rows and injection-gate rows
/// of every token, `pre` applied on the fly) then `hc3_upfold`
/// (up-projection and fold; with `pre` the injected state is written back
/// in place, which is what the separate inject would have left). Cache
/// slots `base + 1..=4`. None, with nothing encoded, where the device or the
/// shape does not take it.
#[allow(clippy::too_many_arguments)]
fn encode_hc_v3(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    g: &Geom,
    nw: &wgpu::Buffer,
    down: &WeightRef,
    up: &WeightRef,
    inj_w: Option<&WeightRef>,
    hyper: &wgpu::Buffer,
    x: &wgpu::Buffer,
    low: &wgpu::Buffer,
    injb: &wgpu::Buffer,
    pre: Option<&PreInject>,
    nt: usize,
    bc: Bc,
    base: u16,
) -> Option<()> {
    let h3 = p.hc3.as_ref()?;
    let hh = g.hc * g.hidden;
    let lr = down.rows;
    if g.hc != 4
        || g.hidden % 8 != 0
        || lr % 8 != 0
        || down.cols != hh
        || nt == 0
        || nt > TMAX
        || inj_w.is_some_and(|b| b.dtype != TensorDtype::F16 || b.cols != hh)
        || pre.is_some_and(|q| q.bs % 4 != 0)
    {
        return None;
    }
    let rows_b = inj_w.map_or(0, |b| b.rows);
    let groups = (lr + rows_b).div_ceil(2);
    if groups > MAX_WG as usize || g.hidden / 8 > MAX_WG as usize {
        return None;
    }
    // loop counts: 8-column chunks per lane (256 lanes), low/8 words per octet lane
    let kd = (hh / 8).div_ceil(256) as u32;
    let pd = hc3_pipe(c, h3, true, if hc1(nt) { kd | 0x8000 } else { kd })?;
    let pu = hc3_pipe(c, h3, false, (lr / 8).div_ceil(8) as u32)?;
    // the per-token stream inverses, [t·4 + s], from workgroup 0 of the down
    let hinv = tbuf_exact(c, T_HINV, g.hc * 4, false);
    let inv = (1.0 / g.hc as f32).to_bits();
    let flags = u32::from(pre.is_some());
    // No pending block: the inject bindings are never read. The stream
    // weights stand in (read-only in both kernels, so no usage conflict with
    // the up-fold's read-write hyper state or the down's outputs).
    let (blk, bs, gate, gs) = match pre {
        Some(q) => (q.blk.clone(), q.bs, q.gate.clone(), q.gs),
        None => (nw.clone(), 0, nw.clone(), 0),
    };
    let bbuf = inj_w.map_or_else(|| down.buf.clone(), |b| b.buf.clone());
    // the uniforms and the bindings differ with and without the inject
    let (s_down, s_up) = if pre.is_some() {
        (base + 3, base + 4)
    } else {
        (base + 1, base + 2)
    };
    let (ls, hs, os) = (es(lr * 4), es(hh * 4), es(g.hidden * 4));
    if hc4() && rows_b <= 64 {
        // stream-split: the four streams' shares of every row, added in the up-fold
        let pd4 = hc3_pipe(c, h3, true, kd | 0x4000)?;
        let pu4 = hc3_pipe(c, h3, false, (lr / 8).div_ceil(8) as u32 | 0x4000)?;
        let (ls4s, ybs4) = (es(4 * lr * 4), es(4 * rows_b.max(1) * 4));
        let low4 = tbuf(c, T_LOW4, 4 * lr * 4, false);
        let ybp = tbuf(c, T_HCYB, 4 * rows_b.max(1) * 4, false);
        let (blk2, gate2) = (blk.clone(), gate.clone());
        // with and without the inject the bindings differ: their own cache slots
        let (s4_down, s4_up) = if pre.is_some() { (base + 12, base + 13) } else { (base + 10, base + 11) };
        dispatch(
            c,
            pass,
            &pd4,
            bc,
            s4_down,
            "qwen4t-hc3-down4",
            || {
                vec![
                    down.buf.clone(),
                    bbuf,
                    hyper.clone(),
                    nw.clone(),
                    low4.clone(),
                    ybp.clone(),
                    hinv.clone(),
                    uniform_u32x16(
                        c,
                        [
                            hh as u32,
                            lr as u32,
                            rows_b as u32,
                            g.hidden as u32,
                            g.eps.to_bits(),
                            inv,
                            nt as u32,
                            hs / 4,
                            ls4s,
                            ybs4,
                            flags,
                            bs / 4,
                            gs,
                            0,
                            0,
                            0,
                        ],
                    ),
                    blk2,
                    gate2,
                ]
            },
            (groups as u32, 4, 1),
        );
        dispatch(
            c,
            pass,
            &pu4,
            bc,
            s4_up,
            "qwen4t-hc3-upfold4",
            || {
                vec![
                    up.buf.clone(),
                    low4.clone(),
                    hyper.clone(),
                    nw.clone(),
                    hinv.clone(),
                    x.clone(),
                    uniform_u32x16(
                        c,
                        [
                            g.hidden as u32,
                            lr as u32,
                            nt as u32,
                            inv,
                            ls4s / 4,
                            hs,
                            os,
                            flags,
                            bs,
                            gs,
                            rows_b as u32,
                            ybs4,
                            es(g.hc * 4),
                            0,
                            0,
                            0,
                        ],
                    ),
                    blk,
                    gate,
                    ybp.clone(),
                    injb.clone(),
                ]
            },
            ((g.hidden / 8) as u32, 1, 1),
        );
        return Some(());
    }
    dispatch(
        c,
        pass,
        &pd,
        bc,
        s_down,
        "qwen4t-hc3-down",
        || {
            vec![
                down.buf.clone(),
                bbuf,
                hyper.clone(),
                nw.clone(),
                low.clone(),
                injb.clone(),
                hinv.clone(),
                uniform_u32x16(
                    c,
                    [
                        hh as u32,
                        lr as u32,
                        rows_b as u32,
                        g.hidden as u32,
                        g.eps.to_bits(),
                        inv,
                        nt as u32,
                        hs / 4,
                        ls,
                        es(g.hc * 4),
                        flags,
                        bs / 4,
                        gs,
                        0,
                        0,
                        0,
                    ],
                ),
                blk.clone(),
                gate.clone(),
            ]
        },
        (groups as u32, 1, 1),
    );
    dispatch(
        c,
        pass,
        &pu,
        bc,
        s_up,
        "qwen4t-hc3-upfold",
        || {
            vec![
                up.buf.clone(),
                low.clone(),
                hyper.clone(),
                nw.clone(),
                hinv.clone(),
                x.clone(),
                uniform_u32x16(
                    c,
                    [
                        g.hidden as u32,
                        lr as u32,
                        nt as u32,
                        inv,
                        ls / 4,
                        hs,
                        os,
                        flags,
                        bs,
                        gs,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                ),
                blk,
                gate,
            ]
        },
        ((g.hidden / 8) as u32, 1, 1),
    );
    Some(())
}

/// The pre-v3 mixes: group norm + `q4t_f16_pair` + `q4t_hc_upfold` (three
/// dispatches), or with `CMF_QWEN_HC_FUSE=1` the norm folded into
/// `q4t_hc_down` + `q4t_hc_upfold2` (two).
#[allow(clippy::too_many_arguments)]
fn encode_hc_old(
    c: &Ctx,
    p: &Pipes,
    pass: &mut wgpu::ComputePass<'_>,
    g: &Geom,
    nw: &wgpu::Buffer,
    down: &WeightRef,
    up: &WeightRef,
    inj_w: Option<&WeightRef>,
    hyper: &wgpu::Buffer,
    x: &wgpu::Buffer,
    low: &wgpu::Buffer,
    injb: &wgpu::Buffer,
    nt: usize,
    bc: Bc,
    base: u16,
) -> Option<()> {
    let hh = g.hc * g.hidden;
    if hc_fused() && g.hc <= 8 && g.hidden % 2 == 0 {
        // the norm folded into both kernels: two dispatches a mix
        if inj_w.is_some_and(|b| b.dtype != TensorDtype::F16 || b.cols != hh) {
            return None;
        }
        let rows_b = inj_w.map_or(0, |b| b.rows);
        let total = down.rows + rows_b;
        if total > MAX_WG as usize {
            return None;
        }
        let bbuf = inj_w.map_or_else(|| down.buf.clone(), |b| b.buf.clone());
        let hinv = tbuf_exact(c, T_HINV, g.hc * 4, false);
        let inv = 1.0 / g.hc as f32;
        dispatch(
            c,
            pass,
            &p.t_hc_down,
            bc,
            base + 8,
            "qwen4t-hc-down",
            || {
                vec![
                    down.buf.clone(),
                    bbuf,
                    hyper.clone(),
                    nw.clone(),
                    low.clone(),
                    injb.clone(),
                    hinv.clone(),
                    uniform_u32x16(
                        c,
                        [
                            hh as u32,
                            down.rows as u32,
                            rows_b as u32,
                            1,
                            inv.to_bits(),
                            0,
                            nt as u32,
                            es(hh * 4),
                            es(down.rows * 4),
                            es(g.hc * 4),
                            g.hidden as u32,
                            g.eps.to_bits(),
                            g.hc as u32,
                            0,
                            0,
                            0,
                        ],
                    ),
                ]
            },
            (total as u32, 1, 1),
        );
        dispatch(
            c,
            pass,
            &p.t_hc_upfold2,
            bc,
            base + 9,
            "qwen4t-upfold2",
            || {
                vec![
                    up.buf.clone(),
                    low.clone(),
                    hyper.clone(),
                    nw.clone(),
                    hinv.clone(),
                    x.clone(),
                    uniform_u32x8(
                        c,
                        [
                            g.hc as u32,
                            g.hidden as u32,
                            down.rows as u32,
                            inv.to_bits(),
                            nt as u32,
                            es(down.rows * 4),
                            es(hh * 4),
                            es(g.hidden * 4),
                        ],
                    ),
                ]
            },
            (g.hidden as u32, 1, 1),
        );
        return Some(());
    }
    let normed = tbuf(c, T_NORMED, hh * 4, false);
    gn_t(
        c,
        p,
        pass,
        hyper,
        es(hh * 4),
        nw,
        &normed,
        es(hh * 4),
        g.hc,
        g.hidden,
        g.eps,
        nt,
        bc,
        base,
    );
    pair_w(
        c,
        p,
        pass,
        down,
        inj_w,
        &normed,
        es(hh * 4),
        low,
        es(down.rows * 4),
        injb,
        es(g.hc * 4),
        1,
        1.0 / g.hc as f32,
        0,
        nt,
        bc,
        base + 6,
    )?;
    let inv = (1.0 / g.hc as f32).to_bits();
    if nt > TMAX {
        // up-projection as a GEMM into [nt][hh], then the σ-fold with the norms
        let up_y = tbuf_exact(c, T_HCUP, hh * 4, false);
        let zero = zero_buf(c);
        pair_w(
            c, p, pass, up, None, low, es(down.rows * 4), &up_y, hh as u32, &zero, 0, 0, 1.0, 0, nt, bc, base + 5,
        )?;
        dispatch(
            c,
            pass,
            &p.t_hc_fold_t,
            bc,
            base + 4,
            "qwen4t-hc-fold",
            || {
                vec![
                    up_y.clone(),
                    normed.clone(),
                    x.clone(),
                    uniform_u32x8(
                        c,
                        [
                            g.hc as u32,
                            g.hidden as u32,
                            nt as u32,
                            inv,
                            hh as u32,
                            es(hh * 4),
                            es(g.hidden * 4),
                            0,
                        ],
                    ),
                ]
            },
            ((g.hidden as u32).div_ceil(256), nt as u32, 1),
        );
        return Some(());
    }
    dispatch(
        c,
        pass,
        &p.t_hc_upfold,
        bc,
        base + 7,
        "qwen4t-upfold",
        || {
            vec![
                up.buf.clone(),
                low.clone(),
                normed.clone(),
                x.clone(),
                uniform_u32x8(
                    c,
                    [
                        g.hc as u32,
                        g.hidden as u32,
                        down.rows as u32,
                        inv,
                        nt as u32,
                        es(down.rows * 4),
                        es(hh * 4),
                        es(g.hidden * 4),
                    ],
                ),
            ]
        },
        (g.hidden as u32, 1, 1),
    );
    Some(())
}

/// A submitted frame whose readback has not been waited for yet.
pub(crate) struct Pending {
    stage: wgpu::Buffer,
    total: u64,
    parts: Vec<(u64, u64)>,
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Submit the encoder with copies of `parts` into one of the staging
/// buffers and start the map; `Pending::wait` finishes it. The host can
/// encode the next frame in between.
pub(crate) fn submit_async(
    dev: &mut Dev,
    mut enc: wgpu::CommandEncoder,
    parts: &[(&wgpu::Buffer, u64)],
) -> Option<Pending> {
    let c = ctx()?;
    let stage = dev.stages[dev.stage_ix].clone();
    dev.stage_ix ^= 1;
    let mut offs = Vec::with_capacity(parts.len());
    let mut total = 0u64;
    for (_, bytes) in parts {
        offs.push((total, *bytes));
        total += bytes.div_ceil(16) * 16;
    }
    if total > stage.size() {
        return None;
    }
    flush_pass(&enc);
    for ((buf, bytes), (off, _)) in parts.iter().zip(&offs) {
        enc.copy_buffer_to_buffer(buf, 0, &stage, *off, *bytes);
    }
    submit(c, finish_enc(enc));
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d2 = done.clone();
    stage
        .slice(..total.max(16))
        .map_async(wgpu::MapMode::Read, move |_| {
            d2.store(true, std::sync::atomic::Ordering::Release);
        });
    Some(Pending {
        stage,
        total: total.max(16),
        parts: offs,
        done,
    })
}

/// Submit a chain whose result copies into `dev.stage()` the caller already
/// encoded (`total` bytes), and start the map.
pub(crate) fn submit_chain(
    dev: &mut Dev,
    enc: wgpu::CommandEncoder,
    total: u64,
) -> Option<Pending> {
    submit_frame(dev, None, finish_frame(enc), total)
}

/// Finish a frame's encoder into a command buffer. wgpu records the real
/// (HAL) commands here, replaying every dispatch of the frame, so the
/// driver finishes the frame it encodes ahead right away, while the card
/// still runs the previous one, instead of on the critical path between
/// that frame's fence and this frame's submit.
pub(crate) fn finish_frame(enc: wgpu::CommandEncoder) -> wgpu::CommandBuffer {
    finish_enc(enc)
}

/// Submit a finished frame (after `pre`, the staged expert copies, in the
/// same queue submission) and map its readback stage.
pub(crate) fn submit_frame(
    dev: &mut Dev,
    pre: Option<wgpu::CommandBuffer>,
    cb: wgpu::CommandBuffer,
    total: u64,
) -> Option<Pending> {
    let c = ctx()?;
    let stage = dev.stages[dev.stage_ix].clone();
    dev.stage_ix ^= 1;
    let total = total.div_ceil(16) * 16;
    if total > stage.size() {
        // the frame is refused, but the arena already counts the staged
        // experts as resident: their copies must still land
        if let Some(pre) = pre {
            submit(c, pre);
        }
        return None;
    }
    note_submit(c);
    c.queue.submit(pre.into_iter().chain(std::iter::once(cb)));
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d2 = done.clone();
    stage
        .slice(..total.max(16))
        .map_async(wgpu::MapMode::Read, move |_| {
            d2.store(true, std::sync::atomic::Ordering::Release);
        });
    Some(Pending {
        stage,
        total: total.max(16),
        parts: vec![(0, total.max(16))],
        done,
    })
}

impl Pending {
    pub(crate) fn wait(self) -> Option<Vec<u8>> {
        let c = ctx()?;
        let slice = self.stage.slice(..self.total);
        if spin_wait() {
            let t0 = std::time::Instant::now();
            loop {
                let _ = c.device.poll(wgpu::PollType::Poll);
                if self.done.load(std::sync::atomic::Ordering::Acquire) {
                    break;
                }
                if t0.elapsed() > std::time::Duration::from_millis(2) {
                    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
                        self.stage.unmap();
                        return None;
                    }
                    break;
                }
                std::hint::spin_loop();
            }
        } else if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            self.stage.unmap();
            return None;
        }
        let out = {
            let Ok(data) = slice.get_mapped_range() else {
                self.stage.unmap();
                return None;
            };
            let mut v = Vec::with_capacity(self.total as usize);
            for &(off, bytes) in &self.parts {
                let o = off as usize;
                v.extend_from_slice(&data[o..o + bytes as usize]);
                let pad = (bytes.div_ceil(16) * 16 - bytes) as usize;
                v.extend(std::iter::repeat_n(0u8, pad));
            }
            v
        };
        self.stage.unmap();
        Some(out)
    }
}

/// Grow a QSA layer's caches so position `need` fits. Returns whether a
/// buffer was replaced (cached bind groups of the layer are then stale).
fn ensure_qsa(c: &Ctx, g: &Geom, q: &mut QsaDev, need: usize) -> bool {
    if need < q.cap && need < q.kcap {
        return false;
    }
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen4-kv-grow"),
        });
    if need >= q.cap {
        let new_cap = (q.cap * 2).max(need + 1).next_power_of_two();
        let hd = g.head_dim;
        let nk = storage_buf(c, "qwen4-k", (g.n_kv_heads * new_cap * hd * 4) as u64);
        let nv = storage_buf(c, "qwen4-v", (g.n_kv_heads * new_cap * hd * 4) as u64);
        for h in 0..g.n_kv_heads {
            let bytes = (q.cap * hd * 4) as u64;
            enc.copy_buffer_to_buffer(
                &q.k,
                (h * q.cap * hd * 4) as u64,
                &nk,
                (h * new_cap * hd * 4) as u64,
                bytes,
            );
            enc.copy_buffer_to_buffer(
                &q.v,
                (h * q.cap * hd * 4) as u64,
                &nv,
                (h * new_cap * hd * 4) as u64,
                bytes,
            );
        }
        q.k = nk;
        q.v = nv;
        q.cap = new_cap;
    }
    if need >= q.kcap {
        let new_cap = (q.kcap * 2).max(need + 1).next_power_of_two();
        let cr = g.compress_ratio.max(1);
        let nr = storage_buf(c, "qwen4-rawk", (new_cap * g.index_dim * 4) as u64);
        let nc = storage_buf(c, "qwen4-ckeys", ((new_cap / cr) * g.index_dim * 4) as u64);
        enc.copy_buffer_to_buffer(&q.rawk, 0, &nr, 0, (q.kcap * g.index_dim * 4) as u64);
        enc.copy_buffer_to_buffer(
            &q.ckeys,
            0,
            &nc,
            0,
            ((q.kcap / cr) * g.index_dim * 4) as u64,
        );
        q.rawk = nr;
        q.ckeys = nc;
        q.kcap = new_cap;
    }
    submit(c, enc.finish());
    true
}

/// What a layer frame leaves for the host: the per-token cold lists
/// (`nt` rows at `cold_stride(g)` bytes) and the MoE inputs (`nt` rows at
/// `hidden·4` bytes).
pub(crate) struct LayerOut {
    pub cold: wgpu::Buffer,
    pub x2: wgpu::Buffer,
}

/// Bytes per token row of the cold list a frame reads back.
pub(crate) fn cold_stride(g: &Geom) -> usize {
    tstride(4 * g.top_k * 4)
}

/// Encode one whole layer for the `nt` tokens of a frame: PLE, attention
/// half, mixer, MoE half up to the resident experts. Token-wide where the
/// tokens are independent (every projection, the hyper-connection mixes,
/// routing, the experts), token by token only through the recurrent state
/// (the GDN conv ring and S, the PLE history, the QSA caches). The MoE
/// output waits in `mo` for the next frame's inject (see `encode_pending`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_layer(
    enc: &mut wgpu::CommandEncoder,
    dev: &mut Dev,
    model: &Arc<CmfModel>,
    g: &Geom,
    w: &LayerW,
    li: usize,
    pos0: usize,
    nt: usize,
    inv_freq: &[f32],
    remap: &[u32],
    shared_slot: u32,
    ple_rows: &[Vec<f32>],
    inject_prev: bool,
    snapshot: bool,
) -> Option<LayerOut> {
    let c = ctx()?;
    let p = pipes(c)?;
    let uid = dev.uid;
    if nt == 0 || nt > TROWS {
        return None;
    }
    let (hidden, hc) = (g.hidden, g.hc);
    let hh = hc * hidden;
    // the expert kernels fix the MoE input/output row stride at `hidden`
    if (hidden * 4) % 256 != 0 || (hh * 4) % 256 != 0 {
        return None;
    }
    // Caches that grow with the position are sized before the pass opens —
    // for the whole chunk: a growth encoded between two tokens of one frame
    // would leave the earlier tokens' dispatches on the replaced buffers.
    // A sealed layer's caches left the card: nothing to grow.
    let o1_sealed = dev.o1.as_ref().is_some_and(|o| o.sealed && !o.refused);
    let grew = !o1_sealed
        && dev.layers[li]
            .qsa
            .as_mut()
            .is_some_and(|q| ensure_qsa(c, g, q, pos0 + TROWS));
    if grew {
        dev.forget_layer(li);
    }
    let global = c.dsv4_global_moe.lock().unwrap().get(&uid).cloned()?;
    if global.gu_q2 != g.gu_q2 || global.fp4 {
        return None;
    }
    let (p_gu, p_dn) = dsv4_global_moe_pipelines(c, g.gu_q2, global.segments)?;
    // Per layer, not the shared frame pool: a chain writes every layer's
    // remap before one submit, and one buffer would route all of them with
    // the last layer's table.
    let remapb = store_slot(
        c,
        T_REMAP,
        uid,
        li,
        bytemuck::cast_slice(&remap[..g.n_experts]),
    );

    let d: &Dev = &*dev;
    let bc = Bc {
        dev: d,
        li,
        tok: TW + nt,
        direct: false,
    };
    let hyper = &d.hyper;
    let zero = zero_buf(c);
    let mo = tbuf(c, T_MO, hidden * 4, true);
    let inj_mlp = tbuf(c, T_INJ_MLP, hc * 4, true);
    let x = tbuf(c, T_X, hidden * 4, false);
    let x2 = tbuf(c, T_X2, hidden * 4, false);
    let blk = tbuf(c, T_BLK, hidden * 4, false);
    let inj_attn = tbuf(c, T_INJ_ATTN, hc * 4, false);
    let hs = es(hidden * 4);
    let is = es(hc * 4);
    let mut ple_pushes = 0usize;
    // (token slot, head, rows) of the PLE snapshots taken in this frame
    let mut ple_snapped: Vec<(usize, usize, usize)> = Vec::new();
    // Does the PLE pass below read and update the hyper rows?
    let ple_runs =
        w.ple.is_some() && d.layers[li].ple.is_some() && !skip("ple") && !ple_rows.is_empty();
    // The previous frame's MoE output rides in the attention mix's kernels
    // (HC v3) when nothing reads the state in between, i.e. no PLE pass.
    let defer_inject = inject_prev && !ple_runs && hc_v3_fuses(p);
    {
        let mut pass = begin_pass(enc);
        encode_gate(c, p, &mut pass, d, li);
        if inject_prev && !defer_inject {
            // the previous frame's MoE output enters the state here (gated
            // with this frame: a miss before it stops the chain)
            inject_t(
                c,
                p,
                &mut pass,
                g,
                hyper,
                &mo,
                hs,
                &inj_mlp,
                is,
                (&zero, 0),
                (&zero, 0),
                0,
                &zero,
                nt,
                bc,
                2,
            );
        }
    }

    // ── PLE (its embedding rows are gathered on the host), token by token
    // through the history ring ──
    if let (Some(pw), Some(pd)) = (&w.ple, d.layers[li].ple.as_ref())
        && !skip("ple")
        && !ple_rows.is_empty()
    {
        let nk = const_buf(c, bytemuck::cast_slice(&pw.norm_key[..hh]));
        let nq = const_buf(c, bytemuck::cast_slice(&pw.norm_query[..hh]));
        let nc = const_buf(c, bytemuck::cast_slice(&pw.norm_conv[..hh]));
        let taps = const_buf(c, bytemuck::cast_slice(&pw.conv[..hh * g.ple_kernel]));
        let (mut ph, mut pr) = (pd.head, pd.rows);
        for _ in 0..d.ple_touched[li] {
            ph = (ph + 1) % pd.cap;
            pr = (pr + 1).min(pd.cap);
        }
        for (t, rows) in ple_rows.iter().enumerate().take(nt) {
            let bt = Bc {
                dev: d,
                li,
                tok: t,
                direct: false,
            };
            if snapshot && let Some(slot) = d.ple_snaps[li].get(t).map(|(b, _, _)| b.clone()) {
                // the ring as it stands before this token's push
                flush_pass(&*enc);
                enc.copy_buffer_to_buffer(&pd.hist, 0, &slot, 0, (pd.cap * hh * 4) as u64);
                ple_snapped.push((t, ph, pr));
            }
            let emb = frame_buf_t(c, T_EMB, t, rows.len() * 4, true);
            c.queue.write_buffer(&emb, 0, bytemuck::cast_slice(rows));
            let keyraw = frame_buf_t(c, T_KEYRAW, t, hh * 4, false);
            let val = frame_buf_t(c, T_VAL, t, hidden * 4, false);
            let keyn = frame_buf_t(c, T_KEYN, t, hh * 4, false);
            let qn = frame_buf_t(c, T_QN, t, hh * 4, false);
            let gated = frame_buf_t(c, T_GATED, t, hh * 4, false);
            let pnorm = frame_buf_t(c, T_PNORM, t, hh * 4, false);
            let mut pass = begin_pass(enc);
            mv(c, p, &mut pass, model, pw.key_proj, &emb, &keyraw, bt, 10)?;
            mv(c, p, &mut pass, model, pw.value_proj, &emb, &val, bt, 11)?;
            group_rmsnorm(
                c, p, &mut pass, &keyraw, &nk, &keyn, hc, hidden, g.eps, bt, 12,
            );
            // the query norm reads this token's hyper row
            dispatch_r(
                c,
                &mut pass,
                &p.group_rmsnorm,
                bt,
                13,
                "qwen4-gn-row",
                || {
                    vec![
                        Rng::row(hyper, t, hh * 4),
                        Rng::all(&nq),
                        Rng::all(&qn),
                        Rng::all(&uniform_u32x4(
                            c,
                            [hc as u32, hidden as u32, g.eps.to_bits(), 0],
                        )),
                    ]
                },
                (hc as u32, 1, 1),
            );
            dispatch(
                c,
                &mut pass,
                &p.ple_gate,
                bt,
                14,
                "qwen4-ple-gate",
                || {
                    vec![
                        keyn.clone(),
                        qn.clone(),
                        val.clone(),
                        gated.clone(),
                        uniform_u32x4(
                            c,
                            [
                                hc as u32,
                                hidden as u32,
                                (hidden as f32).sqrt().recip().to_bits(),
                                0,
                            ],
                        ),
                    ]
                },
                (hc as u32, 1, 1),
            );
            group_rmsnorm(
                c, p, &mut pass, &gated, &nc, &pnorm, hc, hidden, g.eps, bt, 15,
            );
            let pb = uni_slot8(
                c,
                U_PLE,
                uid,
                bt.key(),
                [
                    hh as u32,
                    g.ple_kernel as u32,
                    g.ple_dilation as u32,
                    pd.cap as u32,
                    ph as u32,
                    pr as u32,
                    0,
                    0,
                ],
            );
            dispatch_r(
                c,
                &mut pass,
                &p.ple_conv,
                bt,
                16,
                "qwen4-ple-conv",
                || {
                    vec![
                        Rng::all(&pnorm),
                        Rng::all(&pd.hist),
                        Rng::all(&taps),
                        Rng::all(&gated),
                        Rng::row(hyper, t, hh * 4),
                        Rng::all(&pb),
                    ]
                },
                ((hh as u32).div_ceil(256), 1, 1),
            );
            drop(pass);
            // The normalized vector joins the history after the conv read it.
            flush_pass(&*enc);
            enc.copy_buffer_to_buffer(&pnorm, 0, &pd.hist, (ph * hh * 4) as u64, (hh * 4) as u64);
            ph = (ph + 1) % pd.cap;
            pr = (pr + 1).min(pd.cap);
            ple_pushes += 1;
        }
    }

    // ── attention half ──
    let mut pass = begin_pass(enc);
    ts_pass(&mut pass, 2);
    let pre_attn = defer_inject.then_some(PreInject {
        blk: &mo,
        bs: hs,
        gate: &inj_mlp,
        gs: is,
        step: 2,
    });
    if !skip("hc") {
        encode_hc_t(
            c,
            p,
            &mut pass,
            model,
            g,
            &w.attn_hc,
            hyper,
            &x,
            Some(&inj_attn),
            pre_attn,
            nt,
            bc,
            100,
        )?;
    } else if let Some(q) = &pre_attn {
        inject_pre(c, p, &mut pass, g, hyper, q, nt, bc);
    }
    ts_pass(&mut pass, 3);
    match &w.mixer {
        MixerW::Gdn {
            qkv,
            z,
            a,
            b,
            out,
            conv1d,
            a_log,
            dt_bias,
            norm,
        } => {
            if !skip("gdn") {
                let gd = g.gdn;
                let cdim = 2 * gd.nk * gd.dk + gd.nv * gd.dv;
                let (ring, st) = d.layers[li].gdn.as_ref()?;
                // the looped kernels fix every row stride: cdim, nv, nv·dv
                let qkvb = tbuf_exact(c, T_QKV, cdim * 4, false);
                let zb = tbuf_exact(c, T_Z, gd.nv * gd.dv * 4, false);
                let ab = tbuf_exact(c, T_A, gd.nv * 4, false);
                let bb = tbuf_exact(c, T_B, gd.nv * 4, false);
                let cq = tbuf_exact(c, T_CQ, cdim * 4, false);
                let gdo = tbuf_exact(c, T_GDO, gd.nv * gd.dv * 4, false);
                let (qs, zs, abs_, gs) = (
                    cdim as u32,
                    (gd.nv * gd.dv) as u32,
                    gd.nv as u32,
                    (gd.nv * gd.dv) as u32,
                );
                if cdim % 4 != 0 || (gd.nv * gd.dv) % 4 != 0 {
                    return None;
                }
                mv_t(c, p, &mut pass, model, *qkv, &x, hs, &qkvb, qs, nt, bc, 400)?;
                mv_t(c, p, &mut pass, model, *z, &x, hs, &zb, zs, nt, bc, 401)?;
                if pair_t(
                    c,
                    p,
                    &mut pass,
                    model,
                    *a,
                    Some(*b),
                    &x,
                    hs,
                    &ab,
                    abs_,
                    &bb,
                    abs_,
                    0,
                    1.0,
                    0,
                    nt,
                    bc,
                    408,
                )
                .is_none()
                {
                    mv_t(c, p, &mut pass, model, *a, &x, hs, &ab, abs_, nt, bc, 402)?;
                    mv_t(c, p, &mut pass, model, *b, &x, hs, &bb, abs_, nt, bc, 403)?;
                }
                ts_pass(&mut pass, 9);
                let taps = const_buf(c, bytemuck::cast_slice(&conv1d[..cdim * gd.kk]));
                let alog = const_buf(c, bytemuck::cast_slice(&a_log[..gd.nv]));
                let dtb = const_buf(c, bytemuck::cast_slice(&dt_bias[..gd.nv]));
                let gnorm = const_buf(c, bytemuck::cast_slice(&norm[..gd.dv]));
                // The shared step kernel leaves the un-normed per-head
                // output; the fused `gdn_step` / `gdn_step_norm` apply a SiLU
                // gate, and this model gates with a sigmoid, so the norm is
                // ours.
                if !c.gdn_par {
                    return None;
                }
                // Position-looped conv and step: one dispatch each for the
                // whole frame; in a verify window every position's (ring, S)
                // lands in the layer's snapshot buffer as it is produced.
                let stride = snap_stride(g);
                let ring_els = (gd.kk - 1) * cdim;
                let snap = if snapshot {
                    d.snaps[li].as_ref().map(|(b, _)| b.clone())
                } else {
                    None
                };
                let snapping = snap.is_some();
                let snapb = snap.unwrap_or_else(|| zero.clone());
                let gc_p = uniform_u32x8(
                    c,
                    [
                        cdim as u32,
                        gd.kk as u32,
                        nt as u32,
                        u32::from(snapping),
                        stride as u32,
                        0,
                        0,
                        0,
                    ],
                );
                let gd_p = uniform_u32x16(
                    c,
                    [
                        gd.nv as u32,
                        gd.dk as u32,
                        gd.dv as u32,
                        (gd.nk * gd.dk) as u32,
                        (gd.nv / gd.nk) as u32,
                        cdim as u32,
                        g.eps.to_bits(),
                        nt as u32,
                        if snapping { stride as u32 } else { 0 },
                        ring_els as u32,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                );
                // snapshot on/off changes the bind group: its own cache slots
                let bs = Bc {
                    dev: d,
                    li,
                    tok: TW + nt + usize::from(snapping) * TMAX,
                    direct: false,
                };
                let bg_conv = bs.get(404, || {
                    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("qwen4-gdn-conv-k"),
                        layout: &c.gdn_conv_k.get_bind_group_layout(0),
                        entries: &[
                            bind_buf(0, &qkvb),
                            bind_buf(1, &taps),
                            bind_buf(2, ring),
                            bind_buf(3, &cq),
                            bind_buf(4, &gc_p),
                            bind_buf(5, &snapb),
                        ],
                    })
                });
                pass.set_pipeline(&c.gdn_conv_k);
                pass.set_bind_group(0, &bg_conv, &[]);
                bs.launch(&mut pass, 404, ((cdim as u32).div_ceil(256), 1, 1));
                // the subgroup-tree twin where it came up (the same sums,
                // `CMF_GDN_SG=0` keeps the loops)
                // a prompt frame past eight tokens: the register-resident scan
                // (`gdn_scan_k`) where it applies; window snapshots keep the step
                let scan = (nt > TMAX && !snapping && gd.dk == super::dense_mv::GDN_SCAN_DK)
                    .then(|| super::dense_mv::gdn(c).and_then(|g| g.scan.as_ref()))
                    .flatten();
                // `CMF_QWEN_SCAN3=1`: the staged scan (`gdn_prep_k` + `gdn_scan3_k`) of the
                // Qwen3.5 hybrids instead of `gdn_scan_k`. Off: on Flash-Next's 48 v-heads over
                // 16 k-heads it read a 4K prompt at 893 tok/s against 910 (PRO 6000), and its
                // logits after a 9K prompt sit 0.2 off the register scan's
                let scan3 = (nt > TMAX
                    && !snapping
                    && gd.dk == super::dense_mv::GDN_SCAN_DK
                    && 2 * gd.nk * gd.dk + gd.nv * gd.dv == cdim
                    && std::env::var("CMF_QWEN_SCAN3").as_deref() == Ok("1"))
                .then(|| super::dense_mv::gdn(c).and_then(|g| g.scan2.as_ref()))
                .flatten();
                if let Some((prep, scan2, cols)) = scan3 {
                    let qk = frame_buf_t(c, T_GQK, 0, TROWS * gd.nk * 256 * 4, false);
                    let kq = frame_buf_t(c, T_GKQ, 0, TROWS * gd.nk * 4, false);
                    let gb = frame_buf_t(c, T_GGB, 0, TROWS * gd.nv * 2 * 4, false);
                    let bg_prep = bs.get(430, || {
                        let ents = [
                            bind_buf(0, &cq),
                            bind_buf(2, &ab),
                            bind_buf(3, &bb),
                            bind_buf(4, &alog),
                            bind_buf(5, &dtb),
                            bind_buf(9, &gd_p),
                            bind_buf(10, &qk),
                            bind_buf(11, &kq),
                            bind_buf(12, &gb),
                        ];
                        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("qwen4-gdn-prep"),
                            layout: &prep.get_bind_group_layout(0),
                            entries: &ents,
                        })
                    });
                    pass.set_pipeline(prep);
                    pass.set_bind_group(0, &bg_prep, &[]);
                    bs.launch(&mut pass, 430, (gd.nk as u32, nt as u32, 1));
                    let bg_s3 = bs.get(431, || {
                        let ents = [
                            bind_buf(0, &cq),
                            bind_buf(7, st),
                            bind_buf(8, &gdo),
                            bind_buf(9, &gd_p),
                            bind_buf(10, &qk),
                            bind_buf(11, &kq),
                            bind_buf(12, &gb),
                        ];
                        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("qwen4-gdn-scan3"),
                            layout: &scan2.get_bind_group_layout(0),
                            entries: &ents,
                        })
                    });
                    pass.set_pipeline(scan2);
                    pass.set_bind_group(0, &bg_s3, &[]);
                    bs.launch(&mut pass, 431, (gd.nv as u32, (gd.dv as u32).div_ceil(*cols as u32), 1));
                } else if let Some((scan, cols)) = scan {
                    let bg_scan = bs.get(409, || {
                        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("qwen4-gdn-scan"),
                            layout: &scan.get_bind_group_layout(0),
                            entries: &[
                                bind_buf(0, &cq),
                                bind_buf(2, &ab),
                                bind_buf(3, &bb),
                                bind_buf(4, &alog),
                                bind_buf(5, &dtb),
                                bind_buf(7, st),
                                bind_buf(8, &gdo),
                                bind_buf(9, &gd_p),
                            ],
                        })
                    });
                    pass.set_pipeline(scan);
                    pass.set_bind_group(0, &bg_scan, &[]);
                    bs.launch(&mut pass, 409, (gd.nv as u32, (gd.dv as u32).div_ceil(*cols as u32), 1));
                } else {
                    let park = super::dense_mv::gdn(c).map_or(&c.gdn_step_par_k, |s| &s.park);
                    let bg_step = bs.get(405, || {
                        // the auto layout keeps only what the entry point touches
                        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("qwen4-gdn-step-k"),
                            layout: &park.get_bind_group_layout(0),
                            entries: &[
                                bind_buf(0, &cq),
                                bind_buf(2, &ab),
                                bind_buf(3, &bb),
                                bind_buf(4, &alog),
                                bind_buf(5, &dtb),
                                bind_buf(7, st),
                                bind_buf(8, &gdo),
                                bind_buf(9, &gd_p),
                                bind_buf(10, &snapb),
                            ],
                        })
                    });
                    pass.set_pipeline(park);
                    pass.set_bind_group(0, &bg_step, &[]);
                    bs.launch(
                        &mut pass,
                        405,
                        (gd.nv as u32, (gd.dv as u32).div_ceil(4), 1),
                    );
                }
                dispatch(
                    c,
                    &mut pass,
                    &p.t_gdn_norm,
                    bc,
                    406,
                    "qwen4t-gdn-norm",
                    || {
                        vec![
                            gdo.clone(),
                            zb.clone(),
                            gnorm.clone(),
                            uniform_u32x8(
                                c,
                                [gd.nv as u32, gd.dv as u32, g.eps.to_bits(), gs, zs, 0, 0, 0],
                            ),
                        ]
                    },
                    (gd.nv as u32, nt as u32, 1),
                );
                ts_pass(&mut pass, 10);
                mv_t(
                    c, p, &mut pass, model, *out, &gdo, gs, &blk, hs, nt, bc, 407,
                )?;
            }
        }
        MixerW::Qsa {
            q,
            k,
            v,
            o,
            index_qk,
            q_norm,
            k_norm,
            iq_norm,
            ik_norm,
        } => {
            if !skip("qsa") {
                let (nh, nkv, hd, rd) = (g.n_heads, g.n_kv_heads, g.head_dim, g.rotary_dim);
                let (ih, idim) = (g.index_heads, g.index_dim);
                let ird = rd.min(idim);
                let cr = g.compress_ratio.max(1);
                if inv_freq.len() * 2 < rd {
                    return None;
                }
                let qd = d.layers[li].qsa.as_ref()?;
                let iqk_len = (ih + 1) * idim * 4;
                let iqk = tbuf(c, T_IQK, iqk_len, false);
                let qg = tbuf(c, T_QG, nh * hd * 2 * 4, false);
                let kb = tbuf(c, T_K, nkv * hd * 4, false);
                let vb = tbuf(c, T_V, nkv * hd * 4, false);
                let iq = tbuf(c, T_IQ, ih * idim * 4, false);
                let qb = tbuf(c, T_Q, nh * hd * 4, false);
                let gate = tbuf(c, T_GATE, nh * hd * 4, false);
                let scores = tbuf(c, T_SCORES, MAX_INDEX_BLOCKS * 4, false);
                let pick_len = (g.index_budget / cr).max(1) * 4;
                let pick = tbuf(c, T_PICK, pick_len, false);
                let cnt = tbuf(c, T_CNT, 16, false);
                let idx = tbuf(c, T_IDX, MAX_ATTEND * 4, false);
                let att = tbuf(c, T_ATT, nh * hd * 4, false);
                mv_t(
                    c,
                    p,
                    &mut pass,
                    model,
                    *index_qk,
                    &x,
                    hs,
                    &iqk,
                    es(iqk_len),
                    nt,
                    bc,
                    500,
                )?;
                mv_t(
                    c,
                    p,
                    &mut pass,
                    model,
                    *q,
                    &x,
                    hs,
                    &qg,
                    es(nh * hd * 2 * 4),
                    nt,
                    bc,
                    501,
                )?;
                mv_t(
                    c,
                    p,
                    &mut pass,
                    model,
                    *k,
                    &x,
                    hs,
                    &kb,
                    es(nkv * hd * 4),
                    nt,
                    bc,
                    502,
                )?;
                mv_t(
                    c,
                    p,
                    &mut pass,
                    model,
                    *v,
                    &x,
                    hs,
                    &vb,
                    es(nkv * hd * 4),
                    nt,
                    bc,
                    503,
                )?;
                ts_pass(&mut pass, 9);
                if qsa_tw() {
                    let freq = const_buf(c, bytemuck::cast_slice(&inv_freq[..rd / 2]));
                    let qnw = const_buf(c, bytemuck::cast_slice(&q_norm[..hd]));
                    let knw = const_buf(c, bytemuck::cast_slice(&k_norm[..hd]));
                    let iqnw = const_buf(c, bytemuck::cast_slice(&iq_norm[..idim]));
                    let iknw = const_buf(c, bytemuck::cast_slice(&ik_norm[..idim]));
                    let ixw = const_buf(c, bytemuck::cast_slice(&d.ixw[..ih]));
                    // The frame's token table: (pos, complete blocks, kept blocks,
                    // attended) per slot. Always TMAX rows: `store_slot` replaces
                    // its buffer when a larger size is asked for, and the bind
                    // groups cached for smaller frames would keep the old one and
                    // read a stale table (the first multi-token frame after
                    // single-token ones left every later token attending at the
                    // wrong position).
                    let mut tab: Vec<u32> = vec![0; TROWS * 4];
                    let (mut max_complete, mut max_m) = (0usize, 0usize);
                    let mut all_kept = true;
                    for t in 0..nt {
                        let pos = pos0 + t;
                        let npos = pos + 1;
                        let complete = npos / cr;
                        let keep = (g.index_budget / cr).min(complete);
                        all_kept &= keep == complete;
                        let m = keep * cr + (npos - complete * cr);
                        // the indexer's tables cap the exact path; a sealed layer
                        // attends its far field and never reads them
                        if !o1_sealed && (complete > index_blocks_cap() || m > MAX_ATTEND) {
                            return None;
                        }
                        max_complete = max_complete.max(complete);
                        max_m = max_m.max(m);
                        tab[t * 4..t * 4 + 4].copy_from_slice(&[
                            pos as u32,
                            complete as u32,
                            keep as u32,
                            m as u32,
                        ]);
                    }
                    let tabb = store_slot(c, T_QTAB, uid, 0, bytemuck::cast_slice(&tab));
                    let (qgs, kvs, iqs2, qbs, gts, iqks) = (
                        es(nh * hd * 2 * 4),
                        es(nkv * hd * 4),
                        es(ih * idim * 4),
                        es(nh * hd * 4),
                        es(nh * hd * 4),
                        es(iqk_len),
                    );
                    // indexer query: per-head norm (1+w) + partial rope, no gate, no K
                    dispatch(
                        c,
                        &mut pass,
                        &p.t_rope,
                        bc,
                        504,
                        "qwen4t-rope-iq",
                        || {
                            vec![
                                iqk.clone(),
                                zero.clone(),
                                iq.clone(),
                                zero.clone(),
                                iqnw.clone(),
                                iknw.clone(),
                                freq.clone(),
                                tabb.clone(),
                                uniform_u32x16(
                                    c,
                                    [
                                        ih as u32,
                                        0,
                                        idim as u32,
                                        ird as u32,
                                        2 | 8,
                                        g.eps.to_bits(),
                                        iqks,
                                        0,
                                        iqs2,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                    ],
                                ),
                            ]
                        },
                        (ih as u32, nt as u32, 1),
                    );
                    // attention q/gate split, q/k norm (1+w), partial rope, K in place
                    dispatch(
                        c,
                        &mut pass,
                        &p.t_rope,
                        bc,
                        505,
                        "qwen4t-rope-qk",
                        || {
                            vec![
                                qg.clone(),
                                kb.clone(),
                                qb.clone(),
                                gate.clone(),
                                qnw.clone(),
                                knw.clone(),
                                freq.clone(),
                                tabb.clone(),
                                uniform_u32x16(
                                    c,
                                    [
                                        nh as u32,
                                        nkv as u32,
                                        hd as u32,
                                        rd as u32,
                                        1 | 2 | 4 | 8,
                                        g.eps.to_bits(),
                                        qgs,
                                        kvs,
                                        qbs,
                                        gts,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                    ],
                                ),
                            ]
                        },
                        ((nh + nkv) as u32, nt as u32, 1),
                    );
                    // The attention layers' O(1) far field (`crate::qwen4_o1`): while the
                    // prompt runs exact its rows leave a query trace; past the seal the
                    // frame's rows go through the engine's far-field kernels in place of
                    // the indexer and the sparse attention, a verify window's keys and
                    // values waiting in the state's spec stash for the host's commit.
                    let o1_on = d.o1.as_ref().is_some_and(|o| !o.refused);
                    let (ngq, ngk) = (nh * hd, nkv * hd);
                    if o1_on {
                        let o = d.o1.as_ref()?;
                        let spec = o1_sealed && snapshot;
                        let (pk, pv, step) = if spec {
                            let map = c.o1m.lock().unwrap();
                            let d1 = map.get(&(uid, li))?;
                            (d1.stages.spec_k.clone(), d1.stages.spec_v.clone(), 522u16)
                        } else {
                            (o.pk.clone(), o.pv.clone(), if o1_sealed { 521 } else { 520 })
                        };
                        let qt = o.qtrace[li].clone().unwrap_or_else(|| zero.clone());
                        let (pq, qcap) = (o.pq.clone(), o.qcap as u32);
                        let trace = u32::from(!o1_sealed);
                        dispatch(
                            c,
                            &mut pass,
                            &p.t_o1_pack,
                            bc,
                            step,
                            "qwen4t-o1-pack",
                            || {
                                vec![
                                    qb.clone(),
                                    kb.clone(),
                                    vb.clone(),
                                    pq,
                                    pk,
                                    pv,
                                    qt,
                                    tabb.clone(),
                                    uniform_u32x16(
                                        c,
                                        [
                                            nh as u32,
                                            nkv as u32,
                                            hd as u32,
                                            qbs,
                                            kvs,
                                            kvs,
                                            trace,
                                            crate::dory::QT_STRIDE as u32,
                                            qcap,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                        ],
                                    ),
                                ]
                            },
                            ((ngq.max(ngk) as u32).div_ceil(256), nt as u32, 1),
                        );
                    }
                    if o1_sealed {
                        let o = d.o1.as_ref()?;
                        let ob = o1_batch::pipes(c)?;
                        let (meta, rk, rv, sk, sv, kt, qt2, mu, mz, th, spk, spv, geom) = {
                            let map = c.o1m.lock().unwrap();
                            let d1 = map.get(&(uid, li))?;
                            (
                                d1.meta.clone(),
                                d1.ring_k.clone(),
                                d1.ring_v.clone(),
                                d1.sink_k.clone(),
                                d1.sink_v.clone(),
                                d1.k_tilde.clone(),
                                d1.qt.clone(),
                                d1.mu.clone(),
                                d1.mz.clone(),
                                d1.that.clone(),
                                d1.stages.spec_k.clone(),
                                d1.stages.spec_v.clone(),
                                o1_batch::Geom {
                                    g: d1.g,
                                    hpg: d1.h,
                                    m: d1.m,
                                    w: d1.w,
                                    ns: d1.ns,
                                    rect_fm: false,
                                    d: d1.dk,
                                    dv: d1.dk,
                                    scale: d1.scale,
                                },
                            )
                        };
                        if geom.g != nkv || geom.hpg * nkv != nh || geom.d != hd {
                            return None;
                        }
                        let st = o1_batch::State {
                            meta: &meta,
                            ring_k: &rk,
                            ring_v: &rv,
                            sink_k: &sk,
                            sink_v: &sv,
                            k_tilde: &kt,
                            qt: &qt2,
                            mu: &mu,
                            mz: &mz,
                            that: &th,
                        };
                        let (kb2, vb2) = if snapshot { (&spk, &spv) } else { (&o.pk, &o.pv) };
                        for (pipe, bg, x, y) in o1_batch::steps(c, ob, &st, geom, &o.pq, kb2, vb2, &o.ob, &o.out, nt, !snapshot) {
                            pass.set_pipeline(&pipe);
                            pass.set_bind_group(0, &bg, &[]);
                            pass.dispatch_workgroups(x, y, 1);
                        }
                        let (out, ats) = (o.out.clone(), es(nh * hd * 4));
                        dispatch(
                            c,
                            &mut pass,
                            &p.t_o1_gate,
                            bc,
                            523,
                            "qwen4t-o1-gate",
                            || vec![out, gate.clone(), att.clone(), uniform_u32x8(c, [ngq as u32, gts, ats, 0, 0, 0, 0, 0])],
                            ((ngq as u32).div_ceil(256), nt as u32, 1),
                        );
                    } else {
                        // K/V and the raw indexer keys into the caches at their positions
                        dispatch(
                            c,
                            &mut pass,
                            &p.t_kv_append,
                            bc,
                            506,
                            "qwen4t-kv-append",
                            || {
                                vec![
                                    kb.clone(),
                                    vb.clone(),
                                    qd.k.clone(),
                                    qd.v.clone(),
                                    tabb.clone(),
                                    uniform_u32x8(
                                        c,
                                        [nkv as u32, hd as u32, qd.cap as u32, kvs, kvs, 0, 0, 0],
                                    ),
                                ]
                            },
                            (((nkv * hd) as u32).div_ceil(256), nt as u32, 1),
                        );
                        dispatch(
                            c,
                            &mut pass,
                            &p.t_rawk,
                            bc,
                            513,
                            "qwen4t-rawk",
                            || {
                                vec![
                                    iqk.clone(),
                                    qd.rawk.clone(),
                                    tabb.clone(),
                                    uniform_u32x4(c, [idim as u32, (ih * idim) as u32, iqks, 0]),
                                ]
                            },
                            ((idim as u32).div_ceil(256), nt as u32, 1),
                        );
                        // blocks completed within the frame: their compressed keys
                        for t in 0..nt {
                            let npos = pos0 + t + 1;
                            if npos % cr != 0 {
                                continue;
                            }
                            let complete = npos / cr;
                            let bt = Bc {
                                dev: d,
                                li,
                                tok: t,
                                direct: false,
                            };
                            let pb = uni_slot8(
                                c,
                                U_BK,
                                uid,
                                bt.key(),
                                [
                                    cr as u32,
                                    idim as u32,
                                    (complete - 1) as u32,
                                    ird as u32,
                                    g.eps.to_bits(),
                                    0,
                                    0,
                                    0,
                                ],
                            );
                            dispatch(
                                c,
                                &mut pass,
                                &p.block_key,
                                bt,
                                507,
                                "qwen4-block-key",
                                || {
                                    vec![
                                        qd.rawk.clone(),
                                        iknw.clone(),
                                        freq.clone(),
                                        qd.ckeys.clone(),
                                        pb,
                                    ]
                                },
                                (1, 1, 1),
                            );
                        }
                        let (scs, pks, cts, ids) = (
                            es(MAX_INDEX_BLOCKS * 4),
                            es(pick_len),
                            es(16),
                            es(MAX_ATTEND * 4),
                        );
                        // every block of every token kept and the split attention
                        // taking them: no scores, no ranking, no index list
                        let groups = nh / nkv.max(1);
                        let split = attend_split()
                            && groups <= if attn_sp2() { 12 } else { 16 }
                            && hd <= 256
                            && hd % if attn_sp2() { 32 } else { 4 } == 0
                            && nh % nkv.max(1) == 0;
                        let identity = all_kept && split && topk_r() && qsa_identity();
                        if max_complete > 0 && !identity {
                            dispatch(
                                c,
                                &mut pass,
                                if topk_r() { &p.t_ix_scores_l } else { &p.t_ix_scores },
                                bc,
                                508,
                                "qwen4t-ix-scores",
                                || {
                                    vec![
                                        iq.clone(),
                                        qd.ckeys.clone(),
                                        ixw.clone(),
                                        scores.clone(),
                                        tabb.clone(),
                                        uniform_u32x4(c, [ih as u32, idim as u32, iqs2, scs]),
                                    ]
                                },
                                if topk_r() {
                                    ((max_complete as u32).div_ceil(256), nt as u32, 1)
                                } else {
                                    (max_complete as u32, nt as u32, 1)
                                },
                            );
                            dispatch(
                                c,
                                &mut pass,
                                if topk_r() { &p.t_topk_r } else { &p.t_topk },
                                bc,
                                509,
                                "qwen4t-topk",
                                || {
                                    vec![
                                        scores.clone(),
                                        pick.clone(),
                                        cnt.clone(),
                                        tabb.clone(),
                                        uniform_u32x4(c, [scs, pks, cts, 0]),
                                    ]
                                },
                                (nt as u32, 1, 1),
                            );
                        }
                        if !identity {
                            dispatch(
                                c,
                                &mut pass,
                                &p.t_idx_build,
                                bc,
                                510,
                                "qwen4t-idx",
                                || {
                                    vec![
                                        pick.clone(),
                                        idx.clone(),
                                        tabb.clone(),
                                        uniform_u32x4(c, [cr as u32, pks, ids, 0]),
                                    ]
                                },
                                ((max_m as u32).div_ceil(256).max(1), nt as u32, 1),
                            );
                        }
                        if split {
                            // split-key GQA attention: (KV head, chunk of 32 keys, token)
                            // partials, then one fold per (head, token)
                            const CH: usize = 32;
                            let nsp = max_m.div_ceil(CH).max(1);
                            let ps = hd;
                            let po = frame_buf_t(c, T_ATSP_O, 0, TROWS * nsp.max(MAX_ATTEND.div_ceil(CH)) * nh * ps * 4, false);
                            let pml = frame_buf_t(c, T_ATSP_ML, 0, TROWS * nsp.max(MAX_ATTEND.div_ceil(CH)) * nh * 8, false);
                            let nsp_cap = nsp.max(MAX_ATTEND.div_ceil(CH));
                            dispatch(
                                c,
                                &mut pass,
                                &p.t_attend_sp,
                                bc,
                                524,
                                "qwen4t-attend-sp",
                                || {
                                    vec![
                                        qb.clone(),
                                        qd.k.clone(),
                                        qd.v.clone(),
                                        idx.clone(),
                                        tabb.clone(),
                                        po.clone(),
                                        pml.clone(),
                                        uniform_u32x16(
                                            c,
                                            [
                                                nh as u32,
                                                hd as u32,
                                                (hd as f32).sqrt().recip().to_bits(),
                                                groups as u32,
                                                qd.cap as u32,
                                                qbs,
                                                ids,
                                                CH as u32,
                                                nsp_cap as u32,
                                                ps as u32,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                            ],
                                        ),
                                    ]
                                },
                                (nkv as u32, nsp as u32, nt as u32),
                            );
                            dispatch(
                                c,
                                &mut pass,
                                &p.t_attend_cmb,
                                bc,
                                525,
                                "qwen4t-attend-cmb",
                                || {
                                    vec![
                                        po.clone(),
                                        pml.clone(),
                                        gate.clone(),
                                        tabb.clone(),
                                        att.clone(),
                                        uniform_u32x8(
                                            c,
                                            [
                                                nh as u32,
                                                hd as u32,
                                                CH as u32,
                                                nsp_cap as u32,
                                                ps as u32,
                                                gts,
                                                es(nh * hd * 4),
                                                0,
                                            ],
                                        ),
                                    ]
                                },
                                (nh as u32, nt as u32, 1),
                            );
                        } else {
                        dispatch(
                                c,
                                &mut pass,
                                &p.t_attend,
                                bc,
                                511,
                                "qwen4t-attend",
                                || {
                                    vec![
                                        qb.clone(),
                                        qd.k.clone(),
                                        qd.v.clone(),
                                        idx.clone(),
                                        gate.clone(),
                                        att.clone(),
                                        tabb.clone(),
                                        uniform_u32x16(
                                            c,
                                            [
                                                nh as u32,
                                                hd as u32,
                                                (hd as f32).sqrt().recip().to_bits(),
                                                (nh / nkv) as u32,
                                                qd.cap as u32,
                                                qbs,
                                                ids,
                                                gts,
                                                es(nh * hd * 4),
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                            ],
                                        ),
                                    ]
                                },
                                (nh as u32, nt as u32, 1),
                            );
                        }
                    }
                    ts_pass(&mut pass, 10);
                } else {
                    let freq = const_buf(c, bytemuck::cast_slice(&inv_freq[..rd / 2]));
                    let qnw = const_buf(c, bytemuck::cast_slice(&q_norm[..hd]));
                    let knw = const_buf(c, bytemuck::cast_slice(&k_norm[..hd]));
                    let iqnw = const_buf(c, bytemuck::cast_slice(&iq_norm[..idim]));
                    let iknw = const_buf(c, bytemuck::cast_slice(&ik_norm[..idim]));
                    let ixw = const_buf(c, bytemuck::cast_slice(&d.ixw[..ih]));
                    let dummy = zero.clone();
                    for t in 0..nt {
                        let pos = pos0 + t;
                        let npos = pos + 1;
                        let complete = npos / cr;
                        let keep = (g.index_budget / cr).min(complete);
                        let m = keep * cr + (npos - complete * cr);
                        // the indexer's tables cap the exact path; a sealed layer
                        // attends its far field and never reads them
                        if !o1_sealed && (complete > MAX_INDEX_BLOCKS_OLD || m > MAX_ATTEND) {
                            return None;
                        }
                        let bt = Bc {
                            dev: d,
                            li,
                            tok: t,
                            direct: false,
                        };
                        let lk = bt.key();
                        // indexer query: per-head norm (1+w) + partial rope, no gate, no K
                        let pb = uni_slot12(
                            c,
                            U_ROPE_IQ,
                            uid,
                            lk,
                            [
                                ih as u32,
                                0,
                                idim as u32,
                                ird as u32,
                                pos as u32,
                                2 | 8,
                                g.eps.to_bits(),
                                0,
                                1.0f32.to_bits(),
                                0,
                                0,
                                0,
                            ],
                        );
                        dispatch_r(
                            c,
                            &mut pass,
                            &c.attn_rope,
                            bt,
                            504,
                            "qwen4-rope-iq",
                            || {
                                vec![
                                    Rng::row(&iqk, t, iqk_len),
                                    Rng::all(&dummy),
                                    Rng::row(&iq, t, ih * idim * 4),
                                    Rng::all(&dummy),
                                    Rng::all(&iqnw),
                                    Rng::all(&iknw),
                                    Rng::all(&freq),
                                    Rng::all(&pb),
                                ]
                            },
                            (ih as u32, 1, 1),
                        );
                        // attention q/gate split, q/k norm (1+w), partial rope, K in place
                        let pb = uni_slot12(
                            c,
                            U_ROPE_QK,
                            uid,
                            lk,
                            [
                                nh as u32,
                                nkv as u32,
                                hd as u32,
                                rd as u32,
                                pos as u32,
                                1 | 2 | 4 | 8,
                                g.eps.to_bits(),
                                0,
                                1.0f32.to_bits(),
                                0,
                                0,
                                0,
                            ],
                        );
                        dispatch_r(
                            c,
                            &mut pass,
                            &c.attn_rope,
                            bt,
                            505,
                            "qwen4-rope-qk",
                            || {
                                vec![
                                    Rng::row(&qg, t, nh * hd * 2 * 4),
                                    Rng::row(&kb, t, nkv * hd * 4),
                                    Rng::row(&qb, t, nh * hd * 4),
                                    Rng::row(&gate, t, nh * hd * 4),
                                    Rng::all(&qnw),
                                    Rng::all(&knw),
                                    Rng::all(&freq),
                                    Rng::all(&pb),
                                ]
                            },
                            ((nh + nkv) as u32, 1, 1),
                        );
                        // K/V into the caches at `pos`
                        let pb = uni_slot(
                            c,
                            U_KV,
                            uid,
                            lk,
                            [nkv as u32, hd as u32, qd.cap as u32, pos as u32],
                        );
                        dispatch_r(
                            c,
                            &mut pass,
                            &c.kv_append,
                            bt,
                            506,
                            "qwen4-kv-append",
                            || {
                                vec![
                                    Rng::row(&kb, t, nkv * hd * 4),
                                    Rng::row(&vb, t, nkv * hd * 4),
                                    Rng::all(&qd.k),
                                    Rng::all(&qd.v),
                                    Rng::all(&pb),
                                ]
                            },
                            (((nkv * hd) as u32).div_ceil(256), 1, 1),
                        );
                        drop(pass);
                        // the raw indexer key joins its cache (a copy: not a pass op)
                        flush_pass(&*enc);
                        enc.copy_buffer_to_buffer(
                            &iqk,
                            (t * tstride(iqk_len) + ih * idim * 4) as u64,
                            &qd.rawk,
                            (pos * idim * 4) as u64,
                            (idim * 4) as u64,
                        );
                        pass = begin_pass(enc);
                        // a block completed with this position: its compressed key
                        if npos % cr == 0 {
                            let pb = uni_slot8(
                                c,
                                U_BK,
                                uid,
                                lk,
                                [
                                    cr as u32,
                                    idim as u32,
                                    (complete - 1) as u32,
                                    ird as u32,
                                    g.eps.to_bits(),
                                    0,
                                    0,
                                    0,
                                ],
                            );
                            dispatch(
                                c,
                                &mut pass,
                                &p.block_key,
                                bt,
                                507,
                                "qwen4-block-key",
                                || {
                                    vec![
                                        qd.rawk.clone(),
                                        iknw.clone(),
                                        freq.clone(),
                                        qd.ckeys.clone(),
                                        pb,
                                    ]
                                },
                                (1, 1, 1),
                            );
                        }
                        if complete > 0 {
                            let pb = uni_slot(
                                c,
                                U_IX,
                                uid,
                                lk,
                                [ih as u32, idim as u32, complete as u32, complete as u32],
                            );
                            dispatch_r(
                                c,
                                &mut pass,
                                &c.index_scores,
                                bt,
                                508,
                                "qwen4-ix-scores",
                                || {
                                    vec![
                                        Rng::row(&iq, t, ih * idim * 4),
                                        Rng::all(&qd.ckeys),
                                        Rng::all(&ixw),
                                        Rng::row(&scores, t, MAX_INDEX_BLOCKS * 4),
                                        Rng::all(&pb),
                                    ]
                                },
                                (complete as u32, 1, 1),
                            );
                            let pb =
                                uni_slot(c, U_TK, uid, lk, [complete as u32, keep as u32, 0, 0]);
                            dispatch_r(
                                c,
                                &mut pass,
                                &c.top_k_index,
                                bt,
                                509,
                                "qwen4-topk",
                                || {
                                    vec![
                                        Rng::row(&scores, t, MAX_INDEX_BLOCKS * 4),
                                        Rng::row(&pick, t, pick_len),
                                        Rng::row(&cnt, t, 16),
                                        Rng::all(&pb),
                                    ]
                                },
                                (1, 1, 1),
                            );
                        }
                        let pb = uni_slot(
                            c,
                            U_IB,
                            uid,
                            lk,
                            [keep as u32, cr as u32, complete as u32, npos as u32],
                        );
                        dispatch_r(
                            c,
                            &mut pass,
                            &p.idx_build,
                            bt,
                            510,
                            "qwen4-idx",
                            || {
                                vec![
                                    Rng::row(&pick, t, pick_len),
                                    Rng::row(&idx, t, MAX_ATTEND * 4),
                                    Rng::all(&pb),
                                ]
                            },
                            ((m as u32).div_ceil(256).max(1), 1, 1),
                        );
                        let pb = uni_slot8(
                            c,
                            U_QA,
                            uid,
                            lk,
                            [
                                nh as u32,
                                hd as u32,
                                m as u32,
                                (hd as f32).sqrt().recip().to_bits(),
                                (nh / nkv) as u32,
                                qd.cap as u32,
                                0,
                                0,
                            ],
                        );
                        dispatch_r(
                            c,
                            &mut pass,
                            &p.qsa_attend,
                            bt,
                            511,
                            "qwen4-attend",
                            || {
                                vec![
                                    Rng::row(&qb, t, nh * hd * 4),
                                    Rng::all(&qd.k),
                                    Rng::all(&qd.v),
                                    Rng::row(&idx, t, MAX_ATTEND * 4),
                                    Rng::row(&gate, t, nh * hd * 4),
                                    Rng::row(&att, t, nh * hd * 4),
                                    Rng::all(&pb),
                                ]
                            },
                            (nh as u32, 1, 1),
                        );
                    }
                }
                mv_t(
                    c,
                    p,
                    &mut pass,
                    model,
                    *o,
                    &att,
                    es(nh * hd * 4),
                    &blk,
                    hs,
                    nt,
                    bc,
                    512,
                )?;
            }
        }
    }
    ts_pass(&mut pass, 4);
    // the attention block enters the state: inside the MoE mix's kernels
    // (HC v3), or by its own dispatch in front of it
    let attn_blk = PreInject {
        blk: &blk,
        bs: hs,
        gate: &inj_attn,
        gs: is,
        step: 600,
    };
    let fuse_attn = !skip("hc") && hc_v3_fuses(p);
    if !fuse_attn {
        inject_pre(c, p, &mut pass, g, hyper, &attn_blk, nt, bc);
    }

    // ── MoE half ──
    if !skip("hc") {
        encode_hc_t(
            c,
            p,
            &mut pass,
            model,
            g,
            &w.mlp_hc,
            hyper,
            &x2,
            Some(&inj_mlp),
            fuse_attn.then_some(attn_blk),
            nt,
            bc,
            200,
        )?;
    }
    ts_pass(&mut pass, 5);
    let slots = g.top_k + 1;
    let logits = tbuf(c, T_LOGITS, g.n_experts * 4, false);
    let forced = tbuf_exact(c, T_FORCED, slots * 4, false);
    let msel = tbuf_exact(c, T_MSEL, slots * 4, false);
    let mwt = tbuf_exact(c, T_MWT, slots * 4, false);
    let mcnt = tbuf_exact(c, T_MCNT, 16, false);
    let mact = tbuf_exact(c, T_MACT, slots * g.inter * 4, false);
    let cold = tbuf(c, T_COLD, 4 * g.top_k * 4, false);
    if !skip("route") {
        // router logits and the shared expert's gate in one dispatch
        pair_t(
            c,
            p,
            &mut pass,
            model,
            w.router,
            w.shared_gate,
            &x2,
            hs,
            &logits,
            es(g.n_experts * 4),
            &forced,
            slots as u32,
            2,
            1.0,
            g.top_k,
            nt,
            bc,
            708,
        )?;
        ts_pass(&mut pass, 11);
        // route on the card: Qwen ranking + softmax over the chosen ten, the
        // arena's remap turns winners into slots or hands them back cold
        let rflags: u32 = 8
            | 16
            | 32
            | (u32::from(w.shared_gate.is_some()) << 6)
            | (u32::from(d.masked) << 7)
            | (shared_slot << 8);
        let (route_pipe, route_step) = match &p.route2 {
            Some(r2) if g.n_experts <= 512 && g.top_k <= 64 => (r2, 714),
            _ => (&p.t_route, 703),
        };
        dispatch(
            c,
            &mut pass,
            route_pipe,
            bc,
            route_step,
            "qwen4t-route",
            || {
                vec![
                    logits.clone(),
                    forced.clone(),
                    msel.clone(),
                    mwt.clone(),
                    mcnt.clone(),
                    uniform_u32x8(
                        c,
                        [
                            g.n_experts as u32,
                            g.top_k as u32,
                            rflags,
                            1.0f32.to_bits(),
                            es(g.n_experts * 4),
                            slots as u32,
                            slots as u32,
                            es(4 * g.top_k * 4),
                        ],
                    ),
                    remapb.clone(),
                    cold.clone(),
                ]
            },
            (nt as u32, 1, 1),
        );
    }
    ts_pass(&mut pass, 6);
    capture_rows(c, &mut pass, li, nt, &x2, hs, &logits, es(g.n_experts * 4), hidden, g.n_experts);
    if !skip("experts") {
        // resident experts straight from the global arena, every token in
        // one dispatch (the kernels take the token as their batch index)
        let stride16 = |rows: usize, cols: usize, q2: bool| -> u32 {
            let dt = if q2 {
                TensorDtype::Q2TiledP
            } else {
                TensorDtype::Q4TiledP
            };
            (cortiq_core::quant::expected_nbytes(dt, &[rows, cols]).unwrap_or(0) / 2) as u32
        };
        let gu_u = uniform_u32x8(
            c,
            [
                (hidden / 32) as u32,
                g.inter as u32,
                slots as u32,
                stride16(g.inter, hidden, g.gu_q2),
                0.0f32.to_bits(),
                global.segment_slots as u32,
                0,
                0,
            ],
        );
        let dn_u = uniform_u32x8(
            c,
            [
                (g.inter / 32) as u32,
                hidden as u32,
                slots as u32,
                stride16(hidden, g.inter, false),
                global.segment_slots as u32,
                0,
                0,
                0,
            ],
        );
        // q2tp gate/up banks take the sorted path only on the matrix units
        let mm_ok = p.mm.is_some() && g.inter % 64 == 0 && hidden % 64 == 0 && (!g.gu_q2 || p.mm_gu_q2.is_some());
        if blocked_experts(g, global.segments)
            && nt > TMAX
            && (!g.gu_q2 || mm_ok)
            && sorted_experts()
            && hidden % 16 == 0
            && (nt * slots <= 4096 || mm_ok)
        {
            // a prompt frame past eight tokens: the winners sorted by arena
            // slot, so one expert's weights are read by back-to-back
            // workgroups (L2) instead of once per token from DRAM
            let nw = nt * slots;
            let ord = tbuf_exact(c, T_ORD, slots * 4, false);
            let part = frame_buf_t(c, T_EPART, 0, TROWS * slots * hidden * 4, false);
            if let Some((mm_gu4, mm_dn)) = p.mm.as_ref().filter(|_| mm_ok) {
                let mm_gu = if g.gu_q2 { p.mm_gu_q2.as_ref().unwrap_or(mm_gu4) } else { mm_gu4 };
                // the matrix units: one workgroup per (64-row tile, expert segment)
                let segb = frame_buf_t(c, T_ESEG, 0, 4097 * 16, false);
                if nw <= 4096 {
                    dispatch(
                        c,
                        &mut pass,
                        &p.g_sort,
                        bc,
                        715,
                        "qwen4g-sort",
                        || vec![msel.clone(), ord.clone(), uniform_u32x4(c, [nw as u32, 0, 0, 0])],
                        (1, 1, 1),
                    );
                    dispatch(
                        c,
                        &mut pass,
                        &p.g_segs,
                        bc,
                        734,
                        "qwen4g-segs",
                        || vec![msel.clone(), ord.clone(), segb.clone(), uniform_u32x4(c, [nw as u32, 0, 0, 0])],
                        (1, 1, 1),
                    );
                } else {
                    // past 4096 winners: the counting sort by arena slot
                    let nslots = global.capacity;
                    let cnt = frame_buf_t(c, T_CSCNT, 0, nslots * 4, false);
                    let u = || uniform_u32x4(c, [nw as u32, nslots as u32, 0, 0]);
                    let groups = |n: usize| ((n as u32).div_ceil(256), 1, 1);
                    for (i, (step, grid)) in [(739u16, groups(nslots)), (740, groups(nw)), (741, (1, 1, 1)), (742, groups(nw))]
                        .into_iter()
                        .enumerate()
                    {
                        dispatch(
                            c,
                            &mut pass,
                            &p.g_cs[i],
                            bc,
                            step,
                            "qwen4g-cs",
                            || vec![msel.clone(), cnt.clone(), ord.clone(), segb.clone(), u()],
                            grid,
                        );
                    }
                }
                // a segment per expert of the layer and one for its shared expert
                let nseg = nw.min(g.n_experts + 1) as u32;
                let bg_gu = bc.get(735, || {
                    let gate_b: Vec<_> = global.gate.iter().map(wgpu::Buffer::as_entire_buffer_binding).collect();
                    let up_b: Vec<_> = global.up.iter().map(wgpu::Buffer::as_entire_buffer_binding).collect();
                    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("qwen4g-mm-gu"),
                        layout: &mm_gu.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::BufferArray(&gate_b) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::BufferArray(&up_b) },
                            bind_buf(2, &x2),
                            bind_buf(4, &mact),
                            bind_buf(5, &ord),
                            bind_buf(6, &segb),
                        ],
                    })
                });
                let bg_gu_p = bc.get(736, || {
                    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("qwen4g-mm-gu-p"),
                        layout: &mm_gu.get_bind_group_layout(1),
                        entries: &[bind_buf(0, &gu_u)],
                    })
                });
                pass.set_pipeline(mm_gu);
                pass.set_bind_group(0, &bg_gu, &[]);
                pass.set_bind_group(1, &bg_gu_p, &[]);
                pass.dispatch_workgroups((g.inter / 64) as u32, nseg, 1);
                let bg_dn = bc.get(737, || {
                    let down_b: Vec<_> = global.down.iter().map(wgpu::Buffer::as_entire_buffer_binding).collect();
                    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("qwen4g-mm-dn"),
                        layout: &mm_dn.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::BufferArray(&down_b) },
                            bind_buf(1, &mact),
                            bind_buf(3, &mwt),
                            bind_buf(4, &part),
                            bind_buf(5, &ord),
                            bind_buf(6, &segb),
                        ],
                    })
                });
                let bg_dn_p = bc.get(738, || {
                    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("qwen4g-mm-dn-p"),
                        layout: &mm_dn.get_bind_group_layout(1),
                        entries: &[bind_buf(0, &dn_u)],
                    })
                });
                pass.set_pipeline(mm_dn);
                pass.set_bind_group(0, &bg_dn, &[]);
                pass.set_bind_group(1, &bg_dn_p, &[]);
                pass.dispatch_workgroups((hidden / 64) as u32, nseg, 1);
                dispatch(
                    c,
                    &mut pass,
                    &p.g_sum,
                    bc,
                    727,
                    "qwen4g-sum",
                    || vec![part.clone(), msel.clone(), mo.clone(), uniform_u32x4(c, [hidden as u32, slots as u32, nt as u32, es(hidden * 4)])],
                    ((hidden as u32).div_ceil(256), nt as u32, 1),
                );
            } else {
            dispatch(
                c,
                &mut pass,
                &p.g_sort,
                bc,
                715,
                "qwen4g-sort",
                || vec![msel.clone(), ord.clone(), uniform_u32x4(c, [nw as u32, 0, 0, 0])],
                (1, 1, 1),
            );
            let bg_gu = bc.get(716, || {
                let gate_b: Vec<_> = global.gate.iter().map(wgpu::Buffer::as_entire_buffer_binding).collect();
                let up_b: Vec<_> = global.up.iter().map(wgpu::Buffer::as_entire_buffer_binding).collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4g-gu"),
                    layout: &p.g_gu_q4tp4.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::BufferArray(&gate_b) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::BufferArray(&up_b) },
                        bind_buf(2, &x2),
                        bind_buf(3, &msel),
                        bind_buf(4, &mact),
                        bind_buf(5, &ord),
                    ],
                })
            });
            let bg_gu_p = bc.get(717, || {
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4g-gu-p"),
                    layout: &p.g_gu_q4tp4.get_bind_group_layout(1),
                    entries: &[bind_buf(0, &gu_u)],
                })
            });
            pass.set_pipeline(&p.g_gu_q4tp4);
            pass.set_bind_group(0, &bg_gu, &[]);
            pass.set_bind_group(1, &bg_gu_p, &[]);
            pass.dispatch_workgroups((g.inter / 4) as u32, nw as u32, 1);
            let bg_dn = bc.get(718, || {
                let down_b: Vec<_> = global.down.iter().map(wgpu::Buffer::as_entire_buffer_binding).collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4g-dn"),
                    layout: &p.g_dn_q4tp4.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::BufferArray(&down_b) },
                        bind_buf(1, &mact),
                        bind_buf(2, &msel),
                        bind_buf(3, &mwt),
                        bind_buf(4, &part),
                        bind_buf(5, &ord),
                    ],
                })
            });
            let bg_dn_p = bc.get(719, || {
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4g-dn-p"),
                    layout: &p.g_dn_q4tp4.get_bind_group_layout(1),
                    entries: &[bind_buf(0, &dn_u)],
                })
            });
            pass.set_pipeline(&p.g_dn_q4tp4);
            pass.set_bind_group(0, &bg_dn, &[]);
            pass.set_bind_group(1, &bg_dn_p, &[]);
            pass.dispatch_workgroups((hidden as u32).div_ceil(16), nw as u32, 1);
            dispatch(
                c,
                &mut pass,
                &p.g_sum,
                bc,
                727,
                "qwen4g-sum",
                || vec![part.clone(), msel.clone(), mo.clone(), uniform_u32x4(c, [hidden as u32, slots as u32, nt as u32, es(hidden * 4)])],
                ((hidden as u32).div_ceil(256), nt as u32, 1),
            );
            }
        } else if blocked_experts(g, global.segments) {
            // four (eight) rows a workgroup: x read once per group for all
            let (gu4, dn4) = p.experts_blocked(g);
            let gr = p.gu_rows(g);
            let bg_gu = bc.get(710, || {
                let gate_b: Vec<_> = global
                    .gate
                    .iter()
                    .map(wgpu::Buffer::as_entire_buffer_binding)
                    .collect();
                let up_b: Vec<_> = global
                    .up
                    .iter()
                    .map(wgpu::Buffer::as_entire_buffer_binding)
                    .collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-gu4"),
                    layout: &gu4.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::BufferArray(&gate_b),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::BufferArray(&up_b),
                        },
                        bind_buf(2, &x2),
                        bind_buf(3, &msel),
                        bind_buf(4, &mact),
                    ],
                })
            });
            let bg_gu_p = bc.get(711, || {
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-gu4-p"),
                    layout: &gu4.get_bind_group_layout(1),
                    entries: &[bind_buf(0, &gu_u)],
                })
            });
            pass.set_pipeline(gu4);
            pass.set_bind_group(0, &bg_gu, &[]);
            pass.set_bind_group(1, &bg_gu_p, &[]);
            bc.launch(
                &mut pass,
                704,
                ((g.inter / gr) as u32, slots as u32, nt as u32),
            );
            ts_pass(&mut pass, 12);
            let bg_dn = bc.get(712, || {
                let down_b: Vec<_> = global
                    .down
                    .iter()
                    .map(wgpu::Buffer::as_entire_buffer_binding)
                    .collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-dn4"),
                    layout: &dn4.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::BufferArray(&down_b),
                        },
                        bind_buf(1, &mact),
                        bind_buf(2, &msel),
                        bind_buf(3, &mwt),
                        bind_buf(4, &mo),
                    ],
                })
            });
            let bg_dn_p = bc.get(713, || {
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-dn4-p"),
                    layout: &dn4.get_bind_group_layout(1),
                    entries: &[bind_buf(0, &dn_u)],
                })
            });
            pass.set_pipeline(dn4);
            pass.set_bind_group(0, &bg_dn, &[]);
            pass.set_bind_group(1, &bg_dn_p, &[]);
            bc.launch(&mut pass, 706, ((hidden / 4) as u32, nt as u32, 1));
        } else {
            let bg_gu = bc.get(704, || {
                let gate_b: Vec<_> = global
                    .gate
                    .iter()
                    .map(wgpu::Buffer::as_entire_buffer_binding)
                    .collect();
                let up_b: Vec<_> = global
                    .up
                    .iter()
                    .map(wgpu::Buffer::as_entire_buffer_binding)
                    .collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-global-gu"),
                    layout: &p_gu.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::BufferArray(&gate_b),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::BufferArray(&up_b),
                        },
                        bind_buf(2, &x2),
                        bind_buf(3, &msel),
                        bind_buf(4, &mact),
                        bind_buf(5, &mwt),
                    ],
                })
            });
            let bg_gu_p = bc.get(705, || {
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-global-gu-p"),
                    layout: &p_gu.get_bind_group_layout(1),
                    entries: &[bind_buf(0, &gu_u)],
                })
            });
            pass.set_pipeline(p_gu);
            pass.set_bind_group(0, &bg_gu, &[]);
            pass.set_bind_group(1, &bg_gu_p, &[]);
            bc.launch(&mut pass, 704, (g.inter as u32, slots as u32, nt as u32));
            let bg_dn = bc.get(706, || {
                let down_b: Vec<_> = global
                    .down
                    .iter()
                    .map(wgpu::Buffer::as_entire_buffer_binding)
                    .collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-global-dn"),
                    layout: &p_dn.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::BufferArray(&down_b),
                        },
                        bind_buf(1, &mact),
                        bind_buf(2, &msel),
                        bind_buf(3, &mwt),
                        bind_buf(4, &mo),
                    ],
                })
            });
            let bg_dn_p = bc.get(707, || {
                c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("qwen4-global-dn-p"),
                    layout: &p_dn.get_bind_group_layout(1),
                    entries: &[bind_buf(0, &dn_u)],
                })
            });
            pass.set_pipeline(p_dn);
            pass.set_bind_group(0, &bg_dn, &[]);
            pass.set_bind_group(1, &bg_dn_p, &[]);
            bc.launch(&mut pass, 706, (hidden as u32, nt as u32, 1));
        }
    }
    // the winners' hot neurons: their tubes against the MoE input, into the output
    if let (Some(tb), Some(act)) = (d.tubes.get(li).and_then(Option::as_ref), d.tube_act.as_ref()) {
        if !skip("experts") {
            let (hidden32, h32, k32, e32) = (hidden as u32, tb.h as u32, g.top_k as u32, g.n_experts as u32);
            let tu = uniform_u32x16(
                c,
                [hidden32, h32, k32, e32, hs, es(4 * g.top_k * 4), tb.gseg, tb.dseg, es(hidden * 4), k32, slots as u32, 0, 0, 0, 0, 0],
            );
            let (tg, tup, tdn, act2, tu2, sl) = (tb.gate.clone(), tb.up.clone(), tb.down.clone(), act.clone(), tu.clone(), tb.slots.clone());
            dispatch(
                c,
                &mut pass,
                &p.t_tube_gu,
                bc,
                720,
                "qwen4t-tube-gu",
                || vec![tg, tup, x2.clone(), cold.clone(), act2, tu, sl],
                (g.top_k as u32, nt as u32, (tb.h as u32).div_ceil(32)),
            );
            let (act3, sl2) = (act.clone(), tb.slots.clone());
            dispatch(
                c,
                &mut pass,
                &p.t_tube_dn,
                bc,
                721,
                "qwen4t-tube-dn",
                || vec![tdn, act3, cold.clone(), mo.clone(), tu2, mwt.clone(), sl2],
                ((hidden as u32).div_ceil(256), nt as u32, 1),
            );
        }
    }
    // the shared expert's tube (the whole expert at eight bits), with the route's σ
    if let (Some(tb), Some(act)) = (d.tubes_shared.get(li).and_then(Option::as_ref), d.tube_act_s.as_ref()) {
        if !skip("experts") {
            let (hidden32, h32, k32) = (hidden as u32, tb.h as u32, g.top_k as u32);
            let tu = uniform_u32x16(
                c,
                [hidden32, h32, 0, 1, hs, es(4 * g.top_k * 4), tb.gseg, tb.dseg, es(hidden * 4), 1, slots as u32, k32, 1, 0, 0, 0],
            );
            let (tg, tup, tdn, act2, tu2, sl) = (tb.gate.clone(), tb.up.clone(), tb.down.clone(), act.clone(), tu.clone(), tb.slots.clone());
            dispatch(
                c,
                &mut pass,
                &p.t_tube_gu,
                bc,
                722,
                "qwen4t-tube-gu-s",
                || vec![tg, tup, x2.clone(), cold.clone(), act2, tu, sl],
                (1, nt as u32, (tb.h as u32).div_ceil(32)),
            );
            let (act3, sl2) = (act.clone(), tb.slots.clone());
            dispatch(
                c,
                &mut pass,
                &p.t_tube_dn,
                bc,
                723,
                "qwen4t-tube-dn-s",
                || vec![tdn, act3, cold.clone(), mo.clone(), tu2, mwt.clone(), sl2],
                ((hidden as u32).div_ceil(256), nt as u32, 1),
            );
        }
    }
    // a cold winner here gates every later frame of the chain off
    if d.gated {
        let bind = bc.get(709, || {
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("qwen4-miss"),
                layout: &p.miss.get_bind_group_layout(0),
                entries: &[
                    bind_buf(0, &cold),
                    bind_buf(1, &d.miss),
                    bind_buf(2, &uniform_u32x4(c, [g.top_k as u32, 0, 0, 0])),
                ],
            })
        });
        pass.set_pipeline(&p.miss);
        pass.set_bind_group(0, &bind, &[]);
        bc.launch(&mut pass, 709, (1, 1, 1));
    }
    ts_pass(&mut pass, 7);
    drop(pass);
    for (t, h, r) in ple_snapped {
        if let Some(e) = dev.ple_snaps[li].get_mut(t) {
            e.1 = h;
            e.2 = r;
        }
    }
    dev.ple_touched[li] += ple_pushes;
    dev.pending_inject = true;
    Some(LayerOut { cold, x2 })
}

/// The head: final hyper-connection fold and the vocabulary projection of
/// every token slot. Returns the logits buffer (`nt` rows of `vocab`
/// floats at the returned row stride, in bytes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_head(
    enc: &mut wgpu::CommandEncoder,
    dev: &Dev,
    model: &Arc<CmfModel>,
    g: &Geom,
    head_hc: &HcW,
    lm_head: usize,
    inject_prev: bool,
    nt: usize,
) -> Option<(wgpu::Buffer, usize, usize)> {
    encode_head_with(enc, dev, model, model, g, head_hc, lm_head, inject_prev, nt)
}

/// `encode_head` with the vocabulary projection taken from `lm_model` (the
/// MTP sidecar's mixer folds the state, the main file's lm_head reads it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_head_with(
    enc: &mut wgpu::CommandEncoder,
    dev: &Dev,
    model: &Arc<CmfModel>,
    lm_model: &Arc<CmfModel>,
    g: &Geom,
    head_hc: &HcW,
    lm_head: usize,
    inject_prev: bool,
    nt: usize,
) -> Option<(wgpu::Buffer, usize, usize)> {
    let c = ctx()?;
    let p = pipes(c)?;
    let bc = Bc {
        dev,
        li: dev.rows - 1,
        tok: TW + nt,
        direct: false,
    };
    let hid = tbuf(c, T_HID, g.hidden * 4, false);
    let head = weight(c, lm_model, lm_head)?;
    let logits = tbuf(c, T_LMLOGITS, head.rows * 4, false);
    let mut pass = begin_pass(enc);
    encode_gate(c, p, &mut pass, dev, dev.rows - 1);
    // the last layer's MoE output: inside the final mix's kernels (HC v3),
    // or by its own dispatch in front of it
    let mo = tbuf(c, T_MO, g.hidden * 4, true);
    let inj_mlp = tbuf(c, T_INJ_MLP, g.hc * 4, true);
    let mut pre = inject_prev.then_some(PreInject {
        blk: &mo,
        bs: es(g.hidden * 4),
        gate: &inj_mlp,
        gs: es(g.hc * 4),
        step: 2,
    });
    if let Some(q) = pre.filter(|_| !hc_v3_fuses(p)) {
        inject_pre(c, p, &mut pass, g, &dev.hyper, &q, nt, bc);
        pre = None;
    }
    encode_hc_t(
        c, p, &mut pass, model, g, head_hc, &dev.hyper, &hid, None, pre, nt, bc, 300,
    )?;
    if !skip("head")
        && !q82_t(
            c,
            p,
            &mut pass,
            &head,
            &hid,
            es(g.hidden * 4),
            &logits,
            es(head.rows * 4),
            nt,
            bc,
            800,
        )
    {
        return None;
    }
    drop(pass);
    Some((logits, head.rows, tstride(head.rows * 4)))
}

/// The previous layer's MoE output enters the hyper state of every token
/// slot, together with the cold winners the arena admitted after the route
/// (the same expert kernels over their fresh slots, into a second output
/// the inject adds) and the host's completion. Sized by indirect arguments
/// and flags written once the cold lists are known, so the frame is
/// encoded before the previous one has been read back. Must run before
/// anything else of the next frame reads the state.
pub(crate) fn encode_pending(
    enc: &mut wgpu::CommandEncoder,
    dev: &Dev,
    g: &Geom,
    nt: usize,
) -> bool {
    if !dev.pending_inject {
        return true;
    }
    let Some(c) = ctx() else { return false };
    let Some(p) = pipes(c) else { return false };
    let (hidden, k) = (g.hidden, g.top_k);
    let mo = tbuf(c, T_MO, hidden * 4, true);
    let inj = tbuf(c, T_INJ_MLP, g.hc * 4, true);
    let coldvec = tbuf(c, T_COLDVEC, hidden * 4, true);
    let csel = tbuf_exact(c, T_CSEL, k * 4, true);
    let cwt = tbuf_exact(c, T_CWT, k * 4, true);
    let mact = tbuf_exact(c, T_MACT2, k * g.inter * 4, false);
    let mo2 = tbuf(c, T_MOCOLD, hidden * 4, false);
    let x2 = tbuf(c, T_X2, hidden * 4, false);
    let bc = Bc {
        dev,
        li: LI_PENDING,
        tok: TW + nt,
        direct: true,
    };
    let mut pass = begin_pass(enc);
    let global = c.dsv4_global_moe.lock().unwrap().get(&dev.uid).cloned();
    let Some(global) = global else { return false };
    let Some((p_gu, p_dn)) = dsv4_global_moe_pipelines(c, g.gu_q2, global.segments) else {
        return false;
    };
    let stride16 = |rows: usize, cols: usize, q2: bool| -> u32 {
        let dt = if q2 {
            TensorDtype::Q2TiledP
        } else {
            TensorDtype::Q4TiledP
        };
        (cortiq_core::quant::expected_nbytes(dt, &[rows, cols]).unwrap_or(0) / 2) as u32
    };
    let gu_w = [
        (hidden / 32) as u32,
        g.inter as u32,
        k as u32,
        stride16(g.inter, hidden, g.gu_q2),
        0.0f32.to_bits(),
        global.segment_slots as u32,
        0,
        0,
    ];
    let dn_w = [
        (g.inter / 32) as u32,
        hidden as u32,
        k as u32,
        stride16(hidden, g.inter, false),
        global.segment_slots as u32,
        0,
        0,
        0,
    ];
    let (gu_u, dn_u) = if cold_k_env() {
        // the slot word is rewritten per frame at finalize
        dev.cold_tpl.set(Some((gu_w, dn_w)));
        (dev.cold_u[0].clone(), dev.cold_u[1].clone())
    } else {
        (uniform_u32x8(c, gu_w), uniform_u32x8(c, dn_w))
    };
    let blocked = blocked_experts(g, global.segments);
    let (pipe_gu, pipe_dn): (&wgpu::ComputePipeline, &wgpu::ComputePipeline) = if blocked {
        p.experts_blocked(g)
    } else {
        (p_gu, p_dn)
    };
    let (s_gu, s_gup, s_dn, s_dnp) = if blocked {
        (14, 15, 16, 17)
    } else {
        (10, 11, 12, 13)
    };
    let bg_gu = bc.get(s_gu, || {
        let gate_b: Vec<_> = global
            .gate
            .iter()
            .map(wgpu::Buffer::as_entire_buffer_binding)
            .collect();
        let up_b: Vec<_> = global
            .up
            .iter()
            .map(wgpu::Buffer::as_entire_buffer_binding)
            .collect();
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::BufferArray(&gate_b),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::BufferArray(&up_b),
            },
            bind_buf(2, &x2),
            bind_buf(3, &csel),
            bind_buf(4, &mact),
        ];
        if !blocked {
            entries.push(bind_buf(5, &cwt));
        }
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qwen4-cold-gu"),
            layout: &pipe_gu.get_bind_group_layout(0),
            entries: &entries,
        })
    });
    let bg_gu_p = bc.get(s_gup, || {
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qwen4-cold-gu-p"),
            layout: &pipe_gu.get_bind_group_layout(1),
            entries: &[bind_buf(0, &gu_u)],
        })
    });
    pass.set_pipeline(pipe_gu);
    pass.set_bind_group(0, &bg_gu, &[]);
    pass.set_bind_group(1, &bg_gu_p, &[]);
    pass.dispatch_workgroups_indirect(&dev.cold_args, 0);
    let bg_dn = bc.get(s_dn, || {
        let down_b: Vec<_> = global
            .down
            .iter()
            .map(wgpu::Buffer::as_entire_buffer_binding)
            .collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qwen4-cold-dn"),
            layout: &pipe_dn.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::BufferArray(&down_b),
                },
                bind_buf(1, &mact),
                bind_buf(2, &csel),
                bind_buf(3, &cwt),
                bind_buf(4, &mo2),
            ],
        })
    });
    let bg_dn_p = bc.get(s_dnp, || {
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qwen4-cold-dn-p"),
            layout: &pipe_dn.get_bind_group_layout(1),
            entries: &[bind_buf(0, &dn_u)],
        })
    });
    pass.set_pipeline(pipe_dn);
    pass.set_bind_group(0, &bg_dn, &[]);
    pass.set_bind_group(1, &bg_dn_p, &[]);
    pass.dispatch_workgroups_indirect(&dev.cold_args, 16);
    // the inject takes its cold flags from `inj_flags`, rewritten at finalize
    inject_t(
        c,
        p,
        &mut pass,
        g,
        &dev.hyper,
        &mo,
        es(hidden * 4),
        &inj,
        es(g.hc * 4),
        (&coldvec, es(hidden * 4)),
        (&mo2, es(hidden * 4)),
        0,
        &dev.inj_flags,
        nt,
        bc,
        1,
    );
    drop(pass);
    true
}

/// Every token slot's pending inject has been encoded for this frame.
pub(crate) fn pending_done(dev: &mut Dev) {
    dev.pending_inject = false;
}

/// Fill in what the pending inject of the frame about to be submitted
/// needs: the admitted cold slots and weights of every token, the host's
/// completions, the indirect sizes of the cold pass and the inject flags.
/// A token without cold winners gets slot 0 at weight zero (the kernels
/// run over every token slot of the frame).
pub(crate) fn finalize_pending(
    dev: &Dev,
    g: &Geom,
    nt: usize,
    cold_host: &[Option<Vec<f32>>],
    cold_slots: &[Vec<(u32, f32)>],
) -> bool {
    let Some(c) = ctx() else { return false };
    let k = g.top_k;
    let k_full = k;
    let any_dev = cold_slots.iter().take(nt).any(|s| !s.is_empty());
    let any_host = cold_host.iter().take(nt).any(Option::is_some);
    // Slots per token in the cold pass: the frame's longest cold list
    // (`CMF_QWEN_COLD_K`, default) or always top_k. The padding slots carry
    // weight zero, so the down kernel's sums only lose trailing zero terms.
    let tpl = dev.cold_tpl.get().filter(|_| cold_k_env());
    let k = match tpl {
        Some(_) if any_dev => cold_slots
            .iter()
            .take(nt)
            .map(|s| s.len().min(k))
            .max()
            .unwrap_or(k)
            .max(1),
        _ => k,
    };
    if let Some((gu_w, dn_w)) = tpl
        && any_dev
        && dev.cold_k.get() != k
    {
        let (mut a, mut b) = (gu_w, dn_w);
        a[2] = k as u32;
        b[2] = k as u32;
        c.queue.write_buffer(&dev.cold_u[0], 0, bytemuck::cast_slice(&a));
        c.queue.write_buffer(&dev.cold_u[1], 0, bytemuck::cast_slice(&b));
        dev.cold_k.set(k);
    }
    if any_dev {
        let mut sel = vec![0u32; nt * k];
        let mut wt = vec![0.0f32; nt * k];
        for t in 0..nt {
            let s = cold_slots.get(t).map_or(&[][..], |v| v.as_slice());
            let n = s.len().min(k);
            for (i, &(sl, w)) in s.iter().take(n).enumerate() {
                sel[t * k + i] = sl;
                wt[t * k + i] = w;
            }
            let first = if n > 0 { sel[t * k] } else { 0 };
            for i in n..k {
                sel[t * k + i] = first;
            }
        }
        c.queue.write_buffer(
            &tbuf_exact(c, T_CSEL, k_full * 4, true),
            0,
            bytemuck::cast_slice(&sel),
        );
        c.queue.write_buffer(
            &tbuf_exact(c, T_CWT, k_full * 4, true),
            0,
            bytemuck::cast_slice(&wt),
        );
    }
    let blocked = any_dev
        && c.dsv4_global_moe
            .lock()
            .unwrap()
            .get(&dev.uid)
            .is_some_and(|b| blocked_experts(g, b.segments));
    let (div_gu, div) = match (blocked, pipes(c)) {
        (true, Some(p)) => (p.gu_rows(g), 4),
        (true, None) => (4, 4),
        (false, _) => (1, 1),
    };
    let args: [u32; 8] = if any_dev {
        [
            (g.inter / div_gu) as u32,
            k as u32,
            nt as u32,
            0,
            (g.hidden / div) as u32,
            nt as u32,
            1,
            0,
        ]
    } else {
        [0; 8]
    };
    let flags = u32::from(any_host) | (u32::from(any_dev) << 1);
    if dev.fin_last.get() != Some((args, flags)) {
        c.queue
            .write_buffer(&dev.cold_args, 0, bytemuck::cast_slice(&args));
        c.queue
            .write_buffer(&dev.inj_flags, 0, bytemuck::cast_slice(&[flags, 0, 0, 0]));
        dev.fin_last.set(Some((args, flags)));
    }
    if any_host {
        let coldvec = tbuf(c, T_COLDVEC, g.hidden * 4, true);
        let zeros = vec![0.0f32; g.hidden];
        for t in 0..nt {
            let row = cold_host
                .get(t)
                .and_then(|v| v.as_deref())
                .unwrap_or(&zeros);
            c.queue.write_buffer(
                &coldvec,
                (t * tstride(g.hidden * 4)) as u64,
                bytemuck::cast_slice(&row[..g.hidden]),
            );
        }
    }
    true
}

/// Submit the encoder and read `parts` back through one staging buffer and
/// one fence. Returns the concatenated bytes (each part 16-byte aligned).
pub(crate) fn submit_readback(
    mut enc: wgpu::CommandEncoder,
    parts: &[(&wgpu::Buffer, u64)],
) -> Option<Vec<u8>> {
    let c = ctx()?;
    let mut offs = Vec::with_capacity(parts.len());
    let mut total = 0u64;
    for (_, bytes) in parts {
        offs.push(total);
        total += bytes.div_ceil(16) * 16;
    }
    let mut sc = c.scratch.lock().unwrap();
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        total.max(16),
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "qwen4-stage",
    );
    flush_pass(&enc);
    for ((buf, bytes), off) in parts.iter().zip(&offs) {
        enc.copy_buffer_to_buffer(buf, 0, &stage, *off, *bytes);
    }
    submit(c, finish_enc(enc));
    let slice = stage.slice(..total.max(16));
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d2 = done.clone();
    slice.map_async(wgpu::MapMode::Read, move |_| {
        d2.store(true, std::sync::atomic::Ordering::Release);
    });
    if spin_wait() {
        let t0 = std::time::Instant::now();
        loop {
            let _ = c.device.poll(wgpu::PollType::Poll);
            if done.load(std::sync::atomic::Ordering::Acquire) {
                break;
            }
            if t0.elapsed() > std::time::Duration::from_millis(2) {
                if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
                    stage.unmap();
                    return None;
                }
                break;
            }
            std::hint::spin_loop();
        }
    } else if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        stage.unmap();
        return None;
    }
    let out = {
        let Ok(data) = slice.get_mapped_range() else {
            stage.unmap();
            return None;
        };
        let mut v = Vec::with_capacity(total as usize);
        for ((_, bytes), off) in parts.iter().zip(&offs) {
            let o = *off as usize;
            v.extend_from_slice(&data[o..o + *bytes as usize]);
            let pad = (bytes.div_ceil(16) * 16 - bytes) as usize;
            v.extend(std::iter::repeat_n(0u8, pad));
        }
        v
    };
    stage.unmap();
    drop(sc);
    Some(out)
}

/// The MTP cell input: R' = fc_hidden(rms_{hc·hidden}(R)·(1+w_h)) per stream
/// + fc_embedding(rms(embed(tok))·(1+w_e)), written into the MTP state's
/// token slot `tok`. `r` is the main model's final hyper state of the cell's
/// position (or the MTP's own output for a chained draft).
#[allow(clippy::too_many_arguments)]

/// The MTP cell's input: R' = fc_hidden(rms(R)·(1+w_h)) per stream +
/// fc_embedding(rms(embed(tok))·(1+w_e)), into token slot 0's hyper state.
/// `r` is the main model's last R or the previous cell's output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mtp_input(
    enc: &mut wgpu::CommandEncoder,
    dev: &Dev,
    side: &Arc<CmfModel>,
    g: &Geom,
    emb: Option<&[f32]>,
    enorm: &[f32],
    hnorm: &[f32],
    fc_e: usize,
    fc_h: usize,
    r: Rng,
) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(p) = pipes(c) else { return false };
    let (hidden, hc) = (g.hidden, g.hc);
    let hh = hc * hidden;
    let tok = 0usize;
    let bc = Bc {
        dev,
        li: LI_MTP_IN,
        tok,
        direct: true,
    };
    // the host's embedding row, or the one `encode_draft_gather` left here
    let embb = frame_buf_t(c, T_MEMB, tok, hidden * 4, true);
    if let Some(emb) = emb {
        c.queue
            .write_buffer(&embb, 0, bytemuck::cast_slice(&emb[..hidden]));
    }
    let en = frame_buf_t(c, T_MEN, tok, hidden * 4, false);
    let e = frame_buf_t(c, T_ME, tok, hidden * 4, false);
    let rn = frame_buf_t(c, T_MRN, tok, hh * 4, false);
    let rs: Vec<wgpu::Buffer> = (0..hc)
        .map(|sidx| frame_buf_t(c, T_MRS0 + sidx as u8, tok, hidden * 4, true))
        .collect();
    let hs: Vec<wgpu::Buffer> = (0..hc)
        .map(|sidx| frame_buf_t(c, T_MH0 + sidx as u8, tok, hidden * 4, false))
        .collect();
    let enw = const_buf(c, bytemuck::cast_slice(&enorm[..hidden]));
    let hnw = const_buf(c, bytemuck::cast_slice(&hnorm[..hh]));
    {
        let mut pass = begin_pass(enc);
        group_rmsnorm(c, p, &mut pass, &embb, &enw, &en, 1, hidden, g.eps, bc, 900);
        if mv(c, p, &mut pass, side, fc_e, &en, &e, bc, 901).is_none() {
            return false;
        }
        // one bind group per R source (the slot key tells them apart)
        let step = if r.len == 0 { 902 } else { 912 };
        dispatch_r(
            c,
            &mut pass,
            &p.group_rmsnorm,
            bc,
            step,
            "qwen4-gn-r",
            || {
                vec![
                    r.clone(),
                    Rng::all(&hnw),
                    Rng::all(&rn),
                    Rng::all(&uniform_u32x4(c, [1, hh as u32, g.eps.to_bits(), 0])),
                ]
            },
            (1, 1, 1),
        );
    }
    // the normalized streams, one per projection input
    flush_pass(&*enc);
    for (sidx, rsb) in rs.iter().enumerate() {
        enc.copy_buffer_to_buffer(&rn, (sidx * hidden * 4) as u64, rsb, 0, (hidden * 4) as u64);
    }
    let mut pass = begin_pass(enc);
    for sidx in 0..hc {
        if mv(
            c,
            p,
            &mut pass,
            side,
            fc_h,
            &rs[sidx],
            &hs[sidx],
            bc,
            903 + sidx as u16,
        )
        .is_none()
        {
            return false;
        }
    }
    if hc != 4 {
        return false;
    }
    dispatch_r(
        c,
        &mut pass,
        &p.mtp_fuse,
        bc,
        910,
        "qwen4-mtp-fuse",
        || {
            vec![
                Rng::all(&hs[0]),
                Rng::all(&hs[1]),
                Rng::all(&hs[2]),
                Rng::all(&hs[3]),
                Rng::all(&e),
                Rng::row(&dev.hyper, 0, hh * 4),
                Rng::all(&uniform_u32x4(c, [hc as u32, hidden as u32, 0, 0])),
            ]
        },
        ((hh as u32).div_ceil(256), 1, 1),
    );
    drop(pass);
    true
}

/// The draft chain's token ids: slot 0 the host's input token, slot j+1
/// the argmax of cell j (what the chain hands back as its drafts).
pub(crate) fn draft_ids(host_tok: u32) -> Option<wgpu::Buffer> {
    let c = ctx()?;
    let b = frame_buf_t(c, T_MIDS, 0, (TMAX + 1) * 4, true);
    c.queue
        .write_buffer(&b, 0, bytemuck::cast_slice(&[host_tok]));
    Some(b)
}

/// Re-embed `ids[st]` from the main model's q8_2f embedding table into the
/// MTP input slot (what `encode_mtp_input` reads when given no host row).
/// None when the table is not resident on the card.
pub(crate) fn encode_draft_gather(
    enc: &mut wgpu::CommandEncoder,
    dev: &Dev,
    model: &Arc<CmfModel>,
    g: &Geom,
    embed_idx: usize,
    ids: &wgpu::Buffer,
    st: usize,
) -> Option<()> {
    let c = ctx()?;
    let p = pipes(c)?;
    let w = weight(c, model, embed_idx)?;
    if w.dtype != TensorDtype::Q8_2f || w.cols != g.hidden || w.cols % 4 != 0 {
        return None;
    }
    let embb = frame_buf_t(c, T_MEMB, 0, g.hidden * 4, true);
    let bc = Bc {
        dev,
        li: LI_MTP_IN,
        tok: st,
        direct: true,
    };
    let mut pass = begin_pass(enc);
    dispatch(
        c,
        &mut pass,
        &p.embed_gather_q82,
        bc,
        930,
        "qwen4-embed-gather",
        || {
            vec![
                w.buf.clone(),
                ids.clone(),
                embb.clone(),
                uniform_u32x4(c, [w.cols as u32, w.rows as u32, st as u32, 0]),
            ]
        },
        ((g.hidden as u32).div_ceil(256), 1, 1),
    );
    drop(pass);
    Some(())
}

/// argmax over `vocab` logits in row 0 of `lb`, into `ids[st]` (the main
/// module's two-stage reduction).
pub(crate) fn encode_argmax(
    enc: &mut wgpu::CommandEncoder,
    dev: &Dev,
    lb: &wgpu::Buffer,
    vocab: usize,
    ids: &wgpu::Buffer,
    st: usize,
) -> Option<()> {
    let c = ctx()?;
    const PARTS: u32 = 512;
    let pv = frame_buf_t(c, T_AMPV, 0, PARTS as usize * 4, false);
    let pi = frame_buf_t(c, T_AMPI, 0, PARTS as usize * 4, false);
    let bc = Bc {
        dev,
        li: LI_MTP_IN,
        tok: st,
        direct: true,
    };
    let u = uniform_u32x4(c, [vocab as u32, PARTS, st as u32, 0]);
    let mut pass = begin_pass(enc);
    dispatch(
        c,
        &mut pass,
        &c.argmax_part,
        bc,
        931,
        "qwen4-argmax-part",
        || vec![lb.clone(), pv.clone(), pi.clone(), u.clone()],
        (PARTS, 1, 1),
    );
    dispatch(
        c,
        &mut pass,
        &c.argmax_final,
        bc,
        932,
        "qwen4-argmax-final",
        || vec![pv.clone(), pi.clone(), ids.clone(), u.clone()],
        (1, 1, 1),
    );
    drop(pass);
    Some(())
}

/// Copy `bytes` of `src` from `src_off` into the current stage at `off`
/// (ends any open pass). False, with nothing recorded, when either range
/// is out of bounds: wgpu would only report that when the encoder is
/// finished, as a validation error that takes the process down.
#[must_use]
pub(crate) fn copy_to_stage(
    enc: &mut wgpu::CommandEncoder,
    dev: &Dev,
    src: &wgpu::Buffer,
    src_off: u64,
    off: u64,
    bytes: u64,
) -> bool {
    let fits = |o: u64, size: u64| o.checked_add(bytes).is_some_and(|end| end <= size);
    if !fits(off, dev.stage().size()) || !fits(src_off, src.size()) {
        return false;
    }
    flush_pass(&*enc);
    enc.copy_buffer_to_buffer(src, src_off, dev.stage(), off, bytes);
    true
}

// ── GPU stage timestamps (`CMF_QWEN_TS=1` with `CMF_GPU_TS=2`): a probe,
// not a release path. Every frame writes the same fixed marks (`TS_*`), so
// slot i always means the same frame point; the frame resolves them into
// the context's timestamp stage, which the host reads after the fence. ──

/// Mark slots of a frame: chain head, after the pending cold pass, after
/// PLE, after the attention mix, after the mixer, after the MoE mix, after
/// routing, after the experts, after the stage copies.
pub(crate) const TS_MARKS: u32 = 13;

pub(crate) fn ts_probe() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CMF_QWEN_TS").as_deref() == Ok("1")
            && ctx().is_some_and(|c| {
                c.ts_query.is_some()
                    && c.device
                        .features()
                        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            })
    })
}

/// Write mark `i` inside the frame's open pass.
fn ts_pass(pass: &mut wgpu::ComputePass<'_>, i: u32) {
    if !ts_probe() {
        return;
    }
    if let Some((qs, _, _)) = ctx().and_then(|c| c.ts_query.as_ref()) {
        pass.write_timestamp(qs, i);
    }
}

/// Write mark `i` on the encoder (through the merged pass).
pub(crate) fn ts_mark(enc: &mut wgpu::CommandEncoder, i: u32) {
    if !ts_probe() {
        return;
    }
    let mut pass = begin_pass(enc);
    ts_pass(&mut pass, i);
}

/// Resolve the frame's marks into the timestamp stage (end of the frame).
pub(crate) fn ts_resolve(enc: &mut wgpu::CommandEncoder) {
    if !ts_probe() {
        return;
    }
    if let Some((qs, resolve, tstage)) = ctx().and_then(|c| c.ts_query.as_ref()) {
        flush_pass(&*enc);
        enc.resolve_query_set(qs, 0..TS_MARKS, resolve, 0);
        enc.copy_buffer_to_buffer(resolve, 0, tstage, 0, TS_MARKS as u64 * 8);
    }
}

/// The marks of the frame whose fence just passed, in nanoseconds.
pub(crate) fn ts_read() -> Option<Vec<f64>> {
    if !ts_probe() {
        return None;
    }
    let c = ctx()?;
    let (_, _, tstage) = c.ts_query.as_ref()?;
    let slice = tstage.slice(..TS_MARKS as u64 * 8);
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d2 = done.clone();
    slice.map_async(wgpu::MapMode::Read, move |_| {
        d2.store(true, std::sync::atomic::Ordering::Release)
    });
    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    if !done.load(std::sync::atomic::Ordering::Acquire) {
        tstage.unmap();
        return None;
    }
    let v: Vec<f64> = {
        let data = slice.get_mapped_range().ok()?;
        data.chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as f64 * c.ts_period as f64)
            .collect()
    };
    tstage.unmap();
    Some(v)
}

/// Debug: every token slot's hyper state (`hc·hidden` floats each).
pub(crate) fn read_hyper(dev: &Dev, ntok: usize, hh: usize) -> Option<Vec<Vec<f32>>> {
    let enc = new_encoder("qwen4-dump")?;
    let bytes = submit_readback(enc, &[(&dev.hyper, (ntok * hh * 4) as u64)])?;
    Some(
        (0..ntok)
            .map(|t| {
                bytes[t * hh * 4..(t + 1) * hh * 4]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            })
            .collect(),
    )
}

/// Frame buffers a parity check reads back after a layer (token slot 0,
/// which sits at offset 0 of every strided buffer): the attention input,
/// the mixer output, the MoE input, the resident MoE output and the hyper
/// state.
pub(crate) fn tap_bufs(dev: &Dev, g: &Geom) -> [wgpu::Buffer; 5] {
    let c = ctx().expect("tap needs the context the frame ran on");
    [
        tbuf(c, T_X, g.hidden * 4, false),
        tbuf(c, T_BLK, g.hidden * 4, false),
        tbuf(c, T_X2, g.hidden * 4, false),
        tbuf(c, T_MO, g.hidden * 4, true),
        dev.hyper.clone(),
    ]
}

/// A whole buffer as a bind group entry.
pub(crate) fn whole(b: &wgpu::Buffer) -> Rng {
    Rng::all(b)
}

/// Submit `pre` (staged expert uploads) and `cb` in one queue submission; the index to
/// wait on.
pub(crate) fn submit_indexed(pre: Option<wgpu::CommandBuffer>, cb: wgpu::CommandBuffer) -> Option<wgpu::SubmissionIndex> {
    let c = ctx()?;
    note_submit(c);
    Some(c.queue.submit(pre.into_iter().chain(std::iter::once(cb))))
}

/// Block until submission `idx` has run on the card.
pub(crate) fn wait_submission(idx: wgpu::SubmissionIndex) -> bool {
    let Some(c) = ctx() else { return false };
    c.device.poll(wgpu::PollType::Wait { submission_index: Some(idx), timeout: None }).is_ok()
}

/// Whether a GPTQ calibration capture is recording layer rows (they must come in frame
/// order, which a streamed prompt does not keep).
pub(crate) fn capture_on() -> bool {
    CAPTURE.lock().map(|g| g.is_some()).unwrap_or(true)
}

/// Submit one finished command buffer on its own.
pub(crate) fn submit_cb(cb: wgpu::CommandBuffer) {
    if let Some(c) = ctx() {
        submit(c, cb);
    }
}

/// Submit without a readback (a token whose logits nobody wants).
pub(crate) fn submit_only(enc: wgpu::CommandEncoder) {
    if let Some(c) = ctx() {
        submit(c, finish_enc(enc));
    }
}

/// Pass merging for a frame encoder: every `begin_pass` on it hands back
/// one open pass until something uses the encoder directly (a copy, the
/// submit), which flushes it first. A pass boundary costs ~9 µs on the
/// Vulkan stack; a frame used to open three or four per layer. Drop it
/// before handing the encoder on.
pub(crate) struct MergeGuard {
    _g: super::PassMergeGuard,
}

pub(crate) fn merge_guard(enc: &wgpu::CommandEncoder) -> MergeGuard {
    MergeGuard {
        _g: super::PassMergeGuard::new(enc),
    }
}

// ── pinned staging ring for expert admissions ──

/// Two host-visible staging buffers: admissions memcpy into the mapped one
/// from parallel threads (no queue work per expert), and one copy per
/// matrix is recorded at the next flush, submitted ahead of the frame that
/// reads the slots. `write_buffer` per expert measured ~0.25 ms each and
/// serialized across threads on the NVIDIA Vulkan stack; 150 misses a
/// token at a 12 GB budget were 38 ms of a 70 ms token.
pub(crate) struct Stager {
    bufs: [wgpu::Buffer; 2],
    cap: u64,
    cur: usize,
    used: std::sync::atomic::AtomicU64,
    ready: [std::sync::Arc<std::sync::atomic::AtomicBool>; 2],
    copies: std::sync::Mutex<Vec<(u64, wgpu::Buffer, u64, u64)>>,
    pub(crate) staged: std::sync::atomic::AtomicU64,
    /// `take` handed out copies whose ring half `rearm` has to remap
    armed: bool,
}

impl Stager {
    /// `cap_mb` per buffer (two of them, pinned host memory).
    pub(crate) fn new(cap_mb: u64) -> Option<Self> {
        let c = ctx()?;
        if cap_mb == 0 {
            return None;
        }
        let cap = cap_mb << 20;
        let mk = || {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("qwen4-stage-ring"),
                size: cap,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            })
        };
        let flag = || std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        Some(Self {
            bufs: [mk(), mk()],
            cap,
            cur: 0,
            used: std::sync::atomic::AtomicU64::new(0),
            ready: [flag(), flag()],
            copies: std::sync::Mutex::new(Vec::new()),
            staged: std::sync::atomic::AtomicU64::new(0),
            armed: false,
        })
    }

    /// Copy `src` into `dst[dst_off..]` through the ring. False when the
    /// ring is full or still mapping (the caller uploads directly).
    pub(crate) fn put(&self, dst: &wgpu::Buffer, dst_off: u64, src: &[u8]) -> bool {
        use std::sync::atomic::Ordering;
        if !self.ready[self.cur].load(Ordering::Acquire) {
            return false;
        }
        let len = src.len() as u64;
        if len == 0 || len % 4 != 0 || dst_off % 4 != 0 {
            return false;
        }
        let rounded = len.div_ceil(256) * 256;
        let off = self.used.fetch_add(rounded, Ordering::AcqRel);
        if off + rounded > self.cap {
            self.used.fetch_sub(rounded, Ordering::AcqRel);
            return false;
        }
        {
            let Ok(mut view) = self.bufs[self.cur]
                .slice(off..off + len)
                .get_mapped_range_mut()
            else {
                return false;
            };
            view.copy_from_slice(src);
        }
        self.copies
            .lock()
            .unwrap()
            .push((off, dst.clone(), dst_off, len));
        self.staged.fetch_add(len, Ordering::Relaxed);
        true
    }

    /// Submit the pending copies ahead of the next frame, hand the buffer
    /// to the card and start mapping it again; the other one fills next.
    pub(crate) fn flush(&mut self) {
        let Some(c) = ctx() else { return };
        if let Some(cb) = self.take() {
            submit(c, cb);
            self.rearm();
        }
    }

    /// The staged copies as one command buffer, for the caller to submit
    /// (ahead of its frame, in the same queue submission); `rearm` must
    /// follow that submit. `None` when nothing was staged.
    pub(crate) fn take(&mut self) -> Option<wgpu::CommandBuffer> {
        use std::sync::atomic::Ordering;
        let c = ctx()?;
        let copies = std::mem::take(&mut *self.copies.lock().unwrap());
        if copies.is_empty() {
            self.used.store(0, Ordering::Release);
            return None;
        }
        let buf = self.bufs[self.cur].clone();
        buf.unmap();
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qwen4-stage-copies"),
            });
        for (so, dst, doff, len) in &copies {
            enc.copy_buffer_to_buffer(&buf, *so, dst, *doff, *len);
        }
        self.armed = true;
        Some(enc.finish())
    }

    /// Map the ring half whose copies were just submitted for the next
    /// round of writes, and switch to the other half.
    pub(crate) fn rearm(&mut self) {
        use std::sync::atomic::Ordering;
        if !std::mem::take(&mut self.armed) {
            return;
        }
        let cur = self.cur;
        let flag = self.ready[cur].clone();
        flag.store(false, Ordering::Release);
        self.bufs[cur]
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |r| {
                if r.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
        self.cur ^= 1;
        self.used.store(0, Ordering::Release);
    }
}

/// Upload one expert's (gate, up, down) bytes into arena `slot` of the
/// model's global bank: through the staging ring when there is one with
/// room, else (part by part) straight through the queue.
pub(crate) fn upload_expert_parts(
    st: Option<&Stager>,
    model: &Arc<CmfModel>,
    slot: usize,
    parts: [&[u8]; 3],
) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(b) = c.dsv4_global_moe.lock().unwrap().get(&model.uid()).cloned() else {
        return false;
    };
    if slot >= b.capacity
        || parts[0].len() != b.gu_len
        || parts[1].len() != b.gu_len
        || parts[2].len() != b.d_len
    {
        return false;
    }
    let seg = slot / b.segment_slots;
    let local = slot % b.segment_slots;
    let dst = [
        (&b.gate[seg], (local * b.gu_len) as u64),
        (&b.up[seg], (local * b.gu_len) as u64),
        (&b.down[seg], (local * b.d_len) as u64),
    ];
    for (part, (buf, off)) in parts.iter().zip(dst) {
        if !st.is_some_and(|s| s.put(buf, off, part)) {
            c.queue.write_buffer(buf, off, part);
        }
    }
    true
}

/// Hand the queue's pending `write_buffer` data to the GPU and wait, so
/// its staging memory is released (a bulk upload would otherwise hold all
/// of it until the next frame).
pub(crate) fn flush_writes() {
    if let Some(c) = ctx() {
        note_submit(c);
        c.queue.submit(std::iter::empty());
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

/// Stage one expert's three matrices into arena `slot` of the model's
/// global bank. False when the ring cannot take it (nothing was staged
/// that a direct upload would not simply repeat).
pub(crate) fn stage_expert(
    st: &Stager,
    model: &Arc<CmfModel>,
    slot: usize,
    t: (usize, usize, usize),
) -> bool {
    let Some(c) = ctx() else { return false };
    let Some(b) = c.dsv4_global_moe.lock().unwrap().get(&model.uid()).cloned() else {
        return false;
    };
    if slot >= b.capacity {
        return false;
    }
    let seg = slot / b.segment_slots;
    let local = slot % b.segment_slots;
    let bytes = model.primary_bytes();
    let put = |buf: &wgpu::Buffer, idx: usize, plen: usize| -> bool {
        let Some(e) = model.tensors.get(idx) else {
            return false;
        };
        if e.nbytes as usize != plen {
            return false;
        }
        let Some(abs) = model.entry_abs_offset(e) else {
            return false;
        };
        let Some(src) = bytes.get(abs..abs + plen) else {
            return false;
        };
        st.put(buf, (local * plen) as u64, src)
    };
    put(&b.gate[seg], t.0, b.gu_len)
        && put(&b.up[seg], t.1, b.gu_len)
        && put(&b.down[seg], t.2, b.d_len)
}

// ── host-heap expert tier: admissions as DMA copies ──

/// Routed experts in GPU-visible host memory (`host_mem::sysmem_buffer`,
/// the cached type), one slot per expert: gate, up and down at 256-byte
/// aligned offsets. An admission whose expert sits here is three
/// `copy_buffer_to_buffer` commands, recorded now and submitted ahead of the
/// frame that reads the arena slot — the card's copy engine pulls the bytes
/// over PCIe and the CPU copies nothing. A miss copies the expert once from
/// the mapping into a free slot (CLOCK eviction once the tier is full)
/// first: parallel writes into system RAM, not into the PCIe window.
///
/// Why (RTX 3090, PCIe 4.0 x16, no resizable BAR, measured with the
/// `qwen4_xfer_bench` example): `write_buffer` stages every expert in the
/// 256 MB BAR window and serializes across threads (an 8-token prompt frame
/// spent ~80 ms admitting ~410 experts, ~9 GB/s, the card idle meanwhile);
/// a DMA copy from host memory runs at 24.5 GB/s and a parallel memcpy from
/// the page-cached mapping into host memory at 45-60 GB/s.
///
/// Segments (1 GiB) are allocated by a background thread — the driver pins
/// and clears ~0.3 s per GiB — so the tier comes online in steps; an
/// admission it cannot place yet takes the `write_buffer` path. The thread
/// starts after the first completed submission (`done`), so the first
/// encode creates its resources unhindered; the pinning still holds up the
/// frames that run alongside it (~7 s in all for 28 GiB on the 3090), which
/// is why the tier is not on by default (§8.7 of the device doc).
pub(crate) struct HostTier {
    bank: Arc<Dsv4GlobalMoeBufs>,
    segs: Arc<Vec<std::sync::OnceLock<super::host_mem::SysBuf>>>,
    grow_started: std::sync::atomic::AtomicBool,
    seg_slots: usize,
    stride: u64,
    off: [u64; 3],
    cap: usize,
    online: Arc<std::sync::atomic::AtomicUsize>,
    /// no more segments are coming (all allocated, or the driver refused
    /// one: the tier then stays at `online` slots)
    grown: Arc<std::sync::atomic::AtomicBool>,
    meta: std::sync::Mutex<TierMeta>,
    copies: std::sync::Mutex<Vec<(u32, u32)>>,
    /// copy batches handed to the queue (`take`) / known complete (`done`)
    taken: std::sync::atomic::AtomicU64,
    completed: std::sync::atomic::AtomicU64,
    pub(crate) hits: std::sync::atomic::AtomicU64,
    pub(crate) fills: std::sync::atomic::AtomicU64,
    pub(crate) fill_ns: std::sync::atomic::AtomicU64,
    pub(crate) misses: std::sync::atomic::AtomicU64,
    pub(crate) bg_fills: std::sync::atomic::AtomicU64,
}

struct TierMeta {
    slot_of: Vec<u32>,
    key_of: Vec<u32>,
    ready: Vec<bool>,
    refbit: Vec<bool>,
    /// the copy batch that last read the slot: it may be evicted only once
    /// that batch is complete
    epoch: Vec<u64>,
    bump: usize,
    hand: usize,
}

impl HostTier {
    /// A tier of at most `cap_bytes` over `n_keys` experts of `model`'s
    /// global bank. None off Vulkan, without host-visible system memory, or
    /// when the first segment cannot be allocated.
    pub(crate) fn new(model: &CmfModel, n_keys: usize, cap_bytes: u64) -> Option<Arc<Self>> {
        use std::sync::atomic::{AtomicU64, AtomicUsize};
        let c = ctx()?;
        let bank = c.dsv4_global_moe.lock().unwrap().get(&model.uid()).cloned()?;
        let al = |n: usize| (n as u64).div_ceil(256) * 256;
        let off = [0, al(bank.gu_len), 2 * al(bank.gu_len)];
        let stride = off[2] + al(bank.d_len);
        let seg_bytes = std::env::var("CMF_QWEN_HTIER_SEG_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(1 << 30, |mb| mb << 20)
            .min(c.device.limits().max_buffer_size);
        let seg_slots = (seg_bytes / stride) as usize;
        let cap = ((cap_bytes / stride) as usize).min(n_keys).min(u32::MAX as usize - 1);
        if seg_slots == 0 || cap < 64 || n_keys >= u32::MAX as usize {
            return None;
        }
        let nseg = cap.div_ceil(seg_slots);
        let seg_len = seg_slots as u64 * stride;
        let first = super::host_mem::sysmem_buffer(&c.device, seg_len, true)?;
        let segs: Arc<Vec<std::sync::OnceLock<super::host_mem::SysBuf>>> =
            Arc::new((0..nseg).map(|_| std::sync::OnceLock::new()).collect());
        let _ = segs[0].set(first);
        let online = Arc::new(AtomicUsize::new(seg_slots.min(cap)));
        Some(Arc::new(Self {
            bank,
            segs,
            grow_started: std::sync::atomic::AtomicBool::new(nseg <= 1),
            seg_slots,
            stride,
            off,
            cap,
            online,
            grown: Arc::new(std::sync::atomic::AtomicBool::new(nseg <= 1)),
            meta: std::sync::Mutex::new(TierMeta {
                slot_of: vec![u32::MAX; n_keys],
                key_of: vec![u32::MAX; cap],
                ready: vec![false; cap],
                refbit: vec![false; cap],
                epoch: vec![0; cap],
                bump: 0,
                hand: 0,
            }),
            copies: std::sync::Mutex::new(Vec::new()),
            taken: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            fills: AtomicU64::new(0),
            fill_ns: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bg_fills: AtomicU64::new(0),
        }))
    }

    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }

    /// Slots whose segment is allocated.
    pub(crate) fn online(&self) -> usize {
        self.online.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Bytes per slot.
    pub(crate) fn stride(&self) -> u64 {
        self.stride
    }

    /// Every slot is taken (further fills evict).
    pub(crate) fn full(&self) -> bool {
        let limit = if self.grown.load(std::sync::atomic::Ordering::Acquire) {
            self.online()
        } else {
            self.cap
        };
        self.meta.lock().unwrap().bump >= limit
    }

    /// The tier has (or is writing) expert `key`.
    pub(crate) fn has(&self, key: usize) -> bool {
        self.meta
            .lock()
            .unwrap()
            .slot_of
            .get(key)
            .is_some_and(|&s| s != u32::MAX)
    }

    /// Fill the tier in the background with `order` (keys, most wanted
    /// first; `triples[key]` its tensors in `model`) on `threads` threads,
    /// while it has free slots. Admissions that miss before the loader
    /// reaches their expert still fill it themselves.
    pub(crate) fn start_fill(
        self: &Arc<Self>,
        model: Arc<CmfModel>,
        triples: Arc<Vec<(usize, usize, usize)>>,
        order: Vec<usize>,
        threads: usize,
    ) {
        let order = Arc::new(order);
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for _ in 0..threads.max(1) {
            let (me, order, next, model, triples) = (
                Arc::downgrade(self),
                order.clone(),
                next.clone(),
                model.clone(),
                triples.clone(),
            );
            let _ = std::thread::Builder::new()
                .name("qwen4-tier-fill".into())
                .spawn(move || {
                    let bytes = model.primary_bytes();
                    let part = |i: usize| -> Option<&[u8]> {
                        let e = model.tensors.get(i)?;
                        let abs = model.entry_abs_offset(e)?;
                        bytes.get(abs..abs + e.nbytes as usize)
                    };
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&key) = order.get(i) else { return };
                        let Some(t) = me.upgrade() else { return };
                        if t.has(key) {
                            continue;
                        }
                        let Some(&(g, u, d)) = triples.get(key) else {
                            continue;
                        };
                        let (Some(g), Some(u), Some(d)) = (part(g), part(u), part(d)) else {
                            continue;
                        };
                        loop {
                            // never evict for a guess: stop once full
                            if t.full() {
                                return;
                            }
                            if t.fill(key, None, [g, u, d]) || t.has(key) {
                                break;
                            }
                            // the next segment is still being allocated
                            std::thread::sleep(std::time::Duration::from_millis(5));
                        }
                    }
                });
        }
    }

    fn queue_copy(&self, m: &mut TierMeta, s: usize, arena_slot: usize) {
        m.refbit[s] = true;
        m.epoch[s] = self.taken.load(std::sync::atomic::Ordering::Acquire) + 1;
        self.copies
            .lock()
            .unwrap()
            .push((s as u32, arena_slot as u32));
    }

    /// Queue the copy of expert `key` into `arena_slot` when the tier holds
    /// it. False when it does not (or it is still being written).
    pub(crate) fn hit(&self, key: usize, arena_slot: usize) -> bool {
        if arena_slot >= self.bank.capacity {
            return false;
        }
        let mut m = self.meta.lock().unwrap();
        let Some(&s) = m.slot_of.get(key) else {
            return false;
        };
        if s == u32::MAX || !m.ready[s as usize] {
            return false;
        }
        self.queue_copy(&mut m, s as usize, arena_slot);
        drop(m);
        self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Copy expert `key` (`parts`: gate, up, down) into a tier slot, then,
    /// with `arena_slot`, queue its copy into the arena. False when the
    /// tier has no slot for it now (the caller uploads it another way).
    pub(crate) fn fill(&self, key: usize, arena_slot: Option<usize>, parts: [&[u8]; 3]) -> bool {
        use std::sync::atomic::Ordering;
        let b = &self.bank;
        if parts[0].len() != b.gu_len
            || parts[1].len() != b.gu_len
            || parts[2].len() != b.d_len
            || arena_slot.is_some_and(|a| a >= b.capacity)
        {
            return false;
        }
        let t0 = std::time::Instant::now();
        let s = {
            let mut m = self.meta.lock().unwrap();
            if m.slot_of.get(key).is_none_or(|&s| s != u32::MAX) {
                return false;
            }
            let online = self.online();
            let s = if m.bump < online {
                m.bump += 1;
                m.bump - 1
            } else if online < self.cap && !self.grown.load(Ordering::Acquire) {
                // more segments are coming: do not evict yet
                self.misses.fetch_add(1, Ordering::Relaxed);
                return false;
            } else {
                // CLOCK over the whole tier: skip slots being written, slots
                // a pending copy reads, and (once) recently used ones
                let done = self.completed.load(Ordering::Acquire);
                let mut pick = None;
                for _ in 0..2 * self.cap {
                    let s = m.hand;
                    m.hand = (m.hand + 1) % self.cap;
                    if !m.ready[s] || m.epoch[s] > done {
                        continue;
                    }
                    if m.refbit[s] {
                        m.refbit[s] = false;
                        continue;
                    }
                    pick = Some(s);
                    break;
                }
                let Some(s) = pick else {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    return false;
                };
                let old = m.key_of[s] as usize;
                if let Some(o) = m.slot_of.get_mut(old) {
                    *o = u32::MAX;
                }
                s
            };
            m.key_of[s] = key as u32;
            m.slot_of[key] = s as u32;
            m.ready[s] = false;
            s
        };
        let Some(seg) = self.segs[s / self.seg_slots].get() else {
            // cannot happen (online covers it); undo
            let mut m = self.meta.lock().unwrap();
            m.slot_of[key] = u32::MAX;
            m.key_of[s] = u32::MAX;
            return false;
        };
        let base = (s % self.seg_slots) as u64 * self.stride;
        let ok = (0..3).all(|i| seg.write(base + self.off[i], parts[i]));
        let mut m = self.meta.lock().unwrap();
        if !ok {
            m.slot_of[key] = u32::MAX;
            m.key_of[s] = u32::MAX;
            return false;
        }
        m.ready[s] = true;
        if let Some(a) = arena_slot {
            self.queue_copy(&mut m, s, a);
        } else {
            m.refbit[s] = true;
        }
        drop(m);
        if arena_slot.is_some() {
            self.fills.fetch_add(1, Ordering::Relaxed);
            self.fill_ns
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        } else {
            self.bg_fills.fetch_add(1, Ordering::Relaxed);
        }
        true
    }

    /// The queued copies as one command buffer, to go ahead of the frame
    /// that reads their arena slots (same queue submission). None when
    /// nothing is queued.
    pub(crate) fn take(&self) -> Option<wgpu::CommandBuffer> {
        let c = ctx()?;
        let m = self.meta.lock().unwrap();
        let copies = std::mem::take(&mut *self.copies.lock().unwrap());
        if copies.is_empty() {
            return None;
        }
        let b = &self.bank;
        let mut enc = c
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qwen4-host-tier-copies"),
            });
        for (s, a) in copies {
            let (s, a) = (s as usize, a as usize);
            let Some(seg) = self.segs[s / self.seg_slots].get() else {
                continue;
            };
            let base = (s % self.seg_slots) as u64 * self.stride;
            let (aseg, local) = (a / b.segment_slots, a % b.segment_slots);
            let dst = [
                (&b.gate[aseg], (local * b.gu_len) as u64, b.gu_len as u64),
                (&b.up[aseg], (local * b.gu_len) as u64, b.gu_len as u64),
                (&b.down[aseg], (local * b.d_len) as u64, b.d_len as u64),
            ];
            for (i, (buf, doff, len)) in dst.into_iter().enumerate() {
                enc.copy_buffer_to_buffer(&seg.buffer, base + self.off[i], buf, doff, len);
            }
        }
        self.taken
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        drop(m);
        Some(enc.finish())
    }

    /// Every batch `take` handed out so far has executed (the caller waited
    /// on a submission that followed them).
    pub(crate) fn done(&self) {
        use std::sync::atomic::Ordering;
        self.completed
            .store(self.taken.load(Ordering::Acquire), Ordering::Release);
        if !self.grow_started.swap(true, Ordering::AcqRel) {
            self.grow();
        }
    }

    /// Allocate the segments after the first in the background.
    fn grow(&self) {
        use std::sync::atomic::Ordering;
        let Some(c) = ctx() else {
            self.grown.store(true, Ordering::Release);
            return;
        };
        let (segs, online, grown) = (self.segs.clone(), self.online.clone(), self.grown.clone());
        let (seg_slots, cap) = (self.seg_slots, self.cap);
        let seg_len = seg_slots as u64 * self.stride;
        let spawned = std::thread::Builder::new()
            .name("qwen4-host-tier".into())
            .spawn(move || {
                for i in 1..segs.len() {
                    let Some(b) = super::host_mem::sysmem_buffer(&c.device, seg_len, true) else {
                        break;
                    };
                    let _ = segs[i].set(b);
                    online.store(((i + 1) * seg_slots).min(cap), Ordering::Release);
                }
                grown.store(true, Ordering::Release);
            });
        if spawned.is_err() {
            self.grown.store(true, Ordering::Release);
        }
    }
}

/// A frame salt for cell `j` of a chain that records several positions
/// before one submit: every mutable per-position uniform and cached bind
/// group gets its own identity while the guard lives. Salt 0 (dropped) is
/// the plain frame.
pub(crate) struct FrameSalt {
    _g: super::Dsv4FrameSalt,
}

pub(crate) fn frame_salt(j: usize) -> FrameSalt {
    FrameSalt {
        _g: super::Dsv4FrameSalt::enter(j + 1),
    }
}

pub(crate) fn new_encoder(label: &'static str) -> Option<wgpu::CommandEncoder> {
    let c = ctx()?;
    Some(
        c.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) }),
    )
}

#[cfg(test)]
mod shader_tests {
    /// The module the device path compiles: both sources, every entry
    /// point the pipelines name.
    /// The subgroup module as the device builds it (the qwen4 sources first).
    #[test]
    fn qwen4_sg_module_validates() {
        let src = format!("{}{}{}", super::QWEN4_WGSL, super::QWEN4T_WGSL, super::QWEN4_SG_WGSL);
        let module = wgpu::naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("qwen4 subgroup WGSL must parse: {}", e.emit_to_string(&src)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("qwen4 subgroup WGSL must validate: {}", e.emit_to_string(&src)));
        for ep in ["q4_dn_q4tp4s", "q4_dn_q4tp4sv", "q4_dn_q4tp4sv128", "q4_q82_sg", "q4_q82_sg1", "q4t_route2"] {
            assert!(module.entry_points.iter().any(|e| e.name == ep), "entry point {ep}");
        }
    }

    #[test]
    fn qwen4_mm_shader_validates() {
        let src = super::QWEN4_MM_WGSL;
        let module = wgpu::naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("qwen4 mm WGSL must parse: {}", e.emit_to_string(src)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("qwen4 mm WGSL must validate: {}", e.emit_to_string(src)));
    }

    #[test]
    fn qwen4_shaders_validate() {
        let src = format!("{}{}", super::QWEN4_WGSL, super::QWEN4T_WGSL);
        let module = wgpu::naga::front::wgsl::parse_str(&src).expect("qwen4 WGSL parses");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("qwen4 WGSL validates");
        for ep in [
            "q4_group_rmsnorm",
            "q4_f16_matvec",
            "q4_ple_gate",
            "q4_ple_conv",
            "q4_qsa_block_key",
            "q4_qsa_idx_build",
            "q4_qsa_attend",
            "q4_gate",
            "q4_miss",
            "q4_mtp_fuse",
            "q4t_group_rmsnorm",
            "q4t_f16_pair",
            "q4t_hc_upfold",
            "q4t_q82_matvec",
            "q4t_inject",
            "q4t_gdn_norm",
            "q4t_route",
            "q4_embed_gather_q82",
            "q4_gu_q2tp4",
            "q4_gu_q4tp4",
            "q4_gu_q4tp4v",
            "q4_dn_q4tp4",
            "q4t_hc_down",
            "q4t_hc_upfold2",
            "q4t_rope",
            "q4t_kv_append",
            "q4t_rawk",
            "q4t_ix_scores",
            "q4t_ix_scores_l",
            "q4t_topk_r",
            "q4t_topk",
            "q4t_idx_build",
            "q4t_attend",
            "q4t_attend_sp",
            "q4t_attend_sp2",
            "q4t_attend_cmb",
            "q4t_f16_pair_mm",
            "q4t_pair_sum",
            "q4g_gu_q4tp4",
            "q4g_dn_q4tp4",
            "q4g_sum",
            "q4g_sort",
            "q4g_segs",
            "q4g_cs_clear",
            "q4g_cs_count",
            "q4g_cs_scan",
            "q4g_cs_scatter",
            "q4t_hc_fold_t",
            "q4t_o1_pack",
            "q4t_o1_gate",
            "q4t_tube_gu",
            "q4t_tube_dn",
        ] {
            assert!(
                module.entry_points.iter().any(|e| e.name == ep),
                "entry point {ep} missing"
            );
        }
    }

    /// HC v3, its own module: it validates once the subgroup capability is
    /// there (no `enable subgroups;`), not without it, and again after the
    /// override specialization its pipelines get (`H3_KD` / `H3_KU`: 5 / 5
    /// for Qwen3.8-Flash-Next's cols 10240 and low 320).
    #[test]
    fn qwen4_hc3_shaders_validate() {
        use wgpu::naga;
        let flags = naga::valid::ValidationFlags::all();
        let module = naga::front::wgsl::parse_str(super::HC3_WGSL).expect("HC3 WGSL parses");
        let info = naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
            .validate(&module)
            .expect("HC3 WGSL validates");
        for ep in ["hc3_down", "hc3_down1", "hc3_down4", "hc3_upfold", "hc3_upfold4"] {
            assert!(
                module.entry_points.iter().any(|e| e.name == ep),
                "entry point {ep} missing"
            );
        }
        for knob in ["H3_KD", "H3_KU"] {
            assert!(
                module
                    .overrides
                    .iter()
                    .any(|(_, o)| o.name.as_deref() == Some(knob)),
                "override {knob} missing"
            );
        }
        let no_sg = naga::valid::Capabilities::all() - naga::valid::Capabilities::SUBGROUP;
        assert!(
            naga::valid::Validator::new(flags, no_sg)
                .validate(&module)
                .is_err(),
            "HC3 must need the subgroup capability"
        );
        for (ep, knob) in [("hc3_down", "H3_KD"), ("hc3_down1", "H3_KD"), ("hc3_down4", "H3_KD"), ("hc3_upfold", "H3_KU"), ("hc3_upfold4", "H3_KU")] {
            for k in [0.0, 1.0, 5.0, 8.0] {
                let mut pc = naga::back::PipelineConstants::default();
                pc.insert(knob.to_string(), k);
                let (m2, _) = naga::back::pipeline_constants::process_overrides(
                    &module,
                    &info,
                    Some((naga::ShaderStage::Compute, ep)),
                    &pc,
                )
                .unwrap_or_else(|e| panic!("{ep} with {knob} = {k}: {e:?}"));
                naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
                    .validate(&m2)
                    .unwrap_or_else(|e| panic!("{ep} with {knob} = {k} validates: {e:?}"));
            }
        }
    }
}

/// HC v3 against the pre-v3 kernels through the frame encoders themselves
/// (bindings, uniforms, row strides, cache slots), on synthetic f16 weights
/// of Qwen3.8-Flash-Next's shape, for 1..8 token rows, with and without the
/// injection-gate rows and the fused pending inject. Needs a card that
/// admits HC v3:
/// `cargo test --release -p cortiq-engine --features gpu hc3_matches -- --ignored --nocapture`
#[cfg(test)]
mod hc3_device_tests {
    use super::*;

    pub(super) fn f2h(f: f32) -> u16 {
        // round-to-nearest-even into f16 (the values here stay normal)
        let x = f.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let e = ((x >> 23) & 0xff) as i32 - 127 + 15;
        let m = x & 0x7f_ffff;
        if e <= 0 {
            return sign;
        }
        if e >= 31 {
            return sign | 0x7c00;
        }
        let mut h = (((e as u32) << 10) | (m >> 13)) as u16;
        let rem = m & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
            h += 1;
        }
        sign | h
    }

    pub(super) struct Rnd(pub(super) u64);
    impl Rnd {
        /// 64 raw xorshift bits
        pub(super) fn bits(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        pub(super) fn n(&mut self) -> f32 {
            // sum of four uniforms, centred: close enough to a normal
            let mut s = 0.0f32;
            for _ in 0..4 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                s += (self.0 >> 40) as f32 / (1u64 << 24) as f32;
            }
            (s - 2.0) * 1.7
        }
    }

    pub(super) fn upload(c: &Ctx, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        let b = storage_buf(c, label, bytes.len() as u64);
        c.queue.write_buffer(&b, 0, bytes);
        b
    }

    fn f16_weight(c: &Ctx, r: &mut Rnd, rows: usize, cols: usize, scale: f32) -> WeightRef {
        let h: Vec<u16> = (0..rows * cols).map(|_| f2h(r.n() * scale)).collect();
        WeightRef {
            buf: upload(c, "hc3-test-w", bytemuck::cast_slice(&h)),
            dtype: TensorDtype::F16,
            rows,
            cols,
        }
    }

    /// Written over every output buffer before each run, so a value a
    /// kernel failed to write cannot pass for the other arm's.
    pub(super) const POISON: f32 = 7.7e30;

    /// max |a - b| over the first `n` floats of `nt` rows at `stride`,
    /// relative to max |b|; infinite where either side kept the poison
    /// the run wrote first (a value the kernel never produced) or is not finite
    pub(super) fn rel(a: &[f32], b: &[f32], nt: usize, stride: usize, n: usize) -> f32 {
        let (mut d, mut m) = (0.0f32, 0.0f32);
        for t in 0..nt {
            for i in 0..n {
                let (x, y) = (a[t * stride + i], b[t * stride + i]);
                if !x.is_finite() || !y.is_finite() || x == POISON || y == POISON {
                    return f32::INFINITY;
                }
                d = d.max((x - y).abs());
                m = m.max(y.abs());
            }
        }
        d / m.max(1e-30)
    }

    /// The O(1) glue kernels: the frame's strided rows packed dense (and the query
    /// trace at the rows' positions), the far-field rows through the sigmoid gate.
    #[test]
    fn o1_pack_and_gate_match_host() {
        let Some(c) = ctx() else {
            eprintln!("no wgpu device: skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("qwen4 kernels unavailable on this adapter: skipped");
            return;
        };
        let (nh, nkv, hd, nt) = (16usize, 2usize, 256usize, 3usize);
        let (nq, nk) = (nh * hd, nkv * hd);
        let (qs, ks) = (es(nq * 4) as usize, es(nk * 4) as usize);
        let stride = crate::dory::QT_STRIDE;
        let qcap = 16usize;
        let pos0 = 38usize; // rows 38, 39, 40: one of them a trace row
        let mut r = Rnd(0x51A5_0001);
        let fill = |r: &mut Rnd, n: usize| -> Vec<f32> { (0..n).map(|_| r.n()).collect() };
        let (q, k, v, gate, far) = (fill(&mut r, nt * qs), fill(&mut r, nt * ks), fill(&mut r, nt * ks), fill(&mut r, nt * qs), fill(&mut r, nt * nq));
        let up = |label: &str, data: &[f32]| -> wgpu::Buffer {
            let b = storage_buf(c, label, (data.len() * 4) as u64);
            c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };
        let (qb, kb, vb, gb, fb) = (up("t-q", &q), up("t-k", &k), up("t-v", &v), up("t-gate", &gate), up("t-far", &far));
        let (pq, pk, pv, qt, att) = (
            storage_buf(c, "t-pq", (nt * nq * 4) as u64),
            storage_buf(c, "t-pk", (nt * nk * 4) as u64),
            storage_buf(c, "t-pv", (nt * nk * 4) as u64),
            storage_buf(c, "t-qt", (qcap * nq * 4) as u64),
            storage_buf(c, "t-att", (nt * qs * 4) as u64),
        );
        let mut tab = vec![0u32; TMAX * 4];
        for t in 0..nt {
            tab[t * 4] = (pos0 + t) as u32;
        }
        let tabb = up("t-tab", bytemuck::cast_slice::<u32, f32>(&tab));
        let bind = |pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| {
            let ents: Vec<_> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
            c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &ents })
        };
        let u_pack = uniform_u32x16(c, [nh as u32, nkv as u32, hd as u32, qs as u32, ks as u32, ks as u32, 1, stride as u32, qcap as u32, 0, 0, 0, 0, 0, 0, 0]);
        let u_gate = uniform_u32x8(c, [nq as u32, qs as u32, qs as u32, 0, 0, 0, 0, 0]);
        let bg_pack = bind(&p.t_o1_pack, &[&qb, &kb, &vb, &pq, &pk, &pv, &qt, &tabb, &u_pack]);
        let bg_gate = bind(&p.t_o1_gate, &[&fb, &gb, &att, &u_gate]);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t-o1") });
        {
            let mut pass = begin_pass(&mut enc);
            pass.set_pipeline(&p.t_o1_pack);
            pass.set_bind_group(0, &bg_pack, &[]);
            pass.dispatch_workgroups((nq.max(nk) as u32).div_ceil(256), nt as u32, 1);
            pass.set_pipeline(&p.t_o1_gate);
            pass.set_bind_group(0, &bg_gate, &[]);
            pass.dispatch_workgroups((nq as u32).div_ceil(256), nt as u32, 1);
        }
        submit(c, enc.finish());
        let (gpq, gpk, gpv, gqt, gatt) = (
            train_ops::download(c, &pq, nt * nq).unwrap(),
            train_ops::download(c, &pk, nt * nk).unwrap(),
            train_ops::download(c, &pv, nt * nk).unwrap(),
            train_ops::download(c, &qt, qcap * nq).unwrap(),
            train_ops::download(c, &att, nt * qs).unwrap(),
        );
        for t in 0..nt {
            assert_eq!(&gpq[t * nq..(t + 1) * nq], &q[t * qs..t * qs + nq], "packed q row {t}");
            assert_eq!(&gpk[t * nk..(t + 1) * nk], &k[t * ks..t * ks + nk], "packed k row {t}");
            assert_eq!(&gpv[t * nk..(t + 1) * nk], &v[t * ks..t * ks + nk], "packed v row {t}");
            let pos = pos0 + t;
            if pos % stride == 0 {
                let rr = pos / stride;
                assert_eq!(&gqt[rr * nq..(rr + 1) * nq], &q[t * qs..t * qs + nq], "trace row {rr}");
            }
            for i in 0..nq {
                let want = far[t * nq + i] / (1.0 + (-gate[t * qs + i]).exp());
                let got = gatt[t * qs + i];
                assert!((got - want).abs() <= 1e-5 * want.abs().max(1.0), "gate row {t} lane {i}: {got} vs {want}");
            }
        }
        // rows the table did not name stay untouched
        let other: Vec<usize> = (0..qcap).filter(|rr| !(pos0..pos0 + nt).any(|pos| pos % stride == 0 && pos / stride == *rr)).collect();
        for rr in other {
            assert!(gqt[rr * nq..(rr + 1) * nq].iter().all(|&x| x == 0.0), "trace row {rr} written");
        }
    }

    /// The split-key GQA attention (`q4t_attend_sp` + `q4t_attend_cmb`) against the
    /// per-head kernel it replaces: three tokens attending 70, 33 and 1 positions of a
    /// two-KV-head cache, 24 query heads of 256, each through its sigmoid gate.
    #[test]
    fn split_attend_matches_per_head_kernel() {
        let Some(c) = ctx() else {
            eprintln!("no wgpu device: skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("qwen4 kernels unavailable on this adapter: skipped");
            return;
        };
        let (nh, nkv, hd, cap, nt) = (24usize, 2usize, 256usize, 96usize, 3usize);
        let ms = [70usize, 33, 1];
        let is = 128usize;
        let mut r = Rnd(0x5A11_7E2D);
        let fill = |r: &mut Rnd, n: usize| -> Vec<f32> { (0..n).map(|_| r.n()).collect() };
        let q = fill(&mut r, nt * nh * hd);
        let k = fill(&mut r, nkv * cap * hd);
        let v = fill(&mut r, nkv * cap * hd);
        let gate = fill(&mut r, nt * nh * hd);
        let mut idx = vec![0u32; nt * is];
        let mut tab = vec![0u32; nt * 4];
        for t in 0..nt {
            for j in 0..ms[t] {
                idx[t * is + j] = ((j * 7 + t * 3) % cap) as u32;
            }
            // one complete block, none kept: the kernels read the index list
            // (kept == complete would take the identity list)
            tab[t * 4 + 1] = 1;
            tab[t * 4 + 3] = ms[t] as u32;
        }
        let up_f = |label: &str, data: &[f32]| -> wgpu::Buffer {
            let b = storage_buf(c, label, (data.len() * 4) as u64);
            c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };
        let up_u = |label: &str, data: &[u32]| -> wgpu::Buffer {
            let b = storage_buf(c, label, (data.len() * 4) as u64);
            c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };
        let (qb, kb, vb, gb) = (up_f("a-q", &q), up_f("a-k", &k), up_f("a-v", &v), up_f("a-g", &gate));
        let (ib, tb) = (up_u("a-idx", &idx), up_u("a-tab", &tab));
        let out_a = storage_buf(c, "a-out-a", (nt * nh * hd * 4) as u64);
        let out_b = storage_buf(c, "a-out-b", (nt * nh * hd * 4) as u64);
        const CH: usize = 32;
        let nsp = ms.iter().max().unwrap().div_ceil(CH);
        let po = storage_buf(c, "a-po", (nt * nsp * nh * hd * 4) as u64);
        let pml = storage_buf(c, "a-pml", (nt * nsp * nh * 8) as u64);
        let scale = (hd as f32).sqrt().recip().to_bits();
        let (nh32, hd32, g32, cap32, qs, is32) = (nh as u32, hd as u32, (nh / nkv) as u32, cap as u32, (nh * hd) as u32, is as u32);
        let ua = uniform_u32x16(c, [nh32, hd32, scale, g32, cap32, qs, is32, qs, qs, 0, 0, 0, 0, 0, 0, 0]);
        let usp = uniform_u32x16(c, [nh32, hd32, scale, g32, cap32, qs, is32, CH as u32, nsp as u32, hd32, 0, 0, 0, 0, 0, 0]);
        let ucb = uniform_u32x8(c, [nh32, hd32, CH as u32, nsp as u32, hd32, qs, qs, 0]);
        let bind = |pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| {
            let ents: Vec<_> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
            c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &ents })
        };
        let bga = bind(&p.t_attend, &[&qb, &kb, &vb, &ib, &gb, &out_a, &tb, &ua]);
        let bgs = bind(&p.t_attend_sp, &[&qb, &kb, &vb, &ib, &tb, &po, &pml, &usp]);
        let bgc = bind(&p.t_attend_cmb, &[&po, &pml, &gb, &tb, &out_b, &ucb]);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t-attend") });
        {
            let mut pass = begin_pass(&mut enc);
            pass.set_pipeline(&p.t_attend);
            pass.set_bind_group(0, &bga, &[]);
            pass.dispatch_workgroups(nh as u32, nt as u32, 1);
            pass.set_pipeline(&p.t_attend_sp);
            pass.set_bind_group(0, &bgs, &[]);
            pass.dispatch_workgroups(nkv as u32, nsp as u32, nt as u32);
            pass.set_pipeline(&p.t_attend_cmb);
            pass.set_bind_group(0, &bgc, &[]);
            pass.dispatch_workgroups(nh as u32, nt as u32, 1);
        }
        submit(c, enc.finish());
        let a = train_ops::download(c, &out_a, nt * nh * hd).unwrap();
        let b = train_ops::download(c, &out_b, nt * nh * hd).unwrap();
        let mut worst = 0f32;
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let d = (x - y).abs();
            assert!(d <= 2e-5 * x.abs().max(1.0), "element {i} (token {}, head {}): per-head {x} vs split {y}", i / (nh * hd), (i / hd) % nh);
            worst = worst.max(d);
        }
        eprintln!("split attention vs per-head kernel: max |diff| {worst:.2e}");
    }

    /// The radix top-k (both its paths: everything kept, and the 8-bit digit
    /// search) against a host sort: ties to the lower index, non-positive
    /// scores all equal, the kept blocks in index order.
    #[test]
    fn radix_topk_matches_host() {
        let Some(c) = ctx() else {
            eprintln!("no wgpu device: skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("qwen4 kernels unavailable on this adapter: skipped");
            return;
        };
        let cases = [(5000usize, 2048usize), (2048, 2048), (300, 2048), (3000, 1000), (4096, 1), (1500, 1499)];
        let nt = cases.len();
        let (ss, ps) = (8192usize, 4096usize);
        let mut r = Rnd(0x7095_1D3A);
        let mut sc = vec![0f32; nt * ss];
        let mut tab = vec![0u32; nt * 4];
        for (t, &(n, k)) in cases.iter().enumerate() {
            for i in 0..n {
                // a coarse ladder: many exact ties, some negative and zero scores
                sc[t * ss + i] = ((r.n() * 6.0).round() / 4.0).max(-0.5) + if t % 2 == 0 { 0.0 } else { (r.n() * 1e-3).abs() };
            }
            tab[t * 4 + 1] = n as u32;
            tab[t * 4 + 2] = k as u32;
        }
        let up = |label: &str, data: &[u8]| -> wgpu::Buffer {
            let b = storage_buf(c, label, data.len() as u64);
            c.queue.write_buffer(&b, 0, data);
            b
        };
        let sb = up("tk-s", bytemuck::cast_slice(&sc));
        let tb = up("tk-tab", bytemuck::cast_slice(&tab));
        let ib = storage_buf(c, "tk-idx", (nt * ps * 4) as u64);
        let cb = storage_buf(c, "tk-cnt", (nt * 4 * 4) as u64);
        let u = uniform_u32x4(c, [ss as u32, ps as u32, 4, 0]);
        let ents: Vec<_> = [&sb, &ib, &cb, &tb, &u].iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
        let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.t_topk_r.get_bind_group_layout(0), entries: &ents });
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t-topk") });
        {
            let mut pass = begin_pass(&mut enc);
            pass.set_pipeline(&p.t_topk_r);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(nt as u32, 1, 1);
        }
        submit(c, enc.finish());
        let idx: Vec<u32> = bytemuck::cast_slice(&train_ops::download(c, &ib, nt * ps).unwrap()).to_vec();
        let cnt: Vec<u32> = bytemuck::cast_slice(&train_ops::download(c, &cb, nt * 4).unwrap()).to_vec();
        for (t, &(n, k)) in cases.iter().enumerate() {
            let key = |i: usize| -> u32 { let v = sc[t * ss + i]; if v <= 0.0 { 0 } else { v.to_bits() } };
            let mut ord: Vec<usize> = (0..n).collect();
            ord.sort_by(|&a, &b| key(b).cmp(&key(a)).then(a.cmp(&b)));
            ord.truncate(k.min(n));
            ord.sort();
            let got: Vec<usize> = idx[t * ps..t * ps + cnt[t * 4] as usize].iter().map(|&v| v as usize).collect();
            assert_eq!(got.len(), ord.len(), "case {t} (n {n}, k {k}): count");
            assert_eq!(got, ord, "case {t} (n {n}, k {k}): kept blocks");
        }
        eprintln!("radix top-k matches the host sort on {nt} cases");
    }

    /// The experts' tube kernels against the host: random q8_2f tubes of six experts,
    /// two tokens with four winners each (one rank empty), the MoE output accumulated.
    #[test]
    fn tube_kernels_match_host() {
        let Some(c) = ctx() else {
            eprintln!("no wgpu device: skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("qwen4 kernels unavailable on this adapter: skipped");
            return;
        };
        // Six experts, one of them (4) without a tube; h = 64 so the gate/up kernel runs
        // two row blocks; a shared expert's tube with its own bank.
        let (hidden, h, k, ne, nt) = (256usize, 64usize, 4usize, 6usize, 2usize);
        let slots = k + 1;
        let notube = 4usize;
        let mut r = Rnd(0x7B3E_0002);
        let fill = |r: &mut Rnd, n: usize| -> Vec<f32> { (0..n).map(|_| r.n() * 0.2).collect() };
        let seg_g = cortiq_core::quant::expected_nbytes(TensorDtype::Q8_2f, &[h, hidden]).unwrap().div_ceil(16) * 16;
        let seg_d = cortiq_core::quant::expected_nbytes(TensorDtype::Q8_2f, &[hidden, h]).unwrap().div_ceil(16) * 16;
        // a bank of `n` tubes, float and encoded
        let mut bank = |r: &mut Rnd, n: usize| -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
            let (mut gb, mut ub, mut db) = (vec![0u8; n * seg_g], vec![0u8; n * seg_g], vec![0u8; n * seg_d]);
            let (mut gf, mut uf, mut df) = (Vec::new(), Vec::new(), Vec::new());
            for e in 0..n {
                for (dst, seg, rows, cols, keep) in [(&mut gb, seg_g, h, hidden, &mut gf), (&mut ub, seg_g, h, hidden, &mut uf), (&mut db, seg_d, hidden, h, &mut df)] {
                    let vals = fill(r, rows * cols);
                    let enc = cortiq_core::quant::encode_q8_2f(&vals, rows, cols);
                    let mut deq = vec![0f32; rows * cols];
                    cortiq_core::quant::dequant_q8_2f(&enc, rows, cols, &mut deq);
                    dst[e * seg..e * seg + enc.len()].copy_from_slice(&enc);
                    keep.push(deq);
                }
            }
            (gb, ub, db, gf, uf, df)
        };
        let (gb, ub, db, gf, uf, df) = bank(&mut r, ne - 1);
        let (sgb, sub, sdb, sgf, suf, sdf) = bank(&mut r, 1);
        // expert e's entry in the packed bank
        let slot: Vec<u32> = (0..ne).map(|e| if e == notube { u32::MAX } else if e < notube { e as u32 } else { e as u32 - 1 }).collect();
        let xs = es(hidden * 4) as usize;
        let cs = es(4 * k * 4) as usize;
        let x = fill(&mut r, nt * xs);
        let mo0 = fill(&mut r, nt * xs);
        // winners: (expert, weight) pairs in the cold list's second block; token 1 picks the
        // expert without a tube
        let picks: [[(u32, f32); 4]; 2] = [[(3, 0.4), (0, 0.3), (5, 0.2), (1, 0.1)], [(2, 0.5), (4, 0.3), (u32::MAX, 0.0), (u32::MAX, 0.0)]];
        let mut win = vec![0u32; nt * cs];
        for t in 0..nt {
            for j in 0..k {
                let (e, w) = picks[t][j];
                win[t * cs + 2 * k + 2 * j] = e;
                win[t * cs + 2 * k + 2 * j + 1] = w.to_bits();
            }
        }
        // the route's weight table: the shared expert's σ at column k
        let sigma = [0.7f32, 0.35];
        let mut mwt = vec![0f32; nt * slots];
        for t in 0..nt {
            mwt[t * slots + k] = sigma[t];
        }
        let up_f = |label: &str, data: &[f32]| -> wgpu::Buffer {
            let b = storage_buf(c, label, (data.len() * 4) as u64);
            c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };
        let up_b = |label: &str, data: &[u8]| -> wgpu::Buffer {
            let b = storage_buf(c, label, data.len() as u64);
            c.queue.write_buffer(&b, 0, data);
            b
        };
        let (bg_, bu_, bd_) = (up_b("t-tg", &gb), up_b("t-tu", &ub), up_b("t-td", &db));
        let (sg_, su_, sd_) = (up_b("t-sg", &sgb), up_b("t-su", &sub), up_b("t-sd", &sdb));
        let slotb = up_b("t-slot", bytemuck::cast_slice(&slot));
        let sslotb = up_b("t-sslot", bytemuck::cast_slice(&[0u32]));
        let xb = up_f("t-x", &x);
        let mob = up_f("t-mo", &mo0);
        let mwtb = up_f("t-mwt", &mwt);
        let winb = up_b("t-win", bytemuck::cast_slice(&win));
        let act = storage_buf(c, "t-act", (nt * k * h * 4) as u64);
        let sact = storage_buf(c, "t-sact", (nt * h * 4) as u64);
        let (hid32, h32, k32) = (hidden as u32, h as u32, k as u32);
        let u = uniform_u32x16(c, [hid32, h32, k32, ne as u32, xs as u32, cs as u32, (seg_g / 4) as u32, (seg_d / 4) as u32, xs as u32, k32, slots as u32, 0, 0, 0, 0, 0]);
        let su = uniform_u32x16(c, [hid32, h32, 0, 1, xs as u32, cs as u32, (seg_g / 4) as u32, (seg_d / 4) as u32, xs as u32, 1, slots as u32, k32, 1, 0, 0, 0]);
        let bind = |pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| {
            let ents: Vec<_> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
            c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &ents })
        };
        let bg1 = bind(&p.t_tube_gu, &[&bg_, &bu_, &xb, &winb, &act, &u, &slotb]);
        let bg2 = bind(&p.t_tube_dn, &[&bd_, &act, &winb, &mob, &u, &mwtb, &slotb]);
        let bg3 = bind(&p.t_tube_gu, &[&sg_, &su_, &xb, &winb, &sact, &su, &sslotb]);
        let bg4 = bind(&p.t_tube_dn, &[&sd_, &sact, &winb, &mob, &su, &mwtb, &sslotb]);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("t-tube") });
        {
            let mut pass = begin_pass(&mut enc);
            pass.set_pipeline(&p.t_tube_gu);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(k as u32, nt as u32, (h as u32).div_ceil(32));
            pass.set_pipeline(&p.t_tube_dn);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups((hidden as u32).div_ceil(256), nt as u32, 1);
            pass.set_pipeline(&p.t_tube_gu);
            pass.set_bind_group(0, &bg3, &[]);
            pass.dispatch_workgroups(1, nt as u32, (h as u32).div_ceil(32));
            pass.set_pipeline(&p.t_tube_dn);
            pass.set_bind_group(0, &bg4, &[]);
            pass.dispatch_workgroups((hidden as u32).div_ceil(256), nt as u32, 1);
        }
        submit(c, enc.finish());
        let got_act = train_ops::download(c, &act, nt * k * h).unwrap();
        let got_sact = train_ops::download(c, &sact, nt * h).unwrap();
        let got_mo = train_ops::download(c, &mob, nt * xs).unwrap();
        let tube = |gf: &[f32], uf: &[f32], xv: &[f32], i: usize| -> f32 {
            let g: f32 = (0..hidden).map(|d| gf[i * hidden + d] * xv[d]).sum();
            let uu: f32 = (0..hidden).map(|d| uf[i * hidden + d] * xv[d]).sum();
            g / (1.0 + (-g).exp()) * uu
        };
        for t in 0..nt {
            let xv = &x[t * xs..t * xs + hidden];
            let mut want = mo0[t * xs..t * xs + hidden].to_vec();
            for j in 0..k {
                let (e, w) = picks[t][j];
                if e == u32::MAX || e as usize == notube {
                    assert!(got_act[(t * k + j) * h..(t * k + j + 1) * h].iter().all(|&a| a == 0.0), "rank {j} of token {t} should be empty");
                    continue;
                }
                let s = slot[e as usize] as usize;
                for i in 0..h {
                    let a = tube(&gf[s], &uf[s], xv, i);
                    let got = got_act[(t * k + j) * h + i];
                    assert!((got - a).abs() <= 1e-4 * a.abs().max(1.0), "act t{t} j{j} i{i}: {got} vs {a}");
                    for d in 0..hidden {
                        want[d] += w * a * df[s][d * h + i];
                    }
                }
            }
            for i in 0..h {
                let a = tube(&sgf[0], &suf[0], xv, i);
                let got = got_sact[t * h + i];
                assert!((got - a).abs() <= 1e-4 * a.abs().max(1.0), "shared act t{t} i{i}: {got} vs {a}");
                for d in 0..hidden {
                    want[d] += sigma[t] * a * sdf[0][d * h + i];
                }
            }
            for d in 0..hidden {
                let got = got_mo[t * xs + d];
                assert!((got - want[d]).abs() <= 2e-4 * want[d].abs().max(1.0), "out t{t} d{d}: {got} vs {}", want[d]);
            }
        }
    }

    /// The matrix-unit f16 pair GEMM against the scalar one on random data: an
    /// A/B pair of 300 + 4 rows over 1024 columns, 100 tokens, SiLU rows and
    /// sigmoid gate rows, whole-K and split-K.
    #[test]
    fn pair_coop_matches_scalar() {
        let Some(c) = ctx() else {
            eprintln!("no wgpu device: skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("qwen4 kernels unavailable: skipped");
            return;
        };
        let Some(pc) = p.pair_coop.as_ref() else {
            eprintln!("no matrix-unit pair on {}: skipped", c.adapter_info.name);
            return;
        };
        let (ra, rb, cols, nt) = (300usize, 4usize, 1024usize, 100usize);
        let mut r = Rnd(0x9A1C_00F1);
        let h16 = |v: f32| -> u16 { super::hc3_device_tests::f2h(v) };
        let wa: Vec<u16> = (0..ra * cols).map(|_| h16(r.n() * 0.05)).collect();
        let wb: Vec<u16> = (0..rb * cols).map(|_| h16(r.n() * 0.05)).collect();
        let x: Vec<f32> = (0..nt * cols).map(|_| r.n()).collect();
        let up = |label: &str, data: &[u8]| -> wgpu::Buffer {
            let b = storage_buf(c, label, data.len() as u64);
            c.queue.write_buffer(&b, 0, data);
            b
        };
        let (ab, bb, xb) = (up("pc-a", bytemuck::cast_slice(&wa)), up("pc-b", bytemuck::cast_slice(&wb)), up("pc-x", bytemuck::cast_slice(&x)));
        for ks in [1usize, 4] {
            let kc = cols.div_ceil(ks).div_ceil(64) * 64;
            let u = uniform_u32x16(c, [cols as u32, ra as u32, rb as u32, 3, 0.25f32.to_bits(), 0, nt as u32, cols as u32, ra as u32, rb as u32, kc as u32, ks as u32, 0, 0, 0, 0]);
            let mut outs = Vec::new();
            for (pipe, tile) in [(&p.t_f16_pair_mm, 16usize), (pc, 64usize)] {
                let ya = storage_buf(c, "pc-ya", (nt * ra * 4) as u64);
                let yb = storage_buf(c, "pc-yb", (nt * rb * 4) as u64);
                let part = storage_buf(c, "pc-part", (ks * nt * (ra + rb) * 4).max(4) as u64);
                let ents: Vec<_> = [&ab, &bb, &xb, &ya, &yb, &u, &part].iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
                let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &ents });
                let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc") });
                {
                    let mut pass = begin_pass(&mut enc);
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups((ra + rb).div_ceil(64) as u32, nt.div_ceil(tile) as u32, ks as u32);
                    if ks > 1 {
                        let ents2: Vec<_> = [&part, &ya, &yb, &u].iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
                        let bg2 = c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.t_pair_sum.get_bind_group_layout(0), entries: &ents2 });
                        pass.set_pipeline(&p.t_pair_sum);
                        pass.set_bind_group(0, &bg2, &[]);
                        pass.dispatch_workgroups(((ra + rb) as u32).div_ceil(256), nt as u32, 1);
                    }
                }
                submit(c, enc.finish());
                let a: Vec<f32> = bytemuck::cast_slice(&train_ops::download(c, &ya, nt * ra).unwrap()).to_vec();
                let b: Vec<f32> = bytemuck::cast_slice(&train_ops::download(c, &yb, nt * rb).unwrap()).to_vec();
                outs.push((a, b));
            }
            let (mut wa_, mut wb_) = (0f32, 0f32);
            for (x, y) in outs[0].0.iter().zip(&outs[1].0) {
                wa_ = wa_.max((x - y).abs() / x.abs().max(0.05));
            }
            for (x, y) in outs[0].1.iter().zip(&outs[1].1) {
                wb_ = wb_.max((x - y).abs());
            }
            eprintln!("pair coop vs scalar, ks {ks}: rows_a rel {wa_:.2e}, gate rows abs {wb_:.2e}");
            assert!(wa_ < 3e-2 && wb_ < 1e-2, "coop pair departs: {wa_} {wb_}");
        }
    }

    /// Kernel timing probe (`--ignored --nocapture` on the card): the cost of one
    /// dependent dispatch (a chain of 1000 one-workgroup sums, each waiting on
    /// the last) against the same dispatches with no hazard between them, and
    /// the HC v3 pair at the decode shape (one token, pending inject, 13 MB of
    /// f16 weights a call).
    #[test]
    #[ignore = "timing probe for the card"]
    fn decode_kernel_bench() {
        let Some(c) = ctx() else {
            eprintln!("no wgpu device: skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("qwen4 kernels unavailable: skipped");
            return;
        };
        let wait = || {
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        };
        let time = |label: &str, n: usize, enc_fn: &dyn Fn(&mut wgpu::ComputePass<'_>, usize)| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..4 {
                let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kbench") });
                {
                    let mut pass = begin_pass(&mut enc);
                    for i in 0..n {
                        enc_fn(&mut pass, i);
                    }
                }
                let cmd = enc.finish();
                wait();
                let t0 = std::time::Instant::now();
                c.queue.submit([cmd]);
                wait();
                best = best.min(t0.elapsed().as_secs_f64() * 1e6 / n as f64);
            }
            eprintln!("[kbench] {label}: {best:.2} us a dispatch (best of 4, {n} in one pass)");
            best
        };
        // 1. one-workgroup q4g_sum: dependent (one output) vs independent (16 outputs)
        let part = upload(c, "kb-part", &vec![0u8; 64 * 10 * 4]);
        let sel = upload(c, "kb-sel", bytemuck::cast_slice(&vec![0u32; 10]));
        let outs: Vec<wgpu::Buffer> = (0..16).map(|_| upload(c, "kb-mo", &vec![0u8; 64 * 4])).collect();
        let u = uniform_u32x4(c, [64, 10, 1, 64]);
        let bgs: Vec<wgpu::BindGroup> = outs
            .iter()
            .map(|o| {
                let ents: Vec<_> = [&part, &sel, o, &u].iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.g_sum.get_bind_group_layout(0), entries: &ents })
            })
            .collect();
        time("q4g_sum 1 WG, dependent chain", 1000, &|pass, _| {
            pass.set_pipeline(&p.g_sum);
            pass.set_bind_group(0, &bgs[0], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        });
        time("q4g_sum 1 WG, 16 independent outputs", 1000, &|pass, i| {
            pass.set_pipeline(&p.g_sum);
            pass.set_bind_group(0, &bgs[i % 16], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        });
        // 2. HC v3 pair, decode shape
        if p.hc3.is_none() {
            eprintln!("HC v3 not admitted: HC timing skipped");
            return;
        }
        let (hidden, hc, lr) = (2560usize, 4usize, 320usize);
        let hh = hc * hidden;
        let g = Geom {
            hidden,
            hc,
            eps: 1e-6,
            n_heads: 24,
            n_kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            index_heads: 4,
            index_dim: 128,
            index_budget: 2048,
            compress_ratio: 4,
            gdn: GdnGeom { nv: 48, nk: 16, dk: 128, dv: 128, kk: 4 },
            ple_kernel: 4,
            ple_dilation: 1,
            top_k: 10,
            n_experts: 256,
            inter: 640,
            gu_q2: false,
        };
        let dev = Dev::new(0xC3C3_0B01, &g, &[]).expect("device state");
        let mut r = Rnd(0x51ED_2701);
        // a ring of weight sets, so the timing reads DRAM and not the 13 MB in L2
        let sets: Vec<(WeightRef, WeightRef, WeightRef)> = (0..12)
            .map(|_| (f16_weight(c, &mut r, lr, hh, 0.02), f16_weight(c, &mut r, hc, hh, 0.02), f16_weight(c, &mut r, hh, lr, 0.3)))
            .collect();
        let norm: Vec<f32> = (0..hh).map(|_| r.n() * 0.1).collect();
        let nw = const_buf(c, bytemuck::cast_slice(&norm));
        let os = es(hidden * 4) as usize;
        let gsd = es(hc * 4) as usize;
        let ls = es(lr * 4) as usize;
        let hs = es(hh * 4) as usize;
        let h0: Vec<f32> = (0..TMAX * hs).map(|_| r.n()).collect();
        c.queue.write_buffer(&dev.hyper, 0, bytemuck::cast_slice(&h0));
        let blk = upload(c, "kb-blk", bytemuck::cast_slice(&(0..TMAX * os).map(|_| r.n()).collect::<Vec<f32>>()));
        let gate = upload(c, "kb-gate", bytemuck::cast_slice(&(0..TMAX * gsd).map(|_| r.n()).collect::<Vec<f32>>()));
        let x = upload(c, "kb-x", &vec![0u8; TMAX * os * 4]);
        let y = upload(c, "kb-y", &vec![0u8; TMAX * gsd * 4]);
        let low = upload(c, "kb-low", &vec![0u8; TMAX * ls * 4]);
        for nt in [1usize, 4] {
            let us = time(&format!("HC v3 pair, nt {nt}, pending inject"), 240, &|pass, i| {
                let (down, inj, up) = &sets[i % sets.len()];
                let bc = Bc { dev: &dev, li: i % sets.len(), tok: TW + nt, direct: true };
                let pre = PreInject { blk: &blk, bs: os as u32, gate: &gate, gs: gsd as u32, step: 2 };
                encode_hc_v3(c, p, pass, &g, &nw, down, up, Some(inj), &dev.hyper, &x, &low, &y, Some(&pre), nt, bc, 100)
                    .expect("HC v3 takes this shape");
            });
            let mb = (2 * lr * hh * 2 + hc * hh * 2) as f64 / 1e6;
            eprintln!("[kbench]   = {:.0} GB/s over {mb:.1} MB of weights (two dispatches a call)", mb * 1e3 / us);
        }
        // 3. the router-shaped one-token f16 pair: 256 + 1 rows of 2560
        let ra: Vec<WeightRef> = (0..12).map(|_| f16_weight(c, &mut r, 256, hidden, 0.02)).collect();
        let rb: Vec<WeightRef> = (0..12).map(|_| f16_weight(c, &mut r, 1, hidden, 0.02)).collect();
        let xr = upload(c, "kb-xr", bytemuck::cast_slice(&(0..hidden).map(|_| r.n()).collect::<Vec<f32>>()));
        let ya = upload(c, "kb-ya", &vec![0u8; 512 * 4]);
        let yb = upload(c, "kb-yb", &vec![0u8; 64 * 4]);
        let ur = uniform_u32x8(c, [hidden as u32, 256, 1, 2, 1.0f32.to_bits(), 10, 0, 0]);
        let bgr: Vec<wgpu::BindGroup> = (0..12)
            .map(|i| {
                let ents: Vec<_> = [&ra[i].buf, &rb[i].buf, &xr, &ya, &yb, &ur].iter().enumerate().map(|(k, b)| bind_buf(k as u32, b)).collect();
                c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.f16_matvec2.get_bind_group_layout(0), entries: &ents })
            })
            .collect();
        time("router pair f16_matvec2 257 x 2560, one token", 480, &|pass, i| {
            pass.set_pipeline(&p.f16_matvec2);
            pass.set_bind_group(0, &bgr[i % 12], &[]);
            pass.dispatch_workgroups(257, 1, 1);
        });
        // 4. split attention + fold at the decode shape: 24 heads, 2 KV heads of 256,
        // 2051 keys (the 2048-position budget and the open tail)
        let (nh, nkv, hd, cap, m) = (24usize, 2usize, 256usize, 4096usize, 2051usize);
        let qv = upload(c, "kb-q", bytemuck::cast_slice(&(0..nh * hd).map(|_| r.n()).collect::<Vec<f32>>()));
        let kc = upload(c, "kb-k", bytemuck::cast_slice(&(0..nkv * cap * hd).map(|_| r.n()).collect::<Vec<f32>>()));
        let vc = upload(c, "kb-v", bytemuck::cast_slice(&(0..nkv * cap * hd).map(|_| r.n()).collect::<Vec<f32>>()));
        let gt = upload(c, "kb-g", bytemuck::cast_slice(&(0..nh * hd).map(|_| r.n()).collect::<Vec<f32>>()));
        let is = 4096usize;
        let idxv: Vec<u32> = (0..is as u32).map(|j| (j * 7) % m as u32).collect();
        let ib = upload(c, "kb-idx", bytemuck::cast_slice(&idxv));
        let tb = upload(c, "kb-tab", bytemuck::cast_slice(&[3000u32, 750, 512, m as u32]));
        const CH: usize = 32;
        let nsp = m.div_ceil(CH);
        let po = upload(c, "kb-po", &vec![0u8; nsp * nh * hd * 4]);
        let pml = upload(c, "kb-pml", &vec![0u8; nsp * nh * 8]);
        let att = upload(c, "kb-att", &vec![0u8; nh * hd * 4]);
        let scale = (hd as f32).sqrt().recip().to_bits();
        let (nh32, hd32, g32, cap32, qs, is32) = (nh as u32, hd as u32, (nh / nkv) as u32, cap as u32, (nh * hd) as u32, is as u32);
        let usp = uniform_u32x16(c, [nh32, hd32, scale, g32, cap32, qs, is32, CH as u32, nsp as u32, hd32, 0, 0, 0, 0, 0, 0]);
        let ucb = uniform_u32x8(c, [nh32, hd32, CH as u32, nsp as u32, hd32, qs, qs, 0]);
        let bind = |pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| {
            let ents: Vec<_> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
            c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &ents })
        };
        let bgs = bind(&p.t_attend_sp, &[&qv, &kc, &vc, &ib, &tb, &po, &pml, &usp]);
        let bgc = bind(&p.t_attend_cmb, &[&po, &pml, &gt, &tb, &att, &ucb]);
        time("split attention partials, 2051 keys", 240, &|pass, _| {
            pass.set_pipeline(&p.t_attend_sp);
            pass.set_bind_group(0, &bgs, &[]);
            pass.dispatch_workgroups(nkv as u32, nsp as u32, 1);
        });
        // 5. one-token q8_2f matvecs at the GDN shapes (qkv [10240, 2560], out [2560, 6144])
        for (shape, (rows, cols)) in [(10240usize, 2560usize), (2560, 6144), (12288, 2560)].into_iter().enumerate() {
            let ws: Vec<WeightRef> = (0..6)
                .map(|_| {
                    let vals: Vec<f32> = (0..rows * cols).map(|_| r.n() * 0.02).collect();
                    let bytes = cortiq_core::quant::encode_q8_2f(&vals, rows, cols);
                    WeightRef { buf: upload(c, "kb-q82", &bytes), dtype: TensorDtype::Q8_2f, rows, cols }
                })
                .collect();
            let xq = upload(c, "kb-xq", bytemuck::cast_slice(&(0..cols).map(|_| r.n()).collect::<Vec<f32>>()));
            let yq = upload(c, "kb-yq", &vec![0u8; rows * 4]);
            let mb = (rows * cols) as f64 * 1.0625 / 1e6;
            let us = time(&format!("q8_2f matvec [{rows}, {cols}], one token"), 240, &|pass, i| {
                // a layer slot per (shape, weight set): the bind cache keys on it
                let bc = Bc { dev: &dev, li: 20 + shape * 8 + i % ws.len(), tok: TW + 1, direct: true };
                assert!(q82_t(c, p, pass, &ws[i % ws.len()], &xq, cols as u32, &yq, rows as u32, 1, bc, 400));
            });
            eprintln!("[kbench]   = {:.0} GB/s over {mb:.1} MB", mb * 1e3 / us);
            // the output of the last weight set, to compare kernel variants across runs
            let yv: Vec<f32> = bytemuck::cast_slice(&train_ops::download(c, &yq, rows).unwrap()).to_vec();
            let sum: f64 = yv.iter().map(|v| f64::from(*v).abs()).sum();
            eprintln!("[kbench]   y[0..3] {:.6} {:.6} {:.6}, sum|y| {sum:.4}", yv[0], yv[1], yv[2]);
        }
        // 6. decode experts (blocked gate/up + down, 11 slots: ten routed and the shared)
        //    over an arena of 8 segments x 24 experts (~490 MB, past the L2)
        {
            const PS: usize = 24;
            let (inter, gsl) = (640usize, 11usize);
            let gu_len = cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[inter, hidden]).unwrap();
            let d_len = cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[hidden, inter]).unwrap();
            let rnd_bank = |r: &mut Rnd, len: usize| -> Vec<wgpu::Buffer> {
                (0..8)
                    .map(|_| {
                        let v: Vec<u8> = (0..len * PS)
                            .map(|_| ((r.bits() >> 40) as u8) & 0x77)
                            .collect();
                        upload(c, "kb-bank", &v)
                    })
                    .collect()
            };
            let (gb, ub, db) = (rnd_bank(&mut r, gu_len), rnd_bank(&mut r, gu_len), rnd_bank(&mut r, d_len));
            fn arr4(v: &[wgpu::Buffer]) -> Vec<wgpu::BufferBinding<'_>> {
                v.iter().map(wgpu::Buffer::as_entire_buffer_binding).collect()
            }
            fn ent<'a>(b: u32, a: &'a [wgpu::BufferBinding<'a>]) -> wgpu::BindGroupEntry<'a> {
                wgpu::BindGroupEntry { binding: b, resource: wgpu::BindingResource::BufferArray(a) }
            }
            let (ga, ua, da) = (arr4(&gb), arr4(&ub), arr4(&db));
            let ge = Geom { inter, ..g };
            let (gu4, dn4) = p.experts_blocked(&ge);
            let xe = upload(c, "kb-xe", bytemuck::cast_slice(&(0..hidden).map(|_| r.n()).collect::<Vec<f32>>()));
            let wte = upload(c, "kb-wte", bytemuck::cast_slice(&vec![0.1f32; gsl]));
            let acte = upload(c, "kb-acte", &vec![0u8; gsl * inter * 4]);
            let ye = upload(c, "kb-ye", &vec![0u8; hidden * 4]);
            let gu_u = uniform_u32x8(c, [(hidden / 32) as u32, inter as u32, gsl as u32, (gu_len / 2) as u32, 0, PS as u32, 0, 0]);
            let dn_u = uniform_u32x8(c, [(inter / 32) as u32, hidden as u32, gsl as u32, (d_len / 2) as u32, PS as u32, 0, 0, 0]);
            let n_exp = 8 * PS;
            let sels: Vec<wgpu::Buffer> = (0..16)
                .map(|k| {
                    let v: Vec<u32> = (0..gsl).map(|s| ((k * 11 + s * 17) % n_exp) as u32).collect();
                    upload(c, "kb-sel", bytemuck::cast_slice(&v))
                })
                .collect();
            let mk = |pipe: &wgpu::ComputePipeline, ents: &[wgpu::BindGroupEntry]| {
                c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: ents })
            };
            let mkp = |pipe: &wgpu::ComputePipeline, u: &wgpu::Buffer| {
                c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(1), entries: &[bind_buf(0, u)] })
            };
            let bgu: Vec<wgpu::BindGroup> = sels
                .iter()
                .map(|sl| mk(gu4, &[ent(0, &ga), ent(1, &ua), bind_buf(2, &xe), bind_buf(3, sl), bind_buf(4, &acte)]))
                .collect();
            let bdn: Vec<wgpu::BindGroup> = sels
                .iter()
                .map(|sl| mk(dn4, &[ent(0, &da), bind_buf(1, &acte), bind_buf(2, sl), bind_buf(3, &wte), bind_buf(4, &ye)]))
                .collect();
            let (pgu, pdn) = (mkp(gu4, &gu_u), mkp(dn4, &dn_u));
            let gr = p.gu_rows(&ge) as u32;
            let us_gu = time("decode experts gate/up (11 slots)", 480, &|pass, i| {
                pass.set_pipeline(gu4);
                pass.set_bind_group(0, &bgu[i % 16], &[]);
                pass.set_bind_group(1, &pgu, &[]);
                pass.dispatch_workgroups(inter as u32 / gr, gsl as u32, 1);
            });
            eprintln!("[kbench]   = {:.0} GB/s over {:.1} MB", (2 * gu_len * gsl) as f64 / 1e3 / us_gu, (2 * gu_len * gsl) as f64 / 1e6);
            let us_dn = time("decode experts down (11 slots)", 480, &|pass, i| {
                pass.set_pipeline(dn4);
                pass.set_bind_group(0, &bdn[i % 16], &[]);
                pass.set_bind_group(1, &pdn, &[]);
                pass.dispatch_workgroups((hidden / 4) as u32, 1, 1);
            });
            eprintln!("[kbench]   = {:.0} GB/s over {:.1} MB", (d_len * gsl) as f64 / 1e3 / us_dn, (d_len * gsl) as f64 / 1e6);
        }
        time("split attention partials + fold", 240, &|pass, i| {
            if i % 2 == 0 {
                pass.set_pipeline(&p.t_attend_sp);
                pass.set_bind_group(0, &bgs, &[]);
                pass.dispatch_workgroups(nkv as u32, nsp as u32, 1);
            } else {
                pass.set_pipeline(&p.t_attend_cmb);
                pass.set_bind_group(0, &bgc, &[]);
                pass.dispatch_workgroups(nh as u32, 1, 1);
            }
        });
    }

    #[test]
    #[ignore = "needs a GPU that admits HC v3"]
    fn hc3_matches_pre_v3_kernels() {
        let Some(c) = ctx() else {
            eprintln!("no wgpu device: skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("qwen4 kernels unavailable on this adapter: skipped");
            return;
        };
        if p.hc3.is_none() {
            eprintln!("HC v3 not admitted on {}: skipped", c.adapter_info.name);
            return;
        }
        let (hidden, hc, lr) = (2560usize, 4usize, 320usize);
        let hh = hc * hidden;
        let g = Geom {
            hidden,
            hc,
            eps: 1e-6,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            index_heads: 4,
            index_dim: 128,
            index_budget: 2048,
            compress_ratio: 4,
            gdn: GdnGeom {
                nv: 32,
                nk: 16,
                dk: 128,
                dv: 128,
                kk: 4,
            },
            ple_kernel: 4,
            ple_dilation: 1,
            top_k: 10,
            n_experts: 512,
            inter: 512,
            gu_q2: true,
        };
        let dev = Dev::new(0xC3C3_0001, &g, &[]).expect("device state");
        let mut r = Rnd(0x9E37_79B9_7F4A_7C15);
        let down = f16_weight(c, &mut r, lr, hh, 0.02);
        let inj_w = f16_weight(c, &mut r, hc, hh, 0.02);
        let up = f16_weight(c, &mut r, hh, lr, 0.3);
        let norm: Vec<f32> = (0..hh).map(|_| r.n() * 0.1).collect();
        let (hs, ls, os) = (
            es(hh * 4) as usize,
            es(lr * 4) as usize,
            es(hidden * 4) as usize,
        );
        let gsd = es(hc * 4) as usize;
        let scales = [1.0f32, 2.0, 0.5, 3.0];
        let mut h0 = vec![0.0f32; TMAX * hs];
        for t in 0..TMAX {
            for i in 0..hh {
                h0[t * hs + i] = r.n() * scales[i / hidden];
            }
        }
        let mut blk0 = vec![0.0f32; TMAX * os];
        for t in 0..TMAX {
            for i in 0..hidden {
                blk0[t * os + i] = r.n();
            }
        }
        let mut gate0 = vec![0.0f32; TMAX * gsd];
        for t in 0..TMAX {
            for s in 0..hc {
                gate0[t * gsd + s] = r.n() * 2.0;
            }
        }
        let blk = upload(c, "hc3-test-blk", bytemuck::cast_slice(&blk0));
        let gate = upload(c, "hc3-test-gate", bytemuck::cast_slice(&gate0));
        let zeros = |n: usize| vec![0u8; n * 4];
        let x_old = upload(c, "hc3-test-x-old", &zeros(TMAX * os));
        let x_new = upload(c, "hc3-test-x-new", &zeros(TMAX * os));
        let y_old = upload(c, "hc3-test-inj-old", &zeros(TMAX * gsd));
        let y_new = upload(c, "hc3-test-inj-new", &zeros(TMAX * gsd));
        let nw = const_buf(c, bytemuck::cast_slice(&norm));
        // one `low` per arm (the frame pool's would carry the reference
        // arm's values into the v3 readback wherever v3 did not write)
        let low_old = upload(c, "hc3-test-low-old", &zeros(TMAX * ls));
        let low_new = upload(c, "hc3-test-low-new", &zeros(TMAX * ls));
        let poison = |b: &wgpu::Buffer, n: usize| {
            c.queue
                .write_buffer(b, 0, bytemuck::cast_slice(&vec![POISON; n]));
        };
        let f32s = |b: &[u8]| -> Vec<f32> { bytemuck::cast_slice(b).to_vec() };
        let mut worst = 0.0f32;
        for nt in [1usize, 2, 3, 4, 5, 8] {
            for with_b in [true, false] {
                for with_pre in [false, true] {
                    let bc = Bc {
                        dev: &dev,
                        li: usize::from(with_b),
                        tok: TW + nt,
                        direct: true,
                    };
                    let pre = with_pre.then_some(PreInject {
                        blk: &blk,
                        bs: os as u32,
                        gate: &gate,
                        gs: gsd as u32,
                        step: 2,
                    });
                    let b = with_b.then_some(&inj_w);
                    let run = |v3: bool| -> Vec<Vec<f32>> {
                        c.queue
                            .write_buffer(&dev.hyper, 0, bytemuck::cast_slice(&h0));
                        let (x, y, low) = if v3 {
                            (&x_new, &y_new, &low_new)
                        } else {
                            (&x_old, &y_old, &low_old)
                        };
                        poison(x, TMAX * os);
                        poison(y, TMAX * gsd);
                        poison(low, TMAX * ls);
                        let mut enc = new_encoder("hc3-test").expect("encoder");
                        {
                            let mut pass = begin_pass(&mut enc);
                            if v3 {
                                encode_hc_v3(
                                    c,
                                    p,
                                    &mut pass,
                                    &g,
                                    &nw,
                                    &down,
                                    &up,
                                    b,
                                    &dev.hyper,
                                    x,
                                    low,
                                    y,
                                    pre.as_ref(),
                                    nt,
                                    bc,
                                    100,
                                )
                                .expect("HC v3 takes this shape");
                            } else {
                                if let Some(q) = &pre {
                                    inject_pre(c, p, &mut pass, &g, &dev.hyper, q, nt, bc);
                                }
                                encode_hc_old(
                                    c, p, &mut pass, &g, &nw, &down, &up, b, &dev.hyper, x, low, y,
                                    nt, bc, 100,
                                )
                                .expect("pre-v3 mix");
                            }
                        }
                        let out = submit_readback(
                            enc,
                            &[
                                (x, (TMAX * os * 4) as u64),
                                (low, (TMAX * ls * 4) as u64),
                                (y, (TMAX * gsd * 4) as u64),
                                (&dev.hyper, (TMAX * hs * 4) as u64),
                            ],
                        )
                        .expect("readback");
                        let mut parts = Vec::new();
                        let mut o = 0usize;
                        for n in [TMAX * os, TMAX * ls, TMAX * gsd, TMAX * hs] {
                            parts.push(f32s(&out[o..o + n * 4]));
                            o += (n * 4).div_ceil(16) * 16;
                        }
                        parts
                    };
                    let old = run(false);
                    let new = run(true);
                    let ex = rel(&new[0], &old[0], nt, os, hidden);
                    // the stream-split pair keeps its shares in its own buffer
                    let el = if hc4() { 0.0 } else { rel(&new[1], &old[1], nt, ls, lr) };
                    let ei = if with_b {
                        rel(&new[2], &old[2], nt, gsd, hc)
                    } else {
                        0.0
                    };
                    let eh = rel(&new[3], &old[3], nt, hs, hh);
                    eprintln!(
                        "nt {nt} gate rows {with_b:5} inject {with_pre:5}: x {ex:.2e} low {el:.2e} \
                         gate {ei:.2e} hyper {eh:.2e}"
                    );
                    worst = worst.max(ex).max(el).max(ei).max(eh);
                    // the rows past nt stay as they were
                    for t in nt..TMAX {
                        assert_eq!(
                            &new[3][t * hs..t * hs + hh],
                            &h0[t * hs..t * hs + hh],
                            "hyper row {t} past nt = {nt} touched"
                        );
                    }
                }
            }
        }
        assert!(worst < 1e-4, "HC v3 departs from the pre-v3 mix: {worst:e}");
    }
}

/// q4tp resident experts on the row-blocked kernels. The device test runs
/// `q4_gu_q4tp4` (picked by `Pipes::gu4`) against the arena's one-row
/// `dsv4_global_gate_up_q4tp`, then the blocked down kernel against the
/// one-row down kernel on each arm's activations, over a synthetic
/// eight-segment bank of Qwen3.8-Flash-Next's expert shape (inter 640,
/// hidden 2560) for 1, 4 and 8 tokens of ten slots spread over every
/// segment, with the host decode (`dequant_q4tp`) as a third witness for
/// gate/up. Needs an adapter with binding arrays (Metal and Vulkan have them):
/// `CMF_GPU=wgpu cargo test --release -p cortiq-engine --features gpu expert4_ -- --include-ignored --nocapture`
#[cfg(test)]
mod expert4_tests {
    use super::hc3_device_tests::{POISON, Rnd, f2h, rel, upload};
    use super::*;
    use cortiq_core::quant::{dequant_q4tp, expected_nbytes, q4tp_put_code, q4tp_sections};

    const INTER: usize = 640;
    const HIDDEN: usize = 2560;
    const SEGS: usize = 8;
    const PER_SEG: usize = 3;
    const SLOTS: usize = 10;

    fn geom(inter: usize, gu_q2: bool) -> Geom {
        Geom {
            hidden: HIDDEN,
            hc: 4,
            eps: 1e-6,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            index_heads: 4,
            index_dim: 128,
            index_budget: 2048,
            compress_ratio: 4,
            gdn: GdnGeom {
                nv: 32,
                nk: 16,
                dk: 128,
                dv: 128,
                kk: 4,
            },
            ple_kernel: 4,
            ple_dilation: 1,
            top_k: SLOTS,
            n_experts: 512,
            inter,
            gu_q2,
        }
    }

    /// q4tp banks take the blocked kernels on the same terms as q2tp ones.
    #[test]
    fn blocked_experts_take_q4tp_banks() {
        if std::env::var("CMF_QWEN_EXPERT4").as_deref() == Ok("0") {
            eprintln!("CMF_QWEN_EXPERT4=0 in the environment: skipped");
            return;
        }
        for gu_q2 in [true, false] {
            assert!(blocked_experts(&geom(INTER, gu_q2), 8));
            assert!(!blocked_experts(&geom(INTER, gu_q2), 16));
            assert!(!blocked_experts(&geom(INTER + 2, gu_q2), 8));
        }
        // the blocked gate/up kernel reads whole words: mat16 must be even
        let n = expected_nbytes(TensorDtype::Q4TiledP, &[INTER, HIDDEN]).unwrap();
        assert_eq!(n % 4, 0);
        for rows in (4..256).step_by(4) {
            for cols in [32, 64, 96, 160, 2560] {
                let n = expected_nbytes(TensorDtype::Q4TiledP, &[rows, cols]).unwrap();
                assert_eq!(n % 4, 0, "q4tp [{rows}, {cols}]");
            }
        }
    }

    /// A random q4tp matrix: uniform nibbles, rung codes over the whole
    /// ladder, per-row (lo, step) so the scales span 2^-10 .. 2^-2.
    fn q4tp_matrix(r: &mut Rnd, rows: usize, cols: usize) -> Vec<u8> {
        let mut b = vec![0u8; expected_nbytes(TensorDtype::Q4TiledP, &[rows, cols]).unwrap()];
        let gpr = cols / 32;
        let (poff, coff, cst) = q4tp_sections(rows, cols);
        for x in &mut b[..poff] {
            *x = (r.bits() >> 32) as u8;
        }
        for row in 0..rows {
            let lo = f2h(-10.0 + 0.3 * r.n());
            let st = f2h(0.25 + 0.01 * r.n());
            b[poff + row * 4..poff + row * 4 + 2].copy_from_slice(&lo.to_le_bytes());
            b[poff + row * 4 + 2..poff + row * 4 + 4].copy_from_slice(&st.to_le_bytes());
            let codes = &mut b[coff + row * cst..coff + (row + 1) * cst];
            for g in 0..gpr {
                q4tp_put_code(codes, g, (r.bits() >> 40) as usize & 31);
            }
        }
        b
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice(b).to_vec()
    }

    fn arr(v: &[wgpu::Buffer]) -> Vec<wgpu::BufferBinding<'_>> {
        v.iter()
            .map(wgpu::Buffer::as_entire_buffer_binding)
            .collect()
    }

    fn arr_entry<'a>(b: u32, a: &'a [wgpu::BufferBinding<'a>]) -> wgpu::BindGroupEntry<'a> {
        wgpu::BindGroupEntry {
            binding: b,
            resource: wgpu::BindingResource::BufferArray(a),
        }
    }

    #[test]
    #[ignore = "needs a GPU with binding arrays"]
    fn expert4_q4tp_matches_one_row_kernels() {
        let Some(c) = ctx() else {
            eprintln!("no wgpu device (CMF_GPU=wgpu on macOS): skipped");
            return;
        };
        let Some(p) = pipes(c) else {
            eprintln!("qwen4 kernels unavailable on this adapter: skipped");
            return;
        };
        let Some((p_gu, p_dn)) = dsv4_global_moe_pipelines(c, false, SEGS) else {
            eprintln!(
                "no global q4tp arena kernels on {}: skipped",
                c.adapter_info.name
            );
            return;
        };
        let g = geom(INTER, false);
        assert!(blocked_experts(&g, SEGS) || std::env::var_os("CMF_QWEN_EXPERT4").is_some());
        eprintln!(
            "adapter: {} ({:?})",
            c.adapter_info.name, c.adapter_info.backend
        );
        let mut r = Rnd(0x51F1_5EED_0F4E_0E4D);
        let n_exp = SEGS * PER_SEG;
        let (gu_len, d_len) = (
            expected_nbytes(TensorDtype::Q4TiledP, &[INTER, HIDDEN]).unwrap(),
            expected_nbytes(TensorDtype::Q4TiledP, &[HIDDEN, INTER]).unwrap(),
        );
        // expert e lives in segment e / PER_SEG at local slot e % PER_SEG
        let mut gate_m = Vec::new();
        let mut up_m = Vec::new();
        let mut down_m = Vec::new();
        for _ in 0..n_exp {
            gate_m.push(q4tp_matrix(&mut r, INTER, HIDDEN));
            up_m.push(q4tp_matrix(&mut r, INTER, HIDDEN));
            down_m.push(q4tp_matrix(&mut r, HIDDEN, INTER));
        }
        let bank = |mats: &[Vec<u8>], label: &str| -> Vec<wgpu::Buffer> {
            (0..SEGS)
                .map(|s| upload(c, label, &mats[s * PER_SEG..(s + 1) * PER_SEG].concat()))
                .collect()
        };
        let (gate_b, up_b, down_b) = (
            bank(&gate_m, "e4-gate"),
            bank(&up_m, "e4-up"),
            bank(&down_m, "e4-down"),
        );
        let (gate_a, up_a, down_a) = (arr(&gate_b), arr(&up_b), arr(&down_b));
        let tmax = 8usize;
        let x0: Vec<f32> = (0..tmax * HIDDEN).map(|_| r.n()).collect();
        // every token's ten slots: distinct experts, all segments in use
        let mut sel0 = vec![0u32; tmax * SLOTS];
        for t in 0..tmax {
            let off = (r.bits() >> 33) as usize % n_exp;
            for s in 0..SLOTS {
                sel0[t * SLOTS + s] = ((off + s * 7) % n_exp) as u32;
            }
        }
        let wt0: Vec<f32> = (0..tmax * SLOTS)
            .map(|_| 0.05 + r.n().abs() * 0.1)
            .collect();
        let x = upload(c, "e4-x", bytemuck::cast_slice(&x0));
        let sel = upload(c, "e4-sel", bytemuck::cast_slice(&sel0));
        let wt = upload(c, "e4-wt", bytemuck::cast_slice(&wt0));
        let act_n = tmax * SLOTS * INTER;
        let (act_old, act_new) = (
            storage_buf(c, "e4-act-old", (act_n * 4) as u64),
            storage_buf(c, "e4-act-new", (act_n * 4) as u64),
        );
        let (y_old, y_new) = (
            storage_buf(c, "e4-y-old", (tmax * HIDDEN * 4) as u64),
            storage_buf(c, "e4-y-new", (tmax * HIDDEN * 4) as u64),
        );
        let gu_u = uniform_u32x8(
            c,
            [
                (HIDDEN / 32) as u32,
                INTER as u32,
                SLOTS as u32,
                (gu_len / 2) as u32,
                0.0f32.to_bits(),
                PER_SEG as u32,
                0,
                0,
            ],
        );
        let dn_u = uniform_u32x8(
            c,
            [
                (INTER / 32) as u32,
                HIDDEN as u32,
                SLOTS as u32,
                (d_len / 2) as u32,
                PER_SEG as u32,
                0,
                0,
                0,
            ],
        );
        // the vec4-load twins where this shape takes them (CMF_QWEN_V4LOAD=0: the u32 ones)
        let v4 = v4load() && q4tp_words16(INTER, HIDDEN) && q4tp_words16(HIDDEN, INTER);
        let gu4 = if v4 { p.experts_blocked(&g).0 } else { p.gu4(false) };
        eprintln!("blocked arm: {}", if v4 { "q4_gu_q4tp4v + the vec4 down" } else { "q4_gu_q4tp4 + q4_dn_q4tp4" });
        let bg = |pipe: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]| {
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("e4-test"),
                layout: &pipe.get_bind_group_layout(0),
                entries,
            })
        };
        let bgp = |pipe: &wgpu::ComputePipeline, u: &wgpu::Buffer| {
            c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("e4-test-p"),
                layout: &pipe.get_bind_group_layout(1),
                entries: &[bind_buf(0, u)],
            })
        };
        // (pipeline, group 0, group 1) for the one-row and the blocked arm
        let gu_old = (
            p_gu,
            bg(
                p_gu,
                &[
                    arr_entry(0, &gate_a),
                    arr_entry(1, &up_a),
                    bind_buf(2, &x),
                    bind_buf(3, &sel),
                    bind_buf(4, &act_old),
                    bind_buf(5, &wt),
                ],
            ),
            bgp(p_gu, &gu_u),
        );
        let gu_new = (
            gu4,
            bg(
                gu4,
                &[
                    arr_entry(0, &gate_a),
                    arr_entry(1, &up_a),
                    bind_buf(2, &x),
                    bind_buf(3, &sel),
                    bind_buf(4, &act_new),
                ],
            ),
            bgp(gu4, &gu_u),
        );
        let dn = |pipe: &'static wgpu::ComputePipeline, act: &wgpu::Buffer, y: &wgpu::Buffer| {
            (
                pipe,
                bg(
                    pipe,
                    &[
                        arr_entry(0, &down_a),
                        bind_buf(1, act),
                        bind_buf(2, &sel),
                        bind_buf(3, &wt),
                        bind_buf(4, y),
                    ],
                ),
                bgp(pipe, &dn_u),
            )
        };
        let dn_old = dn(p_dn, &act_old, &y_old);
        let dn_new = dn(if v4 { p.dn4sv.as_ref().unwrap_or(&p.dn_q4tp4) } else { &p.dn_q4tp4 }, &act_new, &y_new);
        type Arm<'a> = (&'a wgpu::ComputePipeline, wgpu::BindGroup, wgpu::BindGroup);
        let launch = |pass: &mut PassHandle, a: &Arm, wg: (u32, u32, u32)| {
            pass.set_pipeline(a.0);
            pass.set_bind_group(0, &a.1, &[]);
            pass.set_bind_group(1, &a.2, &[]);
            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
        };
        let poison = |b: &wgpu::Buffer, n: usize| {
            c.queue
                .write_buffer(b, 0, bytemuck::cast_slice(&vec![POISON; n]));
        };
        let mut worst = (0.0f32, 0.0f32, 0.0f32);
        for nt in [1usize, 4, 8] {
            for b in [&act_old, &act_new] {
                poison(b, act_n);
            }
            for b in [&y_old, &y_new] {
                poison(b, tmax * HIDDEN);
            }
            let mut enc = new_encoder("e4-test").expect("encoder");
            {
                let mut pass = begin_pass(&mut enc);
                let n = nt as u32;
                launch(&mut pass, &gu_old, (INTER as u32, SLOTS as u32, n));
                launch(&mut pass, &dn_old, (HIDDEN as u32, n, 1));
                launch(&mut pass, &gu_new, ((INTER / 4) as u32, SLOTS as u32, n));
                launch(&mut pass, &dn_new, ((HIDDEN / 4) as u32, n, 1));
            }
            let (an, yn) = ((act_n * 4) as u64, (tmax * HIDDEN * 4) as u64);
            let out = submit_readback(
                enc,
                &[(&act_old, an), (&act_new, an), (&y_old, yn), (&y_new, yn)],
            )
            .expect("readback");
            let (an, yn) = (an as usize, yn as usize);
            let a_old = f32s(&out[..an]);
            let a_new = f32s(&out[an..2 * an]);
            let y_o = f32s(&out[2 * an..2 * an + yn]);
            let y_n = f32s(&out[2 * an + yn..2 * an + 2 * yn]);
            // the host decode of every (token, slot)'s gate/up, f64 sums
            let mut a_host = vec![0.0f32; nt * SLOTS * INTER];
            let (mut wg, mut wu) = (vec![0.0f32; INTER * HIDDEN], vec![0.0f32; INTER * HIDDEN]);
            for e in 0..n_exp {
                let users: Vec<usize> = (0..nt * SLOTS)
                    .filter(|&bs| sel0[bs] as usize == e)
                    .collect();
                if users.is_empty() {
                    continue;
                }
                dequant_q4tp(&gate_m[e], INTER, HIDDEN, &mut wg);
                dequant_q4tp(&up_m[e], INTER, HIDDEN, &mut wu);
                for bs in users {
                    let xt = &x0[(bs / SLOTS) * HIDDEN..(bs / SLOTS + 1) * HIDDEN];
                    for row in 0..INTER {
                        let (mut sg, mut su) = (0.0f64, 0.0f64);
                        for (j, &xv) in xt.iter().enumerate() {
                            sg += f64::from(wg[row * HIDDEN + j]) * f64::from(xv);
                            su += f64::from(wu[row * HIDDEN + j]) * f64::from(xv);
                        }
                        a_host[bs * INTER + row] = ((sg / (1.0 + (-sg).exp())) * su) as f32;
                    }
                }
            }
            let rows = nt * SLOTS;
            let e_blk = rel(&a_new, &a_old, rows, INTER, INTER);
            let e_old_h = rel(&a_old, &a_host, rows, INTER, INTER);
            let e_new_h = rel(&a_new, &a_host, rows, INTER, INTER);
            let e_y = rel(&y_n, &y_o, nt, HIDDEN, HIDDEN);
            eprintln!(
                "nt {nt}: gate/up blocked vs one-row {e_blk:.2e} (one-row vs host {e_old_h:.2e}, \
                 blocked vs host {e_new_h:.2e}); down output blocked vs one-row {e_y:.2e}"
            );
            // tokens past nt untouched by either arm
            for bs in rows * INTER..act_n {
                assert!(
                    a_old[bs] == POISON && a_new[bs] == POISON,
                    "act {bs} past nt = {nt}"
                );
            }
            worst.0 = worst.0.max(e_blk);
            worst.1 = worst.1.max(e_new_h).max(e_old_h);
            worst.2 = worst.2.max(e_y);
        }
        assert!(
            worst.0 < 1e-5,
            "blocked q4tp gate/up departs: {:e}",
            worst.0
        );
        assert!(
            worst.1 < 1e-4,
            "q4tp gate/up departs from the host: {:e}",
            worst.1
        );
        assert!(
            worst.2 < 1e-5,
            "blocked q4tp down chain departs: {:e}",
            worst.2
        );
        // wall time of the gate/up dispatch alone, many in one pass (the
        // adapter here is not the target card: a ratio, not a number to quote)
        const REPS: usize = 200;
        for nt in [1usize, 8] {
            let mut times = [0.0f64; 2];
            for _round in 0..2 {
                for (i, (arm, wg)) in [
                    (&gu_old, (INTER as u32, SLOTS as u32, nt as u32)),
                    (&gu_new, ((INTER / 4) as u32, SLOTS as u32, nt as u32)),
                ]
                .into_iter()
                .enumerate()
                {
                    let mut enc = new_encoder("e4-time").expect("encoder");
                    {
                        let mut pass = begin_pass(&mut enc);
                        for _ in 0..REPS {
                            launch(&mut pass, arm, wg);
                        }
                    }
                    let t0 = std::time::Instant::now();
                    submit_readback(enc, &[(&sel, 16)]).expect("readback");
                    times[i] = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;
                }
            }
            eprintln!(
                "nt {nt}: gate/up {:.1} µs one-row, {:.1} µs blocked a dispatch",
                times[0], times[1]
            );
        }
    }
}

// ───────────────────── GPTQ calibration capture ─────────────────────
//
// `capture_begin(lo, hi)` makes every layer frame in [lo, hi) copy its MoE
// input rows (`x2`, the vector the router and the experts read) and its
// router logits into a per-layer buffer right after routing; the caller
// reads them after each `forward_tokens` (≤ TMAX tokens). The experts'
// input Hessians are then built from these rows on the host side.

struct Capture {
    lo: usize,
    hi: usize,
    width: usize,
    bufs: Vec<wgpu::Buffer>,
    written: usize,
    pipe: wgpu::ComputePipeline,
}

static CAPTURE: std::sync::Mutex<Option<Capture>> = std::sync::Mutex::new(None);

const CAPTURE_SRC: &str = r#"
struct P { nt: u32, hidden: u32, ne: u32, xs: u32, ls: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x2: array<f32>;
@group(0) @binding(2) var<storage, read> lg: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn cap_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = p.hidden + p.ne;
    let i = g.x;
    if (i >= p.nt * w) { return; }
    let t = i / w;
    let c = i % w;
    if (c < p.hidden) { dst[i] = x2[t * p.xs + c]; } else { dst[i] = lg[t * p.ls + (c - p.hidden)]; }
}
"#;

/// Start capturing layers `lo..hi` (`hidden` input width, `ne` experts).
pub fn capture_begin(lo: usize, hi: usize, hidden: usize, ne: usize) -> bool {
    let Some(c) = ctx() else { return false };
    let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("qwen4-capture"),
        source: wgpu::ShaderSource::Wgsl(CAPTURE_SRC.into()),
    });
    let pipe = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("qwen4-capture"),
        layout: None,
        module: &m,
        entry_point: Some("cap_rows"),
        compilation_options: crate::gpu_wgpu::pco(),
        cache: None,
    });
    let width = hidden + ne;
    let bufs = (lo..hi)
        .map(|_| {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("qwen4-capture"),
                size: (TMAX * width * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    *CAPTURE.lock().unwrap() = Some(Capture { lo, hi, width, bufs, written: 0, pipe });
    true
}

pub fn capture_end() {
    *CAPTURE.lock().unwrap() = None;
}

#[allow(clippy::too_many_arguments)]
fn capture_rows(c: &Ctx, pass: &mut wgpu::ComputePass<'_>, li: usize, nt: usize, x2: &wgpu::Buffer, xs: u32, lg: &wgpu::Buffer, ls: u32, hidden: usize, ne: usize) {
    let mut g = CAPTURE.lock().unwrap();
    let Some(cap) = g.as_mut() else { return };
    if li < cap.lo || li >= cap.hi || cap.width != hidden + ne {
        return;
    }
    let u = uniform_u32x8(c, [nt as u32, hidden as u32, ne as u32, xs, ls, 0, 0, 0]);
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qwen4-capture"),
        layout: &cap.pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: lg.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cap.bufs[li - cap.lo].as_entire_binding() },
        ],
    });
    pass.set_pipeline(&cap.pipe);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(((nt * cap.width).div_ceil(256)) as u32, 1, 1);
    cap.written += 1;
}

/// The captured rows of the last forward (`nt` tokens): per layer of the
/// range, `nt × (hidden + ne)` floats (MoE input then router logits), and
/// how many layer frames wrote since the last read.
pub fn capture_read(nt: usize) -> Option<(Vec<Vec<f32>>, usize)> {
    let c = ctx()?;
    let mut g = CAPTURE.lock().unwrap();
    let cap = g.as_mut()?;
    let n = nt * cap.width;
    let size = (n * 4) as u64;
    let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("qwen4-capture-rb"),
        size: size * cap.bufs.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("qwen4-capture-rb") });
    for (k, b) in cap.bufs.iter().enumerate() {
        enc.copy_buffer_to_buffer(b, 0, &stage, size * k as u64, size);
    }
    submit(c, enc.finish());
    let slice = stage.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let out = {
        let data = slice.get_mapped_range().ok()?;
        let all: &[f32] = bytemuck::cast_slice(&data);
        (0..cap.bufs.len()).map(|k| all[k * n..(k + 1) * n].to_vec()).collect()
    };
    stage.unmap();
    let w = cap.written;
    cap.written = 0;
    Some((out, w))
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    /// The Qwen3.5-family vocabulary the Flash-Next `lm_head` projects to.
    const VOCAB: usize = 248_320;

    #[test]
    fn verify_window_logits_fit_the_stage() {
        let ls = tstride(VOCAB * 4);
        assert_eq!(ls, 993_280);
        // the initial stage holds a window of four rows and not five: what
        // made MTP k = 4 (a five-position verify window) crash
        assert!(chain_stage_bytes(0, 0, 4, ls) <= STAGE_MIN);
        assert!(chain_stage_bytes(0, 0, 5, ls) > STAGE_MIN);
        for rows in 1..=TMAX {
            let need = chain_stage_bytes(0, 0, rows, ls);
            let size = stage_size(need);
            assert!(size >= need, "window {rows}: stage {size} < {need}");
            assert!(size >= STAGE_MIN && size % (1 << 20) == 0);
        }
        assert_eq!(stage_size(0), STAGE_MIN);
        assert_eq!(stage_size(STAGE_MIN + 1), STAGE_MIN + (1 << 20));
        // layer frames first, the logits rows 16-byte aligned after them
        assert_eq!(chain_stage_bytes(3, 1024, 2, 10), 3 * 1024 + 32);
        assert_eq!(chain_stage_bytes(2, 512, 0, ls), 1024);
    }

    #[test]
    fn snapshot_bytes_follow_the_window() {
        let g = Geom {
            hidden: 2048,
            hc: 4,
            eps: 1e-6,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            index_heads: 4,
            index_dim: 128,
            index_budget: 2048,
            compress_ratio: 4,
            gdn: GdnGeom {
                nv: 32,
                nk: 16,
                dk: 128,
                dv: 128,
                kk: 4,
            },
            ple_kernel: 4,
            ple_dilation: 1,
            top_k: 10,
            n_experts: 512,
            inter: 512,
            gu_q2: true,
        };
        // conv ring (kk-1)·cdim then S = nv·dk·dv, f32
        let gdn_row = ((4 - 1) * (2 * 16 * 128 + 32 * 128) + 32 * 128 * 128) * 4;
        // the PLE history ring, (kernel-1)·dilation rows of hc·hidden
        let ple_row = 3 * 4 * 2048 * 4;
        let kinds = [(true, false), (false, true), (false, false), (true, true)];
        assert_eq!(Dev::snap_bytes(&g, &kinds, 0), 0);
        assert_eq!(
            Dev::snap_bytes(&g, &kinds, 1),
            (2 * gdn_row + 2 * ple_row) as u64
        );
        assert_eq!(
            Dev::snap_bytes(&g, &kinds, 8),
            2 * Dev::snap_bytes(&g, &kinds, 4)
        );
    }
}
