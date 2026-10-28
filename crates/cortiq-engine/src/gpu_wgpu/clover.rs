//! Block-diffusion preview kernels of the batched graph (Clover).
//!
//! A preview runs the whole block over the committed state without
//! advancing it. Three kernels differ from the causal commit:
//!
//! * `bdp_conv` — the GDN conv over [ring | block] plus the right taps
//!   that read the next kk−1 rows of the block, then SiLU; the ring is
//!   read, never shifted;
//! * `bdp_mix` — per (value head, row) the block softmax over the rows'
//!   l2-normalized keys (logits scaled per head, no β) and one history
//!   slot whose value is the clean state's readout decayed by the row's
//!   gate, then the gated RMSNorm of the causal step — S is read only;
//! * `bdp_coreset` — the block's expert coreset over the router logits
//!   of every row: the top-k softmax mass per expert, the smallest prefix
//!   (descending) holding `top_p`, every other expert's logit to −1e30 so
//!   the ordinary select kernel routes inside the coreset.
//!
//! Host twins: `linear_core::gdn_preview_batch`, `pipeline::clover`.

use super::*;

pub(crate) const CLOVER_WGSL: &str = r#"
// ── preview conv: causal taps over [ring | block] + right taps, SiLU ──
struct BcP { cdim: u32, kk: u32, kb: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       bc_qkv  : array<f32>;
@group(0) @binding(1) var<storage, read>       bc_taps : array<f32>;
@group(0) @binding(2) var<storage, read>       bc_right: array<f32>;
@group(0) @binding(3) var<storage, read>       bc_ring : array<f32>;
@group(0) @binding(4) var<storage, read_write> bc_cq   : array<f32>;
@group(0) @binding(5) var<uniform>             bc_p    : BcP;
@compute @workgroup_size(256)
fn bdp_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let t = gid.y;
    let cdim = bc_p.cdim;
    let kk = bc_p.kk;
    let kb = bc_p.kb;
    if (c >= cdim || t >= kb) { return; }
    var acc = 0.0;
    for (var j = 0u; j < kk; j = j + 1u) {
        // index into [ring (kk-1 rows) | block]
        let i = t + j;
        var x: f32;
        if (i + 1u < kk) {
            x = bc_ring[i * cdim + c];
        } else {
            x = bc_qkv[(i + 1u - kk) * cdim + c];
        }
        acc = acc + x * bc_taps[c * kk + j];
    }
    for (var d = 1u; d < kk; d = d + 1u) {
        if (t + d < kb) {
            acc = acc + bc_qkv[(t + d) * cdim + c] * bc_right[c * (kk - 1u) + d - 1u];
        }
    }
    bc_cq[t * cdim + c] = acc / (1.0 + exp(-acc));
}

// ── preview mixer: block softmax + history slot, gated RMSNorm ──
struct BmP {
    nv: u32, dk: u32, dv: u32, kd: u32,
    rep: u32, cdim: u32, eps: f32, kb: u32,
    len_log: f32, _p0: u32, _p1: u32, _p2: u32,
};
@group(0) @binding(0)  var<storage, read>       bm_cq   : array<f32>;
@group(0) @binding(1)  var<storage, read>       bm_z    : array<f32>;
@group(0) @binding(2)  var<storage, read>       bm_a    : array<f32>;
@group(0) @binding(3)  var<storage, read>       bm_alog : array<f32>;
@group(0) @binding(4)  var<storage, read>       bm_dtb  : array<f32>;
@group(0) @binding(5)  var<storage, read>       bm_norm : array<f32>;
@group(0) @binding(6)  var<storage, read>       bm_S    : array<f32>;
@group(0) @binding(7)  var<storage, read_write> bm_o    : array<f32>;
@group(0) @binding(8)  var<uniform>             bm_p    : BmP;
@group(0) @binding(9)  var<storage, read>       bm_lsc  : array<f32>;
@group(0) @binding(10) var<storage, read>       bm_hb   : array<f32>;
@group(0) @binding(11) var<storage, read>       bm_hc   : array<f32>;
// Four rows a workgroup (BMR): the key norms once for all of them, each S
// column read once for two rows (a lane owns a column and two rows).
const BMR: u32 = 4u;
var<workgroup> bm_q  : array<f32, 1024>;
var<workgroup> bm_kn : array<f32, 256>;
var<workgroup> bm_w  : array<f32, 1028>;
var<workgroup> bm_ov : array<f32, 1024>;
var<workgroup> bm_nr : array<f32, 4>;
fn bm_softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}
@compute @workgroup_size(256)
fn bdp_mix(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let t0 = wid.y * BMR;
    let l = lid.x;
    let dk = bm_p.dk;
    let dv = bm_p.dv;
    let kb = bm_p.kb;
    let cdim = bm_p.cdim;
    let ko = h / bm_p.rep;
    let nr = min(BMR, kb - t0);
    // |q| of the rows (lane rr) and |k| of every block row (lane s)
    if (l < nr) {
        let qs = (t0 + l) * cdim + ko * dk;
        var nq = 0.0;
        for (var d = 0u; d < dk; d = d + 1u) { nq = nq + bm_cq[qs + d] * bm_cq[qs + d]; }
        bm_nr[l] = 1.0 / sqrt(nq + 1e-6);
    }
    if (l < kb) {
        let ks = l * cdim + bm_p.kd + ko * dk;
        var nk = 0.0;
        for (var d = 0u; d < dk; d = d + 1u) { nk = nk + bm_cq[ks + d] * bm_cq[ks + d]; }
        bm_kn[l] = 1.0 / sqrt(nk + 1e-6);
    }
    workgroupBarrier();
    for (var i = l; i < nr * dk; i = i + 256u) {
        let rr = i / dk;
        let d = i % dk;
        bm_q[rr * dk + d] = bm_cq[(t0 + rr) * cdim + ko * dk + d] * bm_nr[rr];
    }
    workgroupBarrier();
    // logits: lane p = (row, key); the history slot after each row's keys
    let w1 = kb + 1u;
    for (var p = l; p < nr * kb; p = p + 256u) {
        let rr = p / kb;
        let sx = p % kb;
        let ks = sx * cdim + bm_p.kd + ko * dk;
        var dot = 0.0;
        for (var d = 0u; d < dk; d = d + 1u) { dot = dot + bm_q[rr * dk + d] * bm_cq[ks + d]; }
        bm_w[rr * w1 + sx] = dot * bm_kn[sx] * bm_lsc[h];
    }
    if (l < nr) { bm_w[l * w1 + kb] = bm_hb[h] + bm_hc[h] * bm_p.len_log; }
    workgroupBarrier();
    if (l < nr) {
        let b0 = l * w1;
        var mx = bm_w[b0];
        for (var sx = 1u; sx <= kb; sx = sx + 1u) { mx = max(mx, bm_w[b0 + sx]); }
        var sum = 0.0;
        for (var sx = 0u; sx <= kb; sx = sx + 1u) {
            let e = exp(bm_w[b0 + sx] - mx);
            bm_w[b0 + sx] = e;
            sum = sum + e;
        }
        for (var sx = 0u; sx <= kb; sx = sx + 1u) { bm_w[b0 + sx] = bm_w[b0 + sx] / sum; }
        // the history weight times the row's decay and 1/√dk
        let g = -exp(bm_alog[h]) * bm_softplus(bm_a[(t0 + l) * bm_p.nv + h] + bm_dtb[h]);
        bm_w[b0 + kb] = bm_w[b0 + kb] * exp(g) / sqrt(f32(dk));
    }
    workgroupBarrier();
    // outputs: a lane owns column c and rows r0, r0 + 2 (dv ≤ 128) or
    // strides the (row, column) pairs
    let sb = h * dk * dv;
    let vo = 2u * bm_p.kd + h * dv;
    for (var i = l; i < 2u * dv; i = i + 256u) {
        let c = i % dv;
        let r0 = i / dv;
        let r1 = r0 + 2u;
        var o0 = 0.0;
        var o1 = 0.0;
        for (var di = 0u; di < dk; di = di + 1u) {
            let sv = bm_S[sb + di * dv + c];
            o0 = o0 + bm_q[r0 * dk + di] * sv;
            o1 = o1 + bm_q[min(r1, 3u) * dk + di] * sv;
        }
        o0 = o0 * bm_w[r0 * w1 + kb];
        o1 = o1 * bm_w[min(r1, 3u) * w1 + kb];
        for (var sx = 0u; sx < kb; sx = sx + 1u) {
            let v = bm_cq[sx * cdim + vo + c];
            o0 = o0 + bm_w[r0 * w1 + sx] * v;
            o1 = o1 + bm_w[min(r1, 3u) * w1 + sx] * v;
        }
        if (r0 < nr) { bm_ov[r0 * dv + c] = o0; }
        if (r1 < nr) { bm_ov[r1 * dv + c] = o1; }
    }
    workgroupBarrier();
    // gated RMSNorm per row
    if (l < nr) {
        var ss = 0.0;
        for (var c = 0u; c < dv; c = c + 1u) { ss = ss + bm_ov[l * dv + c] * bm_ov[l * dv + c]; }
        bm_nr[l] = 1.0 / sqrt(ss / f32(dv) + bm_p.eps);
    }
    workgroupBarrier();
    for (var i = l; i < nr * dv; i = i + 256u) {
        let rr = i / dv;
        let c = i % dv;
        let zo = (t0 + rr) * bm_p.nv * dv + h * dv + c;
        let zz = bm_z[zo];
        bm_o[zo] = bm_ov[rr * dv + c] * bm_nr[rr] * bm_norm[c] * (zz / (1.0 + exp(-zz)));
    }
}

// ── block expert coreset over the router logits [kb × ne], two phases ──
// rows: one workgroup a row — the row's softmax probability of each of its
// top-k experts (0 elsewhere) into `cs_part`; pick: one workgroup — the
// per-expert totals, the descending prefix that holds `top_p`, every other
// expert's logit to -1e30 in every row.
struct CsP { ne: u32, kb: u32, topk: u32, top_p: f32 };
@group(0) @binding(0) var<storage, read_write> cs_lg  : array<f32>;
@group(0) @binding(1) var<uniform>             cs_p   : CsP;
@group(0) @binding(2) var<storage, read_write> cs_part: array<f32>;
var<workgroup> cs_row: array<f32, 256>;
var<workgroup> cs_red: array<f32, 256>;
fn cs_max(t: u32) -> f32 {
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { cs_red[t] = max(cs_red[t], cs_red[t + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return cs_red[0];
}
fn cs_sum(t: u32) -> f32 {
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { cs_red[t] = cs_red[t] + cs_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return cs_red[0];
}
@compute @workgroup_size(256)
fn bdp_coreset_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let e = lid.x;
    let t = wid.x;
    let ne = cs_p.ne;
    let v = select(-3.0e38, cs_lg[t * ne + min(e, ne - 1u)], e < ne);
    cs_row[e] = v;
    cs_red[e] = v;
    workgroupBarrier();
    let mx = cs_max(e);
    workgroupBarrier();
    let ex = select(0.0, exp(v - mx), e < ne);
    cs_red[e] = ex;
    workgroupBarrier();
    let sum = cs_sum(e);
    // rank within the row: descending, the lower index first on ties
    var rank = 0u;
    for (var j = 0u; j < ne; j = j + 1u) {
        let w = cs_row[j];
        if (w > v || (w == v && j < e)) { rank = rank + 1u; }
    }
    if (e < ne) { cs_part[t * ne + e] = select(0.0, ex / sum, rank < cs_p.topk); }
}
var<workgroup> cs_tot: array<f32, 256>;
@compute @workgroup_size(256)
fn bdp_coreset_pick(@builtin(local_invocation_id) lid: vec3<u32>) {
    let e = lid.x;
    let ne = cs_p.ne;
    var acc = 0.0;
    if (e < ne) {
        for (var t = 0u; t < cs_p.kb; t = t + 1u) { acc = acc + cs_part[t * ne + e]; }
    }
    cs_tot[e] = acc;
    cs_red[e] = acc;
    workgroupBarrier();
    let total = max(cs_sum(e), 1e-9);
    var rank = 0u;
    var before = 0.0;
    for (var j = 0u; j < ne; j = j + 1u) {
        let w = cs_tot[j];
        if (w > acc || (w == acc && j < e)) {
            rank = rank + 1u;
            before = before + w;
        }
    }
    let keep = rank == 0u || before / total < cs_p.top_p;
    if (e < ne && !keep) {
        for (var t = 0u; t < cs_p.kb; t = t + 1u) { cs_lg[t * ne + e] = -1.0e30; }
    }
}
"#;

/// Expert-grouped MoE of a block over its non-empty tiles only. The
/// grouped prompt kernels (`moe_r4::moeg_gu` / `moeg_dn`, a row a thread,
/// eight tokens a tile) launch a buckets × tiles grid — 257 × 4 for a
/// 32-row block — of which a 15-expert coreset fills a few dozen; the
/// empty workgroups cost a fifth of the MoE time. `bdp_moe_tiles` lists the
/// non-empty (expert, tile) pairs and writes the indirect grids, and the
/// same kernels read their pair from the list (`moeg_tiles_wgsl`).
pub(crate) const MOE_WGSL: &str = r#"
const BMT: u32 = 8u;

// ── the non-empty (expert, tile) pairs and the indirect grids ──
// After `moeg_group`: bucket e holds off[e+1]−off[e] entries, a tile is
// BMT of them. The pairs land in `tl_list` as (e << 16) | tile, and the
// two grids [rows_gu, n, 1], [rows_dn, n, 1] in `tl_args`.
struct TlP { buckets: u32, rows_gu: u32, rows_dn: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       tl_off : array<u32>;
@group(0) @binding(1) var<storage, read_write> tl_list: array<u32>;
@group(0) @binding(2) var<storage, read_write> tl_args: array<u32>;
@group(0) @binding(3) var<uniform>             tl_p   : TlP;
var<workgroup> tl_n: array<u32, 512>;
var<workgroup> tl_base: array<u32, 512>;
@compute @workgroup_size(256)
fn bdp_moe_tiles(@builtin(local_invocation_index) lid: u32) {
    let nb = min(tl_p.buckets, 512u);
    for (var e = lid; e < nb; e = e + 256u) {
        tl_n[e] = (tl_off[e + 1u] - tl_off[e] + BMT - 1u) / BMT;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var s = 0u;
        for (var e = 0u; e < nb; e = e + 1u) {
            tl_base[e] = s;
            s = s + tl_n[e];
        }
        tl_args[0] = tl_p.rows_gu;
        tl_args[1] = s;
        tl_args[2] = 1u;
        tl_args[3] = tl_p.rows_dn;
        tl_args[4] = s;
        tl_args[5] = 1u;
    }
    workgroupBarrier();
    for (var e = lid; e < nb; e = e + 256u) {
        for (var t = 0u; t < tl_n[e]; t = t + 1u) {
            tl_list[tl_base[e] + t] = (e << 16u) | t;
        }
    }
}
"#;

/// The greedy pick of a block preview on the device: per row the response
/// penalties (`pipeline::clover::Penalties::apply`), the first maximum and
/// its softmax probability — 8 bytes a row back instead of the logits.
pub(crate) const PICK_WGSL: &str = r#"
struct PkP {
    vocab: u32, stride: u32, n_stop: u32, n_forbid: u32,
    think: u32, close_open: u32, len0: u32, _p: u32,
    freq: f32, pres: f32, rep: f32, lp: f32,
    lp_start: f32, lp_win: f32, tb: f32, tb_start: f32,
    tb_win: f32, _q0: f32, _q1: f32, _q2: f32,
};
@group(0) @binding(0) var<storage, read_write> pk_lg    : array<f32>;
@group(0) @binding(1) var<storage, read>       pk_cnt   : array<u32>;
@group(0) @binding(2) var<storage, read>       pk_stop  : array<u32>;
@group(0) @binding(3) var<storage, read>       pk_forbid: array<u32>;
@group(0) @binding(4) var<storage, read_write> pk_out   : array<u32>;
@group(0) @binding(5) var<uniform>             pk_p     : PkP;
var<workgroup> pk_m: array<f32, 256>;
var<workgroup> pk_i: array<u32, 256>;
@compute @workgroup_size(256)
fn bdp_pick(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let r = wid.x;
    let base = r * pk_p.stride;
    let pos = f32(pk_p.len0 + r + 1u);
    let ramp_l = clamp((pos - pk_p.lp_start) / pk_p.lp_win, 0.0, 1.0) * pk_p.lp;
    let ramp_t = select(0.0, clamp((pos - pk_p.tb_start) / pk_p.tb_win, 0.0, 1.0) * pk_p.tb,
                        pk_p.close_open != 0u);
    var m = -3.4e38;
    var mi = 0u;
    for (var v = lid; v < pk_p.vocab; v = v + 256u) {
        var l = pk_lg[base + v];
        var forbidden = false;
        for (var i = 0u; i < pk_p.n_forbid; i = i + 1u) {
            if (pk_forbid[i] == v) { forbidden = true; }
        }
        if (forbidden) {
            l = -3.4e38;
        } else {
            let c = pk_cnt[v];
            if (c > 0u) {
                l = l - f32(c) * pk_p.freq;
                l = l - pk_p.pres;
                if (pk_p.rep != 1.0) { l = select(l / pk_p.rep, l * pk_p.rep, l < 0.0); }
            }
            if (pk_p.lp != 0.0) {
                for (var i = 0u; i < pk_p.n_stop; i = i + 1u) {
                    if (pk_stop[i] == v) { l = l + ramp_l; }
                }
            }
            if (v == pk_p.think) { l = l + ramp_t; }
        }
        pk_lg[base + v] = l;
        if (l > m) { m = l; mi = v; }
    }
    pk_m[lid] = m;
    pk_i[lid] = mi;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            let om = pk_m[lid + stride];
            let oi = pk_i[lid + stride];
            if (om > pk_m[lid] || (om == pk_m[lid] && oi < pk_i[lid])) {
                pk_m[lid] = om;
                pk_i[lid] = oi;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let mx = pk_m[0];
    let id = pk_i[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var v = lid; v < pk_p.vocab; v = v + 256u) { sum = sum + exp(pk_lg[base + v] - mx); }
    pk_m[lid] = sum;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { pk_m[lid] = pk_m[lid] + pk_m[lid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid == 0u) {
        pk_out[2u * r] = id;
        pk_out[2u * r + 1u] = bitcast<u32>(1.0 / pk_m[0]);
    }
}
"#;

/// GPTQ calibration capture on the batched graph: a layer's MoE input rows
/// and router logits, interleaved `[k][hidden + ne]` (the layout
/// `qwen4::capture_read` hands `quantize-moe`).
pub(crate) const CAP_WGSL: &str = r#"
struct CpP { k: u32, hidden: u32, ne: u32, _p: u32 };
@group(0) @binding(0) var<uniform>             cp_p  : CpP;
@group(0) @binding(1) var<storage, read>       cp_x  : array<f32>;
@group(0) @binding(2) var<storage, read>       cp_lg : array<f32>;
@group(0) @binding(3) var<storage, read_write> cp_dst: array<f32>;
@compute @workgroup_size(256)
fn bdp_cap_rows(@builtin(global_invocation_id) g: vec3<u32>) {
    let w = cp_p.hidden + cp_p.ne;
    let i = g.x;
    if (i >= cp_p.k * w) { return; }
    let t = i / w;
    let c = i % w;
    if (c < cp_p.hidden) {
        cp_dst[i] = cp_x[t * cp_p.hidden + c];
    } else {
        cp_dst[i] = cp_lg[t * cp_p.ne + c - cp_p.hidden];
    }
}
"#;

struct Capture {
    lo: usize,
    hi: usize,
    width: usize,
    rows: usize,
    bufs: Vec<wgpu::Buffer>,
    written: usize,
}

static CAPTURE: std::sync::Mutex<Option<Capture>> = std::sync::Mutex::new(None);

/// Capture layers `lo..hi` of the next batched-graph calls (up to `rows`
/// tokens a call). False without a device or its kernels.
pub fn moe_capture_begin(lo: usize, hi: usize, rows: usize, hidden: usize, ne: usize) -> bool {
    let Some(c) = ctx() else { return false };
    if pipes(c).is_none() || lo >= hi {
        return false;
    }
    let width = hidden + ne;
    let bufs = (lo..hi)
        .map(|_| {
            c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("clover-capture"),
                size: (rows * width * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    *CAPTURE.lock().unwrap() = Some(Capture { lo, hi, width, rows, bufs, written: 0 });
    true
}

pub fn moe_capture_end() {
    *CAPTURE.lock().unwrap() = None;
}

/// Inside the graph's MoE arm, after the router: copy layer `li`'s rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_capture_rows(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    li: usize,
    k: usize,
    x: &wgpu::Buffer,
    logits: &wgpu::Buffer,
    hidden: usize,
    ne: usize,
) {
    let mut g = CAPTURE.lock().unwrap();
    let Some(cap) = g.as_mut() else { return };
    let Some(p) = pipes(c) else { return };
    if li < cap.lo || li >= cap.hi || cap.width != hidden + ne || k > cap.rows {
        return;
    }
    let u = uniform(c, &[k as u32, hidden as u32, ne as u32, 0]);
    let bg = bind(c, &p.cap, &[(0, &u), (1, x), (2, logits), (3, &cap.bufs[li - cap.lo])]);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.cap);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(((k * cap.width) as u32).div_ceil(256), 1, 1);
    cap.written += 1;
}

/// Whether layer `li` is being captured.
pub(crate) fn moe_capturing(li: usize) -> bool {
    CAPTURE.lock().unwrap().as_ref().is_some_and(|c| li >= c.lo && li < c.hi)
}

/// The rows the last call captured (`k` tokens): per layer of the range
/// `k × (hidden + ne)` floats, and how many layers wrote since the last read.
pub fn moe_capture_read(k: usize) -> Option<(Vec<Vec<f32>>, usize)> {
    let c = ctx()?;
    let mut g = CAPTURE.lock().unwrap();
    let cap = g.as_mut()?;
    let n = k * cap.width;
    let size = (n * 4) as u64;
    let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("clover-capture-rb"),
        size: size * cap.bufs.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for (i, b) in cap.bufs.iter().enumerate() {
        enc.copy_buffer_to_buffer(b, 0, &stage, i as u64 * size, size);
    }
    c.queue.submit(Some(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    stage.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().ok()?.ok()?;
    let raw = stage.slice(..).get_mapped_range().ok()?;
    let vals: &[f32] = bytemuck::cast_slice(&raw);
    let out = vals.chunks_exact(n).map(|v| v.to_vec()).collect();
    drop(raw);
    stage.unmap();
    let written = std::mem::take(&mut cap.written);
    Some((out, written))
}

pub(crate) struct Pipes {
    conv: wgpu::ComputePipeline,
    mix: wgpu::ComputePipeline,
    coreset_rows: wgpu::ComputePipeline,
    coreset_pick: wgpu::ComputePipeline,
    moe_tiles: wgpu::ComputePipeline,
    pick: wgpu::ComputePipeline,
    cap: wgpu::ComputePipeline,
    /// `moe_r4`'s grouped kernels reading their (expert, tile) from the
    /// tile list instead of a full buckets × tiles grid.
    moeg_gu: wgpu::ComputePipeline,
    moeg_dn: wgpu::ComputePipeline,
}

/// `moe_r4::moeg_wgsl` with the workgroup's expert and tile read from the
/// tile list (`bdp_moe_tiles`): the bucket axis of the grid becomes the
/// list index, so only non-empty tiles launch.
fn moeg_tiles_wgsl() -> String {
    let src = super::moe_r4::moeg_wgsl();
    let nt = super::moe_r4::MOEG_TOKENS;
    let gu_old = format!("    let ex = wid.y;\n    let beg = gg_off[ex] + wid.z * {nt}u;");
    let dn_old = format!("    let ex = wid.y;\n    let beg = gd_off[ex] + wid.z * {nt}u;");
    assert!(src.contains(&gu_old) && src.contains(&dn_old), "moeg template changed");
    src.replace(
        &gu_old,
        &format!("    let pk = gg_tl[wid.y];\n    let ex = pk >> 16u;\n    let beg = gg_off[ex] + (pk & 0xFFFFu) * {nt}u;"),
    )
    .replace(
        &dn_old,
        &format!("    let pk = gd_tl[wid.y];\n    let ex = pk >> 16u;\n    let beg = gd_off[ex] + (pk & 0xFFFFu) * {nt}u;"),
    )
    .replace(
        "@group(0) @binding(8) var<uniform>             gg_p   : GgP;",
        "@group(0) @binding(8) var<uniform>             gg_p   : GgP;\n@group(0) @binding(9) var<storage, read>       gg_tl  : array<u32>;",
    )
    .replace(
        "@group(0) @binding(6) var<uniform>             gd_p   : GdP;",
        "@group(0) @binding(6) var<uniform>             gd_p   : GdP;\n@group(0) @binding(7) var<storage, read>       gd_tl  : array<u32>;",
    )
}

fn build(c: &Ctx) -> Option<Pipes> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-clover"),
        source: wgpu::ShaderSource::Wgsl(CLOVER_WGSL.into()),
    });
    let pipe = |entry: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let moe_module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-clover-moe"),
        source: wgpu::ShaderSource::Wgsl(MOE_WGSL.into()),
    });
    let pick_module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-clover-pick"),
        source: wgpu::ShaderSource::Wgsl(PICK_WGSL.into()),
    });
    let pick = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bdp_pick"),
        layout: None,
        module: &pick_module,
        entry_point: Some("bdp_pick"),
        compilation_options: Default::default(),
        cache: c.pipeline_cache.as_ref(),
    });
    let cap_module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-clover-capture"),
        source: wgpu::ShaderSource::Wgsl(CAP_WGSL.into()),
    });
    let cap = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bdp_cap_rows"),
        layout: None,
        module: &cap_module,
        entry_point: Some("bdp_cap_rows"),
        compilation_options: Default::default(),
        cache: c.pipeline_cache.as_ref(),
    });
    let grouped = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-clover-moeg"),
        source: wgpu::ShaderSource::Wgsl(moeg_tiles_wgsl().into()),
    });
    let grouped_pipe = |entry: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &grouped,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let moe_pipe = |entry: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &moe_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let p = Pipes {
        moe_tiles: moe_pipe("bdp_moe_tiles"),
        pick,
        cap,
        moeg_gu: grouped_pipe("moeg_gu"),
        moeg_dn: grouped_pipe("moeg_dn"),
        conv: pipe("bdp_conv"),
        mix: pipe("bdp_mix"),
        coreset_rows: pipe("bdp_coreset_rows"),
        coreset_pick: pipe("bdp_coreset_pick"),
    };
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-clover module rejected ({e}): block previews run on the host");
        return None;
    }
    Some(p)
}

/// The module's pipelines, or None when the device rejected them.
pub(crate) fn pipes(c: &Ctx) -> Option<&Pipes> {
    c.clover_pipes.get_or_init(|| build(c)).as_ref()
}

fn uniform(c: &Ctx, words: &[u32]) -> wgpu::Buffer {
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("clover-u"),
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn bind(c: &Ctx, p: &wgpu::ComputePipeline, pairs: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    bind_pairs(c, &p.get_bind_group_layout(0), pairs)
}

/// The geometry of one GDN layer's preview.
pub(crate) struct GdnDims {
    pub nv: usize,
    pub nk: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
    pub cdim: usize,
    pub kb: usize,
    pub eps: f32,
}

/// One GDN layer's preview: `qkv`/`z` [kb × …] and `a` [kb × nv] from the
/// projections, `ring`/`s` the committed device state (read only),
/// `cq` scratch [kb × cdim]; the gated output lands in `out` [kb × nv·dv].
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gdn_preview(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes,
    d: &GdnDims,
    qkv: &wgpu::Buffer,
    z: &wgpu::Buffer,
    a: &wgpu::Buffer,
    taps: &wgpu::Buffer,
    right: &wgpu::Buffer,
    a_log: &wgpu::Buffer,
    dt_bias: &wgpu::Buffer,
    norm: &wgpu::Buffer,
    logit_scale: &wgpu::Buffer,
    history_bias: &wgpu::Buffer,
    history_c: &wgpu::Buffer,
    ring: &wgpu::Buffer,
    s: &wgpu::Buffer,
    cq: &wgpu::Buffer,
    out: &wgpu::Buffer,
) {
    let cu = uniform(c, &[d.cdim as u32, d.kk as u32, d.kb as u32, 0]);
    let len_log = (d.kb as f32 / 32.0).ln();
    let mu = uniform(
        c,
        &[
            d.nv as u32,
            d.dk as u32,
            d.dv as u32,
            (d.nk * d.dk) as u32,
            (d.nv / d.nk) as u32,
            d.cdim as u32,
            d.eps.to_bits(),
            d.kb as u32,
            len_log.to_bits(),
            0,
            0,
            0,
        ],
    );
    let bg_conv = bind(c, &p.conv, &[(0, qkv), (1, taps), (2, right), (3, ring), (4, cq), (5, &cu)]);
    let bg_mix = bind(
        c,
        &p.mix,
        &[
            (0, cq),
            (1, z),
            (2, a),
            (3, a_log),
            (4, dt_bias),
            (5, norm),
            (6, s),
            (7, out),
            (8, &mu),
            (9, logit_scale),
            (10, history_bias),
            (11, history_c),
        ],
    );
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.conv);
    pass.set_bind_group(0, &bg_conv, &[]);
    pass.dispatch_workgroups((d.cdim as u32).div_ceil(256), d.kb as u32, 1);
    pass.set_pipeline(&p.mix);
    pass.set_bind_group(0, &bg_mix, &[]);
    pass.dispatch_workgroups(d.nv as u32, (d.kb as u32).div_ceil(4), 1);
}

/// The expert-grouped MoE of a block after `moeg_group` filled `off`/`ent`:
/// the tile list, then gate/up and down over the non-empty tiles only
/// (indirect grids). Bindings as `moeg_gu` / `moeg_dn` take them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_moe(
    c: &Ctx,
    pass: &mut wgpu::ComputePass<'_>,
    p: &Pipes,
    gate_all: &wgpu::Buffer,
    up_all: &wgpu::Buffer,
    down_all: &wgpu::Buffer,
    x: &wgpu::Buffer,
    off: &wgpu::Buffer,
    ent: &wgpu::Buffer,
    act: &wgpu::Buffer,
    y: &wgpu::Buffer,
    gu_u: &wgpu::Buffer,
    dn_u: &wgpu::Buffer,
    buckets: usize,
    n_ent: usize,
    inter: usize,
    hidden: usize,
) {
    let list = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("clover-moe-tiles"),
        size: (n_ent * 4).max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let args = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("clover-moe-args"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });
    let (gu_p, dn_p, gu_x, dn_x) = (&p.moeg_gu, &p.moeg_dn, inter.div_ceil(64), hidden.div_ceil(64));
    let tu = uniform(c, &[buckets as u32, gu_x as u32, dn_x as u32, 0]);
    let bt = bind(c, &p.moe_tiles, &[(0, off), (1, &list), (2, &args), (3, &tu)]);
    let bgu = bind(
        c,
        gu_p,
        &[
            (0, gate_all),
            (1, up_all),
            (2, gate_all),
            (3, up_all),
            (4, x),
            (5, off),
            (6, ent),
            (7, act),
            (8, gu_u),
            (9, &list),
        ],
    );
    let bdn = bind(
        c,
        dn_p,
        &[(0, down_all), (1, down_all), (2, act), (3, off), (4, ent), (5, y), (6, dn_u), (7, &list)],
    );
    pass.set_pipeline(&p.moe_tiles);
    pass.set_bind_group(0, &bt, &[]);
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(gu_p);
    pass.set_bind_group(0, &bgu, &[]);
    pass.dispatch_workgroups_indirect(&args, 0);
    pass.set_pipeline(dn_p);
    pass.set_bind_group(0, &bdn, &[]);
    pass.dispatch_workgroups_indirect(&args, 12);
}

/// Penalize and pick every row of `logits` `[kb × stride]` in place; the
/// (id, probability) pairs land in `out` `[kb × 2]` u32.
pub(crate) fn encode_pick(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes,
    pk: &crate::gpu::PreviewPick<'_>,
    logits: &wgpu::Buffer,
    stride: usize,
    kb: usize,
    out: &wgpu::Buffer,
) {
    let init = |label: &'static str, words: &[u32]| {
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(if words.is_empty() { &[u32::MAX] } else { words }),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let counts = init("clover-pick-counts", pk.counts);
    let stop = init("clover-pick-stop", pk.stop);
    let forbid = init("clover-pick-forbid", pk.forbid);
    let f = |x: f32| x.to_bits();
    let u = uniform(
        c,
        &[
            pk.vocab.min(stride) as u32,
            stride as u32,
            pk.stop.len() as u32,
            pk.forbid.len() as u32,
            pk.think_end.unwrap_or(u32::MAX),
            u32::from(pk.close_open),
            pk.len0 as u32,
            0,
            f(pk.frequency),
            f(pk.presence),
            f(pk.repetition),
            f(pk.length_ramp.0),
            f(pk.length_ramp.1),
            f(pk.length_ramp.2.max(1.0)),
            f(pk.think_ramp.0),
            f(pk.think_ramp.1),
            f(pk.think_ramp.2.max(1.0)),
            0,
            0,
            0,
        ],
    );
    let bg = bind(c, &p.pick, &[(0, logits), (1, &counts), (2, &stop), (3, &forbid), (4, out), (5, &u)]);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.pick);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(kb as u32, 1, 1);
}

/// Mask the router logits `[kb × ne]` in place to the block coreset.
pub(crate) fn encode_coreset(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &Pipes,
    logits: &wgpu::Buffer,
    ne: usize,
    kb: usize,
    top_k: usize,
    top_p: f32,
) {
    let u = uniform(c, &[ne as u32, kb as u32, top_k as u32, top_p.to_bits()]);
    let part = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("clover-coreset"),
        size: (kb * ne * 4).max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bg_rows = bind(c, &p.coreset_rows, &[(0, logits), (1, &u), (2, &part)]);
    let bg_pick = bind(c, &p.coreset_pick, &[(0, logits), (1, &u), (2, &part)]);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.coreset_rows);
    pass.set_bind_group(0, &bg_rows, &[]);
    pass.dispatch_workgroups(kb as u32, 1, 1);
    pass.set_pipeline(&p.coreset_pick);
    pass.set_bind_group(0, &bg_pick, &[]);
    pass.dispatch_workgroups(1, 1, 1);
}

/// Shapes the kernels serve: four rows of a ≤256-wide key head and a
/// ≤128-wide value head in one workgroup's arrays, the block plus its
/// history slot within `bm_w` (and ≤ 256 experts for the coreset).
pub(crate) fn fits(dk: usize, dv: usize, kb: usize) -> bool {
    dk <= 256 && dv <= 128 && kb > 0 && kb <= 255
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clover_preview_wgsl_validates() {
        let moeg = moeg_tiles_wgsl();
        for (name, src) in [
            ("preview", CLOVER_WGSL),
            ("moe", MOE_WGSL),
            ("moeg", moeg.as_str()),
            ("pick", PICK_WGSL),
            ("capture", CAP_WGSL),
        ] {
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
