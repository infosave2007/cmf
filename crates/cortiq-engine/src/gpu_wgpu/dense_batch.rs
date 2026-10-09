//! Batch-graph kernels for the dense GDN hybrids (Qwen3.5 / Qwen3.8).
//!
//! The batched prompt graph ran a gated full-attention layer as a loop of
//! six dispatches per position (rope, K/V append, split attend, merge,
//! output gate, blit). On an RTX 3090 with Qwen3.8-27B q4tp that loop cost
//! 317 of a 128-position chunk's 1,155 ms at a 1,000-token prompt, and it
//! grows with the context. These are the same kernels with a TOKEN AXIS:
//! one dispatch each for the whole run. Every token's arithmetic is the
//! per-position kernel's, expression for expression, so the hidden rows
//! are the loop's to the bit; only the dispatch count changes.
//!
//! `CMF_ATTN_BT=0` keeps the per-position loop.

use super::*;

pub(crate) const ATTN_BT_WGSL: &str = r#"
// ── RoPE + qk-norm + gate split with a token axis (wid.y = token t of the
// run): `attn_rope_qkn` with tok = rq_p.tok + t and pos = rq_p.pos + t; Q
// and gate rows land at t·nh·hd of the batched outputs.
struct RqP {
    nh: u32, nkv: u32, hd: u32, rd: u32,
    pos: u32, flags: u32, eps: f32, tok: u32,
    rope_scale: f32, _p0: u32, _p1: u32, _p2: u32,
};
@group(0) @binding(0) var<storage, read>       rq_qraw : array<f32>;
@group(0) @binding(1) var<storage, read_write> rq_k    : array<f32>;
@group(0) @binding(2) var<storage, read_write> rq_qout : array<f32>;
@group(0) @binding(3) var<storage, read_write> rq_gout : array<f32>;
@group(0) @binding(4) var<storage, read>       rq_qnw  : array<f32>;
@group(0) @binding(5) var<storage, read>       rq_knw  : array<f32>;
@group(0) @binding(6) var<storage, read>       rq_invf : array<f32>;
@group(0) @binding(7) var<uniform>             rq_p    : RqP;
var<workgroup> rq_red: array<f32, 32>;
var<workgroup> rq_head: array<f32, 256>;
@compute @workgroup_size(32)
fn attn_rope_qkn_bt(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let lane = lid.x;
    let nh = rq_p.nh;
    let hd = rq_p.hd;
    if (head >= nh + rq_p.nkv) { return; }
    let tok = rq_p.tok + wid.y;
    let pos = rq_p.pos + wid.y;
    let isq = head < nh;
    let gate = (rq_p.flags & 1u) != 0u;
    let src_base = select((head - nh) * hd, head * select(1u, 2u, gate) * hd, isq);
    let qoff = tok * nh * select(1u, 2u, gate) * hd;
    let koff = tok * rq_p.nkv * hd;
    let ooff = tok * nh * hd;
    let nt = (hd + 31u) / 32u;
    var xv: array<f32, 8>;
    var ss = 0.0;
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        var val = 0.0;
        if (d < hd) { val = select(rq_k[koff + src_base + d], rq_qraw[qoff + src_base + d], isq); }
        xv[t] = val;
        ss = ss + val * val;
    }
    rq_red[lane] = ss;
    workgroupBarrier();
    var stride = 16u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) { rq_red[lane] = rq_red[lane] + rq_red[lane + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let normed = select((rq_p.flags & 4u) != 0u, (rq_p.flags & 2u) != 0u, isq);
    let late = (rq_p.flags & 32u) != 0u;
    let hlf = rq_p.rd / 2u;
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) { rq_head[d] = xv[t]; }
    }
    workgroupBarrier();
    if (late) {
        var ri0 = lane;
        loop {
            if (ri0 >= hlf) { break; }
            let angle0 = f32(pos) * rq_invf[ri0];
            let cc0 = cos(angle0);
            let sf0 = sin(angle0);
            let y0 = rq_head[ri0];
            let y1 = rq_head[ri0 + hlf];
            rq_head[ri0] = (y0 * cc0 - y1 * sf0) * rq_p.rope_scale;
            rq_head[ri0 + hlf] = (y0 * sf0 + y1 * cc0) * rq_p.rope_scale;
            ri0 = ri0 + 32u;
        }
    }
    workgroupBarrier();
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) { xv[t] = rq_head[d]; }
    }
    workgroupBarrier();
    if (normed) {
        let inv = 1.0 / sqrt(rq_red[0] / f32(hd) + rq_p.eps);
        let gemma = (rq_p.flags & 8u) != 0u;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) {
                var wd = select(rq_knw[d], rq_qnw[d], isq);
                if (gemma) { wd = 1.0 + wd; }
                xv[t] = xv[t] * inv * wd;
            }
        }
    }
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) { rq_head[d] = xv[t]; }
    }
    workgroupBarrier();
    if (!late) {
        var ri = lane;
        loop {
            if (ri >= hlf) { break; }
            let angle = f32(pos) * rq_invf[ri];
            let cc = cos(angle);
            let sfac = sin(angle);
            let x0 = rq_head[ri];
            let x1 = rq_head[ri + hlf];
            rq_head[ri] = (x0 * cc - x1 * sfac) * rq_p.rope_scale;
            rq_head[ri + hlf] = (x0 * sfac + x1 * cc) * rq_p.rope_scale;
            ri = ri + 32u;
        }
    }
    workgroupBarrier();
    let dst_base = select((head - nh) * hd, head * hd, isq);
    for (var t = 0u; t < nt; t = t + 1u) {
        let d = t * 32u + lane;
        if (d < hd) {
            if (isq) { rq_qout[ooff + dst_base + d] = rq_head[d]; } else { rq_k[koff + dst_base + d] = rq_head[d]; }
        }
    }
    if (isq && gate) {
        let gbase = head * 2u * hd + hd;
        for (var t = 0u; t < nt; t = t + 1u) {
            let d = t * 32u + lane;
            if (d < hd) { rq_gout[ooff + head * hd + d] = rq_qraw[qoff + gbase + d]; }
        }
    }
}

// ── K/V rows of the run into the mirror (gid.y = token): `kv_append` with
// stored = pos + t, the batch row tok + t.
struct KvP { nkv: u32, hd: u32, cap: u32, pos: u32, tok: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(8)  var<storage, read>       kv_k  : array<f32>;
@group(0) @binding(9)  var<storage, read>       kv_v  : array<f32>;
@group(0) @binding(10) var<storage, read_write> kv_kb : array<f32>;
@group(0) @binding(11) var<storage, read_write> kv_vb : array<f32>;
@group(0) @binding(12) var<uniform>             kv_p  : KvP;
@compute @workgroup_size(256)
fn kv_append_bt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= kv_p.nkv * kv_p.hd) { return; }
    let stored = kv_p.pos + gid.y;
    let toff = (kv_p.tok + gid.y) * kv_p.nkv * kv_p.hd;
    let h = i / kv_p.hd;
    let d = i % kv_p.hd;
    let dst = (h * kv_p.cap + stored) * kv_p.hd + d;
    kv_kb[dst] = kv_k[toff + i];
    kv_vb[dst] = kv_v[toff + i];
}

// ── GQA-shared split attend with a token axis: `gqa_attend_gpart` for token
// t = t0 + wid.z of the batch (context n = pos0 + t + 1), its queries at row
// t of the batched Q, its partials at slot ((wid.z·nh + h)·nc + ch).
struct AtP {
    nh: u32, hpk: u32, hd: u32, cap: u32,
    pos0: u32, ck: u32, nc: u32, scale: f32,
    k: u32, t0: u32, _a: u32, _b: u32,
};
@group(0) @binding(13) var<storage, read>       ap_q  : array<vec4<f32>>;
@group(0) @binding(14) var<storage, read>       ap_k  : array<vec4<f32>>;
@group(0) @binding(15) var<storage, read>       ap_v  : array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> ap_acc: array<f32>;
@group(0) @binding(17) var<storage, read_write> ap_ml : array<vec2<f32>>;
@group(0) @binding(18) var<uniform>             ap_p  : AtP;
@group(0) @binding(19) var<storage, read_write> ap_o  : array<f32>;
const AG_CK: u32 = 256u;
var<workgroup> ag_q: array<vec4<f32>, 512>;
var<workgroup> ag_sc: array<f32, 2048>;
var<workgroup> ag_red: array<f32, 2048>;
@compute @workgroup_size(256)
fn gqa_attend_gpart_bt(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x;
    let ch = wid.y;
    let tl = wid.z;
    let hpk = ap_p.hpk;
    let nkv = ap_p.nh / hpk;
    if (g >= nkv || tl >= ap_p.k) { return; }
    let t = ap_p.t0 + tl;
    let n = ap_p.pos0 + t + 1u;
    let hd = ap_p.hd;
    let hd4 = hd / 4u;
    let p0 = ch * AG_CK;
    if (p0 >= n) { return; }
    let pend = min(n, p0 + AG_CK);
    let cn = select(0u, pend - p0, pend > p0);
    let kbase = g * ap_p.cap * hd4;
    let scale = ap_p.scale;
    let qrow = t * ap_p.nh * hd4;
    for (var i = lid; i < 512u; i = i + 256u) {
        let hq = i / hd4;
        let d4 = i % hd4;
        var qv = vec4<f32>(0.0);
        if (hq < hpk && i < hpk * hd4) { qv = ap_q[qrow + (g * hpk + hq) * hd4 + d4]; }
        ag_q[i] = qv;
    }
    workgroupBarrier();
    var d0 = vec4<f32>(0.0); var d1 = vec4<f32>(0.0);
    var d2 = vec4<f32>(0.0); var d3 = vec4<f32>(0.0);
    var d4 = vec4<f32>(0.0); var d5 = vec4<f32>(0.0);
    var d6 = vec4<f32>(0.0); var d7 = vec4<f32>(0.0);
    if (lid < cn) {
        let krow = kbase + (p0 + lid) * hd4;
        for (var d = 0u; d < hd4; d = d + 1u) {
            let kv = ap_k[krow + d];
            d0 = d0 + ag_q[d] * kv;
            d1 = d1 + ag_q[hd4 + d] * kv;
            d2 = d2 + ag_q[2u * hd4 + d] * kv;
            d3 = d3 + ag_q[3u * hd4 + d] * kv;
            d4 = d4 + ag_q[4u * hd4 + d] * kv;
            d5 = d5 + ag_q[5u * hd4 + d] * kv;
            d6 = d6 + ag_q[6u * hd4 + d] * kv;
            d7 = d7 + ag_q[7u * hd4 + d] * kv;
        }
    }
    let live = lid < cn;
    var sc: array<f32, 8>;
    sc[0] = select(-1e30, (d0.x + d0.y + d0.z + d0.w) * scale, live);
    sc[1] = select(-1e30, (d1.x + d1.y + d1.z + d1.w) * scale, live);
    sc[2] = select(-1e30, (d2.x + d2.y + d2.z + d2.w) * scale, live);
    sc[3] = select(-1e30, (d3.x + d3.y + d3.z + d3.w) * scale, live);
    sc[4] = select(-1e30, (d4.x + d4.y + d4.z + d4.w) * scale, live);
    sc[5] = select(-1e30, (d5.x + d5.y + d5.z + d5.w) * scale, live);
    sc[6] = select(-1e30, (d6.x + d6.y + d6.z + d6.w) * scale, live);
    sc[7] = select(-1e30, (d7.x + d7.y + d7.z + d7.w) * scale, live);
    for (var h = 0u; h < 8u; h = h + 1u) {
        ag_sc[h * 256u + lid] = sc[h];
        ag_red[lid * 8u + h] = sc[h];
    }
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) {
            for (var h = 0u; h < 8u; h = h + 1u) {
                ag_red[lid * 8u + h] = max(ag_red[lid * 8u + h], ag_red[(lid + st) * 8u + h]);
            }
        }
        workgroupBarrier();
        st = st >> 1u;
    }
    var cm: array<f32, 8>;
    for (var h = 0u; h < 8u; h = h + 1u) { cm[h] = ag_red[h]; }
    workgroupBarrier();
    for (var h = 0u; h < 8u; h = h + 1u) {
        let w = select(0.0, exp(sc[h] - cm[h]), live);
        ag_sc[h * 256u + lid] = w;
        ag_red[lid * 8u + h] = w;
    }
    workgroupBarrier();
    st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) {
            for (var h = 0u; h < 8u; h = h + 1u) {
                ag_red[lid * 8u + h] = ag_red[lid * 8u + h] + ag_red[(lid + st) * 8u + h];
            }
        }
        workgroupBarrier();
        st = st >> 1u;
    }
    let pb = tl * ap_p.nh;
    if (lid < hd) {
        var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
        var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
        let dw = lid >> 2u;
        let dc = lid & 3u;
        for (var p = 0u; p < cn; p = p + 1u) {
            let v = ap_v[kbase + (p0 + p) * hd4 + dw][dc];
            a0 = a0 + ag_sc[p] * v;
            a1 = a1 + ag_sc[256u + p] * v;
            a2 = a2 + ag_sc[512u + p] * v;
            a3 = a3 + ag_sc[768u + p] * v;
            a4 = a4 + ag_sc[1024u + p] * v;
            a5 = a5 + ag_sc[1280u + p] * v;
            a6 = a6 + ag_sc[1536u + p] * v;
            a7 = a7 + ag_sc[1792u + p] * v;
        }
        var acc: array<f32, 8>;
        acc[0] = a0; acc[1] = a1; acc[2] = a2; acc[3] = a3;
        acc[4] = a4; acc[5] = a5; acc[6] = a6; acc[7] = a7;
        for (var h = 0u; h < hpk; h = h + 1u) {
            let idx = (pb + g * hpk + h) * ap_p.nc + ch;
            ap_acc[idx * hd + lid] = acc[h];
        }
    }
    if (lid == 0u) {
        for (var h = 0u; h < hpk; h = h + 1u) {
            let idx = (pb + g * hpk + h) * ap_p.nc + ch;
            ap_ml[idx] = vec2<f32>(cm[h], ag_red[h]);
        }
    }
}

// ── `gqa_attend_dec` with a token axis (wid.y = slot tl, token t = t0 +
// tl of context pos0 + t + 1): the one-workgroup-a-head decode attend the
// position loop runs up to ATTEND_SPLIT_MIN positions, same order.
var<workgroup> ad_sc: array<f32, 256>;
var<workgroup> ad_red: array<f32, 256>;
@compute @workgroup_size(256)
fn gqa_attend_dec_bt(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let tl = wid.y;
    if (h >= ap_p.nh || tl >= ap_p.k) { return; }
    let t = ap_p.t0 + tl;
    let hd = ap_p.hd;
    let hd4 = hd / 4u;
    let n = ap_p.pos0 + t + 1u;
    let kbase = (h / ap_p.hpk) * ap_p.cap * hd4;
    let qbase = (t * ap_p.nh + h) * hd4;
    let scale = ap_p.scale;
    var m = -1.0e30;
    var l = 0.0;
    var acc = 0.0;
    var c0 = 0u;
    loop {
        if (c0 >= n) { break; }
        let cn = min(256u, n - c0);
        var sc = -1.0e30;
        if (lid < cn) {
            let krow = kbase + (c0 + lid) * hd4;
            var dot4 = vec4<f32>(0.0);
            for (var d = 0u; d < hd4; d = d + 1u) {
                dot4 = dot4 + ap_q[qbase + d] * ap_k[krow + d];
            }
            sc = (dot4.x + dot4.y + dot4.z + dot4.w) * scale;
        }
        ad_sc[lid] = sc;
        ad_red[lid] = sc;
        workgroupBarrier();
        var st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { ad_red[lid] = max(ad_red[lid], ad_red[lid + st]); }
            workgroupBarrier();
            st = st >> 1u;
        }
        let cm = ad_red[0];
        workgroupBarrier();
        let mp = max(m, cm);
        let f = exp(m - mp);
        let w = select(0.0, exp(ad_sc[lid] - mp), lid < cn);
        ad_sc[lid] = w;
        ad_red[lid] = w;
        workgroupBarrier();
        st = 128u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { ad_red[lid] = ad_red[lid] + ad_red[lid + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        l = l * f + ad_red[0];
        workgroupBarrier();
        if (lid < hd) {
            acc = acc * f;
            let dw = lid >> 2u;
            let dc = lid & 3u;
            for (var p = 0u; p < cn; p = p + 1u) {
                acc = acc + ad_sc[p] * ap_v[kbase + (c0 + p) * hd4 + dw][dc];
            }
        }
        m = mp;
        c0 = c0 + 256u;
        workgroupBarrier();
    }
    if (lid < hd) {
        ap_o[(t * ap_p.nh + h) * hd + lid] = acc / l;
    }
}

// ── `gqa_attend_merge` with a token axis (wid.y = slot tl): token t's
// chunks of context pos0 + t + 1, output row t of the batched attention.
@compute @workgroup_size(32)
fn gqa_attend_merge_bt(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let tl = wid.y;
    let lane = lid.x;
    if (h >= ap_p.nh || tl >= ap_p.k) { return; }
    let t = ap_p.t0 + tl;
    let n = ap_p.pos0 + t + 1u;
    let hd = ap_p.hd;
    let nc = (n + ap_p.ck - 1u) / ap_p.ck;
    let hb = (tl * ap_p.nh + h) * ap_p.nc;
    var mg = -1e30;
    for (var ci = 0u; ci < nc; ci = ci + 1u) { mg = max(mg, ap_ml[hb + ci].x); }
    var lg = 0.0;
    for (var ci = 0u; ci < nc; ci = ci + 1u) {
        let ml = ap_ml[hb + ci];
        lg = lg + ml.y * exp(ml.x - mg);
    }
    let invl = select(0.0, 1.0 / lg, lg > 0.0);
    let ob = (t * ap_p.nh + h) * hd;
    for (var d = lane; d < hd; d = d + 32u) {
        var a = 0.0;
        for (var ci = 0u; ci < nc; ci = ci + 1u) {
            let idx = hb + ci;
            a = a + ap_acc[idx * hd + d] * exp(ap_ml[idx].x - mg);
        }
        ap_o[ob + d] = a * invl;
    }
}
"#;

pub(crate) struct AttnBt {
    pub(crate) rope: wgpu::ComputePipeline,
    pub(crate) rope_l: wgpu::BindGroupLayout,
    pub(crate) kv: wgpu::ComputePipeline,
    pub(crate) kv_l: wgpu::BindGroupLayout,
    pub(crate) part: wgpu::ComputePipeline,
    pub(crate) part_l: wgpu::BindGroupLayout,
    pub(crate) merge: wgpu::ComputePipeline,
    pub(crate) merge_l: wgpu::BindGroupLayout,
    pub(crate) dec: wgpu::ComputePipeline,
    pub(crate) dec_l: wgpu::BindGroupLayout,
}

pub(crate) struct Pipes {
    pub(crate) attn: Option<AttnBt>,
    /// `q4tp_mm_r2` (`q4tp_mm_r` under `CMF_Q4MM_R=1`); None on rejection,
    /// a small workgroup-storage limit or `CMF_Q4MM_R=0`.
    pub(crate) mm_r: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
}

fn build_attn(c: &Ctx) -> Option<AttnBt> {
    if std::env::var("CMF_ATTN_BT").as_deref() == Ok("0") {
        return None;
    }
    if c.device.limits().max_compute_workgroup_storage_size < 2 * 2048 * 4 + 512 * 16 {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-attn-bt"),
        source: wgpu::ShaderSource::Wgsl(ATTN_BT_WGSL.into()),
    });
    let pipe = |ep: &str| {
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let rope = pipe("attn_rope_qkn_bt");
    let kv = pipe("kv_append_bt");
    let part = pipe("gqa_attend_gpart_bt");
    let merge = pipe("gqa_attend_merge_bt");
    let dec = pipe("gqa_attend_dec_bt");
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-attn-bt module rejected ({e}): gated attention keeps the per-position loop");
        return None;
    }
    Some(AttnBt {
        rope_l: rope.get_bind_group_layout(0),
        rope,
        kv_l: kv.get_bind_group_layout(0),
        kv,
        part_l: part.get_bind_group_layout(0),
        part,
        merge_l: merge.get_bind_group_layout(0),
        merge,
        dec_l: dec.get_bind_group_layout(0),
        dec,
    })
}

fn build_mm_r(c: &Ctx) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    if std::env::var("CMF_Q4MM_R").as_deref() == Ok("0") {
        return None;
    }
    if c.device.limits().max_compute_workgroup_storage_size < 2 * 1024 * 16 {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-q4mm-r"),
        source: wgpu::ShaderSource::Wgsl(MM_R_WGSL.into()),
    });
    // The software-pipelined twin by default (same sums): 1000-token
    // prefill on Qwen3.8-27B / RTX 3090 176.9/175.9 -> 183.7/186.6 tok/s.
    // `CMF_Q4MM_R=1`: the single-buffered kernel.
    let entry = if std::env::var("CMF_Q4MM_R").as_deref() == Ok("1") {
        "q4tp_mm_r"
    } else {
        "q4tp_mm_r2"
    };
    let p = c
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: c.pipeline_cache.as_ref(),
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-q4mm-r module rejected ({e}): the prompt GEMM keeps q4tp_mul_mm");
        return None;
    }
    let l = p.get_bind_group_layout(0);
    Some((p, l))
}

fn build(c: &Ctx) -> Option<Pipes> {
    Some(Pipes {
        attn: build_attn(c),
        mm_r: build_mm_r(c),
    })
}

/// Encode `y[nb][rows] = x[nb][cols] · Wᵀ` for a q4tp weight through
/// `q4tp_mm_r`. False (nothing encoded) when the kernel is missing or the
/// width is not a multiple of its 64-column step.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q4tp_mm_r(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    weight: &wgpu::Buffer,
    xs: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
    nb: usize,
) -> bool {
    let Some((p, l)) = pipes(c).and_then(|p| p.mm_r.as_ref()) else {
        return false;
    };
    if cols % 64 != 0 || rows == 0 || nb == 0 || rows.div_ceil(64) > MAX_WG as usize {
        return false;
    }
    let u = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, nb as u32, 0]);
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("q4mm-r"),
        layout: l,
        entries: &[bind_buf(0, weight), bind_buf(1, xs), bind_buf(2, y), bind_buf(3, &u)],
    });
    let mut pass = begin_pass(enc);
    pass.set_pipeline(p);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups(rows.div_ceil(64) as u32, nb.div_ceil(64) as u32, 1);
    true
}

pub(crate) fn pipes(c: &Ctx) -> Option<&Pipes> {
    c.dense_batch_pipes.get_or_init(|| build(c)).as_ref()
}

/// The token-axis attention kernels, when the module came up.
pub(crate) fn attn(c: &Ctx) -> Option<&AttnBt> {
    pipes(c).and_then(|p| p.attn.as_ref())
}

/// Positions per chunk of `gqa_attend_gpart_bt` (the token kernel's AG_CK).
pub(crate) const BT_CK: usize = 256;

/// Encode the whole run's gated full attention for batch tokens
/// `t_from..k` (positions `pos0 + t`): rope + K/V append, the decode
/// attend for tokens before `t_dec` and the split attend + merge from it,
/// writing attention rows `t` of `attn_bb`. The caller applies the
/// output gate over the same rows. `qall`/`gall` are [k][nh][hd] scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_attn_bt(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    p: &AttnBt,
    rope_words: [u32; 12],
    qraw: &wgpu::Buffer,
    kb: &wgpu::Buffer,
    vb: &wgpu::Buffer,
    qall: &wgpu::Buffer,
    gall: &wgpu::Buffer,
    qnw: &wgpu::Buffer,
    knw: &wgpu::Buffer,
    invf: &wgpu::Buffer,
    kcache: &wgpu::Buffer,
    vcache: &wgpu::Buffer,
    pacc: &wgpu::Buffer,
    pml: &wgpu::Buffer,
    attn_bb: &wgpu::Buffer,
    nh: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    pos0: usize,
    t_from: usize,
    t_dec: usize,
    k: usize,
    sub: usize,
    nc: usize,
    scale: f32,
) {
    let run = k - t_from;
    if run == 0 {
        return;
    }
    let mut rw = rope_words;
    rw[4] = (pos0 + t_from) as u32;
    rw[7] = t_from as u32;
    let rope_u = c
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attn-bt-rope"),
            contents: bytemuck::cast_slice(&rw),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let kv_u = uniform_u32x8(
        c,
        [
            nkv as u32,
            hd as u32,
            cap as u32,
            (pos0 + t_from) as u32,
            t_from as u32,
            0,
            0,
            0,
        ],
    );
    let bind = |l: &wgpu::BindGroupLayout, e: &[(u32, &wgpu::Buffer)]| {
        let entries: Vec<_> = e.iter().map(|(i, b)| bind_buf(*i, b)).collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: l,
            entries: &entries,
        })
    };
    let bg_rope = bind(
        &p.rope_l,
        &[
            (0, qraw),
            (1, kb),
            (2, qall),
            (3, gall),
            (4, qnw),
            (5, knw),
            (6, invf),
            (7, &rope_u),
        ],
    );
    let bg_kv = bind(&p.kv_l, &[(8, kb), (9, vb), (10, kcache), (11, vcache), (12, &kv_u)]);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.rope);
    pass.set_bind_group(0, &bg_rope, &[]);
    pass.dispatch_workgroups((nh + nkv) as u32, run as u32, 1);
    pass.set_pipeline(&p.kv);
    pass.set_bind_group(0, &bg_kv, &[]);
    pass.dispatch_workgroups(((nkv * hd) as u32).div_ceil(256), run as u32, 1);
    if t_dec > t_from {
        // Tokens whose context is within ATTEND_SPLIT_MIN: the dec twin.
        let kk = t_dec - t_from;
        let at_u = c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("attn-bt-dec"),
                contents: bytemuck::cast_slice(&[
                    nh as u32,
                    (nh / nkv) as u32,
                    hd as u32,
                    cap as u32,
                    pos0 as u32,
                    BT_CK as u32,
                    nc as u32,
                    scale.to_bits(),
                    kk as u32,
                    t_from as u32,
                    0,
                    0,
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg_dec = bind(
            &p.dec_l,
            &[(13, qall), (14, kcache), (15, vcache), (18, &at_u), (19, attn_bb)],
        );
        pass.set_pipeline(&p.dec);
        pass.set_bind_group(0, &bg_dec, &[]);
        pass.dispatch_workgroups(nh as u32, kk as u32, 1);
    }
    let mut t0 = t_dec.max(t_from);
    while t0 < k {
        let kk = sub.min(k - t0);
        let n_last = pos0 + t0 + kk;
        let at_u = c
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("attn-bt-at"),
                contents: bytemuck::cast_slice(&[
                    nh as u32,
                    (nh / nkv) as u32,
                    hd as u32,
                    cap as u32,
                    pos0 as u32,
                    BT_CK as u32,
                    nc as u32,
                    scale.to_bits(),
                    kk as u32,
                    t0 as u32,
                    0,
                    0,
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg_part = bind(
            &p.part_l,
            &[
                (13, qall),
                (14, kcache),
                (15, vcache),
                (16, pacc),
                (17, pml),
                (18, &at_u),
            ],
        );
        let bg_merge = bind(&p.merge_l, &[(16, pacc), (17, pml), (18, &at_u), (19, attn_bb)]);
        pass.set_pipeline(&p.part);
        pass.set_bind_group(0, &bg_part, &[]);
        pass.dispatch_workgroups(nkv as u32, n_last.div_ceil(BT_CK) as u32, kk as u32);
        pass.set_pipeline(&p.merge);
        pass.set_bind_group(0, &bg_merge, &[]);
        pass.dispatch_workgroups(nh as u32, kk as u32, 1);
        t0 += kk;
    }
}

pub(crate) const MM_R_WGSL: &str = r#"
struct MmP { cols4: u32, rows: u32, nb: u32, pad: u32 };
@group(0) @binding(0) var<storage, read>       qmm  : array<u32>;
@group(0) @binding(1) var<storage, read>       xmm4 : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> ymm  : array<f32>;
@group(0) @binding(3) var<uniform>             pmm  : MmP;
// 64 k x 16 four-row groups each, XOR-swizzled by the k block so both the
// staging stores and the compute loads stay conflict-free.
var<workgroup> rx: array<vec4<f32>, 1024>;
var<workgroup> rw: array<vec4<f32>, 1024>;
fn rb(off: u32) -> u32 {
    return (qmm[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}
// Register-blocked q4tp GEMM for the batched prompt: a 64 (batch rows) x 64
// (output rows) tile per 128-lane workgroup, 64 columns (two q4tp groups) a
// step, four batch rows x eight output rows per lane. Every output is the
// sequential k-order sum `a = a + x*w` of `q4tp_mul_mm`, with the same
// weight values, so the two kernels agree to the bit; this one stages each
// weight word once per tile, takes one scale per row and group instead of
// one per four weights, and feeds 32 FMAs from three shared loads.
@compute @workgroup_size(128)
fn q4tp_mm_r(@builtin(workgroup_id) wid: vec3<u32>,
             @builtin(local_invocation_index) t: u32) {
    let cols4 = pmm.cols4;
    let cols = cols4 * 4u;
    let gpr = cols >> 5u;
    let rows = pmm.rows;
    let nb = pmm.nb;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    // Weight staging role: word kw of the step (group kw/4, word kw%4) of
    // rows 4ng..4ng+3. Lanes of a warp share rows, so a row's 32 bytes of a
    // step come in one sector.
    let kw = t & 7u;
    let ng = t >> 3u;
    let wr = n0 + ng * 4u;
    // Compute role: batch rows 4tm..4tm+3, output rows 8tn..8tn+7.
    let tm = t & 15u;
    let tn = t >> 4u;
    var a00 = 0.0;
    var a01 = 0.0;
    var a02 = 0.0;
    var a03 = 0.0;
    var a04 = 0.0;
    var a05 = 0.0;
    var a06 = 0.0;
    var a07 = 0.0;
    var a10 = 0.0;
    var a11 = 0.0;
    var a12 = 0.0;
    var a13 = 0.0;
    var a14 = 0.0;
    var a15 = 0.0;
    var a16 = 0.0;
    var a17 = 0.0;
    var a20 = 0.0;
    var a21 = 0.0;
    var a22 = 0.0;
    var a23 = 0.0;
    var a24 = 0.0;
    var a25 = 0.0;
    var a26 = 0.0;
    var a27 = 0.0;
    var a30 = 0.0;
    var a31 = 0.0;
    var a32 = 0.0;
    var a33 = 0.0;
    var a34 = 0.0;
    var a35 = 0.0;
    var a36 = 0.0;
    var a37 = 0.0;
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        for (var u = t; u < 256u; u = u + 128u) {
            let kq = u & 15u;
            let mg = u >> 4u;
            let m = m0 + mg * 4u;
            let ci = (k0 >> 2u) + kq;
            var v0 = vec4<f32>(0.0);
            var v1 = vec4<f32>(0.0);
            var v2 = vec4<f32>(0.0);
            var v3 = vec4<f32>(0.0);
            if (m < nb) { v0 = xmm4[m * cols4 + ci]; }
            if (m + 1u < nb) { v1 = xmm4[(m + 1u) * cols4 + ci]; }
            if (m + 2u < nb) { v2 = xmm4[(m + 2u) * cols4 + ci]; }
            if (m + 3u < nb) { v3 = xmm4[(m + 3u) * cols4 + ci]; }
            let kb = kq * 64u;
            let sx = mg ^ kq;
            rx[kb + sx] = vec4<f32>(v0.x, v1.x, v2.x, v3.x);
            rx[kb + 16u + sx] = vec4<f32>(v0.y, v1.y, v2.y, v3.y);
            rx[kb + 32u + sx] = vec4<f32>(v0.z, v1.z, v2.z, v3.z);
            rx[kb + 48u + sx] = vec4<f32>(v0.w, v1.w, v2.w, v3.w);
        }
        {
            let g = (k0 >> 5u) + (kw >> 2u);
            let word = kw & 3u;
            let bit = g * 5u;
            let sh = bit & 7u;
            let r0 = min(wr + 0u, rows - 1u);
            let cb0 = codes_b + r0 * cstride + (bit >> 3u);
            var cv0 = rb(cb0);
            if (sh > 3u) { cv0 = cv0 | (rb(cb0 + 1u) << 8u); }
            let pr0 = unpack2x16float(qmm[params_w + r0]);
            let s0 = exp2(pr0.x + f32((cv0 >> sh) & 31u) * pr0.y);
            let q0 = qmm[(r0 * gpr + g) * 4u + word];
            let r1 = min(wr + 1u, rows - 1u);
            let cb1 = codes_b + r1 * cstride + (bit >> 3u);
            var cv1 = rb(cb1);
            if (sh > 3u) { cv1 = cv1 | (rb(cb1 + 1u) << 8u); }
            let pr1 = unpack2x16float(qmm[params_w + r1]);
            let s1 = exp2(pr1.x + f32((cv1 >> sh) & 31u) * pr1.y);
            let q1 = qmm[(r1 * gpr + g) * 4u + word];
            let r2 = min(wr + 2u, rows - 1u);
            let cb2 = codes_b + r2 * cstride + (bit >> 3u);
            var cv2 = rb(cb2);
            if (sh > 3u) { cv2 = cv2 | (rb(cb2 + 1u) << 8u); }
            let pr2 = unpack2x16float(qmm[params_w + r2]);
            let s2 = exp2(pr2.x + f32((cv2 >> sh) & 31u) * pr2.y);
            let q2 = qmm[(r2 * gpr + g) * 4u + word];
            let r3 = min(wr + 3u, rows - 1u);
            let cb3 = codes_b + r3 * cstride + (bit >> 3u);
            var cv3 = rb(cb3);
            if (sh > 3u) { cv3 = cv3 | (rb(cb3 + 1u) << 8u); }
            let pr3 = unpack2x16float(qmm[params_w + r3]);
            let s3 = exp2(pr3.x + f32((cv3 >> sh) & 31u) * pr3.y);
            let q3 = qmm[(r3 * gpr + g) * 4u + word];
            let sw = ng ^ kw;
            let kb = kw * 128u;
            rw[kb + 0u + sw] = vec4<f32>((f32((q0 >> 0u) & 0xFu) - 8.0) * s0, (f32((q1 >> 0u) & 0xFu) - 8.0) * s1, (f32((q2 >> 0u) & 0xFu) - 8.0) * s2, (f32((q3 >> 0u) & 0xFu) - 8.0) * s3);
            rw[kb + 16u + sw] = vec4<f32>((f32((q0 >> 4u) & 0xFu) - 8.0) * s0, (f32((q1 >> 4u) & 0xFu) - 8.0) * s1, (f32((q2 >> 4u) & 0xFu) - 8.0) * s2, (f32((q3 >> 4u) & 0xFu) - 8.0) * s3);
            rw[kb + 32u + sw] = vec4<f32>((f32((q0 >> 8u) & 0xFu) - 8.0) * s0, (f32((q1 >> 8u) & 0xFu) - 8.0) * s1, (f32((q2 >> 8u) & 0xFu) - 8.0) * s2, (f32((q3 >> 8u) & 0xFu) - 8.0) * s3);
            rw[kb + 48u + sw] = vec4<f32>((f32((q0 >> 12u) & 0xFu) - 8.0) * s0, (f32((q1 >> 12u) & 0xFu) - 8.0) * s1, (f32((q2 >> 12u) & 0xFu) - 8.0) * s2, (f32((q3 >> 12u) & 0xFu) - 8.0) * s3);
            rw[kb + 64u + sw] = vec4<f32>((f32((q0 >> 16u) & 0xFu) - 8.0) * s0, (f32((q1 >> 16u) & 0xFu) - 8.0) * s1, (f32((q2 >> 16u) & 0xFu) - 8.0) * s2, (f32((q3 >> 16u) & 0xFu) - 8.0) * s3);
            rw[kb + 80u + sw] = vec4<f32>((f32((q0 >> 20u) & 0xFu) - 8.0) * s0, (f32((q1 >> 20u) & 0xFu) - 8.0) * s1, (f32((q2 >> 20u) & 0xFu) - 8.0) * s2, (f32((q3 >> 20u) & 0xFu) - 8.0) * s3);
            rw[kb + 96u + sw] = vec4<f32>((f32((q0 >> 24u) & 0xFu) - 8.0) * s0, (f32((q1 >> 24u) & 0xFu) - 8.0) * s1, (f32((q2 >> 24u) & 0xFu) - 8.0) * s2, (f32((q3 >> 24u) & 0xFu) - 8.0) * s3);
            rw[kb + 112u + sw] = vec4<f32>((f32((q0 >> 28u) & 0xFu) - 8.0) * s0, (f32((q1 >> 28u) & 0xFu) - 8.0) * s1, (f32((q2 >> 28u) & 0xFu) - 8.0) * s2, (f32((q3 >> 28u) & 0xFu) - 8.0) * s3);
        }
        workgroupBarrier();
        for (var kq = 0u; kq < 16u; kq = kq + 1u) {
            let sx = tm ^ kq;
            let sw = kq >> 1u;
            let na = (tn * 2u) ^ sw;
            let nbb = (tn * 2u + 1u) ^ sw;
            {
                let kk = kq * 4u + 0u;
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
            {
                let kk = kq * 4u + 1u;
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
            {
                let kk = kq * 4u + 2u;
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
            {
                let kk = kq * 4u + 3u;
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
        k0 = k0 + 64u;
    }
    let nq = n0 + tn * 8u;
    {
        let m = m0 + tm * 4u + 0u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a00; }
            if (nq + 1u < rows) { ymm[o + 1u] = a01; }
            if (nq + 2u < rows) { ymm[o + 2u] = a02; }
            if (nq + 3u < rows) { ymm[o + 3u] = a03; }
            if (nq + 4u < rows) { ymm[o + 4u] = a04; }
            if (nq + 5u < rows) { ymm[o + 5u] = a05; }
            if (nq + 6u < rows) { ymm[o + 6u] = a06; }
            if (nq + 7u < rows) { ymm[o + 7u] = a07; }
        }
    }
    {
        let m = m0 + tm * 4u + 1u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a10; }
            if (nq + 1u < rows) { ymm[o + 1u] = a11; }
            if (nq + 2u < rows) { ymm[o + 2u] = a12; }
            if (nq + 3u < rows) { ymm[o + 3u] = a13; }
            if (nq + 4u < rows) { ymm[o + 4u] = a14; }
            if (nq + 5u < rows) { ymm[o + 5u] = a15; }
            if (nq + 6u < rows) { ymm[o + 6u] = a16; }
            if (nq + 7u < rows) { ymm[o + 7u] = a17; }
        }
    }
    {
        let m = m0 + tm * 4u + 2u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a20; }
            if (nq + 1u < rows) { ymm[o + 1u] = a21; }
            if (nq + 2u < rows) { ymm[o + 2u] = a22; }
            if (nq + 3u < rows) { ymm[o + 3u] = a23; }
            if (nq + 4u < rows) { ymm[o + 4u] = a24; }
            if (nq + 5u < rows) { ymm[o + 5u] = a25; }
            if (nq + 6u < rows) { ymm[o + 6u] = a26; }
            if (nq + 7u < rows) { ymm[o + 7u] = a27; }
        }
    }
    {
        let m = m0 + tm * 4u + 3u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a30; }
            if (nq + 1u < rows) { ymm[o + 1u] = a31; }
            if (nq + 2u < rows) { ymm[o + 2u] = a32; }
            if (nq + 3u < rows) { ymm[o + 3u] = a33; }
            if (nq + 4u < rows) { ymm[o + 4u] = a34; }
            if (nq + 5u < rows) { ymm[o + 5u] = a35; }
            if (nq + 6u < rows) { ymm[o + 6u] = a36; }
            if (nq + 7u < rows) { ymm[o + 7u] = a37; }
        }
    }
}
// Software-pipelined twin of `q4tp_mm_r`: the next step's activation
// vec4s and weight words are loaded into registers while the current step
// computes from workgroup memory, so a step's global latency hides behind
// the previous step's FMAs. Same staging values, same sequential sums.
@compute @workgroup_size(128)
fn q4tp_mm_r2(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) t: u32) {
    let cols4 = pmm.cols4;
    let cols = cols4 * 4u;
    let gpr = cols >> 5u;
    let rows = pmm.rows;
    let nb = pmm.nb;
    let m0 = wid.y * 64u;
    let n0 = wid.x * 64u;
    let params_w = rows * gpr * 4u;
    let codes_b = rows * gpr * 16u + rows * 4u;
    let cstride = (gpr * 5u + 7u) / 8u;
    let kw = t & 7u;
    let ng = t >> 3u;
    let wr = n0 + ng * 4u;
    let tm = t & 15u;
    let tn = t >> 4u;
    let word = kw & 3u;
    let r0 = min(wr + 0u, rows - 1u);
    let pr0 = unpack2x16float(qmm[params_w + r0]);
    let cr0 = codes_b + r0 * cstride;
    let r1 = min(wr + 1u, rows - 1u);
    let pr1 = unpack2x16float(qmm[params_w + r1]);
    let cr1 = codes_b + r1 * cstride;
    let r2 = min(wr + 2u, rows - 1u);
    let pr2 = unpack2x16float(qmm[params_w + r2]);
    let cr2 = codes_b + r2 * cstride;
    let r3 = min(wr + 3u, rows - 1u);
    let pr3 = unpack2x16float(qmm[params_w + r3]);
    let cr3 = codes_b + r3 * cstride;
    var a00 = 0.0;
    var a01 = 0.0;
    var a02 = 0.0;
    var a03 = 0.0;
    var a04 = 0.0;
    var a05 = 0.0;
    var a06 = 0.0;
    var a07 = 0.0;
    var a10 = 0.0;
    var a11 = 0.0;
    var a12 = 0.0;
    var a13 = 0.0;
    var a14 = 0.0;
    var a15 = 0.0;
    var a16 = 0.0;
    var a17 = 0.0;
    var a20 = 0.0;
    var a21 = 0.0;
    var a22 = 0.0;
    var a23 = 0.0;
    var a24 = 0.0;
    var a25 = 0.0;
    var a26 = 0.0;
    var a27 = 0.0;
    var a30 = 0.0;
    var a31 = 0.0;
    var a32 = 0.0;
    var a33 = 0.0;
    var a34 = 0.0;
    var a35 = 0.0;
    var a36 = 0.0;
    var a37 = 0.0;
    // x staging tasks u = t and t + 128: kq = u & 15, mg = u >> 4.
    let kqa = t & 15u;
    let mga = t >> 4u;
    let mgb = mga + 8u;
    let ma = m0 + mga * 4u;
    let mb = m0 + mgb * 4u;
    var xa0: vec4<f32>;
    var xa1: vec4<f32>;
    var xa2: vec4<f32>;
    var xa3: vec4<f32>;
    var xb0: vec4<f32>;
    var xb1: vec4<f32>;
    var xb2: vec4<f32>;
    var xb3: vec4<f32>;
    var cb0: u32;
    var q0: u32;
    var cb1: u32;
    var q1: u32;
    var cb2: u32;
    var q2: u32;
    var cb3: u32;
    var q3: u32;
    var sh: u32;
    {
        xa0 = select(vec4<f32>(0.0), xmm4[min(ma + 0u, nb - 1u) * cols4 + (0u >> 2u) + kqa], ma + 0u < nb);
        xa1 = select(vec4<f32>(0.0), xmm4[min(ma + 1u, nb - 1u) * cols4 + (0u >> 2u) + kqa], ma + 1u < nb);
        xa2 = select(vec4<f32>(0.0), xmm4[min(ma + 2u, nb - 1u) * cols4 + (0u >> 2u) + kqa], ma + 2u < nb);
        xa3 = select(vec4<f32>(0.0), xmm4[min(ma + 3u, nb - 1u) * cols4 + (0u >> 2u) + kqa], ma + 3u < nb);
        xb0 = select(vec4<f32>(0.0), xmm4[min(mb + 0u, nb - 1u) * cols4 + (0u >> 2u) + kqa], mb + 0u < nb);
        xb1 = select(vec4<f32>(0.0), xmm4[min(mb + 1u, nb - 1u) * cols4 + (0u >> 2u) + kqa], mb + 1u < nb);
        xb2 = select(vec4<f32>(0.0), xmm4[min(mb + 2u, nb - 1u) * cols4 + (0u >> 2u) + kqa], mb + 2u < nb);
        xb3 = select(vec4<f32>(0.0), xmm4[min(mb + 3u, nb - 1u) * cols4 + (0u >> 2u) + kqa], mb + 3u < nb);
    }
    {
        let gn_ = (0u >> 5u) + (kw >> 2u);
        let bitn_ = gn_ * 5u;
        cb0 = rb(cr0 + (bitn_ >> 3u)) | (rb(cr0 + (bitn_ >> 3u) + 1u) << 8u);
        q0 = qmm[(r0 * gpr + gn_) * 4u + word];
        cb1 = rb(cr1 + (bitn_ >> 3u)) | (rb(cr1 + (bitn_ >> 3u) + 1u) << 8u);
        q1 = qmm[(r1 * gpr + gn_) * 4u + word];
        cb2 = rb(cr2 + (bitn_ >> 3u)) | (rb(cr2 + (bitn_ >> 3u) + 1u) << 8u);
        q2 = qmm[(r2 * gpr + gn_) * 4u + word];
        cb3 = rb(cr3 + (bitn_ >> 3u)) | (rb(cr3 + (bitn_ >> 3u) + 1u) << 8u);
        q3 = qmm[(r3 * gpr + gn_) * 4u + word];
        sh = bitn_ & 7u;
    }
    var k0 = 0u;
    loop {
        if (k0 >= cols) { break; }
        {
            let kb = kqa * 64u;
            let sx = mga ^ kqa;
            rx[kb + sx] = vec4<f32>(xa0.x, xa1.x, xa2.x, xa3.x);
            rx[kb + 16u + sx] = vec4<f32>(xa0.y, xa1.y, xa2.y, xa3.y);
            rx[kb + 32u + sx] = vec4<f32>(xa0.z, xa1.z, xa2.z, xa3.z);
            rx[kb + 48u + sx] = vec4<f32>(xa0.w, xa1.w, xa2.w, xa3.w);
        }
        {
            let kb = kqa * 64u;
            let sx = mgb ^ kqa;
            rx[kb + sx] = vec4<f32>(xb0.x, xb1.x, xb2.x, xb3.x);
            rx[kb + 16u + sx] = vec4<f32>(xb0.y, xb1.y, xb2.y, xb3.y);
            rx[kb + 32u + sx] = vec4<f32>(xb0.z, xb1.z, xb2.z, xb3.z);
            rx[kb + 48u + sx] = vec4<f32>(xb0.w, xb1.w, xb2.w, xb3.w);
        }
        {
            let s0 = exp2(pr0.x + f32((cb0 >> sh) & 31u) * pr0.y);
            let s1 = exp2(pr1.x + f32((cb1 >> sh) & 31u) * pr1.y);
            let s2 = exp2(pr2.x + f32((cb2 >> sh) & 31u) * pr2.y);
            let s3 = exp2(pr3.x + f32((cb3 >> sh) & 31u) * pr3.y);
            let sw = ng ^ kw;
            let kb = kw * 128u;
            rw[kb + 0u + sw] = vec4<f32>((f32((q0 >> 0u) & 0xFu) - 8.0) * s0, (f32((q1 >> 0u) & 0xFu) - 8.0) * s1, (f32((q2 >> 0u) & 0xFu) - 8.0) * s2, (f32((q3 >> 0u) & 0xFu) - 8.0) * s3);
            rw[kb + 16u + sw] = vec4<f32>((f32((q0 >> 4u) & 0xFu) - 8.0) * s0, (f32((q1 >> 4u) & 0xFu) - 8.0) * s1, (f32((q2 >> 4u) & 0xFu) - 8.0) * s2, (f32((q3 >> 4u) & 0xFu) - 8.0) * s3);
            rw[kb + 32u + sw] = vec4<f32>((f32((q0 >> 8u) & 0xFu) - 8.0) * s0, (f32((q1 >> 8u) & 0xFu) - 8.0) * s1, (f32((q2 >> 8u) & 0xFu) - 8.0) * s2, (f32((q3 >> 8u) & 0xFu) - 8.0) * s3);
            rw[kb + 48u + sw] = vec4<f32>((f32((q0 >> 12u) & 0xFu) - 8.0) * s0, (f32((q1 >> 12u) & 0xFu) - 8.0) * s1, (f32((q2 >> 12u) & 0xFu) - 8.0) * s2, (f32((q3 >> 12u) & 0xFu) - 8.0) * s3);
            rw[kb + 64u + sw] = vec4<f32>((f32((q0 >> 16u) & 0xFu) - 8.0) * s0, (f32((q1 >> 16u) & 0xFu) - 8.0) * s1, (f32((q2 >> 16u) & 0xFu) - 8.0) * s2, (f32((q3 >> 16u) & 0xFu) - 8.0) * s3);
            rw[kb + 80u + sw] = vec4<f32>((f32((q0 >> 20u) & 0xFu) - 8.0) * s0, (f32((q1 >> 20u) & 0xFu) - 8.0) * s1, (f32((q2 >> 20u) & 0xFu) - 8.0) * s2, (f32((q3 >> 20u) & 0xFu) - 8.0) * s3);
            rw[kb + 96u + sw] = vec4<f32>((f32((q0 >> 24u) & 0xFu) - 8.0) * s0, (f32((q1 >> 24u) & 0xFu) - 8.0) * s1, (f32((q2 >> 24u) & 0xFu) - 8.0) * s2, (f32((q3 >> 24u) & 0xFu) - 8.0) * s3);
            rw[kb + 112u + sw] = vec4<f32>((f32((q0 >> 28u) & 0xFu) - 8.0) * s0, (f32((q1 >> 28u) & 0xFu) - 8.0) * s1, (f32((q2 >> 28u) & 0xFu) - 8.0) * s2, (f32((q3 >> 28u) & 0xFu) - 8.0) * s3);
        }
        workgroupBarrier();
        let kn = k0 + 64u;
        if (kn < cols) {
            xa0 = select(vec4<f32>(0.0), xmm4[min(ma + 0u, nb - 1u) * cols4 + (kn >> 2u) + kqa], ma + 0u < nb);
            xa1 = select(vec4<f32>(0.0), xmm4[min(ma + 1u, nb - 1u) * cols4 + (kn >> 2u) + kqa], ma + 1u < nb);
            xa2 = select(vec4<f32>(0.0), xmm4[min(ma + 2u, nb - 1u) * cols4 + (kn >> 2u) + kqa], ma + 2u < nb);
            xa3 = select(vec4<f32>(0.0), xmm4[min(ma + 3u, nb - 1u) * cols4 + (kn >> 2u) + kqa], ma + 3u < nb);
            xb0 = select(vec4<f32>(0.0), xmm4[min(mb + 0u, nb - 1u) * cols4 + (kn >> 2u) + kqa], mb + 0u < nb);
            xb1 = select(vec4<f32>(0.0), xmm4[min(mb + 1u, nb - 1u) * cols4 + (kn >> 2u) + kqa], mb + 1u < nb);
            xb2 = select(vec4<f32>(0.0), xmm4[min(mb + 2u, nb - 1u) * cols4 + (kn >> 2u) + kqa], mb + 2u < nb);
            xb3 = select(vec4<f32>(0.0), xmm4[min(mb + 3u, nb - 1u) * cols4 + (kn >> 2u) + kqa], mb + 3u < nb);
            let gn_ = (kn >> 5u) + (kw >> 2u);
            let bitn_ = gn_ * 5u;
            cb0 = rb(cr0 + (bitn_ >> 3u)) | (rb(cr0 + (bitn_ >> 3u) + 1u) << 8u);
            q0 = qmm[(r0 * gpr + gn_) * 4u + word];
            cb1 = rb(cr1 + (bitn_ >> 3u)) | (rb(cr1 + (bitn_ >> 3u) + 1u) << 8u);
            q1 = qmm[(r1 * gpr + gn_) * 4u + word];
            cb2 = rb(cr2 + (bitn_ >> 3u)) | (rb(cr2 + (bitn_ >> 3u) + 1u) << 8u);
            q2 = qmm[(r2 * gpr + gn_) * 4u + word];
            cb3 = rb(cr3 + (bitn_ >> 3u)) | (rb(cr3 + (bitn_ >> 3u) + 1u) << 8u);
            q3 = qmm[(r3 * gpr + gn_) * 4u + word];
            sh = bitn_ & 7u;

        }
        for (var kq = 0u; kq < 16u; kq = kq + 1u) {
            let sx = tm ^ kq;
            let sw = kq >> 1u;
            let na = (tn * 2u) ^ sw;
            let nbb = (tn * 2u + 1u) ^ sw;
            {
                let kk = kq * 4u + 0u;
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
            {
                let kk = kq * 4u + 1u;
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
            {
                let kk = kq * 4u + 2u;
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
            {
                let kk = kq * 4u + 3u;
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
    let nq = n0 + tn * 8u;
    {
        let m = m0 + tm * 4u + 0u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a00; }
            if (nq + 1u < rows) { ymm[o + 1u] = a01; }
            if (nq + 2u < rows) { ymm[o + 2u] = a02; }
            if (nq + 3u < rows) { ymm[o + 3u] = a03; }
            if (nq + 4u < rows) { ymm[o + 4u] = a04; }
            if (nq + 5u < rows) { ymm[o + 5u] = a05; }
            if (nq + 6u < rows) { ymm[o + 6u] = a06; }
            if (nq + 7u < rows) { ymm[o + 7u] = a07; }
        }
    }
    {
        let m = m0 + tm * 4u + 1u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a10; }
            if (nq + 1u < rows) { ymm[o + 1u] = a11; }
            if (nq + 2u < rows) { ymm[o + 2u] = a12; }
            if (nq + 3u < rows) { ymm[o + 3u] = a13; }
            if (nq + 4u < rows) { ymm[o + 4u] = a14; }
            if (nq + 5u < rows) { ymm[o + 5u] = a15; }
            if (nq + 6u < rows) { ymm[o + 6u] = a16; }
            if (nq + 7u < rows) { ymm[o + 7u] = a17; }
        }
    }
    {
        let m = m0 + tm * 4u + 2u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a20; }
            if (nq + 1u < rows) { ymm[o + 1u] = a21; }
            if (nq + 2u < rows) { ymm[o + 2u] = a22; }
            if (nq + 3u < rows) { ymm[o + 3u] = a23; }
            if (nq + 4u < rows) { ymm[o + 4u] = a24; }
            if (nq + 5u < rows) { ymm[o + 5u] = a25; }
            if (nq + 6u < rows) { ymm[o + 6u] = a26; }
            if (nq + 7u < rows) { ymm[o + 7u] = a27; }
        }
    }
    {
        let m = m0 + tm * 4u + 3u;
        if (m < nb) {
            let o = m * rows + nq;
            if (nq + 0u < rows) { ymm[o + 0u] = a30; }
            if (nq + 1u < rows) { ymm[o + 1u] = a31; }
            if (nq + 2u < rows) { ymm[o + 2u] = a32; }
            if (nq + 3u < rows) { ymm[o + 3u] = a33; }
            if (nq + 4u < rows) { ymm[o + 4u] = a34; }
            if (nq + 5u < rows) { ymm[o + 5u] = a35; }
            if (nq + 6u < rows) { ymm[o + 6u] = a36; }
            if (nq + 7u < rows) { ymm[o + 7u] = a37; }
        }
    }
}
"#;
