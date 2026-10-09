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
}

pub(crate) struct Pipes {
    pub(crate) attn: Option<AttnBt>,
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
    })
}

fn build(c: &Ctx) -> Option<Pipes> {
    Some(Pipes {
        attn: build_attn(c),
    })
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
/// `t_from..k` (positions `pos0 + t`): rope + K/V append + split attend +
/// merge, writing attention rows `t` of `attn_bb`. The caller applies the
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
    let mut t0 = t_from;
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

