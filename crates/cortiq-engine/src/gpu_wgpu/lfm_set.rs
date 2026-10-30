//! d1's one-pass forward on wgpu (`pipeline_masked::forward_masked`), the
//! twin of `gpu_metal::lfm_set`: every layer of an LFM3 stack over the whole
//! packed row in one submission — the batch graph's q4tp / q8_2f GEMMs, the
//! short convolution walked over the positions from an empty ring, Q/K norm
//! + RoPE per row, the `set` attention (or causal within a sliding window)
//! and the per-head sigmoid gate. The host path ran each projection as its
//! own submission: ~0.7 s for a 228-token request on an A40.
//!
//! No subgroup operations: every reduction goes through workgroup memory,
//! so the module compiles on every adapter the engine runs.
//! `CMF_D1_DEVICE=0` keeps the host path.

use super::*;
use crate::gpu::{SetDims, SetHead, SetLayer, SetMixer, SetW};

const LS_WGSL: &str = r#"
// h += rms(d)·w per row (the sandwich norm before the residual add), or
// h += d when `hasw` is 0. One 256-lane group a row.
struct NaP { n: u32, gemma: u32, eps: f32, hasw: u32 };
@group(0) @binding(0) var<storage, read_write> na_h: array<f32>;
@group(0) @binding(1) var<storage, read>       na_d: array<f32>;
@group(0) @binding(2) var<storage, read>       na_w: array<f32>;
@group(0) @binding(3) var<uniform>             na_p: NaP;
var<workgroup> na_part: array<f32, 256>;
@compute @workgroup_size(256)
fn ls_norm_add_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = na_p.n;
    let base = wid.x * n;
    var inv = 1.0;
    if (na_p.hasw != 0u) {
        var acc = 0.0;
        var i = tid;
        loop { if (i >= n) { break; } let v = na_d[base + i]; acc = acc + v * v; i = i + 256u; }
        na_part[tid] = acc;
        workgroupBarrier();
        var s = 128u;
        loop { if (s == 0u) { break; } if (tid < s) { na_part[tid] = na_part[tid] + na_part[tid + s]; } workgroupBarrier(); s = s / 2u; }
        inv = inverseSqrt(na_part[0] / f32(n) + na_p.eps);
    }
    var i = tid;
    loop {
        if (i >= n) { break; }
        var f = na_d[base + i];
        if (na_p.hasw != 0u) {
            var wv = na_w[i];
            if (na_p.gemma != 0u) { wv = wv + 1.0; }
            f = f * inv * wv;
        }
        na_h[base + i] = na_h[base + i] + f;
        i = i + 256u;
    }
}

// The gated short convolution over `t` positions from an empty ring, one
// lane a channel: bcx rows [B | C | x] (3h), taps [h][k] with tap k-1 on
// the current position; y = C·(tap[k-1]·B·x + Σ tap[k-2-s]·ring[s]).
struct ScP { h: u32, k: u32, t: u32, pad: u32 };
@group(0) @binding(0) var<storage, read>       sc_bcx:  array<f32>;
@group(0) @binding(1) var<storage, read>       sc_taps: array<f32>;
@group(0) @binding(2) var<storage, read_write> sc_y:    array<f32>;
@group(0) @binding(3) var<uniform>             sc_p:    ScP;
@compute @workgroup_size(256)
fn ls_sconv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    let h = sc_p.h;
    if (ch >= h) { return; }
    let k = sc_p.k;
    let rl = k - 1u;
    var ring: array<f32, 15>;
    for (var s = 0u; s < 15u; s = s + 1u) { ring[s] = 0.0; }
    let tb = ch * k;
    for (var r = 0u; r < sc_p.t; r = r + 1u) {
        let rb = r * 3u * h;
        let bx = sc_bcx[rb + ch] * sc_bcx[rb + 2u * h + ch];
        var acc = sc_taps[tb + k - 1u] * bx;
        for (var s = 0u; s < rl; s = s + 1u) { acc = acc + sc_taps[tb + k - 2u - s] * ring[s]; }
        sc_y[r * h + ch] = sc_bcx[rb + h + ch] * acc;
        var j = rl;
        loop { if (j <= 1u) { break; } ring[j - 1u] = ring[j - 2u]; j = j - 1u; }
        if (rl > 0u) { ring[0] = bx; }
    }
}

// Per (head, row): the head's RMS norm (when `hasw`), then RoPE at the
// row's own position over the first `rd` dims (pairs i, i + rd/2), in
// place on x [t][heads·hd]. One 32-lane group a head.
struct QrP { heads: u32, hd: u32, rd: u32, hasw: u32, gemma: u32, eps: f32, pa: u32, pb: u32 };
@group(0) @binding(0) var<storage, read_write> qr_x:    array<f32>;
@group(0) @binding(1) var<storage, read>       qr_w:    array<f32>;
@group(0) @binding(2) var<storage, read>       qr_invf: array<f32>;
@group(0) @binding(3) var<uniform>             qr_p:    QrP;
var<workgroup> qr_a: array<f32, 512>;
var<workgroup> qr_red: array<f32, 32>;
@compute @workgroup_size(32)
fn ls_qk_rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let row = wid.y;
    let lane = lid.x;
    let hd = qr_p.hd;
    let base = (row * qr_p.heads + head) * hd;
    var ss = 0.0;
    var d = lane;
    loop { if (d >= hd) { break; } let e = qr_x[base + d]; qr_a[d] = e; ss = ss + e * e; d = d + 32u; }
    qr_red[lane] = ss;
    workgroupBarrier();
    var s = 16u;
    loop { if (s == 0u) { break; } if (lane < s) { qr_red[lane] = qr_red[lane] + qr_red[lane + s]; } workgroupBarrier(); s = s / 2u; }
    if (qr_p.hasw != 0u) {
        let inv = inverseSqrt(qr_red[0] / f32(hd) + qr_p.eps);
        d = lane;
        loop {
            if (d >= hd) { break; }
            var wv = qr_w[d];
            if (qr_p.gemma != 0u) { wv = wv + 1.0; }
            qr_a[d] = qr_a[d] * inv * wv;
            d = d + 32u;
        }
    }
    workgroupBarrier();
    let hr = qr_p.rd / 2u;
    let pos = f32(row);
    var i = lane;
    loop {
        if (i >= hr) { break; }
        let ang = pos * qr_invf[i];
        let sn = sin(ang);
        let cs = cos(ang);
        let x0 = qr_a[i];
        let x1 = qr_a[i + hr];
        qr_a[i] = x0 * cs - x1 * sn;
        qr_a[i + hr] = x0 * sn + x1 * cs;
        i = i + 32u;
    }
    workgroupBarrier();
    d = lane;
    loop { if (d >= hd) { break; } qr_x[base + d] = qr_a[d]; d = d + 32u; }
}

// Attention of row i (wid.x), head hh (wid.y) over the packed row: online
// softmax in 128-key blocks, one key a lane for the scores, one dim a lane
// for P·V. `window` 0: d1's `set` mask (the state reads every row and is
// read by every row; a question row reads its own question); otherwise
// causal within the window. hd ≤ 256.
struct AtP { t: u32, nh: u32, nkv: u32, hd: u32, state: u32, window: u32, scale: f32, pad: u32 };
@group(0) @binding(0) var<storage, read>       at_q: array<f32>;
@group(0) @binding(1) var<storage, read>       at_k: array<f32>;
@group(0) @binding(2) var<storage, read>       at_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> at_o: array<f32>;
@group(0) @binding(4) var<storage, read>       at_g: array<i32>;
@group(0) @binding(5) var<uniform>             at_p: AtP;
var<workgroup> at_qs: array<f32, 256>;
var<workgroup> at_pv: array<f32, 128>;
var<workgroup> at_red: array<f32, 128>;
const NEG: f32 = -3.0e38;
@compute @workgroup_size(128)
fn ls_attn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = wid.x;
    let hh = wid.y;
    let tid = lid.x;
    let hd = at_p.hd;
    let nkv = at_p.nkv;
    let g = hh / (at_p.nh / nkv);
    let qb = (i * at_p.nh + hh) * hd;
    var d = tid;
    loop { if (d >= hd) { break; } at_qs[d] = at_q[qb + d]; d = d + 128u; }
    workgroupBarrier();
    let gi = at_g[i];
    let kvs = nkv * hd;
    var jlo = 0u;
    var jhi = at_p.t;
    if (at_p.window != 0u) {
        jlo = select(0u, i + 1u - at_p.window, i + 1u > at_p.window);
        jhi = i + 1u;
    }
    var m = NEG;
    var l = 0.0;
    var acc0 = 0.0;
    var acc1 = 0.0;
    var j0 = jlo;
    loop {
        if (j0 >= jhi) { break; }
        let j = j0 + tid;
        var ok = j < jhi;
        if (ok && at_p.window == 0u) { ok = (i < at_p.state) || (j < at_p.state) || (at_g[j] == gi); }
        var sc = NEG;
        if (ok) {
            let kb = j * kvs + g * hd;
            var dot = 0.0;
            for (var e = 0u; e < hd; e = e + 1u) { dot = dot + at_qs[e] * at_k[kb + e]; }
            sc = dot * at_p.scale;
        }
        at_red[tid] = sc;
        workgroupBarrier();
        var st = 64u;
        loop { if (st == 0u) { break; } if (tid < st) { at_red[tid] = max(at_red[tid], at_red[tid + st]); } workgroupBarrier(); st = st / 2u; }
        let bm = workgroupUniformLoad(&at_red[0]);
        let mn = max(m, bm);
        if (mn > -1.0e38) {
            var pj = 0.0;
            if (ok) { pj = exp(sc - mn); }
            at_pv[tid] = pj;
            at_red[tid] = pj;
            workgroupBarrier();
            st = 64u;
            loop { if (st == 0u) { break; } if (tid < st) { at_red[tid] = at_red[tid] + at_red[tid + st]; } workgroupBarrier(); st = st / 2u; }
            let bs = at_red[0];
            var corr = 0.0;
            if (m > -1.0e38) { corr = exp(m - mn); }
            l = l * corr + bs;
            acc0 = acc0 * corr;
            acc1 = acc1 * corr;
            let nb = min(128u, jhi - j0);
            for (var jj = 0u; jj < nb; jj = jj + 1u) {
                let pv = at_pv[jj];
                if (pv != 0.0) {
                    let vb = (j0 + jj) * kvs + g * hd;
                    if (tid < hd) { acc0 = acc0 + pv * at_v[vb + tid]; }
                    if (tid + 128u < hd) { acc1 = acc1 + pv * at_v[vb + tid + 128u]; }
                }
            }
            m = mn;
            workgroupBarrier();
        }
        j0 = j0 + 128u;
    }
    let ob = (i * at_p.nh + hh) * hd;
    var il = 0.0;
    if (l > 0.0) { il = 1.0 / l; }
    if (tid < hd) { at_o[ob + tid] = acc0 * il; }
    if (tid + 128u < hd) { at_o[ob + tid + 128u] = acc1 * il; }
}

// The per-head sigmoid gate: σ(x_row · gw[head]) scales the head's
// attention output, one 256-lane group per (head, row).
struct HgP { nh: u32, hd: u32, hidden: u32, pad: u32 };
@group(0) @binding(0) var<storage, read_write> hg_ao: array<f32>;
@group(0) @binding(1) var<storage, read>       hg_x:  array<f32>;
@group(0) @binding(2) var<storage, read>       hg_w:  array<f32>;
@group(0) @binding(3) var<uniform>             hg_p:  HgP;
var<workgroup> hg_part: array<f32, 256>;
@compute @workgroup_size(256)
fn ls_head_gate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hh = wid.x;
    let row = wid.y;
    let tid = lid.x;
    let n = hg_p.hidden;
    var acc = 0.0;
    var i = tid;
    loop { if (i >= n) { break; } acc = acc + hg_x[row * n + i] * hg_w[hh * n + i]; i = i + 256u; }
    hg_part[tid] = acc;
    workgroupBarrier();
    var s = 128u;
    loop { if (s == 0u) { break; } if (tid < s) { hg_part[tid] = hg_part[tid] + hg_part[tid + s]; } workgroupBarrier(); s = s / 2u; }
    let gate = 1.0 / (1.0 + exp(-hg_part[0]));
    let ab = (row * hg_p.nh + hh) * hg_p.hd;
    var d = tid;
    loop { if (d >= hg_p.hd) { break; } hg_ao[ab + d] = hg_ao[ab + d] * gate; d = d + 256u; }
}

// dst[r] = src[idx[r]] for rows of n floats (the head's answer slots).
struct GaP { n: u32, pa: u32, pb: u32, pc: u32 };
@group(0) @binding(0) var<storage, read>       ga_src: array<f32>;
@group(0) @binding(1) var<storage, read>       ga_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> ga_dst: array<f32>;
@group(0) @binding(3) var<uniform>             ga_p:   GaP;
@compute @workgroup_size(256)
fn ls_gather_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = ga_p.n;
    if (gid.x >= n) { return; }
    ga_dst[gid.y * n + gid.x] = ga_src[ga_idx[gid.y] * n + gid.x];
}

// a = silu(g)·u over n elements, grid-stride.
struct SmP { n: u32, stride: u32, pa: u32, pb: u32 };
@group(0) @binding(0) var<storage, read>       sm_g: array<f32>;
@group(0) @binding(1) var<storage, read>       sm_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> sm_a: array<f32>;
@group(0) @binding(3) var<uniform>             sm_p: SmP;
@compute @workgroup_size(256)
fn ls_silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    var i = gid.x;
    loop {
        if (i >= sm_p.n) { break; }
        let x = sm_g[i];
        sm_a[i] = x / (1.0 + exp(-x)) * sm_u[i];
        i = i + sm_p.stride;
    }
}
"#;

struct Pipe {
    p: wgpu::ComputePipeline,
    l: wgpu::BindGroupLayout,
}

struct Pipes {
    norm_add: Pipe,
    sconv: Pipe,
    qk_rope: Pipe,
    attn: Pipe,
    head_gate: Pipe,
    silu_mul: Pipe,
    gather: Pipe,
}

static PIPES: std::sync::OnceLock<Option<Pipes>> = std::sync::OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cmf-d1-set"),
                source: wgpu::ShaderSource::Wgsl(LS_WGSL.into()),
            });
            let pipe = |ep: &str| {
                let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(ep),
                    layout: None,
                    module: &module,
                    entry_point: Some(ep),
                    compilation_options: pco(),
                    cache: c.pipeline_cache.as_ref(),
                });
                Pipe {
                    l: p.get_bind_group_layout(0),
                    p,
                }
            };
            let pipes = Pipes {
                norm_add: pipe("ls_norm_add_rows"),
                sconv: pipe("ls_sconv"),
                qk_rope: pipe("ls_qk_rope"),
                attn: pipe("ls_attn"),
                head_gate: pipe("ls_head_gate"),
                silu_mul: pipe("ls_silu_mul"),
                gather: pipe("ls_gather_rows"),
            };
            if let Some(e) = pollster::block_on(scope.pop()) {
                tracing::warn!("cmf-d1-set module rejected ({e}): d1 keeps the host path");
                return None;
            }
            Some(pipes)
        })
        .as_ref()
}

/// A resident projection: q4tp tiles or the whole q8_2f tensor.
enum W {
    Q4(wgpu::Buffer),
    Q8(wgpu::Buffer),
}

struct Proj {
    w: W,
    rows: usize,
    cols: usize,
}

fn resolve(c: &Ctx, model: &Arc<CmfModel>, w: SetW) -> Option<Proj> {
    let (idx, rows, cols) = w;
    let e = model.tensors.get(idx)?;
    if e.shape.as_slice() != [rows, cols] || cols % 32 != 0 {
        return None;
    }
    let w = match e.dtype {
        cortiq_core::TensorDtype::Q4TiledP => {
            let (b, r, cc) = tile_weight(c, model, idx)?;
            if r != rows || cc != cols {
                return None;
            }
            W::Q4(b)
        }
        cortiq_core::TensorDtype::Q8_2f if c.graph_x.is_some() => {
            let abs = model.entry_abs_offset(e)?;
            let plen = e.nbytes as usize;
            let bytes = model.primary_bytes();
            if abs + plen > bytes.len() {
                return None;
            }
            W::Q8(weight_buffer_l(
                c,
                (model.uid() as usize, idx),
                &bytes[abs..abs + plen],
                layer_of_name(&e.name),
            )?)
        }
        _ => return None,
    };
    Some(Proj { w, rows, cols })
}

/// y[k][rows] = x[k][cols] · Wᵀ through the batch graph's kernels.
fn mm(c: &Ctx, enc: &mut wgpu::CommandEncoder, w: &Proj, xs: &wgpu::Buffer, y: &wgpu::Buffer, k: usize) {
    let (rows, cols) = (w.rows, w.cols);
    match &w.w {
        W::Q4(b) => {
            if !(k > 16 && dense_batch::encode_q4tp_mm_r(c, enc, b, xs, y, rows, cols, k)) {
                encode_q4_tile_mm(c, enc, &c.q4tp_mm, b, xs, y, rows, cols, k);
            }
        }
        W::Q8(b) => {
            if k > 16
                && gemma_batch::pipes(c)
                    .is_some_and(|gp| gemma_batch::encode_q82_mm(c, enc, gp, b, xs, y, rows, cols, k))
            {
                return;
            }
            let gx = c.graph_x.as_ref().expect("checked in resolve");
            let p_buf = uniform_u32x4(c, [(cols / 4) as u32, rows as u32, cols as u32, k as u32]);
            let bind = bind_pairs(c, &gx.q82_l, &[(0, b), (1, xs), (2, y), (3, &p_buf)]);
            let mut pass = begin_pass(enc);
            pass.set_pipeline(&gx.q82_b);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                (rows as u32).div_ceil(4).min(MAX_WG),
                (k as u32).div_ceil(8),
                1,
            );
        }
    }
}

/// A grow-only scratch buffer per role (the call's sizes vary).
fn scratch(c: &Ctx, role: u8, bytes: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    static SCR: std::sync::Mutex<Vec<(u8, wgpu::Buffer)>> = std::sync::Mutex::new(Vec::new());
    let mut m = SCR.lock().unwrap();
    if let Some((_, b)) = m.iter().find(|(r, b)| *r == role && b.size() >= bytes as u64) {
        return b.clone();
    }
    m.retain(|(r, _)| *r != role);
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("d1-set"),
        size: bytes.max(16) as u64,
        usage,
        mapped_at_creation: false,
    });
    m.push((role, b.clone()));
    b
}

/// The final-normed hidden rows `[t, hidden]` of `rows` under d1's mask
/// and, with `head`, the logits of its rows; None when the stack or a
/// weight is outside these kernels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    model: &Arc<CmfModel>,
    layers: &[SetLayer],
    d: &SetDims,
    final_norm: &[f32],
    rows: &[f32],
    t: usize,
    state: usize,
    group: &[i32],
    head: Option<&SetHead>,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let c = ctx()?;
    let p = pipes(c)?;
    let hs = d.hidden;
    if t < 8
        || t > 32768
        || rows.len() != t * hs
        || group.len() != t
        || state > t
        || d.hd % 4 != 0
        || d.hd > 256
        || d.nh == 0
    {
        return None;
    }
    struct Prep {
        mix: Vec<Proj>,
        ffn: [Proj; 3],
    }
    let mut preps = Vec::with_capacity(layers.len());
    let (mut wide, mut inter_max, mut kv_max) = (3 * hs, 0usize, 4usize);
    for l in layers {
        let ffn = [
            resolve(c, model, l.gate)?,
            resolve(c, model, l.up)?,
            resolve(c, model, l.down)?,
        ];
        let inter = ffn[0].rows;
        if ffn[1].rows != inter || ffn[2].cols != inter || ffn[0].cols != hs || ffn[2].rows != hs
        {
            return None;
        }
        inter_max = inter_max.max(inter);
        let mix = match &l.mixer {
            SetMixer::Conv {
                in_proj,
                out_proj,
                taps,
                kernel,
            } => {
                let (a, b) = (resolve(c, model, *in_proj)?, resolve(c, model, *out_proj)?);
                if !(1..=16).contains(kernel)
                    || taps.len() != hs * kernel
                    || a.rows != 3 * hs
                    || a.cols != hs
                    || b.rows != hs
                    || b.cols != hs
                {
                    return None;
                }
                vec![a, b]
            }
            SetMixer::Attn {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                head_gate,
                inv_freq,
                rd,
                nkv,
                window,
            } => {
                let w = vec![
                    resolve(c, model, *wq)?,
                    resolve(c, model, *wk)?,
                    resolve(c, model, *wv)?,
                    resolve(c, model, *wo)?,
                ];
                let (qd, kvd) = (d.nh * d.hd, nkv * d.hd);
                if *nkv == 0
                    || d.nh % nkv != 0
                    || w[0].rows != qd
                    || w[1].rows != kvd
                    || w[2].rows != kvd
                    || w[3].cols != qd
                    || w[3].rows != hs
                    || w.iter().take(3).any(|x| x.cols != hs)
                    || *rd > d.hd
                    || rd % 2 != 0
                    || inv_freq.len() < rd / 2
                    || q_norm.is_some_and(|n| n.len() != d.hd)
                    || k_norm.is_some_and(|n| n.len() != d.hd)
                    || head_gate.is_some_and(|g| g.len() != d.nh * hs)
                    || *window == Some(0)
                {
                    return None;
                }
                wide = wide.max(qd);
                kv_max = kv_max.max(kvd);
                w
            }
        };
        preps.push(Prep { mix, ffn });
    }
    let wide = wide.max(inter_max);
    let head = match head {
        Some(hd) if !hd.rows.is_empty() => {
            let w = resolve(c, model, hd.w)?;
            if w.cols != hs || hd.rows.iter().any(|&r| r >= t) || hd.rows.len() > t {
                return None;
            }
            Some((w, hd.rows.iter().map(|&r| r as u32).collect::<Vec<u32>>()))
        }
        _ => None,
    };

    use wgpu::BufferUsages as U;
    let st = U::STORAGE | U::COPY_DST | U::COPY_SRC;
    let h_b = scratch(c, 1, t * hs * 4, st);
    let n_b = scratch(c, 2, t * hs * 4, st);
    let d_b = scratch(c, 3, t * hs * 4, st);
    let x_b = scratch(c, 4, t * wide * 4, st);
    let u_b = scratch(c, 5, t * inter_max.max(d.nh * d.hd) * 4, st);
    let k_b = scratch(c, 6, t * kv_max * 4, st);
    let v_b = scratch(c, 7, t * kv_max * 4, st);
    let a_b = scratch(c, 8, t * wide * 4, st);
    let g_b = scratch(c, 9, t * 4, st);
    let staging = scratch(c, 10, t * hs * 4, U::MAP_READ | U::COPY_DST);
    c.queue.write_buffer(&h_b, 0, bytemuck::cast_slice(rows));
    c.queue.write_buffer(&g_b, 0, bytemuck::cast_slice(group));

    let stor = |data: &[f32]| const_buf(c, bytemuck::cast_slice(data));
    let dummy = stor(&[0f32; 4]);
    let unif = |words: &[u32]| {
        c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(words),
            usage: U::UNIFORM,
        })
    };
    let bg = |l: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]| {
        let e: Vec<_> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
        c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: l,
            entries: &e,
        })
    };
    let go = |enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, b: &wgpu::BindGroup, x: u32, y: u32| {
        let mut pass = begin_pass(enc);
        pass.set_pipeline(p);
        pass.set_bind_group(0, b, &[]);
        pass.dispatch_workgroups(x, y, 1);
    };
    let rms_u = unif(&[hs as u32, d.gemma as u32, d.eps.to_bits(), 0]);
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("d1-set"),
    });
    let _merge_guard = PassMergeGuard::new(&enc);
    let rms = |enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, w: &[f32], o: &wgpu::Buffer| {
        go(enc, &c.rmsnorm_b, &bg(&c.layout_rmsnorm_b, &[x, &stor(w), o, &rms_u]), t as u32, 1);
    };
    let norm_add = |enc: &mut wgpu::CommandEncoder, dl: &wgpu::Buffer, w: Option<&[f32]>| {
        let wb = w.map(stor).unwrap_or_else(|| dummy.clone());
        let u = unif(&[hs as u32, d.gemma as u32, d.eps.to_bits(), w.is_some() as u32]);
        go(enc, &p.norm_add.p, &bg(&p.norm_add.l, &[&h_b, dl, &wb, &u]), t as u32, 1);
    };
    for (l, pr) in layers.iter().zip(&preps) {
        rms(&mut enc, &h_b, l.input_norm, &n_b);
        match &l.mixer {
            SetMixer::Conv { taps, kernel, .. } => {
                mm(c, &mut enc, &pr.mix[0], &n_b, &x_b, t);
                let u = unif(&[hs as u32, *kernel as u32, t as u32, 0]);
                go(
                    &mut enc,
                    &p.sconv.p,
                    &bg(&p.sconv.l, &[&x_b, &stor(*taps), &a_b, &u]),
                    (hs as u32).div_ceil(256),
                    1,
                );
                mm(c, &mut enc, &pr.mix[1], &a_b, &d_b, t);
            }
            SetMixer::Attn {
                q_norm,
                k_norm,
                head_gate,
                inv_freq,
                rd,
                nkv,
                window,
                ..
            } => {
                mm(c, &mut enc, &pr.mix[0], &n_b, &x_b, t);
                mm(c, &mut enc, &pr.mix[1], &n_b, &k_b, t);
                mm(c, &mut enc, &pr.mix[2], &n_b, &v_b, t);
                let invf = stor(&inv_freq[..(rd / 2).max(1)]);
                for (buf, norm, heads) in [(&x_b, q_norm, d.nh), (&k_b, k_norm, *nkv)] {
                    let wb = norm.map(stor).unwrap_or_else(|| dummy.clone());
                    let u = unif(&[
                        heads as u32,
                        d.hd as u32,
                        *rd as u32,
                        norm.is_some() as u32,
                        d.gemma as u32,
                        d.eps.to_bits(),
                        0,
                        0,
                    ]);
                    go(
                        &mut enc,
                        &p.qk_rope.p,
                        &bg(&p.qk_rope.l, &[buf, &wb, &invf, &u]),
                        heads as u32,
                        t as u32,
                    );
                }
                let u = unif(&[
                    t as u32,
                    d.nh as u32,
                    *nkv as u32,
                    d.hd as u32,
                    state as u32,
                    window.unwrap_or(0) as u32,
                    d.scale.to_bits(),
                    0,
                ]);
                go(
                    &mut enc,
                    &p.attn.p,
                    &bg(&p.attn.l, &[&x_b, &k_b, &v_b, &a_b, &g_b, &u]),
                    t as u32,
                    d.nh as u32,
                );
                if let Some(gw) = head_gate {
                    let u = unif(&[d.nh as u32, d.hd as u32, hs as u32, 0]);
                    go(
                        &mut enc,
                        &p.head_gate.p,
                        &bg(&p.head_gate.l, &[&a_b, &n_b, &stor(*gw), &u]),
                        d.nh as u32,
                        t as u32,
                    );
                }
                mm(c, &mut enc, &pr.mix[3], &a_b, &d_b, t);
            }
        }
        norm_add(&mut enc, &d_b, l.attn_out_norm);
        rms(&mut enc, &h_b, l.post_norm, &n_b);
        let inter = pr.ffn[0].rows;
        mm(c, &mut enc, &pr.ffn[0], &n_b, &x_b, t);
        mm(c, &mut enc, &pr.ffn[1], &n_b, &u_b, t);
        let n = (t * inter) as u32;
        let groups = n.div_ceil(256).min(4096);
        let u = unif(&[n, groups * 256, 0, 0]);
        go(&mut enc, &p.silu_mul.p, &bg(&p.silu_mul.l, &[&x_b, &u_b, &a_b, &u]), groups, 1);
        mm(c, &mut enc, &pr.ffn[2], &a_b, &d_b, t);
        norm_add(&mut enc, &d_b, l.ffn_out_norm);
    }
    rms(&mut enc, &h_b, final_norm, &n_b);
    let mut out = vec![0f32; t * hs];
    let Some((w, idx)) = &head else {
        return readback(c, enc, &n_b, &staging, (t * hs * 4) as u64, &mut out)
            .then_some((out, Vec::new()));
    };
    let nb = idx.len();
    let ib = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("d1-set-idx"),
        contents: bytemuck::cast_slice(idx),
        usage: U::STORAGE,
    });
    let u = unif(&[hs as u32, 0, 0, 0]);
    go(
        &mut enc,
        &p.gather.p,
        &bg(&p.gather.l, &[&n_b, &ib, &d_b, &u]),
        (hs as u32).div_ceil(256),
        nb as u32,
    );
    let lb = scratch(c, 11, nb * w.rows * 4, st);
    mm(c, &mut enc, w, &d_b, &lb, nb);
    let mut logits = vec![0f32; nb * w.rows];
    readback2(c, enc, (&n_b, &mut out), (&lb, &mut logits)).then_some((out, logits))
}
