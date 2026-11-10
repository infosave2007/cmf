//! The DFlash 2 block on the wgpu device as ONE submission.
//!
//! The host block (`dflash::Drafter::block`) ran a matrix a submission and
//! the rest — the kernel projections, the row convolutions, the norms, the
//! attention over the context — on the host: 15 device round trips and
//! 72–78 ms a block on an RTX PRO 6000 for 2.3 GB of weights that stream in
//! 1.3 ms. Here every stage is a dispatch in one command encoder, the
//! context's keys and values live on the device (appended as rows settle),
//! the target's head and the selector's projection run in the same encoder,
//! and one readback brings the logits of the draft rows and the selector's
//! rows back. The arithmetic is the host block's in f32 (the attention and
//! the norms the same way, order aside); the matrices take the verify's
//! kernels: int8 activations on dp4a for the four-bit ones
//! (`q4tp_matvec4_bk8h`), the eight-row q8_2f kernel for fc, the kernel
//! projections, the selector and the head. `CMF_DFLASH_DEV=0` keeps the
//! block on the host.

use super::{
    Ctx, begin_pass, bind_buf, bind_buf_off, encode_act_absmax, encode_q4_tile_mm_full, finish_enc, flush_pass,
    layer_of_name, mv_grid, q4tp_mv_params, submit, weight_buffer_l, MAX_WG,
};
use cortiq_core::{CmfModel, TensorDtype};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SRC: &str = r#"
// ── rms norm, one workgroup a row ─────────────────────────────────────────
struct RP { n: u32, rows: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<storage, read> rx: array<f32>;
@group(0) @binding(1) var<storage, read> rw: array<f32>;
@group(0) @binding(2) var<storage, read_write> ry: array<f32>;
@group(0) @binding(3) var<uniform> rp: RP;
var<workgroup> rsh: array<f32, 256>;
@compute @workgroup_size(256)
fn dflash_rms(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let n = rp.n;
    let base = wid.x * n;
    var s = 0.0;
    for (var i = lid; i < n; i += 256u) { let v = rx[base + i]; s += v * v; }
    rsh[lid] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st >>= 1u) {
        if (lid < st) { rsh[lid] += rsh[lid + st]; }
        workgroupBarrier();
    }
    let sc = inverseSqrt(rsh[0] / f32(n) + rp.eps);
    for (var i = lid; i < n; i += 256u) { ry[base + i] = rx[base + i] * sc * rw[i]; }
}

// ── grouped causal row convolution: y[r] (+)= Σ_o (base[st][o] + d[r][st][o][group]) ⊙ x[r − o]
struct CP { hs: u32, kn: u32, groups: u32, gs: u32, stage: u32, rows: u32, add: u32, pr: u32 }
@group(0) @binding(0) var<storage, read> cx: array<f32>;
@group(0) @binding(1) var<storage, read> cd: array<f32>;
@group(0) @binding(2) var<storage, read> cb: array<f32>;
@group(0) @binding(3) var<storage, read_write> cy: array<f32>;
@group(0) @binding(4) var<uniform> cp: CP;
@compute @workgroup_size(256)
fn dflash_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let r = gid.y;
    if (i >= cp.hs || r >= cp.rows) { return; }
    let g = i / cp.gs;
    var acc = 0.0;
    let omax = min(cp.kn, r + 1u);
    for (var o = 0u; o < omax; o++) {
        let so = cp.stage * cp.kn + o;
        let k = cb[so * cp.hs + i] + cd[r * cp.pr + so * cp.groups + g];
        acc += k * cx[(r - o) * cp.hs + i];
    }
    let idx = r * cp.hs + i;
    if (cp.add == 1u) { cy[idx] = cy[idx] + acc; } else { cy[idx] = acc; }
}

// ── q and k: rms over a head, then the rotation of the row's position ────
struct QP { rows: u32, nh: u32, kv: u32, hd: u32, pos0: u32, rotary: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<storage, read_write> pq: array<f32>;
@group(0) @binding(1) var<storage, read_write> pk: array<f32>;
@group(0) @binding(2) var<storage, read> pqw: array<f32>;
@group(0) @binding(3) var<storage, read> pkw: array<f32>;
@group(0) @binding(4) var<storage, read> pinv: array<f32>;
@group(0) @binding(5) var<uniform> pp: QP;
var<workgroup> psh: array<f32, 128>;
var<workgroup> pv: array<f32, 128>;
@compute @workgroup_size(128)
fn dflash_qk(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let r = wid.y;
    let hd = pp.hd;
    let isq = h < pp.nh;
    var base = 0u;
    if (isq) { base = (r * pp.nh + h) * hd; } else { base = (r * pp.kv + (h - pp.nh)) * hd; }
    var v = 0.0;
    var w = 0.0;
    if (lid < hd) {
        if (isq) { v = pq[base + lid]; w = pqw[lid]; } else { v = pk[base + lid]; w = pkw[lid]; }
    }
    psh[lid] = v * v;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st >>= 1u) {
        if (lid < st) { psh[lid] += psh[lid + st]; }
        workgroupBarrier();
    }
    let sc = inverseSqrt(psh[0] / f32(hd) + pp.eps);
    let nv = v * sc * w;
    pv[lid] = nv;
    workgroupBarrier();
    if (lid >= hd) { return; }
    var out = nv;
    if (pp.rotary == 1u) {
        let half = hd / 2u;
        let p = f32(pp.pos0 + r);
        if (lid < half) {
            let ang = p * pinv[lid];
            out = pv[lid] * cos(ang) - pv[lid + half] * sin(ang);
        } else {
            let i = lid - half;
            let ang = p * pinv[i];
            out = pv[lid] * cos(ang) + pv[i] * sin(ang);
        }
    }
    if (isq) { pq[base + lid] = out; } else { pk[base + lid] = out; }
}

// ── attention of one (kv head, block row, slice of the group's q heads):
//    the context's keys from `lo`, `n` of them, then the block's own rows
//    (not causal); a row further on sees less far back (window). Scores to
//    a scratch, softmax, then the values. One workgroup per (kv head, row,
//    head slice): `nz` slices of `rep / nz` heads each, so a 16-row block of
//    eight kv heads fills the card with 256 workgroups rather than 128.
struct AP {
    rows: u32, nh: u32, kv: u32, hd: u32,
    n: u32, lo: u32, pos0: u32, window: u32,
    first: u32, span: u32, rep: u32, nz: u32,
    scale: f32, pad1: u32, pad2: u32, pad3: u32,
}
@group(0) @binding(0) var<storage, read> aq: array<f32>;
@group(0) @binding(1) var<storage, read> akb: array<f32>;
@group(0) @binding(2) var<storage, read> avb: array<f32>;
@group(0) @binding(3) var<storage, read> akc: array<f32>;
@group(0) @binding(4) var<storage, read> avc: array<f32>;
@group(0) @binding(5) var<storage, read_write> asc: array<f32>;
@group(0) @binding(6) var<storage, read_write> ay: array<f32>;
@group(0) @binding(7) var<uniform> ap: AP;
// q of this slice's heads, head-fastest: lanes of neighbouring heads read
// neighbouring words
var<workgroup> qsh: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;
var<workgroup> stat: array<f32, 32>;
@compute @workgroup_size(256)
fn dflash_attn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let gi = wid.x;
    let r = wid.y;
    let hd = ap.hd;
    let rep = ap.rep;
    let repw = rep / ap.nz;
    let hoff = wid.z * repw;
    let n = ap.n;
    let span = ap.span;
    for (var i = lid; i < repw * hd; i += 256u) {
        let h = i / hd;
        let d = i - h * hd;
        qsh[d * repw + h] = aq[(r * ap.nh + gi * rep + hoff + h) * hd + d];
    }
    workgroupBarrier();
    let sbase = (gi * ap.rows + r) * rep * span + hoff * span;
    var lo_pos = 0u;
    if (ap.window > 0u) {
        let a = ap.pos0 + r + 1u;
        if (a > ap.window) { lo_pos = a - ap.window; }
    }
    // scores: a lane per (key, head), heads fastest so a key's row is read
    // once for the slice
    let total = repw * span;
    for (var t = lid; t < total; t += 256u) {
        let j = t / repw;
        let h = t - j * repw;
        let ctxk = j < n;
        var acc = 0.0;
        if (ctxk) {
            let kbase = ((ap.lo + j) * ap.kv + gi) * hd;
            for (var d = 0u; d < hd; d++) { acc += qsh[d * repw + h] * akc[kbase + d]; }
        } else {
            let kbase = ((j - n) * ap.kv + gi) * hd;
            for (var d = 0u; d < hd; d++) { acc += qsh[d * repw + h] * akb[kbase + d]; }
        }
        var s = acc * ap.scale;
        if (ctxk && ap.window > 0u && (ap.first + ap.lo + j) < lo_pos) { s = -1e30; }
        asc[sbase + h * span + j] = s;
    }
    workgroupBarrier();
    storageBarrier();
    // softmax of each head: the max, then exp(s - max) written back and
    // its sum
    for (var h = 0u; h < repw; h++) {
        let hb = sbase + h * span;
        var m = -1e30;
        for (var j = lid; j < span; j += 256u) { m = max(m, asc[hb + j]); }
        red[lid] = m;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st >>= 1u) {
            if (lid < st) { red[lid] = max(red[lid], red[lid + st]); }
            workgroupBarrier();
        }
        let mx = red[0];
        workgroupBarrier();
        var sum = 0.0;
        for (var j = lid; j < span; j += 256u) {
            let e = exp(asc[hb + j] - mx);
            asc[hb + j] = e;
            sum += e;
        }
        red[lid] = sum;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st >>= 1u) {
            if (lid < st) { red[lid] += red[lid + st]; }
            workgroupBarrier();
        }
        if (lid == 0u) { stat[h] = 1.0 / red[0]; }
        workgroupBarrier();
    }
    storageBarrier();
    // the values: a lane owns one dimension and a slice of the heads, one
    // head at a time (no indexed register array)
    let ngroups = 256u / hd;
    let d = lid % hd;
    let hg = lid / hd;
    let hpg = (repw + ngroups - 1u) / ngroups;
    let h0 = hg * hpg;
    let h1 = min(repw, h0 + hpg);
    for (var h = h0; h < h1; h++) {
        let hb = sbase + h * span;
        var o = 0.0;
        for (var j = 0u; j < n; j++) { o += asc[hb + j] * avc[((ap.lo + j) * ap.kv + gi) * hd + d]; }
        for (var j = n; j < span; j++) { o += asc[hb + j] * avb[((j - n) * ap.kv + gi) * hd + d]; }
        ay[(r * ap.nh + gi * rep + hoff + h) * hd + d] = o * stat[h];
    }
}

// ── silu(g) ⊙ u ───────────────────────────────────────────────────────────
struct SP { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> sg: array<f32>;
@group(0) @binding(1) var<storage, read> su: array<f32>;
@group(0) @binding(2) var<storage, read_write> sa: array<f32>;
@group(0) @binding(3) var<uniform> sp: SP;
@compute @workgroup_size(256)
fn dflash_silu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= sp.n) { return; }
    let g = sg[i];
    sa[i] = g / (1.0 + exp(-g)) * su[i];
}
"#;

/// One layer of the block as the device needs it.
pub struct LayerSpec {
    pub in_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// `[stage][offset][hidden]` of the attention and the MLP convolution.
    pub attn_base: Vec<f32>,
    pub mlp_base: Vec<f32>,
    pub window: usize,
    pub rotary: bool,
    /// Directory indices in the sidecar: q, k, v, o, gate, up, down, the
    /// attention conv's kernel projection, the MLP conv's.
    pub q: usize,
    pub k: usize,
    pub v: usize,
    pub o: usize,
    pub gate: usize,
    pub up: usize,
    pub down: usize,
    pub attn_proj: usize,
    pub mlp_proj: usize,
}

pub struct Spec {
    pub hs: usize,
    pub ffn: usize,
    pub hd: usize,
    pub nh: usize,
    pub kv: usize,
    pub kn: usize,
    pub gs: usize,
    pub block: usize,
    pub eps: f32,
    pub layers: Vec<LayerSpec>,
    pub norm: Vec<f32>,
    pub inv_freq: Vec<f32>,
    pub sel: usize,
    pub sel_rank: usize,
}

struct Pipes {
    rms: wgpu::ComputePipeline,
    conv: wgpu::ComputePipeline,
    qk: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
}

struct LayerBufs {
    in_norm: wgpu::Buffer,
    post_norm: wgpu::Buffer,
    q_norm: wgpu::Buffer,
    k_norm: wgpu::Buffer,
    attn_base: wgpu::Buffer,
    mlp_base: wgpu::Buffer,
    /// The context's keys and values, `[position − first][kv][hd]`.
    ctx_k: wgpu::Buffer,
    ctx_v: wgpu::Buffer,
    cap: usize,
}

/// The block's device state: pipelines, the sidecar's matrices resident on
/// the card, the context per layer, the activations of one block.
pub struct Dev {
    model: Arc<CmfModel>,
    spec: Spec,
    pipes: Pipes,
    layers: Vec<LayerBufs>,
    norm: wgpu::Buffer,
    inv_freq: wgpu::Buffer,
    /// Resident matrices by directory index.
    weights: std::collections::HashMap<usize, wgpu::Buffer>,
    h: wgpu::Buffer,
    x: wgpu::Buffer,
    d: wgpu::Buffer,
    xc: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    y: wgpu::Buffer,
    o: wgpu::Buffer,
    g: wgpu::Buffer,
    u: wgpu::Buffer,
    a: wgpu::Buffer,
    scores: wgpu::Buffer,
    scores_cap: usize,
    /// the activation scale of the cooperative GEMM (one f32)
    asc: wgpu::Buffer,
    /// rows of the block being encoded (the geometry's block, or fewer
    /// under `CMF_DFLASH_DEV_ROWS`)
    rows: usize,
    logits: wgpu::Buffer,
    logits_cap: usize,
    hp: wgpu::Buffer,
    stage: wgpu::Buffer,
    stage_cap: u64,
}

/// What a block of the device returns: the logits of the draft rows under
/// the head (`rows − 1` rows of `width`) and the selector's projection of
/// the same rows (`rows − 1` rows of `sel_rank`).
pub struct Out {
    pub logits: Vec<f32>,
    pub hp: Vec<f32>,
}

/// The context range a layer's block reads: `lo` as an index into the
/// device context (position − first), `n` rows.
pub struct CtxRange {
    pub lo: usize,
    pub n: usize,
}

fn storage(c: &Ctx, label: &str, bytes: u64) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn storage_f32(c: &Ctx, label: &str, v: &[f32]) -> wgpu::Buffer {
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(v),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn unif(c: &Ctx, words: &[u32]) -> wgpu::Buffer {
    let mut w = words.to_vec();
    while w.len() % 4 != 0 {
        w.push(0);
    }
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dflash-u"),
        contents: bytemuck::cast_slice(&w),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn bg(c: &Ctx, p: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry<'_>]) -> wgpu::BindGroup {
    c.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries })
}

/// The block's matrix kernel for the q4tp projections (`CMF_DFLASH_DEV_MM`).
fn mm_mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("CMF_DFLASH_DEV_MM").as_deref() {
        Ok("coop") => 2,
        Ok("r") => 1,
        _ => 0,
    })
}

/// `CMF_DFLASH_DEV_ROWS`: rows of the block to run (see `Dev::block`).
fn dev_rows() -> Option<usize> {
    static R: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("CMF_DFLASH_DEV_ROWS").ok().and_then(|v| v.parse().ok()).filter(|&n| n >= 2))
}

/// Whether the device block is wanted (`CMF_DFLASH_DEV=0` keeps the host's).
pub fn wanted() -> bool {
    std::env::var("CMF_DFLASH_DEV").as_deref() != Ok("0")
}

impl Dev {
    /// Build the block's device state for a sidecar, or say why not.
    pub fn new(model: Arc<CmfModel>, spec: Spec) -> Result<Self, String> {
        let c = super::ctx().ok_or("no wgpu context")?;
        if spec.hd > 128 || spec.hd % 4 != 0 || spec.nh % spec.kv != 0 || spec.nh / spec.kv > 16 || spec.hs % 64 != 0 {
            return Err(format!("geometry (hd {}, heads {}/{}, hidden {}) outside the kernels' range", spec.hd, spec.nh, spec.kv, spec.hs));
        }
        if c.graph_x.is_none() {
            return Err("the graph's eight-bit kernels are not built".into());
        }
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dflash-dev"),
            source: wgpu::ShaderSource::Wgsl(SRC.into()),
        });
        let pipe = |entry: &str| {
            c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &m,
                entry_point: Some(entry),
                compilation_options: super::pco(),
                cache: None,
            })
        };
        let pipes = Pipes {
            rms: pipe("dflash_rms"),
            conv: pipe("dflash_conv"),
            qk: pipe("dflash_qk"),
            attn: pipe("dflash_attn"),
            silu: pipe("dflash_silu"),
        };
        if let Some(e) = pollster::block_on(sc.pop()) {
            return Err(format!("kernels rejected: {e}"));
        }
        let b = spec.block;
        let (hs, ffn, hd, nh, kv) = (spec.hs, spec.ffn, spec.hd, spec.nh, spec.kv);
        let groups = hs / spec.gs;
        let pr = 2 * spec.kn * groups;
        let f = |n: usize| (n * 4) as u64;
        let layers = spec
            .layers
            .iter()
            .map(|l| LayerBufs {
                in_norm: storage_f32(c, "dflash-in-norm", &l.in_norm),
                post_norm: storage_f32(c, "dflash-post-norm", &l.post_norm),
                q_norm: storage_f32(c, "dflash-q-norm", &l.q_norm),
                k_norm: storage_f32(c, "dflash-k-norm", &l.k_norm),
                attn_base: storage_f32(c, "dflash-attn-base", &l.attn_base),
                mlp_base: storage_f32(c, "dflash-mlp-base", &l.mlp_base),
                ctx_k: storage(c, "dflash-ctx-k", 16),
                ctx_v: storage(c, "dflash-ctx-v", 16),
                cap: 0,
            })
            .collect();
        let mut dev = Self {
            norm: storage_f32(c, "dflash-norm", &spec.norm),
            inv_freq: storage_f32(c, "dflash-invf", &spec.inv_freq),
            weights: std::collections::HashMap::new(),
            h: storage(c, "dflash-h", f(b * hs)),
            x: storage(c, "dflash-x", f(b * hs)),
            d: storage(c, "dflash-d", f(b * pr)),
            xc: storage(c, "dflash-xc", f(b * hs)),
            q: storage(c, "dflash-q", f(b * nh * hd)),
            k: storage(c, "dflash-k", f(b * kv * hd)),
            v: storage(c, "dflash-v", f(b * kv * hd)),
            y: storage(c, "dflash-y", f(b * nh * hd)),
            o: storage(c, "dflash-o", f(b * hs)),
            g: storage(c, "dflash-g", f(b * ffn)),
            u: storage(c, "dflash-u", f(b * ffn)),
            a: storage(c, "dflash-a", f(b * ffn)),
            scores: storage(c, "dflash-scores", 16),
            scores_cap: 0,
            asc: storage(c, "dflash-asc", 4),
            rows: spec.block,
            logits: storage(c, "dflash-logits", 16),
            logits_cap: 0,
            hp: storage(c, "dflash-hp", f(b * spec.sel_rank)),
            stage: c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dflash-stage"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stage_cap: 16,
            layers,
            pipes,
            spec,
            model,
        };
        // every matrix resident before the first block
        let mut idx: Vec<usize> = vec![dev.spec.sel];
        for l in &dev.spec.layers {
            idx.extend([l.q, l.k, l.v, l.o, l.gate, l.up, l.down, l.attn_proj, l.mlp_proj]);
        }
        for i in idx {
            dev.weight(c, &dev.model.clone(), i).ok_or_else(|| format!("tensor {i} of the sidecar did not reach the device"))?;
        }
        // The uploads above are staged; without this they ride the first
        // block's submission (measured 480 ms for a 2.3 GB sidecar) and
        // the first round pays for them.
        c.queue.submit(std::iter::empty());
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(dev)
    }

    /// A matrix of `model` on the device, cached here.
    fn weight(&mut self, c: &Ctx, model: &Arc<CmfModel>, idx: usize) -> Option<wgpu::Buffer> {
        let own = Arc::ptr_eq(model, &self.model);
        if own && let Some(b) = self.weights.get(&idx) {
            return Some(b.clone());
        }
        let e = &model.tensors[idx];
        let bytes = model.primary_bytes();
        let abs = model.entry_abs_offset(e)?;
        let plen = e.nbytes as usize;
        if abs + plen > bytes.len() {
            return None;
        }
        let b = weight_buffer_l(c, (model.uid() as usize, idx), &bytes[abs..abs + plen], layer_of_name(&e.name))?;
        if own {
            self.weights.insert(idx, b.clone());
        }
        Some(b)
    }

    /// Append the context rows at device index `at` (position − first):
    /// `k` and `v` are `n` rows of `[kv][hd]` each, normed and rotated as
    /// the block reads them. The buffer grows when it has to; the host
    /// re-sends what it holds then (`refill`).
    pub fn push_rows(&mut self, layer: usize, at: usize, k: &[f32], v: &[f32]) -> bool {
        let Some(c) = super::ctx() else { return false };
        let row = self.spec.kv * self.spec.hd;
        let n = k.len() / row;
        if v.len() != k.len() || k.len() % row != 0 {
            return false;
        }
        if at + n > self.layers[layer].cap {
            return false;
        }
        let off = (at * row * 4) as u64;
        c.queue.write_buffer(&self.layers[layer].ctx_k, off, bytemuck::cast_slice(k));
        c.queue.write_buffer(&self.layers[layer].ctx_v, off, bytemuck::cast_slice(v));
        true
    }

    /// Room for `rows` context rows of a layer. Growing loses what is there:
    /// the caller refills from its own copy.
    pub fn reserve(&mut self, layer: usize, rows: usize) -> bool {
        if rows <= self.layers[layer].cap {
            return true;
        }
        let Some(c) = super::ctx() else { return false };
        let cap = rows.max(2 * self.layers[layer].cap).max(8192);
        let bytes = (cap * self.spec.kv * self.spec.hd * 4) as u64;
        self.layers[layer].ctx_k = storage(c, "dflash-ctx-k", bytes);
        self.layers[layer].ctx_v = storage(c, "dflash-ctx-v", bytes);
        self.layers[layer].cap = cap;
        false
    }

    pub fn capacity(&self, layer: usize) -> usize {
        self.layers[layer].cap
    }

    fn q4(&mut self, c: &Ctx, enc: &mut wgpu::CommandEncoder, idx: usize, xs: &wgpu::Buffer, y: &wgpu::Buffer, rows: usize, cols: usize) -> bool {
        let model = self.model.clone();
        self.q4_of(c, enc, &model, idx, xs, y, rows, cols)
    }

    /// A four-bit matrix over the block's rows: the verify's int8 kernel,
    /// eight rows a dispatch.
    #[allow(clippy::too_many_arguments)]
    fn q4_of(
        &mut self,
        c: &Ctx,
        enc: &mut wgpu::CommandEncoder,
        model: &Arc<CmfModel>,
        idx: usize,
        xs: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        cols: usize,
    ) -> bool {
        let Some(w) = self.weight(c, model, idx) else { return false };
        let b = self.rows;
        let gpr = cols / 32;
        if cols % 32 != 0 {
            return false;
        }
        // CMF_DFLASH_DEV_MM: `i8` (default) runs the eight-row int8 matvec
        // in chunks, so a 16-row block reads every weight twice; `coop`
        // puts the whole block on the cooperative-matrix GEMM (f16
        // operands, the weight read once); `r` on the exact 64x64 tile
        // GEMM. Measured per stage with CMF_DFLASH_DEV_TS=1.
        match mm_mode() {
            2 if c.q4tp_mm_coop_s.is_some() => {
                if encode_act_absmax(c, enc, xs, b * cols, &self.asc) {
                    encode_q4_tile_mm_full(c, enc, c.q4tp_mm_coop_s.as_ref().unwrap(), &w, xs, y, rows, cols, b, 0.0, Some(&self.asc));
                    return true;
                }
            }
            1 => {
                if super::dense_batch::encode_q4tp_mm_r(c, enc, &w, xs, y, rows, cols, b) {
                    return true;
                }
            }
            _ => {}
        }
        // Narrow matrices (a 0.3B drafter's 896 columns) are below the int8
        // kernel's split: the exact 64x64 tile GEMM, the weight read once.
        if gpr <= 64 {
            return super::dense_batch::encode_q4tp_mm_r(c, enc, &w, xs, y, rows, cols, b);
        }
        let mut done = 0;
        while done < b {
            let n = (b - done).min(8);
            if n < 2 {
                return false;
            }
            let need = (n * gpr) as u64;
            let (x8, xsb) = {
                let mut g = c.i8x.lock().unwrap();
                let grow = match g.as_ref() {
                    Some((_, _, cap)) => *cap < need,
                    None => true,
                };
                if grow {
                    let cap = need.max(8 * 1024);
                    let x8 = c.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("i8-x8"),
                        size: cap * 32,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    let xsb = c.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("i8-xs"),
                        size: cap * 8,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    });
                    *g = Some((x8, xsb, cap));
                }
                let (a, b, _) = g.as_ref().unwrap();
                (a.clone(), b.clone())
            };
            let x_off = (done * cols * 4) as u64;
            let y_off = (done * rows * 4) as u64;
            let qp = unif(c, &[gpr as u32, n as u32, 0, 0]);
            let qbind = bg(
                c,
                &c.x_quant_i8,
                &[
                    bind_buf(3, &qp),
                    bind_buf_off(5, xs, x_off, (n * cols * 4) as u64),
                    bind_buf(11, &x8),
                    bind_buf(12, &xsb),
                ],
            );
            let p_buf = q4tp_mv_params(c, gpr, rows, n);
            let pipe8 = &c.q4tp_mv4_bk8h[n];
            let bind = bg(
                c,
                pipe8,
                &[
                    bind_buf(0, &w),
                    bind_buf_off(2, y, y_off, ((b - done) * rows * 4) as u64),
                    bind_buf(3, &p_buf),
                    bind_buf(4, &w),
                    bind_buf(9, &x8),
                    bind_buf(10, &xsb),
                ],
            );
            let mut pass = begin_pass(enc);
            pass.set_pipeline(&c.x_quant_i8);
            pass.set_bind_group(0, &qbind, &[]);
            pass.dispatch_workgroups(((n * gpr) as u32).div_ceil(256), 1, 1);
            pass.set_pipeline(pipe8);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(mv_grid((rows as u32).div_ceil(8)), 1, 1);
            done += n;
        }
        true
    }

    /// An eight-bit (q8_2f) matrix over `nb` rows of `xs` (from `x_off`
    /// floats) into `y`: the graph's eight-row kernel.
    #[allow(clippy::too_many_arguments)]
    fn q8(
        &mut self,
        c: &Ctx,
        enc: &mut wgpu::CommandEncoder,
        model: &Arc<CmfModel>,
        idx: usize,
        xs: &wgpu::Buffer,
        x_off: usize,
        y: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        nb: usize,
    ) -> bool {
        let Some(w) = self.weight(c, model, idx) else { return false };
        let Some(gx) = c.graph_x.as_ref() else { return false };
        if cols % 16 != 0 || nb == 0 {
            return false;
        }
        let p = unif(c, &[(cols / 4) as u32, rows as u32, cols as u32, nb as u32]);
        let xb = if x_off == 0 {
            bind_buf(1, xs)
        } else {
            bind_buf_off(1, xs, (x_off * 4) as u64, (nb * cols * 4) as u64)
        };
        // the decode kernel batched where it applies (each row the decode's to the bit)
        let (pipe, layout) = gx.q82_unr(c.use_q82_mv4k, cols, nb).unwrap_or((&gx.q82_b, &gx.q82_l));
        let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[bind_buf(0, &w), xb, bind_buf(2, y), bind_buf(3, &p)],
        });
        let mut pass = begin_pass(enc);
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rows as u32).div_ceil(4).min(MAX_WG), (nb as u32).div_ceil(8), 1);
        true
    }

    fn rms(&self, c: &Ctx, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, w: &wgpu::Buffer, y: &wgpu::Buffer) {
        let p = unif(c, &[self.spec.hs as u32, self.rows as u32, self.spec.eps.to_bits(), 0]);
        let bind = bg(c, &self.pipes.rms, &[bind_buf(0, x), bind_buf(1, w), bind_buf(2, y), bind_buf(3, &p)]);
        let mut pass = begin_pass(enc);
        pass.set_pipeline(&self.pipes.rms);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(self.rows as u32, 1, 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&self, c: &Ctx, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, base: &wgpu::Buffer, stage: u32, add: bool, y: &wgpu::Buffer) {
        let s = &self.spec;
        let groups = s.hs / s.gs;
        let pr = 2 * s.kn * groups;
        let p = unif(
            c,
            &[s.hs as u32, s.kn as u32, groups as u32, s.gs as u32, stage, self.rows as u32, u32::from(add), pr as u32],
        );
        let bind = bg(c, &self.pipes.conv, &[bind_buf(0, x), bind_buf(1, &self.d), bind_buf(2, base), bind_buf(3, y), bind_buf(4, &p)]);
        let mut pass = begin_pass(enc);
        pass.set_pipeline(&self.pipes.conv);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((s.hs as u32).div_ceil(256), self.rows as u32, 1);
    }

    /// One block: `h` is `block` rows of the embeddings (the anchor, then
    /// MASK rows), at positions `anchor_pos ..`; `ranges[l]` the context each
    /// layer reads, `first` the position of device index 0. The head's
    /// matrix (`model`, index, rows, cols, kind: 7 = q8_2f) scores the
    /// `n_draft` rows after the anchor over its `width` (= all) rows; the
    /// selector's projection runs over the same rows. None: a stage
    /// declined, the host block runs.
    #[allow(clippy::too_many_arguments)]
    pub fn block(
        &mut self,
        anchor_pos: usize,
        first: usize,
        h: &[f32],
        ranges: &[CtxRange],
        head: (&Arc<CmfModel>, usize, usize, usize, u8),
        width: usize,
        n_draft: usize,
    ) -> Option<Out> {
        let c = super::ctx()?;
        let _op = c.op_gate.enter();
        let s = &self.spec;
        let (hs, hd, nh, kv, ffn) = (s.hs, s.hd, s.nh, s.kv, s.ffn);
        let (qd, kd) = (nh * hd, kv * hd);
        let groups = hs / s.gs;
        let pr = 2 * s.kn * groups;
        let rep = nh / kv;
        if h.len() != s.block * hs || ranges.len() != s.layers.len() {
            return None;
        }
        // The block cut to its first rows: the anchor, the draft rows and
        // mask rows up to eight in all (`CMF_DFLASH_DEV_ROWS=N` for another
        // count, 16 = the sidecar's whole block). The sidecar was trained
        // with the whole block, so its later mask rows do shape the earlier
        // ones through the block's attention — but on Ripple-65B eight rows
        // kept exactly the drafts of sixteen (56 of 80 on code, the same
        // rounds) at half the price (14.4 → 8.4 ms), and eight rows is one
        // pass of the int8 matvec where sixteen read every weight twice.
        let b = dev_rows().unwrap_or(8).clamp(n_draft + 1, s.block);
        self.rows = b;
        let h = &h[..b * hs];
        let s = &self.spec;
        let (hmodel, hidx, hrows, hcols, hkind) = head;
        // the eight-row q8_2f kernel finds the scale planes by rows × cols,
        // so the head is scored over its whole table
        if hcols != hs || width != hrows || hkind != 7 || n_draft == 0 || n_draft >= b {
            return None;
        }
        // scratch for the widest attention of this block
        let span_max = ranges.iter().map(|r| r.n + b).max().unwrap_or(b);
        // head slices of the attention workgroups: two when the group
        // splits evenly, so the block fills the card
        let nz = if rep % 2 == 0 && rep >= 4 { 2 } else { 1 };
        let need = nh * b * span_max;
        if need > self.scores_cap {
            self.scores = storage(c, "dflash-scores", (need * 4) as u64);
            self.scores_cap = need;
        }
        if n_draft * hrows > self.logits_cap {
            self.logits = storage(c, "dflash-logits", (n_draft * hrows * 4) as u64);
            self.logits_cap = n_draft * hrows;
        }
        let l_bytes = (n_draft * width * 4) as u64;
        let hp_bytes = (n_draft * s.sel_rank * 4) as u64;
        let hp_off = l_bytes.next_multiple_of(256);
        let total = hp_off + hp_bytes;
        if total > self.stage_cap {
            self.stage = c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dflash-stage"),
                size: total,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.stage_cap = total;
        }
        c.queue.write_buffer(&self.h, 0, bytemuck::cast_slice(h));
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dflash-block") });
        let _merge = super::PassMergeGuard::new(&enc);
        // CMF_DFLASH_DEV_TS=1 (with CMF_GPU_TS=1): the card's clock at every
        // stage boundary of the block, summed over the layers and printed
        // once a block.
        let ts_on = c.ts_query.is_some() && std::env::var("CMF_DFLASH_DEV_TS").as_deref() == Ok("1");
        let mut ts_lbl: Vec<usize> = Vec::new();
        macro_rules! st {
            ($lbl:expr) => {
                if ts_on {
                    if let Some((qs, _, _)) = c.ts_query.as_ref() {
                        if ts_lbl.len() < 200 {
                            flush_pass(&enc);
                            enc.write_timestamp(qs, ts_lbl.len() as u32);
                            ts_lbl.push($lbl);
                        }
                    }
                }
            };
        }
        st!(0);
        let model = self.model.clone();
        let scale = (hd as f32).powf(-0.5);
        let sel_rank = s.sel_rank;
        let eps_bits = s.eps.to_bits();
        let nlayers = s.layers.len();
        for li in 0..nlayers {
            let (window, rotary) = (self.spec.layers[li].window, self.spec.layers[li].rotary);
            let (q_i, k_i, v_i, o_i, g_i, u_i, d_i, ap_i, mp_i) = {
                let l = &self.spec.layers[li];
                (l.q, l.k, l.v, l.o, l.gate, l.up, l.down, l.attn_proj, l.mlp_proj)
            };
            let (in_norm, post_norm, q_norm, k_norm, attn_base, mlp_base, ctx_k, ctx_v) = {
                let lb = &self.layers[li];
                (
                    lb.in_norm.clone(),
                    lb.post_norm.clone(),
                    lb.q_norm.clone(),
                    lb.k_norm.clone(),
                    lb.attn_base.clone(),
                    lb.mlp_base.clone(),
                    lb.ctx_k.clone(),
                    lb.ctx_v.clone(),
                )
            };
            let (x, d, xc, q, k, v, y, o, g, u, a, hb) = (
                self.x.clone(),
                self.d.clone(),
                self.xc.clone(),
                self.q.clone(),
                self.k.clone(),
                self.v.clone(),
                self.y.clone(),
                self.o.clone(),
                self.g.clone(),
                self.u.clone(),
                self.a.clone(),
                self.h.clone(),
            );
            // attention: norm, kernels, conv in, q k v
            self.rms(c, &mut enc, &hb, &in_norm, &x);
            if !self.q8(c, &mut enc, &model, ap_i, &x, 0, &d, pr, hs, b) {
                return None;
            }
            self.conv(c, &mut enc, &x, &attn_base, 0, false, &xc);
            st!(1);
            if !(self.q4(c, &mut enc, q_i, &xc, &q, qd, hs) && self.q4(c, &mut enc, k_i, &xc, &k, kd, hs) && self.q4(c, &mut enc, v_i, &xc, &v, kd, hs)) {
                return None;
            }
            st!(2);
            {
                let p = unif(c, &[b as u32, nh as u32, kv as u32, hd as u32, anchor_pos as u32, u32::from(rotary), eps_bits, 0]);
                let bind = bg(
                    c,
                    &self.pipes.qk,
                    &[bind_buf(0, &q), bind_buf(1, &k), bind_buf(2, &q_norm), bind_buf(3, &k_norm), bind_buf(4, &self.inv_freq), bind_buf(5, &p)],
                );
                let mut pass = begin_pass(&mut enc);
                pass.set_pipeline(&self.pipes.qk);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups((nh + kv) as u32, b as u32, 1);
            }
            st!(3);
            {
                let r = &ranges[li];
                if r.n > 0 && r.lo + r.n > self.layers[li].cap {
                    return None;
                }
                let span = r.n + b;
                let p = unif(
                    c,
                    &[
                        b as u32,
                        nh as u32,
                        kv as u32,
                        hd as u32,
                        r.n as u32,
                        r.lo as u32,
                        anchor_pos as u32,
                        window as u32,
                        first as u32,
                        span as u32,
                        rep as u32,
                        nz as u32,
                        scale.to_bits(),
                        0,
                        0,
                        0,
                    ],
                );
                let bind = bg(
                    c,
                    &self.pipes.attn,
                    &[
                        bind_buf(0, &q),
                        bind_buf(1, &k),
                        bind_buf(2, &v),
                        bind_buf(3, &ctx_k),
                        bind_buf(4, &ctx_v),
                        bind_buf(5, &self.scores),
                        bind_buf(6, &y),
                        bind_buf(7, &p),
                    ],
                );
                let mut pass = begin_pass(&mut enc);
                pass.set_pipeline(&self.pipes.attn);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(kv as u32, b as u32, nz as u32);
            }
            st!(4);
            if !self.q4(c, &mut enc, o_i, &y, &o, hs, qd) {
                return None;
            }
            self.conv(c, &mut enc, &o, &attn_base, 1, true, &hb);
            st!(5);
            // the MLP the same way
            self.rms(c, &mut enc, &hb, &post_norm, &x);
            if !self.q8(c, &mut enc, &model, mp_i, &x, 0, &d, pr, hs, b) {
                return None;
            }
            self.conv(c, &mut enc, &x, &mlp_base, 0, false, &xc);
            st!(6);
            if !(self.q4(c, &mut enc, g_i, &xc, &g, ffn, hs) && self.q4(c, &mut enc, u_i, &xc, &u, ffn, hs)) {
                return None;
            }
            {
                let n = b * ffn;
                let p = unif(c, &[n as u32, 0, 0, 0]);
                let bind = bg(c, &self.pipes.silu, &[bind_buf(0, &g), bind_buf(1, &u), bind_buf(2, &a), bind_buf(3, &p)]);
                let wgs = (n as u32).div_ceil(256);
                let gx = wgs.min(MAX_WG);
                let gy = wgs.div_ceil(gx);
                let mut pass = begin_pass(&mut enc);
                pass.set_pipeline(&self.pipes.silu);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            st!(7);
            if !self.q4(c, &mut enc, d_i, &a, &o, hs, ffn) {
                return None;
            }
            self.conv(c, &mut enc, &o, &mlp_base, 1, true, &hb);
            st!(8);
        }
        // the last norm, the head over the draft rows, the selector's rows
        let (hb, x, norm, logits, hp) = (self.h.clone(), self.x.clone(), self.norm.clone(), self.logits.clone(), self.hp.clone());
        self.rms(c, &mut enc, &hb, &norm, &x);
        let sel = self.spec.sel;
        if !self.q8(c, &mut enc, &model, sel, &x, hs, &hp, sel_rank, hs, n_draft) {
            return None;
        }
        st!(9);
        let hmodel = hmodel.clone();
        if !self.q8(c, &mut enc, &hmodel, hidx, &x, hs, &logits, hrows, hs, n_draft) {
            return None;
        }
        st!(10);
        flush_pass(&enc);
        enc.copy_buffer_to_buffer(&self.logits, 0, &self.stage, 0, l_bytes);
        enc.copy_buffer_to_buffer(&self.hp, 0, &self.stage, hp_off, hp_bytes);
        if ts_on && ts_lbl.len() > 1 {
            if let Some((qs, resolve, tstage)) = c.ts_query.as_ref() {
                let n = ts_lbl.len() as u32;
                enc.resolve_query_set(qs, 0..n, resolve, 0);
                enc.copy_buffer_to_buffer(resolve, 0, tstage, 0, n as u64 * 8);
            }
        }
        drop(_merge);
        submit(c, finish_enc(enc));
        let slice = self.stage.slice(..total);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d2 = done.clone();
        slice.map_async(wgpu::MapMode::Read, move |_| d2.store(true, std::sync::atomic::Ordering::Release));
        if ts_on && ts_lbl.len() > 1 {
            if let Some((_, _, tstage)) = c.ts_query.as_ref() {
                let bytes = ts_lbl.len() as u64 * 8;
                tstage.slice(..bytes).map_async(wgpu::MapMode::Read, |_| {});
            }
        }
        if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            self.stage.unmap();
            return None;
        }
        if ts_on && ts_lbl.len() > 1 {
            if let Some((_, _, tstage)) = c.ts_query.as_ref() {
                let bytes = ts_lbl.len() as u64 * 8;
                if let Ok(raw) = tstage.slice(..bytes).get_mapped_range() {
                    let t: &[u64] = bytemuck::cast_slice(&raw);
                    const NAMES: [&str; 11] = [
                        "start",
                        "norm+proj+conv",
                        "qkv",
                        "rope",
                        "attn",
                        "o+conv",
                        "norm+proj+conv",
                        "gate-up+silu",
                        "down+conv",
                        "norm+sel",
                        "head",
                    ];
                    let mut sum = [0u64; 11];
                    for i in 1..ts_lbl.len() {
                        let d = t[i].saturating_sub(t[i - 1]);
                        sum[ts_lbl[i]] += (d as f64 * c.ts_period as f64) as u64;
                    }
                    let total: u64 = sum.iter().sum();
                    let mut line = format!("dflash-dev-ts: block {:.0} us ({} rows, {} layers)", total as f64 / 1e3, b, nlayers);
                    for (i, name) in NAMES.iter().enumerate().skip(1) {
                        line.push_str(&format!(" | {} {:.0}", name, sum[i] as f64 / 1e3));
                    }
                    eprintln!("{line}");
                    drop(raw);
                }
                tstage.unmap();
            }
        }
        let out = {
            let Ok(data) = slice.get_mapped_range() else {
                self.stage.unmap();
                return None;
            };
            let logits: Vec<f32> = bytemuck::cast_slice(&data[..l_bytes as usize]).to_vec();
            let hp: Vec<f32> = bytemuck::cast_slice(&data[hp_off as usize..(hp_off + hp_bytes) as usize]).to_vec();
            Out { logits, hp }
        };
        self.stage.unmap();
        Some(out)
    }
}

/// The dtype of a directory entry as the block's kernels name it: 6 for
/// four-bit tiled, 7 for q8_2f, None otherwise.
pub fn kind_of(model: &CmfModel, idx: usize) -> Option<u8> {
    match model.tensors.get(idx)?.dtype {
        TensorDtype::Q4TiledP => Some(6),
        TensorDtype::Q8_2f => Some(7),
        _ => None,
    }
}

#[cfg(test)]
mod shader_tests {
    /// The DFlash block module parses and validates without a device,
    /// with every entry point `Dev::new` asks for — the one check of the
    /// WGSL a machine without the card can run (`from` is a reserved
    /// word in WGSL; the driver found that one).
    #[test]
    fn dflash_dev_shader_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(super::SRC).expect("dflash-dev WGSL parses");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("dflash-dev WGSL validates");
        for ep in ["dflash_rms", "dflash_conv", "dflash_qk", "dflash_attn", "dflash_silu"] {
            assert!(module.entry_points.iter().any(|e| e.name == ep), "entry point {ep} missing");
        }
    }
}
