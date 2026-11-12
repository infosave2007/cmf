//! LTX-2.5 over Vulkan: the 48 blocks of a denoising step as one resident
//! chain, on the H3 chain's parts (a child of `h3.rs`).
//!
//! Both residual streams go up once and come back once. Every projection is
//! `zi_mm` reading the q4tp tensor in place on the 16x16 matrix units (f16
//! operands, f32 accumulation); attention is `vit_flash` over a `[q | k | v]`
//! panel (heads of 128 for video, 64 for audio and the fusion), queries and
//! keys of different lengths for the prompt and fusion attentions. Between
//! the products, kernels of their own: the timestep-modulated norm into f16
//! (`lx_ada`), bias, q/k RMS norm over the whole inner width and the split
//! rotation in place on the panel (`lx_qk`), the per-head gate (`lx_gate`),
//! the GELU of the feed-forward (`lx_gelu`) and the gated residual
//! (`lx_addg`).
//!
//! Range guards as in the H3 chain (`Guards`): activations into a product
//! x2^-act, the panel x2^-qkv (the norm is scale-free, the values keep the
//! factor to the output projection), the feed-forward's first product
//! x2^-gu and its middle x2^-hid. A non-finite stream makes the chain stand
//! down and the per-operation path redo the step.
//!
//! The weights follow the H3 plan: on the card in their codec between steps
//! when they fit the adapter's budget (`CMF_GPU_VRAM_MB` sets it), else the
//! first blocks that fit stay and the rest pass through a slot every step.

use super::*;
use crate::gpu::{LtxAttn, LtxChain, LtxLin, LtxSide};

/// q4tp tensors of a block, in the order `spans` lists them: per stream the
/// four projections of self-attention, of prompt attention and the two of
/// the feed-forward, then the four of each fusion direction.
const PER: usize = 28;
const NONE: u32 = u32::MAX;
/// The widest row `lx_qk` keeps in workgroup memory.
const QK_MAX: usize = 4096;

/// `out = f16(((x · inv) · (1 + scale) + shift) · oscale)` a row: `inv` the
/// RMS norm's under flag 1 (else 1), scale and shift under flag 2 from the
/// table at `tab` (row `ri[ri_off + row]` under flag 4, else row 0, rows
/// `stride` apart; chunks at `sh` and `sc`). One workgroup a row.
const ADA_SRC: &str = r#"
enable f16;
struct AP { w: u32, stride: u32, sh: u32, sc: u32, flags: u32, tab: u32, ri_off: u32, eps: f32, oscale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<storage, read> ri: array<u32>;
@group(0) @binding(4) var<uniform> p: AP;
var<workgroup> red: array<f32, 256>;

fn wsum(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn lx_ada(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let base = row * p.w;
    var inv = 1.0;
    if ((p.flags & 1u) != 0u) {
        var ss = 0.0;
        for (var i = lid; i < p.w; i = i + 256u) {
            let v = x[base + i];
            ss = ss + v * v;
        }
        inv = inverseSqrt(wsum(ss, lid) / f32(p.w) + p.eps);
    }
    let modded = (p.flags & 2u) != 0u;
    var t = p.tab;
    if (modded && (p.flags & 4u) != 0u) { t = t + ri[p.ri_off + row] * p.stride; }
    for (var i = lid; i < p.w; i = i + 256u) {
        var v = x[base + i] * inv;
        if (modded) { v = v * (1.0 + cw[t + p.sc + i]) + cw[t + p.sh + i]; }
        o[base + i] = f16(v * p.oscale);
    }
}
"#;

/// In place on a `[q | k | v]` f16 panel (`ld` wide, parts `inner` wide):
/// workgroup `(part, row)` takes its row of the part back to scale (`gin`),
/// adds the part's bias; q and k then take the RMS norm over the whole
/// inner width with their weight and the split rotation (pairs `i`, `i +
/// half` inside groups of `2 · half`, cos and sin `[rows][inner / 2]` in
/// `pe`), stored unscaled; v is stored times `vscale`. Rows past `nq` (q) or
/// `nk` (k, v) are left alone.
const QK_SRC: &str = r#"
enable f16;
struct QP { ld: u32, inner: u32, nq: u32, nk: u32, gin: f32, eps: f32, vscale: f32, qhalf: u32,
            khalf: u32, qb: u32, kb: u32, vb: u32, qn: u32, kn: u32, qcos: u32, qsin: u32,
            kcos: u32, ksin: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read_write> pan: array<f16>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(2) var<storage, read> pe: array<f32>;
@group(0) @binding(3) var<uniform> p: QP;
var<workgroup> red: array<f32, 256>;
var<workgroup> nv: array<f32, 4096>;

fn wsum(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn lx_qk(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let part = wid.x;
    let row = wid.y;
    var cnt = p.nk;
    if (part == 0u) { cnt = p.nq; }
    if (row >= cnt) { return; }
    var bo = p.vb;
    if (part == 0u) { bo = p.qb; }
    if (part == 1u) { bo = p.kb; }
    let base = row * p.ld + part * p.inner;
    var ss = 0.0;
    for (var i = lid; i < p.inner; i = i + 256u) {
        var v = f32(pan[base + i]) * p.gin;
        if (bo != 0xFFFFFFFFu) { v = v + cw[bo + i]; }
        nv[i] = v;
        ss = ss + v * v;
    }
    if (part == 2u) {
        for (var i = lid; i < p.inner; i = i + 256u) { pan[base + i] = f16(nv[i] * p.vscale); }
        return;
    }
    let inv = inverseSqrt(wsum(ss, lid) / f32(p.inner) + p.eps);
    var nw = p.qn;
    var co = p.qcos;
    var so = p.qsin;
    var half = p.qhalf;
    if (part == 1u) { nw = p.kn; co = p.kcos; so = p.ksin; half = p.khalf; }
    let hp = p.inner / 2u;
    for (var e = lid; e < p.inner; e = e + 256u) {
        var v = nv[e] * inv * cw[nw + e];
        if (co != 0xFFFFFFFFu) {
            let g = e / (2u * half);
            let i = e - g * 2u * half;
            let first = i < half;
            let lo = select(e - half, e, first);
            let ci = g * half + select(i - half, i, first);
            let a = nv[lo] * inv * cw[nw + lo];
            let b = nv[lo + half] * inv * cw[nw + lo + half];
            let c = pe[co + row * hp + ci];
            let s = pe[so + row * hp + ci];
            v = select(b * c + a * s, a * c - b * s, first);
        }
        pan[base + e] = f16(v);
    }
}
"#;

/// Workgroup `(head, row)`: the head's gate logit from the query's input
/// row (f16, times `gin`) and the gate weights (f16 `[heads][cols]` at
/// `gw`) plus its bias, then the head's slice of the attention output
/// times `2 · sigmoid(logit)`, in place.
const GATE_SRC: &str = r#"
enable f16;
struct GP { cols: u32, dh: u32, o_ld: u32, gw: u32, gb: u32, gin: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> xq: array<f16>;
@group(0) @binding(1) var<storage, read> gw: array<f16>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<storage, read_write> att: array<f16>;
@group(0) @binding(4) var<uniform> p: GP;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn lx_gate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let h = wid.x;
    let row = wid.y;
    let xb = row * p.cols;
    let wb = p.gw + h * p.cols;
    var acc = 0.0;
    for (var i = lid; i < p.cols; i = i + 64u) { acc = acc + f32(xq[xb + i]) * f32(gw[wb + i]); }
    red[lid] = acc;
    workgroupBarrier();
    var st = 32u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    var l = red[0] * p.gin;
    if (p.gb != 0xFFFFFFFFu) { l = l + cw[p.gb + h]; }
    let g = 2.0 / (1.0 + exp(-clamp(l, -60.0, 60.0)));
    let ab = row * p.o_ld + h * p.dh;
    for (var d = lid; d < p.dh; d = d + 64u) { att[ab + d] = f16(f32(att[ab + d]) * g); }
}
"#;

/// `hid = f16(gelu_tanh(gu · gin + bias) · oscale)`, one thread an element.
const GELU_SRC: &str = r#"
enable f16;
struct EP { n: u32, wide: u32, gin: f32, oscale: f32, b: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> gu: array<f16>;
@group(0) @binding(1) var<storage, read_write> hid: array<f16>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<uniform> p: EP;

@compute @workgroup_size(256)
fn lx_gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * (65535u * 256u) + gid.x;
    if (i >= p.n) { return; }
    var v = f32(gu[i]) * p.gin;
    if (p.b != 0xFFFFFFFFu) { v = v + cw[p.b + i % p.wide]; }
    let t = tanh(clamp(0.7978845608 * (v + 0.044715 * v * v * v), -15.0, 15.0));
    hid[i] = f16(0.5 * v * (1.0 + t) * p.oscale);
}
"#;

/// `x += g · (y + bias)` over `n` values of rows `w` wide: `g` a chunk at
/// `go` of the table at `tab` under flag 1 (row `ri[ri_off + row]` under
/// flag 2, rows `stride` apart, else the table's first row), else one.
const ADDG_SRC: &str = r#"
struct GP { n: u32, w: u32, flags: u32, tab: u32, stride: u32, go: u32, ri_off: u32, b: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<storage, read> ri: array<u32>;
@group(0) @binding(4) var<uniform> p: GP;

@compute @workgroup_size(256)
fn lx_addg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * (65535u * 256u) + gid.x;
    if (i >= p.n) { return; }
    let r = i / p.w;
    let e = i - r * p.w;
    var v = y[i];
    if (p.b != 0xFFFFFFFFu) { v = v + cw[p.b + e]; }
    var g = 1.0;
    if ((p.flags & 1u) != 0u) {
        var t = p.tab;
        if ((p.flags & 2u) != 0u) { t = t + ri[p.ri_off + r] * p.stride; }
        g = cw[t + p.go + e];
    }
    x[i] = x[i] + g * v;
}
"#;

/// Grid of a one-thread-an-element kernel over `n` elements.
fn grid1(n: usize) -> (u32, u32, u32) {
    let wgs = (n as u32).div_ceil(256);
    (wgs.min(65535), wgs.div_ceil(65535), 1)
}

/// The small arrays of a step (biases, norm weights, modulation tables), in
/// upload order; offsets in floats, `NONE` for an empty one.
#[derive(Default)]
struct Cw(Vec<f32>);

impl Cw {
    fn put(&mut self, x: &[f32]) -> u32 {
        if x.is_empty() {
            return NONE;
        }
        let at = self.0.len() as u32;
        self.0.extend_from_slice(x);
        at
    }
}

/// Where an attention's constants sit: biases of q, k, v, o and the gate,
/// the q and k norm weights, the gate weights in the gate buffer.
#[derive(Clone, Copy)]
struct AOff {
    qb: u32,
    kb: u32,
    vb: u32,
    ob: u32,
    qn: u32,
    kn: u32,
    gb: u32,
    gw: u32,
}

/// The pipelines of the chain.
struct Pipes {
    ada: Arc<wgpu::ComputePipeline>,
    qk: Arc<wgpu::ComputePipeline>,
    gate: Arc<wgpu::ComputePipeline>,
    gelu: Arc<wgpu::ComputePipeline>,
    addg: Arc<wgpu::ComputePipeline>,
    flash128: Arc<wgpu::ComputePipeline>,
    flash64: Arc<wgpu::ComputePipeline>,
    f16: MmCfg,
    f32: MmCfg,
}

/// The working buffers of a step.
struct Bufs {
    xv: wgpu::Buffer,
    xa: wgpu::Buffer,
    /// f16 operands: queries (`nq`, `nq2`) and keys (`nk`, `nk2`).
    nq: wgpu::Buffer,
    nk: wgpu::Buffer,
    nq2: wgpu::Buffer,
    nk2: wgpu::Buffer,
    panel: wgpu::Buffer,
    att: wgpu::Buffer,
    y: wgpu::Buffer,
    gu: wgpu::Buffer,
    hid: wgpu::Buffer,
    cw: wgpu::Buffer,
    gw: wgpu::Buffer,
    ri: wgpu::Buffer,
    pe: wgpu::Buffer,
    vctx: wgpu::Buffer,
    actx: wgpu::Buffer,
}

/// A rotation table in `pe`: cos and sin offsets and the half width of a
/// group.
#[derive(Clone, Copy)]
struct Rot {
    cos: u32,
    sin: u32,
    half: u32,
}

/// What an attention adds its output into.
#[derive(Clone, Copy)]
struct Add<'a> {
    x: &'a wgpu::Buffer,
    w: usize,
    flags: u32,
    tab: u32,
    stride: usize,
    go: usize,
    ri_off: u32,
}

fn ltx_enabled() -> bool {
    std::env::var("CMF_LTX_WGPU").as_deref() != Ok("0")
}

/// Whether this adapter can run the chain at all (matrix units, f16
/// shaders, 32-wide subgroups).
pub(crate) fn ltx_available() -> bool {
    ltx_enabled() && zi::zctx().is_some()
}

/// A dense gate's weights as f16 bits, rows of `cols` (None: the gate is a
/// container matrix, which the chain does not read).
fn gate_half(g: &LtxLin<'_>) -> Option<Vec<u16>> {
    (g.w.is_none() && g.dense.len() == g.rows * g.cols)
        .then(|| g.dense.iter().map(|&v| cortiq_core::quant::f32_to_f16(v.clamp(-65504.0, 65504.0))).collect())
}

/// Every attention of a block: (attention, query rows, key rows, query
/// width, key width, output width).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn shapes<'a, 'b>(
    b: &'b crate::gpu::LtxBlock<'a>,
    n: usize,
    m: usize,
    dim: usize,
    adim: usize,
    vctx: usize,
    actx: usize,
) -> [(&'b LtxAttn<'a>, usize, usize, usize, usize, usize); 6] {
    [
        (&b.video.attn1, n, n, dim, dim, dim),
        (&b.video.attn2, n, vctx, dim, dim, dim),
        (&b.audio.attn1, m, m, adim, adim, adim),
        (&b.audio.attn2, m, actx, adim, adim, adim),
        (&b.a2v, n, m, dim, adim, dim),
        (&b.v2a, m, n, adim, dim, adim),
    ]
}

/// The blocks over both streams on the card. `false`: nothing was written
/// and the caller runs its per-op path.
pub(crate) fn ltx_chain(job: &LtxChain<'_>, vx: &mut [f32], ax: &mut [f32]) -> bool {
    if !ltx_enabled() {
        return false;
    }
    let (v, a) = (&job.video, &job.audio);
    let (n, m, dim, adim) = (v.n, a.n, v.dim, a.dim);
    let nb = job.blocks.len();
    if nb == 0 || n == 0 || m == 0 || vx.len() != n * dim || ax.len() != m * adim {
        return not_mine("an LTX step of the wrong shape");
    }
    if !v.mask.is_empty() || !a.mask.is_empty() {
        return not_mine("an LTX prompt with a key mask");
    }
    let Some(c) = zi::zctx() else {
        return not_mine("no adapter with 16x16 f16 cooperative matrices, f16 shaders and 32-wide subgroups");
    };
    let b0 = &job.blocks[0];
    let xw = b0.a2v.heads * b0.a2v.dh;
    let shapes = |b| shapes(b, n, m, dim, adim, v.ctx_len, a.ctx_len);
    for b in job.blocks {
        for (at, _, _, qc, kc, oc) in shapes(b) {
            let inner = at.heads * at.dh;
            let ok = (at.dh == 64 || at.dh == 128)
                && inner <= QK_MAX
                && inner.is_multiple_of(128)
                && at.q.rows == inner
                && at.k.rows == inner
                && at.v.rows == inner
                && at.q.cols == qc
                && at.k.cols == kc
                && at.v.cols == kc
                && at.o.rows == oc
                && at.o.cols == inner
                && at.q_norm.len() == inner
                && at.k_norm.len() == inner
                && at.gate.as_ref().is_none_or(|g| g.rows == at.heads && g.cols == qc && gate_half(g).is_some());
            if !ok {
                return not_mine("an LTX attention outside the chain's shapes");
            }
        }
        for s in [&b.video, &b.audio] {
            if s.ff_in.cols != s.ff_out.rows || s.ff_out.cols != s.ff_in.rows || !s.ff_in.rows.is_multiple_of(128) {
                return not_mine("an LTX feed-forward outside the chain's shapes");
            }
        }
    }
    if [n, m, v.ctx_len, a.ctx_len].iter().any(|&r| r > MAX_ROWS) || !dim.is_multiple_of(128) || !adim.is_multiple_of(128) {
        return not_mine("more rows than a dispatch addresses, or widths the tiles do not divide");
    }

    // ── the weights: q4tp tensors in block order ──
    let mut spans = Vec::with_capacity(nb * PER);
    let lin = |l: &LtxLin<'_>| -> Option<(usize, std::ops::Range<usize>)> { span(job.model, l.w?, l.rows, l.cols) };
    for b in job.blocks {
        let mut ls: Vec<&LtxLin<'_>> = Vec::with_capacity(PER);
        for s in [&b.video, &b.audio] {
            for at in [&s.attn1, &s.attn2] {
                ls.extend([&at.q, &at.k, &at.v, &at.o]);
            }
            ls.extend([&s.ff_in, &s.ff_out]);
        }
        for at in [&b.a2v, &b.v2a] {
            ls.extend([&at.q, &at.k, &at.v, &at.o]);
        }
        for l in ls {
            match lin(l) {
                Some(sp) => spans.push(sp),
                None => return not_mine("an LTX projection that is not a q4tp matrix"),
            }
        }
    }

    let _op = c.op_gate.enter();
    let _gate = c.mm_gate.lock().unwrap();
    let gd = Guards::from_env();
    let fc = flash_cfg();
    let lim = c.device.limits();
    if vit_flash_shared(fc, 128) > lim.max_compute_workgroup_storage_size {
        return decline("the attention tile does not fit the workgroup memory");
    }

    // ── constants: one upload ──
    let mut cw = Cw::default();
    let mut gwv: Vec<u16> = Vec::new();
    let mut put_attn = |at: &LtxAttn<'_>, cw: &mut Cw| -> AOff {
        let (gb, gw) = match &at.gate {
            Some(g) => {
                let off = gwv.len() as u32;
                gwv.extend(gate_half(g).unwrap_or_default());
                (cw.put(g.bias), off)
            }
            None => (NONE, NONE),
        };
        AOff {
            qb: cw.put(at.q.bias),
            kb: cw.put(at.k.bias),
            vb: cw.put(at.v.bias),
            ob: cw.put(at.o.bias),
            qn: cw.put(at.q_norm),
            kn: cw.put(at.k_norm),
            gb,
            gw,
        }
    };
    struct SOff {
        a1: AOff,
        a2: AOff,
        fb1: u32,
        fb2: u32,
        mods: u32,
        ctx: u32,
    }
    struct BOff {
        v: SOff,
        a: SOff,
        a2v: AOff,
        v2a: AOff,
        xv: u32,
        xa: u32,
        ga: u32,
        gv: u32,
    }
    let mut lay: Vec<BOff> = Vec::with_capacity(nb);
    for b in job.blocks {
        let mut stream = |s: &crate::gpu::LtxStream<'_>, cw: &mut Cw| SOff {
            a1: put_attn(&s.attn1, cw),
            a2: put_attn(&s.attn2, cw),
            fb1: cw.put(s.ff_in.bias),
            fb2: cw.put(s.ff_out.bias),
            mods: cw.put(s.mods),
            ctx: cw.put(s.ctx_mod),
        };
        let vo = stream(&b.video, &mut cw);
        let ao = stream(&b.audio, &mut cw);
        let a2v = put_attn(&b.a2v, &mut cw);
        let v2a = put_attn(&b.v2a, &mut cw);
        lay.push(BOff {
            v: vo,
            a: ao,
            a2v,
            v2a,
            xv: cw.put(b.xmods_v),
            xa: cw.put(b.xmods_a),
            ga: cw.put(b.gate_a2v),
            gv: cw.put(b.gate_v2a),
        });
    }
    if gwv.len() < 2 {
        gwv.resize(2, 0);
    }
    // row indices: video, video fusion, audio, audio fusion
    let ri: Vec<u32> = [v.rows, v.xrows, a.rows, a.xrows].concat();
    let (r_v, r_vx, r_a, r_ax) = (0u32, n as u32, 2 * n as u32, (2 * n + m) as u32);
    // rotations: cos then sin of each table
    let mut pev: Vec<f32> = Vec::new();
    let mut rot = |t: (&[f32], &[f32], usize)| -> Rot {
        let r = Rot { cos: pev.len() as u32, sin: (pev.len() + t.0.len()) as u32, half: t.2 as u32 };
        pev.extend_from_slice(t.0);
        pev.extend_from_slice(t.1);
        r
    };
    let (v_pe, a_pe, v_xpe, a_xpe) = (rot(v.pe), rot(a.pe), rot(v.xpe), rot(a.xpe));
    let side_ok = |s: &LtxSide<'_>, w: usize| {
        s.pe.0.len() == s.n * s.dim / 2 && s.xpe.0.len() == s.n * w / 2 && s.ctx.len() == s.ctx_len * s.dim && s.rows.len() == s.n && s.xrows.len() == s.n
    };
    if !side_ok(v, xw) || !side_ok(a, xw) {
        return not_mine("an LTX stream's tables of the wrong size");
    }
    if (cw.0.len() * 4) as u64 > lim.max_storage_buffer_binding_size {
        return decline("the step's constants pass the largest storage binding");
    }

    // ── the working set ──
    let wide = b0.video.ff_in.rows.max(b0.audio.ff_in.rows);
    let rq = zi::pad_rows(n.max(m));
    let rk = zi::pad_rows(n.max(m).max(v.ctx_len).max(a.ctx_len));
    let wmax = dim.max(adim);
    let inner_max = job
        .blocks
        .iter()
        .flat_map(|b| shapes(b).map(|s| s.0.heads * s.0.dh))
        .max()
        .unwrap_or(0);
    let acts: u64 = (4 * (n * dim + m * adim)
        + 2 * (2 * rq + 2 * rk) * wmax
        + 2 * rk * 3 * inner_max
        + 2 * rq * inner_max
        + 4 * rq * wmax
        + 2 * 2 * rq * wide) as u64
        + (cw.0.len() * 4 + gwv.len() * 2 + pev.len() * 4 + (v.ctx.len() + a.ctx.len()) * 4) as u64;
    if (2 * rk * 3 * inner_max) as u64 > lim.max_storage_buffer_binding_size
        || (2 * rq * wide) as u64 > lim.max_storage_buffer_binding_size
    {
        return decline("an LTX panel passes the largest storage binding");
    }
    let plan = match plan(c, job.model, &spans, PER, acts) {
        Ok(p) => p,
        Err(said) => return said,
    };
    let sc_oom = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let bf = Bufs {
        xv: zi::sbuf(c, (n * dim * 4) as u64, "lx_xv"),
        xa: zi::sbuf(c, (m * adim * 4) as u64, "lx_xa"),
        nq: zi::sbuf(c, (rq * wmax * 2) as u64, "lx_nq"),
        nk: zi::sbuf(c, (rk * wmax * 2) as u64, "lx_nk"),
        nq2: zi::sbuf(c, (rq * wmax * 2) as u64, "lx_nq2"),
        nk2: zi::sbuf(c, (rk * wmax * 2) as u64, "lx_nk2"),
        panel: zi::sbuf(c, (rk * 3 * inner_max * 2) as u64, "lx_panel"),
        att: zi::sbuf(c, (rq * inner_max * 2) as u64, "lx_att"),
        y: zi::sbuf(c, (rq * wmax * 4) as u64, "lx_y"),
        gu: zi::sbuf(c, (rq * wide * 2) as u64, "lx_gu"),
        hid: zi::sbuf(c, (rq * wide * 2) as u64, "lx_hid"),
        cw: zi::sbuf_init(c, bytemuck::cast_slice(&cw.0), "lx_cw"),
        gw: zi::sbuf_init(c, bytemuck::cast_slice(&gwv), "lx_gw"),
        ri: zi::sbuf_init(c, bytemuck::cast_slice(&ri), "lx_ri"),
        pe: zi::sbuf_init(c, bytemuck::cast_slice(&pev), "lx_pe"),
        vctx: zi::sbuf_init(c, bytemuck::cast_slice(v.ctx), "lx_vctx"),
        actx: zi::sbuf_init(c, bytemuck::cast_slice(a.ctx), "lx_actx"),
    };
    if let Some(e) = pollster::block_on(sc_oom.pop()) {
        eprintln!("ltx wgpu: {e}");
        return decline("the LTX working set did not fit the card");
    }
    let t = tile();
    let pipes = || -> Option<Pipes> {
        let f128 = format!("vit_flash_{}_{}_128_{}", fc.nw, fc.bc, zi::flash_thr());
        let f64 = format!("vit_flash_{}_{}_64_{}", fc.nw, fc.bc, zi::flash_thr());
        Some(Pipes {
            ada: zi::pipeline(c, "lx_ada", ADA_SRC, "lx_ada")?,
            qk: zi::pipeline(c, "lx_qk", QK_SRC, "lx_qk")?,
            gate: zi::pipeline(c, "lx_gate", GATE_SRC, "lx_gate")?,
            gelu: zi::pipeline(c, "lx_gelu", GELU_SRC, "lx_gelu")?,
            addg: zi::pipeline(c, "lx_addg", ADDG_SRC, "lx_addg")?,
            flash128: zi::pipeline(c, &f128, &vit_flash_src(fc, 128)?, "vit_flash")?,
            flash64: zi::pipeline(c, &f64, &vit_flash_src(fc, 64)?, "vit_flash")?,
            f16: site_cfg(t, Epi::F16),
            f32: site_cfg(t, Epi::F32),
        })
    };
    let Some(pp) = pipes() else {
        return decline("a kernel of the LTX chain was rejected by the device");
    };

    // ── the dispatches ──
    let ada = |calls: &mut ZCalls, x: &wgpu::Buffer, out: &wgpu::Buffer, rows: usize, w: usize, tab: u32, stride: usize, sh: usize, sc: usize, ri_off: Option<u32>, rms: bool| {
        let mut flags = rms as u32;
        if tab != NONE {
            flags |= 2;
        }
        if ri_off.is_some() {
            flags |= 4;
        }
        let u = zi::ubuf(
            c,
            &[
                w as u32,
                stride as u32,
                sh as u32,
                sc as u32,
                flags,
                if tab == NONE { 0 } else { tab },
                ri_off.unwrap_or(0),
                1e-6f32.to_bits(),
                p2(-gd.act).to_bits(),
                0,
                0,
                0,
            ],
        );
        let bg = zi::bg(c, &pp.ada, &[x, out, &bf.cw, &bf.ri, &u]);
        calls.push(Class::Rows, Call { pipe: pp.ada.clone(), bg, grid: (rows as u32, 1, 1) });
    };
    let addg = |calls: &mut ZCalls, add: &Add<'_>, rows: usize, b: u32| {
        let total = rows * add.w;
        let u = zi::ubuf(
            c,
            &[total as u32, add.w as u32, add.flags, add.tab, add.stride as u32, add.go as u32, add.ri_off, b, 0, 0, 0, 0],
        );
        let bg = zi::bg(c, &pp.addg, &[add.x, &bf.y, &bf.cw, &bf.ri, &u]);
        calls.push(Class::Rows, Call { pipe: pp.addg.clone(), bg, grid: grid1(total) });
    };
    // one attention: projections into the panel, norm and rotation,
    // attention, the gate, the output projection, the residual
    let attention = |calls: &mut ZCalls,
                     at: &LtxAttn<'_>,
                     o: &AOff,
                     w: &[wgpu::Buffer],
                     xq: &wgpu::Buffer,
                     nq: usize,
                     xk: &wgpu::Buffer,
                     nk: usize,
                     qpe: Option<Rot>,
                     kpe: Option<Rot>,
                     add: &Add<'_>|
     -> Option<()> {
        let (heads, dh) = (at.heads, at.dh);
        let inner = heads * dh;
        let ld = 3 * inner;
        let qs = p2(gd.act - gd.qkv);
        calls.push_mm(Class::MmQkv, mm(c, pp.f16, &margs(nq, inner, at.q.cols, ld, 0, qs), 0, &w[0], xq, &bf.panel)?);
        calls.push_mm(Class::MmQkv, mm(c, pp.f16, &margs(nk, inner, at.k.cols, ld, inner, qs), 0, &w[1], xk, &bf.panel)?);
        calls.push_mm(Class::MmQkv, mm(c, pp.f16, &margs(nk, inner, at.v.cols, ld, 2 * inner, qs), 0, &w[2], xk, &bf.panel)?);
        {
            let none = Rot { cos: NONE, sin: NONE, half: 1 };
            let (q, k) = (qpe.unwrap_or(none), kpe.unwrap_or(none));
            let u = zi::ubuf(
                c,
                &[
                    ld as u32,
                    inner as u32,
                    nq as u32,
                    nk as u32,
                    p2(gd.qkv).to_bits(),
                    1e-6f32.to_bits(),
                    p2(-gd.qkv).to_bits(),
                    q.half,
                    k.half,
                    o.qb,
                    o.kb,
                    o.vb,
                    o.qn,
                    o.kn,
                    q.cos,
                    q.sin,
                    k.cos,
                    k.sin,
                    0,
                    0,
                ],
            );
            let bg = zi::bg(c, &pp.qk, &[&bf.panel, &bf.cw, &bf.pe, &u]);
            calls.push(Class::Rows, Call { pipe: pp.qk.clone(), bg, grid: (3, nq.max(nk) as u32, 1) });
        }
        {
            let scl = std::f32::consts::LOG2_E / (dh as f32).sqrt();
            let u = zi::ubuf(
                c,
                &[nq as u32, nk as u32, ld as u32, inner as u32, (2 * inner) as u32, inner as u32, scl.to_bits(), 1.0f32.to_bits()],
            );
            let fp = if dh == 128 { &pp.flash128 } else { &pp.flash64 };
            let bg = zi::bg(c, fp, &[&bf.panel, &bf.panel, &bf.att, &u]);
            calls.push(Class::Flash, Call { pipe: fp.clone(), bg, grid: ((nq as u32).div_ceil(fc.nw * 16), heads as u32, 1) });
        }
        if let Some(g) = &at.gate {
            let u = zi::ubuf(c, &[g.cols as u32, dh as u32, inner as u32, o.gw, o.gb, p2(gd.act).to_bits(), 0, 0]);
            let bg = zi::bg(c, &pp.gate, &[xq, &bf.gw, &bf.cw, &bf.att, &u]);
            calls.push(Class::Rows, Call { pipe: pp.gate.clone(), bg, grid: (heads as u32, nq as u32, 1) });
        }
        let out = at.o.rows;
        calls.push_mm(Class::MmO, mm(c, pp.f32, &margs(nq, out, inner, out, 0, p2(gd.qkv)), 0, &w[3], &bf.att, &bf.y)?);
        addg(calls, add, nq, o.ob);
        Some(())
    };
    let feed_forward = |calls: &mut ZCalls, s: &crate::gpu::LtxStream<'_>, so: &SOff, w: &[wgpu::Buffer], x: &wgpu::Buffer, rows: usize, width: usize, ri_off: u32| -> Option<()> {
        let wd = s.ff_in.rows;
        ada(calls, x, &bf.nq, rows, width, so.mods, 9 * width, 3 * width, 4 * width, Some(ri_off), true);
        calls.push_mm(Class::MmW13, mm(c, pp.f16, &margs(rows, wd, width, wd, 0, p2(gd.act - gd.gu)), 0, &w[0], &bf.nq, &bf.gu)?);
        {
            let total = rows * wd;
            let u = zi::ubuf(c, &[total as u32, wd as u32, p2(gd.gu).to_bits(), p2(-gd.hid).to_bits(), so.fb1, 0, 0, 0]);
            let bg = zi::bg(c, &pp.gelu, &[&bf.gu, &bf.hid, &bf.cw, &u]);
            calls.push(Class::Rows, Call { pipe: pp.gelu.clone(), bg, grid: grid1(total) });
        }
        calls.push_mm(Class::MmW2, mm(c, pp.f32, &margs(rows, width, wd, width, 0, p2(gd.hid)), 0, &w[1], &bf.hid, &bf.y)?);
        let add = Add { x, w: width, flags: 3, tab: so.mods, stride: 9 * width, go: 5 * width, ri_off };
        addg(calls, &add, rows, so.fb2);
        Some(())
    };
    let build = |blocks: std::ops::Range<usize>, wb: &[wgpu::Buffer]| -> Option<ZCalls> {
        let mut calls = ZCalls::default();
        for (j, bi) in blocks.clone().enumerate() {
            let (blk, l) = (&job.blocks[bi], &lay[bi]);
            let w = &wb[j * PER..(j + 1) * PER];
            // a stream's self-attention, then its attention to the prompt
            let streams = [
                (&blk.video, &l.v, &bf.xv, n, dim, r_v, v_pe, &bf.vctx, v.ctx_len, &w[0..10]),
                (&blk.audio, &l.a, &bf.xa, m, adim, r_a, a_pe, &bf.actx, a.ctx_len, &w[10..20]),
            ];
            for (s, so, x, rows, width, ro, pe, ctx, cl, ws) in streams {
                let add = |go: usize| Add { x, w: width, flags: 3, tab: so.mods, stride: 9 * width, go, ri_off: ro };
                ada(&mut calls, x, &bf.nq, rows, width, so.mods, 9 * width, 0, width, Some(ro), true);
                attention(&mut calls, &s.attn1, &so.a1, &ws[0..4], &bf.nq, rows, &bf.nq, rows, Some(pe), Some(pe), &add(2 * width))?;
                ada(&mut calls, x, &bf.nq, rows, width, so.mods, 9 * width, 6 * width, 7 * width, Some(ro), true);
                ada(&mut calls, ctx, &bf.nk, cl, width, so.ctx, 2 * width, 0, width, None, false);
                attention(&mut calls, &s.attn2, &so.a2, &ws[4..8], &bf.nq, rows, &bf.nk, cl, None, None, &add(8 * width))?;
            }
            // the fusion: all four operands off the streams as they are now
            ada(&mut calls, &bf.xv, &bf.nq, n, dim, l.xv, 4 * dim, dim, 0, Some(r_vx), true);
            ada(&mut calls, &bf.xa, &bf.nk, m, adim, l.xa, 4 * adim, adim, 0, Some(r_ax), true);
            ada(&mut calls, &bf.xa, &bf.nq2, m, adim, l.xa, 4 * adim, 3 * adim, 2 * adim, Some(r_ax), true);
            ada(&mut calls, &bf.xv, &bf.nk2, n, dim, l.xv, 4 * dim, 3 * dim, 2 * dim, Some(r_vx), true);
            let add_v = Add { x: &bf.xv, w: dim, flags: 1, tab: l.ga, stride: dim, go: 0, ri_off: 0 };
            attention(&mut calls, &blk.a2v, &l.a2v, &w[20..24], &bf.nq, n, &bf.nk, m, Some(v_xpe), Some(a_xpe), &add_v)?;
            let add_a = Add { x: &bf.xa, w: adim, flags: 1, tab: l.gv, stride: adim, go: 0, ri_off: 0 };
            attention(&mut calls, &blk.v2a, &l.v2a, &w[24..28], &bf.nq2, m, &bf.nk2, n, Some(a_xpe), Some(v_xpe), &add_a)?;
            feed_forward(&mut calls, &blk.video, &l.v, &w[8..10], &bf.xv, n, dim, r_v)?;
            feed_forward(&mut calls, &blk.audio, &l.a, &w[18..20], &bf.xa, m, adim, r_a)?;
        }
        Some(calls)
    };

    // ── the step: streams up, the runs of blocks, streams back ──
    let t0 = std::time::Instant::now();
    let uid = job.model.uid() as usize;
    c.queue.write_buffer(&bf.xv, 0, bytemuck::cast_slice(vx));
    c.queue.write_buffer(&bf.xa, 0, bytemuck::cast_slice(ax));
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut submitted = false;
    for (kept, blocks) in plan.runs() {
        let mine = &spans[blocks.start * PER..blocks.end * PER];
        if !kept {
            if submitted {
                zi::wait(c);
            }
            let stale: Vec<(usize, usize)> = spans
                .iter()
                .enumerate()
                .filter(|(i, _)| !plan.keep.contains(&(i / PER)) && !blocks.contains(&(i / PER)))
                .map(|(_, (idx, _))| (uid, *idx))
                .collect();
            super::super::weight_evict(c, &stale);
        }
        let wb = match fetch(c, job.model, mine) {
            Ok(wb) => wb,
            Err(said) => {
                let _ = pollster::block_on(sc.pop());
                return said;
            }
        };
        let Some(calls) = build(blocks, &wb) else {
            let _ = pollster::block_on(sc.pop());
            return decline("a kernel of the LTX chain was rejected by the device");
        };
        let mut enc = c.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            calls.record(&mut pass, None);
        }
        c.queue.submit(Some(enc.finish()));
        submitted = true;
    }
    let rv = zi::read_bytes(c, &bf.xv, (n * dim * 4) as u64);
    let ra = zi::read_bytes(c, &bf.xa, (m * adim * 4) as u64);
    if let Some(e) = pollster::block_on(sc.pop()) {
        return decline(&format!("the LTX chain's command buffer failed: {e}"));
    }
    let (Some(rv), Some(ra)) = (rv, ra) else {
        return decline("the readback failed");
    };
    let (ov, oa): (&[f32], &[f32]) = (bytemuck::cast_slice(&rv), bytemuck::cast_slice(&ra));
    if ov.iter().chain(oa.iter()).any(|x| !x.is_finite()) {
        return decline("the LTX chain produced a non-finite stream (an f16 range guard overflowed?)");
    }
    vx.copy_from_slice(ov);
    ax.copy_from_slice(oa);
    if std::env::var("CMF_LTX_WGPU_PROF").as_deref() == Ok("1") {
        eprintln!(
            "ltx wgpu chain: {:.3} s, {} blocks kept on the card, {} passing",
            t0.elapsed().as_secs_f64(),
            plan.keep.len(),
            plan.pass.iter().map(|r| r.len()).sum::<usize>()
        );
    }
    static FLUSHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !FLUSHED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        super::super::pipeline_cache_flush();
    }
    true
}

// ───────────────────────────── the video VAE ─────────────────────────────

/// `out[p][c] = f16(silu(y · inv) )` with `inv` the RMS over the channels of
/// position `p` (`flags` 1), or `f16(y)` (0). One workgroup a position.
const VNS_SRC: &str = r#"
enable f16;
struct NP { c: u32, p: u32, flags: u32, eps: f32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@group(0) @binding(2) var<uniform> q: NP;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn vx_ns(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let pos = wid.y * 65535u + wid.x;
    if (pos >= q.p) { return; }
    let base = pos * q.c;
    var inv = 1.0;
    if ((q.flags & 1u) != 0u) {
        var ss = 0.0;
        for (var i = lid; i < q.c; i = i + 64u) { let v = y[base + i]; ss = ss + v * v; }
        red[lid] = ss;
        workgroupBarrier();
        var st = 32u;
        loop {
            if (st == 0u) { break; }
            if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
            workgroupBarrier();
            st = st >> 1u;
        }
        inv = inverseSqrt(red[0] / f32(q.c) + q.eps);
    }
    for (var i = lid; i < q.c; i = i + 64u) {
        var v = y[base + i] * inv;
        if ((q.flags & 1u) != 0u) { v = v / (1.0 + exp(-v)); }
        o[base + i] = f16(clamp(v, -65504.0, 65504.0));
    }
}
"#;

/// `o[i] = (flags & 1 ? o[i] : 0) + y[i] + (flags & 2 ? b[i % c] : 0)`.
const VADD_SRC: &str = r#"
struct AP { n: u32, c: u32, flags: u32, _a: u32 };
@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<uniform> q: AP;

@compute @workgroup_size(256)
fn vx_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * (65535u * 256u) + gid.x;
    if (i >= q.n) { return; }
    var v = y[i];
    if ((q.flags & 2u) != 0u) { v = v + b[i % q.c]; }
    if ((q.flags & 1u) != 0u) { v = v + o[i]; }
    o[i] = v;
}
"#;

/// Depth to space of `y` `[f·h·w][c·p1·p2·p3]` (channels laid out
/// `(c p1 p2 p3)`) into `o` `[(f·p1 − drop)·h·p2·w·p3][c]`.
const VD2S_SRC: &str = r#"
struct DP { n: u32, c: u32, f: u32, h: u32, w: u32, p1: u32, p2: u32, p3: u32, drop: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> q: DP;

@compute @workgroup_size(256)
fn vx_d2s(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * (65535u * 256u) + gid.x;
    if (i >= q.n) { return; }
    let cc = i % q.c;
    var r = i / q.c;
    let w2 = q.w * q.p3;
    let h2 = q.h * q.p2;
    let ow = r % w2;
    r = r / w2;
    let oh = r % h2;
    let tf = r / h2 + q.drop;
    let fr = tf / q.p1;
    let a = tf % q.p1;
    let hh = oh / q.p2;
    let b = oh % q.p2;
    let ww = ow / q.p3;
    let d = ow % q.p3;
    let src_c = ((cc * q.p1 + a) * q.p2 + b) * q.p3 + d;
    let src_p = (fr * q.h + hh) * q.w + ww;
    o[i] = y[src_p * (q.c * q.p1 * q.p2 * q.p3) + src_c];
}
"#;

/// A convolution's weights `[out][in · 27]` in (channel, tap) order as the
/// f16 plane the convolution GEMM reads: `[out_pad][27][in]`, rows past
/// `out` zero.
fn conv_plane(w: &[f32], out: usize, cin: usize, out_pad: usize) -> Vec<u16> {
    let kk = cin * 27;
    let mut p = vec![0u16; out_pad * kk];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 16);
    let per = out.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (t, rows) in p[..out * kk].chunks_mut(per * kk).enumerate() {
            sc.spawn(move || {
                for (i, row) in rows.chunks_exact_mut(kk).enumerate() {
                    let src = &w[(t * per + i) * kk..][..kk];
                    for tap in 0..27 {
                        for ci in 0..cin {
                            row[tap * cin + ci] = cortiq_core::quant::f32_to_f16(src[ci * 27 + tap].clamp(-65504.0, 65504.0));
                        }
                    }
                }
            });
        }
    });
    p
}

/// The LTX-2.5 video VAE's stages over a volume `x` `[c][f][h][w]` on the
/// card, the activation position-major (`[f·h·w][c]`) inside: the input goes
/// up once and the output of the last stage comes back once, as `[c][f][h][w]`
/// with its `(c, f, h, w)`. `None`: the caller runs its host path.
pub(crate) fn ltx_vae_chain(
    steps: &[crate::gpu::Vae3dStep<'_>],
    x: &[f32],
    c0: usize,
    f0: usize,
    h0: usize,
    w0: usize,
) -> Option<(Vec<f32>, usize, usize, usize, usize)> {
    use crate::gpu::Vae3dStep;
    if std::env::var("CMF_LTX_WGPU_VAE").as_deref() == Ok("0") || steps.is_empty() || x.len() != c0 * f0 * h0 * w0 {
        return None;
    }
    let c = zi::zctx()?;
    let lim = c.device.limits();
    // the shapes every stage leaves, the largest activation and product
    let (mut ch, mut fr, mut hh, mut ww) = (c0, f0, h0, w0);
    let (mut act_max, mut y_max) = (x.len(), 0usize);
    let cfg = MmCfg { conv: 6, ..zi::default_cfg(Epi::F32) };
    if !cfg.valid() {
        return None;
    }
    for s in steps {
        match s {
            Vae3dStep::Conv { w, b, in_ch, out_ch } => {
                if *in_ch != ch || w.len() != out_ch * in_ch * 27 || b.len() != *out_ch || !in_ch.is_multiple_of(cfg.bk as usize) {
                    return None;
                }
                let p = fr * hh * ww;
                if p.div_ceil(cfg.bm as usize) > 65535 || fr > 0xFFFF {
                    return None;
                }
                ch = *out_ch;
                y_max = y_max.max(p * out_ch.next_multiple_of(cfg.bn as usize));
            }
            Vae3dStep::DepthToSpace { p } => {
                let q = p.0 * p.1 * p.2;
                if q == 0 || !ch.is_multiple_of(q) {
                    return None;
                }
                ch /= q;
                fr = fr * p.0 - (p.0 == 2) as usize;
                hh *= p.1;
                ww *= p.2;
            }
            Vae3dStep::NormSilu { .. } | Vae3dStep::Save | Vae3dStep::AddSaved => {}
        }
        act_max = act_max.max(ch * fr * hh * ww);
    }
    let room = lim.max_storage_buffer_binding_size;
    if (act_max.max(y_max) * 4) as u64 > room {
        return None;
    }
    if (act_max * 4 * 3 + act_max * 2 + y_max * 4) as u64 + (512 << 20) > c.vram_budget.saturating_sub(super::super::resident_bytes()) {
        return None;
    }
    let _op = c.op_gate.enter();
    let _gate = c.mm_gate.lock().unwrap();
    let t0 = std::time::Instant::now();
    let pns = zi::pipeline(c, "vx_ns", VNS_SRC, "vx_ns")?;
    let padd = zi::pipeline(c, "vx_add", VADD_SRC, "vx_add")?;
    let pd2s = zi::pipeline(c, "vx_d2s", VD2S_SRC, "vx_d2s")?;
    let pmm = zi::mm_pipe(c, cfg)?;
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut cur = zi::sbuf(c, (act_max * 4) as u64, "vx_a");
    let mut tmp = zi::sbuf(c, (act_max * 4) as u64, "vx_b");
    let saved = zi::sbuf(c, (act_max * 4) as u64, "vx_s");
    let h16 = zi::sbuf(c, (act_max * 2) as u64, "vx_h");
    let y = zi::sbuf(c, (y_max.max(16) * 4) as u64, "vx_y");
    // the input, position-major
    {
        let p = f0 * h0 * w0;
        let mut t = vec![0f32; x.len()];
        for (ci, plane) in x.chunks_exact(p).enumerate() {
            for (pi, &v) in plane.iter().enumerate() {
                t[pi * c0 + ci] = v;
            }
        }
        c.queue.write_buffer(&cur, 0, bytemuck::cast_slice(&t));
    }
    let mut calls = ZCalls::default();
    let mut keep: Vec<wgpu::Buffer> = Vec::new();
    let (mut ch, mut fr, mut hh, mut ww) = (c0, f0, h0, w0);
    let mut normed = false;
    let mut saved_len = 0usize;
    let dummy = zi::sbuf(c, 16, "vx_dummy");
    let add = |calls: &mut ZCalls, o: &wgpu::Buffer, src: &wgpu::Buffer, b: Option<&wgpu::Buffer>, n: usize, ch: usize, accumulate: bool| {
        let flags = accumulate as u32 | ((b.is_some() as u32) << 1);
        let u = zi::ubuf(c, &[n as u32, ch as u32, flags, 0]);
        let bg = zi::bg(c, &padd, &[o, src, b.unwrap_or(&dummy), &u]);
        calls.push(Class::Rows, Call { pipe: padd.clone(), bg, grid: grid1(n) });
    };
    for s in steps {
        let p = fr * hh * ww;
        match s {
            Vae3dStep::NormSilu { eps } => {
                // folded into the next convolution's operand
                let _ = eps;
                normed = true;
            }
            Vae3dStep::Save => {
                add(&mut calls, &saved, &cur, None, ch * p, ch, false);
                saved_len = ch * p;
            }
            Vae3dStep::AddSaved => {
                if saved_len != ch * p {
                    return None;
                }
                add(&mut calls, &cur, &saved, None, ch * p, ch, true);
            }
            Vae3dStep::Conv { w, b, in_ch, out_ch } => {
                let eps = steps
                    .iter()
                    .find_map(|s| if let Vae3dStep::NormSilu { eps } = s { Some(*eps) } else { None })
                    .unwrap_or(1e-8);
                // the operand: the activation normed and through SiLU, or as it is
                let u = zi::ubuf(c, &[ch as u32, p as u32, normed as u32, eps.to_bits()]);
                let bg = zi::bg(c, &pns, &[&cur, &h16, &u]);
                let pc = p as u32;
                calls.push(Class::Rows, Call { pipe: pns.clone(), bg, grid: (pc.min(65535), pc.div_ceil(65535), 1) });
                normed = false;
                let op = out_ch.next_multiple_of(cfg.bn as usize);
                let plane = conv_plane(w, *out_ch, *in_ch, op);
                let wb = zi::sbuf_init(c, bytemuck::cast_slice(&plane), "vx_w");
                let mut bias = b.to_vec();
                bias.resize(op, 0.0);
                let bb = zi::sbuf_init(c, bytemuck::cast_slice(&bias), "vx_bias");
                let args = MmArgs {
                    m: p as u32,
                    n: op as u32,
                    k: (in_ch * 27) as u32,
                    ldo: op as u32,
                    ocol: 0,
                    arow: fr as u32,
                    oscale: 1.0,
                    conv: [ww as u32, hh as u32, *in_ch as u32],
                };
                let u = zi::mm_uniform(c, &args);
                let bg = zi::bg(c, &pmm, &[&wb, &h16, &y, &u]);
                calls.push_mm(
                    Class::MmW13,
                    MmCall { pipe: pmm.clone(), bg, grid: (op as u32 / cfg.bn, (p as u32).div_ceil(cfg.bm)), mnk: [p as u32, op as u32, (in_ch * 27) as u32], bits: false },
                );
                // the result with its bias becomes the activation
                add(&mut calls, &cur, &y, Some(&bb), op * p, op, false);
                keep.extend([wb, bb]);
                ch = op;
                if op != *out_ch {
                    // only the last convolution is padded: its rows stay padded
                    // and the readback keeps the first `out_ch` channels
                    if !std::ptr::eq(s, steps.last()?) {
                        return None;
                    }
                }
            }
            Vae3dStep::DepthToSpace { p: (p1, p2, p3) } => {
                let drop = (*p1 == 2) as usize;
                let (oc, of, oh, ow) = (ch / (p1 * p2 * p3), fr * p1 - drop, hh * p2, ww * p3);
                let n = oc * of * oh * ow;
                let u = zi::ubuf(c, &[n as u32, oc as u32, fr as u32, hh as u32, ww as u32, *p1 as u32, *p2 as u32, *p3 as u32, drop as u32, 0, 0, 0]);
                let bg = zi::bg(c, &pd2s, &[&cur, &tmp, &u]);
                calls.push(Class::Rows, Call { pipe: pd2s.clone(), bg, grid: grid1(n) });
                std::mem::swap(&mut cur, &mut tmp);
                (ch, fr, hh, ww) = (oc, of, oh, ow);
            }
        }
    }
    if let Some(e) = pollster::block_on(sc.pop()) {
        eprintln!("ltx wgpu vae: {e}");
        return None;
    }
    let sv = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        calls.record(&mut pass, None);
    }
    c.queue.submit(Some(enc.finish()));
    let p = fr * hh * ww;
    let raw = zi::read_bytes(c, &cur, (p * ch * 4) as u64);
    if let Some(e) = pollster::block_on(sv.pop()) {
        eprintln!("ltx wgpu vae: {e}");
        return None;
    }
    let raw = raw?;
    let v: &[f32] = bytemuck::cast_slice(&raw);
    // the last convolution's real channels, channel-major
    let out_ch = match steps.last()? {
        Vae3dStep::Conv { out_ch, .. } => *out_ch,
        _ => ch,
    };
    let mut out = vec![0f32; out_ch * p];
    for pi in 0..p {
        for ci in 0..out_ch {
            out[ci * p + pi] = v[pi * ch + ci];
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return None;
    }
    if std::env::var("CMF_LTX_WGPU_PROF").as_deref() == Ok("1") {
        eprintln!("ltx wgpu vae chain: {:.3} s, {} dispatches", t0.elapsed().as_secs_f64(), calls.len());
    }
    drop(keep);
    Some((out, out_ch, fr, hh, ww))
}

// ───────────────────────────── Qwen-Image ─────────────────────────────

/// `out = f16(((x − mean) · inv · (1 + scale) + shift) · oscale)` a row, the
/// affine-free LayerNorm; shift and scale at `sh` and `sc` of `cw`. One
/// workgroup a row.
const QLN_SRC: &str = r#"
enable f16;
struct LP { w: u32, sh: u32, sc: u32, _a: u32, eps: f32, oscale: f32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<uniform> p: LP;
var<workgroup> red: array<f32, 256>;

fn wsum(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn qw_ln(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let base = wid.x * p.w;
    var s = 0.0;
    for (var i = lid; i < p.w; i = i + 256u) { s = s + x[base + i]; }
    let mean = wsum(s, lid) / f32(p.w);
    var q = 0.0;
    for (var i = lid; i < p.w; i = i + 256u) { let d = x[base + i] - mean; q = q + d * d; }
    let inv = inverseSqrt(wsum(q, lid) / f32(p.w) + p.eps);
    for (var i = lid; i < p.w; i = i + 256u) {
        let v = (x[base + i] - mean) * inv * (1.0 + cw[p.sc + i]) + cw[p.sh + i];
        o[base + i] = f16(clamp(v * p.oscale, -65504.0, 65504.0));
    }
}
"#;

/// In place on the joint `[q | k | v]` f16 panel (`ld` wide, parts `hs` wide,
/// text rows first): workgroup `(part · heads + head, row)` takes its head of
/// the row back to scale (`gin`) and adds the bias of the row's stream; q and
/// k then take the head's RMS norm with the stream's weight and the
/// interleaved rotation (pairs 2j, 2j + 1; cos at `co`, sin at `so`, rows of
/// `hd / 2`), stored unscaled; v is stored times `vscale`.
const QQK_SRC: &str = r#"
enable f16;
struct QP { ld: u32, hs: u32, hd: u32, heads: u32, nt: u32, n: u32, gin: f32, vscale: f32,
            tb: u32, ib: u32, tn: u32, in_: u32, tco: u32, tso: u32, ico: u32, iso: u32, eps: f32, split: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read_write> pan: array<f16>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(2) var<storage, read> pe: array<f32>;
@group(0) @binding(3) var<uniform> p: QP;
var<workgroup> red: array<f32, 64>;
var<workgroup> nv: array<f32, 256>;

@compute @workgroup_size(64)
fn qw_qk(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let part = wid.x / p.heads;
    let h = wid.x - part * p.heads;
    let row = wid.y;
    if (row >= p.n) { return; }
    let text = row < p.nt;
    // biases: [q | k | v] of the stream, norms: [q | k] of the stream
    var bo = select(p.ib, p.tb, text);
    var no = select(p.in_, p.tn, text);
    let base = row * p.ld + part * p.hs + h * p.hd;
    var ss = 0.0;
    for (var i = lid; i < p.hd; i = i + 64u) {
        var v = f32(pan[base + i]) * p.gin;
        if (bo != 0xFFFFFFFFu) { v = v + cw[bo + part * p.hs + h * p.hd + i]; }
        nv[i] = v;
        ss = ss + v * v;
    }
    if (part == 2u) {
        for (var i = lid; i < p.hd; i = i + 64u) { pan[base + i] = f16(clamp(nv[i] * p.vscale, -65504.0, 65504.0)); }
        return;
    }
    red[lid] = ss;
    workgroupBarrier();
    var st = 32u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let inv = inverseSqrt(red[0] / f32(p.hd) + p.eps);
    let wo = no + part * p.hd;
    let hp = p.hd / 2u;
    var co = p.ico;
    var so = p.iso;
    var r = row - p.nt;
    if (text) { co = p.tco; so = p.tso; r = row; }
    let split = p.split != 0u;
    for (var j = lid; j < hp; j = j + 64u) {
        let i0 = select(2u * j, j, split);
        let i1 = select(2u * j + 1u, j + hp, split);
        let a = nv[i0] * inv * cw[wo + i0];
        let b = nv[i1] * inv * cw[wo + i1];
        let c = pe[co + r * hp + j];
        let s = pe[so + r * hp + j];
        pan[base + i0] = f16(a * c - b * s);
        pan[base + i1] = f16(a * s + b * c);
    }
}
"#;

/// A Qwen-Image step's blocks over both streams (`image` `[ni][hs]`, `text`
/// `[nt][hs]`) on the card: the zi_mm products reading the q4tp tensors in
/// place, joint attention over [text | image] on `vit_flash`. `false`: nothing
/// was written.
pub(crate) fn qwen_chain(model: &Arc<CmfModel>, a: &mut crate::gpu::QwenImageChainArgs<'_>) -> bool {
    if !ltx_enabled() || std::env::var("CMF_QWEN_WGPU_H3").as_deref() == Ok("0") {
        return false;
    }
    let (ni, nt, heads, hd) = (a.image_tokens, a.text_tokens, a.heads, a.head_dim);
    let hs = heads * hd;
    let n = ni + nt;
    let nb = a.blocks.len();
    if nb == 0 || ni == 0 || nt == 0 || hd != 128 || !hs.is_multiple_of(128) || n > MAX_ROWS || a.image.len() != ni * hs || a.text.len() != nt * hs {
        return not_mine("a Qwen-Image step outside the chain's shapes");
    }
    if a.image_cos.len() != ni * hd / 2 || a.text_cos.len() != nt * hd / 2 {
        return not_mine("Qwen-Image rotation tables of the wrong size");
    }
    let Some(c) = zi::zctx() else {
        return not_mine("no adapter with 16x16 f16 cooperative matrices");
    };
    // the twelve q4tp tensors of a block, in the order the chain reads them
    let order = |b: &crate::gpu::QwenImageChainBlock<'_>| {
        [b.text_q, b.text_k, b.text_v, b.image_q, b.image_k, b.image_v, b.text_out, b.image_out, b.text_mlp_in, b.text_mlp_out, b.image_mlp_in, b.image_mlp_out]
    };
    const QPER: usize = 12;
    let mut spans = Vec::with_capacity(nb * QPER);
    let mut inter = 0usize;
    for b in a.blocks {
        for (j, ix) in order(b).into_iter().enumerate() {
            let e = match model.tensors.get(ix) {
                Some(e) if e.shape.len() == 2 => e,
                _ => return not_mine("a Qwen-Image projection that is not a matrix"),
            };
            if j == 8 {
                inter = e.shape[0];
            }
            match span(model, ix, e.shape[0], e.shape[1]) {
                Some(sp) => spans.push(sp),
                None => return not_mine("a Qwen-Image projection that is not a q4tp matrix"),
            }
        }
        if b.image_mod.len() != 6 * hs || b.text_mod.len() != 6 * hs {
            return not_mine("a Qwen-Image modulation of the wrong width");
        }
    }
    if inter == 0 || !inter.is_multiple_of(128) {
        return not_mine("a Qwen-Image MLP width the tiles do not divide");
    }
    let _op = c.op_gate.enter();
    let _gate = c.mm_gate.lock().unwrap();
    let gd = Guards::from_env();
    let fc = flash_cfg();
    let lim = c.device.limits();

    // ── constants ──
    let mut cw = Cw::default();
    struct BO {
        tm: u32,
        im: u32,
        tb: u32,
        ib: u32,
        tn: u32,
        in_: u32,
        tob: u32,
        iob: u32,
        tf1: u32,
        tf2: u32,
        if1: u32,
        if2: u32,
    }
    let cat = |x: &[&[f32]]| -> Vec<f32> { x.iter().flat_map(|v| v.iter().copied()).collect() };
    let mut lay = Vec::with_capacity(nb);
    for b in a.blocks {
        let qkv_t = cat(&[b.text_q_bias, b.text_k_bias, b.text_v_bias]);
        let qkv_i = cat(&[b.image_q_bias, b.image_k_bias, b.image_v_bias]);
        if qkv_t.len() != 3 * hs || qkv_i.len() != 3 * hs || b.text_q_norm.len() != hd || b.image_q_norm.len() != hd {
            return not_mine("Qwen-Image biases or norms of the wrong width");
        }
        lay.push(BO {
            tm: cw.put(b.text_mod),
            im: cw.put(b.image_mod),
            tb: cw.put(&qkv_t),
            ib: cw.put(&qkv_i),
            tn: cw.put(&cat(&[b.text_q_norm, b.text_k_norm])),
            in_: cw.put(&cat(&[b.image_q_norm, b.image_k_norm])),
            tob: cw.put(b.text_out_bias),
            iob: cw.put(b.image_out_bias),
            tf1: cw.put(b.text_mlp_in_bias),
            tf2: cw.put(b.text_mlp_out_bias),
            if1: cw.put(b.image_mlp_in_bias),
            if2: cw.put(b.image_mlp_out_bias),
        });
    }
    let mut pev: Vec<f32> = Vec::new();
    let mut put_pe = |v: &[f32]| -> u32 {
        let at = pev.len() as u32;
        pev.extend_from_slice(v);
        at
    };
    let (tco, tso, ico, iso) = (put_pe(a.text_cos), put_pe(a.text_sin), put_pe(a.image_cos), put_pe(a.image_sin));
    if (cw.0.len() * 4) as u64 > lim.max_storage_buffer_binding_size {
        return decline("the Qwen-Image constants pass the largest storage binding");
    }

    // ── the working set ──
    let rmax = zi::pad_rows(ni.max(nt));
    let rn = zi::pad_rows(n) + 128;
    let acts: u64 = (4 * n * hs + 2 * 2 * rmax * hs + 2 * rn * 3 * hs + 2 * rn * hs + 4 * rmax * hs + 2 * 2 * rmax * inter) as u64
        + (cw.0.len() * 4 + pev.len() * 4) as u64;
    if (2 * rn * 3 * hs) as u64 > lim.max_storage_buffer_binding_size || (2 * rmax * inter) as u64 > lim.max_storage_buffer_binding_size {
        return decline("a Qwen-Image panel passes the largest storage binding");
    }
    let plan = match plan(c, model, &spans, QPER, acts) {
        Ok(p) => p,
        Err(said) => return said,
    };
    let sc_oom = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let xt = zi::sbuf(c, (nt * hs * 4) as u64, "qw_xt");
    let xi = zi::sbuf(c, (ni * hs * 4) as u64, "qw_xi");
    let nt_b = zi::sbuf(c, (rmax * hs * 2) as u64, "qw_nt");
    let ni_b = zi::sbuf(c, (rmax * hs * 2) as u64, "qw_ni");
    let panel = zi::sbuf(c, (rn * 3 * hs * 2) as u64, "qw_panel");
    let att = zi::sbuf(c, (rn * hs * 2) as u64, "qw_att");
    let y = zi::sbuf(c, (rmax * hs * 4) as u64, "qw_y");
    let gu = zi::sbuf(c, (rmax * inter * 2) as u64, "qw_gu");
    let hid = zi::sbuf(c, (rmax * inter * 2) as u64, "qw_hid");
    let cwb = zi::sbuf_init(c, bytemuck::cast_slice(&cw.0), "qw_cw");
    let peb = zi::sbuf_init(c, bytemuck::cast_slice(&pev), "qw_pe");
    let rib = zi::sbuf(c, 16, "qw_ri");
    if let Some(e) = pollster::block_on(sc_oom.pop()) {
        eprintln!("qwen-image wgpu: {e}");
        return decline("the Qwen-Image working set did not fit the card");
    }
    let t = tile();
    let (f16c, f32c) = (site_cfg(t, Epi::F16), site_cfg(t, Epi::F32));
    let fkey = format!("vit_flash_{}_{}_128_{}", fc.nw, fc.bc, zi::flash_thr());
    let pipes = || -> Option<_> {
        Some((
            zi::pipeline(c, "qw_ln", QLN_SRC, "qw_ln")?,
            zi::pipeline(c, "qw_qk", QQK_SRC, "qw_qk")?,
            zi::pipeline(c, &fkey, &vit_flash_src(fc, 128)?, "vit_flash")?,
            zi::pipeline(c, "lx_gelu", GELU_SRC, "lx_gelu")?,
            zi::pipeline(c, "lx_addg", ADDG_SRC, "lx_addg")?,
        ))
    };
    let Some((pln, pqk, pfl, pgelu, paddg)) = pipes() else {
        return decline("a kernel of the Qwen-Image chain was rejected by the device");
    };
    // a product into rows from `orow` of an output `ldo` wide at column `ocol`
    let prod = |g: MmCfg, m: usize, nn: usize, k: usize, ldo: usize, orow: usize, ocol: usize, arow: usize, oscale: f32, w: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer| {
        let args = MmArgs {
            m: m as u32,
            n: nn as u32,
            k: k as u32,
            ldo: ldo as u32,
            ocol: (orow * ldo + ocol) as u32,
            arow: arow as u32,
            oscale,
            conv: [0; 3],
        };
        mm(c, g, &args, 0, w, act, out)
    };
    let ln = |calls: &mut ZCalls, x: &wgpu::Buffer, out: &wgpu::Buffer, rows: usize, sh: u32, sc: u32| {
        let u = zi::ubuf(c, &[hs as u32, sh, sc, 0, 1e-6f32.to_bits(), p2(-gd.act).to_bits(), 0, 0]);
        let bg = zi::bg(c, &pln, &[x, out, &cwb, &u]);
        calls.push(Class::Rows, Call { pipe: pln.clone(), bg, grid: (rows as u32, 1, 1) });
    };
    let addg = |calls: &mut ZCalls, x: &wgpu::Buffer, rows: usize, gate: u32, b: u32| {
        let total = rows * hs;
        let u = zi::ubuf(c, &[total as u32, hs as u32, 1, gate, hs as u32, 0, 0, b, 0, 0, 0, 0]);
        let bg = zi::bg(c, &paddg, &[x, &y, &cwb, &rib, &u]);
        calls.push(Class::Rows, Call { pipe: paddg.clone(), bg, grid: grid1(total) });
    };
    let hs32 = hs as u32;
    let build = |blocks: std::ops::Range<usize>, wb: &[wgpu::Buffer]| -> Option<ZCalls> {
        let mut calls = ZCalls::default();
        let ld = 3 * hs;
        let qs = p2(gd.act - gd.qkv);
        for (j, bi) in blocks.clone().enumerate() {
            let (l, w) = (&lay[bi], &wb[j * QPER..(j + 1) * QPER]);
            // each stream normed and modulated (shift chunk 0, scale chunk 1)
            ln(&mut calls, &xt, &nt_b, nt, l.tm, l.tm + hs32);
            ln(&mut calls, &xi, &ni_b, ni, l.im, l.im + hs32);
            // q, k, v of the text rows, then of the image rows below them
            for (part, wi) in [0usize, 1, 2].into_iter().zip([0usize, 1, 2]) {
                calls.push_mm(Class::MmQkv, prod(f16c, nt, hs, hs, ld, 0, part * hs, 0, qs, &w[wi], &nt_b, &panel)?);
            }
            for (part, wi) in [0usize, 1, 2].into_iter().zip([3usize, 4, 5]) {
                calls.push_mm(Class::MmQkv, prod(f16c, ni, hs, hs, ld, nt, part * hs, 0, qs, &w[wi], &ni_b, &panel)?);
            }
            {
                let u = zi::ubuf(
                    c,
                    &[
                        ld as u32,
                        hs32,
                        hd as u32,
                        heads as u32,
                        nt as u32,
                        n as u32,
                        p2(gd.qkv).to_bits(),
                        p2(-gd.qkv).to_bits(),
                        l.tb,
                        l.ib,
                        l.tn,
                        l.in_,
                        tco,
                        tso,
                        ico,
                        iso,
                        1e-6f32.to_bits(),
                        0,
                        0,
                        0,
                    ],
                );
                let bg = zi::bg(c, &pqk, &[&panel, &cwb, &peb, &u]);
                calls.push(Class::Rows, Call { pipe: pqk.clone(), bg, grid: ((3 * heads) as u32, n as u32, 1) });
            }
            {
                let scl = std::f32::consts::LOG2_E / (hd as f32).sqrt();
                let u = zi::ubuf(c, &[n as u32, n as u32, ld as u32, hs32, (2 * hs) as u32, hs32, scl.to_bits(), 1.0f32.to_bits()]);
                let bg = zi::bg(c, &pfl, &[&panel, &panel, &att, &u]);
                calls.push(Class::Flash, Call { pipe: pfl.clone(), bg, grid: ((n as u32).div_ceil(fc.nw * 16), heads as u32, 1) });
            }
            // each stream's output product (its rows of the attention) and gated residual
            calls.push_mm(Class::MmO, prod(f32c, nt, hs, hs, hs, 0, 0, 0, p2(gd.qkv), &w[6], &att, &y)?);
            addg(&mut calls, &xt, nt, l.tm + 2 * hs32, l.tob);
            calls.push_mm(Class::MmO, prod(f32c, ni, hs, hs, hs, 0, 0, nt, p2(gd.qkv), &w[7], &att, &y)?);
            addg(&mut calls, &xi, ni, l.im + 2 * hs32, l.iob);
            // each stream's MLP (shift chunk 3, scale 4, gate 5)
            for (x, nb_, rows, m, wi, wo, b1, b2) in [(&xt, &nt_b, nt, l.tm, 8usize, 9usize, l.tf1, l.tf2), (&xi, &ni_b, ni, l.im, 10, 11, l.if1, l.if2)] {
                ln(&mut calls, x, nb_, rows, m + 3 * hs32, m + 4 * hs32);
                calls.push_mm(Class::MmW13, prod(f16c, rows, inter, hs, inter, 0, 0, 0, p2(gd.act - gd.gu), &w[wi], nb_, &gu)?);
                let total = rows * inter;
                let u = zi::ubuf(c, &[total as u32, inter as u32, p2(gd.gu).to_bits(), p2(-gd.hid).to_bits(), b1, 0, 0, 0]);
                let bg = zi::bg(c, &pgelu, &[&gu, &hid, &cwb, &u]);
                calls.push(Class::Rows, Call { pipe: pgelu.clone(), bg, grid: grid1(total) });
                calls.push_mm(Class::MmW2, prod(f32c, rows, hs, inter, hs, 0, 0, 0, p2(gd.hid), &w[wo], &hid, &y)?);
                addg(&mut calls, x, rows, m + 5 * hs32, b2);
            }
        }
        Some(calls)
    };

    // ── the step ──
    let uid = model.uid() as usize;
    c.queue.write_buffer(&xt, 0, bytemuck::cast_slice(&*a.text));
    c.queue.write_buffer(&xi, 0, bytemuck::cast_slice(&*a.image));
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut submitted = false;
    for (kept, blocks) in plan.runs() {
        let mine = &spans[blocks.start * QPER..blocks.end * QPER];
        if !kept {
            if submitted {
                zi::wait(c);
            }
            let stale: Vec<(usize, usize)> = spans
                .iter()
                .enumerate()
                .filter(|(i, _)| !plan.keep.contains(&(i / QPER)) && !blocks.contains(&(i / QPER)))
                .map(|(_, (idx, _))| (uid, *idx))
                .collect();
            super::super::weight_evict(c, &stale);
        }
        let wb = match fetch(c, model, mine) {
            Ok(wb) => wb,
            Err(said) => {
                let _ = pollster::block_on(sc.pop());
                return said;
            }
        };
        // a submission a few blocks at a time keeps the command buffers small
        let mut start = blocks.start;
        while start < blocks.end {
            let end = (start + 4).min(blocks.end);
            let Some(calls) = build(start..end, &wb[(start - blocks.start) * QPER..(end - blocks.start) * QPER]) else {
                let _ = pollster::block_on(sc.pop());
                return decline("a kernel of the Qwen-Image chain was rejected by the device");
            };
            let mut enc = c.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                calls.record(&mut pass, None);
            }
            c.queue.submit(Some(enc.finish()));
            submitted = true;
            start = end;
        }
    }
    let rt = zi::read_bytes(c, &xt, (nt * hs * 4) as u64);
    let ri = zi::read_bytes(c, &xi, (ni * hs * 4) as u64);
    if let Some(e) = pollster::block_on(sc.pop()) {
        return decline(&format!("the Qwen-Image chain's command buffer failed: {e}"));
    }
    let (Some(rt), Some(ri)) = (rt, ri) else {
        return decline("the readback failed");
    };
    let (ot, oi): (&[f32], &[f32]) = (bytemuck::cast_slice(&rt), bytemuck::cast_slice(&ri));
    if ot.iter().chain(oi.iter()).any(|x| !x.is_finite()) {
        return decline("the Qwen-Image chain produced a non-finite stream (an f16 range guard overflowed?)");
    }
    a.text.copy_from_slice(ot);
    a.image.copy_from_slice(oi);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chain's kernels are valid WGSL for naga — a rejected module only
    /// shows on a card as "the chain declined".
    #[test]
    fn ltx_kernels_validate() {
        use wgpu::naga;
        let check = |name: &str, src: &str| {
            let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&m)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        };
        check("lx_ada", ADA_SRC);
        check("lx_qk", QK_SRC);
        check("lx_gate", GATE_SRC);
        check("lx_gelu", GELU_SRC);
        check("lx_addg", ADDG_SRC);
        check("vx_ns", VNS_SRC);
        check("vx_add", VADD_SRC);
        check("vx_d2s", VD2S_SRC);
        check("qw_ln", QLN_SRC);
        check("qw_qk", QQK_SRC);
        let g = MmCfg { conv: 6, ..zi::default_cfg(Epi::F32) };
        assert!(g.valid(), "{g:?}");
        check("conv6", &zi::mm_src(g));
        for hd in [64, 128] {
            check(&format!("vit_flash {hd}"), &vit_flash_src(flash_cfg(), hd).unwrap());
        }
    }
}
