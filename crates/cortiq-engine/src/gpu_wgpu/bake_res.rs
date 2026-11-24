//! A whole Phase-A step of the skill bake resident on a wgpu device — the
//! twin of `gpu_cuda::bake_phase_a`: the ids' embeddings go up, every layer
//! (frozen attention — full or GDN —, the masked FFN, both norms), the final
//! norm and the cross-entropy over the scored rows run on the card, and with
//! `grad` the backward of the FFN path (through each frozen attention or GDN
//! branch too under `through_attn`) down to every layer's mask gradient; only
//! the per-target losses, their argmax and the mask gradients come down.
//!
//! It reuses the bake chains' kernels (the GEMMs — coop, or the plain-f32
//! tiles in the strict phase —, `bake_qkr`, `bake_attn_head`, the gate, the
//! attention-core backward, the FFN activations and the mask gradient) and the
//! NVG trainer's GDN operator (`gdn_train`, one tape per layer), and adds the
//! few pieces the chains left to the host: RMSNorm both ways, the residual
//! adds, the scored rows' gather/scatter, the LM head's cross-entropy, and the
//! backward of `bake_qkr` (RoPE and the q/k norm) and of the output gate.
//! Discrete cards only (on unified memory the planes would be RAM).

use super::gdn_train::{self, GdnTape, GdnTrainCfg};
use super::{BAKE_LOCK, Ctx, bake_nn, bake_nt, bake_weight, begin_pass, bind_buf, buffers_fit, const_buf, ctx, finish_enc, flush_pass, submit};
use std::collections::HashMap;
use std::sync::Mutex;

/// A full-attention layer (no projection bias, no projected head gate).
pub struct WgFull<'a> {
    pub wqkv: &'a [f32],
    pub wo: &'a [f32],
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub gated: bool,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    /// (cos, sin) per position and frequency, `[t][half][2]`.
    pub rope: &'a [f32],
    pub half: usize,
    pub window: Option<usize>,
}

/// A GDN layer: the operator's constants and the frozen projections
/// (qkv `[c_dim × hsz]`, z `[vd × hsz]`, a/b `[nv × hsz]`, out `[hsz × vd]`).
pub struct WgGdn<'a> {
    pub cfg: GdnTrainCfg<'a>,
    pub wqkv: &'a [f32],
    pub wz: &'a [f32],
    pub wa: &'a [f32],
    pub wb: &'a [f32],
    pub wout: &'a [f32],
}

pub enum WgAttn<'a> {
    Full(WgFull<'a>),
    Gdn(WgGdn<'a>),
}

pub struct WgLayer<'a> {
    pub iln: &'a [f32],
    pub pln: &'a [f32],
    pub attn: WgAttn<'a>,
    /// `[gate; up]` `[2·inter × hsz]`, down `[hsz × inter]`.
    pub gu: &'a [f32],
    pub down: &'a [f32],
    pub inter: usize,
    /// The mask's gain σ(m) per neuron.
    pub gain: &'a [f32],
}

pub struct WgStep<'a> {
    pub layers: Vec<WgLayer<'a>>,
    pub ids: &'a [u32],
    pub b: usize,
    pub t: usize,
    pub hsz: usize,
    pub eps: f32,
    pub gemma: bool,
    pub act: crate::fcd_ops::FfnAct,
    pub embed: &'a [f32],
    pub final_norm: &'a [f32],
    pub lm: &'a [f32],
    /// Scored target rows (the token predicted from row − 1).
    pub targets: &'a [usize],
    /// The exact classifier: `lm` holds only the class rows and this is each
    /// target's class (`gpu_cuda::BkStep::classes`).
    pub classes: Option<&'a [u32]>,
    pub scale: f32,
    pub grad: bool,
    pub through_attn: bool,
}

/// nll and argmax per scored target, the mask gradient Σ dact·act per layer
/// (empty without `grad`).
pub struct WgOut {
    pub nll: Vec<f32>,
    pub argmax: Vec<u32>,
    pub dmask: Vec<Vec<f32>>,
}

// ───────────────────────────── kernels ─────────────────────────────

const RMS_SRC: &str = r#"
struct P { d: u32, n: u32, eps: f32, gemma: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> inv: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn res_rms(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let r = wid.y * 32768u + wid.x;
    if (r >= p.n) { return; }
    let base = r * p.d;
    var ss = 0.0;
    for (var c = j; c < p.d; c = c + 256u) { let v = x[base + c]; ss = ss + v * v; }
    red[j] = ss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (j < s) { red[j] = red[j] + red[j + s]; }
        workgroupBarrier();
    }
    let iv = 1.0 / sqrt(red[0] / f32(p.d) + p.eps);
    if (j == 0u) { inv[r] = iv; }
    for (var c = j; c < p.d; c = c + 256u) {
        var ww = w[c];
        if (p.gemma != 0u) { ww = 1.0 + ww; }
        y[base + c] = x[base + c] * iv * ww;
    }
}
"#;

const RMS_BWD_SRC: &str = r#"
struct P { d: u32, n: u32, gemma: u32, acc: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> inv: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
fn wv(c: u32) -> f32 { if (p.gemma != 0u) { return 1.0 + w[c]; } return w[c]; }
// dx (+)= inv·w'·dy − x·inv³/d·Σ dy·w'·x
@compute @workgroup_size(256)
fn res_rms_bwd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let r = wid.y * 32768u + wid.x;
    if (r >= p.n) { return; }
    let base = r * p.d;
    var s = 0.0;
    for (var c = j; c < p.d; c = c + 256u) { s = s + dy[base + c] * wv(c) * x[base + c]; }
    red[j] = s;
    workgroupBarrier();
    for (var h = 128u; h > 0u; h = h >> 1u) {
        if (j < h) { red[j] = red[j] + red[j + h]; }
        workgroupBarrier();
    }
    let iv = inv[r];
    let k = red[0] * iv * iv * iv / f32(p.d);
    for (var c = j; c < p.d; c = c + 256u) {
        let g = iv * wv(c) * dy[base + c] - x[base + c] * k;
        if (p.acc != 0u) { dx[base + c] = dx[base + c] + g; } else { dx[base + c] = g; }
    }
}
"#;

const ADD_SRC: &str = r#"
struct P { total: u32, stride: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn res_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * p.stride + gid.x;
    if (i < p.total) { y[i] = a[i] + b[i]; }
}
"#;

const ACC_SRC: &str = r#"
struct P { total: u32, stride: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn res_acc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * p.stride + gid.x;
    if (i < p.total) { y[i] = y[i] + a[i]; }
}
"#;

const RED_SRC: &str = r#"
struct P { total: u32, stride: u32, parts: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
// y = Σ_z part[z][i], the split-K GEMM's partial planes in order
@compute @workgroup_size(256)
fn res_red(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * p.stride + gid.x;
    if (i >= p.total) { return; }
    var acc = part[i];
    for (var z = 1u; z < p.parts; z = z + 1u) { acc = acc + part[z * p.total + i]; }
    y[i] = acc;
}
"#;

/// `bake_mask_grad` over row chunks: partial[ch][j] = Σ_{r in chunk ch}
/// dact·act(g)·u (summed in order by `res_red`) — the one-thread-a-neuron
/// kernel ran 12 workgroups down 256 rows.
const MASKG_SRC: &str = r#"
struct P { inter: u32, n: u32, act: u32, rows: u32 };
@group(0) @binding(0) var<storage, read> plane: array<f32>;
@group(0) @binding(1) var<storage, read> dact: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
fn erf_(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449674) * t
            + 0.2548296)
            * t)
            * exp(-a * a);
    return select(y, -y, x < 0.0);
}
fn act_(g: f32) -> f32 {
    if (p.act == 1u) { return 0.5 * g * (1.0 + erf_(g * 0.70710678)); }
    return g / (1.0 + exp(-g));
}
@compute @workgroup_size(256)
fn res_mask_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= p.inter) { return; }
    let r0 = gid.y * p.rows;
    let r1 = min(r0 + p.rows, p.n);
    var acc = 0.0;
    for (var r = r0; r < r1; r = r + 1u) {
        let g = plane[r * 2u * p.inter + j];
        let u = plane[r * 2u * p.inter + p.inter + j];
        acc = acc + dact[r * p.inter + j] * act_(g) * u;
    }
    part[gid.y * p.inter + j] = acc;
}
"#;

const GATHER_SRC: &str = r#"
struct P { k: u32, d: u32, stride: u32, scatter: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> rows: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
// gather: dst[k] = src[rows[k]]; scatter: dst[rows[k]] += src[k] (rows distinct)
@compute @workgroup_size(256)
fn res_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * p.stride + gid.x;
    if (i >= p.k * p.d) { return; }
    let r = rows[i / p.d];
    let c = i % p.d;
    if (p.scatter == 0u) { dst[i] = src[r * p.d + c]; } else { dst[r * p.d + c] = dst[r * p.d + c] + src[i]; }
}
"#;

const CE_SRC: &str = r#"
struct P { vocab: u32, grad: u32, scale: f32, k: u32 };
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> tg: array<u32>;
@group(0) @binding(2) var<storage, read_write> nll: array<f32>;
@group(0) @binding(3) var<storage, read_write> arg: array<u32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> ro: array<f32, 256>;
// log(1 + x) without the cancellation of log near 1 (Goldberg)
fn log1p_(x: f32) -> f32 {
    let u = 1.0 + x;
    if (u == 1.0) { return x; }
    return log(u) * x / (u - 1.0);
}
// One row of logits [k][vocab] a workgroup: nll = logsumexp − logit[target],
// the argmax (smallest index on ties), and with grad (softmax − onehot)·scale
// in place. The target's own terms come from the sum over the OTHER columns
// (nll = log1p(Σ_other/e_t), dlogit_t = −scale·Σ_other/Σ): when the target is
// near certain, p_t − 1 in f32 keeps few digits and Adam magnifies them.
@compute @workgroup_size(256)
fn res_ce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let r = wid.y * 32768u + wid.x;
    if (r >= p.k) { return; }
    let base = r * p.vocab;
    let t = tg[r];
    var mx = -3.402823e38;
    var am = 0u;
    for (var c = j; c < p.vocab; c = c + 256u) {
        let v = logits[base + c];
        if (v > mx) { mx = v; am = c; }
    }
    rv[j] = mx;
    ri[j] = am;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (j < s) {
            let ov = rv[j + s];
            let oi = ri[j + s];
            if (ov > rv[j] || (ov == rv[j] && oi < ri[j])) { rv[j] = ov; ri[j] = oi; }
        }
        workgroupBarrier();
    }
    mx = rv[0];
    am = ri[0];
    workgroupBarrier();
    var so = 0.0;
    for (var c = j; c < p.vocab; c = c + 256u) {
        if (c != t) { so = so + exp(logits[base + c] - mx); }
    }
    ro[j] = so;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (j < s) { ro[j] = ro[j] + ro[j + s]; }
        workgroupBarrier();
    }
    let others = ro[0];
    let lt = logits[base + t];
    let et = exp(lt - mx);
    let tot = others + et;
    workgroupBarrier();
    if (j == 0u) {
        nll[r] = log1p_(others / et);
        arg[r] = am;
    }
    if (p.grad != 0u) {
        let iv = 1.0 / tot;
        for (var c = j; c < p.vocab; c = c + 256u) {
            if (c == t) {
                logits[base + c] = -p.scale * others * iv;
            } else {
                logits[base + c] = exp(logits[base + c] - mx) * iv * p.scale;
            }
        }
    }
}
"#;

/// `bake_qkr`'s backward: dqrot, dkrot (rotated, normed), dv and dgate back
/// to the projection plane's layout (overwritten), one thread a (row, unit)
/// as the forward: RoPE transposed, then the per-head RMSNorm's backward
/// from the raw projection.
const QKR_BWD_SRC: &str = r#"
struct P {
    n: u32, t: u32, nh: u32, nkv: u32,
    hd: u32, qrows: u32, half: u32, flags: u32,
    eps: f32, _a: u32, _b: u32, _c: u32,
};
@group(0) @binding(0) var<storage, read> plane: array<f32>;
@group(0) @binding(1) var<storage, read> qn: array<f32>;
@group(0) @binding(2) var<storage, read> kn: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<f32>;
@group(0) @binding(4) var<storage, read> dq: array<f32>;
@group(0) @binding(5) var<storage, read> dk: array<f32>;
@group(0) @binding(6) var<storage, read> dv: array<f32>;
@group(0) @binding(7) var<storage, read> dgate: array<f32>;
@group(0) @binding(8) var<storage, read_write> dplane: array<f32>;
@group(0) @binding(9) var<uniform> p: P;
const FL_GATED: u32 = 1u;
const FL_QNORM: u32 = 2u;
const FL_KNORM: u32 = 4u;
const FL_GEMMA: u32 = 8u;
var<private> xb: array<f32, 256>;
var<private> db: array<f32, 256>;

fn rope_t(pos: u32) {
    for (var i = 0u; i < p.half; i = i + 1u) {
        let c = rope[(pos * p.half + i) * 2u];
        let s = rope[(pos * p.half + i) * 2u + 1u];
        let d0 = db[i];
        let d1 = db[i + p.half];
        db[i] = d0 * c + d1 * s;
        db[i + p.half] = -d0 * s + d1 * c;
    }
}

fn norm_bwd(is_q: bool) {
    let hd = p.hd;
    var ss = 0.0;
    var sd = 0.0;
    for (var j = 0u; j < hd; j = j + 1u) {
        var w: f32;
        if (is_q) { w = qn[j]; } else { w = kn[j]; }
        if ((p.flags & FL_GEMMA) != 0u) { w = 1.0 + w; }
        ss = ss + xb[j] * xb[j];
        sd = sd + db[j] * w * xb[j];
    }
    let iv = 1.0 / sqrt(ss / f32(hd) + p.eps);
    let k = sd * iv * iv * iv / f32(hd);
    for (var j = 0u; j < hd; j = j + 1u) {
        var w: f32;
        if (is_q) { w = qn[j]; } else { w = kn[j]; }
        if ((p.flags & FL_GEMMA) != 0u) { w = 1.0 + w; }
        db[j] = iv * w * db[j] - xb[j] * k;
    }
}

@compute @workgroup_size(64)
fn res_qkr_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let units = p.nh + 2u * p.nkv;
    if (gid.x >= p.n * units) { return; }
    let r = gid.x / units;
    let u = gid.x % units;
    let pos = r % p.t;
    let hd = p.hd;
    let qdim = p.nh * hd;
    let kvdim = p.nkv * hd;
    let fused = p.qrows + 2u * kvdim;
    let base = r * fused;
    if (u < p.nh) {
        let h = u;
        var src: u32;
        if ((p.flags & FL_GATED) != 0u) { src = base + 2u * h * hd; } else { src = base + h * hd; }
        for (var j = 0u; j < hd; j = j + 1u) {
            xb[j] = plane[src + j];
            db[j] = dq[r * qdim + h * hd + j];
        }
        rope_t(pos);
        if ((p.flags & FL_QNORM) != 0u) { norm_bwd(true); }
        for (var j = 0u; j < hd; j = j + 1u) { dplane[src + j] = db[j]; }
        if ((p.flags & FL_GATED) != 0u) {
            for (var j = 0u; j < hd; j = j + 1u) { dplane[src + hd + j] = dgate[r * qdim + h * hd + j]; }
        }
    } else if (u < p.nh + p.nkv) {
        let g = u - p.nh;
        let src = base + p.qrows + g * hd;
        for (var j = 0u; j < hd; j = j + 1u) {
            xb[j] = plane[src + j];
            db[j] = dk[r * kvdim + g * hd + j];
        }
        rope_t(pos);
        if ((p.flags & FL_KNORM) != 0u) { norm_bwd(false); }
        for (var j = 0u; j < hd; j = j + 1u) { dplane[src + j] = db[j]; }
    } else {
        let g = u - p.nh - p.nkv;
        let dst = base + p.qrows + kvdim + g * hd;
        for (var j = 0u; j < hd; j = j + 1u) { dplane[dst + j] = dv[r * kvdim + g * hd + j]; }
    }
}
"#;

/// `QKR_BWD_SRC` on 32-wide subgroups, `bake_qkr`'s subgroup geometry: one
/// subgroup per (row, unit), lanes on vec4s, the norm's two sums by
/// subgroupAdd (RTX 3060, Qwen3-0.6B layer: 0.37 → 0.06 ms).
const QKR_BWD_SG_SRC: &str = r#"
struct P {
    n: u32, t: u32, nh: u32, nkv: u32,
    hd: u32, qrows: u32, half: u32, flags: u32,
    eps: f32, _a: u32, _b: u32, _c: u32,
};
@group(0) @binding(0) var<storage, read> plane: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> qn: array<f32>;
@group(0) @binding(2) var<storage, read> kn: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<f32>;
@group(0) @binding(4) var<storage, read> dq: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> dk: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> dv: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> dgate: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> dplane: array<vec4<f32>>;
@group(0) @binding(9) var<uniform> p: P;
const FL_GATED: u32 = 1u;
const FL_QNORM: u32 = 2u;
const FL_KNORM: u32 = 4u;
const FL_GEMMA: u32 = 8u;

fn nw(is_q: bool, c: u32) -> vec4<f32> {
    var w: vec4<f32>;
    if (is_q) { w = vec4<f32>(qn[c], qn[c + 1u], qn[c + 2u], qn[c + 3u]); }
    else { w = vec4<f32>(kn[c], kn[c + 1u], kn[c + 2u], kn[c + 3u]); }
    if ((p.flags & FL_GEMMA) != 0u) { w = w + vec4<f32>(1.0); }
    return w;
}

fn grad4(is_q: bool, g0: u32, c: u32) -> vec4<f32> {
    if (is_q) { return dq[(g0 + c) >> 2u]; }
    return dk[(g0 + c) >> 2u];
}

// the rotated gradient at c … c+3 (from `g0`, the head's start in dq/dk)
// back through RoPE at `pos`
fn unrope4(is_q: bool, g0: u32, c: u32, pos: u32) -> vec4<f32> {
    let half = p.half;
    let d = grad4(is_q, g0, c);
    if (c >= 2u * half) { return d; }
    let i = c % half;
    let rb = (pos * half + i) * 2u;
    let cs = vec4<f32>(rope[rb], rope[rb + 2u], rope[rb + 4u], rope[rb + 6u]);
    let sn = vec4<f32>(rope[rb + 1u], rope[rb + 3u], rope[rb + 5u], rope[rb + 7u]);
    if (c < half) { return d * cs + grad4(is_q, g0, c + half) * sn; }
    return d * cs - grad4(is_q, g0, c - half) * sn;
}

// `bake_qkr`'s backward with its geometry: one subgroup per (row, unit),
// lanes on vec4s; RoPE transposed (partners read again), then the per-head
// RMSNorm's backward from the raw projection, its two sums by subgroupAdd.
@compute @workgroup_size(64)
fn res_qkr_bwd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let units = p.nh + 2u * p.nkv;
    let pair = (wid.y * 32768u + wid.x) * 2u + lid / 32u;
    if (pair >= p.n * units) { return; }
    let lane = lid % 32u;
    let r = pair / units;
    let u = pair % units;
    let pos = r % p.t;
    let hd = p.hd;
    let qdim = p.nh * hd;
    let kvdim = p.nkv * hd;
    let base = r * (p.qrows + 2u * kvdim);
    let ca = lane * 4u;
    let cb = 128u + lane * 4u;
    let oka = ca < hd;
    let okb = cb < hd;
    if (u < p.nh + p.nkv) {
        let is_q = u < p.nh;
        var src: u32;
        var g0: u32;
        var norm: bool;
        if (is_q) {
            if ((p.flags & FL_GATED) != 0u) { src = base + 2u * u * hd; } else { src = base + u * hd; }
            g0 = r * qdim + u * hd;
            norm = (p.flags & FL_QNORM) != 0u;
        } else {
            let g = u - p.nh;
            src = base + p.qrows + g * hd;
            g0 = r * kvdim + g * hd;
            norm = (p.flags & FL_KNORM) != 0u;
        }
        var da = vec4<f32>();
        var db = vec4<f32>();
        if (oka) { da = unrope4(is_q, g0, ca, pos); }
        if (okb) { db = unrope4(is_q, g0, cb, pos); }
        var xa = vec4<f32>();
        var xb = vec4<f32>();
        var wa = vec4<f32>();
        var wb = vec4<f32>();
        var ss = 0.0;
        var sd = 0.0;
        if (norm) {
            if (oka) { xa = plane[(src + ca) >> 2u]; wa = nw(is_q, ca); ss = dot(xa, xa); sd = dot(da * wa, xa); }
            if (okb) { xb = plane[(src + cb) >> 2u]; wb = nw(is_q, cb); ss = ss + dot(xb, xb); sd = sd + dot(db * wb, xb); }
        }
        ss = subgroupAdd(ss);
        sd = subgroupAdd(sd);
        if (norm) {
            let iv = 1.0 / sqrt(ss / f32(hd) + p.eps);
            let k = sd * iv * iv * iv / f32(hd);
            da = iv * wa * da - xa * k;
            db = iv * wb * db - xb * k;
        }
        if (oka) { dplane[(src + ca) >> 2u] = da; }
        if (okb) { dplane[(src + cb) >> 2u] = db; }
        if (is_q && (p.flags & FL_GATED) != 0u) {
            if (oka) { dplane[(src + hd + ca) >> 2u] = dgate[(g0 + ca) >> 2u]; }
            if (okb) { dplane[(src + hd + cb) >> 2u] = dgate[(g0 + cb) >> 2u]; }
        }
    } else {
        let g = u - p.nh - p.nkv;
        let dst = base + p.qrows + kvdim + g * hd;
        if (oka) { dplane[(dst + ca) >> 2u] = dv[(r * kvdim + g * hd + ca) >> 2u]; }
        if (okb) { dplane[(dst + cb) >> 2u] = dv[(r * kvdim + g * hd + cb) >> 2u]; }
    }
}
"#;

const GATE_BWD_SRC: &str = r#"
struct P { total: u32, stride: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read_write> d: array<f32>;
@group(0) @binding(1) var<storage, read> ao: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> dgate: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
// d ← d·σ(g) (the pre-gate ao's gradient), dgate = d·ao·σ(g)(1 − σ(g))
@compute @workgroup_size(256)
fn res_gate_bwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * p.stride + gid.x;
    if (i >= p.total) { return; }
    let s = 1.0 / (1.0 + exp(-gate[i]));
    dgate[i] = d[i] * ao[i] * s * (1.0 - s);
    d[i] = d[i] * s;
}
"#;

struct Pipes {
    rms: wgpu::ComputePipeline,
    rms_bwd: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    acc: wgpu::ComputePipeline,
    red: wgpu::ComputePipeline,
    maskg: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    ce: wgpu::ComputePipeline,
    qkr_bwd: wgpu::ComputePipeline,
    /// `QKR_BWD_SG_SRC` where the device has the subgroup kernels.
    qkr_bwd_sg: Option<wgpu::ComputePipeline>,
    gate_bwd: wgpu::ComputePipeline,
}

fn build_pipes(c: &Ctx) -> Pipes {
    let mk = |src: &str, entry: &'static str| {
        let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &m,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    Pipes {
        rms: mk(RMS_SRC, "res_rms"),
        rms_bwd: mk(RMS_BWD_SRC, "res_rms_bwd"),
        add: mk(ADD_SRC, "res_add"),
        acc: mk(ACC_SRC, "res_acc"),
        red: mk(RED_SRC, "res_red"),
        maskg: mk(MASKG_SRC, "res_mask_grad"),
        gather: mk(GATHER_SRC, "res_gather"),
        ce: mk(CE_SRC, "res_ce"),
        qkr_bwd: mk(QKR_BWD_SRC, "res_qkr_bwd"),
        qkr_bwd_sg: c.bake_sg.as_ref().map(|_| mk(QKR_BWD_SG_SRC, "res_qkr_bwd")),
        gate_bwd: mk(GATE_BWD_SRC, "res_gate_bwd"),
    }
}

/// The step's device state on one card: named buffers grown to the largest
/// call, one GDN tape every GDN layer shares (its forward is recomputed in
/// the backward, as the CUDA step does — a tape per layer is ~80 MB at
/// batch 4 × 64), and each GDN layer's parameters in the tape's PRM layout.
struct State {
    pipes: &'static Pipes,
    bufs: HashMap<String, (u64, wgpu::Buffer)>,
    tape: Option<(TapeKey, GdnTape)>,
    prms: HashMap<(usize, usize, u64), wgpu::Buffer>,
}

/// The shared tape's geometry: (b, t, nv, nk, kk).
type TapeKey = (usize, usize, usize, usize, usize);

/// Per device (`ctx()` is per selected card), under `BAKE_LOCK`.
static STATES: Mutex<Option<HashMap<usize, State>>> = Mutex::new(None);

fn storage(c: &Ctx, floats: usize, label: &'static str) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (floats.max(4) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl State {
    fn buf(&mut self, c: &Ctx, name: &str, floats: usize) -> wgpu::Buffer {
        let bytes = (floats.max(4) * 4) as u64;
        if let Some((_, b)) = self.bufs.get(name).filter(|(cap, _)| *cap >= bytes) {
            return b.clone();
        }
        let cap = bytes.next_power_of_two().max(256);
        let b = storage(c, (cap / 4) as usize, "bake-res");
        self.bufs.insert(name.to_string(), (cap, b.clone()));
        b
    }

    /// `name` holding `data` (written now — lands before the next submit).
    fn upload(&mut self, c: &Ctx, name: &str, data: &[f32]) -> wgpu::Buffer {
        let b = self.buf(c, name, data.len());
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        b
    }

    fn prm(&mut self, c: &Ctx, li: usize, g: &GdnTrainCfg) -> wgpu::Buffer {
        let key = (li, g.a_log.as_ptr() as usize, crate::gpu::fp_bytes(bytemuck::cast_slice(g.conv)));
        if let Some(b) = self.prms.get(&key) {
            return b.clone();
        }
        let mut v = Vec::with_capacity(2 * g.nv + g.norm.len() + g.conv.len());
        v.extend_from_slice(g.a_log);
        v.extend_from_slice(g.dt_bias);
        v.extend_from_slice(g.norm);
        v.extend_from_slice(g.conv);
        let b = storage(c, v.len(), "bake-res-gdn-prm");
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(&v));
        self.prms.retain(|k, _| k.0 != li);
        self.prms.insert(key, b.clone());
        b
    }
}

/// The step's command encoder (one compute pass per dispatch; copies and
/// submits in queue order between them).
struct Enc<'c> {
    c: &'c Ctx,
    enc: wgpu::CommandEncoder,
    pp: &'static Pipes,
    /// The split-K partial planes (`SPLIT_FLOATS`).
    part: wgpu::Buffer,
    split: bool,
}

/// Workgroups a GEMM should launch to fill the card: below this a long-K
/// product with a small output (the 256 × 1024 of a 4 × 64 batch is 64 tiles)
/// splits its K. Two waves of the ~224 tiles an RTX 3060 keeps resident.
const SPLIT_TILES: usize = 448;
/// The partial planes' size: S·nb·rows ≤ (SPLIT_TILES/tiles + 1)·tiles·4096.
const SPLIT_FLOATS: usize = 2 * SPLIT_TILES * 4096;

/// (parts, K per part) for an f32-tile GEMM of `tiles` 64×64 tiles over
/// `cols`: whole K (1, 0) when the tiles fill the card or K is short; each
/// part at least 256 columns, a multiple of 16.
fn split_k(tiles: usize, cols: usize) -> (usize, usize) {
    if tiles >= SPLIT_TILES || cols < 512 {
        return (1, 0);
    }
    let want = SPLIT_TILES.div_ceil(tiles).min(cols / 256).max(1);
    let span = cols.div_ceil(want).div_ceil(16) * 16;
    let parts = cols.div_ceil(span);
    if parts <= 1 { (1, 0) } else { (parts, span) }
}

fn new_enc(c: &Ctx) -> wgpu::CommandEncoder {
    c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bake-res") })
}

impl<'c> Enc<'c> {
    fn uni<T: bytemuck::Pod>(&self, v: &[T]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn run(&mut self, pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer], grid: (u32, u32, u32)) {
        let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| bind_buf(i as u32, b)).collect();
        let bind = self.c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = begin_pass(&mut self.enc);
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
    }

    /// NT: y[nb][rows] = x[nb][cols]·w[rows][cols]ᵀ; NN (the backward's
    /// pipeline): y[nb][rows] = x[nb][cols]·w[cols][rows]. Both overwrite y.
    /// On the f32 tile kernels a product too small to fill the card splits
    /// its K into partial planes and sums them (`CMF_BAKE_SPLITK=0`: never).
    #[allow(clippy::too_many_arguments)]
    fn gemm(&mut self, pipe: &wgpu::ComputePipeline, w: &wgpu::Buffer, x: &wgpu::Buffer, y: &wgpu::Buffer, nb: usize, cols: usize, rows: usize) {
        let (gx, gy) = ((rows as u32).div_ceil(64), (nb as u32).div_ceil(64));
        let f32t = std::ptr::eq(pipe, &self.c.gemm_nt_f32t) || std::ptr::eq(pipe, &self.c.gemm_nn_f32t);
        let (parts, span) = if f32t && self.split { split_k((gx * gy) as usize, cols) } else { (1, 0) };
        if parts > 1 && parts * nb * rows <= SPLIT_FLOATS {
            let u = self.uni(&[(cols / 4) as u32, rows as u32, nb as u32, span as u32]);
            let part = self.part.clone();
            self.run(pipe, &[w, x, &part, &u], (gx, gy, parts as u32));
            let total = nb * rows;
            let (grid, stride) = Self::elem_grid(total);
            let ur = self.uni(&[total as u32, stride, parts as u32, 0u32]);
            let pp = self.pp;
            self.run(&pp.red, &[&part, y, &ur], grid);
            return;
        }
        let u = self.uni(&[(cols / 4) as u32, rows as u32, nb as u32, 0u32]);
        self.run(pipe, &[w, x, y, &u], (gx, gy, 1));
    }

    /// An element-wise kernel's grid (256 a workgroup, folded under 32 768
    /// in x) and its row stride.
    fn elem_grid(total: usize) -> ((u32, u32, u32), u32) {
        let wgs = (total as u32).div_ceil(256);
        let gx = wgs.min(32_768);
        ((gx, wgs.div_ceil(32_768), 1), gx * 256)
    }

    /// One workgroup a row.
    fn rows_grid(n: usize) -> (u32, u32, u32) {
        let n = n as u32;
        (n.min(32_768), n.div_ceil(32_768), 1)
    }

    fn add(&mut self, pp: &Pipes, a: &wgpu::Buffer, b: &wgpu::Buffer, y: &wgpu::Buffer, total: usize) {
        let (grid, stride) = Self::elem_grid(total);
        let u = self.uni(&[total as u32, stride, 0u32, 0u32]);
        self.run(&pp.add, &[a, b, y, &u], grid);
    }

    /// y += a
    fn acc(&mut self, pp: &Pipes, a: &wgpu::Buffer, y: &wgpu::Buffer, total: usize) {
        let (grid, stride) = Self::elem_grid(total);
        let u = self.uni(&[total as u32, stride, 0u32, 0u32]);
        self.run(&pp.acc, &[a, y, &u], grid);
    }

    #[allow(clippy::too_many_arguments)]
    fn rms(&mut self, pp: &Pipes, x: &wgpu::Buffer, w: &wgpu::Buffer, y: &wgpu::Buffer, inv: &wgpu::Buffer, n: usize, d: usize, eps: f32, gemma: bool) {
        let u = self.uni(&[d as u32, n as u32, eps.to_bits(), gemma as u32]);
        self.run(&pp.rms, &[x, w, y, inv, &u], Self::rows_grid(n));
    }

    /// dx (+)= the RMSNorm backward of dy at x.
    #[allow(clippy::too_many_arguments)]
    fn rms_bwd(&mut self, pp: &Pipes, x: &wgpu::Buffer, w: &wgpu::Buffer, inv: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer, n: usize, d: usize, gemma: bool, acc: bool) {
        let u = self.uni(&[d as u32, n as u32, gemma as u32, acc as u32]);
        self.run(&pp.rms_bwd, &[x, w, inv, dy, dx, &u], Self::rows_grid(n));
    }

    fn copy(&mut self, src: &wgpu::Buffer, so: u64, dst: &wgpu::Buffer, d_o: u64, bytes: u64) {
        flush_pass(&self.enc);
        self.enc.copy_buffer_to_buffer(src, so, dst, d_o, bytes);
    }

    /// Submit what is encoded so far (before an operator that submits its
    /// own encoder), and go on in a fresh one.
    fn submit_now(&mut self) {
        let enc = std::mem::replace(&mut self.enc, new_enc(self.c));
        submit(self.c, finish_enc(enc));
    }
}

/// `CMF_BAKE_RES_PROF=1`: the step drains the queue after each section and
/// prints the wall time per section (the drains remove the overlap between
/// sections, so the sum runs a little over the plain step).
struct Prof {
    on: bool,
    t: std::time::Instant,
    secs: Vec<(&'static str, f64)>,
}

impl Prof {
    fn new() -> Self {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("CMF_BAKE_RES_PROF").as_deref() == Ok("1"));
        Self { on, t: std::time::Instant::now(), secs: Vec::new() }
    }

    fn mark(&mut self, e: &mut Enc, name: &'static str) {
        if !self.on {
            return;
        }
        e.submit_now();
        let _ = e.c.device.poll(wgpu::PollType::wait_indefinitely());
        let dt = self.t.elapsed().as_secs_f64() * 1e3;
        match self.secs.iter_mut().find(|(n, _)| *n == name) {
            Some(x) => x.1 += dt,
            None => self.secs.push((name, dt)),
        }
        self.t = std::time::Instant::now();
    }

    fn report(&self) {
        if self.on {
            let total: f64 = self.secs.iter().map(|x| x.1).sum();
            let parts: Vec<String> = self.secs.iter().map(|(n, ms)| format!("{n} {ms:.1}")).collect();
            eprintln!("[bake-res-prof] {total:.1} ms: {}", parts.join(" | "));
        }
    }
}

/// The planes a full-attention forward leaves for its backward.
struct FullPlanes {
    plane: wgpu::Buffer,
    qrot: wgpu::Buffer,
    krot: wgpu::Buffer,
    vproj: wgpu::Buffer,
    gate: wgpu::Buffer,
    aoh: wgpu::Buffer,
    qn: wgpu::Buffer,
    kn: wgpu::Buffer,
    rope: wgpu::Buffer,
}

static ZERO4: [u8; 16] = [0; 16];

fn full_flags(a: &WgFull, gemma: bool) -> u32 {
    (a.gated as u32) | ((a.q_norm.is_some() as u32) << 1) | ((a.k_norm.is_some() as u32) << 2) | ((gemma as u32) << 3)
}

/// A full-attention layer's planes: the shared slots (`tag` empty), or the
/// layer's own when the forward keeps them for the backward (`tag` = layer).
fn full_planes(c: &Ctx, s: &mut State, st: &WgStep, li: usize, a: &WgFull, tag: &str) -> FullPlanes {
    let (t, n, hd) = (st.t, st.b * st.t, a.hd);
    let (qdim, kvdim) = (a.nh * hd, a.nkv * hd);
    let fused = if a.gated { 2 * qdim } else { qdim } + 2 * kvdim;
    FullPlanes {
        plane: s.buf(c, &format!("qkv{tag}"), n * fused),
        qrot: s.buf(c, &format!("qrot{tag}"), n * qdim),
        krot: s.buf(c, &format!("krot{tag}"), n * kvdim),
        vproj: s.buf(c, &format!("vproj{tag}"), n * kvdim),
        // an ungated layer's gate slot is bound but never written
        gate: if a.gated { s.buf(c, &format!("gate{tag}"), n * qdim) } else { s.buf(c, "gate", n * qdim) },
        aoh: s.buf(c, &format!("aoh{tag}"), n * qdim),
        qn: match a.q_norm {
            Some(w) => const_buf(c, bytemuck::cast_slice(&w[..hd])),
            None => const_buf(c, &ZERO4),
        },
        kn: match a.k_norm {
            Some(w) => const_buf(c, bytemuck::cast_slice(&w[..hd])),
            None => const_buf(c, &ZERO4),
        },
        rope: s.buf(c, &format!("rope{li}"), t * a.half * 2),
    }
}

/// Floats a full-attention layer keeps for its backward (`full_planes` with
/// the layer's tag, and its input norm's scale).
fn full_keep(a: &WgFull, n: usize) -> usize {
    let (qdim, kvdim) = (a.nh * a.hd, a.nkv * a.hd);
    let fused = if a.gated { 2 * qdim } else { qdim } + 2 * kvdim;
    n * (fused + qdim + 2 * kvdim + if a.gated { qdim } else { 0 } + qdim + 1)
}

/// The full-attention forward from the normed rows `x` into `y` (the
/// branch output before the residual); None `y`: the planes only.
#[allow(clippy::too_many_arguments)]
fn full_fwd(e: &mut Enc, s: &mut State, st: &WgStep, li: usize, a: &WgFull, nt_p: &wgpu::ComputePipeline, x: &wgpu::Buffer, y: Option<&wgpu::Buffer>, tag: &str) -> FullPlanes {
    let c = e.c;
    let (b, t, hsz, n) = (st.b, st.t, st.hsz, st.b * st.t);
    let (nh, nkv, hd) = (a.nh, a.nkv, a.hd);
    let (qdim, kvdim) = (nh * hd, nkv * hd);
    let qrows = if a.gated { 2 * qdim } else { qdim };
    let fused = qrows + 2 * kvdim;
    let wq = bake_weight(c, a.wqkv, "bake-wqkv");
    let wo = if y.is_some() { Some(bake_weight(c, a.wo, "bake-wo")) } else { None };
    let p = full_planes(c, s, st, li, a, tag);
    let (qi, ki) = (s.buf(c, "qinv", n * nh), s.buf(c, "kinv", n * nkv));
    let zero = s.buf(c, "zero", fused);
    e.gemm(nt_p, &wq, x, &p.plane, n, hsz, fused);
    let u = e.uni(&[n as u32, t as u32, nh as u32, nkv as u32, hd as u32, qrows as u32, a.half as u32, full_flags(a, st.gemma), st.eps.to_bits(), 0, 0, 0]);
    let ka = c.bake_attn(hd, a.half);
    e.run(ka.qkr, &[&p.plane, &p.qn, &p.kn, &zero, &p.rope, &p.qrot, &p.krot, &p.vproj, &p.gate, &qi, &ki, &u], ka.qkr_grid(n * (nh + 2 * nkv)));
    let u2 = e.uni(&[t as u32, nh as u32, nkv as u32, hd as u32, b as u32, a.window.unwrap_or(0) as u32, 0, 0]);
    e.run(ka.head, &[&p.qrot, &p.krot, &p.vproj, &p.aoh, &u2], (t as u32, (b * nh) as u32, 1));
    let (Some(y), Some(wo)) = (y, wo) else { return p };
    if a.gated {
        let aoe = s.buf(c, "aoe", n * qdim);
        let (grid, stride) = Enc::elem_grid(n * qdim);
        let u3 = e.uni(&[(n * qdim) as u32, stride, 0u32, 0u32]);
        e.run(&c.bake_attn_gate, &[&p.aoh, &p.gate, &aoe, &u3], grid);
        e.gemm(nt_p, &wo, &aoe, y, n, qdim, hsz);
    } else {
        e.gemm(nt_p, &wo, &p.aoh, y, n, qdim, hsz);
    }
    p
}

/// The GDN projections of the normed rows `x` into the shared tape (raw
/// qkv, z, a|b) and the layer's parameters into its PRM slot; returns the
/// frozen weights [qkv, z, a, b, out]. The caller submits before the
/// operator runs.
#[allow(clippy::too_many_arguments)]
fn gdn_in(e: &mut Enc, s: &mut State, st: &WgStep, li: usize, g: &WgGdn, tape: &GdnTape, nt_p: &wgpu::ComputePipeline, x: &wgpu::Buffer) -> [wgpu::Buffer; 5] {
    let c = e.c;
    let (hsz, n) = (st.hsz, st.b * st.t);
    let nv = g.cfg.nv;
    let (vd, cdim) = (nv * 128, 2 * g.cfg.nk * 128 + nv * 128);
    let f4 = |x: usize| (x * 4) as u64;
    let ws = gdn_w(c, g);
    let prm = s.prm(c, li, &g.cfg);
    let prm_len = 2 * nv + g.cfg.norm.len() + g.cfg.conv.len();
    let (ta, tb) = (s.buf(c, "gdn-a", n * nv), s.buf(c, "gdn-b", n * nv));
    e.copy(&prm, 0, tape.prm(), 0, f4(prm_len));
    e.gemm(nt_p, &ws[0], x, tape.raw(), n, hsz, cdim);
    e.gemm(nt_p, &ws[1], x, tape.z(), n, hsz, vd);
    e.gemm(nt_p, &ws[2], x, &ta, n, hsz, nv);
    e.gemm(nt_p, &ws[3], x, &tb, n, hsz, nv);
    e.copy(&ta, 0, tape.ab(), 0, f4(n * nv));
    e.copy(&tb, 0, tape.ab(), f4(n * nv), f4(n * nv));
    ws
}

fn gdn_w(c: &Ctx, g: &WgGdn) -> [wgpu::Buffer; 5] {
    [
        bake_weight(c, g.wqkv, "bake-gdn-qkv"),
        bake_weight(c, g.wz, "bake-gdn-z"),
        bake_weight(c, g.wa, "bake-gdn-a"),
        bake_weight(c, g.wb, "bake-gdn-b"),
        bake_weight(c, g.wout, "bake-gdn-out"),
    ]
}

/// A GDN layer's kept inputs: raw qkv, z, a|b (`[n][c_dim]`, `[n][vd]`,
/// `[2][n][nv]`) and its input norm's scale.
fn gdn_keep(g: &WgGdn, n: usize) -> (usize, usize, usize) {
    let nv = g.cfg.nv;
    (n * (2 * g.cfg.nk * 128 + nv * 128), n * nv * 128, 2 * n * nv)
}

/// The step. None: something the resident step does not take, or the device
/// declined — the layer-by-layer path runs.
pub fn phase_a(st: &WgStep) -> Option<WgOut> {
    if std::env::var("CMF_BAKE_GPU").as_deref() == Ok("0") {
        return None;
    }
    let c = ctx()?;
    if !c.discrete {
        return None;
    }
    let (nt_p, nn_p) = (bake_nt(c)?, bake_nn(c)?);
    let (b, t, hsz) = (st.b, st.t, st.hsz);
    let n = b * t;
    if b == 0 || st.ids.len() != n || hsz % 4 != 0 || st.lm.len() % hsz != 0 || st.targets.is_empty() || st.final_norm.len() < hsz {
        return None;
    }
    let vocab = st.lm.len() / hsz;
    let target_of = |k: usize| -> usize { st.classes.map_or(st.ids[st.targets[k]] as usize, |c| c[k] as usize) };
    if vocab % 4 != 0
        || st.classes.is_some_and(|c| c.len() != st.targets.len())
        || st.targets.iter().any(|&r| r == 0 || r >= n)
        || (0..st.targets.len()).any(|k| target_of(k) >= vocab)
        || st.ids.iter().any(|&i| (i as usize + 1) * hsz > st.embed.len())
    {
        return None;
    }
    let lim = c.device.limits();
    let max_bind = lim.max_storage_buffer_binding_size;
    if !buffers_fit(lim.max_buffer_size, &[(st.lm.len() * 4) as u64]) || (st.lm.len() * 4) as u64 > max_bind {
        return None;
    }
    let mut gdn_key = None;
    let mut gdn_first: Option<&GdnTrainCfg> = None;
    for l in &st.layers {
        if l.inter % 4 != 0 || l.gain.len() < l.inter || l.gu.len() < 2 * l.inter * hsz || l.down.len() < hsz * l.inter || l.iln.len() < hsz || l.pln.len() < hsz {
            return None;
        }
        match &l.attn {
            WgAttn::Full(a) => {
                let qrows = if a.gated { 2 * a.nh * a.hd } else { a.nh * a.hd };
                let fused = qrows + 2 * a.nkv * a.hd;
                if a.hd > 256
                    || t > 2048
                    || a.nkv == 0
                    || a.nh % a.nkv != 0
                    || (a.nh * a.hd) % 4 != 0
                    || a.wqkv.len() < fused * hsz
                    || a.wo.len() < hsz * a.nh * a.hd
                    || a.rope.len() < t * a.half * 2
                    || a.q_norm.is_some_and(|w| w.len() < a.hd)
                    || a.k_norm.is_some_and(|w| w.len() < a.hd)
                {
                    return None;
                }
            }
            WgAttn::Gdn(g) => {
                let (nv, nk) = (g.cfg.nv, g.cfg.nk);
                let (vd, cdim) = (nv * 128, 2 * nk * 128 + nv * 128);
                let key = (b, t, nv, nk, g.cfg.kk);
                if nk == 0
                    || nv % nk != 0
                    || nv % 4 != 0
                    || gdn_key.is_some_and(|k| k != key)
                    || g.wqkv.len() < cdim * hsz
                    || g.wz.len() < vd * hsz
                    || g.wa.len() < nv * hsz
                    || g.wb.len() < nv * hsz
                    || g.wout.len() < hsz * vd
                {
                    return None;
                }
                gdn_key = Some(key);
                gdn_first.get_or_insert(&g.cfg);
            }
        }
    }
    let _op = c.op_gate.enter();
    let _bake = BAKE_LOCK.lock().unwrap();
    let mut states = STATES.lock().unwrap();
    let s = states.get_or_insert_with(HashMap::new).entry(c as *const Ctx as usize).or_insert_with(|| State {
        // one set per card for the process
        pipes: Box::leak(Box::new(build_pipes(c))),
        bufs: HashMap::new(),
        tape: None,
        prms: HashMap::new(),
    });
    let mut tape = s.tape.take();
    if let Some((key, cfg)) = gdn_key.zip(gdn_first).filter(|(key, _)| tape.as_ref().is_none_or(|(k, _)| k != key)) {
        drop(tape.take());
        tape = Some((key, gdn_train::gdn_tape(cfg, b, t)?));
    }
    let out = run(st, c, s, nt_p, nn_p, tape.as_ref().map(|(_, tp)| tp), vocab, max_bind);
    s.tape = tape;
    out
}

#[allow(clippy::too_many_arguments)]
fn run(st: &WgStep, c: &Ctx, s: &mut State, nt_p: &wgpu::ComputePipeline, nn_p: &wgpu::ComputePipeline, tape: Option<&GdnTape>, vocab: usize, max_bind: u64) -> Option<WgOut> {
    let pp = s.pipes;
    let (b, t, hsz) = (st.b, st.t, st.hsz);
    let n = b * t;
    let k = st.targets.len();
    let f4 = |x: usize| (x * 4) as u64;
    let max_inter = st.layers.iter().map(|l| l.inter).max().unwrap_or(4);
    let max_fused = st
        .layers
        .iter()
        .filter_map(|l| match &l.attn {
            WgAttn::Full(a) => Some((if a.gated { 2 } else { 1 }) * a.nh * a.hd + 2 * a.nkv * a.hd),
            WgAttn::Gdn(_) => None,
        })
        .max()
        .unwrap_or(4);
    let keep_in = st.grad && st.through_attn;
    // The backward of a frozen branch reads what its forward made: kept per
    // layer within `CMF_BAKE_RES_SAVE_MB` (default 1024), recomputed from
    // the saved layer input otherwise.
    let keep = keep_in && {
        static MB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let mb = *MB.get_or_init(|| std::env::var("CMF_BAKE_RES_SAVE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1024));
        let floats: usize = st
            .layers
            .iter()
            .map(|l| match &l.attn {
                WgAttn::Full(a) => full_keep(a, n),
                WgAttn::Gdn(g) => {
                    let (r, z, ab) = gdn_keep(g, n);
                    r + z + ab + n
                }
            })
            .sum();
        floats * 4 <= mb << 20
    };
    let act_link = match st.act {
        crate::fcd_ops::FfnAct::Silu => &c.bake_silu,
        crate::fcd_ops::FfnAct::GeluErf => &c.bake_gelu,
    };
    let act_bwd_link = match st.act {
        crate::fcd_ops::FfnAct::Silu => &c.bake_silu_bwd,
        crate::fcd_ops::FfnAct::GeluErf => &c.bake_gelu_bwd,
    };
    let gelu = matches!(st.act, crate::fcd_ops::FfnAct::GeluErf) as u32;

    // uploads: the ids' embeddings, the mask gains, the rope tables, the
    // scored rows — all written before the first submit of the step
    let mut h0 = vec![0f32; n * hsz];
    for (r, &id) in st.ids.iter().enumerate() {
        h0[r * hsz..(r + 1) * hsz].copy_from_slice(&st.embed[id as usize * hsz..(id as usize + 1) * hsz]);
    }
    let h = s.upload(c, "h", &h0);
    drop(h0);
    let gains: Vec<wgpu::Buffer> = st.layers.iter().enumerate().map(|(li, l)| s.upload(c, &format!("gain{li}"), &l.gain[..l.inter])).collect();
    for (li, l) in st.layers.iter().enumerate() {
        if let WgAttn::Full(a) = &l.attn {
            s.upload(c, &format!("rope{li}"), &a.rope[..t * a.half * 2]);
        }
    }
    let src_rows: Vec<u32> = st.targets.iter().map(|&r| (r - 1) as u32).collect();
    let tgt: Vec<u32> = (0..k).map(|i| st.classes.map_or(st.ids[st.targets[i]], |c| c[i])).collect();
    let rows_b = s.buf(c, "rows", k);
    c.queue.write_buffer(&rows_b, 0, bytemuck::cast_slice(&src_rows));
    let tgt_b = s.buf(c, "tgt", k);
    c.queue.write_buffer(&tgt_b, 0, bytemuck::cast_slice(&tgt));

    let nrm = s.buf(c, "nrm", n * hsz);
    let inv1 = s.buf(c, "inv1", n);
    let ao = s.buf(c, "ao", n * hsz);
    let act = s.buf(c, "act", n * max_inter);
    static SPLIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let split = *SPLIT.get_or_init(|| std::env::var("CMF_BAKE_SPLITK").as_deref() != Ok("0"));
    let part = s.buf(c, "splitk", SPLIT_FLOATS);
    let mut e = Enc { c, enc: new_enc(c), pp, part, split };
    let mut prof = Prof::new();
    {
        // the zero plane `bake_qkr` adds as its bias
        let z = s.buf(c, "zero", max_fused);
        e.enc.clear_buffer(&z, 0, None);
    }
    prof.mark(&mut e, "upload");

    // ── forward ──
    for (li, l) in st.layers.iter().enumerate() {
        let inter = l.inter;
        let iln = const_buf(c, bytemuck::cast_slice(&l.iln[..hsz]));
        let pln = const_buf(c, bytemuck::cast_slice(&l.pln[..hsz]));
        if keep_in {
            let hin = s.buf(c, &format!("hin{li}"), n * hsz);
            e.copy(&h, 0, &hin, 0, f4(n * hsz));
        }
        let tag = if keep { format!("{li}") } else { String::new() };
        let inv_l = if keep { s.buf(c, &format!("inv1_{li}"), n) } else { inv1.clone() };
        e.rms(pp, &h, &iln, &nrm, &inv_l, n, hsz, st.eps, st.gemma);
        match &l.attn {
            WgAttn::Full(a) => {
                full_fwd(&mut e, s, st, li, a, nt_p, &nrm, Some(&ao), &tag);
                prof.mark(&mut e, "fwd.attn");
            }
            WgAttn::Gdn(g) => {
                let tape = tape?;
                let (nv, nk) = (g.cfg.nv, g.cfg.nk);
                let (vd, cdim) = (nv * 128, 2 * nk * 128 + nv * 128);
                let ws = gdn_in(&mut e, s, st, li, g, tape, nt_p, &nrm);
                if keep {
                    let (r, z, ab) = gdn_keep(g, n);
                    let (kr, kz, kab) = (s.buf(c, &format!("graw{li}"), r), s.buf(c, &format!("gz{li}"), z), s.buf(c, &format!("gab{li}"), ab));
                    e.copy(tape.raw(), 0, &kr, 0, f4(r));
                    e.copy(tape.z(), 0, &kz, 0, f4(z));
                    e.copy(tape.ab(), 0, &kab, 0, f4(ab));
                }
                prof.mark(&mut e, "fwd.gdn.proj");
                e.submit_now();
                gdn_train::gdn_forward_dev(tape, nk, nv, cdim, false)?;
                prof.mark(&mut e, "fwd.gdn.op");
                e.gemm(nt_p, &ws[4], tape.out(), &ao, n, vd, hsz);
                prof.mark(&mut e, "fwd.gdn.out");
            }
        }
        let h1 = s.buf(c, &format!("h1_{li}"), n * hsz);
        let inv2 = s.buf(c, &format!("inv2_{li}"), n);
        let both = if st.grad { s.buf(c, &format!("both{li}"), n * 2 * inter) } else { s.buf(c, "both", n * 2 * inter) };
        e.add(pp, &h, &ao, &h1, n * hsz);
        e.rms(pp, &h1, &pln, &nrm, &inv2, n, hsz, st.eps, st.gemma);
        let wgu = bake_weight(c, l.gu, "bake-gu");
        let wdn = bake_weight(c, l.down, "bake-dn");
        e.gemm(nt_p, &wgu, &nrm, &both, n, hsz, 2 * inter);
        let (grid, stride) = Enc::elem_grid(n * inter);
        let ug = e.uni(&[inter as u32, (n * inter) as u32, 1u32, stride]);
        e.run(act_link, &[&both, &gains[li], &act, &ug], grid);
        e.gemm(nt_p, &wdn, &act, &ao, n, inter, hsz);
        e.add(pp, &h1, &ao, &h, n * hsz);
        prof.mark(&mut e, "fwd.ffn");
    }

    // ── final norm, the scored rows' cross-entropy (the LM head in bands
    // that keep the logits within one binding) ──
    let fnw = const_buf(c, bytemuck::cast_slice(&st.final_norm[..hsz]));
    let finv = s.buf(c, "finv", n);
    e.rms(pp, &h, &fnw, &nrm, &finv, n, hsz, st.eps, st.gemma);
    let hs = s.buf(c, "hs", k * hsz);
    {
        let (grid, stride) = Enc::elem_grid(k * hsz);
        let ug = e.uni(&[k as u32, hsz as u32, stride, 0u32]);
        e.run(&pp.gather, &[&nrm, &rows_b, &hs, &ug], grid);
    }
    let wl = bake_weight(c, st.lm, "bake-lm");
    let band = ((max_bind as usize) / (vocab * 4)).clamp(1, k).min(32_768);
    let logits = s.buf(c, "logits", band * vocab);
    let nll = s.buf(c, "nll", k);
    let arg = s.buf(c, "arg", k);
    let dhs = s.buf(c, "dhs", k * hsz);
    let mut r0 = 0usize;
    while r0 < k {
        let nb = band.min(k - r0);
        let whole = nb == k;
        let (hs_v, tg_v, nll_v, arg_v, dhs_v) = if whole {
            (hs.clone(), tgt_b.clone(), nll.clone(), arg.clone(), dhs.clone())
        } else {
            let v = (s.buf(c, "hs_band", nb * hsz), s.buf(c, "tg_band", nb), s.buf(c, "nll_band", nb), s.buf(c, "arg_band", nb), s.buf(c, "dhs_band", nb * hsz));
            e.copy(&hs, f4(r0 * hsz), &v.0, 0, f4(nb * hsz));
            e.copy(&tgt_b, f4(r0), &v.1, 0, f4(nb));
            v
        };
        e.gemm(nt_p, &wl, &hs_v, &logits, nb, hsz, vocab);
        let uc = e.uni(&[vocab as u32, st.grad as u32, st.scale.to_bits(), nb as u32]);
        e.run(&pp.ce, &[&logits, &tg_v, &nll_v, &arg_v, &uc], Enc::rows_grid(nb));
        prof.mark(&mut e, "head.fwd");
        if st.grad {
            e.gemm(nn_p, &wl, &logits, &dhs_v, nb, vocab, hsz);
            prof.mark(&mut e, "head.bwd");
        }
        if !whole {
            if st.grad {
                e.copy(&dhs_v, 0, &dhs, f4(r0 * hsz), f4(nb * hsz));
            }
            e.copy(&nll_v, 0, &nll, f4(r0), f4(nb));
            e.copy(&arg_v, 0, &arg, f4(r0), f4(nb));
        }
        r0 += nb;
    }

    // ── backward: the FFN path down to every mask, through the frozen
    // branches under `through_attn` ──
    let mut dms: Vec<wgpu::Buffer> = Vec::new();
    if st.grad {
        let dhn = s.buf(c, "dhn", n * hsz);
        let dh = s.buf(c, "dh", n * hsz);
        let dn = s.buf(c, "dn", n * hsz);
        let dn_b = s.buf(c, "dn_b", n * hsz);
        let dact = s.buf(c, "dact", n * max_inter);
        let dgu = s.buf(c, "dgu", n * 2 * max_inter);
        flush_pass(&e.enc);
        e.enc.clear_buffer(&dhn, 0, Some(f4(n * hsz)));
        {
            let (grid, stride) = Enc::elem_grid(k * hsz);
            let us = e.uni(&[k as u32, hsz as u32, stride, 1u32]);
            e.run(&pp.gather, &[&dhs, &rows_b, &dhn, &us], grid);
        }
        e.rms_bwd(pp, &h, &fnw, &finv, &dhn, &dh, n, hsz, st.gemma, false);
        for (li, l) in st.layers.iter().enumerate().rev() {
            let inter = l.inter;
            let wgu = bake_weight(c, l.gu, "bake-gu");
            let wdn = bake_weight(c, l.down, "bake-dn");
            let pln = const_buf(c, bytemuck::cast_slice(&l.pln[..hsz]));
            let (h1, inv2, both) = (s.buf(c, &format!("h1_{li}"), n * hsz), s.buf(c, &format!("inv2_{li}"), n), s.buf(c, &format!("both{li}"), n * 2 * inter));
            let dm = s.buf(c, &format!("dm{li}"), inter);
            // dact2 = dh·down; dgu and the mask gradient; dn2 = dgu·gu
            e.gemm(nn_p, &wdn, &dh, &dact, n, hsz, inter);
            prof.mark(&mut e, "bwd.ffn.dact");
            let (grid, stride) = Enc::elem_grid(n * inter);
            let ug = e.uni(&[inter as u32, (n * inter) as u32, 1u32, stride]);
            e.run(act_bwd_link, &[&both, &dact, &dgu, &ug, &gains[li]], grid);
            // 16 rows a chunk: ≥ 16 chunks over a 4 × 64 batch
            let chunks = n.div_ceil(16).min(SPLIT_FLOATS / inter);
            let rows_per = n.div_ceil(chunks);
            let chunks = n.div_ceil(rows_per);
            let um = e.uni(&[inter as u32, n as u32, gelu, rows_per as u32]);
            let part = e.part.clone();
            e.run(&pp.maskg, &[&both, &dact, &part, &um], ((inter as u32).div_ceil(256), chunks as u32, 1));
            let (grid, stride) = Enc::elem_grid(inter);
            let ur = e.uni(&[inter as u32, stride, chunks as u32, 0u32]);
            e.run(&pp.red, &[&part, &dm, &ur], grid);
            dms.push(dm);
            prof.mark(&mut e, "bwd.ffn.mask");
            e.gemm(nn_p, &wgu, &dgu, &dn, n, 2 * inter, hsz);
            e.rms_bwd(pp, &h1, &pln, &inv2, &dn, &dh, n, hsz, st.gemma, true);
            prof.mark(&mut e, "bwd.ffn.dn");
            if !st.through_attn {
                // dh_in = dh1: the frozen branch passes no gradient
                continue;
            }
            // dh_in = dh1 + the input norm's backward of the branch's dX,
            // the branch recomputed from the saved layer input
            let iln = const_buf(c, bytemuck::cast_slice(&l.iln[..hsz]));
            let hin = s.buf(c, &format!("hin{li}"), n * hsz);
            let inv_l = if keep {
                s.buf(c, &format!("inv1_{li}"), n)
            } else {
                e.rms(pp, &hin, &iln, &nrm, &inv1, n, hsz, st.eps, st.gemma);
                inv1.clone()
            };
            match &l.attn {
                WgAttn::Full(a) => {
                    let (nh, nkv, hd) = (a.nh, a.nkv, a.hd);
                    let (qdim, kvdim) = (nh * hd, nkv * hd);
                    let qrows = if a.gated { 2 * qdim } else { qdim };
                    let fused = qrows + 2 * kvdim;
                    let p = if keep { full_planes(c, s, st, li, a, &format!("{li}")) } else { full_fwd(&mut e, s, st, li, a, nt_p, &nrm, None, "") };
                    prof.mark(&mut e, "bwd.attn.recompute");
                    let wq = bake_weight(c, a.wqkv, "bake-wqkv");
                    let wo = bake_weight(c, a.wo, "bake-wo");
                    let dao = s.buf(c, "dao", n * qdim);
                    let dgt = s.buf(c, "dgate", n * qdim);
                    let (dq, dk, dv) = (s.buf(c, "dq", n * qdim), s.buf(c, "dk", n * kvdim), s.buf(c, "dv", n * kvdim));
                    let stat = s.buf(c, "stat", n * nh * 4);
                    let dplane = s.buf(c, "dqkv", n * fused);
                    // d(ao·σ(gate)) = dh·Wo, then through the output gate
                    e.gemm(nn_p, &wo, &dh, &dao, n, hsz, qdim);
                    if a.gated {
                        let (grid, stride) = Enc::elem_grid(n * qdim);
                        let u3 = e.uni(&[(n * qdim) as u32, stride, 0u32, 0u32]);
                        e.run(&pp.gate_bwd, &[&dao, &p.aoh, &p.gate, &dgt, &u3], grid);
                    }
                    let up = e.uni(&[t as u32, nh as u32, nkv as u32, hd as u32, b as u32, a.window.unwrap_or(0) as u32, 0, 0]);
                    let ka = c.bake_attn(hd, a.half);
                    e.run(ka.bwd_q, &[&p.qrot, &p.krot, &p.vproj, &dao, &dq, &stat, &up], (t as u32, (b * nh) as u32, 1));
                    e.run(ka.bwd_kv, &[&p.qrot, &p.krot, &p.vproj, &dao, &stat, &dk, &dv, &up], (t as u32, (b * nkv) as u32, 1));
                    prof.mark(&mut e, "bwd.attn.core");
                    let uq = e.uni(&[n as u32, t as u32, nh as u32, nkv as u32, hd as u32, qrows as u32, a.half as u32, full_flags(a, st.gemma), st.eps.to_bits(), 0, 0, 0]);
                    let qb = if ka.qkr_sg { pp.qkr_bwd_sg.as_ref() } else { None };
                    e.run(qb.unwrap_or(&pp.qkr_bwd), &[&p.plane, &p.qn, &p.kn, &p.rope, &dq, &dk, &dv, &dgt, &dplane, &uq], if qb.is_some() { ka.qkr_grid(n * (nh + 2 * nkv)) } else { (((n * (nh + 2 * nkv)) as u32).div_ceil(64), 1, 1) });
                    e.gemm(nn_p, &wq, &dplane, &dn, n, fused, hsz);
                    prof.mark(&mut e, "bwd.attn.proj");
                }
                WgAttn::Gdn(g) => {
                    let tape = tape?;
                    let (nv, nk) = (g.cfg.nv, g.cfg.nk);
                    let (vd, cdim) = (nv * 128, 2 * nk * 128 + nv * 128);
                    let ws = if keep {
                        // the forward's inputs back into the shared tape
                        let (r, z, ab) = gdn_keep(g, n);
                        let prm = s.prm(c, li, &g.cfg);
                        let prm_len = 2 * nv + g.cfg.norm.len() + g.cfg.conv.len();
                        e.copy(&prm, 0, tape.prm(), 0, f4(prm_len));
                        e.copy(&s.buf(c, &format!("graw{li}"), r), 0, tape.raw(), 0, f4(r));
                        e.copy(&s.buf(c, &format!("gz{li}"), z), 0, tape.z(), 0, f4(z));
                        e.copy(&s.buf(c, &format!("gab{li}"), ab), 0, tape.ab(), 0, f4(ab));
                        gdn_w(c, g)
                    } else {
                        gdn_in(&mut e, s, st, li, g, tape, nt_p, &nrm)
                    };
                    // the upstream gradient dof = dh·Wout into the tape's slot
                    e.gemm(nn_p, &ws[4], &dh, tape.dout(), n, hsz, vd);
                    prof.mark(&mut e, "bwd.gdn.proj");
                    e.submit_now();
                    gdn_train::gdn_forward_dev(tape, nk, nv, cdim, true)?;
                    prof.mark(&mut e, "bwd.gdn.fwd");
                    gdn_train::gdn_backward_dev(tape, nk, nv, cdim)?;
                    prof.mark(&mut e, "bwd.gdn.bptt");
                    // dn1 = draw·Wqkv + dz·Wz + da·Wa + db·Wb
                    let (ta, tb) = (s.buf(c, "gdn-a", n * nv), s.buf(c, "gdn-b", n * nv));
                    e.copy(tape.dab(), 0, &ta, 0, f4(n * nv));
                    e.copy(tape.dab(), f4(n * nv), &tb, 0, f4(n * nv));
                    e.gemm(nn_p, &ws[0], tape.draw(), &dn, n, cdim, hsz);
                    e.gemm(nn_p, &ws[1], tape.dz(), &dn_b, n, vd, hsz);
                    e.acc(pp, &dn_b, &dn, n * hsz);
                    e.gemm(nn_p, &ws[2], &ta, &dn_b, n, nv, hsz);
                    e.acc(pp, &dn_b, &dn, n * hsz);
                    e.gemm(nn_p, &ws[3], &tb, &dn_b, n, nv, hsz);
                    e.acc(pp, &dn_b, &dn, n * hsz);
                    prof.mark(&mut e, "bwd.gdn.dn");
                }
            }
            e.rms_bwd(pp, &hin, &iln, &inv_l, &dn, &dh, n, hsz, st.gemma, true);
        }
        dms.reverse();
    }

    // ── one readback: nll, argmax, the mask gradients ──
    let mut offs: Vec<(u64, u64)> = vec![(0, f4(k))];
    let next = |o: &[(u64, u64)]| o.last().map(|(a, l)| (a + l).div_ceil(16) * 16).unwrap_or(0);
    offs.push((next(&offs), f4(k)));
    for dm_len in st.layers.iter().map(|l| f4(l.inter)).take(dms.len()) {
        offs.push((next(&offs), dm_len));
    }
    let need = next(&offs);
    let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bake-res-stage"),
        size: need,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    e.copy(&nll, 0, &stage, offs[0].0, offs[0].1);
    e.copy(&arg, 0, &stage, offs[1].0, offs[1].1);
    for (i, dm) in dms.iter().enumerate() {
        e.copy(dm, 0, &stage, offs[2 + i].0, offs[2 + i].1);
    }
    e.submit_now();
    let slice = stage.slice(..need);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let out = {
        let data = slice.get_mapped_range().ok()?;
        let at = |(o, len): (u64, u64)| &data[o as usize..(o + len) as usize];
        WgOut {
            nll: bytemuck::cast_slice(at(offs[0])).to_vec(),
            argmax: bytemuck::cast_slice(at(offs[1])).to_vec(),
            dmask: (0..dms.len()).map(|i| bytemuck::cast_slice(at(offs[2 + i])).to_vec()).collect(),
        }
    };
    stage.unmap();
    prof.mark(&mut e, "readback");
    prof.report();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{Enc, SPLIT_FLOATS, SPLIT_TILES, build_pipes, ctx, new_enc, split_k, storage};

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn read(c: &super::Ctx, b: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let st = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = new_enc(c);
        enc.copy_buffer_to_buffer(b, 0, &st, 0, (n * 4) as u64);
        c.queue.submit(Some(enc.finish()));
        let sl = st.slice(..);
        sl.map_async(wgpu::MapMode::Read, |_| {});
        c.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let v = bytemuck::cast_slice(&sl.get_mapped_range().unwrap()).to_vec();
        v
    }

    /// The split-K product against an f64 reference and the whole-K one, on
    /// a discrete card (`CMF_GPU=1`): the long-K shapes of the bake's head
    /// and FFN backward.
    #[test]
    fn split_k_gemm_matches_f64() {
        if std::env::var("CMF_GPU").is_err() {
            eprintln!("no backend requested — skip");
            return;
        }
        let Some(c) = ctx().filter(|c| c.discrete) else {
            eprintln!("no discrete wgpu card — skip");
            return;
        };
        let pp: &'static super::Pipes = Box::leak(Box::new(build_pipes(c)));
        // (nn, nb, cols, rows): the head's dhs (K = a vocabulary), a down/dgu
        // product, an NT product with a short K
        for &(nn, nb, cols, rows) in &[(true, 64usize, 65_536usize, 256usize), (true, 256, 7168, 1024), (false, 256, 3584, 1024), (false, 252, 1024, 2048)] {
            let x = synth(nb * cols, 1);
            let w = synth(cols * rows, 2);
            let mut r64 = vec![0f64; nb * rows];
            for m in 0..nb {
                for k in 0..cols {
                    let xv = x[m * cols + k] as f64;
                    let wr = if nn { &w[k * rows..(k + 1) * rows] } else { &w[..0] };
                    for q in 0..rows {
                        let wv = if nn { wr[q] as f64 } else { w[q * cols + k] as f64 };
                        r64[m * rows + q] += xv * wv;
                    }
                }
            }
            let (xb, wb, yb) = (storage(c, nb * cols, "t"), storage(c, cols * rows, "t"), storage(c, nb * rows, "t"));
            c.queue.write_buffer(&xb, 0, bytemuck::cast_slice(&x));
            c.queue.write_buffer(&wb, 0, bytemuck::cast_slice(&w));
            let pipe = if nn { &c.gemm_nn_f32t } else { &c.gemm_nt_f32t };
            let mut errs = Vec::new();
            for split in [false, true] {
                let mut e = Enc { c, enc: new_enc(c), pp, part: storage(c, SPLIT_FLOATS, "t"), split };
                e.gemm(pipe, &wb, &xb, &yb, nb, cols, rows);
                e.submit_now();
                let y = read(c, &yb, nb * rows);
                let (mut num, mut den) = (0f64, 0f64);
                for (a, r) in y.iter().zip(&r64) {
                    num += (*a as f64 - r) * (*a as f64 - r);
                    den += r * r;
                }
                errs.push((num / den).sqrt());
            }
            let tiles = rows.div_ceil(64) * nb.div_ceil(64);
            eprintln!("{} nb {nb} cols {cols} rows {rows}: split {:?} | rel. error whole K {:.3e}, split {:.3e}", if nn { "NN" } else { "NT" }, split_k(tiles, cols), errs[0], errs[1]);
            assert!(errs[0] < 1e-5 && errs[1] < 1e-5, "{errs:?}");
            assert!(errs[1] <= errs[0] * 1.05, "split-K less accurate than whole K: {errs:?}");
        }
    }

    #[test]
    fn split_k_covers_k_once_in_aligned_parts() {
        for &(rows, nb, cols) in &[(1024usize, 256usize, 3584usize), (1024, 256, 248_070 + 2), (2048, 256, 1024), (16, 256, 1024), (1024, 256, 16), (7168, 256, 1024), (1024, 16, 6144)] {
            let tiles = rows.div_ceil(64) * nb.div_ceil(64);
            let (parts, span) = split_k(tiles, cols);
            if parts == 1 {
                assert_eq!(span, 0);
                continue;
            }
            assert_eq!(span % 16, 0, "{rows}x{nb}x{cols}");
            assert!(span >= 256 && (parts - 1) * span < cols && parts * span >= cols, "{rows}x{nb}x{cols}: {parts} × {span}");
            assert!(parts * tiles <= 2 * SPLIT_TILES + tiles);
            assert!(parts * nb * rows <= SPLIT_FLOATS, "{rows}x{nb}x{cols}");
        }
        // a product that fills the card keeps its K whole
        assert_eq!(split_k(SPLIT_TILES, 1 << 20), (1, 0));
    }
}
