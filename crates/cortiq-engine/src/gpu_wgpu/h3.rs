//! MiniMax-H3 stacks on wgpu/Vulkan as resident chains: the DiT of
//! MiniMax-H3 / MiniMax-H3-Image with its prompt refiner ([`chain`]) and
//! the prompt encoder ([`te_chain`]).
//!
//! Both serve stacks whose projections are q4tp tensors with 128-wide
//! heads. The residual stream goes up once and comes back once; between
//! them every block runs on the card:
//!
//! - `h3_rowop`: the gated residual `x += g ⊙ y` and the next RMS norm with
//!   its adaLN modulation, `xn = f16((x·inv·w·(1 + scale) + shift)·2^-act)`,
//!   one pass over the row (a token's modulation row is looked up per row;
//!   the encoder's residuals are plain and its norms unmodulated);
//! - the products through `zi_mm` (`zimage.rs`: tensor cores, f32
//!   accumulation) reading the q4tp tensor itself ([`MmCfg::q4`]) — the
//!   weights stay on the card in their codec, 222 MB a block of the 20 B
//!   DiT, and no f16 plane is ever made. The DiT's qkv writes the f16 panel,
//!   its output projection and fc2 write f32, fc1 runs the fused SwiGLU
//!   epilogue on the tensor's `[gate | up]` rows ([`MmCfg::q4h`]). The
//!   encoder's q, k and v are three tensors into one `[q | k | v]` panel,
//!   its gate and up two tensors into one panel that `h3_swiglu` folds;
//! - `h3_qkrope`: per-head RMS norm·w and the split-half rotation of the
//!   first `2·pairs` dims, in place on the panel's q and k columns;
//! - `qi_flash` (`qi21.rs`): flash attention straight from the panel, any
//!   row count (keys past the sequence are masked exactly). The encoder
//!   runs it causal (query `i` sees keys `[0, i]`) and grouped (a key head
//!   serves `heads / kv_heads` query heads).
//!
//! Range guards, as powers of two the consumer multiplies back: the normed
//! activations ×2⁻², the qkv panel ×2⁻³ (the q/k norm is scale-free with
//! its eps scaled, v carries the factor to the output projection), the
//! SwiGLU middle ×2⁻⁸, the encoder's gate and up panel ×2⁻⁴. Stored maxima
//! of the image DiT at 1024² (`CMF_H3_WGPU_AMAX=1`, six steps): activations
//! 23, qkv 276, attention output 101, the middle 4332 with a 2⁻⁶ guard —
//! every site two orders below the f16 limit. `CMF_H3_WGPU_{ACT,QKV,HID,
//! GU}_SHIFT` override. A non-finite result makes the chain stand down and
//! the per-op path redo the call.
//!
//! Measured, RTX A5000 (24 GB, NVIDIA 595.91, Vulkan; pure-MMA ceiling
//! 103 TF), `zimage_gemmbench h3`: the four products of a DiT block take
//! 17.1 ms at 1178 rows (1024² and its prompt) and 57.2 ms at 4250 rows
//! (2048²), 50–58 TF each; over an f16 plane the same tile is 39–55 TF, and
//! wider tiles (256 rows) 10–16 TF. Attention is 0.8 and 10.6 ms. A whole
//! step of the DiT: 1.01 s at 1024² and 4.0 s at 2048², against 10.2 and
//! 36.2 s through the per-op kernels.
//!
//! **Memory is a ceiling.** A stack's weights stay on the card in their
//! codec between calls when they fit the adapter's weight budget beside the
//! activations (10 GB of the DiT, 12.7 GB of the encoder, one stage at a
//! time). When they do not — a smaller card, or `CMF_GPU_VRAM_MB=N` — the
//! first blocks that fit are kept and the rest pass through a slot of at
//! most 1 GB in runs: read from the file, run, dropped, every call
//! ([`plan`]). The stream never leaves the card for it.
//!
//! Knobs: `CMF_H3_WGPU=0` (the per-op path for both stacks),
//! `CMF_H3_WGPU_TE=0` (for the encoder), `CMF_H3_WGPU_PROF=1` (device time
//! per kernel class after a call: each class alone, 3 repetitions),
//! `CMF_H3_WGPU_TILE=bm,bn,bk,wm,wn`, `CMF_H3_WGPU_FLASH=nw,bc`.

use crate::gpu::{H3Chain, TeChain};
use cortiq_core::{CmfModel, TensorDtype};
use std::sync::{Arc, Mutex};

use super::qi21;
use super::zimage::{self as zi, Call, Class, Epi, FlashCfg, MmArgs, MmCall, MmCfg, ZCalls};
use super::Ctx;

/// Rows one dispatch may address (the row kernels index rows by
/// `workgroup_id`).
const MAX_ROWS: usize = 65_535;
/// Chunks of a block's modulation row: shift, scale, gate for the attention
/// half, then for the MLP.
const EXPAND: usize = 6;

fn enabled() -> bool {
    std::env::var("CMF_H3_WGPU").as_deref() != Ok("0")
}

/// Whether this device can run the chains at all.
pub(crate) fn available() -> bool {
    enabled() && zi::zctx().is_some()
}

/// Say once why a chain stood down (the caller runs its per-op path).
fn decline(reason: &str) -> bool {
    static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
    if let Ok(mut v) = SAID.lock()
        && !v.iter().any(|r| r == reason)
    {
        eprintln!("minimax-h3: wgpu chain declined: {reason}");
        v.push(reason.to_string());
    }
    false
}

/// A stack the chains are not written for (the VAE's ViT: biases, 64-wide
/// heads, tile batches): no message unless asked.
fn not_mine(reason: &str) -> bool {
    if std::env::var("CMF_GPU_DEBUG").is_ok() {
        decline(reason)
    } else {
        false
    }
}

#[derive(Clone, Copy, Debug)]
struct Guards {
    act: i32,
    qkv: i32,
    hid: i32,
    /// The prompt encoder's gate and up panel.
    gu: i32,
}

impl Guards {
    fn from_env() -> Guards {
        let e = |k: &str, d: i32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d).clamp(0, 14);
        Guards {
            act: e("CMF_H3_WGPU_ACT_SHIFT", 2),
            qkv: e("CMF_H3_WGPU_QKV_SHIFT", 3),
            hid: e("CMF_H3_WGPU_HID_SHIFT", 8),
            gu: e("CMF_H3_WGPU_GU_SHIFT", 4),
        }
    }
}

fn p2(e: i32) -> f32 {
    (2.0f32).powi(e)
}

// ───────────────────────────── kernels ─────────────────────────────

const ROW_RES: u32 = 1;
const ROW_NORM: u32 = 2;
const ROW_MOD: u32 = 4;
const ROW_BIAS: u32 = 8;

/// mode bit 0: `x += g ⊙ y` (g = the row's gate chunk under bit 2, else 1;
/// under bit 3 `x += s ⊙ (y + b)` with the bias at `g_base` and the scale
/// at `sh_base` of `cw` — a ViT's projection bias and layer scale);
/// bit 1: `xn = f16((x·inv·w·(1 + scale) + shift)·oscale)` with
/// `inv = 1/sqrt(mean(x²) + eps)` (scale and shift under bit 2, else the
/// plain weighted norm). One workgroup per row.
const ROWOP_SRC: &str = r#"
enable f16;
struct RP { h: u32, mode: u32, g_base: u32, sh_base: u32, sc_base: u32, w_off: u32, eps: f32, oscale: f32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xn: array<f16>;
@group(0) @binding(3) var<storage, read> mods: array<f32>;
@group(0) @binding(4) var<storage, read> mrow: array<u32>;
@group(0) @binding(5) var<storage, read> cw: array<f32>;
@group(0) @binding(6) var<uniform> rp: RP;
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
fn h3_rowop(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let base = row * rp.h;
    let modded = (rp.mode & 4u) != 0u;
    var mo = 0u;
    if (modded) { mo = mrow[row] * 6u * rp.h; }
    var ss = 0.0;
    for (var i = lid; i < rp.h; i = i + 256u) {
        var v = x[base + i];
        if ((rp.mode & 1u) != 0u) {
            var g = 1.0;
            var bb = 0.0;
            if (modded) { g = mods[rp.g_base + mo + i]; }
            if ((rp.mode & 8u) != 0u) { bb = cw[rp.g_base + i]; g = cw[rp.sh_base + i]; }
            v = v + g * (y[base + i] + bb);
            x[base + i] = v;
        }
        ss = ss + v * v;
    }
    if ((rp.mode & 2u) == 0u) { return; }
    let inv = inverseSqrt(wsum(ss, lid) / f32(rp.h) + rp.eps);
    for (var i = lid; i < rp.h; i = i + 256u) {
        var v = x[base + i] * inv * cw[rp.w_off + i];
        if (modded) { v = v * (1.0 + mods[rp.sc_base + mo + i]) + mods[rp.sh_base + mo + i]; }
        xn[base + i] = f16(v * rp.oscale);
    }
}
"#;

/// One workgroup (two dims a lane) per (token, head) of the f16 panel —
/// `nq` query heads from column 0, then `nk` key heads from `k_col`: RMS
/// norm over the head's `hd` dims (eps from `QP`), the head's weight, then
/// dims `j` and `j + pairs` (`j < pairs`) turn by the token's angle `j` —
/// `(cos, sin)` pairs in `cs`, a table per sequence of `nseq` rows (0: one
/// row a token). Dims past `2·pairs` only norm. With a bias (`b_off` into
/// `cw`, `[3·heads·hd]` in the panel's column order, times `bscale`) every
/// part gets it first, and the value heads from `v_col` are dispatched too
/// (they only take the bias).
const QKROPE_SRC: &str = r#"
struct QP { ld: u32, nq: u32, eps: f32, pairs: u32, w_off: u32, k_col: u32, hd: u32, nk: u32, b_off: u32, bscale: f32, nseq: u32, v_col: u32 };
@group(0) @binding(0) var<storage, read_write> qkv: array<u32>;
@group(0) @binding(1) var<storage, read> cw: array<f32>;
@group(0) @binding(2) var<storage, read> cs: array<f32>;
@group(0) @binding(3) var<uniform> qp: QP;
var<workgroup> red: array<f32, 64>;
var<workgroup> nv: array<f32, 128>;

fn turn(d: u32, row: u32, own: f32) -> f32 {
    let np = qp.pairs;
    if (d >= 2u * np) { return own; }
    let j = select(d - np, d, d < np);
    let c = cs[(row * np + j) * 2u];
    let s = cs[(row * np + j) * 2u + 1u];
    let lo = nv[j];
    let hi = nv[j + np];
    return select(lo * s + hi * c, lo * c - hi * s, d < np);
}

@compute @workgroup_size(64)
fn h3_qkrope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.y;
    let hv = wid.x;
    let hd = qp.hd;
    var part = 0u;
    var col = hv * hd;
    if (hv >= qp.nq) { part = 1u; col = qp.k_col + (hv - qp.nq) * hd; }
    if (hv >= qp.nq + qp.nk) { part = 2u; col = qp.v_col + (hv - qp.nq - qp.nk) * hd; }
    let live = 2u * lid < hd;
    let idx = (row * qp.ld + col) / 2u + lid;
    var v = vec2<f32>(0.0);
    if (live) {
        v = unpack2x16float(qkv[idx]);
        if (qp.b_off != 0xFFFFFFFFu) {
            let bo = qp.b_off + col + 2u * lid;
            v = v + vec2<f32>(cw[bo], cw[bo + 1u]) * qp.bscale;
        }
    }
    red[lid] = v.x * v.x + v.y * v.y;
    workgroupBarrier();
    var st = 32u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let rr = inverseSqrt(red[0] / f32(hd) + qp.eps);
    var a = v.x;
    var b = v.y;
    if (live && part < 2u) {
        let wo = qp.w_off + part * hd + 2u * lid;
        a = v.x * rr * cw[wo];
        b = v.y * rr * cw[wo + 1u];
    }
    nv[2u * lid] = a;
    nv[2u * lid + 1u] = b;
    workgroupBarrier();
    if (live) {
        var o = vec2<f32>(a, b);
        if (part < 2u) {
            var crow = row;
            if (qp.nseq != 0u) { crow = row % qp.nseq; }
            o = vec2<f32>(turn(2u * lid, crow, a), turn(2u * lid + 1u, crow, b));
        }
        qkv[idx] = pack2x16float(o);
    }
}
"#;

/// `hid = f16(silu(g)·u·oscale)` from a `[gate | up]` f16 panel stored
/// divided by `gin`, plus the projection's bias `[gate | up]` at `b_off` of
/// `cw` when it has one. One thread an element.
const SWIGLU_SRC: &str = r#"
enable f16;
struct SP { n: u32, inter: u32, gin: f32, oscale: f32, b_off: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> gu: array<f16>;
@group(0) @binding(1) var<storage, read_write> hid: array<f16>;
@group(0) @binding(2) var<storage, read> cw: array<f32>;
@group(0) @binding(3) var<uniform> sp: SP;

@compute @workgroup_size(256)
fn h3_swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * (65535u * 256u) + gid.x;
    if (i >= sp.n) { return; }
    let row = i / sp.inter;
    let col = i - row * sp.inter;
    let b = row * 2u * sp.inter + col;
    var g = f32(gu[b]) * sp.gin;
    var u = f32(gu[b + sp.inter]) * sp.gin;
    if (sp.b_off != 0xFFFFFFFFu) {
        g = g + cw[sp.b_off + col];
        u = u + cw[sp.b_off + sp.inter + col];
    }
    hid[i] = f16(g / (1.0 + exp(-g)) * u * sp.oscale);
}
"#;

/// The GEMM tile: 128×128×32 with 2×2 subgroups. A q4 kernel needs a row a
/// thread (`bn` = the thread count); on the RTX A5000 the 256-row tiles are
/// 4–5 times slower and 128×128 with 4×1 subgroups ties.
fn tile() -> (u32, u32, u32, u32, u32) {
    if let Ok(s) = std::env::var("CMF_H3_WGPU_TILE") {
        let v: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 5 && site_cfg((v[0], v[1], v[2], v[3], v[4]), Epi::F32).valid() {
            return (v[0], v[1], v[2], v[3], v[4]);
        }
    }
    (128, 128, 32, 2, 2)
}

fn site_cfg(t: (u32, u32, u32, u32, u32), epi: Epi) -> MmCfg {
    MmCfg { q4: true, q4h: epi.is_swiglu(), ..MmCfg::new(t.0, t.1, t.2, t.3, t.4, epi) }
}

/// The attention tile (`nw` subgroups × 16 queries, `bc` keys a block):
/// 4×16 on the RTX A5000 (0.82 ms at 1184 rows and 10.6 ms at 4256, against
/// 1.17 / 14.2 for 8×16 and 2.20 / 14.3 for 4×32, 56 heads).
fn flash_cfg() -> FlashCfg {
    if let Ok(s) = std::env::var("CMF_H3_WGPU_FLASH") {
        let v: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 2 && qi21::flash_src(FlashCfg { nw: v[0], bc: v[1] }).is_some() {
            return FlashCfg { nw: v[0], bc: v[1] };
        }
    }
    FlashCfg { nw: 4, bc: 16 }
}

/// `qi_flash` whose keys and values are shared by `rep` query heads each
/// (grouped-query attention): head `h` reads key head `h / rep`.
fn flash_src(f: FlashCfg, rep: usize) -> Option<String> {
    let s = qi21::flash_src(f)?;
    if rep == 1 {
        return Some(s);
    }
    let own = "let kq4 = (p.k_col + h * 128u) / 4u; let vq4 = (p.v_col + h * 128u) / 4u;";
    let shared =
        format!("let kq4 = (p.k_col + (h / {rep}u) * 128u) / 4u; let vq4 = (p.v_col + (h / {rep}u) * 128u) / 4u;");
    s.contains(own).then(|| s.replace(own, &shared))
}

/// `QP` of `h3_qkrope`: the panel's row width, the query and key head
/// counts, the norm's eps, the rotated pairs, the q/k norm weights' offset
/// in `cw`, the key columns' start, the head width, and for a stack with a
/// qkv bias its offset in `cw` and its scale; `nseq`: rows of one sequence
/// of a batch (0: the rotation table has a row a token).
#[allow(clippy::too_many_arguments)]
fn qk_uniform(
    c: &Ctx,
    ld: usize,
    nq: usize,
    nk: usize,
    eps: f32,
    pairs: usize,
    w_off: usize,
    k_col: usize,
    hd: usize,
    bias: Option<(usize, f32, usize)>,
    nseq: usize,
) -> wgpu::Buffer {
    let (b_off, bscale, v_col) = bias.map_or((u32::MAX, 0.0, 0), |(o, s, v)| (o as u32, s, v as u32));
    zi::ubuf(
        c,
        &[
            ld as u32,
            nq as u32,
            eps.to_bits(),
            pairs as u32,
            w_off as u32,
            k_col as u32,
            hd as u32,
            nk as u32,
            b_off,
            bscale.to_bits(),
            nseq as u32,
            v_col,
        ],
    )
}

/// `zi_mm` with the uniform's last word set (`Epi::SwiGluIn`'s input
/// multiplier; 0 for the other epilogues).
fn mm(c: &Ctx, g: MmCfg, a: &MmArgs, extra: u32, w: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer) -> Option<MmCall> {
    if !a.n.is_multiple_of(g.bn) || !a.k.is_multiple_of(g.bk) {
        return None;
    }
    let pipe = zi::mm_pipe(c, g)?;
    let u = zi::ubuf(c, &[a.m, a.n, a.k, a.ldo, a.ocol, a.arow, a.oscale.to_bits(), extra]);
    let bg = zi::bg(c, &pipe, &[w, act, out, &u]);
    Some(MmCall {
        pipe,
        bg,
        grid: (a.n / g.bn, a.m.div_ceil(g.bm)),
        mnk: [a.m, a.n, a.k],
        bits: false,
    })
}

/// A product `[m][k] · [n][k]ᵀ` written at column `ocol` of rows `ldo` wide.
fn margs(m: usize, n: usize, k: usize, ldo: usize, ocol: usize, oscale: f32) -> MmArgs {
    MmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ldo: ldo as u32,
        ocol: ocol as u32,
        arow: 0,
        oscale,
        conv: [0; 3],
    }
}

// ───────────────────────────── state ─────────────────────────────

/// Row widths of a stack's activations: the stream, the q/k/v panel, the
/// attention output, the MLP's middle, and the `[gate | up]` panel of a
/// stack whose SwiGLU is not a GEMM epilogue (0 otherwise).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Widths {
    hs: usize,
    panel: usize,
    att: usize,
    mid: usize,
    gu: usize,
}

impl Widths {
    fn bytes(&self, mp: usize) -> u64 {
        (mp * (10 * self.hs + 2 * (self.panel + self.att + self.mid + self.gu))) as u64
    }
}

/// Activations of a chain call, `mp = pad_rows(rows)` rows each so the
/// unchecked kernels' tile over-reads stay inside. Kept between calls and
/// regrown when a call needs more rows or another geometry.
struct Acts {
    mp: usize,
    w: Widths,
    x: wgpu::Buffer,     // f32 [mp][hs]
    xn: wgpu::Buffer,    // f16 [mp][hs]
    panel: wgpu::Buffer, // f16 [mp][panel]
    att: wgpu::Buffer,   // f16 [mp][att]
    y: wgpu::Buffer,     // f32 [mp][hs]
    hid: wgpu::Buffer,   // f16 [mp][mid]
    gu: wgpu::Buffer,    // f16 [mp][gu]
}

impl Acts {
    fn new(c: &Ctx, mp: usize, w: Widths) -> Acts {
        let m = mp as u64;
        let b = |width: usize, bytes: u64, label: &str| zi::sbuf(c, m * width as u64 * bytes, label);
        Acts {
            mp,
            w,
            x: b(w.hs, 4, "h3_x"),
            xn: b(w.hs, 2, "h3_xn"),
            panel: b(w.panel, 2, "h3_panel"),
            att: b(w.att, 2, "h3_att"),
            y: b(w.hs, 4, "h3_y"),
            hid: b(w.mid, 2, "h3_hid"),
            gu: b(w.gu, 2, "h3_gu"),
        }
    }
}

static STATE: Mutex<Option<Acts>> = Mutex::new(None);

/// Drop the activations (a stage is done). Touches module-local state only:
/// never brings a device up.
pub(crate) fn release() {
    if let Ok(mut st) = STATE.lock() {
        *st = None;
    }
    if let Ok(mut v) = VIT.lock() {
        *v = None;
    }
}

/// A q4tp tensor of the given shape: its index and its bytes in the file.
fn span(model: &CmfModel, idx: usize, rows: usize, cols: usize) -> Option<(usize, std::ops::Range<usize>)> {
    let e = model.tensors.get(idx)?;
    if e.dtype != TensorDtype::Q4TiledP || e.shape.len() != 2 || e.shape[0] != rows || e.shape[1] != cols {
        return None;
    }
    let abs = model.entry_abs_offset(e)?;
    let len = e.nbytes as usize;
    (len.is_multiple_of(16) && abs + len <= model.primary_bytes().len()).then_some((idx, abs..abs + len))
}

/// The activations for `mp` rows of these widths, kept or regrown. `Err`
/// carries the answer for the caller (declined).
fn acts_for<'a>(c: &Ctx, st: &'a mut Option<Acts>, mp: usize, w: Widths) -> Result<&'a Acts, bool> {
    let lim = c.device.limits();
    if (mp * w.panel.max(w.gu) * 2) as u64 > lim.max_storage_buffer_binding_size {
        return Err(decline("an activation panel passes the largest storage binding"));
    }
    if !st.as_ref().is_some_and(|a| a.mp >= mp && a.w == w) {
        *st = None;
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let a = Acts::new(c, mp, w);
        if let Some(e) = pollster::block_on(sc.pop()) {
            eprintln!("minimax-h3 wgpu: {e}");
            return Err(decline("the activations did not fit the card"));
        }
        *st = Some(a);
    }
    Ok(st.as_ref().unwrap())
}

/// How a stack's blocks reach the card under the adapter's weight budget
/// (`CMF_GPU_VRAM_MB` sets it): the blocks kept there between calls, then
/// the runs of blocks that pass through — read from the file, run, dropped,
/// every call. One kept run and nothing else when the whole stack fits
/// beside the activations and what the card already holds.
struct Plan {
    keep: std::ops::Range<usize>,
    pass: Vec<std::ops::Range<usize>>,
}

impl Plan {
    fn all(nb: usize) -> Plan {
        Plan { keep: 0..nb, pass: Vec::new() }
    }
    /// The runs in block order, the kept one first when it has blocks.
    fn runs(&self) -> impl Iterator<Item = (bool, std::ops::Range<usize>)> + '_ {
        (!self.keep.is_empty())
            .then(|| (true, self.keep.clone()))
            .into_iter()
            .chain(self.pass.iter().map(|r| (false, r.clone())))
    }
}

/// The plan for blocks of these byte sizes with `room` bytes for weights:
/// everything kept when it fits; else a slot the passing runs go through —
/// a quarter of the room, at most 1 GB, at least the largest block — the
/// first blocks that fit beside it kept, the rest in runs of at most the
/// slot. None: the largest block does not fit the room.
fn split(sizes: &[u64], room: u64) -> Option<Plan> {
    let nb = sizes.len();
    if sizes.iter().sum::<u64>() <= room {
        return Some(Plan::all(nb));
    }
    let largest = sizes.iter().copied().max().unwrap_or(0);
    let slot = (room / 4).min(1 << 30).max(largest);
    if room < slot {
        return None;
    }
    let (mut b, mut held) = (0usize, 0u64);
    while b < nb && held + sizes[b] <= room - slot {
        held += sizes[b];
        b += 1;
    }
    let keep = 0..b;
    let mut pass = Vec::new();
    while b < nb {
        let (start, mut run) = (b, 0u64);
        while b < nb && (b == start || run + sizes[b] <= slot) {
            run += sizes[b];
            b += 1;
        }
        pass.push(start..b);
    }
    Some(Plan { keep, pass })
}

/// `spans`: `per` tensors a block, in block order; `acts`: the bytes of the
/// stack's activations. `Err`: not even one block fits (declined, said
/// once).
fn plan(c: &Ctx, model: &Arc<CmfModel>, spans: &[(usize, std::ops::Range<usize>)], per: usize, acts: u64) -> Result<Plan, bool> {
    let uid = model.uid() as usize;
    let sizes: Vec<u64> = spans.chunks(per).map(|b| b.iter().map(|(_, r)| r.len() as u64).sum()).collect();
    let own: u64 = spans
        .iter()
        .filter(|(idx, _)| super::weight_resident(c, (uid, *idx)))
        .map(|(_, r)| r.len() as u64)
        .sum();
    let other = super::resident_bytes().saturating_sub(own);
    let room = c.vram_budget.saturating_sub(other + acts + (64 << 20));
    split(&sizes, room).ok_or_else(|| {
        decline(&format!(
            "a block of the stack is {:.2} GB and {:.2} GB of the adapter's {:.1} GB budget are free",
            sizes.iter().copied().max().unwrap_or(0) as f64 / 1e9,
            room as f64 / 1e9,
            c.vram_budget as f64 / 1e9
        ))
    })
}

/// These tensors on the card in their codec (uploaded, or as the residency
/// cache holds them).
fn fetch(c: &Ctx, model: &Arc<CmfModel>, spans: &[(usize, std::ops::Range<usize>)]) -> Result<Vec<wgpu::Buffer>, bool> {
    let uid = model.uid() as usize;
    let bytes = model.primary_bytes();
    let _chunk = super::UploadChunk::mb(4);
    let mut wb = Vec::with_capacity(spans.len());
    for (idx, r) in spans {
        let layer = super::layer_of_name(&model.tensors[*idx].name);
        match super::weight_buffer_l(c, (uid, *idx), &bytes[r.clone()], layer) {
            Some(b) => wb.push(b),
            None => return Err(decline("a weight did not fit the card")),
        }
    }
    Ok(wb)
}

/// `(cos, sin)` of every angle, as `h3_qkrope` reads them.
fn cos_sin(angles: &[f32]) -> Vec<f32> {
    angles
        .iter()
        .flat_map(|&v| {
            let (s, c) = v.sin_cos();
            [c, s]
        })
        .collect()
}

/// The probe buffer of `CMF_H3_WGPU_AMAX` and its sites: a name and the
/// site's guard.
type Amax<'a> = (&'a wgpu::Buffer, &'a [(&'a str, i32)]);

/// The dispatches of a run of blocks, given the run's weights on the card.
type Build<'a> = dyn Fn(std::ops::Range<usize>, &[wgpu::Buffer]) -> Option<ZCalls> + 'a;

/// The stream up, the stack's runs of blocks in order — each run's weights
/// fetched, its dispatches built by `build(blocks, weights)` and submitted,
/// a passing run's weights dropped before the next — and the stream back
/// into `h`. Under `CMF_H3_WGPU_PROF` a stack that ran in one piece is
/// timed class by class. `false`: `h` is untouched.
#[allow(clippy::too_many_arguments)]
fn run(
    c: &Ctx,
    model: &Arc<CmfModel>,
    spans: &[(usize, std::ops::Range<usize>)],
    per: usize,
    plan: &Plan,
    a: &Acts,
    h: &mut [f32],
    what: &str,
    t0: std::time::Instant,
    amax: Option<Amax<'_>>,
    build: &Build<'_>,
) -> bool {
    let prof = std::env::var("CMF_H3_WGPU_PROF").as_deref() == Ok("1");
    let uid = model.uid() as usize;
    c.queue.write_buffer(&a.x, 0, bytemuck::cast_slice(h));
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let (mut t_fetch, mut dispatches, mut submitted) = (0f64, 0usize, false);
    let mut whole: Option<ZCalls> = None;
    for (kept, blocks) in plan.runs() {
        // a stack with planes of its own has no cached tensors to fetch
        let mine = if spans.is_empty() { spans } else { &spans[blocks.start * per..blocks.end * per] };
        if !kept {
            // the card finishes the run before, then lets go of every
            // weight of this stack that is neither kept nor this run's
            if submitted {
                zi::wait(c);
            }
            let stale: Vec<(usize, usize)> = spans
                .iter()
                .enumerate()
                .filter(|(i, _)| !plan.keep.contains(&(i / per)) && !blocks.contains(&(i / per)))
                .map(|(_, (idx, _))| (uid, *idx))
                .collect();
            super::weight_evict(c, &stale);
        }
        let t = std::time::Instant::now();
        let wb = match fetch(c, model, mine) {
            Ok(wb) => wb,
            Err(said) => {
                let _ = pollster::block_on(sc.pop());
                return said;
            }
        };
        t_fetch += t.elapsed().as_secs_f64();
        let Some(calls) = build(blocks, &wb) else {
            let _ = pollster::block_on(sc.pop());
            return decline("a kernel of the chain was rejected by the device");
        };
        let mut enc = c.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            calls.record(&mut pass, None);
        }
        c.queue.submit(Some(enc.finish()));
        submitted = true;
        dispatches += calls.len();
        if plan.pass.is_empty() {
            whole = Some(calls);
        }
    }
    let raw = zi::read_bytes(c, &a.x, std::mem::size_of_val(h) as u64);
    if let Some(e) = pollster::block_on(sc.pop()) {
        return decline(&format!("the chain's command buffer failed: {e}"));
    }
    let Some(raw) = raw else {
        return decline("the readback failed");
    };
    let v: &[f32] = bytemuck::cast_slice(&raw);
    if let Some((ab, sites)) = amax
        && let Some(r) = zi::read_bytes(c, ab, 64)
    {
        let m: &[f32] = bytemuck::cast_slice(&r);
        let s: Vec<String> = sites
            .iter()
            .enumerate()
            .map(|(i, (name, sh))| format!("{name} {:.1} (×2^{sh} = {:.3e})", m[i], m[i] * p2(*sh)))
            .collect();
        eprintln!("h3 wgpu amax, {what} (stored): {}", s.join(" · "));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return decline("a chain produced a non-finite stream (an f16 range guard overflowed?)");
    }
    h.copy_from_slice(v);
    if prof {
        let mut line = format!(
            "h3 wgpu chain, {what}: {:.3} s (weights to the card {t_fetch:.3} s, {dispatches} dispatches",
            t0.elapsed().as_secs_f64()
        );
        match &whole {
            Some(calls) => {
                line += "); classes alone:";
                let mut tot = 0.0;
                for cl in Class::ALL {
                    let t = std::time::Instant::now();
                    let mut enc = c.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        for _ in 0..3 {
                            calls.record(&mut pass, Some(cl));
                        }
                    }
                    c.queue.submit(Some(enc.finish()));
                    zi::wait(c);
                    let ms = t.elapsed().as_secs_f64() * 1e3 / 3.0;
                    tot += ms;
                    line += &format!(" {cl:?} {ms:.1}");
                }
                line += &format!(" · sum {tot:.1} ms");
            }
            None => {
                line += &format!(
                    "; {} blocks kept on the card, {} passing in {} runs)",
                    plan.keep.len(),
                    plan.pass.iter().map(|r| r.len()).sum::<usize>(),
                    plan.pass.len()
                );
            }
        }
        eprintln!("{line}");
    }
    static FLUSHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !FLUSHED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        super::pipeline_cache_flush();
    }
    true
}

/// The row kernel's dispatch over `n` rows. `res`: the absolute offset of
/// the gate chunk in `mods` (unread without modulation); `bias`: instead,
/// the offsets in `cw` of the bias added to `y` and of the scale of the
/// sum; `norm`: the offsets of the shift and scale chunks and of the norm's
/// weights in `cw`.
#[allow(clippy::too_many_arguments)]
fn rowop(
    c: &Ctx,
    pipe: &Arc<wgpu::ComputePipeline>,
    a: &Acts,
    aux: (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
    n: usize,
    modded: bool,
    res: Option<usize>,
    bias: Option<(usize, usize)>,
    norm: Option<(usize, usize, usize)>,
    eps: f32,
    oscale: f32,
) -> Call {
    let mut mode = if modded { ROW_MOD } else { 0 };
    let (mut g_base, mut sh_base, mut sc_base, mut w_off) = (0usize, 0usize, 0usize, 0usize);
    if let Some(g) = res {
        mode |= ROW_RES;
        g_base = g;
    }
    if let Some((sh, sc, w)) = norm {
        mode |= ROW_NORM;
        (sh_base, sc_base, w_off) = (sh, sc, w);
    }
    if let Some((b, sc)) = bias {
        // an unmodulated stack: the two fields carry the bias and the scale
        mode |= ROW_RES | ROW_BIAS;
        (g_base, sh_base) = (b, sc);
    }
    let u = zi::ubuf(
        c,
        &[
            a.w.hs as u32,
            mode,
            g_base as u32,
            sh_base as u32,
            sc_base as u32,
            w_off as u32,
            eps.to_bits(),
            oscale.to_bits(),
        ],
    );
    let bg = zi::bg(c, pipe, &[&a.y, &a.x, &a.xn, aux.0, aux.1, aux.2, &u]);
    Call { pipe: pipe.clone(), bg, grid: (n as u32, 1, 1) }
}

/// The attention dispatch over the `[q | k | v]` panel: `nh` query heads,
/// keys at `k_col`, values at `v_col`, the output `[n][nh·128]` into `att`.
#[allow(clippy::too_many_arguments)]
fn flash_call(
    c: &Ctx,
    pipe: &Arc<wgpu::ComputePipeline>,
    fc: FlashCfg,
    a: &Acts,
    vis: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    n: usize,
    nh: usize,
    k_col: usize,
    v_col: usize,
) -> Call {
    let scl = std::f32::consts::LOG2_E / (128f32).sqrt();
    let u = zi::ubuf(
        c,
        &[
            n as u32,
            0,
            n as u32,
            a.w.panel as u32,
            k_col as u32,
            v_col as u32,
            a.w.att as u32,
            scl.to_bits(),
            vis.is_some() as u32,
            a.w.panel as u32,
            0,
            1.0f32.to_bits(),
        ],
    );
    let bg = zi::bg(c, pipe, &[&a.panel, &a.panel, &a.panel, &a.att, vis.unwrap_or(dummy), &u]);
    Call { pipe: pipe.clone(), bg, grid: ((n as u32).div_ceil(fc.nw * 16), nh as u32, 1) }
}

// ───────────────────────────── the DiT ─────────────────────────────

/// Probe sites of `CMF_H3_WGPU_AMAX=1`.
const AMAX_SITES: [&str; 6] = ["attn-in", "qkv", "attn-out", "mlp-in", "mlp-mid", "gate|up"];

/// The stack over `h` `[n][hidden]`, in place. `false`: nothing was written
/// and the caller runs its per-op path.
pub(crate) fn chain(job: &H3Chain<'_>, h: &mut [f32]) -> bool {
    if !enabled() {
        return false;
    }
    if job.qkv_interleaved {
        return vit_chain(job, h);
    }
    let (n, hs, nh, hd, ffn) = (job.n, job.hidden, job.heads, job.head_dim, job.ffn);
    let inner = nh * hd;
    let nb = job.blocks.len();
    if job.groups != 1 || hd != 128 || nb == 0 || n == 0 || h.len() != n * hs {
        return not_mine("a stack of batches or heads that are not 128 wide");
    }
    if job.blocks.iter().any(|b| {
        !b.qkv_bias.is_empty()
            || !b.out_bias.is_empty()
            || !b.fc1_bias.is_empty()
            || !b.fc2_bias.is_empty()
            || !b.scale1.is_empty()
            || !b.scale2.is_empty()
            || b.q_norm.len() != hd
            || b.k_norm.len() != hd
            || b.norm1.len() != hs
            || b.norm2.len() != hs
    }) {
        return not_mine("a stack with biases, residual scales or no q/k norm");
    }
    if !hs.is_multiple_of(128) || !ffn.is_multiple_of(64) || n > MAX_ROWS {
        return not_mine("widths the tiles do not divide, or more than 65535 rows");
    }
    let m0 = job.blocks[0].mods.len();
    let modded = m0 != 0 && !job.rows.is_empty();
    let mrows = if modded { m0 / (EXPAND * hs) } else { 0 };
    if modded
        && (!m0.is_multiple_of(EXPAND * hs)
            || job.rows.len() != n
            || job.blocks.iter().any(|b| b.mods.len() != m0)
            || job.rows.iter().any(|&r| r as usize >= mrows))
    {
        return decline("the modulation tables do not match the rows");
    }
    let pairs = if job.angles.is_empty() { 0 } else { job.angles.len() / n };
    if pairs * n != job.angles.len() || 2 * pairs > hd {
        return decline("the rotation angles do not match the rows");
    }
    let Some(c) = zi::zctx() else {
        return not_mine("no adapter with 16×16 f16 cooperative matrices, f16 shaders and 32-wide subgroups");
    };
    let _op = c.op_gate.enter();
    let _gate = c.mm_gate.lock().unwrap();
    let t0 = std::time::Instant::now();
    let gd = Guards::from_env();
    let t = tile();
    let fc = flash_cfg();
    if qi21::flash_shared(fc) > c.device.limits().max_compute_workgroup_storage_size {
        return decline("the attention tile does not fit the workgroup memory");
    }

    let mut spans = Vec::with_capacity(4 * nb);
    for b in job.blocks {
        for s in [
            span(job.model, b.qkv, 3 * inner, hs),
            span(job.model, b.out, hs, inner),
            span(job.model, b.fc1, 2 * ffn, hs),
            span(job.model, b.fc2, hs, ffn),
        ] {
            match s {
                Some(s) => spans.push(s),
                None => return not_mine("a projection that is not a q4tp tensor of the stack's shape"),
            }
        }
    }
    let mp = zi::pad_rows(n);
    let widths = Widths { hs, panel: 3 * inner, att: inner, mid: ffn, gu: 0 };
    let Ok(mut st) = STATE.lock() else { return false };
    let a = match acts_for(c, &mut st, mp, widths) {
        Ok(a) => a,
        Err(said) => return said,
    };
    let plan = match plan(c, job.model, &spans, 4, a.w.bytes(a.mp)) {
        Ok(p) => p,
        Err(said) => return said,
    };

    // Per call: the norms' weights, the modulation tables, the rows' angles.
    let cstride = 2 * hs + 2 * hd;
    let mut cw = Vec::with_capacity(nb * cstride);
    for b in job.blocks {
        cw.extend_from_slice(b.norm1);
        cw.extend_from_slice(b.norm2);
        cw.extend_from_slice(b.q_norm);
        cw.extend_from_slice(b.k_norm);
    }
    let cw_buf = zi::sbuf_init(c, bytemuck::cast_slice(&cw), "h3_cw");
    let dummy = zi::sbuf(c, 16, "h3_dummy");
    let (mods_buf, mrow_buf) = if modded {
        let mut m = Vec::with_capacity(nb * m0);
        for b in job.blocks {
            m.extend_from_slice(b.mods);
        }
        (
            zi::sbuf_init(c, bytemuck::cast_slice(&m), "h3_mods"),
            zi::sbuf_init(c, bytemuck::cast_slice(job.rows), "h3_mrow"),
        )
    } else {
        (dummy.clone(), dummy.clone())
    };
    let cs_buf = if pairs > 0 {
        zi::sbuf_init(c, bytemuck::cast_slice(&cos_sin(job.angles)), "h3_cs")
    } else {
        dummy.clone()
    };
    let amax = (std::env::var("CMF_H3_WGPU_AMAX").as_deref() == Ok("1")).then(|| zi::sbuf(c, 64, "h3_amax"));

    let build = |blocks: std::ops::Range<usize>, wb: &[wgpu::Buffer]| -> Option<ZCalls> {
        let mut calls = ZCalls::default();
        let rpipe = zi::pipeline(c, "h3_rowop", ROWOP_SRC, "h3_rowop")?;
        let qkpipe = zi::pipeline(c, "h3_qkrope", QKROPE_SRC, "h3_qkrope")?;
        let fpipe = zi::pipeline(c, &qi21::flash_key(fc), &qi21::flash_src(fc)?, "qi_flash")?;
        let aux = (&mods_buf, &mrow_buf, &cw_buf);
        // a norm of block `l`: 0 = before the attention, 1 = before the MLP
        let norm = |l: usize, half: usize| (l * m0 + 3 * half * hs, l * m0 + (3 * half + 1) * hs, l * cstride + half * hs);
        let row = |res: Option<usize>, nm: Option<(usize, usize, usize)>| {
            rowop(c, &rpipe, a, aux, n, modded, res, None, nm, job.eps, p2(-gd.act))
        };
        let probe = |calls: &mut ZCalls, buf: &wgpu::Buffer, halves: usize, site: u32| -> Option<()> {
            if let Some(ab) = amax.as_ref() {
                calls.push(Class::Io, zi::amax_call(c, buf, halves / 2, false, ab, site)?);
            }
            Some(())
        };
        if blocks.start == 0 {
            calls.push(Class::Rows, row(None, Some(norm(0, 0))));
        }
        for l in blocks.clone() {
            let w = &wb[4 * (l - blocks.start)..][..4];
            probe(&mut calls, &a.xn, n * hs, 0)?;
            let aq = margs(n, 3 * inner, hs, 3 * inner, 0, p2(gd.act - gd.qkv));
            calls.push_mm(Class::MmQkv, mm(c, site_cfg(t, Epi::F16), &aq, 0, &w[0], &a.xn, &a.panel)?);
            probe(&mut calls, &a.panel, n * 3 * inner, 1)?;
            {
                // q/k arrive ×2^-qkv: eps ×2^-2qkv keeps the norm exact
                let eps = job.qk_eps * p2(-2 * gd.qkv);
                let u = qk_uniform(c, 3 * inner, nh, nh, eps, pairs, l * cstride + 2 * hs, inner, 128, None, 0);
                let bg = zi::bg(c, &qkpipe, &[&a.panel, &cw_buf, &cs_buf, &u]);
                calls.push(Class::Rows, Call { pipe: qkpipe.clone(), bg, grid: (2 * nh as u32, n as u32, 1) });
            }
            calls.push(Class::Flash, flash_call(c, &fpipe, fc, a, None, &dummy, n, nh, inner, 2 * inner));
            probe(&mut calls, &a.att, n * inner, 2)?;
            let ao = margs(n, hs, inner, hs, 0, p2(gd.qkv));
            calls.push_mm(Class::MmO, mm(c, site_cfg(t, Epi::F32), &ao, 0, &w[1], &a.att, &a.y)?);
            calls.push(Class::Rows, row(Some(l * m0 + 2 * hs), Some(norm(l, 1))));
            probe(&mut calls, &a.xn, n * hs, 3)?;
            let a1 = margs(n, 2 * ffn, hs, ffn, 0, p2(-gd.hid));
            calls.push_mm(
                Class::MmW13,
                mm(c, site_cfg(t, Epi::SwiGluIn), &a1, p2(gd.act).to_bits(), &w[2], &a.xn, &a.hid)?,
            );
            probe(&mut calls, &a.hid, n * ffn, 4)?;
            let a2 = margs(n, hs, ffn, hs, 0, p2(gd.hid));
            calls.push_mm(Class::MmW2, mm(c, site_cfg(t, Epi::F32), &a2, 0, &w[3], &a.hid, &a.y)?);
            let next = (l + 1 < nb).then(|| norm(l + 1, 0));
            calls.push(Class::Rows, row(Some(l * m0 + 5 * hs), next));
        }
        Some(calls)
    };
    let sites = [
        (AMAX_SITES[0], gd.act),
        (AMAX_SITES[1], gd.qkv),
        (AMAX_SITES[2], gd.qkv),
        (AMAX_SITES[3], gd.act),
        (AMAX_SITES[4], gd.hid),
    ];
    run(
        c,
        job.model,
        &spans,
        4,
        &plan,
        a,
        h,
        &format!("{n} rows x {nb} blocks"),
        t0,
        amax.as_ref().map(|b| (b, &sites[..])),
        &build,
    )
}

// ───────────────────────── the prompt encoder ─────────────────────────

/// The layers of a language-model prompt encoder (grouped-query attention
/// with per-head q/k norms and a rotation over the whole head, a SwiGLU MLP
/// with separate gate and up projections, plain residuals) over one causal
/// sequence `h` `[n][hidden]`, in place. `false`: nothing was written.
pub(crate) fn te_chain(job: &TeChain<'_>, h: &mut [f32]) -> bool {
    if !enabled() || std::env::var("CMF_H3_WGPU_TE").as_deref() == Ok("0") {
        return false;
    }
    let (n, hs, nh, nkv, hd, inter) = (job.n, job.hidden, job.heads, job.kv_heads, job.head_dim, job.inter);
    let nl = job.layers.len();
    if n == 0 || nl == 0 || h.len() != n * hs || hd != 128 || nkv == 0 || !nh.is_multiple_of(nkv) {
        return not_mine("an encoder whose heads are not 128 wide or do not group");
    }
    if !hs.is_multiple_of(128) || !inter.is_multiple_of(128) || n > MAX_ROWS {
        return not_mine("widths the tiles do not divide, or more than 65535 rows");
    }
    let pairs = hd / 2;
    if job.cs.len() != n * pairs * 2 {
        return decline("the encoder's rotation table does not match the rows");
    }
    if job
        .layers
        .iter()
        .any(|l| l.input_norm.len() != hs || l.post_norm.len() != hs || l.q_norm.len() != hd || l.k_norm.len() != hd)
    {
        return not_mine("an encoder layer without q/k norms");
    }
    let (qd, kd) = (nh * hd, nkv * hd);
    let pw = qd + 2 * kd;
    let Some(c) = zi::zctx() else {
        return not_mine("no adapter with 16×16 f16 cooperative matrices, f16 shaders and 32-wide subgroups");
    };
    let _op = c.op_gate.enter();
    let _gate = c.mm_gate.lock().unwrap();
    let t0 = std::time::Instant::now();
    let gd = Guards::from_env();
    let t = tile();
    let fc = flash_cfg();
    if qi21::flash_shared(fc) > c.device.limits().max_compute_workgroup_storage_size {
        return decline("the attention tile does not fit the workgroup memory");
    }

    let mut spans = Vec::with_capacity(7 * nl);
    for l in job.layers {
        for s in [
            span(job.model, l.q, qd, hs),
            span(job.model, l.k, kd, hs),
            span(job.model, l.v, kd, hs),
            span(job.model, l.o, hs, qd),
            span(job.model, l.gate, inter, hs),
            span(job.model, l.up, inter, hs),
            span(job.model, l.down, hs, inter),
        ] {
            match s {
                Some(s) => spans.push(s),
                None => return not_mine("an encoder projection that is not a q4tp tensor of the stack's shape"),
            }
        }
    }
    let mp = zi::pad_rows(n);
    let widths = Widths { hs, panel: pw, att: qd, mid: inter, gu: 2 * inter };
    let Ok(mut st) = STATE.lock() else { return false };
    let a = match acts_for(c, &mut st, mp, widths) {
        Ok(a) => a,
        Err(said) => return said,
    };
    let plan = match plan(c, job.model, &spans, 7, a.w.bytes(a.mp)) {
        Ok(p) => p,
        Err(said) => return said,
    };

    let cstride = 2 * hs + 2 * hd;
    let mut cw = Vec::with_capacity(nl * cstride);
    for l in job.layers {
        cw.extend_from_slice(l.input_norm);
        cw.extend_from_slice(l.post_norm);
        cw.extend_from_slice(l.q_norm);
        cw.extend_from_slice(l.k_norm);
    }
    let cw_buf = zi::sbuf_init(c, bytemuck::cast_slice(&cw), "h3_te_cw");
    let cs_buf = zi::sbuf_init(c, bytemuck::cast_slice(job.cs), "h3_te_cs");
    // causal: query i sees keys [0, i]
    let vis: Vec<u32> = (1..=n as u32).collect();
    let vis_buf = zi::sbuf_init(c, bytemuck::cast_slice(&vis), "h3_te_vis");
    let dummy = zi::sbuf(c, 16, "h3_dummy");
    let amax = (std::env::var("CMF_H3_WGPU_AMAX").as_deref() == Ok("1")).then(|| zi::sbuf(c, 64, "h3_amax"));

    let build = |blocks: std::ops::Range<usize>, wb: &[wgpu::Buffer]| -> Option<ZCalls> {
        let mut calls = ZCalls::default();
        let rep = nh / nkv;
        let rpipe = zi::pipeline(c, "h3_rowop", ROWOP_SRC, "h3_rowop")?;
        let qkpipe = zi::pipeline(c, "h3_qkrope", QKROPE_SRC, "h3_qkrope")?;
        let spipe = zi::pipeline(c, "h3_swiglu", SWIGLU_SRC, "h3_swiglu")?;
        let fpipe = zi::pipeline(c, &format!("{}_g{rep}", qi21::flash_key(fc)), &flash_src(fc, rep)?, "qi_flash")?;
        let aux = (&dummy, &dummy, &cw_buf);
        let row = |res: bool, w_off: Option<usize>| {
            rowop(c, &rpipe, a, aux, n, false, res.then_some(0), None, w_off.map(|w| (0, 0, w)), job.eps, p2(-gd.act))
        };
        let probe = |calls: &mut ZCalls, buf: &wgpu::Buffer, halves: usize, site: u32| -> Option<()> {
            if let Some(ab) = amax.as_ref() {
                calls.push(Class::Io, zi::amax_call(c, buf, halves / 2, false, ab, site)?);
            }
            Some(())
        };
        let (f16, f32e) = (site_cfg(t, Epi::F16), site_cfg(t, Epi::F32));
        if blocks.start == 0 {
            calls.push(Class::Rows, row(false, Some(0)));
        }
        for l in blocks.clone() {
            let w = &wb[7 * (l - blocks.start)..][..7];
            probe(&mut calls, &a.xn, n * hs, 0)?;
            let os = p2(gd.act - gd.qkv);
            for (wi, rows, ocol) in [(0, qd, 0), (1, kd, qd), (2, kd, qd + kd)] {
                let aq = margs(n, rows, hs, pw, ocol, os);
                calls.push_mm(Class::MmQkv, mm(c, f16, &aq, 0, &w[wi], &a.xn, &a.panel)?);
            }
            probe(&mut calls, &a.panel, n * pw, 1)?;
            {
                let eps = job.eps * p2(-2 * gd.qkv);
                let u = qk_uniform(c, pw, nh, nkv, eps, pairs, l * cstride + 2 * hs, qd, 128, None, 0);
                let bg = zi::bg(c, &qkpipe, &[&a.panel, &cw_buf, &cs_buf, &u]);
                calls.push(Class::Rows, Call { pipe: qkpipe.clone(), bg, grid: ((nh + nkv) as u32, n as u32, 1) });
            }
            calls.push(Class::Flash, flash_call(c, &fpipe, fc, a, Some(&vis_buf), &dummy, n, nh, qd, qd + kd));
            probe(&mut calls, &a.att, n * qd, 2)?;
            let ao = margs(n, hs, qd, hs, 0, p2(gd.qkv));
            calls.push_mm(Class::MmO, mm(c, f32e, &ao, 0, &w[3], &a.att, &a.y)?);
            calls.push(Class::Rows, row(true, Some(l * cstride + hs)));
            probe(&mut calls, &a.xn, n * hs, 3)?;
            for (wi, ocol) in [(4, 0), (5, inter)] {
                let ag = margs(n, inter, hs, 2 * inter, ocol, p2(gd.act - gd.gu));
                calls.push_mm(Class::MmW13, mm(c, f16, &ag, 0, &w[wi], &a.xn, &a.gu)?);
            }
            probe(&mut calls, &a.gu, n * 2 * inter, 5)?;
            {
                let u = zi::ubuf(
                    c,
                    &[(n * inter) as u32, inter as u32, p2(gd.gu).to_bits(), p2(-gd.hid).to_bits(), u32::MAX, 0, 0, 0],
                );
                let bg = zi::bg(c, &spipe, &[&a.gu, &a.hid, &cw_buf, &u]);
                let wgs = ((n * inter) as u32).div_ceil(256);
                calls.push(Class::Rows, Call { pipe: spipe.clone(), bg, grid: (wgs.min(65535), wgs.div_ceil(65535), 1) });
            }
            probe(&mut calls, &a.hid, n * inter, 4)?;
            let ad = margs(n, hs, inter, hs, 0, p2(gd.hid));
            calls.push_mm(Class::MmW2, mm(c, f32e, &ad, 0, &w[6], &a.hid, &a.y)?);
            calls.push(Class::Rows, row(true, (l + 1 < nl).then(|| (l + 1) * cstride)));
        }
        Some(calls)
    };
    let sites = [
        (AMAX_SITES[0], gd.act),
        (AMAX_SITES[1], gd.qkv),
        (AMAX_SITES[2], gd.qkv),
        (AMAX_SITES[3], gd.act),
        (AMAX_SITES[4], gd.hid),
        (AMAX_SITES[5], gd.gu),
    ];
    run(
        c,
        job.model,
        &spans,
        7,
        &plan,
        a,
        h,
        &format!("encoder, {n} rows x {nl} layers"),
        t0,
        amax.as_ref().map(|b| (b, &sites[..])),
        &build,
    )
}

// ───────────────────────── the VAE's ViT decoder ─────────────────────────

/// Flash attention for a batch of independent sequences and a head width
/// of 64 or 128: `qi_flash`'s schedule (Q·Kᵀ and P·V on the matrix units in
/// f16, the online softmax in f32 with the lazy workgroup-uniform rescale)
/// with `hd / 16` fragments a head, every row offset by the workgroup's
/// sequence (`workgroup_id.z · nseq`), and keys past the sequence masked.
/// Bindings: 0 panel (f16), 1 panel (vec4<f16>), 2 output (vec4<u32>),
/// 3 `FP`.
fn vit_flash_src(f: FlashCfg, hd: u32) -> Option<String> {
    use std::fmt::Write;
    let (nw, bc) = (f.nw, f.bc);
    if nw == 0 || bc == 0 || !bc.is_multiple_of(16) || nw * 32 > 1024 || !(hd == 64 || hd == 128) {
        return None;
    }
    let nt = nw * 32;
    let nf = hd / 16; // fragments across a head
    let v4 = hd / 4; // vec4<f16> a row of a head
    let ldk = hd + 8;
    let ldp = bc + 8;
    let nkf = bc / 16;
    let vk = bc * v4;
    if !vk.is_multiple_of(nt) || !nf.is_multiple_of(nkf) || !(bc / 2).is_multiple_of(8) {
        return None;
    }
    let per = vk / nt;
    let half = bc / 2;
    let qw = nw * 16;
    let rounds = nf / nkf;
    let thr = zi::flash_thr();
    let mut s = String::new();
    let _ = writeln!(s, "enable f16;\nenable wgpu_cooperative_matrix;\ndiagnostic(off, derivative_uniformity);");
    let _ = writeln!(s, "struct FP {{ nq: u32, nseq: u32, ld: u32, k_col: u32, v_col: u32, o_ld: u32, scl: f32, oscale: f32 }};");
    let _ = writeln!(s, "@group(0) @binding(0) var<storage, read> qh: array<f16>;");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> qv: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read_write> oh: array<vec4<u32>>;");
    let _ = writeln!(s, "@group(0) @binding(3) var<uniform> p: FP;");
    // Every workgroup array is a multiple of 16 bytes (the zimage.rs trap:
    // cooperative loads ignore the low address bits).
    let _ = writeln!(s, "var<workgroup> sk: array<f16, {}>;", bc * ldk);
    let _ = writeln!(s, "var<workgroup> sv: array<f16, {}>;", bc * ldk);
    let _ = writeln!(s, "var<workgroup> ss: array<f32, {}>;", nw * 16 * bc);
    let _ = writeln!(s, "var<workgroup> sp: array<f16, {}>;", nw * 16 * ldp);
    let _ = writeln!(s, "var<workgroup> smx: array<f32, {nt}>;");
    let _ = writeln!(s, "var<workgroup> sfl: array<u32, 4>;");
    let _ = writeln!(
        s,
        "@compute @workgroup_size({nt})\nfn vit_flash(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sg: u32) {{"
    );
    let _ = writeln!(s, "  let h = wid.y;");
    let _ = writeln!(s, "  let gb = wid.z * p.nseq;");
    let _ = writeln!(s, "  let q0 = wid.x * {qw}u + sg * 16u;");
    let _ = writeln!(s, "  let lane = tid - sg * 32u;");
    let _ = writeln!(s, "  let r = lane & 15u; let hf = lane >> 4u;");
    let _ = writeln!(s, "  let ldq = p.ld / 4u;");
    let _ = writeln!(s, "  let kq4 = (p.k_col + h * {hd}u) / 4u; let vq4 = (p.v_col + h * {hd}u) / 4u;");
    let _ = writeln!(s, "  let ntot = p.nseq;");
    for d in 0..nf {
        let _ = writeln!(
            s,
            "  let qa{d} = coopLoadT<coop_mat16x16<f16, A>>(&qh[(gb + q0) * p.ld + h * {hd}u + {}u], p.ld);",
            d * 16
        );
    }
    for j in 0..nf {
        let _ = writeln!(s, "  var o{j}: coop_mat16x16<f32, C>;");
    }
    // Never assigned: the zero every block's S accumulation starts from.
    let _ = writeln!(s, "  var zc: coop_mat16x16<f32, C>;");
    let _ = writeln!(s, "  var mu = -1.0e30; var ls = 0.0;");
    let _ = writeln!(s, "  let qi = q0 + r;");
    let _ = writeln!(s, "  let live = qi < p.nq;");
    let _ = writeln!(s, "  if (tid < 2u) {{ sfl[tid] = 0u; }}");
    let _ = writeln!(s, "  let nkb = (ntot + {}u) / {bc}u;", bc - 1);
    let load = |s: &mut String, kbx: &str, ind: &str| {
        for t in 0..per {
            let _ = writeln!(
                s,
                "{ind}{{ let idx = tid + {o}u; let row = gb + min({kbx} * {bc}u + idx / {v4}u, ntot - 1u); let c4 = idx % {v4}u; rk{t} = qv[row * ldq + kq4 + c4]; rv{t} = qv[row * ldq + vq4 + c4]; }}",
                o = t * nt
            );
        }
    };
    for t in 0..per {
        let _ = writeln!(s, "  var rk{t}: vec4<f16>; var rv{t}: vec4<f16>;");
    }
    load(&mut s, "0u", "  ");
    let _ = writeln!(s, "  for (var kb = 0u; kb < nkb; kb = kb + 1u) {{");
    for t in 0..per {
        let _ = writeln!(
            s,
            "    {{ let idx = tid + {o}u; let d = (idx / {v4}u) * {ldk}u + (idx % {v4}u) * 4u; sk[d] = rk{t}.x; sk[d + 1u] = rk{t}.y; sk[d + 2u] = rk{t}.z; sk[d + 3u] = rk{t}.w; sv[d] = rv{t}.x; sv[d + 1u] = rv{t}.y; sv[d + 2u] = rv{t}.z; sv[d + 3u] = rv{t}.w; }}",
            o = t * nt
        );
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    if (kb + 1u < nkb) {{");
    load(&mut s, "(kb + 1u)", "      ");
    let _ = writeln!(s, "    }}");
    // S = Q Kᵀ
    for j in 0..nkf {
        let _ = writeln!(s, "    let kf{j}_0 = coopLoad<coop_mat16x16<f16, B>>(&sk[{}u], {ldk}u);", j * 16 * ldk);
        let _ = writeln!(s, "    var s{j} = coopMultiplyAdd(qa0, kf{j}_0, zc);");
        for d in 1..nf {
            let _ = writeln!(s, "    let kf{j}_{d} = coopLoad<coop_mat16x16<f16, B>>(&sk[{}u], {ldk}u);", j * 16 * ldk + d * 16);
            let _ = writeln!(s, "    s{j} = coopMultiplyAdd(qa{d}, kf{j}_{d}, s{j});");
        }
        let _ = writeln!(s, "    coopStoreT(s{j}, &ss[sg * {}u + {}u], {bc}u);", 16 * bc, j * 16);
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    // this lane's scores (scaled, masked) into registers
    let _ = writeln!(s, "    let sbase = sg * {}u + r * {bc}u + hf * {half}u;", 16 * bc);
    let _ = writeln!(s, "    let kbase = kb * {bc}u + hf * {half}u;");
    for e in 0..half {
        let _ = writeln!(s, "    let x{e} = select(-1.0e30, ss[sbase + {e}u] * p.scl, kbase + {e}u < ntot);");
    }
    let _ = writeln!(s, "    var pm = x0;");
    for e in 1..half {
        let _ = writeln!(s, "    pm = max(pm, x{e});");
    }
    let _ = writeln!(s, "    smx[tid] = pm;");
    // Only live rows vote for the workgroup-wide rescale (zimage.rs, vk2).
    let _ = writeln!(s, "    if (pm > mu + {thr} && live) {{ sfl[kb & 1u] = 1u; }}");
    let _ = writeln!(s, "    let need = workgroupUniformLoad(&sfl[kb & 1u]);");
    let _ = writeln!(s, "    if (tid == 0u) {{ sfl[(kb + 1u) & 1u] = 0u; }}");
    let _ = writeln!(s, "    let rmax = max(pm, smx[tid ^ 16u]);");
    let _ = writeln!(s, "    if (need != 0u) {{");
    let _ = writeln!(s, "      let mn = max(mu, rmax);");
    let _ = writeln!(s, "      let alpha = exp2(mu - mn);");
    let _ = writeln!(s, "      ls = ls * alpha; mu = mn;");
    let _ = writeln!(s, "      if (kb > 0u) {{");
    for rd in 0..rounds {
        for q in 0..nkf {
            let _ = writeln!(s, "        coopStoreT(o{}, &ss[sg * {}u + {}u], {bc}u);", rd * nkf + q, 16 * bc, q * 16);
        }
        let _ = writeln!(s, "        workgroupBarrier();");
        for e in 0..half {
            let _ = writeln!(s, "        ss[sbase + {e}u] = ss[sbase + {e}u] * alpha;");
        }
        let _ = writeln!(s, "        workgroupBarrier();");
        for q in 0..nkf {
            let _ = writeln!(
                s,
                "        o{} = coopLoadT<coop_mat16x16<f32, C>>(&ss[sg * {}u + {}u], {bc}u);",
                rd * nkf + q,
                16 * bc,
                q * 16
            );
        }
        let _ = writeln!(s, "        workgroupBarrier();");
    }
    let _ = writeln!(s, "      }}");
    let _ = writeln!(s, "    }}");
    // P = 2^(s − mu) → f16 (masked keys exactly 0), row sums
    let _ = writeln!(s, "    let pbase = sg * {}u + r * {ldp}u + hf * {half}u;", 16 * ldp);
    for e in 0..half {
        let _ = writeln!(s, "    let pv{e} = select(0.0, exp2(x{e} - mu), kbase + {e}u < ntot);");
    }
    for e in 0..half {
        let _ = writeln!(s, "    ls = ls + pv{e}; sp[pbase + {e}u] = f16(pv{e});");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    // O += P V
    for kk in 0..nkf {
        let _ = writeln!(s, "    {{ let pa = coopLoadT<coop_mat16x16<f16, A>>(&sp[sg * {}u + {}u], {ldp}u);", 16 * ldp, kk * 16);
        for j in 0..nf {
            let _ = writeln!(s, "      let vf{j} = coopLoadT<coop_mat16x16<f16, B>>(&sv[{}u], {ldk}u);", kk * 16 * ldk + j * 16);
            let _ = writeln!(s, "      o{j} = coopMultiplyAdd(pa, vf{j}, o{j});");
        }
        let _ = writeln!(s, "    }}");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "  }}");
    // finalize: O / l · oscale → f16
    let _ = writeln!(s, "  smx[tid] = ls;");
    let _ = writeln!(s, "  workgroupBarrier();");
    let _ = writeln!(s, "  let lt = ls + smx[tid ^ 16u];");
    let _ = writeln!(s, "  let inv = select(0.0, p.oscale / lt, lt > 0.0);");
    let _ = writeln!(s, "  let sbase = sg * {}u + r * {bc}u + hf * {half}u;", 16 * bc);
    for rd in 0..rounds {
        for q in 0..nkf {
            let _ = writeln!(s, "  coopStoreT(o{}, &ss[sg * {}u + {}u], {bc}u);", rd * nkf + q, 16 * bc, q * 16);
        }
        let _ = writeln!(s, "  workgroupBarrier();");
        let _ = writeln!(s, "  if (live) {{");
        for c8 in 0..half / 8 {
            let _ = writeln!(
                s,
                "    {{ let b = sbase + {}u; oh[((gb + qi) * p.o_ld + h * {hd}u + {}u + hf * {half}u + {}u) / 8u] = vec4<u32>(pack2x16float(vec2<f32>(ss[b], ss[b + 1u]) * inv), pack2x16float(vec2<f32>(ss[b + 2u], ss[b + 3u]) * inv), pack2x16float(vec2<f32>(ss[b + 4u], ss[b + 5u]) * inv), pack2x16float(vec2<f32>(ss[b + 6u], ss[b + 7u]) * inv)); }}",
                c8 * 8,
                rd * bc,
                c8 * 8
            );
        }
        let _ = writeln!(s, "  }}");
        let _ = writeln!(s, "  workgroupBarrier();");
    }
    let _ = writeln!(s, "}}");
    Some(s)
}

/// Workgroup-shared bytes of `vit_flash`.
fn vit_flash_shared(f: FlashCfg, hd: u32) -> u32 {
    2 * f.bc * (hd + 8) * 2 + f.nw * 16 * f.bc * 4 + f.nw * 16 * (f.bc + 8) * 2 + f.nw * 32 * 4 + 16
}

/// The f16 planes of a ViT stack, built once from the container (any codec
/// `zi::tensor_to_plane` expands): per block qkv with its rows regrouped
/// from per-head `[q k v]` triples into `[q | k | v]`, the output
/// projection, fc1 `[gate | up]` as stored, fc2.
struct VitDev {
    key: (u64, usize, usize),
    planes: Vec<[wgpu::Buffer; 4]>,
}

static VIT: Mutex<Option<VitDev>> = Mutex::new(None);

/// A ViT stack — the decoder of the MiniMax-H3 VAEs: projections with
/// biases, a layer scale on each residual, a weightless q/k norm, heads of
/// 64 or 128, per-head `[q k v]` triples — over a batch of `groups`
/// independent sequences of `n` rows, `h` `[groups·n][hidden]` in place.
/// Its weights are small (34 MB of f16 a block of the image VAE), so they
/// are kept as planes whatever their codec. `false`: nothing was written.
fn vit_chain(job: &H3Chain<'_>, h: &mut [f32]) -> bool {
    if std::env::var("CMF_H3_WGPU_VIT").as_deref() == Ok("0") {
        return false;
    }
    let (n, hs, nh, hd, ffn, groups) = (job.n, job.hidden, job.heads, job.head_dim, job.ffn, job.groups);
    let inner = nh * hd;
    let nb = job.blocks.len();
    let rows = groups * n;
    if nb == 0 || n == 0 || groups == 0 || h.len() != rows * hs || !(hd == 64 || hd == 128) {
        return not_mine("a ViT stack whose heads are not 64 or 128 wide");
    }
    if !hs.is_multiple_of(128) || !inner.is_multiple_of(128) || !ffn.is_multiple_of(64) || rows > MAX_ROWS {
        return not_mine("widths the tiles do not divide, or more than 65535 rows");
    }
    if !job.rows.is_empty()
        || job.blocks.iter().any(|b| {
            !b.mods.is_empty()
                || b.qkv_bias.len() != 3 * inner
                || b.out_bias.len() != hs
                || b.fc1_bias.len() != 2 * ffn
                || b.fc2_bias.len() != hs
                || b.scale1.len() != hs
                || b.scale2.len() != hs
                || b.norm1.len() != hs
                || b.norm2.len() != hs
                || !(b.q_norm.is_empty() || b.q_norm.len() == hd)
                || !(b.k_norm.is_empty() || b.k_norm.len() == hd)
        })
    {
        return not_mine("a ViT stack without biases and layer scales, or with modulation");
    }
    let pairs = if job.angles.is_empty() { 0 } else { job.angles.len() / n };
    if pairs * n != job.angles.len() || 2 * pairs > hd {
        return decline("the rotation angles do not match the rows");
    }
    let Some(c) = zi::zctx() else {
        return not_mine("no adapter with 16×16 f16 cooperative matrices, f16 shaders and 32-wide subgroups");
    };
    let _op = c.op_gate.enter();
    let _gate = c.mm_gate.lock().unwrap();
    let t0 = std::time::Instant::now();
    let gd = Guards::from_env();
    let fc = flash_cfg();
    if vit_flash_shared(fc, hd as u32) > c.device.limits().max_compute_workgroup_storage_size {
        return decline("the attention tile does not fit the workgroup memory");
    }

    // The planes, once per stack.
    let key = (job.model.uid(), job.blocks[0].qkv, nb);
    let Ok(mut vit) = VIT.lock() else { return false };
    if vit.as_ref().is_none_or(|d| d.key != key) {
        *vit = None;
        let bytes = (nb * 2 * (3 * inner * hs + hs * inner + 2 * ffn * hs + hs * ffn)) as u64;
        if super::resident_bytes() + bytes > c.vram_budget {
            return decline("the ViT's planes do not fit the adapter's budget");
        }
        let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let built: Option<Vec<[wgpu::Buffer; 4]>> = job
            .blocks
            .iter()
            .map(|b| {
                let plane = |rows: usize, cols: usize, label: &str| zi::sbuf(c, (rows * cols * 2) as u64, label);
                let qkv = plane(3 * inner, hs, "h3_vit_qkv");
                // tensor rows are per-head [q k v] triples; the plane is [q | k | v]
                zi::tensor_to_plane(c, job.model, b.qkv, 3 * inner, hs, &qkv, &|p| {
                    let r = 16 * p;
                    (r % (3 * hd)) / hd * inner + r / (3 * hd) * hd + r % hd
                })?;
                let out = plane(hs, inner, "h3_vit_o");
                zi::tensor_to_plane(c, job.model, b.out, hs, inner, &out, &|p| 16 * p)?;
                let fc1 = plane(2 * ffn, hs, "h3_vit_fc1");
                zi::tensor_to_plane(c, job.model, b.fc1, 2 * ffn, hs, &fc1, &|p| 16 * p)?;
                let fc2 = plane(hs, ffn, "h3_vit_fc2");
                zi::tensor_to_plane(c, job.model, b.fc2, hs, ffn, &fc2, &|p| 16 * p)?;
                // flush the staged writes block by block
                c.queue.submit(std::iter::empty());
                zi::wait(c);
                Some([qkv, out, fc1, fc2])
            })
            .collect();
        if let Some(e) = pollster::block_on(sc.pop()) {
            eprintln!("minimax-h3 wgpu: {e}");
            return decline("the ViT's planes did not fit the card");
        }
        match built {
            Some(planes) => *vit = Some(VitDev { key, planes }),
            None => return not_mine("a ViT projection in a codec the plane builder cannot expand"),
        }
    }
    let planes = &vit.as_ref().unwrap().planes;

    let mp = zi::pad_rows(rows);
    let widths = Widths { hs, panel: 3 * inner, att: inner, mid: ffn, gu: 2 * ffn };
    let Ok(mut st) = STATE.lock() else { return false };
    let a = match acts_for(c, &mut st, mp, widths) {
        Ok(a) => a,
        Err(said) => return said,
    };

    // Per block in `cw`: the two norms, the q and k norm weights (ones when
    // the stack has none), the qkv bias in the panel's [q | k | v] order,
    // the output bias and its layer scale, fc1's bias, fc2's and its scale.
    let (o_n2, o_qk, o_qb) = (hs, 2 * hs, 2 * hs + 2 * hd);
    let o_ob = o_qb + 3 * inner;
    let o_s1 = o_ob + hs;
    let o_f1 = o_s1 + hs;
    let o_f2 = o_f1 + 2 * ffn;
    let o_s2 = o_f2 + hs;
    let cstride = o_s2 + hs;
    let ones = vec![1f32; hd];
    let mut cw = Vec::with_capacity(nb * cstride);
    for b in job.blocks {
        cw.extend_from_slice(b.norm1);
        cw.extend_from_slice(b.norm2);
        cw.extend_from_slice(if b.q_norm.is_empty() { &ones } else { b.q_norm });
        cw.extend_from_slice(if b.k_norm.is_empty() { &ones } else { b.k_norm });
        for part in 0..3 {
            for head in 0..nh {
                let at = head * 3 * hd + part * hd;
                cw.extend_from_slice(&b.qkv_bias[at..at + hd]);
            }
        }
        cw.extend_from_slice(b.out_bias);
        cw.extend_from_slice(b.scale1);
        cw.extend_from_slice(b.fc1_bias);
        cw.extend_from_slice(b.fc2_bias);
        cw.extend_from_slice(b.scale2);
    }
    let cw_buf = zi::sbuf_init(c, bytemuck::cast_slice(&cw), "h3_vit_cw");
    let dummy = zi::sbuf(c, 16, "h3_dummy");
    let cs_buf = if pairs > 0 {
        zi::sbuf_init(c, bytemuck::cast_slice(&cos_sin(job.angles)), "h3_vit_cs")
    } else {
        dummy.clone()
    };

    // the planes are the stack's own: one run, nothing to fetch
    let build = |_: std::ops::Range<usize>, _: &[wgpu::Buffer]| -> Option<ZCalls> {
        let mut calls = ZCalls::default();
        let rpipe = zi::pipeline(c, "h3_rowop", ROWOP_SRC, "h3_rowop")?;
        let qkpipe = zi::pipeline(c, "h3_qkrope", QKROPE_SRC, "h3_qkrope")?;
        let spipe = zi::pipeline(c, "h3_swiglu", SWIGLU_SRC, "h3_swiglu")?;
        let fkey = format!("vit_flash_{}_{}_{hd}_{}", fc.nw, fc.bc, zi::flash_thr());
        let fpipe = zi::pipeline(c, &fkey, &vit_flash_src(fc, hd as u32)?, "vit_flash")?;
        let aux = (&dummy, &dummy, &cw_buf);
        let row = |bias: Option<(usize, usize)>, w_off: Option<usize>| {
            rowop(c, &rpipe, a, aux, rows, false, None, bias, w_off.map(|w| (0, 0, w)), job.eps, p2(-gd.act))
        };
        let (f16, f32e) = (zi::default_cfg(Epi::F16), zi::default_cfg(Epi::F32));
        calls.push(Class::Rows, row(None, Some(0)));
        for (l, w) in planes.iter().enumerate() {
            let base = l * cstride;
            let aq = margs(rows, 3 * inner, hs, 3 * inner, 0, p2(gd.act - gd.qkv));
            calls.push_mm(Class::MmQkv, mm(c, f16, &aq, 0, &w[0], &a.xn, &a.panel)?);
            {
                let u = qk_uniform(
                    c,
                    3 * inner,
                    nh,
                    nh,
                    job.qk_eps * p2(-2 * gd.qkv),
                    pairs,
                    base + o_qk,
                    inner,
                    hd,
                    Some((base + o_qb, p2(-gd.qkv), 2 * inner)),
                    n,
                );
                let bg = zi::bg(c, &qkpipe, &[&a.panel, &cw_buf, &cs_buf, &u]);
                calls.push(Class::Rows, Call { pipe: qkpipe.clone(), bg, grid: (3 * nh as u32, rows as u32, 1) });
            }
            {
                let scl = std::f32::consts::LOG2_E / (hd as f32).sqrt();
                let u = zi::ubuf(
                    c,
                    &[
                        n as u32,
                        n as u32,
                        (3 * inner) as u32,
                        inner as u32,
                        (2 * inner) as u32,
                        inner as u32,
                        scl.to_bits(),
                        1.0f32.to_bits(),
                    ],
                );
                let bg = zi::bg(c, &fpipe, &[&a.panel, &a.panel, &a.att, &u]);
                calls.push(
                    Class::Flash,
                    Call { pipe: fpipe.clone(), bg, grid: ((n as u32).div_ceil(fc.nw * 16), nh as u32, groups as u32) },
                );
            }
            let ao = margs(rows, hs, inner, hs, 0, p2(gd.qkv));
            calls.push_mm(Class::MmO, mm(c, f32e, &ao, 0, &w[1], &a.att, &a.y)?);
            calls.push(Class::Rows, row(Some((base + o_ob, base + o_s1)), Some(base + o_n2)));
            let a1 = margs(rows, 2 * ffn, hs, 2 * ffn, 0, p2(gd.act - gd.gu));
            calls.push_mm(Class::MmW13, mm(c, f16, &a1, 0, &w[2], &a.xn, &a.gu)?);
            {
                let u = zi::ubuf(
                    c,
                    &[
                        (rows * ffn) as u32,
                        ffn as u32,
                        p2(gd.gu).to_bits(),
                        p2(-gd.hid).to_bits(),
                        (base + o_f1) as u32,
                        0,
                        0,
                        0,
                    ],
                );
                let bg = zi::bg(c, &spipe, &[&a.gu, &a.hid, &cw_buf, &u]);
                let wgs = ((rows * ffn) as u32).div_ceil(256);
                calls.push(Class::Rows, Call { pipe: spipe.clone(), bg, grid: (wgs.min(65535), wgs.div_ceil(65535), 1) });
            }
            let a2 = margs(rows, hs, ffn, hs, 0, p2(gd.hid));
            calls.push_mm(Class::MmW2, mm(c, f32e, &a2, 0, &w[3], &a.hid, &a.y)?);
            calls.push(Class::Rows, row(Some((base + o_f2, base + o_s2)), (l + 1 < nb).then(|| (l + 1) * cstride)));
        }
        Some(calls)
    };
    run(
        c,
        job.model,
        &[],
        1,
        &Plan::all(nb),
        a,
        h,
        &format!("ViT, {groups} x {n} rows x {nb} blocks"),
        t0,
        None,
        &build,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chains' GEMM variants and kernels are valid WGSL for naga — a
    /// rejected module only shows on a card as "the chain declined".
    #[test]
    fn chain_kernels_validate() {
        use wgpu::naga;
        let check = |name: &str, src: &str| {
            let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&m)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        };
        check("h3_rowop", ROWOP_SRC);
        check("h3_qkrope", QKROPE_SRC);
        check("h3_swiglu", SWIGLU_SRC);
        for epi in [Epi::F16, Epi::F32, Epi::SwiGluIn] {
            let g = site_cfg((128, 128, 32, 2, 2), epi);
            assert!(g.valid(), "{g:?}");
            check(&format!("{epi:?}"), &zi::mm_src(g));
        }
        let f = FlashCfg { nw: 4, bc: 16 };
        check("qi_flash", &flash_src(f, 1).expect("flash source"));
        let grouped = flash_src(f, 8).expect("grouped flash source");
        assert!(grouped.contains("(h / 8u) * 128u"));
        check("qi_flash grouped", &grouped);
        for hd in [64, 128] {
            let v = vit_flash_src(f, hd).expect("ViT flash source");
            assert!(vit_flash_shared(f, hd) < 48 * 1024);
            check(&format!("vit_flash {hd}"), &v);
        }
        for epi in [Epi::F16, Epi::F32] {
            check(&format!("plane {epi:?}"), &zi::mm_src(zi::default_cfg(epi)));
        }
        assert!(vit_flash_src(f, 96).is_none());
    }

    /// A stack under a ceiling: every block once, in order, the kept ones
    /// first; a run never past the slot unless it is one block.
    #[test]
    fn a_stack_larger_than_the_room_is_split_in_order() {
        let sizes = vec![200u64 << 20; 50];
        // everything fits
        let p = split(&sizes, 11 << 30).unwrap();
        assert_eq!((p.keep.clone(), p.pass.len()), (0..50, 0));
        // 4 GB of room: a 1 GB slot, 15 blocks kept, the rest in runs of 5
        let p = split(&sizes, 4 << 30).unwrap();
        assert_eq!(p.keep, 0..15);
        assert!(p.pass.iter().all(|r| r.len() == 5));
        let order: Vec<usize> = p.runs().flat_map(|(_, r)| r).collect();
        assert_eq!(order, (0..50).collect::<Vec<_>>());
        assert!(p.runs().next().is_some_and(|(kept, _)| kept));
        // room for one block and nothing kept: every block passes alone
        let p = split(&sizes, 250 << 20).unwrap();
        assert!(p.keep.is_empty());
        assert_eq!(p.pass.len(), 50);
        assert!(p.runs().all(|(kept, r)| !kept && r.len() == 1));
        // a block larger than the room
        assert!(split(&sizes, 100 << 20).is_none());
        // blocks of different sizes: the slot is at least the largest
        let p = split(&[100 << 20, 900 << 20, 100 << 20], 1000 << 20).unwrap();
        assert_eq!(p.keep, 0..1);
        assert_eq!(p.pass, vec![1..2, 2..3]);
    }

    /// `[gate | up]` rows under the SwiGLU epilogue's 16-row panels.
    #[test]
    fn q4h_reads_gate_then_up() {
        let s = zi::mm_src(site_cfg((128, 128, 32, 2, 2), Epi::SwiGluIn));
        assert!(s.contains("let qrow = (qpl / 32u) * 16u + (qpl % 16u) + ((qpl / 16u) & 1u) * (p.n / 2u);"));
        let row = |qpl: usize, n: usize| (qpl / 32) * 16 + qpl % 16 + ((qpl / 16) & 1) * (n / 2);
        // panel 0 = gate rows 0..16, panel 1 = up rows 0..16, panel 2 = gate 16..32
        assert_eq!(row(0, 64), 0);
        assert_eq!(row(15, 64), 15);
        assert_eq!(row(16, 64), 32);
        assert_eq!(row(31, 64), 47);
        assert_eq!(row(32, 64), 16);
        assert_eq!(row(63, 64), 63);
    }
}
