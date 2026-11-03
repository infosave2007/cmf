//! Kandinsky S2V on wgpu/Vulkan: the whole denoiser forward resident on the
//! card, built from the Z-Image kernels (`zi_mm`, `zi_flash`).
//!
//! The weights do not fit as planes: 21 B parameters at f16 are 42 GB on a
//! 32 GB card. So the container's quantized bytes stay resident as they are
//! (q4tp 10.7 GB, q8_2f 21.4 GB) and one block's f16 planes are expanded into
//! a shared scratch right before that block runs, in the same command stream.
//! Every GEMM site's bind group is then the same for all 60 blocks.
//!
//! One forward:
//! - text: the 4 modulated blocks, once per segment and cached across chunks;
//! - audio: the 4 unmodulated blocks, once per chunk;
//! - visual: `copy clean rows · embed noisy rows · 60 blocks · final GEMM`, one
//!   upload (patches) and one readback (the head output).
//!
//! Per block: `dq · row op (LN·(1+s)+sh → f16) · qkv GEMM · bias + qk-norm + RoPE
//! · flash · O GEMM · row op (gated residual, next LN) · q GEMM · k|v GEMM over
//! the f16 prompt · norms · cross attention · O GEMM · row op`, optionally the
//! speech adapter (affine LN over the noisy tail, frame-aligned attention,
//! ungated residual) and the feed-forward (`w1` GEMM with a GELU epilogue, `w2`
//! GEMM, row op). Short attentions (prompt, speech) use a plain kernel: at 186
//! keys a block's cross-attention is 0.3 % of its multiply-adds.

use super::Ctx;
use super::zimage::{
    Epi, MmArgs, MmCall, MmCfg, bg, default_cfg, default_flash, flash_calls, mm_call, pad_rows, pipeline, sbuf, sbuf_init,
    ubuf, zctx,
};
use crate::dit::Proj;
use crate::s2v::{self, S2vDit, S2vGeom, audio_groups};
use crate::s2vgen::S2vCond;
use cortiq_core::{CmfModel, TensorDtype as T};
use std::sync::Arc;
use std::time::Instant;

const ROW_SRC: &str = r#"
struct RP {
  h: u32, mode: u32, nclean: u32, xoff: u32,
  g0: u32, g1: u32, sh0: u32, sh1: u32,
  sc0: u32, sc1: u32, bias: u32, aw: u32,
  ab: u32, _p0: u32, eps: f32, oscale: f32,
};
@group(0) @binding(0) var<storage, read> br: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xn: array<u32>;
@group(0) @binding(3) var<storage, read> mods: array<f32>;
@group(0) @binding(4) var<storage, read> aux: array<f32>;
@group(0) @binding(5) var<uniform> rp: RP;
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

// Modes: 1 x += g·(br + bias), 2 the gate comes from mods (else g = 1),
// 8 xn = LN(x)·(1 + scale) + shift, 16 xn = LN(x)·w + b, 32 xn = x.
// One workgroup per row; each of the 256 threads owns 8 column pairs.
@compute @workgroup_size(256)
fn s2v_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let row = wid.x + wid.y * 65535u;
  let xrow = row + rp.xoff;
  let clean = xrow < rp.nclean;
  let base = xrow * rp.h;
  let bb = row * rp.h;
  let np = rp.h / 512u;
  var xv: array<vec2<f32>, 8>;
  if ((rp.mode & 1u) != 0u) {
    let g = select(rp.g1, rp.g0, clean);
    for (var j = 0u; j < np; j = j + 1u) {
      let c = 2u * (lid + j * 256u);
      var gv = vec2<f32>(1.0, 1.0);
      if ((rp.mode & 2u) != 0u) { gv = vec2<f32>(mods[g + c], mods[g + c + 1u]); }
      let o = vec2<f32>(x[base + c], x[base + c + 1u])
        + gv * (vec2<f32>(br[bb + c], br[bb + c + 1u]) + vec2<f32>(aux[rp.bias + c], aux[rp.bias + c + 1u]));
      x[base + c] = o.x;
      x[base + c + 1u] = o.y;
      xv[j] = o;
    }
  } else {
    for (var j = 0u; j < np; j = j + 1u) {
      let c = 2u * (lid + j * 256u);
      xv[j] = vec2<f32>(x[base + c], x[base + c + 1u]);
    }
  }
  if ((rp.mode & 56u) == 0u) { return; }
  var mean = 0.0;
  var inv = 1.0;
  if ((rp.mode & 24u) != 0u) {
    var s = 0.0;
    for (var j = 0u; j < np; j = j + 1u) { s = s + xv[j].x + xv[j].y; }
    mean = wsum(s, lid) / f32(rp.h);
    var v = 0.0;
    for (var j = 0u; j < np; j = j + 1u) { let d = xv[j] - vec2<f32>(mean); v = v + dot(d, d); }
    inv = inverseSqrt(wsum(v, lid) / f32(rp.h) + rp.eps);
  }
  let sh = select(rp.sh1, rp.sh0, clean);
  let sc = select(rp.sc1, rp.sc0, clean);
  for (var j = 0u; j < np; j = j + 1u) {
    let c = 2u * (lid + j * 256u);
    var y = xv[j];
    if ((rp.mode & 8u) != 0u) {
      y = (y - vec2<f32>(mean)) * inv * (vec2<f32>(mods[sc + c], mods[sc + c + 1u]) + vec2<f32>(1.0)) + vec2<f32>(mods[sh + c], mods[sh + c + 1u]);
    } else if ((rp.mode & 16u) != 0u) {
      y = (y - vec2<f32>(mean)) * inv * vec2<f32>(aux[rp.aw + c], aux[rp.aw + c + 1u]) + vec2<f32>(aux[rp.ab + c], aux[rp.ab + c + 1u]);
    }
    xn[(row * rp.h + c) / 2u] = pack2x16float(y * rp.oscale);
  }
}
"#;

const NORM_SRC: &str = r#"
struct NP { ld: u32, col: u32, flags: u32, w: u32, b: u32, eps: f32, bscale: f32, nrows: u32 };
@group(0) @binding(0) var<storage, read_write> qkv: array<u32>;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> rc: array<f32>;
@group(0) @binding(3) var<storage, read> rs: array<f32>;
@group(0) @binding(4) var<uniform> np: NP;
var<workgroup> red: array<f32, 64>;

// One workgroup (64 lanes = 64 pairs of a 128-wide head) per (token, head):
// the bias, then (flag 1) RMSNorm + weight, then (flag 2) the rotation.
@compute @workgroup_size(64)
fn s2v_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let row = wid.y + wid.z * 65535u;
  let hh = wid.x;
  let idx = (row * np.ld + np.col + hh * 128u) / 2u + lid;
  let bi = np.b + hh * 128u + 2u * lid;
  var v = unpack2x16float(qkv[idx]) + vec2<f32>(aux[bi], aux[bi + 1u]) * np.bscale;
  if ((np.flags & 1u) != 0u) {
    red[lid] = v.x * v.x + v.y * v.y;
    workgroupBarrier();
    var st = 32u;
    loop {
      if (st == 0u) { break; }
      if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
      workgroupBarrier();
      st = st >> 1u;
    }
    let rr = inverseSqrt(red[0] / 128.0 + np.eps);
    v = v * rr * vec2<f32>(aux[np.w + 2u * lid], aux[np.w + 2u * lid + 1u]);
  }
  if ((np.flags & 2u) != 0u) {
    let c = rc[row * 64u + lid];
    let s = rs[row * 64u + lid];
    v = vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
  }
  qkv[idx] = pack2x16float(v);
}
"#;

/// Short attention: q `[m][ld_q]`, keys and values in one f16 panel. A row's
/// keys are the group its frame owns (`per` rows a frame; a single group with
/// `per` = rows for the prompt).
fn xattn_src(kmax: usize) -> String {
    format!(
        r#"
enable f16;
struct XP {{ m: u32, nh: u32, ldq: u32, qcol: u32, ldk: u32, kcol: u32, vcol: u32, per: u32, scl: f32, oscale: f32, _p0: u32, _p1: u32 }};
@group(0) @binding(0) var<storage, read> qp: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> kv: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> oh: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> groups: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> xp: XP;

@compute @workgroup_size(32)
fn s2v_xattn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{
  let row = wid.x * 32u + lid;
  if (row >= xp.m) {{ return; }}
  let hh = wid.y;
  let g = groups[row / xp.per];
  let q0 = (row * xp.ldq + xp.qcol + hh * 128u) / 4u;
  var qr: array<vec4<f32>, 32>;
  for (var t = 0u; t < 32u; t = t + 1u) {{ qr[t] = vec4<f32>(qp[q0 + t]); }}
  var s: array<f32, {kmax}>;
  var mx = -1.0e30;
  for (var j = 0u; j < g.y; j = j + 1u) {{
    let k0 = ((g.x + j) * xp.ldk + xp.kcol + hh * 128u) / 4u;
    var d = 0.0;
    for (var t = 0u; t < 32u; t = t + 1u) {{ d = d + dot(qr[t], vec4<f32>(kv[k0 + t])); }}
    let v = d * xp.scl;
    s[j] = v;
    mx = max(mx, v);
  }}
  var den = 0.0;
  for (var j = 0u; j < g.y; j = j + 1u) {{
    let e = exp(s[j] - mx);
    s[j] = e;
    den = den + e;
  }}
  var out: array<vec4<f32>, 32>;
  for (var j = 0u; j < g.y; j = j + 1u) {{
    let v0 = ((g.x + j) * xp.ldk + xp.vcol + hh * 128u) / 4u;
    let pv = s[j] / den;
    for (var t = 0u; t < 32u; t = t + 1u) {{ out[t] = out[t] + pv * vec4<f32>(kv[v0 + t]); }}
  }}
  let o0 = (row * xp.nh * 128u + hh * 128u) / 4u;
  for (var t = 0u; t < 32u; t = t + 1u) {{
    let a = out[t] * xp.oscale;
    oh[o0 + t] = vec2<u32>(pack2x16float(a.xy), pack2x16float(a.zw));
  }}
}}
"#
    )
}

const TAIL_SRC: &str = r#"
struct TP { h: u32, off: u32, bias: u32, mask: u32 };
@group(0) @binding(0) var<storage, read> br: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> aux: array<f32>;
@group(0) @binding(3) var<uniform> tp: TP;

// The noisy rows of the visual stream: embedder output + bias + mask row.
@compute @workgroup_size(256)
fn s2v_tail(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * (65535u * 256u);
  let row = i / tp.h;
  let c = i % tp.h;
  x[(row + tp.off) * tp.h + c] = br[i] + aux[tp.bias + c] + aux[tp.mask + c];
}
"#;

/// q4tp bytes → packed f16 plane, with source and destination offsets so one
/// raw buffer holds a block and one scratch holds a site's rows.
const DQ4_SRC: &str = r#"
struct DP { cols: u32, rows: u32, src_b: u32, dst_w: u32 };
@group(0) @binding(0) var<storage, read> qsrc: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> dp: DP;
fn dq_byte(off: u32) -> u32 { return (qsrc[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu; }
@compute @workgroup_size(256)
fn s2v_dq4(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let ppr = dp.cols / 2u;
  let idx = gid.y * (nwg.x * 256u) + gid.x;
  if (idx >= dp.rows * ppr) { return; }
  let row = idx / ppr;
  let pair = idx % ppr;
  let col0 = pair * 2u;
  let gpr = dp.cols >> 5u;
  let params_b = dp.rows * gpr * 16u;
  let codes_b = params_b + dp.rows * 4u;
  let cstride = (gpr * 5u + 7u) / 8u;
  let g = col0 >> 5u;
  let bit = g * 5u;
  let cb = dp.src_b + codes_b + row * cstride + (bit >> 3u);
  let sh = bit & 7u;
  var cv = dq_byte(cb);
  if (sh > 3u) { cv = cv | (dq_byte(cb + 1u) << 8u); }
  let pr = unpack2x16float(qsrc[((dp.src_b + params_b) >> 2u) + row]);
  let scale = exp2(pr.x + f32((cv >> sh) & 31u) * pr.y);
  let toff = (row * gpr + g) * 16u;
  let pp = col0 - g * 32u;
  let b0 = dq_byte(dp.src_b + toff + pp / 2u);
  let w0 = (f32(b0 & 0xFu) - 8.0) * scale;
  let w1 = (f32(b0 >> 4u) - 8.0) * scale;
  dst[dp.dst_w + idx] = pack2x16float(vec2<f32>(w0, w1));
}
"#;

const DQ8_SRC: &str = r#"
struct DP { cols: u32, rows: u32, has_col: u32, src_w: u32, rs_w: u32, cs_w: u32, dst_w: u32, _p: u32 };
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> p: DP;
fn f16at(w0: u32, i: u32) -> f32 {
  let v = unpack2x16float(src[w0 + i / 2u]);
  return select(v.x, v.y, (i & 1u) == 1u);
}
fn s8(w: u32, j: u32) -> f32 {
  let b = (w >> (8u * j)) & 0xFFu;
  return select(f32(b), f32(b) - 256.0, b > 127u);
}
@compute @workgroup_size(256)
fn s2v_dq8(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let q4 = p.cols / 4u;
  let idx = gid.y * (nwg.x * 256u) + gid.x;
  if (idx >= p.rows * q4) { return; }
  let r = idx / q4;
  let c4 = idx % q4;
  let w = src[p.src_w + idx];
  let rs = f16at(p.rs_w, r);
  var cs = vec4<f32>(1.0);
  if (p.has_col != 0u) {
    let a = unpack2x16float(src[p.cs_w + 2u * c4]);
    let b = unpack2x16float(src[p.cs_w + 2u * c4 + 1u]);
    cs = vec4<f32>(a.x, a.y, b.x, b.y);
  }
  let v = vec4<f32>(s8(w, 0u) * rs * cs.x, s8(w, 1u) * rs * cs.y, s8(w, 2u) * rs * cs.z, s8(w, 3u) * rs * cs.w);
  dst[p.dst_w / 2u + idx] = vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}
"#;

/// A dispatch that isn't a GEMM. Bind groups are prebuilt.
struct Disp {
    name: &'static str,
    pipe: Arc<wgpu::ComputePipeline>,
    bg: wgpu::BindGroup,
    grid: (u32, u32, u32),
}

enum Op {
    D(Disp),
    M(MmCall),
    F(super::zimage::Call),
}

impl Op {
    /// A name for the per-class profile.
    fn class(&self) -> String {
        match self {
            Op::D(d) => d.name.to_string(),
            Op::M(m) => {
                let [_, n, k] = m.mnk;
                format!("gemm n{n} k{k}")
            }
            Op::F(_) => "flash".to_string(),
        }
    }

    fn record(&self, pass: &mut wgpu::ComputePass) {
        match self {
            Op::D(d) => {
                pass.set_pipeline(&d.pipe);
                pass.set_bind_group(0, &d.bg, &[]);
                pass.dispatch_workgroups(d.grid.0, d.grid.1, d.grid.2);
            }
            Op::M(m) => m.record(pass),
            Op::F(f) => f.record(pass),
        }
    }
}

/// Grows an `aux` array of f32 parameters; `push` returns the word offset.
#[derive(Default)]
struct Aux(Vec<f32>);
impl Aux {
    fn push(&mut self, v: &[f32]) -> u32 {
        let o = self.0.len() as u32;
        self.0.extend_from_slice(v);
        o
    }
}

/// One stream's activations.
struct Seq {
    x: wgpu::Buffer,
    xn: wgpu::Buffer,
    qkv: wgpu::Buffer,
    att: wgpu::Buffer,
    br: wgpu::Buffer,
    hid: wgpu::Buffer,
    rc: wgpu::Buffer,
    rs: wgpu::Buffer,
}

impl Seq {
    fn new(c: &Ctx, m: usize, h: usize, ff: usize, tag: &str) -> Seq {
        let mp = pad_rows(m) as u64;
        let (h, ff) = (h as u64, ff as u64);
        Seq {
            x: sbuf(c, mp * h * 4, &format!("{tag}_x")),
            xn: sbuf(c, mp * h * 2, &format!("{tag}_xn")),
            qkv: sbuf(c, mp * 3 * h * 2, &format!("{tag}_qkv")),
            att: sbuf(c, mp * h * 2, &format!("{tag}_att")),
            br: sbuf(c, mp * h * 4, &format!("{tag}_br")),
            hid: sbuf(c, mp * ff * 2, &format!("{tag}_hid")),
            rc: sbuf(c, mp * 64 * 4, &format!("{tag}_rc")),
            rs: sbuf(c, mp * 64 * 4, &format!("{tag}_rs")),
        }
    }
}

/// The f16 planes one block expands into, shared by all blocks.
struct Planes {
    qkv: wgpu::Buffer,
    o: wgpu::Buffer,
    cq: wgpu::Buffer,
    ckv: wgpu::Buffer,
    co: wgpu::Buffer,
    aq: wgpu::Buffer,
    akv: wgpu::Buffer,
    ao: wgpu::Buffer,
    w1: wgpu::Buffer,
    w2: wgpu::Buffer,
}

impl Planes {
    fn new(c: &Ctx, h: usize, ff: usize) -> Planes {
        let hh = (h * h * 2) as u64;
        let hf = (h * ff * 2) as u64;
        Planes {
            qkv: sbuf(c, 3 * hh, "s2v_p_qkv"),
            o: sbuf(c, hh, "s2v_p_o"),
            cq: sbuf(c, hh, "s2v_p_cq"),
            ckv: sbuf(c, 2 * hh, "s2v_p_ckv"),
            co: sbuf(c, hh, "s2v_p_co"),
            aq: sbuf(c, hh, "s2v_p_aq"),
            akv: sbuf(c, 2 * hh, "s2v_p_akv"),
            ao: sbuf(c, hh, "s2v_p_ao"),
            w1: sbuf(c, hf, "s2v_p_w1"),
            w2: sbuf(c, hf, "s2v_p_w2"),
        }
    }
}

/// f16 range guards, as log2 of the factor the site is stored divided by.
#[derive(Clone, Copy, Debug)]
pub struct Guards {
    pub attn: i32,
    pub qkv: i32,
    pub hid: i32,
}

fn guards() -> Guards {
    let e = |k: &str, d: i32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    Guards { attn: e("CMF_S2V_ATTN_SHIFT", 0), qkv: e("CMF_S2V_QKV_SHIFT", 0), hid: e("CMF_S2V_HID_SHIFT", 0) }
}

/// One tensor's bytes in a block's raw buffer and where its plane goes.
struct RawT {
    idx: usize,
    rows: usize,
    cols: usize,
    plane: usize,
    row0: usize,
}

fn p2(e: i32) -> f32 {
    (e as f32).exp2()
}

struct Builder<'a> {
    c: &'a Ctx,
    aux: wgpu::Buffer,
    mods: wgpu::Buffer,
    groups_all: wgpu::Buffer,
    groups_audio: wgpu::Buffer,
    row: Arc<wgpu::ComputePipeline>,
    norm: Arc<wgpu::ComputePipeline>,
    xattn: Arc<wgpu::ComputePipeline>,
    h: usize,
    nh: usize,
    eps: f32,
    g: Guards,
    tiles: (MmCfg, MmCfg, MmCfg),
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn row_op(
        &self,
        s: &Seq,
        rows: usize,
        mode: u32,
        nclean: usize,
        xoff: usize,
        gs: (u32, u32),
        shs: (u32, u32),
        scs: (u32, u32),
        bias: u32,
        affine: (u32, u32),
        oscale: f32,
    ) -> Op {
        let u = ubuf(
            self.c,
            &[
                self.h as u32,
                mode,
                nclean as u32,
                xoff as u32,
                gs.0,
                gs.1,
                shs.0,
                shs.1,
                scs.0,
                scs.1,
                bias,
                affine.0,
                affine.1,
                0,
                1e-5f32.to_bits(),
                oscale.to_bits(),
            ],
        );
        let b = bg(self.c, &self.row, &[&s.br, &s.x, &s.xn, &self.mods, &self.aux, &u]);
        let r = rows as u32;
        Op::D(Disp { name: "row", pipe: self.row.clone(), bg: b, grid: (r.min(65535), r.div_ceil(65535), 1) })
    }

    #[allow(clippy::too_many_arguments)]
    fn norm_op(&self, panel: &wgpu::Buffer, s: &Seq, rows: usize, ld: usize, col: usize, flags: u32, w: u32, b_off: u32) -> Op {
        // The panel is stored ×2⁻ᵠ: the bias goes in at the same scale and the
        // norm's eps is scaled so the RMSNorm is exactly the unscaled one.
        let q = self.g.qkv;
        let eps = self.eps * p2(-2 * q);
        let u = ubuf(
            self.c,
            &[ld as u32, col as u32, flags, w, b_off, eps.to_bits(), p2(-q).to_bits(), rows as u32],
        );
        let b = bg(self.c, &self.norm, &[panel, &self.aux, &s.rc, &s.rs, &u]);
        let r = rows as u32;
        Op::D(Disp { name: "norm", pipe: self.norm.clone(), bg: b, grid: (self.nh as u32, r.min(65535), r.div_ceil(65535)) })
    }

    #[allow(clippy::too_many_arguments)]
    fn xattn_op(
        &self,
        qp: &wgpu::Buffer,
        ldq: usize,
        kv: &wgpu::Buffer,
        ldk: usize,
        kcol: usize,
        vcol: usize,
        out: &wgpu::Buffer,
        m: usize,
        per: usize,
        groups: &wgpu::Buffer,
    ) -> Op {
        let scl = 1.0 / (128f32).sqrt();
        let u = ubuf(
            self.c,
            &[
                m as u32,
                self.nh as u32,
                ldq as u32,
                0,
                ldk as u32,
                kcol as u32,
                vcol as u32,
                per as u32,
                scl.to_bits(),
                1f32.to_bits(),
                0,
                0,
            ],
        );
        let b = bg(self.c, &self.xattn, &[qp, kv, out, groups, &u]);
        Op::D(Disp {
            name: "xattn",
            pipe: self.xattn.clone(),
            bg: b,
            grid: ((m as u32).div_ceil(32), self.nh as u32, 1),
        })
    }

    /// Prompt cross-attention on the matrix units: `qi_flash` with the prompt's
    /// K|V panel as its whole key set.
    fn prompt_flash(&self, qp: &wgpu::Buffer, ldq: usize, kv: &wgpu::Buffer, nt: usize, out: &wgpu::Buffer, m: usize) -> Option<Op> {
        let f = super::qi21::prompt_flash_cfg();
        if super::qi21::flash_shared(f) > self.c.device.limits().max_compute_workgroup_storage_size {
            return None;
        }
        let pipe = pipeline(self.c, &super::qi21::flash_key(f), &super::qi21::flash_src(f)?, "qi_flash")?;
        let h = self.h;
        let scl = std::f32::consts::LOG2_E / (128f32).sqrt();
        let u = ubuf(
            self.c,
            &[
                m as u32,
                nt as u32,
                0,
                ldq as u32,
                h as u32,
                (2 * h) as u32,
                h as u32,
                scl.to_bits(),
                0,
                (2 * h) as u32,
                h as u32,
                1f32.to_bits(),
            ],
        );
        let b = bg(self.c, &pipe, &[qp, qp, kv, out, &self.groups_all, &u]);
        Some(Op::D(Disp {
            name: "prompt flash",
            pipe,
            bg: b,
            grid: ((m as u32).div_ceil(f.nw * 16), self.nh as u32, 1),
        }))
    }

    fn mm(&self, cfg: MmCfg, a: MmArgs, plane: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer) -> Option<Op> {
        Some(Op::M(mm_call(self.c, cfg, &a, plane, act, out)?))
    }
}

fn args(m: usize, n: usize, k: usize, ldo: usize, oscale: f32) -> MmArgs {
    MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: ldo as u32, ocol: 0, arow: 0, oscale, conv: [0; 3] }
}

/// Everything prepared for one render's geometry.
pub struct S2vDev {
    c: &'static Ctx,
    /// Dequant ops per block, then the block's ops: text, audio, visual.
    text_prog: Vec<(Vec<Op>, Vec<Op>)>,
    audio_prog: Vec<(Vec<Op>, Vec<Op>)>,
    visual_prog: Vec<(Vec<Op>, Vec<Op>)>,
    keep_raw: Vec<wgpu::Buffer>,
    s_small: Seq,
    s_vis: Seq,
    x0: wgpu::Buffer,
    tok: wgpu::Buffer,
    tcond: wgpu::Buffer,
    acond: wgpu::Buffer,
    tcache: wgpu::Buffer,
    t0: wgpu::Buffer,
    a0: wgpu::Buffer,
    mods: wgpu::Buffer,
    out: wgpu::Buffer,
    stage: wgpu::Buffer,
    embed: Vec<Op>,
    fin: Vec<Op>,
    text_tail: Vec<Op>,
    audio_tail: Vec<Op>,
    h: usize,
    g: S2vGeom,
    ntp: usize,
    nvis: usize,
    /// `[steps][n_visual·9h]` noisy-row modulation, `[steps][n_text·6h]`, `[steps][2h]`.
    vm: Vec<Vec<f32>>,
    tm: Vec<Vec<f32>>,
    fm: Vec<Vec<f32>>,
    offs: ModOffs,
    text_done: Vec<bool>,
    audio_for: Option<usize>,
    audio_feats: Vec<Vec<f32>>,
    out_b: Vec<f32>,
    dit: *const S2vDit,
    reference: Vec<f32>,
    cfg: s2v::S2vConfig,
    prof: bool,
}

#[derive(Clone, Copy)]
struct ModOffs {
    text: usize,
    fin: usize,
}

unsafe impl Send for S2vDev {}

fn encode_ops(c: &Ctx, ops: &[&[Op]]) -> wgpu::CommandBuffer {
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for list in ops {
            for op in *list {
                op.record(&mut pass);
            }
        }
    }
    enc.finish()
}

fn bytes_of(model: &CmfModel, idx: usize) -> &[u8] {
    model.entry_bytes(&model.tensors[idx])
}

/// Upload a block's raw tensor bytes as one buffer and build the expand ops.
fn raw_block(c: &Ctx, model: &Arc<CmfModel>, items: &[RawT], planes: &[&wgpu::Buffer]) -> Option<(wgpu::Buffer, Vec<Op>)> {
    let mut offs = Vec::with_capacity(items.len());
    let mut total = 0usize;
    for it in items {
        let e = model.tensors.get(it.idx)?;
        if !matches!(e.dtype, T::Q4TiledP | T::Q8_2f | T::Q8Row) || e.shape != [it.rows, it.cols] {
            return None;
        }
        offs.push(total);
        total += bytes_of(model, it.idx).len().next_multiple_of(256);
    }
    let raw = sbuf(c, total as u64, "s2v_raw");
    for (it, &o) in items.iter().zip(&offs) {
        let b = bytes_of(model, it.idx);
        if b.len() % 4 == 0 {
            c.queue.write_buffer(&raw, o as u64, b);
        } else {
            let mut v = b.to_vec();
            v.resize(v.len().next_multiple_of(4), 0);
            c.queue.write_buffer(&raw, o as u64, &v);
        }
    }
    // Flush the staged writes: left to the first real submit they pile up in
    // host-visible memory, 22 GB of them before the first dispatch.
    c.queue.submit(std::iter::empty());
    super::zimage::wait(c);
    let dq4 = pipeline(c, "s2v_dq4", DQ4_SRC, "s2v_dq4")?;
    let dq8 = pipeline(c, "s2v_dq8", DQ8_SRC, "s2v_dq8")?;
    let mut ops = Vec::with_capacity(items.len());
    for (it, &o) in items.iter().zip(&offs) {
        let e = &model.tensors[it.idx];
        let dst = planes[it.plane];
        let dst_w = (it.row0 * it.cols / 2) as u32;
        if e.dtype == T::Q4TiledP {
            let u = ubuf(c, &[it.cols as u32, it.rows as u32, o as u32, dst_w]);
            let b = bg(c, &dq4, &[&raw, dst, &u]);
            let wgs = ((it.rows * it.cols / 2) as u32).div_ceil(256);
            ops.push(Op::D(Disp { name: "dequant", pipe: dq4.clone(), bg: b, grid: (wgs.min(65535), wgs.div_ceil(65535), 1) }));
        } else {
            let sw = (o / 4) as u32;
            let n = it.rows * it.cols;
            let rs_w = sw + (n / 4) as u32;
            let cs_w = rs_w + (it.rows / 2) as u32;
            let u = ubuf(c, &[it.cols as u32, it.rows as u32, u32::from(e.dtype == T::Q8_2f), sw, rs_w, cs_w, dst_w, 0]);
            let b = bg(c, &dq8, &[&raw, dst, &u]);
            let wgs = ((n / 4) as u32).div_ceil(256);
            ops.push(Op::D(Disp { name: "dequant", pipe: dq8.clone(), bg: b, grid: (wgs.min(65535), wgs.div_ceil(65535), 1) }));
        }
    }
    Some((raw, ops))
}

/// Host tensor lookups for a block.
fn idx_of(model: &CmfModel, name: &str) -> Option<usize> {
    model.tensors.iter().position(|e| e.name == name)
}

fn lin_bias(l: &s2v::Lin) -> Vec<f32> {
    l.b.clone().unwrap_or_else(|| vec![0.0; l.rows()])
}

/// The render's device program, or `None` to run on the host.
pub fn prepare(
    dit: &S2vDit,
    g: &S2vGeom,
    cond: &S2vCond,
    ts: &[f32],
    sf: [f32; 3],
) -> Option<S2vDev> {
    if std::env::var("CMF_S2V_WGPU").as_deref() == Ok("0") {
        return None;
    }
    let c = zctx()?;
    let t_all = Instant::now();
    let model = dit.model.clone();
    let cfg = dit.cfg.clone();
    let (h, ff, nh) = (cfg.dim, cfg.ff, cfg.heads());
    let hg = guards();
    let m = g.total();
    let nt = cond.text.len() / cfg.in_text;
    let na = cond.audio[0].len() / cfg.in_audio;
    let small = nt.max(na);
    if nt > 512 || m > 70000 || m % 16 != 0 {
        eprintln!("s2v: no device path for {m} tokens and {nt} prompt tokens: running on the CPU");
        return None;
    }
    let mp = pad_rows(m);
    let budget = crate::gpu::vram_budget();
    let need = (mp * h * (4 + 2 + 6 + 2 + 4) + mp * ff * 2) as u64 + (h * h * 2 * 12 + h * ff * 4) as u64;
    let raw_total: u64 = model
        .tensors
        .iter()
        .filter(|e| e.name.starts_with("dit.") && matches!(e.dtype, T::Q4TiledP | T::Q8_2f | T::Q8Row))
        .map(|e| model.entry_bytes(e).len() as u64)
        .sum();
    if budget > 0 && need + raw_total > budget {
        eprintln!(
            "s2v: device path needs {:.1} GB, budget {:.1} GB: running on the CPU",
            (need + raw_total) as f64 / 1e9,
            budget as f64 / 1e9
        );
        return None;
    }

    // ── modulations for the whole render ──
    let t0 = Instant::now();
    let pe = dit.pooled_embed(&cond.pooled);
    let te_all: Vec<Vec<f32>> = ts.iter().map(|&t| dit.time_embed(t, &pe)).collect();
    let te0 = dit.time_embed(0.0, &pe);
    let si = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| s2v::silu(x)).collect() };
    let n_t = ts.len();
    let mut rows_all = si(&te0);
    for te in &te_all {
        rows_all.extend(si(te));
    }
    let mut clean_mods = Vec::with_capacity(cfg.n_visual * 9 * h);
    let mut vm: Vec<Vec<f32>> = vec![Vec::with_capacity(cfg.n_visual * 9 * h); n_t];
    let vnames: Vec<String> = (0..cfg.n_visual).map(|b| format!("dit.visual_transformer_blocks.{b}.visual_modulation.out_layer")).collect();
    for y in dit.modulate_all(&vnames, &rows_all, n_t + 1).ok()? {
        clean_mods.extend_from_slice(&y[..9 * h]);
        for (s, v) in vm.iter_mut().enumerate() {
            v.extend_from_slice(&y[(s + 1) * 9 * h..(s + 2) * 9 * h]);
        }
    }
    let mut tm: Vec<Vec<f32>> = vec![Vec::with_capacity(cfg.n_text * 6 * h); n_t];
    let t_rows = &rows_all[cfg.time_dim..];
    for b in 0..cfg.n_text {
        let y = dit
            .modulate(&format!("dit.text_transformer_blocks.{b}.text_modulation.out_layer"), t_rows, n_t)
            .ok()?;
        for (s, v) in tm.iter_mut().enumerate() {
            v.extend_from_slice(&y[s * 6 * h..(s + 1) * 6 * h]);
        }
    }
    let y = dit.modulate("dit.out_layer.modulation.out_layer", t_rows, n_t).ok()?;
    let fm: Vec<Vec<f32>> = (0..n_t).map(|s| y[s * 2 * h..(s + 1) * 2 * h].to_vec()).collect();
    crate::s2vgen::prof("modulations", t0);

    // Mods buffer: [visual][clean, noisy][9h] · [text][6h] · [2h].
    let text_off = cfg.n_visual * 2 * 9 * h;
    let fin_off = text_off + cfg.n_text * 6 * h;
    let mods = sbuf(c, ((fin_off + 2 * h) * 4) as u64, "s2v_mods");
    for b in 0..cfg.n_visual {
        c.queue.write_buffer(
            &mods,
            (b * 2 * 9 * h * 4) as u64,
            bytemuck::cast_slice(&clean_mods[b * 9 * h..(b + 1) * 9 * h]),
        );
    }

    // ── per-block parameters and resident raw weights ──
    let t0 = Instant::now();
    let w1_tile = std::env::var("CMF_S2V_TILE_W1")
        .ok()
        .and_then(|s| {
            let v: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            (v.len() == 5).then(|| MmCfg::new(v[0], v[1], v[2], v[3], v[4], Epi::Gelu))
        })
        .filter(|g| g.valid())
        .unwrap_or_else(|| default_cfg(Epi::Gelu));
    // Measured on an L40S at 36736 rows: the 3h and FF output widths prefer the
    // taller tile, the h-wide O GEMMs the square one.
    let site = |env: &str, epi: Epi, d: (u32, u32, u32, u32, u32)| -> MmCfg {
        let v: Vec<u32> = std::env::var(env)
            .ok()
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_default();
        let g = if v.len() == 5 {
            MmCfg::new(v[0], v[1], v[2], v[3], v[4], epi)
        } else {
            MmCfg::new(d.0, d.1, d.2, d.3, d.4, epi)
        };
        if g.valid() { g } else { default_cfg(epi) }
    };
    let qkv_tile = site("CMF_S2V_TILE_QKV", Epi::F16, (256, 128, 32, 4, 2));
    let w2_tile = site("CMF_S2V_TILE_W2", Epi::F32, (256, 128, 32, 4, 2));
    let tiles = (qkv_tile, default_cfg(Epi::F32), w1_tile);
    let flash = default_flash();
    let s_vis = Seq::new(c, m, h, ff, "s2v_v");
    let s_small = Seq::new(c, small, h, ff, "s2v_t");
    let planes = Planes::new(c, h, ff);
    let mut aux = Aux::default();
    let zero_h = aux.push(&vec![0f32; h]);
    let pl_list: Vec<&wgpu::Buffer> = vec![
        &planes.qkv, &planes.o, &planes.cq, &planes.ckv, &planes.co, &planes.aq, &planes.akv, &planes.ao, &planes.w1, &planes.w2,
    ];
    // (dq ops, per-block parameter offsets) gathered first; the bind groups
    // need the finished `aux` buffer.
    struct BlockP {
        dq: Vec<Op>,
        off: [u32; 16],
    }
    let attn_names = |p: &str| -> Option<(usize, usize, usize, usize)> {
        Some((
            idx_of(&model, &format!("{p}.to_query.weight"))?,
            idx_of(&model, &format!("{p}.to_key.weight"))?,
            idx_of(&model, &format!("{p}.to_value.weight"))?,
            idx_of(&model, &format!("{p}.out_layer.weight"))?,
        ))
    };
    let mut keep_raw = Vec::new();
    let push_attn = |aux: &mut Aux, a: &s2v::Attn| -> [u32; 6] {
        [
            aux.push(&a.qn),
            aux.push(&a.kn),
            aux.push(&lin_bias(&a.q)),
            aux.push(&lin_bias(&a.k)),
            aux.push(&lin_bias(&a.v)),
            aux.push(&lin_bias(&a.o)),
        ]
    };
    let mut visual_p: Vec<BlockP> = Vec::with_capacity(cfg.n_visual);
    let mut v2a_rest: Vec<[u32; 6]> = Vec::with_capacity(cfg.n_visual);
    for (bi, b) in dit.visual.iter().enumerate() {
        let p = format!("dit.visual_transformer_blocks.{bi}");
        let (q, k, v, o) = attn_names(&format!("{p}.self_attention"))?;
        let (cq, ck, cv, co) = attn_names(&format!("{p}.cross_attention"))?;
        let mut items = vec![
            RawT { idx: q, rows: h, cols: h, plane: 0, row0: 0 },
            RawT { idx: k, rows: h, cols: h, plane: 0, row0: h },
            RawT { idx: v, rows: h, cols: h, plane: 0, row0: 2 * h },
            RawT { idx: o, rows: h, cols: h, plane: 1, row0: 0 },
            RawT { idx: cq, rows: h, cols: h, plane: 2, row0: 0 },
            RawT { idx: ck, rows: h, cols: h, plane: 3, row0: 0 },
            RawT { idx: cv, rows: h, cols: h, plane: 3, row0: h },
            RawT { idx: co, rows: h, cols: h, plane: 4, row0: 0 },
            RawT { idx: idx_of(&model, &format!("{p}.feed_forward.in_layer.weight"))?, rows: ff, cols: h, plane: 8, row0: 0 },
            RawT { idx: idx_of(&model, &format!("{p}.feed_forward.out_layer.weight"))?, rows: h, cols: ff, plane: 9, row0: 0 },
        ];
        if b.v2a.is_some() {
            let (aq, ak, av, ao) = attn_names(&format!("{p}.cross_attention_v2a"))?;
            items.extend([
                RawT { idx: aq, rows: h, cols: h, plane: 5, row0: 0 },
                RawT { idx: ak, rows: h, cols: h, plane: 6, row0: 0 },
                RawT { idx: av, rows: h, cols: h, plane: 6, row0: h },
                RawT { idx: ao, rows: h, cols: h, plane: 7, row0: 0 },
            ]);
        }
        let (raw, dq) = raw_block(c, &model, &items, &pl_list)?;
        keep_raw.push(raw);
        let s = push_attn(&mut aux, &b.sa);
        let ca = push_attn(&mut aux, &b.ca);
        let (av2, nw, nb) = match &b.v2a {
            Some((n, a)) => (push_attn(&mut aux, a), aux.push(&n.w), aux.push(&n.b)),
            None => ([0; 6], 0, 0),
        };
        let mut off = [0u32; 16];
        off[..6].copy_from_slice(&s);
        off[6..12].copy_from_slice(&ca);
        off[13] = nw;
        off[14] = nb;
        v2a_rest.push(av2);
        visual_p.push(BlockP { dq, off });
    }
    let mut text_p = Vec::with_capacity(cfg.n_text);
    for (bi, b) in dit.text.iter().enumerate() {
        let p = format!("dit.text_transformer_blocks.{bi}");
        let (q, k, v, o) = attn_names(&format!("{p}.self_attention"))?;
        let items = vec![
            RawT { idx: q, rows: h, cols: h, plane: 0, row0: 0 },
            RawT { idx: k, rows: h, cols: h, plane: 0, row0: h },
            RawT { idx: v, rows: h, cols: h, plane: 0, row0: 2 * h },
            RawT { idx: o, rows: h, cols: h, plane: 1, row0: 0 },
            RawT { idx: idx_of(&model, &format!("{p}.feed_forward.in_layer.weight"))?, rows: ff, cols: h, plane: 8, row0: 0 },
            RawT { idx: idx_of(&model, &format!("{p}.feed_forward.out_layer.weight"))?, rows: h, cols: ff, plane: 9, row0: 0 },
        ];
        let (raw, dq) = raw_block(c, &model, &items, &pl_list)?;
        keep_raw.push(raw);
        let s = push_attn(&mut aux, &b.attn);
        text_p.push((dq, s));
    }
    let mut audio_p = Vec::with_capacity(cfg.n_audio);
    for (bi, b) in dit.audio.iter().enumerate() {
        let p = format!("dit.audio_transformer_blocks.{bi}");
        let (q, k, v, o) = attn_names(&format!("{p}.self_attention"))?;
        let items = vec![
            RawT { idx: q, rows: h, cols: h, plane: 0, row0: 0 },
            RawT { idx: k, rows: h, cols: h, plane: 0, row0: h },
            RawT { idx: v, rows: h, cols: h, plane: 0, row0: 2 * h },
            RawT { idx: o, rows: h, cols: h, plane: 1, row0: 0 },
            RawT { idx: idx_of(&model, &format!("{p}.feed_forward.in_layer.weight"))?, rows: ff, cols: h, plane: 8, row0: 0 },
            RawT { idx: idx_of(&model, &format!("{p}.feed_forward.out_layer.weight"))?, rows: h, cols: ff, plane: 9, row0: 0 },
        ];
        let (raw, dq) = raw_block(c, &model, &items, &pl_list)?;
        keep_raw.push(raw);
        let s = push_attn(&mut aux, &b.attn);
        let n1 = (aux.push(&b.sa_norm.w), aux.push(&b.sa_norm.b));
        let n2 = (aux.push(&b.ff_norm.w), aux.push(&b.ff_norm.b));
        audio_p.push((dq, s, n1, n2));
    }
    let emb_b = aux.push(&lin_bias(&dit.emb1));
    let mask0 = aux.push(&dit.mask[..h]);
    let out_b = lin_bias(&dit.out_w);
    let aux_buf = sbuf_init(c, bytemuck::cast_slice(&aux.0), "s2v_aux");
    crate::s2vgen::prof("raw weights", t0);
    if std::env::var("CMF_S2V_PROF").is_ok() {
        let l = c.device.limits();
        eprintln!(
            "s2v device: binding limit {} MB, buffer limit {} MB, raw weights {:.1} GB",
            l.max_storage_buffer_binding_size >> 20,
            l.max_buffer_size >> 20,
            raw_total as f64 / 1e9
        );
    }

    // ── conditioning streams ──
    let ntp = small;
    let tcond = sbuf(c, (pad_rows(ntp) * h * 2) as u64, "s2v_tcond");
    let acond = sbuf(c, (pad_rows(ntp) * h * 2) as u64, "s2v_acond");
    let tcache = sbuf(c, (n_t * pad_rows(ntp) * h * 2) as u64, "s2v_tcache");
    let t0b = sbuf_init(c, bytemuck::cast_slice(&dit.embed_text(&cond.text)), "s2v_t0");
    let a0b = sbuf(c, (pad_rows(ntp) * h * 4) as u64, "s2v_a0");
    let tkv = sbuf(c, (pad_rows(nt) * 2 * h * 2) as u64, "s2v_tkv");
    let akv = sbuf(c, (pad_rows(na) * 2 * h * 2) as u64, "s2v_akv");

    // Rope tables.
    let (vc, vs) = s2v::rope_visual(&cfg, g, sf);
    c.queue.write_buffer(&s_vis.rc, 0, bytemuck::cast_slice(&vc));
    c.queue.write_buffer(&s_vis.rs, 0, bytemuck::cast_slice(&vs));
    drop((vc, vs));

    let groups_audio = {
        let v: Vec<u32> = audio_groups(na, g.lt).into_iter().flat_map(|(a, b)| [a as u32, b as u32]).collect();
        sbuf_init(c, bytemuck::cast_slice(&v), "s2v_groups")
    };
    let groups_all = sbuf_init(c, bytemuck::cast_slice(&[0u32, nt as u32]), "s2v_gall");
    let groups_t = groups_all.clone();
    let groups_a = sbuf_init(c, bytemuck::cast_slice(&[0u32, na as u32]), "s2v_gsm");

    let bld = Builder {
        c,
        aux: aux_buf,
        mods: mods.clone(),
        groups_all: groups_all.clone(),
        groups_audio: groups_audio.clone(),
        row: pipeline(c, "s2v_row", ROW_SRC, "s2v_row")?,
        norm: pipeline(c, "s2v_norm", NORM_SRC, "s2v_norm")?,
        xattn: pipeline(c, "s2v_xattn_512", &xattn_src(512), "s2v_xattn")?,
        h,
        nh,
        eps: 1e-5,
        g: hg,
        tiles,
    };
    let (attn_sc, qkv_sh, hid_sh) = (p2(-hg.attn), hg.qkv, hg.hid);
    let mm16 = |a: MmArgs, p: &wgpu::Buffer, act: &wgpu::Buffer, o: &wgpu::Buffer| bld.mm(bld.tiles.0, a, p, act, o);
    let mm32 = |a: MmArgs, p: &wgpu::Buffer, act: &wgpu::Buffer, o: &wgpu::Buffer| bld.mm(bld.tiles.1, a, p, act, o);

    // ── the short streams: one block program over `Seq` rows ──
    let short_block = |s: &Seq, nrows: usize, off: [u32; 6], pre: (u32, u32), pre_ffn: (u32, u32), bias_gate: (u32, u32), modul: bool, gr: &wgpu::Buffer| -> Option<Vec<Op>> {
        // pre: (shift word, scale word) or an affine (w, b) when unmodulated.
        let (mode_pre, sh, sc, af): (u32, (u32, u32), (u32, u32), (u32, u32)) = if modul {
            (8, (pre.0, pre.0), (pre.1, pre.1), (0, 0))
        } else {
            (16, (0, 0), (0, 0), pre)
        };
        let mut v = Vec::new();
        v.push(bld.row_op(s, nrows, mode_pre, nrows + 1, 0, (0, 0), sh, sc, zero_h, af, attn_sc));
        let aq = args(nrows, 3 * h, h, 3 * h, p2(hg.attn - qkv_sh));
        v.push(mm16(aq, &planes.qkv, &s.xn, &s.qkv)?);
        v.push(bld.norm_op(&s.qkv, s, nrows, 3 * h, 0, 3, off[0], off[2]));
        v.push(bld.norm_op(&s.qkv, s, nrows, 3 * h, h, 3, off[1], off[3]));
        v.push(bld.norm_op(&s.qkv, s, nrows, 3 * h, 2 * h, 0, 0, off[4]));
        v.push(bld.xattn_op(&s.qkv, 3 * h, &s.qkv, 3 * h, h, 2 * h, &s.att, nrows, nrows, gr));
        v.push(mm32(args(nrows, h, h, h, p2(qkv_sh)), &planes.o, &s.att, &s.br)?);
        let (gm, gs): (u32, (u32, u32)) = if modul { (2, (bias_gate.0, bias_gate.0)) } else { (0, (0, 0)) };
        let (mode_ffn, sh2, sc2, af2): (u32, (u32, u32), (u32, u32), (u32, u32)) = if modul {
            (8, (pre_ffn.0, pre_ffn.0), (pre_ffn.1, pre_ffn.1), (0, 0))
        } else {
            (16, (0, 0), (0, 0), pre_ffn)
        };
        v.push(bld.row_op(s, nrows, 1 | gm | mode_ffn, nrows + 1, 0, gs, sh2, sc2, off[5], af2, 1.0));
        let gcfg = bld.tiles.2;
        v.push(bld.mm(gcfg, args(nrows, ff, h, ff, p2(-hid_sh)), &planes.w1, &s.xn, &s.hid)?);
        v.push(mm32(args(nrows, h, ff, h, p2(hid_sh)), &planes.w2, &s.hid, &s.br)?);
        let (gs2, _): ((u32, u32), u32) = if modul { ((bias_gate.1, bias_gate.1), 0) } else { ((0, 0), 0) };
        v.push(bld.row_op(s, nrows, 1 | gm, nrows + 1, 0, gs2, (0, 0), (0, 0), zero_h, (0, 0), 1.0));
        Some(v)
    };

    let text_mods = |b: usize, chunk: usize| -> u32 { (text_off + b * 6 * h + chunk * h) as u32 };
    let mut text_prog = Vec::with_capacity(cfg.n_text);
    for (bi, (dq, off)) in text_p.into_iter().enumerate() {
        let ops = short_block(
            &s_small,
            nt,
            off,
            (text_mods(bi, 0), text_mods(bi, 1)),
            (text_mods(bi, 3), text_mods(bi, 4)),
            (text_mods(bi, 2), text_mods(bi, 5)),
            true,
            &groups_t,
        )?;
        text_prog.push((dq, ops));
    }
    let mut audio_prog = Vec::with_capacity(cfg.n_audio);
    for (dq, off, n1, n2) in audio_p {
        let ops = short_block(&s_small, na, off, n1, n2, (0, 0), false, &groups_a)?;
        audio_prog.push((dq, ops));
    }
    // Stream → f16 conditioning panel, with a row-op copy.
    let tail_to = |s: &Seq, rows: usize| bld.row_op(s, rows, 32, rows + 1, 0, (0, 0), (0, 0), (0, 0), zero_h, (0, 0), 1.0);
    let text_tail = vec![tail_to(&s_small, nt)];
    let audio_tail = vec![tail_to(&s_small, na)];

    // ── the visual program ──
    let vmods = |b: usize, row: usize, chunk: usize| -> u32 { (b * 2 * 9 * h + row * 9 * h + chunk * h) as u32 };
    let two = |b: usize, chunk: usize| -> (u32, u32) { (vmods(b, 0, chunk), vmods(b, 1, chunk)) };
    let nclean = g.clean();
    let mut visual_prog = Vec::with_capacity(cfg.n_visual);
    for (bi, bp) in visual_p.into_iter().enumerate() {
        let off = bp.off;
        let mut v = Vec::with_capacity(32);
        // self-attention
        v.push(bld.row_op(&s_vis, m, 8, nclean, 0, (0, 0), two(bi, 0), two(bi, 1), zero_h, (0, 0), attn_sc));
        v.push(mm16(args(m, 3 * h, h, 3 * h, p2(hg.attn - qkv_sh)), &planes.qkv, &s_vis.xn, &s_vis.qkv)?);
        v.push(bld.norm_op(&s_vis.qkv, &s_vis, m, 3 * h, 0, 3, off[0], off[2]));
        v.push(bld.norm_op(&s_vis.qkv, &s_vis, m, 3 * h, h, 3, off[1], off[3]));
        v.push(bld.norm_op(&s_vis.qkv, &s_vis, m, 3 * h, 2 * h, 0, 0, off[4]));
        for f in flash_calls(c, flash, nh, &s_vis.qkv, &s_vis.att, &[(0, m)])? {
            v.push(Op::F(f));
        }
        v.push(mm32(args(m, h, h, h, p2(qkv_sh)), &planes.o, &s_vis.att, &s_vis.br)?);
        // residual, then the prompt cross-attention's pre-norm
        v.push(bld.row_op(&s_vis, m, 1 | 2 | 8, nclean, 0, two(bi, 2), two(bi, 3), two(bi, 4), off[5], (0, 0), attn_sc));
        v.push(mm16(args(m, h, h, 3 * h, p2(hg.attn - qkv_sh)), &planes.cq, &s_vis.xn, &s_vis.qkv)?);
        v.push(bld.norm_op(&s_vis.qkv, &s_vis, m, 3 * h, 0, 1, off[6], off[8]));
        v.push(mm16(args(nt, 2 * h, h, 2 * h, p2(-qkv_sh)), &planes.ckv, &tcond, &tkv)?);
        v.push(bld.norm_op(&tkv, &s_vis, nt, 2 * h, 0, 1, off[7], off[9]));
        v.push(bld.norm_op(&tkv, &s_vis, nt, 2 * h, h, 0, 0, off[10]));
        match bld.prompt_flash(&s_vis.qkv, 3 * h, &tkv, nt, &s_vis.att, m) {
            Some(op) => v.push(op),
            None => v.push(bld.xattn_op(&s_vis.qkv, 3 * h, &tkv, 2 * h, 0, h, &s_vis.att, m, m, &bld.groups_all)),
        }
        v.push(mm32(args(m, h, h, h, p2(qkv_sh)), &planes.co, &s_vis.att, &s_vis.br)?);
        let has_a = dit.visual[bi].v2a.is_some();
        if has_a {
            // residual only; the adapter reads a new norm of the noisy tail.
            v.push(bld.row_op(&s_vis, m, 1 | 2, nclean, 0, two(bi, 5), (0, 0), (0, 0), off[11], (0, 0), 1.0));
            let a = v2a_rest[bi];
            v.push(bld.row_op(&s_vis, g.nvis, 16, nclean, nclean, (0, 0), (0, 0), (0, 0), zero_h, (off[13], off[14]), attn_sc));
            v.push(mm16(args(g.nvis, h, h, 3 * h, p2(hg.attn - qkv_sh)), &planes.aq, &s_vis.xn, &s_vis.qkv)?);
            v.push(bld.norm_op(&s_vis.qkv, &s_vis, g.nvis, 3 * h, 0, 1, a[0], a[2]));
            v.push(mm16(args(na, 2 * h, h, 2 * h, p2(-qkv_sh)), &planes.akv, &acond, &akv)?);
            v.push(bld.norm_op(&akv, &s_vis, na, 2 * h, 0, 1, a[1], a[3]));
            v.push(bld.norm_op(&akv, &s_vis, na, 2 * h, h, 0, 0, a[4]));
            v.push(bld.xattn_op(&s_vis.qkv, 3 * h, &akv, 2 * h, 0, h, &s_vis.att, g.nvis, g.per_frame(), &bld.groups_audio));
            v.push(mm32(args(g.nvis, h, h, h, p2(qkv_sh)), &planes.ao, &s_vis.att, &s_vis.br)?);
            v.push(bld.row_op(&s_vis, g.nvis, 1, nclean, nclean, (0, 0), (0, 0), (0, 0), a[5], (0, 0), 1.0));
            v.push(bld.row_op(&s_vis, m, 8, nclean, 0, (0, 0), two(bi, 6), two(bi, 7), zero_h, (0, 0), 1.0));
        } else {
            v.push(bld.row_op(&s_vis, m, 1 | 2 | 8, nclean, 0, two(bi, 5), two(bi, 6), two(bi, 7), off[11], (0, 0), 1.0));
        }
        // feed-forward
        v.push(bld.mm(bld.tiles.2, args(m, ff, h, ff, p2(-hid_sh)), &planes.w1, &s_vis.xn, &s_vis.hid)?);
        v.push(bld.mm(w2_tile, args(m, h, ff, h, p2(hid_sh)), &planes.w2, &s_vis.hid, &s_vis.br)?);
        v.push(bld.row_op(&s_vis, m, 1 | 2, nclean, 0, two(bi, 8), (0, 0), (0, 0), zero_h, (0, 0), 1.0));
        visual_prog.push((bp.dq, v));
    }

    // Embedder of the noisy rows and the final head, each a GEMM on a plane
    // of the same f16 kind; the head's 320 columns pad to the tile's 384.
    let nvis = g.nvis;
    let to_f16 = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|x| cortiq_core::quant::f32_to_f16(*x).to_le_bytes()).collect()
    };
    let emb_plane = {
        let Proj::F32 { w, .. } = &dit.emb1.w else { return None };
        sbuf_init(c, &to_f16(w), "s2v_emb")
    };
    let tok = sbuf(c, (pad_rows(nvis) * 64 * 2) as u64, "s2v_tok");
    let tail = pipeline(c, "s2v_tail", TAIL_SRC, "s2v_tail")?;
    let mut embed = vec![mm32(args(nvis, h, 64, h, 1.0), &emb_plane, &tok, &s_vis.br)?];
    {
        let u = ubuf(c, &[h as u32, nclean as u32, emb_b, mask0]);
        let b = bg(c, &tail, &[&s_vis.br, &s_vis.x, &bld.aux, &u]);
        let wgs = ((nvis * h) as u32).div_ceil(256);
        embed.push(Op::D(Disp { name: "tail", pipe: tail, bg: b, grid: (wgs.min(65535), wgs.div_ceil(65535), 1) }));
    }
    let head_plane = {
        let Proj::F32 { w, .. } = &dit.out_w.w else { return None };
        let mut padded = w.clone();
        padded.resize(384 * h, 0.0);
        sbuf_init(c, &to_f16(&padded), "s2v_head")
    };
    let out = sbuf(c, (pad_rows(nvis) * 384 * 4) as u64, "s2v_out");
    let fin = vec![
        bld.row_op(&s_vis, nvis, 8, nclean, nclean, (0, 0), ((fin_off) as u32, (fin_off) as u32), ((fin_off + h) as u32, (fin_off + h) as u32), zero_h, (0, 0), 1.0),
        mm32(args(nvis, 384, h, 384, 1.0), &head_plane, &s_vis.xn, &out)?,
    ];
    let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("s2v_stage"),
        size: (nvis * 384 * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let x0 = sbuf(c, (g.clean().max(1) * h * 4) as u64, "s2v_x0");
    // Rows of the short streams' RoPE are per stream: text once, audio per chunk.
    let (tc, tsn) = s2v::rope_1d(&(0..small as i32).collect::<Vec<_>>());
    c.queue.write_buffer(&s_small.rc, 0, bytemuck::cast_slice(&tc));
    c.queue.write_buffer(&s_small.rs, 0, bytemuck::cast_slice(&tsn));

    c.queue.submit(std::iter::empty());
    super::zimage::wait(c);
    crate::s2vgen::prof("device program", t_all);
    Some(S2vDev {
        c,
        text_prog,
        audio_prog,
        visual_prog,
        keep_raw,
        s_small,
        s_vis,
        x0,
        tok,
        tcond,
        acond,
        tcache,
        t0: t0b,
        a0: a0b,
        mods,
        out,
        stage,
        embed,
        fin,
        text_tail,
        audio_tail,
        h,
        g: *g,
        ntp,
        nvis,
        vm,
        tm,
        fm,
        offs: ModOffs { text: text_off, fin: fin_off },
        text_done: vec![false; n_t],
        audio_for: None,
        audio_feats: cond.audio.clone(),
        out_b,
        dit: dit as *const S2vDit,
        reference: cond.reference.clone(),
        cfg,
        prof: std::env::var("CMF_S2V_PROF").is_ok(),
    })
}

impl S2vDev {
    /// `CMF_S2V_CLASS=1`: time every op of two visual blocks (one with the speech
    /// adapter), one submission each, the second of two runs. `=exit` stops there.
    fn class_profile(&self) {
        let mut by: std::collections::BTreeMap<String, f64> = Default::default();
        for b in [0usize, 1usize] {
            let (dq, ops) = &self.visual_prog[b];
            for op in dq.iter().chain(ops.iter()) {
                let mut t = 0.0;
                for _ in 0..2 {
                    let t0 = Instant::now();
                    let cb = encode_ops(self.c, &[std::slice::from_ref(op)]);
                    self.c.queue.submit(Some(cb));
                    super::zimage::wait(self.c);
                    t = t0.elapsed().as_secs_f64();
                }
                *by.entry(op.class()).or_default() += t;
            }
        }
        let tot: f64 = by.values().sum();
        for (k, v) in &by {
            eprintln!("  {k:<22} {:>9.1} ms  {:>5.1} %", v * 1e3, v / tot * 100.0);
        }
        eprintln!("  two blocks: {:.1} ms", tot * 1e3);
        if std::env::var("CMF_S2V_CLASS").as_deref() == Ok("exit") {
            std::process::exit(0);
        }
    }

    fn submit_blocks(&self, blocks: &[(Vec<Op>, Vec<Op>)]) {
        for (dq, ops) in blocks {
            let cb = encode_ops(self.c, &[dq, ops]);
            self.c.queue.submit(Some(cb));
        }
    }

    fn copy(&self, src: &wgpu::Buffer, s_off: u64, dst: &wgpu::Buffer, d_off: u64, n: u64) {
        let mut enc = self.c.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, s_off, dst, d_off, n);
        self.c.queue.submit(Some(enc.finish()));
    }

    fn write_mods(&self, step: usize) {
        let h = self.h;
        let (nv, mods) = (self.cfg.n_visual, &self.mods);
        // Noisy rows only: the clean rows never change.
        for b in 0..nv {
            let at = (b * 2 * 9 * h + 9 * h) * 4;
            self.c.queue.write_buffer(mods, at as u64, bytemuck::cast_slice(&self.vm[step][b * 9 * h..(b + 1) * 9 * h]));
        }
        self.c.queue.write_buffer(mods, (self.offs.text * 4) as u64, bytemuck::cast_slice(&self.tm[step]));
        self.c.queue.write_buffer(mods, (self.offs.fin * 4) as u64, bytemuck::cast_slice(&self.fm[step]));
    }

    fn run_audio(&mut self, chunk: usize) {
        let h = self.h;
        // SAFETY: the DiT outlives the stepper: `generate` drops the stepper first.
        let dit = unsafe { &*self.dit };
        let a0 = dit.embed_audio(&self.audio_feats[chunk]);
        let n = a0.len() / h;
        self.c.queue.write_buffer(&self.s_small.x, 0, bytemuck::cast_slice(&a0));
        let (ac, asn) = s2v::rope_1d(&s2v::audio_rope_pos(n, self.g.lt));
        self.c.queue.write_buffer(&self.s_small.rc, 0, bytemuck::cast_slice(&ac));
        self.c.queue.write_buffer(&self.s_small.rs, 0, bytemuck::cast_slice(&asn));
        self.submit_blocks(&self.audio_prog);
        let cb = encode_ops(self.c, &[&self.audio_tail]);
        self.c.queue.submit(Some(cb));
        self.copy(&self.s_small.xn, 0, &self.acond, 0, (n * h * 2) as u64);
        self.audio_for = Some(chunk);
        self.text_rope();
    }

    fn text_rope(&self) {
        let (tc, tsn) = s2v::rope_1d(&(0..self.ntp as i32).collect::<Vec<_>>());
        self.c.queue.write_buffer(&self.s_small.rc, 0, bytemuck::cast_slice(&tc));
        self.c.queue.write_buffer(&self.s_small.rs, 0, bytemuck::cast_slice(&tsn));
    }

    fn run_text(&mut self, step: usize) {
        let h = self.h;
        let per = (pad_rows(self.ntp) * h * 2) as u64;
        if !self.text_done[step] {
            let nt = self.t0.size() as usize / (h * 4);
            self.copy(&self.t0, 0, &self.s_small.x, 0, (nt * h * 4) as u64);
            self.submit_blocks(&self.text_prog);
            let cb = encode_ops(self.c, &[&self.text_tail]);
            self.c.queue.submit(Some(cb));
            self.copy(&self.s_small.xn, 0, &self.tcache, step as u64 * per, (nt * h * 2) as u64);
            self.text_done[step] = true;
        }
        self.copy(&self.tcache, step as u64 * per, &self.tcond, 0, per);
    }

    fn step_inner(&mut self, step: usize, chunk: usize, x: &[f32], memory: &[Vec<f32>; 3]) -> Result<Vec<f32>, String> {
        let h = self.h;
        let g = self.g;
        let tt = Instant::now();
        self.write_mods(step);
        if self.audio_for != Some(chunk) {
            self.run_audio(chunk);
            // SAFETY: as in `run_audio`.
            let dit = unsafe { &*self.dit };
            let clean = dit.embed_clean(&self.reference, memory, &g);
            self.c.queue.write_buffer(&self.x0, 0, bytemuck::cast_slice(&clean));
        }
        self.run_text(step);
        let toks = s2v::patchify(x, self.cfg.in_visual, 1, 2, g.lh, g.lw);
        let packed: Vec<u8> = toks.iter().flat_map(|v| cortiq_core::quant::f32_to_f16(*v).to_le_bytes()).collect();
        self.c.queue.write_buffer(&self.tok, 0, &packed);
        self.copy(&self.x0, 0, &self.s_vis.x, 0, (g.clean() * h * 4) as u64);
        let cb = encode_ops(self.c, &[&self.embed]);
        self.c.queue.submit(Some(cb));
        if std::env::var("CMF_S2V_CLASS").is_ok() && step == 0 {
            self.class_profile();
        }
        self.submit_blocks(&self.visual_prog);
        let cb = encode_ops(self.c, &[&self.fin]);
        let mut enc = self.c.device.create_command_encoder(&Default::default());
        let bytes = (self.nvis * 384 * 4) as u64;
        enc.copy_buffer_to_buffer(&self.out, 0, &self.stage, 0, bytes);
        self.c.queue.submit([cb, enc.finish()]);
        let slice = self.stage.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.c.device.poll(wgpu::PollType::wait_indefinitely());
        let mapped = slice.get_mapped_range().map_err(|e| format!("s2v readback: {e:?}"))?;
        let raw: &[f32] = bytemuck::cast_slice(&mapped);
        let mut y = vec![0f32; self.nvis * 320];
        for r in 0..self.nvis {
            for c in 0..320 {
                y[r * 320 + c] = raw[r * 384 + c] + self.out_b[c];
            }
        }
        drop(mapped);
        self.stage.unmap();
        if self.prof {
            eprintln!("s2v device step {step}: {:.2} s", tt.elapsed().as_secs_f64());
        }
        Ok(s2v::unpatchify_out(&y, &g, self.cfg.out_visual))
    }
}

impl crate::gpu::S2vStepper for S2vDev {
    fn step(&mut self, step: usize, chunk: usize, x: &[f32], memory: &[Vec<f32>; 3]) -> Result<Vec<f32>, String> {
        let _ = (&self.keep_raw, &self.a0);
        self.step_inner(step, chunk, x, memory)
    }
}
