//! HunyuanVideo VAE decoder on wgpu, resident: one latent upload and one RGB
//! readback a tile. The Z-Image VAE recipe (`zimage.rs`) moved to 3-D:
//! - every 3×3×3 causal conv is an implicit GEMM on `zi_mm` (`MmCfg::conv = 3`):
//!   the A tile is gathered from the `[F·H·W][C]` f16 volume with the causal
//!   replicate padding and the nearest upsample folded into the gather;
//! - the whole stream is f16: at 512² the last stages hold 4.4 M positions of
//!   256 channels, which at f32 would not fit a 4 GB binding. The reference
//!   decodes under bf16 autocast, so f16 is the closer of the two anyway;
//! - GroupNorm is the two-pass reduction, over a whole tile as in the class;
//! - the mid attention runs one query frame at a time over the keys of that
//!   frame and the frames before it;
//! - a tensor past the storage binding limit (2 GB here, and 4.4 M × 256 × 2 B
//!   is 2.2 GB) runs in slabs of frames: every kernel is row-local or, for the
//!   convs, reads two frames back, so the buffers only need windowed bindings.

use super::Ctx;
use super::zimage::{Epi, MmArgs, MmCall, MmCfg, VAE_SOFTMAX_SRC, bg, default_cfg, mm_call, mm_pipe, mm_uniform, pipeline, sbuf, sbuf_init, ubuf, zctx};
use crate::ltxvae::Vol;
use crate::s2v_vae::{Conv, GroupNorm, Mid, Res, S2vVae};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const GN_PART_SRC: &str = r#"
enable f16;
struct VP { m: u32, c: u32, cpg: u32, rows: u32, nchunk: u32, flags: u32, g: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> stat: array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<uniform> p: VP;
var<workgroup> sh: array<f32, 512>;
// One workgroup per chunk of `rows` positions. flags: 1 = add bias on read,
// 4 = pass 2 (Σ (v − mean)²).
@compute @workgroup_size(256)
fn s2v_gn_part(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let chunk = wid.x + p._b;
  let r0 = wid.x * p.rows;
  let r1 = min(r0 + p.rows, p.m);
  let n = (r1 - r0) * p.c;
  let base = r0 * p.c;
  let hb = (p.flags & 1u) != 0u;
  let pass2 = (p.flags & 4u) != 0u;
  var a0 = 0.0;
  var a1 = 0.0;
  for (var e = t; e < n; e = e + 256u) {
    let ch = e % p.c;
    var v = f32(x[base + e]);
    if (hb) { v = v + bias[ch]; }
    if (pass2) { let d = v - stat[ch / p.cpg]; v = d * d; }
    if (p.c <= 256u || ch == t) { a0 = a0 + v; } else { a1 = a1 + v; }
  }
  sh[t] = a0;
  sh[256u + t] = a1;
  workgroupBarrier();
  if (t < p.g) {
    var s = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
      var c0 = i;
      if (p.c <= 256u) { c0 = i % p.c; }
      if (c0 / p.cpg == t) { s = s + sh[i]; }
      if (p.c > 256u && (i + 256u) / p.cpg == t) { s = s + sh[256u + i]; }
    }
    part[t * p.nchunk + chunk] = s;
  }
}
"#;

const GN_FIN_SRC: &str = r#"
struct VP { m: u32, c: u32, cpg: u32, rows: u32, nchunk: u32, flags: u32, g: u32, eps: f32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> stat: array<f32>;
@group(0) @binding(2) var<uniform> p: VP;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn s2v_gn_fin(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let g = wid.x;
  var s = 0.0;
  for (var i = t; i < p.nchunk; i = i + 256u) { s = s + part[g * p.nchunk + i]; }
  sh[t] = s;
  workgroupBarrier();
  var st = 128u;
  loop {
    if (st == 0u) { break; }
    if (t < st) { sh[t] = sh[t] + sh[t + st]; }
    workgroupBarrier();
    st = st >> 1u;
  }
  if (t == 0u) {
    let n = f32(p.m) * f32(p.cpg);
    if ((p.flags & 4u) != 0u) {
      stat[p.g + g] = inverseSqrt(sh[0] / n + p.eps);
    } else {
      stat[g] = sh[0] / n;
    }
  }
}
"#;

const GN_APPLY_SRC: &str = r#"
enable f16;
struct VP { m: u32, c: u32, cpg: u32, rows: u32, nchunk: u32, flags: u32, g: u32, mp: u32 };
@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> stat: array<f32>;
@group(0) @binding(3) var<storage, read> gw: array<f32>;
@group(0) @binding(4) var<storage, read> gb: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<vec2<u32>>;
@group(0) @binding(6) var<uniform> p: VP;
// y = ((x + bias) − mean)·rstd·w + b, then SiLU (flags 2), f16.
@compute @workgroup_size(256)
fn s2v_gn_apply(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c4 = p.c / 4u;
  let idx = gid.y * (nwg.x * 256u) + gid.x;
  if (idx >= p.mp * c4) { return; }
  let row = idx / c4;
  if (row >= p.m) { outp[idx] = vec2<u32>(0u, 0u); return; }
  let ch = (idx % c4) * 4u;
  let v = vec4<f32>(x[idx]);
  var o = vec4<f32>(0.0);
  for (var j = 0u; j < 4u; j = j + 1u) {
    var a = v[j];
    if ((p.flags & 1u) != 0u) { a = a + bias[ch + j]; }
    let g = (ch + j) / p.cpg;
    var y = (a - stat[g]) * stat[p.g + g] * gw[ch + j] + gb[ch + j];
    if ((p.flags & 2u) != 0u) { y = y / (1.0 + exp(-y)); }
    o[j] = y;
  }
  outp[idx] = vec2<u32>(pack2x16float(o.xy), pack2x16float(o.zw));
}
"#;

const COMBINE_SRC: &str = r#"
enable f16;
struct CP { n: u32, c: u32, mode: u32, _a: u32 };
@group(0) @binding(0) var<storage, read_write> xo: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> h: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> hb: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> sb: array<f32>;
@group(0) @binding(5) var<uniform> p: CP;
// mode 0: x += h + hb;  mode 1: x = s + sb + h + hb;  mode 2: x = h + hb.
@compute @workgroup_size(256)
fn s2v_combine(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * 256u) + gid.x;
  if (i >= p.n) { return; }
  let ch = (i % (p.c / 4u)) * 4u;
  let hv = vec4<f32>(h[i]) + vec4<f32>(hb[ch], hb[ch + 1u], hb[ch + 2u], hb[ch + 3u]);
  var o = hv;
  if (p.mode == 0u) {
    let a = unpack2x16float(xo[i].x);
    let b = unpack2x16float(xo[i].y);
    o = o + vec4<f32>(a.x, a.y, b.x, b.y);
  } else if (p.mode == 1u) {
    o = o + vec4<f32>(s[i]) + vec4<f32>(sb[ch], sb[ch + 1u], sb[ch + 2u], sb[ch + 3u]);
  }
  xo[i] = vec2<u32>(pack2x16float(o.xy), pack2x16float(o.zw));
}
"#;

const CAST_SRC: &str = r#"
struct KP { m: u32, c: u32, hasb: u32, _p: u32 };
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> p: KP;
@compute @workgroup_size(256)
fn s2v_cast(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c4 = p.c / 4u;
  let idx = gid.y * (nwg.x * 256u) + gid.x;
  if (idx >= p.m * c4) { return; }
  var v = x[idx];
  if (p.hasb != 0u) {
    let ch = (idx % c4) * 4u;
    v = v + vec4<f32>(b[ch], b[ch + 1u], b[ch + 2u], b[ch + 3u]);
  }
  outp[idx] = vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}
"#;

const OUT_SRC: &str = r#"
enable f16;
struct OP { m: u32, ld: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> y: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform> p: OP;
@compute @workgroup_size(256)
fn s2v_vout(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * 256u) + gid.x;
  if (i >= p.m) { return; }
  for (var ch = 0u; ch < 3u; ch = ch + 1u) {
    outp[ch * p.m + i] = f32(y[i * p.ld + ch]) + b[ch];
  }
}
"#;

struct VC {
    plane: wgpu::Buffer,
    bias: wgpu::Buffer,
    cin_p: usize,
    cout: usize,
    cout_p: usize,
    k3: bool,
}

struct VN {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
}

struct VR {
    n1: VN,
    c1: VC,
    n2: VN,
    c2: VC,
    sc: Option<VC>,
}

struct VMid {
    r0: VR,
    gn: VN,
    q: VC,
    k: VC,
    vw: wgpu::Buffer,
    vb: wgpu::Buffer,
    o: VC,
    r1: VR,
    c: usize,
}

struct VDev {
    conv_in: VC,
    mid: VMid,
    ups: Vec<(Vec<VR>, Option<(VC, [usize; 3])>)>,
    norm: VN,
    conv_out: VC,
}

fn vc(c: &Ctx, cv: &Conv) -> VC {
    let taps = cv.k[0] * cv.k[1] * cv.k[2];
    let cin_p = cv.cin.next_multiple_of(32);
    let cout_p = cv.cout.next_multiple_of(128);
    let kd = taps * cin_p;
    let mut plane = vec![0u16; cout_p * kd];
    for o in 0..cv.cout {
        for ci in 0..cv.cin {
            for tap in 0..taps {
                plane[o * kd + tap * cin_p + ci] = cortiq_core::quant::f32_to_f16(cv.w[(o * cv.cin + ci) * taps + tap]);
            }
        }
    }
    let mut bias = vec![0f32; cout_p];
    bias[..cv.cout].copy_from_slice(&cv.b);
    VC {
        plane: sbuf_init(c, bytemuck::cast_slice(&plane), "s2v_vp"),
        bias: sbuf_init(c, bytemuck::cast_slice(&bias), "s2v_vb"),
        cin_p,
        cout: cv.cout,
        cout_p,
        k3: taps == 27,
    }
}

fn vl(c: &Ctx, l: &(Vec<f32>, Vec<f32>)) -> VC {
    let cc = l.1.len();
    let mut plane = vec![0u16; cc * cc];
    for (d, &s) in plane.iter_mut().zip(&l.0) {
        *d = cortiq_core::quant::f32_to_f16(s);
    }
    VC {
        plane: sbuf_init(c, bytemuck::cast_slice(&plane), "s2v_vl"),
        bias: sbuf_init(c, bytemuck::cast_slice(&l.1), "s2v_vlb"),
        cin_p: cc,
        cout: cc,
        cout_p: cc,
        k3: false,
    }
}

fn vn(c: &Ctx, g: &GroupNorm) -> VN {
    VN { w: sbuf_init(c, bytemuck::cast_slice(&g.w), "s2v_gw"), b: sbuf_init(c, bytemuck::cast_slice(&g.b), "s2v_gb") }
}

fn vr(c: &Ctx, r: &Res) -> VR {
    VR { n1: vn(c, &r.n1), c1: vc(c, &r.c1), n2: vn(c, &r.n2), c2: vc(c, &r.c2), sc: r.short.as_ref().map(|s| vc(c, s)) }
}

fn vmid(c: &Ctx, m: &Mid) -> VMid {
    let cc = m.q.1.len();
    let mut vw = vec![0u16; cc * cc];
    for (d, &s) in vw.iter_mut().zip(&m.v.0) {
        *d = cortiq_core::quant::f32_to_f16(s);
    }
    VMid {
        r0: vr(c, &m.r0),
        gn: vn(c, &m.gn),
        q: vl(c, &m.q),
        k: vl(c, &m.k),
        vw: sbuf_init(c, bytemuck::cast_slice(&vw), "s2v_vw"),
        vb: sbuf_init(c, bytemuck::cast_slice(&m.v.1), "s2v_vvb"),
        o: vl(c, &m.o),
        r1: vr(c, &m.r1),
        c: cc,
    }
}

struct VS {
    dev: VDev,
    progs: HashMap<(usize, usize, usize), Arc<Prog>>,
}
// SAFETY: the context is one device and the program is only used under the mutex.
unsafe impl Send for VS {}

static STATE: Mutex<Option<VS>> = Mutex::new(None);

pub(crate) fn release() {
    if let Ok(mut g) = STATE.lock() {
        *g = None;
    }
}

enum Op {
    D(Arc<wgpu::ComputePipeline>, wgpu::BindGroup, (u32, u32, u32)),
    M(MmCall),
}

struct Prog {
    ops: Vec<Op>,
    z: wgpu::Buffer,
    out: wgpu::Buffer,
    stage: wgpu::Buffer,
    bytes: u64,
    f_out: usize,
    h_out: usize,
    w_out: usize,
    mp0: usize,
    cin0: usize,
    _keep: Vec<wgpu::Buffer>,
}

fn grid1(n: usize) -> (u32, u32, u32) {
    let wgs = (n as u32).div_ceil(256).max(1);
    (wgs.min(65535), wgs.div_ceil(65535), 1)
}

struct Rec<'a> {
    c: &'a Ctx,
    ops: Vec<Op>,
    dummy: wgpu::Buffer,
    stat: wgpu::Buffer,
    part: wgpu::Buffer,
    x: wgpu::Buffer,
    h: wgpu::Buffer,
    s: wgpu::Buffer,
    xn: wgpu::Buffer,
}

const GN_ROWS: usize = 1024;

fn mp_of(m: usize) -> usize {
    m.next_multiple_of(128)
}

impl Rec<'_> {
    fn k(&mut self, key: &str, src: &str, entry: &str, bufs: &[&wgpu::Buffer], grid: (u32, u32, u32)) -> Option<()> {
        let pipe = pipeline(self.c, key, src, entry)?;
        let b = bg(self.c, &pipe, bufs);
        self.ops.push(Op::D(pipe, b, grid));
        Some(())
    }


    fn limit(&self) -> usize {
        let l = self.c.device.limits();
        (l.max_storage_buffer_binding_size as u64).min(l.max_buffer_size) as usize
    }

    /// `(first frame, frames)` slabs of a `[F·H·W][ch]` f16 tensor that each
    /// fit one binding.
    fn slabs(&self, f: usize, per: usize, ch: usize) -> Vec<(usize, usize)> {
        let frame = per * ch * 2;
        let room = (self.limit() / frame).max(1).min(f);
        (0..f.div_ceil(room)).map(|i| (i * room, room.min(f - i * room))).collect()
    }

    fn kw(&mut self, key: &str, src: &str, entry: &str, bufs: &[(&wgpu::Buffer, u64, u64)], grid: (u32, u32, u32)) -> Option<()> {
        let pipe = pipeline(self.c, key, src, entry)?;
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                if b.2 == 0 {
                    super::bind_buf(i as u32, b.0)
                } else {
                    super::bind_buf_off(i as u32, b.0, b.1, b.2)
                }
            })
            .collect();
        let bgr = self.c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        self.ops.push(Op::D(pipe, bgr, grid));
        Some(())
    }

    /// GroupNorm of `x` `[F·H·W][ch]` → f16 `out`, optional bias on read and SiLU.
    #[allow(clippy::too_many_arguments)]
    fn gn(&mut self, x: &wgpu::Buffer, bias: Option<&wgpu::Buffer>, n: &VN, f: usize, per: usize, ch: usize, silu: bool, out: &wgpu::Buffer) -> Option<()> {
        let g = 32usize;
        let cpg = ch / g;
        let m = f * per;
        let hb = bias.is_some() as u32;
        let d = self.dummy.clone();
        let bb = bias.unwrap_or(&d);
        let (stat, part) = (self.stat.clone(), self.part.clone());
        let sl = self.slabs(f, per, ch);
        let nchunk_all: usize = sl.iter().map(|s| (s.1 * per).div_ceil(GN_ROWS)).sum();
        for pass2 in [false, true] {
            let flags = hb | if pass2 { 4 } else { 0 };
            let mut c0 = 0usize;
            for &(f0, nf) in &sl {
                let rows = nf * per;
                let nc = rows.div_ceil(GN_ROWS);
                let u = ubuf(self.c, &[rows as u32, ch as u32, cpg as u32, GN_ROWS as u32, nchunk_all as u32, flags, g as u32, c0 as u32]);
                let off = (f0 * per * ch * 2) as u64;
                let size = (rows * ch * 2) as u64;
                self.kw("s2v_gn_part", GN_PART_SRC, "s2v_gn_part", &[(x, off, size), (bb, 0, 0), (&stat, 0, 0), (&part, 0, 0), (&u, 0, 0)], (nc as u32, 1, 1))?;
                c0 += nc;
            }
            let u = ubuf(self.c, &[m as u32, ch as u32, cpg as u32, GN_ROWS as u32, nchunk_all as u32, flags, g as u32, 1e-6f32.to_bits()]);
            self.k("s2v_gn_fin", GN_FIN_SRC, "s2v_gn_fin", &[&part, &stat, &u], (g as u32, 1, 1))?;
        }
        let flags = hb | if silu { 2 } else { 0 };
        for &(f0, nf) in &sl {
            let rows = nf * per;
            let u = ubuf(self.c, &[rows as u32, ch as u32, cpg as u32, 0, 0, flags, g as u32, rows as u32]);
            let off = (f0 * per * ch * 2) as u64;
            let size = (rows * ch * 2) as u64;
            self.kw(
                "s2v_gn_apply",
                GN_APPLY_SRC,
                "s2v_gn_apply",
                &[(x, off, size), (bb, 0, 0), (&stat, 0, 0), (&n.w, 0, 0), (&n.b, 0, 0), (out, off, size), (&u, 0, 0)],
                grid1(rows * ch / 4),
            )?;
        }
        Some(())
    }

    /// conv of the f16 volume `act` → raw f16 `[m][cout_p]` (no bias). `act`
    /// holds the source frames (before the nearest upsample `up`).
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, cv: &VC, act: &wgpu::Buffer, f: usize, h: usize, w: usize, up: (bool, bool), out: &wgpu::Buffer) -> Option<()> {
        let per = h * w;
        let (sh, sw) = if up.0 { (h / 2, w / 2) } else { (h, w) };
        let src_f = |of: usize| if up.1 && of > 0 { (of - 1) / 2 + 1 } else { of };
        let sper = sh * sw * cv.cin_p * 2;
        let taps = if cv.k3 { 27 } else { 1 };
        let g = MmCfg { conv: if cv.k3 { 3 } else { 0 }, ..default_cfg(Epi::F16) };
        let sl = self.slabs(f, per, cv.cout_p.max(if up.0 { 0 } else { cv.cin_p }));
        for (f0, nf) in sl {
            let rows = nf * per;
            let s0 = if cv.k3 { src_f(f0.saturating_sub(2)) } else { f0 };
            let s1 = src_f(f0 + nf - 1) + 1;
            let a = MmArgs {
                m: rows as u32,
                n: cv.cout_p as u32,
                k: (taps * cv.cin_p) as u32,
                ldo: cv.cout_p as u32,
                ocol: 0,
                arow: if cv.k3 {
                    (f as u32) | ((f0 as u32) << 8) | ((s0 as u32) << 16) | (u32::from(up.0) << 24) | (u32::from(up.1) << 25)
                } else {
                    0
                },
                oscale: 1.0,
                conv: [w as u32, h as u32, cv.cin_p as u32],
            };
            // The tile over-reads rows past the slab: keep them in the window.
            let oo = (f0 * per * cv.cout_p * 2) as u64;
            let cap = out.size() - oo;
            let osize = ((rows.next_multiple_of(128) * cv.cout_p * 2) as u64).min(cap);
            let ao = (s0 * sper) as u64;
            let acap = act.size() - ao;
            let asize = (((s1 - s0) * sper) as u64).next_multiple_of(16).min(acap).max(16);
            let mc = mm_win(self.c, g, &a, &cv.plane, act, ao, asize, out, oo, osize)?;
            self.ops.push(Op::M(mc));
        }
        Some(())
    }

    #[allow(clippy::too_many_arguments)]
    fn combine(&mut self, x: &wgpu::Buffer, h: &wgpu::Buffer, hb: &wgpu::Buffer, s: Option<(&wgpu::Buffer, &wgpu::Buffer)>, mode: u32, f: usize, per: usize, ch: usize) -> Option<()> {
        let d = self.dummy.clone();
        let (sb, sbb) = s.unwrap_or((&d, &d));
        for (f0, nf) in self.slabs(f, per, ch) {
            let rows = nf * per;
            let c4n = rows * ch / 4;
            let u = ubuf(self.c, &[c4n as u32, ch as u32, mode, 0]);
            let off = (f0 * per * ch * 2) as u64;
            let size = (rows * ch * 2) as u64;
            let sw = if s.is_some() { (sb, off, size) } else { (sb, 0, 0) };
            self.kw(
                "s2v_combine",
                COMBINE_SRC,
                "s2v_combine",
                &[(x, off, size), (h, off, size), (hb, 0, 0), sw, (sbb, 0, 0), (&u, 0, 0)],
                grid1(c4n),
            )?;
        }
        Some(())
    }

    fn resnet(&mut self, r: &VR, f: usize, h: usize, w: usize, ic: usize) -> Option<usize> {
        let per = h * w;
        let oc = r.c1.cout;
        let (x, hb, s, xn) = (self.x.clone(), self.h.clone(), self.s.clone(), self.xn.clone());
        self.gn(&x, None, &r.n1, f, per, ic, true, &xn)?;
        self.conv(&r.c1, &xn, f, h, w, (false, false), &hb)?;
        self.gn(&hb, Some(&r.c1.bias), &r.n2, f, per, oc, true, &xn)?;
        self.conv(&r.c2, &xn, f, h, w, (false, false), &hb)?;
        match &r.sc {
            Some(sc) => {
                self.conv(sc, &x, f, h, w, (false, false), &s)?;
                self.combine(&x, &hb, &r.c2.bias, Some((&s, &sc.bias)), 1, f, per, oc)?;
            }
            None => {
                self.combine(&x, &hb, &r.c2.bias, None, 0, f, per, oc)?;
            }
        }
        Some(oc)
    }
}

#[allow(clippy::too_many_arguments)]
fn mm_win(c: &Ctx, g: MmCfg, a: &MmArgs, plane: &wgpu::Buffer, act: &wgpu::Buffer, ao: u64, asize: u64, out: &wgpu::Buffer, oo: u64, osize: u64) -> Option<MmCall> {
    if a.n % g.bn != 0 || a.k % g.bk != 0 {
        return None;
    }
    let pipe = mm_pipe(c, g)?;
    let u = mm_uniform(c, a);
    let bgr = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("s2v_conv"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            super::bind_buf(0, plane),
            super::bind_buf_off(1, act, ao, asize),
            super::bind_buf_off(2, out, oo, osize),
            super::bind_buf(3, &u),
        ],
    });
    Some(MmCall { pipe, bg: bgr, grid: (a.n / g.bn, a.m.div_ceil(g.bm)), mnk: [a.m, a.n, a.k], bits: false })
}

fn mm_off(c: &Ctx, g: MmCfg, a: &MmArgs, plane: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer, off: u64, size: u64) -> Option<MmCall> {
    if a.n % g.bn != 0 || a.k % g.bk != 0 {
        return None;
    }
    let pipe = mm_pipe(c, g)?;
    let u = mm_uniform(c, a);
    let bgr = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("s2v_pv"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            super::bind_buf(0, plane),
            super::bind_buf(1, act),
            super::bind_buf_off(2, out, off, size),
            super::bind_buf(3, &u),
        ],
    });
    Some(MmCall { pipe, bg: bgr, grid: (a.n / g.bn, a.m.div_ceil(g.bm)), mnk: [a.m, a.n, a.k], bits: false })
}

fn build_prog(c: &Ctx, d: &VDev, f0: usize, h0: usize, w0: usize) -> Option<Prog> {
    // Shapes walk: size every buffer for the largest.
    let mut elems = mp_of(f0 * h0 * w0) * 512;
    let (mut ff, mut hh, mut ww) = (f0, h0, w0);
    for (res, up) in &d.ups {
        for r in res {
            elems = elems.max(mp_of(ff * hh * ww) * r.c1.cin_p.max(r.c1.cout_p));
        }
        if let Some((u, fac)) = up {
            ff = if fac[0] == 2 { 1 + (ff - 1) * 2 } else { ff };
            hh *= fac[1];
            ww *= fac[2];
            elems = elems.max(mp_of(ff * hh * ww) * u.cin_p.max(u.cout_p));
        }
    }
    let nchunk_max = (elems / 128).div_ceil(GN_ROWS) + 1;
    let mut r = Rec {
        c,
        ops: Vec::new(),
        dummy: sbuf(c, 64, "s2v_d"),
        stat: sbuf(c, 1024 * 4, "s2v_stat"),
        part: sbuf(c, (32 * nchunk_max * 4) as u64, "s2v_part"),
        x: sbuf(c, ((elems + 128 * 512) * 2) as u64, "s2v_vx"),
        h: sbuf(c, ((elems + 128 * 512) * 2) as u64, "s2v_vh"),
        s: sbuf(c, ((elems + 128 * 512) * 2) as u64, "s2v_vs"),
        xn: sbuf(c, ((elems + 128 * 512) * 2) as u64, "s2v_vxn"),
    };
    let (x, hb, xn) = (r.x.clone(), r.h.clone(), r.xn.clone());
    let cin0 = d.conv_in.cin_p;
    let m0 = f0 * h0 * w0;
    let mp0 = mp_of(m0);
    let z = sbuf(c, ((mp0 + 128) * cin0 * 2) as u64, "s2v_vz");
    r.conv(&d.conv_in, &z, f0, h0, w0, (false, false), &hb)?;
    let mut ch = d.conv_in.cout_p;
    r.combine(&x, &hb, &d.conv_in.bias, None, 2, f0, h0 * w0, ch)?;
    ch = r.resnet(&d.mid.r0, f0, h0, w0, ch)?;

    // Mid attention, one query frame at a time.
    let mut keep = Vec::new();
    {
        let at = &d.mid;
        let cc = at.c;
        let m = m0;
        let mp = mp0;
        let pf = h0 * w0;
        // Small buffers of their own: `xn` and `h` are sized for the last stages
        // and can be past a binding as a whole.
        let xa = sbuf(c, ((mp + 128) * cc * 2) as u64, "s2v_xa");
        let ha = sbuf(c, ((mp + 128) * cc * 2) as u64, "s2v_ha");
        r.gn(&x, None, &at.gn, f0, pf, cc, false, &xa)?;
        let q16 = sbuf(c, (mp * cc * 2) as u64, "s2v_q");
        let k16 = sbuf(c, (mp * cc * 2) as u64, "s2v_k");
        let o32 = sbuf(c, (mp * cc * 4) as u64, "s2v_o");
        let vt16 = sbuf(c, (cc * mp * 2) as u64, "s2v_vt");
        let q32 = sbuf(c, (mp * cc * 4) as u64, "s2v_q32");
        let cast = pipeline(c, "s2v_vcast", CAST_SRC, "s2v_cast")?;
        // q and k as plain GEMMs with their bias, f16.
        for (cv, dst) in [(&at.q, &q16), (&at.k, &k16)] {
            let a = MmArgs { m: m as u32, n: cc as u32, k: cc as u32, ldo: cc as u32, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
            r.ops.push(Op::M(mm_call(c, default_cfg(Epi::F32), &a, &cv.plane, &xa, &q32)?));
            let u = ubuf(c, &[m as u32, cc as u32, 1, 0]);
            let b = bg(c, &cast, &[&q32, &cv.bias, dst, &u]);
            r.ops.push(Op::D(cast.clone(), b, grid1(m * cc / 4)));
        }
        let scale = 1.0 / (cc as f32).sqrt();
        let sc = sbuf(c, (pf * m * 4) as u64, "s2v_sc");
        let pr = sbuf(c, (pf * m * 2) as u64, "s2v_pr");
        for fr in 0..f0 {
            let live = (fr + 1) * pf;
            let a = MmArgs { m: pf as u32, n: live as u32, k: cc as u32, ldo: live as u32, ocol: 0, arow: (fr * pf) as u32, oscale: scale, conv: [0; 3] };
            r.ops.push(Op::M(mm_call(c, default_cfg(Epi::F32), &a, &k16, &q16, &sc)?));
            let u = ubuf(c, &[live as u32, live as u32, 0, 0]);
            let pipe = pipeline(c, "s2v_softmax", VAE_SOFTMAX_SRC, "vae_softmax")?;
            let b = bg(c, &pipe, &[&sc, &pr, &u]);
            r.ops.push(Op::D(pipe, b, ((pf as u32).min(65535), (pf as u32).div_ceil(65535), 1)));
            // Vᵀ of the live keys: Wv·xnᵀ.
            let a = MmArgs { m: cc as u32, n: live as u32, k: cc as u32, ldo: live as u32, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
            r.ops.push(Op::M(mm_call(c, default_cfg(Epi::F16), &a, &xa, &at.vw, &vt16)?));
            let a = MmArgs { m: pf as u32, n: cc as u32, k: live as u32, ldo: cc as u32, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
            let off = (fr * pf * cc * 4) as u64;
            r.ops.push(Op::M(mm_off(c, default_cfg(Epi::F32), &a, &vt16, &pr, &o32, off, (pf * cc * 4) as u64)?));
        }
        let u = ubuf(c, &[mp as u32, cc as u32, 1, 0]);
        let b = bg(c, &cast, &[&o32, &at.vb, &xa, &u]);
        r.ops.push(Op::D(cast.clone(), b, grid1(mp * cc / 4)));
        let a = MmArgs { m: m as u32, n: cc as u32, k: cc as u32, ldo: cc as u32, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
        r.ops.push(Op::M(mm_call(c, default_cfg(Epi::F16), &a, &at.o.plane, &xa, &ha)?));
        r.combine(&x, &ha, &at.o.bias, None, 0, f0, pf, cc)?;
        keep.extend([q16, k16, o32, vt16, q32, sc, pr, xa, ha]);
    }
    ch = r.resnet(&d.mid.r1, f0, h0, w0, ch)?;
    let (mut ff, mut hh, mut ww) = (f0, h0, w0);
    for (res, up) in &d.ups {
        for rs in res {
            ch = r.resnet(rs, ff, hh, ww, ch)?;
        }
        if let Some((u, fac)) = up {
            ff = if fac[0] == 2 { 1 + (ff - 1) * 2 } else { ff };
            hh *= fac[1];
            ww *= fac[2];
            r.conv(u, &x, ff, hh, ww, (fac[1] == 2, fac[0] == 2), &hb)?;
            ch = u.cout_p;
            r.combine(&x, &hb, &u.bias, None, 2, ff, hh * ww, ch)?;
        }
    }
    let m = ff * hh * ww;
    r.gn(&x, None, &d.norm, ff, hh * ww, ch, true, &xn)?;
    r.conv(&d.conv_out, &xn, ff, hh, ww, (false, false), &hb)?;
    let out = sbuf(c, (3 * m * 4) as u64, "s2v_vout");
    let u = ubuf(c, &[m as u32, d.conv_out.cout_p as u32, 0, 0]);
    let hsize = (m * d.conv_out.cout_p * 2) as u64;
    r.kw("s2v_vout", OUT_SRC, "s2v_vout", &[(&hb, 0, hsize), (&d.conv_out.bias, 0, 0), (&out, 0, 0), (&u, 0, 0)], grid1(m))?;
    let bytes = (3 * m * 4) as u64;
    let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("s2v_vstage"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    keep.extend([r.x.clone(), r.h.clone(), r.s.clone(), r.xn.clone(), r.part.clone(), r.stat.clone(), r.dummy.clone()]);
    Some(Prog { ops: r.ops, z, out, stage, bytes, f_out: ff, h_out: hh, w_out: ww, mp0, cin0, _keep: keep })
}

/// `z` is one temporal tile `[16, F, H, W]`, past the post-quant conv.
fn decode_tile(c: &Ctx, p: &Prog, z: &Vol) -> Option<Vol> {
    let n = z.positions();
    let mut zin = vec![0u16; p.mp0 * p.cin0];
    for ch in 0..z.c {
        for i in 0..n {
            zin[i * p.cin0 + ch] = cortiq_core::quant::f32_to_f16(z.data[ch * n + i]);
        }
    }
    c.queue.write_buffer(&p.z, 0, bytemuck::cast_slice(&zin));
    // One submission per block of work keeps each under the driver's patience.
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for op in &p.ops {
            match op {
                Op::D(pipe, b, g) => {
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, b, &[]);
                    pass.dispatch_workgroups(g.0, g.1, g.2);
                }
                Op::M(m) => m.record(&mut pass),
            }
        }
    }
    enc.copy_buffer_to_buffer(&p.out, 0, &p.stage, 0, p.bytes);
    c.queue.submit(Some(enc.finish()));
    let slice = p.stage.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    let mapped = slice.get_mapped_range().ok()?;
    let mut v = Vol::zeros(3, p.f_out, p.h_out, p.w_out);
    v.data.copy_from_slice(bytemuck::cast_slice(&mapped));
    drop(mapped);
    p.stage.unmap();
    Some(v)
}

/// The whole tiled decode, or `None` to run on the CPU.
pub(crate) fn decode(vae: &S2vVae, z: &Vol) -> Option<Vol> {
    if std::env::var("CMF_S2V_VAE_WGPU").as_deref() == Ok("0") {
        return None;
    }
    let c = zctx()?;
    let t0 = Instant::now();
    let mut g = STATE.lock().ok()?;
    if g.is_none() {
        let mut ups = Vec::new();
        for u in &vae.ups {
            let res = u.res.iter().map(|r| vr(c, r)).collect();
            let conv = u.conv.as_ref().map(|(cv, f)| (vc(c, cv), *f));
            ups.push((res, conv));
        }
        let dev = VDev {
            conv_in: vc(c, &vae.conv_in),
            mid: vmid(c, &vae.mid),
            ups,
            norm: vn(c, &vae.norm_out),
            conv_out: vc(c, &vae.conv_out),
        };
        *g = Some(VS { dev, progs: HashMap::new() });
        crate::s2vgen::prof("vae planes", t0);
    }
    let st = g.as_mut()?;
    let mut failed = false;
    let v = crate::s2v_vae::tiled(z, &mut |tile| {
        let key = (tile.f, tile.h, tile.w);
        if !st.progs.contains_key(&key) {
            match build_prog(c, &st.dev, tile.f, tile.h, tile.w) {
                Some(p) => {
                    st.progs.insert(key, Arc::new(p));
                }
                None => {
                    failed = true;
                    return Vol::zeros(3, 1, tile.h * 8, tile.w * 8);
                }
            }
        }
        let pq = vae.post_quant_host(tile);
        match decode_tile(c, &st.progs[&key], &pq) {
            Some(d) => d,
            None => {
                failed = true;
                Vol::zeros(3, 1, tile.h * 8, tile.w * 8)
            }
        }
    });
    crate::s2vgen::prof("vae device decode", t0);
    if failed {
        return None;
    }
    Some(v)
}
