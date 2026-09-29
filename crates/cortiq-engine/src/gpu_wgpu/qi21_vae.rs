//! Qwen-Image-2.1 VAE decoder on wgpu, resident (the `gpu::qi21_vae_decode`
//! contract): the whole decoder on the device — one latent upload, one
//! submission per band, one `[4][H·W]` readback — instead of
//! `gpu::vae_conv2d` per conv with the activations crossing the bus and
//! the RMS norms on the host.
//!
//! The Z-Image resident VAE's recipe (`zimage.rs`, "Resident Flux-VAE
//! decoder"): NHWC activations, every 3×3 conv an implicit GEMM on `zi_mm`
//! (`MmCfg::conv`; the nearest-2× upsample folded into the A-tile gather),
//! 1×1 convs as plain GEMMs, f16 conv inputs with f32 accumulation and an
//! f32 residual stream, the mid attention as QKᵀ → softmax → P·V over
//! query chunks. Its combine and softmax kernels are reused as they are.
//! What differs here:
//! - `qv_rmsn`: the Wan `RMS_norm` (x/max(‖x‖₂, 1e-12)·√C·γ per pixel over
//!   the channels) with the conv bias added on read and an optional SiLU,
//!   → the next conv's f16 input;
//! - `qv_dupup`: the `DupUp3D` shortcut of an up block (a channel shuffle
//!   onto the 2× grid) added into the upsampled stream;
//! - channels are padded to multiples of 64 (1152, 576, 288 → 320,
//!   144 → 192, 4 → 64: zero weights, zero γ), and a GEMM whose padded
//!   output width is not a multiple of 128 uses a 128×64 tile;
//! - the two f16 casts of the raw residual stream carry a range guard
//!   (see `guard`);
//! - large frames run in horizontal bands (see "bands" below): the leading
//!   up blocks run on the whole frame while every tensor fits a binding,
//!   the rest band by band with exact halos, so a banded decode is bit for
//!   bit the whole-frame one (1024², forced with `CMF_QI21_VAE_BAND_ROWS`
//!   at 94–512 kept rows from blocks 0–3: max |Δ| = 0 in all five cases).
//!
//! Measured (RTX PRO 4000 Blackwell, 2 GB binding limit), decode only;
//! the weights (f16 planes, once per model) take 0.26–0.47 s more:
//!
//! | frame      | layout                                   | decode  |
//! |------------|------------------------------------------|---------|
//! | 256²       | whole                                    | 0.11 s  |
//! | 1024²      | whole (4.8 GB of activations)            | 0.49–0.61 s |
//! | 2048²      | 2 blocks whole + 3 bands of 683 rows     | 1.84 s  |
//! | 1536×2752  | 2 blocks whole + 3 bands of 512 rows     | 1.87 s  |
//! | 4096²      | 1 block whole + 11 bands of 373 rows     | 8.3 s   |
//!
//! The per-conv path took 43 s at 1024² and could not run 2048² at all on
//! the device (a 2.4 GB tensor against the 2 GB binding). u8 PSNR against
//! the per-conv path: 66.9 dB on a real 1024² latent, 61.2 dB on a random
//! one; against the fp32 diffusers decoder at 256² 66.0 dB (per-conv 69.1).
//!
//! Knobs: `CMF_QI21_VAE_CHAIN=0` (the per-conv path),
//! `CMF_QI21_VAE_BAND_ROWS=k` (force bands of ≤ k kept output rows),
//! `CMF_QI21_VAE_BAND_FROM=s` (band from up block s on; forced bands
//! default to 1), `CMF_QI21_VAE_SHIFT` (the stream guard, log2),
//! `CMF_QI21_VAE_DUMP=<file>` (the raw f32 `[4][H·W]` output).

use crate::gpu::{Qi21VaeConvRef, Qi21VaeDecodeArgs, Qi21VaeResRef};
use std::sync::Mutex;

use super::zimage::{self as zi, Call, Class, Epi, MmCall, MmCfg, ZCalls};
use super::Ctx;

fn enabled() -> bool {
    std::env::var("CMF_QI21_VAE_CHAIN").as_deref() != Ok("0")
}

fn decline(reason: &str) -> bool {
    static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
    if let Ok(mut v) = SAID.lock()
        && !v.iter().any(|r| r == reason)
    {
        eprintln!("qwen-image-2.1: resident VAE decoder declined: {reason}; the per-conv path runs");
        v.push(reason.to_string());
    }
    false
}

fn prof_on() -> bool {
    std::env::var("CMF_QI21_PROF").is_ok_and(|v| v != "0")
}

/// The range guard of the f16 casts of the raw residual stream (the 1×1
/// shortcut's and the upsample conv's inputs): stored ×2⁻ᵍ, the conv's
/// f32 epilogue multiplies 2ᵍ back. Stream maxima (host decoder): 120 on
/// a real 1024² latent, 1.5e5 (block 4) on a random N(mean, std) one —
/// past the f16 range, which turned 68 % of that image into NaN unguarded.
/// Every other f16 site is RMS-normalised. `CMF_QI21_VAE_SHIFT` (default 8).
fn guard() -> f32 {
    let g = std::env::var("CMF_QI21_VAE_SHIFT").ok().and_then(|v| v.parse::<i32>().ok()).unwrap_or(8).clamp(0, 14);
    (2.0f32).powi(g)
}

/// Channel padding of every NHWC tensor (the GEMM's N tile 64 / K slice 32).
fn cpad(c: usize) -> usize {
    c.next_multiple_of(64)
}

/// Rows of an NHWC tensor, padded to the GEMM's M tile.
fn mp_of(m: usize) -> usize {
    m.next_multiple_of(128)
}

// ───────────────────────────── kernels ─────────────────────────────

/// RMS_norm over the channels of one NHWC pixel (+ bias on read, flags 1;
/// SiLU, flags 2) → f16 `[mp][cp]`, rows ≥ m zeroed. `c` = the real channel
/// count (√C), `cp` = the padded row (padded channels are zero: weights
/// and γ are zero there).
const RMSN_SRC: &str = r#"
struct NP { m: u32, mp: u32, cp: u32, c: u32, flags: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> x: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> gam: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> outp: array<u32>;
@group(0) @binding(4) var<uniform> p: NP;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn qv_rmsn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wid.x + wid.y * 65535u;
  if (row >= p.mp) { return; }
  let np = p.cp / 2u;
  let base = row * np;
  if (row >= p.m) {
    for (var i = t; i < np; i = i + 64u) { outp[base + i] = 0u; }
    return;
  }
  let hb = (p.flags & 1u) != 0u;
  var ss = 0.0;
  for (var i = t; i < np; i = i + 64u) {
    var v = x[base + i];
    if (hb) { v = v + bias[i]; }
    ss = ss + dot(v, v);
  }
  red[t] = ss;
  workgroupBarrier();
  var st = 32u;
  loop {
    if (st == 0u) { break; }
    if (t < st) { red[t] = red[t] + red[t + st]; }
    workgroupBarrier();
    st = st >> 1u;
  }
  let inv = sqrt(f32(p.c)) / max(sqrt(red[0]), 1e-12);
  for (var i = t; i < np; i = i + 64u) {
    var v = x[base + i];
    if (hb) { v = v + bias[i]; }
    var y = v * inv * gam[i];
    if ((p.flags & 2u) != 0u) { y = y / (vec2<f32>(1.0) + exp(-y)); }
    outp[base + i] = pack2x16float(y);
  }
}
"#;

/// f32 `[m][c]` (+ bias) × `scale` → f16 `[mp][c]`, rows ≥ m zeroed.
const CAST_SRC: &str = r#"
enable f16;
struct KP { m: u32, mp: u32, c: u32, hasb: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> p: KP;
@compute @workgroup_size(256)
fn qv_cast(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c4 = p.c / 4u;
  let idx = gid.y * (nwg.x * 256u) + gid.x;
  if (idx >= p.mp * c4) { return; }
  if (idx / c4 >= p.m) { outp[idx] = vec2<u32>(0u, 0u); return; }
  var v = x[idx];
  if (p.hasb != 0u) {
    let ch = (idx % c4) * 4u;
    v = v + vec4<f32>(b[ch], b[ch + 1u], b[ch + 2u], b[ch + 3u]);
  }
  v = v * p.scale;
  outp[idx] = vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}
"#;

/// DupUp3D on the first chunk, added into the upsampled stream:
/// `x[2y+sy, 2x+sx, o] += src[y, x, (o·factor + (ft−1)·4 + 2·sy + sx) / repeats]`.
const DUPUP_SRC: &str = r#"
struct DP { h: u32, w: u32, cin_p: u32, cout: u32, cout_p: u32, factor: u32, repeats: u32, ft: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<uniform> p: DP;
@compute @workgroup_size(256)
fn qv_dupup(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * 256u) + gid.x;
  if (i >= 4u * p.h * p.w * p.cout) { return; }
  let w2 = 2u * p.w;
  let o = i % p.cout;
  let pix = i / p.cout;
  let yy = pix / w2;
  let xx = pix % w2;
  let j = o * p.factor + (p.ft - 1u) * 4u + 2u * (yy & 1u) + (xx & 1u);
  let s = ((yy >> 1u) * p.w + (xx >> 1u)) * p.cin_p + j / p.repeats;
  x[pix * p.cout_p + o] = x[pix * p.cout_p + o] + src[s];
}
"#;

/// NHWC `[rows][ld]` (first `nch` channels) + bias → NCHW `[nch][tot]`:
/// the `n` pixels from band pixel `src0` land at frame pixel `dst0`.
const OUT_SRC: &str = r#"
struct OP { n: u32, ld: u32, nch: u32, src0: u32, dst0: u32, tot: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform> p: OP;
@compute @workgroup_size(256)
fn qv_out(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * 256u) + gid.x;
  if (i >= p.n) { return; }
  for (var ch = 0u; ch < p.nch; ch = ch + 1u) {
    outp[ch * p.tot + p.dst0 + i] = y[(p.src0 + i) * p.ld + ch] + b[ch];
  }
}
"#;

// ───────────────────────────── weights ─────────────────────────────

/// One conv: an f16 plane `[cout_p][k²·cin_p]` in (tap, channel) order
/// (zero padding) and an f32 bias `[cout_p]`.
struct QConv {
    plane: wgpu::Buffer,
    bias: wgpu::Buffer,
    cin_p: usize,
    cout: usize,
    cout_p: usize,
    k: usize,
}

/// `rows` = the output channels of `r` this conv takes (the attention
/// splits `to_qkv` into three).
fn qconv(c: &Ctx, r: &Qi21VaeConvRef, rows: std::ops::Range<usize>) -> Option<QConv> {
    let (ic, k) = (r.ci, r.k);
    if (k != 1 && k != 3) || r.w.len() != r.co * ic * k * k || r.b.len() != r.co || rows.end > r.co {
        return None;
    }
    let oc = rows.len();
    let (cin_p, cout_p) = (cpad(ic), cpad(oc));
    let kk = k * k;
    let kd = kk * cin_p;
    let mut plane = vec![0u16; cout_p * kd];
    let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 16);
    let per = oc.div_ceil(nt).max(1);
    std::thread::scope(|sc| {
        for (ti, chunk) in plane[..oc * kd].chunks_mut(per * kd).enumerate() {
            let r0 = rows.start + ti * per;
            sc.spawn(move || {
                for (ol, dst) in chunk.chunks_mut(kd).enumerate() {
                    let o = r0 + ol;
                    for ci in 0..ic {
                        for tap in 0..kk {
                            dst[tap * cin_p + ci] = cortiq_core::quant::f32_to_f16(r.w[(o * ic + ci) * kk + tap]);
                        }
                    }
                }
            });
        }
    });
    let mut bias = vec![0f32; cout_p];
    bias[..oc].copy_from_slice(&r.b[rows]);
    Some(QConv {
        plane: zi::sbuf_init(c, bytemuck::cast_slice(&plane), "qv_plane"),
        bias: zi::sbuf_init(c, bytemuck::cast_slice(&bias), "qv_bias"),
        cin_p,
        cout: oc,
        cout_p,
        k,
    })
}

/// γ padded with zeros to the padded channel count.
fn gamma(c: &Ctx, g: &[f32]) -> wgpu::Buffer {
    let mut v = g.to_vec();
    v.resize(cpad(g.len()), 0.0);
    zi::sbuf_init(c, bytemuck::cast_slice(&v), "qv_gamma")
}

struct QRes {
    g1: wgpu::Buffer,
    c1: QConv,
    g2: wgpu::Buffer,
    c2: QConv,
    sc: Option<QConv>,
    cin: usize,
}

fn qres(c: &Ctx, r: &Qi21VaeResRef) -> Option<QRes> {
    let cin = r.c1.ci;
    if r.g1.len() != cin || r.g2.len() != r.c1.co || r.c2.ci != r.c1.co || r.c2.co != r.c1.co {
        return None;
    }
    if r.shortcut.is_none() && cin != r.c2.co {
        return None;
    }
    Some(QRes {
        g1: gamma(c, r.g1),
        c1: qconv(c, &r.c1, 0..r.c1.co)?,
        g2: gamma(c, r.g2),
        c2: qconv(c, &r.c2, 0..r.c2.co)?,
        sc: match &r.shortcut {
            Some(s) if s.k == 1 && s.ci == cin && s.co == r.c2.co => Some(qconv(c, s, 0..s.co)?),
            Some(_) => return None,
            None => None,
        },
        cin,
    })
}

struct QUp {
    res: Vec<QRes>,
    up: Option<(QConv, usize)>,
    in_dim: usize,
    out_dim: usize,
}

struct QVae {
    key: u64,
    pq: QConv,
    conv_in: QConv,
    mid: [QRes; 2],
    ag: wgpu::Buffer,
    aq: QConv,
    ak: QConv,
    av: QConv,
    ap: QConv,
    ac: usize,
    ups: Vec<QUp>,
    no: wgpu::Buffer,
    no_c: usize,
    conv_out: QConv,
    /// zeros, as wide as the widest padded row (the copy's bias)
    zero: wgpu::Buffer,
}

impl QVae {
    fn build(c: &Ctx, a: &Qi21VaeDecodeArgs) -> Option<QVae> {
        let ac = a.attn_proj.co;
        if a.attn_qkv.co != 3 * ac || a.attn_qkv.ci != ac || a.attn_qkv.k != 1 || a.attn_proj.k != 1 || a.attn_gamma.len() != ac {
            return None;
        }
        let ups = a
            .ups
            .iter()
            .map(|u| -> Option<QUp> {
                Some(QUp {
                    res: u.resnets.iter().map(|r| qres(c, r)).collect::<Option<Vec<_>>>()?,
                    up: match &u.up {
                        Some((cv, ft)) if cv.k == 3 && *ft >= 1 => Some((qconv(c, cv, 0..cv.co)?, *ft)),
                        Some(_) => return None,
                        None => None,
                    },
                    in_dim: u.in_dim,
                    out_dim: u.out_dim,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let widest = ups
            .iter()
            .flat_map(|u| u.res.iter().map(|r| r.c1.cout_p.max(cpad(r.cin))))
            .chain([cpad(ac), cpad(a.conv_in.co), cpad(a.post_quant.co)])
            .max()?;
        Some(QVae {
            key: a.key,
            pq: qconv(c, &a.post_quant, 0..a.post_quant.co)?,
            conv_in: qconv(c, &a.conv_in, 0..a.conv_in.co)?,
            mid: [qres(c, &a.mid_res[0])?, qres(c, &a.mid_res[1])?],
            ag: gamma(c, a.attn_gamma),
            aq: qconv(c, &a.attn_qkv, 0..ac)?,
            ak: qconv(c, &a.attn_qkv, ac..2 * ac)?,
            av: qconv(c, &a.attn_qkv, 2 * ac..3 * ac)?,
            ap: qconv(c, &a.attn_proj, 0..ac)?,
            ac,
            ups,
            no: gamma(c, a.norm_out),
            no_c: a.norm_out.len(),
            conv_out: qconv(c, &a.conv_out, 0..a.conv_out.co)?,
            zero: zi::sbuf(c, (widest * 4) as u64, "qv_zero"),
        })
    }
}

static VSTATE: Mutex<Option<QVae>> = Mutex::new(None);

// ───────────────────────────── recording ─────────────────────────────

fn grid1(n: usize) -> (u32, u32, u32) {
    let wgs = (n as u32).div_ceil(256).max(1);
    (wgs.min(65535), wgs.div_ceil(65535), 1)
}


/// A binding: the whole buffer, or `(offset, size)` bytes of it.
type Bind<'a> = (&'a wgpu::Buffer, Option<(u64, u64)>);

fn entry<'a>(i: u32, b: &Bind<'a>) -> wgpu::BindGroupEntry<'a> {
    match b.1 {
        Some((off, size)) => super::bind_buf_off(i, b.0, off, size),
        None => super::bind_buf(i, b.0),
    }
}

/// A `zi_mm` dispatch with the full 12-word uniform (the conv variants
/// read `cw, ch, cin` from the last words).
fn mmc(c: &Ctx, g: MmCfg, w: [u32; 12], plane: &wgpu::Buffer, act: Bind, out: Bind) -> Option<MmCall> {
    let (m, n, k) = (w[0], w[1], w[2]);
    if !n.is_multiple_of(g.bn) || !k.is_multiple_of(g.bk) {
        return None;
    }
    let pipe = zi::mm_pipe(c, g)?;
    let u = zi::ubuf(c, &w);
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qv_mm"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[super::bind_buf(0, plane), entry(1, &act), entry(2, &out), super::bind_buf(3, &u)],
    });
    Some(MmCall { pipe, bg, grid: (n / g.bn, m.div_ceil(g.bm)) })
}

/// The GEMM tile of a conv writing `cout_p` channels (128×128 when it
/// divides, else 128×64).
fn tile(cout_p: usize, conv: u32, epi: Epi) -> MmCfg {
    let bn = if cout_p.is_multiple_of(128) { 128 } else { 64 };
    MmCfg { conv, ..MmCfg::new(128, bn, 32, 2, 2, epi) }
}

struct Rec<'a> {
    c: &'a Ctx,
    calls: ZCalls,
    zero: &'a wgpu::Buffer,
    /// the device's storage-offset alignment
    align: u64,
}

/// The activation buffers a pass works in (NHWC, `mp_of` rows; `x` the
/// f32 residual stream, `h`/`s` f32 conv outputs, `xn` the f16 conv input,
/// `cp` the up block's saved input).
struct Bufs<'b> {
    x: &'b wgpu::Buffer,
    h: &'b wgpu::Buffer,
    s: &'b wgpu::Buffer,
    xn: &'b wgpu::Buffer,
    cp: &'b wgpu::Buffer,
}

impl Rec<'_> {
    fn k(&mut self, key: &str, src: &str, entry_point: &str, bufs: &[Bind], grid: (u32, u32, u32)) -> Option<()> {
        let pipe = zi::pipeline(self.c, key, src, entry_point)?;
        if bufs.iter().any(|b| b.1.is_some_and(|(off, _)| off % self.align != 0)) {
            return None;
        }
        let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| entry(i as u32, b)).collect();
        let bg = self.c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qv"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        self.calls.push(Class::Io, Call { pipe, bg, grid });
        Some(())
    }

    /// conv of the f16 NHWC image `act` (`h × w` output; `up` = the input
    /// is at half size) → raw (bias-free) f32 `[mp][cout_p]`.
    fn conv(&mut self, cv: &QConv, act: Bind, h: usize, w: usize, up: bool, out: &wgpu::Buffer) -> Option<()> {
        self.conv_s(cv, act, h, w, up, out, 1.0)
    }

    /// `conv` with the f32 epilogue times `oscale`.
    #[allow(clippy::too_many_arguments)]
    fn conv_s(&mut self, cv: &QConv, act: Bind, h: usize, w: usize, up: bool, out: &wgpu::Buffer, oscale: f32) -> Option<()> {
        let conv = match (cv.k, up) {
            (3, false) => 1,
            (3, true) => 2,
            (1, false) => 0,
            _ => return None,
        };
        if act.1.is_some_and(|(off, _)| off % self.align != 0) {
            return None;
        }
        let m = (h * w) as u32;
        let words = [
            m,
            cv.cout_p as u32,
            (cv.k * cv.k * cv.cin_p) as u32,
            cv.cout_p as u32,
            0,
            0,
            oscale.to_bits(),
            0,
            w as u32,
            h as u32,
            cv.cin_p as u32,
            0,
        ];
        let mc = mmc(self.c, tile(cv.cout_p, conv, Epi::F32), words, &cv.plane, act, (out, None))?;
        self.calls.push_mm(Class::Io, mc);
        Some(())
    }

    /// RMS_norm (+ bias, + SiLU) of `x` `[m][cp]` → f16 `out` `[mp][cp]`.
    #[allow(clippy::too_many_arguments)]
    fn rmsn(&mut self, x: &wgpu::Buffer, bias: Option<&wgpu::Buffer>, g: &wgpu::Buffer, c_real: usize, cp: usize, m: usize, silu: bool, out: &wgpu::Buffer) -> Option<()> {
        let mp = mp_of(m);
        let flags = bias.is_some() as u32 | if silu { 2 } else { 0 };
        let u = zi::ubuf(self.c, &[m as u32, mp as u32, cp as u32, c_real as u32, flags, 0, 0, 0]);
        let z = self.zero;
        self.k(
            "qv_rmsn",
            RMSN_SRC,
            "qv_rmsn",
            &[(x, None), (bias.unwrap_or(z), None), (g, None), (out, None), (&u, None)],
            ((mp as u32).min(65535), (mp as u32).div_ceil(65535), 1),
        )
    }

    /// mode 0: x += h + hb; mode 1: x = s + sb + h + hb; mode 2: x = h + hb.
    #[allow(clippy::too_many_arguments)]
    fn combine(&mut self, x: &wgpu::Buffer, h: Bind, hb: &wgpu::Buffer, s: Option<(&wgpu::Buffer, &wgpu::Buffer)>, mode: u32, n: usize, ch: usize) -> Option<()> {
        let u = zi::ubuf(self.c, &[n as u32, ch as u32, mode, 0]);
        let z = self.zero;
        let (sb, sbb) = s.unwrap_or((z, z));
        self.k(
            "zv_combine",
            zi::VAE_COMBINE_SRC,
            "vae_combine",
            &[(x, None), h, (hb, None), (sb, None), (sbb, None), (&u, None)],
            grid1(n),
        )
    }

    /// f32 → f16 conv input, times `scale` (a power of two: the raw
    /// stream's range guard, undone in the consuming conv's epilogue).
    #[allow(clippy::too_many_arguments)]
    fn cast(&mut self, x: &wgpu::Buffer, b: Option<&wgpu::Buffer>, m: usize, ch: usize, out: &wgpu::Buffer, scale: f32) -> Option<()> {
        let mp = mp_of(m);
        let u = zi::ubuf(self.c, &[m as u32, mp as u32, ch as u32, b.is_some() as u32, scale.to_bits(), 0, 0, 0]);
        let z = self.zero;
        self.k("qv_cast", CAST_SRC, "qv_cast", &[(x, None), (b.unwrap_or(z), None), (out, None), (&u, None)], grid1(mp * ch / 4))
    }

    /// One resnet on `x` `[h·w][cin_p]` → `x` `[h·w][cout_p]`.
    fn resnet(&mut self, b: &Bufs, r: &QRes, h: usize, w: usize, cin_p: usize) -> Option<usize> {
        let (m, mp) = (h * w, mp_of(h * w));
        let cout_p = r.c2.cout_p;
        if r.c1.cin_p != cin_p {
            return None;
        }
        self.rmsn(b.x, None, &r.g1, r.cin, cin_p, m, true, b.xn)?;
        self.conv(&r.c1, (b.xn, None), h, w, false, b.h)?;
        self.rmsn(b.h, Some(&r.c1.bias), &r.g2, r.c1.cout, r.c1.cout_p, m, true, b.xn)?;
        self.conv(&r.c2, (b.xn, None), h, w, false, b.h)?;
        match &r.sc {
            Some(sc) => {
                let g = guard();
                self.cast(b.x, None, m, cin_p, b.xn, 1.0 / g)?;
                self.conv_s(sc, (b.xn, None), h, w, false, b.s, g)?;
                self.combine(b.x, (b.h, None), &r.c2.bias, Some((b.s, &sc.bias)), 1, mp * cout_p, cout_p)?;
            }
            None => self.combine(b.x, (b.h, None), &r.c2.bias, None, 0, mp * cout_p, cout_p)?,
        }
        Some(cout_p)
    }
}

/// The mid attention (single head over the `m` pixels) on `b.x`, in place.
fn attention(r: &mut Rec, v: &QVae, b: &Bufs, m: usize) -> Option<()> {
    let c = r.c;
    let cc = v.ac;
    let cp = cpad(cc);
    let mp = mp_of(m);
    if cp != cc || !cc.is_multiple_of(128) {
        return None;
    }
    r.rmsn(b.x, None, &v.ag, cc, cp, m, false, b.xn)?;
    let q16 = zi::sbuf(c, (mp * cc * 2) as u64, "qv_q");
    let k16 = zi::sbuf(c, (mp * cc * 2) as u64, "qv_k");
    let vt16 = zi::sbuf(c, (cc * mp * 2) as u64, "qv_vt");
    let o32 = zi::sbuf(c, (mp * cc * 4) as u64, "qv_o");
    r.conv(&v.aq, (b.xn, None), m, 1, false, b.h)?;
    r.cast(b.h, Some(&v.aq.bias), m, cc, &q16, 1.0)?;
    r.conv(&v.ak, (b.xn, None), m, 1, false, b.h)?;
    r.cast(b.h, Some(&v.ak.bias), m, cc, &k16, 1.0)?;
    // Vᵀ [c][mp] = Wv · xnᵀ (the bias is added after P·V: P's rows sum to 1)
    let w = [cc as u32, mp as u32, cc as u32, mp as u32, 0, 0, 1f32.to_bits(), 0, 0, 0, 0, 0];
    let mc = mmc(c, zi::default_cfg(Epi::F16), w, b.xn, (&v.av.plane, None), (&vt16, None))?;
    r.calls.push_mm(Class::Io, mc);
    // query chunks: S [rows][mp] f32 ≤ 128 MB
    let rows = ((32usize << 20) / mp).clamp(128, mp) / 128 * 128;
    let sc = zi::sbuf(c, (rows * mp * 4) as u64, "qv_sc");
    let pr = zi::sbuf(c, (rows * mp * 2) as u64, "qv_p");
    let scale = 1.0 / (cc as f32).sqrt();
    let mut q0 = 0;
    while q0 < mp {
        let rq = rows.min(mp - q0);
        let w = [rq as u32, mp as u32, cc as u32, mp as u32, 0, q0 as u32, scale.to_bits(), 0, 0, 0, 0, 0];
        let mc = mmc(c, zi::default_cfg(Epi::F32), w, &k16, (&q16, None), (&sc, None))?;
        r.calls.push_mm(Class::Io, mc);
        let u = zi::ubuf(c, &[mp as u32, m as u32, 0, 0]);
        r.k(
            "zv_softmax",
            zi::VAE_SOFTMAX_SRC,
            "vae_softmax",
            &[(&sc, None), (&pr, None), (&u, None)],
            ((rq as u32).min(65535), (rq as u32).div_ceil(65535), 1),
        )?;
        let w = [rq as u32, cc as u32, mp as u32, cc as u32, 0, 0, 1f32.to_bits(), 0, 0, 0, 0, 0];
        let mc = mmc(c, zi::default_cfg(Epi::F32), w, &vt16, (&pr, None), (&o32, Some(((q0 * cc * 4) as u64, (rq * cc * 4) as u64))))?;
        r.calls.push_mm(Class::Io, mc);
        q0 += rq;
    }
    r.cast(&o32, Some(&v.av.bias), m, cc, b.xn, 1.0)?;
    r.conv(&v.ap, (b.xn, None), m, 1, false, b.h)?;
    r.combine(b.x, (b.h, None), &v.ap.bias, None, 0, mp * cc, cc)
}

// ───────────────────────────── bands ─────────────────────────────
//
// Everything after the mid block is spatially local: per-pixel RMS norms,
// 3×3 / 1×1 convs, nearest-2× upsampling, the DupUp shortcut. So the
// high-resolution stages can run in horizontal bands. The leading up
// blocks run on the whole frame while their tensors fit a binding; from
// block S on every band starts from the saved whole-frame stream with a
// halo, and each stage is fed only the rows the next one needs: walking
// back from a band's kept output rows, a 3×3 conv needs one more row on
// each side, an upsample halves the range (rounded outward). Rows next to
// a band edge are wrong (the conv sees zero padding there) but never
// reach a kept row; at the frame's own top and bottom the band edge IS
// the frame edge, so the zero padding is the real one. Every output
// pixel is computed by the same kernels on the same inputs as in the
// whole-frame decode, so the kept rows are bit-for-bit the same.

/// Rows `[lo, hi)` of one resolution level of the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rows {
    lo: usize,
    hi: usize,
}

impl Rows {
    fn n(&self) -> usize {
        self.hi - self.lo
    }
}

fn grow(r: Rows, k: usize, h: usize) -> Rows {
    Rows {
        lo: r.lo.saturating_sub(k),
        hi: (r.hi + k).min(h),
    }
}

fn half(r: Rows) -> Rows {
    Rows {
        lo: r.lo / 2,
        hi: r.hi.div_ceil(2),
    }
}

/// The level of every up block and of the output.
fn levels(v: &QVae) -> (Vec<usize>, usize) {
    let mut l = 0;
    let mut out = Vec::with_capacity(v.ups.len());
    for u in &v.ups {
        out.push(l);
        l += u.up.is_some() as usize;
    }
    (out, l)
}

/// One band: the rows of the saved stream it starts from (the input
/// level of block S), the rows each up block feeds to its upsample, the
/// kept output rows.
#[derive(Clone, Debug)]
struct Band {
    input: Rows,
    feed: Vec<Rows>,
    keep: Rows,
}

/// Walk back from the kept output rows through blocks `s..`.
fn plan_band(v: &QVae, s: usize, keep: Rows, hs: &[usize]) -> Band {
    let (lvl, lf) = levels(v);
    let mut e = grow(keep, v.conv_out.k / 2, hs[lf]);
    let mut feed = vec![Rows { lo: 0, hi: 0 }; v.ups.len()];
    for i in (s..v.ups.len()).rev() {
        let u = &v.ups[i];
        if let Some((cv, _)) = &u.up {
            // the up conv needs its (upsampled) input one row wider; the
            // upsample reads half the rows
            let a = half(grow(e, cv.k / 2, hs[lvl[i] + 1]));
            feed[i] = a;
            e = a;
        }
        let convs: usize = u.res.iter().map(|r| r.c1.k / 2 + r.c2.k / 2).sum();
        e = grow(e, convs, hs[lvl[i]]);
    }
    Band { input: e, feed, keep }
}

/// Largest element counts one pass needs: (x/h/xn, shortcut s, saved input cp).
#[derive(Clone, Copy, Default)]
struct Need {
    big: usize,
    sc: usize,
    cp: usize,
}

impl Need {
    fn max(self, o: Need) -> Need {
        Need {
            big: self.big.max(o.big),
            sc: self.sc.max(o.sc),
            cp: self.cp.max(o.cp),
        }
    }
    fn fits(&self, lim: usize) -> bool {
        self.big.max(self.sc).max(self.cp) * 4 <= lim
    }
}

/// What blocks `blocks` (and the output stage, when `fin`) need, starting
/// from `cur` rows at the first block's level; `feed(i, cur)` = the rows
/// block i feeds to its upsample.
fn need_blocks(v: &QVae, blocks: std::ops::Range<usize>, mut cur: Rows, w0: usize, feed: &dyn Fn(usize, Rows) -> Rows, fin: bool) -> Need {
    let (lvl, lf) = levels(v);
    let mut n = Need::default();
    for i in blocks {
        let u = &v.ups[i];
        let w = w0 << lvl[i];
        let mp = mp_of(cur.n() * w);
        if u.up.is_some() {
            n.cp = n.cp.max(mp * cpad(u.in_dim));
        }
        for r in &u.res {
            n.big = n.big.max(mp * r.c1.cin_p.max(r.c2.cout_p));
            if r.sc.is_some() {
                n.sc = n.sc.max(mp * r.c2.cout_p);
            }
        }
        if let Some((cv, _)) = &u.up {
            let f = feed(i, cur);
            cur = Rows { lo: 2 * f.lo, hi: 2 * f.hi };
            n.big = n.big.max(mp_of(cur.n() * 2 * w) * cv.cout_p);
        }
    }
    if fin {
        let w = w0 << lf;
        n.big = n.big.max(mp_of(cur.n() * w) * v.conv_out.cout_p.max(v.conv_out.cin_p));
    }
    n
}

/// Run up blocks `blocks` on `b` whose `x` holds rows `cur` of the first
/// block's level (buffer row 0 = `cur.lo`); `feed(i, cur)` = the rows block
/// i feeds to its upsample (⊂ `cur`). Returns the rows `x` then holds and
/// their channel padding.
#[allow(clippy::too_many_arguments)]
fn run_blocks(
    r: &mut Rec,
    v: &QVae,
    b: &Bufs,
    blocks: std::ops::Range<usize>,
    mut cur: Rows,
    w0: usize,
    mut ch: usize,
    feed: &dyn Fn(usize, Rows) -> Rows,
) -> Option<(Rows, usize)> {
    let (lvl, _) = levels(v);
    for i in blocks {
        let u = &v.ups[i];
        let w = w0 << lvl[i];
        let hh = cur.n();
        let (m, mp) = (hh * w, mp_of(hh * w));
        if u.up.is_some() {
            if ch != cpad(u.in_dim) {
                return None;
            }
            // save the block's input for the DupUp shortcut
            r.combine(b.cp, (b.x, None), r.zero, None, 2, mp * ch, ch)?;
        }
        let cin = ch;
        for rs in &u.res {
            ch = r.resnet(b, rs, hh, w, ch)?;
        }
        if let Some((cv, ft)) = &u.up {
            if cv.cin_p != ch || cv.cout != u.out_dim {
                return None;
            }
            let f = feed(i, cur);
            if f.lo < cur.lo || f.hi > cur.hi || f.n() == 0 {
                return None;
            }
            let skip = (f.lo - cur.lo) as u64;
            let g = guard();
            r.cast(b.x, None, m, ch, b.xn, 1.0 / g)?;
            let row16 = (w * ch * 2) as u64;
            let (h2, w2) = (2 * f.n(), 2 * w);
            r.conv_s(cv, (b.xn, Some((skip * row16, f.n() as u64 * row16))), h2, w2, true, b.h, g)?;
            ch = cv.cout_p;
            r.combine(b.x, (b.h, None), &cv.bias, None, 2, mp_of(h2 * w2) * ch, ch)?;
            let factor = ft * 4;
            if !(u.out_dim * factor).is_multiple_of(u.in_dim) {
                return None;
            }
            let repeats = u.out_dim * factor / u.in_dim;
            let uu = zi::ubuf(
                r.c,
                &[f.n() as u32, w as u32, cin as u32, u.out_dim as u32, ch as u32, factor as u32, repeats as u32, *ft as u32],
            );
            let row32 = (w * cin * 4) as u64;
            r.k(
                "qv_dupup",
                DUPUP_SRC,
                "qv_dupup",
                &[(b.cp, Some((skip * row32, f.n() as u64 * row32))), (b.x, None), (&uu, None)],
                grid1(h2 * w2 * u.out_dim),
            )?;
            cur = Rows { lo: 2 * f.lo, hi: 2 * f.hi };
        }
    }
    Some((cur, ch))
}

/// norm_out + SiLU → conv_out on `x` (rows `cur` of the output level) →
/// the kept rows into the frame output `ob` `[nch][hf·wf]`.
#[allow(clippy::too_many_arguments)]
fn output_rows(r: &mut Rec, v: &QVae, b: &Bufs, cur: Rows, w: usize, ch: usize, keep: Rows, ob: &wgpu::Buffer, tot: usize, nch: usize) -> Option<()> {
    if v.conv_out.cin_p != ch || cpad(v.no_c) != ch || keep.lo < cur.lo || keep.hi > cur.hi {
        return None;
    }
    let m = cur.n() * w;
    r.rmsn(b.x, None, &v.no, v.no_c, ch, m, true, b.xn)?;
    r.conv(&v.conv_out, (b.xn, None), cur.n(), w, false, b.h)?;
    let n = keep.n() * w;
    let u = zi::ubuf(
        r.c,
        &[n as u32, v.conv_out.cout_p as u32, nch as u32, ((keep.lo - cur.lo) * w) as u32, (keep.lo * w) as u32, tot as u32, 0, 0],
    );
    r.k("qv_out", OUT_SRC, "qv_out", &[(b.h, None), (&v.conv_out.bias, None), (ob, None), (&u, None)], grid1(n))
}

fn env_usize(k: &str) -> Option<usize> {
    std::env::var(k).ok().and_then(|v| v.trim().parse().ok())
}

/// Record, submit and wait for one list under a validation scope.
fn run_list(c: &Ctx, calls: &ZCalls) -> bool {
    let vs = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let ran = calls.run().is_some();
    if let Some(e) = pollster::block_on(vs.pop()) {
        eprintln!("qwen-image-2.1 vae chain: {e}");
        return false;
    }
    ran
}

/// Resident decode; see the module doc. `false` = declined, `out` untouched.
pub(crate) fn decode(a: &Qi21VaeDecodeArgs, z: &[f32], h0: usize, w0: usize, out: &mut [f32]) -> bool {
    if !enabled() {
        return false;
    }
    let Some(c) = zi::zctx() else { return decline("no cooperative-matrix device") };
    let zc = a.post_quant.ci;
    let ups = a.ups.iter().filter(|u| u.up.is_some()).count();
    let (hf, wf) = (h0 << ups, w0 << ups);
    let nch = a.conv_out.co;
    if z.len() != zc * h0 * w0 || out.len() != nch * hf * wf || a.post_quant.k != 1 || nch > 64 || h0 == 0 || w0 == 0 {
        return decline("decode args do not match the decoder");
    }
    let t0 = std::time::Instant::now();
    let Ok(mut g) = VSTATE.lock() else { return false };
    if !g.as_ref().is_some_and(|v| v.key == a.key) {
        *g = None;
        match QVae::build(c, a) {
            Some(v) => *g = Some(v),
            None => return decline("the decoder's shapes are not the ones the chain is written for"),
        }
        if prof_on() {
            eprintln!("qi21 vae chain: weights {:.2}s", t0.elapsed().as_secs_f64());
        }
    }
    let v = g.as_ref().unwrap();
    let t1 = std::time::Instant::now();
    let nb = v.ups.len();
    let (_, lf) = levels(v);
    let hs: Vec<usize> = (0..=lf).map(|l| h0 << l).collect();
    let lim_dev = c.device.limits();
    let lim = lim_dev.max_storage_buffer_binding_size.min(lim_dev.max_buffer_size) as usize;
    let full = |l: usize| Rows { lo: 0, hi: hs[l] };
    let whole = |_: usize, cur: Rows| cur;
    // the head (conv_in, mid block, attention) at the latent resolution
    let head = Need {
        big: mp_of(h0 * w0) * v.conv_in.cout_p.max(v.pq.cout_p).max(cpad(v.ac)),
        ..Default::default()
    };
    // S = how many up blocks run on the whole frame
    let forced = env_usize("CMF_QI21_VAE_BAND_ROWS").filter(|&k| k > 0);
    let auto_s = (0..=nb)
        .rev()
        .find(|&s| head.max(need_blocks(v, 0..s, full(0), w0, &whole, s == nb)).fits(lim));
    let Some(auto_s) = auto_s else {
        return decline(&format!("the latent-resolution tensors exceed the {} MB binding limit", lim >> 20));
    };
    let s = match (forced, env_usize("CMF_QI21_VAE_BAND_FROM")) {
        (_, Some(f)) => f.min(auto_s),
        (Some(_), None) => auto_s.min(1),
        (None, None) => auto_s,
    };
    let need_whole = head.max(need_blocks(v, 0..s, full(0), w0, &whole, s == nb));
    // bands: the fewest (equal) bands whose tensors fit a binding
    let hfin = hs[lf];
    let plan = |nbands: usize| -> Vec<Band> {
        let k = hfin.div_ceil(nbands.max(1));
        (0..hfin.div_ceil(k))
            .map(|j| plan_band(v, s, Rows { lo: j * k, hi: ((j + 1) * k).min(hfin) }, &hs))
            .collect()
    };
    let band_need = |bands: &[Band]| -> Need {
        bands.iter().fold(Need::default(), |acc, bd| {
            let f = |i: usize, _: Rows| bd.feed[i];
            acc.max(need_blocks(v, s..nb, bd.input, w0, &f, true))
        })
    };
    let mut bands: Vec<Band> = Vec::new();
    let mut need_band = Need::default();
    if s < nb {
        let mut nbands = forced.map_or(1, |k| hfin.div_ceil(k));
        loop {
            bands = plan(nbands);
            need_band = band_need(&bands);
            if need_band.fits(lim) || forced.is_some() {
                break;
            }
            if hfin / nbands <= 16 {
                return decline(&format!("no band of {hf}×{wf} fits the {} MB binding limit", lim >> 20));
            }
            nbands += 1;
        }
        if !need_band.fits(lim) {
            return decline(&format!("CMF_QI21_VAE_BAND_ROWS bands exceed the {} MB binding limit", lim >> 20));
        }
    }
    let all = need_whole.max(need_band);
    let band_x = if s < nb { need_band.big } else { 0 };
    let bytes = (all.big * (4 + 2) + need_whole.big * 4 + band_x * 4 + all.sc * 4 + all.cp * 4 + nch * hf * wf * 4) as u64;
    let budget = super::device_vram_budget();
    if budget > 0 && bytes + super::resident_bytes() > budget {
        return decline(&format!("the activations need {:.1} GB", bytes as f64 / 1e9));
    }
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let xw = zi::sbuf(c, (need_whole.big * 4) as u64, "qv_x");
    let xb = zi::sbuf(c, (band_x.max(1) * 4) as u64, "qv_xb");
    let h = zi::sbuf(c, (all.big * 4) as u64, "qv_h");
    let sbuf = zi::sbuf(c, (all.sc.max(1) * 4) as u64, "qv_s");
    let xn = zi::sbuf(c, (all.big * 2) as u64, "qv_xn");
    let cp = zi::sbuf(c, (all.cp.max(1) * 4) as u64, "qv_cp");
    let ob = zi::sbuf(c, (nch * hf * wf * 4) as u64, "qv_out");
    if pollster::block_on(sc.pop()).is_some() {
        return decline("the activations did not fit");
    }
    let wbufs = Bufs { x: &xw, h: &h, s: &sbuf, xn: &xn, cp: &cp };
    let bbufs = Bufs { x: &xb, h: &h, s: &sbuf, xn: &xn, cp: &cp };
    let align = lim_dev.min_storage_buffer_offset_alignment as u64;
    let mut dispatches = 0usize;
    // the whole-frame part: head + blocks 0..s (+ the output when s = nb)
    let mut r = Rec { c, calls: ZCalls::default(), zero: &v.zero, align };
    let whole_part = (|| -> Option<(Rows, usize)> {
        let (m0, mp0) = (h0 * w0, mp_of(h0 * w0));
        let zp = v.pq.cin_p;
        let mut zin = vec![0u16; mp0 * zp];
        for ch in 0..zc {
            for p in 0..m0 {
                zin[p * zp + ch] = cortiq_core::quant::f32_to_f16(z[ch * m0 + p]);
            }
        }
        let zbuf = zi::sbuf_init(c, bytemuck::cast_slice(&zin), "qv_z");
        let b = &wbufs;
        r.conv(&v.pq, (&zbuf, None), m0, 1, false, b.h)?;
        r.cast(b.h, Some(&v.pq.bias), m0, v.pq.cout_p, b.xn, 1.0)?;
        if v.conv_in.cin_p != v.pq.cout_p {
            return None;
        }
        r.conv(&v.conv_in, (b.xn, None), h0, w0, false, b.h)?;
        let mut ch = v.conv_in.cout_p;
        r.combine(b.x, (b.h, None), &v.conv_in.bias, None, 2, mp0 * ch, ch)?;
        ch = r.resnet(b, &v.mid[0], h0, w0, ch)?;
        if ch != cpad(v.ac) {
            return None;
        }
        attention(&mut r, v, b, m0)?;
        ch = r.resnet(b, &v.mid[1], h0, w0, ch)?;
        let (cur, ch) = run_blocks(&mut r, v, b, 0..s, full(0), w0, ch, &whole)?;
        if s == nb {
            output_rows(&mut r, v, b, cur, w0 << lf, ch, full(lf), &ob, hf * wf, nch)?;
        }
        Some((cur, ch))
    })();
    let Some((cur_s, ch_s)) = whole_part else {
        return decline("the chain could not be recorded for this decoder");
    };
    dispatches += r.calls.len();
    if !run_list(c, &r.calls) {
        return decline("a decode command buffer failed");
    }
    // the bands: blocks s..nb from the saved stream in `xw`
    for bd in &bands {
        let mut r = Rec { c, calls: ZCalls::default(), zero: &v.zero, align };
        let ls = levels(v).0[s];
        let w = w0 << ls;
        let recorded = (|| -> Option<()> {
            if cur_s != full(ls) || bd.input.hi > cur_s.hi {
                return None;
            }
            let row32 = (w * ch_s * 4) as u64;
            let n = bd.input.n() * w * ch_s;
            r.combine(
                &xb,
                (&xw, Some((bd.input.lo as u64 * row32, bd.input.n() as u64 * row32))),
                &v.zero,
                None,
                2,
                n,
                ch_s,
            )?;
            let f = |i: usize, _: Rows| bd.feed[i];
            let (cur, ch) = run_blocks(&mut r, v, &bbufs, s..nb, bd.input, w0, ch_s, &f)?;
            output_rows(&mut r, v, &bbufs, cur, w0 << lf, ch, bd.keep, &ob, hf * wf, nch)
        })();
        if recorded.is_none() {
            return decline("a band could not be recorded (offset alignment or shapes)");
        }
        dispatches += r.calls.len();
        if !run_list(c, &r.calls) {
            return decline("a decode command buffer failed");
        }
    }
    let Some(raw) = zi::read_bytes(c, &ob, (nch * hf * wf * 4) as u64) else {
        return decline("the decode readback failed");
    };
    out.copy_from_slice(bytemuck::cast_slice(&raw));
    if let Ok(path) = std::env::var("CMF_QI21_VAE_DUMP") {
        let _ = std::fs::write(&path, &raw);
    }
    // the chain's pipelines were compiled at first use: keep them for the
    // next process (once per process)
    static FLUSHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !FLUSHED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        super::pipeline_cache_flush();
    }
    if prof_on() {
        let bandinfo = if bands.is_empty() {
            "whole frame".to_string()
        } else {
            format!("{} blocks whole, then {} bands of {} rows", s, bands.len(), bands[0].keep.n())
        };
        eprintln!(
            "qi21 vae chain: {hf}×{wf} decode {:.2}s ({bandinfo}; {dispatches} dispatches, {:.1} GB activations)",
            t1.elapsed().as_secs_f64(),
            bytes as f64 / 1e9
        );
    }
    true
}

/// Drop the decoder's weights (module-local state only).
pub(crate) fn release() {
    if let Ok(mut g) = VSTATE.lock() {
        *g = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_rows_walk_back_through_the_stages() {
        // one 3×3 conv at the output, one up block (two 3×3 resnet convs,
        // then 2× + a 3×3 conv): keep rows [100, 164) of 256
        let k = Rows { lo: 100, hi: 164 };
        let e = grow(k, 1, 256); // conv_out
        let e = grow(e, 0, 256); // a final block without resnets
        let a = half(grow(e, 1, 256)); // the up conv, then the upsample
        assert_eq!(a, Rows { lo: 49, hi: 83 });
        let input = grow(a, 2, 128);
        assert_eq!(input, Rows { lo: 47, hi: 85 });
        // at the frame edges the band stops at the edge
        assert_eq!(grow(Rows { lo: 0, hi: 10 }, 3, 12), Rows { lo: 0, hi: 12 });
    }
}
