//! Qwen-Image-2.1 VAE decoder on wgpu, resident (the `gpu::qi21_vae_decode`
//! contract): the whole decoder in one submission — one latent upload, one
//! `[4][H·W]` readback — instead of `gpu::vae_conv2d` per conv with the
//! activations crossing the bus and the RMS norms on the host.
//!
//! The Z-Image resident VAE's recipe (`zimage.rs`, "Resident Flux-VAE
//! decoder"): NHWC activations, every 3×3 conv an implicit GEMM on `zi_mm`
//! (`MmCfg::conv`; the nearest-2× upsample folded into the A-tile gather),
//! 1×1 convs as plain GEMMs, f16 conv inputs with f32 accumulation and an
//! f32 residual stream, the mid attention as QKᵀ → softmax → P·V over
//! query chunks. Its combine / cast / softmax kernels are reused as they
//! are. What differs here:
//! - `qv_rmsn`: the Wan `RMS_norm` (x/max(‖x‖₂, 1e-12)·√C·γ per pixel over
//!   the channels) with the conv bias added on read and an optional SiLU,
//!   → the next conv's f16 input;
//! - `qv_dupup`: the `DupUp3D` shortcut of an up block (a channel shuffle
//!   onto the 2× grid) added into the upsampled stream;
//! - channels are padded to multiples of 64 (1152, 576, 288 → 320,
//!   144 → 192, 4 → 64: zero weights, zero γ), and a GEMM whose padded
//!   output width is not a multiple of 128 uses a 128×64 tile.
//!
//! Measured (RTX PRO 4000 Blackwell): the 1024² decode 0.49–0.64 s plus
//! 0.26–0.47 s for the weights (f16 planes, once per model) against 43 s
//! for the per-conv path; 256² 0.11 s. u8 PSNR against the per-conv path
//! 67.4 dB at 1024²; against the fp32 diffusers decoder at 256² 66.0 dB
//! (the per-conv path: 69.1 dB). The 1024² activations take 5.5 GB; at
//! 2048² one of them (5.4 GB) exceeds the 2 GB binding limit and the chain
//! declines (the per-conv path runs).
//!
//! Knob: `CMF_QI21_VAE_CHAIN=0` (the per-conv path).

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

/// NHWC `[m][ld]` (first `nch` channels) + bias → NCHW `[nch][m]`.
const OUT_SRC: &str = r#"
struct OP { m: u32, ld: u32, nch: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform> p: OP;
@compute @workgroup_size(256)
fn qv_out(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * 256u) + gid.x;
  if (i >= p.m) { return; }
  for (var ch = 0u; ch < p.nch; ch = ch + 1u) {
    outp[ch * p.m + i] = y[i * p.ld + ch] + b[ch];
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

/// A `zi_mm` dispatch with the full 12-word uniform (the conv variants
/// read `cw, ch, cin` from the last words).
#[allow(clippy::too_many_arguments)]
fn mmc(c: &Ctx, g: MmCfg, w: [u32; 12], plane: &wgpu::Buffer, act: &wgpu::Buffer, out: &wgpu::Buffer, out_off: Option<(u64, u64)>) -> Option<MmCall> {
    let (m, n, k) = (w[0], w[1], w[2]);
    if !n.is_multiple_of(g.bn) || !k.is_multiple_of(g.bk) {
        return None;
    }
    let pipe = zi::mm_pipe(c, g)?;
    let u = zi::ubuf(c, &w);
    let ob = match out_off {
        Some((off, size)) => super::bind_buf_off(2, out, off, size),
        None => super::bind_buf(2, out),
    };
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("qv_mm"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[super::bind_buf(0, plane), super::bind_buf(1, act), ob, super::bind_buf(3, &u)],
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
}

/// The activation buffers of one decode (NHWC, `mp_of` rows).
struct Bufs {
    x: wgpu::Buffer,
    h: wgpu::Buffer,
    s: wgpu::Buffer,
    xn: wgpu::Buffer,
    cp: wgpu::Buffer,
}

impl Rec<'_> {
    fn k(&mut self, key: &str, src: &str, entry: &str, bufs: &[&wgpu::Buffer], grid: (u32, u32, u32)) -> Option<()> {
        let pipe = zi::pipeline(self.c, key, src, entry)?;
        let b = zi::bg(self.c, &pipe, bufs);
        self.calls.push(Class::Io, Call { pipe, bg: b, grid });
        Some(())
    }

    /// conv of the f16 NHWC image `act` (`h × w` output; `up` = the input
    /// is at half size) → raw (bias-free) f32 `[mp][cout_p]`.
    fn conv(&mut self, cv: &QConv, act: &wgpu::Buffer, h: usize, w: usize, up: bool, out: &wgpu::Buffer) -> Option<()> {
        let conv = match (cv.k, up) {
            (3, false) => 1,
            (3, true) => 2,
            (1, false) => 0,
            _ => return None,
        };
        let m = (h * w) as u32;
        let words = [
            m,
            cv.cout_p as u32,
            (cv.k * cv.k * cv.cin_p) as u32,
            cv.cout_p as u32,
            0,
            0,
            1f32.to_bits(),
            0,
            w as u32,
            h as u32,
            cv.cin_p as u32,
            0,
        ];
        let mc = mmc(self.c, tile(cv.cout_p, conv, Epi::F32), words, &cv.plane, act, out, None)?;
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
        self.k("qv_rmsn", RMSN_SRC, "qv_rmsn", &[x, bias.unwrap_or(z), g, out, &u], ((mp as u32).min(65535), (mp as u32).div_ceil(65535), 1))
    }

    /// mode 0: x += h + hb; mode 1: x = s + sb + h + hb; mode 2: x = h + hb.
    #[allow(clippy::too_many_arguments)]
    fn combine(&mut self, x: &wgpu::Buffer, h: &wgpu::Buffer, hb: &wgpu::Buffer, s: Option<(&wgpu::Buffer, &wgpu::Buffer)>, mode: u32, n: usize, ch: usize) -> Option<()> {
        let u = zi::ubuf(self.c, &[n as u32, ch as u32, mode, 0]);
        let z = self.zero;
        let (sb, sbb) = s.unwrap_or((z, z));
        self.k("zv_combine", zi::VAE_COMBINE_SRC, "vae_combine", &[x, h, hb, sb, sbb, &u], grid1(n))
    }

    fn cast(&mut self, x: &wgpu::Buffer, b: Option<&wgpu::Buffer>, m: usize, ch: usize, out: &wgpu::Buffer) -> Option<()> {
        let mp = mp_of(m);
        let u = zi::ubuf(self.c, &[m as u32, mp as u32, ch as u32, b.is_some() as u32]);
        let z = self.zero;
        self.k("zv_cast", zi::VAE_CAST_SRC, "vae_cast", &[x, b.unwrap_or(z), out, &u], grid1(mp * ch / 4))
    }

    /// One resnet on `x` `[h·w][cin_p]` → `x` `[h·w][cout_p]`.
    fn resnet(&mut self, b: &Bufs, r: &QRes, h: usize, w: usize, cin_p: usize) -> Option<usize> {
        let (m, mp) = (h * w, mp_of(h * w));
        let cout_p = r.c2.cout_p;
        if r.c1.cin_p != cin_p {
            return None;
        }
        self.rmsn(&b.x, None, &r.g1, r.cin, cin_p, m, true, &b.xn)?;
        self.conv(&r.c1, &b.xn, h, w, false, &b.h)?;
        self.rmsn(&b.h, Some(&r.c1.bias), &r.g2, r.c1.cout, r.c1.cout_p, m, true, &b.xn)?;
        self.conv(&r.c2, &b.xn, h, w, false, &b.h)?;
        match &r.sc {
            Some(sc) => {
                self.cast(&b.x, None, m, cin_p, &b.xn)?;
                self.conv(sc, &b.xn, h, w, false, &b.s)?;
                self.combine(&b.x, &b.h, &r.c2.bias, Some((&b.s, &sc.bias)), 1, mp * cout_p, cout_p)?;
            }
            None => self.combine(&b.x, &b.h, &r.c2.bias, None, 0, mp * cout_p, cout_p)?,
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
    r.rmsn(&b.x, None, &v.ag, cc, cp, m, false, &b.xn)?;
    let q16 = zi::sbuf(c, (mp * cc * 2) as u64, "qv_q");
    let k16 = zi::sbuf(c, (mp * cc * 2) as u64, "qv_k");
    let vt16 = zi::sbuf(c, (cc * mp * 2) as u64, "qv_vt");
    let o32 = zi::sbuf(c, (mp * cc * 4) as u64, "qv_o");
    r.conv(&v.aq, &b.xn, m, 1, false, &b.h)?;
    r.cast(&b.h, Some(&v.aq.bias), m, cc, &q16)?;
    r.conv(&v.ak, &b.xn, m, 1, false, &b.h)?;
    r.cast(&b.h, Some(&v.ak.bias), m, cc, &k16)?;
    // Vᵀ [c][mp] = Wv · xnᵀ (the bias is added after P·V: P's rows sum to 1)
    let w = [cc as u32, mp as u32, cc as u32, mp as u32, 0, 0, 1f32.to_bits(), 0, 0, 0, 0, 0];
    let mc = mmc(c, zi::default_cfg(Epi::F16), w, &b.xn, &v.av.plane, &vt16, None)?;
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
        let mc = mmc(c, zi::default_cfg(Epi::F32), w, &k16, &q16, &sc, None)?;
        r.calls.push_mm(Class::Io, mc);
        let u = zi::ubuf(c, &[mp as u32, m as u32, 0, 0]);
        r.k("zv_softmax", zi::VAE_SOFTMAX_SRC, "vae_softmax", &[&sc, &pr, &u], ((rq as u32).min(65535), (rq as u32).div_ceil(65535), 1))?;
        let w = [rq as u32, cc as u32, mp as u32, cc as u32, 0, 0, 1f32.to_bits(), 0, 0, 0, 0, 0];
        let mc = mmc(c, zi::default_cfg(Epi::F32), w, &vt16, &pr, &o32, Some(((q0 * cc * 4) as u64, (rq * cc * 4) as u64)))?;
        r.calls.push_mm(Class::Io, mc);
        q0 += rq;
    }
    r.cast(&o32, Some(&v.av.bias), m, cc, &b.xn)?;
    r.conv(&v.ap, &b.xn, m, 1, false, &b.h)?;
    r.combine(&b.x, &b.h, &v.ap.bias, None, 0, mp * cc, cc)
}

/// Largest NHWC element counts of one decode: (x/h/xn, shortcut s, the
/// up block's saved input).
fn sizes(v: &QVae, h0: usize, w0: usize) -> (usize, usize, usize) {
    let mut big = mp_of(h0 * w0) * v.conv_in.cout_p.max(v.pq.cout_p).max(cpad(v.ac));
    let (mut sc, mut cp) = (0usize, 0usize);
    let (mut h, mut w) = (h0, w0);
    for u in &v.ups {
        let mp = mp_of(h * w);
        cp = cp.max(mp * cpad(u.in_dim));
        for r in &u.res {
            big = big.max(mp * r.c1.cin_p.max(r.c2.cout_p));
            if r.sc.is_some() {
                sc = sc.max(mp * r.c2.cout_p);
            }
        }
        if let Some((cv, _)) = &u.up {
            h *= 2;
            w *= 2;
            big = big.max(mp_of(h * w) * cv.cout_p);
        }
    }
    big = big.max(mp_of(h * w) * v.conv_out.cout_p.max(v.conv_out.cin_p));
    (big, sc.max(1), cp.max(1))
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
    let (big, scn, cpn) = sizes(v, h0, w0);
    let lim = c.device.limits();
    let maxb = lim.max_storage_buffer_binding_size.min(lim.max_buffer_size);
    if (big * 4) as u64 > maxb {
        return decline(&format!(
            "a {} MB activation exceeds the {} MB binding limit at {hf}×{wf}",
            (big * 4) >> 20,
            maxb >> 20
        ));
    }
    let budget = super::device_vram_budget();
    let need = (big * (4 + 4 + 2) + scn * 4 + cpn * 4) as u64;
    if budget > 0 && need + super::resident_bytes() > budget {
        return decline(&format!("the activations need {:.1} GB", need as f64 / 1e9));
    }
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let bufs = Bufs {
        x: zi::sbuf(c, (big * 4) as u64, "qv_x"),
        h: zi::sbuf(c, (big * 4) as u64, "qv_h"),
        s: zi::sbuf(c, (scn * 4) as u64, "qv_s"),
        xn: zi::sbuf(c, (big * 2) as u64, "qv_xn"),
        cp: zi::sbuf(c, (cpn * 4) as u64, "qv_cp"),
    };
    if pollster::block_on(sc.pop()).is_some() {
        return decline("the activations did not fit");
    }
    let mut r = Rec {
        c,
        calls: ZCalls::default(),
        zero: &v.zero,
    };
    let recorded = (|| -> Option<wgpu::Buffer> {
        // the latent: NHWC f16 [mp0][cpad(z)]
        let (m0, mp0) = (h0 * w0, mp_of(h0 * w0));
        let zp = v.pq.cin_p;
        let mut zin = vec![0u16; mp0 * zp];
        for ch in 0..zc {
            for p in 0..m0 {
                zin[p * zp + ch] = cortiq_core::quant::f32_to_f16(z[ch * m0 + p]);
            }
        }
        let zbuf = zi::sbuf_init(c, bytemuck::cast_slice(&zin), "qv_z");
        r.conv(&v.pq, &zbuf, m0, 1, false, &bufs.h)?;
        r.cast(&bufs.h, Some(&v.pq.bias), m0, v.pq.cout_p, &bufs.xn)?;
        if v.conv_in.cin_p != v.pq.cout_p {
            return None;
        }
        r.conv(&v.conv_in, &bufs.xn, h0, w0, false, &bufs.h)?;
        let mut ch = v.conv_in.cout_p;
        r.combine(&bufs.x, &bufs.h, &v.conv_in.bias, None, 2, mp0 * ch, ch)?;
        ch = r.resnet(&bufs, &v.mid[0], h0, w0, ch)?;
        if ch != cpad(v.ac) {
            return None;
        }
        attention(&mut r, v, &bufs, m0)?;
        ch = r.resnet(&bufs, &v.mid[1], h0, w0, ch)?;
        let (mut hh, mut ww) = (h0, w0);
        for u in &v.ups {
            let (m, mp) = (hh * ww, mp_of(hh * ww));
            if u.up.is_some() {
                if ch != cpad(u.in_dim) {
                    return None;
                }
                // save the block's input for the DupUp shortcut
                r.combine(&bufs.cp, &bufs.x, &v.zero, None, 2, mp * ch, ch)?;
            }
            for rs in &u.res {
                ch = r.resnet(&bufs, rs, hh, ww, ch)?;
            }
            if let Some((cv, ft)) = &u.up {
                if cv.cin_p != ch || cv.cout != u.out_dim {
                    return None;
                }
                r.cast(&bufs.x, None, m, ch, &bufs.xn)?;
                hh *= 2;
                ww *= 2;
                r.conv(cv, &bufs.xn, hh, ww, true, &bufs.h)?;
                ch = cv.cout_p;
                let (m2, mp2) = (hh * ww, mp_of(hh * ww));
                r.combine(&bufs.x, &bufs.h, &cv.bias, None, 2, mp2 * ch, ch)?;
                let factor = ft * 4;
                if !(u.out_dim * factor).is_multiple_of(u.in_dim) {
                    return None;
                }
                let repeats = u.out_dim * factor / u.in_dim;
                let uu = zi::ubuf(
                    c,
                    &[(hh / 2) as u32, (ww / 2) as u32, cpad(u.in_dim) as u32, u.out_dim as u32, ch as u32, factor as u32, repeats as u32, *ft as u32],
                );
                r.k("qv_dupup", DUPUP_SRC, "qv_dupup", &[&bufs.cp, &bufs.x, &uu], grid1(m2 * u.out_dim))?;
            }
        }
        let m = hh * ww;
        if v.conv_out.cin_p != ch || cpad(v.no_c) != ch {
            return None;
        }
        r.rmsn(&bufs.x, None, &v.no, v.no_c, ch, m, true, &bufs.xn)?;
        r.conv(&v.conv_out, &bufs.xn, hh, ww, false, &bufs.h)?;
        let ob = zi::sbuf(c, (nch * m * 4) as u64, "qv_out");
        let u = zi::ubuf(c, &[m as u32, v.conv_out.cout_p as u32, nch as u32, 0]);
        r.k("qv_out", OUT_SRC, "qv_out", &[&bufs.h, &v.conv_out.bias, &ob, &u], grid1(m))?;
        Some(ob)
    })();
    let Some(ob) = recorded else {
        return decline("the chain could not be recorded for this decoder");
    };
    let vs = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let ran = r.calls.run().is_some();
    if let Some(e) = pollster::block_on(vs.pop()) {
        eprintln!("qwen-image-2.1 vae chain: {e}");
        return decline("a decode command buffer failed");
    }
    if !ran {
        return false;
    }
    let Some(raw) = zi::read_bytes(c, &ob, (nch * hf * wf * 4) as u64) else {
        return decline("the decode readback failed");
    };
    out.copy_from_slice(bytemuck::cast_slice(&raw));
    // the chain's pipelines were compiled at first use: keep them for the
    // next process (once per process)
    static FLUSHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !FLUSHED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        super::pipeline_cache_flush();
    }
    if prof_on() {
        eprintln!(
            "qi21 vae chain: {hf}×{wf} decode {:.2}s ({} dispatches, {:.1} GB activations)",
            t1.elapsed().as_secs_f64(),
            r.calls.len(),
            need as f64 / 1e9
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
