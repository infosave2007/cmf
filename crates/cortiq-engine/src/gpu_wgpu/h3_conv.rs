//! A chain of convolution stages on wgpu/Vulkan (`gpu::conv_chain`): the
//! CNN encoder of the MiniMax-H3 VAEs, which turns an edit's reference
//! picture into latents. The tile goes up once, every stage runs on the
//! card, and only the last activation comes home.
//!
//! Built from the Z-Image VAE decoder's pieces (`zimage.rs`): activations
//! are NHWC, f32 between stages and f16 where a convolution reads them; a
//! convolution is `zi_mm` gathering its taps from the image (no im2col
//! panel) — 3×3 with one reflected row and column on every side
//! ([`MmCfg::conv`] 4), 3×3 stride 2 with the reflection on the far side
//! only (5), or 1×1 (a plain GEMM); GroupNorm is two passes of partial sums
//! (`h3_gn_part`, any channel count: a workgroup per chunk of rows and
//! group) folded by `vae_gn_fin`, then `vae_gn_apply` writes the normed,
//! SiLU-ed f16 image. The kernels' f16 planes are made once from the host's
//! f32 weights and kept by the slice's address.
//!
//! Measured, RTX A5000, a 1024 x 1024 reference picture: 0.8 s, against
//! 84.4 s for the encoder on the host (7.65 cores); the edit's picture is
//! 50.9 dB from the host encoder's. `CMF_H3_WGPU_CONV=0` keeps the host
//! path.

use crate::gpu::ConvStep;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::zimage::{self as zi, Call, Class, Epi, MmArgs, MmCfg, ZCalls};
use super::Ctx;

/// Rows of pixels a GroupNorm workgroup sums.
const GN_ROWS: usize = 1024;
const GROUPS: usize = 32;

/// Partial sums of a GroupNorm over NHWC `x [m][c]`: one workgroup per
/// (chunk of `rows` pixels, group). flags: 1 = add `bias[ch]` on read,
/// 4 = pass 2 (Σ (v − mean)² against `stat[group]`).
const GN_PART_SRC: &str = r#"
struct VP { m: u32, c: u32, cpg: u32, rows: u32, nchunk: u32, flags: u32, g: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> stat: array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<uniform> p: VP;
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn h3_gn_part(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let chunk = wid.x;
  let grp = wid.y;
  let r0 = chunk * p.rows;
  let r1 = min(r0 + p.rows, p.m);
  let n = (r1 - r0) * p.cpg;
  let hb = (p.flags & 1u) != 0u;
  let pass2 = (p.flags & 4u) != 0u;
  var a = 0.0;
  for (var e = t; e < n; e = e + 256u) {
    let ch = grp * p.cpg + e % p.cpg;
    var v = x[(r0 + e / p.cpg) * p.c + ch];
    if (hb) { v = v + bias[ch]; }
    if (pass2) { let d = v - stat[grp]; v = d * d; }
    a = a + v;
  }
  sh[t] = a;
  workgroupBarrier();
  var st = 128u;
  loop {
    if (st == 0u) { break; }
    if (t < st) { sh[t] = sh[t] + sh[t + st]; }
    workgroupBarrier();
    st = st >> 1u;
  }
  if (t == 0u) { part[grp * p.nchunk + chunk] = sh[0]; }
}
"#;

fn enabled() -> bool {
    std::env::var("CMF_H3_WGPU").as_deref() != Ok("0") && std::env::var("CMF_H3_WGPU_CONV").as_deref() != Ok("0")
}

/// A convolution's device weights: the f16 plane `[cout_p][k²·cin_p]` in
/// (tap, channel) order with zero rows and columns for the padding, and
/// the f32 bias `[cout_p]`.
struct ConvDev {
    plane: wgpu::Buffer,
    bias: wgpu::Buffer,
    cin_p: usize,
    cout_p: usize,
}

/// Planes by the address and length of the host weight they were made from.
type ConvMap = HashMap<(usize, usize), Arc<ConvDev>>;
static CONVS: Mutex<Option<ConvMap>> = Mutex::new(None);

/// Drop the planes (a stage is done). Never brings a device up.
pub(crate) fn release() {
    if let Ok(mut g) = CONVS.lock() {
        *g = None;
    }
}

fn conv_dev(c: &Ctx, w: &[f32], b: &[f32], oc: usize, ic: usize, k: usize) -> Option<Arc<ConvDev>> {
    let key = (w.as_ptr() as usize, w.len());
    let mut g = CONVS.lock().ok()?;
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(d) = map.get(&key) {
        return Some(d.clone());
    }
    let kk = k * k;
    if w.len() != oc * ic * kk || b.len() != oc {
        return None;
    }
    let cin_p = ic.next_multiple_of(32);
    let cout_p = oc.next_multiple_of(128);
    let kd = kk * cin_p;
    let mut plane = vec![0u16; cout_p * kd];
    for o in 0..oc {
        let row = &mut plane[o * kd..(o + 1) * kd];
        for ci in 0..ic {
            for tap in 0..kk {
                row[tap * cin_p + ci] = cortiq_core::quant::f32_to_f16(w[(o * ic + ci) * kk + tap]);
            }
        }
    }
    let mut bias = vec![0f32; cout_p];
    bias[..oc].copy_from_slice(b);
    let d = Arc::new(ConvDev {
        plane: zi::sbuf_init(c, bytemuck::cast_slice(&plane), "h3cv_plane"),
        bias: zi::sbuf_init(c, bytemuck::cast_slice(&bias), "h3cv_bias"),
        cin_p,
        cout_p,
    });
    map.insert(key, d.clone());
    Some(d)
}

fn mp_of(m: usize) -> usize {
    m.next_multiple_of(128)
}

fn grid1(n: usize) -> (u32, u32, u32) {
    let wgs = (n as u32).div_ceil(256).max(1);
    (wgs.min(65535), wgs.div_ceil(65535), 1)
}

/// Where the current activation lives while the stages are recorded.
enum Cur {
    /// f16 in `xn` (an input image, or a GroupNorm's output).
    Half,
    /// f32 in `x`, complete.
    Full,
    /// f32 in `h`, a convolution's output whose bias is still to be added.
    Raw(Arc<ConvDev>),
}

/// The recorder of one chain: its buffers and the dispatches.
struct Rec<'a> {
    c: &'a Ctx,
    calls: ZCalls,
    x: wgpu::Buffer,
    h: wgpu::Buffer,
    s: wgpu::Buffer,
    xn: wgpu::Buffer,
    dummy: wgpu::Buffer,
    stat: wgpu::Buffer,
    part: wgpu::Buffer,
}

impl Rec<'_> {
    fn k(&mut self, key: &str, src: &str, entry: &str, bufs: &[&wgpu::Buffer], grid: (u32, u32, u32)) -> Option<()> {
        let pipe = zi::pipeline(self.c, key, src, entry)?;
        let bg = zi::bg(self.c, &pipe, bufs);
        self.calls.push(Class::Io, Call { pipe, bg, grid });
        Some(())
    }

    /// GroupNorm(32) of `src [m][ch]` (+ `bias` on read) with SiLU → f16 `xn`.
    fn gn_silu(&mut self, raw: Option<&ConvDev>, w: &[f32], b: &[f32], m: usize, ch: usize) -> Option<()> {
        if w.len() != ch || b.len() != ch || !ch.is_multiple_of(4 * GROUPS) {
            return None;
        }
        let cpg = ch / GROUPS;
        let nchunk = m.div_ceil(GN_ROWS);
        let mp = mp_of(m);
        let hb = raw.is_some() as u32;
        let src = if raw.is_some() { self.h.clone() } else { self.x.clone() };
        let bias = raw.map_or(self.dummy.clone(), |r| r.bias.clone());
        let (stat, part) = (self.stat.clone(), self.part.clone());
        for pass2 in [false, true] {
            let flags = hb | if pass2 { 4 } else { 0 };
            let head = [m as u32, ch as u32, cpg as u32, GN_ROWS as u32, nchunk as u32, flags, GROUPS as u32];
            let u = zi::ubuf(self.c, &[head[0], head[1], head[2], head[3], head[4], head[5], head[6], 0]);
            self.k("h3_gn_part", GN_PART_SRC, "h3_gn_part", &[&src, &bias, &stat, &part, &u], (nchunk as u32, GROUPS as u32, 1))?;
            let u = zi::ubuf(
                self.c,
                &[head[0], head[1], head[2], head[3], head[4], head[5], head[6], 1e-6f32.to_bits()],
            );
            self.k("zv_gn_fin", zi::VAE_GN_FIN_SRC, "vae_gn_fin", &[&part, &stat, &u], (GROUPS as u32, 1, 1))?;
        }
        let gw = zi::sbuf_init(self.c, bytemuck::cast_slice(w), "h3cv_gw");
        let gb = zi::sbuf_init(self.c, bytemuck::cast_slice(b), "h3cv_gb");
        let u = zi::ubuf(self.c, &[m as u32, ch as u32, cpg as u32, 0, 0, hb | 2, GROUPS as u32, mp as u32]);
        let xn = self.xn.clone();
        self.k(
            "zv_gn_apply",
            zi::VAE_GN_APPLY_SRC,
            "vae_gn_apply",
            &[&src, &bias, &stat, &gw, &gb, &xn, &u],
            grid1(mp * ch / 4),
        )
    }

    /// The f32 image `src [m][ch]` (+ `bias`) as f16 into `xn`.
    fn cast(&mut self, raw: Option<&ConvDev>, m: usize, ch: usize) -> Option<()> {
        let mp = mp_of(m);
        let src = if raw.is_some() { self.h.clone() } else { self.x.clone() };
        let bias = raw.map_or(self.dummy.clone(), |r| r.bias.clone());
        let u = zi::ubuf(self.c, &[m as u32, mp as u32, ch as u32, raw.is_some() as u32]);
        let xn = self.xn.clone();
        self.k("zv_cast", zi::VAE_CAST_SRC, "vae_cast", &[&src, &bias, &xn, &u], grid1(mp * ch / 4))
    }

    /// A convolution of the f16 image in `xn` into `out`, raw (no bias).
    /// `mode`: the `zi_mm` gather (0 = 1×1); `(oh, ow)`: the output's size.
    fn conv(&mut self, cv: &ConvDev, mode: u32, k: usize, oh: usize, ow: usize, out: &wgpu::Buffer) -> Option<()> {
        let g = MmCfg { conv: mode, ..zi::default_cfg(Epi::F32) };
        let a = MmArgs {
            m: (oh * ow) as u32,
            n: cv.cout_p as u32,
            k: (k * k * cv.cin_p) as u32,
            ldo: cv.cout_p as u32,
            ocol: 0,
            arow: 0,
            oscale: 1.0,
            conv: [ow as u32, oh as u32, cv.cin_p as u32],
        };
        let xn = self.xn.clone();
        let mc = zi::mm_call(self.c, g, &a, &cv.plane, &xn, out)?;
        self.calls.push_mm(Class::Io, mc);
        Some(())
    }

    /// mode 0: `x += h + hb`; 1: `x = s + sb + h + hb`; 2: `x = h + hb`.
    fn combine(&mut self, hb: &wgpu::Buffer, sb: Option<&wgpu::Buffer>, mode: u32, n: usize, ch: usize) -> Option<()> {
        let u = zi::ubuf(self.c, &[n as u32, ch as u32, mode, 0]);
        let (x, h, s, d) = (self.x.clone(), self.h.clone(), self.s.clone(), self.dummy.clone());
        self.k(
            "zv_combine",
            zi::VAE_COMBINE_SRC,
            "vae_combine",
            &[&x, &h, hb, &s, sb.unwrap_or(&d), &u],
            grid1(n),
        )
    }
}

/// The stages over `x` `[c][h][w]`; `([c'][h'][w'], c', h', w')`, or None
/// where this chain does not take them (the caller runs its own).
pub(crate) fn conv_chain(steps: &[ConvStep<'_>], x: &[f32], c0: usize, h0: usize, w0: usize) -> Option<(Vec<f32>, usize, usize, usize)> {
    if !enabled() || x.len() != c0 * h0 * w0 || c0 > 32 || steps.is_empty() {
        return None;
    }
    let c = zi::zctx()?;
    let _op = c.op_gate.enter();
    let _gate = c.mm_gate.lock().unwrap();

    // Walk the shapes once: what the chain takes, and the largest image.
    // Between stages an activation is `[pixels][channels]` with the
    // channels of the convolution that made it, so every convolution but
    // the last must make a multiple of 128 of them.
    let (mut ch, mut hh, mut ww) = (c0.next_multiple_of(32), h0, w0);
    let mut real = c0;
    let mut elems = mp_of(h0 * w0) * ch;
    let last_conv = steps.iter().rposition(|s| matches!(s, ConvStep::Conv { .. }))?;
    for (i, s) in steps.iter().enumerate() {
        if let ConvStep::Conv { out_ch, in_ch, k, stride, pad, far, .. } = s {
            let shape_ok = matches!((*k, *stride, *pad, *far), (3, 1, 1, false) | (3, 2, 0, true) | (1, 1, 0, false));
            if !shape_ok || *in_ch != real || (i != last_conv && !out_ch.is_multiple_of(128)) {
                return None;
            }
            if *stride == 2 {
                if !hh.is_multiple_of(2) || !ww.is_multiple_of(2) {
                    return None;
                }
                (hh, ww) = (hh / 2, ww / 2);
            }
            real = *out_ch;
            ch = out_ch.next_multiple_of(128);
            elems = elems.max(mp_of(hh * ww) * ch);
        }
    }
    let lim = c.device.limits();
    if (elems * 4) as u64 > lim.max_storage_buffer_binding_size.min(lim.max_buffer_size) {
        return None;
    }
    if super::resident_bytes() + (elems * 14) as u64 > c.vram_budget {
        return None;
    }
    let nchunk_max = (h0 * w0).div_ceil(GN_ROWS) + 1;
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut r = Rec {
        c,
        calls: ZCalls::default(),
        x: zi::sbuf(c, (elems * 4) as u64, "h3cv_x"),
        h: zi::sbuf(c, (elems * 4) as u64, "h3cv_h"),
        s: zi::sbuf(c, (elems * 4) as u64, "h3cv_s"),
        xn: zi::sbuf(c, (elems * 2) as u64, "h3cv_xn"),
        dummy: zi::sbuf(c, 64, "h3cv_dummy"),
        stat: zi::sbuf(c, (2 * GROUPS * 4) as u64, "h3cv_stat"),
        part: zi::sbuf(c, (GROUPS * nchunk_max * 4) as u64, "h3cv_part"),
    };
    if let Some(e) = pollster::block_on(sc.pop()) {
        eprintln!("minimax-h3 wgpu: {e}");
        return None;
    }

    // The input image as f16 NHWC, its channels padded to 32.
    let (mut ch, mut hh, mut ww) = (c0.next_multiple_of(32), h0, w0);
    {
        let m0 = h0 * w0;
        let mut xin = vec![0u16; mp_of(m0) * ch];
        for cc in 0..c0 {
            for p in 0..m0 {
                xin[p * ch + cc] = cortiq_core::quant::f32_to_f16(x[cc * m0 + p]);
            }
        }
        c.queue.write_buffer(&r.xn, 0, bytemuck::cast_slice(&xin));
    }
    let mut cur = Cur::Half;
    let mut real = c0;
    // whether `x` holds a saved activation the stages must not overwrite
    let mut saved = false;
    for s in steps {
        let m = hh * ww;
        match s {
            ConvStep::GnSilu { w, b } => {
                match &cur {
                    Cur::Full => r.gn_silu(None, w, b, m, ch)?,
                    Cur::Raw(cv) => r.gn_silu(Some(cv.as_ref()), w, b, m, ch)?,
                    Cur::Half => return None,
                }
                cur = Cur::Half;
            }
            ConvStep::Conv { w, b, out_ch, in_ch, k, stride, .. } => {
                match &cur {
                    Cur::Half => {}
                    Cur::Full => r.cast(None, m, ch)?,
                    Cur::Raw(cv) => r.cast(Some(cv.as_ref()), m, ch)?,
                }
                let cv = conv_dev(c, w, b, *out_ch, *in_ch, *k)?;
                if cv.cin_p != ch {
                    return None;
                }
                let mode = match (*k, *stride) {
                    (1, _) => 0,
                    (_, 2) => 5,
                    _ => 4,
                };
                if *stride == 2 {
                    (hh, ww) = (hh / 2, ww / 2);
                }
                let out = r.h.clone();
                r.conv(&cv, mode, *k, hh, ww, &out)?;
                (ch, real) = (cv.cout_p, *out_ch);
                cur = Cur::Raw(cv);
            }
            ConvStep::Save => {
                match &cur {
                    Cur::Full => {}
                    Cur::Raw(cv) => {
                        let bias = cv.bias.clone();
                        r.combine(&bias, None, 2, mp_of(m) * ch, ch)?;
                    }
                    Cur::Half => return None,
                }
                cur = Cur::Full;
                saved = true;
            }
            ConvStep::AddSaved { via } => {
                // `x` is the saved activation, `h` the branch's last
                // convolution with its bias still to add
                let Cur::Raw(cv) = &cur else { return None };
                if !saved {
                    return None;
                }
                let cv = cv.clone();
                match via {
                    None => r.combine(&cv.bias, None, 0, mp_of(m) * ch, ch)?,
                    Some((w, b)) => {
                        // the saved activation through a 1×1 convolution
                        let ic = w.len() / b.len().max(1);
                        let sc = conv_dev(c, w, b, b.len(), ic, 1)?;
                        if sc.cout_p != ch {
                            return None;
                        }
                        r.cast(None, m, sc.cin_p)?;
                        let out = r.s.clone();
                        r.conv(&sc, 0, 1, hh, ww, &out)?;
                        r.combine(&cv.bias, Some(&sc.bias), 1, mp_of(m) * ch, ch)?;
                    }
                }
                cur = Cur::Full;
                saved = false;
            }
        }
    }

    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        r.calls.record(&mut pass, None);
    }
    c.queue.submit(Some(enc.finish()));
    let m = hh * ww;
    let (src, bias) = match &cur {
        Cur::Full => (r.x.clone(), None),
        Cur::Raw(cv) => (r.h.clone(), Some(cv.clone())),
        Cur::Half => return None,
    };
    let raw = zi::read_bytes(c, &src, (m * ch * 4) as u64);
    if let Some(e) = pollster::block_on(sc.pop()) {
        eprintln!("minimax-h3 wgpu: the convolution chain failed: {e}");
        return None;
    }
    let raw = raw?;
    let v: &[f32] = bytemuck::cast_slice(&raw);
    // the bias of a last convolution was never added on the card
    let last_b: Vec<f32> = match (&bias, steps.iter().rev().find_map(|s| match s {
        ConvStep::Conv { b, .. } => Some(*b),
        _ => None,
    })) {
        (Some(_), Some(b)) => b.to_vec(),
        _ => vec![0f32; real],
    };
    let mut out = vec![0f32; real * m];
    for cc in 0..real {
        let b = last_b.get(cc).copied().unwrap_or(0.0);
        for p in 0..m {
            out[cc * m + p] = v[p * ch + cc] + b;
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some((out, real, hh, ww))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chain's kernels are valid WGSL for naga.
    #[test]
    fn conv_kernels_validate() {
        use wgpu::naga;
        let check = |name: &str, src: &str| {
            let m = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&m)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        };
        check("h3_gn_part", GN_PART_SRC);
        for conv in [0, 4, 5] {
            let g = MmCfg { conv, ..zi::default_cfg(Epi::F32) };
            assert!(g.valid(), "{g:?}");
            check(&format!("conv {conv}"), &zi::mm_src(g));
        }
    }
}
