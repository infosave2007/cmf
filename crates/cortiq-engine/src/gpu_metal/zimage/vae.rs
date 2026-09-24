//! Resident Flux-VAE decoder on Metal (plan WP3, M6).
//!
//! NHWC activations: the residual stream in f32, every conv input in half.
//! Each 3×3 / 1×1 conv is an implicit GEMM (`zv_conv`, the DiT GEMM's
//! 64×64×32 tile, half weights `[oc][tap][ic]`, f32 accumulation, bias and
//! residual in the epilogue); the nearest-2× upsample is folded into the
//! gather. GroupNorm is two exact passes (sum, then centred sum of squares)
//! over per-block partials, fused with the affine, SiLU and the half cast.
//! The mid attention (one head, c = 512) runs as GEMMs: q (pre-scaled by
//! 1/√c), k, vᵀ, then per query chunk S = q·kᵀ (f32), row softmax → half P,
//! O = P·v, out projection + residual. One latent upload, one RGB readback.

use super::{buf_from, buf_zeroed, pipes, set_p, Pipes};
use crate::vae::{VaeChainArgs, VaeConvRef, VaeNormRef, VaeResnetRef};
use metal::{Buffer, ComputeCommandEncoderRef, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::Mutex;

use super::super::Ctx;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PVc {
    n: u32,
    rows: u32,
    k: u32,
    ic: u32,
    h: u32,
    w: u32,
    taps: u32,
    up: u32,
    epi: u32,
    ldy: u32,
    has_bias: u32,
    has_res: u32,
    mul: f32,
    ldx: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PGn {
    n: u32,
    c: u32,
    groups: u32,
    nblk: u32,
    pass: u32,
    eps: f32,
    silu: u32,
    half_out: u32,
}

/// One conv (or linear, taps = 0) with half weights [ocp][taps·icp] and
/// f32 bias [ocp] (zero-padded channels).
struct VConv {
    w: Buffer,
    b: Buffer,
    ocp: usize,
    icp: usize,
    taps: usize,
}

struct VNorm {
    w: Buffer,
    b: Buffer,
    groups: usize,
}

struct VRes {
    n1: VNorm,
    c1: VConv,
    n2: VNorm,
    c2: VConv,
    sc: Option<VConv>,
    cin: usize,
    cout: usize,
}

struct VAttn {
    norm: VNorm,
    q: VConv,
    k: VConv,
    v: VConv,
    o: VConv,
    c: usize,
}

struct VUp {
    res: Vec<VRes>,
    up: Option<VConv>,
}

struct VaeDev {
    key: u64,
    conv_in: VConv,
    mid1: VRes,
    attn: VAttn,
    mid2: VRes,
    ups: Vec<VUp>,
    norm_out: VNorm,
    conv_out: VConv,
    latent_c: usize,
}

struct VState(Option<VaeDev>);
unsafe impl Send for VState {}

static VSTATE: Mutex<VState> = Mutex::new(VState(None));

fn half_buf(c: &Ctx, v: &[f32]) -> Buffer {
    let h: Vec<u16> = v.iter().map(|&x| cortiq_core::quant::f32_to_f16(x)).collect();
    c._device.new_buffer_with_data(
        h.as_ptr() as *const c_void,
        (h.len().max(8) * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// conv weight [oc][ic][k][k] → half [ocp][k·k][icp]; `scale` multiplies
/// weight and bias (the attention's 1/√c for q).
fn conv_of(c: &Ctx, r: &VaeConvRef, scale: f32) -> Result<VConv, String> {
    let (oc, ic, k) = (r.oc, r.ic, r.k);
    if r.w.len() != oc * ic * k * k || r.b.len() != oc || (k != 1 && k != 3) {
        return Err(format!("vae conv shape oc {oc} ic {ic} k {k}"));
    }
    let taps = k * k;
    let ocp = oc.div_ceil(64) * 64;
    let icp = ic.div_ceil(32) * 32;
    let mut w = vec![0f32; ocp * taps * icp];
    for o in 0..oc {
        for i in 0..ic {
            for t in 0..taps {
                w[(o * taps + t) * icp + i] = r.w[(o * ic + i) * taps + t] * scale;
            }
        }
    }
    let mut b = vec![0f32; ocp];
    for o in 0..oc {
        b[o] = r.b[o] * scale;
    }
    Ok(VConv {
        w: half_buf(c, &w),
        b: buf_from(c, &b),
        ocp,
        icp,
        taps,
    })
}

/// A [c_out, c_in] linear as a plain-mode "conv" (taps 0).
fn lin_of(c: &Ctx, w: &[f32], b: &[f32], ch: usize, scale: f32) -> Result<VConv, String> {
    if w.len() != ch * ch || b.len() != ch || ch % 64 != 0 {
        return Err(format!("vae attention linear {ch}"));
    }
    let ws: Vec<f32> = w.iter().map(|v| v * scale).collect();
    let bs: Vec<f32> = b.iter().map(|v| v * scale).collect();
    Ok(VConv {
        w: half_buf(c, &ws),
        b: buf_from(c, &bs),
        ocp: ch,
        icp: ch,
        taps: 0,
    })
}

fn norm_of(c: &Ctx, r: &VaeNormRef) -> VNorm {
    VNorm {
        w: buf_from(c, r.w),
        b: buf_from(c, r.b),
        groups: r.groups,
    }
}

fn res_of(c: &Ctx, r: &VaeResnetRef) -> Result<VRes, String> {
    let (cin, cout) = (r.conv1.ic, r.conv1.oc);
    if cin % 32 != 0 || cout % 64 != 0 || cin % (4 * r.norm1.groups) != 0 || cout % (4 * r.norm2.groups) != 0 {
        return Err(format!("vae resnet {cin}->{cout}"));
    }
    Ok(VRes {
        n1: norm_of(c, &r.norm1),
        c1: conv_of(c, &r.conv1, 1.0)?,
        n2: norm_of(c, &r.norm2),
        c2: conv_of(c, &r.conv2, 1.0)?,
        sc: match &r.shortcut {
            Some(s) => Some(conv_of(c, s, 1.0)?),
            None => None,
        },
        cin,
        cout,
    })
}

fn build(c: &Ctx, a: &VaeChainArgs) -> Result<VaeDev, String> {
    let at = &a.mid_attn;
    let ch = at.c;
    let scale = 1.0 / (ch as f32).sqrt();
    Ok(VaeDev {
        key: a.key,
        conv_in: conv_of(c, &a.conv_in, 1.0)?,
        mid1: res_of(c, &a.mid_res1)?,
        attn: VAttn {
            norm: norm_of(c, &at.norm),
            q: lin_of(c, at.q.0, at.q.1, ch, scale)?,
            k: lin_of(c, at.k.0, at.k.1, ch, 1.0)?,
            v: lin_of(c, at.v.0, at.v.1, ch, 1.0)?,
            o: lin_of(c, at.out.0, at.out.1, ch, 1.0)?,
            c: ch,
        },
        mid2: res_of(c, &a.mid_res2)?,
        ups: a
            .ups
            .iter()
            .map(|u| {
                Ok(VUp {
                    res: u.resnets.iter().map(|r| res_of(c, r)).collect::<Result<Vec<_>, String>>()?,
                    up: match &u.upsample {
                        Some(s) => Some(conv_of(c, s, 1.0)?),
                        None => None,
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        norm_out: norm_of(c, &a.norm_out),
        conv_out: conv_of(c, &a.conv_out, 1.0)?,
        latent_c: a.latent_channels,
    })
}

fn ensure<'s>(st: &'s mut VState, c: &Ctx, a: &VaeChainArgs) -> Result<&'s VaeDev, String> {
    if st.0.as_ref().is_none_or(|d| d.key != a.key) {
        st.0 = None;
        st.0 = Some(build(c, a)?);
    }
    Ok(st.0.as_ref().unwrap())
}

pub(crate) fn prewarm(a: &VaeChainArgs) -> bool {
    let Some(c) = super::super::ctx() else { return false };
    if pipes(c).is_none() {
        return false;
    }
    let mut st = VSTATE.lock().unwrap();
    match ensure(&mut st, c, a) {
        Ok(_) => true,
        Err(e) => super::decline(&e),
    }
}

pub(crate) fn release() {
    if let Ok(mut st) = VSTATE.lock() {
        st.0 = None;
    }
}

/// Encoder helper: one encoder per command buffer, a new buffer at every
/// `cut` (committed without a wait).
struct VRec<'a> {
    c: &'a Ctx,
    p: &'static Pipes,
    done: Vec<metal::CommandBuffer>,
    cur: Option<(metal::CommandBuffer, metal::ComputeCommandEncoder)>,
    part: Buffer,
    stat: Buffer,
}

impl VRec<'_> {
    fn enc(&mut self) -> &ComputeCommandEncoderRef {
        if self.cur.is_none() {
            let cmd = self.c.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder().to_owned();
            self.cur = Some((cmd, enc));
        }
        &self.cur.as_ref().unwrap().1
    }
    fn cut(&mut self) {
        if let Some((cmd, enc)) = self.cur.take() {
            enc.end_encoding();
            cmd.commit();
            self.done.push(cmd);
        }
    }
    fn finish(&mut self) -> bool {
        self.cut();
        let mut ok = true;
        for cmd in &self.done {
            cmd.wait_until_completed();
            ok &= cmd.status() == metal::MTLCommandBufferStatus::Completed;
        }
        ok
    }

    /// Y = conv(X) (+bias) (+R). `hw` = output (H, W); `up` = the source is
    /// (H/2, W/2). Plain mode (taps 0): X [n][ldx], `n` rows.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        cv: &VConv,
        x: (&Buffer, u64),
        y: (&Buffer, u64),
        res: Option<(&Buffer, u64)>,
        n: usize,
        hw: (usize, usize),
        up: bool,
        epi: u32,
        ldy: usize,
        ldx: usize,
        k_plain: usize,
    ) {
        let k = if cv.taps == 0 { k_plain } else { cv.taps * cv.icp };
        let pv = PVc {
            n: n as u32,
            rows: cv.ocp as u32,
            k: k as u32,
            ic: cv.icp as u32,
            h: hw.0 as u32,
            w: hw.1 as u32,
            taps: cv.taps as u32,
            up: up as u32,
            epi,
            ldy: ldy as u32,
            has_bias: 1,
            has_res: res.is_some() as u32,
            mul: 1.0,
            ldx: ldx as u32,
            ..Default::default()
        };
        let p = self.p;
        let enc = self.enc();
        enc.set_compute_pipeline_state(&p.vconv);
        enc.set_buffer(0, Some(&cv.w), 0);
        enc.set_buffer(1, Some(&cv.b), 0);
        enc.set_buffer(2, Some(x.0), x.1);
        enc.set_buffer(3, Some(y.0), y.1);
        let r = res.unwrap_or(y);
        enc.set_buffer(4, Some(r.0), r.1);
        set_p(enc, 5, &pv);
        enc.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(64) as u64, (cv.ocp / 64) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// Plain GEMM with an explicit weight buffer (the attention's k / vᵀ
    /// as "weights"): Y[n][rows] = X[n][K] · W[rows][K]ᵀ, no bias.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&mut self, w: (&Buffer, u64), rows: usize, k: usize, x: (&Buffer, u64), ldx: usize, y: (&Buffer, u64), ldy: usize, n: usize, epi: u32) {
        let pv = PVc {
            n: n as u32,
            rows: rows as u32,
            k: k as u32,
            ic: k as u32,
            epi,
            ldy: ldy as u32,
            mul: 1.0,
            ldx: ldx as u32,
            ..Default::default()
        };
        let p = self.p;
        let enc = self.enc();
        enc.set_compute_pipeline_state(&p.vconv);
        enc.set_buffer(0, Some(w.0), w.1);
        enc.set_buffer(1, Some(y.0), y.1);
        enc.set_buffer(2, Some(x.0), x.1);
        enc.set_buffer(3, Some(y.0), y.1);
        enc.set_buffer(4, Some(y.0), y.1);
        set_p(enc, 5, &pv);
        enc.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(64) as u64, (rows / 64) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// GroupNorm (+SiLU) of x [n][ch] f32 → y (half or f32).
    fn gn(&mut self, nm: &VNorm, x: &Buffer, y: &Buffer, n: usize, ch: usize, silu: bool, half_out: bool) {
        let nblk = n.div_ceil(1024).clamp(1, 256);
        let mut pg = PGn {
            n: n as u32,
            c: ch as u32,
            groups: nm.groups as u32,
            nblk: nblk as u32,
            pass: 0,
            eps: 1e-6,
            silu: silu as u32,
            half_out: half_out as u32,
        };
        let p = self.p;
        let (part, stat) = (self.part.clone(), self.stat.clone());
        let enc = self.enc();
        for pass in 0..2u32 {
            pg.pass = pass;
            enc.set_compute_pipeline_state(&p.vgnpart);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&part), 0);
            enc.set_buffer(2, Some(&stat), 0);
            set_p(enc, 3, &pg);
            enc.dispatch_thread_groups(MTLSize::new(nm.groups as u64, nblk as u64, 1), MTLSize::new(256, 1, 1));
            enc.set_compute_pipeline_state(&p.vgnfin);
            enc.set_buffer(0, Some(&part), 0);
            enc.set_buffer(1, Some(&stat), 0);
            set_p(enc, 2, &pg);
            enc.dispatch_thread_groups(MTLSize::new(nm.groups as u64, 1, 1), MTLSize::new(32, 1, 1));
        }
        enc.set_compute_pipeline_state(&p.vgnapply);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_buffer(2, Some(&stat), 0);
        enc.set_buffer(3, Some(&nm.w), 0);
        enc.set_buffer(4, Some(&nm.b), 0);
        set_p(enc, 5, &pg);
        let n4 = (n * ch / 4) as u64;
        enc.dispatch_thread_groups(MTLSize::new(n4.div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    }

    fn cvt(&mut self, x: &Buffer, y: &Buffer, count: usize) {
        let p = self.p;
        let n4 = (count / 4) as u32;
        let enc = self.enc();
        enc.set_compute_pipeline_state(&p.vcvt);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(y), 0);
        set_p(enc, 2, &n4);
        enc.dispatch_thread_groups(MTLSize::new((n4 as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    }
}

struct Bufs {
    xa: Buffer,
    t: Buffer,
    hh: Buffer,
}

/// One resnet on xa [n][cin] (f32) → xa [n][cout].
fn resnet(r: &mut VRec, b: &Bufs, rs: &VRes, n: usize, hw: (usize, usize)) {
    r.gn(&rs.n1, &b.xa, &b.hh, n, rs.cin, true, true);
    r.conv(&rs.c1, (&b.hh, 0), (&b.t, 0), None, n, hw, false, 0, rs.cout, 0, 0);
    if let Some(sc) = &rs.sc {
        r.cvt(&b.xa, &b.hh, n * rs.cin);
        r.conv(sc, (&b.hh, 0), (&b.xa, 0), None, n, hw, false, 0, rs.cout, 0, 0);
    }
    r.gn(&rs.n2, &b.t, &b.hh, n, rs.cout, true, true);
    r.conv(&rs.c2, (&b.hh, 0), (&b.xa, 0), Some((&b.xa, 0)), n, hw, false, 0, rs.cout, 0, 0);
}

/// The f16-plane arm of the codec A/B (plan M5c): the same 64×64×32 tile
/// with half weights (`zv_conv`, plain mode) on y[n, rows] = x[n, k]·Wᵀ.
/// Returns (min ms, median ms) of GPU time.
pub(crate) fn bench_plane(rows: usize, k: usize, n: usize, reps: usize) -> Option<(f64, f64)> {
    let c = super::super::ctx()?;
    let p = pipes(c)?;
    let w: Vec<u16> = (0..rows * k).map(|i| cortiq_core::quant::f32_to_f16(((i * 7) % 255) as f32 - 127.0)).collect();
    let x: Vec<u16> = (0..(n.div_ceil(64) * 64 + 64) * k).map(|i| cortiq_core::quant::f32_to_f16(((i * 5) % 17) as f32 * 0.01)).collect();
    let wb = c._device.new_buffer_with_data(w.as_ptr() as *const c_void, (w.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
    let xb = c._device.new_buffer_with_data(x.as_ptr() as *const c_void, (x.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
    let yb = buf_zeroed(c, n * rows * 4);
    let pv = PVc {
        n: n as u32,
        rows: rows as u32,
        k: k as u32,
        ic: k as u32,
        ldy: rows as u32,
        mul: 1.0,
        ldx: k as u32,
        ..Default::default()
    };
    let mut t = Vec::new();
    for _ in 0..reps + 1 {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&p.vconv);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&yb), 0);
        enc.set_buffer(2, Some(&xb), 0);
        enc.set_buffer(3, Some(&yb), 0);
        enc.set_buffer(4, Some(&yb), 0);
        set_p(enc, 5, &pv);
        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(64) as u64, (rows / 64) as u64, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        t.push(super::super::cmd_gpu_ms(cmd));
    }
    t.remove(0);
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((t[0], t[t.len() / 2]))
}

/// Resident Flux-VAE decoder; `z` is already de-normalised, [lc, h, w];
/// `out` [3, 8h, 8w].
pub(crate) fn decode(a: &VaeChainArgs, z: &[f32], h: usize, w: usize, out: &mut [f32]) -> bool {
    if std::env::var("CMF_ZI_VAE").as_deref() == Ok("0") {
        return false;
    }
    let Some(c) = super::super::ctx() else { return false };
    let Some(p) = pipes(c) else { return false };
    let mut st = VSTATE.lock().unwrap();
    let d = match ensure(&mut st, c, a) {
        Ok(d) => d,
        Err(e) => return super::decline(&e),
    };
    let lc = d.latent_c;
    let n0 = h * w;
    if z.len() != lc * n0 || out.len() != 3 * 64 * n0 || d.attn.c % 64 != 0 || n0 % 4 != 0 {
        return false;
    }
    // attention keys padded to the GEMM tile (zero keys, masked in the softmax)
    let n0p = n0.div_ceil(64) * 64;
    // largest n·C of any stage
    let mut cmax = 0usize;
    {
        let (mut hh, mut ww, mut ch) = (h, w, d.mid1.cout);
        cmax = cmax.max(hh * ww * ch.max(d.conv_in.icp));
        for u in &d.ups {
            for r in &u.res {
                cmax = cmax.max(hh * ww * r.cin.max(r.cout));
                ch = r.cout;
            }
            if u.up.is_some() {
                hh *= 2;
                ww *= 2;
                cmax = cmax.max(hh * ww * ch);
            }
        }
        cmax = cmax.max(hh * ww * 64);
    }
    // Metal returns nil past maxBufferLength (13.6 GB on a 24 GB M4) and the
    // uploads below would write through a null pointer: decline instead.
    if (cmax * 4) as u64 > c._device.max_buffer_length() {
        return super::decline(&format!(
            "the VAE activation buffer ({:.1} GB) exceeds the device's maxBufferLength",
            (cmax * 4) as f64 / 1e9
        ));
    }
    let bufs = Bufs {
        xa: buf_zeroed(c, cmax * 4),
        t: buf_zeroed(c, cmax * 4),
        hh: buf_zeroed(c, cmax * 2),
    };
    // latent → NHWC half, channels padded to the conv's icp
    {
        let icp = d.conv_in.icp;
        let mut zh = vec![0u16; n0 * icp];
        for ch in 0..lc {
            for px in 0..n0 {
                zh[px * icp + ch] = cortiq_core::quant::f32_to_f16(z[ch * n0 + px]);
            }
        }
        unsafe {
            std::ptr::copy_nonoverlapping(zh.as_ptr(), bufs.hh.contents() as *mut u16, zh.len());
        }
    }
    let ac = d.attn.c;
    let chunk = n0.min(env_chunk());
    let (qb, kb, vt, ob) = (
        buf_zeroed(c, n0p * ac * 2),
        buf_zeroed(c, n0p * ac * 2),
        buf_zeroed(c, n0p * ac * 2),
        buf_zeroed(c, n0p * ac * 2),
    );
    let sb = buf_zeroed(c, chunk * n0p * 4);
    let pb = buf_zeroed(c, chunk * n0p * 2);
    let mut r = VRec {
        c,
        p,
        done: Vec::new(),
        cur: None,
        part: buf_zeroed(c, 64 * 256 * 4),
        stat: buf_zeroed(c, 64 * 2 * 4),
    };
    let t0 = std::time::Instant::now();
    let b = &bufs;
    // conv_in
    r.conv(&d.conv_in, (&b.hh, 0), (&b.xa, 0), None, n0, (h, w), false, 0, d.conv_in.ocp, 0, 0);
    resnet(&mut r, b, &d.mid1, n0, (h, w));
    r.cut();
    // mid attention
    {
        let at = &d.attn;
        r.gn(&at.norm, &b.xa, &b.hh, n0, ac, false, true);
        r.conv(&at.q, (&b.hh, 0), (&qb, 0), None, n0, (0, 0), false, 1, ac, ac, ac);
        r.conv(&at.k, (&b.hh, 0), (&kb, 0), None, n0, (0, 0), false, 1, ac, ac, ac);
        r.conv(&at.v, (&b.hh, 0), (&vt, 0), None, n0, (0, 0), false, 2, n0p, ac, ac);
        let mut q0 = 0;
        while q0 < n0 {
            let m = chunk.min(n0 - q0);
            r.gemm((&kb, 0), n0p, ac, (&qb, (q0 * ac * 2) as u64), ac, (&sb, 0), n0p, m, 0);
            {
                let pp = r.p;
                let (n0u, ldu) = (n0 as u32, n0p as u32);
                let enc = r.enc();
                enc.set_compute_pipeline_state(&pp.vsoftmax);
                enc.set_buffer(0, Some(&sb), 0);
                enc.set_buffer(1, Some(&pb), 0);
                set_p(enc, 2, &n0u);
                set_p(enc, 3, &ldu);
                enc.dispatch_thread_groups(MTLSize::new(m as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            r.gemm((&vt, 0), ac, n0p, (&pb, 0), n0p, (&ob, (q0 * ac * 2) as u64), ac, m, 1);
            q0 += m;
        }
        r.conv(&at.o, (&ob, 0), (&b.xa, 0), Some((&b.xa, 0)), n0, (0, 0), false, 0, ac, ac, ac);
    }
    resnet(&mut r, b, &d.mid2, n0, (h, w));
    r.cut();
    let (mut hh, mut ww) = (h, w);
    let mut ch = d.mid2.cout;
    let mut bufs_swapped = false;
    let (mut xa, mut tt) = (b.xa.clone(), b.t.clone());
    for u in &d.ups {
        for rs in &u.res {
            let bb = Bufs { xa: xa.clone(), t: tt.clone(), hh: b.hh.clone() };
            resnet(&mut r, &bb, rs, hh * ww, (hh, ww));
            ch = rs.cout;
            r.cut();
        }
        if let Some(upc) = &u.up {
            r.cvt(&xa, &b.hh, hh * ww * ch);
            hh *= 2;
            ww *= 2;
            r.conv(upc, (&b.hh, 0), (&tt, 0), None, hh * ww, (hh, ww), true, 0, upc.ocp, 0, 0);
            std::mem::swap(&mut xa, &mut tt);
            bufs_swapped = !bufs_swapped;
            r.cut();
        }
    }
    let n = hh * ww;
    r.gn(&d.norm_out, &xa, &b.hh, n, ch, true, true);
    r.conv(&d.conv_out, (&b.hh, 0), (&tt, 0), None, n, (hh, ww), false, 0, 64, 0, 0);
    let rgb = buf_zeroed(c, 3 * n * 4);
    {
        let pp = r.p;
        let nu = n as u32;
        let enc = r.enc();
        enc.set_compute_pipeline_state(&pp.vrgb);
        enc.set_buffer(0, Some(&tt), 0);
        enc.set_buffer(1, Some(&rgb), 0);
        set_p(enc, 2, &nu);
        enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    }
    let _ = bufs_swapped;
    if !r.finish() {
        return super::decline("a Metal command buffer failed (VAE)");
    }
    if std::env::var("CMF_ZIMAGE_PROF").is_ok_and(|v| v != "0") {
        let gpu: f64 = r.done.iter().map(|cmd| super::super::cmd_gpu_ms(cmd)).sum();
        eprintln!(
            "zimage metal vae: {:.3}s wall, {:.1} ms gpu ({}x{})",
            t0.elapsed().as_secs_f64(),
            gpu,
            hh,
            ww
        );
    }
    let v = unsafe { std::slice::from_raw_parts(rgb.contents() as *const f32, 3 * n) };
    if v.iter().any(|x| !x.is_finite()) {
        return super::decline("the VAE output is not finite");
    }
    out.copy_from_slice(v);
    true
}

fn env_chunk() -> usize {
    std::env::var("CMF_ZI_VAE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(4096).max(64) / 64 * 64
}
