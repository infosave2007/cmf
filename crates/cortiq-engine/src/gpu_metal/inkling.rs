//! Inkling-2-Mini on Metal, the twin of the CUDA arm (`gpu_cuda/hyb/inkling.rs`;
//! `inkling` has the semantics). Weights stay in the file mapping (the
//! backend's arena): the dense matrices through the backend's q4tp / q8_2f
//! matvecs, the routed experts through this arm's kernels, which pick them on
//! the GPU (router, top-k, every pick's gate/up with Inkling's softcapped
//! SwiGLU, the weighted downs) — nothing returns to the host inside a token.
//! Every one of the 48 layer applications keeps its own keys and values (f16;
//! a local layer's in a ring of its window, a full layer's growing) and its
//! four convolution rings. A call (a token, or a prompt chunk of up to 16
//! rows) is one command buffer.

use super::{Ctx, WeightArena, ctx, file_buffer, set_bytes_t};
use crate::inkling::{Cfg, Globals, Layer};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MSL: &str = include_str!("inkling_msl.metal");
/// Rows of a call.
const NMAX: usize = 16;
/// Block states a row holds.
const MAXB: usize = 4;

/// `IP` of the shaders.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IP {
    h: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
    dr: u32,
    kc: u32,
    n: u32,
    pos0: u32,
    nb: u32,
    bstride: u32,
    ostride: u32,
    ring: u32,
    e: u32,
    nq: u32,
    full: u32,
    rows: u32,
    r: u32,
    inter: u32,
    v: u32,
    bslot: u32,
    topk: u32,
    gpr: u32,
    ostride2: u32,
    win: u32,
    eps: f32,
    qclip: f32,
    rclip: f32,
    alpha: f32,
    floor_: f32,
    scale: f32,
    cap: f32,
    ucap: f32,
    rscale: f32,
    pad1: f32,
    pad2: f32,
    pad3: f32,
}

struct Pipes {
    mix: ComputePipelineState,
    rms: ComputePipelineState,
    gate_down: ComputePipelineState,
    gate_out: ComputePipelineState,
    fmv: ComputePipelineState,
    qr: ComputePipelineState,
    kv: ComputePipelineState,
    attn: ComputePipelineState,
    ogate: ComputePipelineState,
    post: ComputePipelineState,
    softcap: ComputePipelineState,
    copy: ComputePipelineState,
    router: ComputePipelineState,
    select: ComputePipelineState,
    moe_gu: ComputePipelineState,
    moe_down: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c._device.new_library_with_source(MSL, &opts).map_err(|e| format!("inkling MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            c._device.new_compute_pipeline_state_with_function(&f).map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            mix: pso("ink_mix")?,
            rms: pso("ink_rms")?,
            gate_down: pso("ink_gate_down")?,
            gate_out: pso("ink_gate_out")?,
            fmv: pso("ink_fmv")?,
            qr: pso("ink_qr")?,
            kv: pso("ink_kv")?,
            attn: pso("ink_attn")?,
            ogate: pso("ink_ogate")?,
            post: pso("ink_post")?,
            softcap: pso("ink_softcap")?,
            copy: pso("ink_copy")?,
            router: pso("ink_router")?,
            select: pso("ink_select")?,
            moe_gu: pso("ink_moe_gu")?,
            moe_down: pso("ink_moe_down")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| tracing::warn!("inkling2_loop on Metal: {e}"));
            None
        }
    }
}

#[derive(Clone)]
enum Mat {
    Q4 { abs: usize, rows: usize, cols: usize },
    Q8 { abs: usize, rows: usize, cols: usize, rs: Buffer, col: Buffer },
}

impl Mat {
    fn rows(&self) -> usize {
        match self {
            Mat::Q4 { rows, .. } | Mat::Q8 { rows, .. } => *rows,
        }
    }
}

fn shared(c: &Ctx, bytes: usize) -> Buffer {
    c._device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
}

fn up<T: Copy>(c: &Ctx, v: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(v).max(16);
    if v.is_empty() {
        return shared(c, 16);
    }
    c._device.new_buffer_with_data(v.as_ptr() as *const std::ffi::c_void, bytes as u64, MTLResourceOptions::StorageModeShared)
}

fn mat(c: &Ctx, model: &Arc<cortiq_core::CmfModel>, t: &crate::qtensor::QTensor) -> Option<Mat> {
    let (rows, cols) = (t.rows(), t.cols());
    if let Some((_, idx)) = t.q4tp_mapped() {
        let abs = model.entry_abs_offset(&model.tensors[idx])?;
        return Some(Mat::Q4 { abs, rows, cols });
    }
    let (bytes, rs, col) = t.q8_2f_codes()?;
    let base = model.primary_bytes().as_ptr() as usize;
    let abs = (bytes.as_ptr() as usize).checked_sub(base)?;
    Some(Mat::Q8 { abs, rows, cols, rs: up(c, rs), col: up(c, col) })
}

struct LayerM {
    sliding: bool,
    boundary: bool,
    aq: Option<(Buffer, usize)>,
    mq: Buffer,
    norms: [Buffer; 3],
    post_norm: Buffer,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    gate: Mat,
    rel: Mat,
    sink: Buffer,
    q_scale: Buffer,
    rel_scale: Buffer,
    rel_table: Buffer,
    extent: usize,
    tk: Buffer,
    tv: Buffer,
    dk: Option<Buffer>,
    dv: Option<Buffer>,
    tpa: Buffer,
    tpm: Buffer,
    router: Buffer,
    bias: Buffer,
    /// Absolute offsets of every expert's gate, up and down `[E][3]`.
    bases: Buffer,
    e: usize,
    inter: usize,
    gates: Option<[(Buffer, Buffer, Buffer, usize); 4]>,
}

struct SlotM {
    k: Buffer,
    v: Buffer,
    cap: usize,
    ring: usize,
    rings: [Buffer; 4],
}

struct Dev {
    fbuf: Arc<WeightArena>,
    wstride: u64,
    layers: Vec<LayerM>,
    slots: Vec<SlotM>,
    head: Mat,
    msg_q: Buffer,
    final_q: Buffer,
    final_norm: Buffer,
    h: [Buffer; 2],
    blk: Buffer,
    pre_h: Buffer,
    bank: Buffer,
    bank_out: Buffer,
    msg: Buffer,
    in3: Buffer,
    gin: Buffer,
    gd: Buffer,
    x: [Buffer; 3],
    yq: Buffer,
    yg: Buffer,
    yr: Buffer,
    yk: Buffer,
    yv: Buffer,
    sk: Buffer,
    dkb: Buffer,
    dvb: Buffer,
    att: Buffer,
    ya: Buffer,
    r: Buffer,
    m: Buffer,
    mg: Buffer,
    xm: Buffer,
    lg: Buffer,
    pick: Buffer,
    wt: Buffer,
    act: Buffer,
    y: Buffer,
    fin: Buffer,
    xf: Buffer,
    logits: Buffer,
    dummy: Buffer,
}
unsafe impl Send for Dev {}

static DEVICES: Mutex<Option<HashMap<u64, Dev>>> = Mutex::new(None);
static DECLINED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn admits(layers: &[Layer], cfg: &Cfg) -> bool {
    cfg.hd == 128
        && cfg.nh % cfg.nkv == 0
        && cfg.dim % 256 == 0
        && cfg.d_rel <= 32
        && cfg.window <= 1024
        && cfg.rel_extent <= 1024
        && cfg.top_k <= 16
        && cfg.layers / cfg.block.max(1) <= MAXB
        && layers.iter().all(|l| l.moe.bias.len() <= 128)
        && layers.iter().all(|l| l.moe.experts.iter().all(|[g, u, d]| g.q4tp_mapped().is_some() && u.q4tp_mapped().is_some() && d.q4tp_mapped().is_some()) && l.attn.k_conv.kernel <= 8)
}

fn build(c: &Ctx, g: &Globals, layers: &[Layer], cfg: &Cfg) -> Result<Dev, String> {
    pipes(c).ok_or("pipelines")?;
    let model = &g.model;
    let (fbuf, _) = file_buffer(c, model).ok_or("no file arena")?;
    // the experts' kernels address the mapping by absolute offsets: one window,
    // or up to four `stride` apart
    if !fbuf.cuts.is_empty() || fbuf.bufs.len() > 4 {
        return Err(format!("the file's {} windows are not the experts' (at most four, evenly spaced)", fbuf.bufs.len()));
    }
    let wstride = if fbuf.bufs.len() > 1 { fbuf.stride as u64 } else { 0 };
    let (hidden, nh, nkv, hd, dr) = (cfg.dim, cfg.nh, cfg.nkv, cfg.hd, cfg.d_rel);
    let mut dl = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter().enumerate() {
        let a = &l.attn;
        let mm = |t: &crate::qtensor::QTensor| mat(c, model, t).ok_or_else(|| format!("layer {li}: a matrix outside q4tp / q8_2f"));
        let mut bases: Vec<u64> = Vec::with_capacity(l.moe.experts.len() * 3);
        for x in &l.moe.experts {
            for t in x {
                let (_, idx) = t.q4tp_mapped().ok_or("expert codec")?;
                bases.push(model.entry_abs_offset(&model.tensors[idx]).ok_or("expert offset")? as u64);
            }
        }
        let gates = match &l.gates {
            Some(gs) => {
                let mk = |x: &crate::inkling::Gate| (up(c, &x.down), up(c, &x.forget), up(c, &x.gain), x.down.len() / (2 * hidden));
                Some([mk(&gs[0]), mk(&gs[1]), mk(&gs[2]), mk(&gs[3])])
            }
            None => None,
        };
        dl.push(LayerM {
            sliding: l.sliding,
            boundary: li % cfg.block == 0,
            aq: l.attn_query.as_ref().map(|q| (up(c, q), q.len() / hidden)),
            mq: up(c, &l.mlp_query),
            norms: [up(c, &l.in_norm[0]), up(c, &l.in_norm[1]), up(c, &l.in_norm[2])],
            post_norm: up(c, &l.post_norm),
            q: mm(&a.q)?,
            k: mm(&a.k)?,
            v: mm(&a.v)?,
            o: mm(&a.o)?,
            gate: mm(&a.gate)?,
            rel: mm(&a.rel)?,
            sink: up(c, &a.sink),
            q_scale: up(c, &a.q_scale),
            rel_scale: up(c, &a.rel_scale),
            rel_table: up(c, &a.rel_table),
            extent: a.rel_table.len() / dr,
            tk: up(c, &a.k_conv.taps),
            tv: up(c, &a.v_conv.taps),
            dk: a.k_conv.proj.as_ref().map(|p| up(c, p)),
            dv: a.v_conv.proj.as_ref().map(|p| up(c, p)),
            tpa: up(c, &l.post_attn_conv.taps),
            tpm: up(c, &l.post_mlp_conv.taps),
            router: up(c, &l.moe.router),
            bias: up(c, &l.moe.bias),
            bases: up(c, &bases),
            e: l.moe.bias.len(),
            inter: l.moe.experts.first().map_or(0, |x| x[0].rows()),
            gates,
        });
    }
    let head = mat(c, model, &g.lm_head).ok_or("head codec")?;
    let kvs = nkv * hd;
    let mut slots: Vec<Option<SlotM>> = (0..cfg.slots()).map(|_| None).collect();
    for (li, pass) in cfg.plan() {
        // a local slot's ring: its window and a call's rows (the call writes
        // every row's key before any row reads)
        let ring = if layers[li].sliding { cfg.window + NMAX } else { 0 };
        let cap = if ring > 0 { ring } else { 4096 };
        slots[cfg.slot(li, pass)] = Some(SlotM {
            k: shared(c, cap * kvs * 2),
            v: shared(c, cap * kvs * 2),
            cap,
            ring,
            rings: [shared(c, 8 * kvs * 4), shared(c, 8 * kvs * 4), shared(c, 8 * hidden * 4), shared(c, 8 * hidden * 4)],
        });
    }
    let slots: Vec<SlotM> = slots.into_iter().map(|s| s.ok_or("a slot without its application")).collect::<Result<_, _>>()?;
    let f4 = |n: usize| shared(c, n * 4);
    let rows = NMAX;
    let (qs, rs) = (nh * hd, nh * dr);
    let e = dl.first().map_or(0, |l| l.e);
    let inter = dl.first().map_or(0, |l| l.inter);
    let vocab = head.rows();
    Ok(Dev {
        fbuf,
        wstride,
        layers: dl,
        slots,
        head,
        msg_q: up(c, &g.message_query),
        final_q: up(c, &g.final_query),
        final_norm: up(c, &g.norm),
        h: [f4(rows * hidden), f4(rows * hidden)],
        blk: f4(rows * MAXB * hidden),
        pre_h: f4(rows * hidden),
        bank: f4(rows * MAXB * hidden),
        bank_out: f4(rows * hidden),
        msg: f4(4 * rows * hidden),
        in3: f4(3 * rows * hidden),
        gin: f4(3 * rows * hidden),
        gd: f4(rows * 256),
        x: [f4(rows * hidden), f4(rows * hidden), f4(rows * hidden)],
        yq: f4(rows * qs),
        yg: f4(rows * qs),
        yr: f4(rows * rs),
        yk: f4(rows * kvs),
        yv: f4(rows * kvs),
        sk: f4(rows * nh),
        dkb: f4(rows * nkv * 8),
        dvb: f4(rows * nkv * 8),
        att: f4(rows * qs),
        ya: f4(rows * hidden),
        r: f4(rows * hidden),
        m: f4(rows * hidden),
        mg: f4(rows * hidden),
        xm: f4(rows * hidden),
        lg: f4(rows * e),
        pick: shared(c, rows * cfg.top_k * 4),
        wt: f4(rows * cfg.top_k),
        act: f4(rows * cfg.top_k * inter),
        y: f4(rows * hidden),
        fin: f4(rows * hidden),
        xf: f4(rows * hidden),
        logits: f4(rows * vocab),
        dummy: shared(c, 64),
    })
}

impl Dev {
    /// Room for positions `..need` in every full slot (shared memory: a copy).
    fn ensure(&mut self, c: &Ctx, need: usize, kvs: usize) {
        for s in self.slots.iter_mut().filter(|s| s.ring == 0 && s.cap < need) {
            let cap = need.max(s.cap * 2).next_power_of_two();
            let (k, v) = (shared(c, cap * kvs * 2), shared(c, cap * kvs * 2));
            let old = s.cap * kvs * 2;
            unsafe {
                std::ptr::copy_nonoverlapping(s.k.contents() as *const u8, k.contents() as *mut u8, old);
                std::ptr::copy_nonoverlapping(s.v.contents() as *const u8, v.contents() as *mut u8, old);
            }
            *s = SlotM { k, v, cap, ring: 0, rings: std::mem::replace(&mut s.rings, [shared(c, 16), shared(c, 16), shared(c, 16), shared(c, 16)]) };
        }
    }

    fn zero_rings(&self) {
        for s in &self.slots {
            for r in &s.rings {
                unsafe { std::ptr::write_bytes(r.contents() as *mut u8, 0, r.length() as usize) };
            }
        }
    }
}

/// `true` when this call ran on the GPU. `ids` from position `pos0`; the last
/// row's logits (`all`: every row's) when `want`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(id: u64, g: &Globals, layers: &[Layer], cfg: &Cfg, ids: &[u32], pos0: usize, want: bool, all: bool, logits: &mut Vec<f32>) -> bool {
    let Some(c) = ctx() else { return false };
    if std::env::var("CMF_INKLING_METAL").as_deref() == Ok("0") || !admits(layers, cfg) {
        return false;
    }
    if DECLINED.lock().unwrap().contains(&id) {
        return false;
    }
    let mut guard = DEVICES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if !map.contains_key(&id) {
        // one sequence at a time: a new one takes the old one's buffers' place
        let old = map.drain().next();
        let built = match old {
            Some((_, d)) => Ok(d),
            None => {
                let t0 = std::time::Instant::now();
                let r = build(c, g, layers, cfg);
                if r.is_ok() {
                    tracing::info!("inkling2_loop on Metal: built in {:.1} s", t0.elapsed().as_secs_f64());
                }
                r
            }
        };
        match built {
            Ok(d) => {
                map.insert(id, d);
            }
            Err(e) => {
                tracing::warn!("inkling2_loop on Metal declined: {e}; the host runs this sequence");
                DECLINED.lock().unwrap().push(id);
                return false;
            }
        }
    }
    let d = map.get_mut(&id).expect("device");
    if pos0 == 0 {
        d.zero_rings();
    }
    d.ensure(c, pos0 + ids.len(), cfg.nkv * cfg.hd);
    if all {
        logits.clear();
    }
    let hidden = cfg.dim;
    let vocab = d.head.rows();
    let p = pipes(c).expect("pipelines");
    let mut at = 0usize;
    while at < ids.len() {
        let n = (ids.len() - at).min(NMAX);
        let last = at + n == ids.len();
        let lrows = if all { n } else if want && last { 1 } else { 0 };
        let mut e = vec![0f32; hidden];
        let hp = d.h[0].contents() as *mut f32;
        for (i, &tid) in ids[at..at + n].iter().enumerate() {
            e.fill(0.0);
            if (tid as usize) < g.embed.rows() {
                g.embed.row_f32(&g.model, tid as usize, &mut e);
            }
            let ms = e.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
            let r = (1.0 / (ms + cfg.eps).sqrt()) as f32;
            for (j, (&v, &w)) in e.iter().zip(&g.embed_norm).enumerate() {
                unsafe { *hp.add(i * hidden + j) = w * (v * r) };
            }
        }
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_call(c, p, d, cfg, enc, n, pos0 + at, lrows);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if lrows > 0 {
            let v = unsafe { std::slice::from_raw_parts(d.logits.contents() as *const f32, lrows * vocab) };
            if all {
                logits.extend_from_slice(v);
            } else {
                *logits = v.to_vec();
            }
        }
        at += n;
    }
    true
}

/// A new sequence: its device state goes (the buffers stay for the next one).
pub(crate) fn reset(id: u64) {
    DECLINED.lock().unwrap().retain(|&x| x != id);
}

fn grid(enc: &ComputeCommandEncoderRef, x: u64, y: u64, z: u64, threads: u64) {
    enc.dispatch_thread_groups(MTLSize::new(x.max(1), y.max(1), z.max(1)), MTLSize::new(threads, 1, 1));
}

/// `m · x` for `n` rows: x rows `cols` apart from `xoff` bytes, y rows `rows` apart.
#[allow(clippy::too_many_arguments)]
fn mv(c: &Ctx, enc: &ComputeCommandEncoderRef, fbuf: &WeightArena, m: &Mat, xs: &Buffer, xoff: usize, y: &Buffer, yoff: usize, n: usize) {
    for j in 0..n {
        match m {
            Mat::Q4 { abs, rows, cols } => {
                enc.set_compute_pipeline_state(&c.q4tp_m);
                fbuf.bind(enc, 0, *abs);
                enc.set_buffer(1, Some(xs), (xoff + j * cols * 4) as u64);
                enc.set_buffer(2, Some(y), (yoff + j * rows * 4) as u64);
                let (gpr, r, addy) = ((cols / 32) as u32, *rows as u32, 0u32);
                set_bytes_t(enc, 3, &gpr);
                set_bytes_t(enc, 4, &r);
                set_bytes_t(enc, 5, &addy);
                enc.dispatch_thread_groups(MTLSize::new((*rows as u64).div_ceil(16), 1, 1), MTLSize::new(128, 1, 1));
            }
            Mat::Q8 { abs, rows, cols, rs, col } => {
                enc.set_compute_pipeline_state(&c.q8f_r4);
                fbuf.bind(enc, 0, *abs);
                enc.set_buffer(1, Some(xs), (xoff + j * cols * 4) as u64);
                enc.set_buffer(2, Some(rs), 0);
                enc.set_buffer(3, Some(y), (yoff + j * rows * 4) as u64);
                enc.set_buffer(4, Some(col), 0);
                let (c4, r) = ((cols / 4) as u32, *rows as u32);
                set_bytes_t(enc, 5, &c4);
                set_bytes_t(enc, 6, &r);
                enc.dispatch_thread_groups(MTLSize::new((*rows as u64).div_ceil(32), 1, 1), MTLSize::new(256, 1, 1));
            }
        }
    }
}

fn base_ip(cfg: &Cfg, n: usize, pos0: usize) -> IP {
    IP {
        h: cfg.dim as u32,
        nh: cfg.nh as u32,
        nkv: cfg.nkv as u32,
        hd: cfg.hd as u32,
        dr: cfg.d_rel as u32,
        kc: 4,
        n: n as u32,
        pos0: pos0 as u32,
        bstride: (MAXB * cfg.dim) as u32,
        ostride: (NMAX * cfg.dim) as u32,
        topk: cfg.top_k as u32,
        eps: cfg.eps as f32,
        qclip: cfg.qk_clip,
        rclip: cfg.rel_clip,
        alpha: cfg.log_alpha as f32,
        floor_: cfg.log_floor as f32,
        scale: 1.0 / (cfg.hd as f32).sqrt(),
        cap: cfg.gate_cap,
        ucap: cfg.up_cap,
        rscale: cfg.route_scale,
        bslot: u32::MAX,
        ..IP::default()
    }
}

fn setp(enc: &ComputeCommandEncoderRef, ip: &IP) {
    set_bytes_t(enc, 15, ip);
}

/// The dispatches of one call of `n` rows from `pos0`: the 48 applications,
/// the final mixture, the head over the last `lrows` rows.
#[allow(clippy::too_many_arguments)]
fn encode_call(c: &Ctx, p: &Pipes, d: &Dev, cfg: &Cfg, enc: &ComputeCommandEncoderRef, n: usize, pos0: usize, lrows: usize) {
    let hidden = cfg.dim;
    let base = base_ip(cfg, n, pos0);
    let hb = (n * hidden * 4) as u64;
    let mut cur = 0usize;
    let mut hin: (&Buffer, u64) = (&d.h[0], 0);
    let mut nb = 0usize;
    let mut npre = 0usize;
    for (li, pass) in cfg.plan() {
        let l = &d.layers[li];
        let s = &d.slots[cfg.slot(li, pass)];
        if li == cfg.core_start {
            if pass == 0 {
                copy(p, enc, hin.0, &d.pre_h, hb);
                npre = nb;
            } else {
                copy(p, enc, &d.blk, &d.bank, (n * MAXB * hidden * 4) as u64);
                copy(p, enc, hin.0, &d.bank_out, hb);
                hin = (&d.pre_h, 0);
                nb = npre;
            }
        }
        let hout = &d.h[1 - cur];
        app(c, p, d, cfg, enc, l, s, li, pass, n, &base, hin.0, hout, &mut nb, npre);
        cur = 1 - cur;
        hin = (hout, 0);
    }
    // the final mixture, its norm and the head
    let mut ip = base;
    ip.nb = nb as u32;
    enc.set_compute_pipeline_state(&p.mix);
    enc.set_buffer(0, Some(&d.blk), 0);
    enc.set_buffer(1, Some(hin.0), 0);
    enc.set_buffer(2, Some(&d.final_q), 0);
    enc.set_buffer(3, Some(&d.fin), 0);
    setp(enc, &ip);
    grid(enc, 1, n as u64, 1, 256);
    if lrows > 0 {
        let r0 = n - lrows;
        enc.set_compute_pipeline_state(&p.rms);
        enc.set_buffer(0, Some(&d.fin), (r0 * hidden * 4) as u64);
        enc.set_buffer(1, Some(&d.final_norm), 0);
        enc.set_buffer(2, Some(&d.xf), 0);
        setp(enc, &ip);
        grid(enc, lrows as u64, 1, 1, 256);
        mv(c, enc, &d.fbuf, &d.head, &d.xf, 0, &d.logits, 0, lrows);
        if cfg.logit_cap > 0.0 {
            let mut ipc = ip;
            ipc.v = d.head.rows() as u32;
            ipc.cap = cfg.logit_cap;
            enc.set_compute_pipeline_state(&p.softcap);
            enc.set_buffer(0, Some(&d.logits), 0);
            setp(enc, &ipc);
            grid(enc, (d.head.rows() as u64).div_ceil(256), lrows as u64, 1, 256);
        }
    }
}

/// `bytes` of `src` into `dst`, in order with the dispatches around it.
fn copy(p: &Pipes, enc: &ComputeCommandEncoderRef, src: &Buffer, dst: &Buffer, bytes: u64) {
    let floats = bytes / 4;
    let ip = IP { rows: floats as u32, ..IP::default() };
    enc.set_compute_pipeline_state(&p.copy);
    enc.set_buffer(0, Some(src), 0);
    enc.set_buffer(1, Some(dst), 0);
    setp(enc, &ip);
    grid(enc, floats.div_ceil(256), 1, 1, 256);
}

/// One layer application of the call's rows (`inkling.cu`'s `app` on Metal).
#[allow(clippy::too_many_arguments)]
fn app(c: &Ctx, p: &Pipes, d: &Dev, cfg: &Cfg, enc: &ComputeCommandEncoderRef, l: &LayerM, s: &SlotM, li: usize, pass: usize, n: usize, base: &IP, hin: &Buffer, hout: &Buffer, nb: &mut usize, npre: usize) {
    let hidden = cfg.dim;
    let nu = n as u64;
    let plane = (NMAX * hidden * 4) as u64;
    // 1. the attention inputs
    let mut inp: [(&Buffer, u64); 3] = [(hin, 0), (hin, 0), (hin, 0)];
    if let Some((aq, nq)) = &l.aq {
        let mut ip = *base;
        ip.nb = *nb as u32;
        enc.set_compute_pipeline_state(&p.mix);
        enc.set_buffer(0, Some(&d.blk), 0);
        enc.set_buffer(1, Some(hin), 0);
        enc.set_buffer(2, Some(aq), 0);
        enc.set_buffer(3, Some(&d.in3), 0);
        setp(enc, &ip);
        grid(enc, *nq as u64, nu, 1, 256);
        inp = if *nq == 3 { [(&d.in3, 0), (&d.in3, plane), (&d.in3, 2 * plane)] } else { [(&d.in3, 0); 3] };
    }
    // 2. the messages of a later pass, gated into q, k, v
    let gates = if pass > 0 { l.gates.as_ref() } else { None };
    if let Some(gs) = gates {
        let i = li - cfg.core_start;
        let mut ip = *base;
        ip.nb = (cfg.loop_len() / cfg.block - 1) as u32;
        enc.set_compute_pipeline_state(&p.mix);
        enc.set_buffer(0, Some(&d.bank), ((npre + 1) * hidden * 4) as u64);
        enc.set_buffer(1, Some(&d.bank_out), 0);
        enc.set_buffer(2, Some(&d.msg_q), (i * 4 * hidden * 4) as u64);
        enc.set_buffer(3, Some(&d.msg), 0);
        setp(enc, &ip);
        grid(enc, 4, nu, 1, 256);
        for j in 0..3 {
            gate(p, enc, base, &gs[j], inp[j], (&d.msg, j as u64 * plane), &d.gd, (&d.gin, j as u64 * plane), n, hidden);
            inp[j] = (&d.gin, j as u64 * plane);
        }
    }
    // 3. the input norms
    let one = l.sliding && gates.is_none();
    for j in 0..(if one { 1 } else { 3 }) {
        enc.set_compute_pipeline_state(&p.rms);
        enc.set_buffer(0, Some(inp[j].0), inp[j].1);
        enc.set_buffer(1, Some(&l.norms[j]), 0);
        enc.set_buffer(2, Some(&d.x[j]), 0);
        setp(enc, base);
        grid(enc, nu, 1, 1, 256);
    }
    let (xk, xv) = if one { (0, 0) } else { (1, 2) };
    // 4. the projections
    mv(c, enc, &d.fbuf, &l.q, &d.x[0], 0, &d.yq, 0, n);
    mv(c, enc, &d.fbuf, &l.gate, &d.x[0], 0, &d.yg, 0, n);
    mv(c, enc, &d.fbuf, &l.rel, &d.x[0], 0, &d.yr, 0, n);
    mv(c, enc, &d.fbuf, &l.k, &d.x[xk], 0, &d.yk, 0, n);
    mv(c, enc, &d.fbuf, &l.v, &d.x[xv], 0, &d.yv, 0, n);
    let fmv = |w: &Buffer, rows: usize, x: &Buffer, y: &Buffer| {
        let mut ip = *base;
        ip.rows = rows as u32;
        enc.set_compute_pipeline_state(&p.fmv);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), 0);
        setp(enc, &ip);
        grid(enc, (rows as u64).div_ceil(8), nu, 1, 256);
    };
    fmv(&l.sink, cfg.nh, &d.x[0], &d.sk);
    let full = l.dk.is_some();
    if let (Some(wk), Some(wv)) = (&l.dk, &l.dv) {
        fmv(wk, cfg.nkv * 4, &d.x[xk], &d.dkb);
        fmv(wv, cfg.nkv * 4, &d.x[xv], &d.dvb);
    }
    // 5. queries and relative states; keys and values into the cache
    let mut ip = *base;
    ip.full = (!l.sliding) as u32;
    enc.set_compute_pipeline_state(&p.qr);
    enc.set_buffer(0, Some(&d.yq), 0);
    enc.set_buffer(1, Some(&d.yr), 0);
    enc.set_buffer(2, Some(&l.q_scale), 0);
    enc.set_buffer(3, Some(&l.rel_scale), 0);
    setp(enc, &ip);
    grid(enc, cfg.nh as u64, nu, 1, cfg.hd as u64);
    let mut ip = *base;
    ip.full = full as u32;
    ip.ring = s.ring as u32;
    enc.set_compute_pipeline_state(&p.kv);
    enc.set_buffer(0, Some(&d.yk), 0);
    enc.set_buffer(1, Some(&d.yv), 0);
    enc.set_buffer(2, Some(if full { &d.dkb } else { &d.dummy }), 0);
    enc.set_buffer(3, Some(if full { &d.dvb } else { &d.dummy }), 0);
    enc.set_buffer(4, Some(&l.tk), 0);
    enc.set_buffer(5, Some(&l.tv), 0);
    enc.set_buffer(6, Some(&s.rings[0]), 0);
    enc.set_buffer(7, Some(&s.rings[1]), 0);
    enc.set_buffer(8, Some(&s.k), 0);
    enc.set_buffer(9, Some(&s.v), 0);
    setp(enc, &ip);
    enc.dispatch_thread_groups(MTLSize::new(cfg.nkv as u64, 2, 1), MTLSize::new(cfg.hd as u64, 1, 1));
    // 6. the attention, its gate, the O projection
    let mut ip = *base;
    ip.ring = s.ring as u32;
    ip.win = if s.ring > 0 { cfg.window as u32 } else { 0 };
    ip.e = l.extent as u32;
    enc.set_compute_pipeline_state(&p.attn);
    enc.set_buffer(0, Some(&d.yq), 0);
    enc.set_buffer(1, Some(&d.yr), 0);
    enc.set_buffer(2, Some(&d.sk), 0);
    enc.set_buffer(3, Some(&l.rel_table), 0);
    enc.set_buffer(4, Some(&s.k), 0);
    enc.set_buffer(5, Some(&s.v), 0);
    enc.set_buffer(6, Some(&d.att), 0);
    setp(enc, &ip);
    grid(enc, cfg.nh as u64, nu, 1, 256);
    let qs = cfg.nh * cfg.hd;
    enc.set_compute_pipeline_state(&p.ogate);
    enc.set_buffer(0, Some(&d.att), 0);
    enc.set_buffer(1, Some(&d.yg), 0);
    setp(enc, base);
    grid(enc, (qs as u64).div_ceil(256), nu, 1, 256);
    mv(c, enc, &d.fbuf, &l.o, &d.att, 0, &d.ya, 0, n);
    // 7. the output convolution; a boundary closes the running sum into a block
    let mut ip = *base;
    ip.bslot = if l.boundary { *nb as u32 } else { u32::MAX };
    enc.set_compute_pipeline_state(&p.post);
    enc.set_buffer(0, Some(&d.ya), 0);
    enc.set_buffer(1, Some(&l.tpa), 0);
    enc.set_buffer(2, Some(&s.rings[2]), 0);
    enc.set_buffer(3, Some(hin), 0);
    enc.set_buffer(4, Some(&d.blk), 0);
    enc.set_buffer(5, Some(&d.r), 0);
    setp(enc, &ip);
    grid(enc, (hidden as u64).div_ceil(256), 1, 1, 256);
    if l.boundary {
        *nb += 1;
    }
    // 8. the MoE input
    let mut ip = *base;
    ip.nb = *nb as u32;
    enc.set_compute_pipeline_state(&p.mix);
    enc.set_buffer(0, Some(&d.blk), 0);
    enc.set_buffer(1, Some(&d.r), 0);
    enc.set_buffer(2, Some(&l.mq), 0);
    enc.set_buffer(3, Some(&d.m), 0);
    setp(enc, &ip);
    grid(enc, 1, nu, 1, 256);
    let mut mi: (&Buffer, u64) = (&d.m, 0);
    if let Some(gs) = gates {
        gate(p, enc, base, &gs[3], mi, (&d.msg, 3 * plane), &d.gd, (&d.mg, 0), n, hidden);
        mi = (&d.mg, 0);
    }
    enc.set_compute_pipeline_state(&p.rms);
    enc.set_buffer(0, Some(mi.0), mi.1);
    enc.set_buffer(1, Some(&l.post_norm), 0);
    enc.set_buffer(2, Some(&d.xm), 0);
    setp(enc, base);
    grid(enc, nu, 1, 1, 256);
    // 9. the routed experts into y
    let mut ip = *base;
    ip.e = l.e as u32;
    ip.inter = l.inter as u32;
    enc.set_compute_pipeline_state(&p.router);
    enc.set_buffer(0, Some(&l.router), 0);
    enc.set_buffer(1, Some(&d.xm), 0);
    enc.set_buffer(2, Some(&d.lg), 0);
    setp(enc, &ip);
    grid(enc, (l.e as u64).div_ceil(8), nu, 1, 256);
    enc.set_compute_pipeline_state(&p.select);
    enc.set_buffer(0, Some(&d.lg), 0);
    enc.set_buffer(1, Some(&l.bias), 0);
    enc.set_buffer(2, Some(&d.pick), 0);
    enc.set_buffer(3, Some(&d.wt), 0);
    setp(enc, &ip);
    grid(enc, nu, 1, 1, 128);
    let wins: Vec<&Buffer> = (0..4).map(|i| d.fbuf.window(i.min(d.fbuf.bufs.len() - 1))).collect();
    enc.set_compute_pipeline_state(&p.moe_gu);
    for (i, w) in wins.iter().enumerate() {
        enc.set_buffer(i as u64, Some(w), 0);
    }
    enc.set_buffer(4, Some(&l.bases), 0);
    enc.set_buffer(5, Some(&d.pick), 0);
    enc.set_buffer(6, Some(&d.xm), 0);
    enc.set_buffer(7, Some(&d.act), 0);
    set_bytes_t(enc, 8, &d.wstride);
    setp(enc, &ip);
    grid(enc, (l.inter as u64).div_ceil(16), cfg.top_k as u64, nu, 128);
    enc.set_compute_pipeline_state(&p.moe_down);
    for (i, w) in wins.iter().enumerate() {
        enc.set_buffer(i as u64, Some(w), 0);
    }
    enc.set_buffer(4, Some(&l.bases), 0);
    enc.set_buffer(5, Some(&d.pick), 0);
    enc.set_buffer(6, Some(&d.wt), 0);
    enc.set_buffer(7, Some(&d.act), 0);
    enc.set_buffer(8, Some(&d.y), 0);
    set_bytes_t(enc, 9, &d.wstride);
    setp(enc, &ip);
    grid(enc, (hidden as u64).div_ceil(16), nu, 1, 128);
    // 10. the output convolution onto the running sum
    let ip = *base;
    enc.set_compute_pipeline_state(&p.post);
    enc.set_buffer(0, Some(&d.y), 0);
    enc.set_buffer(1, Some(&l.tpm), 0);
    enc.set_buffer(2, Some(&s.rings[3]), 0);
    enc.set_buffer(3, Some(&d.r), 0);
    enc.set_buffer(4, Some(&d.dummy), 0);
    enc.set_buffer(5, Some(hout), 0);
    setp(enc, &ip);
    grid(enc, (hidden as u64).div_ceil(256), 1, 1, 256);
}

/// A message gate on the encoder.
#[allow(clippy::too_many_arguments)]
fn gate(p: &Pipes, enc: &ComputeCommandEncoderRef, base: &IP, g: &(Buffer, Buffer, Buffer, usize), x: (&Buffer, u64), m: (&Buffer, u64), dbuf: &Buffer, out: (&Buffer, u64), n: usize, hidden: usize) {
    let mut ip = *base;
    ip.r = g.3 as u32;
    enc.set_compute_pipeline_state(&p.gate_down);
    enc.set_buffer(0, Some(x.0), x.1);
    enc.set_buffer(1, Some(m.0), m.1);
    enc.set_buffer(2, Some(&g.0), 0);
    enc.set_buffer(3, Some(dbuf), 0);
    set_bytes_t(enc, 15, &ip);
    grid(enc, g.3 as u64, n as u64, 1, 256);
    enc.set_compute_pipeline_state(&p.gate_out);
    enc.set_buffer(0, Some(x.0), x.1);
    enc.set_buffer(1, Some(m.0), m.1);
    enc.set_buffer(2, Some(dbuf), 0);
    enc.set_buffer(3, Some(&g.1), 0);
    enc.set_buffer(4, Some(&g.2), 0);
    enc.set_buffer(5, Some(out.0), out.1);
    set_bytes_t(enc, 15, &ip);
    grid(enc, (hidden as u64).div_ceil(256), n as u64, 1, 256);
}
