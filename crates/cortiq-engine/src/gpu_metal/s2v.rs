//! Kandinsky S2V on native Metal: the denoiser and the VAE decoder.
//!
//! Weights stay in the file mapping: q4tp tiles are dequantized to half as the
//! GEMM stages them, q8 int8 is staged times the column field with the row scale
//! in the epilogue (the `qi_mm_*` kernels of the Qwen-Image-2.1 library), so the
//! 21 B parameters never become planes on a shared 24 GB. The block order is the
//! wgpu chain's (`gpu_wgpu/s2v.rs`); the short attentions are `qi_flash` with the
//! prompt or the speech frame's keys as its prefix.
//!
//! Every long dispatch is cut: a GEMM of 36736 rows goes in bands and the flash
//! attention a few heads at a time, each on its own command buffer, because a
//! single buffer of several seconds risks the GPU watchdog on a laptop.

use super::qi21::{Rec, buf_from, buf_zeroed, set_p, write_f32};
use super::{Ctx, WeightArena};
use crate::ltxvae::Vol;
use crate::s2v::{self, S2vDit, S2vGeom, audio_groups};
use crate::s2v_vae::{Conv, GroupNorm, Mid, Res, S2vVae, S2vVaeEncoder};
use crate::s2vgen::S2vCond;
use cortiq_core::{CmfModel, TensorDtype};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const MSL: &str = concat!(
    include_str!("qi21_msl.metal"),
    "\n",
    include_str!("zimage_msl.metal"),
    "\n",
    include_str!("s2v_msl.metal")
);

struct Pipes {
    mm_q4tp: ComputePipelineState,
    mm_q8: ComputePipelineState,
    flash: ComputePipelineState,
    row: ComputePipelineState,
    norm: ComputePipelineState,
    gelu: ComputePipelineState,
    embed: ComputePipelineState,
    addv: ComputePipelineState,
    head: ComputePipelineState,
    vconv: ComputePipelineState,
    gnpart: ComputePipelineState,
    gnfin: ComputePipelineState,
    gnapply: ComputePipelineState,
    cvt: ComputePipelineState,
    softmax: ComputePipelineState,
    rgb: ComputePipelineState,
}
unsafe impl Send for Pipes {}
unsafe impl Sync for Pipes {}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c
            ._device
            .new_library_with_source(MSL, &opts)
            .map_err(|e| format!("s2v MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            c._device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            mm_q4tp: pso("qi_mm_q4tp")?,
            mm_q8: pso("qi_mm_q8")?,
            flash: pso("qi_flash")?,
            row: pso("s2v_row")?,
            norm: pso("s2v_norm")?,
            gelu: pso("s2v_gelu")?,
            embed: pso("qi_embed")?,
            addv: pso("s2v_addv")?,
            head: pso("s2v_head")?,
            vconv: pso("s2v_vconv")?,
            gnpart: pso("zv_gn_part")?,
            gnfin: pso("zv_gn_fin")?,
            gnapply: pso("zv_gn_apply")?,
            cvt: pso("zv_cvt")?,
            softmax: pso("s2v_softmax")?,
            rgb: pso("s2v_rgb")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("s2v: Metal device path declined: {e}");
            None
        }
    }
}

fn prof_on() -> bool {
    std::env::var("CMF_S2V_METAL_PROF").is_ok()
}

// ───────────────────────── weights and parameters ─────────────────────────

#[derive(Clone, Copy, Default)]
struct Tens {
    abs: usize,
    rows: usize,
    cols: usize,
    fmt: u8,
    params_off: usize,
    codes_off: usize,
    stride: usize,
    rs: usize,
    col: usize,
}

fn f16s(b: &[u8]) -> impl Iterator<Item = f32> + '_ {
    b.chunks_exact(2).map(|c| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
}

/// Grows the float `aux` buffer, aligned for the kernels' float4 reads.
#[derive(Default)]
struct Aux(Vec<f32>);
impl Aux {
    fn push(&mut self, v: &[f32]) -> usize {
        while self.0.len() % 4 != 0 {
            self.0.push(0.0);
        }
        let o = self.0.len();
        self.0.extend_from_slice(v);
        o
    }
}

fn tens(model: &CmfModel, name: &str, rows: usize, cols: usize, aux: &mut Aux) -> Result<Tens, String> {
    let e = model.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if e.shape != [rows, cols] {
        return Err(format!("{name}: shape {:?}", e.shape));
    }
    let abs = model.entry_abs_offset(e).ok_or_else(|| format!("{name}: not in the primary shard"))?;
    if abs % 16 != 0 {
        return Err(format!("{name}: unaligned ({abs})"));
    }
    let bytes = model.entry_bytes(e);
    let q = rows * cols;
    let mut t = Tens { abs, rows, cols, ..Default::default() };
    match e.dtype {
        TensorDtype::Q4TiledP => {
            let (po, co, st) = cortiq_core::quant::q4tp_sections(rows, cols);
            t.params_off = po;
            t.codes_off = co;
            t.stride = st;
        }
        TensorDtype::Q8_2f if bytes.len() == q + 2 * rows + 2 * cols => {
            t.fmt = 1;
            let rs: Vec<f32> = f16s(&bytes[q..q + 2 * rows]).collect();
            let cf: Vec<f32> = f16s(&bytes[q + 2 * rows..]).collect();
            t.rs = aux.push(&rs);
            t.col = aux.push(&cf);
        }
        TensorDtype::Q8Row if bytes.len() == q + 2 * rows => {
            t.fmt = 1;
            let rs: Vec<f32> = f16s(&bytes[q..]).collect();
            t.rs = aux.push(&rs);
            t.col = aux.push(&vec![1.0f32; cols]);
        }
        d => return Err(format!("{name}: codec {d:?} (the Metal chain reads q4tp / q8_2f / q8_row)")),
    }
    Ok(t)
}

/// The tensors and the parameter offsets of one attention.
#[derive(Clone, Copy, Default)]
struct At {
    q: Tens,
    k: Tens,
    v: Tens,
    o: Tens,
    qn: usize,
    kn: usize,
    qb: usize,
    kb: usize,
    vb: usize,
    ob: usize,
}

fn attn_of(model: &CmfModel, p: &str, a: &s2v::Attn, h: usize, aux: &mut Aux) -> Result<At, String> {
    let bias = |l: &s2v::Lin| l.b.clone().unwrap_or_else(|| vec![0.0; l.rows()]);
    Ok(At {
        q: tens(model, &format!("{p}.to_query.weight"), h, h, aux)?,
        k: tens(model, &format!("{p}.to_key.weight"), h, h, aux)?,
        v: tens(model, &format!("{p}.to_value.weight"), h, h, aux)?,
        o: tens(model, &format!("{p}.out_layer.weight"), h, h, aux)?,
        qn: aux.push(&a.qn),
        kn: aux.push(&a.kn),
        qb: aux.push(&bias(&a.q)),
        kb: aux.push(&bias(&a.k)),
        vb: aux.push(&bias(&a.v)),
        ob: aux.push(&bias(&a.o)),
    })
}

struct VisB {
    sa: At,
    ca: At,
    v2a: Option<(At, usize, usize)>,
    w1: Tens,
    w2: Tens,
}

struct ShortB {
    a: At,
    w1: Tens,
    w2: Tens,
    /// LayerNorm (w, b) of the speech blocks.
    aff: Option<[usize; 4]>,
}

struct Seq {
    x: Buffer,
    xn: Buffer,
    panel: Buffer,
    att: Buffer,
    y: Buffer,
    hid: Buffer,
    cs: Buffer,
    sn: Buffer,
}

impl Seq {
    fn new(c: &Ctx, m: usize, h: usize, ff: usize) -> Seq {
        let a = m.div_ceil(128) * 128 + 128;
        Seq {
            x: buf_zeroed(c, a * h * 4),
            xn: buf_zeroed(c, a * h * 2),
            panel: buf_zeroed(c, a * 3 * h * 2),
            att: buf_zeroed(c, a * h * 2),
            y: buf_zeroed(c, a * h * 4),
            hid: buf_zeroed(c, a * ff * 2),
            cs: buf_zeroed(c, a * 64 * 4),
            sn: buf_zeroed(c, a * 64 * 4),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PMm {
    n: u32,
    rows: u32,
    k: u32,
    ldx: u32,
    ldy: u32,
    epi: u32,
    mul: f32,
    params_off: u32,
    codes_off: u32,
    code_stride: u32,
    pad0: u32,
    pad1: u32,
    x_off: [u32; 4],
    y_off: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PFa {
    q_row0: u32,
    nq: u32,
    n_pre: u32,
    n_own: u32,
    ldp: u32,
    h: u32,
    ldo: u32,
    oscale: f32,
    use_vis: u32,
    head0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PRow {
    h: u32,
    row0: u32,
    orow0: u32,
    mode: u32,
    eps: f32,
    oscale: f32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PNorm {
    ld: u32,
    col: u32,
    nh: u32,
    flags: u32,
    eps: f32,
    bscale: f32,
    qmul: f32,
    row0: u32,
}

/// Rows of one GEMM band: about a second of work on an M4.
const BAND_ROWS: usize = 4096;
/// Heads of one flash dispatch.
const FLASH_HEADS: usize = 8;

struct E<'r, 'a> {
    rec: &'r mut Rec<'a>,
    p: &'static Pipes,
    arena: &'r WeightArena,
    aux: &'r Buffer,
    h: usize,
    nh: usize,
}

impl E<'_, '_> {
    /// `Y = X·Wᵀ`. `ts` share a shape and a codec; several of them run as one
    /// z-batched dispatch. Cut into bands on the host's side by offsets.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&mut self, ts: &[Tens], x: &Buffer, ldx: usize, y: &Buffer, y_off: &[usize], ldy: usize, n: usize, half_out: bool, mul: f32) {
        let t0 = ts[0];
        let mut start = 0usize;
        while start < n {
            let band = (n - start).min(BAND_ROWS);
            let mut pm = PMm {
                n: band as u32,
                rows: t0.rows as u32,
                k: t0.cols as u32,
                ldx: ldx as u32,
                ldy: ldy as u32,
                epi: half_out as u32,
                mul,
                params_off: t0.params_off as u32,
                codes_off: t0.codes_off as u32,
                code_stride: t0.stride as u32,
                ..Default::default()
            };
            for z in 0..ts.len() {
                pm.x_off[z] = (start * ldx) as u32;
                pm.y_off[z] = (start * ldy + y_off[z]) as u32;
            }
            let (p, aux, arena) = (self.p, self.aux, self.arena);
            let enc = self.rec.enc("gemm");
            enc.set_compute_pipeline_state(if t0.fmt == 1 { &p.mm_q8 } else { &p.mm_q4tp });
            for s in 0..3usize {
                let t = ts[s.min(ts.len() - 1)];
                arena.bind(enc, s as u64, t.abs);
                enc.set_buffer(3 + s as u64, Some(aux), (t.rs * 4) as u64);
                enc.set_buffer(9 + s as u64, Some(aux), (t.col * 4) as u64);
            }
            enc.set_buffer(6, Some(x), 0);
            enc.set_buffer(7, Some(y), 0);
            set_p(enc, 8, &pm);
            enc.dispatch_thread_groups(
                MTLSize::new(band.div_ceil(64) as u64, (t0.rows / 64) as u64, ts.len() as u64),
                MTLSize::new(128, 1, 1),
            );
            self.rec.cut();
            start += band;
        }
    }

    /// One row op over `n` rows; buffers `(mods-or-aux, word offset)`.
    #[allow(clippy::too_many_arguments)]
    fn row(&mut self, s: &Seq, row0: usize, orow0: usize, n: usize, mode: u32, g: (&Buffer, usize), sh: (&Buffer, usize), sc: (&Buffer, usize), b: usize, oscale: f32) {
        let pr = PRow { h: self.h as u32, row0: row0 as u32, orow0: orow0 as u32, mode, eps: 1e-5, oscale, ..Default::default() };
        let (p, aux) = (self.p, self.aux);
        let enc = self.rec.enc("row");
        enc.set_compute_pipeline_state(&p.row);
        enc.set_buffer(0, Some(&s.x), 0);
        enc.set_buffer(1, Some(&s.y), 0);
        enc.set_buffer(2, Some(&s.xn), 0);
        enc.set_buffer(3, Some(g.0), (g.1 * 4) as u64);
        enc.set_buffer(4, Some(sh.0), (sh.1 * 4) as u64);
        enc.set_buffer(5, Some(sc.0), (sc.1 * 4) as u64);
        enc.set_buffer(6, Some(aux), (b * 4) as u64);
        set_p(enc, 7, &pr);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(&mut self, panel: &Buffer, cs: &Buffer, sn: &Buffer, n: usize, ld: usize, col: usize, flags: u32, w: usize, b: usize, qmul: f32) {
        let pn = PNorm {
            ld: ld as u32,
            col: col as u32,
            nh: self.nh as u32,
            flags,
            eps: 1e-5,
            bscale: 1.0,
            qmul,
            row0: 0,
        };
        let (p, aux) = (self.p, self.aux);
        let enc = self.rec.enc("norm");
        enc.set_compute_pipeline_state(&p.norm);
        enc.set_buffer(0, Some(panel), 0);
        enc.set_buffer(1, Some(aux), (w * 4) as u64);
        enc.set_buffer(2, Some(aux), (b * 4) as u64);
        enc.set_buffer(3, Some(cs), 0);
        enc.set_buffer(4, Some(sn), 0);
        set_p(enc, 5, &pn);
        enc.dispatch_thread_groups(
            MTLSize::new(n as u64, self.nh.div_ceil(4) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// Flash attention over `n` query rows of `panel`, a few heads a buffer.
    #[allow(clippy::too_many_arguments)]
    fn flash(&mut self, panel: &Buffer, ldp: usize, att: &Buffer, q_row0: usize, n: usize, keys: Option<(&Buffer, usize, usize)>) {
        let p = self.p;
        let (h, nh) = (self.h, self.nh);
        let mut head0 = 0;
        while head0 < nh {
            let nheads = FLASH_HEADS.min(nh - head0);
            let pf = PFa {
                q_row0: q_row0 as u32,
                nq: n as u32,
                n_pre: keys.map_or(0, |k| k.2) as u32,
                n_own: if keys.is_some() { 0 } else { n as u32 },
                ldp: ldp as u32,
                h: h as u32,
                ldo: h as u32,
                oscale: 1.0,
                use_vis: 0,
                head0: head0 as u32,
                ..Default::default()
            };
            let enc = self.rec.enc("flash");
            enc.set_compute_pipeline_state(&p.flash);
            enc.set_buffer(0, Some(panel), 0);
            enc.set_buffer(1, Some(att), 0);
            match keys {
                Some((b, off, _)) => enc.set_buffer(2, Some(b), off as u64),
                None => enc.set_buffer(2, Some(panel), 0),
            }
            enc.set_buffer(3, Some(panel), 0);
            set_p(enc, 4, &pf);
            enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(64) as u64, nheads as u64, 1), MTLSize::new(256, 1, 1));
            head0 += nheads;
            if n >= BAND_ROWS {
                self.rec.cut();
            }
        }
    }

    fn gelu(&mut self, hid: &Buffer, n4: usize) {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct G {
            n4: u32,
            scale: f32,
            iscale: f32,
            pad: u32,
        }
        let g = G { n4: n4 as u32, scale: 1.0, iscale: 1.0, pad: 0 };
        let p = self.p;
        let enc = self.rec.enc("gelu");
        enc.set_compute_pipeline_state(&p.gelu);
        enc.set_buffer(0, Some(hid), 0);
        set_p(enc, 1, &g);
        enc.dispatch_thread_groups(MTLSize::new((n4 as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    }
}

pub struct S2vDev {
    c: &'static Ctx,
    p: &'static Pipes,
    _model: Arc<CmfModel>,
    arena: Arc<WeightArena>,
    aux: Buffer,
    mods: Buffer,
    visual: Vec<VisB>,
    text: Vec<ShortB>,
    audio: Vec<ShortB>,
    s_vis: Seq,
    s_small: Seq,
    tkv: Buffer,
    akv: Buffer,
    tcache: Buffer,
    t0: Vec<f32>,
    x0: Buffer,
    tok: Buffer,
    emb: Buffer,
    head_w: Buffer,
    head_b: Buffer,
    out: Buffer,
    h: usize,
    ff: usize,
    nh: usize,
    g: S2vGeom,
    nt: usize,
    na: usize,
    emb_b: usize,
    mask0: usize,
    vm: Vec<Vec<f32>>,
    tm: Vec<Vec<f32>>,
    fm: Vec<Vec<f32>>,
    text_off: usize,
    fin_off: usize,
    text_done: Vec<bool>,
    audio_for: Option<usize>,
    audio_feats: Vec<Vec<f32>>,
    dit: *const S2vDit,
    reference: Vec<f32>,
    cfg: s2v::S2vConfig,
    prof: bool,
}

unsafe impl Send for S2vDev {}

pub fn prepare(dit: &S2vDit, g: &S2vGeom, cond: &S2vCond, ts: &[f32], sf: [f32; 3]) -> Option<S2vDev> {
    if std::env::var("CMF_S2V_METAL").as_deref() == Ok("0") {
        return None;
    }
    let c = super::ctx()?;
    let p = pipes(c)?;
    let t_all = Instant::now();
    let model = dit.model.clone();
    let cfg = dit.cfg.clone();
    let (h, ff, nh) = (cfg.dim, cfg.ff, cfg.heads());
    let m = g.total();
    let nt = cond.text.len() / cfg.in_text;
    let na = cond.audio[0].len() / cfg.in_audio;
    let small = nt.max(na);
    let alloc = m.div_ceil(128) * 128 + 128;
    let largest = (alloc * ff * 2).max(alloc * 3 * h * 2);
    let total = alloc * (ff * 2 + 3 * h * 2 + 2 * h * 4 + 2 * h * 2);
    if largest as u64 > c._device.max_buffer_length() || total as u64 > c._device.recommended_max_working_set_size() {
        eprintln!(
            "s2v: the Metal program needs {:.1} GB in all (largest {:.1} GB), past this device: running on the CPU",
            total as f64 / 1e9,
            largest as f64 / 1e9
        );
        return None;
    }
    let (arena, _) = super::file_buffer(c, &model)?;

    // Modulations, as on wgpu.
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
    let text_off = cfg.n_visual * 2 * 9 * h;
    let fin_off = text_off + cfg.n_text * 6 * h;
    let mods = buf_zeroed(c, (fin_off + 2 * h) * 4);
    let mut vm: Vec<Vec<f32>> = vec![Vec::with_capacity(cfg.n_visual * 9 * h); n_t];
    let vnames: Vec<String> = (0..cfg.n_visual).map(|b| format!("dit.visual_transformer_blocks.{b}.visual_modulation.out_layer")).collect();
    for (b, y) in dit.modulate_all(&vnames, &rows_all, n_t + 1).ok()?.into_iter().enumerate() {
        write_f32(&mods, b * 2 * 9 * h, &y[..9 * h]);
        for (s, v) in vm.iter_mut().enumerate() {
            v.extend_from_slice(&y[(s + 1) * 9 * h..(s + 2) * 9 * h]);
        }
    }
    let t_rows = &rows_all[cfg.time_dim..];
    let mut tm: Vec<Vec<f32>> = vec![Vec::with_capacity(cfg.n_text * 6 * h); n_t];
    for b in 0..cfg.n_text {
        let y = dit.modulate(&format!("dit.text_transformer_blocks.{b}.text_modulation.out_layer"), t_rows, n_t).ok()?;
        for (s, v) in tm.iter_mut().enumerate() {
            v.extend_from_slice(&y[s * 6 * h..(s + 1) * 6 * h]);
        }
    }
    let y = dit.modulate("dit.out_layer.modulation.out_layer", t_rows, n_t).ok()?;
    let fm: Vec<Vec<f32>> = (0..n_t).map(|s| y[s * 2 * h..(s + 1) * 2 * h].to_vec()).collect();
    crate::s2vgen::prof("modulations", t0);

    let mut aux = Aux::default();
    let mut visual = Vec::with_capacity(cfg.n_visual);
    let fail = |e: String| {
        eprintln!("s2v: Metal device path declined: {e}");
    };
    for (i, b) in dit.visual.iter().enumerate() {
        let pp = format!("dit.visual_transformer_blocks.{i}");
        let r = (|| -> Result<VisB, String> {
            let v2a = match &b.v2a {
                Some((n, a)) => Some((
                    attn_of(&model, &format!("{pp}.cross_attention_v2a"), a, h, &mut aux)?,
                    aux.push(&n.w),
                    aux.push(&n.b),
                )),
                None => None,
            };
            Ok(VisB {
                sa: attn_of(&model, &format!("{pp}.self_attention"), &b.sa, h, &mut aux)?,
                ca: attn_of(&model, &format!("{pp}.cross_attention"), &b.ca, h, &mut aux)?,
                v2a,
                w1: tens(&model, &format!("{pp}.feed_forward.in_layer.weight"), ff, h, &mut aux)?,
                w2: tens(&model, &format!("{pp}.feed_forward.out_layer.weight"), h, ff, &mut aux)?,
            })
        })();
        match r {
            Ok(v) => visual.push(v),
            Err(e) => {
                fail(e);
                return None;
            }
        }
    }
    let mut text = Vec::new();
    for (i, b) in dit.text.iter().enumerate() {
        let pp = format!("dit.text_transformer_blocks.{i}");
        let r = (|| -> Result<ShortB, String> {
            Ok(ShortB {
                a: attn_of(&model, &format!("{pp}.self_attention"), &b.attn, h, &mut aux)?,
                w1: tens(&model, &format!("{pp}.feed_forward.in_layer.weight"), ff, h, &mut aux)?,
                w2: tens(&model, &format!("{pp}.feed_forward.out_layer.weight"), h, ff, &mut aux)?,
                aff: None,
            })
        })();
        match r {
            Ok(v) => text.push(v),
            Err(e) => {
                fail(e);
                return None;
            }
        }
    }
    let mut audio = Vec::new();
    for (i, b) in dit.audio.iter().enumerate() {
        let pp = format!("dit.audio_transformer_blocks.{i}");
        let r = (|| -> Result<ShortB, String> {
            let aff = [aux.push(&b.sa_norm.w), aux.push(&b.sa_norm.b), aux.push(&b.ff_norm.w), aux.push(&b.ff_norm.b)];
            Ok(ShortB {
                a: attn_of(&model, &format!("{pp}.self_attention"), &b.attn, h, &mut aux)?,
                w1: tens(&model, &format!("{pp}.feed_forward.in_layer.weight"), ff, h, &mut aux)?,
                w2: tens(&model, &format!("{pp}.feed_forward.out_layer.weight"), h, ff, &mut aux)?,
                aff: Some(aff),
            })
        })();
        match r {
            Ok(v) => audio.push(v),
            Err(e) => {
                fail(e);
                return None;
            }
        }
    }
    let lin_b = |l: &s2v::Lin| l.b.clone().unwrap_or_else(|| vec![0.0; l.rows()]);
    let emb_b = aux.push(&lin_b(&dit.emb1));
    let mask0 = aux.push(&dit.mask[..h]);
    let s_vis = Seq::new(c, m, h, ff);
    let s_small = Seq::new(c, small, h, ff);
    let (vc, vs) = s2v::rope_visual(&cfg, g, sf);
    write_f32(&s_vis.cs, 0, &vc);
    write_f32(&s_vis.sn, 0, &vs);
    let crate::dit::Proj::F32 { w: emb, .. } = &dit.emb1.w else { return None };
    let crate::dit::Proj::F32 { w: head, .. } = &dit.out_w.w else { return None };
    crate::s2vgen::prof("metal program", t_all);
    Some(S2vDev {
        c,
        p,
        _model: model,
        arena,
        aux: buf_from(c, &aux.0),
        mods,
        visual,
        text,
        audio,
        s_vis,
        s_small,
        tkv: buf_zeroed(c, (nt.div_ceil(64) * 64 + 64) * 2 * h * 2),
        akv: buf_zeroed(c, (na.div_ceil(64) * 64 + 64) * 2 * h * 2),
        tcache: buf_zeroed(c, n_t * small * h * 2),
        t0: dit.embed_text(&cond.text),
        x0: buf_zeroed(c, g.clean().max(1) * h * 4),
        tok: buf_zeroed(c, g.nvis * 64 * 4),
        emb: buf_from(c, emb),
        head_w: buf_from(c, head),
        head_b: buf_from(c, &lin_b(&dit.out_w)),
        out: buf_zeroed(c, g.nvis * 320 * 4),
        h,
        ff,
        nh,
        g: *g,
        nt,
        na,
        emb_b,
        mask0,
        vm,
        tm,
        fm,
        text_off,
        fin_off,
        text_done: vec![false; n_t],
        audio_for: None,
        audio_feats: cond.audio.clone(),
        dit: dit as *const S2vDit,
        reference: cond.reference.clone(),
        cfg,
        prof: std::env::var("CMF_S2V_PROF").is_ok(),
    })
}

impl S2vDev {
    fn e<'r, 'a>(&'r self, rec: &'r mut Rec<'a>) -> E<'r, 'a> {
        E { rec, p: self.p, arena: &self.arena, aux: &self.aux, h: self.h, nh: self.nh }
    }

    fn short_block(&self, rec: &mut Rec<'_>, b: &ShortB, n: usize, mods_at: Option<usize>) {
        let (h, ff) = (self.h, self.ff);
        let s = &self.s_small;
        let aux = &self.aux;
        let mut e = self.e(rec);
        let m = &self.mods;
        match (mods_at, b.aff) {
            (Some(o), _) => e.row(s, 0, 0, n, 8, (m, 0), (m, o), (m, o + h), 0, 1.0),
            (None, Some(a)) => e.row(s, 0, 0, n, 16, (m, 0), (aux, a[1]), (aux, a[0]), 0, 1.0),
            _ => {}
        }
        e.gemm(&[b.a.q, b.a.k, b.a.v], &s.xn, h, &s.panel, &[0, h, 2 * h], 3 * h, n, true, 1.0);
        let qmul = std::f32::consts::LOG2_E / 128f32.sqrt();
        e.norm(&s.panel, &s.cs, &s.sn, n, 3 * h, 0, 3, b.a.qn, b.a.qb, qmul);
        e.norm(&s.panel, &s.cs, &s.sn, n, 3 * h, h, 3, b.a.kn, b.a.kb, 1.0);
        e.norm(&s.panel, &s.cs, &s.sn, n, 3 * h, 2 * h, 0, b.a.qn, b.a.vb, 1.0);
        e.flash(&s.panel, 3 * h, &s.att, 0, n, None);
        e.gemm(&[b.a.o], &s.att, h, &s.y, &[0], h, n, false, 1.0);
        match (mods_at, b.aff) {
            (Some(o), _) => e.row(s, 0, 0, n, 1 | 2 | 8, (m, o + 2 * h), (m, o + 3 * h), (m, o + 4 * h), b.a.ob, 1.0),
            (None, Some(a)) => e.row(s, 0, 0, n, 1 | 16, (m, 0), (aux, a[3]), (aux, a[2]), b.a.ob, 1.0),
            _ => {}
        }
        e.gemm(&[b.w1], &s.xn, h, &s.hid, &[0], ff, n, true, 1.0);
        e.gelu(&s.hid, n * ff / 4);
        e.gemm(&[b.w2], &s.hid, ff, &s.y, &[0], h, n, false, 1.0);
        // The feed-forward has no bias; the zero row of `aux` past the mask is
        // not needed because `mask0` is only read through `s2v_addv`.
        let zero = self.mask0 + h;
        match (mods_at, b.aff) {
            (Some(o), _) => e.row(s, 0, 0, n, 1 | 2, (m, o + 5 * h), (m, 0), (m, 0), zero, 1.0),
            _ => e.row(s, 0, 0, n, 1, (m, 0), (m, 0), (m, 0), zero, 1.0),
        }
    }

    fn run_audio(&mut self, chunk: usize) -> Result<(), String> {
        let h = self.h;
        // SAFETY: the DiT outlives the stepper.
        let dit = unsafe { &*self.dit };
        let a0 = dit.embed_audio(&self.audio_feats[chunk]);
        let n = a0.len() / h;
        write_f32(&self.s_small.x, 0, &a0);
        let (ac, asn) = s2v::rope_1d(&s2v::audio_rope_pos(n, self.g.lt));
        write_f32(&self.s_small.cs, 0, &ac);
        write_f32(&self.s_small.sn, 0, &asn);
        let mut rec = Rec::new(self.c);
        for b in &self.audio {
            self.short_block(&mut rec, b, n, None);
            rec.cut();
        }
        // The last residual is in x: f16 for the speech adapters' k|v GEMMs.
        {
            let mut e = self.e(&mut rec);
            let m = &self.mods;
            e.row(&self.s_small, 0, 0, n, 32, (m, 0), (m, 0), (m, 0), 0, 1.0);
        }
        if !rec.finish("audio") {
            return Err("a Metal command buffer failed".into());
        }
        // Keep the speech stream for the whole chunk in its own panel input.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.s_small.xn.contents() as *const u16,
                self.s_small.att.contents() as *mut u16,
                n * h,
            );
        }
        self.audio_for = Some(chunk);
        Ok(())
    }

    fn step_inner(&mut self, step: usize, chunk: usize, x: &[f32], memory: &[Vec<f32>; 3]) -> Result<Vec<f32>, String> {
        let (h, ff, g) = (self.h, self.ff, self.g);
        let tt = Instant::now();
        // Noisy rows and the per-step tables of this segment.
        for b in 0..self.cfg.n_visual {
            write_f32(&self.mods, b * 2 * 9 * h + 9 * h, &self.vm[step][b * 9 * h..(b + 1) * 9 * h]);
        }
        write_f32(&self.mods, self.text_off, &self.tm[step]);
        write_f32(&self.mods, self.fin_off, &self.fm[step]);
        // Speech stream of this chunk, kept as f16 in a private buffer.
        let speech_changed = self.audio_for != Some(chunk);
        if speech_changed {
            self.run_audio(chunk)?;
            // SAFETY: as in `run_audio`.
            let dit = unsafe { &*self.dit };
            let clean = dit.embed_clean(&self.reference, memory, &g);
            write_f32(&self.x0, 0, &clean);
        }
        let speech: Vec<u16> = unsafe {
            std::slice::from_raw_parts(self.s_small.att.contents() as *const u16, self.na * h).to_vec()
        };
        // Prompt stream of this segment, computed once and cached across chunks.
        let per = self.nt * h * 2;
        if !self.text_done[step] {
            write_f32(&self.s_small.x, 0, &self.t0);
            let (tc, tsn) = s2v::rope_1d(&(0..self.nt as i32).collect::<Vec<_>>());
            write_f32(&self.s_small.cs, 0, &tc);
            write_f32(&self.s_small.sn, 0, &tsn);
            let mut rec = Rec::new(self.c);
            for (bi, b) in self.text.iter().enumerate() {
                self.short_block(&mut rec, b, self.nt, Some(self.text_off + bi * 6 * h));
                rec.cut();
            }
            {
                let mut e = self.e(&mut rec);
                let m = &self.mods;
                e.row(&self.s_small, 0, 0, self.nt, 32, (m, 0), (m, 0), (m, 0), 0, 1.0);
            }
            if !rec.finish("text") {
                return Err("a Metal command buffer failed".into());
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.s_small.xn.contents() as *const u8,
                    (self.tcache.contents() as *mut u8).add(step * per),
                    per,
                );
            }
            self.text_done[step] = true;
        }
        // The two condition panels the GEMMs read as activations.
        let tcond = self.c._device.new_buffer_with_data(
            unsafe { (self.tcache.contents() as *const u8).add(step * per) } as *const c_void,
            ((self.nt.div_ceil(64) * 64 + 64) * h * 2).max(per) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let acond = {
            let mut v = speech;
            v.resize((self.na.div_ceil(64) * 64 + 64) * h, 0);
            self.c._device.new_buffer_with_data(
                v.as_ptr() as *const c_void,
                (v.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        // Stream start: clean rows, then the noisy rows' embedder.
        let s = &self.s_vis;
        unsafe {
            std::ptr::copy_nonoverlapping(self.x0.contents() as *const f32, s.x.contents() as *mut f32, g.clean() * h);
        }
        let toks = s2v::patchify(x, self.cfg.in_visual, 1, 2, g.lh, g.lw);
        write_f32(&self.tok, 0, &toks);
        let mut rec = Rec::new(self.c);
        {
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct Em {
                h: u32,
                n: u32,
                p0: u32,
                p1: u32,
            }
            let pe = Em { h: h as u32, n: g.nvis as u32, p0: 0, p1: 0 };
            let p = self.p;
            let enc = rec.enc("embed");
            enc.set_compute_pipeline_state(&p.embed);
            enc.set_buffer(0, Some(&self.tok), 0);
            enc.set_buffer(1, Some(&self.emb), 0);
            enc.set_buffer(2, Some(&s.x), (g.clean() * h * 4) as u64);
            set_p(enc, 3, &pe);
            enc.dispatch_threads(MTLSize::new(h as u64, g.nvis as u64, 1), MTLSize::new(256, 1, 1));
            enc.set_compute_pipeline_state(&p.addv);
            enc.set_buffer(0, Some(&s.x), (g.clean() * h * 4) as u64);
            enc.set_buffer(1, Some(&self.aux), (self.emb_b * 4) as u64);
            enc.set_buffer(2, Some(&self.aux), (self.mask0 * 4) as u64);
            set_p(enc, 3, &pe);
            enc.dispatch_threads(MTLSize::new(h as u64, g.nvis as u64, 1), MTLSize::new(256, 1, 1));
        }
        let qmul = std::f32::consts::LOG2_E / 128f32.sqrt();
        let m = g.total();
        let cl = g.clean();
        let (mods, aux) = (&self.mods, &self.aux);
        // Modulated pre-norm: clean rows and noisy rows are separate dispatches.
        let pre = |e: &mut E<'_, '_>, bi: usize, chunk_i: usize, mode: u32, gate: Option<usize>, bias: usize| {
            let at = |row: usize, c: usize| bi * 2 * 9 * h + row * 9 * h + c * h;
            for (row, (r0, n)) in [(0usize, (0usize, cl)), (1usize, (cl, g.nvis))] {
                let gw = gate.map_or(0, |gc| at(row, gc));
                e.row(s, r0, r0, n, mode, (mods, gw), (mods, at(row, chunk_i)), (mods, at(row, chunk_i + 1)), bias, 1.0);
            }
        };
        let zero = self.mask0 + h;
        for (bi, b) in self.visual.iter().enumerate() {
            let mut e = self.e(&mut rec);
            // self-attention
            pre(&mut e, bi, 0, 8, None, zero);
            e.gemm(&[b.sa.q, b.sa.k, b.sa.v], &s.xn, h, &s.panel, &[0, h, 2 * h], 3 * h, m, true, 1.0);
            e.norm(&s.panel, &s.cs, &s.sn, m, 3 * h, 0, 3, b.sa.qn, b.sa.qb, qmul);
            e.norm(&s.panel, &s.cs, &s.sn, m, 3 * h, h, 3, b.sa.kn, b.sa.kb, 1.0);
            e.norm(&s.panel, &s.cs, &s.sn, m, 3 * h, 2 * h, 0, b.sa.qn, b.sa.vb, 1.0);
            e.flash(&s.panel, 3 * h, &s.att, 0, m, None);
            e.gemm(&[b.sa.o], &s.att, h, &s.y, &[0], h, m, false, 1.0);
            // residual + the prompt attention's pre-norm
            pre(&mut e, bi, 3, 1 | 2 | 8, Some(2), b.sa.ob);
            e.gemm(&[b.ca.q], &s.xn, h, &s.panel, &[0], 3 * h, m, true, 1.0);
            e.norm(&s.panel, &s.cs, &s.sn, m, 3 * h, 0, 1, b.ca.qn, b.ca.qb, qmul);
            e.gemm(&[b.ca.k, b.ca.v], &tcond, h, &self.tkv, &[0, h], 2 * h, self.nt, true, 1.0);
            e.norm(&self.tkv, &s.cs, &s.sn, self.nt, 2 * h, 0, 1, b.ca.kn, b.ca.kb, 1.0);
            e.norm(&self.tkv, &s.cs, &s.sn, self.nt, 2 * h, h, 0, b.ca.qn, b.ca.vb, 1.0);
            e.flash(&s.panel, 3 * h, &s.att, 0, m, Some((&self.tkv, 0, self.nt)));
            e.gemm(&[b.ca.o], &s.att, h, &s.y, &[0], h, m, false, 1.0);
            if let Some((a, nw, nb)) = &b.v2a {
                pre(&mut e, bi, 6, 1 | 2, Some(5), b.ca.ob);
                e.row(s, cl, 0, g.nvis, 16, (mods, 0), (aux, *nb), (aux, *nw), zero, 1.0);
                e.gemm(&[a.q], &s.xn, h, &s.panel, &[0], 3 * h, g.nvis, true, 1.0);
                e.norm(&s.panel, &s.cs, &s.sn, g.nvis, 3 * h, 0, 1, a.qn, a.qb, qmul);
                e.gemm(&[a.k, a.v], &acond, h, &self.akv, &[0, h], 2 * h, self.na, true, 1.0);
                e.norm(&self.akv, &s.cs, &s.sn, self.na, 2 * h, 0, 1, a.kn, a.kb, 1.0);
                e.norm(&self.akv, &s.cs, &s.sn, self.na, 2 * h, h, 0, a.qn, a.vb, 1.0);
                let pf_rows = g.per_frame();
                for (fi, (k0, kn)) in audio_groups(self.na, g.lt).into_iter().enumerate() {
                    e.flash(&s.panel, 3 * h, &s.att, fi * pf_rows, pf_rows, Some((&self.akv, k0 * 2 * h * 2, kn)));
                }
                e.gemm(&[a.o], &s.att, h, &s.y, &[0], h, g.nvis, false, 1.0);
                e.row(s, cl, 0, g.nvis, 1, (mods, 0), (mods, 0), (mods, 0), a.ob, 1.0);
                pre(&mut e, bi, 6, 8, None, zero);
            } else {
                pre(&mut e, bi, 6, 1 | 2 | 8, Some(5), b.ca.ob);
            }
            // feed-forward
            e.gemm(&[b.w1], &s.xn, h, &s.hid, &[0], ff, m, true, 1.0);
            e.gelu(&s.hid, m * ff / 4);
            e.gemm(&[b.w2], &s.hid, ff, &s.y, &[0], h, m, false, 1.0);
            pre(&mut e, bi, 7, 1 | 2, Some(8), zero);
            rec.cut();
        }
        {
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct Hd {
                h: u32,
                row0: u32,
                eps: f32,
                p: u32,
            }
            let ph = Hd { h: h as u32, row0: cl as u32, eps: 1e-5, p: 0 };
            let p = self.p;
            let enc = rec.enc("head");
            enc.set_compute_pipeline_state(&p.head);
            enc.set_buffer(0, Some(&s.x), 0);
            enc.set_buffer(1, Some(mods), (self.fin_off * 4) as u64);
            enc.set_buffer(2, Some(mods), ((self.fin_off + h) * 4) as u64);
            enc.set_buffer(3, Some(&self.head_w), 0);
            enc.set_buffer(4, Some(&self.head_b), 0);
            enc.set_buffer(5, Some(&self.out), 0);
            set_p(enc, 6, &ph);
            enc.dispatch_thread_groups(MTLSize::new(g.nvis as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        if !rec.finish("step") {
            return Err("a Metal command buffer failed".into());
        }
        let y = unsafe { std::slice::from_raw_parts(self.out.contents() as *const f32, g.nvis * 320) };
        if self.prof {
            eprintln!("s2v metal step {step}: {:.1} s", tt.elapsed().as_secs_f64());
        }
        Ok(s2v::unpatchify_out(y, &g, self.cfg.out_visual))
    }
}

impl crate::gpu::S2vStepper for S2vDev {
    fn step(&mut self, step: usize, chunk: usize, x: &[f32], memory: &[Vec<f32>; 3]) -> Result<Vec<f32>, String> {
        self.step_inner(step, chunk, x, memory)
    }
}

// ───────────────────────────── VAE decoder ─────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PVc {
    n: u32,
    rows: u32,
    k: u32,
    ic: u32,
    f: u32,
    h: u32,
    w: u32,
    taps: u32,
    up_s: u32,
    up_t: u32,
    epi: u32,
    ldy: u32,
    has_bias: u32,
    has_res: u32,
    mul: f32,
    ldx: u32,
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

struct VC {
    w: Buffer,
    b: Buffer,
    ocp: usize,
    icp: usize,
    taps: usize,
}

struct VN {
    w: Buffer,
    b: Buffer,
}

struct VR {
    n1: VN,
    c1: VC,
    n2: VN,
    c2: VC,
    sc: Option<VC>,
    cin: usize,
    cout: usize,
}

struct VDev {
    conv_in: VC,
    mid0: VR,
    gn: VN,
    q: VC,
    k: VC,
    v: VC,
    o: VC,
    mid1: VR,
    ac: usize,
    ups: Vec<(Vec<VR>, Option<(VC, [usize; 3])>)>,
    norm: VN,
    conv_out: VC,
}

struct VState(Option<VDev>);
unsafe impl Send for VState {}
static VSTATE: Mutex<VState> = Mutex::new(VState(None));

pub(crate) fn release() {
    if let Ok(mut st) = VSTATE.lock() {
        st.0 = None;
    }
}

fn half_buf(c: &Ctx, v: &[f32]) -> Buffer {
    let h: Vec<u16> = v.iter().map(|&x| cortiq_core::quant::f32_to_f16(x)).collect();
    c._device.new_buffer_with_data(h.as_ptr() as *const c_void, (h.len().max(8) * 2) as u64, MTLResourceOptions::StorageModeShared)
}

fn vc(c: &Ctx, cv: &Conv) -> VC {
    let taps = cv.k[0] * cv.k[1] * cv.k[2];
    let ocp = cv.cout.div_ceil(64) * 64;
    let icp = cv.cin.div_ceil(32) * 32;
    let mut w = vec![0f32; ocp * taps * icp];
    for o in 0..cv.cout {
        for i in 0..cv.cin {
            for t in 0..taps {
                w[(o * taps + t) * icp + i] = cv.w[(o * cv.cin + i) * taps + t];
            }
        }
    }
    let mut b = vec![0f32; ocp];
    b[..cv.cout].copy_from_slice(&cv.b);
    VC { w: half_buf(c, &w), b: buf_from(c, &b), ocp, icp, taps }
}

fn vl(c: &Ctx, l: &(Vec<f32>, Vec<f32>), scale: f32) -> VC {
    let ch = l.1.len();
    let ws: Vec<f32> = l.0.iter().map(|v| v * scale).collect();
    let bs: Vec<f32> = l.1.iter().map(|v| v * scale).collect();
    VC { w: half_buf(c, &ws), b: buf_from(c, &bs), ocp: ch, icp: ch, taps: 0 }
}

fn vn(c: &Ctx, g: &GroupNorm) -> VN {
    VN { w: buf_from(c, &g.w), b: buf_from(c, &g.b) }
}

fn vr(c: &Ctx, r: &Res) -> VR {
    VR {
        n1: vn(c, &r.n1),
        c1: vc(c, &r.c1),
        n2: vn(c, &r.n2),
        c2: vc(c, &r.c2),
        sc: r.short.as_ref().map(|s| vc(c, s)),
        cin: r.c1.cin,
        cout: r.c1.cout,
    }
}

fn build_vae(c: &Ctx, v: &S2vVae) -> VDev {
    let m: &Mid = &v.mid;
    let ac = m.q.1.len();
    VDev {
        conv_in: vc(c, &v.conv_in),
        mid0: vr(c, &m.r0),
        gn: vn(c, &m.gn),
        q: vl(c, &m.q, 1.0 / (ac as f32).sqrt()),
        k: vl(c, &m.k, 1.0),
        v: vl(c, &m.v, 1.0),
        o: vl(c, &m.o, 1.0),
        mid1: vr(c, &m.r1),
        ac,
        ups: v
            .ups
            .iter()
            .map(|u| (u.res.iter().map(|r| vr(c, r)).collect(), u.conv.as_ref().map(|(cv, f)| (vc(c, cv), *f))))
            .collect(),
        norm: vn(c, &v.norm_out),
        conv_out: vc(c, &v.conv_out),
    }
}

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
        self.done.clear();
        ok
    }

    /// `Y = conv(X)` (+bias) (+R); `up` = (space, time) nearest upsample of the source.
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, cv: &VC, x: &Buffer, y: &Buffer, res: Option<&Buffer>, fhw: (usize, usize, usize), up: (u32, bool), epi: u32, ldy: usize, plain: Option<(usize, usize)>) {
        let (n, ldx, k) = match plain {
            Some((n, ldx)) => (n, ldx, ldx),
            None => (fhw.0 * fhw.1 * fhw.2, cv.icp, cv.taps * cv.icp),
        };
        let pv = PVc {
            n: n as u32,
            rows: (if epi == 2 { cv.ocp } else { ldy }) as u32,
            k: k as u32,
            ic: cv.icp as u32,
            f: fhw.0 as u32,
            h: fhw.1 as u32,
            w: fhw.2 as u32,
            taps: cv.taps as u32,
            up_s: up.0,
            up_t: up.1 as u32,
            epi,
            ldy: ldy as u32,
            has_bias: 1,
            has_res: res.is_some() as u32,
            mul: 1.0,
            ldx: ldx as u32,
        };
        let p = self.p;
        let enc = self.enc();
        enc.set_compute_pipeline_state(&p.vconv);
        enc.set_buffer(0, Some(&cv.w), 0);
        enc.set_buffer(1, Some(&cv.b), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(y), 0);
        enc.set_buffer(4, Some(res.unwrap_or(y)), 0);
        set_p(enc, 5, &pv);
        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(64) as u64, (cv.ocp / 64) as u64, 1), MTLSize::new(128, 1, 1));
        self.cut();
    }

    /// Plain GEMM with an explicit weight: `Y[n][rows] = X[n][k]·Wᵀ`.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&mut self, w: &Buffer, rows: usize, k: usize, x: (&Buffer, u64), ldx: usize, y: (&Buffer, u64), ldy: usize, n: usize, epi: u32) {
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
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(y.0), y.1);
        enc.set_buffer(2, Some(x.0), x.1);
        enc.set_buffer(3, Some(y.0), y.1);
        enc.set_buffer(4, Some(y.0), y.1);
        set_p(enc, 5, &pv);
        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(64) as u64, (rows / 64) as u64, 1), MTLSize::new(128, 1, 1));
        self.cut();
    }

    #[allow(clippy::too_many_arguments)]
    fn gn(&mut self, nm: &VN, x: &Buffer, y: &Buffer, n: usize, ch: usize, silu: bool, half_out: bool) {
        let nblk = n.div_ceil(1024).clamp(1, 1024);
        let mut pg = PGn { n: n as u32, c: ch as u32, groups: 32, nblk: nblk as u32, pass: 0, eps: 1e-6, silu: silu as u32, half_out: half_out as u32 };
        let p = self.p;
        let (part, stat) = (self.part.clone(), self.stat.clone());
        let enc = self.enc();
        for pass in 0..2u32 {
            pg.pass = pass;
            enc.set_compute_pipeline_state(&p.gnpart);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&part), 0);
            enc.set_buffer(2, Some(&stat), 0);
            set_p(enc, 3, &pg);
            enc.dispatch_thread_groups(MTLSize::new(32, nblk as u64, 1), MTLSize::new(256, 1, 1));
            enc.set_compute_pipeline_state(&p.gnfin);
            enc.set_buffer(0, Some(&part), 0);
            enc.set_buffer(1, Some(&stat), 0);
            set_p(enc, 2, &pg);
            enc.dispatch_thread_groups(MTLSize::new(32, 1, 1), MTLSize::new(32, 1, 1));
        }
        enc.set_compute_pipeline_state(&p.gnapply);
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
        enc.set_compute_pipeline_state(&p.cvt);
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

fn resnet(r: &mut VRec, b: &Bufs, rs: &VR, fhw: (usize, usize, usize)) {
    let n = fhw.0 * fhw.1 * fhw.2;
    r.gn(&rs.n1, &b.xa, &b.hh, n, rs.cin, true, true);
    r.conv(&rs.c1, &b.hh, &b.t, None, fhw, (0, false), 0, rs.cout, None);
    if let Some(sc) = &rs.sc {
        r.cvt(&b.xa, &b.hh, n * rs.cin);
        r.conv(sc, &b.hh, &b.xa, None, fhw, (0, false), 0, rs.cout, None);
    }
    r.gn(&rs.n2, &b.t, &b.hh, n, rs.cout, true, true);
    r.conv(&rs.c2, &b.hh, &b.xa, Some(&b.xa), fhw, (0, false), 0, rs.cout, None);
}

/// The VAE's mid attention: one query frame at a time.
#[allow(clippy::too_many_arguments)]
fn mid_attention(c: &Ctx, r: &mut VRec, b: &Bufs, gn: &VN, w: (&VC, &VC, &VC, &VC), ac: usize, f0: usize, pf: usize) -> Option<()> {
    let n0 = f0 * pf;
    let n0p = n0.div_ceil(64) * 64;
    let (qb, kb, vt, ob) = (
        buf_zeroed(c, n0p * ac * 2),
        buf_zeroed(c, n0p * ac * 2),
        buf_zeroed(c, n0p * ac * 2),
        buf_zeroed(c, n0p * ac * 2),
    );
    let sb = buf_zeroed(c, pf * n0p * 4);
    let pb = buf_zeroed(c, pf * n0p * 2);
    r.gn(gn, &b.xa, &b.hh, n0, ac, false, true);
    r.conv(w.0, &b.hh, &qb, None, (0, 0, 0), (0, false), 1, ac, Some((n0, ac)));
    r.conv(w.1, &b.hh, &kb, None, (0, 0, 0), (0, false), 1, ac, Some((n0, ac)));
    r.conv(w.2, &b.hh, &vt, None, (0, 0, 0), (0, false), 2, n0p, Some((n0, ac)));
    for fr in 0..f0 {
        let live = (fr + 1) * pf;
        r.gemm(&kb, live.div_ceil(64) * 64, ac, (&qb, (fr * pf * ac * 2) as u64), ac, (&sb, 0), n0p, pf, 0);
        {
            let pp = r.p;
            let (nv, ld) = (live as u32, n0p as u32);
            let enc = r.enc();
            enc.set_compute_pipeline_state(&pp.softmax);
            enc.set_buffer(0, Some(&sb), 0);
            enc.set_buffer(1, Some(&pb), 0);
            set_p(enc, 2, &nv);
            set_p(enc, 3, &ld);
            enc.dispatch_thread_groups(MTLSize::new(pf as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        r.gemm(&vt, ac, n0p, (&pb, 0), n0p, (&ob, (fr * pf * ac * 2) as u64), ac, pf, 1);
    }
    r.conv(w.3, &ob, &b.xa, Some(&b.xa), (0, 0, 0), (0, false), 0, ac, Some((n0, ac)));
    Some(())
}

/// One temporal tile `[16, F, H, W]`, past the post-quant conv.
fn decode_tile(c: &Ctx, p: &'static Pipes, d: &VDev, z: &Vol) -> Option<Vol> {
    let (f0, h0, w0) = (z.f, z.h, z.w);
    let n0 = z.positions();
    let pf = h0 * w0;
    let mut cmax = n0 * 512;
    {
        let (mut ff, mut hh, mut ww) = (f0, h0, w0);
        for (res, up) in &d.ups {
            for r in res {
                cmax = cmax.max(ff * hh * ww * r.cin.max(r.cout));
            }
            if let Some((_, fac)) = up {
                ff = if fac[0] == 2 { 1 + (ff - 1) * 2 } else { ff };
                hh *= fac[1];
                ww *= fac[2];
                cmax = cmax.max(ff * hh * ww * d.ups[0].0[0].cin.min(256));
            }
        }
        cmax = cmax.max(ff * hh * ww * 256).max(ff * hh * ww * 64);
    }
    if (cmax * 4) as u64 > c._device.max_buffer_length() {
        eprintln!("s2v: the VAE buffer ({:.1} GB) is past maxBufferLength: decoding on the CPU", (cmax * 4) as f64 / 1e9);
        return None;
    }
    let b = Bufs { xa: buf_zeroed(c, cmax * 4), t: buf_zeroed(c, cmax * 4), hh: buf_zeroed(c, cmax * 2) };
    {
        let icp = d.conv_in.icp;
        let mut zh = vec![0u16; n0 * icp];
        for ch in 0..z.c {
            for px in 0..n0 {
                zh[px * icp + ch] = cortiq_core::quant::f32_to_f16(z.data[ch * n0 + px]);
            }
        }
        unsafe { std::ptr::copy_nonoverlapping(zh.as_ptr(), b.hh.contents() as *mut u16, zh.len()) };
    }
    let ac = d.ac;
    let mut r = VRec { c, p, done: Vec::new(), cur: None, part: buf_zeroed(c, 32 * 1024 * 4), stat: buf_zeroed(c, 32 * 2 * 4) };
    let fhw0 = (f0, h0, w0);
    r.conv(&d.conv_in, &b.hh, &b.xa, None, fhw0, (0, false), 0, d.conv_in.ocp, None);
    resnet(&mut r, &b, &d.mid0, fhw0);
    mid_attention(c, &mut r, &b, &d.gn, (&d.q, &d.k, &d.v, &d.o), ac, f0, pf)?;
    resnet(&mut r, &b, &d.mid1, fhw0);
    let (mut ff, mut hh, mut ww) = (f0, h0, w0);
    let mut ch = d.mid1.cout;
    let (mut xa, mut tt) = (b.xa.clone(), b.t.clone());
    for (res, up) in &d.ups {
        for rs in res {
            let bb = Bufs { xa: xa.clone(), t: tt.clone(), hh: b.hh.clone() };
            resnet(&mut r, &bb, rs, (ff, hh, ww));
            ch = rs.cout;
        }
        if let Some((u, fac)) = up {
            r.cvt(&xa, &b.hh, ff * hh * ww * ch);
            ff = if fac[0] == 2 { 1 + (ff - 1) * 2 } else { ff };
            hh *= fac[1];
            ww *= fac[2];
            r.conv(u, &b.hh, &tt, None, (ff, hh, ww), (u32::from(fac[1] == 2), fac[0] == 2), 0, u.ocp, None);
            std::mem::swap(&mut xa, &mut tt);
            ch = u.ocp;
        }
    }
    let n = ff * hh * ww;
    r.gn(&d.norm, &xa, &b.hh, n, ch, true, true);
    r.conv(&d.conv_out, &b.hh, &tt, None, (ff, hh, ww), (0, false), 0, 64, None);
    let out = buf_zeroed(c, 3 * n * 4);
    {
        let pp = r.p;
        let nu = n as u32;
        let enc = r.enc();
        enc.set_compute_pipeline_state(&pp.rgb);
        enc.set_buffer(0, Some(&tt), 0);
        enc.set_buffer(1, Some(&out), 0);
        set_p(enc, 2, &nu);
        enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
    }
    if !r.finish() {
        return None;
    }
    let mut v = Vol::zeros(3, ff, hh, ww);
    let src = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, 3 * n) };
    v.data.copy_from_slice(src);
    Some(v)
}

/// The reference frame through the encoder: `[3, 1, H, W]` → `(mean, logvar)`.
pub(crate) fn encode(enc: &S2vVaeEncoder, img: &Vol) -> Option<(Vol, Vol)> {
    if std::env::var("CMF_S2V_VAE_METAL").as_deref() == Ok("0") || img.f != 1 {
        return None;
    }
    let c = super::ctx()?;
    let p = pipes(c)?;
    let (h0, w0) = (img.h, img.w);
    let conv_in = vc(c, &enc.conv_in);
    let downs: Vec<(Vec<VR>, Option<VC>)> = enc
        .downs
        .iter()
        .map(|(res, d)| (res.iter().map(|r| vr(c, r)).collect(), d.as_ref().map(|cv| vc(c, cv))))
        .collect();
    let m: &Mid = &enc.mid;
    let ac = m.q.1.len();
    let (mr0, gn, q, k, v, o, mr1) = (
        vr(c, &m.r0),
        vn(c, &m.gn),
        vl(c, &m.q, 1.0 / (ac as f32).sqrt()),
        vl(c, &m.k, 1.0),
        vl(c, &m.v, 1.0),
        vl(c, &m.o, 1.0),
        vr(c, &m.r1),
    );
    let norm = vn(c, &enc.norm_out);
    let conv_out = vc(c, &enc.conv_out);
    let n0 = h0 * w0;
    let cmax = n0 * 512;
    let b = Bufs { xa: buf_zeroed(c, cmax * 4), t: buf_zeroed(c, cmax * 4), hh: buf_zeroed(c, cmax * 2) };
    {
        let icp = conv_in.icp;
        let mut zh = vec![0u16; n0 * icp];
        for ch in 0..3 {
            for px in 0..n0 {
                zh[px * icp + ch] = cortiq_core::quant::f32_to_f16(img.data[ch * n0 + px]);
            }
        }
        unsafe { std::ptr::copy_nonoverlapping(zh.as_ptr(), b.hh.contents() as *mut u16, zh.len()) };
    }
    let mut r = VRec { c, p, done: Vec::new(), cur: None, part: buf_zeroed(c, 32 * 1024 * 4), stat: buf_zeroed(c, 32 * 2 * 4) };
    r.conv(&conv_in, &b.hh, &b.xa, None, (1, h0, w0), (0, false), 0, conv_in.ocp, None);
    let (mut hh, mut ww) = (h0, w0);
    let mut ch = conv_in.ocp;
    let (mut xa, mut tt) = (b.xa.clone(), b.t.clone());
    for (res, dn) in &downs {
        for rs in res {
            let bb = Bufs { xa: xa.clone(), t: tt.clone(), hh: b.hh.clone() };
            resnet(&mut r, &bb, rs, (1, hh, ww));
            ch = rs.cout;
        }
        if let Some(d) = dn {
            r.cvt(&xa, &b.hh, hh * ww * ch);
            hh /= 2;
            ww /= 2;
            r.conv(d, &b.hh, &tt, None, (1, hh, ww), (2, false), 0, d.ocp, None);
            std::mem::swap(&mut xa, &mut tt);
            ch = d.ocp;
        }
    }
    let bb = Bufs { xa: xa.clone(), t: tt.clone(), hh: b.hh.clone() };
    resnet(&mut r, &bb, &mr0, (1, hh, ww));
    mid_attention(c, &mut r, &bb, &gn, (&q, &k, &v, &o), ac, 1, hh * ww)?;
    resnet(&mut r, &bb, &mr1, (1, hh, ww));
    let n = hh * ww;
    r.gn(&norm, &xa, &b.hh, n, ac, true, true);
    r.conv(&conv_out, &b.hh, &tt, None, (1, hh, ww), (0, false), 0, conv_out.ocp, None);
    if !r.finish() {
        return None;
    }
    let cout = enc.conv_out.cout;
    let raw = unsafe { std::slice::from_raw_parts(tt.contents() as *const f32, n * conv_out.ocp) };
    let mut h = Vol::zeros(cout, 1, hh, ww);
    for co in 0..cout {
        for i in 0..n {
            h.data[co * n + i] = raw[i * conv_out.ocp + co];
        }
    }
    Some(enc.finish_host(&h))
}

pub(crate) fn decode(vae: &S2vVae, z: &Vol) -> Option<Vol> {
    if std::env::var("CMF_S2V_VAE_METAL").as_deref() == Ok("0") {
        return None;
    }
    let c = super::ctx()?;
    let p = pipes(c)?;
    let t0 = Instant::now();
    let mut st = VSTATE.lock().ok()?;
    if st.0.is_none() {
        st.0 = Some(build_vae(c, vae));
        crate::s2vgen::prof("vae planes", t0);
    }
    let d = st.0.as_ref()?;
    let mut failed = false;
    let v = crate::s2v_vae::tiled(z, &mut |tile| {
        let pq = vae.post_quant_host(tile);
        match decode_tile(c, p, d, &pq) {
            Some(x) => x,
            None => {
                failed = true;
                Vol::zeros(3, 1, tile.h * 8, tile.w * 8)
            }
        }
    });
    if prof_on() {
        crate::s2vgen::prof("vae metal decode", t0);
    }
    if failed { None } else { Some(v) }
}
