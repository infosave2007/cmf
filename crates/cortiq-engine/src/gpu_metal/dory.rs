//! Dory's token path on Metal, the twin of `gpu_wgpu/dory.rs` (`dory` has the
//! semantics). Weights stay in the file mapping; a sliding ring holds the window,
//! a chunk and a turn's rewind.

use super::{Ctx, O1MetalDev, ProjKind, WeightArena, ctx, file_buffer, note_weight_bytes, o1_ensure_metal, set_bytes_t};
use crate::dory::{Dory, ExitRate, Gate, KvMode, Layer, Want, top_prob};
use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MSL: &str = include_str!("dory_msl.metal");
/// A prompt this long seals before it ends: 8 KB a row per full layer and loop.
const SEAL_MAX: usize = 32768;
/// One row in this many keeps its queries for the seal (`8u` in the shader).
const QT_STRIDE: usize = 8;
/// Rows of a prompt chunk: `q4tp_mul_mm_n8` takes eight a dispatch, q8's GEMM tiles 32.
const CHUNK: usize = 32;
const CHUNK_Q4: usize = 8;
/// Pieces of a value sum (`8u` in the shader).
const CH: usize = 8;

struct Pipes {
    prep: ComputePipelineState,
    score: ComputePipelineState,
    soft: ComputePipelineState,
    sum: ComputePipelineState,
    gate: ComputePipelineState,
    silu: ComputePipelineState,
    slerp: ComputePipelineState,
    mix: ComputePipelineState,
    qcopy: ComputePipelineState,
    o1_back: ComputePipelineState,
    os: ComputePipelineState,
    ol: ComputePipelineState,
    omax: ComputePipelineState,
    ofar: ComputePipelineState,
    osoft: ComputePipelineState,
    osum: ComputePipelineState,
    ogate: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c._device.new_library_with_source(MSL, &opts).map_err(|e| format!("dory MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            c._device.new_compute_pipeline_state_with_function(&f).map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            prep: pso("dory_prep")?,
            score: pso("dory_score")?,
            soft: pso("dory_soft")?,
            sum: pso("dory_sum")?,
            gate: pso("dory_gate")?,
            silu: pso("dory_silu")?,
            slerp: pso("dory_slerp")?,
            mix: pso("dory_mix")?,
            qcopy: pso("dory_qcopy")?,
            o1_back: pso("dory_o1_back")?,
            os: pso("dory_os")?,
            ol: pso("dory_ol")?,
            omax: pso("dory_omax")?,
            ofar: pso("dory_ofar")?,
            osoft: pso("dory_osoft")?,
            osum: pso("dory_osum")?,
            ogate: pso("dory_ogate")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            tracing::warn!("{e}");
            None
        }
    }
}

#[derive(Clone)]
enum Mat {
    Q4 { abs: usize, rows: usize, cols: usize },
    Q8 { abs: usize, rows: usize, cols: usize, rs: Buffer, col: Buffer },
}

struct AttnDev {
    q: Mat,
    k: Mat,
    v: Mat,
    g: Mat,
    o: Mat,
    sq: Buffer,
    sg: Buffer,
    alpha: Buffer,
    sliding: bool,
    ordinal: usize,
}

struct MlpDev {
    gate: Mat,
    up: Mat,
    down: Mat,
    sg: Buffer,
    su: Buffer,
    alpha: Buffer,
}

enum LayerDev {
    Attn(Box<AttnDev>),
    Mlp(Box<MlpDev>),
}

struct Slot {
    k: Buffer,
    v: Buffer,
    cap: usize,
}

struct Device {
    fbuf: Arc<WeightArena>,
    layers: Vec<LayerDev>,
    head: Mat,
    slots: Vec<Vec<Slot>>,
    h: Buffer,
    input: Buffer,
    state: Buffer,
    hlast: Buffer,
    yq: Buffer,
    yk: Buffer,
    yv: Buffer,
    yg: Buffer,
    qo: Buffer,
    z: Buffer,
    pz: Buffer,
    sinv: Buffer,
    fg: Buffer,
    fu: Buffer,
    o: Buffer,
    logits: Buffer,
    xsc: Buffer,
    scores: Buffer,
    scores_cap: usize,
    /// Rotary table per layer kind: sliding, full.
    rope: [Buffer; 2],
    gate_tokens: u64,
    gate_one: u64,
    exits: ExitRate,
    id: u64,
    sealed: bool,
    o1_refused: bool,
    seal_epoch: u64,
    /// The queries of a full slot up to the seal.
    qtrace: HashMap<(usize, usize), Buffer>,
    far: HashMap<(usize, usize), O1MetalDev>,
    kt: Buffer,
    vt: Buffer,
    /// The sealed state's sink rows and its rectifier: rows chosen at the seal can pass
    /// what `nsrect` holds.
    far_ns: usize,
    far_rect: u32,
    /// The far stages': landmark scores, near maxima, per head and gains.
    ofs: Buffer,
    onm: Buffer,
    ohd: Buffer,
    ogn: Buffer,
}

static DEVICES: Mutex<Option<HashMap<u64, Device>>> = Mutex::new(None);

fn shared(c: &Ctx, bytes: usize) -> Buffer {
    c._device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
}

fn up(c: &Ctx, v: &[f32]) -> Buffer {
    c._device.new_buffer_with_data(v.as_ptr() as *const std::ffi::c_void, (v.len().max(1) * 4) as u64, MTLResourceOptions::StorageModeShared)
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

/// Can the device path run this file?
pub(crate) fn admits(m: &Dory) -> bool {
    let c = &m.cfg;
    if c.nh != c.nk * 4 || c.hd != 256 || c.hidden % 64 != 0 || c.ffn % 64 != 0 {
        return false;
    }
    let plain = |t: &crate::qtensor::QTensor| t.q4tp_mapped().is_some() || t.q8_2f_codes().is_some();
    plain(&m.head)
        && m.layers.iter().all(|l| match l {
            Layer::Attn(w) => plain(&w.q) && plain(&w.k) && plain(&w.v) && plain(&w.g) && plain(&w.o),
            Layer::Mlp(w) => plain(&w.gate) && plain(&w.up) && plain(&w.down),
        })
}

fn build(c: &Ctx, model: &Arc<cortiq_core::CmfModel>, m: &Dory) -> Option<Device> {
    pipes(c)?;
    super::moe_q8::pipes(c)?;
    let (fbuf, _) = file_buffer(c, model)?;
    let cf = &m.cfg;
    let mut layers = Vec::with_capacity(m.layers.len());
    for l in &m.layers {
        layers.push(match l {
            Layer::Attn(w) => LayerDev::Attn(Box::new(AttnDev {
                q: mat(c, model, &w.q)?,
                k: mat(c, model, &w.k)?,
                v: mat(c, model, &w.v)?,
                g: mat(c, model, &w.g)?,
                o: mat(c, model, &w.o)?,
                sq: up(c, &w.sq),
                sg: up(c, &w.sg),
                alpha: up(c, &w.alpha),
                sliding: w.sliding,
                ordinal: w.ordinal,
            })),
            Layer::Mlp(w) => LayerDev::Mlp(Box::new(MlpDev {
                gate: mat(c, model, &w.gate)?,
                up: mat(c, model, &w.up)?,
                down: mat(c, model, &w.down)?,
                sg: up(c, &w.sg),
                su: up(c, &w.su),
                alpha: up(c, &w.alpha),
            })),
        });
    }
    let k = CHUNK;
    let q_size = cf.nh * cf.hd;
    let kv_row = cf.nk * cf.hd;
    let hk = k * cf.hidden * 4;
    Some(Device {
        fbuf,
        head: mat(c, model, &m.head)?,
        layers,
        slots: (0..m.attn_layers).map(|_| Vec::new()).collect(),
        h: shared(c, hk),
        input: shared(c, hk),
        state: shared(c, hk),
        hlast: shared(c, cf.hidden * 4),
        yq: shared(c, k * q_size * 4),
        yk: shared(c, k * kv_row * 4),
        yv: shared(c, k * kv_row * 4),
        yg: shared(c, k * q_size * 4),
        qo: shared(c, k * q_size * 4),
        z: shared(c, k * q_size * 4),
        pz: shared(c, k * q_size * CH * 4),
        sinv: shared(c, k * cf.nk * 16),
        fg: shared(c, k * cf.ffn * 4),
        fu: shared(c, k * cf.ffn * 4),
        o: shared(c, hk),
        logits: shared(c, k * cf.vocab * 4),
        xsc: up(c, &vec![1.0f32; k]),
        scores: shared(c, (cf.window + k) * k * cf.nk * 16),
        scores_cap: cf.window + k,
        rope: [shared(c, k * cf.hd * 4), shared(c, k * cf.hd * 4)],
        gate_tokens: 0,
        gate_one: 0,
        exits: ExitRate::default(),
        id: 0,
        sealed: false,
        o1_refused: false,
        seal_epoch: 0,
        qtrace: HashMap::new(),
        far: HashMap::new(),
        kt: shared(c, kv_row * 4),
        vt: shared(c, kv_row * 4),
        far_ns: 0,
        far_rect: 0,
        ofs: shared(c, cf.nk * 32 * 16),
        onm: shared(c, cf.nk * 16),
        ohd: shared(c, cf.nh * 16),
        ogn: shared(c, cf.nh * 32 * 4),
    })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct P {
    t: u32,
    pos0: u32,
    cap: u32,
    window: u32,
    nh: u32,
    nk: u32,
    hd: u32,
    hidden: u32,
    ffn: u32,
    qsize: u32,
    ncap: u32,
    ch: u32,
    scale: f32,
    eps: f32,
    pad0: u32,
    pad1: u32,
}

fn enc_mv(c: &Ctx, enc: &metal::ComputeCommandEncoderRef, fbuf: &WeightArena, m: &Mat, xs: &Buffer, y: &Buffer) {
    match m {
        Mat::Q4 { abs, rows, cols } => {
            note_weight_bytes(&ProjKind::Q4tp, *rows, cols / 32);
            enc.set_compute_pipeline_state(&c.q4tp_m);
            fbuf.bind(enc, 0, *abs);
            enc.set_buffer(1, Some(xs), 0);
            enc.set_buffer(2, Some(y), 0);
            let (gpr, r, addy) = ((cols / 32) as u32, *rows as u32, 0u32);
            set_bytes_t(enc, 3, &gpr);
            set_bytes_t(enc, 4, &r);
            set_bytes_t(enc, 5, &addy);
            enc.dispatch_thread_groups(MTLSize::new((*rows as u64).div_ceil(16), 1, 1), MTLSize::new(128, 1, 1));
        }
        Mat::Q8 { abs, rows, cols, rs, col } => {
            enc.set_compute_pipeline_state(&c.q8f_r4);
            fbuf.bind(enc, 0, *abs);
            enc.set_buffer(1, Some(xs), 0);
            enc.set_buffer(2, Some(rs), 0);
            enc.set_buffer(3, Some(y), 0);
            enc.set_buffer(4, Some(col), 0);
            let (c4, r) = ((cols / 4) as u32, *rows as u32);
            set_bytes_t(enc, 5, &c4);
            set_bytes_t(enc, 6, &r);
            enc.dispatch_thread_groups(MTLSize::new((*rows as u64).div_ceil(32), 1, 1), MTLSize::new(256, 1, 1));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn enc_mm(c: &Ctx, enc: &metal::ComputeCommandEncoderRef, fbuf: &WeightArena, m: &Mat, xs: &Buffer, y: &Buffer, xsc: &Buffer, k: usize) -> bool {
    match m {
        Mat::Q4 { abs, rows, cols } => {
            super::encode_q4tp_mm_n8(c, enc, fbuf, *abs, xs, y, xsc, *rows, *cols, k);
            true
        }
        Mat::Q8 { abs, rows, cols, .. } => {
            let Some(p) = super::moe_q8::pipes(c) else { return false };
            super::moe_q8::enc_mm_rows(p, enc, fbuf, *abs, xs, 0, y, 0, k, *rows, *cols, true);
            true
        }
    }
}

/// `CMF_DORY_KEEP=qkv,g,prep,score,soft,sum,gate,o,gu,silu,d,slerp,mix`: only these
/// stages run, for a keep-one table (the logits are then garbage, the timing is not).
fn keep(stage: &str) -> bool {
    static K: OnceLock<Option<Vec<String>>> = OnceLock::new();
    K.get_or_init(|| std::env::var("CMF_DORY_KEEP").ok().map(|v| v.split(',').map(str::to_string).collect()))
        .as_ref()
        .is_none_or(|l| l.iter().any(|x| x == stage))
}

/// Pieces of a value sum over `n` keys.
fn pieces(n: usize) -> usize {
    (n / 8).clamp(1, CH)
}

fn ensure_slot(c: &Ctx, d: &mut Device, ord: usize, sl: usize, need: usize, floor: usize, kv_row: usize) {
    let row = &mut d.slots[ord];
    while row.len() <= sl {
        let cap = need.max(floor);
        row.push(Slot { k: shared(c, cap * kv_row * 4), v: shared(c, cap * kv_row * 4), cap });
    }
    if row[sl].cap >= need {
        return;
    }
    // Double a full-attention slot: shared memory, a plain copy.
    let cap = need.max(row[sl].cap * 2);
    let (k, v) = (shared(c, cap * kv_row * 4), shared(c, cap * kv_row * 4));
    let old = row[sl].cap * kv_row * 4;
    unsafe {
        std::ptr::copy_nonoverlapping(row[sl].k.contents() as *const u8, k.contents() as *mut u8, old);
        std::ptr::copy_nonoverlapping(row[sl].v.contents() as *const u8, v.contents() as *mut u8, old);
    }
    row[sl] = Slot { k, v, cap };
}

/// `true` when this forward ran on the device.
pub(crate) fn forward(
    id: u64,
    model: &Arc<cortiq_core::CmfModel>,
    m: &Dory,
    ids: &[u32],
    pos0: usize,
    loops: usize,
    want: Want,
    logits: &mut Vec<f32>,
) -> bool {
    let Some(c) = ctx() else { return false };
    if !admits(m) {
        return false;
    }
    let mut guard = DEVICES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if !map.contains_key(&id) {
        match build(c, model, m) {
            Some(mut d) => {
                tracing::info!("dory graph on Metal");
                d.id = id;
                map.insert(id, d);
            }
            None => return false,
        }
    }
    let d = map.get_mut(&id).expect("just inserted");
    let mut at = 0usize;
    while at < ids.len() {
        let top = if matches!(d.head, Mat::Q4 { .. }) { CHUNK_Q4 } else { CHUNK };
        let k = (ids.len() - at).min(top);
        let k = if k < 2 || d.sealed { 1 } else { k };
        let last = at + k == ids.len();
        let w = if last || want == Want::All { want } else { Want::None };
        if !run(c, d, m, &ids[at..at + k], pos0 + at, m.chunk_loops(loops, w), w, logits) {
            return false;
        }
        at += k;
        if let (Some(o1), false, false) = (m.o1, d.sealed, d.o1_refused) {
            let rows = pos0 + at;
            // At a prompt's end: its rows go fastest exact, and a long one seals
            // anyway before they outgrow the card. A refused state only costs the
            // layer its far field: it stays exact.
            let due = at == ids.len() || rows >= SEAL_MAX;
            if due && rows >= o1.seal && !seal(c, d, m, o1, rows) {
                d.o1_refused = true;
                d.qtrace.clear();
            }
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn run(c: &Ctx, d: &mut Device, m: &Dory, ids: &[u32], pos: usize, loops: usize, want: Want, logits: &mut Vec<f32>) -> bool {
    let Some(p) = pipes(c) else { return false };
    let cf = &m.cfg;
    let k = ids.len();
    let kv_row = cf.nk * cf.hd;
    let sl_cap = cf.window + 64 + crate::dory::REWIND;
    let need_full = pos + k;
    let per_loop = cf.kv == KvMode::PerLoop;
    let attn: Vec<(usize, usize, bool)> = d
        .layers
        .iter()
        .enumerate()
        .filter_map(|(li, l)| match l {
            LayerDev::Attn(a) => Some((li, a.ordinal, a.sliding)),
            _ => None,
        })
        .collect();
    for (li, ord, sliding) in attn {
        let rec = li >= cf.n_in && li < cf.n_in + cf.n_rec;
        let n = if rec && per_loop { loops } else { 1 };
        for sl in 0..n {
            if !sliding && d.sealed {
                continue;
            }
            if let (false, Some(o1), false, false) = (sliding, m.o1, d.o1_refused, d.qtrace.contains_key(&(ord, sl))) {
                d.qtrace.insert((ord, sl), shared(c, (SEAL_MAX + CHUNK) / QT_STRIDE * cf.nh * cf.hd * 4));
            }
            // A sliding ring has its size; a full slot grows from 4096 rows, or is a
            // ring of its own when the full layers are cut to a window.
            if sliding {
                ensure_slot(c, d, ord, sl, sl_cap, sl_cap, kv_row);
            } else if let Some(w) = cf.full_window {
                let cap = w + 64 + crate::dory::REWIND;
                ensure_slot(c, d, ord, sl, cap, cap, kv_row);
            } else {
                ensure_slot(c, d, ord, sl, need_full, 4096, kv_row);
            }
        }
    }
    if pos + k > d.scores_cap && !d.sealed {
        let cap = (pos + k).max(d.scores_cap * 2);
        d.scores = shared(c, cap * if matches!(d.head, Mat::Q4 { .. }) { CHUNK_Q4 } else { CHUNK } * cf.nk * 16);
        d.scores_cap = cap;
    }
    let full_cap = d
        .layers
        .iter()
        .find_map(|l| match l {
            LayerDev::Attn(a) if !a.sliding => Some(d.slots[a.ordinal][0].cap),
            _ => None,
        })
        .unwrap_or(need_full);
    let make = |cap: usize, window: u32, n: usize| P {
        t: k as u32,
        pos0: pos as u32,
        cap: cap as u32,
        window,
        nh: cf.nh as u32,
        nk: cf.nk as u32,
        hd: cf.hd as u32,
        hidden: cf.hidden as u32,
        ffn: cf.ffn as u32,
        qsize: (cf.nh * cf.hd) as u32,
        ncap: d.scores_cap as u32,
        ch: pieces(n) as u32,
        scale: cf.attn_scale,
        eps: crate::dory::EPS,
        pad0: 0,
        pad1: 0,
    };
    let n_sl = (pos + k).min(cf.window);
    let n_fu = (pos + k).min(cf.full_window.unwrap_or(usize::MAX));
    let p_sl = make(sl_cap, cf.window as u32, n_sl);
    let p_fu = make(full_cap, cf.full_window.map_or(u32::MAX, |w| w as u32), n_fu);
    let p_o1 = make(1, u32::MAX, 1);
    // The rotary tables and the embedded rows, straight into shared memory.
    let half = cf.hd / 2;
    for (ti, inv) in [&m.inv_sliding, &m.inv_full].iter().enumerate() {
        let tab = unsafe { std::slice::from_raw_parts_mut(d.rope[ti].contents() as *mut f32, k * cf.hd) };
        for r in 0..k {
            let at = (pos + r) as f64;
            let row = &mut tab[r * cf.hd..(r + 1) * cf.hd];
            for j in 0..half {
                let (sn, cs) = (at * inv[j] as f64).sin_cos();
                row[j] = cs as f32;
                row[half + j] = sn as f32;
            }
        }
    }
    let rows = unsafe { std::slice::from_raw_parts_mut(d.h.contents() as *mut f32, k * cf.hidden) };
    for (i, row) in rows.chunks_exact_mut(cf.hidden).enumerate() {
        if (ids[i] as usize) < m.embed.rows() {
            m.embed.row_f32(ids[i] as usize, row);
        }
    }
    let g_mode = if k == 1 && want != Want::None { m.gate.clone() } else { None };
    // Which stages one command buffer runs: all of a forward, or a gate's part.
    #[derive(Clone, Copy, PartialEq)]
    enum Stage {
        All,
        First,
        FirstRead,
        Read,
        SecondRead,
        /// The second pass of a token that had no read-out yet.
        SecondFresh,
    }
    let run_cmd = |d: &Device, stage: Stage| -> bool {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let fb: &WeightArena = &d.fbuf;
        let (kx, tg32) = (k as u64, MTLSize::new(32, 1, 1));
        let layer = |li: usize, lp: usize| -> bool {
            match &d.layers[li] {
                LayerDev::Attn(a) => {
                    let rec = li >= cf.n_in && li < cf.n_in + cf.n_rec;
                    let sl = if rec && per_loop { lp } else { 0 };
                    if d.sealed && !a.sliding {
                        let Some(od) = d.far.get(&(a.ordinal, sl)) else { return false };
                        if keep("qkv") {
                            enc_mv(c, enc, fb, &a.q, &d.h, &d.yq);
                            enc_mv(c, enc, fb, &a.k, &d.h, &d.yk);
                            enc_mv(c, enc, fb, &a.v, &d.h, &d.yv);
                        }
                        if keep("g") {
                            enc_mv(c, enc, fb, &a.g, &d.h, &d.yg);
                        }
                        enc.set_compute_pipeline_state(&p.prep);
                        for (i, b) in [&d.yq, &d.yk, &d.yv, &d.rope[1], &a.sq, &d.qo, &d.kt, &d.vt].iter().enumerate() {
                            enc.set_buffer(i as u64, Some(b), 0);
                        }
                        set_bytes_t(enc, 8, &p_o1);
                        enc.dispatch_thread_groups(MTLSize::new((cf.nh + cf.nk) as u64, 1, 1), tg32);
                        let (gg, hh, mm) = (od.g as u64, od.h as u64, od.m as u64);
                        if lp > 0 && !per_loop {
                            enc.set_compute_pipeline_state(&p.o1_back);
                            enc.set_buffer(0, Some(&od.meta), 0);
                            let w = od.w as u32;
                            set_bytes_t(enc, 1, &w);
                            enc.dispatch_threads(MTLSize::new(gg, 1, 1), MTLSize::new(1, 1, 1));
                        } else if keep("score") {
                            super::disp(
                                enc,
                                &c.o1far,
                                &[(&od.meta, 0), (&od.ring_k, 0), (&od.ring_v, 0), (&od.qt, 0), (&od.mz, 0), (&od.that, 0)],
                                &[od.h as u32, od.m as u32, od.w as u32, od.d as u32, od.dv as u32],
                                &[od.scale],
                                (gg * hh * mm * 64, 64),
                            );
                        }
                        super::disp(
                            enc,
                            &c.o1push,
                            &[(&od.meta, 0), (&d.kt, 0), (&d.vt, 0), (&od.ring_k, 0), (&od.ring_v, 0)],
                            &[od.w as u32, od.d as u32, od.dv as u32],
                            &[],
                            (gg * 256, 256),
                        );
                        // The far field's attention in short threads, as in the exact path.
                        let ns = d.far_ns as u64;
                        let ou: [u32; 8] = [od.h as u32, od.m as u32, od.w as u32, ns as u32, od.d as u32, od.dv as u32, od.scale.to_bits(), d.far_rect];
                        let stage = |pso: &ComputePipelineState, bufs: &[&Buffer]| {
                            enc.set_compute_pipeline_state(pso);
                            for (i, b) in bufs.iter().enumerate() {
                                enc.set_buffer(i as u64, Some(b), 0);
                            }
                            let at = bufs.len() as u64;
                            set_bytes_t(enc, at, &p_fu);
                            set_bytes_t(enc, at + 1, &ou);
                        };
                        let e4 = (cf.hd / 4) as u64;
                        let nk = cf.nk as u64;
                        stage(&p.os, &[&od.meta, &d.qo, &od.ring_k, &od.sink_k, &d.scores]);
                        enc.dispatch_thread_groups(MTLSize::new((od.w as u64 + ns).div_ceil(32), nk, 1), tg32);
                        stage(&p.ol, &[&d.qo, &od.k_tilde, &d.ofs]);
                        enc.dispatch_thread_groups(MTLSize::new(mm.div_ceil(32), nk, 1), tg32);
                        stage(&p.omax, &[&od.meta, &d.scores, &d.onm]);
                        enc.dispatch_thread_groups(MTLSize::new(nk, 1, 1), tg32);
                        stage(&p.ofar, &[&od.meta, &d.ofs, &d.onm, &od.mu, &od.mz, &d.ohd, &d.ogn]);
                        enc.dispatch_thread_groups(MTLSize::new((cf.nh as u64).div_ceil(8), 1, 1), MTLSize::new(8, 1, 1));
                        stage(&p.osoft, &[&od.meta, &d.scores, &d.ohd, &d.sinv]);
                        enc.dispatch_thread_groups(MTLSize::new(nk, 1, 1), tg32);
                        stage(&p.osum, &[&od.meta, &d.scores, &od.ring_v, &od.sink_v, &d.pz]);
                        enc.dispatch_thread_groups(MTLSize::new((nk * 8 * e4).div_ceil(32), 1, 1), tg32);
                        stage(&p.ogate, &[&d.pz, &d.ohd, &d.ogn, &od.that, &d.sinv, &d.yg, &a.sg, &d.z]);
                        enc.dispatch_thread_groups(MTLSize::new(((cf.nh as u64) * e4).div_ceil(32), 1, 1), tg32);
                        if keep("o") {
                            enc_mv(c, enc, fb, &a.o, &d.z, &d.o);
                        }
                        slerp(p, enc, d, &a.alpha, &p_sl, kx);
                        return true;
                    }
                    let slot = &d.slots[a.ordinal][sl];
                    let (pp, n) = if a.sliding { (&p_sl, n_sl) } else { (&p_fu, n_fu) };
                    if k == 1 {
                        if keep("qkv") {
                            enc_mv(c, enc, fb, &a.q, &d.h, &d.yq);
                            enc_mv(c, enc, fb, &a.k, &d.h, &d.yk);
                            enc_mv(c, enc, fb, &a.v, &d.h, &d.yv);
                        }
                        if keep("g") {
                            enc_mv(c, enc, fb, &a.g, &d.h, &d.yg);
                        }
                    } else if !(enc_mm(c, enc, fb, &a.q, &d.h, &d.yq, &d.xsc, k)
                        && enc_mm(c, enc, fb, &a.k, &d.h, &d.yk, &d.xsc, k)
                        && enc_mm(c, enc, fb, &a.v, &d.h, &d.yv, &d.xsc, k)
                        && enc_mm(c, enc, fb, &a.g, &d.h, &d.yg, &d.xsc, k))
                    {
                        return false;
                    }
                    enc.set_compute_pipeline_state(&p.prep);
                    let r = if a.sliding { &d.rope[0] } else { &d.rope[1] };
                    for (i, b) in [&d.yq, &d.yk, &d.yv, r, &a.sq, &d.qo, &slot.k, &slot.v].iter().enumerate() {
                        enc.set_buffer(i as u64, Some(b), 0);
                    }
                    set_bytes_t(enc, 8, pp);
                    if keep("prep") {
                        enc.dispatch_thread_groups(MTLSize::new((cf.nh + cf.nk) as u64, kx, 1), tg32);
                    }
                    if let Some(tr) = d.qtrace.get(&(a.ordinal, sl)) {
                        enc.set_compute_pipeline_state(&p.qcopy);
                        enc.set_buffer(0, Some(&d.qo), 0);
                        enc.set_buffer(1, Some(tr), 0);
                        set_bytes_t(enc, 2, &p_fu);
                        enc.dispatch_thread_groups(MTLSize::new(((cf.nh * cf.hd / 4) as u64).div_ceil(32), kx, 1), tg32);
                    }
                    enc.set_compute_pipeline_state(&p.score);
                    enc.set_buffer(0, Some(&d.qo), 0);
                    enc.set_buffer(1, Some(&slot.k), 0);
                    enc.set_buffer(2, Some(&d.scores), 0);
                    set_bytes_t(enc, 3, pp);
                    if keep("score") {
                        enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(32), cf.nk as u64, kx), tg32);
                    }
                    enc.set_compute_pipeline_state(&p.soft);
                    enc.set_buffer(0, Some(&d.scores), 0);
                    enc.set_buffer(1, Some(&d.sinv), 0);
                    set_bytes_t(enc, 2, pp);
                    if keep("soft") {
                        enc.dispatch_thread_groups(MTLSize::new(cf.nk as u64, kx, 1), tg32);
                    }
                    enc.set_compute_pipeline_state(&p.sum);
                    enc.set_buffer(0, Some(&slot.v), 0);
                    enc.set_buffer(1, Some(&d.scores), 0);
                    enc.set_buffer(2, Some(&d.pz), 0);
                    set_bytes_t(enc, 3, pp);
                    let threads = (cf.nk * pieces(n) * cf.hd / 4) as u64;
                    if keep("sum") {
                        enc.dispatch_thread_groups(MTLSize::new(threads.div_ceil(32), kx, 1), tg32);
                    }
                    enc.set_compute_pipeline_state(&p.gate);
                    for (i, b) in [&d.yg, &a.sg, &d.pz, &d.sinv, &d.z].iter().enumerate() {
                        enc.set_buffer(i as u64, Some(b), 0);
                    }
                    set_bytes_t(enc, 5, pp);
                    if keep("gate") {
                        enc.dispatch_thread_groups(MTLSize::new(((cf.nh * cf.hd / 4) as u64).div_ceil(32), kx, 1), tg32);
                    }
                    if k == 1 {
                        if keep("o") {
                            enc_mv(c, enc, fb, &a.o, &d.z, &d.o);
                        }
                    } else if !enc_mm(c, enc, fb, &a.o, &d.z, &d.o, &d.xsc, k) {
                        return false;
                    }
                    slerp(p, enc, d, &a.alpha, &p_sl, kx);
                }
                LayerDev::Mlp(a) => {
                    if k == 1 {
                        if keep("gu") {
                            enc_mv(c, enc, fb, &a.gate, &d.h, &d.fg);
                            enc_mv(c, enc, fb, &a.up, &d.h, &d.fu);
                        }
                    } else if !(enc_mm(c, enc, fb, &a.gate, &d.h, &d.fg, &d.xsc, k) && enc_mm(c, enc, fb, &a.up, &d.h, &d.fu, &d.xsc, k)) {
                        return false;
                    }
                    enc.set_compute_pipeline_state(&p.silu);
                    for (i, b) in [&d.fg, &d.fu, &a.sg, &a.su].iter().enumerate() {
                        enc.set_buffer(i as u64, Some(b), 0);
                    }
                    set_bytes_t(enc, 4, &p_sl);
                    if keep("silu") {
                        enc.dispatch_thread_groups(MTLSize::new(((cf.ffn / 4) as u64).div_ceil(32), kx, 1), tg32);
                    }
                    if k == 1 {
                        if keep("d") {
                            enc_mv(c, enc, fb, &a.down, &d.fg, &d.o);
                        }
                    } else if !enc_mm(c, enc, fb, &a.down, &d.fg, &d.o, &d.xsc, k) {
                        return false;
                    }
                    slerp(p, enc, d, &a.alpha, &p_sl, kx);
                }
            }
            true
        };
        let mix = |mode: u32| {
            enc.set_compute_pipeline_state(&p.mix);
            enc.set_buffer(0, Some(&d.h), 0);
            enc.set_buffer(1, Some(&d.input), 0);
            enc.set_buffer(2, Some(&d.state), 0);
            set_bytes_t(enc, 3, &mode);
            set_bytes_t(enc, 4, &p_sl);
            if keep("mix") {
                enc.dispatch_thread_groups(MTLSize::new(kx, 1, 1), tg32);
            }
        };
        let mut ok = true;
        if matches!(stage, Stage::All | Stage::First | Stage::FirstRead) {
            for li in 0..cf.n_in {
                ok &= layer(li, 0);
            }
            mix(0);
        }
        if stage == Stage::All {
            for i in 0..loops {
                for li in cf.n_in..cf.n_in + cf.n_rec {
                    ok &= layer(li, i);
                }
                mix(if i + 1 == loops { 2 } else { 1 });
            }
        } else if matches!(stage, Stage::First | Stage::FirstRead) {
            for li in cf.n_in..cf.n_in + cf.n_rec {
                ok &= layer(li, 0);
            }
            mix(3);
        } else if matches!(stage, Stage::SecondRead | Stage::SecondFresh) {
            mix(4);
            for li in cf.n_in..cf.n_in + cf.n_rec {
                ok &= layer(li, 1);
            }
            mix(2);
        }
        if stage != Stage::First {
            // A second read-out rewrites this position in the output block.
            let again = stage == Stage::SecondRead;
            for li in cf.n_in + cf.n_rec..d.layers.len() {
                ok &= layer(li, usize::from(again));
            }
        }
        if stage == Stage::First {
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            return ok;
        }
        match want {
            Want::None => {}
            Want::Last => {
                let src = d.h.contents() as *const u8;
                let _ = src;
                if k == 1 {
                    enc_mv(c, enc, fb, &d.head, &d.h, &d.logits);
                } else {
                    // The head of the chunk's last row alone.
                    enc.end_encoding();
                    let blit = cmd.new_blit_command_encoder();
                    blit.copy_from_buffer(&d.h, ((k - 1) * cf.hidden * 4) as u64, &d.hlast, 0, (cf.hidden * 4) as u64);
                    blit.end_encoding();
                    let enc2 = cmd.new_compute_command_encoder();
                    enc_mv(c, enc2, fb, &d.head, &d.hlast, &d.logits);
                    enc2.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    return ok;
                }
            }
            Want::All => {
                if k == 1 {
                    enc_mv(c, enc, fb, &d.head, &d.h, &d.logits);
                } else {
                    ok &= enc_mm(c, enc, fb, &d.head, &d.h, &d.logits, &d.xsc, k);
                }
            }
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        ok
    };
    let hb = cf.hidden;
    let top_read = |d: &Device| -> Vec<f32> {
        let y = unsafe { std::slice::from_raw_parts(d.logits.contents() as *const f32, cf.vocab) };
        y.iter().zip(&m.logit_scale).map(|(x, s)| *x * *s).collect()
    };
    let ok = metal::objc::rc::autoreleasepool(|| match &g_mode {
        None => run_cmd(d, Stage::All),
        Some(Gate::Conf(th)) => {
            d.gate_tokens += 1;
            if !d.exits.probe() {
                return run_cmd(d, Stage::First) && run_cmd(d, Stage::SecondFresh);
            }
            if !run_cmd(d, Stage::FirstRead) {
                return false;
            }
            let one = top_prob(&top_read(d)) >= *th;
            d.exits.note(one);
            d.gate_one += u64::from(one);
            one || run_cmd(d, Stage::SecondRead)
        }
        Some(Gate::Recon(t, bias)) => {
            if !run_cmd(d, Stage::First) {
                return false;
            }
            let phi = unsafe { std::slice::from_raw_parts(d.state.contents() as *const f32, hb) };
            let one = t.score(phi) >= *bias;
            d.gate_tokens += 1;
            d.gate_one += u64::from(one);
            run_cmd(d, if one { Stage::Read } else { Stage::SecondFresh })
        }
    });
    if !ok {
        return false;
    }
    if g_mode.is_some() && d.gate_tokens % 64 == 0 && std::env::var_os("CMF_DORY_GATE_LOG").is_some() {
        eprintln!("dory gate: {} tokens, {:.1} % on one pass", d.gate_tokens, 100.0 * d.gate_one as f64 / d.gate_tokens as f64);
    }
    match want {
        Want::None => {}
        Want::Last => {
            logits.clear();
            let y = unsafe { std::slice::from_raw_parts(d.logits.contents() as *const f32, cf.vocab) };
            logits.extend(y.iter().zip(&m.logit_scale).map(|(x, s)| *x * *s));
        }
        Want::All => {
            let from = logits.len();
            let y = unsafe { std::slice::from_raw_parts(d.logits.contents() as *const f32, k * cf.vocab) };
            logits.extend_from_slice(y);
            for row in logits[from..].chunks_exact_mut(cf.vocab) {
                for (x, s) in row.iter_mut().zip(&m.logit_scale) {
                    *x *= *s;
                }
            }
        }
    }
    true
}

/// O(1) on the full layers from here on: freeze landmarks over the slots' rows
/// (`nystrom`), hand over the states, free the exact rows.
fn seal(c: &Ctx, d: &mut Device, m: &Dory, o1: crate::dory::DoryO1, rows: usize) -> bool {
    let cf = &m.cfg;
    let (nh, nk, hd) = (cf.nh, cf.nk, cf.hd);
    let rep = nh / nk;
    let t0 = std::time::Instant::now();
    let full: Vec<usize> = d
        .layers
        .iter()
        .filter_map(|l| match l {
            LayerDev::Attn(a) if !a.sliding => Some(a.ordinal),
            _ => None,
        })
        .collect();
    let keep: Vec<usize> = match &m.exact {
        Some(x) => {
            let view = |b: &Buffer, n: usize| unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) };
            let layers: Vec<(&[f32], &[f32])> = full.iter().map(|&o| (view(&d.slots[o][0].k, rows * nk * hd), view(&d.slots[o][0].v, rows * nk * hd))).collect();
            crate::dory::pick_exact_rows(x, &layers, rows, nk, hd, o1.m, o1.sink, rows.saturating_sub(o1.w))
        }
        None => Vec::new(),
    };
    let order = crate::dory::seal_order(rows, o1.sink, &keep);
    let (keep, order) = (&keep, &order);
    let (mut far_ns, mut far_rect) = (0usize, 0u32);
    let mut far = HashMap::new();
    for ord in full {
        for sl in 0..d.slots[ord].len() {
            let slot = &d.slots[ord][sl];
            let Some(tr) = d.qtrace.get(&(ord, sl)) else { return false };
            let view = |b: &Buffer, n: usize| unsafe { std::slice::from_raw_parts(b.contents() as *const f32, n) };
            let (ks, vs, qs) = (view(&slot.k, rows * nk * hd), view(&slot.v, rows * nk * hd), view(tr, rows.div_ceil(QT_STRIDE) * nh * hd));
            let mut states = Vec::with_capacity(nk);
            std::thread::scope(|sc| {
                let mut hs = Vec::with_capacity(nk);
                for kh in 0..nk {
                    hs.push(sc.spawn(move || {
                        let mut kg = vec![0f32; rows * hd];
                        let mut vg = vec![0f32; rows * hd];
                        let mut qg: Vec<Vec<f32>> = (0..rep).map(|_| vec![0f32; rows * hd]).collect();
                        for (r, &from) in order.iter().enumerate() {
                            let from = from as usize;
                            let at = (from * nk + kh) * hd;
                            kg[r * hd..(r + 1) * hd].copy_from_slice(&ks[at..at + hd]);
                            vg[r * hd..(r + 1) * hd].copy_from_slice(&vs[at..at + hd]);
                            for (h, q) in qg.iter_mut().enumerate() {
                                let at = (from / QT_STRIDE * nh + kh * rep + h) * hd;
                                q[r * hd..(r + 1) * hd].copy_from_slice(&qs[at..at + hd]);
                            }
                        }
                        let mut n = crate::nystrom::NystromState::new_group(o1.m, o1.w, o1.sink + keep.len(), rep);
                        let refs: Vec<&[f32]> = qg.iter().map(|q| &q[..]).collect();
                        n.prefill_group(&refs, &kg, &vg, rows, hd, hd);
                        n
                    }));
                }
                for h in hs {
                    states.push(h.join().expect("seal worker"));
                }
            });
            let views: Vec<_> = states.iter().map(|n| n.device_view()).collect();
            far_ns = views[0].sink_len;
            far_rect = u32::from(views[0].heads[0].rect_fm);
            let Some(dev) = o1_ensure_metal(c, d.id, ord * 16 + sl, &views, d.seal_epoch + 1, crate::dory::MAX_NEAR) else {
                tracing::warn!("dory: the far state does not fit the kernels, the layer stays exact");
                return false;
            };
            far.insert((ord, sl), dev);
        }
    }
    let keys: Vec<(usize, usize)> = far.keys().copied().collect();
    d.far = far;
    for (o, sl) in keys {
        d.slots[o][sl].k = shared(c, 16);
        d.slots[o][sl].v = shared(c, 16);
        d.slots[o][sl].cap = 1;
    }
    d.qtrace.clear();
    d.far_ns = far_ns;
    d.far_rect = far_rect;
    d.sealed = true;
    d.seal_epoch += 1;
    tracing::info!("dory: O(1) sealed {rows} rows in {:.2} s", t0.elapsed().as_secs_f64());
    true
}

/// A new sequence: O(1) states go, and exact slots come back on demand.
pub(crate) fn reset(id: u64) {
    if let Some(map) = DEVICES.lock().unwrap().as_mut() {
        if let Some(d) = map.get_mut(&id) {
            if d.sealed {
                d.sealed = false;
                d.far.clear();
                d.slots.iter_mut().for_each(|r| r.clear());
            }
            d.o1_refused = false;
        }
    }
}

fn slerp(p: &Pipes, enc: &metal::ComputeCommandEncoderRef, d: &Device, alpha: &Buffer, u: &P, kx: u64) {
    enc.set_compute_pipeline_state(&p.slerp);
    enc.set_buffer(0, Some(&d.h), 0);
    enc.set_buffer(1, Some(&d.o), 0);
    enc.set_buffer(2, Some(alpha), 0);
    set_bytes_t(enc, 3, u);
    if keep("slerp") {
        enc.dispatch_thread_groups(MTLSize::new(kx, 1, 1), MTLSize::new(32, 1, 1));
    }
}

/// The KV bytes of a sequence on the device.
/// Has this sequence gone over to the far field?
pub(crate) fn sealed(id: u64) -> bool {
    DEVICES.lock().unwrap().as_ref().and_then(|g| g.get(&id)).is_some_and(|d| d.sealed)
}

pub(crate) fn kv_bytes(id: u64) -> Option<u64> {
    let g = DEVICES.lock().unwrap();
    let d = g.as_ref()?.get(&id)?;
    let exact: u64 = d.slots.iter().flatten().map(|s| s.k.length() + s.v.length()).sum();
    let far: u64 = d
        .far
        .values()
        .map(|o| {
            [&o.meta, &o.ring_k, &o.ring_v, &o.sink_k, &o.sink_v, &o.k_tilde, &o.qt, &o.mu, &o.mz, &o.that]
                .iter()
                .map(|b| b.length())
                .sum::<u64>()
        })
        .sum();
    Some(exact + far)
}
