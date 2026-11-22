//! Inkling-2-Mini on Metal, the twin of the CUDA arm (`gpu_cuda/hyb/inkling.rs`;
//! `inkling` has the semantics). Weights stay in the file mapping (the
//! backend's arena): the dense matrices through the backend's q4tp / q8_2f
//! matvecs, the routed experts through this arm's kernels, which pick them on
//! the GPU (router, top-k, every pick's gate/up with Inkling's softcapped
//! SwiGLU, the weighted downs) — nothing returns to the host inside a token.
//! Every one of the 48 layer applications keeps its own keys and values (f16;
//! a local layer's in a ring of its window, a full layer's growing) and its
//! four convolution rings. A call (a token, or a prompt chunk of up to
//! `CMF_INKLING_ROWS`, 256, rows) is one command buffer. A prompt chunk's
//! dense matrices go through the backend's simdgroup GEMM (`q4tp_mul_mm`), its
//! routed experts grouped by expert, so a chunk reads each expert once.

use super::{Ctx, WeightArena, ctx, file_buffer, set_bytes_t};
use crate::inkling::{Cfg, Globals, Layer};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MSL: &str = include_str!("inkling_msl.metal");
/// Rows of a call (`CMF_INKLING_ROWS`, 256: M4, a 512-token prompt 117 tok/s
/// at 128, 140 at 256, 145 at 512).
fn rows_cap() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("CMF_INKLING_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).map_or(256, |n| n.clamp(1, 1024)))
}
/// Block states a row holds.
const MAXB: usize = 4;
/// Rows of a draft verify (the drafter's block: the anchor and 7 drafts).
const VROWS: usize = 8;
/// Taps of the short convolutions (their rings keep `KC - 1` rows).
const KC: usize = 4;

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
    /// A draft verify: the convolutions write every row's input (`SlotM::hist`).
    hist: u32,
    pad2: f32,
    pad3: f32,
}

struct Pipes {
    mix: ComputePipelineState,
    rms: ComputePipelineState,
    gate_down: ComputePipelineState,
    gate_out: ComputePipelineState,
    gate_norm: ComputePipelineState,
    gate_mix: ComputePipelineState,
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
    moe_gu8: ComputePipelineState,
    moe_dn8: ComputePipelineState,
    group: ComputePipelineState,
    moe_gu_g: ComputePipelineState,
    moe_down_g: ComputePipelineState,
    moe_mix: ComputePipelineState,
    moe_mm: ComputePipelineState,
    moe_act: ComputePipelineState,
    moe_mixq: ComputePipelineState,
    qrkv: ComputePipelineState,
    fmv3: ComputePipelineState,
    mvq: ComputePipelineState,
    gate_down3: ComputePipelineState,
    gate_out3: ComputePipelineState,
    attn_fl: ComputePipelineState,
    xcol: ComputePipelineState,
    q8p: ComputePipelineState,
    ds_mix: ComputePipelineState,
    ds_ctx: ComputePipelineState,
    ds_qk: ComputePipelineState,
    ds_attn: ComputePipelineState,
    ds_act: ComputePipelineState,
    ds_fmv: ComputePipelineState,
    ds_markov: ComputePipelineState,
    ds_pick: ComputePipelineState,
    argmax: ComputePipelineState,
    q8mm8: ComputePipelineState,
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
            gate_norm: pso("ink_gate_norm")?,
            gate_mix: pso("ink_gate_mix")?,
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
            moe_gu8: pso("ink_moe_gu8")?,
            moe_dn8: pso("ink_moe_dn8")?,
            group: pso("ink_group")?,
            moe_gu_g: pso("ink_moe_gu_g")?,
            moe_down_g: pso("ink_moe_down_g")?,
            moe_mix: pso("ink_moe_mix")?,
            moe_mm: pso("ink_moe_mm")?,
            moe_act: pso("ink_moe_act")?,
            moe_mixq: pso("ink_moe_mixq")?,
            qrkv: pso("ink_qrkv")?,
            fmv3: pso("ink_fmv3")?,
            mvq: pso("ink_mvq")?,
            gate_down3: pso("ink_gate_down3")?,
            gate_out3: pso("ink_gate_out3")?,
            attn_fl: pso("ink_attn_fl")?,
            xcol: pso("ink_xcol")?,
            q8p: pso("ink_q8p")?,
            ds_mix: pso("ink_ds_mix")?,
            ds_ctx: pso("ink_ds_ctx")?,
            ds_qk: pso("ink_ds_qk")?,
            ds_attn: pso("ink_ds_attn")?,
            ds_act: pso("ink_ds_act")?,
            ds_fmv: pso("ink_ds_fmv")?,
            ds_markov: pso("ink_ds_markov")?,
            ds_pick: pso("ink_ds_pick")?,
            argmax: pso("ink_argmax")?,
            q8mm8: pso("ink_q8mm8")?,
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
    /// q8_2f rows gathered into a buffer of their own (the drafter's table).
    Q8B { buf: Buffer, rows: usize, cols: usize, rs: Buffer, col: Buffer },
}

impl Mat {
    fn rows(&self) -> usize {
        match self {
            Mat::Q4 { rows, .. } | Mat::Q8 { rows, .. } | Mat::Q8B { rows, .. } => *rows,
        }
    }
}

/// `DsP` of the shaders (the drafter's kernels).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DsP {
    eps: f32,
    theta: f32,
    cap: u32,
    ring: u32,
    pos0: u32,
    b: u32,
    win: u32,
    r: u32,
}

/// Positions whose Markov context the ring keeps.
const FR: usize = 16;

/// One layer of the DSpark drafter on the GPU (q8_2f matrices in the mapping).
struct DLayerM {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    kc: Mat,
    vc: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
    in_norm: Buffer,
    post_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    kc_norm: Buffer,
    /// Its keys and values (f16 `[cell][nkv·hd]`): the context's from every
    /// target call, the block's at a round's positions; a sliding layer's a
    /// ring of its window and more.
    ck: Buffer,
    cv: Buffer,
    cap: usize,
    ring: bool,
}

/// The DSpark drafter (`g.draft`) on the GPU: `inkling.cu`'s drafter on Metal.
struct DraftM {
    layers: Vec<DLayerM>,
    /// Applications whose input rows are taps (the final normed state the last).
    taps: Vec<usize>,
    /// The call's taps `[tap][rows][H]`, the mixes `[layer][rows][H]`, the
    /// Markov context ring `[FR][H]`.
    tap: Buffer,
    mix: Buffer,
    fr: Buffer,
    alpha: Buffer,
    norms: Buffer,
    fnorm: Buffer,
    masks: Vec<f32>,
    x: Buffer,
    xn: Buffer,
    dq: Buffer,
    dk: Buffer,
    dv: Buffer,
    att: Buffer,
    gb: Buffer,
    ub: Buffer,
    ab: Buffer,
    hf: Buffer,
    /// x·col of a q8_2f product (`[8][widest]`), a commit's context keys and values.
    xc: Buffer,
    kcb: Buffer,
    vcb: Buffer,
    dlog: Buffer,
    rnn_w: Buffer,
    rnn_b: Buffer,
    ch: Buffer,
    cf: Buffer,
    state: Buffer,
    prev: Buffer,
    ids: Buffer,
    /// Each draft's probability under the drafter (`ink_ds_pick`).
    probs: Buffer,
    w1: Buffer,
    fold: Mat,
    fold_b: Buffer,
    /// `CMF_INKLING_DRAFT_TABLE=N`: drafts scored on N tokens only (the head's
    /// and the fold's rows of the N largest fold_b, the Markov head's unigram
    /// prior), `map` their ids; the verify keeps the whole head.
    table: Option<(Mat, Mat, Buffer, Buffer, usize)>,
    lat: Buffer,
    block: usize,
    rank: usize,
    window: usize,
    theta: f32,
    eps: f32,
    inter: usize,
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

/// One matrix of `ink_mvq`'s decode-row launch (the shader's `MvJob`).
#[repr(C)]
#[derive(Clone, Copy)]
struct MvJob {
    abs: u64,
    rows: u32,
    xs: u32,
    ys: u32,
    tg0: u32,
}

struct LayerM {
    /// q, gate, rel, k, v as one decode-row launch (`ink_mvq`): its jobs, their
    /// count and threadgroups — [0] all from input 0, [1] k from 1 and v from 2
    mvj: Option<[(Buffer, u32, u64); 2]>,
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
    /// A draft verify's input rows of the four rings (`[VROWS][width]` each,
    /// at [`hist_off`]): the rollback rebuilds the rings from them.
    hist: Buffer,
}

/// Byte offsets of ring `j`'s rows in `SlotM::hist` (k and v `kvs` wide, the
/// two output convolutions `hidden`).
fn hist_off(j: usize, kvs: usize, hidden: usize) -> u64 {
    (match j {
        0 => 0,
        1 => VROWS * kvs,
        2 => 2 * VROWS * kvs,
        _ => 2 * VROWS * kvs + VROWS * hidden,
    } * 4) as u64
}

struct Dev {
    /// Rows a call holds.
    rows: usize,
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
    /// A prompt call's gate as GEMMs: the normed rows side by side, F·d, G·d.
    gxn: Buffer,
    gtf: Buffer,
    gtg: Buffer,
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
    /// A multi-row call's (row, pick) entries grouped by expert, and the
    /// picks' down outputs `[row·topk + k][H]`.
    grp: Buffer,
    yp: Buffer,
    /// The places' gate and up rows of the grouped GEMMs `[place][inter]`.
    gq: Buffer,
    uq: Buffer,
    y: Buffer,
    fin: Buffer,
    xf: Buffer,
    logits: Buffer,
    dummy: Buffer,
    /// The rings before a draft verify (`ring_snap`), every slot's four in turn.
    snap: Vec<f32>,
    /// A verify's head input x·col (`ink_xcol`), and the narrow GEMM's
    /// activation pre-scales (ones: the verify's rows are normed).
    pxc: Buffer,
    ones8: Buffer,
    /// A verify's greedy picks (`ink_argmax`).
    vpick: Buffer,
    /// The rings' widths: k and v `nkv·hd`, the two output convolutions `hidden`.
    rw: [usize; 4],
    /// The DSpark drafter, when the file has one and its matrices are q8_2f.
    draft: Option<DraftM>,
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
        let (q, k, v, gt, rl) = (mm(&a.q)?, mm(&a.k)?, mm(&a.v)?, mm(&a.gate)?, mm(&a.rel)?);
        let mvj = {
            let all = [(&q, 0u32), (&gt, 1), (&rl, 2), (&k, 3), (&v, 4)];
            if all.iter().all(|(m, _)| matches!(m, Mat::Q4 { cols, .. } if *cols == hidden)) {
                let mk = |two: bool| {
                    let mut jobs = Vec::with_capacity(5);
                    let mut tg0 = 0u32;
                    for (m, ys) in all {
                        let Mat::Q4 { abs, rows, .. } = m else { unreachable!() };
                        let xs = if two && ys >= 3 { ys - 2 } else { 0 };
                        jobs.push(MvJob { abs: *abs as u64, rows: *rows as u32, xs, ys, tg0 });
                        tg0 += rows.div_ceil(16) as u32;
                    }
                    (up(c, &jobs), jobs.len() as u32, tg0 as u64)
                };
                Some([mk(false), mk(true)])
            } else {
                None
            }
        };
        let o = mm(&a.o)?;
        dl.push(LayerM {
            mvj,
            sliding: l.sliding,
            boundary: li % cfg.block == 0,
            aq: l.attn_query.as_ref().map(|q| (up(c, q), q.len() / hidden)),
            mq: up(c, &l.mlp_query),
            norms: [up(c, &l.in_norm[0]), up(c, &l.in_norm[1]), up(c, &l.in_norm[2])],
            post_norm: up(c, &l.post_norm),
            q,
            k,
            v,
            o,
            gate: gt,
            rel: rl,
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
        let ring = if layers[li].sliding { cfg.window + rows_cap() } else { 0 };
        let cap = if ring > 0 { ring } else { 4096 };
        slots[cfg.slot(li, pass)] = Some(SlotM {
            k: shared(c, cap * kvs * 2),
            v: shared(c, cap * kvs * 2),
            cap,
            ring,
            rings: [shared(c, 8 * kvs * 4), shared(c, 8 * kvs * 4), shared(c, 8 * hidden * 4), shared(c, 8 * hidden * 4)],
            hist: shared(c, 2 * VROWS * (kvs + hidden) * 4),
        });
    }
    let slots: Vec<SlotM> = slots.into_iter().map(|s| s.ok_or("a slot without its application")).collect::<Result<_, _>>()?;
    let f4 = |n: usize| shared(c, n * 4);
    let rows = rows_cap();
    let (qs, rs) = (nh * hd, nh * dr);
    let e = dl.first().map_or(0, |l| l.e);
    let inter = dl.first().map_or(0, |l| l.inter);
    let vocab = head.rows();
    Ok(Dev {
        rows,
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
        gxn: f4(rows * 2 * hidden),
        gtf: f4(rows * hidden),
        gtg: f4(rows * hidden),
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
        grp: shared(c, (e + 1 + 2 * rows * cfg.top_k) * 4),
        yp: f4(rows * cfg.top_k * hidden),
        gq: f4(rows * cfg.top_k * inter),
        uq: f4(rows * cfg.top_k * inter),
        y: f4(rows * hidden),
        fin: f4(rows * hidden),
        xf: f4(rows * hidden),
        logits: f4(rows.min(16) * vocab),
        dummy: shared(c, 64),
        snap: Vec::new(),
        pxc: f4(VROWS * hidden),
        ones8: up(c, &[1.0f32; 8]),
        vpick: shared(c, VROWS * 4),
        rw: [kvs, kvs, hidden, hidden],
        draft: match g.draft.as_ref() {
            Some(dr) if spec_on() => build_draft(c, g, dr, cfg, rows),
            _ => None,
        },
    })
}

/// `CMF_INKLING_DRAFT_TABLE=[RANKING:]N`: the drafter's vocabulary, N tokens:
/// the first of RANKING (token ids most frequent first, one a line), then by
/// fold_b (the Markov head's unigram prior). Unset: the whole head.
fn draft_table() -> Option<(Option<String>, usize)> {
    static V: OnceLock<Option<(Option<String>, usize)>> = OnceLock::new();
    V.get_or_init(|| {
        let v = std::env::var("CMF_INKLING_DRAFT_TABLE").ok()?;
        let (path, n) = match v.rsplit_once(':') {
            Some((p, n)) => (Some(p.to_string()), n),
            None => (None, v.as_str()),
        };
        n.parse::<usize>().ok().filter(|&n| n > 0).map(|n| (path, n))
    })
    .clone()
}

/// The drafter's table of `n` tokens (in id order): the head's and the fold's
/// q8_2f rows, their fold_b, and their ids.
fn gather_table(c: &Ctx, g: &Globals, dr: &crate::inkling::Draft, (path, n): (Option<String>, usize)) -> Option<(Mat, Mat, Buffer, Buffer, usize)> {
    let vocab = g.lm_head.rows().min(dr.fold_b.len());
    if n >= vocab {
        return None;
    }
    let mut keep = vec![false; vocab];
    let mut ids: Vec<u32> = Vec::with_capacity(n);
    if let Some(p) = path {
        let text = std::fs::read_to_string(&p).map_err(|e| tracing::warn!("draft table {p}: {e}")).ok()?;
        for id in text.lines().filter_map(|l| l.trim().parse::<u32>().ok()) {
            if ids.len() < n && (id as usize) < vocab && !keep[id as usize] {
                keep[id as usize] = true;
                ids.push(id);
            }
        }
    }
    let mut prior: Vec<u32> = (0..vocab as u32).filter(|&i| !keep[i as usize]).collect();
    prior.sort_by(|&a, &b| dr.fold_b[b as usize].total_cmp(&dr.fold_b[a as usize]));
    ids.extend(prior.into_iter().take(n - ids.len()));
    ids.sort_unstable();
    let gather = |t: &crate::qtensor::QTensor| -> Option<Mat> {
        let (codes, rs, col) = t.q8_2f_codes()?;
        let cols = t.cols();
        let mut q = Vec::with_capacity(n * cols);
        let mut r = Vec::with_capacity(n);
        for &id in &ids {
            q.extend_from_slice(&codes[id as usize * cols..(id as usize + 1) * cols]);
            r.push(rs[id as usize]);
        }
        Some(Mat::Q8B { buf: up(c, &q), rows: n, cols, rs: up(c, &r), col: up(c, col) })
    };
    let fb: Vec<f32> = ids.iter().map(|&id| dr.fold_b[id as usize]).collect();
    Some((gather(&g.lm_head)?, gather(&dr.fold)?, up(c, &fb), up(c, &ids), n))
}

/// The drafter on the GPU, or `None` (and plain decoding) when its matrices
/// are not q8_2f or its geometry is not the arm's.
fn build_draft(c: &Ctx, g: &Globals, dr: &crate::inkling::Draft, cfg: &Cfg, rows: usize) -> Option<DraftM> {
    let model = &g.model;
    let hidden = cfg.dim;
    let (nh, nkv, hd) = (cfg.nh, cfg.nkv, cfg.hd);
    let (qs, kvs) = (nh * hd, nkv * hd);
    let b = dr.block;
    let q8 = |t: &crate::qtensor::QTensor| mat(c, model, t).filter(|m| matches!(m, Mat::Q8 { .. }));
    if dr.layers.is_empty() || dr.targets.len() < 2 || !(2..=VROWS).contains(&b) || dr.rank * 3 > 768 || hidden > 4096 || hd > 256 {
        return None;
    }
    let f4 = |n: usize| shared(c, n * 4);
    let inter = dr.layers[0].gate.rows();
    let mut layers = Vec::with_capacity(dr.layers.len());
    for dl in &dr.layers {
        let cap = if dl.local { dr.window + 64 } else { 4096 };
        layers.push(DLayerM {
            q: q8(&dl.q)?,
            k: q8(&dl.k)?,
            v: q8(&dl.v)?,
            o: q8(&dl.o)?,
            kc: q8(&dl.k_ctx)?,
            vc: q8(&dl.v_ctx)?,
            gate: q8(&dl.gate)?,
            up: q8(&dl.up)?,
            down: q8(&dl.down)?,
            in_norm: up(c, &dl.in_norm),
            post_norm: up(c, &dl.post_norm),
            q_norm: up(c, &dl.q_norm),
            k_norm: up(c, &dl.k_norm),
            kc_norm: up(c, &dl.k_ctx_norm),
            ck: shared(c, cap * kvs * 2),
            cv: shared(c, cap * kvs * 2),
            cap,
            ring: dl.local,
        });
    }
    let nt = dr.targets.len();
    let mut norms = Vec::with_capacity((dr.layers.len() + 1) * hidden);
    for dl in &dr.layers {
        norms.extend_from_slice(&dl.dflare_norm);
    }
    norms.extend_from_slice(&dr.markov_norm);
    let vocab = g.lm_head.rows();
    let rank = dr.rank;
    let mut w1 = vec![0u16; dr.w1.rows() * rank];
    let mut row = vec![0f32; rank];
    for r in 0..dr.w1.rows() {
        dr.w1.row_f32(r, &mut row);
        for (o, &v) in w1[r * rank..(r + 1) * rank].iter_mut().zip(&row) {
            *o = cortiq_core::quant::f32_to_f16(v);
        }
    }
    let widest = hidden.max(qs).max(inter).max(rank);
    Some(DraftM {
        layers,
        taps: dr.targets[..nt - 1].to_vec(),
        tap: f4(nt * rows * hidden),
        mix: f4(dr.layers.len() * rows * hidden),
        fr: f4(FR * hidden),
        alpha: up(c, &dr.alpha),
        norms: up(c, &norms),
        fnorm: up(c, &dr.norm),
        masks: dr.mask.clone(),
        x: f4(b * hidden),
        xn: f4(b * hidden),
        dq: f4(b * qs),
        dk: f4(b * kvs),
        dv: f4(b * kvs),
        att: f4(b * qs),
        gb: f4(b * inter),
        ub: f4(b * inter),
        ab: f4(b * inter),
        hf: f4(b * hidden),
        xc: f4(VROWS * widest),
        kcb: f4(VROWS * kvs),
        vcb: f4(VROWS * kvs),
        dlog: f4((b - 1) * vocab),
        rnn_w: up(c, &dr.rnn_w),
        rnn_b: up(c, &dr.rnn_b),
        ch: f4((b - 1) * 3 * rank),
        cf: f4(3 * rank),
        state: f4(rank),
        prev: shared(c, 64),
        ids: shared(c, 64),
        probs: shared(c, 64),
        w1: up(c, &w1),
        fold: q8(&dr.fold)?,
        fold_b: up(c, &dr.fold_b),
        table: draft_table().and_then(|n| gather_table(c, g, dr, n)),
        lat: f4(rank),
        block: b,
        rank,
        window: dr.window,
        theta: dr.rope_theta as f32,
        eps: dr.eps,
        inter,
    })
}

impl Dev {
    /// Room for positions `..need` in every full slot (shared memory: a copy).
    fn ensure(&mut self, c: &Ctx, need: usize, kvs: usize) {
        if let Some(dr) = self.draft.as_mut() {
            let need = need + dr.block;
            for l in dr.layers.iter_mut().filter(|l| !l.ring && l.cap < need) {
                let cap = need.max(l.cap * 2).next_power_of_two();
                let (k, v) = (shared(c, cap * kvs * 2), shared(c, cap * kvs * 2));
                let old = l.cap * kvs * 2;
                unsafe {
                    std::ptr::copy_nonoverlapping(l.ck.contents() as *const u8, k.contents() as *mut u8, old);
                    std::ptr::copy_nonoverlapping(l.cv.contents() as *const u8, v.contents() as *mut u8, old);
                }
                (l.ck, l.cv, l.cap) = (k, v, cap);
            }
        }
        for s in self.slots.iter_mut().filter(|s| s.ring == 0 && s.cap < need) {
            let cap = need.max(s.cap * 2).next_power_of_two();
            let (k, v) = (shared(c, cap * kvs * 2), shared(c, cap * kvs * 2));
            let old = s.cap * kvs * 2;
            unsafe {
                std::ptr::copy_nonoverlapping(s.k.contents() as *const u8, k.contents() as *mut u8, old);
                std::ptr::copy_nonoverlapping(s.v.contents() as *const u8, v.contents() as *mut u8, old);
            }
            let rings = std::mem::replace(&mut s.rings, [shared(c, 16), shared(c, 16), shared(c, 16), shared(c, 16)]);
            let hist = std::mem::replace(&mut s.hist, shared(c, 16));
            *s = SlotM { k, v, cap, ring: 0, rings, hist };
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
    // `CMF_INKLING_WINDOW_PROBE=1` (a measurement, not a run): after a prompt,
    // the time of a call of 1..8 rows at the tip (the last row's logits), as a
    // draft verify's trunk would take it; the sequence's state is spoiled
    if pos0 > 0 && std::env::var("CMF_INKLING_WINDOW_PROBE").as_deref() == Ok("1") {
        let mut line = Vec::new();
        for n in [1usize, 2, 3, 4, 6, 8] {
            let mut best = f64::MAX;
            for t in 0..12 {
                let t0 = std::time::Instant::now();
                let cmd = c.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                encode_call(c, p, d, cfg, enc, n, pos0, 1, probe_vm());
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                if t > 1 {
                    best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            line.push(format!("{n}: {best:.2} ms"));
        }
        tracing::info!("inkling2_loop Metal window probe (a call, the last row's logits, best of 10): {}", line.join(", "));
        std::process::exit(0);
    }
    let mut at = 0usize;
    while at < ids.len() {
        // every row's logits come 16 rows a call (the logits buffer's rows)
        let n = (ids.len() - at).min(if all { d.rows.min(16) } else { d.rows });
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
        encode_call(c, p, d, cfg, enc, n, pos0 + at, lrows, false);
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

/// `CMF_INKLING_VPROJ=rows`: a verify's projections and O a decode matvec a
/// row instead of the narrow GEMM.
fn vproj_rows() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_VPROJ").as_deref() == Ok("rows"))
}

/// DSpark on Metal only when `CMF_INKLING_SPEC` names it (`auto` or a depth):
/// on the M4 a verify's second row costs ~40 % of a step (the experts'
/// unpacking), so drafts gain ~7 % on code and JSON at the best depth and lose
/// on prose; without it the drafter is not built and nothing is committed.
fn spec_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_SPEC").is_ok_and(|v| v != "0"))
}

/// `CMF_INKLING_PROBE_VM=1`: the window probe times a draft verify's call
/// (every row through the decode kernels) instead of a prompt chunk's.
fn probe_vm() -> bool {
    std::env::var("CMF_INKLING_PROBE_VM").as_deref() == Ok("1")
}

/// A draft verify: `ids` (the anchor and its drafts, up to `VROWS`) from
/// `pos0` through the target, every row's sums those of decoding it alone;
/// each row's greedy pick into `picks`, the logits left for `row_logits`.
pub(crate) fn verify(id: u64, g: &Globals, layers: &[Layer], cfg: &Cfg, ids: &[u32], pos0: usize, picks: &mut Vec<u32>) -> bool {
    let Some(c) = ctx() else { return false };
    let n = ids.len();
    if n == 0 || n > VROWS || !admits(layers, cfg) {
        return false;
    }
    let mut guard = DEVICES.lock().unwrap();
    let Some(d) = guard.as_mut().and_then(|m| m.get_mut(&id)) else { return false };
    d.ensure(c, pos0 + n, cfg.nkv * cfg.hd);
    let hidden = cfg.dim;
    let vocab = d.head.rows();
    let p = pipes(c).expect("pipelines");
    let mut e = vec![0f32; hidden];
    let hp = d.h[0].contents() as *mut f32;
    for (i, &tid) in ids.iter().enumerate() {
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
    let t0 = std::time::Instant::now();
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_call(c, p, d, cfg, enc, n, pos0, n, true);
    // every row's greedy pick on the GPU (the first of equal maxima)
    enc.set_compute_pipeline_state(&p.argmax);
    enc.set_buffer(0, Some(&d.logits), 0);
    enc.set_buffer(1, Some(&d.vpick), 0);
    set_bytes_t(enc, 2, &(vocab as u32));
    grid(enc, n as u64, 1, 1, 1024);
    enc.end_encoding();
    let t1 = std::time::Instant::now();
    cmd.commit();
    cmd.wait_until_completed();
    let t2 = std::time::Instant::now();
    picks.clear();
    picks.extend_from_slice(unsafe { std::slice::from_raw_parts(d.vpick.contents() as *const u32, n) });
    if std::env::var("CMF_INKLING_SPEC_TIME").is_ok() {
        static T: Mutex<(f64, f64, f64, u32)> = Mutex::new((0.0, 0.0, 0.0, 0));
        let mut t = T.lock().unwrap();
        t.0 += (t1 - t0).as_secs_f64() * 1e3;
        t.1 += (t2 - t1).as_secs_f64() * 1e3;
        t.2 += t2.elapsed().as_secs_f64() * 1e3;
        t.3 += 1;
        if t.3 % 64 == 0 {
            let k = t.3 as f64;
            eprintln!("metal verify: encode {:.2} ms, GPU {:.2} ms, picks {:.2} ms", t.0 / k, t.1 / k, t.2 / k);
        }
    }
    true
}

/// Row `r`'s logits of the last verify.
pub(crate) fn row_logits(id: u64, r: usize, vocab: usize, out: &mut Vec<f32>) -> bool {
    let guard = DEVICES.lock().unwrap();
    let Some(d) = guard.as_ref().and_then(|m| m.get(&id)) else { return false };
    if r >= VROWS || d.head.rows() != vocab {
        return false;
    }
    let lg = unsafe { std::slice::from_raw_parts(d.logits.contents() as *const f32, VROWS.min(16) * vocab) };
    out.clear();
    out.extend_from_slice(&lg[r * vocab..(r + 1) * vocab]);
    true
}

/// Every application's convolution rings saved before a draft verify (shared
/// memory: copied on the host).
pub(crate) fn ring_snap(id: u64) -> bool {
    let mut guard = DEVICES.lock().unwrap();
    let Some(d) = guard.as_mut().and_then(|m| m.get_mut(&id)) else { return false };
    let kc = KC;
    let w = d.rw;
    let per: usize = w.iter().map(|&x| (kc - 1) * x).sum();
    let mut snap = std::mem::take(&mut d.snap);
    snap.resize(d.slots.len() * per, 0.0);
    let mut at = 0usize;
    for s in &d.slots {
        for (j, r) in s.rings.iter().enumerate() {
            let len = (kc - 1) * w[j];
            let src = unsafe { std::slice::from_raw_parts(r.contents() as *const f32, len) };
            snap[at..at + len].copy_from_slice(src);
            at += len;
        }
    }
    d.snap = snap;
    true
}

/// The rings put back as they stand after the verify's first `r` rows: ring
/// row l−1 is the call's input row r−l, or the snapshot's row l−1−r before it.
pub(crate) fn ring_back(id: u64, r: usize) -> bool {
    let guard = DEVICES.lock().unwrap();
    let Some(d) = guard.as_ref().and_then(|m| m.get(&id)) else { return false };
    let kc = KC;
    let w = d.rw;
    let per: usize = w.iter().map(|&x| (kc - 1) * x).sum();
    if r == 0 || r > VROWS || d.snap.len() != d.slots.len() * per {
        return false;
    }
    let mut at = 0usize;
    for s in &d.slots {
        let hist = s.hist.contents() as *const u8;
        for (j, ring) in s.rings.iter().enumerate() {
            let wj = w[j];
            let live = ring.contents() as *mut f32;
            let hrows = unsafe { hist.add(hist_off(j, w[0], w[2]) as usize) as *const f32 };
            for l in 1..kc {
                let src = if r >= l { unsafe { hrows.add((r - l) * wj) } } else { d.snap[at + (l - 1 - r) * wj..].as_ptr() };
                unsafe { std::ptr::copy_nonoverlapping(src, live.add((l - 1) * wj), wj) };
            }
            at += (kc - 1) * wj;
        }
    }
    true
}

/// One block of the drafter from the anchor at `pos` (the target has run every
/// position before it): up to `k` drafts (`inkling.cu`'s `draft_block`), one
/// command buffer.
pub(crate) fn draft(id: u64, g: &Globals, cfg: &Cfg, anchor: u32, pos: usize, k: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    let c = ctx()?;
    let mut guard = DEVICES.lock().unwrap();
    let d = guard.as_mut()?.get_mut(&id)?;
    if pos == 0 || k == 0 {
        return None;
    }
    d.ensure(c, pos + VROWS, cfg.nkv * cfg.hd);
    let dr = d.draft.as_ref()?;
    let p = pipes(c)?;
    let b = dr.block;
    let k = k.min(b - 1);
    let hidden = cfg.dim;
    let vocab = d.head.rows();
    let (nh, nkv, hd) = (cfg.nh, cfg.nkv, cfg.hd);
    // the anchor's normed embedding, as the target's input row, and the masks
    let mut e = vec![0f32; hidden];
    if (anchor as usize) < g.embed.rows() {
        g.embed.row_f32(&g.model, anchor as usize, &mut e);
    }
    let ms = e.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
    let r = (1.0 / (ms + cfg.eps).sqrt()) as f32;
    unsafe {
        let xp = dr.x.contents() as *mut f32;
        for (j, (&v, &w)) in e.iter().zip(&g.embed_norm).enumerate() {
            *xp.add(j) = w * (v * r);
        }
        for i in 1..b {
            std::ptr::copy_nonoverlapping(dr.masks.as_ptr(), xp.add(i * hidden), hidden);
        }
        std::ptr::write_bytes(dr.state.contents() as *mut u8, 0, dr.rank * 4);
        *(dr.prev.contents() as *mut u32) = anchor;
    }
    let mut ip = base_ip(cfg, b, pos, d.rows);
    ip.eps = dr.eps;
    let dp = DsP { eps: dr.eps, theta: dr.theta, pos0: pos as u32, b: b as u32, r: dr.rank as u32, ..DsP::default() };
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let rms = |x: &Buffer, w: &Buffer, o: &Buffer| {
        enc.set_compute_pipeline_state(&p.rms);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(o), 0);
        setp(enc, &ip);
        grid(enc, b as u64, 1, 1, 256);
    };
    let dskip = |what: &str| std::env::var("CMF_INKLING_DSKIP").is_ok_and(|v| v.split(',').any(|x| x == what));
    for l in dr.layers.iter().filter(|_| !dskip("layers")) {
        let dpl = DsP { cap: l.cap as u32, ring: l.ring as u32, win: if l.ring { dr.window as u32 } else { 0 }, ..dp };
        rms(&dr.x, &l.in_norm, &dr.xn);
        q8_rows(p, enc, d, &l.q, (&dr.xn, 0), (&dr.dq, 0), b, false);
        q8_rows(p, enc, d, &l.k, (&dr.xn, 0), (&dr.dk, 0), b, false);
        q8_rows(p, enc, d, &l.v, (&dr.xn, 0), (&dr.dv, 0), b, false);
        enc.set_compute_pipeline_state(&p.ds_qk);
        enc.set_buffer(0, Some(&dr.dq), 0);
        enc.set_buffer(1, Some(&dr.dk), 0);
        enc.set_buffer(2, Some(&dr.dv), 0);
        enc.set_buffer(3, Some(&l.q_norm), 0);
        enc.set_buffer(4, Some(&l.k_norm), 0);
        enc.set_buffer(5, Some(&l.ck), 0);
        enc.set_buffer(6, Some(&l.cv), 0);
        set_bytes_t(enc, 14, &dpl);
        setp(enc, &ip);
        grid(enc, b as u64, (nh + nkv) as u64, 1, hd as u64);
        enc.set_compute_pipeline_state(&p.ds_attn);
        enc.set_buffer(0, Some(&dr.dq), 0);
        enc.set_buffer(1, Some(&l.ck), 0);
        enc.set_buffer(2, Some(&l.cv), 0);
        enc.set_buffer(3, Some(&dr.att), 0);
        set_bytes_t(enc, 14, &dpl);
        setp(enc, &ip);
        grid(enc, nh as u64, b as u64, 1, 256);
        q8_rows(p, enc, d, &l.o, (&dr.att, 0), (&dr.x, 0), b, true);
        rms(&dr.x, &l.post_norm, &dr.xn);
        q8_rows(p, enc, d, &l.gate, (&dr.xn, 0), (&dr.gb, 0), b, false);
        q8_rows(p, enc, d, &l.up, (&dr.xn, 0), (&dr.ub, 0), b, false);
        enc.set_compute_pipeline_state(&p.ds_act);
        enc.set_buffer(0, Some(&dr.gb), 0);
        enc.set_buffer(1, Some(&dr.ub), 0);
        enc.set_buffer(2, Some(&dr.ab), 0);
        set_bytes_t(enc, 3, &(dr.inter as u32));
        grid(enc, (dr.inter as u64).div_ceil(256), b as u64, 1, 256);
        q8_rows(p, enc, d, &l.down, (&dr.ab, 0), (&dr.x, 0), b, true);
    }
    // h; the rows after the anchor's through the target's head (the drafts' base logits)
    rms(&dr.x, &dr.fnorm, &dr.hf);
    let (thead, tfold, tfb, tmap, tv) = match &dr.table {
        Some((h, f, b, m, n)) => (h, f, b, Some(m), *n),
        None => (&d.head, &dr.fold, &dr.fold_b, None, vocab),
    };
    if !dskip("head") {
        q8_rows(p, enc, d, thead, (&dr.hf, (hidden * 4) as u64), (&dr.dlog, 0), k, false);
    }
    // the recurrence's input terms: W_h · h_{i+1} for every draft, W_f · f
    // (the Markov context of position pos − 1)
    let rk = dr.rank;
    let ws = 2 * rk + 2 * hidden;
    let fmv = |x: (&Buffer, u64), y: &Buffer, off: usize, rows: usize| {
        let fm: [u32; 4] = [(3 * rk) as u32, ws as u32, off as u32, hidden as u32];
        enc.set_compute_pipeline_state(&p.ds_fmv);
        enc.set_buffer(0, Some(&dr.rnn_w), 0);
        enc.set_buffer(1, Some(x.0), x.1);
        enc.set_buffer(2, Some(y), 0);
        set_bytes_t(enc, 3, &fm);
        grid(enc, ((3 * rk) as u64).div_ceil(8), rows as u64, 1, 256);
    };
    fmv((&dr.hf, (hidden * 4) as u64), &dr.ch, 2 * rk, k);
    fmv((&dr.fr, (((pos - 1) % FR) * hidden * 4) as u64), &dr.cf, 2 * rk + hidden, 1);
    let wr: [u32; 2] = [ws as u32, rk as u32];
    for i in 0..k {
        enc.set_compute_pipeline_state(&p.ds_markov);
        enc.set_buffer(0, Some(&dr.ch), (i * 3 * rk * 4) as u64);
        enc.set_buffer(1, Some(&dr.cf), 0);
        enc.set_buffer(2, Some(&dr.rnn_b), 0);
        enc.set_buffer(3, Some(&dr.rnn_w), 0);
        enc.set_buffer(4, Some(&dr.state), 0);
        enc.set_buffer(5, Some(&dr.prev), 0);
        enc.set_buffer(6, Some(&dr.w1), 0);
        enc.set_buffer(7, Some(&dr.lat), 0);
        set_bytes_t(enc, 8, &wr);
        grid(enc, 1, 1, 1, (3 * rk) as u64);
        if !dskip("fold") {
            q8_rows(p, enc, d, tfold, (&dr.lat, 0), (&dr.dlog, (i * tv * 4) as u64), 1, true);
        }
        #[repr(C)]
        struct DsPick {
            v: u32,
            i: u32,
            map: u32,
            pad: u32,
            cap: f32,
        }
        let vi = DsPick { v: tv as u32, i: i as u32, map: tmap.is_some() as u32, pad: 0, cap: cfg.logit_cap };
        enc.set_compute_pipeline_state(&p.ds_pick);
        enc.set_buffer(0, Some(&dr.dlog), (i * tv * 4) as u64);
        enc.set_buffer(1, Some(tfb), 0);
        enc.set_buffer(2, Some(&dr.ids), 0);
        enc.set_buffer(3, Some(&dr.prev), 0);
        enc.set_buffer(5, Some(tmap.unwrap_or(&dr.ids)), 0);
        enc.set_buffer(6, Some(&dr.probs), 0);
        set_bytes_t(enc, 4, &vi);
        grid(enc, 1, 1, 1, 1024);
    }
    enc.end_encoding();
    let t1 = std::time::Instant::now();
    cmd.commit();
    cmd.wait_until_completed();
    if std::env::var("CMF_INKLING_SPEC_TIME").is_ok() {
        static T: Mutex<(f64, u32)> = Mutex::new((0.0, 0));
        let mut t = T.lock().unwrap();
        t.0 += t1.elapsed().as_secs_f64() * 1e3;
        t.1 += 1;
        if t.1 % 64 == 0 {
            eprintln!("metal draft: GPU {:.2} ms", t.0 / t.1 as f64);
        }
    }
    let ids = unsafe { std::slice::from_raw_parts(dr.ids.contents() as *const u32, k) };
    let probs = unsafe { std::slice::from_raw_parts(dr.probs.contents() as *const f32, k) };
    Some((ids.to_vec(), probs.to_vec()))
}

fn grid(enc: &ComputeCommandEncoderRef, x: u64, y: u64, z: u64, threads: u64) {
    enc.dispatch_thread_groups(MTLSize::new(x.max(1), y.max(1), z.max(1)), MTLSize::new(threads, 1, 1));
}

/// `m · x` for `n` rows: x rows `cols` apart from `xoff` bytes, y rows `rows` apart.
#[allow(clippy::too_many_arguments)]
fn mv(c: &Ctx, enc: &ComputeCommandEncoderRef, fbuf: &WeightArena, m: &Mat, xs: &Buffer, xoff: usize, y: &Buffer, yoff: usize, n: usize, vm: bool) {
    // a prompt chunk's q4tp matrix: the backend's simdgroup GEMM reads it once
    // for every row (its activations staged as half)
    if let Mat::Q4 { abs, rows, cols } = m {
        // (a draft verify: a decode matvec a row, a row's sums as decoding's)
        if n > 1 && !vm && gemm_on() && xoff % (cols * 4) == 0 && yoff % (rows * 4) == 0 {
            super::enc_q4tp_mm_rows(c, enc, fbuf, *abs, xs, xoff / (cols * 4), y, yoff / (rows * 4), n, *rows, *cols);
            return;
        }
    }
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
            Mat::Q8B { .. } => unreachable!("a gathered table goes through q8_rows"),
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

/// `CMF_INKLING_SKIP=attn,gates,proj,o,moe` (a measurement, not a run): the
/// named stages are not encoded, so a call's time without them shows what
/// they cost (the output is wrong).
fn skip(stage: &str) -> bool {
    static S: OnceLock<Vec<String>> = OnceLock::new();
    S.get_or_init(|| std::env::var("CMF_INKLING_SKIP").map(|v| v.split(',').map(|x| x.trim().to_string()).collect()).unwrap_or_default()).iter().any(|x| x == stage)
}

/// `CMF_INKLING_FLASH=0`: a prompt call's attention a threadgroup a (head, row)
/// (`ink_attn`) instead of the flash kernel on the simdgroup matrices.
fn flash_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_FLASH").as_deref() != Ok("0"))
}

/// `CMF_INKLING_FUSE=0`: a call's stages one launch each (the mixes and norms,
/// the five projections, the small matvecs, qr and kv, the attention and its
/// gate) instead of the fused launches.
fn fuse_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_FUSE").as_deref() != Ok("0"))
}

/// `CMF_INKLING_DEC8=0`: the decode's experts on `ink_moe_gu` / `ink_moe_down`
/// (a simdgroup's 32 lanes over a row block's groups) instead of
/// `ink_moe_gu8` / `ink_moe_dn8` (halves and quarters of a simdgroup, every
/// lane busy in every pass).
fn dec8_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_DEC8").as_deref() != Ok("0"))
}

/// `CMF_INKLING_GATE_GEMM=0`: a prompt call's message gates a channel a
/// simdgroup instead of three f32 GEMMs on the simdgroup matrices.
fn gate_gemm_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_GATE_GEMM").as_deref() != Ok("0"))
}

/// `CMF_INKLING_GEMM=0`: a chunk's dense matrices a matvec a row.
fn gemm_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_GEMM").as_deref() != Ok("0"))
}

/// A chunk's routed experts (`CMF_INKLING_MOE`): `gemm` (the default) groups
/// the (row, pick) entries by expert and runs each expert as a simdgroup GEMM
/// over its rows; `loop` an expert's rows a matvec at a time; `pick` the decode
/// kernels, a (row, pick) at a time.
fn moe_mode() -> u8 {
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("CMF_INKLING_MOE").as_deref() {
        Ok("pick") => 0,
        Ok("loop") => 1,
        _ => 2,
    })
}

fn base_ip(cfg: &Cfg, n: usize, pos0: usize, rows: usize) -> IP {
    IP {
        h: cfg.dim as u32,
        nh: cfg.nh as u32,
        nkv: cfg.nkv as u32,
        hd: cfg.hd as u32,
        dr: cfg.d_rel as u32,
        kc: KC as u32,
        n: n as u32,
        pos0: pos0 as u32,
        bstride: (MAXB * cfg.dim) as u32,
        ostride: (rows * cfg.dim) as u32,
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
fn encode_call(c: &Ctx, p: &Pipes, d: &Dev, cfg: &Cfg, enc: &ComputeCommandEncoderRef, n: usize, pos0: usize, lrows: usize, vm: bool) {
    let hidden = cfg.dim;
    let mut base = base_ip(cfg, n, pos0, d.rows);
    base.hist = vm as u32;
    let hb = (n * hidden * 4) as u64;
    let mut cur = 0usize;
    let mut hin: (&Buffer, u64) = (&d.h[0], 0);
    let mut nb = 0usize;
    let mut npre = 0usize;
    for (ai, (li, pass)) in cfg.plan().into_iter().enumerate() {
        let l = &d.layers[li];
        let s = &d.slots[cfg.slot(li, pass)];
        // the drafter's taps: this application's input rows
        if let Some(dr) = &d.draft {
            if let Some(k) = dr.taps.iter().position(|&t| t == ai) {
                copy_at(p, enc, (hin.0, hin.1), (&dr.tap, (k * d.rows * hidden * 4) as u64), hb);
            }
        }
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
        app(c, p, d, cfg, enc, l, s, li, pass, n, &base, hin.0, hout, &mut nb, npre, vm);
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
    if let Some(dr) = d.draft.as_ref().filter(|_| spec_on()) {
        draft_commit(p, d, dr, cfg, enc, &base, n, pos0);
    }
    if lrows > 0 {
        let r0 = n - lrows;
        enc.set_compute_pipeline_state(&p.rms);
        enc.set_buffer(0, Some(&d.fin), (r0 * hidden * 4) as u64);
        enc.set_buffer(1, Some(&d.final_norm), 0);
        enc.set_buffer(2, Some(&d.xf), 0);
        setp(enc, &ip);
        grid(enc, lrows as u64, 1, 1, 256);
        match &d.head {
            // a verify's rows: the head read once for all of them
            Mat::Q8 { .. } if vm && std::env::var("CMF_INKLING_VHEAD").as_deref() != Ok("rows") => q8_rows(p, enc, d, &d.head, (&d.xf, 0), (&d.logits, 0), lrows, false),
            _ => mv(c, enc, &d.fbuf, &d.head, &d.xf, 0, &d.logits, 0, lrows, vm),
        }
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

/// `m · x` for `n` ≤ 8 rows of a q8_2f matrix (x `[n][cols]`, y `[n][rows]`;
/// `add`: y += m · x): x·col once into the drafter's or the device's scratch,
/// then `ink_q8p`, the matrix read once for all the rows.
#[allow(clippy::too_many_arguments)]
fn q8_rows(p: &Pipes, enc: &ComputeCommandEncoderRef, d: &Dev, m: &Mat, x: (&Buffer, u64), y: (&Buffer, u64), n: usize, add: bool) {
    let (wbuf, rows, cols, rs, col) = match m {
        Mat::Q8 { abs, rows, cols, rs, col } => (Err(*abs), rows, cols, rs, col),
        Mat::Q8B { buf, rows, cols, rs, col } => (Ok(buf), rows, cols, rs, col),
        Mat::Q4 { .. } => return,
    };
    let xc = d.draft.as_ref().map_or(&d.pxc, |dr| &dr.xc);
    let (c4, r) = ((cols / 4) as u32, *rows as u32);
    enc.set_compute_pipeline_state(&p.xcol);
    enc.set_buffer(0, Some(x.0), x.1);
    enc.set_buffer(1, Some(col), 0);
    enc.set_buffer(2, Some(xc), 0);
    set_bytes_t(enc, 3, &c4);
    grid(enc, ((cols / 4) as u64).div_ceil(256), n as u64, 1, 256);
    let ip = IP { n: n as u32, v: add as u32, ..IP::default() };
    // four rows and more (the drafter's block): the matrix unit, x·col as half
    let mm = n >= q8_mm_min() && cols % 64 == 0 && q8_mm_on();
    enc.set_compute_pipeline_state(if mm { &p.q8mm8 } else { &p.q8p });
    match wbuf {
        Ok(b) => enc.set_buffer(0, Some(b), 0),
        Err(abs) => d.fbuf.bind(enc, 0, abs),
    }
    enc.set_buffer(1, Some(xc), 0);
    enc.set_buffer(2, Some(rs), 0);
    enc.set_buffer(3, Some(y.0), y.1);
    if mm {
        set_bytes_t(enc, 4, &(*cols as u32));
        set_bytes_t(enc, 5, &r);
        setp(enc, &ip);
        grid(enc, (*rows as u64).div_ceil(64), 1, 1, 128);
    } else {
        set_bytes_t(enc, 4, &c4);
        set_bytes_t(enc, 5, &r);
        setp(enc, &ip);
        grid(enc, (*rows as u64).div_ceil(32), 1, 1, 256);
    }
}

/// Rows from which a q8_2f product takes the matrix unit (`CMF_INKLING_Q8MM_MIN`).
fn q8_mm_min() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("CMF_INKLING_Q8MM_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(4))
}

/// `CMF_INKLING_Q8MM=0`: q8_2f products of four rows and more on `ink_q8p`
/// (f32 lanes) instead of the matrix unit.
fn q8_mm_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_Q8MM").as_deref() != Ok("0"))
}

/// `bytes` of `src` from its offset into `dst` at its offset.
fn copy_at(p: &Pipes, enc: &ComputeCommandEncoderRef, src: (&Buffer, u64), dst: (&Buffer, u64), bytes: u64) {
    let floats = bytes / 4;
    let ip = IP { rows: floats as u32, ..IP::default() };
    enc.set_compute_pipeline_state(&p.copy);
    enc.set_buffer(0, Some(src.0), src.1);
    enc.set_buffer(1, Some(dst.0), dst.1);
    setp(enc, &ip);
    grid(enc, floats.div_ceil(256), 1, 1, 256);
}

/// The drafter's context of a call's `n` rows from `pos0` (`inkling.cu`'s
/// `draft_commit`): the final normed state as the last tap, the mixes, every
/// drafter layer's context keys and values at the rows' positions.
#[allow(clippy::too_many_arguments)]
fn draft_commit(p: &Pipes, d: &Dev, dr: &DraftM, cfg: &Cfg, enc: &ComputeCommandEncoderRef, base: &IP, n: usize, pos0: usize) {
    let hidden = cfg.dim;
    let nt = dr.taps.len() + 1;
    let nl = dr.layers.len();
    enc.set_compute_pipeline_state(&p.rms);
    enc.set_buffer(0, Some(&d.fin), 0);
    enc.set_buffer(1, Some(&d.final_norm), 0);
    enc.set_buffer(2, Some(&dr.tap), ((nt - 1) * d.rows * hidden * 4) as u64);
    setp(enc, base);
    grid(enc, n as u64, 1, 1, 256);
    let dp = DsP { eps: dr.eps, theta: dr.theta, pos0: pos0 as u32, ..DsP::default() };
    let dm: [u32; 4] = [nt as u32, d.rows as u32, (nl + 1) as u32, FR as u32];
    enc.set_compute_pipeline_state(&p.ds_mix);
    enc.set_buffer(0, Some(&dr.tap), 0);
    enc.set_buffer(1, Some(&dr.alpha), 0);
    enc.set_buffer(2, Some(&dr.norms), 0);
    enc.set_buffer(3, Some(&dr.mix), 0);
    enc.set_buffer(4, Some(&dr.fr), 0);
    set_bytes_t(enc, 5, &dm);
    set_bytes_t(enc, 14, &dp);
    setp(enc, base);
    grid(enc, n as u64, (nl + 1) as u64, 1, 256);
    for (li, l) in dr.layers.iter().enumerate() {
        let mut c0 = 0usize;
        while c0 < n {
            let m = (n - c0).min(VROWS);
            let xo = ((li * d.rows + c0) * hidden * 4) as u64;
            q8_rows(p, enc, d, &l.kc, (&dr.mix, xo), (&dr.kcb, 0), m, false);
            q8_rows(p, enc, d, &l.vc, (&dr.mix, xo), (&dr.vcb, 0), m, false);
            let dpl = DsP { cap: l.cap as u32, ring: l.ring as u32, pos0: (pos0 + c0) as u32, ..dp };
            enc.set_compute_pipeline_state(&p.ds_ctx);
            enc.set_buffer(0, Some(&dr.kcb), 0);
            enc.set_buffer(1, Some(&dr.vcb), 0);
            enc.set_buffer(2, Some(&l.kc_norm), 0);
            enc.set_buffer(3, Some(&l.ck), 0);
            enc.set_buffer(4, Some(&l.cv), 0);
            set_bytes_t(enc, 14, &dpl);
            setp(enc, base);
            grid(enc, m as u64, cfg.nkv as u64, 1, cfg.hd as u64);
            c0 += m;
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
/// `vm`, a draft verify (up to `VROWS` rows): every row through the decode
/// kernels, so its sums are those of decoding it alone, and the convolutions
/// keep their input rows for a rollback.
fn app(c: &Ctx, p: &Pipes, d: &Dev, cfg: &Cfg, enc: &ComputeCommandEncoderRef, l: &LayerM, s: &SlotM, li: usize, pass: usize, n: usize, base: &IP, hin: &Buffer, hout: &Buffer, nb: &mut usize, npre: usize, vm: bool) {
    let hidden = cfg.dim;
    let kvs = cfg.nkv * cfg.hd;
    let nu = n as u64;
    let plane = (d.rows * hidden * 4) as u64;
    let fuse = fuse_on();
    let gates = if pass > 0 { l.gates.as_ref() } else { None };
    let one = l.sliding && gates.is_none();
    // 1. the attention inputs (fused: a local layer's one norm in the mix)
    let mut inp: [(&Buffer, u64); 3] = [(hin, 0), (hin, 0), (hin, 0)];
    let mut normed = false;
    if let Some((aq, nq)) = &l.aq {
        let mut ip = *base;
        ip.nb = *nb as u32;
        enc.set_compute_pipeline_state(&p.mix);
        enc.set_buffer(0, Some(&d.blk), 0);
        enc.set_buffer(1, Some(hin), 0);
        enc.set_buffer(2, Some(aq), 0);
        enc.set_buffer(3, Some(&d.in3), 0);
        if fuse && one {
            ip.v = 1;
            enc.set_buffer(4, Some(&l.norms[0]), 0);
            enc.set_buffer(5, Some(&d.x[0]), 0);
            normed = true;
        }
        setp(enc, &ip);
        grid(enc, *nq as u64, nu, 1, 256);
        inp = if *nq == 3 { [(&d.in3, 0), (&d.in3, plane), (&d.in3, 2 * plane)] } else { [(&d.in3, 0); 3] };
    }
    // 2. the messages of a later pass, gated into q, k, v
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
        let r3 = gs[0].3;
        if fuse && (n == 1 || vm) && !skip("gates") && gs[..3].iter().all(|g| g.3 == r3) && inp.iter().all(|x| std::ptr::eq(x.0, inp[0].0)) {
            // the three gates' halves a launch each
            #[repr(C)]
            struct G3Off {
                xo: [u32; 3],
                mo: [u32; 3],
                oo: [u32; 3],
            }
            let pf = (plane / 4) as u32;
            let go = G3Off { xo: [(inp[0].1 / 4) as u32, (inp[1].1 / 4) as u32, (inp[2].1 / 4) as u32], mo: [0, pf, 2 * pf], oo: [0, pf, 2 * pf] };
            let mut ip = *base;
            ip.r = r3 as u32;
            enc.set_compute_pipeline_state(&p.gate_down3);
            enc.set_buffer(0, Some(inp[0].0), 0);
            enc.set_buffer(1, Some(&d.msg), 0);
            enc.set_buffer(2, Some(&gs[0].0), 0);
            enc.set_buffer(3, Some(&gs[1].0), 0);
            enc.set_buffer(4, Some(&gs[2].0), 0);
            enc.set_buffer(5, Some(&d.gd), 0);
            set_bytes_t(enc, 6, &go);
            setp(enc, &ip);
            grid(enc, r3 as u64, nu, 3, 256);
            enc.set_compute_pipeline_state(&p.gate_out3);
            enc.set_buffer(0, Some(inp[0].0), 0);
            enc.set_buffer(1, Some(&d.msg), 0);
            enc.set_buffer(2, Some(&d.gd), 0);
            enc.set_buffer(3, Some(&gs[0].1), 0);
            enc.set_buffer(4, Some(&gs[1].1), 0);
            enc.set_buffer(5, Some(&gs[2].1), 0);
            enc.set_buffer(6, Some(&gs[0].2), 0);
            enc.set_buffer(7, Some(&gs[1].2), 0);
            enc.set_buffer(8, Some(&gs[2].2), 0);
            enc.set_buffer(9, Some(&d.gin), 0);
            set_bytes_t(enc, 10, &go);
            setp(enc, &ip);
            grid(enc, (hidden as u64).div_ceil(8), nu, 3, 256);
            for (j, x) in inp.iter_mut().enumerate() {
                *x = (&d.gin, j as u64 * plane);
            }
        } else {
            for j in 0..3 {
                if !skip("gates") {
                    gate(c, p, d, enc, base, &gs[j], inp[j], (&d.msg, j as u64 * plane), (&d.gin, j as u64 * plane), n, hidden, vm);
                }
                inp[j] = (&d.gin, j as u64 * plane);
            }
        }
    }
    // 3. the input norms
    for j in 0..(if one { 1 } else { 3 }) {
        if normed && j == 0 {
            continue;
        }
        enc.set_compute_pipeline_state(&p.rms);
        enc.set_buffer(0, Some(inp[j].0), inp[j].1);
        enc.set_buffer(1, Some(&l.norms[j]), 0);
        enc.set_buffer(2, Some(&d.x[j]), 0);
        setp(enc, base);
        grid(enc, nu, 1, 1, 256);
    }
    let (xk, xv) = if one { (0, 0) } else { (1, 2) };
    // 4. the projections
    let narrow = |m: &Mat, x: &Buffer, y: &Buffer| match m {
        Mat::Q4 { abs, rows, cols } if cols % 64 == 0 => super::encode_q4tp_mm_n8(c, enc, &d.fbuf, *abs, x, y, &d.ones8, *rows, *cols, n),
        _ => mv(c, enc, &d.fbuf, m, x, 0, y, 0, n, true),
    };
    if !skip("proj") && vm && vproj_rows() && fuse && l.mvj.is_some() {
        // a verify's rows a decode launch each
        let (jobs, nj, tgs) = &l.mvj.as_ref().unwrap()[if one { 0 } else { 1 }];
        enc.set_compute_pipeline_state(&p.mvq);
        for i in 0..4u64 {
            enc.set_buffer(i, Some(d.fbuf.window((i as usize).min(d.fbuf.bufs.len() - 1))), 0);
        }
        enc.set_buffer(4, Some(jobs), 0);
        set_bytes_t(enc, 5, nj);
        set_bytes_t(enc, 6, &d.wstride);
        setp(enc, base);
        let xr = (hidden * 4) as u64;
        let ys = [l.q.rows(), l.gate.rows(), l.rel.rows(), l.k.rows(), l.v.rows()].map(|r| (r * 4) as u64);
        for j in 0..nu {
            enc.set_buffer(7, Some(&d.x[0]), j * xr);
            enc.set_buffer(8, Some(&d.x[1]), j * xr);
            enc.set_buffer(9, Some(&d.x[2]), j * xr);
            enc.set_buffer(10, Some(&d.yq), j * ys[0]);
            enc.set_buffer(11, Some(&d.yg), j * ys[1]);
            enc.set_buffer(12, Some(&d.yr), j * ys[2]);
            enc.set_buffer(13, Some(&d.yk), j * ys[3]);
            enc.set_buffer(14, Some(&d.yv), j * ys[4]);
            enc.dispatch_thread_groups(MTLSize::new(*tgs, 1, 1), MTLSize::new(128, 1, 1));
        }
    } else if !skip("proj") && vm {
        // a verify's rows: every matrix read once on the matrix unit
        narrow(&l.q, &d.x[0], &d.yq);
        narrow(&l.gate, &d.x[0], &d.yg);
        narrow(&l.rel, &d.x[0], &d.yr);
        narrow(&l.k, &d.x[xk], &d.yk);
        narrow(&l.v, &d.x[xv], &d.yv);
    } else if !skip("proj") && fuse && n == 1 && l.mvj.is_some() {
        // the five matrices in one launch
        let (jobs, nj, tgs) = &l.mvj.as_ref().unwrap()[if one { 0 } else { 1 }];
        enc.set_compute_pipeline_state(&p.mvq);
        for i in 0..4u64 {
            enc.set_buffer(i, Some(d.fbuf.window((i as usize).min(d.fbuf.bufs.len() - 1))), 0);
        }
        {
            enc.set_buffer(4, Some(jobs), 0);
            set_bytes_t(enc, 5, nj);
            set_bytes_t(enc, 6, &d.wstride);
            enc.set_buffer(7, Some(&d.x[0]), 0);
            enc.set_buffer(8, Some(&d.x[1]), 0);
            enc.set_buffer(9, Some(&d.x[2]), 0);
            enc.set_buffer(10, Some(&d.yq), 0);
            enc.set_buffer(11, Some(&d.yg), 0);
            enc.set_buffer(12, Some(&d.yr), 0);
            enc.set_buffer(13, Some(&d.yk), 0);
            enc.set_buffer(14, Some(&d.yv), 0);
            setp(enc, base);
        }
        enc.dispatch_thread_groups(MTLSize::new(*tgs, 1, 1), MTLSize::new(128, 1, 1));
    } else if !skip("proj") {
        mv(c, enc, &d.fbuf, &l.q, &d.x[0], 0, &d.yq, 0, n, vm);
        mv(c, enc, &d.fbuf, &l.gate, &d.x[0], 0, &d.yg, 0, n, vm);
        mv(c, enc, &d.fbuf, &l.rel, &d.x[0], 0, &d.yr, 0, n, vm);
        mv(c, enc, &d.fbuf, &l.k, &d.x[xk], 0, &d.yk, 0, n, vm);
        mv(c, enc, &d.fbuf, &l.v, &d.x[xv], 0, &d.yv, 0, n, vm);
    }
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
    let full = l.dk.is_some();
    if fuse {
        // the sinks and the per-token taps in one launch
        let kr = if full { (cfg.nkv * 4) as u32 } else { 0 };
        let rr: [u32; 4] = [cfg.nh as u32, kr, kr, 0];
        enc.set_compute_pipeline_state(&p.fmv3);
        enc.set_buffer(0, Some(&l.sink), 0);
        enc.set_buffer(1, Some(&d.x[0]), 0);
        enc.set_buffer(2, Some(&d.sk), 0);
        enc.set_buffer(3, Some(l.dk.as_ref().unwrap_or(&d.dummy)), 0);
        enc.set_buffer(4, Some(&d.x[xk]), 0);
        enc.set_buffer(5, Some(&d.dkb), 0);
        enc.set_buffer(6, Some(l.dv.as_ref().unwrap_or(&d.dummy)), 0);
        enc.set_buffer(7, Some(&d.x[xv]), 0);
        enc.set_buffer(8, Some(&d.dvb), 0);
        set_bytes_t(enc, 9, &rr);
        setp(enc, base);
        grid(enc, ((cfg.nh as u32 + 2 * kr) as u64).div_ceil(8), nu, 1, 256);
    } else {
        fmv(&l.sink, cfg.nh, &d.x[0], &d.sk);
        if let (Some(wk), Some(wv)) = (&l.dk, &l.dv) {
            fmv(wk, cfg.nkv * 4, &d.x[xk], &d.dkb);
            fmv(wv, cfg.nkv * 4, &d.x[xv], &d.dvb);
        }
    }
    // 5. queries and relative states; keys and values into the cache
    // (fused: one launch, a full layer's log scale and per-token taps alike)
    if fuse && l.sliding == l.dk.is_none() {
        let mut ip = *base;
        ip.full = full as u32;
        ip.ring = s.ring as u32;
        enc.set_compute_pipeline_state(&p.qrkv);
        enc.set_buffer(0, Some(&d.yq), 0);
        enc.set_buffer(1, Some(&d.yr), 0);
        enc.set_buffer(2, Some(&l.q_scale), 0);
        enc.set_buffer(3, Some(&l.rel_scale), 0);
        enc.set_buffer(4, Some(&d.yk), 0);
        enc.set_buffer(5, Some(&d.yv), 0);
        enc.set_buffer(6, Some(if full { &d.dkb } else { &d.dummy }), 0);
        enc.set_buffer(7, Some(if full { &d.dvb } else { &d.dummy }), 0);
        enc.set_buffer(8, Some(&l.tk), 0);
        enc.set_buffer(9, Some(&l.tv), 0);
        enc.set_buffer(10, Some(&s.rings[0]), 0);
        enc.set_buffer(11, Some(&s.rings[1]), 0);
        enc.set_buffer(12, Some(&s.k), 0);
        enc.set_buffer(13, Some(&s.v), 0);
        enc.set_buffer(16, Some(&s.hist), hist_off(0, kvs, hidden));
        enc.set_buffer(17, Some(&s.hist), hist_off(1, kvs, hidden));
        setp(enc, &ip);
        enc.dispatch_thread_groups(MTLSize::new((cfg.nh * n + 2 * cfg.nkv) as u64, 1, 1), MTLSize::new(cfg.hd as u64, 1, 1));
    } else {
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
    enc.set_buffer(16, Some(&s.hist), hist_off(0, kvs, hidden));
    enc.set_buffer(17, Some(&s.hist), hist_off(1, kvs, hidden));
    setp(enc, &ip);
    enc.dispatch_thread_groups(MTLSize::new(cfg.nkv as u64, 2, 1), MTLSize::new(cfg.hd as u64, 1, 1));
    }
    // 6. the attention, its gate, the O projection (fused: the gate in the attention's epilogue)
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
    // a prompt call: the flash kernel on the simdgroup matrices (the gate after it)
    let flash = n > 1 && !vm && flash_on() && cfg.hd == 128 && cfg.d_rel <= 32;
    let gated = fuse && !flash;
    enc.set_buffer(7, Some(if gated { &d.yg } else { &d.dummy }), 0);
    if gated {
        ip.v = 1;
    }
    setp(enc, &ip);
    if !skip("attn") {
        if flash {
            enc.set_compute_pipeline_state(&p.attn_fl);
            grid(enc, nu.div_ceil(32), cfg.nh as u64, 1, 128);
        } else {
            grid(enc, cfg.nh as u64, nu, 1, 256);
        }
    }
    let qs = cfg.nh * cfg.hd;
    if !gated {
        enc.set_compute_pipeline_state(&p.ogate);
        enc.set_buffer(0, Some(&d.att), 0);
        enc.set_buffer(1, Some(&d.yg), 0);
        setp(enc, base);
        grid(enc, (qs as u64).div_ceil(256), nu, 1, 256);
    }
    if !skip("o") {
        if vm && !vproj_rows() {
            narrow(&l.o, &d.att, &d.ya);
        } else {
            mv(c, enc, &d.fbuf, &l.o, &d.att, 0, &d.ya, 0, n, vm);
        }
    }
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
    enc.set_buffer(6, Some(&s.hist), hist_off(2, kvs, hidden));
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
    // fused, no gate: the MoE input's norm in the mix
    let mnorm = fuse && gates.is_none();
    if mnorm {
        ip.v = 1;
        enc.set_buffer(4, Some(&l.post_norm), 0);
        enc.set_buffer(5, Some(&d.xm), 0);
    }
    setp(enc, &ip);
    grid(enc, 1, nu, 1, 256);
    let mut mi: (&Buffer, u64) = (&d.m, 0);
    if let Some(gs) = gates {
        if !skip("gates") {
            gate(c, p, d, enc, base, &gs[3], mi, (&d.msg, 3 * plane), (&d.mg, 0), n, hidden, vm);
        }
        mi = (&d.mg, 0);
    }
    if !mnorm {
        enc.set_compute_pipeline_state(&p.rms);
        enc.set_buffer(0, Some(mi.0), mi.1);
        enc.set_buffer(1, Some(&l.post_norm), 0);
        enc.set_buffer(2, Some(&d.xm), 0);
        setp(enc, base);
        grid(enc, nu, 1, 1, 256);
    }
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
    if n > 1 && !vm && moe_mode() > 0 {
        // the entries by expert, an expert's rows against its weights once
        enc.set_compute_pipeline_state(&p.group);
        enc.set_buffer(0, Some(&d.pick), 0);
        enc.set_buffer(1, Some(&d.grp), 0);
        setp(enc, &ip);
        grid(enc, 1, 1, 1, 256);
    }
    if skip("moe") {
    } else if vm {
        // a verify: the decode kernels a (row, pick), a row's sums as decoding's
        // (the experts' unpacking is the M4's limit: reading a shared expert
        // once for its rows measured slower, on the lanes or the matrix unit)
        enc.set_compute_pipeline_state(&p.moe_gu8);
        for (i, w) in wins.iter().enumerate() {
            enc.set_buffer(i as u64, Some(w), 0);
        }
        enc.set_buffer(4, Some(&l.bases), 0);
        enc.set_buffer(5, Some(&d.pick), 0);
        enc.set_buffer(6, Some(&d.xm), 0);
        enc.set_buffer(7, Some(&d.act), 0);
        set_bytes_t(enc, 8, &d.wstride);
        setp(enc, &ip);
        grid(enc, (l.inter as u64) / 32, cfg.top_k as u64, nu, 128);
        enc.set_compute_pipeline_state(&p.moe_dn8);
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
    } else if n > 1 && moe_mode() == 2 {
        let nk = (n * cfg.top_k) as u64;
        let tiles = nu.div_ceil(32);
        let mm = |which: u32, x: &Buffer, y: &Buffer, out_rows: usize| {
            let mut ipm = ip;
            ipm.v = which;
            enc.set_compute_pipeline_state(&p.moe_mm);
            for (i, w) in wins.iter().enumerate() {
                enc.set_buffer(i as u64, Some(w), 0);
            }
            enc.set_buffer(4, Some(&l.bases), 0);
            enc.set_buffer(5, Some(&d.grp), 0);
            enc.set_buffer(6, Some(x), 0);
            enc.set_buffer(7, Some(y), 0);
            set_bytes_t(enc, 8, &d.wstride);
            setp(enc, &ipm);
            grid(enc, tiles, (out_rows as u64).div_ceil(64), l.e as u64, 128);
        };
        mm(0, &d.xm, &d.gq, l.inter);
        mm(1, &d.xm, &d.uq, l.inter);
        enc.set_compute_pipeline_state(&p.moe_act);
        enc.set_buffer(0, Some(&d.gq), 0);
        enc.set_buffer(1, Some(&d.uq), 0);
        enc.set_buffer(2, Some(&d.act), 0);
        setp(enc, &ip);
        grid(enc, (l.inter as u64).div_ceil(256), nk, 1, 256);
        mm(2, &d.act, &d.yp, hidden);
        enc.set_compute_pipeline_state(&p.moe_mixq);
        enc.set_buffer(0, Some(&d.yp), 0);
        enc.set_buffer(1, Some(&d.wt), 0);
        enc.set_buffer(2, Some(&d.y), 0);
        enc.set_buffer(3, Some(&d.grp), 0);
        setp(enc, &ip);
        grid(enc, (hidden as u64).div_ceil(256), nu, 1, 256);
    } else if n > 1 && moe_mode() == 1 {
        enc.set_compute_pipeline_state(&p.moe_gu_g);
        for (i, w) in wins.iter().enumerate() {
            enc.set_buffer(i as u64, Some(w), 0);
        }
        enc.set_buffer(4, Some(&l.bases), 0);
        enc.set_buffer(5, Some(&d.grp), 0);
        enc.set_buffer(6, Some(&d.xm), 0);
        enc.set_buffer(7, Some(&d.act), 0);
        set_bytes_t(enc, 8, &d.wstride);
        setp(enc, &ip);
        grid(enc, (l.inter as u64).div_ceil(16), l.e as u64, 1, 128);
        enc.set_compute_pipeline_state(&p.moe_down_g);
        for (i, w) in wins.iter().enumerate() {
            enc.set_buffer(i as u64, Some(w), 0);
        }
        enc.set_buffer(4, Some(&l.bases), 0);
        enc.set_buffer(5, Some(&d.grp), 0);
        enc.set_buffer(6, Some(&d.act), 0);
        enc.set_buffer(7, Some(&d.yp), 0);
        set_bytes_t(enc, 8, &d.wstride);
        setp(enc, &ip);
        grid(enc, (hidden as u64).div_ceil(16), l.e as u64, 1, 128);
        enc.set_compute_pipeline_state(&p.moe_mix);
        enc.set_buffer(0, Some(&d.yp), 0);
        enc.set_buffer(1, Some(&d.wt), 0);
        enc.set_buffer(2, Some(&d.y), 0);
        setp(enc, &ip);
        grid(enc, (hidden as u64).div_ceil(256), nu, 1, 256);
    } else if dec8_on() && l.inter % 32 == 0 && hidden % 16 == 0 {
        enc.set_compute_pipeline_state(&p.moe_gu8);
        for (i, w) in wins.iter().enumerate() {
            enc.set_buffer(i as u64, Some(w), 0);
        }
        enc.set_buffer(4, Some(&l.bases), 0);
        enc.set_buffer(5, Some(&d.pick), 0);
        enc.set_buffer(6, Some(&d.xm), 0);
        enc.set_buffer(7, Some(&d.act), 0);
        set_bytes_t(enc, 8, &d.wstride);
        setp(enc, &ip);
        grid(enc, (l.inter as u64) / 32, cfg.top_k as u64, nu, 128);
        enc.set_compute_pipeline_state(&p.moe_dn8);
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
    } else {
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
    }
    // 10. the output convolution onto the running sum
    let ip = *base;
    enc.set_compute_pipeline_state(&p.post);
    enc.set_buffer(0, Some(&d.y), 0);
    enc.set_buffer(1, Some(&l.tpm), 0);
    enc.set_buffer(2, Some(&s.rings[3]), 0);
    enc.set_buffer(3, Some(&d.r), 0);
    enc.set_buffer(4, Some(&d.dummy), 0);
    enc.set_buffer(5, Some(hout), 0);
    enc.set_buffer(6, Some(&s.hist), hist_off(3, kvs, hidden));
    setp(enc, &ip);
    grid(enc, (hidden as u64).div_ceil(256), 1, 1, 256);
}

/// A message gate on the encoder.
#[allow(clippy::too_many_arguments)]
fn gate(c: &Ctx, p: &Pipes, d: &Dev, enc: &ComputeCommandEncoderRef, base: &IP, g: &(Buffer, Buffer, Buffer, usize), x: (&Buffer, u64), m: (&Buffer, u64), out: (&Buffer, u64), n: usize, hidden: usize, vm: bool) {
    let dbuf = &d.gd;
    let mut ip = *base;
    ip.r = g.3 as u32;
    // a prompt call: the rows normed side by side, then three f32 GEMMs on the
    // simdgroup matrices (d = xn·downᵀ, F·d, G·d) and the mixture
    if n > 1 && !vm && gate_gemm_on() && g.3 % 4 == 0 {
        let r = g.3;
        enc.set_compute_pipeline_state(&p.gate_norm);
        enc.set_buffer(0, Some(x.0), x.1);
        enc.set_buffer(1, Some(m.0), m.1);
        enc.set_buffer(2, Some(&d.gxn), 0);
        setp(enc, &ip);
        grid(enc, n as u64, 1, 1, 256);
        let f32mm = |w: &Buffer, xs: &Buffer, y: &Buffer, rows: usize, cols: usize| {
            enc.set_compute_pipeline_state(&c.mmf32nt);
            enc.set_buffer(0, Some(w), 0);
            enc.set_buffer(1, Some(xs), 0);
            enc.set_buffer(2, Some(y), 0);
            let (cols_u, rows_u, nb_u, one) = (cols as u32, rows as u32, n as u32, 1.0f32);
            enc.set_bytes(3, 4, &cols_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &rows_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &nb_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &one as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(32), (rows as u64).div_ceil(64), 1), MTLSize::new(128, 1, 1));
        };
        f32mm(&g.0, &d.gxn, dbuf, r, 2 * hidden);
        f32mm(&g.1, dbuf, &d.gtf, hidden, r);
        f32mm(&g.2, dbuf, &d.gtg, hidden, r);
        enc.set_compute_pipeline_state(&p.gate_mix);
        enc.set_buffer(0, Some(x.0), x.1);
        enc.set_buffer(1, Some(m.0), m.1);
        enc.set_buffer(2, Some(&d.gtf), 0);
        enc.set_buffer(3, Some(&d.gtg), 0);
        enc.set_buffer(4, Some(out.0), out.1);
        setp(enc, &ip);
        grid(enc, ((n * hidden) as u64).div_ceil(256), 1, 1, 256);
        return;
    }
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
    grid(enc, (hidden as u64).div_ceil(8), n as u64, 1, 256);
}
