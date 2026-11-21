//! Inkling-2-Mini on CUDA, beside the hybrid arm whose matvecs and routed
//! experts it borrows (`inkling` has the semantics).
//!
//! Every matrix goes through the arm's tensor-core matvec in its codec (q4tp
//! or q8_2f); the routed experts are the arm's slot launches (`hy_moe_router`,
//! `hy_moe_select`, a gate/up pair whose epilogue is Inkling's softcapped
//! SwiGLU, the down launch, `hy_moe_combine`); the stages between them are
//! `inkling.cu`. A decode or verify call takes at most 16 rows (`NMAX`); its
//! launches are recorded once per row count and replayed (CUDA graphs). A
//! prompt goes up to `CMF_INKLING_ROWS` (512) rows a call on the prompt path:
//! every weight expanded to f16 and multiplied by cuBLAS, the routed experts
//! grouped by expert (the hybrid prompt path's `hy_moe_group`), so a chunk reads
//! each expert once instead of once every 16 rows. Every one of the 48 layer
//! applications keeps its own keys and values (f16; a local layer's in a ring of
//! its window and a call's rows, a full layer's for the context cap) and its four
//! convolution rings.

use super::*;
use crate::inkling::{Cfg, Globals, Layer};
use std::collections::HashMap;
use std::sync::Mutex;

const SRC: &str = include_str!("../inkling.cu");
/// Block states a row holds (the embedding and the closed blocks).
const MAXB: usize = 4;

/// Keys a tile of the decode attention takes (`AT` in `inkling.cu`).
const ATILE: usize = 256;

/// Bytes of a prompt chunk's attention scores (`attn_gemm`): the rows of a
/// chunk are as many as fit.
const ATT_SCORES: usize = 128 << 20;

/// `CMF_INKLING_MOE_MMA=0`: a prompt call's experts through the hybrid arm's
/// grouped tensor-core launches (32 rows a job, two bf16 planes) instead of
/// `ink_moe_mma` (64 rows a job, f16 rows, 128 output rows a block).
fn moe_mma_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_MOE_MMA").as_deref() != Ok("0"))
}

/// Output tiles of 16 rows an `ink_moe_mma` block takes (4 warps × `MT`).
const MMA_TILES: usize = 8;

/// `CMF_INKLING_ATTN_GEMM=0`: a prompt call's attention a block a head and row
/// (`ink_attn`) instead of the cuBLAS products.
fn attn_gemm_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_ATTN_GEMM").as_deref() != Ok("0"))
}

/// `CMF_INKLING_SPLIT=0`: a decode call's attention a block a head and row
/// (`ink_attn`) instead of a block a tile of keys.
fn split_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_SPLIT").as_deref() != Ok("0"))
}

/// Rows of a prompt call (`CMF_INKLING_ROWS`, 512; 16 reads a prompt on the
/// decode path, 16 rows a call).
fn bmax() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("CMF_INKLING_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).map_or(512, |n| n.clamp(NMAX, 4096)))
}

struct Fx {
    mix: cs::CUfunction,
    rms: cs::CUfunction,
    gate_down: cs::CUfunction,
    gate_out: cs::CUfunction,
    fmv: cs::CUfunction,
    qr: cs::CUfunction,
    kv: cs::CUfunction,
    attn: cs::CUfunction,
    ogate: cs::CUfunction,
    post: cs::CUfunction,
    softcap: cs::CUfunction,
    attn_split: cs::CUfunction,
    attn_merge: cs::CUfunction,
    h16: cs::CUfunction,
    f16: cs::CUfunction,
    attn_soft: cs::CUfunction,
    gather16: cs::CUfunction,
    moe_mma: cs::CUfunction,
    ring_snap: cs::CUfunction,
    ring_back: cs::CUfunction,
    ds_mix: cs::CUfunction,
    ds_ctx: cs::CUfunction,
    ds_qk: cs::CUfunction,
    ds_attn: cs::CUfunction,
    planes: cs::CUfunction,
    ds_markov: cs::CUfunction,
    ds_pick: cs::CUfunction,
    argmax_rows: cs::CUfunction,
    gate_norm: cs::CUfunction,
    gate_mix: cs::CUfunction,
    kv_rows: cs::CUfunction,
    ring_rows: cs::CUfunction,
    post_rows: cs::CUfunction,
}

/// A loop layer's message gate: down `[R][2H]`, forget and gain `[H][R]`.
struct GateD {
    down: Buf,
    forget: Buf,
    gain: Buf,
}

struct LayerD {
    sliding: bool,
    boundary: bool,
    /// The attention input's queries (1 or 3); None on layer 0.
    aq: Option<(Buf, usize)>,
    mq: Buf,
    norms: [Buf; 3],
    post_norm: Buf,
    /// q, gate, rel from the query input's planes; k; v; o.
    qset: MmaSet,
    /// q, gate, rel, k and v together, for a local layer's first pass (one input row).
    qkv: Option<MmaSet>,
    kset: MmaSet,
    vset: MmaSet,
    oset: MmaSet,
    sink: Buf,
    q_scale: Buf,
    rel_scale: Buf,
    rel_table: Buf,
    extent: usize,
    tk: Buf,
    tv: Buf,
    dk: Option<Buf>,
    dv: Option<Buf>,
    tpa: Buf,
    tpm: Buf,
    router: Buf,
    bias: Buf,
    e: usize,
    inter: usize,
    gu_t: Buf,
    dn_t: Buf,
    kd: (Option<usize>, Option<usize>),
    bpj_gu: u32,
    bpj_dn: u32,
    gates: Option<[GateD; 4]>,
    /// q, gate, rel, k, v, o for the prompt path's GEMMs.
    dense: [Arc<DevMat>; 6],
    _mats: Vec<Arc<DevMat>>,
}

/// One application's caches: K and V (f16 `[row][NKV·HD]`, `ring` > 0 a ring
/// of that many rows), the k, v, post-attention and post-MoE rings.
struct SlotD {
    k: Buf,
    v: Buf,
    ring: usize,
    rings: [Buf; 4],
    /// A decode call's input rows of each ring (`NMAX` rows) and the rings'
    /// snapshot: a draft verify rolls them back to its accepted prefix.
    hist: [Buf; 4],
    snap: [Buf; 4],
}

/// `struct RingRef` of inkling.cu.
#[repr(C)]
#[derive(Clone, Copy)]
struct RingRef {
    live: u64,
    snap: u64,
    hist: u64,
    width: i32,
    pad: i32,
}

/// One dense layer of the DSpark drafter on the card.
struct DLayerD {
    qkv: MmaSet,
    kctx: MmaSet,
    vctx: MmaSet,
    kctx_m: Arc<DevMat>,
    vctx_m: Arc<DevMat>,
    o: MmaSet,
    gu: MmaSet,
    down: MmaSet,
    in_norm: Buf,
    post_norm: Buf,
    q_norm: Buf,
    k_norm: Buf,
    kc_norm: Buf,
    /// The layer's keys and values (f16 `[cell][NKV·HD]`; a sliding layer's a
    /// ring of its window and more).
    ck: Buf,
    cv: Buf,
    cap: usize,
    ring: bool,
    _mats: Vec<Arc<DevMat>>,
}

/// The DSpark drafter on the card: its layers and caches, the taps the target
/// leaves in every call, the block's rows and the Markov head.
struct DraftD {
    layers: Vec<DLayerD>,
    /// Application indices whose input rows are tapped (the last target, the
    /// final normed state, is every call's).
    taps: Vec<usize>,
    /// `[targets][rows][H]` f32.
    tap: Buf,
    alpha: Buf,
    /// The mixes' norms `[mixes][H]`.
    norms: Buf,
    /// The block's final norm.
    fnorm: Buf,
    mixp: [Buf; 2],
    mixh: Buf,
    ckb: Buf,
    cvb: Buf,
    /// The Markov head's context of recent positions `[FR][H]`.
    fr: Buf,
    masks: Buf,
    x: Buf,
    xp: [Buf; 2],
    dq: Buf,
    dk: Buf,
    dv: Buf,
    att: Buf,
    ap: [Buf; 2],
    gp: [Buf; 2],
    hf: Buf,
    hp: [Buf; 2],
    dcr: Buf,
    head: MmaSet,
    dlog: Buf,
    rnn_w: Buf,
    rnn_b: Buf,
    ch: Buf,
    cf: Buf,
    state: Buf,
    prev: Buf,
    ids: Buf,
    w1: Buf,
    folds: Vec<MmaSet>,
    fold_b: Buf,
    latp: [Buf; 2],
    block: usize,
    rank: usize,
    window: usize,
    theta: f32,
    eps: f32,
    inter: usize,
    _mats: Vec<Arc<DevMat>>,
}

/// Positions whose Markov context the ring keeps.
const FR: usize = 16;

/// The prompt path's scratch: the f16 inputs of the weight GEMMs, the band a
/// weight expands into, the grouped experts' entries, planes and jobs.
struct Big {
    x16: [Buf; 3],
    o16: Buf,
    wd: Buf,
    wd_elems: usize,
    lg: Buf,
    pidx: Buf,
    pwt: Buf,
    pos: Buf,
    src: Buf,
    xg: [Buf; 2],
    qq: [Buf; 2],
    ys: Buf,
    gu_a: Buf,
    dn_a: Buf,
    /// A message gate's normed halves `[rows][2H]` and its F and G products
    /// `[rows][H]` (`gate_big`).
    gxn: Buf,
    gtf: Buf,
    gtg: Buf,
    /// The attention through cuBLAS (`attn_gemm`): the queries as f16, the
    /// scores of a chunk of rows `[NH][rows][keys]` f32 and their softmax f16.
    q16: Buf,
    sc: Buf,
    pr: Buf,
}

struct Dev {
    fx: Fx,
    hfx: Arc<HFuncs>,
    /// Rows the row buffers hold: a prompt call's cap (`NMAX` without the
    /// prompt path).
    rows: usize,
    big: Option<Big>,
    /// A prompt call's first position (the host's copy of `cr[0]`).
    big_pos0: std::cell::Cell<usize>,
    /// Every application's four rings as `RingRef`s (`ink_ring_snap` / `back`).
    rings_tab: Buf,
    nrings: usize,
    draft: Option<DraftD>,
    layers: Vec<LayerD>,
    slots: Vec<SlotD>,
    head: MmaSet,
    msg_q: Buf,
    final_q: Buf,
    final_norm: Buf,
    /// The running sums (two, alternating), the block states, the prelude's
    /// output, the previous pass's blocks and output.
    h: [Buf; 2],
    blk: Buf,
    pre_h: Buf,
    bank: Buf,
    bank_out: Buf,
    msg: Buf,
    in3: Buf,
    gin: Buf,
    gd: Buf,
    x: [Buf; 3],
    xp: [[Buf; 2]; 3],
    yq: Buf,
    yg: Buf,
    yr: Buf,
    yk: Buf,
    yv: Buf,
    sk: Buf,
    /// The key tiles' softmax partials of a decode call's attention
    /// (`ink_attn_split`): `[NMAX][NH][tiles of the cap][HD + 2]`.
    part: Buf,
    dkb: Buf,
    dvb: Buf,
    att: Buf,
    po: [Buf; 2],
    ya: Buf,
    r: Buf,
    m: Buf,
    mg: Buf,
    xm: Buf,
    pm: [Buf; 2],
    logits_e: Buf,
    sg: Buf,
    wsel: Buf,
    gu_a: Buf,
    dn_a: Buf,
    qq: [Buf; 2],
    dns: Buf,
    y: Buf,
    fin: Buf,
    pf: [Buf; 2],
    logits: Buf,
    cr: Buf,
    /// A verify call's greedy pick of each row (`NMAX` ids).
    pick: Buf,
    smax: usize,
    /// Recorded calls by (rows, logits rows) and the pointers they hold.
    graphs: HashMap<(usize, usize), GraphRec>,
    /// Positions the full slots hold.
    cap: usize,
}
unsafe impl Send for Dev {}

static DEVICES: Mutex<Option<HashMap<u64, Dev>>> = Mutex::new(None);

/// Stages of `CMF_INKLING_PROF=1` (each closed by a stream sync; the calls
/// then run without graphs).
const STAGES: [&str; 14] = ["mix", "msg+gates", "norms", "proj", "qr+kv", "attn", "o", "post", "moe in", "moe", "head", "moe route", "moe gu", "moe down"];
static STAGE_NS: [std::sync::atomic::AtomicU64; 14] = [const { std::sync::atomic::AtomicU64::new(0) }; 14];
static PROF_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// A prompt call's stages, apart from the decode calls' (`CMF_INKLING_PROF=1`).
static BIG_NS: [std::sync::atomic::AtomicU64; 14] = [const { std::sync::atomic::AtomicU64::new(0) }; 14];
static IN_BIG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn prof() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_INKLING_PROF").as_deref() == Ok("1"))
}

/// Closes stage `i` when profiling: syncs and adds its time.
fn st(c: &Ctx, i: usize, t: &mut std::time::Instant) -> Res<()> {
    if prof() {
        c.sync()?;
        let to = if IN_BIG.load(Ordering::Relaxed) { &BIG_NS } else { &STAGE_NS };
        to[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        *t = std::time::Instant::now();
    }
    Ok(())
}

/// The stage table of the decode calls so far (`CMF_INKLING_PROF=1`).
fn prof_report() {
    let calls = PROF_CALLS.load(Ordering::Relaxed).max(1);
    let row: Vec<String> = STAGES.iter().zip(&STAGE_NS).map(|(n, v)| format!("{n} {:.2}", v.load(Ordering::Relaxed) as f64 / 1e6 / calls as f64)).collect();
    tracing::info!("inkling2_loop CUDA ms a call ({calls} calls): {}", row.join(", "));
}

/// One prompt call's stage table (`CMF_INKLING_PROF=1`), then cleared.
fn prof_report_big(n: usize) {
    let tot: u64 = BIG_NS.iter().map(|v| v.load(Ordering::Relaxed)).sum();
    let row: Vec<String> = STAGES.iter().zip(&BIG_NS).filter(|(_, v)| v.load(Ordering::Relaxed) > 0).map(|(name, v)| format!("{name} {:.1}", v.swap(0, Ordering::Relaxed) as f64 / 1e6)).collect();
    tracing::info!("inkling2_loop CUDA prompt call of {n} rows, {:.1} ms ({:.0} rows/s): {}", tot as f64 / 1e6, n as f64 / (tot as f64 / 1e9), row.join(", "));
}
static DECLINED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn gw_of(t: &crate::qtensor::QTensor) -> Option<(Arc<CmfModel>, GraphW<'static>)> {
    let (model, idx, kind, _) = t.graph_weight()?;
    // q4tp (6) and q8_2f (7): the codecs the arm's matvecs read from the file
    if !matches!(kind, 6 | 7) {
        return None;
    }
    Some((model.clone(), GraphW { idx, kind, row_scale: &[], data: &[], prism: crate::gpu::GraphPrismOp::None, affine: false }))
}

fn mat(c: &Ctx, st: &mut State, t: &crate::qtensor::QTensor) -> Res<Arc<DevMat>> {
    let (model, gw) = gw_of(t).ok_or("an Inkling matrix outside q4tp / q8_2f")?;
    hmat(c, st, &model, &gw, t.rows(), t.cols())
}

/// The stack fits the arm: its geometry and every matrix's codec.
pub(crate) fn admits(g: &Globals, layers: &[Layer], cfg: &Cfg) -> bool {
    let ok_geom = cfg.hd == 128 && cfg.nh % cfg.nkv == 0 && cfg.dim % 256 == 0 && cfg.d_rel <= 32 && cfg.window <= 1024 && cfg.rel_extent <= 1024 && cfg.block > 0 && cfg.layers / cfg.block <= MAXB;
    let q = |t: &crate::qtensor::QTensor| gw_of(t).is_some();
    ok_geom
        && q(&g.lm_head)
        && layers.iter().all(|l| {
            let a = &l.attn;
            q(&a.q) && q(&a.k) && q(&a.v) && q(&a.o) && q(&a.gate) && q(&a.rel) && l.moe.experts.iter().all(|[x, y, z]| q(x) && q(y) && q(z)) && a.k_conv.kernel == 4 && a.v_conv.kernel == 4
        })
}

fn f32b(v: &[f32]) -> Res<Buf> {
    Buf::from_slice(v)
}

fn job(m: &DevMat, rows: usize, kind: i32) -> HyJob {
    HyJob { w: m.w.ptr, aux0: m.aux0(), aux1: m.aux1(), y: 0, res: 0, rows: rows as i32, kind, blk0: 0, ystride: 0, xoff: 0, nr: 0, lim: 0.0, pad: [0; 3] }
}

fn build(c: &Ctx, st: &mut State, g: &Globals, layers: &[Layer], cfg: &Cfg) -> Res<Dev> {
    let t0 = std::time::Instant::now();
    // the 6144 expert matrices (1.3 and 2.7 MB) as pieces of shared slabs: the
    // driver rounds each allocation above a megabyte up to a multiple of two,
    // which took 6 GB of an L4 beside 13.6 GB of weights
    let slabs = super::super::SlabScope::enter();
    let (hidden, nh, nkv, hd, dr) = (cfg.dim, cfg.nh, cfg.nkv, cfg.hd, cfg.d_rel);
    let hpk = nh / nkv;
    let hfx = match st.hyb.funcs.get(&(hpk, hd)) {
        Some(f) => f.clone(),
        None => {
            let f = Arc::new(load_funcs(c, hpk, hd)?);
            st.hyb.funcs.insert((hpk, hd), f.clone());
            f
        }
    };
    let module = compile(SRC, vec![format!("-DH={hidden}"), format!("-DHD={hd}"), format!("-DNKV={nkv}"), format!("-DNH={nh}"), format!("-DDR={dr}"), "-DKC=4".into()], c.arch)?;
    let fx = Fx {
        mix: func(module, "ink_mix")?,
        rms: func(module, "ink_rms")?,
        gate_down: func(module, "ink_gate_down")?,
        gate_out: func(module, "ink_gate_out")?,
        fmv: func(module, "ink_fmv")?,
        qr: func(module, "ink_qr")?,
        kv: func(module, "ink_kv")?,
        attn: func(module, "ink_attn")?,
        ogate: func(module, "ink_ogate")?,
        post: func(module, "ink_post")?,
        softcap: func(module, "ink_softcap")?,
        attn_split: func(module, "ink_attn_split")?,
        attn_merge: func(module, "ink_attn_merge")?,
        h16: func(module, "ink_h16")?,
        f16: func(module, "ink_f16")?,
        gather16: func(module, "ink_gather16")?,
        moe_mma: func(module, "ink_moe_mma")?,
        ring_snap: func(module, "ink_ring_snap")?,
        ring_back: func(module, "ink_ring_back")?,
        ds_mix: func(module, "ink_ds_mix")?,
        ds_ctx: func(module, "ink_ds_ctx")?,
        ds_qk: func(module, "ink_ds_qk")?,
        ds_attn: func(module, "ink_ds_attn")?,
        planes: func(module, "ink_planes")?,
        ds_markov: func(module, "ink_ds_markov")?,
        ds_pick: func(module, "ink_ds_pick")?,
        argmax_rows: func(module, "ink_argmax_rows")?,
        attn_soft: func(module, "ink_attn_soft")?,
        gate_norm: func(module, "ink_gate_norm")?,
        gate_mix: func(module, "ink_gate_mix")?,
        kv_rows: func(module, "ink_kv_rows")?,
        ring_rows: func(module, "ink_ring_rows")?,
        post_rows: func(module, "ink_post_rows")?,
    };
    let f4 = |n: usize| Buf::new(n * 4);
    let h2 = |n: usize| Buf::new(n * 2 + 64);
    let (qs, kvs, rs) = (nh * hd, nkv * hd, nh * dr);
    // the prompt path's rows (f16 weight GEMMs, as the hybrid's)
    let big_on = bmax() > NMAX && prec().0 == Gemm::Half;
    let rows = if big_on { bmax() } else { NMAX };
    let (yq, yg, yr, yk, yv) = (f4(rows * qs)?, f4(rows * qs)?, f4(rows * rs)?, f4(rows * kvs)?, f4(rows * kvs)?);
    let ya = f4(rows * hidden)?;
    let vocab = g.lm_head.rows();
    let logits = f4(NMAX * vocab)?;
    let e = layers.first().map_or(0, |l| l.moe.bias.len());
    let k = cfg.top_k;
    let inter = layers.first().and_then(|l| l.moe.experts.first()).map_or(0, |x| x[0].rows());
    let smax = (NMAX * k).min(e);
    let mut dl = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter().enumerate() {
        let a = &l.attn;
        let (q, gt, rl, km, vm, om) = (mat(c, st, &a.q)?, mat(c, st, &a.gate)?, mat(c, st, &a.rel)?, mat(c, st, &a.k)?, mat(c, st, &a.v)?, mat(c, st, &a.o)?);
        let qset = mmaset(&[(&q, yq.ptr, 0, qs, qs), (&gt, yg.ptr, 0, qs, qs), (&rl, yr.ptr, 0, rs, rs)])?;
        let qkv = if l.sliding { Some(mmaset(&[(&q, yq.ptr, 0, qs, qs), (&gt, yg.ptr, 0, qs, qs), (&rl, yr.ptr, 0, rs, rs), (&km, yk.ptr, 0, kvs, kvs), (&vm, yv.ptr, 0, kvs, kvs)])?) } else { None };
        let kset = mmaset(&[(&km, yk.ptr, 0, kvs, kvs)])?;
        let vset = mmaset(&[(&vm, yv.ptr, 0, kvs, kvs)])?;
        let oset = mmaset(&[(&om, ya.ptr, 0, hidden, hidden)])?;
        let dense = [q.clone(), gt.clone(), rl.clone(), km.clone(), vm.clone(), om.clone()];
        let mut mats = vec![q, gt, rl, km, vm, om];
        // the routed experts: Inkling's softcapped SwiGLU in the pair's epilogue
        // (kind bit 0x2000: the gate's cap in `lim`, the up branch's in pad[0])
        let mut gu_t = Vec::with_capacity(e);
        let mut dn_t = Vec::with_capacity(e);
        EXPERT_MAT.with(|x| x.set(true));
        let built = (|| -> Res<()> {
            for [eg, eu, ed] in &l.moe.experts {
                let (mg, gwg) = gw_of(eg).ok_or("expert gate codec")?;
                let (_, gwu) = gw_of(eu).ok_or("expert up codec")?;
                let pm = hmat_pair(c, st, &mg, &gwg, &gwu, inter, hidden)?;
                let dmx = mat(c, st, ed)?;
                gu_t.push(HyJob { lim: cfg.gate_cap, pad: [cfg.up_cap.to_bits() as i32, 0, 0], ..job(&pm, 2 * inter, pm.kind as i32 | 0x100 | 0x2000) });
                dn_t.push(job(&dmx, hidden, dmx.kind as i32));
                mats.push(pm);
                mats.push(dmx);
            }
            Ok(())
        })();
        EXPERT_MAT.with(|x| x.set(false));
        built?;
        let one_kd = |t: &[HyJob]| {
            let d = t.first().and_then(|j| kd_of(j.kind & 0xFF))?;
            t.iter().all(|j| kd_of(j.kind & 0xFF) == Some(d)).then_some(d)
        };
        let kd = (one_kd(&gu_t), one_kd(&dn_t));
        let gates = match &l.gates {
            Some(gs) => {
                let mk = |x: &crate::inkling::Gate| -> Res<GateD> { Ok(GateD { down: f32b(&x.down)?, forget: f32b(&x.forget)?, gain: f32b(&x.gain)? }) };
                Some([mk(&gs[0])?, mk(&gs[1])?, mk(&gs[2])?, mk(&gs[3])?])
            }
            None => None,
        };
        dl.push(LayerD {
            sliding: l.sliding,
            boundary: li % cfg.block == 0,
            aq: match &l.attn_query {
                Some(q) => Some((f32b(q)?, q.len() / hidden)),
                None => None,
            },
            mq: f32b(&l.mlp_query)?,
            norms: [f32b(&l.in_norm[0])?, f32b(&l.in_norm[1])?, f32b(&l.in_norm[2])?],
            post_norm: f32b(&l.post_norm)?,
            qset,
            qkv,
            kset,
            vset,
            oset,
            sink: f32b(&a.sink)?,
            q_scale: f32b(&a.q_scale)?,
            rel_scale: f32b(&a.rel_scale)?,
            rel_table: f32b(&a.rel_table)?,
            extent: a.rel_table.len() / dr,
            tk: f32b(&a.k_conv.taps)?,
            tv: f32b(&a.v_conv.taps)?,
            dk: match &a.k_conv.proj {
                Some(p) => Some(f32b(p)?),
                None => None,
            },
            dv: match &a.v_conv.proj {
                Some(p) => Some(f32b(p)?),
                None => None,
            },
            tpa: f32b(&l.post_attn_conv.taps)?,
            tpm: f32b(&l.post_mlp_conv.taps)?,
            router: f32b(&l.moe.router)?,
            bias: f32b(&l.moe.bias)?,
            e,
            inter,
            gu_t: Buf::from_slice(&gu_t)?,
            dn_t: Buf::from_slice(&dn_t)?,
            kd,
            bpj_gu: (2 * inter).div_ceil(16) as u32,
            bpj_dn: hidden.div_ceil(16) as u32,
            gates,
            dense,
            _mats: mats,
        });
    }
    let hm = mat(c, st, &g.lm_head)?;
    let head = mmaset(&[(&hm, logits.ptr, 0, vocab, vocab)])?;
    let resident = c.resident.load(Ordering::Relaxed);
    // the caches: a local slot's window, a full slot's context cap
    let cap = cfg.max_seq;
    let mut slots = Vec::with_capacity(cfg.slots());
    for (li, pass) in cfg.plan() {
        let _ = pass;
        // a local slot's ring: its window and a call's rows (the call writes
        // every row's key before any row reads)
        let ring = if layers[li].sliding { cfg.window + rows } else { 0 };
        let rows_kv = if ring > 0 { ring } else { cap };
        let s = SlotD {
            k: h2(rows_kv * kvs)?,
            v: h2(rows_kv * kvs)?,
            ring,
            rings: [f4(3 * kvs)?, f4(3 * kvs)?, f4(3 * hidden)?, f4(3 * hidden)?],
            hist: [f4(NMAX * kvs)?, f4(NMAX * kvs)?, f4(NMAX * hidden)?, f4(NMAX * hidden)?],
            snap: [f4(3 * kvs)?, f4(3 * kvs)?, f4(3 * hidden)?, f4(3 * hidden)?],
        };
        c.kv_bytes.fetch_add((s.k.bytes + s.v.bytes) as u64, Ordering::Relaxed);
        slots.push(s);
    }
    // slots by application order → by slot index
    let mut by_slot: Vec<Option<SlotD>> = (0..cfg.slots()).map(|_| None).collect();
    for ((li, pass), s) in cfg.plan().into_iter().zip(slots) {
        by_slot[cfg.slot(li, pass)] = Some(s);
    }
    let slots: Vec<SlotD> = by_slot.into_iter().map(|s| s.ok_or_else(|| "a slot without its application".to_string())).collect::<Res<_>>()?;
    let job_b = std::mem::size_of::<HyJob>();
    let mut tab = Vec::with_capacity(slots.len() * 4);
    for sl in &slots {
        for j in 0..4 {
            tab.push(RingRef { live: sl.rings[j].ptr, snap: sl.snap[j].ptr, hist: sl.hist[j].ptr, width: if j < 2 { kvs } else { hidden } as i32, pad: 0 });
        }
    }
    let rings_tab = Buf::from_slice(&tab)?;
    let nrings = tab.len();
    let draft = match &g.draft {
        Some(dr) => match build_draft(c, st, g, dr, cfg, rows, &hm, cap) {
            Ok(x) => Some(x),
            Err(e) => {
                tracing::warn!("inkling2_loop on CUDA: the DSpark drafter does not build ({e}); plain decoding");
                None
            }
        },
        None => None,
    };
    // the prompt path: every dense weight in a codec it expands (q4tp, int8)
    let big = if big_on && dl.iter().all(|l| l.dense.iter().all(|m| matches!(m.kind, 5 | 6 | 15))) && inter % 32 == 0 && e <= 1024 {
        let wmax = dl.iter().flat_map(|l| l.dense.iter().map(|m| m.rows * m.cols)).max().unwrap_or(0);
        let wd_bytes = (wmax * 2).min(wd_cap().max(1 << 20));
        let nk = rows * k;
        let jmax = e + nk.div_ceil(32);
        Some(Big {
            x16: [h2(rows * hidden)?, h2(rows * hidden)?, h2(rows * hidden)?],
            o16: h2(rows * qs)?,
            wd: Buf::new(wd_bytes + 64)?,
            wd_elems: wd_bytes / 2,
            lg: f4(rows * e)?,
            pidx: f4(nk)?,
            pwt: f4(nk)?,
            pos: f4(nk)?,
            src: f4(nk)?,
            xg: [h2(nk * hidden)?, h2(nk * hidden)?],
            qq: [h2(nk * inter)?, h2(nk * inter)?],
            ys: f4(nk * hidden)?,
            gu_a: Buf::new(jmax * job_b)?,
            dn_a: Buf::new(jmax * job_b)?,
            gxn: f4(rows * 2 * hidden)?,
            gtf: f4(rows * hidden)?,
            gtg: f4(rows * hidden)?,
            q16: h2(rows * qs)?,
            sc: Buf::new(ATT_SCORES)?,
            pr: Buf::new(ATT_SCORES / 2)?,
        })
    } else {
        None
    };
    let d = Dev {
        fx,
        hfx,
        rows,
        big,
        big_pos0: std::cell::Cell::new(0),
        rings_tab,
        nrings,
        draft,
        layers: dl,
        slots,
        head,
        msg_q: f32b(&g.message_query)?,
        final_q: f32b(&g.final_query)?,
        final_norm: f32b(&g.norm)?,
        h: [f4(rows * hidden)?, f4(rows * hidden)?],
        blk: f4(rows * MAXB * hidden)?,
        pre_h: f4(rows * hidden)?,
        bank: f4(rows * MAXB * hidden)?,
        bank_out: f4(rows * hidden)?,
        msg: f4(4 * rows * hidden)?,
        in3: f4(3 * rows * hidden)?,
        gin: f4(4 * rows * hidden)?,
        gd: f4(rows * 256)?,
        x: [f4(rows * hidden)?, f4(rows * hidden)?, f4(rows * hidden)?],
        xp: [[h2(rows * hidden)?, h2(rows * hidden)?], [h2(rows * hidden)?, h2(rows * hidden)?], [h2(rows * hidden)?, h2(rows * hidden)?]],
        yq,
        yg,
        yr,
        yk,
        yv,
        sk: f4(rows * nh)?,
        part: f4(NMAX * nh * cap.div_ceil(ATILE) * (hd + 2))?,
        dkb: f4(rows * nkv * 4)?,
        dvb: f4(rows * nkv * 4)?,
        att: f4(rows * qs)?,
        po: [h2(rows * qs)?, h2(rows * qs)?],
        ya,
        r: f4(rows * hidden)?,
        m: f4(rows * hidden)?,
        mg: f4(rows * hidden)?,
        xm: f4(rows * hidden)?,
        pm: [h2(rows * hidden)?, h2(rows * hidden)?],
        logits_e: f4(rows * e)?,
        sg: f4(rows)?,
        wsel: f4(NMAX * smax)?,
        gu_a: Buf::new(smax * job_b)?,
        dn_a: Buf::new(smax * job_b)?,
        qq: [Buf::new(smax * NMAX * hidden * 2)?, Buf::new(smax * NMAX * hidden * 2)?],
        dns: f4(smax * NMAX * hidden)?,
        y: f4(rows * hidden)?,
        fin: f4(rows * hidden)?,
        pf: [h2(rows * hidden)?, h2(rows * hidden)?],
        logits,
        cr: Buf::new(64)?,
        pick: Buf::new(NMAX * 4)?,
        smax,
        graphs: HashMap::new(),
        cap,
    };
    tracing::info!(
        "inkling2_loop on CUDA: {} layers resident, {:.1} GB of weights, {:.2} GB of caches for {} positions, prompt calls of {} rows, built in {:.1} s; {}",
        layers.len(),
        resident as f64 / 1e9,
        d.slots.iter().map(|s| (s.k.bytes + s.v.bytes) as f64).sum::<f64>() / 1e9,
        cap,
        if d.big.is_some() { d.rows } else { NMAX },
        t0.elapsed().as_secs_f64(),
        slabs.report()
    );
    Ok(d)
}

impl Dev {
    fn zero_state(&self, c: &Ctx) -> Res<()> {
        unsafe {
            for s in &self.slots {
                for r in &s.rings {
                    cu::memset_d8_async(r.ptr, 0, r.bytes, c.stream).map_err(derr)?;
                }
            }
        }
        c.sync()
    }
}

/// `true` when this call ran on the card. `ids` from position `pos0`; the
/// logits of the last row (`all`: of every row, `[n][vocab]`) when `want`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(id: u64, g: &Globals, layers: &[Layer], cfg: &Cfg, ids: &[u32], pos0: usize, want: bool, all: bool, logits: &mut Vec<f32>) -> bool {
    forward_with(id, g, layers, cfg, ids, pos0, want, all, logits, None)
}

/// A draft verify: every row of `ids` through the target, each row's greedy
/// pick into `picks` on the card; the logits stay there for `row_logits`.
pub(crate) fn verify(id: u64, g: &Globals, layers: &[Layer], cfg: &Cfg, ids: &[u32], pos0: usize, picks: &mut Vec<u32>) -> bool {
    if ids.len() > NMAX {
        return false;
    }
    let mut none = Vec::new();
    forward_with(id, g, layers, cfg, ids, pos0, true, true, &mut none, Some(picks))
}

/// Row `r`'s logits of the last verify.
pub(crate) fn row_logits(id: u64, r: usize, vocab: usize, out: &mut Vec<f32>) -> bool {
    let Some(c) = super::super::ctx() else { return false };
    let guard = DEVICES.lock().unwrap();
    let Some(d) = guard.as_ref().and_then(|m| m.get(&id)) else { return false };
    if r >= NMAX {
        return false;
    }
    out.resize(vocab, 0.0);
    unsafe { cu::memcpy_dtoh_async(out, d.logits.ptr + (r * vocab * 4) as u64, c.stream).is_ok() && c.sync().is_ok() }
}

#[allow(clippy::too_many_arguments)]
fn forward_with(id: u64, g: &Globals, layers: &[Layer], cfg: &Cfg, ids: &[u32], pos0: usize, want: bool, all: bool, logits: &mut Vec<f32>, pick: Option<&mut Vec<u32>>) -> bool {
    let Some(c) = super::super::ctx() else { return false };
    if std::env::var("CMF_INKLING_CUDA").as_deref() == Ok("0") {
        return false;
    }
    if !admits(g, layers, cfg) {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| tracing::warn!("inkling2_loop on CUDA: the file's codecs or geometry are not the arm's (q4tp / q8_2f matrices, 128-wide heads); the host runs it"));
        return false;
    }
    if DECLINED.lock().unwrap().contains(&id) {
        return false;
    }
    if c.current().is_err() {
        return false;
    }
    let mut guard = DEVICES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if !map.contains_key(&id) {
        // one sequence's state on the card at a time: a new sequence takes the
        // device over (its weights, caches and recorded calls stay)
        let old = map.keys().next().copied();
        match old {
            Some(k) => {
                let d = map.remove(&k).expect("device");
                map.insert(id, d);
            }
            None => {
                let built = {
                    let mut st = c.st.lock().unwrap();
                    build(c, &mut st, g, layers, cfg)
                };
                match built {
                    Ok(d) => {
                        map.insert(id, d);
                    }
                    Err(e) => {
                        tracing::warn!("inkling2_loop on CUDA declined: {e}; the host runs this sequence");
                        DECLINED.lock().unwrap().push(id);
                        return false;
                    }
                }
            }
        }
    }
    let d = map.get_mut(&id).expect("device");
    if pos0 == 0 && d.zero_state(c).is_err() {
        return false;
    }
    if pos0 + ids.len() > d.cap {
        tracing::warn!("inkling2_loop on CUDA: {} positions past the cap of {}", pos0 + ids.len(), d.cap);
        return false;
    }
    // `CMF_INKLING_WINDOW_PROBE=1` (a measurement, not a run): after a prompt,
    // the time of a call of 1..16 rows with every row's logits, as a draft
    // verify would take it; the sequence's state is spoiled
    if pos0 > 0 && std::env::var("CMF_INKLING_WINDOW_PROBE").as_deref() == Ok("1") {
        let vocab = g.lm_head.rows();
        let mut line = Vec::new();
        for n in [1usize, 2, 3, 4, 5, 6, 8, 12, 16] {
            let mut lg = Vec::new();
            let ids_n = vec![ids.first().copied().unwrap_or(0); n];
            let _ = forward_(c, d, g, cfg, &ids_n, pos0, true, true, &mut lg, None);
            let t = std::time::Instant::now();
            for _ in 0..10 {
                let _ = forward_(c, d, g, cfg, &ids_n, pos0, true, true, &mut lg, None);
            }
            line.push(format!("{n}: {:.2} ms", t.elapsed().as_secs_f64() * 100.0));
            let _ = vocab;
        }
        tracing::info!("inkling2_loop CUDA window probe (a call with every row's logits): {}", line.join(", "));
        std::process::exit(0);
    }
    match forward_(c, d, g, cfg, ids, pos0, want, all, logits, pick) {
        Ok(()) => true,
        Err(e) => {
            tracing::error!("inkling2_loop on CUDA: {e}");
            false
        }
    }
}

/// A sequence ends: the device stays for the next one (its rings are cleared
/// when that one starts at position 0).
pub(crate) fn reset(id: u64) {
    DECLINED.lock().unwrap().retain(|&x| x != id);
}

#[allow(clippy::too_many_arguments)]
fn forward_(c: &Ctx, d: &mut Dev, g: &Globals, cfg: &Cfg, ids: &[u32], pos0: usize, want: bool, all: bool, logits: &mut Vec<f32>, mut pick: Option<&mut Vec<u32>>) -> Res<()> {
    let hidden = cfg.dim;
    let vocab = g.lm_head.rows();
    if all {
        logits.clear();
    }
    let mut at = 0usize;
    while at < ids.len() {
        // a prompt's chunk of more than NMAX rows on the prompt path (every
        // row's logits then come from the head 16 rows at a time)
        let big = d.big.is_some() && ids.len() - at > NMAX;
        let n = (ids.len() - at).min(if big { d.rows } else { NMAX });
        let part = &ids[at..at + n];
        let last = at + n == ids.len();
        let lrows = if all { n } else if want && last { 1 } else { 0 };
        // the call's embedded rows, normed on the host
        let mut rows = vec![0f32; n * hidden];
        let mut e = vec![0f32; hidden];
        for (i, &id) in part.iter().enumerate() {
            e.fill(0.0);
            if (id as usize) < g.embed.rows() {
                g.embed.row_f32(&g.model, id as usize, &mut e);
            }
            let ms = e.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
            let r = (1.0 / (ms + cfg.eps).sqrt()) as f32;
            for (o, (&v, &w)) in rows[i * hidden..(i + 1) * hidden].iter_mut().zip(e.iter().zip(&g.embed_norm)) {
                *o = w * (v * r);
            }
        }
        unsafe {
            cu::memcpy_htod_async(d.h[0].ptr, &rows, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(d.cr.ptr, &[(pos0 + at) as i32, n as i32], c.stream).map_err(derr)?;
        }
        let key = (n, lrows);
        d.big_pos0.set(pos0 + at);
        if big && all {
            unsafe { call(c, d, cfg, n, 0, vocab)? };
            let mut r0 = 0usize;
            while r0 < n {
                let m = (n - r0).min(NMAX);
                unsafe { head_rows(c, d, cfg, r0, m, vocab)? };
                let mut v = vec![0f32; m * vocab];
                unsafe { cu::memcpy_dtoh_async(&mut v, d.logits.ptr, c.stream).map_err(derr)? };
                c.sync()?;
                logits.extend_from_slice(&v);
                r0 += m;
            }
            at += n;
            continue;
        }
        if big {
            unsafe { call(c, d, cfg, n, lrows, vocab)? };
        } else if graphs_on() && !prof() {
            if !d.graphs.contains_key(&key) {
                let dd: &Dev = d;
                let exec = unsafe { capture(c, || call(c, dd, cfg, n, lrows, vocab))? };
                d.graphs.insert(key, GraphRec { exec, sig: vec![n as u64, lrows as u64] });
            }
            unsafe { curesult(cs::cuGraphLaunch(d.graphs[&key].exec, c.stream), "graph launch")? };
        } else {
            unsafe { call(c, d, cfg, n, lrows, vocab)? };
        }
        if lrows > 0 && all && !big {
            if let Some(p) = pick.as_deref_mut() {
                let mut v = vec![0i32; lrows];
                let v_i = vocab as i32;
                unsafe {
                    c.launch(d.fx.argmax_rows, (lrows as u32, 1), 1024, 0, kargs!(d.logits.ptr, v_i, d.pick.ptr))?;
                    cu::memcpy_dtoh_async(&mut v, d.pick.ptr, c.stream).map_err(derr)?;
                }
                c.sync()?;
                p.extend(v.iter().map(|&x| x as u32));
                at += n;
                continue;
            }
        }
        if lrows > 0 {
            let mut v = vec![0f32; lrows * vocab];
            unsafe { cu::memcpy_dtoh_async(&mut v, d.logits.ptr, c.stream).map_err(derr)? };
            c.sync()?;
            if all {
                logits.extend_from_slice(&v);
            } else {
                *logits = v;
            }
        }
        at += n;
    }
    c.sync()
}

/// The launches of one call of `n` rows (positions from `cr[0]`): the 48
/// applications, the final mixture and, for the last `lrows` rows, the head.
unsafe fn call(c: &Ctx, d: &Dev, cfg: &Cfg, n: usize, lrows: usize, vocab: usize) -> Res<()> {
    if prof() {
        IN_BIG.store(n > NMAX, Ordering::Relaxed);
    }
    let hidden = cfg.dim;
    let hb = n * hidden * 4;
    let ost = (d.rows * hidden) as i32;
    let bst = (MAXB * hidden) as i32;
    let mut cur = 0usize;
    let mut hin = d.h[0].ptr;
    let mut nb = 0usize;
    let mut npre = 0usize;
    for (ai, (li, pass)) in cfg.plan().into_iter().enumerate() {
        let l = &d.layers[li];
        let s = &d.slots[cfg.slot(li, pass)];
        // the drafter's taps: this application's input rows
        if let Some(dr) = &d.draft {
            if let Some(k) = dr.taps.iter().position(|&t| t == ai) {
                unsafe { cu::memcpy_dtod_async(dr.tap.ptr + ((k * d.rows * hidden) * 4) as u64, hin, hb, c.stream).map_err(derr)? };
            }
        }
        if li == cfg.core_start {
            unsafe {
                if pass == 0 {
                    cu::memcpy_dtod_async(d.pre_h.ptr, hin, hb, c.stream).map_err(derr)?;
                    npre = nb;
                } else {
                    // the previous pass's blocks after its first, and its output
                    cu::memcpy_dtod_async(d.bank.ptr, d.blk.ptr, n * MAXB * hidden * 4, c.stream).map_err(derr)?;
                    cu::memcpy_dtod_async(d.bank_out.ptr, hin, hb, c.stream).map_err(derr)?;
                    hin = d.pre_h.ptr;
                }
            }
        }
        let bank_nb = if pass > 0 { cfg.loop_len() / cfg.block - 1 } else { 0 };
        if pass > 0 && li == cfg.core_start {
            nb = npre;
        }
        let hout = d.h[1 - cur].ptr;
        match &d.big {
            Some(b) if n > NMAX => unsafe { app_big(c, d, b, cfg, l, s, li, pass, n, hin, hout, &mut nb, npre, bank_nb)? },
            _ => unsafe { app(c, d, cfg, l, s, li, pass, n, hin, hout, &mut nb, npre, bank_nb)? },
        }
        cur = 1 - cur;
        hin = hout;
    }
    // the final mixture, its norm and the head
    let f = &d.fx;
    let mut tp = std::time::Instant::now();
    unsafe {
        let nb_i = nb as i32;
        c.launch(f.mix, (1, n as u32), 256, 0, kargs!(d.blk.ptr, nb_i, bst, hin, d.final_q.ptr, d.fin.ptr, ost))?;
        if let Some(dr) = &d.draft {
            draft_commit(c, d, dr, cfg, n)?;
        }
        if lrows > 0 {
            head_rows(c, d, cfg, n - lrows, lrows, vocab)?;
        }
    }
    st(c, 10, &mut tp)?;
    if prof() && n > NMAX {
        IN_BIG.store(false, Ordering::Relaxed);
        prof_report_big(n);
    }
    if prof() && n == 1 {
        let k = PROF_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if k % 64 == 0 {
            prof_report();
        }
    }
    Ok(())
}

/// The drafter's context of a call's rows: the final normed state as the last
/// tap, the mixes, every layer's context keys and values at the rows'
/// positions, the Markov head's context into its ring.
unsafe fn draft_commit(c: &Ctx, d: &Dev, dr: &DraftD, cfg: &Cfg, n: usize) -> Res<()> {
    let f = &d.fx;
    let hidden = cfg.dim;
    let rows = d.rows;
    let kvs = cfg.nkv * cfg.hd;
    let nt = dr.taps.len() + 1;
    let (nul, hs, eps) = (0u64, hidden as i32, cfg.eps as f32);
    let last = dr.tap.ptr + (((nt - 1) * rows * hidden) * 4) as u64;
    let (nt_i, rows_i, nmix_i, fr_i) = (nt as i32, rows as i32, (dr.layers.len() + 1) as i32, FR as i32);
    unsafe {
        c.launch(f.rms, (n as u32, 1), 256, 0, kargs!(d.fin.ptr, d.final_norm.ptr, eps, last, nul, nul, hs))?;
        c.launch(
            f.ds_mix,
            (n as u32, (dr.layers.len() + 1) as u32),
            256,
            0,
            kargs!(dr.tap.ptr, nt_i, rows_i, dr.alpha.ptr, dr.norms.ptr, dr.eps, nmix_i, dr.mixp[0].ptr, dr.mixp[1].ptr, dr.mixh.ptr, dr.fr.ptr, fr_i, d.cr.ptr),
        )?;
        for (l, dl) in dr.layers.iter().enumerate() {
            let off = ((l * rows * hidden) * 2) as u64;
            if n <= NMAX {
                let pl = Planes { p0: dr.mixp[0].ptr + off, p1: dr.mixp[1].ptr + off, p2: dr.mixp[1].ptr + off, stride: hidden };
                dl.kctx.run(c, &d.hfx, &pl, hidden, n)?;
                dl.vctx.run(c, &d.hfx, &pl, hidden, n)?;
            } else {
                let b = d.big.as_ref().ok_or("a prompt call without the prompt path")?;
                wmul(c, &d.hfx, b, &dl.kctx_m, dr.mixh.ptr + off, dr.ckb.ptr, kvs, n)?;
                wmul(c, &d.hfx, b, &dl.vctx_m, dr.mixh.ptr + off, dr.cvb.ptr, kvs, n)?;
            }
            let (cap_i, ring_i) = (dl.cap as i32, dl.ring as i32);
            c.launch(f.ds_ctx, (n as u32, cfg.nkv as u32), cfg.hd as u32, 0, kargs!(dr.ckb.ptr, dr.cvb.ptr, dl.kc_norm.ptr, dr.eps, dr.theta, dl.ck.ptr, dl.cv.ptr, cap_i, ring_i, d.cr.ptr))?;
        }
    }
    Ok(())
}

/// The drafter on the card (`g.draft`): its matrices in the arm's codecs, its
/// caches for `cap` positions, the target's head for its block.
#[allow(clippy::too_many_arguments)]
fn build_draft(c: &Ctx, st: &mut State, g: &Globals, dr: &crate::inkling::Draft, cfg: &Cfg, rows: usize, hm: &Arc<DevMat>, cap: usize) -> Res<DraftD> {
    let hidden = cfg.dim;
    let (nh, nkv, hd) = (cfg.nh, cfg.nkv, cfg.hd);
    let (qs, kvs) = (nh * hd, nkv * hd);
    let b = dr.block;
    if dr.layers.is_empty() || dr.targets.len() < 2 || b < 2 || b > NMAX + 1 || dr.rank * 3 > 1024 {
        return Err("the drafter's geometry".into());
    }
    let f4 = |n: usize| Buf::new(n * 4);
    let h2 = |n: usize| Buf::new(n * 2 + 64);
    let f32b = |v: &[f32]| Buf::from_slice(v);
    let inter = dr.layers[0].gate.rows();
    let x = f4(b * hidden)?;
    let xp = [h2(b * hidden)?, h2(b * hidden)?];
    let (dq, dk, dv) = (f4(b * qs)?, f4(b * kvs)?, f4(b * kvs)?);
    let att = f4(b * qs)?;
    let ap = [h2(b * qs)?, h2(b * qs)?];
    let gp = [h2(b * inter)?, h2(b * inter)?];
    let (ckb, cvb) = (f4(rows * kvs)?, f4(rows * kvs)?);
    let mut layers = Vec::new();
    let mut all = Vec::new();
    for dl in &dr.layers {
        let (q, k, v, o) = (mat(c, st, &dl.q)?, mat(c, st, &dl.k)?, mat(c, st, &dl.v)?, mat(c, st, &dl.o)?);
        let (kc, vc, dn) = (mat(c, st, &dl.k_ctx)?, mat(c, st, &dl.v_ctx)?, mat(c, st, &dl.down)?);
        let (mg, gwg) = gw_of(&dl.gate).ok_or("drafter gate codec")?;
        let (_, gwu) = gw_of(&dl.up).ok_or("drafter up codec")?;
        let pm = hmat_pair(c, st, &mg, &gwg, &gwu, inter, hidden)?;
        let qkv = mmaset(&[(&q, dq.ptr, 0, qs, qs), (&k, dk.ptr, 0, kvs, kvs), (&v, dv.ptr, 0, kvs, kvs)])?;
        let kctx = mmaset(&[(&kc, ckb.ptr, 0, kvs, kvs)])?;
        let vctx = mmaset(&[(&vc, cvb.ptr, 0, kvs, kvs)])?;
        let oset = mmaset(&[(&o, x.ptr, x.ptr, hidden, hidden)])?;
        let gu = mmaset_pair(&pm, gp[0].ptr, gp[1].ptr, inter)?;
        let down = mmaset(&[(&dn, x.ptr, x.ptr, hidden, hidden)])?;
        let lcap = if dl.local { dr.window + 64 } else { cap + b };
        let ck = Buf::new(lcap * kvs * 2 + 64)?;
        let cv = Buf::new(lcap * kvs * 2 + 64)?;
        c.kv_bytes.fetch_add((ck.bytes + cv.bytes) as u64, Ordering::Relaxed);
        layers.push(DLayerD {
            qkv,
            kctx,
            vctx,
            kctx_m: kc.clone(),
            vctx_m: vc.clone(),
            o: oset,
            gu,
            down,
            in_norm: f32b(&dl.in_norm)?,
            post_norm: f32b(&dl.post_norm)?,
            q_norm: f32b(&dl.q_norm)?,
            k_norm: f32b(&dl.k_norm)?,
            kc_norm: f32b(&dl.k_ctx_norm)?,
            ck,
            cv,
            cap: lcap,
            ring: dl.local,
            _mats: vec![q, k, v, o, kc, vc, dn, pm],
        });
    }
    let nt = dr.targets.len();
    let mut norms = Vec::with_capacity((dr.layers.len() + 1) * hidden);
    for dl in &dr.layers {
        norms.extend_from_slice(&dl.dflare_norm);
    }
    norms.extend_from_slice(&dr.markov_norm);
    let mut masks = Vec::with_capacity((b - 1) * hidden);
    for _ in 1..b {
        masks.extend_from_slice(&dr.mask);
    }
    let vocab = g.lm_head.rows();
    let dlog = f4((b - 1) * vocab)?;
    let head = mmaset(&[(hm, dlog.ptr, 0, vocab, vocab)])?;
    let fold = mat(c, st, &dr.fold)?;
    let latp = [h2(dr.rank)?, h2(dr.rank)?];
    let mut folds = Vec::with_capacity(b - 1);
    for i in 0..b - 1 {
        let y = dlog.ptr + ((i * vocab) * 4) as u64;
        folds.push(mmaset(&[(&fold, y, y, vocab, vocab)])?);
    }
    all.push(fold);
    // the previous token's rows, f16
    let rank = dr.rank;
    let mut w1 = vec![0u16; dr.w1.rows() * rank];
    let mut row = vec![0f32; rank];
    for r in 0..dr.w1.rows() {
        dr.w1.row_f32(r, &mut row);
        for (o, &v) in w1[r * rank..(r + 1) * rank].iter_mut().zip(&row) {
            *o = cortiq_core::quant::f32_to_f16(v);
        }
    }
    Ok(DraftD {
        layers,
        taps: dr.targets[..nt - 1].to_vec(),
        tap: f4(nt * rows * hidden)?,
        alpha: f32b(&dr.alpha)?,
        norms: f32b(&norms)?,
        fnorm: f32b(&dr.norm)?,
        mixp: [h2(dr.layers.len() * rows * hidden)?, h2(dr.layers.len() * rows * hidden)?],
        mixh: h2(dr.layers.len() * rows * hidden)?,
        ckb,
        cvb,
        fr: f4(FR * hidden)?,
        masks: f32b(&masks)?,
        x,
        xp,
        dq,
        dk,
        dv,
        att,
        ap,
        gp,
        hf: f4(b * hidden)?,
        hp: [h2(b * hidden)?, h2(b * hidden)?],
        dcr: Buf::new(64)?,
        head,
        dlog,
        rnn_w: f32b(&dr.rnn_w)?,
        rnn_b: f32b(&dr.rnn_b)?,
        ch: f4((b - 1) * 3 * rank)?,
        cf: f4(3 * rank)?,
        state: f4(rank)?,
        prev: Buf::new(64)?,
        ids: Buf::new(64)?,
        w1: Buf::from_slice(&w1)?,
        folds,
        fold_b: f32b(&dr.fold_b)?,
        latp,
        block: b,
        rank,
        window: dr.window,
        theta: dr.rope_theta as f32,
        eps: dr.eps,
        inter,
        _mats: all,
    })
}

/// One block of the drafter from the anchor at `pos` (the target has run every
/// position before it): up to `k` drafts, on the card.
pub(crate) fn draft_round(id: u64, g: &Globals, cfg: &Cfg, anchor: u32, pos: usize, k: usize) -> Option<Vec<u32>> {
    let c = super::super::ctx()?;
    let guard = DEVICES.lock().unwrap();
    let d = guard.as_ref()?.get(&id)?;
    let dr = d.draft.as_ref()?;
    if pos == 0 || k == 0 || pos + dr.block > d.cap {
        return None;
    }
    let k = k.min(dr.block - 1);
    match unsafe { draft_block(c, d, dr, g, cfg, anchor, pos, k) } {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!("inkling2_loop on CUDA: a draft failed: {e}");
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn draft_block(c: &Ctx, d: &Dev, dr: &DraftD, g: &Globals, cfg: &Cfg, anchor: u32, pos: usize, k: usize) -> Res<Vec<u32>> {
    let f = &d.fx;
    let hidden = cfg.dim;
    let (nh, nkv, hd) = (cfg.nh, cfg.nkv, cfg.hd);
    let qs = nh * hd;
    let b = dr.block;
    let vocab = g.lm_head.rows();
    let (bu, eps, hs) = (b as u32, dr.eps, hidden as i32);
    // the anchor's normed embedding, as the target's input row
    let mut e = vec![0f32; hidden];
    if (anchor as usize) < g.embed.rows() {
        g.embed.row_f32(&g.model, anchor as usize, &mut e);
    }
    let ms = e.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hidden as f64;
    let r = (1.0 / (ms + cfg.eps).sqrt()) as f32;
    let row0: Vec<f32> = e.iter().zip(&g.embed_norm).map(|(&v, &w)| w * (v * r)).collect();
    unsafe {
        cu::memcpy_htod_async(dr.x.ptr, &row0, c.stream).map_err(derr)?;
        cu::memcpy_dtod_async(dr.x.ptr + (hidden * 4) as u64, dr.masks.ptr, (b - 1) * hidden * 4, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(dr.dcr.ptr, &[pos as i32, b as i32], c.stream).map_err(derr)?;
        cu::memcpy_htod_async(dr.prev.ptr, &[anchor as i32], c.stream).map_err(derr)?;
        cu::memset_d8_async(dr.state.ptr, 0, dr.rank * 4, c.stream).map_err(derr)?;
        let xpl = Planes { p0: dr.xp[0].ptr, p1: dr.xp[1].ptr, p2: dr.xp[1].ptr, stride: hidden };
        let nul = 0u64;
        for dl in &dr.layers {
            c.launch(f.rms, (bu, 1), 256, 0, kargs!(dr.x.ptr, dl.in_norm.ptr, eps, nul, dr.xp[0].ptr, dr.xp[1].ptr, hs))?;
            dl.qkv.run(c, &d.hfx, &xpl, hidden, b)?;
            let (cap_i, ring_i) = (dl.cap as i32, dl.ring as i32);
            c.launch(f.ds_qk, (bu, (nh + nkv) as u32), hd as u32, 0, kargs!(dr.dq.ptr, dr.dk.ptr, dr.dv.ptr, dl.q_norm.ptr, dl.k_norm.ptr, eps, dr.theta, dl.ck.ptr, dl.cv.ptr, cap_i, ring_i, dr.dcr.ptr))?;
            let (win_i, b_i) = (if dl.ring { dr.window as i32 } else { 0 }, b as i32);
            c.launch(f.ds_attn, (nh as u32, bu), 256, 0, kargs!(dr.dq.ptr, dl.ck.ptr, dl.cv.ptr, cap_i, ring_i, win_i, b_i, dr.att.ptr, dr.dcr.ptr))?;
            let tot = (b * qs) as i32;
            c.launch(f.planes, ((b * qs).div_ceil(256) as u32, 1), 256, 0, kargs!(dr.att.ptr, dr.ap[0].ptr, dr.ap[1].ptr, tot))?;
            let apl = Planes { p0: dr.ap[0].ptr, p1: dr.ap[1].ptr, p2: dr.ap[1].ptr, stride: qs };
            dl.o.run(c, &d.hfx, &apl, qs, b)?;
            c.launch(f.rms, (bu, 1), 256, 0, kargs!(dr.x.ptr, dl.post_norm.ptr, eps, nul, dr.xp[0].ptr, dr.xp[1].ptr, hs))?;
            dl.gu.run(c, &d.hfx, &xpl, hidden, b)?;
            let gpl = Planes { p0: dr.gp[0].ptr, p1: dr.gp[1].ptr, p2: dr.gp[1].ptr, stride: dr.inter };
            dl.down.run(c, &d.hfx, &gpl, dr.inter, b)?;
        }
        // h; the rows after the anchor's through the target's head (the drafts' base logits)
        c.launch(f.rms, (bu, 1), 256, 0, kargs!(dr.x.ptr, dr.fnorm.ptr, eps, dr.hf.ptr, dr.hp[0].ptr, dr.hp[1].ptr, hs))?;
        let hpl = Planes { p0: dr.hp[0].ptr + (hidden * 2) as u64, p1: dr.hp[1].ptr + (hidden * 2) as u64, p2: dr.hp[1].ptr + (hidden * 2) as u64, stride: hidden };
        dr.head.run(c, &d.hfx, &hpl, hidden, k)?;
        // the recurrence's input terms: W_h · h_{i+1} for every draft, W_f · f
        // (the Markov context of position pos − 1)
        let rk = dr.rank;
        let ws = 2 * rk + 2 * hidden;
        let w_h = dr.rnn_w.ptr + ((2 * rk) * 4) as u64;
        let w_f = dr.rnn_w.ptr + ((2 * rk + hidden) * 4) as u64;
        c.gemm(true, false, 3 * rk, k, hidden, 1.0, w_h, ws, dr.hf.ptr + (hidden * 4) as u64, hidden, 0.0, dr.ch.ptr, 3 * rk, Gemm::F32)?;
        let fp = dr.fr.ptr + ((((pos - 1) % FR) * hidden) * 4) as u64;
        c.gemm(true, false, 3 * rk, 1, hidden, 1.0, w_f, ws, fp, hidden, 0.0, dr.cf.ptr, 3 * rk, Gemm::F32)?;
        let (ws_i, rk_i, v_i) = (ws as i32, rk as i32, vocab as i32);
        let lpl = Planes { p0: dr.latp[0].ptr, p1: dr.latp[1].ptr, p2: dr.latp[1].ptr, stride: rk };
        for i in 0..k {
            let chi = dr.ch.ptr + ((i * 3 * rk) * 4) as u64;
            c.launch(f.ds_markov, (1, 1), (3 * rk) as u32, 5 * rk * 4, kargs!(chi, dr.cf.ptr, dr.rnn_b.ptr, dr.rnn_w.ptr, ws_i, rk_i, dr.state.ptr, dr.prev.ptr, dr.w1.ptr, dr.latp[0].ptr, dr.latp[1].ptr))?;
            dr.folds[i].run(c, &d.hfx, &lpl, rk, 1)?;
            let lg = dr.dlog.ptr + ((i * vocab) * 4) as u64;
            let i_i = i as i32;
            c.launch(f.ds_pick, (1, 1), 1024, 0, kargs!(lg, dr.fold_b.ptr, v_i, i_i, dr.ids.ptr, dr.prev.ptr))?;
        }
        let mut ids = vec![0i32; k];
        cu::memcpy_dtoh_async(&mut ids, dr.ids.ptr, c.stream).map_err(derr)?;
        c.sync()?;
        Ok(ids.into_iter().map(|v| v as u32).collect())
    }
}

/// The rings of every application saved before a draft verify.
pub(crate) fn ring_snap(id: u64) -> bool {
    let Some(c) = super::super::ctx() else { return false };
    let guard = DEVICES.lock().unwrap();
    let Some(d) = guard.as_ref().and_then(|m| m.get(&id)) else { return false };
    let maxw = (d.slots.first().map_or(0, |s| s.hist[2].bytes / 4 / NMAX)) as u32;
    unsafe { c.launch(d.fx.ring_snap, (maxw.div_ceil(256), d.nrings as u32), 256, 0, kargs!(d.rings_tab.ptr)).is_ok() && c.sync().is_ok() }
}

/// The rings put back as they stand after the verify call's first `r` rows.
pub(crate) fn ring_back(id: u64, r: usize) -> bool {
    let Some(c) = super::super::ctx() else { return false };
    let guard = DEVICES.lock().unwrap();
    let Some(d) = guard.as_ref().and_then(|m| m.get(&id)) else { return false };
    let maxw = (d.slots.first().map_or(0, |s| s.hist[2].bytes / 4 / NMAX)) as u32;
    let r_i = r as i32;
    unsafe { c.launch(d.fx.ring_back, (maxw.div_ceil(256), d.nrings as u32), 256, 0, kargs!(d.rings_tab.ptr, r_i)).is_ok() && c.sync().is_ok() }
}

/// The head over rows `r0 .. r0 + m` of the final mixture (`m` ≤ `NMAX`):
/// the norm, the logits into `logits`, the softcap.
unsafe fn head_rows(c: &Ctx, d: &Dev, cfg: &Cfg, r0: usize, m: usize, vocab: usize) -> Res<()> {
    let f = &d.fx;
    let hidden = cfg.dim;
    let fin = d.fin.ptr + (r0 * hidden * 4) as u64;
    let (nul, hs) = (0u64, hidden as i32);
    unsafe {
        c.launch(f.rms, (m as u32, 1), 256, 0, kargs!(fin, d.final_norm.ptr, cfg.eps as f32, nul, d.pf[0].ptr, d.pf[1].ptr, hs))?;
        let pl = Planes { p0: d.pf[0].ptr, p1: d.pf[1].ptr, p2: d.pf[1].ptr, stride: hidden };
        d.head.run(c, &d.hfx, &pl, hidden, m)?;
        let (v_i, cap) = (vocab as i32, cfg.logit_cap);
        if cap > 0.0 {
            c.launch(f.softcap, (vocab.div_ceil(256) as u32, m as u32), 256, 0, kargs!(d.logits.ptr, v_i, cap))?;
        }
    }
    Ok(())
}

/// One layer application of the call's rows: `hin` the running sums in,
/// `hout` out; `nb` the block states the rows hold (a boundary layer adds one).
#[allow(clippy::too_many_arguments)]
unsafe fn app(c: &Ctx, d: &Dev, cfg: &Cfg, l: &LayerD, s: &SlotD, li: usize, pass: usize, n: usize, hin: u64, hout: u64, nb: &mut usize, npre: usize, bank_nb: usize) -> Res<()> {
    let f = &d.fx;
    let hidden = cfg.dim;
    let rows = d.rows;
    let ost = (rows * hidden) as i32;
    let bst = (MAXB * hidden) as i32;
    let (ni, nu) = (n as i32, n as u32);
    let eps = cfg.eps as f32;
    let hs = hidden as i32;
    let fl = |b: &Buf, i: usize| b.ptr + (i * rows * hidden * 4) as u64;
    let mut tp = std::time::Instant::now();
    unsafe {
        // 1. the attention inputs
        let mut inp = [hin, hin, hin];
        if let Some((aq, nq)) = &l.aq {
            let nb_i = *nb as i32;
            c.launch(f.mix, (*nq as u32, nu), 256, 0, kargs!(d.blk.ptr, nb_i, bst, hin, aq.ptr, d.in3.ptr, ost))?;
            inp = if *nq == 3 { [fl(&d.in3, 0), fl(&d.in3, 1), fl(&d.in3, 2)] } else { [fl(&d.in3, 0); 3] };
        }
        st(c, 0, &mut tp)?;
        // 2. the messages of a later pass, gated into q, k, v
        let gates = if pass > 0 { l.gates.as_ref() } else { None };
        if let Some(gs) = gates {
            let i = li - cfg.core_start;
            let q = d.msg_q.ptr + (i * 4 * hidden * 4) as u64;
            let bnb = bank_nb as i32;
            let bank = d.bank.ptr + ((npre + 1) * hidden * 4) as u64;
            c.launch(f.mix, (4, nu), 256, 0, kargs!(bank, bnb, bst, d.bank_out.ptr, q, d.msg.ptr, ost))?;
            for j in 0..3 {
                gate(c, f, &gs[j], inp[j], fl(&d.msg, j), d.gd.ptr, fl(&d.gin, j), n, hidden, eps)?;
                inp[j] = fl(&d.gin, j);
            }
        }
        st(c, 1, &mut tp)?;
        // 3. the input norms (a local layer's three are one weight; without
        // messages its inputs are one row)
        let one = l.sliding && gates.is_none();
        for j in 0..(if one { 1 } else { 3 }) {
            c.launch(f.rms, (nu, 1), 256, 0, kargs!(inp[j], l.norms[j].ptr, eps, d.x[j].ptr, d.xp[j][0].ptr, d.xp[j][1].ptr, hs))?;
        }
        let (xk, xv) = if one { (0, 0) } else { (1, 2) };
        let pl = |j: usize| Planes { p0: d.xp[j][0].ptr, p1: d.xp[j][1].ptr, p2: d.xp[j][1].ptr, stride: hidden };
        st(c, 2, &mut tp)?;
        // 4. the projections
        match (&l.qkv, one) {
            (Some(all), true) => all.run(c, &d.hfx, &pl(0), hidden, n)?,
            _ => {
                l.qset.run(c, &d.hfx, &pl(0), hidden, n)?;
                l.kset.run(c, &d.hfx, &pl(xk), hidden, n)?;
                l.vset.run(c, &d.hfx, &pl(xv), hidden, n)?;
            }
        }
        let nh = cfg.nh;
        let (nh_i, kp_i) = (nh as i32, (cfg.nkv * 4) as i32);
        c.launch(f.fmv, (nh.div_ceil(8) as u32, nu), 256, 0, kargs!(l.sink.ptr, nh_i, d.x[0].ptr, d.sk.ptr))?;
        let (mut dk, mut dv) = (0u64, 0u64);
        if let (Some(wk), Some(wv)) = (&l.dk, &l.dv) {
            c.launch(f.fmv, ((cfg.nkv * 4).div_ceil(8) as u32, nu), 256, 0, kargs!(wk.ptr, kp_i, d.x[xk].ptr, d.dkb.ptr))?;
            c.launch(f.fmv, ((cfg.nkv * 4).div_ceil(8) as u32, nu), 256, 0, kargs!(wv.ptr, kp_i, d.x[xv].ptr, d.dvb.ptr))?;
            dk = d.dkb.ptr;
            dv = d.dvb.ptr;
        }
        st(c, 3, &mut tp)?;
        // 5. queries and relative states; keys and values into the cache
        let full = !l.sliding as i32;
        let (alpha, floor) = (cfg.log_alpha as f32, cfg.log_floor as f32);
        c.launch(f.qr, (nh as u32, nu), cfg.hd as u32, 0, kargs!(d.yq.ptr, d.yr.ptr, l.q_scale.ptr, l.rel_scale.ptr, cfg.qk_clip, cfg.rel_clip, full, alpha, floor, d.cr.ptr))?;
        let ring = s.ring as i32;
        c.launch(
            f.kv,
            (cfg.nkv as u32, 2),
            cfg.hd as u32,
            0,
            kargs!(d.yk.ptr, d.yv.ptr, dk, dv, l.tk.ptr, l.tv.ptr, s.rings[0].ptr, s.rings[1].ptr, s.k.ptr, s.v.ptr, ring, ni, cfg.qk_clip, eps, d.cr.ptr, s.hist[0].ptr, s.hist[1].ptr),
        )?;
        st(c, 4, &mut tp)?;
        // 6. the attention, its gate, the O projection
        let (ext, scale) = (l.extent as i32, 1.0f32 / (cfg.hd as f32).sqrt());
        let win = if s.ring > 0 { cfg.window as i32 } else { 0 };
        if split_on() {
            // the tiles of the longest span the slot holds: a recorded call
            // replays at any position
            let tiles = if s.ring > 0 { cfg.window } else { d.cap }.div_ceil(ATILE);
            let t_i = tiles as i32;
            c.launch_3d(
                f.attn_split,
                (nh as u32, nu, tiles as u32),
                256,
                0,
                kargs!(d.yq.ptr, d.yr.ptr, l.rel_table.ptr, ext, s.k.ptr, s.v.ptr, ring, win, scale, d.part.ptr, t_i, d.cr.ptr),
            )?;
            c.launch(f.attn_merge, (nh as u32, nu), cfg.hd as u32, 0, kargs!(d.part.ptr, t_i, d.sk.ptr, win, d.att.ptr, d.cr.ptr))?;
        } else {
            c.launch(f.attn, (nh as u32, nu), 256, 0, kargs!(d.yq.ptr, d.yr.ptr, d.sk.ptr, l.rel_table.ptr, ext, s.k.ptr, s.v.ptr, ring, win, scale, d.att.ptr, d.cr.ptr))?;
        }
        st(c, 5, &mut tp)?;
        let qs = nh * cfg.hd;
        let qs_i = qs as i32;
        c.launch(f.ogate, (qs.div_ceil(256) as u32, nu), 256, 0, kargs!(d.att.ptr, d.yg.ptr, d.po[0].ptr, d.po[1].ptr, qs_i))?;
        let po = Planes { p0: d.po[0].ptr, p1: d.po[1].ptr, p2: d.po[1].ptr, stride: qs };
        l.oset.run(c, &d.hfx, &po, qs, n)?;
        st(c, 6, &mut tp)?;
        // 7. the output convolution; a boundary closes the running sum into a block
        let bslot = if l.boundary { *nb as i32 } else { -1 };
        let hb = hidden.div_ceil(256) as u32;
        c.launch(f.post, (hb, 1), 256, 0, kargs!(d.ya.ptr, l.tpa.ptr, s.rings[2].ptr, hin, d.blk.ptr, bslot, bst, d.r.ptr, ni, s.hist[2].ptr))?;
        if l.boundary {
            *nb += 1;
        }
        st(c, 7, &mut tp)?;
        // 8. the MoE input
        let nb_i = *nb as i32;
        c.launch(f.mix, (1, nu), 256, 0, kargs!(d.blk.ptr, nb_i, bst, d.r.ptr, l.mq.ptr, d.m.ptr, ost))?;
        let mut mi = d.m.ptr;
        if let Some(gs) = gates {
            gate(c, f, &gs[3], mi, fl(&d.msg, 3), d.gd.ptr, d.mg.ptr, n, hidden, eps)?;
            mi = d.mg.ptr;
        }
        c.launch(f.rms, (nu, 1), 256, 0, kargs!(mi, l.post_norm.ptr, eps, d.xm.ptr, d.pm[0].ptr, d.pm[1].ptr, hs))?;
        st(c, 8, &mut tp)?;
        // 9. the routed experts into y
        moe(c, d, cfg, l, n)?;
        st(c, 9, &mut tp)?;
        // 10. the output convolution onto the running sum
        let (nul, m1) = (0u64, -1i32);
        c.launch(f.post, (hb, 1), 256, 0, kargs!(d.y.ptr, l.tpm.ptr, s.rings[3].ptr, d.r.ptr, nul, m1, bst, hout, ni, s.hist[3].ptr))?;
        st(c, 7, &mut tp)?;
    }
    Ok(())
}

/// One layer application of a prompt call (more than `NMAX` rows): `app`'s
/// stages with the weights as cuBLAS GEMMs (`wmul`), the convolutions a row a
/// block (their rings moved on after the call) and the routed experts grouped
/// by expert (`moe_big`).
#[allow(clippy::too_many_arguments)]
unsafe fn app_big(c: &Ctx, d: &Dev, b: &Big, cfg: &Cfg, l: &LayerD, s: &SlotD, li: usize, pass: usize, n: usize, hin: u64, hout: u64, nb: &mut usize, npre: usize, bank_nb: usize) -> Res<()> {
    let f = &d.fx;
    let fx = &d.hfx;
    let hidden = cfg.dim;
    let rows = d.rows;
    let ost = (rows * hidden) as i32;
    let bst = (MAXB * hidden) as i32;
    let (ni, nu) = (n as i32, n as u32);
    let eps = cfg.eps as f32;
    let hs = hidden as i32;
    let (nh, nkv, hd) = (cfg.nh, cfg.nkv, cfg.hd);
    let (qs, kvs, rs) = (nh * hd, nkv * hd, nh * cfg.d_rel);
    let fl = |bb: &Buf, i: usize| bb.ptr + (i * rows * hidden * 4) as u64;
    let h16 = |p0: u64, p1: u64, out: u64, width: usize| -> Res<()> {
        let tot = n * width;
        let tot_i = tot as i32;
        unsafe { c.launch(f.h16, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(p0, p1, out, tot_i)) }
    };
    let ring = |y: u64, r: &Buf, width: usize| -> Res<()> {
        let w_i = width as i32;
        unsafe { c.launch(f.ring_rows, (width.div_ceil(256) as u32, 1), 256, 0, kargs!(y, r.ptr, w_i, ni)) }
    };
    let mut tp = std::time::Instant::now();
    unsafe {
        // 1. the attention inputs
        let mut inp = [hin, hin, hin];
        if let Some((aq, nq)) = &l.aq {
            let nb_i = *nb as i32;
            c.launch(f.mix, (*nq as u32, nu), 256, 0, kargs!(d.blk.ptr, nb_i, bst, hin, aq.ptr, d.in3.ptr, ost))?;
            inp = if *nq == 3 { [fl(&d.in3, 0), fl(&d.in3, 1), fl(&d.in3, 2)] } else { [fl(&d.in3, 0); 3] };
        }
        st(c, 0, &mut tp)?;
        // 2. the messages of a later pass, gated into q, k, v
        let gates = if pass > 0 { l.gates.as_ref() } else { None };
        if let Some(gs) = gates {
            let i = li - cfg.core_start;
            let q = d.msg_q.ptr + (i * 4 * hidden * 4) as u64;
            let bnb = bank_nb as i32;
            let bank = d.bank.ptr + ((npre + 1) * hidden * 4) as u64;
            c.launch(f.mix, (4, nu), 256, 0, kargs!(bank, bnb, bst, d.bank_out.ptr, q, d.msg.ptr, ost))?;
            for j in 0..3 {
                gate_big(c, f, b, &gs[j], inp[j], fl(&d.msg, j), d.gd.ptr, fl(&d.gin, j), n, hidden, eps)?;
                inp[j] = fl(&d.gin, j);
            }
        }
        st(c, 1, &mut tp)?;
        // 3. the input norms and their f16 rows
        let one = l.sliding && gates.is_none();
        for j in 0..(if one { 1 } else { 3 }) {
            c.launch(f.rms, (nu, 1), 256, 0, kargs!(inp[j], l.norms[j].ptr, eps, d.x[j].ptr, d.xp[j][0].ptr, d.xp[j][1].ptr, hs))?;
            h16(d.xp[j][0].ptr, d.xp[j][1].ptr, b.x16[j].ptr, hidden)?;
        }
        let (xk, xv) = if one { (0, 0) } else { (1, 2) };
        st(c, 2, &mut tp)?;
        // 4. the projections
        let [wq, wg, wr, wk, wv, wo] = &l.dense;
        wmul(c, fx, b, wq, b.x16[0].ptr, d.yq.ptr, qs, n)?;
        wmul(c, fx, b, wg, b.x16[0].ptr, d.yg.ptr, qs, n)?;
        wmul(c, fx, b, wr, b.x16[0].ptr, d.yr.ptr, rs, n)?;
        wmul(c, fx, b, wk, b.x16[xk].ptr, d.yk.ptr, kvs, n)?;
        wmul(c, fx, b, wv, b.x16[xv].ptr, d.yv.ptr, kvs, n)?;
        let (nh_i, kp_i) = (nh as i32, (nkv * 4) as i32);
        c.launch(f.fmv, (nh.div_ceil(8) as u32, nu), 256, 0, kargs!(l.sink.ptr, nh_i, d.x[0].ptr, d.sk.ptr))?;
        let (mut dk, mut dv) = (0u64, 0u64);
        if let (Some(wdk), Some(wdv)) = (&l.dk, &l.dv) {
            c.launch(f.fmv, ((nkv * 4).div_ceil(8) as u32, nu), 256, 0, kargs!(wdk.ptr, kp_i, d.x[xk].ptr, d.dkb.ptr))?;
            c.launch(f.fmv, ((nkv * 4).div_ceil(8) as u32, nu), 256, 0, kargs!(wdv.ptr, kp_i, d.x[xv].ptr, d.dvb.ptr))?;
            dk = d.dkb.ptr;
            dv = d.dvb.ptr;
        }
        st(c, 3, &mut tp)?;
        // 5. queries and relative states; keys and values into the cache, a row a block
        let full = !l.sliding as i32;
        let (alpha, floor) = (cfg.log_alpha as f32, cfg.log_floor as f32);
        c.launch(f.qr, (nh as u32, nu), hd as u32, 0, kargs!(d.yq.ptr, d.yr.ptr, l.q_scale.ptr, l.rel_scale.ptr, cfg.qk_clip, cfg.rel_clip, full, alpha, floor, d.cr.ptr))?;
        let ring_i = s.ring as i32;
        c.launch(
            f.kv_rows,
            ((2 * nkv) as u32, nu),
            hd as u32,
            0,
            kargs!(d.yk.ptr, d.yv.ptr, dk, dv, l.tk.ptr, l.tv.ptr, s.rings[0].ptr, s.rings[1].ptr, s.k.ptr, s.v.ptr, ring_i, cfg.qk_clip, eps, d.cr.ptr),
        )?;
        ring(d.yk.ptr, &s.rings[0], kvs)?;
        ring(d.yv.ptr, &s.rings[1], kvs)?;
        st(c, 4, &mut tp)?;
        // 6. the attention, its gate, the O projection
        let (ext, scale) = (l.extent as i32, 1.0f32 / (hd as f32).sqrt());
        let win = if s.ring > 0 { cfg.window as i32 } else { 0 };
        if attn_gemm_on() {
            attn_gemm(c, d, b, cfg, l, s, n, ext, win, scale)?;
        } else {
            c.launch(f.attn, (nh as u32, nu), 256, 0, kargs!(d.yq.ptr, d.yr.ptr, d.sk.ptr, l.rel_table.ptr, ext, s.k.ptr, s.v.ptr, ring_i, win, scale, d.att.ptr, d.cr.ptr))?;
        }
        st(c, 5, &mut tp)?;
        let qs_i = qs as i32;
        c.launch(f.ogate, (qs.div_ceil(256) as u32, nu), 256, 0, kargs!(d.att.ptr, d.yg.ptr, d.po[0].ptr, d.po[1].ptr, qs_i))?;
        h16(d.po[0].ptr, d.po[1].ptr, b.o16.ptr, qs)?;
        wmul(c, fx, b, wo, b.o16.ptr, d.ya.ptr, hidden, n)?;
        st(c, 6, &mut tp)?;
        // 7. the output convolution; a boundary closes the running sum into a block
        let bslot = if l.boundary { *nb as i32 } else { -1 };
        let hb = hidden.div_ceil(256) as u32;
        c.launch(f.post_rows, (hb, nu), 256, 0, kargs!(d.ya.ptr, l.tpa.ptr, s.rings[2].ptr, hin, d.blk.ptr, bslot, bst, d.r.ptr))?;
        ring(d.ya.ptr, &s.rings[2], hidden)?;
        if l.boundary {
            *nb += 1;
        }
        st(c, 7, &mut tp)?;
        // 8. the MoE input
        let nb_i = *nb as i32;
        c.launch(f.mix, (1, nu), 256, 0, kargs!(d.blk.ptr, nb_i, bst, d.r.ptr, l.mq.ptr, d.m.ptr, ost))?;
        let mut mi = d.m.ptr;
        if let Some(gs) = gates {
            gate_big(c, f, b, &gs[3], mi, fl(&d.msg, 3), d.gd.ptr, d.mg.ptr, n, hidden, eps)?;
            mi = d.mg.ptr;
        }
        let nul = 0u64;
        c.launch(f.rms, (nu, 1), 256, 0, kargs!(mi, l.post_norm.ptr, eps, d.xm.ptr, nul, nul, hs))?;
        st(c, 8, &mut tp)?;
        // 9. the routed experts into y
        moe_big(c, d, b, cfg, l, n, &mut tp)?;
        st(c, 13, &mut tp)?;
        // 10. the output convolution onto the running sum
        let m1 = -1i32;
        c.launch(f.post_rows, (hb, nu), 256, 0, kargs!(d.y.ptr, l.tpm.ptr, s.rings[3].ptr, d.r.ptr, nul, m1, bst, hout))?;
        ring(d.y.ptr, &s.rings[3], hidden)?;
        st(c, 7, &mut tp)?;
    }
    Ok(())
}

/// A prompt call's attention through cuBLAS: the queries as f16 rows, per KV
/// group S = Q·Kᵀ over the slots the call reads (a local layer's whole ring,
/// a full layer's positions to the call's last), `ink_attn_soft` (scale,
/// relative bias, window, ring positions, sink, softmax), O = P·V into `att`;
/// rows in chunks whose scores fit `ATT_SCORES`.
#[allow(clippy::too_many_arguments)]
unsafe fn attn_gemm(c: &Ctx, d: &Dev, b: &Big, cfg: &Cfg, l: &LayerD, s: &SlotD, n: usize, ext: i32, win: i32, scale: f32) -> Res<()> {
    let f = &d.fx;
    let (nh, nkv, hd) = (cfg.nh, cfg.nkv, cfg.hd);
    let hpk = nh / nkv;
    let (qs, kvs) = (nh * hd, nkv * hd);
    let nk = if s.ring > 0 { s.ring } else { d.big_pos0.get() + n };
    let chunk = (ATT_SCORES / (nh * nk * 4)).clamp(1, n);
    let (tot_i, nk_i, ring_i, n_i) = ((n * qs) as i32, nk as i32, s.ring as i32, n as i32);
    unsafe {
        c.launch(f.f16, ((n * qs).div_ceil(256) as u32, 1), 256, 0, kargs!(d.yq.ptr, b.q16.ptr, tot_i))?;
        let mut r0 = 0usize;
        while r0 < n {
            let rows = chunk.min(n - r0);
            for g in 0..nkv {
                let kp = s.k.ptr + (g * hd * 2) as u64;
                let qp = b.q16.ptr + ((r0 * qs + g * hpk * hd) * 2) as u64;
                let sp = b.sc.ptr + ((g * hpk * rows * nk) * 4) as u64;
                gemm_batched_h(c, true, nk, rows, hd, 1.0, kp, kvs, 0, qp, qs, hd, sp, nk, rows * nk, hpk)?;
            }
            let (rows_i, r0_i) = (rows as i32, r0 as i32);
            c.launch(
                f.attn_soft,
                (rows as u32, nh as u32),
                256,
                0,
                kargs!(b.sc.ptr, b.pr.ptr, d.yr.ptr, d.sk.ptr, l.rel_table.ptr, ext, nk_i, ring_i, win, scale, rows_i, r0_i, n_i, d.cr.ptr),
            )?;
            for g in 0..nkv {
                let vp = s.v.ptr + (g * hd * 2) as u64;
                let pp = b.pr.ptr + ((g * hpk * rows * nk) * 2) as u64;
                let op = d.att.ptr + ((r0 * qs + g * hpk * hd) * 4) as u64;
                gemm_batched_h(c, false, hd, rows, nk, 1.0, vp, kvs, 0, pp, nk, rows * nk, op, qs, hd, hpk)?;
            }
            r0 += rows;
        }
    }
    Ok(())
}

/// `y [n][ldy] = W x` for a prompt call's rows (`x` f16 `[n][cols]`): the
/// weight expanded to f16 (whole when it fits the band, else a band of rows at
/// a time) and cuBLAS on the tensor cores, as the hybrid prompt path's `mul`.
#[allow(clippy::too_many_arguments)]
unsafe fn wmul(c: &Ctx, fx: &HFuncs, b: &Big, m: &DevMat, x: u64, y: u64, ldy: usize, n: usize) -> Res<()> {
    let (rows_i, cols_i) = (m.rows as i32, m.cols as i32);
    let wd = b.wd.ptr;
    unsafe {
        if m.kind == 6 && m.rows * m.cols <= b.wd_elems {
            let threads = (m.rows.div_ceil(16) * (m.cols / 32) * 32) as u64;
            c.launch(fx.expand_p, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd, rows_i, cols_i))?;
            return c.gemm(true, false, m.rows, n, m.cols, 1.0, wd, m.cols, x, m.cols, 0.0, y, ldy, Gemm::Half);
        }
        let tiles = m.rows.div_ceil(16);
        let band = (b.wd_elems / m.cols / 16).max(1);
        let mut t0 = 0usize;
        while t0 < tiles {
            let nt = band.min(tiles - t0);
            let r0 = t0 * 16;
            let nr = (nt * 16).min(m.rows - r0);
            let threads = (nt * (m.cols / 32) * 32) as u64;
            let (t0_i, nt_i) = (t0 as i32, nt as i32);
            if m.kind == 5 || m.kind == 15 {
                let dual_i = (m.kind == 15) as i32;
                c.launch(fx.expand_i8, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux1(), m.aux0(), wd, rows_i, cols_i, t0_i, nt_i, dual_i))?;
            } else {
                c.launch(fx.expand_pr, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(m.w.ptr, m.aux0(), m.aux1(), wd, rows_i, cols_i, t0_i, nt_i))?;
            }
            c.gemm(true, false, nr, n, m.cols, 1.0, wd, m.cols, x, m.cols, 0.0, y + (r0 * 4) as u64, ldy, Gemm::Half)?;
            t0 += nt;
        }
    }
    Ok(())
}

/// A prompt call's routed experts, grouped (the hybrid prompt path's
/// `moe_grouped`): the router as an f32 GEMM, every row's picks
/// (`hy_moe_topk`, the arithmetic of the decode select), the (row, pick)
/// entries sorted by expert and laid out as planes, one tensor-core launch over
/// every expert's gate/up pair (Inkling's activation in the epilogue) and one
/// over its down, the weighted mixture into `y`.
unsafe fn moe_big(c: &Ctx, d: &Dev, b: &Big, cfg: &Cfg, l: &LayerD, n: usize, tp: &mut std::time::Instant) -> Res<()> {
    let fx = &d.hfx;
    let (hidden, e, k, inter) = (cfg.dim, l.e, cfg.top_k, l.inter);
    let nk = n * k;
    let mma = moe_mma_on();
    let rmax = if mma { 64 } else { moe_rmax() };
    let jmax = e + nk.div_ceil(rmax);
    let (gi, tb) = if rmax == 64 { (1usize, 1u32) } else { (0, 2) };
    let (bpg, bpd) = (l.bpj_gu.div_ceil(tb), l.bpj_dn.div_ceil(tb));
    let flags = 1 | 2 | 4 | 32;
    let (n_i, e_i, k_i, nk_i, rmax_i, jmax_i, inter_i, hid_i) = (n as i32, e as i32, k as i32, nk as i32, rmax as i32, jmax as i32, inter as i32, hidden as i32);
    let (bg_i, bd_i) = (bpg as i32, bpd as i32);
    let (nul, zero) = (0u64, 0i32);
    unsafe {
        c.gemm(true, false, e, n, hidden, 1.0, l.router.ptr, hidden, d.xm.ptr, hidden, 0.0, b.lg.ptr, e, Gemm::F32)?;
        c.launch(fx.moe_topk, (n.div_ceil(8) as u32, 1), 256, 0, kargs!(b.lg.ptr, l.bias.ptr, n_i, e_i, k_i, flags, cfg.route_scale, b.pidx.ptr, b.pwt.ptr))?;
        c.launch(
            fx.moe_group,
            (1, 1),
            1024,
            0,
            kargs!(b.pidx.ptr, nk_i, k_i, e_i, rmax_i, jmax_i, l.gu_t.ptr, l.dn_t.ptr, b.gu_a.ptr, b.dn_a.ptr, b.pos.ptr, b.src.ptr, b.qq[0].ptr, b.qq[1].ptr, hid_i, inter_i, b.ys.ptr, bg_i, bd_i),
        )?;
        let tot = (nk * hidden) as u64;
        if mma {
            // the rows in their places as f16; gate/up (the activation in the
            // epilogue) into f16 rows, then down into ys
            let (tot_i, f) = (tot as i32, &d.fx);
            c.launch(f.gather16, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(d.xm.ptr, b.src.ptr, tot_i, b.xg[0].ptr))?;
            st(c, 11, tp)?;
            let (one, zero_i, nul) = (1i32, 0i32, 0u64);
            let gx = ((2 * inter).div_ceil(16)).div_ceil(MMA_TILES) as u32;
            c.launch(f.moe_mma, (gx, jmax as u32), 128, 0, kargs!(b.gu_a.ptr, b.xg[0].ptr, hid_i, one, b.qq[0].ptr, inter_i, nul, zero_i))?;
            st(c, 12, tp)?;
            let dx = (hidden.div_ceil(16)).div_ceil(MMA_TILES) as u32;
            c.launch(f.moe_mma, (dx, jmax as u32), 128, 0, kargs!(b.dn_a.ptr, b.qq[0].ptr, inter_i, zero_i, nul, zero_i, b.ys.ptr, hid_i))?;
            cu::memset_d8_async(d.y.ptr, 0, n * hidden * 4, c.stream).map_err(derr)?;
            c.launch(fx.moe_gcombine, (hidden.div_ceil(256) as u32, n as u32), 256, 0, kargs!(d.y.ptr, b.ys.ptr, b.pos.ptr, b.pwt.ptr, nul, nul, zero, n_i, k_i, hid_i))?;
            return Ok(());
        }
        c.launch(fx.moe_gather, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(d.xm.ptr, b.src.ptr, nk_i, hid_i, b.xg[0].ptr, b.xg[1].ptr, b.xg[1].ptr))?;
        st(c, 11, tp)?;
        let fg = fx.mmp_g[gi];
        c.launch(fg, (jmax as u32 * bpg, 1), 32 * 8, 0, kargs!(b.gu_a.ptr, jmax_i, b.xg[0].ptr, b.xg[1].ptr, b.xg[1].ptr, hid_i, rmax_i, hid_i))?;
        st(c, 12, tp)?;
        c.launch(fg, (jmax as u32 * bpd, 1), 32 * 8, 0, kargs!(b.dn_a.ptr, jmax_i, b.qq[0].ptr, b.qq[1].ptr, b.qq[1].ptr, inter_i, rmax_i, inter_i))?;
        cu::memset_d8_async(d.y.ptr, 0, n * hidden * 4, c.stream).map_err(derr)?;
        c.launch(fx.moe_gcombine, (hidden.div_ceil(256) as u32, n as u32), 256, 0, kargs!(d.y.ptr, b.ys.ptr, b.pos.ptr, b.pwt.ptr, nul, nul, zero, n_i, k_i, hid_i))?;
    }
    Ok(())
}

/// A prompt call's message gate (`gate`'s arithmetic): the normed halves of
/// every row, `d = xn · downᵀ`, `F·d` and `G·d` as f32 GEMMs, then the mix.
#[allow(clippy::too_many_arguments)]
unsafe fn gate_big(c: &Ctx, f: &Fx, b: &Big, g: &GateD, x: u64, m: u64, dbuf: u64, out: u64, n: usize, hidden: usize, eps: f32) -> Res<()> {
    let r = g.down.bytes / (2 * hidden * 4);
    let tot = n * hidden;
    let tot_i = tot as i32;
    unsafe {
        c.launch(f.gate_norm, (n as u32, 1), 256, 0, kargs!(x, m, eps, b.gxn.ptr))?;
        c.gemm(true, false, r, n, 2 * hidden, 1.0, g.down.ptr, 2 * hidden, b.gxn.ptr, 2 * hidden, 0.0, dbuf, r, Gemm::F32)?;
        c.gemm(true, false, hidden, n, r, 1.0, g.forget.ptr, r, dbuf, r, 0.0, b.gtf.ptr, hidden, Gemm::F32)?;
        c.gemm(true, false, hidden, n, r, 1.0, g.gain.ptr, r, dbuf, r, 0.0, b.gtg.ptr, hidden, Gemm::F32)?;
        c.launch(f.gate_mix, (tot.div_ceil(256) as u32, 1), 256, 0, kargs!(x, m, b.gtf.ptr, b.gtg.ptr, out, tot_i))
    }
}

/// A message gate: `out = (1 − tanh F·d)·x + tanh G·d·m`, `d` the rank-R
/// projection of `[x/rms x, m/rms m]`.
#[allow(clippy::too_many_arguments)]
unsafe fn gate(c: &Ctx, f: &Fx, g: &GateD, x: u64, m: u64, dbuf: u64, out: u64, n: usize, hidden: usize, eps: f32) -> Res<()> {
    let r = g.down.bytes / (2 * hidden * 4);
    let r_i = r as i32;
    unsafe {
        c.launch(f.gate_down, (r as u32, n as u32), 256, 0, kargs!(x, m, g.down.ptr, r_i, eps, dbuf))?;
        c.launch(f.gate_out, (hidden.div_ceil(8) as u32, n as u32), 256, 0, kargs!(x, m, dbuf, g.forget.ptr, g.gain.ptr, r_i, out))
    }
}

/// The routed experts of the call's rows (the hybrid arm's `moe_routed` over
/// this arm's buffers): the router, the picks laid out as slots, every slot's
/// gate/up pair (Inkling's activation in its epilogue) and down, the mixture
/// into `y`.
unsafe fn moe(c: &Ctx, d: &Dev, cfg: &Cfg, l: &LayerD, n: usize) -> Res<()> {
    let fx = &d.hfx;
    let hidden = cfg.dim;
    let e = l.e;
    let k = cfg.top_k;
    let s_n = (n * k).min(e).min(d.smax);
    let (ni, ei, hi, ki, si) = (n as i32, e as i32, hidden as i32, k as i32, s_n as i32);
    // renormalized σ scores, a selection bias, the floor 1e-20
    let flags = 1 | 2 | 4 | 32;
    let ps_i = hidden as i32;
    let inter_i = l.inter as i32;
    let tb = if n > 1 && std::env::var("CMF_HYB_MOE_TB").as_deref() != Ok("1") { 2u32 } else { 1u32 };
    let (bpg, bpd) = (l.bpj_gu.div_ceil(tb), l.bpj_dn.div_ceil(tb));
    let (bg, bd) = (bpg as i32, bpd as i32);
    let nul = 0u64;
    unsafe {
        c.launch(fx.moe_router, ((e + 1).div_ceil(8) as u32, n as u32), 256, 0, kargs!(d.xm.ptr, l.router.ptr, nul, d.logits_e.ptr, d.sg.ptr, ni, ei, hi))?;
        c.launch(
            fx.moe_select,
            (1, 1),
            if n == 1 && select512_on() { 512 } else { (32 * n.max(1)) as u32 },
            0,
            kargs!(
                d.logits_e.ptr, l.bias.ptr, ni, ei, ki, flags, cfg.route_scale, l.gu_t.ptr, l.dn_t.ptr, d.gu_a.ptr, d.dn_a.ptr, d.wsel.ptr, si,
                d.qq[0].ptr, d.qq[1].ptr, ps_i, d.dns.ptr, hi, bg, bd
            ),
        )?;
        let t = (tb - 1) as usize;
        let pick = |kd: Option<usize>| match kd {
            Some(dd) => fx.mmpk[nt_of(n)][t][1][dd],
            None => fx.mmp[nt_of(n)][t][1],
        };
        let xs_i = hidden as i32;
        c.launch(pick(l.kd.0), (s_n as u32 * bpg, 1), 32 * 16, 0, kargs!(d.gu_a.ptr, si, d.pm[0].ptr, d.pm[1].ptr, d.pm[1].ptr, xs_i, ni, hi))?;
        c.launch(pick(l.kd.1), (s_n as u32 * bpd, 1), 32 * 16, 0, kargs!(d.dn_a.ptr, si, d.qq[0].ptr, d.qq[1].ptr, d.qq[1].ptr, ps_i, ni, inter_i))?;
        cu::memset_d8_async(d.y.ptr, 0, n * hidden * 4, c.stream).map_err(derr)?;
        let zero = 0i32;
        c.launch(fx.moe_combine, (hidden.div_ceil(256) as u32, n as u32), 256, 0, kargs!(d.y.ptr, d.dns.ptr, d.wsel.ptr, nul, d.sg.ptr, zero, ni, hi, si))?;
    }
    Ok(())
}
