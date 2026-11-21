//! Inkling-2-Mini on CUDA, beside the hybrid arm whose matvecs and routed
//! experts it borrows (`inkling` has the semantics).
//!
//! Every matrix goes through the arm's tensor-core matvec in its codec (q4tp
//! or q8_2f); the routed experts are the arm's slot launches (`hy_moe_router`,
//! `hy_moe_select`, a gate/up pair whose epilogue is Inkling's softcapped
//! SwiGLU, the down launch, `hy_moe_combine`); the stages between them are
//! `inkling.cu`. A call takes at most 16 rows (`NMAX`); a prompt goes 16 rows
//! at a time. Every one of the 48 layer applications keeps its own keys and
//! values (f16; a local layer's in a ring of its window, a full layer's for
//! the context cap) and its four convolution rings. A call's launches are
//! recorded once per row count and replayed (CUDA graphs).

use super::*;
use crate::inkling::{Cfg, Globals, Layer};
use std::collections::HashMap;
use std::sync::Mutex;

const SRC: &str = include_str!("../inkling.cu");
/// Block states a row holds (the embedding and the closed blocks).
const MAXB: usize = 4;

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
    _mats: Vec<Arc<DevMat>>,
}

/// One application's caches: K and V (f16 `[row][NKV·HD]`, `ring` > 0 a ring
/// of that many rows), the k, v, post-attention and post-MoE rings.
struct SlotD {
    k: Buf,
    v: Buf,
    ring: usize,
    rings: [Buf; 4],
}

struct Dev {
    fx: Fx,
    hfx: Arc<HFuncs>,
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
    smax: usize,
    /// Recorded calls by (rows, logits rows) and the pointers they hold.
    graphs: HashMap<(usize, usize), GraphRec>,
    /// Positions the full slots hold.
    cap: usize,
}
unsafe impl Send for Dev {}

static DEVICES: Mutex<Option<HashMap<u64, Dev>>> = Mutex::new(None);
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
    };
    let f4 = |n: usize| Buf::new(n * 4);
    let h2 = |n: usize| Buf::new(n * 2 + 64);
    let (qs, kvs, rs) = (nh * hd, nkv * hd, nh * dr);
    let rows = NMAX;
    let (yq, yg, yr, yk, yv) = (f4(rows * qs)?, f4(rows * qs)?, f4(rows * rs)?, f4(rows * kvs)?, f4(rows * kvs)?);
    let ya = f4(rows * hidden)?;
    let vocab = g.lm_head.rows();
    let logits = f4(rows * vocab)?;
    let e = layers.first().map_or(0, |l| l.moe.bias.len());
    let k = cfg.top_k;
    let inter = layers.first().and_then(|l| l.moe.experts.first()).map_or(0, |x| x[0].rows());
    let smax = (rows * k).min(e);
    let mut dl = Vec::with_capacity(layers.len());
    for (li, l) in layers.iter().enumerate() {
        let a = &l.attn;
        let (q, gt, rl, km, vm, om) = (mat(c, st, &a.q)?, mat(c, st, &a.gate)?, mat(c, st, &a.rel)?, mat(c, st, &a.k)?, mat(c, st, &a.v)?, mat(c, st, &a.o)?);
        let qset = mmaset(&[(&q, yq.ptr, 0, qs, qs), (&gt, yg.ptr, 0, qs, qs), (&rl, yr.ptr, 0, rs, rs)])?;
        let kset = mmaset(&[(&km, yk.ptr, 0, kvs, kvs)])?;
        let vset = mmaset(&[(&vm, yv.ptr, 0, kvs, kvs)])?;
        let oset = mmaset(&[(&om, ya.ptr, 0, hidden, hidden)])?;
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
        let ring = if layers[li].sliding { cfg.window } else { 0 };
        let rows_kv = if ring > 0 { ring } else { cap };
        let s = SlotD {
            k: h2(rows_kv * kvs)?,
            v: h2(rows_kv * kvs)?,
            ring,
            rings: [f4(3 * kvs)?, f4(3 * kvs)?, f4(3 * hidden)?, f4(3 * hidden)?],
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
    let d = Dev {
        fx,
        hfx,
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
        wsel: f4(rows * smax)?,
        gu_a: Buf::new(smax * job_b)?,
        dn_a: Buf::new(smax * job_b)?,
        qq: [Buf::new(smax * rows * hidden * 2)?, Buf::new(smax * rows * hidden * 2)?],
        dns: f4(smax * rows * hidden)?,
        y: f4(rows * hidden)?,
        fin: f4(rows * hidden)?,
        pf: [h2(rows * hidden)?, h2(rows * hidden)?],
        logits,
        cr: Buf::new(64)?,
        smax,
        graphs: HashMap::new(),
        cap,
    };
    tracing::info!(
        "inkling2_loop on CUDA: {} layers resident, {:.1} GB of weights, {:.2} GB of caches for {} positions, built in {:.1} s",
        layers.len(),
        resident as f64 / 1e9,
        d.slots.iter().map(|s| (s.k.bytes + s.v.bytes) as f64).sum::<f64>() / 1e9,
        cap,
        t0.elapsed().as_secs_f64()
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
    let fresh = !map.contains_key(&id);
    if fresh {
        // one sequence's device at a time: a new one takes the old one's place
        map.clear();
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
    let d = map.get_mut(&id).expect("device");
    if pos0 == 0 && d.zero_state(c).is_err() {
        return false;
    }
    if pos0 + ids.len() > d.cap {
        tracing::warn!("inkling2_loop on CUDA: {} positions past the cap of {}", pos0 + ids.len(), d.cap);
        return false;
    }
    match forward_(c, d, g, cfg, ids, pos0, want, all, logits) {
        Ok(()) => true,
        Err(e) => {
            tracing::error!("inkling2_loop on CUDA: {e}");
            false
        }
    }
}

/// A new sequence: its device state goes.
pub(crate) fn reset(id: u64) {
    if let Some(m) = DEVICES.lock().unwrap().as_mut() {
        m.remove(&id);
    }
    DECLINED.lock().unwrap().retain(|&x| x != id);
}

#[allow(clippy::too_many_arguments)]
fn forward_(c: &Ctx, d: &mut Dev, g: &Globals, cfg: &Cfg, ids: &[u32], pos0: usize, want: bool, all: bool, logits: &mut Vec<f32>) -> Res<()> {
    let hidden = cfg.dim;
    let vocab = g.lm_head.rows();
    if all {
        logits.clear();
    }
    let mut at = 0usize;
    while at < ids.len() {
        let n = (ids.len() - at).min(NMAX);
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
        if graphs_on() {
            if !d.graphs.contains_key(&key) {
                let dd: &Dev = d;
                let exec = unsafe { capture(c, || call(c, dd, cfg, n, lrows, vocab))? };
                d.graphs.insert(key, GraphRec { exec, sig: vec![n as u64, lrows as u64] });
            }
            unsafe { curesult(cs::cuGraphLaunch(d.graphs[&key].exec, c.stream), "graph launch")? };
        } else {
            unsafe { call(c, d, cfg, n, lrows, vocab)? };
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
    let hidden = cfg.dim;
    let hb = n * hidden * 4;
    let ost = (NMAX * hidden) as i32;
    let bst = (MAXB * hidden) as i32;
    let mut cur = 0usize;
    let mut hin = d.h[0].ptr;
    let mut nb = 0usize;
    let mut npre = 0usize;
    for (li, pass) in cfg.plan() {
        let l = &d.layers[li];
        let s = &d.slots[cfg.slot(li, pass)];
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
        unsafe { app(c, d, cfg, l, s, li, pass, n, hin, hout, &mut nb, npre, bank_nb)? };
        cur = 1 - cur;
        hin = hout;
    }
    // the final mixture, its norm and the head
    let f = &d.fx;
    unsafe {
        let nb_i = nb as i32;
        c.launch(f.mix, (1, n as u32), 256, 0, kargs!(d.blk.ptr, nb_i, bst, hin, d.final_q.ptr, d.fin.ptr, ost))?;
        if lrows > 0 {
            let r0 = n - lrows;
            let fin = d.fin.ptr + (r0 * hidden * 4) as u64;
            let (nul, hs) = (0u64, hidden as i32);
            c.launch(f.rms, (lrows as u32, 1), 256, 0, kargs!(fin, d.final_norm.ptr, cfg.eps as f32, nul, d.pf[0].ptr, d.pf[1].ptr, hs))?;
            let pl = Planes { p0: d.pf[0].ptr, p1: d.pf[1].ptr, p2: d.pf[1].ptr, stride: hidden };
            d.head.run(c, &d.hfx, &pl, hidden, lrows)?;
            let (v_i, cap) = (vocab as i32, cfg.logit_cap);
            if cap > 0.0 {
                c.launch(f.softcap, (vocab.div_ceil(256) as u32, lrows as u32), 256, 0, kargs!(d.logits.ptr, v_i, cap))?;
            }
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
    let rows = NMAX;
    let ost = (rows * hidden) as i32;
    let bst = (MAXB * hidden) as i32;
    let (ni, nu) = (n as i32, n as u32);
    let eps = cfg.eps as f32;
    let hs = hidden as i32;
    let fl = |b: &Buf, i: usize| b.ptr + (i * rows * hidden * 4) as u64;
    unsafe {
        // 1. the attention inputs
        let mut inp = [hin, hin, hin];
        if let Some((aq, nq)) = &l.aq {
            let nb_i = *nb as i32;
            c.launch(f.mix, (*nq as u32, nu), 256, 0, kargs!(d.blk.ptr, nb_i, bst, hin, aq.ptr, d.in3.ptr, ost))?;
            inp = if *nq == 3 { [fl(&d.in3, 0), fl(&d.in3, 1), fl(&d.in3, 2)] } else { [fl(&d.in3, 0); 3] };
        }
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
        // 3. the input norms (a local layer's three are one weight; without
        // messages its inputs are one row)
        let one = l.sliding && gates.is_none();
        for j in 0..(if one { 1 } else { 3 }) {
            c.launch(f.rms, (nu, 1), 256, 0, kargs!(inp[j], l.norms[j].ptr, eps, d.x[j].ptr, d.xp[j][0].ptr, d.xp[j][1].ptr, hs))?;
        }
        let (xk, xv) = if one { (0, 0) } else { (1, 2) };
        let pl = |j: usize| Planes { p0: d.xp[j][0].ptr, p1: d.xp[j][1].ptr, p2: d.xp[j][1].ptr, stride: hidden };
        // 4. the projections
        l.qset.run(c, &d.hfx, &pl(0), hidden, n)?;
        l.kset.run(c, &d.hfx, &pl(xk), hidden, n)?;
        l.vset.run(c, &d.hfx, &pl(xv), hidden, n)?;
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
            kargs!(d.yk.ptr, d.yv.ptr, dk, dv, l.tk.ptr, l.tv.ptr, s.rings[0].ptr, s.rings[1].ptr, s.k.ptr, s.v.ptr, ring, ni, cfg.qk_clip, eps, d.cr.ptr),
        )?;
        // 6. the attention, its gate, the O projection
        let (ext, scale) = (l.extent as i32, 1.0f32 / (cfg.hd as f32).sqrt());
        c.launch(f.attn, (nh as u32, nu), 256, 0, kargs!(d.yq.ptr, d.yr.ptr, d.sk.ptr, l.rel_table.ptr, ext, s.k.ptr, s.v.ptr, ring, scale, d.att.ptr, d.cr.ptr))?;
        let qs = nh * cfg.hd;
        let qs_i = qs as i32;
        c.launch(f.ogate, (qs.div_ceil(256) as u32, nu), 256, 0, kargs!(d.att.ptr, d.yg.ptr, d.po[0].ptr, d.po[1].ptr, qs_i))?;
        let po = Planes { p0: d.po[0].ptr, p1: d.po[1].ptr, p2: d.po[1].ptr, stride: qs };
        l.oset.run(c, &d.hfx, &po, qs, n)?;
        // 7. the output convolution; a boundary closes the running sum into a block
        let bslot = if l.boundary { *nb as i32 } else { -1 };
        let hb = hidden.div_ceil(256) as u32;
        c.launch(f.post, (hb, 1), 256, 0, kargs!(d.ya.ptr, l.tpa.ptr, s.rings[2].ptr, hin, d.blk.ptr, bslot, bst, d.r.ptr, ni))?;
        if l.boundary {
            *nb += 1;
        }
        // 8. the MoE input
        let nb_i = *nb as i32;
        c.launch(f.mix, (1, nu), 256, 0, kargs!(d.blk.ptr, nb_i, bst, d.r.ptr, l.mq.ptr, d.m.ptr, ost))?;
        let mut mi = d.m.ptr;
        if let Some(gs) = gates {
            gate(c, f, &gs[3], mi, fl(&d.msg, 3), d.gd.ptr, d.mg.ptr, n, hidden, eps)?;
            mi = d.mg.ptr;
        }
        c.launch(f.rms, (nu, 1), 256, 0, kargs!(mi, l.post_norm.ptr, eps, d.xm.ptr, d.pm[0].ptr, d.pm[1].ptr, hs))?;
        // 9. the routed experts into y
        moe(c, d, cfg, l, n)?;
        // 10. the output convolution onto the running sum
        let (nul, m1) = (0u64, -1i32);
        c.launch(f.post, (hb, 1), 256, 0, kargs!(d.y.ptr, l.tpm.ptr, s.rings[3].ptr, d.r.ptr, nul, m1, bst, hout, ni))?;
    }
    Ok(())
}

/// A message gate: `out = (1 − tanh F·d)·x + tanh G·d·m`, `d` the rank-R
/// projection of `[x/rms x, m/rms m]`.
#[allow(clippy::too_many_arguments)]
unsafe fn gate(c: &Ctx, f: &Fx, g: &GateD, x: u64, m: u64, dbuf: u64, out: u64, n: usize, hidden: usize, eps: f32) -> Res<()> {
    let r = g.down.bytes / (2 * hidden * 4);
    let r_i = r as i32;
    unsafe {
        c.launch(f.gate_down, (r as u32, n as u32), 256, 0, kargs!(x, m, g.down.ptr, r_i, eps, dbuf))?;
        c.launch(f.gate_out, (hidden.div_ceil(256) as u32, n as u32), 256, 0, kargs!(x, m, dbuf, g.forget.ptr, g.gain.ptr, r_i, out))
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
