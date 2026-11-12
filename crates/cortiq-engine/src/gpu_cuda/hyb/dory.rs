//! Nemotron Labs Dory on CUDA, the twin of `gpu_metal/dory.rs` (`dory` has
//! the semantics), beside the hybrid arm whose kernels it borrows.
//!
//! Every four-bit matrix goes through the arm's tensor-core matvec, repacked
//! on the card; the stages between them are `dory.cu`. K and V rows are f16:
//! a sliding layer's in a ring of the window and a turn's rewind, a full
//! layer's at their position. A turbo file's router decides after a token's
//! first pass whether it takes a second (`Gate`), and at a prompt's end its
//! full layers seal (`seal`): the rows the far field would lose are chosen
//! and kept exact, the rest goes into landmarks (the arm's far-field kernels
//! and its `o1_ensure`), and a token's cost stops growing with the text.
//!
//! A long prompt keeps a fixed ceiling on the card: its first rows are read
//! exact (`Dev::prefix`, 8192 on a card with room), then sealed, and the rest
//! is read past the seal in chunks on the GEMMs, each row after the far
//! field's near rows (the hybrid arm's sealed prompt kernels). The rows leave
//! the card for an archive on the host as they are read, and before the
//! prompt's last rows the far field is built again over all of them, with the
//! rows it would lose kept exact (`rebuild`).

use super::*;
use crate::dory::{Dory, ExitRate, Gate, KvMode, Layer, QT_STRIDE, Want, top_prob};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

const SRC: &str = include_str!("../dory.cu");
/// Rows of a call: the tensor-core kernel takes 16.
const CHUNK: usize = NMAX;
/// Rows of a prompt chunk on the GEMM path: the weights expand to f16 once
/// for them.
const BIG: usize = 512;
/// A prompt's last rows that take every pass (the wgpu path's chunk).
const LAST_CHUNK: usize = 64;
/// Key blocks a row's attention may split into (`attn_rb` keeps it under 130),
/// and one more for a far field's.
const NBCAP: usize = 192;
/// A prompt read whole (`CMF_DORY_BIG=0`, or every row's logits wanted) seals
/// before it ends at this length: its full layers' rows are on the card until
/// the seal, 36 KB a row for the nine of Dory 1.0.
const SEAL_MAX: usize = 131072;
/// Rows of a long prompt read exact before its first seal, on a card with
/// room (the wgpu path's prefix; `CMF_O1_PREFIX_MAX`, and less on a small
/// card: `o1_seal_for_card_rows`). Past them the rows leave the card.
const PREFIX_MAX: usize = 8192;
/// Rows of a sealed prompt the archive holds: a longer one has its exact rows
/// chosen there, and the rows after it go to the window and the far field.
const ARCH_MAX: usize = 131072;

struct Fx {
    qk: cs::CUfunction,
    silu: cs::CUfunction,
    slerp: cs::CUfunction,
    mix: cs::CUfunction,
    softmax_w: cs::CUfunction,
    attn: cs::CUfunction,
    finish_m: cs::CUfunction,
}

struct AttnD {
    qkvg: MmaSet,
    o: MmaSet,
    /// q, k, v, g, o for the prompt's GEMMs.
    mats: [Arc<DevMat>; 5],
    sq: Buf,
    sg: Buf,
    alpha: Buf,
    sliding: bool,
    ordinal: usize,
}

struct MlpD {
    gu: MmaSet,
    down: MmaSet,
    /// gate and up as one pair with their scales in the ladders: the matvec
    /// writes silu(gate) * up as down's planes, no activation kernel
    fused: bool,
    /// gate, up, down for the prompt's GEMMs.
    mats: [Arc<DevMat>; 3],
    sa: Buf,
    sb: Buf,
    alpha: Buf,
}

enum Ld {
    Attn(Box<AttnD>),
    Mlp(Box<MlpD>),
}

/// One layer's (and loop's) K/V rows, f16 `[row][kv head][hd]`; `ring` > 0:
/// a ring of that many rows by position.
struct Slot {
    k: Buf,
    v: Buf,
    cap: usize,
    ring: usize,
}

impl Slot {
    fn new(cap: usize, ring: usize, kv_row: usize) -> Res<Self> {
        Ok(Self { k: Buf::new(cap * kv_row * 2)?, v: Buf::new(cap * kv_row * 2)?, cap, ring })
    }
    fn bytes(&self) -> u64 {
        (self.k.bytes + self.v.bytes) as u64
    }
}

struct Dev {
    id: u64,
    fx: Fx,
    hfx: Arc<HFuncs>,
    layers: Vec<Ld>,
    head: MmaSet,
    slots: Vec<Vec<Slot>>,
    h: Buf,
    input: Buf,
    state: Buf,
    /// h's planes (the hidden-wide matvecs read them).
    p: [Buf; 2],
    /// The attention output's planes (the output projection), the MLP
    /// middle's (down).
    zp: [Buf; 2],
    mp: [Buf; 2],
    /// The fused gate/up pairs the job tables point into.
    #[allow(dead_code)]
    pairs: Vec<Arc<DevMat>>,
    yq: Buf,
    yk: Buf,
    yv: Buf,
    yg: Buf,
    qg: Buf,
    gate: Buf,
    o: Buf,
    fg: Buf,
    fu: Buf,
    logits: Buf,
    rope: [Buf; 2],
    s: Buf,
    s_keys: usize,
    mx: Buf,
    pp: Buf,
    ee: Buf,
    /// A sealed layer's call record.
    ci: Buf,
    /// The call's record the kernels read their positions from: [0] pos0.
    cr: Buf,
    /// Recorded decode stages: (stage, loops, want) to the graph and the
    /// pointers it was recorded with.
    graphs: HashMap<(u8, usize, u8), GraphRec>,
    /// The prompt's GEMM path, made on its first chunk.
    big: Option<Big>,
    gate_tokens: u64,
    gate_one: u64,
    exits: ExitRate,
    sealed: bool,
    o1_refused: bool,
    seal_epoch: u64,
    /// The queries of a full slot, one row in QT_STRIDE, until the seal.
    qtrace: HashMap<(usize, usize), (Buf, usize)>,
    /// The sealed slots (their far fields are the arm's, keyed (id, ordinal
    /// * 16 + slot)).
    far: Vec<(usize, usize)>,
    /// A long prompt's rows past its first seal, until its far field is
    /// built again over all of them.
    arch: Option<Archive>,
    /// Rows a long prompt reads exact before it seals on this card.
    prefix: usize,
    /// The rows a prompt ends at (0 for a decode step): a full slot grows to
    /// them and no further while the prompt is read.
    rows_hint: usize,
}
unsafe impl Send for Dev {}

/// One full slot's rows past the first seal, off the card: keys and values
/// `[row][kv head][hd]`, queries one row in QT_STRIDE `[row][nh][hd]`.
struct ArchSlot {
    ord: usize,
    sl: usize,
    k: crate::pipeline::ArchStore,
    v: crate::pipeline::ArchStore,
    q: crate::pipeline::ArchStore,
}

/// A long prompt's rows from position 0 (in RAM, or in files under
/// `CMF_O1_SPILL`), and the card's staging for one chunk of every slot: K
/// and V (f16, BIG rows each) then the kept queries (f32).
struct Archive {
    slots: Vec<ArchSlot>,
    rows: usize,
    staging: Buf,
    /// Bytes of one slot's part of the staging.
    per: usize,
}

/// A sealed slot's far field as a prompt chunk reads it: the buffers (near K,
/// near V, key landmarks, query landmarks, inverse, maxima and masses,
/// numerators), the geometry and the window's cursor, moved by `commit`.
#[derive(Clone, Copy)]
struct FarCur {
    p: [u64; 7],
    ns: usize,
    w: usize,
    m: usize,
    scale: f32,
    rect: bool,
    win_len: usize,
    win_head: usize,
    far_len: usize,
}

impl FarCur {
    fn of(o: &HO1) -> Self {
        Self {
            p: [o.nk.ptr, o.nv.ptr, o.kt.ptr, o.qt.ptr, o.mu.ptr, o.mz.ptr, o.th.ptr],
            ns: o.ns,
            w: o.w,
            m: o.m,
            scale: o.scale,
            rect: o.rect_fm,
            win_len: o.win_len,
            win_head: o.win_head,
            far_len: o.far_len,
        }
    }

    fn store(&self, o: &mut HO1) {
        o.win_len = self.win_len;
        o.win_head = self.win_head;
        o.far_len = self.far_len;
    }

    /// The mirror's rows before a chunk: the sinks with the kept rows, then
    /// the window (unwrapped until it is full).
    fn near(&self) -> usize {
        self.ns + self.win_len
    }

    /// The chunk of `b` rows written after the near rows goes into the
    /// window: the first fill its room where they are; past a full window
    /// each takes the oldest's slot, which goes into the far sums first (the
    /// hybrid arm's prompt commit).
    unsafe fn commit(&mut self, c: &Ctx, hfx: &HFuncs, b: usize, nk: usize, kv_row: usize) -> Res<()> {
        let (n0, w) = (self.near(), self.w);
        if b > w {
            return Err(format!("a chunk of {b} rows over a window of {w}"));
        }
        let fill = b.min(w - self.win_len);
        let rest = b - fill;
        let head0 = if self.win_len == w { self.win_head } else { 0 };
        if rest > 0 {
            let [kp, vp, _, qt, _, mz, th] = self.p;
            let (m_i, nk_i, ns_i, w_i, h0_i, e_i) = (self.m as i32, nk as i32, self.ns as i32, w as i32, head0 as i32, rest as i32);
            let warps = nk * 4 * self.m;
            let row_b = kv_row * 2;
            unsafe {
                c.launch(hfx.far_insert_n, ((warps * 32).div_ceil(256) as u32, 1), 256, 0, kargs!(kp, vp, qt, mz, th, m_i, nk_i, self.scale, ns_i, w_i, h0_i, e_i))?;
                // the rows past the window's room into the freed slots: two runs at most
                let src = n0 + fill;
                let r1 = rest.min(w - head0);
                for (dst, from, n) in [(self.ns + head0, src, r1), (self.ns, src + r1, rest - r1)] {
                    if n == 0 {
                        continue;
                    }
                    cu::memcpy_dtod_async(kp + (dst * row_b) as u64, kp + (from * row_b) as u64, n * row_b, c.stream).map_err(derr)?;
                    cu::memcpy_dtod_async(vp + (dst * row_b) as u64, vp + (from * row_b) as u64, n * row_b, c.stream).map_err(derr)?;
                }
            }
            self.win_head = (head0 + rest) % w;
            self.far_len += rest;
        }
        self.win_len = (self.win_len + b).min(w);
        Ok(())
    }
}

static DEVICES: Mutex<Option<HashMap<u64, Dev>>> = Mutex::new(None);
/// Sequences whose build on the card failed: the host, which has read their
/// rows from the start, runs the rest of each until `reset` starts a new one.
/// A build that came through on a later call would run on rows the card
/// never read.
static DECLINED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// A sealed slot's far field for the launches: its buffers (near K, near V,
/// key landmarks, query landmarks, inverse, maxima and masses, numerators),
/// landmarks, scale and rectification.
type FarDev = ([u64; 7], usize, f32, bool);

/// A full slot's rows on the host for its seal: ordinal, slot, keys, values
/// and the queries kept one in QT_STRIDE.
type SlotRows = (usize, usize, Vec<f32>, Vec<f32>, Vec<f32>);

fn gw_of(t: &crate::qtensor::QTensor) -> Option<(Arc<CmfModel>, GraphW<'static>)> {
    let (model, idx) = t.q4tp_mapped()?;
    Some((model.clone(), GraphW { idx, kind: 6, row_scale: &[], data: &[], prism: crate::gpu::GraphPrismOp::None, affine: false }))
}

fn mat(c: &Ctx, st: &mut State, t: &crate::qtensor::QTensor) -> Res<Arc<DevMat>> {
    let (model, gw) = gw_of(t).ok_or("a Dory matrix that is not q4tp")?;
    hmat(c, st, &model, &gw, t.rows(), t.cols())
}

/// An MLP's gate and up as the arm's pair (16-row tiles of 8 gate rows then
/// the same 8 up rows), each row's ladder shifted by log2 of its scale: the
/// pair's epilogue then makes silu(g * sa) * (u * sb). None unless every
/// scale is positive.
fn scaled_pair(c: &Ctx, st: &mut State, fx: &HFuncs, g: &crate::qtensor::QTensor, u: &crate::qtensor::QTensor, sa: &[f32], sb: &[f32]) -> Res<Option<Arc<DevMat>>> {
    use cortiq_core::quant::{f16_to_f32, q4tp_sections};
    if sa.iter().chain(sb).any(|&x| !(x > 0.0 && x.is_finite())) {
        return Ok(None);
    }
    let ((mg, ig), (mu, iu)) = (g.q4tp_mapped().ok_or("gate not q4tp")?, u.q4tp_mapped().ok_or("up not q4tp")?);
    // the arm's pair cache, under a key of its own (the scales are in it)
    let key = (mg.uid() ^ 0xD0A7_5CA1_ED00_0000, ig, iu);
    if let Some(m) = st.hyb.pairs.get(&key) {
        return Ok(Some(m.clone()));
    }
    let (inter, cols) = (g.rows(), g.cols());
    if u.rows() != inter || u.cols() != cols || inter % 8 != 0 || sa.len() != inter || sb.len() != inter {
        return Ok(None);
    }
    let (gb, ub) = (mg.entry_bytes(&mg.tensors[ig]), mu.entry_bytes(&mu.tensors[iu]));
    let rows = 2 * inter;
    let (w, aux0, _) = repack_on_card(c, fx, &[gb, ub], inter, rows, cols)?;
    let (params_off, _, _) = q4tp_sections(inter, cols);
    let mut lad = vec![0f32; rows * 2];
    for r in 0..rows {
        let (src, sr) = (((r & 15) >= 8) as usize, (r >> 4) * 8 + (r & 7));
        let (b, sc) = if src == 0 { (gb, sa[sr]) } else { (ub, sb[sr]) };
        let p = &b[params_off + sr * 4..params_off + sr * 4 + 4];
        lad[r * 2] = f16_to_f32(u16::from_le_bytes([p[0], p[1]])) + sc.log2();
        lad[r * 2 + 1] = f16_to_f32(u16::from_le_bytes([p[2], p[3]]));
    }
    let held = (gb.len() + ub.len()) as u64;
    c.resident.fetch_add(held, Ordering::Relaxed);
    let m = Arc::new(DevMat { kind: 6, rows, cols, w: Arc::new(w), aux0: Some(aux0), aux1: Some(Buf::from_slice(&lad)?), held });
    st.hyb.pairs.insert(key, m.clone());
    Ok(Some(m))
}

/// Can the card run this file? Decided before anything moves.
pub(crate) fn admits(m: &Dory) -> bool {
    let c = &m.cfg;
    if c.nh != c.nk * 4 || c.hd != 256 || !c.hidden.is_multiple_of(64) || !c.ffn.is_multiple_of(64) || c.hidden > 8192 {
        return false;
    }
    let q4 = |t: &crate::qtensor::QTensor| t.q4tp_mapped().is_some();
    q4(&m.head)
        && m.layers.iter().all(|l| match l {
            Layer::Attn(w) => q4(&w.q) && q4(&w.k) && q4(&w.v) && q4(&w.g) && q4(&w.o),
            Layer::Mlp(w) => q4(&w.gate) && q4(&w.up) && q4(&w.down),
        })
}

fn build(c: &Ctx, st: &mut State, m: &Dory) -> Res<Dev> {
    let cf = &m.cfg;
    let t0 = std::time::Instant::now();
    let (hidden, ffn, nh, nk, hd, vocab) = (cf.hidden, cf.ffn, cf.nh, cf.nk, cf.hd, cf.vocab);
    let hfx = match st.hyb.funcs.get(&(4, hd)) {
        Some(f) => f.clone(),
        None => {
            let f = Arc::new(load_funcs(c, 4, hd)?);
            st.hyb.funcs.insert((4, hd), f.clone());
            f
        }
    };
    let module = compile(SRC, vec!["-DHPK=4".into(), format!("-DHD={hd}")], c.arch)?;
    let fx = Fx {
        qk: func(module, "dk_qk")?,
        silu: func(module, "dk_silu")?,
        slerp: func(module, "dk_slerp")?,
        mix: func(module, "dk_mix")?,
        softmax_w: func(module, "dk_softmax_w")?,
        attn: func(module, "dk_attn")?,
        finish_m: func(module, "dk_finish_m")?,
    };
    let (qs, kvs) = (nh * hd, nk * hd);
    let f4 = |n: usize| Buf::new(n * 4);
    let h2 = |n: usize| Buf::new(n * 2);
    let (yq, yk, yv, yg) = (f4(BIG * qs)?, f4(BIG * kvs)?, f4(BIG * kvs)?, f4(BIG * qs)?);
    let (o, fg, fu) = (f4(BIG * hidden)?, f4(BIG * ffn)?, f4(BIG * ffn)?);
    let logits = f4(CHUNK * vocab)?;
    let mp = [h2(CHUNK * ffn)?, h2(CHUNK * ffn)?];
    let fuse = std::env::var("CMF_DORY_FUSE").as_deref() != Ok("0");
    // the pairs are the Dory arm's own: held here for their job tables
    let mut pairs: Vec<Arc<DevMat>> = Vec::new();
    let mut layers = Vec::with_capacity(m.layers.len());
    for l in &m.layers {
        layers.push(match l {
            Layer::Attn(w) => {
                let (q, k, v, g, om) = (mat(c, st, &w.q)?, mat(c, st, &w.k)?, mat(c, st, &w.v)?, mat(c, st, &w.g)?, mat(c, st, &w.o)?);
                Ld::Attn(Box::new(AttnD {
                    qkvg: mmaset(&[(&q, yq.ptr, 0, qs, qs), (&k, yk.ptr, 0, kvs, kvs), (&v, yv.ptr, 0, kvs, kvs), (&g, yg.ptr, 0, qs, qs)])?,
                    o: mmaset(&[(&om, o.ptr, 0, hidden, hidden)])?,
                    mats: [q.clone(), k.clone(), v.clone(), g.clone(), om.clone()],
                    sq: Buf::from_slice(&w.sq)?,
                    sg: Buf::from_slice(&w.sg)?,
                    alpha: Buf::from_slice(&w.alpha)?,
                    sliding: w.sliding,
                    ordinal: w.ordinal,
                }))
            }
            Layer::Mlp(w) => {
                let (g, u, d) = (mat(c, st, &w.gate)?, mat(c, st, &w.up)?, mat(c, st, &w.down)?);
                let pair = if fuse { scaled_pair(c, st, &hfx, &w.gate, &w.up, &w.sg, &w.su)? } else { None };
                let fused = pair.is_some();
                let gu = match &pair {
                    Some(pm) => mmaset_pair(pm, mp[0].ptr, mp[1].ptr, ffn)?,
                    None => mmaset(&[(&g, fg.ptr, 0, ffn, ffn), (&u, fu.ptr, 0, ffn, ffn)])?,
                };
                pairs.extend(pair);
                Ld::Mlp(Box::new(MlpD {
                    gu,
                    fused,
                    down: mmaset(&[(&d, o.ptr, 0, hidden, hidden)])?,
                    mats: [g.clone(), u.clone(), d.clone()],
                    sa: Buf::from_slice(&w.sg)?,
                    sb: Buf::from_slice(&w.su)?,
                    alpha: Buf::from_slice(&w.alpha)?,
                }))
            }
        });
    }
    let hm = mat(c, st, &m.head)?;
    let head = mmaset(&[(&hm, logits.ptr, 0, vocab, vocab)])?;
    // Dory's matrices are narrow (2560 to 7168 columns): one tile a block and
    // eight warps a tile do best on a one-row call (3090: 189 against 186
    // tok/s with the arm's choice)
    for l in layers.iter_mut() {
        let sets: Vec<&mut MmaSet> = match l {
            Ld::Attn(a) => vec![&mut a.qkvg, &mut a.o],
            Ld::Mlp(a) => vec![&mut a.gu, &mut a.down],
        };
        for s in sets {
            s.tb1_n1 = true;
            s.ks16 = false;
        }
    }
    tracing::info!("dory on CUDA: {} sub-layers resident, {:.1} GB of weights, built in {:.1} s", layers.len(), c.resident.load(Ordering::Relaxed) as f64 / 1e9, t0.elapsed().as_secs_f64());
    // a prompt row's bytes on the card before the seal: every full slot's K
    // and V in f16 and its query trace
    let full_slots: usize = m
        .layers
        .iter()
        .enumerate()
        .map(|(li, l)| match l {
            Layer::Attn(w) if !w.sliding => {
                if cf.kv == KvMode::PerLoop && li >= cf.n_in && li < cf.n_in + cf.n_rec { cf.loops.max(1) } else { 1 }
            }
            _ => 0,
        })
        .sum();
    let row_bytes = (full_slots * (kvs * 4 + qs * 4 / QT_STRIDE)) as u64;
    let prefix = crate::pipeline::o1_seal_for_card_rows(PREFIX_MAX, c.resident.load(Ordering::Relaxed), row_bytes);
    Ok(Dev {
        id: 0,
        fx,
        hfx,
        layers,
        head,
        slots: (0..m.attn_layers).map(|_| Vec::new()).collect(),
        h: f4(BIG * hidden)?,
        input: f4(BIG * hidden)?,
        state: f4(BIG * hidden)?,
        p: [h2(BIG * hidden)?, h2(BIG * hidden)?],
        zp: [h2(CHUNK * qs)?, h2(CHUNK * qs)?],
        mp,
        pairs,
        yq,
        yk,
        yv,
        yg,
        qg: f4(BIG * qs)?,
        gate: f4(BIG * qs)?,
        o,
        fg,
        fu,
        logits,
        rope: [f4(BIG * hd)?, f4(BIG * hd)?],
        s: f4(64)?,
        s_keys: 0,
        mx: f4(CHUNK * nh * NBCAP)?,
        pp: f4(CHUNK * NBCAP * qs)?,
        ee: f4(CHUNK * NBCAP * nh)?,
        ci: Buf::new(64)?,
        cr: Buf::new(64)?,
        graphs: HashMap::new(),
        big: None,
        gate_tokens: 0,
        gate_one: 0,
        exits: ExitRate::default(),
        sealed: false,
        o1_refused: false,
        seal_epoch: 0,
        qtrace: HashMap::new(),
        far: Vec::new(),
        arch: None,
        prefix,
        rows_hint: 0,
    })
}

impl Dev {
    fn planes(&self) -> Planes {
        Planes { p0: self.p[0].ptr, p1: self.p[1].ptr, p2: self.p[1].ptr, stride: 0 }
    }

    /// Room for `need` rows in a slot; a full layer's grows, keeping its rows.
    fn ensure_slot(&mut self, c: &Ctx, ord: usize, sl: usize, need: usize, ring: usize, kv_row: usize) -> Res<()> {
        let row = &mut self.slots[ord];
        while row.len() <= sl {
            let s = Slot::new(if ring > 0 { ring } else { need.max(4096) }, ring, kv_row)?;
            c.kv_bytes.fetch_add(s.bytes(), Ordering::Relaxed);
            row.push(s);
        }
        let s = &mut row[sl];
        if s.cap >= need || s.ring > 0 {
            return Ok(());
        }
        // doubling, but no further than the prompt's end while one is read
        let hint = self.rows_hint;
        let cap = need.max(s.cap * 2);
        let cap = if hint >= need { cap.min(hint) } else { cap };
        let n = Slot::new(cap, 0, kv_row)?;
        let old = s.cap * kv_row * 2;
        unsafe {
            cu::memcpy_dtod_async(n.k.ptr, s.k.ptr, old, c.stream).map_err(derr)?;
            cu::memcpy_dtod_async(n.v.ptr, s.v.ptr, old, c.stream).map_err(derr)?;
        }
        c.sync()?;
        c.kv_bytes.fetch_add(n.bytes(), Ordering::Relaxed);
        c.kv_bytes.fetch_sub(s.bytes(), Ordering::Relaxed);
        *s = n;
        Ok(())
    }

    /// Scores for `keys` keys a row.
    fn ensure_scores(&mut self, c: &Ctx, nh: usize, keys: usize) -> Res<()> {
        if keys <= self.s_keys {
            return Ok(());
        }
        let cap = keys.next_multiple_of(4096).max(self.s_keys * 2);
        c.sync()?;
        self.s = Buf::new(CHUNK * nh * cap * 4)?;
        self.s_keys = cap;
        Ok(())
    }
}

/// `true` when this forward ran on the card.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(id: u64, m: &Dory, ids: &[u32], pos0: usize, loops: usize, want: Want, logits: &mut Vec<f32>) -> bool {
    let Some(c) = super::super::ctx() else { return false };
    if !admits(m) || std::env::var("CMF_DORY_CUDA").as_deref() == Ok("0") {
        return false;
    }
    if DECLINED.lock().unwrap().contains(&id) {
        return false;
    }
    let mut guard = DEVICES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    if c.current().is_err() {
        return false;
    }
    let d = match map.entry(id) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => {
            let built = {
                let mut st = c.st.lock().unwrap();
                build(c, &mut st, m)
            };
            match built {
                Ok(mut d) => {
                    d.id = id;
                    e.insert(d)
                }
                Err(e) => {
                    tracing::warn!("dory on CUDA declined: {e}; the host runs this sequence");
                    DECLINED.lock().unwrap().push(id);
                    return false;
                }
            }
        }
    };
    match forward_(c, d, m, ids, pos0, loops, want, logits) {
        Ok(()) => true,
        Err(e) => {
            tracing::error!("dory on CUDA: {e}");
            false
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_(c: &Ctx, d: &mut Dev, m: &Dory, ids: &[u32], pos0: usize, loops: usize, want: Want, logits: &mut Vec<f32>) -> Res<()> {
    let mut at = 0usize;
    let big_on = std::env::var("CMF_DORY_BIG").as_deref() != Ok("0");
    let n = ids.len();
    let end = pos0 + n;
    // the prompt's last rows take every pass, as the last 64-row chunk of the
    // wgpu path does: the rows before them by the GEMMs, BIG at a time
    let tail = (n - 1) % LAST_CHUNK + 1;
    // a prompt shorter than this stays exact; a longer one seals at its end,
    // or after its first `seal_at` rows when they are not the most of it
    let seal_at = m.o1.map_or(usize::MAX, |o| o.seal.min(d.prefix));
    // A long prompt keeps the card at its prefix's memory: the prefix exact,
    // a seal, the rest past it, and the far field built again with its exact
    // rows before the last EXACT_TAIL rows (they read the chosen ones).
    let staged = m.o1.is_some()
        && far_chunks_on()
        && big_on
        && want != Want::All
        && !d.sealed
        && !d.o1_refused
        && pos0 < seal_at
        && end > seal_at + crate::dory::EXACT_TAIL;
    // a full slot grows to the rows it will hold before the seal and no
    // further (a prompt that stays exact doubles it, for its decode)
    d.rows_hint = match () {
        _ if n <= 1 => 0,
        _ if staged => seal_at,
        _ if end >= seal_at => end,
        _ => 0,
    };
    let rebuild_at = end.saturating_sub(crate::dory::EXACT_TAIL).min(ARCH_MAX);
    while at < n {
        let left = n - at;
        let pos = pos0 + at;
        // the rows before the tail by the GEMMs, BIG at a time, one pass on a
        // turbo file (exact, or past the seal after the near rows); the tail
        // (and a decode step) by the tensor-core kernel, 16 rows at a time,
        // every one with every pass
        let gemm = big_on && want != Want::All && left > tail;
        if d.arch.is_some()
            && (!gemm || pos >= rebuild_at)
            && let Some(o1) = m.o1
        {
            rebuild(c, d, m, o1, pos);
        }
        let (k, w, lp) = if gemm {
            let mut k = (left - tail).min(BIG);
            if staged && !d.sealed {
                k = k.min(seal_at - pos);
            }
            if d.arch.is_some() {
                k = k.min(rebuild_at - pos);
            }
            (k, Want::None, m.chunk_loops(loops, Want::None))
        } else {
            let k = left.min(CHUNK);
            let last = at + k == n;
            let w = if last || want == Want::All { want } else { Want::None };
            // a turbo prompt's tail takes every pass, as on the wgpu path
            (k, w, if want == Want::All || !big_on { m.chunk_loops(loops, w) } else { loops })
        };
        if gemm {
            run_big(c, d, m, &ids[at..at + k], pos, lp, w, logits)?;
        } else {
            run(c, d, m, &ids[at..at + k], pos, lp, w, logits)?;
        }
        at += k;
        if let (Some(o1), false, false) = (m.o1, d.sealed, d.o1_refused) {
            let rows = pos0 + at;
            // At a prompt's end its rows go exact first; a long one seals
            // after its prefix (or, read whole, before its rows outgrow the
            // card). A refused seal only costs the layers their far field:
            // they stay exact.
            let early = staged && rows == seal_at;
            let due = early || at == n || rows >= SEAL_MAX;
            if due && rows >= seal_at {
                match seal(c, d, m, o1, rows, early) {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::warn!("dory on CUDA: no far field ({e}), the full layers stay exact");
                        d.o1_refused = true;
                        d.qtrace.clear();
                    }
                }
            }
        }
    }
    if d.arch.is_some()
        && let Some(o1) = m.o1
    {
        rebuild(c, d, m, o1, end);
    }
    Ok(())
}

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

/// The slots a call of `k` rows at `pos` writes, and the queries a seal will want.
fn prepare_slots(c: &Ctx, d: &mut Dev, m: &Dory, pos: usize, k: usize, loops: usize) -> Res<()> {
    let cf = &m.cfg;
    let (nh, nk, hd) = (cf.nh, cf.nk, cf.hd);
    let kv_row = nk * hd;
    let per_loop = cf.kv == KvMode::PerLoop;
    let sl_ring = cf.window + 64 + crate::dory::REWIND;
    let full_ring = cf.full_window.map(|w| w + 64 + crate::dory::REWIND);
    let attn: Vec<(usize, usize, bool)> = d
        .layers
        .iter()
        .enumerate()
        .filter_map(|(li, l)| match l {
            Ld::Attn(a) => Some((li, a.ordinal, a.sliding)),
            _ => None,
        })
        .collect();
    for &(li, ord, sliding) in &attn {
        let rec = li >= cf.n_in && li < cf.n_in + cf.n_rec;
        let n = if rec && per_loop { loops } else { 1 };
        for sl in 0..n {
            if !sliding && d.sealed {
                continue;
            }
            if sliding {
                d.ensure_slot(c, ord, sl, sl_ring, sl_ring, kv_row)?;
            } else if let Some(r) = full_ring {
                d.ensure_slot(c, ord, sl, r, r, kv_row)?;
            } else {
                d.ensure_slot(c, ord, sl, pos + k, 0, kv_row)?;
                if m.o1.is_some() && !d.o1_refused {
                    let need = (pos + k).div_ceil(QT_STRIDE) * nh * hd;
                    let hint = d.rows_hint.div_ceil(QT_STRIDE) * nh * hd;
                    let e = d.qtrace.entry((ord, sl)).or_insert_with(|| (Buf::new(64).unwrap(), 0));
                    if e.1 < need {
                        let cap = need.max(e.1 * 2).max(4096 / QT_STRIDE * nh * hd);
                        let cap = if hint >= need { cap.min(hint.max(4096 / QT_STRIDE * nh * hd)) } else { cap };
                        let nb = Buf::new(cap * 4)?;
                        if e.1 > 0 {
                            unsafe { cu::memcpy_dtod_async(nb.ptr, e.0.ptr, e.1 * 4, c.stream).map_err(derr)? };
                            c.sync()?;
                        }
                        *e = (nb, cap);
                    }
                }
            }
        }
    }
    Ok(())
}

/// The rotary tables of positions `pos ..` (sliding, full) and the embedded rows.
fn inputs(m: &Dory, ids: &[u32], pos: usize) -> ([Vec<f32>; 2], Vec<f32>) {
    let cf = &m.cfg;
    let (k, hd, hidden) = (ids.len(), cf.hd, cf.hidden);
    let half = hd / 2;
    let mut tabs = [vec![0f32; k * hd], vec![0f32; k * hd]];
    for (ti, inv) in [&m.inv_sliding, &m.inv_full].iter().enumerate() {
        for r in 0..k {
            let at = (pos + r) as f64;
            let row = &mut tabs[ti][r * hd..(r + 1) * hd];
            for j in 0..half {
                let (sn, cs) = (at * inv[j] as f64).sin_cos();
                row[j] = cs as f32;
                row[half + j] = sn as f32;
            }
        }
    }
    let mut rows = vec![0f32; k * hidden];
    for (i, row) in rows.chunks_exact_mut(hidden).enumerate() {
        if (ids[i] as usize) < m.embed.rows() {
            m.embed.row_f32(ids[i] as usize, row);
        }
    }
    (tabs, rows)
}

/// The prompt path's buffers: f16 operands of the GEMMs, the expanded
/// weight, the attention's scratch.
struct Big {
    xh: Buf,
    qgh: Buf,
    ao: Buf,
    mh: Buf,
    aog: Buf,
    isum: Buf,
    wd: Buf,
    /// full layers' scores and exponentials, grown as the context does
    bs: Buf,
    bp: Buf,
    bcap: usize,
    /// a windowed layer's keys and values unwrapped from its ring
    kc: Buf,
    vc: Buf,
    /// past the seal, per row and head: the near part's factor, the far
    /// numerator's, the far scale and mass, then the far numerators
    fa: Buf,
    fc: Buf,
    fm: Buf,
    fz: Buf,
    fnum: Buf,
}

fn make_big(m: &Dory, d: &Dev) -> Res<Big> {
    let cf = &m.cfg;
    let (hidden, ffn, nh, hd) = (cf.hidden, cf.ffn, cf.nh, cf.hd);
    let qs = nh * hd;
    let wmax = d
        .layers
        .iter()
        .flat_map(|l| match l {
            Ld::Attn(a) => a.mats.iter().map(|m| m.rows * m.cols).collect::<Vec<_>>(),
            Ld::Mlp(a) => a.mats.iter().map(|m| m.rows * m.cols).collect(),
        })
        .max()
        .unwrap_or(0);
    let win = cf.window.max(cf.full_window.unwrap_or(0)) + BIG;
    Ok(Big {
        xh: Buf::new(BIG * hidden * 2)?,
        qgh: Buf::new(BIG * qs * 2)?,
        ao: Buf::new(BIG * qs * 2)?,
        mh: Buf::new(BIG * ffn * 2)?,
        aog: Buf::new(BIG * qs * 4)?,
        isum: Buf::new(BIG * nh * 4)?,
        wd: Buf::new(wmax * 2)?,
        bs: Buf::new(64)?,
        bp: Buf::new(64)?,
        bcap: 0,
        kc: Buf::new(win * cf.nk * hd * 2)?,
        vc: Buf::new(win * cf.nk * hd * 2)?,
        fa: Buf::new(BIG * nh * 4)?,
        fc: Buf::new(BIG * nh * 4)?,
        fm: Buf::new(BIG * nh * 4)?,
        fz: Buf::new(BIG * nh * 4)?,
        fnum: Buf::new(BIG * qs * 4)?,
    })
}

/// Rows of a prompt chunk one attention GEMM pass takes at most
/// (`CMF_DORY_ATT_ROWS`): the score and weight buffers grow with rows x keys,
/// and at the exact prefix's last chunk they set the card's peak. 128 rows
/// against the chunk's 512: 412 MiB less at the peak, the same first token
/// within 1.5 % from 4 000 to 32 000 tokens (RTX 2000 Ada).
fn att_rows() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("CMF_DORY_ATT_ROWS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n >= 16).unwrap_or(128))
}

/// A prompt chunk of up to BIG rows by GEMMs: each weight expanded to f16 once
/// for the chunk (`hy_expand_p`), a full layer's attention as two batched
/// GEMMs around a masked softmax (the hybrid's prompt path), a windowed one
/// by the split-K kernels. The last row's head when `want` is `Last`.
///
/// Past the seal a full layer's rows go after its far field's near rows: their
/// near part by the same GEMMs (all the near rows visible, the chunk's
/// causally), the far part from the landmarks (`hy_far_rows`), the two
/// joined by `hy_softmax_hf` and `hy_attn_out_f`; then the rows enter the
/// window (`FarCur::commit`), and while an archive is open they leave the
/// card for it first.
#[allow(clippy::too_many_arguments)]
fn run_big(c: &Ctx, d: &mut Dev, m: &Dory, ids: &[u32], pos: usize, loops: usize, want: Want, logits: &mut Vec<f32>) -> Res<()> {
    let cf = &m.cfg;
    let b = ids.len();
    let (hidden, ffn, nh, nk, hd, vocab) = (cf.hidden, cf.ffn, cf.nh, cf.nk, cf.hd, cf.vocab);
    let (qs, kvs) = (nh * hd, nk * hd);
    let hpk = nh / nk;
    let per_loop = cf.kv == KvMode::PerLoop;
    if b > BIG {
        return Err("a prompt chunk over BIG rows".into());
    }
    prepare_slots(c, d, m, pos, b, loops)?;
    if d.big.is_none() {
        d.big = Some(make_big(m, d)?);
    }
    // the sealed slots' far fields, their cursors moved as the chunk commits
    let far: std::cell::RefCell<HashMap<(usize, usize), FarCur>> = std::cell::RefCell::new(HashMap::new());
    if d.sealed {
        let st = c.st.lock().unwrap();
        for &(ord, sl) in &d.far {
            let o = st.hyb.o1.get(&(d.id, ord * 16 + sl)).ok_or("a sealed slot without its far field")?;
            if o.win_len + b > o.w + SEAL_ROWS {
                return Err(format!("a chunk of {b} rows after {} near rows", o.ns + o.win_len));
            }
            far.borrow_mut().insert((ord, sl), FarCur::of(o));
        }
    }
    let near_max = far.borrow().values().map(|f| f.ns + f.w).max().unwrap_or(0);
    // the full layers' scores of this chunk: hy's budget, halved rows past it
    let win_keys = cf.window.max(cf.full_window.unwrap_or(0)) + b;
    let keys = if d.sealed { (near_max + b).max(win_keys) } else { (pos + b).max(win_keys) };
    let fit = |n: usize| keys * n * nh * 4 <= 384 << 20;
    let mut bn_max = b.min(att_rows());
    while bn_max > 16 && !fit(bn_max) {
        bn_max = bn_max.div_ceil(2);
    }
    {
        let bg = d.big.as_mut().unwrap();
        let need = keys * bn_max * nh;
        if need > bg.bcap {
            c.sync()?;
            let cap = need.max(bg.bcap + bg.bcap / 2);
            bg.bs = Buf::new(cap * 4)?;
            bg.bp = Buf::new(cap * 2)?;
            bg.bcap = cap;
        }
    }
    let (tabs, rows) = inputs(m, ids, pos);
    let kt = std::env::var("CMF_DORY_KT_AT").ok().and_then(|v| v.parse::<usize>().ok()).is_some_and(|p| p >= pos && p < pos + b);
    if kt {
        super::super::kt_start();
    }
    let bg = d.big.as_ref().unwrap();
    let hfx = &*d.hfx;
    let fx = &d.fx;
    let eps = crate::dory::EPS;
    let (b_i, hid_i, nh_i, nk_i, one) = (b as i32, hidden as i32, nh as i32, nk as i32, 1i32);
    let nul = 0u64;
    // the archive's parts this chunk staged: (slot, rows, query rows)
    let caps: std::cell::RefCell<Vec<(usize, usize, usize)>> = std::cell::RefCell::new(Vec::new());
    unsafe {
        cu::memcpy_htod_async(d.rope[0].ptr, &tabs[0], c.stream).map_err(derr)?;
        cu::memcpy_htod_async(d.rope[1].ptr, &tabs[1], c.stream).map_err(derr)?;
        cu::memcpy_htod_async(d.h.ptr, &rows, c.stream).map_err(derr)?;
        let n = (b * hidden) as u64;
        c.launch(hfx.to_half, (n.div_ceil(256) as u32, 1), 256, 0, kargs!(d.h.ptr, bg.xh.ptr, n))?;
        let pl = Planes { stride: hidden, ..d.planes() };
        split(c, hfx, &pl, d.h.ptr, hidden, hidden, b)?;
    }
    // y [b][ldy] = W x: W expanded to f16, one GEMM
    let mul = |mm: &DevMat, x: u64, y: u64, ldy: usize| -> Res<()> {
        let (rows_i, cols_i) = (mm.rows as i32, mm.cols as i32);
        let threads = (mm.rows.div_ceil(16) * (mm.cols / 32) * 32) as u64;
        unsafe {
            c.launch(hfx.expand_p, (threads.div_ceil(256) as u32, 1), 256, 0, kargs!(mm.w.ptr, mm.aux0(), mm.aux1(), bg.wd.ptr, rows_i, cols_i))?;
            c.gemm(true, false, mm.rows, b, mm.cols, 1.0, bg.wd.ptr, mm.cols, x, mm.cols, 0.0, y, ldy, Gemm::Half)
        }
    };
    let slerp = |alpha: &Buf| -> Res<()> {
        let threads = hidden.div_ceil(8).next_multiple_of(32).min(1024) as u32;
        unsafe { c.launch(fx.slerp, (b as u32, 1), threads, 0, kargs!(d.h.ptr, d.o.ptr, alpha.ptr, d.p[0].ptr, d.p[1].ptr, hid_i, eps, bg.xh.ptr)) }
    };
    let mix = |mode: i32| -> Res<()> {
        unsafe { c.launch(fx.mix, (b as u32, 1), 256, 0, kargs!(d.h.ptr, d.input.ptr, d.state.ptr, mode, d.p[0].ptr, d.p[1].ptr, hid_i, eps, bg.xh.ptr)) }
    };
    let layer = |li: usize, lp: usize| -> Res<()> {
        match &d.layers[li] {
            Ld::Attn(a) => {
                let [qm, km, vm, gm, om] = &a.mats;
                mul(qm, bg.xh.ptr, d.yq.ptr, qs)?;
                mul(km, bg.xh.ptr, d.yk.ptr, kvs)?;
                mul(vm, bg.xh.ptr, d.yv.ptr, kvs)?;
                mul(gm, bg.xh.ptr, d.yg.ptr, qs)?;
                let rec = li >= cf.n_in && li < cf.n_in + cf.n_rec;
                let sl = if rec && per_loop { lp } else { 0 };
                let tab = if a.sliding { d.rope[0].ptr } else { d.rope[1].ptr };
                let window = if a.sliding { cf.window } else { cf.full_window.unwrap_or(0) };
                let sealed = if a.sliding { None } else { far.borrow().get(&(a.ordinal, sl)).copied() };
                // the rows' K and V: a slot's at their position (a ring's by
                // it), or a far field's mirror after its near rows
                let (kp, vp, qtr, at_i, cap_i) = match &sealed {
                    Some(f) => (f.p[0], f.p[1], 0u64, f.near() as i32, 0i32),
                    None => {
                        let slot = &d.slots[a.ordinal][sl];
                        (slot.k.ptr, slot.v.ptr, d.qtrace.get(&(a.ordinal, sl)).map_or(0, |x| x.0.ptr), pos as i32, slot.ring as i32)
                    }
                };
                let win_i = window as i32;
                unsafe {
                    c.launch(
                        fx.qk,
                        ((nh + nk) as u32, b as u32),
                        hd as u32,
                        0,
                        kargs!(d.yq.ptr, d.yk.ptr, d.yv.ptr, d.yg.ptr, tab, a.sq.ptr, a.sg.ptr, d.qg.ptr, d.gate.ptr, kp, vp, qtr, at_i, cap_i, nh_i, nk_i, eps, b_i, nul, nul, bg.qgh.ptr),
                    )?;
                }
                if let Some(mut f) = sealed {
                    // past the seal: the near rows before the chunk all
                    // visible, the chunk's causally (a near row r sits at
                    // position col_base + r for the mask), and the far part
                    let n0 = f.near();
                    let [_, _, kt, _, mu, mz, th] = f.p;
                    let far_on = f.far_len > 0;
                    if far_on {
                        let (m_i, rf) = (f.m as i32, f.rect as i32);
                        unsafe {
                            c.launch(hfx.far_rows, (nh as u32, b as u32), 256, 0, kargs!(d.qg.ptr, kt, mu, mz, th, bg.fm.ptr, bg.fz.ptr, bg.fnum.ptr, m_i, nh_i, f.scale, rf, b_i))?;
                        }
                    }
                    let lda = kvs;
                    let grow = hpk * hd;
                    let col_base = pos - n0;
                    let mut b0 = 0usize;
                    while b0 < b {
                        let bn = (b - b0).min(bn_max);
                        let p0 = pos + b0;
                        let l_rows = n0 + b0 + bn;
                        let cols = bn * hpk;
                        let (l_i, p0_i, cb_i, hpk_i, cols_i, istride) = (l_rows as i32, p0 as i32, col_base as i32, hpk as i32, cols as i32, (b * hpk) as i32);
                        let off = (b0 * hpk * 4) as u64;
                        let (fmp, fzp) = if far_on { (bg.fm.ptr + (b0 * nh * 4) as u64, bg.fz.ptr + (b0 * nh * 4) as u64) } else { (0u64, 0u64) };
                        unsafe {
                            gemm_batched_h(c, true, l_rows, cols, hd, cf.attn_scale, kp, lda, hd, bg.qgh.ptr + (b0 * grow * 2) as u64, hd, b * grow, bg.bs.ptr, l_rows, l_rows * cols, nk)?;
                            c.launch(
                                hfx.softmax_hf,
                                ((nk * cols) as u32, 1),
                                256,
                                0,
                                kargs!(bg.bs.ptr, bg.bp.ptr, bg.isum.ptr + off, bg.fa.ptr + off, bg.fc.ptr + off, l_i, p0_i, cb_i, hpk_i, cols_i, istride, fmp, fzp, nh_i),
                            )?;
                            gemm_batched_h(c, false, hd, cols, l_rows, 1.0, vp, lda, hd, bg.bp.ptr, l_rows, l_rows * cols, bg.aog.ptr + (b0 * grow * 4) as u64, hd, b * grow, nk)?;
                        }
                        b0 += bn;
                    }
                    let n_ao = (b * qs) as u64;
                    let fnp = if far_on { bg.fnum.ptr } else { 0u64 };
                    unsafe {
                        c.launch(hfx.attn_out_f, (n_ao.div_ceil(256) as u32, 1), 256, 0, kargs!(bg.aog.ptr, bg.isum.ptr, bg.fa.ptr, bg.fc.ptr, fnp, d.gate.ptr, bg.ao.ptr, nh_i, b_i, one))?;
                    }
                    // the last-loop cache keeps a recurrent layer's last pass:
                    // the earlier passes' rows are written over in place
                    if !rec || per_loop || lp + 1 == loops {
                        if let Some(ar) = d.arch.as_ref() {
                            let i = ar.slots.iter().position(|s| (s.ord, s.sl) == (a.ordinal, sl)).ok_or("an archived slot missing")?;
                            unsafe { stage_rows(c, ar, i, kp, vp, n0, b, d.qg.ptr, nh, nk, hd, pos)? };
                            caps.borrow_mut().push((i, b, kept_queries(pos, b)));
                        }
                        unsafe { f.commit(c, hfx, b, nk, kvs)? };
                        far.borrow_mut().insert((a.ordinal, sl), f);
                    }
                } else if window > 0 {
                    let slot = &d.slots[a.ordinal][sl];
                    // a window: its keys unwrapped from the ring (two copies
                    // at most), then the full layer's GEMMs with a windowed
                    // softmax
                    let lo_min = (pos + 1).saturating_sub(window);
                    let l_rows = pos + b - lo_min;
                    let row_b = kvs * 2;
                    let ring = slot.ring;
                    let mut at = 0usize;
                    while at < l_rows {
                        let src = (lo_min + at) % ring;
                        let n = (l_rows - at).min(ring - src);
                        unsafe {
                            cu::memcpy_dtod_async(bg.kc.ptr + (at * row_b) as u64, slot.k.ptr + (src * row_b) as u64, n * row_b, c.stream).map_err(derr)?;
                            cu::memcpy_dtod_async(bg.vc.ptr + (at * row_b) as u64, slot.v.ptr + (src * row_b) as u64, n * row_b, c.stream).map_err(derr)?;
                        }
                        at += n;
                    }
                    let lda = kvs;
                    let grow = hpk * hd;
                    let mut b0 = 0usize;
                    while b0 < b {
                        let bn = (b - b0).min(bn_max);
                        let cols = bn * hpk;
                        let (l_i, p0_i, cb_i, cols_i, istride) = (l_rows as i32, (pos + b0) as i32, lo_min as i32, cols as i32, (b * hpk) as i32);
                        let isum = bg.isum.ptr + (b0 * hpk * 4) as u64;
                        unsafe {
                            gemm_batched_h(c, true, l_rows, cols, hd, cf.attn_scale, bg.kc.ptr, lda, hd, bg.qgh.ptr + (b0 * grow * 2) as u64, hd, b * grow, bg.bs.ptr, l_rows, l_rows * cols, nk)?;
                            c.launch(fx.softmax_w, ((nk * cols) as u32, 1), 256, 0, kargs!(bg.bs.ptr, bg.bp.ptr, isum, l_i, p0_i, cb_i, cols_i, istride, win_i))?;
                            gemm_batched_h(c, false, hd, cols, l_rows, 1.0, bg.vc.ptr, lda, hd, bg.bp.ptr, l_rows, l_rows * cols, bg.aog.ptr + (b0 * grow * 4) as u64, hd, b * grow, nk)?;
                        }
                        b0 += bn;
                    }
                    let n_ao = (b * qs) as u64;
                    unsafe { c.launch(hfx.attn_out, (n_ao.div_ceil(256) as u32, 1), 256, 0, kargs!(bg.aog.ptr, bg.isum.ptr, d.gate.ptr, bg.ao.ptr, nh_i, b_i, one))? };
                } else {
                    // a full layer: scores of every key by GEMM, the masked
                    // softmax, values by GEMM (rows from position 0)
                    let slot = &d.slots[a.ordinal][sl];
                    let lda = kvs;
                    let grow = hpk * hd;
                    let mut b0 = 0usize;
                    while b0 < b {
                        let bn = (b - b0).min(bn_max);
                        let p0 = pos + b0;
                        let l_rows = p0 + bn;
                        let cols = bn * hpk;
                        let (l_i, p0_i, cb_i, hpk_i, cols_i, istride) = (l_rows as i32, p0 as i32, 0i32, hpk as i32, cols as i32, (b * hpk) as i32);
                        let isum = bg.isum.ptr + (b0 * hpk * 4) as u64;
                        unsafe {
                            gemm_batched_h(c, true, l_rows, cols, hd, cf.attn_scale, slot.k.ptr, lda, hd, bg.qgh.ptr + (b0 * grow * 2) as u64, hd, b * grow, bg.bs.ptr, l_rows, l_rows * cols, nk)?;
                            c.launch(hfx.softmax_h, ((nk * cols) as u32, 1), 256, 0, kargs!(bg.bs.ptr, bg.bp.ptr, isum, l_i, p0_i, cb_i, hpk_i, cols_i, istride))?;
                            gemm_batched_h(c, false, hd, cols, l_rows, 1.0, slot.v.ptr, lda, hd, bg.bp.ptr, l_rows, l_rows * cols, bg.aog.ptr + (b0 * grow * 4) as u64, hd, b * grow, nk)?;
                        }
                        b0 += bn;
                    }
                    let n_ao = (b * qs) as u64;
                    unsafe { c.launch(hfx.attn_out, (n_ao.div_ceil(256) as u32, 1), 256, 0, kargs!(bg.aog.ptr, bg.isum.ptr, d.gate.ptr, bg.ao.ptr, nh_i, b_i, one))? };
                }
                mul(om, bg.ao.ptr, d.o.ptr, hidden)?;
                slerp(&a.alpha)
            }
            Ld::Mlp(a) => {
                let [gm, um, dm] = &a.mats;
                let (ffn_i, tot) = (ffn as i32, (b * ffn) as u32);
                mul(gm, bg.xh.ptr, d.fg.ptr, ffn)?;
                mul(um, bg.xh.ptr, d.fu.ptr, ffn)?;
                unsafe { c.launch(fx.silu, (tot.div_ceil(256), 1), 256, 0, kargs!(d.fg.ptr, d.fu.ptr, a.sa.ptr, a.sb.ptr, bg.mh.ptr, nul, ffn_i, b_i, one))? };
                mul(dm, bg.mh.ptr, d.o.ptr, hidden)?;
                slerp(&a.alpha)
            }
        }
    };
    for li in 0..cf.n_in {
        layer(li, 0)?;
    }
    mix(0)?;
    for i in 0..loops {
        for li in cf.n_in..cf.n_in + cf.n_rec {
            layer(li, i)?;
        }
        mix(if i + 1 == loops { 2 } else { 1 })?;
    }
    for li in cf.n_in + cf.n_rec..d.layers.len() {
        layer(li, 0)?;
    }
    match want {
        Want::Last => {
            let pl = Planes { stride: hidden, ..d.planes() };
            let last = Planes { p0: pl.p0 + ((b - 1) * hidden * 2) as u64, p1: pl.p1 + ((b - 1) * hidden * 2) as u64, p2: pl.p2 + ((b - 1) * hidden * 2) as u64, stride: hidden };
            unsafe { d.head.run(c, hfx, &last, hidden, 1)? };
            let mut v = vec![0f32; vocab];
            unsafe { cu::memcpy_dtoh_async(&mut v, d.logits.ptr, c.stream).map_err(derr)? };
            c.sync()?;
            for (x, s) in v.iter_mut().zip(&m.logit_scale) {
                *x *= *s;
            }
            logits.clear();
            logits.extend(v);
        }
        _ => c.sync()?,
    }
    if kt {
        super::super::kt_stop();
    }
    if d.sealed {
        let mut st = c.st.lock().unwrap();
        for (&(ord, sl), f) in far.borrow().iter() {
            if let Some(o) = st.hyb.o1.get_mut(&(d.id, ord * 16 + sl)) {
                f.store(o);
            }
        }
    }
    let caps = caps.into_inner();
    if !caps.is_empty()
        && let Err(e) = archive_take(c, d, cf, &caps, pos, b)
    {
        give_up_rows(d, &e);
    }
    Ok(())
}

/// The queries of `b` rows from `pos` the archive keeps: one in QT_STRIDE by position.
fn kept_queries(pos: usize, b: usize) -> usize {
    let first = (QT_STRIDE - pos % QT_STRIDE) % QT_STRIDE;
    if first < b { (b - first).div_ceil(QT_STRIDE) } else { 0 }
}

/// A sealed chunk's rows of slot `i` into the archive's staging: `b` K and V
/// rows from row `n0` of the far field's mirror, and the kept queries of
/// `qg` (`[kv head][b][head of the group][hd]`).
#[allow(clippy::too_many_arguments)]
unsafe fn stage_rows(c: &Ctx, ar: &Archive, i: usize, kp: u64, vp: u64, n0: usize, b: usize, qg: u64, nh: usize, nk: usize, hd: usize, pos: usize) -> Res<()> {
    let kv_row = nk * hd;
    let base = ar.staging.ptr + (i * ar.per) as u64;
    let (bytes, at) = (b * kv_row * 2, (n0 * kv_row * 2) as u64);
    let kept = kept_queries(pos, b);
    unsafe {
        cu::memcpy_dtod_async(base, kp + at, bytes, c.stream).map_err(derr)?;
        cu::memcpy_dtod_async(base + (BIG * kv_row * 2) as u64, vp + at, bytes, c.stream).map_err(derr)?;
        if kept > 0 {
            let first = (QT_STRIDE - pos % QT_STRIDE) % QT_STRIDE;
            let (nh_i, nk_i, hd_i, b_i, first_i, stride_i) = (nh as i32, nk as i32, hd as i32, b as i32, first as i32, QT_STRIDE as i32);
            let qd = base + (2 * BIG * kv_row * 2) as u64;
            c.launch(c.f.q_rows_strided, (kept as u32, 1), 256, 0, kargs!(qg, qd, nh_i, nk_i, hd_i, b_i, first_i, stride_i))?;
        }
    }
    Ok(())
}

/// One slot's rows of a chunk off the card: K and V in f16, the kept queries.
type Staged = (Vec<u16>, Vec<u16>, Vec<f32>);

/// The staged rows of a chunk (the stream is done with it) into the archive's
/// stores, the slots on threads of their own.
fn archive_take(c: &Ctx, d: &mut Dev, cf: &crate::dory::DoryCfg, caps: &[(usize, usize, usize)], pos: usize, b: usize) -> Res<()> {
    let Some(ar) = d.arch.as_mut() else { return Ok(()) };
    let (kv_row, qs) = (cf.nk * cf.hd, cf.nh * cf.hd);
    if ar.rows != pos || caps.len() != ar.slots.len() || caps.iter().any(|&(_, n, _)| n != b) {
        return Err(format!("the archive holds {} rows, a chunk of {b} at {pos} staged {} of {} slots", ar.rows, caps.len(), ar.slots.len()));
    }
    let mut got: Vec<Staged> = Vec::with_capacity(caps.len());
    for &(i, n, kept) in caps {
        let base = ar.staging.ptr + (i * ar.per) as u64;
        let (mut k, mut v, mut q) = (vec![0u16; n * kv_row], vec![0u16; n * kv_row], vec![0f32; kept * qs]);
        unsafe {
            cu::memcpy_dtoh_async(k.as_mut_slice(), base, c.stream).map_err(derr)?;
            cu::memcpy_dtoh_async(v.as_mut_slice(), base + (BIG * kv_row * 2) as u64, c.stream).map_err(derr)?;
            if kept > 0 {
                cu::memcpy_dtoh_async(q.as_mut_slice(), base + (2 * BIG * kv_row * 2) as u64, c.stream).map_err(derr)?;
            }
        }
        got.push((k, v, q));
    }
    c.sync()?;
    let mut order: Vec<usize> = caps.iter().map(|x| x.0).collect();
    let mut slots: Vec<Option<&mut ArchSlot>> = ar.slots.iter_mut().map(Some).collect();
    let jobs: Vec<(&mut ArchSlot, Staged)> = order.drain(..).zip(got).map(|(i, g)| (slots[i].take().expect("a slot staged twice"), g)).collect();
    let errs: Vec<String> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .into_iter()
            .map(|(s, (k, v, q))| {
                sc.spawn(move || -> Result<(), String> {
                    let wide = |h: &[u16]| -> Vec<f32> { h.iter().map(|&x| cortiq_core::quant::f16_to_f32(x)).collect() };
                    s.k.push(&wide(&k))?;
                    s.v.push(&wide(&v))?;
                    s.q.push(&q)
                })
            })
            .collect();
        hs.into_iter().filter_map(|h| h.join().unwrap_or_else(|_| Err("an archive thread panicked".into())).err()).collect()
    });
    if let Some(e) = errs.into_iter().next() {
        return Err(e);
    }
    ar.rows = pos + b;
    Ok(())
}

/// The rows did not fit where they were going: the far field stays as first
/// sealed, and no row stays exact past it.
fn give_up_rows(d: &mut Dev, why: &str) {
    tracing::warn!("dory on CUDA: {why}: the prompt's rows are dropped, and none stays exact past the seal");
    d.arch = None;
}

#[allow(clippy::too_many_arguments)]
fn run(c: &Ctx, d: &mut Dev, m: &Dory, ids: &[u32], pos: usize, loops: usize, want: Want, logits: &mut Vec<f32>) -> Res<()> {
    let cf = &m.cfg;
    let k = ids.len();
    let (hidden, nh, nk, hd, vocab) = (cf.hidden, cf.nh, cf.nk, cf.hd, cf.vocab);
    let per_loop = cf.kv == KvMode::PerLoop;
    // the slots this call writes, and the queries a seal will want
    prepare_slots(c, d, m, pos, k, loops)?;
    // a row's keys are bounded by its slot: a sliding layer's window, a full
    // layer's capacity (or the near field once sealed), so a recorded graph
    // covers every position up to them
    let full_bound = d.slots.iter().flatten().filter(|s| s.ring == 0).map(|s| s.cap).max().unwrap_or(0);
    let keys_max = full_bound.max(cf.window).max(cf.full_window.unwrap_or(0)).max(if d.sealed { crate::dory::MAX_NEAR } else { 0 });
    d.ensure_scores(c, nh, keys_max)?;
    // the rotary tables and the embedded rows
    let half = hd / 2;
    let mut tabs = [vec![0f32; k * hd], vec![0f32; k * hd]];
    for (ti, inv) in [&m.inv_sliding, &m.inv_full].iter().enumerate() {
        for r in 0..k {
            let at = (pos + r) as f64;
            let row = &mut tabs[ti][r * hd..(r + 1) * hd];
            for j in 0..half {
                let (sn, cs) = (at * inv[j] as f64).sin_cos();
                row[j] = cs as f32;
                row[half + j] = sn as f32;
            }
        }
    }
    let mut rows = vec![0f32; k * hidden];
    for (i, row) in rows.chunks_exact_mut(hidden).enumerate() {
        if (ids[i] as usize) < m.embed.rows() {
            m.embed.row_f32(ids[i] as usize, row);
        }
    }
    unsafe {
        cu::memcpy_htod_async(d.rope[0].ptr, &tabs[0], c.stream).map_err(derr)?;
        cu::memcpy_htod_async(d.rope[1].ptr, &tabs[1], c.stream).map_err(derr)?;
        cu::memcpy_htod_async(d.h.ptr, &rows, c.stream).map_err(derr)?;
        cu::memcpy_htod_async(d.cr.ptr, &[pos as i32, k as i32], c.stream).map_err(derr)?;
        let pl = Planes { stride: hidden, ..d.planes() };
        split(c, &d.hfx, &pl, d.h.ptr, hidden, hidden, k)?;
    }
    let g_mode = if k == 1 && want != Want::None { m.gate.clone() } else { None };
    let read_state = |c: &Ctx, d: &Dev| -> Res<Vec<f32>> {
        let mut v = vec![0f32; hidden];
        unsafe { cu::memcpy_dtoh_async(&mut v, d.state.ptr, c.stream).map_err(derr)? };
        c.sync()?;
        Ok(v)
    };
    let read_logits = |c: &Ctx, d: &Dev, nrows: usize| -> Res<Vec<f32>> {
        let mut v = vec![0f32; nrows * vocab];
        unsafe { cu::memcpy_dtoh_async(&mut v, d.logits.ptr, c.stream).map_err(derr)? };
        c.sync()?;
        for row in v.chunks_exact_mut(vocab) {
            for (x, s) in row.iter_mut().zip(&m.logit_scale) {
                *x *= *s;
            }
        }
        Ok(v)
    };
    // the far fields' call records of this call, made once per sealed slot:
    // a token's row takes its window slot now (the oldest row into the far
    // sums first, once the window is full); several rows (a prompt's last
    // ones) go after the near rows, read by the rows after them as a verify
    // window's on the hybrid arm, and enter the window once the call is done
    let mut far_ci: HashMap<(usize, usize), (u64, [i32; 8], bool)> = HashMap::new();
    let mut ci_host: Vec<i32> = Vec::new();
    if d.sealed {
        let mut st = c.st.lock().unwrap();
        for &(ord, sl) in &d.far {
            let key = (d.id, ord * 16 + sl);
            let o = st.hyb.o1.get_mut(&key).ok_or("a sealed slot without its far field")?;
            if k > 1 {
                let base = o.ns + o.win_len;
                if o.win_len + k > o.w + SEAL_ROWS || k > o.w {
                    return Err(format!("{k} rows after {base} near rows of a far field"));
                }
                let rb = attn_rb(base + k);
                let rec = [base as i32, base as i32, rb as i32, d.s_keys as i32, NBCAP as i32, 1, base as i32, (o.far_len > 0) as i32];
                far_ci.insert((ord, sl), ((ci_host.len() * 4) as u64, rec, false));
                ci_host.extend_from_slice(&rec);
                continue;
            }
            let full = o.win_len == o.w;
            let row = o.ns + if full { o.win_head } else { o.win_len };
            if full {
                o.far_len += 1;
                o.win_head = (o.win_head + 1) % o.w;
            } else {
                o.win_len += 1;
            }
            let keys = o.ns + o.win_len;
            let rb = attn_rb(keys);
            let rec = [(keys - 1) as i32, row as i32, rb as i32, d.s_keys as i32, NBCAP as i32, 1, row as i32, (o.far_len > 0) as i32];
            // a full ring's evicted row goes into the far sums first
            far_ci.insert((ord, sl), ((ci_host.len() * 4) as u64, rec, full));
            ci_host.extend_from_slice(&rec);
        }
        drop(st);
        if ci_host.len() * 4 > d.ci.bytes {
            d.ci = Buf::new(ci_host.len() * 4 * 2)?;
        }
        unsafe { cu::memcpy_htod_async(d.ci.ptr, &ci_host, c.stream).map_err(derr)? };
    }
    let o1_state: HashMap<(usize, usize), FarDev> = if d.sealed {
        let st = c.st.lock().unwrap();
        d.far
            .iter()
            .filter_map(|&(ord, sl)| {
                let o = st.hyb.o1.get(&(d.id, ord * 16 + sl))?;
                Some(((ord, sl), ([o.nk.ptr, o.nv.ptr, o.kt.ptr, o.qt.ptr, o.mu.ptr, o.mz.ptr, o.th.ptr], o.m, o.scale, o.rect_fm)))
            })
            .collect()
    } else {
        HashMap::new()
    };
    let qs = nh * hd;
    let eps = crate::dory::EPS;
    let run_stage = |d: &Dev, stage: Stage| -> Res<()> {
        let fx = &d.fx;
        let hfx = &*d.hfx;
        let pl = Planes { stride: hidden, ..d.planes() };
        let zpl = Planes { p0: d.zp[0].ptr, p1: d.zp[1].ptr, p2: d.zp[1].ptr, stride: qs };
        let mpl = Planes { p0: d.mp[0].ptr, p1: d.mp[1].ptr, p2: d.mp[1].ptr, stride: cf.ffn };
        let (k_i, hid_i, nh_i, nk_i, qs_i) = (k as i32, hidden as i32, nh as i32, nk as i32, qs as i32);
        let nul = 0u64;
        let crp = d.cr.ptr;
        let slerp = |alpha: &Buf| -> Res<()> {
            let threads = hidden.div_ceil(8).next_multiple_of(32).min(1024) as u32;
            unsafe { c.launch(fx.slerp, (k as u32, 1), threads, 0, kargs!(d.h.ptr, d.o.ptr, alpha.ptr, d.p[0].ptr, d.p[1].ptr, hid_i, eps, nul)) }
        };
        let layer = |li: usize, lp: usize| -> Res<()> {
            match &d.layers[li] {
                Ld::Attn(a) => {
                    let rec = li >= cf.n_in && li < cf.n_in + cf.n_rec;
                    let sl = if rec && per_loop { lp } else { 0 };
                    unsafe { a.qkvg.run(c, hfx, &pl, hidden, k)? };
                    let tab = if a.sliding { d.rope[0].ptr } else { d.rope[1].ptr };
                    if d.sealed && !a.sliding {
                        let (cio, w, ins) = far_ci.get(&(a.ordinal, sl)).ok_or("no call record for a sealed slot")?;
                        let (lb, fm, fscale, frect) = o1_state.get(&(a.ordinal, sl)).ok_or("a sealed slot without its far field")?;
                        let cip = d.ci.ptr + cio;
                        // a later pass of this position on the last-loop
                        // cache rewrites its row: no second eviction
                        let first = lp == 0 || per_loop;
                        let (fm_i, scale, frect_i) = (*fm as i32, cf.attn_scale, *frect as i32);
                        let (n0_i, r0, rb_i, ncap_i, nbcap_i, one_i) = (w[0], 0i32, w[2], w[3], w[4], 1i32);
                        unsafe {
                            if first && *ins {
                                c.launch(hfx.far_insert, ((nk * 4 * fm).div_ceil(8) as u32, 1), 256, 0, kargs!(lb[0], lb[1], lb[3], lb[5], lb[6], cip, fm_i, nk_i, *fscale))?;
                            }
                            // row b's K and V at the record's row + b
                            c.launch(
                                fx.qk,
                                ((nh + nk) as u32, k as u32),
                                hd as u32,
                                0,
                                kargs!(d.yq.ptr, d.yk.ptr, d.yv.ptr, d.yg.ptr, tab, a.sq.ptr, a.sg.ptr, d.qg.ptr, d.gate.ptr, lb[0], lb[1], nul, 0i32, 0i32, nh_i, nk_i, eps, k_i, nul, cip, nul),
                            )?;
                            let nblk = (d.s_keys / rb_i as usize).min(NBCAP - 1) as u32;
                            let _ = one_i;
                            c.launch_3d(hfx.scores, (nblk, nk as u32, k as u32), 256, 0, kargs!(d.qg.ptr, lb[0], d.s.ptr, d.mx.ptr, n0_i, r0, nk_i, scale, rb_i, ncap_i, nbcap_i, k_i, cip))?;
                            if k == 1 {
                                c.launch(hfx.far_prep, (nh as u32, 1), 256, 0, kargs!(d.qg.ptr, lb[2], lb[4], lb[5], lb[6], d.mx.ptr, d.pp.ptr, d.ee.ptr, cip, fm_i, nh_i, *fscale, frect_i))?;
                            } else {
                                c.launch(hfx.far_prep_b, (nh as u32, k as u32), 256, 0, kargs!(d.qg.ptr, lb[2], lb[4], lb[5], lb[6], d.mx.ptr, d.pp.ptr, d.ee.ptr, cip, fm_i, nh_i, *fscale, frect_i, k_i))?;
                            }
                            c.launch_3d(hfx.values, (nblk, nk as u32, k as u32), 256, 0, kargs!(d.s.ptr, d.mx.ptr, lb[1], d.pp.ptr, d.ee.ptr, n0_i, r0, nk_i, rb_i, ncap_i, nbcap_i, k_i, cip))?;
                            c.launch(hfx.finish, (nh as u32, k as u32), 256, 0, kargs!(d.pp.ptr, d.ee.ptr, d.gate.ptr, zpl.p0, zpl.p1, zpl.p1, qs_i, n0_i, rb_i, nh_i, nbcap_i, cip))?;
                        }
                    } else {
                        let slot = &d.slots[a.ordinal][sl];
                        let window = if a.sliding { cf.window } else { cf.full_window.unwrap_or(0) };
                        let bound = if window > 0 { window } else { slot.cap };
                        // smaller key blocks than the arm's, more of them (at most 128)
                        let rb = (attn_rb(bound) / 2).max(32).max(bound.div_ceil(128));
                        let nblk = bound.div_ceil(rb) as u32;
                        let qtr = d.qtrace.get(&(a.ordinal, sl)).map_or(0, |b| b.0.ptr);
                        let (pos_i, cap_i, win_i, rb_i, nbcap_i) = (pos as i32, slot.ring as i32, window as i32, rb as i32, NBCAP as i32);
                        unsafe {
                            c.launch(
                                fx.qk,
                                ((nh + nk) as u32, k as u32),
                                hd as u32,
                                0,
                                kargs!(d.yq.ptr, d.yk.ptr, d.yv.ptr, d.yg.ptr, tab, a.sq.ptr, a.sg.ptr, d.qg.ptr, d.gate.ptr, slot.k.ptr, slot.v.ptr, qtr, pos_i, cap_i, nh_i, nk_i, eps, k_i, crp, nul, nul),
                            )?;
                            c.launch_3d(fx.attn, (k as u32, nblk, nk as u32), 256, 0, kargs!(d.qg.ptr, slot.k.ptr, slot.v.ptr, d.pp.ptr, d.ee.ptr, d.mx.ptr, pos_i, cap_i, win_i, nk_i, cf.attn_scale, rb_i, nbcap_i, k_i, crp))?;
                            c.launch(fx.finish_m, (nh as u32, k as u32), 256, 0, kargs!(d.pp.ptr, d.ee.ptr, d.mx.ptr, d.gate.ptr, zpl.p0, zpl.p1, qs_i, pos_i, win_i, rb_i, nh_i, nbcap_i, crp))?;
                        }
                    }
                    unsafe { a.o.run(c, hfx, &zpl, qs, k)? };
                    slerp(&a.alpha)
                }
                Ld::Mlp(a) => {
                    let ffn_i = cf.ffn as i32;
                    unsafe {
                        a.gu.run(c, hfx, &pl, hidden, k)?;
                        if !a.fused {
                            let tot = (k * cf.ffn) as u32;
                            c.launch(fx.silu, (tot.div_ceil(256), 1), 256, 0, kargs!(d.fg.ptr, d.fu.ptr, a.sa.ptr, a.sb.ptr, mpl.p0, mpl.p1, ffn_i, k_i, 0i32))?;
                        }
                        a.down.run(c, hfx, &mpl, cf.ffn, k)?;
                    }
                    slerp(&a.alpha)
                }
            }
        };
        let mix = |mode: i32| -> Res<()> {
            unsafe { c.launch(fx.mix, (k as u32, 1), 256, 0, kargs!(d.h.ptr, d.input.ptr, d.state.ptr, mode, d.p[0].ptr, d.p[1].ptr, hid_i, eps, nul)) }
        };
        if matches!(stage, Stage::All | Stage::First | Stage::FirstRead) {
            for li in 0..cf.n_in {
                layer(li, 0)?;
            }
            mix(0)?;
        }
        if stage == Stage::All {
            for i in 0..loops {
                for li in cf.n_in..cf.n_in + cf.n_rec {
                    layer(li, i)?;
                }
                mix(if i + 1 == loops { 2 } else { 1 })?;
            }
        } else if matches!(stage, Stage::First | Stage::FirstRead) {
            for li in cf.n_in..cf.n_in + cf.n_rec {
                layer(li, 0)?;
            }
            mix(3)?;
        } else if matches!(stage, Stage::SecondRead | Stage::SecondFresh) {
            mix(4)?;
            for li in cf.n_in..cf.n_in + cf.n_rec {
                layer(li, 1)?;
            }
            mix(2)?;
        }
        if stage == Stage::First {
            return Ok(());
        }
        // a second read-out rewrites this position in the output block
        let again = usize::from(stage == Stage::SecondRead);
        for li in cf.n_in + cf.n_rec..d.layers.len() {
            layer(li, again)?;
        }
        match want {
            Want::None => {}
            Want::Last => {
                let last = Planes { p0: pl.p0 + ((k - 1) * hidden * 2) as u64, p1: pl.p1 + ((k - 1) * hidden * 2) as u64, p2: pl.p2 + ((k - 1) * hidden * 2) as u64, stride: hidden };
                unsafe { d.head.run(c, hfx, &last, hidden, 1)? };
            }
            Want::All => unsafe { d.head.run(c, hfx, &pl, hidden, k)? },
        }
        Ok(())
    };
    // a one-row stage replays its recorded graph while the pointers it was
    // recorded with hold; the inputs (the row, the tables, the records) are
    // on the stream already
    let sig = |d: &Dev| -> Vec<u64> {
        let mut v: Vec<u64> = d.slots.iter().flatten().flat_map(|s| [s.k.ptr, s.v.ptr, s.cap as u64]).collect();
        let mut tr: Vec<_> = d.qtrace.iter().map(|(&(o, s), b)| ((o * 16 + s) as u64, b.0.ptr)).collect();
        tr.sort_unstable();
        v.extend(tr.into_iter().flat_map(|(a, b)| [a, b]));
        v.extend([d.s.ptr, d.s_keys as u64, d.ci.ptr, d.sealed as u64]);
        let mut fc: Vec<_> = far_ci.iter().map(|(&(o, s), &(off, _, ins))| ((o * 16 + s) as u64, off, ins as u64)).collect();
        fc.sort_unstable();
        v.extend(fc.into_iter().flat_map(|(a, b, i)| [a, b, i]));
        // a far field built again (a long prompt's rebuild, a new sequence's
        // seal) has buffers of its own
        let mut fp: Vec<_> = o1_state.iter().map(|(&(o, s), f)| ((o * 16 + s) as u64, f.0)).collect();
        fp.sort_unstable();
        v.extend(fp.into_iter().flat_map(|(a, p)| std::iter::once(a).chain(p)));
        v
    };
    // `CMF_DORY_KT=N`: the kernel times of the N-th one-row call
    static ONE_ROW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    // `CMF_DORY_KT_AT=P`: of the prompt chunk that starts at position P
    let kt = (k == 1 && {
        let n = ONE_ROW.fetch_add(1, Ordering::Relaxed);
        std::env::var("CMF_DORY_KT").ok().and_then(|v| v.parse::<usize>().ok()) == Some(n)
    }) || (k > 1 && std::env::var("CMF_DORY_KT_AT").ok().and_then(|v| v.parse::<usize>().ok()) == Some(pos));
    if kt {
        super::super::kt_start();
    }
    let graphs = k == 1 && graphs_on() && !kt;
    let stage_run = |d: &mut Dev, stage: Stage| -> Res<()> {
        if !graphs {
            return run_stage(d, stage);
        }
        let key = (stage as u8, loops, want as u8);
        let sg = sig(d);
        if !d.graphs.get(&key).is_some_and(|g| g.sig == sg) {
            let exec = unsafe { capture(c, || run_stage(d, stage))? };
            d.graphs.insert(key, GraphRec { exec, sig: sg });
        }
        unsafe { curesult(cs::cuGraphLaunch(d.graphs[&key].exec, c.stream), "graph launch") }
    };
    match &g_mode {
        None => stage_run(d, Stage::All)?,
        Some(Gate::Conf(th)) => {
            d.gate_tokens += 1;
            if !d.exits.probe() {
                stage_run(d, Stage::First)?;
                stage_run(d, Stage::SecondFresh)?;
            } else {
                stage_run(d, Stage::FirstRead)?;
                let one = top_prob(&read_logits(c, d, 1)?) >= *th;
                d.exits.note(one);
                d.gate_one += u64::from(one);
                if !one {
                    stage_run(d, Stage::SecondRead)?;
                }
            }
        }
        Some(Gate::Recon(t, bias)) => {
            stage_run(d, Stage::First)?;
            let phi = read_state(c, d)?;
            let one = t.score(&phi) >= *bias;
            d.gate_tokens += 1;
            d.gate_one += u64::from(one);
            stage_run(d, if one { Stage::Read } else { Stage::SecondFresh })?;
        }
    }
    if kt {
        c.sync()?;
        super::super::kt_stop();
        eprintln!("dory cuda: that call took {} passes", if d.gate_one > 0 && g_mode.is_some() && d.gate_tokens > 0 { "1 or 2" } else { "2" });
    }
    if d.sealed && k > 1 {
        // the rows written after the near rows go into the window (the last
        // pass's on the last-loop cache: the earlier ones were written over)
        let hfx = d.hfx.clone();
        let mut st = c.st.lock().unwrap();
        for &(ord, sl) in &d.far {
            let o = st.hyb.o1.get_mut(&(d.id, ord * 16 + sl)).ok_or("a sealed slot without its far field")?;
            let mut f = FarCur::of(o);
            unsafe { f.commit(c, &hfx, k, nk, nk * hd)? };
            f.store(o);
        }
    }
    if g_mode.is_some() && d.gate_tokens.is_multiple_of(64) && std::env::var_os("CMF_DORY_GATE_LOG").is_some() {
        eprintln!("dory gate: {} tokens, {:.1} % on one pass", d.gate_tokens, 100.0 * d.gate_one as f64 / d.gate_tokens as f64);
    }
    match want {
        Want::None => c.sync()?,
        Want::Last => {
            logits.clear();
            logits.extend(read_logits(c, d, 1)?);
        }
        Want::All => logits.extend(read_logits(c, d, k)?),
    }
    Ok(())
}

/// The full slots of the card's layers: (ordinal, slot).
fn full_slots(d: &Dev) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for l in &d.layers {
        if let Ld::Attn(a) = l
            && !a.sliding
        {
            out.extend((0..d.slots[a.ordinal].len()).map(|sl| (a.ordinal, sl)));
        }
    }
    out
}

/// The full layers go over to the far field (`early`: after a long prompt's
/// prefix, the rest still to come): the slots' rows come to the host, the
/// rows the far field would lose are chosen and kept exact at a prompt's end
/// (`pick_exact_rows`), the far fields go up (`far_up`) and the exact rows are
/// freed. After a prefix whose file chooses exact rows the rows go on to an
/// archive, for the far field to be built again over all of them (`rebuild`).
fn seal(c: &Ctx, d: &mut Dev, m: &Dory, o1: crate::dory::DoryO1, rows: usize, early: bool) -> Res<()> {
    let cf = &m.cfg;
    let (nh, nk, hd) = (cf.nh, cf.nk, cf.hd);
    if cf.full_window.is_some() {
        return Err("full layers cut to a window".into());
    }
    let t0 = std::time::Instant::now();
    let kv_row = nk * hd;
    let pull = |b: &Buf, n: usize| -> Res<Vec<f32>> {
        let mut h = vec![0u16; n];
        unsafe { cu::memcpy_dtoh_async(&mut h, b.ptr, c.stream).map_err(derr)? };
        c.sync()?;
        let mut out = vec![0f32; n];
        let per = n.div_ceil(16).max(1 << 16);
        std::thread::scope(|sc| {
            for (o, i) in out.chunks_mut(per).zip(h.chunks(per)) {
                sc.spawn(move || {
                    for (a, b) in o.iter_mut().zip(i) {
                        *a = cortiq_core::quant::f16_to_f32(*b);
                    }
                });
            }
        });
        Ok(out)
    };
    // every full slot's rows and queries on the host
    let mut slots: Vec<SlotRows> = Vec::new();
    for (ord, sl) in full_slots(d) {
        let s = &d.slots[ord][sl];
        if s.cap < rows {
            return Err(format!("a full slot of {} rows sealed at {rows}", s.cap));
        }
        let (ks, vs) = (pull(&s.k, rows * kv_row)?, pull(&s.v, rows * kv_row)?);
        let (tr, have) = d.qtrace.get(&(ord, sl)).ok_or("a full slot without its queries")?;
        let nq = rows.div_ceil(QT_STRIDE) * nh * hd;
        if *have < nq {
            return Err("a full slot's queries short of its rows".into());
        }
        let mut qs = vec![0f32; nq];
        unsafe { cu::memcpy_dtoh_async(&mut qs, tr.ptr, c.stream).map_err(derr)? };
        c.sync()?;
        slots.push((ord, sl, ks, vs, qs));
    }
    let t_pull = t0.elapsed().as_secs_f64();
    let keep: Vec<usize> = match (&m.exact, early) {
        (Some(x), false) => {
            let views: Vec<crate::dory::FlatRows> = slots.iter().filter(|s| s.1 == 0).map(|s| crate::dory::FlatRows { k: &s.2, v: &s.3, nk, hd }).collect();
            let layers: Vec<&dyn crate::dory::LayerRows> = views.iter().map(|v| v as &dyn crate::dory::LayerRows).collect();
            crate::dory::pick_exact_rows(x, &layers, rows, nk, hd, o1.m, o1.sink, rows.saturating_sub(o1.w))
        }
        _ => Vec::new(),
    };
    let t_pick = t0.elapsed().as_secs_f64() - t_pull;
    let order = crate::dory::seal_order(rows, o1.sink, &keep);
    let epoch = d.seal_epoch + 1;
    // the rows are on the host now: they leave the card before the far fields
    // come up (the two never sit there together), with the prompt's score
    // buffers, which a sealed chunk needs far less of
    c.sync()?;
    let far: Vec<(usize, usize)> = slots.iter().map(|s| (s.0, s.1)).collect();
    for &(ord, sl) in &far {
        let s = &mut d.slots[ord][sl];
        let n = Slot::new(0, 0, kv_row)?;
        c.kv_bytes.fetch_add(n.bytes(), Ordering::Relaxed);
        c.kv_bytes.fetch_sub(s.bytes(), Ordering::Relaxed);
        *s = n;
    }
    d.qtrace.clear();
    if let Some(bg) = d.big.as_mut() {
        bg.bs = Buf::new(64)?;
        bg.bp = Buf::new(64)?;
        bg.bcap = 0;
    }
    {
        let refs: Vec<RowsRef> = slots.iter().map(|(o, sl, k, v, q)| (*o, *sl, &k[..], &v[..], &q[..])).collect();
        if let Err(e) = far_up(c, d, m, o1, &refs, rows, &order, keep.len(), epoch) {
            far_drop(c, d);
            // the layers stay exact: their rows go back
            restore_slots(c, d, &slots, rows, kv_row)?;
            return Err(e);
        }
    }
    let t_far = t0.elapsed().as_secs_f64();
    d.far = far;
    d.sealed = true;
    d.seal_epoch = epoch;
    if early && m.exact.is_some() {
        // the prefix's rows open the archive the prompt's rows go on to
        let dir = m.spill.as_deref();
        let qs_w = nh * hd;
        let per = (2 * BIG * kv_row * 2 + (BIG / QT_STRIDE + 1) * qs_w * 4).next_multiple_of(256);
        let opened = (|| -> Result<Archive, String> {
            let mut out = Vec::with_capacity(slots.len());
            for (ord, sl, ks, vs, qs) in slots {
                let name = |kind: &str| format!("dory-cuda{}-{ord}-{sl}-{kind}", d.id);
                let put = |kind: &str, rows: Vec<f32>| -> Result<crate::pipeline::ArchStore, String> {
                    if dir.is_none() {
                        return Ok(crate::pipeline::ArchStore::Ram(rows));
                    }
                    let mut s = crate::pipeline::ArchStore::new(dir, name(kind))?;
                    s.push(&rows)?;
                    Ok(s)
                };
                out.push(ArchSlot { ord, sl, k: put("k", ks)?, v: put("v", vs)?, q: put("q", qs)? });
            }
            let staging = Buf::new(out.len().max(1) * per).map_err(|e| e.to_string())?;
            Ok(Archive { slots: out, rows, staging, per })
        })();
        match opened {
            Ok(a) => d.arch = Some(a),
            Err(e) => give_up_rows(d, &e),
        }
    }
    hyb_mem(c, &format!("dory's seal of {rows} rows"));
    tracing::info!(
        "dory on CUDA: far field over {rows} rows{}, {} kept exact (rows read {t_pull:.2} s, chosen {t_pick:.2} s, far fields by {t_far:.2} s), {:.2} s in all",
        if d.arch.is_some() { ", the rest to come past it" } else { "" },
        keep.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// A failed seal's rows back on the card, as they were before it.
fn restore_slots(c: &Ctx, d: &mut Dev, slots: &[SlotRows], rows: usize, kv_row: usize) -> Res<()> {
    let h16 = cortiq_core::quant::f32_to_f16;
    for (ord, sl, ks, vs, qs) in slots {
        let n = Slot::new(rows.max(4096), 0, kv_row)?;
        let (kh, vh): (Vec<u16>, Vec<u16>) = (ks.iter().map(|&x| h16(x)).collect(), vs.iter().map(|&x| h16(x)).collect());
        unsafe {
            cu::memcpy_htod_async(n.k.ptr, &kh, c.stream).map_err(derr)?;
            cu::memcpy_htod_async(n.v.ptr, &vh, c.stream).map_err(derr)?;
        }
        let q = Buf::from_slice(qs)?;
        c.sync()?;
        c.kv_bytes.fetch_add(n.bytes(), Ordering::Relaxed);
        let s = &mut d.slots[*ord][*sl];
        c.kv_bytes.fetch_sub(s.bytes(), Ordering::Relaxed);
        *s = n;
        d.qtrace.insert((*ord, *sl), (q, qs.len()));
    }
    Ok(())
}

/// The far field built again over every row of the archive, the rows it
/// would lose chosen and kept exact (they leave it), before a prompt's last
/// rows. On a failure the far field stays as first sealed, with no exact rows.
fn rebuild(c: &Ctx, d: &mut Dev, m: &Dory, o1: crate::dory::DoryO1, rows: usize) {
    let Some(a) = d.arch.take() else { return };
    if let Err(e) = rebuild_(c, d, m, o1, rows, a) {
        tracing::warn!("dory on CUDA: no far field over the whole prompt ({e}); it stays as sealed after the prefix, with no exact rows");
    }
}

fn rebuild_(c: &Ctx, d: &mut Dev, m: &Dory, o1: crate::dory::DoryO1, rows: usize, a: Archive) -> Res<()> {
    let cf = &m.cfg;
    let (nh, nk, hd) = (cf.nh, cf.nk, cf.hd);
    let t0 = std::time::Instant::now();
    if a.rows != rows {
        return Err(format!("the archive holds {} rows of {rows}", a.rows));
    }
    drop(a.staging);
    let mut frozen = Vec::with_capacity(a.slots.len());
    for s in a.slots {
        frozen.push((s.ord, s.sl, s.k.freeze()?, s.v.freeze()?, s.q.freeze()?));
    }
    let kv_row = nk * hd;
    for f in &frozen {
        if f.2.as_slice().len() != rows * kv_row || f.3.as_slice().len() != rows * kv_row || f.4.as_slice().len() != rows.div_ceil(QT_STRIDE) * nh * hd {
            return Err(format!("the archive of layer {}: {} {} {} values for {rows} rows", f.0, f.2.as_slice().len(), f.3.as_slice().len(), f.4.as_slice().len()));
        }
    }
    // the chosen rows leave the far field: a copy of a fact there drowns the exact one
    let keep = match &m.exact {
        Some(x) => {
            let views: Vec<crate::dory::FlatRows> = frozen.iter().filter(|e| e.1 == 0).map(|e| crate::dory::FlatRows { k: e.2.as_slice(), v: e.3.as_slice(), nk, hd }).collect();
            let layers: Vec<&dyn crate::dory::LayerRows> = views.iter().map(|v| v as &dyn crate::dory::LayerRows).collect();
            crate::dory::pick_exact_rows(x, &layers, rows, nk, hd, o1.m, o1.sink, rows.saturating_sub(o1.w))
        }
        None => Vec::new(),
    };
    let t_pick = t0.elapsed().as_secs_f64();
    let order = crate::dory::seal_order(rows, o1.sink, &keep);
    let epoch = d.seal_epoch + 1;
    let refs: Vec<RowsRef> = frozen.iter().map(|(o, sl, k, v, q)| (*o, *sl, k.as_slice(), v.as_slice(), q.as_slice())).collect();
    far_up(c, d, m, o1, &refs, rows, &order, keep.len(), epoch)?;
    d.seal_epoch = epoch;
    hyb_mem(c, &format!("dory's far field rebuilt over {rows} rows"));
    tracing::info!("dory on CUDA: far field rebuilt over {rows} rows, {} kept exact, chosen in {t_pick:.2} s, {:.2} s in all", keep.len(), t0.elapsed().as_secs_f64());
    Ok(())
}

/// A full slot's rows for its far field: ordinal, slot, keys, values, queries.
type RowsRef<'a> = (usize, usize, &'a [f32], &'a [f32], &'a [f32]);

/// The far fields of full slots over `rows` rows taken in `order` (`kept`
/// rows after the sinks stay exact), up on the card under `epoch`. A slot's
/// rows are `[row][kv head][hd]`, its queries one row in QT_STRIDE. The
/// landmarks and skeletons come from the rows by index on the host (the
/// slots side by side, `seal_gathered_rows`), the far sums from the card
/// (`far_sums`, `CMF_O1_FAR_GPU=0`: the host's replay); a slot the gathered
/// seal does not take seals over ordered copies (`far_states`).
#[allow(clippy::too_many_arguments)]
fn far_up(c: &Ctx, d: &Dev, m: &Dory, o1: crate::dory::DoryO1, slots: &[RowsRef], rows: usize, order: &[u32], kept: usize, epoch: u64) -> Res<()> {
    let cf = &m.cfg;
    let (nh, nk, hd) = (cf.nh, cf.nk, cf.hd);
    let hpk = nh / nk;
    let kv_row = nk * hd;
    let t0 = std::time::Instant::now();
    type Gathered = (Vec<crate::nystrom::NystromState>, Option<std::ops::Range<usize>>);
    let states: Vec<Gathered> = std::thread::scope(|sc| {
        let hs: Vec<_> = slots
            .iter()
            .map(|&(_, _, ks, vs, qs)| {
                sc.spawn(move || -> Gathered {
                    let mut groups = Vec::with_capacity(nk);
                    let mut range: Option<std::ops::Range<usize>> = None;
                    for g in 0..nk {
                        let mut st = crate::nystrom::NystromState::new_group(o1.m, o1.w, o1.sink + kept, hpk);
                        let r = st.seal_gathered_rows(&ks[g * hd..], kv_row, &vs[g * hd..], kv_row, qs, QT_STRIDE, nh, g * hpk, order, rows, hd, hd);
                        match (r, &range) {
                            (Some(r), None) if g == 0 => range = Some(r),
                            (Some(r), Some(r0)) if r == *r0 => {}
                            _ => return (crate::dory::far_states(cf, o1, ks, vs, qs, QT_STRIDE, rows, order, kept), None),
                        }
                        groups.push(st);
                    }
                    (groups, range)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("a far state's thread")).collect()
    });
    let t_host = t0.elapsed().as_secs_f64();
    let card = crate::gpu::o1_far_sums_on();
    for (&(ord, sl, ks, vs, _), (mut groups, range)) in slots.iter().zip(states) {
        if let Some(range) = range {
            let (m_eff, _, scale) = groups[0].far_geometry();
            let mut done = false;
            if card && groups.iter().all(|g| g.far_geometry().0 == m_eff) {
                let mut qt = Vec::with_capacity(nh * m_eff * hd);
                for g in &groups {
                    for h in 0..hpk {
                        qt.extend_from_slice(g.far_q_tilde(h));
                    }
                }
                let kg: Vec<&[f32]> = (0..nk).map(|g| &ks[g * hd..]).collect();
                let vg: Vec<&[f32]> = (0..nk).map(|g| &vs[g * hd..]).collect();
                match far_sums(c, nk, hpk, hd, m_eff, scale, &qt, &order[range.clone()], &kg, &vg, kv_row) {
                    Ok((mz, th)) => {
                        for (gi, g) in groups.iter_mut().enumerate() {
                            for h in 0..hpk {
                                let hh = gi * hpk + h;
                                let z = &mz[hh * 2 * m_eff..(hh + 1) * 2 * m_eff];
                                g.set_far_sums(h, &z[..m_eff], &z[m_eff..], &th[hh * m_eff * hd..(hh + 1) * m_eff * hd], range.len());
                            }
                        }
                        done = true;
                    }
                    Err(e) => tracing::warn!("dory on CUDA: far sums on the host ({e})"),
                }
            }
            if !done {
                let idx = &order[range];
                std::thread::scope(|sc| {
                    for (gi, g) in groups.iter_mut().enumerate() {
                        let (k, v) = (&ks[gi * hd..], &vs[gi * hd..]);
                        sc.spawn(move || g.replay_far_idx_rows(k, kv_row, v, kv_row, idx));
                    }
                });
            }
        }
        let views: Vec<_> = groups.iter().map(|n| n.device_view()).collect();
        let mut st = c.st.lock().unwrap();
        o1_ensure(c, &mut st, (d.id, ord * 16 + sl), &views, epoch, 4, nk, hd)?;
    }
    if std::env::var_os("CMF_PREFILL_PROF").is_some() {
        eprintln!("dory seal: {} slots over {rows} rows, landmarks {t_host:.2} s, far sums and uploads {:.2} s ({})", slots.len(), t0.elapsed().as_secs_f64() - t_host, if card { "card" } else { "host" });
    }
    Ok(())
}

/// A seal that failed half way: the far fields it put up go.
fn far_drop(c: &Ctx, d: &Dev) {
    let mut st = c.st.lock().unwrap();
    for (ord, sl) in full_slots(d) {
        if let Some(o) = st.hyb.o1.remove(&(d.id, ord * 16 + sl)) {
            c.kv_bytes.fetch_sub(o.bytes, Ordering::Relaxed);
        }
    }
}

/// `CMF_CUDA_MEM=1`: what the card holds.
fn hyb_mem(c: &Ctx, what: &str) {
    let st = c.st.lock().unwrap();
    mem_report(c, &st, what);
}

/// A new sequence: the far fields go, and exact slots come back on demand.
pub(crate) fn reset(id: u64) {
    DECLINED.lock().unwrap().retain(|&x| x != id);
    let Some(c) = super::super::ctx() else { return };
    let mut g = DEVICES.lock().unwrap();
    let Some(d) = g.as_mut().and_then(|g| g.get_mut(&id)) else { return };
    if c.current().is_err() {
        return;
    }
    if d.sealed {
        let mut st = c.st.lock().unwrap();
        for &(ord, sl) in &d.far {
            if let Some(o) = st.hyb.o1.remove(&(id, ord * 16 + sl)) {
                c.kv_bytes.fetch_sub(o.bytes, Ordering::Relaxed);
            }
        }
        drop(st);
        for row in d.slots.iter_mut() {
            for s in row.drain(..) {
                c.kv_bytes.fetch_sub(s.bytes(), Ordering::Relaxed);
            }
        }
        d.far.clear();
        d.sealed = false;
    }
    d.arch = None;
    d.qtrace.clear();
    d.o1_refused = false;
}

/// Has this sequence gone over to the far field?
pub(crate) fn sealed(id: u64) -> bool {
    DEVICES.lock().unwrap().as_ref().and_then(|g| g.get(&id)).is_some_and(|d| d.sealed)
}

/// The K/V bytes of a sequence on the card (exact rows and far fields).
pub(crate) fn kv_bytes(id: u64) -> Option<u64> {
    let c = super::super::ctx()?;
    let g = DEVICES.lock().unwrap();
    let d = g.as_ref()?.get(&id)?;
    let exact: u64 = d.slots.iter().flatten().map(|s| (s.k.bytes + s.v.bytes) as u64).sum();
    let st = c.st.lock().unwrap();
    let far: u64 = d.far.iter().filter_map(|&(o, sl)| st.hyb.o1.get(&(id, o * 16 + sl)).map(|x| x.bytes)).sum();
    Some(exact + far)
}
