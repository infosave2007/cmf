//! EmbeddingGemma 2 on Metal: the text encoder's 24 layers and the vision
//! tower's 16, each forward of a packed batch in one command buffer, f32
//! throughout.
//!
//! * **Text** ([`TextGpu`]): the host hands over the merged input rows
//!   (`E[id]·√512`, media soft tokens in their placeholder rows) and the
//!   sequence lengths; the device runs every layer and returns the
//!   mean-pooled, final-normed hidden state per sequence (`[nseq, 512]`);
//!   the 512→768 head and the L2 norm stay on the host.
//! * **Vision** ([`VisionGpu`]): the host embeds the patches (one GEMM and
//!   the position tables) and builds the 2-D RoPE rows; the device runs the
//!   encoder layers in place; pooling and `embed_vision` stay on the host.
//! * **Weights** are the encoders' own f32 matrices, wrapped without a copy
//!   (`newBufferWithBytesNoCopy` over the host allocation — large mallocs are
//!   page aligned on macOS; an unaligned one is copied once). The CPU path
//!   keeps working from the same memory.
//! * **GEMM** `eg_mm_nt` / `eg_mm_nn`: 64×64×32 tiles, f32 operands staged
//!   as dense 8×8 blocks, the next tile prefetched in registers (2.6-2.8
//!   TF/s on the M4's 10-core GPU; the f32 MMA ceiling there is ~3.4). One
//!   dispatch takes an item table (one item per grid z), so the three q/k/v
//!   projections are one dispatch and so is every (sequence, head, query
//!   block) of an attention. Its sums run in the CPU's order: on random
//!   operands it returns Accelerate's sgemm bit for bit.
//! * **Attention** is two GEMMs and a masked softmax: `S = Q·Kᵀ` per item
//!   (keys limited to the sliding window's span), the softmax in place
//!   (zeros outside the window and in the row padding), `O = P·V` straight
//!   into the attention output's head columns. Items are grouped into
//!   rounds that fit the score buffer (`S_BUDGET` floats).
//! * **Row ops** (RMS norms, the post-norm residual add fused with the next
//!   pre-norm, GELU-tanh gating, q/k norm + RoPE) are a simdgroup per row
//!   (per head for q/k/v).
//!
//! The device forwards equal the CPU forwards up to summation order (checked
//! by the parity tests, which run whichever path is the default, and against
//! the host path in-process). `CMF_EGEMMA2_GPU=0` keeps the CPU path;
//! `CMF_EGEMMA2_GPU_PROF=1` splits a forward into one command buffer per op
//! class and prints their GPU times.

use super::Ctx;
use metal::{
    Buffer, CommandBuffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions,
    MTLSize,
};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

const EMSL: &str = include_str!("egemma2_msl.metal");

/// Score-buffer budget per attention round, in floats (128 MB).
const S_BUDGET: usize = 32 << 20;
/// Query rows per attention item.
const QB: usize = 512;

struct Pipes {
    mm_nt: ComputePipelineState,
    mm_nn: ComputePipelineState,
    rms: ComputePipelineState,
    add_norm: ComputePipelineState,
    gelu_mul: ComputePipelineState,
    qkv_prep: ComputePipelineState,
    softmax: ComputePipelineState,
    row_inv: ComputePipelineState,
    pool: ComputePipelineState,
    flash64: ComputePipelineState,
    flash64s: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn msl_source() -> Result<String, String> {
    // `CMF_EGEMMA2_MSL=<file>`: kernel development without a rebuild
    match std::env::var("CMF_EGEMMA2_MSL") {
        Ok(path) => std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}")),
        Err(_) => Ok(EMSL.to_string()),
    }
}

fn pipes(c: &Ctx) -> Result<&'static Pipes, String> {
    PIPES
        .get_or_init(|| {
            let opts = metal::CompileOptions::new();
            opts.set_language_version(metal::MTLLanguageVersion::V3_0);
            let lib = c
                ._device
                .new_library_with_source(&msl_source()?, &opts)
                .map_err(|e| format!("egemma2 MSL compile: {e}"))?;
            let pso = |name: &str| -> Result<ComputePipelineState, String> {
                let f = lib
                    .get_function(name, None)
                    .map_err(|e| format!("kernel {name}: {e}"))?;
                c._device
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(|e| format!("pipeline {name}: {e}"))
            };
            // the fragment layout the GEMM epilogue assumes
            {
                let probe = pso("eg_fragprobe")?;
                let out = c
                    ._device
                    .new_buffer(128 * 4, MTLResourceOptions::StorageModeShared);
                let cmd = c.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&probe);
                enc.set_buffer(0, Some(&out), 0);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                let v = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, 128) };
                for lane in 0..32 {
                    let r = &v[lane * 4..lane * 4 + 4];
                    if r[0] != r[2] || r[1] != r[3] {
                        return Err(format!(
                            "unexpected simdgroup fragment layout (lane {lane}: {r:?})"
                        ));
                    }
                }
            }
            Ok(Pipes {
                mm_nt: pso("eg_mm_nt")?,
                mm_nn: pso("eg_mm_nn")?,
                rms: pso("eg_rms")?,
                add_norm: pso("eg_add_norm")?,
                gelu_mul: pso("eg_gelu_mul")?,
                qkv_prep: pso("eg_qkv_prep")?,
                softmax: pso("eg_softmax")?,
                row_inv: pso("eg_row_inv")?,
                pool: pso("eg_pool")?,
                flash64: pso("eg_flash64")?,
                flash64s: pso("eg_flash64s")?,
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

// ───────────────────────────── kernel parameters ─────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct EgMm {
    n: u32,
    m: u32,
    k: u32,
    ldx: u32,
    ldw: u32,
    ldy: u32,
    x_off: u32,
    w_off: u32,
    y_off: u32,
    alpha: f32,
    wrows: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EgRow {
    d: u32,
    ldx: u32,
    ldy: u32,
    eps: f32,
    pre: f32,
    post: f32,
    has_w: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EgAct {
    d: u32,
    ldg: u32,
    ldu: u32,
    ldo: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EgQkv {
    ld: u32,
    hd: u32,
    nq: u32,
    nkv: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    eps: f32,
    /// RoPE section: `rotate_half` within each `sec` channels of a head
    sec: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EgFlash {
    ld: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    ldo: u32,
    nq: u32,
    group: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct EgAtt {
    s_off: u32,
    nq: u32,
    nk: u32,
    lds: u32,
    q0: u32,
    k0: u32,
    len: u32,
    window: u32,
}

fn set_p<T>(enc: &ComputeCommandEncoderRef, idx: u64, v: &T) {
    enc.set_bytes(
        idx,
        std::mem::size_of::<T>() as u64,
        v as *const T as *const c_void,
    );
}

// ───────────────────────────── buffers ─────────────────────────────

/// A host f32 slice as a device buffer: wrapped in place when it starts on
/// a page (macOS hands large allocations whole pages; the wrap's length is
/// rounded up to the page, inside the same allocation), copied otherwise.
fn wrap(c: &Ctx, v: &[f32]) -> Buffer {
    let page = super::page_size();
    let addr = v.as_ptr() as usize;
    let bytes = v.len() * 4;
    if bytes >= page && addr.is_multiple_of(page) {
        return c._device.new_buffer_with_bytes_no_copy(
            v.as_ptr() as *const c_void,
            (bytes.div_ceil(page) * page) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
    }
    c._device.new_buffer_with_data(
        v.as_ptr() as *const c_void,
        bytes.max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn write<T: Copy>(b: &Buffer, v: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(v.as_ptr(), b.contents() as *mut T, v.len());
    }
}

/// Scratch buffers, grown on demand and kept; one set for every forward
/// (text and vision programs never overlap: the lock is held for one).
struct Scratch {
    bufs: Vec<(Buffer, usize)>,
}

static SCRATCH: Mutex<Scratch> = Mutex::new(Scratch { bufs: Vec::new() });

impl Scratch {
    fn get(&mut self, c: &Ctx, slot: usize, bytes: usize) -> Buffer {
        while self.bufs.len() <= slot {
            self.bufs.push((
                c._device
                    .new_buffer(16, MTLResourceOptions::StorageModeShared),
                16,
            ));
        }
        if self.bufs[slot].1 < bytes {
            let b = bytes.next_power_of_two().max(bytes);
            self.bufs[slot] = (
                c._device
                    .new_buffer(b as u64, MTLResourceOptions::StorageModeShared),
                b,
            );
        }
        self.bufs[slot].0.clone()
    }
}

// scratch slots
const B_X: usize = 0;
const B_H: usize = 1;
const B_A: usize = 2;
const B_QKV: usize = 3;
const B_ATT: usize = 4;
const B_T: usize = 5;
const B_GU: usize = 6;
const B_M: usize = 7;
const B_PIN: usize = 8;
const B_Z: usize = 9;
const B_S: usize = 10;
const B_POS: usize = 11;
const B_SEG: usize = 12;
const B_INV: usize = 13;
const B_OUT: usize = 14;
const B_ITEMS: usize = 15;
const B_COS: usize = 16;
const B_SIN: usize = 17;

/// `CMF_EGEMMA2_GPU=0` turns the device paths off.
pub fn enabled() -> bool {
    std::env::var("CMF_EGEMMA2_GPU").as_deref() != Ok("0")
}

fn prof_on() -> bool {
    std::env::var("CMF_EGEMMA2_GPU_PROF").as_deref() == Ok("1")
}

/// Command-buffer recorder: one encoder; under `CMF_EGEMMA2_GPU_PROF=1` a
/// buffer per op class so each class's GPU time can be read.
struct Rec<'a> {
    c: &'a Ctx,
    prof: bool,
    done: Vec<(&'static str, CommandBuffer)>,
    cur: Option<(CommandBuffer, metal::ComputeCommandEncoder, &'static str)>,
}

impl<'a> Rec<'a> {
    fn new(c: &'a Ctx) -> Self {
        Rec {
            c,
            prof: prof_on(),
            done: Vec::new(),
            cur: None,
        }
    }

    fn close(&mut self) {
        if let Some((cmd, enc, label)) = self.cur.take() {
            enc.end_encoding();
            cmd.commit();
            self.done.push((label, cmd));
        }
    }

    fn enc(&mut self, label: &'static str) -> &ComputeCommandEncoderRef {
        if self.prof && self.cur.as_ref().is_some_and(|(_, _, l)| *l != label) {
            self.close();
        }
        if self.cur.is_none() {
            let cmd = self.c.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder().to_owned();
            self.cur = Some((cmd, enc, label));
        }
        &self.cur.as_ref().unwrap().1
    }

    /// Commit what is left and wait for all of it; the GPU ms per class.
    fn finish(mut self) -> Result<Vec<(&'static str, f64)>, String> {
        self.close();
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        for (label, cmd) in &self.done {
            cmd.wait_until_completed();
            if cmd.status() != metal::MTLCommandBufferStatus::Completed {
                return Err(format!(
                    "egemma2 command buffer failed ({:?})",
                    cmd.status()
                ));
            }
            let ms = super::cmd_gpu_ms(cmd);
            match out.iter_mut().find(|(l, _)| l == label) {
                Some(e) => e.1 += ms,
                None => out.push((label, ms)),
            }
        }
        Ok(out)
    }
}

/// Item tables of one forward, packed into one shared buffer.
struct Arena {
    buf: Buffer,
    cap: usize,
    off: usize,
}

impl Arena {
    fn push<T: Copy>(&mut self, items: &[T]) -> Result<u64, String> {
        let bytes = std::mem::size_of_val(items);
        let at = self.off.div_ceil(256) * 256;
        if at + bytes > self.cap {
            return Err("egemma2: item arena overflow".into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                items.as_ptr() as *const u8,
                (self.buf.contents() as *mut u8).add(at),
                bytes,
            );
        }
        self.off = at + bytes;
        Ok(at as u64)
    }
}

// ───────────────────────────── attention plans ─────────────────────────────

/// One attention item: a block of one sequence's queries of one head.
#[derive(Clone, Copy)]
struct AttItem {
    s0: usize,
    h: usize,
    kvh: usize,
    q0: usize,
    nq: usize,
    k0: usize,
    nk: usize,
}

type Rounds = Vec<Vec<(EgAtt, AttItem)>>;

/// The items of one layer layout over `segs` (start, len), grouped into
/// rounds that fit the score buffer; and the floats the largest round needs.
fn att_plan(
    segs: &[(usize, usize)],
    nq_heads: usize,
    nkv: usize,
    window: usize,
) -> (Rounds, usize) {
    let group = nq_heads / nkv;
    let mut rounds: Rounds = vec![Vec::new()];
    let (mut used, mut need) = (0usize, 0usize);
    for &(s0, len) in segs {
        for h in 0..nq_heads {
            let mut q0 = 0usize;
            while q0 < len {
                let nq = QB.min(len - q0);
                let (k0, k1) = if window == 0 {
                    (0, len)
                } else {
                    (
                        q0.saturating_sub(window),
                        (q0 + nq - 1 + window + 1).min(len),
                    )
                };
                let nk = k1 - k0;
                let lds = nk.div_ceil(32) * 32;
                let sz = nq * lds;
                if used > 0 && used + sz > S_BUDGET {
                    rounds.push(Vec::new());
                    used = 0;
                }
                rounds.last_mut().unwrap().push((
                    EgAtt {
                        s_off: used as u32,
                        nq: nq as u32,
                        nk: nk as u32,
                        lds: lds as u32,
                        q0: q0 as u32,
                        k0: k0 as u32,
                        len: len as u32,
                        window: window as u32,
                    },
                    AttItem {
                        s0,
                        h,
                        kvh: h / group,
                        q0,
                        nq,
                        k0,
                        nk,
                    },
                ));
                used += sz;
                need = need.max(used);
                q0 += nq;
            }
        }
    }
    (rounds, need)
}

/// Item-table bytes a forward of `layers` dispatch sets needs: per layer up
/// to `gemms` GEMM dispatches (≤ 3 items each) and, per attention round,
/// three tables of the round's items.
fn arena_bytes(layers: &[&Rounds], gemms: usize) -> usize {
    let mm = std::mem::size_of::<EgMm>();
    let att = std::mem::size_of::<EgAtt>();
    layers
        .iter()
        .map(|rounds| {
            gemms * (3 * mm + 256)
                + rounds
                    .iter()
                    .map(|r| r.len() * (2 * mm + att) + 3 * 256)
                    .sum::<usize>()
        })
        .sum::<usize>()
        + 4096
}

// ───────────────────────────── the program builder ─────────────────────────────

/// One device program (a forward) being encoded.
struct Prog<'a> {
    p: &'static Pipes,
    rec: Rec<'a>,
    arena: Arena,
    eps: f32,
}

fn proj(n: usize, m: usize, k: usize, ldx: usize, ldy: usize, y_off: usize) -> EgMm {
    EgMm {
        n: n as u32,
        m: m as u32,
        k: k as u32,
        ldx: ldx as u32,
        ldw: k as u32,
        ldy: ldy as u32,
        x_off: 0,
        w_off: 0,
        y_off: y_off as u32,
        alpha: 1.0,
        wrows: m as u32,
        pad1: 0,
    }
}

impl<'a> Prog<'a> {
    #[allow(clippy::too_many_arguments)]
    fn mm(
        &mut self,
        nn: bool,
        x: &Buffer,
        ws: [&Buffer; 3],
        y: &Buffer,
        items: &[EgMm],
        label: &'static str,
    ) -> Result<(), String> {
        if items.is_empty() {
            return Ok(());
        }
        let at = self.arena.push(items)?;
        let gx = items.iter().map(|i| i.n as u64).max().unwrap().div_ceil(64);
        let gy = items.iter().map(|i| i.m as u64).max().unwrap().div_ceil(64);
        let p = self.p;
        let enc = self.rec.enc(label);
        enc.set_compute_pipeline_state(if nn { &p.mm_nn } else { &p.mm_nt });
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(ws[0]), 0);
        enc.set_buffer(2, Some(ws[1]), 0);
        enc.set_buffer(3, Some(ws[2]), 0);
        enc.set_buffer(4, Some(y), 0);
        enc.set_buffer(5, Some(&self.arena.buf), at);
        enc.dispatch_thread_groups(
            MTLSize::new(gx, gy, items.len() as u64),
            MTLSize::new(128, 1, 1),
        );
        Ok(())
    }

    fn row(&self, d: usize, n: usize, pre: f32, post: f32, has_w: bool) -> EgRow {
        EgRow {
            d: d as u32,
            ldx: d as u32,
            ldy: d as u32,
            eps: self.eps,
            pre,
            post,
            has_w: has_w as u32,
            n: n as u32,
        }
    }

    /// Y = rms(pre·X)·w
    fn rms(&mut self, x: &Buffer, w: &Buffer, y: &Buffer, d: usize, n: usize, pre: f32) {
        let prm = self.row(d, n, pre, 1.0, true);
        let p = self.p;
        let enc = self.rec.enc("rows");
        enc.set_compute_pipeline_state(&p.rms);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(y), 0);
        set_p(enc, 3, &prm);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(4), 1, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// H = (H + rms(T)·w)·post; then A = rms(H)·w2 when given
    #[allow(clippy::too_many_arguments)]
    fn add_norm(
        &mut self,
        h: &Buffer,
        w: &Buffer,
        t: &Buffer,
        a: &Buffer,
        w2: Option<&Buffer>,
        d: usize,
        n: usize,
        post: f32,
    ) {
        let prm = self.row(d, n, 1.0, post, w2.is_some());
        let p = self.p;
        let enc = self.rec.enc("rows");
        enc.set_compute_pipeline_state(&p.add_norm);
        enc.set_buffer(0, Some(h), 0);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(t), 0);
        enc.set_buffer(3, Some(a), 0);
        enc.set_buffer(4, Some(w2.unwrap_or(w)), 0);
        set_p(enc, 5, &prm);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(4), 1, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// O = gelu_tanh(G)·U over `[n, act.d]`
    fn gelu(&mut self, g: (&Buffer, u64), u: (&Buffer, u64), o: &Buffer, act: EgAct, n: usize) {
        let p = self.p;
        let enc = self.rec.enc("act");
        enc.set_compute_pipeline_state(&p.gelu_mul);
        enc.set_buffer(0, Some(g.0), g.1);
        enc.set_buffer(1, Some(u.0), u.1);
        enc.set_buffer(2, Some(o), 0);
        set_p(enc, 3, &act);
        let w = (act.d as u64).div_ceil(4);
        enc.dispatch_threads(MTLSize::new(w, n as u64, 1), MTLSize::new(w.min(256), 1, 1));
    }

    /// q/k norm + RoPE, v norm, in place on the q|k|v rows
    #[allow(clippy::too_many_arguments)]
    fn qkv(
        &mut self,
        qkv: &Buffer,
        qn: &Buffer,
        kn: &Buffer,
        rope: (&Buffer, &Buffer),
        pos: &Buffer,
        prm: EgQkv,
        n: usize,
    ) {
        let heads = (prm.nq + 2 * prm.nkv) as u64;
        let p = self.p;
        let enc = self.rec.enc("rows");
        enc.set_compute_pipeline_state(&p.qkv_prep);
        enc.set_buffer(0, Some(qkv), 0);
        enc.set_buffer(1, Some(qn), 0);
        enc.set_buffer(2, Some(kn), 0);
        enc.set_buffer(3, Some(rope.0), 0);
        enc.set_buffer(4, Some(rope.1), 0);
        enc.set_buffer(5, Some(pos), 0);
        set_p(enc, 6, &prm);
        enc.dispatch_thread_groups(
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(32 * heads.min(8), 1, 1),
        );
    }

    /// Attention of one layer: `att[:, h·hd..] = softmax(Q_h·K_kvᵀ)·V_kv`
    /// over every item of `rounds`, q|k|v rows of width `ld`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        rounds: &Rounds,
        qkv: &Buffer,
        s: &Buffer,
        att: &Buffer,
        (ld, qw, kw, hd, attw): (usize, usize, usize, usize, usize),
    ) -> Result<(), String> {
        for round in rounds {
            if round.is_empty() {
                continue;
            }
            let s_items: Vec<EgMm> = round
                .iter()
                .map(|(a, it)| EgMm {
                    n: it.nq as u32,
                    m: it.nk as u32,
                    k: hd as u32,
                    ldx: ld as u32,
                    ldw: ld as u32,
                    ldy: a.lds,
                    x_off: ((it.s0 + it.q0) * ld + it.h * hd) as u32,
                    w_off: ((it.s0 + it.k0) * ld + qw + it.kvh * hd) as u32,
                    y_off: a.s_off,
                    alpha: 1.0,
                    wrows: it.nk as u32,
                    pad1: 0,
                })
                .collect();
            self.mm(false, qkv, [qkv, qkv, qkv], s, &s_items, "attn")?;
            let atts: Vec<EgAtt> = round.iter().map(|(a, _)| *a).collect();
            let at = self.arena.push(&atts)?;
            {
                let p = self.p;
                let maxq = round.iter().map(|(_, it)| it.nq).max().unwrap() as u64;
                let enc = self.rec.enc("attn");
                enc.set_compute_pipeline_state(&p.softmax);
                enc.set_buffer(0, Some(s), 0);
                enc.set_buffer(1, Some(&self.arena.buf), at);
                enc.dispatch_thread_groups(
                    MTLSize::new(maxq.div_ceil(4), round.len() as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
            let o_items: Vec<EgMm> = round
                .iter()
                .map(|(a, it)| EgMm {
                    n: it.nq as u32,
                    m: hd as u32,
                    k: a.lds,
                    ldx: a.lds,
                    ldw: ld as u32,
                    ldy: attw as u32,
                    x_off: a.s_off,
                    w_off: ((it.s0 + it.k0) * ld + qw + kw + it.kvh * hd) as u32,
                    y_off: ((it.s0 + it.q0) * attw + it.h * hd) as u32,
                    alpha: 1.0,
                    wrows: it.nk as u32,
                    pad1: 0,
                })
                .collect();
            self.mm(true, s, [qkv, qkv, qkv], att, &o_items, "attn")?;
        }
        Ok(())
    }
}

fn report(what: &str, times: &[(&'static str, f64)], host_ms: f64, wait_ms: f64) {
    if prof_on() || crate::egemma2::prof::on() {
        let total: f64 = times.iter().map(|(_, ms)| ms).sum();
        let split: Vec<String> = times.iter().map(|(l, ms)| format!("{l} {ms:.1}")).collect();
        eprintln!(
            "egemma2 gpu {what} — GPU {total:.1} ms ({}), encode {host_ms:.1} ms, wait {wait_ms:.1} ms",
            split.join(", ")
        );
    }
}

// ───────────────────────────── the text encoder ─────────────────────────────

/// One text layer's host weights, borrowed for the wrap.
pub struct LayerSpec<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub o: &'a [f32],
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
    pub ple_in: &'a [f32],
    pub ple_gate: &'a [f32],
    pub ple_proj: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub in_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub pre_ff_norm: &'a [f32],
    pub post_ff_norm: &'a [f32],
    pub ple_norm: &'a [f32],
    pub head_dim: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub window: Option<usize>,
    pub scalar: f32,
    /// index into [`TextSpec::ropes`]
    pub rope: usize,
}

/// The text encoder's host weights and shapes.
pub struct TextSpec<'a> {
    pub hidden: usize,
    pub inter: usize,
    pub n_ple: usize,
    pub eps: f32,
    pub ple_scale: f32,
    pub ple_norm: &'a [f32],
    pub norm: &'a [f32],
    pub layers: Vec<LayerSpec<'a>>,
    /// RoPE tables `(head_dim, cos, sin)`, each `[max_tokens, head_dim/2]`
    pub ropes: Vec<(usize, Vec<f32>, Vec<f32>)>,
    pub max_tokens: usize,
}

struct GLayer {
    q: Buffer,
    k: Buffer,
    v: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    ple_in: Buffer,
    ple_gate: Buffer,
    ple_proj: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    in_norm: Buffer,
    post_attn_norm: Buffer,
    pre_ff_norm: Buffer,
    post_ff_norm: Buffer,
    ple_norm: Buffer,
    hd: usize,
    nq: usize,
    nkv: usize,
    window: usize,
    scalar: f32,
    rope: usize,
}

/// The device-side text encoder.
pub struct TextGpu {
    layers: Vec<GLayer>,
    ple_norm: Buffer,
    norm: Buffer,
    ropes: Vec<(usize, Buffer, Buffer)>,
    hidden: usize,
    inter: usize,
    n_ple: usize,
    eps: f32,
    ple_scale: f32,
    max_tokens: usize,
}

impl TextGpu {
    /// Wrap the encoder's weights for the device. `Err`: no device, or a
    /// shape this path does not take (the CPU path runs).
    pub fn new(spec: &TextSpec) -> Result<TextGpu, String> {
        let c = super::ctx().ok_or("no Metal device")?;
        pipes(c)?;
        let d = spec.hidden;
        if ![d, spec.inter, spec.n_ple]
            .iter()
            .all(|x| x.is_multiple_of(64))
        {
            return Err(format!(
                "widths {d}/{}/{} are not multiples of 64",
                spec.inter, spec.n_ple
            ));
        }
        let mut layers = Vec::with_capacity(spec.layers.len());
        for (i, l) in spec.layers.iter().enumerate() {
            let hd = l.head_dim;
            if !hd.is_multiple_of(64) || l.kv_heads == 0 || !l.q_heads.is_multiple_of(l.kv_heads) {
                return Err(format!("layer {i}: head layout {}x{hd}", l.q_heads));
            }
            if l.rope >= spec.ropes.len() || spec.ropes[l.rope].0 != hd {
                return Err(format!("layer {i}: no RoPE table for head_dim {hd}"));
            }
            let want = [
                (l.q.len(), l.q_heads * hd * d),
                (l.k.len(), l.kv_heads * hd * d),
                (l.v.len(), l.kv_heads * hd * d),
                (l.o.len(), d * l.q_heads * hd),
                (l.gate.len(), spec.inter * d),
                (l.up.len(), spec.inter * d),
                (l.down.len(), d * spec.inter),
                (l.ple_in.len(), spec.n_ple * d),
                (l.ple_gate.len(), spec.n_ple * d),
                (l.ple_proj.len(), d * spec.n_ple),
            ];
            if want.iter().any(|(a, b)| a != b) {
                return Err(format!("layer {i}: unexpected weight sizes"));
            }
            layers.push(GLayer {
                q: wrap(c, l.q),
                k: wrap(c, l.k),
                v: wrap(c, l.v),
                o: wrap(c, l.o),
                gate: wrap(c, l.gate),
                up: wrap(c, l.up),
                down: wrap(c, l.down),
                ple_in: wrap(c, l.ple_in),
                ple_gate: wrap(c, l.ple_gate),
                ple_proj: wrap(c, l.ple_proj),
                q_norm: wrap(c, l.q_norm),
                k_norm: wrap(c, l.k_norm),
                in_norm: wrap(c, l.in_norm),
                post_attn_norm: wrap(c, l.post_attn_norm),
                pre_ff_norm: wrap(c, l.pre_ff_norm),
                post_ff_norm: wrap(c, l.post_ff_norm),
                ple_norm: wrap(c, l.ple_norm),
                hd,
                nq: l.q_heads,
                nkv: l.kv_heads,
                window: l.window.unwrap_or(0),
                scalar: l.scalar,
                rope: l.rope,
            });
        }
        // copied: the spec's tables are dropped after this
        let copy = |v: &[f32]| {
            c._device.new_buffer_with_data(
                v.as_ptr() as *const c_void,
                (v.len() * 4).max(4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let ropes = spec
            .ropes
            .iter()
            .map(|(hd, cs, sn)| (*hd, copy(cs), copy(sn)))
            .collect();
        Ok(TextGpu {
            layers,
            ple_norm: wrap(c, spec.ple_norm),
            norm: wrap(c, spec.norm),
            ropes,
            hidden: d,
            inter: spec.inter,
            n_ple: spec.n_ple,
            eps: spec.eps,
            ple_scale: spec.ple_scale,
            max_tokens: spec.max_tokens,
        })
    }

    /// The forward over merged rows `x` (`[Σ lens, hidden]`): the
    /// final-normed hidden state mean-pooled per sequence, `[nseq, hidden]`.
    pub fn forward(&self, x: &[f32], lens: &[usize]) -> Result<Vec<f32>, String> {
        let c = super::ctx().ok_or("no Metal device")?;
        let p = pipes(c)?;
        let t_host = std::time::Instant::now();
        let d = self.hidden;
        let n: usize = lens.iter().sum();
        if n == 0 || x.len() != n * d {
            return Err("egemma2 gpu: empty or misshapen input".into());
        }
        if lens.iter().any(|&l| l == 0 || l > self.max_tokens) {
            return Err("egemma2 gpu: a sequence is empty or past the context".into());
        }
        let mut segs = Vec::with_capacity(lens.len());
        let mut pos: Vec<u32> = Vec::with_capacity(n);
        let mut off = 0usize;
        for &l in lens {
            segs.push((off, l));
            pos.extend(0..l as u32);
            off += l;
        }
        let qkv_w = self
            .layers
            .iter()
            .map(|l| (l.nq + 2 * l.nkv) * l.hd)
            .max()
            .unwrap_or(0);
        let att_w = self.layers.iter().map(|l| l.nq * l.hd).max().unwrap_or(0);
        // attention plans, one per distinct (hd, heads, window) layout
        type Key = (usize, usize, usize, usize);
        let mut plans: Vec<(Key, Rounds)> = Vec::new();
        let mut s_need = 0usize;
        for l in &self.layers {
            let key = (l.hd, l.nq, l.nkv, l.window);
            if !plans.iter().any(|(k, _)| *k == key) {
                let (r, need) = att_plan(&segs, l.nq, l.nkv, l.window);
                s_need = s_need.max(need);
                plans.push((key, r));
            }
        }
        let plan_of = |l: &GLayer| -> &Rounds {
            &plans
                .iter()
                .find(|(k, _)| *k == (l.hd, l.nq, l.nkv, l.window))
                .unwrap()
                .1
        };
        let per_layer: Vec<&Rounds> = self.layers.iter().map(plan_of).collect();
        let arena_cap = arena_bytes(&per_layer, 8);

        let mut sc = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        let f = 4usize;
        let bx = sc.get(c, B_X, n * d * f);
        let bh = sc.get(c, B_H, n * d * f);
        let ba = sc.get(c, B_A, n * d * f);
        let bqkv = sc.get(c, B_QKV, n * qkv_w * f);
        let batt = sc.get(c, B_ATT, n * att_w * f);
        let bt = sc.get(c, B_T, n * d * f);
        let bgu = sc.get(c, B_GU, n * 2 * self.inter * f);
        let bm = sc.get(c, B_M, n * self.inter * f);
        let bpin = sc.get(c, B_PIN, n * self.n_ple * f);
        let bz = sc.get(c, B_Z, n * self.n_ple * f);
        let bs = sc.get(c, B_S, s_need.max(1) * f);
        let bpos = sc.get(c, B_POS, n * 4);
        let bseg = sc.get(c, B_SEG, segs.len() * 8);
        let binv = sc.get(c, B_INV, n * f);
        let bout = sc.get(c, B_OUT, segs.len() * d * f);
        let bitems = sc.get(c, B_ITEMS, arena_cap);
        write(&bx, x);
        write(&bh, x);
        write(&bpos, &pos);
        let segw: Vec<u32> = segs
            .iter()
            .flat_map(|&(s0, l)| [s0 as u32, l as u32])
            .collect();
        write(&bseg, &segw);
        let mut g = Prog {
            p,
            rec: Rec::new(c),
            arena: Arena {
                buf: bitems,
                cap: arena_cap,
                off: 0,
            },
            eps: self.eps,
        };

        let (inter, np) = (self.inter, self.n_ple);
        if let Some(l0) = self.layers.first() {
            g.rms(&bh, &l0.in_norm, &ba, d, n, 1.0);
        }
        for (li, l) in self.layers.iter().enumerate() {
            let hd = l.hd;
            let (qw, kw) = (l.nq * hd, l.nkv * hd);
            let ld = qw + 2 * kw;
            // ── attention block (A = rms(H)·in_norm already)
            g.mm(
                false,
                &ba,
                [&l.q, &l.k, &l.v],
                &bqkv,
                &[
                    proj(n, qw, d, d, ld, 0),
                    proj(n, kw, d, d, ld, qw),
                    proj(n, kw, d, d, ld, qw + kw),
                ],
                "gemm",
            )?;
            let (_, rc, rs) = &self.ropes[l.rope];
            g.qkv(
                &bqkv,
                &l.q_norm,
                &l.k_norm,
                (rc, rs),
                &bpos,
                EgQkv {
                    ld: ld as u32,
                    hd: hd as u32,
                    nq: l.nq as u32,
                    nkv: l.nkv as u32,
                    q_off: 0,
                    k_off: qw as u32,
                    v_off: (qw + kw) as u32,
                    eps: self.eps,
                    sec: hd as u32,
                    pad0: 0,
                },
                n,
            );
            g.attention(plan_of(l), &bqkv, &bs, &batt, (ld, qw, kw, hd, qw))?;
            g.mm(
                false,
                &batt,
                [&l.o, &l.o, &l.o],
                &bt,
                &[proj(n, d, qw, qw, d, 0)],
                "gemm",
            )?;
            g.add_norm(
                &bh,
                &l.post_attn_norm,
                &bt,
                &ba,
                Some(&l.pre_ff_norm),
                d,
                n,
                1.0,
            );
            // ── gated FFN
            g.mm(
                false,
                &ba,
                [&l.gate, &l.up, &l.up],
                &bgu,
                &[
                    proj(n, inter, d, d, 2 * inter, 0),
                    proj(n, inter, d, d, 2 * inter, inter),
                ],
                "gemm",
            )?;
            g.gelu(
                (&bgu, 0),
                (&bgu, (inter * 4) as u64),
                &bm,
                EgAct {
                    d: inter as u32,
                    ldg: (2 * inter) as u32,
                    ldu: (2 * inter) as u32,
                    ldo: inter as u32,
                },
                n,
            );
            g.mm(
                false,
                &bm,
                [&l.down, &l.down, &l.down],
                &bt,
                &[proj(n, d, inter, inter, d, 0)],
                "gemm",
            )?;
            g.add_norm(&bh, &l.post_ff_norm, &bt, &ba, None, d, n, 1.0);
            // ── per-layer input
            g.mm(
                false,
                &bx,
                [&l.ple_in, &l.ple_in, &l.ple_in],
                &bpin,
                &[proj(n, np, d, d, np, 0)],
                "gemm",
            )?;
            g.rms(&bpin, &self.ple_norm, &bpin, np, n, self.ple_scale);
            g.mm(
                false,
                &bh,
                [&l.ple_gate, &l.ple_gate, &l.ple_gate],
                &bz,
                &[proj(n, np, d, d, np, 0)],
                "gemm",
            )?;
            let same = EgAct {
                d: np as u32,
                ldg: np as u32,
                ldu: np as u32,
                ldo: np as u32,
            };
            g.gelu((&bz, 0), (&bpin, 0), &bz, same, n);
            g.mm(
                false,
                &bz,
                [&l.ple_proj, &l.ple_proj, &l.ple_proj],
                &bt,
                &[proj(n, d, np, np, d, 0)],
                "gemm",
            )?;
            // and the next layer's input norm
            let next = self.layers.get(li + 1).map(|nl| &nl.in_norm);
            g.add_norm(&bh, &l.ple_norm, &bt, &ba, next, d, n, l.scalar);
        }
        // ── final norm + mean pool
        {
            let inv = g.row(d, n, 1.0, 1.0, false);
            let pool = g.row(d, n, 1.0, 1.0, true);
            let enc = g.rec.enc("rows");
            enc.set_compute_pipeline_state(&p.row_inv);
            enc.set_buffer(0, Some(&bh), 0);
            enc.set_buffer(1, Some(&binv), 0);
            set_p(enc, 2, &inv);
            enc.dispatch_thread_groups(
                MTLSize::new((n as u64).div_ceil(4), 1, 1),
                MTLSize::new(128, 1, 1),
            );
            enc.set_compute_pipeline_state(&p.pool);
            enc.set_buffer(0, Some(&bh), 0);
            enc.set_buffer(1, Some(&binv), 0);
            enc.set_buffer(2, Some(&self.norm), 0);
            enc.set_buffer(3, Some(&bseg), 0);
            enc.set_buffer(4, Some(&bout), 0);
            set_p(enc, 5, &pool);
            enc.dispatch_threads(
                MTLSize::new(d as u64, segs.len() as u64, 1),
                MTLSize::new(128, 1, 1),
            );
        }
        let host_ms = t_host.elapsed().as_secs_f64() * 1e3;
        let t_wait = std::time::Instant::now();
        let times = g.rec.finish()?;
        let mut out = vec![0f32; segs.len() * d];
        unsafe {
            std::ptr::copy_nonoverlapping(
                bout.contents() as *const f32,
                out.as_mut_ptr(),
                out.len(),
            );
        }
        report(
            &format!("text: {n} tokens / {} seqs", segs.len()),
            &times,
            host_ms,
            t_wait.elapsed().as_secs_f64() * 1e3,
        );
        Ok(out)
    }
}

// ───────────────────────────── the vision tower ─────────────────────────────

/// One vision layer's host weights, borrowed for the wrap.
pub struct VLayerSpec<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub o: &'a [f32],
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
    pub in_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub pre_ff_norm: &'a [f32],
    pub post_ff_norm: &'a [f32],
}

struct VLayer {
    q: Buffer,
    k: Buffer,
    v: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    in_norm: Buffer,
    post_attn_norm: Buffer,
    pre_ff_norm: Buffer,
    post_ff_norm: Buffer,
}

/// The device-side vision encoder layers (`gemma4_vision`: Gemma sandwich
/// norms, bidirectional attention per image with 2-D RoPE, gelu-tanh FFN).
pub struct VisionGpu {
    layers: Vec<VLayer>,
    hidden: usize,
    heads: usize,
    hd: usize,
    inter: usize,
    eps: f32,
}

impl VisionGpu {
    pub fn new(
        layers: &[VLayerSpec],
        hidden: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<VisionGpu, String> {
        let c = super::ctx().ok_or("no Metal device")?;
        pipes(c)?;
        let d = hidden;
        let w = heads * head_dim;
        let inter = layers.first().map(|l| l.gate.len() / d.max(1)).unwrap_or(0);
        if ![d, w, inter].iter().all(|x| x.is_multiple_of(64)) || !head_dim.is_multiple_of(32) {
            return Err(format!("vision widths {d}/{w}/{inter}/{head_dim}"));
        }
        let mut out = Vec::with_capacity(layers.len());
        for (i, l) in layers.iter().enumerate() {
            let want = [
                (l.q.len(), w * d),
                (l.k.len(), w * d),
                (l.v.len(), w * d),
                (l.o.len(), d * w),
                (l.gate.len(), inter * d),
                (l.up.len(), inter * d),
                (l.down.len(), d * inter),
                (l.q_norm.len(), head_dim),
                (l.k_norm.len(), head_dim),
            ];
            if want.iter().any(|(a, b)| a != b) {
                return Err(format!("vision layer {i}: unexpected weight sizes"));
            }
            out.push(VLayer {
                q: wrap(c, l.q),
                k: wrap(c, l.k),
                v: wrap(c, l.v),
                o: wrap(c, l.o),
                gate: wrap(c, l.gate),
                up: wrap(c, l.up),
                down: wrap(c, l.down),
                q_norm: wrap(c, l.q_norm),
                k_norm: wrap(c, l.k_norm),
                in_norm: wrap(c, l.in_norm),
                post_attn_norm: wrap(c, l.post_attn_norm),
                pre_ff_norm: wrap(c, l.pre_ff_norm),
                post_ff_norm: wrap(c, l.post_ff_norm),
            });
        }
        Ok(VisionGpu {
            layers: out,
            hidden: d,
            heads,
            hd: head_dim,
            inter,
            eps,
        })
    }

    /// Every encoder layer over the embedded patches `h` (`[Σ lens, hidden]`,
    /// images back to back), in place. `cos` / `sin`: each row's RoPE
    /// angles, `[Σ lens, head_dim/2]` (the column half, then the row half).
    pub fn forward(
        &self,
        h: &mut [f32],
        lens: &[usize],
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(), String> {
        let c = super::ctx().ok_or("no Metal device")?;
        let p = pipes(c)?;
        let t_host = std::time::Instant::now();
        let (d, hd, nh) = (self.hidden, self.hd, self.heads);
        let n: usize = lens.iter().sum();
        let half = hd / 2;
        if n == 0 || h.len() != n * d || cos.len() != n * half || sin.len() != n * half {
            return Err("egemma2 vision gpu: misshapen input".into());
        }
        let mut segs = Vec::with_capacity(lens.len());
        let mut off = 0usize;
        for &l in lens {
            segs.push((off, l));
            off += l;
        }
        let w = nh * hd;
        let ld = 3 * w;
        // head_dim 64: flash attention, no score matrix (the GEMM form
        // writes and reads n² floats a head); `CMF_EGEMMA2_VISION_FLASH=0`
        // keeps the GEMM form
        let flash = hd == 64 && std::env::var("CMF_EGEMMA2_VISION_FLASH").as_deref() != Ok("0");
        let direct_flash = std::env::var("CMF_EGEMMA2_FLASH").as_deref() == Ok("direct");
        let (rounds, s_need) = if flash {
            (Vec::new(), 0)
        } else {
            att_plan(&segs, nh, nh, 0)
        };
        let per_layer: Vec<&Rounds> = self.layers.iter().map(|_| &rounds).collect();
        let arena_cap = arena_bytes(&per_layer, 6);
        let inter = self.inter;
        let mut sc = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        let f = 4usize;
        let bh = sc.get(c, B_H, n * d * f);
        let ba = sc.get(c, B_A, n * d * f);
        // 32 rows of padding: a flash tail block reads past the last image
        let bqkv = sc.get(c, B_QKV, (n + 32) * ld * f);
        let bseg = sc.get(c, B_SEG, segs.len() * 8);
        let segw: Vec<u32> = segs
            .iter()
            .flat_map(|&(s0, l)| [s0 as u32, l as u32])
            .collect();
        write(&bseg, &segw);
        let max_len = lens.iter().copied().max().unwrap_or(0);
        let batt = sc.get(c, B_ATT, n * w * f);
        let bt = sc.get(c, B_T, n * d * f);
        let bgu = sc.get(c, B_GU, n * 2 * inter * f);
        let bm = sc.get(c, B_M, n * inter * f);
        let bs = sc.get(c, B_S, s_need.max(1) * f);
        let bpos = sc.get(c, B_POS, n * 4);
        let bcos = sc.get(c, B_COS, n * half * f);
        let bsin = sc.get(c, B_SIN, n * half * f);
        let bitems = sc.get(c, B_ITEMS, arena_cap);
        write(&bh, h);
        write(&bcos, cos);
        write(&bsin, sin);
        let pos: Vec<u32> = (0..n as u32).collect();
        write(&bpos, &pos);
        let mut g = Prog {
            p,
            rec: Rec::new(c),
            arena: Arena {
                buf: bitems,
                cap: arena_cap,
                off: 0,
            },
            eps: self.eps,
        };
        if let Some(l0) = self.layers.first() {
            g.rms(&bh, &l0.in_norm, &ba, d, n, 1.0);
        }
        for (li, l) in self.layers.iter().enumerate() {
            g.mm(
                false,
                &ba,
                [&l.q, &l.k, &l.v],
                &bqkv,
                &[
                    proj(n, w, d, d, ld, 0),
                    proj(n, w, d, d, ld, w),
                    proj(n, w, d, d, ld, 2 * w),
                ],
                "gemm",
            )?;
            g.qkv(
                &bqkv,
                &l.q_norm,
                &l.k_norm,
                (&bcos, &bsin),
                &bpos,
                EgQkv {
                    ld: ld as u32,
                    hd: hd as u32,
                    nq: nh as u32,
                    nkv: nh as u32,
                    q_off: 0,
                    k_off: w as u32,
                    v_off: (2 * w) as u32,
                    eps: self.eps,
                    // two halves: columns, then rows
                    sec: (hd / 2) as u32,
                    pad0: 0,
                },
                n,
            );
            if flash {
                // staged K/V shared by 8 simdgroups (64 queries a group);
                // `CMF_EGEMMA2_FLASH=direct`: 4 simdgroups reading K/V each
                let (pso, q_tile, threads) = if direct_flash {
                    (&p.flash64, 32u64, 128u64)
                } else {
                    (&p.flash64s, 64, 256)
                };
                let enc = g.rec.enc("attn");
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(&bqkv), 0);
                enc.set_buffer(1, Some(&batt), 0);
                enc.set_buffer(2, Some(&bseg), 0);
                set_p(
                    enc,
                    3,
                    &EgFlash {
                        ld: ld as u32,
                        q_off: 0,
                        k_off: w as u32,
                        v_off: (2 * w) as u32,
                        ldo: w as u32,
                        nq: nh as u32,
                        group: 1,
                        scale: 1.0,
                    },
                );
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        (max_len as u64).div_ceil(q_tile),
                        nh as u64,
                        segs.len() as u64,
                    ),
                    MTLSize::new(threads, 1, 1),
                );
            } else {
                g.attention(&rounds, &bqkv, &bs, &batt, (ld, w, w, hd, w))?;
            }
            g.mm(
                false,
                &batt,
                [&l.o, &l.o, &l.o],
                &bt,
                &[proj(n, d, w, w, d, 0)],
                "gemm",
            )?;
            g.add_norm(
                &bh,
                &l.post_attn_norm,
                &bt,
                &ba,
                Some(&l.pre_ff_norm),
                d,
                n,
                1.0,
            );
            g.mm(
                false,
                &ba,
                [&l.gate, &l.up, &l.up],
                &bgu,
                &[
                    proj(n, inter, d, d, 2 * inter, 0),
                    proj(n, inter, d, d, 2 * inter, inter),
                ],
                "gemm",
            )?;
            g.gelu(
                (&bgu, 0),
                (&bgu, (inter * 4) as u64),
                &bm,
                EgAct {
                    d: inter as u32,
                    ldg: (2 * inter) as u32,
                    ldu: (2 * inter) as u32,
                    ldo: inter as u32,
                },
                n,
            );
            g.mm(
                false,
                &bm,
                [&l.down, &l.down, &l.down],
                &bt,
                &[proj(n, d, inter, inter, d, 0)],
                "gemm",
            )?;
            let next = self.layers.get(li + 1).map(|nl| &nl.in_norm);
            g.add_norm(&bh, &l.post_ff_norm, &bt, &ba, next, d, n, 1.0);
        }
        let host_ms = t_host.elapsed().as_secs_f64() * 1e3;
        let t_wait = std::time::Instant::now();
        let times = g.rec.finish()?;
        unsafe {
            std::ptr::copy_nonoverlapping(bh.contents() as *const f32, h.as_mut_ptr(), n * d);
        }
        report(
            &format!("vision: {n} patches / {} images", segs.len()),
            &times,
            host_ms,
            t_wait.elapsed().as_secs_f64() * 1e3,
        );
        Ok(())
    }
}

// ───────────────────────────── kernel development ─────────────────────────────

/// GEMM micro-benchmark (kernel development): `y = x·wᵀ` with random
/// operands, `n×k · (m×k)ᵀ`, `reps` timed runs of the NT kernel named
/// `kernel` (default `eg_mm_nt`; any kernel with its signature, from
/// `CMF_EGEMMA2_MSL` when set; `CMF_EGEMMA2_TILE=<rows>x<features>` and
/// `CMF_EGEMMA2_THREADS` set its grid). Returns (best TF/s, max |Δ| /
/// max |y| against Accelerate).
#[doc(hidden)]
pub fn bench_gemm(
    n: usize,
    k: usize,
    m: usize,
    reps: usize,
    kernel: Option<&str>,
) -> Result<(f64, f64), String> {
    let c = super::ctx().ok_or("no Metal device")?;
    let p = pipes(c)?;
    let custom;
    let pso = match kernel {
        None => &p.mm_nt,
        Some(name) => {
            let opts = metal::CompileOptions::new();
            opts.set_language_version(metal::MTLLanguageVersion::V3_0);
            let lib = c
                ._device
                .new_library_with_source(&msl_source()?, &opts)
                .map_err(|e| e.to_string())?;
            let f = lib.get_function(name, None).map_err(|e| e.to_string())?;
            custom = c
                ._device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| e.to_string())?;
            &custom
        }
    };
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let x: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
    let w: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    let (bx, bw) = (wrap(c, &x), wrap(c, &w));
    let by = c
        ._device
        .new_buffer((n * m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let item = EgMm {
        n: n as u32,
        m: m as u32,
        k: k as u32,
        ldx: k as u32,
        ldw: k as u32,
        ldy: m as u32,
        alpha: 1.0,
        wrows: m as u32,
        ..Default::default()
    };
    let bi = c._device.new_buffer_with_data(
        &item as *const EgMm as *const c_void,
        std::mem::size_of::<EgMm>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let (tm, tn) = std::env::var("CMF_EGEMMA2_TILE")
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once('x')?;
            Some((a.parse::<u64>().ok()?, b.parse::<u64>().ok()?))
        })
        .unwrap_or((64, 64));
    let threads = std::env::var("CMF_EGEMMA2_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128u64);
    let mut best = 0f64;
    for _ in 0..reps.max(1) {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&bx), 0);
        enc.set_buffer(1, Some(&bw), 0);
        enc.set_buffer(2, Some(&bw), 0);
        enc.set_buffer(3, Some(&bw), 0);
        enc.set_buffer(4, Some(&by), 0);
        enc.set_buffer(5, Some(&bi), 0);
        enc.dispatch_thread_groups(
            MTLSize::new((n as u64).div_ceil(tm), (m as u64).div_ceil(tn), 1),
            MTLSize::new(threads, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ms = super::cmd_gpu_ms(cmd);
        best = best.max(2.0 * n as f64 * k as f64 * m as f64 / ms / 1e9);
    }
    let mut want = vec![0f32; n * m];
    crate::fcd_ops::gemm_nt_host(&x, &w, &mut want, n, k, m, None);
    let got = unsafe { std::slice::from_raw_parts(by.contents() as *const f32, n * m) };
    let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
    let err = got
        .iter()
        .zip(&want)
        .fold(0f32, |a, (&g, &w)| a.max((g - w).abs()));
    Ok((best, (err / scale) as f64))
}
