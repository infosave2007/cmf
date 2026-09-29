//! Qwen-Image-2.1 denoiser on native Metal.
//!
//! Implements the `gpu::qi21_*` contract: [`prefill`] runs the prompt
//! prefix (text and condition-image rows, t = 0 modulation, block-causal
//! mask) through all blocks once and keeps every layer's keys/values on
//! the device; [`step`] runs the target rows against `[cached prefix,
//! target]` — the pipeline's KV cache.
//!
//! Weights are read in place from the file mapping (the parent's no-copy
//! arena): q4tp tiles are dequantized to half as the GEMM stages them,
//! q8_2f/q8_row int8 are staged times the column field with the row scale
//! in the epilogue — so a container can mix both per tensor. Everything
//! else follows the Z-Image chain (`zimage.rs`): half activations behind
//! power-of-two guards, one fused qkv panel, qk-norm + RoPE in place,
//! flash attention over 32-key blocks, f32 residual stream.
//!
//! Per block: row op (gated residual + modulated LayerNorm) → qkv GEMM
//! (z = 3) → qk-norm/RoPE → flash → O GEMM → row op → gate/up GEMM (z = 2)
//! → SwiGLU → down GEMM. A command buffer covers two blocks.
//!
//! Measured (Mac mini M4, 512², 1024 target rows, in-process GPU timers):
//! a step is 5.3–5.9 s = GEMM 91 % (≈ 2.9 TF/s on q4tp, 83 % of the half
//! MMA peak) + flash 7 %; host work per step < 1 %. A 128-token GEMM tile
//! (256 threads, half the q4tp dequantization per FLOP, half-bit-trick
//! nibble decode) measured equal (4.87 vs 4.80 s) — the staging is not
//! the bottleneck — and was dropped.
//!
//! Knobs: `CMF_QI21_METAL=0` (device path off), `CMF_QI21_METAL_PROF=1`
//! (per-class GPU ms, one command buffer per op), `CMF_QI21_AMAX=1` (the
//! largest |value| at each half site, per step), `CMF_QI21_{ATTN,QKV,AO,
//! FFN,HID}_SHIFT` (the guards).

use crate::gpu::{Qi21BlockRef, Qi21Geom, Qi21PrefillArgs};
use cortiq_core::{CmfModel, TensorDtype};
use metal::{Buffer, CommandBuffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use super::{Ctx, WeightArena};

const QMSL: &str = include_str!("qi21_msl.metal");

fn enabled() -> bool {
    std::env::var("CMF_QI21_METAL").as_deref() != Ok("0")
}

fn decline(reason: &str) -> bool {
    static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
    if let Ok(mut v) = SAID.lock() {
        if !v.iter().any(|r| r == reason) {
            eprintln!("qwen-image-2.1: Metal device path declined: {reason}");
            v.push(reason.to_string());
        }
    }
    false
}

struct Pipes {
    mm_q4tp: ComputePipelineState,
    mm_q8: ComputePipelineState,
    flash: ComputePipelineState,
    qkrope: ComputePipelineState,
    rowop: ComputePipelineState,
    swiglu: ComputePipelineState,
    embed: ComputePipelineState,
    fin: ComputePipelineState,
    kvcopy: ComputePipelineState,
    amax: ComputePipelineState,
    probe: ComputePipelineState,
}
unsafe impl Send for Pipes {}
unsafe impl Sync for Pipes {}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn build_pipes(c: &Ctx) -> Result<Pipes, String> {
    let opts = metal::CompileOptions::new();
    opts.set_language_version(metal::MTLLanguageVersion::V3_0);
    let lib = c
        ._device
        .new_library_with_source(QMSL, &opts)
        .map_err(|e| format!("qi21 MSL compile: {e}"))?;
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
        qkrope: pso("qi_qkrope")?,
        rowop: pso("qi_rowop")?,
        swiglu: pso("qi_swiglu")?,
        embed: pso("qi_embed")?,
        fin: pso("qi_final")?,
        kvcopy: pso("qi_kvcopy")?,
        amax: pso("qi_amax")?,
        probe: pso("qi_fragprobe")?,
    })
}

fn frag_layout_ok(c: &Ctx, p: &Pipes) -> bool {
    let out = c._device.new_buffer(128 * 4, MTLResourceOptions::StorageModeShared);
    let cmd = c.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&p.probe);
    enc.set_buffer(0, Some(&out), 0);
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let v = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, 128) };
    (0..32).all(|l| v[l * 4] == v[l * 4 + 2] && v[l * 4 + 1] == v[l * 4 + 3])
}

fn device() -> Option<(&'static Ctx, &'static Pipes)> {
    if !enabled() {
        return None;
    }
    let c = super::ctx()?;
    match PIPES.get_or_init(|| {
        let p = build_pipes(c)?;
        if !frag_layout_ok(c, &p) {
            return Err("the simdgroup fragment layout differs from the one the kernels assume".into());
        }
        Ok(p)
    }) {
        Ok(p) => Some((c, p)),
        Err(e) => {
            decline(e);
            None
        }
    }
}

// ───────────────────────────── guards ─────────────────────────────

/// Power-of-two guards of the half sites (stored value = x·2^-s).
#[derive(Clone, Copy, Debug)]
struct Guards {
    attn: i32,
    qkv: i32,
    ao: i32,
    ffn: i32,
    hid: i32,
}

impl Guards {
    fn from_env() -> Guards {
        let e = |k: &str, d: i32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d).clamp(0, 14);
        Guards {
            attn: e("CMF_QI21_ATTN_SHIFT", 2),
            qkv: e("CMF_QI21_QKV_SHIFT", 4),
            ao: e("CMF_QI21_AO_SHIFT", 4),
            ffn: e("CMF_QI21_FFN_SHIFT", 2),
            hid: e("CMF_QI21_HID_SHIFT", 8),
        }
    }
}

fn p2(e: i32) -> f32 {
    (2.0f32).powi(e)
}

// ───────────────────────────── weights ─────────────────────────────

#[derive(Clone, Copy, Default)]
struct Tens {
    abs: usize,
    rows: usize,
    cols: usize,
    /// 0 = q4tp, 1 = q8 (int8 + row scale + column field)
    fmt: u8,
    params_off: usize,
    codes_off: usize,
    stride: usize,
    /// q8: float offsets of the row scale / column field in `aux`
    rs: usize,
    col: usize,
}

struct Blk {
    t: [Tens; 7],
    aux: Buffer,
    nq: usize,
    nk: usize,
}

const TQ: usize = 0;
const TK: usize = 1;
const TV: usize = 2;
const TO: usize = 3;
const TG: usize = 4;
const TU: usize = 5;
const TD: usize = 6;

fn f16s(b: &[u8]) -> impl Iterator<Item = f32> + '_ {
    b.chunks_exact(2)
        .map(|c| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
}

fn pad4(v: &mut Vec<f32>) {
    while v.len() % 4 != 0 {
        v.push(0.0);
    }
}

fn build_block(c: &Ctx, model: &CmfModel, r: &Qi21BlockRef, g: &Qi21Geom) -> Result<Blk, String> {
    let (h, inter) = (g.hidden, g.inter);
    let shapes = [(h, h), (h, h), (h, h), (h, h), (inter, h), (inter, h), (h, inter)];
    let mut t = [Tens::default(); 7];
    let mut aux: Vec<f32> = Vec::new();
    for (i, (&ti, &(rows, cols))) in r.w.iter().zip(&shapes).enumerate() {
        let e = model.tensors.get(ti).ok_or("tensor index out of range")?;
        if e.shape != [rows, cols] {
            return Err(format!("{}: shape {:?}, expected [{rows}, {cols}]", e.name, e.shape));
        }
        let abs = model
            .entry_abs_offset(e)
            .ok_or_else(|| format!("{}: not in the primary shard", e.name))?;
        if abs % 16 != 0 {
            return Err(format!("{}: unaligned ({abs})", e.name));
        }
        let bytes = model.entry_bytes(e);
        let q = rows * cols;
        let mut tt = Tens {
            abs,
            rows,
            cols,
            ..Default::default()
        };
        match e.dtype {
            TensorDtype::Q4TiledP => {
                let (po, co, st) = cortiq_core::quant::q4tp_sections(rows, cols);
                if bytes.len() < co + rows * st {
                    return Err(format!("{}: q4tp payload too short", e.name));
                }
                tt.fmt = 0;
                tt.params_off = po;
                tt.codes_off = co;
                tt.stride = st;
            }
            TensorDtype::Q8_2f if bytes.len() == q + 2 * rows + 2 * cols => {
                tt.fmt = 1;
                tt.rs = aux.len();
                aux.extend(f16s(&bytes[q..q + 2 * rows]));
                pad4(&mut aux);
                tt.col = aux.len();
                aux.extend(f16s(&bytes[q + 2 * rows..]));
                pad4(&mut aux);
            }
            TensorDtype::Q8Row if bytes.len() == q + 2 * rows => {
                tt.fmt = 1;
                tt.rs = aux.len();
                aux.extend(f16s(&bytes[q..]));
                pad4(&mut aux);
                tt.col = aux.len();
                aux.extend(std::iter::repeat_n(1.0f32, cols));
                pad4(&mut aux);
            }
            d => return Err(format!("{}: codec {d:?} (the Metal chain reads q4tp / q8_2f / q8_row)", e.name)),
        }
        t[i] = tt;
    }
    let nq = aux.len();
    aux.extend_from_slice(r.norm_q);
    pad4(&mut aux);
    let nk = aux.len();
    aux.extend_from_slice(r.norm_k);
    pad4(&mut aux);
    Ok(Blk {
        t,
        aux: buf_from(c, &aux),
        nq,
        nk,
    })
}

fn buf_from(c: &Ctx, v: &[f32]) -> Buffer {
    c._device.new_buffer_with_data(
        v.as_ptr() as *const c_void,
        (v.len().max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn buf_u32(c: &Ctx, v: &[u32]) -> Buffer {
    c._device.new_buffer_with_data(
        v.as_ptr() as *const c_void,
        (v.len().max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn buf_zeroed(c: &Ctx, bytes: usize) -> Buffer {
    c._device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
}

fn write_f32(b: &Buffer, off_floats: usize, v: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(v.as_ptr(), (b.contents() as *mut f32).add(off_floats), v.len());
    }
}

// ───────────────────────────── state ─────────────────────────────

struct Acts {
    alloc: usize,
    x: Buffer,     // f32 [alloc][H]
    xn: Buffer,    // half [alloc][H]
    panel: Buffer, // half [alloc][3H]
    attn: Buffer,  // half [alloc][H]
    y: Buffer,     // f32 [alloc][H]
    gu: Buffer,    // f32 [2][alloc][inter]
    h: Buffer,     // half [alloc][inter]
}

impl Acts {
    fn new(c: &Ctx, rows: usize, g: &Qi21Geom) -> Acts {
        let alloc = rows.div_ceil(128) * 128 + 128;
        let (h, i) = (g.hidden, g.inter);
        Acts {
            alloc,
            x: buf_zeroed(c, alloc * h * 4),
            xn: buf_zeroed(c, alloc * h * 2),
            panel: buf_zeroed(c, alloc * 3 * h * 2),
            attn: buf_zeroed(c, alloc * h * 2),
            y: buf_zeroed(c, alloc * h * 4),
            gu: buf_zeroed(c, 2 * alloc * i * 4),
            h: buf_zeroed(c, alloc * i * 2),
        }
    }
}

struct Prog {
    key: u64,
    lp: usize,
    n: usize,
    acts: Acts,
    /// per layer [lp][2H] half: K then V
    pkv: Buffer,
    rope_t: (Buffer, Buffer),
    mods: Buffer,
    fs: Buffer,
    xtok: Buffer,
    out: Buffer,
}

struct Dev {
    uid: u64,
    _model: Arc<CmfModel>,
    arena: Arc<WeightArena>,
    geom: Qi21Geom,
    guards: Guards,
    blocks: Vec<Blk>,
    emb: Buffer,
    fin: Buffer,
    progs: Vec<Prog>,
}

struct State(Option<Dev>);
unsafe impl Send for State {}

static STATE: Mutex<State> = Mutex::new(State(None));

/// Programs kept at once (the positive and the negative prompt).
const MAX_PROGS: usize = 2;

fn geom_ok(g: &Qi21Geom) -> Result<(), String> {
    if g.hd != 128 || g.nh * g.hd != g.hidden || g.hidden % 256 != 0 || g.hidden > 4096 {
        return Err(format!("geometry {g:?} (the kernels need hd 128, hidden % 256 == 0, ≤ 4096)"));
    }
    if g.inter % 64 != 0 || g.in_ch != 64 {
        return Err(format!("geometry {g:?} (inter % 64, 64 latent channels)"));
    }
    Ok(())
}

// ───────────────────────────── encoding ─────────────────────────────

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
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PQk {
    h: u32,
    nh: u32,
    ldp: u32,
    row0: u32,
    rope0: u32,
    eps: f32,
    qmul: f32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PRow {
    h: u32,
    row0: u32,
    mode: u32,
    eps: f32,
    oscale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PSw {
    n4: u32,
    g_off: u32,
    u_off: u32,
    h_off: u32,
    oscale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PEm {
    h: u32,
    n: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PFi {
    h: u32,
    row0: u32,
    eps: f32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PKv {
    h: u32,
    row0: u32,
    n: u32,
    ldp: u32,
}

fn set_p<T>(enc: &ComputeCommandEncoderRef, idx: u64, v: &T) {
    enc.set_bytes(idx, std::mem::size_of::<T>() as u64, v as *const T as *const c_void);
}

fn prof_on() -> bool {
    std::env::var("CMF_QI21_METAL_PROF").as_deref() == Ok("1")
}

fn amax_on() -> bool {
    std::env::var("CMF_QI21_AMAX").as_deref() == Ok("1")
}

/// Command-buffer recorder (the Z-Image one): a new buffer at every cut;
/// under profiling every op class gets its own buffer.
struct Rec<'a> {
    c: &'a Ctx,
    prof: bool,
    done: Vec<(&'static str, CommandBuffer)>,
    cur: Option<(CommandBuffer, Option<metal::ComputeCommandEncoder>, &'static str)>,
}

impl<'a> Rec<'a> {
    fn new(c: &'a Ctx) -> Rec<'a> {
        Rec {
            c,
            prof: prof_on(),
            done: Vec::new(),
            cur: None,
        }
    }
    fn close(&mut self) {
        if let Some((cmd, enc, label)) = self.cur.take() {
            if let Some(e) = enc {
                e.end_encoding();
            }
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
            self.cur = Some((cmd, None, label));
        }
        let cur = self.cur.as_mut().unwrap();
        if cur.1.is_none() {
            cur.1 = Some(cur.0.new_compute_command_encoder().to_owned());
        }
        cur.1.as_ref().unwrap()
    }
    fn cut(&mut self) {
        if !self.prof {
            self.close();
        }
    }
    fn finish(&mut self, what: &str) -> bool {
        self.close();
        let mut ok = true;
        for (_, cmd) in &self.done {
            cmd.wait_until_completed();
            if cmd.status() != metal::MTLCommandBufferStatus::Completed {
                ok = false;
            }
        }
        if self.prof {
            let mut by: Vec<(&'static str, f64)> = Vec::new();
            for (l, cmd) in &self.done {
                let ms = super::cmd_gpu_ms(cmd);
                match by.iter_mut().find(|(k, _)| k == l) {
                    Some(e) => e.1 += ms,
                    None => by.push((l, ms)),
                }
            }
            let tot: f64 = by.iter().map(|e| e.1).sum();
            let s: Vec<String> = by.iter().map(|(l, ms)| format!("{l} {ms:.1}")).collect();
            eprintln!("qi21 metal {what} gpu ms: total {tot:.1} · {}", s.join(" · "));
        }
        self.done.clear();
        ok
    }
}

struct Enc<'r, 'a> {
    rec: &'r mut Rec<'a>,
    p: &'static Pipes,
    d: &'r Dev,
    amax: Option<&'r Buffer>,
}

/// One program's views for a pass (prefill or step).
struct Pass<'b> {
    a: &'b Acts,
    rows: usize,
    rope: (&'b Buffer, &'b Buffer),
    mods: &'b Buffer,
    pkv: &'b Buffer,
    lp: usize,
    prefill: Option<&'b Buffer>, // the visibility buffer
}

impl Enc<'_, '_> {
    /// y = GEMM over `n` rows; the tensors in `ts` share a shape. Tensors
    /// of one codec go out as one z-batched dispatch.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        blk: &Blk,
        ts: &[usize],
        x: &Buffer,
        x_off: &[usize],
        ldx: usize,
        y: &Buffer,
        y_off: &[usize],
        ldy: usize,
        n: usize,
        half_out: bool,
        mul: f32,
    ) {
        for fmt in [0u8, 1u8] {
            let sel: Vec<usize> = (0..ts.len()).filter(|&i| blk.t[ts[i]].fmt == fmt).collect();
            if sel.is_empty() {
                continue;
            }
            let t0 = blk.t[ts[sel[0]]];
            let mut pm = PMm {
                n: n as u32,
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
            for (z, &i) in sel.iter().enumerate() {
                pm.x_off[z] = x_off[i] as u32;
                pm.y_off[z] = y_off[i] as u32;
            }
            let arena = self.d.arena.clone();
            let p = self.p;
            let enc = self.rec.enc("gemm");
            enc.set_compute_pipeline_state(if fmt == 1 { &p.mm_q8 } else { &p.mm_q4tp });
            for s in 0..3usize {
                let i = sel[s.min(sel.len() - 1)];
                let t = blk.t[ts[i]];
                arena.bind(enc, s as u64, t.abs);
                enc.set_buffer(3 + s as u64, Some(&blk.aux), (t.rs * 4) as u64);
                enc.set_buffer(9 + s as u64, Some(&blk.aux), (t.col * 4) as u64);
            }
            enc.set_buffer(6, Some(x), 0);
            enc.set_buffer(7, Some(y), 0);
            set_p(enc, 8, &pm);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(64) as u64, (t0.rows / 64) as u64, sel.len() as u64),
                MTLSize::new(128, 1, 1),
            );
        }
    }

    /// mode bit 0: x += tanh(g)·y; bit 1: xn = half(LN(x)·(1+s)·oscale).
    fn rowop(&mut self, a: &Acts, n: usize, gate: Option<(&Buffer, usize)>, scale: Option<(&Buffer, usize, f32)>) {
        let g = self.d.geom;
        let mut pr = PRow {
            h: g.hidden as u32,
            row0: 0,
            eps: g.eps,
            ..Default::default()
        };
        let p = self.p;
        let enc = self.rec.enc("row");
        enc.set_compute_pipeline_state(&p.rowop);
        enc.set_buffer(0, Some(&a.x), 0);
        enc.set_buffer(1, Some(&a.y), 0);
        enc.set_buffer(2, Some(&a.xn), 0);
        enc.set_buffer(3, Some(&a.y), 0);
        enc.set_buffer(4, Some(&a.y), 0);
        if let Some((b, off)) = gate {
            pr.mode |= 1;
            enc.set_buffer(3, Some(b), (off * 4) as u64);
        }
        if let Some((b, off, os)) = scale {
            pr.mode |= 2;
            pr.oscale = os;
            enc.set_buffer(4, Some(b), (off * 4) as u64);
        }
        set_p(enc, 5, &pr);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn amax(&mut self, buf: &Buffer, count: usize, site: usize) {
        let Some(ab) = self.amax else { return };
        let p = self.p;
        let enc = self.rec.enc("amax");
        enc.set_compute_pipeline_state(&p.amax);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_buffer(1, Some(ab), 0);
        let n = count as u32;
        let slot = site as u32;
        set_p(enc, 2, &n);
        set_p(enc, 3, &slot);
        enc.dispatch_thread_groups(MTLSize::new(256, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Every block over the pass's rows. `xn` must already hold the first
    /// block's modulated LayerNorm.
    fn blocks(&mut self, s: &Pass) {
        let g = self.d.geom;
        let gd = self.d.guards;
        let (h, inter) = (g.hidden, g.inter);
        let a = s.a;
        let n = s.rows;
        let gl = a.alloc * inter;
        let nb = self.d.blocks.len();
        for l in 0..nb {
            let blk = &self.d.blocks[l];
            // qkv → panel
            self.gemm(blk, &[TQ, TK, TV], &a.xn, &[0, 0, 0], h, &a.panel, &[0, h, 2 * h], 3 * h, n, true, p2(gd.attn - gd.qkv));
            self.amax(&a.panel, n * 3 * h, 1);
            {
                let p = self.p;
                let pq = PQk {
                    h: h as u32,
                    nh: g.nh as u32,
                    ldp: (3 * h) as u32,
                    row0: 0,
                    rope0: 0,
                    eps: g.eps * p2(-2 * gd.qkv),
                    qmul: std::f32::consts::LOG2_E / (g.hd as f32).sqrt(),
                    pad0: 0,
                };
                let enc = self.rec.enc("rope");
                enc.set_compute_pipeline_state(&p.qkrope);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(&blk.aux), (blk.nq * 4) as u64);
                enc.set_buffer(2, Some(&blk.aux), (blk.nk * 4) as u64);
                enc.set_buffer(3, Some(s.rope.0), 0);
                enc.set_buffer(4, Some(s.rope.1), 0);
                set_p(enc, 5, &pq);
                enc.dispatch_thread_groups(
                    MTLSize::new(n as u64, (2 * g.nh).div_ceil(4) as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
            let kv_off = (l * s.lp * 2 * h * 2) as u64;
            if s.prefill.is_some() {
                let p = self.p;
                let pk = PKv {
                    h: h as u32,
                    row0: 0,
                    n: n as u32,
                    ldp: (3 * h) as u32,
                };
                let enc = self.rec.enc("kv");
                enc.set_compute_pipeline_state(&p.kvcopy);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(s.pkv), kv_off);
                set_p(enc, 2, &pk);
                enc.dispatch_threads(MTLSize::new((2 * h / 8) as u64, n as u64, 1), MTLSize::new(64, 4, 1));
            }
            {
                let p = self.p;
                let pf = PFa {
                    q_row0: 0,
                    nq: n as u32,
                    n_pre: if s.prefill.is_some() { 0 } else { s.lp as u32 },
                    n_own: n as u32,
                    ldp: (3 * h) as u32,
                    h: h as u32,
                    ldo: h as u32,
                    oscale: p2(gd.qkv - gd.ao),
                    use_vis: s.prefill.is_some() as u32,
                    ..Default::default()
                };
                let enc = self.rec.enc("flash");
                enc.set_compute_pipeline_state(&p.flash);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(&a.attn), 0);
                enc.set_buffer(2, Some(s.pkv), kv_off);
                enc.set_buffer(3, Some(s.prefill.unwrap_or(&a.y)), 0);
                set_p(enc, 4, &pf);
                enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(64) as u64, g.nh as u64, 1), MTLSize::new(256, 1, 1));
            }
            self.amax(&a.attn, n * h, 2);
            self.gemm(blk, &[TO], &a.attn, &[0], h, &a.y, &[0], h, n, false, p2(gd.ao));
            self.rowop(a, n, Some((s.mods, h)), Some((s.mods, 2 * h, p2(-gd.ffn))));
            self.amax(&a.xn, n * h, 3);
            self.gemm(blk, &[TG, TU], &a.xn, &[0, 0], h, &a.gu, &[0, gl], inter, n, false, p2(gd.ffn));
            {
                let p = self.p;
                let ps = PSw {
                    n4: (n * inter / 4) as u32,
                    g_off: 0,
                    u_off: gl as u32,
                    h_off: 0,
                    oscale: p2(-gd.hid),
                    ..Default::default()
                };
                let enc = self.rec.enc("swiglu");
                enc.set_compute_pipeline_state(&p.swiglu);
                enc.set_buffer(0, Some(&a.gu), 0);
                enc.set_buffer(1, Some(&a.h), 0);
                set_p(enc, 2, &ps);
                enc.dispatch_thread_groups(MTLSize::new((ps.n4 as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
            }
            self.amax(&a.h, n * inter, 4);
            self.gemm(blk, &[TD], &a.h, &[0], inter, &a.y, &[0], h, n, false, p2(gd.hid));
            if l + 1 < nb {
                self.rowop(a, n, Some((s.mods, 3 * h)), Some((s.mods, 0, p2(-gd.attn))));
                self.amax(&a.xn, n * h, 0);
            } else {
                self.rowop(a, n, Some((s.mods, 3 * h)), None);
            }
            if l % 2 == 1 {
                self.rec.cut();
            }
        }
    }
}

fn read_amax(b: &Buffer, what: &str, g: Guards) {
    let v = unsafe { std::slice::from_raw_parts(b.contents() as *const u32, 5) };
    let f: Vec<f32> = v.iter().map(|&x| f32::from_bits(x)).collect();
    eprintln!(
        "qi21 {what} amax (stored, guards attn {} qkv {} ao {} ffn {} hid {}): attn-in {:.1} · qkv {:.1} · attn-out {:.1} · ffn-in {:.1} · hidden {:.1}",
        g.attn, g.qkv, g.ao, g.ffn, g.hid, f[0], f[1], f[2], f[3], f[4]
    );
}

// ───────────────────────────── contract ─────────────────────────────

/// Build the program for `a.key` and run its prefix through every block,
/// keeping each layer's keys and values.
pub(crate) fn prefill(a: &Qi21PrefillArgs) -> bool {
    let Some((c, p)) = device() else { return false };
    let g = a.geom;
    if let Err(e) = geom_ok(&g) {
        return decline(&e);
    }
    if a.lp < 8 || a.n < 8 {
        return decline("fewer than 8 prefix or target rows");
    }
    let h = g.hidden;
    if a.x.len() != a.lp * h
        || a.rope_p.0.len() != a.lp * 64
        || a.rope_t.0.len() != a.n * 64
        || a.vis.len() != a.lp
        || a.mods0.len() != 4 * h
    {
        return decline("prefill args do not match the lengths");
    }
    let mut st = STATE.lock().unwrap();
    if st.0.as_ref().is_some_and(|d| d.uid != a.model.uid() || d.geom != g) {
        st.0 = None;
    }
    if st.0.is_none() {
        let Some((arena, _)) = super::file_buffer(c, a.model) else {
            return decline("the file mapping cannot be wrapped as a Metal buffer");
        };
        let mut blocks = Vec::with_capacity(a.blocks.len());
        for r in a.blocks {
            match build_block(c, a.model, r, &g) {
                Ok(b) => blocks.push(b),
                Err(e) => return decline(&e),
            }
        }
        st.0 = Some(Dev {
            uid: a.model.uid(),
            _model: a.model.clone(),
            arena,
            geom: g,
            guards: Guards::from_env(),
            blocks,
            emb: buf_from(c, a.img_in),
            fin: buf_from(c, a.proj_out),
            progs: Vec::new(),
        });
    }
    let d = st.0.as_mut().unwrap();
    let rows = a.lp.max(a.n);
    // Every buffer must fit one MTLBuffer (newBufferWithLength answers nil
    // past maxBufferLength — 13.6 GB on a 24 GB M4, less on smaller Macs):
    // many or large condition images grow the prefix cache without bound.
    {
        let alloc = rows.div_ceil(128) * 128 + 128;
        let largest = [
            d.blocks.len() * a.lp * 2 * h * 2, // prefix K/V
            2 * alloc * g.inter * 4,           // gate|up
            alloc * g.inter * 2,               // SwiGLU hidden
            alloc * 3 * h * 2,                 // qkv panel
            alloc * h * 4,                     // x / y
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let total = d.blocks.len() * a.lp * 2 * h * 2 + alloc * (2 * g.inter * 4 + g.inter * 2 + 3 * h * 2 + 2 * h * 4 + 2 * h * 2);
        let max_len = c._device.max_buffer_length() as usize;
        let budget = c._device.recommended_max_working_set_size() as usize;
        if largest > max_len || total > budget {
            return decline(&format!(
                "the program needs a {:.1} GB buffer ({:.1} GB in all) — past this device's {:.1} GB buffer / {:.1} GB working-set limit",
                largest as f64 / 1e9,
                total as f64 / 1e9,
                max_len as f64 / 1e9,
                budget as f64 / 1e9
            ));
        }
    }
    let prog = Prog {
        key: a.key,
        lp: a.lp,
        n: a.n,
        acts: Acts::new(c, rows, &g),
        pkv: buf_zeroed(c, d.blocks.len() * a.lp * 2 * h * 2),
        rope_t: (buf_from(c, a.rope_t.0), buf_from(c, a.rope_t.1)),
        mods: buf_zeroed(c, 4 * h * 4),
        fs: buf_zeroed(c, h * 4),
        xtok: buf_zeroed(c, a.n * 64 * 4),
        out: buf_zeroed(c, a.n * 64 * 4),
    };
    d.progs.retain(|q| q.key != a.key);
    if d.progs.len() >= MAX_PROGS {
        d.progs.remove(0);
    }
    // prefix-only buffers live for this call
    let rope_p = (buf_from(c, a.rope_p.0), buf_from(c, a.rope_p.1));
    let vis = buf_u32(c, a.vis);
    let mods0 = buf_from(c, a.mods0);
    write_f32(&prog.acts.x, 0, a.x);
    let amax = amax_on().then(|| buf_zeroed(c, 64));
    let d_ref: &Dev = d;
    let mut rec = Rec::new(c);
    {
        let mut e = Enc {
            rec: &mut rec,
            p,
            d: d_ref,
            amax: amax.as_ref(),
        };
        let gd = d_ref.guards;
        e.rowop(&prog.acts, a.lp, None, Some((&mods0, 0, p2(-gd.attn))));
        e.blocks(&Pass {
            a: &prog.acts,
            rows: a.lp,
            rope: (&rope_p.0, &rope_p.1),
            mods: &mods0,
            pkv: &prog.pkv,
            lp: a.lp,
            prefill: Some(&vis),
        });
    }
    if !rec.finish("prefill") {
        return decline("a prefill command buffer failed");
    }
    if let Some(b) = &amax {
        read_amax(b, "prefill", d_ref.guards);
    }
    d.progs.push(prog);
    true
}

/// One denoiser call for the prepared `key`: latent tokens `[n, 64]` →
/// velocity `[n, 64]`. `mods` = the step's `[s1|g1|s2|g2]`, `fs` = 1 + the
/// final norm's scale.
pub(crate) fn step(key: u64, xtok: &[f32], mods: &[f32], fs: &[f32], out: &mut [f32]) -> bool {
    let Some((c, p)) = device() else { return false };
    let mut st = STATE.lock().unwrap();
    let Some(d) = st.0.as_mut() else { return false };
    let Some(pi) = d.progs.iter().position(|q| q.key == key) else { return false };
    let g = d.geom;
    let h = g.hidden;
    let d_ref: &Dev = d;
    let prog = &d_ref.progs[pi];
    let n = prog.n;
    if xtok.len() != n * 64 || out.len() != n * 64 || mods.len() != 4 * h || fs.len() != h {
        return decline("step args do not match the program");
    }
    write_f32(&prog.xtok, 0, xtok);
    write_f32(&prog.mods, 0, mods);
    write_f32(&prog.fs, 0, fs);
    let amax = amax_on().then(|| buf_zeroed(c, 64));
    let mut rec = Rec::new(c);
    {
        let mut e = Enc {
            rec: &mut rec,
            p,
            d: d_ref,
            amax: amax.as_ref(),
        };
        let gd = d_ref.guards;
        {
            let pe = PEm {
                h: h as u32,
                n: n as u32,
                ..Default::default()
            };
            let enc = e.rec.enc("embed");
            enc.set_compute_pipeline_state(&p.embed);
            enc.set_buffer(0, Some(&prog.xtok), 0);
            enc.set_buffer(1, Some(&d_ref.emb), 0);
            enc.set_buffer(2, Some(&prog.acts.x), 0);
            set_p(enc, 3, &pe);
            enc.dispatch_threads(MTLSize::new(h as u64, n as u64, 1), MTLSize::new(256, 1, 1));
        }
        e.rowop(&prog.acts, n, None, Some((&prog.mods, 0, p2(-gd.attn))));
        e.blocks(&Pass {
            a: &prog.acts,
            rows: n,
            rope: (&prog.rope_t.0, &prog.rope_t.1),
            mods: &prog.mods,
            pkv: &prog.pkv,
            lp: prog.lp,
            prefill: None,
        });
        {
            let pf = PFi {
                h: h as u32,
                row0: 0,
                eps: g.eps,
                pad0: 0,
            };
            let enc = e.rec.enc("final");
            enc.set_compute_pipeline_state(&p.fin);
            enc.set_buffer(0, Some(&prog.acts.x), 0);
            enc.set_buffer(1, Some(&prog.fs), 0);
            enc.set_buffer(2, Some(&d_ref.fin), 0);
            enc.set_buffer(3, Some(&prog.out), 0);
            set_p(enc, 4, &pf);
            enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
    }
    if !rec.finish("step") {
        return decline("a step command buffer failed");
    }
    if let Some(b) = &amax {
        read_amax(b, "step", d_ref.guards);
    }
    let src = unsafe { std::slice::from_raw_parts(prog.out.contents() as *const f32, n * 64) };
    out.copy_from_slice(src);
    true
}

/// Drop one program (a prompt is done).
pub(crate) fn release_key(key: u64) {
    if let Ok(mut st) = STATE.lock() {
        if let Some(d) = st.0.as_mut() {
            d.progs.retain(|q| q.key != key);
        }
    }
}

/// Drop everything (weights views, programs).
pub(crate) fn release() {
    if let Ok(mut st) = STATE.lock() {
        st.0 = None;
    }
}
