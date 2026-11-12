//! MiniMax-H3 stacks on native Metal: the DiT of MiniMax-H3 /
//! MiniMax-H3-Image (and its prompt refiner) as a resident chain.
//!
//! Implements `gpu::h3_chain` for a stack whose four projections are q4tp
//! tensors, with 128-wide heads and no biases. The residual stream is
//! written once and read once; between them every block runs on the GPU,
//! with the kernels of the Qwen-Image-2.1 chain (`qi21_msl.metal`) where the
//! arithmetic is the same and three of its own (`h3_msl.metal`):
//!
//! - `h3_rowop`: the gated residual `x += g ⊙ y` and the next RMS norm with
//!   its adaLN modulation into half, a token's modulation row looked up per
//!   row;
//! - `qi_mm_q4tp`: the four products, the weights read in place from the
//!   file mapping and dequantized to half as the GEMM stages them. qkv
//!   writes the half panel, the output projection and fc2 write f32, fc1
//!   writes the `[gate | up]` panel in f32 for `h3_swiglu`;
//! - `h3_qkrope`: per-head RMS norm·w and the split-half rotation of the
//!   first `2·pairs` dims, in place on the panel's q and k columns;
//! - `qi_flash`: flash attention over 32-key blocks.
//!
//! Range guards as on the wgpu chain (`gpu_wgpu/h3.rs`): the normed
//! activations ×2⁻², the qkv panel ×2⁻³, the SwiGLU middle ×2⁻⁸; each
//! consumer multiplies its factor back. A non-finite result makes the chain
//! stand down and the per-op path redo the call.
//!
//! The prompt encoder ([`te_chain`]: grouped causal attention through
//! `h3_flash_128`) and the VAE decoder's ViT ([`vit_chain`]: int8 weights,
//! 64-wide heads, tile batches) have chains of the same making. Both keep
//! their weights in buffers of their own: see their notes.
//!
//! Knobs: `CMF_H3_METAL=0` (the per-op path for every stack),
//! `CMF_H3_METAL_TE=0`, `CMF_H3_METAL_VIT=0` (for one),
//! `CMF_H3_METAL_PROF=1` (GPU ms
//! per kernel class, one command buffer per class), `CMF_H3_METAL_AMAX=1`
//! (the largest stored value of each half site),
//! `CMF_H3_METAL_{ACT,QKV,HID}_SHIFT` (the guards).

use crate::gpu::{H3Chain, TeChain};
use cortiq_core::{CmfModel, TensorDtype};
use metal::{
    Buffer, CommandBuffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions,
    MTLSize,
};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use super::{Ctx, WeightArena};

const QI_MSL: &str = include_str!("qi21_msl.metal");
const H3_MSL: &str = include_str!("h3_msl.metal");

/// Chunks of a block's modulation row: shift, scale, gate for the attention
/// half, then for the MLP.
const EXPAND: usize = 6;

fn enabled() -> bool {
    std::env::var("CMF_H3_METAL").as_deref() != Ok("0")
}

/// Say once why the chain stood down (the caller runs its per-op path).
fn decline(reason: &str) -> bool {
    static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
    if let Ok(mut v) = SAID.lock()
        && !v.iter().any(|r| r == reason)
    {
        eprintln!("minimax-h3: Metal chain declined: {reason}");
        v.push(reason.to_string());
    }
    false
}

/// A stack the chain is not written for (the VAE's ViT): no message unless
/// asked.
fn not_mine(reason: &str) -> bool {
    if std::env::var("CMF_GPU_DEBUG").is_ok() {
        decline(reason)
    } else {
        false
    }
}

struct Pipes {
    mm: ComputePipelineState,
    flash: ComputePipelineState,
    rowop: ComputePipelineState,
    qkrope: ComputePipelineState,
    swiglu: ComputePipelineState,
    /// `qi_swiglu`: the gate and up panels as two planes (the encoder's).
    swiglu2: ComputePipelineState,
    /// `h3_flash_128`: grouped, causal attention (the encoder's).
    flash_g: ComputePipelineState,
    /// The ViT decoder's: the int8 GEMM, attention over 64-wide heads and
    /// tile batches, its q/k/v step.
    mm_q8: ComputePipelineState,
    flash64: ComputePipelineState,
    vit_qk: ComputePipelineState,
    amax: ComputePipelineState,
    probe: ComputePipelineState,
}
unsafe impl Send for Pipes {}
unsafe impl Sync for Pipes {}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn build_pipes(c: &Ctx) -> Result<Pipes, String> {
    let opts = metal::CompileOptions::new();
    opts.set_language_version(metal::MTLLanguageVersion::V3_0);
    let src = format!("{QI_MSL}\n{H3_MSL}");
    let lib = c
        ._device
        .new_library_with_source(&src, &opts)
        .map_err(|e| format!("h3 MSL compile: {e}"))?;
    let pso = |name: &str| -> Result<ComputePipelineState, String> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| format!("kernel {name}: {e}"))?;
        c._device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline {name}: {e}"))
    };
    Ok(Pipes {
        mm: pso("qi_mm_q4tp")?,
        flash: pso("qi_flash")?,
        rowop: pso("h3_rowop")?,
        qkrope: pso("h3_qkrope")?,
        swiglu: pso("h3_swiglu")?,
        swiglu2: pso("qi_swiglu")?,
        flash_g: pso("h3_flash_128")?,
        mm_q8: pso("qi_mm_q8")?,
        flash64: pso("h3_flash_64")?,
        vit_qk: pso("h3_vit_qk")?,
        amax: pso("qi_amax")?,
        probe: pso("qi_fragprobe")?,
    })
}

/// Whether a simdgroup matrix's elements sit in the lanes where `qi_fc`
/// says (the GEMM and the attention store through it).
fn frag_layout_ok(c: &Ctx, p: &Pipes) -> bool {
    let out = c
        ._device
        .new_buffer(128 * 4, MTLResourceOptions::StorageModeShared);
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
            return Err(
                "the simdgroup fragment layout differs from the one the kernels assume".into(),
            );
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

/// Whether this device can run the chain at all.
pub(crate) fn available() -> bool {
    device().is_some()
}

#[derive(Clone, Copy, Debug)]
struct Guards {
    act: i32,
    qkv: i32,
    hid: i32,
}

impl Guards {
    fn from_env() -> Guards {
        let e = |k: &str, d: i32| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
                .clamp(0, 14)
        };
        Guards {
            act: e("CMF_H3_METAL_ACT_SHIFT", 2),
            qkv: e("CMF_H3_METAL_QKV_SHIFT", 3),
            hid: e("CMF_H3_METAL_HID_SHIFT", 8),
        }
    }
}

fn p2(e: i32) -> f32 {
    (2.0f32).powi(e)
}

// ───────────────────────────── weights ─────────────────────────────

/// A weight tensor where it lies in the file mapping.
#[derive(Clone, Copy)]
struct Tens {
    abs: usize,
    rows: usize,
    cols: usize,
    /// 0 = q4tp, 1 = int8 with a row scale and a column field
    fmt: u8,
    params_off: usize,
    codes_off: usize,
    stride: usize,
    /// int8: float offsets of the row scale and the column field in the
    /// call's `aux` buffer
    rs: usize,
    col: usize,
}

fn tens(model: &CmfModel, idx: usize, rows: usize, cols: usize, safe_len: usize) -> Option<Tens> {
    let e = model.tensors.get(idx)?;
    if e.dtype != TensorDtype::Q4TiledP || e.shape != [rows, cols] {
        return None;
    }
    let abs = model.entry_abs_offset(e)?;
    let (params_off, codes_off, stride) = cortiq_core::quant::q4tp_sections(rows, cols);
    let need = codes_off + rows * stride;
    (abs.is_multiple_of(16) && (e.nbytes as usize) >= need && abs + need <= safe_len).then_some(Tens {
        abs,
        rows,
        cols,
        fmt: 0,
        params_off,
        codes_off,
        stride,
        rs: 0,
        col: 0,
    })
}

/// [`tens`] for a stack that may be int8 (q8_2f / q8_row): the row scales
/// and the column field go to `aux` as f32.
fn tens_any(model: &CmfModel, idx: usize, rows: usize, cols: usize, safe_len: usize, aux: &mut Vec<f32>) -> Option<Tens> {
    let e = model.tensors.get(idx)?;
    if e.dtype == TensorDtype::Q4TiledP {
        return tens(model, idx, rows, cols, safe_len);
    }
    if e.shape != [rows, cols] {
        return None;
    }
    let abs = model.entry_abs_offset(e)?;
    let bytes = model.entry_bytes(e);
    let q = rows * cols;
    let f16s = |b: &[u8]| -> Vec<f32> {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| cortiq_core::quant::f16_to_f32(u16::from_le_bytes(*c)))
            .collect()
    };
    let pad4 = |v: &mut Vec<f32>| {
        while !v.len().is_multiple_of(4) {
            v.push(0.0);
        }
    };
    let mut t = Tens { abs, rows, cols, fmt: 1, params_off: 0, codes_off: 0, stride: 0, rs: aux.len(), col: 0 };
    match e.dtype {
        TensorDtype::Q8_2f if bytes.len() == q + 2 * rows + 2 * cols => {
            aux.extend(f16s(&bytes[q..q + 2 * rows]));
            pad4(aux);
            t.col = aux.len();
            aux.extend(f16s(&bytes[q + 2 * rows..]));
        }
        TensorDtype::Q8Row if bytes.len() == q + 2 * rows => {
            aux.extend(f16s(&bytes[q..]));
            pad4(aux);
            t.col = aux.len();
            aux.extend(std::iter::repeat_n(1.0f32, cols));
        }
        _ => return None,
    }
    pad4(aux);
    (abs.is_multiple_of(16) && abs + q <= safe_len).then_some(t)
}

/// `src` into the buffer at `dst`, across eight threads: the source is the
/// file mapping, usually not in memory yet, and one thread faults it in at
/// 0.76 GB/s on a MacBook Air M4 (12.7 GB of the encoder in 16.6 s).
///
/// # Safety
/// `dst` must be valid for `src.len()` bytes and not alias `src`.
unsafe fn par_copy(src: &[u8], dst: *mut u8) {
    const THREADS: usize = 8;
    if src.len() < (4 << 20) {
        // SAFETY: the caller's contract.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        return;
    }
    let part = src.len().div_ceil(THREADS).next_multiple_of(16384);
    let base = dst as usize;
    std::thread::scope(|s| {
        for (i, piece) in src.chunks(part).enumerate() {
            let at = base + i * part;
            s.spawn(move || {
                // SAFETY: disjoint ranges of the destination, one a thread.
                unsafe { std::ptr::copy_nonoverlapping(piece.as_ptr(), at as *mut u8, piece.len()) }
            });
        }
    });
}

fn buf_from<T: Copy>(c: &Ctx, v: &[T]) -> Buffer {
    if v.is_empty() {
        return buf_zeroed(c, 16);
    }
    c._device.new_buffer_with_data(
        v.as_ptr() as *const c_void,
        std::mem::size_of_val(v) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn buf_zeroed(c: &Ctx, bytes: usize) -> Buffer {
    c._device
        .new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
}

// ───────────────────────────── state ─────────────────────────────

/// Activations of a chain call, `alloc` rows each (padded so the GEMM's
/// 64-row tiles and the attention's query blocks stay inside). Kept between
/// calls and regrown when a call needs more rows or another geometry.
struct Acts {
    alloc: usize,
    /// Row widths: the stream, the q/k/v panel, the attention output, the
    /// MLP's middle.
    dims: (usize, usize, usize, usize),
    x: Buffer,     // f32 [alloc][hidden]
    xn: Buffer,    // half [alloc][hidden]
    panel: Buffer, // half [alloc][panel]
    attn: Buffer,  // half [alloc][att]
    y: Buffer,     // f32 [alloc][hidden]
    gu: Buffer,    // f32, 2·ffn a row: [gate | up] rows, or a gate and an up plane
    hid: Buffer,   // half [alloc][ffn]
}

fn acts_sizes(alloc: usize, hs: usize, panel: usize, att: usize, ffn: usize) -> [usize; 7] {
    [
        alloc * hs * 4,
        alloc * hs * 2,
        alloc * panel * 2,
        alloc * att * 2,
        alloc * hs * 4,
        alloc * 2 * ffn * 4,
        alloc * ffn * 2,
    ]
}

impl Acts {
    fn new(c: &Ctx, alloc: usize, hs: usize, panel: usize, att: usize, ffn: usize) -> Acts {
        let s = acts_sizes(alloc, hs, panel, att, ffn);
        Acts {
            alloc,
            dims: (hs, panel, att, ffn),
            x: buf_zeroed(c, s[0]),
            xn: buf_zeroed(c, s[1]),
            panel: buf_zeroed(c, s[2]),
            attn: buf_zeroed(c, s[3]),
            y: buf_zeroed(c, s[4]),
            gu: buf_zeroed(c, s[5]),
            hid: buf_zeroed(c, s[6]),
        }
    }
}

struct State(Option<Acts>);
unsafe impl Send for State {}

static STATE: Mutex<State> = Mutex::new(State(None));

/// A ViT stack's weights in a buffer of their own, by (model, first
/// tensor, blocks): the tensors' offsets there.
struct VitW {
    key: (u64, usize, usize),
    buf: Buffer,
    offs: Vec<usize>,
}
unsafe impl Send for VitW {}

static VITW: Mutex<Option<VitW>> = Mutex::new(None);

/// Drop the activations and the ViT's weights (a stage is done).
pub(crate) fn release() {
    if let Ok(mut st) = STATE.lock() {
        st.0 = None;
    }
    if let Ok(mut v) = VITW.lock() {
        *v = None;
    }
}

/// The activations for `n` rows of these widths, kept or regrown. `false`
/// (said once): they do not fit this device.
fn ensure_acts(c: &Ctx, st: &mut State, n: usize, hs: usize, panel: usize, att: usize, ffn: usize) -> bool {
    let alloc = n.div_ceil(128) * 128 + 128;
    if st.0.as_ref().is_some_and(|a| a.alloc >= alloc && a.dims == (hs, panel, att, ffn)) {
        return true;
    }
    st.0 = None;
    // Every buffer must fit one MTLBuffer, and all of them the working set.
    let sizes = acts_sizes(alloc, hs, panel, att, ffn);
    let (largest, total) = (*sizes.iter().max().unwrap(), sizes.iter().sum::<usize>());
    let max_len = c._device.max_buffer_length() as usize;
    let budget = c._device.recommended_max_working_set_size() as usize;
    if largest > max_len || total > budget {
        return decline(&format!(
            "the activations need a {:.1} GB buffer ({:.1} GB in all), past this device's {:.1} GB buffer / {:.1} GB working set",
            largest as f64 / 1e9,
            total as f64 / 1e9,
            max_len as f64 / 1e9,
            budget as f64 / 1e9
        ));
    }
    st.0 = Some(Acts::new(c, alloc, hs, panel, att, ffn));
    true
}

// ───────────────────────────── encoding ─────────────────────────────

/// `QMm` of qi21_msl.metal.
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

/// `QFa` of qi21_msl.metal.
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

/// `HFa` of h3_msl.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PFg {
    nq: u32,
    nseq: u32,
    ldp: u32,
    k_col: u32,
    v_col: u32,
    rep: u32,
    ldo: u32,
    oscale: f32,
    causal: u32,
    hstride: u32,
    pad1: u32,
    pad2: u32,
}

/// `HVq` of h3_msl.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PVq {
    ldp: u32,
    nh: u32,
    pairs: u32,
    b_off: u32,
    nseq: u32,
    bscale: f32,
    eps: f32,
    qmul: f32,
}

/// `QSw` of qi21_msl.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PSw2 {
    n4: u32,
    g_off: u32,
    u_off: u32,
    h_off: u32,
    oscale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `HRow` of h3_msl.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PRow {
    h: u32,
    mode: u32,
    eps: f32,
    oscale: f32,
    g_off: u32,
    sh_off: u32,
    sc_off: u32,
    w_off: u32,
}

/// `HQk` of h3_msl.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PQk {
    ldp: u32,
    nq: u32,
    nk: u32,
    k_col: u32,
    pairs: u32,
    w_off: u32,
    eps: f32,
    qmul: f32,
}

/// `HSw` of h3_msl.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PSw {
    n4: u32,
    ffn: u32,
    oscale: f32,
    has_b: u32,
}

fn set_p<T>(enc: &ComputeCommandEncoderRef, idx: u64, v: &T) {
    enc.set_bytes(
        idx,
        std::mem::size_of::<T>() as u64,
        v as *const T as *const c_void,
    );
}

/// Command-buffer recorder: a new buffer at every cut; under profiling
/// every kernel class gets buffers of its own.
struct Rec<'a> {
    c: &'a Ctx,
    prof: bool,
    done: Vec<(&'static str, CommandBuffer)>,
    /// command buffers of `done` already waited for
    waited: usize,
    cur: Option<(
        CommandBuffer,
        Option<metal::ComputeCommandEncoder>,
        &'static str,
    )>,
}

impl<'a> Rec<'a> {
    fn new(c: &'a Ctx) -> Rec<'a> {
        Rec {
            c,
            prof: std::env::var("CMF_H3_METAL_PROF").as_deref() == Ok("1"),
            done: Vec::new(),
            waited: 0,
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
    /// Commit what is recorded and wait for it: a buffer the next commands
    /// are about to overwrite must be idle.
    fn drain(&mut self) -> bool {
        self.close();
        let mut ok = true;
        for (_, cmd) in &self.done[self.waited..] {
            cmd.wait_until_completed();
            if cmd.status() != metal::MTLCommandBufferStatus::Completed {
                ok = false;
            }
        }
        self.waited = self.done.len();
        ok
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
            eprintln!("h3 metal chain, {what}, gpu ms: total {tot:.1} · {}", s.join(" · "));
        }
        self.done.clear();
        self.waited = 0;
        ok
    }
}

/// What every dispatch of a call shares.
struct Enc<'r, 'a> {
    rec: &'r mut Rec<'a>,
    p: &'static Pipes,
    arena: &'r Arc<WeightArena>,
    a: &'r Acts,
    n: usize,
    hs: usize,
    cw: &'r Buffer,
    /// the int8 tensors' row scales and column fields (`cw` when none)
    aux: &'r Buffer,
    /// the stack's weights copied into a buffer of their own (a tensor's
    /// `abs` is then its offset there); None: read in place from the file
    own: Option<&'r Buffer>,
    mods: &'r Buffer,
    mrow: &'r Buffer,
    modded: bool,
    eps: f32,
    amax: Option<&'r Buffer>,
}

impl Enc<'_, '_> {
    /// `y[n][ldy] = mul · x[n][K] · Wᵀ`, f32 or half, written `y_off`
    /// elements into `y`.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&mut self, t: Tens, x: &Buffer, y: &Buffer, y_off: usize, ldy: usize, half_out: bool, mul: f32) {
        let mut pm = PMm {
            n: self.n as u32,
            rows: t.rows as u32,
            k: t.cols as u32,
            ldx: t.cols as u32,
            ldy: ldy as u32,
            epi: half_out as u32,
            mul,
            params_off: t.params_off as u32,
            codes_off: t.codes_off as u32,
            code_stride: t.stride as u32,
            ..Default::default()
        };
        pm.y_off[0] = y_off as u32;
        let (p, arena, aux, n, own) = (self.p, self.arena.clone(), self.aux, self.n, self.own);
        let enc = self.rec.enc("gemm");
        enc.set_compute_pipeline_state(if t.fmt == 1 { &p.mm_q8 } else { &p.mm });
        for s in 0..3u64 {
            match own {
                Some(b) => enc.set_buffer(s, Some(b), t.abs as u64),
                None => arena.bind(enc, s, t.abs),
            }
            // the int8 codec's row scales and column field (unread by q4tp)
            enc.set_buffer(3 + s, Some(aux), (t.rs * 4) as u64);
            enc.set_buffer(9 + s, Some(aux), (t.col * 4) as u64);
        }
        enc.set_buffer(6, Some(x), 0);
        enc.set_buffer(7, Some(y), 0);
        set_p(enc, 8, &pm);
        enc.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(64) as u64, (t.rows / 64) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// `res`: the gate chunk's offset in `mods`; `norm`: the offsets of the
    /// shift and scale chunks and of the norm's weights in `cw`.
    fn rowop(&mut self, res: Option<usize>, norm: Option<(usize, usize, usize)>, oscale: f32) {
        self.rowop_b(res, None, norm, oscale)
    }

    /// [`Self::rowop`], or with `bias` the residual `x += s ⊙ (y + b)`: the
    /// offsets of `b` and `s` in `cw` (an unmodulated stack).
    fn rowop_b(
        &mut self,
        res: Option<usize>,
        bias: Option<(usize, usize)>,
        norm: Option<(usize, usize, usize)>,
        oscale: f32,
    ) {
        let mut pr = PRow {
            h: self.hs as u32,
            mode: if self.modded { 4 } else { 0 },
            eps: self.eps,
            oscale,
            ..Default::default()
        };
        if let Some(g) = res {
            pr.mode |= 1;
            pr.g_off = g as u32;
        }
        if let Some((sh, sc, w)) = norm {
            pr.mode |= 2;
            pr.sh_off = sh as u32;
            pr.sc_off = sc as u32;
            pr.w_off = w as u32;
        }
        if let Some((b, sc)) = bias {
            pr.mode |= 1 | 8;
            pr.g_off = b as u32;
            pr.sh_off = sc as u32;
        }
        let (p, a, n) = (self.p, self.a, self.n);
        let (mods, mrow, cw) = (self.mods, self.mrow, self.cw);
        let enc = self.rec.enc("row");
        enc.set_compute_pipeline_state(&p.rowop);
        enc.set_buffer(0, Some(&a.x), 0);
        enc.set_buffer(1, Some(&a.y), 0);
        enc.set_buffer(2, Some(&a.xn), 0);
        enc.set_buffer(3, Some(mods), 0);
        enc.set_buffer(4, Some(mrow), 0);
        enc.set_buffer(5, Some(cw), 0);
        set_p(enc, 6, &pr);
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
}

/// Probe sites of `CMF_H3_METAL_AMAX=1`.
const AMAX_SITES: [&str; 5] = ["attn-in", "qkv", "attn-out", "mlp-in", "mlp-mid"];

// ───────────────────────────── the chain ─────────────────────────────

/// The stack over `h` `[n][hidden]`, in place. `false`: nothing was written
/// and the caller runs its per-op path.
pub(crate) fn chain(job: &H3Chain<'_>, h: &mut [f32]) -> bool {
    if !enabled() {
        return false;
    }
    if job.qkv_interleaved {
        return vit_chain(job, h);
    }
    let (n, hs, nh, hd, ffn) = (job.n, job.hidden, job.heads, job.head_dim, job.ffn);
    let inner = nh * hd;
    let nb = job.blocks.len();
    if job.groups != 1 || hd != 128 || nb == 0 || n < 8 || h.len() != n * hs {
        return not_mine("a stack of batches or heads that are not 128 wide");
    }
    if job.blocks.iter().any(|b| {
        !b.qkv_bias.is_empty()
            || !b.out_bias.is_empty()
            || !b.fc1_bias.is_empty()
            || !b.fc2_bias.is_empty()
            || !b.scale1.is_empty()
            || !b.scale2.is_empty()
            || b.q_norm.len() != hd
            || b.k_norm.len() != hd
            || b.norm1.len() != hs
            || b.norm2.len() != hs
    }) {
        return not_mine("a stack with biases, residual scales or no q/k norm");
    }
    if !hs.is_multiple_of(256) || hs > 8192 || !ffn.is_multiple_of(64) {
        return not_mine("widths the kernels do not divide");
    }
    let m0 = job.blocks[0].mods.len();
    let modded = m0 != 0 && !job.rows.is_empty();
    let mrows = if modded { m0 / (EXPAND * hs) } else { 0 };
    if modded
        && (!m0.is_multiple_of(EXPAND * hs)
            || job.rows.len() != n
            || job.blocks.iter().any(|b| b.mods.len() != m0)
            || job.rows.iter().any(|&r| r as usize >= mrows))
    {
        return decline("the modulation tables do not match the rows");
    }
    let pairs = if job.angles.is_empty() { 0 } else { job.angles.len() / n };
    if pairs * n != job.angles.len() || 2 * pairs > hd || !pairs.is_multiple_of(4) {
        return decline("the rotation angles do not match the rows");
    }
    let Some((c, p)) = device() else { return false };
    let Some((arena, safe_len)) = super::file_buffer(c, job.model) else {
        return decline("the file mapping cannot be wrapped as a Metal buffer");
    };
    let mut ts = Vec::with_capacity(4 * nb);
    for b in job.blocks {
        for t in [
            tens(job.model, b.qkv, 3 * inner, hs, safe_len),
            tens(job.model, b.out, hs, inner, safe_len),
            tens(job.model, b.fc1, 2 * ffn, hs, safe_len),
            tens(job.model, b.fc2, hs, ffn, safe_len),
        ] {
            match t {
                Some(t) => ts.push(t),
                None => return not_mine("a projection that is not a q4tp tensor of the stack's shape"),
            }
        }
    }
    let t0 = std::time::Instant::now();
    let gd = Guards::from_env();

    let mut st = STATE.lock().unwrap();
    if !ensure_acts(c, &mut st, n, hs, 3 * inner, inner, ffn) {
        return false;
    }
    let a = st.0.as_ref().unwrap();

    // Per call: the norms' weights, the modulation tables, the rows' angles.
    let cstride = 2 * hs + 2 * hd;
    let mut cwv = Vec::with_capacity(nb * cstride);
    for b in job.blocks {
        cwv.extend_from_slice(b.norm1);
        cwv.extend_from_slice(b.norm2);
        cwv.extend_from_slice(b.q_norm);
        cwv.extend_from_slice(b.k_norm);
    }
    let cw = buf_from(c, &cwv);
    let (mods, mrow) = if modded {
        let mut m = Vec::with_capacity(nb * m0);
        for b in job.blocks {
            m.extend_from_slice(b.mods);
        }
        (buf_from(c, &m), buf_from(c, job.rows))
    } else {
        (buf_zeroed(c, 16), buf_zeroed(c, 16))
    };
    let csv: Vec<f32> = job
        .angles
        .iter()
        .flat_map(|&v| {
            let (s, c) = v.sin_cos();
            [c, s]
        })
        .collect();
    let cs = buf_from(c, &csv);
    let amax = (std::env::var("CMF_H3_METAL_AMAX").as_deref() == Ok("1")).then(|| buf_zeroed(c, 64));
    unsafe {
        std::ptr::copy_nonoverlapping(h.as_ptr(), a.x.contents() as *mut f32, n * hs);
    }

    let mut rec = Rec::new(c);
    {
        let mut e = Enc {
            rec: &mut rec,
            p,
            arena: &arena,
            a,
            n,
            hs,
            cw: &cw,
            aux: &cw,
            own: None,
            mods: &mods,
            mrow: &mrow,
            modded,
            eps: job.eps,
            amax: amax.as_ref(),
        };
        // a norm of block `l`: 0 = before the attention, 1 = before the MLP
        let norm = |l: usize, half: usize| (l * m0 + 3 * half * hs, l * m0 + (3 * half + 1) * hs, l * cstride + half * hs);
        let act = p2(-gd.act);
        // long passes one block a command buffer, short ones two
        let per_cut = if n > 2048 { 1 } else { 2 };
        e.rowop(None, Some(norm(0, 0)), act);
        for l in 0..nb {
            let w = &ts[4 * l..4 * l + 4];
            e.amax(&a.xn, n * hs, 0);
            e.gemm(w[0], &a.xn, &a.panel, 0, 3 * inner, true, p2(gd.act - gd.qkv));
            e.amax(&a.panel, n * 3 * inner, 1);
            {
                let pq = PQk {
                    ldp: (3 * inner) as u32,
                    nq: nh as u32,
                    nk: nh as u32,
                    k_col: inner as u32,
                    pairs: pairs as u32,
                    w_off: (l * cstride + 2 * hs) as u32,
                    // q/k arrive ×2^-qkv: eps ×2^-2qkv keeps the norm exact
                    eps: job.qk_eps * p2(-2 * gd.qkv),
                    qmul: std::f32::consts::LOG2_E / (hd as f32).sqrt(),
                };
                let enc = e.rec.enc("rope");
                enc.set_compute_pipeline_state(&p.qkrope);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(&cw), 0);
                enc.set_buffer(2, Some(&cs), 0);
                set_p(enc, 3, &pq);
                enc.dispatch_thread_groups(
                    MTLSize::new(n as u64, (2 * nh).div_ceil(4) as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
            {
                let pf = PFa {
                    q_row0: 0,
                    nq: n as u32,
                    n_pre: 0,
                    n_own: n as u32,
                    ldp: (3 * inner) as u32,
                    h: inner as u32,
                    ldo: inner as u32,
                    oscale: 1.0,
                    ..Default::default()
                };
                let enc = e.rec.enc("flash");
                enc.set_compute_pipeline_state(&p.flash);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(&a.attn), 0);
                // the prefix cache and the visibility list: unread here
                enc.set_buffer(2, Some(&a.panel), 0);
                enc.set_buffer(3, Some(&a.y), 0);
                set_p(enc, 4, &pf);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(64) as u64, nh as u64, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            e.amax(&a.attn, n * inner, 2);
            e.gemm(w[1], &a.attn, &a.y, 0, hs, false, p2(gd.qkv));
            e.rowop(Some(l * m0 + 2 * hs), Some(norm(l, 1)), act);
            e.amax(&a.xn, n * hs, 3);
            e.gemm(w[2], &a.xn, &a.gu, 0, 2 * ffn, false, p2(gd.act));
            {
                let ps = PSw {
                    n4: (n * ffn / 4) as u32,
                    ffn: ffn as u32,
                    oscale: p2(-gd.hid),
                    has_b: 0,
                };
                let enc = e.rec.enc("swiglu");
                enc.set_compute_pipeline_state(&p.swiglu);
                enc.set_buffer(0, Some(&a.gu), 0);
                enc.set_buffer(1, Some(&a.hid), 0);
                set_p(enc, 2, &ps);
                enc.set_buffer(3, Some(&cw), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new((ps.n4 as u64).div_ceil(256), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            e.amax(&a.hid, n * ffn, 4);
            e.gemm(w[3], &a.hid, &a.y, 0, hs, false, p2(gd.hid));
            let next = (l + 1 < nb).then(|| norm(l + 1, 0));
            e.rowop(Some(l * m0 + 5 * hs), next, act);
            if (l + 1).is_multiple_of(per_cut) {
                e.rec.cut();
            }
        }
    }
    let what = format!("{n} rows x {nb} blocks");
    if !rec.finish(&what) {
        return decline("a command buffer of the chain failed");
    }
    if let Some(b) = &amax {
        let v = unsafe { std::slice::from_raw_parts(b.contents() as *const u32, 5) };
        let sh = [gd.act, gd.qkv, gd.qkv, gd.act, gd.hid];
        let s: Vec<String> = AMAX_SITES
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let m = f32::from_bits(v[i]);
                format!("{name} {m:.1} (×2^{} = {:.3e})", sh[i], m * p2(sh[i]))
            })
            .collect();
        eprintln!("h3 metal amax, {what} (stored): {}", s.join(" · "));
    }
    let out = unsafe { std::slice::from_raw_parts(a.x.contents() as *const f32, n * hs) };
    if out.iter().any(|v| !v.is_finite()) {
        return decline("the chain produced a non-finite stream (a half range guard overflowed?)");
    }
    h.copy_from_slice(out);
    if std::env::var("CMF_H3_METAL_PROF").as_deref() == Ok("1") {
        eprintln!("h3 metal chain, {what}: {:.3} s", t0.elapsed().as_secs_f64());
    }
    true
}

// ───────────────────────── the prompt encoder ─────────────────────────

/// The layers of a language-model prompt encoder (grouped-query attention
/// with per-head q/k norms and a rotation over the whole head, a SwiGLU MLP
/// with separate gate and up projections, plain residuals) over one causal
/// sequence `h` `[n][hidden]`, in place. `false`: nothing was written.
///
/// The weights pass through a buffer of their own, a layer at a time
/// (253 MB of the 12.7 GB), copied out of the file mapping by eight
/// threads: read in place they sit in a window of the mapping the GPU first
/// makes resident whole, and on a 24 GB MacBook Air M4, next to the DiT's
/// pages, that took 35 s of a 38 s call whose kernels ran 3.5 s — against
/// 17–21 s for the stage on the host path.
/// `CMF_H3_METAL_TE=0` keeps the host path.
pub(crate) fn te_chain(job: &TeChain<'_>, h: &mut [f32]) -> bool {
    if !enabled() || std::env::var("CMF_H3_METAL_TE").as_deref() == Ok("0") {
        return false;
    }
    let (n, hs, nh, nkv, hd, inter) = (job.n, job.hidden, job.heads, job.kv_heads, job.head_dim, job.inter);
    let nl = job.layers.len();
    if n < 8 || nl == 0 || h.len() != n * hs || hd != 128 || nkv == 0 || !nh.is_multiple_of(nkv) {
        return not_mine("an encoder whose heads are not 128 wide or do not group");
    }
    if !hs.is_multiple_of(256) || hs > 8192 || !inter.is_multiple_of(64) {
        return not_mine("widths the kernels do not divide");
    }
    let pairs = hd / 2;
    if job.cs.len() != n * pairs * 2 {
        return decline("the encoder's rotation table does not match the rows");
    }
    if job
        .layers
        .iter()
        .any(|l| l.input_norm.len() != hs || l.post_norm.len() != hs || l.q_norm.len() != hd || l.k_norm.len() != hd)
    {
        return not_mine("an encoder layer without q/k norms");
    }
    let (qd, kd) = (nh * hd, nkv * hd);
    let pw = qd + 2 * kd;
    let Some((c, p)) = device() else { return false };
    let Some((arena, safe_len)) = super::file_buffer(c, job.model) else {
        return decline("the file mapping cannot be wrapped as a Metal buffer");
    };
    let mut ts = Vec::with_capacity(7 * nl);
    for l in job.layers {
        for t in [
            tens(job.model, l.q, qd, hs, safe_len),
            tens(job.model, l.k, kd, hs, safe_len),
            tens(job.model, l.v, kd, hs, safe_len),
            tens(job.model, l.o, hs, qd, safe_len),
            tens(job.model, l.gate, inter, hs, safe_len),
            tens(job.model, l.up, inter, hs, safe_len),
            tens(job.model, l.down, hs, inter, safe_len),
        ] {
            match t {
                Some(t) => ts.push(t),
                None => return not_mine("an encoder projection that is not a q4tp tensor of the stack's shape"),
            }
        }
    }
    let t0 = std::time::Instant::now();
    let gd = Guards::from_env();
    let mut st = STATE.lock().unwrap();
    if !ensure_acts(c, &mut st, n, hs, pw, qd, inter) {
        return false;
    }
    let a = st.0.as_ref().unwrap();
    let gl = a.alloc * inter; // the up plane's start in `gu`, floats

    let cstride = 2 * hs + 2 * hd;
    let mut cwv = Vec::with_capacity(nl * cstride);
    for l in job.layers {
        cwv.extend_from_slice(l.input_norm);
        cwv.extend_from_slice(l.post_norm);
        cwv.extend_from_slice(l.q_norm);
        cwv.extend_from_slice(l.k_norm);
    }
    let cw = buf_from(c, &cwv);
    let cs = buf_from(c, job.cs);
    let none = buf_zeroed(c, 16);
    unsafe {
        std::ptr::copy_nonoverlapping(h.as_ptr(), a.x.contents() as *mut f32, n * hs);
    }
    // one layer's tensors, back to back
    let bytes = job.model.primary_bytes();
    let tlen = |t: &Tens| (t.codes_off + t.rows * t.stride).next_multiple_of(16);
    let layer_bytes = ts.chunks(7).map(|l| l.iter().map(tlen).sum::<usize>()).max().unwrap_or(0);
    let wown = buf_zeroed(c, layer_bytes + 64);

    let mut rec = Rec::new(c);
    {
        let mut e = Enc {
            rec: &mut rec,
            p,
            arena: &arena,
            a,
            n,
            hs,
            cw: &cw,
            aux: &cw,
            own: Some(&wown),
            mods: &none,
            mrow: &none,
            modded: false,
            eps: job.eps,
            amax: None,
        };
        let act = p2(-gd.act);
        e.rowop(None, Some((0, 0, 0)), act);
        for l in 0..nl {
            // the layer's weights into the buffer the previous layer is
            // done with
            if !e.rec.drain() {
                return decline("a command buffer of the encoder chain failed");
            }
            let mut w = [ts[7 * l]; 7];
            let mut at = 0usize;
            for (k, t) in ts[7 * l..7 * l + 7].iter().enumerate() {
                let len = t.codes_off + t.rows * t.stride;
                // SAFETY: the layer's tensors fit `wown` (sized by the
                // largest layer); `tens` checked the source range.
                unsafe {
                    par_copy(&bytes[t.abs..t.abs + len], (wown.contents() as *mut u8).add(at));
                }
                w[k] = Tens { abs: at, ..*t };
                at += tlen(t);
            }
            let os = p2(gd.act - gd.qkv);
            e.gemm(w[0], &a.xn, &a.panel, 0, pw, true, os);
            e.gemm(w[1], &a.xn, &a.panel, qd, pw, true, os);
            e.gemm(w[2], &a.xn, &a.panel, qd + kd, pw, true, os);
            {
                let pq = PQk {
                    ldp: pw as u32,
                    nq: nh as u32,
                    nk: nkv as u32,
                    k_col: qd as u32,
                    pairs: pairs as u32,
                    w_off: (l * cstride + 2 * hs) as u32,
                    eps: job.eps * p2(-2 * gd.qkv),
                    qmul: std::f32::consts::LOG2_E / (hd as f32).sqrt(),
                };
                let enc = e.rec.enc("rope");
                enc.set_compute_pipeline_state(&p.qkrope);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(&cw), 0);
                enc.set_buffer(2, Some(&cs), 0);
                set_p(enc, 3, &pq);
                enc.dispatch_thread_groups(
                    MTLSize::new(n as u64, (nh + nkv).div_ceil(4) as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
            {
                let pf = PFg {
                    nq: n as u32,
                    nseq: 0,
                    ldp: pw as u32,
                    k_col: qd as u32,
                    v_col: (qd + kd) as u32,
                    rep: (nh / nkv) as u32,
                    ldo: qd as u32,
                    oscale: 1.0,
                    causal: 1,
                    hstride: hd as u32,
                    ..Default::default()
                };
                let enc = e.rec.enc("flash");
                enc.set_compute_pipeline_state(&p.flash_g);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(&a.attn), 0);
                set_p(enc, 2, &pf);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(64) as u64, nh as u64, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            e.gemm(w[3], &a.attn, &a.y, 0, hs, false, p2(gd.qkv));
            e.rowop(Some(0), Some((0, 0, l * cstride + hs)), act);
            e.gemm(w[4], &a.xn, &a.gu, 0, inter, false, p2(gd.act));
            e.gemm(w[5], &a.xn, &a.gu, gl, inter, false, p2(gd.act));
            {
                let ps = PSw2 {
                    n4: (n * inter / 4) as u32,
                    g_off: 0,
                    u_off: gl as u32,
                    h_off: 0,
                    oscale: p2(-gd.hid),
                    ..Default::default()
                };
                let enc = e.rec.enc("swiglu");
                enc.set_compute_pipeline_state(&p.swiglu2);
                enc.set_buffer(0, Some(&a.gu), 0);
                enc.set_buffer(1, Some(&a.hid), 0);
                set_p(enc, 2, &ps);
                enc.dispatch_thread_groups(
                    MTLSize::new((ps.n4 as u64).div_ceil(256), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            e.gemm(w[6], &a.hid, &a.y, 0, hs, false, p2(gd.hid));
            let next = (l + 1 < nl).then(|| (0, 0, (l + 1) * cstride));
            e.rowop(Some(0), next, act);
        }
    }
    let what = format!("encoder, {n} rows x {nl} layers");
    if !rec.finish(&what) {
        return decline("a command buffer of the encoder chain failed");
    }
    let out = unsafe { std::slice::from_raw_parts(a.x.contents() as *const f32, n * hs) };
    if out.iter().any(|v| !v.is_finite()) {
        return decline("the encoder chain produced a non-finite stream (a half range guard overflowed?)");
    }
    h.copy_from_slice(out);
    if std::env::var("CMF_H3_METAL_PROF").as_deref() == Ok("1") {
        eprintln!("h3 metal chain, {what}: {:.3} s", t0.elapsed().as_secs_f64());
    }
    true
}

// ───────────────────────── the VAE's ViT decoder ─────────────────────────

/// A ViT stack — the decoder of the MiniMax-H3 VAEs: projections with
/// biases, a layer scale on each residual, a weightless q/k norm, 64-wide
/// heads as per-head `[q k v]` triples — over a batch of `groups`
/// independent sequences of `n` rows, `h` `[groups·n][hidden]` in place.
/// Its weights are read in place (q4tp, or int8 with the row scale in the
/// GEMM's epilogue). `false`: nothing was written.
/// `CMF_H3_METAL_VIT=0` keeps the per-op path.
fn vit_chain(job: &H3Chain<'_>, h: &mut [f32]) -> bool {
    if std::env::var("CMF_H3_METAL_VIT").as_deref() == Ok("0") {
        return false;
    }
    let (n, hs, nh, hd, ffn, groups) = (job.n, job.hidden, job.heads, job.head_dim, job.ffn, job.groups);
    let inner = nh * hd;
    let nb = job.blocks.len();
    let rows = groups * n;
    if nb == 0 || n < 8 || groups == 0 || h.len() != rows * hs || hd != 64 {
        return not_mine("a ViT stack whose heads are not 64 wide");
    }
    if !hs.is_multiple_of(256) || hs > 8192 || !inner.is_multiple_of(64) || !ffn.is_multiple_of(64) {
        return not_mine("widths the kernels do not divide");
    }
    if !job.rows.is_empty()
        || job.blocks.iter().any(|b| {
            !b.mods.is_empty()
                || b.qkv_bias.len() != 3 * inner
                || b.out_bias.len() != hs
                || b.fc1_bias.len() != 2 * ffn
                || b.fc2_bias.len() != hs
                || b.scale1.len() != hs
                || b.scale2.len() != hs
                || b.norm1.len() != hs
                || b.norm2.len() != hs
                || !b.q_norm.is_empty()
                || !b.k_norm.is_empty()
        })
    {
        return not_mine("a ViT stack without biases and layer scales, or with q/k norm weights");
    }
    let pairs = if job.angles.is_empty() { 0 } else { job.angles.len() / n };
    if pairs * n != job.angles.len() || 2 * pairs > hd || !pairs.is_multiple_of(2) {
        return decline("the rotation angles do not match the rows");
    }
    let Some((c, p)) = device() else { return false };
    let Some((arena, safe_len)) = super::file_buffer(c, job.model) else {
        return decline("the file mapping cannot be wrapped as a Metal buffer");
    };
    let mut auxv: Vec<f32> = Vec::new();
    let mut ts = Vec::with_capacity(4 * nb);
    for b in job.blocks {
        for (idx, r, k) in [(b.qkv, 3 * inner, hs), (b.out, hs, inner), (b.fc1, 2 * ffn, hs), (b.fc2, hs, ffn)] {
            match tens_any(job.model, idx, r, k, safe_len, &mut auxv) {
                Some(t) => ts.push(t),
                None => return not_mine("a ViT projection in a codec the chain does not read"),
            }
        }
    }
    let t0 = std::time::Instant::now();
    let gd = Guards::from_env();
    // The decoder's weights in a buffer of their own (0.4 GB for the image
    // VAE). Read in place, the GPU first makes their whole 13.6 GB window
    // of the file mapping resident: 5.6 s of a 7.6 s call whose kernels ran
    // 2.0 s (MacBook Air M4, 25 tiles).
    let mut vw = VITW.lock().unwrap();
    let key = (job.model.uid(), job.blocks[0].qkv, nb);
    if !vw.as_ref().is_some_and(|w| w.key == key) {
        *vw = None;
        let bytes = job.model.primary_bytes();
        let len = |t: &Tens| if t.fmt == 1 { t.rows * t.cols } else { t.codes_off + t.rows * t.stride };
        let mut offs = Vec::with_capacity(ts.len());
        let mut total = 0usize;
        for t in &ts {
            offs.push(total);
            total += len(t).next_multiple_of(16);
        }
        let buf = buf_zeroed(c, total + 64);
        for (t, &o) in ts.iter().zip(&offs) {
            // SAFETY: `o + len` lies inside the buffer sized above; the
            // source range was checked against the mapping by `tens_any`.
            unsafe {
                par_copy(&bytes[t.abs..t.abs + len(t)], (buf.contents() as *mut u8).add(o));
            }
        }
        *vw = Some(VitW { key, buf, offs });
    }
    let vw = vw.as_ref().unwrap();
    for (t, &o) in ts.iter_mut().zip(&vw.offs) {
        t.abs = o;
    }
    let wown = &vw.buf;
    let mut st = STATE.lock().unwrap();
    if !ensure_acts(c, &mut st, rows, hs, 3 * inner, inner, ffn) {
        return false;
    }
    let a = st.0.as_ref().unwrap();

    // Per block in `cw`: the two norms, the qkv bias (the projection's own
    // order), the output bias and its layer scale, fc1's bias, fc2's and
    // its scale.
    let (o_n2, o_qb) = (hs, 2 * hs);
    let o_ob = o_qb + 3 * inner;
    let o_s1 = o_ob + hs;
    let o_f1 = o_s1 + hs;
    let o_f2 = o_f1 + 2 * ffn;
    let o_s2 = o_f2 + hs;
    let cstride = o_s2 + hs;
    let mut cwv = Vec::with_capacity(nb * cstride);
    for b in job.blocks {
        cwv.extend_from_slice(b.norm1);
        cwv.extend_from_slice(b.norm2);
        cwv.extend_from_slice(b.qkv_bias);
        cwv.extend_from_slice(b.out_bias);
        cwv.extend_from_slice(b.scale1);
        cwv.extend_from_slice(b.fc1_bias);
        cwv.extend_from_slice(b.fc2_bias);
        cwv.extend_from_slice(b.scale2);
    }
    let cw = buf_from(c, &cwv);
    let aux = buf_from(c, &auxv);
    let csv: Vec<f32> = job
        .angles
        .iter()
        .flat_map(|&v| {
            let (s, c) = v.sin_cos();
            [c, s]
        })
        .collect();
    let cs = buf_from(c, &csv);
    let none = buf_zeroed(c, 16);
    unsafe {
        std::ptr::copy_nonoverlapping(h.as_ptr(), a.x.contents() as *mut f32, rows * hs);
    }

    let mut rec = Rec::new(c);
    {
        let mut e = Enc {
            rec: &mut rec,
            p,
            arena: &arena,
            a,
            n: rows,
            hs,
            cw: &cw,
            aux: &aux,
            own: Some(wown),
            mods: &none,
            mrow: &none,
            modded: false,
            eps: job.eps,
            amax: None,
        };
        let act = p2(-gd.act);
        let per_cut = if rows > 2048 { 2 } else { 4 };
        e.rowop(None, Some((0, 0, 0)), act);
        for l in 0..nb {
            let w = &ts[4 * l..4 * l + 4];
            let base = l * cstride;
            e.gemm(w[0], &a.xn, &a.panel, 0, 3 * inner, true, p2(gd.act - gd.qkv));
            {
                let pq = PVq {
                    ldp: (3 * inner) as u32,
                    nh: nh as u32,
                    pairs: pairs as u32,
                    b_off: (base + o_qb) as u32,
                    nseq: n as u32,
                    bscale: p2(-gd.qkv),
                    eps: job.qk_eps * p2(-2 * gd.qkv),
                    qmul: std::f32::consts::LOG2_E / (hd as f32).sqrt(),
                };
                let enc = e.rec.enc("rope");
                enc.set_compute_pipeline_state(&p.vit_qk);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(&cw), 0);
                enc.set_buffer(2, Some(&cs), 0);
                set_p(enc, 3, &pq);
                enc.dispatch_thread_groups(
                    MTLSize::new(rows as u64, (3 * nh).div_ceil(4) as u64, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
            {
                let pf = PFg {
                    nq: n as u32,
                    nseq: n as u32,
                    ldp: (3 * inner) as u32,
                    k_col: hd as u32,
                    v_col: (2 * hd) as u32,
                    rep: 1,
                    ldo: inner as u32,
                    oscale: 1.0,
                    causal: 0,
                    hstride: (3 * hd) as u32,
                    ..Default::default()
                };
                let enc = e.rec.enc("flash");
                enc.set_compute_pipeline_state(&p.flash64);
                enc.set_buffer(0, Some(&a.panel), 0);
                enc.set_buffer(1, Some(&a.attn), 0);
                set_p(enc, 2, &pf);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(64) as u64, nh as u64, groups as u64),
                    MTLSize::new(256, 1, 1),
                );
            }
            e.gemm(w[1], &a.attn, &a.y, 0, hs, false, p2(gd.qkv));
            e.rowop_b(None, Some((base + o_ob, base + o_s1)), Some((0, 0, base + o_n2)), act);
            e.gemm(w[2], &a.xn, &a.gu, 0, 2 * ffn, false, p2(gd.act));
            {
                let ps = PSw {
                    n4: (rows * ffn / 4) as u32,
                    ffn: ffn as u32,
                    oscale: p2(-gd.hid),
                    has_b: 1,
                };
                let enc = e.rec.enc("swiglu");
                enc.set_compute_pipeline_state(&p.swiglu);
                enc.set_buffer(0, Some(&a.gu), 0);
                enc.set_buffer(1, Some(&a.hid), 0);
                set_p(enc, 2, &ps);
                enc.set_buffer(3, Some(&cw), ((base + o_f1) * 4) as u64);
                enc.dispatch_thread_groups(
                    MTLSize::new((ps.n4 as u64).div_ceil(256), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            e.gemm(w[3], &a.hid, &a.y, 0, hs, false, p2(gd.hid));
            let next = (l + 1 < nb).then(|| (0, 0, (l + 1) * cstride));
            e.rowop_b(None, Some((base + o_f2, base + o_s2)), next, act);
            if (l + 1).is_multiple_of(per_cut) {
                e.rec.cut();
            }
        }
    }
    let what = format!("ViT, {groups} x {n} rows x {nb} blocks");
    if !rec.finish(&what) {
        return decline("a command buffer of the ViT chain failed");
    }
    let out = unsafe { std::slice::from_raw_parts(a.x.contents() as *const f32, rows * hs) };
    if out.iter().any(|v| !v.is_finite()) {
        return decline("the ViT chain produced a non-finite stream (a half range guard overflowed?)");
    }
    h.copy_from_slice(out);
    if std::env::var("CMF_H3_METAL_PROF").as_deref() == Ok("1") {
        eprintln!("h3 metal chain, {what}: {:.3} s", t0.elapsed().as_secs_f64());
    }
    true
}
