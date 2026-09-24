//! Z-Image-Turbo / Z-Image on native Metal (plan WP3). OWNER: WP3 — this file only.
//!
//! Implements the `gpu::zimage_*` / `gpu::vae_decode_chain` contract
//! (`gpu.rs`, "Z-Image-Turbo device contract"). Rules (plan §2.2):
//! - reach the parent's private kernels, MSL sources, `ctx()` and buffer
//!   pools through `super::`; new MSL goes into this module's own library
//!   (`zimage_msl.metal`); never edit parent functions;
//! - build pipelines into a module-local, lazily created cache (`OnceLock`);
//!   never add fields to the parent's context struct;
//! - `release` touches module-local state only and must never bring a
//!   device up (it is called unconditionally by `gpu::zimage_release`);
//! - every entry returns `false` before changing any output when it cannot
//!   honour the full contract — the caller then runs the CPU path.
//!
//! # The chain
//!
//! Weights are the file's q8_2f / q8_row bytes, read in place through the
//! parent's no-copy file arena (no planes: the 24 GB of the M4 are shared
//! with the text encoder and the page cache). Each GEMM stages the int8
//! tile as half (exact) and applies the row scale in its f32 epilogue; the
//! q8_2f column field is folded into the half activation by the kernel
//! that produces it (the row ops for q/k/v/w1/w3, the flash epilogue for
//! the O projection, SwiGLU for w2). Every half site carries a
//! power-of-two guard ([`Guards`]) the consumer multiplies back.
//!
//! One step is: x_embed → 2 noise-refiner blocks (image rows) → caption
//! rows copied in → 30 layers → final layer; per block: row op (post-norm
//! residual + next pre-norm) → qkv GEMM (z = 3) → qk-norm + RoPE in place →
//! flash attention → O GEMM → row op → w1/w3 GEMM (z = 2) → SwiGLU → w2
//! GEMM. The CFG pair is one batch-2 program: rows [img0 | img1 | cap0 |
//! cap1], every row-local kernel runs once over all rows, attention runs
//! per item over its two row ranges. The context refiner runs the same
//! block encoder unmodulated; the VAE is `vae.rs` (resident, NHWC).
//!
//! # Measured (Mac mini M4, 10-core GPU, 24 GB; in-process GPU timers)
//!
//! `examples/zimage_metal_bench.rs` (peak / gemm / plane / flash / corun)
//! and `examples/zimage_metal_check.rs` (one forward vs the CPU path and
//! the fp32 oracle, per-class GPU ms with `CMF_ZI_METAL_PROF=1`).
//!
//! - Ceiling: a pure simdgroup-MMA loop (16 independent 8×8 accumulators,
//!   no loads) issues 3.55 TF/s with half operands, 3.36-3.46 with float
//!   operands, 3.69 with half accumulators (not used: +4% for f16 sums).
//! - GEMM `zi_q8mm`: 3.2-3.35 TF/s on all four DiT shapes at 1056 and 4224
//!   tokens = 91-94% of that ceiling; rel 5e-7 against f64 on the same
//!   operands. Measured and rejected (do not reopen): the transposed-W
//!   staging (`CMF_ZI_MM=wt`, equal); 32-accumulator tiles 64×128 /
//!   128×64 (register spill, 0.5 TF/s); 256-thread 64×128 (−2%); 32×128
//!   (−5%); f16 weight planes (the same tile with half weights: equal or up
//!   to −6%, and 11.6 GB the shared 24 GB cannot spare).
//! - Flash `zi_flash_q64pf` (64 queries = 8 simdgroups × 8, 32-key blocks
//!   staged in threadgroup memory, register prefetch of the next block):
//!   1.50 TF/s at 1056 tokens, 1.65-1.73 at 4224; 2.6e-4 against f64 (the
//!   f16 output floor). Attribution at 4224 (per layer): full 162 ms, Q·Kᵀ
//!   ≈ 57 and P·V ≈ 60 against an ideal 39 each (one threadgroup load per
//!   MMA), the MMA-free skeleton (staging, softmax, barriers) 45 ms.
//!   Rejected: 32 queries (v1, −7..−11%), head-dim-outer Q·Kᵀ (−6%),
//!   transposed-K staging (equal), split partial sums (−20%), direct device
//!   loads without threadgroup memory (−15..−40%), 128 queries (−7%),
//!   double-buffered K/V tiles (32 KB, −7%), K/V tiles packed as dense
//!   8×8 blocks (−2%: not bank conflicts). What is left is structural: one
//!   threadgroup fragment load per MMA (the GEMM has one per two); the next
//!   step would split the head dim across simdgroup pairs (16 queries a
//!   pair, partial Q·Kᵀ exchanged through threadgroup memory).
//! - One step, Turbo 512² (1056 rows): 4.1-4.2 s = GEMM 3.70 + flash 0.35
//!   + row ops 0.09 (3.0 TF/s effective); 1024² (4128 rows): 19.7 s = GEMM
//!   14.4 + flash 4.84 + row ops 0.37 (2.8 TF/s), rising to 21-23 s as the
//!   machine heats up over a run. Host work per step (upload, patchify,
//!   readback, Euler) is < 1%: wall ≈ GPU time.
//! - Command buffers: one per step vs one per two blocks (default, each
//!   buffer ≤ ~1.3 s at 1024²): equal (4.11 vs 4.11 s, 3 alternating pairs).
//! - CFG: the batch-2 program costs 2 × the single forward (8.11 s vs
//!   4.06 s at 512²; compute-bound, no batching gain) and is bit-identical
//!   to the two single forwards.
//! - CPU share (M8, `cpu.rs`): Accelerate sgemm runs 1.65 TF/s alone and
//!   1.14–1.23 beside the busy GPU (30 s co-run, the GPU 3.28 → 3.10), so
//!   the CPU computes the last output features of every GEMM of ≥ 256
//!   rows, ordered with the chain by an `MTLSharedEvent`. One-step sweep at
//!   512² (6 steps a value, alternating): share 0 → 3.99–4.04 s, 0.20 →
//!   3.30–3.43, 0.25 → 3.14–3.36, 0.30 → 3.10–3.65, 0.35 → 3.43–3.92.
//!   Whole CLI, alternating with cool-downs: Turbo 512² 34.5 → 30.1 s per
//!   image at 0.25 (−13 %), 1024² 165.4 → 155.2 s at 0.20 (−6 %: the gain
//!   is largest on the cool first steps and fades as the package heats);
//!   base CFG pair at 512² 8.11 → 6.66 s a step. The CPU's features are
//!   f32, so a step is closer to the CPU path (Turbo r512 v 1.15e-3 →
//!   7.0e-4, base 2.7e-4 → 2.2e-4). The default is fixed per chip (a
//!   seed gives the same image every run; measured on "Apple M4" only,
//!   off elsewhere); `CMF_ZI_CPU_FRAC=<x>` fixes a share, `=auto` runs a
//!   controller that moves it ±0.03 a forward toward the CPU and GPU parts
//!   finishing together (it settles at 0.20–0.27 on the M4 at 512²; the
//!   partition then depends on timing and the image is not bit-stable).
//!
//! Range guards ([`Guards`], measured with `CMF_ZI_AMAX=1`, stored value
//! after the guard, worst block): Turbo over 8 steps at 512² — q/k/v input
//! 102, qkv panel 5720 (guard 2⁻¹ → 2860), attention output 202, FFN input
//! 19, hidden 310 (with 2⁻⁶); base r512 c3 step 0 — 784, 9192 (at 2⁻⁷;
//! now 2⁻⁸), 734, 9, 1350 (at 2⁻¹¹).
//!
//! Knobs: `CMF_ZI_METAL=0` (device path off), `CMF_ZI_MM` / `CMF_ZI_FLASH`
//! (kernel variants, A/B only), `CMF_ZI_METAL_CHUNK` (blocks per command
//! buffer, default 2), `CMF_ZI_METAL_PROF=1` (per-class GPU ms; one command
//! buffer per op), `CMF_ZI_AMAX=1`, `CMF_ZI_{ATTN,QKV,AO,FFN,HID}_SHIFT`,
//! `CMF_ZI_METAL_REFINE=0` (CPU context refiner), `CMF_ZI_VAE=0` (the
//! per-conv VAE), `CMF_ZI_VAE_CHUNK` (attention query chunk),
//! `CMF_ZI_CPU_FRAC=<x>|auto` (the CPU share; 0 = GPU only),
//! `CMF_ZI_CPU_THREADS` (conversion workers, 4), `CMF_ZI_CPU_PROF=1`.

use crate::gpu::{ZBlockRef, ZGeom, ZPrepareArgs, ZStepArgs};
use cortiq_core::{CmfModel, TensorDtype};
use metal::{Buffer, CommandBuffer, ComputeCommandEncoderRef, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use super::{Ctx, WeightArena};

const ZMSL: &str = include_str!("zimage_msl.metal");

mod cpu;
mod vae;

/// `CMF_ZI_METAL=0` turns the device path off (the caller runs the CPU).
fn zi_enabled() -> bool {
    std::env::var("CMF_ZI_METAL").as_deref() != Ok("0")
}

/// Says why the device path declined, once per reason per process.
fn decline(reason: &str) -> bool {
    static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
    if let Ok(mut v) = SAID.lock() {
        if !v.iter().any(|r| r == reason) {
            eprintln!("zimage: Metal device path declined: {reason}; the DiT runs on the CPU");
            v.push(reason.to_string());
        }
    }
    false
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

// ───────────────────────────── pipelines ─────────────────────────────

/// A GEMM kernel variant: pipeline, tile tokens, tile features, threads.
struct MmVar {
    name: &'static str,
    pso: ComputePipelineState,
    bt: usize,
    bo: usize,
    threads: u64,
}

/// (name, kernel, tile tokens, tile features, threads) of every GEMM
/// variant; "base" is the v1 kernel (`zi_q8mm`), "wt" its transposed-W twin.
const MM_VARIANTS: &[(&str, &str, usize, usize, u64)] = &[
    ("base", "zi_q8mm", 64, 64, 128),
    ("wt", "zi_q8mm_wt", 64, 64, 128),
];

/// Flash-attention variants (name, kernel); the first is the default.
const FLASH_VARIANTS: &[(&str, &str, usize)] = &[
    ("q64pf", "zi_flash_q64pf", 8),
    ("q32", "zi_flash_q32", 4),
    ("nopv", "zi_flash_nopv", 8),
    ("noqk", "zi_flash_noqk", 8),
    ("nomma", "zi_flash_nomma", 8),
    ("noload", "zi_flash_noload", 8),
];

/// (pipeline, simdgroups) of the chosen flash variant.
fn flash_var(p: &Pipes) -> (&ComputePipelineState, usize) {
    static V: OnceLock<String> = OnceLock::new();
    let want = V.get_or_init(|| std::env::var("CMF_ZI_FLASH").unwrap_or_default());
    let f = p.flashes.iter().find(|f| f.0 == want).unwrap_or(&p.flashes[0]);
    (&f.1, f.2)
}

struct Pipes {
    mms: Vec<MmVar>,
    /// (name, pipeline) of the flash variants; `CMF_ZI_FLASH=<name>`.
    flashes: Vec<(&'static str, ComputePipelineState, usize)>,
    qkrope: ComputePipelineState,
    rowop: ComputePipelineState,
    swiglu: ComputePipelineState,
    embed: ComputePipelineState,
    fin: ComputePipelineState,
    copy4: ComputePipelineState,
    amax: ComputePipelineState,
    probe: ComputePipelineState,
    // resident VAE
    vconv: ComputePipelineState,
    vgnpart: ComputePipelineState,
    vgnfin: ComputePipelineState,
    vgnapply: ComputePipelineState,
    vcvt: ComputePipelineState,
    vsoftmax: ComputePipelineState,
    vrgb: ComputePipelineState,
}
// metal-rs objects are retained ObjC pointers; used under the state mutex.
unsafe impl Send for Pipes {}
unsafe impl Sync for Pipes {}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn build_pipes(c: &Ctx) -> Result<Pipes, String> {
    let opts = metal::CompileOptions::new();
    opts.set_language_version(metal::MTLLanguageVersion::V3_0);
    let lib = c
        ._device
        .new_library_with_source(ZMSL, &opts)
        .map_err(|e| format!("zimage MSL compile: {e}"))?;
    let pso = |name: &str| -> Result<ComputePipelineState, String> {
        let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
        c._device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline {name}: {e}"))
    };
    Ok(Pipes {
        mms: MM_VARIANTS
            .iter()
            .map(|&(name, k, bt, bo, threads)| Ok(MmVar { name, pso: pso(k)?, bt, bo, threads }))
            .collect::<Result<Vec<_>, String>>()?,
        flashes: FLASH_VARIANTS
            .iter()
            .map(|&(n, k, nsg)| Ok((n, pso(k)?, nsg)))
            .collect::<Result<Vec<_>, String>>()?,
        qkrope: pso("zi_qkrope")?,
        rowop: pso("zi_rowop")?,
        swiglu: pso("zi_swiglu")?,
        embed: pso("zi_embed")?,
        fin: pso("zi_final")?,
        copy4: pso("zi_copy4")?,
        amax: pso("zi_amax")?,
        probe: pso("zi_fragprobe")?,
        vconv: pso("zv_conv")?,
        vgnpart: pso("zv_gn_part")?,
        vgnfin: pso("zv_gn_fin")?,
        vgnapply: pso("zv_gn_apply")?,
        vcvt: pso("zv_cvt")?,
        vsoftmax: pso("zv_softmax")?,
        vrgb: pso("zv_rgb")?,
    })
}

/// The module's pipelines, compiled once; the fragment layout the kernels
/// assume is checked on the device the first time (a mismatch declines).
fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    match PIPES.get_or_init(|| {
        let p = build_pipes(c)?;
        if !frag_layout_ok(c, &p) {
            return Err("the simdgroup fragment layout differs from the one the kernels assume".into());
        }
        Ok(p)
    }) {
        Ok(p) => Some(p),
        Err(e) => {
            decline(e);
            None
        }
    }
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

// ───────────────────────────── weights ─────────────────────────────

/// Guards: the half sites store x·2^-s (the consumer multiplies 2^s back).
/// `attn` = the q/k/v GEMM input, `qkv` = the qkv panel, `ao` = the
/// attention output (O GEMM input), `ffn` = the w1/w3 input, `hid` = the
/// SwiGLU hidden (w2 input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Guards {
    pub attn: i32,
    pub qkv: i32,
    pub ao: i32,
    pub ffn: i32,
    pub hid: i32,
}

impl Guards {
    pub const TURBO: Guards = Guards { attn: 0, qkv: 1, ao: 0, ffn: 0, hid: 6 };
    pub const BASE: Guards = Guards { attn: 6, qkv: 8, ao: 6, ffn: 6, hid: 11 };

    fn for_model(model: &CmfModel) -> Guards {
        let variant = model
            .tensor_bytes("zimage.config_json")
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .and_then(|v| v["variant"].as_str().map(str::to_string));
        let g = if variant.as_deref() == Some("base") { Self::BASE } else { Self::TURBO };
        let e = |k: &str, d: i32| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d).clamp(0, 14)
        };
        Guards {
            attn: e("CMF_ZI_ATTN_SHIFT", g.attn),
            qkv: e("CMF_ZI_QKV_SHIFT", g.qkv),
            ao: e("CMF_ZI_AO_SHIFT", g.ao),
            ffn: e("CMF_ZI_FFN_SHIFT", g.ffn),
            hid: e("CMF_ZI_HID_SHIFT", g.hid),
        }
    }
}

fn p2(e: i32) -> f32 {
    (2.0f32).powi(e)
}

#[derive(Clone, Copy)]
struct Tens {
    abs: usize,
    rows: usize,
    cols: usize,
}

/// One block's device view: the seven int8 tensors in the file arena and
/// one f32 aux buffer [rs×7 | col×7 | norm1 | norm2 | ffn_norm1 | ffn_norm2 |
/// norm_q | norm_k].
struct ZBlk {
    t: [Tens; 7],
    aux: Buffer,
    rs: [usize; 7],
    col: [usize; 7],
    norm: [usize; 6],
}

const TQ: usize = 0;
const TK: usize = 1;
const TV: usize = 2;
const TO: usize = 3;
const T1: usize = 4;
const T3: usize = 5;
const T2: usize = 6;

fn f16s(b: &[u8]) -> impl Iterator<Item = f32> + '_ {
    b.chunks_exact(2)
        .map(|c| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
}

fn build_block(c: &Ctx, model: &CmfModel, r: &ZBlockRef, g: &ZGeom) -> Result<ZBlk, String> {
    let idx = [r.wq, r.wk, r.wv, r.wo, r.w1, r.w3, r.w2];
    let (h, inter) = (g.hidden, g.inter);
    let shapes = [(h, h), (h, h), (h, h), (h, h), (inter, h), (inter, h), (h, inter)];
    let mut t = [Tens { abs: 0, rows: 0, cols: 0 }; 7];
    let mut aux: Vec<f32> = Vec::new();
    let mut rs = [0usize; 7];
    let mut col = [0usize; 7];
    let mut rs_data: Vec<Vec<f32>> = Vec::new();
    let mut col_data: Vec<Vec<f32>> = Vec::new();
    for (i, (&ti, &(rows, cols))) in idx.iter().zip(&shapes).enumerate() {
        let e = model.tensors.get(ti).ok_or("tensor index out of range")?;
        if e.shape != [rows, cols] {
            return Err(format!("{}: shape {:?}, expected [{rows}, {cols}]", e.name, e.shape));
        }
        let abs = model.entry_abs_offset(e).ok_or_else(|| format!("{}: not in the primary shard", e.name))?;
        if abs % 16 != 0 {
            return Err(format!("{}: unaligned ({abs})", e.name));
        }
        let bytes = model.entry_bytes(e);
        let q = rows * cols;
        match e.dtype {
            TensorDtype::Q8_2f if bytes.len() == q + 2 * rows + 2 * cols => {
                rs_data.push(f16s(&bytes[q..q + 2 * rows]).collect());
                col_data.push(f16s(&bytes[q + 2 * rows..]).collect());
            }
            TensorDtype::Q8Row if bytes.len() == q + 2 * rows => {
                rs_data.push(f16s(&bytes[q..]).collect());
                col_data.push(vec![1.0; cols]);
            }
            d => return Err(format!("{}: codec {d:?} (the Metal chain reads q8_2f / q8_row)", e.name)),
        }
        t[i] = Tens { abs, rows, cols };
    }
    for i in 0..7 {
        rs[i] = aux.len();
        aux.extend_from_slice(&rs_data[i]);
    }
    for i in 0..7 {
        col[i] = aux.len();
        aux.extend_from_slice(&col_data[i]);
    }
    let mut norm = [0usize; 6];
    for (k, v) in [r.norm1, r.norm2, r.ffn_norm1, r.ffn_norm2, r.norm_q, r.norm_k].iter().enumerate() {
        norm[k] = aux.len();
        aux.extend_from_slice(v);
        while aux.len() % 4 != 0 {
            aux.push(0.0);
        }
    }
    Ok(ZBlk {
        t,
        aux: buf_from(c, &aux),
        rs,
        col,
        norm,
    })
}

fn buf_from(c: &Ctx, v: &[f32]) -> Buffer {
    c._device.new_buffer_with_data(
        v.as_ptr() as *const c_void,
        (v.len().max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn buf_zeroed(c: &Ctx, bytes: usize) -> Buffer {
    // new_buffer returns zero-filled memory for shared storage
    c._device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared)
}

fn write_f32(b: &Buffer, off_floats: usize, v: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(v.as_ptr(), (b.contents() as *mut f32).add(off_floats), v.len());
    }
}

// ───────────────────────────── state ─────────────────────────────

/// Activations of one program (rows = all items' rows, alloc = rows + 64).
struct Acts {
    rows: usize,
    alloc: usize,
    x: Buffer,
    xn: Buffer,    // half [3][alloc][H]
    panel: Buffer, // half [alloc][3H]
    attn: Buffer,  // half [alloc][H]
    y: Buffer,     // f32 [alloc][H]
    gu: Buffer,    // f32 [2][alloc][inter]
    h: Buffer,     // half [alloc][inter]
}

impl Acts {
    fn new(c: &Ctx, rows: usize, g: &ZGeom) -> Acts {
        let alloc = rows.div_ceil(128) * 128 + 128;
        let (h, i) = (g.hidden, g.inter);
        Acts {
            rows,
            alloc,
            x: buf_zeroed(c, alloc * h * 4),
            xn: buf_zeroed(c, 3 * alloc * h * 2),
            panel: buf_zeroed(c, alloc * 3 * h * 2),
            attn: buf_zeroed(c, alloc * h * 2),
            y: buf_zeroed(c, alloc * h * 4),
            gu: buf_zeroed(c, 2 * alloc * i * 4),
            h: buf_zeroed(c, alloc * i * 2),
        }
    }
}

/// Per-model device weights: kept across prompts until `release`.
struct ZDev {
    uid: u64,
    _model: Arc<CmfModel>,
    arena: Arc<WeightArena>,
    geom: ZGeom,
    guards: Guards,
    /// noise_refiner (2) then layers (30).
    step_blocks: Vec<ZBlk>,
    n_refiner: usize,
    ctx_blocks: Vec<ZBlk>,
    /// x_embedder W [H][64] | b [H] | x_pad [H]; final W [64][H] | b [64].
    emb: Option<Buffer>,
    fin: Option<Buffer>,
    progs: Vec<ZProg>,
}

/// One prepared (prompt(s), resolution) program.
struct ZProg {
    key: u64,
    n_img: usize,
    n_img_p: usize,
    /// (n_cap_p, first caption row) per item; image rows of item i start
    /// at i·n_img_p.
    items: Vec<(usize, usize)>,
    acts: Acts,
    cap: Buffer,
    rope_img: (Buffer, Buffer),
    rope_joint: (Buffer, Buffer),
    xtok: Buffer,
    mods: Buffer,
    fs: Buffer,
    out: Buffer,
}

struct ZState(Option<ZDev>);
unsafe impl Send for ZState {}

static ZSTATE: Mutex<ZState> = Mutex::new(ZState(None));

const MAX_PROGS: usize = 2;

fn geom_ok(g: &ZGeom) -> Result<(), String> {
    if g.hd != 128 || g.nh * g.hd != g.hidden || g.hidden % 256 != 0 || g.hidden > 4096 {
        return Err(format!("geometry {g:?} (the kernels need hd 128, hidden % 256 == 0, ≤ 4096)"));
    }
    if g.inter % 128 != 0 || g.hidden % 128 != 0 || g.patch_dim != 64 {
        return Err(format!("geometry {g:?} (inter % 128, patch 64)"));
    }
    Ok(())
}

/// The per-model device state for `model`, (re)built when the model
/// changes. The step blocks are built when missing; `ctx` adds the
/// context-refiner blocks.
fn ensure_dev<'s>(
    st: &'s mut ZState,
    c: &Ctx,
    model: &Arc<CmfModel>,
    geom: &ZGeom,
    step: Option<(&[ZBlockRef], &[ZBlockRef])>,
    ctx_blocks: Option<&[ZBlockRef]>,
) -> Result<&'s mut ZDev, String> {
    geom_ok(geom)?;
    if st.0.as_ref().is_some_and(|d| d.uid != model.uid() || d.geom != *geom) {
        st.0 = None;
    }
    if st.0.is_none() {
        let (arena, _len) = super::file_buffer(c, model).ok_or("the file mapping cannot be wrapped as a Metal buffer")?;
        st.0 = Some(ZDev {
            uid: model.uid(),
            _model: model.clone(),
            arena,
            geom: *geom,
            guards: Guards::for_model(model),
            step_blocks: Vec::new(),
            n_refiner: 0,
            ctx_blocks: Vec::new(),
            emb: None,
            fin: None,
            progs: Vec::new(),
        });
    }
    let d = st.0.as_mut().unwrap();
    if let Some((nr, layers)) = step {
        if d.step_blocks.len() != nr.len() + layers.len() {
            let mut v = Vec::with_capacity(nr.len() + layers.len());
            for r in nr.iter().chain(layers) {
                v.push(build_block(c, model, r, geom)?);
            }
            d.step_blocks = v;
            d.n_refiner = nr.len();
        }
    }
    if let Some(cb) = ctx_blocks {
        if d.ctx_blocks.len() != cb.len() {
            let mut v = Vec::with_capacity(cb.len());
            for r in cb {
                v.push(build_block(c, model, r, geom)?);
            }
            d.ctx_blocks = v;
        }
    }
    Ok(d)
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
    pad0: u32,
    x_off: [u32; 4],
    y_off: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PFa {
    img_off: u32,
    n_img: u32,
    cap_off: u32,
    n_cap: u32,
    ldp: u32,
    h: u32,
    ldo: u32,
    oscale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PQk {
    h: u32,
    nh: u32,
    ldp: u32,
    row0: u32,
    eps: f32,
    qmul: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PRow {
    h: u32,
    row0: u32,
    mode: u32,
    nout: u32,
    has_g: u32,
    has_s: u32,
    eps1: f32,
    eps2: f32,
    oscale: f32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PSw {
    n4: u32,
    inter: u32,
    g_off: u32,
    u_off: u32,
    h_off: u32,
    oscale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PEm {
    h: u32,
    n_img: u32,
    n_img_p: u32,
    nitems: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PFi {
    h: u32,
    row0: u32,
    out0: u32,
    eps: f32,
}

fn set_p<T>(enc: &ComputeCommandEncoderRef, idx: u64, v: &T) {
    enc.set_bytes(idx, std::mem::size_of::<T>() as u64, v as *const T as *const c_void);
}

/// Command-buffer recorder: one encoder per command buffer, a new buffer
/// every `chunk` blocks (committed without a wait; the queue orders them).
/// Under `CMF_ZI_METAL_PROF=1` every op gets its own buffer so its GPU
/// time can be attributed to a class.
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
    /// The encoder for the next op of class `label`.
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
    /// Signal (`signal`) or wait for `v` on the shared event, between the
    /// encoders of the current command buffer.
    fn event(&mut self, ev: &metal::SharedEventRef, v: u64, signal: bool) {
        if self.cur.is_none() {
            let cmd = self.c.queue.new_command_buffer().to_owned();
            self.cur = Some((cmd, None, "event"));
        }
        let cur = self.cur.as_mut().unwrap();
        if let Some(e) = cur.1.take() {
            e.end_encoding();
        }
        if signal {
            cur.0.encode_signal_event(ev, v);
        } else {
            cur.0.encode_wait_for_event(ev, v);
        }
    }
    /// Chunk boundary (normal mode only).
    fn cut(&mut self) {
        if !self.prof {
            self.close();
        }
    }
    /// Commit what is open, wait for everything; false on a GPU error.
    fn finish(&mut self) -> bool {
        self.close();
        let mut ok = true;
        if let Some((_, last)) = self.done.last() {
            last.wait_until_completed();
        }
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
            eprintln!("zimage metal gpu ms: total {tot:.1} · {}", s.join(" · "));
        }
        self.done.clear();
        ok
    }
    fn gpu_ms(&self) -> f64 {
        self.done.iter().map(|(_, c)| super::cmd_gpu_ms(c)).sum()
    }
}

/// The process-wide event of the CPU GEMM share and its monotone counter.
struct ZEvent(metal::SharedEvent);
unsafe impl Send for ZEvent {}
unsafe impl Sync for ZEvent {}

fn zevent(c: &Ctx) -> &'static metal::SharedEventRef {
    static E: OnceLock<ZEvent> = OnceLock::new();
    &E.get_or_init(|| ZEvent(c._device.new_shared_event())).0
}

/// GPU-only event: signaled after the GPU's part of every split GEMM (the
/// controller's "who finished first").
fn zevent2(c: &Ctx) -> &'static metal::SharedEventRef {
    static E: OnceLock<ZEvent> = OnceLock::new();
    &E.get_or_init(|| ZEvent(c._device.new_shared_event())).0
}

static ZEVENT_NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static ZEVENT2_NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The CPU jobs of one step (M8, `cpu.rs`).
struct CpuPlan {
    frac: f64,
    ev: &'static metal::SharedEventRef,
    ev2: &'static metal::SharedEventRef,
    jobs: Vec<cpu::CpuJob>,
}

fn prof_on() -> bool {
    std::env::var("CMF_ZI_METAL_PROF").as_deref() == Ok("1")
}

/// The GEMM kernel variant (`CMF_ZI_MM=<name>`, see `MM_VARIANTS`).
fn mm_var(p: &Pipes) -> &MmVar {
    static V: OnceLock<String> = OnceLock::new();
    let want = V.get_or_init(|| std::env::var("CMF_ZI_MM").unwrap_or_else(|_| "base".into()));
    p.mms.iter().find(|m| m.name == want).unwrap_or(&p.mms[0])
}

struct Enc<'r, 'a> {
    rec: &'r mut Rec<'a>,
    p: &'static Pipes,
    d: &'r ZDev,
    amax: Option<(&'r Buffer, usize)>,
    cpu: Option<CpuPlan>,
}

impl Enc<'_, '_> {
    /// y = GEMM over `n` rows starting at `row0`, z = tensors.len() (≤ 3).
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        blk: &ZBlk,
        ts: &[usize],
        x: &Buffer,
        x_off: &[usize],
        ldx: usize,
        y: &Buffer,
        y_off: &[usize],
        ldy: usize,
        row0: usize,
        n: usize,
        half_out: bool,
        mul: f32,
    ) {
        let t0 = blk.t[ts[0]];
        let mut pm = PMm {
            n: n as u32,
            rows: t0.rows as u32,
            k: t0.cols as u32,
            ldx: ldx as u32,
            ldy: ldy as u32,
            epi: half_out as u32,
            mul,
            ..Default::default()
        };
        for (z, _) in ts.iter().enumerate() {
            pm.x_off[z] = (x_off[z] + row0 * ldx) as u32;
            pm.y_off[z] = (y_off[z] + row0 * ldy) as u32;
        }
        let p = self.p;
        let mv = mm_var(p);
        let arena = self.d.arena.clone();
        // M8: the last output features go to the CPU (large GEMMs only)
        let rows = t0.rows;
        let mut rg = rows;
        if let Some(plan) = self.cpu.as_mut() {
            if n >= 256 && plan.frac > 0.0 {
                let r = ((rows as f64 * (1.0 - plan.frac)) / 64.0).round() as usize * 64;
                rg = r.clamp(64, rows);
            }
            if rg < rows {
                let v = ZEVENT_NEXT.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
                let base = self.d._model.primary_bytes().as_ptr();
                let auxp = blk.aux.contents() as *const f32;
                let mut job = cpu::CpuJob {
                    ready: v,
                    done: v + 1,
                    gdone: ZEVENT2_NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    x: x.contents() as *const u16,
                    x_off: [0; 3],
                    ldx,
                    w: [std::ptr::null(); 3],
                    rs: [std::ptr::null(); 3],
                    nz: ts.len(),
                    k: t0.cols,
                    r0: rg,
                    r1: rows,
                    n,
                    mul,
                    y: y.contents() as *mut u8,
                    y_half: half_out,
                    y_off: [0; 3],
                    ldy,
                };
                for (z, &ti) in ts.iter().enumerate() {
                    job.x_off[z] = x_off[z] + row0 * ldx;
                    job.y_off[z] = y_off[z] + row0 * ldy;
                    // SAFETY: offsets inside the mapped file / the aux buffer
                    job.w[z] = unsafe { base.add(blk.t[ti].abs) } as *const i8;
                    job.rs[z] = unsafe { auxp.add(blk.rs[ti]) };
                }
                let ev = plan.ev;
                plan.jobs.push(job);
                self.rec.event(ev, v, true);
            }
        }
        let enc = self.rec.enc("gemm");
        enc.set_compute_pipeline_state(&mv.pso);
        for s in 0..3 {
            let t = blk.t[ts[s.min(ts.len() - 1)]];
            arena.bind(enc, s as u64, t.abs);
            enc.set_buffer(3 + s as u64, Some(&blk.aux), (blk.rs[ts[s.min(ts.len() - 1)]] * 4) as u64);
        }
        enc.set_buffer(6, Some(x), 0);
        enc.set_buffer(7, Some(y), 0);
        set_p(enc, 8, &pm);
        enc.dispatch_thread_groups(
            MTLSize::new(n.div_ceil(mv.bt) as u64, (rg / mv.bo) as u64, ts.len() as u64),
            MTLSize::new(mv.threads, 1, 1),
        );
        if rg < rows {
            let (ev, ev2, v, g) = {
                let plan = self.cpu.as_ref().unwrap();
                let j = plan.jobs.last().unwrap();
                (plan.ev, plan.ev2, j.done, j.gdone)
            };
            self.rec.event(ev2, g, true);
            self.rec.event(ev, v, false);
        }
    }

    /// Row op over rows [row0, row0 + n).
    #[allow(clippy::too_many_arguments)]
    fn rowop(
        &mut self,
        a: &Acts,
        row0: usize,
        n: usize,
        post: Option<(&ZBlk, usize /*norm idx*/, Option<(&Buffer, usize)> /*gate*/)>,
        pre: Option<(&ZBlk, usize /*norm idx*/, Option<(&Buffer, usize)> /*scale*/, &[usize] /*col tensors*/, f32)>,
    ) {
        let h = self.d.geom.hidden;
        let eps = self.d.geom.eps;
        let mut pr = PRow {
            h: h as u32,
            row0: row0 as u32,
            eps1: eps,
            eps2: eps,
            ..Default::default()
        };
        let p = self.p;
        let enc = self.rec.enc("row");
        enc.set_compute_pipeline_state(&p.rowop);
        enc.set_buffer(0, Some(&a.x), 0);
        enc.set_buffer(1, Some(&a.y), 0);
        for s in 0..3u64 {
            enc.set_buffer(2 + s, Some(&a.xn), s * (a.alloc * h * 2) as u64);
        }
        // defaults for unused slots
        let dummy = &a.y;
        for s in 5..12u64 {
            enc.set_buffer(s, Some(dummy), 0);
        }
        if let Some((blk, ni, gate)) = post {
            pr.mode |= 1;
            enc.set_buffer(5, Some(&blk.aux), (blk.norm[ni] * 4) as u64);
            if let Some((gb, go)) = gate {
                pr.has_g = 1;
                enc.set_buffer(6, Some(gb), (go * 4) as u64);
            }
        }
        if let Some((blk, ni, scale, cols, oscale)) = pre {
            pr.mode |= 2;
            pr.nout = cols.len() as u32;
            pr.oscale = oscale;
            enc.set_buffer(7, Some(&blk.aux), (blk.norm[ni] * 4) as u64);
            if let Some((sb, so)) = scale {
                pr.has_s = 1;
                enc.set_buffer(8, Some(sb), (so * 4) as u64);
            }
            for (j, &t) in cols.iter().enumerate() {
                enc.set_buffer(9 + j as u64, Some(&blk.aux), (blk.col[t] * 4) as u64);
            }
        }
        set_p(enc, 12, &pr);
        enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    fn amax(&mut self, buf: &Buffer, off_halves: usize, count: usize, site: usize) {
        let Some((ab, base)) = self.amax else { return };
        let p = self.p;
        let enc = self.rec.enc("amax");
        enc.set_compute_pipeline_state(&p.amax);
        enc.set_buffer(0, Some(buf), (off_halves * 2) as u64);
        enc.set_buffer(1, Some(ab), 0);
        let n = count as u32;
        let slot = (base + site) as u32;
        set_p(enc, 2, &n);
        set_p(enc, 3, &slot);
        enc.dispatch_thread_groups(MTLSize::new(256, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// One block over rows [row0, row0+n). The block's attention input
    /// (xn slots 0..3 = q/k/v inputs) is already written. `mods` = float
    /// offset of this block's [s_msa|g_msa|s_mlp|g_mlp] in `mbuf`;
    /// `next` = the next block (its q/k/v inputs are produced by this
    /// block's last row op).
    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        a: &Acts,
        blk: &ZBlk,
        row0: usize,
        n: usize,
        rope: (&Buffer, &Buffer),
        segs: &[PFa],
        mods: Option<(&Buffer, usize)>,
        next: Option<(&ZBlk, Option<(&Buffer, usize)>)>,
    ) {
        let g = self.d.geom;
        let gd = self.d.guards;
        let (h, inter) = (g.hidden, g.inter);
        let md = |k: usize| mods.map(|(b, o)| (b, o + k * h));
        let sl = a.alloc * h;
        // qkv
        self.gemm(blk, &[TQ, TK, TV], &a.xn, &[0, sl, 2 * sl], h, &a.panel, &[0, h, 2 * h], 3 * h, row0, n, true, p2(gd.attn - gd.qkv));
        self.amax(&a.panel, row0 * 3 * h, n * 3 * h, 1);
        {
            let p = self.p;
            let pq = PQk {
                h: h as u32,
                nh: g.nh as u32,
                ldp: (3 * h) as u32,
                row0: row0 as u32,
                eps: g.eps * p2(-2 * gd.qkv),
                qmul: std::f32::consts::LOG2_E / (g.hd as f32).sqrt(),
            };
            let enc = self.rec.enc("rope");
            enc.set_compute_pipeline_state(&p.qkrope);
            enc.set_buffer(0, Some(&a.panel), 0);
            enc.set_buffer(1, Some(&blk.aux), (blk.norm[4] * 4) as u64);
            enc.set_buffer(2, Some(&blk.aux), (blk.norm[5] * 4) as u64);
            enc.set_buffer(3, Some(rope.0), 0);
            enc.set_buffer(4, Some(rope.1), 0);
            set_p(enc, 5, &pq);
            enc.dispatch_thread_groups(
                MTLSize::new(n as u64, (2 * g.nh).div_ceil(4) as u64, 1),
                MTLSize::new(128, 1, 1),
            );
        }
        {
            let p = self.p;
            let enc = self.rec.enc("flash");
            let (fpso, nsg) = flash_var(p);
            enc.set_compute_pipeline_state(fpso);
            enc.set_buffer(0, Some(&a.panel), 0);
            enc.set_buffer(1, Some(&a.attn), 0);
            enc.set_buffer(2, Some(&blk.aux), (blk.col[TO] * 4) as u64);
            for s in segs {
                let mut pf = *s;
                pf.ldp = (3 * h) as u32;
                pf.h = h as u32;
                pf.ldo = h as u32;
                pf.oscale = p2(gd.qkv - gd.ao);
                set_p(enc, 3, &pf);
                let nt = (pf.n_img + pf.n_cap) as u64;
                let qt = 8 * nsg as u64;
                enc.dispatch_thread_groups(MTLSize::new(nt.div_ceil(qt), g.nh as u64, 1), MTLSize::new(32 * nsg as u64, 1, 1));
            }
        }
        self.amax(&a.attn, row0 * h, n * h, 2);
        self.gemm(blk, &[TO], &a.attn, &[0], h, &a.y, &[0], h, row0, n, false, p2(gd.ao));
        self.rowop(a, row0, n, Some((blk, 1, md(1))), Some((blk, 2, md(2), &[T1, T3], p2(-gd.ffn))));
        self.amax(&a.xn, row0 * h, n * h, 3);
        let gl = a.alloc * inter;
        self.gemm(blk, &[T1, T3], &a.xn, &[0, sl], h, &a.gu, &[0, gl], inter, row0, n, false, p2(gd.ffn));
        {
            let p = self.p;
            let ps = PSw {
                n4: (n * inter / 4) as u32,
                inter: inter as u32,
                g_off: (row0 * inter) as u32,
                u_off: (gl + row0 * inter) as u32,
                h_off: (row0 * inter) as u32,
                oscale: p2(-gd.hid),
            };
            let enc = self.rec.enc("swiglu");
            enc.set_compute_pipeline_state(&p.swiglu);
            enc.set_buffer(0, Some(&a.gu), 0);
            enc.set_buffer(1, Some(&a.h), 0);
            enc.set_buffer(2, Some(&blk.aux), (blk.col[T2] * 4) as u64);
            set_p(enc, 3, &ps);
            enc.dispatch_thread_groups(MTLSize::new((ps.n4 as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
        }
        self.amax(&a.h, row0 * inter, n * inter, 4);
        self.gemm(blk, &[T2], &a.h, &[0], inter, &a.y, &[0], h, row0, n, false, p2(gd.hid));
        match next {
            Some((nb, nmods)) => {
                self.rowop(a, row0, n, Some((blk, 3, md(3))), Some((nb, 0, nmods, &[TQ, TK, TV], p2(-gd.attn))));
                self.amax(&a.xn, row0 * h, n * h, 0);
            }
            None => self.rowop(a, row0, n, Some((blk, 3, md(3))), None),
        }
    }
}

// ───────────────────────────── contract ─────────────────────────────

fn device() -> Option<(&'static Ctx, &'static Pipes)> {
    if !zi_enabled() {
        return None;
    }
    let c = super::ctx()?;
    let p = pipes(c)?;
    Some((c, p))
}

/// Upload the per-block row scales, column fields and norms of every block
/// (the int8 weights stay in the file mapping).
pub(crate) fn preload(
    model: &Arc<CmfModel>,
    geom: &ZGeom,
    noise_refiner: &[ZBlockRef],
    layers: &[ZBlockRef],
    context_refiner: &[ZBlockRef],
) -> bool {
    let Some((c, _)) = device() else { return false };
    let mut st = ZSTATE.lock().unwrap();
    match ensure_dev(&mut st, c, model, geom, Some((noise_refiner, layers)), Some(context_refiner)) {
        Ok(_) => true,
        Err(e) => decline(&e),
    }
}

pub(crate) fn warmup() -> bool {
    device().is_some()
}

fn rope_buf(c: &Ctx, parts: &[&[f32]]) -> Buffer {
    let mut v = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        v.extend_from_slice(p);
    }
    buf_from(c, &v)
}

/// Build/refresh the per-(prompt, resolution) state for `a.key`.
pub(crate) fn prepare(a: &ZPrepareArgs) -> bool {
    let Some((c, _)) = device() else { return false };
    let g = a.geom;
    if a.n_img_p % 32 != 0 || a.n_cap_p % 32 != 0 || a.n_img == 0 || a.n_img > a.n_img_p {
        return decline("sequence lengths not padded to 32");
    }
    if let Some(n) = &a.neg {
        if n.n_cap_p % 32 != 0 {
            return decline("negative caption length not padded to 32");
        }
    }
    let mut st = ZSTATE.lock().unwrap();
    let d = match ensure_dev(&mut st, c, a.model, &g, Some((a.noise_refiner, a.layers)), None) {
        Ok(d) => d,
        Err(e) => return decline(&e),
    };
    let h = g.hidden;
    if d.emb.is_none() {
        let mut v = a.x_emb_w.to_vec();
        v.extend_from_slice(a.x_emb_b);
        v.extend_from_slice(a.x_pad);
        d.emb = Some(buf_from(c, &v));
        let mut f = a.final_w.to_vec();
        f.extend_from_slice(a.final_b);
        d.fin = Some(buf_from(c, &f));
    }
    // items: positive, then the negative
    let pd = g.patch_dim;
    let hp = g.hd / 2;
    let mut caps: Vec<(usize, &[f32], (&[f32], &[f32]), (&[f32], &[f32]))> =
        vec![(a.n_cap_p, a.cap, a.rope_img, a.rope_joint)];
    if let Some(n) = &a.neg {
        caps.push((n.n_cap_p, n.cap, n.rope_img, n.rope_joint));
    }
    let ni = caps.len();
    for (ncp, cap, ri, rj) in &caps {
        if cap.len() != ncp * h
            || ri.0.len() != a.n_img_p * hp
            || rj.0.len() != (a.n_img_p + ncp) * hp
        {
            return decline("prepare args: caption / rope sizes do not match the lengths");
        }
    }
    let img_rows = ni * a.n_img_p;
    let rows = img_rows + caps.iter().map(|x| x.0).sum::<usize>();
    let mut items = Vec::new();
    let mut r = img_rows;
    let mut capv = Vec::with_capacity((rows - img_rows) * h);
    for (ncp, cap, _, _) in &caps {
        items.push((*ncp, r));
        r += ncp;
        capv.extend_from_slice(cap);
    }
    // rope tables in row order
    let mut ri_c: Vec<&[f32]> = Vec::new();
    let mut ri_s: Vec<&[f32]> = Vec::new();
    for (_, _, ri, _) in &caps {
        ri_c.push(ri.0);
        ri_s.push(ri.1);
    }
    let mut rj_c: Vec<&[f32]> = Vec::new();
    let mut rj_s: Vec<&[f32]> = Vec::new();
    for (_, _, _, rj) in &caps {
        rj_c.push(&rj.0[..a.n_img_p * hp]);
        rj_s.push(&rj.1[..a.n_img_p * hp]);
    }
    for (_, _, _, rj) in &caps {
        rj_c.push(&rj.0[a.n_img_p * hp..]);
        rj_s.push(&rj.1[a.n_img_p * hp..]);
    }
    let nb = d.step_blocks.len();
    let prog = ZProg {
        key: a.key,
        n_img: a.n_img,
        n_img_p: a.n_img_p,
        items,
        acts: Acts::new(c, rows, &g),
        cap: buf_from(c, &capv),
        rope_img: (rope_buf(c, &ri_c), rope_buf(c, &ri_s)),
        rope_joint: (rope_buf(c, &rj_c), rope_buf(c, &rj_s)),
        xtok: buf_zeroed(c, a.n_img_p * pd * 4),
        mods: buf_zeroed(c, nb * 4 * h * 4),
        fs: buf_zeroed(c, h * 4),
        out: buf_zeroed(c, ni * a.n_img * pd * 4),
    };
    d.progs.retain(|p| p.key != a.key);
    if d.progs.len() >= MAX_PROGS {
        d.progs.remove(0);
    }
    d.progs.push(prog);
    true
}

fn amax_on() -> bool {
    std::env::var("CMF_ZI_AMAX").as_deref() == Ok("1")
}

/// One DiT forward for a prepared `a.key`; writes `a.out` (and `out_neg`).
pub(crate) fn step(a: &mut ZStepArgs) -> bool {
    let Some((c, p)) = device() else { return false };
    let mut st = ZSTATE.lock().unwrap();
    let Some(d) = st.0.as_mut() else { return false };
    let Some(pi) = d.progs.iter().position(|p| p.key == a.key) else { return false };
    let d: &ZDev = d;
    let pr = &d.progs[pi];
    let g = d.geom;
    let (h, pd) = (g.hidden, g.patch_dim);
    let ni = pr.items.len();
    let nb = d.step_blocks.len();
    if a.x_tok.len() != pr.n_img_p * pd
        || a.mods.len() != nb * 4 * h
        || a.final_scale.len() != h
        || a.out.len() != pr.n_img * pd
        || (ni == 2) != a.out_neg.is_some()
        || a.out_neg.as_ref().is_some_and(|o| o.len() != pr.n_img * pd)
    {
        return false;
    }
    write_f32(&pr.xtok, 0, a.x_tok);
    write_f32(&pr.mods, 0, a.mods);
    write_f32(&pr.fs, 0, a.final_scale);
    let amax_buf = amax_on().then(|| buf_zeroed(c, nb * 5 * 4));
    let t0 = std::time::Instant::now();
    let mut rec = Rec::new(c);
    let chunk = env_usize("CMF_ZI_METAL_CHUNK", 2).max(1);
    let ac = &pr.acts;
    let emb = d.emb.as_ref().unwrap();
    let fin = d.fin.as_ref().unwrap();
    let frac = cpu::frac(&c._device.name(), pr.acts.rows);
    let mut jobs: Vec<cpu::CpuJob> = Vec::new();
    {
        let mut e = Enc {
            rec: &mut rec,
            p,
            d,
            amax: None,
            cpu: (frac > 0.0).then(|| CpuPlan { frac, ev: zevent(c), ev2: zevent2(c), jobs: Vec::new() }),
        };
        // embed (every item's image rows)
        {
            let pe = PEm {
                h: h as u32,
                n_img: pr.n_img as u32,
                n_img_p: pr.n_img_p as u32,
                nitems: ni as u32,
            };
            let enc = e.rec.enc("embed");
            enc.set_compute_pipeline_state(&p.embed);
            enc.set_buffer(0, Some(&pr.xtok), 0);
            enc.set_buffer(1, Some(emb), 0);
            enc.set_buffer(2, Some(emb), (h * 64 * 4) as u64);
            enc.set_buffer(3, Some(emb), (h * 65 * 4) as u64);
            enc.set_buffer(4, Some(&ac.x), 0);
            set_p(enc, 5, &pe);
            enc.dispatch_thread_groups(
                MTLSize::new(h.div_ceil(64) as u64, pr.n_img_p as u64, 1),
                MTLSize::new(64, 1, 1),
            );
        }
        let img_rows = ni * pr.n_img_p;
        let mo = |bi: usize| Some((&pr.mods, bi * 4 * h));
        let nr = d.n_refiner;
        // noise refiner over the image rows of every item
        let segs_img: Vec<PFa> = (0..ni)
            .map(|i| PFa {
                img_off: (i * pr.n_img_p) as u32,
                n_img: pr.n_img_p as u32,
                ..Default::default()
            })
            .collect();
        let rope_i = (&pr.rope_img.0, &pr.rope_img.1);
        let rope_j = (&pr.rope_joint.0, &pr.rope_joint.1);
        for bi in 0..nb {
            let blk = &d.step_blocks[bi];
            let amax_slot = amax_buf.as_ref().map(|b| (b, bi * 5));
            e.amax = amax_slot;
            if bi == 0 || bi == nr {
                if bi == nr {
                    // caption rows in
                    let n4 = ((ac.rows - img_rows) * h / 4) as u32;
                    let enc = e.rec.enc("copy");
                    enc.set_compute_pipeline_state(&p.copy4);
                    enc.set_buffer(0, Some(&pr.cap), 0);
                    enc.set_buffer(1, Some(&ac.x), (img_rows * h * 4) as u64);
                    set_p(enc, 2, &n4);
                    enc.dispatch_thread_groups(MTLSize::new((n4 as u64).div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
                }
                let n = if bi < nr { img_rows } else { ac.rows };
                e.rowop(ac, 0, n, None, Some((blk, 0, mo(bi).map(|(b, o)| (b, o)), &[TQ, TK, TV], p2(-d.guards.attn))));
                e.amax(&ac.xn, 0, n * h, 0);
            }
            let refiner = bi < nr;
            let (n, rope) = if refiner { (img_rows, rope_i) } else { (ac.rows, rope_j) };
            let segs: Vec<PFa> = if refiner {
                segs_img.clone()
            } else {
                pr.items
                    .iter()
                    .enumerate()
                    .map(|(i, &(ncp, cr))| PFa {
                        img_off: (i * pr.n_img_p) as u32,
                        n_img: pr.n_img_p as u32,
                        cap_off: cr as u32,
                        n_cap: ncp as u32,
                        ..Default::default()
                    })
                    .collect()
            };
            // the refiner's last block hands over to the caption copy
            let next = if bi + 1 < nb && bi + 1 != nr {
                Some((&d.step_blocks[bi + 1], mo(bi + 1)))
            } else {
                None
            };
            e.block(ac, blk, 0, n, rope, &segs, mo(bi), next);
            if (bi + 1) % chunk == 0 {
                e.rec.cut();
            }
        }
        // final layer per item
        for i in 0..ni {
            let pf = PFi {
                h: h as u32,
                row0: (i * pr.n_img_p) as u32,
                out0: (i * pr.n_img) as u32,
                eps: g.final_eps,
            };
            let enc = e.rec.enc("final");
            enc.set_compute_pipeline_state(&p.fin);
            enc.set_buffer(0, Some(&ac.x), 0);
            enc.set_buffer(1, Some(&pr.fs), 0);
            enc.set_buffer(2, Some(fin), 0);
            enc.set_buffer(3, Some(fin), (64 * h * 4) as u64);
            enc.set_buffer(4, Some(&pr.out), 0);
            set_p(enc, 5, &pf);
            enc.dispatch_thread_groups(MTLSize::new(pr.n_img as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        if let Some(plan) = e.cpu.take() {
            jobs = plan.jobs;
        }
    }
    let mut cpu_ok = true;
    if !jobs.is_empty() {
        rec.close();
        let cmds: Vec<CommandBuffer> = rec.done.iter().map(|(_, c)| c.clone()).collect();
        cpu_ok = cpu::execute(&jobs, zevent(c), zevent2(c), ac.rows, &cmds);
    }
    let ok = rec.finish() && cpu_ok;
    if std::env::var("CMF_ZIMAGE_PROF").is_ok_and(|v| v != "0") && !rec.prof {
        eprintln!(
            "zimage metal step: {:.3}s wall ({} rows, {} item(s))",
            t0.elapsed().as_secs_f64(),
            ac.rows,
            ni
        );
    }
    let _ = rec.gpu_ms();
    if !ok {
        return decline("a Metal command buffer failed");
    }
    if let Some(ab) = &amax_buf {
        let v = unsafe { std::slice::from_raw_parts(ab.contents() as *const u32, nb * 5) };
        let names = ["xn_qkv", "qkv", "attn_o", "xn_ffn", "hidden"];
        let mut worst = [0f32; 5];
        for bi in 0..nb {
            let row: Vec<String> = (0..5)
                .map(|s| {
                    let f = f32::from_bits(v[bi * 5 + s]);
                    worst[s] = worst[s].max(f);
                    format!("{}={:.3e}", names[s], f)
                })
                .collect();
            eprintln!("zi amax block {bi:2}: {}", row.join(" "));
        }
        eprintln!("zi amax worst (stored, after guards): {:?}", worst);
    }
    let out = unsafe { std::slice::from_raw_parts(pr.out.contents() as *const f32, ni * pr.n_img * pd) };
    if out.iter().any(|v| !v.is_finite()) {
        return decline("the device output is not finite (an f16 overflow: raise the CMF_ZI_*_SHIFT guards)");
    }
    a.out.copy_from_slice(&out[..pr.n_img * pd]);
    if let Some(on) = a.out_neg.as_mut() {
        on.copy_from_slice(&out[pr.n_img * pd..]);
    }
    true
}

/// Drop all module-local device state (prepared states, VAE chain).
pub(crate) fn release() {
    if let Ok(mut st) = ZSTATE.lock() {
        st.0 = None;
    }
    vae::release();
}

/// Drop the DiT state (keep the VAE chain).
pub(crate) fn release_dit() {
    if let Ok(mut st) = ZSTATE.lock() {
        st.0 = None;
    }
}

/// Unmodulated context refiner on the device (s = 0, gate = 1).
pub(crate) fn refine_caption(
    model: &Arc<CmfModel>,
    geom: &ZGeom,
    blocks: &[ZBlockRef],
    rope_cap: (&[f32], &[f32]),
    cap: &mut [f32],
) -> bool {
    if std::env::var("CMF_ZI_METAL_REFINE").as_deref() == Ok("0") || blocks.is_empty() {
        return false;
    }
    let Some((c, p)) = device() else { return false };
    let h = geom.hidden;
    let n = cap.len() / h.max(1);
    if n == 0 || n % 32 != 0 || cap.len() != n * h || rope_cap.0.len() != n * geom.hd / 2 {
        return false;
    }
    let mut st = ZSTATE.lock().unwrap();
    let d = match ensure_dev(&mut st, c, model, geom, None, Some(blocks)) {
        Ok(d) => d,
        Err(e) => return decline(&e),
    };
    let d: &ZDev = d;
    let acts = Acts::new(c, n, geom);
    write_f32(&acts.x, 0, cap);
    let rc = buf_from(c, rope_cap.0);
    let rs = buf_from(c, rope_cap.1);
    let segs = [PFa {
        img_off: 0,
        n_img: n as u32,
        ..Default::default()
    }];
    let mut rec = Rec::new(c);
    {
        let mut e = Enc { rec: &mut rec, p, d, amax: None, cpu: None };
        let nbk = d.ctx_blocks.len();
        e.rowop(&acts, 0, n, None, Some((&d.ctx_blocks[0], 0, None, &[TQ, TK, TV], p2(-d.guards.attn))));
        for bi in 0..nbk {
            let next = (bi + 1 < nbk).then(|| (&d.ctx_blocks[bi + 1], None));
            e.block(&acts, &d.ctx_blocks[bi], 0, n, (&rc, &rs), &segs, None, next);
        }
    }
    if !rec.finish() {
        return decline("a Metal command buffer failed (context refiner)");
    }
    let x = unsafe { std::slice::from_raw_parts(acts.x.contents() as *const f32, n * h) };
    if x.iter().any(|v| !v.is_finite()) {
        return decline("the refined caption is not finite");
    }
    cap.copy_from_slice(x);
    true
}

/// Upload the VAE weights and compile its kernels ahead of the decode.
pub(crate) fn vae_prewarm(a: &crate::vae::VaeChainArgs) -> bool {
    zi_enabled() && vae::prewarm(a)
}

/// Resident Flux-VAE decoder; `z` is already de-normalised.
pub(crate) fn vae_decode_chain(
    a: &crate::vae::VaeChainArgs,
    z: &[f32],
    h: usize,
    w: usize,
    out: &mut [f32],
) -> bool {
    zi_enabled() && vae::decode(a, z, h, w, out)
}

// ───────────────────────────── bench hooks ─────────────────────────────

/// Pure simdgroup-MMA issue rate (no memory traffic), `ty` "h" (half
/// operands) or "f" (float operands), f32 accumulation. Returns TFLOP/s
/// (best of `reps`).
#[doc(hidden)]
pub fn bench_mma_peak(ty: &str, reps: usize) -> Option<f64> {
    let c = super::ctx()?;
    let opts = metal::CompileOptions::new();
    opts.set_language_version(metal::MTLLanguageVersion::V3_0);
    let lib = c._device.new_library_with_source(ZMSL, &opts).ok()?;
    let f = lib.get_function(match ty { "f" => "zi_peak_f", "hh" => "zi_peak_hh", _ => "zi_peak_h" }, None).ok()?;
    let pso = c._device.new_compute_pipeline_state_with_function(&f).ok()?;
    let out = buf_zeroed(c, 1 << 20);
    let iters = 4096u32;
    let tgs = 2048u64;
    let mut best = 0f64;
    for _ in 0..reps {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&out), 0);
        set_p(enc, 1, &iters);
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ms = super::cmd_gpu_ms(cmd);
        let fl = tgs as f64 * 4.0 * iters as f64 * 16.0 * 1024.0;
        best = best.max(fl / ms / 1e9);
    }
    Some(best)
}

/// GEMM microbench: y[n, rows] = x[n, k] · W[rows, k]ᵀ with synthetic int8
/// weights, `reps` timed dispatches (GPU time per dispatch, min and
/// median), variant "base" or "wt". Returns (min ms, median ms, rel err vs
/// an f64 host reference on a sample of outputs).
#[doc(hidden)]
pub fn bench_gemm(rows: usize, k: usize, n: usize, reps: usize, variant: &str) -> Option<(f64, f64, f64)> {
    let c = super::ctx()?;
    let p = pipes(c)?;
    let mv = p.mms.iter().find(|m| m.name == variant)?;
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let wq: Vec<i8> = (0..rows * k).map(|_| (rnd() % 255) as i8).collect();
    let rs: Vec<f32> = (0..rows).map(|i| 0.001 + (i % 7) as f32 * 1e-4).collect();
    let xh: Vec<u16> = (0..(n.div_ceil(128) * 128 + 128) * k)
        .map(|_| cortiq_core::quant::f32_to_f16(((rnd() % 2001) as f32 - 1000.0) / 1000.0))
        .collect();
    let wb = c._device.new_buffer_with_data(wq.as_ptr() as *const c_void, wq.len() as u64, MTLResourceOptions::StorageModeShared);
    let rb = buf_from(c, &rs);
    let xb = c._device.new_buffer_with_data(xh.as_ptr() as *const c_void, (xh.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
    let yb = buf_zeroed(c, n * rows * 4);
    let pm = PMm {
        n: n as u32,
        rows: rows as u32,
        k: k as u32,
        ldx: k as u32,
        ldy: rows as u32,
        epi: 0,
        mul: 1.0,
        ..Default::default()
    };
    let mut times = Vec::new();
    for _ in 0..reps + 1 {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&mv.pso);
        for s in 0..3u64 {
            enc.set_buffer(s, Some(&wb), 0);
            enc.set_buffer(3 + s, Some(&rb), 0);
        }
        enc.set_buffer(6, Some(&xb), 0);
        enc.set_buffer(7, Some(&yb), 0);
        set_p(enc, 8, &pm);
        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(mv.bt) as u64, (rows / mv.bo) as u64, 1), MTLSize::new(mv.threads, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        times.push(super::cmd_gpu_ms(cmd));
    }
    times.remove(0);
    let y = unsafe { std::slice::from_raw_parts(yb.contents() as *const f32, n * rows) };
    let (mut dd, mut rr) = (0f64, 0f64);
    for s in 0..64usize {
        let t = (s * 7919) % n;
        let o = (s * 104729) % rows;
        let mut acc = 0f64;
        for kk in 0..k {
            acc += wq[o * k + kk] as f64 * cortiq_core::quant::f16_to_f32(xh[t * k + kk]) as f64;
        }
        acc *= rs[o] as f64;
        dd += (y[t * rows + o] as f64 - acc).powi(2);
        rr += acc * acc;
    }
    let mut s = times.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((s[0], s[s.len() / 2], (dd / rr.max(1e-300)).sqrt()))
}

/// Codec A/B arm: the f16-plane GEMM (half weights, same tile) — (min ms,
/// median ms) of GPU time.
#[doc(hidden)]
pub fn bench_plane(rows: usize, k: usize, n: usize, reps: usize) -> Option<(f64, f64)> {
    vae::bench_plane(rows, k, n, reps)
}

/// Flash-attention microbench over one item of `n` rows (30 heads × 128):
/// synthetic normalised q/k and v, GPU ms (min, median), rel error of a
/// sample of (row, head) outputs against an f64 host softmax.
#[doc(hidden)]
pub fn bench_flash(n: usize, reps: usize, variant: &str) -> Option<(f64, f64, f64)> {
    let c = super::ctx()?;
    let p = pipes(c)?;
    let fv = p.flashes.iter().find(|f| f.0 == variant)?;
    let (pso, nsg) = (&fv.1, fv.2);
    let (h, nh) = (3840usize, 30usize);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    let qmul = std::f32::consts::LOG2_E / (128f32).sqrt();
    let panel_f: Vec<f32> = (0..n * 3 * h)
        .map(|i| {
            let col = i % (3 * h);
            let v = rnd() * 2.0;
            if col < h { v * qmul } else { v }
        })
        .collect();
    let panel: Vec<u16> = panel_f.iter().map(|&v| cortiq_core::quant::f32_to_f16(v)).collect();
    let pb = c._device.new_buffer_with_data(panel.as_ptr() as *const c_void, (panel.len() * 2) as u64, MTLResourceOptions::StorageModeShared);
    let ob = buf_zeroed(c, n * h * 2);
    let colo = buf_from(c, &vec![1.0f32; h]);
    let pf = PFa {
        img_off: 0,
        n_img: n as u32,
        cap_off: 0,
        n_cap: 0,
        ldp: (3 * h) as u32,
        h: h as u32,
        ldo: h as u32,
        oscale: 1.0,
    };
    let mut times = Vec::new();
    for _ in 0..reps + 1 {
        let cmd = c.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&pb), 0);
        enc.set_buffer(1, Some(&ob), 0);
        enc.set_buffer(2, Some(&colo), 0);
        set_p(enc, 3, &pf);
        enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(8 * nsg) as u64, nh as u64, 1), MTLSize::new(32 * nsg as u64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        times.push(super::cmd_gpu_ms(cmd));
    }
    times.remove(0);
    let out = unsafe { std::slice::from_raw_parts(ob.contents() as *const u16, n * h) };
    let hf = |i: usize| cortiq_core::quant::f16_to_f32(panel[i]) as f64;
    let (mut dd, mut rr) = (0f64, 0f64);
    for sidx in 0..8usize {
        let row = (sidx * 7919) % n;
        let head = (sidx * 13) % nh;
        let sc: Vec<f64> = (0..n)
            .map(|j| {
                (0..128).map(|d| hf(row * 3 * h + head * 128 + d) * hf(j * 3 * h + h + head * 128 + d)).sum::<f64>()
            })
            .collect();
        let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
        let w: Vec<f64> = sc.iter().map(|s| (2f64).powf(s - mx)).collect();
        let l: f64 = w.iter().sum();
        for d in 0..128 {
            let r: f64 = (0..n).map(|j| w[j] * hf(j * 3 * h + 2 * h + head * 128 + d)).sum::<f64>() / l;
            let g = cortiq_core::quant::f16_to_f32(out[row * h + head * 128 + d]) as f64;
            dd += (g - r).powi(2);
            rr += r * r;
        }
    }
    let mut s = times.clone();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((s[0], s[s.len() / 2], (dd / rr.max(1e-300)).sqrt()))
}
