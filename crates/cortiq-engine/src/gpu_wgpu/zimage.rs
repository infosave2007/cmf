//! Z-Image-Turbo / Z-Image on wgpu/Vulkan (plan WP2). OWNER: WP2 — this file only.
//!
//! Implements the `gpu::zimage_*` / `gpu::vae_decode_chain` contract
//! (`gpu.rs`, "Z-Image-Turbo device contract"). Rules (plan §2.2):
//! - reach the parent's private kernels, shader sources (`super::SOME_SRC`),
//!   `ctx()` and buffer pools through `super::`; never edit parent functions;
//! - build pipelines into a module-local, lazily created cache (`OnceLock`
//!   keyed by device); never add fields to the parent's context struct;
//! - `release` touches module-local state only and must never bring a
//!   device up (it is called unconditionally by `gpu::zimage_release`);
//! - every entry returns `false` before changing any output when it cannot
//!   honour the full contract — the caller then runs the CPU path.
//!
//! The module is `pub` (doc-hidden) so WP2's examples/tests can reach
//! `pub` helpers added here without touching `gpu_wgpu.rs`.
//!
//! # B1: the fast-path kernels and the resident chain
//!
//! Measured on the pod's RTX 3090 (Ampere GA102, NVIDIA 610.43, Vulkan
//! 1.4), `examples/zimage_gemmbench.rs`, in-process device time, medians:
//!
//! - Ceilings (S0): coop is available (16×16×16 f16 in, f32 or f16 acc).
//!   A pure-MMA loop reaches 81.5 TFLOPS with f32 accumulation and 162.9
//!   with f16 accumulation. The parent's existing arms at the Z-Image
//!   shapes reach: scalar `q4tp_mm` 7.4–8.3 TF, `q4tp_mm_coop` (in-kernel
//!   dequant) 15.5–19.5 TF, `q4tp_mm_coop_f16` (plane) 16–19 TF.
//! - [`MmCfg`] / `zi_mm`: f16 plane × f16 activation, f32 accumulation.
//!   The default tile is 128×128×32 with 2×2 subgroups (a 64×64 warp tile)
//!   and a register prefetch of the next K slice. It reaches 55–61 TF on
//!   all four sites at 1024² (qkv, o, w13 with SwiGLU, w2), which is
//!   70–75 % of the f32-accumulate ceiling and 3.1× the parent's coop
//!   arms. At 512² it reaches 50–55 TF.
//!   Precision against an f64 reference on f16 operands: the f32 epilogue
//!   gives 6.2e-6 at K=3840 and 1.6e-5 at K=10240; the f16 and SwiGLU
//!   epilogues give 2.1e-4, which is the f16 output rounding.
//!   Measured and rejected (kept only as `MmCfg` fields):
//!   - two shared stages: −5…−12 %;
//!   - direct global coop loads (`direct`): 25 TF;
//!   - 2×4 / 4×2 / 256-wide tiles: 42–52 TF;
//!   - BK=64: equal or −3 %.
//!   `acc16_probe` (f16 accumulators with no flush, so a wrong answer)
//!   runs the BK=64 tile at 80 TF. That is the ceiling a flushed
//!   f16-accumulate arm could approach. It is not built, because its
//!   precision at K=10240 needs the real-weight gates first.
//! - `zi_flash`: bidirectional flash attention, heads 30×128, read straight
//!   out of the fused qkv panel (no head-major pack). Q·K and P·V are f16
//!   on the matrix units. The softmax is f32 online with the lazy
//!   (threshold 2⁸) rescale, which is exact. Output goes straight into the
//!   `[token][head·128]` layout the O GEMM reads. There is one dispatch per
//!   segment, so batch 2 (CFG cond + uncond) keeps each item's own padded
//!   length, which is exactly diffusers' key mask.
//!   The default is 4 subgroups × 16 queries and 16 keys per block, with V
//!   staged row-major. It measures 51–54 TF at 1024² (5.4 ms per layer)
//!   and 38 TF at 512².
//!   Error against f64 is 2.1e-4, the f16 output floor, including the
//!   rescale path.
//!   Rescaling on every max increase (`CMF_ZI_FLASH_THR=0`) runs at
//!   22 TF. A transposed V (`CMF_ZI_FLASH_VT=1`) runs at 35 TF.
//! - `zi_rowop` fuses the gated residual with the post-norm
//!   (x += tanh(g)·RMS(br)·w) and the next pre-norm·(1+s) → f16 into one
//!   pass over the row.
//! - `zi_qkrope` applies the per-head qk-RMSNorm and the complex-interleaved
//!   RoPE, in place on the f16 qkv panel.
//! - `zi_embed` is the x_embedder (64→3840 + b, pad rows := x_pad).
//!   `zi_final` is the final LayerNorm·scale + Linear(3840→64) + b.
//!   Both run in f32.
//!   Row kernels together take 2.7 % of a 1024² step, so fusing them into
//!   the GEMM epilogues is not worth doing.
//! - [`ZStepDev`] holds the planes, the per-(prompt, resolution) state and
//!   the prebuilt bind groups. The hidden state never leaves the device
//!   inside a step. The modulation of all 32 blocks is one buffer, written
//!   once per step. A step needs one upload (x_tok, mods, scale) and one
//!   1 MB readback.
//!
//! Synthetic whole step (32 blocks, distinct random f16 planes, 11.6 GB):
//!
//! | case          | zi chain            | Lumina path (`dit_block_seg` as is) |
//! |---------------|---------------------|-------------------------------------|
//! | 512², b1      | 0.250 s (50 TF eff) | 1.06 s (0.815 s with resident x)    |
//! | 512², b2      | 0.479 s (52 TF)     | 2.66 s                              |
//! | 1024², b1     | 1.034 s (55 TF)     | 4.95 s                              |
//! | 1024², b2     | 2.063 s (55 TF)     | —                                   |
//!
//! At 1024² b1, per class: GEMMs 821 ms (w13 368, qkv 206, w2 180, o 67),
//! flash 164 ms, row kernels 29 ms, embed/final/copies 16 ms.
//!
//! Correctness of the assembled pieces:
//! - one block against an f64 host block: 1.5e-4 of x (5e-4 of the
//!   block's update);
//! - the whole contract path (`gpu::zimage_prepare` + `zimage_step`) on a
//!   tiny F16/BF16 container (x_pad rows, 2 refiner blocks, [img, cap]
//!   assembly, 2 layers, final layer) against an f64 host forward:
//!   4e-4 (`zimage_gemmbench stepcheck`).
//!
//! # Integration recipe (for the core package)
//!
//! - `prepare` and `step` implement the contract for batch 1 (Turbo).
//!   `prepare` builds the f16 planes once per model, using
//!   [`ZBlockDev::from_model`]: F16 as stored, Bf16/F32 converted, Q4TiledP
//!   dequantized by the parent's `q4tp_dq_f16`. Any other codec declines.
//!   The contract then builds a [`ZStepDev`] for (n_img, n_cap_p) and
//!   uploads both RoPE tables. `step` uploads x_tok, mods and the final
//!   scale, replays the program and reads back `[n_img][64]`.
//! - The base model with CFG (batch 2) goes through
//!   `ZStepDev::new(.., n_cap_p = &[cond, uncond], cap = both stacked)`
//!   plus `upload(x_tok for both items, ..)` and `run(out [2][n_img][64])`.
//!   The contract has no batch-2 entry yet. Adding one means an
//!   `Option<…>` field agreed with the WP1 lead (plan §2.1). Unequal
//!   caption lengths are supported: segments, per-item final dispatch.
//! - Still to do on the device side:
//!   - the context refiner (`refine_caption` still declines; use
//!     `block_calls(.., modulated = false)` on a `ZSeq` of the caption);
//!   - the resident VAE (`vae_decode_chain`);
//!   - q8 plane codecs;
//!   - the text encoder.
//! - Knobs:
//!   - `CMF_ZI_WGPU=0`: device path off;
//!   - `CMF_ZI_TILE=bm,bn,bk,wm,wn`: GEMM tile;
//!   - `CMF_ZI_FLASH=nw,bc`: flash tile;
//!   - `CMF_ZI_FLASH_THR`, `CMF_ZI_FLASH_VT`, `CMF_ZI_FLASH_PAD`,
//!     `CMF_ZI_FLASH_DBG`: flash debug arms;
//!   - `CMF_ZI_CHECKED=1`: build with naga bounds checks.
//!
//! Traps found here (they cost time; keep them):
//! - A workgroup array whose byte size is not a multiple of 16 misaligns
//!   every array after it. Cooperative loads ignore the low address bits,
//!   so each 16-half row is read 4 halves early.
//! - naga 30 does not emit a runtime `coopStore` stride or pointer index
//!   before the store and panics with "Expression is not cached". Bind
//!   both to `let`s first.
//! - A `var` of cooperative-matrix type declared inside a loop is zeroed
//!   once, at function entry, not once per iteration.

use crate::gpu::{ZBlockRef, ZGeom, ZPrepareArgs, ZStepArgs};
use cortiq_core::CmfModel;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::util::DeviceExt;

use super::Ctx;

/// Module-local device state: the f16 planes of the 32 per-step blocks
/// (keyed by the model, they survive prompts) and the prepared step
/// program of the current (prompt, resolution) key.
struct ZState {
    model_uid: u64,
    blocks: Vec<ZBlockDev>,
    key: Option<u64>,
    prog: Option<ZStepDev>,
}

static ZSTATE: Mutex<Option<ZState>> = Mutex::new(None);

/// `CMF_ZI_WGPU=0` turns the whole device path off (the caller runs CPU).
fn zi_enabled() -> bool {
    std::env::var("CMF_ZI_WGPU").as_deref() != Ok("0")
}

/// Build/refresh the per-(prompt, resolution) state for `a.key` (batch 1,
/// the contract's shape). Planes are built once per model: every codec
/// `ZBlockDev::from_model` knows is expanded to f16 (11.6 GB for Turbo).
pub(crate) fn prepare(a: &ZPrepareArgs) -> bool {
    if !zi_enabled() {
        return false;
    }
    let Some(d) = ZDims::from_geom(&a.geom) else { return false };
    let (n_img, n_img_p, n_cap_p) = (a.n_img, a.n_img_p, a.n_cap_p);
    let s_len = n_img_p + n_cap_p;
    if n_img_p % 32 != 0
        || n_cap_p % 32 != 0
        || n_img > n_img_p
        || n_img == 0
        || d.pd != 64
        || a.noise_refiner.len() != 2
        || a.layers.is_empty()
        || a.cap.len() < n_cap_p * d.h
        || a.rope_img.0.len() < n_img_p * 64
        || a.rope_img.1.len() < n_img_p * 64
        || a.rope_joint.0.len() < s_len * 64
        || a.rope_joint.1.len() < s_len * 64
        || a.x_emb_w.len() < d.h * 64
        || a.x_emb_b.len() < d.h
        || a.x_pad.len() < d.h
        || a.final_w.len() < 64 * d.h
        || a.final_b.len() < 64
    {
        return false;
    }
    if zctx().is_none() {
        return false;
    }
    let Ok(mut g) = ZSTATE.lock() else { return false };
    let uid = a.model.uid();
    let nblk = a.noise_refiner.len() + a.layers.len();
    let have = g.as_ref().is_some_and(|st| st.model_uid == uid && st.blocks.len() == nblk);
    if !have {
        *g = None; // free the old planes before building new ones
        let mut blocks = Vec::with_capacity(nblk);
        for r in a.noise_refiner.iter().chain(a.layers.iter()) {
            match ZBlockDev::from_model(a.model, &d, r) {
                Some(b) => blocks.push(b),
                None => return false,
            }
        }
        *g = Some(ZState { model_uid: uid, blocks, key: None, prog: None });
    }
    let st = g.as_mut().unwrap();
    st.prog = None;
    st.key = None;
    let io = ZIo { x_emb_w: a.x_emb_w, x_emb_b: a.x_emb_b, x_pad: a.x_pad, final_w: a.final_w, final_b: a.final_b };
    let t = ZTiles::default();
    let (nr, layers) = st.blocks.split_at(2);
    let Some(prog) = ZStepDev::new(d, &t, nr, layers, &io, n_img, &[n_cap_p], &a.cap[..n_cap_p * d.h]) else {
        return false;
    };
    prog.img.set_rope(&a.rope_img.0[..n_img_p * 64], &a.rope_img.1[..n_img_p * 64]);
    prog.joint.set_rope(&a.rope_joint.0[..s_len * 64], &a.rope_joint.1[..s_len * 64]);
    st.prog = Some(prog);
    st.key = Some(a.key);
    true
}

/// One DiT forward for a prepared `a.key`; writes `a.out`.
pub(crate) fn step(a: &mut ZStepArgs) -> bool {
    let Ok(g) = ZSTATE.lock() else { return false };
    let Some(st) = g.as_ref() else { return false };
    let (Some(key), Some(prog)) = (st.key, st.prog.as_ref()) else { return false };
    let d = prog.d;
    let nblk = st.blocks.len();
    if key != a.key
        || a.x_tok.len() < prog.n_img_p * d.pd
        || a.mods.len() < nblk * 4 * d.h
        || a.final_scale.len() < d.h
        || a.out.len() < prog.n_img * d.pd
    {
        return false;
    }
    prog.upload(&a.x_tok[..prog.n_img_p * d.pd], &a.mods[..nblk * 4 * d.h], &a.final_scale[..d.h]);
    prog.run(&mut a.out[..prog.n_img * d.pd]).is_some()
}

/// Drop all module-local device state (planes, prepared states, VAE chain).
/// Never brings a device up: it only drops what this module holds.
pub(crate) fn release() {
    if let Ok(mut g) = ZSTATE.lock() {
        *g = None;
    }
    if let Ok(mut g) = ZREFINER.lock() {
        *g = None;
    }
}

/// Unmodulated context refiner on the device (s = 0, gate = 1).
/// The planes of the context-refiner blocks are cached apart from the 32
/// per-step blocks: (model uid, first weight index) → planes.
static ZREFINER: Mutex<Option<(u64, usize, Vec<ZBlockDev>)>> = Mutex::new(None);

pub(crate) fn refine_caption(
    model: &Arc<CmfModel>,
    geom: &ZGeom,
    blocks: &[ZBlockRef],
    rope_cap: (&[f32], &[f32]),
    cap: &mut [f32],
) -> bool {
    if !zi_enabled() || blocks.is_empty() {
        return false;
    }
    let Some(d) = ZDims::from_geom(geom) else { return false };
    let n = cap.len() / d.h;
    if n == 0 || n % 32 != 0 || cap.len() != n * d.h || rope_cap.0.len() < n * 64 || rope_cap.1.len() < n * 64 {
        return false;
    }
    let Some(c) = zctx() else { return false };
    let Ok(mut g) = ZREFINER.lock() else { return false };
    let key = (model.uid(), blocks[0].wq);
    if !g.as_ref().is_some_and(|(u, w, v)| (*u, *w) == key && v.len() == blocks.len()) {
        *g = None;
        let mut v = Vec::with_capacity(blocks.len());
        for r in blocks {
            match ZBlockDev::from_model(model, &d, r) {
                Some(b) => v.push(b),
                None => return false,
            }
        }
        *g = Some((key.0, key.1, v));
    }
    let devs = &g.as_ref().unwrap().2;
    let Some(seq) = ZSeq::new(&d, &[(0, n)]) else { return false };
    seq.set_rope(&rope_cap.0[..n * 64], &rope_cap.1[..n * 64]);
    seq.write_x(cap);
    // Unmodulated: the row ops read no mods (scale 0, gate 1).
    let mods = sbuf(c, 16, "zi_nomods");
    let t = ZTiles::default();
    let mut calls = ZCalls::default();
    for (k, blk) in devs.iter().enumerate() {
        let next = devs.get(k + 1).map(|nb| (nb, 0));
        match block_calls(&d, &t, &seq, blk, &mods, 0, k == 0, next, false) {
            Some(cl) => calls.extend(cl),
            None => return false,
        }
    }
    if calls.run().is_none() {
        return false;
    }
    match seq.read_x(&d) {
        Some(x) => {
            cap.copy_from_slice(&x[..n * d.h]);
            true
        }
        None => false,
    }
}

/// Resident Flux-VAE decoder; `z` is already de-normalised.
pub(crate) fn vae_decode_chain(
    _a: &crate::vae::VaeChainArgs,
    _z: &[f32],
    _h: usize,
    _w: usize,
    _out: &mut [f32],
) -> bool {
    false
}

// ════════════════════════════════════════════════════════════════════
// Pipeline cache (module-local; never a field of the parent's Ctx).
// ════════════════════════════════════════════════════════════════════

struct ZPipes {
    pipes: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
}

static ZP: OnceLock<ZPipes> = OnceLock::new();

/// The device context, but only where the fast path can run at all:
/// cooperative matrices with the 16×16×16 f16→f32 shape (the parent's
/// init checked the shape before raising `COOP_OK`), f16 in shaders, and
/// 32-wide subgroups (the epilogues index lanes as `tid − 32·sg`).
fn zctx() -> Option<&'static Ctx> {
    let c = super::ctx()?;
    if !super::coop_matrix_active() {
        return None;
    }
    let f = c.device.features();
    if !f.contains(wgpu::Features::SHADER_F16)
        || !f.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return None;
    }
    let i = c._adapter.get_info();
    if i.subgroup_min_size != 32 || i.subgroup_max_size != 32 {
        // Only NVIDIA-style 32-wide subgroups are written for; AMD wave64
        // or Intel would need a lane map. Decline, CPU/Lumina path runs.
        if std::env::var("CMF_ZI_ANY_SUBGROUP").as_deref() != Ok("1") {
            return None;
        }
    }
    Some(c)
}

/// Compile (once) and return a pipeline. Performance kernels are built
/// WITHOUT naga's injected bounds checks and loop bounding: every buffer the
/// chain binds is padded so the kernels stay in range by construction (M to
/// the tile + one flash block, N and K to the tile). `CMF_ZI_CHECKED=1`
/// builds them checked, for debugging an out-of-range suspicion.
fn pipeline(c: &Ctx, key: &str, src: &str, entry: &str) -> Option<Arc<wgpu::ComputePipeline>> {
    let zp = ZP.get_or_init(|| ZPipes {
        pipes: Mutex::new(HashMap::new()),
    });
    if let Some(p) = zp.pipes.lock().ok()?.get(key) {
        return Some(p.clone());
    }
    let checked = std::env::var("CMF_ZI_CHECKED").as_deref() == Ok("1");
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let desc = wgpu::ShaderModuleDescriptor {
        label: Some(key),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    };
    let m = if checked {
        c.device.create_shader_module(desc)
    } else {
        // SAFETY: the kernels in this file index only inside buffers the
        // chain sizes for them (see the padding rules on `ZState`), and
        // every loop has a uniform, finite trip count.
        unsafe {
            c.device
                .create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
        }
    };
    if let Some(e) = pollster::block_on(sc.pop()) {
        eprintln!("zimage: shader {key} rejected: {e}");
        if std::env::var("CMF_ZI_DUMP_WGSL").is_ok() {
            eprintln!("{src}");
        }
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        return None;
    }
    let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(key),
        layout: None,
        module: &m,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: c.pipeline_cache.as_ref(),
    });
    if let Some(e) = pollster::block_on(sc.pop()) {
        eprintln!("zimage: pipeline {key} rejected: {e}");
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        return None;
    }
    let p = Arc::new(p);
    zp.pipes.lock().ok()?.insert(key.to_string(), p.clone());
    Some(p)
}

// ════════════════════════════════════════════════════════════════════
// zi_mm — the tensor-core GEMM (plan S2/S4).
// ════════════════════════════════════════════════════════════════════

/// What a `zi_mm` tile does with its f32 accumulators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Epi {
    /// `out[m][ocol+n]` f32, stored straight from the accumulators.
    F32,
    /// `out[m][ocol+n]` f16 (qkv panel).
    F16,
    /// Plane rows interleaved in 16-row panels (gate, up, gate, up …):
    /// writes `silu(gate)·up` as f16, output width N/2.
    SwiGlu,
}

/// Tile geometry of one `zi_mm` pipeline. `bm × bn` output tile per
/// workgroup, `bk` K slice per stage, `wm × wn` subgroups (32 lanes each),
/// so every subgroup owns a `(bm/wm) × (bn/wn)` block of 16×16 accumulators.
/// `direct` = coop-load the operands straight from global memory (no
/// shared staging) — the A/B arm for the staging choreography.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MmCfg {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub wm: u32,
    pub wn: u32,
    pub epi: Epi,
    pub direct: bool,
    /// PROBE ONLY: f16 accumulators, stored raw as f16 (no flush to f32) —
    /// measures the speed ceiling of the f16-accumulate rate. Wrong answer
    /// at real K; never used by the chain.
    pub acc16_probe: bool,
    /// 2 = double-buffered shared staging (one barrier per K slice instead
    /// of two); 1 = single buffer + register prefetch.
    pub stages: u32,
}

impl MmCfg {
    pub const fn new(bm: u32, bn: u32, bk: u32, wm: u32, wn: u32, epi: Epi) -> Self {
        Self { bm, bn, bk, wm, wn, epi, direct: false, acc16_probe: false, stages: 1 }
    }
    pub fn valid(&self) -> bool {
        let nt = self.wm * self.wn * 32;
        let (tm, tn) = (self.bm / self.wm.max(1), self.bn / self.wn.max(1));
        let vpr = self.bk / 4;
        nt > 0
            && nt <= 1024
            && self.bk % 16 == 0
            && tm % 16 == 0
            && tn % 16 == 0
            && tm * self.wm == self.bm
            && tn * self.wn == self.bn
            && (self.bm * vpr) % nt == 0
            && (self.bn * vpr) % nt == 0
            && (self.epi != Epi::SwiGlu || (tn / 16) % 2 == 0)
    }
    fn key(&self) -> String {
        format!(
            "zi_mm_{}x{}x{}_{}x{}_{:?}{}",
            self.bm,
            self.bn,
            self.bk,
            self.wm,
            self.wn,
            self.epi,
            if self.direct {
                "_d".to_string()
            } else if self.acc16_probe {
                "_h".to_string()
            } else if self.stages == 2 {
                "_s2".to_string()
            } else {
                String::new()
            }
        )
    }
    /// Workgroup-shared bytes this variant declares.
    pub fn shared_bytes(&self) -> u32 {
        let lds = self.bk + 8;
        let stage = if self.direct { 0 } else { (self.bm + self.bn) * lds * 2 * self.stages.max(1) };
        let nw = self.wm * self.wn;
        let sc = match self.epi {
            Epi::F32 => 0,
            Epi::F16 => nw * 256 * 4,
            Epi::SwiGlu => nw * 512 * 4,
        };
        stage + sc
    }
}

/// The default tiles, picked by `zimage_gemmbench mm` on the RTX 3090
/// (see the module doc of the report). Overridable per epilogue with
/// `CMF_ZI_TILE=bm,bn,bk,wm,wn`.
pub fn default_cfg(epi: Epi) -> MmCfg {
    if let Ok(s) = std::env::var("CMF_ZI_TILE") {
        let v: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 5 {
            let c = MmCfg::new(v[0], v[1], v[2], v[3], v[4], epi);
            if c.valid() {
                return c;
            }
        }
    }
    MmCfg::new(128, 128, 32, 2, 2, epi)
}

/// WGSL of one `zi_mm` variant. Bindings: 0 plane `[N][K]` f16 (as
/// `vec4<f16>`), 1 activation `[M][K]` f16, 2 output, 3 `MmP`.
pub fn mm_src(g: MmCfg) -> String {
    use std::fmt::Write;
    assert!(g.valid(), "invalid zi_mm cfg {g:?}");
    let nw = g.wm * g.wn;
    let nt = nw * 32;
    let tm = g.bm / g.wm;
    let tn = g.bn / g.wn;
    let fm = tm / 16;
    let fnn = tn / 16;
    let lds = g.bk + 8;
    let vpr = g.bk / 4;
    let la = g.bm * vpr / nt;
    let lb = g.bn * vpr / nt;
    let mut s = String::new();
    let _ = writeln!(s, "enable f16;\nenable wgpu_cooperative_matrix;");
    let _ = writeln!(s, "diagnostic(off, derivative_uniformity);");
    let _ = writeln!(
        s,
        "struct MmP {{ m: u32, n: u32, k: u32, ldo: u32, ocol: u32, arow: u32, oscale: f32, _p: u32 }};"
    );
    if g.direct {
        let _ = writeln!(s, "@group(0) @binding(0) var<storage, read> wt: array<f16>;");
        let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> act: array<f16>;");
    } else {
        let _ = writeln!(s, "@group(0) @binding(0) var<storage, read> wt: array<vec4<f16>>;");
        let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> act: array<vec4<f16>>;");
    }
    match g.epi {
        _ if g.acc16_probe => {
            let _ = writeln!(s, "@group(0) @binding(2) var<storage, read_write> outp: array<f16>;");
        }
        Epi::F32 => {
            let _ = writeln!(s, "@group(0) @binding(2) var<storage, read_write> outp: array<f32>;");
        }
        _ => {
            let _ = writeln!(s, "@group(0) @binding(2) var<storage, read_write> outp: array<vec4<u32>>;");
        }
    }
    let _ = writeln!(s, "@group(0) @binding(3) var<uniform> p: MmP;");
    if !g.direct {
        let st = g.stages.max(1);
        let _ = writeln!(s, "var<workgroup> sa: array<f16, {}>;", g.bm * lds * st);
        let _ = writeln!(s, "var<workgroup> sb: array<f16, {}>;", g.bn * lds * st);
    }
    match g.epi {
        _ if g.acc16_probe => {}
        Epi::F32 => {}
        Epi::F16 => {
            let _ = writeln!(s, "var<workgroup> sc: array<f32, {}>;", nw * 256);
        }
        Epi::SwiGlu => {
            let _ = writeln!(s, "var<workgroup> sc: array<f32, {}>;", nw * 512);
        }
    }
    let _ = writeln!(
        s,
        "@compute @workgroup_size({nt})\nfn zi_mm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sg: u32) {{"
    );
    let _ = writeln!(s, "  let m0 = wid.y * {}u; let n0 = wid.x * {}u;", g.bm, g.bn);
    let _ = writeln!(s, "  let wy = sg / {}u; let wx = sg % {}u;", g.wn, g.wn);
    for i in 0..fm {
        for j in 0..fnn {
            let _ = writeln!(s, "  var c{i}_{j}: coop_mat16x16<{}, C>;", if g.acc16_probe { "f16" } else { "f32" });
        }
    }
    let _ = writeln!(s, "  let nkt = p.k / {}u;", g.bk);
    if g.direct {
        // Operands straight from global memory: no barriers at all.
        let _ = writeln!(s, "  let ar = p.arow + m0 + wy * {tm}u; let br = n0 + wx * {tn}u;");
        let _ = writeln!(s, "  for (var kt = 0u; kt < nkt; kt = kt + 1u) {{");
        for kk in 0..g.bk / 16 {
            let _ = writeln!(s, "    {{ let k0 = kt * {}u + {}u;", g.bk, kk * 16);
            for j in 0..fnn {
                let _ = writeln!(
                    s,
                    "      let b{j} = coopLoad<coop_mat16x16<f16, B>>(&wt[(br + {}u) * p.k + k0], p.k);",
                    j * 16
                );
            }
            for i in 0..fm {
                let _ = writeln!(
                    s,
                    "      let a{i} = coopLoadT<coop_mat16x16<f16, A>>(&act[(ar + {}u) * p.k + k0], p.k);",
                    i * 16
                );
                for j in 0..fnn {
                    let _ = writeln!(s, "      c{i}_{j} = coopMultiplyAdd(a{i}, b{j}, c{i}_{j});");
                }
            }
            let _ = writeln!(s, "    }}");
        }
        let _ = writeln!(s, "  }}");
    } else {
        let _ = writeln!(s, "  let kq = p.k / 4u;");
        for t in 0..la {
            let _ = writeln!(
                s,
                "  var ra{t} = act[(p.arow + m0 + (tid + {o}u) / {vpr}u) * kq + (tid + {o}u) % {vpr}u];",
                o = t * nt
            );
        }
        for t in 0..lb {
            let _ = writeln!(
                s,
                "  var rb{t} = wt[(n0 + (tid + {o}u) / {vpr}u) * kq + (tid + {o}u) % {vpr}u];",
                o = t * nt
            );
        }
        if g.stages == 2 {
            mm_loop_2stage(&mut s, g, nt, tm, tn, fm, fnn, lds, vpr, la, lb);
        } else {
            let _ = writeln!(s, "  for (var kt = 0u; kt < nkt; kt = kt + 1u) {{");
            for (arr, reg, cnt) in [("sa", "ra", la), ("sb", "rb", lb)] {
                for t in 0..cnt {
                    let _ = writeln!(
                        s,
                        "    {{ let d = ((tid + {o}u) / {vpr}u) * {lds}u + ((tid + {o}u) % {vpr}u) * 4u; {arr}[d] = {reg}{t}.x; {arr}[d + 1u] = {reg}{t}.y; {arr}[d + 2u] = {reg}{t}.z; {arr}[d + 3u] = {reg}{t}.w; }}",
                        o = t * nt
                    );
                }
            }
            let _ = writeln!(s, "    workgroupBarrier();");
            let _ = writeln!(s, "    if (kt + 1u < nkt) {{ let kb = (kt + 1u) * {vpr}u;");
            for t in 0..la {
                let _ = writeln!(
                    s,
                    "      ra{t} = act[(p.arow + m0 + (tid + {o}u) / {vpr}u) * kq + kb + (tid + {o}u) % {vpr}u];",
                    o = t * nt
                );
            }
            for t in 0..lb {
                let _ = writeln!(
                    s,
                    "      rb{t} = wt[(n0 + (tid + {o}u) / {vpr}u) * kq + kb + (tid + {o}u) % {vpr}u];",
                    o = t * nt
                );
            }
            let _ = writeln!(s, "    }}");
            for kk in 0..g.bk / 16 {
                let _ = writeln!(s, "    {{");
                for j in 0..fnn {
                    let _ = writeln!(
                        s,
                        "      let b{j} = coopLoad<coop_mat16x16<f16, B>>(&sb[(wx * {tn}u + {}u) * {lds}u + {}u], {lds}u);",
                        j * 16,
                        kk * 16
                    );
                }
                for i in 0..fm {
                    let _ = writeln!(
                        s,
                        "      let a{i} = coopLoadT<coop_mat16x16<f16, A>>(&sa[(wy * {tm}u + {}u) * {lds}u + {}u], {lds}u);",
                        i * 16,
                        kk * 16
                    );
                    for j in 0..fnn {
                        let _ = writeln!(s, "      c{i}_{j} = coopMultiplyAdd(a{i}, b{j}, c{i}_{j});");
                    }
                }
                let _ = writeln!(s, "    }}");
            }

        let _ = writeln!(s, "    workgroupBarrier();");
        let _ = writeln!(s, "  }}");
        }
    }
    // ── epilogue
    let _ = writeln!(s, "  let orow = m0 + wy * {tm}u;");
    // naga 30 does not emit a runtime coopStore stride before the store
    // (panics "Expression is not cached"): bind it to a `let` first.
    let _ = writeln!(s, "  let ldo = p.ldo;");
    match g.epi {
        _ if g.acc16_probe => {
            let _ = writeln!(s, "  let ocol = p.ocol + n0 + wx * {tn}u;");
            for i in 0..fm {
                for j in 0..fnn {
                    let _ = writeln!(s, "  {{ let oi = (orow + {}u) * ldo + ocol + {}u; coopStoreT(c{i}_{j}, &outp[oi], ldo); }}", i * 16, j * 16);
                }
            }
        }
        Epi::F32 => {
            let _ = writeln!(s, "  let ocol = p.ocol + n0 + wx * {tn}u;");
            for i in 0..fm {
                for j in 0..fnn {
                    let _ = writeln!(
                        s,
                        "  {{ let oi = (orow + {}u) * ldo + ocol + {}u; coopStoreT(c{i}_{j}, &outp[oi], ldo); }}",
                        i * 16,
                        j * 16
                    );
                }
            }
        }
        Epi::F16 => {
            // One 16×16 fragment at a time through a 1 KB per-subgroup
            // scratch: lane → (row lane/2, 8 columns) → one 16-byte store.
            let _ = writeln!(s, "  let ocol = p.ocol + n0 + wx * {tn}u;");
            let _ = writeln!(s, "  let lane = tid - sg * 32u;");
            let _ = writeln!(s, "  let sb0 = sg * 256u;");
            let _ = writeln!(s, "  let er = lane / 2u; let ec = (lane % 2u) * 8u; let eb = sb0 + er * 16u + ec;");
            for i in 0..fm {
                for j in 0..fnn {
                    let _ = writeln!(s, "  coopStoreT(c{i}_{j}, &sc[sb0], 16u);");
                    let _ = writeln!(s, "  workgroupBarrier();");
                    let _ = writeln!(s, "  outp[((orow + {}u + er) * ldo + ocol + {}u + ec) / 8u] = vec4<u32>(pack2x16float(vec2<f32>(sc[eb], sc[eb + 1u]) * p.oscale), pack2x16float(vec2<f32>(sc[eb + 2u], sc[eb + 3u]) * p.oscale), pack2x16float(vec2<f32>(sc[eb + 4u], sc[eb + 5u]) * p.oscale), pack2x16float(vec2<f32>(sc[eb + 6u], sc[eb + 7u]) * p.oscale));", i * 16, j * 16);
                    let _ = writeln!(s, "  workgroupBarrier();");
                }
            }
        }
        Epi::SwiGlu => {
            // Fragment pair (gate j=2q, up j=2q+1) → 16 output columns.
            let _ = writeln!(s, "  let ocol = (p.ocol + n0 + wx * {tn}u) / 2u;");
            let _ = writeln!(s, "  let lane = tid - sg * 32u;");
            let _ = writeln!(s, "  let sb0 = sg * 512u;");
            let _ = writeln!(s, "  let er = lane / 2u; let ec = (lane % 2u) * 8u; let eb = sb0 + er * 16u + ec;");
            for i in 0..fm {
                for q in 0..fnn / 2 {
                    let _ = writeln!(s, "  coopStoreT(c{i}_{}, &sc[sb0], 16u);", 2 * q);
                    let _ = writeln!(s, "  coopStoreT(c{i}_{}, &sc[sb0 + 256u], 16u);", 2 * q + 1);
                    let _ = writeln!(s, "  workgroupBarrier();");
                    let _ = writeln!(s, "  {{ var hv: array<f32, 8>;");
                    let _ = writeln!(s, "    for (var e = 0u; e < 8u; e = e + 1u) {{ let gg = sc[eb + e]; hv[e] = gg / (1.0 + exp(-gg)) * sc[eb + 256u + e] * p.oscale; }}");
                    let _ = writeln!(s, "    outp[((orow + {}u + er) * ldo + ocol + {}u + ec) / 8u] = vec4<u32>(pack2x16float(vec2<f32>(hv[0], hv[1])), pack2x16float(vec2<f32>(hv[2], hv[3])), pack2x16float(vec2<f32>(hv[4], hv[5])), pack2x16float(vec2<f32>(hv[6], hv[7]))); }}", i * 16, q * 16);
                    let _ = writeln!(s, "  workgroupBarrier();");
                }
            }
        }
    }
    let _ = writeln!(s, "}}");
    s
}

/// The K loop with two shared stages: while slice kt is multiplied out of
/// stage kt%2, slice kt+1 (already in registers) is stored into the other
/// stage and slice kt+2 is fetched from global memory — one barrier per
/// slice. The first slice is staged before the loop.
#[allow(clippy::too_many_arguments)]
fn mm_loop_2stage(
    s: &mut String,
    g: MmCfg,
    nt: u32,
    tm: u32,
    tn: u32,
    fm: u32,
    fnn: u32,
    lds: u32,
    vpr: u32,
    la: u32,
    lb: u32,
) {
    use std::fmt::Write;
    let (sza, szb) = (g.bm * lds, g.bn * lds);
    let store = |s: &mut String, stage: &str| {
        for (arr, reg, cnt, sz) in [("sa", "ra", la, sza), ("sb", "rb", lb, szb)] {
            for t in 0..cnt {
                let _ = writeln!(
                    s,
                    "    {{ let d = {stage} * {sz}u + ((tid + {o}u) / {vpr}u) * {lds}u + ((tid + {o}u) % {vpr}u) * 4u; {arr}[d] = {reg}{t}.x; {arr}[d + 1u] = {reg}{t}.y; {arr}[d + 2u] = {reg}{t}.z; {arr}[d + 3u] = {reg}{t}.w; }}",
                    o = t * nt
                );
            }
        }
    };
    let fetch = |s: &mut String, kb: &str| {
        for t in 0..la {
            let _ = writeln!(s, "      ra{t} = act[(p.arow + m0 + (tid + {o}u) / {vpr}u) * kq + {kb} + (tid + {o}u) % {vpr}u];", o = t * nt);
        }
        for t in 0..lb {
            let _ = writeln!(s, "      rb{t} = wt[(n0 + (tid + {o}u) / {vpr}u) * kq + {kb} + (tid + {o}u) % {vpr}u];", o = t * nt);
        }
    };
    store(s, "0u");
    let _ = writeln!(s, "  workgroupBarrier();");
    let _ = writeln!(s, "  if (1u < nkt) {{");
    fetch(s, &format!("{vpr}u"));
    let _ = writeln!(s, "  }}");
    let _ = writeln!(s, "  for (var kt = 0u; kt < nkt; kt = kt + 1u) {{");
    let _ = writeln!(s, "    let ca = (kt & 1u) * {sza}u; let cb = (kt & 1u) * {szb}u;");
    for kk in 0..g.bk / 16 {
        let _ = writeln!(s, "    {{");
        for j in 0..fnn {
            let _ = writeln!(s, "      let b{j} = coopLoad<coop_mat16x16<f16, B>>(&sb[cb + (wx * {tn}u + {}u) * {lds}u + {}u], {lds}u);", j * 16, kk * 16);
        }
        for i in 0..fm {
            let _ = writeln!(s, "      let a{i} = coopLoadT<coop_mat16x16<f16, A>>(&sa[ca + (wy * {tm}u + {}u) * {lds}u + {}u], {lds}u);", i * 16, kk * 16);
            for j in 0..fnn {
                let _ = writeln!(s, "      c{i}_{j} = coopMultiplyAdd(a{i}, b{j}, c{i}_{j});");
            }
        }
        let _ = writeln!(s, "    }}");
    }
    let _ = writeln!(s, "    if (kt + 1u < nkt) {{");
    store(s, "((kt + 1u) & 1u)");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    if (kt + 2u < nkt) {{ let kb2 = (kt + 2u) * {vpr}u;");
    fetch(s, "kb2");
    let _ = writeln!(s, "    }}");
    let _ = writeln!(s, "  }}");
}

fn mm_pipe(c: &Ctx, g: MmCfg) -> Option<Arc<wgpu::ComputePipeline>> {
    if !g.valid() || g.shared_bytes() > c.device.limits().max_compute_workgroup_storage_size {
        return None;
    }
    pipeline(c, &g.key(), &mm_src(g), "zi_mm")
}

/// Uniform of one `zi_mm` dispatch.
#[derive(Clone, Copy, Debug)]
pub struct MmArgs {
    /// Rows the dispatch covers (padded to `bm` by the caller's buffers).
    pub m: u32,
    /// Plane rows (output columns before SwiGLU halving).
    pub n: u32,
    pub k: u32,
    /// Output row stride in elements, and output column offset (for SwiGLU:
    /// in plane-row units; the kernel halves it).
    pub ldo: u32,
    pub ocol: u32,
    /// First activation row.
    pub arow: u32,
    /// Output multiplier (1.0 = none); for the f16 epilogues it is the
    /// power-of-two range guard the next GEMM divides back out.
    pub oscale: f32,
}

fn mm_uniform(c: &Ctx, a: &MmArgs) -> wgpu::Buffer {
    let w: [u32; 8] = [a.m, a.n, a.k, a.ldo, a.ocol, a.arow, a.oscale.to_bits(), 0];
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zi_mm_p"),
        contents: bytemuck::cast_slice(&w),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// One prebuilt GEMM dispatch.
struct MmCall {
    pipe: Arc<wgpu::ComputePipeline>,
    bg: wgpu::BindGroup,
    grid: (u32, u32),
}

fn mm_call(
    c: &Ctx,
    g: MmCfg,
    a: &MmArgs,
    plane: &wgpu::Buffer,
    act: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> Option<MmCall> {
    if a.n % g.bn != 0 || a.k % g.bk != 0 {
        return None;
    }
    let pipe = mm_pipe(c, g)?;
    let u = mm_uniform(c, a);
    let bg = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zi_mm"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            super::bind_buf(0, plane),
            super::bind_buf(1, act),
            super::bind_buf(2, out),
            super::bind_buf(3, &u),
        ],
    });
    Some(MmCall {
        pipe,
        bg,
        grid: (a.n / g.bn, a.m.div_ceil(g.bm)),
    })
}

impl MmCall {
    fn record(&self, pass: &mut wgpu::ComputePass) {
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }
}

// ════════════════════════════════════════════════════════════════════
// zi_flash — bidirectional flash attention on the matrix units (plan S3).
// ════════════════════════════════════════════════════════════════════

/// Flash-attention variant: `nw` subgroups × 16 query rows per workgroup,
/// `bc` keys per block (multiple of 16; Z-Image segment lengths are
/// multiples of 32, so bc ∈ {16, 32} never needs a key mask).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlashCfg {
    pub nw: u32,
    pub bc: u32,
}

impl FlashCfg {
    pub fn shared_bytes(&self) -> u32 {
        let ldk = 128 + 8;
        let ldp = self.bc + flash_pad();
        self.bc * ldk * 2 + (128 * (self.bc + flash_pad()) * 2).max(self.bc * ldk * 2) + self.nw * 16 * self.bc * 4 + self.nw * 16 * ldp * 2
            + self.nw * 32 * 4 * 2
            + 8
    }
    fn key(&self) -> String {
        format!(
            "zi_flash_{}_{}_{}_{}_{}_{}",
            self.nw,
            self.bc,
            flash_thr(),
            flash_vt(),
            flash_pad(),
            std::env::var("CMF_ZI_FLASH_DBG").unwrap_or_default()
        )
    }
}

/// Row padding (halves) of the P and transposed-V staging tiles.
fn flash_pad() -> u32 {
    std::env::var("CMF_ZI_FLASH_PAD").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

fn flash_vt() -> bool {
    std::env::var("CMF_ZI_FLASH_VT").as_deref() == Ok("1")
}

/// Lazy-rescale threshold in log2 units (P ≤ 2^thr in f16). `CMF_ZI_FLASH_THR`
/// overrides (0 = rescale whenever the max grows: the exact-classic path).
fn flash_thr() -> String {
    let v: f32 = std::env::var("CMF_ZI_FLASH_THR").ok().and_then(|v| v.parse().ok()).unwrap_or(8.0);
    format!("{:.1}", v.clamp(0.0, 14.0))
}

pub fn default_flash() -> FlashCfg {
    if let Ok(s) = std::env::var("CMF_ZI_FLASH") {
        let v: Vec<u32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 2 {
            return FlashCfg { nw: v[0], bc: v[1] };
        }
    }
    FlashCfg { nw: 4, bc: 16 }
}

/// WGSL of `zi_flash`. hd is fixed at 128 (8 fragments of 16).
/// Bindings: 0 qkv panel as f16 (Q fragments load straight from it),
/// 1 the same buffer as `vec4<f16>` (K/V tiles), 2 output `[M][o_ld]` f16
/// (as `vec4<u32>`), 3 `FP`.
///
/// Per block of `bc` keys, per subgroup (16 query rows):
///   S = Q·Kᵀ (f32 acc) → shared → lanes (row = lane&15, half = lane>>4)
///   take the row max; if ANY row of the workgroup grew past the max the
///   exponentials are anchored at by > 8 (log2 units) every subgroup
///   rescales its O accumulators through shared (the FA4 lazy rescale —
///   exact; P ≤ 2⁸ so f16 P is safe), P = 2^(s·scale·log2e − m) → f16 →
///   O += P·V. The workgroup-uniform decision costs one
///   `workgroupUniformLoad` per block instead of a per-row rescale.
pub fn flash_src(f: FlashCfg) -> String {
    use std::fmt::Write;
    let nt = f.nw * 32;
    let bc = f.bc;
    let ldk = 136u32;
    let ldp = bc + flash_pad();
    let nkf = bc / 16;
    let vk = bc * 16; // vec4<f16> per K (or V) tile: bc rows × 128/8… (128 halves = 32 vec4<f16>)
    let vk = vk * 2; // 128 halves / 4 = 32 vec4<f16> per row
    let per = vk / nt;
    assert!(per * nt == vk, "flash cfg {f:?}");
    let half = bc / 2;
    // `CMF_ZI_FLASH_VT=1`: V staged transposed ([dim][key]) and read as a
    // column-major B. Measured slower (35 vs 51 TF at 1024²): row-major B.
    let vt = flash_vt();
    let ldv = bc + flash_pad();
    let dbg = std::env::var("CMF_ZI_FLASH_DBG").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    let mut s = String::new();
    let _ = writeln!(s, "enable f16;\nenable wgpu_cooperative_matrix;\ndiagnostic(off, derivative_uniformity);");
    let _ = writeln!(s, "struct FP {{ q_off: u32, len: u32, ld: u32, k_col: u32, v_col: u32, o_ld: u32, scl: f32, nh: u32 }};");
    let _ = writeln!(s, "@group(0) @binding(0) var<storage, read> qh: array<f16>;");
    let _ = writeln!(s, "@group(0) @binding(1) var<storage, read> qv: array<vec4<f16>>;");
    let _ = writeln!(s, "@group(0) @binding(2) var<storage, read_write> oh: array<vec4<u32>>;");
    let _ = writeln!(s, "@group(0) @binding(3) var<uniform> p: FP;");
    let _ = writeln!(s, "var<workgroup> sk: array<f16, {}>;", bc * ldk);
    if vt {
        let _ = writeln!(s, "var<workgroup> sv: array<f16, {}>;", 128 * ldv);
    } else {
        let _ = writeln!(s, "var<workgroup> sv: array<f16, {}>;", bc * ldk);
    }
    let _ = writeln!(s, "var<workgroup> ss: array<f32, {}>;", f.nw * 16 * bc);
    let _ = writeln!(s, "var<workgroup> sp: array<f16, {}>;", f.nw * 16 * ldp);
    let _ = writeln!(s, "var<workgroup> smx: array<f32, {}>;", nt);
    // 16 bytes, not 8: an 8-byte array here shifted every workgroup array
    // after it off 16-byte alignment, and the cooperative loads (ldmatrix-
    // class, low address bits ignored) then read each 16-half row 4 halves
    // early — the last 4 keys of every block vanished and row 0 read the
    // previous array (inf/NaN). Every workgroup array in this file is a
    // multiple of 16 bytes.
    let _ = writeln!(s, "var<workgroup> sfl: array<u32, 4>;");
    let _ = writeln!(s, "@compute @workgroup_size({nt})\nfn zi_flash(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sg: u32) {{");
    let _ = writeln!(s, "  let h = wid.y;");
    let _ = writeln!(s, "  let q0 = wid.x * {}u + sg * 16u;", f.nw * 16);
    let _ = writeln!(s, "  let qrow = p.q_off + q0;");
    let _ = writeln!(s, "  let lane = tid - sg * 32u;");
    let _ = writeln!(s, "  let r = lane & 15u; let hf = lane >> 4u;");
    let _ = writeln!(s, "  let ldq = p.ld / 4u;");
    for d in 0..8 {
        let _ = writeln!(s, "  let qa{d} = coopLoadT<coop_mat16x16<f16, A>>(&qh[qrow * p.ld + h * 128u + {}u], p.ld);", d * 16);
    }
    for j in 0..8 {
        let _ = writeln!(s, "  var o{j}: coop_mat16x16<f32, C>;");
    }
    // Never assigned: the zero every block's S accumulation starts from.
    // (A `var` declared INSIDE the loop is zeroed once at function entry by
    // naga, not per iteration — S then summed across key blocks.)
    let _ = writeln!(s, "  var zc: coop_mat16x16<f32, C>;");
    let _ = writeln!(s, "  var mu = -1.0e30; var ls = 0.0;");
    let _ = writeln!(s, "  if (tid < 2u) {{ sfl[tid] = 0u; }}");
    let _ = writeln!(s, "  let nkb = p.len / {bc}u;");
    // prefetch registers for K and V tiles
    for t in 0..per {
        let _ = writeln!(s, "  var rk{t}: vec4<f16>; var rv{t}: vec4<f16>;");
        let _ = writeln!(s, "  {{ let idx = tid + {o}u; let row = p.q_off + idx / 32u; let c4 = idx % 32u; rk{t} = qv[row * ldq + (p.k_col + h * 128u) / 4u + c4]; rv{t} = qv[row * ldq + (p.v_col + h * 128u) / 4u + c4]; }}", o = t * nt);
    }
    let _ = writeln!(s, "  for (var kb = 0u; kb < nkb; kb = kb + 1u) {{");
    for t in 0..per {
        if vt {
            let _ = writeln!(s, "    {{ let idx = tid + {o}u; let d = (idx / 32u) * {ldk}u + (idx % 32u) * 4u; sk[d] = rk{t}.x; sk[d + 1u] = rk{t}.y; sk[d + 2u] = rk{t}.z; sk[d + 3u] = rk{t}.w; let vd = (idx % 32u) * 4u * {ldv}u + idx / 32u; sv[vd] = rv{t}.x; sv[vd + {ldv}u] = rv{t}.y; sv[vd + {}u] = rv{t}.z; sv[vd + {}u] = rv{t}.w; }}", 2 * ldv, 3 * ldv, o = t * nt);
        } else {
            let _ = writeln!(s, "    {{ let idx = tid + {o}u; let d = (idx / 32u) * {ldk}u + (idx % 32u) * 4u; sk[d] = rk{t}.x; sk[d + 1u] = rk{t}.y; sk[d + 2u] = rk{t}.z; sk[d + 3u] = rk{t}.w; sv[d] = rv{t}.x; sv[d + 1u] = rv{t}.y; sv[d + 2u] = rv{t}.z; sv[d + 3u] = rv{t}.w; }}", o = t * nt);
        }
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "    if (kb + 1u < nkb) {{");
    for t in 0..per {
        let _ = writeln!(s, "      {{ let idx = tid + {o}u; let row = p.q_off + (kb + 1u) * {bc}u + idx / 32u; let c4 = idx % 32u; rk{t} = qv[row * ldq + (p.k_col + h * 128u) / 4u + c4]; rv{t} = qv[row * ldq + (p.v_col + h * 128u) / 4u + c4]; }}", o = t * nt);
    }
    let _ = writeln!(s, "    }}");
    // S = Q K^T
    for j in 0..nkf {
        if dbg == 7 {
            let _ = writeln!(s, "    var s{j} = zc;");
            let _ = writeln!(s, "    coopStoreT(s{j}, &ss[sg * {}u + {}u], {bc}u);", 16 * bc, j * 16);
            continue;
        }
        let _ = writeln!(s, "    let kf{j}_0 = coopLoad<coop_mat16x16<f16, B>>(&sk[{}u], {ldk}u);", j * 16 * ldk);
        let _ = writeln!(s, "    var s{j} = coopMultiplyAdd(qa0, kf{j}_0, zc);");
        for d in 1..8 {
            let _ = writeln!(s, "    let kf{j}_{d} = coopLoad<coop_mat16x16<f16, B>>(&sk[{}u], {ldk}u);", j * 16 * ldk + d * 16);
            let _ = writeln!(s, "    s{j} = coopMultiplyAdd(qa{d}, kf{j}_{d}, s{j});");
        }
        let _ = writeln!(s, "    coopStoreT(s{j}, &ss[sg * {}u + {}u], {bc}u);", 16 * bc, j * 16);
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    // partial row max
    let _ = writeln!(s, "    let sbase = sg * {}u + r * {bc}u + hf * {half}u;", 16 * bc);
    let _ = writeln!(s, "    var pm = -1.0e30;");
    // Lane loops are unrolled by the generator (fixed trip count).
    for e in 0..half {
        let _ = writeln!(s, "    pm = max(pm, ss[sbase + {e}u] * p.scl);");
    }
    let _ = writeln!(s, "    smx[tid] = pm;");
    let _ = writeln!(s, "    if (pm > mu + {}) {{ sfl[kb & 1u] = 1u; }}", flash_thr());
    let _ = writeln!(s, "    let need = workgroupUniformLoad(&sfl[kb & 1u]);");
    let _ = writeln!(s, "    if (tid == 0u) {{ sfl[(kb + 1u) & 1u] = 0u; }}");
    let _ = writeln!(s, "    let rmax = max(pm, smx[tid ^ 16u]);");
    let _ = writeln!(s, "    if (need != 0u) {{");
    let _ = writeln!(s, "      let mn = max(mu, rmax);");
    let _ = writeln!(s, "      let alpha = exp2(mu - mn);");
    let _ = writeln!(s, "      ls = ls * alpha; mu = mn;");
    let _ = writeln!(s, "      if (kb > 0u) {{");
    // rescale O through ss, nkf accumulators per round
    let rounds = 8 / nkf;
    for rd in 0..if dbg >= 5 { 0 } else { rounds } {
        for q in 0..nkf {
            let j = rd * nkf + q;
            let _ = writeln!(s, "        coopStoreT(o{j}, &ss[sg * {}u + {}u], {bc}u);", 16 * bc, q * 16);
        }
        let _ = writeln!(s, "        workgroupBarrier();");
        for e in 0..half {
            let _ = writeln!(s, "        ss[sbase + {e}u] = ss[sbase + {e}u] * alpha;");
        }
        let _ = writeln!(s, "        workgroupBarrier();");
        for q in 0..nkf {
            let j = rd * nkf + q;
            let _ = writeln!(s, "        o{j} = coopLoadT<coop_mat16x16<f32, C>>(&ss[sg * {}u + {}u], {bc}u);", 16 * bc, q * 16);
        }
        let _ = writeln!(s, "        workgroupBarrier();");
    }
    let _ = writeln!(s, "      }}");
    // the S values were overwritten by the rescale rounds when kb > 0:
    // recompute S is too costly; instead the rescale rounds run BEFORE
    // reading S? — no: S lives in ss too. Store S again from registers.
    for j in 0..if dbg >= 6 { 0 } else { nkf } {
        let _ = writeln!(s, "      coopStoreT(s{j}, &ss[sg * {}u + {}u], {bc}u);", 16 * bc, j * 16);
    }
    let _ = writeln!(s, "      workgroupBarrier();");
    let _ = writeln!(s, "    }}");
    // P = exp2(s*scl - mu) → f16, row sums
    let _ = writeln!(s, "    let pbase = sg * {}u + r * {ldp}u + hf * {half}u;", 16 * ldp);
    for e in 0..half {
        let _ = writeln!(s, "    let pv{e} = exp2(ss[sbase + {e}u] * p.scl - mu);");
    }
    for e in 0..half {
        let _ = writeln!(s, "    ls = ls + pv{e}; sp[pbase + {e}u] = f16(pv{e});");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    // O += P V
    for kk in 0..nkf {
        let _ = writeln!(s, "    {{ let pa = coopLoadT<coop_mat16x16<f16, A>>(&sp[sg * {}u + {}u], {ldp}u);", 16 * ldp, kk * 16);
        for j in 0..8 {
            if vt {
                let _ = writeln!(s, "      let vf{j} = coopLoad<coop_mat16x16<f16, B>>(&sv[{}u], {ldv}u);", j * 16 * ldv + kk * 16);
            } else {
                let _ = writeln!(s, "      let vf{j} = coopLoadT<coop_mat16x16<f16, B>>(&sv[{}u], {ldk}u);", kk * 16 * ldk + j * 16);
            }
            let _ = writeln!(s, "      o{j} = coopMultiplyAdd(pa, vf{j}, o{j});");
        }
        let _ = writeln!(s, "    }}");
    }
    let _ = writeln!(s, "    workgroupBarrier();");
    let _ = writeln!(s, "  }}");
    // finalize: l per row, O / l → f16 out
    let _ = writeln!(s, "  smx[tid] = ls;");
    let _ = writeln!(s, "  workgroupBarrier();");
    let _ = writeln!(s, "  let lt = ls + smx[tid ^ 16u];");
    let _ = writeln!(s, "  let inv = select(0.0, 1.0 / lt, lt > 0.0);");
    // CMF_ZI_FLASH_DBG: 1 = write the row sum l, 2 = raw O (no 1/l),
    // 3 = the anchor max mu; 5 = no O rescale block, 6 = no coop op in the
    // rescale branch at all, 7 = 6 + no S MMAs (S = 0).
    let _ = writeln!(s, "  let sbase = sg * {}u + r * {bc}u + hf * {half}u;", 16 * bc);
    let _ = writeln!(s, "  let orow = qrow + r;");
    let _ = writeln!(s, "  let live = q0 + r < p.len;");
    for rd in 0..rounds {
        for q in 0..nkf {
            let j = rd * nkf + q;
            let _ = writeln!(s, "  coopStoreT(o{j}, &ss[sg * {}u + {}u], {bc}u);", 16 * bc, q * 16);
        }
        let _ = writeln!(s, "  workgroupBarrier();");
        // this lane's `half` columns → half/8 vec4<u32> stores
        let _ = writeln!(s, "  if (live) {{");
        match dbg {
            1 => { let _ = writeln!(s, "    for (var e = 0u; e < {half}u; e = e + 1u) {{ ss[sbase + e] = lt * lt; }}"); }
            2 => { let _ = writeln!(s, "    for (var e = 0u; e < {half}u; e = e + 1u) {{ ss[sbase + e] = ss[sbase + e] * lt; }}"); }
            3 => { let _ = writeln!(s, "    for (var e = 0u; e < {half}u; e = e + 1u) {{ ss[sbase + e] = mu * lt; }}"); }
            _ => {}
        }
        for c8 in 0..half / 8 {
            let b = format!("sbase + {}u", c8 * 8);
            let _ = writeln!(s, "    {{ let b = {b}; oh[(orow * p.o_ld + h * 128u + {}u + hf * {half}u + {}u) / 8u] = vec4<u32>(pack2x16float(vec2<f32>(ss[b], ss[b + 1u]) * inv), pack2x16float(vec2<f32>(ss[b + 2u], ss[b + 3u]) * inv), pack2x16float(vec2<f32>(ss[b + 4u], ss[b + 5u]) * inv), pack2x16float(vec2<f32>(ss[b + 6u], ss[b + 7u]) * inv)); }}", rd * bc, c8 * 8);
        }
        let _ = writeln!(s, "  }}");
        let _ = writeln!(s, "  workgroupBarrier();");
    }
    let _ = writeln!(s, "}}");
    s
}

// ════════════════════════════════════════════════════════════════════
// Row kernels: gated residual + next pre-norm, qk-norm + RoPE, embed,
// final layer, fills.
// ════════════════════════════════════════════════════════════════════

/// `zi_rowop` mode bits.
pub const ROW_GRES: u32 = 1; // x += gate · RMS(br)·w_post
pub const ROW_GATE_MOD: u32 = 2; // gate = tanh(mods[g_off..]) (else 1)
pub const ROW_SCALE_MOD: u32 = 4; // pre-norm ·(1 + mods[s_off..]) (else ·1)
pub const ROW_PRE: u32 = 8; // write the next pre-norm as f16 into xn

const ROWOP_SRC: &str = r#"
struct RP { h: u32, mode: u32, g_off: u32, s_off: u32, eps_post: f32, eps_pre: f32, oscale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read> br: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xn: array<u32>;
@group(0) @binding(3) var<storage, read> wpost: array<f32>;
@group(0) @binding(4) var<storage, read> wpre: array<f32>;
@group(0) @binding(5) var<storage, read> mods: array<f32>;
@group(0) @binding(6) var<uniform> rp: RP;
var<workgroup> red: array<f32, 256>;

fn wsum(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// One workgroup per token row; each thread owns up to 8 pairs (hidden ≤ 4096).
@compute @workgroup_size(256)
fn zi_rowop(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x + wid.y * 65535u;
    let np = rp.h / 2u;
    let base = row * rp.h;
    var xv: array<vec2<f32>, 8>;
    if ((rp.mode & 1u) != 0u) {
        var bv: array<vec2<f32>, 8>;
        var ss = 0.0;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let pi = lid + j * 256u;
            if (pi < np) {
                let b = vec2<f32>(br[base + 2u * pi], br[base + 2u * pi + 1u]);
                bv[j] = b;
                ss = ss + b.x * b.x + b.y * b.y;
            }
        }
        let rr = inverseSqrt(wsum(ss, lid) / f32(rp.h) + rp.eps_post);
        for (var j = 0u; j < 8u; j = j + 1u) {
            let pi = lid + j * 256u;
            if (pi < np) {
                let c = 2u * pi;
                var g = vec2<f32>(1.0, 1.0);
                if ((rp.mode & 2u) != 0u) {
                    g = tanh(vec2<f32>(mods[rp.g_off + c], mods[rp.g_off + c + 1u]));
                }
                let w = vec2<f32>(wpost[c], wpost[c + 1u]);
                let xo = vec2<f32>(x[base + c], x[base + c + 1u]) + g * (bv[j] * rr) * w;
                x[base + c] = xo.x;
                x[base + c + 1u] = xo.y;
                xv[j] = xo;
            }
        }
    } else {
        for (var j = 0u; j < 8u; j = j + 1u) {
            let pi = lid + j * 256u;
            if (pi < np) {
                xv[j] = vec2<f32>(x[base + 2u * pi], x[base + 2u * pi + 1u]);
            }
        }
    }
    if ((rp.mode & 8u) == 0u) { return; }
    var ss2 = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
        let pi = lid + j * 256u;
        if (pi < np) { ss2 = ss2 + xv[j].x * xv[j].x + xv[j].y * xv[j].y; }
    }
    let r2 = inverseSqrt(wsum(ss2, lid) / f32(rp.h) + rp.eps_pre);
    for (var j = 0u; j < 8u; j = j + 1u) {
        let pi = lid + j * 256u;
        if (pi < np) {
            let c = 2u * pi;
            var s = vec2<f32>(1.0, 1.0);
            if ((rp.mode & 4u) != 0u) {
                s = s + vec2<f32>(mods[rp.s_off + c], mods[rp.s_off + c + 1u]);
            }
            let y = xv[j] * r2 * vec2<f32>(wpre[c], wpre[c + 1u]) * s * rp.oscale;
            xn[row * np + pi] = pack2x16float(y);
        }
    }
}
"#;

const QKROPE_SRC: &str = r#"
struct QP { ld: u32, nh: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> qkv: array<u32>;
@group(0) @binding(1) var<storage, read> wq: array<f32>;
@group(0) @binding(2) var<storage, read> wk: array<f32>;
@group(0) @binding(3) var<storage, read> rc: array<f32>;
@group(0) @binding(4) var<storage, read> rs: array<f32>;
@group(0) @binding(5) var<uniform> qp: QP;
var<workgroup> red: array<f32, 64>;

// One workgroup (64 lanes = 64 complex pairs of hd 128) per (token, q|k
// head). RMSNorm over the head (eps from QP), weight, then the
// complex-interleaved rotation by the token's (cos, sin) row.
@compute @workgroup_size(64)
fn zi_qkrope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.y + wid.z * 65535u;
    let hv = wid.x;
    let isk = hv >= qp.nh;
    let hh = hv % qp.nh;
    let col = select(0u, qp.nh * 128u, isk) + hh * 128u;
    let idx = (row * qp.ld + col) / 2u + lid;
    let v = unpack2x16float(qkv[idx]);
    red[lid] = v.x * v.x + v.y * v.y;
    workgroupBarrier();
    var st = 32u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let rr = inverseSqrt(red[0] / 128.0 + qp.eps);
    var w = vec2<f32>(wq[2u * lid], wq[2u * lid + 1u]);
    if (isk) { w = vec2<f32>(wk[2u * lid], wk[2u * lid + 1u]); }
    let a = v.x * rr * w.x;
    let b = v.y * rr * w.y;
    let c = rc[row * 64u + lid];
    let s = rs[row * 64u + lid];
    qkv[idx] = pack2x16float(vec2<f32>(a * c - b * s, a * s + b * c));
}
"#;

const EMBED_SRC: &str = r#"
struct EP { h: u32, n_img: u32, seg: u32, pd: u32 };
@group(0) @binding(0) var<storage, read> tok: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> xpad: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<uniform> ep: EP;
var<workgroup> tr: array<f32, 64>;

// x_embedder: one workgroup per token row, 256 threads over the hidden
// width; rows whose index inside their segment is ≥ n_img get x_pad.
@compute @workgroup_size(256)
fn zi_embed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let pad = (row % ep.seg) >= ep.n_img;
    if (lid < ep.pd) { tr[lid] = tok[row * ep.pd + lid]; }
    workgroupBarrier();
    for (var c = lid; c < ep.h; c = c + 256u) {
        var acc = b[c];
        for (var k = 0u; k < ep.pd; k = k + 1u) { acc = acc + tr[k] * w[c * ep.pd + k]; }
        x[row * ep.h + c] = select(acc, xpad[c], pad);
    }
}
"#;

const FINAL_SRC: &str = r#"
struct FP { h: u32, pd: u32, eps: f32, n_img: u32, seg: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> fsc: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<uniform> fp: FP;
var<workgroup> y: array<f32, 4096>;
var<workgroup> red: array<f32, 256>;

fn wsum(v: f32, lid: u32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
        st = st >> 1u;
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

// Final layer on IMAGE rows only: LayerNorm(eps, no affine)·scale, then
// Linear(h → pd) + b, all f32. Workgroup = one output row; the output is
// compacted to [batch][n_img][pd].
@compute @workgroup_size(256)
fn zi_final(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let bi = wid.x / fp.n_img;
    let r = wid.x % fp.n_img;
    let row = bi * fp.seg + r;
    let base = row * fp.h;
    var s = 0.0;
    for (var c = lid; c < fp.h; c = c + 256u) { s = s + x[base + c]; }
    let mean = wsum(s, lid) / f32(fp.h);
    var v = 0.0;
    for (var c = lid; c < fp.h; c = c + 256u) { let d = x[base + c] - mean; v = v + d * d; }
    let rs = inverseSqrt(wsum(v, lid) / f32(fp.h) + fp.eps);
    for (var c = lid; c < fp.h; c = c + 256u) { y[c] = (x[base + c] - mean) * rs * fsc[c]; }
    workgroupBarrier();
    // pd outputs × 4 partial sums each (pd ≤ 64).
    let o = lid & 63u;
    let part = lid >> 6u;
    var acc = 0.0;
    if (o < fp.pd) {
        let q = fp.h / 4u;
        for (var k = part * q; k < part * q + q; k = k + 1u) { acc = acc + y[k] * w[o * fp.h + k]; }
    }
    red[lid] = acc;
    workgroupBarrier();
    if (lid < 64u && lid < fp.pd) {
        outp[wid.x * fp.pd + lid] = red[lid] + red[lid + 64u] + red[lid + 128u] + red[lid + 192u] + b[lid];
    }
}
"#;

const FILL_SRC: &str = r#"
struct FlP { n: u32, seed: u32, amp: f32, mode: u32 };
@group(0) @binding(0) var<storage, read_write> dst: array<u32>;
@group(0) @binding(1) var<uniform> fl: FlP;

fn hash(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u); x = x * 0x7feb352du;
    x = x ^ (x >> 15u); x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}
fn uni(i: u32) -> f32 { return f32(hash(i) >> 8u) * (2.0 / 16777216.0) - 1.0; }

// Synthetic weights on the device: mode 0 = u32 words of two f16 in
// [-amp, amp], mode 1 = f32 words in [-amp, amp].
@compute @workgroup_size(256)
fn zi_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    if (i >= fl.n) { return; }
    let s = fl.seed * 0x9e3779b9u;
    if (fl.mode == 0u) {
        dst[i] = pack2x16float(vec2<f32>(uni(2u * i ^ s), uni((2u * i + 1u) ^ s)) * fl.amp);
    } else {
        dst[i] = bitcast<u32>(uni(i ^ s) * fl.amp);
    }
}
"#;

/// Pure-MMA ceiling: fragments loaded once, 8 independent accumulator
/// chains per subgroup, no global traffic inside the loop.
fn peak_src(acc16: bool) -> String {
    let cty = if acc16 { "f16" } else { "f32" };
    let mut s = String::from(
        "enable f16;\nenable wgpu_cooperative_matrix;\ndiagnostic(off, derivative_uniformity);\n",
    );
    s += &format!(
        "struct PP {{ iters: u32, a: u32, b: u32, c: u32 }};\n@group(0) @binding(0) var<storage, read_write> outp: array<{cty}>;\n@group(0) @binding(1) var<uniform> pp: PP;\nvar<workgroup> sa: array<f16, 512>;\n"
    );
    s += "@compute @workgroup_size(256)\nfn zi_peak(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_id) sg: u32) {\n";
    s += "  sa[tid] = f16(f32(tid % 7u) * 0.001); sa[tid + 256u] = f16(f32(tid % 5u) * 0.001);\n  workgroupBarrier();\n";
    s += "  let a = coopLoadT<coop_mat16x16<f16, A>>(&sa[0], 16u);\n  let b = coopLoad<coop_mat16x16<f16, B>>(&sa[256], 16u);\n";
    for j in 0..8 {
        s += &format!("  var c{j}: coop_mat16x16<{cty}, C>;\n");
    }
    s += "  for (var i = 0u; i < pp.iters; i = i + 1u) {\n";
    for j in 0..8 {
        s += &format!("    c{j} = coopMultiplyAdd(a, b, c{j});\n");
    }
    s += "  }\n";
    s += "  let o = (wid.x * 8u + sg) * 8u * 256u;\n";
    for j in 0..8 {
        s += &format!("  coopStoreT(c{j}, &outp[o + {}u], 16u);\n", j * 256);
    }
    s += "}\n";
    s
}

// ════════════════════════════════════════════════════════════════════
// Small host helpers.
// ════════════════════════════════════════════════════════════════════

fn sbuf(c: &Ctx, bytes: u64, label: &str) -> wgpu::Buffer {
    c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn sbuf_init(c: &Ctx, data: &[u8], label: &str) -> wgpu::Buffer {
    let b = sbuf(c, data.len() as u64, label);
    c.queue.write_buffer(&b, 0, data);
    b
}

fn ubuf(c: &Ctx, words: &[u32]) -> wgpu::Buffer {
    c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zi_u"),
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn bg(c: &Ctx, pipe: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<_> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| super::bind_buf(i as u32, b))
        .collect();
    c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zi_bg"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn wait(c: &Ctx) {
    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
}

/// Read a device buffer back (blocking).
fn read_bytes(c: &Ctx, src: &wgpu::Buffer, bytes: u64) -> Option<Vec<u8>> {
    let st = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zi_rb"),
        size: bytes.next_multiple_of(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = c.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, 0, &st, 0, bytes.next_multiple_of(4));
    c.queue.submit(Some(enc.finish()));
    let sl = st.slice(..);
    sl.map_async(wgpu::MapMode::Read, |_| {});
    wait(c);
    let v = sl.get_mapped_range().ok()?.to_vec();
    st.unmap();
    Some(v[..bytes as usize].to_vec())
}

fn fill(c: &Ctx, dst: &wgpu::Buffer, words: u64, seed: u32, amp: f32, f32_mode: bool) -> Option<()> {
    let pipe = pipeline(c, "zi_fill", FILL_SRC, "zi_fill")?;
    let u = ubuf(c, &[words as u32, seed, amp.to_bits(), f32_mode as u32]);
    let b = bg(c, &pipe, &[dst, &u]);
    let mut enc = c.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &b, &[]);
        let wgs = (words as u32).div_ceil(256);
        pass.dispatch_workgroups(wgs.min(65535), wgs.div_ceil(65535), 1);
    }
    c.queue.submit(Some(enc.finish()));
    Some(())
}

/// Median seconds per repetition of `rec` (recorded `reps` times into one
/// pass, `rounds` submissions, after one warm-up submission).
fn time_pass(c: &Ctx, reps: usize, rounds: usize, rec: &dyn Fn(&mut wgpu::ComputePass)) -> f64 {
    let run = || {
        let mut enc = c.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for _ in 0..reps {
                rec(&mut pass);
            }
        }
        let t = std::time::Instant::now();
        c.queue.submit(Some(enc.finish()));
        wait(c);
        t.elapsed().as_secs_f64() / reps as f64
    };
    run();
    let mut v: Vec<f64> = (0..rounds.max(1)).map(|_| run()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_enc(c: &Ctx, reps: usize, rounds: usize, rec: &dyn Fn(&mut wgpu::CommandEncoder)) -> f64 {
    let run = || {
        let mut enc = c.device.create_command_encoder(&Default::default());
        for _ in 0..reps {
            rec(&mut enc);
        }
        let cb = super::finish_enc(enc);
        let t = std::time::Instant::now();
        c.queue.submit(Some(cb));
        wait(c);
        t.elapsed().as_secs_f64() / reps as f64
    };
    run();
    let mut v: Vec<f64> = (0..rounds.max(1)).map(|_| run()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Round `m` up so every kernel's reads past the last real row stay inside
/// the buffer: the GEMM tile (128) plus one flash query block (128).
pub fn pad_rows(m: usize) -> usize {
    m.div_ceil(128) * 128 + 128
}

// ════════════════════════════════════════════════════════════════════
// Bench / test API (used by examples/zimage_gemmbench.rs and
// tests/zimage_wgpu.rs). Doc-hidden, not part of the engine surface.
// ════════════════════════════════════════════════════════════════════

#[doc(hidden)]
pub mod bench {
    use super::*;

    /// Adapter, driver, coop shapes, limits.
    pub fn info() -> Option<String> {
        let c = super::super::ctx()?;
        let i = c._adapter.get_info();
        let l = c.device.limits();
        let mut s = format!(
            "adapter {} / {:?} / driver {} {}\ncoop active: {}  f16: {}  subgroup {}..{}\nmax wg storage {} B, max storage binding {} MB, max buffer {} MB\n",
            i.name,
            i.backend,
            i.driver,
            i.driver_info,
            super::super::coop_matrix_active(),
            c.device.features().contains(wgpu::Features::SHADER_F16),
            i.subgroup_min_size,
            i.subgroup_max_size,
            l.max_compute_workgroup_storage_size,
            l.max_storage_buffer_binding_size >> 20,
            l.max_buffer_size >> 20,
        );
        for p in c._adapter.cooperative_matrix_properties() {
            s += &format!(
                "  coop shape {}x{}x{} ab {:?} c {:?} sat {}\n",
                p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type, p.saturating_accumulation
            );
        }
        s += &format!("zi fast path usable: {}\n", zctx().is_some());
        Some(s)
    }

    /// Tensor-core ceiling in TFLOPS (f16 operands, f32 or f16 accumulate).
    pub fn peak(acc16: bool) -> Option<f64> {
        let c = zctx()?;
        let src = peak_src(acc16);
        let pipe = pipeline(c, if acc16 { "zi_peak16" } else { "zi_peak32" }, &src, "zi_peak")?;
        let wgs = 82 * 8u32;
        let iters = 4096u32;
        let out = sbuf(c, (wgs as u64) * 8 * 8 * 256 * 4, "peak");
        let u = ubuf(c, &[iters, 0, 0, 0]);
        let b = bg(c, &pipe, &[&out, &u]);
        let t = time_pass(c, 4, 5, &|pass| {
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &b, &[]);
            pass.dispatch_workgroups(wgs, 1, 1);
        });
        let flops = wgs as f64 * 8.0 * 8.0 * iters as f64 * 2.0 * 4096.0;
        Some(flops / t / 1e12)
    }

    /// The parent's existing GEMMs at M×K×N (activation f32 [M][K]):
    /// `arm` = "scalar" (q4tp_mm), "coop_q4" (q4tp_mm_coop, in-kernel
    /// dequant), "coop_f16" (q4tp_mm_coop_f16 on an f16 plane).
    /// `q4tp` = the Q4TP payload for the first two arms. Seconds per GEMM.
    pub fn existing(arm: &str, m: usize, k: usize, n: usize, q4tp: &[u8]) -> Option<f64> {
        let c = super::super::ctx()?;
        let x = sbuf(c, (m * k * 4) as u64, "x");
        fill(c, &x, (m * k) as u64, 3, 1.0, true)?;
        let y = sbuf(c, (m * n * 4) as u64, "y");
        let (pipe, w) = match arm {
            "scalar" => (&c.q4tp_mm, sbuf_init(c, q4tp, "w")),
            "coop_q4" => (c.q4tp_mm_coop.as_ref()?, sbuf_init(c, q4tp, "w")),
            "coop_f16" => {
                let w = sbuf(c, (n * k * 2) as u64, "plane");
                fill(c, &w, (n * k / 2) as u64, 5, 0.02, false)?;
                (c.q4tp_mm_coop_f16.as_ref()?, w)
            }
            _ => return None,
        };
        wait(c);
        let reps = if m * n * k > 50_000_000_000 { 3 } else { 6 };
        Some(time_enc(c, reps, 3, &|enc| {
            super::super::encode_q4_tile_mm_full(c, enc, pipe, &w, &x, &y, n, k, m, 0.0, None)
        }))
    }

    /// Run one `zi_mm` on host data (f16 bit patterns) and return the
    /// output as f32 (`[m][n]`, or `[m][n/2]` for SwiGLU).
    pub fn mm_run(cfg: MmCfg, m: usize, k: usize, n: usize, act: &[u16], plane: &[u16]) -> Option<Vec<f32>> {
        let c = zctx()?;
        let mp = pad_rows(m);
        let mut a = act.to_vec();
        a.resize(mp * k, 0);
        let ab = sbuf_init(c, bytemuck::cast_slice(&a), "act");
        let wb = sbuf_init(c, bytemuck::cast_slice(plane), "plane");
        let ncol = if cfg.epi == Epi::SwiGlu { n / 2 } else { n };
        let ob = sbuf(c, (mp * ncol * 4) as u64, "out");
        let args = MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: ncol as u32, ocol: 0, arow: 0, oscale: 1.0 };
        let call = mm_call(c, cfg, &args, &wb, &ab, &ob)?;
        let mut enc = c.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            call.record(&mut pass);
        }
        c.queue.submit(Some(enc.finish()));
        let raw = read_bytes(c, &ob, (m * ncol * if cfg.epi == Epi::F32 { 4 } else { 2 }) as u64)?;
        Some(match cfg.epi {
            Epi::F32 => bytemuck::cast_slice::<u8, f32>(&raw).to_vec(),
            _ => bytemuck::cast_slice::<u8, u16>(&raw)
                .iter()
                .map(|&h| cortiq_core::quant::f16_to_f32(h))
                .collect(),
        })
    }

    /// Seconds per `zi_mm` at M×K×N on device-filled data.
    pub fn mm_time(cfg: MmCfg, m: usize, k: usize, n: usize) -> Option<f64> {
        let c = zctx()?;
        let mp = pad_rows(m);
        let ab = sbuf(c, (mp * k * 2) as u64, "act");
        fill(c, &ab, (mp * k / 2) as u64, 7, 1.0, false)?;
        let wb = sbuf(c, (n * k * 2) as u64, "plane");
        fill(c, &wb, (n * k / 2) as u64, 9, 0.02, false)?;
        let ncol = if cfg.epi == Epi::SwiGlu { n / 2 } else { n };
        let ob = sbuf(c, (mp * ncol * 4) as u64, "out");
        let args = MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: ncol as u32, ocol: 0, arow: 0, oscale: 1.0 };
        let call = mm_call(c, cfg, &args, &wb, &ab, &ob)?;
        wait(c);
        let reps = if m * n * k > 50_000_000_000 { 4 } else { 10 };
        Some(time_pass(c, reps, 5, &|pass| call.record(pass)))
    }

    /// Flash attention over a packed qkv panel `[m][3·nh·128]` (f16 bits),
    /// segments `(offset, len)`; returns `[m][nh·128]` f32.
    pub fn flash_run(f: FlashCfg, nh: usize, qkv: &[u16], segs: &[(usize, usize)]) -> Option<Vec<f32>> {
        let c = zctx()?;
        let ld = 3 * nh * 128;
        let m = qkv.len() / ld;
        let mp = pad_rows(m);
        let mut q = qkv.to_vec();
        q.resize(mp * ld, 0);
        let qb = sbuf_init(c, bytemuck::cast_slice(&q), "qkv");
        let ob = sbuf(c, (mp * nh * 128 * 2) as u64, "att");
        let calls = flash_calls(c, f, nh, &qb, &ob, segs)?;
        let mut enc = c.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for cl in &calls {
                cl.record(&mut pass);
            }
        }
        c.queue.submit(Some(enc.finish()));
        let raw = read_bytes(c, &ob, (m * nh * 128 * 2) as u64)?;
        Some(
            bytemuck::cast_slice::<u8, u16>(&raw)
                .iter()
                .map(|&h| cortiq_core::quant::f16_to_f32(h))
                .collect(),
        )
    }

    /// Seconds per full attention (all segments, all heads).
    pub fn flash_time(f: FlashCfg, nh: usize, segs: &[(usize, usize)]) -> Option<f64> {
        let c = zctx()?;
        let ld = 3 * nh * 128;
        let m: usize = segs.iter().map(|s| s.0 + s.1).max()?;
        let mp = pad_rows(m);
        let qb = sbuf(c, (mp * ld * 2) as u64, "qkv");
        fill(c, &qb, (mp * ld / 2) as u64, 11, 1.0, false)?;
        let ob = sbuf(c, (mp * nh * 128 * 2) as u64, "att");
        let calls = flash_calls(c, f, nh, &qb, &ob, segs)?;
        wait(c);
        Some(time_pass(c, 5, 5, &|pass| {
            for cl in &calls {
                cl.record(pass);
            }
        }))
    }
}

/// One prebuilt dispatch of any of the small kernels.
struct Call {
    pipe: Arc<wgpu::ComputePipeline>,
    bg: wgpu::BindGroup,
    grid: (u32, u32, u32),
}

impl Call {
    fn record(&self, pass: &mut wgpu::ComputePass) {
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.dispatch_workgroups(self.grid.0, self.grid.1, self.grid.2);
    }
}

fn flash_calls(
    c: &Ctx,
    f: FlashCfg,
    nh: usize,
    qkv: &wgpu::Buffer,
    out: &wgpu::Buffer,
    segs: &[(usize, usize)],
) -> Option<Vec<Call>> {
    if f.shared_bytes() > c.device.limits().max_compute_workgroup_storage_size {
        return None;
    }
    let pipe = pipeline(c, &f.key(), &flash_src(f), "zi_flash")?;
    let hsz = nh * 128;
    let scl = (1.0f32 / (128f32).sqrt()) * std::f32::consts::LOG2_E;
    let mut v = Vec::new();
    for &(off, len) in segs {
        if len % f.bc as usize != 0 {
            return None;
        }
        let u = ubuf(
            c,
            &[
                off as u32,
                len as u32,
                (3 * hsz) as u32,
                hsz as u32,
                (2 * hsz) as u32,
                hsz as u32,
                scl.to_bits(),
                nh as u32,
            ],
        );
        let b = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zi_flash"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                super::bind_buf(0, qkv),
                super::bind_buf(1, qkv),
                super::bind_buf(2, out),
                super::bind_buf(3, &u),
            ],
        });
        v.push(Call {
            pipe: pipe.clone(),
            bg: b,
            grid: ((len as u32).div_ceil(f.nw * 16), nh as u32, 1),
        });
    }
    Some(v)
}

// ════════════════════════════════════════════════════════════════════
// The resident chain (plan S1–S5 machinery): planes, sequence state,
// prebuilt dispatch lists. The integration package builds these from a
// container in `prepare` and records them in `step`.
// ════════════════════════════════════════════════════════════════════

/// Z-Image dimensions the chain is built for.
#[derive(Clone, Copy, Debug)]
pub struct ZDims {
    pub h: usize,
    pub nh: usize,
    pub inter: usize,
    pub eps: f32,
    pub final_eps: f32,
    /// Patch vector width (64).
    pub pd: usize,
}

impl ZDims {
    pub const TURBO: ZDims = ZDims { h: 3840, nh: 30, inter: 10240, eps: 1e-5, final_eps: 1e-6, pd: 64 };
    pub fn from_geom(g: &ZGeom) -> Option<ZDims> {
        (g.hd == 128 && g.hidden == g.nh * 128 && g.hidden % 128 == 0 && g.inter % 128 == 0).then_some(ZDims {
            h: g.hidden,
            nh: g.nh,
            inter: g.inter,
            eps: g.eps,
            final_eps: g.final_eps,
            pd: g.patch_dim,
        })
    }
}

/// One block's device weights: row-major f16 planes (the `q4tp_dq_f16`
/// layout: two halves per u32) and f32 norm vectors.
pub struct ZBlockDev {
    /// `[3h][h]`: to_q rows, then to_k, then to_v.
    pub qkv: wgpu::Buffer,
    /// `[h][h]`: to_out.0.
    pub o: wgpu::Buffer,
    /// `[2·inter][h]`: w1 (gate) and w3 (up) interleaved in 16-row panels,
    /// plane row `32q + t` = gate row `16q + t` (t < 16) or up row
    /// `16q + t − 16` — what the SwiGLU epilogue expects.
    pub w13: wgpu::Buffer,
    /// `[h][inter]`: w2.
    pub w2: wgpu::Buffer,
    pub norm1: wgpu::Buffer,
    pub norm2: wgpu::Buffer,
    pub ffn_norm1: wgpu::Buffer,
    pub ffn_norm2: wgpu::Buffer,
    pub norm_q: wgpu::Buffer,
    pub norm_k: wgpu::Buffer,
}

/// Plane row of the interleaved w1‖w3 plane that holds gate row `j`
/// (`up = false`) or up row `j` (`up = true`).
pub fn w13_plane_row(j: usize, up: bool) -> usize {
    32 * (j / 16) + (j % 16) + if up { 16 } else { 0 }
}

impl ZBlockDev {
    /// Random weights generated on the device (synthetic benchmarks).
    pub fn synthetic(d: &ZDims, seed: u32) -> Option<ZBlockDev> {
        let c = zctx()?;
        let (h, i) = (d.h as u64, d.inter as u64);
        let mk = |rows: u64, cols: u64, s: u32| -> Option<wgpu::Buffer> {
            let b = sbuf(c, rows * cols * 2, "zi_plane");
            fill(c, &b, rows * cols / 2, s, 1.7 / (cols as f32).sqrt(), false)?;
            Some(b)
        };
        let norm = |n: usize, s: u32| -> wgpu::Buffer {
            let v: Vec<f32> = (0..n).map(|j| 1.0 + 0.1 * (((j as u32 ^ s) % 7) as f32 - 3.0) / 3.0).collect();
            sbuf_init(c, bytemuck::cast_slice(&v), "zi_norm")
        };
        Some(ZBlockDev {
            qkv: mk(3 * h, h, seed * 8 + 1)?,
            o: mk(h, h, seed * 8 + 2)?,
            w13: mk(2 * i, h, seed * 8 + 3)?,
            w2: mk(h, i, seed * 8 + 4)?,
            norm1: norm(d.h, seed + 1),
            norm2: norm(d.h, seed + 2),
            ffn_norm1: norm(d.h, seed + 3),
            ffn_norm2: norm(d.h, seed + 4),
            norm_q: norm(128, seed + 5),
            norm_k: norm(128, seed + 6),
        })
    }

    /// Planes from host f16 bit patterns (tests; the integration package
    /// adds the codec paths: q4tp/q8 via the parent's dequant kernels,
    /// F16 = direct upload, Bf16 = convert). `w1`, `w3` are `[inter][h]`
    /// and are interleaved here. `norms` = norm1, norm2, ffn_norm1,
    /// ffn_norm2, norm_q, norm_k.
    #[allow(clippy::too_many_arguments)]
    pub fn from_host(
        d: &ZDims,
        wq: &[u16],
        wk: &[u16],
        wv: &[u16],
        wo: &[u16],
        w1: &[u16],
        w3: &[u16],
        w2: &[u16],
        norms: [&[f32]; 6],
    ) -> Option<ZBlockDev> {
        let c = zctx()?;
        let h = d.h;
        let mut qkv = Vec::with_capacity(3 * h * h);
        qkv.extend_from_slice(wq);
        qkv.extend_from_slice(wk);
        qkv.extend_from_slice(wv);
        let mut w13 = vec![0u16; 2 * d.inter * h];
        for j in 0..d.inter {
            let g = w13_plane_row(j, false);
            let u = w13_plane_row(j, true);
            w13[g * h..(g + 1) * h].copy_from_slice(&w1[j * h..(j + 1) * h]);
            w13[u * h..(u + 1) * h].copy_from_slice(&w3[j * h..(j + 1) * h]);
        }
        let up = |v: &[u16]| sbuf_init(c, bytemuck::cast_slice(v), "zi_plane");
        let upf = |v: &[f32]| sbuf_init(c, bytemuck::cast_slice(v), "zi_norm");
        Some(ZBlockDev {
            qkv: up(&qkv),
            o: up(wo),
            w13: up(&w13),
            w2: up(w2),
            norm1: upf(norms[0]),
            norm2: upf(norms[1]),
            ffn_norm1: upf(norms[2]),
            ffn_norm2: upf(norms[3]),
            norm_q: upf(norms[4]),
            norm_k: upf(norms[5]),
        })
    }
}

/// f16 plane rows of one weight tensor, written into `dst`: tensor row
/// panel `p` (rows 16p..16p+16) lands at plane row `panel_row(p)`.
/// F16 = bytes as stored; Bf16/F32 converted on the host; Q4TiledP
/// dequantized on the device by the parent's `q4tp_dq_f16` (the plane
/// layout the parent's coop GEMM eats). Other codecs → `None`.
fn tensor_to_plane(
    c: &Ctx,
    model: &Arc<CmfModel>,
    idx: usize,
    rows: usize,
    cols: usize,
    dst: &wgpu::Buffer,
    panel_row: &dyn Fn(usize) -> usize,
) -> Option<()> {
    use cortiq_core::TensorDtype as T;
    let e = model.tensors.get(idx)?;
    if e.shape.len() != 2 || e.shape[0] != rows || e.shape[1] != cols || rows % 16 != 0 {
        return None;
    }
    let bytes = model.entry_bytes(e);
    let panel_bytes = (16 * cols * 2) as u64;
    let contiguous = (0..rows / 16).all(|p| panel_row(p) == panel_row(0) + 16 * p);
    match e.dtype {
        T::F16 | T::Bf16 | T::F32 => {
            let h: std::borrow::Cow<[u8]> = match e.dtype {
                T::F16 => std::borrow::Cow::Borrowed(&bytes[..rows * cols * 2]),
                T::Bf16 => std::borrow::Cow::Owned(
                    bytes[..rows * cols * 2]
                        .chunks_exact(2)
                        .flat_map(|b| {
                            let f = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
                            cortiq_core::quant::f32_to_f16(f).to_le_bytes()
                        })
                        .collect(),
                ),
                _ => std::borrow::Cow::Owned(
                    bytes[..rows * cols * 4]
                        .chunks_exact(4)
                        .flat_map(|b| cortiq_core::quant::f32_to_f16(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes())
                        .collect(),
                ),
            };
            if contiguous {
                c.queue.write_buffer(dst, (panel_row(0) * cols * 2) as u64, &h);
            } else {
                for p in 0..rows / 16 {
                    let src = &h[p * panel_bytes as usize..(p + 1) * panel_bytes as usize];
                    c.queue.write_buffer(dst, (panel_row(p) * cols * 2) as u64, src);
                }
            }
            Some(())
        }
        T::Q4TiledP => {
            let dq = c.q4tp_dq_f16.as_ref()?;
            let need = cortiq_core::quant::expected_nbytes(T::Q4TiledP, &[rows, cols])?;
            if bytes.len() < need {
                return None;
            }
            let mut payload = bytes[..need].to_vec();
            payload.resize(need.next_multiple_of(4), 0);
            let src = sbuf_init(c, &payload, "zi_q4tp");
            let tmp = sbuf(c, (rows * cols * 2) as u64, "zi_dq");
            let u = ubuf(c, &[cols as u32, rows as u32, 0, 0]);
            let b = bg(c, dq, &[&src, &tmp, &u]);
            let mut enc = c.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(dq);
                pass.set_bind_group(0, &b, &[]);
                let wgs = ((rows * cols / 2) as u32).div_ceil(256);
                pass.dispatch_workgroups(wgs.min(65535), wgs.div_ceil(65535), 1);
            }
            if contiguous {
                enc.copy_buffer_to_buffer(&tmp, 0, dst, (panel_row(0) * cols * 2) as u64, (rows * cols * 2) as u64);
            } else {
                for p in 0..rows / 16 {
                    enc.copy_buffer_to_buffer(&tmp, p as u64 * panel_bytes, dst, (panel_row(p) * cols * 2) as u64, panel_bytes);
                }
            }
            c.queue.submit(Some(enc.finish()));
            Some(())
        }
        _ => None,
    }
}

impl ZBlockDev {
    /// Planes of one block straight from the container (`ZBlockRef`
    /// indices, diffusers names). F16 / Bf16 / F32 / Q4TiledP; any other
    /// codec → `None` (the caller declines and the CPU path runs). The
    /// integration package adds q8_row / q8_2f here via the parent's
    /// `q8_dq_f16` when the WP4 codec policy picks them.
    pub fn from_model(model: &Arc<CmfModel>, d: &ZDims, r: &ZBlockRef) -> Option<ZBlockDev> {
        let c = zctx()?;
        let (h, i) = (d.h, d.inter);
        if r.norm1.len() < h || r.norm2.len() < h || r.ffn_norm1.len() < h || r.ffn_norm2.len() < h
            || r.norm_q.len() < 128 || r.norm_k.len() < 128
        {
            return None;
        }
        let qkv = sbuf(c, (3 * h * h * 2) as u64, "zi_plane_qkv");
        for (k, idx) in [r.wq, r.wk, r.wv].into_iter().enumerate() {
            tensor_to_plane(c, model, idx, h, h, &qkv, &|p| k * h + 16 * p)?;
        }
        let o = sbuf(c, (h * h * 2) as u64, "zi_plane_o");
        tensor_to_plane(c, model, r.wo, h, h, &o, &|p| 16 * p)?;
        let w13 = sbuf(c, (2 * i * h * 2) as u64, "zi_plane_w13");
        tensor_to_plane(c, model, r.w1, i, h, &w13, &|p| w13_plane_row(16 * p, false))?;
        tensor_to_plane(c, model, r.w3, i, h, &w13, &|p| w13_plane_row(16 * p, true))?;
        let w2 = sbuf(c, (h * i * 2) as u64, "zi_plane_w2");
        tensor_to_plane(c, model, r.w2, h, i, &w2, &|p| 16 * p)?;
        let upf = |v: &[f32]| sbuf_init(c, bytemuck::cast_slice(v), "zi_norm");
        let b = ZBlockDev {
            qkv,
            o,
            w13,
            w2,
            norm1: upf(&r.norm1[..h]),
            norm2: upf(&r.norm2[..h]),
            ffn_norm1: upf(&r.ffn_norm1[..h]),
            ffn_norm2: upf(&r.ffn_norm2[..h]),
            norm_q: upf(&r.norm_q[..128]),
            norm_k: upf(&r.norm_k[..128]),
        };
        // Flush the staged writes and bound their memory block by block.
        c.queue.submit(std::iter::empty());
        wait(c);
        Some(b)
    }
}

/// Activation state of one stacked sequence layout (all batch items).
/// Every buffer has `mp = pad_rows(m)` rows so the unchecked kernels'
/// tile over-reads stay inside it.
pub struct ZSeq {
    pub segs: Vec<(usize, usize)>,
    pub m: usize,
    pub mp: usize,
    pub x: wgpu::Buffer,
    xn: wgpu::Buffer,
    qkv: wgpu::Buffer,
    att: wgpu::Buffer,
    br: wgpu::Buffer,
    hid: wgpu::Buffer,
    pub rope_c: wgpu::Buffer,
    pub rope_s: wgpu::Buffer,
}

impl ZSeq {
    /// `segs` = (row offset, rows) per batch item; lengths multiples of 32.
    pub fn new(d: &ZDims, segs: &[(usize, usize)]) -> Option<ZSeq> {
        let c = zctx()?;
        let m = segs.iter().map(|s| s.0 + s.1).max()?;
        if segs.iter().any(|s| s.1 % 32 != 0) {
            return None;
        }
        let mp = pad_rows(m) as u64;
        let (h, i) = (d.h as u64, d.inter as u64);
        Some(ZSeq {
            segs: segs.to_vec(),
            m,
            mp: mp as usize,
            x: sbuf(c, mp * h * 4, "zi_x"),
            xn: sbuf(c, mp * h * 2, "zi_xn"),
            qkv: sbuf(c, mp * 3 * h * 2, "zi_qkv"),
            att: sbuf(c, mp * h * 2, "zi_att"),
            br: sbuf(c, mp * h * 4, "zi_br"),
            hid: sbuf(c, mp * i * 2, "zi_hid"),
            rope_c: sbuf(c, mp * 64 * 4, "zi_rc"),
            rope_s: sbuf(c, mp * 64 * 4, "zi_rs"),
        })
    }

    /// Upload the RoPE table rows `[m][64]` (cos, sin).
    pub fn set_rope(&self, cos: &[f32], sin: &[f32]) {
        if let Some(c) = zctx() {
            c.queue.write_buffer(&self.rope_c, 0, bytemuck::cast_slice(cos));
            c.queue.write_buffer(&self.rope_s, 0, bytemuck::cast_slice(sin));
        }
    }

    /// Upload / read the residual stream `[m][h]` f32 (tests, and the
    /// context refiner's in/out).
    pub fn write_x(&self, x: &[f32]) {
        if let Some(c) = zctx() {
            c.queue.write_buffer(&self.x, 0, bytemuck::cast_slice(x));
        }
    }
    pub fn read_x(&self, d: &ZDims) -> Option<Vec<f32>> {
        let c = zctx()?;
        let raw = read_bytes(c, &self.x, (self.m * d.h * 4) as u64)?;
        Some(bytemuck::cast_slice(&raw).to_vec())
    }
}

/// Kernel class of a recorded dispatch (for the per-class profile).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Class {
    MmQkv,
    Flash,
    MmO,
    MmW13,
    MmW2,
    Rows,
    Io,
}

impl Class {
    pub const ALL: [Class; 7] =
        [Class::MmQkv, Class::Flash, Class::MmO, Class::MmW13, Class::MmW2, Class::Rows, Class::Io];
}

enum Rec {
    Mm(MmCall),
    K(Call),
}

/// A prebuilt list of dispatches (one or more blocks).
#[derive(Default)]
pub struct ZCalls {
    list: Vec<(Class, Rec)>,
}

impl ZCalls {
    pub fn record(&self, pass: &mut wgpu::ComputePass, only: Option<Class>) {
        for (cl, r) in &self.list {
            if only.is_some_and(|o| o != *cl) {
                continue;
            }
            match r {
                Rec::Mm(m) => m.record(pass),
                Rec::K(k) => k.record(pass),
            }
        }
    }
    pub fn len(&self) -> usize {
        self.list.len()
    }
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
    fn push_mm(&mut self, cl: Class, m: MmCall) {
        self.list.push((cl, Rec::Mm(m)));
    }
    fn push(&mut self, cl: Class, k: Call) {
        self.list.push((cl, Rec::K(k)));
    }
    pub fn extend(&mut self, o: ZCalls) {
        self.list.extend(o.list);
    }
    /// Record into a fresh encoder, submit, wait (tests / context refiner).
    pub fn run(&self) -> Option<()> {
        let c = zctx()?;
        let mut enc = c.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            self.record(&mut pass, None);
        }
        c.queue.submit(Some(enc.finish()));
        wait(c);
        Some(())
    }
}

/// A device buffer holding `data` (mods, caption rows …) — test helper.
pub fn upload_f32(data: &[f32]) -> Option<wgpu::Buffer> {
    let c = zctx()?;
    Some(sbuf_init(c, bytemuck::cast_slice(data), "zi_host"))
}

/// Offsets of one block's raw modulation chunks inside the mods buffer
/// (`[block][scale_msa, gate_msa, scale_mlp, gate_mlp][h]`, f32 words).
fn mod_off(d: &ZDims, bi: usize, chunk: usize) -> u32 {
    ((bi * 4 + chunk) * d.h) as u32
}

#[allow(clippy::too_many_arguments)]
fn rowop_call(
    c: &Ctx,
    d: &ZDims,
    seq: &ZSeq,
    mode: u32,
    wpost: &wgpu::Buffer,
    wpre: &wgpu::Buffer,
    mods: &wgpu::Buffer,
    g_off: u32,
    s_off: u32,
) -> Option<Call> {
    let pipe = pipeline(c, "zi_rowop", ROWOP_SRC, "zi_rowop")?;
    let u = ubuf(
        c,
        &[d.h as u32, mode, g_off, s_off, d.eps.to_bits(), d.eps.to_bits(), 1f32.to_bits(), 0],
    );
    let b = bg(c, &pipe, &[&seq.br, &seq.x, &seq.xn, wpost, wpre, mods, &u]);
    Some(Call { pipe, bg: b, grid: ((seq.m as u32).min(65535), (seq.m as u32).div_ceil(65535), 1) })
}

/// Tiles per site (defaults from the S0/S4 sweep; env overrides).
#[derive(Clone, Copy, Debug)]
pub struct ZTiles {
    pub qkv: MmCfg,
    pub o: MmCfg,
    pub w13: MmCfg,
    pub w2: MmCfg,
    pub flash: FlashCfg,
}

impl Default for ZTiles {
    fn default() -> Self {
        ZTiles {
            qkv: default_cfg(Epi::F16),
            o: default_cfg(Epi::F32),
            w13: default_cfg(Epi::SwiGlu),
            w2: default_cfg(Epi::F32),
            flash: default_flash(),
        }
    }
}

/// The dispatches of one transformer block on `seq`.
///
/// `bi` = this block's index in the mods buffer; `first` = also compute
/// this block's own pre-norm from `seq.x` (the chain entry); `next` = the
/// following block (its norm1 and scale_msa index) whose pre-norm the last
/// row op writes, or `None` (the last block: gated residual only).
/// `modulated = false` is the context refiner (scale 0, gate 1).
#[allow(clippy::too_many_arguments)]
pub fn block_calls(
    d: &ZDims,
    t: &ZTiles,
    seq: &ZSeq,
    blk: &ZBlockDev,
    mods: &wgpu::Buffer,
    bi: usize,
    first: bool,
    next: Option<(&ZBlockDev, usize)>,
    modulated: bool,
) -> Option<ZCalls> {
    let c = zctx()?;
    let (h, i, m) = (d.h as u32, d.inter as u32, seq.m as u32);
    let gm = if modulated { ROW_GATE_MOD } else { 0 };
    let sm = if modulated { ROW_SCALE_MOD } else { 0 };
    let mut v = ZCalls::default();
    if first {
        v.push(
            Class::Rows,
            rowop_call(c, d, seq, ROW_PRE | sm, &blk.norm1, &blk.norm1, mods, 0, mod_off(d, bi, 0))?,
        );
    }
    let a = |n: u32, k: u32, ldo: u32| MmArgs { m, n, k, ldo, ocol: 0, arow: 0, oscale: 1.0 };
    v.push_mm(Class::MmQkv, mm_call(c, t.qkv, &a(3 * h, h, 3 * h), &blk.qkv, &seq.xn, &seq.qkv)?);
    {
        let pipe = pipeline(c, "zi_qkrope", QKROPE_SRC, "zi_qkrope")?;
        let u = ubuf(c, &[3 * h, d.nh as u32, d.eps.to_bits(), 0]);
        let b = bg(c, &pipe, &[&seq.qkv, &blk.norm_q, &blk.norm_k, &seq.rope_c, &seq.rope_s, &u]);
        v.push(Class::Rows, Call { pipe, bg: b, grid: (2 * d.nh as u32, m.min(65535), m.div_ceil(65535)) });
    }
    for cl in flash_calls(c, t.flash, d.nh, &seq.qkv, &seq.att, &seq.segs)? {
        v.push(Class::Flash, cl);
    }
    v.push_mm(Class::MmO, mm_call(c, t.o, &a(h, h, h), &blk.o, &seq.att, &seq.br)?);
    v.push(
        Class::Rows,
        rowop_call(
            c,
            d,
            seq,
            ROW_GRES | gm | ROW_PRE | sm,
            &blk.norm2,
            &blk.ffn_norm1,
            mods,
            mod_off(d, bi, 1),
            mod_off(d, bi, 2),
        )?,
    );
    v.push_mm(Class::MmW13, mm_call(c, t.w13, &a(2 * i, h, i), &blk.w13, &seq.xn, &seq.hid)?);
    v.push_mm(Class::MmW2, mm_call(c, t.w2, &a(h, i, h), &blk.w2, &seq.hid, &seq.br)?);
    let (mode, wpre, s_off) = match next {
        Some((nb, nbi)) => (ROW_GRES | gm | ROW_PRE | sm, &nb.norm1, mod_off(d, nbi, 0)),
        None => (ROW_GRES | gm, &blk.ffn_norm2, 0),
    };
    v.push(Class::Rows, rowop_call(c, d, seq, mode, &blk.ffn_norm2, wpre, mods, mod_off(d, bi, 3), s_off)?);
    Some(v)
}

/// The per-step device program for one image (or a CFG pair): x_embed →
/// noise refiner (image rows) → [img, cap] assembly → 30 layers → final
/// layer on image rows. Built once per (prompt, resolution, batch); a
/// step then only writes `x_tok` and the step's mods and replays it.
pub struct ZStepDev {
    pub d: ZDims,
    pub n_img: usize,
    pub n_img_p: usize,
    pub n_cap_p: Vec<usize>,
    pub img: ZSeq,
    pub joint: ZSeq,
    pub mods: wgpu::Buffer,
    tok: wgpu::Buffer,
    cap: wgpu::Buffer,
    fscale: wgpu::Buffer,
    out: wgpu::Buffer,
    pre_a: ZCalls,
    joint_calls: ZCalls,
    fin: Vec<Call>,
    out_rows: usize,
}

/// Host-side small weights of the embed and final layers.
pub struct ZIo<'a> {
    pub x_emb_w: &'a [f32],
    pub x_emb_b: &'a [f32],
    pub x_pad: &'a [f32],
    pub final_w: &'a [f32],
    pub final_b: &'a [f32],
}

impl ZStepDev {
    /// `nr` = the 2 noise-refiner blocks, `layers` = the 30 main layers
    /// (mods indices 0,1 and 2..32). `n_cap_p[b]` = padded caption rows of
    /// batch item b (CFG: cond, uncond — may differ). `cap` = the refined
    /// captions stacked `[Σ n_cap_p][h]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d: ZDims,
        t: &ZTiles,
        nr: &[ZBlockDev],
        layers: &[ZBlockDev],
        io: &ZIo,
        n_img: usize,
        n_cap_p: &[usize],
        cap: &[f32],
    ) -> Option<ZStepDev> {
        let c = zctx()?;
        let b = n_cap_p.len();
        let n_img_p = n_img.div_ceil(32) * 32;
        let img_segs: Vec<(usize, usize)> = (0..b).map(|i| (i * n_img_p, n_img_p)).collect();
        let mut joint_segs = Vec::new();
        let mut off = 0;
        for &cp in n_cap_p {
            joint_segs.push((off, n_img_p + cp));
            off += n_img_p + cp;
        }
        let img = ZSeq::new(&d, &img_segs)?;
        let joint = ZSeq::new(&d, &joint_segs)?;
        let nblk = nr.len() + layers.len();
        let mods = sbuf(c, (nblk * 4 * d.h * 4) as u64, "zi_mods");
        let tok = sbuf(c, (img.mp * d.pd * 4) as u64, "zi_tok");
        let capb = sbuf_init(c, bytemuck::cast_slice(cap), "zi_cap");
        let fscale = sbuf(c, (d.h * 4) as u64, "zi_fscale");
        let out_rows = b * n_img;
        let out = sbuf(c, (out_rows * d.pd * 4) as u64, "zi_out");
        let ew = sbuf_init(c, bytemuck::cast_slice(io.x_emb_w), "zi_ew");
        let eb = sbuf_init(c, bytemuck::cast_slice(io.x_emb_b), "zi_eb");
        let ep = sbuf_init(c, bytemuck::cast_slice(io.x_pad), "zi_ep");
        let fw = sbuf_init(c, bytemuck::cast_slice(io.final_w), "zi_fw");
        let fb = sbuf_init(c, bytemuck::cast_slice(io.final_b), "zi_fb");
        let mut pre_a = ZCalls::default();
        {
            let pipe = pipeline(c, "zi_embed", EMBED_SRC, "zi_embed")?;
            let u = ubuf(c, &[d.h as u32, n_img as u32, n_img_p as u32, d.pd as u32]);
            let bgr = bg(c, &pipe, &[&tok, &ew, &eb, &ep, &img.x, &u]);
            pre_a.push(Class::Io, Call { pipe, bg: bgr, grid: (img.m as u32, 1, 1) });
        }
        for (k, blk) in nr.iter().enumerate() {
            let next = nr.get(k + 1).map(|nb| (nb, k + 1));
            pre_a.extend(block_calls(&d, t, &img, blk, &mods, k, k == 0, next, true)?);
        }
        let mut joint_calls = ZCalls::default();
        for (k, blk) in layers.iter().enumerate() {
            let bi = nr.len() + k;
            let next = layers.get(k + 1).map(|nb| (nb, bi + 1));
            joint_calls.extend(block_calls(&d, t, &joint, blk, &mods, bi, k == 0, next, true)?);
        }
        // Final layer, one dispatch per batch item (rows of item bi start
        // at its segment offset; output compacted to [bi][n_img][pd]).
        let mut fin = Vec::new();
        let pipe = pipeline(c, "zi_final", FINAL_SRC, "zi_final")?;
        for (bi, &(off, _)) in joint_segs.iter().enumerate() {
            let u = ubuf(c, &[d.h as u32, d.pd as u32, d.final_eps.to_bits(), n_img as u32, 1u32 << 30, 0, 0, 0]);
            let xo = off;
            let ob = (bi * n_img * d.pd * 4) as u64;
            let entries = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &joint.x,
                        offset: (xo * d.h * 4) as u64,
                        size: std::num::NonZeroU64::new((n_img * d.h * 4) as u64),
                    }),
                },
                super::bind_buf(1, &fscale),
                super::bind_buf(2, &fw),
                super::bind_buf(3, &fb),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &out,
                        offset: ob,
                        size: std::num::NonZeroU64::new((n_img * d.pd * 4) as u64),
                    }),
                },
                super::bind_buf(5, &u),
            ];
            let bgr = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zi_final"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &entries,
            });
            fin.push(Call { pipe: pipe.clone(), bg: bgr, grid: (n_img as u32, 1, 1) });
        }
        Some(ZStepDev {
            d,
            n_img,
            n_img_p,
            n_cap_p: n_cap_p.to_vec(),
            img,
            joint,
            mods,
            tok,
            cap: capb,
            fscale,
            out,
            pre_a,
            joint_calls,
            fin,
            out_rows,
        })
    }

    /// Record one whole forward into `enc` (two passes + the assembly
    /// copies). `only` restricts the recorded kernels to one class (the
    /// profile; the result is then meaningless).
    pub fn encode(&self, enc: &mut wgpu::CommandEncoder, only: Option<Class>) {
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            self.pre_a.record(&mut pass, only);
        }
        if only.is_none() || only == Some(Class::Io) {
            let h4 = (self.d.h * 4) as u64;
            let mut cap_off = 0u64;
            for (bi, &(off, _)) in self.joint.segs.iter().enumerate() {
                enc.copy_buffer_to_buffer(
                    &self.img.x,
                    (bi * self.n_img_p) as u64 * h4,
                    &self.joint.x,
                    off as u64 * h4,
                    self.n_img_p as u64 * h4,
                );
                let cp = self.n_cap_p[bi] as u64;
                enc.copy_buffer_to_buffer(
                    &self.cap,
                    cap_off * h4,
                    &self.joint.x,
                    (off + self.n_img_p) as u64 * h4,
                    cp * h4,
                );
                cap_off += cp;
            }
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            self.joint_calls.record(&mut pass, only);
            if only.is_none() || only == Some(Class::Io) {
                for f in &self.fin {
                    f.record(&mut pass);
                }
            }
        }
    }

    /// Upload one step's inputs: `x_tok` `[batch][n_img_p][pd]`, the raw
    /// mods `[blocks][4][h]`, the final scale `[h]` (already 1 + …).
    pub fn upload(&self, x_tok: &[f32], mods: &[f32], final_scale: &[f32]) {
        if let Some(c) = zctx() {
            c.queue.write_buffer(&self.tok, 0, bytemuck::cast_slice(x_tok));
            c.queue.write_buffer(&self.mods, 0, bytemuck::cast_slice(mods));
            c.queue.write_buffer(&self.fscale, 0, bytemuck::cast_slice(final_scale));
        }
    }

    /// One forward: record, submit, read back `[batch][n_img][pd]`.
    pub fn run(&self, out: &mut [f32]) -> Option<()> {
        let c = zctx()?;
        let mut enc = c.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, None);
        c.queue.submit(Some(enc.finish()));
        let raw = read_bytes(c, &self.out, (self.out_rows * self.d.pd * 4) as u64)?;
        out[..self.out_rows * self.d.pd].copy_from_slice(bytemuck::cast_slice(&raw));
        Some(())
    }

    /// Seconds per forward (device: one submit per forward, fence at the
    /// end of `reps`), median of `rounds`; `only` = one kernel class.
    pub fn time(&self, reps: usize, rounds: usize, only: Option<Class>) -> Option<f64> {
        let c = zctx()?;
        let run = || {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                let mut enc = c.device.create_command_encoder(&Default::default());
                self.encode(&mut enc, only);
                c.queue.submit(Some(enc.finish()));
            }
            wait(c);
            t.elapsed().as_secs_f64() / reps as f64
        };
        run();
        let mut v: Vec<f64> = (0..rounds.max(1)).map(|_| run()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[v.len() / 2])
    }

    /// Dispatches per forward.
    pub fn dispatches(&self) -> usize {
        self.pre_a.len() + self.joint_calls.len() + self.fin.len()
    }
}
