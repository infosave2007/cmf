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
//! Real weights (the core package's q8 Turbo container, run through its
//! own `zimagegen` with this file dropped in; its tree otherwise
//! untouched):
//! - One oracle forward (r512 p0, step 0, fp32 caption), device against
//!   the core's `step_cpu_taps` on the same inputs: v 8.8e-4. Per block
//!   1e-5 to 1.4e-3, growing with depth. Refined caption 4.4e-5.
//! - Whole pipeline with the same (device) text encoder in both arms:
//!   v_0 1.3e-3 against the CPU DiT.
//!   The remaining image-level gap to an all-CPU run (v_0 3–7 %) comes from
//!   the text encoder running on the device, not from this path.
//! - Step times on real weights: 0.243–0.256 s at 512² and 0.984–1.028 s
//!   at 1024². diffusers bf16 on the same 3090 (all resident, SDPA) takes
//!   0.261 s and 0.997 s per DiT step.
//! - Range guard: the SwiGLU hidden reaches 6.3e4 at step 0 and overflowed
//!   f16 (inf, then NaN) from step 2. It is stored ×2⁻⁶ and w2's f32
//!   epilogue multiplies the 2⁶ back in. Every other f16 site has ≥ 10×
//!   headroom (qkv ≤ 5.7e3, attention input ≤ 622, attention output
//!   ≤ 1.6e3). `CMF_ZI_AMAX=1` prints the per-block, per-site max|x|;
//!   `CMF_ZI_TAPS=<dir>` writes every block's residual stream.
//!
//! # Integration recipe (for the core package)
//!
//! - `prepare` and `step` implement the contract for batch 1 (Turbo).
//!   `prepare` builds the f16 planes once per model, using
//!   [`ZBlockDev::from_model`]: F16 as stored, Bf16/F32 converted,
//!   Q4TiledP / Q8Row / Q8_2f dequantized by the parent's `q4tp_dq_f16` /
//!   `q8_dq_f16`. Any other codec declines.
//!   The contract then builds a [`ZStepDev`] for (n_img, n_cap_p) and
//!   uploads both RoPE tables. `step` uploads x_tok, mods and the final
//!   scale, replays the program and reads back `[n_img][64]`.
//! - The base model with CFG (batch 2) goes through
//!   `ZStepDev::new(.., n_cap_p = &[cond, uncond], cap = both stacked)`
//!   plus `upload(x_tok for both items, ..)` and `run(out [2][n_img][64])`.
//!   The contract has no batch-2 entry yet. Adding one means an
//!   `Option<…>` field agreed with the WP1 lead (plan §2.1). Unequal
//!   caption lengths are supported: segments, per-item final dispatch.
//! - `refine_caption` runs the context refiner on the device: the same
//!   chain, unmodulated, with its planes cached separately.
//! - Still to do on the device side, by measured cost in a real 1024²
//!   image:
//!   - the resident VAE (`vae_decode_chain` declines). The existing wgpu
//!     VAE takes 40–44 s at 1024² and 9 s at 512², against 8 s for all
//!     8 DiT steps. It is the largest item left.
//!   - `prepare` builds the planes cold in 9–12 s (q8 upload + dequant,
//!     once per process).
//!   - The text encoder on the device takes 3.7 s, slower than on the CPU,
//!     and moves v_0 by 3–7 %. That code belongs to the core package.
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
/// programs by key — CFG prepares cond and uncond under two keys and then
/// alternates their steps, so the last `MAX_PROGS` stay live.
struct ZState {
    model_uid: u64,
    blocks: Vec<ZBlockDev>,
    progs: Vec<(u64, ZStepDev)>,
}

const MAX_PROGS: usize = 2;

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
    if !ensure_planes(&mut g, a.model, &d, a.noise_refiner, a.layers) {
        return false;
    }
    let t0 = std::time::Instant::now();
    let st = g.as_mut().unwrap();
    st.progs.retain(|(k, _)| *k != a.key);
    while st.progs.len() >= MAX_PROGS {
        st.progs.remove(0);
    }
    let io = ZIo { x_emb_w: a.x_emb_w, x_emb_b: a.x_emb_b, x_pad: a.x_pad, final_w: a.final_w, final_b: a.final_b };
    let t = ZTiles { guards: ZGuards::for_model(a.model), ..ZTiles::default() };
    let (nr, layers) = st.blocks.split_at(2);
    let prog = match &a.neg {
        None => {
            let Some(prog) = ZStepDev::new(d, &t, nr, layers, &io, n_img, &[n_cap_p], &a.cap[..n_cap_p * d.h]) else {
                return false;
            };
            prog.img.set_rope(&a.rope_img.0[..n_img_p * 64], &a.rope_img.1[..n_img_p * 64]);
            prog.joint.set_rope(&a.rope_joint.0[..s_len * 64], &a.rope_joint.1[..s_len * 64]);
            prog
        }
        Some(ng) => {
            // CFG pair: item 0 = this prompt, item 1 = the negative, each
            // with its own caption length and RoPE rows (batch 2, B1).
            let nc = ng.n_cap_p;
            let s2 = n_img_p + nc;
            if nc % 32 != 0
                || ng.cap.len() < nc * d.h
                || ng.rope_img.0.len() < n_img_p * 64
                || ng.rope_img.1.len() < n_img_p * 64
                || ng.rope_joint.0.len() < s2 * 64
                || ng.rope_joint.1.len() < s2 * 64
            {
                return false;
            }
            let mut cap = Vec::with_capacity((n_cap_p + nc) * d.h);
            cap.extend_from_slice(&a.cap[..n_cap_p * d.h]);
            cap.extend_from_slice(&ng.cap[..nc * d.h]);
            let Some(prog) = ZStepDev::new(d, &t, nr, layers, &io, n_img, &[n_cap_p, nc], &cap) else {
                return false;
            };
            let cat = |x: &[f32], y: &[f32]| -> Vec<f32> { x.iter().chain(y).copied().collect() };
            prog.img.set_rope(
                &cat(&a.rope_img.0[..n_img_p * 64], &ng.rope_img.0[..n_img_p * 64]),
                &cat(&a.rope_img.1[..n_img_p * 64], &ng.rope_img.1[..n_img_p * 64]),
            );
            prog.joint.set_rope(
                &cat(&a.rope_joint.0[..s_len * 64], &ng.rope_joint.0[..s2 * 64]),
                &cat(&a.rope_joint.1[..s_len * 64], &ng.rope_joint.1[..s2 * 64]),
            );
            prog
        }
    };
    prof(if a.neg.is_some() { "program (batch 2)" } else { "program (batch 1)" }, t0);
    st.progs.push((a.key, prog));
    true
}

/// `CMF_ZIMAGE_PROF=1`: in-process sub-stage times of the device path.
fn prof(what: &str, t0: std::time::Instant) {
    if std::env::var("CMF_ZIMAGE_PROF").is_ok_and(|v| v != "0") {
        eprintln!("zimage wgpu: {what} {:.3}s", t0.elapsed().as_secs_f64());
    }
}

/// The 32 per-step blocks' planes of `model` in `g` (built if missing).
fn ensure_planes(g: &mut Option<ZState>, model: &Arc<CmfModel>, d: &ZDims, nr: &[ZBlockRef], layers: &[ZBlockRef]) -> bool {
    let uid = model.uid();
    let nblk = nr.len() + layers.len();
    if g.as_ref().is_some_and(|st| st.model_uid == uid && st.blocks.len() == nblk) {
        return true;
    }
    *g = None; // free the old planes before building new ones
    let t0 = std::time::Instant::now();
    let refs: Vec<&ZBlockRef> = nr.iter().chain(layers.iter()).collect();
    let Some(blocks) = ZBlockDev::from_model_all(model, d, &refs) else { return false };
    prof(&format!("planes {} blocks", blocks.len()), t0);
    *g = Some(ZState { model_uid: uid, blocks, progs: Vec::new() });
    true
}

/// The context refiner's planes (cached apart: model uid, first weight).
fn ensure_refiner(g: &mut Option<(u64, usize, Vec<ZBlockDev>)>, model: &Arc<CmfModel>, d: &ZDims, blocks: &[ZBlockRef]) -> bool {
    let key = (model.uid(), blocks[0].wq);
    if g.as_ref().is_some_and(|(u, w, v)| (*u, *w) == key && v.len() == blocks.len()) {
        return true;
    }
    *g = None;
    let t0 = std::time::Instant::now();
    let refs: Vec<&ZBlockRef> = blocks.iter().collect();
    let Some(v) = ZBlockDev::from_model_all(model, d, &refs) else { return false };
    prof("context-refiner planes", t0);
    *g = Some((key.0, key.1, v));
    true
}

/// Build every plane ahead of `prepare` (B2): the caller overlaps this
/// with the CPU text encoder, which the planes do not depend on.
pub(crate) fn preload(model: &Arc<CmfModel>, geom: &ZGeom, nr: &[ZBlockRef], layers: &[ZBlockRef], cr: &[ZBlockRef]) -> bool {
    if !zi_enabled() || nr.len() != 2 || layers.is_empty() || cr.is_empty() {
        return false;
    }
    let Some(d) = ZDims::from_geom(geom) else { return false };
    if zctx().is_none() {
        return false;
    }
    {
        let Ok(mut g) = ZREFINER.lock() else { return false };
        if !ensure_refiner(&mut g, model, &d, cr) {
            return false;
        }
    }
    let Ok(mut g) = ZSTATE.lock() else { return false };
    ensure_planes(&mut g, model, &d, nr, layers)
}

/// One DiT forward for a prepared `a.key`; writes `a.out`.
pub(crate) fn step(a: &mut ZStepArgs) -> bool {
    let Ok(g) = ZSTATE.lock() else { return false };
    let Some(st) = g.as_ref() else { return false };
    let Some((_, prog)) = st.progs.iter().find(|(k, _)| *k == a.key) else { return false };
    let d = prog.d;
    let nblk = st.blocks.len();
    let batch = prog.n_cap_p.len();
    let (ni, np) = (prog.n_img * d.pd, prog.n_img_p * d.pd);
    if a.x_tok.len() < np
        || a.mods.len() < nblk * 4 * d.h
        || a.final_scale.len() < d.h
        || a.out.len() < ni
        || (batch == 2) != a.out_neg.is_some()
        || batch > 2
        || a.out_neg.as_ref().is_some_and(|o| o.len() < ni)
    {
        return false;
    }
    // Both CFG items denoise the same latent: x_tok twice.
    let x2: Vec<f32>;
    let x_in: &[f32] = if batch == 2 {
        x2 = a.x_tok[..np].iter().chain(&a.x_tok[..np]).copied().collect();
        &x2
    } else {
        &a.x_tok[..np]
    };
    prog.upload(x_in, &a.mods[..nblk * 4 * d.h], &a.final_scale[..d.h]);
    let mut outb = vec![0f32; batch * ni];
    let ok = match std::env::var("CMF_ZI_TAPS") {
        Ok(dir) if !dir.is_empty() => match prog.run_taps(&mut outb) {
            Some(taps) => {
                let dir = std::path::Path::new(&dir).join(format!("step{}", a.step));
                let _ = std::fs::create_dir_all(&dir);
                for (name, v) in taps {
                    let _ = std::fs::write(dir.join(format!("{name}.f32")), bytemuck::cast_slice(&v));
                }
                true
            }
            None => false,
        },
        _ => prog.run(&mut outb).is_some(),
    };
    if ok {
        a.out[..ni].copy_from_slice(&outb[..ni]);
        if let Some(o) = a.out_neg.as_mut() {
            o[..ni].copy_from_slice(&outb[ni..2 * ni]);
        }
    }
    if let Some(v) = prog.amax_read() {
        let ns = AMAX_SITES.len();
        let mut line = format!("zi amax step {}:", a.step);
        for (si, name) in AMAX_SITES.iter().enumerate() {
            // ffn_hid is stored ×2⁻ᵏ (range guard): report the true value.
            let g = match si {
                0 => (prog.guards.attn as f32).exp2(),
                1 | 2 => (prog.guards.qkv as f32).exp2(),
                5 => (prog.guards.hid as f32).exp2(),
                _ => 1.0,
            };
            let (bi, m) = (0..v.len() / ns).map(|b| (b, v[b * ns + si] * g)).fold((0, 0f32), |acc, x| if x.1 > acc.1 { x } else { acc });
            line += &format!(" {name} {m:.3e}@{bi}");
        }
        eprintln!("{line}");
        if std::env::var("CMF_ZI_AMAX_ALL").is_ok() {
            for b in 0..v.len() / ns {
                eprintln!("  blk {b:2}: {:?}", &v[b * ns..(b + 1) * ns]);
            }
        }
    }
    ok
}

/// The DiT half of `release`: planes, prepared programs, the context
/// refiner — the VAE chain stays.
pub(crate) fn release_dit() {
    if let Ok(mut g) = ZSTATE.lock() {
        *g = None;
    }
    if let Ok(mut g) = ZREFINER.lock() {
        *g = None;
    }
}

/// Upload the VAE weights and compile every kernel the decode uses.
pub(crate) fn vae_prewarm(a: &crate::vae::VaeChainArgs) -> bool {
    if !zi_enabled() || std::env::var("CMF_ZI_VAE").as_deref() == Ok("0") {
        return false;
    }
    let Some(c) = zctx() else { return false };
    let t0 = std::time::Instant::now();
    {
        let Ok(mut g) = ZVAE.lock() else { return false };
        if !g.as_ref().is_some_and(|v| v.key == a.key) {
            *g = None;
            match VaeDev::build(c, a) {
                Some(v) => *g = Some(v),
                None => return false,
            }
        }
    }
    let kernels: [(&str, &str, &str); 7] = [
        ("zv_gn_part", VAE_GN_PART_SRC, "vae_gn_part"),
        ("zv_gn_fin", VAE_GN_FIN_SRC, "vae_gn_fin"),
        ("zv_gn_apply", VAE_GN_APPLY_SRC, "vae_gn_apply"),
        ("zv_combine", VAE_COMBINE_SRC, "vae_combine"),
        ("zv_cast", VAE_CAST_SRC, "vae_cast"),
        ("zv_softmax", VAE_SOFTMAX_SRC, "vae_softmax"),
        ("zv_out", VAE_OUT_SRC, "vae_out"),
    ];
    for (k, src, e) in kernels {
        if pipeline(c, k, src, e).is_none() {
            return false;
        }
    }
    for g in [
        MmCfg { conv: 1, ..default_cfg(Epi::F32) },
        MmCfg { conv: 2, ..default_cfg(Epi::F32) },
        default_cfg(Epi::F32),
        default_cfg(Epi::F16),
    ] {
        if mm_pipe(c, g).is_none() {
            return false;
        }
    }
    prof("vae prewarm (weights + kernels)", t0);
    true
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
    if let Ok(mut g) = ZVAE.lock() {
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
    if !ensure_refiner(&mut g, model, &d, blocks) {
        return false;
    }
    let devs = &g.as_ref().unwrap().2;
    let Some(seq) = ZSeq::new(&d, &[(0, n)]) else { return false };
    seq.set_rope(&rope_cap.0[..n * 64], &rope_cap.1[..n * 64]);
    seq.write_x(cap);
    // Unmodulated: the row ops read no mods (scale 0, gate 1).
    let mods = sbuf(c, 16, "zi_nomods");
    let t = ZTiles { guards: ZGuards::for_model(model), ..ZTiles::default() };
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
/// `CMF_ZI_VAE=0` declines (the parent's per-conv path runs).
pub(crate) fn vae_decode_chain(
    a: &crate::vae::VaeChainArgs,
    z: &[f32],
    h: usize,
    w: usize,
    out: &mut [f32],
) -> bool {
    if !zi_enabled() || std::env::var("CMF_ZI_VAE").as_deref() == Ok("0") {
        return false;
    }
    vae_decode_dev(a, z, h, w, out).is_some()
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
    /// Implicit-GEMM 3×3 convolution (the VAE, B2): 0 = plain GEMM,
    /// 1 = 3×3 pad-1 conv over an NHWC f16 image (`act` = the image,
    /// K = 9·cin in (tap, channel) order), 2 = the same on the nearest-2×
    /// upsample of `act` (the image is at half the output size).
    pub conv: u32,
}

impl MmCfg {
    pub const fn new(bm: u32, bn: u32, bk: u32, wm: u32, wn: u32, epi: Epi) -> Self {
        Self { bm, bn, bk, wm, wn, epi, direct: false, acc16_probe: false, stages: 1, conv: 0 }
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
            && (self.conv == 0 || (!self.direct && self.stages == 1 && !self.acc16_probe && nt % vpr == 0))
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
            } else if self.conv != 0 {
                format!("_conv{}", self.conv)
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
    if g.conv == 0 {
        let _ = writeln!(
            s,
            "struct MmP {{ m: u32, n: u32, k: u32, ldo: u32, ocol: u32, arow: u32, oscale: f32, _p: u32 }};"
        );
    } else {
        // cw/ch = OUTPUT width/height, cin = input channels (a multiple of
        // bk, so one K slice never straddles two taps).
        let _ = writeln!(
            s,
            "struct MmP {{ m: u32, n: u32, k: u32, ldo: u32, ocol: u32, arow: u32, oscale: f32, _p: u32, cw: u32, ch: u32, cin: u32, _q: u32 }};"
        );
    }
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
    if g.conv != 0 {
        let up = if g.conv == 2 { 1 } else { 0 };
        let _ = writeln!(
            s,
            "fn ldc(y: i32, x: i32, kt: u32, vc: u32) -> vec4<f16> {{
  let k0 = kt * {bk}u;
  let tap = k0 / p.cin;
  let ci = k0 - tap * p.cin + vc * 4u;
  let sy = y + i32(tap / 3u) - 1;
  let sx = x + i32(tap % 3u) - 1;
  if (sy < 0 || sx < 0 || sy >= i32(p.ch) || sx >= i32(p.cw)) {{ return vec4<f16>(0.0h); }}
  let src = (u32(sy) >> {up}u) * (p.cw >> {up}u) + (u32(sx) >> {up}u);
  return act[(src * p.cin + ci) / 4u];
}}",
            bk = g.bk
        );
    }
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
        if g.conv != 0 {
            // Output pixel of each loaded row, fixed for the whole K loop.
            let _ = writeln!(s, "  let vc = tid % {vpr}u;");
            for t in 0..la {
                let _ = writeln!(
                    s,
                    "  let r{t} = m0 + (tid + {o}u) / {vpr}u; let y{t} = i32(r{t} / p.cw); let x{t} = i32(r{t} % p.cw);\n  var ra{t} = ldc(y{t}, x{t}, 0u, vc);",
                    o = t * nt
                );
            }
        } else {
            for t in 0..la {
                let _ = writeln!(
                    s,
                    "  var ra{t} = act[(p.arow + m0 + (tid + {o}u) / {vpr}u) * kq + (tid + {o}u) % {vpr}u];",
                    o = t * nt
                );
            }
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
                if g.conv != 0 {
                    let _ = writeln!(s, "      ra{t} = ldc(y{t}, x{t}, kt + 1u, vc);");
                } else {
                    let _ = writeln!(
                        s,
                        "      ra{t} = act[(p.arow + m0 + (tid + {o}u) / {vpr}u) * kq + kb + (tid + {o}u) % {vpr}u];",
                        o = t * nt
                    );
                }
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
            // `oscale` undoes the producer's range guard (w2 reads the
            // SwiGLU hidden stored ×2⁻ᵏ): one scalar multiply per fragment.
            let _ = writeln!(s, "  let ocol = p.ocol + n0 + wx * {tn}u;");
            let _ = writeln!(s, "  let osc = p.oscale;");
            for i in 0..fm {
                for j in 0..fnn {
                    let _ = writeln!(
                        s,
                        "  {{ let oi = (orow + {}u) * ldo + ocol + {}u; let cv = c{i}_{j} * osc; coopStoreT(cv, &outp[oi], ldo); }}",
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
    /// Conv variants only: [output width, output height, input channels].
    pub conv: [u32; 3],
}

fn mm_uniform(c: &Ctx, a: &MmArgs) -> wgpu::Buffer {
    let w: [u32; 12] =
        [a.m, a.n, a.k, a.ldo, a.ocol, a.arow, a.oscale.to_bits(), 0, a.conv[0], a.conv[1], a.conv[2], 0];
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

/// max|x| of a panel into one slot of a u32 array (float bits, atomicMax;
/// NaN counts as +inf). `is_f32 = 0`: the panel is packed f16 pairs.
const AMAX_SRC: &str = r#"
struct AP { n: u32, slot: u32, is_f32: u32, _p: u32 };
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> outm: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> ap: AP;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn zi_amax(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    var m = 0.0;
    for (var i = gid.x; i < ap.n; i = i + nwg.x * 256u) {
        var v = 0.0;
        if (ap.is_f32 != 0u) {
            v = abs(bitcast<f32>(src[i]));
        } else {
            let p = unpack2x16float(src[i]);
            v = max(abs(p.x), abs(p.y));
            if (p.x != p.x || p.y != p.y) { v = 3.0e38; }
        }
        if (v != v) { v = 3.0e38; }
        m = max(m, v);
    }
    red[lid] = m;
    workgroupBarrier();
    var st = 128u;
    loop {
        if (st == 0u) { break; }
        if (lid < st) { red[lid] = max(red[lid], red[lid + st]); }
        workgroupBarrier();
        st = st >> 1u;
    }
    if (lid == 0u) { atomicMax(&outm[ap.slot], bitcast<u32>(red[0])); }
}
"#;

/// Probe sites recorded per block when `CMF_ZI_AMAX=1` (block-major slots).
pub const AMAX_SITES: [&str; 8] = ["attn_in", "qkv", "attn_out", "o_proj", "ffn_in", "ffn_hid", "w2_out", "x"];

fn amax_call(c: &Ctx, src: &wgpu::Buffer, words: usize, is_f32: bool, dst: &wgpu::Buffer, slot: u32) -> Option<Call> {
    let pipe = pipeline(c, "zi_amax", AMAX_SRC, "zi_amax")?;
    let u = ubuf(c, &[words as u32, slot, is_f32 as u32, 0]);
    let b = bg(c, &pipe, &[src, dst, &u]);
    Some(Call { pipe, bg: b, grid: (256, 1, 1) })
}

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
        let args = MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: ncol as u32, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
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
        let args = MmArgs { m: m as u32, n: n as u32, k: k as u32, ldo: ncol as u32, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
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
/// F16 = bytes as stored; Bf16/F32 converted on the host; Q4TiledP,
/// Q8Row and Q8_2f dequantized on the device by the parent's
/// `q4tp_dq_f16` / `q8_dq_f16` (the plane layout the parent's coop GEMM
/// eats). Other codecs → `None`.
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
        T::Q4TiledP | T::Q8Row | T::Q8_2f => {
            // Device dequant into a temporary [rows][cols] plane with the
            // parent's kernels, then copy the panels into place.
            let need = cortiq_core::quant::expected_nbytes(e.dtype, &[rows, cols])?;
            if bytes.len() < need {
                return None;
            }
            let n = rows * cols;
            let tmp = sbuf(c, (n * 2) as u64, "zi_dq");
            let pad4 = |v: &[u8]| {
                let mut v = v.to_vec();
                v.resize(v.len().next_multiple_of(4), 0);
                v
            };
            let (pipe, bufs): (&wgpu::ComputePipeline, Vec<wgpu::Buffer>) = if e.dtype == T::Q4TiledP {
                let src = sbuf_init(c, &pad4(&bytes[..need]), "zi_q4tp");
                (c.q4tp_dq_f16.as_ref()?, vec![src, tmp.clone(), ubuf(c, &[cols as u32, rows as u32, 0, 0])])
            } else {
                // q8_row: int8 [rows][cols] + f16 row scales; q8_2f adds f16
                // column scales. The kernel takes both fields as f32.
                let h2f = |b: &[u8]| -> Vec<f32> {
                    b.chunks_exact(2).map(|x| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([x[0], x[1]]))).collect()
                };
                let rsc = h2f(&bytes[n..n + rows * 2]);
                let col = if e.dtype == T::Q8_2f { Some(h2f(&bytes[n + rows * 2..n + rows * 2 + cols * 2])) } else { None };
                let q = sbuf_init(c, &pad4(&bytes[..n]), "zi_q8");
                let rs = sbuf_init(c, bytemuck::cast_slice(&rsc), "zi_q8_rs");
                let cs = sbuf_init(c, bytemuck::cast_slice(col.as_deref().unwrap_or(&[1.0f32])), "zi_q8_cs");
                let u = ubuf(c, &[cols as u32, rows as u32, u32::from(col.is_some()), 0]);
                (c.q8_dq_f16.as_ref()?, vec![q, tmp.clone(), u, rs, cs])
            };
            let refs: Vec<&wgpu::Buffer> = bufs.iter().collect();
            let b = bg(c, pipe, &refs);
            let mut enc = c.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &b, &[]);
                let wgs = ((n / 2) as u32).div_ceil(256);
                pass.dispatch_workgroups(wgs.min(65535), wgs.div_ceil(65535), 1);
            }
            if contiguous {
                enc.copy_buffer_to_buffer(&tmp, 0, dst, (panel_row(0) * cols * 2) as u64, (n * 2) as u64);
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
    /// indices, diffusers names). F16 / Bf16 / F32 / Q4TiledP / Q8Row /
    /// Q8_2f; any other codec → `None` (the caller declines and the CPU
    /// path runs).
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

/// `zi_dq8`: q8_row / q8_2f tensor bytes exactly as stored in the file
/// (`[int8 rows·cols][f16 row scale][f16 col scale]`) → f16 plane rows,
/// 4 weights per thread, `w = (q·rs)·cs` in f32 (the CPU order), with the
/// destination row remap of the plane (mode 0 = linear from `dst_row0`,
/// 1/2 = the w13 gate/up panel interleave).
const DQ8_SRC: &str = r#"
struct DP { cols: u32, rows: u32, has_col: u32, mode: u32, src_w: u32, rs_w: u32, cs_w: u32, dst_row0: u32 };
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> p: DP;
fn f16at(w0: u32, i: u32) -> f32 {
  let v = unpack2x16float(src[w0 + i / 2u]);
  return select(v.x, v.y, (i & 1u) == 1u);
}
fn s8(w: u32, j: u32) -> f32 {
  let b = (w >> (8u * j)) & 0xFFu;
  return select(f32(b), f32(b) - 256.0, b > 127u);
}
@compute @workgroup_size(256)
fn zi_dq8(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let q4 = p.cols / 4u;
  let idx = gid.y * (nwg.x * 256u) + gid.x;
  if (idx >= p.rows * q4) { return; }
  let r = idx / q4;
  let c4 = idx % q4;
  let w = src[p.src_w + idx];
  let rs = f16at(p.rs_w, r);
  var cs = vec4<f32>(1.0);
  if (p.has_col != 0u) {
    let a = unpack2x16float(src[p.cs_w + 2u * c4]);
    let b = unpack2x16float(src[p.cs_w + 2u * c4 + 1u]);
    cs = vec4<f32>(a.x, a.y, b.x, b.y);
  }
  let v = vec4<f32>(s8(w, 0u) * rs * cs.x, s8(w, 1u) * rs * cs.y, s8(w, 2u) * rs * cs.z, s8(w, 3u) * rs * cs.w);
  var pr = r;
  if (p.mode == 1u) { pr = 32u * (r / 16u) + r % 16u; }
  if (p.mode == 2u) { pr = 32u * (r / 16u) + r % 16u + 16u; }
  dst[(p.dst_row0 + pr) * q4 + c4] = vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}
"#;

/// One q8 tensor → plane job of the streamed builder.
struct Dq8Job {
    /// Which plane of the block (0 qkv, 1 o, 2 w13, 3 w2).
    plane: usize,
    rows: usize,
    cols: usize,
    has_col: bool,
    mode: u32,
    dst_row0: usize,
    /// The tensor's bytes in the file (mmap).
    bytes: &'static [u8],
    /// Their byte offset in the block's source buffer.
    src_off: usize,
}

impl ZBlockDev {
    /// Planes of MANY blocks, streamed (B2). For q8_row / q8_2f weights the
    /// raw file bytes of one block (7 tensors, 177 MB for Turbo) go into a
    /// staging view filled by parallel threads straight from the mmap (one
    /// host copy), and `zi_dq8` expands them on the device while the next
    /// block is being filled (two alternating source buffers). Any other
    /// codec falls back to `from_model` for that block.
    /// `CMF_ZI_PLANE_FAST=0` = the B1 per-tensor path (A/B arm).
    pub fn from_model_all(model: &Arc<CmfModel>, d: &ZDims, refs: &[&ZBlockRef]) -> Option<Vec<ZBlockDev>> {
        use cortiq_core::TensorDtype as T;
        let c = zctx()?;
        if std::env::var("CMF_ZI_PLANE_FAST").as_deref() == Ok("0") {
            return refs.iter().map(|r| ZBlockDev::from_model(model, d, r)).collect();
        }
        let (h, i) = (d.h, d.inter);
        let jobs_of = |r: &ZBlockRef| -> Option<(Vec<Dq8Job>, usize)> {
            let spec: [(usize, usize, usize, usize, u32, usize); 7] = [
                (r.wq, 0, h, h, 0, 0),
                (r.wk, 0, h, h, 0, h),
                (r.wv, 0, h, h, 0, 2 * h),
                (r.wo, 1, h, h, 0, 0),
                (r.w1, 2, i, h, 1, 0),
                (r.w3, 2, i, h, 2, 0),
                (r.w2, 3, h, i, 0, 0),
            ];
            let mut jobs = Vec::with_capacity(7);
            let mut off = 0usize;
            for (idx, plane, rows, cols, mode, dst_row0) in spec {
                let e = model.tensors.get(idx)?;
                if e.shape.len() != 2 || e.shape[0] != rows || e.shape[1] != cols || rows % 32 != 0 || cols % 4 != 0 {
                    return None;
                }
                if !matches!(e.dtype, T::Q8_2f | T::Q8Row) {
                    return None;
                }
                let need = cortiq_core::quant::expected_nbytes(e.dtype, &[rows, cols])?;
                let b = model.entry_bytes(e);
                if b.len() < need {
                    return None;
                }
                // SAFETY: the mmap lives as long as `model` (an Arc the
                // caller holds for the whole build); the slice never
                // outlives this function.
                let bytes: &'static [u8] = unsafe { std::slice::from_raw_parts(b.as_ptr(), need) };
                jobs.push(Dq8Job { plane, rows, cols, has_col: e.dtype == T::Q8_2f, mode, dst_row0, bytes, src_off: off });
                off += need.next_multiple_of(256);
            }
            Some((jobs, off))
        };
        let plans: Vec<Option<(Vec<Dq8Job>, usize)>> = refs.iter().map(|r| jobs_of(r)).collect();
        let max_src = plans.iter().flatten().map(|p| p.1).max().unwrap_or(0);
        let pipe = pipeline(c, "zi_dq8", DQ8_SRC, "zi_dq8")?;
        let srcs: Vec<wgpu::Buffer> = if max_src > 0 {
            (0..2).map(|_| sbuf(c, max_src as u64, "zi_dq8_src")).collect()
        } else {
            Vec::new()
        };
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 16);
        let mut out = Vec::with_capacity(refs.len());
        let mut last: Option<wgpu::SubmissionIndex> = None;
        let (mut t_view, mut t_copy, mut t_wait, mut t_alloc) = (0f64, 0f64, 0f64, 0f64);
        // Two persistent host staging buffers (mapped at creation for their
        // first block) and the submission that last read each.
        let stg: Vec<wgpu::Buffer> = if max_src > 0 {
            (0..2)
                .map(|_| {
                    c.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("zi_dq8_stage"),
                        size: (max_src as u64).next_multiple_of(4),
                        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut stg_sub: [Option<wgpu::SubmissionIndex>; 2] = [None, None];
        for (bi, (r, plan)) in refs.iter().zip(plans).enumerate() {
            let Some((jobs, total)) = plan else {
                if let Some(ix) = last.take() {
                    let _ = c.device.poll(wgpu::PollType::Wait { submission_index: Some(ix), timeout: None });
                }
                out.push(ZBlockDev::from_model(model, d, r)?);
                continue;
            };
            if r.norm1.len() < h || r.norm2.len() < h || r.ffn_norm1.len() < h || r.ffn_norm2.len() < h
                || r.norm_q.len() < 128 || r.norm_k.len() < 128
            {
                return None;
            }
            let k = bi % 2;
            let src = &srcs[k];
            let st = &stg[k];
            {
                // The staging buffer is reused: wait for the block that
                // last read it, then map it again (a fresh 177 MB staging
                // allocation per block — `write_buffer_with` — cost 28 ms
                // each, 0.9 s of the 1.9 s build).
                let tv = std::time::Instant::now();
                if let Some(ix) = stg_sub[k].take() {
                    let _ = c.device.poll(wgpu::PollType::Wait { submission_index: Some(ix), timeout: None });
                    st.slice(..).map_async(wgpu::MapMode::Write, |_| {});
                    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
                }
                let mut view = st.slice(..total as u64).get_mapped_range_mut().ok()?;
                t_view += tv.elapsed().as_secs_f64();
                let tv = std::time::Instant::now();
                // Cut the view into ≤ 8 MB pieces, one list pulled by the
                // threads (the mmap page faults run in parallel too).
                // (destination address, source) — `WriteOnly<[u8]>` is not
                // Send, so the threads get the raw staging address.
                let mut pieces: Vec<(usize, &[u8])> = Vec::new();
                let mut rest = view.slice(..);
                let mut cur = 0usize;
                for j in &jobs {
                    let (_, r2) = rest.split_at(j.src_off - cur);
                    let (mut piece, r3) = r2.split_at(j.bytes.len());
                    rest = r3;
                    cur = j.src_off + j.bytes.len();
                    let mut sb = j.bytes;
                    while !sb.is_empty() {
                        let n = sb.len().min(8 << 20);
                        let (mut head, tail) = piece.split_at(n);
                        pieces.push((head.as_raw_ptr().as_ptr() as *mut u8 as usize, &sb[..n]));
                        piece = tail;
                        sb = &sb[n..];
                    }
                }
                let work = Mutex::new(pieces);
                std::thread::scope(|sc| {
                    for _ in 0..nthreads {
                        sc.spawn(|| loop {
                            let job = work.lock().ok().and_then(|mut v| v.pop());
                            match job {
                                // SAFETY: disjoint pieces of one live
                                // write mapping, each exactly `srcb.len()`.
                                Some((dst, srcb)) => unsafe {
                                    std::ptr::copy_nonoverlapping(srcb.as_ptr(), dst as *mut u8, srcb.len())
                                },
                                None => break,
                            }
                        });
                    }
                });
                t_copy += tv.elapsed().as_secs_f64();
            }
            st.unmap();
            let ta = std::time::Instant::now();
            let planes = [
                sbuf(c, (3 * h * h * 2) as u64, "zi_plane_qkv"),
                sbuf(c, (h * h * 2) as u64, "zi_plane_o"),
                sbuf(c, (2 * i * h * 2) as u64, "zi_plane_w13"),
                sbuf(c, (h * i * 2) as u64, "zi_plane_w2"),
            ];
            t_alloc += ta.elapsed().as_secs_f64();
            let mut enc = c.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(st, 0, src, 0, total as u64);
            let mut keep = Vec::new();
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipe);
                for j in &jobs {
                    let sw = (j.src_off / 4) as u32;
                    let rs_w = sw + (j.rows * j.cols / 4) as u32;
                    let cs_w = rs_w + (j.rows / 2) as u32;
                    let u = ubuf(c, &[j.cols as u32, j.rows as u32, j.has_col as u32, j.mode, sw, rs_w, cs_w, j.dst_row0 as u32]);
                    let bgr = bg(c, &pipe, &[src, &planes[j.plane], &u]);
                    pass.set_bind_group(0, &bgr, &[]);
                    let wgs = ((j.rows * j.cols / 4) as u32).div_ceil(256);
                    pass.dispatch_workgroups(wgs.min(65535), wgs.div_ceil(65535), 1);
                    keep.push((u, bgr));
                }
            }
            let ix = c.queue.submit(Some(enc.finish()));
            stg_sub[k] = Some(ix.clone());
            drop(keep);
            // Bound the staging memory: block bi−1 must be done before
            // bi+1 reuses its source buffer.
            if let Some(prev) = last.replace(ix) {
                let tw = std::time::Instant::now();
                let _ = c.device.poll(wgpu::PollType::Wait { submission_index: Some(prev), timeout: None });
                t_wait += tw.elapsed().as_secs_f64();
            }
            let upf = |v: &[f32]| sbuf_init(c, bytemuck::cast_slice(v), "zi_norm");
            let [qkv, o, w13, w2] = planes;
            out.push(ZBlockDev {
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
            });
        }
        wait(c);
        if std::env::var("CMF_ZIMAGE_PROF").is_ok_and(|v| v != "0") {
            eprintln!("zimage wgpu: planes: staging maps {t_view:.3}s · parallel copy {t_copy:.3}s ({nthreads} threads) · plane allocs {t_alloc:.3}s · waits {t_wait:.3}s");
        }
        Some(out)
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

/// f16 range guards of the chain, log2 of the power-of-two each f16 site is
/// stored divided by (the consumer multiplies it back, or is scale-free):
/// - `attn`: the attention input (pre-norm·(1+scale_msa)) the qkv GEMM
///   reads; the qkv GEMM's epilogue applies 2^(attn − qkv);
/// - `qkv`: the q/k/v panel; the qk-RMSNorm is scale-free (its eps is
///   scaled by 2⁻²ᵠ), v carries the factor into the attention output and
///   the O GEMM's f32 epilogue multiplies 2ᵠ back;
/// - `hid`: the SwiGLU hidden; w2's f32 epilogue multiplies 2ʰ back.
///
/// Measured maxima over all steps, both prompts, 512² and 1024²
/// (`CMF_ZI_AMAX=1`, q8 containers): Turbo attn 6.3e2, qkv 5.8e3,
/// hidden 3.1e5 → (0, 0, 6). Base (CFG pair, 28 steps): attn 3.0e5, qkv
/// 1.25e6, attention output 3.4e5, hidden 7.1e6, all at layer 28 (it was
/// inf unguarded) → (6, 7, 11), which keeps every stored f16 ≤ 1.1e4.
/// One forward's v against the CPU: base 3.0e-4 / 1.8e-4 (r512 i0 / i2)
/// with (4, 5, 10);
/// Turbo 8.8e-4 / 5.7e-4 / 1.7e-3 (r512 i0, i5, r1024 i0) with (0, 0, 6)
/// and 1.1e-3 / 6.1e-4 / 1.6e-3 with the base guards — the same within
/// noise; each model keeps the smallest guards that hold.
/// `CMF_ZI_ATTN_SHIFT` / `CMF_ZI_QKV_SHIFT` / `CMF_ZI_HID_SHIFT` override.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZGuards {
    pub attn: i32,
    pub qkv: i32,
    pub hid: i32,
}

impl ZGuards {
    pub const TURBO: ZGuards = ZGuards { attn: 0, qkv: 0, hid: 6 };
    pub const BASE: ZGuards = ZGuards { attn: 6, qkv: 7, hid: 11 };

    /// The guards of a container's variant (`zimage.config_json`), env
    /// overrides applied.
    pub fn for_model(model: &CmfModel) -> ZGuards {
        let variant = model
            .tensor_bytes("zimage.config_json")
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .and_then(|v| v["variant"].as_str().map(str::to_string));
        let base = if variant.as_deref() == Some("base") { Self::BASE } else { Self::TURBO };
        base.with_env()
    }

    pub fn with_env(self) -> ZGuards {
        let e = |k: &str, d: i32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d).clamp(0, 14);
        ZGuards {
            attn: e("CMF_ZI_ATTN_SHIFT", self.attn),
            qkv: e("CMF_ZI_QKV_SHIFT", self.qkv),
            hid: e("CMF_ZI_HID_SHIFT", self.hid),
        }
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
    oscale: f32,
) -> Option<Call> {
    let pipe = pipeline(c, "zi_rowop", ROWOP_SRC, "zi_rowop")?;
    let u = ubuf(
        c,
        &[d.h as u32, mode, g_off, s_off, d.eps.to_bits(), d.eps.to_bits(), oscale.to_bits(), 0],
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
    /// f16 range guards (per model variant).
    pub guards: ZGuards,
}

impl Default for ZTiles {
    fn default() -> Self {
        ZTiles {
            qkv: default_cfg(Epi::F16),
            o: default_cfg(Epi::F32),
            w13: default_cfg(Epi::SwiGlu),
            w2: default_cfg(Epi::F32),
            flash: default_flash(),
            guards: ZGuards::TURBO.with_env(),
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
    block_calls_probe(d, t, seq, blk, mods, bi, first, next, modulated, None)
}

/// `block_calls` with an optional max|x| probe: after every producer a
/// `zi_amax` dispatch folds the site's panel into `probe.0[probe.1 + site]`
/// (sites: [`AMAX_SITES`]).
#[allow(clippy::too_many_arguments)]
pub fn block_calls_probe(
    d: &ZDims,
    t: &ZTiles,
    seq: &ZSeq,
    blk: &ZBlockDev,
    mods: &wgpu::Buffer,
    bi: usize,
    first: bool,
    next: Option<(&ZBlockDev, usize)>,
    modulated: bool,
    probe: Option<(&wgpu::Buffer, u32)>,
) -> Option<ZCalls> {
    let c = zctx()?;
    let (mh, mhalf) = (seq.m * d.h, seq.m * d.h / 2);
    let pr = |v: &mut ZCalls, site: u32, buf: &wgpu::Buffer, words: usize, f32w: bool| -> Option<()> {
        if let Some((pb, base)) = probe {
            v.push(Class::Io, amax_call(c, buf, words, f32w, pb, base + site)?);
        }
        Some(())
    };
    let (h, i, m) = (d.h as u32, d.inter as u32, seq.m as u32);
    let gm = if modulated { ROW_GATE_MOD } else { 0 };
    // Range guards (B2): the attention input is stored ×2⁻ᵃ, the qkv panel
    // ×2⁻ᵠ (so the qkv GEMM scales by 2^(a−q)), the SwiGLU hidden ×2⁻ʰ.
    let gd = t.guards;
    let att_in_scale = (-(gd.attn as f32)).exp2();
    let sm = if modulated { ROW_SCALE_MOD } else { 0 };
    let mut v = ZCalls::default();
    if first {
        v.push(
            Class::Rows,
            rowop_call(c, d, seq, ROW_PRE | sm, &blk.norm1, &blk.norm1, mods, 0, mod_off(d, bi, 0), att_in_scale)?,
        );
    }
    pr(&mut v, 0, &seq.xn, mhalf, false)?;
    let a = |n: u32, k: u32, ldo: u32| MmArgs { m, n, k, ldo, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
    let qs = gd.qkv;
    let aq = MmArgs { oscale: ((gd.attn - qs) as f32).exp2(), ..a(3 * h, h, 3 * h) };
    let ao = MmArgs { oscale: (qs as f32).exp2(), ..a(h, h, h) };
    v.push_mm(Class::MmQkv, mm_call(c, t.qkv, &aq, &blk.qkv, &seq.xn, &seq.qkv)?);
    pr(&mut v, 1, &seq.qkv, 3 * mhalf, false)?;
    {
        let pipe = pipeline(c, "zi_qkrope", QKROPE_SRC, "zi_qkrope")?;
        // q/k arrive ×2⁻ᵏ (qkv range guard): scale eps by 2⁻²ᵏ so the
        // RMSNorm is exactly the unscaled one.
        let eps = d.eps * (-2.0 * qs as f32).exp2();
        let u = ubuf(c, &[3 * h, d.nh as u32, eps.to_bits(), 0]);
        let b = bg(c, &pipe, &[&seq.qkv, &blk.norm_q, &blk.norm_k, &seq.rope_c, &seq.rope_s, &u]);
        v.push(Class::Rows, Call { pipe, bg: b, grid: (2 * d.nh as u32, m.min(65535), m.div_ceil(65535)) });
    }
    for cl in flash_calls(c, t.flash, d.nh, &seq.qkv, &seq.att, &seq.segs)? {
        v.push(Class::Flash, cl);
    }
    pr(&mut v, 2, &seq.att, mhalf, false)?;
    v.push_mm(Class::MmO, mm_call(c, t.o, &ao, &blk.o, &seq.att, &seq.br)?);
    pr(&mut v, 3, &seq.br, mh, true)?;
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
            1.0,
        )?,
    );
    pr(&mut v, 4, &seq.xn, mhalf, false)?;
    // Range guard: the SwiGLU hidden of real Z-Image weights reaches 6.3e4
    // at t≈0 and overflows f16 (inf) from step 2 (measured, CMF_ZI_AMAX on
    // the q8 Turbo container, 512²). It is stored ×2⁻ᵏ and w2 multiplies
    // 2ᵏ back in its f32 epilogue. `CMF_ZI_HID_SHIFT=k` (default 6).
    let hs = gd.hid;
    let a13 = MmArgs { oscale: (-(hs as f32)).exp2(), ..a(2 * i, h, i) };
    let a2 = MmArgs { oscale: (hs as f32).exp2(), ..a(h, i, h) };
    v.push_mm(Class::MmW13, mm_call(c, t.w13, &a13, &blk.w13, &seq.xn, &seq.hid)?);
    pr(&mut v, 5, &seq.hid, seq.m * d.inter / 2, false)?;
    v.push_mm(Class::MmW2, mm_call(c, t.w2, &a2, &blk.w2, &seq.hid, &seq.br)?);
    pr(&mut v, 6, &seq.br, mh, true)?;
    let (mode, wpre, s_off) = match next {
        Some((nb, nbi)) => (ROW_GRES | gm | ROW_PRE | sm, &nb.norm1, mod_off(d, nbi, 0)),
        None => (ROW_GRES | gm, &blk.ffn_norm2, 0),
    };
    v.push(Class::Rows, rowop_call(c, d, seq, mode, &blk.ffn_norm2, wpre, mods, mod_off(d, bi, 3), s_off, att_in_scale)?);
    pr(&mut v, 7, &seq.x, mh, true)?;
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
    /// Call-list lengths after each block (noise refiner in `pre_a`, main
    /// layers in `joint_calls`): the tap points of `run_taps`.
    pre_ends: Vec<usize>,
    joint_ends: Vec<usize>,
    fin: Vec<Call>,
    out_rows: usize,
    /// `CMF_ZI_AMAX=1`: per (block, site) max|x| of the last forward.
    amax: Option<wgpu::Buffer>,
    /// The range guards the program was built with.
    pub guards: ZGuards,
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
        let amax = (std::env::var("CMF_ZI_AMAX").as_deref() == Ok("1"))
            .then(|| sbuf(c, (nblk * AMAX_SITES.len() * 4) as u64, "zi_amax"));
        let probe = |bi: usize| amax.as_ref().map(|b| (b, (bi * AMAX_SITES.len()) as u32));
        let mut pre_a = ZCalls::default();
        {
            let pipe = pipeline(c, "zi_embed", EMBED_SRC, "zi_embed")?;
            let u = ubuf(c, &[d.h as u32, n_img as u32, n_img_p as u32, d.pd as u32]);
            let bgr = bg(c, &pipe, &[&tok, &ew, &eb, &ep, &img.x, &u]);
            pre_a.push(Class::Io, Call { pipe, bg: bgr, grid: (img.m as u32, 1, 1) });
        }
        let mut pre_ends = Vec::new();
        for (k, blk) in nr.iter().enumerate() {
            let next = nr.get(k + 1).map(|nb| (nb, k + 1));
            pre_a.extend(block_calls_probe(&d, t, &img, blk, &mods, k, k == 0, next, true, probe(k))?);
            pre_ends.push(pre_a.len());
        }
        let mut joint_calls = ZCalls::default();
        let mut joint_ends = Vec::new();
        for (k, blk) in layers.iter().enumerate() {
            let bi = nr.len() + k;
            let next = layers.get(k + 1).map(|nb| (nb, bi + 1));
            joint_calls.extend(block_calls_probe(&d, t, &joint, blk, &mods, bi, k == 0, next, true, probe(bi))?);
            joint_ends.push(joint_calls.len());
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
            pre_ends,
            joint_ends,
            fin,
            out_rows,
            amax,
            guards: t.guards,
        })
    }

    /// The probe of the last forward: `[blocks][AMAX_SITES]` max|x|
    /// (None unless built with `CMF_ZI_AMAX=1`). Resets the slots.
    pub fn amax_read(&self) -> Option<Vec<f32>> {
        let c = zctx()?;
        let b = self.amax.as_ref()?;
        let raw = read_bytes(c, b, b.size())?;
        c.queue.write_buffer(b, 0, &vec![0u8; b.size() as usize]);
        Some(bytemuck::cast_slice(&raw).to_vec())
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

    /// One forward that also returns the residual stream after every block
    /// (the oracle's `nr{i}_out` [n_img_p·batch rows] and `l{i}_out`
    /// [S·batch rows] taps, f32). Debug path: one submit per block.
    pub fn run_taps(&self, out: &mut [f32]) -> Option<Vec<(String, Vec<f32>)>> {
        let c = zctx()?;
        let mut taps = Vec::new();
        let rec = |calls: &ZCalls, from: usize, to: usize| {
            let mut enc = c.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (_, r) in &calls.list[from..to] {
                    match r {
                        Rec::Mm(m) => m.record(&mut pass),
                        Rec::K(k) => k.record(&mut pass),
                    }
                }
            }
            c.queue.submit(Some(enc.finish()));
        };
        let mut from = 0;
        for (i, &e) in self.pre_ends.iter().enumerate() {
            rec(&self.pre_a, from, e);
            from = e;
            let raw = read_bytes(c, &self.img.x, (self.img.m * self.d.h * 4) as u64)?;
            taps.push((format!("nr{i}_out"), bytemuck::cast_slice(&raw).to_vec()));
        }
        // the assembly copies
        let mut enc = c.device.create_command_encoder(&Default::default());
        let h4 = (self.d.h * 4) as u64;
        let mut cap_off = 0u64;
        for (bi, &(off, _)) in self.joint.segs.iter().enumerate() {
            enc.copy_buffer_to_buffer(&self.img.x, (bi * self.n_img_p) as u64 * h4, &self.joint.x, off as u64 * h4, self.n_img_p as u64 * h4);
            let cp = self.n_cap_p[bi] as u64;
            enc.copy_buffer_to_buffer(&self.cap, cap_off * h4, &self.joint.x, (off + self.n_img_p) as u64 * h4, cp * h4);
            cap_off += cp;
        }
        c.queue.submit(Some(enc.finish()));
        let mut from = 0;
        for (i, &e) in self.joint_ends.iter().enumerate() {
            rec(&self.joint_calls, from, e);
            from = e;
            let raw = read_bytes(c, &self.joint.x, (self.joint.m * self.d.h * 4) as u64)?;
            taps.push((format!("l{i}_out"), bytemuck::cast_slice(&raw).to_vec()));
        }
        let mut enc = c.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for f in &self.fin {
                f.record(&mut pass);
            }
        }
        c.queue.submit(Some(enc.finish()));
        let raw = read_bytes(c, &self.out, (self.out_rows * self.d.pd * 4) as u64)?;
        out[..self.out_rows * self.d.pd].copy_from_slice(bytemuck::cast_slice(&raw));
        Some(taps)
    }

    /// Dispatches per forward.
    pub fn dispatches(&self) -> usize {
        self.pre_a.len() + self.joint_calls.len() + self.fin.len()
    }
}

// ════════════════════════════════════════════════════════════════════
// Resident Flux-VAE decoder (plan S6, B2).
//
// The whole decoder in one submission: NHWC activations stay on the
// device, every 3×3 conv is `zi_mm` as an implicit GEMM (the A tile is
// gathered from the NHWC f16 image, K = 9·cin in (tap, channel) order;
// the nearest-2× upsample is folded into that gather), GroupNorm is a
// two-pass f32 reduction (sum, then centred sum of squares) fused with
// the affine, SiLU and the f16 cast of the next conv's input, and the
// mid-block single-head attention is QKᵀ → softmax → P·V as three
// `zi_mm` passes over query chunks. The host uploads the latent once and
// reads back [3][H·W] once.
// ════════════════════════════════════════════════════════════════════

const VAE_GN_PART_SRC: &str = r#"
struct VP { m: u32, c: u32, cpg: u32, rows: u32, nchunk: u32, flags: u32, g: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> stat: array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<uniform> p: VP;
var<workgroup> sh: array<f32, 512>;
// One workgroup per chunk of `rows` pixels, all groups at once (coalesced
// NHWC rows). flags: 1 = add bias[c] on read, 4 = pass 2 (Σ (v − mean)²).
// Thread t always sees channel t % c (c ≤ 256) or channels t, t + 256
// (c = 512).
@compute @workgroup_size(256)
fn vae_gn_part(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let chunk = wid.x;
  let r0 = chunk * p.rows;
  let r1 = min(r0 + p.rows, p.m);
  let n = (r1 - r0) * p.c;
  let base = r0 * p.c;
  let hb = (p.flags & 1u) != 0u;
  let pass2 = (p.flags & 4u) != 0u;
  var a0 = 0.0;
  var a1 = 0.0;
  for (var e = t; e < n; e = e + 256u) {
    let ch = e % p.c;
    var v = x[base + e];
    if (hb) { v = v + bias[ch]; }
    if (pass2) { let d = v - stat[ch / p.cpg]; v = d * d; }
    if (p.c <= 256u || ch == t) { a0 = a0 + v; } else { a1 = a1 + v; }
  }
  sh[t] = a0;
  sh[256u + t] = a1;
  workgroupBarrier();
  if (t < p.g) {
    var s = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
      var c0 = i;
      if (p.c <= 256u) { c0 = i % p.c; }
      if (c0 / p.cpg == t) { s = s + sh[i]; }
      if (p.c > 256u && (i + 256u) / p.cpg == t) { s = s + sh[256u + i]; }
    }
    part[t * p.nchunk + chunk] = s;
  }
}
"#;

const VAE_GN_FIN_SRC: &str = r#"
struct VP { m: u32, c: u32, cpg: u32, rows: u32, nchunk: u32, flags: u32, g: u32, eps: f32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> stat: array<f32>;
@group(0) @binding(2) var<uniform> p: VP;
var<workgroup> sh: array<f32, 256>;
// One workgroup per group: Σ over chunks → mean (pass 1) or rstd (pass 2),
// stat = [mean[g] …, rstd[g] …].
@compute @workgroup_size(256)
fn vae_gn_fin(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let g = wid.x;
  var s = 0.0;
  for (var i = t; i < p.nchunk; i = i + 256u) { s = s + part[g * p.nchunk + i]; }
  sh[t] = s;
  workgroupBarrier();
  var st = 128u;
  loop {
    if (st == 0u) { break; }
    if (t < st) { sh[t] = sh[t] + sh[t + st]; }
    workgroupBarrier();
    st = st >> 1u;
  }
  if (t == 0u) {
    let n = f32(p.m) * f32(p.cpg);
    if ((p.flags & 4u) != 0u) {
      stat[p.g + g] = inverseSqrt(sh[0] / n + p.eps);
    } else {
      stat[g] = sh[0] / n;
    }
  }
}
"#;

const VAE_GN_APPLY_SRC: &str = r#"
enable f16;
struct VP { m: u32, c: u32, cpg: u32, rows: u32, nchunk: u32, flags: u32, g: u32, mp: u32 };
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> stat: array<f32>;
@group(0) @binding(3) var<storage, read> gw: array<f32>;
@group(0) @binding(4) var<storage, read> gb: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<vec2<u32>>;
@group(0) @binding(6) var<uniform> p: VP;
// y = ((x + bias) − mean)·rstd·w + b, then SiLU (flags 2), stored f16;
// rows ≥ m are written as zeros (the conv gathers never read them, but
// the plain GEMMs and the attention keys do).
@compute @workgroup_size(256)
fn vae_gn_apply(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c4 = p.c / 4u;
  let idx = gid.y * (nwg.x * 256u) + gid.x;
  if (idx >= p.mp * c4) { return; }
  let row = idx / c4;
  if (row >= p.m) { outp[idx] = vec2<u32>(0u, 0u); return; }
  let ch = (idx % c4) * 4u;
  var v = x[idx];
  var o = vec4<f32>(0.0);
  for (var j = 0u; j < 4u; j = j + 1u) {
    var a = v[j];
    if ((p.flags & 1u) != 0u) { a = a + bias[ch + j]; }
    let g = (ch + j) / p.cpg;
    var y = (a - stat[g]) * stat[p.g + g] * gw[ch + j] + gb[ch + j];
    if ((p.flags & 2u) != 0u) { y = y / (1.0 + exp(-y)); }
    o[j] = y;
  }
  outp[idx] = vec2<u32>(pack2x16float(o.xy), pack2x16float(o.zw));
}
"#;

const VAE_COMBINE_SRC: &str = r#"
struct CP { n: u32, c: u32, mode: u32, _a: u32 };
@group(0) @binding(0) var<storage, read_write> xo: array<f32>;
@group(0) @binding(1) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read> hb: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read> sb: array<f32>;
@group(0) @binding(5) var<uniform> p: CP;
// mode 0: x += h + hb;  mode 1: x = s + sb + h + hb;  mode 2: x = h + hb.
@compute @workgroup_size(256)
fn vae_combine(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * 256u) + gid.x;
  if (i >= p.n) { return; }
  let ch = i % p.c;
  let hv = h[i] + hb[ch];
  if (p.mode == 0u) { xo[i] = xo[i] + hv; }
  else if (p.mode == 1u) { xo[i] = s[i] + sb[ch] + hv; }
  else { xo[i] = hv; }
}
"#;

const VAE_CAST_SRC: &str = r#"
enable f16;
struct KP { m: u32, mp: u32, c: u32, hasb: u32 };
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> p: KP;
@compute @workgroup_size(256)
fn vae_cast(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c4 = p.c / 4u;
  let idx = gid.y * (nwg.x * 256u) + gid.x;
  if (idx >= p.mp * c4) { return; }
  if (idx / c4 >= p.m) { outp[idx] = vec2<u32>(0u, 0u); return; }
  var v = x[idx];
  if (p.hasb != 0u) {
    let ch = (idx % c4) * 4u;
    v = v + vec4<f32>(b[ch], b[ch + 1u], b[ch + 2u], b[ch + 3u]);
  }
  outp[idx] = vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}
"#;

const VAE_SOFTMAX_SRC: &str = r#"
enable f16;
struct SP { ld: u32, nv: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> sc: array<f32>;
@group(0) @binding(1) var<storage, read_write> pr: array<u32>;
@group(0) @binding(2) var<uniform> p: SP;
var<workgroup> red: array<f32, 256>;
// One workgroup per score row: softmax over the first nv columns in f32,
// P stored f16 (columns ≥ nv = 0).
@compute @workgroup_size(256)
fn vae_softmax(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wid.x + wid.y * 65535u;
  let base = row * p.ld;
  var mx = -3.0e38;
  for (var j = t; j < p.nv; j = j + 256u) { mx = max(mx, sc[base + j]); }
  red[t] = mx;
  workgroupBarrier();
  var st = 128u;
  loop {
    if (st == 0u) { break; }
    if (t < st) { red[t] = max(red[t], red[t + st]); }
    workgroupBarrier();
    st = st >> 1u;
  }
  let m = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var j = t; j < p.nv; j = j + 256u) { s = s + exp(sc[base + j] - m); }
  red[t] = s;
  workgroupBarrier();
  st = 128u;
  loop {
    if (st == 0u) { break; }
    if (t < st) { red[t] = red[t] + red[t + st]; }
    workgroupBarrier();
    st = st >> 1u;
  }
  let inv = 1.0 / red[0];
  for (var j = 2u * t; j < p.ld; j = j + 512u) {
    var a = 0.0;
    var b = 0.0;
    if (j < p.nv) { a = exp(sc[base + j] - m) * inv; }
    if (j + 1u < p.nv) { b = exp(sc[base + j + 1u] - m) * inv; }
    pr[(base + j) / 2u] = pack2x16float(vec2<f32>(a, b));
  }
}
"#;

const VAE_OUT_SRC: &str = r#"
struct OP { m: u32, ld: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
@group(0) @binding(3) var<uniform> p: OP;
// NHWC [m][ld] (first 3 channels) + bias → NCHW [3][m].
@compute @workgroup_size(256)
fn vae_out(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.y * (nwg.x * 256u) + gid.x;
  if (i >= p.m) { return; }
  for (var ch = 0u; ch < 3u; ch = ch + 1u) {
    outp[ch * p.m + i] = y[i * p.ld + ch] + b[ch];
  }
}
"#;

/// One conv's device weights: an f16 plane [cout_p][k²·cin_p] in (tap,
/// channel) order (zero rows/columns for the padding) and an f32 bias
/// [cout_p].
struct VConv {
    plane: wgpu::Buffer,
    bias: wgpu::Buffer,
    cin_p: usize,
    cout: usize,
    cout_p: usize,
    k: usize,
}

struct VNorm {
    w: wgpu::Buffer,
    b: wgpu::Buffer,
    groups: usize,
}

struct VRes {
    n1: VNorm,
    c1: VConv,
    n2: VNorm,
    c2: VConv,
    sc: Option<VConv>,
}

/// The attention's linears as 1×1 convs (planes [c][c]).
struct VAttn {
    norm: VNorm,
    q: VConv,
    k: VConv,
    v: VConv,
    o: VConv,
    c: usize,
}

struct VaeDev {
    key: u64,
    conv_in: VConv,
    mid1: VRes,
    attn: VAttn,
    mid2: VRes,
    ups: Vec<(Vec<VRes>, Option<VConv>)>,
    norm_out: VNorm,
    conv_out: VConv,
}

static ZVAE: Mutex<Option<VaeDev>> = Mutex::new(None);

fn vconv(c: &Ctx, r: &crate::vae::VaeConvRef) -> Option<VConv> {
    let (ic, oc, k) = (r.ic, r.oc, r.k);
    if (k != 1 && k != 3) || r.w.len() < oc * ic * k * k || r.b.len() < oc {
        return None;
    }
    let cin_p = ic.next_multiple_of(32);
    let cout_p = oc.next_multiple_of(128);
    let kk = k * k;
    let kd = kk * cin_p;
    let mut plane = vec![0u16; cout_p * kd];
    for o in 0..oc {
        for ci in 0..ic {
            for tap in 0..kk {
                plane[o * kd + tap * cin_p + ci] = cortiq_core::quant::f32_to_f16(r.w[(o * ic + ci) * kk + tap]);
            }
        }
    }
    let mut bias = vec![0f32; cout_p];
    bias[..oc].copy_from_slice(&r.b[..oc]);
    Some(VConv {
        plane: sbuf_init(c, bytemuck::cast_slice(&plane), "zv_plane"),
        bias: sbuf_init(c, bytemuck::cast_slice(&bias), "zv_bias"),
        cin_p,
        cout: oc,
        cout_p,
        k,
    })
}

fn vlin(c: &Ctx, w: &[f32], b: &[f32], n: usize) -> Option<VConv> {
    vconv(c, &crate::vae::VaeConvRef { w, b, oc: n, ic: n, k: 1 })
}

fn vnorm(c: &Ctx, r: &crate::vae::VaeNormRef) -> VNorm {
    VNorm {
        w: sbuf_init(c, bytemuck::cast_slice(r.w), "zv_gw"),
        b: sbuf_init(c, bytemuck::cast_slice(r.b), "zv_gb"),
        groups: r.groups,
    }
}

fn vres(c: &Ctx, r: &crate::vae::VaeResnetRef) -> Option<VRes> {
    Some(VRes {
        n1: vnorm(c, &r.norm1),
        c1: vconv(c, &r.conv1)?,
        n2: vnorm(c, &r.norm2),
        c2: vconv(c, &r.conv2)?,
        sc: match &r.shortcut {
            Some(s) => Some(vconv(c, s)?),
            None => None,
        },
    })
}

impl VaeDev {
    fn build(c: &Ctx, a: &crate::vae::VaeChainArgs) -> Option<VaeDev> {
        let at = &a.mid_attn;
        Some(VaeDev {
            key: a.key,
            conv_in: vconv(c, &a.conv_in)?,
            mid1: vres(c, &a.mid_res1)?,
            attn: VAttn {
                norm: vnorm(c, &at.norm),
                q: vlin(c, at.q.0, at.q.1, at.c)?,
                k: vlin(c, at.k.0, at.k.1, at.c)?,
                v: vlin(c, at.v.0, at.v.1, at.c)?,
                o: vlin(c, at.out.0, at.out.1, at.c)?,
                c: at.c,
            },
            mid2: vres(c, &a.mid_res2)?,
            ups: a
                .ups
                .iter()
                .map(|u| -> Option<(Vec<VRes>, Option<VConv>)> {
                    Some((
                        u.resnets.iter().map(|r| vres(c, r)).collect::<Option<Vec<_>>>()?,
                        match &u.upsample {
                            Some(s) => Some(vconv(c, s)?),
                            None => None,
                        },
                    ))
                })
                .collect::<Option<Vec<_>>>()?,
            norm_out: vnorm(c, &a.norm_out),
            conv_out: vconv(c, &a.conv_out)?,
        })
    }
}

/// Dispatch grid for `n` threads of 256.
fn grid1(n: usize) -> (u32, u32, u32) {
    let wgs = (n as u32).div_ceil(256).max(1);
    (wgs.min(65535), wgs.div_ceil(65535), 1)
}

/// The recorder of one decode: every buffer it binds and the calls.
struct VRec<'a> {
    c: &'a Ctx,
    calls: ZCalls,
    dummy: wgpu::Buffer,
    stat: wgpu::Buffer,
    part: wgpu::Buffer,
    nchunk_max: usize,
}

const GN_ROWS: usize = 1024;

impl VRec<'_> {
    fn k(&mut self, key: &str, src: &str, entry: &str, bufs: &[&wgpu::Buffer], grid: (u32, u32, u32)) -> Option<()> {
        let pipe = pipeline(self.c, key, src, entry)?;
        let b = bg(self.c, &pipe, bufs);
        self.calls.push(Class::Io, Call { pipe, bg: b, grid });
        Some(())
    }

    /// GroupNorm (+ optional bias on read) of x [m][ch] → f16 `out`
    /// [mp][ch], SiLU when `silu`.
    #[allow(clippy::too_many_arguments)]
    fn gn(&mut self, x: &wgpu::Buffer, bias: Option<&wgpu::Buffer>, n: &VNorm, m: usize, mp: usize, ch: usize, silu: bool, out: &wgpu::Buffer) -> Option<()> {
        let g = n.groups;
        if ch % g != 0 || g > 256 || ch % 4 != 0 || !(ch <= 256 && 256 % ch == 0 || ch == 512) {
            return None;
        }
        let cpg = ch / g;
        let nchunk = m.div_ceil(GN_ROWS);
        if nchunk > self.nchunk_max {
            return None;
        }
        let hb = bias.is_some() as u32;
        let d = self.dummy.clone();
        let bb = bias.unwrap_or(&d);
        for pass2 in [false, true] {
            let flags = hb | if pass2 { 4 } else { 0 };
            let u = ubuf(self.c, &[m as u32, ch as u32, cpg as u32, GN_ROWS as u32, nchunk as u32, flags, g as u32, 0]);
            let (stat, part) = (self.stat.clone(), self.part.clone());
            self.k("zv_gn_part", VAE_GN_PART_SRC, "vae_gn_part", &[x, bb, &stat, &part, &u], (nchunk as u32, 1, 1))?;
            let u = ubuf(self.c, &[m as u32, ch as u32, cpg as u32, GN_ROWS as u32, nchunk as u32, flags, g as u32, 1e-6f32.to_bits()]);
            self.k("zv_gn_fin", VAE_GN_FIN_SRC, "vae_gn_fin", &[&part, &stat, &u], (g as u32, 1, 1))?;
        }
        let flags = hb | if silu { 2 } else { 0 };
        let u = ubuf(self.c, &[m as u32, ch as u32, cpg as u32, 0, 0, flags, g as u32, mp as u32]);
        let stat = self.stat.clone();
        self.k("zv_gn_apply", VAE_GN_APPLY_SRC, "vae_gn_apply", &[x, bb, &stat, &n.w, &n.b, out, &u], grid1(mp * ch / 4))
    }

    /// conv of the f16 NHWC image `act` [(h·w or h/2·w/2)][cin_p] → raw
    /// (bias-free) f32 [mp][cout_p]. `up` = the input is at half size.
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, cv: &VConv, act: &wgpu::Buffer, h: usize, w: usize, up: bool, out: &wgpu::Buffer) -> Option<()> {
        let m = h * w;
        let conv = match (cv.k, up) {
            (3, false) => 1,
            (3, true) => 2,
            (1, false) => 0,
            _ => return None,
        };
        let g = MmCfg { conv, ..default_cfg(Epi::F32) };
        let a = MmArgs {
            m: m as u32,
            n: cv.cout_p as u32,
            k: (cv.k * cv.k * cv.cin_p) as u32,
            ldo: cv.cout_p as u32,
            ocol: 0,
            arow: 0,
            oscale: 1.0,
            conv: [w as u32, h as u32, cv.cin_p as u32],
        };
        let mc = mm_call(self.c, g, &a, &cv.plane, act, out)?;
        self.calls.push_mm(Class::Io, mc);
        Some(())
    }

    #[allow(clippy::too_many_arguments)]
    fn combine(&mut self, x: &wgpu::Buffer, h: &wgpu::Buffer, hb: &wgpu::Buffer, s: Option<(&wgpu::Buffer, &wgpu::Buffer)>, mode: u32, n: usize, ch: usize) -> Option<()> {
        let u = ubuf(self.c, &[n as u32, ch as u32, mode, 0]);
        let d = self.dummy.clone();
        let (sb, sbb) = s.unwrap_or((&d, &d));
        self.k("zv_combine", VAE_COMBINE_SRC, "vae_combine", &[x, h, hb, sb, sbb, &u], grid1(n))
    }

    fn cast(&mut self, x: &wgpu::Buffer, b: Option<&wgpu::Buffer>, m: usize, mp: usize, ch: usize, out: &wgpu::Buffer) -> Option<()> {
        let u = ubuf(self.c, &[m as u32, mp as u32, ch as u32, b.is_some() as u32]);
        let d = self.dummy.clone();
        self.k("zv_cast", VAE_CAST_SRC, "vae_cast", &[x, b.unwrap_or(&d), out, &u], grid1(mp * ch / 4))
    }
}

/// Activation buffers of one decode (NHWC, rows padded to 128).
struct VBufs {
    x: wgpu::Buffer,
    h: wgpu::Buffer,
    s: wgpu::Buffer,
    xn: wgpu::Buffer,
}

fn mp_of(m: usize) -> usize {
    m.next_multiple_of(128)
}

/// One resnet on x [h·w][ic] (f32) → x [h·w][oc].
fn vae_resnet(r: &mut VRec, b: &VBufs, rs: &VRes, h: usize, w: usize, ic: usize) -> Option<usize> {
    let (m, mp) = (h * w, mp_of(h * w));
    let oc = rs.c1.cout;
    if rs.c1.cin_p != ic || rs.c1.cout_p != oc || rs.c2.cout_p != oc {
        return None;
    }
    r.gn(&b.x, None, &rs.n1, m, mp, ic, true, &b.xn)?;
    r.conv(&rs.c1, &b.xn, h, w, false, &b.h)?;
    r.gn(&b.h, Some(&rs.c1.bias), &rs.n2, m, mp, oc, true, &b.xn)?;
    r.conv(&rs.c2, &b.xn, h, w, false, &b.h)?;
    match &rs.sc {
        Some(sc) => {
            if sc.cin_p != ic || sc.cout_p != oc {
                return None;
            }
            r.cast(&b.x, None, m, mp, ic, &b.xn)?;
            r.conv(sc, &b.xn, h, w, false, &b.s)?;
            r.combine(&b.x, &b.h, &rs.c2.bias, Some((&b.s, &sc.bias)), 1, mp * oc, oc)?;
        }
        None => {
            if ic != oc {
                return None;
            }
            r.combine(&b.x, &b.h, &rs.c2.bias, None, 0, mp * oc, oc)?;
        }
    }
    Some(oc)
}

/// Resident decode; see the section comment. `None` = declined (nothing
/// was written to `out`).
fn vae_decode_dev(a: &crate::vae::VaeChainArgs, z: &[f32], h0: usize, w0: usize, out: &mut [f32]) -> Option<()> {
    let c = zctx()?;
    let t0 = std::time::Instant::now();
    let lc = a.latent_channels;
    if z.len() < lc * h0 * w0 || out.len() < 3 * 64 * h0 * w0 || lc > 32 {
        return None;
    }
    let mut g = ZVAE.lock().ok()?;
    if !g.as_ref().is_some_and(|v| v.key == a.key) {
        *g = None;
        *g = Some(VaeDev::build(c, a)?);
        prof("vae weights", t0);
    }
    let t0 = std::time::Instant::now();
    let v = g.as_ref().unwrap();
    // Walk the shapes once: the largest NHWC tensor sizes every buffer.
    let mut elems = mp_of(h0 * w0) * 512;
    let (mut hh, mut ww) = (h0, w0);
    for (res, upc) in &v.ups {
        for r in res {
            elems = elems.max(mp_of(hh * ww) * r.c1.cin_p.max(r.c1.cout_p));
        }
        if let Some(u) = upc {
            hh *= 2;
            ww *= 2;
            elems = elems.max(mp_of(hh * ww) * u.cout_p);
        }
    }
    elems = elems.max(mp_of(hh * ww) * v.conv_out.cout_p);
    let lim = c.device.limits();
    let maxb = lim.max_storage_buffer_binding_size.min(lim.max_buffer_size);
    if (elems * 4) as u64 > maxb {
        prof(&format!("vae declined: {} MB tensor over the {} MB binding limit", elems * 4 >> 20, maxb >> 20), t0);
        return None;
    }
    let bufs = VBufs {
        x: sbuf(c, (elems * 4) as u64, "zv_x"),
        h: sbuf(c, (elems * 4) as u64, "zv_h"),
        s: sbuf(c, (elems * 4) as u64, "zv_s"),
        xn: sbuf(c, (elems * 2) as u64, "zv_xn"),
    };
    let nchunk_max = (elems / 128).div_ceil(GN_ROWS) + 1;
    let mut r = VRec {
        c,
        calls: ZCalls::default(),
        dummy: sbuf(c, 64, "zv_dummy"),
        stat: sbuf(c, 1024 * 4, "zv_stat"),
        part: sbuf(c, (512 * nchunk_max * 4) as u64, "zv_part"),
        nchunk_max,
    };
    // conv_in input: NHWC f16 with the channels padded to cin_p.
    let cin0 = v.conv_in.cin_p;
    let (m0, mp0) = (h0 * w0, mp_of(h0 * w0));
    let mut zin = vec![0u16; mp0 * cin0];
    for ch in 0..lc {
        for p in 0..m0 {
            zin[p * cin0 + ch] = cortiq_core::quant::f32_to_f16(z[ch * m0 + p]);
        }
    }
    let zbuf = sbuf_init(c, bytemuck::cast_slice(&zin), "zv_z");
    r.conv(&v.conv_in, &zbuf, h0, w0, false, &bufs.h)?;
    let mut ch = v.conv_in.cout_p;
    r.combine(&bufs.x, &bufs.h, &v.conv_in.bias, None, 2, mp0 * ch, ch)?;
    ch = vae_resnet(&mut r, &bufs, &v.mid1, h0, w0, ch)?;
    // ── mid attention (single head over the h0·w0 grid)
    {
        let at = &v.attn;
        let cc = at.c;
        if cc != ch || cc % 128 != 0 {
            return None;
        }
        let (m, mp) = (m0, mp0);
        r.gn(&bufs.x, None, &at.norm, m, mp, cc, false, &bufs.xn)?;
        let q16 = sbuf(c, (mp * cc * 2) as u64, "zv_q");
        let k16 = sbuf(c, (mp * cc * 2) as u64, "zv_k");
        let vt16 = sbuf(c, (cc * mp * 2) as u64, "zv_vt");
        let o32 = sbuf(c, (mp * cc * 4) as u64, "zv_o");
        r.conv(&at.q, &bufs.xn, m, 1, false, &bufs.h)?;
        r.cast(&bufs.h, Some(&at.q.bias), m, mp, cc, &q16)?;
        r.conv(&at.k, &bufs.xn, m, 1, false, &bufs.h)?;
        r.cast(&bufs.h, Some(&at.k.bias), m, mp, cc, &k16)?;
        // Vᵀ [c][mp] = Wv · xnᵀ (the bias is added after P·V: rows of P
        // sum to one).
        {
            let a = MmArgs { m: cc as u32, n: mp as u32, k: cc as u32, ldo: mp as u32, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
            let mc = mm_call(c, default_cfg(Epi::F16), &a, &bufs.xn, &at.v.plane, &vt16)?;
            r.calls.push_mm(Class::Io, mc);
        }
        // Query chunks: S [rows][mp] f32 ≤ 128 MB.
        let rows = ((32usize << 20) / mp).clamp(128, mp) / 128 * 128;
        let sc = sbuf(c, (rows * mp * 4) as u64, "zv_sc");
        let pr = sbuf(c, (rows * mp * 2) as u64, "zv_p");
        let scale = 1.0 / (cc as f32).sqrt();
        let mut q0 = 0;
        while q0 < mp {
            let rq = rows.min(mp - q0);
            let a = MmArgs { m: rq as u32, n: mp as u32, k: cc as u32, ldo: mp as u32, ocol: 0, arow: q0 as u32, oscale: scale, conv: [0; 3] };
            let mc = mm_call(c, default_cfg(Epi::F32), &a, &k16, &q16, &sc)?;
            r.calls.push_mm(Class::Io, mc);
            let u = ubuf(c, &[mp as u32, m as u32, 0, 0]);
            r.k("zv_softmax", VAE_SOFTMAX_SRC, "vae_softmax", &[&sc, &pr, &u], ((rq as u32).min(65535), (rq as u32).div_ceil(65535), 1))?;
            // O rows q0.. = P · V (output bound at the chunk's row offset).
            let a = MmArgs { m: rq as u32, n: cc as u32, k: mp as u32, ldo: cc as u32, ocol: 0, arow: 0, oscale: 1.0, conv: [0; 3] };
            let g = default_cfg(Epi::F32);
            if a.n % g.bn != 0 || a.k % g.bk != 0 {
                return None;
            }
            let pipe = mm_pipe(c, g)?;
            let u = mm_uniform(c, &a);
            let off = (q0 * cc * 4) as u64;
            let bgr = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("zv_pv"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    super::bind_buf(0, &vt16),
                    super::bind_buf(1, &pr),
                    super::bind_buf_off(2, &o32, off, (rq * cc * 4) as u64),
                    super::bind_buf(3, &u),
                ],
            });
            r.calls.push_mm(Class::Io, MmCall { pipe, bg: bgr, grid: (a.n / g.bn, a.m.div_ceil(g.bm)) });
            q0 += rq;
        }
        r.cast(&o32, Some(&at.v.bias), m, mp, cc, &bufs.xn)?;
        r.conv(&at.o, &bufs.xn, m, 1, false, &bufs.h)?;
        r.combine(&bufs.x, &bufs.h, &at.o.bias, None, 0, mp * cc, cc)?;
    }
    ch = vae_resnet(&mut r, &bufs, &v.mid2, h0, w0, ch)?;
    let (mut hh, mut ww) = (h0, w0);
    for (res, upc) in &v.ups {
        for rs in res {
            ch = vae_resnet(&mut r, &bufs, rs, hh, ww, ch)?;
        }
        if let Some(u) = upc {
            if u.cin_p != ch {
                return None;
            }
            r.cast(&bufs.x, None, hh * ww, mp_of(hh * ww), ch, &bufs.xn)?;
            hh *= 2;
            ww *= 2;
            r.conv(u, &bufs.xn, hh, ww, true, &bufs.h)?;
            ch = u.cout_p;
            r.combine(&bufs.x, &bufs.h, &u.bias, None, 2, mp_of(hh * ww) * ch, ch)?;
        }
    }
    let (m, mp) = (hh * ww, mp_of(hh * ww));
    if v.conv_out.cin_p != ch || v.conv_out.cout != 3 {
        return None;
    }
    r.gn(&bufs.x, None, &v.norm_out, m, mp, ch, true, &bufs.xn)?;
    r.conv(&v.conv_out, &bufs.xn, hh, ww, false, &bufs.h)?;
    let ob = sbuf(c, (3 * m * 4) as u64, "zv_out");
    let u = ubuf(c, &[m as u32, v.conv_out.cout_p as u32, 0, 0]);
    r.k("zv_out", VAE_OUT_SRC, "vae_out", &[&bufs.h, &v.conv_out.bias, &ob, &u], grid1(m))?;
    prof(&format!("vae record ({} dispatches)", r.calls.len()), t0);
    r.calls.run()?;
    let raw = read_bytes(c, &ob, (3 * m * 4) as u64)?;
    out[..3 * m].copy_from_slice(bytemuck::cast_slice(&raw));
    prof("vae decode", t0);
    Some(())
}
