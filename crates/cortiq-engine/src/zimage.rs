//! Z-Image-Turbo DiT (Tongyi-MAI, 6.15B single-stream DiT) — the CPU-exact
//! reference path and the host glue around the `gpu::zimage_*` contract.
//! OWNER: WP1 (plan §5). WP0 state: public API skeleton, bodies `todo!()`.
//!
//! New code, NOT a mode of Lumina (`dit.rs` stays bit-identical): helpers
//! are copied, never shared. Sequence order is diffusers' [img, cap], so
//! every intermediate lines up row-for-row with the oracle dumps.
//!
//! Semantics (zimage-spec.md): adaLN has NO SiLU before it (the final
//! layer does); t is scaled by 1000; RoPE θ = 256, axes [32, 48, 48],
//! complex-interleaved; caption ids (1+j, 0, 0) over the padded L_p,
//! image ids (L_p+1, r, c), image pad ids (0, 0, 0); pad rows are real
//! keys/queries (no mask at batch 1); caption pad rows := cap_pad_token
//! after the embedder, image pad rows := x_pad_token after the embed.
//!
//! Per image the host precomputes everything latent-independent: caption
//! embed + context refiner (per prompt), the modulation of all steps, the
//! rope tables (per prompt and resolution). Per step the device (or
//! `step_cpu`) runs exactly x_embed → 2 noise refiners → 30 layers → final.
#![allow(dead_code, unused_variables)]

use crate::gpu::{ZBlockRef, ZGeom};
use cortiq_core::CmfModel;
use std::path::Path;
use std::sync::Arc;

/// Sequence padding multiple (diffusers `SEQ_MULTI_OF`).
pub const SEQ_MULTI_OF: usize = 32;
/// Default scheduler shift (static, `use_dynamic_shifting: false`).
pub const DEFAULT_SHIFT: f32 = 3.0;
/// Default number of DiT forwards (Turbo).
pub const DEFAULT_STEPS: usize = 8;

/// The Turbo geometry — for random-weight kernel tests in WP2/WP3 that
/// must not wait for a container.
pub const ZGEOM_TURBO: ZGeom = ZGeom {
    hidden: 3840,
    nh: 30,
    hd: 128,
    inter: 10240,
    eps: 1e-5,
    final_eps: 1e-6,
    patch_dim: 64,
};

/// `dit.config_json` / transformer `config.json` fields the runtime uses.
#[derive(Clone, Debug)]
pub struct ZConfig {
    /// 3840
    pub dim: usize,
    /// 30 main layers
    pub n_layers: usize,
    /// 2 noise-refiner and 2 context-refiner blocks
    pub n_refiner: usize,
    /// 30 heads, 30 kv heads, head_dim 128
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// int(dim / 3 · 8) = 10240
    pub ffn_dim: usize,
    /// 2560 (Qwen3-4B hidden)
    pub cap_feat_dim: usize,
    /// 256 (`ADALN_EMBED_DIM`) and the t-embedder MLP width 1024
    pub t_embed_dim: usize,
    pub t_hidden: usize,
    /// patch 2, latent channels 16 → patch vector 64
    pub patch: usize,
    pub in_channels: usize,
    /// 1e-5 (all RMSNorms), 1e-6 (final LayerNorm)
    pub norm_eps: f32,
    pub final_eps: f32,
    /// 256.0 and [32, 48, 48]
    pub rope_theta: f64,
    pub axes_dims: [usize; 3],
    /// 1000.0
    pub t_scale: f32,
}

impl ZConfig {
    /// Parse the diffusers transformer `config.json` (the same JSON is
    /// stored as `dit.config_json` in the container).
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        todo!("WP1")
    }

    pub fn geom(&self) -> ZGeom {
        todo!("WP1")
    }
}

/// Token geometry of one (resolution, prompt length).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZShape {
    /// Latent size (H/8, W/8).
    pub h_lat: usize,
    pub w_lat: usize,
    /// Patch grid (H/16, W/16).
    pub grid: (usize, usize),
    /// n_img = grid.0 · grid.1, n_img_p = ceil32(n_img).
    pub n_img: usize,
    pub n_img_p: usize,
    /// Caption tokens L and ceil32(L).
    pub l: usize,
    pub l_p: usize,
}

impl ZShape {
    /// From the image size in pixels (multiples of 16) and the caption length.
    pub fn new(height: usize, width: usize, l: usize) -> Self {
        todo!("WP1")
    }

    /// S = n_img_p + l_p.
    pub fn seq(&self) -> usize {
        todo!("WP1")
    }
}

/// RoPE tables, each `(cos, sin)` of [rows · hd/2] f32.
pub struct ZRope {
    /// Noise refiner, [n_img_p] rows (image ids).
    pub img: (Vec<f32>, Vec<f32>),
    /// Main layers, [n_img_p + l_p] rows ordered [img, cap].
    pub joint: (Vec<f32>, Vec<f32>),
    /// Context refiner, [l_p] rows (caption ids).
    pub cap: (Vec<f32>, Vec<f32>),
}

/// Which block of the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZBlockId {
    NoiseRefiner(usize),
    ContextRefiner(usize),
    Layer(usize),
}

/// Device views of every block (for `ZPrepareArgs` and
/// `gpu::zimage_refine_caption`). Borrowed from a `ZImageDit::from_cmf`.
pub struct ZBlockRefs<'a> {
    pub noise_refiner: Vec<ZBlockRef<'a>>,
    pub context_refiner: Vec<ZBlockRef<'a>>,
    pub layers: Vec<ZBlockRef<'a>>,
}

/// Per-(prompt, resolution) state: the refined caption and the rope
/// tables, plus whether the device accepted `gpu::zimage_prepare`.
pub struct ZPrepared {
    pub key: u64,
    pub shape: ZShape,
    /// [l_p, dim], after cap_embedder, pad-token rows and the context refiner.
    pub cap: Vec<f32>,
    pub rope: ZRope,
    pub device: bool,
}

/// The Z-Image transformer. Tensor names are the ORIGINAL diffusers names
/// under `dit.` (no rename map).
pub struct ZImageDit {
    pub cfg: ZConfig,
    /// The container this was loaded from (None for `load_dir`) — device
    /// paths need tensor indices into it.
    pub model: Option<Arc<CmfModel>>,
    // WP1: weights (Proj-like, copied from dit.rs), norms, pad tokens,
    // t-embedder, adaLN, final layer, per-block tensor indices.
}

impl ZImageDit {
    /// Load from a diffusers `transformer/` directory (bf16/fp32 safetensors).
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        todo!("WP1")
    }

    /// Load from a packaged `.cmf` (`dit.*` tensors + `dit.config_json`).
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        todo!("WP1")
    }

    pub fn geom(&self) -> ZGeom {
        self.cfg.geom()
    }

    /// Timestep embedding for `t_model` = (1000 − 1000σ)/1000: sinusoid of
    /// t·1000 (cos first, f32 args) → mlp.0 → SiLU → mlp.2. Returns [256].
    pub fn temb(&self, t_model: f32) -> Vec<f32> {
        todo!("WP1")
    }

    /// Raw `adaLN_modulation.0(temb)` for every step and block:
    /// [steps][2 + n_layers][4 · dim], chunks [scale_msa, gate_msa,
    /// scale_mlp, gate_mlp] (no +1, no tanh). f64 accumulation.
    pub fn mods_for_steps(&self, t_models: &[f32]) -> Vec<f32> {
        todo!("WP1")
    }

    /// 1 + `all_final_layer.2-1.adaLN_modulation.1`(SiLU(temb)) per step:
    /// [steps][dim].
    pub fn final_scale_for_steps(&self, t_models: &[f32]) -> Vec<f32> {
        todo!("WP1")
    }

    /// Caption features [l, cap_feat_dim] → [l_p, dim]: pad rows are copies
    /// of the last row, RMSNorm(w, 1e-5) → Linear + b, rows ≥ l :=
    /// cap_pad_token.
    pub fn embed_caption(&self, cap_feats: &[f32], l: usize) -> Vec<f32> {
        todo!("WP1")
    }

    /// The two unmodulated context-refiner blocks on the host, in place on
    /// `cap` [l_p, dim].
    pub fn refine_caption_cpu(&self, cap: &mut [f32], rope_cap: (&[f32], &[f32])) {
        todo!("WP1")
    }

    /// One block on the host, in place on `x` [n, dim]. `m` = the block's
    /// raw modulation [4 · dim] (None = unmodulated: s = 0, gate = 1).
    pub fn block_cpu(
        &self,
        blk: ZBlockId,
        x: &mut [f32],
        n: usize,
        rope: (&[f32], &[f32]),
        m: Option<&[f32]>,
    ) {
        todo!("WP1")
    }

    /// Device views of all blocks (requires `from_cmf`).
    pub fn block_refs(&self) -> Option<ZBlockRefs<'_>> {
        todo!("WP1")
    }

    /// Once per (prompt, resolution): caption embed → context refiner
    /// (`gpu::zimage_refine_caption`, else CPU) → rope tables →
    /// `gpu::zimage_prepare` (sets `ZPrepared::device`). `mods_all` =
    /// (mods_for_steps, final_scale_for_steps) forwarded to the backend.
    pub fn prepare(
        &self,
        cap_feats: &[f32],
        shape: ZShape,
        key: u64,
        mods_all: Option<(&[f32], &[f32])>,
    ) -> Result<ZPrepared, String> {
        todo!("WP1")
    }

    /// One DiT forward: `gpu::zimage_step` if `p.device`, else `step_cpu`.
    /// `x_tok` [n_img_p, 64] (pad rows = copies of the last row), `mods`
    /// [(2 + n_layers) · 4 · dim] of this step, `final_scale` [dim].
    /// Returns v [n_img, 64] (before the pipeline's negation).
    pub fn step(
        &self,
        p: &ZPrepared,
        step: usize,
        x_tok: &[f32],
        mods: &[f32],
        final_scale: &[f32],
    ) -> Vec<f32> {
        todo!("WP1")
    }

    /// The host reference forward (WP2/WP3 gate against this):
    /// embed → pad rows := x_pad_token → noise refiner (img only) →
    /// concat [img, cap] → layers → LayerNorm(1e-6)·final_scale → Linear →
    /// image rows.
    pub fn step_cpu(&self, p: &ZPrepared, x_tok: &[f32], mods: &[f32], final_scale: &[f32]) -> Vec<f32> {
        todo!("WP1")
    }
}

/// ceil to a multiple of `SEQ_MULTI_OF`.
pub fn ceil32(n: usize) -> usize {
    todo!("WP1")
}

/// σ schedule in torch f32 order: linspace(1, 1/n, n) (`a + step·i` for
/// i < n/2, `b − step·(n−1−i)` otherwise), then shift·s/(1+(shift−1)·s),
/// then a terminal 0. Length n + 1. N=4, shift 3 → [1, .9, .75, .5, 0].
pub fn sigmas_torch_f32(n: usize, shift: f32) -> Vec<f32> {
    todo!("WP1")
}

/// The pipeline's timestep: (1000 − σ·1000)/1000 in f32.
pub fn t_model(sigma: f32) -> f32 {
    todo!("WP1")
}

/// Position ids and RoPE tables for a patch grid and a caption of `l`
/// tokens (θ, axes from the config; f64 freqs, f32 angles, f32 cos/sin).
pub fn ids_and_rope(grid: (usize, usize), l: usize, theta: f64, axes: [usize; 3]) -> ZRope {
    todo!("WP1")
}

/// latent [c, h, w] → tokens [(h/2)·(w/2), 4c], feature (dy·2+dx)·c + ch.
pub fn patchify(latent: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    todo!("WP1")
}

/// Exact inverse of `patchify`.
pub fn unpatchify(tok: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    todo!("WP1")
}

/// Pad `rows` rows of `width` to `rows_p` by repeating the last row.
pub fn pad_rows_repeat_last(x: &[f32], rows: usize, rows_p: usize, width: usize) -> Vec<f32> {
    todo!("WP1")
}
