//! MiMo-V2 multimodal towers: the companion container and its data holder.
//!
//! # Packaging
//!
//! The text model stays a plain `mimo_v2` CMF. The towers live in a
//! companion file `<stem>.mm.cmf` (header `arch_name` = [`MIMO_MM_ARCH`],
//! `provenance.mimo_mm.base_arch` = [`MIMO_BASE_ARCH`]) written by
//! `cortiq convert --mimo-towers mm-only`. One companion serves every text
//! variant: it binds by configuration (hidden size, the nine pinned special
//! token ids), not by the text file's directory hash. A single-file
//! multimodal CMF (`--mimo-towers multimodal`, header `arch_name` =
//! `mimo_v2`) carries the SAME tensor names beside the text tensors;
//! [`MimoMm::from_model`] accepts either and detects the towers by the
//! presence of the [`MM_CONFIG_BLOB`] tensor.
//!
//! Discovery ([`MimoMm::discover`]): an explicit `--mm PATH`, else
//! `<text-stem>.mm.cmf` beside the text file, else the only `*.mm.cmf` in
//! that directory (two or more is an error: pass `--mm`).
//!
//! # Tensor-name contract
//!
//! Tower tensors keep their checkpoint names; the audio tokenizer's own
//! file (`audio_tokenizer/model.safetensors`) is stored under the
//! `audio_tokenizer.` prefix, encoder only. Shapes are those of the
//! release config (MiMo-V2.6-Flash); [`mimo_tower_inventory`] derives the
//! exact list from any config and is what the loader and the converter
//! check against. Codec column = the default (q4tp) companion; the exact
//! development companion (`--quant f16`) stores every "q4tp" row as the
//! source BF16 instead.
//!
//! Config blobs (U8, raw JSON bytes of the source files):
//!
//! | name | content |
//! |---|---|
//! | `mm.config_json` | the full `config.json` (vision_config, audio_config, processor_config, token ids) |
//! | `audio_tokenizer.config_json` | `audio_tokenizer/config.json` |
//!
//! Vision — 364 tensors (`i` in 0..28; sinks only on the windowed blocks
//! {1-8, 10-17, 19-26}, i.e. every block not in `fullatt_block_indexes`):
//!
//! | name | shape | codec |
//! |---|---|---|
//! | `visual.patch_embed.proj.weight` | [1280, 3, 2, 16, 16] | BF16 (Conv3d, no bias; flatten to [1280, 1536] in (c, t, y, x) order) |
//! | `visual.blocks.{i}.norm1.weight` | [1280] | BF16 |
//! | `visual.blocks.{i}.norm2.weight` | [1280] | BF16 |
//! | `visual.blocks.{i}.attn.qkv.weight` | [3072, 1280] | q4tp (rows [q 32×64 \| k 8×64 \| v 8×64]) |
//! | `visual.blocks.{i}.attn.qkv.bias` | [3072] | BF16 |
//! | `visual.blocks.{i}.attn.proj.weight` | [1280, 2048] | q4tp |
//! | `visual.blocks.{i}.attn.proj.bias` | [1280] | BF16 |
//! | `visual.blocks.{i}.attn.sinks` | [32] | BF16 (windowed blocks only) |
//! | `visual.blocks.{i}.mlp.gate_proj.weight` | [4608, 1280] | q4tp |
//! | `visual.blocks.{i}.mlp.gate_proj.bias` | [4608] | BF16 |
//! | `visual.blocks.{i}.mlp.up_proj.weight` | [4608, 1280] | q4tp |
//! | `visual.blocks.{i}.mlp.up_proj.bias` | [4608] | BF16 |
//! | `visual.blocks.{i}.mlp.down_proj.weight` | [1280, 4608] | q4tp |
//! | `visual.blocks.{i}.mlp.down_proj.bias` | [1280] | BF16 |
//! | `visual.merger.ln_q.weight` | [1280] | BF16 (RMSNorm weight; no bias exists) |
//! | `visual.merger.mlp.0.weight` | [5120, 5120] | q4tp (no bias) |
//! | `visual.merger.mlp.2.weight` | [4096, 5120] | q4tp (no bias) |
//!
//! Audio encoder — 75 tensors (`l` in 0..6) — and speech embeddings — 20
//! tensors (`c` in 0..20):
//!
//! | name | shape | codec |
//! |---|---|---|
//! | `audio_encoder.input_local_transformer.layers.{l}.input_layernorm.weight` | [1024] | BF16 |
//! | `audio_encoder.input_local_transformer.layers.{l}.post_attention_layernorm.weight` | [1024] | BF16 |
//! | `audio_encoder.input_local_transformer.layers.{l}.self_attn.{q,k,v}_proj.weight` | [1024, 1024] | q4tp |
//! | `audio_encoder.input_local_transformer.layers.{l}.self_attn.{q,k,v}_proj.bias` | [1024] | BF16 |
//! | `audio_encoder.input_local_transformer.layers.{l}.self_attn.o_proj.weight` | [1024, 1024] | q4tp (no bias) |
//! | `audio_encoder.input_local_transformer.layers.{l}.mlp.{gate,up}_proj.weight` | [4096, 1024] | q4tp |
//! | `audio_encoder.input_local_transformer.layers.{l}.mlp.down_proj.weight` | [1024, 4096] | q4tp |
//! | `audio_encoder.input_local_transformer.norm.weight` | [1024] | BF16 |
//! | `audio_encoder.projection.mlp.0.weight` | [16384, 4096] | q4tp (no bias) |
//! | `audio_encoder.projection.mlp.2.weight` | [4096, 16384] | q4tp (no bias) |
//! | `speech_embeddings.{c}.weight` | [1280, 1024] | BF16 (gather table, never quantized) |
//!
//! `audio_encoder.input_local_transformer.embed_tokens` does not exist in
//! the checkpoint (the local transformer takes embeddings) and is never
//! required.
//!
//! Audio tokenizer encoder — 389 tensors (`l` in 0..24, `q` in 0..20;
//! codebook rows 1024, 1024, 256, then 128 × 17):
//!
//! | name | shape | codec |
//! |---|---|---|
//! | `audio_tokenizer.encoder.conv1.weight` | [1024, 128, 3] | BF16 |
//! | `audio_tokenizer.encoder.conv1.bias` | [1024] | BF16 |
//! | `audio_tokenizer.encoder.conv2.weight` | [1024, 1024, 3] | BF16 (stride 2) |
//! | `audio_tokenizer.encoder.conv2.bias` | [1024] | BF16 |
//! | `audio_tokenizer.encoder.layers.{l}.self_attn.{q,v,out}_proj.weight` | [1024, 1024] | q4tp |
//! | `audio_tokenizer.encoder.layers.{l}.self_attn.{q,v,out}_proj.bias` | [1024] | BF16 |
//! | `audio_tokenizer.encoder.layers.{l}.self_attn.k_proj.weight` | [1024, 1024] | q4tp (no bias) |
//! | `audio_tokenizer.encoder.layers.{l}.self_attn_layer_norm.{weight,bias}` | [1024] | BF16 |
//! | `audio_tokenizer.encoder.layers.{l}.final_layer_norm.{weight,bias}` | [1024] | BF16 |
//! | `audio_tokenizer.encoder.layers.{l}.fc1.weight` | [4096, 1024] | q4tp |
//! | `audio_tokenizer.encoder.layers.{l}.fc1.bias` | [4096] | BF16 |
//! | `audio_tokenizer.encoder.layers.{l}.fc2.weight` | [1024, 4096] | q4tp |
//! | `audio_tokenizer.encoder.layers.{l}.fc2.bias` | [1024] | BF16 |
//! | `audio_tokenizer.encoder.layer_norm.{weight,bias}` | [1024] | BF16 |
//! | `audio_tokenizer.encoder.down_sample_layer.0.weight` | [1024, 1024, 2] | BF16 (Conv1d k2 s2, no bias) |
//! | `audio_tokenizer.encoder.down_sample_norm.{weight,bias}` | [1024] | BF16 |
//! | `audio_tokenizer.encoder.quantizer.vq.layers.{q}._codebook.embed` | [bins_q, 1024] | F32, bit-exact |
//!
//! Dropped from the tokenizer file: every `decoder.*` tensor and the
//! codebook training state (`cluster_size`, `embed_avg`, `inited`).
//!
//! "BF16" = the source bytes verbatim, bit-exact by construction. F16 is
//! not used: 195 467 of the 1 297.6 M BF16 values the release towers keep
//! do not survive BF16 → F16 (|Δ| ≤ 2^-25 on the visual/audio/speech
//! ones, all below F16's subnormal step). Readers
//! dispatch on the directory dtype, which [`MimoMm::f32`] and
//! [`MimoMm::linear`] do (BF16/F16/F32 → f32 once; q4tp/q8_2f stay mapped).
//!
//! # API for the tower modules
//!
//! ```ignore
//! let mm = MimoMm::attach(&text, None)?.expect("no companion");   // or MimoMm::open(path)?
//! let geo = &mm.vision;                                              // parsed geometry
//! let qkv = mm.linear(&names::vis_block(3, "attn.qkv.weight"), geo.qkv_rows(), geo.hidden)?;
//! let b   = mm.f32_shaped(&names::vis_block(3, "attn.qkv.bias"), &[geo.qkv_rows()])?;
//! let cb  = mm.f32_shaped(&names::at_codebook(0), &[1024, 1024])?;  // F32 RVQ codebook
//! ```

use crate::qtensor::QTensor;
use crate::tokenizer::Tokenizer;
use cortiq_core::format::TensorEntry;
use cortiq_core::{CmfModel, TensorDtype};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Header `arch_name` of the companion file.
pub const MIMO_MM_ARCH: &str = "mimo_v2_mm";
/// The text architecture a companion attaches to.
pub const MIMO_BASE_ARCH: &str = "mimo_v2";
/// U8 tensor holding the full source `config.json`.
pub const MM_CONFIG_BLOB: &str = "mm.config_json";
/// U8 tensor holding `audio_tokenizer/config.json`.
pub const AUDIO_TOKENIZER_CONFIG_BLOB: &str = "audio_tokenizer.config_json";
/// Namespace of the audio tokenizer file's tensors inside a CMF.
pub const AUDIO_TOKENIZER_PREFIX: &str = "audio_tokenizer.";
/// File suffix of the companion.
pub const MM_SUFFIX: &str = ".mm.cmf";

/// The nine special tokens the multimodal prompt layout depends on, with
/// the ids the release pins (config.json + tokenizer_config.json).
pub const MIMO_SPECIAL_TOKENS: [(&str, u32); 9] = [
    ("<|vision_start|>", 151652),
    ("<|vision_end|>", 151653),
    ("<|image_pad|>", 151655),
    ("<|video_pad|>", 151656),
    ("<|audio_pad|>", 151669),
    ("<|mimo_video_start|>", 151670),
    ("<|mimo_video_end|>", 151671),
    ("<|mimo_audio_start|>", 151673),
    ("<|mimo_audio_end|>", 151674),
];

/// The special token ids, by role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MimoTokenIds {
    pub vision_start: u32,
    pub vision_end: u32,
    pub image_pad: u32,
    pub video_pad: u32,
    pub audio_pad: u32,
    pub video_start: u32,
    pub video_end: u32,
    pub audio_start: u32,
    pub audio_end: u32,
}

impl MimoTokenIds {
    /// The release ids ([`MIMO_SPECIAL_TOKENS`]).
    pub const PINNED: Self = Self {
        vision_start: 151652,
        vision_end: 151653,
        image_pad: 151655,
        video_pad: 151656,
        audio_pad: 151669,
        video_start: 151670,
        video_end: 151671,
        audio_start: 151673,
        audio_end: 151674,
    };

    /// `(token spelling, id)` in the order of [`MIMO_SPECIAL_TOKENS`].
    pub fn named(&self) -> [(&'static str, u32); 9] {
        [
            (MIMO_SPECIAL_TOKENS[0].0, self.vision_start),
            (MIMO_SPECIAL_TOKENS[1].0, self.vision_end),
            (MIMO_SPECIAL_TOKENS[2].0, self.image_pad),
            (MIMO_SPECIAL_TOKENS[3].0, self.video_pad),
            (MIMO_SPECIAL_TOKENS[4].0, self.audio_pad),
            (MIMO_SPECIAL_TOKENS[5].0, self.video_start),
            (MIMO_SPECIAL_TOKENS[6].0, self.video_end),
            (MIMO_SPECIAL_TOKENS[7].0, self.audio_start),
            (MIMO_SPECIAL_TOKENS[8].0, self.audio_end),
        ]
    }

    /// Read the ids from a MiMo `config.json` (top-level keys, the video
    /// markers from `processor_config`). Every key is required.
    pub fn from_config(cfg: &Value) -> Result<Self, String> {
        let top = |k: &str| -> Result<u32, String> {
            cfg.get(k)
                .and_then(Value::as_u64)
                .map(|v| v as u32)
                .ok_or_else(|| format!("config.json: missing integer '{k}'"))
        };
        let proc = |k: &str| -> Result<u32, String> {
            cfg.get("processor_config")
                .and_then(|p| p.get(k))
                .and_then(Value::as_u64)
                .map(|v| v as u32)
                .ok_or_else(|| format!("config.json: missing integer 'processor_config.{k}'"))
        };
        Ok(Self {
            vision_start: top("vision_start_token_id")?,
            vision_end: top("vision_end_token_id")?,
            image_pad: top("image_token_id")?,
            video_pad: top("video_token_id")?,
            audio_pad: top("audio_token_id")?,
            video_start: proc("video_start_token_id")?,
            video_end: proc("video_end_token_id")?,
            audio_start: top("audio_start_token_id")?,
            audio_end: top("audio_end_token_id")?,
        })
    }

    /// Hard error unless every id equals the pinned one.
    pub fn check_pinned(&self, what: &str) -> Result<(), String> {
        for ((name, got), (_, want)) in self.named().iter().zip(Self::PINNED.named()) {
            if *got != want {
                return Err(format!(
                    "{what}: special token {name} has id {got}, the MiMo layout pins {want}"
                ));
            }
        }
        Ok(())
    }
}

fn cfg_usize(cfg: &Value, key: &str, what: &str) -> Result<usize, String> {
    match cfg.get(key) {
        Some(Value::Number(n)) => n
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("{what}.{key}: not a non-negative integer")),
        // The release stores two audio keys as strings ("1280", "1024").
        Some(Value::String(s)) => s
            .trim()
            .parse()
            .map_err(|_| format!("{what}.{key}: '{s}' is not an integer")),
        _ => Err(format!("{what}: missing integer '{key}'")),
    }
}

fn cfg_usize_or(cfg: &Value, key: &str, what: &str, default: usize) -> Result<usize, String> {
    if cfg.get(key).is_none_or(Value::is_null) {
        Ok(default)
    } else {
        cfg_usize(cfg, key, what)
    }
}

fn cfg_f64_or(cfg: &Value, key: &str, default: f64) -> f64 {
    match cfg.get(key) {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(default),
        Some(Value::String(s)) => s.trim().parse().unwrap_or(default),
        _ => default,
    }
}

fn cfg_bool_or(cfg: &Value, key: &str, default: bool) -> bool {
    cfg.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn cfg_usize_list(cfg: &Value, key: &str, what: &str) -> Result<Vec<usize>, String> {
    cfg.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{what}: missing list '{key}'"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| format!("{what}.{key}: non-integer entry"))
        })
        .collect()
}

/// `speech_vocab_size` / `speech_zeroemb_idx`: one value for every channel,
/// or a "a-b-c" list (HF `_parse_maybe_list`).
fn per_channel(cfg: &Value, key: &str, channels: usize) -> Result<Vec<usize>, String> {
    let bad = || format!("audio_config.{key}: expected an integer or 'a-b-…' list");
    let one = |s: &str| s.trim().parse::<usize>().map_err(|_| bad());
    match cfg.get(key) {
        Some(Value::Number(n)) => Ok(vec![n.as_u64().ok_or_else(bad)? as usize; channels]),
        Some(Value::String(s)) if s.contains('-') => {
            let v = s.split('-').map(one).collect::<Result<Vec<_>, _>>()?;
            if v.len() != channels {
                return Err(format!(
                    "audio_config.{key}: {} entries for {channels} channels",
                    v.len()
                ));
            }
            Ok(v)
        }
        Some(Value::String(s)) => Ok(vec![one(s)?; channels]),
        _ => Err(bad()),
    }
}

/// ViT geometry (`config.vision_config`, HF `MiMoVisionTransformer`).
#[derive(Clone, Debug, PartialEq)]
pub struct MimoVisionGeom {
    pub depth: usize,
    /// Residual width (1280).
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    /// `qk_channels`, default 64 — NOT hidden/heads (= 40).
    pub head_dim: usize,
    /// LLM hidden the merger projects to (4096).
    pub out_hidden: usize,
    pub in_channels: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    /// Blocks with full (per-frame) attention.
    pub fullatt: Vec<usize>,
    /// `vit_window_attn_types` (−1 full, 0 row order, 1 column order).
    pub window_types: Vec<i64>,
    /// Band half-width `visual_token_window_size` (|i−j| ≤ 64).
    pub band: usize,
    pub use_sink: bool,
    pub rms_eps: f64,
}

impl MimoVisionGeom {
    pub fn from_config(cfg: &Value) -> Result<Self, String> {
        let v = cfg
            .get("vision_config")
            .ok_or("config.json: no vision_config")?;
        let w = "vision_config";
        let depth = cfg_usize(v, "depth", w)?;
        let heads = cfg_usize(v, "num_heads", w)?;
        let window_types: Vec<i64> = match v.get("vit_window_attn_types") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|x| {
                    x.as_i64()
                        .ok_or("vision_config.vit_window_attn_types: non-integer")
                })
                .collect::<Result<_, _>>()?,
            _ => vec![-1; depth],
        };
        if window_types.len() != depth {
            return Err(format!(
                "vision_config: {} vit_window_attn_types for depth {depth}",
                window_types.len()
            ));
        }
        let fullatt = match v.get("fullatt_block_indexes") {
            Some(Value::Array(_)) => cfg_usize_list(v, "fullatt_block_indexes", w)?,
            _ => Vec::new(),
        };
        if let Some(&bad) = fullatt.iter().find(|&&i| i >= depth) {
            return Err(format!(
                "vision_config: full-attention block {bad} >= depth {depth}"
            ));
        }
        let band = match v.get("visual_token_window_size").and_then(Value::as_i64) {
            Some(b) if b > 0 => b as usize,
            _ => return Err("vision_config: visual_token_window_size must be positive".into()),
        };
        let g = Self {
            depth,
            hidden: cfg_usize(v, "hidden_size", w)?,
            intermediate: cfg_usize(v, "intermediate_size", w)?,
            heads,
            kv_heads: cfg_usize_or(v, "num_key_value_heads", w, heads)?,
            head_dim: cfg_usize_or(v, "qk_channels", w, 64)?,
            out_hidden: cfg_usize(v, "out_hidden_size", w)?,
            in_channels: match v.get("in_channels") {
                Some(x) if !x.is_null() => cfg_usize(v, "in_channels", w)?,
                _ => cfg_usize_or(v, "in_chans", w, 3)?,
            },
            patch: cfg_usize(v, "patch_size", w)?,
            temporal_patch: cfg_usize(v, "temporal_patch_size", w)?,
            merge: cfg_usize_or(v, "spatial_merge_size", w, 2)?,
            fullatt,
            window_types,
            band,
            use_sink: cfg_bool_or(v, "use_sink", false),
            rms_eps: cfg_f64_or(v, "rms_norm_eps", 1e-6),
        };
        if g.heads == 0 || g.kv_heads == 0 || g.heads % g.kv_heads != 0 {
            return Err(format!(
                "vision_config: {} heads over {} kv heads",
                g.heads, g.kv_heads
            ));
        }
        if g.head_dim == 0 || g.head_dim % 4 != 0 {
            return Err(format!(
                "vision_config: head_dim {} must be a positive multiple of 4 (2-D RoPE)",
                g.head_dim
            ));
        }
        Ok(g)
    }

    /// Rows of the fused qkv: `(heads + 2·kv_heads)·head_dim`.
    pub fn qkv_rows(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    /// Width of one merge unit: `hidden·merge²`.
    pub fn merge_width(&self) -> usize {
        self.hidden * self.merge * self.merge
    }

    /// Values per patch row: `in_channels·temporal_patch·patch²`.
    pub fn patch_width(&self) -> usize {
        self.in_channels * self.temporal_patch * self.patch * self.patch
    }

    /// Block `i` carries a key-0 sink (every non-full block when `use_sink`).
    pub fn has_sink(&self, i: usize) -> bool {
        self.use_sink && !self.fullatt.contains(&i)
    }
}

/// Audio patch encoder geometry (`config.audio_config`, HF `MiMoAudioEncoder`).
#[derive(Clone, Debug, PartialEq)]
pub struct MimoAudioGeom {
    /// RVQ channels summed per frame (20).
    pub channels: usize,
    /// Frames per LLM token (4).
    pub group_size: usize,
    /// Local transformer width (1024).
    pub local_dim: usize,
    pub local_layers: usize,
    pub local_heads: usize,
    pub local_head_dim: usize,
    pub local_intermediate: usize,
    pub rope_theta: f64,
    pub partial_rotary: f64,
    /// Bidirectional attention inside a group.
    pub full_attention: bool,
    pub post_norm: bool,
    pub projection_layers: usize,
    /// LLM hidden (4096).
    pub out_hidden: usize,
    pub segment_size: usize,
    /// Rows of each `speech_embeddings.{c}` table.
    pub speech_vocab: Vec<usize>,
    /// `padding_idx` of each table (the "empty" code).
    pub speech_zero: Vec<usize>,
}

impl MimoAudioGeom {
    pub fn from_config(cfg: &Value) -> Result<Self, String> {
        let a = cfg
            .get("audio_config")
            .ok_or("config.json: no audio_config")?;
        let w = "audio_config";
        let channels = cfg_usize(a, "audio_channels", w)?;
        let heads = cfg_usize(a, "input_local_attn_heads", w)?;
        let dim = cfg_usize(a, "input_local_dim", w)?;
        let g = Self {
            channels,
            group_size: cfg_usize(a, "group_size", w)?,
            local_dim: dim,
            local_layers: cfg_usize(a, "input_local_layers", w)?,
            local_heads: heads,
            local_head_dim: cfg_usize_or(a, "input_local_head_dim", w, dim / heads.max(1))?,
            local_intermediate: cfg_usize(a, "input_local_intermediate_size", w)?,
            rope_theta: cfg_f64_or(a, "rope_theta", 640000.0),
            partial_rotary: cfg_f64_or(a, "partial_rotary_factor", 1.0),
            full_attention: cfg_bool_or(a, "input_full_attention", true),
            post_norm: cfg_bool_or(a, "add_post_norm", true),
            projection_layers: cfg_usize_or(a, "projection_layers", w, 2)?,
            out_hidden: cfg_usize(a, "out_hidden_size", w)?,
            segment_size: cfg_usize_or(a, "audio_segment_size", w, 6000)?,
            speech_vocab: per_channel(a, "speech_vocab_size", channels)?,
            speech_zero: per_channel(a, "speech_zeroemb_idx", channels)?,
        };
        if g.local_heads * g.local_head_dim != g.local_dim {
            return Err(format!(
                "audio_config: {} heads × {} != input_local_dim {}",
                g.local_heads, g.local_head_dim, g.local_dim
            ));
        }
        if !matches!(g.projection_layers, 1 | 2) {
            return Err(format!(
                "audio_config: projection_layers {} (expected 1 or 2)",
                g.projection_layers
            ));
        }
        Ok(g)
    }

    /// Input width of the projection: `local_dim·group_size` (4096).
    pub fn proj_in(&self) -> usize {
        self.local_dim * self.group_size
    }
}

/// Audio tokenizer encoder geometry (`audio_tokenizer/config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct MimoAudioTokenizerGeom {
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub n_mels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub avg_pooler: usize,
    /// `encoder_skip_layer_id` as stored (HF saves h after layer index
    /// `skip_layer_id − 1`).
    pub skip_layer_id: usize,
    pub causal: bool,
    /// `encoder_attn_window_size` (left, right).
    pub window: (i64, i64),
    pub hybrid_attention: bool,
    pub swa_per_block: usize,
    pub rope_theta: f64,
    /// Rows of each RVQ codebook.
    pub codebook_sizes: Vec<usize>,
    pub sampling_rate: usize,
    pub hop_length: usize,
    pub nfft: usize,
    pub window_size: usize,
}

impl MimoAudioTokenizerGeom {
    pub fn from_config(at: &Value) -> Result<Self, String> {
        let w = "audio_tokenizer/config.json";
        let win = at
            .get("encoder_attn_window_size")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .and_then(|a| Some((a[0].as_i64()?, a[1].as_i64()?)))
            .ok_or_else(|| format!("{w}: encoder_attn_window_size must be [left, right]"))?;
        let ln = at
            .get("ln_type")
            .and_then(Value::as_str)
            .unwrap_or("LayerNorm");
        if ln != "LayerNorm" {
            return Err(format!("{w}: ln_type '{ln}' (only LayerNorm)"));
        }
        let num_q = cfg_usize(at, "num_quantizers", w)?;
        let codebook_sizes = cfg_usize_list(at, "codebook_size", w)?;
        if codebook_sizes.len() != num_q {
            return Err(format!(
                "{w}: {} codebook sizes for {num_q} quantizers",
                codebook_sizes.len()
            ));
        }
        let g = Self {
            d_model: cfg_usize(at, "d_model", w)?,
            layers: cfg_usize(at, "encoder_layers", w)?,
            heads: cfg_usize(at, "encoder_attention_heads", w)?,
            ffn: cfg_usize(at, "encoder_ffn_dim", w)?,
            n_mels: cfg_usize(at, "n_mels", w)?,
            kernel_size: cfg_usize(at, "kernel_size", w)?,
            stride: cfg_usize(at, "stride_size", w)?,
            avg_pooler: cfg_usize_or(at, "avg_pooler", w, 1)?,
            skip_layer_id: cfg_usize(at, "encoder_skip_layer_id", w)?,
            causal: cfg_bool_or(at, "encoder_causal", false),
            window: win,
            hybrid_attention: cfg_bool_or(at, "hybrid_attention", false),
            swa_per_block: cfg_usize_or(at, "swa_per_block", w, 1)?,
            rope_theta: cfg_f64_or(at, "rope_theta", 10000.0),
            codebook_sizes,
            sampling_rate: cfg_usize(at, "sampling_rate", w)?,
            hop_length: cfg_usize(at, "hop_length", w)?,
            nfft: cfg_usize(at, "nfft", w)?,
            window_size: cfg_usize(at, "window_size", w)?,
        };
        if g.heads == 0 || g.d_model % g.heads != 0 {
            return Err(format!("{w}: d_model {} over {} heads", g.d_model, g.heads));
        }
        if at.get("scale_embedding").and_then(Value::as_bool) == Some(true) {
            return Err(format!("{w}: scale_embedding=true is not supported"));
        }
        Ok(g)
    }
}

/// The four tower groups of a MiMo-V2 checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MimoTowerGroup {
    Vision,
    AudioEncoder,
    SpeechEmbeddings,
    AudioTokenizer,
}

impl MimoTowerGroup {
    pub const ALL: [Self; 4] = [
        Self::Vision,
        Self::AudioEncoder,
        Self::SpeechEmbeddings,
        Self::AudioTokenizer,
    ];

    /// The group a (CMF) tensor name belongs to, if any. Config blobs are
    /// not tower tensors.
    pub fn of(name: &str) -> Option<Self> {
        if name.starts_with("visual.") {
            Some(Self::Vision)
        } else if name.starts_with("audio_encoder.") {
            Some(Self::AudioEncoder)
        } else if name.starts_with("speech_embeddings.") {
            Some(Self::SpeechEmbeddings)
        } else if name.starts_with("audio_tokenizer.encoder.") {
            Some(Self::AudioTokenizer)
        } else {
            None
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Vision => "visual",
            Self::AudioEncoder => "audio_encoder",
            Self::SpeechEmbeddings => "speech_embeddings",
            Self::AudioTokenizer => "audio_tokenizer",
        }
    }
}

/// An RVQ codebook of the audio tokenizer (stored F32, never quantized).
pub fn is_codebook(name: &str) -> bool {
    name.starts_with("audio_tokenizer.encoder.quantizer.") && name.ends_with("._codebook.embed")
}

/// Tensor-name builders (see the module table).
pub mod names {
    use super::AUDIO_TOKENIZER_PREFIX;

    pub const VIS_PATCH_EMBED: &str = "visual.patch_embed.proj.weight";
    pub const VIS_MERGER_NORM: &str = "visual.merger.ln_q.weight";
    pub const VIS_MERGER_FC1: &str = "visual.merger.mlp.0.weight";
    pub const VIS_MERGER_FC2: &str = "visual.merger.mlp.2.weight";
    pub const AUD_LOCAL_NORM: &str = "audio_encoder.input_local_transformer.norm.weight";
    pub const AUD_PROJ_FC1: &str = "audio_encoder.projection.mlp.0.weight";
    pub const AUD_PROJ_FC2: &str = "audio_encoder.projection.mlp.2.weight";
    /// Single-linear projection (`projection_layers == 1`).
    pub const AUD_PROJ_SINGLE: &str = "audio_encoder.projection.weight";

    /// `visual.blocks.{i}.{leaf}`, e.g. `vis_block(3, "attn.qkv.weight")`.
    pub fn vis_block(i: usize, leaf: &str) -> String {
        format!("visual.blocks.{i}.{leaf}")
    }

    /// `audio_encoder.input_local_transformer.layers.{l}.{leaf}`.
    pub fn aud_layer(l: usize, leaf: &str) -> String {
        format!("audio_encoder.input_local_transformer.layers.{l}.{leaf}")
    }

    /// `speech_embeddings.{c}.weight`.
    pub fn speech_embedding(c: usize) -> String {
        format!("speech_embeddings.{c}.weight")
    }

    /// `audio_tokenizer.encoder.{leaf}`, e.g. `at("conv1.weight")`.
    pub fn at(leaf: &str) -> String {
        format!("{AUDIO_TOKENIZER_PREFIX}encoder.{leaf}")
    }

    /// `audio_tokenizer.encoder.layers.{l}.{leaf}`.
    pub fn at_layer(l: usize, leaf: &str) -> String {
        format!("{AUDIO_TOKENIZER_PREFIX}encoder.layers.{l}.{leaf}")
    }

    /// `audio_tokenizer.encoder.quantizer.vq.layers.{q}._codebook.embed`.
    pub fn at_codebook(q: usize) -> String {
        format!("{AUDIO_TOKENIZER_PREFIX}encoder.quantizer.vq.layers.{q}._codebook.embed")
    }
}

/// The exact tower inventory `(name, shape)` implied by the configs, in
/// the order the converter writes it (vision, audio encoder, speech
/// embeddings, audio tokenizer). Release: 364 + 75 + 20 + 389 = 848.
pub fn mimo_tower_inventory(
    config: &Value,
    at_config: &Value,
) -> Result<Vec<(String, Vec<usize>)>, String> {
    let v = MimoVisionGeom::from_config(config)?;
    let a = MimoAudioGeom::from_config(config)?;
    let t = MimoAudioTokenizerGeom::from_config(at_config)?;
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |n: String, s: Vec<usize>| out.push((n, s));

    // Vision.
    push(
        names::VIS_PATCH_EMBED.into(),
        vec![v.hidden, v.in_channels, v.temporal_patch, v.patch, v.patch],
    );
    let (h, i, hd) = (v.hidden, v.intermediate, v.head_dim);
    for b in 0..v.depth {
        let n = |leaf: &str| names::vis_block(b, leaf);
        push(n("norm1.weight"), vec![h]);
        push(n("norm2.weight"), vec![h]);
        push(n("attn.qkv.weight"), vec![v.qkv_rows(), h]);
        push(n("attn.qkv.bias"), vec![v.qkv_rows()]);
        push(n("attn.proj.weight"), vec![h, v.heads * hd]);
        push(n("attn.proj.bias"), vec![h]);
        if v.has_sink(b) {
            push(n("attn.sinks"), vec![v.heads]);
        }
        push(n("mlp.gate_proj.weight"), vec![i, h]);
        push(n("mlp.gate_proj.bias"), vec![i]);
        push(n("mlp.up_proj.weight"), vec![i, h]);
        push(n("mlp.up_proj.bias"), vec![i]);
        push(n("mlp.down_proj.weight"), vec![h, i]);
        push(n("mlp.down_proj.bias"), vec![h]);
    }
    push(names::VIS_MERGER_NORM.into(), vec![h]);
    push(
        names::VIS_MERGER_FC1.into(),
        vec![v.merge_width(), v.merge_width()],
    );
    push(
        names::VIS_MERGER_FC2.into(),
        vec![v.out_hidden, v.merge_width()],
    );

    // Audio patch encoder (a Qwen2 stack: q/k/v biased, o_proj not).
    let (d, ai) = (a.local_dim, a.local_intermediate);
    let qd = a.local_heads * a.local_head_dim;
    for l in 0..a.local_layers {
        let n = |leaf: &str| names::aud_layer(l, leaf);
        push(n("input_layernorm.weight"), vec![d]);
        push(n("post_attention_layernorm.weight"), vec![d]);
        for p in ["q_proj", "k_proj", "v_proj"] {
            push(n(&format!("self_attn.{p}.weight")), vec![qd, d]);
            push(n(&format!("self_attn.{p}.bias")), vec![qd]);
        }
        push(n("self_attn.o_proj.weight"), vec![d, qd]);
        push(n("mlp.gate_proj.weight"), vec![ai, d]);
        push(n("mlp.up_proj.weight"), vec![ai, d]);
        push(n("mlp.down_proj.weight"), vec![d, ai]);
    }
    if a.post_norm {
        push(names::AUD_LOCAL_NORM.into(), vec![d]);
    }
    if a.projection_layers == 2 {
        push(
            names::AUD_PROJ_FC1.into(),
            vec![4 * a.proj_in(), a.proj_in()],
        );
        push(
            names::AUD_PROJ_FC2.into(),
            vec![a.out_hidden, 4 * a.proj_in()],
        );
    } else {
        push(
            names::AUD_PROJ_SINGLE.into(),
            vec![a.out_hidden, a.proj_in()],
        );
    }
    for c in 0..a.channels {
        push(names::speech_embedding(c), vec![a.speech_vocab[c], d]);
    }

    // Audio tokenizer encoder.
    let (m, f) = (t.d_model, t.ffn);
    push(names::at("conv1.weight"), vec![m, t.n_mels, t.kernel_size]);
    push(names::at("conv1.bias"), vec![m]);
    push(names::at("conv2.weight"), vec![m, m, t.kernel_size]);
    push(names::at("conv2.bias"), vec![m]);
    for l in 0..t.layers {
        let n = |leaf: &str| names::at_layer(l, leaf);
        push(n("self_attn.k_proj.weight"), vec![m, m]);
        for p in ["q_proj", "v_proj", "out_proj"] {
            push(n(&format!("self_attn.{p}.weight")), vec![m, m]);
            push(n(&format!("self_attn.{p}.bias")), vec![m]);
        }
        push(n("self_attn_layer_norm.weight"), vec![m]);
        push(n("self_attn_layer_norm.bias"), vec![m]);
        push(n("final_layer_norm.weight"), vec![m]);
        push(n("final_layer_norm.bias"), vec![m]);
        push(n("fc1.weight"), vec![f, m]);
        push(n("fc1.bias"), vec![f]);
        push(n("fc2.weight"), vec![m, f]);
        push(n("fc2.bias"), vec![m]);
    }
    push(names::at("layer_norm.weight"), vec![m]);
    push(names::at("layer_norm.bias"), vec![m]);
    if t.avg_pooler != 1 {
        push(
            names::at("down_sample_layer.0.weight"),
            vec![m, m, t.avg_pooler],
        );
        push(names::at("down_sample_norm.weight"), vec![m]);
        push(names::at("down_sample_norm.bias"), vec![m]);
    }
    for (q, &bins) in t.codebook_sizes.iter().enumerate() {
        push(names::at_codebook(q), vec![bins, m]);
    }
    Ok(out)
}

fn json_blob(model: &CmfModel, name: &str) -> Result<Value, String> {
    let e = model
        .tensor(name)
        .ok_or_else(|| format!("{}: no '{name}' tensor", model.path.display()))?;
    if e.dtype != TensorDtype::U8 {
        return Err(format!("'{name}' is {:?}, expected U8 JSON bytes", e.dtype));
    }
    serde_json::from_slice(model.entry_bytes(e)).map_err(|e| format!("'{name}': {e}"))
}

/// Where the tower tensors came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MimoMmSource {
    /// A `mimo_v2_mm` companion file.
    Companion,
    /// A single-file multimodal `mimo_v2` CMF.
    SingleFile,
}

/// Data holder for the MiMo-V2 towers: the container handle, the parsed
/// configs and geometry, and typed tensor access. Building one validates
/// the whole tower inventory (names, shapes, codebook dtype) and the pinned
/// special ids; [`Self::validate_text`] checks the pairing with a text model.
pub struct MimoMm {
    model: Arc<CmfModel>,
    pub source: MimoMmSource,
    /// The full source `config.json`.
    pub config: Value,
    /// `audio_tokenizer/config.json`.
    pub audio_tokenizer_config: Value,
    pub vision: MimoVisionGeom,
    pub audio: MimoAudioGeom,
    pub audio_tokenizer: MimoAudioTokenizerGeom,
    pub tokens: MimoTokenIds,
    /// LLM hidden size the towers project into (4096).
    pub hidden_size: usize,
    /// `provenance.mimo_mm` as written by the converter (codec per group,
    /// config sha256, source revision), when present.
    pub provenance: Option<Value>,
}

impl std::fmt::Debug for MimoMm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MimoMm")
            .field("path", &self.model.path)
            .field("source", &self.source)
            .field("hidden_size", &self.hidden_size)
            .finish()
    }
}

impl MimoMm {
    /// Open a companion (or single-file multimodal CMF) by path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let model = CmfModel::open(path)
            .map_err(|e| format!("open MiMo companion {}: {e}", path.display()))?;
        Self::from_model(&Arc::new(model))
    }

    /// Build from an open container: a `mimo_v2_mm` companion, or a
    /// `mimo_v2` file that carries the towers. Every inconsistency is an
    /// error; nothing is defaulted.
    pub fn from_model(model: &Arc<CmfModel>) -> Result<Self, String> {
        let arch = model.arch();
        let where_ = model.path.display().to_string();
        let prov = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("mimo_mm"))
            .cloned();
        let source = match arch.arch_name.as_str() {
            MIMO_MM_ARCH => {
                let base = prov
                    .as_ref()
                    .and_then(|p| p.get("base_arch"))
                    .and_then(Value::as_str);
                if base != Some(MIMO_BASE_ARCH) {
                    return Err(format!(
                        "{where_}: companion base_arch {base:?}, expected '{MIMO_BASE_ARCH}'"
                    ));
                }
                MimoMmSource::Companion
            }
            MIMO_BASE_ARCH => {
                if model.tensor(MM_CONFIG_BLOB).is_none() {
                    return Err(format!(
                        "{where_}: text-only mimo_v2 file (no '{MM_CONFIG_BLOB}'); \
                         attach a {MIMO_MM_ARCH} companion instead"
                    ));
                }
                MimoMmSource::SingleFile
            }
            other => {
                return Err(format!(
                    "{where_}: arch '{other}' is neither '{MIMO_MM_ARCH}' nor '{MIMO_BASE_ARCH}'"
                ));
            }
        };
        let config = json_blob(model, MM_CONFIG_BLOB)?;
        let at_config = json_blob(model, AUDIO_TOKENIZER_CONFIG_BLOB)?;
        let mt = config.get("model_type").and_then(Value::as_str);
        if mt != Some(MIMO_BASE_ARCH) {
            return Err(format!(
                "{where_}: {MM_CONFIG_BLOB} model_type {mt:?}, expected '{MIMO_BASE_ARCH}'"
            ));
        }
        let hidden = cfg_usize(&config, "hidden_size", "config.json")?;
        if hidden != arch.hidden_size {
            return Err(format!(
                "{where_}: header hidden_size {} != config hidden_size {hidden}",
                arch.hidden_size
            ));
        }
        let tokens = MimoTokenIds::from_config(&config)?;
        tokens.check_pinned(&where_)?;
        let vision = MimoVisionGeom::from_config(&config)?;
        let audio = MimoAudioGeom::from_config(&config)?;
        let audio_tokenizer = MimoAudioTokenizerGeom::from_config(&at_config)?;
        if vision.out_hidden != hidden || audio.out_hidden != hidden {
            return Err(format!(
                "{where_}: tower outputs (vision {}, audio {}) != LLM hidden {hidden}",
                vision.out_hidden, audio.out_hidden
            ));
        }

        // The inventory: every expected tensor present with its shape, and
        // no unknown tower tensor (a newer or older converter layout).
        let inv = mimo_tower_inventory(&config, &at_config)?;
        let mut missing = Vec::new();
        for (name, shape) in &inv {
            match model.tensor(name) {
                None => missing.push(name.clone()),
                Some(e) if &e.shape != shape => {
                    return Err(format!(
                        "{where_}: '{name}' has shape {:?}, expected {shape:?}",
                        e.shape
                    ));
                }
                Some(e) if is_codebook(name) && e.dtype != TensorDtype::F32 => {
                    return Err(format!(
                        "{where_}: RVQ codebook '{name}' is {:?}; it must be F32",
                        e.dtype
                    ));
                }
                Some(_) => {}
            }
        }
        if !missing.is_empty() {
            return Err(format!(
                "{where_}: {} tower tensors missing, e.g. {:?}",
                missing.len(),
                &missing[..missing.len().min(4)]
            ));
        }
        let known: std::collections::HashSet<&str> = inv.iter().map(|(n, _)| n.as_str()).collect();
        let unknown: Vec<&str> = model
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .filter(|n| MimoTowerGroup::of(n).is_some() && !known.contains(n))
            .collect();
        if !unknown.is_empty() {
            return Err(format!(
                "{where_}: {} unexpected tower tensors, e.g. {:?}",
                unknown.len(),
                &unknown[..unknown.len().min(4)]
            ));
        }
        if source == MimoMmSource::Companion {
            let extra: Vec<&str> = model
                .tensors
                .iter()
                .map(|t| t.name.as_str())
                .filter(|n| {
                    MimoTowerGroup::of(n).is_none()
                        && *n != MM_CONFIG_BLOB
                        && *n != AUDIO_TOKENIZER_CONFIG_BLOB
                })
                .collect();
            if !extra.is_empty() {
                return Err(format!(
                    "{where_}: companion carries non-tower tensors, e.g. {:?}",
                    &extra[..extra.len().min(4)]
                ));
            }
        }
        Ok(Self {
            model: model.clone(),
            source,
            config,
            audio_tokenizer_config: at_config,
            vision,
            audio,
            audio_tokenizer,
            tokens,
            hidden_size: hidden,
            provenance: prov,
        })
    }

    /// Find the companion of a text CMF: `explicit` (must exist), else
    /// `<stem>.mm.cmf` beside it (the stem without `.cmf` and without a
    /// `-NNNNN-of-NNNNN` shard suffix), else the only `*.mm.cmf` in the
    /// directory. `Ok(None)` when there is none; two candidates are an error.
    pub fn discover(text_path: &Path, explicit: Option<&Path>) -> Result<Option<PathBuf>, String> {
        if let Some(p) = explicit {
            if !p.is_file() {
                return Err(format!("--mm {}: no such file", p.display()));
            }
            return Ok(Some(p.to_path_buf()));
        }
        let dir = match text_path.parent() {
            Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let file = text_path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        if file.ends_with(MM_SUFFIX) {
            return Ok(None);
        }
        let stem = file.strip_suffix(".cmf").unwrap_or(&file);
        let stem = strip_shard_suffix(stem);
        let sibling = dir.join(format!("{stem}{MM_SUFFIX}"));
        if sibling.is_file() {
            return Ok(Some(sibling));
        }
        let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| format!("scan {} for {MM_SUFFIX}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.file_name()
                        .is_some_and(|f| f.to_string_lossy().ends_with(MM_SUFFIX))
            })
            .collect();
        found.sort();
        match found.len() {
            0 => Ok(None),
            1 => Ok(found.pop()),
            _ => Err(format!(
                "{} MiMo companions next to {} ({}); pass --mm PATH",
                found.len(),
                text_path.display(),
                found
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }

    /// Towers for a text model: the file's own towers (single-file
    /// multimodal), else a discovered or explicit companion — validated
    /// against the text model ([`Self::validate_text_model`]). `Ok(None)`:
    /// the model is not `mimo_v2`, or no towers exist anywhere. An explicit
    /// `--mm` on a non-MiMo model is an error.
    pub fn attach(text: &Arc<CmfModel>, explicit: Option<&Path>) -> Result<Option<Self>, String> {
        if text.arch().arch_name != MIMO_BASE_ARCH {
            if let Some(p) = explicit {
                return Err(format!(
                    "--mm {}: the model is '{}', not {MIMO_BASE_ARCH}",
                    p.display(),
                    text.arch().arch_name
                ));
            }
            return Ok(None);
        }
        if explicit.is_none() && text.tensor(MM_CONFIG_BLOB).is_some() {
            let mm = Self::from_model(text)?;
            mm.validate_text_model(text)?;
            return Ok(Some(mm));
        }
        let Some(path) = Self::discover(&text.path, explicit)? else {
            return Ok(None);
        };
        let mm = Self::open(&path)?;
        mm.validate_text_model(text)?;
        Ok(Some(mm))
    }

    /// [`Self::validate_text`] with the tokenizer embedded in the text CMF
    /// (a text file without one cannot be checked and is refused).
    pub fn validate_text_model(&self, text: &CmfModel) -> Result<(), String> {
        let vocab = text.vocab.as_deref().ok_or_else(|| {
            format!(
                "{}: no embedded tokenizer — the MiMo special ids cannot be checked",
                text.path.display()
            )
        })?;
        let tok = Tokenizer::from_bytes(vocab)
            .map_err(|e| format!("{}: tokenizer: {e}", text.path.display()))?;
        self.validate_text(text, &tok)
    }

    /// Hard checks that this tower set belongs to `text`: arch `mimo_v2`,
    /// the same hidden size, and a tokenizer that maps the nine special
    /// tokens to the pinned ids.
    pub fn validate_text(&self, text: &CmfModel, tok: &Tokenizer) -> Result<(), String> {
        let a = text.arch();
        let where_ = text.path.display();
        if a.arch_name != MIMO_BASE_ARCH {
            return Err(format!(
                "{where_}: arch '{}' — MiMo towers attach only to {MIMO_BASE_ARCH}",
                a.arch_name
            ));
        }
        if a.hidden_size != self.hidden_size {
            return Err(format!(
                "{where_}: text hidden_size {} != tower hidden_size {} ({})",
                a.hidden_size,
                self.hidden_size,
                self.model.path.display()
            ));
        }
        for (name, want) in self.tokens.named() {
            match tok.token_to_id(name) {
                Some(got) if got == want => {}
                got => {
                    return Err(format!(
                        "{where_}: tokenizer maps {name} to {got:?}, the towers need {want}"
                    ));
                }
            }
            if want as usize >= a.vocab_size {
                return Err(format!(
                    "{where_}: special id {want} ({name}) outside vocab_size {}",
                    a.vocab_size
                ));
            }
        }
        Ok(())
    }

    /// The container the tower tensors live in.
    pub fn model(&self) -> &Arc<CmfModel> {
        &self.model
    }

    pub fn has(&self, name: &str) -> bool {
        self.model.tensor(name).is_some()
    }

    /// Directory entry (dtype, shape) of a tensor.
    pub fn entry(&self, name: &str) -> Result<&TensorEntry, String> {
        self.model
            .tensor(name)
            .ok_or_else(|| format!("{}: missing tensor '{name}'", self.model.path.display()))
    }

    /// Storage dtype of a tensor, if present.
    pub fn dtype(&self, name: &str) -> Option<TensorDtype> {
        self.model.tensor(name).map(|e| e.dtype)
    }

    /// A 2-D weight `[rows, cols]` on the engine's kernels (quantized
    /// payloads stay memory-mapped; F16/F32 dequantize to f32 once).
    pub fn linear(&self, name: &str, rows: usize, cols: usize) -> Result<QTensor, String> {
        let e = self.entry(name)?;
        if e.shape != [rows, cols] {
            return Err(format!(
                "'{name}' has shape {:?}, expected [{rows}, {cols}]",
                e.shape
            ));
        }
        QTensor::from_model(&self.model, name)
    }

    /// As [`Self::linear`], as the tower-side `Proj` (exact f32 GEMM arm for
    /// float payloads).
    #[allow(dead_code)] // the vision/audio tower modules are its callers
    pub(crate) fn proj(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<crate::dit::Proj, String> {
        self.linear(name, rows, cols)?;
        crate::dit::Proj::from_model(&self.model, name)
    }

    /// Any tensor dequantized to f32 (row-major, directory shape).
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        crate::dit::cmf_f32(&self.model, name)
    }

    /// [`Self::f32`] with a shape check.
    pub fn f32_shaped(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let e = self.entry(name)?;
        if e.shape != shape {
            return Err(format!(
                "'{name}' has shape {:?}, expected {shape:?}",
                e.shape
            ));
        }
        self.f32(name)
    }

    /// Codec recorded by the converter for a tower group ("q4tp", "f16", …).
    pub fn group_codec(&self, group: MimoTowerGroup) -> Option<&str> {
        self.provenance
            .as_ref()?
            .get("codec")?
            .get(group.label())?
            .get("matrices")?
            .as_str()
    }
}

/// `name-00001-of-00003` → `name`.
fn strip_shard_suffix(stem: &str) -> &str {
    let b = stem.as_bytes();
    // "-NNNNN-of-NNNNN" is 15 bytes.
    if b.len() > 15 {
        let tail = &stem[stem.len() - 15..];
        let t = tail.as_bytes();
        let digits = |r: std::ops::Range<usize>| t[r].iter().all(u8::is_ascii_digit);
        if t[0] == b'-' && digits(1..6) && &tail[6..10] == "-of-" && digits(10..15) {
            return &stem[..stem.len() - 15];
        }
    }
    stem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_suffix_is_stripped_only_when_exact() {
        assert_eq!(strip_shard_suffix("mimo-q4tp-00001-of-00004"), "mimo-q4tp");
        assert_eq!(strip_shard_suffix("mimo-q4tp"), "mimo-q4tp");
        assert_eq!(strip_shard_suffix("x-0001-of-00004"), "x-0001-of-00004");
    }

    #[test]
    fn per_channel_accepts_scalar_string_and_list() {
        let c = serde_json::json!({"a": "1280", "b": 7, "c": "1-2-3"});
        assert_eq!(per_channel(&c, "a", 2).unwrap(), vec![1280, 1280]);
        assert_eq!(per_channel(&c, "b", 3).unwrap(), vec![7, 7, 7]);
        assert_eq!(per_channel(&c, "c", 3).unwrap(), vec![1, 2, 3]);
        assert!(per_channel(&c, "c", 2).is_err());
        assert!(per_channel(&c, "missing", 2).is_err());
    }
}
