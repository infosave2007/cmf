//! Native Qwen-Image-Edit-2509 denoising transformer.
//!
//! This module mirrors the pinned Diffusers 0.36
//! `QwenImageTransformer2DModel`: image and text streams are jointly
//! attended in `[text, image]` order, Q/K use per-head RMSNorm plus the
//! scaled three-axis Qwen RoPE, and every stream has its own modulated MLP
//! and gated residual.  The text encoder, VAE, scheduler and CLI are kept in
//! their own components; this module accepts the already-packed image/text
//! activations described by the native runtime spine.

use crate::pool::Pool;
use crate::qwen_image_ops::{Linear, add_bias, forward_qkv};
use cortiq_core::CmfModel;
use std::sync::Arc;
use std::time::Instant;

const TIME_INPUT: usize = 256;
const TIME_HALF: usize = TIME_INPUT / 2;
const MAX_TOTAL_TOKENS: usize = 16_384;
const MAX_TEXT_TOKENS: usize = 4_096;
const ROPE_THETA: f64 = 10_000.0;
const NORM_EPS: f64 = 1e-6;
const GELU_C: f32 = 0.797_884_6;
const GELU_K: f32 = 0.044_715;

const PROFILE_INPUT: usize = 0;
const PROFILE_TIME: usize = 1;
const PROFILE_MODULATION: usize = 2;
const PROFILE_QKV: usize = 3;
const PROFILE_QK_NORM_ROPE: usize = 4;
const PROFILE_ATTENTION: usize = 5;
const PROFILE_ATTN_OUTPUT: usize = 6;
const PROFILE_MLP: usize = 7;
const PROFILE_OUTPUT: usize = 8;
const PROFILE_STAGES: usize = 9;
const PROFILE_NAMES: [&str; PROFILE_STAGES] = [
    "input-proj",
    "time-proj",
    "modulation",
    "qkv-proj",
    "qknorm-rope",
    "attention",
    "attn-output",
    "mlp",
    "output",
];

/// Optional per-forward wall-clock totals.  The default path only checks the
/// once-cached flag and never calls `Instant::now`; each enabled forward emits
/// one aggregate line instead of logging individual projections or kernels.
struct QwenProfile {
    enabled: bool,
    started: Option<Instant>,
    nanos: [u64; PROFILE_STAGES],
    blocks: usize,
}

impl QwenProfile {
    #[inline]
    fn new() -> Self {
        let enabled = qwen_profile_enabled();
        Self {
            enabled,
            started: enabled.then(Instant::now),
            nanos: [0; PROFILE_STAGES],
            blocks: 0,
        }
    }

    #[inline]
    fn begin(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    #[inline]
    fn finish(&mut self, stage: usize, started: Option<Instant>) {
        if let Some(started) = started {
            self.nanos[stage] = self.nanos[stage]
                .saturating_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        }
    }

    #[inline]
    fn emit(self) {
        let Some(started) = self.started else {
            return;
        };
        let mut line = format!(
            "qwen_image profile: forwards=1 blocks={} total_ms={:.3}",
            self.blocks,
            started.elapsed().as_secs_f64() * 1e3,
        );
        for (name, nanos) in PROFILE_NAMES.iter().zip(self.nanos) {
            line.push_str(&format!(" {name}_ms={:.3}", nanos as f64 / 1e6));
        }
        eprintln!("{line}");
    }
}

fn qwen_profile_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CMF_QWEN_IMAGE_PROFILE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("on"))
            .unwrap_or(false)
    })
}

/// The native dense Qwen Image transformer.
pub struct QwenImageTransformer {
    model: Arc<CmfModel>,
    img_in: Linear,
    img_in_bias: Vec<f32>,
    txt_in: Linear,
    txt_in_bias: Vec<f32>,
    txt_norm: Vec<f32>,
    time_linear_1: Linear,
    time_linear_1_bias: Vec<f32>,
    time_linear_2: Linear,
    time_linear_2_bias: Vec<f32>,
    blocks: Vec<Block>,
    norm_out: Linear,
    norm_out_bias: Vec<f32>,
    proj_out: Linear,
    proj_out_bias: Vec<f32>,
    hidden: usize,
    in_channels: usize,
    out_channels: usize,
    patch_size: usize,
    heads: usize,
    head_dim: usize,
    joint_attention_dim: usize,
    axes_dim: [usize; 3],
    pool: Option<Arc<Pool>>,
}

/// Short alias used by integration code that treats image components as a
/// family of transformers.
pub type Transformer = QwenImageTransformer;

struct Block {
    img_mod: Linear,
    img_mod_bias: Vec<f32>,
    txt_mod: Linear,
    txt_mod_bias: Vec<f32>,
    attn_to_q: Linear,
    attn_to_q_bias: Vec<f32>,
    attn_to_k: Linear,
    attn_to_k_bias: Vec<f32>,
    attn_to_v: Linear,
    attn_to_v_bias: Vec<f32>,
    attn_add_q: Linear,
    attn_add_q_bias: Vec<f32>,
    attn_add_k: Linear,
    attn_add_k_bias: Vec<f32>,
    attn_add_v: Linear,
    attn_add_v_bias: Vec<f32>,
    attn_to_out: Linear,
    attn_to_out_bias: Vec<f32>,
    attn_to_add_out: Linear,
    attn_to_add_out_bias: Vec<f32>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    norm_added_q: Vec<f32>,
    norm_added_k: Vec<f32>,
    img_mlp_in: Linear,
    img_mlp_in_bias: Vec<f32>,
    img_mlp_out: Linear,
    img_mlp_out_bias: Vec<f32>,
    txt_mlp_in: Linear,
    txt_mlp_in_bias: Vec<f32>,
    txt_mlp_out: Linear,
    txt_mlp_out_bias: Vec<f32>,
}

impl std::fmt::Debug for QwenImageTransformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenImageTransformer")
            .field("hidden", &self.hidden)
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("patch_size", &self.patch_size)
            .field("heads", &self.heads)
            .field("head_dim", &self.head_dim)
            .field("blocks", &self.blocks.len())
            .finish()
    }
}

impl QwenImageTransformer {
    /// Open a standalone Qwen Image transformer CMF.
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let model = Arc::new(CmfModel::open(path).map_err(|e| e.to_string())?);
        Self::from_cmf(&model)
    }

    /// Load an already-open CMF. Large matrices remain mmap-backed; only
    /// vector controls (biases, norms) are copied into small F32 buffers.
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("image.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("image.config_json: {e}"))?;

        let patch_size = cfg_usize(&cfg, "patch_size")?;
        let in_channels = cfg_usize(&cfg, "in_channels")?;
        let out_channels = cfg_usize(&cfg, "out_channels")?;
        let heads = cfg_usize(&cfg, "num_attention_heads")?;
        let head_dim = cfg_usize(&cfg, "attention_head_dim")?;
        let hidden = heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Qwen Image hidden size overflows".to_string())?;
        let joint_attention_dim = cfg_usize(&cfg, "joint_attention_dim")?;
        let num_layers = cfg_usize(&cfg, "num_layers")?;
        if patch_size == 0
            || in_channels == 0
            || out_channels == 0
            || heads == 0
            || head_dim == 0
            || num_layers == 0
        {
            return Err("Qwen Image config has a zero geometry field".into());
        }
        if cfg["guidance_embeds"].as_bool().unwrap_or(false) {
            return Err("Qwen Image guidance_embeds is unsupported by this API".into());
        }
        if head_dim % 2 != 0 {
            return Err(format!(
                "Qwen Image head_dim {head_dim} must be even for RoPE"
            ));
        }
        let axes = parse_axes(&cfg, head_dim)?;
        let pool = Pool::from_env();

        let img_in = load_linear(model, "img_in.weight", hidden, in_channels)?;
        let img_in_bias = load_vector(model, "img_in.bias", hidden)?;
        let txt_in = load_linear(model, "txt_in.weight", hidden, joint_attention_dim)?;
        let txt_in_bias = load_vector(model, "txt_in.bias", hidden)?;
        let txt_norm = load_vector(model, "txt_norm.weight", joint_attention_dim)?;
        let time_linear_1 = load_linear(
            model,
            "time_text_embed.timestep_embedder.linear_1.weight",
            hidden,
            TIME_INPUT,
        )?;
        let time_linear_1_bias = load_vector(
            model,
            "time_text_embed.timestep_embedder.linear_1.bias",
            hidden,
        )?;
        let time_linear_2 = load_linear(
            model,
            "time_text_embed.timestep_embedder.linear_2.weight",
            hidden,
            hidden,
        )?;
        let time_linear_2_bias = load_vector(
            model,
            "time_text_embed.timestep_embedder.linear_2.bias",
            hidden,
        )?;

        let mut blocks = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            blocks.push(load_block(model, layer, hidden, head_dim)?);
        }

        let norm_out = load_linear(model, "norm_out.linear.weight", 2 * hidden, hidden)?;
        let norm_out_bias = load_vector(model, "norm_out.linear.bias", 2 * hidden)?;
        let output_features = patch_size
            .checked_mul(patch_size)
            .and_then(|x| x.checked_mul(out_channels))
            .ok_or_else(|| "Qwen Image output geometry overflows".to_string())?;
        let proj_out = load_linear(model, "proj_out.weight", output_features, hidden)?;
        let proj_out_bias = load_vector(model, "proj_out.bias", output_features)?;

        Ok(Self {
            model: model.clone(),
            img_in,
            img_in_bias,
            txt_in,
            txt_in_bias,
            txt_norm,
            time_linear_1,
            time_linear_1_bias,
            time_linear_2,
            time_linear_2_bias,
            blocks,
            norm_out,
            norm_out_bias,
            proj_out,
            proj_out_bias,
            hidden,
            in_channels,
            out_channels,
            patch_size,
            heads,
            head_dim,
            joint_attention_dim,
            axes_dim: axes,
            pool,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub fn input_channels(&self) -> usize {
        self.in_channels
    }

    pub fn output_channels(&self) -> usize {
        self.out_channels
    }

    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Borrow the backing CMF mapping retained by the transformer.
    pub fn model(&self) -> &Arc<CmfModel> {
        &self.model
    }

    /// Stable cache key used by the bounded Metal weight-buffer lifecycle.
    pub fn model_uid(&self) -> u64 {
        self.model.uid()
    }

    /// Forward one packed image/text batch through the full double-stream
    /// denoiser.
    ///
    /// `image` is row-major `[image_tokens, 64]` (or the configured
    /// `in_channels` for a focused fixture), `text` is row-major
    /// `[text_len, 3584]`, and `image_shapes` gives the concatenated target
    /// then reference streams as `(frames, token_height, token_width)`.
    /// `normalized_timestep` is the Diffusers timestep divided by 1000, so a
    /// sigma of 1.0 enters the sinusoidal projection as 1000.0.
    pub fn forward(
        &self,
        image: &[f32],
        text: &[f32],
        image_shapes: &[[usize; 3]],
        text_len: usize,
        normalized_timestep: f32,
    ) -> Result<Vec<f32>, String> {
        if !normalized_timestep.is_finite() {
            return Err("Qwen Image timestep must be finite".into());
        }
        if image.is_empty() || image.len() % self.in_channels != 0 {
            return Err(format!(
                "Qwen Image image input has {} values, expected a non-empty multiple of {}",
                image.len(),
                self.in_channels
            ));
        }
        if text.len() != text_len.saturating_mul(self.joint_attention_dim) {
            return Err(format!(
                "Qwen Image text input length {} != text_len {text_len} × width {}",
                text.len(),
                self.joint_attention_dim
            ));
        }
        if text_len > MAX_TEXT_TOKENS {
            return Err(format!(
                "Qwen Image text length {text_len} exceeds {MAX_TEXT_TOKENS}"
            ));
        }
        let image_tokens = image.len() / self.in_channels;
        let shape_tokens = checked_shape_tokens(image_shapes)?;
        if shape_tokens != image_tokens {
            return Err(format!(
                "Qwen Image shape tokens {shape_tokens} != image rows {image_tokens}"
            ));
        }
        let total = text_len
            .checked_add(image_tokens)
            .ok_or_else(|| "Qwen Image sequence length overflows".to_string())?;
        if total > MAX_TOTAL_TOKENS {
            return Err(format!(
                "Qwen Image sequence length {total} exceeds bounded limit {MAX_TOTAL_TOKENS}"
            ));
        }

        let mut profile = QwenProfile::new();
        let pool = self.pool.as_deref();
        let profile_span = profile.begin();
        let mut img = vec![0.0f32; image_tokens * self.hidden];
        self.img_in.forward(image, image_tokens, &mut img, pool)?;
        add_bias(&mut img, image_tokens, &self.img_in_bias)?;

        let mut txt_normed = vec![0.0f32; text.len()];
        for r in 0..text_len {
            rms_norm_into(
                &text[r * self.joint_attention_dim..(r + 1) * self.joint_attention_dim],
                &self.txt_norm,
                NORM_EPS,
                &mut txt_normed[r * self.joint_attention_dim..(r + 1) * self.joint_attention_dim],
            );
        }
        let mut txt = vec![0.0f32; text_len * self.hidden];
        self.txt_in.forward(&txt_normed, text_len, &mut txt, pool)?;
        add_bias(&mut txt, text_len, &self.txt_in_bias)?;
        profile.finish(PROFILE_INPUT, profile_span);

        let profile_span = profile.begin();
        let temb = self.time_embed(normalized_timestep)?;
        profile.finish(PROFILE_TIME, profile_span);

        let profile_span = profile.begin();
        let (img_cos, img_sin, txt_cos, txt_sin) =
            rope_angles(image_shapes, text_len, &self.axes_dim)?;
        profile.finish(PROFILE_QK_NORM_ROPE, profile_span);
        for block in &self.blocks {
            block.forward(
                &mut img,
                &mut txt,
                image_tokens,
                text_len,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                self.hidden,
                self.heads,
                self.head_dim,
                pool,
                &mut profile,
            )?;
        }

        // AdaLayerNormContinuous: SiLU(temb) -> [scale, shift] projection,
        // affine-free LayerNorm over the image stream, then modulation.
        let profile_span = profile.begin();
        let mut cond = temb.clone();
        silu_inplace(&mut cond);
        let mut final_mod = vec![0.0f32; 2 * self.hidden];
        self.norm_out.forward_one(&cond, &mut final_mod, pool)?;
        add_bias(&mut final_mod, 1, &self.norm_out_bias)?;
        let mut final_norm = vec![0.0f32; img.len()];
        for r in 0..image_tokens {
            layer_norm_into(
                &img[r * self.hidden..(r + 1) * self.hidden],
                NORM_EPS,
                &mut final_norm[r * self.hidden..(r + 1) * self.hidden],
            );
        }
        let mut modulated = vec![0.0f32; img.len()];
        for r in 0..image_tokens {
            for d in 0..self.hidden {
                modulated[r * self.hidden + d] = final_norm[r * self.hidden + d]
                    * (1.0 + final_mod[d])
                    + final_mod[self.hidden + d];
            }
        }
        let output_features = self.patch_size * self.patch_size * self.out_channels;
        let mut out = vec![0.0f32; image_tokens * output_features];
        self.proj_out
            .forward(&modulated, image_tokens, &mut out, pool)?;
        add_bias(&mut out, image_tokens, &self.proj_out_bias)?;
        profile.finish(PROFILE_OUTPUT, profile_span);
        profile.emit();
        Ok(out)
    }

    fn time_embed(&self, timestep: f32) -> Result<Vec<f32>, String> {
        let mut sinusoid = vec![0.0f32; TIME_INPUT];
        let scaled = timestep * 1000.0;
        for i in 0..TIME_HALF {
            let exponent = -((ROPE_THETA.ln()) * i as f64 / TIME_HALF as f64);
            let angle = scaled as f64 * exponent.exp();
            // flip_sin_to_cos=True: cosine half first, sine half second.
            sinusoid[i] = angle.cos() as f32;
            sinusoid[TIME_HALF + i] = angle.sin() as f32;
        }
        let pool = self.pool.as_deref();
        let mut h = vec![0.0f32; self.hidden];
        self.time_linear_1.forward_one(&sinusoid, &mut h, pool)?;
        add_bias(&mut h, 1, &self.time_linear_1_bias)?;
        silu_inplace(&mut h);
        let mut out = vec![0.0f32; self.hidden];
        self.time_linear_2.forward_one(&h, &mut out, pool)?;
        add_bias(&mut out, 1, &self.time_linear_2_bias)?;
        Ok(out)
    }
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: &mut [f32],
        txt: &mut [f32],
        image_tokens: usize,
        text_len: usize,
        temb: &[f32],
        img_cos: &[f32],
        img_sin: &[f32],
        txt_cos: &[f32],
        txt_sin: &[f32],
        hidden: usize,
        heads: usize,
        head_dim: usize,
        pool: Option<&Pool>,
        profile: &mut QwenProfile,
    ) -> Result<(), String> {
        profile.blocks += 1;
        let profile_span = profile.begin();
        let mut mod_cond = temb.to_vec();
        silu_inplace(&mut mod_cond);
        let mut img_mod = vec![0.0f32; 6 * hidden];
        let mut txt_mod = vec![0.0f32; 6 * hidden];
        self.img_mod.forward_one(&mod_cond, &mut img_mod, pool)?;
        self.txt_mod.forward_one(&mod_cond, &mut txt_mod, pool)?;
        add_bias(&mut img_mod, 1, &self.img_mod_bias)?;
        add_bias(&mut txt_mod, 1, &self.txt_mod_bias)?;

        // First normalized/modulated stream inputs.
        let mut img_n = vec![0.0f32; img.len()];
        let mut txt_n = vec![0.0f32; txt.len()];
        for r in 0..image_tokens {
            layer_norm_into(
                &img[r * hidden..(r + 1) * hidden],
                NORM_EPS,
                &mut img_n[r * hidden..(r + 1) * hidden],
            );
            modulate_row(
                &mut img_n[r * hidden..(r + 1) * hidden],
                &img_mod[..3 * hidden],
            );
        }
        for r in 0..text_len {
            layer_norm_into(
                &txt[r * hidden..(r + 1) * hidden],
                NORM_EPS,
                &mut txt_n[r * hidden..(r + 1) * hidden],
            );
            modulate_row(
                &mut txt_n[r * hidden..(r + 1) * hidden],
                &txt_mod[..3 * hidden],
            );
        }
        profile.finish(PROFILE_MODULATION, profile_span);

        // Six affine projections feed the joint attention. The two streams
        // have distinct Q/K/V matrices, exactly as the official processor.
        let profile_span = profile.begin();
        let mut img_q = vec![0.0f32; image_tokens * hidden];
        let mut img_k = vec![0.0f32; image_tokens * hidden];
        let mut img_v = vec![0.0f32; image_tokens * hidden];
        let mut txt_q = vec![0.0f32; text_len * hidden];
        let mut txt_k = vec![0.0f32; text_len * hidden];
        let mut txt_v = vec![0.0f32; text_len * hidden];
        forward_qkv(
            &self.attn_to_q,
            &self.attn_to_k,
            &self.attn_to_v,
            &img_n,
            image_tokens,
            &mut img_q,
            &mut img_k,
            &mut img_v,
            pool,
        )?;
        forward_qkv(
            &self.attn_add_q,
            &self.attn_add_k,
            &self.attn_add_v,
            &txt_n,
            text_len,
            &mut txt_q,
            &mut txt_k,
            &mut txt_v,
            pool,
        )?;
        add_bias(&mut img_q, image_tokens, &self.attn_to_q_bias)?;
        add_bias(&mut img_k, image_tokens, &self.attn_to_k_bias)?;
        add_bias(&mut img_v, image_tokens, &self.attn_to_v_bias)?;
        add_bias(&mut txt_q, text_len, &self.attn_add_q_bias)?;
        add_bias(&mut txt_k, text_len, &self.attn_add_k_bias)?;
        add_bias(&mut txt_v, text_len, &self.attn_add_v_bias)?;
        profile.finish(PROFILE_QKV, profile_span);

        let profile_span = profile.begin();
        normalize_rope(&mut img_q, heads, head_dim, &self.norm_q, img_cos, img_sin);
        normalize_rope(&mut img_k, heads, head_dim, &self.norm_k, img_cos, img_sin);
        normalize_rope(
            &mut txt_q,
            heads,
            head_dim,
            &self.norm_added_q,
            txt_cos,
            txt_sin,
        );
        normalize_rope(
            &mut txt_k,
            heads,
            head_dim,
            &self.norm_added_k,
            txt_cos,
            txt_sin,
        );
        profile.finish(PROFILE_QK_NORM_ROPE, profile_span);

        let profile_span = profile.begin();
        let (img_attn, txt_attn) = joint_attention(
            &txt_q,
            &txt_k,
            &txt_v,
            &img_q,
            &img_k,
            &img_v,
            text_len,
            image_tokens,
            hidden,
            heads,
            head_dim,
            pool,
        );
        profile.finish(PROFILE_ATTENTION, profile_span);

        let profile_span = profile.begin();
        let mut img_proj = vec![0.0f32; img.len()];
        let mut txt_proj = vec![0.0f32; txt.len()];
        self.attn_to_out
            .forward(&img_attn, image_tokens, &mut img_proj, pool)?;
        self.attn_to_add_out
            .forward(&txt_attn, text_len, &mut txt_proj, pool)?;
        add_bias(&mut img_proj, image_tokens, &self.attn_to_out_bias)?;
        add_bias(&mut txt_proj, text_len, &self.attn_to_add_out_bias)?;
        gated_residual(img, &img_proj, &img_mod[2 * hidden..3 * hidden]);
        gated_residual(txt, &txt_proj, &txt_mod[2 * hidden..3 * hidden]);
        profile.finish(PROFILE_ATTN_OUTPUT, profile_span);

        // Second normalized/modulated input and independent GELU-tanh MLPs.
        let profile_span = profile.begin();
        for r in 0..image_tokens {
            layer_norm_into(
                &img[r * hidden..(r + 1) * hidden],
                NORM_EPS,
                &mut img_n[r * hidden..(r + 1) * hidden],
            );
            modulate_row(
                &mut img_n[r * hidden..(r + 1) * hidden],
                &img_mod[3 * hidden..6 * hidden],
            );
        }
        for r in 0..text_len {
            layer_norm_into(
                &txt[r * hidden..(r + 1) * hidden],
                NORM_EPS,
                &mut txt_n[r * hidden..(r + 1) * hidden],
            );
            modulate_row(
                &mut txt_n[r * hidden..(r + 1) * hidden],
                &txt_mod[3 * hidden..6 * hidden],
            );
        }
        let inter = self.img_mlp_in.rows();
        if self.img_mlp_out.cols() != inter || self.txt_mlp_in.rows() != inter {
            return Err(format!(
                "Qwen Image block MLP dimensions disagree: image {}/{} text {}/{}",
                self.img_mlp_in.rows(),
                self.img_mlp_out.cols(),
                self.txt_mlp_in.rows(),
                self.txt_mlp_out.cols()
            ));
        }
        let mut img_ff = vec![0.0f32; img.len()];
        let mut txt_ff = vec![0.0f32; txt.len()];
        // On a wide Q4TP batch the WGPU arm keeps the intermediate panel on
        // the device and applies both biases plus exact tanh-GELU there.  A
        // refusal (unsupported codec/limits/backend/probe) falls through to
        // the original bounded projection + pooled pointwise sequence.
        let img_fused = fused_qwen_gelu_ffn(
            &self.img_mlp_in,
            &self.img_mlp_out,
            &img_n,
            image_tokens,
            hidden,
            inter,
            &self.img_mlp_in_bias,
            &self.img_mlp_out_bias,
            &mut img_ff,
        );
        let txt_fused = fused_qwen_gelu_ffn(
            &self.txt_mlp_in,
            &self.txt_mlp_out,
            &txt_n,
            text_len,
            hidden,
            inter,
            &self.txt_mlp_in_bias,
            &self.txt_mlp_out_bias,
            &mut txt_ff,
        );
        if !img_fused {
            let mut img_mlp = vec![0.0f32; image_tokens * inter];
            self.img_mlp_in
                .forward(&img_n, image_tokens, &mut img_mlp, pool)?;
            // Bias and tanh-GELU are one row-parallel pass.  The helper
            // preserves the original add-then-GELU order.
            gelu_tanh_bias_inplace(&mut img_mlp, image_tokens, &self.img_mlp_in_bias, pool)?;
            self.img_mlp_out
                .forward(&img_mlp, image_tokens, &mut img_ff, pool)?;
        }
        if !txt_fused {
            let mut txt_mlp = vec![0.0f32; text_len * inter];
            self.txt_mlp_in
                .forward(&txt_n, text_len, &mut txt_mlp, pool)?;
            gelu_tanh_bias_inplace(&mut txt_mlp, text_len, &self.txt_mlp_in_bias, pool)?;
            self.txt_mlp_out
                .forward(&txt_mlp, text_len, &mut txt_ff, pool)?;
        }
        if img_fused {
            gated_residual(img, &img_ff, &img_mod[5 * hidden..6 * hidden]);
        } else {
            gated_residual_bias(
                img,
                &img_ff,
                &self.img_mlp_out_bias,
                &img_mod[5 * hidden..6 * hidden],
                pool,
            )?;
        }
        if txt_fused {
            gated_residual(txt, &txt_ff, &txt_mod[5 * hidden..6 * hidden]);
        } else {
            gated_residual_bias(
                txt,
                &txt_ff,
                &self.txt_mlp_out_bias,
                &txt_mod[5 * hidden..6 * hidden],
                pool,
            )?;
        }
        profile.finish(PROFILE_MLP, profile_span);
        Ok(())
    }
}

/// Try the device-resident Qwen tanh-GELU FFN for one stream.  All four
/// projection/bias shapes are checked by the backend; this helper only joins
/// the two mapped Q4TP identities so a mixed stream can fall back alone.
#[allow(clippy::too_many_arguments)]
fn fused_qwen_gelu_ffn(
    input: &Linear,
    output: &Linear,
    x: &[f32],
    batch: usize,
    hidden: usize,
    inter: usize,
    bias_in: &[f32],
    bias_out: &[f32],
    out: &mut [f32],
) -> bool {
    if std::env::var("CMF_QWEN_IMAGE_FUSED_MLP").as_deref() == Ok("0") {
        return false;
    }
    let (Some((im, ii)), Some((om, oi))) = (input.mapped_q4tp(), output.mapped_q4tp()) else {
        return false;
    };
    if im.uid() != om.uid() {
        return false;
    }
    crate::gpu::q4tp_gelu_ffn(im, ii, oi, x, batch, hidden, inter, bias_in, bias_out, out)
}

fn load_block(
    model: &Arc<CmfModel>,
    layer: usize,
    hidden: usize,
    head_dim: usize,
) -> Result<Block, String> {
    let p = format!("transformer_blocks.{layer}");
    let l = |name: &str, rows: usize, cols: usize| {
        load_linear(model, &format!("{p}.{name}"), rows, cols)
    };
    let b = |name: &str, len: usize| load_vector(model, &format!("{p}.{name}"), len);
    let img_mlp_in = Linear::load(model, &format!("{p}.img_mlp.net.0.proj.weight"))?;
    let inter = img_mlp_in.rows();
    if inter == 0 || img_mlp_in.cols() != hidden {
        return Err(format!(
            "linear '{}.img_mlp.net.0.proj.weight' has shape [{}, {}], expected [inter, {hidden}]",
            p,
            img_mlp_in.rows(),
            img_mlp_in.cols()
        ));
    }
    Ok(Block {
        img_mod: l("img_mod.1.weight", 6 * hidden, hidden)?,
        img_mod_bias: b("img_mod.1.bias", 6 * hidden)?,
        txt_mod: l("txt_mod.1.weight", 6 * hidden, hidden)?,
        txt_mod_bias: b("txt_mod.1.bias", 6 * hidden)?,
        attn_to_q: l("attn.to_q.weight", hidden, hidden)?,
        attn_to_q_bias: b("attn.to_q.bias", hidden)?,
        attn_to_k: l("attn.to_k.weight", hidden, hidden)?,
        attn_to_k_bias: b("attn.to_k.bias", hidden)?,
        attn_to_v: l("attn.to_v.weight", hidden, hidden)?,
        attn_to_v_bias: b("attn.to_v.bias", hidden)?,
        attn_add_q: l("attn.add_q_proj.weight", hidden, hidden)?,
        attn_add_q_bias: b("attn.add_q_proj.bias", hidden)?,
        attn_add_k: l("attn.add_k_proj.weight", hidden, hidden)?,
        attn_add_k_bias: b("attn.add_k_proj.bias", hidden)?,
        attn_add_v: l("attn.add_v_proj.weight", hidden, hidden)?,
        attn_add_v_bias: b("attn.add_v_proj.bias", hidden)?,
        attn_to_out: l("attn.to_out.0.weight", hidden, hidden)?,
        attn_to_out_bias: b("attn.to_out.0.bias", hidden)?,
        attn_to_add_out: l("attn.to_add_out.weight", hidden, hidden)?,
        attn_to_add_out_bias: b("attn.to_add_out.bias", hidden)?,
        norm_q: b("attn.norm_q.weight", head_dim)?,
        norm_k: b("attn.norm_k.weight", head_dim)?,
        norm_added_q: b("attn.norm_added_q.weight", head_dim)?,
        norm_added_k: b("attn.norm_added_k.weight", head_dim)?,
        img_mlp_in,
        img_mlp_in_bias: b("img_mlp.net.0.proj.bias", inter)?,
        img_mlp_out: l("img_mlp.net.2.weight", hidden, inter)?,
        img_mlp_out_bias: b("img_mlp.net.2.bias", hidden)?,
        txt_mlp_in: l("txt_mlp.net.0.proj.weight", inter, hidden)?,
        txt_mlp_in_bias: b("txt_mlp.net.0.proj.bias", inter)?,
        txt_mlp_out: l("txt_mlp.net.2.weight", hidden, inter)?,
        txt_mlp_out_bias: b("txt_mlp.net.2.bias", hidden)?,
    })
}

fn load_linear(
    model: &Arc<CmfModel>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Linear, String> {
    let linear = Linear::load(model, name)?;
    if linear.rows() != rows || linear.cols() != cols {
        return Err(format!(
            "linear '{name}' has shape [{}, {}], expected [{rows}, {cols}]",
            linear.rows(),
            linear.cols()
        ));
    }
    Ok(linear)
}

fn load_vector(model: &Arc<CmfModel>, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| format!("missing vector tensor '{name}'"))?;
    if entry.shape.len() != 1 || entry.shape[0] != len {
        return Err(format!(
            "vector '{name}' has shape {:?}, expected [{len}]",
            entry.shape
        ));
    }
    crate::dit::cmf_f32(model, name)
}

fn cfg_usize(cfg: &serde_json::Value, key: &str) -> Result<usize, String> {
    cfg[key]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("Qwen Image config missing integer '{key}'"))
}

fn parse_axes(cfg: &serde_json::Value, head_dim: usize) -> Result<[usize; 3], String> {
    let a = cfg["axes_dims_rope"]
        .as_array()
        .ok_or_else(|| "Qwen Image config missing axes_dims_rope".to_string())?;
    if a.len() != 3 {
        return Err(format!("Qwen Image axes_dims_rope has {} entries", a.len()));
    }
    let axes = [
        a[0].as_u64().and_then(|v| usize::try_from(v).ok()),
        a[1].as_u64().and_then(|v| usize::try_from(v).ok()),
        a[2].as_u64().and_then(|v| usize::try_from(v).ok()),
    ];
    let [Some(a0), Some(a1), Some(a2)] = axes else {
        return Err("Qwen Image axes_dims_rope must contain integers".into());
    };
    if a0 % 2 != 0 || a1 % 2 != 0 || a2 % 2 != 0 || a0 + a1 + a2 != head_dim {
        return Err(format!(
            "Qwen Image RoPE axes [{a0}, {a1}, {a2}] do not partition head_dim {head_dim}"
        ));
    }
    Ok([a0, a1, a2])
}

fn checked_shape_tokens(shapes: &[[usize; 3]]) -> Result<usize, String> {
    if shapes.is_empty() {
        return Err("Qwen Image requires at least one image shape".into());
    }
    let mut total = 0usize;
    for (i, &[frames, height, width]) in shapes.iter().enumerate() {
        if frames == 0 || height == 0 || width == 0 {
            return Err(format!(
                "Qwen Image shape {i} contains zero: [{frames}, {height}, {width}]"
            ));
        }
        let n = frames
            .checked_mul(height)
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| format!("Qwen Image shape {i} overflows"))?;
        total = total
            .checked_add(n)
            .ok_or_else(|| "Qwen Image image-shape token count overflows".to_string())?;
    }
    Ok(total)
}

/// Construct Qwen's scaled three-axis RoPE. The image rows are ordered by
/// `(frame,height,width)` and each reference receives the same centered
/// spatial coordinates but its own positive frame index. Text positions
/// begin at the largest half-resolution image side.
fn rope_angles(
    shapes: &[[usize; 3]],
    text_len: usize,
    axes_dim: &[usize; 3],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let image_tokens = checked_shape_tokens(shapes)?;
    let pairs = axes_dim.iter().sum::<usize>() / 2;
    let mut img_cos = Vec::with_capacity(image_tokens * pairs);
    let mut img_sin = Vec::with_capacity(image_tokens * pairs);
    let mut max_vid_index = 0usize;
    for &[_, height, width] in shapes {
        max_vid_index = max_vid_index.max(height / 2).max(width / 2);
    }
    let mut append = |ids: [i64; 3], cos: &mut Vec<f32>, sin: &mut Vec<f32>| {
        for (axis, &dim) in axes_dim.iter().enumerate() {
            for j in 0..dim / 2 {
                let freq = 1.0 / ROPE_THETA.powf(2.0 * j as f64 / dim as f64);
                let angle = ids[axis] as f64 * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    };
    for (shape_idx, &[frames, height, width]) in shapes.iter().enumerate() {
        let hneg = height - height / 2;
        let wneg = width - width / 2;
        for frame in 0..frames {
            let frame_id = (shape_idx + frame) as i64;
            for h in 0..height {
                let h_id = if h < hneg {
                    h as i64 - hneg as i64
                } else {
                    (h - hneg) as i64
                };
                for w in 0..width {
                    let w_id = if w < wneg {
                        w as i64 - wneg as i64
                    } else {
                        (w - wneg) as i64
                    };
                    append([frame_id, h_id, w_id], &mut img_cos, &mut img_sin);
                }
            }
        }
    }
    debug_assert_eq!(img_cos.len(), image_tokens * pairs);
    let mut txt_cos = Vec::with_capacity(text_len * pairs);
    let mut txt_sin = Vec::with_capacity(text_len * pairs);
    for pos in max_vid_index..max_vid_index.saturating_add(text_len) {
        append(
            [pos as i64, pos as i64, pos as i64],
            &mut txt_cos,
            &mut txt_sin,
        );
    }
    Ok((img_cos, img_sin, txt_cos, txt_sin))
}

fn rms_norm_into(x: &[f32], weight: &[f32], eps: f64, dst: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), dst.len());
    let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len().max(1) as f64;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for ((d, &v), &w) in dst.iter_mut().zip(x).zip(weight) {
        *d = (v as f64 * inv) as f32 * w;
    }
}

fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f64) {
    debug_assert_eq!(x.len(), weight.len());
    let mean_sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len().max(1) as f64;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for (v, &w) in x.iter_mut().zip(weight) {
        *v = (*v as f64 * inv) as f32 * w;
    }
}

fn layer_norm_into(x: &[f32], eps: f64, dst: &mut [f32]) {
    debug_assert_eq!(x.len(), dst.len());
    let n = x.len().max(1) as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let inv = 1.0 / (var + eps).sqrt();
    for (d, &v) in dst.iter_mut().zip(x) {
        *d = ((v as f64 - mean) * inv) as f32;
    }
}

fn modulate_row(row: &mut [f32], params: &[f32]) {
    let n = row.len();
    debug_assert_eq!(params.len(), 3 * n);
    for i in 0..n {
        row[i] = row[i] * (1.0 + params[n + i]) + params[i];
    }
}

fn gated_residual(dst: &mut [f32], src: &[f32], gate: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (row, srow) in dst
        .chunks_exact_mut(gate.len())
        .zip(src.chunks_exact(gate.len()))
    {
        for ((d, &s), &g) in row.iter_mut().zip(srow).zip(gate) {
            *d += g * s;
        }
    }
}

/// A raw mutable panel whose disjoint row ranges may be processed by the
/// persistent pool.  The caller joins the pool before returning, so no range
/// outlives the original slice and workers never alias one another.
struct PanelMut(*mut f32);

unsafe impl Send for PanelMut {}
unsafe impl Sync for PanelMut {}

impl PanelMut {
    fn as_ptr(&self) -> *mut f32 {
        self.0
    }
}

/// Apply the Qwen MLP input bias and exact tanh GELU in one row-parallel
/// pass.  The scalar expression intentionally matches the historical helper;
/// the optimization only changes which worker visits each row.
fn gelu_tanh_bias_inplace(
    values: &mut [f32],
    batch: usize,
    bias: &[f32],
    pool: Option<&Pool>,
) -> Result<(), String> {
    let expected = batch
        .checked_mul(bias.len())
        .ok_or_else(|| "Qwen Image GELU panel size overflows".to_string())?;
    if bias.is_empty() || values.len() != expected {
        return Err(format!(
            "Qwen Image GELU panel length {} != batch {batch} × width {}",
            values.len(),
            bias.len()
        ));
    }
    let width = bias.len();
    let panel = PanelMut(values.as_mut_ptr());
    let run = |start: usize, end: usize| {
        // The pool hands out rows, so every worker gets a whole number of
        // bias vectors and the source slice remains immutable to its peers.
        let rows = unsafe {
            std::slice::from_raw_parts_mut(
                panel.as_ptr().add(start * width),
                (end - start) * width,
            )
        };
        for row in rows.chunks_exact_mut(width) {
            for (value, &b) in row.iter_mut().zip(bias) {
                *value += b;
                let x = *value;
                let x3 = x * x * x;
                *value = 0.5 * x * (1.0 + (GELU_C * (x + GELU_K * x3)).tanh());
            }
        }
    };
    match pool {
        Some(pool) if batch >= 256 => pool.run_rows(batch, &run),
        _ => run(0, batch),
    }
    Ok(())
}

/// Consume a projection bias directly in its gated residual.  It preserves
/// the two original f32 operations per element while avoiding a temporary
/// write/read pass over the large output panel.
fn gated_residual_bias(
    dst: &mut [f32],
    src: &[f32],
    bias: &[f32],
    gate: &[f32],
    pool: Option<&Pool>,
) -> Result<(), String> {
    if bias.len() != gate.len() {
        return Err(format!(
            "Qwen Image residual bias width {} != gate width {}",
            bias.len(),
            gate.len()
        ));
    }
    if bias.is_empty() {
        return Err("Qwen Image residual bias is empty".into());
    }
    let expected = dst.len();
    if src.len() != expected || expected % bias.len() != 0 {
        return Err(format!(
            "Qwen Image residual buffers have dst={} src={} width={}",
            dst.len(),
            src.len(),
            bias.len()
        ));
    }
    let width = bias.len();
    let batch = expected / width;
    let dst_panel = PanelMut(dst.as_mut_ptr());
    let run = |start: usize, end: usize| {
        let rows = unsafe {
            std::slice::from_raw_parts_mut(
                dst_panel.as_ptr().add(start * width),
                (end - start) * width,
            )
        };
        let src_rows = &src[start * width..end * width];
        for (dst_row, src_row) in rows
            .chunks_exact_mut(width)
            .zip(src_rows.chunks_exact(width))
        {
            for (((d, &s), &b), &g) in dst_row.iter_mut().zip(src_row).zip(bias).zip(gate) {
                let biased = s + b;
                *d += g * biased;
            }
        }
    };
    match pool {
        Some(pool) if batch >= 256 => pool.run_rows(batch, &run),
        _ => run(0, batch),
    }
    Ok(())
}

fn silu_inplace(v: &mut [f32]) {
    for x in v {
        *x = *x / (1.0 + (-*x).exp());
    }
}

fn normalize_rope(
    data: &mut [f32],
    heads: usize,
    head_dim: usize,
    weight: &[f32],
    cos: &[f32],
    sin: &[f32],
) {
    let tokens = data.len() / (heads * head_dim);
    let pairs = head_dim / 2;
    debug_assert_eq!(weight.len(), head_dim);
    debug_assert_eq!(cos.len(), tokens * pairs);
    for t in 0..tokens {
        for h in 0..heads {
            let row = &mut data[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
            rms_norm_inplace(row, weight, NORM_EPS);
            for j in 0..pairs {
                let c = cos[t * pairs + j];
                let s = sin[t * pairs + j];
                let (a, b) = (row[2 * j], row[2 * j + 1]);
                row[2 * j] = a * c - b * s;
                row[2 * j + 1] = a * s + b * c;
            }
        }
    }
}

/// CPU/GPU joint full attention. The score matrix is allocated once per
/// head and reused, bounding peak memory independently of the number of
/// heads. Inputs and output are token-major; image/text outputs are split at
/// the end to feed their separate output projections.
fn joint_attention(
    txt_q: &[f32],
    txt_k: &[f32],
    txt_v: &[f32],
    img_q: &[f32],
    img_k: &[f32],
    img_v: &[f32],
    text_len: usize,
    image_tokens: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    pool: Option<&Pool>,
) -> (Vec<f32>, Vec<f32>) {
    let n = text_len + image_tokens;
    let mut q = Vec::with_capacity(n * hidden);
    let mut k = Vec::with_capacity(n * hidden);
    let mut v = Vec::with_capacity(n * hidden);
    q.extend_from_slice(txt_q);
    q.extend_from_slice(img_q);
    k.extend_from_slice(txt_k);
    k.extend_from_slice(img_k);
    v.extend_from_slice(txt_v);
    v.extend_from_slice(img_v);

    let mut qh_all = vec![0.0f32; heads * n * head_dim];
    let mut kh_all = vec![0.0f32; heads * n * head_dim];
    let mut vh_all = vec![0.0f32; heads * n * head_dim];
    for t in 0..n {
        for h in 0..heads {
            let src = &q[t * hidden + h * head_dim..(t * hidden + (h + 1) * head_dim)];
            qh_all[h * n * head_dim + t * head_dim..h * n * head_dim + (t + 1) * head_dim]
                .copy_from_slice(src);
            let src = &k[t * hidden + h * head_dim..(t * hidden + (h + 1) * head_dim)];
            kh_all[h * n * head_dim + t * head_dim..h * n * head_dim + (t + 1) * head_dim]
                .copy_from_slice(src);
            let src = &v[t * hidden + h * head_dim..(t * hidden + (h + 1) * head_dim)];
            vh_all[h * n * head_dim + t * head_dim..h * n * head_dim + (t + 1) * head_dim]
                .copy_from_slice(src);
        }
    }

    // The existing GPU attention path consumes exactly this head-major input
    // layout and returns token-major output. It is a capability probe, so a
    // CPU-only build or a refused device naturally falls through below.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut device_out = vec![0.0f32; n * hidden];
    if n >= 128
        && crate::gpu::enabled_here()
        && !crate::gpu::mm_killed()
        && crate::gpu::dit_attention(
            &qh_all,
            &kh_all,
            &vh_all,
            heads,
            heads,
            n,
            head_dim,
            scale,
            &mut device_out,
        )
    {
        return (
            device_out[text_len * hidden..].to_vec(),
            device_out[..text_len * hidden].to_vec(),
        );
    }

    let mut image_out = vec![0.0f32; image_tokens * hidden];
    let mut text_out = vec![0.0f32; text_len * hidden];
    let mut qh = vec![0.0f32; n * head_dim];
    let mut kh = vec![0.0f32; n * head_dim];
    let mut vt = vec![0.0f32; head_dim * n];
    let mut scores = vec![0.0f32; n * n];
    let mut oh = vec![0.0f32; n * head_dim];
    for h in 0..heads {
        let qsrc = &qh_all[h * n * head_dim..(h + 1) * n * head_dim];
        let ksrc = &kh_all[h * n * head_dim..(h + 1) * n * head_dim];
        let vsrc = &vh_all[h * n * head_dim..(h + 1) * n * head_dim];
        for t in 0..n {
            for d in 0..head_dim {
                qh[t * head_dim + d] = qsrc[t * head_dim + d] * scale;
                kh[t * head_dim + d] = ksrc[t * head_dim + d];
                vt[d * n + t] = vsrc[t * head_dim + d];
            }
        }
        crate::fcd_ops::gemm_nt(&qh, &kh, &mut scores, n, head_dim, n, pool);
        for row in scores.chunks_exact_mut(n) {
            softmax_inplace(row);
        }
        crate::fcd_ops::gemm_nt(&scores, &vt, &mut oh, n, n, head_dim, pool);
        for t in 0..text_len {
            text_out[t * hidden + h * head_dim..t * hidden + (h + 1) * head_dim]
                .copy_from_slice(&oh[t * head_dim..(t + 1) * head_dim]);
        }
        for t in 0..image_tokens {
            image_out[t * hidden + h * head_dim..t * hidden + (h + 1) * head_dim]
                .copy_from_slice(&oh[(text_len + t) * head_dim..(text_len + t + 1) * head_dim]);
        }
    }
    (image_out, text_out)
}

fn softmax_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in row {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        checked_shape_tokens, gated_residual_bias, gelu_tanh_bias_inplace, rope_angles, Pool,
        GELU_C, GELU_K,
    };

    #[test]
    fn rope_shape_and_centered_scaled_positions_cover_reference_streams() {
        let shapes = [[1, 2, 3], [1, 1, 2]];
        let (c, s, tc, ts) = rope_angles(&shapes, 4, &[2, 2, 0]).unwrap();
        assert_eq!(checked_shape_tokens(&shapes).unwrap(), 8);
        assert_eq!(c.len(), 8 * 2);
        assert_eq!(s.len(), 8 * 2);
        assert_eq!(tc.len(), 4 * 2);
        assert_eq!(ts.len(), 4 * 2);
        // The first image token is frame 0, row -1, col -2.  The first
        // text token starts at max(height//2,width//2)=1.
        assert!((c[0] - 1.0).abs() < 1e-7);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn shape_validation_rejects_empty_and_zero_dimensions() {
        assert!(checked_shape_tokens(&[]).is_err());
        assert!(checked_shape_tokens(&[[1, 0, 2]]).is_err());
    }

    #[test]
    fn pooled_mlp_pointwise_paths_are_bit_exact() {
        let batch = 320;
        let width = 64;
        let bias: Vec<f32> = (0..width).map(|i| (i as f32 - 31.0) * 0.003).collect();
        let gate: Vec<f32> = (0..width).map(|i| 0.2 + i as f32 * 0.001).collect();
        let source: Vec<f32> = (0..batch * width)
            .map(|i| ((i as f32 * 0.017).sin()) * 0.7)
            .collect();

        let mut want_gelu = source.clone();
        for row in want_gelu.chunks_exact_mut(width) {
            for (value, &b) in row.iter_mut().zip(&bias) {
                *value += b;
                let x = *value;
                let x3 = x * x * x;
                *value = 0.5 * x * (1.0 + (GELU_C * (x + GELU_K * x3)).tanh());
            }
        }
        let mut got_gelu = source.clone();
        let pool = Pool::with_spin(2, 0);
        gelu_tanh_bias_inplace(&mut got_gelu, batch, &bias, Some(&pool)).unwrap();
        assert_eq!(got_gelu, want_gelu);

        let mut want_residual = source.clone();
        for (row, src_row) in want_residual
            .chunks_exact_mut(width)
            .zip(source.chunks_exact(width))
        {
            for (((dst, &src), &b), &g) in row.iter_mut().zip(src_row).zip(&bias).zip(&gate) {
                *dst += g * (src + b);
            }
        }
        let mut got_residual = source.clone();
        gated_residual_bias(&mut got_residual, &source, &bias, &gate, Some(&pool)).unwrap();
        assert_eq!(got_residual, want_residual);
    }
}
