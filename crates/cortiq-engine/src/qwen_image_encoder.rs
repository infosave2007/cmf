//! Native Qwen2.5-VL conditioning for Qwen-Image-Edit-2509.
//!
//! This module deliberately owns the complete prompt path: tokenizer JSON,
//! the official Qwen Image prompt template, Qwen2VL image placeholders and
//! vision tower, followed by the causal Qwen2.5-VL language tower.  The
//! final hidden state is the tensor consumed by the Qwen Image transformer;
//! there is no Qwen3 substitution or text-only approximation here.

use crate::qtensor::QTensor;
use crate::qwen_image_ops::Linear;
use crate::qwen_image_vision::{prepare_image, PreparedImage, ProcessorConfig, VisionTower};
use crate::tokenizer::Tokenizer;
use cortiq_core::{CmfModel, TensorDtype};
use image::RgbImage;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

const PROMPT_TEMPLATE_HEAD: &str =
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
const PROMPT_TEMPLATE_TAIL: &str = "<|im_end|>\n<|im_start|>assistant\n";
const DROP_PREFIX: usize = 64;
const EPS: f64 = 1e-6;

/// Conditioning rows for one prompt.  `hidden` is row-major and contains
/// only valid rows after the official 64-token instruction prefix is dropped.
#[derive(Clone, Debug)]
pub struct Conditioning {
    pub hidden: Vec<f32>,
    pub seq_len: usize,
    pub hidden_size: usize,
}

struct MappedEmbedding {
    model: Arc<CmfModel>,
    idx: usize,
    dtype: TensorDtype,
    rows: usize,
    cols: usize,
}

enum Embedding {
    Mapped(MappedEmbedding),
    Quant(QTensor),
}

impl Embedding {
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        let idx = model
            .tensor_index(name)
            .ok_or_else(|| format!("missing embedding tensor '{name}'"))?;
        let entry = &model.tensors[idx];
        if entry.shape.len() != 2 {
            return Err(format!(
                "embedding tensor '{name}' must be rank 2, got shape {:?}",
                entry.shape
            ));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        match entry.dtype {
            TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16 => {
                Ok(Self::Mapped(MappedEmbedding {
                    model: model.clone(),
                    idx,
                    dtype: entry.dtype,
                    rows,
                    cols,
                }))
            }
            _ => Ok(Self::Quant(QTensor::from_model(model, name)?)),
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::Mapped(e) => e.rows,
            Self::Quant(q) => q.rows(),
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::Mapped(e) => e.cols,
            Self::Quant(q) => q.cols(),
        }
    }

    fn row(&self, id: usize, dst: &mut [f32]) -> Result<(), String> {
        if id >= self.rows() {
            return Err(format!(
                "token id {id} exceeds embedding rows {}",
                self.rows()
            ));
        }
        if dst.len() != self.cols() {
            return Err(format!(
                "embedding destination width {} != {}",
                dst.len(),
                self.cols()
            ));
        }
        match self {
            Self::Quant(q) => {
                q.row_f32(id, dst);
                Ok(())
            }
            Self::Mapped(e) => {
                let entry = &e.model.tensors[e.idx];
                let bytes = e.model.entry_bytes(entry);
                let elem_bytes = match e.dtype {
                    TensorDtype::F32 => 4,
                    TensorDtype::F16 | TensorDtype::Bf16 => 2,
                    _ => unreachable!(),
                };
                let expected = e
                    .rows
                    .checked_mul(e.cols)
                    .and_then(|n| n.checked_mul(elem_bytes))
                    .ok_or_else(|| {
                        format!("embedding tensor '{}' byte size overflow", entry.name)
                    })?;
                if bytes.len() != expected {
                    return Err(format!(
                        "embedding tensor '{}' has {} bytes, expected {}",
                        entry.name,
                        bytes.len(),
                        expected
                    ));
                }
                let start = id * e.cols * elem_bytes;
                let src = &bytes[start..start + e.cols * elem_bytes];
                for i in 0..e.cols {
                    let off = i * elem_bytes;
                    dst[i] = match e.dtype {
                        TensorDtype::F32 => {
                            f32::from_le_bytes(src[off..off + 4].try_into().unwrap())
                        }
                        TensorDtype::F16 => cortiq_core::quant::f16_to_f32(u16::from_le_bytes([
                            src[off],
                            src[off + 1],
                        ])),
                        TensorDtype::Bf16 => cortiq_core::quant::bf16_to_f32(u16::from_le_bytes([
                            src[off],
                            src[off + 1],
                        ])),
                        _ => unreachable!(),
                    };
                }
                Ok(())
            }
        }
    }
}

struct TextLayer {
    input_norm: Vec<f32>,
    q: Linear,
    q_bias: Vec<f32>,
    k: Linear,
    k_bias: Vec<f32>,
    v: Linear,
    v_bias: Vec<f32>,
    o: Linear,
    post_norm: Vec<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
    sliding: bool,
}

/// Qwen2.5-VL prompt encoder.  The component CMF retains the official
/// `model.*` and `visual.*` tensor names, plus U8 `image.*_json` assets.
pub struct QwenImageEncoder {
    model: Arc<CmfModel>,
    tokenizer: Tokenizer,
    processor: ProcessorConfig,
    embed: Embedding,
    layers: Vec<TextLayer>,
    final_norm: Vec<f32>,
    vision: VisionTower,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    eps: f64,
    rope_theta: f64,
    mrope_section: [usize; 3],
    sliding_window: usize,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
}

/// Short contract name retained for callers that use the phase document's
/// `Encoder::open` spelling.
pub type Encoder = QwenImageEncoder;

fn json_blob<'a>(model: &'a CmfModel, name: &str) -> Result<&'a [u8], String> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| format!("missing required U8 asset '{name}'"))?;
    if entry.dtype != TensorDtype::U8 {
        return Err(format!(
            "asset '{name}' must use U8 storage, got {}",
            entry.dtype.name()
        ));
    }
    if entry.shape.len() != 1 || entry.shape[0] != entry.n_elems() {
        return Err(format!("asset '{name}' must be a one-dimensional U8 blob"));
    }
    Ok(model.entry_bytes(entry))
}

fn value_usize(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("Qwen config missing integer '{key}'"))
}

fn value_usize_default(v: &Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .unwrap_or(default)
}

fn value_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn text_config<'a>(root: &'a Value) -> &'a Value {
    root.get("text_config").unwrap_or(root)
}

fn vision_config<'a>(root: &'a Value) -> Result<&'a Value, String> {
    root.get("vision_config")
        .ok_or_else(|| "Qwen2.5-VL config has no vision_config".into())
}

fn vector(model: &CmfModel, name: &str, expected: usize) -> Result<Vec<f32>, String> {
    let v = crate::dit::cmf_f32(model, name)?;
    if v.len() != expected {
        return Err(format!(
            "tensor '{name}' has {} values, expected {expected}",
            v.len()
        ));
    }
    Ok(v)
}

fn linear(model: &Arc<CmfModel>, name: &str, rows: usize, cols: usize) -> Result<Linear, String> {
    let p = Linear::load(model, name)?;
    if p.rows() != rows || p.cols() != cols {
        return Err(format!(
            "tensor '{name}' shape [{},{}] != expected [{rows},{cols}]",
            p.rows(),
            p.cols()
        ));
    }
    Ok(p)
}

fn special_id(
    root: &Value,
    text: &Value,
    tokenizer: &Tokenizer,
    config_key: &str,
    token: &str,
) -> Result<u32, String> {
    root.get(config_key)
        .and_then(Value::as_u64)
        .or_else(|| text.get(config_key).and_then(Value::as_u64))
        .map(|v| v as u32)
        .or_else(|| tokenizer.token_to_id(token))
        .ok_or_else(|| format!("missing Qwen special token '{token}' / config key '{config_key}'"))
}

fn parse_mrope(text: &Value, head_dim: usize) -> Result<[usize; 3], String> {
    let values = text
        .get("rope_scaling")
        .and_then(|v| v.get("mrope_section"))
        .and_then(Value::as_array);
    let mut out = if let Some(values) = values {
        if values.len() != 3 {
            return Err("rope_scaling.mrope_section must contain three integers".into());
        }
        [
            values[0]
                .as_u64()
                .ok_or("mrope_section[0] is not an integer")? as usize,
            values[1]
                .as_u64()
                .ok_or("mrope_section[1] is not an integer")? as usize,
            values[2]
                .as_u64()
                .ok_or("mrope_section[2] is not an integer")? as usize,
        ]
    } else if head_dim == 128 {
        // The canonical Qwen2.5-VL config carries this field.  Retaining the
        // known value also lets a minimal seeded fixture omit only metadata
        // that is fixed by the canonical head width.
        [16, 24, 24]
    } else {
        return Err("Qwen2.5-VL config missing rope_scaling.mrope_section".into());
    };
    if out.iter().any(|&x| x == 0) || out.iter().sum::<usize>() != head_dim / 2 {
        return Err(format!(
            "mrope_section {:?} must be positive and sum to head_dim/2={}",
            out,
            head_dim / 2
        ));
    }
    Ok(out)
}

impl QwenImageEncoder {
    /// Open a standalone Qwen2.5-VL component CMF.  Sharded CMFs are opened
    /// through the core loader so the language and vision mappings remain
    /// lazy and no giant F32 embedding copy is created.
    pub fn open(path: &Path) -> Result<Self, String> {
        let model = Arc::new(CmfModel::open_sharded(path).map_err(|e| e.to_string())?);
        let config_bytes = json_blob(&model, "image.config_json")?;
        let root: Value =
            serde_json::from_slice(config_bytes).map_err(|e| format!("image.config_json: {e}"))?;
        let text = text_config(&root);
        let vision_cfg = vision_config(&root)?;
        let processor =
            ProcessorConfig::from_json(json_blob(&model, "image.processor_config_json")?)?;
        let tokenizer = Tokenizer::from_bytes(json_blob(&model, "image.tokenizer_json")?)
            .map_err(|e| format!("image.tokenizer_json: {e}"))?;
        // An optional tokenizer config is retained as a load-time contract:
        // if present it must be valid JSON, but prompt behavior comes from
        // the pipeline's pinned template rather than a guessed chat template.
        if let Some(entry) = model.tensor("image.tokenizer_config_json") {
            if entry.dtype != TensorDtype::U8 {
                return Err("image.tokenizer_config_json must use U8 storage".into());
            }
            serde_json::from_slice::<Value>(model.entry_bytes(entry))
                .map_err(|e| format!("image.tokenizer_config_json: {e}"))?;
        }

        let hidden = value_usize(text, "hidden_size")?;
        let intermediate = value_usize(text, "intermediate_size")?;
        let n_layers = value_usize(text, "num_hidden_layers")?;
        let heads = value_usize(text, "num_attention_heads")?;
        let kv_heads = value_usize_default(text, "num_key_value_heads", heads);
        let head_dim = hidden
            .checked_div(heads)
            .ok_or_else(|| "Qwen text heads must be positive".to_string())?;
        if hidden == 0
            || heads == 0
            || hidden % heads != 0
            || kv_heads == 0
            || heads % kv_heads != 0
            || head_dim < 2
            || head_dim % 2 != 0
        {
            return Err(format!(
                "invalid Qwen text geometry hidden={hidden}, heads={heads}, kv_heads={kv_heads}"
            ));
        }
        if text
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or("silu")
            != "silu"
        {
            return Err("Qwen2.5-VL text MLP requires hidden_act='silu'".into());
        }
        let eps = text
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(EPS);
        let rope_theta = text
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(1_000_000.0);
        if !eps.is_finite() || eps <= 0.0 || !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err("Qwen text epsilon/rope_theta must be finite and positive".into());
        }
        let mrope_section = parse_mrope(text, head_dim)?;
        let layer_types = if let Some(a) = text.get("layer_types").and_then(Value::as_array) {
            if a.len() != n_layers {
                return Err(format!(
                    "layer_types length {} != num_hidden_layers {n_layers}",
                    a.len()
                ));
            }
            a.iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "layer_types contains a non-string".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let use_sliding = value_bool(text, "use_sliding_window", false);
            let max_window_layers = value_usize_default(text, "max_window_layers", n_layers);
            (0..n_layers)
                .map(|i| {
                    if use_sliding && i >= max_window_layers {
                        "sliding_attention".into()
                    } else {
                        "full_attention".into()
                    }
                })
                .collect()
        };
        if layer_types
            .iter()
            .any(|kind| kind != "full_attention" && kind != "sliding_attention")
        {
            return Err("Qwen layer_types contains an unsupported attention type".into());
        }
        let sliding_window = value_usize_default(text, "sliding_window", 4096);
        if layer_types.iter().any(|x| x == "sliding_attention") && sliding_window == 0 {
            return Err("Qwen sliding_window must be positive".into());
        }

        let image_token_id =
            special_id(&root, text, &tokenizer, "image_token_id", "<|image_pad|>")?;
        let vision_start_token_id = special_id(
            &root,
            text,
            &tokenizer,
            "vision_start_token_id",
            "<|vision_start|>",
        )?;
        let vision_end_token_id = special_id(
            &root,
            text,
            &tokenizer,
            "vision_end_token_id",
            "<|vision_end|>",
        )?;
        for (token, id) in [
            ("<|image_pad|>", image_token_id),
            ("<|vision_start|>", vision_start_token_id),
            ("<|vision_end|>", vision_end_token_id),
        ] {
            if let Some(tok_id) = tokenizer.token_to_id(token) {
                if tok_id != id {
                    return Err(format!(
                        "config token id {id} for '{token}' disagrees with tokenizer id {tok_id}"
                    ));
                }
            }
        }

        let embed = Embedding::load(&model, "model.embed_tokens.weight")?;
        if embed.cols() != hidden {
            return Err(format!(
                "model.embed_tokens width {} != text hidden {hidden}",
                embed.cols()
            ));
        }
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("model.layers.{i}");
            let q_rows = heads * head_dim;
            let kv_rows = kv_heads * head_dim;
            layers.push(TextLayer {
                input_norm: vector(&model, &format!("{p}.input_layernorm.weight"), hidden)?,
                q: linear(
                    &model,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_rows,
                    hidden,
                )?,
                q_bias: vector(&model, &format!("{p}.self_attn.q_proj.bias"), q_rows)?,
                k: linear(
                    &model,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_rows,
                    hidden,
                )?,
                k_bias: vector(&model, &format!("{p}.self_attn.k_proj.bias"), kv_rows)?,
                v: linear(
                    &model,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_rows,
                    hidden,
                )?,
                v_bias: vector(&model, &format!("{p}.self_attn.v_proj.bias"), kv_rows)?,
                o: linear(
                    &model,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                    q_rows,
                )?,
                post_norm: vector(
                    &model,
                    &format!("{p}.post_attention_layernorm.weight"),
                    hidden,
                )?,
                gate: linear(
                    &model,
                    &format!("{p}.mlp.gate_proj.weight"),
                    intermediate,
                    hidden,
                )?,
                up: linear(
                    &model,
                    &format!("{p}.mlp.up_proj.weight"),
                    intermediate,
                    hidden,
                )?,
                down: linear(
                    &model,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                    intermediate,
                )?,
                sliding: layer_types[i] == "sliding_attention",
            });
        }
        let final_norm = vector(&model, "model.norm.weight", hidden)?;
        let vision = VisionTower::from_cmf(&model, vision_cfg)?;
        if vision.out_hidden != hidden {
            return Err(format!(
                "vision merger output {} != text hidden {hidden}; image features cannot be spliced",
                vision.out_hidden
            ));
        }
        Ok(Self {
            model,
            tokenizer,
            processor,
            embed,
            layers,
            final_norm,
            vision,
            hidden,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            eps,
            rope_theta,
            mrope_section,
            sliding_window,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Stable CMF identity for the caller's targeted GPU buffer release
    /// after this encoder's stage scope ends.  The encoder keeps this value
    /// available without exposing its internal model mapping.
    pub fn model_uid(&self) -> u64 {
        self.model.uid()
    }

    fn rms_norm(&self, x: &[f32], w: &[f32], dst: &mut [f32]) {
        let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ss + self.eps).sqrt();
        for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
            *d = (v as f64 * inv) as f32 * g;
        }
    }

    fn build_prompt(
        &self,
        prompt: &str,
        images: &[PreparedImage],
    ) -> Result<(String, Vec<usize>), String> {
        let mut image_text = String::new();
        let mut token_counts = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            let count = image.image_tokens(self.processor.merge_size)?;
            if count == 0 {
                return Err(format!("image {} produced zero image tokens", i + 1));
            }
            token_counts.push(count);
            image_text.push_str(&format!("Picture {}: <|vision_start|>", i + 1));
            for _ in 0..count {
                image_text.push_str("<|image_pad|>");
            }
            image_text.push_str("<|vision_end|>");
        }
        let mut out = String::with_capacity(
            PROMPT_TEMPLATE_HEAD.len()
                + image_text.len()
                + prompt.len()
                + PROMPT_TEMPLATE_TAIL.len(),
        );
        out.push_str(PROMPT_TEMPLATE_HEAD);
        out.push_str(&image_text);
        out.push_str(prompt);
        out.push_str(PROMPT_TEMPLATE_TAIL);
        Ok((out, token_counts))
    }

    fn locate_images(
        &self,
        ids: &[u32],
        token_counts: &[usize],
        features: &[Vec<f32>],
    ) -> Result<Vec<ImagePlacement>, String> {
        if token_counts.len() != features.len() {
            return Err("image token/feature count mismatch".into());
        }
        let mut found = Vec::with_capacity(token_counts.len());
        let mut cursor = 0usize;
        for (image_idx, (&count, feature)) in token_counts.iter().zip(features).enumerate() {
            if cursor >= ids.len() {
                return Err(format!(
                    "image {} is missing vision_start token",
                    image_idx + 1
                ));
            }
            let start_marker = ids[cursor..]
                .iter()
                .position(|&id| id == self.vision_start_token_id)
                .map(|p| cursor + p)
                .ok_or_else(|| format!("image {} is missing vision_start token", image_idx + 1))?;
            let start = start_marker + 1;
            if start + count >= ids.len() {
                return Err(format!(
                    "image {} placeholder exceeds token sequence",
                    image_idx + 1
                ));
            }
            if ids[start..start + count]
                .iter()
                .any(|&id| id != self.image_token_id)
            {
                return Err(format!(
                    "image {} placeholder run has a non-image token",
                    image_idx + 1
                ));
            }
            if ids[start + count] != self.vision_end_token_id {
                return Err(format!(
                    "image {} is missing vision_end after {} image tokens",
                    image_idx + 1,
                    count
                ));
            }
            if feature.len() != count * self.hidden {
                return Err(format!(
                    "image {} feature width {} != {} × {}",
                    image_idx + 1,
                    feature.len(),
                    count,
                    self.hidden
                ));
            }
            found.push(ImagePlacement {
                start,
                count,
                feature: feature.clone(),
            });
            cursor = start + count + 1;
        }
        let extra = ids[cursor..]
            .iter()
            .filter(|&&id| id == self.image_token_id)
            .count();
        if extra != 0 {
            return Err("prompt contains more image placeholders than supplied images".into());
        }
        Ok(found)
    }

    fn mrope_positions(
        &self,
        ids: &[u32],
        grids: &[Grid],
        placements: &[ImagePlacement],
    ) -> Result<Vec<[i64; 3]>, String> {
        if grids.len() != placements.len() {
            return Err("grid/placement count mismatch".into());
        }
        let mut pos = vec![[0i64; 3]; ids.len()];
        let mut cursor = 0usize;
        let mut prior_max = -1i64;
        for (image_idx, (grid, placement)) in grids.iter().zip(placements).enumerate() {
            if placement.start < cursor || placement.start + placement.count > ids.len() {
                return Err(format!("image {image_idx} placement is outside prompt"));
            }
            let text_len = placement.start - cursor;
            let st_idx = prior_max + 1;
            for j in 0..text_len {
                let v = st_idx + j as i64;
                pos[cursor + j] = [v; 3];
            }
            let llm_h = grid.h / self.processor.merge_size;
            let llm_w = grid.w / self.processor.merge_size;
            let expected = grid
                .t
                .checked_mul(llm_h)
                .and_then(|n| n.checked_mul(llm_w))
                .ok_or_else(|| "image MRoPE grid overflow".to_string())?;
            if expected != placement.count {
                return Err(format!(
                    "image {image_idx} grid yields {expected} tokens, placeholder has {}",
                    placement.count
                ));
            }
            for local in 0..placement.count {
                let per_t = llm_h * llm_w;
                let t = local / per_t;
                let rem = local % per_t;
                let h = rem / llm_w;
                let w = rem % llm_w;
                let base = text_len as i64 + st_idx;
                pos[placement.start + local] = [base + t as i64, base + h as i64, base + w as i64];
            }
            let end = placement.start + placement.count;
            prior_max = pos[cursor..end]
                .iter()
                .flat_map(|p| p.iter())
                .copied()
                .max()
                .ok_or_else(|| "empty Qwen prompt segment".to_string())?;
            // Keep the vision_end token in the next text segment, exactly as
            // Qwen2.5-VL's `st = ed + image_count` does.
            cursor = end;
        }
        if cursor < ids.len() {
            let st_idx = prior_max + 1;
            for j in cursor..ids.len() {
                let v = st_idx + (j - cursor) as i64;
                pos[j] = [v; 3];
            }
        }
        Ok(pos)
    }

    fn apply_text_rope(&self, q: &mut [f32], k: &mut [f32], positions: &[[i64; 3]]) {
        let half = self.head_dim / 2;
        for (token, p) in positions.iter().enumerate() {
            let mut cos = vec![0f32; self.head_dim];
            let mut sin = vec![0f32; self.head_dim];
            // Transformers builds `freqs = position @ inv_freq`, duplicates
            // it (`cat(freqs, freqs)`), then replaces each mRoPE section with
            // the corresponding temporal/height/width section.  Preserve
            // that split before rotate-half; a direct `[T,H,W]` assignment
            // to the first/second head halves is a different rotation.
            let mut axis_emb = vec![vec![0f32; self.head_dim]; 3];
            let mut axis_sin = vec![vec![0f32; self.head_dim]; 3];
            for axis in 0..3 {
                for j in 0..half {
                    let freq = 1.0 / self.rope_theta.powf(2.0 * j as f64 / self.head_dim as f64);
                    let (s, c) = (p[axis] as f64 * freq).sin_cos();
                    axis_emb[axis][j] = c as f32;
                    axis_sin[axis][j] = s as f32;
                    axis_emb[axis][j + half] = c as f32;
                    axis_sin[axis][j + half] = s as f32;
                }
            }
            let section_widths = [
                self.mrope_section[0] * 2,
                self.mrope_section[1] * 2,
                self.mrope_section[2] * 2,
            ];
            let mut offset = 0usize;
            for (axis, &width) in section_widths.iter().enumerate() {
                // `cos.split([section*2])` is applied before the axis
                // replacement.  Therefore height starts at source offset
                // 32 and width at 80 in the canonical 128-wide head.
                cos[offset..offset + width]
                    .copy_from_slice(&axis_emb[axis][offset..offset + width]);
                sin[offset..offset + width]
                    .copy_from_slice(&axis_sin[axis][offset..offset + width]);
                offset += width;
            }
            for head in 0..self.heads {
                let qv = &mut q[token * self.hidden + head * self.head_dim
                    ..token * self.hidden + (head + 1) * self.head_dim];
                for d in 0..half {
                    let (a, b) = (qv[d], qv[d + half]);
                    qv[d] = a * cos[d] - b * sin[d];
                    qv[d + half] = a * sin[d + half] + b * cos[d + half];
                }
            }
            for head in 0..self.kv_heads {
                let kv_offset = token * self.kv_heads * self.head_dim + head * self.head_dim;
                let kv = &mut k[kv_offset..kv_offset + self.head_dim];
                for d in 0..half {
                    let (a, b) = (kv[d], kv[d + half]);
                    kv[d] = a * cos[d] - b * sin[d];
                    kv[d + half] = a * sin[d + half] + b * cos[d + half];
                }
            }
        }
    }

    fn text_forward(
        &self,
        ids: &[u32],
        positions: &[[i64; 3]],
        placements: &[ImagePlacement],
    ) -> Result<Vec<f32>, String> {
        if ids.is_empty() || ids.len() != positions.len() {
            return Err("Qwen prompt token/position lengths do not match".into());
        }
        let n = ids.len();
        let mut h = vec![0f32; n * self.hidden];
        for (i, &id) in ids.iter().enumerate() {
            self.embed
                .row(id as usize, &mut h[i * self.hidden..(i + 1) * self.hidden])?;
        }
        for placement in placements {
            h[placement.start * self.hidden..(placement.start + placement.count) * self.hidden]
                .copy_from_slice(&placement.feature);
        }
        let pool = crate::pool::Pool::from_env();
        let pool_ref = pool.as_deref();
        let mut norm = vec![0f32; n * self.hidden];
        let mut q = vec![0f32; n * self.hidden];
        let mut k = vec![0f32; n * self.kv_heads * self.head_dim];
        let mut v = vec![0f32; n * self.kv_heads * self.head_dim];
        let mut attn = vec![0f32; n * self.hidden];
        let mut proj = vec![0f32; n * self.hidden];
        let mut gate = vec![0f32; n * self.intermediate];
        let mut up = vec![0f32; n * self.intermediate];
        let mut down = vec![0f32; n * self.hidden];
        for layer in &self.layers {
            for i in 0..n {
                self.rms_norm(
                    &h[i * self.hidden..(i + 1) * self.hidden],
                    &layer.input_norm,
                    &mut norm[i * self.hidden..(i + 1) * self.hidden],
                );
            }
            layer.q.forward(&norm, n, &mut q, pool_ref)?;
            layer.k.forward(&norm, n, &mut k, pool_ref)?;
            layer.v.forward(&norm, n, &mut v, pool_ref)?;
            crate::qwen_image_ops::add_bias(&mut q, n, &layer.q_bias)?;
            crate::qwen_image_ops::add_bias(&mut k, n, &layer.k_bias)?;
            crate::qwen_image_ops::add_bias(&mut v, n, &layer.v_bias)?;
            self.apply_text_rope(&mut q, &mut k, positions);
            attn.fill(0.0);
            let groups = self.heads / self.kv_heads;
            for head in 0..self.heads {
                let kv_head = head / groups;
                for i in 0..n {
                    let first = if layer.sliding {
                        i.saturating_add(1).saturating_sub(self.sliding_window)
                    } else {
                        0
                    };
                    let qv = &q[i * self.hidden + head * self.head_dim
                        ..i * self.hidden + (head + 1) * self.head_dim];
                    let mut scores = vec![0f32; i - first + 1];
                    let mut mx = f32::NEG_INFINITY;
                    for (slot, j) in (first..=i).enumerate() {
                        let kv = &k[j * self.kv_heads * self.head_dim + kv_head * self.head_dim
                            ..j * self.kv_heads * self.head_dim + (kv_head + 1) * self.head_dim];
                        let score =
                            crate::attention::dot_f32(qv, kv) / (self.head_dim as f32).sqrt();
                        scores[slot] = score;
                        mx = mx.max(score);
                    }
                    let mut den = 0.0f32;
                    for score in &mut scores {
                        *score = (*score - mx).exp();
                        den += *score;
                    }
                    if den == 0.0 {
                        return Err("Qwen text attention softmax underflowed".into());
                    }
                    let inv = 1.0 / den;
                    let out = &mut attn[i * self.hidden + head * self.head_dim
                        ..i * self.hidden + (head + 1) * self.head_dim];
                    for (slot, j) in (first..=i).enumerate() {
                        let vv = &v[j * self.kv_heads * self.head_dim + kv_head * self.head_dim
                            ..j * self.kv_heads * self.head_dim + (kv_head + 1) * self.head_dim];
                        let weight = scores[slot] * inv;
                        for (o, &vv) in out.iter_mut().zip(vv) {
                            *o += weight * vv;
                        }
                    }
                }
            }
            layer.o.forward(&attn, n, &mut proj, pool_ref)?;
            for (a, &b) in h.iter_mut().zip(&proj) {
                *a += b;
            }
            for i in 0..n {
                self.rms_norm(
                    &h[i * self.hidden..(i + 1) * self.hidden],
                    &layer.post_norm,
                    &mut norm[i * self.hidden..(i + 1) * self.hidden],
                );
            }
            layer.gate.forward(&norm, n, &mut gate, pool_ref)?;
            layer.up.forward(&norm, n, &mut up, pool_ref)?;
            for (g, &u) in gate.iter_mut().zip(&up) {
                *g = (*g / (1.0 + (-*g).exp())) * u;
            }
            layer.down.forward(&gate, n, &mut down, pool_ref)?;
            for (a, &b) in h.iter_mut().zip(&down) {
                *a += b;
            }
        }
        for i in 0..n {
            self.rms_norm(
                &h[i * self.hidden..(i + 1) * self.hidden],
                &self.final_norm,
                &mut norm[i * self.hidden..(i + 1) * self.hidden],
            );
        }
        Ok(norm)
    }

    /// Encode a positive or negative prompt with the same reference image
    /// list.  The caller invokes this once for each true-CFG branch; image
    /// preprocessing and vision placeholders therefore remain identical.
    pub fn encode(&self, prompt: &str, images: &[RgbImage]) -> Result<Conditioning, String> {
        let prepared: Vec<PreparedImage> = images
            .iter()
            .map(|image| prepare_image(image, &self.processor))
            .collect::<Result<_, _>>()?;
        let features = self.vision.forward(&prepared)?;
        let (prompt_text, token_counts) = self.build_prompt(prompt, &prepared)?;
        let ids = self.tokenizer.encode(&prompt_text);
        if ids.iter().any(|&id| id as usize >= self.embed.rows()) {
            return Err("Qwen tokenizer emitted an id outside the embedding table".into());
        }
        let mut grids = Vec::with_capacity(prepared.len());
        for image in &prepared {
            grids.push(Grid {
                t: image.grid_t,
                h: image.grid_h,
                w: image.grid_w,
            });
        }
        let placements = self.locate_images(&ids, &token_counts, &features)?;
        let positions = self.mrope_positions(&ids, &grids, &placements)?;
        let all = self.text_forward(&ids, &positions, &placements)?;
        if all.len() != ids.len() * self.hidden || ids.len() <= DROP_PREFIX {
            return Err(format!(
                "Qwen final hidden shape {}×{} cannot drop required {}-token prefix",
                ids.len(),
                self.hidden,
                DROP_PREFIX
            ));
        }
        let hidden = all[DROP_PREFIX * self.hidden..].to_vec();
        Ok(Conditioning {
            seq_len: hidden.len() / self.hidden,
            hidden_size: self.hidden,
            hidden,
        })
    }

    /// Name matching the pipeline's positive/negative branch operation while
    /// keeping one shared image list and one immutable encoder mapping.
    pub fn encode_pair(
        &self,
        prompt: &str,
        negative_prompt: &str,
        images: &[RgbImage],
    ) -> Result<(Conditioning, Conditioning), String> {
        Ok((
            self.encode(prompt, images)?,
            self.encode(negative_prompt, images)?,
        ))
    }
}

struct ImagePlacement {
    start: usize,
    count: usize,
    feature: Vec<f32>,
}

#[derive(Clone, Copy)]
struct Grid {
    t: usize,
    h: usize,
    w: usize,
}

#[cfg(test)]
mod tests {
    use super::{parse_mrope, DROP_PREFIX, PROMPT_TEMPLATE_HEAD, PROMPT_TEMPLATE_TAIL};
    use serde_json::json;

    #[test]
    fn canonical_template_and_prefix_contract_are_fixed() {
        assert!(PROMPT_TEMPLATE_HEAD.starts_with("<|im_start|>system\nDescribe"));
        assert!(PROMPT_TEMPLATE_TAIL.ends_with("<|im_start|>assistant\n"));
        assert_eq!(DROP_PREFIX, 64);
    }

    #[test]
    fn tiny_hidden_width_uses_explicit_mrope_sections() {
        let cfg = json!({"rope_scaling":{"mrope_section":[1,1,2]}});
        assert_eq!(parse_mrope(&cfg, 8).unwrap(), [1, 1, 2]);
        assert!(parse_mrope(&json!({}), 8).is_err());
    }
}
