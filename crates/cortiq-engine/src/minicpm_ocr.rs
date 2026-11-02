//! MiniCPM-OCR-1.0 (`minicpm_ocr`): the SigLIP vision encoder, its 2×2 merger
//! into the dense Qwen3.5 text model, the slicing image processor and the
//! prompt assembly that replaces `<|image_pad|>` with the tower's rows.
//!
//! The text model is the ordinary hybrid Qwen3.5 decoder with plain positions,
//! so the prompt rows go through the embedded-prompt prefill. A file is
//! vision-capable when it carries `vis.*` and the checkpoint and processor
//! configs under `provenance.minicpm_ocr`.
//!
//! ## Preprocessing (`MiniCPMOcrImageProcessor`)
//!
//! An image larger than `scale_resolution²` (448²) becomes an overview plus up
//! to `max_slice_nums` (9) slices on a grid picked by aspect ratio. Everything
//! is resized with torchvision's uint8 antialiased bicubic to multiples of
//! `patch·2` and standardized with mean and std 0.5. A smaller image is
//! upscaled to the same budget (`small_image_upsize = "always"`).
//!
//! ## The tower (`MiniCPMOcrVisionModel`, `downsample_mode = "4x"`)
//!
//! * a conv patch embedding (a linear with bias over the 588-wide patch) plus
//!   a position table read at the nearest bucket of a 70×70 grid;
//! * 27 pre-norm blocks with separate q/k/v projections, no rotary, and the
//!   tanh GELU;
//! * `post_layernorm`, then the merger: four rows of a 2×2 block joined into
//!   one 4608-wide row, LayerNorm, linear, exact GELU, linear to the text
//!   width.
//!
//! The checkpoint's `vit_merger` serves only `downsample_mode = "16x"` and is
//! not converted. Each slice runs through the tower on its own.
//!
//! Parity: `scripts/ocr/ref_minicpm_ocr.py` dumps the processor's pixels, the
//! tower rows and the prompt logits from Transformers 5.21.

use crate::Pipeline;
use crate::media::{self, RgbFrame};
use crate::onevision2::{Mat, gelu_rows, layer_norm_rows};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use serde_json::{Value, json};
use std::sync::Arc;

/// The tower and processor settings.
#[derive(Clone, Debug)]
pub struct OcrConfig {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub patch_size: usize,
    pub side: usize,
    pub eps: f64,
    pub text_hidden: usize,
    pub image_token_id: u32,
    pub scale_resolution: usize,
    pub max_slice_nums: usize,
    pub slice_mode: bool,
    pub use_image_id: bool,
}

/// Whether `model` carries the MiniCPM-OCR tower.
pub fn has_vision(model: &CmfModel) -> bool {
    model.header.arch.arch_name == "minicpm_ocr" && model.tensor("vis.embeddings.patch_embedding.weight").is_some()
}

fn need(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| format!("minicpm_ocr config: missing '{key}'"))
}

impl OcrConfig {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let p = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("minicpm_ocr"))
            .ok_or("minicpm_ocr: the file carries no provenance.minicpm_ocr (reconvert it)")?;
        let cfg = p.get("config").ok_or("minicpm_ocr: provenance has no config")?;
        let processor = p.get("processor").ok_or("minicpm_ocr: provenance has no processor")?;
        Self::from_json(cfg, processor)
    }

    pub fn from_json(cfg: &Value, processor: &Value) -> Result<Self, String> {
        let vc = cfg.get("vision_config").ok_or("minicpm_ocr: config has no vision_config")?;
        let tc = cfg.get("text_config").ok_or("minicpm_ocr: config has no text_config")?;
        let mode = |v: &Value| v.get("downsample_mode").and_then(Value::as_str).map(str::to_owned);
        if mode(cfg).as_deref().unwrap_or("4x") != "4x" || mode(processor).as_deref().unwrap_or("4x") != "4x" {
            return Err("minicpm_ocr: only downsample_mode 4x is supported".into());
        }
        let act = vc.get("hidden_act").and_then(Value::as_str).unwrap_or("");
        if act != "gelu_pytorch_tanh" {
            return Err(format!("minicpm_ocr: unsupported tower activation '{act}'"));
        }
        let hidden = need(vc, "hidden_size")?;
        let heads = need(vc, "num_attention_heads")?;
        if heads == 0 || hidden % heads != 0 {
            return Err("minicpm_ocr: the tower width does not split into heads".into());
        }
        let patch_size = need(vc, "patch_size")?;
        if need(processor, "patch_size")? != patch_size {
            return Err("minicpm_ocr: processor and tower patch sizes disagree".into());
        }
        Ok(Self {
            hidden,
            heads,
            head_dim: hidden / heads,
            layers: need(vc, "num_hidden_layers")?,
            intermediate: need(vc, "intermediate_size")?,
            patch_size,
            side: need(vc, "image_size")? / patch_size,
            eps: vc.get("layer_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            text_hidden: need(tc, "hidden_size")?,
            image_token_id: need(cfg, "image_token_id")? as u32,
            scale_resolution: need(processor, "scale_resolution")?,
            max_slice_nums: need(processor, "max_slice_nums")?,
            slice_mode: processor.get("slice_mode").and_then(Value::as_bool).unwrap_or(true),
            use_image_id: processor.get("use_image_id").and_then(Value::as_bool).unwrap_or(true),
        })
    }
}

/// `ensure_divide`: `max(round(length / divisor) · divisor, divisor)`, with
/// Python's half-to-even `round`.
fn ensure_divide(length: f64, divisor: usize) -> usize {
    let q = length / divisor as f64;
    let r = q.round();
    let r = if (q - q.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - q.signum() } else { r };
    (r as usize * divisor).max(divisor)
}

/// `find_best_resize` for `downsample_mode = "4x"`: `(height, width)`.
fn best_resize(h: f64, w: f64, scale: usize, patch: usize, allow_upscale: bool) -> (usize, usize) {
    let (mut h, mut w) = (h, w);
    if h * w > (scale * scale) as f64 || allow_upscale {
        let aspect = w / h;
        h = (scale as f64 / aspect.sqrt()).trunc();
        w = (h * aspect).trunc();
    }
    (ensure_divide(h, patch * 2), ensure_divide(w, patch * 2))
}

/// `get_sliced_grid`: `(rows, cols)`, or None for an image that needs no slices.
fn sliced_grid(h: usize, w: usize, max_slices: usize, scale: usize) -> Option<(usize, usize)> {
    let log_ratio = (w as f64 / h as f64).ln();
    let ratio = (w * h) as f64 / (scale * scale) as f64;
    let multiple = (ratio.ceil() as usize).min(max_slices);
    if multiple <= 1 {
        return None;
    }
    let (mut best, mut min_error) = ((1usize, 1usize), f64::INFINITY);
    for slices in [multiple - 1, multiple, multiple + 1] {
        if slices == 1 || slices > max_slices {
            continue;
        }
        for rows in 1..=slices {
            if slices % rows == 0 {
                let cols = slices / rows;
                let error = (log_ratio - (cols as f64 / rows as f64).ln()).abs();
                if error < min_error {
                    best = (rows, cols);
                    min_error = error;
                } else if error == min_error && rows > best.0 {
                    best = (rows, cols);
                }
            }
        }
    }
    Some(best)
}

/// One view of an image: raster patch rows `[h·w, 3·p·p]` on an `h`×`w` grid.
pub struct OcrView {
    pub rows: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

/// The views of an image, the overview first, and its slice grid.
pub fn prepare_views(frame: &RgbFrame, c: &OcrConfig) -> Result<(Vec<OcrView>, Option<(usize, usize)>), String> {
    let (h0, w0) = (frame.height, frame.width);
    let (scale, patch) = (c.scale_resolution, c.patch_size);
    let grid = c.slice_mode.then(|| sliced_grid(h0, w0, c.max_slice_nums, scale)).flatten();
    let (sh, sw) = best_resize(h0 as f64, w0 as f64, scale, patch, grid.is_none());
    let resize = |h: usize, w: usize| {
        if (h, w) == (h0, w0) { frame.clone() } else { crate::egemma2_vision::resize_bicubic_aa(frame, h, w) }
    };
    let mut views = vec![patchify(&resize(sh, sw), 0, 0, sh, sw, patch)];
    if let Some((grid_y, grid_x)) = grid {
        let (ry, rx) = (ensure_divide(h0 as f64, grid_y), ensure_divide(w0 as f64, grid_x));
        let (bh, bw) = best_resize(ry as f64 / grid_y as f64, rx as f64 / grid_x as f64, scale, patch, true);
        let refine = resize(bh * grid_y, bw * grid_x);
        for y in 0..grid_y {
            for x in 0..grid_x {
                views.push(patchify(&refine, y * bh, x * bw, bh, bw, patch));
            }
        }
    }
    Ok((views, grid))
}

/// Standardize a crop of `img` and lay it out as raster patches `(c, py, px)`.
fn patchify(img: &RgbFrame, top: usize, left: usize, h: usize, w: usize, p: usize) -> OcrView {
    let (gh, gw) = (h / p, w / p);
    let mut rows = vec![0f32; gh * gw * 3 * p * p];
    let mut k = 0usize;
    for by in 0..gh {
        for bx in 0..gw {
            for c in 0..3 {
                for y in 0..p {
                    let line = ((top + by * p + y) * img.width + left + bx * p) * 3 + c;
                    for x in 0..p {
                        rows[k] = (img.data[line + x * 3] as f32 - 127.5) / 127.5;
                        k += 1;
                    }
                }
            }
        }
    }
    OcrView { rows, h: gh, w: gw }
}

/// `get_vision_nearest_position_ids` for one view: f32 coordinates and f32
/// bucket boundaries, as torch computes them.
fn position_ids(h: usize, w: usize, side: usize) -> Vec<usize> {
    let step = 1.0 / side as f64;
    let boundaries: Vec<f32> = (0..).map(|i| step + i as f64 * step).take_while(|&b| b < 1.0).map(|b| b as f32).collect();
    let bucket = |i: usize, n: usize| {
        let v = i as f32 / n as f32;
        boundaries.partition_point(|&b| b <= v)
    };
    let mut out = Vec::with_capacity(h * w);
    for y in 0..h {
        let by = bucket(y, h);
        for x in 0..w {
            out.push(by * side + bucket(x, w));
        }
    }
    out
}

fn vec_of(model: &Arc<CmfModel>, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let v = crate::dit::cmf_f32(model, name)?;
    if v.len() != len {
        return Err(format!("tensor '{name}' has {} values, expected {len}", v.len()));
    }
    Ok(v)
}

/// The separate q, k and v projections as one `[3d, d]` matrix, the layout the
/// resident ViT kernels take.
fn fused_qkv(model: &Arc<CmfModel>, p: &str, d: usize) -> Result<(Mat, Vec<f32>), String> {
    let mut w = Vec::with_capacity(3 * d * d);
    let mut b = Vec::with_capacity(3 * d);
    for s in ["q_proj", "k_proj", "v_proj"] {
        w.extend(vec_of(model, &format!("{p}.self_attn.{s}.weight"), d * d)?);
        b.extend(vec_of(model, &format!("{p}.self_attn.{s}.bias"), d)?);
    }
    Ok((Mat::from_vec(w, 3 * d, d), b))
}

struct Layer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    qkv: Mat,
    qkv_b: Vec<f32>,
    proj: Mat,
    proj_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1: Mat,
    fc1_b: Vec<f32>,
    fc2: Mat,
    fc2_b: Vec<f32>,
}

/// The vision encoder and its merger.
pub struct OcrTower {
    pub cfg: OcrConfig,
    patch: Mat,
    patch_b: Vec<f32>,
    pos: Vec<f32>,
    layers: Vec<Layer>,
    post_w: Vec<f32>,
    post_b: Vec<f32>,
    pre_w: Vec<f32>,
    pre_b: Vec<f32>,
    fc1: Mat,
    fc1_b: Vec<f32>,
    fc2: Mat,
    fc2_b: Vec<f32>,
    pool: Option<Arc<Pool>>,
}

/// `CMF_OCR_GPU=0` keeps the tower on the host.
fn device_on() -> bool {
    std::env::var("CMF_OCR_GPU").as_deref() != Ok("0") && crate::gpu::enabled_here()
}

impl OcrTower {
    pub fn from_model(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let cfg = OcrConfig::from_model(model)?;
        let (d, inter) = (cfg.hidden, cfg.intermediate);
        let merged = d * 4;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("vis.encoder.layers.{i}");
            let (qkv, qkv_b) = fused_qkv(model, &p, d)?;
            layers.push(Layer {
                ln1_w: vec_of(model, &format!("{p}.layer_norm1.weight"), d)?,
                ln1_b: vec_of(model, &format!("{p}.layer_norm1.bias"), d)?,
                qkv,
                qkv_b,
                proj: Mat::load(model, &format!("{p}.self_attn.out_proj.weight"), d, d)?,
                proj_b: vec_of(model, &format!("{p}.self_attn.out_proj.bias"), d)?,
                ln2_w: vec_of(model, &format!("{p}.layer_norm2.weight"), d)?,
                ln2_b: vec_of(model, &format!("{p}.layer_norm2.bias"), d)?,
                fc1: Mat::load(model, &format!("{p}.mlp.fc1.weight"), inter, d)?,
                fc1_b: vec_of(model, &format!("{p}.mlp.fc1.bias"), inter)?,
                fc2: Mat::load(model, &format!("{p}.mlp.fc2.weight"), d, inter)?,
                fc2_b: vec_of(model, &format!("{p}.mlp.fc2.bias"), d)?,
            });
        }
        let pd = 3 * cfg.patch_size * cfg.patch_size;
        Ok(Self {
            patch: Mat::load(model, "vis.embeddings.patch_embedding.weight", d, pd)?,
            patch_b: vec_of(model, "vis.embeddings.patch_embedding.bias", d)?,
            pos: vec_of(model, "vis.embeddings.position_embedding.weight", cfg.side * cfg.side * d)?,
            layers,
            post_w: vec_of(model, "vis.post_layernorm.weight", d)?,
            post_b: vec_of(model, "vis.post_layernorm.bias", d)?,
            pre_w: vec_of(model, "vis.merger.mlp.0.pre_norm.weight", merged)?,
            pre_b: vec_of(model, "vis.merger.mlp.0.pre_norm.bias", merged)?,
            fc1: Mat::load(model, "vis.merger.mlp.0.linear_1.weight", merged, merged)?,
            fc1_b: vec_of(model, "vis.merger.mlp.0.linear_1.bias", merged)?,
            fc2: Mat::load(model, "vis.merger.mlp.0.linear_2.weight", cfg.text_hidden, merged)?,
            fc2_b: vec_of(model, "vis.merger.mlp.0.linear_2.bias", cfg.text_hidden)?,
            cfg,
            pool,
        })
    }

    /// Raster patch rows of one view → merged rows `[h·w/4, text_hidden]`.
    pub fn forward(&self, v: &OcrView) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (d, inter) = (c.hidden, c.intermediate);
        let n = v.h * v.w;
        if n == 0 || v.h % 2 != 0 || v.w % 2 != 0 {
            return Err("minicpm_ocr: a view's grid is not divisible by the 2×2 merge".into());
        }
        let pool = self.pool.as_deref();
        let prof = std::env::var_os("CMF_OCR_PROF").is_some();
        let t0 = std::time::Instant::now();
        let mut x = vec![0f32; n * d];
        self.patch.matmat(&v.rows, n, Some(&self.patch_b), &mut x, pool);
        for (row, id) in x.chunks_exact_mut(d).zip(position_ids(v.h, v.w, c.side)) {
            for (a, &p) in row.iter_mut().zip(&self.pos[id * d..(id + 1) * d]) {
                *a += p;
            }
        }
        let t_embed = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        // The resident stack: identity rotations, the kernels' only door.
        let resident = device_on() && {
            let (cos, sin) = (vec![1f32; n * c.head_dim], vec![0f32; n * c.head_dim]);
            let layers = self
                .layers
                .iter()
                .map(|l| crate::gpu::VitLayer {
                    ln1_w: &l.ln1_w,
                    ln1_b: &l.ln1_b,
                    qkv: l.qkv.weights(),
                    qkv_b: &l.qkv_b,
                    proj: l.proj.weights(),
                    proj_b: &l.proj_b,
                    ln2_w: &l.ln2_w,
                    ln2_b: &l.ln2_b,
                    fc1: l.fc1.weights(),
                    fc1_b: &l.fc1_b,
                    fc2: l.fc2.weights(),
                    fc2_b: &l.fc2_b,
                })
                .collect();
            let args = crate::gpu::VitArgs {
                key: self as *const Self as u64,
                layers,
                n,
                d,
                heads: c.heads,
                head_dim: c.head_dim,
                inter,
                eps: c.eps,
                gelu_tanh: true,
                quick_gelu: false,
                rope_halves: true,
                cos: &cos,
                sin: &sin,
                prof,
            };
            crate::gpu::vit_encode(&args, &mut x)
        };
        if !resident {
            self.encode_host(&mut x, n, pool);
        }
        let t_enc = t1.elapsed().as_secs_f64();
        let t2 = std::time::Instant::now();
        let mut xn = vec![0f32; n * d];
        layer_norm_rows(&x, &self.post_w, &self.post_b, c.eps, &mut xn, pool);
        // (h/2, 2, w/2, 2) → (h/2, w/2, 2, 2): four raster rows per cell.
        let nm = n / 4;
        let merged = d * 4;
        let mut cells = vec![0f32; nm * merged];
        for (k, cell) in cells.chunks_exact_mut(merged).enumerate() {
            let (by, bx) = (k / (v.w / 2), k % (v.w / 2));
            for iy in 0..2 {
                for ix in 0..2 {
                    let s = ((by * 2 + iy) * v.w + bx * 2 + ix) * d;
                    cell[(iy * 2 + ix) * d..(iy * 2 + ix + 1) * d].copy_from_slice(&xn[s..s + d]);
                }
            }
        }
        let mut norm = vec![0f32; nm * merged];
        layer_norm_rows(&cells, &self.pre_w, &self.pre_b, 1e-6, &mut norm, pool);
        let mut h1 = vec![0f32; nm * merged];
        self.fc1.matmat(&norm, nm, Some(&self.fc1_b), &mut h1, pool);
        gelu_rows(&mut h1, pool);
        let mut out = vec![0f32; nm * c.text_hidden];
        self.fc2.matmat(&h1, nm, Some(&self.fc2_b), &mut out, pool);
        if prof {
            eprintln!(
                "minicpm_ocr: {}x{} {n} patches, embed {t_embed:.3} s, encoder {t_enc:.3} s ({}), merger {:.3} s",
                v.h,
                v.w,
                if resident { "resident" } else { "per-op" },
                t2.elapsed().as_secs_f64()
            );
        }
        Ok(out)
    }

    /// The encoder blocks on the host GEMM, attention on the device when it
    /// takes it.
    fn encode_host(&self, x: &mut [f32], n: usize, pool: Option<&Pool>) {
        let c = &self.cfg;
        let (d, inter, nh, hd) = (c.hidden, c.intermediate, c.heads, c.head_dim);
        let mut xn = vec![0f32; n * d];
        let mut qkv = vec![0f32; n * 3 * d];
        let (mut qh, mut kh, mut vh) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
        let mut att = vec![0f32; n * d];
        let mut tmp = vec![0f32; n * d];
        let mut ff = vec![0f32; n * inter];
        let scale = (hd as f32).powf(-0.5);
        for l in &self.layers {
            layer_norm_rows(x, &l.ln1_w, &l.ln1_b, c.eps, &mut xn, pool);
            l.qkv.matmat(&xn, n, Some(&l.qkv_b), &mut qkv, pool);
            for p in 0..n {
                let row = &qkv[p * 3 * d..(p + 1) * 3 * d];
                for h in 0..nh {
                    let at = (h * n + p) * hd;
                    qh[at..at + hd].copy_from_slice(&row[h * hd..(h + 1) * hd]);
                    kh[at..at + hd].copy_from_slice(&row[d + h * hd..d + (h + 1) * hd]);
                    vh[at..at + hd].copy_from_slice(&row[2 * d + h * hd..2 * d + (h + 1) * hd]);
                }
            }
            let dev = device_on() && n >= 64 && crate::gpu::dit_attention(&qh, &kh, &vh, nh, nh, n, hd, scale, &mut att);
            if !dev {
                crate::onevision2::attend_host(&qh, &kh, &vh, nh, hd, n, scale, &mut att, pool);
            }
            l.proj.matmat(&att, n, Some(&l.proj_b), &mut tmp, pool);
            for (a, &b) in x.iter_mut().zip(&tmp) {
                *a += b;
            }
            layer_norm_rows(x, &l.ln2_w, &l.ln2_b, c.eps, &mut xn, pool);
            l.fc1.matmat(&xn, n, Some(&l.fc1_b), &mut ff, pool);
            for v in ff.iter_mut() {
                const C: f32 = 0.797_884_6; // √(2/π)
                *v = 0.5 * *v * (1.0 + (C * (*v + 0.044715 * *v * *v * *v)).tanh());
            }
            l.fc2.matmat(&ff, n, Some(&l.fc2_b), &mut tmp, pool);
            for (a, &b) in x.iter_mut().zip(&tmp) {
                *a += b;
            }
        }
    }
}

/// The tower of `model`, built once and kept for the next request.
fn tower_for(model: &Arc<CmfModel>) -> Result<Arc<OcrTower>, String> {
    static TOWER: std::sync::Mutex<Option<(usize, Arc<OcrTower>)>> = std::sync::Mutex::new(None);
    let key = Arc::as_ptr(model) as usize;
    let mut g = TOWER.lock().unwrap();
    if let Some((k, t)) = g.as_ref() {
        if *k == key {
            return Ok(t.clone());
        }
    }
    let t = Arc::new(OcrTower::from_model(model, Pool::from_env())?);
    *g = Some((key, t.clone()));
    Ok(t)
}

/// The prepared prompt: token ids and, with media, one row per position.
pub struct PreparedOcrPrompt {
    pub token_ids: Vec<u32>,
    pub rows: Option<Vec<f32>>,
}

/// Replace image parts with `{"type": "image"}` and collect the sources.
fn normalize(messages: &[Value]) -> Result<(Vec<Value>, Vec<Value>), String> {
    let mut messages = messages.to_vec();
    let mut items = Vec::new();
    for message in &mut messages {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            if part.is_string() {
                continue;
            }
            let typ = part.get("type").and_then(Value::as_str).unwrap_or("");
            if typ == "image" || typ == "image_url" || part.get("image").is_some() || part.get("image_url").is_some() {
                let source = part.get("image_url").or_else(|| part.get("image")).unwrap_or(part);
                items.push(match source.as_str() {
                    Some(url) => json!({"url": url}),
                    None => source.clone(),
                });
                *part = json!({"type": "image"});
            } else if typ != "text" && part.get("text").is_none() {
                return Err(format!("MiniCPM-OCR: unsupported content part '{typ}' (text and images only)"));
            }
        }
    }
    Ok((messages, items))
}

/// Whether any message carries an image part.
pub fn has_media(messages: &[Value]) -> Result<bool, String> {
    Ok(!normalize(messages)?.1.is_empty())
}

/// [`prepare`] on the pipeline's own file.
pub fn prepare_for_pipeline(
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
) -> Result<PreparedOcrPrompt, String> {
    let model = pipeline.model.as_ref().ok_or("MiniCPM-OCR images require a file-backed pipeline")?;
    prepare(model, pipeline, messages, tools, thinking)
}

/// Render `messages`, run every view through the tower, expand each
/// `<|image_pad|>` the way `MiniCPMOcrProcessor` does and assemble the rows.
pub fn prepare(
    model: &Arc<CmfModel>,
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
) -> Result<PreparedOcrPrompt, String> {
    let (messages, items) = normalize(messages)?;
    let tok = &pipeline.tokenizer;
    if items.is_empty() {
        let ids = tok.try_apply_chat_template_json(&messages, tools, thinking)?;
        return Ok(PreparedOcrPrompt { token_ids: ids, rows: None });
    }
    let text = tok
        .render_chat_json(&messages, tools, thinking)
        .ok_or("MiniCPM-OCR: the chat template failed to render")?;
    let tower = tower_for(model)?;
    let cfg = &tower.cfg;
    if cfg.text_hidden != pipeline.hidden_size {
        return Err("MiniCPM-OCR: the merger width differs from the text model".into());
    }
    // CMF_OCR_DUMP=DIR: the pixels and rows of every image as raw f32.
    let dump = std::env::var_os("CMF_OCR_DUMP").map(std::path::PathBuf::from);
    const PAD: &str = "<|image_pad|>";
    let mut sources = items.into_iter();
    let mut expanded = String::with_capacity(text.len());
    let mut media_rows: Vec<f32> = Vec::new();
    let mut rest = text.as_str();
    let mut k = 0usize;
    while let Some(at) = rest.find(PAD) {
        expanded.push_str(&rest[..at]);
        rest = &rest[at + PAD.len()..];
        let source = sources.next().ok_or("MiniCPM-OCR: more <|image_pad|> placeholders than images")?;
        let t0 = std::time::Instant::now();
        let frame = media::decode_rgb(&media::load_image_bytes(&source)?)?;
        let (views, grid) = prepare_views(&frame, cfg)?;
        let t_pre = t0.elapsed().as_secs_f64();
        if cfg.use_image_id {
            expanded.push_str(&format!("<image_id>{k}</image_id>"));
        }
        let mut tokens = Vec::with_capacity(views.len());
        for (vi, v) in views.iter().enumerate() {
            let rows = tower.forward(v)?;
            if let Some(dir) = &dump {
                let write = |name: String, data: &[f32]| {
                    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(dir.join(name), bytes).map_err(|e| format!("CMF_OCR_DUMP: {e}"))
                };
                write(format!("pixels_{k}_{vi}.f32"), &v.rows)?;
                write(format!("rows_{k}_{vi}.f32"), &rows)?;
            }
            tokens.push(v.h * v.w / 4);
            media_rows.extend(rows);
        }
        expanded.push_str("<image>");
        expanded.push_str(&PAD.repeat(tokens[0]));
        expanded.push_str("</image>");
        if let Some((grid_y, grid_x)) = grid {
            for y in 0..grid_y {
                if y > 0 {
                    expanded.push('\n');
                }
                for x in 0..grid_x {
                    expanded.push_str("<slice>");
                    expanded.push_str(&PAD.repeat(tokens[1 + y * grid_x + x]));
                    expanded.push_str("</slice>");
                }
            }
        }
        if dump.is_some() || std::env::var_os("CMF_OCR_PROF").is_some() {
            eprintln!(
                "minicpm_ocr: image {k} {}x{}, {} views, grid {grid:?}, {} tokens, preprocess {t_pre:.3} s, total {:.2} s",
                frame.width,
                frame.height,
                views.len(),
                tokens.iter().sum::<usize>(),
                t0.elapsed().as_secs_f64()
            );
        }
        k += 1;
    }
    expanded.push_str(rest);
    if sources.next().is_some() {
        return Err("MiniCPM-OCR: the template rendered fewer placeholders than images".into());
    }
    let ids = tok.with_bos(tok.encode(&expanded));
    let h = pipeline.hidden_size;
    let mut rows = Vec::with_capacity(ids.len() * h);
    let mut at = 0usize;
    for &id in &ids {
        if id == cfg.image_token_id {
            let row = media_rows.get(at..at + h).ok_or("MiniCPM-OCR: placeholder and image row counts differ")?;
            rows.extend_from_slice(row);
            at += h;
        } else {
            rows.extend(pipeline.embed_id(id));
        }
    }
    if at != media_rows.len() {
        return Err("MiniCPM-OCR: unused image rows".into());
    }
    Ok(PreparedOcrPrompt { token_ids: ids, rows: Some(rows) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divide_rounds_half_to_even() {
        assert_eq!(ensure_divide(14.0, 28), 28); // round(0.5) = 0, then the floor
        assert_eq!(ensure_divide(42.0, 28), 56); // round(1.5) = 2
        assert_eq!(ensure_divide(70.0, 28), 56); // round(2.5) = 2
        assert_eq!(ensure_divide(100.0, 28), 112);
    }

    #[test]
    fn a_tall_page_gets_a_three_by_three_grid() {
        assert_eq!(sliced_grid(1584, 1224, 9, 448), Some((3, 3)));
        assert_eq!(sliced_grid(400, 400, 9, 448), None);
        assert_eq!(sliced_grid(448, 1344, 9, 448), Some((1, 3)));
    }

    #[test]
    fn position_buckets_follow_the_grid() {
        let ids = position_ids(2, 2, 70);
        assert_eq!(ids, vec![0, 35, 35 * 70, 35 * 70 + 35]);
    }

    #[test]
    fn patches_are_raster_with_channel_first_lines() {
        let (w, h, p) = (4usize, 2usize, 2usize);
        let data: Vec<u8> = (0..w * h * 3).map(|i| i as u8).collect();
        let img = RgbFrame::new(w, h, data).unwrap();
        let v = patchify(&img, 0, 2, 2, 2, p);
        assert_eq!((v.h, v.w), (1, 1));
        let px = |b: u8| (b as f32 - 127.5) / 127.5;
        // channel 0, y 0, x 0 of the patch sits at pixel (0, 2)
        assert_eq!(v.rows[0], px(6));
        // channel 1, y 1, x 1 sits at pixel (1, 3)
        assert_eq!(v.rows[4 + 3], px(4 * 3 + 3 * 3 + 1));
    }
}
