//! Mage-VL-Cowork (`llava_onevision2`): the OneVision vision encoder, its
//! patch merger into the Qwen3 text model, the Qwen2-VL image preprocessing
//! and the prompt assembly that replaces the `<|image_pad|>` placeholders
//! with the tower's rows.
//!
//! The text model is plain Qwen3 with one-dimensional positions (the
//! checkpoint's `LlavaOnevision2Model.forward` builds `arange` position ids
//! even for image tokens), so the prompt rows go through the ordinary
//! embedded-prompt prefill. A file is vision-capable when it carries
//! `visual.*` and the checkpoint and processor configs under
//! `provenance.onevision2`.
//!
//! ## Preprocessing (`Qwen2VLImageProcessorFast`)
//!
//! 1. **Resize** to `smart_resize(h, w, factor 32, min_pixels, max_pixels)`
//!    with torchvision's uint8 antialiased bicubic
//!    (`egemma2_vision::resize_bicubic_aa`).
//! 2. **Standardize** with the CLIP mean and std (the 1/255 rescale folded
//!    in, as the fast processor does).
//! 3. **Patchify** like Qwen2-VL with a temporal patch of 1: 16×16 patches,
//!    rows in 2×2 merge-block order, each row `(c, py, px)`.
//!
//! ## The tower (`LlavaOnevision2VisionPretrainedModel`)
//!
//! * a bias-free linear over the 768-wide patch row, then `layernorm_pre`;
//! * 24 pre-norm blocks: LayerNorm → fused `qkv` (with bias) → RoPE →
//!   bidirectional attention over the whole image → `proj`; LayerNorm →
//!   `fc1` → exact GELU → `fc2`;
//! * RoPE is three-axis (t, h, w with a 4:6:6 split of the 32 frequencies,
//!   θ = 10000, each axis `θ^(-i/n_axis)`), the 32 angles concatenated with
//!   themselves to 64 and applied with the INTERLEAVED `rotate_half`
//!   (`(x0, x1) → (-x1, x0)` on adjacent lanes) — so the two lanes of a pair
//!   use different angles, exactly as the checkpoint does. Patch positions
//!   are the raster `(0, row, col)` of each patch, reordered into merge
//!   blocks with the rows;
//! * the merger LayerNorms each row, joins the four rows of a merge block
//!   into one 4096-wide row and runs `mlp.0 → exact GELU → mlp.2` to the
//!   text width.
//!
//! The matrices are stored f16 and run in f32: on the device GEMM where
//! one exists (`gpu::vision_gemm_nt`), else the host GEMM; attention on
//! `gpu::dit_attention`, else the host.
//!
//! Parity: `scripts/four/mage_vl_ref.py` dumps the processor's pixels, the
//! tower rows and the image-prompt logits from Transformers 4.57.3.

use crate::Pipeline;
use crate::media::{self, RgbFrame};
use crate::mimo_vision::{MimoProcessorConfig, VisualInput, VisualKind};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use serde_json::{Value, json};
use std::sync::Arc;

/// Tensor whose presence marks a vision-capable Mage-VL file.
pub const PATCH: &str = "visual.embeddings.patch_embedding.weight";

/// Everything the vision path reads from `provenance.onevision2`.
#[derive(Clone, Debug)]
pub struct Ov2Config {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub eps: f64,
    pub rope_theta: f64,
    pub text_hidden: usize,
    pub image_token_id: u32,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale: f32,
}

/// Whether `model` carries the OneVision tower.
pub fn has_vision(model: &CmfModel) -> bool {
    model.header.arch.arch_name == "llava_onevision2" && model.tensor(PATCH).is_some()
}

fn need_usize(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| format!("onevision2 config: missing '{key}'"))
}

fn triple(v: &Value, key: &str) -> Result<[f32; 3], String> {
    let a = v
        .get(key)
        .and_then(Value::as_array)
        .filter(|a| a.len() == 3)
        .ok_or_else(|| format!("onevision2 processor: '{key}' must hold 3 numbers"))?;
    let mut out = [0f32; 3];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x
            .as_f64()
            .ok_or_else(|| format!("onevision2 processor: '{key}' must hold numbers"))? as f32;
    }
    Ok(out)
}

impl Ov2Config {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let p = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("onevision2"))
            .ok_or("onevision2: the file carries no provenance.onevision2 (reconvert it)")?;
        let cfg = p.get("config").ok_or("onevision2: provenance has no config")?;
        let processor = p.get("processor").ok_or("onevision2: provenance has no processor")?;
        Self::from_json(cfg, processor)
    }

    pub fn from_json(cfg: &Value, processor: &Value) -> Result<Self, String> {
        let vc = cfg.get("vision_config").ok_or("onevision2: config has no vision_config")?;
        let tc = cfg.get("text_config").ok_or("onevision2: config has no text_config")?;
        let hidden = need_usize(vc, "hidden_size")?;
        let heads = need_usize(vc, "num_attention_heads")?;
        if heads == 0 || hidden % heads != 0 || (hidden / heads) % 32 != 0 {
            return Err("onevision2: head_dim must split 4:6:6 into whole frequencies".into());
        }
        let norm = vc.get("layer_norm_type").and_then(Value::as_str).unwrap_or("layer_norm");
        if norm != "layer_norm" || vc.get("use_head").and_then(Value::as_bool).unwrap_or(false) {
            return Err("onevision2: only the LayerNorm tower without a pooling head is supported".into());
        }
        let act = vc.get("hidden_act").and_then(Value::as_str).unwrap_or("gelu");
        if act != "gelu" {
            return Err(format!("onevision2: unsupported tower activation '{act}'"));
        }
        let c = Ov2Config {
            hidden,
            heads,
            head_dim: hidden / heads,
            layers: need_usize(vc, "num_hidden_layers")?,
            intermediate: need_usize(vc, "intermediate_size")?,
            patch_size: need_usize(vc, "patch_size")?,
            merge_size: need_usize(vc, "spatial_merge_size")?,
            temporal_patch_size: need_usize(vc, "temporal_patch_size")?,
            eps: vc.get("layer_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6),
            rope_theta: vc.get("rope_theta").and_then(Value::as_f64).unwrap_or(10000.0),
            text_hidden: need_usize(tc, "hidden_size")?,
            image_token_id: need_usize(cfg, "image_token_id")? as u32,
            min_pixels: processor
                .get("min_pixels")
                .and_then(Value::as_u64)
                .or_else(|| processor.pointer("/size/shortest_edge").and_then(Value::as_u64))
                .ok_or("onevision2 processor: missing min_pixels")? as usize,
            max_pixels: processor
                .get("max_pixels")
                .and_then(Value::as_u64)
                .or_else(|| processor.pointer("/size/longest_edge").and_then(Value::as_u64))
                .ok_or("onevision2 processor: missing max_pixels")? as usize,
            mean: triple(processor, "image_mean")?,
            std: triple(processor, "image_std")?,
            rescale: processor
                .get("rescale_factor")
                .and_then(Value::as_f64)
                .unwrap_or(1.0 / 255.0) as f32,
        };
        if c.temporal_patch_size != 1 {
            return Err("onevision2: only temporal_patch_size 1 (still images) is supported".into());
        }
        if need_usize(processor, "patch_size")? != c.patch_size
            || need_usize(processor, "merge_size")? != c.merge_size
        {
            return Err("onevision2: processor and tower patch geometry disagree".into());
        }
        Ok(c)
    }

    fn patch_cfg(&self) -> MimoProcessorConfig {
        MimoProcessorConfig {
            patch_size: self.patch_size,
            merge_size: self.merge_size,
            temporal_patch_size: self.temporal_patch_size,
            ..Default::default()
        }
    }

    fn patch_dim(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

/// Preprocess one image into tower rows. `max_pixels` lowers the
/// processor's ceiling (`--image-max-pixels`).
pub fn prepare_image(
    frame: &RgbFrame,
    cfg: &Ov2Config,
    max_pixels: Option<usize>,
) -> Result<VisualInput, String> {
    let factor = cfg.patch_size * cfg.merge_size;
    let max_px = max_pixels.map_or(cfg.max_pixels, |m| m.min(cfg.max_pixels));
    let (hb, wb) = crate::lfm3_vl::smart_resize(frame.height, frame.width, factor, cfg.min_pixels, max_px)?;
    let resized = if (hb, wb) == (frame.height, frame.width) {
        frame.clone()
    } else {
        crate::egemma2_vision::resize_bicubic_aa(frame, hb, wb)
    };
    let plane = hb * wb;
    let mut chw = vec![0f32; 3 * plane];
    for c in 0..3 {
        let m = cfg.mean[c] / cfg.rescale;
        let s = cfg.std[c] / cfg.rescale;
        for (i, px) in resized.data.chunks_exact(3).enumerate() {
            chw[c * plane + i] = (px[c] as f32 - m) / s;
        }
    }
    let (rows, grid_t, grid_h, grid_w) = crate::mimo_vision::patchify(&[chw.as_slice()], hb, wb, &cfg.patch_cfg())?;
    Ok(VisualInput {
        kind: VisualKind::Image,
        rows,
        grid_t,
        grid_h,
        grid_w,
        patch_dim: cfg.patch_dim(),
        merge_size: cfg.merge_size,
        timestamps: Vec::new(),
        frame_indices: Vec::new(),
        resized: (hb, wb),
    })
}

/// A tower matrix as an f32 operand: the device GEMM when the backend has
/// one and the job is big enough, the host GEMM otherwise.
pub(crate) struct Mat {
    w: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Mat {
    pub(crate) fn load(model: &Arc<CmfModel>, name: &str, rows: usize, cols: usize) -> Result<Self, String> {
        let w = crate::dit::cmf_f32(model, name)?;
        if w.len() != rows * cols {
            return Err(format!("tensor '{name}' has {} values, expected [{rows}, {cols}]", w.len()));
        }
        Ok(Mat { w, rows, cols })
    }

    /// `y[n, rows] = x[n, cols] · Wᵀ (+ bias)`.
    pub(crate) fn matmat(&self, x: &[f32], n: usize, bias: Option<&[f32]>, y: &mut [f32], pool: Option<&Pool>) {
        let (k, m) = (self.cols, self.rows);
        if device_gemm_on() && crate::gpu::vision_gemm_nt(x, &self.w, y, n, k, m) {
            GEMM_DEV.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            GEMM_HOST.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::fcd_ops::gemm_nt_host(x, &self.w, y, n, k, m, pool);
        }
        if let Some(b) = bias {
            add_bias(y, b, pool);
        }
    }
}

/// GEMMs the device took / refused (the `CMF_OV2_PROF` report).
pub(crate) static GEMM_DEV: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub(crate) static GEMM_HOST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `CMF_OV2_GPU=0` keeps the tower's GEMMs and attention on the host.
fn device_gemm_on() -> bool {
    std::env::var("CMF_OV2_GPU").as_deref() != Ok("0") && crate::gpu::enabled_here()
}

fn add_bias(y: &mut [f32], b: &[f32], pool: Option<&Pool>) {
    let d = b.len();
    let rows = y.len() / d;
    let dst = crate::pool::SendMut::new(y.as_mut_ptr());
    let f = |lo: usize, hi: usize| {
        for r in lo..hi {
            // SAFETY: disjoint rows per worker.
            let row = unsafe { std::slice::from_raw_parts_mut(dst.at(r * d), d) };
            for (v, &bb) in row.iter_mut().zip(b) {
                *v += bb;
            }
        }
    };
    match pool {
        Some(p) => p.run_rows(rows, &f),
        None => f(0, rows),
    }
}

/// Row-wise LayerNorm with weight and bias (two-pass variance, f64 sums).
pub(crate) fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], eps: f64, out: &mut [f32], pool: Option<&Pool>) {
    let d = w.len();
    let rows = x.len() / d;
    let dst = crate::pool::SendMut::new(out.as_mut_ptr());
    let f = |lo: usize, hi: usize| {
        for r in lo..hi {
            let src = &x[r * d..(r + 1) * d];
            // SAFETY: disjoint rows per worker.
            let o = unsafe { std::slice::from_raw_parts_mut(dst.at(r * d), d) };
            let mean = src.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = src.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..d {
                o[i] = ((src[i] as f64 - mean) * inv) as f32 * w[i] + b[i];
            }
        }
    };
    match pool {
        Some(p) => p.run_rows(rows, &f),
        None => f(0, rows),
    }
}

fn erf(x: f32) -> f32 {
    crate::lfm3_vl::erf(x as f64) as f32
}

/// Exact (erf) GELU in place.
pub(crate) fn gelu_rows(y: &mut [f32], pool: Option<&Pool>) {
    let n = y.len();
    let chunk = 1 << 14;
    let dst = crate::pool::SendMut::new(y.as_mut_ptr());
    let f = |lo: usize, hi: usize| {
        for c in lo..hi {
            let a = c * chunk;
            let b = (a + chunk).min(n);
            // SAFETY: disjoint chunks per worker.
            let s = unsafe { std::slice::from_raw_parts_mut(dst.at(a), b - a) };
            for v in s {
                *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
            }
        }
    };
    let chunks = n.div_ceil(chunk);
    match pool {
        Some(p) => p.run_rows(chunks, &f),
        None => f(0, chunks),
    }
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

/// The OneVision tower and merger.
pub struct Ov2Tower {
    pub cfg: Ov2Config,
    patch: Mat,
    ln_pre_w: Vec<f32>,
    ln_pre_b: Vec<f32>,
    layers: Vec<Layer>,
    ln_q_w: Vec<f32>,
    ln_q_b: Vec<f32>,
    mlp0: Mat,
    mlp0_b: Vec<f32>,
    mlp2: Mat,
    mlp2_b: Vec<f32>,
    pool: Option<Arc<Pool>>,
}

fn vec_of(model: &Arc<CmfModel>, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let v = crate::dit::cmf_f32(model, name)?;
    if v.len() != len {
        return Err(format!("tensor '{name}' has {} values, expected {len}", v.len()));
    }
    Ok(v)
}

impl Ov2Tower {
    pub fn from_model(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let cfg = Ov2Config::from_model(model)?;
        let (d, inter) = (cfg.hidden, cfg.intermediate);
        let merged = d * cfg.merge_size * cfg.merge_size;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("visual.encoder.layers.{i}");
            layers.push(Layer {
                ln1_w: vec_of(model, &format!("{p}.layer_norm1.weight"), d)?,
                ln1_b: vec_of(model, &format!("{p}.layer_norm1.bias"), d)?,
                qkv: Mat::load(model, &format!("{p}.self_attn.qkv.weight"), 3 * d, d)?,
                qkv_b: vec_of(model, &format!("{p}.self_attn.qkv.bias"), 3 * d)?,
                proj: Mat::load(model, &format!("{p}.self_attn.proj.weight"), d, d)?,
                proj_b: vec_of(model, &format!("{p}.self_attn.proj.bias"), d)?,
                ln2_w: vec_of(model, &format!("{p}.layer_norm2.weight"), d)?,
                ln2_b: vec_of(model, &format!("{p}.layer_norm2.bias"), d)?,
                fc1: Mat::load(model, &format!("{p}.mlp.fc1.weight"), inter, d)?,
                fc1_b: vec_of(model, &format!("{p}.mlp.fc1.bias"), inter)?,
                fc2: Mat::load(model, &format!("{p}.mlp.fc2.weight"), d, inter)?,
                fc2_b: vec_of(model, &format!("{p}.mlp.fc2.bias"), d)?,
            });
        }
        Ok(Ov2Tower {
            patch: Mat::load(model, PATCH, d, cfg.patch_dim())?,
            ln_pre_w: vec_of(model, "visual.layernorm_pre.weight", d)?,
            ln_pre_b: vec_of(model, "visual.layernorm_pre.bias", d)?,
            layers,
            ln_q_w: vec_of(model, "visual.merger.ln_q.weight", d)?,
            ln_q_b: vec_of(model, "visual.merger.ln_q.bias", d)?,
            mlp0: Mat::load(model, "visual.merger.mlp.0.weight", merged, merged)?,
            mlp0_b: vec_of(model, "visual.merger.mlp.0.bias", merged)?,
            mlp2: Mat::load(model, "visual.merger.mlp.2.weight", cfg.text_hidden, merged)?,
            mlp2_b: vec_of(model, "visual.merger.mlp.2.bias", cfg.text_hidden)?,
            cfg,
            pool,
        })
    }

    /// cos / sin of the 64 rotary lanes of every patch, `[n, head_dim]`, in
    /// merge-block row order.
    fn rope_tables(&self, gt: usize, gh: usize, gw: usize) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let hd = c.head_dim;
        let half = hd / 2;
        let unit = half / 16;
        let (nt, nh, nw) = (4 * unit, 6 * unit, 6 * unit);
        let inv = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (1.0 / c.rope_theta.powf(i as f64 / n as f64)) as f32)
                .collect()
        };
        let (it, ih, iw) = (inv(nt), inv(nh), inv(nw));
        let m = c.merge_size;
        let n = gt * gh * gw;
        let (mut cos, mut sin) = (vec![0f32; n * hd], vec![0f32; n * hd]);
        let mut k = 0usize;
        let mut ang = vec![0f32; half];
        for t in 0..gt {
            for a in 0..gh / m {
                for b in 0..gw / m {
                    for mh in 0..m {
                        for mw in 0..m {
                            let (r, col) = ((a * m + mh) as f32, (b * m + mw) as f32);
                            // f32 outer products, as torch builds them.
                            for (j, &f) in it.iter().enumerate() {
                                ang[j] = t as f32 * f;
                            }
                            for (j, &f) in ih.iter().enumerate() {
                                ang[nt + j] = r * f;
                            }
                            for (j, &f) in iw.iter().enumerate() {
                                ang[nt + nh + j] = col * f;
                            }
                            for j in 0..hd {
                                let (s, co) = ang[j % half].sin_cos();
                                cos[k * hd + j] = co;
                                sin[k * hd + j] = s;
                            }
                            k += 1;
                        }
                    }
                }
            }
        }
        (cos, sin)
    }

    /// Split the fused qkv rows into head-major planes, rotating q and k.
    fn split_rope(&self, qkv: &[f32], n: usize, cos: &[f32], sin: &[f32], qh: &mut [f32], kh: &mut [f32], vh: &mut [f32]) {
        let (nh, hd) = (self.cfg.heads, self.cfg.head_dim);
        let d = nh * hd;
        let (qp, kp, vp) = (
            crate::pool::SendMut::new(qh.as_mut_ptr()),
            crate::pool::SendMut::new(kh.as_mut_ptr()),
            crate::pool::SendMut::new(vh.as_mut_ptr()),
        );
        let f = |lo: usize, hi: usize| {
            for p in lo..hi {
                let row = &qkv[p * 3 * d..(p + 1) * 3 * d];
                let (cs, sn) = (&cos[p * hd..(p + 1) * hd], &sin[p * hd..(p + 1) * hd]);
                for h in 0..nh {
                    let dst = (h * n + p) * hd;
                    // SAFETY: (head, patch) rows are disjoint per worker.
                    let (q, k, v) = unsafe {
                        (
                            std::slice::from_raw_parts_mut(qp.at(dst), hd),
                            std::slice::from_raw_parts_mut(kp.at(dst), hd),
                            std::slice::from_raw_parts_mut(vp.at(dst), hd),
                        )
                    };
                    let (qs, ks, vs) = (
                        &row[h * hd..(h + 1) * hd],
                        &row[d + h * hd..d + (h + 1) * hd],
                        &row[2 * d + h * hd..2 * d + (h + 1) * hd],
                    );
                    for i in (0..hd).step_by(2) {
                        q[i] = qs[i] * cs[i] - qs[i + 1] * sn[i];
                        q[i + 1] = qs[i + 1] * cs[i + 1] + qs[i] * sn[i + 1];
                        k[i] = ks[i] * cs[i] - ks[i + 1] * sn[i];
                        k[i + 1] = ks[i + 1] * cs[i + 1] + ks[i] * sn[i + 1];
                    }
                    v.copy_from_slice(vs);
                }
            }
        };
        match self.pool.as_deref() {
            Some(pool) => pool.run_rows(n, &f),
            None => f(0, n),
        }
    }

    /// Bidirectional attention over head-major planes → `[n, heads·hd]`.
    fn attend(&self, qh: &[f32], kh: &[f32], vh: &[f32], n: usize, out: &mut [f32]) {
        let (nh, hd) = (self.cfg.heads, self.cfg.head_dim);
        let scale = (hd as f32).powf(-0.5);
        if device_gemm_on() && n >= 64 && crate::gpu::dit_attention(qh, kh, vh, nh, nh, n, hd, scale, out) {
            return;
        }
        let pool = self.pool.as_deref();
        let tile = n.min(1024);
        let mut vt = vec![0f32; hd * n];
        let mut qs = vec![0f32; tile * hd];
        let mut sc = vec![0f32; tile * n];
        let mut oh = vec![0f32; tile * hd];
        for h in 0..nh {
            let kp = &kh[h * n * hd..(h + 1) * n * hd];
            for j in 0..n {
                for e in 0..hd {
                    vt[e * n + j] = vh[(h * n + j) * hd + e];
                }
            }
            for t0 in (0..n).step_by(tile) {
                let tq = tile.min(n - t0);
                for i in 0..tq {
                    for e in 0..hd {
                        qs[i * hd + e] = qh[(h * n + t0 + i) * hd + e] * scale;
                    }
                }
                crate::fcd_ops::gemm_nt_host(&qs[..tq * hd], kp, &mut sc[..tq * n], tq, hd, n, pool);
                let sp = crate::pool::SendMut::new(sc.as_mut_ptr());
                let f = |lo: usize, hi: usize| {
                    for i in lo..hi {
                        // SAFETY: disjoint score rows per worker.
                        let row = unsafe { std::slice::from_raw_parts_mut(sp.at(i * n), n) };
                        let mx = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                        let mut sum = 0f32;
                        for x in row.iter_mut() {
                            *x = (*x - mx).exp();
                            sum += *x;
                        }
                        let inv = 1.0 / sum;
                        for x in row.iter_mut() {
                            *x *= inv;
                        }
                    }
                };
                match pool {
                    Some(p) => p.run_rows(tq, &f),
                    None => f(0, tq),
                }
                crate::fcd_ops::gemm_nt_host(&sc[..tq * n], &vt, &mut oh[..tq * hd], tq, n, hd, pool);
                for i in 0..tq {
                    let r = t0 + i;
                    out[r * nh * hd + h * hd..r * nh * hd + (h + 1) * hd].copy_from_slice(&oh[i * hd..(i + 1) * hd]);
                }
            }
        }
    }

    /// Patch rows `[n, 768]` in merge-block order → merged rows
    /// `[n/4, text_hidden]`.
    pub fn forward(&self, input: &VisualInput) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (gt, gh, gw) = (input.grid_t, input.grid_h, input.grid_w);
        let n = gt * gh * gw;
        let m2 = c.merge_size * c.merge_size;
        if input.rows.len() != n * c.patch_dim() || n % m2 != 0 {
            return Err("onevision2: patch rows do not match the grid".into());
        }
        let pool = self.pool.as_deref();
        let (d, inter) = (c.hidden, c.intermediate);
        let prof = std::env::var("CMF_OV2_PROF").is_ok();
        let t0 = std::time::Instant::now();
        let mut stamps: Vec<(&str, f64)> = Vec::new();
        let mut stamp = |what: &'static str, t: &mut std::time::Instant| {
            stamps.push((what, t.elapsed().as_secs_f64()));
            *t = std::time::Instant::now();
        };
        let mut ts = std::time::Instant::now();
        let mut x = vec![0f32; n * d];
        self.patch.matmat(&input.rows, n, None, &mut x, pool);
        stamp("patch", &mut ts);
        let mut xn = vec![0f32; n * d];
        layer_norm_rows(&x, &self.ln_pre_w, &self.ln_pre_b, c.eps, &mut xn, pool);
        std::mem::swap(&mut x, &mut xn);
        let (cos, sin) = self.rope_tables(gt, gh, gw);
        stamp("ln_pre+rope", &mut ts);
        let mut qkv = vec![0f32; n * 3 * d];
        let (mut qh, mut kh, mut vh) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
        let mut att = vec![0f32; n * d];
        let mut tmp = vec![0f32; n * d];
        let mut ff = vec![0f32; n * inter];
        let (mut t_mm, mut t_att) = (0f64, 0f64);
        // The whole encoder stack resident on the device when it takes it.
        let resident = device_gemm_on() && {
            let layers = self
                .layers
                .iter()
                .map(|l| crate::gpu::VitLayer {
                    ln1_w: &l.ln1_w,
                    ln1_b: &l.ln1_b,
                    qkv: &l.qkv.w,
                    qkv_b: &l.qkv_b,
                    proj: &l.proj.w,
                    proj_b: &l.proj_b,
                    ln2_w: &l.ln2_w,
                    ln2_b: &l.ln2_b,
                    fc1: &l.fc1.w,
                    fc1_b: &l.fc1_b,
                    fc2: &l.fc2.w,
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
                gelu_tanh: false,
                rope_halves: false,
                cos: &cos,
                sin: &sin,
                prof,
            };
            crate::gpu::vit_encode(&args, &mut x)
        };
        stamp("encoder", &mut ts);
        for l in self.layers.iter().filter(|_| !resident) {
            layer_norm_rows(&x, &l.ln1_w, &l.ln1_b, c.eps, &mut xn, pool);
            let tm = std::time::Instant::now();
            l.qkv.matmat(&xn, n, Some(&l.qkv_b), &mut qkv, pool);
            t_mm += tm.elapsed().as_secs_f64();
            self.split_rope(&qkv, n, &cos, &sin, &mut qh, &mut kh, &mut vh);
            let ta = std::time::Instant::now();
            self.attend(&qh, &kh, &vh, n, &mut att);
            t_att += ta.elapsed().as_secs_f64();
            let tm = std::time::Instant::now();
            l.proj.matmat(&att, n, Some(&l.proj_b), &mut tmp, pool);
            for (a, &b) in x.iter_mut().zip(&tmp) {
                *a += b;
            }
            layer_norm_rows(&x, &l.ln2_w, &l.ln2_b, c.eps, &mut xn, pool);
            l.fc1.matmat(&xn, n, Some(&l.fc1_b), &mut ff, pool);
            gelu_rows(&mut ff, pool);
            l.fc2.matmat(&ff, n, Some(&l.fc2_b), &mut tmp, pool);
            t_mm += tm.elapsed().as_secs_f64();
            for (a, &b) in x.iter_mut().zip(&tmp) {
                *a += b;
            }
        }
        layer_norm_rows(&x, &self.ln_q_w, &self.ln_q_b, c.eps, &mut xn, pool);
        let nm = n / m2;
        let mut h1 = vec![0f32; nm * d * m2];
        self.mlp0.matmat(&xn, nm, Some(&self.mlp0_b), &mut h1, pool);
        gelu_rows(&mut h1, pool);
        let mut out = vec![0f32; nm * c.text_hidden];
        self.mlp2.matmat(&h1, nm, Some(&self.mlp2_b), &mut out, pool);
        stamp("merger", &mut ts);
        if prof {
            eprintln!("onevision2 stages: {}", stamps.iter().map(|(w, s)| format!("{w} {s:.3} s")).collect::<Vec<_>>().join(", "));
            eprintln!(
                "onevision2: {n} patches, tower {:.3} s (gemm {:.3} s: {} on the device, {} on the host; attention {:.3} s)",
                t0.elapsed().as_secs_f64(),
                t_mm,
                GEMM_DEV.load(std::sync::atomic::Ordering::Relaxed),
                GEMM_HOST.load(std::sync::atomic::Ordering::Relaxed),
                t_att
            );
        }
        Ok(out)
    }
}

/// The tower of `model`, built once and kept for the next request (a server
/// answers many images with one file; the f32 matrices and their device
/// copies are the expensive part).
fn tower_for(model: &Arc<CmfModel>) -> Result<Arc<Ov2Tower>, String> {
    static TOWER: std::sync::Mutex<Option<(usize, Arc<Ov2Tower>)>> = std::sync::Mutex::new(None);
    let key = Arc::as_ptr(model) as usize;
    let mut g = TOWER.lock().unwrap();
    if let Some((k, t)) = g.as_ref() {
        if *k == key {
            return Ok(t.clone());
        }
    }
    let t = Arc::new(Ov2Tower::from_model(model, crate::pool::Pool::from_env())?);
    *g = Some((key, t.clone()));
    Ok(t)
}

/// The prepared prompt: token ids and, with media, one row per position.
pub struct PreparedOv2Prompt {
    pub token_ids: Vec<u32>,
    pub rows: Option<Vec<f32>>,
}

/// Replace every image part of `messages` with `{"type": "image"}` (what
/// the template renders as `<|vision_start|><|image_pad|><|vision_end|>`)
/// and collect the sources in order. Video and audio are refused.
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
                return Err(format!("Mage-VL: unsupported content part '{typ}' (text and images only)"));
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
    image_max_pixels: Option<usize>,
) -> Result<PreparedOv2Prompt, String> {
    let model = pipeline.model.as_ref().ok_or("Mage-VL images require a file-backed pipeline")?;
    prepare(model, pipeline, messages, tools, thinking, image_max_pixels)
}

/// Render `messages` with the file's template, run every image through the
/// tower, expand each `<|image_pad|>` to the image's merged-token count the
/// way `Qwen2_5_VLProcessor` does, and assemble the prompt rows.
pub fn prepare(
    model: &Arc<CmfModel>,
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
    image_max_pixels: Option<usize>,
) -> Result<PreparedOv2Prompt, String> {
    let (messages, items) = normalize(messages)?;
    let tok = &pipeline.tokenizer;
    if items.is_empty() {
        let ids = tok.try_apply_chat_template_json(&messages, tools, thinking)?;
        return Ok(PreparedOv2Prompt { token_ids: ids, rows: None });
    }
    let text = tok
        .render_chat_json(&messages, tools, thinking)
        .ok_or("Mage-VL: the chat template failed to render")?;
    let tower = tower_for(model)?;
    let cfg = &tower.cfg;
    if cfg.text_hidden != pipeline.hidden_size {
        return Err("Mage-VL: merger width differs from the text model".into());
    }
    // CMF_OV2_DUMP=DIR: each image's patch rows and tower rows as raw
    // little-endian f32 (`pixels_I.f32`, `rows_I.f32`) for parity checks.
    let dump = std::env::var_os("CMF_OV2_DUMP").map(std::path::PathBuf::from);
    const PAD: &str = "<|image_pad|>";
    let mut sources = items.into_iter();
    let mut expanded = String::with_capacity(text.len());
    let mut media_rows: Vec<f32> = Vec::new();
    let mut rest = text.as_str();
    let mut k = 0usize;
    while let Some(at) = rest.find(PAD) {
        expanded.push_str(&rest[..at]);
        rest = &rest[at + PAD.len()..];
        let source = sources.next().ok_or("Mage-VL: more <|image_pad|> placeholders than images")?;
        let t0 = std::time::Instant::now();
        let frame = media::decode_rgb(&media::load_image_bytes(&source)?)?;
        let input = prepare_image(&frame, cfg, image_max_pixels)?;
        let rows = tower.forward(&input)?;
        let tokens = input.patches() / (cfg.merge_size * cfg.merge_size);
        if let Some(dir) = &dump {
            let write = |name: String, v: &[f32]| {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(name), bytes).map_err(|e| format!("CMF_OV2_DUMP: {e}"))
            };
            write(format!("pixels_{k}.f32"), &input.rows)?;
            write(format!("rows_{k}.f32"), &rows)?;
        }
        if dump.is_some() || std::env::var_os("CMF_OV2_PROF").is_some() {
            eprintln!(
                "mage-vl: image {k} {}x{} resized to {}x{}, grid {}x{}, {tokens} tokens, tower {:.2} s",
                frame.width,
                frame.height,
                input.resized.1,
                input.resized.0,
                input.grid_h,
                input.grid_w,
                t0.elapsed().as_secs_f64()
            );
        }
        expanded.push_str(&PAD.repeat(tokens));
        media_rows.extend(rows);
        k += 1;
    }
    expanded.push_str(rest);
    if sources.next().is_some() {
        return Err("Mage-VL: the template rendered fewer placeholders than images".into());
    }
    let ids = tok.with_bos(tok.encode(&expanded));
    let h = pipeline.hidden_size;
    let mut rows = Vec::with_capacity(ids.len() * h);
    let mut at = 0usize;
    for &id in &ids {
        if id == cfg.image_token_id {
            let row = media_rows.get(at..at + h).ok_or("Mage-VL: placeholder and image row counts differ")?;
            rows.extend_from_slice(row);
            at += h;
        } else {
            rows.extend(pipeline.embed_id(id));
        }
    }
    if at != media_rows.len() {
        return Err("Mage-VL: unused image rows".into());
    }
    Ok(PreparedOv2Prompt { token_ids: ids, rows: Some(rows) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Ov2Config {
        let config = json!({
            "image_token_id": 151655,
            "text_config": {"hidden_size": 2560},
            "vision_config": {
                "hidden_size": 1024, "num_attention_heads": 16, "num_hidden_layers": 24,
                "intermediate_size": 4096, "patch_size": 16, "spatial_merge_size": 2,
                "temporal_patch_size": 1, "layer_norm_eps": 1e-6, "rope_theta": 10000.0,
                "hidden_act": "gelu", "layer_norm_type": "layer_norm", "use_head": false
            }
        });
        let processor = json!({
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "min_pixels": 3136, "max_pixels": 4000000, "patch_size": 16, "merge_size": 2,
            "rescale_factor": 0.00392156862745098
        });
        Ov2Config::from_json(&config, &processor).unwrap()
    }

    #[test]
    fn config_reads_the_checkpoint_geometry() {
        let c = cfg();
        assert_eq!((c.head_dim, c.patch_dim(), c.text_hidden), (64, 768, 2560));
        assert_eq!((c.min_pixels, c.max_pixels), (3136, 4_000_000));
    }

    #[test]
    fn image_parts_become_template_images_in_order() {
        let msgs = vec![json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "a.png"}},
            {"type": "text", "text": "hi"},
            {"type": "image", "image": "b.png"}
        ]})];
        let (out, items) = normalize(&msgs).unwrap();
        assert_eq!(items, vec![json!({"url": "a.png"}), json!({"url": "b.png"})]);
        assert_eq!(out[0]["content"][0], json!({"type": "image"}));
        assert!(normalize(&[json!({"role": "user", "content": [{"type": "audio", "audio": "x"}]})]).is_err());
    }
}
