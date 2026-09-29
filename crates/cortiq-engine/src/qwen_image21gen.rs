//! Qwen-Image-2.1 pipeline: text-to-image (RGBA) and image-conditioned
//! generation/editing from ONE container.
//!
//! Mirrors diffusers `QwenImage21Pipeline.__call__`:
//! - the raw template `<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n
//!   {prompt}<|im_end|>\n<|im_start|>assistant\n` (with `<imageN><|vision_start|>
//!   <|image_pad|><|vision_end|>` per condition image), Qwen3-VL-8B's last
//!   layer BEFORE its final norm, the system rows dropped;
//! - condition images are resized to `output_resolution²` area (sides
//!   multiples of 32, the aspect of the image), fed to the vision tower
//!   composited over white and to the VAE as RGBA; their VAE latents
//!   replace the vision slots (one slot = 2×2 latent tokens);
//! - height/width are floored to multiples of 32; with condition images
//!   they default to the last image's aspect at `output_resolution²`;
//! - σ = linspace(1, 1/N, N) through the exponential dynamic shift at
//!   μ(target tokens) and the terminal stretch; the model is called at
//!   `timestep/1000` and `x ← x + (σᵢ₊₁ − σᵢ)·v`;
//! - true CFG only when `cfg > 1` and a negative prompt is given:
//!   `neg + g·(pos − neg)`;
//! - the prefix (text + condition images) is encoded once per prompt and
//!   its keys/values are reused every step (the pipeline's KV cache).
//!
//! Stages load and drop in sequence (text encoder → DiT → VAE).
//!
//! Parity/profiling knobs: `CMF_QI21_PROF=1` (stage times);
//! `CMF_INIT_LATENT=<raw f32 [h·w, 64] tokens>` (oracle noise);
//! `CMF_QI21_EMBEDS=<dir>` (oracle `prompt_embeds.f32` + `meta.json`
//! instead of the text encoder, text-to-image only);
//! `CMF_QI21_TRACE=<dir>` (`v_i`, `lat_i` per step);
//! `CMF_QI21_DUMP=<dir>` (the prompt's encoder features, text rows; with
//! `CMF_QI21_TE_ONLY=1` the run stops there);
//! `CMF_QI21_LATENT_IN=<raw f32 tokens>` (decode this latent instead);
//! `CMF_QI21_COND_LAT=<raw f32 [h·w, 64]>` (the condition latent instead of
//! the VAE encoder; one image);
//! `CMF_QI21_TE_DEV=all|q,k,…` (text-encoder projections on the device GEMM);
//! `CMF_QI21_FORCE=<dir>` (teacher forcing: step i ≥ 1 starts from
//! `<dir>/lat_i.f32` when it exists — the per-step error of one denoiser
//! call, without the trajectory's accumulation).

use crate::qwen_image21::{Qi21Dit, Qi21Layout, Seg, ARCH_NAME};
use crate::qwen_image21_vae::Qi21Vae;
use crate::tokenizer::Tokenizer;
use image::imageops::FilterType;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub const SYSTEM_PROMPT: &str = "Comprehend and analyze the provided prompt.";

/// Per-model defaults stored in the container (`qi21.config_json`).
#[derive(Clone, Debug)]
pub struct Qi21Defaults {
    pub steps: usize,
    pub height: usize,
    pub width: usize,
    pub cfg: f32,
    pub output_resolution: usize,
    pub system_prompt: String,
}

impl Default for Qi21Defaults {
    fn default() -> Self {
        Self {
            steps: 40,
            height: 1024,
            width: 1024,
            cfg: 1.0,
            output_resolution: 1024,
            system_prompt: SYSTEM_PROMPT.into(),
        }
    }
}

impl Qi21Defaults {
    pub fn of(model: &cortiq_core::CmfModel) -> Self {
        let d = Self::default();
        let Some(v) = model
            .tensor_bytes("qi21.config_json")
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
        else {
            return d;
        };
        let u = |k: &str, dv: usize| v[k].as_u64().map(|x| x as usize).unwrap_or(dv);
        Self {
            steps: u("steps", d.steps),
            height: u("height", d.height),
            width: u("width", d.width),
            cfg: v["cfg"].as_f64().map(|x| x as f32).unwrap_or(d.cfg),
            output_resolution: u("output_resolution", d.output_resolution),
            system_prompt: v["system_prompt"].as_str().unwrap_or(&d.system_prompt).to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qi21Params {
    /// None = the container default (or the condition image's aspect).
    pub height: Option<usize>,
    pub width: Option<usize>,
    pub steps: usize,
    pub seed: u64,
    /// True CFG scale; ≤ 1 (the default) = one forward per step.
    pub cfg: f32,
    pub negative_prompt: Option<String>,
    /// Condition images (edit / reference generation).
    pub images: Vec<PathBuf>,
    pub output_resolution: usize,
    pub num_images: usize,
}

impl Qi21Params {
    pub fn from_defaults(d: &Qi21Defaults) -> Self {
        Self {
            height: None,
            width: None,
            steps: d.steps,
            seed: 42,
            cfg: d.cfg,
            negative_prompt: None,
            images: Vec::new(),
            output_resolution: d.output_resolution,
            num_images: 1,
        }
    }
}

/// One generated image: RGBA u8 `[height, width, 4]`.
pub struct Qi21Image {
    pub rgba: Vec<u8>,
    pub height: usize,
    pub width: usize,
    pub seed: u64,
}

impl Qi21Image {
    /// Whether the image is meaningfully transparent: more than 1 % of the
    /// pixels below alpha 250 (an opaque generation lands at 253–255).
    pub fn has_alpha(&self) -> bool {
        let n = self.rgba.len() / 4;
        self.rgba.chunks_exact(4).filter(|p| p[3] < 250).count() * 100 > n
    }

    /// PNG keeps the alpha channel; JPEG and PPM are composited over white.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "png" {
            return image::RgbaImage::from_raw(self.width as u32, self.height as u32, self.rgba.clone())
                .ok_or("output image dimensions overflow")?
                .save(path)
                .map_err(|e| format!("{}: {e}", path.display()));
        }
        let rgb: Vec<u8> = self
            .rgba
            .chunks_exact(4)
            .flat_map(|p| {
                let a = p[3] as f32 / 255.0;
                [0, 1, 2].map(|c| (p[c] as f32 * a + 255.0 * (1.0 - a)).round() as u8)
            })
            .collect();
        if ext == "ppm" {
            let mut ppm = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
            ppm.extend_from_slice(&rgb);
            return std::fs::write(path, ppm).map_err(|e| e.to_string());
        }
        image::RgbImage::from_raw(self.width as u32, self.height as u32, rgb)
            .ok_or("output image dimensions overflow")?
            .save(path)
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Qi21Timings {
    pub text_encode: f64,
    pub vae_encode: f64,
    pub dit_load: f64,
    pub prefill: f64,
    pub steps: Vec<f64>,
    pub vae: f64,
    pub total: f64,
}

impl Qi21Timings {
    pub fn median_step(&self) -> f64 {
        let mut s = self.steps.clone();
        if s.is_empty() {
            return 0.0;
        }
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }
}

fn prof_on() -> bool {
    std::env::var("CMF_QI21_PROF").is_ok_and(|v| v != "0")
}

/// diffusers `calculate_dimensions`: the (w, h) of `area` at `ratio`,
/// rounded to multiples of 32.
pub fn calculate_dimensions(area: f64, ratio: f64) -> (usize, usize) {
    let w = (area * ratio).sqrt();
    let h = w / ratio;
    let r32 = |v: f64| ((v / 32.0).round_ties_even() * 32.0).max(32.0) as usize;
    (r32(w), r32(h))
}

/// The prompt template (text-to-image or with `n_images` vision slots).
pub fn template(system: &str, prompt: &str, n_images: usize) -> String {
    // Qwen has no BOS: an empty prompt leaves the encoder nothing to read
    let prompt = if prompt.is_empty() { " " } else { prompt };
    let mut slots = String::new();
    for i in 1..=n_images {
        if i > 1 {
            slots.push(' ');
        }
        slots.push_str(&format!("<image{i}><|vision_start|><|image_pad|><|vision_end|>"));
    }
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{slots}{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Rows of the system message (`apply_chat_template` of it alone) — the
/// encoder rows the pipeline drops.
pub fn drop_idx(tok: &Tokenizer, system: &str) -> usize {
    tok.encode(&format!("<|im_start|>system\n{system}<|im_end|>\n")).len()
}

fn gauss(n: usize, seed: u64) -> Result<Vec<f32>, String> {
    if let Ok(path) = std::env::var("CMF_INIT_LATENT") {
        let b = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
        if b.len() != n * 4 {
            return Err(format!("{path}: {} floats, the latent needs {n}", b.len() / 4));
        }
        return Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
    }
    let mut rng = crate::sampler::SplitMix64::new(seed);
    let mut u = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let (a, b) = (u().max(1e-300), u());
        let r = (-2.0 * a.ln()).sqrt();
        let ang = 2.0 * std::f64::consts::PI * b;
        out.push((r * ang.cos()) as f32);
        if out.len() < n {
            out.push((r * ang.sin()) as f32);
        }
    }
    Ok(out)
}

fn write_f32(dir: &str, name: &str, v: &[f32]) -> Result<(), String> {
    let _ = std::fs::create_dir_all(dir);
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{dir}/{name}.f32"), b).map_err(|e| format!("{dir}/{name}: {e}"))
}

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// A condition image, resized once for both the vision tower and the VAE.
struct Cond {
    /// RGBA in [0, 1], channel planes `[4, h, w]`.
    rgba: Vec<f32>,
    h: usize,
    w: usize,
}

fn load_cond(path: &Path, output_resolution: usize) -> Result<Cond, String> {
    let img = image::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .to_rgba8();
    let (iw, ih) = (img.width() as f64, img.height() as f64);
    let area = (output_resolution * output_resolution) as f64;
    let (w, h) = calculate_dimensions(area, iw / ih);
    // PIL resizes RGBA in premultiplied form (RGBa) so fully transparent
    // pixels do not bleed their colour into the edges; same here, and a
    // same-size "resize" stays a plain copy as it is in PIL
    let img = if (img.width() as usize, img.height() as usize) == (w, h) {
        img
    } else {
        let mut pm = img;
        for p in pm.pixels_mut() {
            let a = p.0[3] as u32;
            for c in 0..3 {
                p.0[c] = ((p.0[c] as u32 * a + 127) / 255) as u8;
            }
        }
        let mut r = image::imageops::resize(&pm, w as u32, h as u32, FilterType::Lanczos3);
        for p in r.pixels_mut() {
            let a = p.0[3] as u32;
            if a != 0 && a != 255 {
                for c in 0..3 {
                    p.0[c] = ((255 * p.0[c] as u32) / a).min(255) as u8;
                }
            }
        }
        r
    };
    let plane = w * h;
    let mut rgba = vec![0f32; 4 * plane];
    for (i, p) in img.pixels().enumerate() {
        for c in 0..4 {
            rgba[c * plane + i] = p.0[c] as f32 / 255.0;
        }
    }
    Ok(Cond { rgba, h, w })
}

/// The encoded prompt: the encoder rows after the system message, and the
/// joint-sequence layout (text runs and condition image blocks, target
/// excluded).
struct Encoded {
    /// `[rows, 4096]` encoder features of the text rows only, in order.
    text: Vec<f32>,
    /// Prefix runs (text lengths and condition image blocks).
    segs: Vec<Seg>,
}

/// Prompt → encoder features. `conds` supply the vision slots.
fn encode_prompt(
    model: &Arc<cortiq_core::CmfModel>,
    tok: &Tokenizer,
    system: &str,
    prompt: &str,
    conds: &[Cond],
) -> Result<Encoded, String> {
    let text = template(system, prompt, conds.len());
    let mut ids = tok.encode(&text);
    let pad_id = *tok
        .encode("<|image_pad|>")
        .first()
        .ok_or("tokenizer has no <|image_pad|>")?;
    let slots = ids.iter().filter(|&&t| t == pad_id).count();
    if slots != conds.len() {
        return Err(format!(
            "the prompt has {slots} <|image_pad|> slot(s) for {} condition image(s) — \
             the prompt text must not contain vision tokens",
            conds.len()
        ));
    }
    let drop = drop_idx(tok, system);
    let mut enc = crate::qwen3te::Qwen3Encoder::from_cmf(model)?;
    // `CMF_QI21_TE_DEV=all|q,k,…`: those projections through the device GEMM
    if let Ok(spec) = std::env::var("CMF_QI21_TE_DEV") {
        enc.set_device_ops(&spec);
    }
    let t_enc = Instant::now();
    let hs = enc.out_hidden();
    let hidden = if conds.is_empty() {
        enc.encode(&ids)
    } else {
        let vis = crate::qwen3vis::VisionTower::from_cmf(model)?;
        // expand each <|image_pad|> into its merged vision tokens
        let mut spans = Vec::new();
        let mut embeds = Vec::new();
        let mut deep: Vec<Vec<f32>> = Vec::new();
        let mut out_ids = Vec::with_capacity(ids.len());
        let mut ci = 0usize;
        for &id in &ids {
            if id != pad_id {
                out_ids.push(id);
                continue;
            }
            let c = conds.get(ci).ok_or("more <|image_pad|> slots than images")?;
            ci += 1;
            // composite over white for the vision tower
            let plane = c.h * c.w;
            let mut rgb = vec![0f32; 3 * plane];
            for p in 0..plane {
                let a = c.rgba[3 * plane + p];
                for ch in 0..3 {
                    rgb[ch * plane + p] = c.rgba[ch * plane + p] * a + (1.0 - a);
                }
            }
            let (patches, gh, gw) = crate::qwen3vis::preprocess(&rgb, c.h, c.w, 16, 2, 2);
            let tv = Instant::now();
            let (merged, ds) = vis.forward(&patches, gh, gw);
            if prof_on() {
                eprintln!("qi21: vision tower {}x{} patches {:.2}s", gh, gw, tv.elapsed().as_secs_f64());
            }
            let n = gh * gw / 4;
            spans.push(crate::qwen3te::ImageSpan {
                start: out_ids.len(),
                len: n,
                merged_h: gh / 2,
                merged_w: gw / 2,
            });
            out_ids.extend(std::iter::repeat_n(pad_id, n));
            embeds.push(merged);
            if deep.is_empty() {
                deep = ds;
            } else {
                for (d, s) in deep.iter_mut().zip(ds) {
                    d.extend(s);
                }
            }
        }
        ids = out_ids;
        let tl = Instant::now();
        let h = enc.encode_with_images(&ids, &spans, &embeds, &deep);
        if prof_on() {
            eprintln!("qi21: language model {} tokens {:.2}s", ids.len(), tl.elapsed().as_secs_f64());
        }
        h
    };
    if prof_on() {
        eprintln!("qi21: prompt encode {} tokens {:.2}s", ids.len(), t_enc.elapsed().as_secs_f64());
    }
    if ids.len() <= drop {
        return Err("the prompt encodes to nothing past the system message".into());
    }
    let mut feats = Vec::new();
    let mut segs = Vec::new();
    let mut run = 0usize;
    let mut ci = 0usize;
    let mut i = drop;
    while i < ids.len() {
        if ids[i] == pad_id {
            if run > 0 {
                segs.push(Seg::Text(run));
                run = 0;
            }
            let c = &conds[ci];
            ci += 1;
            segs.push(Seg::Image(c.h / 16, c.w / 16));
            i += c.h * c.w / (32 * 32);
        } else {
            feats.extend_from_slice(&hidden[i * hs..(i + 1) * hs]);
            run += 1;
            i += 1;
        }
    }
    if run > 0 {
        segs.push(Seg::Text(run));
    }
    Ok(Encoded { text: feats, segs })
}

/// Oracle prompt embeddings (parity only): `prompt_embeds.f32` over all
/// rows after the system message, `image_pad_mask.f32` marking the vision
/// slots (one condition image, its pixel size in `meta.cond_wh`).
fn embeds_from_dir(dir: &Path) -> Result<Encoded, String> {
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("meta.json")).map_err(|e| format!("{}: {e}", dir.display()))?,
    )
    .map_err(|e| e.to_string())?;
    let shape = meta["prompt_embeds"].as_array().ok_or("meta: prompt_embeds")?;
    let rows = shape[0].as_u64().unwrap_or(0) as usize;
    let all = read_f32(&dir.join("prompt_embeds.f32"))?;
    let hs = all.len() / rows.max(1);
    let mask = read_f32(&dir.join("image_pad_mask.f32")).unwrap_or_else(|_| vec![0.0; rows]);
    let (cw, ch) = match meta["cond_wh"].as_array() {
        Some(a) if a.len() == 2 => (a[0].as_u64().unwrap_or(0) as usize, a[1].as_u64().unwrap_or(0) as usize),
        _ => (0, 0),
    };
    let mut text = Vec::new();
    let mut segs = Vec::new();
    let (mut run, mut i) = (0usize, 0usize);
    while i < rows {
        if mask[i] > 0.5 {
            if run > 0 {
                segs.push(Seg::Text(run));
                run = 0;
            }
            segs.push(Seg::Image(ch / 16, cw / 16));
            while i < rows && mask[i] > 0.5 {
                i += 1;
            }
        } else {
            text.extend_from_slice(&all[i * hs..(i + 1) * hs]);
            run += 1;
            i += 1;
        }
    }
    if run > 0 {
        segs.push(Seg::Text(run));
    }
    Ok(Encoded { text, segs })
}

/// Generate `p.num_images` images. `progress(image, step, steps)` after
/// every DiT step.
pub fn generate_images(
    model_path: &Path,
    prompt: &str,
    p: &Qi21Params,
    mut progress: impl FnMut(usize, usize, usize),
) -> Result<(Vec<Qi21Image>, Qi21Timings), String> {
    let t_all = Instant::now();
    let mut tm = Qi21Timings::default();
    let model = Arc::new(
        cortiq_core::CmfModel::open(model_path).map_err(|e| format!("{}: {e}", model_path.display()))?,
    );
    if model.header.arch.arch_name != ARCH_NAME {
        return Err(format!(
            "{}: architecture '{}' is not {ARCH_NAME}",
            model_path.display(),
            model.header.arch.arch_name
        ));
    }
    let d = Qi21Defaults::of(&model);
    if p.steps == 0 {
        return Err("steps must be at least 1".into());
    }
    let conds = p
        .images
        .iter()
        .map(|path| load_cond(path, p.output_resolution))
        .collect::<Result<Vec<_>, _>>()?;
    // diffusers: height = height or output_resolution (a condition image
    // sets its own aspect at that area); the container's default size is
    // the same 1024 unless --reference-size moves it
    let (dh, dw) = match conds.last() {
        Some(c) => (c.h, c.w),
        None if p.output_resolution != d.output_resolution => (p.output_resolution, p.output_resolution),
        None => (d.height, d.width),
    };
    let height = p.height.unwrap_or(dh) / 32 * 32;
    let width = p.width.unwrap_or(dw) / 32 * 32;
    if height == 0 || width == 0 || height > 4096 || width > 4096 {
        return Err(format!("height and width must be in 32..=4096 (got {width}x{height})"));
    }
    let (lh, lw) = (height / 16, width / 16);
    let n = lh * lw;
    let do_cfg = p.cfg > 1.0 && p.negative_prompt.is_some();

    // ── prompt ──
    let t0 = Instant::now();
    let vocab = model.vocab.as_deref().ok_or("Qwen-Image-2.1 .cmf has no embedded tokenizer")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
    let (pos, neg) = {
        let _stage = crate::gpu::image_stage_scope();
        let pos = match std::env::var("CMF_QI21_EMBEDS") {
            Ok(dir) => embeds_from_dir(Path::new(&dir))?,
            _ => encode_prompt(&model, &tok, &d.system_prompt, prompt, &conds)?,
        };
        let neg = if do_cfg {
            Some(encode_prompt(
                &model,
                &tok,
                &d.system_prompt,
                p.negative_prompt.as_deref().unwrap_or(""),
                &conds,
            )?)
        } else {
            None
        };
        (pos, neg)
    };
    tm.text_encode = t0.elapsed().as_secs_f64();
    if let Ok(dir) = std::env::var("CMF_QI21_DUMP") {
        write_f32(&dir, "prompt_embeds", &pos.text)?;
        if std::env::var("CMF_QI21_TE_ONLY").as_deref() == Ok("1") {
            return Ok((Vec::new(), tm));
        }
    }

    // ── condition latents ──
    let t0 = Instant::now();
    let cond_tokens: Vec<Vec<f32>> = if conds.is_empty() {
        Vec::new()
    } else if let Ok(path) = std::env::var("CMF_QI21_COND_LAT") {
        vec![read_f32(Path::new(&path))?]
    } else {
        let _stage = crate::gpu::image_stage_scope();
        let vae = Qi21Vae::from_cmf(&model, true, false)?;
        conds
            .iter()
            .map(|c| {
                let img: Vec<f32> = c.rgba.iter().map(|&v| v * 2.0 - 1.0).collect();
                let planes = vae.encode_mean(&img, c.h, c.w)?;
                Ok(vae.normalize_to_tokens(&planes, (c.h / 16) * (c.w / 16)))
            })
            .collect::<Result<_, String>>()?
    };
    tm.vae_encode = t0.elapsed().as_secs_f64();
    if let Ok(dir) = std::env::var("CMF_QI21_DUMP") {
        for (i, c) in cond_tokens.iter().enumerate() {
            write_f32(&dir, &format!("cond_lat_{i}"), c)?;
        }
    }

    // ── DiT ──
    let t0 = Instant::now();
    let _stage = crate::gpu::image_stage_scope();
    let dit = Qi21Dit::from_cmf(&model)?;
    tm.dit_load = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let build = |e: &Encoded| -> Result<(Qi21Layout, Vec<f32>), String> {
        let hs = dit.cfg.dim;
        let mut segs = e.segs.clone();
        segs.push(Seg::Image(lh, lw));
        let layout = Qi21Layout { segs };
        let n_text: usize = e.segs.iter().filter_map(|s| if let Seg::Text(k) = s { Some(*k) } else { None }).sum();
        let text_rows = dit.embed_text(&e.text, n_text);
        let mut rows = Vec::with_capacity(layout.prefix_len() * hs);
        let (mut ti, mut ci) = (0usize, 0usize);
        for s in &e.segs {
            match *s {
                Seg::Text(k) => {
                    rows.extend_from_slice(&text_rows[ti * hs..(ti + k) * hs]);
                    ti += k;
                }
                Seg::Image(h, w) => {
                    rows.extend(dit.embed_image(&cond_tokens[ci], h * w));
                    ci += 1;
                }
            }
        }
        Ok((layout, rows))
    };
    let (layout, rows) = build(&pos)?;
    let prefix = dit.prefill(rows, &layout)?;
    let nprefix = match &neg {
        Some(e) => {
            let (l, r) = build(e)?;
            Some(dit.prefill(r, &l)?)
        }
        None => None,
    };
    tm.prefill = t0.elapsed().as_secs_f64();
    if prof_on() {
        eprintln!(
            "qi21: prefix {} rows ({} segments), target {lh}x{lw}, prefill {:.2}s",
            layout.prefix_len(),
            layout.segs.len() - 1,
            tm.prefill
        );
    }

    let sched = model
        .tensor_bytes("qi21.scheduler_json")
        .ok()
        .and_then(|b| serde_json::from_slice::<crate::qwen_imagegen::FlowMatchConfig>(b).ok())
        .unwrap_or_default();
    if p.steps < 2 {
        return Err("Qwen-Image-2.1 needs at least 2 steps".into());
    }
    let sig = crate::qwen_imagegen::flow_match_sigmas(p.steps, n, &sched)?;
    // the model sees timestep/1000 with timestep = σ·1000 (f32)
    let t_models: Vec<f32> = sig[..p.steps].iter().map(|&s| (s * 1000.0) / 1000.0).collect();
    let mods: Vec<_> = t_models.iter().map(|&t| dit.mods(t)).collect();
    let trace = std::env::var("CMF_QI21_TRACE").ok();
    let force = std::env::var("CMF_QI21_FORCE").ok();
    let c = dit.cfg.in_channels;
    let mut latents = Vec::with_capacity(p.num_images);
    for img in 0..p.num_images {
        let mut lat = gauss(n * c, p.seed.wrapping_add(img as u64))?;
        for i in 0..p.steps {
            if let Some(dir) = force.as_deref().filter(|_| i > 0) {
                let path = Path::new(dir).join(format!("lat_{i}.f32"));
                if path.exists() {
                    lat = read_f32(&path)?;
                    if lat.len() != n * c {
                        return Err(format!("{}: {} floats, the latent needs {}", path.display(), lat.len(), n * c));
                    }
                }
            }
            let ts = Instant::now();
            let mut v = dit.step(&prefix, &lat, &mods[i])?;
            if let Some(np) = &nprefix {
                let nv = dit.step(np, &lat, &mods[i])?;
                for (pv, &q) in v.iter_mut().zip(&nv) {
                    *pv = q + p.cfg * (*pv - q);
                }
            }
            let dt = sig[i + 1] - sig[i];
            for (x, &vv) in lat.iter_mut().zip(&v) {
                *x += dt * vv;
            }
            if let Some(dir) = &trace {
                write_f32(dir, &format!("v_{i}"), &v)?;
                write_f32(dir, &format!("lat_{}", i + 1), &lat)?;
            }
            tm.steps.push(ts.elapsed().as_secs_f64());
            if prof_on() {
                eprintln!("qi21: image {} step {}/{} {:.3}s", img + 1, i + 1, p.steps, ts.elapsed().as_secs_f64());
            }
            progress(img, i + 1, p.steps);
        }
        latents.push(lat);
    }
    drop(prefix);
    drop(nprefix);
    drop(dit);
    crate::gpu::qi21_release();
    drop(_stage);

    // ── VAE ──
    let t0 = Instant::now();
    let mut out = Vec::with_capacity(latents.len());
    {
        let _stage = crate::gpu::image_stage_scope();
        let vae = Qi21Vae::from_cmf(&model, false, true)?;
        for (img, lat) in latents.iter().enumerate() {
            if lat.iter().any(|v| !v.is_finite()) {
                return Err(format!("image {img}: the final latent is not finite"));
            }
            let lat_in;
            let lat = match std::env::var("CMF_QI21_LATENT_IN") {
                Ok(path) => {
                    lat_in = read_f32(Path::new(&path))?;
                    if lat_in.len() != lat.len() {
                        return Err(format!("{path}: {} floats, the latent needs {}", lat_in.len(), lat.len()));
                    }
                    &lat_in
                }
                Err(_) => lat,
            };
            let z = vae.denormalize_tokens(lat, n);
            let px = vae.decode(&z, lh, lw)?;
            let ch = vae.cfg.out_channels;
            let plane = height * width;
            let mut rgba = vec![255u8; plane * 4];
            for i in 0..plane {
                for k in 0..ch.min(4) {
                    let v = (px[k * plane + i] / 2.0 + 0.5).clamp(0.0, 1.0);
                    rgba[i * 4 + k] = (v * 255.0).round_ties_even() as u8;
                }
            }
            out.push(Qi21Image {
                rgba,
                height,
                width,
                seed: p.seed.wrapping_add(img as u64),
            });
        }
    }
    tm.vae = t0.elapsed().as_secs_f64();
    tm.total = t_all.elapsed().as_secs_f64();
    if prof_on() {
        eprintln!(
            "qi21 stages: text {:.2}s · vae-enc {:.2}s · dit-load {:.2}s · prefill {:.2}s · steps {:.2}s (median {:.3}s) · vae {:.2}s · total {:.2}s",
            tm.text_encode,
            tm.vae_encode,
            tm.dit_load,
            tm.prefill,
            tm.steps.iter().sum::<f64>(),
            tm.median_step(),
            tm.vae,
            tm.total
        );
    }
    Ok((out, tm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_round_to_32_like_the_reference() {
        assert_eq!(calculate_dimensions(1024.0 * 1024.0, 1.0), (1024, 1024));
        // 4:3 at 1024² → w = 1182.4 → 1184, h = 886.8 → 896
        assert_eq!(calculate_dimensions(1024.0 * 1024.0, 4.0 / 3.0), (1184, 896));
    }

    #[test]
    fn template_places_the_vision_slots_before_the_prompt() {
        let t = template("S", "hi", 2);
        assert!(t.contains(
            "user\n<image1><|vision_start|><|image_pad|><|vision_end|> <image2><|vision_start|><|image_pad|><|vision_end|>hi<|im_end|>"
        ));
        assert!(template("S", "", 0).contains("user\n <|im_end|>"));
    }
}
