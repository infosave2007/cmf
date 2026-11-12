//! End-to-end MiniMax-H3-Image text → image: Qwen3-VL prompt encode →
//! six steps over the packed DiT on the sigma grid its output heads were
//! distilled for → the 2-D ViT decoder.
//!
//! The model is the MiniMax-H3 stack without the audio stream; the
//! sequence is `[text | reference images | target image]` and one latent
//! frame is denoised. Stages load and drop one at a time, as the video
//! pipeline next door does: the peak is one component, not their sum.
//!
//! ## Two modes, one checkpoint
//!
//! Text-to-image, and image editing: with reference pictures the prompt
//! is preceded by `Picture 1: <vision block>`, `Picture 2: …` — the
//! encoder sees each picture through the Qwen3-VL vision tower at a
//! 1024-pixel short edge — and the DiT also gets each picture's VAE
//! latent as a block of condition rows ahead of the target. Picture 1
//! is the picture to edit; the output takes its size unless one is
//! given.
//!
//! ## Six steps, no guidance
//!
//! The checkpoint is distilled: its output layer is a bank of six heads,
//! one per interval of the grid `1 → 0.9675 → 0.8875 → 0.75 → 0.5328 →
//! 0.2320 → 0`, and there is no classifier-free guidance. Fewer steps
//! walk the same grid in longer strides and average the heads they span,
//! which is the reference's own rule for a step that covers more than one
//! interval.

use crate::mmh3::{Layout, MiniMaxH3};
use crate::qwen3te::{ImageSpan, Qwen3Encoder};
use crate::qwen3vis::{self, VisionTower};
use crate::sampler::SplitMix64;
use crate::tokenizer::Tokenizer;
use crate::vae3d::{VideoVae, VideoVaeEncoder};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// `arch_name` of a packed MiniMax-H3-Image container.
pub const ARCH_NAME: &str = "minimax-h3-image";

/// The canvas the reference nodes default to; the model is tuned for
/// this area and up.
pub const DEFAULT_SIDE: usize = 2048;

/// Editing: the short edge the prompt encoder sees a reference at, and
/// the range the VAE's copy of it is brought into.
const REF_SHORT_EDGE: usize = 1024;
const REF_VAE_MAX_SHORT_EDGE: usize = 2048;
/// The reference's posterior-sample seed for a picture's latent.
const IMAGE_ENCODE_SEED: u64 = 42;
/// The vision-block token ids the H3 presentation flanks a picture with.
const VISION_START: u32 = 151_652;
const VISION_END: u32 = 151_653;

/// An aspect-preserving size with the short edge inside `[low, high]`,
/// each side rounded to the 32-pixel canvas grid.
pub fn size_short_edge(w: usize, h: usize, low: usize, high: usize) -> (usize, usize) {
    let short = w.min(h).max(1);
    let scale = short.clamp(low, high) as f64 / short as f64;
    let side = |s: usize| (((s as f64 * scale / 32.0).round() as usize).max(1)) * 32;
    (side(w), side(h))
}

/// The size an edit of this picture comes out at when none is asked for:
/// the picture's own, within the VAE's reference range.
pub fn edit_default_size(path: &Path) -> Result<(usize, usize), String> {
    let (w, h) = image::image_dimensions(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(size_short_edge(w as usize, h as usize, REF_SHORT_EDGE, REF_VAE_MAX_SHORT_EDGE))
}

/// The low end of the model's working range, as an area in pixels: the
/// reference server's default `min_generation_area`.
pub const MIN_AREA: usize = 2048 * 2048;

/// The canvas a `width`×`height` request is generated on: its own when it
/// has at least `min_area` pixels, otherwise the same aspect ratio grown
/// to that area, each side rounded to the 32-pixel grid — the reference's
/// `get_image_canvas_size`. The picture is then resized back.
pub fn canvas_size(width: usize, height: usize, min_area: usize) -> (usize, usize) {
    if width * height >= min_area || width == 0 || height == 0 {
        return (width, height);
    }
    let scale = (min_area as f64 / (width * height) as f64).sqrt();
    let side = |s: usize| ((s as f64 * scale / 32.0).round() as usize).max(1) * 32;
    (side(width), side(height))
}

/// An image file as RGB in [0, 1], `[3, h, w]`, resized (Lanczos) to
/// `w`×`h` when it is not that size already.
fn load_rgb(img: &image::RgbImage, w: usize, h: usize) -> Vec<f32> {
    let resized;
    let src = if (img.width() as usize, img.height() as usize) == (w, h) {
        img
    } else {
        resized = image::imageops::resize(img, w as u32, h as u32, image::imageops::FilterType::Lanczos3);
        &resized
    };
    let hw = w * h;
    let mut out = vec![0f32; 3 * hw];
    for (i, px) in src.pixels().enumerate() {
        for c in 0..3 {
            out[c * hw + i] = px.0[c] as f32 / 255.0;
        }
    }
    out
}

#[derive(Clone)]
pub struct H3ImageParams {
    pub width: usize,
    pub height: usize,
    /// Denoising steps: the grid's own count (6) unless fewer are asked.
    pub steps: usize,
    pub seed: u64,
    /// Prompt token cap (the reference truncates at 4096).
    pub max_tokens: usize,
    pub num_images: usize,
    /// Reference pictures (files), `Picture 1`, `Picture 2`, … in the
    /// prompt; the first is the picture to edit. Empty: text-to-image.
    pub images: Vec<String>,
    /// The smallest canvas the model is run at, in pixels ([`MIN_AREA`]);
    /// a smaller request is generated larger and resized back.
    pub min_area: usize,
}

impl Default for H3ImageParams {
    fn default() -> Self {
        Self {
            width: DEFAULT_SIDE,
            height: DEFAULT_SIDE,
            steps: 0,
            seed: 42,
            max_tokens: 4096,
            num_images: 1,
            images: Vec::new(),
            min_area: MIN_AREA,
        }
    }
}

pub struct H3Image {
    /// `[height][width][3]`, 8 bits a channel.
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
}

impl H3Image {
    /// PNG/JPEG by extension; `.ppm` writes P6.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("ppm"))
        {
            let mut ppm = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
            ppm.extend_from_slice(&self.rgb);
            return std::fs::write(path, ppm).map_err(|e| e.to_string());
        }
        image::RgbImage::from_raw(self.width as u32, self.height as u32, self.rgb.clone())
            .ok_or("output image dimensions overflow")?
            .save(path)
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Seconds per stage of one run.
#[derive(Default, Clone)]
pub struct H3ImageTimings {
    pub text_encode: f64,
    pub dit_load: f64,
    pub steps: Vec<f64>,
    pub vae: f64,
    pub total: f64,
    pub prompt_tokens: usize,
    pub image_tokens: usize,
    /// Editing: seconds in the vision tower and in the VAE encoder, and
    /// the condition rows the references add to the DiT's sequence.
    pub ref_vision: f64,
    pub ref_encode: f64,
    pub ref_tokens: usize,
}

impl H3ImageTimings {
    pub fn median_step(&self) -> f64 {
        let mut v = self.steps.clone();
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
    pub fn denoise(&self) -> f64 {
        self.steps.iter().sum()
    }
}

/// The decoder's `[3, h, w]` in [0, 1] as 8-bit interleaved RGB.
fn to_image(px: &[f32], width: usize, height: usize, seed: u64) -> H3Image {
    let hw = height * width;
    let mut rgb = vec![0u8; hw * 3];
    for i in 0..hw {
        for c in 0..3 {
            rgb[i * 3 + c] = (px[c * hw + i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    H3Image { rgb, width, height, seed }
}

/// Latent channels of the container's VAE.
fn vae_channels(model: &Arc<cortiq_core::CmfModel>) -> usize {
    model
        .tensor_bytes("vvae.config_json")
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
        .and_then(|c| c["z_channels"].as_u64())
        .unwrap_or(24) as usize
}

fn gauss(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
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
    out
}

/// The sigma walk of `steps` steps over a head bank's grid: the grid
/// itself at its own step count, and otherwise evenly spaced grid points
/// (a step then spans several intervals and averages their heads).
pub fn sigma_walk(grid: &[f64], steps: usize) -> Vec<f64> {
    let n = grid.len() - 1;
    if steps == 0 || steps >= n {
        return grid.to_vec();
    }
    let mut idx: Vec<usize> = (0..=steps)
        .map(|i| ((i * n) as f64 / steps as f64).round() as usize)
        .collect();
    idx.dedup();
    idx.iter().map(|&i| grid[i]).collect()
}

/// Text → images from a packed `.cmf`. `progress(image, step, steps)` is
/// called after every DiT step.
pub fn generate(
    path: &Path,
    prompt: &str,
    p: &H3ImageParams,
    mut progress: impl FnMut(usize, usize, usize),
) -> Result<(Vec<H3Image>, H3ImageTimings), String> {
    if p.width % 32 != 0 || p.height % 32 != 0 || p.width == 0 || p.height == 0 {
        return Err(format!(
            "width and height must be positive multiples of 32 (got {}x{})",
            p.width, p.height
        ));
    }
    // The device is trusted per stack, not by name: this file's own first
    // qkv weight against the host at DiT-scale activations, as the video
    // pipeline does. `CMF_MMH3_GPU=1/0` forces either way.
    let use_gpu = match std::env::var("CMF_MMH3_GPU").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => crate::videogen::mmh3_gpu_parity_probe(path).unwrap_or(false),
    };
    // Below the model's working range the picture is generated on a
    // larger canvas of the same aspect ratio and resized back.
    let (cw, ch) = canvas_size(p.width, p.height, p.min_area);
    let canvas = H3ImageParams {
        width: cw,
        height: ch,
        ..p.clone()
    };
    let (mut imgs, tm) = if use_gpu {
        generate_inner(path, prompt, &canvas, &mut progress)?
    } else {
        crate::gpu::cpu_scope(|| generate_inner(path, prompt, &canvas, &mut progress))?
    };
    if (cw, ch) != (p.width, p.height) {
        for img in imgs.iter_mut() {
            let full = image::RgbImage::from_raw(cw as u32, ch as u32, std::mem::take(&mut img.rgb))
                .ok_or("output image dimensions overflow")?;
            img.rgb = image::imageops::resize(
                &full,
                p.width as u32,
                p.height as u32,
                image::imageops::FilterType::Lanczos3,
            )
            .into_raw();
            img.width = p.width;
            img.height = p.height;
        }
    }
    Ok((imgs, tm))
}

fn generate_inner(
    path: &Path,
    prompt: &str,
    p: &H3ImageParams,
    progress: &mut dyn FnMut(usize, usize, usize),
) -> Result<(Vec<H3Image>, H3ImageTimings), String> {
    let t_all = Instant::now();
    let mut tm = H3ImageTimings::default();
    let model = Arc::new(
        cortiq_core::CmfModel::open(path).map_err(|e| format!("{}: {e}", path.display()))?,
    );
    let (lat_h, lat_w) = (p.height / 16, p.width / 16);

    // `CMF_H3_DECODE_ONLY=1` with `CMF_INIT_LATENT=<file>`: decode that
    // latent and nothing else — the VAE decoder alone, for comparing its
    // paths and codecs on one latent.
    if std::env::var("CMF_H3_DECODE_ONLY").as_deref() == Ok("1") {
        let f = std::env::var("CMF_INIT_LATENT").map_err(|_| "CMF_H3_DECODE_ONLY needs CMF_INIT_LATENT")?;
        let b = std::fs::read(&f).map_err(|e| format!("{f}: {e}"))?;
        let x: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let vae = VideoVae::from_cmf(&model)?;
        if x.len() != vae_channels(&model) * lat_h * lat_w {
            return Err(format!("{f}: {} floats, not a latent of {}x{}", x.len(), p.width, p.height));
        }
        let t = Instant::now();
        let px = vae.decode_image(&x, lat_h, lat_w);
        tm.vae = t.elapsed().as_secs_f64();
        tm.total = t_all.elapsed().as_secs_f64();
        if let Some(rep) = crate::vae3d::vae3d_prof_report() {
            eprint!("{rep}");
        }
        return Ok((vec![to_image(&px, p.width, p.height, p.seed)], tm));
    }

    // ── prompt ──
    // The H3 presentation is raw text: no chat template, no BOS, no
    // special tokens at all.
    let t = Instant::now();
    let vocab = model
        .vocab
        .as_deref()
        .ok_or("packaged .cmf has no embedded tokenizer")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
    let mut ids: Vec<u32> = Vec::new();
    let mut tags: Vec<u8> = Vec::new();
    let mut spans: Vec<ImageSpan> = Vec::new();
    let mut embeds: Vec<Vec<f32>> = Vec::new();
    let mut deepstack: Vec<Vec<f32>> = Vec::new();
    // each reference's latent with its (lat_h, lat_w)
    let mut refs: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    if !p.images.is_empty() {
        // Editing: every picture is presented to the encoder as
        // "Picture i: " and a vision block BEFORE the prompt, and
        // separately conditions the DiT as a latent. Both halves come
        // from the same file, each at its own size policy.
        // On the native CUDA backend the tower's products and attention run
        // on the card (its matrices as f32 planes); elsewhere it keeps the
        // host path the video pipeline uses.
        let tower = if crate::gpu::cuda_active() && crate::gpu::enabled_here() {
            VisionTower::from_cmf_device(&model)?
        } else {
            VisionTower::from_cmf(&model)?
        };
        let venc = VideoVaeEncoder::from_cmf(&model)?;
        for (i, file) in p.images.iter().enumerate() {
            let img = image::open(file).map_err(|e| format!("{file}: {e}"))?.to_rgb8();
            let (sw, sh) = (img.width() as usize, img.height() as usize);
            let tv = Instant::now();
            let (vw, vh) = size_short_edge(sw, sh, REF_SHORT_EDGE, REF_SHORT_EDGE);
            let seen = load_rgb(&img, vw, vh);
            for t in tok.encode(&format!("Picture {}: ", i + 1)) {
                ids.push(t);
                tags.push(1);
            }
            let (patches, gh, gw) =
                qwen3vis::preprocess(&seen, vh, vw, tower.patch_size, tower.temporal_patch, tower.merge);
            let (merged, deep) = tower.forward(&patches, gh, gw);
            let n_img = merged.len() / tower.out_hidden;
            // The whole block carries the VIDEO tag, the flanking markers
            // included.
            ids.push(VISION_START);
            tags.push(0);
            let start = ids.len();
            for _ in 0..n_img {
                ids.push(VISION_START); // a placeholder the embed replaces
                tags.push(0);
            }
            ids.push(VISION_END);
            tags.push(0);
            spans.push(ImageSpan {
                start,
                len: n_img,
                merged_h: gh / tower.merge,
                merged_w: gw / tower.merge,
            });
            embeds.push(merged);
            if deepstack.is_empty() {
                deepstack = deep;
            } else {
                for (a, b) in deepstack.iter_mut().zip(deep) {
                    a.extend_from_slice(&b);
                }
            }
            tm.ref_vision += tv.elapsed().as_secs_f64();
            let te = Instant::now();
            let (zw, zh) = size_short_edge(sw, sh, REF_SHORT_EDGE, REF_VAE_MAX_SHORT_EDGE);
            let pic: Vec<f32> = load_rgb(&img, zw, zh).iter().map(|&v| v * 2.0 - 1.0).collect();
            // `CMF_H3_REF_MEAN=1`: the posterior mean instead of a sample (a
            // measurement switch).
            let (z, lh, lw) = if std::env::var("CMF_H3_REF_MEAN").as_deref() == Ok("1") {
                venc.encode_frame(&pic, zh, zw)
            } else {
                venc.encode_frame_sampled(&pic, zh, zw, IMAGE_ENCODE_SEED)
            };
            // `CMF_H3_REF_DECODE=<prefix>`: each reference's latent decoded
            // straight back to `<prefix><i>.png` — the VAE's round trip on
            // the picture, to tell an encoder fault from a denoiser one.
            if let Ok(prefix) = std::env::var("CMF_H3_REF_DECODE") {
                let px = VideoVae::from_cmf(&model)?.decode_image(&z, lh, lw);
                let hw = zh * zw;
                let mut rgb = vec![0u8; hw * 3];
                for k in 0..hw {
                    for c in 0..3 {
                        rgb[k * 3 + c] = (px[c * hw + k] * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
                H3Image { rgb, width: zw, height: zh, seed: 0 }
                    .save(Path::new(&format!("{prefix}{}.png", i + 1)))?;
                let mut b = Vec::with_capacity(z.len() * 4);
                for v in &z {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(format!("{prefix}{}.f32", i + 1), &b).map_err(|e| e.to_string())?;
            }
            refs.push((z, lh, lw));
            tm.ref_encode += te.elapsed().as_secs_f64();
            tracing::info!(
                "reference {}: {sw}x{sh}, seen at {vw}x{vh} ({n_img} tokens, {:.1}s), encoded at {zw}x{zh} ({:.1}s)",
                i + 1,
                te.duration_since(tv).as_secs_f64(),
                te.elapsed().as_secs_f64()
            );
        }
    }
    for t in tok.encode(prompt) {
        ids.push(t);
        tags.push(1);
    }
    if ids.is_empty() {
        ids.push(151643); // the pad id, as the reference does for ""
        tags.push(1);
    }
    ids.truncate(p.max_tokens);
    tags.truncate(ids.len());
    if spans.last().is_some_and(|s| s.start + s.len > ids.len()) {
        return Err(format!(
            "the reference pictures alone take more than --max-sequence-length {} tokens",
            p.max_tokens
        ));
    }
    tm.prompt_tokens = ids.len();
    // The prompt encoder is a one-shot pass over 12 GB of weights; the
    // GEMM contention kill stays disarmed until it is done.
    crate::gpu::mm_kill_arm(false);
    // `CMF_TE_LOAD=<file>`: the conditioning from a `CMF_TE_DUMP` of the
    // same prompt instead of this file's encoder — another encoder's
    // states under this file's DiT, to price the encoder's codec alone.
    let states = match std::env::var("CMF_TE_LOAD") {
        Ok(f) => {
            let b = std::fs::read(&f).map_err(|e| format!("CMF_TE_LOAD {f}: {e}"))?;
            if b.len() < 16 {
                return Err(format!("CMF_TE_LOAD {f}: not a conditioning dump"));
            }
            let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
            let w = u64::from_le_bytes(b[8..16].try_into().unwrap()) as usize;
            if n != ids.len() || b.len() != 16 + n * w * 4 {
                return Err(format!(
                    "CMF_TE_LOAD {f}: {n} tokens x {w}, this prompt is {} tokens",
                    ids.len()
                ));
            }
            b[16..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        Err(_) => {
            // `CMF_TE_FROM=<file.cmf>`: the prompt encoder (`te.*`) of
            // another container under this file's vision tower and DiT —
            // another codec of the encoder on an edit, where the dump of
            // a text-only run does not reach.
            let te_model = match std::env::var("CMF_TE_FROM") {
                Ok(f) => Arc::new(cortiq_core::CmfModel::open(Path::new(&f)).map_err(|e| format!("CMF_TE_FROM {f}: {e}"))?),
                Err(_) => model.clone(),
            };
            let enc = Qwen3Encoder::from_cmf(&te_model)?;
            let states = enc.encode_with_images(&ids, &spans, &embeds, &deepstack);
            drop(enc);
            if !Arc::ptr_eq(&te_model, &model) {
                te_model.advise_done(|n| n.starts_with("te."));
                crate::gpu::stage_done(&te_model);
            }
            states
        }
    };
    drop(embeds);
    drop(deepstack);
    crate::gpu::mm_kill_arm(true);
    if let Ok(dump) = std::env::var("CMF_TE_DUMP") {
        let w = states.len() / ids.len().max(1);
        let mut b = Vec::with_capacity(16 + states.len() * 4);
        b.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        b.extend_from_slice(&(w as u64).to_le_bytes());
        for v in &states {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&dump, &b).map_err(|e| format!("CMF_TE_DUMP {dump}: {e}"))?;
    }
    tm.text_encode = t.elapsed().as_secs_f64();
    // The encoder ran its once-per-generation pass: release its page
    // cache and whatever the device kept of it before the DiT loads.
    model.advise_done(|n| n.starts_with("te.") || n.starts_with("vis."));
    crate::gpu::stage_done(&model);

    // ── denoise ──
    let t = Instant::now();
    let dit = MiniMaxH3::from_cmf(&model)?;
    if !dit.image {
        return Err(format!(
            "{}: not a MiniMax-H3-Image container (its DiT has an audio stream)",
            path.display()
        ));
    }
    let grid: Vec<f64> = if dit.pdd_steps > 1 {
        dit.pdd_sigmas.clone()
    } else {
        crate::videogen::sigmas(p.steps.max(1), dit.shift_video)
    };
    let sg = sigma_walk(&grid, p.steps);
    let steps = sg.len() - 1;
    let ref_dims: Vec<(usize, usize)> = refs.iter().map(|r| (r.1, r.2)).collect();
    let layout = Layout::image(ids.len(), lat_h, lat_w, &ref_dims, &tags);
    tm.image_tokens = (lat_h / 2) * (lat_w / 2);
    tm.ref_tokens = layout.seq_len - ids.len() - tm.image_tokens;
    let ref_rows: Vec<(&[f32], usize, usize)> = refs.iter().map(|r| (&r.0[..], r.1, r.2)).collect();
    if let Ok(dir) = std::env::var("CMF_H3_DUMP") {
        // the sequence's geometry, beside the attention operands the chain
        // dumps there
        let _ = std::fs::create_dir_all(&dir);
        let meta = format!(
            "{{\"text\": {}, \"refs\": {}, \"hp\": {}, \"wp\": {}, \"sigmas\": {:?}}}",
            ids.len(),
            tm.ref_tokens,
            lat_h / 2,
            lat_w / 2,
            sg
        );
        let _ = std::fs::write(format!("{dir}/layout.json"), meta);
    }
    let text = dit.refine_text(&states, ids.len());
    drop(states);
    tm.dit_load = t.elapsed().as_secs_f64();
    let prof = std::env::var_os("CMF_ANIM_PROF").is_some();
    let rms = |x: &[f32]| {
        (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len().max(1) as f64).sqrt()
    };
    if prof {
        eprintln!(
            "  text {} tok, refined rms {:.4}, sigmas {:?}",
            ids.len(),
            rms(&text),
            sg.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
        );
    }
    let mut latents: Vec<(Vec<f32>, u64)> = Vec::new();
    for img in 0..p.num_images.max(1) {
        let seed = p.seed.wrapping_add(img as u64);
        let mut x = match std::env::var("CMF_INIT_LATENT") {
            Ok(f) => {
                let b = std::fs::read(&f).map_err(|e| format!("{f}: {e}"))?;
                if b.len() != dit.latents_dim * lat_h * lat_w * 4 {
                    return Err(format!(
                        "{f}: {} floats, the latent needs {}",
                        b.len() / 4,
                        dit.latents_dim * lat_h * lat_w
                    ));
                }
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
            Err(_) => gauss(dit.latents_dim * lat_h * lat_w, seed),
        };
        for i in 0..steps {
            let ts = Instant::now();
            let (s, s_next) = (sg[i], sg[i + 1]);
            let d = dit.forward_image(&layout, &text, &x, &ref_rows, s, s_next);
            let step = (s_next - s) as f32;
            for (v, &dv) in x.iter_mut().zip(&d) {
                *v += step * dv;
            }
            tm.steps.push(ts.elapsed().as_secs_f64());
            if prof {
                eprintln!(
                    "  step {i}: sigma {s:.4}->{s_next:.4} vel {:.4} | latent {:.4}",
                    rms(&d),
                    rms(&x)
                );
            }
            progress(img, i + 1, steps);
        }
        if let Ok(f) = std::env::var("CMF_LATENT_OUT") {
            let mut b = Vec::with_capacity(x.len() * 4);
            for v in &x {
                b.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&f, &b).map_err(|e| format!("CMF_LATENT_OUT {f}: {e}"))?;
        }
        latents.push((x, seed));
    }
    drop(dit);
    model.advise_done(|n| n.starts_with("dit."));
    crate::gpu::stage_done(&model);

    // ── decode ──
    let t = Instant::now();
    let vae = VideoVae::from_cmf(&model)?;
    let mut out = Vec::with_capacity(latents.len());
    for (x, seed) in latents {
        let px = vae.decode_image(&x, lat_h, lat_w);
        out.push(to_image(&px, p.width, p.height, seed));
    }
    tm.vae = t.elapsed().as_secs_f64();
    tm.total = t_all.elapsed().as_secs_f64();
    if let Some(rep) = crate::mmh3::mmh3_prof_report() {
        eprintln!("{rep}");
    }
    if let Some(rep) = crate::gpu::h3_prof_report() {
        eprintln!("{rep}");
    }
    if let Some(rep) = crate::vae3d::vae3d_prof_report() {
        eprint!("{rep}");
    }
    Ok((out, tm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_below_the_working_range_gets_a_larger_canvas() {
        // at or above the area: the request's own size
        assert_eq!(canvas_size(2048, 2048, MIN_AREA), (2048, 2048));
        assert_eq!(canvas_size(4096, 1024, MIN_AREA), (4096, 1024));
        // below: the same aspect ratio at the area, on the 32-pixel grid
        assert_eq!(canvas_size(1024, 1024, MIN_AREA), (2048, 2048));
        assert_eq!(canvas_size(512, 512, MIN_AREA), (2048, 2048));
        assert_eq!(canvas_size(1280, 720, MIN_AREA), (2720, 1536));
        assert_eq!(canvas_size(768, 1344, MIN_AREA), (1536, 2720));
        // 1024 turns it off for every real size
        assert_eq!(canvas_size(1024, 1024, 1024), (1024, 1024));
        assert_eq!(canvas_size(32, 32, 1024), (32, 32));
    }

    #[test]
    fn the_walk_is_the_grid_at_its_own_step_count() {
        let g = [1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232, 0.0];
        assert_eq!(sigma_walk(&g, 0), g.to_vec());
        assert_eq!(sigma_walk(&g, 6), g.to_vec());
        assert_eq!(sigma_walk(&g, 9), g.to_vec());
        assert_eq!(sigma_walk(&g, 3), vec![1.0, 0.8875, 0.5328, 0.0]);
        assert_eq!(sigma_walk(&g, 2), vec![1.0, 0.75, 0.0]);
        assert_eq!(sigma_walk(&g, 1), vec![1.0, 0.0]);
    }

    #[test]
    fn a_reference_keeps_its_aspect_on_the_canvas_grid() {
        // already inside the range: untouched
        assert_eq!(size_short_edge(1024, 1024, 1024, 2048), (1024, 1024));
        assert_eq!(size_short_edge(1536, 2048, 1024, 2048), (1536, 2048));
        // a small picture comes up to the 1024 short edge, a large one
        // down to 2048, each side to a multiple of 32
        assert_eq!(size_short_edge(512, 768, 1024, 2048), (1024, 1536));
        assert_eq!(size_short_edge(4000, 3000, 1024, 2048), (2720, 2048));
        // what the prompt encoder sees: the short edge at exactly 1024
        assert_eq!(size_short_edge(4000, 3000, 1024, 1024), (1376, 1024));
        assert_eq!(size_short_edge(640, 480, 1024, 1024), (1376, 1024));
    }
}
