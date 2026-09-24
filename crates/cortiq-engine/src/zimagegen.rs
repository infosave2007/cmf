//! Z-Image / Z-Image-Turbo text-to-image pipeline: prompt → Qwen3 chat
//! template → Qwen3-4B encoder (35 layers, `hidden_states[-2]`) → DiT
//! Euler loop (static-shift flow matching, optional CFG) → Flux VAE → RGB.
//!
//! Mirrors diffusers `ZImagePipeline.__call__` (0.40):
//! - σ = shift·s/(1+(shift−1)·s) over linspace(1, 1/N, N), terminal 0
//!   (Turbo shift 3, base shift 6 — the container stores its scheduler);
//! - the model is called at t = (1000 − 1000σ)/1000 and its output v is
//!   negated before the Euler step: x ← x + (σᵢ₊₁ − σᵢ)·(−pred);
//! - CFG (guidance > 0): pred = pos + g·(pos − neg) — NOT Lumina's
//!   uncond + g(cond − uncond); optional `cfg_normalization` c > 0 clips
//!   ‖pred‖ (the norm over the whole tensor) to c·‖pos‖; `cfg_truncation`
//!   τ ≤ 1 turns CFG off at steps whose t_norm = (1000 − t)/1000 > τ.
//! - negative prompt default "" through the same chat template.
//!
//! Stages load and drop in sequence (text encoder → DiT → VAE), each under
//! `gpu::image_stage_scope()`; several images of one prompt share the
//! text encoding and the prepared caption, and every image's latent is
//! finished before the VAE loads.
//!
//! Profiling: `CMF_ZIMAGE_PROF=1` prints in-process stage times (tokenize ·
//! text-encode · load · prepare · steps · vae · total) and the median step.
//! `CMF_INIT_LATENT=<raw f32 [1,16,H/8,W/8]>` injects the oracle noise.
//! `CMF_ZIMAGE_DIT_DIR=<diffusers transformer dir>` runs the DiT from the
//! source weights instead of the container (parity work only).

use crate::tokenizer::Tokenizer;
use crate::zimage::{ZImageDit, ZShape};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// `header.arch.arch_name` of a Z-Image container.
pub const ARCH_NAME: &str = "z_image";
/// Prompt token cap (diffusers `max_sequence_length`), applied to the
/// TEMPLATED string.
pub const MAX_TOKENS: usize = 512;

/// Per-model defaults stored in the container (`zimage.config_json`), so
/// the CLI needs no flags. Missing keys fall back to the Turbo recipe.
#[derive(Clone, Debug)]
pub struct ZDefaults {
    /// "turbo" or "base".
    pub variant: String,
    pub steps: usize,
    pub guidance: f32,
    pub shift: f32,
    pub height: usize,
    pub width: usize,
    /// 0 = off; c > 0 clips ‖pred‖ to c·‖pos‖.
    pub cfg_normalization: f32,
    pub cfg_truncation: f32,
    pub negative_prompt: String,
    pub max_sequence_length: usize,
}

impl Default for ZDefaults {
    fn default() -> Self {
        Self {
            variant: "turbo".into(),
            steps: crate::zimage::DEFAULT_STEPS,
            guidance: 0.0,
            shift: crate::zimage::DEFAULT_SHIFT,
            height: 1024,
            width: 1024,
            cfg_normalization: 0.0,
            cfg_truncation: 1.0,
            negative_prompt: String::new(),
            max_sequence_length: MAX_TOKENS,
        }
    }
}

impl ZDefaults {
    pub fn from_json(v: &serde_json::Value) -> Self {
        let d = Self::default();
        let f = |k: &str, dv: f32| v[k].as_f64().map(|x| x as f32).unwrap_or(dv);
        let u = |k: &str, dv: usize| v[k].as_u64().map(|x| x as usize).unwrap_or(dv);
        Self {
            variant: v["variant"].as_str().unwrap_or(&d.variant).to_string(),
            steps: u("steps", d.steps),
            guidance: f("guidance", d.guidance),
            shift: f("shift", d.shift),
            height: u("height", d.height),
            width: u("width", d.width),
            cfg_normalization: match &v["cfg_normalization"] {
                serde_json::Value::Bool(b) => *b as u8 as f32,
                x => x.as_f64().map(|x| x as f32).unwrap_or(d.cfg_normalization),
            },
            cfg_truncation: f("cfg_truncation", d.cfg_truncation),
            negative_prompt: v["negative_prompt"]
                .as_str()
                .unwrap_or(&d.negative_prompt)
                .to_string(),
            max_sequence_length: u("max_sequence_length", d.max_sequence_length),
        }
    }

    /// The defaults a container carries (`zimage.config_json`).
    pub fn of(model: &cortiq_core::CmfModel) -> Self {
        model
            .tensor_bytes("zimage.config_json")
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok())
            .map(|v| Self::from_json(&v))
            .unwrap_or_default()
    }
}

/// Generation parameters. `ZParams::from_defaults` fills them from the
/// container; every field can be overridden.
#[derive(Clone, Debug)]
pub struct ZParams {
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub seed: u64,
    pub shift: f32,
    pub max_tokens: usize,
    /// Classifier-free guidance scale; 0 disables CFG (one forward/step).
    pub guidance: f32,
    /// None = the container's default negative prompt ("" by default).
    pub negative_prompt: Option<String>,
    /// 0 = off; c > 0 clips ‖pred‖ to c·‖pos‖ (diffusers True = 1.0).
    pub cfg_normalization: f32,
    /// CFG only while t_norm ≤ this (1.0 = every step).
    pub cfg_truncation: f32,
    /// Images per prompt; image i uses seed + i.
    pub num_images: usize,
}

impl Default for ZParams {
    fn default() -> Self {
        Self::from_defaults(&ZDefaults::default())
    }
}

impl ZParams {
    pub fn from_defaults(d: &ZDefaults) -> Self {
        Self {
            height: d.height,
            width: d.width,
            steps: d.steps,
            seed: 42,
            shift: d.shift,
            max_tokens: d.max_sequence_length,
            guidance: d.guidance,
            negative_prompt: None,
            cfg_normalization: d.cfg_normalization,
            cfg_truncation: d.cfg_truncation,
            num_images: 1,
        }
    }
}

/// One generated image: RGB u8 [height, width, 3] row-major.
pub struct ZImage {
    pub rgb: Vec<u8>,
    pub height: usize,
    pub width: usize,
    pub seed: u64,
}

impl ZImage {
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

/// `"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"` —
/// the Qwen3 template with `enable_thinking=True` (no think block), no
/// system prompt.
pub fn chat_template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// Template → ids (Qwen2 BPE, NFC, no BOS), truncated to `max_tokens`.
pub fn prompt_ids(tok: &Tokenizer, prompt: &str, max_tokens: usize) -> Vec<u32> {
    let mut ids = tok.encode(&chat_template(prompt));
    ids.truncate(max_tokens);
    ids
}

/// Standard normal draw (SplitMix64 + Box-Muller), or the raw f32 file in
/// `CMF_INIT_LATENT`.
fn gauss_latent(n: usize, seed: u64) -> Result<Vec<f32>, String> {
    if let Ok(path) = std::env::var("CMF_INIT_LATENT") {
        let b = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
        if b.len() != n * 4 {
            return Err(format!("{path}: {} floats, the latent needs {n}", b.len() / 4));
        }
        return Ok(b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect());
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

fn trace_write(dir: &str, img: usize, name: &str, v: &[f32]) -> Result<(), String> {
    let path = if img == 0 {
        format!("{dir}/{name}.f32")
    } else {
        format!("{dir}/img{img}_{name}.f32")
    };
    let _ = std::fs::create_dir_all(dir);
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&path, b).map_err(|e| format!("{path}: {e}"))
}

fn prof_on() -> bool {
    std::env::var("CMF_ZIMAGE_PROF").is_ok_and(|v| v != "0")
}

/// Classifier-free guidance combine, in place on `pos` (diffusers order
/// and f32 arithmetic): pred = pos + g·(pos − neg); with c > 0 the whole
/// tensor is scaled down to ‖pred‖ ≤ c·‖pos‖.
pub fn cfg_combine(pos: &mut [f32], neg: &[f32], g: f32, norm_clip: f32) {
    let pos_norm = if norm_clip > 0.0 {
        pos.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt()
    } else {
        0.0
    };
    for (p, &n) in pos.iter_mut().zip(neg) {
        *p += g * (*p - n);
    }
    if norm_clip > 0.0 {
        let new_norm = pos.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        let max_norm = (pos_norm as f32 * norm_clip) as f64;
        if new_norm > max_norm {
            let f = (max_norm / new_norm) as f32;
            for p in pos.iter_mut() {
                *p *= f;
            }
        }
    }
}

/// Stage times of one `generate_images` call (seconds).
#[derive(Clone, Debug, Default)]
pub struct ZTimings {
    pub text_encode: f64,
    pub dit_load: f64,
    pub prepare: f64,
    pub steps: Vec<f64>,
    pub vae: f64,
    pub total: f64,
}

impl ZTimings {
    pub fn median_step(&self) -> f64 {
        let mut s = self.steps.clone();
        if s.is_empty() {
            return 0.0;
        }
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }
}

/// Generate one image (the first of `generate_images`). RGB u8
/// [height, width, 3].
pub fn generate(
    model_path: &Path,
    prompt: &str,
    p: &ZParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, String> {
    let p1 = ZParams {
        num_images: 1,
        ..p.clone()
    };
    let (mut imgs, _) = generate_images(model_path, prompt, &p1, |_, i, n| progress(i, n))?;
    Ok(imgs.remove(0).rgb)
}

/// Generate `p.num_images` images for one prompt. `progress(image, step,
/// steps)` is called after every DiT step.
pub fn generate_images(
    model_path: &Path,
    prompt: &str,
    p: &ZParams,
    mut progress: impl FnMut(usize, usize, usize),
) -> Result<(Vec<ZImage>, ZTimings), String> {
    if p.height % 16 != 0 || p.width % 16 != 0 || p.height == 0 || p.width == 0 {
        return Err(format!(
            "height and width must be positive multiples of 16 (got {}x{})",
            p.width, p.height
        ));
    }
    if p.steps == 0 {
        return Err("steps must be at least 1".into());
    }
    let t_all = Instant::now();
    let mut tm = ZTimings::default();
    let model = Arc::new(
        cortiq_core::CmfModel::open(model_path)
            .map_err(|e| format!("{}: {e}", model_path.display()))?,
    );
    if model.header.arch.arch_name != ARCH_NAME {
        return Err(format!(
            "{}: architecture '{}' is not {ARCH_NAME}",
            model_path.display(),
            model.header.arch.arch_name
        ));
    }
    let vocab = model
        .vocab
        .as_deref()
        .ok_or("Z-Image .cmf has no embedded tokenizer")?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
    let defaults = ZDefaults::of(&model);
    let do_cfg = p.guidance > 0.0;
    // Device up + kernels compiled beside the host-side loading below.
    let warm = crate::zimage::gpu_allowed().then(|| std::thread::spawn(crate::gpu::zimage_warmup));

    // ── DiT host weights, then the text encoder on the CPU while a helper
    // thread uploads the device planes (they do not depend on the caption).
    let t0 = Instant::now();
    let _stage = crate::gpu::image_stage_scope();
    let dit = match std::env::var("CMF_ZIMAGE_DIT_DIR") {
        Ok(dir) => ZImageDit::load_dir(Path::new(&dir))?,
        Err(_) => ZImageDit::from_cmf(&model)?,
    };
    tm.dit_load = t0.elapsed().as_secs_f64();
    let ids = prompt_ids(&tok, prompt, p.max_tokens);
    let neg_text = p
        .negative_prompt
        .clone()
        .unwrap_or_else(|| defaults.negative_prompt.clone());
    let neg_ids = do_cfg.then(|| prompt_ids(&tok, &neg_text, p.max_tokens));
    let t0 = Instant::now();
    let overlap = crate::zimage::gpu_allowed()
        && std::env::var("CMF_ZIMAGE_OVERLAP").as_deref() != Ok("0")
        && std::env::var("CMF_ZIMAGE_TE_GPU").as_deref() != Ok("1")
        && std::env::var("CMF_ZIMAGE_TE_DEV").is_err();
    let (te, preload) = std::thread::scope(|sc| {
        let helper = overlap.then(|| {
            sc.spawn(|| {
                let t = Instant::now();
                let ok = dit.preload_device();
                (ok, t.elapsed().as_secs_f64())
            })
        });
        let t = Instant::now();
        let te = (|| -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
            // The text encoder runs on the CPU (measured, B2): on the 3090
            // the per-op device path took 3.2–4.2 s against 0.6 s here and
            // moved v_0 by 4–10 %. vk2 traced that to its host a8w8 arm
            // (the probe's CPU turns and the sub-gate projections), not the
            // device; with exact host fallback the device projections land
            // at the DiT's own floor, but stay 3–4 s (docs/ZIMAGE.md, "Text
            // encoder on the device"). `pause_gpu` is process-wide, so
            // the pool's workers stay off the device too (`cpu_scope` would
            // not); the plane upload beside it goes straight to the device
            // context, which the pause does not gate.
            // `CMF_ZIMAGE_TE_GPU=1` restores the device arm for A/B work.
            // `CMF_ZIMAGE_TE_DEV=all|q,k,…`: those projections through the
            // device GEMM of their codec, the rest exact on the host (the
            // device-TE experiment, vk2; see qwen3te `set_device_ops`).
            let te_dev = std::env::var("CMF_ZIMAGE_TE_DEV").ok();
            let _cpu = (std::env::var("CMF_ZIMAGE_TE_GPU").as_deref() != Ok("1") && te_dev.is_none())
                .then(crate::gpu::pause_gpu);
            let mut enc = crate::qwen3te::Qwen3Encoder::from_cmf(&model)?;
            // Weight-only exact q8 projections (B2): the default a8w8
            // kernel triples the caption error (h_m2 1.35e-2 vs 4.1e-3).
            // `CMF_ZIMAGE_TE_EXACT=0` = the a8w8 arm.
            enc.set_exact_q8(std::env::var("CMF_ZIMAGE_TE_EXACT").as_deref() != Ok("0"));
            if let Some(spec) = &te_dev {
                enc.set_device_ops(spec);
            }
            let cap = enc.encode(&ids);
            let ncap = neg_ids.as_ref().map(|n| enc.encode(n));
            Ok((cap, ncap))
        })();
        let te_s = t.elapsed().as_secs_f64();
        (te.map(|v| (v, te_s)), helper.map(|h| h.join().unwrap_or((false, 0.0))))
    });
    let ((cap, ncap), te_s) = te?;
    if let Some(w) = warm {
        let _ = w.join();
    }
    tm.text_encode = te_s;
    let overlapped = t0.elapsed().as_secs_f64();
    if prof_on() {
        match preload {
            Some((ok, s)) => eprintln!(
                "zimage: text-encode {te_s:.3}s beside the plane upload {s:.3}s (ok {ok}) -> {overlapped:.3}s"
            ),
            None => eprintln!("zimage: text-encode {te_s:.3}s"),
        }
    }
    let t0 = Instant::now();
    let sig = crate::zimage::sigmas_torch_f32(p.steps, p.shift);
    let t_models: Vec<f32> = sig[..p.steps]
        .iter()
        .map(|&s| crate::zimage::t_model(s))
        .collect();
    let mods = dit.mods_for_steps(&t_models);
    let fscale = dit.final_scale_for_steps(&t_models);
    let t_mods = t0.elapsed().as_secs_f64();
    let per_mod = dit.cfg.n_mod_blocks() * 4 * dit.cfg.dim;
    let dim = dit.cfg.dim;
    static KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let next_key = || KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let shape = ZShape::new(p.height, p.width, ids.len());
    // Which steps run CFG (cfg truncation: t_norm = (1000 − t)/1000 > τ
    // turns it off).
    let cfg_at: Vec<bool> = (0..p.steps)
        .map(|i| do_cfg && !(p.cfg_truncation <= 1.0 && t_models[i] > p.cfg_truncation))
        .collect();
    let dev = crate::zimage::gpu_allowed();
    let mods_all = Some((&mods[..], &fscale[..]));
    let mut prep = dit.prepare_host(&cap, shape, next_key(), dev)?;
    let mut nprep = match (&ncap, &neg_ids) {
        (Some(nc), Some(ni)) => Some(dit.prepare_host(
            nc,
            ZShape::new(p.height, p.width, ni.len()),
            next_key(),
            dev,
        )?),
        _ => None,
    };
    let t_caps = t0.elapsed().as_secs_f64();
    // CFG steps run cond + uncond as ONE batch-2 device forward; steps
    // without CFG (Turbo, or past the truncation) use the single program.
    let pair_key = next_key();
    let pair_dev = match &nprep {
        Some(np) if dev && cfg_at.iter().any(|&c| c) => {
            dit.attach_device_pair(&prep, np, pair_key, mods_all)
        }
        _ => false,
    };
    let mut on_device = pair_dev;
    if dev && (cfg_at.iter().any(|&c| !c) || (do_cfg && !pair_dev)) {
        on_device |= dit.attach_device(&mut prep, mods_all);
    }
    if let Some(np) = nprep.as_mut() {
        if dev && !pair_dev {
            on_device |= dit.attach_device(np, mods_all);
        }
    }
    if dev && !on_device {
        // the device module says why (once); this is the consequence
        eprintln!(
            "zimage: the DiT runs on the CPU — expect minutes per image; \
             CMF_ZIMAGE_PROF=1 prints the stages"
        );
    }
    tm.prepare = t0.elapsed().as_secs_f64();
    if prof_on() {
        eprintln!(
            "zimage prepare: mods {:.3}s · captions {:.3}s · device {:.3}s",
            t_mods,
            t_caps - t_mods,
            tm.prepare - t_caps
        );
    }
    let c = dit.cfg.in_channels;
    let (lh, lw) = (shape.h_lat, shape.w_lat);
    let pd = dit.geom().patch_dim;
    // `CMF_ZIMAGE_TRACE=<dir>`: raw f32 [16,h,w] per step — `v_i` (the
    // guided prediction before the negation), `lat_{i+1}`, and under CFG
    // `vpos_i`/`vneg_i` — the oracle `run_*` names, for parity scripts.
    let trace = std::env::var("CMF_ZIMAGE_TRACE").ok();
    // The VAE loads (host) and uploads/compiles (device) on a helper thread
    // while the steps run — the device is busy with the DiT and the CPU is
    // idle then (B2: 0.3–0.4 s off the critical path).
    let vae_warm = {
        let m = model.clone();
        let dev = crate::zimage::gpu_allowed();
        std::thread::spawn(move || -> Result<crate::vae::VaeDecoder, String> {
            let vae = crate::vae::VaeDecoder::from_cmf(&m)?;
            if dev && crate::gpu::enabled() {
                crate::gpu::vae_prewarm(&vae.chain_args());
            }
            Ok(vae)
        })
    };
    let mut latents: Vec<Vec<f32>> = Vec::with_capacity(p.num_images);
    for img in 0..p.num_images {
        let mut lat = gauss_latent(c * lh * lw, p.seed.wrapping_add(img as u64))?;
        for i in 0..p.steps {
            let ts = Instant::now();
            let x_tok = crate::zimage::pad_rows_repeat_last(
                &crate::zimage::patchify(&lat, c, lh, lw),
                shape.n_img,
                shape.n_img_p,
                pd,
            );
            let m = &mods[i * per_mod..(i + 1) * per_mod];
            let fs = &fscale[i * dim..(i + 1) * dim];
            let apply_cfg = cfg_at[i];
            let pair = if apply_cfg && pair_dev {
                dit.step_pair_device(pair_key, shape.n_img, i, &x_tok, m, fs)
            } else {
                None
            };
            let (mut pred, neg) = match pair {
                Some((a, b)) => (a, Some(b)),
                None => (dit.step(&prep, i, &x_tok, m, fs), None),
            };
            if apply_cfg {
                let neg = match neg {
                    Some(n) => n,
                    None => dit.step(nprep.as_ref().expect("negative prepared"), i, &x_tok, m, fs),
                };
                if let Some(d) = &trace {
                    trace_write(d, img, &format!("vpos_{i}"), &crate::zimage::unpatchify(&pred, c, lh, lw))?;
                    trace_write(d, img, &format!("vneg_{i}"), &crate::zimage::unpatchify(&neg, c, lh, lw))?;
                }
                cfg_combine(&mut pred, &neg, p.guidance, p.cfg_normalization);
            }
            let v = crate::zimage::unpatchify(&pred, c, lh, lw);
            let dt = sig[i + 1] - sig[i];
            for (x, &vv) in lat.iter_mut().zip(&v) {
                *x += dt * (-vv);
            }
            if let Some(d) = &trace {
                trace_write(d, img, &format!("v_{i}"), &v)?;
                trace_write(d, img, &format!("lat_{}", i + 1), &lat)?;
            }
            tm.steps.push(ts.elapsed().as_secs_f64());
            if prof_on() {
                eprintln!(
                    "zimage: image {} step {}/{} {:.3}s{}",
                    img + 1,
                    i + 1,
                    p.steps,
                    ts.elapsed().as_secs_f64(),
                    if apply_cfg { " (cfg)" } else { "" }
                );
            }
            progress(img, i + 1, p.steps);
        }
        if let Ok(path) = std::env::var("CMF_ZIMAGE_LATENT_OUT") {
            let path = if p.num_images > 1 {
                format!("{path}.{img}")
            } else {
                path
            };
            let b: Vec<u8> = lat.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, b).map_err(|e| format!("{path}: {e}"))?;
        }
        latents.push(lat);
    }
    drop(prep);
    drop(nprep);
    drop(dit);
    crate::gpu::zimage_release_dit();
    drop(_stage);

    // ── VAE ──
    let t0 = Instant::now();
    let mut out = Vec::with_capacity(latents.len());
    {
        let _stage = crate::gpu::image_stage_scope();
        let vae = vae_warm.join().map_err(|_| "VAE loader panicked".to_string())??;
        for (img, lat) in latents.iter().enumerate() {
            if lat.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "image {img}: the final latent is not finite (an f16 overflow on the \
                     device path, or a corrupt file); rerun with CMF_ZIMAGE_GPU=0 to \
                     use the CPU DiT"
                ));
            }
            let rgb = vae.decode_fast(lat, lh, lw);
            let (h, w) = (p.height, p.width);
            let plane = h * w;
            let mut u8s = vec![0u8; plane * 3];
            for px in 0..plane {
                for ch in 0..3 {
                    let v = (rgb[ch * plane + px] / 2.0 + 0.5).clamp(0.0, 1.0);
                    // numpy `(x*255).round()`: half to even
                    u8s[px * 3 + ch] = (v * 255.0).round_ties_even() as u8;
                }
            }
            out.push(ZImage {
                rgb: u8s,
                height: h,
                width: w,
                seed: p.seed.wrapping_add(img as u64),
            });
        }
    }
    tm.vae = t0.elapsed().as_secs_f64();
    tm.total = t_all.elapsed().as_secs_f64();
    if crate::zimage::gpu_allowed() {
        crate::gpu::zimage_flush_pipelines();
    }
    if prof_on() {
        eprintln!(
            "zimage stages: text-encode {:.2}s · dit-load {:.2}s · prepare {:.2}s · steps {:.2}s (median {:.3}s, {} forwards/step) · vae {:.2}s · total {:.2}s",
            tm.text_encode,
            tm.dit_load,
            tm.prepare,
            tm.steps.iter().sum::<f64>(),
            tm.median_step(),
            if do_cfg { 2 } else { 1 },
            tm.vae,
            tm.total
        );
    }
    Ok((out, tm))
}
