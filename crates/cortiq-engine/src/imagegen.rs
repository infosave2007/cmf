//! End-to-end Lumina-Image 2.0 text→image: Gemma-2 prompt encode →
//! flow-matching Next-DiT loop → FLUX-VAE decode.
//!
//! Fourth increment of the image-generation runtime
//! (docs/GENERATIVE.ru.md). Mirrors diffusers Lumina2Pipeline: the
//! system-prompt template `"<sys> <Prompt Start> <prompt>"`, Gemma
//! hidden_states[-2] as caption features, FlowMatchEulerDiscrete with
//! static shift 6 (σ' = 6σ/(1+5σ) over linspace(1, 1/N, N), terminal
//! 0), the model called at t = 1−σ, CFG with per-row norm
//! rescaling and the sign flip before the Euler step. Loads stages
//! sequentially and drops each when done — peak RSS is one component
//! (Gemma 10.4 GB f32), not the sum.

use crate::dit::NextDit;
use crate::sampler::SplitMix64;
use crate::textenc::GemmaEncoder;
use crate::tokenizer::Tokenizer;
use crate::vae::VaeDecoder;
use std::path::Path;

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are an assistant designed to generate superior \
     images with the superior degree of image-text alignment based on textual prompts or user \
     prompts.";

pub struct GenParams {
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    /// ≤1 disables classifier-free guidance (single forward per step).
    pub guidance_scale: f32,
    /// Fraction of steps that run with CFG; past it only cond runs.
    pub cfg_trunc_ratio: f32,
    /// Rescale the guided prediction to the cond-branch norm per row.
    pub cfg_normalization: bool,
    pub seed: u64,
    pub system_prompt: Option<String>,
    /// Gemma prompt token cap (diffusers max_sequence_length).
    pub max_tokens: usize,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            height: 512,
            width: 512,
            steps: 30,
            guidance_scale: 4.0,
            cfg_trunc_ratio: 1.0,
            cfg_normalization: true,
            seed: 42,
            system_prompt: None,
            max_tokens: 256,
        }
    }
}

/// σ schedule: linspace(1, 1/N, N) through the static shift, plus the
/// terminal 0.
fn sigmas(steps: usize, shift: f64) -> Vec<f64> {
    let n = steps;
    let mut out: Vec<f64> = (0..n)
        .map(|i| {
            let s = if n == 1 {
                1.0
            } else {
                1.0 - i as f64 * (1.0 - 1.0 / n as f64) / (n - 1) as f64
            };
            shift * s / (1.0 + (shift - 1.0) * s)
        })
        .collect();
    out.push(0.0);
    out
}

fn gauss_latent(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut u = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        // Box-Muller; guard log(0).
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

/// (features, token count) of the conditional prompt plus the same
/// pair for the "" uncond prompt when CFG runs.
type CapFeats = (Vec<f32>, usize, Option<(Vec<f32>, usize)>);

/// Prompt → caption features (cond + optional uncond) through Gemma;
/// the encoder is dropped on return.
fn encode_prompt(
    tok: &Tokenizer,
    enc: &GemmaEncoder,
    prompt: &str,
    p: &GenParams,
    want_uncond: bool,
) -> CapFeats {
    let sys = p.system_prompt.as_deref().unwrap_or(DEFAULT_SYSTEM_PROMPT);
    let full = format!("{sys} <Prompt Start> {prompt}");
    let mut ids = tok.with_bos(tok.encode(&full));
    ids.truncate(p.max_tokens);
    // diffusers takes hidden_states[-2] — the stream entering the
    // last layer.
    let (_, streams) = enc.encode(&ids, true);
    let cap = streams[streams.len() - 2].clone();
    let cap_u = if want_uncond {
        let uncond_ids = tok.with_bos(tok.encode(""));
        let (_, s) = enc.encode(&uncond_ids, true);
        Some((s[s.len() - 2].clone(), uncond_ids.len()))
    } else {
        None
    };
    (cap, ids.len(), cap_u)
}

/// The flow-matching Euler loop over the Next-DiT.
#[allow(clippy::too_many_arguments)]
fn denoise(
    dit: &NextDit,
    cap: &[f32],
    cap_n: usize,
    cap_u: Option<&(Vec<f32>, usize)>,
    lh: usize,
    lw: usize,
    p: &GenParams,
    progress: &mut dyn FnMut(usize, usize),
) -> Vec<f32> {
    let mut latents = gauss_latent(dit.in_channels * lh * lw, p.seed);
    let sg = sigmas(p.steps, 6.0);
    // The caption embedding and its refiner blocks depend on the prompt
    // alone — not on the timestep, not on the latents. Refined once here
    // instead of inside every model call: at 30 steps under CFG that is
    // 2 evaluations where the loop used to do 60.
    let cap_r = dit.refine_caption(cap, cap_n);
    let cap_u_r = cap_u.map(|(cu, un)| (dit.refine_caption(cu, *un), *un));
    for i in 0..p.steps {
        let t = (1.0 - sg[i]) as f32;
        let mut pred = dit.forward_with_cap(&latents, lh, lw, &cap_r, cap_n, t);
        if let Some((cu, un)) = &cap_u_r {
            // CFG truncation: past the ratio only cond runs.
            if (i + 1) as f32 / p.steps as f32 <= p.cfg_trunc_ratio {
                let uncond = dit.forward_with_cap(&latents, lh, lw, cu, *un, t);
                let gs = p.guidance_scale;
                let mut comb: Vec<f32> = uncond
                    .iter()
                    .zip(&pred)
                    .map(|(&u, &c)| u + gs * (c - u))
                    .collect();
                if p.cfg_normalization {
                    // ‖cond‖/‖comb‖ per row (last dim), as diffusers.
                    for (cr, gr) in pred.chunks_exact(lw).zip(comb.chunks_exact_mut(lw)) {
                        let cn = cr.iter().map(|&v| v * v).sum::<f32>().sqrt();
                        let gn = gr.iter().map(|&v| v * v).sum::<f32>().sqrt();
                        if gn > 0.0 {
                            let f = cn / gn;
                            for v in gr.iter_mut() {
                                *v *= f;
                            }
                        }
                    }
                }
                pred = comb;
            }
        }
        // Lumina predicts toward the image (t=1); the scheduler
        // steps σ→0, hence the sign flip: x += (σ₊ − σ)·(−pred).
        let d = (sg[i + 1] - sg[i]) as f32;
        for (x, &v) in latents.iter_mut().zip(&pred) {
            *x -= d * v;
        }
        progress(i + 1, p.steps);
    }
    latents
}

fn to_rgb01(img: Vec<f32>) -> Vec<f32> {
    img.iter()
        .map(|&v| (v / 2.0 + 0.5).clamp(0.0, 1.0))
        .collect()
}

/// Generate an RGB image `[3, height, width]` in [0, 1] from `root`:
/// either a diffusers Lumina-Image 2.0 directory (tokenizer/
/// text_encoder/ transformer/ vae/, exact f32) or a packaged .cmf
/// file (`cortiq imagine-pack`, quantized + mmap). `progress` is
/// called after every denoise step.
pub fn generate(
    root: &Path,
    prompt: &str,
    p: &GenParams,
    mut progress: impl FnMut(usize, usize),
) -> Result<Vec<f32>, String> {
    if p.height % 16 != 0 || p.width % 16 != 0 {
        return Err("height/width must be multiples of 16".into());
    }
    let (lh, lw) = (p.height / 8, p.width / 8);
    let want_uncond = p.guidance_scale > 1.0;

    if root.is_file() {
        // ── packaged .cmf: one mmap, components stay quantized ──
        let model = std::sync::Arc::new(
            cortiq_core::CmfModel::open(root).map_err(|e| format!("{}: {e}", root.display()))?,
        );
        let vocab = model
            .vocab
            .as_deref()
            .ok_or("packaged .cmf has no embedded tokenizer")?;
        let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
        let (cap, cap_n, cap_u) = {
            let enc = GemmaEncoder::from_cmf(&model)?;
            encode_prompt(&tok, &enc, prompt, p, want_uncond)
        };
        let latents = {
            let dit = NextDit::from_cmf(&model)?;
            denoise(&dit, &cap, cap_n, cap_u.as_ref(), lh, lw, p, &mut progress)
        };
        let vae = VaeDecoder::from_cmf(&model)?;
        return Ok(to_rgb01(vae.decode(&latents, lh, lw)));
    }

    // ── diffusers directory: exact f32, stages load-and-drop ──
    let tok = Tokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))
        .map_err(|e| format!("tokenizer: {e}"))?;
    let (cap, cap_n, cap_u) = {
        let enc = GemmaEncoder::load_dir(&root.join("text_encoder"))?;
        encode_prompt(&tok, &enc, prompt, p, want_uncond)
    };
    let latents = {
        let dit = NextDit::load_dir(&root.join("transformer"))?;
        denoise(&dit, &cap, cap_n, cap_u.as_ref(), lh, lw, p, &mut progress)
    };
    let vae = VaeDecoder::load_dir(&root.join("vae"))?;
    Ok(to_rgb01(vae.decode(&latents, lh, lw)))
}
