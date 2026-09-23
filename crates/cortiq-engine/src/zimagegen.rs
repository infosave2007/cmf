//! Z-Image-Turbo text-to-image pipeline: prompt → Qwen3 chat template →
//! Qwen3-4B encoder (35 layers, `hidden_states[-2]`) → DiT, 8 Euler steps
//! (shift 3, guidance 0) → Flux VAE → RGB u8.
//! OWNER: WP1 (plan §5). WP0 state: public API skeleton, bodies `todo!()`.
//!
//! Runs every stage under `gpu::image_stage_scope()`. Profiling:
//! `CMF_ZIMAGE_PROF=1` prints in-process stage times (tokenize ·
//! text-encode · caption · prepare · step[0..7] · vae · total) and the
//! median of steps 1..7. `CMF_INIT_LATENT=<raw f32 [1,16,H/8,W/8]>`
//! injects the oracle noise.
#![allow(dead_code, unused_variables)]

use crate::tokenizer::Tokenizer;
use std::path::Path;

/// `header.arch.arch_name` of a Z-Image container.
pub const ARCH_NAME: &str = "z_image";
/// Prompt token cap (diffusers `max_sequence_length`), applied to the
/// TEMPLATED string.
pub const MAX_TOKENS: usize = 512;

/// Generation parameters. Defaults: 1024², 8 steps, shift 3, seed 42.
#[derive(Clone, Debug)]
pub struct ZParams {
    pub height: usize,
    pub width: usize,
    pub steps: usize,
    pub seed: u64,
    pub shift: f32,
    pub max_tokens: usize,
}

impl Default for ZParams {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            steps: crate::zimage::DEFAULT_STEPS,
            seed: 42,
            shift: crate::zimage::DEFAULT_SHIFT,
            max_tokens: MAX_TOKENS,
        }
    }
}

/// `"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"` —
/// the Qwen3 template with `enable_thinking=True` (no think block), no
/// system prompt.
pub fn chat_template(prompt: &str) -> String {
    todo!("WP1")
}

/// Template → ids (Qwen2 BPE, NFC, no BOS), truncated to `max_tokens`.
pub fn prompt_ids(tok: &Tokenizer, prompt: &str, max_tokens: usize) -> Vec<u32> {
    todo!("WP1")
}

/// Generate one image from a Z-Image `.cmf`. Returns RGB u8 in
/// [height, width, 3] row-major order (PNG order): (x/2+0.5).clamp(0,1),
/// ·255, round. `progress(step, steps)` is called after every DiT step.
pub fn generate(
    model_path: &Path,
    prompt: &str,
    p: &ZParams,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, String> {
    todo!("WP1")
}
