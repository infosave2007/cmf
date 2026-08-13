//! `cortiq music`: caption + lyrics → 44.1 kHz stereo, from one .cmf.
//!
//! Three stacks in sequence. The AR stack GENERATES the conditioning
//! rather than encoding it — audio tokens sampled frame by frame, whose
//! eight codebook hidden states become what the DiT is conditioned on —
//! then the flow sampler denoises a latent and the DAV vocoder turns it
//! into sound at 512 samples a frame.

use anyhow::{Context, anyhow};
use cortiq_core::CmfModel;
use cortiq_engine::audiovae::Music3Dav;
use cortiq_engine::music3::{Music3Ar, Music3Dit, tokens};
use cortiq_engine::tokenizer::Tokenizer;
use std::sync::Arc;

/// `comfy/ldm/minimax_music/prompt.py`, minus the markdown scrubbing —
/// that only ever removes characters a caption should not carry.
fn normalize_lyrics(lyrics: &str) -> String {
    let mut out = String::from("[start]\n");
    for line in lyrics.replace(" ^ ", "\n").lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        // Section tags are lowercased; everything else is left alone.
        if l.starts_with('[') && l.ends_with(']') {
            out.push_str(&l.to_lowercase());
        } else {
            out.push_str(l);
        }
        out.push('\n');
    }
    out
}

/// The prompt is special tokens around plain text, so it is tokenized in
/// pieces: the ids are fixed and must not be re-derived from the vocab.
fn build_ids(tok: &Tokenizer, caption: &str, lyrics: &str) -> Vec<u32> {
    let mut ids = vec![tokens::IM_START, tokens::CAPTION_START];
    ids.extend(tok.encode(caption.trim()));
    ids.push(tokens::CAPTION_END);
    ids.push(tokens::LYRICS_START);
    ids.extend(tok.encode(&normalize_lyrics(lyrics)));
    ids.push(tokens::LYRICS_END);
    ids.push(tokens::IM_END);
    ids.push(tokens::AUDIO_START);
    ids
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_music(
    model_path: &str,
    caption: &str,
    lyrics: &str,
    seconds: f32,
    steps: usize,
    seed: u64,
    out: &str,
) -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let model = Arc::new(CmfModel::open(model_path).map_err(|e| anyhow!("{model_path}: {e}"))?);
    let vocab = model
        .vocab
        .as_deref()
        .ok_or_else(|| anyhow!("{model_path} carries no tokenizer"))?;
    let tok = Tokenizer::from_bytes(vocab).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let ar = Music3Ar::from_cmf(&model).map_err(|e| anyhow!("AR stack: {e}"))?;
    let ids = build_ids(&tok, caption, lyrics);
    let frames = ((seconds * ar.fps as f32).round() as usize)
        .clamp(1, ar.max_frames)
        .min(ar.max_frames);
    eprintln!(
        "prompt {} tokens, asking for {frames} audio frames ({:.1} s at {} fps)",
        ids.len(),
        frames as f32 / ar.fps as f32,
        ar.fps
    );

    let (hidden, got) = ar
        .generate(&ids, seed, frames, |i, n| {
            if i > 0 && (i % 10 == 0 || i == n) {
                eprint!("\r  ar {i}/{n} ({:.1}s)   ", t0.elapsed().as_secs_f32());
            }
        })
        .map_err(|e| anyhow!("{e}"))?;
    eprintln!("\n  ar done: {got} frames ({:.1}s)", t0.elapsed().as_secs_f32());

    let dit = Music3Dit::from_cmf(&model).map_err(|e| anyhow!("DiT: {e}"))?;
    let (cond, n) = dit.aligned_condition(&hidden, got);
    eprintln!("  condition {} x {n} latent frames", Music3Dit::COND_CH);

    // Deterministic noise: the seed names the song, and a song that
    // changes between runs of the same seed is not reproducible.
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let noise: Vec<f32> = (0..Music3Dit::IN_CH * n)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Box-Muller from two uniforms would be exact; a sum of
            // three is within a per cent and needs no transcendentals.
            let u = |x: u64| ((x >> 40) as f32) / (1u32 << 24) as f32 - 0.5;
            (u(rng) + u(rng.rotate_left(17)) + u(rng.rotate_left(34))) * 2.0
        })
        .collect();
    let latent = dit.sample(&noise, &cond, n, steps, |i, t| {
        eprint!("\r  denoise {i}/{t} ({:.1}s)   ", t0.elapsed().as_secs_f32());
    });
    eprintln!();

    let dav = Music3Dav::from_cmf(&model).map_err(|e| anyhow!("vocoder: {e}"))?;
    // The vocoder is 30 convolutions over 512x the latent length; giving
    // it the pool is the difference between seconds and minutes.
    let pool = cortiq_engine::pool::Pool::from_env();
    let pcm = dav.decode(&latent, n, pool.as_deref());
    let samples = pcm.len() / 2;
    // `wav_bytes` wants planar; the vocoder hands back interleaved.
    let mut planar = vec![0f32; pcm.len()];
    for (i, c) in pcm.chunks_exact(2).enumerate() {
        planar[i] = c[0];
        planar[samples + i] = c[1];
    }
    let wav = crate::avout::wav_bytes(&planar, samples, Music3Dav::SAMPLE_RATE);
    std::fs::write(out, &wav).with_context(|| out.to_string())?;
    let secs = samples as f32 / Music3Dav::SAMPLE_RATE as f32;
    println!(
        "{out}: {secs:.2}s stereo at {} Hz, {:.1} MB in {:.1}s",
        Music3Dav::SAMPLE_RATE,
        wav.len() as f32 / 1e6,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
