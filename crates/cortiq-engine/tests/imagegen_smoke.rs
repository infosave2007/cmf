//! End-to-end Lumina-Image 2.0 smoke on REAL weights.
//!
//!     CMF_LUMINA_DIR=<root with tokenizer/ text_encoder/ transformer/ vae/> \
//!         cargo test -p cortiq-engine --release --test imagegen_smoke -- --nocapture
//!
//! Skips silently when the env var is absent (CI has no weights).
use std::path::Path;

/// HF `tokenizers` ids for the full Lumina template around
/// "a cat sitting on a windowsill at sunset" (BOS included).
const GOLDEN_IDS: [u32; 40] = [
    2, 2045, 708, 671, 20409, 6869, 577, 11941, 11257, 5191, 675, 573, 11257, 7151, 576, 2416,
    235290, 1082, 30236, 3482, 611, 95984, 73815, 689, 2425, 73815, 235265, 968, 55440, 7248,
    235313, 476, 4401, 10191, 611, 476, 5912, 81036, 696, 22097,
];

#[test]
fn lumina_tokenizer_matches_hf() {
    let Ok(dir) = std::env::var("CMF_LUMINA_DIR") else {
        eprintln!("CMF_LUMINA_DIR not set — skipping");
        return;
    };
    let tok = cortiq_engine::tokenizer::Tokenizer::from_file(
        Path::new(&dir).join("tokenizer").join("tokenizer.json"),
    )
    .expect("tokenizer");
    let full = format!(
        "{} <Prompt Start> {}",
        cortiq_engine::imagegen::DEFAULT_SYSTEM_PROMPT,
        "a cat sitting on a windowsill at sunset"
    );
    let ids = tok.with_bos(tok.encode(&full));
    assert_eq!(ids, GOLDEN_IDS, "Gemma tokenization diverged from HF");
}

#[test]
fn lumina_end_to_end_generates_finite_image() {
    let Ok(dir) = std::env::var("CMF_LUMINA_DIR") else {
        eprintln!("CMF_LUMINA_DIR not set — skipping");
        return;
    };
    let params = cortiq_engine::imagegen::GenParams {
        height: 128,
        width: 128,
        steps: 2,
        guidance_scale: 0.0, // no CFG — halves the smoke cost
        seed: 7,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let img = cortiq_engine::imagegen::generate(
        Path::new(&dir),
        "a red square on a white background",
        &params,
        |i, n| eprintln!("step {i}/{n} ({:.1}s)", t0.elapsed().as_secs_f64()),
    )
    .expect("generate");
    assert_eq!(img.len(), 3 * 128 * 128);
    assert!(img.iter().all(|v| v.is_finite()));
    let mean = img.iter().sum::<f32>() / img.len() as f32;
    // 2 steps of denoising won't make art, but the pipeline must land
    // strictly inside (0,1) on average — all-0/all-1 means a broken
    // stage, NaNs a broken kernel.
    assert!(mean > 0.02 && mean < 0.98, "degenerate image, mean {mean}");
    println!(
        "e2e smoke: 128x128 in {:.1}s, mean {mean:.3}",
        t0.elapsed().as_secs_f64()
    );
}

#[test]
fn lumina_packaged_cmf_generates() {
    // CMF_LUMINA_CMF = a `cortiq imagine-pack` output file.
    let Ok(cmf) = std::env::var("CMF_LUMINA_CMF") else {
        eprintln!("CMF_LUMINA_CMF not set — skipping");
        return;
    };
    let params = cortiq_engine::imagegen::GenParams {
        height: 128,
        width: 128,
        steps: 2,
        guidance_scale: 0.0,
        seed: 7,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let img = cortiq_engine::imagegen::generate(
        Path::new(&cmf),
        "a red square on a white background",
        &params,
        |i, n| eprintln!("step {i}/{n} ({:.1}s)", t0.elapsed().as_secs_f64()),
    )
    .expect("generate from .cmf");
    assert_eq!(img.len(), 3 * 128 * 128);
    assert!(img.iter().all(|v| v.is_finite()));
    let mean = img.iter().sum::<f32>() / img.len() as f32;
    assert!(mean > 0.02 && mean < 0.98, "degenerate image, mean {mean}");
    println!(
        "packaged smoke: 128x128 in {:.1}s, mean {mean:.3}",
        t0.elapsed().as_secs_f64()
    );
}
