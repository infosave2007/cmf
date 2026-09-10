//! FLUX-VAE decoder parity against the numpy reference on REAL weights.
//!
//!     python3 python/vae_ref.py <vae_dir> <out_dir> 8
//!     CMF_VAE_DIR=<vae_dir> CMF_VAE_REF=<out_dir> \
//!         cargo test -p cortiq-engine --release --test vae_parity -- --nocapture
//!
//! Skips silently when the env vars are absent (CI has no weights).

use std::path::Path;

#[test]
fn vae_decoder_matches_numpy_reference() {
    let (Ok(dir), Ok(refs)) = (std::env::var("CMF_VAE_DIR"), std::env::var("CMF_VAE_REF")) else {
        eprintln!("CMF_VAE_DIR/CMF_VAE_REF not set — skipping VAE parity");
        return;
    };
    let dec = cortiq_engine::vae::VaeDecoder::load_dir(Path::new(&dir)).expect("load VAE");
    let read_f32 = |p: &str| -> Vec<f32> {
        std::fs::read(p)
            .unwrap_or_else(|e| panic!("{p}: {e}"))
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let z = read_f32(&format!("{refs}/vae_latent.bin"));
    let want = read_f32(&format!("{refs}/vae_ref.bin"));
    let hw = ((z.len() / dec.latent_channels) as f64).sqrt() as usize;
    assert_eq!(z.len(), dec.latent_channels * hw * hw);
    let t0 = std::time::Instant::now();
    let got = dec.decode(&z, hw, hw);
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(got.len(), want.len(), "output shape mismatch");
    let mut max_d = 0f32;
    for (a, b) in got.iter().zip(&want) {
        max_d = max_d.max((a - b).abs());
    }
    println!(
        "vae parity: latent {hw}x{hw} -> {}x{} decode {dt:.2}s, max|Δ| = {max_d:.2e}",
        hw * 8,
        hw * 8
    );
    assert!(
        max_d < 2e-3,
        "VAE decoder ≠ numpy reference: max|Δ| = {max_d}"
    );
}

/// Speed at real sizes (no reference — timing + range sanity).
///     CMF_VAE_DIR=... cargo test --release --test vae_parity vae_decode_speed -- --ignored --nocapture
#[test]
#[ignore]
fn vae_decode_speed() {
    let Ok(dir) = std::env::var("CMF_VAE_DIR") else {
        eprintln!("CMF_VAE_DIR not set");
        return;
    };
    let dec = cortiq_engine::vae::VaeDecoder::load_dir(Path::new(&dir)).expect("load VAE");
    for hw in [64usize, 128] {
        let z: Vec<f32> = (0..dec.latent_channels * hw * hw)
            .map(|i| ((i * 2654435761usize) as f32 / usize::MAX as f32 - 0.5) * 2.0)
            .collect();
        let t0 = std::time::Instant::now();
        let img = dec.decode(&z, hw, hw);
        let (mn, mx) = img
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        println!(
            "decode {hw}x{hw} -> {}px: {:.2}s (range [{mn:.2}, {mx:.2}])",
            hw * 8,
            t0.elapsed().as_secs_f64()
        );
    }
}
