//! MiniMax-H3's two VAE decoders against the reference, on toys that
//! keep the release's real schedules (16× spatial, 4× temporal, 800
//! samples a latent audio frame) and shrink only the widths.
//!
//! `tools/mk_vae_toy.py` builds the fixture; `CMF_MMH3_VAE_TOY` points
//! at the packed directory. Without it both tests skip.

use cortiq_engine::audiovae::AudioVae;
use cortiq_engine::vae3d::VideoVae;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn report(name: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut mx = 0f32;
    let mut se = 0f64;
    let mut rs = 0f64;
    for (&g, &w) in got.iter().zip(want) {
        mx = mx.max((g - w).abs());
        se += ((g - w) as f64).powi(2);
        rs += (w as f64).powi(2);
    }
    println!(
        "{name}: max {mx:.3e} rms {:.3e} over signal rms {:.3e}",
        (se / got.len() as f64).sqrt(),
        (rs / want.len() as f64).sqrt()
    );
    mx
}

fn toy() -> Option<(PathBuf, serde_json::Value)> {
    let dir = PathBuf::from(std::env::var_os("CMF_MMH3_VAE_TOY")?);
    let meta = serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    Some((dir, meta))
}

#[test]
fn video_decoder_matches_the_reference() {
    let Some((dir, meta)) = toy() else {
        eprintln!("CMF_MMH3_VAE_TOY unset — skipping");
        return;
    };
    let v = &meta["video"];
    let u = |k: &str| v[k].as_u64().unwrap() as usize;
    let model = Arc::new(cortiq_core::CmfModel::open(dir.join("vvae.cmf")).unwrap());
    let vae = VideoVae::from_cmf(&model).unwrap();
    let (rgb, frames) = vae.decode(
        &read_f32(&dir.join("video_z.bin")),
        u("latent_t"),
        u("lat_h"),
        u("lat_w"),
    );
    assert_eq!(frames, u("frames"), "frame count");
    // The tiling schedule is the risk, and a wrong tile boundary shows
    // up as a seam — a handful of columns far off, not a drift.
    let mx = report("video rgb", &rgb, &read_f32(&dir.join("video_rgb.bin")));
    assert!(mx < 2e-3, "video decoder diverges: max {mx:.3e}");
}

#[test]
fn audio_decoder_matches_the_reference() {
    let Some((dir, meta)) = toy() else {
        eprintln!("CMF_MMH3_VAE_TOY unset — skipping");
        return;
    };
    let a = &meta["audio"];
    let u = |k: &str| a[k].as_u64().unwrap() as usize;
    let model = Arc::new(cortiq_core::CmfModel::open(dir.join("avae.cmf")).unwrap());
    let vae = AudioVae::from_cmf(&model).unwrap();
    let (wav, n) = vae.decode(
        &read_f32(&dir.join("audio_z.bin")),
        u("vae_latent_channels"),
        u("audio_t"),
    );
    assert_eq!(n, u("samples"), "sample count");
    let mx = report("audio wav", &wav, &read_f32(&dir.join("audio_wav.bin")));
    assert!(mx < 2e-3, "audio decoder diverges: max {mx:.3e}");
}

#[test]
fn keyframe_encoder_matches_the_reference() {
    let Some((dir, meta)) = toy() else {
        eprintln!("CMF_MMH3_VAE_TOY unset — skipping");
        return;
    };
    let v = &meta["video"];
    let u = |k: &str| v[k].as_u64().unwrap() as usize;
    let model = Arc::new(cortiq_core::CmfModel::open(dir.join("vvae.cmf")).unwrap());
    let enc = cortiq_engine::vae3d::VideoVaeEncoder::from_cmf(&model).unwrap();
    let (z, zh, zw) = enc.encode_frame(
        &read_f32(&dir.join("frame_in.bin")),
        u("frame_h"),
        u("frame_w"),
    );
    assert_eq!((zh, zw), (u("enc_zh"), u("enc_zw")), "latent shape");
    let mx = report("keyframe z", &z, &read_f32(&dir.join("frame_z.bin")));
    assert!(mx < 2e-3, "encoder diverges: max {mx:.3e}");
}
