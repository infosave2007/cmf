//! MiniMax-H3 port vs the reference forward, on a toy checkpoint that
//! carries the release's real tensor names.
//!
//! `tools/mk_mmh3_toy.py` builds the fixture from ComfyUI's own
//! `MiniMaxH3Model`; `tools/mmh3_toy_gate.sh` packs it and points this
//! test at the directory. Without `CMF_MMH3_TOY` the test skips — the
//! fixture needs torch, which the repository does not.

use cortiq_engine::mmh3::{Layout, MiniMaxH3};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// (max |Δ|, rms Δ, rms of the reference)
fn diff(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
    assert_eq!(got.len(), want.len(), "length");
    let mut mx = 0f32;
    let mut se = 0f64;
    let mut rs = 0f64;
    for (&g, &w) in got.iter().zip(want) {
        let d = (g - w).abs();
        mx = mx.max(d);
        se += (d as f64) * (d as f64);
        rs += (w as f64) * (w as f64);
    }
    (
        mx,
        (se / got.len() as f64).sqrt() as f32,
        (rs / want.len() as f64).sqrt() as f32,
    )
}

#[test]
fn matches_the_reference_forward() {
    let Some(dir) = std::env::var_os("CMF_MMH3_TOY") else {
        eprintln!("CMF_MMH3_TOY unset — skipping (run tools/mmh3_toy_gate.sh)");
        return;
    };
    let dir = PathBuf::from(dir);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let u = |k: &str| meta[k].as_u64().unwrap() as usize;
    let (text_len, latent_t, lat_h, lat_w, audio_t) = (
        u("text_len"),
        u("latent_t"),
        u("lat_h"),
        u("lat_w"),
        u("audio_t"),
    );
    let sigma = meta["sigma"].as_f64().unwrap();

    let model = Arc::new(cortiq_core::CmfModel::open(dir.join("toy.cmf")).unwrap());
    let dit = MiniMaxH3::from_cmf(&model).unwrap();
    let layout = Layout::t2va(text_len, latent_t, lat_h, lat_w, audio_t);

    // ── the refiner, on its own ──
    let text_in = read_f32(&dir.join("text_in.bin"));
    let refined = dit.refine_text(&text_in, text_len);
    let (mx, rms, sig) = diff(&refined, &read_f32(&dir.join("text_refined.bin")));
    println!("refined text: max {mx:.3e} rms {rms:.3e} over signal rms {sig:.3e}");
    assert!(mx < 2e-4, "token refiner diverges: max {mx:.3e}");

    // ── the whole forward ──
    let video_in = read_f32(&dir.join("video_in.bin"));
    let audio_in = read_f32(&dir.join("audio_in.bin"));
    let (v, a) = dit.forward(&layout, &refined, &video_in, &audio_in, sigma, &[]);

    let (mx, rms, sig) = diff(&v, &read_f32(&dir.join("video_out.bin")));
    println!("video velocity: max {mx:.3e} rms {rms:.3e} over signal rms {sig:.3e}");
    assert!(mx < 5e-4, "video head diverges: max {mx:.3e}");

    let (mx, rms, sig) = diff(&a, &read_f32(&dir.join("audio_out.bin")));
    println!("audio velocity: max {mx:.3e} rms {rms:.3e} over signal rms {sig:.3e}");
    assert!(mx < 5e-4, "audio head diverges: max {mx:.3e}");
}

/// `fl2va`: the same clip conditioned on a first and a last keyframe.
///
/// The condition rows go in between the text and the audio, share the
/// target's spatial grid, sit at their own timestep near 1, and must not
/// move where the audio and video streams begin. The golden turns the
/// reference's 0.1% noise blend off (`visual_cond_noise_aug = 1.0`), so
/// this compares everything except a torch-seeded random stream.
#[test]
fn keyframes_match_the_reference() {
    let Some(dir) = std::env::var_os("CMF_MMH3_TOY") else {
        eprintln!("CMF_MMH3_TOY unset — skipping");
        return;
    };
    let dir = PathBuf::from(dir);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    if meta.get("n_keyframes").is_none() {
        eprintln!("fixture predates the keyframe golden — skipping");
        return;
    }
    let u = |k: &str| meta[k].as_u64().unwrap() as usize;
    let (text_len, latent_t, lat_h, lat_w, audio_t) = (
        u("text_len"), u("latent_t"), u("lat_h"), u("lat_w"), u("audio_t"),
    );
    let fc = u("frame_count");
    let sigma = meta["sigma"].as_f64().unwrap();

    let model = Arc::new(cortiq_core::CmfModel::open(dir.join("toy.cmf")).unwrap());
    let mut dit = MiniMaxH3::from_cmf(&model).unwrap();
    // The golden turned the blend off, which also puts the condition
    // rows at timestep 1.0 rather than 0.999.
    dit.cond_aug = 1.0;
    let layout = Layout::fl2va(
        text_len, latent_t, lat_h, lat_w, audio_t,
        &[(0, fc), (fc - 1, fc)],
        &[],
    );
    let refined = dit.refine_text(&read_f32(&dir.join("text_in.bin")), text_len);
    let cond: Vec<Vec<f32>> = (0..u("n_keyframes"))
        .map(|i| read_f32(&dir.join(format!("kf{i}.bin"))))
        .collect();
    let (v, a) = dit.forward(
        &layout,
        &refined,
        &read_f32(&dir.join("video_in.bin")),
        &read_f32(&dir.join("audio_in.bin")),
        sigma,
        &cond,
    );
    let (mx, _, _) = diff(&v, &read_f32(&dir.join("kf_video_out.bin")));
    println!("fl2va video velocity: max {mx:.3e}");
    assert!(mx < 5e-4, "keyframe video head diverges: max {mx:.3e}");
    let (mx, _, _) = diff(&a, &read_f32(&dir.join("kf_audio_out.bin")));
    println!("fl2va audio velocity: max {mx:.3e}");
    assert!(mx < 5e-4, "keyframe audio head diverges: max {mx:.3e}");
}
