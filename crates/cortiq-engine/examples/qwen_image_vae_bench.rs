//! Bounded native Qwen Image VAE timing probe.
//!
//! This opens only the VAE component and exercises one deterministic frame.
//! It never invokes Python/Diffusers and is intended for CPU or serialized
//! GPU A/B measurements:
//!
//! ```text
//! qwen_image_vae_bench <vae.cmf> <height> <width> [encode|decode|both]
//! ```

use cortiq_engine::qwen_image_vae::QwenImageVae;
use std::path::PathBuf;
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage: qwen_image_vae_bench <vae.cmf> <height> <width> [encode|decode|both]");
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let height: usize = args
        .next()
        .unwrap_or_else(|| usage())
        .parse()
        .unwrap_or_else(|_| usage());
    let width: usize = args
        .next()
        .unwrap_or_else(|| usage())
        .parse()
        .unwrap_or_else(|_| usage());
    let mode_arg = args.next();
    let mode = mode_arg.as_deref().unwrap_or("both");
    if args.next().is_some() || !matches!(mode, "encode" | "decode" | "both") {
        usage();
    }
    if height == 0 || width == 0 || height % 8 != 0 || width % 8 != 0 {
        eprintln!("height and width must be positive multiples of 8");
        std::process::exit(2);
    }

    let vae = QwenImageVae::open(&path).unwrap_or_else(|e| panic!("open VAE: {e}"));
    let frame: Vec<f32> = (0..3 * height * width)
        .map(|i| ((i as f32 * 0.017).sin() * 0.8).clamp(-1.0, 1.0))
        .collect();
    let latent_len =
        vae.z_dim * (height / vae.spatial_compression()) * (width / vae.spatial_compression());
    let latent: Vec<f32> = (0..latent_len)
        .map(|i| (i as f32 * 0.013).cos() * 0.5)
        .collect();
    println!(
        "qwen_image_vae_bench shape={}x{} latent={} mode={} threads={}",
        height,
        width,
        latent_len,
        mode,
        std::env::var("CMF_THREADS").unwrap_or_else(|_| "auto".into())
    );
    if mode == "encode" || mode == "both" {
        let started = Instant::now();
        let got = vae
            .encode_mean(&frame, height, width)
            .unwrap_or_else(|e| panic!("encode_mean: {e}"));
        println!(
            "encode_mean {:.3}s values={} first={:.7e}",
            started.elapsed().as_secs_f64(),
            got.len(),
            got[0]
        );
    }
    if mode == "decode" || mode == "both" {
        let started = Instant::now();
        let got = vae
            .decode(
                &latent,
                height / vae.spatial_compression(),
                width / vae.spatial_compression(),
            )
            .unwrap_or_else(|e| panic!("decode: {e}"));
        println!(
            "decode {:.3}s values={} first={:.7e}",
            started.elapsed().as_secs_f64(),
            got.len(),
            got[0]
        );
    }
}
