//! `cortiq vae`: run a standalone autoencoder file (Primal VAE) — encode an
//! image to its latent, decode a latent back to an image, or round-trip an
//! image and report the reconstruction PSNR.

use anyhow::{Context, anyhow};
use cortiq_engine::primal_vae::{self, PrimalVae};
use std::path::Path;
use std::sync::Arc;

/// Write a C-order float32 `.npy`.
fn write_npy(path: &Path, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let tuple = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {tuple}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).with_context(|| path.display().to_string())
}

fn load_image(path: &str, multiple: usize) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let frame = cortiq_engine::media::read_rgb(Path::new(path)).map_err(anyhow::Error::msg)?;
    let (h, w) = (frame.height, frame.width);
    anyhow::ensure!(
        h % multiple == 0 && w % multiple == 0,
        "{path}: {w}x{h} — Primal VAE image sides must be multiples of {multiple}"
    );
    Ok((primal_vae::rgb8_to_chw(&frame.data, h, w), h, w))
}

fn save_image(path: &str, x: &[f32], h: usize, w: usize) -> anyhow::Result<()> {
    let frame = cortiq_engine::media::RgbFrame { width: w, height: h, data: primal_vae::chw_to_rgb8(x, h, w) };
    cortiq_engine::media::save_png(Path::new(path), &frame).map_err(anyhow::Error::msg)
}

pub fn run(model: &str, mode: &str, input: &str, output: &str) -> anyhow::Result<()> {
    let m = Arc::new(cortiq_core::CmfModel::open(model).map_err(|e| anyhow!("{model}: {e}"))?);
    anyhow::ensure!(primal_vae::is_primal_vae(&m), "{model}: not a Primal VAE file");
    let t_load = std::time::Instant::now();
    let vae = PrimalVae::from_model(&m, cortiq_engine::pool::Pool::from_env()).map_err(anyhow::Error::msg)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let c = vae.compression;
    let lc = vae.latent_channels;
    let device = if cortiq_engine::gpu::enabled_here() { cortiq_engine::gpu::backend_name() } else { "cpu".into() };
    match mode {
        "encode" => {
            let (img, h, w) = load_image(input, vae.image_multiple())?;
            let t0 = std::time::Instant::now();
            let z = vae.encode(&img, h, w).map_err(anyhow::Error::msg)?;
            let dt = t0.elapsed().as_secs_f64();
            write_npy(Path::new(output), &[1, lc, h / c, w / c], &z)?;
            let mean = z.iter().map(|&v| v as f64).sum::<f64>() / z.len() as f64;
            let std = (z.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / z.len() as f64).sqrt();
            println!("encode {w}x{h} -> [1, {lc}, {}, {}] in {dt:.3} s ({device}); latent mean {mean:.3} std {std:.3}; wrote {output}", h / c, w / c);
        }
        "decode" => {
            let npy = crate::npy::read(Path::new(input))?;
            let crate::npy::NpyData::F32(z) = npy.data else {
                return Err(anyhow!("{input}: the latent must be float"));
            };
            let s = &npy.shape;
            let (zc, zh, zw) = match s.as_slice() {
                [1, a, b, d] | [a, b, d] => (*a, *b, *d),
                _ => return Err(anyhow!("{input}: expected a [1, C, h, w] or [C, h, w] latent, got {s:?}")),
            };
            anyhow::ensure!(zc == lc, "{input}: {zc} latent channels, the model takes {lc}");
            let (h, w) = (zh * c, zw * c);
            let t0 = std::time::Instant::now();
            let x = vae.decode(&z, h, w).map_err(anyhow::Error::msg)?;
            let dt = t0.elapsed().as_secs_f64();
            save_image(output, &x, h, w)?;
            println!("decode [1, {lc}, {zh}, {zw}] -> {w}x{h} in {dt:.3} s ({device}); wrote {output}");
        }
        "roundtrip" => {
            let (img, h, w) = load_image(input, vae.image_multiple())?;
            let t0 = std::time::Instant::now();
            let z = vae.encode(&img, h, w).map_err(anyhow::Error::msg)?;
            let te = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let x = vae.decode(&z, h, w).map_err(anyhow::Error::msg)?;
            let td = t1.elapsed().as_secs_f64();
            save_image(output, &x, h, w)?;
            println!(
                "roundtrip {w}x{h}: encode {te:.3} s, decode {td:.3} s ({device}, weights {load_s:.2} s); PSNR {:.2} dB; wrote {output}",
                primal_vae::psnr(&img, &x)
            );
        }
        other => return Err(anyhow!("--mode {other}: expected encode, decode or roundtrip")),
    }
    Ok(())
}
