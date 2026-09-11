//! Full-network parity gate for the native Qwen Image VAE.
//!
//! The fixture directory is produced by `python/qwen_image_vae_ref.py`; the
//! first argument must be a real CMF produced from that fixture's safetensors
//! and `image.config_json`.  This deliberately exercises the mmap/dequant
//! loader, every encoder and decoder block, and both public APIs.
//!
//! ```text
//! cargo run -p cortiq-engine --example qwen_image_vae_parity -- \
//!   /path/to/qwen-vae.cmf /path/to/qwen-image-vae-fixture
//! ```

use cortiq_core::{CmfModel, TensorDtype, TensorSpec};
use cortiq_engine::qwen_image_vae::QwenImageVae;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Manifest {
    encode_input: String,
    encode_reference: String,
    decode_input: String,
    decode_reference: String,
    encode_input_shape: Vec<usize>,
    decode_input_shape: Vec<usize>,
    encode_output_shape: Vec<usize>,
    decode_output_shape: Vec<usize>,
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len() % 4 == 0,
        "{} is not f32-aligned",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn check(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name} length mismatch");
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (&g, &r) in got.iter().zip(want) {
        assert!(g.is_finite(), "{name} contains non-finite output");
        let d = g as f64 - r as f64;
        max_abs = max_abs.max(d.abs());
        sum_sq += d * d;
        ref_sq += (r as f64) * (r as f64);
    }
    let rel_rms = (sum_sq / ref_sq.max(1.0e-30)).sqrt();
    println!("{name}: max_abs={max_abs:.6e} rel_rms={rel_rms:.6e}");
    assert!(max_abs < 5.0e-3, "{name} max_abs={max_abs:.6e}");
    assert!(rel_rms < 2.0e-3, "{name} rel_rms={rel_rms:.6e}");
}

fn joined(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

fn pack_fixture(fixture: &Path, base: &Path, output: &Path, f16: bool) {
    let weights = joined(fixture, "vae.safetensors");
    let mut tensors = Vec::new();
    cortiq_engine::vae::read_safetensors_each(&weights, &mut |name, shape, values| {
        let mut data = Vec::with_capacity(values.len() * if f16 { 2 } else { 4 });
        for value in values {
            if f16 {
                data.extend_from_slice(&cortiq_core::quant::f32_to_f16(value).to_le_bytes());
            } else {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        tensors.push(TensorSpec {
            name: name.to_string(),
            dtype: if f16 {
                TensorDtype::F16
            } else {
                TensorDtype::F32
            },
            shape,
            data,
        });
        Ok(())
    })
    .unwrap_or_else(|e| panic!("read {}: {e}", weights.display()));
    tensors.push(TensorSpec {
        name: "image.config_json".into(),
        dtype: TensorDtype::U8,
        shape: vec![joined(fixture, "config.json").metadata().unwrap().len() as usize],
        data: std::fs::read(joined(fixture, "config.json")).expect("read config.json"),
    });
    let base_model = CmfModel::open(base).unwrap_or_else(|e| panic!("open base CMF: {e}"));
    CmfModel::write(output, &base_model.header, &tensors, None, None)
        .unwrap_or_else(|e| panic!("write {}: {e}", output.display()));
    println!(
        "wrote tiny Qwen Image VAE CMF {} ({} tensors)",
        output.display(),
        tensors.len()
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next().expect(
        "usage: qwen_image_vae_parity <vae.cmf> <fixture-dir> | pack <fixture-dir> <base.cmf> <out.cmf> [f16]",
    );
    if first == "pack" {
        let fixture = PathBuf::from(args.next().expect("pack needs <fixture-dir>"));
        let base = PathBuf::from(args.next().expect("pack needs <base.cmf>"));
        let output = PathBuf::from(args.next().expect("pack needs <out.cmf>"));
        let f16 = args.next().as_deref() == Some("f16");
        pack_fixture(&fixture, &base, &output, f16);
        return;
    }
    let cmf = PathBuf::from(first);
    let fixture = PathBuf::from(args.next().expect(
        "usage: qwen_image_vae_parity <vae.cmf> <fixture-dir> | pack <fixture-dir> <base.cmf> <out.cmf> [f16]",
    ));
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(fixture.join("manifest.json")).expect("read manifest.json"),
    )
    .expect("parse manifest.json");
    assert_eq!(manifest.encode_input_shape.len(), 3);
    assert_eq!(manifest.decode_input_shape.len(), 3);

    let vae = QwenImageVae::open(&cmf).expect("open qwen image VAE CMF");
    let input = read_f32(&joined(&fixture, &manifest.encode_input));
    let expected_encode = read_f32(&joined(&fixture, &manifest.encode_reference));
    let got_encode = vae
        .encode_mean(
            &input,
            manifest.encode_input_shape[1],
            manifest.encode_input_shape[2],
        )
        .expect("encode_mean");
    assert_eq!(
        got_encode.len(),
        manifest.encode_output_shape.iter().product::<usize>()
    );
    check("encode_mean", &got_encode, &expected_encode);

    let latent = read_f32(&joined(&fixture, &manifest.decode_input));
    let expected_decode = read_f32(&joined(&fixture, &manifest.decode_reference));
    let got_decode = vae
        .decode(
            &latent,
            manifest.decode_input_shape[1],
            manifest.decode_input_shape[2],
        )
        .expect("decode");
    assert_eq!(
        got_decode.len(),
        manifest.decode_output_shape.iter().product::<usize>()
    );
    check("decode", &got_decode, &expected_decode);
    println!(
        "qwen image VAE parity PASS: z_dim={} compression={} latent_stats={}",
        vae.z_dim,
        vae.spatial_compression_ratio,
        vae.latents_mean.len()
    );
}
