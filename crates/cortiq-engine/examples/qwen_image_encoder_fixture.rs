//! Full tiny Qwen2.5-VL text+vision parity gate.
//!
//! Generate the fixture with `qwen_image_encoder_oracle.py` in the bundled
//! reference environment, then this example writes a real CMF with the
//! official tensor names/assets and compares native `Conditioning` against
//! the reference model's final hidden state after the 64-token drop:
//!
//! ```text
//! /private/tmp/cmf-qwen-image-ref-venv/bin/python \
//!   crates/cortiq-engine/examples/qwen_image_encoder_oracle.py /tmp/qwen-encoder.json
//! cargo run -p cortiq-engine --example qwen_image_encoder_fixture -- \
//!   /tmp/qwen-encoder.json /tmp/qwen-encoder.cmf
//! ```

use cortiq_core::{CmfHeader, CmfModel, TensorDtype, TensorSpec, CMF_VERSION};
use cortiq_engine::qwen_image_encoder::QwenImageEncoder;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Fixture {
    config: serde_json::Value,
    processor: serde_json::Value,
    tokenizer: serde_json::Value,
    prompt: String,
    image_width: u32,
    image_height: u32,
    image_rgb: Vec<u8>,
    ids: Vec<u32>,
    grid: Vec<usize>,
    shapes: HashMap<String, Vec<usize>>,
    weights: HashMap<String, Vec<f32>>,
    expected: Vec<f32>,
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn blob(name: &str, value: &serde_json::Value) -> Result<TensorSpec, Box<dyn Error>> {
    let data = serde_json::to_vec(value)?;
    Ok(TensorSpec {
        name: name.into(),
        dtype: TensorDtype::U8,
        shape: vec![data.len()],
        data,
    })
}

fn tiny_header() -> Result<CmfHeader, Box<dyn Error>> {
    Ok(serde_json::from_value(json!({
        "version": CMF_VERSION,
        "arch": {
            "arch_name": "qwen-image-encoder-tiny-fixture",
            "hidden_size": 12,
            "intermediate_size": 24,
            "num_layers": 1,
            "num_attention_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 6,
            "vocab_size": 305,
            "layer_types": ["FullAttention"],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 512
        },
        "quant_type": "F32"
    }))?)
}

fn pack(fixture: &Fixture, output: &Path) -> Result<(), Box<dyn Error>> {
    let mut tensors = vec![
        blob("image.config_json", &fixture.config)?,
        blob("image.processor_config_json", &fixture.processor)?,
        blob("image.tokenizer_json", &fixture.tokenizer)?,
    ];
    let mut names: Vec<&String> = fixture.weights.keys().collect();
    names.sort();
    for name in names {
        let values = fixture.weights.get(name).expect("weight key disappeared");
        let shape = fixture
            .shapes
            .get(name)
            .ok_or_else(|| format!("missing shape for weight '{name}'"))?;
        if shape.iter().product::<usize>() != values.len() {
            return Err(format!(
                "weight '{name}' shape {:?} has {} elements, fixture stores {}",
                shape,
                shape.iter().product::<usize>(),
                values.len()
            )
            .into());
        }
        tensors.push(TensorSpec {
            name: name.clone(),
            dtype: TensorDtype::F32,
            shape: shape.clone(),
            data: f32_bytes(values),
        });
    }
    CmfModel::write(output, &tiny_header()?, &tensors, None, None)?;
    Ok(())
}

fn check(got: &[f32], expected: &[f32]) -> Result<(), Box<dyn Error>> {
    if got.len() != expected.len() {
        return Err(format!(
            "native hidden length {} != oracle {}",
            got.len(),
            expected.len()
        )
        .into());
    }
    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (&a, &b) in got.iter().zip(expected) {
        if !a.is_finite() {
            return Err("native hidden contains non-finite values".into());
        }
        max_abs = max_abs.max((a - b).abs());
        let d = a as f64 - b as f64;
        sum_sq += d * d;
        ref_sq += (b as f64) * (b as f64);
    }
    let rel_rms = (sum_sq / ref_sq.max(1e-30)).sqrt();
    println!("encoder tiny parity: max_abs={max_abs:.6e} rel_rms={rel_rms:.6e}");
    if max_abs > 5e-4 || rel_rms > 2e-4 {
        return Err(
            format!("native/oracle mismatch max_abs={max_abs:.6e} rel_rms={rel_rms:.6e}").into(),
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let fixture_path = PathBuf::from(args.next().ok_or("missing fixture JSON")?);
    let cmf_path = PathBuf::from(args.next().ok_or("missing output CMF")?);
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&fixture_path)?)?;
    if fixture.grid != vec![1usize, 6, 10] || fixture.ids.len() <= 64 {
        return Err(format!(
            "unexpected fixture geometry grid={:?} ids={}",
            fixture.grid,
            fixture.ids.len()
        )
        .into());
    }
    if fixture.image_rgb.len() != fixture.image_width as usize * fixture.image_height as usize * 3 {
        return Err("fixture RGB byte count does not match image dimensions".into());
    }
    pack(&fixture, &cmf_path)?;
    let model = CmfModel::open(&cmf_path)?;
    if !model.verify().is_empty() {
        return Err(format!("fixture CMF failed verification: {:?}", model.verify()).into());
    }
    let encoder = QwenImageEncoder::open(&cmf_path)?;
    let image = image::RgbImage::from_raw(
        fixture.image_width,
        fixture.image_height,
        fixture.image_rgb.clone(),
    )
    .ok_or("fixture RGB image could not be constructed")?;
    let output = encoder.encode(&fixture.prompt, &[image])?;
    if output.hidden_size != 12 || output.seq_len * output.hidden_size != fixture.expected.len() {
        return Err(format!(
            "native conditioning shape {}×{} does not match oracle {} values",
            output.seq_len,
            output.hidden_size,
            fixture.expected.len()
        )
        .into());
    }
    check(&output.hidden, &fixture.expected)?;
    println!(
        "encoder tiny fixture PASS: tokens={} rows={} grid={:?}",
        fixture.ids.len(),
        output.seq_len,
        fixture.grid
    );
    Ok(())
}
