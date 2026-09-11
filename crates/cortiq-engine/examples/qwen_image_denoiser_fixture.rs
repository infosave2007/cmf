//! Run the full native Qwen denoiser against the seeded official oracle.
//!
//! The companion `qwen_image_denoiser_oracle.py` writes a tiny state dict and
//! expected output.  This example repacks those tensors into an actual CMF,
//! opens it through the public transformer API, and exercises F16 plus Q8_2f
//! and Q4TP mapped projections while checking the full double-stream forward.
//!
//! ```text
//! /private/tmp/cmf-qwen-image-ref-venv/bin/python \
//!   crates/cortiq-engine/examples/qwen_image_denoiser_oracle.py /tmp/qwen.json
//! cargo run -p cortiq-engine --example qwen_image_denoiser_fixture -- \
//!   /tmp/qwen.json /tmp/qwen.cmf
//! ```

use cortiq_core::{CMF_VERSION, CmfHeader, CmfModel, TensorDtype, TensorSpec};
use cortiq_engine::qwen_image::QwenImageTransformer;
use cortiq_engine::qwen_image_ops::Linear;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Deserialize)]
struct Fixture {
    config: serde_json::Value,
    shapes: Vec<[usize; 3]>,
    text_len: usize,
    timestep: f32,
    image: Vec<f32>,
    text: Vec<f32>,
    weights: HashMap<String, Vec<f32>>,
    expected: Vec<f32>,
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| cortiq_core::quant::f32_to_f16(v).to_le_bytes())
        .collect()
}

fn q8_2f_bytes(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    let mut body = Vec::with_capacity(values.len());
    let mut row_scales = Vec::with_capacity(rows);
    for row in values.chunks_exact(cols) {
        let absmax = row.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let scale = if absmax == 0.0 {
            1.0 / 127.0
        } else {
            absmax / 127.0
        };
        row_scales.push(scale);
        body.extend(
            row.iter()
                .map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8 as u8),
        );
    }
    let mut out = body;
    out.extend(
        row_scales
            .iter()
            .flat_map(|&v| cortiq_core::quant::f32_to_f16(v).to_le_bytes()),
    );
    // A unit column field keeps the expected projection easy to state while
    // still traversing the native two-field quantized codec.
    out.extend((0..cols).flat_map(|_| cortiq_core::quant::f32_to_f16(1.0).to_le_bytes()));
    out
}

/// Minimal one-row Q4TP payload with one scale-ladder tile. The nibbles are
/// deliberately nonuniform, so this checks the native Q4TP reader/GEMM path
/// without pulling in the CLI converter crate.
fn q4tp_bytes() -> Vec<u8> {
    let mut out = vec![0u8; 16 + 4 + 1];
    out[16..18].copy_from_slice(&cortiq_core::quant::f32_to_f16(-3.0f32).to_le_bytes());
    out[18..20].copy_from_slice(&cortiq_core::quant::f32_to_f16(0.0).to_le_bytes());
    for k in 0..16 {
        let q0 = k as i8 - 8;
        let q1 = 7 - k as i8;
        out[k] = ((q0 + 8) as u8 & 0x0f) | (((q1 + 8) as u8 & 0x0f) << 4);
    }
    out
}

fn q4tp_expected() -> f32 {
    // Each of the sixteen nibble pairs sums to -1; rung 0 is 2^-3.
    -16.0 * cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(-3.0)).exp2()
}

fn tiny_header() -> Result<CmfHeader, Box<dyn Error>> {
    Ok(serde_json::from_value(json!({
        "version": CMF_VERSION,
        "arch": {
            "arch_name": "qwen-image-tiny-fixture",
            "hidden_size": 12,
            "intermediate_size": 48,
            "num_layers": 2,
            "num_attention_heads": 2,
            "num_kv_heads": 2,
            "head_dim": 6,
            "vocab_size": 0,
            "layer_types": [],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 64
        },
        "quant_type": "F32"
    }))?)
}

fn write_cmf(fixture: &Fixture, path: &Path) -> Result<(), Box<dyn Error>> {
    let mut tensors = Vec::with_capacity(fixture.weights.len() + 4);
    let config = serde_json::to_vec(&fixture.config)?;
    tensors.push(TensorSpec {
        name: "image.config_json".into(),
        dtype: TensorDtype::U8,
        shape: vec![config.len()],
        data: config,
    });
    for (name, values) in &fixture.weights {
        tensors.push(TensorSpec {
            name: name.clone(),
            dtype: TensorDtype::F32,
            shape: match name.as_str() {
                n if n.ends_with("linear_1.weight") => vec![12, 256],
                n if n.ends_with("linear_2.weight") => vec![12, 12],
                n if n == "txt_norm.weight" => vec![4],
                n if n == "img_in.weight" => vec![12, 2],
                n if n == "txt_in.weight" => vec![12, 4],
                n if n.ends_with("img_mod.1.weight") || n.ends_with("txt_mod.1.weight") => {
                    vec![72, 12]
                }
                n if n.ends_with("attn.norm_q.weight")
                    || n.ends_with("attn.norm_k.weight")
                    || n.ends_with("attn.norm_added_q.weight")
                    || n.ends_with("attn.norm_added_k.weight") =>
                {
                    vec![6]
                }
                n if n.ends_with("mlp.net.0.proj.weight") => vec![48, 12],
                n if n.ends_with("mlp.net.2.weight") => vec![12, 48],
                n if n == "norm_out.linear.weight" => vec![24, 12],
                n if n == "proj_out.weight" => vec![2, 12],
                n if n.ends_with(".weight") => vec![12, 12],
                n if n.ends_with("linear_1.bias") || n.ends_with("linear_2.bias") => vec![12],
                n if n == "proj_out.bias" => vec![2],
                n if n == "norm_out.linear.bias" => vec![24],
                n if n.ends_with("img_mod.1.bias") || n.ends_with("txt_mod.1.bias") => vec![72],
                n if n.ends_with("mlp.net.0.proj.bias") => vec![48],
                n if n.ends_with(".bias") => vec![12],
                _ => return Err(format!("cannot infer shape for {name}").into()),
            },
            data: f32_bytes(values),
        });
    }
    let q8_values: Vec<f32> = (0..64)
        .map(|i| ((i as f32 * 0.17).sin() * 0.25) + 0.01)
        .collect();
    tensors.push(TensorSpec {
        name: "fixture.f16.weight".into(),
        dtype: TensorDtype::F16,
        shape: vec![2, 2],
        data: f16_bytes(&[0.25, -0.5, 0.75, -1.0]),
    });
    tensors.push(TensorSpec {
        name: "fixture.q8_2f.weight".into(),
        dtype: TensorDtype::Q8_2f,
        shape: vec![2, 32],
        data: q8_2f_bytes(&q8_values, 2, 32),
    });
    tensors.push(TensorSpec {
        name: "fixture.q4tp.weight".into(),
        dtype: TensorDtype::Q4TiledP,
        shape: vec![1, 32],
        data: q4tp_bytes(),
    });
    CmfModel::write(path, &tiny_header()?, &tensors, None, None)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let fixture_path = PathBuf::from(args.next().ok_or("missing oracle JSON path")?);
    let cmf_path = PathBuf::from(args.next().ok_or("missing output CMF path")?);
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&fixture_path)?)?;
    write_cmf(&fixture, &cmf_path)?;

    let model = Arc::new(CmfModel::open(&cmf_path)?);
    if !model.verify().is_empty() {
        return Err(format!("fixture CMF failed verification: {:?}", model.verify()).into());
    }
    let transformer = QwenImageTransformer::from_cmf(&model)?;
    let output = transformer.forward(
        &fixture.image,
        &fixture.text,
        &fixture.shapes,
        fixture.text_len,
        fixture.timestep,
    )?;
    if output.len() != fixture.expected.len() {
        return Err(format!(
            "native output length {} != oracle {}",
            output.len(),
            fixture.expected.len()
        )
        .into());
    }
    let max_abs = output
        .iter()
        .zip(&fixture.expected)
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 5e-5 {
        return Err(format!("native/oracle max_abs={max_abs:.8e} > 5e-5").into());
    }

    let f16 = Linear::load(&model, "fixture.f16.weight")?;
    let mut f16_out = [0.0f32; 2];
    f16.forward_one(&[1.0, 2.0], &mut f16_out, None)?;
    let expected_f16 = [0.25 + 2.0 * -0.5, 0.75 + 2.0 * -1.0];
    let f16_err = f16_out
        .iter()
        .zip(expected_f16)
        .map(|(&got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    if f16_err > 2e-3 {
        return Err(format!("mapped F16 projection max_abs={f16_err:.8e}").into());
    }

    let q8 = Linear::load(&model, "fixture.q8_2f.weight")?;
    let mut q8_out = [0.0f32; 2];
    q8.forward_one(&[1.0; 32], &mut q8_out, None)?;
    let expected_q8: [f32; 2] = [q8_values_for_check(0), q8_values_for_check(1)];
    let q8_err = q8_out
        .iter()
        .zip(expected_q8)
        .map(|(&got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    if q8_err > 2e-2 {
        return Err(format!("Q8_2f projection max_abs={q8_err:.8e}").into());
    }
    let q4tp = Linear::load(&model, "fixture.q4tp.weight")?;
    let mut q4tp_out = [0.0f32; 1];
    q4tp.forward_one(&[1.0; 32], &mut q4tp_out, None)?;
    let q4tp_err = (q4tp_out[0] - q4tp_expected()).abs();
    if q4tp_err > 1e-6 {
        return Err(format!("Q4TP projection max_abs={q4tp_err:.8e}").into());
    }
    println!(
        "qwen_image_denoiser_fixture: cmf={} tokens={} text={} max_abs={max_abs:.8e} f16={f16_err:.8e} q8_2f={q8_err:.8e} q4tp={q4tp_err:.8e}",
        cmf_path.display(),
        fixture.expected.len() / 2,
        fixture.text_len,
    );
    Ok(())
}

fn q8_values_for_check(row: usize) -> f32 {
    let values: Vec<f32> = (0..64)
        .map(|i| ((i as f32 * 0.17).sin() * 0.25) + 0.01)
        .collect();
    let source = &values[row * 32..(row + 1) * 32];
    let absmax = source.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let scale = if absmax == 0.0 {
        1.0 / 127.0
    } else {
        absmax / 127.0
    };
    let scale = cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(scale));
    source
        .iter()
        .map(|&v| (v / scale).round().clamp(-128.0, 127.0) * scale)
        .sum()
}
