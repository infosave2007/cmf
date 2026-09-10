//! Compare the V4.1 tokenwise runtime against the deterministic tiny oracle.
//!
//! Usage:
//!   cargo run -p cortiq-engine --example dsv41_tiny_oracle -- <model.cmf> <oracle.json>

use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

fn json_f32s(value: &Value) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(value
        .as_array()
        .ok_or("expected JSON array")?
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|v| v as f32)
                .ok_or_else(|| "expected JSON number".into())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?)
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn compare(got: &[f32], expected: &[f32]) -> (f32, f64, bool) {
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (&a, &b) in got.iter().zip(expected) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let mean_abs = sum_abs / got.len().max(1) as f64;
    (max_abs, mean_abs, argmax(got) == argmax(expected))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_path = args.next().ok_or("missing CMF model path")?;
    let oracle_path = args.next().ok_or("missing oracle JSON path")?;
    if args.next().is_some() {
        return Err("usage: dsv41_tiny_oracle <model.cmf> <oracle.json>".into());
    }

    let oracle: Value = serde_json::from_reader(File::open(&oracle_path)?)?;
    let ids: Vec<u32> = oracle["input_ids"]
        .as_array()
        .ok_or("oracle input_ids is not an array")?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|v| v as u32)
                .ok_or_else(|| "oracle input id is not an integer".into())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let serial = oracle["serial_golden"]
        .as_array()
        .ok_or("oracle serial_golden is not an array")?;
    let prefill_len = oracle["prefill_length"]
        .as_u64()
        .ok_or("oracle prefill_length is not an integer")? as usize;

    let model = Arc::new(CmfModel::open(Path::new(&model_path))?);
    let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default())
        .map_err(|e| format!("pipeline load failed: {e}"))?;
    let rows = pipeline
        .dsv41_serial_logits(&ids)
        .map_err(|e| format!("serial forward failed: {e}"))?;
    if rows.len() != ids.len() {
        return Err(format!("runtime returned {} rows for {} ids", rows.len(), ids.len()).into());
    }

    let mut serial_max = 0.0f32;
    let mut serial_sum = 0.0f64;
    let mut serial_count = 0usize;
    let mut serial_argmax_equal = true;
    for item in serial {
        let position = item["position"]
            .as_u64()
            .ok_or("serial golden position is not an integer")? as usize;
        let expected = json_f32s(&item["logits"])?;
        let got = rows
            .get(position)
            .ok_or_else(|| format!("runtime has no row at position {position}"))?;
        if got.len() != expected.len() {
            return Err(format!(
                "vocab width mismatch at position {position}: runtime={} oracle={}",
                got.len(),
                expected.len()
            )
            .into());
        }
        let (max_abs, mean_abs, argmax_equal) = compare(got, &expected);
        serial_max = serial_max.max(max_abs);
        serial_sum += mean_abs as f64 * got.len() as f64;
        serial_count += got.len();
        serial_argmax_equal &= argmax_equal;
        println!(
            "serial position={position} max_abs={max_abs:.8e} mean_abs={mean_abs:.8e} argmax_equal={argmax_equal}"
        );
    }
    println!(
        "serial summary positions={} max_abs={serial_max:.8e} mean_abs={:.8e} argmax_equal={serial_argmax_equal}",
        serial.len(),
        serial_sum / serial_count.max(1) as f64,
    );

    let final_golden = oracle["golden"]
        .as_array()
        .and_then(|x| x.first())
        .ok_or("oracle golden is empty")?;
    let expected_final = json_f32s(&final_golden["logits"])?;
    let final_logits = pipeline
        .forward_ids(&ids[..prefill_len], None)
        .map_err(|e| format!("prefill forward failed: {e}"))?;
    let (max_abs, mean_abs, argmax_equal) = compare(&final_logits, &expected_final);
    println!(
        "prefill position={} max_abs={max_abs:.8e} mean_abs={mean_abs:.8e} argmax_equal={argmax_equal}",
        final_golden["position"].as_u64().unwrap_or(0),
    );
    if rows.iter().flatten().any(|x| !x.is_finite()) || final_logits.iter().any(|x| !x.is_finite())
    {
        return Err("runtime produced non-finite logits".into());
    }
    Ok(())
}
