//! Compare the converted Q4TP V4.1 experts with the unchanged official
//! MXFP4 expert oracle.
//!
//! The example deliberately uses the engine's `QTensor` row kernels and the
//! V4 SwiGLU helper.  It reports residual and cosine metrics instead of
//! deciding an arbitrary quality threshold; non-finite output, malformed
//! fixtures, and missing canonical tensors remain hard errors.
//!
//! Usage (on the conversion pod):
//!
//! ```text
//! cargo run --release -p cortiq-engine --example dsv41_expert_oracle -- \
//!   /root/dsv41/expert-conversion-probe/probe.cmf \
//!   /root/dsv41/expert-component-reference/oracle.safetensors \
//!   /root/dsv41/expert-component-reference/oracle.json \
//!   /root/dsv41/expert-conversion-probe/model.safetensors
//! ```

use anyhow::{Context, ensure};
use cortiq_core::{CmfModel, TensorDtype};
use cortiq_engine::dsv4::expert_swiglu;
use cortiq_engine::pool::Pool;
use cortiq_engine::qtensor::QTensor;
use cortiq_engine::vae::{StTensor, read_safetensors};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const EXPERT_IDS: [usize; 4] = [0, 1, 10, 100];
const TOKEN_ROWS: usize = 24;

fn read_safetensors_header(path: &Path) -> anyhow::Result<Map<String, Value>> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("read header length from {}", path.display()))?;
    let len = u64::from_le_bytes(len_bytes);
    ensure!(
        len <= 64 * 1024 * 1024,
        "{}: safetensors header is too large ({len} bytes)",
        path.display()
    );
    let mut bytes = vec![0u8; len as usize];
    file.read_exact(&mut bytes)
        .with_context(|| format!("read header from {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse safetensors header {}", path.display()))?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("{}: header is not an object", path.display()))
}

fn shape(meta: &Value, name: &str) -> anyhow::Result<Vec<usize>> {
    meta.get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("source tensor {name}: missing shape"))?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| anyhow::anyhow!("source tensor {name}: invalid shape dim {i}"))
        })
        .collect()
}

/// Check the packed source fixture without reading its 75 MB payload.  This
/// pins the exact layer-0 expert set and proves that the CMF being compared
/// came from the expected MXFP4 weight/scale pairs.
fn verify_source_header(path: &Path, dim: usize, inter: usize) -> anyhow::Result<()> {
    let header = read_safetensors_header(path)?;
    let tensors: Vec<(&String, &Value)> = header
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
        .collect();
    ensure!(
        tensors.len() == EXPERT_IDS.len() * 3 * 2,
        "{}: expected 24 expert tensors, found {}",
        path.display(),
        tensors.len()
    );
    for &expert in &EXPERT_IDS {
        for (matrix, rows, cols) in [("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)] {
            let packed_name = format!("layers.0.ffn.experts.{expert}.{matrix}.weight");
            let scale_name = format!("layers.0.ffn.experts.{expert}.{matrix}.scale");
            let packed = header
                .get(&packed_name)
                .ok_or_else(|| anyhow::anyhow!("source is missing {packed_name}"))?;
            let scales = header
                .get(&scale_name)
                .ok_or_else(|| anyhow::anyhow!("source is missing {scale_name}"))?;
            ensure!(
                packed.get("dtype").and_then(Value::as_str) == Some("I8"),
                "{packed_name}: expected packed I8"
            );
            ensure!(
                shape(packed, &packed_name)? == [rows, cols / 2],
                "{packed_name}: unexpected packed shape"
            );
            ensure!(
                scales.get("dtype").and_then(Value::as_str) == Some("F8_E8M0"),
                "{scale_name}: expected F8_E8M0 scales"
            );
            ensure!(
                shape(scales, &scale_name)? == [rows, cols / 32],
                "{scale_name}: unexpected scale shape"
            );
        }
    }
    Ok(())
}

fn oracle_tensor<'a>(
    tensors: &'a HashMap<String, StTensor>,
    name: &str,
) -> anyhow::Result<&'a StTensor> {
    tensors
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("oracle is missing tensor {name}"))
}

fn check_shape(tensor: &StTensor, want: &[usize], name: &str) -> anyhow::Result<()> {
    ensure!(
        tensor.shape == want,
        "{name}: shape {:?}, expected {want:?}",
        tensor.shape
    );
    let n = want.iter().product::<usize>();
    ensure!(
        tensor.data.len() == n,
        "{name}: {} values, expected {n}",
        tensor.data.len()
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    elements: usize,
    max_abs: f32,
    mean_abs: f64,
    rmse: f64,
    cosine: f64,
    finite_got: bool,
    finite_expected: bool,
}

fn measure(got: &[f32], expected: &[f32]) -> anyhow::Result<Metrics> {
    ensure!(
        got.len() == expected.len(),
        "comparison length mismatch: got {}, expected {}",
        got.len(),
        expected.len()
    );
    let finite_got = got.iter().all(|v| v.is_finite());
    let finite_expected = expected.iter().all(|v| v.is_finite());
    let mut max_abs = 0.0f32;
    let mut abs = 0.0f64;
    let mut sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut got_sq = 0.0f64;
    let mut expected_sq = 0.0f64;
    for (&a, &b) in got.iter().zip(expected) {
        let d = (a - b).abs();
        if d.is_finite() {
            max_abs = max_abs.max(d);
            abs += d as f64;
            sq += (d as f64) * (d as f64);
        }
        dot += a as f64 * b as f64;
        got_sq += a as f64 * a as f64;
        expected_sq += b as f64 * b as f64;
    }
    let n = got.len().max(1) as f64;
    let cosine_den = (got_sq * expected_sq).sqrt();
    Ok(Metrics {
        elements: got.len(),
        max_abs,
        mean_abs: abs / n,
        rmse: (sq / n).sqrt(),
        cosine: if cosine_den > 0.0 {
            dot / cosine_den
        } else {
            f64::NAN
        },
        finite_got,
        finite_expected,
    })
}

fn forward_expert(
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    x: &[f32],
    route_weight: f32,
    inter: usize,
    dim: usize,
    swiglu_limit: f32,
    pool: &Pool,
) -> (Vec<f32>, Vec<f32>, bool) {
    let mut plain = vec![0.0f32; dim];
    let mut weighted = vec![0.0f32; dim];
    let mut mid = vec![0.0f32; inter];
    let fused = QTensor::matvec_silu_mul_limited(gate, up, x, &mut mid, swiglu_limit, Some(pool));
    if fused {
        down.matvec(&mid, &mut plain, Some(pool));
        for value in &mut mid {
            *value *= route_weight;
        }
        down.matvec(&mid, &mut weighted, Some(pool));
    } else {
        expert_swiglu(
            x,
            &|input, output| gate.matvec(input, output, Some(pool)),
            &|input, output| up.matvec(input, output, Some(pool)),
            &|input, output| down.matvec(input, output, Some(pool)),
            inter,
            1.0,
            swiglu_limit,
            &mut plain,
        );
        expert_swiglu(
            x,
            &|input, output| gate.matvec(input, output, Some(pool)),
            &|input, output| up.matvec(input, output, Some(pool)),
            &|input, output| down.matvec(input, output, Some(pool)),
            inter,
            route_weight,
            swiglu_limit,
            &mut weighted,
        );
    }
    (plain, weighted, fused)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(
        args.len() == 4,
        "usage: dsv41_expert_oracle <probe.cmf> <oracle.safetensors> <oracle.json> <source.safetensors>"
    );
    // This diagnostic measures the portable QTensor path.  It must not
    // silently turn into a device probe whose upload/cache cost obscures the
    // quantization comparison; production GPU parity has separate fixtures.
    unsafe { std::env::set_var("CMF_GPU", "0") };

    let model_path = &args[0];
    let oracle_path = &args[1];
    let report_path = &args[2];
    let source_path = &args[3];
    let report: Value = serde_json::from_reader(
        File::open(report_path).with_context(|| format!("open {}", report_path.display()))?,
    )
    .with_context(|| format!("parse {}", report_path.display()))?;
    let dim = report["dim"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| anyhow::anyhow!("oracle report has no integer dim"))?;
    let inter = report["inter_dim"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| anyhow::anyhow!("oracle report has no integer inter_dim"))?;
    let swiglu_limit = report["swiglu_limit"]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("oracle report has no numeric swiglu_limit"))?
        as f32;
    ensure!(dim > 0 && inter > 0 && swiglu_limit.is_finite());
    let expert_report = report["experts"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("oracle report has no experts array"))?;
    for &expert in &EXPERT_IDS {
        ensure!(
            expert_report.iter().any(|row| {
                row["expert"].as_u64() == Some(expert as u64)
                    && row["shape"] == serde_json::json!([TOKEN_ROWS, dim])
            }),
            "oracle report has no shape record for expert {expert}"
        );
    }
    verify_source_header(source_path, dim, inter)?;
    println!(
        "source_header=ok tensors={} dim={} inter={} swiglu_limit={swiglu_limit}",
        EXPERT_IDS.len() * 3 * 2,
        dim,
        inter
    );

    let oracle = read_safetensors(oracle_path).map_err(anyhow::Error::msg)?;
    let input = oracle_tensor(&oracle, "input")?;
    let route = oracle_tensor(&oracle, "route_weights")?;
    check_shape(input, &[TOKEN_ROWS, dim], "input")?;
    check_shape(route, &[TOKEN_ROWS, 1], "route_weights")?;
    ensure!(
        input.data.iter().all(|v| v.is_finite()) && route.data.iter().all(|v| v.is_finite()),
        "oracle inputs contain non-finite values"
    );

    let model = Arc::new(CmfModel::open(model_path)?);
    let integrity = model.verify();
    ensure!(
        integrity.is_empty(),
        "probe CMF integrity errors: {integrity:?}"
    );
    let pool = Pool::new(3);
    let mut aggregate_plain_got = Vec::with_capacity(EXPERT_IDS.len() * TOKEN_ROWS * dim);
    let mut aggregate_plain_expected = Vec::with_capacity(aggregate_plain_got.capacity());
    let mut aggregate_weighted_got = Vec::with_capacity(aggregate_plain_got.capacity());
    let mut aggregate_weighted_expected = Vec::with_capacity(aggregate_plain_got.capacity());
    let mut fused_calls = 0usize;
    let mut forward_calls = 0usize;

    for &expert in &EXPERT_IDS {
        let prefix = format!("model.layers.0.mlp.experts.{expert}");
        let gate_name = format!("{prefix}.gate_proj.weight");
        let up_name = format!("{prefix}.up_proj.weight");
        let down_name = format!("{prefix}.down_proj.weight");
        let gate = QTensor::from_model(&model, &gate_name).map_err(anyhow::Error::msg)?;
        let up = QTensor::from_model(&model, &up_name).map_err(anyhow::Error::msg)?;
        let down = QTensor::from_model(&model, &down_name).map_err(anyhow::Error::msg)?;
        ensure!(
            gate.model_dtype() == Some(TensorDtype::Q4TiledP)
                && up.model_dtype() == Some(TensorDtype::Q4TiledP)
                && down.model_dtype() == Some(TensorDtype::Q4TiledP),
            "expert {expert}: expected Q4TiledP gate/up/down tensors"
        );
        ensure!(
            gate.rows() == inter
                && gate.cols() == dim
                && up.rows() == inter
                && up.cols() == dim
                && down.rows() == dim
                && down.cols() == inter,
            "expert {expert}: unexpected canonical matrix shapes"
        );

        let expected_plain = oracle_tensor(&oracle, &format!("expert{expert}.output"))?;
        let expected_weighted = oracle_tensor(&oracle, &format!("expert{expert}.weighted_output"))?;
        check_shape(
            expected_plain,
            &[TOKEN_ROWS, dim],
            &format!("expert{expert}.output"),
        )?;
        check_shape(
            expected_weighted,
            &[TOKEN_ROWS, dim],
            &format!("expert{expert}.weighted_output"),
        )?;
        let mut got_plain = Vec::with_capacity(TOKEN_ROWS * dim);
        let mut got_weighted = Vec::with_capacity(TOKEN_ROWS * dim);
        for token in 0..TOKEN_ROWS {
            let x = &input.data[token * dim..(token + 1) * dim];
            let route_weight = route.data[token];
            let (plain, weighted, fused) = forward_expert(
                &gate,
                &up,
                &down,
                x,
                route_weight,
                inter,
                dim,
                swiglu_limit,
                &pool,
            );
            forward_calls += 1;
            fused_calls += usize::from(fused);
            got_plain.extend_from_slice(&plain);
            got_weighted.extend_from_slice(&weighted);
        }
        let plain = measure(&got_plain, &expected_plain.data)?;
        let weighted = measure(&got_weighted, &expected_weighted.data)?;
        println!(
            "expert={expert} mode=plain elements={} max_abs={:.8e} mean_abs={:.8e} rmse={:.8e} cosine={:.9} finite={}",
            plain.elements,
            plain.max_abs,
            plain.mean_abs,
            plain.rmse,
            plain.cosine,
            plain.finite_got && plain.finite_expected
        );
        println!(
            "expert={expert} mode=route_weighted elements={} max_abs={:.8e} mean_abs={:.8e} rmse={:.8e} cosine={:.9} finite={}",
            weighted.elements,
            weighted.max_abs,
            weighted.mean_abs,
            weighted.rmse,
            weighted.cosine,
            weighted.finite_got && weighted.finite_expected
        );
        ensure!(
            plain.finite_got
                && plain.finite_expected
                && weighted.finite_got
                && weighted.finite_expected,
            "expert {expert}: non-finite forward output"
        );
        aggregate_plain_got.extend_from_slice(&got_plain);
        aggregate_plain_expected.extend_from_slice(&expected_plain.data);
        aggregate_weighted_got.extend_from_slice(&got_weighted);
        aggregate_weighted_expected.extend_from_slice(&expected_weighted.data);
    }

    let plain = measure(&aggregate_plain_got, &aggregate_plain_expected)?;
    let weighted = measure(&aggregate_weighted_got, &aggregate_weighted_expected)?;
    println!(
        "summary mode=plain experts={} tokens={} elements={} max_abs={:.8e} mean_abs={:.8e} rmse={:.8e} cosine={:.9} finite={} fused_calls={}/{}",
        EXPERT_IDS.len(),
        TOKEN_ROWS,
        plain.elements,
        plain.max_abs,
        plain.mean_abs,
        plain.rmse,
        plain.cosine,
        plain.finite_got && plain.finite_expected,
        fused_calls,
        forward_calls
    );
    println!(
        "summary mode=route_weighted experts={} tokens={} elements={} max_abs={:.8e} mean_abs={:.8e} rmse={:.8e} cosine={:.9} finite={}",
        EXPERT_IDS.len(),
        TOKEN_ROWS,
        weighted.elements,
        weighted.max_abs,
        weighted.mean_abs,
        weighted.rmse,
        weighted.cosine,
        weighted.finite_got && weighted.finite_expected
    );
    ensure!(
        plain.finite_got && weighted.finite_got,
        "aggregate forward produced non-finite values"
    );
    Ok(())
}
