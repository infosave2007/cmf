//! Real-shape Qwen Image projection fixture for the WGPU speed gate.
//!
//! This deliberately exercises the exact Q4TP tensor layout used by the
//! converter without constructing a full transformer or a full f32 copy of
//! either matrix.  The default geometry is the measured Qwen prefill shape:
//! 5120 rows, hidden 3072, intermediate 12288.  The fixture compares the
//! existing two-GEMM fallback with the device-resident tanh-GELU chain and
//! can also compare the three-projection QKV submission with three fallback
//! submissions.
//!
//! Build once through the repository wrapper, then run the binary under the
//! shared GPU lock:
//!
//! ```text
//! CMF_GPU=wgpu CMF_GPU_PROBE=0 CMF_GPU_DEBUG=1 \
//!   CMF_QWEN_IMAGE_FUSED_MLP_COOP=1 \
//!   ./target/debug/examples/qwen_image_denoiser_speed_fixture \
//!   /tmp/qwen-image-speed-q4tp.cmf 5120 3072 12288 qkv
//! ```
//!
//! The fourth argument is optional (`qkv` enables the Q/K/V comparison).
//! This fixture is an operator gate; it does not claim full-image quality or
//! production throughput.

use cortiq_core::{CmfHeader, CmfModel, TensorDtype, TensorSpec, CMF_VERSION};
use serde_json::json;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const GROUP_SIZE: usize = 32;
const Q4TP_LMAX: usize = 31;
const F16_TINY: f32 = 6.103515625e-5;
const GELU_C: f32 = 0.7978846;
const GELU_K: f32 = 0.044715;

fn f16_scale(raw: f32) -> f32 {
    cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(raw)).max(F16_TINY)
}

/// Deterministic, nonuniform fixture values.  The bytes are generated
/// directly, so no dense f32 matrix is kept alive while the CMF is written.
fn fixture_value(seed: usize, row: usize, col: usize) -> f32 {
    let z = row
        .wrapping_mul(97)
        .wrapping_add(col.wrapping_mul(13))
        .wrapping_add(seed)
        % 257;
    (z as f32 - 128.0) / 512.0
}

/// Encode the converter's Q4TP plane layout: packed nibbles, one f16
/// `(log2(lo), step)` pair per row, and row-aligned five-bit group codes.
fn q4tp_bytes(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert!(rows > 0 && cols % GROUP_SIZE == 0);
    let groups = cols / GROUP_SIZE;
    let stride = (groups * 5).div_ceil(8);
    let nib_len = rows * groups * 16;
    let params_len = rows * 4;
    let mut out = vec![0u8; nib_len + params_len + rows * stride];
    let mut logs = vec![0.0f32; groups];

    for r in 0..rows {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for (g, log) in logs.iter_mut().enumerate() {
            let mut absmax = 0.0f32;
            for c in 0..GROUP_SIZE {
                absmax = absmax.max(fixture_value(seed, r, g * GROUP_SIZE + c).abs());
            }
            *log = f16_scale(absmax / 7.0).log2();
            lo = lo.min(*log);
            hi = hi.max(*log);
        }
        if !lo.is_finite() {
            lo = 0.0;
            hi = 0.0;
        }
        let params_off = nib_len + r * 4;
        let lo_h = cortiq_core::quant::f32_to_f16(lo);
        let lo_r = cortiq_core::quant::f16_to_f32(lo_h);
        let span = (hi - lo_r).max(0.0);
        let mut step_h = cortiq_core::quant::f32_to_f16(span / Q4TP_LMAX as f32);
        for _ in 0..64 {
            let step = cortiq_core::quant::f16_to_f32(step_h);
            if step > 0.0 && lo_r + Q4TP_LMAX as f32 * step >= hi {
                break;
            }
            step_h = step_h.saturating_add(1);
        }
        out[params_off..params_off + 2].copy_from_slice(&lo_h.to_le_bytes());
        out[params_off + 2..params_off + 4].copy_from_slice(&step_h.to_le_bytes());
        let table = cortiq_core::quant::q4tp_ladder(&out[nib_len..nib_len + params_len], r);
        let codes_off = nib_len + params_len + r * stride;
        for (group, &log) in logs.iter().enumerate() {
            let step = cortiq_core::quant::f16_to_f32(step_h);
            let code = if step <= 0.0 {
                0
            } else {
                ((log - lo_r) / step)
                    .round_ties_even()
                    .clamp(0.0, Q4TP_LMAX as f32) as usize
            };
            cortiq_core::quant::q4tp_put_code(&mut out[codes_off..codes_off + stride], group, code);
            let scale = table[code];
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let dst = &mut out[(r * groups + group) * 16..(r * groups + group + 1) * 16];
            for k in 0..16 {
                let c0 = group * GROUP_SIZE + 2 * k;
                let q0 = (fixture_value(seed, r, c0) * inv)
                    .round_ties_even()
                    .clamp(-8.0, 7.0) as i8;
                let q1 = (fixture_value(seed, r, c0 + 1) * inv)
                    .round_ties_even()
                    .clamp(-8.0, 7.0) as i8;
                dst[k] = ((q0 + 8) as u8 & 0x0f) | (((q1 + 8) as u8 & 0x0f) << 4);
            }
        }
    }
    out
}

fn header(hidden: usize, inter: usize) -> Result<CmfHeader, Box<dyn Error>> {
    Ok(serde_json::from_value(json!({
        "version": CMF_VERSION,
        "arch": {
            "arch_name": "qwen-image-speed-fixture",
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_layers": 2,
            "num_attention_heads": 24,
            "num_kv_heads": 24,
            "head_dim": hidden / 24,
            "vocab_size": 0,
            "layer_types": [],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192
        },
        // The envelope enum has no Q4TP variant; per-tensor dtypes carry the
        // actual codec, as they do in the production mixed CMF.
        "quant_type": "F32"
    }))?)
}

fn write_cmf(path: &Path, hidden: usize, inter: usize, qkv: bool) -> Result<(), Box<dyn Error>> {
    let mut tensors = Vec::with_capacity(if qkv { 8 } else { 3 });
    tensors.push(TensorSpec {
        name: "speed.mlp_in.weight".into(),
        dtype: TensorDtype::Q4TiledP,
        shape: vec![inter, hidden],
        data: q4tp_bytes(inter, hidden, 17),
    });
    tensors.push(TensorSpec {
        name: "speed.mlp_out.weight".into(),
        dtype: TensorDtype::Q4TiledP,
        shape: vec![hidden, inter],
        data: q4tp_bytes(hidden, inter, 31),
    });
    if qkv {
        for (name, rows, seed) in [
            ("speed.q.weight", hidden, 43usize),
            ("speed.k.weight", hidden, 59),
            ("speed.v.weight", hidden, 71),
        ] {
            tensors.push(TensorSpec {
                name: name.into(),
                dtype: TensorDtype::Q4TiledP,
                shape: vec![rows, hidden],
                data: q4tp_bytes(rows, hidden, seed),
            });
        }
    }
    CmfModel::write(path, &header(hidden, inter)?, &tensors, None, None)?;
    Ok(())
}

fn input_values(b: usize, hidden: usize) -> Vec<f32> {
    (0..b * hidden)
        .map(|i| {
            let z = (i.wrapping_mul(19).wrapping_add(7) % 257) as f32;
            (z - 128.0) / 1024.0
        })
        .collect()
}

fn bias_values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let z = (i.wrapping_mul(23).wrapping_add(seed) % 101) as f32;
            (z - 50.0) / 10000.0
        })
        .collect()
}

fn gelu_fallback(mid: &mut [f32], b: usize, inter: usize, bias: &[f32]) {
    for row in mid.chunks_exact_mut(inter).take(b) {
        for (j, x) in row.iter_mut().enumerate() {
            let v = *x + bias[j];
            *x = 0.5 * v * (1.0 + (GELU_C * (v + GELU_K * v * v * v)).tanh());
        }
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut d2, mut b2) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = x as f64 - y as f64;
        d2 += d * d;
        b2 += y as f64 * y as f64;
    }
    (d2 / b2.max(1e-30)).sqrt()
}

fn fallback_mlp(
    model: &Arc<CmfModel>,
    x: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
    bias_in: &[f32],
    bias_out: &[f32],
) -> Result<(Vec<f32>, f64), Box<dyn Error>> {
    // Force the old scalar Q4TP dispatch, so the comparison isolates the
    // resident chain rather than comparing two f16 implementations.
    unsafe { std::env::set_var("CMF_Q4TP_PLANE_MIN", "18446744073709551615") };
    let mut mid = vec![0.0f32; b * inter];
    let mut out = vec![0.0f32; b * hidden];
    let t0 = Instant::now();
    if !cortiq_engine::gpu::q4tp_matmat(model, 0, x, b, inter, hidden, &mut mid) {
        return Err("fallback input Q4TP matmat refused".into());
    }
    gelu_fallback(&mut mid, b, inter, bias_in);
    if !cortiq_engine::gpu::q4tp_matmat(model, 1, &mid, b, hidden, inter, &mut out) {
        return Err("fallback output Q4TP matmat refused".into());
    }
    for row in out.chunks_exact_mut(hidden).take(b) {
        for (j, y) in row.iter_mut().enumerate() {
            *y += bias_out[j];
        }
    }
    Ok((out, t0.elapsed().as_secs_f64() * 1e3))
}

fn fused_mlp(
    model: &Arc<CmfModel>,
    x: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
    bias_in: &[f32],
    bias_out: &[f32],
) -> Result<(Vec<f32>, f64), Box<dyn Error>> {
    unsafe {
        std::env::set_var("CMF_QWEN_IMAGE_FUSED_MLP", "1");
        std::env::set_var("CMF_QWEN_IMAGE_FUSED_MLP_COOP", "1");
    }
    let mut out = vec![0.0f32; b * hidden];
    let t0 = Instant::now();
    if !cortiq_engine::gpu::q4tp_gelu_ffn(
        model, 0, 1, x, b, hidden, inter, bias_in, bias_out, &mut out,
    ) {
        return Err("fused Q4TP GELU FFN refused".into());
    }
    Ok((out, t0.elapsed().as_secs_f64() * 1e3))
}

fn fallback_qkv(
    model: &Arc<CmfModel>,
    x: &[f32],
    b: usize,
    hidden: usize,
) -> Result<(Vec<f32>, f64), Box<dyn Error>> {
    let mut out = vec![0.0f32; b * hidden * 3];
    let t0 = Instant::now();
    for (idx, dst) in [(2usize, 0usize), (3, 1), (4, 2)] {
        if !cortiq_engine::gpu::q4tp_matmat(
            model,
            idx,
            x,
            b,
            hidden,
            hidden,
            &mut out[dst * b * hidden..(dst + 1) * b * hidden],
        ) {
            return Err(format!("fallback QKV Q4TP matmat refused at index {idx}").into());
        }
    }
    Ok((out, t0.elapsed().as_secs_f64() * 1e3))
}

fn fused_qkv(
    model: &Arc<CmfModel>,
    x: &[f32],
    b: usize,
    hidden: usize,
) -> Result<(Vec<f32>, f64), Box<dyn Error>> {
    let mut out = vec![0.0f32; b * hidden * 3];
    let t0 = Instant::now();
    let panel = b * hidden;
    let (q_out, rest) = out.split_at_mut(panel);
    let (k_out, v_out) = rest.split_at_mut(panel);
    if !cortiq_engine::gpu::dit_qkv(
        model, 2, 3, 4, x, b, hidden, hidden, hidden, q_out, k_out, v_out,
    ) {
        return Err("fused QKV Q4TP submission refused".into());
    }
    Ok((out, t0.elapsed().as_secs_f64() * 1e3))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let cmf_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/tmp/qwen-image-speed-q4tp.cmf".into()),
    );
    let b = args
        .next()
        .map(|v| v.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(5120);
    let hidden = args
        .next()
        .map(|v| v.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(3072);
    let inter = args
        .next()
        .map(|v| v.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(12288);
    let qkv = args.next().is_some_and(|v| v == "qkv");
    if b < 32 || hidden == 0 || inter == 0 || hidden % 32 != 0 || inter % 32 != 0 {
        return Err("geometry requires b>=32 and hidden/inter multiples of 32".into());
    }

    write_cmf(&cmf_path, hidden, inter, qkv)?;
    let model = Arc::new(CmfModel::open(&cmf_path)?);
    if !model.verify().is_empty() {
        return Err(format!("fixture CMF failed verification: {:?}", model.verify()).into());
    }
    unsafe {
        std::env::set_var("CMF_GPU", "wgpu");
        std::env::set_var("CMF_GPU_PROBE", "0");
    }
    if !cortiq_engine::gpu::enabled() {
        return Err(
            "WGPU backend did not initialize; run with a real Vulkan/Metal WGPU adapter".into(),
        );
    }

    let x = input_values(b, hidden);
    let bias_in = bias_values(inter, 13);
    let bias_out = bias_values(hidden, 29);
    let x_mb = (x.len() * 4) as f64 / (1024.0 * 1024.0);
    let mid_mb = (b * inter * 4) as f64 / (1024.0 * 1024.0);
    println!(
        "speed fixture: backend=wgpu b={b} hidden={hidden} inter={inter} x_f32_mib={x_mb:.1} intermediate_f32_mib={mid_mb:.1} qkv={qkv}"
    );

    let (fallback, fallback_ms) = fallback_mlp(&model, &x, b, hidden, inter, &bias_in, &bias_out)?;
    let (fused, fused_ms) = fused_mlp(&model, &x, b, hidden, inter, &bias_in, &bias_out)?;
    let mlp_abs = max_abs(&fused, &fallback);
    let mlp_rel = rel_rms(&fused, &fallback);
    println!(
        "mlp A/B: fallback_scalar_ms={fallback_ms:.3} fused_resident_ms={fused_ms:.3} max_abs={mlp_abs:.8e} rel_rms={mlp_rel:.8e}"
    );
    if !fused.iter().all(|v| v.is_finite()) || mlp_abs > 0.25 || mlp_rel > 5.0e-3 {
        return Err(format!(
            "fused MLP parity failed: max_abs={mlp_abs:.8e} rel_rms={mlp_rel:.8e}"
        )
        .into());
    }

    if qkv {
        let (qkv_fallback, fallback_ms) = fallback_qkv(&model, &x, b, hidden)?;
        let (qkv_fused, fused_ms) = fused_qkv(&model, &x, b, hidden)?;
        let abs = max_abs(&qkv_fused, &qkv_fallback);
        let rel = rel_rms(&qkv_fused, &qkv_fallback);
        println!(
            "qkv A/B: fallback_3x_scalar_ms={fallback_ms:.3} fused_one_submit_ms={fused_ms:.3} max_abs={abs:.8e} rel_rms={rel:.8e}"
        );
        if !qkv_fused.iter().all(|v| v.is_finite()) || abs > 0.25 || rel > 5.0e-3 {
            return Err(
                format!("fused QKV parity failed: max_abs={abs:.8e} rel_rms={rel:.8e}").into(),
            );
        }
    }
    Ok(())
}
