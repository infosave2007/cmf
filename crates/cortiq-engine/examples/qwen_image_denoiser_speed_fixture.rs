//! Real-shape Qwen Image projection fixture for the WGPU speed gate.
//!
//! This deliberately exercises the exact Q4TP tensor layout used by the
//! converter without constructing a full transformer or a full f32 copy of
//! either matrix.  The default geometry is the measured Qwen prefill shape:
//! 5120 rows, hidden 3072, intermediate 12288.  The fixture compares the
//! existing two-GEMM fallback with the device-resident tanh-GELU chain and
//! can also compare the three-projection QKV submission with three fallback
//! submissions. `qwen-block` exercises the complete two-stream Qwen block
//! contract on a small nonuniform token panel, including both MLP streams.
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
//! The fourth argument is optional (`qkv` enables the Q/K/V comparison,
//! `qwen` runs the two-stream resident attention A/B, and `qwen-block` runs
//! the complete resident block against a decomposed reference).
//! This fixture is an operator gate; it does not claim full-image quality or
//! production throughput.

use cortiq_core::{CMF_VERSION, CmfHeader, CmfModel, TensorDtype, TensorSpec};
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

fn header(hidden: usize, inter: usize, heads: usize) -> Result<CmfHeader, Box<dyn Error>> {
    Ok(serde_json::from_value(json!({
        "version": CMF_VERSION,
        "arch": {
            "arch_name": "qwen-image-speed-fixture",
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_layers": 2,
            "num_attention_heads": heads,
            "num_kv_heads": heads,
            "head_dim": hidden / heads,
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

fn write_cmf(
    path: &Path,
    hidden: usize,
    inter: usize,
    qkv: bool,
    qwen: bool,
    qwen_block: bool,
) -> Result<(), Box<dyn Error>> {
    let mut tensors = Vec::with_capacity(if qwen_block {
        12
    } else if qwen {
        10
    } else if qkv {
        8
    } else {
        3
    });
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
    if qwen {
        for (name, seed) in [
            ("qwen.image_q.weight", 101usize),
            ("qwen.image_k.weight", 113),
            ("qwen.image_v.weight", 127),
            ("qwen.text_q.weight", 139),
            ("qwen.text_k.weight", 151),
            ("qwen.text_v.weight", 163),
            ("qwen.image_out.weight", 173),
            ("qwen.text_out.weight", 181),
        ] {
            tensors.push(TensorSpec {
                name: name.into(),
                dtype: TensorDtype::Q4TiledP,
                shape: vec![hidden, hidden],
                data: q4tp_bytes(hidden, hidden, seed),
            });
        }
    }
    if qwen_block {
        for (name, seed, rows, cols) in [
            ("qwen.text_mlp_in.weight", 191usize, inter, hidden),
            ("qwen.text_mlp_out.weight", 197, hidden, inter),
        ] {
            tensors.push(TensorSpec {
                name: name.into(),
                dtype: TensorDtype::Q4TiledP,
                shape: vec![rows, cols],
                data: q4tp_bytes(rows, cols, seed),
            });
        }
    }
    let heads = if qwen_block { 4 } else { 24 };
    CmfModel::write(path, &header(hidden, inter, heads)?, &tensors, None, None)?;
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

fn qwen_mlp_reference(
    model: &Arc<CmfModel>,
    x: &[f32],
    b: usize,
    hidden: usize,
    inter: usize,
    bias_in: &[f32],
    bias_out: &[f32],
    modulation: &[f32],
    gate: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    if modulation.len() != 2 * hidden || gate.len() != hidden {
        return Err("Qwen MLP reference modulation/gate shape mismatch".into());
    }
    let mut normed = vec![0.0f32; x.len()];
    for (r, src) in x.chunks_exact(hidden).take(b).enumerate() {
        let dst = &mut normed[r * hidden..(r + 1) * hidden];
        let mean = src.iter().map(|&v| v as f64).sum::<f64>() / hidden as f64;
        let var = src
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / hidden as f64;
        let inv = 1.0f64 / (var + 1.0e-6).sqrt();
        for (j, (&v, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
            let n = (v as f64 - mean) * inv;
            *d = (n as f32) * (1.0 + modulation[hidden + j]) + modulation[j];
        }
    }
    let mut mid = vec![0.0f32; b * inter];
    if !cortiq_engine::gpu::q4tp_matmat(model, 0, &normed, b, inter, hidden, &mut mid) {
        return Err("Qwen MLP reference input Q4TP refused".into());
    }
    gelu_fallback(&mut mid, b, inter, bias_in);
    let mut projected = vec![0.0f32; b * hidden];
    if !cortiq_engine::gpu::q4tp_matmat(model, 1, &mid, b, hidden, inter, &mut projected) {
        return Err("Qwen MLP reference output Q4TP refused".into());
    }
    let mut out = x.to_vec();
    for (i, y) in projected.iter().enumerate().take(b * hidden) {
        let j = i % hidden;
        out[i] += gate[j] * (*y + bias_out[j]);
    }
    Ok(out)
}

fn qwen_mlp_ab(
    model: &Arc<CmfModel>,
    b: usize,
    hidden: usize,
    inter: usize,
) -> Result<(), Box<dyn Error>> {
    let x = input_values(b, hidden);
    let bias_in = bias_values(inter, 197);
    let bias_out = bias_values(hidden, 211);
    let mut modulation = bias_values(hidden, 223);
    modulation.extend(bias_values(hidden, 227));
    let gate = bias_values(hidden, 229);
    let expected = qwen_mlp_reference(
        model,
        &x,
        b,
        hidden,
        inter,
        &bias_in,
        &bias_out,
        &modulation,
        &gate,
    )?;
    unsafe {
        std::env::set_var("CMF_QWEN_IMAGE_RESIDENT", "1");
        std::env::set_var("CMF_QWEN_IMAGE_FUSED_MLP", "1");
    }
    let mut got = x.clone();
    let t0 = Instant::now();
    if !cortiq_engine::gpu::qwen_image_mlp_inplace(
        model,
        0,
        1,
        &mut got,
        b,
        hidden,
        inter,
        &bias_in,
        &bias_out,
        &modulation,
        &gate,
    ) {
        return Err("resident Qwen MLP block refused".into());
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let abs = max_abs(&got, &expected);
    let rel = rel_rms(&got, &expected);
    println!(
        "qwen mlp block A/B: b={b} hidden={hidden} inter={inter} resident_ms={ms:.3} max_abs={abs:.8e} rel_rms={rel:.8e}"
    );
    if !got.iter().all(|v| v.is_finite()) || abs > 0.35 || rel > 1.0e-2 {
        return Err(format!(
            "resident Qwen MLP parity failed: max_abs={abs:.8e} rel_rms={rel:.8e}"
        )
        .into());
    }
    Ok(())
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

fn attention_values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let z = (i
                .wrapping_mul(37)
                .wrapping_add(seed.wrapping_mul(11))
                .wrapping_add((i / 17).wrapping_mul(5))
                % 257) as f32;
            (z - 128.0) / 1024.0
        })
        .collect()
}

fn qwen_angles(tokens: usize, pairs: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(tokens * pairs);
    let mut sin = Vec::with_capacity(tokens * pairs);
    for t in 0..tokens {
        for j in 0..pairs {
            let angle = ((t + 1 + seed) as f32) * (j + 1) as f32 * 0.0031;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

fn qwen_norm_rope_host(
    x: &[f32],
    tokens: usize,
    heads: usize,
    hd: usize,
    bias: &[f32],
    norm: &[f32],
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    let hidden = heads * hd;
    let pairs = hd / 2;
    let mut out = vec![0.0f32; x.len()];
    for t in 0..tokens {
        for h in 0..heads {
            let off = t * hidden + h * hd;
            let mut sum = 0.0f64;
            for d in 0..hd {
                let v = x[off + d] + bias[h * hd + d];
                sum += (v as f64) * (v as f64);
                out[off + d] = v;
            }
            let inv = 1.0f64 / (sum / hd as f64 + 1e-6).sqrt();
            for d in 0..hd {
                out[off + d] = (out[off + d] as f64 * inv) as f32 * norm[d];
            }
            for j in 0..pairs {
                let a = out[off + 2 * j];
                let b = out[off + 2 * j + 1];
                let c = cos[t * pairs + j];
                let s = sin[t * pairs + j];
                out[off + 2 * j] = a * c - b * s;
                out[off + 2 * j + 1] = a * s + b * c;
            }
        }
    }
    out
}

fn qwen_attention_host(
    txt_q: &[f32],
    txt_k: &[f32],
    txt_v: &[f32],
    img_q: &[f32],
    img_k: &[f32],
    img_v: &[f32],
    text_tokens: usize,
    image_tokens: usize,
    heads: usize,
    hd: usize,
) -> Vec<f32> {
    let hidden = heads * hd;
    let n = text_tokens + image_tokens;
    let mut q = Vec::with_capacity(n * hidden);
    let mut k = Vec::with_capacity(n * hidden);
    let mut v = Vec::with_capacity(n * hidden);
    q.extend_from_slice(txt_q);
    q.extend_from_slice(img_q);
    k.extend_from_slice(txt_k);
    k.extend_from_slice(img_k);
    v.extend_from_slice(txt_v);
    v.extend_from_slice(img_v);
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; n * hidden];
    let mut scores = vec![0.0f32; n];
    for t in 0..n {
        for h in 0..heads {
            let qoff = t * hidden + h * hd;
            let mut max = f32::NEG_INFINITY;
            for u in 0..n {
                let koff = u * hidden + h * hd;
                let mut dot = 0.0;
                for d in 0..hd {
                    dot += q[qoff + d] * k[koff + d];
                }
                scores[u] = dot * scale;
                max = max.max(scores[u]);
            }
            let mut z = 0.0;
            for s in &mut scores {
                *s = (*s - max).exp();
                z += *s;
            }
            let inv = 1.0 / z.max(1e-30);
            for d in 0..hd {
                let mut acc = 0.0;
                for u in 0..n {
                    acc += scores[u] * inv * v[u * hidden + h * hd + d];
                }
                out[qoff + d] = acc;
            }
        }
    }
    out
}

fn qwen_attention_ab(
    model: &Arc<CmfModel>,
    image_tokens: usize,
    text_tokens: usize,
    hidden: usize,
) -> Result<(), Box<dyn Error>> {
    let heads = 24;
    let hd = hidden / heads;
    if hd == 0 || hd % 2 != 0 || hidden % heads != 0 {
        return Err("qwen mode requires hidden divisible by 24 with even head_dim".into());
    }
    let image = input_values(image_tokens, hidden);
    let text = input_values(text_tokens, hidden)
        .into_iter()
        .map(|v| v * 0.7 + 0.013)
        .collect::<Vec<_>>();
    let (image_cos, image_sin) = qwen_angles(image_tokens, hd / 2, 7);
    let (text_cos, text_sin) = qwen_angles(text_tokens, hd / 2, 19);
    let image_q_bias = bias_values(hidden, 3);
    let image_k_bias = bias_values(hidden, 5);
    let image_v_bias = bias_values(hidden, 7);
    let text_q_bias = bias_values(hidden, 11);
    let text_k_bias = bias_values(hidden, 13);
    let text_v_bias = bias_values(hidden, 17);
    let image_out_bias = bias_values(hidden, 23);
    let text_out_bias = bias_values(hidden, 29);
    let image_q_norm = bias_values(hd, 31)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let image_k_norm = bias_values(hd, 37)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let text_q_norm = bias_values(hd, 41)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let text_k_norm = bias_values(hd, 43)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    // The independent f64 attention oracle is intentionally bounded to the
    // small nonuniform parity fixture.  At the production prefill shape an
    // O(tokens²) host oracle would dominate the measurement and allocate a
    // misleading second copy of the attention panels; the resident path is
    // still checked for finite output there and is timed cold/warm.
    let expected = if image_tokens.saturating_add(text_tokens) <= 512 {
        let mut qkv = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        for idx in 0..6 {
            let (source, batch) = if idx < 3 {
                (&image, image_tokens)
            } else {
                (&text, text_tokens)
            };
            let dst = &mut qkv[idx];
            dst.resize(batch * hidden, 0.0);
            if !cortiq_engine::gpu::q4tp_matmat(model, idx + 2, source, batch, hidden, hidden, dst)
            {
                return Err(format!("qwen host QKV refused at index {}", idx + 2).into());
            }
        }
        let img_q = qwen_norm_rope_host(
            &qkv[0],
            image_tokens,
            heads,
            hd,
            &image_q_bias,
            &image_q_norm,
            &image_cos,
            &image_sin,
        );
        let img_k = qwen_norm_rope_host(
            &qkv[1],
            image_tokens,
            heads,
            hd,
            &image_k_bias,
            &image_k_norm,
            &image_cos,
            &image_sin,
        );
        let img_v = qkv[2]
            .chunks_exact(hidden)
            .flat_map(|row| row.iter().enumerate().map(|(d, &v)| v + image_v_bias[d]))
            .collect::<Vec<_>>();
        let txt_q = qwen_norm_rope_host(
            &qkv[3],
            text_tokens,
            heads,
            hd,
            &text_q_bias,
            &text_q_norm,
            &text_cos,
            &text_sin,
        );
        let txt_k = qwen_norm_rope_host(
            &qkv[4],
            text_tokens,
            heads,
            hd,
            &text_k_bias,
            &text_k_norm,
            &text_cos,
            &text_sin,
        );
        let txt_v = qkv[5]
            .chunks_exact(hidden)
            .flat_map(|row| row.iter().enumerate().map(|(d, &v)| v + text_v_bias[d]))
            .collect::<Vec<_>>();
        let baseline = qwen_attention_host(
            &txt_q,
            &txt_k,
            &txt_v,
            &img_q,
            &img_k,
            &img_v,
            text_tokens,
            image_tokens,
            heads,
            hd,
        );
        let mut img_out = vec![0.0f32; image_tokens * hidden];
        let mut txt_out = vec![0.0f32; text_tokens * hidden];
        if !cortiq_engine::gpu::q4tp_matmat(
            model,
            8,
            &baseline[text_tokens * hidden..],
            image_tokens,
            hidden,
            hidden,
            &mut img_out,
        ) || !cortiq_engine::gpu::q4tp_matmat(
            model,
            9,
            &baseline[..text_tokens * hidden],
            text_tokens,
            hidden,
            hidden,
            &mut txt_out,
        ) {
            return Err("qwen host output projection refused".into());
        }
        for row in img_out.chunks_exact_mut(hidden) {
            for (v, &b) in row.iter_mut().zip(&image_out_bias) {
                *v += b;
            }
        }
        for row in txt_out.chunks_exact_mut(hidden) {
            for (v, &b) in row.iter_mut().zip(&text_out_bias) {
                *v += b;
            }
        }
        let mut expected = img_out;
        expected.extend_from_slice(&txt_out);
        Some(expected)
    } else {
        None
    };

    let resident_once = || -> Result<(f64, Vec<f32>, Vec<f32>), Box<dyn Error>> {
        let mut resident_img = vec![0.0f32; image_tokens * hidden];
        let mut resident_txt = vec![0.0f32; text_tokens * hidden];
        let mut args = cortiq_engine::gpu::QwenImageAttentionArgs {
            image: &image,
            text: &text,
            image_tokens,
            text_tokens,
            heads,
            head_dim: hd,
            image_q: 2,
            image_k: 3,
            image_v: 4,
            text_q: 5,
            text_k: 6,
            text_v: 7,
            image_out: 8,
            text_out: 9,
            image_q_norm: &image_q_norm,
            image_k_norm: &image_k_norm,
            text_q_norm: &text_q_norm,
            text_k_norm: &text_k_norm,
            image_cos: &image_cos,
            image_sin: &image_sin,
            text_cos: &text_cos,
            text_sin: &text_sin,
            image_q_bias: &image_q_bias,
            image_k_bias: &image_k_bias,
            image_v_bias: &image_v_bias,
            text_q_bias: &text_q_bias,
            text_k_bias: &text_k_bias,
            text_v_bias: &text_v_bias,
            image_out_bias: &image_out_bias,
            text_out_bias: &text_out_bias,
            image_proj: &mut resident_img,
            text_proj: &mut resident_txt,
        };
        unsafe {
            std::env::set_var("CMF_QWEN_IMAGE_RESIDENT", "1");
        }
        let t0 = Instant::now();
        if !cortiq_engine::gpu::qwen_image_attention(model, &mut args) {
            return Err("resident Qwen attention refused".into());
        }
        Ok((t0.elapsed().as_secs_f64() * 1e3, resident_img, resident_txt))
    };
    let (cold_ms, cold_img, cold_txt) = resident_once()?;
    let (warm_ms, resident_img, resident_txt) = resident_once()?;
    let mut got = resident_img;
    got.extend_from_slice(&resident_txt);
    if let Some(expected) = expected {
        let abs = max_abs(&got, &expected);
        let rel = rel_rms(&got, &expected);
        println!(
            "qwen attention A/B: cold_ms={cold_ms:.3} warm_ms={warm_ms:.3} image_tokens={image_tokens} text_tokens={text_tokens} hidden={hidden} heads={heads} head_dim={hd} max_abs={abs:.8e} rel_rms={rel:.8e}"
        );
        if !got.iter().all(|v| v.is_finite()) || abs > 0.35 || rel > 1.0e-2 {
            return Err(format!(
                "resident Qwen parity failed: max_abs={abs:.8e} rel_rms={rel:.8e}"
            )
            .into());
        }
    } else {
        let finite = cold_img.iter().chain(&cold_txt).all(|v| v.is_finite())
            && got.iter().all(|v| v.is_finite());
        let max_value = got.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        println!(
            "qwen attention warm: cold_ms={cold_ms:.3} warm_ms={warm_ms:.3} image_tokens={image_tokens} text_tokens={text_tokens} hidden={hidden} heads={heads} head_dim={hd} parity=skipped finite={finite} max_abs_value={max_value:.8e}"
        );
        if !finite {
            return Err("resident Qwen real-shape output was non-finite".into());
        }
    }
    Ok(())
}

fn qwen_layer_norm_mod_host(
    x: &[f32],
    tokens: usize,
    hidden: usize,
    modulation: &[f32],
) -> Vec<f32> {
    assert_eq!(x.len(), tokens * hidden);
    assert_eq!(modulation.len(), 2 * hidden);
    let mut out = vec![0.0f32; x.len()];
    for (r, src) in x.chunks_exact(hidden).take(tokens).enumerate() {
        let mean = src.iter().map(|&v| v as f64).sum::<f64>() / hidden as f64;
        let var = src
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / hidden as f64;
        let inv = 1.0f64 / (var + 1.0e-6).sqrt();
        let dst = &mut out[r * hidden..(r + 1) * hidden];
        for (j, (&v, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
            let n = (v as f64 - mean) * inv;
            *d = (n as f32) * (1.0 + modulation[hidden + j]) + modulation[j];
        }
    }
    out
}

fn qwen_mlp_stream_reference(
    model: &Arc<CmfModel>,
    input_idx: usize,
    output_idx: usize,
    x: &[f32],
    tokens: usize,
    hidden: usize,
    inter: usize,
    bias_in: &[f32],
    bias_out: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut mid = vec![0.0f32; tokens * inter];
    if !cortiq_engine::gpu::q4tp_matmat(model, input_idx, x, tokens, inter, hidden, &mut mid) {
        return Err(format!("Qwen full-block MLP input refused at index {input_idx}").into());
    }
    gelu_fallback(&mut mid, tokens, inter, bias_in);
    let mut out = vec![0.0f32; tokens * hidden];
    if !cortiq_engine::gpu::q4tp_matmat(model, output_idx, &mid, tokens, hidden, inter, &mut out) {
        return Err(format!("Qwen full-block MLP output refused at index {output_idx}").into());
    }
    for row in out.chunks_exact_mut(hidden).take(tokens) {
        for (v, &b) in row.iter_mut().zip(bias_out) {
            *v += b;
        }
    }
    Ok(out)
}

/// Decomposed reference for the complete device block. The six Q4TP
/// projections and both MLPs use the established projection API, while all
/// stream joins, norms, RoPE, attention, residuals and GELU stay on this
/// side of the comparison. This catches ordering and buffer-window errors
/// without retaining a dense f32 weight copy.
#[allow(clippy::too_many_arguments)]
fn qwen_block_reference(
    model: &Arc<CmfModel>,
    image: &[f32],
    text: &[f32],
    image_tokens: usize,
    text_tokens: usize,
    heads: usize,
    head_dim: usize,
    hidden: usize,
    inter: usize,
    image_mod: &[f32],
    text_mod: &[f32],
    image_cos: &[f32],
    image_sin: &[f32],
    text_cos: &[f32],
    text_sin: &[f32],
    image_q_norm: &[f32],
    image_k_norm: &[f32],
    text_q_norm: &[f32],
    text_k_norm: &[f32],
    image_q_bias: &[f32],
    image_k_bias: &[f32],
    image_v_bias: &[f32],
    text_q_bias: &[f32],
    text_k_bias: &[f32],
    text_v_bias: &[f32],
    image_out_bias: &[f32],
    text_out_bias: &[f32],
    image_mlp_in_bias: &[f32],
    image_mlp_out_bias: &[f32],
    text_mlp_in_bias: &[f32],
    text_mlp_out_bias: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let image_norm =
        qwen_layer_norm_mod_host(image, image_tokens, hidden, &image_mod[..2 * hidden]);
    let text_norm = qwen_layer_norm_mod_host(text, text_tokens, hidden, &text_mod[..2 * hidden]);

    let mut qkv = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    for idx in 0..6 {
        let (source, tokens) = if idx < 3 {
            (&image_norm, image_tokens)
        } else {
            (&text_norm, text_tokens)
        };
        let dst = &mut qkv[idx];
        dst.resize(tokens * hidden, 0.0);
        if !cortiq_engine::gpu::q4tp_matmat(model, idx + 2, source, tokens, hidden, hidden, dst) {
            return Err(format!("Qwen full-block QKV refused at index {}", idx + 2).into());
        }
    }
    let image_q = qwen_norm_rope_host(
        &qkv[0],
        image_tokens,
        heads,
        head_dim,
        image_q_bias,
        image_q_norm,
        image_cos,
        image_sin,
    );
    let image_k = qwen_norm_rope_host(
        &qkv[1],
        image_tokens,
        heads,
        head_dim,
        image_k_bias,
        image_k_norm,
        image_cos,
        image_sin,
    );
    let image_v = qkv[2]
        .chunks_exact(hidden)
        .flat_map(|row| row.iter().enumerate().map(|(d, &v)| v + image_v_bias[d]))
        .collect::<Vec<_>>();
    let text_q = qwen_norm_rope_host(
        &qkv[3],
        text_tokens,
        heads,
        head_dim,
        text_q_bias,
        text_q_norm,
        text_cos,
        text_sin,
    );
    let text_k = qwen_norm_rope_host(
        &qkv[4],
        text_tokens,
        heads,
        head_dim,
        text_k_bias,
        text_k_norm,
        text_cos,
        text_sin,
    );
    let text_v = qkv[5]
        .chunks_exact(hidden)
        .flat_map(|row| row.iter().enumerate().map(|(d, &v)| v + text_v_bias[d]))
        .collect::<Vec<_>>();
    let attention = qwen_attention_host(
        &text_q,
        &text_k,
        &text_v,
        &image_q,
        &image_k,
        &image_v,
        text_tokens,
        image_tokens,
        heads,
        head_dim,
    );
    let mut image_attn = vec![0.0f32; image_tokens * hidden];
    let mut text_attn = vec![0.0f32; text_tokens * hidden];
    if !cortiq_engine::gpu::q4tp_matmat(
        model,
        8,
        &attention[text_tokens * hidden..],
        image_tokens,
        hidden,
        hidden,
        &mut image_attn,
    ) || !cortiq_engine::gpu::q4tp_matmat(
        model,
        9,
        &attention[..text_tokens * hidden],
        text_tokens,
        hidden,
        hidden,
        &mut text_attn,
    ) {
        return Err("Qwen full-block output projection refused".into());
    }
    for row in image_attn.chunks_exact_mut(hidden) {
        for (v, &b) in row.iter_mut().zip(image_out_bias) {
            *v += b;
        }
    }
    for row in text_attn.chunks_exact_mut(hidden) {
        for (v, &b) in row.iter_mut().zip(text_out_bias) {
            *v += b;
        }
    }

    let mut image_state = image.to_vec();
    let mut text_state = text.to_vec();
    for (i, v) in image_state.iter_mut().enumerate() {
        *v += image_mod[2 * hidden + i % hidden] * image_attn[i];
    }
    for (i, v) in text_state.iter_mut().enumerate() {
        *v += text_mod[2 * hidden + i % hidden] * text_attn[i];
    }
    let image_mlp_norm = qwen_layer_norm_mod_host(
        &image_state,
        image_tokens,
        hidden,
        &image_mod[3 * hidden..5 * hidden],
    );
    let text_mlp_norm = qwen_layer_norm_mod_host(
        &text_state,
        text_tokens,
        hidden,
        &text_mod[3 * hidden..5 * hidden],
    );
    let image_mlp = qwen_mlp_stream_reference(
        model,
        0,
        1,
        &image_mlp_norm,
        image_tokens,
        hidden,
        inter,
        image_mlp_in_bias,
        image_mlp_out_bias,
    )?;
    let text_mlp = qwen_mlp_stream_reference(
        model,
        10,
        11,
        &text_mlp_norm,
        text_tokens,
        hidden,
        inter,
        text_mlp_in_bias,
        text_mlp_out_bias,
    )?;
    for (i, v) in image_state.iter_mut().enumerate() {
        *v += image_mod[5 * hidden + i % hidden] * image_mlp[i];
    }
    for (i, v) in text_state.iter_mut().enumerate() {
        *v += text_mod[5 * hidden + i % hidden] * text_mlp[i];
    }
    Ok((image_state, text_state))
}

#[allow(clippy::too_many_arguments)]
fn qwen_block_ab(
    model: &Arc<CmfModel>,
    image_tokens: usize,
    text_tokens: usize,
    hidden: usize,
    inter: usize,
) -> Result<(), Box<dyn Error>> {
    let heads = 4;
    let head_dim = hidden / heads;
    if hidden % heads != 0 || head_dim % 2 != 0 {
        return Err(
            "qwen-block geometry requires hidden divisible by four and even head_dim".into(),
        );
    }
    let image = input_values(image_tokens, hidden);
    let text = input_values(text_tokens, hidden)
        .into_iter()
        .map(|v| v * 0.7 + 0.013)
        .collect::<Vec<_>>();
    let mut image_mod = bias_values(6 * hidden, 307);
    let mut text_mod = bias_values(6 * hidden, 311);
    // Keep shifts, scales and gates distinct across the two streams and all
    // three modulation sections; repeated zeros would miss stream aliasing.
    for (i, v) in image_mod.iter_mut().enumerate() {
        *v += ((i % 7) as f32 - 3.0) * 0.0007;
    }
    for (i, v) in text_mod.iter_mut().enumerate() {
        *v += ((i % 11) as f32 - 5.0) * 0.0006;
    }
    let image_norm =
        qwen_layer_norm_mod_host(&image, image_tokens, hidden, &image_mod[..2 * hidden]);
    let text_norm = qwen_layer_norm_mod_host(&text, text_tokens, hidden, &text_mod[..2 * hidden]);
    let (image_cos, image_sin) = qwen_angles(image_tokens, head_dim / 2, 17);
    let (text_cos, text_sin) = qwen_angles(text_tokens, head_dim / 2, 23);
    let image_q_bias = bias_values(hidden, 313);
    let image_k_bias = bias_values(hidden, 317);
    let image_v_bias = bias_values(hidden, 331);
    let text_q_bias = bias_values(hidden, 337);
    let text_k_bias = bias_values(hidden, 347);
    let text_v_bias = bias_values(hidden, 349);
    let image_out_bias = bias_values(hidden, 353);
    let text_out_bias = bias_values(hidden, 359);
    let image_q_norm = bias_values(head_dim, 361)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let image_k_norm = bias_values(head_dim, 367)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let text_q_norm = bias_values(head_dim, 373)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let text_k_norm = bias_values(head_dim, 379)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let image_mlp_in_bias = bias_values(inter, 383);
    let image_mlp_out_bias = bias_values(hidden, 389);
    let text_mlp_in_bias = bias_values(inter, 397);
    let text_mlp_out_bias = bias_values(hidden, 401);

    let (expected_img, expected_txt) = qwen_block_reference(
        model,
        &image,
        &text,
        image_tokens,
        text_tokens,
        heads,
        head_dim,
        hidden,
        inter,
        &image_mod,
        &text_mod,
        &image_cos,
        &image_sin,
        &text_cos,
        &text_sin,
        &image_q_norm,
        &image_k_norm,
        &text_q_norm,
        &text_k_norm,
        &image_q_bias,
        &image_k_bias,
        &image_v_bias,
        &text_q_bias,
        &text_k_bias,
        &text_v_bias,
        &image_out_bias,
        &text_out_bias,
        &image_mlp_in_bias,
        &image_mlp_out_bias,
        &text_mlp_in_bias,
        &text_mlp_out_bias,
    )?;

    unsafe {
        std::env::set_var("CMF_QWEN_IMAGE_RESIDENT", "1");
        std::env::set_var("CMF_QWEN_IMAGE_FUSED_MLP", "1");
        std::env::set_var("CMF_QWEN_IMAGE_FUSED_MLP_COOP", "1");
    }
    let mut got_img = image.clone();
    let mut got_txt = text.clone();
    let t0 = Instant::now();
    let mut args = cortiq_engine::gpu::QwenImageBlockArgs {
        image: &mut got_img,
        text: &mut got_txt,
        image_norm: &image_norm,
        text_norm: &text_norm,
        image_tokens,
        text_tokens,
        heads,
        head_dim,
        image_cos: &image_cos,
        image_sin: &image_sin,
        text_cos: &text_cos,
        text_sin: &text_sin,
        image_q: 2,
        image_k: 3,
        image_v: 4,
        text_q: 5,
        text_k: 6,
        text_v: 7,
        image_out: 8,
        text_out: 9,
        image_q_norm: &image_q_norm,
        image_k_norm: &image_k_norm,
        text_q_norm: &text_q_norm,
        text_k_norm: &text_k_norm,
        image_q_bias: &image_q_bias,
        image_k_bias: &image_k_bias,
        image_v_bias: &image_v_bias,
        text_q_bias: &text_q_bias,
        text_k_bias: &text_k_bias,
        text_v_bias: &text_v_bias,
        image_out_bias: &image_out_bias,
        text_out_bias: &text_out_bias,
        image_attn_gate: &image_mod[2 * hidden..3 * hidden],
        text_attn_gate: &text_mod[2 * hidden..3 * hidden],
        image_mlp_in: 0,
        image_mlp_out: 1,
        text_mlp_in: 10,
        text_mlp_out: 11,
        image_mlp_in_bias: &image_mlp_in_bias,
        image_mlp_out_bias: &image_mlp_out_bias,
        text_mlp_in_bias: &text_mlp_in_bias,
        text_mlp_out_bias: &text_mlp_out_bias,
        image_mlp_mod: &image_mod[3 * hidden..5 * hidden],
        text_mlp_mod: &text_mod[3 * hidden..5 * hidden],
        image_mlp_gate: &image_mod[5 * hidden..6 * hidden],
        text_mlp_gate: &text_mod[5 * hidden..6 * hidden],
    };
    if !cortiq_engine::gpu::qwen_image_block(model, &mut args) {
        return Err("resident Qwen full block refused".into());
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut expected = expected_img;
    expected.extend_from_slice(&expected_txt);
    let mut got = got_img;
    got.extend_from_slice(&got_txt);
    let abs = max_abs(&got, &expected);
    let rel = rel_rms(&got, &expected);
    println!(
        "qwen full block A/B: image_tokens={image_tokens} text_tokens={text_tokens} hidden={hidden} heads={heads} head_dim={head_dim} inter={inter} resident_ms={ms:.3} max_abs={abs:.8e} rel_rms={rel:.8e}"
    );
    if !got.iter().all(|v| v.is_finite()) || abs > 0.35 || rel > 1.0e-2 {
        return Err(format!(
            "resident Qwen full-block parity failed: max_abs={abs:.8e} rel_rms={rel:.8e}"
        )
        .into());
    }
    Ok(())
}

/// Two consecutive Qwen blocks over the same device state.  The controls are
/// deliberately different per layer so a stale modulation buffer or a host
/// boundary that accidentally resets the second block is visible.  The
/// projections are reused only because this focused CMF is small; production
/// layers carry distinct directory indices.
#[allow(clippy::too_many_arguments)]
fn qwen_chain_ab(
    model: &Arc<CmfModel>,
    image_tokens: usize,
    text_tokens: usize,
    hidden: usize,
    inter: usize,
) -> Result<(), Box<dyn Error>> {
    let heads = 4;
    let head_dim = hidden / heads;
    if hidden % heads != 0 || head_dim % 2 != 0 {
        return Err("qwen-chain geometry requires hidden divisible by four and even head_dim".into());
    }
    let image = input_values(image_tokens, hidden);
    let text = input_values(text_tokens, hidden)
        .into_iter()
        .map(|v| v * 0.7 + 0.013)
        .collect::<Vec<_>>();
    let mut image_mods = (0..2)
        .map(|layer| bias_values(6 * hidden, 307 + layer * 17))
        .collect::<Vec<_>>();
    let mut text_mods = (0..2)
        .map(|layer| bias_values(6 * hidden, 311 + layer * 19))
        .collect::<Vec<_>>();
    for (layer, values) in image_mods.iter_mut().enumerate() {
        for (i, v) in values.iter_mut().enumerate() {
            *v += ((i % 7) as f32 - 3.0) * (0.0007 + layer as f32 * 0.00011);
        }
    }
    for (layer, values) in text_mods.iter_mut().enumerate() {
        for (i, v) in values.iter_mut().enumerate() {
            *v += ((i % 11) as f32 - 5.0) * (0.0006 + layer as f32 * 0.00009);
        }
    }
    let (image_cos, image_sin) = qwen_angles(image_tokens, head_dim / 2, 17);
    let (text_cos, text_sin) = qwen_angles(text_tokens, head_dim / 2, 23);
    let image_q_bias = bias_values(hidden, 313);
    let image_k_bias = bias_values(hidden, 317);
    let image_v_bias = bias_values(hidden, 331);
    let text_q_bias = bias_values(hidden, 337);
    let text_k_bias = bias_values(hidden, 347);
    let text_v_bias = bias_values(hidden, 349);
    let image_out_bias = bias_values(hidden, 353);
    let text_out_bias = bias_values(hidden, 359);
    let image_q_norm = bias_values(head_dim, 361)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let image_k_norm = bias_values(head_dim, 367)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let text_q_norm = bias_values(head_dim, 373)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let text_k_norm = bias_values(head_dim, 379)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let image_mlp_in_bias = bias_values(inter, 383);
    let image_mlp_out_bias = bias_values(hidden, 389);
    let text_mlp_in_bias = bias_values(inter, 397);
    let text_mlp_out_bias = bias_values(hidden, 401);

    let (expected_img0, expected_txt0) = qwen_block_reference(
        model,
        &image,
        &text,
        image_tokens,
        text_tokens,
        heads,
        head_dim,
        hidden,
        inter,
        &image_mods[0],
        &text_mods[0],
        &image_cos,
        &image_sin,
        &text_cos,
        &text_sin,
        &image_q_norm,
        &image_k_norm,
        &text_q_norm,
        &text_k_norm,
        &image_q_bias,
        &image_k_bias,
        &image_v_bias,
        &text_q_bias,
        &text_k_bias,
        &text_v_bias,
        &image_out_bias,
        &text_out_bias,
        &image_mlp_in_bias,
        &image_mlp_out_bias,
        &text_mlp_in_bias,
        &text_mlp_out_bias,
    )?;
    let (expected_img, expected_txt) = qwen_block_reference(
        model,
        &expected_img0,
        &expected_txt0,
        image_tokens,
        text_tokens,
        heads,
        head_dim,
        hidden,
        inter,
        &image_mods[1],
        &text_mods[1],
        &image_cos,
        &image_sin,
        &text_cos,
        &text_sin,
        &image_q_norm,
        &image_k_norm,
        &text_q_norm,
        &text_k_norm,
        &image_q_bias,
        &image_k_bias,
        &image_v_bias,
        &text_q_bias,
        &text_k_bias,
        &text_v_bias,
        &image_out_bias,
        &text_out_bias,
        &image_mlp_in_bias,
        &image_mlp_out_bias,
        &text_mlp_in_bias,
        &text_mlp_out_bias,
    )?;

    let make_block = |layer: usize| cortiq_engine::gpu::QwenImageChainBlock {
        image_mod: &image_mods[layer],
        text_mod: &text_mods[layer],
        image_q: 2,
        image_k: 3,
        image_v: 4,
        text_q: 5,
        text_k: 6,
        text_v: 7,
        image_out: 8,
        text_out: 9,
        image_q_norm: &image_q_norm,
        image_k_norm: &image_k_norm,
        text_q_norm: &text_q_norm,
        text_k_norm: &text_k_norm,
        image_q_bias: &image_q_bias,
        image_k_bias: &image_k_bias,
        image_v_bias: &image_v_bias,
        text_q_bias: &text_q_bias,
        text_k_bias: &text_k_bias,
        text_v_bias: &text_v_bias,
        image_out_bias: &image_out_bias,
        text_out_bias: &text_out_bias,
        image_attn_gate: &image_mods[layer][2 * hidden..3 * hidden],
        text_attn_gate: &text_mods[layer][2 * hidden..3 * hidden],
        image_mlp_in: 0,
        image_mlp_out: 1,
        text_mlp_in: 10,
        text_mlp_out: 11,
        image_mlp_in_bias: &image_mlp_in_bias,
        image_mlp_out_bias: &image_mlp_out_bias,
        text_mlp_in_bias: &text_mlp_in_bias,
        text_mlp_out_bias: &text_mlp_out_bias,
    };
    let specs = vec![make_block(0), make_block(1)];
    unsafe {
        std::env::set_var("CMF_QWEN_IMAGE_RESIDENT", "1");
        std::env::set_var("CMF_QWEN_IMAGE_FUSED_MLP", "1");
        std::env::set_var("CMF_QWEN_IMAGE_FUSED_MLP_COOP", "1");
    }
    let mut got_img = image.clone();
    let mut got_txt = text.clone();
    let t0 = Instant::now();
    let mut args = cortiq_engine::gpu::QwenImageChainArgs {
        image: &mut got_img,
        text: &mut got_txt,
        image_tokens,
        text_tokens,
        heads,
        head_dim,
        image_cos: &image_cos,
        image_sin: &image_sin,
        text_cos: &text_cos,
        text_sin: &text_sin,
        blocks: &specs,
    };
    if !cortiq_engine::gpu::qwen_image_chain(model, &mut args) {
        return Err("resident Qwen chain refused".into());
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut expected = expected_img;
    expected.extend_from_slice(&expected_txt);
    let mut got = got_img;
    got.extend_from_slice(&got_txt);
    let abs = max_abs(&got, &expected);
    let rel = rel_rms(&got, &expected);
    println!(
        "qwen full chain A/B: layers=2 image_tokens={image_tokens} text_tokens={text_tokens} hidden={hidden} heads={heads} head_dim={head_dim} inter={inter} resident_ms={ms:.3} max_abs={abs:.8e} rel_rms={rel:.8e}"
    );
    if !got.iter().all(|v| v.is_finite()) || abs > 0.35 || rel > 1.0e-2 {
        return Err(format!(
            "resident Qwen chain parity failed: max_abs={abs:.8e} rel_rms={rel:.8e}"
        )
        .into());
    }
    Ok(())
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
    let mode = args
        .next()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let qkv = mode == "qkv";
    let qwen = mode == "qwen";
    let qwen_block = mode == "qwen-block";
    let qwen_chain = mode == "qwen-chain";
    if !mode.is_empty() && !qkv && !qwen && !qwen_block && !qwen_chain {
        return Err("optional mode must be qkv, qwen, qwen-block, or qwen-chain".into());
    }
    if b < 32 || hidden == 0 || inter == 0 || hidden % 32 != 0 || inter % 32 != 0 {
        return Err("geometry requires b>=32 and hidden/inter multiples of 32".into());
    }

    write_cmf(
        &cmf_path,
        hidden,
        inter,
        qkv,
        qwen || qwen_block || qwen_chain,
        qwen_block || qwen_chain,
    )?;
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
        "speed fixture: backend=wgpu b={b} hidden={hidden} inter={inter} x_f32_mib={x_mb:.1} intermediate_f32_mib={mid_mb:.1} qkv={qkv} qwen={qwen}"
    );

    if qwen_chain {
        let text_tokens = (b / 4).max(1);
        qwen_chain_ab(&model, b, text_tokens, hidden, inter)?;
        return Ok(());
    }
    if qwen_block {
        let text_tokens = (b / 4).max(1);
        qwen_block_ab(&model, b, text_tokens, hidden, inter)?;
        return Ok(());
    }

    if qwen {
        let text_tokens = (b / 4).max(1);
        // The full image geometry is reserved for the resident attention
        // timing; a bounded 128-row panel proves the second Qwen sub-block
        // against its decomposed host reference without another large
        // activation allocation.
        qwen_mlp_ab(&model, b.min(128).max(32), hidden, inter)?;
        qwen_attention_ab(&model, b, text_tokens, hidden)?;
        return Ok(());
    }

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
