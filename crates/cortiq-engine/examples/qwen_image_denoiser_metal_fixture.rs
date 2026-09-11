//! Exercise the native Metal attention and Q4TP full-forward paths.
//!
//! The companion Python oracle emits the same seeded official Diffusers state
//! as the tiny proof, but with hidden=128, two layers, 4,102 image tokens and
//! 8 text tokens.  That makes the combined attention sequence 4,110 (above the
//! Metal attention gate) and the image MLP GEMMs large enough to cross the
//! native Q4TP matmat gate.  The fixture writes two CMFs from identical float
//! weights: an F16 matrix model and a Q4TP matrix model.
//!
//! Run the CPU/GPU comparison under the repository's exclusive GPU lock:
//!
//! ```text
//! flock /private/tmp/cmf-qwen-image-model/gpu.lock \
//!   env CMF_GPU=1 CMF_GPU_PROBE=0 CMF_SDOT=0 CMF_METAL_MMPROF=1 \
//!   cargo run -p cortiq-engine --example qwen_image_denoiser_metal_fixture -- \
//!   /tmp/qwen-image-denoiser-metal-fixture.json \
//!   /tmp/qwen-image-denoiser-metal-f16.cmf \
//!   /tmp/qwen-image-denoiser-metal-q4tp.cmf
//! ```
//!
//! If the host exposes no Metal device, the same complete CPU/oracle loop can
//! be run explicitly with `CMF_GPU=0 CMF_REQUIRE_METAL=0`; that mode reports
//! `backend=cpu-only` and does not satisfy the native-kernel counters.
//! Set `CMF_SDOT=0` for the exact scalar quantized reference contract.

use cortiq_core::{CMF_VERSION, CmfHeader, CmfModel, TensorDtype, TensorSpec};
use cortiq_engine::gpu;
use cortiq_engine::qwen_image::QwenImageTransformer;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const GROUP_SIZE: usize = 32;
const Q4TP_LMAX: usize = 31;
const F16_TINY: f32 = 6.103_515_6e-5;

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
    #[serde(default)]
    expected_q4tp: Option<Vec<f32>>,
}

#[derive(Clone, Copy)]
struct Geometry {
    patch_size: usize,
    in_channels: usize,
    out_channels: usize,
    heads: usize,
    head_dim: usize,
    hidden: usize,
    joint_attention_dim: usize,
    layers: usize,
    intermediate: usize,
}

fn cfg_usize(cfg: &serde_json::Value, key: &str) -> Result<usize, Box<dyn Error>> {
    cfg[key]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("config missing integer '{key}'").into())
}

fn geometry(fixture: &Fixture) -> Result<Geometry, Box<dyn Error>> {
    let patch_size = cfg_usize(&fixture.config, "patch_size")?;
    let in_channels = cfg_usize(&fixture.config, "in_channels")?;
    let out_channels = cfg_usize(&fixture.config, "out_channels")?;
    let heads = cfg_usize(&fixture.config, "num_attention_heads")?;
    let head_dim = cfg_usize(&fixture.config, "attention_head_dim")?;
    let hidden = heads
        .checked_mul(head_dim)
        .ok_or("hidden dimension overflow")?;
    let joint_attention_dim = cfg_usize(&fixture.config, "joint_attention_dim")?;
    let layers = cfg_usize(&fixture.config, "num_layers")?;
    let first_mlp = fixture
        .weights
        .get("transformer_blocks.0.img_mlp.net.0.proj.weight")
        .ok_or("missing first image MLP weight")?;
    let intermediate = first_mlp
        .len()
        .checked_div(hidden)
        .filter(|&n| n > 0)
        .ok_or("invalid first image MLP weight")?;
    Ok(Geometry {
        patch_size,
        in_channels,
        out_channels,
        heads,
        head_dim,
        hidden,
        joint_attention_dim,
        layers,
        intermediate,
    })
}

fn product(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn shape_for(name: &str, values_len: usize, g: Geometry) -> Result<Vec<usize>, Box<dyn Error>> {
    let output_features = g
        .patch_size
        .checked_mul(g.patch_size)
        .and_then(|v| v.checked_mul(g.out_channels))
        .ok_or("output geometry overflow")?;
    let shape = if name == "txt_norm.weight" {
        vec![g.joint_attention_dim]
    } else if name == "img_in.weight" {
        vec![g.hidden, g.in_channels]
    } else if name == "txt_in.weight" {
        vec![g.hidden, g.joint_attention_dim]
    } else if name.ends_with("linear_1.weight") {
        vec![g.hidden, 256]
    } else if name.ends_with("linear_2.weight") {
        vec![g.hidden, g.hidden]
    } else if name.ends_with("img_mod.1.weight") || name.ends_with("txt_mod.1.weight") {
        vec![6 * g.hidden, g.hidden]
    } else if name.ends_with("attn.norm_q.weight")
        || name.ends_with("attn.norm_k.weight")
        || name.ends_with("attn.norm_added_q.weight")
        || name.ends_with("attn.norm_added_k.weight")
    {
        vec![g.head_dim]
    } else if name.ends_with("img_mlp.net.0.proj.weight")
        || name.ends_with("txt_mlp.net.0.proj.weight")
    {
        vec![g.intermediate, g.hidden]
    } else if name.ends_with("img_mlp.net.2.weight") || name.ends_with("txt_mlp.net.2.weight") {
        vec![g.hidden, g.intermediate]
    } else if name == "norm_out.linear.weight" {
        vec![2 * g.hidden, g.hidden]
    } else if name == "proj_out.weight" {
        vec![output_features, g.hidden]
    } else if name.ends_with("img_mod.1.bias") || name.ends_with("txt_mod.1.bias") {
        vec![6 * g.hidden]
    } else if name.ends_with("linear_1.bias") || name.ends_with("linear_2.bias") {
        vec![g.hidden]
    } else if name == "proj_out.bias" {
        vec![output_features]
    } else if name == "norm_out.linear.bias" {
        vec![2 * g.hidden]
    } else if name.ends_with("img_mlp.net.0.proj.bias") || name.ends_with("txt_mlp.net.0.proj.bias")
    {
        vec![g.intermediate]
    } else if name.ends_with(".bias") {
        vec![g.hidden]
    } else if name.ends_with(".weight") {
        vec![g.hidden, g.hidden]
    } else {
        return Err(format!("cannot infer shape for {name}").into());
    };
    if product(&shape) != values_len {
        return Err(format!(
            "{name} inferred shape {shape:?} has {} values, JSON has {values_len}",
            product(&shape)
        )
        .into());
    }
    Ok(shape)
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

fn f16_scale(raw: f32) -> f32 {
    cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(raw)).max(F16_TINY)
}

/// Encode the matrix with the same q4tp plane layout used by the converter:
/// all nibbles first, then one `(f16 lo, f16 step)` ladder per row, then the
/// row-aligned five-bit tile codes. This is fixture arithmetic only; no
/// production dtype or loader code is added here.
fn q4tp_bytes(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols % GROUP_SIZE, 0);
    let groups = cols / GROUP_SIZE;
    let stride = (groups * 5).div_ceil(8);
    let nib_len = rows * groups * 16;
    let params_len = rows * 4;
    let mut out = vec![0u8; nib_len + params_len + rows * stride];
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let mut logs = vec![0.0f32; groups];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for (g, log) in logs.iter_mut().enumerate() {
            let tile = &row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let absmax = tile.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            *log = f16_scale(absmax / 7.0).log2();
            if absmax != 0.0 {
                lo = lo.min(*log);
                hi = hi.max(*log);
            }
        }
        if !lo.is_finite() {
            lo = logs[0];
            hi = lo;
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
            let tile = &row[group * GROUP_SIZE..(group + 1) * GROUP_SIZE];
            let dst = &mut out[(r * groups + group) * 16..(r * groups + group + 1) * 16];
            for k in 0..16 {
                let q0 = (tile[2 * k] * inv).round_ties_even().clamp(-8.0, 7.0) as i8;
                let q1 = (tile[2 * k + 1] * inv).round_ties_even().clamp(-8.0, 7.0) as i8;
                dst[k] = ((q0 + 8) as u8 & 0x0f) | (((q1 + 8) as u8 & 0x0f) << 4);
            }
        }
    }
    out
}

fn is_matrix(shape: &[usize], name: &str) -> bool {
    shape.len() == 2 && name.ends_with(".weight")
}

fn header(g: Geometry, dtype_name: &str) -> Result<CmfHeader, Box<dyn Error>> {
    Ok(serde_json::from_value(json!({
        "version": CMF_VERSION,
        "arch": {
            "arch_name": "qwen-image-metal-fixture",
            "hidden_size": g.hidden,
            "intermediate_size": g.intermediate,
            "num_layers": g.layers,
            "num_attention_heads": g.heads,
            "num_kv_heads": g.heads,
            "head_dim": g.head_dim,
            "vocab_size": 0,
            "layer_types": [],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192
        },
        "quant_type": dtype_name
    }))?)
}

fn write_cmf(
    fixture: &Fixture,
    g: Geometry,
    path: &Path,
    q4tp: bool,
) -> Result<(), Box<dyn Error>> {
    let mut names: Vec<&String> = fixture.weights.keys().collect();
    names.sort();
    let mut tensors = Vec::with_capacity(names.len() + 2);
    let config = serde_json::to_vec(&fixture.config)?;
    tensors.push(TensorSpec {
        name: "image.config_json".into(),
        dtype: TensorDtype::U8,
        shape: vec![config.len()],
        data: config,
    });
    for name in names {
        let values = fixture.weights.get(name).unwrap();
        let shape = shape_for(name, values.len(), g)?;
        let matrix = is_matrix(&shape, name);
        let (dtype, data) = if q4tp && matrix {
            if shape[1] % GROUP_SIZE != 0 {
                return Err(format!("Q4TP matrix {name} has nonmultiple-32 cols").into());
            }
            (
                TensorDtype::Q4TiledP,
                q4tp_bytes(values, shape[0], shape[1]),
            )
        } else if matrix {
            (TensorDtype::F16, f16_bytes(values))
        } else {
            (TensorDtype::F32, f32_bytes(values))
        };
        tensors.push(TensorSpec {
            name: name.clone(),
            dtype,
            shape,
            data,
        });
    }
    // Keep the final q4tp matrix wholly inside the page-truncated no-copy
    // arena used by Metal. Production CMFs naturally have trailing tensors;
    // this focused file adds an inert page-sized tail for the same contract.
    tensors.push(TensorSpec {
        name: "fixture.padding".into(),
        dtype: TensorDtype::F32,
        shape: vec![1024, 4],
        data: vec![0u8; 1024 * 4 * 4],
    });
    // The file-level field is informational; per-tensor dtypes carry the
    // mixed F16/Q4TP truth and QuantType has no Q4TP variant.
    CmfModel::write(path, &header(g, "F32")?, &tensors, None, None)?;
    Ok(())
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_finite(values: &[f32]) -> f32 {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(f32::abs)
        .fold(0.0, f32::max)
}

fn run(
    transformer: &QwenImageTransformer,
    fixture: &Fixture,
    cpu: bool,
) -> Result<(Vec<f32>, Duration), String> {
    let start = Instant::now();
    let output = if cpu {
        gpu::cpu_scope(|| {
            transformer.forward(
                &fixture.image,
                &fixture.text,
                &fixture.shapes,
                fixture.text_len,
                fixture.timestep,
            )
        })?
    } else {
        transformer.forward(
            &fixture.image,
            &fixture.text,
            &fixture.shapes,
            fixture.text_len,
            fixture.timestep,
        )?
    };
    Ok((output, start.elapsed()))
}

fn main() -> Result<(), Box<dyn Error>> {
    if !cfg!(target_os = "macos") {
        return Err("this fixture requires the native macOS Metal backend".into());
    }
    let require_metal = !matches!(
        std::env::var("CMF_REQUIRE_METAL").as_deref(),
        Ok("0") | Ok("off")
    );
    if require_metal && std::env::var("CMF_GPU").as_deref() != Ok("1") {
        return Err("run with CMF_GPU=1 to exercise native Metal (or set CMF_REQUIRE_METAL=0 for an explicit CPU-only oracle run)".into());
    }
    let metal_active = gpu::enabled();
    if require_metal && !metal_active {
        #[cfg(target_os = "macos")]
        return Err(format!(
            "CMF_GPU=1 did not initialize a Metal backend: {:?}",
            cortiq_engine::gpu_metal::initialization_error()
        )
        .into());
        #[cfg(not(target_os = "macos"))]
        return Err("CMF_GPU=1 did not initialize a Metal backend".into());
    }
    // Enable the existing q4tp phase counters; the fixture reports counts but
    // never claims a performance result from this correctness run.
    unsafe { std::env::set_var("CMF_METAL_MMPROF", "1") };

    let mut args = std::env::args_os().skip(1);
    let fixture_path = PathBuf::from(args.next().ok_or("missing oracle JSON path")?);
    let f16_path = PathBuf::from(args.next().ok_or("missing F16 CMF path")?);
    let q4tp_path = PathBuf::from(args.next().ok_or("missing Q4TP CMF path")?);
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&fixture_path)?)?;
    let g = geometry(&fixture)?;
    let image_tokens = fixture.image.len() / g.in_channels;
    let combined = image_tokens
        .checked_add(fixture.text_len)
        .ok_or("combined sequence overflow")?;
    if image_tokens < 32 || combined < 128 || g.hidden < 64 || g.head_dim % 2 != 0 {
        return Err(format!(
            "fixture does not cross Metal gates: image_tokens={image_tokens} combined={combined} hidden={} head_dim={}",
            g.hidden, g.head_dim
        )
        .into());
    }
    write_cmf(&fixture, g, &f16_path, false)?;
    write_cmf(&fixture, g, &q4tp_path, true)?;

    let f16_model = Arc::new(CmfModel::open(&f16_path)?);
    if !f16_model.verify().is_empty() {
        return Err(format!("F16 CMF failed verification: {:?}", f16_model.verify()).into());
    }
    let q4tp_model = Arc::new(CmfModel::open(&q4tp_path)?);
    if !q4tp_model.verify().is_empty() {
        return Err(format!("Q4TP CMF failed verification: {:?}", q4tp_model.verify()).into());
    }
    let f16_transformer = QwenImageTransformer::from_cmf(&f16_model)?;
    let q4tp_transformer = QwenImageTransformer::from_cmf(&q4tp_model)?;

    #[cfg(target_os = "macos")]
    let submit_before =
        cortiq_engine::gpu_metal::METAL_SUBMITS.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_os = "macos")]
    let mm_before = cortiq_engine::gpu_metal::MM_N.load(std::sync::atomic::Ordering::Relaxed);

    let (f16_gpu, f16_gpu_time) = run(&f16_transformer, &fixture, false)?;
    let (f16_cpu, f16_cpu_time) = run(&f16_transformer, &fixture, true)?;
    let (q4tp_gpu, q4tp_gpu_time) = run(&q4tp_transformer, &fixture, false)?;
    let (q4tp_cpu, q4tp_cpu_time) = run(&q4tp_transformer, &fixture, true)?;

    #[cfg(target_os = "macos")]
    let submit_after =
        cortiq_engine::gpu_metal::METAL_SUBMITS.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_os = "macos")]
    let mm_after = cortiq_engine::gpu_metal::MM_N.load(std::sync::atomic::Ordering::Relaxed);

    if f16_gpu.len() != fixture.expected.len() || f16_cpu.len() != fixture.expected.len() {
        return Err(format!(
            "F16 output length GPU/CPU={}/{} != oracle {}",
            f16_gpu.len(),
            f16_cpu.len(),
            fixture.expected.len()
        )
        .into());
    }
    let f16_oracle_err = max_abs(&f16_cpu, &fixture.expected);
    let f16_cpu_gpu_err = max_abs(&f16_cpu, &f16_gpu);
    let q4tp_cpu_gpu_err = max_abs(&q4tp_cpu, &q4tp_gpu);
    let q4tp_vs_f16_err = max_abs(&q4tp_cpu, &f16_cpu);
    let q4tp_oracle = fixture
        .expected_q4tp
        .as_deref()
        .ok_or("oracle JSON is missing expected_q4tp")?;
    if q4tp_cpu.len() != q4tp_oracle.len() {
        return Err(format!(
            "Q4TP output length {} != dequantized oracle {}",
            q4tp_cpu.len(),
            q4tp_oracle.len()
        )
        .into());
    }
    let q4tp_oracle_err = max_abs(&q4tp_cpu, q4tp_oracle);
    if f16_gpu.iter().any(|v| !v.is_finite())
        || f16_cpu.iter().any(|v| !v.is_finite())
        || q4tp_gpu.iter().any(|v| !v.is_finite())
        || q4tp_cpu.iter().any(|v| !v.is_finite())
    {
        return Err("Metal or CPU full-forward output contained non-finite values".into());
    }
    // The native attention kernels use f32 arithmetic but reduction ordering
    // differs from the CPU path. Keep this a correctness gate, not a perf
    // claim; a multi-ulp spread is expected on a 4k sequence.
    if f16_cpu_gpu_err > 2e-3 {
        return Err(format!("F16 CPU/GPU max_abs={f16_cpu_gpu_err:.8e} > 2e-3").into());
    }
    if f16_oracle_err > 2e-3 {
        return Err(
            format!("F16 CPU/dequantized Torch max_abs={f16_oracle_err:.8e} > 2e-3").into(),
        );
    }
    if q4tp_cpu_gpu_err > 5e-3 {
        return Err(format!("Q4TP CPU/GPU max_abs={q4tp_cpu_gpu_err:.8e} > 5e-3").into());
    }
    if q4tp_oracle_err > 2e-3 {
        return Err(
            format!("Q4TP CPU/dequantized Torch max_abs={q4tp_oracle_err:.8e} > 2e-3").into(),
        );
    }

    #[cfg(target_os = "macos")]
    let (metal_submits, q4tp_matmat_calls) = (submit_after - submit_before, mm_after - mm_before);
    #[cfg(not(target_os = "macos"))]
    let (metal_submits, q4tp_matmat_calls) = (0, 0);
    if metal_active {
        if metal_submits == 0 {
            return Err("GPU forward made no Metal command submissions".into());
        }
        if q4tp_matmat_calls == 0 {
            return Err("Q4TP forward did not enter the Metal q4tp matmat path".into());
        }
    }

    println!(
        "qwen_image_denoiser_metal_fixture: backend={} image_tokens={image_tokens} combined={combined} hidden={} layers={} f16_oracle={f16_oracle_err:.8e} f16_cpu_gpu={f16_cpu_gpu_err:.8e} q4tp_oracle={q4tp_oracle_err:.8e} q4tp_cpu_gpu={q4tp_cpu_gpu_err:.8e} q4tp_vs_f16={q4tp_vs_f16_err:.8e} f16_gpu_ms={:.3} f16_cpu_ms={:.3} q4tp_gpu_ms={:.3} q4tp_cpu_ms={:.3} metal_submits={metal_submits} q4tp_matmat_calls={q4tp_matmat_calls} f16_peak={:.3e} q4tp_peak={:.3e}",
        if metal_active { "metal" } else { "cpu-only" },
        g.hidden,
        g.layers,
        f16_gpu_time.as_secs_f64() * 1e3,
        f16_cpu_time.as_secs_f64() * 1e3,
        q4tp_gpu_time.as_secs_f64() * 1e3,
        q4tp_cpu_time.as_secs_f64() * 1e3,
        max_abs_finite(&f16_cpu),
        max_abs_finite(&q4tp_cpu),
    );
    Ok(())
}
