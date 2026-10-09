//! Generic Q4TP MoE execution on the portable GPU paths.
//!
//! Mellum2.1 routes eight of sixty-four experts.  The generic GPU block only
//! receives the selected jobs, so this builds the full 64-expert catalogue and
//! submits eight deliberately non-contiguous winners.  It proves that the
//! selected backend reads the Q4TP triples, keeps their intermediates on the
//! device for the block, and blends them like the scalar CMF definition.
//! It does *not* claim that routing itself is a device kernel: the production
//! router intentionally remains on the host for this generic path.
//!
//! By default this selects WGPU (Vulkan/DX12/Metal).  On macOS, run the same
//! oracle against native Metal in a separate process with
//! `CMF_MOE_TEST_BACKEND=metal`; contexts deliberately remain process-global.

#![cfg(feature = "gpu")]

use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::quant::{dequant_tensor, f32_to_f16, q4tp_put_code, q4tp_sections};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use cortiq_core::{CMF_VERSION, CmfModel};
use cortiq_engine::gpu::MoeJob;
use std::sync::Arc;

const EXPERTS: usize = 64;
const TOP_K: usize = 8;
// Non-contiguous on purpose: a top-k router need not choose adjacent experts.
const WINNERS: [usize; TOP_K] = [63, 51, 44, 31, 19, 10, 6, 0];
const MIX: [f32; TOP_K] = [0.03, 0.07, 0.09, 0.11, 0.13, 0.16, 0.18, 0.23];

fn payload(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let gpr = cols / 32;
    let (params_off, codes_off, stride) = q4tp_sections(rows, cols);
    let mut out = vec![0u8; codes_off + rows * stride];
    for r in 0..rows {
        for g in 0..gpr {
            let tile = &mut out[(r * gpr + g) * 16..(r * gpr + g + 1) * 16];
            for (k, byte) in tile.iter_mut().enumerate() {
                // Both signed nibbles vary; an all-zero/constant payload could
                // make a wrong tensor index or an omitted expert invisible.
                *byte = ((seed * 53 + r * 29 + g * 17 + k * 11) % 251) as u8;
            }
            q4tp_put_code(
                &mut out[codes_off + r * stride..codes_off + (r + 1) * stride],
                g,
                (seed * 7 + r * 5 + g * 3) % 32,
            );
        }
        let p = params_off + r * 4;
        let lo = f32_to_f16(-6.0 + (seed % 3) as f32 * 0.125);
        let step = f32_to_f16(0.02 + (r % 5) as f32 * 0.002);
        out[p..p + 2].copy_from_slice(&lo.to_le_bytes());
        out[p + 2..p + 4].copy_from_slice(&step.to_le_bytes());
    }
    out
}

fn header(hidden: usize, inter: usize) -> CmfHeader {
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "moe-top8-gpu-test",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_layers": 1,
        "num_attention_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "vocab_size": 8,
        "layer_types": ["FullAttention"],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 8,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))
    .unwrap();
    CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type: QuantType::Q4Block,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
        genome: None,
        lineage: Vec::new(),
        router: None,
        segments: Vec::new(),
    }
}

fn mv(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
        .collect()
}

fn dequant(model: &CmfModel, idx: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    dequant_tensor(
        &model.tensors[idx],
        model.entry_bytes(&model.tensors[idx]),
        &mut out,
    )
    .unwrap();
    out
}

#[test]
fn generic_q4tp_moe_top8_of_64_matches_scalar_reference() {
    // Run the universal WGPU backend even on Apple Silicon by default, so
    // this covers the code used through Vulkan/DX12 on non-Apple systems.
    // Native Metal is separately selectable because both backends own
    // process-global contexts and cannot be switched inside one test process.
    let test_backend = std::env::var("CMF_MOE_TEST_BACKEND").unwrap_or_else(|_| "wgpu".into());
    let run: for<'a> fn(&Arc<CmfModel>, &[MoeJob<'a>], &mut [f32]) -> bool =
        match test_backend.as_str() {
            "wgpu" => {
                unsafe { std::env::set_var("CMF_GPU", "wgpu") };
                if !cortiq_engine::gpu_wgpu::enabled() {
                    eprintln!("skipped: no WGPU adapter");
                    return;
                }
                cortiq_engine::gpu_wgpu::moe_block
            }
            "metal" => {
                #[cfg(target_os = "macos")]
                {
                    unsafe { std::env::set_var("CMF_GPU", "1") };
                    if !cortiq_engine::gpu_metal::enabled() {
                        eprintln!("skipped: no native Metal adapter");
                        return;
                    }
                    cortiq_engine::gpu_metal::moe_block
                }
                #[cfg(not(target_os = "macos"))]
                {
                    panic!("CMF_MOE_TEST_BACKEND=metal requires macOS")
                }
            }
            other => panic!("unknown CMF_MOE_TEST_BACKEND={other:?}; use wgpu or metal"),
        };

    // Both Mellum dimensions are multiples of 32.  The compact test geometry
    // keeps the full 64-expert directory cheap while exercising the same Q4TP
    // three-plane layout and the selected top-8 batch contract.
    let (hidden, inter) = (96usize, 128usize);
    let mut specs = Vec::with_capacity(EXPERTS * 3 + 1);
    for e in 0..EXPERTS {
        for (part, rows, cols, salt) in [
            ("gate", inter, hidden, 1usize),
            ("up", inter, hidden, 2),
            ("down", hidden, inter, 3),
        ] {
            specs.push(TensorSpec {
                name: format!("experts.{e}.{part}"),
                dtype: TensorDtype::Q4TiledP,
                shape: vec![rows, cols],
                data: payload(rows, cols, e * 13 + salt),
            });
        }
    }
    // Keep the last mapped Q4TP tensor away from the end-of-file page edge;
    // this matches the file-layout condition used by the runtime's no-copy
    // weight cache.
    specs.push(TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0; 8192 * 8],
    });
    let dir = std::env::temp_dir().join(format!(
        "cmf-wgpu-moe-top8-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("top8.cmf");
    CmfModel::write(&path, &header(hidden, inter), &specs, None, None).unwrap();
    let model = Arc::new(CmfModel::open(&path).unwrap());

    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i * 19 + 3) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let mut jobs = Vec::with_capacity(TOP_K);
    for (&e, &w) in WINNERS.iter().zip(&MIX) {
        let gi = model.tensor_index(&format!("experts.{e}.gate")).unwrap();
        let ui = model.tensor_index(&format!("experts.{e}.up")).unwrap();
        let di = model.tensor_index(&format!("experts.{e}.down")).unwrap();
        jobs.push(MoeJob {
            gate: (gi, inter, hidden, &[]),
            up: (ui, inter, hidden, &[]),
            down: (di, hidden, inter, &[]),
            xs_gate: x.clone(),
            xs_up: x.clone(),
            down_col: &[],
            w,
            q1: false,
            q4t: false,
            q4tp: true,
            gu_q2: false,
            swiglu_limit: 0.0,
        });
    }

    let mut got = vec![0.0; hidden];
    assert!(
        run(&model, &jobs, &mut got),
        "{test_backend} refused a well-formed Q4TP top-8 MoE block"
    );

    let mut want = vec![0.0; hidden];
    for (&e, &w) in WINNERS.iter().zip(&MIX) {
        let gi = model.tensor_index(&format!("experts.{e}.gate")).unwrap();
        let ui = model.tensor_index(&format!("experts.{e}.up")).unwrap();
        let di = model.tensor_index(&format!("experts.{e}.down")).unwrap();
        let g = mv(&dequant(&model, gi, inter, hidden), inter, hidden, &x);
        let u = mv(&dequant(&model, ui, inter, hidden), inter, hidden, &x);
        let a: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
            .collect();
        let d = mv(&dequant(&model, di, hidden, inter), hidden, inter, &a);
        for (dst, v) in want.iter_mut().zip(d) {
            *dst += w * v;
        }
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(&want) {
        assert!(a.is_finite(), "GPU produced a non-finite MoE output");
        num += (a as f64 - b as f64).powi(2);
        den += (b as f64).powi(2);
    }
    let rel = (num / den.max(1e-30)).sqrt();
    eprintln!("{test_backend} q4tp top-8/64 MoE relative RMS error: {rel:.3e}");
    // The generic kernel changes reduction order but must stay well below the
    // quantized-model error budget.  This is a chain test, not a trivially
    // zero-valued matvec test.
    assert!(
        rel < 3e-4,
        "Q4TP top-8 MoE drifted from scalar CMF: {rel:.3e}"
    );

    drop(model);
    std::fs::remove_dir_all(dir).ok();
}

fn q8_row_payload(rows: usize, cols: usize, seed: usize) -> (Vec<u8>, Vec<f32>) {
    let mut out = Vec::with_capacity(rows * cols + rows * 2);
    for r in 0..rows {
        for c in 0..cols {
            out.push((((seed * 31 + r * 13 + c * 7) % 255) as i32 - 127) as i8 as u8);
        }
    }
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let h = f32_to_f16(0.002 + ((seed + r) % 7) as f32 * 0.0005);
        out.extend_from_slice(&h.to_le_bytes());
        scales.push(cortiq_core::quant::f16_to_f32(h));
    }
    (out, scales)
}

/// q8 jobs whose gate/up inputs differ per job — what a q8_2f expert
/// sends (x times that expert's own column field) — plus a per-job down
/// column field. The whole block is one command buffer, so a staging
/// buffer shared between jobs hands every expert the last job's input
/// (native Metal before the per-job key: Mellum2.1 q8_2f experts decoded
/// at wiki ppl 7.39 instead of 7.21).
#[test]
fn q8_moe_jobs_keep_their_own_inputs() {
    let test_backend = std::env::var("CMF_MOE_TEST_BACKEND").unwrap_or_else(|_| "wgpu".into());
    let run: for<'a> fn(&Arc<CmfModel>, &[MoeJob<'a>], &mut [f32]) -> bool =
        match test_backend.as_str() {
            "wgpu" => {
                unsafe { std::env::set_var("CMF_GPU", "wgpu") };
                if !cortiq_engine::gpu_wgpu::enabled() {
                    eprintln!("skipped: no WGPU adapter");
                    return;
                }
                cortiq_engine::gpu_wgpu::moe_block
            }
            "metal" => {
                #[cfg(target_os = "macos")]
                {
                    unsafe { std::env::set_var("CMF_GPU", "1") };
                    if !cortiq_engine::gpu_metal::enabled() {
                        eprintln!("skipped: no native Metal adapter");
                        return;
                    }
                    cortiq_engine::gpu_metal::moe_block
                }
                #[cfg(not(target_os = "macos"))]
                {
                    panic!("CMF_MOE_TEST_BACKEND=metal requires macOS")
                }
            }
            other => panic!("unknown CMF_MOE_TEST_BACKEND={other:?}; use wgpu or metal"),
        };

    let (hidden, inter, n_jobs) = (96usize, 128usize, 4usize);
    let mut specs = Vec::new();
    let mut scales: Vec<[Vec<f32>; 3]> = Vec::new();
    for e in 0..n_jobs {
        let mut s3: [Vec<f32>; 3] = Default::default();
        for (k, (part, rows, cols)) in [("gate", inter, hidden), ("up", inter, hidden), ("down", hidden, inter)]
            .into_iter()
            .enumerate()
        {
            let (data, rs) = q8_row_payload(rows, cols, e * 5 + k);
            specs.push(TensorSpec {
                name: format!("experts.{e}.{part}"),
                dtype: TensorDtype::Q8Row,
                shape: vec![rows, cols],
                data,
            });
            s3[k] = rs;
        }
        scales.push(s3);
    }
    specs.push(TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0; 8192 * 8],
    });
    let dir = std::env::temp_dir().join(format!(
        "cmf-moe-q8-inputs-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("q8jobs.cmf");
    CmfModel::write(&path, &header(hidden, inter), &specs, None, None).unwrap();
    let model = Arc::new(CmfModel::open(&path).unwrap());

    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i * 19 + 3) % 97) as f32 / 97.0 - 0.5)
        .collect();
    // Per-job column fields, as q8_2f experts carry them.
    let col = |e: usize, n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 + ((e * 11 + i * 3 + salt) % 17) as f32 / 16.0)
            .collect()
    };
    let xs_g: Vec<Vec<f32>> = (0..n_jobs)
        .map(|e| x.iter().zip(col(e, hidden, 1)).map(|(a, b)| a * b).collect())
        .collect();
    let xs_u: Vec<Vec<f32>> = (0..n_jobs)
        .map(|e| x.iter().zip(col(e, hidden, 2)).map(|(a, b)| a * b).collect())
        .collect();
    let dcols: Vec<Vec<f32>> = (0..n_jobs).map(|e| col(e, inter, 3)).collect();
    let mix = [0.4f32, 0.3, 0.2, 0.1];
    let mut jobs = Vec::with_capacity(n_jobs);
    for e in 0..n_jobs {
        let gi = model.tensor_index(&format!("experts.{e}.gate")).unwrap();
        let ui = model.tensor_index(&format!("experts.{e}.up")).unwrap();
        let di = model.tensor_index(&format!("experts.{e}.down")).unwrap();
        jobs.push(MoeJob {
            gate: (gi, inter, hidden, &scales[e][0]),
            up: (ui, inter, hidden, &scales[e][1]),
            down: (di, hidden, inter, &scales[e][2]),
            xs_gate: xs_g[e].clone(),
            xs_up: xs_u[e].clone(),
            down_col: &dcols[e],
            w: mix[e],
            q1: false,
            q4t: false,
            q4tp: false,
            gu_q2: false,
            swiglu_limit: 0.0,
        });
    }
    let mut got = vec![0.0; hidden];
    assert!(
        run(&model, &jobs, &mut got),
        "{test_backend} refused a well-formed q8 MoE block"
    );

    let mut want = vec![0.0f32; hidden];
    for e in 0..n_jobs {
        let gi = model.tensor_index(&format!("experts.{e}.gate")).unwrap();
        let ui = model.tensor_index(&format!("experts.{e}.up")).unwrap();
        let di = model.tensor_index(&format!("experts.{e}.down")).unwrap();
        let g = mv(&dequant(&model, gi, inter, hidden), inter, hidden, &xs_g[e]);
        let u = mv(&dequant(&model, ui, inter, hidden), inter, hidden, &xs_u[e]);
        let a: Vec<f32> = g
            .iter()
            .zip(&u)
            .zip(&dcols[e])
            .map(|((&gv, &uv), &c)| gv / (1.0 + (-gv).exp()) * uv * c)
            .collect();
        let d = mv(&dequant(&model, di, hidden, inter), hidden, inter, &a);
        for (dst, v) in want.iter_mut().zip(d) {
            *dst += mix[e] * v;
        }
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(&want) {
        assert!(a.is_finite(), "GPU produced a non-finite MoE output");
        num += (a as f64 - b as f64).powi(2);
        den += (b as f64).powi(2);
    }
    let rel = (num / den.max(1e-30)).sqrt();
    eprintln!("{test_backend} q8 per-job-input MoE relative RMS error: {rel:.3e}");
    assert!(rel < 3e-4, "q8 MoE jobs mixed up their inputs: {rel:.3e}");

    drop(model);
    std::fs::remove_dir_all(dir).ok();
}
