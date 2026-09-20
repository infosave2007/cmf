//! Small native Metal q2tp affine operator gate.
//!
//! The production payload remains ordinary Q2TP bytes; the descriptor selects
//! the affine centre only at the operator boundary. This test deliberately
//! exercises both matrix tails and the single-token matvec.

#![cfg(all(feature = "gpu", target_os = "macos"))]
#![recursion_limit = "256"]

use cortiq_core::CmfModel;
use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::quant::{f32_to_f16, q2tp_sections, q4tp_code, q4tp_put_code};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::sync::Arc;

const GROUP: usize = 32;

fn payload(rows: usize, cols: usize) -> Vec<u8> {
    let gpr = cols / GROUP;
    let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
    let mut out = vec![0u8; codes_off + rows * stride];
    let mut seen_codes = [false; 3];
    let mut seen_rungs = [false; 32];
    for r in 0..rows {
        // Every ladder rung, including exact zero, gets a real row. Varying
        // nonzero f16 steps prevents the GEMM from hiding a decoder offset.
        let rung = r % 32;
        let row_step = 0.03125f32 + (r % 7) as f32 * 0.0078125f32;
        seen_rungs[rung] = true;
        for g in 0..gpr {
            let dst = &mut out[(r * gpr + g) * 8..(r * gpr + g + 1) * 8];
            // Affine Q2 reserves code 3. Keep the fixture on the legal
            // 0/1/2 alphabet while exercising zero, centre, and positive
            // codes without relying on a dequant helper.
            for (k, byte) in dst.iter_mut().enumerate() {
                let mut packed = 0u8;
                for j in 0..4 {
                    let code = ((r + g + 2 * k + j) % 3) as u8;
                    assert!(code <= 2);
                    seen_codes[code as usize] = true;
                    packed |= code << (2 * j);
                }
                *byte = packed;
            }
            q4tp_put_code(
                &mut out[codes_off + r * stride..codes_off + (r + 1) * stride],
                g,
                rung,
            );
        }
        // q2tp stores log2(scale), not log2(log2(scale)). The old fixture
        // evaluated log2(-3), producing NaNs that f32::max silently ignored.
        let lo = f32_to_f16(-3.0f32);
        out[params_off + r * 4..params_off + r * 4 + 2].copy_from_slice(&lo.to_le_bytes());
        let step = f32_to_f16(row_step);
        out[params_off + r * 4 + 2..params_off + r * 4 + 4]
            .copy_from_slice(&step.to_le_bytes());
    }
    assert!(seen_codes.into_iter().all(|seen| seen));
    assert!(seen_rungs.into_iter().all(|seen| seen));
    out
}

fn arch(cols: usize) -> ModelArch {
    let signs: Vec<f32> = (0..cols)
        .map(|i| if i & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    assert!(signs.iter().any(|&s| s < 0.0));
    assert!(signs.iter().any(|&s| s > 0.0));
    serde_json::from_value(serde_json::json!({
        "arch_name": "prism_hadamard_qwen35", "hidden_size": cols, "intermediate_size": 65,
        "num_layers": 1, "num_attention_heads": 1, "num_kv_heads": 1,
        "head_dim": 1, "vocab_size": 1, "layer_types": ["FullAttention"],
        "rms_norm_eps": 1e-6, "max_position_embeddings": 1,
        "linear_conv_kernel_dim": 0, "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
        "prism_hadamard": {
            "version": 1, "block_size": 32,
            "transform": "normalized-sylvester-walsh-hadamard",
            "axis": "input-last-dimension", "sign_mode": "explicit",
            "widths": [cols], "signs": signs,
            "forward_weight_names": ["w"], "inverse_weight_names": [],
            "gdn_v_grouped": true, "activation_f16": true,
            "affine": {"version": 1, "profile": "q2tp_affine",
                        "group_size": 32, "correction_scale": 0.5,
                        "target_names": ["w"]}
        }
    }))
    .unwrap()
}

fn model() -> (Arc<CmfModel>, usize, usize, usize) {
    let rows = 1025;
    let cols = 64;
    let model_arch = arch(cols);
    let arch_json = serde_json::to_value(&model_arch).unwrap();
    assert_eq!(arch_json["prism_hadamard"]["activation_f16"], true);
    assert_eq!(arch_json["prism_hadamard"]["sign_mode"], "explicit");
    let path = std::env::temp_dir().join(format!(
        "cmf-q2tp-metal-affine-{}-{}.cmf",
        std::process::id(),
        rows
    ));
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch: model_arch,
        quant_type: QuantType::Q4Block,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    CmfModel::write(
        &path,
        &header,
        &[
            TensorSpec {
                name: "w".into(),
                dtype: TensorDtype::Q2TiledP,
                shape: vec![rows, cols],
                data: payload(rows, cols),
            },
            // Keep the mapped primary section past the final partial page;
            // Metal's no-copy arena intentionally rounds its safe length down.
            TensorSpec {
                name: "padding".into(),
                dtype: TensorDtype::U8,
                shape: vec![20_000],
                data: vec![0u8; 20_000],
            },
        ],
        None,
        None,
    )
    .unwrap();
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensors.iter().position(|t| t.name == "w").unwrap();
    (model, idx, rows, cols)
}

fn input(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u32).wrapping_mul(747_796_405).wrapping_add(seed) >> 9) as f32 / 4096.0 - 4.0)
        .collect()
}

fn affine_reference(model: &CmfModel, idx: usize, xs: &[f32], b: usize, rows: usize, cols: usize) -> Vec<f32> {
    let entry = &model.tensors[idx];
    let bytes = model.entry_bytes(entry);
    let gpr = cols / GROUP;
    let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
    let mut out = vec![0.0; b * rows];
    for r in 0..rows {
        let lo = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([
            bytes[params_off + r * 4],
            bytes[params_off + r * 4 + 1],
        ]));
        let step = cortiq_core::quant::f16_to_f32(u16::from_le_bytes([
            bytes[params_off + r * 4 + 2],
            bytes[params_off + r * 4 + 3],
        ]));
        let codes = &bytes[codes_off + r * stride..codes_off + (r + 1) * stride];
        let rung = q4tp_code(codes, 0);
        let scale = if rung == 0 {
            0.0
        } else {
            2.0f32.powf(lo + (rung as f32 - 1.0) * step)
        };
        for bi in 0..b {
            let x = &xs[bi * cols..(bi + 1) * cols];
            let mut acc = 0.0;
            for g in 0..gpr {
                let chunk = &bytes[(r * gpr + g) * 8..(r * gpr + g + 1) * 8];
                for k in 0..GROUP {
                    let code = (chunk[k / 4] >> (2 * (k % 4))) & 3;
                    assert!(code <= 2, "reserved affine Q2 code {code}");
                    acc += ((code as f32) - 1.0) * scale * x[g * GROUP + k];
                }
            }
            out[bi * rows + r] = acc;
        }
    }
    out
}

#[test]
fn affine_matmat_and_matvec_match_q2_operator() {
    let (model, idx, rows, cols) = model();
    for b in [1usize, 2, 3, 7, 8, 9, 31, 32, 33] {
        let xs = input(b * cols, 17 + b as u32);
        let want = affine_reference(&model, idx, &xs, b, rows, cols);
        assert!(want.iter().all(|v| v.is_finite()));
        assert!(want.iter().map(|v| v.abs()).sum::<f32>() > 0.0);
        let mut got = vec![0.0; b * rows];
        assert!(
            cortiq_engine::gpu::q2tp_affine_matmat(&model, idx, &xs, b, rows, cols, &mut got),
            "native Metal q2 affine matmat declined for b={b}"
        );
        assert!(got.iter().all(|v| v.is_finite()));
        let rel = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / want.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1.0);
        println!("q2tp affine matmat rows={rows} cols={cols} b={b} rel_max={rel:.6e}");
        assert!(rel < 2e-3, "affine matmat relative max {rel:.3e} at b={b}");
    }

    let xs = input(cols, 17);
    let want = affine_reference(&model, idx, &xs, 1, rows, cols);
    let mut mv = vec![0.0; rows];
    assert!(cortiq_engine::gpu::q2tp_affine_matvec(
        &model, idx, &xs[..cols], rows, cols, &mut mv
    ));
    assert!(mv.iter().all(|v| v.is_finite()));
    let mv_rel = mv
        .iter()
        .zip(&want[..rows])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / want[..rows]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1.0);
    println!("q2tp affine matvec rows={rows} cols={cols} rel_max={mv_rel:.6e}");
    assert!(mv_rel < 2e-3, "affine matvec relative max {mv_rel:.3e}");
}
