//! The Metal q4tp kernel against the canonical scalar dequant.
//!
//! A wrong GPU kernel is the expensive kind of bug: the model still emits
//! fluent text, so nothing looks broken until someone measures quality. This
//! pins the kernel to `dequant_q4tp`, which is the format's definition.
#![cfg(target_os = "macos")]

use cortiq_core::format::{CMF_VERSION, CmfHeader, CmfModel, TensorSpec};
use cortiq_core::quant::{
    GROUP_SIZE, dequant_q4tp, f32_to_f16, q4tp_code_stride, q4tp_put_code, q4tp_sections,
};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};

/// Random nibbles plus a per-row ladder whose span varies row to row, so the
/// codes cover the full 0..31 range instead of clustering on one rung.
fn synth(rows: usize, cols: usize) -> Vec<u8> {
    let gpr = cols / GROUP_SIZE;
    let stride = q4tp_code_stride(gpr);
    let (params_off, codes_off, _) = q4tp_sections(rows, cols);
    let mut b = vec![0u8; codes_off + rows * stride];
    for r in 0..rows {
        for g in 0..gpr {
            let t = (r * gpr + g) * 16;
            for k in 0..16 {
                b[t + k] = ((r * 31 + g * 7 + k * 13) % 251) as u8;
            }
        }
        let p = params_off + r * 4;
        b[p..p + 2].copy_from_slice(&f32_to_f16(-6.0 - 0.03 * (r % 17) as f32).to_le_bytes());
        b[p + 2..p + 4]
            .copy_from_slice(&f32_to_f16(0.01 + 0.004 * (r % 11) as f32).to_le_bytes());
        let crow = &mut b[codes_off + r * stride..codes_off + (r + 1) * stride];
        for g in 0..gpr {
            q4tp_put_code(crow, g, (r * 5 + g * 3) % 32);
        }
    }
    b
}

fn tiny_model(rows: usize, cols: usize, payload: Vec<u8>) -> (std::sync::Arc<CmfModel>, usize) {
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "tiny",
        "hidden_size": cols,
        "intermediate_size": rows,
        "num_layers": 1,
        "num_attention_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "vocab_size": rows,
        "layer_types": ["FullAttention"],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 8,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))
    .unwrap();
    let header = CmfHeader {
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
    };
    let spec = TensorSpec {
        name: "w".into(),
        dtype: TensorDtype::Q4TiledP,
        shape: vec![rows, cols],
        data: payload,
    };
    // The Metal path maps the file and refuses tensors whose payload would
    // run past the mapped span, so leave slack behind the weight.
    let pad = TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0u8; 8192 * 8],
    };
    let dir = std::env::temp_dir().join(format!("cmf-q4tp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();
    (model, idx)
}

#[test]
fn metal_q4tp_matvec_matches_dequant_reference() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    if !cortiq_engine::gpu_metal::enabled() {
        eprintln!("skipped: Metal disabled");
        return;
    }
    let (rows, cols) = (512usize, 1024usize);
    let payload = synth(rows, cols);
    let mut w = vec![0f32; rows * cols];
    dequant_q4tp(&payload, rows, cols, &mut w);
    let (model, idx) = tiny_model(rows, cols, payload);

    let xs: Vec<f32> = (0..cols)
        .map(|i| ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5)
        .collect();
    let mut got = vec![0f32; rows];
    assert!(
        cortiq_engine::gpu_metal::q4tp_matvec_for_test(&model, idx, &xs, rows, cols, &mut got),
        "GPU refused a well-formed q4tp tensor"
    );

    for r in 0..rows {
        let want: f32 = (0..cols).map(|c| w[r * cols + c] * xs[c]).sum();
        // Tolerance against the summed magnitude, not the result: these dot
        // products cancel, and GPU/CPU differ only in summation order.
        let mag: f32 = (0..cols).map(|c| (w[r * cols + c] * xs[c]).abs()).sum();
        assert!(
            (got[r] - want).abs() <= 1e-5 * mag,
            "row {r}: GPU {} vs dequant {want}",
            got[r]
        );
    }
}
