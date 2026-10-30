//! q8_2f projections in the native Metal rows graph (speculative verify,
//! batched prefill). `cortiq compress` keeps the K/V projections of the
//! full-attention layers at q8_2f beside a q4tp trunk; the rows graph used
//! to refuse such a file outright. Its b-row q8_2f GEMM must reproduce the
//! whole-token graph's `q8f_matvec` bit for bit on every row, so a verified
//! draft can never change the greedy text.

#![cfg(target_os = "macos")]

use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use cortiq_core::{CMF_VERSION, CmfModel};
use std::sync::Arc;

// cols/4 = 272 crosses the 128-float4 unrolled loop and leaves a tail;
// rows leave a partial last threadgroup of 8 simdgroups.
const ROWS: usize = 1031;
const COLS: usize = 1088;

fn header() -> CmfHeader {
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "q8f-rows-metal-test",
        "hidden_size": COLS,
        "intermediate_size": ROWS,
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
        quant_type: QuantType::Q8_2f,
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

#[test]
fn q8_2f_rows_gemm_matches_the_token_matvec_bitwise() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    if !cortiq_engine::gpu_metal::enabled() {
        eprintln!(
            "skipped: {}",
            cortiq_engine::gpu_metal::initialization_error().unwrap_or("Metal disabled")
        );
        return;
    }
    // [rows×cols i8][rows f16 row scales][cols f16 input field]
    let mut data = Vec::with_capacity(ROWS * COLS + ROWS * 2 + COLS * 2);
    for r in 0..ROWS {
        for c in 0..COLS {
            data.push((((r * 13 + c * 7 + 5) % 255) as i32 - 127) as i8 as u8);
        }
    }
    let mut rs = Vec::with_capacity(ROWS);
    for r in 0..ROWS {
        let h = f32_to_f16(0.002 + (r % 7) as f32 * 0.0005);
        data.extend_from_slice(&h.to_le_bytes());
        rs.push(f16_to_f32(h));
    }
    let mut col = Vec::with_capacity(COLS);
    for c in 0..COLS {
        let h = f32_to_f16(0.5 + ((c * 3) % 17) as f32 / 16.0);
        data.extend_from_slice(&h.to_le_bytes());
        col.push(f16_to_f32(h));
    }
    let specs = vec![
        TensorSpec {
            name: "w".into(),
            dtype: TensorDtype::Q8_2f,
            shape: vec![ROWS, COLS],
            data,
        },
        // Keep the tensor away from the page-truncated end of the window.
        TensorSpec {
            name: "pad".into(),
            dtype: TensorDtype::F32,
            shape: vec![8192, 2],
            data: vec![0; 8192 * 8],
        },
    ];
    let dir = std::env::temp_dir().join(format!("cmf-metal-q8-rows-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("q8f.cmf");
    CmfModel::write(&path, &header(), &specs, None, None).unwrap();
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();

    // Verify-sized and prefill-sized batches.
    for b in [1usize, 3, 8, 13] {
        let xs: Vec<f32> = (0..b * COLS)
            .map(|i| ((i * 37 + b) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let mut rows_out = vec![0f32; b * ROWS];
        assert!(cortiq_engine::gpu_metal::q8_2f_matmat(
            &model,
            idx,
            &rs,
            &col,
            &xs,
            b,
            ROWS,
            COLS,
            &mut rows_out,
        ));
        for bi in 0..b {
            let mut one = vec![0f32; ROWS];
            assert!(cortiq_engine::gpu_metal::q8_2f_matvec(
                &model,
                idx,
                &rs,
                &col,
                &xs[bi * COLS..(bi + 1) * COLS],
                ROWS,
                COLS,
                &mut one,
            ));
            for r in 0..ROWS {
                assert_eq!(
                    rows_out[bi * ROWS + r].to_bits(),
                    one[r].to_bits(),
                    "b={b} row {bi} out {r}: {} vs matvec {}",
                    rows_out[bi * ROWS + r],
                    one[r]
                );
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
