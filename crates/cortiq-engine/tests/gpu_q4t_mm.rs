//! Metal batched q4t GEMM (dequant pass + f32nt simdgroup GEMM) vs an
//! exact f64 dequant reference. Half shared-memory tiles inside the
//! GEMM make it tolerance-class — the bound here is loose but honest.
//!
//!     cargo test -p cortiq-engine --release --test gpu_q4t_mm -- --nocapture
#![cfg(target_os = "macos")]

use cortiq_core::quant::{GROUP_SIZE, f16_to_f32, f32_to_f16};
use cortiq_core::*;

fn require_metal() -> bool {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    if cortiq_engine::gpu_metal::enabled() {
        true
    } else {
        eprintln!(
            "gpu q4t mm test skipped: {}",
            cortiq_engine::gpu_metal::initialization_error().unwrap_or("Metal disabled")
        );
        false
    }
}

#[test]
fn gpu_q4t_matmat_matches_dequant_reference() {
    if !require_metal() {
        return;
    }
    let (rows, cols, b) = (512usize, 1024usize, 64usize);
    let gpr = cols / GROUP_SIZE;
    const TILE: usize = 18;
    let mut payload = vec![0u8; rows * gpr * TILE];
    for t in 0..rows * gpr {
        let sc = 0.02 + 0.0005 * (t % 64) as f32;
        payload[t * TILE..t * TILE + 2].copy_from_slice(&f32_to_f16(sc).to_le_bytes());
        for k in 0..16 {
            payload[t * TILE + 2 + k] = ((t * 31 + k * 13) % 251) as u8;
        }
    }
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "tiny",
        "hidden_size": cols,
        "intermediate_size": cols * 2,
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
        dtype: TensorDtype::Q4Tiled,
        shape: vec![rows, cols],
        data: payload.clone(),
    };
    // Tail pad: the no-copy file window rounds DOWN to a page.
    let pad = TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0u8; 8192 * 8],
    };
    let dir = std::env::temp_dir().join(format!("cmf-q4tmm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();

    let x: Vec<f32> = (0..b * cols)
        .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let mut got = vec![0f32; b * rows];
    assert!(
        cortiq_engine::gpu::q4t_matmat(&model, idx, &x, b, rows, cols, &mut got),
        "gpu q4t_matmat refused"
    );

    // Exact reference: f64 matmul over the dequantized tiles.
    let mut max_rel = 0f64;
    for bi in 0..b {
        for r in 0..rows {
            let mut want = 0f64;
            for g in 0..gpr {
                let t = (r * gpr + g) * TILE;
                let s = f16_to_f32(u16::from_le_bytes([payload[t], payload[t + 1]])) as f64;
                for (k, &bb) in payload[t + 2..t + TILE].iter().enumerate() {
                    let w0 = ((bb & 0x0F) as f64 - 8.0) * s;
                    let w1 = (((bb >> 4) & 0x0F) as f64 - 8.0) * s;
                    want += w0 * x[bi * cols + g * GROUP_SIZE + 2 * k] as f64;
                    want += w1 * x[bi * cols + g * GROUP_SIZE + 2 * k + 1] as f64;
                }
            }
            let d = (got[bi * rows + r] as f64 - want).abs();
            max_rel = max_rel.max(d / want.abs().max(1.0));
        }
    }
    println!("gpu q4t GEMM max rel dev {max_rel:.2e}");
    assert!(max_rel < 2e-2, "gpu q4t GEMM diverged: {max_rel}");
    let _ = std::fs::remove_dir_all(&dir);
}
