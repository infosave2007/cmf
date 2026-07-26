//! Micro-benchmark of the batched q4t GEMM at the Lumina FFN shape.
//! Env-gated so CI never runs it:
//!
//!     CMF_BENCH=1 cargo test -p cortiq-engine --release --test gpu_q4t_bench -- --nocapture
#![cfg(target_os = "macos")]

use cortiq_core::quant::{GROUP_SIZE, f32_to_f16};
use cortiq_core::*;

#[test]
fn bench_q4t_matmat_ffn_shape() {
    if std::env::var("CMF_BENCH").is_err() {
        eprintln!("bench skipped: set CMF_BENCH=1");
        return;
    }
    unsafe { std::env::set_var("CMF_GPU", "1") };
    if !cortiq_engine::gpu_metal::enabled() {
        eprintln!("bench skipped: Metal disabled");
        return;
    }
    // Lumina FFN w1/w3 shape at 512px.
    let (rows, cols, b) = (9216usize, 2304usize, 1024usize);
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
        dtype: TensorDtype::Q4Tiled,
        shape: vec![rows, cols],
        data: payload,
    };
    let pad = TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0u8; 8192 * 8],
    };
    let dir = std::env::temp_dir().join(format!("cmf-q4tbench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();

    let x: Vec<f32> = (0..b * cols)
        .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let mut out = vec![0f32; b * rows];
    for _ in 0..3 {
        assert!(cortiq_engine::gpu::q4t_matmat(
            &model, idx, &x, b, rows, cols, &mut out
        ));
    }
    let n_iter = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        cortiq_engine::gpu::q4t_matmat(&model, idx, &x, b, rows, cols, &mut out);
    }
    let per = t0.elapsed().as_secs_f64() / n_iter as f64;
    let flops = 2.0 * b as f64 * rows as f64 * cols as f64;
    println!(
        "q4t GEMM {rows}x{cols} b={b}: {:.2} ms/op, {:.2} TFLOP/s",
        per * 1e3,
        flops / per / 1e12
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// VAE decode at 512px from the packed Lumina file (set CMF_VAE_PROF=1
/// for the per-stage split). Needs the user's model file; skips
/// without it.
#[test]
fn bench_vae_decode() {
    if std::env::var("CMF_BENCH").is_err() {
        eprintln!("bench skipped: set CMF_BENCH=1");
        return;
    }
    let path = std::env::var("CMF_LUMINA")
        .unwrap_or_else(|_| "/Users/oleg/Documents/cortiq-bot/lumina-q4t.cmf".into());
    let Ok(model) = CmfModel::open(&path) else {
        eprintln!("bench skipped: no model at {path}");
        return;
    };
    let vae = cortiq_engine::vae::VaeDecoder::from_cmf(&model).unwrap();
    // 64 → 512px out; CMF_VAE_BENCH_HW=128 → 1024px.
    let hw_lat: usize = std::env::var("CMF_VAE_BENCH_HW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let (h, w) = (hw_lat, hw_lat);
    let z: Vec<f32> = (0..vae.latent_channels * h * w)
        .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let t0 = std::time::Instant::now();
    let img = vae.decode(&z, h, w);
    println!(
        "vae decode {}px: {:.2}s ({} px out)",
        h * 8,
        t0.elapsed().as_secs_f64(),
        img.len() / 3
    );
}

#[test]
fn bench_dit_attention_shapes() {
    if std::env::var("CMF_BENCH").is_err() {
        eprintln!("bench skipped: set CMF_BENCH=1");
        return;
    }
    unsafe { std::env::set_var("CMF_GPU", "1") };
    if !cortiq_engine::gpu_metal::enabled() {
        eprintln!("bench skipped: Metal disabled");
        return;
    }
    // Lumina heads; n = joint sequence at 512px and 1024px.
    let (nh, nkv, hd) = (24usize, 8usize, 96usize);
    for n in [1064usize, 4136] {
        let mk = |seed: usize, len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 29 + seed * 31 + 7) % 83) as f32 / 83.0 - 0.5)
                .collect()
        };
        let qh = mk(1, nh * n * hd);
        let kh = mk(2, nkv * n * hd);
        let vh = mk(3, nkv * n * hd);
        let mut out = vec![0f32; n * nh * hd];
        let scale = 1.0 / (hd as f32).sqrt();
        for _ in 0..2 {
            assert!(cortiq_engine::gpu::dit_attention(
                &qh, &kh, &vh, nh, nkv, n, hd, scale, &mut out
            ));
        }
        let n_iter = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iter {
            cortiq_engine::gpu::dit_attention(&qh, &kh, &vh, nh, nkv, n, hd, scale, &mut out);
        }
        let per = t0.elapsed().as_secs_f64() / n_iter as f64;
        let flops = 4.0 * nh as f64 * (n as f64) * (n as f64) * hd as f64;
        println!(
            "dit attention n={n} nh={nh} hd={hd}: {:.2} ms/op, {:.2} TFLOP/s",
            per * 1e3,
            flops / per / 1e12
        );
    }
}
