//! Paired in-process A/B of the blocked 2×4 x86 prefill GEMM against
//! the per-row path — same tensor, same activations, alternating env
//! toggle, min-of timing (shared-vCPU hosts jitter ±60% across
//! processes; a paired micro inside one process still ranks the two).
#![cfg(target_arch = "x86_64")]

use cortiq_core::quant::f32_to_f16;
use cortiq_core::*;

#[test]
fn blocked_vs_per_row() {
    let (rows, cols, b) = (4864usize, 896usize, 256usize);
    let mut payload = vec![0u8; rows * cols + rows * 2];
    for (i, byte) in payload[..rows * cols].iter_mut().enumerate() {
        *byte = ((i * 37 + 11) % 251) as u8;
    }
    for r in 0..rows {
        let sc = f32_to_f16(0.01).to_le_bytes();
        payload[rows * cols + r * 2..rows * cols + r * 2 + 2].copy_from_slice(&sc);
    }
    let arch = ModelArch {
        arch_name: "tiny".into(), hidden_size: cols, intermediate_size: cols * 2,
        num_layers: 1, num_attention_heads: 2, num_kv_heads: 1, head_dim: 4,
        vocab_size: rows, layer_types: vec![LayerType::FullAttention],
        rms_norm_eps: 1e-6, norm_style: NormStyle::Qwen, rope_theta: 1e4,
        tie_word_embeddings: false, partial_rotary_factor: 1.0, mtp: None,
        moe: None, linear_core: None, max_position_embeddings: 8,
        linear_conv_kernel_dim: None, linear_num_key_heads: None,
        linear_num_value_heads: None, linear_key_head_dim: None, linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: None,
        sliding_window_pattern: None,
        rope_local_base_freq: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: None,
        attn_v_norm: false,
    };
    let header = CmfHeader {
        format: "cmf".into(), version: CMF_VERSION, arch, quant_type: QuantType::Q8Row,
        provenance: None, tokenizer_config: None, section_hashes: None,
        skills: Vec::new(), shard: None, calibration: None,
    };
    let spec = TensorSpec {
        name: "w".into(), dtype: TensorDtype::Q8Row, shape: vec![rows, cols], data: payload,
    };
    let dir = std::env::temp_dir().join(format!("cmf-x86mm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();
    let rs = vec![0.01f32; rows];
    let x: Vec<f32> = (0..b * cols).map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5).collect();
    let qt = cortiq_engine::qtensor::QTensor::Mapped {
        model: model.clone(), idx, dtype: TensorDtype::Q8Row, rows, cols,
        row_scale: rs, col_field: Vec::new(), vbit_offsets: Vec::new(),
        repack: Vec::new(),
    };
    let mut y_a = vec![0f32; b * rows];
    let mut y_b = vec![0f32; b * rows];

    // Parity first.
    unsafe { std::env::set_var("CMF_X86_BLOCKED", "1") };
    qt.matmat(&x, b, &mut y_a, None);
    unsafe { std::env::set_var("CMF_X86_BLOCKED", "0") };
    qt.matmat(&x, b, &mut y_b, None);
    let max_d = y_a.iter().zip(&y_b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    assert!(max_d < 1e-3, "blocked ≠ per-row: max|Δ| = {max_d}");

    // Paired timing, min-of interleaved rounds.
    let (mut t_blk, mut t_row) = (f64::MAX, f64::MAX);
    for _ in 0..6 {
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "1") };
        let t0 = std::time::Instant::now();
        qt.matmat(&x, b, &mut y_a, None);
        t_blk = t_blk.min(t0.elapsed().as_secs_f64() * 1000.0);
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "0") };
        let t1 = std::time::Instant::now();
        qt.matmat(&x, b, &mut y_b, None);
        t_row = t_row.min(t1.elapsed().as_secs_f64() * 1000.0);
    }
    let gflop = 2.0 * (b * rows * cols) as f64 / 1e9;
    println!(
        "x86 q8 matmat {rows}x{cols} b={b}: blocked {t_blk:.1} ms ({:.0} GF/s) | per-row {t_row:.1} ms ({:.0} GF/s)",
        gflop / t_blk * 1e3, gflop / t_row * 1e3
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Paired A/B for the q4_block leg (the current q4 default): the row is
/// unpacked once already; blocking removes the per-activation reload
/// and reduce rounds.
#[test]
fn q4b_blocked_vs_per_row() {
    let (rows, cols, b) = (4864usize, 896usize, 256usize);
    let gpr = cols / 32;
    let groups = rows * gpr;
    // Split layout: packed nibbles first (16B per group), then f16 scales.
    let mut payload = vec![0u8; groups * 16 + groups * 2];
    for g in 0..groups {
        for k in 0..16 {
            payload[g * 16 + k] = ((g * 13 + k * 29 + 5) % 251) as u8;
        }
        payload[groups * 16 + g * 2..groups * 16 + g * 2 + 2]
            .copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
    }
    let arch = ModelArch {
        arch_name: "tiny".into(), hidden_size: cols, intermediate_size: cols * 2,
        num_layers: 1, num_attention_heads: 2, num_kv_heads: 1, head_dim: 4,
        vocab_size: rows, layer_types: vec![LayerType::FullAttention],
        rms_norm_eps: 1e-6, norm_style: NormStyle::Qwen, rope_theta: 1e4,
        tie_word_embeddings: false, partial_rotary_factor: 1.0, mtp: None,
        moe: None, linear_core: None, max_position_embeddings: 8,
        linear_conv_kernel_dim: None, linear_num_key_heads: None,
        linear_num_value_heads: None, linear_key_head_dim: None, linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: None,
        sliding_window_pattern: None,
        rope_local_base_freq: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: None,
        attn_v_norm: false,
    };
    let header = CmfHeader {
        format: "cmf".into(), version: CMF_VERSION, arch, quant_type: QuantType::Q4Block,
        provenance: None, tokenizer_config: None, section_hashes: None,
        skills: Vec::new(), shard: None, calibration: None,
    };
    let spec = TensorSpec {
        name: "w".into(), dtype: TensorDtype::Q4Block, shape: vec![rows, cols], data: payload,
    };
    let dir = std::env::temp_dir().join(format!("cmf-x86q4b-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();
    let x: Vec<f32> = (0..b * cols).map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5).collect();
    let qt = cortiq_engine::qtensor::QTensor::Mapped {
        model: model.clone(), idx, dtype: TensorDtype::Q4Block, rows, cols,
        row_scale: Vec::new(), col_field: Vec::new(), vbit_offsets: Vec::new(),
        repack: Vec::new(),
    };
    let mut y_a = vec![0f32; b * rows];
    let mut y_b = vec![0f32; b * rows];
    unsafe { std::env::set_var("CMF_X86_BLOCKED", "1") };
    qt.matmat(&x, b, &mut y_a, None);
    unsafe { std::env::set_var("CMF_X86_BLOCKED", "0") };
    qt.matmat(&x, b, &mut y_b, None);
    let max_d = y_a.iter().zip(&y_b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    assert!(max_d < 1e-3, "q4b blocked ≠ per-row: max|Δ| = {max_d}");
    let (mut t_blk, mut t_row) = (f64::MAX, f64::MAX);
    for _ in 0..6 {
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "1") };
        let t0 = std::time::Instant::now();
        qt.matmat(&x, b, &mut y_a, None);
        t_blk = t_blk.min(t0.elapsed().as_secs_f64() * 1000.0);
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "0") };
        let t1 = std::time::Instant::now();
        qt.matmat(&x, b, &mut y_b, None);
        t_row = t_row.min(t1.elapsed().as_secs_f64() * 1000.0);
    }
    let gflop = 2.0 * (b * rows * cols) as f64 / 1e9;
    println!(
        "x86 q4b matmat {rows}x{cols} b={b}: blocked {t_blk:.1} ms ({:.0} GF/s) | per-row {t_row:.1} ms ({:.0} GF/s)",
        gflop / t_blk * 1e3, gflop / t_row * 1e3
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Same paired A/B for the q4_tiled leg (unpack reuse across four
/// activation streams).
#[test]
fn q4t_blocked_vs_per_row() {
    let (rows, cols, b) = (4864usize, 896usize, 256usize);
    let gpr = cols / 32;
    // 18B tiles: [f16 scale][16B nibbles].
    let mut payload = vec![0u8; rows * gpr * 18];
    for r in 0..rows {
        for g in 0..gpr {
            let t = (r * gpr + g) * 18;
            payload[t..t + 2].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
            for k in 0..16 {
                payload[t + 2 + k] = ((r * 31 + g * 7 + k * 13) % 251) as u8;
            }
        }
    }
    let arch = ModelArch {
        arch_name: "tiny".into(), hidden_size: cols, intermediate_size: cols * 2,
        num_layers: 1, num_attention_heads: 2, num_kv_heads: 1, head_dim: 4,
        vocab_size: rows, layer_types: vec![LayerType::FullAttention],
        rms_norm_eps: 1e-6, norm_style: NormStyle::Qwen, rope_theta: 1e4,
        tie_word_embeddings: false, partial_rotary_factor: 1.0, mtp: None,
        moe: None, linear_core: None, max_position_embeddings: 8,
        linear_conv_kernel_dim: None, linear_num_key_heads: None,
        linear_num_value_heads: None, linear_key_head_dim: None, linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: None,
        sliding_window_pattern: None,
        rope_local_base_freq: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: None,
        attn_v_norm: false,
    };
    let header = CmfHeader {
        format: "cmf".into(), version: CMF_VERSION, arch, quant_type: QuantType::Q4Block,
        provenance: None, tokenizer_config: None, section_hashes: None,
        skills: Vec::new(), shard: None, calibration: None,
    };
    let spec = TensorSpec {
        name: "w".into(), dtype: TensorDtype::Q4Tiled, shape: vec![rows, cols], data: payload,
    };
    let dir = std::env::temp_dir().join(format!("cmf-x86q4t-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();
    let x: Vec<f32> = (0..b * cols).map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5).collect();
    let qt = cortiq_engine::qtensor::QTensor::Mapped {
        model: model.clone(), idx, dtype: TensorDtype::Q4Tiled, rows, cols,
        row_scale: Vec::new(), col_field: Vec::new(), vbit_offsets: Vec::new(),
        repack: Vec::new(),
    };
    let mut y_a = vec![0f32; b * rows];
    let mut y_b = vec![0f32; b * rows];
    unsafe { std::env::set_var("CMF_X86_BLOCKED", "1") };
    qt.matmat(&x, b, &mut y_a, None);
    unsafe { std::env::set_var("CMF_X86_BLOCKED", "0") };
    qt.matmat(&x, b, &mut y_b, None);
    let max_d = y_a.iter().zip(&y_b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    assert!(max_d < 1e-3, "q4t blocked ≠ per-row: max|Δ| = {max_d}");
    let (mut t_blk, mut t_row) = (f64::MAX, f64::MAX);
    for _ in 0..6 {
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "1") };
        let t0 = std::time::Instant::now();
        qt.matmat(&x, b, &mut y_a, None);
        t_blk = t_blk.min(t0.elapsed().as_secs_f64() * 1000.0);
        unsafe { std::env::set_var("CMF_X86_BLOCKED", "0") };
        let t1 = std::time::Instant::now();
        qt.matmat(&x, b, &mut y_b, None);
        t_row = t_row.min(t1.elapsed().as_secs_f64() * 1000.0);
    }
    let gflop = 2.0 * (b * rows * cols) as f64 / 1e9;
    println!(
        "x86 q4t matmat {rows}x{cols} b={b}: blocked {t_blk:.1} ms ({:.0} GF/s) | per-row {t_row:.1} ms ({:.0} GF/s)",
        gflop / t_blk * 1e3, gflop / t_row * 1e3
    );
    std::fs::remove_dir_all(&dir).ok();
}
