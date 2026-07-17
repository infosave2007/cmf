//! Metal q1 micro-decomposition (manual): fixed submit cost vs kernel
//! time. Run: cargo test -p cortiq-engine --features gpu --test gpu_micro -- --nocapture
#[cfg(target_os = "macos")]
#[test]
fn q1_gpu_micro() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    use cortiq_core::quant::{f32_to_f16, GROUP_SIZE};
    use cortiq_core::*;
    let (rows, cols) = (17408usize, 5120usize);
    let gpr = cols / GROUP_SIZE;
    let mut payload = vec![0u8; rows * gpr * 6];
    for t in 0..rows * gpr {
        payload[t * 6..t * 6 + 2].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
        payload[t * 6 + 2] = (t % 251) as u8;
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
        format: "cmf".into(), version: CMF_VERSION, arch, quant_type: QuantType::Vbit,
        provenance: None, tokenizer_config: None, section_hashes: None,
        skills: Vec::new(), shard: None, calibration: None,
    };
    let spec = TensorSpec { name: "w".into(), dtype: TensorDtype::Q1, shape: vec![rows, cols], data: payload };
    let pad = TensorSpec { name: "pad".into(), dtype: TensorDtype::F32, shape: vec![8192, 2], data: vec![0u8; 8192 * 8] };
    let dir = std::env::temp_dir().join(format!("cmf-q1micro-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();
    let x = vec![0.1f32; cols];
    let mut y = vec![0f32; rows];
    // warm
    for _ in 0..3 {
        assert!(cortiq_engine::gpu_q1_matvec_for_test(&model, idx, &x, rows, cols, &mut y));
    }
    let t0 = std::time::Instant::now();
    let n = 20;
    for _ in 0..n {
        cortiq_engine::gpu_q1_matvec_for_test(&model, idx, &x, rows, cols, &mut y);
    }
    let per = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let mb = (rows * gpr * 6) as f64 / 1e6;
    println!("q1 single matvec {rows}x{cols} ({mb:.1} MB): {per:.3} ms/op → {:.1} GB/s", mb / per);
    std::fs::remove_dir_all(&dir).ok();
}

/// q8_row twin of `q1_gpu_micro` — the whole-token q8 graph question
/// is whether this kernel's warm GB/s clears the CPU decode rate.
#[cfg(target_os = "macos")]
#[test]
fn q8_gpu_micro() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    use cortiq_core::quant::f32_to_f16;
    use cortiq_core::*;
    let (rows, cols) = (17408usize, 5120usize);
    let mut payload = vec![0u8; rows * cols + rows * 2];
    for (i, b) in payload[..rows * cols].iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    for r in 0..rows {
        let s = f32_to_f16(0.01).to_le_bytes();
        payload[rows * cols + r * 2..rows * cols + r * 2 + 2].copy_from_slice(&s);
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
    let spec =
        TensorSpec { name: "w".into(), dtype: TensorDtype::Q8Row, shape: vec![rows, cols], data: payload };
    let pad = TensorSpec { name: "pad".into(), dtype: TensorDtype::F32, shape: vec![8192, 2], data: vec![0u8; 8192 * 8] };
    let dir = std::env::temp_dir().join(format!("cmf-q8micro-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();
    let rs = vec![0.01f32; rows];
    let x = vec![0.1f32; cols];
    let mut y = vec![0f32; rows];
    for _ in 0..3 {
        assert!(cortiq_engine::gpu_metal::q8_matvec(&model, idx, &rs, &x, rows, cols, &mut y));
    }
    let t0 = std::time::Instant::now();
    let n = 20;
    for _ in 0..n {
        cortiq_engine::gpu_metal::q8_matvec(&model, idx, &rs, &x, rows, cols, &mut y);
    }
    let per = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let mb = (rows * cols) as f64 / 1e6;
    println!("q8 single matvec {rows}x{cols} ({mb:.1} MB): {per:.3} ms/op → {:.1} GB/s", mb / per);
    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(target_os = "macos")]
#[test]
fn empty_submit_cost() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    let s = cortiq_engine::gpu_empty_submit_for_test(50);
    println!("empty submit+wait: {:.3} ms/op", s * 1000.0 / 50.0);
}

#[cfg(target_os = "macos")]
#[test]
fn pipelined_submit_cost() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    let s = cortiq_engine::gpu_pipelined_submit_for_test(50);
    println!("pipelined empty submit: {:.3} ms/op", s * 1000.0 / 50.0);
}

/// Direct parity of the metal q1 CHAIN (moe_block) and q1 matvec_batch
/// against a dequantized f32 reference — the paths a small single-op
/// parity test never exercises.
#[cfg(target_os = "macos")]
#[test]
fn q1_chain_and_batch_parity() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    use cortiq_core::quant::{f32_to_f16, f16_to_f32, GROUP_SIZE};
    use cortiq_core::*;
    let (hidden, inter) = (256usize, 512usize);
    // Deterministic binary weights ±s.
    let mk = |rows: usize, cols: usize, seed: usize| -> (Vec<u8>, Vec<f32>) {
        let gpr = cols / GROUP_SIZE;
        let mut payload = Vec::with_capacity(rows * gpr * 6);
        let mut w = vec![0f32; rows * cols];
        for o in 0..rows {
            for g in 0..gpr {
                let s = 0.01 + ((o * 3 + g + seed) % 7) as f32 * 0.004;
                let s = f16_to_f32(f32_to_f16(s));
                payload.extend_from_slice(&f32_to_f16(s).to_le_bytes());
                for j in 0..4 {
                    let mut byte = 0u8;
                    for k in 0..8 {
                        let i = g * GROUP_SIZE + j * 8 + k;
                        let bit = ((o * 11 + i * 7 + seed) % 3) != 0;
                        if bit { byte |= 1 << k; }
                        w[o * cols + i] = if bit { s } else { -s };
                    }
                    payload.push(byte);
                }
            }
        }
        (payload, w)
    };
    let (gp, gw) = mk(inter, hidden, 1);
    let (up_, uw) = mk(inter, hidden, 2);
    let (dp, dw) = mk(hidden, inter, 3);
    let arch = ModelArch {
        arch_name: "tiny".into(), hidden_size: hidden, intermediate_size: inter,
        num_layers: 1, num_attention_heads: 2, num_kv_heads: 1, head_dim: 4,
        vocab_size: 64, layer_types: vec![LayerType::FullAttention],
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
        format: "cmf".into(), version: CMF_VERSION, arch, quant_type: QuantType::Vbit,
        provenance: None, tokenizer_config: None, section_hashes: None,
        skills: Vec::new(), shard: None, calibration: None,
    };
    let specs = vec![
        TensorSpec { name: "g".into(), dtype: TensorDtype::Q1, shape: vec![inter, hidden], data: gp },
        TensorSpec { name: "u".into(), dtype: TensorDtype::Q1, shape: vec![inter, hidden], data: up_ },
        TensorSpec { name: "d".into(), dtype: TensorDtype::Q1, shape: vec![hidden, inter], data: dp },
        TensorSpec { name: "pad".into(), dtype: TensorDtype::F32, shape: vec![8192, 2], data: vec![0u8; 8192 * 8] },
    ];
    let dir = std::env::temp_dir().join(format!("cmf-q1chain-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &specs, None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let (gi, ui, di) = (
        model.tensor_index("g").unwrap(),
        model.tensor_index("u").unwrap(),
        model.tensor_index("d").unwrap(),
    );
    let x: Vec<f32> = (0..hidden).map(|i| ((i * 19 + 3) % 83) as f32 / 83.0 - 0.5).collect();

    // f32 reference chain.
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mv = |w: &Vec<f32>, rows: usize, cols: usize, x: &[f32]| -> Vec<f32> {
        (0..rows).map(|o| (0..cols).map(|i| w[o * cols + i] * x[i]).sum()).collect()
    };
    let g = mv(&gw, inter, hidden, &x);
    let u = mv(&uw, inter, hidden, &x);
    let act: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a) * b).collect();
    let want = mv(&dw, hidden, inter, &act);

    // GPU chain.
    let job = cortiq_engine::gpu_moe_job_for_test(gi, ui, di, inter, hidden, x.clone());
    let mut got = vec![0f32; hidden];
    assert!(cortiq_engine::gpu_moe_block_for_test(&model, job, &mut got), "moe_block refused");
    let mut max_d = 0f32;
    for i in 0..hidden {
        max_d = max_d.max((want[i] - got[i]).abs());
    }
    println!("q1 chain max|Δ| = {max_d}");
    assert!(max_d < 1e-3, "q1 chain diverged: {max_d}");

    // GPU batch: three independent matvecs of one input.
    let want_g = mv(&gw, inter, hidden, &x);
    let mut o1 = vec![0f32; inter];
    let mut o2 = vec![0f32; inter];
    let mut o3 = vec![0f32; hidden];
    let xi: Vec<f32> = (0..inter).map(|i| ((i * 7 + 1) % 61) as f32 / 61.0 - 0.5).collect();
    let want_d = mv(&dw, hidden, inter, &xi);
    assert!(
        cortiq_engine::gpu_batch_q1_for_test(&model, &[(gi, inter, hidden), (ui, inter, hidden), (di, hidden, inter)], &x, &xi, &mut [o1.as_mut_slice(), o2.as_mut_slice(), o3.as_mut_slice()]),
        "matvec_batch refused"
    );
    let mut bd = 0f32;
    for i in 0..inter { bd = bd.max((want_g[i] - o1[i]).abs()); }
    for i in 0..hidden { bd = bd.max((want_d[i] - o3[i]).abs()); }
    println!("q1 batch max|Δ| = {bd}");
    assert!(bd < 1e-3, "q1 batch diverged: {bd}");
    std::fs::remove_dir_all(&dir).ok();
}
