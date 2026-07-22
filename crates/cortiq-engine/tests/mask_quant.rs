//! Mask × quantized mmap: the sparse-FFN-on-quant path
//! (`sparse_ffn_quant`) must agree with the same weights dequantized to
//! f32. This gates the q8_2f branches of row_dot / add_col_scaled — the
//! per-neuron reads that let a masked model run at quantized RSS instead
//! of the old whole-model-f32 blowup.

use cortiq_core::quant::f32_to_f16;
use cortiq_core::{
    CMF_VERSION, CmfHeader, CmfModel, LayerType, ModelArch, NormStyle, QuantType, TensorDtype,
    TensorSpec,
};
use cortiq_engine::pipeline::{DenseFfn, sparse_ffn_quant_for_test};
use cortiq_engine::qtensor::QTensor;
use std::sync::Arc;

/// Reference q8_2f encoder (matches the converter / spec §3.2):
/// col[i] = RMS over rows, row_scale[o] = max|w/col|/127, both f16.
/// Layout: [int8 : out·in][f16 row_scale : out][f16 col : in].
fn encode_q8_2f(w: &[f32], out: usize, inn: usize) -> Vec<u8> {
    let mut col = vec![0f32; inn];
    for i in 0..inn {
        let mut acc = 0f64;
        for o in 0..out {
            let v = w[o * inn + i] as f64;
            acc += v * v;
        }
        let c = (acc / out as f64).sqrt().max(1e-12) as f32;
        // Quantize against the f16-rounded field (decoder multiplies by it).
        col[i] = f32::from(half_round(c)).max(6.104e-5);
    }
    let mut q = vec![0i8; out * inn];
    let mut rs = vec![0f32; out];
    for o in 0..out {
        let mut amax = 1e-12f32;
        for i in 0..inn {
            let wn = w[o * inn + i] / col[i];
            amax = amax.max(wn.abs());
        }
        let s = f32::from(half_round(amax / 127.0)).max(6.104e-5);
        rs[o] = s;
        for i in 0..inn {
            let wn = w[o * inn + i] / col[i];
            q[o * inn + i] = (wn / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let mut bytes = Vec::with_capacity(out * inn + out * 2 + inn * 2);
    bytes.extend(q.iter().map(|&b| b as u8));
    for &s in &rs {
        bytes.extend_from_slice(&f32_to_f16(s).to_le_bytes());
    }
    for &c in &col {
        bytes.extend_from_slice(&f32_to_f16(c).to_le_bytes());
    }
    bytes
}

/// f32 → f16 → f32 round-trip (the fields are stored f16).
fn half_round(x: f32) -> HalfF32 {
    HalfF32(cortiq_core::quant::f16_to_f32(f32_to_f16(x)))
}
struct HalfF32(f32);
impl From<HalfF32> for f32 {
    fn from(h: HalfF32) -> f32 {
        h.0
    }
}

fn spec_q8(name: &str, out: usize, inn: usize, w: &[f32]) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::Q8_2f,
        shape: vec![out, inn],
        data: encode_q8_2f(w, out, inn),
    }
}

#[test]
fn sparse_ffn_quant_agrees_with_dequant() {
    let (hidden, inter) = (32usize, 96usize);
    let synth = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 41 + salt * 13 + 7) % 103) as f32 / 103.0 - 0.5) * 0.5)
            .collect()
    };
    let gate = synth(inter * hidden, 1);
    let up = synth(inter * hidden, 2);
    let down = synth(hidden * inter, 3);

    let arch = ModelArch {
        arch_name: "tiny".into(),
        hidden_size: hidden,
        intermediate_size: inter,
        num_layers: 1,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 8,
        vocab_size: 16,
        layer_types: vec![LayerType::FullAttention],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 1e4,
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: 8,
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
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
    };
    let tensors = vec![
        spec_q8("g", inter, hidden, &gate),
        spec_q8("u", inter, hidden, &up),
        spec_q8("d", hidden, inter, &down),
    ];
    let dir = std::env::temp_dir().join(format!("cmf-maskq-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &tensors, None, None).unwrap();
    let model = Arc::new(CmfModel::open(&path).unwrap());

    // Mapped (quantized, zero-copy) DenseFfn.
    let d_mapped = DenseFfn {
        act: cortiq_engine::pipeline::Act::Silu,
        gate_proj: QTensor::from_model(&model, "g").unwrap(),
        up_proj: QTensor::from_model(&model, "u").unwrap(),
        down_proj: QTensor::from_model(&model, "d").unwrap(),
    };
    assert!(
        d_mapped.down_proj.sparse_col_ok(),
        "q8_2f must allow col reads"
    );

    // Dequantized-to-f32 copy of the SAME weights.
    let deq = |t: &QTensor| {
        let (r, c) = (t.rows(), t.cols());
        let mut o = vec![0f32; r * c];
        for row in 0..r {
            t.row_f32(row, &mut o[row * c..(row + 1) * c]);
        }
        o
    };
    let d_f32 = DenseFfn {
        act: cortiq_engine::pipeline::Act::Silu,
        gate_proj: QTensor::from_f32(deq(&d_mapped.gate_proj), inter, hidden),
        up_proj: QTensor::from_f32(deq(&d_mapped.up_proj), inter, hidden),
        down_proj: QTensor::from_f32(deq(&d_mapped.down_proj), hidden, inter),
    };

    let x = synth(hidden, 9);
    let active: Vec<u16> = (0..inter as u16).filter(|i| i % 2 == 0).collect();
    let q = sparse_ffn_quant_for_test(&d_mapped, &x, &active, hidden);
    let f = sparse_ffn_quant_for_test(&d_f32, &x, &active, hidden);

    let max_d = q
        .iter()
        .zip(&f)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Same underlying values (q8 branch vs dequant-then-f32) → only float
    // accumulation-order noise separates them.
    assert!(
        max_d < 1e-4,
        "q8 sparse != dequant sparse: max|Δ| = {max_d}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
