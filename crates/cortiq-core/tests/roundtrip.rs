//! Stage-1 tests: envelope roundtrip, strict validation, hash64
//! cross-vectors, canonical quant layouts, bitwise mask diff.

use cortiq_core::format::{build_sparse_index, decode_sparse_index, encode_sparse_index};
use cortiq_core::mask::zero_tail_bits;
use cortiq_core::quant::{GROUP_SIZE, dequant_q4_block, dequant_q8_row, f16_to_f32, f32_to_f16};
use cortiq_core::{
    CMF_VERSION, CmfError, CmfHeader, CmfModel, LayerType, MaskCatalog, MaskPriority, ModelArch,
    NormStyle, Quality, QuantType, TaskMask, TensorDtype, TensorSpec, hash64,
};

// ───────────────────────── helpers ─────────────────────────

fn tiny_arch() -> ModelArch {
    ModelArch {
        arch_name: "tiny-test".into(),
        hidden_size: 8,
        intermediate_size: 16,
        num_layers: 2,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: 10,
        layer_types: vec![LayerType::FullAttention; 2],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: 64,
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
        num_loops: 1,
        loop_final_norm: false,
    }
}

fn tiny_header() -> CmfHeader {
    CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        arch: tiny_arch(),
        quant_type: QuantType::F32,
        provenance: None,
    }
}

/// Reference q8_row encoder (canonical layout: quants then row scales).
fn encode_q8_row(vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert_eq!(vals.len(), out_dim * in_dim);
    let mut q = Vec::with_capacity(out_dim * in_dim);
    let mut scales = Vec::with_capacity(out_dim * 2);
    for o in 0..out_dim {
        let row = &vals[o * in_dim..(o + 1) * in_dim];
        let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if absmax == 0.0 { 1e-10 } else { absmax / 127.0 };
        for &v in row {
            q.push((v / scale).round().clamp(-128.0, 127.0) as i8 as u8);
        }
        scales.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
    }
    q.extend_from_slice(&scales);
    q
}

/// Reference q4_block encoder.
fn encode_q4_block(vals: &[f32]) -> Vec<u8> {
    let n_groups = vals.len().div_ceil(GROUP_SIZE);
    let mut padded = vals.to_vec();
    padded.resize(n_groups * GROUP_SIZE, 0.0);
    let mut packed = Vec::with_capacity(n_groups * 16);
    let mut scales = Vec::with_capacity(n_groups * 2);
    for g in 0..n_groups {
        let group = &padded[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let absmax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if absmax == 0.0 { 1e-10 } else { absmax / 7.0 };
        for k in 0..16 {
            let q0 = ((group[k * 2] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q1 = ((group[k * 2 + 1] / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            packed.push((q0 & 0x0F) | (q1 << 4));
        }
        scales.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
    }
    packed.extend_from_slice(&scales);
    packed
}

fn mask_with_bits(
    task_id: u32,
    name: &str,
    arch: &ModelArch,
    quality: Option<Quality>,
) -> TaskMask {
    let ffn_b = arch.ffn_mask_bytes();
    let head_b = arch.head_mask_bytes();
    // Layer 0: every even neuron active; layer 1: first half active.
    let mut ffn0 = vec![0b0101_0101u8; ffn_b];
    zero_tail_bits(&mut ffn0, arch.intermediate_size);
    let mut ffn1 = vec![0u8; ffn_b];
    for i in 0..arch.intermediate_size / 2 {
        ffn1[i / 8] |= 1 << (i % 8);
    }
    let mut head = vec![0xFFu8; head_b];
    zero_tail_bits(&mut head, arch.num_attention_heads);
    TaskMask {
        task_id,
        name: name.into(),
        description: Some("test".into()),
        sparsity: 0.5,
        quality,
        ffn_masks: vec![ffn0, ffn1],
        head_masks: vec![head.clone(), head],
        layer_gates: vec![true, true],
        parent: None,
        has_hot_pack: false,
        priority: if task_id == 0 {
            MaskPriority::Fallback
        } else {
            MaskPriority::Normal
        },
    }
}

fn write_tiny_file(path: &std::path::Path) -> (Vec<TensorSpec>, MaskCatalog, Vec<u8>) {
    let arch = tiny_arch();
    let embed: Vec<f32> = (0..arch.vocab_size * arch.hidden_size)
        .map(|i| (i as f32 * 0.13).sin())
        .collect();
    let gate: Vec<f32> = (0..arch.intermediate_size * arch.hidden_size)
        .map(|i| (i as f32 * 0.07).cos())
        .collect();
    let norm: Vec<f32> = (0..arch.hidden_size)
        .map(|i| 1.0 + i as f32 * 0.01)
        .collect();

    let tensors = vec![
        TensorSpec {
            name: "model.embed_tokens.weight".into(),
            dtype: TensorDtype::F32,
            shape: vec![arch.vocab_size, arch.hidden_size],
            data: embed.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        TensorSpec {
            name: "model.layers.0.mlp.gate_proj.weight".into(),
            dtype: TensorDtype::Q8Row,
            shape: vec![arch.intermediate_size, arch.hidden_size],
            data: encode_q8_row(&gate, arch.intermediate_size, arch.hidden_size),
        },
        TensorSpec {
            name: "model.layers.1.mlp.gate_proj.weight".into(),
            dtype: TensorDtype::Q4Block,
            shape: vec![arch.intermediate_size, arch.hidden_size],
            data: encode_q4_block(&gate),
        },
        TensorSpec {
            name: "model.norm.weight".into(),
            dtype: TensorDtype::F16,
            shape: vec![arch.hidden_size],
            data: norm
                .iter()
                .flat_map(|v| f32_to_f16(*v).to_le_bytes())
                .collect(),
        },
    ];

    let quality = Quality {
        metric: "heldout_ppl_ratio".into(),
        value: 0.97,
        baseline_dense: Some(6.1),
        n_samples: Some(128),
        dataset_sha256: None,
    };
    let catalog = MaskCatalog {
        masks: vec![
            mask_with_bits(0, "general", &tiny_arch(), Some(quality)),
            mask_with_bits(1, "coding", &tiny_arch(), None),
        ],
        default_task: "general".into(),
    };
    let vocab = br#"{"model":{"type":"BPE","vocab":{},"merges":[]}}"#.to_vec();

    CmfModel::write(path, &tiny_header(), &tensors, Some(&catalog), Some(&vocab)).unwrap();
    (tensors, catalog, vocab)
}

// ───────────────────────── hash64 ─────────────────────────

/// Vectors generated by `vmfcore.hash64` (Python/numpy) — bit-for-bit
/// cross-format compatibility contract.
#[test]
fn hash64_matches_vmfcore_reference() {
    assert_eq!(hash64(b""), 0x0);
    assert_eq!(hash64(b"CMF"), 0xb59aa39a074033ec);
    assert_eq!(hash64(b"0123456789abcdef"), 0x30d401310062ea9f);
    let all: Vec<u8> = (0..=255u8).collect();
    assert_eq!(hash64(&all), 0x9955bb876c0706bf);
}

// ───────────────────────── f16 / quant ─────────────────────────

#[test]
fn f16_roundtrip() {
    for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0, 1e-4, -std::f32::consts::PI] {
        let back = f16_to_f32(f32_to_f16(v));
        assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
    }
}

#[test]
fn q8_row_dequant_accuracy() {
    let vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.31).sin() * 3.0).collect();
    let bytes = encode_q8_row(&vals, 4, 16);
    let mut out = vec![0f32; 64];
    dequant_q8_row(&bytes, 4, 16, &mut out);
    for (a, b) in vals.iter().zip(&out) {
        assert!((a - b).abs() < 3.0 / 127.0 * 1.5, "{a} vs {b}");
    }
}

#[test]
fn q4_block_dequant_accuracy() {
    // 40 elements: one full group + one padded group.
    let vals: Vec<f32> = (0..40).map(|i| (i as f32 * 0.17).cos() * 2.0).collect();
    let bytes = encode_q4_block(&vals);
    let mut out = vec![0f32; 40];
    dequant_q4_block(&bytes, &mut out);
    for (a, b) in vals.iter().zip(&out) {
        assert!((a - b).abs() < 2.0 / 7.0 * 0.75, "{a} vs {b}");
    }
}

// ───────────────────────── file roundtrip ─────────────────────────

#[test]
fn full_file_roundtrip() {
    let dir = tempdir();
    let path = dir.join("tiny.cmf");
    let (tensors, catalog, vocab) = write_tiny_file(&path);

    let model = CmfModel::open(&path).unwrap();

    // Header + arch
    assert_eq!(model.header.version, CMF_VERSION);
    assert_eq!(model.arch().arch_name, "tiny-test");
    assert_eq!(model.arch().norm_style, NormStyle::Qwen);

    // Tensors: names, dtypes, shapes, bytes identical.
    assert_eq!(model.tensors.len(), tensors.len());
    for spec in &tensors {
        let e = model.tensor(&spec.name).expect(&spec.name);
        assert_eq!(e.dtype, spec.dtype);
        assert_eq!(e.shape, spec.shape);
        assert_eq!(model.tensor_bytes(&spec.name).unwrap(), &spec.data[..]);
        assert_eq!(e.off % 64, 0, "tensor offset must be 64-aligned");
    }

    // Hash verification: intact file has zero problems.
    assert!(model.verify().is_empty());

    // Masks: bit-identical, quality contract round-trips.
    assert_eq!(model.masks.masks.len(), 2);
    assert_eq!(model.masks.default_task, "general");
    for (orig, got) in catalog.masks.iter().zip(&model.masks.masks) {
        assert_eq!(orig.name, got.name);
        assert_eq!(orig.ffn_masks, got.ffn_masks);
        assert_eq!(orig.head_masks, got.head_masks);
        assert_eq!(orig.layer_gates, got.layer_gates);
    }
    let q = model.masks.masks[0].quality.as_ref().unwrap();
    assert_eq!(q.metric, "heldout_ppl_ratio");
    assert!(
        model.masks.masks[1].quality.is_none(),
        "unmeasured stays None"
    );

    // Vocab embedded verbatim.
    assert_eq!(model.vocab.as_deref(), Some(&vocab[..]));

    // Sparse index was built and decodes.
    assert!(!model.sparse_index.is_empty());
    let e0 = &model.sparse_index[0];
    assert_eq!(e0.task_id, 0);
    // Layer 0 of "general": even neurons active in all 32-neuron... here
    // intermediate=16 → 1 group, active.
    assert_eq!(e0.active_ffn_groups, vec![0u16]);
    assert_eq!(e0.active_heads, vec![0u8, 1]);

    // total_param_count counts matrices only.
    let expect = (10 * 8 + 16 * 8 + 16 * 8) as u64;
    assert_eq!(model.total_param_count(), expect);
}

/// Large tensors get page-aligned so cold skill/MoE/mask weights don't share a
/// page with the backbone; small tensors stay 64-aligned (no padding bloat).
/// The file must still round-trip and verify, and — since 4096 % 64 == 0 —
/// remain readable by the existing (64-aligned-only) contract.
#[test]
fn large_tensors_are_page_aligned() {
    use cortiq_core::format::{LARGE_TENSOR_ALIGN, LARGE_TENSOR_MIN};
    let dir = tempdir();
    let path = dir.join("align.cmf");
    let (rows, cols) = (512usize, 64usize); // q8_row = 512*64 + 512*2 = 33 792 B > 16 KB
    let vals: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.01).sin()).collect();
    let norm: Vec<f32> = (0..8).map(|i| 1.0 + i as f32).collect();
    let tensors = vec![
        TensorSpec {
            name: "big.weight".into(),
            dtype: TensorDtype::Q8Row,
            shape: vec![rows, cols],
            data: encode_q8_row(&vals, rows, cols),
        },
        TensorSpec {
            name: "small.norm".into(),
            dtype: TensorDtype::F16,
            shape: vec![8],
            data: norm
                .iter()
                .flat_map(|v| f32_to_f16(*v).to_le_bytes())
                .collect(),
        },
    ];
    CmfModel::write(&path, &tiny_header(), &tensors, None, None).unwrap();

    let model = CmfModel::open(&path).unwrap();
    let big = model.tensor("big.weight").unwrap();
    assert!(big.nbytes >= LARGE_TENSOR_MIN);
    assert_eq!(
        big.off % LARGE_TENSOR_ALIGN,
        0,
        "large tensor must be page-aligned (off = {})",
        big.off
    );
    // Every tensor still satisfies the legacy 64-alignment contract.
    for e in &model.tensors {
        assert_eq!(e.off % 64, 0);
    }
    assert_eq!(
        model.tensor_bytes("big.weight").unwrap(),
        &tensors[0].data[..]
    );
    assert!(model.verify().is_empty());
}

// ───────────────────────── strict validation ─────────────────────────

#[test]
fn open_rejects_garbage_and_corruption() {
    let dir = tempdir();

    // Not a CMF file at all.
    let garbage = dir.join("garbage.bin");
    std::fs::write(&garbage, b"not a cmf, definitely not a 27B model").unwrap();
    assert!(matches!(
        CmfModel::open(&garbage),
        Err(CmfError::InvalidMagic) | Err(CmfError::Bounds(_))
    ));

    // Too small.
    let small = dir.join("small.bin");
    std::fs::write(&small, b"CMF\x01").unwrap();
    assert!(matches!(CmfModel::open(&small), Err(CmfError::Bounds(_))));

    // Metadata tamper: flip one header byte inside a JSON string value
    // (keeps JSON valid) — section hashes must catch it (spec §8.1).
    let path = dir.join("tamper.cmf");
    write_tiny_file(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    let hpos = bytes
        .windows(5)
        .position(|w| w == b"\"cmf\"")
        .expect("format field in header JSON");
    bytes[hpos + 2] = b'x'; // "cmf" → "cxf"
    std::fs::write(&path, &bytes).unwrap();
    let tampered = CmfModel::open(&path).unwrap(); // opens (JSON valid)…
    let problems = tampered.verify();
    assert!(
        problems.iter().any(|p| p.contains("section 'header'")),
        "tampered header must fail verify, got: {problems:?}"
    );

    // Valid file, then corrupt version.
    let path = dir.join("v.cmf");
    write_tiny_file(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[4] = 1; // version = 1
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        CmfModel::open(&path),
        Err(CmfError::UnsupportedVersion(1))
    ));

    // Unknown required feature bit.
    let path = dir.join("f.cmf");
    write_tiny_file(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[12] |= 1 << 7;
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        CmfModel::open(&path),
        Err(CmfError::UnsupportedFeature(f)) if f == 1 << 7
    ));

    // A hostile directory count must return an error, never overflow/panic.
    let path = dir.join("dir-overflow.cmf");
    write_tiny_file(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    let dir_off = u64::from_le_bytes(bytes[0x20..0x28].try_into().unwrap()) as usize;
    bytes[dir_off..dir_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(CmfModel::open(&path), Err(CmfError::Parse(_))));

    // Truncated data section.
    let path = dir.join("t.cmf");
    write_tiny_file(&path);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    assert!(matches!(CmfModel::open(&path), Err(CmfError::Bounds(_))));

    // Flipped weight byte → verify() reports the tensor.
    let path = dir.join("h.cmf");
    write_tiny_file(&path);
    let model_ok = CmfModel::open(&path).unwrap();
    assert!(model_ok.verify().is_empty());
    let embed_off = model_ok.tensor("model.embed_tokens.weight").unwrap().off;
    drop(model_ok);
    let mut bytes = std::fs::read(&path).unwrap();
    // data_off is at envelope [0x30..0x38]
    let data_off = u64::from_le_bytes(bytes[0x30..0x38].try_into().unwrap());
    let idx = (data_off + embed_off) as usize;
    bytes[idx] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();
    let model_bad = CmfModel::open(&path).unwrap();
    let problems = model_bad.verify();
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("embed_tokens"));
}

// ───────────────────────── masks ─────────────────────────

#[test]
fn mask_diff_is_bitwise() {
    let arch = tiny_arch();
    let a = mask_with_bits(0, "a", &arch, None);
    let mut b = a.clone();
    // Same number of active neurons in layer 0, but different bits:
    // shift the even-pattern to odd.
    for byte in &mut b.ffn_masks[0] {
        *byte = 0b1010_1010;
    }
    zero_tail_bits(&mut b.ffn_masks[0], arch.intermediate_size);

    let diff = a.diff(&b);
    assert!(
        diff.changed_layers.contains(&0),
        "equal counts must still diff"
    );
    assert_eq!(diff.neurons_added, 8);
    assert_eq!(diff.neurons_removed, 8);
    assert!(diff.ffn_delta[0].iter().any(|&x| x != 0));

    // Identical masks → empty diff.
    let none = a.diff(&a.clone());
    assert!(none.changed_layers.is_empty());
    assert_eq!(none.neurons_added + none.neurons_removed, 0);
}

#[test]
fn tail_bits_are_zeroed() {
    let mut bits = vec![0xFFu8; 3]; // 24 bits for 18 real
    zero_tail_bits(&mut bits, 18);
    assert_eq!(bits, vec![0xFF, 0xFF, 0b0000_0011]);
}

#[test]
fn sparse_index_encode_decode() {
    let arch = tiny_arch();
    let catalog = MaskCatalog {
        masks: vec![mask_with_bits(0, "general", &arch, None)],
        default_task: "general".into(),
    };
    let idx = build_sparse_index(&catalog, &arch);
    let bytes = encode_sparse_index(&idx);
    let back = decode_sparse_index(&bytes).unwrap();
    assert_eq!(idx, back);
}

// ───────────────────────── defrag (spec §11) ─────────────────────────

/// A physically-defragged file (spec §11, Patent 2 claims 9/10): each
/// layer keeps its OWN reduced FFN neuron count and the masks section is
/// absent. The directory is the size authority, so open() must accept
/// per-layer-different FFN shapes that diverge from the nominal arch
/// scalar — that is exactly what makes pruned neurons "not stored".
#[test]
fn defrag_per_layer_ffn_shapes_roundtrip() {
    let dir = tempdir();
    let path = dir.join("defrag.cmf");
    let arch = tiny_arch(); // hidden=8, nominal intermediate=16
    let hidden = arch.hidden_size;

    // Full FFN triple per layer, q8_row (row-wise → any inter' is legal).
    // Layer 0 keeps 12 neurons, layer 1 keeps 8 — both below the nominal
    // 16 and different from each other.
    let mk_triple = |li: usize, inter: usize| -> Vec<TensorSpec> {
        let gate: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i + li) as f32 * 0.07).cos())
            .collect();
        let up: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i + li) as f32 * 0.05).sin())
            .collect();
        let down: Vec<f32> = (0..hidden * inter)
            .map(|i| ((i + li) as f32 * 0.03).cos())
            .collect();
        vec![
            TensorSpec {
                name: format!("model.layers.{li}.mlp.gate_proj.weight"),
                dtype: TensorDtype::Q8Row,
                shape: vec![inter, hidden],
                data: encode_q8_row(&gate, inter, hidden),
            },
            TensorSpec {
                name: format!("model.layers.{li}.mlp.up_proj.weight"),
                dtype: TensorDtype::Q8Row,
                shape: vec![inter, hidden],
                data: encode_q8_row(&up, inter, hidden),
            },
            TensorSpec {
                name: format!("model.layers.{li}.mlp.down_proj.weight"),
                dtype: TensorDtype::Q8Row,
                shape: vec![hidden, inter],
                data: encode_q8_row(&down, hidden, inter),
            },
        ]
    };
    let mut tensors = mk_triple(0, 12);
    tensors.extend(mk_triple(1, 8));

    // No masks (None) — identity after physical pruning.
    CmfModel::write(&path, &tiny_header(), &tensors, None, None).unwrap();

    let model = CmfModel::open(&path).unwrap();
    assert!(
        model.masks.masks.is_empty(),
        "defragged file carries no masks"
    );

    let g0 = model.tensor("model.layers.0.mlp.gate_proj.weight").unwrap();
    let u0 = model.tensor("model.layers.0.mlp.up_proj.weight").unwrap();
    let d0 = model.tensor("model.layers.0.mlp.down_proj.weight").unwrap();
    assert_eq!(g0.shape, vec![12, hidden]);
    assert_eq!(u0.shape, vec![12, hidden]);
    assert_eq!(d0.shape, vec![hidden, 12], "down cols == inter'");

    let g1 = model.tensor("model.layers.1.mlp.gate_proj.weight").unwrap();
    assert_eq!(
        g1.shape,
        vec![8, hidden],
        "layer 1 keeps its OWN smaller size"
    );

    // The nominal arch scalar is untouched (per-layer truth is in shapes).
    assert!(model.arch().intermediate_size >= 12);
    assert!(model.verify().is_empty(), "per-tensor hashes intact");
}

// ───────────────────────── util ─────────────────────────

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cmf-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
