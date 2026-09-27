//! The DECISION feature bit (`required_features` bit 11, 0x800): writers
//! derive it from the `cortiq-decision-` arch prefix, readers refuse a
//! file whose bit and prefix disagree, and a language-model file is
//! written byte-for-byte as it was before the bit existed.

use cortiq_core::format::{self, CmfStreamWriter, features};
use cortiq_core::{CmfError, CmfHeader, CmfModel, TensorDtype, TensorSpec, hash64};
use std::path::{Path, PathBuf};

/// Offset of the `required_features` u32 in the envelope.
const FEATURES_OFFSET: usize = 12;

fn header(arch_name: &str, hidden_size: usize, num_layers: usize) -> CmfHeader {
    serde_json::from_value(serde_json::json!({
        "format": "cmf",
        "version": 2,
        "quant_type": "F32",
        "arch": {
            "arch_name": arch_name,
            "hidden_size": hidden_size,
            "intermediate_size": if num_layers == 0 { 0 } else { 16 },
            "num_layers": num_layers,
            "num_attention_heads": if num_layers == 0 { 0 } else { 2 },
            "num_kv_heads": if num_layers == 0 { 0 } else { 1 },
            "head_dim": if num_layers == 0 { 0 } else { 4 },
            "vocab_size": if num_layers == 0 { 0 } else { 10 },
            "layer_types": vec!["FullAttention"; num_layers],
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 64
        }
    }))
    .expect("toy header")
}

fn decision_header() -> CmfHeader {
    header("cortiq-decision-ph-v1", 4480, 0)
}

fn llm_header() -> CmfHeader {
    header("tiny-test", 8, 2)
}

/// A scratch directory unique to this process and test.
fn scratch(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("cmf-decision-feature-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The LLM toy: an F32 matrix, an F16 vector and a U8 blob, plus a vocab
/// section — the plain write path of a language-model container.
fn llm_tensors() -> Vec<TensorSpec> {
    let w: Vec<f32> = (0..8 * 16).map(|i| (i as f32 - 64.0) / 32.0).collect();
    let norm: Vec<u8> = (0..8u16)
        .flat_map(|i| (0x3c00u16 + i).to_le_bytes())
        .collect();
    vec![
        TensorSpec {
            name: "model.layers.0.mlp.up_proj.weight".into(),
            dtype: TensorDtype::F32,
            shape: vec![16, 8],
            data: f32_bytes(&w),
        },
        TensorSpec {
            name: "model.norm.weight".into(),
            dtype: TensorDtype::F16,
            shape: vec![8],
            data: norm,
        },
        TensorSpec {
            name: "aux.blob".into(),
            dtype: TensorDtype::U8,
            shape: vec![13],
            data: (0..13u8).map(|i| i.wrapping_mul(29)).collect(),
        },
    ]
}

const LLM_VOCAB: &[u8] = br#"{"model":{"type":"BPE","vocab":{},"merges":[]}}"#;

fn write_llm_toy(path: &Path) {
    CmfModel::write(path, &llm_header(), &llm_tensors(), None, Some(LLM_VOCAB)).unwrap();
}

fn write_llm_toy_streamed(path: &Path) {
    let gap = CmfStreamWriter::head_reserve_for(3, 40);
    let mut w = CmfStreamWriter::new(path, gap).unwrap();
    for t in llm_tensors() {
        w.push(&t.name, t.dtype, &t.shape, &t.data).unwrap();
    }
    w.finish(&llm_header(), None, Some(LLM_VOCAB)).unwrap();
}

fn decision_tensors() -> Vec<TensorSpec> {
    vec![
        TensorSpec {
            name: "decision.manifest".into(),
            dtype: TensorDtype::U8,
            shape: vec![2],
            data: b"{}".to_vec(),
        },
        TensorSpec {
            name: "decision.skill.toy.task.0.mean".into(),
            dtype: TensorDtype::F32,
            shape: vec![4],
            data: f32_bytes(&[0.5, -0.25, 1.0, 0.0]),
        },
    ]
}

fn write_decision_toy(path: &Path, arch_name: &str) {
    CmfModel::write(
        path,
        &header(arch_name, 4480, 0),
        &decision_tensors(),
        None,
        None,
    )
    .unwrap();
}

fn envelope_features(path: &Path) -> u32 {
    let bytes = std::fs::read(path).unwrap();
    u32::from_le_bytes(
        bytes[FEATURES_OFFSET..FEATURES_OFFSET + 4]
            .try_into()
            .unwrap(),
    )
}

/// Rewrite the envelope's `required_features` in place. The envelope's
/// integrity hashes cover the header and directory, not this word, so the
/// patched file differs from a valid one ONLY in the bit under test.
fn patch_features(path: &Path, f: impl Fn(u32) -> u32) {
    let mut bytes = std::fs::read(path).unwrap();
    let old = u32::from_le_bytes(
        bytes[FEATURES_OFFSET..FEATURES_OFFSET + 4]
            .try_into()
            .unwrap(),
    );
    bytes[FEATURES_OFFSET..FEATURES_OFFSET + 4].copy_from_slice(&f(old).to_le_bytes());
    std::fs::write(path, bytes).unwrap();
}

fn expect_disagreement(result: Result<CmfModel, CmfError>) {
    match result {
        Ok(_) => panic!("a file whose DECISION bit and arch prefix disagree was opened"),
        Err(CmfError::Parse(msg)) => assert!(
            msg.contains("decision profile and DECISION feature bit disagree"),
            "unexpected refusal: {msg}"
        ),
        Err(e) => panic!("wrong refusal kind: {e}"),
    }
}

/// FNV-1a 64 — a second, independent digest next to the format's own
/// `hash64`, so the byte-identity golden does not rest on one function.
fn fnv1a64(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn decision_bit_is_bit_eleven_and_supported() {
    assert_eq!(features::DECISION, 0x800);
    assert_ne!(features::SUPPORTED & features::DECISION, 0);
    // Bits 7-10 keep their v0.7.7 meaning; DECISION overlaps none of them.
    assert_eq!(features::PRISM_HADAMARD, 1 << 7);
    assert_eq!(features::PRISM_AFFINE, 1 << 8);
    assert_eq!(features::DECISION & ((1 << 11) - 1), 0);
    assert_eq!(format::DECISION_ARCH_PREFIX, "cortiq-decision-");
    assert!(format::is_decision_profile("cortiq-decision-ph-v1"));
    assert!(format::is_decision_profile("cortiq-decision-overlay-v1"));
    assert!(!format::is_decision_profile("cortiq-decision"));
    assert!(!format::is_decision_profile("xcortiq-decision-ph-v1"));
    assert!(!format::is_decision_profile("Cortiq-Decision-ph-v1"));
    assert!(!format::is_decision_profile("qwen3"));
}

#[test]
fn writer_sets_decision_bit_from_arch_prefix() {
    let dir = scratch("write");
    for arch in ["cortiq-decision-ph-v1", "cortiq-decision-overlay-v1"] {
        let path = dir.join(format!("{arch}.cmf"));
        write_decision_toy(&path, arch);
        let raw = envelope_features(&path);
        assert_eq!(raw & 0x800, 0x800, "{arch}: envelope {raw:#x}");
        assert_eq!(raw, features::TENSOR_DIR | features::DECISION, "{arch}");
        let m = CmfModel::open(&path).unwrap();
        assert_eq!(
            m.required_features,
            features::TENSOR_DIR | features::DECISION
        );
        assert_eq!(m.header.arch.arch_name, arch);
        assert!(m.verify().is_empty());
        assert_eq!(m.tensor_bytes("decision.manifest").unwrap(), b"{}");
        drop(m);
        // `open_sharded` of an unsharded file is a plain open.
        let m = CmfModel::open_sharded(&path).unwrap();
        assert_ne!(m.required_features & features::DECISION, 0);
    }
    // A name that only resembles the prefix stays an ordinary file.
    for arch in ["cortiq-decision", "my-cortiq-decision-ph-v1", "tiny-test"] {
        let path = dir.join(format!("{arch}.cmf"));
        write_decision_toy(&path, arch);
        assert_eq!(envelope_features(&path) & features::DECISION, 0, "{arch}");
        let m = CmfModel::open(&path).unwrap();
        assert_eq!(m.required_features & features::DECISION, 0, "{arch}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn streaming_writer_sets_decision_bit_from_arch_prefix() {
    let dir = scratch("stream");
    let path = dir.join("streamed.cmf");
    let gap = CmfStreamWriter::head_reserve_for(2, 32);
    let mut w = CmfStreamWriter::new(&path, gap).unwrap();
    for t in decision_tensors() {
        w.push(&t.name, t.dtype, &t.shape, &t.data).unwrap();
    }
    w.finish(&decision_header(), None, None).unwrap();
    assert_eq!(
        envelope_features(&path),
        features::TENSOR_DIR | features::DECISION
    );
    let m = CmfModel::open(&path).unwrap();
    assert_ne!(m.required_features & features::DECISION, 0);
    assert!(m.verify().is_empty());
    drop(m);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn open_refuses_bit_without_prefix() {
    let dir = scratch("bit-no-prefix");
    let path = dir.join("llm.cmf");
    write_llm_toy(&path);
    // The unpatched file is valid and carries no DECISION bit.
    let m = CmfModel::open(&path).unwrap();
    assert_eq!(m.required_features & features::DECISION, 0);
    drop(m);
    patch_features(&path, |f| f | features::DECISION);
    expect_disagreement(CmfModel::open(&path));
    expect_disagreement(CmfModel::open_sharded(&path));

    // The same for a file that only resembles the prefix.
    let near = dir.join("near.cmf");
    write_decision_toy(&near, "cortiq-decision");
    CmfModel::open(&near).unwrap();
    patch_features(&near, |f| f | features::DECISION);
    expect_disagreement(CmfModel::open(&near));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn open_refuses_prefix_without_bit() {
    let dir = scratch("prefix-no-bit");
    for arch in ["cortiq-decision-ph-v1", "cortiq-decision-overlay-v1"] {
        let path = dir.join(format!("{arch}.cmf"));
        write_decision_toy(&path, arch);
        CmfModel::open(&path).unwrap();
        patch_features(&path, |f| f & !features::DECISION);
        assert_eq!(envelope_features(&path) & features::DECISION, 0);
        expect_disagreement(CmfModel::open(&path));
        expect_disagreement(CmfModel::open_sharded(&path));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// Byte identity of LLM files across the change. The goldens were taken
/// from the v0.7.7 writer (commit 2f9ea52d) BEFORE the DECISION bit was
/// added; the same toy written by this build must reproduce them exactly.
#[test]
fn llm_toy_bytes_are_unchanged_by_the_decision_bit() {
    // (len, hash64, fnv1a64) of the v0.7.7 output.
    const PLAIN: (usize, u64, u64) = (4732, 0x7dac_2100_a42c_3fad, 0xe4e0_73de_a061_0fb0);
    const STREAMED: (usize, u64, u64) = (3_150_460, 0xf2cb_83d6_e670_c4a1, 0xd1b2_0c75_0c59_2f30);

    let dir = scratch("llm-bytes");
    let plain = dir.join("plain.cmf");
    write_llm_toy(&plain);
    let streamed = dir.join("streamed.cmf");
    write_llm_toy_streamed(&streamed);

    for (path, golden) in [(&plain, PLAIN), (&streamed, STREAMED)] {
        let bytes = std::fs::read(path).unwrap();
        let got = (bytes.len(), hash64(&bytes), fnv1a64(&bytes));
        assert_eq!(
            got,
            golden,
            "{}: got ({}, {:#018x}, {:#018x})",
            path.display(),
            got.0,
            got.1,
            got.2
        );
        assert_eq!(envelope_features(path), features::TENSOR_DIR);
        let m = CmfModel::open(path).unwrap();
        assert_eq!(m.required_features, features::TENSOR_DIR);
        assert!(m.verify().is_empty());
    }
    std::fs::remove_dir_all(dir).unwrap();
}
