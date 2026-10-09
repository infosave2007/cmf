//! EmbeddingGemma 2 text parity against the float32 transformers /
//! sentence-transformers reference.
//!
//! Needs two env vars (the test passes trivially without them):
//! * `CORTIQ_EGEMMA2_MODEL` — a `.cmf` from `cortiq convert --model
//!   <google/embeddinggemma-2 dir>`;
//! * `CORTIQ_EGEMMA2_REF` — the reference dir: `manifest.json` (cases with
//!   `modality`, `st_call {input, prompt_name}`, `input_ids`, `row`) and
//!   `embeddings.npy` (`[cases, 768]` float32).
//!
//! The cosine floor is 0.99999 for an exact (bf16 / f32) file and 0.995 for
//! a quantized one; `CORTIQ_EGEMMA2_MIN_COS` overrides it.
//!
//! `cargo test --release -p cortiq-engine --test egemma2_parity -- --nocapture`

use cortiq_core::CmfModel;
use cortiq_engine::egemma2::{EmbeddingGemma2, MATRYOSHKA_DIMS, TextInput, cosine, matryoshka};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A float32 C-order .npy as (shape, data).
fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "not a .npy");
    let (hl, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[off..off + hl]).unwrap();
    assert!(header.contains("'<f4'"), "expected float32: {header}");
    assert!(header.contains("'fortran_order': False"));
    let shape_s = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape: Vec<usize> = shape_s
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = b[off + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

#[test]
fn egemma2_text_parity() {
    let (Some(model), Some(refdir)) = (
        std::env::var_os("CORTIQ_EGEMMA2_MODEL").map(PathBuf::from),
        std::env::var_os("CORTIQ_EGEMMA2_REF").map(PathBuf::from),
    ) else {
        eprintln!("egemma2_text_parity: set CORTIQ_EGEMMA2_MODEL and CORTIQ_EGEMMA2_REF to run");
        return;
    };
    let cmf = Arc::new(CmfModel::open(&model).expect("open model"));
    let enc = EmbeddingGemma2::load(&cmf, cortiq_engine::pool::Pool::from_env()).expect("load");
    let floor: f64 = std::env::var("CORTIQ_EGEMMA2_MIN_COS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if matches!(enc.quant.as_str(), "bf16" | "f32") {
            0.99999
        } else {
            0.995
        });
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(refdir.join("manifest.json")).unwrap()).unwrap();
    let (shape, refs) = read_npy_f32(&refdir.join("embeddings.npy"));
    let d = shape[1];
    let cases: Vec<&serde_json::Value> = manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["modality"] == "text")
        .collect();
    assert!(!cases.is_empty(), "no text cases in the manifest");

    let inputs: Vec<TextInput> = cases
        .iter()
        .map(|c| TextInput {
            text: c["st_call"]["input"].as_str().unwrap().to_string(),
            prompt_name: c["st_call"]["prompt_name"].as_str().map(str::to_string),
            ..Default::default()
        })
        .collect();
    // tokenization is exact
    for (c, inp) in cases.iter().zip(&inputs) {
        let want: Vec<u32> = c["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            enc.input_ids(inp).unwrap(),
            want,
            "token ids of {}",
            c["id"]
        );
    }
    // all cases in one packed batch
    let got = enc.embed_texts(&inputs).unwrap();
    let mut worst = (1.0f64, String::new());
    for (c, g) in cases.iter().zip(&got) {
        let row = c["row"].as_u64().unwrap() as usize;
        let r = &refs[row * d..(row + 1) * d];
        for dim in MATRYOSHKA_DIMS {
            let a = matryoshka(g, dim).unwrap();
            let cs = cosine(&a, &r[..dim]);
            if cs < worst.0 {
                worst = (cs, format!("{} @{dim}", c["id"]));
            }
            assert!(
                cs >= floor,
                "{} dim {dim}: cosine {cs:.7} < {floor}",
                c["id"]
            );
        }
        eprintln!("{} cos {:.7}", c["id"], cosine(g, r));
    }
    eprintln!("worst {:.7} ({}), floor {floor}", worst.0, worst.1);
    // a member of a batch embeds as it does alone
    let alone = enc.embed_texts(&inputs[..1]).unwrap();
    let cs = cosine(&alone[0], &got[0]);
    assert!(cs > 0.999_999, "batch vs single: {cs}");
}
