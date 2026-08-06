//! Qwen3-VL prompt encoder against the reference forward.
//!
//! `tools/mk_qwen3te_toy.py` builds the fixture from ComfyUI's own
//! `Llama2_` under `Qwen3VL_32BConfig`; point `CMF_QWEN3TE_TOY` at the
//! packed directory. Without it the test skips.

use cortiq_engine::qwen3te::Qwen3Encoder;
use std::path::PathBuf;
use std::sync::Arc;

#[test]
fn matches_the_reference_forward() {
    let Some(dir) = std::env::var_os("CMF_QWEN3TE_TOY") else {
        eprintln!("CMF_QWEN3TE_TOY unset — skipping");
        return;
    };
    let dir = PathBuf::from(dir);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let model = Arc::new(cortiq_core::CmfModel::open(dir.join("toy.cmf")).unwrap());
    let enc = Qwen3Encoder::from_cmf(&model).unwrap();
    let got = enc.encode(&ids);

    let want: Vec<f32> = std::fs::read(dir.join("hidden.bin"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got.len(), want.len());
    let mut mx = 0f32;
    let mut se = 0f64;
    let mut rs = 0f64;
    for (&g, &w) in got.iter().zip(&want) {
        mx = mx.max((g - w).abs());
        se += ((g - w) as f64).powi(2);
        rs += (w as f64).powi(2);
    }
    println!(
        "hidden: max {mx:.3e} rms {:.3e} over signal rms {:.3e}",
        (se / got.len() as f64).sqrt(),
        (rs / want.len() as f64).sqrt()
    );
    assert!(mx < 2e-4, "encoder diverges: max {mx:.3e}");
}
