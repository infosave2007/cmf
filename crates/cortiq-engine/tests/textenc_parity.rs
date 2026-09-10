//! Gemma-2 encoder parity against the numpy reference on REAL weights.
//!
//!     python3 python/gemma_ref.py <text_encoder_dir> <out_dir>
//!     CMF_GEMMA_DIR=... CMF_GEMMA_REF=... \
//!         cargo test -p cortiq-engine --release --test textenc_parity -- --nocapture
use std::path::Path;

#[test]
fn gemma_encoder_matches_numpy_reference() {
    let (Ok(dir), Ok(refs)) = (
        std::env::var("CMF_GEMMA_DIR"),
        std::env::var("CMF_GEMMA_REF"),
    ) else {
        eprintln!("CMF_GEMMA_DIR/CMF_GEMMA_REF not set — skipping");
        return;
    };
    let enc = cortiq_engine::textenc::GemmaEncoder::load_dir(Path::new(&dir)).expect("load");
    let ids: Vec<u32> = std::fs::read(format!("{refs}/gemma_ids.bin"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let want: Vec<f32> = std::fs::read(format!("{refs}/gemma_ref.bin"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let t0 = std::time::Instant::now();
    let (got, _) = enc.encode(&ids, false);
    let warm = std::time::Instant::now();
    let _ = enc.encode(&ids, false);
    eprintln!("warm encode: {:.2}s", warm.elapsed().as_secs_f64());
    assert_eq!(got.len(), want.len());
    let mut max_d = 0f32;
    let mut max_rel = 0f32;
    for (a, b) in got.iter().zip(&want) {
        max_d = max_d.max((a - b).abs());
        max_rel = max_rel.max((a - b).abs() / b.abs().max(1.0));
    }
    println!(
        "gemma parity: {} tokens in {:.2}s, max|Δ| = {max_d:.2e} (rel {max_rel:.2e})",
        ids.len(),
        t0.elapsed().as_secs_f64()
    );
    assert!(max_rel < 1e-3, "Gemma encoder ≠ numpy: max|Δ| = {max_d}");
}
