//! Next-DiT parity against the numpy reference on REAL weights.
//!
//!     python3 python/nextdit_ref.py <transformer_dir> <out_dir>
//!     CMF_DIT_DIR=<transformer_dir> CMF_DIT_REF=<out_dir> \
//!         cargo test -p cortiq-engine --release --test dit_parity -- --nocapture
//!
//! Skips silently when the env vars are absent (CI has no weights).
use std::path::Path;

// Must match python/nextdit_ref.py.
const C: usize = 16;
const H: usize = 16;
const W: usize = 16;
const CAP_N: usize = 8;
const T: f32 = 0.7;

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn nextdit_matches_numpy_reference() {
    let (Ok(dir), Ok(refs)) = (std::env::var("CMF_DIT_DIR"), std::env::var("CMF_DIT_REF")) else {
        eprintln!("CMF_DIT_DIR/CMF_DIT_REF not set — skipping");
        return;
    };
    let dit = cortiq_engine::dit::NextDit::load_dir(Path::new(&dir)).expect("load DiT");
    let latent = read_f32(&format!("{refs}/dit_latent.bin"));
    let cap = read_f32(&format!("{refs}/dit_cap.bin"));
    let want = read_f32(&format!("{refs}/dit_out.bin"));
    assert_eq!(latent.len(), C * H * W);
    assert_eq!(want.len(), C * H * W);
    let t0 = std::time::Instant::now();
    let got = dit.forward(&latent, H, W, &cap, CAP_N, T);
    assert_eq!(got.len(), want.len());
    let mut max_d = 0f32;
    let mut max_rel = 0f32;
    for (a, b) in got.iter().zip(&want) {
        max_d = max_d.max((a - b).abs());
        max_rel = max_rel.max((a - b).abs() / b.abs().max(1.0));
    }
    println!(
        "dit parity: {}x{} latent + {CAP_N} cap tokens in {:.2}s, max|Δ| = {max_d:.2e} (rel {max_rel:.2e})",
        H,
        W,
        t0.elapsed().as_secs_f64()
    );
    assert!(max_rel < 1e-3, "Next-DiT ≠ numpy: max|Δ| = {max_d}");
}
