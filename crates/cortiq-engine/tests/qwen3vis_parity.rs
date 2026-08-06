//! Qwen3-VL's vision tower against the reference forward.
//!
//! `tools/mk_qwen3vis_toy.py` builds the fixture from ComfyUI's own
//! `Qwen3VLVisionModel` on a grid big enough to exercise the three
//! things a one-patch check cannot see: the bilinearly interpolated
//! 48×48 position table, the 2×2 merge-block order the patches are
//! permuted into, and the two different GELUs. Point `CMF_QWEN3VIS_TOY`
//! at the packed directory; without it the test skips.

use cortiq_engine::qwen3vis::{VisionTower, preprocess};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn report(name: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut mx = 0f32;
    let mut se = 0f64;
    let mut rs = 0f64;
    for (&g, &w) in got.iter().zip(want) {
        mx = mx.max((g - w).abs());
        se += ((g - w) as f64).powi(2);
        rs += (w as f64).powi(2);
    }
    println!(
        "{name}: max {mx:.3e} rms {:.3e} over signal rms {:.3e}",
        (se / got.len() as f64).sqrt(),
        (rs / want.len() as f64).sqrt()
    );
    mx
}

#[test]
fn vision_tower_matches_the_reference() {
    let Some(dir) = std::env::var_os("CMF_QWEN3VIS_TOY") else {
        eprintln!("CMF_QWEN3VIS_TOY unset — skipping");
        return;
    };
    let dir = PathBuf::from(dir);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("golden.json")).unwrap()).unwrap();
    let u = |k: &str| meta[k].as_u64().unwrap() as usize;

    let model = Arc::new(cortiq_core::CmfModel::open(dir.join("vis.cmf")).unwrap());
    let tower = VisionTower::from_cmf(&model).unwrap();

    // The preprocessor first: a wrong resize or patch order would show
    // up in the tower's output as noise with no obvious cause.
    let (patches, gh, gw) = preprocess(
        &read_f32(&dir.join("image.bin")),
        u("img_h"),
        u("img_w"),
        tower.patch_size,
        tower.temporal_patch,
        tower.merge,
    );
    assert_eq!((gh, gw), (u("grid_h"), u("grid_w")), "patch grid");
    let mx = report("patches", &patches, &read_f32(&dir.join("patches.bin")));
    assert!(mx < 2e-5, "preprocessing diverges: max {mx:.3e}");

    let (merged, deep) = tower.forward(&patches, gh, gw);
    assert_eq!(merged.len(), u("merged_n") * tower.out_hidden, "merged shape");
    let mx = report("merged", &merged, &read_f32(&dir.join("merged.bin")));
    assert!(mx < 5e-4, "vision tower diverges: max {mx:.3e}");

    assert_eq!(deep.len(), u("n_deep"), "deepstack count");
    for (i, d) in deep.iter().enumerate() {
        let mx = report(&format!("deepstack {i}"), d, &read_f32(&dir.join(format!("deep{i}.bin"))));
        assert!(mx < 5e-4, "deepstack {i} diverges: max {mx:.3e}");
    }
}
