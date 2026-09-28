//! macOS: a natively bounded / GDN Embryo file runs on the CPU bounded
//! path (the default there), and the native Metal block and chunk paths
//! (`q1_graph_gpu`, `prefill_batch_metal`) refuse it cleanly — no panic,
//! finite logits, batched prefill == per-position walk, generation runs.
//! `CMF_GPU=1` selects Metal explicitly so the refusal is exercised, not
//! skipped.

#![cfg(target_os = "macos")]

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::sync::Arc;

#[path = "common/embryo_synth.rs"]
mod embryo_synth;

fn greedy() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        seed: Some(0),
        ..Default::default()
    }
}

#[test]
fn metal_paths_refuse_bounded_and_gdn_embryo_cleanly() {
    // Before any pipeline: the backend is chosen at first use.
    unsafe {
        std::env::set_var("CMF_GPU", "1");
        std::env::remove_var("CMF_EMBRYO_RESIDENT");
        std::env::remove_var("CMF_GPU_WGPU_GRAPH");
    }
    for g in [
        embryo_synth::SynthGeom::tiny_bounded(),
        embryo_synth::SynthGeom::tiny_gdn_bounded(),
    ] {
        let path = embryo_synth::synth_genome_path(&g);
        let model = Arc::new(CmfModel::open(&path).expect("open synthetic genome"));
        let mut p = Pipeline::from_model(&model, greedy()).expect("bounded/GDN file loads on macOS");
        assert!(p.bounded_native(), "{}: anchor_core must be native", g.tag());
        let ids = embryo_synth::synth_ids(300, 5, g.vocab);
        // Batched prefill: the Metal chunk GEMM path is offered the
        // prompt (> 8 tokens, gdn_cfg present for the GDN genome) and
        // must hand it back to the CPU chunk prefill.
        let a = p.forward_ids(&ids, None).expect("forward_ids");
        assert!(a.iter().all(|v| v.is_finite()), "{}: non-finite logits", g.tag());
        // Per-position walk: the Metal block path scans the layers and
        // must end its run at the first bounded/GDN layer.
        let b = p.prefill_next_logits(&ids, None);
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max <= 1e-4,
            "{}: batched vs per-position CPU logits differ by {max}",
            g.tag()
        );
        let r = p
            .generate_from_ids(&ids, 8, None, None)
            .expect("generate on the CPU bounded path");
        assert_eq!(r.tokens_generated, 8, "{}", g.tag());
        // The whole state is the rings + the recurrent record the header
        // predicts; nothing per position was appended and no zero record
        // was planted on the anchor by a half-taken device path
        // (`metal_rows_run` used to size every layer before refusing).
        let want = cortiq_engine::loader::per_sequence_state_bytes(&model).unwrap();
        assert_eq!(want.growing_layers, 0);
        assert_eq!(
            p.kv_cache.bounded_state_bytes(),
            want.bounded_bytes + 8 * g.anchor_layers.len(),
            "{}: ring bytes",
            g.tag()
        );
        assert_eq!(
            p.kv_cache.recurrent_state_bytes(),
            want.recurrent_bytes,
            "{}: recurrent bytes must be the header's record, nothing planted",
            g.tag()
        );
        assert_eq!(p.device_state_bytes(), None, "{}: no device sequence expected", g.tag());
        eprintln!(
            "{}: metal refused cleanly; batched-vs-stepwise max|Δ|={max:.3e}, state {} B",
            g.tag(),
            p.kv_cache.bounded_state_bytes() + p.kv_cache.recurrent_state_bytes()
        );
    }
}
