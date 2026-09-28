//! The genome exported to .cmf and run by the RUNTIME (CPU pipeline:
//! vmf_phase mixer with κ, GQA anchor, resonance-routed experts + shared,
//! hierarchical head) must produce the trainer's log-probabilities.
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, Mixer, init_params};
use cortiq_embryo::ops::lcg_vec;
use cortiq_embryo::train::Checkpoint;

#[test]
fn runtime_matches_trainer_logprobs() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 4;
    run_runtime_matches_trainer_logprobs(cfg);
}

/// The new operator tag is a mixed-layer contract: selected hybrids use the
/// normalized in-place update while the other hybrids and the anchor retain
/// their established paths.  Keep this beside the legacy test so an export
/// cannot accidentally route every layer through one operator.
#[test]
fn runtime_matches_mixed_phase_delta_trainer_logprobs() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 4;
    cfg.layers = 8;
    cfg.anchor_every = 8;
    cfg.phase_delta_layers = Some(vec![3, 6]);
    run_runtime_matches_trainer_logprobs(cfg);
}

/// The GDN mixer exported as `linear_core.kind = "gated_delta_net"` +
/// `linear_attn.*` tensors must reproduce the trainer's log-probs through
/// the runtime's CPU `gdn_step` (the plan's trainer ⇔ runtime gate).
#[test]
fn runtime_matches_gdn_mixer_trainer_logprobs() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 4;
    cfg.mixer = Mixer::Gdn;
    cfg.gdn_heads = 2;
    cfg.gdn_dk = 32;
    cfg.gdn_dv = 32;
    cfg.anchor_window = 16;
    cfg.anchor_sink = 2;
    run_runtime_matches_trainer_logprobs(cfg);
}

fn run_runtime_matches_trainer_logprobs(cfg: EmbryoCfg) {
    let Some(_) = cortiq_embryo::metal::ctx() else {
        return;
    };
    // the CPU pipeline is the reference operator here
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let (b, t) = (1usize, 64usize);
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 5);
    let gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).expect("gpu");
    let tokens: Vec<u32> = lcg_vec(21, t)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * cfg.vocab as f32) as u32 % cfg.vocab as u32)
        .collect();
    // seed the descriptors from this batch and read them back — routing state
    // is part of the model; run one training forward with updates on
    let targets: Vec<u32> = tokens.iter().cycle().skip(1).take(t).cloned().collect();
    let mut gpu = gpu;
    let _ = gpu.train_step(&tokens, &targets, 0.0, 0.0, 1e9); // lr 0: params unchanged, descriptors seeded
    gpu.desc_updates.set(false);
    let params = gpu.params_host();
    let extras: Vec<(String, Vec<f32>)> = gpu
        .desc_host()
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    let xf = gpu.forward_hidden(&tokens);
    // trainer-side hierarchical log-probs from xf
    let (h, v, ncl) = (cfg.hidden, cfg.vocab, cfg.head_clusters);
    let cs = v / ncl;
    let e = &params[lay.embed..lay.embed + v * h];
    let cm = &params[lay.head_clusters..lay.head_clusters + ncl * h];
    let logprobs = |x: &[f32]| -> Vec<f32> {
        let mut lc: Vec<f32> = (0..ncl)
            .map(|c| (0..h).map(|j| cm[c * h + j] * x[j]).sum())
            .collect();
        let mx = lc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx + lc.iter().map(|z| (z - mx).exp()).sum::<f32>().ln();
        for z in &mut lc {
            *z -= lse;
        }
        let mut out = vec![0.0f32; v];
        for c in 0..ncl {
            let lg: Vec<f32> = (0..cs)
                .map(|s| (0..h).map(|j| e[(c * cs + s) * h + j] * x[j]).sum())
                .collect();
            let bm = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let bl = bm + lg.iter().map(|z| (z - bm).exp()).sum::<f32>().ln();
            for s in 0..cs {
                out[c * cs + s] = lc[c] + lg[s] - bl;
            }
        }
        out
    };
    // export
    let tok_json = {
        // a minimal byte-level tokenizer.json (vocab must cover the ids we feed)
        let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
        let mut counts = std::collections::HashMap::new();
        cortiq_embryo::tokenizer::count_words("hello world hello embryo", &re, &mut counts);
        cortiq_embryo::tokenizer::train(&counts, cfg.vocab, false).to_hf_json()
    };
    let ck = Checkpoint {
        cfg: cfg.clone(),
        step: 0,
        params: params.clone(),
        m: None,
        v: None,
        extras,
    };
    // one directory per test invocation (the tests run in parallel)
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "embryo_parity_{}_{}_{}",
        std::process::id(),
        cfg.layers,
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for (profile, dtype, tolerance) in [
        ("f32", cortiq_core::TensorDtype::F32, 2e-3f32),
        ("f16", cortiq_core::TensorDtype::F16, 1e-3f32),
    ] {
        let path = dir.join(format!("tiny-{profile}.cmf"));
        cortiq_embryo::export::export_with_dtype(&ck, tok_json.as_bytes(), &path, dtype)
            .expect("export");
        // runtime
        let model =
            std::sync::Arc::new(cortiq_core::format::CmfModel::open(&path).expect("open cmf"));
        let mut pipe = cortiq_engine::pipeline::Pipeline::from_model(
            &model,
            cortiq_engine::sampler::SamplerConfig::default(),
        )
        .expect("pipeline");
        let mut worst = 0.0f32;
        for pos in [0usize, 1, 5, 17, 40, 63] {
            let want = logprobs(&xf[pos * h..(pos + 1) * h]);
            let got = pipe.prefill_next_logits(&tokens[..=pos], None);
            assert_eq!(got.len(), v);
            let d = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{profile} pos {pos}: max|Δ logprob| = {d:.3e}  (argmax trainer {} runtime {})",
                want.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0,
                got.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            );
            worst = worst.max(d);
        }
        assert!(
            worst < tolerance,
            "{profile} runtime vs trainer log-probs differ: max {worst}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
