//! End-to-end numerical contract for the opt-in resident Embryo graph.
//!
//! This test deliberately runs in two separate processes.  The wgpu context
//! is process-global, so a CPU reference and the parallel graph must not race
//! one another while changing `CMF_GPU`.  Run the CPU leg first with
//! `CMF_EMBRYO_PARITY_MODE=cpu` and `CMF_EMBRYO_PARITY_OUT=/tmp/ref.f32`, then
//! the Vulkan leg with `CMF_EMBRYO_PARITY_MODE=parallel` and
//! `CMF_EMBRYO_PARITY_REFERENCE=/tmp/ref.f32`.

#![cfg(feature = "gpu")]

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use cortiq_engine::tokenizer::Tokenizer;
use std::sync::Arc;

#[path = "common/embryo_synth.rs"]
mod embryo_synth;

/// `CMF_EMBRYO_PARITY_IDS`: an explicit comma list, `synth:N[:seed]`
/// for N deterministic pseudo-random ids (the bounded gate's 512-token
/// prefix on a synthetic genome, which has no natural text), or
/// `natural` for the first 512 tokens of `CMF_EMBRYO_PARITY_PROMPT_FILE`
/// through `CMF_EMBRYO_PARITY_TOKENIZER` (a real genome's gate).
fn env_ids_for(vocab: usize) -> Vec<u32> {
    let spec = std::env::var("CMF_EMBRYO_PARITY_IDS").unwrap_or_else(|_| "1,2,3,4,5,6,7,8".to_string());
    if spec == "natural" {
        return natural_prefix_ids().expect("natural ids need CMF_EMBRYO_PARITY_TOKENIZER + _PROMPT_FILE");
    }
    if let Some(rest) = spec.strip_prefix("synth:") {
        let mut it = rest.split(':');
        let n: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(512);
        let seed: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(11);
        return embryo_synth::synth_ids(n, seed, vocab);
    }
    spec.split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

/// `CMF_EMBRYO_PARITY_MODEL`: a path, or `synth-bounded` / `synth-bounded-tiny`
/// (vmf_phase mixers) / `synth-gdn-bounded` / `synth-gdn-bounded-tiny`
/// (gated_delta_net mixers, plan variant B) for a synthetic genome with the
/// native bounded anchor (written once into the temp dir, shared by the
/// CPU and GPU legs).
fn resolve_model_path(spec: &str) -> String {
    match spec {
        "synth-bounded" => embryo_synth::synth_genome_path(&embryo_synth::SynthGeom::embryo0_bounded()),
        "synth-bounded-tiny" => embryo_synth::synth_genome_path(&embryo_synth::SynthGeom::tiny_bounded()),
        "synth-gdn-bounded" => {
            embryo_synth::synth_genome_path(&embryo_synth::SynthGeom::embryo3_gdn_bounded())
        }
        "synth-gdn-bounded-tiny" => {
            embryo_synth::synth_genome_path(&embryo_synth::SynthGeom::tiny_gdn_bounded())
        }
        other => return other.to_string(),
    }
    .to_string_lossy()
    .into_owned()
}

/// Write a synthetic bounded genome to `$CMF_SYNTH_OUT` for `cortiq bench`
/// / `cortiq info` / `cortiq requant` on a server: `CMF_SYNTH_GEOM` =
/// `tiny` | `gdn` (6×GDN + 2×bounded, 56M class) | `gdn-tiny` |
/// unset (Embryo-0 vmf_phase + 1 bounded anchor).
#[test]
#[ignore]
fn write_synth_bounded_genome() {
    let out = std::env::var("CMF_SYNTH_OUT").expect("set CMF_SYNTH_OUT=<path.cmf>");
    let g = match std::env::var("CMF_SYNTH_GEOM").as_deref() {
        Ok("tiny") => embryo_synth::SynthGeom::tiny_bounded(),
        Ok("gdn") => embryo_synth::SynthGeom::embryo3_gdn_bounded(),
        Ok("gdn-tiny") => embryo_synth::SynthGeom::tiny_gdn_bounded(),
        _ => embryo_synth::SynthGeom::embryo0_bounded(),
    };
    embryo_synth::write_synth_genome(std::path::Path::new(&out), &g);
    let m = CmfModel::open(&out).expect("reopen");
    let st = cortiq_engine::loader::per_sequence_state_bytes(&m).unwrap();
    eprintln!(
        "wrote {out}: {} tensors, per-sequence state {} B bounded + {} B recurrent, growing layers {}",
        m.tensors.len(),
        st.bounded_bytes,
        st.recurrent_bytes,
        st.growing_layers
    );
    assert_eq!(st.growing_layers, 0);
}

fn read_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read logits reference");
    assert_eq!(bytes.len() % 4, 0, "reference is not f32-aligned");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn write_f32(path: &str, values: &[f32]) {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &x in values {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write logits reference");
}

fn assert_finite(label: &str, values: &[f32]) {
    for (i, &value) in values.iter().enumerate() {
        assert!(
            value.is_finite(),
            "{label}[{i}] is not finite: {value:?}"
        );
    }
}

fn read_u32(path: &str) -> Vec<u32> {
    std::fs::read_to_string(path)
        .expect("read token reference")
        .split_whitespace()
        .map(|s| s.parse::<u32>().expect("token reference is u32"))
        .collect()
}

fn write_u32(path: &str, values: &[u32]) {
    let text = values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    std::fs::write(path, text).expect("write token reference");
}

fn natural_prefix_ids() -> Option<Vec<u32>> {
    // A synthetic genome has no natural text: the 512-id synthetic prefix
    // stands in for it (`CMF_EMBRYO_PARITY_IDS=synth:512`).
    if std::env::var("CMF_EMBRYO_PARITY_TOKENIZER").is_err() {
        let spec = std::env::var("CMF_EMBRYO_PARITY_IDS").ok()?;
        let rest = spec.strip_prefix("synth:")?;
        let mut it = rest.split(':');
        let n: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(512);
        let seed: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(11);
        let vocab: usize = std::env::var("CMF_EMBRYO_PARITY_VOCAB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        if n < 512 {
            return None;
        }
        return Some(embryo_synth::synth_ids(n, seed, vocab)[..512].to_vec());
    }
    let tokenizer_path = std::env::var("CMF_EMBRYO_PARITY_TOKENIZER").ok()?;
    let prompt_path = std::env::var("CMF_EMBRYO_PARITY_PROMPT_FILE").ok()?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("load parity tokenizer");
    let prompt = std::fs::read_to_string(prompt_path).expect("read natural parity prompt");
    let ids = tokenizer.with_bos(tokenizer.encode(&prompt));
    assert!(
        ids.len() >= 512,
        "natural parity prompt has only {} tokens",
        ids.len()
    );
    Some(ids[..512].to_vec())
}

fn greedy_config() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        min_p: 0.0,
        seed: Some(0),
        // The Embryo CMF declares eos=32760. Keep the state probe at its
        // requested length so a natural prefix cannot terminate the contract
        // early; this is only the deterministic regression sampler.
        suppress_tokens: vec![32760],
        penalty_window: 0,
    }
}

fn greedy_ids(pipeline: &mut Pipeline, ids: &[u32], max_tokens: usize) -> Vec<u32> {
    pipeline.set_sampler_config(greedy_config());
    let result = pipeline
        .generate_from_ids(ids, max_tokens, None, None)
        .expect("greedy continuation");
    assert_eq!(result.tokens_generated, max_tokens);
    assert_eq!(result.token_ids.len(), max_tokens);
    result.token_ids
}

fn assert_reused_and_interleaved_state(
    model: &Arc<CmfModel>,
    pipeline: &mut Pipeline,
    ids: &[u32],
) {
    // The helper itself is stepwise. Compare it with the normal (possibly
    // chunked/pair) forward so the CPU prefill and resident token walk share
    // one teacher-forced contract.
    let stepwise = pipeline.prefill_next_logits(ids, None);
    assert_finite("stepwise logits", &stepwise);

    let short = ids.len().min(64);
    assert!(short >= 8, "state probe needs at least eight IDs");
    let a = ids[..short / 2].to_vec();
    let b = ids[short / 2..short].to_vec();
    let mut pa = Pipeline::from_model(model, greedy_config()).expect("pipeline A");
    let mut pb = Pipeline::from_model(model, greedy_config()).expect("pipeline B");
    let mut fresh = Pipeline::from_model(model, greedy_config()).expect("fresh pipeline");

    // A and B remain live simultaneously. Extending each prompt after the
    // other sequence ran exercises cached resident state and catches a
    // sequence-key collision that a single reset-only test cannot see.
    let a1 = greedy_ids(&mut pa, &a, 1);
    let b1 = greedy_ids(&mut pb, &b, 1);
    let mut a_ext = a.clone();
    a_ext.extend_from_slice(&a1);
    let mut b_ext = b.clone();
    b_ext.extend_from_slice(&b1);
    let a2 = greedy_ids(&mut pa, &a_ext, 1);
    let b2 = greedy_ids(&mut pb, &b_ext, 1);
    let a2_fresh = greedy_ids(&mut fresh, &a_ext, 1);
    let b2_fresh = greedy_ids(&mut fresh, &b_ext, 1);
    assert_eq!(a2, a2_fresh, "sequence A cached continuation changed");
    assert_eq!(b2, b2_fresh, "sequence B cached continuation changed");
}

#[test]
fn resident_parallel_logits_and_reset_match_cpu() {
    let (Ok(model_path), Ok(mode)) = (
        std::env::var("CMF_EMBRYO_PARITY_MODEL"),
        std::env::var("CMF_EMBRYO_PARITY_MODE"),
    ) else {
        eprintln!("resident parity skipped: set CMF_EMBRYO_PARITY_MODEL and _MODE");
        return;
    };
    assert!(matches!(mode.as_str(), "cpu" | "parallel"));
    // Refusals of the resident path (shader validation, eligibility) are
    // `tracing::warn!`s: `RUST_LOG=warn` makes them visible here.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();
    let model_path = resolve_model_path(&model_path);
    let vocab = CmfModel::open(&model_path).expect("open model").arch().vocab_size;
    unsafe { std::env::set_var("CMF_EMBRYO_PARITY_VOCAB", vocab.to_string()) };
    let ids = env_ids_for(vocab);
    assert!(!ids.is_empty(), "parity token sequence is empty");

    // The mode is selected before Pipeline performs its first forward.  The
    // test is intentionally one process per mode; see the module comment.
    unsafe {
        if mode == "cpu" {
            std::env::set_var("CMF_GPU", "0");
            std::env::remove_var("CMF_EMBRYO_RESIDENT");
            std::env::remove_var("CMF_GPU_WGPU_GRAPH");
        } else {
            std::env::set_var("CMF_GPU", "wgpu");
            std::env::set_var("CMF_GPU_PROBE", "0");
            std::env::set_var("CMF_GPU_WGPU_GRAPH", "1");
            std::env::set_var("CMF_EMBRYO_RESIDENT", "parallel");
        }
    }
    let model = Arc::new(CmfModel::open(&model_path).expect("open model"));
    let mut pipeline = Pipeline::from_model(&model, SamplerConfig::default()).expect("pipeline");
    if mode == "parallel" {
        eprintln!(
            "resident parity: gpu enabled_here={} graph_on(prefill)={} graph_on(decode)={}",
            cortiq_engine::gpu::enabled_here(),
            cortiq_engine::gpu::wgpu_graph_on(cortiq_engine::gpu::GraphPhase::Prefill),
            cortiq_engine::gpu::wgpu_graph_on(cortiq_engine::gpu::GraphPhase::Decode),
        );
    }

    let first = pipeline.forward_ids(&ids, None).expect("first forward");
    assert_finite("first logits", &first);
    // A bounded genome on the resident graph: the device holds exactly the
    // recurrent state + the anchor rings, and both are the same numbers at
    // every context depth (the O(1) claim as measured bytes).
    let device_before = pipeline.device_state_bytes();
    if mode == "parallel" && pipeline.bounded_native() {
        let (st, kv) = device_before.expect("bounded genome must run resident");
        let m = CmfModel::open(&model_path).expect("open model");
        let want = cortiq_engine::loader::per_sequence_state_bytes(&m).unwrap();
        assert_eq!(kv as usize, want.bounded_bytes, "device ring bytes = header rings");
        assert_eq!(st as usize, want.recurrent_bytes, "device state bytes = header recurrent state");
        assert!(st > 0 && kv > 0);
        eprintln!("device state after {} ids: state {st} B, rings {kv} B", ids.len());
    }
    let stepwise = pipeline.prefill_next_logits(&ids, None);
    assert_finite("prefill-stepwise logits", &stepwise);
    if mode == "parallel" && pipeline.bounded_native() {
        assert_eq!(pipeline.device_state_bytes(), device_before, "device state grew");
    }
    assert_eq!(first.len(), stepwise.len());
    let stepwise_max = first
        .iter()
        .zip(&stepwise)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(stepwise_max.is_finite(), "stepwise error is not finite");
    // On the resident graph `forward_ids` runs the chunked prefill and
    // `prefill_next_logits` the per-position walk: this is the chunked ==
    // token-by-token gate (`CMF_EMBRYO_CHUNK_TOL`, plan: ≤ 1e-5).
    let chunk_tol = std::env::var("CMF_EMBRYO_CHUNK_TOL")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(2e-3);
    eprintln!("prefill parity: chunked-vs-stepwise max|Δ|={stepwise_max:.6e} (tol {chunk_tol:.1e})");
    assert!(
        stepwise_max <= chunk_tol,
        "stepwise logits diverged: {stepwise_max} > {chunk_tol}"
    );
    pipeline.reset_session();
    let second = pipeline.forward_ids(&ids, None).expect("reset forward");
    assert_finite("reset logits", &second);
    assert_eq!(first.len(), second.len());
    let reset_max = first
        .iter()
        .zip(&second)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(reset_max.is_finite(), "reset error is not finite");
    assert!(
        reset_max <= 1e-5,
        "reset changed logits: max|Δ|={reset_max}"
    );

    if let Some(path) = std::env::var_os("CMF_EMBRYO_PARITY_OUT") {
        write_f32(path.to_str().expect("reference path utf8"), &first);
    }
    if let Ok(path) = std::env::var("CMF_EMBRYO_PARITY_REFERENCE") {
        let reference = read_f32(&path);
        assert_finite("CPU reference logits", &reference);
        assert_eq!(first.len(), reference.len());
        let mut max_diff = 0.0f32;
        let mut sum_sq = 0.0f64;
        for (&got, &want) in first.iter().zip(&reference) {
            let d = (got - want).abs();
            assert!(d.is_finite(), "logit error is not finite: {got:?} vs {want:?}");
            max_diff = max_diff.max(d);
            sum_sq += f64::from(d) * f64::from(d);
        }
        let rms = (sum_sq / first.len() as f64).sqrt();
        let tol = std::env::var("CMF_EMBRYO_PARITY_TOL")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(2e-3);
        assert!(max_diff.is_finite(), "max logit error is not finite");
        assert!(sum_sq.is_finite() && rms.is_finite(), "RMS error is not finite");
        assert!(tol.is_finite() && tol >= 0.0, "parity tolerance is not finite");
        assert!(
            max_diff <= tol,
            "resident logits diverged: max|Δ|={max_diff:.6e}, rms={rms:.6e}, tol={tol:.6e}"
        );
        eprintln!(
            "resident parity: logits={} max|Δ|={max_diff:.6e} rms={rms:.6e} reset_max={reset_max:.6e}",
            first.len()
        );
    }

    // This is opt-in because it runs several real decode sequences. The
    // terminal gate enables it with a natural held-out 512-token prefix and
    // a CPU-produced 32-token reference, rather than synthetic IDs alone.
    if std::env::var("CMF_EMBRYO_PARITY_STATE").as_deref() == Ok("1") {
        assert_reused_and_interleaved_state(&model, &mut pipeline, &ids);
        if let Some(natural) = natural_prefix_ids() {
            let generated = greedy_ids(&mut pipeline, &natural, 32);
            if let Some(path) = std::env::var_os("CMF_EMBRYO_PARITY_GENERATION_OUT") {
                write_u32(path.to_str().expect("generation path utf8"), &generated);
            }
            if let Ok(path) = std::env::var("CMF_EMBRYO_PARITY_GENERATION_REFERENCE") {
                let reference = read_u32(&path);
                assert_eq!(generated, reference, "natural continuation changed");
            }
            eprintln!(
                "resident state: natural_prefix={} continuation={} reset+stepwise+interleaved=ok",
                natural.len(),
                generated.len()
            );
        }
    }
}
