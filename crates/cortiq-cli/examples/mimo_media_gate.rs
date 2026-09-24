//! One model load for embedding-ingress parity and a suite of MiMo media
//! prompts. Usage: MODEL CASES.json COMPANION OUTPUT.json [MAX_TOKENS].
//! Cases are [{"label":..., "messages":[...]}]. Results contain exact IDs,
//! answers, sizes and timings; task-specific OCR/WER assertions stay outside
//! this diagnostic rather than embedding the expected answer in the prompt.
use cortiq_core::CmfModel;
use cortiq_engine::{
    Pipeline, SamplerConfig,
    mimo_ingress::{self, MediaOptions},
};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Instant};
fn main() -> anyhow::Result<()> {
    let a: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        (5..=6).contains(&a.len()),
        "usage: mimo_media_gate MODEL CASES.json COMPANION OUTPUT.json [MAX_TOKENS]"
    );
    let n: usize = a.get(5).map(|s| s.parse()).transpose()?.unwrap_or(128);
    let model = Arc::new(CmfModel::open_sharded(&a[1])?);
    let mut p = Pipeline::from_model(
        &model,
        SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 1.0,
            seed: Some(42),
            ..Default::default()
        },
    )?;
    p.speculative = false;
    // Actual model/GPU Hidden ingress must equal the established ID path.
    let ids = p.tokenizer.apply_chat_template_opts(
        &[(
            "user".into(),
            "Explain why the sky is blue in one sentence.".into(),
        )],
        Some(false),
    );
    let embeddings: Vec<_> = ids.iter().flat_map(|&id| p.embed_id(id)).collect();
    p.ignore_eos = true;
    let ordinary = p
        .generate_from_ids(&ids, 32, None, None)
        .map_err(anyhow::Error::msg)?;
    let injected = p
        .generate_from_embeds(&ids, &embeddings, 32, None, None)
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        ordinary.token_ids == injected.token_ids,
        "real-model ID/Hidden ingress token mismatch"
    );
    p.speculative = true;
    let speculative = p
        .generate_from_embeds(&ids, &embeddings, 32, None, None)
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        ordinary.token_ids == speculative.token_ids,
        "real-model Hidden plain/MTP token mismatch"
    );
    p.ignore_eos = false;
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let options = MediaOptions {
        companion: Some(PathBuf::from(&a[3])),
        ..Default::default()
    };
    let mut results = vec![
        json!({"gate":"real-model embedding ingress + MTP", "identical":true, "tokens":ordinary.token_ids}),
    ];
    std::fs::write(&a[4], serde_json::to_vec_pretty(&results)?)?;
    for case in cases {
        let label = case["label"].as_str().unwrap_or("case");
        eprintln!("media gate: {label}");
        let messages = case["messages"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing messages for {label}"))?;
        let t = Instant::now();
        let prepared = mimo_ingress::prepare(&model, &p, messages, None, Some(false), &options)
            .map_err(anyhow::Error::msg)?;
        let prep_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let output = match prepared.rows.as_deref() {
            Some(rows) => p.generate_from_embeds(&prepared.token_ids, rows, n, None, None),
            None => p.generate_from_ids(&prepared.token_ids, n, None, None),
        }
        .map_err(anyhow::Error::msg)?;
        eprintln!("{label}: {}", output.text);
        results.push(
            json!({"label":label,"prompt_tokens":prepared.token_ids.len(), "prepare_s":prep_s,
            "generate_s":t.elapsed().as_secs_f64(),"text":output.text,"token_ids":output.token_ids,
            "finish_reason":output.finish_reason}),
        );
        std::fs::write(&a[4], serde_json::to_vec_pretty(&results)?)?;
    }
    #[cfg(feature = "gpu")]
    cortiq_engine::gpu_wgpu::shutdown();
    Ok(())
}
