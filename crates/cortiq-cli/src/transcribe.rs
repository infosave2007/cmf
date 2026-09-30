use anyhow::{Context, Result};

pub fn run(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    audio: &str,
    language: &str,
    task: &str,
    max_new_tokens: usize,
) -> Result<String> {
    cortiq_engine::whisper::transcribe(
        model,
        std::path::Path::new(audio),
        language,
        task,
        max_new_tokens,
    )
    .map_err(anyhow::Error::msg)
    .with_context(|| format!("transcribing {audio}"))
}
