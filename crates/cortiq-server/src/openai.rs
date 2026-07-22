//! OpenAI-compatible API endpoints, backed by the real inference
//! pipeline. Generation runs in `spawn_blocking` behind a Mutex — a
//! panic inside the pipeline becomes a 500, never a dead process.

use crate::AppState;
use crate::streaming::{self, ChatStream};
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use cortiq_core::TaskMask;
use cortiq_engine::SamplerConfig;
use cortiq_engine::pipeline::GenerateResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Register OpenAI-compatible routes.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
}

// ─── Models ──────────────────────────────────────────────

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let arch = state.runtime.model().arch();
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelEntry {
            id: format!("{}-cortiq", arch.arch_name),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            owned_by: "cortiq".to_string(),
        }],
    })
}

// ─── Shared types ────────────────────────────────────────

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct CortiqExtension {
    task: Option<String>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct CortiqResponseMeta {
    task_used: String,
    sparsity: f32,
    active_layers: usize,
    execution_mode: String,
    tokens_per_second: f64,
}

#[derive(Serialize)]
struct ApiError {
    error: ApiErrorBody,
}

#[derive(Serialize)]
struct ApiErrorBody {
    message: String,
    r#type: String,
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ApiError {
            error: ApiErrorBody {
                message: message.into(),
                r#type: "invalid_request_error".to_string(),
            },
        }),
    )
        .into_response()
}

/// Run one generation on the shared pipeline (blocking thread).
/// Returns the result plus wall-clock milliseconds.
async fn run_generation(
    state: Arc<AppState>,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    mask: Option<TaskMask>,
    sampler_config: SamplerConfig,
    on_token: Option<cortiq_engine::TokenCallback>,
) -> Result<(GenerateResult, f64), Response> {
    let started = std::time::Instant::now();

    // Check a pipeline slot out for this generation: up to
    // `slots` requests decode concurrently, the rest queue here.
    let mut slot = state.slots.acquire().await;
    let outcome = tokio::task::spawn_blocking(move || {
        let p = &mut *slot.pipe;
        // A pooled pipeline must not inherit sampling state or RNG position
        // from the request that previously occupied this slot.
        p.set_sampler_config(sampler_config);
        p.generate_from_ids(&prompt_ids, max_tokens, mask.as_ref(), on_token)
    })
    .await;

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match outcome {
        Ok(Ok(result)) => Ok((result, elapsed_ms)),
        Ok(Err(e)) => Err(error_response(StatusCode::BAD_REQUEST, e)),
        Err(join_err) => {
            tracing::error!("generation task panicked: {join_err}");
            Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation failed",
            ))
        }
    }
}

fn request_sampler(
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
) -> Result<SamplerConfig, Response> {
    if temperature.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "temperature must be finite and >= 0",
        ));
    }
    if top_p.is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v)) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "top_p must be finite and between 0 and 1",
        ));
    }
    let mut config = SamplerConfig::default();
    if let Some(v) = temperature {
        config.temperature = v;
    }
    if let Some(v) = top_p {
        config.top_p = v;
    }
    config.seed = seed;
    Ok(config)
}

// ─── Chat Completions ────────────────────────────────────

#[derive(Deserialize)]
struct ChatCompletionsRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    stream: bool,
    /// Reasoning-model switch: `false` renders the chat template with
    /// `enable_thinking=false` (e.g. Qwen3/3.5 prefill an empty <think> block,
    /// so the model answers directly). Absent = the template's default.
    #[serde(default)]
    enable_thinking: Option<bool>,
    /// vLLM-style alternative: {"enable_thinking": false} — the explicit
    /// top-level field above wins when both are present.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Value>,
    /// Cortiq extension: task routing
    #[serde(default)]
    cortiq: Option<CortiqExtension>,
}

impl ChatCompletionsRequest {
    /// Effective enable_thinking: top-level field, else chat_template_kwargs.
    fn thinking(&self) -> Option<bool> {
        self.enable_thinking.or_else(|| {
            self.chat_template_kwargs
                .as_ref()
                .and_then(|k| k.get("enable_thinking"))
                .and_then(|v| v.as_bool())
        })
    }
}

#[derive(Serialize)]
struct ChatCompletionsResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    cortiq: Option<CortiqResponseMeta>,
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: String,
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response {
    if req.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must not be empty");
    }

    // Resolve task selection into request-local state. Mutating the runtime's
    // global active task here made concurrent requests use each other's mask.
    let (task_used, request_mask) =
        if let Some(task) = req.cortiq.as_ref().and_then(|c| c.task.as_deref()) {
            let Some(mask) = state.runtime.masks().get(task).cloned() else {
                return error_response(
                    StatusCode::NOT_FOUND,
                    format!("Task mask '{task}' not found"),
                );
            };
            (task.to_string(), Some(mask))
        } else {
            state.runtime.active_selection().await
        };
    let sampler_config = match request_sampler(req.temperature, req.top_p, req.seed) {
        Ok(config) => config,
        Err(response) => return response,
    };

    // Chat template → prompt ids (uses real special tokens).
    let prompt_ids = {
        let msgs: Vec<(String, String)> = req
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        state
            .tokenizer
            .apply_chat_template_opts(&msgs, req.thinking())
    };

    let request_id = format!("cmf-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp() as u64;
    let max_tokens = req.max_tokens as usize;

    if req.stream {
        let (tx, stream) = ChatStream::new(64);
        let model = req.model.clone();
        let id = request_id.clone();
        let state2 = state.clone();

        tokio::spawn(async move {
            // Role prelude chunk.
            let _ = tx
                .send(streaming::StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![streaming::StreamChoice {
                        index: 0,
                        delta: streaming::StreamDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                })
                .await;

            // Real tokens flow from the generation thread through the
            // channel; a closed channel (client gone) cancels generation.
            let tx_tokens = tx.clone();
            let id2 = id.clone();
            let model2 = model.clone();
            let callback: cortiq_engine::TokenCallback = Box::new(move |token: &str| {
                let chunk = streaming::token_chunk(&id2, &model2, token, created);
                tx_tokens.blocking_send(chunk).is_ok()
            });

            let outcome = run_generation(
                state2.clone(),
                prompt_ids,
                max_tokens,
                request_mask,
                sampler_config,
                Some(callback),
            )
            .await;

            match outcome {
                Ok((result, elapsed_ms)) => {
                    state2
                        .runtime
                        .record_generation(result.tokens_generated, elapsed_ms, elapsed_ms)
                        .await;
                    let _ = tx
                        .send(streaming::finish_chunk(
                            &id,
                            &model,
                            &result.finish_reason,
                            created,
                        ))
                        .await;
                }
                Err(_) => {
                    let _ = tx
                        .send(streaming::finish_chunk(&id, &model, "error", created))
                        .await;
                }
            }
        });

        stream.into_sse().into_response()
    } else {
        let (result, elapsed_ms) = match run_generation(
            state.clone(),
            prompt_ids,
            max_tokens,
            request_mask,
            sampler_config,
            None,
        )
        .await
        {
            Ok(r) => r,
            Err(resp) => return resp,
        };

        state
            .runtime
            .record_generation(result.tokens_generated, elapsed_ms, elapsed_ms)
            .await;
        let status = state.runtime.status().await;
        let task_mask = state.runtime.masks().get(&task_used);

        let cortiq_meta = req.cortiq.as_ref().map(|_| CortiqResponseMeta {
            task_used,
            sparsity: task_mask.map(|m| m.sparsity).unwrap_or(0.0),
            active_layers: task_mask
                .map(|m| m.active_layer_count())
                .unwrap_or(state.runtime.model().arch().num_layers),
            execution_mode: format!("{:?}", status.execution_mode),
            tokens_per_second: result.tokens_generated as f64 / (elapsed_ms / 1000.0).max(1e-9),
        });

        Json(ChatCompletionsResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created,
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: result.text.clone(),
                },
                finish_reason: result.finish_reason.clone(),
            }],
            usage: Usage {
                prompt_tokens: result.prompt_tokens as u32,
                completion_tokens: result.tokens_generated as u32,
                total_tokens: (result.prompt_tokens + result.tokens_generated) as u32,
            },
            cortiq: cortiq_meta,
        })
        .into_response()
    }
}

// ─── Completions (legacy) ────────────────────────────────

#[derive(Deserialize)]
struct CompletionsRequest {
    model: String,
    prompt: String,
    temperature: Option<f32>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
}

#[derive(Serialize)]
struct CompletionsResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: u32,
    finish_reason: String,
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionsRequest>,
) -> Response {
    let prompt_ids = state.tokenizer.encode(&req.prompt);

    let sampler_config = match request_sampler(req.temperature, None, None) {
        Ok(config) => config,
        Err(response) => return response,
    };
    let (_, request_mask) = state.runtime.active_selection().await;

    let (result, elapsed_ms) = match run_generation(
        state.clone(),
        prompt_ids,
        req.max_tokens as usize,
        request_mask,
        sampler_config,
        None,
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    state
        .runtime
        .record_generation(result.tokens_generated, elapsed_ms, elapsed_ms)
        .await;

    Json(CompletionsResponse {
        id: format!("cmf-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: chrono::Utc::now().timestamp() as u64,
        model: req.model,
        choices: vec![CompletionChoice {
            text: result.text,
            index: 0,
            finish_reason: result.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: result.prompt_tokens as u32,
            completion_tokens: result.tokens_generated as u32,
            total_tokens: (result.prompt_tokens + result.tokens_generated) as u32,
        },
    })
    .into_response()
}

fn default_max_tokens() -> u32 {
    256
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_options_start_from_defaults_and_validate_ranges() {
        let changed = request_sampler(Some(0.2), Some(0.5), Some(7)).unwrap();
        assert_eq!(changed.temperature, 0.2);
        assert_eq!(changed.top_p, 0.5);
        assert_eq!(changed.seed, Some(7));

        let fresh = request_sampler(None, None, None).unwrap();
        let defaults = SamplerConfig::default();
        assert_eq!(fresh.temperature, defaults.temperature);
        assert_eq!(fresh.top_p, defaults.top_p);
        assert_eq!(fresh.seed, None);

        assert!(request_sampler(Some(-1.0), None, None).is_err());
        assert!(request_sampler(None, Some(1.1), None).is_err());
    }
}
