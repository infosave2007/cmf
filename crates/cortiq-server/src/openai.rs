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
    /// Nullable: an assistant turn that made tool calls has
    /// `content: null` in the OpenAI shape, and a required field here
    /// 422'd the whole conversation on the SECOND request of every
    /// agent loop — the one that carries the history.
    #[serde(default)]
    content: Option<MessageContent>,
    /// Assistant history: the calls it made, echoed back by the client.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
    /// `role: "tool"` results reference the call they answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

/// `content` in the shape clients actually send it.
///
/// The OpenAI schema allows a bare string OR an array of typed blocks, and
/// coding assistants (Cline, Roo/Zoo Code) switch to the array form as soon
/// as they attach file context or a system preamble. A `String`-only field
/// rejected those with `422 Failed to deserialize the JSON body ... expected
/// a string` before the model was ever reached — which is why the admin
/// playground worked (flat prompt) and the IDE did not.
///
/// Untagged: serde tries the string first, then the block list.
#[derive(Deserialize, Serialize, Clone)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// One block of a structured `content` array. Deliberately permissive —
/// every field optional, unknown fields ignored — so a new block type from
/// some future client degrades to "no text" instead of a 422.
#[derive(Deserialize, Serialize, Clone)]
struct ContentBlock {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

impl MessageContent {
    /// Flatten to the plain prompt text the pipeline consumes. Text blocks
    /// join with newlines, in order; non-text blocks (images and friends)
    /// contribute nothing rather than failing the turn.
    fn text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(bs) => bs
                .iter()
                .filter_map(|b| b.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
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
    // The organism's day side: idle marker + OOD buffer (CMF_OOD_DIR).
    crate::ood::touch_last_request();
    let ood_on = crate::ood::ood_dir().is_some();
    let ood_state = state.clone();

    // Check a pipeline slot out for this generation: up to
    // `slots` requests decode concurrently, the rest queue here.
    let mut slot = state.slots.acquire().await;
    let remote = state.remote.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        if ood_on {
            let text = ood_state.tokenizer.decode(&prompt_ids);
            let p = &mut *slot.pipe;
            crate::ood::record_if_ood(ood_state.runtime.model(), p, &prompt_ids, &text);
        }
        // Replica mode: the compute happens HERE, on a blocking-pool
        // thread that never saw the pin `acquire` set on the async
        // thread. Pin it again or every replica quietly shares card 0
        // (measured: 400 tokens across two "replicas" ran 103 tok/s,
        // exactly one card's worth, with the second card at 17 MB).
        cortiq_engine::gpu::set_current_device(slot.device);
        let p = &mut *slot.pipe;
        // A pooled pipeline must not inherit sampling state or RNG position
        // from the request that previously occupied this slot.
        p.set_sampler_config(sampler_config);
        match remote {
            Some(rm) => {
                // Task masks would apply to this side's layers only —
                // refuse rather than run half a mask (same rule as
                // `run --peer`).
                if mask.is_some() {
                    return Err(
                        "this server runs a network split (--peer): task masks are not \
                         supported yet — use task 'general'"
                            .to_string(),
                    );
                }
                let mut rm = rm.lock().expect("remote segment mutex");
                cortiq_net::generate_split(p, &mut rm, &prompt_ids, max_tokens, None, on_token)
                    .map(
                    |(r, st)| {
                        if st.remote_steps > 0 {
                            tracing::info!(
                                "net: prefill {:.0} ms ({} of {} pos) · {} trips · {:.2} ms avg · {:.0}% of decode",
                                st.prefill_s * 1e3,
                                st.prefilled,
                                r.prompt_tokens,
                                st.remote_steps,
                                st.net_s * 1e3 / st.remote_steps as f64,
                                100.0 * st.net_s / st.decode_s.max(1e-9),
                            );
                        }
                        r
                    },
                )
            }
            None => p.generate_from_ids(&prompt_ids, max_tokens, mask.as_ref(), on_token),
        }
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
    /// OpenAI function calling. Passed through to the FILE's chat
    /// template, whose `{%- if tools %}` branch has been waiting for
    /// them since the first Qwen-family convert.
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    /// "none" suppresses the tool prompt; "auto"/absent lets the model
    /// decide. A forced {"function": {...}} is honoured as "auto" —
    /// grammar-constrained forcing is not implemented, and pretending
    /// otherwise would be worse than saying so.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
}

impl ChatCompletionsRequest {
    /// Tools the template should see: none when absent, empty, or
    /// explicitly refused via tool_choice: "none".
    fn effective_tools(&self) -> Option<&[serde_json::Value]> {
        if matches!(
            self.tool_choice.as_ref().and_then(|v| v.as_str()),
            Some("none")
        ) {
            return None;
        }
        match self.tools.as_deref() {
            Some([]) | None => None,
            Some(ts) => Some(ts),
        }
    }

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
    let mut sampler_config = match request_sampler(req.temperature, req.top_p, req.seed) {
        Ok(config) => config,
        Err(response) => return response,
    };
    if req.thinking() == Some(false) {
        let think_tokens = state.tokenizer.encode("<think>");
        sampler_config.suppress_tokens.extend(think_tokens);
    }

    // Chat template → prompt ids (uses real special tokens).
    let prompt_ids = {
        let mut msgs: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|m| {
                let mut o = serde_json::json!({
                    "role": m.role,
                    "content": m.content.as_ref().map(|c| c.text()).unwrap_or_default(),
                });
                if let Some(tc) = &m.tool_calls {
                    // OpenAI sends function.arguments as a STRING of
                    // JSON; some templates (Nanbeige's XML history
                    // branch) iterate it as an object. Normalise:
                    // parseable strings become objects, everything else
                    // passes through untouched. Qwen-style templates
                    // tojson the object back to the identical text.
                    let mut tc = tc.clone();
                    if let Some(arr) = tc.as_array_mut() {
                        for call in arr {
                            if let Some(args) = call
                                .get_mut("function")
                                .and_then(|f| f.get_mut("arguments"))
                            {
                                if let Some(s) = args.as_str() {
                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
                                        if v.is_object() {
                                            *args = v;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    o["tool_calls"] = tc;
                }
                if let Some(id) = &m.tool_call_id {
                    o["tool_call_id"] = serde_json::json!(id);
                }
                if let Some(n) = &m.name {
                    o["name"] = serde_json::json!(n);
                }
                o
            })
            .collect();
        // Hard thinking suppression: when enable_thinking=false, inject a
        // system-level directive so even models that ignore the empty
        //  block still produce direct answers.
        eprintln!("[serve] thinking={:?}", req.thinking());
        if req.thinking() == Some(false) {
            let has_system = msgs.iter().any(|m| m["role"] == "system");
            let directive = "Answer directly and concisely. Do NOT reason, think step-by-step, or explain your process. Output ONLY the final answer.";
            if has_system {
                // Prepend to existing system message
                if let Some(m) = msgs.iter_mut().find(|m| m["role"] == "system") {
                    let cur = m["content"].as_str().unwrap_or_default();
                    m["content"] = serde_json::json!(format!("{directive}\n\n{cur}"));
                }
            } else {
                msgs.insert(
                    0,
                    serde_json::json!({"role": "system", "content": directive}),
                );
            }
        }
        eprintln!("[serve] msgs[0]={:?}", msgs.first());
        state
            .tokenizer
            .apply_chat_template_json(&msgs, req.effective_tools(), req.thinking())
    };

    let request_id = format!("cmf-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp() as u64;
    let max_tokens = req.max_tokens as usize;

    if req.stream {
        let tool_names: Vec<String> = req
            .effective_tools()
            .map(|ts| {
                ts.iter()
                    .filter_map(|t| t["function"]["name"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
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
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                })
                .await;

            // Real tokens flow from the generation thread through the
            // channel; a closed channel (client gone) cancels generation.
            let tx_tokens = tx.clone();
            let id2 = id.clone();
            let model2 = model.clone();
            // Shared with the post-generation flush: a short reply that
            // never opens a <think> block (the template prefilled an
            // empty one) used to be swallowed whole — the filter waited
            // for a </think> that never comes.
            let filter_shared = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
            let filter_cb = filter_shared.clone();
            let mut filter_passthrough = req.thinking() != Some(false);
            // Tool-call holdback: once the model opens a <tool_call>
            // block, nothing more goes out as content — the calls are
            // parsed whole at the end and shipped as a tool_calls delta.
            // Until the marker is certain, the last few characters stay
            // buffered so a marker split across tokens cannot leak.
            let tools_active = req.effective_tools().is_some();
            let mut tool_tail = String::new();
            let mut tool_holding = false;
            const MARK: &str = "<tool_call>";

            let callback: cortiq_engine::TokenCallback = Box::new(move |token: &str| {
                if filter_passthrough {
                    if tools_active {
                        if tool_holding {
                            return !tx_tokens.is_closed();
                        }
                        tool_tail.push_str(token);
                        if let Some(pos) = tool_tail.find(MARK) {
                            tool_holding = true;
                            let before = tool_tail[..pos].to_string();
                            if !before.is_empty() {
                                let chunk = streaming::token_chunk(&id2, &model2, &before, created);
                                return tx_tokens.blocking_send(chunk).is_ok();
                            }
                            return !tx_tokens.is_closed();
                        }
                        // Flush all but a marker's worth of tail.
                        if tool_tail.len() > MARK.len() {
                            let cut = tool_tail.len() - (MARK.len() - 1);
                            let safe_cut = (0..=cut)
                                .rev()
                                .find(|&c| tool_tail.is_char_boundary(c))
                                .unwrap_or(0);
                            if safe_cut > 0 {
                                let out: String = tool_tail.drain(..safe_cut).collect();
                                let chunk = streaming::token_chunk(&id2, &model2, &out, created);
                                return tx_tokens.blocking_send(chunk).is_ok();
                            }
                        }
                        return !tx_tokens.is_closed();
                    }
                    let chunk = streaming::token_chunk(&id2, &model2, token, created);
                    return tx_tokens.blocking_send(chunk).is_ok();
                }
                let mut filter_buf = filter_cb.lock().expect("think filter buf");
                filter_buf.push_str(token);
                if let Some(pos) = filter_buf.find("</think>") {
                    let tail = filter_buf[pos + "</think>".len()..].to_string();
                    filter_buf.clear();
                    filter_passthrough = true;
                    let tail_trimmed = tail.trim_start_matches('\n');
                    if !tail_trimmed.is_empty() {
                        let chunk = streaming::token_chunk(&id2, &model2, tail_trimmed, created);
                        return tx_tokens.blocking_send(chunk).is_ok();
                    }
                    return true;
                }
                if filter_buf.len() > 100 && !filter_buf.contains("<think>") {
                    let b = std::mem::take(&mut *filter_buf);
                    filter_passthrough = true;
                    let chunk = streaming::token_chunk(&id2, &model2, &b, created);
                    return tx_tokens.blocking_send(chunk).is_ok();
                }
                true
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
                    // End-of-generation flush of the think filter, by the
                    // filter's own rules: a buffer that never opened a
                    // <think> block IS the answer; a closed block ships
                    // its tail; an unterminated block stays private.
                    let leftover = std::mem::take(&mut *filter_shared.lock().expect("filter buf"));
                    if !leftover.is_empty() {
                        let out = if !leftover.contains("<think>") {
                            leftover
                        } else if let Some(pos) = leftover.find("</think>") {
                            leftover[pos + "</think>".len()..]
                                .trim_start_matches('\n')
                                .to_string()
                        } else {
                            String::new()
                        };
                        if !out.is_empty() {
                            let _ = tx
                                .send(streaming::token_chunk(&id, &model, &out, created))
                                .await;
                        }
                    }
                    state2
                        .runtime
                        .record_generation(result.tokens_generated, elapsed_ms, elapsed_ms)
                        .await;
                    let (plain2, mut calls) = extract_tool_calls(&result.text);
                    if calls.is_empty() {
                        if let Some(c) = bare_call_fallback(&plain2, &tool_names) {
                            calls = vec![c];
                        }
                    }
                    let finish = if calls.is_empty() {
                        result.finish_reason.clone()
                    } else {
                        let _ = tx
                            .send(streaming::tool_calls_chunk(
                                &id,
                                &model,
                                serde_json::Value::Array(
                                    calls
                                        .into_iter()
                                        .enumerate()
                                        .map(|(i, mut c)| {
                                            c["index"] = serde_json::json!(i);
                                            c
                                        })
                                        .collect(),
                                ),
                                created,
                            ))
                            .await;
                        "tool_calls".to_string()
                    };
                    // exact counts ahead of the finish chunk (OpenAI include_usage shape)
                    let _ = tx
                        .send(streaming::usage_chunk(
                            &id,
                            &model,
                            created,
                            result.prompt_tokens as u32,
                            result.tokens_generated as u32,
                        ))
                        .await;
                    let _ = tx
                        .send(streaming::finish_chunk(&id, &model, &finish, created))
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

        let content = if req.thinking() == Some(false) {
            strip_think_block(&result.text)
        } else {
            result.text.clone()
        };

        let (mut plain, mut calls) = extract_tool_calls(&content);
        if calls.is_empty() {
            if let Some(names) = req.effective_tools().map(|ts| {
                ts.iter()
                    .filter_map(|t| t["function"]["name"].as_str().map(String::from))
                    .collect::<Vec<_>>()
            }) {
                if let Some(c) = bare_call_fallback(&plain, &names) {
                    calls = vec![c];
                    plain = String::new();
                }
            }
        }
        let made_calls = !calls.is_empty();
        Json(ChatCompletionsResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created,
            model: req.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    // OpenAI shape: a pure tool-call turn has null content.
                    content: if made_calls && plain.is_empty() {
                        None
                    } else {
                        Some(plain.into())
                    },
                    tool_calls: made_calls.then_some(serde_json::Value::Array(calls)),
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: if made_calls {
                    "tool_calls".to_string()
                } else {
                    result.finish_reason.clone()
                },
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

/// Small-model fallback: the whole reply is ONE bare JSON object that
/// names a REQUESTED tool. Qwen-family minis often emit the call
/// without its <tool_call> wrapper; vLLM and llama.cpp both accept
/// this shape, and refusing it here would fail every agent loop on a
/// small model while a human can see the call sitting in the text.
/// Conditions are strict on purpose: tools were requested, the text
/// parses as a single object, `name` is a string matching a declared
/// tool, and `arguments` (when present) is an object.
fn bare_call_fallback(text: &str, allowed: &[String]) -> Option<serde_json::Value> {
    let t = text.trim();
    if !t.starts_with('{') || !t.ends_with('}') {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(t).ok()?;
    let name = v.get("name")?.as_str()?;
    if !allowed.iter().any(|a| a == name) {
        return None;
    }
    let args = v.get("arguments").cloned().unwrap_or(serde_json::json!({}));
    if !args.is_object() {
        return None;
    }
    Some(serde_json::json!({
        "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
        "type": "function",
        "function": {
            "name": name,
            "arguments": serde_json::to_string(&args).unwrap_or_else(|_| "{}".into()),
        }
    }))
}

/// Nanbeige's XML tool grammar, normalised to the JSON shape:
/// `<function=NAME>\n<parameter=K>\nV\n</parameter>...</function>`.
/// Parameter values keep inner newlines (the format allows multi-line
/// values); the surrounding single newline the grammar inserts is
/// trimmed.
fn parse_xml_function(body: &str) -> Option<serde_json::Value> {
    let t = body.trim();
    let name_start = t.find("<function=")? + "<function=".len();
    let name_end = t[name_start..].find(['>', '\n'])? + name_start;
    let name = t[name_start..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut rest = &t[name_end..];
    while let Some(ps) = rest.find("<parameter=") {
        let key_start = ps + "<parameter=".len();
        let key_end = rest[key_start..].find('>')? + key_start;
        let key = rest[key_start..key_end].trim().to_string();
        let val_start = key_end + 1;
        let val_end = rest[val_start..].find("</parameter>")? + val_start;
        let val = rest[val_start..val_end]
            .strip_prefix('\n')
            .unwrap_or(&rest[val_start..val_end])
            .strip_suffix('\n')
            .unwrap_or(&rest[val_start..val_end])
            .to_string();
        args.insert(key, serde_json::Value::String(val));
        rest = &rest[val_end + "</parameter>".len()..];
    }
    Some(serde_json::json!({"name": name, "arguments": args}))
}

/// Extract `<tool_call>{...}</tool_call>` blocks from generated text —
/// the format every Qwen-family template (Nanbeige included) trains the
/// model to emit. Returns the text OUTSIDE the blocks and the calls in
/// OpenAI shape. `arguments` stays a STRING of JSON per the OpenAI
/// contract; a block whose body does not parse as JSON is left in the
/// text rather than shipped as a broken call — a client can read prose,
/// but it cannot execute garbage.
fn extract_tool_calls(text: &str) -> (String, Vec<serde_json::Value>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut rest = text;
    let mut plain = String::new();
    let mut calls = Vec::new();
    while let Some(i) = rest.find(OPEN) {
        let Some(j) = rest[i + OPEN.len()..].find(CLOSE) else {
            break; // unterminated block: keep as text (truncated output)
        };
        let body = rest[i + OPEN.len()..i + OPEN.len() + j].trim();
        let after = &rest[i + OPEN.len() + j + CLOSE.len()..];
        // Two trained grammars share the <tool_call> wrapper: the JSON
        // object, and Nanbeige's XML `<function=name><parameter=k>v...`.
        // Parse whichever arrived.
        let parsed = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .or_else(|| parse_xml_function(body));
        match parsed {
            Some(v) if v.get("name").map(|n| n.is_string()) == Some(true) => {
                plain.push_str(&rest[..i]);
                let args = v.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                calls.push(serde_json::json!({
                    "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                    "type": "function",
                    "function": {
                        "name": v["name"],
                        "arguments": serde_json::to_string(&args).unwrap_or_else(|_| "{}".into()),
                    }
                }));
            }
            _ => {
                // Not a call: keep the whole block verbatim as text.
                plain.push_str(&rest[..i + OPEN.len() + j + CLOSE.len()]);
            }
        }
        rest = after;
    }
    plain.push_str(rest);
    (plain.trim().to_string(), calls)
}

fn strip_think_block(s: &str) -> String {
    let mut rest = s;
    if let Some(pos) = rest.find("</think>") {
        rest = &rest[pos + "</think>".len()..];
    } else if rest.starts_with("<think>") {
        return String::new();
    }
    rest.trim_start_matches('\n').to_string()
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

    #[test]
    fn tool_calls_extract_single() {
        let (text, calls) = extract_tool_calls(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
        );
        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        // arguments is a STRING of JSON per the OpenAI contract
        let args: serde_json::Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
        assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
    }

    #[test]
    fn tool_calls_extract_text_and_multiple() {
        let (text, calls) = extract_tool_calls(
            "Let me check both.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": 1}}\n</tool_call>",
        );
        assert_eq!(text, "Let me check both.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1]["function"]["name"], "b");
    }

    #[test]
    fn tool_calls_malformed_body_stays_text() {
        let (text, calls) = extract_tool_calls("<tool_call>\nnot json at all\n</tool_call> done");
        assert!(calls.is_empty());
        assert!(
            text.contains("not json at all"),
            "broken call must stay readable text"
        );
    }

    #[test]
    fn tool_calls_unterminated_stays_text() {
        let (text, calls) = extract_tool_calls("<tool_call>\n{\"name\": \"a\"");
        assert!(calls.is_empty());
        assert!(
            text.contains("<tool_call>"),
            "truncated output must not vanish"
        );
    }

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

    /// Cline / Roo-style clients send `content` as a block array once they
    /// attach file context. Both shapes must deserialize, and the array must
    /// flatten to the same prompt text a flat string would give.
    #[test]
    fn content_accepts_both_a_string_and_a_block_array() {
        let flat: ChatMessage =
            serde_json::from_str(r#"{"role":"user","content":"hello"}"#).unwrap();
        assert_eq!(flat.content.as_ref().unwrap().text(), "hello");

        let blocks: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[
                 {"type":"text","text":"file context"},
                 {"type":"text","text":"the question"}]}"#,
        )
        .unwrap();
        assert_eq!(
            blocks.content.as_ref().unwrap().text(),
            "file context\nthe question"
        );

        // A non-text block must not fail the turn — it contributes nothing.
        let mixed: ChatMessage = serde_json::from_str(
            r#"{"role":"user","content":[
                 {"type":"image_url","image_url":{"url":"data:x"}},
                 {"type":"text","text":"describe"}]}"#,
        )
        .unwrap();
        assert_eq!(mixed.content.as_ref().unwrap().text(), "describe");

        // And a whole request round-trips, which is what 422'd before.
        let req: ChatCompletionsRequest = serde_json::from_str(
            r#"{"model":"m","messages":[
                 {"role":"system","content":[{"type":"text","text":"sys"}]},
                 {"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(req.messages[0].content.as_ref().unwrap().text(), "sys");
        assert_eq!(req.messages[1].content.as_ref().unwrap().text(), "hi");
    }
}
