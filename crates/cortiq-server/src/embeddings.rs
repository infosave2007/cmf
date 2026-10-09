//! `POST /v1/embeddings` for EmbeddingGemma 2 files — the OpenAI embeddings
//! API over `cortiq_engine::egemma2`.
//!
//! An embedding file is an encoder: no pipeline, no KV cache, no chat. `cortiq
//! serve` detects the architecture and runs this router instead of the LLM
//! one.
//!
//! Request (OpenAI): `input` is a string, an array of strings, an array of
//! token ids or an array of token-id arrays; `dimensions` is 768 (default) or
//! a Matryoshka prefix 512 | 256 | 128 (re-normalized); `encoding_format` is
//! `float` (default) or `base64` (little-endian f32). `model` and `user` are
//! accepted and ignored.
//!
//! Cortiq extensions — at the top level or inside a `cortiq` object:
//! * `prompt_name` (alias `task`): one of the model's task prompts —
//!   `SearchQuery`, `Document`, `QuestionAnswering`, `FactChecking`,
//!   `CodeRetrieval`, `Classification`, `Clustering`, `SentenceSimilarity`
//!   … (`GET /v1/embeddings/prompts` lists them);
//! * `title`: the title of a Document (`title: {title} | text: …`);
//! * `prompt`: a raw prefix instead of a named prompt.
//!
//! An `input` array may also hold objects `{"text", "prompt_name"|"task",
//! "title", "prompt"}`, so one request can mix queries and documents.
//! Texts are tokenized as `[BOS] + prompt + text + [EOS]` and capped at 8192
//! tokens; token-id inputs get BOS/EOS added when missing and are not
//! prompted.

use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use base64::Engine as _;
use cortiq_core::CmfModel;
use cortiq_engine::egemma2::{EmbeddingGemma2, MATRYOSHKA_DIMS, Prompts, TextInput, matryoshka};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;

/// OpenAI's own cap on inputs per request.
const MAX_INPUTS: usize = 2048;

pub struct EmbedState {
    pub enc: Arc<EmbeddingGemma2>,
    pub model_id: String,
    sem: Arc<Semaphore>,
}

impl EmbedState {
    pub fn new(enc: EmbeddingGemma2, model_id: String) -> Self {
        EmbedState {
            enc: Arc::new(enc),
            model_id,
            // one forward at a time: it already uses every core
            sem: Arc::new(Semaphore::new(1)),
        }
    }
}

/// One parsed input.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Text(TextInputSpec),
    Ids(Vec<u32>),
}

/// A text input with its prompt options (owned, comparable).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextInputSpec {
    pub text: String,
    pub prompt_name: Option<String>,
    pub title: Option<String>,
    pub prompt: Option<String>,
}

impl TextInputSpec {
    fn to_input(&self) -> TextInput {
        TextInput {
            text: self.text.clone(),
            prompt_name: self.prompt_name.clone(),
            title: self.title.clone(),
            prompt: self.prompt.clone(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Parsed {
    pub items: Vec<Item>,
    pub dim: usize,
    pub base64: bool,
}

/// An OpenAI-shaped 400.
#[derive(Debug, PartialEq)]
pub struct ApiError {
    pub message: String,
    pub param: Option<&'static str>,
}

fn bad(message: impl Into<String>, param: Option<&'static str>) -> ApiError {
    ApiError {
        message: message.into(),
        param,
    }
}

fn error_response(status: StatusCode, e: ApiError, kind: &str) -> Response {
    (
        status,
        Json(json!({"error": {
            "message": e.message,
            "type": kind,
            "param": e.param,
            "code": Value::Null,
        }})),
    )
        .into_response()
}

fn opt_str(v: &Value, key: &'static str) -> Result<Option<String>, ApiError> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(bad(format!("'{key}' must be a string"), Some(key))),
    }
}

/// The prompt options at one level (top level, `cortiq`, or an item).
fn options(v: &Value) -> Result<TextInputSpec, ApiError> {
    let pn = opt_str(v, "prompt_name")?;
    let task = opt_str(v, "task")?;
    if pn.is_some() && task.is_some() && pn != task {
        return Err(bad(
            "'prompt_name' and 'task' disagree; give one",
            Some("prompt_name"),
        ));
    }
    Ok(TextInputSpec {
        text: String::new(),
        prompt_name: pn.or(task),
        title: opt_str(v, "title")?,
        prompt: opt_str(v, "prompt")?,
    })
}

/// `b` over `a`, field by field.
fn overlay(a: &TextInputSpec, b: &TextInputSpec) -> TextInputSpec {
    TextInputSpec {
        text: String::new(),
        prompt_name: b.prompt_name.clone().or_else(|| a.prompt_name.clone()),
        title: b.title.clone().or_else(|| a.title.clone()),
        prompt: b.prompt.clone().or_else(|| a.prompt.clone()),
    }
}

fn token_ids(a: &[Value]) -> Option<Vec<u32>> {
    a.iter()
        .map(|x| {
            x.as_u64()
                .filter(|&t| t <= u32::MAX as u64)
                .map(|t| t as u32)
        })
        .collect()
}

/// Parse an embeddings request body.
pub fn parse_request(body: &Value, prompts: &Prompts) -> Result<Parsed, ApiError> {
    if !body.is_object() {
        return Err(bad("the request body must be a JSON object", None));
    }
    let mut defaults = options(body)?;
    if let Some(c) = body.get("cortiq") {
        if !c.is_object() {
            return Err(bad("'cortiq' must be an object", Some("cortiq")));
        }
        defaults = overlay(&options(c)?, &defaults);
    }
    let dim = match body.get("dimensions") {
        None | Some(Value::Null) => 768,
        Some(v) => {
            let d = v
                .as_u64()
                .ok_or_else(|| bad("'dimensions' must be an integer", Some("dimensions")))?
                as usize;
            if !MATRYOSHKA_DIMS.contains(&d) {
                return Err(bad(
                    format!("dimensions {d} is not supported: use 768, 512, 256 or 128"),
                    Some("dimensions"),
                ));
            }
            d
        }
    };
    let base64 = match body.get("encoding_format").and_then(|v| v.as_str()) {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(bad(
                format!("encoding_format '{other}': use float or base64"),
                Some("encoding_format"),
            ));
        }
    };
    let text_item = |text: &str, opts: &TextInputSpec| -> Result<Item, ApiError> {
        let spec = TextInputSpec {
            text: text.to_string(),
            ..opts.clone()
        };
        // validate the prompt/title now: a typo is a 400, not a 500
        prompts
            .format(
                &spec.text,
                spec.prompt_name.as_deref(),
                spec.title.as_deref(),
                spec.prompt.as_deref(),
            )
            .map_err(|e| bad(e, Some("prompt_name")))?;
        Ok(Item::Text(spec))
    };
    let input = body
        .get("input")
        .ok_or_else(|| bad("'input' is required", Some("input")))?;
    let items: Vec<Item> = match input {
        Value::String(s) => vec![text_item(s, &defaults)?],
        Value::Array(a) if a.is_empty() => return Err(bad("'input' is empty", Some("input"))),
        Value::Array(a) if a.iter().all(|x| x.is_number()) => {
            vec![Item::Ids(token_ids(a).ok_or_else(|| {
                bad("token ids must be non-negative integers", Some("input"))
            })?)]
        }
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::String(s) => text_item(s, &defaults),
                Value::Array(ids) => token_ids(ids)
                    .filter(|v| !v.is_empty())
                    .map(Item::Ids)
                    .ok_or_else(|| {
                        bad(
                            "token-id arrays must hold non-negative integers",
                            Some("input"),
                        )
                    }),
                Value::Object(_) => {
                    let text = x.get("text").and_then(|t| t.as_str()).ok_or_else(|| {
                        bad("an input object needs a 'text' string", Some("input"))
                    })?;
                    let o = overlay(&defaults, &options(x)?);
                    text_item(text, &o)
                }
                _ => Err(bad(
                    "each input must be a string, a token-id array or an object",
                    Some("input"),
                )),
            })
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(bad("'input' must be a string or an array", Some("input")));
        }
    };
    if items.len() > MAX_INPUTS {
        return Err(bad(
            format!(
                "{} inputs > the {MAX_INPUTS} allowed per request",
                items.len()
            ),
            Some("input"),
        ));
    }
    Ok(Parsed { items, dim, base64 })
}

fn f32_base64(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

async fn embeddings(State(st): State<Arc<EmbedState>>, body: Bytes) -> Response {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                bad(format!("invalid JSON: {e}"), None),
                "invalid_request_error",
            );
        }
    };
    let parsed = match parse_request(&v, st.enc.prompts()) {
        Ok(p) => p,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e, "invalid_request_error"),
    };
    let Ok(_permit) = st.sem.clone().acquire_owned().await else {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            bad("server is shutting down", None),
            "server_error",
        );
    };
    let enc = st.enc.clone();
    let job = tokio::task::spawn_blocking(move || -> Result<(Vec<Vec<f32>>, usize), ApiError> {
        let mut seqs = Vec::with_capacity(parsed.items.len());
        for (i, it) in parsed.items.iter().enumerate() {
            let ids = match it {
                Item::Text(spec) => enc
                    .input_ids(&spec.to_input())
                    .map_err(|e| bad(format!("input {i}: {e}"), Some("input")))?,
                Item::Ids(raw) => {
                    let mut ids = raw.clone();
                    if ids.first() != Some(&enc.bos) {
                        ids.insert(0, enc.bos);
                    }
                    if ids.last() != Some(&enc.eos) {
                        ids.push(enc.eos);
                    }
                    if ids.len() > enc.max_tokens {
                        return Err(bad(
                            format!(
                                "input {i}: {} tokens > the model's {} token context",
                                ids.len(),
                                enc.max_tokens
                            ),
                            Some("input"),
                        ));
                    }
                    ids
                }
            };
            seqs.push(ids);
        }
        let tokens = seqs.iter().map(|s| s.len()).sum();
        let full = enc.embed_ids(&seqs).map_err(|e| bad(e, Some("input")))?;
        let out = full
            .iter()
            .map(|v| matryoshka(v, parsed.dim))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| bad(e, Some("dimensions")))?;
        Ok((out, tokens))
    })
    .await;
    let (vecs, tokens) = match job {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return error_response(StatusCode::BAD_REQUEST, e, "invalid_request_error"),
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                bad(format!("embedding failed: {e}"), None),
                "server_error",
            );
        }
    };
    let data: Vec<Value> = vecs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let emb = if parsed.base64 {
                json!(f32_base64(v))
            } else {
                json!(v)
            };
            json!({"object": "embedding", "index": i, "embedding": emb})
        })
        .collect();
    Json(json!({
        "object": "list",
        "data": data,
        "model": st.model_id,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }))
    .into_response()
}

async fn models(State(st): State<Arc<EmbedState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": st.model_id,
            "object": "model",
            "created": chrono::Utc::now().timestamp(),
            "owned_by": "cortiq",
        }],
    }))
}

async fn prompts(State(st): State<Arc<EmbedState>>) -> Json<Value> {
    let p = st.enc.prompts();
    let map: serde_json::Map<String, Value> = p
        .names()
        .into_iter()
        .map(|n| (n.to_string(), json!(p.get(n).unwrap_or(""))))
        .collect();
    Json(json!({"prompts": map, "dimensions": MATRYOSHKA_DIMS, "max_tokens": st.enc.max_tokens}))
}

async fn healthz(State(st): State<Arc<EmbedState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": st.model_id,
        "capabilities": {"embeddings": true, "dimensions": MATRYOSHKA_DIMS},
    }))
}

/// The embeddings router.
pub fn router(state: Arc<EmbedState>) -> Router {
    Router::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/embeddings/prompts", get(prompts))
        .route("/v1/models", get(models))
        .route("/healthz", get(healthz))
        .layer(DefaultBodyLimit::max(64 << 20))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Load the encoder from `model` and serve it on `addr`.
pub async fn serve(
    model: Arc<CmfModel>,
    model_path: &str,
    addr: std::net::SocketAddr,
) -> anyhow::Result<()> {
    let pool = cortiq_engine::pool::Pool::from_env();
    let enc = tokio::task::spawn_blocking(move || EmbeddingGemma2::load(&model, pool))
        .await?
        .map_err(anyhow::Error::msg)?;
    let model_id = std::path::Path::new(model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "embeddinggemma-2".into());
    println!(
        "    EmbeddingGemma 2 ({}), {} dims, {} task prompts",
        if enc.quant.is_empty() {
            "?"
        } else {
            &enc.quant
        },
        enc.dim,
        enc.prompts().names().len()
    );
    println!("  Embeddings API: POST http://{addr}/v1/embeddings (model \"{model_id}\")");
    let app = router(Arc::new(EmbedState::new(enc, model_id)));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> Prompts {
        Prompts::default()
    }

    #[test]
    fn parses_openai_shapes() {
        let r = parse_request(&json!({"input": "hi", "model": "x"}), &p()).unwrap();
        assert_eq!(r.dim, 768);
        assert!(!r.base64);
        assert_eq!(
            r.items,
            vec![Item::Text(TextInputSpec {
                text: "hi".into(),
                ..Default::default()
            })]
        );
        let r = parse_request(
            &json!({"input": ["a", "b"], "dimensions": 256, "encoding_format": "base64"}),
            &p(),
        )
        .unwrap();
        assert_eq!(r.items.len(), 2);
        assert_eq!(r.dim, 256);
        assert!(r.base64);
        let r = parse_request(&json!({"input": [2, 100, 1]}), &p()).unwrap();
        assert_eq!(r.items, vec![Item::Ids(vec![2, 100, 1])]);
        let r = parse_request(&json!({"input": [[5, 6], [7]]}), &p()).unwrap();
        assert_eq!(r.items, vec![Item::Ids(vec![5, 6]), Item::Ids(vec![7])]);
    }

    #[test]
    fn parses_cortiq_extensions() {
        let r = parse_request(&json!({"input": "q", "prompt_name": "SearchQuery"}), &p()).unwrap();
        let Item::Text(t) = &r.items[0] else { panic!() };
        assert_eq!(t.prompt_name.as_deref(), Some("SearchQuery"));
        let r = parse_request(
            &json!({"input": "d", "cortiq": {"task": "Document", "title": "T"}}),
            &p(),
        )
        .unwrap();
        let Item::Text(t) = &r.items[0] else { panic!() };
        assert_eq!(t.prompt_name.as_deref(), Some("Document"));
        assert_eq!(t.title.as_deref(), Some("T"));
        // mixed items, item options over request options
        let r = parse_request(
            &json!({"input": ["q", {"text": "d", "prompt_name": "Document", "title": "X"}], "task": "SearchQuery"}),
            &p(),
        )
        .unwrap();
        let (Item::Text(a), Item::Text(b)) = (&r.items[0], &r.items[1]) else {
            panic!()
        };
        assert_eq!(a.prompt_name.as_deref(), Some("SearchQuery"));
        assert_eq!(b.prompt_name.as_deref(), Some("Document"));
        assert_eq!(b.title.as_deref(), Some("X"));
    }

    #[test]
    fn rejects_bad_requests() {
        for body in [
            json!({}),
            json!({"input": []}),
            json!({"input": "x", "dimensions": 300}),
            json!({"input": "x", "encoding_format": "hex"}),
            json!({"input": "x", "prompt_name": "Nope"}),
            json!({"input": "x", "prompt_name": "SearchQuery", "title": "T"}),
            json!({"input": [1, -2]}),
            json!({"input": [true]}),
            json!({"input": 5}),
            json!({"input": "x", "prompt_name": "A", "task": "B"}),
            json!("text"),
        ] {
            assert!(parse_request(&body, &p()).is_err(), "{body}");
        }
    }

    #[test]
    fn base64_is_little_endian_f32() {
        let s = f32_base64(&[1.0, -2.0]);
        let b = base64::engine::general_purpose::STANDARD.decode(s).unwrap();
        assert_eq!(&b[..4], &1.0f32.to_le_bytes());
        assert_eq!(&b[4..], &(-2.0f32).to_le_bytes());
    }
}
