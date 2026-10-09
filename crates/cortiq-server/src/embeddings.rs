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
//!
//! **Audio.** An input object with `audio` embeds sound into the same space:
//! `{"audio": "data:audio/wav;base64,…"}`, a local path, an `http(s)://`
//! URL, `{"data": <base64>, "format": "wav"}`, or a list of those; the
//! OpenAI chat part `{"type": "input_audio", "input_audio": {"data",
//! "format"}}` is accepted too. WAV is decoded natively (any rate and
//! channel count: mixed to mono, resampled to 16 kHz); other formats need
//! `ffmpeg` on the server. Clips are cut at 30 s (750 tokens). Audio alone
//! takes no task prompt; with `text`, the text carries one `<|audio|>` per
//! clip (`{"text": "Narration: <|audio|>", "audio": […]}`) and the prompt
//! options apply to it as to any text. The audio tower loads on the first
//! audio request.

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
use cortiq_engine::egemma2_audio::{self as ea, AudioInput, AudioTower};
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;

/// OpenAI's own cap on inputs per request.
const MAX_INPUTS: usize = 2048;

pub struct EmbedState {
    pub enc: Arc<EmbeddingGemma2>,
    pub model_id: String,
    sem: Arc<Semaphore>,
    /// the file, for the towers that load on first use
    model: Option<Arc<CmfModel>>,
    audio: OnceLock<Result<AudioTower, String>>,
}

impl EmbedState {
    pub fn new(enc: EmbeddingGemma2, model_id: String) -> Self {
        EmbedState {
            enc: Arc::new(enc),
            model_id,
            // one forward at a time: it already uses every core
            sem: Arc::new(Semaphore::new(1)),
            model: None,
            audio: OnceLock::new(),
        }
    }

    /// Keep the file so the audio tower can load on the first audio input.
    pub fn with_model(mut self, model: Arc<CmfModel>) -> Self {
        self.model = Some(model);
        self
    }

    fn has_audio(&self) -> bool {
        self.model.as_deref().is_some_and(ea::has_audio)
    }

    /// The audio tower, loaded once.
    fn audio_tower(&self) -> Result<&AudioTower, String> {
        let model = self
            .model
            .as_ref()
            .ok_or("this server was started without audio support")?;
        self.audio
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let r = AudioTower::load(model, cortiq_engine::pool::Pool::from_env());
                if r.is_ok() {
                    eprintln!("  audio tower loaded in {:.2}s", t0.elapsed().as_secs_f64());
                }
                r
            })
            .as_ref()
            .map_err(|e| e.clone())
    }
}

/// One parsed input.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Text(TextInputSpec),
    Ids(Vec<u32>),
    /// audio sources (decoded in the job) with optional `<|audio|>` text
    Audio(AudioSpec),
}

/// An audio input: the sources of its clips, in order, and the text that
/// places them (`None`: the clips alone, unprompted).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSpec {
    pub text: Option<TextInputSpec>,
    pub sources: Vec<Value>,
}

/// The audio sources of an input object, if it has any.
fn audio_sources(x: &Value) -> Result<Option<Vec<Value>>, ApiError> {
    if x.get("type").and_then(Value::as_str) == Some("input_audio")
        || x.get("input_audio").is_some()
    {
        let inner = x
            .get("input_audio")
            .ok_or_else(|| bad("an input_audio part needs 'input_audio'", Some("input")))?;
        return Ok(Some(vec![json!({ "input_audio": inner })]));
    }
    let a = match x.get("audio") {
        None | Some(Value::Null) => return Ok(None),
        Some(a) => a,
    };
    let list = match a {
        Value::Array(v) => v.clone(),
        v @ (Value::String(_) | Value::Object(_)) => vec![v.clone()],
        _ => {
            return Err(bad(
                "'audio' must be a source (data URL, path, URL or {\"data\"}) or a list of them",
                Some("input"),
            ));
        }
    };
    if list.is_empty() {
        return Err(bad("'audio' is empty", Some("input")));
    }
    if !list.iter().all(|v| v.is_string() || v.is_object()) {
        return Err(bad(
            "each audio source must be a string or an object",
            Some("input"),
        ));
    }
    Ok(Some(list))
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
    let object_item = |x: &Value| -> Result<Item, ApiError> {
        if let Some(sources) = audio_sources(x)? {
            let text = match x.get("text") {
                None | Some(Value::Null) => None,
                Some(Value::String(t)) => {
                    let n = t.matches(ea::PLACEHOLDER).count();
                    if n != sources.len() {
                        return Err(bad(
                            format!(
                                "the text holds {n} {} placeholder(s) for {} audio clip(s)",
                                ea::PLACEHOLDER,
                                sources.len()
                            ),
                            Some("input"),
                        ));
                    }
                    let o = overlay(&defaults, &options(x)?);
                    let Item::Text(spec) = text_item(t, &o)? else {
                        unreachable!("text_item returns text")
                    };
                    Some(spec)
                }
                Some(_) => return Err(bad("'text' must be a string", Some("input"))),
            };
            return Ok(Item::Audio(AudioSpec { text, sources }));
        }
        let text = x.get("text").and_then(|t| t.as_str()).ok_or_else(|| {
            bad(
                "an input object needs a 'text' string (or 'audio')",
                Some("input"),
            )
        })?;
        let o = overlay(&defaults, &options(x)?);
        text_item(text, &o)
    };
    let items: Vec<Item> = match input {
        Value::String(s) => vec![text_item(s, &defaults)?],
        Value::Object(_) => vec![object_item(input)?],
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
                Value::Object(_) => object_item(x),
                _ => Err(bad(
                    "each input must be a string, a token-id array or an object",
                    Some("input"),
                )),
            })
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(bad(
                "'input' must be a string, an array or an object",
                Some("input"),
            ));
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
    let state = st.clone();
    let job = tokio::task::spawn_blocking(move || -> Result<(Vec<Vec<f32>>, usize), ApiError> {
        let n_items = parsed.items.len();
        let mut text_idx = Vec::new();
        let mut seqs = Vec::new();
        let mut audio_idx = Vec::new();
        let mut audio_in: Vec<AudioInput> = Vec::new();
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
                Item::Audio(spec) => {
                    let clips = spec
                        .sources
                        .iter()
                        .map(ea::load_audio_source)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| bad(format!("input {i}: {e}"), Some("input")))?;
                    audio_in.push(AudioInput {
                        text: spec.text.as_ref().map(TextInputSpec::to_input),
                        clips,
                    });
                    audio_idx.push(i);
                    continue;
                }
            };
            seqs.push(ids);
            text_idx.push(i);
        }
        let mut tokens: usize = seqs.iter().map(|s| s.len()).sum();
        let mut full: Vec<Vec<f32>> = vec![Vec::new(); n_items];
        if !seqs.is_empty() {
            let v = enc.embed_ids(&seqs).map_err(|e| bad(e, Some("input")))?;
            for (v, &i) in v.into_iter().zip(&text_idx) {
                full[i] = v;
            }
        }
        if !audio_in.is_empty() {
            let tower = state.audio_tower().map_err(|e| bad(e, Some("input")))?;
            for (k, a) in audio_in.iter().enumerate() {
                let n: Vec<usize> = a.clips.iter().map(|c| ea::num_tokens(c.len())).collect();
                tokens += ea::input_ids(
                    &enc,
                    (tower.audio_token, tower.boa_token, tower.eoa_token),
                    a,
                    &n,
                )
                .map_err(|e| bad(format!("input {}: {e}", audio_idx[k]), Some("input")))?
                .len();
            }
            let v = ea::embed_audio_inputs(&enc, tower, &audio_in)
                .map_err(|e| bad(e, Some("input")))?;
            for (v, &i) in v.into_iter().zip(&audio_idx) {
                full[i] = v;
            }
        }
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
        "capabilities": {
            "embeddings": true,
            "dimensions": MATRYOSHKA_DIMS,
            "audio": st.has_audio(),
        },
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
    let m2 = model.clone();
    let enc = tokio::task::spawn_blocking(move || EmbeddingGemma2::load(&m2, pool))
        .await?
        .map_err(anyhow::Error::msg)?;
    let audio = ea::has_audio(&model);
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
    if audio {
        println!("    audio input: on (the tower loads on the first audio request)");
    }
    println!("  Embeddings API: POST http://{addr}/v1/embeddings (model \"{model_id}\")");
    let app = router(Arc::new(EmbedState::new(enc, model_id).with_model(model)));
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
    fn parses_audio_inputs() {
        let r = parse_request(
            &json!({"input": [{"audio": "data:audio/wav;base64,AA=="}, "a text"], "prompt_name": "SearchQuery"}),
            &p(),
        )
        .unwrap();
        // audio alone takes no prompt; the text keeps the request's
        assert_eq!(
            r.items[0],
            Item::Audio(AudioSpec {
                text: None,
                sources: vec![json!("data:audio/wav;base64,AA==")],
            })
        );
        let Item::Text(t) = &r.items[1] else { panic!() };
        assert_eq!(t.prompt_name.as_deref(), Some("SearchQuery"));
        // interleaved: one placeholder per clip, prompt options apply to the text
        let r = parse_request(
            &json!({"input": {"text": "A <|audio|> B <|audio|>", "audio": ["/a.wav", {"data": "AA==", "format": "wav"}], "task": "Clustering"}}),
            &p(),
        )
        .unwrap();
        let Item::Audio(a) = &r.items[0] else {
            panic!()
        };
        assert_eq!(a.sources.len(), 2);
        assert_eq!(
            a.text.as_ref().unwrap().prompt_name.as_deref(),
            Some("Clustering")
        );
        // the OpenAI chat part
        let r = parse_request(
            &json!({"input": [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]}),
            &p(),
        )
        .unwrap();
        assert!(
            matches!(&r.items[0], Item::Audio(AudioSpec { text: None, sources }) if sources.len() == 1)
        );
        for body in [
            json!({"input": [{"audio": []}]}),
            json!({"input": [{"audio": 5}]}),
            json!({"input": [{"audio": "x.wav", "text": "no placeholder"}]}),
            json!({"input": [{"audio": ["a.wav", "b.wav"], "text": "one <|audio|>"}]}),
            json!({"input": [{"audio": "x.wav", "text": 3}]}),
            json!({"input": [{"type": "input_audio"}]}),
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
