//! `POST /v1/embeddings` for EmbeddingGemma 2 files — the OpenAI embeddings
//! API over `cortiq_engine::egemma2`: texts, images, videos and interleaved
//! text + media, one 768-d space.
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
//! * `prompt`: a raw prefix instead of a named prompt;
//! * `image_tokens` / `video_tokens`: the soft-token budget per image /
//!   per video frame (70 | 140 | 280 | 560 | 1120; defaults 280 / 140).
//!
//! An `input` array may also hold:
//! * objects `{"text", "image", "video", "prompt_name"|"task", "title",
//!   "prompt", "image_tokens", "video_tokens"}` — `image` / `video` a source
//!   or a list of sources; a text with `<|image|>` / `<|video|>`
//!   placeholders interleaves them in order (without placeholders the media
//!   come first);
//! * content-part arrays, interleaved in the order given:
//!   `[{"type": "text", "text": …}, {"type": "image_url", "image_url":
//!   {"url": …}}, {"type": "video_url", "video_url": {"url": …}}]`
//!   (`input_image` / `image` / `video` with a `url` or `image_url` string
//!   are accepted too).
//!
//! A source is a `data:` URL (base64), an `http(s)://` URL, or — when the
//! server listens on loopback, or `CMF_EMBED_LOCAL_MEDIA=1` — a local path.
//! Videos are decoded with the `ffmpeg` executable (or a `.y4m` / frame
//! directory path), sampled at 1 fps, at most 32 frames.
//!
//! Task prompts follow sentence-transformers: a request-level prompt
//! applies to text-only inputs; an input with media takes a prompt only
//! from itself. Texts are tokenized as `[BOS] + prompt + text + [EOS]` and
//! capped at 8192 tokens; an input with media that does not fit is a 400.
//! Token-id inputs get BOS/EOS added when missing and are not prompted.

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
use cortiq_engine::egemma2::{MATRYOSHKA_DIMS, Prompts, TextInput, matryoshka};
use cortiq_engine::egemma2_mm::{IMAGE_PLACEHOLDER, MediaEncoder, MixedInput, VIDEO_PLACEHOLDER};
use cortiq_engine::egemma2_vision::{BUDGETS, decode_video};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;

/// OpenAI's own cap on inputs per request.
const MAX_INPUTS: usize = 2048;
/// Media items per request (each image is a ViT forward).
const MAX_MEDIA: usize = 64;

pub struct EmbedState {
    pub enc: Arc<MediaEncoder>,
    pub model_id: String,
    /// local paths are accepted as media sources
    pub local_media: bool,
    sem: Arc<Semaphore>,
}

impl EmbedState {
    pub fn new(enc: MediaEncoder, model_id: String, local_media: bool) -> Self {
        EmbedState {
            enc: Arc::new(enc),
            model_id,
            local_media,
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
    Mixed(MixedSpec),
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

/// An input with media: the text (placeholders laid out) and the sources.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MixedSpec {
    pub text: String,
    pub images: Vec<String>,
    pub videos: Vec<String>,
    pub prompt_name: Option<String>,
    pub title: Option<String>,
    pub prompt: Option<String>,
    pub image_tokens: Option<usize>,
    pub video_tokens: Option<usize>,
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

fn opt_budget(v: &Value, key: &'static str) -> Result<Option<usize>, ApiError> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(x) => {
            let b = x
                .as_u64()
                .map(|b| b as usize)
                .filter(|b| BUDGETS.contains(b))
                .ok_or_else(|| {
                    bad(
                        format!("'{key}' must be one of 70, 140, 280, 560, 1120"),
                        Some(key),
                    )
                })?;
            Ok(Some(b))
        }
    }
}

/// The options at one level (top level, `cortiq`, or an item).
#[derive(Debug, Clone, Default, PartialEq)]
struct Opts {
    prompt_name: Option<String>,
    title: Option<String>,
    prompt: Option<String>,
    image_tokens: Option<usize>,
    video_tokens: Option<usize>,
}

fn options(v: &Value) -> Result<Opts, ApiError> {
    let pn = opt_str(v, "prompt_name")?;
    let task = opt_str(v, "task")?;
    if pn.is_some() && task.is_some() && pn != task {
        return Err(bad(
            "'prompt_name' and 'task' disagree; give one",
            Some("prompt_name"),
        ));
    }
    Ok(Opts {
        prompt_name: pn.or(task),
        title: opt_str(v, "title")?,
        prompt: opt_str(v, "prompt")?,
        image_tokens: opt_budget(v, "image_tokens")?,
        video_tokens: opt_budget(v, "video_tokens")?,
    })
}

/// `b` over `a`, field by field.
fn overlay(a: &Opts, b: &Opts) -> Opts {
    Opts {
        prompt_name: b.prompt_name.clone().or_else(|| a.prompt_name.clone()),
        title: b.title.clone().or_else(|| a.title.clone()),
        prompt: b.prompt.clone().or_else(|| a.prompt.clone()),
        image_tokens: b.image_tokens.or(a.image_tokens),
        video_tokens: b.video_tokens.or(a.video_tokens),
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

/// A media source: a string, or `{"url": …}` (OpenAI's `image_url`).
fn source(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("url").and_then(|u| u.as_str()).map(str::to_string),
        _ => None,
    }
}

/// `image` / `video` of an input object: one source or a list.
fn sources(v: Option<&Value>, key: &'static str) -> Result<Vec<String>, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                source(x).ok_or_else(|| bad(format!("'{key}' entries must be sources"), Some(key)))
            })
            .collect(),
        Some(x) => Ok(vec![source(x).ok_or_else(|| {
            bad(
                format!("'{key}' must be a URL / path string or a list of them"),
                Some(key),
            )
        })?]),
    }
}

/// OpenAI-style content parts, interleaved in order into one input.
fn content_parts(parts: &[Value]) -> Result<MixedSpec, ApiError> {
    let mut texts: Vec<String> = Vec::new();
    let mut spec = MixedSpec::default();
    for p in parts {
        let kind = p
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| bad("each content part needs a 'type'", Some("input")))?;
        match kind {
            "text" | "input_text" => {
                let t = p
                    .get("text")
                    .and_then(|t| t.as_str())
                    .ok_or_else(|| bad("a text part needs 'text'", Some("input")))?;
                texts.push(t.to_string());
            }
            "image_url" | "input_image" | "image" => {
                let s = p
                    .get("image_url")
                    .and_then(source)
                    .or_else(|| p.get("url").and_then(source))
                    .or_else(|| p.get("image").and_then(source))
                    .ok_or_else(|| bad("an image part needs 'image_url'", Some("input")))?;
                spec.images.push(s);
                texts.push(IMAGE_PLACEHOLDER.to_string());
            }
            "video_url" | "input_video" | "video" => {
                let s = p
                    .get("video_url")
                    .and_then(source)
                    .or_else(|| p.get("url").and_then(source))
                    .or_else(|| p.get("video").and_then(source))
                    .ok_or_else(|| bad("a video part needs 'video_url'", Some("input")))?;
                spec.videos.push(s);
                texts.push(VIDEO_PLACEHOLDER.to_string());
            }
            other => {
                return Err(bad(
                    format!("content part type '{other}': use text, image_url or video_url"),
                    Some("input"),
                ));
            }
        }
    }
    // the processor's own layout: parts separated by a space
    spec.text = texts.join(" ");
    Ok(spec)
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
    let check_prompt = |text: &str, o: &Opts| -> Result<(), ApiError> {
        // validate the prompt/title now: a typo is a 400, not a 500
        prompts
            .format(
                text,
                o.prompt_name.as_deref(),
                o.title.as_deref(),
                o.prompt.as_deref(),
            )
            .map(|_| ())
            .map_err(|e| bad(e, Some("prompt_name")))
    };
    let text_item = |text: &str, o: &Opts| -> Result<Item, ApiError> {
        check_prompt(text, o)?;
        Ok(Item::Text(TextInputSpec {
            text: text.to_string(),
            prompt_name: o.prompt_name.clone(),
            title: o.title.clone(),
            prompt: o.prompt.clone(),
        }))
    };
    // media inputs: budgets inherit, prompts only from the item itself
    let mixed_item = |mut spec: MixedSpec, own: &Opts| -> Result<Item, ApiError> {
        if spec.images.is_empty() && spec.videos.is_empty() {
            let o = overlay(&defaults, own);
            return text_item(&spec.text, &o);
        }
        if own.prompt_name.is_some() || own.title.is_some() || own.prompt.is_some() {
            check_prompt(&spec.text, own)?;
        }
        spec.prompt_name = own.prompt_name.clone();
        spec.title = own.title.clone();
        spec.prompt = own.prompt.clone();
        spec.image_tokens = own.image_tokens.or(defaults.image_tokens);
        spec.video_tokens = own.video_tokens.or(defaults.video_tokens);
        Ok(Item::Mixed(spec))
    };
    let object_item = |x: &Value| -> Result<Item, ApiError> {
        let own = options(x)?;
        let images = sources(x.get("image").or_else(|| x.get("images")), "image")?;
        let images = if images.is_empty() {
            // a bare OpenAI part object {"type": "image_url", ...}
            sources(x.get("image_url"), "image_url")?
        } else {
            images
        };
        let videos = sources(x.get("video").or_else(|| x.get("videos")), "video")?;
        let videos = if videos.is_empty() {
            sources(x.get("video_url"), "video_url")?
        } else {
            videos
        };
        let text = match x.get("text") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(t)) => t.clone(),
            Some(_) => return Err(bad("'text' must be a string", Some("input"))),
        };
        if text.is_empty() && images.is_empty() && videos.is_empty() && x.get("text").is_none() {
            return Err(bad(
                "an input object needs 'text', 'image' or 'video'",
                Some("input"),
            ));
        }
        mixed_item(
            MixedSpec {
                text,
                images,
                videos,
                ..Default::default()
            },
            &own,
        )
    };
    let input = body
        .get("input")
        .ok_or_else(|| bad("'input' is required", Some("input")))?;
    let items: Vec<Item> = match input {
        Value::String(s) => vec![text_item(s, &defaults)?],
        Value::Object(_) => vec![object_item(input)?],
        Value::Array(a) if a.is_empty() => return Err(bad("'input' is empty", Some("input"))),
        Value::Array(a) if a.iter().all(|x| x.is_number()) => {
            vec![Item::Ids(token_ids(a).ok_or_else(|| {
                bad("token ids must be non-negative integers", Some("input"))
            })?)]
        }
        // one interleaved input given as content parts
        Value::Array(a) if a.iter().all(|x| x.get("type").is_some()) => {
            vec![mixed_item(content_parts(a)?, &Opts::default())?]
        }
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::String(s) => text_item(s, &defaults),
                Value::Array(parts) if parts.iter().all(|p| p.is_object()) && !parts.is_empty() => {
                    mixed_item(content_parts(parts)?, &Opts::default())
                }
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
                    "each input must be a string, a token-id array, a content-part array or an object",
                    Some("input"),
                )),
            })
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(bad(
                "'input' must be a string, an object or an array",
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
    let media: usize = items
        .iter()
        .map(|i| match i {
            Item::Mixed(m) => m.images.len() + m.videos.len(),
            _ => 0,
        })
        .sum();
    if media > MAX_MEDIA {
        return Err(bad(
            format!("{media} images/videos > the {MAX_MEDIA} allowed per request"),
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

fn is_remote(src: &str) -> bool {
    src.starts_with("data:") || src.starts_with("http://") || src.starts_with("https://")
}

/// The bytes of a source (a data URL, an http(s) URL, or an allowed path).
fn fetch(src: &str, local: bool) -> Result<Vec<u8>, String> {
    if !is_remote(src) && !local {
        return Err(format!(
            "local paths are not accepted by this server ('{src}'): send a data: or http(s) URL \
             (or start it on loopback / with CMF_EMBED_LOCAL_MEDIA=1)"
        ));
    }
    cortiq_engine::media::load_image_bytes(&json!({ "url": src }))
}

/// A temporary file removed on drop (a fetched video for ffmpeg).
struct TempFile(std::path::PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn video_ext(src: &str) -> &'static str {
    let head = src.split(';').next().unwrap_or("");
    if head.contains("webm") {
        "webm"
    } else if head.contains("quicktime") || src.ends_with(".mov") {
        "mov"
    } else if head.contains("matroska") || src.ends_with(".mkv") {
        "mkv"
    } else if src.ends_with(".y4m") {
        "y4m"
    } else {
        "mp4"
    }
}

/// Decode a mixed input's media and build it.
fn build_mixed(enc: &MediaEncoder, m: &MixedSpec, local: bool) -> Result<MixedInput, String> {
    let mut media = Vec::with_capacity(m.images.len() + m.videos.len());
    for (j, s) in m.images.iter().enumerate() {
        let bytes = fetch(s, local).map_err(|e| format!("image {j}: {e}"))?;
        let img = cortiq_engine::media::decode_rgb(&bytes).map_err(|e| format!("image {j}: {e}"))?;
        media.push(enc.prepare_image(&img, m.image_tokens)?);
    }
    for (j, s) in m.videos.iter().enumerate() {
        let (path, _tmp) = if is_remote(s) {
            let bytes = fetch(s, local).map_err(|e| format!("video {j}: {e}"))?;
            static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let p = std::env::temp_dir().join(format!(
                "cortiq-embed-{}-{}.{}",
                std::process::id(),
                SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                video_ext(s)
            ));
            std::fs::write(&p, &bytes).map_err(|e| format!("video {j}: {e}"))?;
            (p.clone(), Some(TempFile(p)))
        } else if local {
            (std::path::PathBuf::from(s.strip_prefix("file://").unwrap_or(s)), None)
        } else {
            return Err(fetch(s, local).err().unwrap_or_default());
        };
        let dv = decode_video(&path, None, &enc.proc).map_err(|e| format!("video {j}: {e}"))?;
        media.push(enc.prepare_frames(&dv.frames, m.video_tokens)?);
    }
    Ok(MixedInput {
        text: m.text.clone(),
        media,
        prompt_name: m.prompt_name.clone(),
        title: m.title.clone(),
        prompt: m.prompt.clone(),
    })
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
    let parsed = match parse_request(&v, st.enc.text.prompts()) {
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
    let local = st.local_media;
    let job = tokio::task::spawn_blocking(move || -> Result<(Vec<Vec<f32>>, usize), ApiError> {
        let mut inputs = Vec::with_capacity(parsed.items.len());
        let mut tokens = 0usize;
        for (i, it) in parsed.items.iter().enumerate() {
            let x = match it {
                Item::Text(spec) => MixedInput::text(spec.to_input()),
                Item::Mixed(m) => build_mixed(&enc, m, local)
                    .map_err(|e| bad(format!("input {i}: {e}"), Some("input")))?,
                Item::Ids(raw) => {
                    let mut ids = raw.clone();
                    if ids.first() != Some(&enc.text.bos) {
                        ids.insert(0, enc.text.bos);
                    }
                    if ids.last() != Some(&enc.text.eos) {
                        ids.push(enc.text.eos);
                    }
                    if ids.len() > enc.text.max_tokens {
                        return Err(bad(
                            format!(
                                "input {i}: {} tokens > the model's {} token context",
                                ids.len(),
                                enc.text.max_tokens
                            ),
                            Some("input"),
                        ));
                    }
                    // token ids embed on their own below; keep the slot
                    inputs.push((i, None, Some(ids)));
                    continue;
                }
            };
            let n = enc
                .input_ids(&x)
                .map_err(|e| bad(format!("input {i}: {e}"), Some("input")))?
                .len();
            tokens += n;
            inputs.push((i, Some(x), None));
        }
        let mixed: Vec<MixedInput> = inputs.iter().filter_map(|(_, x, _)| x.clone()).collect();
        let ids: Vec<Vec<u32>> = inputs.iter().filter_map(|(_, _, s)| s.clone()).collect();
        tokens += ids.iter().map(|s| s.len()).sum::<usize>();
        let mut a = if mixed.is_empty() {
            Vec::new()
        } else {
            enc.embed(&mixed).map_err(|e| bad(e, Some("input")))?
        }
        .into_iter();
        let mut b = if ids.is_empty() {
            Vec::new()
        } else {
            enc.text.embed_ids(&ids).map_err(|e| bad(e, Some("input")))?
        }
        .into_iter();
        let mut out = Vec::with_capacity(inputs.len());
        for (_, x, _) in &inputs {
            let v = if x.is_some() { a.next() } else { b.next() };
            let v = v.ok_or_else(|| bad("embedding count mismatch", None))?;
            out.push(matryoshka(&v, parsed.dim).map_err(|e| bad(e, Some("dimensions")))?);
        }
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

fn modalities(st: &EmbedState) -> Vec<&'static str> {
    if st.enc.has_vision() {
        vec!["text", "image", "video"]
    } else {
        vec!["text"]
    }
}

async fn prompts(State(st): State<Arc<EmbedState>>) -> Json<Value> {
    let p = st.enc.text.prompts();
    let map: serde_json::Map<String, Value> = p
        .names()
        .into_iter()
        .map(|n| (n.to_string(), json!(p.get(n).unwrap_or(""))))
        .collect();
    Json(json!({
        "prompts": map,
        "dimensions": MATRYOSHKA_DIMS,
        "max_tokens": st.enc.text.max_tokens,
        "modalities": modalities(&st),
        "image_tokens": {"default": st.enc.proc.image_budget, "allowed": BUDGETS},
        "video_tokens": {"default": st.enc.proc.video_budget, "allowed": BUDGETS},
        "video": {"fps": st.enc.proc.video_fps, "max_frames": st.enc.proc.max_frames},
    }))
}

async fn healthz(State(st): State<Arc<EmbedState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": st.model_id,
        "capabilities": {
            "embeddings": true,
            "dimensions": MATRYOSHKA_DIMS,
            "modalities": modalities(&st),
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
        // images and short videos travel as base64 data URLs
        .layer(DefaultBodyLimit::max(256 << 20))
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
    let enc = tokio::task::spawn_blocking(move || -> Result<MediaEncoder, String> {
        let enc = MediaEncoder::load(&model, pool)?;
        if enc.has_vision() {
            // the tower's ~0.7 GB of f32 weights, once, before the first request
            enc.warm_vision()?;
        }
        Ok(enc)
    })
    .await?
    .map_err(anyhow::Error::msg)?;
    let model_id = std::path::Path::new(model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "embeddinggemma-2".into());
    let local_media =
        addr.ip().is_loopback() || std::env::var("CMF_EMBED_LOCAL_MEDIA").is_ok_and(|v| v == "1");
    println!(
        "    EmbeddingGemma 2 ({}), {} dims, {} task prompts, modalities: {}",
        if enc.text.quant.is_empty() {
            "?"
        } else {
            &enc.text.quant
        },
        enc.text.dim,
        enc.text.prompts().names().len(),
        if enc.has_vision() {
            "text, image, video"
        } else {
            "text"
        }
    );
    println!(
        "  Embeddings API: POST http://{addr}/v1/embeddings (model \"{model_id}\"; local media paths {})",
        if local_media { "on" } else { "off" }
    );
    let app = router(Arc::new(EmbedState::new(enc, model_id, local_media)));
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
    fn parses_media_inputs() {
        let r = parse_request(
            &json!({"input": [
                "a query",
                {"image": "data:image/png;base64,AAAA"},
                {"text": "A fox: <|image|> in snow", "image": ["https://x/fox.png"], "image_tokens": 560},
                {"video": {"url": "https://x/v.mp4"}},
                [{"type": "text", "text": "Photo:"}, {"type": "image_url", "image_url": {"url": "https://x/a.png"}}],
            ], "task": "SearchQuery", "video_tokens": 70}),
            &p(),
        )
        .unwrap();
        assert_eq!(r.items.len(), 5);
        let Item::Text(t) = &r.items[0] else { panic!() };
        assert_eq!(t.prompt_name.as_deref(), Some("SearchQuery"));
        let Item::Mixed(m) = &r.items[1] else { panic!() };
        // media inputs do not inherit the request's prompt
        assert_eq!(m.prompt_name, None);
        assert_eq!(m.images.len(), 1);
        let Item::Mixed(m) = &r.items[2] else { panic!() };
        assert_eq!(m.text, "A fox: <|image|> in snow");
        assert_eq!(m.image_tokens, Some(560));
        let Item::Mixed(m) = &r.items[3] else { panic!() };
        assert_eq!(m.videos, vec!["https://x/v.mp4".to_string()]);
        assert_eq!(m.video_tokens, Some(70));
        let Item::Mixed(m) = &r.items[4] else { panic!() };
        assert_eq!(m.text, "Photo: <|image|>");
        assert_eq!(m.images, vec!["https://x/a.png".to_string()]);
        // a single content-part array is one input
        let r = parse_request(
            &json!({"input": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}, {"type": "text", "text": "caption"}]}),
            &p(),
        )
        .unwrap();
        let Item::Mixed(m) = &r.items[0] else { panic!() };
        assert_eq!(m.text, "<|image|> caption");
        // a media item may carry its own prompt
        let r = parse_request(
            &json!({"input": [{"image": "data:,", "prompt_name": "Document", "title": "T"}]}),
            &p(),
        )
        .unwrap();
        let Item::Mixed(m) = &r.items[0] else { panic!() };
        assert_eq!(m.title.as_deref(), Some("T"));
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
            json!({"input": [{"image": 5}]}),
            json!({"input": [{"nothing": 1}]}),
            json!({"input": [{"image": "data:,"}], "image_tokens": 300}),
            json!({"input": [[{"type": "audio_url"}]]}),
        ] {
            assert!(parse_request(&body, &p()).is_err(), "{body}");
        }
    }

    #[test]
    fn local_paths_need_permission() {
        assert!(fetch("/etc/hosts", false).is_err());
        assert!(fetch("data:image/png;base64,AAEC", false).is_ok());
    }

    #[test]
    fn base64_is_little_endian_f32() {
        let s = f32_base64(&[1.0, -2.0]);
        let b = base64::engine::general_purpose::STANDARD.decode(s).unwrap();
        assert_eq!(&b[..4], &1.0f32.to_le_bytes());
        assert_eq!(&b[4..], &(-2.0f32).to_le_bytes());
    }
}
