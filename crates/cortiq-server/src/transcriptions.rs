//! Speech-to-text API over `cortiq_engine::gigaam`, in the shape of
//! OpenAI's `POST /v1/audio/transcriptions`.
//!
//! ## Request
//!
//! `multipart/form-data`, as the OpenAI clients send it:
//!
//! * `file` — the recording (WAV is decoded here; mp3, flac, ogg, m4a … go
//!   through `ffmpeg`);
//! * `response_format` — `json` (default: `{"text"}`), `text`,
//!   `verbose_json` (duration, segments and, when asked for, words), `srt`,
//!   `vtt`;
//! * `timestamp_granularities[]` — `segment` and / or `word` (with
//!   `verbose_json`);
//! * `head` — `e2e` (default: case, punctuation, digits) or `raw`
//!   (lower-case words): the model carries both recognition heads;
//! * `model`, `language`, `prompt`, `temperature` are accepted and ignored
//!   (the model detects Russian and English itself and takes no prompt).
//!
//! Or `application/json` with the same fields, where `file` is a `data:`
//! URL, an `http(s)://` URL, an object `{"data": <base64>}`, or — on a
//! loopback server, or with `CMF_ASR_LOCAL_MEDIA=1` — a local path.
//!
//! A recording longer than 30 s is cut by the file's voice-activity model;
//! segments carry their times in the recording.
//!
//! `GET /v1/models` lists the one model; `GET /healthz` reports it.

use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use base64::Engine as _;
use cortiq_core::CmfModel;
use cortiq_engine::gigaam::{self, Heads, Transcriber, Transcript};
use cortiq_engine::pool::Pool;
use serde_json::{Value, json};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

/// The largest upload taken (a multi-hour recording in a compressed container).
const MAX_BODY: usize = 1 << 30;

pub struct AsrState {
    pub tr: Arc<Transcriber>,
    pub model_id: String,
    /// local paths are accepted as the `file` of a JSON request
    pub local_media: bool,
    pool: Option<Arc<Pool>>,
    /// one recording at a time: a transcription already uses every core or the whole card
    gate: tokio::sync::Mutex<()>,
}

impl AsrState {
    pub fn new(tr: Transcriber, model_id: String, local_media: bool, pool: Option<Arc<Pool>>) -> Self {
        Self { tr: Arc::new(tr), model_id, local_media, pool, gate: tokio::sync::Mutex::new(()) }
    }
}

/// How the transcript is returned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
    Json,
    Text,
    Verbose,
    Srt,
    Vtt,
}

/// A parsed request.
#[derive(Debug, PartialEq)]
pub struct Request {
    pub audio: Vec<u8>,
    pub format: Format,
    pub words: bool,
    pub raw: bool,
}

/// An OpenAI-shaped 400.
#[derive(Debug, PartialEq)]
pub struct ApiError {
    pub message: String,
    pub param: Option<&'static str>,
}

fn bad(message: impl Into<String>, param: Option<&'static str>) -> ApiError {
    ApiError { message: message.into(), param }
}

fn error_response(status: StatusCode, e: ApiError, kind: &str) -> Response {
    (status, Json(json!({"error": {"message": e.message, "type": kind, "param": e.param, "code": Value::Null}}))).into_response()
}

/// One part of a `multipart/form-data` body.
#[derive(Debug, PartialEq)]
pub struct Part<'a> {
    pub name: String,
    pub data: &'a [u8],
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() || from > hay.len() - needle.len() {
        return None;
    }
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

/// The boundary of a `multipart/form-data` content type.
pub fn multipart_boundary(content_type: &str) -> Option<String> {
    let (kind, rest) = content_type.split_once(';')?;
    if !kind.trim().eq_ignore_ascii_case("multipart/form-data") {
        return None;
    }
    rest.split(';').find_map(|p| {
        let (k, v) = p.split_once('=')?;
        k.trim().eq_ignore_ascii_case("boundary").then(|| v.trim().trim_matches('"').to_string())
    })
}

/// Split a `multipart/form-data` body into its named parts.
pub fn parse_multipart<'a>(body: &'a [u8], boundary: &str) -> Result<Vec<Part<'a>>, String> {
    let delim = format!("--{boundary}").into_bytes();
    let mut parts = Vec::new();
    let mut at = find(body, &delim, 0).ok_or("multipart body: the boundary does not occur")? + delim.len();
    loop {
        if body[at..].starts_with(b"--") {
            return Ok(parts);
        }
        // the delimiter line ends, the part's headers follow up to an empty line
        at = find(body, b"\r\n", at).ok_or("multipart body: truncated after a boundary")? + 2;
        let head_end = find(body, b"\r\n\r\n", at).ok_or("multipart body: a part has no header end")?;
        let headers = String::from_utf8_lossy(&body[at..head_end]);
        let mut next = b"\r\n".to_vec();
        next.extend_from_slice(&delim);
        let data_start = head_end + 4;
        let data_end = find(body, &next, data_start).ok_or("multipart body: a part is not closed")?;
        let name = headers
            .lines()
            .filter(|l| l.to_ascii_lowercase().starts_with("content-disposition:"))
            .flat_map(|l| l.split(';'))
            .find_map(|p| {
                let (k, v) = p.split_once('=')?;
                (k.trim() == "name").then(|| v.trim().trim_matches('"').to_string())
            })
            .ok_or("multipart body: a part has no name")?;
        parts.push(Part { name, data: &body[data_start..data_end] });
        at = data_end + next.len();
    }
}

fn parse_format(s: &str) -> Result<Format, ApiError> {
    Ok(match s.trim() {
        "" | "json" => Format::Json,
        "text" => Format::Text,
        "verbose_json" => Format::Verbose,
        "srt" => Format::Srt,
        "vtt" => Format::Vtt,
        other => return Err(bad(format!("response_format '{other}': use json, text, verbose_json, srt or vtt"), Some("response_format"))),
    })
}

fn parse_head(s: &str) -> Result<bool, ApiError> {
    match s.trim() {
        "" | "e2e" | "text" => Ok(false),
        "raw" => Ok(true),
        other => Err(bad(format!("head '{other}': use e2e (formatted text) or raw (lower-case words)"), Some("head"))),
    }
}

/// Parse a request body of the given content type. `local` allows a local
/// path as the `file` of a JSON request.
pub fn parse_request(content_type: &str, body: &[u8], local: bool) -> Result<Request, ApiError> {
    if let Some(boundary) = multipart_boundary(content_type) {
        let parts = parse_multipart(body, &boundary).map_err(|e| bad(e, None))?;
        let text = |name: &str| parts.iter().find(|p| p.name == name).map(|p| String::from_utf8_lossy(p.data).trim().to_string());
        let audio = parts.iter().find(|p| p.name == "file").ok_or_else(|| bad("the form has no 'file' part", Some("file")))?.data.to_vec();
        let words = parts
            .iter()
            .filter(|p| p.name == "timestamp_granularities[]" || p.name == "timestamp_granularities")
            .any(|p| String::from_utf8_lossy(p.data).split(',').any(|g| g.trim().trim_matches('"') == "word"));
        return Ok(Request {
            audio,
            format: parse_format(&text("response_format").unwrap_or_default())?,
            words,
            raw: parse_head(&text("head").unwrap_or_default())?,
        });
    }
    if !content_type.to_ascii_lowercase().contains("json") {
        return Err(bad("send multipart/form-data with a 'file' part, or application/json with 'file'", None));
    }
    let v: Value = serde_json::from_slice(body).map_err(|e| bad(format!("invalid JSON: {e}"), None))?;
    let file = v.get("file").or_else(|| v.get("audio")).ok_or_else(|| bad("the request has no 'file'", Some("file")))?;
    let audio = match file {
        Value::String(s) => {
            let remote = s.starts_with("data:") || s.starts_with("http://") || s.starts_with("https://");
            if !remote && !local {
                return Err(bad(
                    "local paths are not accepted by this server: send a data: or http(s) URL, or upload the file as \
                     multipart/form-data (or start the server on loopback / with CMF_ASR_LOCAL_MEDIA=1)",
                    Some("file"),
                ));
            }
            cortiq_engine::media::load_image_bytes(&json!({ "url": s })).map_err(|e| bad(e, Some("file")))?
        }
        Value::Object(m) => match m.get("data").and_then(Value::as_str) {
            Some(d) => base64::engine::general_purpose::STANDARD
                .decode(d.trim())
                .map_err(|e| bad(format!("invalid base64 audio data: {e}"), Some("file")))?,
            None => return Err(bad("a 'file' object needs 'data' (base64)", Some("file"))),
        },
        _ => return Err(bad("'file' must be a string or an object", Some("file"))),
    };
    let str_of = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let words = v
        .get("timestamp_granularities")
        .and_then(Value::as_array)
        .is_some_and(|a| a.iter().any(|g| g.as_str() == Some("word")));
    Ok(Request { audio, format: parse_format(&str_of("response_format"))?, words, raw: parse_head(&str_of("head"))? })
}

fn clock(t: f64, sep: char) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

/// SubRip or WebVTT cues of a transcript, a segment a cue.
pub fn subtitles(t: &Transcript, vtt: bool) -> String {
    let mut s = String::new();
    if vtt {
        s.push_str("WEBVTT\n\n");
    }
    let sep = if vtt { '.' } else { ',' };
    for (i, seg) in t.segments.iter().filter(|s| !s.text.is_empty()).enumerate() {
        if !vtt {
            s.push_str(&format!("{}\n", i + 1));
        }
        s.push_str(&format!("{} --> {}\n{}\n\n", clock(seg.start, sep), clock(seg.end, sep), seg.text));
    }
    s
}

fn render(t: &Transcript, duration: f64, format: Format, words: bool) -> Response {
    match format {
        Format::Json => Json(json!({ "text": t.text })).into_response(),
        Format::Text => ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], format!("{}\n", t.text)).into_response(),
        Format::Srt => ([(header::CONTENT_TYPE, "application/x-subrip; charset=utf-8")], subtitles(t, false)).into_response(),
        Format::Vtt => ([(header::CONTENT_TYPE, "text/vtt; charset=utf-8")], subtitles(t, true)).into_response(),
        Format::Verbose => {
            let segments: Vec<Value> = t
                .segments
                .iter()
                .enumerate()
                .map(|(i, s)| json!({"id": i, "start": s.start, "end": s.end, "text": s.text}))
                .collect();
            let mut body = json!({"task": "transcribe", "duration": duration, "text": t.text, "segments": segments});
            if words {
                body["words"] = t
                    .segments
                    .iter()
                    .flat_map(|s| &s.words)
                    .map(|w| json!({"word": w.text, "start": w.start, "end": w.end}))
                    .collect();
            }
            Json(body).into_response()
        }
    }
}

async fn transcriptions(State(st): State<Arc<AsrState>>, headers: HeaderMap, body: Bytes) -> Response {
    let content_type = headers.get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let _turn = st.gate.lock().await;
    let (tr, pool, local) = (st.tr.clone(), st.pool.clone(), st.local_media);
    let work = tokio::task::spawn_blocking(move || -> Result<Response, (StatusCode, ApiError)> {
        let req = parse_request(&content_type, &body, local).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        drop(body);
        let samples = gigaam::decode_audio(&req.audio).map_err(|e| (StatusCode::BAD_REQUEST, bad(format!("cannot decode the audio: {e}"), Some("file"))))?;
        let opts = gigaam::Options { heads: if req.raw { Heads::Raw } else { Heads::E2e }, words: req.words, ..Default::default() };
        let out = tr.transcribe(&samples, &opts, pool.as_deref()).map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, bad(e, Some("file"))))?;
        let t = out.e2e.as_ref().or(out.raw.as_ref()).cloned().unwrap_or_default();
        Ok(render(&t, out.audio_s, req.format, req.words))
    })
    .await;
    match work {
        Ok(Ok(r)) => r,
        Ok(Err((status, e))) => error_response(status, e, "invalid_request_error"),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, bad(format!("transcription failed: {e}"), None), "server_error"),
    }
}

async fn models(State(st): State<Arc<AsrState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": st.model_id, "object": "model", "created": chrono::Utc::now().timestamp(), "owned_by": "cortiq"}],
    }))
}

async fn healthz(State(st): State<Arc<AsrState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": st.model_id,
        "capabilities": {
            "transcriptions": true,
            "revision": st.tr.revision,
            "long_recordings": st.tr.has_vad(),
            "sample_rate": gigaam::SR,
            "response_formats": ["json", "text", "verbose_json", "srt", "vtt"],
        },
    }))
}

/// The transcription router.
pub fn router(state: Arc<AsrState>) -> Router {
    Router::new()
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/models", get(models))
        .route("/healthz", get(healthz))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Load the model and serve it on `addr`.
pub async fn serve(model: Arc<CmfModel>, model_path: &str, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let pool = Pool::from_env();
    let tr = tokio::task::spawn_blocking(move || Transcriber::load(&model)).await?.map_err(anyhow::Error::msg)?;
    let model_id = std::path::Path::new(model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "gigaam-v4".into());
    let local_media = addr.ip().is_loopback() || std::env::var("CMF_ASR_LOCAL_MEDIA").is_ok_and(|v| v == "1");
    println!(
        "    GigaAM-v4 ({}), device {}, recordings over 30 s: {}",
        tr.revision,
        cortiq_engine::gpu::backend_name(),
        if tr.has_vad() { "cut by voice activity" } else { "not supported by this file" }
    );
    println!("  Transcription API: POST http://{addr}/v1/audio/transcriptions (model \"{model_id}\")");
    let app = router(Arc::new(AsrState::new(tr, model_id, local_media, pool)));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_engine::gigaam::Segment;

    fn form(parts: &[(&str, &[u8])], boundary: &str) -> Vec<u8> {
        let mut b = Vec::new();
        for (name, data) in parts {
            b.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"").as_bytes());
            if *name == "file" {
                b.extend_from_slice(b"; filename=\"a.wav\"\r\nContent-Type: audio/wav");
            }
            b.extend_from_slice(b"\r\n\r\n");
            b.extend_from_slice(data);
            b.extend_from_slice(b"\r\n");
        }
        b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        b
    }

    #[test]
    fn multipart_forms_are_split_into_parts() {
        assert_eq!(multipart_boundary("multipart/form-data; boundary=xYz").as_deref(), Some("xYz"));
        assert_eq!(multipart_boundary("multipart/form-data; charset=utf-8; boundary=\"a b\"").as_deref(), Some("a b"));
        assert_eq!(multipart_boundary("application/json"), None);
        // binary data that holds line breaks and dashes survives
        let audio = b"RIFF\r\n--not-the-boundary\r\n\x00\x01";
        let body = form(&[("model", b"gigaam"), ("file", audio), ("response_format", b"verbose_json")], "xYz");
        let parts = parse_multipart(&body, "xYz").unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[1], Part { name: "file".into(), data: audio });
        assert_eq!(parts[2].data, b"verbose_json");
        assert!(parse_multipart(b"--xYz\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nunclosed", "xYz").is_err());
        assert!(parse_multipart(b"nothing here", "xYz").is_err());
    }

    #[test]
    fn requests_parse_in_both_shapes() {
        let body = form(&[("file", b"abc"), ("timestamp_granularities[]", b"word"), ("head", b"raw"), ("response_format", b"srt")], "B");
        let r = parse_request("multipart/form-data; boundary=B", &body, false).unwrap();
        assert_eq!(r, Request { audio: b"abc".to_vec(), format: Format::Srt, words: true, raw: true });
        let r = parse_request("multipart/form-data; boundary=B", &form(&[("file", b"abc")], "B"), false).unwrap();
        assert_eq!((r.format, r.words, r.raw), (Format::Json, false, false));
        assert_eq!(parse_request("multipart/form-data; boundary=B", &form(&[("model", b"x")], "B"), false).unwrap_err().param, Some("file"));
        assert_eq!(
            parse_request("multipart/form-data; boundary=B", &form(&[("file", b"a"), ("response_format", b"xml")], "B"), false).unwrap_err().param,
            Some("response_format")
        );
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"abc");
        let json_body = serde_json::to_vec(&json!({"file": {"data": b64}, "response_format": "text", "timestamp_granularities": ["segment", "word"]})).unwrap();
        let r = parse_request("application/json", &json_body, false).unwrap();
        assert_eq!(r, Request { audio: b"abc".to_vec(), format: Format::Text, words: true, raw: false });
        // a local path is refused off loopback
        let local = serde_json::to_vec(&json!({"file": "/tmp/a.wav"})).unwrap();
        assert_eq!(parse_request("application/json", &local, false).unwrap_err().param, Some("file"));
        assert!(parse_request("text/plain", b"x", true).is_err());
    }

    #[test]
    fn subtitles_carry_segment_times() {
        let t = Transcript {
            text: "Привет. Как дела?".into(),
            segments: vec![
                Segment { start: 0.2, end: 1.45, text: "Привет.".into(), words: Vec::new() },
                Segment { start: 3661.0, end: 3662.5, text: "Как дела?".into(), words: Vec::new() },
            ],
        };
        assert_eq!(subtitles(&t, false), "1\n00:00:00,200 --> 00:00:01,450\nПривет.\n\n2\n01:01:01,000 --> 01:01:02,500\nКак дела?\n\n");
        assert!(subtitles(&t, true).starts_with("WEBVTT\n\n00:00:00.200 --> 00:00:01.450\nПривет.\n\n"));
    }
}
