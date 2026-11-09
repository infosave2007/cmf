//! SSE (Server-Sent Events) streaming for OpenAI-compatible chat completions.

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use serde::Serialize;
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

/// A streaming chat completion response.
pub struct ChatStream {
    rx: mpsc::Receiver<StreamChunk>,
    state: StreamState,
}

/// SSE termination protocol: after the finish_reason chunk (or channel
/// close) exactly one `data: [DONE]` is emitted, then the stream ends.
#[derive(PartialEq)]
enum StreamState {
    Open,
    Finishing,
    Done,
}

/// A single chunk in the SSE stream.
#[derive(Debug, Clone, Serialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    /// Real token counts, emitted once ahead of the finish chunk (the OpenAI
    /// `include_usage` shape) — clients get exact numbers, not estimates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<StreamUsage>,
    /// Router-v2 decision of the request (`{target, novelty, e_base,
    /// e_skill, reason}`), carried by the usage chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_cortiq_route: Option<serde_json::Value>,
    /// Prompt and decode speed of the request (llama-server's `timings`),
    /// on the usage and finish chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

/// Prompt and decode speed of one request, in llama-server's `timings` shape:
/// `cache_n` prompt tokens came from the reused prefix, `prompt_n` were read
/// up to the first new token, `predicted_n` tokens came after it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Timings {
    pub cache_n: usize,
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
    /// Every new token, the first one included.
    #[serde(skip)]
    pub generated: usize,
}

impl Timings {
    /// `generated` tokens in all: the first closes the prompt's `prompt_ms`,
    /// the rest share `decode_ms`.
    pub fn new(
        cache_n: usize,
        prompt_n: usize,
        prompt_ms: f64,
        generated: usize,
        decode_ms: f64,
    ) -> Self {
        let predicted_n = generated.saturating_sub(1);
        let per = |n: usize, ms: f64| {
            if n > 0 && ms > 0.0 {
                (ms / n as f64, n as f64 * 1000.0 / ms)
            } else {
                (0.0, 0.0)
            }
        };
        let (prompt_per_token_ms, prompt_per_second) = per(prompt_n, prompt_ms);
        let (predicted_per_token_ms, predicted_per_second) = per(predicted_n, decode_ms);
        Self {
            cache_n,
            prompt_n,
            prompt_ms,
            prompt_per_token_ms,
            prompt_per_second,
            predicted_n,
            predicted_ms: if predicted_n > 0 { decode_ms } else { 0.0 },
            predicted_per_token_ms,
            predicted_per_second,
            generated,
        }
    }

    /// The three lines llama-server prints after a request.
    pub fn lines(&self) -> [String; 3] {
        let cached = if self.cache_n > 0 {
            format!(", {} cached", self.cache_n)
        } else {
            String::new()
        };
        [
            format!(
                "prompt eval time = {:10.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second){cached}",
                self.prompt_ms, self.prompt_n, self.prompt_per_token_ms, self.prompt_per_second
            ),
            format!(
                "       eval time = {:10.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)",
                self.predicted_ms,
                self.predicted_n,
                self.predicted_per_token_ms,
                self.predicted_per_second
            ),
            format!(
                "      total time = {:10.2} ms / {:5} tokens",
                self.prompt_ms + self.predicted_ms,
                self.prompt_n + self.generated
            ),
        ]
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls arrive as one aggregated delta right before the
    /// finish chunk — the shape every OpenAI client accumulates anyway.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
}

impl ChatStream {
    /// Create a new stream pair (sender, SSE response).
    pub fn new(buffer: usize) -> (mpsc::Sender<StreamChunk>, Self) {
        let (tx, rx) = mpsc::channel(buffer);
        (
            tx,
            Self {
                rx,
                state: StreamState::Open,
            },
        )
    }

    /// Convert to axum SSE response.
    pub fn into_sse(self) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        Sse::new(SseStream { inner: self })
    }
}

struct SseStream {
    inner: ChatStream,
}

impl Stream for SseStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.state {
            StreamState::Done => return Poll::Ready(None),
            StreamState::Finishing => {
                self.inner.state = StreamState::Done;
                return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
            }
            StreamState::Open => {}
        }

        match self.inner.rx.poll_recv(cx) {
            Poll::Ready(Some(chunk)) => {
                let is_finish = chunk
                    .choices
                    .first()
                    .and_then(|c| c.finish_reason.as_deref())
                    .is_some();
                if is_finish {
                    self.inner.state = StreamState::Finishing;
                }
                let json = serde_json::to_string(&chunk).unwrap_or_default();
                Poll::Ready(Some(Ok(Event::default().data(json))))
            }
            Poll::Ready(None) => {
                // Channel closed without a finish chunk — still terminate
                // the protocol correctly.
                self.inner.state = StreamState::Done;
                Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// One delta carrying the aggregated tool calls, indexed for clients
/// that merge by `index`.
pub fn tool_calls_chunk(
    id: &str,
    model: &str,
    calls: serde_json::Value,
    created: u64,
) -> StreamChunk {
    StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                role: None,
                content: None,
                tool_calls: Some(calls),
            },
            finish_reason: None,
        }],
        usage: None,
        x_cortiq_route: None,
        timings: None,
    }
}

/// A content-delta chunk (built synchronously from the generation thread).
pub fn token_chunk(id: &str, model: &str, token: &str, created: u64) -> StreamChunk {
    StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                role: None,
                content: Some(token.to_string()),
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: None,
        x_cortiq_route: None,
        timings: None,
    }
}

/// The usage chunk (empty `choices`, real counts) sent just before the finish
/// chunk, exactly as OpenAI's `stream_options.include_usage` does.
pub fn usage_chunk(
    id: &str,
    model: &str,
    created: u64,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> StreamChunk {
    StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: Vec::new(),
        usage: Some(StreamUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
        x_cortiq_route: None,
        timings: None,
    }
}

/// The terminal chunk carrying the real finish_reason.
pub fn finish_chunk(id: &str, model: &str, reason: &str, created: u64) -> StreamChunk {
    StreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![StreamChoice {
            index: 0,
            delta: StreamDelta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason: Some(reason.to_string()),
        }],
        usage: None,
        x_cortiq_route: None,
        timings: None,
    }
}

#[cfg(test)]
mod timings_tests {
    use super::Timings;

    #[test]
    fn the_first_token_closes_the_prompt_time() {
        let t = Timings::new(100, 400, 200.0, 51, 500.0);
        assert_eq!((t.cache_n, t.prompt_n, t.predicted_n), (100, 400, 50));
        assert!((t.prompt_per_second - 2000.0).abs() < 1e-9);
        assert!((t.predicted_per_second - 100.0).abs() < 1e-9);
        assert!((t.predicted_per_token_ms - 10.0).abs() < 1e-9);
        let lines = t.lines();
        assert!(
            lines[0].starts_with("prompt eval time =") && lines[0].ends_with(", 100 cached"),
            "{}",
            lines[0]
        );
        assert!(lines[2].ends_with(" 451 tokens"), "{}", lines[2]);
        let json = serde_json::to_value(t).unwrap();
        assert_eq!(json["predicted_n"], 50);
        assert!(json.get("generated").is_none(), "llama-server's shape only");
    }

    #[test]
    fn no_new_token_leaves_the_decode_empty() {
        let t = Timings::new(0, 10, 50.0, 0, 0.0);
        assert_eq!(
            (t.predicted_n, t.predicted_ms, t.predicted_per_second),
            (0, 0.0, 0.0)
        );
        assert!(!t.lines()[0].contains("cached"));
    }
}
