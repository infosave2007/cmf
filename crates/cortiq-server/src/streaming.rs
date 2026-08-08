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

/// A content-delta chunk (built synchronously from the generation thread).

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
    }
}

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
    }
}
