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
            },
            finish_reason: None,
        }],
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
            },
            finish_reason: Some(reason.to_string()),
        }],
    }
}
