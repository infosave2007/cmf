//! Lingering close of the decision server's connections (nginx's
//! `lingering_close`).
//!
//! The server may answer before it has read the request body: 401 or 429
//! after the headers alone, 413 from `Content-Length`, 415 from the content
//! type, a JSON 404 or 405. hyper then closes the connection with the rest of
//! the body unread, and a socket closed with unread bytes (or receiving bytes
//! after its close) makes the kernel reset the connection. A reverse proxy
//! still writing that body — nginx sends a buffered body after the headers
//! (`proxy_request_buffering on`) — fails its write with EPIPE and answers
//! 502 with its own HTML page instead of the server's JSON error.
//!
//! [`Listener`] hands every connection out as a [`Lingering`] stream: when
//! hyper is done with it, the write side is shut down (the answer is
//! complete) and what the client still sends is read and dropped — at most
//! the body limit, [`LINGER_IDLE`] between reads and [`LINGER_TIME`] in all —
//! before the socket is closed, so the client finishes its write and reads
//! the answer. A client that has sent everything already closes at once.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::{TcpListener, TcpStream};

/// The longest a closed connection keeps reading what its client sends.
pub const LINGER_TIME: Duration = Duration::from_secs(30);
/// The longest it waits for the client's next bytes.
pub const LINGER_IDLE: Duration = Duration::from_secs(5);

/// A TCP listener whose connections close lingering ([`Lingering`]).
pub struct Listener {
    inner: TcpListener,
    /// The most a closed connection reads (`limits.body_bytes`).
    limit: usize,
}

impl Listener {
    pub fn new(inner: TcpListener, limit: usize) -> Self {
        Self { inner, limit }
    }
}

impl axum::serve::Listener for Listener {
    type Io = Lingering;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        // axum's own accept loop: an accept error is logged and retried.
        let (stream, addr) = axum::serve::Listener::accept(&mut self.inner).await;
        (
            Lingering {
                stream: Some(stream),
                limit: self.limit,
            },
            addr,
        )
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

/// A connection that closes lingering when dropped (see the module notes).
pub struct Lingering {
    /// `None` only once dropped.
    stream: Option<TcpStream>,
    limit: usize,
}

impl Lingering {
    fn stream(&mut self) -> Pin<&mut TcpStream> {
        Pin::new(self.stream.as_mut().expect("a live connection"))
    }
}

impl AsyncRead for Lingering {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        self.get_mut().stream().poll_read(cx, buf)
    }
}

impl AsyncWrite for Lingering {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.get_mut().stream().poll_write(cx, buf)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        self.get_mut().stream().poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.as_ref().is_some_and(|s| s.is_write_vectored())
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.get_mut().stream().poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.get_mut().stream().poll_shutdown(cx)
    }
}

impl Drop for Lingering {
    fn drop(&mut self) {
        let Some(stream) = self.stream.take() else {
            return;
        };
        // Outside a runtime (or while it shuts down) the socket just closes.
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(linger(stream, self.limit));
        }
    }
}

/// Shut the write side down, then read and drop what the client still sends
/// until it closes, `limit` bytes have come, it pauses [`LINGER_IDLE`] or
/// [`LINGER_TIME`] has passed; the socket closes when this returns.
async fn linger(mut stream: TcpStream, limit: usize) {
    let _ = stream.shutdown().await;
    let end = tokio::time::Instant::now() + LINGER_TIME;
    let mut buf = vec![0u8; 16 * 1024];
    let mut seen = 0usize;
    while seen <= limit {
        let until = end.min(tokio::time::Instant::now() + LINGER_IDLE);
        match tokio::time::timeout_at(until, stream.read(&mut buf)).await {
            Ok(Ok(n)) if n > 0 => seen += n,
            // Closed by the client, an error, or out of time.
            _ => break,
        }
    }
}
