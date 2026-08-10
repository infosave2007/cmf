//! Network pipeline-split: a coordinator drives layer spans across nodes.
//!
//! Topology (v1): the coordinator owns embed + layers `[0..split)` + final
//! norm + lm_head + sampler; ONE remote worker owns layers `[split..n)` and
//! their KV. Attention causality is per-layer, so the whole prompt's
//! boundary hiddens ship as position batches (prefill pays ~one round trip
//! per chunk) and decode ships one hidden vector per token (one round trip
//! per token — the wire cost is latency, not bandwidth).
//!
//! Identity: the worker proves it holds the SAME model by `dir_hash` — the
//! hash of the exact tensor-directory bytes, the same key `skill apply`
//! trusts. A mismatch is refused at handshake; a chimera is never assembled.
//!
//! Wire v2, measured against v1 on the Thunderbolt stand: control frames
//! stay bincode (rare, small); the per-token frames (Step/Hidden/Prefill)
//! are raw little-endian — one buffer, ONE write syscall per frame (v1's
//! separate length write emitted its own TCP segment under NODELAY: four
//! packets per round trip instead of two), an optional f16 payload
//! (negotiated in Assign, never silent), and a bounded busy-poll before
//! blocking reads (`CMF_NET_SPIN` µs, default 3000, `0` disables) — a
//! blocking-read wakeup on a power-managed core costs more than the wire.
//!
//! The `.cmf` container is untouched by all of this: the split is a
//! property of the run, never of the file.

pub mod client;
pub mod worker;

pub use client::{generate_split, RemoteSegment, SessionSpec, SplitStats};
pub use worker::{worker_serve, WorkerConfig};

use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpStream;

/// Bumped on any incompatible frame change; checked first at handshake.
/// v3: Prefill frames are fire-and-forget; `Sync` fetches the last
/// boundary hidden — the coordinator computes chunk k+1 while the worker
/// chews chunk k (prefill ≈ max of the two sides instead of their sum).
/// v4: Assign carries the SESSION SPEC — skill overlay id, task-mask
/// name and O(1)-attention config — so both sides run the same model
/// configuration, each over its own layers.
pub const WIRE_VERSION: u32 = 4;

// Prefill chunking follows the engine's `pipeline::prefill_chunk()` —
// panel width reorders float accumulation, so the network split MUST
// chunk exactly like the local path to reproduce local generations.

/// Hard ceiling on one frame (a corrupt length prefix must not OOM us).
const MAX_FRAME: u32 = 256 * 1024 * 1024;

const TAG_CONTROL: u8 = 0;
const TAG_STEP_F32: u8 = 1;
const TAG_STEP_F16: u8 = 2;
const TAG_HIDDEN_F32: u8 = 3;
const TAG_HIDDEN_F16: u8 = 4;
const TAG_PREFILL_F32: u8 = 5;
const TAG_PREFILL_F16: u8 = 6;

/// Payload float width on the wire. f32 is bit-exact; f16 halves the
/// frames and is negotiated EXPLICITLY in Assign — never a silent default.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WireDtype {
    F32,
    F16,
}

impl WireDtype {
    pub fn code(self) -> u8 {
        match self {
            WireDtype::F32 => 0,
            WireDtype::F16 => 1,
        }
    }
    pub fn from_code(c: u8) -> Result<Self, String> {
        match c {
            0 => Ok(WireDtype::F32),
            1 => Ok(WireDtype::F16),
            _ => Err(format!("unknown wire dtype code {c}")),
        }
    }
}

/// Control frames: rare, small, bincode-encoded under TAG_CONTROL.
#[derive(Serialize, Deserialize, Debug)]
pub enum Frame {
    /// Coordinator → worker, first frame on the wire.
    Hello {
        wire: u32,
        token: String,
        dir_hash: u64,
        arch: String,
        num_layers: u32,
        hidden_size: u32,
    },
    /// Worker's verdict on Hello / Assign / Reset. Empty `err` = accepted.
    Ack { err: String },
    /// Layer span [from ..= upto] this worker runs, the payload dtype
    /// BOTH directions use from here on, and the session configuration:
    /// skill overlay id (worker reloads its pipeline with it), task-mask
    /// name (worker resolves it from ITS copy of the file and masks its
    /// own layers) and O(1)-attention config (worker mirrors it over its
    /// span). All refusals are loud Acks.
    Assign {
        from: u32,
        upto: u32,
        dtype: u8,
        skill: Option<String>,
        task: Option<String>,
        o1: Option<O1Wire>,
    },
    /// Fresh sequence on the worker (clears its KV and state).
    Reset,
    /// Prefill barrier: the worker replies `Hidden` with the output of
    /// the LAST prefill position it has processed.
    Sync,
    Ping,
    Pong,
    /// Fatal worker-side error; the connection closes after this frame.
    Err { msg: String },
}

/// O(1)-attention (Nyström) config on the wire — the CLI flag set,
/// verbatim; each side resolves it with the same `O1Cfg::from_spec`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct O1Wire {
    pub spec: String,
    pub m: Option<u32>,
    pub w: Option<u32>,
    pub sink: Option<u32>,
    pub rect: Option<String>,
}

/// One decoded message. Float payloads land in the caller's `floats`
/// scratch (reused across frames — the hot path does not allocate).
#[derive(Debug)]
pub enum Msg {
    Control(Frame),
    Step { pos: u64 },
    Hidden,
    Prefill { start_pos: u64, count: u32 },
}

fn spin_budget_us() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("CMF_NET_SPIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3000)
    })
}

/// Busy-poll until the stream is readable or the budget runs out, then
/// restore blocking mode. A wakeup from a blocking read on a
/// power-managed core (M1 efficiency core, parked P-core) costs 0.5–2 ms;
/// a peek loop costs a syscall (~1 µs) per iteration on one core.
fn spin_wait(stream: &TcpStream) {
    let budget = spin_budget_us();
    if budget == 0 {
        return;
    }
    if stream.set_nonblocking(true).is_err() {
        return;
    }
    let t0 = std::time::Instant::now();
    let mut probe = [0u8; 1];
    loop {
        match stream.peek(&mut probe) {
            Ok(_) => break,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if t0.elapsed().as_micros() as u64 >= budget {
                    break;
                }
                std::hint::spin_loop();
            }
            Err(_) => break, // real error surfaces in the blocking read
        }
    }
    let _ = stream.set_nonblocking(false);
}

fn push_f32s(raw: &mut Vec<u8>, xs: &[f32]) {
    for &x in xs {
        raw.extend_from_slice(&x.to_le_bytes());
    }
}

fn push_f16s(raw: &mut Vec<u8>, xs: &[f32]) {
    for &x in xs {
        raw.extend_from_slice(&f32_to_f16(x).to_le_bytes());
    }
}

fn push_floats(raw: &mut Vec<u8>, dtype: WireDtype, xs: &[f32]) {
    match dtype {
        WireDtype::F32 => push_f32s(raw, xs),
        WireDtype::F16 => push_f16s(raw, xs),
    }
}

fn pop_floats(body: &[u8], dtype: WireDtype, floats: &mut Vec<f32>) -> Result<(), String> {
    floats.clear();
    match dtype {
        WireDtype::F32 => {
            if body.len() % 4 != 0 {
                return Err(format!("float payload {} B not ×4", body.len()));
            }
            floats.reserve(body.len() / 4);
            for c in body.chunks_exact(4) {
                floats.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        WireDtype::F16 => {
            if body.len() % 2 != 0 {
                return Err(format!("float payload {} B not ×2", body.len()));
            }
            floats.reserve(body.len() / 2);
            for c in body.chunks_exact(2) {
                floats.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
    }
    Ok(())
}

/// Assemble [len][tag][body] in `raw` and ship it with ONE write syscall.
fn ship(stream: &mut TcpStream, raw: &mut Vec<u8>) -> Result<(), String> {
    let len = raw.len() as u64 - 4;
    if len > MAX_FRAME as u64 {
        return Err(format!("wire encode: frame {len} B exceeds cap {MAX_FRAME} B"));
    }
    raw[0..4].copy_from_slice(&(len as u32).to_le_bytes());
    stream
        .write_all(raw)
        .and_then(|()| stream.flush())
        .map_err(|e| format!("wire write: {e}"))
}

fn begin(raw: &mut Vec<u8>, tag: u8) {
    raw.clear();
    raw.extend_from_slice(&[0u8; 4]); // length backpatched in ship()
    raw.push(tag);
}

pub fn send_control(stream: &mut TcpStream, raw: &mut Vec<u8>, f: &Frame) -> Result<(), String> {
    begin(raw, TAG_CONTROL);
    bincode::serialize_into(&mut *raw, f).map_err(|e| format!("wire encode: {e}"))?;
    ship(stream, raw)
}

pub fn send_step(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    dtype: WireDtype,
    pos: u64,
    hidden: &[f32],
) -> Result<(), String> {
    begin(
        raw,
        match dtype {
            WireDtype::F32 => TAG_STEP_F32,
            WireDtype::F16 => TAG_STEP_F16,
        },
    );
    raw.extend_from_slice(&pos.to_le_bytes());
    push_floats(raw, dtype, hidden);
    ship(stream, raw)
}

pub fn send_hidden(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    dtype: WireDtype,
    hidden: &[f32],
) -> Result<(), String> {
    begin(
        raw,
        match dtype {
            WireDtype::F32 => TAG_HIDDEN_F32,
            WireDtype::F16 => TAG_HIDDEN_F16,
        },
    );
    push_floats(raw, dtype, hidden);
    ship(stream, raw)
}

pub fn send_prefill(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    dtype: WireDtype,
    start_pos: u64,
    count: u32,
    flat: &[f32],
) -> Result<(), String> {
    begin(
        raw,
        match dtype {
            WireDtype::F32 => TAG_PREFILL_F32,
            WireDtype::F16 => TAG_PREFILL_F16,
        },
    );
    raw.extend_from_slice(&start_pos.to_le_bytes());
    raw.extend_from_slice(&count.to_le_bytes());
    push_floats(raw, dtype, flat);
    ship(stream, raw)
}

fn rd_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}
fn rd_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().unwrap())
}

/// One message in. `Ok(None)` = clean EOF between frames (peer hung up).
/// Float payloads fill `floats`; `raw` is the reusable byte scratch.
/// `spin` = busy-poll before the blocking read (hot loops), false for
/// idle waits (a worker between sessions must not burn a core).
pub fn recv_msg(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    floats: &mut Vec<f32>,
    spin: bool,
) -> Result<Option<Msg>, String> {
    if spin {
        spin_wait(stream);
    }
    let mut len_bytes = [0u8; 4];
    match stream.read_exact(&mut len_bytes) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(format!("wire read: {e}")),
    }
    let len = u32::from_le_bytes(len_bytes);
    if len < 1 || len > MAX_FRAME {
        return Err(format!("wire read: frame {len} B outside 1..{MAX_FRAME}"));
    }
    raw.clear();
    raw.resize(len as usize, 0);
    stream
        .read_exact(raw)
        .map_err(|e| format!("wire read: {e}"))?;
    let (tag, body) = (raw[0], &raw[1..]);
    match tag {
        TAG_CONTROL => bincode::deserialize(body)
            .map(|f| Some(Msg::Control(f)))
            .map_err(|e| format!("wire decode: {e}")),
        TAG_STEP_F32 | TAG_STEP_F16 => {
            if body.len() < 8 {
                return Err("Step frame truncated".into());
            }
            let dt = if tag == TAG_STEP_F32 { WireDtype::F32 } else { WireDtype::F16 };
            pop_floats(&body[8..], dt, floats)?;
            Ok(Some(Msg::Step { pos: rd_u64(body) }))
        }
        TAG_HIDDEN_F32 | TAG_HIDDEN_F16 => {
            let dt = if tag == TAG_HIDDEN_F32 { WireDtype::F32 } else { WireDtype::F16 };
            pop_floats(body, dt, floats)?;
            Ok(Some(Msg::Hidden))
        }
        TAG_PREFILL_F32 | TAG_PREFILL_F16 => {
            if body.len() < 12 {
                return Err("Prefill frame truncated".into());
            }
            let dt = if tag == TAG_PREFILL_F32 { WireDtype::F32 } else { WireDtype::F16 };
            pop_floats(&body[12..], dt, floats)?;
            Ok(Some(Msg::Prefill {
                start_pos: rd_u64(body),
                count: rd_u32(&body[8..]),
            }))
        }
        other => Err(format!("unknown frame tag {other}")),
    }
}
