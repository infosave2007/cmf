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

pub mod beacon;
pub mod client;
pub mod nodestat;
pub mod worker;

pub use client::{generate_split, prefill_on_peer, RemoteSegment, SessionSpec, SplitStats};
pub use beacon::{discover, Beacon, Found};
pub use nodestat::NodeStats;
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
/// v5: the HEAD can live on the worker (`Assign.head`). The worker then
/// runs final norm + lm_head + sampler and answers a token ID instead of
/// a boundary hidden — measured on an Android coordinator, the head cost
/// 29 ms of a 73 ms token and does not shrink with the split, so a phone
/// driving a desktop was capped by its own head no matter how fast the
/// desktop ran. Sampling on the worker needs the id history the
/// repetition penalty reads, so `Ids` ships the prompt ids once per
/// generation; the worker appends what it samples. With `from == 0` the
/// coordinator sends `StepId` (12 bytes) instead of a hidden and the
/// worker embeds — the whole per-token wire becomes 16 bytes.
/// v6: `Assign.run_ahead` lets the worker return K tokens for one round
/// trip. In head mode with `from == 0` the coordinator contributes
/// NOTHING between tokens — the worker embeds, runs every layer, applies
/// the head and samples — so asking permission each time buys nothing and
/// costs a full round trip. This is not speculation: no draft model, no
/// verification, the same sequential decode with the handshakes removed,
/// and the output is identical token for token. It exists for Wi-Fi,
/// whose p99 round trip measured 95 ms against a cable's 2.9.
/// v7: `KvFetch` pulls the worker's state for a layer range home. This
/// is what makes "prefill over there, keep talking here" possible —
/// measured on this stand, an 1800-token prompt costs 86.9 s on the
/// phone and 11.3 s on the desktop, and the state is 224 KiB a position
/// (measured, `kv_state_bytes`), so f16 on the wire pays for itself
/// several times over. The worker refuses rather than shipping a
/// half-described cache.
pub const WIRE_VERSION: u32 = 7;

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
/// Head-on-worker frames (v5): a token id each way, dtype-free.
const TAG_STEP_ID: u8 = 7;
const TAG_TOKEN: u8 = 8;
/// v6: several tokens for one round trip.
const TAG_TOKENS: u8 = 9;
/// v7: one layer's KV/recurrent state, `[u32 layer][payload]`.
const TAG_KV: u8 = 10;

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
        /// v5: the worker owns final norm + lm_head + sampler and answers
        /// token ids. Refused unless it holds the LAST layer.
        head: bool,
        /// v5, head mode: the coordinator's `SamplerConfig` as JSON. The
        /// worker MUST sample with the caller's settings — a default
        /// temperature on the far side would silently rewrite the answer.
        /// JSON rather than the struct so adding a sampler field cannot
        /// change the frame encoding behind the version check.
        sampler: Option<String>,
        /// v6: how many tokens the worker may generate for one request.
        /// 0 or 1 = the classic one-token round trip. Only legal with
        /// `head` and `from == 0`, because anything else needs the
        /// coordinator's own layers between tokens.
        run_ahead: u32,
    },
    /// v5, head mode only: the id history the sampler's repetition
    /// penalty reads. Sent once per generation, before the first Step;
    /// the worker appends every id it samples after that.
    Ids { ids: Vec<u32> },
    /// Fresh sequence on the worker (clears its KV and state).
    Reset,
    /// Prefill barrier: the worker replies `Hidden` with the output of
    /// the LAST prefill position it has processed.
    Sync,
    Ping,
    Pong,
    /// v7: send this layer range's KV and recurrent state home. The
    /// worker answers `Ack` (empty = accepted) and then one `Kv` frame
    /// per layer, in order.
    KvFetch { from: u32, upto: u32 },
    /// v5: ask the worker what it is worth right now. Cheap enough to
    /// send between turns; the answer is measured, never declared.
    Stats,
    /// v5: the worker's live capacity signals.
    StatsReply(crate::nodestat::NodeStats),
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
    /// v5: decode step carrying the token id itself — only legal when the
    /// worker holds layer 0 and does the embedding. v6 adds `want`: how
    /// many tokens the coordinator can still use, so a run-ahead batch
    /// never advances the worker's KV past what the caller will emit.
    StepId { pos: u64, id: u32, want: u32 },
    /// v5: the worker's sampled token.
    Token { id: u32 },
    /// v7: one layer's state. The payload is NOT copied — it stays in the
    /// caller's `raw` scratch at `raw[5..]`, because a whole cache is
    /// hundreds of megabytes and copying it twice is the difference
    /// between a transfer and an out-of-memory.
    Kv { layer: u32 },
    /// v6: the run-ahead batch. Fewer than requested means the worker
    /// stopped early — end of sequence — and the coordinator must not
    /// ask for more.
    Tokens { ids: Vec<u32>, eos: bool },
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

/// v5 head mode with the worker holding layer 0: the id goes out, the
/// worker embeds it. Twelve bytes where a hidden was kilobytes.
pub fn send_step_id(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    pos: u64,
    id: u32,
    want: u32,
) -> Result<(), String> {
    begin(raw, TAG_STEP_ID);
    raw.extend_from_slice(&pos.to_le_bytes());
    raw.extend_from_slice(&id.to_le_bytes());
    raw.extend_from_slice(&want.to_le_bytes());
    ship(stream, raw)
}

/// v5 head mode: the worker's sampled token, four bytes.
pub fn send_token(stream: &mut TcpStream, raw: &mut Vec<u8>, id: u32) -> Result<(), String> {
    begin(raw, TAG_TOKEN);
    raw.extend_from_slice(&id.to_le_bytes());
    ship(stream, raw)
}

/// v7: one layer's state, raw after a 4-byte layer index.
pub fn send_kv(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    layer: u32,
    payload: &[u8],
) -> Result<(), String> {
    begin(raw, TAG_KV);
    raw.extend_from_slice(&layer.to_le_bytes());
    raw.extend_from_slice(payload);
    ship(stream, raw)
}

/// v6: a run-ahead batch — `[eos u8][count u32][ids…]`.
pub fn send_tokens(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    ids: &[u32],
    eos: bool,
) -> Result<(), String> {
    begin(raw, TAG_TOKENS);
    raw.push(u8::from(eos));
    raw.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for &id in ids {
        raw.extend_from_slice(&id.to_le_bytes());
    }
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
        TAG_STEP_ID => {
            if body.len() < 16 {
                return Err("StepId frame truncated".into());
            }
            floats.clear();
            Ok(Some(Msg::StepId {
                pos: rd_u64(body),
                id: rd_u32(&body[8..]),
                want: rd_u32(&body[12..]),
            }))
        }
        TAG_KV => {
            if body.len() < 4 {
                return Err("Kv frame truncated".into());
            }
            floats.clear();
            Ok(Some(Msg::Kv { layer: rd_u32(body) }))
        }
        TAG_TOKENS => {
            if body.len() < 5 {
                return Err("Tokens frame truncated".into());
            }
            let eos = body[0] != 0;
            let n = rd_u32(&body[1..]) as usize;
            if body.len() < 5 + n * 4 {
                return Err(format!("Tokens frame claims {n} ids, body is {} B", body.len()));
            }
            let ids = (0..n).map(|i| rd_u32(&body[5 + i * 4..])).collect();
            floats.clear();
            Ok(Some(Msg::Tokens { ids, eos }))
        }
        TAG_TOKEN => {
            if body.len() < 4 {
                return Err("Token frame truncated".into());
            }
            floats.clear();
            Ok(Some(Msg::Token { id: rd_u32(body) }))
        }
        other => Err(format!("unknown frame tag {other}")),
    }
}
