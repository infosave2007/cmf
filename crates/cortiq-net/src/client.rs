//! The coordinator side: `RemoteSegment` speaks the wire protocol to one
//! worker; `generate_split` is the split twin of `generate_from_ids` —
//! deliberately the PLAIN path (no speculation, no whole-token graph, no
//! task masks): correctness and honest timing first, the fast paths
//! return per the roadmap once this one is measured.

use crate::{
    recv_msg, send_control, send_prefill, send_step, send_step_id, Frame, Msg, O1Wire, WireDtype,
    WIRE_VERSION,
};
use cortiq_core::TaskMask;
use cortiq_engine::pipeline::{GenerateResult, Pipeline, TokenCallback};
use std::net::TcpStream;
use std::time::Instant;

/// The model configuration a split session runs under — shipped in
/// Assign so the worker mirrors it over its own layers.
#[derive(Clone, Default)]
pub struct SessionSpec {
    /// Skill overlay id (worker reloads its pipeline with it).
    pub skill: Option<String>,
    /// Task-mask name (each side masks the layers IT runs).
    pub task: Option<String>,
    /// O(1)-attention config (worker mirrors it over its span).
    pub o1: Option<O1Wire>,
    /// v5: hand the worker the final norm + lm_head + sampler. It answers
    /// token ids, so the coordinator's per-token cost drops to detokenizing
    /// — measured worth 29 ms of a 73 ms token on an Android coordinator.
    /// Requires the worker to hold the last layer.
    pub head: bool,
    /// v5 head mode: the coordinator's `SamplerConfig` as JSON, so the far
    /// side samples with the caller's settings and not its defaults.
    pub sampler: Option<String>,
    /// v6: how many tokens the worker may return for one round trip.
    /// 0/1 keeps the classic one-token exchange. Needs `head` and a peer
    /// holding the whole stack; the worker refuses it otherwise, loudly.
    pub run_ahead: u32,
}

pub struct RemoteSegment {
    stream: TcpStream,
    pub addr: String,
    /// Layers [from ..= upto] the worker runs.
    pub from: usize,
    pub upto: usize,
    /// Payload dtype both directions (negotiated at Assign).
    pub dtype: WireDtype,
    /// Accumulated wall time inside round trips (the wire + remote compute).
    pub net_s: f64,
    /// v5: the worker owns the head and answers token ids.
    pub head: bool,
    /// v6: tokens per round trip (1 = classic).
    pub run_ahead: usize,
    /// Ids from the last run-ahead batch, and whether it ended on EOS.
    batch: Vec<u32>,
    pub batch_eos: bool,
    raw: Vec<u8>,
    floats: Vec<f32>,
}

impl RemoteSegment {
    /// Connect, handshake (wire version, token, dir_hash, geometry), and
    /// assign the worker its span + wire dtype. Every refusal arrives as
    /// a worker message, not a hang.
    #[allow(clippy::too_many_arguments)]
    pub fn connect(
        addr: &str,
        token: &str,
        dir_hash: u64,
        arch: &str,
        num_layers: usize,
        hidden_size: usize,
        from: usize,
        upto: usize,
        dtype: WireDtype,
        spec: &SessionSpec,
    ) -> Result<Self, String> {
        let stream = TcpStream::connect(addr).map_err(|e| format!("connect {addr}: {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("set_nodelay: {e}"))?;
        let mut rs = Self {
            stream,
            addr: addr.to_string(),
            from,
            upto,
            dtype,
            net_s: 0.0,
            head: spec.head,
            run_ahead: spec.run_ahead.max(1) as usize,
            batch: Vec::new(),
            batch_eos: false,
            raw: Vec::with_capacity(64 * 1024),
            floats: Vec::new(),
        };
        rs.send_ctl(&Frame::Hello {
            wire: WIRE_VERSION,
            token: token.to_string(),
            dir_hash,
            arch: arch.to_string(),
            num_layers: num_layers as u32,
            hidden_size: hidden_size as u32,
        })?;
        rs.expect_ack("hello")?;
        rs.send_ctl(&Frame::Assign {
            from: from as u32,
            upto: upto as u32,
            dtype: dtype.code(),
            skill: spec.skill.clone(),
            task: spec.task.clone(),
            o1: spec.o1.clone(),
            head: spec.head,
            sampler: spec.sampler.clone(),
            run_ahead: spec.run_ahead,
        })?;
        rs.expect_ack("assign")?;
        Ok(rs)
    }

    fn send_ctl(&mut self, f: &Frame) -> Result<(), String> {
        send_control(&mut self.stream, &mut self.raw, f)
    }

    fn recv(&mut self, spin: bool) -> Result<Msg, String> {
        let mut floats = std::mem::take(&mut self.floats);
        let r = recv_msg(&mut self.stream, &mut self.raw, &mut floats, spin);
        self.floats = floats;
        match r? {
            Some(Msg::Control(Frame::Err { msg })) => Err(format!("worker {}: {msg}", self.addr)),
            Some(m) => Ok(m),
            None => Err(format!("worker {} hung up", self.addr)),
        }
    }

    fn expect_ack(&mut self, what: &str) -> Result<(), String> {
        match self.recv(false)? {
            Msg::Control(Frame::Ack { err }) if err.is_empty() => Ok(()),
            Msg::Control(Frame::Ack { err }) => {
                Err(format!("worker {} refused {what}: {err}", self.addr))
            }
            other => Err(format!(
                "worker {}: expected Ack after {what}, got {other:?}",
                self.addr
            )),
        }
    }

    /// Wait for a Hidden reply; the payload lands in `self.floats`.
    fn expect_hidden(&mut self) -> Result<(), String> {
        match self.recv(true)? {
            Msg::Hidden => Ok(()),
            other => Err(format!(
                "worker {}: expected Hidden, got {other:?}",
                self.addr
            )),
        }
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.send_ctl(&Frame::Reset)?;
        self.expect_ack("reset")
    }

    /// Ship `count` boundary hiddens (flattened) fire-and-forget: the
    /// worker chews them while we compute the next chunk; `sync()` is
    /// the barrier.
    pub fn prefill_send(
        &mut self,
        start_pos: usize,
        count: usize,
        flat: &[f32],
    ) -> Result<(), String> {
        let mut raw = std::mem::take(&mut self.raw);
        let r = send_prefill(
            &mut self.stream,
            &mut raw,
            self.dtype,
            start_pos as u64,
            count as u32,
            flat,
        );
        self.raw = raw;
        r
    }

    /// Prefill barrier: the last prefill position's output lands in
    /// `self.floats` (returned as a slice).
    pub fn sync(&mut self) -> Result<&[f32], String> {
        self.send_ctl(&Frame::Sync)?;
        self.expect_hidden()?;
        Ok(&self.floats)
    }

    /// One decode step; the boundary hidden goes out, the worker's output
    /// comes back in `self.floats` (returned as a slice).
    pub fn step(&mut self, pos: usize, hidden: &[f32]) -> Result<&[f32], String> {
        let t = Instant::now();
        let mut raw = std::mem::take(&mut self.raw);
        let r = send_step(&mut self.stream, &mut raw, self.dtype, pos as u64, hidden);
        self.raw = raw;
        r?;
        self.expect_hidden()?;
        self.net_s += t.elapsed().as_secs_f64();
        Ok(&self.floats)
    }

    /// v7: pull the worker's KV and recurrent state for a layer range
    /// into `p`. Returns the bytes that crossed the wire.
    pub fn fetch_kv(
        &mut self,
        p: &mut Pipeline,
        from: usize,
        upto: usize,
    ) -> Result<usize, String> {
        self.send_ctl(&Frame::KvFetch {
            from: from as u32,
            upto: upto as u32,
        })?;
        self.expect_ack("kv fetch")?;
        let mut bytes = 0usize;
        for expect in from..=upto {
            match self.recv(false)? {
                Msg::Kv { layer } => {
                    let li = layer as usize;
                    if li != expect {
                        return Err(format!(
                            "worker {}: KV arrived for layer {li}, expected {expect}",
                            self.addr
                        ));
                    }
                    // The payload lives in the scratch the frame landed
                    // in — hundreds of megabytes, never copied twice.
                    let payload = &self.raw[5..];
                    bytes += payload.len();
                    p.kv_cache.layers[li]
                        .import_wire(payload)
                        .map_err(|e| format!("layer {li}: {e}"))?;
                }
                other => {
                    return Err(format!(
                        "worker {}: expected Kv for layer {expect}, got {other:?}",
                        self.addr
                    ));
                }
            }
        }
        Ok(bytes)
    }

    /// What the worker is worth right now — thermal state, mains power,
    /// the clock its fastest core is actually running at. The planner
    /// reads this instead of trusting a number measured at connect time.
    pub fn stats(&mut self) -> Result<crate::NodeStats, String> {
        self.send_ctl(&Frame::Stats)?;
        match self.recv(false)? {
            Msg::Control(Frame::StatsReply(s)) => Ok(s),
            other => Err(format!(
                "worker {}: expected StatsReply, got {other:?}",
                self.addr
            )),
        }
    }

    /// Head mode: hand the worker the id history its repetition penalty
    /// reads. Once per generation, before the first sample.
    pub fn send_ids(&mut self, ids: &[u32]) -> Result<(), String> {
        self.send_ctl(&Frame::Ids { ids: ids.to_vec() })?;
        self.expect_ack("ids")
    }

    fn expect_token(&mut self) -> Result<u32, String> {
        match self.recv(true)? {
            Msg::Token { id } => Ok(id),
            other => Err(format!("worker {}: expected Token, got {other:?}", self.addr)),
        }
    }

    /// Head mode: the prefill barrier returns the FIRST sampled token
    /// rather than a boundary hidden — the worker owns the sampler.
    pub fn sync_token(&mut self) -> Result<u32, String> {
        self.send_ctl(&Frame::Sync)?;
        self.expect_token()
    }

    /// Head mode with a local span: boundary hidden out, token id back.
    pub fn step_token(&mut self, pos: usize, hidden: &[f32]) -> Result<u32, String> {
        let t = Instant::now();
        let mut raw = std::mem::take(&mut self.raw);
        let r = send_step(&mut self.stream, &mut raw, self.dtype, pos as u64, hidden);
        self.raw = raw;
        r?;
        let id = self.expect_token()?;
        self.net_s += t.elapsed().as_secs_f64();
        Ok(id)
    }

    /// v6: one round trip, up to `run_ahead` tokens back. The ids are the
    /// same the one-at-a-time path would have produced — the worker runs
    /// the identical sequential decode, it just stops asking permission.
    pub fn step_id_many(&mut self, pos: usize, id: u32, want: usize) -> Result<&[u32], String> {
        let t = Instant::now();
        let mut raw = std::mem::take(&mut self.raw);
        let r = send_step_id(&mut self.stream, &mut raw, pos as u64, id, want as u32);
        self.raw = raw;
        r?;
        match self.recv(true)? {
            Msg::Tokens { ids, eos } => {
                if ids.is_empty() {
                    return Err(format!("worker {}: empty run-ahead batch", self.addr));
                }
                self.batch = ids;
                self.batch_eos = eos;
            }
            other => {
                return Err(format!(
                    "worker {}: expected Tokens, got {other:?}",
                    self.addr
                ));
            }
        }
        self.net_s += t.elapsed().as_secs_f64();
        Ok(&self.batch)
    }

    /// Head mode with the worker holding every layer: the id goes out and
    /// the next id comes back. Sixteen bytes of wire for a whole token.
    pub fn step_id(&mut self, pos: usize, id: u32) -> Result<u32, String> {
        let t = Instant::now();
        let mut raw = std::mem::take(&mut self.raw);
        let r = send_step_id(&mut self.stream, &mut raw, pos as u64, id, 1);
        self.raw = raw;
        r?;
        let next = self.expect_token()?;
        self.net_s += t.elapsed().as_secs_f64();
        Ok(next)
    }
}

/// Measured split telemetry for the caller to print — the numbers that
/// decide whether this topology pays on this wire.
pub struct SplitStats {
    pub prefill_s: f64,
    pub decode_s: f64,
    /// Time inside remote round trips (wire + remote compute), all phases.
    pub net_s: f64,
    pub remote_steps: usize,
    /// Positions actually prefilled this call (< prompt length when the
    /// cross-turn KV reuse kicked in).
    pub prefilled: usize,
}

/// Split generation: coordinator runs layers [0..remote.from) plus
/// embed / final norm / head / sampler; the worker runs the rest.
/// `remote.from == 0` is legal (worker runs every layer; the coordinator
/// still embeds and samples) — useful for loading a remote box fully.
pub fn generate_split(
    p: &mut Pipeline,
    remote: &mut RemoteSegment,
    input_ids: &[u32],
    max_tokens: usize,
    task_mask: Option<&TaskMask>,
    mut on_token: Option<TokenCallback>,
) -> Result<(GenerateResult, SplitStats), String> {
    p.split_supported()?;
    if input_ids.is_empty() {
        return Err("empty prompt: nothing to generate from".to_string());
    }
    if remote.upto + 1 != p.num_layers {
        return Err(format!(
            "v1 topology holds the TAIL on the worker: upto {} must be the last layer {}",
            remote.upto,
            p.num_layers - 1
        ));
    }
    let split = remote.from;
    let hs = p.hidden_size;

    // Cross-turn KV reuse, mirroring generate_from_ids: when the new ids
    // strictly EXTEND what both sides already hold, prefill only the
    // tail — chat turn latency stays proportional to the new text, not
    // the whole session. Extension-only (no rollback); the worker's KV
    // advanced in lockstep, so its state is the coordinator's history.
    let reuse_from = {
        let on = !std::env::var("CMF_KV_REUSE").is_ok_and(|v| v == "0");
        let h = &p.kv_history;
        // o1 excluded, as locally: the Nyström insertion is irreversible
        // and re-sealing over a reused prefix is not the same state.
        if on
            && !p.o1_active()
            && task_mask.is_none()
            && !h.is_empty()
            && h.len() < input_ids.len()
            && input_ids[..h.len()] == h[..]
        {
            h.len()
        } else {
            0
        }
    };
    if reuse_from == 0 {
        p.reset_session();
        p.o1_begin();
        remote.reset()?;
    }
    remote.net_s = 0.0;

    // ── Prefill: pipelined — the worker chews chunk k while we compute
    // chunk k+1; one barrier (Sync) at the end. Wall ≈ max(sides), not sum.
    let t_prefill = Instant::now();
    let chunk = cortiq_engine::pipeline::prefill_chunk();
    let mut flat: Vec<f32> = Vec::with_capacity(chunk * hs);
    let mut pos = reuse_from;
    while pos < input_ids.len() {
        let end = (pos + chunk).min(input_ids.len());
        if split > 0 {
            // Layer-major batched walk over the local span — the same
            // GEMM/graph machinery as local prefill, cut at the boundary.
            flat = p.prefill_span_ids(&input_ids[pos..end], pos, split - 1, task_mask)?;
        } else {
            flat.clear();
            for &id in &input_ids[pos..end] {
                flat.extend_from_slice(&p.embed_id(id));
            }
        }
        remote.prefill_send(pos, end - pos, &flat)?;
        pos = end;
        if p.cancel.swap(false, std::sync::atomic::Ordering::Relaxed) {
            // Partial prompt in both KVs — poison the reuse history.
            p.kv_history.clear();
            return Ok((
                result(String::new(), Vec::new(), input_ids.len(), "cancelled"),
                stats(t_prefill.elapsed().as_secs_f64(), 0.0, remote),
            ));
        }
    }
    // Prompt absorbed on the local span — freeze o1 skeletons exactly
    // where the local path would; the worker seals at Sync.
    p.o1_seal();
    // Head mode samples on the worker, so it needs the prompt ids its
    // repetition penalty reads BEFORE the barrier that samples token one.
    let first_token = if remote.head {
        remote.send_ids(input_ids)?;
        Some(remote.sync_token()?)
    } else {
        None
    };
    let last_hidden = match first_token {
        Some(_) => Vec::new(),
        None => remote.sync()?.to_vec(),
    };
    let prefill_s = t_prefill.elapsed().as_secs_f64();
    // From here net_s counts DECODE round trips only — the honest share.
    remote.net_s = 0.0;

    // ── Decode: one local span + one round trip + head per token ──
    let t_decode = Instant::now();
    let mut all_ids = input_ids.to_vec();
    let mut token_ids = Vec::new();
    let mut text = String::new();
    let mut finish_reason = "max_tokens".to_string();
    let mut remote_steps = 0usize;

    let mut next = match first_token {
        Some(id) => id,
        None => {
            let logits = p.logits_from_hidden(&last_hidden);
            p.sample_next(&logits, &all_ids)
        }
    };
    let mut next_pos = input_ids.len();
    // Run-ahead tokens already fetched and not yet committed.
    let mut pending: std::collections::VecDeque<u32> = std::collections::VecDeque::new();

    while token_ids.len() < max_tokens {
        // Commit `next` (mirrors generate_from_ids: EOS stops unemitted).
        all_ids.push(next);
        token_ids.push(next);
        if p.tokenizer.is_eos(next) {
            finish_reason = "stop".to_string();
            break;
        }
        let piece = p.tokenizer.decode_token(next);
        text.push_str(&piece);
        if let Some(cb) = on_token.as_mut() {
            if !cb(&piece) {
                finish_reason = "cancelled".to_string();
                break;
            }
        }
        if p.cancel.swap(false, std::sync::atomic::Ordering::Relaxed) {
            finish_reason = "cancelled".to_string();
            break;
        }
        if token_ids.len() == max_tokens {
            break;
        }

        if remote.head && split == 0 {
            // Thin client: the worker embeds, runs every layer, and
            // samples. Sixteen bytes of wire and no matmul on this side.
            if remote.run_ahead > 1 {
                // Nothing here happens between tokens, so nothing here
                // needs to be asked between tokens: one round trip
                // brings back as many as the caller can still use. The
                // ids are what the one-at-a-time path would have given —
                // same sequential decode, fewer handshakes.
                if pending.is_empty() {
                    let want = max_tokens - token_ids.len();
                    let got = remote.step_id_many(next_pos, next, want)?;
                    pending.extend(got.iter().copied());
                    remote_steps += 1;
                    // The worker consumed one position per token it
                    // produced; the next request starts after them.
                    next_pos += pending.len();
                }
                next = pending.pop_front().expect("batch was non-empty");
            } else {
                next = remote.step_id(next_pos, next)?;
                remote_steps += 1;
                next_pos += 1;
            }
        } else {
            let emb = p.embed_id(next);
            let boundary = if split > 0 {
                p.forward_span(&emb, next_pos, 0, split - 1, task_mask)?
            } else {
                emb
            };
            if remote.head {
                next = remote.step_token(next_pos, &boundary)?;
            } else {
                remote.step(next_pos, &boundary)?;
                let logits = p.logits_from_hidden(&remote.floats);
                next = p.sample_next(&logits, &all_ids);
            }
            remote_steps += 1;
            next_pos += 1;
        }
    }

    let decode_s = t_decode.elapsed().as_secs_f64();
    // The forwarded prefix = everything except the last committed token
    // (sampled but never forwarded) — the exact state both KVs hold, and
    // the reuse key for the next turn.
    let fwd = all_ids.len() - usize::from(!token_ids.is_empty());
    p.kv_history = all_ids[..fwd].to_vec();
    let mut st = stats(prefill_s, decode_s, remote);
    st.remote_steps = remote_steps;
    st.prefilled = input_ids.len() - reuse_from;
    Ok((result(text, token_ids, input_ids.len(), &finish_reason), st))
}

/// Prefill the whole prompt on the peer and bring the state home, so the
/// caller can keep talking with the cable unplugged.
///
/// The asymmetry this exists for, measured on a phone against a laptop:
/// 1800 positions cost 86.9 s locally and 11.3 s over there, because
/// prefill ships whole chunks (7 round trips for the lot) while decode
/// pays one round trip per token. The state that comes back is 224 KiB a
/// position, so the wire dtype decides whether the trade pays — f16 on
/// the wire is the caller's explicit choice, exactly as for hiddens.
///
/// The last prompt position is deliberately NOT prefilled remotely: the
/// local reuse contract wants a history strictly shorter than the prompt
/// (`kv_history.len() < ids.len()`), so the caller's own generate absorbs
/// one position and everything downstream — MTP, o1, task masks — stays
/// on the path it already trusts.
pub fn prefill_on_peer(
    p: &mut Pipeline,
    remote: &mut RemoteSegment,
    input_ids: &[u32],
) -> Result<(usize, f64, f64), String> {
    if remote.from != 0 || remote.upto + 1 != p.num_layers {
        return Err(format!(
            "prefill offload needs the peer to hold the WHOLE stack (it has {}..={} of {})",
            remote.from,
            remote.upto,
            p.num_layers - 1
        ));
    }
    if input_ids.len() < 2 {
        return Err("prefill offload wants a prompt worth offloading".into());
    }
    let hs = p.hidden_size;
    let take = input_ids.len() - 1;
    p.reset_session();
    p.o1_begin();
    remote.reset()?;

    let t0 = Instant::now();
    let chunk = cortiq_engine::pipeline::prefill_chunk();
    let mut flat: Vec<f32> = Vec::with_capacity(chunk * hs);
    let mut pos = 0usize;
    while pos < take {
        let end = (pos + chunk).min(take);
        flat.clear();
        for &id in &input_ids[pos..end] {
            flat.extend_from_slice(&p.embed_id(id));
        }
        remote.prefill_send(pos, end - pos, &flat)?;
        pos = end;
    }
    remote.sync()?;
    let prefill_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let bytes = remote.fetch_kv(p, 0, p.num_layers - 1)?;
    let fetch_s = t1.elapsed().as_secs_f64();
    // The cache now holds exactly these positions; the reuse check reads
    // this and prefills only what is left.
    p.kv_history = input_ids[..take].to_vec();
    Ok((bytes, prefill_s, fetch_s))
}

fn result(
    text: String,
    token_ids: Vec<u32>,
    prompt_tokens: usize,
    finish_reason: &str,
) -> GenerateResult {
    GenerateResult {
        text,
        tokens_generated: token_ids.len(),
        token_ids,
        prompt_tokens,
        finish_reason: finish_reason.to_string(),
        mtp_drafted: 0,
        mtp_accepted: 0,
        token_confidence: Vec::new(),
        traces: Vec::new(),
    }
}

fn stats(prefill_s: f64, decode_s: f64, remote: &RemoteSegment) -> SplitStats {
    SplitStats {
        prefill_s,
        decode_s,
        net_s: remote.net_s,
        remote_steps: 0,
        prefilled: 0,
    }
}
