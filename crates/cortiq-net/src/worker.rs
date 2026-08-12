//! The worker side: hold a layer span of the shared model, serve one
//! coordinator at a time over TCP. Blocking I/O on purpose — a worker
//! has exactly one conversation partner and the hot path is a single
//! request/reply per token (with a bounded busy-poll before each read:
//! the wakeup from a cold blocking read costs more than the wire).

use crate::{
    recv_msg, send_control, send_hidden, send_kv, send_token, send_tokens, Frame, Msg, O1Wire,
    WireDtype, WIRE_VERSION,
};
use cortiq_core::TaskMask;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::sampler::SamplerConfig;
use std::net::{TcpListener, TcpStream};

pub struct WorkerConfig {
    pub model_path: String,
    pub listen: String,
    /// Shared secret; REQUIRED when listening beyond loopback.
    pub token: Option<String>,
}

/// Load the model and serve layer spans until killed. Library entry on
/// purpose: platforms that cannot spawn a binary (an iOS worker app)
/// call this from FFI; the CLI `cortiq worker` is a thin wrapper.
pub fn worker_serve(cfg: WorkerConfig) -> Result<(), String> {
    let loopback = cfg.listen.starts_with("127.")
        || cfg.listen.starts_with("localhost")
        || cfg.listen.starts_with("[::1]");
    if !loopback && cfg.token.is_none() {
        return Err(format!(
            "worker: refusing to listen on {} without --token (loopback needs none)",
            cfg.listen
        ));
    }

    let model = std::sync::Arc::new(
        cortiq_core::CmfModel::open_sharded(&cfg.model_path).map_err(|e| e.to_string())?,
    );
    let dir_hash = model.dir_hash();
    let mut pipeline =
        Pipeline::from_model(&model, SamplerConfig::default()).map_err(|e| e.to_string())?;
    // A span holder never decodes on its own; the coordinator samples.
    pipeline.split_supported()?;

    let listener =
        TcpListener::bind(&cfg.listen).map_err(|e| format!("bind {}: {e}", cfg.listen))?;
    eprintln!(
        "worker: {} | layers {} | hidden {} | dir_hash {:016x} | wire v{} | listening on {}",
        model.arch().arch_name,
        pipeline.num_layers,
        pipeline.hidden_size,
        dir_hash,
        WIRE_VERSION,
        cfg.listen
    );

    // The active overlay: rebuilding a pipeline is the STATIC skill path
    // (correct for every skill kind); the id is cached so a same-skill
    // session reuses the loaded one.
    let mut cur_skill: Option<String> = None;

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("worker: accept failed: {e}");
                continue;
            }
        };
        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "?".into());
        eprintln!("worker: coordinator connected from {peer}");
        match serve_one(
            &model,
            &mut pipeline,
            &mut cur_skill,
            stream,
            dir_hash,
            cfg.token.as_deref(),
        ) {
            Ok(()) => eprintln!("worker: {peer} disconnected"),
            Err(e) => eprintln!("worker: session with {peer} failed: {e}"),
        }
        // Next coordinator starts from a clean sequence either way.
        pipeline.reset_session();
    }
    Ok(())
}

/// Apply an Assign's session spec: reload with the skill if it changed,
/// mirror the o1 config, resolve the task mask. Every failure is a
/// message for the Ack, never a silent downgrade.
fn apply_spec(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    p: &mut Pipeline,
    cur_skill: &mut Option<String>,
    skill: &Option<String>,
    task: &Option<String>,
    o1: &Option<O1Wire>,
) -> Result<Option<TaskMask>, String> {
    if skill != cur_skill {
        *p = Pipeline::from_model_with_skill(model, SamplerConfig::default(), skill.as_deref())
            .map_err(|e| format!("skill {:?}: {e}", skill.as_deref().unwrap_or("<backbone>")))?;
        p.split_supported()?;
        *cur_skill = skill.clone();
        eprintln!(
            "worker: pipeline reloaded with skill {}",
            skill.as_deref().unwrap_or("<backbone>")
        );
    }
    match o1 {
        Some(w) => {
            let rect = match w.rect.as_deref() {
                None => None,
                Some(s) => Some(
                    cortiq_engine::nystrom::O1Cfg::parse_rect(s)
                        .ok_or_else(|| format!("o1 rect '{s}' unknown"))?,
                ),
            };
            let cfg = cortiq_engine::nystrom::O1Cfg::from_spec(
                &w.spec,
                w.m.map(|v| v as usize),
                w.w.map(|v| v as usize),
                w.sink.map(|v| v as usize),
                rect,
            );
            if cfg.is_none() && w.spec != "off" {
                return Err(format!("o1 spec '{}' did not parse", w.spec));
            }
            p.set_o1(cfg);
        }
        None => p.set_o1(None),
    }
    match task {
        Some(t) => model
            .masks
            .get(t)
            .cloned()
            .map(Some)
            .ok_or_else(|| format!("task mask '{t}' not in this file's catalog")),
        None => Ok(None),
    }
}

/// Head mode: final norm + lm_head + sampler, exactly the three steps the
/// coordinator would have run, in the same order and off the same
/// history — so greedy text is identical to the head-local split.
fn sample_here(p: &mut Pipeline, hist: &mut Vec<u32>, hidden: &[f32]) -> u32 {
    let logits = p.logits_from_hidden(hidden);
    let id = p.sample_next(&logits, hist);
    hist.push(id);
    id
}

fn serve_one(
    model: &std::sync::Arc<cortiq_core::CmfModel>,
    p: &mut Pipeline,
    cur_skill: &mut Option<String>,
    mut stream: TcpStream,
    dir_hash: u64,
    token: Option<&str>,
) -> Result<(), String> {
    stream
        .set_nodelay(true)
        .map_err(|e| format!("set_nodelay: {e}"))?;

    // Reusable scratch: the per-token path allocates nothing.
    let mut raw: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut floats: Vec<f32> = Vec::new();
    let mut out: Vec<u8> = Vec::with_capacity(64 * 1024);

    let mut refuse =
        |stream: &mut TcpStream, out: &mut Vec<u8>, msg: String| -> Result<(), String> {
            let _ = send_control(stream, out, &Frame::Ack { err: msg.clone() });
            Err(msg)
        };

    // ── Handshake ──
    match recv_msg(&mut stream, &mut raw, &mut floats, false)? {
        Some(Msg::Control(Frame::Hello {
            wire,
            token: t,
            dir_hash: dh,
            arch,
            num_layers,
            hidden_size,
        })) => {
            if wire != WIRE_VERSION {
                return refuse(
                    &mut stream,
                    &mut out,
                    format!("wire version {wire} ≠ {WIRE_VERSION} — upgrade one side"),
                );
            }
            if let Some(want) = token {
                if t != want {
                    return refuse(&mut stream, &mut out, "bad --token".to_string());
                }
            }
            if dh != dir_hash {
                return refuse(
                    &mut stream,
                    &mut out,
                    format!(
                        "model mismatch: coordinator dir_hash {dh:016x} ≠ mine {dir_hash:016x} \
                         — both sides must hold the SAME .cmf"
                    ),
                );
            }
            if num_layers as usize != p.num_layers || hidden_size as usize != p.hidden_size {
                return refuse(
                    &mut stream,
                    &mut out,
                    format!(
                        "geometry mismatch: peer {num_layers}L/{hidden_size}h ≠ mine {}L/{}h (arch {arch})",
                        p.num_layers, p.hidden_size
                    ),
                );
            }
            send_control(&mut stream, &mut out, &Frame::Ack { err: String::new() })?;
        }
        Some(other) => return Err(format!("expected Hello, got {other:?}")),
        None => return Ok(()), // probe connection, clean hangup
    }

    // ── Session ──
    let mut span: Option<(usize, usize)> = None;
    let mut dtype = WireDtype::F32;
    let mut mask: Option<TaskMask> = None;
    let mut last_prefill: Vec<f32> = Vec::new();
    // v5: head mode — we own final norm + lm_head + sampler, and `hist`
    // is the id history the repetition penalty reads (prompt via `Ids`,
    // then every id we sample).
    let mut head = false;
    let mut run_ahead = 1usize;
    let mut hist: Vec<u32> = Vec::new();
    let hs = p.hidden_size;
    let fatal = |stream: &mut TcpStream, out: &mut Vec<u8>, msg: String| -> Result<(), String> {
        let _ = send_control(stream, out, &Frame::Err { msg: msg.clone() });
        Err(msg)
    };
    loop {
        // Spin only when a generation is live (span assigned): between
        // sessions the worker sits in a plain blocking read, zero CPU.
        let msg = match recv_msg(&mut stream, &mut raw, &mut floats, span.is_some())? {
            Some(m) => m,
            None => return Ok(()), // clean disconnect
        };
        match msg {
            Msg::Control(Frame::Assign {
                from,
                upto,
                dtype: dt,
                skill,
                task,
                o1,
                head: want_head,
                sampler,
                run_ahead: want_ahead,
            }) => {
                let (from, upto) = (from as usize, upto as usize);
                if from > upto || upto >= p.num_layers {
                    send_control(
                        &mut stream,
                        &mut out,
                        &Frame::Ack {
                            err: format!("assign {from}..={upto} outside 0..{}", p.num_layers),
                        },
                    )?;
                    continue;
                }
                dtype = match WireDtype::from_code(dt) {
                    Ok(d) => d,
                    Err(e) => {
                        send_control(&mut stream, &mut out, &Frame::Ack { err: e })?;
                        continue;
                    }
                };
                // The head sits after the LAST layer: a worker that does
                // not hold it cannot produce logits. Refuse loudly.
                if want_head && upto + 1 != p.num_layers {
                    send_control(
                        &mut stream,
                        &mut out,
                        &Frame::Ack {
                            err: format!(
                                "head requested but my span ends at {upto}, not the last layer {}",
                                p.num_layers - 1
                            ),
                        },
                    )?;
                    continue;
                }
                match apply_spec(model, p, cur_skill, &skill, &task, &o1) {
                    Ok(m) => mask = m,
                    Err(e) => {
                        send_control(&mut stream, &mut out, &Frame::Ack { err: e })?;
                        continue;
                    }
                }
                // apply_spec may have rebuilt the pipeline with a skill —
                // the sampler config has to land AFTER that, or a reload
                // silently restores the default temperature.
                if want_head {
                    let json = sampler.as_deref().unwrap_or("{}");
                    match serde_json::from_str::<SamplerConfig>(json) {
                        Ok(cfg) => p.set_sampler_config(cfg),
                        Err(e) => {
                            send_control(
                                &mut stream,
                                &mut out,
                                &Frame::Ack {
                                    err: format!("sampler config: {e}"),
                                },
                            )?;
                            continue;
                        }
                    }
                }
                // Run-ahead needs the coordinator to be out of the loop
                // entirely: we embed, we run every layer, we sample. With
                // a local span on the other side there is a matmul
                // between tokens that only it can do.
                if want_ahead > 1 && !(want_head && from == 0) {
                    send_control(
                        &mut stream,
                        &mut out,
                        &Frame::Ack {
                            err: format!(
                                "run_ahead {want_ahead} needs head mode over the whole stack                                  (head {want_head}, span starts at {from})"
                            ),
                        },
                    )?;
                    continue;
                }
                head = want_head;
                run_ahead = want_ahead.max(1) as usize;
                hist.clear();
                span = Some((from, upto));
                eprintln!(
                    "worker: assigned layers {from}..={upto}, wire {dtype:?}, head {}, skill {}, task {}, o1 {}",
                    if head { "mine" } else { "coordinator's" },
                    skill.as_deref().unwrap_or("—"),
                    task.as_deref().unwrap_or("—"),
                    o1.as_ref().map(|w| w.spec.as_str()).unwrap_or("—"),
                );
                send_control(&mut stream, &mut out, &Frame::Ack { err: String::new() })?;
            }
            Msg::Control(Frame::Reset) => {
                p.reset_session();
                p.o1_begin();
                last_prefill.clear();
                hist.clear();
                send_control(&mut stream, &mut out, &Frame::Ack { err: String::new() })?;
            }
            Msg::Control(Frame::Ids { ids }) => {
                if !head {
                    return fatal(&mut stream, &mut out, "Ids without head mode".into());
                }
                hist = ids;
                send_control(&mut stream, &mut out, &Frame::Ack { err: String::new() })?;
            }
            Msg::Control(Frame::Ping) => send_control(&mut stream, &mut out, &Frame::Pong)?,
            Msg::Control(Frame::KvFetch { from, upto }) => {
                let (lo, hi) = (from as usize, upto as usize);
                let Some((sfrom, supto)) = span else {
                    return fatal(&mut stream, &mut out, "KvFetch before Assign".into());
                };
                if lo > hi || hi >= p.num_layers || lo < sfrom || hi > supto {
                    send_control(
                        &mut stream,
                        &mut out,
                        &Frame::Ack {
                            err: format!(
                                "KvFetch {lo}..={hi} outside my span {sfrom}..={supto}"
                            ),
                        },
                    )?;
                    continue;
                }
                // Probe the first layer before promising anything: the
                // refusal conditions (q8 storage, an o1 overlay, frozen
                // columns) are set the same way for every layer, so one
                // check speaks for the range — and a refusal must arrive
                // as an Ack, not as a truncated stream.
                if let Err(e) = p.kv_cache.layers[lo].export_wire(dtype == WireDtype::F16) {
                    send_control(&mut stream, &mut out, &Frame::Ack { err: e })?;
                    continue;
                }
                send_control(&mut stream, &mut out, &Frame::Ack { err: String::new() })?;
                let mut sent = 0usize;
                for li in lo..=hi {
                    let payload = match p.kv_cache.layers[li].export_wire(dtype == WireDtype::F16)
                    {
                        Ok(b) => b,
                        // Past the Ack there is no way to refuse politely:
                        // a short stream would look like a complete one.
                        Err(e) => {
                            return fatal(
                                &mut stream,
                                &mut out,
                                format!("kv export failed at layer {li}: {e}"),
                            );
                        }
                    };
                    sent += payload.len();
                    send_kv(&mut stream, &mut out, li as u32, &payload)?;
                }
                eprintln!(
                    "worker: shipped layers {lo}..={hi} of KV, {:.1} MB as {:?}",
                    sent as f64 / 1e6,
                    dtype
                );
            }
            Msg::Control(Frame::Stats) => {
                let st = crate::nodestat::read(cortiq_engine::pool::Pool::effective_threads());
                send_control(&mut stream, &mut out, &Frame::StatsReply(st))?;
            }
            // Fire-and-forget: the coordinator streams chunks and computes
            // its next one while we chew this one; `Sync` is the barrier.
            Msg::Prefill { start_pos, count } => {
                let Some((from, upto)) = span else {
                    return fatal(&mut stream, &mut out, "Prefill before Assign".into());
                };
                if floats.len() != count as usize * hs {
                    let msg = format!(
                        "Prefill: {} floats ≠ count {count} × hidden {hs}",
                        floats.len()
                    );
                    return fatal(&mut stream, &mut out, msg);
                }
                // Layer-major batched walk over our span — the same
                // GEMM/graph machinery as local prefill.
                match p.prefill_span_hidden(&floats, start_pos as usize, from, upto, mask.as_ref())
                {
                    Ok(hb) => last_prefill = hb[(count as usize - 1) * hs..].to_vec(),
                    Err(e) => return fatal(&mut stream, &mut out, e),
                }
            }
            Msg::Control(Frame::Sync) => {
                if span.is_none() {
                    return fatal(&mut stream, &mut out, "Sync before Assign".into());
                }
                // Prompt absorbed on our span — freeze the o1 skeletons
                // exactly where the local path would (post-prefill).
                p.o1_seal();
                if head {
                    // The first token of the generation is sampled here,
                    // from the last prefill position — the coordinator
                    // never sees a hidden in this mode.
                    let id = sample_here(p, &mut hist, &last_prefill);
                    send_token(&mut stream, &mut out, id)?;
                } else {
                    send_hidden(&mut stream, &mut out, dtype, &last_prefill)?;
                }
            }
            Msg::Step { pos } => {
                let Some((from, upto)) = span else {
                    return fatal(&mut stream, &mut out, "Step before Assign".into());
                };
                match p.forward_span(&floats, pos as usize, from, upto, mask.as_ref()) {
                    Ok(h) => {
                        if head {
                            let h = h.to_vec();
                            let id = sample_here(p, &mut hist, &h);
                            send_token(&mut stream, &mut out, id)?;
                        } else {
                            send_hidden(&mut stream, &mut out, dtype, &h)?;
                        }
                    }
                    Err(e) => return fatal(&mut stream, &mut out, e),
                }
            }
            // v5: the coordinator kept only the tokenizer — it ships the
            // id, we embed it ourselves. Legal only when we hold layer 0.
            Msg::StepId { pos, id, want } => {
                let Some((from, upto)) = span else {
                    return fatal(&mut stream, &mut out, "StepId before Assign".into());
                };
                if !head || from != 0 {
                    let msg = format!(
                        "StepId needs head mode and layer 0 (head {head}, span starts at {from})"
                    );
                    return fatal(&mut stream, &mut out, msg);
                }
                // One round trip, up to `run_ahead` tokens. Each step is
                // the same sequential decode the coordinator would have
                // driven — embed, layers, head, sample — so the text is
                // identical; what is gone is the handshake between them.
                let mut fed = id;
                let mut at = pos as usize;
                // `want` is the coordinator's remaining budget: running
                // past it would leave our KV holding positions it never
                // emitted, and the next turn's reuse would read a history
                // that does not exist on its side.
                let steps = run_ahead.min((want as usize).max(1));
                let mut batch: Vec<u32> = Vec::with_capacity(steps);
                let mut eos = false;
                for _ in 0..steps {
                    let emb = p.embed_id(fed);
                    let h = match p.forward_span(&emb, at, from, upto, mask.as_ref()) {
                        Ok(h) => h.to_vec(),
                        Err(e) => return fatal(&mut stream, &mut out, e),
                    };
                    let next = sample_here(p, &mut hist, &h);
                    batch.push(next);
                    if p.tokenizer.is_eos(next) {
                        eos = true;
                        break;
                    }
                    fed = next;
                    at += 1;
                }
                if run_ahead > 1 {
                    send_tokens(&mut stream, &mut out, &batch, eos)?;
                } else {
                    send_token(&mut stream, &mut out, batch[0])?;
                }
            }
            other => {
                let msg = format!("unexpected frame: {other:?}");
                return fatal(&mut stream, &mut out, msg);
            }
        }
    }
}
