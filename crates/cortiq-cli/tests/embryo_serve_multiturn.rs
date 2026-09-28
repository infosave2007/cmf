//! Multi-turn dialogue through `cortiq serve` (OpenAI chat endpoint) on a
//! natively bounded Embryo-O1 file: 20 growing turns, then a divergent
//! history, then one more extension.  Per turn, from the server's own
//! telemetry: the prefill covered ONLY the new turn (the `kv-reuse: R of N
//! prompt positions already cached` line `CMF_PREFILL_PROF=1` prints —
//! R equals the whole previously consumed sequence: prompt + answer minus
//! its last emitted token, which the model samples but never forwards), the
//! per-slot state bytes (`/v1/cortiq/status`) are constant, a divergent
//! history triggers exactly one full re-prefill (no reuse line), and the
//! next extension reuses again.
//!
//! Two modes:
//! * `CMF_SERVE_TEST_MODEL=<bounded .cmf with a tokenizer>` — the full
//!   contract above (the echoed assistant text must re-tokenize to the
//!   generated ids, which a real BPE does; run it on the server, CPU:
//!   `CMF_SERVE_TEST_MODEL=/data/cmf-embryo-o1/s6b/recovery-fixed/recovery-1500-f32.cmf
//!    cargo test -p cortiq-cli --release --test embryo_serve_multiturn -- --nocapture`).
//! * unset — the synthetic vmf+bounded genome (byte-level fallback
//!   tokenizer, lossy on random bytes): mechanics only — the server comes
//!   up, answers 22 turns, state bytes stay constant.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;

struct Server {
    child: Child,
    port: u16,
    stderr: Arc<Mutex<Vec<String>>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn start(model: &str) -> Server {
    let port = free_port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_cortiq"))
        .args(["serve", model, "--port", &port.to_string(), "--host", "127.0.0.1"])
        .env("CMF_GPU", "0")
        .env("CMF_PREFILL_PROF", "1")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cortiq serve");
    let stderr = Arc::new(Mutex::new(Vec::new()));
    let sink = stderr.clone();
    let pipe = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(pipe).lines().map_while(Result::ok) {
            sink.lock().unwrap().push(line);
        }
    });
    let t0 = Instant::now();
    loop {
        if let Ok(r) = ureq::get(&format!("http://127.0.0.1:{port}/healthz")).call() {
            if r.status() == 200 {
                break;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(180), "server did not come up");
        std::thread::sleep(Duration::from_millis(200));
    }
    Server {
        child,
        port,
        stderr,
    }
}

fn status_bytes(port: u16) -> (u64, u64) {
    let v: serde_json::Value = ureq::get(&format!("http://127.0.0.1:{port}/v1/cortiq/status"))
        .call()
        .expect("status")
        .into_json()
        .expect("status json");
    let kv = v["slot_attention_kv_bytes"][0].as_u64().unwrap_or(0);
    let rec = v["slot_recurrent_state_bytes"][0].as_u64().unwrap_or(0);
    (kv, rec)
}

/// (assistant content, prompt_tokens, completion_tokens)
fn chat(port: u16, messages: &[(String, String)], max_tokens: u32) -> (String, u64, u64) {
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(r, c)| serde_json::json!({"role": r, "content": c}))
        .collect();
    let body = serde_json::json!({
        "model": "cortiq",
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false,
    });
    let v: serde_json::Value = ureq::post(&format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .timeout(Duration::from_secs(300))
        .send_json(body)
        .expect("chat request")
        .into_json()
        .expect("chat json");
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    (
        content,
        v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
    )
}

/// `(R, N)` of every `kv-reuse: R of N prompt positions already cached`
/// line the server printed after index `from`; returns the new cursor.
fn reuse_lines(server: &Server, from: usize) -> (Vec<(u64, u64)>, usize) {
    // The line is written before generation returns, but the pipe reader
    // is a separate thread: give it a moment.
    std::thread::sleep(Duration::from_millis(150));
    let lines = server.stderr.lock().unwrap();
    let mut out = Vec::new();
    for l in &lines[from.min(lines.len())..] {
        if let Some(rest) = l.trim().strip_prefix("kv-reuse: ") {
            let mut it = rest.split_whitespace();
            let r: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let n: u64 = it.nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            out.push((r, n));
        }
    }
    (out, lines.len())
}

#[test]
fn twenty_turns_reuse_only_the_new_turn_and_a_divergent_history_reprefills_once() {
    let real = std::env::var("CMF_SERVE_TEST_MODEL").ok();
    let model = match &real {
        Some(m) => m.clone(),
        None => embryo_synth::synth_genome_path(&embryo_synth::SynthGeom::bounded_small())
            .to_string_lossy()
            .into_owned(),
    };
    let full = real.is_some();
    let server = start(&model);
    let port = server.port;
    let max_tokens = 6u32;
    let mut messages: Vec<(String, String)> = Vec::new();
    let mut cursor = 0usize;
    let mut prev_consumed = 0u64;
    let mut state_ref: Option<(u64, u64)> = None;
    for turn in 1..=20u32 {
        messages.push((
            "user".into(),
            format!("Turn {turn}: name one thing about the number {}.", turn * 7),
        ));
        let (content, n, c) = chat(port, &messages, max_tokens);
        let (lines, next) = reuse_lines(&server, cursor);
        cursor = next;
        let st = status_bytes(port);
        eprintln!(
            "turn {turn}: prompt_tokens {n} completion {c} reuse {:?} state {:?} answer {:?}",
            lines,
            st,
            content.chars().take(40).collect::<String>()
        );
        assert!(n > 0, "usage.prompt_tokens missing");
        if full {
            if turn == 1 {
                assert!(lines.is_empty(), "turn 1 is a fresh sequence: {lines:?}");
            } else {
                assert_eq!(lines.len(), 1, "turn {turn}: exactly one reuse decision: {lines:?}");
                let (r, nn) = lines[0];
                assert_eq!(nn, n, "turn {turn}: reuse line prompt length");
                assert_eq!(
                    r, prev_consumed,
                    "turn {turn}: prefill must cover only the new turn (cached = previous prompt + answer − its last emitted token)"
                );
                assert!(n > r, "turn {turn}: nothing new to prefill?");
            }
        }
        // State bytes: rings + recurrent record, whatever the turn.
        match state_ref {
            None => state_ref = Some(st),
            Some(s0) => assert_eq!(st, s0, "turn {turn}: state bytes changed"),
        }
        // The last emitted token is sampled but never forwarded: the cache
        // record ends one position before the answer text does.
        prev_consumed = n + c - 1;
        messages.push(("assistant".into(), content));
    }
    // Divergent history: rewrite turn 3's user message → exactly one full
    // re-prefill (no reuse line), then an extension reuses again.
    messages[4].1 = "Turn 3 (edited): a different question entirely.".into();
    messages.push(("user".into(), "Turn 21: continue.".into()));
    let (content, n21, c21) = chat(port, &messages, max_tokens);
    let (lines, next) = reuse_lines(&server, cursor);
    cursor = next;
    eprintln!("turn 21 (divergent): prompt {n21} completion {c21} reuse {lines:?}");
    if full {
        assert!(lines.is_empty(), "divergent history must re-prefill in full: {lines:?}");
    }
    assert_eq!(status_bytes(port), state_ref.unwrap(), "state bytes after re-prefill");
    messages.push(("assistant".into(), content));
    messages.push(("user".into(), "Turn 22: and again.".into()));
    let (_, n22, _) = chat(port, &messages, max_tokens);
    let (lines, _) = reuse_lines(&server, cursor);
    eprintln!("turn 22 (extension): prompt {n22} reuse {lines:?}");
    if full {
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].0, n21 + c21 - 1, "turn 22 reuses the re-prefilled sequence + its answer");
    }
    assert_eq!(status_bytes(port), state_ref.unwrap());
    eprintln!(
        "multi-turn serve: 22 turns, state bytes constant at {:?}, mode={}",
        state_ref.unwrap(),
        if full { "full contract" } else { "mechanics (synthetic genome)" }
    );
}
