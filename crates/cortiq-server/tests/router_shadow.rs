//! Shadow mode of the router API: `cortiq serve FILE --shadow-of URL` (spec
//! decision-v4 §4.15, the switch of production traffic from `cortiq-router`).
//!
//! A loopback mock of the old router records every request it gets and answers
//! with scripted bytes (pretty-printed JSON, repeated and custom headers, a
//! fixed `Date`), so that byte equality proves a pass-through, never a
//! re-serialization. Checked:
//!
//! * the client receives the old answer byte for byte (status, body, headers;
//!   no `x-request-id` added), in process and over TCP; the old router gets the
//!   request unchanged (method, path and query, body, `Authorization`);
//! * `shadow.jsonl` has one line per routed input with exactly the documented
//!   keys and no text (a SHA-256 of it only); `new_label` / `new_confident`
//!   equal what a server without the flag answers locally;
//! * `GET /v1/admin/shadow`: agreement overall, by confidence and per label
//!   equal the counts expected from the scripted old labels, and the same after
//!   a restart (replayed from the file);
//! * no oracle call, no learning, no billing in shadow mode (a server without
//!   the flag, same configuration, does call the mock oracle and bill);
//! * old-router errors (401, 422, 429 with `Retry-After`, 500, 503) pass
//!   through unchanged; no answer → 502 `UPSTREAM_UNAVAILABLE`; a body over
//!   the old router's 8 MiB → its 413 text;
//! * every router-API path is forwarded, the decisions API and this server's
//!   admin API stay local;
//! * without the flag nothing changes.
//!
//! Hermetic: the toy encoder of `cortiq-decision`, loopback only.

#[path = "support/toy_dir.rs"]
mod toy_dir;

use axum::body::Body;
use axum::http::{HeaderMap, Request};
use cortiq_decision::build::{self, TrainOptions};
use cortiq_decision::cascade::CascadeOptions;
use cortiq_decision::config::Config;
use cortiq_decision::manifest::sha256_hex;
use cortiq_decision::oracle::KeyLookup;
use cortiq_server::decisions::{DecisionServer, ServeOptions};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tower::ServiceExt;

const EPOCH: u64 = 1_790_000_000;
const SKILL: &str = "data-assistant";
const LABELS: [&str; 4] = ["Weather", "billing", "cards", "travel"];
const ADMIN: &str = "admin-token-for-router-shadow-0123456789";
const ORACLE_KEY_ENV: &str = "CMF_C3_TEST_ORACLE_KEY";
const ORACLE_MODEL: &str = "deepseek/deepseek-v4.1-flash";
/// A key of the old router (never checked here: the old router is the judge).
const OLD_KEY: &str = "cortiq_00112233445566778899aabbccddeeff00112233";
const OLD_DATE: &str = "Thu, 01 Jan 2026 00:00:00 GMT";
/// The keys of a `shadow.jsonl` line.
const LINE_KEYS: [&str; 13] = [
    "ts",
    "request_id_old",
    "text_sha256",
    "taxonomy",
    "old_label",
    "new_label",
    "agree",
    "old_confident",
    "new_confident",
    "old_latency_ms",
    "new_latency_ms",
    "old_status",
    "new_error",
];

// ------------------------------------------------------------------ toy file

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn pool(label: &str) -> &'static [&'static str] {
    match label {
        "Weather" => &[
            "rain",
            "snow",
            "forecast",
            "sunny",
            "wind",
            "storm",
            "cloudy",
            "temperature",
        ],
        "billing" => &[
            "invoice",
            "charge",
            "refund",
            "payment",
            "bill",
            "fee",
            "receipt",
            "statement",
        ],
        "cards" => &[
            "card",
            "pin",
            "atm",
            "contactless",
            "debit",
            "credit",
            "freeze",
            "replace",
        ],
        "travel" => &[
            "flight", "hotel", "airport", "booking", "passport", "luggage", "visa", "train",
        ],
        _ => unreachable!("{label}"),
    }
}

const FILLER: [&str; 8] = [
    "please", "help", "my", "the", "today", "need", "about", "with",
];

fn synth(per_label: usize, seed: u64, tag: &str) -> Vec<(String, String)> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    for i in 0..per_label {
        for label in LABELS {
            let p = pool(label);
            let mut words = Vec::new();
            for _ in 0..2 + rng.below(3) {
                words.push(p[rng.below(p.len())]);
            }
            for _ in 0..1 + rng.below(2) {
                words.push(FILLER[rng.below(FILLER.len())]);
            }
            let r = rng.below(words.len());
            words.swap(0, r);
            out.push((format!("{} {tag}{i}", words.join(" ")), label.to_string()));
        }
    }
    out
}

fn write_jsonl(dir: &Path, name: &str, rows: &[(String, String)]) -> PathBuf {
    let p = dir.join(name);
    let body: String = rows
        .iter()
        .map(|(t, l)| json!({"text": t, "label": l}).to_string() + "\n")
        .collect();
    std::fs::write(&p, body).unwrap();
    p
}

struct Toy {
    path: PathBuf,
    dev: Vec<(String, String)>,
}

fn toy() -> &'static Toy {
    static TOY: OnceLock<Toy> = OnceLock::new();
    TOY.get_or_init(|| {
        let dir = toy_dir::toy_dir("toy");
        let d = dir.as_path();
        let enc = d.join("enc.cmf");
        let encoder_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../cortiq-decision/tests/fixtures/toy/encoder");
        build::init_encoder(&encoder_dir, &enc, Some(EPOCH)).expect("init toy encoder");
        let train = synth(30, 11, "t");
        let cal = synth(80, 12, "c");
        let dev = synth(12, 13, "d");
        let mut crit = Map::new();
        for l in LABELS {
            crit.insert(l.to_string(), json!(format!("The message is about {l}.")));
        }
        let q = json!({"instructions": "Which topic is the message about?", "criteria": crit});
        let qp = d.join("q.json");
        std::fs::write(&qp, q.to_string()).unwrap();
        let mut o = TrainOptions::new(SKILL, vec![write_jsonl(d, "train.jsonl", &train)]);
        o.calibration = Some(write_jsonl(d, "cal.jsonl", &cal));
        o.dev = Some(write_jsonl(d, "dev.jsonl", &dev));
        o.question = Some(qp);
        o.threads = 2;
        o.created_unix = Some(EPOCH);
        let path = d.join("toy.cmf");
        build::train(&enc, &o, &path).expect("train the toy skill");
        Toy { path, dev }
    })
}

/// Texts the gate rejects on the toy model (none of the toy's topics).
const OFF_TOPIC: [&str; 4] = [
    "cruise ship cabin deck please",
    "zzz qqq xyzzy plugh foobar quux wibble",
    "quantum chromodynamics lecture notes",
    "violin sonata rehearsal schedule",
];

// ------------------------------------------------------------------ mock old router

/// One request the mock old router received.
#[derive(Clone, Debug)]
struct Recorded {
    method: String,
    target: String,
    /// Lowercase names, in order.
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Recorded {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
}

/// A scripted answer: status, headers (without the framing), body.
#[derive(Clone, Debug)]
struct Canned {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

type Handler = Arc<dyn Fn(&Recorded, u64) -> Canned + Send + Sync>;

/// The old router on 127.0.0.1: one connection per request (`Connection:
/// close`), every request recorded.
struct MockOld {
    addr: SocketAddr,
    seen: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl MockOld {
    fn start(handler: Handler) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen2, stop2) = (seen.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            let mut n = 0u64;
            for conn in listener.incoming() {
                if stop2.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut s) = conn else { continue };
                let Some(r) = read_request(&mut s) else {
                    continue;
                };
                n += 1;
                seen2.lock().unwrap().push(r.clone());
                let c = handler(&r, n);
                let mut head = format!("HTTP/1.1 {} Mock\r\n", c.status);
                for (k, v) in &c.headers {
                    head.push_str(&format!("{k}: {v}\r\n"));
                }
                head.push_str(&format!(
                    "Content-Length: {}\r\nConnection: close\r\nKeep-Alive: timeout=5\r\n\r\n",
                    c.body.len()
                ));
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(&c.body);
                let _ = s.flush();
            }
        });
        Self {
            addr,
            seen,
            stop,
            thread: Some(thread),
        }
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn seen(&self) -> Vec<Recorded> {
        self.seen.lock().unwrap().clone()
    }
}

impl Drop for MockOld {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn read_request(s: &mut TcpStream) -> Option<Recorded> {
    s.set_read_timeout(Some(Duration::from_secs(10))).ok()?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 65536];
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        let n = s.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let mut first = lines.next()?.split(' ');
    let method = first.next()?.to_string();
    let target = first.next()?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let len: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = s.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Some(Recorded {
        method,
        target,
        headers,
        body,
    })
}

/// The headers of every scripted answer: a JSON type, custom and repeated
/// headers, a fixed `Date`, a `Server`.
fn old_headers() -> Vec<(String, String)> {
    [
        ("Content-Type", "application/json"),
        ("X-Old-Router", "cortiq-router-rs/0.4.1"),
        ("X-Multi", "first"),
        ("X-Multi", "second"),
        ("Date", OLD_DATE),
        ("Server", "old-nginx"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// A router `RouteResponse` (router `api.rs:95-201`) in the old router's
/// voice, with the label and confidence the script gives the text.
fn old_route_result(id: &str, label: &str, confident: bool) -> Value {
    json!({
        "schema_version": "1.1",
        "request_id": id,
        "decision": {
            "task_id": LABELS.iter().position(|l| *l == label).map_or(-1, |i| i as i64),
            "task_label": label, "taxonomy_id": SKILL, "confidence": 0.91, "confident": confident,
            "raw_confidence": 0.5, "margin": 0.2, "is_novel": false, "novelty_score": 0.3,
            "complexity": {"score": 0.4, "tier": "medium",
                           "factors": {"base": 0.4, "ambiguity": 0.1, "novelty": 0.3, "margin": 0.2, "length": 0.1}},
            "source": "router", "flags": []
        },
        "scores": [],
        "usage": {"billable_decisions": 1, "oracle_calls": 0},
        "meta": {"model_version": "old", "taxonomy_version": "data-assistant@7", "latency_ms": 1.5,
                 "embedding_latency_ms": 3.0, "served_by": "cortiq-router-rs/0.4.1"}
    })
}

/// Pretty-printed with a trailing newline: bytes this server never writes.
fn pretty(v: &Value) -> Vec<u8> {
    let mut b = serde_json::to_vec_pretty(v).unwrap();
    b.push(b'\n');
    b
}

/// The old router of the tests: `/v1/route` and `/v1/route:batch` answered
/// from `script` (text → (label, confident); unknown texts `travel`), any
/// other path echoed as `{"old": true, "method", "target"}`.
fn old_router(script: HashMap<String, (String, bool)>) -> Handler {
    Arc::new(move |r: &Recorded, n: u64| {
        let answer = |text: &str, i: usize| {
            let (label, conf) = script
                .get(text)
                .cloned()
                .unwrap_or_else(|| ("travel".into(), true));
            old_route_result(&format!("req_old_{n:04}_{i}"), &label, conf)
        };
        let body = match (r.method.as_str(), r.target.as_str()) {
            ("POST", "/v1/route") => {
                let v = r.json();
                answer(v["input"]["text"].as_str().unwrap_or_default(), 0)
            }
            ("POST", "/v1/route:batch") => {
                let v = r.json();
                let results: Vec<Value> = v["inputs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, x)| answer(x["text"].as_str().unwrap_or_default(), i))
                    .collect();
                json!({"schema_version": "1.1", "results": results})
            }
            _ => json!({"old": true, "method": r.method, "target": r.target}),
        };
        Canned {
            status: 200,
            headers: old_headers(),
            body: pretty(&body),
        }
    })
}

// ------------------------------------------------------------------ mock oracle

/// Answers every chat completion with `{"task": <label>}` and counts calls.
struct MockOracle {
    addr: SocketAddr,
    hits: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl MockOracle {
    fn answering(label: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (h2, s2) = (hits.clone(), stop.clone());
        let content = json!({"task": label}).to_string();
        let thread = std::thread::spawn(move || {
            for conn in listener.incoming() {
                if s2.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut s) = conn else { continue };
                if read_request(&mut s).is_none() {
                    continue;
                }
                h2.fetch_add(1, Ordering::SeqCst);
                let body = serde_json::to_vec(&json!({
                    "id": "gen-mock", "object": "chat.completion", "model": ORACLE_MODEL,
                    "provider": "Mock",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                              "cost": 1e-5},
                }))
                .unwrap();
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(&body);
                let _ = s.flush();
            }
        });
        Self {
            addr,
            hits,
            stop,
            thread: Some(thread),
        }
    }

    fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

impl Drop for MockOracle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn oracle_config(oracle: &MockOracle) -> Config {
    let mut c = Config::default();
    c.learning.synchronous = true;
    c.oracle.enabled = true;
    c.oracle.base_url = format!("http://{}", oracle.addr);
    c.oracle.api_key_env = ORACLE_KEY_ENV.to_string();
    c.oracle.deadline_s = 5.0;
    c
}

fn oracle_key() -> KeyLookup {
    Arc::new(|name: &str| (name == ORACLE_KEY_ENV).then(|| "sk-test-not-a-real-key".to_string()))
}

fn no_key() -> KeyLookup {
    Arc::new(|_: &str| None)
}

// ------------------------------------------------------------------ server

fn options(cfg: Config, key: KeyLookup, state: &Path, shadow_of: Option<String>) -> ServeOptions {
    let mut o = ServeOptions::new(&toy().path, cfg);
    o.state_dir = Some(state.to_path_buf());
    o.addr = "127.0.0.1:0".parse().unwrap();
    o.cascade = CascadeOptions {
        key,
        threads: 2,
        created_unix: Some(EPOCH),
    };
    o.admin_token = Some(ADMIN.to_string());
    o.shadow_of = shadow_of;
    o.shadow_timeout = Duration::from_secs(10);
    o
}

struct Srv {
    server: Option<DecisionServer>,
    app: axum::Router,
    state: PathBuf,
    _dir: Option<tempfile::TempDir>,
}

impl Srv {
    fn open(cfg: Config, key: KeyLookup, shadow_of: Option<String>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        let mut s = Self::open_at(cfg, key, &state, shadow_of);
        s._dir = Some(dir);
        s
    }

    fn open_at(cfg: Config, key: KeyLookup, state: &Path, shadow_of: Option<String>) -> Self {
        let server = DecisionServer::open(&options(cfg, key, state, shadow_of))
            .expect("open the decision server");
        let app = server.router();
        Self {
            server: Some(server),
            app,
            state: state.to_path_buf(),
            _dir: None,
        }
    }

    async fn call(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: Option<Vec<u8>>,
    ) -> Resp {
        let mut b = Request::builder().method(method).uri(path);
        for (k, v) in headers {
            b = b.header(*k, *v);
        }
        let req = b.body(body.map_or_else(Body::empty, Body::from)).unwrap();
        let resp = self.app.clone().oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec();
        Resp {
            status,
            headers,
            json: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            bytes,
        }
    }

    /// A router client's request: its key in `Authorization`, JSON.
    async fn route(&self, path: &str, body: &Value) -> Resp {
        let auth = format!("Bearer {OLD_KEY}");
        self.call(
            "POST",
            path,
            &[
                ("content-type", "application/json"),
                ("authorization", &auth),
            ],
            Some(serde_json::to_vec(body).unwrap()),
        )
        .await
    }

    async fn admin(&self, method: &str, path: &str) -> Resp {
        self.call(method, path, &[("x-admin-token", ADMIN)], None)
            .await
    }

    fn log_path(&self) -> PathBuf {
        self.state.join("shadow.jsonl")
    }

    /// Wait until the log has `n` lines (the comparison is written after the
    /// client's answer), then return them.
    async fn lines(&self, n: u64) -> Vec<Value> {
        let t0 = Instant::now();
        loop {
            let s = self.admin("GET", "/v1/admin/shadow").await;
            assert_eq!(s.status, 200, "{}", s.text());
            let have = s.json["lines"].as_u64().unwrap();
            assert!(have <= n, "{have} lines, expected {n}");
            if have == n {
                break;
            }
            assert!(
                t0.elapsed() < Duration::from_secs(30),
                "{have} of {n} lines"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        read_lines(&self.log_path())
    }

    /// Stop the server; the state directory stays (its owner is returned).
    fn close(mut self) -> Option<tempfile::TempDir> {
        if let Some(s) = self.server.take() {
            s.close().unwrap();
        }
        self._dir.take()
    }
}

impl Drop for Srv {
    fn drop(&mut self) {
        if let Some(s) = self.server.take() {
            let _ = s.close();
        }
    }
}

fn read_lines(p: &Path) -> Vec<Value> {
    std::fs::read_to_string(p)
        .unwrap_or_default()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[derive(Debug)]
struct Resp {
    status: u16,
    headers: HeaderMap,
    bytes: Vec<u8>,
    json: Value,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).to_string()
    }

    /// Status, body bytes and headers are exactly the scripted answer's, and
    /// nothing was added. The mock frames its body with `Content-Length` (and
    /// `Connection: close`, `Keep-Alive`, which are hop-by-hop); the length is
    /// written for the same bytes on the wire (checked over TCP), so here it
    /// may only be absent or the old router's value.
    fn is_passthrough_of(&self, c: &Canned) {
        assert_eq!(self.status, c.status);
        assert_eq!(self.bytes, c.body, "body is not the old router's bytes");
        let mut want: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (k, v) in &c.headers {
            want.entry(k.to_ascii_lowercase())
                .or_default()
                .push(v.clone());
        }
        let mut got: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (k, v) in &self.headers {
            got.entry(k.as_str().to_string())
                .or_default()
                .push(v.to_str().unwrap().to_string());
        }
        if let Some(len) = got.remove("content-length") {
            assert_eq!(len, [c.body.len().to_string()]);
        }
        assert_eq!(got, want, "headers are not the old router's");
        assert!(self.header("x-request-id").is_none());
    }
}

/// The scripted answer the mock gave to its `n`-th request (1-based), for a
/// route body.
fn canned_route(script: &HashMap<String, (String, bool)>, text: &str, n: u64) -> Canned {
    old_router(script.clone())(
        &Recorded {
            method: "POST".into(),
            target: "/v1/route".into(),
            headers: Vec::new(),
            body: serde_json::to_vec(&json!({"input": {"text": text}})).unwrap(),
        },
        n,
    )
}

/// What a server without the flag answers locally for a text: (task_label,
/// confident) of `/v1/route` with `allow_oracle: false` and this profile.
async fn local_answers(texts: &[&str], profile: &str) -> Vec<(String, bool)> {
    let srv = Srv::open(Config::default(), no_key(), None);
    let mut out = Vec::new();
    for t in texts {
        let r = srv
            .route(
                "/v1/route",
                &json!({"input": {"text": t}, "taxonomy_id": SKILL,
                        "options": {"allow_oracle": false, "policy_profile": profile}}),
            )
            .await;
        assert_eq!(r.status, 200, "{}", r.text());
        out.push((
            r.json["decision"]["task_label"]
                .as_str()
                .unwrap()
                .to_string(),
            r.json["decision"]["confident"].as_bool().unwrap(),
        ));
    }
    out
}

/// Every line has exactly the documented keys, and nothing in it can be a
/// text: labels, `req_…` ids, a 64-hex hash, the taxonomy, reason codes.
fn check_no_text(lines: &[Value], texts: &[&str]) {
    let raw_ok = |s: &str| {
        LABELS.contains(&s)
            || s == "__novel__"
            || s == SKILL
            || s.starts_with("req_old_")
            || (s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            || s.bytes().all(|b| b.is_ascii_uppercase() || b == b'_')
    };
    for l in lines {
        let m = l.as_object().unwrap();
        let keys: BTreeSet<&str> = m.keys().map(String::as_str).collect();
        assert_eq!(keys, LINE_KEYS.into_iter().collect::<BTreeSet<_>>(), "{l}");
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                assert!(raw_ok(s), "{k}: {s:?} could be a text");
            }
        }
        if let Some(h) = l["text_sha256"].as_str() {
            assert!(
                texts.iter().any(|t| sha256_hex(t.as_bytes()) == h),
                "{h} is not the hash of a sent text"
            );
        }
    }
}

fn file_has_no_text(p: &Path, texts: &[&str]) {
    let raw = std::fs::read_to_string(p).unwrap();
    for t in texts {
        assert!(!raw.contains(t), "the log holds a text");
    }
}

// ------------------------------------------------------------------ tests

#[tokio::test]
async fn route_answer_is_the_old_routers_bytes_and_the_line_holds_no_text() {
    let text = toy().dev[1].0.clone();
    let script: HashMap<String, (String, bool)> =
        [(text.clone(), ("billing".to_string(), false))].into();
    let old = MockOld::start(old_router(script.clone()));
    let srv = Srv::open(Config::default(), no_key(), Some(old.url()));
    let body = json!({"input": {"text": text}, "taxonomy_id": SKILL, "client_request_id": "c-1",
                      "options": {"policy_profile": "balanced", "allow_oracle": true, "top_k": 2}});
    let sent = serde_json::to_vec(&body).unwrap();
    let auth = format!("Bearer {OLD_KEY}");
    let r = srv
        .call(
            "POST",
            "/v1/route",
            &[
                ("content-type", "application/json"),
                ("authorization", &auth),
                ("x-cmf-extensions", "1"),
                ("user-agent", "router-client-test/1"),
                ("x-forwarded-for", "10.1.2.3"),
            ],
            Some(sent.clone()),
        )
        .await;
    r.is_passthrough_of(&canned_route(&script, &text, 1));
    // The old router got the request unchanged.
    let seen = old.seen();
    assert_eq!(seen.len(), 1);
    let q = &seen[0];
    assert_eq!(
        (q.method.as_str(), q.target.as_str()),
        ("POST", "/v1/route")
    );
    assert_eq!(q.body, sent);
    assert_eq!(q.header("authorization"), Some(auth.as_str()));
    assert_eq!(q.header("content-type"), Some("application/json"));
    assert_eq!(q.header("user-agent"), Some("router-client-test/1"));
    assert_eq!(q.header("accept-encoding"), Some("identity"));
    assert_eq!(q.header("x-cmf-extensions"), None);
    assert_eq!(q.header("x-forwarded-for"), None);

    let lines = srv.lines(1).await;
    check_no_text(&lines, &[&text]);
    file_has_no_text(&srv.log_path(), &[&text]);
    let l = &lines[0];
    let (new_label, new_confident) = local_answers(&[&text], "balanced").await.remove(0);
    assert_eq!(l["text_sha256"], json!(sha256_hex(text.as_bytes())));
    assert_eq!(l["request_id_old"], "req_old_0001_0");
    assert_eq!(l["taxonomy"], SKILL);
    assert_eq!(l["old_label"], "billing");
    assert_eq!(l["old_confident"], false);
    assert_eq!(l["new_label"], json!(new_label));
    assert_eq!(l["new_confident"], json!(new_confident));
    assert_eq!(l["agree"], json!(new_label == "billing"));
    assert_eq!(l["old_status"], 200);
    assert_eq!(l["new_error"], Value::Null);
    assert!(l["old_latency_ms"].as_f64().unwrap() > 0.0);
    assert!(l["new_latency_ms"].as_f64().unwrap() > 0.0);
    let ts = l["ts"].as_u64().unwrap();
    assert!(ts.abs_diff(cortiq_decision::keys::now_unix()) < 600, "{ts}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(srv.log_path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

#[tokio::test]
async fn batch_lines_and_agreement_statistics_are_correct_and_survive_a_restart() {
    // 13 inputs: 12 dev texts (3 per topic) and an off-topic one. The old
    // label is the dev label, except every third one, which is the next topic.
    let mut cases: Vec<(String, String, bool)> = Vec::new();
    for (j, (t, label)) in toy().dev.iter().take(12).enumerate() {
        let old = if j % 3 == 0 {
            let i = LABELS.iter().position(|l| l == label).unwrap();
            LABELS[(i + 1) % LABELS.len()].to_string()
        } else {
            label.clone()
        };
        cases.push((t.clone(), old, j % 2 == 0));
    }
    cases.push((OFF_TOPIC[0].to_string(), "travel".to_string(), false));
    let script: HashMap<String, (String, bool)> = cases
        .iter()
        .map(|(t, l, c)| (t.clone(), (l.clone(), *c)))
        .collect();
    let texts: Vec<&str> = cases.iter().map(|c| c.0.as_str()).collect();
    let local = local_answers(&texts, "balanced").await;

    let old = MockOld::start(old_router(script.clone()));
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let srv = Srv::open_at(Config::default(), no_key(), &state, Some(old.url()));
    // The first 5 one by one, the other 8 in one batch.
    for t in &texts[..5] {
        let r = srv.route("/v1/route", &json!({"input": {"text": t}})).await;
        assert_eq!(r.status, 200);
    }
    let batch = json!({"taxonomy_id": SKILL,
                       "inputs": texts[5..].iter().map(|t| json!({"text": t})).collect::<Vec<_>>()});
    let r = srv.route("/v1/route:batch", &batch).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json["results"].as_array().unwrap().len(), 8);
    assert_eq!(r.bytes.last(), Some(&b'\n'), "the old pretty bytes");
    let lines = srv.lines(13).await;
    check_no_text(&lines, &texts);
    file_has_no_text(&srv.log_path(), &texts);

    // Line by line, in request order (single requests first, then the batch).
    let mut by_hash: HashMap<String, &Value> = HashMap::new();
    for l in &lines {
        by_hash.insert(l["text_sha256"].as_str().unwrap().to_string(), l);
    }
    assert_eq!(by_hash.len(), 13);
    let batch_ids: BTreeSet<&str> = lines
        .iter()
        .filter_map(|l| l["request_id_old"].as_str())
        .filter(|id| id.starts_with("req_old_0006_"))
        .collect();
    assert_eq!(batch_ids.len(), 8, "one line per batch input");
    // Expected statistics from the script and the local answers.
    let mut compared = 0u64;
    let mut agree = 0u64;
    let (mut oc, mut oca, mut nc, mut nca, mut bc, mut bca) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut per: BTreeMap<String, (u64, u64, BTreeMap<String, u64>)> = BTreeMap::new();
    for ((t, old_label, old_conf), (new_label, new_conf)) in cases.iter().zip(&local) {
        let l = by_hash[&sha256_hex(t.as_bytes())];
        assert_eq!(l["old_label"], json!(old_label));
        assert_eq!(l["new_label"], json!(new_label));
        assert_eq!(l["old_confident"], json!(old_conf));
        assert_eq!(l["new_confident"], json!(new_conf));
        let a = old_label == new_label;
        assert_eq!(l["agree"], json!(a));
        compared += 1;
        agree += u64::from(a);
        if *old_conf {
            oc += 1;
            oca += u64::from(a);
        }
        if *new_conf {
            nc += 1;
            nca += u64::from(a);
        }
        if *old_conf && *new_conf {
            bc += 1;
            bca += u64::from(a);
        }
        let e = per.entry(old_label.clone()).or_default();
        e.0 += 1;
        e.1 += u64::from(a);
        *e.2.entry(new_label.clone()).or_default() += 1;
    }
    assert!(
        agree > 0 && agree < compared,
        "the script mixes agreement and disagreement: {agree}/{compared}"
    );
    let check = |s: &Value| {
        assert_eq!(s["shadow_of"], json!(old.url()));
        assert_eq!(s["log"], "shadow.jsonl");
        assert_eq!(s["lines"], 13);
        assert_eq!(s["malformed_lines"], 0);
        assert_eq!(s["compared"], compared);
        assert_eq!(s["agree"], agree);
        assert_eq!(s["agreement"], json!(agree as f64 / compared as f64));
        assert_eq!(s["old_errors"], 0);
        assert_eq!(s["old_missing"], 0);
        assert_eq!(s["new_missing"], 0);
        for (k, n, a) in [("old", oc, oca), ("new", nc, nca), ("both", bc, bca)] {
            assert_eq!(s["confident"][k]["compared"], n, "{k}");
            assert_eq!(s["confident"][k]["agree"], a, "{k}");
        }
        let labels = s["labels"].as_array().unwrap();
        assert_eq!(labels.len(), per.len());
        for e in labels {
            let (n, a, news) = &per[e["label"].as_str().unwrap()];
            assert_eq!(e["taxonomy"], SKILL);
            assert_eq!(e["lines"], *n);
            assert_eq!(e["compared"], *n);
            assert_eq!(e["agree"], *a);
            assert_eq!(e["agreement"], json!(*a as f64 / *n as f64));
            assert_eq!(e["new_labels"], json!(news));
        }
        let lat_old: f64 = lines
            .iter()
            .map(|l| l["old_latency_ms"].as_f64().unwrap())
            .sum::<f64>()
            / 13.0;
        let got = s["latency_ms"]["old_mean"].as_f64().unwrap();
        assert!((got - lat_old).abs() < 1e-3, "{got} {lat_old}");
    };
    check(&srv.admin("GET", "/v1/admin/shadow").await.json);
    // Only the admin token reads them.
    let r = srv.call("GET", "/v1/admin/shadow", &[], None).await;
    assert_eq!(r.status, 401);
    // The statistics are the file's: the same after a restart.
    srv.close();
    let srv = Srv::open_at(Config::default(), no_key(), &state, Some(old.url()));
    check(&srv.admin("GET", "/v1/admin/shadow").await.json);
}

#[tokio::test]
async fn shadow_mode_never_calls_the_oracle_learns_or_bills() {
    let oracle = MockOracle::answering("travel");
    let old = MockOld::start(old_router(HashMap::new()));
    let texts: Vec<&str> = OFF_TOPIC
        .iter()
        .copied()
        .chain(toy().dev.iter().take(4).map(|(t, _)| t.as_str()))
        .collect();
    let local = local_answers(&texts, "balanced").await;
    assert!(
        local.iter().any(|(_, c)| !c),
        "a text the gate rejects is sent"
    );

    let srv = Srv::open(oracle_config(&oracle), oracle_key(), Some(old.url()));
    for t in &texts[..4] {
        let r = srv
            .route(
                "/v1/route",
                &json!({"input": {"text": t}, "options": {"allow_oracle": true}}),
            )
            .await;
        assert_eq!(r.status, 200);
        assert!(r.header("x-request-id").is_none());
    }
    let batch = json!({"inputs": texts[4..].iter().map(|t| json!({"text": t})).collect::<Vec<_>>(),
                       "options": {"allow_oracle": true}});
    assert_eq!(srv.route("/v1/route:batch", &batch).await.status, 200);
    let lines = srv.lines(texts.len() as u64).await;
    for (t, (label, conf)) in texts.iter().zip(&local) {
        let l = lines
            .iter()
            .find(|l| l["text_sha256"] == json!(sha256_hex(t.as_bytes())))
            .unwrap();
        assert_eq!(
            l["new_label"],
            json!(label),
            "the local winner, not an oracle's"
        );
        assert_eq!(l["new_confident"], json!(conf));
    }
    // No oracle, no cache, no learning, no billing.
    assert_eq!(oracle.hits(), 0, "the oracle was called in shadow mode");
    let usage = srv.admin("GET", "/v1/admin/usage").await;
    assert_eq!(usage.status, 200);
    assert_eq!(usage.json["accounts"], json!({}), "{}", usage.text());
    let learning = srv.admin("GET", "/v1/admin/learning").await;
    assert_eq!(learning.status, 200);
    let lj = &learning.json;
    assert_eq!(lj["buffer"]["examples"], 0, "{lj}");
    assert_eq!(lj["pending_feedback"], 0);
    assert_eq!(lj["attempts"], 0);
    assert_eq!(lj["cache"]["lookups"], 0);
    assert_eq!(lj["cache"]["entries"], 0);
    let state = srv.state.clone();
    let _kept = srv.close();
    let oracle_ledger = std::fs::read_to_string(state.join("oracle.jsonl")).unwrap_or_default();
    assert!(oracle_ledger.is_empty(), "{oracle_ledger}");
    for e in std::fs::read_dir(state.join("usage")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().is_some_and(|x| x == "jsonl") {
            assert_eq!(std::fs::read_to_string(&p).unwrap(), "", "{}", p.display());
        }
    }

    // The same configuration without the flag does call the oracle, bill and
    // keep the example: the checks above are not vacuous.
    let normal = Srv::open(oracle_config(&oracle), oracle_key(), None);
    let rejected = texts.iter().zip(&local).find(|(_, (_, c))| !c).unwrap().0;
    let r = normal
        .route(
            "/v1/route",
            &json!({"input": {"text": rejected}, "options": {"allow_oracle": true}}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert_eq!(r.json["decision"]["source"], "oracle");
    assert_eq!(oracle.hits(), 1);
    let usage = normal.admin("GET", "/v1/admin/usage").await;
    assert_eq!(
        usage.json["accounts"].as_object().unwrap().len(),
        1,
        "{}",
        usage.text()
    );
    let lj = normal.admin("GET", "/v1/admin/learning").await.json;
    assert!(lj["buffer"]["examples"].as_u64().unwrap() >= 1, "{lj}");
    assert!(!normal.log_path().exists());
}

#[tokio::test]
async fn old_router_errors_pass_through_unchanged() {
    let text_of = |r: &Recorded| {
        r.json()["input"]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    };
    let envelope = |code: &str, id: &str, retriable: bool| {
        pretty(&json!({"schema_version": "1.1", "request_id": id,
                       "error": {"code": code, "message": "scripted", "retriable": retriable, "details": null}}))
    };
    let script = move |text: &str| -> Canned {
        let json_h = || vec![("Content-Type".to_string(), "application/json".to_string())];
        match text {
            "e401 invoice" => Canned {
                status: 401,
                headers: json_h(),
                body: envelope("UNAUTHORIZED", "req_old_e401", false),
            },
            "e402 invoice" => Canned {
                status: 402,
                headers: json_h(),
                body: envelope("QUOTA_EXCEEDED", "req_old_e402", false),
            },
            "e429 invoice" => {
                let mut h = json_h();
                h.push(("Retry-After".into(), "17".into()));
                Canned {
                    status: 429,
                    headers: h,
                    body: envelope("RATE_LIMITED", "req_old_e429", true),
                }
            }
            "e422 invoice" => Canned {
                status: 422,
                headers: vec![("Content-Type".into(), "text/plain; charset=utf-8".into())],
                body: b"Failed to deserialize the JSON body into the target type: input: missing field `text`".to_vec(),
            },
            "e500 invoice" => Canned {
                status: 500,
                headers: json_h(),
                body: envelope("INTERNAL", "req_old_e500", true),
            },
            _ => Canned {
                status: 503,
                headers: vec![("X-Old-Router".into(), "down".into())],
                body: Vec::new(),
            },
        }
    };
    let old = MockOld::start(Arc::new(move |r: &Recorded, _: u64| script(&text_of(r))));
    let srv = Srv::open(Config::default(), no_key(), Some(old.url()));
    let cases = [
        "e401 invoice",
        "e402 invoice",
        "e429 invoice",
        "e422 invoice",
        "e500 invoice",
        "e503 invoice",
    ];
    for t in cases {
        let r = srv.route("/v1/route", &json!({"input": {"text": t}})).await;
        r.is_passthrough_of(&script(t));
    }
    let lines = srv.lines(cases.len() as u64).await;
    check_no_text(&lines, &cases);
    for (t, want) in cases.iter().zip([401, 402, 429, 422, 500, 503]) {
        let l = lines
            .iter()
            .find(|l| l["text_sha256"] == json!(sha256_hex(t.as_bytes())))
            .unwrap();
        assert_eq!(l["old_status"], want);
        assert_eq!(l["old_label"], Value::Null);
        assert_eq!(l["agree"], Value::Null);
        assert!(l["new_label"].is_string(), "decided locally all the same");
        let id = match want {
            422 | 503 => Value::Null,
            n => json!(format!("req_old_e{n}")),
        };
        assert_eq!(l["request_id_old"], id);
    }
    let s = srv.admin("GET", "/v1/admin/shadow").await.json;
    assert_eq!(
        (s["lines"].as_u64(), s["compared"].as_u64()),
        (Some(6), Some(0))
    );
    assert_eq!(s["old_errors"], 6);
    assert_eq!(s["agreement"], Value::Null);

    // A body over the old router's 8 MiB: its own 413, nothing forwarded.
    let before = old.seen().len();
    let big = vec![b' '; 8 * 1024 * 1024 + 1];
    let len = big.len().to_string();
    let r = srv
        .call(
            "POST",
            "/v1/route",
            &[
                ("content-type", "application/json"),
                ("content-length", &len),
            ],
            Some(big),
        )
        .await;
    assert_eq!(r.status, 413);
    assert_eq!(r.text(), "length limit exceeded");
    assert_eq!(r.header("content-type"), Some("text/plain; charset=utf-8"));
    assert_eq!(old.seen().len(), before);

    // No answer at all: 502 in the router's envelope, and the comparison
    // still has the local side.
    let closed = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    let dead = Srv::open(
        Config::default(),
        no_key(),
        Some(format!("http://{closed}")),
    );
    let r = dead
        .route("/v1/route", &json!({"input": {"text": toy().dev[0].0}}))
        .await;
    assert_eq!(r.status, 502);
    assert_eq!(r.json["schema_version"], "1.1");
    assert_eq!(r.json["error"]["code"], "UPSTREAM_UNAVAILABLE");
    assert_eq!(r.json["error"]["retriable"], true);
    assert_eq!(r.json["request_id"].as_str(), r.header("x-request-id"));
    let l = &dead.lines(1).await[0];
    assert_eq!(l["old_status"], Value::Null);
    assert_eq!(l["request_id_old"], Value::Null);
    assert!(l["new_label"].is_string());
    assert!(l["old_latency_ms"].as_f64().is_some());
}

#[tokio::test]
async fn every_router_path_is_forwarded_and_the_servers_own_stay_local() {
    let old = MockOld::start(old_router(HashMap::new()));
    let srv = Srv::open(Config::default(), no_key(), Some(old.url()));
    let auth = format!("Bearer {OLD_KEY}");
    let auth = auth.as_str();
    let echo = |method: &str, target: &str| Canned {
        status: 200,
        headers: old_headers(),
        body: pretty(&json!({"old": true, "method": method, "target": target})),
    };
    let fb = serde_json::to_vec(&json!({"request_id": "req_old_1", "correct_task_label": "cards"}))
        .unwrap();
    // (method, path and query, client headers, body)
    type Case<'a> = (&'a str, &'a str, Vec<(&'a str, &'a str)>, Option<Vec<u8>>);
    let forwarded: [Case; 13] = [
        (
            "POST",
            "/v1/feedback",
            vec![
                ("authorization", auth),
                ("content-type", "application/json"),
            ],
            Some(fb.clone()),
        ),
        ("GET", "/v1/taxonomies", vec![("authorization", auth)], None),
        (
            "GET",
            "/v1/taxonomies/data-assistant",
            vec![("x-api-key", OLD_KEY)],
            None,
        ),
        (
            "GET",
            "/v1/taxonomies/a/b",
            vec![("authorization", auth)],
            None,
        ),
        ("GET", "/v1/usage", vec![("authorization", auth)], None),
        (
            "GET",
            "/v1/escalations?limit=5",
            vec![("authorization", auth)],
            None,
        ),
        ("GET", "/v1/healthz", vec![], None),
        ("GET", "/v1/readyz", vec![], None),
        ("GET", "/metrics", vec![], None),
        (
            "POST",
            "/v1/admin/keys",
            vec![
                ("x-admin-token", "old-admin"),
                ("content-type", "application/json"),
            ],
            Some(br#"{"plan":"developer"}"#.to_vec()),
        ),
        (
            "GET",
            "/v1/admin/keys",
            vec![("x-admin-token", "old-admin")],
            None,
        ),
        (
            "DELETE",
            "/v1/admin/keys/acct-1",
            vec![("x-admin-token", "old-admin")],
            None,
        ),
        // A method the old router does not route: its answer too.
        ("PUT", "/v1/route", vec![("authorization", auth)], None),
    ];
    for (i, (method, target, headers, body)) in forwarded.iter().enumerate() {
        let r = srv.call(method, target, headers, body.clone()).await;
        r.is_passthrough_of(&echo(method, target));
        let q = &old.seen()[i];
        assert_eq!((q.method.as_str(), q.target.as_str()), (*method, *target));
        assert_eq!(q.body, body.clone().unwrap_or_default());
        for (k, v) in headers {
            assert_eq!(q.header(k), Some(*v), "{target} {k}");
        }
    }
    let n = old.seen().len();
    assert_eq!(n, forwarded.len());
    // Local: the decisions API, this server's admin API.
    for (method, path, headers) in [
        ("GET", "/v1/models", vec![]),
        ("GET", "/healthz", vec![]),
        ("GET", "/v1/admin/shadow", vec![("x-admin-token", ADMIN)]),
        ("GET", "/v1/admin/learning", vec![("x-admin-token", ADMIN)]),
        (
            "DELETE",
            "/v1/admin/keys/hash/0123456789ab",
            vec![("x-admin-token", ADMIN)],
        ),
    ] {
        let r = srv.call(method, path, &headers, None).await;
        assert!(r.header("x-request-id").is_some(), "{path} answered here");
        assert!(r.header("x-old-router").is_none(), "{path}");
    }
    let r = srv
        .call(
            "POST",
            "/v1/decisions",
            &[("content-type", "application/json")],
            Some(
                serde_json::to_vec(&json!({"model": "cortiq/decision", "state": toy().dev[0].0,
                    "questions": {"task": {"type": "choice", "instructions": "Which topic?",
                        "criteria": {"Weather": "w", "billing": "b", "cards": "c", "travel": "t"}}}}))
                .unwrap(),
            ),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert!(r.json["answers"]["task"].is_object());
    assert_eq!(old.seen().len(), n, "nothing else reached the old router");
    // Only routing writes comparisons.
    let s = srv.admin("GET", "/v1/admin/shadow").await.json;
    assert_eq!(s["lines"], 0);
}

#[tokio::test]
async fn without_the_flag_nothing_changes() {
    let old = MockOld::start(old_router(HashMap::new()));
    let srv = Srv::open(Config::default(), no_key(), None);
    let r = srv
        .route("/v1/route", &json!({"input": {"text": toy().dev[0].0}}))
        .await;
    assert_eq!(r.status, 200, "{}", r.text());
    assert!(
        r.json["meta"]["served_by"]
            .as_str()
            .unwrap()
            .starts_with("cortiq/")
    );
    assert!(r.json["request_id"].as_str().unwrap().starts_with("req_"));
    assert_eq!(r.json["request_id"].as_str(), r.header("x-request-id"));
    let r = srv
        .route(
            "/v1/route:batch",
            &json!({"inputs": [{"text": toy().dev[1].0}]}),
        )
        .await;
    assert_eq!(r.status, 200);
    // `/v1/admin/shadow` is not a route: the decisions surface's JSON 404 for an
    // unknown path (this server's admin paths are not the router's, §4.8).
    let r = srv.admin("GET", "/v1/admin/shadow").await;
    assert_eq!(r.status, 404);
    assert_eq!(r.json["error"]["code"], 404, "{}", r.text());
    assert_eq!(r.json["error"]["metadata"]["reason"], "INVALID_REQUEST");
    assert_eq!(
        r.json["error"]["metadata"]["request_id"].as_str(),
        r.header("x-request-id")
    );
    let r = srv.call("GET", "/v1/healthz", &[], None).await;
    assert_eq!(r.json, json!({"status": "ok"}));
    assert!(old.seen().is_empty());
    assert!(!srv.log_path().exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn over_tcp_the_old_headers_stay_and_a_stopping_server_writes_its_lines() {
    let text = toy().dev[2].0.clone();
    let script: HashMap<String, (String, bool)> =
        [(text.clone(), ("cards".to_string(), true))].into();
    let old = MockOld::start(old_router(script.clone()));
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let server = DecisionServer::open(&options(
        Config::default(),
        no_key(),
        &state,
        Some(old.url()),
    ))
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let run = tokio::spawn(server.run(listener, async {
        let _ = stop_rx.await;
    }));
    let body = serde_json::to_vec(&json!({"input": {"text": text}})).unwrap();
    let raw = tokio::task::spawn_blocking(move || {
        let mut s = TcpStream::connect(addr).unwrap();
        let head = format!(
            "POST /v1/route HTTP/1.1\r\nHost: shadow\r\nContent-Type: application/json\r\nAuthorization: Bearer {OLD_KEY}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        s.write_all(head.as_bytes()).unwrap();
        s.write_all(&body).unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).unwrap();
        out
    })
    .await
    .unwrap();
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
    let body = &raw[split + 4..];
    let want = canned_route(&script, &text, 1);
    assert!(head.starts_with("http/1.1 200"), "{head}");
    assert_eq!(body, want.body.as_slice());
    let header_lines: Vec<&str> = head.lines().skip(1).collect();
    let dates: Vec<&&str> = header_lines
        .iter()
        .filter(|l| l.starts_with("date:"))
        .collect();
    assert_eq!(
        dates,
        [&format!("date: {}", OLD_DATE.to_ascii_lowercase()).as_str()],
        "{head}"
    );
    assert!(header_lines.contains(&"x-old-router: cortiq-router-rs/0.4.1"));
    assert!(header_lines.contains(&"server: old-nginx"));
    assert!(header_lines.contains(&"x-multi: first") && header_lines.contains(&"x-multi: second"));
    assert!(header_lines.contains(&format!("content-length: {}", want.body.len()).as_str()));
    assert!(!head.contains("x-request-id"), "{head}");
    assert!(!head.contains("keep-alive: timeout"), "{head}");
    let _ = stop_tx.send(());
    run.await.unwrap().unwrap();
    // The server stopped after writing the comparison.
    let lines = read_lines(&state.join("shadow.jsonl"));
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["old_label"], "cards");
    assert_eq!(lines[0]["text_sha256"], json!(sha256_hex(text.as_bytes())));
}

#[test]
fn bad_shadow_urls_are_refused_before_anything_is_opened() {
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    for (url, why) in [
        ("http://router.example.com", "loopback"),
        ("https://user:secretpw@router.example.com", "credentials"),
        ("router.example.com", "not a URL"),
    ] {
        let e = DecisionServer::open(&options(
            Config::default(),
            no_key(),
            &state,
            Some(url.to_string()),
        ))
        .unwrap_err()
        .to_string();
        assert!(e.contains(why), "{url}: {e}");
        assert!(!e.contains("secretpw"), "{e}");
        assert!(!state.exists(), "{url}: the state directory was created");
    }
}
