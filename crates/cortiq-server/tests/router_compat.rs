//! Router API compatibility (spec decision-v4 §4.15, §4.7b): every endpoint of
//! the production `cortiq-router` (schema 1.1) answers with exactly the
//! router's keys and JSON types, for success and error cases.
//!
//! The golden shapes below are transcribed from the router's `src/api.rs`
//! (request structs 30-89, response structs 95-201, error envelope 207-284,
//! admin 374-543, handlers 1170-1527) and `src/state.rs` (`AuditRecord`
//! 118-132). A shape is a JSON value: a string names the JSON type(s) of a
//! leaf (`"string"`, `"number"`, `"uint"`, `"int"`, `"bool"`, `"null"`,
//! alternatives with `|`), an object lists every key (a `?` prefix marks a key
//! the router omits when `None`, `skip_serializing_if`), an array holds the
//! shape of every element. [`conforms`] fails on a missing key, an extra key
//! or a wrong type.
//!
//! Where the router's axum extractors reject a body (415, 400, 422, 413, the
//! query string), the router answers axum's `text/plain` messages; those are
//! compared as text, as are the empty 404/405 of unrouted requests.
//!
//! Hermetic: the toy encoder of `cortiq-decision`, one skill named like the
//! production taxonomy (`data-assistant`), requests driven in-process; one test
//! runs a loopback mock of the oracle (no network beyond 127.0.0.1).

#[path = "support/toy_dir.rs"]
mod toy_dir;

use axum::body::Body;
use axum::http::{HeaderMap, Request};
use cortiq_decision::build::{self, TrainOptions};
use cortiq_decision::cascade::CascadeOptions;
use cortiq_decision::config::Config;
use cortiq_decision::container::{DecisionModel, Verify};
use cortiq_decision::keys::now_unix;
use cortiq_decision::oracle::KeyLookup;
use cortiq_decision::service::{Action, DecisionService, LoadedModel, ModelHandle, Principal};
use cortiq_decision::signal::SignalEncoder;
use cortiq_server::decisions::{DecisionServer, ServeOptions};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tower::ServiceExt;

const EPOCH: u64 = 1_790_000_000;
const SKILL: &str = "data-assistant";
const LABELS: [&str; 4] = ["Weather", "billing", "cards", "travel"];
const ADMIN: &str = "admin-token-for-router-compat-0123456789";
const ORACLE_KEY_ENV: &str = "CMF_C1_TEST_ORACLE_KEY";
const ORACLE_MODEL: &str = "deepseek/deepseek-v4.1-flash";

// ------------------------------------------------------------------ golden shapes (router api.rs)

/// `RouteResponse` (router `api.rs:95-201`).
fn route_shape() -> Value {
    json!({
        "schema_version": "string",
        "request_id": "string",
        "?client_request_id": "string",
        "decision": {
            "task_id": "int",
            "task_label": "string",
            "taxonomy_id": "string",
            "confidence": "number",
            "confident": "bool",
            "raw_confidence": "number",
            "margin": "number",
            "is_novel": "bool",
            "novelty_score": "number",
            "complexity": {
                "score": "number",
                "tier": "string",
                "factors": {
                    "base": "number",
                    "?difficulty": "number",
                    "ambiguity": "number",
                    "novelty": "number",
                    "margin": "number",
                    "length": "number",
                },
            },
            "source": "string",
            "flags": ["string"],
        },
        "scores": [{
            "task_id": "uint",
            "task_label": "string",
            "probability": "number",
            "score": "number",
            "reconstruction_error": "number",
        }],
        "?routing": {"target": "string", "reason": "string"},
        "?oracle": {
            "consulted": "bool",
            "model": "string|null",
            "agreement_with_router": "bool|null",
            "latency_ms": "number|null",
        },
        "?explanation": {"top1_vs_top2": "string", "decision_path": "string"},
        "usage": {"billable_decisions": "uint", "oracle_calls": "uint"},
        "meta": {
            "model_version": "string",
            "taxonomy_version": "string",
            "latency_ms": "number",
            "embedding_latency_ms": "number",
            "served_by": "string",
        },
    })
}

/// `ApiError::into_response` (router `api.rs:270-284`): `details` is `null`
/// (`Value::Null`) except for `taxonomy_not_found`.
fn error_shape() -> Value {
    json!({
        "schema_version": "string",
        "request_id": "string",
        "error": {"code": "string", "message": "string", "retriable": "bool", "details": "null"},
    })
}

fn taxonomy_error_shape() -> Value {
    json!({
        "schema_version": "string",
        "request_id": "string",
        "error": {"code": "string", "message": "string", "retriable": "bool",
                  "details": {"taxonomy_id": "string"}},
    })
}

/// `BatchResponse` (router `api.rs:1193-1197`).
fn batch_shape() -> Value {
    json!({"schema_version": "string", "results": [route_shape()]})
}

/// `FeedbackResponse` (router `api.rs:1254-1259`).
fn feedback_shape() -> Value {
    json!({"schema_version": "string", "accepted": "bool", "message": "string"})
}

/// `TaxonomySummary` (router `api.rs:1301-1307`).
fn taxonomy_shape() -> Value {
    json!({"taxonomy_id": "string", "taxonomy_version": "string", "model_version": "string",
           "labels": ["string"]})
}

fn taxonomies_shape() -> Value {
    json!({"schema_version": "string", "taxonomies": [taxonomy_shape()]})
}

/// `usage_handler` (router `api.rs:1337-1363`).
fn usage_shape() -> Value {
    json!({
        "schema_version": "string",
        "account": {"id": "string", "billable_decisions": "uint", "oracle_calls": "uint",
                    "decision_quota": "uint", "rate_per_min": "uint"},
        "usage": {"billable_decisions": "uint", "oracle_calls": "uint", "escalations": "uint",
                  "escalation_rate": "number", "cache_hits": "uint", "novelty_hits": "uint",
                  "refits": "uint", "promotions": "uint"},
    })
}

/// `escalations_handler` (router `api.rs:1371-1395`) with `AuditRecord`
/// (router `state.rs:118-132`).
fn escalations_shape() -> Value {
    json!({
        "schema_version": "string",
        "summary": {"total": "uint", "oracle_calls": "uint", "cache_hits": "uint",
                    "oracle_unavailable": "uint", "labeled_examples": "uint",
                    "cache": {"entries": "uint", "hits": "uint", "lookups": "uint"}},
        "records": [{
            "request_id": "string", "ts": "uint", "taxonomy_id": "string", "source": "string",
            "router_label": "string", "final_label": "string",
            "agreement_with_router": "bool|null", "oracle_model": "string|null",
            "oracle_latency_ms": "number|null", "oracle_calls": "uint",
            "novelty_score": "number", "flags": ["string"],
        }],
    })
}

/// `admin_create_key` (router `api.rs:492-501`).
fn admin_create_shape() -> Value {
    json!({"key": "string", "account": "string", "plan": "string", "rate_per_min": "uint",
           "decision_quota": "uint", "expires_at": "uint|null", "created_at": "uint",
           "persisted": "bool"})
}

/// `admin_list_keys` (router `api.rs:504-529`).
fn admin_list_shape() -> Value {
    json!({"count": "uint", "keys": [{
        "account": "string", "plan": "string", "rate_per_min": "uint", "decision_quota": "uint",
        "expires_at": "uint|null", "expired": "bool", "key_hash_prefix": "string",
        "usage": {"decisions": "uint", "oracle_calls": "uint"},
    }]})
}

/// `admin_revoke_key` (router `api.rs:531-543`).
fn admin_revoke_shape() -> Value {
    json!({"account": "string", "revoked": "uint"})
}

/// The Prometheus names of router `api.rs:1416-1505`, in its order.
const METRICS: [(&str, &str); 15] = [
    ("cortiq_decisions_total", "counter"),
    ("cortiq_escalations_total", "counter"),
    ("cortiq_oracle_calls_total", "counter"),
    ("cortiq_cache_hits_total", "counter"),
    ("cortiq_oracle_unavailable_total", "counter"),
    ("cortiq_novelty_hits_total", "counter"),
    ("cortiq_refits_total", "counter"),
    ("cortiq_promotions_total", "counter"),
    ("cortiq_escalation_rate", "gauge"),
    ("cortiq_oracle_call_rate", "gauge"),
    ("cortiq_labeled_examples", "gauge"),
    ("cortiq_oracle_cache_entries", "gauge"),
    ("cortiq_oracle_cache_hits_total", "counter"),
    ("cortiq_oracle_cache_lookups_total", "counter"),
    ("cortiq_active_tasks", "gauge"),
];

fn type_ok(v: &Value, t: &str) -> bool {
    match t {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "uint" => v.is_u64(),
        "int" => v.is_i64() || v.is_u64(),
        "bool" => v.is_boolean(),
        "null" => v.is_null(),
        other => panic!("unknown type {other}"),
    }
}

/// `v` has exactly the keys and types of `shape` (see the module notes).
fn conforms(v: &Value, shape: &Value) {
    check(v, shape, "$");
}

fn check(v: &Value, shape: &Value, path: &str) {
    match shape {
        Value::String(t) => assert!(
            t.split('|').any(|alt| type_ok(v, alt)),
            "{path}: expected {t}, got {v}"
        ),
        Value::Array(items) => {
            let a = v
                .as_array()
                .unwrap_or_else(|| panic!("{path}: expected an array, got {v}"));
            for (i, x) in a.iter().enumerate() {
                check(x, &items[0], &format!("{path}[{i}]"));
            }
        }
        Value::Object(m) => {
            let o = v
                .as_object()
                .unwrap_or_else(|| panic!("{path}: expected an object, got {v}"));
            let mut known = BTreeSet::new();
            for (k, s) in m {
                let (optional, name) = match k.strip_prefix('?') {
                    Some(n) => (true, n),
                    None => (false, k.as_str()),
                };
                known.insert(name.to_string());
                match o.get(name) {
                    Some(x) => check(x, s, &format!("{path}.{name}")),
                    None => assert!(optional, "{path}: missing key '{name}' in {v}"),
                }
            }
            for k in o.keys() {
                assert!(
                    known.contains(k),
                    "{path}: key '{k}' is not the router's: {v}"
                );
            }
        }
        _ => panic!("bad shape at {path}"),
    }
}

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

fn service() -> DecisionService {
    let m = DecisionModel::open(&toy().path, Verify::Light).unwrap();
    let h = Arc::new(ModelHandle::new(LoadedModel::new(m).unwrap()));
    DecisionService::open(h, Config::default(), None)
        .unwrap()
        .with_loopback(true)
}

fn action_of(svc: &DecisionService, text: &str) -> Action {
    let mut c = Map::new();
    for l in LABELS {
        c.insert(l.to_string(), json!(format!("about {l}")));
    }
    let b = json!({"model": "cortiq/decision", "state": text,
                   "questions": {"task": {"type": "choice", "instructions": "Which topic?", "criteria": c}}});
    svc.decide_body(&serde_json::to_vec(&b).unwrap(), &Principal::open())
        .unwrap()
        .questions[0]
        .action
}

/// A dev text the gate accepts.
fn accepted() -> &'static str {
    static A: OnceLock<String> = OnceLock::new();
    A.get_or_init(|| {
        let svc = service();
        toy()
            .dev
            .iter()
            .map(|(t, _)| t.clone())
            .find(|t| action_of(&svc, t) == Action::Local)
            .expect("a dev text the gate accepts")
    })
}

/// A text the gate rejects.
fn rejected() -> &'static str {
    static R: OnceLock<String> = OnceLock::new();
    R.get_or_init(|| {
        let svc = service();
        [
            "cruise ship cabin deck please",
            "zzz qqq xyzzy plugh foobar quux wibble",
            "quantum chromodynamics lecture notes",
            "violin sonata rehearsal schedule",
        ]
        .iter()
        .map(|t| t.to_string())
        .find(|t| action_of(&svc, t) == Action::Abstain)
        .expect("a text the gate rejects")
    })
}

fn signal(text: &str) -> Vec<f32> {
    static ENC: OnceLock<SignalEncoder> = OnceLock::new();
    ENC.get_or_init(|| {
        let m = DecisionModel::open(&toy().path, Verify::Light).unwrap();
        SignalEncoder::from_model(&m).unwrap().0
    })
    .signal(text)
}

// ------------------------------------------------------------------ server + HTTP

fn cfg() -> Config {
    let mut c = Config::default();
    c.learning.synchronous = true;
    c
}

struct Srv {
    server: Option<DecisionServer>,
    app: axum::Router,
    _dir: tempfile::TempDir,
}

impl Srv {
    fn open(cfg: Config) -> Self {
        Self::open_with(cfg, Arc::new(|_: &str| None), Some(ADMIN))
    }

    fn open_with(cfg: Config, key: KeyLookup, admin: Option<&str>) -> Self {
        Self::open_in(cfg, key, admin, tempfile::tempdir().unwrap())
    }

    /// A server on the state directory `dir/state` as it is (keys imported
    /// beforehand, for example).
    fn open_in(cfg: Config, key: KeyLookup, admin: Option<&str>, dir: tempfile::TempDir) -> Self {
        let mut o = ServeOptions::new(&toy().path, cfg);
        o.state_dir = Some(dir.path().join("state"));
        o.addr = "127.0.0.1:0".parse().unwrap();
        o.cascade = CascadeOptions {
            key,
            threads: 2,
            created_unix: Some(EPOCH),
        };
        o.admin_token = admin.map(str::to_string);
        let server = DecisionServer::open(&o).expect("open the decision server");
        let app = server.router();
        Self {
            server: Some(server),
            app,
            _dir: dir,
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
            .unwrap();
        let text = String::from_utf8_lossy(&bytes).to_string();
        let r = Resp {
            status,
            body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            headers,
            text,
        };
        assert!(
            r.header("x-request-id").is_some(),
            "no x-request-id on {method} {path}"
        );
        r
    }

    async fn post(&self, path: &str, key: Option<&str>, v: &Value) -> Resp {
        let auth = key.map(|k| format!("Bearer {k}"));
        let mut h = vec![("content-type", "application/json")];
        if let Some(a) = &auth {
            h.push(("authorization", a.as_str()));
        }
        self.call("POST", path, &h, Some(serde_json::to_vec(v).unwrap()))
            .await
    }

    async fn post_raw(&self, path: &str, key: Option<&str>, ct: Option<&str>, body: &[u8]) -> Resp {
        let auth = key.map(|k| format!("Bearer {k}"));
        let mut h = Vec::new();
        if let Some(c) = ct {
            h.push(("content-type", c));
        }
        if let Some(a) = &auth {
            h.push(("authorization", a.as_str()));
        }
        self.call("POST", path, &h, Some(body.to_vec())).await
    }

    async fn get(&self, path: &str, key: Option<&str>) -> Resp {
        let auth = key.map(|k| format!("Bearer {k}"));
        let mut h = Vec::new();
        if let Some(a) = &auth {
            h.push(("authorization", a.as_str()));
        }
        self.call("GET", path, &h, None).await
    }

    async fn admin(&self, method: &str, path: &str, v: Option<&Value>) -> Resp {
        let mut h = vec![("x-admin-token", ADMIN)];
        if v.is_some() {
            h.push(("content-type", "application/json"));
        }
        self.call(method, path, &h, v.map(|v| serde_json::to_vec(v).unwrap()))
            .await
    }

    /// A key through the admin API (ends the open mode).
    async fn key(&self, spec: Value) -> String {
        let r = self.admin("POST", "/v1/admin/keys", Some(&spec)).await;
        assert_eq!(r.status, 200, "{}", r.text);
        r.body["key"].as_str().unwrap().to_string()
    }
}

impl Drop for Srv {
    fn drop(&mut self) {
        if let Some(s) = self.server.take() {
            let _ = s.close();
        }
    }
}

#[derive(Debug)]
struct Resp {
    status: u16,
    headers: HeaderMap,
    body: Value,
    text: String,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }

    /// A router error envelope with exactly the router's keys: (status, code).
    fn router_error(&self) -> (u16, String) {
        assert_eq!(self.header("content-type"), Some("application/json"));
        let shape = if self.body["error"]["code"] == "TAXONOMY_NOT_FOUND" {
            taxonomy_error_shape()
        } else {
            error_shape()
        };
        conforms(&self.body, &shape);
        assert_eq!(self.body["schema_version"], "1.1");
        let id = self.body["request_id"].as_str().unwrap();
        assert!(id.starts_with("req_"), "{id}");
        assert_eq!(Some(id), self.header("x-request-id"));
        assert_eq!(
            self.body["error"]["retriable"],
            json!(matches!(self.status, 429 | 500 | 502))
        );
        (
            self.status,
            self.body["error"]["code"].as_str().unwrap().to_string(),
        )
    }

    /// An axum rejection: `text/plain; charset=utf-8` with this text.
    fn plain(&self) -> (u16, &str) {
        assert_eq!(
            self.header("content-type"),
            Some("text/plain; charset=utf-8"),
            "{}",
            self.text
        );
        (self.status, self.text.as_str())
    }
}

fn route_body(text: &str) -> Value {
    json!({"input": {"text": text}})
}

/// router-client.ts's request (`cortiq-opencode-router/src/router-client.ts:159-170`).
fn client_body(text: &str) -> Value {
    json!({
        "input": {"text": text},
        "options": {"policy_profile": "balanced", "allow_oracle": true, "return_explanation": false, "top_k": 3},
        "taxonomy_id": SKILL,
    })
}

async fn wait_for_fresh_minute() {
    let s = now_unix() % 60;
    if s >= 55 {
        tokio::time::sleep(Duration::from_secs(61 - s)).await;
    }
}

// ------------------------------------------------------------------ mock oracle (loopback)

/// Answers every chat completion with `{"task": <label>}`.
struct MockOracle {
    addr: std::net::SocketAddr,
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
                if read_http_request(&mut s).is_none() {
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

    fn url(&self) -> String {
        format!("http://{}", self.addr)
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

fn read_http_request(s: &mut TcpStream) -> Option<()> {
    s.set_read_timeout(Some(Duration::from_secs(10))).ok()?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
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
    let head = String::from_utf8_lossy(&buf[..head_end]).to_lowercase();
    let len: usize = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    let mut have = buf.len() - head_end - 4;
    while have < len {
        let n = s.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        have += n;
    }
    Some(())
}

fn oracle_config(url: &str) -> Config {
    let mut c = cfg();
    c.oracle.enabled = true;
    c.oracle.base_url = url.to_string();
    c.oracle.api_key_env = ORACLE_KEY_ENV.to_string();
    c.oracle.deadline_s = 5.0;
    c
}

fn oracle_key() -> KeyLookup {
    Arc::new(|name: &str| (name == ORACLE_KEY_ENV).then(|| "sk-test-not-a-real-key".to_string()))
}

// ------------------------------------------------------------------ /v1/route: success shapes

#[tokio::test]
async fn route_minimal_request_has_exactly_the_router_keys_and_defaults() {
    let srv = Srv::open(cfg());
    let r = srv.post("/v1/route", None, &route_body(accepted())).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.header("content-type"), Some("application/json"));
    conforms(&r.body, &route_shape());
    // Defaults (router api.rs:63-74): top_k 3, no explanation, no routing,
    // balanced; client_request_id omitted when absent.
    assert_eq!(r.body["scores"].as_array().unwrap().len(), 3);
    for k in [
        "explanation",
        "routing",
        "oracle",
        "client_request_id",
        "cmf",
    ] {
        assert!(r.body.get(k).is_none(), "{k} present");
    }
    assert_eq!(r.body["schema_version"], "1.1");
    let id = r.body["request_id"].as_str().unwrap();
    assert!(id.starts_with("req_") && id.len() == 4 + 16 + 6, "{id}");
    assert_eq!(r.header("x-request-id"), Some(id));
    let d = &r.body["decision"];
    assert_eq!(d["taxonomy_id"], SKILL);
    assert_eq!(d["source"], "router");
    assert_eq!(d["confident"], true);
    assert_eq!(d["flags"], json!([]));
    assert!(d["complexity"]["factors"].get("difficulty").is_none());
    assert_eq!(
        r.body["usage"],
        json!({"billable_decisions": 1, "oracle_calls": 0})
    );
    assert_eq!(r.body["meta"]["taxonomy_version"], format!("{SKILL}@1"));
    assert_eq!(
        r.body["meta"]["served_by"],
        format!("cortiq/{}", env!("CARGO_PKG_VERSION"))
    );
    // Contract invariants (router api.rs tests `contract_invariants_hold`).
    assert_eq!(r.body["scores"][0]["task_label"], d["task_label"]);
    assert_eq!(
        r.body["scores"][0]["task_id"].as_i64(),
        d["task_id"].as_i64()
    );
}

#[tokio::test]
async fn route_opencode_client_request_is_answered_in_the_router_shape() {
    let srv = Srv::open(cfg());
    let r = srv.post("/v1/route", None, &client_body(accepted())).await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &route_shape());
    // What router-client.ts parseRouteDecision() requires.
    let d = &r.body["decision"];
    let unit = |v: &Value| v.as_f64().is_some_and(|x| (0.0..=1.0).contains(&x));
    assert!(unit(&d["complexity"]["score"]) && unit(&d["confidence"]));
    assert!(["low", "medium", "high"].contains(&d["complexity"]["tier"].as_str().unwrap()));
    assert!(!d["task_label"].as_str().unwrap().trim().is_empty());
}

#[tokio::test]
async fn route_optional_keys_appear_only_when_asked() {
    let mut c = cfg();
    for (t, m) in [("low", "small"), ("medium", "mid"), ("high", "big")] {
        c.routing_tiers.insert(t.into(), m.into());
    }
    let srv = Srv::open(c);
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"input": {"text": accepted()}, "client_request_id": "c-7",
            "options": {"return_explanation": true, "routing_table_id": "default", "top_k": 2}}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &route_shape());
    assert_eq!(r.body["client_request_id"], "c-7");
    assert!(
        r.body["explanation"]["top1_vs_top2"]
            .as_str()
            .unwrap()
            .contains(" leads ")
    );
    assert!(["small", "mid", "big"].contains(&r.body["routing"]["target"].as_str().unwrap()));
    assert_eq!(r.body["scores"].as_array().unwrap().len(), 2);
    // Tiers mapped but no routing_table_id: no routing (router api.rs:945-948).
    let r = srv.post("/v1/route", None, &route_body(accepted())).await;
    assert!(r.body.get("routing").is_none());
    // client_request_id null is None: omitted.
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"input": {"text": accepted()}, "client_request_id": null}),
        )
        .await;
    assert!(r.body.get("client_request_id").is_none());
}

#[tokio::test]
async fn route_top_k_is_clamped_like_the_router() {
    let srv = Srv::open(cfg());
    for (k, n) in [(0, 1), (1, 1), (4, 4), (100, 4)] {
        let r = srv
            .post(
                "/v1/route",
                None,
                &json!({"input": {"text": accepted()}, "options": {"top_k": k}}),
            )
            .await;
        assert_eq!(r.status, 200, "{}", r.text);
        conforms(&r.body, &route_shape());
        assert_eq!(r.body["scores"].as_array().unwrap().len(), n, "top_k {k}");
    }
}

#[tokio::test]
async fn route_every_policy_profile_answers_the_router_shape() {
    let srv = Srv::open(cfg());
    for p in ["cost-saver", "balanced", "quality-first"] {
        for text in [accepted(), rejected()] {
            let r = srv
                .post(
                    "/v1/route",
                    None,
                    &json!({"input": {"text": text}, "options": {"policy_profile": p}}),
                )
                .await;
            assert_eq!(r.status, 200, "{p}: {}", r.text);
            conforms(&r.body, &route_shape());
        }
    }
}

#[tokio::test]
async fn route_gate_rejected_without_oracle_degrades_like_the_router() {
    let srv = Srv::open(cfg());
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"input": {"text": rejected()}, "options": {"return_explanation": true}}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &route_shape());
    let d = &r.body["decision"];
    assert_eq!(d["source"], "router");
    assert_eq!(d["confident"], false);
    assert_eq!(d["flags"][0], "low_confidence");
    assert!(
        r.body.get("oracle").is_none(),
        "no oracle block when not consulted"
    );
}

#[tokio::test]
async fn route_byo_embedding_escalation_has_the_router_oracle_block() {
    // Oracle enabled (no key needed: an embedding is never sent anywhere; the
    // base URL is a closed loopback port all the same).
    let mut c = cfg();
    c.oracle.enabled = true;
    c.oracle.base_url = "http://127.0.0.1:9".into();
    let srv = Srv::open(c);
    let r = srv
        .post("/v1/route", None, &json!({"input": {"embedding": signal(rejected()), "embedding_model": "cortiq-decision-ph-v1"}}))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &route_shape());
    let o = &r.body["oracle"];
    assert_eq!(o["consulted"], false);
    assert!(o["model"].is_string());
    assert!(o["agreement_with_router"].is_null() && o["latency_ms"].is_null());
    assert_eq!(
        r.body["decision"]["flags"],
        json!(["low_confidence", "oracle_unavailable"])
    );
    // An accepted signal: the router shape without the oracle block.
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"input": {"embedding": signal(accepted()), "embedding_model": "m"}}),
        )
        .await;
    conforms(&r.body, &route_shape());
    assert!(r.body.get("oracle").is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn route_oracle_and_cache_answers_have_the_router_shape() {
    let mock = MockOracle::answering("travel");
    let srv = Srv::open_with(oracle_config(&mock.url()), oracle_key(), Some(ADMIN));
    let r = srv.post("/v1/route", None, &route_body(rejected())).await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &route_shape());
    assert_eq!(r.body["decision"]["source"], "oracle");
    assert_eq!(r.body["decision"]["task_label"], "travel");
    let o = &r.body["oracle"];
    assert_eq!(o["consulted"], true);
    assert!(o["model"].is_string() && o["agreement_with_router"].is_boolean());
    assert!(o["latency_ms"].is_number());
    assert_eq!(r.body["usage"]["oracle_calls"], 1);
    assert_eq!(mock.hits(), 1);
    // The same text: the semantic cache, no call (router api.rs:680-695).
    let c = srv.post("/v1/route", None, &route_body(rejected())).await;
    conforms(&c.body, &route_shape());
    assert_eq!(c.body["decision"]["source"], "cache");
    assert_eq!(c.body["oracle"]["consulted"], false);
    assert!(c.body["oracle"]["latency_ms"].is_null());
    assert_eq!(c.body["usage"]["oracle_calls"], 0);
    assert_eq!(mock.hits(), 1);
    // The escalation audit in the router's AuditRecord shape.
    let e = srv.get("/v1/escalations", None).await;
    conforms(&e.body, &escalations_shape());
    let rec = e.body["records"].as_array().unwrap();
    assert_eq!(rec.len(), 2);
    assert_eq!(
        (rec[0]["source"].as_str(), rec[1]["source"].as_str()),
        (Some("cache"), Some("oracle"))
    );
}

#[tokio::test]
async fn route_long_text_is_decided_not_rejected() {
    // The router has no text limit (12,000 CJK characters from the opencode
    // client are 36 KB, above limits.state_bytes).
    let srv = Srv::open(cfg());
    let text = "账单发票退款".repeat(2000);
    let r = srv.post("/v1/route", None, &client_body(&text)).await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &route_shape());
}

#[tokio::test]
async fn route_ignores_unknown_keys_like_the_router_serde() {
    let srv = Srv::open(cfg());
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"input": {"text": accepted(), "lang": "en"},
            "options": {"foo": 1}, "extra": true}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &route_shape());
    // A +json media type is JSON to axum.
    let r = srv
        .post_raw(
            "/v1/route",
            None,
            Some("application/cloudevents+json"),
            &serde_json::to_vec(&route_body(accepted())).unwrap(),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
}

// ------------------------------------------------------------------ /v1/route: error shapes

#[tokio::test]
async fn route_router_codes_have_the_router_error_envelope() {
    let srv = Srv::open(cfg());
    for input in [json!({}), json!({"text": "   "}), json!({"text": null})] {
        let r = srv.post("/v1/route", None, &json!({"input": input})).await;
        assert_eq!(r.router_error(), (400, "EMBEDDING_REQUIRED".into()));
    }
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"taxonomy_id": "nope", "input": {"text": "x"}}),
        )
        .await;
    assert_eq!(r.router_error(), (404, "TAXONOMY_NOT_FOUND".into()));
    assert_eq!(r.body["error"]["details"], json!({"taxonomy_id": "nope"}));
    assert_eq!(
        r.body["error"]["message"],
        "taxonomy_id 'nope' not found for account"
    );
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"input": {"embedding": [0.0, 1.0], "embedding_model": "m"}}),
        )
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".into()));
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"input": {"embedding": signal(accepted())}}),
        )
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".into()));
    assert_eq!(
        r.body["error"]["message"],
        "input.embedding_model is required when input.embedding is provided"
    );
}

#[tokio::test]
async fn route_rejected_bodies_are_axum_texts() {
    let srv = Srv::open(cfg());
    let prefix422 = "Failed to deserialize the JSON body into the target type: ";
    let cases = [
        (
            json!({"options": {}}),
            "missing field `input` at line 1 column 14",
        ),
        (
            json!({"input": {"text": "x"}, "options": {"policy_profile": "yolo"}}),
            "options.policy_profile: unknown variant `yolo`, expected one of `cost-saver`, `balanced`, `quality-first` at line 1 column 56",
        ),
        (
            json!({"input": {"text": "x"}, "options": {"top_k": -1}}),
            "options.top_k: invalid value: integer `-1`, expected usize at line 1 column 43",
        ),
        (
            json!({"input": {"text": "x"}, "options": {"allow_oracle": null}}),
            "options.allow_oracle: invalid type: null, expected a boolean at line 1 column 52",
        ),
        (
            json!({"input": "text"}),
            "input: invalid type: string \"text\", expected struct Input at line 1 column 15",
        ),
        (
            json!({"input": {"text": "x"}, "client_request_id": 5}),
            "client_request_id: invalid type: integer `5`, expected a string at line 1 column 43",
        ),
    ];
    for (body, want) in cases {
        let r = srv.post("/v1/route", None, &body).await;
        assert_eq!(
            r.plain(),
            (422, format!("{prefix422}{want}").as_str()),
            "{body}"
        );
    }
    let r = srv
        .post_raw("/v1/route", None, Some("application/json"), b"{\"input\":")
        .await;
    assert_eq!(
        r.plain(),
        (
            400,
            "Failed to parse the request body as JSON: input: EOF while parsing a value at line 1 column 9"
        )
    );
    for ct in [None, Some("text/plain"), Some("text/json")] {
        let r = srv
            .post_raw(
                "/v1/route",
                None,
                ct,
                &serde_json::to_vec(&route_body("x")).unwrap(),
            )
            .await;
        assert_eq!(
            r.plain(),
            (
                415,
                "Expected request with `Content-Type: application/json`"
            )
        );
    }
}

#[tokio::test]
async fn route_body_over_the_limit_is_413_text() {
    let mut c = cfg();
    c.limits.body_bytes = 1024;
    c.limits.state_bytes = 1024;
    let srv = Srv::open(c);
    let r = srv
        .post("/v1/route", None, &route_body(&"invoice ".repeat(400)))
        .await;
    assert_eq!(
        r.plain(),
        (
            413,
            "Failed to buffer the request body: length limit exceeded"
        )
    );
}

#[tokio::test]
async fn keys_401_on_every_keyed_router_endpoint() {
    let srv = Srv::open(cfg());
    let key = srv.key(json!({"account": "acme"})).await;
    let fb = json!({"request_id": "req_x", "correct_task_label": "billing"});
    for (m, p) in [
        ("POST", "/v1/route"),
        ("POST", "/v1/route:batch"),
        ("POST", "/v1/feedback"),
        ("GET", "/v1/taxonomies"),
        ("GET", "/v1/taxonomies/data-assistant"),
        ("GET", "/v1/usage"),
        ("GET", "/v1/escalations"),
    ] {
        for k in [
            None,
            Some("cortiq_0000000000000000000000000000000000000000"),
        ] {
            let r = if m == "POST" {
                srv.post(p, k, &fb).await
            } else {
                srv.get(p, k).await
            };
            assert_eq!(r.router_error(), (401, "UNAUTHORIZED".into()), "{m} {p}");
        }
    }
    let r = srv.post("/v1/route", None, &client_body(accepted())).await;
    assert_eq!(
        r.body["error"]["message"],
        "missing API key (Authorization: Bearer … or x-api-key)"
    );
    // The key works through Authorization and x-api-key.
    assert_eq!(
        srv.post("/v1/route", Some(&key), &client_body(accepted()))
            .await
            .status,
        200
    );
    let r = srv
        .call(
            "GET",
            "/v1/taxonomies",
            &[("x-api-key", key.as_str())],
            None,
        )
        .await;
    assert_eq!(r.status, 200);
}

#[tokio::test]
async fn rate_limit_is_429_and_quota_402_in_the_router_envelope() {
    let srv = Srv::open(cfg());
    let slow = srv.key(json!({"account": "slow", "rate_per_min": 1})).await;
    wait_for_fresh_minute().await;
    assert_eq!(
        srv.post("/v1/route", Some(&slow), &client_body(accepted()))
            .await
            .status,
        200
    );
    let r = srv
        .post("/v1/route", Some(&slow), &client_body(accepted()))
        .await;
    assert_eq!(r.router_error(), (429, "RATE_LIMITED".into()));
    assert_eq!(r.body["error"]["retriable"], true);
    let small = srv
        .key(json!({"account": "small", "decision_quota": 1}))
        .await;
    assert_eq!(
        srv.post("/v1/route", Some(&small), &client_body(accepted()))
            .await
            .status,
        200
    );
    let r = srv
        .post("/v1/route", Some(&small), &client_body(accepted()))
        .await;
    assert_eq!(r.router_error(), (402, "QUOTA_EXCEEDED".into()));
    assert_eq!(r.body["error"]["retriable"], false);
    // The quota gate covers every keyed endpoint (router middleware).
    let r = srv.get("/v1/usage", Some(&small)).await;
    assert_eq!(r.router_error(), (402, "QUOTA_EXCEEDED".into()));
}

// ------------------------------------------------------------------ key import (package C2)

/// Spec decision-v4 §4.15, package C2: keys imported from a synthetic MySQL
/// `api_keys` export and a synthetic router configuration answer with their
/// raw keys on `/v1/route` (router shape) and `/api/alpha/decisions`; an
/// inactive and an expired key get the router's 401 on both; limits of 0
/// are unlimited, the others hold, and imported usage counts against quotas.
#[tokio::test]
async fn imported_router_keys_answer_on_both_apis() {
    use cortiq_decision::keys::{self, ImportFormat, KeyStore};
    use cortiq_decision::ledger::UsageLedger;
    use cortiq_decision::statedir::StateDir;
    let dir = tempfile::tempdir().unwrap();
    let state = StateDir::open(dir.path().join("state")).unwrap();
    let raw = |n: u8| format!("cortiq_{}", format!("{n:x}").repeat(40));
    let (live, off, old, slow, spent, cfgkey) = (raw(1), raw(2), raw(3), raw(4), raw(5), raw(6));
    let export = json!([
        {"key_hash": keys::hash_key(&live), "account": "acct_live", "plan": "pro", "active": 1,
         "rate_per_min": 0, "decision_quota": 0},
        {"key_hash": keys::hash_key(&off), "account": "acct_off", "active": 0},
        {"key_hash": keys::hash_key(&old), "account": "acct_old", "expires_at": 1_000},
        {"key_hash": keys::hash_key(&slow), "account": "acct_slow", "rate_per_min": "1"},
        {"key_hash": keys::hash_key(&spent), "account": "acct_spent", "decision_quota": 3},
    ])
    .to_string();
    let config = format!("[[api_keys]]\nkey = \"{cfgkey}\"\naccount = \"acct_cfg\"\n");
    let store = KeyStore::open(state.keys_path(), "cortiq_").unwrap();
    let now = now_unix();
    for (bytes, format) in [
        (export.as_bytes(), ImportFormat::MysqlJson),
        (config.as_bytes(), ImportFormat::RouterToml),
    ] {
        let k = keys::read_router_keys(bytes, format, now).unwrap();
        assert!(store.import_router_keys(&k, now).unwrap().written);
        let again = store.import_router_keys(&k, now).unwrap();
        assert_eq!((again.imported, again.written), (0, false));
    }
    let on_disk = std::fs::read_to_string(state.keys_path()).unwrap();
    for k in [&live, &off, &old, &slow, &spent, &cfgkey] {
        assert!(!on_disk.contains(k.as_str()) && on_disk.contains(&keys::hash_key(k)));
    }
    let ledger = UsageLedger::open(state.usage_dir()).unwrap();
    let usage = keys::read_router_usage(
        br#"[{"account": "acct_spent", "decisions": 3, "oracle_calls": 1}]"#,
    )
    .unwrap();
    assert_eq!(
        keys::import_router_usage(&ledger, &usage, now)
            .unwrap()
            .carried,
        1
    );
    drop((ledger, store));

    let srv = Srv::open_in(cfg(), Arc::new(|_: &str| None), Some(ADMIN), dir);
    let mut crit = Map::new();
    for l in LABELS {
        crit.insert(l.to_string(), json!(format!("about {l}")));
    }
    let jev = json!({"model": "cortiq/decision", "state": accepted(),
                     "questions": {"task": {"type": "choice", "instructions": "Which topic?",
                                            "criteria": crit}}});
    for key in [&live, &cfgkey] {
        for _ in 0..3 {
            let r = srv
                .post("/v1/route", Some(key), &client_body(accepted()))
                .await;
            assert_eq!(r.status, 200, "{}", r.text);
            conforms(&r.body, &route_shape());
            let r = srv.post("/api/alpha/decisions", Some(key), &jev).await;
            assert_eq!(r.status, 200, "{}", r.text);
        }
    }
    let r = srv.get("/v1/usage", Some(&live)).await;
    conforms(&r.body, &usage_shape());
    assert_eq!(
        (
            &r.body["account"]["billable_decisions"],
            &r.body["account"]["decision_quota"],
            &r.body["account"]["rate_per_min"]
        ),
        (&json!(6), &json!(0), &json!(0)),
        "{}",
        r.text
    );
    for key in [&off, &old] {
        let r = srv
            .post("/v1/route", Some(key), &client_body(accepted()))
            .await;
        assert_eq!(r.router_error(), (401, "UNAUTHORIZED".into()));
        let r = srv.post("/api/alpha/decisions", Some(key), &jev).await;
        assert_eq!(r.status, 401, "{}", r.text);
    }
    wait_for_fresh_minute().await;
    let r = srv
        .post("/v1/route", Some(&slow), &client_body(accepted()))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let r = srv.post("/api/alpha/decisions", Some(&slow), &jev).await;
    assert_eq!(r.status, 429, "{}", r.text);
    let r = srv
        .post("/v1/route", Some(&spent), &client_body(accepted()))
        .await;
    assert_eq!(r.router_error(), (402, "QUOTA_EXCEEDED".into()));
}

// ------------------------------------------------------------------ batch

#[tokio::test]
async fn batch_has_the_router_shape_and_errors() {
    let srv = Srv::open(cfg());
    let r = srv
        .post(
            "/v1/route:batch",
            None,
            &json!({"taxonomy_id": SKILL,
            "inputs": [{"text": accepted()}, {"text": rejected()}], "options": {"top_k": 2}}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &batch_shape());
    assert_eq!(r.body["results"].as_array().unwrap().len(), 2);
    let r = srv
        .post("/v1/route:batch", None, &json!({"inputs": []}))
        .await;
    assert_eq!(r.body, json!({"schema_version": "1.1", "results": []}));
    // Errors: the taxonomy before the size, the size, a failing input.
    let many: Vec<Value> = (0..1025).map(|_| json!({"text": "x"})).collect();
    let r = srv
        .post(
            "/v1/route:batch",
            None,
            &json!({"taxonomy_id": "nope", "inputs": many}),
        )
        .await;
    assert_eq!(r.router_error(), (404, "TAXONOMY_NOT_FOUND".into()));
    let r = srv
        .post("/v1/route:batch", None, &json!({"inputs": many}))
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".into()));
    assert_eq!(r.body["error"]["message"], "batch exceeds 1024 inputs");
    let r = srv
        .post(
            "/v1/route:batch",
            None,
            &json!({"inputs": [{"text": "x"}, {}]}),
        )
        .await;
    assert_eq!(r.router_error(), (400, "EMBEDDING_REQUIRED".into()));
    let r = srv
        .post("/v1/route:batch", None, &json!({"input": {"text": "x"}}))
        .await;
    assert_eq!(
        r.plain(),
        (
            422,
            "Failed to deserialize the JSON body into the target type: missing field `inputs` at line 1 column 22"
        )
    );
}

// ------------------------------------------------------------------ feedback

#[tokio::test]
async fn feedback_has_the_router_shape_and_codes() {
    let srv = Srv::open(cfg());
    let r = srv.post("/v1/route", None, &route_body(accepted())).await;
    let rid = r.body["request_id"].as_str().unwrap().to_string();
    let label = if r.body["decision"]["task_label"] == "cards" {
        "billing"
    } else {
        "cards"
    };
    let fb = srv
        .post(
            "/v1/feedback",
            None,
            &json!({"request_id": rid, "correct_task_label": label}),
        )
        .await;
    assert_eq!(fb.status, 200, "{}", fb.text);
    conforms(&fb.body, &feedback_shape());
    assert_eq!(fb.body["accepted"], true);
    assert_eq!(
        fb.body["message"],
        format!("feedback recorded for '{label}'")
    );
    // Consumed (router api.rs:1265-1272).
    let again = srv
        .post(
            "/v1/feedback",
            None,
            &json!({"request_id": rid, "correct_task_label": label}),
        )
        .await;
    assert_eq!(again.router_error(), (404, "INVALID_REQUEST".into()));
    assert_eq!(
        again.body["error"]["message"],
        "request_id not found or already consumed"
    );
    // A label the taxonomy does not have: accepted (a cold start), as in the router.
    let r = srv.post("/v1/route", None, &route_body(accepted())).await;
    let fb = srv
        .post(
            "/v1/feedback",
            None,
            &json!({"request_id": r.body["request_id"], "correct_task_label": "insurance"}),
        )
        .await;
    assert_eq!(fb.status, 200, "{}", fb.text);
    conforms(&fb.body, &feedback_shape());
    // The router's required fields.
    let bad = srv
        .post("/v1/feedback", None, &json!({"request_id": "req_1"}))
        .await;
    assert_eq!(
        bad.plain(),
        (
            422,
            "Failed to deserialize the JSON body into the target type: missing field `correct_task_label` at line 1 column 22"
        )
    );
}

// ------------------------------------------------------------------ listings

#[tokio::test]
async fn taxonomies_listing_and_detail_have_the_router_shape() {
    let srv = Srv::open(cfg());
    let t = srv.get("/v1/taxonomies", None).await;
    assert_eq!(t.status, 200);
    conforms(&t.body, &taxonomies_shape());
    let list = t.body["taxonomies"].as_array().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["taxonomy_id"], SKILL);
    assert_eq!(list[0]["labels"], json!(LABELS));
    assert_eq!(list[0]["taxonomy_version"], format!("{SKILL}@1"));
    let one = srv.get("/v1/taxonomies/data-assistant", None).await;
    conforms(&one.body, &taxonomy_shape());
    assert_eq!(one.body, list[0]);
    let r = srv.get("/v1/taxonomies/nope", None).await;
    assert_eq!(r.router_error(), (404, "TAXONOMY_NOT_FOUND".into()));
}

#[tokio::test]
async fn usage_has_the_router_shape() {
    let srv = Srv::open(cfg());
    let key = srv
        .key(json!({"account": "acme", "plan": "developer"}))
        .await;
    srv.post("/v1/route", Some(&key), &route_body(accepted()))
        .await;
    srv.post("/v1/route", Some(&key), &route_body(rejected()))
        .await;
    let u = srv.get("/v1/usage", Some(&key)).await;
    assert_eq!(u.status, 200, "{}", u.text);
    conforms(&u.body, &usage_shape());
    assert_eq!(u.body["account"]["id"], "acme");
    assert_eq!(u.body["account"]["billable_decisions"], 2);
    assert_eq!(u.body["account"]["rate_per_min"], 120);
    assert_eq!(u.body["account"]["decision_quota"], 100_000);
    assert_eq!(u.body["usage"]["escalations"], 1);
}

#[tokio::test]
async fn escalations_have_the_router_shape_and_query_errors() {
    let srv = Srv::open(cfg());
    srv.post("/v1/route", None, &route_body(rejected())).await;
    let e = srv.get("/v1/escalations", None).await;
    assert_eq!(e.status, 200);
    conforms(&e.body, &escalations_shape());
    assert_eq!(e.body["records"].as_array().unwrap().len(), 1);
    // A degraded escalation: source router, the response's low_confidence flag.
    let rec = &e.body["records"][0];
    assert_eq!(
        (rec["source"].as_str(), rec["flags"][0].as_str()),
        (Some("router"), Some("low_confidence"))
    );
    let e = srv.get("/v1/escalations?limit=0&other=1", None).await;
    assert_eq!(
        e.body["records"].as_array().unwrap().len(),
        1,
        "limit clamps to 1..1000"
    );
    let e = srv.get("/v1/escalations?limit=x", None).await;
    assert_eq!(
        e.plain(),
        (
            400,
            "Failed to deserialize query string: invalid digit found in string"
        )
    );
}

#[tokio::test]
async fn healthz_readyz_and_metrics_are_the_routers() {
    let srv = Srv::open(cfg());
    let key = srv.key(json!({})).await;
    let h = srv.get("/v1/healthz", None).await;
    assert_eq!((h.status, &h.body), (200, &json!({"status": "ok"})));
    let r = srv.get("/v1/readyz", None).await;
    assert_eq!((r.status, &r.body), (200, &json!({"status": "ready"})));
    srv.post("/v1/route", Some(&key), &route_body(accepted()))
        .await;
    let m = srv.get("/metrics", None).await;
    assert_eq!(m.status, 200);
    assert_eq!(m.header("content-type"), Some("text/plain; version=0.0.4"));
    let lines: Vec<&str> = m.text.lines().collect();
    assert_eq!(lines.len(), 3 * METRICS.len());
    for (i, (name, typ)) in METRICS.iter().enumerate() {
        assert!(
            lines[3 * i].starts_with(&format!("# HELP {name} ")),
            "{}",
            lines[3 * i]
        );
        assert_eq!(lines[3 * i + 1], format!("# TYPE {name} {typ}"));
        let (n, v) = lines[3 * i + 2].split_once(' ').unwrap();
        assert_eq!(n, *name);
        assert!(v.parse::<f64>().is_ok(), "{v}");
    }
    assert!(m.text.contains("\ncortiq_decisions_total 1\n"));
    assert!(m.text.contains("\ncortiq_escalation_rate 0.000000\n"));
}

// ------------------------------------------------------------------ admin keys

#[tokio::test]
async fn admin_key_create_list_revoke_have_the_router_shape() {
    let srv = Srv::open(cfg());
    let r = srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"plan": "pro", "account": "acme",
            "email": "ops@acme.test", "label": "prod", "unknown_field": 1})),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    conforms(&r.body, &admin_create_shape());
    assert_eq!(
        (
            r.body["plan"].as_str(),
            r.body["rate_per_min"].as_u64(),
            r.body["decision_quota"].as_u64()
        ),
        (Some("pro"), Some(600), Some(1_000_000))
    );
    assert_eq!(r.body["persisted"], true);
    let raw = r.body["key"].as_str().unwrap().to_string();
    assert!(raw.starts_with("cortiq_") && raw.len() == 47);
    let r = srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"account": "tmp", "days": 7, "rate_per_min": 5, "decision_quota": 9})),
        )
        .await;
    conforms(&r.body, &admin_create_shape());
    assert_eq!(
        r.body["expires_at"].as_u64(),
        Some(r.body["created_at"].as_u64().unwrap() + 7 * 86_400)
    );
    let l = srv.admin("GET", "/v1/admin/keys", None).await;
    conforms(&l.body, &admin_list_shape());
    assert_eq!(l.body["count"], 2);
    let d = srv.admin("DELETE", "/v1/admin/keys/tmp", None).await;
    assert_eq!(d.status, 200);
    conforms(&d.body, &admin_revoke_shape());
    assert_eq!(d.body, json!({"account": "tmp", "revoked": 1}));
    let d = srv.admin("DELETE", "/v1/admin/keys/nobody", None).await;
    assert_eq!(d.body, json!({"account": "nobody", "revoked": 0}));
    // A revoked key leaves the listing, as it leaves the router's gate.
    let l = srv.admin("GET", "/v1/admin/keys", None).await;
    conforms(&l.body, &admin_list_shape());
    assert_eq!(l.body["count"], 1);
    assert_eq!(l.body["keys"][0]["account"], "acme");
    assert!(!l.text.contains(&raw) && !l.text.contains("ops@acme.test"));
}

#[tokio::test]
async fn admin_errors_are_the_routers() {
    let off = Srv::open_with(cfg(), Arc::new(|_: &str| None), None);
    let r = off.admin("GET", "/v1/admin/keys", None).await;
    assert_eq!(r.router_error(), (404, "ADMIN_DISABLED".into()));
    let srv = Srv::open(cfg());
    for t in [None, Some("wrong")] {
        let mut h = Vec::new();
        if let Some(t) = t {
            h.push(("x-admin-token", t));
        }
        let r = srv.call("GET", "/v1/admin/keys", &h, None).await;
        assert_eq!(r.router_error(), (401, "UNAUTHORIZED".into()));
        let r = srv.call("DELETE", "/v1/admin/keys/acme", &h, None).await;
        assert_eq!(r.router_error(), (401, "UNAUTHORIZED".into()));
    }
    // The router's Json extractor runs before its token check.
    let r = srv.post_raw("/v1/admin/keys", None, None, b"{}").await;
    assert_eq!(
        r.plain(),
        (
            415,
            "Expected request with `Content-Type: application/json`"
        )
    );
    let r = srv.post("/v1/admin/keys", None, &json!({"days": -1})).await;
    assert_eq!(
        r.plain(),
        (
            422,
            "Failed to deserialize the JSON body into the target type: days: invalid value: integer `-1`, expected u32 at line 1 column 10"
        )
    );
    let r = srv.post("/v1/admin/keys", None, &json!({})).await;
    assert_eq!(r.router_error(), (401, "UNAUTHORIZED".into()));
}

// ------------------------------------------------------------------ unrouted, headers, extensions

#[tokio::test]
async fn wrong_method_and_unknown_router_path_are_axum_empty_answers_after_the_key() {
    let srv = Srv::open(cfg());
    let key = srv.key(json!({})).await;
    // Keyed path: the middleware first.
    let r = srv.get("/v1/route", None).await;
    assert_eq!(r.router_error(), (401, "UNAUTHORIZED".into()));
    let r = srv.get("/v1/route", Some(&key)).await;
    assert_eq!((r.status, r.text.as_str()), (405, ""));
    assert_eq!(r.header("allow"), Some("POST"));
    assert!(r.header("content-type").is_none());
    let r = srv.get("/v1/taxonomies/a/b", Some(&key)).await;
    assert_eq!((r.status, r.text.as_str()), (404, ""));
    let r = srv.get("/v1/taxonomies/a/b", None).await;
    assert_eq!(r.router_error(), (401, "UNAUTHORIZED".into()));
    // Probes are open.
    let r = srv.call("POST", "/v1/healthz", &[], None).await;
    assert_eq!((r.status, r.text.as_str()), (405, ""));
}

#[tokio::test]
async fn cmf_extensions_are_opt_in() {
    let srv = Srv::open(cfg());
    let ext = [
        ("content-type", "application/json"),
        ("x-cmf-extensions", "1"),
    ];
    let body = serde_json::to_vec(&route_body(accepted())).unwrap();
    let r = srv.call("POST", "/v1/route", &ext, Some(body)).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.body["cmf"]["action"], "local");
    assert!(r.body["cmf"]["certified"].is_boolean());
    let mut plain = r.body.clone();
    plain.as_object_mut().unwrap().remove("cmf");
    conforms(&plain, &route_shape());
    let t = srv.call("GET", "/v1/taxonomies", &ext[1..], None).await;
    assert!(t.body["taxonomies"][0]["cmf"]["generation"].is_u64());
    let u = srv.call("GET", "/v1/usage", &ext[1..], None).await;
    assert!(u.body["cmf"]["totals"]["input_tokens"].is_u64());
    let h = srv.call("GET", "/v1/healthz", &ext[1..], None).await;
    assert_eq!(h.body["cmf"]["skills"], 1);
    let e = srv
        .call("POST", "/v1/route", &ext, Some(br#"{"input":{}}"#.to_vec()))
        .await;
    assert_eq!(e.body["error"]["metadata"]["reason"], "EMBEDDING_REQUIRED");
    // Without the header none of it.
    let e = srv.post("/v1/route", None, &json!({"input": {}})).await;
    assert!(e.body["error"].get("metadata").is_none());
}

#[tokio::test]
async fn request_ids_are_router_ids_in_the_header_and_the_body() {
    let srv = Srv::open(cfg());
    let is_router_id = |s: &str| {
        s.len() == 26 && s.starts_with("req_") && s[4..].bytes().all(|b| b.is_ascii_hexdigit())
    };
    let r = srv.post("/v1/route", None, &route_body(accepted())).await;
    let id = r.body["request_id"].as_str().unwrap();
    assert!(is_router_id(id), "{id}");
    assert_eq!(r.header("x-request-id"), Some(id));
    let e = srv.post("/v1/route", None, &json!({"input": {}})).await;
    let eid = e.body["request_id"].as_str().unwrap();
    assert!(is_router_id(eid) && eid != id);
    assert_eq!(e.header("x-request-id"), Some(eid));
    let b = srv
        .post(
            "/v1/route:batch",
            None,
            &json!({"inputs": [{"text": accepted()}, {"text": accepted()}]}),
        )
        .await;
    let ids: BTreeSet<&str> = b.body["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["request_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|i| is_router_id(i)));
    assert!(is_router_id(b.header("x-request-id").unwrap()));
    for p in [
        "/v1/taxonomies",
        "/v1/usage",
        "/v1/escalations",
        "/v1/healthz",
        "/v1/readyz",
        "/metrics",
    ] {
        assert!(
            is_router_id(srv.get(p, None).await.header("x-request-id").unwrap()),
            "{p}"
        );
    }
}

#[tokio::test]
async fn admin_create_refuses_unknown_plans_and_bad_accounts_in_the_router_envelope() {
    // Unlike the router (a typo'd plan minted an unlimited key), a plan must be
    // one of auth.plans and an account [A-Za-z0-9_.@-].
    let srv = Srv::open(cfg());
    let r = srv
        .admin("POST", "/v1/admin/keys", Some(&json!({"plan": "platinum"})))
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".into()));
    let r = srv
        .admin("POST", "/v1/admin/keys", Some(&json!({"account": "a b"})))
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".into()));
    // Every router plan name mints (spec §4.10 limits).
    for (plan, rate) in [
        ("starter", 60),
        ("developer", 120),
        ("pro", 600),
        ("scale", 3000),
    ] {
        let r = srv
            .admin("POST", "/v1/admin/keys", Some(&json!({"plan": plan})))
            .await;
        conforms(&r.body, &admin_create_shape());
        assert_eq!(r.body["rate_per_min"], rate);
        assert!(r.body["account"].as_str().unwrap().starts_with("acct_"));
    }
}

#[tokio::test]
async fn feedback_endpoint_still_serves_decisions_api_bodies() {
    let srv = Srv::open(cfg());
    let mut c = Map::new();
    for l in LABELS {
        c.insert(l.to_string(), json!(format!("about {l}")));
    }
    let d = srv
        .post("/v1/decisions", None, &json!({"model": "cortiq/decision", "state": accepted(),
            "questions": {"task": {"type": "choice", "instructions": "Which topic?", "criteria": c}}}))
        .await;
    assert_eq!(d.status, 200, "{}", d.text);
    let r = srv
        .post(
            "/v1/feedback",
            None,
            &json!({"id": d.body["id"], "question": "task", "label": "travel"}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["schema_version"], "1.1");
    assert_eq!(r.body["message"], "feedback recorded for 'travel'");
    // Not an option of that question: the decisions API's rule (400).
    let d2 = srv
        .post("/v1/decisions", None, &json!({"model": "cortiq/decision", "state": accepted(),
            "questions": {"task": {"type": "choice", "instructions": "Which topic?", "criteria": c}}}))
        .await;
    let r = srv
        .post(
            "/v1/feedback",
            None,
            &json!({"id": d2.body["id"], "question": "task", "label": "insurance"}),
        )
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".into()));
}

#[tokio::test]
async fn the_decisions_surface_keeps_its_openrouter_errors() {
    let srv = Srv::open(cfg());
    let r = srv
        .post_raw("/v1/decisions", None, Some("text/plain"), b"{}")
        .await;
    assert_eq!(r.status, 400);
    let e = &r.body["error"];
    assert_eq!(e["code"], 400);
    assert_eq!(e["metadata"]["reason"], "INVALID_REQUEST");
    assert!(r.header("x-request-id").unwrap().starts_with("cmf-dec-"));
    let r = srv.get("/v1/nothing-here", None).await;
    assert_eq!((r.status, &r.body["error"]["code"]), (404, &json!(404)));
}
