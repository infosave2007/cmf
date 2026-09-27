//! The oracle cascade through the HTTP layer (spec decision-v4 §6.5, the
//! HTTP-level cases), hermetic: every oracle call goes to an in-test OpenRouter
//! mock on 127.0.0.1; the key is a fake read through the server's key lookup
//! (and, in one child process, from the real environment).
//!
//! * abstain → oracle with usage, cost and passthrough, over `/v1/decisions` and
//!   `/v1/route` (router fields `oracle`, `source`, promoted score), then the
//!   cache; the escalation audit and the metrics;
//! * gate-accepted questions never reach the oracle, even with `cmf.oracle`;
//! * cache: a repeat and a paraphrase make no call; single flight: two parallel
//!   identical HTTP requests make one call;
//! * learning: 25 oracle answers promote (generation 1, isolation of every other
//!   task, the next similar request local); rollback and restart over HTTP
//!   restore the served generation and the cache; a holdout regression (25
//!   wrong feedbacks over `/v1/feedback`) is refused; a cold start;
//! * budget and stop rules: a reservation above the budget, `max_calls`, HTTP
//!   401/402/403, `unexpected_model`, a cost above the reservation,
//!   `max_errors`; `POST /v1/admin/oracle {"enabled":true}` resumes; untrained
//!   questions get 503;
//! * bad answers: 200 with only `error`, invalid JSON, the deadline; untrained
//!   questions get 502;
//! * consent: `oracle.enabled: false`, `cmf.oracle: false`, a key with
//!   `oracle_allowed: false`, router `allow_oracle: false` — 0 calls;
//! * PII redaction on by default;
//! * the request body for one choice question is byte for byte the v4 driver's
//!   (9 ledger fixtures of `cortiq-decision`, sent over HTTP);
//! * the key comes only from the environment and its bytes are in no response,
//!   log line or state file (child process).
//!
//! Every request of [`Srv`] carries `x-cmf-extensions: 1`, so router-surface
//! answers include the opt-in `cmf` diagnostics (the exact default router
//! shapes are checked in `router_compat.rs`).

#[path = "support/toy_dir.rs"]
mod toy_dir;

use axum::body::Body;
use axum::http::{HeaderMap, Request};
use cortiq_decision::build::{self, TrainOptions};
use cortiq_decision::cascade::CascadeOptions;
use cortiq_decision::config::Config;
use cortiq_decision::container::{DecisionModel, Verify};
use cortiq_decision::oracle::{KeyLookup, process_env};
use cortiq_decision::signal::SignalEncoder;
use cortiq_server::decisions::{DecisionServer, ServeOptions};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tower::ServiceExt;

const EPOCH: u64 = 1_790_000_000;
const TOPICS: [&str; 4] = ["Weather", "billing", "cards", "travel"];
const SHOP: [&str; 3] = ["billing", "cards", "food"];
const ADMIN: &str = "admin-token-for-cascade-tests-0123456789";
/// The fake oracle key (never a real one); tests look for its bytes.
const TEST_KEY: &str = "sk-or-v1-TESTKEY-wp7-http-0123456789abcdef-cmf";
const KEY_ENV: &str = "CMF_WP7_TEST_ORACLE_KEY";
const ORACLE_MODEL: &str = "deepseek/deepseek-v4.1-flash";

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
        "food" => &[
            "pizza", "salad", "bread", "cheese", "soup", "rice", "apple", "coffee",
        ],
        // Out of every training pool: the sub-topic the oracle teaches.
        "cruise" => &[
            "cruise", "ship", "cabin", "deck", "ocean", "port", "voyage", "sail", "yacht",
            "harbor", "ferry", "captain",
        ],
        _ => unreachable!("{label}"),
    }
}

const FILLER: [&str; 8] = [
    "please", "help", "my", "the", "today", "need", "about", "with",
];

fn synth(labels: &[&str], per_label: usize, seed: u64, tag: &str) -> Vec<(String, String)> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    for i in 0..per_label {
        for label in labels {
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

fn jsonl(rows: &[(String, String)]) -> String {
    rows.iter()
        .map(|(t, l)| json!({"text": t, "label": l}).to_string() + "\n")
        .collect()
}

fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

struct Toy {
    path: PathBuf,
    dev: Vec<(String, String)>,
}

fn skill_opts(
    dir: &Path,
    id: &str,
    labels: &[&str],
    seed: u64,
) -> (TrainOptions, Vec<(String, String)>) {
    let train = synth(labels, 30, seed, &format!("{id}t"));
    let cal = synth(labels, 80, seed + 1, &format!("{id}c"));
    let dev = synth(labels, 12, seed + 2, &format!("{id}d"));
    let mut crit = Map::new();
    for l in labels {
        crit.insert(l.to_string(), json!(format!("The message is about {l}.")));
    }
    let q = json!({"instructions": "Which topic is the message about?", "criteria": crit});
    let mut o = TrainOptions::new(
        id,
        vec![write(dir, &format!("{id}-train.jsonl"), &jsonl(&train))],
    );
    o.calibration = Some(write(dir, &format!("{id}-cal.jsonl"), &jsonl(&cal)));
    o.dev = Some(write(dir, &format!("{id}-dev.jsonl"), &jsonl(&dev)));
    o.question = Some(write(dir, &format!("{id}-q.json"), &q.to_string()));
    o.threads = 2;
    o.created_unix = Some(EPOCH);
    (o, dev)
}

fn toy() -> &'static Toy {
    static TOY: OnceLock<Toy> = OnceLock::new();
    TOY.get_or_init(|| {
        let dir = toy_dir::toy_dir("toy");
        let d = dir.as_path();
        let enc = d.join("enc.cmf");
        let export = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../cortiq-decision/tests/fixtures/toy/encoder");
        build::init_encoder(&export, &enc, Some(EPOCH)).expect("init toy encoder");
        let (o1, dev) = skill_opts(d, "topics", &TOPICS, 11);
        let s1 = d.join("s1.cmf");
        build::train(&enc, &o1, &s1).expect("train topics");
        let (o2, _) = skill_opts(d, "shop", &SHOP, 21);
        let path = d.join("toy.cmf");
        build::add_skill(&s1, &o2, &path).expect("add shop");
        Toy { path, dev }
    })
}

fn encoder() -> &'static SignalEncoder {
    static ENC: OnceLock<SignalEncoder> = OnceLock::new();
    ENC.get_or_init(|| {
        let m = DecisionModel::open(&toy().path, Verify::Light).unwrap();
        SignalEncoder::from_model(&m).unwrap().0
    })
}

fn phi_p(text: &str) -> Vec<f32> {
    encoder().features(text).phi_p
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    cortiq_decision::buffer::dot(a, b)
}

/// `count` texts of `label` words, each with cos φ_P < `max_cos` to every
/// earlier one (no cache hit among them).
fn distinct_texts(label: &str, count: usize, seed: u64, tag: &str, max_cos: f32) -> Vec<String> {
    let p = pool(label);
    let mut rng = Lcg(seed);
    let mut out: Vec<String> = Vec::new();
    let mut phis: Vec<Vec<f32>> = Vec::new();
    let mut tries = 0;
    while out.len() < count {
        tries += 1;
        assert!(
            tries < 100_000,
            "could not find {count} distinct '{label}' texts"
        );
        let mut words = Vec::new();
        for _ in 0..3 + rng.below(2) {
            words.push(p[rng.below(p.len())]);
        }
        words.push(FILLER[rng.below(FILLER.len())]);
        let t = format!("{} {tag}{}", words.join(" "), out.len());
        let f = phi_p(&t);
        if phis.iter().all(|q| cos(q, &f) < max_cos) {
            phis.push(f);
            out.push(t);
        }
    }
    out
}

/// Cruise texts the `topics` gate rejects, pairwise cos φ_P < 0.97.
fn rejected() -> &'static [String] {
    static R: OnceLock<Vec<String>> = OnceLock::new();
    R.get_or_init(|| distinct_texts("cruise", 12, 7, "q", 0.97))
}

/// 25 cruise texts to teach (pairwise cos φ_P < 0.97).
fn lesson() -> &'static [String] {
    static L: OnceLock<Vec<String>> = OnceLock::new();
    L.get_or_init(|| distinct_texts("cruise", 25, 7, "q", 0.97))
}

/// Cruise texts never taught.
fn fresh() -> &'static [String] {
    static F: OnceLock<Vec<String>> = OnceLock::new();
    F.get_or_init(|| distinct_texts("cruise", 8, 99, "z", 0.995))
}

fn choice(labels: &[&str]) -> Value {
    let mut c = Map::new();
    for l in labels {
        c.insert(l.to_string(), json!(format!("about {l}")));
    }
    json!({"type": "choice", "instructions": "Which topic?", "criteria": c})
}

fn body(state: Value, questions: Value, cmf: Option<Value>) -> Value {
    let mut v = json!({"model": "cortiq/decision", "state": state, "questions": questions});
    if let Some(c) = cmf {
        v["cmf"] = c;
    }
    v
}

fn topics_body(text: &str) -> Value {
    body(json!(text), json!({"task": choice(&TOPICS)}), None)
}

fn score_question() -> Value {
    json!({"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]})
}

fn untrained_body() -> Value {
    body(json!("x"), json!({"u": score_question()}), None)
}

fn sha256_hex(b: &[u8]) -> String {
    cortiq_decision::manifest::sha256_hex(b)
}

// ------------------------------------------------------------------ mock OpenRouter

#[derive(Clone, Debug)]
struct MockRequest {
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl MockRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("the body is JSON")
    }
    /// The state sent (the user message).
    fn state(&self) -> Value {
        let v = self.json();
        let user: Value =
            serde_json::from_str(v["messages"][1]["content"].as_str().unwrap()).unwrap();
        user["state"].clone()
    }
}

#[derive(Clone, Debug)]
struct MockReply {
    status: u16,
    body: Vec<u8>,
    delay: Duration,
}

type Handler = Arc<dyn Fn(&MockRequest) -> MockReply + Send + Sync>;

struct MockOracle {
    addr: SocketAddr,
    hits: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<MockRequest>>>,
    handler: Arc<Mutex<Handler>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn read_request(s: &mut TcpStream) -> Option<MockRequest> {
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
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let path = lines.next()?.split(' ').nth(1)?.to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let len: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
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
    Some(MockRequest {
        path,
        headers,
        body,
    })
}

impl MockOracle {
    fn start(handler: impl Fn(&MockRequest) -> MockReply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let handler: Arc<Mutex<Handler>> = Arc::new(Mutex::new(Arc::new(handler)));
        let stop = Arc::new(AtomicBool::new(false));
        let (h2, r2, hd2, st2) = (
            hits.clone(),
            requests.clone(),
            handler.clone(),
            stop.clone(),
        );
        let thread = std::thread::spawn(move || {
            for conn in listener.incoming() {
                if st2.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut s) = conn else { continue };
                let (h3, r3, hd3) = (h2.clone(), r2.clone(), hd2.clone());
                std::thread::spawn(move || {
                    let Some(req) = read_request(&mut s) else {
                        return;
                    };
                    h3.fetch_add(1, Ordering::SeqCst);
                    r3.lock().unwrap().push(req.clone());
                    let handler = hd3.lock().unwrap().clone();
                    let reply = handler(&req);
                    std::thread::sleep(reply.delay);
                    let head = format!(
                        "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        reply.status,
                        reply.body.len()
                    );
                    let _ = s.write_all(head.as_bytes());
                    let _ = s.write_all(&reply.body);
                    let _ = s.flush();
                });
            }
        });
        Self {
            addr,
            hits,
            requests,
            handler,
            stop,
            thread: Some(thread),
        }
    }

    /// Every choice answered with `label` when it is an option (else the
    /// first option), every score with 0 and every noul with true.
    fn answering(label: &str) -> Self {
        let label = label.to_string();
        Self::start(move |req| answer_reply(req, |_, opts| pick(opts, &label), 1.3e-5))
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    fn requests(&self) -> Vec<MockRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn set(&self, handler: impl Fn(&MockRequest) -> MockReply + Send + Sync + 'static) {
        *self.handler.lock().unwrap() = Arc::new(handler);
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

fn pick(options: &[String], label: &str) -> Value {
    if options.iter().any(|o| o == label) {
        json!(label)
    } else {
        json!(options[0])
    }
}

fn verdicts(req: &MockRequest, choose: impl Fn(&str, &[String]) -> Value) -> Value {
    let v = req.json();
    let schema = &v["response_format"]["json_schema"]["schema"];
    let mut out = Map::new();
    for q in schema["required"].as_array().unwrap() {
        let qid = q.as_str().unwrap();
        let p = &schema["properties"][qid];
        let a = match p["type"].as_str().unwrap() {
            "string" => {
                let opts: Vec<String> = p["enum"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap().to_string())
                    .collect();
                choose(qid, &opts)
            }
            "integer" => json!(0),
            _ => json!(true),
        };
        out.insert(qid.to_string(), a);
    }
    Value::Object(out)
}

fn completion(content: &str, cost: Value, model: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "id": "gen-mock", "object": "chat.completion", "model": model, "provider": "Mock",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 7, "total_tokens": 1207, "cost": cost,
                  "prompt_tokens_details": {"cached_tokens": 1024}},
    }))
    .unwrap()
}

fn answer_reply(
    req: &MockRequest,
    choose: impl Fn(&str, &[String]) -> Value,
    cost: f64,
) -> MockReply {
    let content = verdicts(req, choose).to_string();
    MockReply {
        status: 200,
        body: completion(&content, json!(cost), ORACLE_MODEL),
        delay: Duration::ZERO,
    }
}

fn raw_reply(status: u16, body: &str) -> MockReply {
    MockReply {
        status,
        body: body.as_bytes().to_vec(),
        delay: Duration::ZERO,
    }
}

// ------------------------------------------------------------------ server + HTTP

fn test_key() -> KeyLookup {
    Arc::new(|name: &str| (name == KEY_ENV).then(|| TEST_KEY.to_string()))
}

/// The oracle configuration of a stand: the mock, the test key's variable,
/// learning synchronous; every limit at its default.
fn stand_config(mock_url: &str) -> Config {
    let mut c = Config::default();
    // The open mode reaches the oracle and teaches only when configured
    // explicitly (a loopback address alone is not enough).
    c.auth.require = Some(false);
    c.oracle.enabled = true;
    c.oracle.base_url = mock_url.to_string();
    c.oracle.api_key_env = KEY_ENV.to_string();
    c.oracle.deadline_s = 5.0;
    c.learning.synchronous = true;
    c
}

struct Srv {
    server: Option<DecisionServer>,
    app: Option<axum::Router>,
    dir: tempfile::TempDir,
    base: PathBuf,
}

impl Srv {
    fn new(cfg: &Config) -> Self {
        Self::open_on(&toy().path, tempfile::tempdir().unwrap(), cfg, test_key())
    }

    fn open_on(base: &Path, dir: tempfile::TempDir, cfg: &Config, key: KeyLookup) -> Self {
        Self::open_with(base, dir, cfg, key, |_| {})
    }

    fn open_with(
        base: &Path,
        dir: tempfile::TempDir,
        cfg: &Config,
        key: KeyLookup,
        tweak: impl FnOnce(&mut ServeOptions),
    ) -> Self {
        let mut o = ServeOptions::new(base, cfg.clone());
        tweak(&mut o);
        o.state_dir = Some(dir.path().join("state"));
        o.cascade = CascadeOptions {
            key,
            threads: 2,
            created_unix: Some(EPOCH),
        };
        o.admin_token = Some(ADMIN.to_string());
        let server = DecisionServer::open(&o).expect("open the decision server");
        let app = server.router();
        Self {
            server: Some(server),
            app: Some(app),
            dir,
            base: base.to_path_buf(),
        }
    }

    /// Close and open again on the same state directory.
    fn restart(mut self, cfg: &Config) -> Self {
        self.app.take();
        self.server.take().unwrap().close().unwrap();
        let base = self.base.clone();
        let dir = std::mem::replace(&mut self.dir, tempfile::tempdir().unwrap());
        Self::open_on(&base, dir, cfg, test_key())
    }

    fn state_root(&self) -> PathBuf {
        self.dir.path().join("state")
    }

    async fn call(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: Option<Vec<u8>>,
    ) -> Resp {
        let mut h = headers.to_vec();
        h.push(("x-cmf-extensions", "1"));
        call(self.app.as_ref().unwrap(), method, path, &h, body).await
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

    async fn decide(&self, v: &Value) -> Resp {
        self.post("/v1/decisions", None, v).await
    }

    async fn get(&self, path: &str) -> Resp {
        self.call("GET", path, &[], None).await
    }

    async fn admin(&self, method: &str, path: &str, v: Option<&Value>) -> Resp {
        let mut h = vec![("x-admin-token", ADMIN)];
        if v.is_some() {
            h.push(("content-type", "application/json"));
        }
        self.call(method, path, &h, v.map(|v| serde_json::to_vec(v).unwrap()))
            .await
    }

    async fn learning(&self) -> Value {
        let r = self.admin("GET", "/v1/admin/learning", None).await;
        assert_eq!(r.status, 200, "{}", r.text);
        r.body
    }

    /// `oracle.state` (`null` when never written).
    fn oracle_state(&self) -> Value {
        std::fs::read(self.state_root().join("oracle.state"))
            .map(|b| serde_json::from_slice(&b).unwrap())
            .unwrap_or(Value::Null)
    }

    /// The oracle ledger lines.
    fn ledger(&self) -> Vec<Value> {
        std::fs::read_to_string(self.state_root().join("oracle.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
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
    fn request_id(&self) -> &str {
        self.headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id")
    }

    fn q(&self, id: &str) -> &Value {
        &self.body["cmf"]["questions"][id]
    }

    fn action(&self) -> &str {
        self.q("task")["action"].as_str().unwrap_or("-")
    }

    fn flags(&self) -> Value {
        self.q("task")["flags"].clone()
    }

    /// OpenRouter error: (status, reason).
    fn error(&self) -> (u16, String) {
        let e = &self.body["error"];
        assert_eq!(e["code"], json!(self.status), "{}", self.text);
        assert_eq!(e["metadata"]["request_id"], json!(self.request_id()));
        assert_eq!(
            e["metadata"]["retriable"],
            json!(matches!(self.status, 429 | 500 | 502))
        );
        (
            self.status,
            e["metadata"]["reason"].as_str().unwrap().to_string(),
        )
    }
}

async fn call(
    app: &axum::Router,
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
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    assert!(headers.contains_key("x-request-id"));
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();
    assert!(
        !text.contains(TEST_KEY),
        "the oracle key in a response to {path}"
    );
    // Correctly rounded floats (the fixtures' costs compare exactly).
    let body = cortiq_decision::canonical::parse(&bytes).unwrap_or(Value::Null);
    Resp {
        status,
        headers,
        body,
        text,
    }
}

fn metric(text: &str, name: &str) -> f64 {
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{name} ")))
        .unwrap_or_else(|| panic!("no metric {name}"))
        .trim()
        .parse()
        .unwrap()
}

/// A dev text the `topics` gate accepts.
async fn accepted(srv: &Srv) -> String {
    for (t, _) in &toy().dev {
        let mut b = topics_body(t);
        b["cmf"] = json!({"oracle": false});
        if srv.decide(&b).await.action() == "local" {
            return t.clone();
        }
    }
    panic!("no dev text the gate accepts")
}

// ------------------------------------------------------------------ cascade

#[tokio::test]
async fn abstain_goes_to_the_oracle_then_the_cache_over_both_apis() {
    let mock = MockOracle::answering("travel");
    let srv = Srv::new(&stand_config(&mock.url()));
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.action(), "oracle");
    assert_eq!(r.q("task")["source"], "oracle");
    assert_eq!(r.q("task")["certified"], false);
    assert_eq!(r.q("task")["gate"]["accepted"], false);
    assert_eq!(r.q("task")["decision_path"], "escalate→oracle");
    assert_eq!(
        r.body["answers"]["task"],
        json!({"type": "choice", "choice": "travel"})
    );
    let u = &r.body["cmf"]["usage"]["oracle"];
    assert_eq!(
        (
            u["calls"].as_u64(),
            u["input_tokens"].as_u64(),
            u["output_tokens"].as_u64()
        ),
        (Some(1), Some(1200), Some(7))
    );
    assert_eq!(u["cost"].as_f64(), Some(1.3e-5));
    assert_eq!(u["passthrough"], true);
    assert_eq!(
        r.body["usage"]["cost"].as_f64(),
        Some(1.3e-5),
        "passthrough cost"
    );
    assert_eq!(r.request_id(), r.body["id"].as_str().unwrap());
    assert_eq!(mock.hits(), 1);
    let sent = &mock.requests()[0];
    assert_eq!(sent.path, "/chat/completions");
    assert_eq!(
        sent.header("authorization"),
        Some(format!("Bearer {TEST_KEY}").as_str())
    );
    assert_eq!(sent.header("x-title"), Some("cortiq-decision"));
    assert_eq!(sent.state(), json!(rejected()[0]));
    let l = srv.ledger();
    assert_eq!(
        (l[0]["status"].as_str(), l[1]["status"].as_str()),
        (Some("reserved"), Some("settled"))
    );
    assert_eq!(l[1]["request_id"], r.body["id"]);

    // The router API: the oracle answer, promoted to the top of the scores.
    let rr = srv
        .post("/v1/route", None, &json!({"taxonomy_id": "topics", "input": {"text": rejected()[1]}, "options": {"return_explanation": true}}))
        .await;
    assert_eq!(rr.status, 200, "{}", rr.text);
    let d = &rr.body["decision"];
    assert_eq!(d["source"], "oracle");
    assert_eq!(d["task_label"], "travel");
    assert_eq!(d["task_id"], 3);
    assert_eq!(d["confidence"].as_f64(), Some(1.0));
    assert_eq!(
        (d["confident"].as_bool(), d["is_novel"].as_bool()),
        (Some(true), Some(false))
    );
    assert_eq!(rr.body["scores"][0]["task_label"], "travel");
    assert_eq!(rr.body["scores"][0]["probability"].as_f64(), Some(1.0));
    let o = &rr.body["oracle"];
    assert_eq!(o["consulted"], true);
    assert_eq!(o["model"], ORACLE_MODEL);
    assert!(o["agreement_with_router"].is_boolean());
    assert!(o["latency_ms"].is_number());
    assert_eq!(rr.body["usage"]["oracle_calls"], 1);
    assert_eq!(rr.body["explanation"]["decision_path"], "escalate→oracle");
    assert_eq!(mock.hits(), 2);
    // The same text again: the cache, no call.
    let rc = srv
        .post(
            "/v1/route",
            None,
            &json!({"taxonomy_id": "topics", "input": {"text": rejected()[1]}}),
        )
        .await;
    let d = &rc.body["decision"];
    assert_eq!(d["source"], "cache");
    assert_eq!(d["task_label"], "travel");
    assert_eq!(rc.body["oracle"]["consulted"], false);
    assert_eq!(rc.body["usage"]["oracle_calls"], 0);
    assert_eq!(mock.hits(), 2);

    // The audit and the metrics.
    let e = srv.get("/v1/escalations").await;
    let rec = e.body["records"].as_array().unwrap();
    assert_eq!(rec.len(), 3);
    assert_eq!(rec[0]["source"], "cache");
    assert_eq!(rec[0]["request_id"], rc.body["request_id"]);
    assert_eq!(rec[1]["source"], "oracle");
    assert_eq!(rec[1]["oracle_calls"], 1);
    assert_eq!(rec[1]["oracle_model"], ORACLE_MODEL);
    assert_eq!(rec[2]["request_id"], r.body["id"]);
    assert_eq!(e.body["summary"]["oracle_calls"], 2);
    assert_eq!(e.body["summary"]["cache_hits"], 1);
    let m = srv.get("/metrics").await.text;
    assert_eq!(metric(&m, "cortiq_decisions_total"), 3.0);
    assert_eq!(metric(&m, "cortiq_escalations_total"), 3.0);
    assert_eq!(metric(&m, "cortiq_oracle_calls_total"), 2.0);
    assert_eq!(metric(&m, "cortiq_cache_hits_total"), 1.0);
    assert!(metric(&m, "cortiq_oracle_cache_entries") >= 2.0);
    let u = srv.get("/v1/usage").await;
    assert_eq!(u.body["usage"]["oracle_calls"], 2);
    assert_eq!(u.body["usage"]["cache_hits"], 1);
}

#[tokio::test]
async fn gate_accepted_questions_never_reach_the_oracle() {
    let mock = MockOracle::answering("travel");
    let srv = Srv::new(&stand_config(&mock.url()));
    let text = accepted(&srv).await;
    for cmf in [None, Some(json!({"oracle": true}))] {
        let r = srv
            .decide(&body(json!(text), json!({"task": choice(&TOPICS)}), cmf))
            .await;
        assert_eq!(r.action(), "local");
    }
    let r = srv
        .post("/v1/route", None, &json!({"taxonomy_id": "topics", "input": {"text": text}, "options": {"allow_oracle": true}}))
        .await;
    assert_eq!(r.body["decision"]["source"], "router");
    assert!(r.body.get("oracle").is_none());
    assert_eq!(mock.hits(), 0);
    // An accepted and an untrained question: the call carries the second only.
    let r = srv
        .decide(&body(
            json!(text),
            json!({"a": choice(&TOPICS), "b": score_question()}),
            Some(json!({"oracle": true})),
        ))
        .await;
    assert_eq!(r.q("a")["action"], "local");
    assert_eq!(r.q("b")["action"], "oracle");
    assert_eq!(mock.hits(), 1);
    let sent = mock.requests()[0].json();
    assert_eq!(
        sent["response_format"]["json_schema"]["schema"]["required"],
        json!(["b"])
    );
}

#[tokio::test]
async fn a_repeat_and_a_paraphrase_are_cache_answers_without_calls() {
    let a = "cruise ship cabin deck please";
    let p = "cruise ship cabin today";
    let c = cos(&phi_p(a), &phi_p(p));
    assert!((0.97..1.0).contains(&c), "paraphrase cos {c}");
    let mock = MockOracle::answering("travel");
    let srv = Srv::new(&stand_config(&mock.url()));
    assert_eq!(srv.decide(&topics_body(a)).await.action(), "oracle");
    let r2 = srv.decide(&topics_body(a)).await;
    assert_eq!(r2.action(), "cache");
    assert_eq!(r2.q("task")["decision_path"], "escalate→cache");
    assert_eq!(r2.body["answers"]["task"]["choice"], "travel");
    assert_eq!(r2.body["usage"]["cost"].as_f64(), Some(0.0));
    assert_eq!(r2.body["cmf"]["usage"]["oracle"]["calls"], 0);
    let r3 = srv.decide(&topics_body(p)).await;
    assert_eq!(r3.action(), "cache", "paraphrase");
    assert_eq!(mock.hits(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_parallel_identical_http_requests_make_one_call() {
    let mock = MockOracle::start(|req| {
        let mut r = answer_reply(req, |_, o| pick(o, "travel"), 1.3e-5);
        r.delay = Duration::from_millis(800);
        r
    });
    let srv = Srv::new(&stand_config(&mock.url()));
    let b = topics_body(&rejected()[2]);
    let (r1, r2) = tokio::join!(srv.decide(&b), async {
        tokio::time::sleep(Duration::from_millis(250)).await;
        srv.decide(&b).await
    });
    assert_eq!(mock.hits(), 1, "one call for two identical requests");
    let mut actions = vec![r1.action().to_string(), r2.action().to_string()];
    actions.sort();
    assert_eq!(actions, ["cache", "oracle"]);
    assert_eq!(r1.body["answers"]["task"]["choice"], "travel");
    assert_eq!(r2.body["answers"]["task"]["choice"], "travel");
}

// ------------------------------------------------------------------ learning

/// (label → (mean sha, basis sha, state)) of a skill in `/v1/admin/learning`.
fn task_shas(l: &Value, skill: &str) -> Vec<(String, Value)> {
    l["skills"][skill]["tasks"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                json!([v["mean_sha256"], v["basis_sha256"], v["state"], v["k"]]),
            )
        })
        .collect()
}

#[tokio::test]
async fn twenty_five_answers_promote_isolate_roll_back_and_survive_restarts() {
    let mock = MockOracle::answering("travel");
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let before = srv.learning().await;
    for t in fresh() {
        let mut b = topics_body(t);
        b["cmf"] = json!({"oracle": false});
        assert_ne!(srv.decide(&b).await.action(), "local", "{t}");
    }

    for t in lesson() {
        let r = srv.decide(&topics_body(t)).await;
        assert_eq!(r.action(), "oracle", "{t}");
    }
    assert_eq!(mock.hits(), 25);
    let l = srv.learning().await;
    assert_eq!(
        (l["attempts"].as_u64(), l["promotions"].as_u64()),
        (Some(1), Some(1))
    );
    assert_eq!(l["isolation_violations"], 0);
    assert_eq!(l["recent"][0]["outcome"], "promoted");
    assert_eq!(l["recent"][0]["holdout"]["passed"], true);
    assert_eq!(l["generation"], 1);
    // Isolation: only topics/travel changed; shop is untouched.
    let (a, b) = (task_shas(&before, "topics"), task_shas(&l, "topics"));
    let changed: Vec<&str> = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x != y)
        .map(|(x, _)| x.0.as_str())
        .collect();
    assert_eq!(changed, vec!["travel"]);
    assert_eq!(task_shas(&before, "shop"), task_shas(&l, "shop"));

    // Generation 1 is served, listed and written.
    let m = srv.get("/v1/models").await;
    assert_eq!(m.body["data"][0]["cmf"]["generation"], 1);
    let g = srv.admin("GET", "/v1/admin/generations", None).await;
    assert_eq!(g.body["current"], 1);
    assert_eq!(g.body["generations"].as_array().unwrap().len(), 1);
    assert!(srv.state_root().join("generations/g000001.cmf").exists());
    let metrics = srv.get("/metrics").await.text;
    assert_eq!(metric(&metrics, "cortiq_promotions_total"), 1.0);
    assert_eq!(metric(&metrics, "cortiq_refits_total"), 1.0);
    // The next similar requests are local.
    let hits = mock.hits();
    for t in fresh() {
        let r = srv.decide(&topics_body(t)).await;
        assert_eq!(r.action(), "local", "{t}");
        assert_eq!(r.body["answers"]["task"]["choice"], "travel");
        assert_eq!(r.body["cmf"]["generation"], 1);
    }
    assert_eq!(mock.hits(), hits);
    let tx = srv.get("/v1/taxonomies/topics").await;
    assert_eq!(tx.body["taxonomy_version"], "topics@1");
    assert_eq!(tx.body["cmf"]["generation"], 1);

    // Rollback to the base over HTTP.
    let r = srv
        .admin(
            "POST",
            "/v1/admin/rollback",
            Some(&json!({"generation": 0})),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["generation"], 0);
    assert_eq!(
        srv.get("/v1/models").await.body["data"][0]["cmf"]["generation"],
        0
    );
    let mut b = topics_body(&fresh()[0]);
    b["cmf"] = json!({"oracle": false});
    assert_ne!(srv.decide(&b).await.action(), "local");

    // Restart: generation 0 stays served; the cache comes back from learn.log.
    let srv = srv.restart(&cfg);
    assert_eq!(
        srv.get("/v1/models").await.body["data"][0]["cmf"]["generation"],
        0
    );
    let hits = mock.hits();
    let r = srv.decide(&topics_body(&lesson()[0])).await;
    assert_eq!(r.action(), "cache", "the cache survived the restart");
    assert_eq!(mock.hits(), hits);
    // Forward to generation 1 again, restart, still generation 1.
    let r = srv
        .admin(
            "POST",
            "/v1/admin/rollback",
            Some(&json!({"generation": 1})),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let srv = srv.restart(&cfg);
    assert_eq!(
        srv.get("/v1/models").await.body["data"][0]["cmf"]["generation"],
        1
    );
    let r = srv.decide(&topics_body(&fresh()[1])).await;
    assert_eq!(r.action(), "local");
    let l = srv.learning().await;
    assert_eq!(l["generation"], 1);
    assert!(
        l["buffer"]["examples"].as_u64().unwrap() >= 25,
        "the buffer is kept"
    );
}

#[tokio::test]
async fn a_holdout_regression_from_wrong_feedback_is_refused() {
    let mock = MockOracle::answering("travel");
    let srv = Srv::new(&stand_config(&mock.url()));
    let k = |r: &Resp| r.body["key"].as_str().unwrap().to_string();
    // No rate window (the starter plan's 60 a minute is below this test's pace).
    let owner = k(&srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"account": "owner", "rate_per_min": 0, "learning_allowed": true})),
        )
        .await);
    let other = k(&srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"account": "other", "rate_per_min": 0})),
        )
        .await);
    let sha0 = srv.get("/v1/models").await.body["data"][0]["cmf"]["model_sha"].clone();
    let bills = distinct_texts("billing", 40, 5, "b", 0.995);
    let mut ids = Vec::new();
    for t in &bills {
        let r = srv
            .post("/v1/decisions", Some(&owner), &topics_body(t))
            .await;
        assert_eq!(r.status, 200, "{}", r.text);
        if r.action() == "local" {
            ids.push(r.body["id"].as_str().unwrap().to_string());
        }
    }
    assert!(ids.len() >= 25, "{} billing texts accepted", ids.len());
    let fb = |id: &str, label: &str| json!({"id": id, "question": "task", "label": label});
    // Scoped to the account; the label must be an option; unknown ids are 404.
    let r = srv
        .post("/v1/feedback", Some(&other), &fb(&ids[0], "cards"))
        .await;
    assert_eq!(r.status, 404);
    assert_eq!(r.body["error"]["code"], "INVALID_REQUEST");
    let r = srv
        .post("/v1/feedback", Some(&owner), &fb(&ids[0], "food"))
        .await;
    assert_eq!(r.status, 400);
    let r = srv
        .post("/v1/feedback", Some(&owner), &fb("cmf-dec-1-x", "cards"))
        .await;
    assert_eq!(r.status, 404);
    let mut last = Value::Null;
    for (n, id) in ids.iter().take(25).enumerate() {
        let r = srv
            .post("/v1/feedback", Some(&owner), &fb(id, "cards"))
            .await;
        assert_eq!(r.status, 200, "feedback {n}: {}", r.text);
        assert_eq!(r.body["accepted"], true);
        assert_eq!(r.body["weight"], 3.0);
        last = r.body;
    }
    let l = &last["learning"];
    assert_eq!(l["outcome"], "rejected");
    assert_eq!(l["reason"], "holdout_regression");
    assert_eq!(l["holdout"]["passed"], false);
    assert_eq!(
        srv.get("/v1/models").await.body["data"][0]["cmf"]["model_sha"],
        sha0
    );
    let g = srv.admin("GET", "/v1/admin/generations", None).await;
    assert_eq!(g.body["generations"], json!([]));
    assert_eq!(srv.learning().await["rejections"], 1);
    assert_eq!(mock.hits(), 0);
}

#[tokio::test]
async fn cold_start_turns_an_oracle_label_into_a_task_after_25_answers() {
    let mock = MockOracle::answering("cruise");
    let srv = Srv::new(&stand_config(&mock.url()));
    let mut l5 = TOPICS.to_vec();
    l5.push("cruise");
    for (n, t) in lesson().iter().enumerate() {
        let r = srv
            .decide(&body(json!(t), json!({"task": choice(&l5)}), None))
            .await;
        assert_eq!(r.status, 200, "{}", r.text);
        assert_eq!(r.action(), "oracle");
        assert_eq!(r.q("task")["match"], "superset");
        if n == 23 {
            let l = srv.learning().await;
            assert_eq!(
                l["quarantine"],
                json!([{"skill": "topics", "label": "cruise", "examples": 24, "new": 24}])
            );
        }
    }
    let l = srv.learning().await;
    assert_eq!(
        (l["promotions"].as_u64(), l["cold_starts"].as_u64()),
        (Some(1), Some(1))
    );
    assert_eq!(l["quarantine"], json!([]));
    assert_eq!(l["isolation_violations"], 0);
    let s = srv.get("/v1/skills/topics").await;
    assert_eq!(
        s.body["labels"],
        json!(["Weather", "billing", "cards", "travel", "cruise"])
    );
    assert_eq!(s.body["taxonomy_version"], 2);
    let t = &s.body["tasks"][4];
    assert_eq!(
        (t["origin"].as_str(), t["state"].as_str()),
        (Some("cold_start"), Some("active"))
    );
    let tx = srv.get("/v1/taxonomies/topics").await;
    assert_eq!(tx.body["taxonomy_version"], "topics@2");
    // Now an exact question: decided locally, never certified.
    let hits = mock.hits();
    for t in fresh() {
        let r = srv
            .decide(&body(json!(t), json!({"task": choice(&l5)}), None))
            .await;
        assert_eq!(r.q("task")["match"], "exact");
        assert_eq!(r.action(), "local", "{t}");
        assert_eq!(r.body["answers"]["task"]["choice"], "cruise");
        assert_eq!(r.q("task")["certified"], false);
    }
    assert_eq!(mock.hits(), hits);
    // The router API sees the new label too.
    let rr = srv
        .post(
            "/v1/route",
            None,
            &json!({"taxonomy_id": "topics", "input": {"text": fresh()[0]}}),
        )
        .await;
    assert_eq!(rr.body["decision"]["task_label"], "cruise");
    assert_eq!(rr.body["decision"]["task_id"], 4);
    assert_eq!(rr.body["cmf"]["certified"], false);
}

// ------------------------------------------------------------------ budget and stop rules

async fn enable(srv: &Srv) {
    let r = srv
        .admin("POST", "/v1/admin/oracle", Some(&json!({"enabled": true})))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
}

#[tokio::test]
async fn budget_max_calls_and_stop_rules_over_http() {
    // A reservation above the budget left: no call.
    let mock = MockOracle::answering("travel");
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.budget_usd = 1e-4;
    let srv = Srv::new(&cfg);
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.action(), "abstain");
    assert_eq!(r.flags(), json!(["budget"]));
    assert_eq!(r.q("task")["decision_path"], "escalate→disabled");
    assert_eq!(
        srv.decide(&untrained_body()).await.error(),
        (503, "ORACLE_BUDGET_EXHAUSTED".to_string())
    );
    assert_eq!(mock.hits(), 0);
    assert!(srv.ledger().is_empty());

    // max_calls.
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.max_calls = 1;
    let srv = Srv::new(&cfg);
    assert_eq!(
        srv.decide(&topics_body(&rejected()[0])).await.action(),
        "oracle"
    );
    assert_eq!(
        srv.decide(&topics_body(&rejected()[1])).await.flags(),
        json!(["budget"])
    );
    assert_eq!(mock.hits(), 1);

    // HTTP 401/402/403 stop the oracle until the admin enables it.
    for status in [401u16, 402, 403] {
        let mock = MockOracle::start(move |_| raw_reply(status, r#"{"error":{"message":"no"}}"#));
        let srv = Srv::new(&stand_config(&mock.url()));
        let r = srv.decide(&topics_body(&rejected()[0])).await;
        assert_eq!(r.action(), "abstain");
        assert_eq!(r.flags(), json!(["oracle_unavailable"]));
        assert_eq!(r.q("task")["decision_path"], "escalate→oracle_unavailable");
        assert_eq!(
            srv.oracle_state()["stop_reason"],
            json!(format!("http_{status}"))
        );
        assert_eq!(
            srv.decide(&topics_body(&rejected()[1])).await.flags(),
            json!(["stopped"])
        );
        assert_eq!(
            srv.decide(&untrained_body()).await.error(),
            (503, "ORACLE_DISABLED".to_string())
        );
        let s = srv.admin("GET", "/v1/admin/oracle", None).await;
        assert_eq!(s.body["stop_reason"], json!(format!("http_{status}")));
        assert_eq!(mock.hits(), 1, "stopped after {status}");
        enable(&srv).await;
        assert_eq!(srv.oracle_state()["stop_reason"], Value::Null);
        mock.set(|req| answer_reply(req, |_, o| pick(o, "travel"), 1e-5));
        assert_eq!(
            srv.decide(&topics_body(&rejected()[1])).await.action(),
            "oracle"
        );
        assert_eq!(mock.hits(), 2);
    }

    // unexpected_model: the answer is not used, the oracle stops.
    let mock = MockOracle::start(|req| {
        let content = verdicts(req, |_, o| pick(o, "travel")).to_string();
        MockReply {
            status: 200,
            body: completion(&content, json!(1e-5), "openai/other-model"),
            delay: Duration::ZERO,
        }
    });
    let srv = Srv::new(&stand_config(&mock.url()));
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.flags(), json!(["oracle_unavailable"]));
    assert_eq!(
        r.body["usage"]["cost"].as_f64(),
        Some(0.0),
        "a failed call is not billed"
    );
    assert_eq!(srv.oracle_state()["stop_reason"], "unexpected_model");
    assert_eq!(
        srv.decide(&topics_body(&rejected()[1])).await.flags(),
        json!(["stopped"])
    );
    assert_eq!(mock.hits(), 1);

    // A cost above the reservation: the answer is used, then the oracle stops.
    let mock = MockOracle::start(|req| answer_reply(req, |_, o| pick(o, "travel"), 0.5));
    let srv = Srv::new(&stand_config(&mock.url()));
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.action(), "oracle");
    assert_eq!(r.body["usage"]["cost"].as_f64(), Some(0.5));
    assert_eq!(srv.oracle_state()["stop_reason"], "cost_above_reservation");
    assert_eq!(
        srv.decide(&topics_body(&rejected()[1])).await.flags(),
        json!(["stopped"])
    );
    assert_eq!(mock.hits(), 1);

    // max_errors failures in a row.
    let fail = Arc::new(AtomicBool::new(true));
    let f2 = fail.clone();
    let mock = MockOracle::start(move |req| {
        if f2.load(Ordering::SeqCst) {
            raw_reply(500, r#"{"error":{"message":"upstream"}}"#)
        } else {
            answer_reply(req, |_, o| pick(o, "travel"), 1e-5)
        }
    });
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.max_errors = 2;
    let srv = Srv::new(&cfg);
    let r = rejected();
    assert_eq!(
        srv.decide(&topics_body(&r[0])).await.flags(),
        json!(["oracle_unavailable"])
    );
    fail.store(false, Ordering::SeqCst);
    assert_eq!(srv.decide(&topics_body(&r[1])).await.action(), "oracle");
    fail.store(true, Ordering::SeqCst);
    assert_eq!(
        srv.decide(&topics_body(&r[2])).await.flags(),
        json!(["oracle_unavailable"])
    );
    assert_eq!(
        srv.oracle_state(),
        Value::Null,
        "one failure after a success does not stop"
    );
    assert_eq!(
        srv.decide(&topics_body(&r[3])).await.flags(),
        json!(["oracle_unavailable"])
    );
    assert_eq!(srv.oracle_state()["stop_reason"], "max_errors");
    assert_eq!(
        srv.decide(&topics_body(&r[4])).await.flags(),
        json!(["stopped"])
    );
    assert_eq!(mock.hits(), 4);
}

// ------------------------------------------------------------------ bad answers

#[tokio::test]
async fn bad_answers_and_the_deadline_are_failures() {
    // 200 whose body is only an error.
    let mock = MockOracle::start(|_| {
        raw_reply(200, r#"{"error":{"code":502,"message":"provider down"}}"#)
    });
    let srv = Srv::new(&stand_config(&mock.url()));
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.action(), "abstain");
    assert_eq!(r.flags(), json!(["oracle_unavailable"]));
    // The abstained answer is the local one (valid distribution).
    assert!(r.body["answers"]["task"]["probabilities"].is_object());
    assert_eq!(srv.ledger()[1]["error"], "error_body");
    let e = srv.decide(&untrained_body()).await;
    assert_eq!(e.error(), (502, "ORACLE_UNAVAILABLE".to_string()));
    let d = &e.body["error"]["metadata"]["details"]["questions"]["u"];
    assert_eq!(d["oracle"], "the oracle call failed");

    // Invalid JSON content: a billed failure, not billed to the client.
    let mock = MockOracle::start(|_| MockReply {
        status: 200,
        body: completion("{\"task\": \"trav", json!(2e-5), ORACLE_MODEL),
        delay: Duration::ZERO,
    });
    let srv = Srv::new(&stand_config(&mock.url()));
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.flags(), json!(["oracle_unavailable"]));
    assert_eq!(r.body["usage"]["cost"].as_f64(), Some(0.0));
    assert_eq!(srv.ledger()[1]["error"], "invalid_json");

    // The deadline bounds a call.
    let mock = MockOracle::start(|req| {
        let mut r = answer_reply(req, |_, o| pick(o, "travel"), 1e-5);
        r.delay = Duration::from_secs(3);
        r
    });
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.deadline_s = 0.5;
    let srv = Srv::new(&cfg);
    let t0 = Instant::now();
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    let took = t0.elapsed();
    assert!(took < Duration::from_millis(2500), "the call took {took:?}");
    assert_eq!(r.flags(), json!(["oracle_unavailable"]));
    assert_eq!(srv.decide(&untrained_body()).await.error().0, 502);
    // The router API degrades like the router: low confidence, no error.
    let rr = srv
        .post(
            "/v1/route",
            None,
            &json!({"taxonomy_id": "topics", "input": {"text": rejected()[3]}}),
        )
        .await;
    assert_eq!(rr.status, 200);
    assert_eq!(
        rr.body["decision"]["flags"],
        json!(["low_confidence", "oracle_unavailable"])
    );
    assert_eq!(rr.body["decision"]["confident"], false);
    assert_eq!(rr.body["oracle"]["consulted"], false);
}

// ------------------------------------------------------------------ consent

#[tokio::test]
async fn consent_switches_make_no_call() {
    let mock = MockOracle::answering("travel");
    let trained =
        |cmf: Option<Value>| body(json!(rejected()[0]), json!({"task": choice(&TOPICS)}), cmf);

    let mut off = stand_config(&mock.url());
    off.oracle.enabled = false;
    let srv = Srv::new(&off);
    assert_eq!(
        srv.decide(&trained(None)).await.flags(),
        json!(["oracle_disabled"])
    );
    assert_eq!(
        srv.decide(&untrained_body()).await.error(),
        (422, "UNSUPPORTED_QUESTION".to_string())
    );

    let srv = Srv::new(&stand_config(&mock.url()));
    assert_eq!(
        srv.decide(&trained(Some(json!({"oracle": false}))))
            .await
            .flags(),
        json!(["consent_off"])
    );
    let mut u = untrained_body();
    u["cmf"] = json!({"oracle": false});
    assert_eq!(
        srv.decide(&u).await.error(),
        (422, "UNSUPPORTED_QUESTION".to_string())
    );
    // A key without oracle_allowed (the default of a new key), then with it.
    let k = |r: &Resp| r.body["key"].as_str().unwrap().to_string();
    let plain = k(&srv
        .admin("POST", "/v1/admin/keys", Some(&json!({"account": "plain"})))
        .await);
    let r = srv
        .post("/v1/decisions", Some(&plain), &trained(None))
        .await;
    assert_eq!(r.flags(), json!(["consent_off"]));
    let rr = srv
        .post(
            "/v1/route",
            Some(&plain),
            &json!({"taxonomy_id": "topics", "input": {"text": rejected()[0]}}),
        )
        .await;
    assert_eq!(
        rr.body["decision"]["flags"],
        json!(["low_confidence", "consent_off"])
    );
    let allowed = k(&srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"account": "allowed", "oracle_allowed": true})),
        )
        .await);
    let rr = srv
        .post("/v1/route", Some(&allowed), &json!({"taxonomy_id": "topics", "input": {"text": rejected()[0]}, "options": {"allow_oracle": false}}))
        .await;
    assert_eq!(
        rr.body["decision"]["flags"],
        json!(["low_confidence", "consent_off"])
    );
    assert_eq!(mock.hits(), 0);
    let r = srv
        .post("/v1/decisions", Some(&allowed), &trained(None))
        .await;
    assert_eq!(r.action(), "oracle");
    assert_eq!(mock.hits(), 1);
    // The admin switch.
    let s = srv
        .admin("POST", "/v1/admin/oracle", Some(&json!({"enabled": false})))
        .await;
    assert_eq!(s.status, 200, "{}", s.text);
    let r = srv
        .post(
            "/v1/decisions",
            Some(&allowed),
            &body(json!(rejected()[1]), json!({"task": choice(&TOPICS)}), None),
        )
        .await;
    assert_eq!(r.flags(), json!(["oracle_disabled"]));
    assert_eq!(mock.hits(), 1);
}

// ------------------------------------------------------------------ PII

#[tokio::test]
async fn pii_is_redacted_by_default() {
    let text = "cruise ship yacht harbor mail john.doe@example.com call +15551234567";
    let mock = MockOracle::answering("travel");
    let srv = Srv::new(&stand_config(&mock.url()));
    let r = srv.decide(&topics_body(text)).await;
    assert_eq!(r.action(), "oracle");
    assert_eq!(r.flags(), json!(["pii_redacted"]));
    assert_eq!(
        mock.requests()[0].state(),
        json!("cruise ship yacht harbor mail [REDACTED] call [REDACTED]")
    );
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    let srv = Srv::new(&cfg);
    let r = srv
        .decide(&body(
            json!(text),
            json!({"task": choice(&TOPICS)}),
            Some(json!({"allow_pii_egress": true})),
        ))
        .await;
    assert_eq!(r.flags(), json!([]));
    assert_eq!(mock.requests()[1].state(), json!(text));
    // The router API's allow_pii_egress (default false).
    let rr = srv
        .post(
            "/v1/route",
            None,
            &json!({"taxonomy_id": "topics", "input": {"text": text}}),
        )
        .await;
    assert_eq!(rr.body["decision"]["flags"], json!(["pii_redacted"]));
    assert!(!String::from_utf8_lossy(&mock.requests()[2].body).contains("john.doe"));
}

// ------------------------------------------------------------------ driver bodies

#[tokio::test]
async fn one_choice_bodies_on_the_wire_equal_the_driver_for_nine_ledger_calls() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../cortiq-decision/tests/fixtures/oracle/deepseek_bodies.json");
    let fx: Value = cortiq_decision::canonical::parse(
        &std::fs::read(&fixtures).unwrap_or_else(|e| panic!("{}: {e}", fixtures.display())),
    )
    .unwrap();
    let mut recorded: HashMap<String, Value> = HashMap::new();
    let mut cases = Vec::new();
    for d in fx["datasets"].as_object().unwrap().values() {
        for r in d["rows"].as_array().unwrap() {
            recorded.insert(
                r["request_sha256"].as_str().unwrap().into(),
                r["oracle"].clone(),
            );
            cases.push((d["question"].clone(), r.clone()));
        }
    }
    assert_eq!(cases.len(), 9);
    // A ledger proxy: the recorded answer for the body's sha256, else 500.
    let mock = MockOracle::start(move |req| {
        let sha = sha256_hex(&req.body);
        let Some(o) = recorded.get(&sha) else {
            return raw_reply(500, r#"{"error":{"message":"unknown body"}}"#);
        };
        let u = &o["usage"];
        let body = serde_json::to_vec(&json!({
            "id": o["id"], "model": o["returned_model"], "provider": o["returned_provider"],
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                          "content": json!({"task": o["choice"]}).to_string()}}],
            "usage": {"prompt_tokens": u["input_tokens"], "completion_tokens": u["output_tokens"],
                      "cost": u["cost"], "prompt_tokens_details": {"cached_tokens": u["cached_tokens"]}},
        }))
        .unwrap();
        MockReply {
            status: 200,
            body,
            delay: Duration::ZERO,
        }
    });
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    let srv = Srv::new(&cfg);
    for (i, (q, row)) in cases.iter().enumerate() {
        let mut question = q.as_object().unwrap().clone();
        question.insert("type".into(), json!("choice"));
        let r = srv
            .decide(&body(row["text"].clone(), json!({"task": question}), None))
            .await;
        assert_eq!(r.status, 200, "row {i}: {}", r.text);
        assert_eq!(r.action(), "oracle", "row {i}");
        assert_eq!(
            r.body["answers"]["task"],
            json!({"type": "choice", "choice": row["oracle"]["choice"]})
        );
        assert_eq!(
            r.body["usage"]["cost"].as_f64(),
            row["oracle"]["usage"]["cost"].as_f64()
        );
        let sent = &mock.requests()[i];
        assert_eq!(
            sha256_hex(&sent.body),
            row["request_sha256"].as_str().unwrap(),
            "row {i}"
        );
    }
    assert_eq!(mock.hits(), 9);
}

// ------------------------------------------------------------------ the key (child process)

mod capture {
    //! A process-wide tracing subscriber that keeps every event field.
    use std::fmt::Write as _;
    use std::sync::Mutex;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Metadata, Subscriber};

    pub static LOG: Mutex<String> = Mutex::new(String::new());

    struct V;
    impl Visit for V {
        fn record_debug(&mut self, f: &Field, v: &dyn std::fmt::Debug) {
            let _ = write!(LOG.lock().unwrap(), "{}={:?} ", f.name(), v);
        }
    }

    pub struct Capture;
    impl Subscriber for Capture {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, a: &Attributes<'_>) -> Id {
            a.record(&mut V);
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, r: &Record<'_>) {
            r.record(&mut V);
        }
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, e: &Event<'_>) {
            let _ = write!(LOG.lock().unwrap(), "\n[{}] ", e.metadata().level());
            e.record(&mut V);
        }
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }
}

fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Runs in a child process with the key in its environment (see
/// `the_oracle_key_comes_only_from_the_environment`).
#[test]
#[ignore = "child process of the_oracle_key_comes_only_from_the_environment"]
fn child_oracle_key_from_the_environment() {
    let Ok(key) = std::env::var(KEY_ENV) else {
        println!("CHILD SKIPPED: {KEY_ENV} not set");
        return;
    };
    tracing::subscriber::set_global_default(capture::Capture).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let flip = Arc::new(AtomicUsize::new(0));
    let f2 = flip.clone();
    let mock = MockOracle::start(move |req| match f2.fetch_add(1, Ordering::SeqCst) {
        0 => answer_reply(req, |_, o| pick(o, "travel"), 1e-5),
        1 => raw_reply(500, r#"{"error":{"message":"boom"}}"#),
        _ => raw_reply(401, r#"{"error":{"message":"bad key"}}"#),
    });
    let (texts, srv) = rt.block_on(async {
        // The server's default key lookup: the process environment.
        let srv = Srv::open_on(
            &toy().path,
            tempfile::tempdir().unwrap(),
            &stand_config(&mock.url()),
            process_env(),
        );
        let r = rejected();
        let mut texts = Vec::new();
        let a = srv.decide(&topics_body(&r[0])).await;
        assert_eq!(a.action(), "oracle");
        assert_eq!(
            srv.decide(&topics_body(&r[1])).await.flags(),
            json!(["oracle_unavailable"])
        );
        assert_eq!(
            srv.decide(&topics_body(&r[2])).await.flags(),
            json!(["oracle_unavailable"])
        );
        let rr = srv
            .post(
                "/v1/route",
                None,
                &json!({"taxonomy_id": "topics", "input": {"text": r[3]}}),
            )
            .await;
        assert_eq!(
            rr.body["decision"]["flags"],
            json!(["low_confidence", "stopped"])
        );
        texts.push(a.text);
        texts.push(rr.text);
        for p in [
            "/v1/admin/oracle",
            "/v1/admin/learning",
            "/v1/admin/generations",
            "/v1/admin/usage",
        ] {
            texts.push(srv.admin("GET", p, None).await.text);
        }
        for p in [
            "/metrics",
            "/v1/escalations",
            "/v1/usage",
            "/v1/models",
            "/healthz",
        ] {
            texts.push(srv.get(p).await.text);
        }
        let status = srv.admin("GET", "/v1/admin/oracle", None).await;
        assert_eq!(status.body["key_present"], true);
        assert_eq!(status.body["stop_reason"], "http_401");
        // A restart re-reads the ledgers and the stop from the state directory.
        let srv = srv.restart(&stand_config(&mock.url()));
        let status = srv.admin("GET", "/v1/admin/oracle", None).await;
        assert_eq!(status.body["stop_reason"], "http_401");
        texts.push(status.text);
        (texts, srv)
    });
    let root = srv.state_root();
    for req in mock.requests() {
        assert_eq!(
            req.header("authorization"),
            Some(format!("Bearer {key}").as_str())
        );
    }
    for t in &texts {
        assert!(!t.contains(&key), "the key is in a response");
    }
    let files = files_under(&root);
    assert!(files.iter().any(|f| f.ends_with("oracle.jsonl")));
    for f in &files {
        let b = std::fs::read(f).unwrap();
        assert!(
            !b.windows(key.len()).any(|w| w == key.as_bytes()),
            "the key is in {}",
            f.display()
        );
    }
    let log = capture::LOG.lock().unwrap().clone();
    assert!(
        log.contains("oracle stopped"),
        "the logs were captured: {log}"
    );
    assert!(
        log.contains("decision request"),
        "request lines were captured"
    );
    assert!(!log.contains(&key), "the key is in the logs");
    for t in rejected().iter().take(4) {
        assert!(!log.contains(t.as_str()), "state text in the logs");
    }
    println!("CHILD OK {} files scanned", files.len());
    drop(srv);
}

#[test]
fn the_oracle_key_comes_only_from_the_environment() {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args([
            "--ignored",
            "--exact",
            "child_oracle_key_from_the_environment",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(toy_dir::TOY_CHILD_ENV, "1")
        .env(KEY_ENV, TEST_KEY)
        .env("CMF_GPU", "0")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "child failed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("CHILD OK"), "{stdout}\n{stderr}");
    assert!(
        !stdout.contains(TEST_KEY) && !stderr.contains(TEST_KEY),
        "the key leaked into the output"
    );
    // Without the variable the oracle is disabled: the key is read from nowhere
    // else, the configuration included.
    assert!(std::env::var(KEY_ENV).is_err());
    let mock = MockOracle::answering("travel");
    let cfg = stand_config(&mock.url());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let srv = Srv::open_on(
            &toy().path,
            tempfile::tempdir().unwrap(),
            &cfg,
            process_env(),
        );
        let r = srv.decide(&topics_body(&rejected()[0])).await;
        assert_eq!(r.flags(), json!(["oracle_disabled", "no_key"]));
        assert_eq!(
            srv.decide(&untrained_body()).await.error(),
            (422, "UNSUPPORTED_QUESTION".to_string())
        );
        let s = srv.admin("GET", "/v1/admin/oracle", None).await;
        assert_eq!(s.body["key_present"], false);
        assert_eq!(s.body["status"], "no_key");
    });
    assert_eq!(mock.hits(), 0);
    assert!(!cfg.to_value().to_string().contains(TEST_KEY));
}

// ------------------------------------------------------------------ the oracle in two steps

/// `status` of `GET /v1/admin/oracle`.
async fn status(srv: &Srv) -> String {
    let s = srv.admin("GET", "/v1/admin/oracle", None).await;
    assert_eq!(s.status, 200, "{}", s.text);
    s.body["status"].as_str().unwrap().to_string()
}

/// `serve --oracle` on loopback without keys or `auth.require`: the open mode
/// may use the oracle (never teach); `status` and `cmf.hint` name every state
/// (ready, disabled by the admin, stopped, budget); the router surface keeps
/// its shape.
#[tokio::test]
async fn oracle_flag_opens_the_loopback_open_mode_and_status_and_hints_name_every_state() {
    let mock = MockOracle::answering("travel");
    let mut cfg = stand_config(&mock.url());
    cfg.auth.require = None;
    // Without --oracle: the implicit open mode may not use the oracle.
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        |_| {},
    );
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.flags(), json!(["consent_off"]));
    assert!(r.body["cmf"].get("hint").is_none(), "{}", r.text);
    assert_eq!(mock.hits(), 0);
    drop(srv);

    // With --oracle it may; its feedback still teaches nothing.
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        |o| o.oracle_from_flag = true,
    );
    assert_eq!(status(&srv).await, "ready");
    assert_eq!(srv.get("/healthz").await.body["oracle_status"], "ready");
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(mock.hits(), 1);
    let id = r.body["id"].as_str().unwrap().to_string();
    let fb = srv
        .post(
            "/v1/feedback",
            None,
            &json!({"id": id, "question": "task", "label": "cards"}),
        )
        .await;
    assert_eq!(fb.status, 200, "{}", fb.text);
    assert_eq!(fb.body["learned"], false, "{}", fb.text);

    // Switched off by the admin.
    let s = srv
        .admin("POST", "/v1/admin/oracle", Some(&json!({"enabled": false})))
        .await;
    assert_eq!(s.body["status"], "disabled", "{}", s.text);
    let r = srv.decide(&topics_body(&rejected()[1])).await;
    assert_eq!(r.flags(), json!(["oracle_disabled"]));
    assert!(
        r.body["cmf"]["hint"]
            .as_str()
            .unwrap()
            .contains("switched off by the admin API"),
        "{}",
        r.text
    );
    enable(&srv).await;
    assert_eq!(status(&srv).await, "ready");

    // A stop rule.
    mock.set(|_| raw_reply(401, r#"{"error":{"message":"bad key"}}"#));
    let r = srv.decide(&topics_body(&rejected()[2])).await;
    assert_eq!(r.flags(), json!(["oracle_unavailable"]));
    assert_eq!(status(&srv).await, "stopped: http_401");
    assert_eq!(
        srv.get("/healthz").await.body["oracle_status"],
        "stopped: http_401"
    );
    let r = srv.decide(&topics_body(&rejected()[3])).await;
    assert_eq!(r.flags(), json!(["stopped"]));
    assert!(
        r.body["cmf"]["hint"]
            .as_str()
            .unwrap()
            .starts_with("the oracle was stopped by a stop rule (http_401)"),
        "{}",
        r.text
    );
    // The router surface: no hint in its answer, with or without extensions.
    let route = srv
        .post(
            "/v1/route",
            None,
            &json!({"input": {"text": rejected()[4]}, "taxonomy_id": "topics"}),
        )
        .await;
    assert_eq!(route.status, 200, "{}", route.text);
    assert!(!route.text.contains("hint"), "{}", route.text);
    let h = srv.get("/v1/healthz").await;
    assert_eq!(h.body["cmf"]["oracle_status"], "stopped: http_401");
    enable(&srv).await;
    assert_eq!(status(&srv).await, "ready");
    assert_eq!(mock.hits(), 2);
}
