//! The oracle cascade through the HTTP layer (spec decision-v4 §6.5, the
//! HTTP-level cases), hermetic: every oracle call goes to an in-test OpenRouter
//! mock on 127.0.0.1; the key is a fake read through the server's key lookup
//! (and, in one child process, from the real environment).
//!
//! * abstain → oracle with usage, cost and passthrough, over `/v1/decisions` and
//!   `/v1/route` (router fields `oracle`, `source`, promoted score), then the
//!   cache; the escalation audit and the metrics;
//! * gate-accepted questions never reach the oracle, even with `cmf.oracle`;
//! * cache: a repeat makes no call, a paraphrase only with near reuse opted
//!   in (0.8.11); single flight: two parallel identical HTTP requests make one
//!   call;
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
//! * the cache without the oracle (0.8.11): without consent, and with the
//!   oracle switched off, stopped or out of budget, a cache hit is served
//!   with no call and a miss keeps its refusal — an error names every
//!   untrained question, as 0.8.9 did, never which ones the cache held;
//!   single flight follows the cache's rule, and a near duplicate's
//!   follower never caches the answer to another input under its own; a
//!   0.8.9 `learn.log` replays and answers exact repeats, and
//!   `cache.legacy_cos: 1` turns its entries off;
//! * PII redaction on by default;
//! * the request body for one choice question is byte for byte the v4 driver's
//!   (9 ledger fixtures of `cortiq-decision`, sent over HTTP);
//! * the key comes only from the environment and its bytes are in no response,
//!   log line or state file (child process);
//! * auto-skills (0.8.6): an untrained 3-label contract is learned from a
//!   keyword-consistent mock into `auto-…`, activated at the trigger
//!   (`auto_start`), answered locally (`exact`, `certified: false`), a
//!   2-of-3 subset and a variant sharing two ids other contracts, an
//!   ambiguous one not learned, two runs byte-identical; the same ids under
//!   two instructions are two auto-skills, each answering its own contract;
//!   positional ids whose descriptions change per request never learn and
//!   `auto_max_skills` bounds them (`auto_skipped`); a random oracle is
//!   rejected (`auto_agreement`) and a consistent contract promoted; a refit
//!   (`auto_refit`) promoted and a regressing one rejected; rollback, restart,
//!   materialize, verify and `decide --labels`; the limits (`auto_max_labels`,
//!   `auto_max_skills`, `auto_skills: false`, score/noul, a key without
//!   `learning_allowed`, `/v1/route` without `taxonomy_id`); the `auto_tau`
//!   floor; a rare label quarantined, probability 0, still taught;
//! * state-less requests (0.8.8): a `state: {}` contract whose text is in the
//!   instructions is registered at its `auto_min_sightings`-th sighting,
//!   learned, answered locally over `/v1/decisions` and `/v1/systemone`, apart
//!   from the stateful contract of the same criteria and kept over a restart;
//!   a one-off contract is never registered, and stateful one-offs do not
//!   take the state-less contracts' slots; capacity errors carry the Decision
//!   Index marker (422 on System One, the oracle's context overflow 422
//!   everywhere, not a stop); PII in state-less instructions is redacted for
//!   the oracle; System One oracle and cache choice answers carry the one-hot
//!   distribution the kit's validator requires, and its `default` model name
//!   is accepted.
//!
//! Every request of [`Srv`] carries `x-cmf-extensions: 1`, so router-surface
//! answers include the opt-in `cmf` diagnostics (the exact default router
//! shapes are checked in `router_compat.rs`).

#[path = "support/toy_dir.rs"]
mod toy_dir;

use axum::body::Body;
use axum::http::{HeaderMap, Request};
use cortiq_decision::buffer::{LogRecord, read_records};
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
        // 0.8.8 (DESIGN C3): the verdict sits beside its distribution; a
        // bare verdict is still read (one-hot).
        let p = &schema["properties"][qid];
        let p = ["choice", "score", "noul"]
            .iter()
            .find_map(|k| p["properties"].get(*k))
            .unwrap_or(p);
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
    cortiq_decision::oracle::key_lookup(|name: &str| {
        (name == KEY_ENV).then(|| TEST_KEY.to_string())
    })
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
    // One explored text in four keeps the quarantined-label test short.
    c.learning.auto_explore_every = 4;
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
    fn restart(self, cfg: &Config) -> Self {
        self.restart_with(cfg, |_| {})
    }

    /// [`Srv::restart`], running `between` on the state directory while the
    /// server is closed.
    fn restart_with(mut self, cfg: &Config, between: impl FnOnce(&Path)) -> Self {
        self.app.take();
        self.server.take().unwrap().close().unwrap();
        between(&self.state_root());
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
    // A verdict without a distribution: one-hot (DESIGN C3).
    assert_eq!(
        r.body["answers"]["task"],
        json!({"type": "choice", "choice": "travel",
               "probabilities": {"Weather": 0, "billing": 0, "cards": 0, "travel": 1}, "confidence": 1})
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
    // A paraphrase is another state: by default (0.8.11) only the same
    // question hits, so it is a call.
    assert_eq!(srv.decide(&topics_body(p)).await.action(), "oracle");
    assert_eq!(mock.hits(), 2);
    // Near reuse opted in (`cache.threshold` 0.97, the default before
    // 0.8.11): the paraphrase is a cache answer.
    let mut near = stand_config(&mock.url());
    near.cache.threshold = 0.97;
    let srv = Srv::new(&near);
    assert_eq!(srv.decide(&topics_body(a)).await.action(), "oracle");
    let r3 = srv.decide(&topics_body(p)).await;
    assert_eq!(r3.action(), "cache", "paraphrase");
    assert_eq!(mock.hits(), 3);
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
    // One failure after a success does not stop; the count of failures in
    // a row is kept in oracle.state (it survives a restart).
    assert_eq!(
        srv.oracle_state(),
        json!({"enabled": true, "stop_reason": null, "stopped_unix": null,
               "budget_usd": null, "max_calls": null, "consecutive_errors": 1,
               "last_error": "http_500"}),
        "one failure after a success does not stop"
    );
    assert_eq!(
        srv.decide(&topics_body(&r[3])).await.flags(),
        json!(["oracle_unavailable"])
    );
    assert_eq!(srv.oracle_state()["stop_reason"], "max_errors");
    // The admin view names the last failure's code.
    let st = srv.admin("GET", "/v1/admin/oracle", None).await;
    assert_eq!(
        (st.body["status"].as_str(), st.body["last_error"].as_str()),
        (Some("stopped: max_errors"), Some("http_500")),
        "{}",
        st.text
    );
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

// ------------------------------------------------------------------ cache without the oracle

/// A key minted over the admin API without `oracle_allowed` (the default).
async fn plain_key(srv: &Srv, account: &str) -> String {
    let r = srv
        .admin("POST", "/v1/admin/keys", Some(&json!({"account": account})))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    r.body["key"].as_str().unwrap().to_string()
}

/// One `topics` question about `text`.
fn trained_body(text: &str, cmf: Option<Value>) -> Value {
    body(json!(text), json!({"task": choice(&TOPICS)}), cmf)
}

/// One untrained score question `u` about `text`.
fn score_body(text: &str, cmf: Option<Value>) -> Value {
    body(json!(text), json!({"u": score_question()}), cmf)
}

/// Question `q` of `r` came from the cache: no call, no cost, no refusal flag.
fn assert_cached(r: &Resp, q: &str) {
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.q(q)["action"], "cache", "{}", r.text);
    assert_eq!(r.q(q)["decision_path"], "escalate→cache");
    assert_eq!(r.q(q)["flags"], json!([]));
    assert_eq!(r.body["usage"]["cost"].as_f64(), Some(0.0));
    assert_eq!(r.body["cmf"]["usage"]["oracle"]["calls"], 0);
}

/// `first`, then 250 ms later (its call in flight) `second`, as `topics`
/// questions.
async fn in_parallel(srv: &Srv, first: &str, second: &str) -> (Resp, Resp) {
    let (b1, b2) = (topics_body(first), topics_body(second));
    tokio::join!(srv.decide(&b1), async {
        tokio::time::sleep(Duration::from_millis(250)).await;
        srv.decide(&b2).await
    })
}

/// The `CachePut` records of a state directory's `learn.log`.
fn logged_cache_puts(state: &Path) -> usize {
    let bytes = std::fs::read(state.join("learn.log")).unwrap();
    let (records, valid) = read_records(&bytes);
    assert_eq!(valid, bytes.len());
    records
        .iter()
        .filter(|r| matches!(r, LogRecord::CachePut(_)))
        .count()
}

/// 0.8.11: the answers the system already holds are served when the oracle
/// may not be called for the request — a key without `oracle_allowed`,
/// `cmf.oracle: false`, the router's `allow_oracle: false` — with no call;
/// a miss keeps the refusal of 0.8.9 (`consent_off`, 422).
#[tokio::test]
async fn a_request_without_oracle_consent_gets_cache_hits_and_no_call() {
    let mock = MockOracle::answering("travel");
    let srv = Srv::new(&stand_config(&mock.url()));
    let held = &rejected()[0];
    // The oracle answers a trained and an untrained question once.
    assert_eq!(
        srv.decide(&trained_body(held, None)).await.action(),
        "oracle"
    );
    let r = srv.decide(&score_body("x", None)).await;
    assert_eq!(r.q("u")["action"], "oracle", "{}", r.text);
    let rr = srv
        .post(
            "/v1/route",
            None,
            &json!({"taxonomy_id": "topics", "input": {"text": rejected()[2]}}),
        )
        .await;
    assert_eq!(rr.body["decision"]["source"], "oracle", "{}", rr.text);
    assert_eq!(mock.hits(), 3);
    let ledger = srv.ledger().len();

    // cmf.oracle: false (the open mode, where the oracle is otherwise
    // allowed; it ends with the first key).
    let off = Some(json!({"oracle": false}));
    assert_cached(&srv.decide(&trained_body(held, off.clone())).await, "task");
    assert_cached(&srv.decide(&score_body("x", off.clone())).await, "u");
    assert_eq!(
        srv.decide(&trained_body(&rejected()[1], off.clone()))
            .await
            .flags(),
        json!(["consent_off"])
    );
    assert_eq!(
        srv.decide(&score_body("y", off)).await.error(),
        (422, "UNSUPPORTED_QUESTION".to_string())
    );
    // The router's allow_oracle: false.
    let rr = srv
        .post("/v1/route", None, &json!({"taxonomy_id": "topics", "input": {"text": rejected()[2]}, "options": {"allow_oracle": false}}))
        .await;
    assert_eq!(rr.body["decision"]["source"], "cache", "{}", rr.text);

    // A key without oracle_allowed (from now on every request needs a key).
    let plain = plain_key(&srv, "plain").await;
    let r = srv
        .post("/v1/decisions", Some(&plain), &trained_body(held, None))
        .await;
    assert_cached(&r, "task");
    assert_eq!(r.body["answers"]["task"]["choice"], "travel");
    assert!(r.body["cmf"].get("hint").is_none(), "{}", r.text);
    let r = srv
        .post("/v1/decisions", Some(&plain), &score_body("x", None))
        .await;
    assert_cached(&r, "u");
    assert_eq!(r.body["answers"]["u"]["score"], 0, "{}", r.text);
    let rr = srv
        .post(
            "/v1/route",
            Some(&plain),
            &json!({"taxonomy_id": "topics", "input": {"text": rejected()[2]}}),
        )
        .await;
    assert_eq!(rr.body["decision"]["source"], "cache", "{}", rr.text);
    assert_eq!(rr.body["decision"]["task_label"], "travel");
    assert_eq!(rr.body["oracle"]["consulted"], false);
    // Misses are refused exactly as before.
    let r = srv
        .post(
            "/v1/decisions",
            Some(&plain),
            &trained_body(&rejected()[1], None),
        )
        .await;
    assert_eq!(r.action(), "abstain");
    assert_eq!(r.flags(), json!(["consent_off"]));
    assert_eq!(r.q("task")["decision_path"], "escalate→disabled");
    assert_eq!(
        srv.post("/v1/decisions", Some(&plain), &score_body("y", None))
            .await
            .error(),
        (422, "UNSUPPORTED_QUESTION".to_string())
    );

    // No call, no reservation; the hits are counted.
    assert_eq!(mock.hits(), 3);
    assert_eq!(srv.ledger().len(), ledger);
    let l = srv.learning().await;
    assert_eq!(l["cache"]["hits"], 6, "{}", l["cache"]);
}

/// 0.8.11: an oracle disabled (in the configuration or by the admin),
/// stopped by a stop rule or out of budget still serves what the cache holds
/// (also replayed after a restart); a miss keeps its flag, 422 or 503.
#[tokio::test]
async fn a_disabled_stopped_or_exhausted_oracle_still_serves_cache_hits() {
    let r = rejected();
    // A fail switch for the stop rule below.
    let fail = Arc::new(AtomicBool::new(false));
    let f2 = fail.clone();
    let mock = MockOracle::start(move |req| {
        if f2.load(Ordering::SeqCst) {
            raw_reply(500, r#"{"error":{"message":"upstream"}}"#)
        } else {
            answer_reply(req, |_, o| pick(o, "travel"), 1e-5)
        }
    });
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.max_errors = 1;
    let srv = Srv::new(&cfg);
    assert_eq!(
        srv.decide(&trained_body(&r[0], None)).await.action(),
        "oracle"
    );
    assert_eq!(
        srv.decide(&score_body("x", None)).await.q("u")["action"],
        "oracle"
    );
    assert_eq!(mock.hits(), 2);

    // The admin switch.
    let s = srv
        .admin("POST", "/v1/admin/oracle", Some(&json!({"enabled": false})))
        .await;
    assert_eq!(s.status, 200, "{}", s.text);
    assert_cached(&srv.decide(&trained_body(&r[0], None)).await, "task");
    assert_cached(&srv.decide(&score_body("x", None)).await, "u");
    assert_eq!(
        srv.decide(&trained_body(&r[1], None)).await.flags(),
        json!(["oracle_disabled"])
    );
    enable(&srv).await;

    // Stopped by max_errors.
    fail.store(true, Ordering::SeqCst);
    assert_eq!(
        srv.decide(&trained_body(&r[1], None)).await.flags(),
        json!(["oracle_unavailable"])
    );
    assert_eq!(srv.oracle_state()["stop_reason"], "max_errors");
    assert_eq!(mock.hits(), 3);
    assert_cached(&srv.decide(&trained_body(&r[0], None)).await, "task");
    assert_cached(&srv.decide(&score_body("x", None)).await, "u");
    assert_eq!(
        srv.decide(&trained_body(&r[2], None)).await.flags(),
        json!(["stopped"])
    );
    assert_eq!(
        srv.decide(&score_body("y", None)).await.error(),
        (503, "ORACLE_DISABLED".to_string())
    );
    assert_eq!(mock.hits(), 3);

    // Disabled in the configuration: the cache comes back from learn.log.
    let mut off = cfg.clone();
    off.oracle.enabled = false;
    let srv = srv.restart(&off);
    assert_cached(&srv.decide(&trained_body(&r[0], None)).await, "task");
    assert_cached(&srv.decide(&score_body("x", None)).await, "u");
    assert_eq!(
        srv.decide(&trained_body(&r[1], None)).await.flags(),
        json!(["oracle_disabled"])
    );
    assert_eq!(
        srv.decide(&score_body("y", None)).await.error(),
        (422, "UNSUPPORTED_QUESTION".to_string())
    );
    assert_eq!(mock.hits(), 3);

    // Out of budget (max_calls).
    fail.store(false, Ordering::SeqCst);
    let mut capped = stand_config(&mock.url());
    capped.oracle.max_calls = 2;
    let srv = Srv::new(&capped);
    assert_eq!(
        srv.decide(&trained_body(&r[0], None)).await.action(),
        "oracle"
    );
    assert_eq!(
        srv.decide(&score_body("x", None)).await.q("u")["action"],
        "oracle"
    );
    assert_eq!(mock.hits(), 5);
    assert_cached(&srv.decide(&trained_body(&r[0], None)).await, "task");
    assert_cached(&srv.decide(&score_body("x", None)).await, "u");
    assert_eq!(
        srv.decide(&trained_body(&r[1], None)).await.flags(),
        json!(["budget"])
    );
    assert_eq!(
        srv.decide(&score_body("y", None)).await.error(),
        (503, "ORACLE_BUDGET_EXHAUSTED".to_string())
    );
    assert_eq!(mock.hits(), 5);
}

/// 0.8.11: single flight follows the cache's rule. By default only the same
/// question waits for a call in flight (a near duplicate leads its own and
/// gets its own answer). With near reuse opted in a paraphrase follows and
/// gets the leader's answer, but nothing is cached under its own input — the
/// oracle never read it: restarted with exact reuse only, the paraphrase is
/// a call and gets its own answer, not the leader's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_flight_follows_the_cache_rule_and_never_caches_another_inputs_answer() {
    let a = "cruise ship cabin deck please";
    let p = "cruise ship cabin today";
    let c = cos(&phi_p(a), &phi_p(p));
    assert!((0.97..1.0).contains(&c), "paraphrase cos {c}");
    // A slow oracle that answers the two texts differently.
    let mock = MockOracle::start(|req| {
        let deck = req.state().as_str().is_some_and(|s| s.contains("deck"));
        let label = if deck { "travel" } else { "cards" };
        let mut r = answer_reply(req, move |_, o| pick(o, label), 1.3e-5);
        r.delay = Duration::from_millis(800);
        r
    });
    // Exact only (the default): the same question follows, once logged.
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let (r1, r2) = in_parallel(&srv, a, a).await;
    let mut actions = vec![r1.action().to_string(), r2.action().to_string()];
    actions.sort();
    assert_eq!(actions, ["cache", "oracle"]);
    assert_eq!(mock.hits(), 1);
    assert_eq!(srv.learning().await["cache"]["entries"], 1);
    assert_eq!(logged_cache_puts(&srv.state_root()), 1);
    // A near duplicate leads its own call and gets its own answer.
    let srv = Srv::new(&cfg);
    let (r1, r2) = in_parallel(&srv, a, p).await;
    assert_eq!((r1.action(), r2.action()), ("oracle", "oracle"));
    assert_eq!(r1.body["answers"]["task"]["choice"], "travel");
    assert_eq!(r2.body["answers"]["task"]["choice"], "cards");
    assert_eq!(mock.hits(), 3);

    // Near reuse opted in: the paraphrase follows and gets the leader's
    // answer; only the leader's input is cached.
    let mut near = stand_config(&mock.url());
    near.cache.threshold = 0.97;
    let srv = Srv::new(&near);
    let (r1, r2) = in_parallel(&srv, a, p).await;
    assert_eq!((r1.action(), r2.action()), ("oracle", "cache"));
    assert_eq!(r2.body["answers"]["task"]["choice"], "travel");
    assert_eq!(mock.hits(), 4);
    assert_eq!(srv.learning().await["cache"]["entries"], 1);
    assert_eq!(logged_cache_puts(&srv.state_root()), 1);
    // While near reuse is on, a repeat of the paraphrase hits the leader's
    // entry (no call).
    assert_eq!(srv.decide(&topics_body(p)).await.action(), "cache");
    assert_eq!(mock.hits(), 4);
    // Restarted with exact reuse only: the leader's text is a cache answer,
    // the paraphrase a call with its own answer.
    let srv = srv.restart(&cfg);
    assert_eq!(srv.learning().await["cache"]["entries"], 1);
    assert_eq!(srv.decide(&topics_body(a)).await.action(), "cache");
    let r = srv.decide(&topics_body(p)).await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(r.body["answers"]["task"]["choice"], "cards");
    assert_eq!(mock.hits(), 5);
    assert_eq!(srv.decide(&topics_body(p)).await.action(), "cache");
    assert_eq!(mock.hits(), 5);
}

/// A `learn.log` written by 0.8.9 (cache puts without an input digest)
/// replays unchanged: its entries answer an exact repeat (cos φ_P ≥ 0.9999)
/// and nothing nearer; a 0.8.9 binary reads the 0.8.11 records (their digest
/// rides in the scope). `cache.legacy_cos: 1` turns those entries off on the
/// same state directory (they are not loaded), so the oracle pass gives each
/// question an entry with its digest, which answers it from then on.
#[tokio::test]
async fn cache_records_of_0_8_9_replay_and_answer_exact_repeats() {
    let a = "cruise ship cabin deck please";
    let p = "cruise ship cabin today";
    let c = cos(&phi_p(a), &phi_p(p));
    assert!(c < cortiq_decision::cache::EXACT_COS, "paraphrase cos {c}");
    let mock = MockOracle::answering("travel");
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    assert_eq!(srv.decide(&topics_body(a)).await.action(), "oracle");
    // The same log as 0.8.9 wrote it: every cache put without its digest.
    let srv = srv.restart_with(&cfg, |state| {
        let path = state.join("learn.log");
        let bytes = std::fs::read(&path).unwrap();
        let (records, valid) = read_records(&bytes);
        assert_eq!(valid, bytes.len());
        let mut old = Vec::new();
        let mut puts = 0;
        for r in records {
            let r = match r {
                LogRecord::CachePut(mut e) => {
                    assert!(e.input.is_some());
                    assert!(e.scope.starts_with("skill:topics:"), "{}", e.scope);
                    e.input = None;
                    puts += 1;
                    LogRecord::CachePut(e)
                }
                other => other,
            };
            old.extend(r.frame());
        }
        assert_eq!(puts, 1);
        std::fs::write(&path, old).unwrap();
    });
    assert_eq!(srv.learning().await["cache"]["entries"], 1);
    let r = srv.decide(&topics_body(a)).await;
    assert_eq!(r.action(), "cache", "{}", r.text);
    assert_eq!(mock.hits(), 1);
    assert_eq!(srv.decide(&topics_body(p)).await.action(), "oracle");
    assert_eq!(mock.hits(), 2);

    // Legacy reuse off: the digest-less entry is not loaded (only p's
    // counts), so `a` is a call, cached with its digest.
    let mut strict = cfg.clone();
    strict.cache.legacy_cos = 1.0;
    let srv = srv.restart(&strict);
    let l = srv.learning().await;
    assert_eq!(l["cache"]["entries"], 1, "{}", l["cache"]);
    assert_eq!(l["cache"]["legacy_cos"], 1.0);
    assert_eq!(srv.decide(&topics_body(a)).await.action(), "oracle");
    assert_eq!(mock.hits(), 3);
    assert_eq!(srv.decide(&topics_body(a)).await.action(), "cache");
    assert_eq!(logged_cache_puts(&srv.state_root()), 3);
    // Back to the default: the log kept every record; `a` hits.
    let srv = srv.restart(&cfg);
    assert_eq!(srv.learning().await["cache"]["entries"], 3);
    assert_eq!(srv.decide(&topics_body(a)).await.action(), "cache");
    assert_eq!(srv.decide(&topics_body(p)).await.action(), "cache");
    assert_eq!(mock.hits(), 3);
}

/// Two untrained questions about one state: `u` (the oracle answered it
/// once) and `v` (never asked).
fn two_untrained(cmf: Option<Value>) -> Value {
    let v = json!({"type": "score", "instructions": "How risky?", "criteria": ["low", "high"]});
    body(json!("x"), json!({"u": score_question(), "v": v}), cmf)
}

/// 0.8.11: a request refused the oracle is answered from the cache alone,
/// and when a question the cache misses fails it, the error names every
/// untrained question — exactly the 0.8.9 error, whether the cache held
/// some of them or nothing — never which ones the cache held (a failed
/// request is neither metered nor recorded). An empty cache fails a request
/// without consent before any work, as in 0.8.9.
#[tokio::test]
async fn a_refused_request_never_tells_which_questions_the_cache_holds() {
    let mock = MockOracle::answering("travel");
    let cfg = stand_config(&mock.url());
    // The reference: a server whose cache is empty.
    let empty = Srv::new(&cfg);
    let srv = Srv::new(&cfg);
    let r = srv.decide(&score_body("x", None)).await;
    assert_eq!(r.q("u")["action"], "oracle", "{}", r.text);
    assert_eq!(mock.hits(), 1);
    let details = |r: &Resp| r.body["error"]["metadata"]["details"].clone();

    // Without consent: 422, both questions named, as on the empty server.
    let off = Some(json!({"oracle": false}));
    let want = empty.decide(&two_untrained(off.clone())).await;
    assert_eq!(want.error(), (422, "UNSUPPORTED_QUESTION".to_string()));
    let got = srv.decide(&two_untrained(off.clone())).await;
    assert_eq!(got.error(), (422, "UNSUPPORTED_QUESTION".to_string()));
    assert_eq!(details(&got), details(&want), "{}", got.text);
    assert_eq!(
        details(&got)["questions"]["u"]["oracle"],
        "the oracle is not allowed for this request or key"
    );
    assert!(details(&got)["questions"]["v"].is_object(), "{}", got.text);
    assert_eq!(got.body["error"]["message"], want.body["error"]["message"]);
    // Every question held: answered.
    assert_cached(&srv.decide(&score_body("x", off)).await, "u");

    // The oracle switched off by the admin (consent given): the same.
    for s in [&empty, &srv] {
        let r = s
            .admin("POST", "/v1/admin/oracle", Some(&json!({"enabled": false})))
            .await;
        assert_eq!(r.status, 200, "{}", r.text);
    }
    let want = empty.decide(&two_untrained(None)).await;
    assert_eq!(want.error(), (422, "UNSUPPORTED_QUESTION".to_string()));
    let got = srv.decide(&two_untrained(None)).await;
    assert_eq!(got.error(), (422, "UNSUPPORTED_QUESTION".to_string()));
    assert_eq!(details(&got), details(&want), "{}", got.text);
    assert_eq!(
        details(&got)["questions"]["u"]["oracle"],
        "the oracle is disabled"
    );
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
    // The driver's body: no distribution asked (DESIGN C3).
    cfg.oracle.probabilities = false;
    let srv = Srv::new(&cfg);
    for (i, (q, row)) in cases.iter().enumerate() {
        let mut question = q.as_object().unwrap().clone();
        question.insert("type".into(), json!("choice"));
        let r = srv
            .decide(&body(row["text"].clone(), json!({"task": question}), None))
            .await;
        assert_eq!(r.status, 200, "row {i}: {}", r.text);
        assert_eq!(r.action(), "oracle", "row {i}");
        let a = &r.body["answers"]["task"];
        assert_eq!(a["choice"], row["oracle"]["choice"]);
        assert_eq!(a["probabilities"][a["choice"].as_str().unwrap()], 1);
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
        // The router surface keeps its flag vocabulary: a missing key is
        // `oracle_disabled` there, in the answer and in the audit records.
        let rr = srv
            .post(
                "/v1/route",
                None,
                &json!({"taxonomy_id": "topics", "input": {"text": rejected()[1]}}),
            )
            .await;
        assert_eq!(
            rr.body["decision"]["flags"],
            json!(["low_confidence", "oracle_disabled"]),
            "{}",
            rr.text
        );
        let e = srv.get("/v1/escalations").await;
        let rec = e.body["records"].as_array().unwrap();
        assert_eq!(rec.len(), 2, "{}", e.text);
        for r in rec {
            assert_eq!(
                r["flags"],
                json!(["low_confidence", "oracle_disabled"]),
                "{r}"
            );
        }
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

/// A key read with surrounding whitespace is trimmed and works; one that is
/// still not a key (a NUL, CR LF, whitespace or a byte outside ASCII inside
/// it, `Bearer `, quotes) is `bad_key`: never sent, named by position in the
/// status and the hint, never echoed in an answer, a status or a state file.
/// A budget that cannot hold one call is `budget_too_small` (with the
/// minimum), one that was spent `budget_exhausted`.
#[tokio::test]
async fn keys_are_trimmed_or_refused_unsent_and_budgets_name_their_state() {
    let mock = MockOracle::answering("travel");
    let lookup = |raw: String| -> KeyLookup {
        cortiq_decision::oracle::key_lookup(move |name: &str| {
            (name == KEY_ENV).then(|| raw.clone())
        })
    };
    let no_key_bytes = |text: &str, what: &str| {
        for needle in [TEST_KEY, "TESTKEY-wp7", "0123456789abcdef"] {
            assert!(!text.contains(needle), "{what} holds key bytes\n{text}");
        }
    };
    // Surrounding whitespace (a .env file's CR LF): trimmed, the key alone
    // is sent.
    let srv = Srv::open_on(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &stand_config(&mock.url()),
        lookup(format!(" {TEST_KEY}\r\n")),
    );
    let s = srv.admin("GET", "/v1/admin/oracle", None).await;
    assert_eq!(
        (s.body["status"].as_str(), s.body["key_trimmed"].as_bool()),
        (Some("ready"), Some(true)),
        "{}",
        s.text
    );
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(
        mock.requests()[0].header("authorization"),
        Some(format!("Bearer {TEST_KEY}").as_str())
    );
    drop(srv);

    let hits = mock.hits();
    let bad = [
        (format!("{TEST_KEY}\0"), "a control character at byte"),
        (
            format!("{}\r\n{}", &TEST_KEY[..12], &TEST_KEY[12..]),
            "a control character at byte 13",
        ),
        (
            format!("{} {}", &TEST_KEY[..12], &TEST_KEY[12..]),
            "whitespace inside it at byte 13",
        ),
        (
            format!("{}\t{}", &TEST_KEY[..12], &TEST_KEY[12..]),
            "whitespace inside it at byte 13",
        ),
        (format!("{TEST_KEY}é"), "a byte outside ASCII"),
        (format!("{TEST_KEY}\u{fffd}"), "a byte outside ASCII"),
        (format!("Bearer {TEST_KEY}"), "it starts with 'Bearer '"),
        (format!("\"{TEST_KEY}\""), "it starts or ends with a quote"),
    ];
    for (raw, problem) in bad {
        let srv = Srv::open_on(
            &toy().path,
            tempfile::tempdir().unwrap(),
            &stand_config(&mock.url()),
            lookup(raw.clone()),
        );
        let s = srv.admin("GET", "/v1/admin/oracle", None).await;
        assert_eq!(s.body["status"], "bad_key", "{raw:?}: {}", s.text);
        assert_eq!(
            (s.body["key_present"].as_bool(), s.body["key_ok"].as_bool()),
            (Some(true), Some(false))
        );
        assert!(
            s.body["key_problem"].as_str().unwrap().contains(problem),
            "{raw:?}: {}",
            s.text
        );
        no_key_bytes(&s.text, "GET /v1/admin/oracle");
        let h = srv.get("/healthz").await;
        assert_eq!(h.body["oracle_status"], "bad_key", "{}", h.text);
        let r = srv.decide(&topics_body(&rejected()[1])).await;
        assert_eq!(r.action(), "abstain", "{}", r.text);
        assert_eq!(r.flags(), json!(["oracle_disabled", "bad_key"]));
        let hint = r.body["cmf"]["hint"].as_str().unwrap();
        assert!(
            hint.starts_with("the oracle key is not usable (")
                && hint.contains(problem)
                && hint.ends_with(&format!(
                    "): fix {KEY_ENV} in the server's environment and restart it"
                )),
            "{hint}"
        );
        no_key_bytes(&r.text, "a decision");
        let e = srv.decide(&untrained_body()).await;
        assert_eq!(e.error(), (422, "UNSUPPORTED_QUESTION".to_string()));
        assert_eq!(
            e.body["error"]["metadata"]["details"]["questions"]["u"]["oracle"],
            "the oracle key in the server's environment is not usable"
        );
        no_key_bytes(&e.text, "an untrained error");
        // The router surface keeps its flag vocabulary.
        let route = srv
            .post(
                "/v1/route",
                None,
                &json!({"input": {"text": rejected()[2]}, "taxonomy_id": "topics"}),
            )
            .await;
        assert_eq!(route.status, 200, "{}", route.text);
        assert_eq!(
            route.body["decision"]["flags"],
            json!(["low_confidence", "oracle_disabled"]),
            "{}",
            route.text
        );
        for f in files_under(&srv.state_root()) {
            let b = std::fs::read(&f).unwrap();
            assert!(
                !b.windows(16).any(|w| w == b"0123456789abcdef"),
                "{} holds key bytes",
                f.display()
            );
        }
    }
    assert_eq!(mock.hits(), hits, "a bad key was sent");

    // A budget below one call's reservation, nothing spent: too small, with
    // the minimum; the hint says so.
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.budget_usd = 1e-7;
    let srv = Srv::new(&cfg);
    let s = srv.admin("GET", "/v1/admin/oracle", None).await;
    assert_eq!(s.body["status"], "budget_too_small", "{}", s.text);
    assert!(
        s.body["min_call_usd"].as_f64().unwrap() > 1e-7,
        "{}",
        s.text
    );
    let r = srv.decide(&topics_body(&rejected()[3])).await;
    assert_eq!(r.flags(), json!(["budget"]));
    let hint = r.body["cmf"]["hint"].as_str().unwrap();
    assert!(
        hint.starts_with("the oracle budget cannot hold one call (it needs $"),
        "{hint}"
    );
    let floor = s.body["min_call_usd"].as_f64().unwrap();
    drop(srv);
    // The advertised minimum (rounded up to the micro-dollar, as printed):
    // ready, since it holds the smallest possible call; a real question
    // reserves more, is refused, and the status becomes budget_too_small
    // with that call's reservation instead of staying "ready".
    let printed = |x: f64| -> f64 {
        cortiq_decision::oracle_setup::usd_ceil(x)[1..]
            .parse()
            .unwrap()
    };
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.budget_usd = printed(floor);
    let srv = Srv::new(&cfg);
    assert_eq!(status(&srv).await, "ready");
    let r = srv.decide(&topics_body(&rejected()[3])).await;
    assert_eq!(r.flags(), json!(["budget"]));
    let s = srv.admin("GET", "/v1/admin/oracle", None).await;
    assert_eq!(s.body["status"], "budget_too_small", "{}", s.text);
    let need = s.body["min_call_usd"].as_f64().unwrap();
    assert!(need > printed(floor), "{}", s.text);
    let hint = r.body["cmf"]["hint"].as_str().unwrap();
    assert!(
        hint.contains(&format!(
            "restart the server with --oracle-budget of at least {}",
            cortiq_decision::oracle_setup::usd_ceil(need)
        )),
        "{hint}"
    );
    drop(srv);
    assert_eq!(mock.hits(), hits, "nothing was sent");
    // Restarted with that figure: the call is made (one hit).
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.budget_usd = printed(need);
    let srv = Srv::new(&cfg);
    assert_eq!(
        srv.decide(&topics_body(&rejected()[3])).await.action(),
        "oracle"
    );
    drop(srv);
    // Spent: exhausted.
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.max_calls = 1;
    let srv = Srv::new(&cfg);
    assert_eq!(status(&srv).await, "ready");
    assert_eq!(
        srv.decide(&topics_body(&rejected()[3])).await.action(),
        "oracle"
    );
    assert_eq!(status(&srv).await, "budget_exhausted");
    let r = srv.decide(&topics_body(&rejected()[4])).await;
    assert_eq!(r.flags(), json!(["budget"]));
    assert!(
        r.body["cmf"]["hint"]
            .as_str()
            .unwrap()
            .starts_with("the oracle budget is used up"),
        "{}",
        r.text
    );
    assert_eq!(mock.hits(), hits + 2);
}

/// An admin limit kept in `oracle.state` binds after a restart with a larger
/// configured budget or call limit: the startup line (with the file's path)
/// and the decisions hint name it and the admin request that lifts it — a
/// figure the admin API takes, rounded up — never "restart the server with
/// --oracle-budget"; following the hint makes the call.
#[tokio::test]
async fn an_admin_limit_outlives_a_restart_and_its_hint_makes_the_call() {
    use cortiq_decision::oracle_setup::{usd_ceil, usd_fine};
    let mock = MockOracle::answering("travel");
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let s = srv
        .admin(
            "POST",
            "/v1/admin/oracle",
            Some(&json!({"budget_usd": 0.0})),
        )
        .await;
    assert_eq!(s.body["status"], "budget_too_small", "{}", s.text);
    // Restarted with a budget of $5.00: the admin limit still binds.
    let mut five = cfg.clone();
    five.oracle.budget_usd = 5.0;
    let srv = srv.restart(&five);
    assert_eq!(status(&srv).await, "budget_too_small");
    let startup = |srv: &Srv, cfg: &Config| {
        let st = srv.server.as_ref().unwrap().state();
        cortiq_server::decisions::oracle_startup_line(st.cascade().unwrap(), cfg, None)
    };
    let file = srv.state_root().join("oracle.state");
    let (ready, line) = startup(&srv, &five);
    assert!(!ready);
    assert!(
        line.starts_with(&format!(
            "oracle: NOT ready — the admin limit budget_usd $0.00 in {} binds (the file keeps it across restarts, so --oracle-budget cannot lift it; the next call needs a budget of $",
            file.display()
        )) && line.contains("(at most the configured $5.00) or lift it with {\"budget_usd\": null} (")
            && !line.contains("restart with")
            && !line.contains("restart the server"),
        "{line}"
    );
    let hits = mock.hits();
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.flags(), json!(["budget"]), "{}", r.text);
    let need = srv.admin("GET", "/v1/admin/oracle", None).await.body["min_call_usd"]
        .as_f64()
        .unwrap();
    let hint = r.body["cmf"]["hint"].as_str().unwrap();
    assert_eq!(
        hint,
        format!(
            "no oracle call fits: the admin limit budget_usd $0.00 in the server's oracle.state binds (the file keeps it across restarts, so --oracle-budget cannot lift it; the next call needs a budget of {}): raise it with POST /v1/admin/oracle {{\"budget_usd\": {}}} (at most the configured $5.00) or lift it with {{\"budget_usd\": null}}",
            usd_fine(need),
            &usd_ceil(need)[1..]
        )
    );
    assert_eq!(mock.hits(), hits, "nothing was sent");
    // Following it: the admin API takes the figure, the call is made.
    let x: f64 = usd_ceil(need)[1..].parse().unwrap();
    let s = srv
        .admin(
            "POST",
            "/v1/admin/oracle",
            Some(&json!({ "budget_usd": x })),
        )
        .await;
    assert_eq!(s.body["status"], "ready", "{}", s.text);
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(mock.hits(), hits + 1);
    // The admin call limit at the calls made, then a restart with a larger
    // configured limit: named likewise; lifting it makes the next call.
    let s = srv
        .admin(
            "POST",
            "/v1/admin/oracle",
            Some(&json!({"max_calls": 1, "budget_usd": null})),
        )
        .await;
    assert_eq!(s.body["status"], "budget_exhausted", "{}", s.text);
    let mut more = five.clone();
    more.oracle.max_calls = 50;
    let srv = srv.restart(&more);
    let (_, line) = startup(&srv, &more);
    assert!(
        line.starts_with(&format!(
            "oracle: NOT ready — the admin limit max_calls 1 in {} binds (the file keeps it across restarts, so --oracle-max-calls cannot lift it): raise it with POST /v1/admin/oracle {{\"max_calls\": 2}} (at most the configured 50) or lift it with {{\"max_calls\": null}} (",
            file.display()
        )),
        "{line}"
    );
    let r = srv.decide(&topics_body(&rejected()[1])).await;
    assert_eq!(r.flags(), json!(["budget"]), "{}", r.text);
    assert_eq!(
        r.body["cmf"]["hint"],
        "the oracle budget is used up: the admin limit max_calls 1 in the server's oracle.state binds (the file keeps it across restarts, so --oracle-max-calls cannot lift it): raise it with POST /v1/admin/oracle {\"max_calls\": 2} (at most the configured 50) or lift it with {\"max_calls\": null}",
        "{}",
        r.text
    );
    let s = srv
        .admin(
            "POST",
            "/v1/admin/oracle",
            Some(&json!({"max_calls": null})),
        )
        .await;
    assert_eq!(s.body["status"], "ready", "{}", s.text);
    let r = srv.decide(&topics_body(&rejected()[1])).await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(mock.hits(), hits + 2);
}

/// `serve --oracle` on loopback without keys or `auth.require`: the implicit
/// open caller reaches the oracle, but nothing it brings teaches the model —
/// not its feedback, and not the oracle's answer to a skill's own question
/// (`/v1/route`), which teaches for any identified caller: the answer is
/// cached, the learning buffer stays empty. An explicit `auth.require: false`
/// keeps teaching as before.
#[tokio::test]
async fn the_implicit_open_mode_with_the_oracle_never_teaches() {
    let route = |text: &str| json!({"input": {"text": text}, "taxonomy_id": "topics"});
    let mock = MockOracle::answering("travel");
    let mut cfg = stand_config(&mock.url());
    cfg.auth.require = None;
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        |o| o.oracle_from_flag = true,
    );
    for text in &rejected()[..3] {
        let r = srv.post("/v1/route", None, &route(text)).await;
        assert_eq!(r.status, 200, "{}", r.text);
        assert_eq!(r.body["decision"]["source"], "oracle", "{}", r.text);
    }
    // A repeat is a cache answer: the oracle's answer was kept.
    let again = srv.post("/v1/route", None, &route(&rejected()[0])).await;
    assert_eq!(again.body["decision"]["source"], "cache", "{}", again.text);
    assert_eq!(mock.hits(), 3);
    let l = srv.learning().await;
    assert_eq!(l["examples_added"], 0, "{l}");
    assert_eq!(l["buffer"]["examples"], 0, "{l}");
    assert_eq!(l["attempts"], 0, "{l}");
    drop(srv);

    // The explicit open mode (auth.require: false): the same route call
    // teaches the skill, as R1 defined it.
    let mut cfg = stand_config(&mock.url());
    cfg.auth.require = Some(false);
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        |o| o.oracle_from_flag = true,
    );
    let r = srv.post("/v1/route", None, &route(&rejected()[5])).await;
    assert_eq!(r.body["decision"]["source"], "oracle", "{}", r.text);
    let l = srv.learning().await;
    assert_eq!(l["examples_added"], 1, "{l}");
    assert_eq!(l["buffer"]["examples"], 1, "{l}");
}

// ------------------------------------------------------------------ auto-skills (0.8.6)

/// The pool of a label the auto-skill tests use, `None` for any other option.
fn pool_of(label: &str) -> Option<&'static [&'static str]> {
    matches!(
        label,
        "Weather" | "billing" | "cards" | "travel" | "food" | "cruise"
    )
    .then(|| pool(label))
}

/// An oracle that answers a choice with the first option whose pool holds a
/// word of the state text (else the first option): consistent by keyword.
fn keyword_mock() -> MockOracle {
    MockOracle::start(|req| {
        let text = req.state().as_str().unwrap_or("").to_string();
        answer_reply(
            req,
            move |_, opts| {
                let words: Vec<&str> = text.split(' ').collect();
                for o in opts {
                    if pool_of(o).is_some_and(|p| p.iter().any(|w| words.contains(w))) {
                        return json!(o);
                    }
                }
                json!(opts[0])
            },
            1e-5,
        )
    })
}

/// `n` texts of `label` with cos φ_P < `max_cos` to every text whose φ_P is
/// in `seen` (and to each other): no cache hit across the labels of one
/// contract, whose answers would otherwise cross over. Extends `seen`.
fn texts_apart(
    label: &str,
    n: usize,
    seed: u64,
    tag: &str,
    max_cos: f32,
    seen: &mut Vec<Vec<f32>>,
) -> Vec<String> {
    let p = pool(label);
    let mut rng = Lcg(seed);
    let mut out: Vec<String> = Vec::new();
    let mut tries = 0;
    while out.len() < n {
        tries += 1;
        assert!(
            tries < 100_000,
            "could not find {n} distinct '{label}' texts"
        );
        let mut words = Vec::new();
        for _ in 0..3 + rng.below(2) {
            words.push(p[rng.below(p.len())]);
        }
        words.push(FILLER[rng.below(FILLER.len())]);
        let t = format!("{} {tag}{}", words.join(" "), out.len());
        let f = phi_p(&t);
        if seen.iter().all(|q| cos(q, &f) < max_cos) {
            seen.push(f);
            out.push(t);
        }
    }
    out
}

/// Up to `n` paraphrases of `text` (its words in another order, another
/// filler): cos φ_P ≥ 0.97 to `text`, < 0.995 to each other (not duplicates).
fn near_variants(text: &str, n: usize, seed: u64, tag: &str) -> Vec<String> {
    let mut words: Vec<&str> = text.split(' ').collect();
    words.pop(); // the tag
    let orig = phi_p(text);
    let mut rng = Lcg(seed);
    let mut out: Vec<String> = Vec::new();
    let mut phis: Vec<Vec<f32>> = Vec::new();
    for k in 0..400 {
        if out.len() >= n {
            break;
        }
        let mut w = words.clone();
        for i in (1..w.len()).rev() {
            w.swap(i, rng.below(i + 1));
        }
        if rng.below(2) == 0 {
            w.push(FILLER[rng.below(FILLER.len())]);
        }
        let t = format!("{} {tag}{k}", w.join(" "));
        let f = phi_p(&t);
        if cos(&orig, &f) >= 0.97 && phis.iter().all(|q| cos(q, &f) < 0.995) {
            phis.push(f);
            out.push(t);
        }
    }
    out
}

/// The lessons of a contract: `n` texts per label, pairwise cos φ_P < 0.97
/// across every label (the oracle's answer to one is never cached for
/// another).
fn lessons_of(labels: &[(&'static str, u64, &str)], n: usize) -> Vec<(&'static str, Vec<String>)> {
    let mut seen = Vec::new();
    labels
        .iter()
        .map(|&(l, seed, tag)| (l, texts_apart(l, n, seed, tag, 0.97, &mut seen)))
        .collect()
}

/// The id of the auto-skill of the contract `q` (its instructions and
/// criteria with the descriptions, DESIGN A18).
fn auto_id_of(q: &Value) -> String {
    cortiq_decision::manifest::auto_skill_id(
        &q["instructions"],
        q["criteria"].as_object().expect("a choice question"),
    )
}

/// The id of the auto-skill of the contract [`choice`] builds over `labels`.
fn auto_id(labels: &[&str]) -> String {
    auto_id_of(&choice(labels))
}

/// Decide `text` under the contract `q` (question id `task`).
async fn ask(srv: &Srv, q: &Value, text: &str) -> Resp {
    let r = srv
        .decide(&body(json!(text), json!({"task": q.clone()}), None))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    r
}

/// Teach the contract `q` with the lessons interleaved (one text of each
/// label in turn) until the first promotion; the answers must be the oracle's
/// (or the cache's) and name the lesson's label. Returns the texts decided.
async fn teach_contract(srv: &Srv, q: &Value, lessons: &[(&str, Vec<String>)]) -> usize {
    teach_until(srv, q, lessons, 1).await
}

/// [`teach_contract`] until the server counts `promotions` promotions.
async fn teach_until(
    srv: &Srv,
    q: &Value,
    lessons: &[(&str, Vec<String>)],
    promotions: u64,
) -> usize {
    let longest = lessons.iter().map(|l| l.1.len()).max().unwrap_or(0);
    let mut n = 0;
    for i in 0..longest {
        for (label, ts) in lessons {
            let Some(t) = ts.get(i) else { continue };
            let r = ask(srv, q, t).await;
            n += 1;
            assert!(
                r.action() == "oracle" || r.action() == "cache",
                "{}",
                r.text
            );
            assert_eq!(r.body["answers"]["task"]["choice"], *label, "{}", r.text);
            if srv.learning().await["promotions"].as_u64().unwrap_or(0) >= promotions {
                return n;
            }
        }
    }
    panic!("no promotion after {n} texts: {}", srv.learning().await);
}

fn food_travel_cruise() -> &'static [(&'static str, Vec<String>)] {
    static L: OnceLock<Vec<(&'static str, Vec<String>)>> = OnceLock::new();
    L.get_or_init(|| {
        lessons_of(
            &[
                ("food", 41, "af"),
                ("travel", 43, "av"),
                ("cruise", 47, "ac"),
            ],
            25,
        )
    })
}

impl Srv {
    /// Close the server, keeping the state directory.
    fn close(mut self) -> (tempfile::TempDir, PathBuf) {
        self.app.take();
        self.server.take().unwrap().close().unwrap();
        let base = self.base.clone();
        let dir = std::mem::replace(&mut self.dir, tempfile::tempdir().unwrap());
        (dir, base)
    }
}

#[tokio::test]
async fn an_untrained_contract_becomes_an_auto_skill_and_answers_locally() {
    let labels = ["food", "travel", "cruise"];
    let q = choice(&labels);
    let id = auto_id(&labels);
    let lessons = food_travel_cruise();
    let lessons = lessons.to_vec();
    // Without an oracle the contract is 422, as before.
    let mock = keyword_mock();
    let mut off = stand_config(&mock.url());
    off.oracle.enabled = false;
    let r = Srv::new(&off)
        .decide(&body(
            json!(lessons[0].1[0]),
            json!({"task": q.clone()}),
            None,
        ))
        .await;
    assert_eq!(r.error(), (422, "UNSUPPORTED_QUESTION".into()));

    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let before = srv.learning().await;
    assert_eq!(before["auto_contracts"], 0);
    let r = ask(&srv, &q, &lessons[0].1[0]).await;
    assert_eq!(r.action(), "oracle");
    assert_eq!(r.q("task")["match"], "untrained");
    assert_eq!(r.q("task")["skill"], Value::Null);
    let l = srv.learning().await;
    assert_eq!(l["auto_contracts"], 1);
    assert_eq!(l["auto_skills"][0]["id"], id);
    assert_eq!(l["auto_skills"][0]["labels"], json!(labels));
    assert_eq!(l["auto_skills"][0]["served"], Value::Null);
    assert_eq!(l["buffer"]["labels"][0]["skill"], id);

    // The rest of the lessons (the first food text was taught above).
    let mut rest = lessons.clone();
    rest[0].1.remove(0);
    let taught = 1 + teach_contract(&srv, &q, &rest).await;
    let calls = mock.hits();
    assert_eq!(calls, taught, "every taught text was an oracle call");
    let l = srv.learning().await;
    assert_eq!(l["promotions"], 1, "{l}");
    assert_eq!(l["generation"], 1);
    assert_eq!(l["isolation_violations"], 0);
    let rec = &l["recent"][0];
    assert_eq!(rec["kind"], "auto_start");
    assert_eq!(rec["outcome"], "promoted");
    assert_eq!(rec["skill"], id);
    assert_eq!(rec["auto"]["eligible"], json!(["cruise", "food", "travel"]));
    assert_eq!(rec["auto"]["quarantined"], json!([]));
    assert!(rec["auto"]["agreement"]["macro"].as_f64().unwrap() >= 0.8);
    assert_eq!(rec["gate_after"]["certified"], false);
    // The recorded T is floored at `auto_temperature_min` (DESIGN A15).
    assert!(
        rec["gate_after"]["temperature"].as_f64().unwrap() as f32 >= 0.02,
        "{rec}"
    );
    assert_eq!(l["auto_skills"][0]["served"]["generation"], 1);
    assert_eq!(
        l["auto_skills"][0]["served"]["active"],
        json!(["cruise", "food", "travel"])
    );
    assert_eq!(l["skills"][&id]["taxonomy_version"], 2);
    // `examples` per label = R per label as the attempt saw it: every taught
    // text is a served learned row AND still in the buffer, counted once
    // (the sum is the attempt's `rows.total`, not buffer + served).
    let per_label = l["auto_skills"][0]["examples"].as_object().unwrap();
    assert_eq!(per_label.len(), 3);
    let total: u64 = per_label.values().map(|v| v.as_u64().unwrap()).sum();
    assert_eq!(total, rec["auto"]["rows"]["total"].as_u64().unwrap());
    assert_eq!(total as usize, taught);
    for b in l["buffer"]["labels"].as_array().unwrap() {
        assert_eq!(b["skill"], id);
        assert_eq!(per_label[b["label"].as_str().unwrap()], b["examples"]);
    }
    eprintln!(
        "auto_start after {taught} texts ({calls} oracle calls): {}",
        rec["auto"]
    );

    // Listings.
    let s = srv.get("/v1/skills").await.body;
    let sk = s["skills"].as_array().unwrap();
    assert_eq!(sk.len(), 3);
    assert_eq!(sk[0]["id"], "topics");
    assert_eq!(sk[0]["auto"], false);
    assert_eq!(sk[2]["id"], id);
    assert_eq!(sk[2]["auto"], true);
    assert_eq!(sk[2]["labels"], json!(["cruise", "food", "travel"]));
    assert_eq!(sk[2]["active_labels"], 3);
    assert_eq!(sk[2]["quarantined_labels"], json!([]));
    assert_eq!(sk[2]["examples"].as_u64().unwrap() as usize, taught);
    assert_eq!(sk[2]["certified"], false);
    assert_eq!(sk[2]["has_rubric"], true);
    let one = srv.get(&format!("/v1/skills/{id}")).await.body;
    assert_eq!(one["rubric"]["instructions"], "Which topic?");
    assert_eq!(one["rubric"]["criteria"]["food"], "about food");
    for t in one["tasks"].as_array().unwrap() {
        assert_eq!(
            (t["origin"].as_str(), t["state"].as_str()),
            (Some("cold_start"), Some("active"))
        );
    }
    let h = srv.get("/healthz").await.body;
    assert_eq!(
        (h["skills"].as_u64(), h["auto_skills"].as_u64()),
        (Some(3), Some(1))
    );
    let g = srv.admin("GET", "/v1/admin/generations", None).await.body;
    assert_eq!(g["current"], 1);
    assert_eq!(g["generations"][0]["auto_skills"], json!([id]));

    // The same contract: exact, the auto-skill, never certified; a fresh text
    // of each label is local at least once (θ of a young skill is weak: the
    // others may abstain and teach, DESIGN A13).
    let hits = mock.hits();
    let mut local = 0;
    let mut escalated = 0;
    for (label, seed, tag) in [
        ("food", 97, "zf"),
        ("travel", 98, "zv"),
        ("cruise", 99, "zc"),
    ] {
        let mut local_here = 0;
        for t in distinct_texts(label, 4, seed, tag, 0.995) {
            let r = ask(&srv, &q, &t).await;
            assert_eq!(r.q("task")["match"], "exact", "{}", r.text);
            assert_eq!(r.q("task")["skill"], id);
            assert_eq!(r.q("task")["certified"], false);
            assert_eq!(r.body["answers"]["task"]["choice"], label, "{}", r.text);
            if r.action() == "local" {
                assert_eq!(r.q("task")["decision_path"], "router:uncertified");
                assert!(r.q("task")["gate"]["p_top"].as_f64().unwrap() >= 0.9);
                local_here += 1;
            } else if r.action() == "oracle" {
                escalated += 1;
            } else {
                assert_eq!(r.action(), "cache", "{}", r.text);
            }
        }
        assert!(local_here >= 1, "no local answer of a fresh {label} text");
        local += local_here;
    }
    assert_eq!(mock.hits() - hits, escalated);
    eprintln!("fresh texts: {local} local, {escalated} escalated");
    // A 2-of-3 subset of the ids is a different contract (DESIGN A18): not
    // served by the auto-skill, untrained, learned separately.
    let subset = ["food", "travel"];
    let r = ask(&srv, &choice(&subset), &lessons[0].1[0]).await;
    assert_eq!(r.q("task")["match"], "untrained", "{}", r.text);
    assert_eq!(r.q("task")["skill"], Value::Null);
    assert_eq!(r.action(), "oracle");
    let sid = auto_id(&subset);
    assert_ne!(sid, id);
    assert_eq!(srv.learning().await["auto_contracts"], 2);
    // A variant sharing two ids: another contract, another auto-skill.
    let variant = ["food", "travel", "Weather", "cards"];
    let r = ask(&srv, &choice(&variant), &lessons[1].1[0]).await;
    assert_eq!(r.q("task")["match"], "untrained");
    assert_eq!(r.action(), "oracle");
    let l = srv.learning().await;
    assert_eq!(l["auto_contracts"], 3);
    assert_ne!(auto_id(&variant), id);
    let ids: Vec<&str> = l["auto_skills"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_str().unwrap())
        .collect();
    let vid = auto_id(&variant);
    let mut want = vec![id.as_str(), sid.as_str(), vid.as_str()];
    want.sort_unstable();
    assert_eq!(ids, want, "contracts listed by id");
    // The same ids under other instructions, or with another description:
    // other contracts too (the id set is not the key).
    let mut other = q.clone();
    other["instructions"] = json!("Is it urgent?");
    let r = ask(&srv, &other, &lessons[2].1[0]).await;
    assert_eq!(r.q("task")["match"], "untrained", "{}", r.text);
    assert_eq!(r.action(), "oracle");
    let mut described = q.clone();
    described["criteria"]["food"] = json!("meals and drinks");
    let r = ask(&srv, &described, &lessons[2].1[1]).await;
    assert_eq!(r.q("task")["match"], "untrained", "{}", r.text);
    assert_eq!(srv.learning().await["auto_contracts"], 5);
    // The criteria in another order: the same contract, served.
    let mut reordered = Map::new();
    for l in ["cruise", "travel", "food"] {
        reordered.insert(l.to_string(), q["criteria"][l].clone());
    }
    let mut same = q.clone();
    same["criteria"] = Value::Object(reordered);
    assert_eq!(auto_id_of(&same), id);
    let r = ask(&srv, &same, &lessons[2].1[2]).await;
    assert_eq!(r.q("task")["match"], "exact", "{}", r.text);
    assert_eq!(r.q("task")["skill"], id);
    assert_eq!(srv.learning().await["auto_contracts"], 5);
    // An ambiguous contract (a subset of two data skills, each with the
    // evidence two one-word labels need: their names, DESIGN C2.1) is not
    // learned.
    let named = json!({"type": "choice", "instructions": "Which topic?",
                       "criteria": {"billing": "billing", "cards": "cards"}});
    let r = ask(&srv, &named, &lessons[0].1[1]).await;
    assert_eq!(r.q("task")["match"], "untrained");
    assert!(
        r.q("task")["reason"]
            .as_str()
            .unwrap()
            .contains("ambiguous")
    );
    assert_eq!(srv.learning().await["auto_contracts"], 5);

    // Determinism: the same lesson on a second server writes the same bytes.
    let sha1 =
        cortiq_decision::generation::file_sha256(&srv.state_root().join("generations/g000001.cmf"))
            .unwrap();
    let again = Srv::new(&cfg);
    ask(&again, &q, &lessons[0].1[0]).await;
    teach_contract(&again, &q, &rest).await;
    let sha2 = cortiq_decision::generation::file_sha256(
        &again.state_root().join("generations/g000001.cmf"),
    )
    .unwrap();
    assert_eq!(sha1, sha2);
}

/// The newest attempt of `GET /v1/admin/learning` (`recent` is oldest first).
fn latest(l: &Value) -> &Value {
    l["recent"].as_array().unwrap().last().unwrap()
}

/// Teach `food_travel_cruise` to a fresh server until the auto-skill is
/// promoted; returns the server and the number of texts taught.
async fn activated(cfg: &Config) -> (Srv, usize) {
    let srv = Srv::new(cfg);
    let q = choice(&["food", "travel", "cruise"]);
    let n = teach_contract(&srv, &q, food_travel_cruise()).await;
    assert_eq!(srv.learning().await["generation"], 1);
    (srv, n)
}

#[tokio::test]
async fn a_random_oracle_is_rejected_and_a_consistent_contract_is_promoted() {
    // Contract X answered at random (by the parity of the text's sha256):
    // the agreement on C stays far below 0.8.
    let mock = MockOracle::start(|req| {
        let text = req.state().as_str().unwrap_or("").to_string();
        let flip = sha256_hex(text.as_bytes()).as_bytes()[0].is_multiple_of(2);
        answer_reply(req, move |_, opts| json!(opts[usize::from(flip)]), 1e-5)
    });
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let x = choice(&["food", "cruise"]);
    let xid = auto_id(&["food", "cruise"]);
    let mut seen = Vec::new();
    let mut lesson = Vec::new();
    for (l, seed, tag) in [
        ("food", 61, "rf"),
        ("cruise", 63, "rc"),
        ("travel", 65, "rv"),
    ] {
        lesson.extend(texts_apart(l, 30, seed, tag, 0.97, &mut seen));
    }
    let mut attempts = 0;
    for t in &lesson {
        let r = ask(&srv, &x, t).await;
        assert_eq!(r.action(), "oracle", "{}", r.text);
        let l = srv.learning().await;
        if l["attempts"].as_u64().unwrap() >= 1 {
            attempts = l["attempts"].as_u64().unwrap();
            break;
        }
    }
    assert_eq!(attempts, 1, "{}", srv.learning().await);
    let l = srv.learning().await;
    assert_eq!(l["rejections"], 1);
    assert_eq!(l["promotions"], 0);
    assert_eq!(l["generation"], 0);
    let rec = &l["recent"][0];
    assert_eq!(rec["skill"], xid);
    assert_eq!(rec["kind"], "auto_start");
    assert_eq!(rec["outcome"], "rejected");
    assert_eq!(rec["reason"], "auto_agreement");
    let agreement = rec["auto"]["agreement"]["macro"].as_f64().unwrap();
    assert!(agreement < 0.8, "{rec}");
    eprintln!(
        "random oracle: agreement {agreement:.3} on {} C rows",
        rec["auto"]["rows"]["calibration"]
    );
    // The trigger label's counter was reset; the examples stay.
    let trigger = rec["label"].as_str().unwrap();
    let lab = l["buffer"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["skill"] == xid && c["label"] == trigger)
        .unwrap();
    assert_eq!(lab["new"], 0);
    assert!(lab["examples"].as_u64().unwrap() >= 25);
    assert!(!srv.state_root().join("generations/g000001.cmf").exists());
    assert_eq!(
        srv.get("/v1/skills").await.body["skills"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    // Contract Y with a consistent oracle: promoted at its trigger.
    mock.set(|req| {
        let text = req.state().as_str().unwrap_or("").to_string();
        answer_reply(
            req,
            move |_, opts| {
                let words: Vec<&str> = text.split(' ').collect();
                for o in opts {
                    if pool_of(o).is_some_and(|p| p.iter().any(|w| words.contains(w))) {
                        return json!(o);
                    }
                }
                json!(opts[0])
            },
            1e-5,
        )
    });
    let y = choice(&["travel", "cruise"]);
    let yl = lessons_of(&[("travel", 71, "yv"), ("cruise", 73, "yc")], 25);
    teach_contract(&srv, &y, &yl).await;
    let l = srv.learning().await;
    assert_eq!(
        (l["promotions"].as_u64(), l["generation"].as_u64()),
        (Some(1), Some(1))
    );
    assert_eq!(latest(&l)["skill"], auto_id(&["travel", "cruise"]));
    assert_eq!(latest(&l)["kind"], "auto_start");
    let g = srv.admin("GET", "/v1/admin/generations", None).await.body;
    assert_eq!(
        g["generations"][0]["auto_skills"],
        json!([auto_id(&["travel", "cruise"])])
    );
}

#[tokio::test]
async fn an_auto_skill_is_refitted_and_a_regressing_challenger_is_rejected() {
    let mock = keyword_mock();
    let mut cfg = stand_config(&mock.url());
    cfg.learning.auto_k = 16;
    let (srv, _) = activated(&cfg).await;
    let labels = ["food", "travel", "cruise"];
    let q = choice(&labels);
    let id = auto_id(&labels);
    // 25 more food examples: a local answer is confirmed by feedback (weight
    // 3), an abstention is answered by the oracle; either way one example.
    let mut seen = Vec::new();
    let more = texts_apart("food", 40, 51, "nf", 0.995, &mut seen);
    let attempts0 = srv.learning().await["attempts"].as_u64().unwrap();
    let mut taught = 0;
    for t in &more {
        let r = ask(&srv, &q, t).await;
        assert_eq!(r.q("task")["skill"], id);
        if r.action() == "local" {
            let fb = json!({"id": r.body["id"], "question": "task", "label": "food"});
            let f = srv.post("/v1/feedback", None, &fb).await;
            assert_eq!(f.status, 200, "{}", f.text);
            assert_eq!(f.body["known_label"], true);
            taught += usize::from(f.body["learned"] == true);
        } else {
            taught += 1;
        }
        if srv.learning().await["attempts"].as_u64().unwrap() > attempts0 {
            break;
        }
    }
    let l = srv.learning().await;
    assert_eq!(l["attempts"].as_u64().unwrap(), attempts0 + 1, "{l}");
    assert!(taught >= 25);
    let rec = latest(&l);
    assert_eq!(rec["kind"], "auto_refit", "{rec}");
    assert_eq!(rec["label"], "food");
    assert_eq!(rec["outcome"], "promoted", "{rec}");
    assert_eq!(rec["holdout"]["gated"], true);
    assert_eq!(rec["holdout"]["passed"], true);
    assert_eq!(l["generation"], 2);
    assert_eq!(l["promotions"], 2);
    let s = srv.get(&format!("/v1/skills/{id}")).await.body;
    assert_eq!(
        s["examples"].as_u64().unwrap(),
        rec["auto"]["rows"]["total"].as_u64().unwrap()
    );
    assert_eq!(s["taxonomy_version"], 2, "the active set did not change");
    eprintln!("auto_refit: {}", rec["holdout"]);

    // A regressing challenger: the oracle (and the feedback) now label
    // paraphrases of cruise texts as `food` — the ones of the cruise rows in
    // C, so that the refitted food topology claims those C rows from cruise
    // and the macro agreement on C drops below the champion's.
    mock.set(|req| answer_reply(req, |_, opts| pick(opts, "food"), 1e-5));
    let cruise_c: Vec<&String> = food_travel_cruise()[2]
        .1
        .iter()
        .filter(|t| cortiq_decision::certify::auto_row_key(&phi_p(t)).1)
        .collect();
    assert!(cruise_c.len() >= 2, "{} cruise rows in C", cruise_c.len());
    let mut flipped = Vec::new();
    for (i, t) in cruise_c.iter().enumerate() {
        flipped.extend(near_variants(t, 16, 53 + i as u64, "ff"));
    }
    assert!(flipped.len() >= 15, "{} paraphrases", flipped.len());
    // Plain food texts make up the trigger count; the paraphrases carry the regression.
    flipped.extend(texts_apart("food", 15, 55, "fg", 0.995, &mut seen));
    for t in &flipped {
        let r = ask(&srv, &q, t).await;
        if r.action() == "local" {
            let fb = json!({"id": r.body["id"], "question": "task", "label": "food"});
            assert_eq!(srv.post("/v1/feedback", None, &fb).await.status, 200);
        }
        if srv.learning().await["attempts"].as_u64().unwrap() > attempts0 + 1 {
            break;
        }
    }
    let l = srv.learning().await;
    assert_eq!(l["attempts"].as_u64().unwrap(), attempts0 + 2, "{l}");
    let rec = latest(&l);
    assert_eq!(rec["kind"], "auto_refit", "{rec}");
    assert_eq!(rec["label"], "food");
    assert_eq!(rec["outcome"], "rejected", "{rec}");
    assert_eq!(rec["reason"], "holdout_regression");
    assert_eq!(rec["holdout"]["passed"], false);
    assert_eq!(l["generation"], 2);
    assert_eq!(l["rejections"], 1);
    eprintln!("regression: {}", rec["holdout"]);
}

#[tokio::test]
async fn rollback_restart_materialize_and_verify_with_an_auto_skill() {
    let mock = keyword_mock();
    let cfg = stand_config(&mock.url());
    let (srv, _) = activated(&cfg).await;
    let labels = ["food", "travel", "cruise"];
    let q = choice(&labels);
    let id = auto_id(&labels);
    let fresh = distinct_texts("food", 6, 97, "zf", 0.995);
    let local = |r: &Resp| r.action() == "local" && r.q("task")["skill"] == id;
    let served_local = {
        let mut v = None;
        for t in &fresh {
            if local(&ask(&srv, &q, t).await) {
                v = Some(t.clone());
                break;
            }
        }
        v.expect("a fresh food text decided locally")
    };

    // Restart: the generation, the contract registry and the buffer come back.
    let srv = srv.restart(&cfg);
    let l = srv.learning().await;
    assert_eq!(
        (l["generation"].as_u64(), l["auto_contracts"].as_u64()),
        (Some(1), Some(1))
    );
    assert_eq!(l["auto_skills"][0]["served"]["generation"], 1);
    assert!(l["buffer"]["examples"].as_u64().unwrap() >= 71);
    let r = ask(&srv, &q, &served_local).await;
    assert!(local(&r), "{}", r.text);
    assert_eq!(srv.get("/healthz").await.body["auto_skills"], 1);

    // Rollback to the base: the contract is untrained again (the oracle), the
    // examples are kept and would start it over at the next trigger.
    let rb = srv
        .admin(
            "POST",
            "/v1/admin/rollback",
            Some(&json!({"generation": 0})),
        )
        .await;
    assert_eq!(rb.status, 200, "{}", rb.text);
    assert_eq!(rb.body["buffer_kept"], true);
    let r = ask(&srv, &q, &served_local).await;
    assert_eq!(r.q("task")["match"], "untrained");
    assert!(
        r.action() == "oracle" || r.action() == "cache",
        "{}",
        r.text
    );
    assert_eq!(
        srv.get("/v1/skills").await.body["skills"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let l = srv.learning().await;
    assert_eq!(l["auto_contracts"], 1);
    assert_eq!(l["auto_skills"][0]["served"], Value::Null);
    // Forward again, and over a restart.
    let rb = srv
        .admin(
            "POST",
            "/v1/admin/rollback",
            Some(&json!({"generation": 1})),
        )
        .await;
    assert_eq!(rb.status, 200, "{}", rb.text);
    let srv = srv.restart(&cfg);
    let hits = mock.hits();
    let r = ask(&srv, &q, &served_local).await;
    assert!(local(&r), "{}", r.text);
    assert_eq!(mock.hits(), hits + usize::from(r.action() == "oracle"));

    // Materialize the served generation (as `cortiq decision materialize`),
    // verify it (`verify`), decide from the file by labels (`decide --labels`).
    let (dir, base) = srv.close();
    let state = cortiq_decision::statedir::StateDir::open(dir.path().join("state")).unwrap();
    let served = cortiq_decision::generation::open_served(&base, &state, Verify::Full).unwrap();
    assert_eq!(served.verify_full().unwrap().skills, 3);
    let out = dir.path().join("materialised.cmf");
    cortiq_decision::container::materialize(&served, &out).unwrap();
    let full = DecisionModel::open(&out, Verify::Full).unwrap();
    assert!(full.skill(&id).unwrap().manifest.is_auto());
    assert_eq!(full.rows(&id).unwrap().rows.len(), 0);
    assert!(full.rows_learned(&id).unwrap().unwrap().rows.len() >= 71);
    let loaded = cortiq_decision::service::LoadedModel::new(full).unwrap();
    // Labels alone never name an auto-skill (its contract is the rubric,
    // DESIGN A18): `decide --labels` needs `--skill`, whose rubric then
    // supplies the contract (`route_question`), an exact match.
    assert!(
        cortiq_decision::matching::skill_for_labels(
            &loaded.skill_labels(),
            &["cruise", "food", "travel"]
        )
        .is_err()
    );
    let own = cortiq_server::decisions::route_question(loaded.skill(&id).unwrap());
    let m =
        cortiq_decision::matching::match_question(&loaded.skill_labels(), &own, Some(&id)).unwrap();
    assert_eq!(m.kind, cortiq_decision::matching::MatchKind::Exact);
    assert_eq!(m.skill.as_deref(), Some(id.as_str()));
    // The file alone needs `--skill` for an unnamed decision (two data skills).
    assert!(cortiq_decision::eval::select_skill(loaded.model(), None).is_err());
    assert_eq!(
        cortiq_decision::eval::select_skill(loaded.model(), Some(&id)).unwrap(),
        id
    );
    drop(loaded);
    // Served from the materialised file: the contract is exact and local.
    // The state directory is fresh (a materialised base refuses the old
    // CURRENT), so learn.log has no contract record: the registry is seeded
    // from the served auto-skill (its labels and rubric) and the skill keeps
    // learning — an abstention's oracle answer and a feedback are stored and
    // an attempt reaches `auto_refit`; before this seeding every example
    // was refused (`full`) and the skill was frozen at its materialised state.
    let mut cfg_m = cfg.clone();
    cfg_m.learning.refit_min_new = 3;
    let srv = Srv::open_on(&out, tempfile::tempdir().unwrap(), &cfg_m, test_key());
    let r = ask(&srv, &q, &served_local).await;
    assert!(local(&r), "{}", r.text);
    assert_eq!(r.q("task")["certified"], false);
    assert_eq!(srv.get("/healthz").await.body["auto_skills"], 1);
    let l = srv.learning().await;
    assert_eq!(l["auto_contracts"], 1, "seeded from the served model: {l}");
    assert_eq!(l["auto_skills"][0]["id"], id);
    assert_eq!(l["auto_skills"][0]["labels"], json!(labels));
    assert_eq!(
        l["auto_skills"][0]["served"]["active"],
        json!(["cruise", "food", "travel"])
    );
    assert_eq!(l["buffer"]["examples"], 0);
    let base_gen = l["generation"].as_u64().unwrap();
    let served_rows: u64 = l["auto_skills"][0]["examples"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_u64().unwrap())
        .sum();
    assert!(served_rows >= 71, "{l}");
    // The learn.log of the fresh state directory holds the seeded record.
    let (log, replayed) =
        cortiq_decision::buffer::LearnLog::open(&srv.state_root().join("learn.log")).unwrap();
    drop(log);
    assert!(matches!(
        replayed.records.as_slice(),
        [cortiq_decision::buffer::LogRecord::Contract(c)] if c.skill == id && c.ids == labels
    ));
    // Feedback on a local answer is learned, an abstention's oracle answer
    // too; the third example triggers an `auto_refit` on the materialised
    // base.
    let fb = json!({"id": r.body["id"], "question": "task", "label": "food"});
    let f = srv.post("/v1/feedback", None, &fb).await;
    assert_eq!(f.status, 200, "{}", f.text);
    assert_eq!(
        (f.body["learned"].as_bool(), f.body["full"].as_bool()),
        (Some(true), Some(false)),
        "{}",
        f.text
    );
    let mut seen = Vec::new();
    let hits = mock.hits();
    let mut taught = 1;
    for t in texts_apart("cruise", 12, 131, "mc", 0.995, &mut seen) {
        let r = ask(&srv, &q, &t).await;
        assert_eq!(r.q("task")["skill"], id, "{}", r.text);
        if r.action() == "oracle" {
            assert_eq!(r.body["answers"]["task"]["choice"], "cruise");
            taught += 1;
        } else if r.action() == "local" {
            let fb = json!({"id": r.body["id"], "question": "task", "label": "cruise"});
            let f = srv.post("/v1/feedback", None, &fb).await;
            assert_eq!(f.body["learned"], true, "{}", f.text);
            taught += 1;
        }
        if srv.learning().await["attempts"].as_u64().unwrap() >= 1 {
            break;
        }
    }
    let l = srv.learning().await;
    assert_eq!(l["examples_added"].as_u64().unwrap(), taught, "{l}");
    // Whether a cruise text abstains (and teaches through the oracle) or is
    // answered locally (and teaches through the feedback) depends on the
    // platform's rounding of the young gate; either way the examples count.
    let _ = hits;
    assert_eq!(l["attempts"], 1, "{l}");
    let rec = latest(&l);
    assert_eq!(rec["kind"], "auto_refit", "{rec}");
    assert_eq!(rec["outcome"], "promoted", "{rec}");
    assert_eq!(l["generation"].as_u64().unwrap(), base_gen + 1);
    let per_label: u64 = l["auto_skills"][0]["examples"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_u64().unwrap())
        .sum();
    assert_eq!(per_label, served_rows + taught, "R counted once: {l}");
    assert_eq!(
        srv.get(&format!("/v1/skills/{id}")).await.body["examples"],
        per_label
    );

    // learn.log lost, generations kept: the served auto-skill is registered
    // again at open and keeps learning.
    let (dir_m, _) = srv.close();
    std::fs::remove_file(dir_m.path().join("state/learn.log")).unwrap();
    let srv = Srv::open_on(&out, dir_m, &cfg_m, test_key());
    let l = srv.learning().await;
    assert_eq!(l["generation"].as_u64().unwrap(), base_gen + 1, "{l}");
    assert_eq!(l["auto_contracts"], 1, "{l}");
    assert_eq!(l["buffer"]["examples"], 0);
    let r = ask(&srv, &q, &served_local).await;
    assert!(local(&r), "{}", r.text);
    // New food texts (`served_local` entered the refit's rows through its
    // feedback above: a duplicate now): an abstention teaches through the
    // oracle, a local answer through its feedback.
    for t in distinct_texts("food", 8, 141, "lf", 0.995) {
        let r = ask(&srv, &q, &t).await;
        assert_eq!(r.q("task")["skill"], id, "{}", r.text);
        if local(&r) {
            let fb = json!({"id": r.body["id"], "question": "task", "label": "food"});
            let f = srv.post("/v1/feedback", None, &fb).await;
            assert_eq!(f.body["learned"], true, "{}", f.text);
            break;
        }
    }
    assert!(
        srv.learning().await["examples_added"].as_u64().unwrap() >= 1,
        "the served auto-skill learns again without its learn.log"
    );
    drop(srv);
    // The offline rollback of the CLI on the closed state directory.
    let cur = cortiq_decision::generation::rollback_state(&state, 0, Some(&base)).unwrap();
    assert_eq!(cur.generation, 0);
}

#[tokio::test]
async fn auto_skill_limits_and_who_teaches() {
    let mock = keyword_mock();
    let labels = ["food", "travel", "cruise"];
    let q = choice(&labels);
    let text = &food_travel_cruise()[0].1[0];
    let none = |l: &Value| {
        assert_eq!(l["examples_added"], 0, "{l}");
        assert_eq!(l["auto_contracts"], 0, "{l}");
        assert_eq!(l["attempts"], 0, "{l}");
    };
    // auto_max_labels: a wider contract is answered and not learned.
    let mut cfg = stand_config(&mock.url());
    cfg.learning.auto_max_labels = 2;
    let srv = Srv::new(&cfg);
    let r = ask(&srv, &q, text).await;
    assert_eq!(r.action(), "oracle");
    let l = srv.learning().await;
    none(&l);
    assert_eq!(l["auto_skipped"], 1, "{l}");
    let r = ask(&srv, &choice(&["food", "cruise"]), text).await;
    assert_eq!(r.action(), "oracle");
    assert_eq!(srv.learning().await["auto_contracts"], 1);
    drop(srv);
    // auto_max_skills: the second contract is not learned.
    let mut cfg = stand_config(&mock.url());
    cfg.learning.auto_max_skills = 1;
    let srv = Srv::new(&cfg);
    ask(&srv, &q, text).await;
    ask(&srv, &choice(&["food", "cruise"]), text).await;
    let l = srv.learning().await;
    assert_eq!(
        (l["auto_contracts"].as_u64(), l["auto_skipped"].as_u64()),
        (Some(1), Some(1))
    );
    assert_eq!(l["examples_added"], 1);
    drop(srv);
    // auto_skills: false — the oracle answers, nothing is recorded.
    let mut cfg = stand_config(&mock.url());
    cfg.learning.auto_skills = false;
    let srv = Srv::new(&cfg);
    let r = ask(&srv, &q, text).await;
    assert_eq!(r.action(), "oracle");
    let l = srv.learning().await;
    none(&l);
    assert_eq!(l["auto_skipped"], 0);
    drop(srv);
    // Score and noul questions stay oracle-only; a key without
    // `learning_allowed` creates no contract and no example.
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let r = srv.decide(&untrained_body()).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.q("u")["action"], "oracle");
    let r = srv
        .decide(&body(
            json!(text),
            json!({"n": {"type": "noul", "instructions": "Is it about food?"}}),
            None,
        ))
        .await;
    assert_eq!(r.q("n")["action"], "oracle", "{}", r.text);
    none(&srv.learning().await);
    drop(srv);
    // A key without `learning_allowed` creates no contract and no example
    // (keys make the server keyed: the open caller is checked first).
    let srv = Srv::new(&cfg);
    let key = srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"account": "guest", "rate_per_min": 0, "oracle_allowed": true})),
        )
        .await;
    let key = key.body["key"].as_str().unwrap().to_string();
    let r = srv
        .post(
            "/v1/decisions",
            Some(&key),
            &body(json!(text), json!({"task": q.clone()}), None),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.action(), "oracle");
    let l = srv.learning().await;
    none(&l);
    assert_eq!(l["auto_skipped"], 0);
    // A key with it teaches the same contract.
    let owner = srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"account": "owner", "rate_per_min": 0, "oracle_allowed": true, "learning_allowed": true})),
        )
        .await;
    let owner = owner.body["key"].as_str().unwrap().to_string();
    let r = srv
        .post(
            "/v1/decisions",
            Some(&owner),
            &body(
                json!(food_travel_cruise()[1].1[0]),
                json!({"task": q.clone()}),
                None,
            ),
        )
        .await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    let l = srv.learning().await;
    assert_eq!(
        (l["auto_contracts"].as_u64(), l["examples_added"].as_u64()),
        (Some(1), Some(1))
    );
    drop(srv);

    // `/v1/route` without `taxonomy_id` on a single-skill file keeps working
    // once an auto-skill is served (it names no implicit taxonomy).
    let single = toy().path.parent().unwrap().join("s1.cmf");
    let srv = Srv::open_on(&single, tempfile::tempdir().unwrap(), &cfg, test_key());
    let route = json!({"input": {"text": rejected()[0]}});
    let r = srv.post("/v1/route", None, &route).await;
    assert_eq!(r.status, 200, "{}", r.text);
    teach_contract(&srv, &q, food_travel_cruise()).await;
    assert_eq!(srv.get("/healthz").await.body["skills"], 2);
    let r = srv.post("/v1/route", None, &route).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert!(
        r.body["meta"]["taxonomy_version"]
            .as_str()
            .unwrap()
            .starts_with("topics@"),
        "{}",
        r.text
    );
    let tx = srv.get("/v1/taxonomies").await.body;
    assert_eq!(tx["taxonomies"].as_array().map(Vec::len), Some(2), "{tx}");
    // Named, the auto-skill routes too.
    let id = auto_id(&labels);
    let r = srv
        .post(
            "/v1/route",
            None,
            &json!({"taxonomy_id": id, "input": {"text": food_travel_cruise()[0].1[0]}}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["decision"]["task_label"], "food");
}

/// The instructions of the `task` question of a mock request (the system
/// message ends with the canonical `{"questions": …}` line).
fn asked_instructions(req: &MockRequest) -> String {
    let v = req.json();
    let system = v["messages"][0]["content"].as_str().unwrap();
    let qs: Value = serde_json::from_str(system.rsplit('\n').next().unwrap()).unwrap();
    qs["questions"]["task"]["instructions"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

/// A `{yes, no}` contract under `instructions` (the same descriptions).
fn yes_no(instructions: &str) -> Value {
    json!({"type": "choice", "instructions": instructions,
           "criteria": {"yes": "it is", "no": "it is not"}})
}

/// The same ids `{yes, no}` under two instructions are two contracts, two
/// auto-skills (DESIGN A18): each answers its own question locally and never
/// the other's — one text gets `yes` under one and `no` under the other.
#[tokio::test]
async fn the_same_ids_under_two_instructions_are_two_auto_skills() {
    const FOOD: &str = "Is it about food?";
    const TRAVEL: &str = "Is it about travel?";
    // The oracle reads the question: yes when the text has a word of the
    // topic the instructions ask about.
    let mock = MockOracle::start(|req| {
        let text = req.state().as_str().unwrap_or("").to_string();
        let topic = if asked_instructions(req) == FOOD {
            "food"
        } else {
            "travel"
        };
        answer_reply(
            req,
            move |_, _| {
                let words: Vec<&str> = text.split(' ').collect();
                json!(if pool(topic).iter().any(|w| words.contains(w)) {
                    "yes"
                } else {
                    "no"
                })
            },
            1e-5,
        )
    });
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let (food_q, travel_q) = (yes_no(FOOD), yes_no(TRAVEL));
    let (fid, tid) = (auto_id_of(&food_q), auto_id_of(&travel_q));
    assert_ne!(fid, tid);
    // Food texts are `yes` and travel texts `no` under the food question.
    let mut seen = Vec::new();
    let food_lessons = vec![
        ("yes", texts_apart("food", 25, 141, "sf", 0.97, &mut seen)),
        ("no", texts_apart("travel", 25, 143, "sv", 0.97, &mut seen)),
    ];
    let n = teach_contract(&srv, &food_q, &food_lessons).await;
    let l = srv.learning().await;
    assert_eq!(l["generation"], 1, "{l}");
    assert_eq!(latest(&l)["skill"], fid);
    assert_eq!(l["auto_contracts"], 1);
    eprintln!("food question active after {n} texts");
    // The travel question: another contract, untrained, the oracle's answer
    // (no under the travel question for a food text), learned separately.
    let fresh_food = distinct_texts("food", 4, 151, "yf", 0.995);
    let r = ask(&srv, &travel_q, &fresh_food[0]).await;
    assert_eq!(r.q("task")["match"], "untrained", "{}", r.text);
    assert_eq!(r.q("task")["skill"], Value::Null);
    assert_eq!(r.action(), "oracle");
    assert_eq!(r.body["answers"]["task"]["choice"], "no");
    assert_eq!(srv.learning().await["auto_contracts"], 2);
    let travel_lessons = vec![
        ("yes", texts_apart("travel", 25, 161, "tv", 0.97, &mut seen)),
        ("no", texts_apart("food", 25, 163, "tf", 0.97, &mut seen)),
    ];
    let n = teach_until(&srv, &travel_q, &travel_lessons, 2).await;
    let l = srv.learning().await;
    assert_eq!(l["generation"], 2, "{l}");
    assert_eq!(latest(&l)["skill"], tid);
    eprintln!("travel question active after {n} texts");
    assert_eq!(srv.get("/healthz").await.body["auto_skills"], 2);
    // Each question is answered by its own skill, whatever the text: a food
    // text is yes under the food question and no under the travel one.
    let mut local = 0;
    for t in fresh_food
        .iter()
        .chain(&distinct_texts("travel", 4, 153, "yv", 0.995))
    {
        let food = pool("food").iter().any(|w| t.split(' ').any(|x| x == *w));
        let r = ask(&srv, &food_q, t).await;
        assert_eq!(r.q("task")["skill"], fid, "{}", r.text);
        assert_eq!(r.q("task")["match"], "exact");
        assert_eq!(
            r.body["answers"]["task"]["choice"],
            if food { "yes" } else { "no" },
            "{}",
            r.text
        );
        local += usize::from(r.action() == "local");
        let r = ask(&srv, &travel_q, t).await;
        assert_eq!(r.q("task")["skill"], tid, "{}", r.text);
        assert_eq!(r.q("task")["match"], "exact");
        assert_eq!(
            r.body["answers"]["task"]["choice"],
            if food { "no" } else { "yes" },
            "{}",
            r.text
        );
        local += usize::from(r.action() == "local");
    }
    assert!(local >= 2, "no local answer of either auto-skill");
    let s = srv.get("/v1/skills").await.body;
    let autos: Vec<&str> = s["skills"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|k| k["auto"] == true)
        .map(|k| k["id"].as_str().unwrap())
        .collect();
    let mut want = vec![fid.as_str(), tid.as_str()];
    want.sort_unstable();
    assert_eq!(autos, want);
    let one = srv.get(&format!("/v1/skills/{tid}")).await.body;
    assert_eq!(one["rubric"]["instructions"], TRAVEL);
    assert_eq!(one["rubric"]["criteria"]["yes"], "it is");
}

/// A multiple-choice question with positional ids `{A, B, C, D}` whose
/// descriptions change with every request (every benchmark): every request
/// is its own contract, so nothing is ever learned as one skill — no attempt
/// runs — and `auto_max_skills` bounds the registry: past the cap the oracle
/// answers, `auto_skipped` counts and the first contracts stay (no eviction).
#[tokio::test]
async fn positional_ids_with_changing_descriptions_never_learn_and_are_bounded() {
    let positional = |i: usize| {
        json!({"type": "choice", "instructions": "Pick the correct answer.",
               "criteria": {"A": format!("answer {i}a"), "B": format!("answer {i}b"),
                            "C": format!("answer {i}c"), "D": format!("answer {i}d")}})
    };
    let mock = keyword_mock();
    let mut cfg = stand_config(&mock.url());
    cfg.learning.auto_max_skills = 3;
    let srv = Srv::new(&cfg);
    let texts = distinct_texts("cruise", 6, 171, "pq", 0.97);
    let mut ids = Vec::new();
    for (i, t) in texts.iter().enumerate().take(5) {
        let q = positional(i);
        let id = auto_id_of(&q);
        assert!(!ids.contains(&id), "every request is its own contract");
        ids.push(id);
        let r = ask(&srv, &q, t).await;
        assert_eq!(r.q("task")["match"], "untrained", "{}", r.text);
        assert_eq!(r.action(), "oracle");
        let l = srv.learning().await;
        assert_eq!(l["auto_contracts"], (i + 1).min(3) as u64, "{l}");
        assert_eq!(l["auto_skipped"], i.saturating_sub(2) as u64, "{l}");
        assert_eq!(l["examples_added"], (i + 1).min(3) as u64, "{l}");
        assert_eq!(l["attempts"], 0, "{l}");
    }
    let l = srv.learning().await;
    let listed: Vec<&str> = l["auto_skills"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_str().unwrap())
        .collect();
    let mut first = ids[..3].to_vec();
    first.sort_unstable();
    assert_eq!(listed, first, "the first contracts stay: {l}");
    assert_eq!(mock.hits(), 5);
    // The same question again (its descriptions unchanged) is the same
    // contract: a second example, no new contract, no skip.
    let r = ask(&srv, &positional(1), &texts[5]).await;
    assert_eq!(r.action(), "oracle");
    let l = srv.learning().await;
    assert_eq!(
        (
            l["auto_contracts"].as_u64(),
            l["auto_skipped"].as_u64(),
            l["examples_added"].as_u64()
        ),
        (Some(3), Some(2), Some(4)),
        "{l}"
    );
}

#[tokio::test]
async fn auto_tau_floors_a_local_answer_of_an_auto_skill() {
    // A noisy oracle (one text in eight gets the next option): the agreement
    // still passes, and T fitted on a C with wrong rows keeps p_top below 1
    // (a clean C drives T to its lower bound and every p_top to 1, DESIGN A13).
    let mock = MockOracle::start(|req| {
        let text = req.state().as_str().unwrap_or("").to_string();
        let noisy = sha256_hex(text.as_bytes()).as_bytes()[1].is_multiple_of(8);
        answer_reply(
            req,
            move |_, opts| {
                let words: Vec<&str> = text.split(' ').collect();
                let hit = opts
                    .iter()
                    .position(|o| pool_of(o).is_some_and(|p| p.iter().any(|w| words.contains(w))))
                    .unwrap_or(0);
                json!(opts[if noisy { (hit + 1) % opts.len() } else { hit }])
            },
            1e-5,
        )
    });
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let labels = ["food", "travel", "cruise"];
    let q = choice(&labels);
    let mut seen = Vec::new();
    let lessons: Vec<(&str, Vec<String>)> = [
        ("food", 41, "af"),
        ("travel", 43, "av"),
        ("cruise", 47, "ac"),
    ]
    .iter()
    .map(|&(l, seed, tag)| (l, texts_apart(l, 30, seed, tag, 0.97, &mut seen)))
    .collect();
    let longest = 30;
    let mut n = 0;
    'teach: for i in 0..longest {
        for (_, ts) in &lessons {
            ask(&srv, &q, &ts[i]).await;
            n += 1;
            if srv.learning().await["promotions"].as_u64().unwrap_or(0) >= 1 {
                break 'teach;
            }
        }
    }
    let l = srv.learning().await;
    let rec = latest(&l);
    assert_eq!(rec["outcome"], "promoted", "{rec}");
    eprintln!(
        "noisy oracle: promoted after {n} texts, agreement {}, T {}",
        rec["auto"]["agreement"]["macro"], rec["gate_after"]["temperature"]
    );
    // The least confident local answer among fresh texts of every label.
    let mut least: Option<(String, f64)> = None;
    for (label, seed, tag) in [
        ("food", 97, "zf"),
        ("travel", 98, "zv"),
        ("cruise", 99, "zc"),
    ] {
        for t in distinct_texts(label, 4, seed, tag, 0.995) {
            let r = ask(&srv, &q, &t).await;
            if r.action() == "local" {
                let p = r.q("task")["gate"]["p_top"].as_f64().unwrap();
                assert!(p >= 0.9);
                if least.as_ref().is_none_or(|(_, q)| p < *q) {
                    least = Some((t.clone(), p));
                }
            }
        }
    }
    let (text, p_top) = least.expect("a local answer");
    eprintln!("least confident local answer: p_top {p_top}");
    assert!(
        p_top < 1.0,
        "no local answer below p_top 1: the floor cannot be exercised"
    );
    let l0 = srv.learning().await;
    // Below the default floor the same text abstains: `auto_tau: 0.5` serves it.
    // Under cost-saver the floor does not apply (θ-only, spec §4.7b).
    let r = srv
        .decide(&body(
            json!(text),
            json!({"task": q.clone()}),
            Some(json!({"profile": "cost-saver"})),
        ))
        .await;
    assert_eq!(r.action(), "local", "{}", r.text);
    // Restart with the floor above that answer: it abstains, escalates and teaches.
    let mut high = cfg.clone();
    high.learning.auto_tau = (p_top as f32 + 1e-3).min(1.0);
    let srv = srv.restart(&high);
    let hits = mock.hits();
    let r = ask(&srv, &q, &text).await;
    assert_eq!(r.q("task")["match"], "exact");
    assert_eq!(r.q("task")["gate"]["accepted"], false, "{}", r.text);
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(r.q("task")["decision_path"], "escalate→oracle");
    assert_eq!(mock.hits(), hits + 1);
    let l = srv.learning().await;
    assert_eq!(
        l["examples_added"].as_u64().unwrap(),
        1,
        "one example since the restart: {l}"
    );
    assert!(l0["examples_added"].as_u64().unwrap() >= 60);
}

/// `n` texts of `label` that also carry words of `neighbour` (the stand's
/// quarantined label: its texts sit inside an active label's cloud and the
/// gate names them as that neighbour with full confidence), pairwise cos
/// φ_P < `max_cos` to `seen`. Extends `seen`.
fn texts_near(
    label: &str,
    neighbour: &str,
    n: usize,
    seed: u64,
    tag: &str,
    max_cos: f32,
    seen: &mut Vec<Vec<f32>>,
) -> Vec<String> {
    let (p, np) = (pool(label), pool(neighbour));
    let mut rng = Lcg(seed);
    let mut out: Vec<String> = Vec::new();
    let mut tries = 0;
    while out.len() < n {
        tries += 1;
        assert!(
            tries < 100_000,
            "could not find {n} '{label}' texts near '{neighbour}'"
        );
        let mut words = vec![p[rng.below(p.len())]];
        for _ in 0..2 + rng.below(2) {
            words.push(np[rng.below(np.len())]);
        }
        words.push(FILLER[rng.below(FILLER.len())]);
        let r = rng.below(words.len());
        words.swap(0, r);
        let t = format!("{} {tag}{}", words.join(" "), out.len());
        let f = phi_p(&t);
        if seen.iter().all(|q| cos(q, &f) < max_cos) {
            seen.push(f);
            out.push(t);
        }
    }
    out
}

/// DESIGN A15–A17: the auto_start gate carries the floored T; while a label
/// is quarantined, texts of it the gate accepts (as a neighbour) are explored
/// — one in four by the hash of φ_P, the oracle's answer served with the
/// flag `explore` and `gate.accepted: true`, and learned — until the label
/// holds enough examples and the next attempt activates it, after which the
/// contract explores no more and the label is answered locally.
#[tokio::test]
async fn a_quarantined_label_is_explored_until_it_activates() {
    let mock = keyword_mock();
    let mut cfg = stand_config(&mock.url());
    // A trigger every 10 new examples of a label keeps the run short.
    cfg.learning.refit_min_new = 10;
    let srv = Srv::new(&cfg);
    // `cruise` first: the keyword oracle answers the first option whose pool
    // holds a word of the text, so a cruise text in food words is `cruise`.
    let labels = ["cruise", "food", "travel"];
    let q = choice(&labels);
    let id = auto_id(&labels);
    let mut seen = Vec::new();
    let food = texts_apart("food", 25, 41, "af", 0.97, &mut seen);
    let travel = texts_apart("travel", 25, 43, "av", 0.97, &mut seen);
    let cruise = texts_near("cruise", "food", 90, 47, "ac", 0.97, &mut seen);
    let lessons = vec![
        ("food", food),
        ("travel", travel),
        ("cruise", cruise[..2].to_vec()),
    ];
    teach_contract(&srv, &q, &lessons).await;
    let l = srv.learning().await;
    let rec = latest(&l);
    assert_eq!(rec["kind"], "auto_start");
    assert_eq!(rec["outcome"], "promoted", "{rec}");
    assert_eq!(rec["auto"]["quarantined"], json!(["cruise"]));
    // A15: the recorded T is floored (a clean C fits T at its lower bound,
    // where every p_top is 1); the served gate carries it.
    let t = rec["gate_after"]["temperature"].as_f64().unwrap() as f32;
    assert!(t >= 0.02, "gate_after T {t}");
    let s = srv.get(&format!("/v1/skills/{id}")).await.body;
    assert!(s["temperature"].as_f64().unwrap() as f32 >= 0.02, "{s}");
    assert_eq!(s["quarantined_labels"], json!(["cruise"]));
    let skill = |sk: &Value| -> Value { sk["quarantined_labels"].clone() };

    // Texts of the quarantined label until it activates: accepted ones
    // answer locally as a neighbour unless explored (then the oracle's
    // `cruise` with the flag), abstentions escalate as before; both teach.
    let (mut explored, mut abstained, mut misnamed) = (0usize, 0usize, 0usize);
    let mut activated_after = None;
    for (i, t) in cruise[2..].iter().enumerate() {
        let r = ask(&srv, &q, t).await;
        assert_eq!(r.q("task")["match"], "exact", "{}", r.text);
        assert_eq!(r.q("task")["skill"], id);
        let g = &r.q("task")["gate"];
        let explore = r.flags().as_array().unwrap().iter().any(|f| f == "explore");
        match r.action() {
            "local" => {
                assert!(!explore, "{}", r.text);
                assert_eq!(g["accepted"], true);
                assert_ne!(r.body["answers"]["task"]["choice"], "cruise", "{}", r.text);
                misnamed += 1;
            }
            "oracle" => {
                assert_eq!(r.body["answers"]["task"]["choice"], "cruise", "{}", r.text);
                assert_eq!(r.q("task")["decision_path"], "escalate→oracle");
                assert_eq!(r.q("task")["certified"], false);
                if explore {
                    assert_eq!(g["accepted"], true, "{}", r.text);
                    explored += 1;
                } else {
                    assert_eq!(g["accepted"], false, "{}", r.text);
                    abstained += 1;
                }
            }
            other => panic!("unexpected action {other}: {}", r.text),
        }
        let s = srv.get(&format!("/v1/skills/{id}")).await.body;
        if skill(&s) == json!([]) {
            activated_after = Some(i + 1);
            break;
        }
    }
    eprintln!(
        "cruise: explored {explored}, abstained {abstained}, misnamed {misnamed}, activated after {activated_after:?}"
    );
    assert!(explored >= 1, "no accepted cruise text was explored");
    let n = activated_after.expect("cruise activated");
    let l = srv.learning().await;
    let rec = latest(&l);
    assert_eq!(rec["kind"], "auto_refit", "{rec}");
    assert_eq!(rec["outcome"], "promoted", "{rec}");
    assert_eq!(rec["label"], "cruise");
    assert_eq!(rec["auto"]["eligible"], json!(["cruise", "food", "travel"]));
    assert_eq!(rec["auto"]["quarantined"], json!([]));
    assert!(rec["gate_after"]["temperature"].as_f64().unwrap() as f32 >= 0.02);
    assert_eq!(l["quarantine"], json!([]));
    let s = srv.get(&format!("/v1/skills/{id}")).await.body;
    assert_eq!(s["labels"], json!(["cruise", "food", "travel"]));
    assert_eq!(s["active_labels"], 3);
    // Every escalated text (explored or abstained) became an example.
    let ex = l["auto_skills"][0]["examples"]["cruise"].as_u64().unwrap() as usize;
    assert_eq!(ex, 2 + explored + abstained, "{l}");
    assert!(n >= ex - 2);

    // Every label active: no exploration any more, and the once-quarantined
    // label is answered locally.
    let hits = mock.hits();
    let mut local_cruise = 0;
    for t in distinct_texts("cruise", 6, 99, "zc", 0.995)
        .iter()
        .chain(distinct_texts("food", 6, 97, "zf", 0.995).iter())
    {
        let r = ask(&srv, &q, t).await;
        assert!(
            !r.flags().as_array().unwrap().iter().any(|f| f == "explore"),
            "{}",
            r.text
        );
        if r.action() == "local" && r.body["answers"]["task"]["choice"] == "cruise" {
            local_cruise += 1;
        }
    }
    assert!(
        local_cruise >= 1,
        "no local cruise answer after the activation"
    );
    eprintln!(
        "after the activation: {local_cruise} local cruise answers, {} oracle calls",
        mock.hits() - hits
    );
}

#[tokio::test]
async fn a_rare_label_stays_quarantined_and_keeps_teaching() {
    let mock = keyword_mock();
    let cfg = stand_config(&mock.url());
    let srv = Srv::new(&cfg);
    let labels = ["food", "travel", "cruise"];
    let q = choice(&labels);
    let id = auto_id(&labels);
    let mut seen = Vec::new();
    let food = texts_apart("food", 25, 41, "af", 0.97, &mut seen);
    let travel = texts_apart("travel", 25, 43, "av", 0.97, &mut seen);
    let cruise = texts_apart("cruise", 8, 47, "ac", 0.97, &mut seen);
    let lessons = vec![
        ("food", food),
        ("travel", travel),
        ("cruise", cruise[..2].to_vec()),
    ];
    teach_contract(&srv, &q, &lessons).await;
    let l = srv.learning().await;
    let rec = latest(&l);
    assert_eq!(rec["kind"], "auto_start");
    assert_eq!(rec["outcome"], "promoted", "{rec}");
    assert_eq!(rec["auto"]["eligible"], json!(["food", "travel"]));
    assert_eq!(rec["auto"]["quarantined"], json!(["cruise"]));
    assert!(rec["auto"]["coverage"].as_f64().unwrap() >= 0.8);
    let s = srv.get(&format!("/v1/skills/{id}")).await.body;
    assert_eq!(s["labels"], json!(["food", "travel"]));
    assert_eq!(s["quarantined_labels"], json!(["cruise"]));
    assert_eq!(s["active_labels"], 2);
    assert_eq!(s["tasks"][0]["state"], "quarantined");
    assert_eq!(
        l["quarantine"],
        json!([{"skill": id, "label": "cruise", "examples": 2, "new": 2}])
    );

    // The whole contract is exact over the active labels; a quarantined
    // option gets probability 0; a text of it abstains and keeps teaching it.
    let mut local_food = None;
    for t in distinct_texts("food", 4, 97, "zf", 0.995) {
        let r = ask(&srv, &q, &t).await;
        assert_eq!(r.q("task")["match"], "exact", "{}", r.text);
        assert_eq!(r.q("task")["skill"], id);
        if r.action() == "local" {
            assert_eq!(r.body["answers"]["task"]["probabilities"]["cruise"], 0.0);
            local_food = Some(r);
            break;
        }
    }
    assert!(local_food.is_some());
    let mut taught = 0;
    // One cruise text in four is explored (a property of the text's features,
    // whose last bits differ between platforms), so keep asking fresh cruise
    // texts until one has reached the oracle: 30 texts leave a 0.75^30 chance
    // of none.
    let mut more = texts_apart("cruise", 24, 53, "ac2", 0.97, &mut seen).into_iter();
    let mut asked = 0;
    for t in cruise[2..]
        .iter()
        .cloned()
        .chain(std::iter::from_fn(|| more.next()))
    {
        let r = ask(&srv, &q, &t).await;
        assert_eq!(r.q("task")["match"], "exact");
        if r.action() == "oracle" {
            assert_eq!(r.body["answers"]["task"]["choice"], "cruise");
            taught += 1;
        }
        asked += 1;
        if taught >= 1 && asked >= cruise.len() - 2 {
            break;
        }
    }
    assert!(taught >= 1, "no cruise text escalated in {asked} texts");
    let l = srv.learning().await;
    let cq = l["quarantine"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["label"] == "cruise")
        .unwrap();
    assert_eq!(cq["examples"].as_u64().unwrap(), 2 + taught);
    // Feedback with the quarantined label teaches it too (a contract id).
    let r = local_food.unwrap();
    let fb = json!({"id": r.body["id"], "question": "task", "label": "cruise"});
    let f = srv.post("/v1/feedback", None, &fb).await;
    assert_eq!(f.status, 200, "{}", f.text);
    assert_eq!(
        (f.body["learned"].as_bool(), f.body["known_label"].as_bool()),
        (Some(true), Some(true))
    );
    // A request with one more id is another contract (DESIGN A18): never a
    // superset of the auto-skill, answered by the oracle and learned on its
    // own — the first skill's label set stays closed.
    let sup = choice(&["food", "travel", "cruise", "Weather"]);
    let sup_id = auto_id_of(&sup);
    let weather = distinct_texts("Weather", 1, 5, "w", 0.97).remove(0);
    let r = ask(&srv, &sup, &weather).await;
    assert_eq!(r.q("task")["match"], "untrained", "{}", r.text);
    assert_eq!(r.q("task")["skill"], Value::Null);
    assert_eq!(r.action(), "oracle");
    assert_eq!(r.body["answers"]["task"]["choice"], "Weather");
    let l2 = srv.learning().await;
    assert_eq!(
        l2["auto_contracts"].as_u64().unwrap(),
        l["auto_contracts"].as_u64().unwrap() + 1,
        "{l2}"
    );
    assert_eq!(
        l2["examples_added"].as_u64().unwrap(),
        l["examples_added"].as_u64().unwrap() + 2,
        "the feedback and the new contract's example: {l2}"
    );
    let weather_of: Vec<&str> = l2["buffer"]["labels"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["label"] == "Weather")
        .map(|c| c["skill"].as_str().unwrap())
        .collect();
    assert_eq!(weather_of, [sup_id.as_str()], "{l2}");
}

// ------------------------------------------------------------------ state-less requests (0.8.8)

/// The Decision Index kit's rendering of a classification item: `state: {}`
/// and the item text inside the instructions after a fixed prefix — a short
/// one here: the toy encoder reads 20 tokens, so a kit-length prefix would
/// leave no room for the text.
const SL_PREFIX: &str = "Q:\n";

/// A state-less choice question over `labels` (the descriptions of
/// [`choice`]) whose instructions carry `text`.
fn sl_question(labels: &[&str], text: &str) -> Value {
    let mut q = choice(labels);
    q["instructions"] = json!(format!("{SL_PREFIX}{text}"));
    q
}

/// The auto-skill id of the state-less contract over `labels` (DESIGN
/// A19.1: the criteria alone).
fn sl_auto_id(labels: &[&str]) -> String {
    cortiq_decision::manifest::stateless_auto_skill_id(
        choice(labels)["criteria"].as_object().unwrap(),
    )
}

/// Decide `text` state-less (`state: {}`) under the criteria of `labels`.
async fn ask_sl(srv: &Srv, labels: &[&str], text: &str) -> Resp {
    let r = srv
        .decide(&body(
            json!({}),
            json!({"task": sl_question(labels, text)}),
            None,
        ))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    r
}

/// An oracle that answers the `task` choice by the keywords of its
/// instructions (where a state-less request carries its text).
fn instructions_keyword_mock() -> MockOracle {
    MockOracle::start(|req| {
        let text = asked_instructions(req);
        answer_reply(
            req,
            move |_, opts| {
                let words: Vec<&str> = text.split([' ', '\n']).collect();
                for o in opts {
                    if pool_of(o).is_some_and(|p| p.iter().any(|w| words.contains(w))) {
                        return json!(o);
                    }
                }
                json!(opts[0])
            },
            1e-5,
        )
    })
}

/// `n` texts of `label` whose state-less input (prefix + text) has cos φ_P
/// < 0.995 to every text in `seen` (never a duplicate of the buffer). The
/// shared prefix pulls the toy encoder's texts above the cache's 0.97, so
/// the learning test runs without the cache (the state-less cache scope is
/// pinned by the one-off test).
fn sl_texts_apart(
    label: &str,
    n: usize,
    seed: u64,
    tag: &str,
    seen: &mut Vec<Vec<f32>>,
) -> Vec<String> {
    let p = pool(label);
    let mut rng = Lcg(seed);
    let mut out: Vec<String> = Vec::new();
    let mut tries = 0;
    while out.len() < n {
        tries += 1;
        assert!(tries < 20_000, "could not find {n} '{label}' texts");
        let mut words = Vec::new();
        for _ in 0..3 + rng.below(2) {
            words.push(p[rng.below(p.len())]);
        }
        words.push(FILLER[rng.below(FILLER.len())]);
        let t = format!("{} {tag}{}", words.join(" "), out.len());
        let f = phi_p(&format!("{SL_PREFIX}{t}"));
        if seen.iter().all(|q| cos(q, &f) < 0.995) {
            seen.push(f);
            out.push(t);
        }
    }
    out
}

/// A state-less 3-label contract (DESIGN A19/A20): the oracle answers by the
/// instructions' keywords; the contract is registered at its second sighting
/// (the first answer is not learned), promoted, and then fresh state-less
/// texts are answered locally — over `/v1/decisions` (the OpenRouter
/// adapter's path) and `/v1/systemone`; the same criteria under a non-empty
/// state are another contract; the auto-skill survives a restart.
#[tokio::test]
async fn a_stateless_contract_is_learned_from_its_instructions() {
    let labels = ["food", "travel", "cruise"];
    let sid = sl_auto_id(&labels);
    assert_ne!(sid, auto_id(&labels));
    let mut seen = Vec::new();
    let lessons: Vec<(&str, Vec<String>)> = [
        ("food", 241, "lf"),
        ("travel", 243, "lv"),
        ("cruise", 247, "lc"),
    ]
    .iter()
    .map(|&(l, seed, tag)| (l, sl_texts_apart(l, 26, seed, tag, &mut seen)))
    .collect();
    let mock = instructions_keyword_mock();
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    // Registered at the second sighting (the default is 5).
    cfg.learning.auto_min_sightings = 2;
    let jev = |o: &mut ServeOptions| o.jev_compatible = true;
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        jev,
    );

    // First sighting: the oracle answers, nothing is registered or learned.
    let r = ask_sl(&srv, &labels, &lessons[0].1[0]).await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(r.q("task")["match"], "untrained");
    assert_eq!(r.body["answers"]["task"]["choice"], "food");
    let l = srv.learning().await;
    assert_eq!(
        (
            l["auto_contracts"].as_u64(),
            l["auto_sightings"].as_u64(),
            l["auto_registered"].as_u64(),
            l["examples_added"].as_u64()
        ),
        (Some(0), Some(1), Some(0), Some(0)),
        "{l}"
    );
    // The second sighting registers it (the sighting is forgotten) and its
    // answer is the first example.
    let r = ask_sl(&srv, &labels, &lessons[1].1[0]).await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    let l = srv.learning().await;
    assert_eq!(
        (
            l["auto_contracts"].as_u64(),
            l["auto_sightings"].as_u64(),
            l["auto_registered"].as_u64(),
            l["examples_added"].as_u64()
        ),
        (Some(1), Some(0), Some(1), Some(1)),
        "{l}"
    );
    assert_eq!(l["auto_skills"][0]["id"], sid);
    assert_eq!(l["auto_skills"][0]["stateless"], true);

    // Teach the rest, interleaved, until the promotion.
    let mut taught = 2;
    'teach: for i in 0..26 {
        for (k, (label, ts)) in lessons.iter().enumerate() {
            if i == 0 && k < 2 {
                continue; // asked above
            }
            let t = &ts[i];
            let r = ask_sl(&srv, &labels, t).await;
            taught += 1;
            assert!(
                r.action() == "oracle" || r.action() == "cache",
                "{}",
                r.text
            );
            assert_eq!(r.body["answers"]["task"]["choice"], *label, "{}", r.text);
            if srv.learning().await["promotions"].as_u64().unwrap_or(0) >= 1 {
                break 'teach;
            }
        }
    }
    let l = srv.learning().await;
    assert_eq!(l["promotions"], 1, "no promotion after {taught} texts: {l}");
    let rec = latest(&l);
    assert_eq!(
        (rec["kind"].as_str(), rec["skill"].as_str()),
        (Some("auto_start"), Some(sid.as_str()))
    );
    eprintln!(
        "state-less auto_start after {taught} texts: {}",
        rec["auto"]
    );
    let one = srv.get(&format!("/v1/skills/{sid}")).await.body;
    assert_eq!(one["rubric"]["instructions"], Value::Null, "{one}");
    assert_eq!(one["rubric"]["input"], "instructions", "{one}");
    assert_eq!(one["rubric"]["criteria"]["food"], "about food");

    // Fresh state-less texts: the auto-skill, exact, never certified, local
    // at least once per label (A13: a young θ may still abstain and teach).
    let mut fresh_seen = seen.clone();
    let hits = mock.hits();
    let mut escalated = 0;
    for (label, seed, tag) in [
        ("food", 251, "zf"),
        ("travel", 253, "zv"),
        ("cruise", 257, "zc"),
    ] {
        let mut local = 0;
        for t in sl_texts_apart(label, 4, seed, tag, &mut fresh_seen) {
            let r = ask_sl(&srv, &labels, &t).await;
            assert_eq!(r.q("task")["match"], "exact", "{}", r.text);
            assert_eq!(r.q("task")["skill"], sid);
            assert_eq!(r.q("task")["certified"], false);
            assert_eq!(r.body["answers"]["task"]["choice"], label, "{}", r.text);
            match r.action() {
                "local" => local += 1,
                "oracle" => escalated += 1,
                other => assert_eq!(other, "cache", "{}", r.text),
            }
        }
        assert!(
            local >= 1,
            "no local answer of a fresh state-less {label} text"
        );
    }
    assert_eq!(mock.hits() - hits, escalated);
    // The System One surface: the same request shape, the same skill.
    let food = sl_texts_apart("food", 1, 259, "so", &mut fresh_seen).remove(0);
    let r = srv
        .post(
            "/v1/systemone",
            None,
            &json!({"state": {}, "questions": {"task": sl_question(&labels, &food)}}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["answers"]["task"]["choice"], "food", "{}", r.text);

    // The same criteria under a non-empty state: another contract (the
    // stateful key includes the instructions), untrained, registered at its
    // first sighting as in 0.8.6.
    let r = ask(&srv, &sl_question(&labels, &food), &food).await;
    assert_eq!(r.q("task")["match"], "untrained", "{}", r.text);
    assert_eq!(r.action(), "oracle");
    let l = srv.learning().await;
    assert_eq!(l["auto_contracts"], 2, "{l}");
    let r = ask(&srv, &choice(&labels), &food).await;
    assert_eq!(r.q("task")["match"], "untrained", "{}", r.text);

    // Restart from learn.log and the generation: the registry knows the
    // contract is state-less, the skill is served.
    let srv = srv.restart(&cfg);
    let l = srv.learning().await;
    let mine = l["auto_skills"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == sid)
        .cloned()
        .unwrap();
    assert_eq!(mine["stateless"], true, "{l}");
    let travel = sl_texts_apart("travel", 1, 261, "rs", &mut fresh_seen).remove(0);
    let r = ask_sl(&srv, &labels, &travel).await;
    assert_eq!(
        (r.q("task")["match"].as_str(), r.q("task")["skill"].as_str()),
        (Some("exact"), Some(sid.as_str()))
    );
}

/// One-off state-less contracts (DESIGN A20) — a multiple-choice item whose
/// option descriptions change with every question — are answered by the
/// oracle and never registered: no `learn.log` record, no example, no slot of
/// `auto_max_stateless_skills`; `auto_min_sightings: 1` registers at once.
/// Each kind has its own cap: stateful one-offs (registered at their first
/// sighting) filling `auto_max_skills` leave the state-less slots free.
#[tokio::test]
async fn a_one_off_stateless_contract_is_never_registered() {
    let item = |i: usize, text: &str| {
        json!({"type": "choice", "instructions": format!("Question: {text}\nAnswer:"),
               "criteria": {"A": format!("answer {i}a"), "B": format!("answer {i}b"),
                            "C": format!("answer {i}c"), "D": format!("answer {i}d")}})
    };
    let mock = keyword_mock();
    let mut cfg = stand_config(&mock.url());
    cfg.learning.auto_max_skills = 2;
    let srv = Srv::new(&cfg);
    let texts = distinct_texts("cruise", 5, 271, "oo", 0.97);
    for (i, t) in texts.iter().enumerate() {
        let r = srv
            .decide(&body(json!({}), json!({"task": item(i, t)}), None))
            .await;
        assert_eq!(r.status, 200, "{}", r.text);
        assert_eq!(r.action(), "oracle");
        let l = srv.learning().await;
        assert_eq!(
            (
                l["auto_contracts"].as_u64(),
                l["auto_sightings"].as_u64(),
                l["examples_added"].as_u64(),
                l["auto_skipped"].as_u64()
            ),
            (Some(0), Some(i as u64 + 1), Some(0), Some(0)),
            "{l}"
        );
    }
    // The cache scope of a state-less question is its criteria with the φ
    // of its instructions (DESIGN A19.4): the same item again is a hit.
    let hits = mock.hits();
    let r = srv
        .decide(&body(json!([]), json!({"task": item(0, &texts[0])}), None))
        .await;
    assert_eq!(r.action(), "cache", "{}", r.text);
    assert_eq!(mock.hits(), hits);
    let log = std::fs::read(srv.state_root().join("learn.log")).unwrap_or_default();
    let (recs, _) = cortiq_decision::buffer::read_records(&log);
    assert!(
        recs.iter().all(|r| !matches!(
            r,
            cortiq_decision::buffer::LogRecord::Contract(_)
                | cortiq_decision::buffer::LogRecord::Example(_)
        )),
        "{recs:?}"
    );
    // With `auto_min_sightings: 1` the first sighting registers.
    let mut once = cfg.clone();
    once.learning.auto_min_sightings = 1;
    let srv = Srv::new(&once);
    let r = srv
        .decide(&body(json!(""), json!({"task": item(9, &texts[0])}), None))
        .await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    let l = srv.learning().await;
    assert_eq!(
        (l["auto_contracts"].as_u64(), l["examples_added"].as_u64()),
        (Some(1), Some(1)),
        "{l}"
    );
    drop(srv);

    let mut caps = once.clone();
    caps.learning.auto_max_skills = 1;
    caps.learning.auto_max_stateless_skills = 1;
    let srv = Srv::new(&caps);
    let counts = |l: &Value| {
        (
            l["auto_contracts"].as_u64(),
            l["auto_skipped"].as_u64(),
            l["examples_added"].as_u64(),
        )
    };
    // Two stateful contracts: the first fills `auto_max_skills`.
    ask(&srv, &choice(&["food", "cruise"]), &texts[1]).await;
    ask(&srv, &choice(&["food", "travel"]), &texts[2]).await;
    let l = srv.learning().await;
    assert_eq!(counts(&l), (Some(1), Some(1), Some(1)), "{l}");
    // A state-less contract still registers, into its own slot ...
    let r = srv
        .decide(&body(json!({}), json!({"task": item(7, &texts[3])}), None))
        .await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    let l = srv.learning().await;
    assert_eq!(counts(&l), (Some(2), Some(1), Some(2)), "{l}");
    // ... and the next one finds `auto_max_stateless_skills` full.
    let r = srv
        .decide(&body(json!({}), json!({"task": item(8, &texts[4])}), None))
        .await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    let l = srv.learning().await;
    assert_eq!(counts(&l), (Some(2), Some(2), Some(2)), "{l}");
}

/// Stateful sightings gate (DESIGN B2): by default (`auto_min_sightings_stateful:
/// 1`) a stateful contract registers at its first sighting, as in 0.8.6, and
/// is never counted; at 3 it is counted, its first two answers are not
/// learned and it registers (and leaves the LRU) at the third.
#[tokio::test]
async fn stateful_sightings_gate_delays_registration() {
    let mock = keyword_mock();
    let cfg = stand_config(&mock.url());
    assert_eq!(cfg.learning.auto_min_sightings_stateful, 1);
    let q = choice(&["food", "cruise"]);
    let texts = distinct_texts("cruise", 3, 281, "sg", 0.97);
    let counts = |l: &Value| {
        (
            l["auto_contracts"].as_u64(),
            l["auto_sightings"].as_u64(),
            l["examples_added"].as_u64(),
        )
    };
    let srv = Srv::new(&cfg);
    assert_eq!(ask(&srv, &q, &texts[0]).await.action(), "oracle");
    let l = srv.learning().await;
    assert_eq!(counts(&l), (Some(1), Some(0), Some(1)), "{l}");
    drop(srv);

    let mut gated = cfg.clone();
    gated.learning.auto_min_sightings_stateful = 3;
    let srv = Srv::new(&gated);
    // `auto_sightings` counts contracts in the LRU, not their sightings.
    for t in &texts[..2] {
        assert_eq!(ask(&srv, &q, t).await.action(), "oracle");
        let l = srv.learning().await;
        assert_eq!(counts(&l), (Some(0), Some(1), Some(0)), "{l}");
    }
    let log = std::fs::read(srv.state_root().join("learn.log")).unwrap_or_default();
    let (recs, _) = cortiq_decision::buffer::read_records(&log);
    assert!(
        recs.iter().all(|r| !matches!(
            r,
            cortiq_decision::buffer::LogRecord::Contract(_)
                | cortiq_decision::buffer::LogRecord::Example(_)
        )),
        "{recs:?}"
    );
    assert_eq!(ask(&srv, &q, &texts[2]).await.action(), "oracle");
    let l = srv.learning().await;
    assert_eq!(counts(&l), (Some(1), Some(0), Some(1)), "{l}");
}

/// The kit's validator (`decision_index/engines/base.py` `validate`) on a
/// choice answer: the chosen id is an option, and `probabilities` holds every
/// option, each in [0, 1], summing to 1 within 0.01.
fn kit_valid_choice(q: &Value, a: &Value) {
    let criteria = q["criteria"].as_object().unwrap();
    assert_eq!(a["type"], "choice", "{a}");
    assert!(criteria.contains_key(a["choice"].as_str().unwrap()), "{a}");
    let p = a["probabilities"].as_object().expect("probabilities");
    let keys: Vec<&String> = p.keys().collect();
    assert_eq!(keys, criteria.keys().collect::<Vec<_>>(), "{a}");
    let sum: f64 = p.values().map(|v| v.as_f64().unwrap()).sum();
    assert!(
        p.values()
            .all(|v| (0.0..=1.0).contains(&v.as_f64().unwrap()))
    );
    assert!((sum - 1.0).abs() <= 0.01, "{a}");
}

/// System One oracle and cache choice answers carry the one-hot distribution
/// (the chosen option 1, the others 0, confidence 1) when the oracle gives
/// none, so the Decision Index kit's validator accepts them; the native
/// answer is the same (0.8.8, DESIGN C3).
/// The kit's default model name `default` is accepted on System One.
#[tokio::test]
async fn systemone_oracle_answers_carry_the_one_hot_distribution() {
    let mock = keyword_mock();
    let cfg = stand_config(&mock.url());
    let jev = |o: &mut ServeOptions| o.jev_compatible = true;
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        jev,
    );
    let q = choice(&["food", "travel", "cruise"]);
    let text = &fresh()[0];
    let ask_s1 = |model: &'static str| {
        let req = json!({"model": model, "state": text, "questions": {"task": q.clone()}});
        let srv = &srv;
        async move { srv.post("/v1/systemone", None, &req).await }
    };
    // The oracle's answer.
    let r = ask_s1("default").await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(mock.hits(), 1);
    let a = &r.body["answers"]["task"];
    kit_valid_choice(&q, a);
    assert_eq!(a["confidence"], 1, "{a}");
    let chosen = a["choice"].as_str().unwrap();
    assert_eq!(a["probabilities"][chosen], 1, "{a}");
    // The same item again: the cache's answer, completed alike.
    let r = ask_s1("jev-latest").await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(mock.hits(), 1, "{}", r.text);
    kit_valid_choice(&q, &r.body["answers"]["task"]);
    assert_eq!(r.body["answers"]["task"], *a);
    // The native surface answers alike (0.8.8, DESIGN C3).
    let r = ask(&srv, &q, text).await;
    assert_eq!(r.action(), "cache", "{}", r.text);
    assert_eq!(r.body["answers"]["task"], *a);
}

/// Capacity errors (DESIGN A21): a state-less request whose instructions
/// exceed the state limit is 422 with the Decision Index kit's marker on the
/// System One surface and keeps 400 (with the marker) on `/v1/decisions`;
/// an oracle that says the prompt does not fit its context answers 422 with
/// the marker on both, and a run of such calls does not stop the oracle.
#[tokio::test]
async fn capacity_errors_carry_the_kit_marker() {
    const MARKER: &str = "maximum context length";
    let overflow = r#"{"error":{"message":"This endpoint's maximum context length is 163840 tokens. However, you requested about 200513 tokens (200449 of text input, 64 of tool input).","code":400}}"#;
    let mock = MockOracle::start(move |_| raw_reply(400, overflow));
    let mut cfg = stand_config(&mock.url());
    cfg.limits.state_bytes = 1024;
    cfg.oracle.max_errors = 2;
    let jev = |o: &mut ServeOptions| o.jev_compatible = true;
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        jev,
    );
    let big = "word ".repeat(400);
    let q = sl_question(&["food", "travel"], &big);
    let s1 = srv
        .post(
            "/v1/systemone",
            None,
            &json!({"state": {}, "questions": {"task": q.clone()}}),
        )
        .await;
    assert_eq!(s1.status, 422, "{}", s1.text);
    assert!(s1.text.contains(MARKER), "{}", s1.text);
    assert_eq!(s1.body["error"]["code"], "INVALID_REQUEST");
    let native = srv
        .decide(&body(json!({}), json!({"task": q.clone()}), None))
        .await;
    assert_eq!(native.error(), (400, "INVALID_REQUEST".into()));
    assert!(native.text.contains(MARKER), "{}", native.text);
    assert_eq!(
        native.body["error"]["metadata"]["details"]["capacity"],
        true
    );
    // The same instructions under a state: no limit of theirs (as 0.8.6);
    // the oracle then says the prompt is too long.
    assert_eq!(mock.hits(), 0);
    for i in 0..3 {
        let r = srv
            .decide(&body(
                json!(format!("text {i}")),
                json!({"task": q.clone()}),
                None,
            ))
            .await;
        assert_eq!(
            r.error(),
            (422, "UNSUPPORTED_QUESTION".into()),
            "{}",
            r.text
        );
        assert!(r.text.contains(MARKER), "{}", r.text);
    }
    assert_eq!(mock.hits(), 3);
    let st = srv.oracle_state();
    assert!(st.is_null() || st["stop_reason"].is_null(), "{st}");
    assert_eq!(status(&srv).await, "ready");
    let small = sl_question(&["food", "travel"], "a short one");
    let s2 = srv
        .post(
            "/v1/systemone",
            None,
            &json!({"state": {}, "questions": {"task": small}}),
        )
        .await;
    assert_eq!(s2.status, 422, "{}", s2.text);
    assert!(s2.text.contains(MARKER), "{}", s2.text);
    assert_eq!(s2.body["error"]["code"], "UNSUPPORTED_QUESTION");
    let ledger = srv.ledger();
    assert!(
        ledger.iter().any(|l| l["error"] == "context_length"),
        "{ledger:?}"
    );
}

/// The question `task` as the oracle request carried it (the system
/// message's last line).
fn asked_question(req: &MockRequest) -> Value {
    let v = req.json();
    let system = v["messages"][0]["content"].as_str().unwrap();
    let qs: Value = serde_json::from_str(system.rsplit('\n').next().unwrap()).unwrap();
    qs["questions"]["task"].clone()
}

/// PII in a question's instructions and in its criteria's descriptions is
/// redacted in the oracle request (DESIGN B4) — the option ids are intact —
/// while the cache scope and the auto-skill key are those of the question as
/// asked: the same question with `allow_pii_egress` (nothing redacted) hits
/// the cache entry of the redacted call, and the registered auto-skill is the
/// original contract's, stateful and state-less alike.
#[tokio::test]
async fn pii_in_instructions_and_descriptions_is_redacted_on_egress_only() {
    let text = "cruise ship yacht harbor";
    let q = json!({"type": "choice",
                   "instructions": "Which topic? Escalations go to ops.lead@example.com",
                   "criteria": {"food": "about food (chef@example.com)",
                                "cruise": "about cruise, call +1 (555) 123-4567"}});
    let mock = MockOracle::answering("cruise");
    let srv = Srv::new(&stand_config(&mock.url()));
    let r = srv
        .decide(&body(json!(text), json!({"task": q.clone()}), None))
        .await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(r.flags(), json!(["pii_redacted"]));
    assert_eq!(r.body["answers"]["task"]["choice"], "cruise", "{}", r.text);
    let sent = &mock.requests()[0];
    assert_eq!(sent.state(), json!(text));
    let asked = asked_question(sent);
    assert_eq!(
        asked["instructions"],
        json!("Which topic? Escalations go to [REDACTED]")
    );
    assert_eq!(
        asked["criteria"],
        json!({"food": "about food ([REDACTED])", "cruise": "about cruise, call [REDACTED]"})
    );
    let mut ids: Vec<&String> = asked["criteria"].as_object().unwrap().keys().collect();
    ids.sort_unstable();
    assert_eq!(ids, ["cruise", "food"]);
    let raw = String::from_utf8_lossy(&sent.body);
    assert!(
        !raw.contains("example.com") && !raw.contains("4567"),
        "{raw}"
    );
    // The learning key is the original contract's.
    let l = srv.learning().await;
    assert_eq!(l["auto_skills"][0]["id"], json!(auto_id_of(&q)), "{l}");
    // The cache scope too: the unredacted question finds the entry.
    let r = srv
        .decide(&body(
            json!(text),
            json!({"task": q.clone()}),
            Some(json!({"allow_pii_egress": true})),
        ))
        .await;
    assert_eq!(r.action(), "cache", "{}", r.text);
    assert_eq!(r.flags(), json!([]));
    assert_eq!(mock.hits(), 1);
    // With `allow_pii_egress` nothing is redacted.
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    let srv = Srv::new(&cfg);
    let r = srv
        .decide(&body(
            json!(text),
            json!({"task": q.clone()}),
            Some(json!({"allow_pii_egress": true})),
        ))
        .await;
    assert_eq!(r.flags(), json!([]));
    let asked = asked_question(&mock.requests()[1]);
    assert_eq!(
        (&asked["instructions"], &asked["criteria"]),
        (&q["instructions"], &q["criteria"])
    );

    // State-less: the descriptions are redacted as well, the key is the
    // original criteria's.
    let mut once = stand_config(&mock.url());
    once.learning.auto_min_sightings = 1;
    let srv = Srv::new(&once);
    let mut sl = q.clone();
    sl["instructions"] = json!(format!("{SL_PREFIX}{text}"));
    let r = srv
        .decide(&body(json!({}), json!({"task": sl.clone()}), None))
        .await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(r.flags(), json!(["pii_redacted"]));
    let asked = asked_question(&mock.requests()[2]);
    assert_eq!(asked["instructions"], json!(format!("{SL_PREFIX}{text}")));
    assert_eq!(
        asked["criteria"],
        json!({"food": "about food ([REDACTED])", "cruise": "about cruise, call [REDACTED]"})
    );
    let l = srv.learning().await;
    assert_eq!(
        l["auto_skills"][0]["id"],
        json!(cortiq_decision::manifest::stateless_auto_skill_id(
            q["criteria"].as_object().unwrap()
        )),
        "{l}"
    );
}

/// A state-less request's instructions are its input: PII in them is
/// redacted in the oracle request like a state (DESIGN A19.3), unless the
/// request allows its egress; the state `{}` is sent as is.
#[tokio::test]
async fn pii_in_stateless_instructions_is_redacted() {
    let text = "cruise ship yacht harbor mail john.doe@example.com call +15551234567";
    let mock = MockOracle::answering("travel");
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    let srv = Srv::new(&cfg);
    let q = sl_question(&["food", "travel"], text);
    let r = srv
        .decide(&body(json!({}), json!({"task": q.clone()}), None))
        .await;
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(r.flags(), json!(["pii_redacted"]));
    let sent = &mock.requests()[0];
    assert_eq!(sent.state(), json!({}));
    assert_eq!(
        asked_instructions(sent),
        format!("{SL_PREFIX}cruise ship yacht harbor mail [REDACTED] call [REDACTED]")
    );
    assert!(!String::from_utf8_lossy(&sent.body).contains("john.doe"));
    let r = srv
        .decide(&body(
            json!({}),
            json!({"task": q}),
            Some(json!({"allow_pii_egress": true})),
        ))
        .await;
    assert_eq!(r.flags(), json!([]));
    assert_eq!(
        asked_instructions(&mock.requests()[1]),
        format!("{SL_PREFIX}{text}")
    );
}

// ------------------------------------------------------------------ description matching (C2)

/// A Decision Index kit row (`layout.py choice_row`): `state: {}`, the text
/// after a lead-in line in the instructions, positional ids `option_N` and
/// the descriptions given.
fn kit_row(lead_in: &str, text: &str, descriptions: &[&str]) -> Value {
    let mut c = Map::new();
    for (i, d) in descriptions.iter().enumerate() {
        c.insert(format!("option_{i}"), json!(d));
    }
    json!({"type": "choice", "instructions": format!("{lead_in}:\n{text}"), "criteria": c})
}

/// The id of the option of `q` described by `description`.
fn option_of(q: &Value, description: &str) -> String {
    q["criteria"]
        .as_object()
        .unwrap()
        .iter()
        .find(|(_, d)| d.as_str() == Some(description))
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| panic!("no option '{description}' in {q}"))
}

/// Dev texts of `topics` its gate accepts, with their labels.
async fn accepted_dev(srv: &Srv, n: usize) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (t, l) in &toy().dev {
        let mut b = topics_body(t);
        b["cmf"] = json!({"oracle": false});
        let r = srv.decide(&b).await;
        if r.action() == "local" && r.body["answers"]["task"]["choice"] == json!(l) {
            out.push((t.clone(), l.clone()));
            if out.len() == n {
                break;
            }
        }
    }
    assert_eq!(out.len(), n, "not enough accepted dev texts");
    out
}

/// Kit-format BANKING77/CLINC rows (DESIGN C1/C2) reach the data skill whose
/// labels their descriptions name: answered LOCALLY in the row's option ids
/// (no oracle call) on System One and the native surface alike, the kit's
/// validator satisfied; a row with the out-of-scope option is answered with
/// it, locally, when the gate rejects the text; without such an option a
/// rejected text goes to the oracle and its answer teaches the skill the
/// option's label, never the option id; descriptions two skills share are no
/// match (the oracle answers).
#[tokio::test]
async fn kit_rows_reach_data_skills_through_their_descriptions() {
    let mock = MockOracle::answering("unused");
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    let jev = |o: &mut ServeOptions| o.jev_compatible = true;
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        jev,
    );
    // The row on System One (the kit's request) and on the native surface
    // (its diagnostics): the answers, and the native question block.
    let ask_both = |q: &Value| {
        let s1 = json!({"model": "default", "state": {}, "questions": {"q1": q.clone()}});
        let native = body(json!({}), json!({"q1": q.clone()}), None);
        let srv = &srv;
        async move {
            let a = srv.post("/v1/systemone", None, &s1).await;
            assert_eq!(a.status, 200, "{}", a.text);
            let n = srv.decide(&native).await;
            assert_eq!(n.status, 200, "{}", n.text);
            (a.body["answers"]["q1"].clone(), n)
        }
    };
    // BANKING77-style: the label names as descriptions, in another order.
    let banking = ["travel", "Weather", "cards", "billing"];
    for (text, label) in accepted_dev(&srv, 4).await {
        let q = kit_row(
            "Classify the banking intent of this user request",
            &text,
            &banking,
        );
        let (a, n) = ask_both(&q).await;
        kit_valid_choice(&q, &a);
        assert_eq!(a["choice"], json!(option_of(&q, &label)), "{a}");
        assert_eq!(n.body["answers"]["q1"], a);
        let m = n.q("q1");
        assert_eq!(
            (&m["action"], &m["match"], &m["by"], &m["skill"]),
            (
                &json!("local"),
                &json!("exact"),
                &json!("descriptions"),
                &json!("topics")
            ),
            "{m}"
        );
        assert_eq!(m["certified"], false, "a state-less row is never certified");
    }
    assert_eq!(mock.hits(), 0);

    // CLINC-style: the labels with spaces and the out-of-scope option.
    let oos = "out of scope: none of the listed intents";
    let clinc = ["weather", "billing", oos, "cards", "travel"];
    let lead = "Classify the intent of this user request, or choose out of scope if none applies";
    let (text, label) = accepted_dev(&srv, 1).await.remove(0);
    let q = kit_row(lead, &text, &clinc);
    let (a, n) = ask_both(&q).await;
    kit_valid_choice(&q, &a);
    assert_eq!(a["choice"], json!(option_of(&q, &label.to_lowercase())));
    assert_eq!(n.q("q1")["decision_path"], "router:uncertified");
    // A text the gate rejects: the none option, locally, never escalated.
    for text in &rejected()[..3] {
        let q = kit_row(lead, text, &clinc);
        let (a, n) = ask_both(&q).await;
        kit_valid_choice(&q, &a);
        assert_eq!(a["choice"], json!(option_of(&q, oos)), "{a}");
        let m = n.q("q1");
        assert_eq!(
            (&m["action"], &m["decision_path"], &m["certified"]),
            (&json!("local"), &json!("router:none_option"), &json!(false)),
            "{m}"
        );
        assert_eq!(m["gate"]["accepted"], false, "{m}");
        // The none option holds 1 − p_top of the rejected winner.
        let p_top = m["gate"]["p_top"].as_f64().unwrap();
        let p_none = a["probabilities"][option_of(&q, oos)].as_f64().unwrap();
        assert!((p_none - (1.0 - p_top)).abs() < 1e-4, "{a} {m}");
    }
    assert_eq!(mock.hits(), 0, "the none option never reaches the oracle");

    // No none option: a rejected text escalates; the oracle answers in the
    // row's ids and the example is the option's label.
    let q = kit_row(lead, &rejected()[3], &banking);
    let (a, n) = ask_both(&q).await;
    assert_eq!(n.q("q1")["action"], "oracle", "{}", n.text);
    kit_valid_choice(&q, &a);
    assert_eq!(a["choice"], "option_0");
    assert_eq!(mock.hits(), 2);
    let l = srv.learning().await;
    let taught = l["buffer"]["labels"].as_array().unwrap();
    assert!(
        taught
            .iter()
            .any(|x| x["skill"] == "topics" && x["label"] == "travel"),
        "{l}"
    );
    assert!(
        taught
            .iter()
            .all(|x| !x["label"].as_str().unwrap().starts_with("option_")),
        "{l}"
    );

    // Feedback names an option id: the example is its label; the none
    // option teaches nothing.
    let q = kit_row(lead, &rejected()[5], &clinc);
    let (_, n) = ask_both(&q).await;
    assert_eq!(n.q("q1")["decision_path"], "router:none_option");
    let id = n.request_id().to_string();
    let fb = |label: String| json!({"id": id, "question": "q1", "label": label});
    let r = srv
        .post("/v1/feedback", None, &fb(option_of(&q, oos)))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(
        (&r.body["learned"], &r.body["accepted"]),
        (&json!(false), &json!(false))
    );
    let (_, n) = ask_both(&q).await;
    let id = n.request_id().to_string();
    let fb = |label: String| json!({"id": id, "question": "q1", "label": label});
    let r = srv
        .post("/v1/feedback", None, &fb(option_of(&q, "cards")))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["learned"], true, "{}", r.text);
    let l = srv.learning().await;
    assert!(
        l["buffer"]["labels"]
            .as_array()
            .unwrap()
            .iter()
            .any(|x| x["skill"] == "topics" && x["label"] == "cards"),
        "{l}"
    );

    // Descriptions two skills share (billing, cards): no description match.
    let q = kit_row(lead, &rejected()[4], &["billing", "cards"]);
    let (_, n) = ask_both(&q).await;
    assert_eq!(n.q("q1")["match"], "untrained", "{}", n.text);
    assert_eq!(n.q("q1")["action"], "oracle");
    assert!(n.q("q1").get("by").is_none());
}

// ------------------------------------------------------------------ oracle distributions (C3)

/// An oracle answering with distributions (DESIGN C3): a choice states its
/// first option but lists the last at 0.6 and the first at 0.3; a score
/// states level 0 and lists 0.1/0.7/0.2; a noul states true with p 0.8.
fn distribution_mock() -> MockOracle {
    MockOracle::start(|req| {
        let v = req.json();
        let schema = &v["response_format"]["json_schema"]["schema"];
        let mut out = Map::new();
        for q in schema["required"].as_array().unwrap() {
            let qid = q.as_str().unwrap();
            let p = &schema["properties"][qid]["properties"];
            let a = if let Some(c) = p.get("choice") {
                let opts = c["enum"].as_array().unwrap();
                let (first, last) = (&opts[0], &opts[opts.len() - 1]);
                json!({"choice": first, "probabilities": [{"id": last, "p": 0.6}, {"id": first, "p": 0.3}]})
            } else if p.get("score").is_some() {
                json!({"score": 0, "probabilities": [0.1, 0.7, 0.2]})
            } else {
                json!({"noul": true, "p_true": 0.8})
            };
            out.insert(qid.to_string(), a);
        }
        MockReply {
            status: 200,
            body: completion(&Value::Object(out).to_string(), json!(2e-5), ORACLE_MODEL),
            delay: Duration::ZERO,
        }
    })
}

/// Oracle distributions end to end (DESIGN C3): normalized (the argmax wins
/// over the stated verdict, the unlisted share the rest), on System One in
/// Jev's forms that the kit's validator accepts (a noul is p(true)), on
/// `/v1/decisions` with `probabilities` and `confidence` = p(choice) beside
/// the documented verdicts; the cache answers a repeat with the stored
/// distribution, also after a restart (replayed from `learn.log`); the
/// example learned is the argmax label.
#[tokio::test]
async fn oracle_distributions_reach_every_surface_and_the_cache() {
    let mock = distribution_mock();
    let cfg = stand_config(&mock.url());
    let jev = |o: &mut ServeOptions| o.jev_compatible = true;
    let srv = Srv::open_with(
        &toy().path,
        tempfile::tempdir().unwrap(),
        &cfg,
        test_key(),
        jev,
    );
    let task = choice(&TOPICS);
    let noul = json!({"type": "noul", "instructions": "Is it urgent?"});
    let text = &rejected()[0];
    let s1 = json!({"model": "default", "state": text, "questions": {"task": task.clone(), "urgent": noul}});
    let r = srv.post("/v1/systemone", None, &s1).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(mock.hits(), 1);
    let sent = mock.requests()[0].json();
    assert!(
        sent["messages"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with(cortiq_decision::oracle::SYSTEM_TYPED_P)
    );
    let a = &r.body["answers"]["task"];
    kit_valid_choice(&task, a);
    assert_eq!(a["choice"], "travel", "the argmax, not the stated Weather");
    assert_eq!(a["confidence"].as_f64(), Some(0.6), "{a}");
    assert_eq!(a["probabilities"]["Weather"].as_f64(), Some(0.3));
    assert_eq!(a["probabilities"]["billing"].as_f64(), Some(0.05), "{a}");
    assert_eq!(
        r.body["answers"]["urgent"],
        json!({"type": "noul", "noul": 0.8})
    );

    // The native surface: the same distribution, the noul verdict beside it,
    // a score with its levels' probabilities. A cache answer for the choice
    // and the noul (the same text and contracts), the oracle for the score.
    let score =
        json!({"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]});
    let native = body(
        json!(text),
        json!({"task": task.clone(), "urgent": json!({"type": "noul", "instructions": "Is it urgent?"}), "level": score}),
        None,
    );
    let r = srv.decide(&native).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(mock.hits(), 2);
    assert_eq!(r.q("task")["action"], "cache");
    assert_eq!(r.body["answers"]["task"], *a);
    assert_eq!(
        r.body["answers"]["urgent"],
        json!({"type": "noul", "noul": 1, "value_semantics": "boolean_verdict_not_probability", "probability": 0.8})
    );
    let l = &r.body["answers"]["level"];
    assert_eq!(l["score"], 1, "{l}");
    assert_eq!(l["probabilities"]["1"].as_f64(), Some(0.7), "{l}");
    assert_eq!(l["confidence"].as_f64(), Some(0.7), "{l}");

    // The example is the argmax label.
    let learning = srv.learning().await;
    assert!(
        learning["buffer"]["labels"]
            .as_array()
            .unwrap()
            .iter()
            .any(|x| x["skill"] == "topics" && x["label"] == "travel"),
        "{learning}"
    );

    // A restart replays the cache with its distributions.
    let srv = srv.restart(&cfg);
    let r = srv.decide(&topics_body(text)).await;
    assert_eq!(r.action(), "cache", "{}", r.text);
    assert_eq!(r.body["answers"]["task"], *a);
    assert_eq!(mock.hits(), 2);
}

// ------------------------------------------------------------------ oracle reasoning (C4)

/// `oracle.reasoning` (DESIGN C4): the request enables OpenRouter's
/// reasoning with the effort (the text excluded) and raises `max_tokens` by
/// `reasoning_max_tokens` — the mock inspects the body —, the reservation
/// follows it, and the reasoning tokens the usage reports reach the ledger
/// and `cmf.usage.oracle`, the cost being OpenRouter's `usage.cost`.
#[tokio::test]
async fn reasoning_effort_is_sent_and_accounted() {
    let mock = MockOracle::start(|req| {
        let v = req.json();
        // Only a reasoning request is answered.
        if v["reasoning"] != json!({"effort": "high", "exclude": true})
            || v["max_tokens"] != json!(64 + 128 + 2048)
        {
            return raw_reply(400, r#"{"error":{"message":"not the expected body"}}"#);
        }
        let content = verdicts(req, |_, o| pick(o, "travel")).to_string();
        let body = serde_json::to_vec(&json!({
            "id": "gen-mock", "model": ORACLE_MODEL, "provider": "Mock",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 812, "cost": 4.2e-4,
                      "completion_tokens_details": {"reasoning_tokens": 800}},
        }))
        .unwrap();
        MockReply {
            status: 200,
            body,
            delay: Duration::ZERO,
        }
    });
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.reasoning = "high".into();
    cfg.oracle.reasoning_max_tokens = 2048;
    let srv = Srv::new(&cfg);
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(r.body["answers"]["task"]["choice"], "travel");
    let u = &r.body["cmf"]["usage"]["oracle"];
    assert_eq!(
        (
            &u["output_tokens"],
            &u["reasoning_tokens"],
            u["cost"].as_f64()
        ),
        (&json!(812), &json!(800), Some(4.2e-4)),
        "{u}"
    );
    let ledger = srv.ledger();
    assert_eq!(ledger[0]["max_tokens"], 64 + 128 + 2048);
    let reserved = ledger[0]["reserved_usd"].as_f64().unwrap();
    let body_len = mock.requests()[0].body.len();
    let expect = cortiq_decision::oracle::reservation_usd(body_len, 64 + 128 + 2048, (0.1, 0.5));
    assert!((reserved - expect).abs() < 1e-12, "{reserved} {expect}");
    assert_eq!(ledger[1]["reasoning_tokens"], 800);
    assert_eq!(ledger[1]["cost_usd"].as_f64(), Some(4.2e-4));
    let st = srv.admin("GET", "/v1/admin/oracle", None).await;
    assert_eq!(st.body["reasoning"], "high", "{}", st.text);
}

/// A reasoning call cut by its token allowance (`finish_length`, billed) is
/// asked once more without reasoning (DESIGN C5): the question is answered
/// by the direct call, both calls are in the ledger, and the stop rules see
/// the answered one last (no consecutive error is left).
#[tokio::test]
async fn a_reasoning_call_cut_by_length_is_answered_without_reasoning() {
    let mock = MockOracle::start(|req| {
        let v = req.json();
        let body = if v["reasoning"] == json!({"effort": "low", "exclude": true}) {
            json!({
                "id": "gen-mock", "model": ORACLE_MODEL, "provider": "Mock",
                "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": ""}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 2048, "cost": 3.0e-4,
                          "completion_tokens_details": {"reasoning_tokens": 2048}},
            })
        } else if v["reasoning"] == json!({"enabled": false}) && v["max_tokens"] == json!(64 + 128) {
            let content = verdicts(req, |_, o| pick(o, "travel")).to_string();
            json!({
                "id": "gen-mock", "model": ORACLE_MODEL, "provider": "Mock",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 12, "cost": 1.0e-5},
            })
        } else {
            return raw_reply(400, r#"{"error":{"message":"not the expected body"}}"#);
        };
        MockReply {
            status: 200,
            body: serde_json::to_vec(&body).unwrap(),
            delay: Duration::ZERO,
        }
    });
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.reasoning = "low".into();
    cfg.oracle.reasoning_max_tokens = 2048;
    let srv = Srv::new(&cfg);
    let r = srv.decide(&topics_body(&rejected()[0])).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.action(), "oracle", "{}", r.text);
    assert_eq!(r.body["answers"]["task"]["choice"], "travel");
    assert_eq!(mock.requests().len(), 2, "one reasoning call, one direct call");
    let ledger = srv.ledger();
    let statuses: Vec<&str> = ledger.iter().map(|l| l["status"].as_str().unwrap()).collect();
    assert_eq!(statuses, ["reserved", "failed_billed", "reserved", "settled"], "{ledger:?}");
    assert_eq!(ledger[1]["error"], "finish_length");
    assert_eq!(ledger[2]["max_tokens"], 64 + 128);
    let st = srv.admin("GET", "/v1/admin/oracle", None).await;
    assert_eq!(st.body["consecutive_errors"], 0, "{}", st.text);
}
