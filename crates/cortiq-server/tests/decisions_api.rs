//! The decisions HTTP API end to end on the toy decision file (spec
//! decision-v4 §6.4, the router API of §4.15), hermetic: the router is driven
//! in-process (`oneshot`), one test serves on a loopback socket, no oracle.
//!
//! The toy file is built once per test binary from the toy encoder of
//! `cortiq-decision` (`tests/fixtures/toy/encoder`): skills `topics` {Weather,
//! billing, cards, travel} and `shop` {billing, cards, food}.
//!
//! * 401: key missing, wrong, expired, revoked; the open mode only on loopback
//!   and only while there is no key;
//! * 429: two requests a minute (`Retry-After`); `max_inflight` overflow while a
//!   request is on the blocking pool (the runtime keeps serving);
//! * 402: decision quota, token quota, credit;
//! * 413 (from `Content-Length` and from a streamed body), 400 (JSON, unknown
//!   keys, duplicate criteria, content type), 404 model, JSON 404/405;
//! * OpenRouter's optional fields are accepted;
//! * skill matching over HTTP: exact, subset, superset, untrained (422 with a
//!   reason per question);
//! * metering: every response's usage, the ledger rows and `/v1/usage` sums;
//!   `/v1/usage` shows only the caller's account;
//! * admin: create, list, revoke (account and hash); disabled without a token;
//! * `/v1/models` shape and price strings, healthz, readyz, metrics, no CORS
//!   header, `x-request-id` on every response;
//! * the request log has the id, status, latency and account, and no state text,
//!   key or admin token (child process with the admin token in its environment);
//! * the router API: every field of router `api.rs` with its JSON type, the
//!   router-client and `scripts/eval.py` recipes, invariants, `top_k`,
//!   taxonomies, batch, bring-your-own embedding, router feedback and error
//!   envelope, escalations scoped to the caller.
//!
//! Every request of [`Srv`] carries `x-cmf-extensions: 1`, so router-surface
//! answers include the opt-in `cmf` diagnostics these tests read (action,
//! certified, gate, totals, error metadata); the exact default router shapes
//! are checked in `router_compat.rs`.

#[path = "support/toy_dir.rs"]
mod toy_dir;

use axum::body::Body;
use axum::http::{HeaderMap, Request};
use cortiq_decision::build::{self, TrainOptions};
use cortiq_decision::cascade::CascadeOptions;
use cortiq_decision::config::Config;
use cortiq_decision::container::{DecisionModel, Verify};
use cortiq_decision::keys::{KEYS_FILE_VERSION, KeyRecord, KeysFile, hash_key, now_unix};
use cortiq_decision::oracle::KeyLookup;
use cortiq_decision::protocol::ApiError;
use cortiq_decision::protocol::FeedbackRequest;
use cortiq_decision::service::{
    Action, AdminCommand, DecisionService, Escalation, EscalationResult, Escalator, LoadedModel,
    ModelHandle, Principal, RefusalReason, Resolution, Resolved,
};
use cortiq_decision::signal::SignalEncoder;
use cortiq_server::decisions::{self, DecisionServer, DecisionState, ServeOptions};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tower::ServiceExt;

const EPOCH: u64 = 1_790_000_000;
const TOPICS: [&str; 4] = ["Weather", "billing", "cards", "travel"];
const SHOP: [&str; 3] = ["billing", "cards", "food"];
const ADMIN: &str = "admin-token-for-tests-0123456789abcdef";
/// A text of words from no training pool: the `topics` gate rejects it.
const REJECTED: &str = "cruise ship cabin deck please";

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

/// The toy encoder export of the `cortiq-decision` crate.
fn toy_encoder_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../cortiq-decision/tests/fixtures/toy/encoder")
}

fn toy() -> &'static Toy {
    static TOY: OnceLock<Toy> = OnceLock::new();
    TOY.get_or_init(|| {
        let dir = toy_dir::toy_dir("toy");
        let d = dir.as_path();
        let enc = d.join("enc.cmf");
        build::init_encoder(&toy_encoder_dir(), &enc, Some(EPOCH)).expect("init toy encoder");
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

/// A dev text the `topics` gate accepts (found with the service directly).
fn accepted() -> &'static str {
    static A: OnceLock<String> = OnceLock::new();
    A.get_or_init(|| {
        let m = DecisionModel::open(&toy().path, Verify::Light).unwrap();
        let h = Arc::new(ModelHandle::new(LoadedModel::new(m).unwrap()));
        let svc = DecisionService::open(h, Config::default(), None)
            .unwrap()
            .with_loopback(true);
        toy()
            .dev
            .iter()
            .map(|(t, _)| t.clone())
            .find(|t| {
                let b = serde_json::to_vec(&topics_body(t)).unwrap();
                svc.decide_body(&b, &Principal::open()).unwrap().questions[0].action
                    == Action::Local
            })
            .expect("a dev text the gate accepts")
    })
}

// ------------------------------------------------------------------ server + HTTP

fn no_key() -> KeyLookup {
    Arc::new(|_: &str| None)
}

/// The default test configuration (learning synchronous, oracle off).
fn cfg() -> Config {
    let mut c = Config::default();
    c.learning.synchronous = true;
    c
}

struct Srv {
    server: Option<DecisionServer>,
    app: Option<axum::Router>,
    dir: tempfile::TempDir,
}

impl Srv {
    fn open(cfg: Config) -> Self {
        Self::open_with(cfg, true, Some(ADMIN), tempfile::tempdir().unwrap())
    }

    fn open_with(cfg: Config, loopback: bool, admin: Option<&str>, dir: tempfile::TempDir) -> Self {
        let mut o = ServeOptions::new(&toy().path, cfg);
        o.state_dir = Some(dir.path().join("state"));
        o.addr = if loopback {
            "127.0.0.1:0".parse().unwrap()
        } else {
            "0.0.0.0:0".parse().unwrap()
        };
        o.cascade = CascadeOptions {
            key: no_key(),
            threads: 2,
            created_unix: Some(EPOCH),
        };
        o.admin_token = admin.map(str::to_string);
        let server = DecisionServer::open(&o).expect("open the decision server");
        let app = server.router();
        Self {
            server: Some(server),
            app: Some(app),
            dir,
        }
    }

    fn server(&self) -> &DecisionServer {
        self.server.as_ref().unwrap()
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

    /// Create a key through the admin API; returns (raw key, account).
    async fn key(&self, spec: Value) -> (String, String) {
        let r = self.admin("POST", "/v1/admin/keys", Some(&spec)).await;
        assert_eq!(r.status, 200, "{}", r.text);
        (
            r.body["key"].as_str().unwrap().to_string(),
            r.body["account"].as_str().unwrap().to_string(),
        )
    }

    fn close(mut self) -> tempfile::TempDir {
        self.app.take();
        self.server.take().unwrap().close().unwrap();
        self.dir
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

    fn request_id(&self) -> &str {
        self.header("x-request-id").expect("x-request-id")
    }

    /// OpenRouter error: (status, reason) and the envelope's invariants.
    fn openrouter_error(&self) -> (u16, String) {
        let e = &self.body["error"];
        assert_eq!(e["code"], json!(self.status), "{}", self.text);
        assert!(e["message"].is_string(), "{}", self.text);
        let m = &e["metadata"];
        assert_eq!(m["request_id"], json!(self.request_id()), "{}", self.text);
        assert_eq!(
            m["retriable"],
            json!(matches!(self.status, 429 | 500 | 502)),
            "{}",
            self.text
        );
        assert!(m["details"].is_object());
        (self.status, m["reason"].as_str().unwrap().to_string())
    }

    /// Router error envelope: (status, code).
    fn router_error(&self) -> (u16, String) {
        assert_eq!(self.body["schema_version"], "1.1", "{}", self.text);
        assert_eq!(self.body["request_id"], json!(self.request_id()));
        assert!(self.request_id().starts_with("req_"));
        let e = &self.body["error"];
        assert!(e["message"].is_string());
        // The router's rule (api.rs:221-225): only 429 and 500.
        assert_eq!(e["retriable"], json!(matches!(self.status, 429 | 500)));
        // The router's `details`: an object for TAXONOMY_NOT_FOUND, else null.
        assert_eq!(
            e["details"].is_object(),
            e["code"] == "TAXONOMY_NOT_FOUND",
            "{}",
            self.text
        );
        assert!(e["details"].is_object() || e["details"].is_null());
        assert_eq!(e["metadata"]["reason"], e["code"]);
        (self.status, e["code"].as_str().unwrap().to_string())
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
    for (k, _) in &headers {
        assert!(
            !k.as_str().starts_with("access-control-"),
            "a CORS header on {method} {path}: {k}"
        );
    }
    assert!(
        headers.contains_key("x-request-id"),
        "no x-request-id on {method} {path}"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Resp {
        status,
        headers,
        body,
        text,
    }
}

fn action(r: &Resp, q: &str) -> String {
    r.body["cmf"]["questions"][q]["action"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

// ------------------------------------------------------------------ 401 and the open mode

#[tokio::test]
async fn keys_missing_wrong_expired_revoked_are_401_and_open_mode_is_loopback_without_keys() {
    let b = topics_body(accepted());

    // Loopback, no key in keys.json: open mode.
    let open = Srv::open(cfg());
    let r = open.post("/v1/decisions", None, &b).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.request_id(), r.body["id"].as_str().unwrap());
    assert!(r.request_id().starts_with("cmf-dec-"));
    let u = open.get("/v1/usage", None).await;
    assert_eq!(u.body["account"]["id"], "anonymous");

    // Not loopback (auth.require null → required): 401 without a key.
    let remote = Srv::open_with(cfg(), false, Some(ADMIN), tempfile::tempdir().unwrap());
    let r = remote.post("/v1/decisions", None, &b).await;
    assert_eq!(r.openrouter_error(), (401, "UNAUTHORIZED".to_string()));
    assert!(
        r.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("missing")
    );

    // Loopback with a key: the open mode ends.
    let (key, account) = open
        .key(json!({"plan": "developer", "account": "acme"}))
        .await;
    assert_eq!(account, "acme");
    let r = open.post("/v1/decisions", None, &b).await;
    assert_eq!(r.openrouter_error(), (401, "UNAUTHORIZED".to_string()));
    let r = open
        .post(
            "/v1/decisions",
            Some("cortiq_0000000000000000000000000000000000000000"),
            &b,
        )
        .await;
    assert_eq!(r.openrouter_error(), (401, "UNAUTHORIZED".to_string()));
    assert_eq!(r.body["error"]["message"], "invalid API key");
    let r = open.post("/v1/decisions", Some(&key), &b).await;
    assert_eq!(r.status, 200, "{}", r.text);
    // x-api-key and a lower-case scheme work too.
    let bytes = serde_json::to_vec(&b).unwrap();
    let r = open
        .call(
            "POST",
            "/v1/decisions",
            &[("content-type", "application/json"), ("x-api-key", &key)],
            Some(bytes.clone()),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let lower = format!("bearer {key}");
    let r = open
        .call(
            "POST",
            "/api/alpha/decisions",
            &[
                ("content-type", "application/json"),
                ("authorization", &lower),
            ],
            Some(bytes),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);

    // Revoked.
    let rv = open.admin("DELETE", "/v1/admin/keys/acme", None).await;
    assert_eq!(rv.body, json!({"account": "acme", "revoked": 1}));
    let r = open.post("/v1/decisions", Some(&key), &b).await;
    assert_eq!(r.openrouter_error(), (401, "UNAUTHORIZED".to_string()));
    assert_eq!(r.body["error"]["message"], "API key revoked");

    // Expired: a record whose expiry has passed, written before the start.
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let raw = "cortiq_1111111111111111111111111111111111111111";
    let rec = KeyRecord {
        hash: hash_key(raw),
        account: "late".into(),
        plan: "starter".into(),
        label: String::new(),
        created: now_unix() - 1000,
        expires: Some(now_unix() - 10),
        active: true,
        rate_per_min: 0,
        decision_quota: 0,
        token_quota: 0,
        credit_usd: None,
        oracle_budget_usd: None,
        oracle_allowed: false,
        learning_allowed: false,
        router: None,
    };
    let file = KeysFile {
        version: KEYS_FILE_VERSION,
        keys: vec![rec],
    };
    std::fs::write(state.join("keys.json"), serde_json::to_vec(&file).unwrap()).unwrap();
    let late = Srv::open_with(cfg(), true, Some(ADMIN), dir);
    let r = late.post("/v1/decisions", Some(raw), &b).await;
    assert_eq!(r.openrouter_error(), (401, "UNAUTHORIZED".to_string()));
    assert_eq!(r.body["error"]["message"], "API key expired");
    // The open mode is off although the only key is expired.
    let r = late.post("/v1/decisions", None, &b).await;
    assert_eq!(r.status, 401);
}

// ------------------------------------------------------------------ 429

async fn wait_for_fresh_minute() {
    // Three requests must fall into one fixed minute window.
    let s = now_unix() % 60;
    if s >= 55 {
        tokio::time::sleep(Duration::from_secs(61 - s)).await;
    }
}

#[tokio::test]
async fn two_requests_a_minute_then_429_with_retry_after() {
    let srv = Srv::open(cfg());
    let (key, _) = srv.key(json!({"plan": "starter", "rate_per_min": 2})).await;
    let b = topics_body(accepted());
    wait_for_fresh_minute().await;
    for _ in 0..2 {
        assert_eq!(srv.post("/v1/decisions", Some(&key), &b).await.status, 200);
    }
    let r = srv.post("/v1/decisions", Some(&key), &b).await;
    assert_eq!(r.openrouter_error(), (429, "RATE_LIMITED".to_string()));
    let retry: u64 = r.header("retry-after").unwrap().parse().unwrap();
    assert!((1..=60).contains(&retry), "Retry-After {retry}");
    assert_eq!(r.body["error"]["metadata"]["details"]["rate_per_min"], 2);
    // The window counts every keyed endpoint, like the router's middleware.
    let r = srv.get("/v1/usage", Some(&key)).await;
    assert_eq!(r.router_error(), (429, "RATE_LIMITED".to_string()));
}

/// An escalator that blocks until released (the request holds its slot on the
/// blocking pool meanwhile).
struct Hold {
    entered: Mutex<std::sync::mpsc::Sender<()>>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl Escalator for Hold {
    fn escalate(&self, e: &Escalation<'_>) -> EscalationResult {
        self.entered.lock().unwrap().send(()).unwrap();
        let _ = self
            .release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(30));
        EscalationResult {
            resolved: e
                .pending
                .iter()
                .map(|_| Resolved::new(Resolution::Refused(RefusalReason::Budget)))
                .collect(),
            usage: Default::default(),
        }
    }
    fn feedback(&self, _: &FeedbackRequest, _: &Principal) -> Result<Value, ApiError> {
        Err(ApiError::not_found("no feedback here"))
    }
    fn admin(&self, _: &AdminCommand) -> Result<Value, ApiError> {
        Err(ApiError::not_found("no admin here"))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_inflight_overflow_is_429_overloaded_while_work_runs_on_the_blocking_pool() {
    // A slot held directly: the next request is refused before any work.
    let mut c = cfg();
    c.limits.max_inflight = 1;
    let srv = Srv::open(c.clone());
    let b = topics_body(accepted());
    {
        let _slot = srv.server().state().service().enter().unwrap();
        let r = srv.post("/v1/decisions", None, &b).await;
        assert_eq!(r.openrouter_error(), (429, "OVERLOADED".to_string()));
        assert_eq!(r.header("retry-after"), Some("1"));
        let r = srv
            .post(
                "/v1/route",
                None,
                &json!({"taxonomy_id": "topics", "input": {"text": accepted()}}),
            )
            .await;
        assert_eq!(r.router_error(), (429, "OVERLOADED".to_string()));
    }
    assert_eq!(srv.post("/v1/decisions", None, &b).await.status, 200);

    // A request blocked inside the decision (an escalator that waits): the
    // runtime still answers, and the second decision finds no slot.
    let (tx_in, rx_in) = std::sync::mpsc::channel();
    let (tx_out, rx_out) = std::sync::mpsc::channel();
    let hold = Arc::new(Hold {
        entered: Mutex::new(tx_in),
        release: Mutex::new(rx_out),
    });
    let mut c2 = c;
    c2.oracle.enabled = true;
    // Open mode with the oracle: only when auth.require is false explicitly
    // (a loopback address alone gives no oracle to a caller without a key).
    c2.auth.require = Some(false);
    let m = DecisionModel::open(&toy().path, Verify::Light).unwrap();
    let h = Arc::new(ModelHandle::new(LoadedModel::new(m).unwrap()));
    let esc: Arc<dyn Escalator> = hold;
    let svc = DecisionService::open(h, c2, Some(esc))
        .unwrap()
        .with_loopback(true);
    let app = decisions::router(DecisionState::new(Arc::new(svc), None).unwrap());
    let slow = serde_json::to_vec(&topics_body(REJECTED)).unwrap();
    let app2 = app.clone();
    let first = tokio::spawn(async move {
        call(
            &app2,
            "POST",
            "/v1/decisions",
            &[("content-type", "application/json")],
            Some(slow),
        )
        .await
    });
    tokio::task::spawn_blocking(move || rx_in.recv_timeout(Duration::from_secs(30)).unwrap())
        .await
        .unwrap();
    let h = call(&app, "GET", "/healthz", &[], None).await;
    assert_eq!(h.status, 200);
    assert_eq!(h.body["inflight"], 1);
    let r = call(
        &app,
        "POST",
        "/v1/decisions",
        &[("content-type", "application/json")],
        Some(serde_json::to_vec(&b).unwrap()),
    )
    .await;
    assert_eq!(r.openrouter_error(), (429, "OVERLOADED".to_string()));
    tx_out.send(()).unwrap();
    let r = first.await.unwrap();
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(action(&r, "task"), "abstain");
    assert_eq!(
        r.body["cmf"]["questions"]["task"]["flags"],
        json!(["budget"])
    );
}

// ------------------------------------------------------------------ 402

#[tokio::test]
async fn decision_quota_token_quota_and_credit_are_402() {
    let mut c = cfg();
    c.pricing.request_usd = "0.001".into();
    let srv = Srv::open(c);
    let b = topics_body(accepted());

    let (k, _) = srv.key(json!({"decision_quota": 1})).await;
    assert_eq!(srv.post("/v1/decisions", Some(&k), &b).await.status, 200);
    let r = srv.post("/v1/decisions", Some(&k), &b).await;
    assert_eq!(r.openrouter_error(), (402, "QUOTA_EXCEEDED".to_string()));
    let d = &r.body["error"]["metadata"]["details"];
    assert_eq!(
        (d["quota"].as_str(), d["used"].as_u64(), d["limit"].as_u64()),
        (Some("decision"), Some(1), Some(1))
    );

    let (k, _) = srv.key(json!({"token_quota": 5})).await;
    let first = srv.post("/v1/decisions", Some(&k), &b).await;
    assert_eq!(first.status, 200);
    assert!(first.body["usage"]["input_tokens"].as_u64().unwrap() > 5);
    let r = srv.post("/v1/decisions", Some(&k), &b).await;
    assert_eq!(r.openrouter_error(), (402, "QUOTA_EXCEEDED".to_string()));
    assert_eq!(r.body["error"]["metadata"]["details"]["quota"], "token");

    let (k, _) = srv.key(json!({"credit_usd": "0.001"})).await;
    let first = srv.post("/v1/decisions", Some(&k), &b).await;
    assert_eq!(first.status, 200);
    assert_eq!(first.body["usage"]["cost"].as_f64(), Some(0.001));
    let r = srv.post("/v1/decisions", Some(&k), &b).await;
    assert_eq!(r.openrouter_error(), (402, "QUOTA_EXCEEDED".to_string()));
    assert_eq!(r.body["error"]["metadata"]["details"]["quota"], "credit");
    // Refused requests are not billed: one decision, one request price.
    let u = srv.get("/v1/usage", Some(&k)).await;
    assert_eq!(u.status, 402, "the credit gate covers every keyed endpoint");
    let au = srv.admin("GET", "/v1/admin/usage", None).await;
    let accounts = au.body["accounts"].as_object().unwrap();
    assert_eq!(accounts.len(), 3);
    for (a, t) in accounts {
        assert_eq!(
            (t["requests"].as_u64(), t["decisions"].as_u64()),
            (Some(1), Some(1)),
            "{a}"
        );
        assert_eq!(t["cost_usd"].as_f64(), Some(0.001), "{a}");
    }

    // Nor can one /v1/decisions request: more questions than the decision
    // quota has left is 402 before any work, and nothing is billed.
    let (k, _) = srv
        .key(json!({"account": "manyq", "decision_quota": 1}))
        .await;
    let two = body(
        json!(accepted()),
        json!({"a": choice(&TOPICS), "b": choice(&SHOP)}),
        None,
    );
    let r = srv.post("/v1/decisions", Some(&k), &two).await;
    assert_eq!(r.openrouter_error(), (402, "QUOTA_EXCEEDED".to_string()));
    let d = &r.body["error"]["metadata"]["details"];
    assert_eq!(
        (
            d["used"].as_u64(),
            d["limit"].as_u64(),
            d["requested"].as_u64()
        ),
        (Some(0), Some(1), Some(2))
    );
    assert_eq!(srv.post("/v1/decisions", Some(&k), &b).await.status, 200);

    // A router batch cannot run past a quota or a credit limit: every input
    // after the first is checked against what the batch used so far (402;
    // the inputs before it are decided and billed).
    let inputs: Vec<Value> = (0..5).map(|_| json!({"text": accepted()})).collect();
    let batch = json!({"taxonomy_id": "topics", "inputs": inputs});
    let (k, _) = srv
        .key(json!({"account": "batchq", "decision_quota": 2}))
        .await;
    let r = srv.post("/v1/route:batch", Some(&k), &batch).await;
    assert_eq!(r.router_error(), (402, "QUOTA_EXCEEDED".to_string()));
    let (k, _) = srv
        .key(json!({"account": "batchc", "credit_usd": "0.002"}))
        .await;
    let r = srv.post("/v1/route:batch", Some(&k), &batch).await;
    assert_eq!(r.router_error(), (402, "QUOTA_EXCEEDED".to_string()));
    let au = srv.admin("GET", "/v1/admin/usage", None).await;
    let accounts = &au.body["accounts"];
    assert_eq!(accounts["batchq"]["decisions"], 2, "{au:?}");
    assert_eq!(accounts["batchc"]["decisions"], 2, "{au:?}");
    assert_eq!(accounts["batchc"]["cost_usd"].as_f64(), Some(0.002));
}

// ------------------------------------------------------------------ 413, 400, 404

#[tokio::test]
async fn payload_too_large_bad_requests_and_unknown_models() {
    let mut c = cfg();
    c.limits.body_bytes = 4096;
    c.limits.state_bytes = 1024;
    let srv = Srv::open(c);
    let big = body(
        json!("x".repeat(5000)),
        json!({"task": choice(&TOPICS)}),
        None,
    );
    let bytes = serde_json::to_vec(&big).unwrap();
    // Declared too large: refused before reading.
    let len = bytes.len().to_string();
    let r = srv
        .call(
            "POST",
            "/v1/decisions",
            &[
                ("content-type", "application/json"),
                ("content-length", &len),
            ],
            Some(bytes.clone()),
        )
        .await;
    assert_eq!(r.openrouter_error(), (413, "PAYLOAD_TOO_LARGE".to_string()));
    // Streamed without a length: refused while reading.
    let r = srv
        .call(
            "POST",
            "/v1/decisions",
            &[("content-type", "application/json")],
            Some(bytes),
        )
        .await;
    assert_eq!(r.openrouter_error(), (413, "PAYLOAD_TOO_LARGE".to_string()));
    // A state over state_bytes inside a small body: 400.
    let r = srv
        .post(
            "/v1/decisions",
            None,
            &body(
                json!("y".repeat(2000)),
                json!({"task": choice(&TOPICS)}),
                None,
            ),
        )
        .await;
    assert_eq!(r.openrouter_error(), (400, "INVALID_REQUEST".to_string()));

    // 400s.
    let bad = |b: &[u8]| {
        srv.call(
            "POST",
            "/v1/decisions",
            &[("content-type", "application/json")],
            Some(b.to_vec()),
        )
    };
    for b in [
        &b"{not json"[..],
        br#"{"model":"cortiq/decision","state":"x","questions":{"t":{"type":"choice","instructions":"i","criteria":{"a":"1","b":"2"}}},"extra":1}"#,
        br#"{"model":"cortiq/decision","state":"x","questions":{"t":{"type":"choice","instructions":"i","criteria":{"a":"1","a":"2","b":"3"}}}}"#,
        br#"{"model":"cortiq/decision","state":"x","questions":{}}"#,
        br#"{"model":"cortiq/decision","state":"x","questions":{"t":{"type":"choice","instructions":"i","criteria":{"a":"1","b":"2"}}},"cmf":{"nope":true}}"#,
    ] {
        let r = bad(b).await;
        assert_eq!(r.openrouter_error(), (400, "INVALID_REQUEST".to_string()), "{}", String::from_utf8_lossy(b));
    }
    let dup = bad(br#"{"model":"cortiq/decision","state":"x","questions":{"t":{"type":"choice","instructions":"i","criteria":{"a":"1","a":"2","b":"3"}}}}"#).await;
    assert!(
        dup.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("duplicate key 'a'")
    );
    // Content type.
    let r = srv
        .call(
            "POST",
            "/v1/decisions",
            &[("content-type", "text/plain")],
            Some(serde_json::to_vec(&topics_body(accepted())).unwrap()),
        )
        .await;
    assert_eq!(r.openrouter_error(), (400, "INVALID_REQUEST".to_string()));
    assert!(
        r.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Content-Type")
    );
    let r = srv
        .call(
            "POST",
            "/v1/decisions",
            &[("content-type", "application/json; charset=utf-8")],
            Some(serde_json::to_vec(&topics_body(accepted())).unwrap()),
        )
        .await;
    assert_eq!(r.status, 200);

    // Models.
    let mut jev = topics_body(accepted());
    jev["model"] = json!("typesafe/jev-1.13");
    let r = srv.post("/v1/decisions", None, &jev).await;
    assert_eq!(r.openrouter_error(), (404, "MODEL_NOT_FOUND".to_string()));
    let models = srv.get("/v1/models", None).await;
    let sha = models.body["data"][0]["cmf"]["model_sha"]
        .as_str()
        .unwrap()
        .to_string();
    let mut pinned = topics_body(accepted());
    pinned["model"] = json!(format!("cortiq/decision@{}", &sha[..12]));
    let r = srv.post("/v1/decisions", None, &pinned).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.body["model"], pinned["model"]);
    pinned["model"] = json!("cortiq/decision@000000000000");
    let r = srv.post("/v1/decisions", None, &pinned).await;
    assert_eq!(r.openrouter_error(), (404, "MODEL_NOT_FOUND".to_string()));

    // No such endpoint, wrong method, no web interface.
    for p in ["/", "/index.html", "/dashboard", "/v1/chat/completions"] {
        let r = srv.get(p, None).await;
        assert_eq!(
            r.openrouter_error(),
            (404, "INVALID_REQUEST".to_string()),
            "{p}"
        );
    }
    let r = srv.get("/v1/decisions", None).await;
    assert_eq!(r.status, 405);
    assert_eq!(r.body["error"]["code"], 405);
}

// ------------------------------------------------------------------ OpenRouter fields, matching

#[tokio::test]
async fn openrouter_optional_fields_are_accepted_and_ignored() {
    let srv = Srv::open(cfg());
    let mut b = topics_body(accepted());
    b["provider"] = json!({"order": ["Cortiq"], "allow_fallbacks": false});
    b["user"] = json!("user-42");
    b["session_id"] = json!("session-7");
    b["trace"] = json!({"trace_id": "t1", "generation_name": "g"});
    let r = srv.post("/v1/decisions", None, &b).await;
    assert_eq!(r.status, 200, "{}", r.text);
    let plain = srv
        .post("/v1/decisions", None, &topics_body(accepted()))
        .await;
    assert_eq!(r.body["answers"], plain.body["answers"]);
    assert_eq!(r.body["provider"], "Cortiq");
    // Ill-typed optional fields are still refused.
    b["provider"] = json!("x");
    assert_eq!(srv.post("/v1/decisions", None, &b).await.status, 400);
}

#[tokio::test]
async fn skill_matching_exact_subset_superset_untrained_over_http() {
    let srv = Srv::open(cfg());
    let q = |r: &Resp, k: &str| r.body["cmf"]["questions"]["task"][k].clone();

    let exact = srv
        .post("/v1/decisions", None, &topics_body(accepted()))
        .await;
    assert_eq!(exact.status, 200);
    assert_eq!(q(&exact, "match"), "exact");
    assert_eq!(q(&exact, "skill"), "topics");
    assert_eq!(q(&exact, "action"), "local");
    assert_eq!(q(&exact, "certified"), true);
    let a = &exact.body["answers"]["task"];
    let probs = a["probabilities"].as_object().unwrap();
    assert_eq!(
        probs.keys().collect::<Vec<_>>(),
        TOPICS.iter().collect::<Vec<_>>()
    );
    let sum: f64 = probs.values().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-5);

    let sub = srv
        .post(
            "/v1/decisions",
            None,
            &body(
                json!(accepted()),
                json!({"task": choice(&["billing", "travel", "cards"])}),
                None,
            ),
        )
        .await;
    assert_eq!(sub.status, 200);
    assert_eq!(q(&sub, "match"), "subset");
    assert_eq!(q(&sub, "certified"), false);

    let mut five = TOPICS.to_vec();
    five.push("cruise");
    let sup = srv
        .post(
            "/v1/decisions",
            None,
            &body(json!(accepted()), json!({"task": choice(&five)}), None),
        )
        .await;
    assert_eq!(
        sup.openrouter_error(),
        (422, "UNSUPPORTED_QUESTION".to_string())
    );
    let d = &sup.body["error"]["metadata"]["details"]["questions"]["task"];
    assert_eq!(
        (d["match"].as_str(), d["skill"].as_str()),
        (Some("superset"), Some("topics"))
    );
    assert!(d["reason"].as_str().unwrap().contains("cruise"));

    let multi = body(
        json!(accepted()),
        json!({
            "team": choice(&["alpha", "beta"]),
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
            "refund": {"type": "noul", "instructions": "Refund asked?"},
            "task": choice(&TOPICS),
        }),
        None,
    );
    let r = srv.post("/v1/decisions", None, &multi).await;
    assert_eq!(
        r.openrouter_error(),
        (422, "UNSUPPORTED_QUESTION".to_string())
    );
    let d = r.body["error"]["metadata"]["details"]["questions"]
        .as_object()
        .unwrap();
    assert_eq!(
        d.keys().collect::<Vec<_>>(),
        vec!["team", "urgency", "refund"]
    );
    for (_, v) in d {
        assert_eq!(v["match"], "untrained");
        assert!(v["reason"].is_string());
    }
    // A forced unknown skill is 400.
    let r = srv
        .post(
            "/v1/decisions",
            None,
            &body(
                json!("x"),
                json!({"task": choice(&TOPICS)}),
                Some(json!({"skill": "nope"})),
            ),
        )
        .await;
    assert_eq!(r.openrouter_error(), (400, "INVALID_REQUEST".to_string()));
}

// ------------------------------------------------------------------ metering

fn ledger_lines(root: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(root.join("usage"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    files.sort();
    for f in files {
        for l in std::fs::read_to_string(f).unwrap().lines() {
            out.push(serde_json::from_str(l).unwrap());
        }
    }
    out
}

#[tokio::test]
async fn usage_rows_and_sums_and_usage_shows_only_the_callers_account() {
    let mut c = cfg();
    c.pricing.input_usd_per_1m = "2".into();
    c.pricing.output_usd_per_1m = "10".into();
    let srv = Srv::open(c);
    let (ka, _) = srv.key(json!({"account": "alpha"})).await;
    let (kb, _) = srv.key(json!({"account": "bravo"})).await;
    let marker = "zebra-state-marker";
    let texts = [
        format!("{} {marker}", accepted()),
        "rain snow storm today".to_string(),
    ];
    let mut mine = Vec::new();
    for t in &texts {
        let r = srv
            .post(
                "/v1/decisions",
                Some(&ka),
                &body(
                    json!(t),
                    json!({"a": choice(&TOPICS), "b": choice(&SHOP)}),
                    None,
                ),
            )
            .await;
        assert_eq!(r.status, 200, "{}", r.text);
        mine.push(r);
    }
    let other = srv
        .post("/v1/decisions", Some(&kb), &topics_body(accepted()))
        .await;
    assert_eq!(other.status, 200);

    for r in &mine {
        let u = &r.body["usage"];
        let local = &r.body["cmf"]["usage"]["local"];
        assert_eq!(u["input_tokens"], local["input_tokens"]);
        assert_eq!(u["output_tokens"], json!(TOPICS.len() + SHOP.len()));
        let expect = u["input_tokens"].as_f64().unwrap() * 2e-6
            + u["output_tokens"].as_f64().unwrap() * 10e-6;
        assert!((u["cost"].as_f64().unwrap() - expect).abs() < 1e-15, "{u}");
        assert_eq!(r.body["cmf"]["usage"]["oracle"]["calls"], 0);
    }

    let ua = srv.get("/v1/usage", Some(&ka)).await;
    assert_eq!(ua.status, 200, "{}", ua.text);
    assert_eq!(ua.body["schema_version"], "1.1");
    assert_eq!(ua.body["account"]["id"], "alpha");
    assert_eq!(ua.body["usage"]["billable_decisions"], 4);
    // The decisions service's totals are the opt-in `cmf.totals`.
    let t = &ua.body["cmf"]["totals"];
    let sum = |k: &str| {
        mine.iter()
            .map(|r| r.body["usage"][k].as_u64().unwrap())
            .sum::<u64>()
    };
    assert_eq!(t["requests"], 2);
    assert_eq!(t["decisions"], 4);
    assert_eq!(ua.body["account"]["billable_decisions"], 4);
    assert_eq!(t["input_tokens"].as_u64(), Some(sum("input_tokens")));
    assert_eq!(t["output_tokens"].as_u64(), Some(sum("output_tokens")));
    let cost: f64 = mine
        .iter()
        .map(|r| r.body["usage"]["cost"].as_f64().unwrap())
        .sum();
    assert!((t["cost_usd"].as_f64().unwrap() - cost).abs() < 1e-12);
    let ub = srv.get("/v1/usage", Some(&kb)).await;
    assert_eq!(ub.body["account"]["id"], "bravo");
    assert_eq!(ub.body["usage"]["billable_decisions"], 1);
    assert_eq!(ub.body["cmf"]["totals"]["decisions"], 1);
    assert!(!ua.text.contains("bravo") && !ub.text.contains("alpha"));

    // The ledger rows: one per request, the response's numbers, no text.
    srv.server()
        .state()
        .service()
        .ledger()
        .unwrap()
        .flush()
        .unwrap();
    let rows = ledger_lines(&srv.state_root());
    assert_eq!(rows.len(), 3);
    for (row, r) in rows.iter().zip(mine.iter().chain([&other])) {
        assert_eq!(row["id"], r.body["id"]);
        assert_eq!(row["input_tokens"], r.body["usage"]["input_tokens"]);
        assert_eq!(row["output_tokens"], r.body["usage"]["output_tokens"]);
        assert_eq!(row["cost_usd"], r.body["usage"]["cost"]);
    }
    assert_eq!(rows[0]["account"], "alpha");
    assert_eq!(rows[2]["account"], "bravo");
    assert_eq!(rows[0]["questions"], 2);
    let all = std::fs::read_dir(srv.state_root().join("usage")).unwrap();
    for f in all {
        let bytes = std::fs::read(f.unwrap().path()).unwrap();
        assert!(
            !String::from_utf8_lossy(&bytes).contains(marker),
            "state text in the ledger"
        );
    }
    // The admin view has every account.
    let au = srv.admin("GET", "/v1/admin/usage", None).await;
    assert_eq!(au.body["accounts"]["alpha"]["decisions"], 4);
    assert_eq!(au.body["accounts"]["bravo"]["decisions"], 1);
    // After a restart the totals come back from the ledger.
    let dir = srv.close();
    let srv = Srv::open_with(cfg(), true, Some(ADMIN), dir);
    let ua2 = srv.get("/v1/usage", Some(&ka)).await;
    assert_eq!(ua2.body["cmf"]["totals"]["input_tokens"], t["input_tokens"]);
}

// ------------------------------------------------------------------ admin

#[tokio::test]
async fn admin_creates_lists_and_revokes_keys() {
    // Without a token in the environment the admin API does not exist.
    let off = Srv::open_with(cfg(), true, None, tempfile::tempdir().unwrap());
    let r = off.admin("GET", "/v1/admin/keys", None).await;
    assert_eq!(r.router_error(), (404, "ADMIN_DISABLED".to_string()));

    let srv = Srv::open(cfg());
    let r = srv
        .call("GET", "/v1/admin/keys", &[("x-admin-token", "wrong")], None)
        .await;
    assert_eq!(r.router_error(), (401, "UNAUTHORIZED".to_string()));
    let r = srv.call("GET", "/v1/admin/keys", &[], None).await;
    assert_eq!(r.router_error(), (401, "UNAUTHORIZED".to_string()));

    let r = srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"plan": "pro", "account": "acme", "label": "prod", "email": "ops@acme.test"})),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let raw = r.body["key"].as_str().unwrap().to_string();
    assert!(raw.starts_with("cortiq_") && raw.len() == 47);
    assert!(raw[7..].bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(r.body["plan"], "pro");
    assert_eq!(r.body["rate_per_min"], 600);
    assert_eq!(r.body["decision_quota"], 1_000_000);
    assert_eq!(r.body["persisted"], true);
    assert_eq!(r.body["cmf"]["hash12"], json!(&hash_key(&raw)[..12]));
    assert!(r.body.get("created_at").is_some() && r.body.get("expires_at").is_some());
    assert!(!r.text.contains("ops@acme.test"), "the email is not stored");
    let (k2, _) = srv
        .key(json!({"account": "beta", "oracle_allowed": true, "days": 7}))
        .await;

    let list = srv.admin("GET", "/v1/admin/keys", None).await;
    assert_eq!(list.body["count"], 2);
    assert!(
        !list.text.contains(&raw) && !list.text.contains(&k2),
        "a raw key in the listing"
    );
    let first = &list.body["keys"][0];
    for f in [
        "account",
        "plan",
        "rate_per_min",
        "decision_quota",
        "expires_at",
        "expired",
        "key_hash_prefix",
        "usage",
    ] {
        assert!(first.get(f).is_some(), "listing field {f}");
    }
    for f in ["hash12", "active", "label", "token_quota"] {
        assert!(first["cmf"].get(f).is_some(), "extension listing field {f}");
    }
    assert_eq!(first["usage"]["decisions"], 0);
    let keys_json = std::fs::read_to_string(srv.state_root().join("keys.json")).unwrap();
    assert!(
        !keys_json.contains(&raw) && !keys_json.contains(&k2),
        "keys.json holds a raw key"
    );
    assert!(keys_json.contains(&hash_key(&raw)));
    assert!(!keys_json.contains("ops@acme.test"));

    // The keys work, then are revoked by account and by hash.
    let b = topics_body(accepted());
    assert_eq!(srv.post("/v1/decisions", Some(&raw), &b).await.status, 200);
    assert_eq!(srv.post("/v1/decisions", Some(&k2), &b).await.status, 200);
    let list = srv.admin("GET", "/v1/admin/keys", None).await;
    assert_eq!(list.body["keys"][0]["usage"]["decisions"], 1);
    let r = srv.admin("DELETE", "/v1/admin/keys/acme", None).await;
    assert_eq!(r.body["revoked"], 1);
    assert_eq!(srv.post("/v1/decisions", Some(&raw), &b).await.status, 401);
    let h12 = &hash_key(&k2)[..12];
    let r = srv
        .admin("DELETE", &format!("/v1/admin/keys/hash/{h12}"), None)
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["revoked"], 1);
    assert_eq!(srv.post("/v1/decisions", Some(&k2), &b).await.status, 401);
    // This server's own admin paths answer with the errors of §4.8
    // (OpenRouter's shape), the router's admin keys paths with the router's.
    let r = srv.admin("DELETE", "/v1/admin/keys/hash/zz", None).await;
    assert_eq!(r.openrouter_error(), (400, "INVALID_REQUEST".to_string()));
    assert!(r.request_id().starts_with("cmf-dec-"));
    // A path parameter axum cannot extract (invalid UTF-8) is a JSON 400 in
    // the path's envelope, never axum's plain text.
    let r = srv.admin("DELETE", "/v1/admin/keys/hash/%FF", None).await;
    assert_eq!(r.openrouter_error(), (400, "INVALID_REQUEST".to_string()));
    let r = srv.admin("DELETE", "/v1/admin/keys/%FF", None).await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".to_string()));
    let (k3, _) = srv.key(json!({"account": "pathcheck"})).await;
    let r = srv.get("/v1/skills/%FF", Some(&k3)).await;
    assert_eq!(r.openrouter_error(), (400, "INVALID_REQUEST".to_string()));
    let r = srv.get("/v1/taxonomies/%FF", Some(&k3)).await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".to_string()));
    // The key check still comes first on a keyed path.
    let r = srv.get("/v1/skills/%FF", None).await;
    assert_eq!(r.openrouter_error(), (401, "UNAUTHORIZED".to_string()));
    for p in ["/v1/admin/usage", "/v1/admin/learning"] {
        let r = srv
            .call("GET", p, &[("x-admin-token", "wrong")], None)
            .await;
        assert_eq!(
            r.openrouter_error(),
            (401, "UNAUTHORIZED".to_string()),
            "{p}"
        );
    }
    let r = off.admin("GET", "/v1/admin/usage", None).await;
    assert_eq!(r.openrouter_error(), (404, "ADMIN_DISABLED".to_string()));
    // Bad requests.
    let r = srv
        .admin("POST", "/v1/admin/keys", Some(&json!({"plan": "platinum"})))
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".to_string()));
    // Unknown fields are ignored like the router's serde; a wrong type is
    // axum's 422 text.
    let r = srv
        .admin("POST", "/v1/admin/keys", Some(&json!({"colour": "red"})))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let r = srv
        .admin(
            "POST",
            "/v1/admin/keys",
            Some(&json!({"rate_per_min": "fast"})),
        )
        .await;
    assert_eq!(r.status, 422, "{}", r.text);
    // The other admin endpoints answer (no oracle traffic here).
    for p in [
        "/v1/admin/usage",
        "/v1/admin/oracle",
        "/v1/admin/learning",
        "/v1/admin/generations",
    ] {
        let r = srv.admin("GET", p, None).await;
        assert_eq!(r.status, 200, "{p}: {}", r.text);
    }
    let r = srv
        .admin(
            "POST",
            "/v1/admin/rollback",
            Some(&json!({"generation": 5})),
        )
        .await;
    assert_eq!(r.openrouter_error(), (404, "INVALID_REQUEST".to_string()));
    let r = srv
        .admin("POST", "/v1/admin/rollback", Some(&json!({"gen": 0})))
        .await;
    assert_eq!(r.openrouter_error(), (400, "INVALID_REQUEST".to_string()));
}

// ------------------------------------------------------------------ service endpoints

#[tokio::test]
async fn models_shape_prices_healthz_readyz_metrics_and_no_cors() {
    let srv = Srv::open(cfg());
    let (key, _) = srv.key(json!({})).await;
    // /v1/models is open even with keys.
    let m = srv.get("/v1/models", None).await;
    assert_eq!(m.status, 200, "{}", m.text);
    assert!(m.request_id().starts_with("cmf-dec-"));
    let d = &m.body["data"][0];
    assert_eq!(d["id"], "cortiq/decision");
    assert_eq!(d["name"], "Cortiq Decision");
    assert_eq!(d["input_modalities"], json!(["text"]));
    assert_eq!(d["output_modalities"], json!(["decisions"]));
    // The encoder's position limit (512 for the release file; the toy's is small).
    assert!(d["context_length"].as_u64().unwrap() >= 8);
    assert_eq!(d["max_output_length"], 255);
    assert_eq!(d["quantization"], "fp32");
    assert_eq!(d["hugging_face_id"], "infosave/cortiq-decision");
    assert_eq!(
        d["pricing"],
        json!({"prompt": "0", "completion": "0", "request": "0", "image": "0"})
    );
    assert_eq!(d["supported_sampling_parameters"], json!([]));
    assert_eq!(d["supported_features"], json!([]));
    assert_eq!(d["cmf"]["model_sha"].as_str().unwrap().len(), 64);
    let skills = d["cmf"]["skills"].as_array().unwrap();
    assert_eq!(skills.len(), 2);
    for s in skills {
        for f in [
            "id",
            "labels",
            "certified",
            "tau",
            "theta",
            "temperature",
            "odd_half",
        ] {
            assert!(s.get(f).is_some(), "skill field {f}");
        }
    }
    let mut c = cfg();
    c.pricing.input_usd_per_1m = "0.5".into();
    c.pricing.output_usd_per_1m = "2".into();
    c.pricing.request_usd = "0.001".into();
    let priced = Srv::open(c);
    let p = &priced.get("/v1/models", None).await.body["data"][0]["pricing"];
    assert_eq!(
        p,
        &json!({"prompt": "0.0000005", "completion": "0.000002", "request": "0.001", "image": "0"})
    );

    let h = srv.get("/healthz", None).await;
    assert_eq!(h.status, 200);
    assert_eq!(h.body["status"], "ok");
    assert_eq!(h.body["skills"], 2);
    // The router's /v1/healthz is {"status":"ok"}; the rest is opt-in.
    let h = srv.get("/v1/healthz", None).await;
    assert_eq!(h.status, 200);
    assert_eq!(h.body["status"], "ok");
    assert_eq!(h.body["cmf"]["skills"], 2);
    let r = srv.get("/v1/readyz", None).await;
    assert_eq!(r.body, json!({"status": "ready"}));
    assert!(r.request_id().starts_with("req_"));
    srv.post("/v1/decisions", Some(&key), &topics_body(accepted()))
        .await;
    let mt = srv.get("/metrics", None).await;
    assert_eq!(mt.status, 200);
    assert_eq!(mt.header("content-type"), Some("text/plain; version=0.0.4"));
    for name in [
        "cortiq_decisions_total 1",
        "cortiq_escalations_total 0",
        "cortiq_oracle_calls_total 0",
        "cortiq_cache_hits_total",
        "cortiq_oracle_unavailable_total",
        "cortiq_novelty_hits_total",
        "cortiq_refits_total",
        "cortiq_promotions_total",
        "cortiq_escalation_rate",
        "cortiq_oracle_call_rate",
        "cortiq_labeled_examples",
        "cortiq_oracle_cache_entries",
        "cortiq_oracle_cache_hits_total",
        "cortiq_oracle_cache_lookups_total",
        "cortiq_active_tasks 7",
    ] {
        assert!(mt.text.contains(name), "metrics lack {name}:\n{}", mt.text);
    }
    // Skills listing (key required).
    assert_eq!(srv.get("/v1/skills", None).await.status, 401);
    let s = srv.get("/v1/skills", Some(&key)).await;
    assert_eq!(s.body["skills"].as_array().unwrap().len(), 2);
    let t = srv.get("/v1/skills/topics", Some(&key)).await;
    assert_eq!(t.body["labels"], json!(TOPICS));
    assert_eq!(
        t.body["rubric"]["instructions"],
        "Which topic is the message about?"
    );
    let r = srv.get("/v1/skills/nope", Some(&key)).await;
    assert_eq!(r.openrouter_error(), (404, "INVALID_REQUEST".to_string()));

    // No CORS: a preflight and a cross-origin request get no CORS header
    // (checked on every response by `call`).
    let pre = srv
        .call(
            "OPTIONS",
            "/v1/decisions",
            &[
                ("origin", "https://evil.example"),
                ("access-control-request-method", "POST"),
            ],
            None,
        )
        .await;
    assert_eq!(pre.status, 405);
    let r = srv
        .call(
            "GET",
            "/v1/models",
            &[("origin", "https://evil.example")],
            None,
        )
        .await;
    assert_eq!(r.status, 200);
}

// ------------------------------------------------------------------ logs (child process)

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

const LOG_ADMIN_ENV: &str = "CMF_WP7_TEST_ADMIN_TOKEN";
const LOG_ADMIN_TOKEN: &str = "wp7-admin-token-FROM-THE-ENVIRONMENT-42";
const STATE_MARKER: &str = "platypus-state-text-marker";

/// Runs in a child process (see `request_logs_carry_no_state_key_or_token`):
/// the admin token comes from the environment, the logs are captured.
#[test]
#[ignore = "child process of request_logs_carry_no_state_key_or_token"]
fn child_request_logs() {
    let Ok(token) = std::env::var(LOG_ADMIN_ENV) else {
        println!("CHILD SKIPPED: {LOG_ADMIN_ENV} not set");
        return;
    };
    tracing::subscriber::set_global_default(capture::Capture).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (raw, ok_id, bad_id) = rt.block_on(async {
        let mut c = cfg();
        c.auth.admin_token_env = LOG_ADMIN_ENV.into();
        // No admin token override: the server reads the environment.
        let srv = Srv::open_with(c, true, None, tempfile::tempdir().unwrap());
        let r = srv
            .call(
                "POST",
                "/v1/admin/keys",
                &[
                    ("x-admin-token", &token),
                    ("content-type", "application/json"),
                ],
                Some(br#"{"account":"logacct"}"#.to_vec()),
            )
            .await;
        assert_eq!(r.status, 200, "{}", r.text);
        let raw = r.body["key"].as_str().unwrap().to_string();
        let text = format!("{} {STATE_MARKER}", accepted());
        let ok = srv
            .post("/v1/decisions", Some(&raw), &topics_body(&text))
            .await;
        assert_eq!(ok.status, 200);
        let bad = srv
            .call(
                "POST",
                "/v1/decisions",
                &[
                    ("content-type", "application/json"),
                    ("authorization", &format!("Bearer {raw}")),
                ],
                Some(format!("{{\"state\":\"{STATE_MARKER}-2\",").into_bytes()),
            )
            .await;
        assert_eq!(bad.status, 400);
        let route = srv
            .post(
                "/v1/route",
                Some(&raw),
                &json!({"taxonomy_id": "topics", "input": {"text": format!("{STATE_MARKER}-3")}}),
            )
            .await;
        assert_eq!(route.status, 200, "{}", route.text);
        // User-controlled path text is not logged either (route templates only).
        let r = srv
            .get(&format!("/v1/skills/{STATE_MARKER}-4"), Some(&raw))
            .await;
        assert_eq!(r.status, 404);
        let r = srv
            .get(&format!("/v1/{STATE_MARKER}-5?q={STATE_MARKER}-6"), None)
            .await;
        assert_eq!(r.status, 404);
        // An account is any text of the router's column, a newline too: its
        // request line stays one line (escaped).
        let r = srv
            .call(
                "POST",
                "/v1/admin/keys",
                &[
                    ("x-admin-token", &token),
                    ("content-type", "application/json"),
                ],
                Some(br#"{"account":"log\nforged=1"}"#.to_vec()),
            )
            .await;
        assert_eq!(r.status, 200, "{}", r.text);
        let odd = r.body["key"].as_str().unwrap().to_string();
        let r = srv
            .post("/v1/decisions", Some(&odd), &topics_body(accepted()))
            .await;
        assert_eq!(r.status, 200);
        srv.close();
        (
            raw,
            ok.request_id().to_string(),
            bad.request_id().to_string(),
        )
    });
    let log = capture::LOG.lock().unwrap().clone();
    for id in [&ok_id, &bad_id] {
        let line = log
            .lines()
            .find(|l| l.contains(id.as_str()) && l.contains("decision request"))
            .unwrap_or_else(|| panic!("no log line for {id}:\n{log}"));
        assert!(line.contains("status="), "{line}");
        assert!(line.contains("latency_ms="), "{line}");
        assert!(line.contains("account=logacct"), "{line}");
    }
    // Spec §4.3: a request line holds only id, status, latency and account
    // (no method, no route template, no path).
    let request_lines: Vec<&str> = log
        .lines()
        .filter(|l| l.contains("message=decision request"))
        .collect();
    assert!(request_lines.len() >= 6, "{log}");
    for l in &request_lines {
        let mut keys: Vec<&str> = l
            .split_whitespace()
            .filter_map(|t| t.split_once('=').map(|(k, _)| k))
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["account", "id", "latency_ms", "message", "status"],
            "{l}"
        );
    }
    assert!(log.contains("account=log\\nforged=1"), "{log}");
    assert!(
        !log.lines().any(|l| l.starts_with("forged=1")),
        "a forged log line:\n{log}"
    );
    assert!(log.contains(&format!("id={ok_id}")) && log.contains("status=200"));
    assert!(log.contains("status=400"));
    assert!(
        !log.contains(STATE_MARKER),
        "state text in the logs:\n{log}"
    );
    assert!(!log.contains(&raw), "an API key in the logs");
    assert!(!log.contains(&token), "the admin token in the logs");
    for l in log.lines().filter(|l| l.contains("decision request")) {
        println!("LOG{l}");
    }
    println!("CHILD OK");
}

#[test]
fn request_logs_carry_no_state_key_or_token() {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args([
            "--ignored",
            "--exact",
            "child_request_logs",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(toy_dir::TOY_CHILD_ENV, "1")
        .env(LOG_ADMIN_ENV, LOG_ADMIN_TOKEN)
        .env("CMF_GPU", "0")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "child failed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("CHILD OK"), "{stdout}\n{stderr}");
}

// ------------------------------------------------------------------ one real socket

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serves_on_a_loopback_socket_and_shuts_down_gracefully() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let dir = tempfile::tempdir().unwrap();
    let mut o = ServeOptions::new(&toy().path, cfg());
    o.state_dir = Some(dir.path().join("state"));
    o.cascade = CascadeOptions {
        key: no_key(),
        threads: 2,
        created_unix: Some(EPOCH),
    };
    let server = DecisionServer::open(&o).unwrap();
    // A second server on the same state directory is refused (LOCK).
    let err = DecisionServer::open(&o).unwrap_err();
    assert!(format!("{err:#}").contains("locked"), "{err:#}");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.run(listener, async {
        let _ = rx.await;
    }));
    let body = serde_json::to_vec(&topics_body(accepted())).unwrap();
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let head = format!(
        "POST /v1/decisions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).await.unwrap();
    s.write_all(&body).await.unwrap();
    let mut resp = Vec::new();
    s.read_to_end(&mut resp).await.unwrap();
    let text = String::from_utf8_lossy(&resp);
    assert!(text.starts_with("HTTP/1.1 200"), "{text}");
    assert!(text.to_ascii_lowercase().contains("x-request-id: cmf-dec-"));
    assert!(!text.to_ascii_lowercase().contains("access-control-"));
    tx.send(()).unwrap();
    task.await.unwrap().unwrap();
    // The ledger was flushed and the lock released.
    assert_eq!(ledger_lines(&dir.path().join("state")).len(), 1);
    assert!(!dir.path().join("state/LOCK").exists());
    DecisionServer::open(&o).unwrap().close().unwrap();
}

/// An answer before the body is read (401 after the headers alone) while the
/// client still sends the body, as nginx sends a buffered one: the server
/// reads and drops the rest instead of resetting the connection, so every
/// write succeeds and the client gets the JSON 401. Before, the unread body
/// made the kernel reset the connection; nginx failed its write with EPIPE
/// and answered 502. The body comes at once (unread when the server closes)
/// and after a pause (arriving after the close).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_early_401_reads_the_rest_of_the_body_instead_of_resetting() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let dir = tempfile::tempdir().unwrap();
    let mut c = cfg();
    c.auth.require = Some(true);
    let mut o = ServeOptions::new(&toy().path, c);
    o.state_dir = Some(dir.path().join("state"));
    o.cascade = CascadeOptions {
        key: no_key(),
        threads: 2,
        created_unix: Some(EPOCH),
    };
    let server = DecisionServer::open(&o).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.run(listener, async {
        let _ = rx.await;
    }));
    // A JSON body of exactly the limit (trailing whitespace).
    let mut body = serde_json::to_vec(&topics_body(accepted())).unwrap();
    body.resize(o.config.limits.body_bytes, b' ');
    for pause_ms in [0, 300] {
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        let head = format!(
            "POST /api/alpha/decisions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        s.write_all(head.as_bytes()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(pause_ms)).await;
        for chunk in body.chunks(64 * 1024) {
            s.write_all(chunk).await.unwrap_or_else(|e| {
                panic!("pause {pause_ms} ms: the connection was reset under the body: {e}")
            });
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        s.shutdown().await.unwrap();
        let mut resp = Vec::new();
        s.read_to_end(&mut resp)
            .await
            .unwrap_or_else(|e| panic!("pause {pause_ms} ms: no answer, a reset: {e}"));
        let text = String::from_utf8_lossy(&resp);
        assert!(text.starts_with("HTTP/1.1 401"), "{text}");
        let (_, json_text) = text.split_once("\r\n\r\n").unwrap();
        let v: Value = serde_json::from_str(json_text).unwrap();
        assert_eq!(
            (
                v["error"]["code"].as_u64(),
                v["error"]["metadata"]["reason"].as_str()
            ),
            (Some(401), Some("UNAUTHORIZED")),
            "{text}"
        );
    }
    tx.send(()).unwrap();
    task.await.unwrap().unwrap();
}

// ------------------------------------------------------------------ router API (spec §4.15)

fn is_num(v: &Value) -> bool {
    v.is_number()
}

/// Every field of router `api.rs:95-201` (and of router-client.ts's
/// `RouteResponse`) with its JSON type.
fn assert_route_shape(v: &Value) {
    assert_eq!(v["schema_version"], "1.1");
    assert!(v["request_id"].as_str().unwrap().starts_with("req_"));
    let d = &v["decision"];
    assert!(d["task_id"].is_i64(), "{d}");
    assert!(d["task_label"].is_string());
    assert!(d["taxonomy_id"].is_string());
    for k in ["confidence", "raw_confidence", "margin", "novelty_score"] {
        assert!(is_num(&d[k]), "decision.{k}: {d}");
    }
    assert!((0.0..=1.0).contains(&d["confidence"].as_f64().unwrap()));
    assert!(d["confident"].is_boolean() && d["is_novel"].is_boolean());
    assert!(["router", "cache", "oracle"].contains(&d["source"].as_str().unwrap()));
    assert!(d["flags"].is_array());
    let c = &d["complexity"];
    assert!(is_num(&c["score"]) && c["tier"].is_string());
    for k in ["base", "ambiguity", "novelty", "margin", "length"] {
        assert!(is_num(&c["factors"][k]), "complexity.factors.{k}");
    }
    let scores = v["scores"].as_array().unwrap();
    assert!(!scores.is_empty());
    for s in scores {
        assert!(s["task_id"].is_u64() && s["task_label"].is_string());
        for k in ["probability", "score", "reconstruction_error"] {
            assert!(is_num(&s[k]), "scores[].{k}");
        }
    }
    for w in scores.windows(2) {
        assert!(
            w[0]["score"].as_f64() >= w[1]["score"].as_f64(),
            "scores sorted"
        );
    }
    if !d["is_novel"].as_bool().unwrap() {
        assert_eq!(scores[0]["task_id"].as_i64(), d["task_id"].as_i64());
        assert_eq!(scores[0]["task_label"], d["task_label"]);
    }
    assert_eq!(v["usage"]["billable_decisions"], 1);
    assert!(v["usage"]["oracle_calls"].is_u64());
    let m = &v["meta"];
    assert!(
        m["model_version"]
            .as_str()
            .unwrap()
            .starts_with("cortiq/decision@")
    );
    assert!(m["taxonomy_version"].as_str().unwrap().contains('@'));
    assert!(is_num(&m["latency_ms"]) && is_num(&m["embedding_latency_ms"]));
    assert_eq!(
        m["served_by"],
        json!(format!("cortiq/{}", env!("CARGO_PKG_VERSION")))
    );
    if let Some(o) = v.get("oracle") {
        assert!(o["consulted"].is_boolean());
    }
    if let Some(e) = v.get("explanation") {
        assert!(e["top1_vs_top2"].is_string() && e["decision_path"].is_string());
    }
}

#[tokio::test]
async fn router_api_route_fields_invariants_and_recipes() {
    let mut c = cfg();
    c.routing_tiers.insert("low".into(), "small-model".into());
    c.routing_tiers.insert("medium".into(), "mid-model".into());
    c.routing_tiers.insert("high".into(), "big-model".into());
    let srv = Srv::open(c);
    let (key, _) = srv.key(json!({"account": "routes"})).await;

    // router-client.ts's request.
    let req = json!({
        "input": {"text": accepted()},
        "options": {"policy_profile": "balanced", "allow_oracle": true, "return_explanation": false, "top_k": 3},
        "taxonomy_id": "topics",
    });
    let r = srv.post("/v1/route", Some(&key), &req).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_route_shape(&r.body);
    assert_eq!(r.body["request_id"], json!(r.request_id()));
    let d = &r.body["decision"];
    assert_eq!(d["taxonomy_id"], "topics");
    assert_eq!(d["source"], "router");
    assert_eq!(d["confident"], true);
    assert_eq!(d["is_novel"], false);
    assert_eq!(d["flags"], json!([]));
    assert_eq!(r.body["scores"].as_array().unwrap().len(), 3, "top_k 3");
    assert!(
        r.body.get("routing").is_none(),
        "routing only with routing_table_id"
    );
    assert!(r.body.get("oracle").is_none());
    assert!(r.body.get("explanation").is_none());
    assert!(r.body.get("client_request_id").is_none());
    assert_eq!(r.body["meta"]["taxonomy_version"], "topics@1");
    assert_eq!(r.body["cmf"]["action"], "local");
    assert_eq!(r.body["cmf"]["certified"], true);
    // The same decision as the decisions API.
    let j = srv
        .post("/v1/decisions", Some(&key), &topics_body(accepted()))
        .await;
    let a = &j.body["answers"]["task"];
    assert_eq!(d["task_label"], a["choice"]);
    assert_eq!(
        r.body["scores"][0]["probability"],
        a["probabilities"][d["task_label"].as_str().unwrap()]
    );
    assert_eq!(
        d["confidence"],
        j.body["cmf"]["questions"]["task"]["gate"]["p_top"]
    );
    assert_eq!(
        d["complexity"],
        j.body["cmf"]["questions"]["task"]["complexity"]
    );
    assert_eq!(
        d["task_id"].as_u64(),
        TOPICS
            .iter()
            .position(|l| *l == d["task_label"].as_str().unwrap())
            .map(|i| i as u64)
    );

    // Options: top_k, explanation, routing table, client id.
    let r = srv
        .post(
            "/v1/route",
            Some(&key),
            &json!({"taxonomy_id": "topics", "input": {"text": accepted()}, "client_request_id": "c-1",
                    "options": {"top_k": 100, "return_explanation": true, "routing_table_id": "default"}}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_route_shape(&r.body);
    assert_eq!(r.body["client_request_id"], "c-1");
    assert_eq!(
        r.body["scores"].as_array().unwrap().len(),
        4,
        "clamped to the candidates"
    );
    let e = &r.body["explanation"];
    assert!(
        e["top1_vs_top2"].as_str().unwrap().contains(" leads "),
        "{e}"
    );
    assert_eq!(e["decision_path"], "router:certified");
    let rt = &r.body["routing"];
    assert!(["small-model", "mid-model", "big-model"].contains(&rt["target"].as_str().unwrap()));
    assert!(rt["reason"].as_str().unwrap().contains("@ complexity"));
    let r1 = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics", "input": {"text": accepted()}, "options": {"top_k": 0}}))
        .await;
    assert_eq!(r1.body["scores"].as_array().unwrap().len(), 1);
    // Profiles (spec §4.7b): cost-saver is the θ gate only and never
    // certified; quality-first adds the router's margin and novelty bounds.
    let cs = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics", "input": {"text": accepted()}, "options": {"policy_profile": "cost-saver", "return_explanation": true}}))
        .await;
    assert_eq!(cs.body["cmf"]["certified"], false);
    assert_eq!(
        cs.body["explanation"]["decision_path"],
        "router:uncertified"
    );
    assert_eq!(cs.body["decision"]["source"], "router");
    let qf = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics", "input": {"text": accepted()}, "options": {"policy_profile": "quality-first"}}))
        .await;
    assert_eq!(qf.status, 200);
    assert_route_shape(&qf.body);
    let g = &qf.body["cmf"]["question"]["gate"];
    assert_eq!(g["profile"], "quality-first");
    let qf_ok = g["margin"].as_f64().unwrap() >= 0.08
        && g["novelty"].as_f64().unwrap() <= g["theta"].as_f64().unwrap().min(0.5)
        && g["p_top"].as_f64().unwrap() >= g["tau"].as_f64().unwrap();
    assert_eq!(qf.body["decision"]["confident"], json!(qf_ok));

    // A gate-rejected text without the oracle: router low_confidence semantics.
    let r = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics", "input": {"text": REJECTED}, "options": {"return_explanation": true}}))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_route_shape(&r.body);
    let d = &r.body["decision"];
    assert_eq!(d["source"], "router");
    assert_eq!(d["confident"], false);
    assert_eq!(d["flags"], json!(["low_confidence", "oracle_disabled"]));
    assert_eq!(r.body["explanation"]["decision_path"], "escalate→disabled");
    let esc = srv.get("/v1/escalations", Some(&key)).await;
    assert_eq!(esc.status, 200);
    assert_eq!(esc.body["schema_version"], "1.1");
    let rec = &esc.body["records"][0];
    assert_eq!(rec["request_id"], r.body["request_id"]);
    assert_eq!(rec["source"], "router");
    assert_eq!(rec["taxonomy_id"], "topics");
    assert_eq!(rec["router_label"], d["task_label"]);
    for f in [
        "ts",
        "final_label",
        "agreement_with_router",
        "oracle_model",
        "oracle_latency_ms",
        "oracle_calls",
        "novelty_score",
        "flags",
    ] {
        assert!(rec.get(f).is_some(), "escalation record field {f}");
    }
    let expect_esc = 1 + u64::from(!qf_ok);
    assert_eq!(esc.body["summary"]["total"].as_u64(), Some(expect_esc));

    // scripts/eval.py's request (open mode, no taxonomy): default_skill.
    let mut dc = cfg();
    dc.default_skill = Some("topics".into());
    let open = Srv::open(dc);
    let r = open
        .post("/v1/route", None, &json!({"input": {"text": accepted()}, "options": {"allow_oracle": false, "policy_profile": "balanced"}}))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert!(r.body["decision"]["task_label"].is_string());
    assert!(r.body["decision"]["source"].is_string());
    assert!(r.body["meta"]["latency_ms"].is_number());
    let tx = open.get("/v1/taxonomies", None).await;
    assert_eq!(
        tx.body["taxonomies"][0]["taxonomy_id"], "topics",
        "default skill first"
    );
    // Without a default and with two skills, the taxonomy is required.
    let r = srv
        .post(
            "/v1/route",
            Some(&key),
            &json!({"input": {"text": accepted()}}),
        )
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".to_string()));

    // Router errors and codes.
    let r = srv
        .post(
            "/v1/route",
            Some(&key),
            &json!({"taxonomy_id": "nope", "input": {"text": "x"}}),
        )
        .await;
    assert_eq!(r.router_error(), (404, "TAXONOMY_NOT_FOUND".to_string()));
    assert_eq!(r.body["error"]["details"]["taxonomy_id"], "nope");
    for input in [json!({}), json!({"text": "   "})] {
        let r = srv
            .post(
                "/v1/route",
                Some(&key),
                &json!({"taxonomy_id": "topics", "input": input}),
            )
            .await;
        assert_eq!(r.router_error(), (400, "EMBEDDING_REQUIRED".to_string()));
    }
    // Bodies the router's axum extractor rejects: 422 with axum's text.
    let r = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics"}))
        .await;
    assert_eq!(r.status, 422, "{}", r.text);
    assert!(r.text.contains("missing field `input`"), "{}", r.text);
    let r = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics", "input": {"text": "x"}, "options": {"policy_profile": "yolo"}}))
        .await;
    assert_eq!(r.status, 422, "{}", r.text);
    assert!(r.text.contains("unknown variant `yolo`"), "{}", r.text);
    let r = srv.post("/v1/route", None, &req).await;
    assert_eq!(r.router_error(), (401, "UNAUTHORIZED".to_string()));
    // Unknown fields are ignored like the router's serde.
    let r = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics", "input": {"text": accepted(), "lang": "en"}, "options": {"foo": 1}, "extra": true}))
        .await;
    assert_eq!(r.status, 200, "{}", r.text);

    // Taxonomies.
    let t = srv.get("/v1/taxonomies", Some(&key)).await;
    assert_eq!(t.status, 200);
    assert_eq!(t.body["schema_version"], "1.1");
    let list = t.body["taxonomies"].as_array().unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0]["taxonomy_id"], "topics");
    assert_eq!(list[0]["labels"], json!(TOPICS));
    assert_eq!(list[0]["taxonomy_version"], "topics@1");
    assert!(
        list[0]["model_version"]
            .as_str()
            .unwrap()
            .starts_with("cortiq/decision@")
    );
    let one = srv.get("/v1/taxonomies/shop", Some(&key)).await;
    assert_eq!(one.body["labels"], json!(SHOP));
    let r = srv.get("/v1/taxonomies/nope", Some(&key)).await;
    assert_eq!(r.router_error(), (404, "TAXONOMY_NOT_FOUND".to_string()));

    // Usage in the router's shape.
    let u = srv.get("/v1/usage", Some(&key)).await;
    assert_eq!(u.body["schema_version"], "1.1");
    let a = &u.body["account"];
    assert_eq!(a["id"], "routes");
    for f in [
        "billable_decisions",
        "oracle_calls",
        "decision_quota",
        "rate_per_min",
    ] {
        assert!(a[f].is_u64(), "account.{f}");
    }
    for f in [
        "billable_decisions",
        "oracle_calls",
        "escalations",
        "escalation_rate",
        "cache_hits",
        "novelty_hits",
        "refits",
        "promotions",
    ] {
        assert!(u.body["usage"][f].is_number(), "usage.{f}");
    }
    assert_eq!(u.body["usage"]["escalations"].as_u64(), Some(expect_esc));
}

#[tokio::test]
async fn router_batch_embeddings_and_feedback() {
    let srv = Srv::open(cfg());
    // This key may teach the model; the one of "someone-else" below may not.
    let (key, _) = srv
        .key(json!({"account": "batcher", "learning_allowed": true}))
        .await;
    let texts = [
        accepted().to_string(),
        REJECTED.to_string(),
        "invoice refund payment".to_string(),
    ];
    let inputs: Vec<Value> = texts.iter().map(|t| json!({"text": t})).collect();
    let r = srv
        .post(
            "/v1/route:batch",
            Some(&key),
            &json!({"taxonomy_id": "topics", "inputs": inputs, "options": {"top_k": 2}}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["schema_version"], "1.1");
    let results = r.body["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    let mut ids = std::collections::HashSet::new();
    for (res, t) in results.iter().zip(&texts) {
        assert_route_shape(res);
        assert!(ids.insert(res["request_id"].as_str().unwrap().to_string()));
        let single = srv
            .post(
                "/v1/route",
                Some(&key),
                &json!({"taxonomy_id": "topics", "input": {"text": t}, "options": {"top_k": 2}}),
            )
            .await;
        assert_eq!(res["decision"], single.body["decision"]);
        assert_eq!(res["scores"], single.body["scores"]);
    }
    let u = srv.get("/v1/usage", Some(&key)).await;
    assert_eq!(
        u.body["usage"]["billable_decisions"], 6,
        "one decision per input"
    );
    let too_many: Vec<Value> = (0..1025).map(|_| json!({"text": "x"})).collect();
    let r = srv
        .post(
            "/v1/route:batch",
            Some(&key),
            &json!({"taxonomy_id": "topics", "inputs": too_many}),
        )
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".to_string()));
    // A failing input fails the batch; the inputs before it were decided and
    // billed (the router's `?` in its batch loop).
    let r = srv
        .post(
            "/v1/route:batch",
            Some(&key),
            &json!({"taxonomy_id": "topics", "inputs": [{"text": "x"}, {}]}),
        )
        .await;
    assert_eq!(r.router_error(), (400, "EMBEDDING_REQUIRED".to_string()));

    // Bring your own embedding: the signal of the text decides the same.
    let x = encoder().signal(accepted());
    let by_text = srv
        .post(
            "/v1/route",
            Some(&key),
            &json!({"taxonomy_id": "topics", "input": {"text": accepted()}}),
        )
        .await;
    let by_vec = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics", "input": {"embedding": x, "embedding_model": "cortiq-decision-ph-v1"}}))
        .await;
    assert_eq!(by_vec.status, 200, "{}", by_vec.text);
    assert_route_shape(&by_vec.body);
    assert_eq!(by_vec.body["scores"], by_text.body["scores"]);
    let (dv, dt) = (&by_vec.body["decision"], &by_text.body["decision"]);
    for f in [
        "task_id",
        "task_label",
        "confidence",
        "raw_confidence",
        "margin",
        "novelty_score",
        "is_novel",
        "confident",
    ] {
        assert_eq!(dv[f], dt[f], "decision.{f}");
    }
    assert_eq!(by_vec.body["cmf"]["certified"], false, "no string state");
    assert_eq!(by_vec.body["cmf"]["usage"]["local"]["input_tokens"], 0);
    let r = srv
        .post("/v1/route", Some(&key), &json!({"taxonomy_id": "topics", "input": {"embedding": [0.0, 1.0], "embedding_model": "m"}}))
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".to_string()));
    assert!(
        r.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("dim 2")
    );
    let r = srv
        .post(
            "/v1/route",
            Some(&key),
            &json!({"taxonomy_id": "topics", "input": {"embedding": x}}),
        )
        .await;
    assert_eq!(r.router_error(), (400, "INVALID_REQUEST".to_string()));
    assert!(
        r.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("embedding_model")
    );
    let u2 = srv.get("/v1/usage", Some(&key)).await;
    assert_eq!(
        u2.body["usage"]["billable_decisions"], 9,
        "6 + the batch's first input + the text + the embedding decision"
    );

    // Router feedback: {request_id, correct_task_label}.
    let rid = by_text.body["request_id"].as_str().unwrap().to_string();
    let other_label = if by_text.body["decision"]["task_label"] == "cards" {
        "billing"
    } else {
        "cards"
    };
    let (other, _) = srv.key(json!({"account": "someone-else"})).await;
    let fb = json!({"request_id": rid, "correct_task_label": other_label});
    let r = srv.post("/v1/feedback", Some(&other), &fb).await;
    assert_eq!(
        r.router_error(),
        (404, "INVALID_REQUEST".to_string()),
        "another account's decision"
    );
    let r = srv.post("/v1/feedback", Some(&key), &fb).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["schema_version"], "1.1");
    assert_eq!(r.body["accepted"], true);
    assert_eq!(
        r.body["message"],
        json!(format!("feedback recorded for '{other_label}'"))
    );
    assert_eq!(r.body["cmf"]["weight"], 3.0);
    let r = srv.post("/v1/feedback", Some(&key), &fb).await;
    assert_eq!(r.router_error(), (404, "INVALID_REQUEST".to_string()));
    assert!(
        r.body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("already consumed")
    );
    let vid = by_vec.body["request_id"].as_str().unwrap();
    let r = srv
        .post(
            "/v1/feedback",
            Some(&key),
            &json!({"request_id": vid, "correct_task_label": other_label}),
        )
        .await;
    assert_eq!(
        r.status, 200,
        "an embedding decision takes feedback too: {}",
        r.text
    );
    // Decisions-API feedback on the same endpoint.
    let j = srv
        .post("/v1/decisions", Some(&key), &topics_body(accepted()))
        .await;
    let r = srv
        .post(
            "/v1/feedback",
            Some(&key),
            &json!({"id": j.body["id"], "question": "task", "label": other_label}),
        )
        .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["schema_version"], "1.1");
    assert!(r.body["message"].is_string());
    let r = srv
        .post(
            "/v1/feedback",
            Some(&key),
            &json!({"id": "cmf-dec-1-nope", "question": "task", "label": "cards"}),
        )
        .await;
    assert_eq!(r.router_error(), (404, "INVALID_REQUEST".to_string()));
    let learning = srv.admin("GET", "/v1/admin/learning", None).await;
    assert_eq!(learning.body["feedback"], 3);
    // A key without learning_allowed: its feedback (a new label here) is
    // answered and consumed, and teaches nothing.
    let mine = srv
        .post(
            "/v1/route",
            Some(&other),
            &json!({"taxonomy_id": "topics", "input": {"text": accepted()}}),
        )
        .await;
    let evil = json!({"request_id": mine.body["request_id"], "correct_task_label": "evil"});
    let r = srv.post("/v1/feedback", Some(&other), &evil).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.body["accepted"], false);
    assert_eq!(
        r.body["message"],
        "feedback for 'evil' not learned: this key may not teach the model"
    );
    let r = srv.post("/v1/feedback", Some(&other), &evil).await;
    assert_eq!(
        r.router_error(),
        (404, "INVALID_REQUEST".to_string()),
        "consumed"
    );
    let after = srv.admin("GET", "/v1/admin/learning", None).await;
    assert_eq!(
        after.body["buffer"]["examples"],
        learning.body["buffer"]["examples"]
    );
    assert!(!after.body["buffer"].to_string().contains("evil"));
    // Escalations are the caller's own.
    let e = srv.get("/v1/escalations?limit=10", Some(&other)).await;
    assert_eq!(e.body["records"], json!([]));
    let e = srv.get("/v1/escalations?limit=1", Some(&key)).await;
    assert_eq!(e.body["records"].as_array().unwrap().len(), 1);
    let e = srv.get("/v1/escalations?limit=x", Some(&key)).await;
    assert_eq!(e.status, 400);
    assert_eq!(
        e.text,
        "Failed to deserialize query string: invalid digit found in string"
    );
}
