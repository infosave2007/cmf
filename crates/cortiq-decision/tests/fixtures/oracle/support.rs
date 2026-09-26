//! Shared helpers of `tests/oracle_mock.rs` and `tests/learning_toy.rs`
//! (included with `#[path]`): the toy decision file, an in-test OpenRouter mock
//! server and a service + cascade stand on a fresh state directory.
#![allow(dead_code)]

use cortiq_decision::build::{self, TrainOptions};
use cortiq_decision::cascade::{Cascade, CascadeOptions};
use cortiq_decision::config::Config;
use cortiq_decision::container::{DecisionModel, Verify};
use cortiq_decision::generation;
use cortiq_decision::oracle::KeyLookup;
use cortiq_decision::protocol::ApiError;
use cortiq_decision::service::{
    Decided, DecisionService, Escalator, LoadedModel, ModelHandle, Principal,
};
use cortiq_decision::signal::SignalEncoder;
use cortiq_decision::statedir::StateDir;
use serde_json::{Map, Value, json};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

pub const EPOCH: u64 = 1_790_000_000;
pub const TOPICS: [&str; 4] = ["Weather", "billing", "cards", "travel"];
pub const SHOP: [&str; 3] = ["billing", "cards", "food"];
/// The fake oracle key (never a real one); tests look for its bytes on disk.
pub const TEST_KEY: &str = "sk-or-v1-TESTKEY-0123456789abcdef-cmf-decision-wp6";
pub const KEY_ENV: &str = "CMF_WP6_TEST_ORACLE_KEY";
pub const ORACLE_MODEL: &str = "deepseek/deepseek-v4.1-flash";

// ------------------------------------------------------------------ toy file

pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub fn pool(label: &str) -> &'static [&'static str] {
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
        // Out of every training pool: a sub-topic the oracle teaches.
        "cruise" => &[
            "cruise", "ship", "cabin", "deck", "ocean", "port", "voyage", "sail", "yacht",
            "harbor", "ferry", "captain",
        ],
        _ => unreachable!("{label}"),
    }
}

pub const FILLER: [&str; 8] = [
    "please", "help", "my", "the", "today", "need", "about", "with",
];

/// `per_label` texts per label: 2–4 pool words and 1–2 fillers, shuffled.
pub fn synth(labels: &[&str], per_label: usize, seed: u64, tag: &str) -> Vec<(String, String)> {
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

pub fn jsonl(rows: &[(String, String)]) -> String {
    rows.iter()
        .map(|(t, l)| json!({"text": t, "label": l}).to_string() + "\n")
        .collect()
}

pub fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

pub struct Toy {
    _dir: tempfile::TempDir,
    pub path: PathBuf,
    pub dev: Vec<(String, String)>,
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

/// The toy file: skills `topics` {Weather, billing, cards, travel} and `shop`
/// {billing, cards, food} on the toy encoder, built once per test binary.
pub fn toy() -> &'static Toy {
    static TOY: OnceLock<Toy> = OnceLock::new();
    TOY.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let enc = d.join("enc.cmf");
        let export = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/encoder");
        build::init_encoder(&export, &enc, Some(EPOCH)).expect("init toy encoder");
        let (o1, dev) = skill_opts(d, "topics", &TOPICS, 11);
        let s1 = d.join("s1.cmf");
        build::train(&enc, &o1, &s1).expect("train topics");
        let (o2, _) = skill_opts(d, "shop", &SHOP, 21);
        let path = d.join("toy.cmf");
        build::add_skill(&s1, &o2, &path).expect("add shop");
        Toy {
            _dir: dir,
            path,
            dev,
        }
    })
}

/// The trap file: one skill `trap` with the four `topics` labels (train 30,
/// calibration 80 each) and four inactive labels `zc1`..`zc4` (one train row,
/// four calibration rows of cruise words each: no holdout row). Teaching
/// `travel` cruise texts makes those calibration rows win `travel` — accepted and
/// wrong — so the recertified gate loses its qualifying tau while the holdout
/// (which has no row of them) does not regress.
pub fn trap() -> &'static Toy {
    static TRAP: OnceLock<Toy> = OnceLock::new();
    TRAP.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let enc = d.join("enc.cmf");
        let export = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/encoder");
        build::init_encoder(&export, &enc, Some(EPOCH)).expect("init toy encoder");
        let mut train = synth(&TOPICS, 30, 31, "trt");
        let mut cal = synth(&TOPICS, 80, 32, "trc");
        let dev = synth(&TOPICS, 12, 33, "trd");
        let cruise = distinct_texts("cruise", 20, 34, "zc", 0.995);
        let zc = ["zc1", "zc2", "zc3", "zc4"];
        for (k, l) in zc.iter().enumerate() {
            train.push((cruise[k * 5].clone(), l.to_string()));
            for j in 1..5 {
                cal.push((cruise[k * 5 + j].clone(), l.to_string()));
            }
        }
        let mut crit = Map::new();
        for l in TOPICS.iter().chain(zc.iter()) {
            crit.insert(l.to_string(), json!(format!("The message is about {l}.")));
        }
        let q = json!({"instructions": "Which topic is the message about?", "criteria": crit});
        let mut o = TrainOptions::new("trap", vec![write(d, "trap-train.jsonl", &jsonl(&train))]);
        o.calibration = Some(write(d, "trap-cal.jsonl", &jsonl(&cal)));
        o.dev = Some(write(d, "trap-dev.jsonl", &jsonl(&dev)));
        o.question = Some(write(d, "trap-q.json", &q.to_string()));
        o.threads = 2;
        o.created_unix = Some(EPOCH);
        let path = d.join("trap.cmf");
        build::train(&enc, &o, &path).expect("train trap");
        Toy {
            _dir: dir,
            path,
            dev,
        }
    })
}

pub fn encoder() -> &'static SignalEncoder {
    static ENC: OnceLock<SignalEncoder> = OnceLock::new();
    ENC.get_or_init(|| {
        let m = DecisionModel::open(&toy().path, Verify::Light).unwrap();
        SignalEncoder::from_model(&m).unwrap().0
    })
}

pub fn phi_p(text: &str) -> Vec<f32> {
    encoder().features(text).phi_p
}

pub fn cos(a: &[f32], b: &[f32]) -> f32 {
    cortiq_decision::buffer::dot(a, b)
}

/// `count` texts of `label` words (distinct first words so that the toy
/// encoder, which reads about 22 word pieces, sees different inputs), each
/// with cos φ_P < `max_cos` to every earlier one.
pub fn distinct_texts(
    label: &str,
    count: usize,
    seed: u64,
    tag: &str,
    max_cos: f32,
) -> Vec<String> {
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

// ------------------------------------------------------------------ mock OpenRouter

/// One request the mock received.
#[derive(Clone, Debug)]
pub struct MockRequest {
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl MockRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("the body is JSON")
    }
    /// The state sent (parsed from the user message).
    pub fn state(&self) -> Value {
        let v = self.json();
        let user: Value =
            serde_json::from_str(v["messages"][1]["content"].as_str().unwrap()).unwrap();
        user["state"].clone()
    }
}

/// The mock's reply.
#[derive(Clone, Debug)]
pub struct MockReply {
    pub status: u16,
    pub body: Vec<u8>,
    pub delay: Duration,
}

type Handler = Arc<dyn Fn(&MockRequest) -> MockReply + Send + Sync>;

pub struct MockOracle {
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
    pub fn start(handler: impl Fn(&MockRequest) -> MockReply + Send + Sync + 'static) -> Self {
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

    /// An oracle that answers every choice with `label` when it is an option
    /// (else the first option), every score with 0 and every noul with true.
    pub fn answering(label: &str) -> Self {
        let label = label.to_string();
        Self::start(move |req| answer_reply(req, |_, opts| pick(opts, &label), 1.3e-5))
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    pub fn requests(&self) -> Vec<MockRequest> {
        self.requests.lock().unwrap().clone()
    }

    pub fn set(&self, handler: impl Fn(&MockRequest) -> MockReply + Send + Sync + 'static) {
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

pub fn pick(options: &[String], label: &str) -> Value {
    if options.iter().any(|o| o == label) {
        json!(label)
    } else {
        json!(options[0])
    }
}

/// The verdict object of a request: `choose(qid, options)` for choice
/// questions, 0 for scores, true for nouls.
pub fn verdicts(req: &MockRequest, choose: impl Fn(&str, &[String]) -> Value) -> Value {
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

/// An OpenRouter chat completion carrying `content`.
pub fn completion(content: &str, cost: Value, model: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "id": "gen-mock", "object": "chat.completion", "model": model, "provider": "Mock",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 7, "total_tokens": 1207, "cost": cost,
                  "prompt_tokens_details": {"cached_tokens": 1024}},
    }))
    .unwrap()
}

pub fn answer_reply(
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

pub fn raw_reply(status: u16, body: &str) -> MockReply {
    MockReply {
        status,
        body: body.as_bytes().to_vec(),
        delay: Duration::ZERO,
    }
}

// ------------------------------------------------------------------ stand

pub fn test_key_lookup() -> KeyLookup {
    Arc::new(|name: &str| (name == KEY_ENV).then(|| TEST_KEY.to_string()))
}

pub fn no_key_lookup() -> KeyLookup {
    Arc::new(|_: &str| None)
}

/// The oracle configuration of a stand (every limit at its default).
pub fn stand_config(mock_url: &str) -> Config {
    let mut c = Config::default();
    c.oracle.enabled = true;
    c.oracle.base_url = mock_url.to_string();
    c.oracle.api_key_env = KEY_ENV.to_string();
    c.oracle.deadline_s = 5.0;
    c.learning.synchronous = true;
    c
}

pub struct Stand {
    pub dir: tempfile::TempDir,
    pub base: PathBuf,
    pub state: StateDir,
    pub handle: Arc<ModelHandle>,
    pub cascade: Arc<Cascade>,
    pub svc: DecisionService,
}

impl Stand {
    pub fn open(dir: tempfile::TempDir, cfg: &Config, key: KeyLookup) -> Self {
        Self::open_on(&toy().path, dir, cfg, key)
    }

    pub fn open_on(base: &Path, dir: tempfile::TempDir, cfg: &Config, key: KeyLookup) -> Self {
        let state = StateDir::open(dir.path().join("state")).unwrap();
        let model = generation::open_served(base, &state, Verify::Full).unwrap();
        let handle = Arc::new(ModelHandle::new(LoadedModel::new(model).unwrap()));
        let cascade = Cascade::open_with(
            Arc::clone(&handle),
            cfg,
            state.clone(),
            CascadeOptions {
                key,
                threads: 2,
                created_unix: Some(EPOCH),
            },
        )
        .unwrap();
        let esc: Arc<dyn Escalator> = cascade.clone();
        let svc = DecisionService::open(Arc::clone(&handle), cfg.clone(), Some(esc))
            .unwrap()
            .with_loopback(true);
        Self {
            dir,
            base: base.to_path_buf(),
            state,
            handle,
            cascade,
            svc,
        }
    }

    pub fn new(cfg: &Config) -> Self {
        Self::open(tempfile::tempdir().unwrap(), cfg, test_key_lookup())
    }

    /// Close the service and cascade and open them again on the same state.
    pub fn restart(self, cfg: &Config) -> Self {
        let Stand {
            dir,
            base,
            svc,
            cascade,
            handle,
            state,
        } = self;
        drop(svc);
        drop(cascade);
        drop(handle);
        drop(state);
        Self::open_on(&base, dir, cfg, test_key_lookup())
    }

    /// The oracle ledger lines.
    pub fn ledger(&self) -> Vec<Value> {
        std::fs::read_to_string(self.state.oracle_ledger_path())
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    /// `oracle.state` (`null` when it was never written).
    pub fn oracle_state(&self) -> Value {
        std::fs::read(self.state.oracle_state_path())
            .map(|b| serde_json::from_slice(&b).unwrap())
            .unwrap_or(Value::Null)
    }

    pub fn decide(&self, body: &[u8]) -> Result<Decided, ApiError> {
        self.svc.decide_body(body, &Principal::open())
    }

    pub fn decide_as(&self, body: &[u8], p: &Principal) -> Result<Decided, ApiError> {
        self.svc.decide_body(body, p)
    }
}

pub fn choice(labels: &[&str]) -> Value {
    let mut c = Map::new();
    for l in labels {
        c.insert(l.to_string(), json!(format!("about {l}")));
    }
    json!({"type": "choice", "instructions": "Which topic?", "criteria": c})
}

pub fn body(state: Value, questions: Value, cmf: Option<Value>) -> Vec<u8> {
    let mut v = json!({"model": "cortiq/decision", "state": state, "questions": questions});
    if let Some(c) = cmf {
        v["cmf"] = c;
    }
    serde_json::to_vec(&v).unwrap()
}

/// One `topics` question about `text`.
pub fn topics_body(text: &str) -> Vec<u8> {
    body(json!(text), json!({"task": choice(&TOPICS)}), None)
}

/// Every file under `dir`, recursively.
pub fn files_under(dir: &Path) -> Vec<PathBuf> {
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

/// Whether `needle` occurs in any file under `dir`.
pub fn bytes_on_disk(dir: &Path, needle: &[u8]) -> Vec<PathBuf> {
    files_under(dir)
        .into_iter()
        .filter(|p| {
            let b = std::fs::read(p).unwrap_or_default();
            b.windows(needle.len()).any(|w| w == needle)
        })
        .collect()
}
