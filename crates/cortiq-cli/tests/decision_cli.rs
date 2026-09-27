//! `cortiq decide`, `cortiq decision …` and `cortiq serve` on a decision file,
//! end to end through the `cortiq` binary on the toy model (spec decision-v4
//! §3.1, §4.1, §4.2, §5.10, §5.14, §6.1), hermetic: the oracle is an in-test
//! mock server, no key of any real service is read or passed.
//!
//! The toy file is built by the CLI itself once per test binary: `decision
//! init` of the toy encoder of `cortiq-decision`, `decision train` of skill
//! `topics` {Weather, billing, cards, travel} from two training files, and
//! `decision add-skill` of `shop` {billing, cards, food} — the data of the
//! library's toy files (same generator, same seeds).
//!
//! * build: init/train/add-skill reports, several `--train` files, `--k`,
//!   carve-out, zero-forgetting of `add-skill`, an existing skill id and an
//!   existing output refused;
//! * `decide -p`: exact, subset (`--labels`), `--json`, `--round 2`, the skill
//!   choice rules and their errors; the answer equals the batch row;
//! * `decide --input`: one row per text without the text, totals on stderr,
//!   the build's dev count reproduced, `--out`, `--bench` percentiles;
//! * `decision info`, `verify` (a corrupted copy fails);
//! * `run` (one-shot and chat) print the DECISION guard; generic `info` and
//!   `verify` still work;
//! * `serve`: the language-model flags are refused; a decision file listens on
//!   127.0.0.1 by default, answers `/healthz` and decisions, learns through the
//!   mock oracle (25 answers → generation 1); then `decide --state`,
//!   `decision verify --state`, `materialize` and `rollback` on its state;
//! * `decision keys create|list|revoke|import` and a keyed server;
//! * `serve --shadow-of`: the old router's answer byte for byte, one
//!   comparison line without the text, whose local label is `decide`'s; a
//!   plain-http non-loopback URL refused before anything is opened;
//! * `decision learn` with the mock oracle: only abstentions are asked, ledger
//!   answers are reused, the reservation ledger holds no key.

#[path = "support/toy_dir.rs"]
mod toy_dir;

use cortiq_decision::config::Config;
use cortiq_decision::container::{DecisionModel, Verify};
use cortiq_decision::eval::Evaluator;
use cortiq_decision::learn;
use cortiq_decision::oracle;
use cortiq_decision::signal::SignalEncoder;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const EPOCH: u64 = 1_790_000_000;
const TOPICS: [&str; 4] = ["Weather", "billing", "cards", "travel"];
const SHOP: [&str; 3] = ["billing", "cards", "food"];
/// The fake oracle key of the mock (never a real one).
const TEST_KEY: &str = "sk-or-v1-TESTKEY-cortiq-cli-wp8-0123456789abcdef";
const KEY_ENV: &str = "CMF_WP8_TEST_ORACLE_KEY";
const ORACLE_MODEL: &str = "deepseek/deepseek-v4.1-flash";
const GUARD: &str = "this is a DECISION model; use `cortiq decide` or `cortiq serve`";

// ------------------------------------------------------------------ the binary

fn cortiq() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_cortiq"));
    c.env("CMF_GPU", "0")
        .env("NO_COLOR", "1")
        .env("SOURCE_DATE_EPOCH", EPOCH.to_string())
        .env_remove("RUST_LOG")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("CORTIQ_DECISION_ADMIN_TOKEN")
        .env_remove(KEY_ENV)
        .stdin(Stdio::null());
    c
}

fn show(o: &Output) -> String {
    format!(
        "status {:?}\n--- stdout\n{}\n--- stderr\n{}",
        o.status.code(),
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn output(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut c = cortiq();
    c.args(args);
    for (k, v) in envs {
        c.env(k, v);
    }
    c.output().expect("run cortiq")
}

/// Run and require success; returns stdout.
fn ok(args: &[&str]) -> String {
    ok_env(args, &[])
}

fn ok_env(args: &[&str], envs: &[(&str, &str)]) -> String {
    let o = output(args, envs);
    assert!(o.status.success(), "cortiq {args:?} failed\n{}", show(&o));
    String::from_utf8(o.stdout).unwrap()
}

/// Run and require failure; returns stderr.
fn fails(args: &[&str]) -> String {
    let o = output(args, &[]);
    assert!(
        !o.status.success(),
        "cortiq {args:?} succeeded\n{}",
        show(&o)
    );
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn json_of(stdout: &str) -> Value {
    serde_json::from_str(stdout.trim()).unwrap_or_else(|e| panic!("not JSON ({e}): {stdout}"))
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn sha256_file(p: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(p).unwrap()))
}

// ------------------------------------------------------------------ toy data

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
        // Out of every training pool: the sub-topic the mock oracle teaches.
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

fn question(labels: &[&str]) -> String {
    let mut crit = Map::new();
    for l in labels {
        crit.insert(l.to_string(), json!(format!("The message is about {l}.")));
    }
    json!({"instructions": "Which topic is the message about?", "criteria": crit}).to_string()
}

/// The data files of one skill (the library toy's generator and seeds).
struct SkillFiles {
    train: Vec<PathBuf>,
    calibration: PathBuf,
    dev: PathBuf,
    question: PathBuf,
    dev_rows: Vec<(String, String)>,
    n_train: usize,
}

fn skill_files(dir: &Path, id: &str, labels: &[&str], seed: u64, parts: usize) -> SkillFiles {
    let train = synth(labels, 30, seed, &format!("{id}t"));
    let cal = synth(labels, 80, seed + 1, &format!("{id}c"));
    let dev = synth(labels, 12, seed + 2, &format!("{id}d"));
    let per = train.len().div_ceil(parts);
    let train_files = train
        .chunks(per)
        .enumerate()
        .map(|(k, c)| write(dir, &format!("{id}-train{k}.jsonl"), &jsonl(c)))
        .collect();
    SkillFiles {
        train: train_files,
        calibration: write(dir, &format!("{id}-cal.jsonl"), &jsonl(&cal)),
        dev: write(dir, &format!("{id}-dev.jsonl"), &jsonl(&dev)),
        question: write(dir, &format!("{id}-q.json"), &question(labels)),
        dev_rows: dev,
        n_train: train.len(),
    }
}

fn skill_args(f: &SkillFiles) -> Vec<String> {
    let mut a = Vec::new();
    for t in &f.train {
        a.push("--train".to_string());
        a.push(s(t).to_string());
    }
    for (flag, p) in [
        ("--calibration", &f.calibration),
        ("--dev", &f.dev),
        ("--question", &f.question),
    ] {
        a.push(flag.to_string());
        a.push(s(p).to_string());
    }
    a.extend(["--threads".to_string(), "2".to_string()]);
    a
}

fn args_ref(v: &[String]) -> Vec<&str> {
    v.iter().map(String::as_str).collect()
}

fn toy_encoder_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../cortiq-decision/tests/fixtures/toy/encoder")
}

struct Toy {
    dir: PathBuf,
    enc: PathBuf,
    s1: PathBuf,
    path: PathBuf,
    topics: SkillFiles,
    shop: SkillFiles,
    init_report: Value,
    train_report: Value,
    add_report: Value,
}

/// The toy file, built through the CLI once per test binary.
fn toy() -> &'static Toy {
    static TOY: OnceLock<Toy> = OnceLock::new();
    TOY.get_or_init(|| {
        let dir = toy_dir::toy_dir("toy");
        let d = dir.as_path();
        let enc = d.join("enc.cmf");
        let init_report = json_of(&ok(&[
            "decision",
            "init",
            "--encoder-dir",
            s(&toy_encoder_dir()),
            "-o",
            s(&enc),
            "--json",
        ]));
        let topics = skill_files(d, "topics", &TOPICS, 11, 2);
        let s1 = d.join("s1.cmf");
        let mut a: Vec<String> = [
            "decision",
            "train",
            "--encoder",
            s(&enc),
            "--skill",
            "topics",
        ]
        .iter()
        .map(|x| x.to_string())
        .collect();
        a.extend(skill_args(&topics));
        a.extend(["-o".into(), s(&s1).into(), "--json".into()]);
        let train_report = json_of(&ok(&args_ref(&a)));
        let shop = skill_files(d, "shop", &SHOP, 21, 1);
        let path = d.join("toy.cmf");
        let mut a: Vec<String> = ["decision", "add-skill", s(&s1), "--skill", "shop"]
            .iter()
            .map(|x| x.to_string())
            .collect();
        a.extend(skill_args(&shop));
        a.extend(["-o".into(), s(&path).into(), "--json".into()]);
        let add_report = json_of(&ok(&args_ref(&a)));
        Toy {
            dir,
            enc,
            s1,
            path,
            topics,
            shop,
            init_report,
            train_report,
            add_report,
        }
    })
}

fn info_json(p: &Path) -> Value {
    json_of(&ok(&["decision", "info", s(p), "--json"]))
}

fn encoder() -> &'static SignalEncoder {
    static ENC: OnceLock<SignalEncoder> = OnceLock::new();
    ENC.get_or_init(|| {
        let m = DecisionModel::open(&toy().path, Verify::Light).unwrap();
        SignalEncoder::from_model(&m).unwrap().0
    })
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `count` cruise texts, each with cos φ_P < `max_cos` to every earlier one
/// (the library's `distinct_texts`).
fn distinct_texts(count: usize, seed: u64, tag: &str, max_cos: f32) -> Vec<String> {
    let p = pool("cruise");
    let mut rng = Lcg(seed);
    let mut out: Vec<String> = Vec::new();
    let mut phis: Vec<Vec<f32>> = Vec::new();
    let mut tries = 0;
    while out.len() < count {
        tries += 1;
        assert!(tries < 100_000, "could not find {count} distinct texts");
        let mut words = Vec::new();
        for _ in 0..3 + rng.below(2) {
            words.push(p[rng.below(p.len())]);
        }
        words.push(FILLER[rng.below(FILLER.len())]);
        let t = format!("{} {tag}{}", words.join(" "), out.len());
        let f = encoder().features(&t).phi_p;
        if phis.iter().all(|q| cos(q, &f) < max_cos) {
            phis.push(f);
            out.push(t);
        }
    }
    out
}

// ------------------------------------------------------------------ mock oracle

struct MockOracle {
    addr: SocketAddr,
    hits: Arc<AtomicUsize>,
    bodies: Arc<Mutex<Vec<Vec<u8>>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn read_request(s: &mut TcpStream) -> Option<Vec<u8>> {
    read_request_parts(s).map(|(_, body)| body)
}

/// One HTTP/1.1 request: (head as sent, body by `Content-Length`).
fn read_request_parts(s: &mut TcpStream) -> Option<(String, Vec<u8>)> {
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
    let len: usize = head
        .to_ascii_lowercase()
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = s.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Some((head, body))
}

/// An OpenRouter chat completion answering every choice with `label` when it
/// is an option (else the first option).
fn completion_for(body: &[u8], label: &str) -> Vec<u8> {
    let v: Value = serde_json::from_slice(body).expect("the oracle body is JSON");
    let schema = &v["response_format"]["json_schema"]["schema"];
    let mut verdicts = Map::new();
    for q in schema["required"].as_array().unwrap() {
        let qid = q.as_str().unwrap();
        let p = &schema["properties"][qid];
        let a = match p["type"].as_str().unwrap() {
            "string" => {
                let opts = p["enum"].as_array().unwrap();
                if opts.iter().any(|o| o == label) {
                    json!(label)
                } else {
                    opts[0].clone()
                }
            }
            "integer" => json!(0),
            _ => json!(true),
        };
        verdicts.insert(qid.to_string(), a);
    }
    serde_json::to_vec(&json!({
        "id": "gen-mock", "object": "chat.completion", "model": ORACLE_MODEL, "provider": "Mock",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": Value::Object(verdicts).to_string()}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 7, "total_tokens": 1207, "cost": 1.3e-5},
    }))
    .unwrap()
}

impl MockOracle {
    fn answering(label: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (h, b, st) = (hits.clone(), bodies.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            for conn in listener.incoming() {
                if st.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(mut s) = conn else { continue };
                let (h, b) = (h.clone(), b.clone());
                std::thread::spawn(move || {
                    let Some(body) = read_request(&mut s) else {
                        return;
                    };
                    h.fetch_add(1, Ordering::SeqCst);
                    let reply = completion_for(&body, label);
                    b.lock().unwrap().push(body);
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        reply.len()
                    );
                    let _ = s.write_all(head.as_bytes());
                    let _ = s.write_all(&reply);
                    let _ = s.flush();
                });
            }
        });
        Self {
            addr,
            hits,
            bodies,
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

    /// The `state` of every request received.
    fn states(&self) -> Vec<Value> {
        self.bodies
            .lock()
            .unwrap()
            .iter()
            .map(|b| {
                let v: Value = serde_json::from_slice(b).unwrap();
                let user: Value =
                    serde_json::from_str(v["messages"][1]["content"].as_str().unwrap()).unwrap();
                user["state"].clone()
            })
            .collect()
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

fn oracle_config(dir: &Path, mock: &MockOracle, extra: Value) -> PathBuf {
    let mut cfg = json!({
        "oracle": {"enabled": true, "base_url": mock.url(), "api_key_env": KEY_ENV, "deadline_s": 5},
    });
    if let (Value::Object(c), Value::Object(e)) = (&mut cfg, extra) {
        c.extend(e);
    }
    write(dir, "oracle-config.json", &cfg.to_string())
}

// ------------------------------------------------------------------ server

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

struct Server {
    child: Child,
    port: u16,
    stdout: PathBuf,
    stderr: PathBuf,
}

impl Server {
    fn start(model: &Path, extra: &[&str], envs: &[(&str, &str)], logs: &Path) -> Self {
        let port = free_port();
        let stdout = logs.join(format!("serve-{port}.out"));
        let stderr = logs.join(format!("serve-{port}.err"));
        let mut c = cortiq();
        c.args(["serve", s(model), "--port", &port.to_string()])
            .args(extra)
            .stdout(std::fs::File::create(&stdout).unwrap())
            .stderr(std::fs::File::create(&stderr).unwrap());
        for (k, v) in envs {
            c.env(k, v);
        }
        let mut srv = Self {
            child: c.spawn().expect("spawn cortiq serve"),
            port,
            stdout,
            stderr,
        };
        let t0 = Instant::now();
        loop {
            if let Some(status) = srv.child.try_wait().unwrap() {
                panic!("cortiq serve exited early ({status})\n{}", srv.logs());
            }
            if TcpStream::connect_timeout(
                &SocketAddr::from(([127, 0, 0, 1], port)),
                Duration::from_millis(200),
            )
            .is_ok()
            {
                break;
            }
            assert!(
                t0.elapsed() < Duration::from_secs(180),
                "cortiq serve did not listen\n{}",
                srv.logs()
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        srv
    }

    fn logs(&self) -> String {
        format!(
            "--- stdout\n{}\n--- stderr\n{}",
            std::fs::read_to_string(&self.stdout).unwrap_or_default(),
            std::fs::read_to_string(&self.stderr).unwrap_or_default()
        )
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    /// SIGTERM (graceful: the usage ledger is flushed) and wait.
    fn stop(mut self) -> String {
        #[cfg(unix)]
        unsafe {
            libc::kill(self.child.id() as i32, libc::SIGTERM);
        }
        #[cfg(not(unix))]
        let _ = self.child.kill();
        let t0 = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                let logs = self.logs();
                #[cfg(unix)]
                assert!(status.success(), "serve exit {status}\n{logs}");
                let _ = status;
                return logs;
            }
            if t0.elapsed() > Duration::from_secs(60) {
                let _ = self.child.kill();
                panic!("cortiq serve did not stop\n{}", self.logs());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One HTTP request: (status, JSON body or Null).
fn http(method: &str, url: &str, key: Option<&str>, body: Option<&Value>) -> (u16, Value) {
    // A fresh agent without idle connections: a graceful shutdown of the
    // server does not wait on a pooled keep-alive connection.
    let agent = ureq::AgentBuilder::new()
        .max_idle_connections(0)
        .timeout(Duration::from_secs(60))
        .build();
    let mut req = agent.request(method, url);
    if let Some(k) = key {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }
    let res = match body {
        Some(b) => req
            .set("Content-Type", "application/json")
            .send_string(&b.to_string()),
        None => req.call(),
    };
    let resp = match res {
        Ok(r) => r,
        Err(ureq::Error::Status(_, r)) => r,
        Err(e) => panic!("{method} {url}: {e}"),
    };
    let status = resp.status();
    let text = resp.into_string().unwrap();
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

fn topics_request(text: &str) -> Value {
    let mut crit = Map::new();
    for l in TOPICS {
        crit.insert(l.to_string(), json!(format!("about {l}")));
    }
    json!({"model": "cortiq/decision", "state": text,
           "questions": {"task": {"type": "choice", "instructions": "Which topic?", "criteria": crit}}})
}

fn decide_json(model: &Path, text: &str, extra: &[&str]) -> Value {
    let mut a = vec!["decide", s(model), "-p", text, "--json"];
    a.extend_from_slice(extra);
    json_of(&ok(&a))
}

// ------------------------------------------------------------------ build

#[test]
fn init_train_and_add_skill_build_the_toy_file() {
    let t = toy();
    // init: an encoder-only file.
    let i = &t.init_report;
    assert_eq!(i["path"], s(&t.enc));
    assert_eq!(i["sha256"], sha256_file(&t.enc));
    assert!(i["model"].as_str().unwrap().starts_with("cortiq/decision@"));
    let enc_info = info_json(&t.enc);
    assert_eq!(enc_info["skills"], json!([]));
    assert_eq!(enc_info["manifest"]["created_unix"], EPOCH);

    // train: two training files read in order, the calibration file, dev.
    let r = &t.train_report;
    assert_eq!(r["skills"], json!(["topics"]));
    assert_eq!(r["skill"]["id"], "topics");
    assert_eq!(r["skill"]["labels"], 4);
    assert_eq!(r["skill"]["K"], 16);
    assert_eq!(r["skill"]["rows"]["train"], t.topics.n_train);
    assert_eq!(r["skill"]["calibration_source"], "file");
    assert_eq!(r["self_check"]["bit_exact"], true);
    assert_eq!(r["dev"]["n"], t.topics.dev_rows.len());
    assert_eq!(r["out"]["sha256"], sha256_file(&t.s1));
    let topics = &info_json(&t.s1)["skills"][0];
    let parts = topics["data"]["train"]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(
        parts.iter().map(|p| p["n"].as_u64().unwrap()).sum::<u64>(),
        t.topics.n_train as u64
    );

    // add-skill: every earlier skill byte for byte (spec §3.7).
    let a = &t.add_report;
    assert_eq!(a["skills"], json!(["topics", "shop"]));
    assert_eq!(a["skill"]["id"], "shop");
    assert_eq!(a["dev"]["n"], t.shop.dev_rows.len());
    let info = info_json(&t.path);
    assert_eq!(info["skills"][0], *topics);
    assert_eq!(
        info["manifest"]["skills"][0],
        info_json(&t.s1)["manifest"]["skills"][0]
    );
    assert_eq!(info["summary"][1]["id"], "shop");
    assert_eq!(info["summary"][1]["labels"], 3);

    // An existing skill id is refused; an existing output is never replaced.
    let d = t.dir.as_path();
    let mut a: Vec<String> = ["decision", "add-skill", s(&t.path), "--skill", "topics"]
        .iter()
        .map(|x| x.to_string())
        .collect();
    a.extend(skill_args(&t.topics));
    let again = d.join("again.cmf");
    a.extend(["-o".into(), s(&again).into()]);
    let e = fails(&args_ref(&a));
    assert!(e.contains("already exists"), "{e}");
    assert!(!again.exists());
    let before = sha256_file(&t.path);
    let mut a: Vec<String> = [
        "decision",
        "train",
        "--encoder",
        s(&t.enc),
        "--skill",
        "other",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    a.extend(skill_args(&t.topics));
    a.extend(["-o".into(), s(&t.path).into()]);
    let e = fails(&args_ref(&a));
    assert!(e.contains("overwrite") || e.contains("exists"), "{e}");
    assert_eq!(sha256_file(&t.path), before);
}

#[test]
fn train_takes_several_train_files_k_and_a_carve_out() {
    let t = toy();
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let f = skill_files(d, "small", &["billing", "food", "travel"], 41, 3);
    let out = d.join("k4.cmf");
    let mut a: Vec<String> = [
        "decision",
        "train",
        "--encoder",
        s(&t.enc),
        "--skill",
        "small",
    ]
    .iter()
    .map(|x| x.to_string())
    .collect();
    for p in &f.train {
        a.extend(["--train".into(), s(p).into()]);
    }
    a.extend(
        [
            "--k",
            "4",
            "--k-source",
            "cv.json chosen_K",
            "--threads",
            "1",
            "-o",
            s(&out),
            "--json",
        ]
        .iter()
        .map(|x| x.to_string()),
    );
    let r = json_of(&ok(&args_ref(&a)));
    assert_eq!(r["skill"]["K"], 4);
    assert!(r["skill"]["k_max"].as_u64().unwrap() <= 4);
    // No --calibration: carved out of the training rows (every fifth row of a
    // label in sha256 order).
    assert_eq!(r["skill"]["calibration_source"], "carve-out");
    let n_train = r["skill"]["rows"]["train"].as_u64().unwrap();
    let n_cal = r["skill"]["rows"]["calibration"].as_u64().unwrap();
    assert_eq!(n_train + n_cal, f.n_train as u64);
    assert_eq!(n_cal, f.n_train as u64 / 5);
    let m = &info_json(&out)["skills"][0];
    assert_eq!(m["recipe"]["K"], 4);
    assert_eq!(m["recipe"]["k_source"], "cv.json chosen_K");
    assert_eq!(m["data"]["train"]["parts"].as_array().unwrap().len(), 3);
    assert!(
        m["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|x| x["k"].as_u64().unwrap() <= 4)
    );
    // The human report.
    let out2 = d.join("k4-human.cmf");
    let n = a.len();
    a[n - 2] = s(&out2).into();
    a.pop();
    let human = ok(&args_ref(&a));
    assert!(
        human.contains("skill small: 3 labels (3 active), K 4"),
        "{human}"
    );
    assert!(human.contains("self-check:  bit-exact"), "{human}");
    // Same inputs, same bytes.
    assert_eq!(sha256_file(&out), sha256_file(&out2));
}

// ------------------------------------------------------------------ decide

#[test]
fn decide_one_text_exact_subset_json_and_round() {
    let t = toy();
    let (text, _) = &t.topics.dev_rows[0];
    let v = decide_json(&t.path, text, &["--skill", "topics"]);
    assert!(v["id"].as_str().unwrap().starts_with("cmf-dec-"));
    assert!(v["model"].as_str().unwrap().starts_with("cortiq/decision@"));
    let ans = &v["answers"]["task"];
    assert_eq!(ans["type"], "choice");
    let probs = ans["probabilities"].as_object().unwrap();
    assert_eq!(
        probs.keys().collect::<Vec<_>>(),
        TOPICS.iter().collect::<Vec<_>>()
    );
    let q = &v["cmf"]["questions"]["task"];
    assert_eq!(
        (q["skill"].as_str(), q["match"].as_str()),
        (Some("topics"), Some("exact"))
    );
    assert!(matches!(q["action"].as_str(), Some("local" | "abstain")));
    assert_eq!(v["cmf"]["usage"]["oracle"]["calls"], 0);

    // --round 2: hundredths, as Jev.
    let r = decide_json(&t.path, text, &["--skill", "topics", "--round", "2"]);
    for p in r["answers"]["task"]["probabilities"]
        .as_object()
        .unwrap()
        .values()
    {
        let x = p.as_f64().unwrap() * 100.0;
        assert!((x - x.round()).abs() < 1e-6, "{p}");
    }
    assert_eq!(r["answers"]["task"]["choice"], ans["choice"]);

    // --labels: a strict subset of one skill's labels, or exactly a skill.
    let sub = decide_json(&t.path, text, &["--labels", "Weather,travel"]);
    let q = &sub["cmf"]["questions"]["task"];
    assert_eq!(
        (q["skill"].as_str(), q["match"].as_str()),
        (Some("topics"), Some("subset"))
    );
    assert_eq!(q["certified"], false);
    let shop = decide_json(
        &t.path,
        "pizza and coffee please",
        &["--labels", "billing,cards,food"],
    );
    let q = &shop["cmf"]["questions"]["task"];
    assert_eq!(
        (q["skill"].as_str(), q["match"].as_str()),
        (Some("shop"), Some("exact"))
    );
    // Two skills have both labels: untrained, and decide never asks an oracle.
    let e = fails(&[
        "decide",
        s(&t.path),
        "-p",
        text,
        "--labels",
        "billing,cards",
    ]);
    assert!(
        e.contains("422") && e.contains("UNSUPPORTED_QUESTION"),
        "{e}"
    );
    // The skill rules of spec §4.1.
    let e = fails(&["decide", s(&t.path), "-p", text]);
    assert!(e.contains("topics, shop"), "{e}");
    let e = fails(&["decide", s(&t.path), "-p", text, "--skill", "nope"]);
    assert!(e.contains("no skill 'nope'"), "{e}");
    assert!(ok(&["decide", s(&t.s1), "-p", text]).contains("skill:      topics (exact match"));

    // The human answer.
    let h = ok(&["decide", s(&t.path), "-p", text, "--skill", "topics"]);
    assert!(
        h.starts_with(&format!(
            "choice:     {}\n",
            ans["choice"].as_str().unwrap()
        )),
        "{h}"
    );
    assert!(h.contains("gate:       p_top "), "{h}");
    assert!(h.contains("model:      cortiq/decision@"), "{h}");

    // A text no skill knows: the gate rejects it, the answer stays local.
    let v = decide_json(
        &t.path,
        "cruise ship cabin deck please",
        &["--skill", "topics"],
    );
    let q = &v["cmf"]["questions"]["task"];
    assert_eq!(q["action"], "abstain");
    assert!(
        q["flags"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f == "oracle_disabled"),
        "{q}"
    );
    let h = ok(&[
        "decide",
        s(&t.path),
        "-p",
        "cruise ship cabin deck please",
        "--skill",
        "topics",
    ]);
    assert!(h.contains("never calls the oracle"), "{h}");

    // An encoder-only file has no skill; a language model is not a decision file.
    let e = fails(&["decide", s(&t.enc), "-p", text]);
    assert!(e.contains("no skill"), "{e}");
}

fn rows_of(stdout: &str) -> Vec<Value> {
    stdout
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn summary_of(stderr: &str) -> Value {
    let line = stderr
        .lines()
        .rev()
        .find(|l| l.starts_with("{\"summary\""))
        .unwrap_or_else(|| panic!("no summary line: {stderr}"));
    serde_json::from_str::<Value>(line).unwrap()["summary"].clone()
}

#[test]
fn decide_batch_rows_summary_bench_and_single_equality() {
    let t = toy();
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let dev = &t.topics.dev;
    let n = t.topics.dev_rows.len();

    let o = output(
        &["decide", s(&t.path), "--input", s(dev), "--skill", "topics"],
        &[],
    );
    assert!(o.status.success(), "{}", show(&o));
    let rows = rows_of(&String::from_utf8(o.stdout.clone()).unwrap());
    assert_eq!(rows.len(), n);
    let keys = [
        "i",
        "text_sha256",
        "skill",
        "choice",
        "p_top",
        "confidence",
        "novelty",
        "margin",
        "accepted",
        "certified",
        "input_tokens",
        "errors_top5",
        "timings_us",
        "correct",
    ];
    for (i, r) in rows.iter().enumerate() {
        let m = r.as_object().unwrap();
        assert_eq!(
            m.keys().collect::<Vec<_>>(),
            keys.iter().collect::<Vec<_>>()
        );
        assert_eq!(r["i"], i);
        assert_eq!(r["skill"], "topics");
        let (text, label) = &t.topics.dev_rows[i];
        assert_eq!(
            r["text_sha256"],
            format!("{:x}", Sha256::digest(text.as_bytes()))
        );
        assert_eq!(r["correct"], json!(r["choice"] == json!(label)));
        assert!(r["errors_top5"].as_object().unwrap().len() <= 5);
    }
    let raw = String::from_utf8_lossy(&o.stdout);
    for (text, _) in &t.topics.dev_rows {
        assert!(!raw.contains(text.as_str()), "a text leaked into the rows");
    }
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("skill topics (4 labels"), "{stderr}");
    let sum = summary_of(&stderr);
    let correct = rows.iter().filter(|r| r["correct"] == true).count();
    let accepted = rows.iter().filter(|r| r["accepted"] == true).count();
    assert_eq!(
        (sum["n"].as_u64(), sum["labelled"].as_u64()),
        (Some(n as u64), Some(n as u64))
    );
    assert_eq!(sum["correct"], correct);
    assert_eq!(sum["accepted"], accepted);
    // The build's own dev count (the same scorer) is reproduced.
    assert_eq!(sum["correct"], t.train_report["dev"]["correct"]);
    assert_eq!(sum["accepted"], t.train_report["dev"]["accepted"]);
    assert!(sum.get("bench").is_none());

    // --bench --out: rows in the file, percentiles of every stage.
    let out = d.join("rows.jsonl");
    let o = output(
        &[
            "decide",
            s(&t.path),
            "--input",
            s(dev),
            "--skill",
            "topics",
            "--bench",
            "--out",
            s(&out),
        ],
        &[],
    );
    assert!(o.status.success(), "{}", show(&o));
    assert!(o.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("bench (warm-up"), "{stderr}");
    let bench = &summary_of(&stderr)["bench"];
    assert_eq!(bench["warmup"], n.min(50));
    for stage in ["tokenize", "encode", "hash", "resonance", "total"] {
        for p in ["p50_us", "p95_us", "p99_us", "mean_us"] {
            assert!(bench[stage][p].as_f64().unwrap() >= 0.0, "{stage} {p}");
        }
        assert!(bench[stage]["p50_us"].as_f64() <= bench[stage]["p99_us"].as_f64());
    }
    let bench_rows = rows_of(&std::fs::read_to_string(&out).unwrap());
    let strip = |r: &Value| {
        let mut r = r.clone();
        r.as_object_mut().unwrap().remove("timings_us");
        r
    };
    assert_eq!(
        bench_rows.iter().map(strip).collect::<Vec<_>>(),
        rows.iter().map(strip).collect::<Vec<_>>()
    );
    // The output is never replaced.
    let e = fails(&[
        "decide",
        s(&t.path),
        "--input",
        s(dev),
        "--skill",
        "topics",
        "--out",
        s(&out),
    ]);
    assert!(e.contains("overwrite"), "{e}");

    // One text through the decisions service equals its batch row.
    for (i, (text, _)) in t.topics.dev_rows.iter().take(4).enumerate() {
        let v = decide_json(&t.path, text, &["--skill", "topics"]);
        let r = &rows[i];
        let a = &v["answers"]["task"];
        let q = &v["cmf"]["questions"]["task"];
        assert_eq!(a["choice"], r["choice"]);
        assert_eq!(
            a["probabilities"][r["choice"].as_str().unwrap()],
            r["p_top"]
        );
        assert_eq!(q["gate"]["p_top"], r["p_top"]);
        assert_eq!(q["gate"]["novelty"], r["novelty"]);
        assert_eq!(q["gate"]["margin"], r["margin"]);
        assert_eq!(q["gate"]["accepted"], r["accepted"]);
        assert_eq!(q["certified"], r["certified"]);
        assert_eq!(q["errors"], r["errors_top5"]);
    }
}

#[test]
fn info_and_verify_describe_and_check_the_file() {
    let t = toy();
    let h = ok(&["decision", "info", s(&t.path)]);
    assert!(h.contains("topics: 4 labels (4 active"), "{h}");
    assert!(h.contains("shop: 3 labels"), "{h}");
    assert!(h.contains("labels: Weather, billing, cards, travel"), "{h}");
    let v = json_of(&ok(&["decision", "verify", s(&t.path), "--json"]));
    assert_eq!(v["ok"], true);
    assert_eq!(v["generation"], 0);
    assert_eq!(v["skills"], 2);
    assert_eq!(v["golden"]["rows"], v["golden"]["bit_exact_rows"]);
    assert!(
        ok(&["decision", "verify", s(&t.path)])
            .trim_end()
            .ends_with("OK")
    );

    // A flipped byte in a tensor fails the full verification.
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.cmf");
    let mut bytes = std::fs::read(&t.path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x40;
    std::fs::write(&bad, &bytes).unwrap();
    fails(&["decision", "verify", s(&bad)]);
}

#[test]
fn run_and_chat_print_the_decision_guard_and_generic_tools_still_work() {
    let t = toy();
    let one_shot = fails(&["run", s(&t.path), "-p", "hello"]);
    assert!(one_shot.contains(GUARD), "{one_shot}");
    // No --prompt: the interactive chat (stdin closed) is refused the same way.
    let chat = fails(&["run", s(&t.path)]);
    assert!(chat.contains(GUARD), "{chat}");
    // `cortiq info` and `cortiq verify` keep working on a decision file.
    let info = ok(&["info", s(&t.path)]);
    assert!(info.contains("cortiq-decision-ph-v1"), "{info}");
    assert!(ok(&["verify", s(&t.path)]).contains("OK"));
}

// ------------------------------------------------------------------ serve

#[test]
fn serve_refuses_language_model_flags_on_a_decision_file() {
    let t = toy();
    for flags in [
        &["--task", "general"][..],
        &["--gpus", "2"],
        &["--o1", "all"],
        &["--o1-m", "8"],
        &["--peer", "127.0.0.1:9"],
        &["--peer-split", "3"],
        &["--compat-port", "11434"],
    ] {
        let mut a = vec!["serve", s(&t.path), "--port", "9"];
        a.extend_from_slice(flags);
        let e = fails(&a);
        assert!(
            e.contains(flags[0]) && e.contains("is a decision file"),
            "{flags:?}: {e}"
        );
    }
}

#[test]
fn serve_on_loopback_learns_then_state_commands_use_its_generations() {
    let t = toy();
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let mock = MockOracle::answering("travel");
    // Open mode with the oracle and learning: `auth.require` false explicitly
    // (the loopback address alone gives a caller without a key neither).
    let cfg = oracle_config(
        d,
        &mock,
        json!({"learning": {"synchronous": true}, "auth": {"require": false}}),
    );
    let state = d.join("state");
    let srv = Server::start(
        &t.path,
        &["--decision-config", s(&cfg), "--state", s(&state)],
        &[(KEY_ENV, TEST_KEY)],
        d,
    );
    // Loopback by default (no --host), port as given.
    let out = std::fs::read_to_string(&srv.stdout).unwrap();
    assert!(
        out.contains(&format!("http://127.0.0.1:{}", srv.port)),
        "{out}"
    );

    let (code, h) = http("GET", &srv.url("/healthz"), None, None);
    assert_eq!(code, 200, "{h}");
    assert_eq!(
        (h["status"].as_str(), h["generation"].as_u64()),
        (Some("ok"), Some(0))
    );
    // A decision (open mode: loopback and no key).
    let (text, _) = &t.topics.dev_rows[1];
    let (code, v) = http(
        "POST",
        &srv.url("/v1/decisions"),
        None,
        Some(&topics_request(text)),
    );
    assert_eq!(code, 200, "{v}");
    assert!(TOPICS.contains(&v["answers"]["task"]["choice"].as_str().unwrap()));
    let base_model = v["model"].as_str().unwrap().to_string();

    // 25 oracle answers of a label promote a challenger (generation 1).
    let lesson = distinct_texts(25, 7, "q", 0.97);
    let fresh = distinct_texts(8, 99, "z", 0.995);
    for q in &lesson {
        let (code, v) = http(
            "POST",
            &srv.url("/v1/decisions"),
            None,
            Some(&topics_request(q)),
        );
        assert_eq!(code, 200, "{v}");
        assert_eq!(v["cmf"]["questions"]["task"]["action"], "oracle", "{v}");
    }
    assert_eq!(mock.hits(), 25);
    let (_, v) = http(
        "POST",
        &srv.url("/v1/decisions"),
        None,
        Some(&topics_request(&fresh[0])),
    );
    assert_eq!(v["cmf"]["generation"], 1, "{v}");
    assert_eq!(v["cmf"]["questions"]["task"]["action"], "local", "{v}");
    assert_eq!(v["answers"]["task"]["choice"], "travel");
    assert_eq!(mock.hits(), 25);
    let logs = srv.stop();
    assert!(!logs.contains(TEST_KEY));
    assert!(
        !std::fs::read_to_string(state.join("oracle.jsonl"))
            .unwrap()
            .contains(TEST_KEY)
    );

    // decide --state: the generation CURRENT names; without it, the base.
    let fresh_text = fresh[1].as_str();
    let served = decide_json(
        &t.path,
        fresh_text,
        &["--skill", "topics", "--state", s(&state)],
    );
    assert_eq!(served["cmf"]["generation"], 1);
    assert_eq!(served["cmf"]["questions"]["task"]["action"], "local");
    assert_eq!(served["answers"]["task"]["choice"], "travel");
    let base = decide_json(&t.path, fresh_text, &["--skill", "topics"]);
    assert_eq!(base["cmf"]["generation"], 0);
    assert_eq!(base["model"].as_str(), Some(base_model.as_str()));
    assert_eq!(base["cmf"]["questions"]["task"]["action"], "abstain");
    let e = fails(&[
        "decide",
        s(&t.path),
        "-p",
        fresh_text,
        "--state",
        s(&d.join("nostate")),
    ]);
    assert!(e.contains("does not exist"), "{e}");

    // verify --state checks the overlay too.
    let v = json_of(&ok(&[
        "decision",
        "verify",
        s(&t.path),
        "--state",
        s(&state),
        "--json",
    ]));
    assert_eq!(
        (v["ok"].as_bool(), v["generation"].as_u64()),
        (Some(true), Some(1))
    );

    // materialize: one self-contained file of the served model.
    let mat = d.join("materialized.cmf");
    let m = ok(&[
        "decision",
        "materialize",
        s(&t.path),
        "--state",
        s(&state),
        "-o",
        s(&mat),
    ]);
    assert!(m.contains("generation 1"), "{m}");
    let v = decide_json(&mat, fresh_text, &["--skill", "topics"]);
    assert_eq!(v["cmf"]["questions"]["task"]["action"], "local");
    assert_eq!(v["answers"]["task"]["choice"], "travel");
    assert_eq!(v["answers"]["task"], served["answers"]["task"]);
    ok(&["decision", "verify", s(&mat)]);
    let e = fails(&[
        "decision",
        "materialize",
        s(&t.path),
        "--state",
        s(&state),
        "-o",
        s(&mat),
    ]);
    assert!(e.contains("overwrite"), "{e}");

    // rollback: 0 serves the base, 1 the generation again; unknown refused.
    let r = ok(&[
        "decision",
        "rollback",
        "--state",
        s(&state),
        "--to",
        "0",
        "--model",
        s(&t.path),
    ]);
    assert!(r.starts_with("CURRENT = g000000"), "{r}");
    assert!(r.contains("g000001"), "{r}");
    let v = decide_json(
        &t.path,
        fresh_text,
        &["--skill", "topics", "--state", s(&state)],
    );
    assert_eq!(v["cmf"]["generation"], 0);
    assert_eq!(v["cmf"]["questions"]["task"]["action"], "abstain");
    let r = ok(&["decision", "rollback", "--state", s(&state), "--to", "1"]);
    assert!(r.contains("g000001") && r.contains("<- CURRENT"), "{r}");
    let v = decide_json(
        &t.path,
        fresh_text,
        &["--skill", "topics", "--state", s(&state)],
    );
    assert_eq!(v["cmf"]["generation"], 1);
    let e = fails(&["decision", "rollback", "--state", s(&state), "--to", "7"]);
    assert!(e.contains("does not exist"), "{e}");
}

#[test]
fn keys_create_list_revoke_import_and_a_keyed_server() {
    let t = toy();
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let state = d.join("state");
    let st = s(&state);
    let created = json_of(&ok(&[
        "decision",
        "keys",
        "create",
        "--state",
        st,
        "--account",
        "acme",
        "--plan",
        "developer",
        "--label",
        "ci",
        "--json",
    ]));
    let acme = created["key"].as_str().unwrap().to_string();
    assert!(
        acme.starts_with("cortiq_") && acme.len() == "cortiq_".len() + 40,
        "{acme}"
    );
    assert_eq!(
        (
            created["account"].as_str(),
            created["plan"].as_str(),
            created["rate_per_min"].as_u64()
        ),
        (Some("acme"), Some("developer"), Some(120))
    );
    // Human form: the key alone on stdout.
    let beta = ok(&[
        "decision",
        "keys",
        "create",
        "--state",
        st,
        "--account",
        "beta",
    ])
    .trim()
    .to_string();
    assert!(beta.starts_with("cortiq_"), "{beta}");
    // A cortiq-router MySQL export: existing keys keep working.
    let legacy = "cortiq_legacy_router_key_0123456789";
    let export = write(
        d,
        "api_keys.json",
        &json!([{"key_hash": format!("{:x}", Sha256::digest(legacy.as_bytes())), "account": "legacy",
                 "plan": "pro", "rate_per_min": "600", "decision_quota": "1000000", "active": 1}])
        .to_string(),
    );
    let r = ok(&[
        "decision",
        "keys",
        "import",
        "--state",
        st,
        "--from",
        s(&export),
    ]);
    assert!(r.contains("1 read; 1 imported (1 active"), "{r}");
    let r = ok(&[
        "decision",
        "keys",
        "import",
        "--state",
        st,
        "--from",
        s(&export),
    ]);
    assert!(
        r.contains("1 read; 0 imported") && r.contains("1 unchanged"),
        "{r}"
    );

    let list = ok(&["decision", "keys", "list", "--state", st, "--json"]);
    let v = json_of(&list);
    let accounts: Vec<&str> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["account"].as_str().unwrap())
        .collect();
    assert_eq!(accounts, ["acme", "beta", "legacy"]);
    let on_disk = std::fs::read_to_string(state.join("keys.json")).unwrap();
    for raw in [&acme, &beta] {
        assert!(!list.contains(raw.as_str()) && !on_disk.contains(raw.as_str()));
        assert!(on_disk.contains(&format!("{:x}", Sha256::digest(raw.as_bytes()))));
    }

    // A server on this state requires a key.
    let srv = Server::start(&t.path, &["--state", st], &[], d);
    let body = topics_request(&t.topics.dev_rows[0].0);
    let (code, v) = http("POST", &srv.url("/v1/decisions"), None, Some(&body));
    assert_eq!(code, 401, "{v}");
    assert_eq!(v["error"]["metadata"]["reason"], "UNAUTHORIZED");
    for key in [acme.as_str(), legacy] {
        let (code, v) = http("POST", &srv.url("/v1/decisions"), Some(key), Some(&body));
        assert_eq!(code, 200, "{v}");
    }
    let (code, _) = http("GET", &srv.url("/healthz"), None, None);
    assert_eq!(code, 200);
    srv.stop();

    // Revoke by account and by hash prefix; nothing left to revoke is an error.
    let r = ok(&[
        "decision",
        "keys",
        "revoke",
        "--state",
        st,
        "--account",
        "acme",
    ]);
    assert!(r.contains("revoked 1 key(s) of account acme"), "{r}");
    let e = fails(&[
        "decision",
        "keys",
        "revoke",
        "--state",
        st,
        "--account",
        "acme",
    ]);
    assert!(e.contains("no active key"), "{e}");
    let beta12 = &format!("{:x}", Sha256::digest(beta.as_bytes()))[..12];
    ok(&[
        "decision", "keys", "revoke", "--state", st, "--hash", beta12,
    ]);
    let v = json_of(&ok(&["decision", "keys", "list", "--state", st, "--json"]));
    let active: Vec<bool> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k["active"].as_bool().unwrap())
        .collect();
    assert_eq!(active, [false, false, true]);
    let table = ok(&["decision", "keys", "list", "--state", st]);
    assert!(
        table.contains("revoked") && table.contains("3 key(s)"),
        "{table}"
    );
}

/// A request the mock received: (head as sent, body).
type SeenRequest = (String, Vec<u8>);

/// A mock of the old router (`cortiq-router`): every request answered with
/// `body` (and `X-Old-Router: yes`), requests recorded as they came; one whose
/// head starts with `slow.0` is answered only after `slow.1`.
fn old_router_mock(
    body: Vec<u8>,
    slow: Option<(&'static str, Duration)>,
) -> (SocketAddr, Arc<Mutex<Vec<SeenRequest>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let Some((head, b)) = read_request_parts(&mut s) else {
                continue;
            };
            if let Some((prefix, wait)) = slow
                && head.starts_with(prefix)
            {
                std::thread::sleep(wait);
            }
            seen2.lock().unwrap().push((head, b));
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-Old-Router: yes\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = s.write_all(head.as_bytes());
            let _ = s.write_all(&body);
        }
    });
    (addr, seen)
}

#[test]
fn serve_shadow_of_answers_with_the_old_router_and_compares_locally() {
    let t = toy();
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    // Pretty-printed: bytes this server never writes.
    let old_body = serde_json::to_vec_pretty(&json!({
        "schema_version": "1.1", "request_id": "req_old_cli",
        "decision": {"task_label": "billing", "taxonomy_id": "topics", "confident": true}
    }))
    .unwrap();
    let (old_addr, seen) = old_router_mock(old_body.clone(), None);
    let old_url = format!("http://{old_addr}");

    // A plain-http router elsewhere than loopback is refused up front.
    let bad_state = d.join("bad-state");
    let e = fails(&[
        "serve",
        s(&t.path),
        "--port",
        "9",
        "--state",
        s(&bad_state),
        "--shadow-of",
        "http://router.example.com",
    ]);
    assert!(e.contains("loopback"), "{e}");
    assert!(!bad_state.exists());
    // Credentials in the URL: refused, and echoed nowhere.
    let o = output(
        &[
            "serve",
            s(&t.path),
            "--port",
            "9",
            "--state",
            s(&bad_state),
            "--shadow-of",
            "https://user:secretpw@router.example.com",
        ],
        &[],
    );
    assert!(!o.status.success());
    assert!(!show(&o).contains("secretpw"), "{}", show(&o));
    assert!(show(&o).contains("credentials"), "{}", show(&o));
    assert!(!bad_state.exists());

    let admin = "admin-token-for-the-cli-shadow-test-0123";
    let state = d.join("state");
    let srv = Server::start(
        &t.path,
        &["--state", s(&state), "--shadow-of", &old_url],
        &[("CORTIQ_DECISION_ADMIN_TOKEN", admin)],
        d,
    );
    let (text, _) = &t.topics.dev_rows[0];
    let req = json!({"taxonomy_id": "topics", "input": {"text": text}}).to_string();
    let agent = ureq::AgentBuilder::new()
        .max_idle_connections(0)
        .timeout(Duration::from_secs(60))
        .build();
    let resp = agent
        .post(&srv.url("/v1/route"))
        .set("Content-Type", "application/json")
        .set(
            "Authorization",
            "Bearer cortiq_00112233445566778899aabbccddeeff00112233",
        )
        .send_string(&req)
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.header("x-old-router"), Some("yes"));
    assert!(resp.header("x-request-id").is_none());
    let mut got = Vec::new();
    resp.into_reader().read_to_end(&mut got).unwrap();
    assert_eq!(got, old_body, "the old router's bytes");
    let bodies: Vec<Vec<u8>> = seen
        .lock()
        .unwrap()
        .iter()
        .map(|(_, b)| b.clone())
        .collect();
    assert_eq!(bodies, [req.clone().into_bytes()]);
    let t0 = Instant::now();
    let stats = loop {
        let v: Value = agent
            .get(&srv.url("/v1/admin/shadow"))
            .set("x-admin-token", admin)
            .call()
            .unwrap()
            .into_json()
            .unwrap();
        if v["lines"] == 1 {
            break v;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "{v}");
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(stats["shadow_of"], json!(old_url));
    assert_eq!(stats["compared"], 1);
    let logs = srv.stop();
    assert!(logs.contains("Shadow mode:"), "{logs}");
    assert!(!logs.contains(text.as_str()), "a text in the logs");
    let raw = std::fs::read_to_string(state.join("shadow.jsonl")).unwrap();
    assert!(!raw.contains(text.as_str()), "a text in the comparison log");
    let line: Value = serde_json::from_str(raw.trim()).unwrap();
    // The digest is keyed (HMAC-SHA256 under <state>/shadow.key), not the
    // text's plain SHA-256.
    let key = cortiq_decision::shadow::read_key(&state.join("shadow.key")).unwrap();
    assert_eq!(
        line["text_hmac"],
        json!(cortiq_decision::shadow::text_hmac(&key, text))
    );
    assert_ne!(
        line["text_hmac"],
        json!(format!("{:x}", Sha256::digest(text.as_bytes())))
    );
    assert_eq!(line["request_id_old"], "req_old_cli");
    assert_eq!(line["taxonomy"], "topics");
    assert_eq!(line["old_label"], "billing");
    assert_eq!(line["old_confident"], true);
    assert_eq!(line["old_status"], 200);
    // The local side is the file's own decision (`cortiq decide`).
    let dec = decide_json(&t.path, text, &["--skill", "topics"]);
    let choice = dec["answers"]["task"]["choice"].as_str().unwrap();
    assert_eq!(line["new_label"], choice);
    assert_eq!(
        line["new_confident"],
        json!(dec["cmf"]["questions"]["task"]["action"] == "local")
    );
    assert_eq!(line["agree"], json!(choice == "billing"));
}

/// F1 item 4: with every debug target on (`RUST_LOG=debug,ureq=trace`), the
/// client secrets a shadow server forwards to the old router — `Authorization`,
/// `x-api-key`, `x-admin-token` — reach the old router and no log line; the
/// deadline of a forwarded request comes from `--shadow-timeout-s`.
#[test]
fn serve_shadow_of_logs_no_forwarded_secret_at_debug_and_takes_its_deadline() {
    let t = toy();
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let old_body = br#"{"schema_version":"1.1","request_id":"req_old_dbg","decision":{"task_label":"billing","taxonomy_id":"topics","confident":true}}"#.to_vec();
    // /v1/readyz is answered only after 8 s: past a 2 s deadline.
    let (old_addr, seen) = old_router_mock(
        old_body.clone(),
        Some(("GET /v1/readyz", Duration::from_secs(8))),
    );
    let bearer = "cortiq_5ec2e75ec2e75ec2e75ec2e75ec2e75ec2e75ec2";
    let xkey = "cortiq_a11ceb0ba11ceb0ba11ceb0ba11ceb0ba11ceb0b";
    let fwd_admin = "forwarded-admin-token-DEBUG-SECRET-9f3e";
    let own_admin = "own-admin-token-for-the-debug-test-5d1c";
    let state = d.join("state");
    let srv = Server::start(
        &t.path,
        &[
            "--state",
            s(&state),
            "--shadow-of",
            &format!("http://{old_addr}"),
            "--shadow-timeout-s",
            "2",
        ],
        &[
            ("RUST_LOG", "debug,ureq=trace"),
            ("CORTIQ_DECISION_ADMIN_TOKEN", own_admin),
        ],
        d,
    );
    let agent = ureq::AgentBuilder::new()
        .max_idle_connections(0)
        .timeout(Duration::from_secs(60))
        .build();
    let (text, _) = &t.topics.dev_rows[0];
    let route = json!({"taxonomy_id": "topics", "input": {"text": text}}).to_string();
    let r = agent
        .post(&srv.url("/v1/route"))
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {bearer}"))
        .set("x-api-key", xkey)
        .send_string(&route)
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = agent
        .get(&srv.url("/v1/taxonomies"))
        .set("x-api-key", xkey)
        .call()
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = agent
        .post(&srv.url("/v1/admin/keys"))
        .set("Content-Type", "application/json")
        .set("x-admin-token", fwd_admin)
        .send_string("{}")
        .unwrap();
    assert_eq!(r.status(), 200);
    // The deadline: the slow old answer gives 502 after about 2 s.
    let t0 = Instant::now();
    let slow = match agent.get(&srv.url("/v1/readyz")).call() {
        Err(ureq::Error::Status(code, r)) => (code, r.into_json::<Value>().unwrap()),
        other => panic!("expected 502, got {other:?}"),
    };
    let took = t0.elapsed();
    assert_eq!(slow.0, 502, "{}", slow.1);
    assert_eq!(slow.1["error"]["code"], "UPSTREAM_UNAVAILABLE");
    assert!(
        took >= Duration::from_millis(1500) && took < Duration::from_secs(7),
        "a 2 s deadline took {took:?}"
    );
    // Wait for the comparison line of the route (the log is written after
    // both sides are done).
    let t0 = Instant::now();
    loop {
        let v: Value = agent
            .get(&srv.url("/v1/admin/shadow"))
            .set("x-admin-token", own_admin)
            .call()
            .unwrap()
            .into_json()
            .unwrap();
        if v["lines"] == 1 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "{v}");
        std::thread::sleep(Duration::from_millis(20));
    }
    // The secrets did go out to the old router (so a header-printing log
    // line would have shown them) ...
    let heads: String = seen
        .lock()
        .unwrap()
        .iter()
        .map(|(h, _)| h.clone())
        .collect::<Vec<_>>()
        .join("\n");
    for secret in [bearer, xkey, fwd_admin] {
        assert!(heads.contains(secret), "{secret} not forwarded:\n{heads}");
    }
    let logs = srv.stop();
    // ... debug logging was on ...
    assert!(
        logs.contains("DEBUG") && logs.contains("shadow mode: deadline of a forwarded request"),
        "{logs}"
    );
    assert!(logs.contains("deadline 2 s per request"), "{logs}");
    // ... and no byte of a secret (whole or a 12-character piece) is in stdout
    // or stderr, nor the header names' values of ureq's request prelude.
    for secret in [bearer, xkey, fwd_admin, own_admin] {
        for i in 0..=secret.len() - 12 {
            let piece = &secret[i..i + 12];
            assert!(
                !logs.contains(piece),
                "'{piece}' of a secret in the logs:\n{logs}"
            );
        }
    }
    let lower = logs.to_ascii_lowercase();
    assert!(!lower.contains("writing prelude"), "{logs}");
    for name in ["authorization:", "x-api-key:", "x-admin-token:"] {
        assert!(!lower.contains(name), "{name} in the logs:\n{logs}");
    }
}

/// Wait for the start of a minute when fewer than 15 s of this one are left
/// (the rate window is a fixed minute).
fn fresh_minute() {
    let now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };
    let left = 60 - now() % 60;
    if left < 15 {
        std::thread::sleep(Duration::from_secs(left + 1));
    }
}

fn sha256_dir(dir: &Path) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().to_string_lossy().into_owned(),
                sha256_file(&e.path()),
            )
        })
        .collect();
    out.sort();
    out
}

/// F1 items 1 and 2: imported router keys may escalate to the oracle unless
/// `--oracle-allowed=false`; an explicit value also reaches keys imported
/// before, a plain re-import leaves them; accounts outside [A-Za-z0-9_.@-]
/// import and are shown escaped in the summary.
#[test]
fn keys_import_oracle_allowed_flag_and_router_accounts() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let st = d.join("state");
    let sha = |k: &str| format!("{:x}", Sha256::digest(k.as_bytes()));
    let export = write(
        d,
        "api_keys.jsonl",
        &[
            json!({"key_hash": sha("cortiq_f1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                   "account": "client+tag@example.com", "plan": "pro"}),
            json!({"key_hash": sha("cortiq_f1bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
                   "account": "Иван\tПетров", "label": "метка".repeat(51)}),
        ]
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n"),
    );
    let import = |extra: &[&str]| {
        let mut a = vec![
            "decision",
            "keys",
            "import",
            "--state",
            s(&st),
            "--from",
            s(&export),
        ];
        a.extend_from_slice(extra);
        ok(&a)
    };
    let oracle = || -> Vec<bool> {
        let v: Value =
            serde_json::from_str(&std::fs::read_to_string(st.join("keys.json")).unwrap()).unwrap();
        v["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k["oracle_allowed"].as_bool().unwrap())
            .collect()
    };
    let r = import(&["--oracle-allowed=false"]);
    assert!(r.contains("2 read; 2 imported (2 active"), "{r}");
    assert!(
        r.contains("accounts of the new keys: client+tag@example.com, Иван\\tПетров"),
        "{r}"
    );
    assert!(r.contains("oracle escalation not allowed"), "{r}");
    assert_eq!(oracle(), [false, false]);
    // A plain re-import does not flip them back.
    let r = import(&[]);
    assert!(
        r.contains("2 unchanged") && r.contains("keys.json unchanged"),
        "{r}"
    );
    assert_eq!(oracle(), [false, false]);
    // An explicit --oracle-allowed (true) reaches the keys imported before.
    let r = import(&["--oracle-allowed"]);
    assert!(
        r.contains("2 keys imported before: oracle escalation allowed"),
        "{r}"
    );
    assert_eq!(oracle(), [true, true]);
    let j = json_of(&import(&["--oracle-allowed=true", "--json"]));
    assert_eq!(
        (&j["keys"]["oracle_allowed"], &j["keys"]["oracle_updated"]),
        (&json!(true), &json!(0))
    );
    // A fresh state: the router's default, allowed.
    let st2 = d.join("state2");
    let r = ok(&[
        "decision",
        "keys",
        "import",
        "--state",
        s(&st2),
        "--from",
        s(&export),
    ]);
    assert!(
        r.contains("new keys: oracle escalation allowed, as in cortiq-router"),
        "{r}"
    );
    // The flag needs --from and a boolean.
    fails(&[
        "decision",
        "keys",
        "import",
        "--state",
        s(&st2),
        "--from",
        s(&export),
        "--oracle-allowed=maybe",
    ]);
    fails(&[
        "decision",
        "keys",
        "import",
        "--state",
        s(&st2),
        "--usage",
        s(&export),
        "--oracle-allowed=false",
    ]);
    // `keys list` shows the key as it is; control characters escaped.
    let list = ok(&["decision", "keys", "list", "--state", s(&st2)]);
    assert!(
        list.contains("Иван\\tПетров") && !list.contains("Иван\tПетров"),
        "{list}"
    );
}

/// Spec decision-v4 §4.15, package C2: keys of the production router (a
/// synthetic MySQL `api_keys` export and a synthetic router configuration,
/// no real data) are imported by the binary and authenticate with their raw
/// keys on the router API and the decisions API; 0 limits are unlimited;
/// inactive and expired keys are refused; the router's usage counters
/// continue; a second import changes nothing; no key and no hash is printed.
#[test]
fn keys_import_moves_router_keys_over_without_reissue() {
    let t = toy();
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let state = d.join("state");
    let st = s(&state);
    // Synthetic router keys (cortiq_ + 40 hex, as the router mints them).
    let mysql_key = "cortiq_c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2";
    let inactive_key = "cortiq_1111111111111111111111111111111111111111";
    let expired_key = "cortiq_2222222222222222222222222222222222222222";
    let limited_key = "cortiq_3333333333333333333333333333333333333333";
    let quota_key = "cortiq_4444444444444444444444444444444444444444";
    let toml_key = "cortiq_5555555555555555555555555555555555555555";
    let bad_key = "cortiq_6666666666666666666666666666666666666666";
    let all = [
        mysql_key,
        inactive_key,
        expired_key,
        limited_key,
        quota_key,
        toml_key,
        bad_key,
    ];
    let sha = |k: &str| format!("{:x}", Sha256::digest(k.as_bytes()));
    let mut secrets: Vec<String> = Vec::new();
    for k in all {
        secrets.push(k.to_string());
        secrets.push(sha(k));
        secrets.push(sha(k).to_ascii_uppercase());
        secrets.push(sha(k)[..12].to_string());
    }
    let no_secret = |what: &str, text: &str| {
        for x in &secrets {
            assert!(
                !text.contains(x.as_str()),
                "{what} shows key material:\n{text}"
            );
        }
    };
    // `mysql -e "SELECT JSON_ARRAYAGG(JSON_OBJECT(...)) FROM api_keys"` shape,
    // numbers as numbers and as strings (phpMyAdmin), store.rs:119-130 columns.
    let export = write(
        d,
        "api_keys.json",
        &json!([
            {"key_hash": sha(mysql_key), "account": "acct_mysql01", "plan": "pro",
             "email": "c2@example.com", "label": "prod", "active": 1, "rate_per_min": 0,
             "decision_quota": 0, "expires_at": null, "created_at": 1_780_000_000},
            {"key_hash": sha(inactive_key), "account": "acct_revoked", "plan": "starter",
             "email": "", "label": "", "active": "0", "rate_per_min": "60",
             "decision_quota": "0", "expires_at": null, "created_at": "1780000000"},
            {"key_hash": sha(expired_key), "account": "acct_expired", "plan": "starter",
             "active": 1, "rate_per_min": 60, "decision_quota": 0, "expires_at": 1_000_000,
             "created_at": 900_000},
            {"key_hash": sha(limited_key), "account": "acct_limited", "plan": "developer",
             "active": 1, "rate_per_min": 2, "decision_quota": 0, "expires_at": null,
             "created_at": 1_780_000_000},
            {"key_hash": sha(quota_key), "account": "acct_quota", "plan": "developer",
             "active": 1, "rate_per_min": 0, "decision_quota": 5, "expires_at": null,
             "created_at": 1_780_000_000},
        ])
        .to_string(),
    );
    let config = write(
        d,
        "router.toml",
        &format!(
            r#"# cortiq-router configuration (synthetic)
bind = "0.0.0.0:8080"
taxonomy_id = "topics"
database_url = ""
complexity_tiers = [ {{ tier = "low", max = 0.33 }}, {{ tier = "high", max = 1.0 }} ]

[[api_keys]]
key            = "{toml_key}"
account        = "acct_toml"
rate_per_min   = 0          # unlimited
decision_quota = 0          # unlimited

[auth]
require = true
[auth.plans.pro]
rate_per_min = 600
decision_quota = 1_000_000
duration_days = 30
"#
        ),
    );
    let usage = write(
        d,
        "usage_counters.json",
        &json!([
            {"account": "acct_quota", "decisions": 5, "oracle_calls": 2},
            {"account": "acct_mysql01", "decisions": "1000000", "oracle_calls": "10"},
        ])
        .to_string(),
    );
    let run = |args: &[&str]| {
        let o = output(args, &[]);
        let text = show(&o);
        no_secret(&format!("cortiq {args:?}"), &text);
        assert!(o.status.success(), "cortiq {args:?} failed\n{text}");
        String::from_utf8(o.stdout).unwrap()
    };
    let import = |extra: &[&str]| {
        let mut a = vec!["decision", "keys", "import", "--state", st];
        a.extend_from_slice(extra);
        run(&a)
    };

    // MySQL export (format by the .json name), then the configuration with usage.
    let r = import(&["--from", s(&export)]);
    assert!(
        r.contains(
            "(mysql-json): 5 read; 5 imported (3 active, 1 inactive, 1 expired), 0 unchanged"
        ),
        "{r}"
    );
    assert!(
        r.contains("accounts of the new keys: acct_expired, acct_limited, acct_mysql01, acct_quota, acct_revoked"),
        "{r}"
    );
    assert!(
        r.contains("1 emails not stored") && r.contains("written"),
        "{r}"
    );
    assert!(
        r.contains("new keys: oracle escalation allowed, as in cortiq-router"),
        "{r}"
    );
    let r = import(&[
        "--from",
        s(&config),
        "--format",
        "router-toml",
        "--usage",
        s(&usage),
    ]);
    assert!(
        r.contains("(router-toml): 1 read; 1 imported (1 active, 0 inactive, 0 expired)"),
        "{r}"
    );
    assert!(
        r.contains("2 read; 2 carried over (+1000005 decisions, +12 oracle calls)"),
        "{r}"
    );

    // keys.json: hashes only, no raw key, no email.
    let keys_json = state.join("keys.json");
    let on_disk = std::fs::read_to_string(&keys_json).unwrap();
    for k in all {
        assert!(!on_disk.contains(k), "raw key in keys.json");
    }
    assert!(!on_disk.contains("example.com") && !on_disk.contains("email"));
    let v: Value = serde_json::from_str(&on_disk).unwrap();
    let stored: Vec<(String, String, bool, u64, u64, String)> = v["keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| {
            (
                k["hash"].as_str().unwrap().to_string(),
                k["account"].as_str().unwrap().to_string(),
                k["active"].as_bool().unwrap(),
                k["rate_per_min"].as_u64().unwrap(),
                k["decision_quota"].as_u64().unwrap(),
                k["plan"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let want = |k: &str, a: &str, active: bool, rate: u64, quota: u64, plan: &str| {
        (sha(k), a.to_string(), active, rate, quota, plan.to_string())
    };
    assert_eq!(
        stored,
        [
            want(mysql_key, "acct_mysql01", true, 0, 0, "pro"),
            want(inactive_key, "acct_revoked", false, 60, 0, "starter"),
            want(expired_key, "acct_expired", true, 60, 0, "starter"),
            want(limited_key, "acct_limited", true, 2, 0, "developer"),
            want(quota_key, "acct_quota", true, 0, 5, "developer"),
            want(toml_key, "acct_toml", true, 0, 0, "static"),
        ]
    );
    for k in v["keys"].as_array().unwrap() {
        let h = k["hash"].as_str().unwrap();
        assert!(h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
        // Every router key escalated in the router: so it may here.
        assert_eq!(k["oracle_allowed"], true, "{k}");
    }

    // A second import of both changes nothing: same bytes, same ledger.
    let keys_sha = sha256_file(&keys_json);
    let keys_mtime = std::fs::metadata(&keys_json).unwrap().modified().unwrap();
    let usage_files = sha256_dir(&state.join("usage"));
    let r = import(&["--from", s(&export)]);
    assert!(
        r.contains("5 read; 0 imported (0 active, 0 inactive, 0 expired), 5 unchanged, 0 revoked")
            && r.contains("keys.json unchanged"),
        "{r}"
    );
    let r = import(&["--from", s(&config), "--usage", s(&usage), "--json"]);
    let j = json_of(&r);
    assert_eq!(
        (
            &j["keys"]["format"],
            &j["keys"]["imported"],
            &j["keys"]["unchanged"],
            &j["keys"]["written"]
        ),
        (&json!("router-toml"), &json!(0), &json!(1), &json!(false)),
        "{j}"
    );
    assert_eq!(
        (
            &j["usage"]["carried"],
            &j["usage"]["unchanged"],
            &j["usage"]["written"]
        ),
        (&json!(0), &json!(2), &json!(false)),
        "{j}"
    );
    assert_eq!(sha256_file(&keys_json), keys_sha);
    assert_eq!(
        std::fs::metadata(&keys_json).unwrap().modified().unwrap(),
        keys_mtime
    );
    assert_eq!(sha256_dir(&state.join("usage")), usage_files);

    // A malformed hash refuses the whole file, naming the row, not the hash.
    let bad = write(
        d,
        "bad.json",
        &json!([
            {"key_hash": sha(bad_key), "account": "acct_bad"},
            {"key_hash": sha(bad_key).to_ascii_uppercase(), "account": "acct_bad2"},
        ])
        .to_string(),
    );
    let o = output(
        &[
            "decision",
            "keys",
            "import",
            "--state",
            st,
            "--from",
            s(&bad),
        ],
        &[],
    );
    let text = show(&o);
    assert!(!o.status.success(), "{text}");
    assert!(
        text.contains("nothing was imported")
            && text.contains("row 2")
            && text.contains("lowercase hex"),
        "{text}"
    );
    no_secret("a refused import", &text);
    assert_eq!(sha256_file(&keys_json), keys_sha);

    // The keys authenticate with their raw keys on both APIs.
    let srv = Server::start(&t.path, &["--state", st], &[], d);
    // The ledger belongs to the running server: a usage import is refused.
    let o = output(
        &[
            "decision",
            "keys",
            "import",
            "--state",
            st,
            "--usage",
            s(&usage),
        ],
        &[],
    );
    assert!(!o.status.success(), "{}", show(&o));
    assert!(show(&o).contains("stop the server"), "{}", show(&o));
    let text = &t.topics.dev_rows[0].0;
    let route = json!({"input": {"text": text}, "taxonomy_id": "topics"});
    let decisions = topics_request(text);
    let call = |key: &str, path: &str| {
        let body = if path == "/v1/route" {
            &route
        } else {
            &decisions
        };
        http("POST", &srv.url(path), Some(key), Some(body))
    };
    // 0 = unlimited: many requests in one minute, a million decisions used.
    for key in [mysql_key, toml_key] {
        for _ in 0..4 {
            let (code, v) = call(key, "/v1/route");
            assert_eq!(code, 200, "{v}");
            assert_eq!(v["decision"]["taxonomy_id"], "topics", "{v}");
            let (code, v) = call(key, "/api/alpha/decisions");
            assert_eq!(code, 200, "{v}");
        }
    }
    let (code, v) = http("GET", &srv.url("/v1/usage"), Some(mysql_key), None);
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["account"]["id"], "acct_mysql01");
    assert_eq!(v["account"]["billable_decisions"], json!(1_000_008), "{v}");
    assert_eq!(v["account"]["oracle_calls"], json!(10), "{v}");
    assert_eq!(v["account"]["decision_quota"], json!(0), "{v}");
    // A limit that is not 0 is enforced: 2 requests per minute, then 429.
    fresh_minute();
    assert_eq!(call(limited_key, "/v1/route").0, 200);
    assert_eq!(call(limited_key, "/api/alpha/decisions").0, 200);
    let (code, v) = call(limited_key, "/v1/route");
    assert_eq!(code, 429, "{v}");
    assert_eq!(v["error"]["code"], "RATE_LIMITED", "{v}");
    // The router's usage continues: 5 of 5 decisions used, so 402 at once.
    let (code, v) = call(quota_key, "/v1/route");
    assert_eq!(code, 402, "{v}");
    assert_eq!(v["error"]["code"], "QUOTA_EXCEEDED", "{v}");
    assert_eq!(call(quota_key, "/api/alpha/decisions").0, 402);
    // Inactive and expired keys are refused on both APIs.
    for key in [inactive_key, expired_key, bad_key] {
        for path in ["/v1/route", "/api/alpha/decisions"] {
            let (code, v) = call(key, path);
            assert_eq!(code, 401, "{path}: {v}");
            no_secret("a 401 body", &v.to_string());
        }
    }
    let logs = srv.stop();
    no_secret("the server logs", &logs);
    assert_eq!(sha256_file(&keys_json), keys_sha);
}

// ------------------------------------------------------------------ learn

#[test]
fn learn_asks_the_mock_oracle_only_about_abstentions() {
    let t = toy();
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let mock = MockOracle::answering("travel");
    let cfg_path = oracle_config(d, &mock, json!({}));

    // Traffic: 4 dev texts the gate accepts, 12 cruise texts it rejects, one repeat.
    let model = DecisionModel::open(&t.path, Verify::Light).unwrap();
    let ev = Evaluator::new(&model, "topics").unwrap();
    let accepted: Vec<&String> = t
        .topics
        .dev_rows
        .iter()
        .map(|(x, _)| x)
        .filter(|x| ev.scorer().accepted(&ev.decide_text(x).unwrap().decision))
        .take(4)
        .collect();
    assert_eq!(accepted.len(), 4);
    // The first 12 texts of the library's lesson (every one abstains).
    let cruise = distinct_texts(12, 7, "q", 0.97);
    for c in &cruise {
        assert!(
            !ev.scorer().accepted(&ev.decide_text(c).unwrap().decision),
            "{c}"
        );
    }
    let mut lines: Vec<Value> = accepted
        .iter()
        .map(|x| json!({"text": x, "label": "ignored"}))
        .collect();
    lines.extend(cruise.iter().map(|x| json!({"text": x})));
    lines.push(json!({"text": cruise[0]}));
    let traffic = write(
        d,
        "traffic.jsonl",
        &lines
            .iter()
            .map(|l| l.to_string() + "\n")
            .collect::<String>(),
    );

    // A driver ledger answering the first 3 cruise texts (by body sha256).
    let cfg = Config::load(&cfg_path).unwrap();
    let q = learn::rubric_question(&model.skill("topics").unwrap().manifest).unwrap();
    let mut ledger = vec![json!({"record_type": "run"})];
    for c in &cruise[..3] {
        let b = oracle::request_body(&cfg.oracle, &[&q], &json!(c));
        ledger.push(json!({"record_type": "oracle_call",
                           "request_sha256": format!("{:x}", Sha256::digest(&b)),
                           "oracle": {"choice": "travel"}}));
    }
    let answers = write(
        d,
        "answers.jsonl",
        &ledger
            .iter()
            .map(|l| l.to_string() + "\n")
            .collect::<String>(),
    );

    let out = d.join("learned.cmf");
    let base_args = [
        "decision",
        "learn",
        s(&t.path),
        "--traffic",
        s(&traffic),
        "--skill",
        "topics",
        "--oracle-config",
        s(&cfg_path),
        "--answers",
        s(&answers),
        "--threads",
        "2",
    ];
    // Live calls enabled without the key in the environment: refused up front.
    let mut a = base_args.to_vec();
    a.extend(["-o", s(&out)]);
    let e = fails(&a);
    assert!(e.contains(KEY_ENV), "{e}");
    assert!(!out.exists() && !d.join("learned.cmf.oracle.jsonl").exists());
    assert_eq!(mock.hits(), 0);

    a.push("--json");
    let r = json_of(&ok_env(&a, &[(KEY_ENV, TEST_KEY)]));
    assert_eq!(
        (r["texts"].as_u64(), r["labelled"].as_u64()),
        (Some(17), Some(4))
    );
    assert_eq!(
        (r["accepted"].as_u64(), r["abstained"].as_u64()),
        (Some(4), Some(13))
    );
    assert_eq!(r["answers_reused"], 3);
    assert_eq!(r["live_calls"], 9);
    assert_eq!(
        mock.hits(),
        9,
        "only the abstentions without a stored answer"
    );
    let mut asked = mock.states();
    asked.sort_by_key(|v| v.to_string());
    let mut expected: Vec<Value> = cruise[3..].iter().map(|c| json!(c)).collect();
    expected.sort_by_key(|v| v.to_string());
    assert_eq!(asked, expected);
    assert_eq!(
        (r["examples"].as_u64(), r["duplicates"].as_u64()),
        (Some(12), Some(1))
    );
    assert_eq!(r["unanswered"], 0);
    let mut decided: Vec<Value> = r["promoted_labels"].as_array().unwrap().clone();
    decided.extend(r["rejected_labels"].as_array().unwrap().iter().cloned());
    assert_eq!(decided, [json!("travel")]);
    // The reservation ledger (default next to the output): reserved + settled
    // per call, no key.
    let led = std::fs::read_to_string(d.join("learned.cmf.oracle.jsonl")).unwrap();
    assert_eq!(led.lines().count(), 18);
    assert!(!led.contains(TEST_KEY));
    // The output: self-contained, verified, with the learned record.
    ok(&["decision", "verify", s(&out)]);
    let info = info_json(&out);
    let l = &info["skills"][0]["learned"];
    assert_eq!(
        (l["calls"].as_u64(), l["answers_reused"].as_u64()),
        (Some(9), Some(3))
    );
    assert_eq!(l["oracle_model"], ORACLE_MODEL);
    assert_eq!(
        info["skills"][1],
        info_json(&t.path)["skills"][1],
        "shop byte for byte"
    );
    // An existing output is refused.
    let e = fails(&a);
    assert!(e.contains("overwrite"), "{e}");
}
