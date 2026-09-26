//! Shadow mode of the router API (spec §4.15, the switch of production traffic
//! from `cortiq-router`): the comparison log, its agreement statistics and the
//! client of the old router.
//!
//! `cortiq serve FILE --shadow-of URL` answers every router-API request with
//! the answer of the router at `URL` ([`Upstream`]), byte for byte, and for
//! `/v1/route` and `/v1/route:batch` also decides the same inputs locally
//! ([`crate::service::DecisionService::decide_local`]: no oracle, no cache, no
//! learning, no billing). Each routed input becomes one [`ShadowLine`] of
//! `<state>/shadow.jsonl` ([`ShadowLog`]); [`ShadowStats`] are the agreement
//! statistics of the whole file, replayed when the log is opened and kept up to
//! date by every append (`GET /v1/admin/shadow`).
//!
//! A line never holds a text: `text_sha256` is the lowercase hex SHA-256 of
//! the input's UTF-8 bytes; the rest are labels, flags, latencies, the old
//! router's HTTP status and reason codes.

use crate::manifest::sha256_hex;
use anyhow::{Context, Result, bail, ensure};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// File name of the comparison log in the state directory.
pub const SHADOW_LOG_FILE: &str = "shadow.jsonl";
/// Most bytes of a request body forwarded to the old router: its own limit
/// (`cortiq-router` `main.rs:441`, `RequestBodyLimitLayer::new(8 * 1024 * 1024)`).
pub const UPSTREAM_MAX_BODY: usize = 8 * 1024 * 1024;
/// The old router's answer to a body over [`UPSTREAM_MAX_BODY`] (tower-http 0.6
/// `limit/body.rs`: 413, `text/plain; charset=utf-8`).
pub const UPSTREAM_LENGTH_LIMIT: &str = "length limit exceeded";
/// Most bytes of an answer of the old router (a larger one is a failure).
pub const UPSTREAM_MAX_RESPONSE: usize = 64 * 1024 * 1024;
/// Deadline of one forwarded request, connect to last byte (nginx's default
/// `proxy_read_timeout`; the old router's own oracle deadline is 30 s,
/// `cortiq-router` `oracle.rs:133`).
pub const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(60);
/// Request headers passed on to the old router: the ones it reads (API key in
/// `Authorization` or `x-api-key`, `x-admin-token`, the JSON content type),
/// `accept` and `user-agent`. `Accept-Encoding: identity` is always sent, so
/// that the body comes back as the router wrote it.
pub const FORWARDED_REQUEST_HEADERS: [&str; 6] = [
    "authorization",
    "x-api-key",
    "x-admin-token",
    "content-type",
    "accept",
    "user-agent",
];
/// Response headers not passed back: hop-by-hop (RFC 9110 §7.6.1) and the
/// framing the serving side sets itself for the same body.
const NOT_FORWARDED_RESPONSE_HEADERS: [&str; 10] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
];

/// Log targets whose DEBUG and TRACE lines print request headers: ureq's
/// `writing prelude` line (ureq 2 `unit.rs`) shows every header it sends but
/// `Authorization` and `Cookie` — the `x-api-key` and `x-admin-token` a
/// shadow server forwards (and the oracle client's request) in clear.
pub const SECRET_BEARING_LOG_TARGETS: [&str; 1] = ["ureq"];

/// Whether a log line of `target` at `level` may carry a forwarded secret
/// ([`SECRET_BEARING_LOG_TARGETS`] above INFO): a subscriber drops it,
/// whatever `RUST_LOG` enables (`cortiq serve` does).
pub fn log_may_carry_secrets(target: &str, level: &tracing::Level) -> bool {
    *level > tracing::Level::INFO
        && SECRET_BEARING_LOG_TARGETS.iter().any(|t| {
            target == *t
                || target
                    .strip_prefix(t)
                    .is_some_and(|rest| rest.starts_with("::"))
        })
}

/// Lowercase hex SHA-256 of a text (the only trace of a text in the log).
pub fn text_sha256(text: &str) -> String {
    sha256_hex(text.as_bytes())
}

// ------------------------------------------------------------------ log line

/// One routed input compared (one line of `shadow.jsonl`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShadowLine {
    /// Unix seconds when the request arrived.
    pub ts: u64,
    /// `request_id` of the old router's answer: of the result in a batch, of
    /// the error envelope when it failed; null when it gave none.
    pub request_id_old: Option<String>,
    /// SHA-256 hex of `input.text`; null without a text or for a
    /// bring-your-own embedding (whose text the router ignores).
    pub text_sha256: Option<String>,
    /// `decision.taxonomy_id` of the old answer, else the request's
    /// `taxonomy_id`, else the skill decided locally.
    pub taxonomy: Option<String>,
    /// `decision.task_label` of the old answer (its oracle's label when it
    /// escalated).
    pub old_label: Option<String>,
    /// The local winner (`__novel__` without a candidate), never an oracle's.
    pub new_label: Option<String>,
    /// `old_label == new_label` when both are present, else null.
    pub agree: Option<bool>,
    /// `decision.confident` of the old answer.
    pub old_confident: Option<bool>,
    /// The local gate accepted under the request's `policy_profile`: what
    /// `/v1/route` would answer without escalating.
    pub new_confident: Option<bool>,
    /// Round trip of the request to the old router, in ms (a batch's lines
    /// repeat the batch's).
    pub old_latency_ms: Option<f64>,
    /// Local decision time of the request, in ms (a batch's lines repeat the
    /// batch's total); null when nothing was decided.
    pub new_latency_ms: Option<f64>,
    /// HTTP status of the old router's answer; null when it did not answer.
    pub old_status: Option<u16>,
    /// Reason code of a missing local decision (`INVALID_REQUEST`,
    /// `TAXONOMY_NOT_FOUND`, `EMBEDDING_REQUIRED`, `EMBEDDING_INPUT`,
    /// `OVERLOADED`, …), else null.
    pub new_error: Option<String>,
}

/// Milliseconds of a duration, rounded to the microsecond.
pub fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e6).round() / 1e3
}

// ------------------------------------------------------------------ statistics

/// Agreement of one old label (one taxonomy).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabelStats {
    /// Lines with this old label.
    pub lines: u64,
    /// Of them, with a local label.
    pub compared: u64,
    pub agree: u64,
    /// Local labels of the compared lines.
    pub new_labels: BTreeMap<String, u64>,
}

/// Agreement statistics of a set of lines ([`ShadowStats::add`] each).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShadowStats {
    pub lines: u64,
    /// Lines of the file that are not a [`ShadowLine`] (a torn last line).
    pub malformed: u64,
    pub first_ts: Option<u64>,
    pub last_ts: Option<u64>,
    /// Lines with both labels, and those that agree.
    pub compared: u64,
    pub agree: u64,
    /// Lines whose old answer was not a 200 (or never came).
    pub old_errors: u64,
    /// Lines without an old label / without a local label.
    pub old_missing: u64,
    pub new_missing: u64,
    /// Compared lines where the old / the local / both sides were confident,
    /// and of those the ones that agree.
    pub old_confident: u64,
    pub old_confident_agree: u64,
    pub new_confident: u64,
    pub new_confident_agree: u64,
    pub both_confident: u64,
    pub both_confident_agree: u64,
    old_latency_sum: f64,
    old_latency_n: u64,
    new_latency_sum: f64,
    new_latency_n: u64,
    /// By (taxonomy, old label).
    pub labels: BTreeMap<(String, String), LabelStats>,
}

fn ratio(a: u64, b: u64) -> Value {
    if b == 0 {
        Value::Null
    } else {
        json!(a as f64 / b as f64)
    }
}

fn mean(sum: f64, n: u64) -> Value {
    if n == 0 {
        Value::Null
    } else {
        json!((sum / n as f64 * 1e3).round() / 1e3)
    }
}

impl ShadowStats {
    pub fn add(&mut self, l: &ShadowLine) {
        self.lines += 1;
        self.first_ts = Some(self.first_ts.map_or(l.ts, |t| t.min(l.ts)));
        self.last_ts = Some(self.last_ts.map_or(l.ts, |t| t.max(l.ts)));
        if l.old_status != Some(200) {
            self.old_errors += 1;
        }
        if l.old_label.is_none() {
            self.old_missing += 1;
        }
        if l.new_label.is_none() {
            self.new_missing += 1;
        }
        if let Some(v) = l.old_latency_ms {
            self.old_latency_sum += v;
            self.old_latency_n += 1;
        }
        if let Some(v) = l.new_latency_ms {
            self.new_latency_sum += v;
            self.new_latency_n += 1;
        }
        let compared = match (&l.old_label, &l.new_label) {
            (Some(o), Some(n)) => Some((o, n, o == n)),
            _ => None,
        };
        if let Some((_, _, agree)) = compared {
            let a = u64::from(agree);
            self.compared += 1;
            self.agree += a;
            let oc = l.old_confident == Some(true);
            let nc = l.new_confident == Some(true);
            if oc {
                self.old_confident += 1;
                self.old_confident_agree += a;
            }
            if nc {
                self.new_confident += 1;
                self.new_confident_agree += a;
            }
            if oc && nc {
                self.both_confident += 1;
                self.both_confident_agree += a;
            }
        }
        if let Some(old) = &l.old_label {
            let key = (l.taxonomy.clone().unwrap_or_default(), old.clone());
            let s = self.labels.entry(key).or_default();
            s.lines += 1;
            if let Some((_, new, agree)) = compared {
                s.compared += 1;
                s.agree += u64::from(agree);
                *s.new_labels.entry(new.clone()).or_default() += 1;
            }
        }
    }

    /// The statistics as `GET /v1/admin/shadow` shows them; `labels` ordered
    /// by compared lines (most first), then taxonomy and label.
    pub fn to_json(&self) -> Value {
        let conf = |n: u64, a: u64| json!({"compared": n, "agree": a, "agreement": ratio(a, n)});
        let mut labels: Vec<(&(String, String), &LabelStats)> = self.labels.iter().collect();
        labels.sort_by(|a, b| b.1.compared.cmp(&a.1.compared).then(a.0.cmp(b.0)));
        let labels: Vec<Value> = labels
            .into_iter()
            .map(|((tax, label), s)| {
                json!({
                    "taxonomy": tax,
                    "label": label,
                    "lines": s.lines,
                    "compared": s.compared,
                    "agree": s.agree,
                    "agreement": ratio(s.agree, s.compared),
                    "new_labels": s.new_labels,
                })
            })
            .collect();
        json!({
            "lines": self.lines,
            "malformed_lines": self.malformed,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "compared": self.compared,
            "agree": self.agree,
            "agreement": ratio(self.agree, self.compared),
            "old_errors": self.old_errors,
            "old_missing": self.old_missing,
            "new_missing": self.new_missing,
            "confident": {
                "old": conf(self.old_confident, self.old_confident_agree),
                "new": conf(self.new_confident, self.new_confident_agree),
                "both": conf(self.both_confident, self.both_confident_agree),
            },
            "latency_ms": {
                "old_mean": mean(self.old_latency_sum, self.old_latency_n),
                "new_mean": mean(self.new_latency_sum, self.new_latency_n),
            },
            "labels": labels,
        })
    }
}

// ------------------------------------------------------------------ the log

struct LogInner {
    file: File,
    stats: ShadowStats,
}

/// `shadow.jsonl`: append-only (mode 0600), one [`ShadowLine`] per line, and
/// the statistics of every line in it.
pub struct ShadowLog {
    path: PathBuf,
    inner: Mutex<LogInner>,
}

impl std::fmt::Debug for ShadowLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShadowLog")
            .field("path", &self.path)
            .field("lines", &self.inner.lock().stats.lines)
            .finish()
    }
}

fn open_append(path: &Path) -> Result<File> {
    let mut o = OpenOptions::new();
    o.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
        .with_context(|| format!("open {}", path.display()))
}

impl ShadowLog {
    /// Open (create) the log and replay the lines already in it into the
    /// statistics. A last line torn by a crash is counted as malformed and
    /// closed with a newline, so the next line starts clean.
    pub fn open(path: &Path) -> Result<Self> {
        let mut stats = ShadowStats::default();
        let mut torn = false;
        match File::open(path) {
            Ok(f) => {
                let mut r = BufReader::new(f);
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    let n = r
                        .read_until(b'\n', &mut buf)
                        .with_context(|| format!("read {}", path.display()))?;
                    if n == 0 {
                        break;
                    }
                    torn = buf.last() != Some(&b'\n');
                    let line = buf.trim_ascii();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_slice::<ShadowLine>(line) {
                        Ok(l) => stats.add(&l),
                        Err(_) => stats.malformed += 1,
                    }
                }
            }
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        }
        let mut file = open_append(path)?;
        if torn {
            file.write_all(b"\n")
                .with_context(|| format!("write {}", path.display()))?;
        }
        Ok(Self {
            path: path.to_path_buf(),
            inner: Mutex::new(LogInner { file, stats }),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append lines (one write) and add them to the statistics.
    pub fn append(&self, lines: &[ShadowLine]) -> Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::with_capacity(256 * lines.len());
        for l in lines {
            serde_json::to_writer(&mut buf, l).context("serialize a shadow line")?;
            buf.push(b'\n');
        }
        let mut inner = self.inner.lock();
        inner
            .file
            .write_all(&buf)
            .with_context(|| format!("write {}", self.path.display()))?;
        for l in lines {
            inner.stats.add(l);
        }
        Ok(())
    }

    pub fn stats(&self) -> ShadowStats {
        self.inner.lock().stats.clone()
    }

    /// `fsync` (at shutdown).
    pub fn sync(&self) -> Result<()> {
        self.inner
            .lock()
            .file
            .sync_data()
            .with_context(|| format!("sync {}", self.path.display()))
    }
}

// ------------------------------------------------------------------ the old router

/// The base URL of `--shadow-of`: `https://host[:port][/prefix]`, or `http://`
/// to a loopback host only (tests, a router on the same machine). No
/// credentials, query or fragment (the client's own key is forwarded). The
/// request path is appended to it; a trailing `/` is dropped.
pub fn upstream_base(url: &str) -> Result<String> {
    let url = url.trim();
    let Some((scheme, rest)) = url.split_once("://") else {
        bail!("--shadow-of is not a URL (https://host[:port][/prefix])");
    };
    let scheme = scheme.to_ascii_lowercase();
    ensure!(
        scheme == "https" || scheme == "http",
        "--shadow-of: the scheme is not https"
    );
    ensure!(
        !rest.contains(['?', '#']),
        "--shadow-of: the URL takes no query or fragment"
    );
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    ensure!(
        !authority.contains('@'),
        "--shadow-of: credentials in the URL are refused (each client's own Authorization header is forwarded)"
    );
    let host = url_host(authority)?;
    if scheme == "http" {
        ensure!(
            is_loopback_host(host),
            "--shadow-of: plain http only to a loopback address; the old router is reached over https"
        );
    }
    ensure!(
        !path.contains(char::is_whitespace),
        "--shadow-of: the path has whitespace"
    );
    Ok(format!(
        "{scheme}://{authority}{}",
        path.trim_end_matches('/')
    ))
}

/// The host of `host[:port]` / `[v6][:port]`.
fn url_host(authority: &str) -> Result<&str> {
    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        let Some((h, after)) = rest.split_once(']') else {
            bail!("--shadow-of: unclosed [ in the host");
        };
        let port = match after {
            "" => None,
            p => Some(
                p.strip_prefix(':')
                    .ok_or_else(|| anyhow::anyhow!("--shadow-of: bad host"))?,
            ),
        };
        ensure!(
            h.parse::<std::net::Ipv6Addr>().is_ok(),
            "--shadow-of: bad IPv6 host"
        );
        (h, port)
    } else {
        let (h, port) = match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        };
        ensure!(
            !h.is_empty()
                && h.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-._".contains(&b)),
            "--shadow-of: bad host"
        );
        (h, port)
    };
    if let Some(p) = port {
        ensure!(
            p.parse::<u16>().is_ok_and(|p| p > 0),
            "--shadow-of: bad port"
        );
    }
    Ok(host)
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// The old router's answer, as it came (hop-by-hop headers and the length
/// dropped, names lowercase, repeated headers kept in order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpstreamResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// Round trip, request sent to last byte read.
    pub latency: Duration,
}

/// No answer from the old router: a transport failure (refused, DNS, TLS,
/// timeout), an unreadable or oversized body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpstreamError {
    /// `transport_<kind>` / `read_<kind>` / `response_too_large`.
    pub reason: String,
    pub latency: Duration,
}

/// The client of the old router (`--shadow-of`): one ureq agent (rustls,
/// pooled connections, no redirects followed, [`UPSTREAM_TIMEOUT`]).
pub struct Upstream {
    base: String,
    agent: ureq::Agent,
}

impl std::fmt::Debug for Upstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upstream")
            .field("base", &self.base)
            .finish()
    }
}

impl Upstream {
    /// The client of the router at `url` ([`upstream_base`]).
    pub fn new(url: &str, timeout: Duration) -> Result<Self> {
        let base = upstream_base(url)?;
        let agent = ureq::AgentBuilder::new()
            .timeout(timeout)
            .redirects(0)
            .build();
        Ok(Self { base, agent })
    }

    /// The base URL requests go to.
    pub fn base(&self) -> &str {
        &self.base
    }

    /// Send `method path_and_query` with `headers` (the client's, filtered by
    /// the caller to [`FORWARDED_REQUEST_HEADERS`]) and `body` unchanged; any
    /// HTTP status is an answer.
    pub fn forward(
        &self,
        method: &str,
        path_and_query: &str,
        headers: &[(String, String)],
        body: &[u8],
    ) -> std::result::Result<UpstreamResponse, UpstreamError> {
        let url = format!("{}{path_and_query}", self.base);
        let mut req = self.agent.request(method, &url);
        for (k, v) in headers {
            req = req.set(k, v);
        }
        req = req.set("accept-encoding", "identity");
        let t0 = Instant::now();
        let fail = |reason: String| UpstreamError {
            reason,
            latency: t0.elapsed(),
        };
        let bodiless = body.is_empty() && matches!(method, "GET" | "HEAD" | "DELETE" | "OPTIONS");
        let res = if bodiless {
            req.call()
        } else {
            req.send_bytes(body)
        };
        let resp = match res {
            Ok(r) => r,
            Err(ureq::Error::Status(_, r)) => r,
            Err(ureq::Error::Transport(t)) => {
                return Err(fail(
                    format!("transport_{:?}", t.kind()).to_ascii_lowercase(),
                ));
            }
        };
        let status = resp.status();
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for name in resp.headers_names() {
            if NOT_FORWARDED_RESPONSE_HEADERS.contains(&name.as_str()) || !seen.insert(name.clone())
            {
                continue;
            }
            for v in resp.all(&name) {
                out.push((name.clone(), v.to_string()));
            }
        }
        let mut buf = Vec::new();
        resp.into_reader()
            .take(UPSTREAM_MAX_RESPONSE as u64 + 1)
            .read_to_end(&mut buf)
            .map_err(|e| fail(format!("read_{:?}", e.kind()).to_ascii_lowercase()))?;
        if buf.len() > UPSTREAM_MAX_RESPONSE {
            return Err(fail("response_too_large".into()));
        }
        Ok(UpstreamResponse {
            status,
            headers: out,
            body: buf,
            latency: t0.elapsed(),
        })
    }
}

// ------------------------------------------------------------------ the old answer

/// What a line takes from one routed result of the old router.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OldResult {
    pub request_id: Option<String>,
    pub label: Option<String>,
    pub confident: Option<bool>,
    pub taxonomy: Option<String>,
}

/// The parts of an old answer the lines need: the envelope's `request_id`,
/// and for a 200 the result(s): the answer itself for `/v1/route`, its
/// `results` for a batch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OldAnswer {
    pub request_id: Option<String>,
    pub results: Vec<OldResult>,
}

impl OldAnswer {
    pub fn parse(status: u16, body: &[u8], batch: bool) -> Self {
        let Ok(v) = serde_json::from_slice::<Value>(body) else {
            return Self::default();
        };
        let result = |r: &Value| {
            let d = &r["decision"];
            OldResult {
                request_id: r["request_id"].as_str().map(str::to_string),
                label: d["task_label"].as_str().map(str::to_string),
                confident: d["confident"].as_bool(),
                taxonomy: d["taxonomy_id"].as_str().map(str::to_string),
            }
        };
        let results = match (status, batch) {
            (200, false) => vec![result(&v)],
            (200, true) => v["results"]
                .as_array()
                .map(|a| a.iter().map(result).collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        Self {
            request_id: v["request_id"].as_str().map(str::to_string),
            results,
        }
    }
}

/// The `cmf` view of the statistics with the log's origin.
pub fn stats_json(log: &ShadowLog, upstream: &Upstream) -> Value {
    let mut v = log.stats().to_json();
    if let Some(m) = v.as_object_mut() {
        let mut head = Map::new();
        head.insert("shadow_of".into(), json!(upstream.base()));
        head.insert("log".into(), json!(SHADOW_LOG_FILE));
        head.append(m);
        return Value::Object(head);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(
        old: Option<&str>,
        new: Option<&str>,
        oc: Option<bool>,
        nc: Option<bool>,
    ) -> ShadowLine {
        ShadowLine {
            ts: 1_790_000_000,
            request_id_old: Some("req_1".into()),
            text_sha256: Some(text_sha256("x")),
            taxonomy: Some("t".into()),
            old_label: old.map(str::to_string),
            new_label: new.map(str::to_string),
            agree: match (old, new) {
                (Some(a), Some(b)) => Some(a == b),
                _ => None,
            },
            old_confident: oc,
            new_confident: nc,
            old_latency_ms: Some(10.0),
            new_latency_ms: new.map(|_| 2.0),
            old_status: Some(if old.is_some() { 200 } else { 500 }),
            new_error: None,
        }
    }

    #[test]
    fn ureq_debug_lines_are_secret_bearing() {
        use tracing::Level;
        for t in ["ureq", "ureq::unit", "ureq::pool", "ureq::stream"] {
            assert!(log_may_carry_secrets(t, &Level::DEBUG), "{t}");
            assert!(log_may_carry_secrets(t, &Level::TRACE), "{t}");
            assert!(!log_may_carry_secrets(t, &Level::INFO), "{t}");
            assert!(!log_may_carry_secrets(t, &Level::WARN), "{t}");
        }
        for t in ["ureqx", "cortiq_server::decisions", "hyper", "rustls"] {
            assert!(!log_may_carry_secrets(t, &Level::TRACE), "{t}");
        }
    }

    #[test]
    fn base_urls_are_https_or_loopback_http() {
        assert_eq!(
            upstream_base("https://router.example.com/").unwrap(),
            "https://router.example.com"
        );
        assert_eq!(
            upstream_base("HTTPS://r.example.com:8443/api/").unwrap(),
            "https://r.example.com:8443/api"
        );
        assert_eq!(
            upstream_base("http://127.0.0.1:9000").unwrap(),
            "http://127.0.0.1:9000"
        );
        assert_eq!(upstream_base("http://[::1]:80").unwrap(), "http://[::1]:80");
        assert_eq!(
            upstream_base("http://localhost:1").unwrap(),
            "http://localhost:1"
        );
        for bad in [
            "http://router.example.com",
            "http://10.0.0.1:80",
            "ftp://127.0.0.1",
            "router.example.com",
            "https://user:pw@router.example.com",
            "https://router.example.com/?a=1",
            "https://router.example.com/#x",
            "https://:80",
            "https://h:0",
            "https://h:99999",
            "https://[::1",
            "https://::1",
            "https://[not-v6]:80",
            "https://a b",
            "user:secretpw@router.example.com",
        ] {
            let e = upstream_base(bad).unwrap_err().to_string();
            assert!(!e.contains(bad) && !e.contains("secretpw"), "{bad}: {e}");
        }
    }

    #[test]
    fn statistics_count_agreement_overall_confident_and_per_label() {
        let mut s = ShadowStats::default();
        s.add(&line(Some("a"), Some("a"), Some(true), Some(true)));
        s.add(&line(Some("a"), Some("b"), Some(true), Some(false)));
        s.add(&line(Some("b"), Some("b"), Some(false), Some(true)));
        s.add(&line(None, Some("b"), None, Some(true)));
        s.add(&line(Some("a"), None, Some(true), None));
        assert_eq!((s.lines, s.compared, s.agree), (5, 3, 2));
        assert_eq!((s.old_errors, s.old_missing, s.new_missing), (1, 1, 1));
        assert_eq!((s.old_confident, s.old_confident_agree), (2, 1));
        assert_eq!((s.new_confident, s.new_confident_agree), (2, 2));
        assert_eq!((s.both_confident, s.both_confident_agree), (1, 1));
        let a = &s.labels[&("t".to_string(), "a".to_string())];
        assert_eq!((a.lines, a.compared, a.agree), (3, 2, 1));
        assert_eq!(a.new_labels.get("b"), Some(&1));
        let j = s.to_json();
        assert_eq!(j["agreement"], json!(2.0 / 3.0));
        assert_eq!(j["labels"][0]["label"], "a");
        assert_eq!(j["latency_ms"]["old_mean"], json!(10.0));
        assert_eq!(j["latency_ms"]["new_mean"], json!(2.0));
        assert_eq!(ShadowStats::default().to_json()["agreement"], Value::Null);
    }

    #[test]
    fn the_log_replays_its_lines_and_survives_a_torn_last_line() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(SHADOW_LOG_FILE);
        let log = ShadowLog::open(&p).unwrap();
        let lines = [
            line(Some("a"), Some("a"), Some(true), Some(true)),
            line(Some("a"), Some("b"), Some(true), Some(false)),
        ];
        log.append(&lines).unwrap();
        let before = log.stats();
        drop(log);
        // A crash in the middle of a line.
        let mut f = OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(br#"{"ts":1,"request_id_o"#).unwrap();
        drop(f);
        let log = ShadowLog::open(&p).unwrap();
        let s = log.stats();
        assert_eq!(s.malformed, 1);
        assert_eq!((s.lines, s.compared, s.agree), (before.lines, 2, 1));
        log.append(&[line(Some("b"), Some("b"), None, Some(true))])
            .unwrap();
        drop(log);
        let log = ShadowLog::open(&p).unwrap();
        let s = log.stats();
        assert_eq!((s.lines, s.malformed, s.compared, s.agree), (3, 1, 3, 2));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn old_answers_give_ids_labels_and_confidence() {
        let one = br#"{"schema_version":"1.1","request_id":"req_a","decision":{"task_label":"x","confident":true,"taxonomy_id":"t"}}"#;
        let a = OldAnswer::parse(200, one, false);
        assert_eq!(a.request_id.as_deref(), Some("req_a"));
        assert_eq!(
            a.results,
            [OldResult {
                request_id: Some("req_a".into()),
                label: Some("x".into()),
                confident: Some(true),
                taxonomy: Some("t".into()),
            }]
        );
        let batch = br#"{"schema_version":"1.1","results":[{"request_id":"req_b","decision":{"task_label":"y"}},{"request_id":"req_c","decision":{"task_label":"z","confident":false}}]}"#;
        let b = OldAnswer::parse(200, batch, true);
        assert_eq!(b.request_id, None);
        assert_eq!(b.results.len(), 2);
        assert_eq!(b.results[1].confident, Some(false));
        let err =
            br#"{"schema_version":"1.1","request_id":"req_e","error":{"code":"UNAUTHORIZED"}}"#;
        let e = OldAnswer::parse(401, err, false);
        assert_eq!(e.request_id.as_deref(), Some("req_e"));
        assert!(e.results.is_empty());
        assert_eq!(
            OldAnswer::parse(200, b"not json", false),
            OldAnswer::default()
        );
    }
}
