//! Shadow mode of the router API: `cortiq serve FILE --shadow-of URL` (spec
//! §4.15, the switch of production traffic from `cortiq-router`).
//!
//! Every request on a path of the router's own API ([`SHADOW_FORWARDED_PATHS`]:
//! `/v1/route`, `/v1/route:batch`, `/v1/feedback`, `/v1/taxonomies[/…]`,
//! `/v1/usage`, `/v1/escalations`, `/v1/healthz`, `/v1/readyz`, `/metrics`,
//! `/v1/admin/keys[/{account}]`) is sent to the old router unchanged — method,
//! path and query, body, the client's `Authorization` / `x-api-key` /
//! `x-admin-token` / `Content-Type` / `Accept` / `User-Agent` — and its answer
//! goes back to the client as it came: status, body bytes and headers
//! (hop-by-hop headers and the length aside; no `x-request-id` is added). Old
//! router errors (401, 402, 404, 413, 415, 422, 429 with `Retry-After`, 5xx)
//! pass through the same way. Only when the old router gives no answer at all
//! (refused connection, TLS, the deadline — [`UPSTREAM_TIMEOUT`], 60 s, or
//! `--shadow-timeout-s` — a body over 64 MiB) does
//! the client get this server's 502 `UPSTREAM_UNAVAILABLE` in the router's
//! envelope, like the 502 of a proxy; a request body over the old router's
//! 8 MiB limit is answered 413 `length limit exceeded`, as it answers it.
//!
//! For `POST /v1/route` and `POST /v1/route:batch` the same inputs are also
//! decided here, while the old router answers, by
//! [`DecisionService::decide_local`](cortiq_decision::service::DecisionService::decide_local):
//! the request's taxonomy and `policy_profile`, and nothing else — no oracle
//! and no cache, no learning (no buffer, no feedback link), no billing (no
//! key check, rate or quota, no usage record). Each routed input becomes one
//! line of `<state>/shadow.jsonl` ([`ShadowLine`]; no text, only its SHA-256)
//! once both sides are done; the client's answer does not wait for the local
//! decision. `GET /v1/admin/shadow` (admin token) returns the agreement
//! statistics of the whole log, overall, by confidence and per old label.
//!
//! The decisions API (`/api/alpha/decisions`, `/v1/decisions`, `/v1/models`,
//! `/v1/skills`, `/healthz`) and this server's own admin API stay local.
//!
//! The forwarded client secrets never reach a log line: the request line has
//! only id, status, latency and account `-`, and `cortiq serve` drops the
//! DEBUG/TRACE lines of the HTTP client that print request headers
//! ([`log_may_carry_secrets`](cortiq_decision::shadow::log_may_carry_secrets))
//! whatever `RUST_LOG` enables.

use super::{
    Ctx, DecisionState, HttpError, NOVEL_LABEL, ROUTER_MAX_BATCH, Reply, Surface, admin_guard,
    error_response, read_body, respond, route_question, route_request, router_request_id,
    truncate_utf8, wire,
};
use anyhow::Result;
use axum::body::{Body, Bytes};
use axum::extract::{Extension, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use cortiq_decision::keys::now_unix;
#[cfg(doc)]
use cortiq_decision::shadow::UPSTREAM_TIMEOUT;
use cortiq_decision::shadow::{
    FORWARDED_REQUEST_HEADERS, OldAnswer, ShadowLine, ShadowLog, UPSTREAM_LENGTH_LIMIT,
    UPSTREAM_MAX_BODY, Upstream, UpstreamError, ms, stats_json, text_sha256,
};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The router-API paths forwarded in shadow mode (router `api.rs:292-307`);
/// `{…}` is one path segment.
pub const SHADOW_FORWARDED_PATHS: [&str; 12] = [
    "/v1/route",
    "/v1/route:batch",
    "/v1/feedback",
    "/v1/taxonomies",
    "/v1/taxonomies/…",
    "/v1/usage",
    "/v1/escalations",
    "/v1/healthz",
    "/v1/readyz",
    "/metrics",
    "/v1/admin/keys",
    "/v1/admin/keys/{account}",
];

/// How long a stopping server waits for comparisons still being decided.
pub(super) const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// The route template of a path shadow mode forwards: the old
/// router's routes (anything under `/v1/taxonomies/`, which it answers
/// itself, and one segment under `/v1/admin/keys/`); `None` for the paths that
/// stay local, this server's own admin paths among them.
pub(super) fn forwarded_route(path: &str) -> Option<&'static str> {
    const EXACT: [&str; 10] = [
        "/v1/route",
        "/v1/route:batch",
        "/v1/feedback",
        "/v1/taxonomies",
        "/v1/usage",
        "/v1/escalations",
        "/v1/healthz",
        "/v1/readyz",
        "/metrics",
        "/v1/admin/keys",
    ];
    if let Some(t) = EXACT.iter().find(|t| **t == path) {
        return Some(t);
    }
    if path.starts_with("/v1/taxonomies/") {
        return Some("/v1/taxonomies/{id}");
    }
    path.strip_prefix("/v1/admin/keys/")
        .is_some_and(|rest| !rest.is_empty() && !rest.contains('/'))
        .then_some("/v1/admin/keys/{account}")
}

/// The old router, the comparison log and the comparisons in progress.
pub(super) struct Shadow {
    upstream: Upstream,
    log: ShadowLog,
    pending: AtomicUsize,
}

impl std::fmt::Debug for Shadow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shadow")
            .field("upstream", &self.upstream)
            .field("log", &self.log)
            .finish()
    }
}

impl Shadow {
    pub(super) fn open(upstream: Upstream, log: &Path) -> Result<Self> {
        Ok(Self {
            upstream,
            log: ShadowLog::open(log)?,
            pending: AtomicUsize::new(0),
        })
    }

    pub(super) fn upstream_base(&self) -> &str {
        self.upstream.base()
    }

    pub(super) fn lines(&self) -> u64 {
        self.log.stats().lines
    }

    /// Wait (at most `timeout`) until every comparison is written.
    pub(super) async fn drain(&self, timeout: Duration) {
        let t0 = Instant::now();
        while self.pending.load(Ordering::SeqCst) > 0 && t0.elapsed() < timeout {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    pub(super) fn sync(&self) -> Result<()> {
        self.log.sync()
    }
}

/// Counts a comparison in progress until dropped.
struct PendingGuard(Arc<Shadow>);

impl PendingGuard {
    fn new(sh: &Arc<Shadow>) -> Self {
        sh.pending.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(sh))
    }
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.0.pending.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Forward a router-API path to the old router (see the module notes), before
/// any routing of this server; any other path goes on to its own routes. One
/// log line per forwarded request with the fields of spec §4.3, like the other
/// requests': id, status, latency and account (`-`: the old router checks the
/// key) — never a method, body, path, query or header.
pub(super) async fn shadow_middleware(
    State(st): State<Arc<DecisionState>>,
    req: Request,
    next: Next,
) -> Response {
    let (Some(sh), Some(_)) = (st.shadow.clone(), forwarded_route(req.uri().path())) else {
        return next.run(req).await;
    };
    let t0 = Instant::now();
    let ctx = Ctx {
        id: router_request_id(),
        surface: Surface::Router,
        ext: false,
        path: req.uri().path().to_string(),
    };
    let resp = forward(st, sh, ctx.clone(), req).await;
    tracing::info!(
        id = %ctx.id,
        status = resp.status().as_u16(),
        latency_ms = t0.elapsed().as_secs_f64() * 1000.0,
        account = "-",
        "shadow: forwarded to the old router"
    );
    resp
}

/// `/v1/route` or `/v1/route:batch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteKind {
    One,
    Batch,
}

/// The old router's answer, shared between the client and the comparison.
#[derive(Clone, Debug)]
struct Answer {
    status: u16,
    headers: Vec<(String, String)>,
    body: Bytes,
}

async fn forward(st: Arc<DecisionState>, sh: Arc<Shadow>, ctx: Ctx, req: Request) -> Response {
    let ts = now_unix();
    let (parts, body) = req.into_parts();
    let path = parts.uri.path().to_string();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map_or_else(|| path.clone(), |p| p.as_str().to_string());
    let bytes = match read_body(&parts.headers, body, UPSTREAM_MAX_BODY).await {
        Ok(b) => Bytes::from(b),
        Err(e) if e.status == 413 => {
            return error_response(&ctx, HttpError::plain(413, UPSTREAM_LENGTH_LIMIT));
        }
        Err(e) => return error_response(&ctx, e),
    };
    let headers = forwarded_headers(&parts.headers);
    let method = parts.method.as_str().to_string();
    let kind = match (parts.method == Method::POST, path.as_str()) {
        (true, "/v1/route") => Some(RouteKind::One),
        (true, "/v1/route:batch") => Some(RouteKind::Batch),
        _ => None,
    };
    let (tx, rx) = tokio::sync::oneshot::channel::<Result<Answer, UpstreamError>>();
    let guard = PendingGuard::new(&sh);
    // One task per request: the comparison is written even when the client
    // goes away before the old router answers.
    tokio::spawn(async move {
        let _guard = guard;
        // The local decision starts first and runs while the old router answers.
        let local = kind.map(|k| {
            let (st, body) = (Arc::clone(&st), bytes.clone());
            tokio::task::spawn_blocking(move || decide_shadow(&st, k, &body))
        });
        let up = {
            let (sh, body) = (Arc::clone(&sh), bytes.clone());
            tokio::task::spawn_blocking(move || {
                sh.upstream
                    .forward(&method, &path_and_query, &headers, &body)
            })
            .await
            .unwrap_or_else(|e| {
                Err(UpstreamError {
                    reason: format!("task_{e}"),
                    latency: Duration::ZERO,
                })
            })
        };
        let (answer, old_status, old_latency) = match up {
            Ok(r) => (
                Ok(Answer {
                    status: r.status,
                    headers: r.headers,
                    body: Bytes::from(r.body),
                }),
                Some(r.status),
                r.latency,
            ),
            Err(e) => {
                let latency = e.latency;
                (Err(e), None, latency)
            }
        };
        let old_body = answer.as_ref().ok().map(|a| (a.status, a.body.clone()));
        let _ = tx.send(answer);
        let (Some(kind), Some(local)) = (kind, local) else {
            return;
        };
        let new = local
            .await
            .unwrap_or_else(|_| NewSide::failed(None, "INTERNAL"));
        // Parsing the old answer and writing the lines: the blocking pool too.
        let written = tokio::task::spawn_blocking(move || {
            let old = old_body
                .map(|(status, body)| OldAnswer::parse(status, &body, kind == RouteKind::Batch))
                .unwrap_or_default();
            sh.log
                .append(&shadow_lines(ts, old_status, old_latency, &old, &new))
        })
        .await;
        match written {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(error = %e, "shadow: a comparison could not be written"),
            Err(e) => tracing::warn!(error = %e, "shadow: a comparison could not be written"),
        }
    });
    match rx.await {
        Ok(Ok(a)) => passthrough(a),
        Ok(Err(e)) => {
            tracing::warn!(
                reason = %e.reason,
                latency_ms = ms(e.latency),
                "shadow: the old router did not answer"
            );
            error_response(&ctx, upstream_unavailable())
        }
        Err(_) => error_response(&ctx, upstream_unavailable()),
    }
}

/// The client headers the old router reads (see
/// [`FORWARDED_REQUEST_HEADERS`]), every value in order.
fn forwarded_headers(h: &HeaderMap) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for name in FORWARDED_REQUEST_HEADERS {
        for v in h.get_all(name) {
            if let Ok(v) = v.to_str() {
                out.push((name.to_string(), v.to_string()));
            }
        }
    }
    out
}

fn passthrough(a: Answer) -> Response {
    let mut resp = Response::new(Body::from(a.body));
    *resp.status_mut() = StatusCode::from_u16(a.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let h = resp.headers_mut();
    for (k, v) in &a.headers {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            h.append(n, v);
        }
    }
    resp
}

fn upstream_unavailable() -> HttpError {
    HttpError::new(
        502,
        "UPSTREAM_UNAVAILABLE",
        "the router this server shadows (--shadow-of) did not answer",
    )
}

// ------------------------------------------------------------------ the local side

/// The local decision of one routed input.
#[derive(Clone, Debug, Default)]
struct NewInput {
    text_sha256: Option<String>,
    label: Option<String>,
    confident: Option<bool>,
    error: Option<&'static str>,
}

/// The local side of a routed request.
#[derive(Clone, Debug, Default)]
struct NewSide {
    /// The request's `taxonomy_id`, else the skill it resolves to.
    taxonomy: Option<String>,
    /// One per input of the request (empty when the body did not parse).
    inputs: Vec<NewInput>,
    /// Why no input was decided (the whole request).
    error: Option<&'static str>,
    /// Wall time of the request's local decisions (when one was made).
    elapsed: Option<Duration>,
}

impl NewSide {
    fn failed(taxonomy: Option<String>, code: &'static str) -> Self {
        Self {
            taxonomy,
            error: Some(code),
            ..Self::default()
        }
    }
}

/// Decide the inputs of a routed request locally, as `/v1/route` would before
/// escalating: [`DecisionService::decide_local`](cortiq_decision::service::DecisionService::decide_local)
/// (no oracle, no learning, no billing).
fn decide_shadow(st: &DecisionState, kind: RouteKind, body: &[u8]) -> NewSide {
    const INVALID: &str = "INVALID_REQUEST";
    let t0 = Instant::now();
    let parsed = match kind {
        RouteKind::One => serde_json::from_slice::<wire::RouteRequest>(body)
            .map(|r| (r.taxonomy_id, vec![r.input], r.options)),
        RouteKind::Batch => serde_json::from_slice::<wire::BatchRequest>(body)
            .map(|r| (r.taxonomy_id, r.inputs, r.options)),
    };
    let Ok((taxonomy_id, inputs, options)) = parsed else {
        return NewSide::failed(None, INVALID);
    };
    if inputs.len() > ROUTER_MAX_BATCH {
        return NewSide::failed(taxonomy_id, INVALID);
    }
    let mut side = NewSide {
        taxonomy: taxonomy_id.clone(),
        inputs: inputs
            .iter()
            .map(|i| NewInput {
                // A bring-your-own embedding makes the router ignore the text.
                text_sha256: match (&i.embedding, &i.text) {
                    (None, Some(t)) => Some(text_sha256(t)),
                    _ => None,
                },
                ..NewInput::default()
            })
            .collect(),
        ..NewSide::default()
    };
    let routing = match st.resolve_routing(taxonomy_id.as_deref(), &options, false) {
        Ok(r) => r,
        Err(e) => {
            side.error = Some(e.code);
            return side;
        }
    };
    side.taxonomy = Some(taxonomy_id.unwrap_or_else(|| routing.skill.clone()));
    let Ok(_slot) = st.svc.enter() else {
        side.error = Some("OVERLOADED");
        return side;
    };
    let model = st.svc.handle().current();
    let Some(skill) = model.skill(&routing.skill) else {
        side.error = Some(super::TAXONOMY_NOT_FOUND);
        return side;
    };
    if skill.scorer().is_empty() {
        side.error = Some(INVALID);
        return side;
    }
    let question = route_question(skill);
    let limit = st.config().limits.state_bytes;
    let mut decided = false;
    for (input, out) in inputs.iter().zip(&mut side.inputs) {
        if input.embedding.is_some() {
            // Another encoder's space: not comparable.
            out.error = Some("EMBEDDING_INPUT");
            continue;
        }
        let Some(text) = input.text.as_deref().filter(|t| !t.trim().is_empty()) else {
            out.error = Some("EMBEDDING_REQUIRED");
            continue;
        };
        let req = route_request(&routing, question.clone(), truncate_utf8(text, limit));
        match st.svc.decide_local(&req) {
            Ok(lo) => match lo.questions.first() {
                Some((_, Some(l))) => {
                    out.label = Some(l.choice.clone().unwrap_or_else(|| NOVEL_LABEL.to_string()));
                    out.confident = Some(l.accepted && !l.is_novel());
                    decided = true;
                }
                _ => out.error = Some("INTERNAL"),
            },
            Err(e) => out.error = Some(e.reason.code()),
        }
    }
    if decided {
        side.elapsed = Some(t0.elapsed());
    }
    side
}

/// The lines of one routed request: one per input (one when the body did not
/// parse), paired with the old results by position.
fn shadow_lines(
    ts: u64,
    old_status: Option<u16>,
    old_latency: Duration,
    old: &OldAnswer,
    new: &NewSide,
) -> Vec<ShadowLine> {
    let n = new.inputs.len().max(1);
    (0..n)
        .map(|i| {
            let o = old.results.get(i);
            let ni = new.inputs.get(i);
            let old_label = o.and_then(|r| r.label.clone());
            let new_label = ni.and_then(|x| x.label.clone());
            let agree = match (&old_label, &new_label) {
                (Some(a), Some(b)) => Some(a == b),
                _ => None,
            };
            let new_error = match new_label {
                Some(_) => None,
                None => ni.and_then(|x| x.error).or(new.error).map(str::to_string),
            };
            ShadowLine {
                ts,
                request_id_old: o
                    .and_then(|r| r.request_id.clone())
                    .or_else(|| old.request_id.clone()),
                text_sha256: ni.and_then(|x| x.text_sha256.clone()),
                taxonomy: o
                    .and_then(|r| r.taxonomy.clone())
                    .or_else(|| new.taxonomy.clone()),
                new_latency_ms: new_label.as_ref().and(new.elapsed).map(ms),
                old_label,
                new_label,
                agree,
                old_confident: o.and_then(|r| r.confident),
                new_confident: ni.and_then(|x| x.confident),
                old_latency_ms: Some(ms(old_latency)),
                old_status,
                new_error,
            }
        })
        .collect()
}

// ------------------------------------------------------------------ admin

/// `GET /v1/admin/shadow` (admin token): the agreement statistics of
/// `shadow.jsonl` ([`cortiq_decision::shadow::ShadowStats::to_json`]) with
/// `shadow_of` and `log`.
pub(super) async fn admin_shadow(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        admin_guard(&st, &headers)?;
        // The log's lock is held across its writes: read it on the blocking pool.
        let s = Arc::clone(&st);
        let v = super::blocking(move || {
            Ok(match &s.shadow {
                Some(sh) => stats_json(&sh.log, &sh.upstream),
                None => Value::Null,
            })
        })
        .await?;
        Ok(Reply::ok(v))
    }
    .await;
    respond(&ctx, out)
}
