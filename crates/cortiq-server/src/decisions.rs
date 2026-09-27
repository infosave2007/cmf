//! Decision server: the HTTP layer of `cortiq serve` on a decision file
//! (spec decision-v4 §4.2–§4.15, §5, §5b).
//!
//! `cortiq serve FILE` on a file with the DECISION feature bit serves this
//! router instead of the language-model router of [`crate::build_router`]: no
//! `Pipeline` is built, no GPU backend is touched. Everything the endpoints do is
//! [`cortiq_decision::service::DecisionService`] (Jev/OpenRouter protocol,
//! matching, metering, keys, usage ledger) and
//! [`cortiq_decision::cascade::Cascade`] (oracle, cache, learning, generations);
//! this module maps HTTP to them.
//!
//! # Endpoints
//!
//! Decisions protocol (Jev / OpenRouter shape, spec §4.3):
//!
//! | Path | Access |
//! |---|---|
//! | `POST /api/alpha/decisions`, `POST /v1/decisions` | key |
//! | `GET /v1/models` | open (OpenRouter provider listing, §4.12) |
//! | `GET /v1/skills`, `GET /v1/skills/{id}` | key |
//! | `GET /healthz` | open |
//!
//! The current `cortiq-router` API (schema `1.1`, spec §4.15; router
//! `api.rs:292-307`), served at the same time so that existing clients switch by
//! changing the backend only:
//!
//! | Path | Access |
//! |---|---|
//! | `POST /v1/route`, `POST /v1/route:batch` | key |
//! | `POST /v1/feedback` (router body or `{"id","question","label"}`, §5.11) | key |
//! | `GET /v1/taxonomies`, `GET /v1/taxonomies/{id}` | key |
//! | `GET /v1/usage` (the caller's account only) | key |
//! | `GET /v1/escalations` (the caller's records only) | key |
//! | `GET /v1/healthz`, `GET /v1/readyz`, `GET /metrics` | open |
//! | `POST, GET /v1/admin/keys`, `DELETE /v1/admin/keys/{account}` | `x-admin-token` |
//!
//! Every router endpoint answers with exactly the router's keys and JSON types
//! (`tests/router_compat.rs` checks them against router `api.rs`); the
//! decision-v4 additions are opt-in: a request with the header
//! `x-cmf-extensions: 1` gets them under an additive `cmf` key (see
//! [`CMF_EXTENSIONS_HEADER`]).
//!
//! Administration (§5b, `x-admin-token`; 404 `ADMIN_DISABLED` when the token's
//! environment variable is unset): `DELETE /v1/admin/keys/hash/{hash12}`,
//! `GET /v1/admin/usage`, `GET|POST /v1/admin/oracle`, `GET /v1/admin/learning`,
//! `GET /v1/admin/generations`, `POST /v1/admin/rollback {"generation": N}`.
//!
//! There is no CORS layer and no web interface of any kind: on the decisions
//! surface an unknown path is a JSON 404, a wrong method a JSON 405 (the router
//! surface answers them as the router does, see Errors).
//!
//! # Shadow mode (`--shadow-of URL`, spec §4.15 migration)
//!
//! [`ServeOptions::shadow_of`]: every request on a router-API path
//! ([`SHADOW_FORWARDED_PATHS`]) is forwarded unchanged to the old router at
//! `URL`, before any routing here, and its answer goes back to the client as
//! it came (status, body bytes, headers; errors included). A `POST /v1/route`
//! or `POST /v1/route:batch` the old router answered 200 is also decided
//! locally, after that answer, without any oracle, learning or billing and
//! within a few comparison slots of its own, and compared in
//! `<state>/shadow.jsonl` (one line per input, never the text);
//! `GET /v1/admin/shadow` (admin token) answers the agreement statistics.
//! Without the flag none of this exists (not even the admin path). Details:
//! the `shadow` submodule.
//!
//! # The oracle in two steps (`--oracle MODEL`)
//!
//! `cortiq serve FILE --oracle MODEL` with the key in `OPENROUTER_API_KEY`
//! turns the oracle on; the command line applies it to the configuration
//! ([`cortiq_decision::oracle_setup`]: budget, max price from the public
//! endpoint listing, …) and sets [`ServeOptions::oracle_from_flag`], which
//! lets the open mode of a loopback address (no keys, `auth.require: null`)
//! use the oracle — never teach the model: its feedback is not learned, and
//! the oracle's answers to its questions (a skill's own question, as
//! `/v1/route` asks it, included) are cached but never become examples. The
//! server logs one startup line of the oracle ([`oracle_startup_line`]:
//! `ready`, `NOT ready — <what to do>` or `off`; a key that had surrounding
//! whitespace, trimmed before use, adds one warning), `GET /v1/admin/oracle`
//! has `status` (`ready`, `no_key`, `bad_key`, `disabled`,
//! `budget_exhausted`, `budget_too_small`, `stopped: <reason>`; with
//! `key_problem`, `last_error` and `min_call_usd`), `/healthz` has
//! `oracle_status` (on `/v1/healthz` only under `cmf`, with
//! `x-cmf-extensions`), and a trained question that abstains because the
//! oracle is not ready carries the reason's flags (`oracle_disabled`,
//! `no_key`, `bad_key`, `budget`, `stopped`) and, on the decisions surface,
//! one line `cmf.hint`; router answers keep their shapes and their flag
//! vocabulary (a missing or unusable key is `oracle_disabled` there, without
//! `no_key` or `bad_key`; the hint is logged: each distinct hint at most once
//! a minute, once per process at INFO on a server without an oracle). No
//! line, status or hint ever holds a byte of the key: a malformed key is
//! never sent, and transport errors are fixed codes.
//!
//! # Every request
//!
//! * **`x-request-id`** on every response: the decision's `cmf-dec-…` id for a
//!   decision, else an id minted when the request arrived (`cmf-dec-…` on the
//!   decisions surface, this server's own admin paths included; `req_…` —
//!   router `ids.rs` — on the router surface); JSON error bodies carry the
//!   same id.
//! * **Keys** (§4.10): `Authorization: Bearer <key>` or `x-api-key`; the open
//!   mode (no key needed) holds only when `keys.json` has no key and
//!   `auth.require` is false (default: false on loopback, true elsewhere). The
//!   open caller may reach the oracle and teach the model only when
//!   `auth.require` is false in the configuration: a loopback address alone
//!   says nothing about the client (a reverse proxy on the same host). On a
//!   keyed endpoint the rate window (429 `RATE_LIMITED`, `Retry-After`) and the
//!   quotas (402 `QUOTA_EXCEEDED`: decisions, tokens, credit) are checked before
//!   any work, like the router's middleware, and again before every input of a
//!   batch after the first; a `/v1/decisions` request with more questions than
//!   the decision quota has left is refused, and an oracle call must fit in
//!   the credit left (`DecisionService::check_decision_room`,
//!   `DecisionService::oracle_credit_left`).
//! * **Body**: `Content-Type: application/json` (else 400; 415 on the router
//!   surface), at most `limits.body_bytes` (else 413 `PAYLOAD_TOO_LARGE`,
//!   checked from `Content-Length` before reading and while reading).
//! * **Work** runs on the blocking pool (`spawn_blocking`): authentication (it
//!   may re-read `keys.json`), decisions, feedback (a learning attempt may run
//!   synchronously), every admin operation and the views that read the usage
//!   ledger or the learning state (`/v1/usage`, `/v1/escalations`, `/metrics`),
//!   whose locks are held across `fsync`. A decision, route or feedback
//!   request takes one of `limits.max_inflight` slots first (else 429
//!   `OVERLOADED`, `Retry-After: 1`).
//! * **Logs**: one line per request with exactly the fields of §4.3: id,
//!   status, latency and account — never a method, a path or query, a body, a
//!   state, a key or a header value.
//!
//! # Errors
//!
//! Decisions surface (`/api/alpha/decisions`, `/v1/decisions`, `/v1/models`,
//! `/v1/skills*`, `/healthz`, this server's own admin paths
//! `/v1/admin/keys/hash/{hash}`, `/v1/admin/usage`, `/oracle`, `/learning`,
//! `/generations`, `/rollback`, `/shadow`, and unknown paths) — OpenRouter's
//! shape with the router's reason codes (§4.8):
//! `{"error":{"code":<HTTP>,"message":"…","metadata":{"reason":"<CODE>","retriable":bool,"request_id":"…","details":{}}}}`.
//! A path parameter axum cannot extract (`%FF`) is a JSON 400
//! `INVALID_REQUEST` in the path's envelope on both surfaces.
//!
//! Router surface (the router's own paths, `/v1/admin/keys` and
//! `DELETE /v1/admin/keys/{account}` among them) — the router's envelope (router
//! `api.rs:207-284`; §4.15: where a path is the router's, its format wins):
//! `{"schema_version":"1.1","request_id":"req_…","error":{"code":"<CODE>","message":"…","retriable":bool,"details":null}}`;
//! `details` is an object only for `TAXONOMY_NOT_FOUND` (`{"taxonomy_id"}`),
//! like the router's; `retriable` is true only for 429 and 500 (router
//! `api.rs:221-225`); the messages of the rate window, the decision quota, a
//! key that is not valid now (expired and revoked alike) and a disabled admin
//! API are the router's texts. The router's own codes `TAXONOMY_NOT_FOUND`
//! (404) and `EMBEDDING_REQUIRED` (400) appear only there. With
//! `x-cmf-extensions: 1` the error also carries the decisions surface's
//! `metadata` object (with this server's own message when it differs).
//!
//! A router body the router's axum extractors would reject is answered as
//! they answer it, as `text/plain; charset=utf-8`: 415 ``Expected request with
//! `Content-Type: application/json` `` (a JSON media type is
//! `application/json` or `application/*+json`), 413 `Failed to buffer the
//! request body: length limit exceeded` (`limits.body_bytes`), 400 `Failed to
//! parse the request body as JSON: …` and 422 `Failed to deserialize the JSON
//! body into the target type: …` (the router's struct names), and on
//! `/v1/escalations` 400 `Failed to deserialize query string: …`. A wrong
//! method on a router path is an empty 405 with `Allow`, an unknown path under
//! a router prefix an empty 404; both after the key check when the path is
//! keyed, as the router's middleware runs first.
//!
//! # Router API on the decision model
//!
//! A `/v1/route` input is one decision of the skill named by `taxonomy_id`
//! (else `default_skill`, else the file's only skill): a `task` choice question
//! over the skill's active labels, with the skill's rubric (instructions and
//! criteria, question-file order) — the Jev request the oracle and the cache
//! see, metered like one. The request is the router's (`api.rs:30-89`, unknown
//! keys ignored): `options.policy_profile` (`balanced`) is the profile of
//! §4.7b, `allow_oracle` (`true`) the consent to the oracle, `allow_pii_egress`
//! (`false`), `return_explanation` (`false`), `top_k` (3, clamped to 1..64) and
//! `routing_table_id` as in the router. `input.text` longer than
//! `limits.state_bytes` is cut there at a character boundary (the router has no
//! text limit; the encoder reads 512 tokens). The response has every field of
//! router `api.rs:95-201` and no other: `decision.confidence` is the calibrated
//! `p_top` (1 for an oracle or cache answer, whose label is promoted to the top
//! of `scores` as the router does), `raw_confidence` the winner's `1/(1+E)`,
//! `scores[]` the candidates by score with `probability`, `score` and
//! `reconstruction_error`, `task_id` the task's index in the skill (−1 with
//! `__novel__` when there is no candidate), `complexity` and `routing` (only
//! with `routing_table_id`, like the router) from §4.7b, `oracle` {consulted,
//! model, agreement_with_router, latency_ms}, `explanation` {top1_vs_top2,
//! decision_path} with the decision paths of §4.7b, `usage`
//! {billable_decisions: 1, oracle_calls}, `meta` {model_version =
//! `cortiq/decision@<sha12>`, taxonomy_version = `<skill>@<n>`, latency_ms (the
//! resonance), embedding_latency_ms (tokenizer, encoder and hash), served_by =
//! `cortiq/<version>`}. A gate-rejected answer that is not escalated carries the
//! router's `low_confidence` flag (so `confident` is false). With
//! `x-cmf-extensions: 1` a `cmf` object adds the `cmf-dec-…` id, the action,
//! `certified` and the gate.
//!
//! `input.embedding` (bring your own) must have the signal's dimension (the
//! encoder's plus the hashing contract's) and `embedding_model`; it is decided
//! locally and never sent to the oracle (no text: `oracle_unavailable` when the
//! oracle would be consulted, as in the router), is not certified, is billed as
//! one decision with 0 input tokens, and can be corrected with feedback.
//!
//! `POST /v1/route:batch` decides up to 1024 inputs in order and fails as a whole
//! on the first failing input, like the router (the inputs before it are
//! decided and billed); a quota or credit used up by the batch's own inputs
//! fails it with 402 at the next input.
//!
//! `POST /v1/feedback` with the router's `{request_id, correct_task_label}`
//! corrects a route decision of the caller's own account (another account's is
//! not found — the router let any key correct any request). Any label is
//! accepted, as in the router: a label of 1..256 bytes the skill does not have
//! starts a cold start (§5.8); an empty or longer one is answered 200 and
//! consumes the request but teaches nothing (it names no task). Only a key
//! with `learning_allowed` teaches the model; anyone else's feedback is
//! answered 200 with `accepted: false` and consumes the request. The response
//! is the router's `{schema_version, accepted, message}`. A body with the
//! decisions API's keys (`id`, `question`, `label`) and none of the router's
//! is decisions-API feedback (§5.11). As in the router (axum 0.7.9), bytes
//! after a router body's first JSON value are ignored.
//!
//! The listings are the router's: `/v1/taxonomies` `{schema_version,
//! taxonomies:[{taxonomy_id, taxonomy_version, model_version, labels}]}`
//! (`default_skill` first), `/v1/usage` `{schema_version, account:{id,
//! billable_decisions, oracle_calls, decision_quota, rate_per_min},
//! usage:{billable_decisions, oracle_calls, escalations, escalation_rate,
//! cache_hits, novelty_hits, refits, promotions}}` with the caller's own
//! numbers, `/v1/escalations` `{schema_version, summary:{total, oracle_calls,
//! cache_hits, oracle_unavailable, labeled_examples, cache:{entries, hits,
//! lookups}}, records:[router AuditRecord]}`, `/v1/healthz` `{"status":"ok"}`.
//! The admin key API takes the router's `CreateKeyReq` (plus the decision-v4
//! limits `token_quota`, `credit_usd`, `oracle_budget_usd`, `oracle_allowed`,
//! `learning_allowed`; `email` is not stored) and answers `{key, account,
//! plan, rate_per_min, decision_quota, expires_at, created_at, persisted}`;
//! its listing has the active keys as `{account, plan, rate_per_min,
//! decision_quota, expires_at, expired, key_hash_prefix, usage:{decisions,
//! oracle_calls}}`. An account is any text of 1..128 characters, as in the
//! router; unlike the router (which mints a key without limits for a plan it
//! does not know), a plan must be one of `auth.plans` (400).

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use axum::Router;
use axum::body::Body;
use axum::extract::rejection::PathRejection;
use axum::extract::{Extension, Path, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{delete, get, post};
use cortiq_decision::cascade::{Cascade, CascadeOptions};
use cortiq_decision::config::Config;
use cortiq_decision::container::Verify;
use cortiq_decision::eval::{f32_json, jev_confidence};
use cortiq_decision::generation;
use cortiq_decision::keys::{AuthFailure, KeyStore, now_unix};
use cortiq_decision::ledger::{Actions, FLUSH_EVERY, Flusher, UsageLedger, UsageRecord};
use cortiq_decision::matching::{MatchKind, SkillMatch};
use cortiq_decision::metering::{Rates, Usd};
use cortiq_decision::protocol::{
    ApiError, CmfOptions, DecisionRequest, ModelRef, Profile, Question, QuestionKind, Reason,
    State as RequestState, new_request_id, parse_json,
};
use cortiq_decision::service::{
    Action, AdminCommand, Decided, DecisionService, Escalator, LoadedModel, LocalDecision,
    ModelHandle, Observation, Principal, QUALITY_FIRST_MARGIN, QUALITY_FIRST_NOVELTY,
    QuestionOutcome, RefusalReason, SkillRuntime, estimate_complexity,
};
use cortiq_decision::signal::Features;
use cortiq_decision::statedir::{StateDir, StateLock};
use futures::StreamExt;
use serde_json::{Map, Value, json};

mod shadow;
pub use shadow::SHADOW_FORWARDED_PATHS;

/// Default listening host of a decision server (spec §4.2): loopback.
pub const DEFAULT_HOST: &str = "127.0.0.1";
/// Default port of a decision server (spec §4.2).
pub const DEFAULT_PORT: u16 = 8080;
/// `schema_version` of the router API (router `api.rs:23`).
pub const ROUTER_SCHEMA_VERSION: &str = "1.1";
/// Most inputs of one `/v1/route:batch` (router `api.rs:1209`).
pub const ROUTER_MAX_BATCH: usize = 1024;
/// `options.top_k` of a router request without it (router `api.rs:68`).
pub const ROUTER_DEFAULT_TOP_K: usize = 3;
/// `options.top_k` is clamped to 1..=64 (router `api.rs:919`).
pub const ROUTER_MAX_TOP_K: usize = 64;
/// Question id of the decision a router input becomes.
pub const ROUTE_QUESTION_ID: &str = "task";
/// Label of a router decision without any candidate (router `api.rs:759`).
pub const NOVEL_LABEL: &str = "__novel__";
/// Escalation records kept in memory (router `config.rs:351`).
pub const AUDIT_CAP: usize = 10_000;
/// Default and largest `limit` of `GET /v1/escalations` (router `api.rs:1377`).
pub const ESCALATIONS_DEFAULT_LIMIT: usize = 100;
pub const ESCALATIONS_MAX_LIMIT: usize = 1000;
/// The request id header of every response.
pub const REQUEST_ID_HEADER: &str = "x-request-id";
/// Request header that opts a router-API call into the decision-v4 additions
/// (`1`, `true`, `yes` or `on`): a `cmf` object in route results, listings,
/// usage, feedback and `/v1/healthz`, revoked keys in the admin listing, and
/// `error.metadata` in router errors. Without it every router endpoint answers
/// with exactly the router's keys.
pub const CMF_EXTENSIONS_HEADER: &str = "x-cmf-extensions";
/// The router's `schema_version` 1.1 answer when its axum `Json` extractor
/// finds no JSON media type (axum 0.7 `MissingJsonContentType`).
pub const ROUTER_MISSING_JSON_CONTENT_TYPE: &str =
    "Expected request with `Content-Type: application/json`";
/// The router's 413 text (axum 0.7 `FailedToBufferBody::LengthLimitError`).
pub const ROUTER_LENGTH_LIMIT: &str = "Failed to buffer the request body: length limit exceeded";
/// Most bytes of a router feedback label that can be learned (the decisions
/// API's option ids); a longer or empty one is accepted, as by the router,
/// but teaches nothing.
pub const ROUTER_MAX_LABEL_BYTES: usize = cortiq_decision::protocol::MAX_LABEL_BYTES;
/// Instructions of the router question of a skill without a rubric.
pub const DEFAULT_ROUTE_INSTRUCTIONS: &str = cortiq_decision::service::DEFAULT_ROUTE_INSTRUCTIONS;

const FLAG_LOW_CONFIDENCE: &str = "low_confidence";
const FLAG_ORACLE_UNAVAILABLE: &str = "oracle_unavailable";

/// `served_by` of router responses: `cortiq/<version>`.
pub fn served_by() -> String {
    format!("cortiq/{}", env!("CARGO_PKG_VERSION"))
}

/// A router request id: `req_<unix nanos hex><counter hex>` (router `ids.rs`).
pub fn router_request_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("req_{nanos:016x}{n:06x}")
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn ms(d: Duration) -> f32 {
    (d.as_secs_f64() * 1000.0) as f32
}

// ------------------------------------------------------------------ surfaces

/// Which error envelope and request id a path uses (see the module notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// The decisions protocol and this server's own admin paths: OpenRouter
    /// errors, `cmf-dec-…` ids.
    Decisions,
    /// The router API (its admin keys paths included): router errors, `req_…`
    /// ids.
    Router,
}

/// Whether a router-surface path is checked by the router's key middleware
/// (router `api.rs:325-331`: probes and the admin API are not).
fn router_path_keyed(path: &str) -> bool {
    !(path == "/v1/healthz"
        || path == "/v1/readyz"
        || path == "/metrics"
        || path.starts_with("/v1/admin"))
}

/// `x-cmf-extensions` is set to a true value.
fn wants_extensions(headers: &HeaderMap) -> bool {
    headers
        .get(CMF_EXTENSIONS_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

/// The surface of a request path: the router's own paths (router
/// `api.rs:292-307`, anything under `/v1/taxonomies/`, and `/v1/admin/keys`
/// with at most one segment after it) are the router surface; everything else,
/// this server's own admin paths among them (`/v1/admin/keys/hash/…`,
/// `/v1/admin/usage`, `/oracle`, `/learning`, `/generations`, `/rollback`,
/// `/shadow`), is the decisions surface with the errors of §4.8.
pub fn surface_of(path: &str) -> Surface {
    let router = matches!(
        path,
        "/v1/route"
            | "/v1/route:batch"
            | "/v1/feedback"
            | "/v1/taxonomies"
            | "/v1/usage"
            | "/v1/escalations"
            | "/v1/healthz"
            | "/v1/readyz"
            | "/metrics"
            | "/v1/admin/keys"
    ) || path.starts_with("/v1/taxonomies/")
        || path
            .strip_prefix("/v1/admin/keys/")
            .is_some_and(|rest| !rest.contains('/'));
    if router {
        Surface::Router
    } else {
        Surface::Decisions
    }
}

/// Per-request context set by the middleware.
#[derive(Clone, Debug)]
struct Ctx {
    id: String,
    surface: Surface,
    /// `x-cmf-extensions` (router surface).
    ext: bool,
    /// The request path (only to tell keyed router paths; never logged).
    path: String,
}

/// The account a response was served to (for the request log).
#[derive(Clone, Debug)]
struct AccountTag(String);

// ------------------------------------------------------------------ errors

/// An error answer of the HTTP layer: a service [`ApiError`] or one of the
/// router's own codes.
#[derive(Clone, Debug)]
pub struct HttpError {
    pub status: u16,
    /// Reason code (`INVALID_REQUEST`, `TAXONOMY_NOT_FOUND`, …).
    pub code: &'static str,
    pub message: String,
    pub details: Option<Box<Map<String, Value>>>,
    pub retry_after: Option<u64>,
    /// Answer `message` as the whole `text/plain` body (the router's axum
    /// extractor rejections), or an empty body when `message` is empty.
    pub plain: bool,
    account: Option<String>,
}

impl From<ApiError> for HttpError {
    fn from(e: ApiError) -> Self {
        Self {
            status: e.status,
            code: e.reason.code(),
            message: e.message,
            details: e.details,
            retry_after: e.retry_after,
            plain: false,
            account: None,
        }
    }
}

impl HttpError {
    pub fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            details: None,
            retry_after: None,
            plain: false,
            account: None,
        }
    }

    /// A router-surface answer in axum's `text/plain` form (`body` empty: no
    /// body and no content type, like axum's own 404/405).
    pub fn plain(status: u16, body: impl Into<String>) -> Self {
        let code = match status {
            413 => Reason::PayloadTooLarge.code(),
            _ => Reason::InvalidRequest.code(),
        };
        let mut e = Self::new(status, code, body);
        e.plain = true;
        e
    }

    fn reason(reason: Reason, message: impl Into<String>) -> Self {
        Self::new(reason.status(), reason.code(), message)
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::reason(Reason::InvalidRequest, message)
    }

    fn internal() -> Self {
        Self::reason(Reason::Internal, "internal error")
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self::new(404, Reason::InvalidRequest.code(), message)
    }

    fn with_detail(mut self, key: &str, value: Value) -> Self {
        self.details
            .get_or_insert_with(Default::default)
            .insert(key.to_string(), value);
        self
    }

    fn details_json(&self) -> Value {
        Value::Object(self.details.as_deref().cloned().unwrap_or_default())
    }

    fn by(mut self, account: &str) -> Self {
        self.account = Some(account.to_string());
        self
    }

    /// Worth retrying unchanged on the decisions surface: 429, 500, 502.
    pub fn retriable(&self) -> bool {
        matches!(self.status, 429 | 500 | 502)
    }

    /// The router's rule: only 429 and 500 are retriable (router
    /// `api.rs:221-225`, `ApiError::new`).
    pub fn router_retriable(&self) -> bool {
        matches!(self.status, 429 | 500)
    }

    fn metadata(&self, request_id: &str, retriable: bool) -> Value {
        json!({
            "reason": self.code,
            "retriable": retriable,
            "request_id": request_id,
            "details": self.details_json(),
        })
    }

    /// OpenRouter's error body (spec §4.8).
    pub fn openrouter_body(&self, request_id: &str) -> Value {
        json!({
            "error": {
                "code": self.status,
                "message": self.message,
                "metadata": self.metadata(request_id, self.retriable()),
            }
        })
    }

    /// The message of the router's envelope: the router's own text where it
    /// has one for the same condition (router `api.rs:254-267`, `:346`,
    /// `:413-417`) — the rate window, the decision quota, a key that is not
    /// valid now (the router does not tell an expired or revoked key from an
    /// unknown one) and a disabled admin API — else this server's message
    /// (a token or credit quota, which the router does not have, among others).
    pub fn router_message(&self) -> &str {
        let detail = |k: &str| {
            self.details
                .as_deref()
                .and_then(|d| d.get(k))
                .and_then(Value::as_str)
        };
        match (self.status, self.code) {
            (429, "RATE_LIMITED") => ROUTER_RATE_LIMITED_MESSAGE,
            (402, "QUOTA_EXCEEDED") if detail("quota") == Some("decision") => {
                ROUTER_QUOTA_EXCEEDED_MESSAGE
            }
            (401, "UNAUTHORIZED")
                if self.message == AuthFailure::Expired.to_string()
                    || self.message == AuthFailure::Revoked.to_string() =>
            {
                ROUTER_INVALID_KEY_MESSAGE
            }
            (404, "ADMIN_DISABLED") => ROUTER_ADMIN_DISABLED_MESSAGE,
            _ => &self.message,
        }
    }

    /// The router's error envelope (router `api.rs:270-284`): `details` is
    /// `null` except for `TAXONOMY_NOT_FOUND` (router `api.rs:226,242`), the
    /// message and `retriable` are the router's ([`HttpError::router_message`],
    /// [`HttpError::router_retriable`]); `extensions` adds the decisions
    /// surface's `metadata` (this server's message in `metadata.message`).
    pub fn router_body(&self, request_id: &str, extensions: bool) -> Value {
        let details = if self.code == TAXONOMY_NOT_FOUND {
            self.details_json()
        } else {
            Value::Null
        };
        let message = self.router_message();
        let mut error = json!({
            "code": self.code,
            "message": message,
            "retriable": self.router_retriable(),
            "details": details,
        });
        if extensions {
            let mut m = self.metadata(request_id, self.router_retriable());
            if message != self.message {
                m["message"] = json!(self.message);
            }
            error["metadata"] = m;
        }
        json!({
            "schema_version": ROUTER_SCHEMA_VERSION,
            "request_id": request_id,
            "error": error,
        })
    }
}

fn internal_error(what: &str, e: impl std::fmt::Display) -> HttpError {
    tracing::error!(error = %e, "{what}");
    HttpError::internal()
}

/// The router's 404 code for an unknown `taxonomy_id`.
pub const TAXONOMY_NOT_FOUND: &str = "TAXONOMY_NOT_FOUND";
/// The router's 429 message (`api.rs:254-260`, `ApiError::rate_limited`).
pub const ROUTER_RATE_LIMITED_MESSAGE: &str = "per-account rate limit exceeded";
/// The router's 402 message (`api.rs:261-267`, `ApiError::quota_exceeded`).
pub const ROUTER_QUOTA_EXCEEDED_MESSAGE: &str = "account decision quota exhausted";
/// The router's 401 message for a key it does not accept (`api.rs:346`: an
/// unknown, expired or revoked key alike).
pub const ROUTER_INVALID_KEY_MESSAGE: &str = "invalid API key";
/// The router's 404 `ADMIN_DISABLED` message (`api.rs:413-417`).
pub const ROUTER_ADMIN_DISABLED_MESSAGE: &str =
    "admin API is disabled (no auth.admin_token configured)";

fn taxonomy_not_found(id: &str) -> HttpError {
    HttpError::new(
        404,
        TAXONOMY_NOT_FOUND,
        format!("taxonomy_id '{id}' not found for account"),
    )
    .with_detail("taxonomy_id", json!(id))
}

/// A path parameter axum could not extract (e.g. `%FF`, invalid UTF-8): a JSON
/// 400 `INVALID_REQUEST` in the envelope of the path's surface, never axum's
/// plain-text rejection.
fn path_rejected(r: PathRejection) -> HttpError {
    HttpError::invalid(format!("invalid path parameter: {}", r.body_text()))
}

fn embedding_required() -> HttpError {
    HttpError::new(
        400,
        "EMBEDDING_REQUIRED",
        "neither input.text nor input.embedding was provided",
    )
}

// ------------------------------------------------------------------ replies

/// A successful answer.
struct Reply {
    status: StatusCode,
    body: Value,
    /// `x-request-id` when it is not the request's own id (a decision).
    id: Option<String>,
    account: Option<String>,
}

impl Reply {
    fn ok(body: Value) -> Self {
        Self {
            status: StatusCode::OK,
            body,
            id: None,
            account: None,
        }
    }

    fn by(mut self, account: &str) -> Self {
        self.account = Some(account.to_string());
        self
    }
}

type Handled = Result<Reply, HttpError>;

fn json_response(
    status: StatusCode,
    body: &Value,
    request_id: &str,
    account: Option<String>,
    retry_after: Option<u64>,
) -> Response {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    let mut resp = Response::new(Body::from(bytes));
    *resp.status_mut() = status;
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    if let Ok(v) = HeaderValue::from_str(request_id) {
        h.insert(HeaderName::from_static(REQUEST_ID_HEADER), v);
    }
    if let Some(s) = retry_after {
        h.insert(header::RETRY_AFTER, HeaderValue::from(s));
    }
    if let Some(a) = account {
        resp.extensions_mut().insert(AccountTag(a));
    }
    resp
}

/// A `text/plain; charset=utf-8` answer (axum's rejection form); an empty
/// `text` is an empty body without a content type.
fn plain_response(
    status: StatusCode,
    text: String,
    request_id: &str,
    account: Option<String>,
) -> Response {
    let empty = text.is_empty();
    let mut resp = Response::new(Body::from(text));
    *resp.status_mut() = status;
    let h = resp.headers_mut();
    if !empty {
        h.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; charset=utf-8"),
        );
    }
    if let Ok(v) = HeaderValue::from_str(request_id) {
        h.insert(HeaderName::from_static(REQUEST_ID_HEADER), v);
    }
    if let Some(a) = account {
        resp.extensions_mut().insert(AccountTag(a));
    }
    resp
}

fn error_response(ctx: &Ctx, e: HttpError) -> Response {
    let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    if e.plain {
        return plain_response(status, e.message, &ctx.id, e.account);
    }
    let body = match ctx.surface {
        Surface::Decisions => e.openrouter_body(&ctx.id),
        Surface::Router => e.router_body(&ctx.id, ctx.ext),
    };
    json_response(status, &body, &ctx.id, e.account, e.retry_after)
}

fn respond(ctx: &Ctx, out: Handled) -> Response {
    match out {
        Ok(r) => json_response(
            r.status,
            &r.body,
            r.id.as_deref().unwrap_or(&ctx.id),
            r.account,
            None,
        ),
        Err(e) => error_response(ctx, e),
    }
}

// ------------------------------------------------------------------ request plumbing

/// Run blocking work on the blocking pool.
async fn blocking<T, F>(f: F) -> Result<T, HttpError>
where
    F: FnOnce() -> Result<T, HttpError> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(r) => r,
        Err(e) => Err(internal_error("decision worker task failed", e)),
    }
}

fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// `Content-Type: application/json` (parameters such as `charset` allowed).
fn require_json(headers: &HeaderMap) -> Result<(), HttpError> {
    let ct = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let mime = ct.split(';').next().unwrap_or("").trim();
    if mime.eq_ignore_ascii_case("application/json") {
        Ok(())
    } else {
        Err(HttpError::invalid("Content-Type must be application/json"))
    }
}

fn too_large(limit: usize) -> HttpError {
    HttpError::reason(
        Reason::PayloadTooLarge,
        format!("request body is larger than {limit} bytes"),
    )
    .with_detail("limit", json!(limit))
}

/// The body, at most `limit` bytes (413 from `Content-Length` before reading,
/// else as soon as the stream passes the limit).
async fn read_body(headers: &HeaderMap, body: Body, limit: usize) -> Result<Vec<u8>, HttpError> {
    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    if declared.is_some_and(|n| n > limit as u64) {
        return Err(too_large(limit));
    }
    let mut stream = body.into_data_stream();
    let mut buf = Vec::with_capacity(declared.map_or(0, |n| n as usize));
    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|e| HttpError::invalid(format!("the request body could not be read: {e}")))?;
        if buf.len() + chunk.len() > limit {
            return Err(too_large(limit));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Authenticate (401) and admit (rate 429, quotas 402) a keyed request.
async fn caller(st: &Arc<DecisionState>, headers: &HeaderMap) -> Result<Principal, HttpError> {
    let auth = header_string(headers, header::AUTHORIZATION.as_str());
    let key = header_string(headers, "x-api-key");
    let s = Arc::clone(st);
    blocking(move || {
        let p = s.svc.authenticate(auth.as_deref(), key.as_deref())?;
        s.svc
            .admit(&p)
            .map_err(|e| HttpError::from(e).by(&p.account))?;
        Ok(p)
    })
    .await
}

/// The `x-admin-token` guard (404 `ADMIN_DISABLED`, 401).
fn admin_guard(st: &DecisionState, headers: &HeaderMap) -> Result<(), HttpError> {
    let token = header_string(headers, "x-admin-token");
    st.svc
        .admin_authorize(token.as_deref())
        .map_err(HttpError::from)
}

/// Keyed JSON body: auth, admission, content type, body.
async fn keyed_body(
    st: &Arc<DecisionState>,
    headers: &HeaderMap,
    body: Body,
) -> Result<(Principal, Vec<u8>), HttpError> {
    let p = caller(st, headers).await?;
    require_json(headers).map_err(|e| e.by(&p.account))?;
    let bytes = read_body(headers, body, st.body_limit())
        .await
        .map_err(|e| e.by(&p.account))?;
    Ok((p, bytes))
}

/// A JSON media type as axum's `Json` extractor sees it: `application/json`
/// or `application/*+json`, parameters allowed (axum `json.rs`
/// `json_content_type`).
fn json_media_type(headers: &HeaderMap) -> bool {
    let Some(ct) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let essence = ct.split(';').next().unwrap_or("").trim();
    let Some((ty, sub)) = essence.split_once('/') else {
        return false;
    };
    let sub = sub.to_ascii_lowercase();
    let valid = |t: &str| {
        !t.is_empty()
            && t.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$&-^_.+".contains(&b))
    };
    valid(ty)
        && valid(&sub)
        && ty.eq_ignore_ascii_case("application")
        && (sub == "json" || sub.ends_with("+json"))
}

/// The router's body extraction (its axum `Json` extractor): 415 without a
/// JSON media type, 413 over `limit`, as `text/plain`.
async fn router_body_bytes(
    headers: &HeaderMap,
    body: Body,
    limit: usize,
) -> Result<Vec<u8>, HttpError> {
    if !json_media_type(headers) {
        return Err(HttpError::plain(415, ROUTER_MISSING_JSON_CONTENT_TYPE));
    }
    read_body(headers, body, limit).await.map_err(|e| {
        if e.status == 413 {
            HttpError::plain(413, ROUTER_LENGTH_LIMIT)
        } else {
            HttpError::plain(
                400,
                format!("Failed to buffer the request body: {}", e.message),
            )
        }
    })
}

/// Deserialize a router body as the router's axum 0.7 `Json` does: 400 for
/// bad JSON syntax, 422 for a body of the wrong shape, with axum's text; and,
/// as axum 0.7.9 `Json::from_bytes` (router `Cargo.lock`), whatever follows
/// the first JSON value is ignored — axum 0.8 refuses trailing characters, so
/// only the first value is handed to it (its errors, positions included, are
/// the same: 0.7.9 stops at the same byte).
fn router_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, HttpError> {
    let mut first =
        serde_json::Deserializer::from_slice(bytes).into_iter::<serde::de::IgnoredAny>();
    let value = match first.next() {
        Some(Ok(_)) => &bytes[..first.byte_offset()],
        // No complete first value: the whole body gives axum's error.
        _ => bytes,
    };
    axum::Json::<T>::from_bytes(value)
        .map(|j| j.0)
        .map_err(|r| HttpError::plain(r.status().as_u16(), r.body_text()))
}

/// Keyed router body: auth and admission (the router's middleware), then its
/// `Json` extractor.
async fn keyed_router_body<T: serde::de::DeserializeOwned>(
    st: &Arc<DecisionState>,
    headers: &HeaderMap,
    body: Body,
) -> Result<(Principal, T), HttpError> {
    let p = caller(st, headers).await?;
    let bytes = router_body_bytes(headers, body, st.body_limit())
        .await
        .map_err(|e| e.by(&p.account))?;
    let v = router_json(&bytes).map_err(|e| e.by(&p.account))?;
    Ok((p, v))
}

/// Admin JSON body: guard, content type, body.
async fn admin_body(
    st: &Arc<DecisionState>,
    headers: &HeaderMap,
    body: Body,
) -> Result<Vec<u8>, HttpError> {
    admin_guard(st, headers)?;
    require_json(headers)?;
    read_body(headers, body, st.body_limit()).await
}

// ------------------------------------------------------------------ state

/// Process counters of the router's `/metrics` and `/v1/escalations`
/// (router `state.rs:47-58`), since the server started.
#[derive(Debug, Default)]
struct Counters {
    decisions: AtomicU64,
    escalations: AtomicU64,
    oracle_calls: AtomicU64,
    cache_hits: AtomicU64,
    oracle_unavailable: AtomicU64,
    novelty_hits: AtomicU64,
}

impl Counters {
    fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }
}

/// One escalated question (router `AuditRecord`, `state.rs:118-132`), with
/// the account it belongs to (not shown) and the question id (shown under
/// `cmf` with `x-cmf-extensions`).
#[derive(Clone, Debug)]
struct AuditEntry {
    account: String,
    question: String,
    record: Value,
}

/// The shared state of the decision router.
pub struct DecisionState {
    svc: Arc<DecisionService>,
    cascade: Option<Arc<Cascade>>,
    rates: Rates,
    counters: Counters,
    novelty_by_account: Mutex<HashMap<String, u64>>,
    audit: Mutex<VecDeque<AuditEntry>>,
    /// Router request id → the decision id its feedback names.
    links: Mutex<VecDeque<(String, String)>>,
    links_cap: usize,
    /// `--shadow-of`: the old router the router API is forwarded to.
    shadow: Option<Arc<shadow::Shadow>>,
}

impl std::fmt::Debug for DecisionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionState")
            .field("service", &self.svc)
            .field("cascade", &self.cascade.is_some())
            .field("shadow", &self.shadow)
            .finish()
    }
}

impl DecisionState {
    /// The state over a service; `cascade` is the escalator the service was
    /// opened with (its learning statistics feed `/metrics` and `/v1/usage`).
    pub fn new(svc: Arc<DecisionService>, cascade: Option<Arc<Cascade>>) -> Result<Arc<Self>> {
        Self::with_shadow(svc, cascade, None)
    }

    fn with_shadow(
        svc: Arc<DecisionService>,
        cascade: Option<Arc<Cascade>>,
        shadow: Option<Arc<shadow::Shadow>>,
    ) -> Result<Arc<Self>> {
        let cfg = svc.config();
        let rates = cfg.rates()?;
        let links_cap = cfg.feedback.pending_cap;
        Ok(Arc::new(Self {
            svc,
            cascade,
            rates,
            counters: Counters::default(),
            novelty_by_account: Mutex::new(HashMap::new()),
            audit: Mutex::new(VecDeque::new()),
            links: Mutex::new(VecDeque::new()),
            links_cap,
            shadow,
        }))
    }

    pub fn service(&self) -> &Arc<DecisionService> {
        &self.svc
    }

    pub fn cascade(&self) -> Option<&Arc<Cascade>> {
        self.cascade.as_ref()
    }

    fn config(&self) -> &Config {
        self.svc.config()
    }

    fn body_limit(&self) -> usize {
        self.config().limits.body_bytes
    }

    /// (refit attempts, promotions) of the cascade.
    fn learning_counts(&self) -> (u64, u64) {
        match &self.cascade {
            Some(c) => {
                let l = c.learning_json();
                (
                    l["attempts"].as_u64().unwrap_or(0),
                    l["promotions"].as_u64().unwrap_or(0),
                )
            }
            None => (0, 0),
        }
    }

    fn link(&self, route_id: &str, decision_id: &str) {
        let mut l = lock(&self.links);
        while l.len() >= self.links_cap {
            l.pop_front();
        }
        l.push_back((route_id.to_string(), decision_id.to_string()));
    }

    fn linked(&self, route_id: &str) -> Option<String> {
        lock(&self.links)
            .iter()
            .find(|(r, _)| r == route_id)
            .map(|(_, d)| d.clone())
    }

    fn unlink(&self, route_id: &str) {
        lock(&self.links).retain(|(r, _)| r != route_id);
    }

    fn push_audit(&self, entries: Vec<AuditEntry>) {
        if entries.is_empty() {
            return;
        }
        let mut a = lock(&self.audit);
        for e in entries {
            if a.len() >= AUDIT_CAP {
                a.pop_front();
            }
            a.push_back(e);
        }
    }

    fn count_novel(&self, account: &str, n: u64) {
        if n > 0 {
            *lock(&self.novelty_by_account)
                .entry(account.to_string())
                .or_default() += n;
        }
    }

    /// Counters and escalation records of a decided request. `audit_id` is the
    /// id the records carry (the router id of a route, else the decision id).
    fn note(&self, d: &Decided, account: &str, audit_id: &str) {
        let c = &self.counters;
        let mut escalated = 0u64;
        let mut unavailable = 0u64;
        let mut novel = 0u64;
        let mut calls_left = d.metered.oracle.calls;
        let mut entries = Vec::new();
        for q in &d.questions {
            if q.local.as_ref().is_some_and(LocalDecision::is_novel) {
                novel += 1;
            }
            let failed = q.flags.iter().any(|f| f == FLAG_ORACLE_UNAVAILABLE);
            if failed {
                unavailable += 1;
            }
            if q.action == Action::Local {
                continue;
            }
            escalated += 1;
            let calls = if q.action == Action::Oracle {
                std::mem::take(&mut calls_left)
            } else {
                0
            };
            let router_label = q
                .local
                .as_ref()
                .and_then(|l| l.choice.clone())
                .unwrap_or_default();
            let final_label = match &q.oracle {
                Some(a) => match a {
                    cortiq_decision::answer::OracleAnswer::Choice(l) => l.clone(),
                    cortiq_decision::answer::OracleAnswer::Score(s) => s.to_string(),
                    cortiq_decision::answer::OracleAnswer::Noul(b) => b.to_string(),
                },
                None => router_label.clone(),
            };
            let consulted_or_cached = matches!(q.action, Action::Oracle | Action::Cache);
            let agreement = match (&q.oracle, &q.local) {
                (Some(a), Some(l)) if consulted_or_cached => {
                    Some(a.label().is_some() && a.label() == l.choice.as_deref())
                }
                _ => None,
            };
            let oracle_model =
                (consulted_or_cached || failed).then(|| self.config().oracle.model.clone());
            let latency =
                (q.action == Action::Oracle || failed).then(|| f32_json(ms(d.timings.oracle)));
            // The router's record carries the response's flags (`api.rs:902`):
            // an unresolved escalation is `low_confidence` there too.
            let flags = audit_flags(q.action, &q.flags);
            entries.push(AuditEntry {
                account: account.to_string(),
                question: q.id.clone(),
                record: json!({
                    "request_id": audit_id,
                    "ts": d.created,
                    "taxonomy_id": q.matched.skill.clone().unwrap_or_default(),
                    "source": q.action.source().replace("local", "router"),
                    "router_label": router_label,
                    "final_label": final_label,
                    "agreement_with_router": agreement,
                    "oracle_model": oracle_model,
                    "oracle_latency_ms": latency,
                    "oracle_calls": calls,
                    "novelty_score": f32_json(q.local.as_ref().map_or(1.0, |l| l.decision.novelty)),
                    "flags": flags,
                }),
            });
        }
        c.decisions
            .fetch_add(d.questions.len() as u64, Ordering::Relaxed);
        c.escalations.fetch_add(escalated, Ordering::Relaxed);
        c.oracle_calls
            .fetch_add(d.metered.oracle.calls, Ordering::Relaxed);
        c.cache_hits
            .fetch_add(d.record.actions.cache, Ordering::Relaxed);
        c.oracle_unavailable
            .fetch_add(unavailable, Ordering::Relaxed);
        c.novelty_hits.fetch_add(novel, Ordering::Relaxed);
        self.count_novel(account, novel);
        self.push_audit(entries);
    }

    // ---------------------------------------------------------------- JSON views

    /// `GET /v1/usage`: the router's shape (`api.rs:1337-1363`) with the
    /// caller's own numbers; `extensions` adds the decisions service's totals
    /// and limits under `cmf`.
    fn usage_json(&self, p: &Principal, extensions: bool) -> Value {
        let t = self.svc.totals(&p.account);
        let escalations = t.decisions.saturating_sub(t.actions.local);
        let rate = escalations as f32 / t.decisions.max(1) as f32;
        let novelty = lock(&self.novelty_by_account)
            .get(&p.account)
            .copied()
            .unwrap_or(0);
        let (refits, promotions) = self.learning_counts();
        let mut v = json!({
            "schema_version": ROUTER_SCHEMA_VERSION,
            "account": {
                "id": p.account,
                "billable_decisions": t.decisions,
                "oracle_calls": t.oracle_calls,
                "decision_quota": p.decision_quota,
                "rate_per_min": p.rate_per_min,
            },
            "usage": {
                "billable_decisions": t.decisions,
                "oracle_calls": t.oracle_calls,
                "escalations": escalations,
                "escalation_rate": f32_json(rate),
                "cache_hits": t.cache_hits,
                "novelty_hits": novelty,
                "refits": refits,
                "promotions": promotions,
            },
        });
        if extensions {
            let base = self.svc.usage_json(p);
            v["cmf"] = json!({
                "plan": base["plan"],
                "totals": base["usage"],
                "limits": base["limits"],
            });
        }
        v
    }

    /// The skills in router-listing order: `default_skill` first.
    fn ordered_skills<'m>(&self, model: &'m LoadedModel) -> Vec<&'m SkillRuntime> {
        let mut v: Vec<&SkillRuntime> = model.skills().iter().collect();
        if let Some(d) = &self.config().default_skill
            && let Some(pos) = v.iter().position(|s| s.id() == d)
        {
            let s = v.remove(pos);
            v.insert(0, s);
        }
        v
    }

    fn metrics_text(&self) -> String {
        let c = &self.counters;
        let decisions = Counters::get(&c.decisions);
        let escalations = Counters::get(&c.escalations);
        let oracle_calls = Counters::get(&c.oracle_calls);
        let (refits, promotions) = self.learning_counts();
        let (buffered, cache_entries, cache_hits, cache_lookups) = match &self.cascade {
            Some(cas) => {
                let l = cas.learning_json();
                (
                    cas.buffer_len() as u64,
                    l["cache"]["entries"].as_u64().unwrap_or(0),
                    l["cache"]["hits"].as_u64().unwrap_or(0),
                    l["cache"]["lookups"].as_u64().unwrap_or(0),
                )
            }
            None => (0, 0, 0, 0),
        };
        let active: usize = self
            .svc
            .handle()
            .current()
            .skills()
            .iter()
            .map(|s| s.scorer().len())
            .sum();
        let d = decisions.max(1) as f32;
        let mut s = String::with_capacity(2048);
        let mut g = |name: &str, help: &str, typ: &str, val: String| {
            s.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} {typ}\n{name} {val}\n"
            ));
        };
        g(
            "cortiq_decisions_total",
            "Total billable decisions",
            "counter",
            decisions.to_string(),
        );
        g(
            "cortiq_escalations_total",
            "Decisions the policy escalated",
            "counter",
            escalations.to_string(),
        );
        g(
            "cortiq_oracle_calls_total",
            "Oracle consultations",
            "counter",
            oracle_calls.to_string(),
        );
        g(
            "cortiq_cache_hits_total",
            "Escalations served from the semantic cache",
            "counter",
            Counters::get(&c.cache_hits).to_string(),
        );
        g(
            "cortiq_oracle_unavailable_total",
            "Oracle failures/degradations",
            "counter",
            Counters::get(&c.oracle_unavailable).to_string(),
        );
        g(
            "cortiq_novelty_hits_total",
            "Inputs flagged novel/OOD",
            "counter",
            Counters::get(&c.novelty_hits).to_string(),
        );
        g(
            "cortiq_refits_total",
            "Self-learn task refits",
            "counter",
            refits.to_string(),
        );
        g(
            "cortiq_promotions_total",
            "Champion/challenger promotions",
            "counter",
            promotions.to_string(),
        );
        g(
            "cortiq_escalation_rate",
            "escalations / decisions",
            "gauge",
            format!("{:.6}", escalations as f32 / d),
        );
        g(
            "cortiq_oracle_call_rate",
            "oracle_calls / decisions",
            "gauge",
            format!("{:.6}", oracle_calls as f32 / d),
        );
        g(
            "cortiq_labeled_examples",
            "Examples in the labeled buffer",
            "gauge",
            buffered.to_string(),
        );
        g(
            "cortiq_oracle_cache_entries",
            "Semantic cache entries",
            "gauge",
            cache_entries.to_string(),
        );
        g(
            "cortiq_oracle_cache_hits_total",
            "Semantic cache hits",
            "counter",
            cache_hits.to_string(),
        );
        g(
            "cortiq_oracle_cache_lookups_total",
            "Semantic cache lookups",
            "counter",
            cache_lookups.to_string(),
        );
        g(
            "cortiq_active_tasks",
            "Active tasks in the taxonomy",
            "gauge",
            active.to_string(),
        );
        s
    }
}

/// The router's `TaxonomySummary` (`api.rs:1301-1335`); `extensions` adds
/// the generation and whether the skill's gate is certified under `cmf`.
fn taxonomy_summary(model: &LoadedModel, s: &SkillRuntime, extensions: bool) -> Value {
    let mut v = json!({
        "taxonomy_id": s.id(),
        "taxonomy_version": format!("{}@{}", s.id(), s.manifest().taxonomy_version),
        "model_version": model.name(),
        "labels": s.scorer().labels(),
    });
    if extensions {
        v["cmf"] = json!({
            "generation": model.generation(),
            "certified": s.gate().certified,
        });
    }
    v
}

// ------------------------------------------------------------------ router

/// The decision router over `state` (see the module notes). It carries its
/// own request-id and logging middleware and no CORS layer.
pub fn router(state: Arc<DecisionState>) -> Router {
    let mut r = Router::new()
        // Decisions protocol (spec §4.3).
        .route("/api/alpha/decisions", post(decisions_handler))
        .route("/v1/decisions", post(decisions_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/skills", get(skills_handler))
        .route("/v1/skills/{id}", get(skill_handler))
        .route("/healthz", get(healthz_handler))
        // Router API (spec §4.15).
        .route("/v1/route", post(route_handler))
        .route("/v1/route:batch", post(batch_handler))
        .route("/v1/feedback", post(feedback_handler))
        .route("/v1/taxonomies", get(taxonomies_handler))
        .route("/v1/taxonomies/{id}", get(taxonomy_handler))
        .route("/v1/usage", get(usage_handler))
        .route("/v1/escalations", get(escalations_handler))
        .route("/v1/healthz", get(router_healthz_handler))
        .route("/v1/readyz", get(readyz_handler))
        .route("/metrics", get(metrics_handler))
        // Admin (spec §5b; router `api.rs:299-304`).
        .route(
            "/v1/admin/keys",
            post(admin_create_key).get(admin_list_keys),
        )
        .route("/v1/admin/keys/{account}", delete(admin_revoke_account))
        .route("/v1/admin/keys/hash/{hash}", delete(admin_revoke_hash))
        .route("/v1/admin/usage", get(admin_usage))
        .route(
            "/v1/admin/oracle",
            get(admin_oracle).post(admin_oracle_update),
        )
        .route("/v1/admin/learning", get(admin_learning))
        .route("/v1/admin/generations", get(admin_generations))
        .route("/v1/admin/rollback", post(admin_rollback));
    // Shadow mode (`--shadow-of`): only then is anything added, so a server
    // without the flag is unchanged.
    let shadow = state.shadow.is_some();
    if shadow {
        r = r.route("/v1/admin/shadow", get(shadow::admin_shadow));
    }
    let app = r
        .fallback(not_found_handler)
        .method_not_allowed_fallback(method_not_allowed_handler)
        .layer(middleware::from_fn(context_middleware))
        .with_state(Arc::clone(&state));
    if !shadow {
        return app;
    }
    // The router API is forwarded before any routing happens here, so that
    // no route of this server touches the old router's answer (axum adds
    // `Allow` to what a route answers for a method it does not have).
    Router::new()
        .fallback_service(app)
        .layer(middleware::from_fn_with_state(
            state,
            shadow::shadow_middleware,
        ))
}

/// Request id, `x-request-id` on every response, one log line per request
/// with exactly the fields of spec §4.3: id, status, latency and account
/// (never a method, path, query, body, header or key).
async fn context_middleware(mut req: Request, next: Next) -> Response {
    let t0 = Instant::now();
    let path = req.uri().path().to_string();
    let surface = surface_of(&path);
    let id = match surface {
        Surface::Decisions => new_request_id(now_unix()),
        Surface::Router => router_request_id(),
    };
    let ext = wants_extensions(req.headers());
    req.extensions_mut().insert(Ctx {
        id: id.clone(),
        surface,
        ext,
        path,
    });
    let mut resp = next.run(req).await;
    let hdr = HeaderName::from_static(REQUEST_ID_HEADER);
    if !resp.headers().contains_key(&hdr)
        && let Ok(v) = HeaderValue::from_str(&id)
    {
        resp.headers_mut().insert(hdr.clone(), v);
    }
    let shown = resp
        .headers()
        .get(&hdr)
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&id)
        .to_string();
    // An account is any text the router's column holds: control characters
    // escaped, so that a line stays one line.
    let account = resp
        .extensions()
        .get::<AccountTag>()
        .map_or_else(|| "-".to_string(), |a| cortiq_decision::keys::shown(&a.0));
    tracing::info!(
        id = %shown,
        status = resp.status().as_u16(),
        latency_ms = t0.elapsed().as_secs_f64() * 1000.0,
        account = %account,
        "decision request"
    );
    resp
}

/// The router's answer to a path or method it does not route: its key
/// middleware first on a keyed path (401/429/402), then axum's empty 404/405
/// (axum adds `Allow` to a 405).
async fn router_unrouted(
    st: &Arc<DecisionState>,
    ctx: &Ctx,
    headers: &HeaderMap,
    status: u16,
) -> Response {
    let mut account = None;
    if router_path_keyed(&ctx.path) {
        match caller(st, headers).await {
            Ok(p) => account = Some(p.account),
            Err(e) => return error_response(ctx, e),
        }
    }
    let mut e = HttpError::plain(status, "");
    e.account = account;
    error_response(ctx, e)
}

async fn not_found_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    match ctx.surface {
        Surface::Router => router_unrouted(&st, &ctx, &headers, 404).await,
        Surface::Decisions => error_response(&ctx, HttpError::not_found("no such endpoint")),
    }
}

async fn method_not_allowed_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    match ctx.surface {
        Surface::Router => router_unrouted(&st, &ctx, &headers, 405).await,
        Surface::Decisions => {
            let mut e = HttpError::invalid("method not allowed on this endpoint");
            e.status = 405;
            error_response(&ctx, e)
        }
    }
}

// ------------------------------------------------------------------ decisions

async fn decisions_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let out = decide_http(&st, &headers, body).await;
    respond(&ctx, out)
}

async fn decide_http(st: &Arc<DecisionState>, headers: &HeaderMap, body: Body) -> Handled {
    let (p, bytes) = keyed_body(st, headers, body).await?;
    let account = p.account.clone();
    let guard = st
        .svc
        .enter()
        .map_err(|e| HttpError::from(e).by(&account))?;
    let s = Arc::clone(st);
    let decided = blocking(move || {
        let _slot = guard;
        s.svc
            .decide_body(&bytes, &p)
            .map_err(|e| HttpError::from(e).by(&p.account))
    })
    .await?;
    st.note(&decided, &account, &decided.id);
    Ok(Reply {
        status: StatusCode::OK,
        body: decided.response,
        id: Some(decided.id),
        account: Some(account),
    })
}

async fn models_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
) -> Response {
    respond(&ctx, Ok(Reply::ok(st.svc.models_json())))
}

async fn healthz_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
) -> Response {
    respond(&ctx, Ok(Reply::ok(st.svc.healthz_json())))
}

/// `GET /v1/healthz`: the router's `{"status":"ok"}` (`api.rs:1397-1399`);
/// `x-cmf-extensions` adds `/healthz`'s fields under `cmf`.
async fn router_healthz_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
) -> Response {
    let mut v = json!({"status": "ok"});
    if ctx.ext {
        let mut h = st.svc.healthz_json();
        if let Some(m) = h.as_object_mut() {
            m.remove("status");
        }
        v["cmf"] = h;
    }
    respond(&ctx, Ok(Reply::ok(v)))
}

async fn skills_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        let p = caller(&st, &headers).await?;
        Ok(Reply::ok(st.svc.skills_json()).by(&p.account))
    }
    .await;
    respond(&ctx, out)
}

async fn skill_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    id: Result<Path<String>, PathRejection>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        let p = caller(&st, &headers).await?;
        let Path(id) = id.map_err(|r| path_rejected(r).by(&p.account))?;
        let v = st
            .svc
            .skill_json(&id)
            .map_err(|e| HttpError::from(e).by(&p.account))?;
        Ok(Reply::ok(v).by(&p.account))
    }
    .await;
    respond(&ctx, out)
}

// ------------------------------------------------------------------ router API: route

/// The router's request types (`api.rs:30-89, 374-393, 1365-1369`) under the
/// router's own names: serde's messages name the types, and a body the router
/// rejects is rejected here with the same text (see [`router_json`]). Unknown
/// keys are ignored, as in the router.
mod wire {
    use serde::Deserialize;

    /// `options.policy_profile` (router `policy.rs:11-17`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub enum PolicyProfile {
        CostSaver,
        Balanced,
        QualityFirst,
    }

    #[derive(Debug, Deserialize)]
    pub struct RouteRequest {
        #[serde(default)]
        pub taxonomy_id: Option<String>,
        pub input: Input,
        #[serde(default)]
        pub options: Options,
        #[serde(default)]
        pub client_request_id: Option<String>,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub struct Input {
        #[serde(default)]
        pub text: Option<String>,
        /// Bring-your-own signal; if present, `text` is ignored.
        #[serde(default)]
        pub embedding: Option<Vec<f32>>,
        #[serde(default)]
        pub embedding_model: Option<String>,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(default)]
    pub struct Options {
        pub policy_profile: PolicyProfile,
        pub allow_oracle: bool,
        pub allow_pii_egress: bool,
        pub top_k: usize,
        pub return_explanation: bool,
        pub routing_table_id: Option<String>,
    }

    /// Router `api.rs:63-74`.
    impl Default for Options {
        fn default() -> Self {
            Self {
                policy_profile: PolicyProfile::Balanced,
                allow_oracle: true,
                allow_pii_egress: false,
                top_k: super::ROUTER_DEFAULT_TOP_K,
                return_explanation: false,
                routing_table_id: None,
            }
        }
    }

    #[derive(Debug, Deserialize)]
    pub struct BatchRequest {
        #[serde(default)]
        pub taxonomy_id: Option<String>,
        pub inputs: Vec<Input>,
        #[serde(default)]
        pub options: Options,
    }

    #[derive(Debug, Deserialize)]
    pub struct FeedbackRequest {
        pub request_id: String,
        pub correct_task_label: String,
    }

    /// `POST /v1/admin/keys` (router `api.rs:374-393`) plus the decision-v4
    /// limits of spec §4.10.
    #[derive(Debug, Deserialize)]
    pub struct CreateKeyReq {
        #[serde(default)]
        pub plan: Option<String>,
        /// Accepted and not stored (no personal data in `keys.json`).
        #[serde(default)]
        #[allow(dead_code)]
        pub email: Option<String>,
        #[serde(default)]
        pub label: Option<String>,
        #[serde(default)]
        pub days: Option<u32>,
        #[serde(default)]
        pub rate_per_min: Option<u32>,
        #[serde(default)]
        pub decision_quota: Option<u64>,
        #[serde(default)]
        pub account: Option<String>,
        #[serde(default)]
        pub token_quota: Option<u64>,
        #[serde(default)]
        pub credit_usd: Option<String>,
        #[serde(default)]
        pub oracle_budget_usd: Option<String>,
        #[serde(default)]
        pub oracle_allowed: Option<bool>,
        #[serde(default)]
        pub learning_allowed: Option<bool>,
    }

    #[derive(Debug, Deserialize)]
    pub struct EscalationsQuery {
        #[serde(default)]
        pub limit: Option<usize>,
    }
}

/// The resolved options of a route.
#[derive(Clone, Debug)]
struct Routing {
    skill: String,
    profile: Profile,
    allow_oracle: bool,
    allow_pii_egress: bool,
    top_k: usize,
    explain: bool,
    routing_table: bool,
    /// `x-cmf-extensions`: add the `cmf` object.
    extensions: bool,
}

impl DecisionState {
    /// The skill of a router request: `taxonomy_id`, else `default_skill`, else
    /// the file's only skill.
    fn resolve_routing(
        &self,
        taxonomy_id: Option<&str>,
        opts: &wire::Options,
        extensions: bool,
    ) -> Result<Routing, HttpError> {
        let model = self.svc.handle().current();
        let skill = match taxonomy_id.or(self.config().default_skill.as_deref()) {
            Some(id) => {
                if model.skill(id).is_none() {
                    return Err(taxonomy_not_found(id));
                }
                id.to_string()
            }
            None => match model.skills() {
                [one] => one.id().to_string(),
                many => {
                    let ids: Vec<&str> = many.iter().map(SkillRuntime::id).collect();
                    return Err(HttpError::invalid(format!(
                        "taxonomy_id is required: this model has skills {} (or set default_skill in the configuration)",
                        ids.join(", ")
                    )));
                }
            },
        };
        let profile = match opts.policy_profile {
            wire::PolicyProfile::CostSaver => Profile::CostSaver,
            wire::PolicyProfile::Balanced => Profile::Balanced,
            wire::PolicyProfile::QualityFirst => Profile::QualityFirst,
        };
        Ok(Routing {
            skill,
            profile,
            allow_oracle: opts.allow_oracle,
            allow_pii_egress: opts.allow_pii_egress,
            top_k: opts.top_k.clamp(1, ROUTER_MAX_TOP_K),
            explain: opts.return_explanation,
            routing_table: opts.routing_table_id.is_some(),
            extensions,
        })
    }
}

/// The flags of a router response (and of its audit record): an escalated
/// question left unanswered is the router's `low_confidence`, then the
/// question's own flags — without `no_key` and `bad_key`, which only the
/// decisions surface names (a missing or unusable oracle key is
/// `oracle_disabled` there too, as before the flags existed), so the router
/// vocabulary stays what it was.
fn audit_flags(action: Action, flags: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(flags.len() + 1);
    if action == Action::Abstain {
        out.push(FLAG_LOW_CONFIDENCE.to_string());
    }
    out.extend(
        flags
            .iter()
            .filter(|f| {
                f.as_str() != RefusalReason::NoKey.flag()
                    && f.as_str() != RefusalReason::BadKey.flag()
            })
            .cloned(),
    );
    out
}

/// The longest prefix of `text` of at most `max` bytes that ends on a
/// character boundary.
fn truncate_utf8(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// The router question of a skill: `task`, choice over its active labels, with
/// the rubric's instructions and descriptions (question-file order, then any
/// active label the rubric does not describe, in task order).
pub fn route_question(s: &SkillRuntime) -> Question {
    s.rubric_question(ROUTE_QUESTION_ID)
}

/// The decision request of a router text input: the skill's `task`
/// question under the request's options.
fn route_request(r: &Routing, question: Question, text: &str) -> DecisionRequest {
    DecisionRequest {
        model: ModelRef::Latest,
        state: RequestState::Text(text.to_string()),
        state_text: text.to_string(),
        questions: vec![question],
        cmf: CmfOptions {
            skill: Some(r.skill.clone()),
            oracle: Some(r.allow_oracle),
            allow_pii_egress: r.allow_pii_egress,
            round: None,
            explain: r.explain,
            profile: r.profile,
        },
        user: None,
        session_id: None,
    }
}

/// One scored candidate of a router response.
#[derive(Clone, Debug)]
struct ScoreRow {
    task_id: usize,
    label: String,
    probability: f32,
    score: f32,
    error: f32,
}

/// What a router response is built from (text and embedding inputs alike).
struct RouteOutcome {
    action: Action,
    local: Option<LocalDecision>,
    oracle_label: Option<String>,
    flags: Vec<String>,
    decision_path: String,
    certified: bool,
    complexity: Value,
    routing: Option<Value>,
    oracle_calls: u64,
    oracle_time: Duration,
    embed_time: Duration,
    route_time: Duration,
    model_name: String,
    model_sha: String,
    generation: u64,
    decision_id: Option<String>,
    cmf_question: Value,
    cmf_usage: Value,
}

/// The router's `promote_oracle_choice` (`api.rs:1012-1023`).
fn promote(scores: &mut [ScoreRow], label: &str, confidence: f32) {
    let top = scores.iter().map(|s| s.score).fold(0.0f32, f32::max);
    if let Some(s) = scores.iter_mut().find(|s| s.label == label) {
        s.score = confidence.max(top);
        s.probability = confidence;
    }
    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn top1_vs_top2(scores: &[ScoreRow]) -> String {
    match scores {
        [a, b, ..] => format!(
            "{} leads {} by {:.3} score",
            a.label,
            b.label,
            a.score - b.score
        ),
        [a] => format!("{} only candidate", a.label),
        [] => "no candidates".into(),
    }
}

impl DecisionState {
    /// Decide one router input (text or embedding) and build its response.
    fn route_one(
        &self,
        p: &Principal,
        r: &Routing,
        input: &wire::Input,
        client_request_id: Option<&str>,
        request_id: &str,
    ) -> Result<Value, HttpError> {
        let model = self.svc.handle().current();
        let s = model
            .skill(&r.skill)
            .ok_or_else(|| taxonomy_not_found(&r.skill))?;
        if s.scorer().is_empty() {
            return Err(HttpError::invalid(format!(
                "taxonomy '{}' has no active task",
                r.skill
            )));
        }
        let taxonomy_version = s.manifest().taxonomy_version;
        let outcome = match (&input.embedding, &input.text) {
            (Some(x), _) => self.route_embedding(p, r, &model, s, input, x, request_id)?,
            (None, Some(text)) => self.route_text(p, r, s, text, request_id)?,
            (None, None) => return Err(embedding_required()),
        };
        Ok(self.route_json(r, &outcome, client_request_id, request_id, taxonomy_version))
    }

    fn route_text(
        &self,
        p: &Principal,
        r: &Routing,
        s: &SkillRuntime,
        text: &str,
        request_id: &str,
    ) -> Result<RouteOutcome, HttpError> {
        if text.trim().is_empty() {
            return Err(embedding_required());
        }
        // The router takes a text of any length; the decision reads its first
        // `limits.state_bytes` (whole characters).
        let text = truncate_utf8(text, self.config().limits.state_bytes);
        let req = route_request(r, route_question(s), text);
        let d = self.svc.decide(&req, p)?;
        self.note(&d, &p.account, request_id);
        self.link(request_id, &d.id);
        let q = d
            .questions
            .first()
            .ok_or_else(|| internal_error("route", "a decision without its question"))?;
        let cq = &d.response["cmf"]["questions"][ROUTE_QUESTION_ID];
        Ok(RouteOutcome {
            action: q.action,
            local: q.local.clone(),
            oracle_label: q
                .oracle
                .as_ref()
                .and_then(|a| a.label())
                .map(str::to_string),
            flags: q.flags.clone(),
            decision_path: q.decision_path.to_string(),
            certified: q.certified,
            complexity: cq["complexity"].clone(),
            routing: cq.get("routing").cloned(),
            oracle_calls: d.metered.oracle.calls,
            oracle_time: d.timings.oracle,
            embed_time: d.timings.tokenize + d.timings.encode + d.timings.hash,
            route_time: d.timings.resonance,
            model_name: d.response["model"].as_str().unwrap_or_default().to_string(),
            model_sha: d.response["cmf"]["model_sha"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            generation: d.response["cmf"]["generation"].as_u64().unwrap_or(0),
            decision_id: Some(d.id.clone()),
            cmf_question: cq.clone(),
            cmf_usage: d.response["cmf"]["usage"].clone(),
        })
    }

    /// A bring-your-own-embedding input: decided locally (see the module notes).
    #[allow(clippy::too_many_arguments)]
    fn route_embedding(
        &self,
        p: &Principal,
        r: &Routing,
        model: &Arc<LoadedModel>,
        s: &SkillRuntime,
        input: &wire::Input,
        x: &[f32],
        request_id: &str,
    ) -> Result<RouteOutcome, HttpError> {
        if input.embedding_model.is_none() {
            return Err(HttpError::invalid(
                "input.embedding_model is required when input.embedding is provided",
            ));
        }
        let dim = model.model().signal_dim();
        if x.len() != dim {
            return Err(HttpError::invalid(format!(
                "embedding has dim {} but model expects {dim} (the encoder's {} + the hashing contract's {})",
                x.len(),
                model.model().encoder_dim(),
                model.model().hashing_dim()
            ))
            .with_detail("expected_dim", json!(dim)));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(HttpError::invalid("input.embedding must be finite numbers"));
        }
        let t0 = Instant::now();
        let scorer = s.scorer();
        let gate = scorer.gate();
        let (errors, decision) = scorer
            .decide(x)
            .map_err(|e| internal_error("route embedding", e))?;
        let gate_accepted = decision.accepted(gate.tau, gate.novelty_theta);
        let accepted = match r.profile {
            Profile::Balanced => gate_accepted,
            Profile::QualityFirst => {
                gate_accepted
                    && decision.margin >= QUALITY_FIRST_MARGIN
                    && decision.novelty <= gate.novelty_theta.min(QUALITY_FIRST_NOVELTY)
            }
            Profile::CostSaver => {
                decision.winner.is_some() && !decision.is_novel(gate.novelty_theta)
            }
        };
        let labels = scorer.labels().to_vec();
        let choice = decision.winner.map(|w| labels[w].clone());
        let local = LocalDecision {
            skill: s.id().to_string(),
            tasks: scorer.tasks().to_vec(),
            errors,
            confidence: if decision.winner.is_some() {
                jev_confidence(decision.p_top, labels.len())
            } else {
                0.0
            },
            labels,
            choice: choice.clone(),
            gate_accepted,
            accepted,
            certified: false,
            gate,
            profile: r.profile,
            decision,
        };
        let route_time = t0.elapsed();
        let cfg = self.config();
        let consent = cfg.oracle.enabled
            && self.svc.escalator().is_some()
            && p.oracle_allowed
            && r.allow_oracle;
        let (action, flags, path) = if accepted {
            (Action::Local, Vec::new(), "router:uncertified")
        } else if consent {
            (
                Action::Abstain,
                vec![FLAG_ORACLE_UNAVAILABLE.to_string()],
                "escalate→oracle_unavailable",
            )
        } else {
            let why = if !cfg.oracle.enabled || self.svc.escalator().is_none() {
                // The router's answer has no hint: the log has it.
                if let Some(h) = self.svc.oracle_hint(RefusalReason::OracleDisabled) {
                    self.svc.log_oracle_hint(&h);
                }
                "oracle_disabled"
            } else {
                "consent_off"
            };
            (Action::Abstain, vec![why.to_string()], "escalate→disabled")
        };
        let label = choice.as_deref();
        let cx = estimate_complexity(
            cfg,
            label,
            local.decision.p_top,
            local.decision.novelty,
            local.decision.margin,
            0,
        );
        let routing = cfg.routing_tiers.get(&cx.tier).map(|target| {
            json!({
                "target": target,
                "reason": format!("{} @ complexity {:.2} ({})", label.unwrap_or("choice"), cx.score, cx.tier),
            })
        });
        // Usage: one decision, no input tokens (no text), the probabilities as
        // output tokens (spec §4.9).
        let out_tokens = local.labels.len() as u64;
        let cost = self
            .rates
            .cost(0, out_tokens, Usd::ZERO)
            .map_err(|e| internal_error("route embedding cost", e))?;
        let mut actions = Actions::default();
        if action == Action::Local {
            actions.local = 1;
        } else {
            actions.abstain = 1;
        }
        let record = UsageRecord {
            ts: now_unix(),
            id: request_id.to_string(),
            account: p.account.clone(),
            key12: p.key12.clone(),
            model: model.name(),
            generation: model.generation(),
            input_tokens: 0,
            output_tokens: out_tokens,
            cost_usd: cost.total.to_f64(),
            cost_local_usd: cost.local.to_f64(),
            cost_oracle_usd: 0.0,
            oracle_calls: 0,
            cache_hits: 0,
            questions: 1,
            actions,
        };
        match self.svc.ledger() {
            Some(l) => l
                .append(&record)
                .map_err(|e| internal_error("usage ledger", e))?,
            None => tracing::warn!(
                request = request_id,
                "no usage ledger: an embedding decision is not recorded"
            ),
        }
        // Counters, audit, and the feedback ring (vectors only).
        let novel = u64::from(local.is_novel());
        let c = &self.counters;
        c.decisions.fetch_add(1, Ordering::Relaxed);
        c.novelty_hits.fetch_add(novel, Ordering::Relaxed);
        self.count_novel(&p.account, novel);
        if action != Action::Local {
            c.escalations.fetch_add(1, Ordering::Relaxed);
            let failed = flags.iter().any(|f| f == FLAG_ORACLE_UNAVAILABLE);
            if failed {
                c.oracle_unavailable.fetch_add(1, Ordering::Relaxed);
            }
            self.push_audit(vec![AuditEntry {
                account: p.account.clone(),
                question: ROUTE_QUESTION_ID.to_string(),
                record: json!({
                    "request_id": request_id,
                    "ts": record.ts,
                    "taxonomy_id": s.id(),
                    "source": "router",
                    "router_label": choice.clone().unwrap_or_default(),
                    "final_label": choice.clone().unwrap_or_default(),
                    "agreement_with_router": Value::Null,
                    "oracle_model": failed.then(|| cfg.oracle.model.clone()),
                    "oracle_latency_ms": Value::Null,
                    "oracle_calls": 0,
                    "novelty_score": f32_json(local.decision.novelty),
                    "flags": audit_flags(action, &flags),
                }),
            }]);
        }
        let question = route_question(s);
        let answer = json!({"type": "choice", "choice": choice});
        let outcome = QuestionOutcome {
            id: ROUTE_QUESTION_ID.to_string(),
            kind: QuestionKind::Choice,
            matched: SkillMatch {
                kind: MatchKind::Exact,
                skill: Some(s.id().to_string()),
                candidates: (0..local.labels.len()).collect(),
                unknown: Vec::new(),
                reason: None,
            },
            action,
            local: Some(local.clone()),
            oracle: None,
            answer,
            certified: false,
            confident: action == Action::Local,
            flags: flags.clone(),
            decision_path: path,
        };
        if let Some(esc) = self.svc.escalator() {
            let dim_p = model.model().encoder_dim();
            let features = Features {
                phi_p: x[..dim_p].to_vec(),
                phi_h: x[dim_p..].iter().map(|v| v * 2.0).collect(),
            };
            let req = DecisionRequest {
                model: ModelRef::Latest,
                state: RequestState::Text(String::new()),
                state_text: String::new(),
                questions: vec![question],
                cmf: CmfOptions {
                    skill: Some(s.id().to_string()),
                    ..CmfOptions::default()
                },
                user: None,
                session_id: None,
            };
            esc.observe(&Observation {
                request_id,
                principal: p,
                model,
                request: &req,
                features: &features,
                questions: std::slice::from_ref(&outcome),
            });
            self.link(request_id, request_id);
        }
        let gate_json = json!({
            "accepted": accepted,
            "p_top": f32_json(local.decision.p_top),
            "tau": f32_json(gate.tau),
            "novelty": f32_json(local.decision.novelty),
            "theta": f32_json(gate.novelty_theta),
            "is_novel": local.is_novel(),
            "margin": f32_json(local.decision.margin),
            "profile": r.profile.as_str(),
        });
        Ok(RouteOutcome {
            action,
            local: Some(local),
            oracle_label: None,
            flags,
            decision_path: path.to_string(),
            certified: false,
            complexity: cx.to_json(),
            routing,
            oracle_calls: 0,
            oracle_time: Duration::ZERO,
            embed_time: Duration::ZERO,
            route_time,
            model_name: model.name(),
            model_sha: model.model_sha().to_string(),
            generation: model.generation(),
            decision_id: None,
            cmf_question: json!({
                "action": action.as_str(), "source": action.source(), "skill": s.id(),
                "match": "exact", "certified": false, "gate": gate_json,
                "decision_path": path, "input": "embedding",
            }),
            cmf_usage: json!({
                "local": {"input_tokens": 0, "output_tokens": out_tokens, "processed_tokens": 0,
                          "cost": cost.local.to_f64()},
                "oracle": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                           "billed": 0.0, "passthrough": self.rates.oracle_passthrough},
            }),
        })
    }

    /// The router response of one decided input (router `api.rs:95-201`).
    fn route_json(
        &self,
        r: &Routing,
        o: &RouteOutcome,
        client_request_id: Option<&str>,
        request_id: &str,
        taxonomy_version: u64,
    ) -> Value {
        let local = o.local.as_ref();
        let mut scores: Vec<ScoreRow> = local
            .map(|l| {
                l.decision
                    .ranked
                    .iter()
                    .map(|k| ScoreRow {
                        task_id: l.tasks[k.index],
                        label: l.labels[k.index].clone(),
                        probability: k.probability,
                        score: k.score,
                        error: k.error,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let router_label = local.and_then(|l| l.choice.clone());
        let escalated_answer = matches!(o.action, Action::Oracle | Action::Cache);
        let (source, chosen, confidence, is_novel) = match (escalated_answer, &o.oracle_label) {
            (true, Some(label)) => {
                promote(&mut scores, label, 1.0);
                (o.action.source(), label.clone(), 1.0f32, false)
            }
            _ => (
                "router",
                router_label
                    .clone()
                    .unwrap_or_else(|| NOVEL_LABEL.to_string()),
                local.map_or(0.0, |l| l.decision.p_top),
                local.is_none_or(LocalDecision::is_novel),
            ),
        };
        let flags = audit_flags(o.action, &o.flags);
        let task_id: i64 = if chosen == NOVEL_LABEL {
            -1
        } else {
            local
                .and_then(|l| {
                    l.labels
                        .iter()
                        .position(|x| *x == chosen)
                        .map(|i| l.tasks[i] as i64)
                })
                .unwrap_or(0)
        };
        let confident = !is_novel && !flags.iter().any(|f| f == FLAG_LOW_CONFIDENCE);
        scores.truncate(r.top_k);
        let oracle_model = &self.config().oracle.model;
        let failed = o.flags.iter().any(|f| f == FLAG_ORACLE_UNAVAILABLE);
        let oracle = match o.action {
            Action::Oracle => json!({
                "consulted": true,
                "model": oracle_model,
                "agreement_with_router": Some(chosen.as_str()) == router_label.as_deref(),
                "latency_ms": f32_json(ms(o.oracle_time)),
            }),
            Action::Cache => json!({
                "consulted": false,
                "model": oracle_model,
                "agreement_with_router": Some(chosen.as_str()) == router_label.as_deref(),
                "latency_ms": Value::Null,
            }),
            Action::Abstain if failed => json!({
                "consulted": false,
                "model": oracle_model,
                "agreement_with_router": Value::Null,
                "latency_ms": if o.oracle_time.is_zero() { Value::Null } else { f32_json(ms(o.oracle_time)) },
            }),
            _ => Value::Null,
        };
        let mut v = Map::new();
        v.insert("schema_version".into(), json!(ROUTER_SCHEMA_VERSION));
        v.insert("request_id".into(), json!(request_id));
        if let Some(c) = client_request_id {
            v.insert("client_request_id".into(), json!(c));
        }
        v.insert(
            "decision".into(),
            json!({
                "task_id": task_id,
                "task_label": chosen,
                "taxonomy_id": r.skill,
                "confidence": f32_json(confidence),
                "confident": confident,
                "raw_confidence": f32_json(local.map_or(0.0, |l| l.decision.raw_confidence)),
                "margin": f32_json(local.map_or(0.0, |l| l.decision.margin)),
                "is_novel": is_novel,
                "novelty_score": f32_json(local.map_or(1.0, |l| l.decision.novelty)),
                "complexity": o.complexity,
                "source": source,
                "flags": flags,
            }),
        );
        v.insert(
            "scores".into(),
            Value::Array(
                scores
                    .iter()
                    .map(|s| {
                        json!({
                            "task_id": s.task_id,
                            "task_label": s.label,
                            "probability": f32_json(s.probability),
                            "score": f32_json(s.score),
                            "reconstruction_error": f32_json(s.error),
                        })
                    })
                    .collect(),
            ),
        );
        if r.routing_table
            && let Some(rt) = &o.routing
        {
            v.insert("routing".into(), rt.clone());
        }
        if !oracle.is_null() {
            v.insert("oracle".into(), oracle);
        }
        if r.explain {
            v.insert(
                "explanation".into(),
                json!({"top1_vs_top2": top1_vs_top2(&scores), "decision_path": o.decision_path}),
            );
        }
        v.insert(
            "usage".into(),
            json!({"billable_decisions": 1, "oracle_calls": o.oracle_calls}),
        );
        v.insert(
            "meta".into(),
            json!({
                "model_version": o.model_name,
                "taxonomy_version": format!("{}@{}", r.skill, taxonomy_version),
                "latency_ms": f32_json(ms(o.route_time)),
                "embedding_latency_ms": f32_json(ms(o.embed_time)),
                "served_by": served_by(),
            }),
        );
        if r.extensions {
            v.insert(
                "cmf".into(),
                json!({
                    "id": o.decision_id,
                    "action": o.action.as_str(),
                    "certified": o.certified,
                    "decision_path": o.decision_path,
                    "generation": o.generation,
                    "model_sha": o.model_sha,
                    "question": o.cmf_question,
                    "usage": o.cmf_usage,
                }),
            );
        }
        Value::Object(v)
    }
}

async fn route_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let id = ctx.id.clone();
    let ext = ctx.ext;
    let out = async {
        let (p, req): (Principal, wire::RouteRequest) =
            keyed_router_body(&st, &headers, body).await?;
        let account = p.account.clone();
        let routing = st
            .resolve_routing(req.taxonomy_id.as_deref(), &req.options, ext)
            .map_err(|e| e.by(&account))?;
        let guard = st
            .svc
            .enter()
            .map_err(|e| HttpError::from(e).by(&account))?;
        let s = Arc::clone(&st);
        let v = blocking(move || {
            let _slot = guard;
            s.route_one(
                &p,
                &routing,
                &req.input,
                req.client_request_id.as_deref(),
                &id,
            )
            .map_err(|e| e.by(&p.account))
        })
        .await?;
        Ok(Reply::ok(v).by(&account))
    }
    .await;
    respond(&ctx, out)
}

async fn batch_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let ext = ctx.ext;
    let out = async {
        let (p, req): (Principal, wire::BatchRequest) =
            keyed_router_body(&st, &headers, body).await?;
        let account = p.account.clone();
        // The router checks the taxonomy before the batch size.
        let routing = st
            .resolve_routing(req.taxonomy_id.as_deref(), &req.options, ext)
            .map_err(|e| e.by(&account))?;
        if req.inputs.len() > ROUTER_MAX_BATCH {
            return Err(
                HttpError::invalid(format!("batch exceeds {ROUTER_MAX_BATCH} inputs")).by(&account),
            );
        }
        let guard = st
            .svc
            .enter()
            .map_err(|e| HttpError::from(e).by(&account))?;
        let s = Arc::clone(&st);
        let results = blocking(move || {
            let _slot = guard;
            let mut out = Vec::with_capacity(req.inputs.len());
            for (i, input) in req.inputs.iter().enumerate() {
                // The quotas and the credit were checked before the request;
                // every later input is checked again against what the batch
                // has used so far (a failing input fails the batch).
                if i > 0 {
                    s.svc
                        .check_quotas(&p)
                        .map_err(|e| HttpError::from(e).by(&p.account))?;
                }
                let id = router_request_id();
                out.push(
                    s.route_one(&p, &routing, input, None, &id)
                        .map_err(|e| e.by(&p.account))?,
                );
            }
            Ok(out)
        })
        .await?;
        Ok(Reply::ok(json!({
            "schema_version": ROUTER_SCHEMA_VERSION,
            "results": results,
        }))
        .by(&account))
    }
    .await;
    respond(&ctx, out)
}

// ------------------------------------------------------------------ feedback

/// Whether a feedback body is the decisions API's `{id, question, label}`: an
/// object with one of its keys and none of the router's. Anything else is
/// the router's `{request_id, correct_task_label}` and is rejected as the
/// router rejects it.
fn is_decisions_feedback(bytes: &[u8]) -> bool {
    // The first JSON value, as `router_json` reads the router's body.
    let first = serde_json::Deserializer::from_slice(bytes)
        .into_iter::<Value>()
        .next();
    match first {
        Some(Ok(Value::Object(m))) => {
            ["id", "question", "label"]
                .iter()
                .any(|k| m.contains_key(*k))
                && !m.contains_key("request_id")
                && !m.contains_key("correct_task_label")
        }
        _ => false,
    }
}

/// A feedback answer of a key without `learning_allowed` (nothing learned).
fn learning_refused(r: &Value) -> bool {
    r["refused"] == json!(cortiq_decision::cascade::LEARNING_NOT_ALLOWED)
}

/// The router's `message` of a feedback answer.
fn feedback_message(r: &Value, label: &str) -> String {
    if learning_refused(r) {
        format!("feedback for '{label}' not learned: this key may not teach the model")
    } else {
        format!("feedback recorded for '{label}'")
    }
}

async fn feedback_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let ext = ctx.ext;
    let out = async {
        let p = caller(&st, &headers).await?;
        let account = p.account.clone();
        let bytes = router_body_bytes(&headers, body, st.body_limit())
            .await
            .map_err(|e| e.by(&account))?;
        if is_decisions_feedback(&bytes) {
            let guard = st
                .svc
                .enter()
                .map_err(|e| HttpError::from(e).by(&account))?;
            let s = Arc::clone(&st);
            let reply = blocking(move || {
                let _slot = guard;
                let mut r = s
                    .svc
                    .feedback(&bytes, &p)
                    .map_err(|e| HttpError::from(e).by(&p.account))?;
                let label = r["label"].as_str().unwrap_or_default().to_string();
                r["schema_version"] = json!(ROUTER_SCHEMA_VERSION);
                r["message"] = json!(feedback_message(&r, &label));
                Ok(r)
            })
            .await?;
            return Ok(Reply::ok(reply).by(&account));
        }
        let fb: wire::FeedbackRequest = router_json(&bytes).map_err(|e| e.by(&account))?;
        // Any label is accepted, as by the router (`api.rs:1261-1299`, no
        // length check): an empty or over-long one (not 1..256 bytes) consumes
        // the request and is answered 200, but teaches nothing (the cascade
        // does not store it).
        let guard = st
            .svc
            .enter()
            .map_err(|e| HttpError::from(e).by(&account))?;
        let s = Arc::clone(&st);
        let reply = blocking(move || {
            let _slot = guard;
            let not_found =
                || HttpError::not_found("request_id not found or already consumed").by(&p.account);
            let decision = s.linked(&fb.request_id).ok_or_else(not_found)?;
            let req = cortiq_decision::protocol::FeedbackRequest {
                id: decision,
                question: ROUTE_QUESTION_ID.to_string(),
                label: fb.correct_task_label.clone(),
                any_label: true,
            };
            let r = s.svc.feedback_request(&req, &p).map_err(|e| {
                let e = HttpError::from(e).by(&p.account);
                if e.status == 404 { not_found() } else { e }
            })?;
            s.unlink(&fb.request_id);
            let mut v = json!({
                "schema_version": ROUTER_SCHEMA_VERSION,
                // False only when the key may not teach the model; an empty or
                // over-long label is `accepted` as by the router.
                "accepted": !learning_refused(&r),
                "message": feedback_message(&r, &fb.correct_task_label),
            });
            if ext {
                v["cmf"] = r;
            }
            Ok(v)
        })
        .await?;
        Ok(Reply::ok(reply).by(&account))
    }
    .await;
    respond(&ctx, out)
}

// ------------------------------------------------------------------ router API: listings

async fn taxonomies_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        let p = caller(&st, &headers).await?;
        let model = st.svc.handle().current();
        let list: Vec<Value> = st
            .ordered_skills(&model)
            .into_iter()
            .map(|s| taxonomy_summary(&model, s, ctx.ext))
            .collect();
        Ok(Reply::ok(json!({
            "schema_version": ROUTER_SCHEMA_VERSION,
            "taxonomies": list,
        }))
        .by(&p.account))
    }
    .await;
    respond(&ctx, out)
}

async fn taxonomy_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    id: Result<Path<String>, PathRejection>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        let p = caller(&st, &headers).await?;
        let Path(id) = id.map_err(|r| path_rejected(r).by(&p.account))?;
        let model = st.svc.handle().current();
        let s = model
            .skill(&id)
            .ok_or_else(|| taxonomy_not_found(&id).by(&p.account))?;
        Ok(Reply::ok(taxonomy_summary(&model, s, ctx.ext)).by(&p.account))
    }
    .await;
    respond(&ctx, out)
}

/// `GET /v1/usage`. The totals come from the usage ledger, whose lock the
/// flusher holds across `fsync`: the view is built on the blocking pool, never
/// on a runtime worker.
async fn usage_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        let p = caller(&st, &headers).await?;
        let (s, ext, account) = (Arc::clone(&st), ctx.ext, p.account.clone());
        let v = blocking(move || Ok(s.usage_json(&p, ext))).await?;
        Ok(Reply::ok(v).by(&account))
    }
    .await;
    respond(&ctx, out)
}

/// The router's `Query<EscalationsQuery>` (axum 0.7): 400 with axum's text
/// (axum 0.7 names no field in it; 0.8 prefixes the field, which is dropped).
fn escalations_query(uri: &Uri) -> Result<wire::EscalationsQuery, HttpError> {
    const PREFIX: &str = "Failed to deserialize query string: ";
    axum::extract::Query::<wire::EscalationsQuery>::try_from_uri(uri)
        .map(|q| q.0)
        .map_err(|r| {
            let text = r.body_text();
            let text = match text.strip_prefix(PREFIX) {
                Some(rest) => format!("{PREFIX}{}", rest.strip_prefix("limit: ").unwrap_or(rest)),
                None => text,
            };
            HttpError::plain(r.status().as_u16(), text)
        })
}

async fn escalations_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let out = async {
        let p = caller(&st, &headers).await?;
        let q = escalations_query(&uri).map_err(|e| e.by(&p.account))?;
        let limit = q
            .limit
            .unwrap_or(ESCALATIONS_DEFAULT_LIMIT)
            .clamp(1, ESCALATIONS_MAX_LIMIT);
        // The learning state's locks are held across disk writes: the view is
        // built on the blocking pool.
        let (s, ext, account) = (Arc::clone(&st), ctx.ext, p.account.clone());
        let v = blocking(move || Ok(s.escalations_json(&p.account, limit, ext))).await?;
        Ok(Reply::ok(v).by(&account))
    }
    .await;
    respond(&ctx, out)
}

impl DecisionState {
    /// `GET /v1/escalations` of `account` (see [`escalations_handler`]).
    fn escalations_json(&self, account: &str, limit: usize, extensions: bool) -> Value {
        let records: Vec<Value> = lock(&self.audit)
            .iter()
            .rev()
            .filter(|e| e.account == account)
            .take(limit)
            .map(|e| {
                let mut r = e.record.clone();
                if extensions {
                    r["cmf"] = json!({"question": e.question});
                }
                r
            })
            .collect();
        let c = &self.counters;
        let (buffered, cache) = match &self.cascade {
            Some(cas) => {
                let l = cas.learning_json();
                (
                    cas.buffer_len() as u64,
                    json!({
                        "entries": l["cache"]["entries"].as_u64().unwrap_or(0),
                        "hits": l["cache"]["hits"].as_u64().unwrap_or(0),
                        "lookups": l["cache"]["lookups"].as_u64().unwrap_or(0),
                    }),
                )
            }
            None => (0, json!({"entries": 0, "hits": 0, "lookups": 0})),
        };
        json!({
            "schema_version": ROUTER_SCHEMA_VERSION,
            "summary": {
                "total": Counters::get(&c.escalations),
                "oracle_calls": Counters::get(&c.oracle_calls),
                "cache_hits": Counters::get(&c.cache_hits),
                "oracle_unavailable": Counters::get(&c.oracle_unavailable),
                "labeled_examples": buffered,
                "cache": cache,
            },
            "records": records,
        })
    }
}

async fn readyz_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
) -> Response {
    let model = st.svc.handle().current();
    let ready = model.skills().iter().any(|s| !s.scorer().is_empty());
    let (status, body) = if ready {
        (StatusCode::OK, json!({"status": "ready"}))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"status": "not_ready"}),
        )
    };
    json_response(status, &body, &ctx.id, None, None)
}

/// `GET /metrics` (Prometheus text), built on the blocking pool like the other
/// views of the learning state.
async fn metrics_handler(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
) -> Response {
    let s = Arc::clone(&st);
    let text = match blocking(move || Ok(s.metrics_text())).await {
        Ok(t) => t,
        Err(e) => return error_response(&ctx, e),
    };
    let mut resp = Response::new(Body::from(text));
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4"),
    );
    if let Ok(v) = HeaderValue::from_str(&ctx.id) {
        h.insert(HeaderName::from_static(REQUEST_ID_HEADER), v);
    }
    resp
}

// ------------------------------------------------------------------ admin

/// The router's view of a key (`api.rs:514-526`).
fn router_key_listing(listing: &Value, usage: &Value) -> Value {
    json!({
        "account": listing["account"],
        "plan": listing["plan"],
        "rate_per_min": listing["rate_per_min"],
        "decision_quota": listing["decision_quota"],
        "expires_at": listing["expires"],
        "expired": listing["expired"],
        "key_hash_prefix": listing["hash12"],
        "usage": {
            "decisions": usage["decisions"].as_u64().unwrap_or(0),
            "oracle_calls": usage["oracle_calls"].as_u64().unwrap_or(0),
        },
    })
}

async fn admin_create_key(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let out = async {
        // The router's `Json` extractor runs before its token check.
        let bytes = router_body_bytes(&headers, body, st.body_limit()).await?;
        let req: wire::CreateKeyReq = router_json(&bytes)?;
        admin_guard(&st, &headers)?;
        let mut new = Map::new();
        for (k, v) in [
            ("plan", json!(req.plan)),
            ("account", json!(req.account)),
            ("label", json!(req.label)),
            ("days", json!(req.days)),
            ("rate_per_min", json!(req.rate_per_min)),
            ("decision_quota", json!(req.decision_quota)),
            ("token_quota", json!(req.token_quota)),
            ("credit_usd", json!(req.credit_usd)),
            ("oracle_budget_usd", json!(req.oracle_budget_usd)),
            ("oracle_allowed", json!(req.oracle_allowed)),
            ("learning_allowed", json!(req.learning_allowed)),
        ] {
            if !v.is_null() {
                new.insert(k.to_string(), v);
            }
        }
        let body = serde_json::to_vec(&new).map_err(|e| internal_error("admin keys", e))?;
        let s = Arc::clone(&st);
        let created = blocking(move || Ok(s.svc.admin_create_key(&body)?)).await?;
        tracing::info!(
            account = created["account"].as_str().unwrap_or_default(),
            "admin: API key created"
        );
        let mut v = json!({
            "key": created["key"],
            "account": created["account"],
            "plan": created["plan"],
            "rate_per_min": created["rate_per_min"],
            "decision_quota": created["decision_quota"],
            "expires_at": created["expires"],
            "created_at": created["created"],
            "persisted": true,
        });
        if ctx.ext {
            let mut listing = created;
            if let Some(m) = listing.as_object_mut() {
                m.remove("key");
            }
            v["cmf"] = listing;
        }
        Ok(Reply::ok(v))
    }
    .await;
    respond(&ctx, out)
}

async fn admin_list_keys(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        admin_guard(&st, &headers)?;
        let s = Arc::clone(&st);
        let list = blocking(move || Ok(s.svc.admin_list_keys()?)).await?;
        // The router's gate drops a revoked key; keys.json keeps it inactive
        // (shown with the extensions).
        let keys: Vec<Value> = list["keys"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .filter(|k| ctx.ext || k["active"].as_bool().unwrap_or(false))
            .map(|k| {
                let mut v = router_key_listing(k, &k["usage"]);
                if ctx.ext {
                    v["cmf"] = k.clone();
                }
                v
            })
            .collect();
        Ok(Reply::ok(json!({"count": keys.len(), "keys": keys})))
    }
    .await;
    respond(&ctx, out)
}

async fn admin_revoke_account(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    account: Result<Path<String>, PathRejection>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        admin_guard(&st, &headers)?;
        let Path(account) = account.map_err(path_rejected)?;
        let s = Arc::clone(&st);
        let v = blocking(move || Ok(s.svc.admin_revoke_account(&account)?)).await?;
        tracing::info!(
            account = v["account"].as_str().unwrap_or_default(),
            "admin: API keys revoked"
        );
        Ok(Reply::ok(v))
    }
    .await;
    respond(&ctx, out)
}

async fn admin_revoke_hash(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    hash: Result<Path<String>, PathRejection>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        admin_guard(&st, &headers)?;
        let Path(hash) = hash.map_err(path_rejected)?;
        let s = Arc::clone(&st);
        let v = blocking(move || Ok(s.svc.admin_revoke_hash(&hash)?)).await?;
        Ok(Reply::ok(v))
    }
    .await;
    respond(&ctx, out)
}

async fn admin_usage(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        admin_guard(&st, &headers)?;
        let s = Arc::clone(&st);
        Ok(Reply::ok(blocking(move || Ok(s.svc.admin_usage())).await?))
    }
    .await;
    respond(&ctx, out)
}

async fn admin_command(st: &Arc<DecisionState>, command: AdminCommand) -> Handled {
    let s = Arc::clone(st);
    let v = blocking(move || Ok(s.svc.admin(&command)?)).await?;
    Ok(Reply::ok(v))
}

async fn admin_oracle(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        admin_guard(&st, &headers)?;
        admin_command(&st, AdminCommand::OracleStatus).await
    }
    .await;
    respond(&ctx, out)
}

async fn admin_oracle_update(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let out = async {
        let bytes = admin_body(&st, &headers, body).await?;
        let v = parse_json(&bytes)?;
        let r = admin_command(&st, AdminCommand::OracleUpdate(v)).await?;
        tracing::info!("admin: oracle settings updated");
        Ok(r)
    }
    .await;
    respond(&ctx, out)
}

async fn admin_learning(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        admin_guard(&st, &headers)?;
        admin_command(&st, AdminCommand::Learning).await
    }
    .await;
    respond(&ctx, out)
}

async fn admin_generations(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
) -> Response {
    let out = async {
        admin_guard(&st, &headers)?;
        admin_command(&st, AdminCommand::Generations).await
    }
    .await;
    respond(&ctx, out)
}

async fn admin_rollback(
    State(st): State<Arc<DecisionState>>,
    Extension(ctx): Extension<Ctx>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let out = async {
        let bytes = admin_body(&st, &headers, body).await?;
        let v = parse_json(&bytes)?;
        let generation = match v.as_object() {
            Some(m) if m.len() == 1 => m.get("generation").and_then(Value::as_u64),
            _ => None,
        }
        .ok_or_else(|| {
            HttpError::invalid("the rollback body is {\"generation\": N} (0 = the base file)")
        })?;
        let r = admin_command(&st, AdminCommand::Rollback { generation }).await?;
        tracing::info!(generation, "admin: rolled back");
        Ok(r)
    }
    .await;
    respond(&ctx, out)
}

// ------------------------------------------------------------------ server

/// How to serve a decision file (`cortiq serve FILE` with the DECISION bit).
#[derive(Clone)]
pub struct ServeOptions {
    /// The base decision file.
    pub model: PathBuf,
    /// `--decision-config` (defaults when absent).
    pub config: Config,
    /// `--state DIR`; else `config.state_dir`, else `<FILE>.state`.
    pub state_dir: Option<PathBuf>,
    /// `--break-lock`: remove a `LOCK` left by a dead process.
    pub break_lock: bool,
    /// Listening address (default `127.0.0.1:8080`). Whether it is loopback
    /// decides `auth.require: null`.
    pub addr: SocketAddr,
    /// Where the oracle key is read from (default: the process environment
    /// variable named by `oracle.api_key_env`), learning threads.
    pub cascade: CascadeOptions,
    /// The admin token; `None` reads the variable named by
    /// `auth.admin_token_env` (the only production source).
    pub admin_token: Option<String>,
    /// `--shadow-of URL`: shadow mode (spec §4.15) — the router API is
    /// answered by the router at `URL`, `/v1/route` and `/v1/route:batch` are
    /// also decided locally and compared in `<state>/shadow.jsonl`.
    pub shadow_of: Option<String>,
    /// Deadline of one request forwarded in shadow mode (`--shadow-timeout-s`;
    /// default [`cortiq_decision::shadow::UPSTREAM_TIMEOUT`], 60 s).
    pub shadow_timeout: Duration,
    /// `--oracle MODEL` was given: the operator enabled the oracle on the
    /// command line, so the open mode of a loopback address without
    /// `auth.require` may use it ([`DecisionService::with_open_oracle`]; it
    /// still never teaches the model, not even through the oracle's answers
    /// to a skill's own question).
    pub oracle_from_flag: bool,
    /// Where the oracle's max price came from (`--oracle`), for the startup
    /// line.
    pub oracle_note: Option<String>,
}

impl std::fmt::Debug for ServeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServeOptions")
            .field("model", &self.model)
            .field("state_dir", &self.state_dir)
            .field("break_lock", &self.break_lock)
            .field("addr", &self.addr)
            .field("cascade", &self.cascade)
            .field(
                "admin_token",
                &self.admin_token.as_ref().map(|_| "<redacted>"),
            )
            .field("shadow_of", &self.shadow_of)
            .field("shadow_timeout", &self.shadow_timeout)
            .field("oracle_from_flag", &self.oracle_from_flag)
            .field("oracle_note", &self.oracle_note)
            .finish()
    }
}

impl ServeOptions {
    /// Defaults for `model` with `config`: loopback, port 8080.
    pub fn new(model: impl Into<PathBuf>, config: Config) -> Self {
        Self {
            model: model.into(),
            config,
            state_dir: None,
            break_lock: false,
            addr: SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
            cascade: CascadeOptions::default(),
            admin_token: None,
            shadow_of: None,
            shadow_timeout: cortiq_decision::shadow::UPSTREAM_TIMEOUT,
            oracle_from_flag: false,
            oracle_note: None,
        }
    }

    /// The state directory these options use.
    pub fn state_root(&self) -> PathBuf {
        self.state_dir
            .clone()
            .unwrap_or_else(|| self.config.state_dir_for(&self.model))
    }
}

/// An opened decision server: the served model (base + `CURRENT`), the state
/// directory under its `LOCK`, keys, the usage ledger with its flusher, the
/// cascade and the service.
pub struct DecisionServer {
    state: Arc<DecisionState>,
    ledger: Arc<UsageLedger>,
    flusher: Option<Flusher>,
    dir: StateDir,
    _lock: StateLock,
}

impl std::fmt::Debug for DecisionServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionServer")
            .field("state_dir", &self.dir.root())
            .field("state", &self.state)
            .finish()
    }
}

impl DecisionServer {
    /// Open everything (blocking: the model is verified in full, spec §2.7).
    pub fn open(opts: &ServeOptions) -> Result<Self> {
        let cfg = &opts.config;
        cfg.validate()?;
        // The URL is checked before anything is opened or created.
        let upstream = opts
            .shadow_of
            .as_deref()
            .map(|u| cortiq_decision::shadow::Upstream::new(u, opts.shadow_timeout))
            .transpose()?;
        // The file is checked (envelope, profile, manifests) before the state
        // directory is created: a file this version refuses leaves nothing
        // behind next to it.
        cortiq_decision::container::DecisionModel::open(&opts.model, Verify::Light)
            .with_context(|| format!("open {}", opts.model.display()))?;
        let root = opts.state_root();
        let dir =
            StateDir::open(&root).with_context(|| format!("state directory {}", root.display()))?;
        let lock = dir.lock(opts.break_lock)?;
        let model = generation::open_served(&opts.model, &dir, Verify::Full)
            .with_context(|| format!("open {}", opts.model.display()))?;
        for w in model.warnings() {
            tracing::warn!("{w}");
        }
        let loaded = LoadedModel::new(model)?;
        if let Some(d) = &cfg.default_skill
            && loaded.skill(d).is_none()
        {
            let ids: Vec<&str> = loaded.skills().iter().map(SkillRuntime::id).collect();
            bail!(
                "default_skill '{d}' is not a skill of {} (skills: {})",
                opts.model.display(),
                ids.join(", ")
            );
        }
        let handle = Arc::new(ModelHandle::new(loaded));
        let keys = Arc::new(KeyStore::open(dir.keys_path(), &cfg.auth.key_prefix)?);
        let ledger = Arc::new(UsageLedger::open(dir.usage_dir())?);
        for w in ledger.warnings() {
            tracing::warn!("{w}");
        }
        let cascade =
            Cascade::open_with(Arc::clone(&handle), cfg, dir.clone(), opts.cascade.clone())?;
        let esc: Arc<dyn Escalator> = cascade.clone();
        let loopback = opts.addr.ip().is_loopback();
        let mut svc = DecisionService::open(Arc::clone(&handle), cfg.clone(), Some(esc))?
            .with_loopback(loopback)
            .with_open_oracle(opts.oracle_from_flag && loopback)
            .with_keys(Arc::clone(&keys))
            .with_ledger(Arc::clone(&ledger));
        if let Some(t) = &opts.admin_token {
            svc = svc.with_admin_token(Some(t.clone()));
        }
        let svc = Arc::new(svc);
        let flusher = ledger.start_flusher(FLUSH_EVERY);
        let model = handle.current();
        tracing::info!(
            model = %model.name(),
            generation = model.generation(),
            skills = model.skills().len(),
            state = %dir.root().display(),
            auth = if svc.auth_enabled() { "keys" } else { "open" },
            keys = keys.len(),
            oracle = cfg.oracle.enabled,
            learning = cfg.learning.enabled,
            "decision server ready"
        );
        if !svc.auth_enabled() && !loopback {
            tracing::warn!(
                "open mode on a non-loopback address: anyone who can connect may decide"
            );
        }
        if !svc.auth_enabled() && cfg.auth.require.is_none() {
            if opts.oracle_from_flag {
                tracing::warn!(
                    "open mode only because the address is loopback (auth.require: null): callers without a key may use the oracle (--oracle, within its budget) but never teach the model (its answers to them are cached, not learned; their feedback is not learned); set auth.require to true behind a reverse proxy"
                );
            } else {
                tracing::warn!(
                    "open mode only because the address is loopback (auth.require: null): callers without a key may not reach the oracle or teach the model; set auth.require to false to allow it, or to true behind a reverse proxy"
                );
            }
        }
        let (ready, line) = oracle_startup_line(&cascade, cfg, opts.oracle_note.as_deref());
        if ready {
            tracing::info!("{line}");
        } else {
            tracing::warn!("{line}");
        }
        // A key read with surrounding whitespace (a `.env` file's CR, LF) is
        // trimmed before every call; said once (never the key).
        let trimmed = cascade.oracle().key_state().trimmed();
        if cfg.oracle.enabled && trimmed > 0 {
            tracing::warn!(
                "{}",
                cortiq_decision::oracle::trimmed_warning(&cfg.oracle.api_key_env, trimmed)
            );
        }
        let shadow = match upstream {
            Some(u) => {
                let sh = shadow::Shadow::open(u, &dir.shadow_log_path())?;
                tracing::info!(
                    shadow_of = %sh.upstream_base(),
                    log = %dir.shadow_log_path().display(),
                    lines = sh.lines(),
                    "shadow mode: the router API is answered by the old router"
                );
                tracing::debug!(
                    deadline_s = opts.shadow_timeout.as_secs_f64(),
                    "shadow mode: deadline of a forwarded request"
                );
                Some(Arc::new(sh))
            }
            None => None,
        };
        let state = DecisionState::with_shadow(svc, Some(cascade), shadow)?;
        Ok(Self {
            state,
            ledger,
            flusher: Some(flusher),
            dir,
            _lock: lock,
        })
    }

    pub fn state(&self) -> &Arc<DecisionState> {
        &self.state
    }

    pub fn state_dir(&self) -> &StateDir {
        &self.dir
    }

    /// The HTTP router of this server.
    pub fn router(&self) -> Router {
        router(Arc::clone(&self.state))
    }

    /// Serve on `listener` until `shutdown` resolves, then close.
    pub async fn run(
        self,
        listener: tokio::net::TcpListener,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<()> {
        let app = self.router();
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
            .context("decision server")?;
        if let Some(sh) = &self.state.shadow {
            sh.drain(shadow::DRAIN_TIMEOUT).await;
        }
        tokio::task::spawn_blocking(move || self.close())
            .await
            .context("close the decision server")?
    }

    /// Flush and snapshot the usage ledger; stop the learning worker; release
    /// the `LOCK`.
    pub fn close(mut self) -> Result<()> {
        if let Some(f) = self.flusher.take() {
            f.stop()?;
        }
        self.ledger.close()?;
        if let Some(sh) = &self.state.shadow {
            sh.sync()?;
        }
        Ok(())
    }
}

/// The one startup line of the oracle (never the key, only its variable's
/// name): `(ready or deliberately off, line)`, e.g. `oracle: ready —
/// deepseek/deepseek-v4.1-flash via openrouter.ai, budget $5.00, max price
/// in/out $0.07/$0.58 per 1M` or `oracle: NOT ready — OPENROUTER_API_KEY is
/// not set …`.
pub fn oracle_startup_line(cascade: &Cascade, cfg: &Config, note: Option<&str>) -> (bool, String) {
    use cortiq_decision::oracle_setup::{host_of, usd, usd_ceil, usd_fine};
    use cortiq_decision::service::OracleStatus;
    let oracle = cascade.oracle();
    let o = &cfg.oracle;
    let st = oracle.status_json();
    let f = |k: &str| st[k].as_f64().unwrap_or(0.0);
    let (budget, spent) = (f("budget_usd"), f("spent_usd"));
    let what = format!("{} via {}", o.model, host_of(&o.base_url));
    let budget_text = if spent > 0.0 {
        format!("budget {} ({} spent)", usd(budget), usd(spent))
    } else {
        format!("budget {}", usd(budget))
    };
    match oracle.status() {
        OracleStatus::Ready => {
            let (p, c) = oracle.max_price();
            let note = note.map(|n| format!(" ({n})")).unwrap_or_default();
            (
                true,
                format!(
                    "oracle: ready — {what}, {budget_text}, max price in/out {}/{} per 1M{note}",
                    usd(p),
                    usd(c)
                ),
            )
        }
        OracleStatus::Disabled { by_admin: false } => (
            true,
            format!(
                "oracle: off — questions the local model cannot decide abstain (start the server with --oracle MODEL and set {})",
                o.api_key_env
            ),
        ),
        OracleStatus::Disabled { by_admin: true } => (
            false,
            format!(
                "oracle: NOT ready — switched off by the admin API ({what}; POST /v1/admin/oracle {{\"enabled\":true}} turns it on)"
            ),
        ),
        OracleStatus::NoKey => (
            false,
            format!(
                "oracle: NOT ready — {} is not set (set it to your OpenRouter key and restart; {what}, {budget_text})",
                o.api_key_env
            ),
        ),
        OracleStatus::BadKey(p) => (
            false,
            format!(
                "oracle: NOT ready — {} (fix the variable and restart; nothing is sent with it; {what}, {budget_text})",
                cortiq_decision::oracle::bad_key_text(&o.api_key_env, &p)
            ),
        ),
        OracleStatus::BudgetExhausted => (
            false,
            format!(
                "oracle: NOT ready — the budget is used up ({what}, {budget_text}, {} of {} calls; restart with a larger --oracle-budget or --oracle-max-calls)",
                st["calls"], st["max_calls"]
            ),
        ),
        OracleStatus::BudgetTooSmall { min_usd } => (
            false,
            if st["max_calls"].as_u64() == Some(0) {
                format!(
                    "oracle: NOT ready — max_calls 0 allows no call ({what}; restart with --oracle-max-calls of at least 1)"
                )
            } else {
                let (p, c) = oracle.max_price();
                format!(
                    "oracle: NOT ready — the budget is too small: {budget_text} cannot hold one call; the smallest possible one (one short question) reserves {} at the max price in/out {}/{} per 1M, a longer one more ({what}; restart with --oracle-budget of at least {}, more for longer questions)",
                    usd_fine(min_usd),
                    usd(p),
                    usd(c),
                    usd_ceil(min_usd)
                )
            },
        ),
        OracleStatus::Stopped(r) => {
            // A `max_errors` stop names the last failure's code.
            let last = match st["last_error"].as_str() {
                Some(e) if r == "max_errors" => format!(", the last error: {e}"),
                _ => String::new(),
            };
            (
                false,
                format!(
                    "oracle: NOT ready — stopped by a stop rule ({r}{last}; {what}); after the fix POST /v1/admin/oracle {{\"enabled\":true}} resumes it"
                ),
            )
        }
    }
}

/// Ctrl-C, or SIGTERM on Unix.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("decision server shutting down");
}

/// `cortiq serve FILE` on a decision file: open (on the blocking pool), bind,
/// serve until Ctrl-C/SIGTERM, flush the ledger.
pub async fn serve(opts: ServeOptions) -> Result<()> {
    let addr = opts.addr;
    let server = tokio::task::spawn_blocking(move || DecisionServer::open(&opts))
        .await
        .context("open the decision server")??;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    tracing::info!(addr = %addr, "decision server listening");
    server.run(listener, shutdown_signal()).await
}
